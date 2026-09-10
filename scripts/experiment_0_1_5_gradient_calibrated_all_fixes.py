from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_0_1_2_regularized_general_comparison as exp012
from scripts import experiment_0_1_4_parallel_regularization_ablation as exp014


EXPERIMENT_ID = "experiment_0_1_5_gradient_calibrated_all_fixes"
PROTOCOL_VERSION = "gradient_calibrated_all_fixes_v1"
BASE_EXPERIMENT_ID = exp01.EXPERIMENT_ID
BASE_PROTOCOL_VERSION = exp01.PROTOCOL_VERSION

DIRECT_FAMILY = exp01.DIRECT_FAMILY
DIRECT_ARCHITECTURES = exp01.DIRECT_ARCHITECTURES
OBJECTIVES = exp01.OBJECTIVES
SEEDS = exp01.SEEDS
VARIANT = "binary"
HIDDEN_CAP = 1
HIDDEN_WIDTH = exp01.HIDDEN_WIDTH
OUTPUT_CAP = exp01.OUTPUT_CAP

TARGET_GRAD_RATIOS = (0.05, 0.10, 0.20)
CALIBRATION_BATCHES = 5
REG_TAU = exp014.REG_TAU
BASE_LAMBDA_P2 = exp014.REG_LAMBDA_P2
BASE_LAMBDA_A1 = exp014.REG_LAMBDA_A1
WARMUP_EPOCHS = exp014.WARMUP_EPOCHS
GRAD_DIAGNOSTIC_EPOCHS = (1, 5, 10, 20, 30, 50, 75, 100)

MIN_EPOCHS = 50
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 30
VAL_BA_MIN_DELTA = 1e-12
VAL_OBJECTIVE_LOSS_MIN_DELTA = 1e-4

BATCH_SIZE = exp01.BATCH_SIZE
LR = exp01.LR
WEIGHT_DECAY = exp01.WEIGHT_DECAY


@dataclass(frozen=True)
class RunSpec:
    target_grad_ratio: float
    architecture: str
    objective: str
    seed: int

    @property
    def target_percent(self) -> int:
        return int(round(100.0 * float(self.target_grad_ratio)))

    @property
    def key(self) -> str:
        return (
            f"{DIRECT_FAMILY}__{self.architecture}__{self.objective}__"
            f"{VARIANT}__hcap{HIDDEN_CAP}__allfix_gr{self.target_percent:02d}__seed{self.seed}"
        )


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    max_epochs: int = MAX_EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def base_results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / BASE_EXPERIMENT_ID / BASE_PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(target_grad_ratio, architecture, objective, seed)
        for target_grad_ratio in TARGET_GRAD_RATIOS
        for architecture in DIRECT_ARCHITECTURES
        for objective in OBJECTIVES
        for seed in SEEDS
    ]


def base_spec(spec: RunSpec) -> exp01.RunSpec:
    return exp01.RunSpec(
        DIRECT_FAMILY,
        spec.architecture,
        spec.objective,
        VARIANT,
        HIDDEN_CAP,
        spec.seed,
    )


def paired_seed(spec: RunSpec, role: str) -> int:
    """Condition-independent stream shared by all target strengths."""
    return exp01.paired_seed(base_spec(spec), role)


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def calibration_path(root: Path, spec: RunSpec) -> Path:
    return root / "calibrations" / f"{spec.key}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _new_model(spec: RunSpec, data: exp01.exp3.Data) -> exp012.RegularizedMultiTauHierarchySNN:
    return exp012.RegularizedMultiTauHierarchySNN(
        layer_shifts=DIRECT_ARCHITECTURES[spec.architecture],
        n_classes=len(data.labels),
        fs=data.fs,
        hidden_cap=HIDDEN_CAP,
    )


def make_loaders(
    data: exp01.exp3.Data,
    spec: RunSpec,
    batch_size: int,
    train_shuffle: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    return exp01.make_loaders(data, base_spec(spec), batch_size, train_shuffle)


def calibration_loader(
    data: exp01.exp3.Data,
    spec: RunSpec,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Independent train-only loader that does not consume the training RNG stream."""
    return exp01.exp3.loader(
        data.Xtr,
        data.ytr,
        data.ltr,
        batch_size,
        True,
        paired_seed(spec, "exp0_1_5_calibration_loader"),
    )


def _hidden_alphas(model: exp012.RegularizedMultiTauHierarchySNN) -> tuple[torch.Tensor, ...]:
    return tuple(getattr(model, name) for name in model._alpha_names)


def _hidden_parameters(
    model: exp012.RegularizedMultiTauHierarchySNN,
) -> tuple[torch.nn.Parameter, ...]:
    params = tuple(
        parameter
        for linear in model.hidden_linears
        for parameter in linear.parameters()
        if parameter.requires_grad
    )
    if not params:
        raise RuntimeError("No trainable hidden SNN parameters found")
    return params


def all_fixes_terms(
    trajectory: dict[str, object],
    lengths: torch.Tensor,
    alphas: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Reuse Exp0.1.4 all-fixes P2/A1 exactly; only strength changes here."""
    return exp014.regularization_terms(
        trajectory,
        lengths,
        alphas,
        condition="all_fixes",
        tau=REG_TAU,
    )


def base_regularizer(p2: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
    return float(BASE_LAMBDA_P2) * p2 + float(BASE_LAMBDA_A1) * a1


def warmup_scale(epoch: int) -> float:
    if epoch < 1:
        raise ValueError("epoch must be >= 1")
    return min(1.0, float(epoch) / float(WARMUP_EPOCHS))


def calibrated_kappa(target_grad_ratio: float, calibration_ratio: float) -> float:
    if not math.isfinite(target_grad_ratio) or target_grad_ratio <= 0.0:
        raise ValueError("target_grad_ratio must be finite and positive")
    if not math.isfinite(calibration_ratio) or calibration_ratio <= 0.0:
        raise ValueError("calibration_ratio must be finite and positive")
    value = float(target_grad_ratio) / float(calibration_ratio)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("calibrated kappa must be finite and positive")
    return value


def _gradient_pair_stats(
    task_loss: torch.Tensor,
    reg_loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> dict[str, float]:
    task_grads = torch.autograd.grad(
        task_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    reg_grads = torch.autograd.grad(
        reg_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    task_sq = 0.0
    reg_sq = 0.0
    dot = 0.0
    for task_grad, reg_grad in zip(task_grads, reg_grads, strict=True):
        if task_grad is not None:
            task_sq += float(task_grad.detach().square().sum().item())
        if reg_grad is not None:
            reg_sq += float(reg_grad.detach().square().sum().item())
        if task_grad is not None and reg_grad is not None:
            dot += float((task_grad.detach() * reg_grad.detach()).sum().item())

    task_norm = math.sqrt(max(task_sq, 0.0))
    reg_norm = math.sqrt(max(reg_sq, 0.0))
    ratio = reg_norm / max(task_norm, 1e-12)
    cosine = dot / max(task_norm * reg_norm, 1e-12)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return {
        "task_hidden_grad_norm": task_norm,
        "reg_hidden_grad_norm": reg_norm,
        "reg_to_task_hidden_grad_ratio": ratio,
        "task_reg_hidden_grad_cosine": cosine,
    }


def calibrate_strength(
    spec: RunSpec,
    model: exp012.RegularizedMultiTauHierarchySNN,
    data: exp01.exp3.Data,
    config: Config,
) -> dict[str, object]:
    """One-time pre-training calibration on 5 independent train-only batches."""
    device = torch.device(config.device)
    loader = calibration_loader(data, spec, config.batch_size)
    alphas = _hidden_alphas(model)
    hidden_params = _hidden_parameters(model)
    rows: list[dict[str, float | int]] = []

    model.eval()
    for batch_index, (X, y, lengths) in enumerate(loader):
        if batch_index >= CALIBRATION_BATCHES:
            break
        X = X.to(device)
        y = y.to(device)
        lengths = lengths.to(device)
        trajectory = model.forward_trajectory(X)
        output_spikes = trajectory["output_spikes"]
        if not isinstance(output_spikes, torch.Tensor):
            raise TypeError("Expected output_spikes tensor")
        task_loss = exp01.exp50.objective_loss(
            output_spikes,
            lengths,
            y,
            spec.objective,
            OUTPUT_CAP,
            data.fs,
        )
        p2, a1, _ = all_fixes_terms(trajectory, lengths, alphas)
        reg0 = base_regularizer(p2, a1)
        stats = _gradient_pair_stats(task_loss, reg0, hidden_params)
        rows.append(
            {
                "batch_index": batch_index,
                "task_loss": float(task_loss.detach().item()),
                "p2": float(p2.detach().item()),
                "a1": float(a1.detach().item()),
                "base_regularizer": float(reg0.detach().item()),
                **stats,
            }
        )

    if len(rows) != CALIBRATION_BATCHES:
        raise RuntimeError(
            f"Calibration expected {CALIBRATION_BATCHES} batches, observed {len(rows)}"
        )
    ratios = np.asarray(
        [float(row["reg_to_task_hidden_grad_ratio"]) for row in rows],
        dtype=float,
    )
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0.0):
        raise RuntimeError("Calibration produced non-finite or non-positive gradient ratios")
    median_ratio = float(np.median(ratios))
    kappa = calibrated_kappa(spec.target_grad_ratio, median_ratio)
    return {
        "target_grad_ratio": float(spec.target_grad_ratio),
        "target_grad_percent": spec.target_percent,
        "calibration_batches": CALIBRATION_BATCHES,
        "calibration_ratio_median": median_ratio,
        "calibration_ratio_mean": float(np.mean(ratios)),
        "calibration_ratio_std": float(np.std(ratios, ddof=1)),
        "kappa": kappa,
        "full_strength_calibrated_ratio_median": kappa * median_ratio,
        "base_lambda_p2": BASE_LAMBDA_P2,
        "base_lambda_a1": BASE_LAMBDA_A1,
        "parameter_scope": "hidden linear weights only; output head excluded",
        "calibration_loader_seed": paired_seed(spec, "exp0_1_5_calibration_loader"),
        "batches": rows,
    }


def _validation_metrics(
    model: exp012.RegularizedMultiTauHierarchySNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    objective: str,
    fs: float,
) -> dict[str, float]:
    """WholeCount deployment metrics plus native objective loss in one pass."""
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    whole_ce_sum = 0.0
    objective_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for X, y, lengths in data_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            output_spikes = trajectory["output_spikes"]
            if not isinstance(output_spikes, torch.Tensor):
                raise TypeError("Expected output_spikes tensor")
            logits = exp01.exp50.deployment_logits(output_spikes, lengths, OUTPUT_CAP)
            ce_logits = exp01.exp50.deployment_ce_logits(output_spikes, lengths, OUTPUT_CAP)
            whole_ce = F.cross_entropy(ce_logits, y)
            objective_loss = exp01.exp50.objective_loss(
                output_spikes,
                lengths,
                y,
                objective,
                OUTPUT_CAP,
                fs,
            )
            n = len(y)
            n_total += n
            whole_ce_sum += float(whole_ce.item()) * n
            objective_sum += float(objective_loss.item()) * n
            labels.append(y.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())

    metrics = exp01._classification_metrics(
        np.concatenate(labels),
        np.concatenate(predictions),
    )
    metrics["whole_count_loss"] = whole_ce_sum / max(n_total, 1)
    metrics["objective_loss"] = objective_sum / max(n_total, 1)
    return metrics


def _early_stop_triggered(epoch: int, last_progress_epoch: int) -> bool:
    return (
        epoch >= MIN_EPOCHS
        and epoch - last_progress_epoch >= EARLY_STOP_PATIENCE
    )


def _experiment_manifest() -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "At what non-destructive but non-trivial hidden-gradient strength does the "
            "Exp0.1.4 all-fixes regularizer help Exp0.1 direct SNN classification?"
        ),
        "all_fixes_formulation": {
            "source": "Exp0.1.4 all_fixes",
            "potential_source": "hidden pre-reset membrane",
            "temporal_normalization": "divide cumulative EW endpoint by valid EW mass",
            "tau_gain_normalization": "multiply membrane by (1-alpha_i) only on the P2 path",
            "p2_reduction": "mean batch/neuron within layer, then mean layers",
            "a1": "Exp0.1.4 hidden valid-neuron-step firing mean, then mean layers",
            "base_regularizer": "0.01 * P2_all_fixes + 0.1 * A1",
            "warmup": "linear scale=min(1, epoch/10)",
        },
        "gradient_calibration": {
            "targets": list(TARGET_GRAD_RATIOS),
            "n_train_batches": CALIBRATION_BATCHES,
            "aggregation": "median per-batch ||grad R0|| / ||grad task||",
            "parameter_scope": "hidden linear weights only; output head excluded",
            "kappa": "target_ratio / calibration_ratio_median",
            "update_policy": "one-time before training; kappa is frozen for the full run",
        },
        "training": {
            "min_epochs": MIN_EPOCHS,
            "max_epochs": MAX_EPOCHS,
            "patience": EARLY_STOP_PATIENCE,
            "progress_signals": [
                "validation WholeCount balanced accuracy",
                "validation native objective loss",
            ],
            "val_ba_min_delta": VAL_BA_MIN_DELTA,
            "val_objective_loss_min_delta": VAL_OBJECTIVE_LOSS_MIN_DELTA,
            "checkpoint_selection": (
                "highest validation WholeCount BA; tie-break lower validation WholeCount CE"
            ),
        },
        "run_matrix": {
            "architectures": {
                key: [list(shifts) for shifts in value]
                for key, value in DIRECT_ARCHITECTURES.items()
            },
            "objectives": list(OBJECTIVES),
            "seeds": list(SEEDS),
            "target_grad_ratios": list(TARGET_GRAD_RATIOS),
            "new_run_count": len(run_specs()),
            "variant": VARIANT,
            "hidden_cap": HIDDEN_CAP,
            "output_cap": OUTPUT_CAP,
        },
        "frozen_baseline": {
            "experiment_id": BASE_EXPERIMENT_ID,
            "protocol_version": BASE_PROTOCOL_VERSION,
            "runs": 30,
            "note": "Exp0.1 100-epoch binary direct-SNN runs are reused; they are not equal-compute controls.",
        },
    }


def train_one(
    spec: RunSpec,
    data: exp01.exp3.Data,
    config: Config,
    force: bool,
) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    if config.max_epochs != MAX_EPOCHS:
        raise ValueError(f"Exp0.1.5 max_epochs is frozen at {MAX_EPOCHS}")

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp01.exp3.seed_all(paired_seed(spec, "model_init"))
    model = _new_model(spec, data).to(device)

    calibration = calibrate_strength(spec, model, data, config)
    _save_json(calibration_path(config.results_dir, spec), calibration)
    kappa = float(calibration["kappa"])

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loaders(data, spec, config.batch_size, train_shuffle=True)["train"]
    val_loader = make_loaders(data, spec, config.batch_size, train_shuffle=False)["val"]
    alphas = _hidden_alphas(model)
    hidden_params = _hidden_parameters(model)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_whole_loss = float("inf")

    best_progress_ba = -float("inf")
    best_progress_objective_loss = float("inf")
    last_progress_epoch = 0
    stop_reason = "max_epochs"
    history: list[dict[str, float | int | bool]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        scale = warmup_scale(epoch)
        task_sum = 0.0
        p2_sum = 0.0
        a1_sum = 0.0
        base_reg_sum = 0.0
        weighted_reg_sum = 0.0
        total_sum = 0.0
        p2_component_sums: list[float] | None = None
        n_total = 0
        gradient_row: dict[str, float] = {
            "first_batch_task_hidden_grad_norm": np.nan,
            "first_batch_base_reg_hidden_grad_norm": np.nan,
            "first_batch_base_reg_to_task_hidden_grad_ratio": np.nan,
            "first_batch_effective_reg_to_task_hidden_grad_ratio": np.nan,
            "first_batch_task_reg_hidden_grad_cosine": np.nan,
        }

        for batch_index, (X, y, lengths) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            output_spikes = trajectory["output_spikes"]
            if not isinstance(output_spikes, torch.Tensor):
                raise TypeError("Expected output_spikes tensor")

            task_loss = exp01.exp50.objective_loss(
                output_spikes,
                lengths,
                y,
                spec.objective,
                OUTPUT_CAP,
                data.fs,
            )
            p2, a1, p2_components = all_fixes_terms(trajectory, lengths, alphas)
            reg0 = base_regularizer(p2, a1)
            multiplier = scale * kappa
            weighted_reg = multiplier * reg0
            total_loss = task_loss + weighted_reg

            if batch_index == 0 and epoch in GRAD_DIAGNOSTIC_EPOCHS:
                stats = _gradient_pair_stats(task_loss, reg0, hidden_params)
                gradient_row = {
                    "first_batch_task_hidden_grad_norm": stats["task_hidden_grad_norm"],
                    "first_batch_base_reg_hidden_grad_norm": stats["reg_hidden_grad_norm"],
                    "first_batch_base_reg_to_task_hidden_grad_ratio": stats[
                        "reg_to_task_hidden_grad_ratio"
                    ],
                    "first_batch_effective_reg_to_task_hidden_grad_ratio": (
                        multiplier * stats["reg_to_task_hidden_grad_ratio"]
                    ),
                    "first_batch_task_reg_hidden_grad_cosine": stats[
                        "task_reg_hidden_grad_cosine"
                    ],
                }

            total_loss.backward()
            optimizer.step()

            n = len(y)
            n_total += n
            task_sum += float(task_loss.item()) * n
            p2_sum += float(p2.item()) * n
            a1_sum += float(a1.item()) * n
            base_reg_sum += float(reg0.item()) * n
            weighted_reg_sum += float(weighted_reg.item()) * n
            total_sum += float(total_loss.item()) * n
            if p2_component_sums is None:
                p2_component_sums = [0.0] * len(p2_components)
            for index, component in enumerate(p2_components):
                p2_component_sums[index] += float(component.item()) * n

        if n_total == 0 or p2_component_sums is None:
            raise RuntimeError("Training loader produced no batches")

        val = _validation_metrics(model, val_loader, device, spec.objective, data.fs)
        val_ba = float(val["balanced_accuracy"])
        val_whole_loss = float(val["whole_count_loss"])
        val_objective_loss = float(val["objective_loss"])

        checkpoint_improved = val_ba > best_val_ba + 1e-12 or (
            abs(val_ba - best_val_ba) <= 1e-12
            and val_whole_loss < best_val_whole_loss
        )
        if checkpoint_improved:
            best_val_ba = val_ba
            best_val_whole_loss = val_whole_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        ba_progress = val_ba > best_progress_ba + VAL_BA_MIN_DELTA
        loss_progress = (
            val_objective_loss
            < best_progress_objective_loss - VAL_OBJECTIVE_LOSS_MIN_DELTA
        )
        if ba_progress:
            best_progress_ba = val_ba
        if loss_progress:
            best_progress_objective_loss = val_objective_loss
        if ba_progress or loss_progress:
            last_progress_epoch = epoch

        should_stop = _early_stop_triggered(epoch, last_progress_epoch)
        task_mean = task_sum / n_total
        base_reg_mean = base_reg_sum / n_total
        weighted_reg_mean = weighted_reg_sum / n_total
        row: dict[str, float | int | bool] = {
            "epoch": epoch,
            "target_grad_ratio": float(spec.target_grad_ratio),
            "calibration_ratio_median": float(calibration["calibration_ratio_median"]),
            "kappa": kappa,
            "warmup_scale": scale,
            "regularizer_multiplier": scale * kappa,
            "effective_lambda_p2": scale * kappa * BASE_LAMBDA_P2,
            "effective_lambda_a1": scale * kappa * BASE_LAMBDA_A1,
            "train_task_loss": task_mean,
            "train_p2": p2_sum / n_total,
            "train_a1": a1_sum / n_total,
            "train_base_regularizer": base_reg_mean,
            "train_weighted_regularizer": weighted_reg_mean,
            "train_total_loss": total_sum / n_total,
            "train_weighted_reg_to_task_loss_ratio": weighted_reg_mean / max(task_mean, 1e-12),
            "val_wholecount_ba": val_ba,
            "val_wholecount_loss": val_whole_loss,
            "val_objective_loss": val_objective_loss,
            "last_progress_epoch": last_progress_epoch,
            "epochs_since_progress": epoch - last_progress_epoch,
            "early_stop_triggered": should_stop,
            **gradient_row,
        }
        for index, value in enumerate(p2_component_sums, start=1):
            row[f"train_p2_hidden_{index}"] = value / n_total
        history.append(row)

        if should_stop:
            stop_reason = "patience_30_after_min50"
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")

    epochs_trained = int(history[-1]["epoch"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "calibration": calibration,
            "best_epoch": best_epoch,
            "best_val_wholecount_ba": best_val_ba,
            "best_val_wholecount_loss": best_val_whole_loss,
            "epochs_trained": epochs_trained,
            "stop_reason": stop_reason,
            "early_stopped": stop_reason != "max_epochs",
            "model_state_dict": best_state,
            "labels": data.labels,
            "split": data.split,
            "architecture": exp01.architecture_manifest(
                base_spec(spec), data.fs, len(data.labels)
            ),
            "experiment_manifest": _experiment_manifest(),
        },
        destination,
    )
    history_file = history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: exp01.exp3.Data,
    config: Config,
) -> tuple[exp012.RegularizedMultiTauHierarchySNN, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp0.1.5 checkpoint: {path}")
    device = torch.device(config.device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong experiment id in {path}")
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong protocol version in {path}")
    if checkpoint.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint identity mismatch for {spec.key}")
    model = _new_model(spec, data).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def evaluate_one(
    spec: RunSpec,
    data: exp01.exp3.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model, checkpoint = load_model(spec, data, config)
    loaders = make_loaders(data, spec, config.batch_size, train_shuffle=False)
    metrics = {
        split: exp01._direct_metrics(model, loader, device)
        for split, loader in loaders.items()
    }
    event_rates = {
        split: exp01._last_hidden_event_rate(model, loader, device, data.fs)
        for split, loader in loaders.items()
    }
    diagnostics = {
        split: exp012._hidden_diagnostics(model, loader, device, data.fs, HIDDEN_CAP)
        for split, loader in loaders.items()
    }
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "calibration": checkpoint["calibration"],
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_wholecount_ba": float(checkpoint["best_val_wholecount_ba"]),
        "best_val_wholecount_loss": float(checkpoint["best_val_wholecount_loss"]),
        "epochs_trained": int(checkpoint["epochs_trained"]),
        "stop_reason": str(checkpoint["stop_reason"]),
        "early_stopped": bool(checkpoint["early_stopped"]),
        "architecture": checkpoint["architecture"],
        "primary_readout": "output_whole_count",
        "metrics": metrics,
        "last_hidden_events_per_neuron_second": event_rates,
        "hidden_diagnostics": diagnostics,
        "provenance": {
            "base_experiment_id": BASE_EXPERIMENT_ID,
            "base_protocol_version": BASE_PROTOCOL_VERSION,
            "paired_frozen_key": base_spec(spec).key,
            "split_seed": int(exp01.exp3.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "input_contract": "identical to Exp0.1 direct SNN: Raw64 30-channel unsigned weighted events",
            "checkpoint_selection": "validation Output WholeCount BA; tie-break normalized WholeCount CE",
        },
    }
    _save_json(destination, payload)
    return payload


def run_one(
    spec: RunSpec,
    data: exp01.exp3.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    train_one(spec, data, config, force)
    return evaluate_one(spec, data, config, force)


def _row_from_payload(spec: RunSpec, payload: dict[str, object]) -> dict[str, object]:
    metrics = payload["metrics"]
    diagnostics = payload["hidden_diagnostics"]
    calibration = payload["calibration"]
    if not isinstance(metrics, dict) or not isinstance(diagnostics, dict):
        raise TypeError("Malformed Exp0.1.5 evaluation payload")
    if not isinstance(calibration, dict):
        raise TypeError("Malformed calibration payload")
    train = metrics["train"]
    val = metrics["val"]
    test = metrics["test"]
    test_diag = diagnostics["test"]
    if not isinstance(test_diag, dict):
        raise TypeError("Malformed test diagnostics")
    layers = test_diag["layers"]
    if not isinstance(layers, list) or not layers:
        raise TypeError("Missing hidden layer diagnostics")
    last = layers[-1]
    return {
        "family": DIRECT_FAMILY,
        "target_grad_ratio": float(spec.target_grad_ratio),
        "target_grad_percent": spec.target_percent,
        "architecture": spec.architecture,
        "objective": spec.objective,
        "variant": VARIANT,
        "hidden_cap": HIDDEN_CAP,
        "output_cap": OUTPUT_CAP,
        "seed": spec.seed,
        "epochs_trained": int(payload["epochs_trained"]),
        "early_stopped": bool(payload["early_stopped"]),
        "stop_reason": str(payload["stop_reason"]),
        "best_epoch": int(payload["best_epoch"]),
        "calibration_ratio_median": float(calibration["calibration_ratio_median"]),
        "kappa": float(calibration["kappa"]),
        "train_ba": float(train["balanced_accuracy"]),
        "val_ba": float(val["balanced_accuracy"]),
        "test_ba": float(test["balanced_accuracy"]),
        "test_accuracy": float(test["accuracy"]),
        "test_macro_f1": float(test["macro_f1"]),
        "test_last_hidden_events_per_neuron_second": float(
            payload["last_hidden_events_per_neuron_second"]["test"]
        ),
        "test_mean_dead_neuron_fraction": float(
            np.mean([float(layer["dead_neuron_fraction"]) for layer in layers])
        ),
        "test_last_hidden_dead_neuron_fraction": float(last["dead_neuron_fraction"]),
        "test_mean_active_step_fraction": float(
            np.mean([float(layer["active_step_fraction"]) for layer in layers])
        ),
        "test_last_hidden_active_step_fraction": float(last["active_step_fraction"]),
    }


def _load_frozen_exp01_binary_runs(repo_root: Path) -> pd.DataFrame:
    path = base_results_dir(repo_root) / "runs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen Exp0.1 runs.csv: {path}")
    runs = pd.read_csv(path)
    direct = runs[
        (runs["family"] == DIRECT_FAMILY)
        & (runs["variant"] == VARIANT)
        & (runs["hidden_cap"] == HIDDEN_CAP)
    ].copy()
    expected = {
        (architecture, objective, seed)
        for architecture in DIRECT_ARCHITECTURES
        for objective in OBJECTIVES
        for seed in SEEDS
    }
    observed = set(
        direct[["architecture", "objective", "seed"]]
        .itertuples(index=False, name=None)
    )
    if len(direct) != 30 or observed != expected:
        raise ValueError("Frozen Exp0.1 binary direct runs do not match the 30 expected identities")
    direct["context_label"] = "frozen_exp0.1_no_reg_100epoch"
    return direct


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in out.columns
    ]
    return out


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    history_frames: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []
    for spec in run_specs():
        eval_file = evaluation_path(root, spec)
        history_file = history_path(root, spec)
        calib_file = calibration_path(root, spec)
        for required in (eval_file, history_file, calib_file):
            if not required.exists():
                raise FileNotFoundError(f"Missing required Exp0.1.5 artifact: {required}")

        payload = json.loads(eval_file.read_text(encoding="utf-8"))
        rows.append(_row_from_payload(spec, payload))

        history = pd.read_csv(history_file)
        history.insert(0, "seed", spec.seed)
        history.insert(0, "objective", spec.objective)
        history.insert(0, "architecture", spec.architecture)
        history_frames.append(history)

        calibration = json.loads(calib_file.read_text(encoding="utf-8"))
        batch_rows = calibration.get("batches")
        if not isinstance(batch_rows, list) or len(batch_rows) != CALIBRATION_BATCHES:
            raise ValueError(f"Malformed calibration batches for {spec.key}")
        cosines = [float(item["task_reg_hidden_grad_cosine"]) for item in batch_rows]
        calibration_rows.append(
            {
                "target_grad_ratio": spec.target_grad_ratio,
                "architecture": spec.architecture,
                "objective": spec.objective,
                "seed": spec.seed,
                "calibration_ratio_median": float(calibration["calibration_ratio_median"]),
                "calibration_ratio_mean": float(calibration["calibration_ratio_mean"]),
                "calibration_ratio_std": float(calibration["calibration_ratio_std"]),
                "kappa": float(calibration["kappa"]),
                "calibration_cosine_mean": float(np.mean(cosines)),
                "calibration_cosine_median": float(np.median(cosines)),
            }
        )

    runs = pd.DataFrame(rows)
    runs_file = root / "runs.csv"
    runs.to_csv(runs_file, index=False)

    history_long = pd.concat(history_frames, ignore_index=True, sort=False)
    history_file = root / "history_long.csv"
    history_long.to_csv(history_file, index=False)

    calibrations = pd.DataFrame(calibration_rows)
    calibrations_file = root / "calibrations.csv"
    calibrations.to_csv(calibrations_file, index=False)

    summary = (
        runs.groupby(["target_grad_ratio", "architecture", "objective"], dropna=False)[
            [
                "train_ba",
                "val_ba",
                "test_ba",
                "test_accuracy",
                "test_macro_f1",
                "test_last_hidden_events_per_neuron_second",
                "test_last_hidden_dead_neuron_fraction",
                "epochs_trained",
                "best_epoch",
                "kappa",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary = _flatten_columns(summary)
    summary_file = root / "summary.csv"
    summary.to_csv(summary_file, index=False)

    frozen = _load_frozen_exp01_binary_runs(repo_root)
    frozen_file = root / "frozen_exp01_binary_context.csv"
    frozen.to_csv(frozen_file, index=False)
    frozen_index = frozen.set_index(["architecture", "objective", "seed"])

    paired_rows: list[dict[str, object]] = []
    for row in runs.itertuples(index=False):
        baseline = frozen_index.loc[(row.architecture, row.objective, row.seed)]
        paired_rows.append(
            {
                "target_grad_ratio": row.target_grad_ratio,
                "architecture": row.architecture,
                "objective": row.objective,
                "seed": row.seed,
                "exp015_test_ba": row.test_ba,
                "frozen_exp01_test_ba": float(baseline["test_ba"]),
                "delta_test_ba_vs_frozen_exp01": row.test_ba - float(baseline["test_ba"]),
                "exp015_last_hidden_events_per_neuron_second": row.test_last_hidden_events_per_neuron_second,
                "frozen_exp01_last_hidden_events_per_neuron_second": float(
                    baseline["test_last_hidden_events_per_neuron_second"]
                ),
                "delta_last_hidden_events_per_neuron_second_vs_frozen_exp01": (
                    row.test_last_hidden_events_per_neuron_second
                    - float(baseline["test_last_hidden_events_per_neuron_second"])
                ),
                "epochs_trained": row.epochs_trained,
                "best_epoch": row.best_epoch,
                "early_stopped": row.early_stopped,
                "kappa": row.kappa,
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired_file = root / "paired_vs_frozen_exp01.csv"
    paired.to_csv(paired_file, index=False)

    paired_summary = (
        paired.groupby(["target_grad_ratio", "architecture", "objective"], dropna=False)[
            [
                "delta_test_ba_vs_frozen_exp01",
                "delta_last_hidden_events_per_neuron_second_vs_frozen_exp01",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    paired_summary = _flatten_columns(paired_summary)
    paired_summary_file = root / "paired_vs_frozen_exp01_summary.csv"
    paired_summary.to_csv(paired_summary_file, index=False)

    calibration_summary = (
        calibrations.groupby(["target_grad_ratio", "architecture", "objective"], dropna=False)[
            [
                "calibration_ratio_median",
                "kappa",
                "calibration_cosine_mean",
                "calibration_cosine_median",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    calibration_summary = _flatten_columns(calibration_summary)
    calibration_summary_file = root / "calibration_summary.csv"
    calibration_summary.to_csv(calibration_summary_file, index=False)

    grad = history_long[
        history_long["first_batch_effective_reg_to_task_hidden_grad_ratio"].notna()
    ].copy()
    gradient_summary = (
        grad.groupby(
            ["target_grad_ratio", "architecture", "objective", "epoch"],
            dropna=False,
        )[
            [
                "first_batch_base_reg_to_task_hidden_grad_ratio",
                "first_batch_effective_reg_to_task_hidden_grad_ratio",
                "first_batch_task_reg_hidden_grad_cosine",
                "val_wholecount_ba",
                "val_objective_loss",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    gradient_summary = _flatten_columns(gradient_summary)
    gradient_summary_file = root / "gradient_trajectory_summary.csv"
    gradient_summary.to_csv(gradient_summary_file, index=False)

    stopping = runs.copy()
    stopping["early_stopped_int"] = stopping["early_stopped"].astype(int)
    stopping_summary = (
        stopping.groupby(["target_grad_ratio", "architecture", "objective"], dropna=False)[
            ["epochs_trained", "best_epoch", "early_stopped_int"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stopping_summary = _flatten_columns(stopping_summary)
    stopping_summary_file = root / "stopping_summary.csv"
    stopping_summary.to_csv(stopping_summary_file, index=False)

    frozen_summary = (
        frozen.groupby(["architecture", "objective"], dropna=False)["test_ba"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    frozen_summary_file = root / "frozen_exp01_summary.csv"
    frozen_summary.to_csv(frozen_summary_file, index=False)

    comparison_rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        comparison_rows.append(
            {
                "system": "exp0.1.5_gradient_calibrated_all_fixes",
                "target_grad_ratio": row.target_grad_ratio,
                "architecture": row.architecture,
                "objective": row.objective,
                "mean_test_ba": row.test_ba_mean,
                "sd_test_ba": row.test_ba_std,
                "n": row.test_ba_count,
                "training_budget": "50-100 epochs, patience-30 after epoch 50",
            }
        )
    for row in frozen_summary.itertuples(index=False):
        comparison_rows.append(
            {
                "system": "frozen_exp0.1_no_reg",
                "target_grad_ratio": 0.0,
                "architecture": row.architecture,
                "objective": row.objective,
                "mean_test_ba": row.mean,
                "sd_test_ba": row.std,
                "n": row.count,
                "training_budget": "frozen Exp0.1 100 epochs",
            }
        )
    comparison_file = root / "comparison_summary.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_file, index=False)

    manifest_file = root / "manifest.json"
    _save_json(manifest_file, _experiment_manifest())

    return {
        "runs": runs_file,
        "summary": summary_file,
        "history_long": history_file,
        "calibrations": calibrations_file,
        "calibration_summary": calibration_summary_file,
        "gradient_trajectory_summary": gradient_summary_file,
        "stopping_summary": stopping_summary_file,
        "paired_vs_frozen_exp01": paired_file,
        "paired_vs_frozen_exp01_summary": paired_summary_file,
        "frozen_exp01_binary_context": frozen_file,
        "frozen_exp01_summary": frozen_summary_file,
        "comparison_summary": comparison_file,
        "manifest": manifest_file,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-one")
    run_parser.add_argument("--array-task-id", type=int, required=True)
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--threads", type=int, default=1)
    run_parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    run_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("finalize")
    return parser


def _config_from_args(args: argparse.Namespace, repo_root: Path) -> Config:
    return Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=getattr(args, "device", "cpu"),
        max_epochs=getattr(args, "max_epochs", MAX_EPOCHS),
        batch_size=getattr(args, "batch_size", BATCH_SIZE),
        threads=getattr(args, "threads", 1),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = exp01.exp3.find_repo_root()
    if args.command == "finalize":
        for name, path in finalize_experiment(repo_root).items():
            print(f"{name}: {path}")
        return

    specs = run_specs()
    if args.array_task_id < 0 or args.array_task_id >= len(specs):
        raise IndexError(
            f"array task {args.array_task_id} outside [0, {len(specs) - 1}]"
        )
    spec = specs[args.array_task_id]
    data = exp01.prepare_data(repo_root)
    config = _config_from_args(args, repo_root)
    payload = run_one(spec, data, config, args.force)
    test = payload["metrics"]["test"]
    print(
        f"completed {spec.key}: test_BA={test['balanced_accuracy']:.6f} "
        f"epochs={payload['epochs_trained']} best_epoch={payload['best_epoch']} "
        f"kappa={payload['calibration']['kappa']:.6g} stop={payload['stop_reason']}"
    )


if __name__ == "__main__":
    main()
