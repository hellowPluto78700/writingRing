from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_0_1_2_regularized_general_comparison as exp012


EXPERIMENT_ID = "experiment_0_1_4_parallel_regularization_ablation"
PROTOCOL_VERSION = "parallel_regularization_ablation_v1"
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

CONDITIONS = (
    "no_reg",
    "original_reg",
    "temporal_norm",
    "tau_gain_norm",
    "warmup",
    "all_fixes",
)

REG_TAU = 0.99
REG_LAMBDA_P2 = 0.01
REG_LAMBDA_A1 = 0.1
WARMUP_EPOCHS = 10
GRAD_DIAGNOSTIC_EPOCHS = (1, 2, 5, 10, 20, 30)

BATCH_SIZE = exp01.BATCH_SIZE
EPOCHS = 30
LR = exp01.LR
WEIGHT_DECAY = exp01.WEIGHT_DECAY


@dataclass(frozen=True)
class ConditionConfig:
    enabled: bool
    temporal_normalization: bool
    tau_gain_normalization: bool
    warmup: bool


CONDITION_CONFIGS: dict[str, ConditionConfig] = {
    "no_reg": ConditionConfig(False, False, False, False),
    "original_reg": ConditionConfig(True, False, False, False),
    "temporal_norm": ConditionConfig(True, True, False, False),
    "tau_gain_norm": ConditionConfig(True, False, True, False),
    "warmup": ConditionConfig(True, False, False, True),
    "all_fixes": ConditionConfig(True, True, True, True),
}


@dataclass(frozen=True)
class RunSpec:
    condition: str
    architecture: str
    objective: str
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{DIRECT_FAMILY}__{self.architecture}__{self.objective}__"
            f"{VARIANT}__hcap{HIDDEN_CAP}__regab_{self.condition}__seed{self.seed}"
        )


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def base_results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / BASE_EXPERIMENT_ID / BASE_PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(condition, architecture, objective, seed)
        for condition in CONDITIONS
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
    """Return a condition-independent seed for strict paired comparisons."""
    return exp01.paired_seed(base_spec(spec), role)


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _condition_config(condition: str) -> ConditionConfig:
    try:
        return CONDITION_CONFIGS[condition]
    except KeyError as exc:
        raise ValueError(f"Unknown condition: {condition}") from exc


def _lambda_scale(condition: str, epoch: int) -> float:
    cfg = _condition_config(condition)
    if not cfg.enabled:
        return 0.0
    if not cfg.warmup:
        return 1.0
    if epoch < 1:
        raise ValueError("epoch must be >= 1")
    return min(1.0, float(epoch) / float(WARMUP_EPOCHS))


def _condition_manifest(condition: str) -> dict[str, object]:
    cfg = _condition_config(condition)
    return {
        "condition": condition,
        "enabled": cfg.enabled,
        "potential_source": "hidden_pre_reset_membrane",
        "potential_layers": "all hidden spiking layers only",
        "activity_layers": "all hidden spiking layers only",
        "output_layer_regularized": False,
        "tau": REG_TAU,
        "lambda_p2_max": REG_LAMBDA_P2 if cfg.enabled else 0.0,
        "lambda_a1_max": REG_LAMBDA_A1 if cfg.enabled else 0.0,
        "temporal_normalization": cfg.temporal_normalization,
        "temporal_normalization_definition": (
            "divide each sample-neuron cumulative endpoint by its valid EW mass"
            if cfg.temporal_normalization
            else "none; Exp0.1.2 cumulative EW sum"
        ),
        "tau_gain_normalization": cfg.tau_gain_normalization,
        "tau_gain_normalization_definition": (
            "multiply each hidden pre-reset membrane by (1-alpha_i) only inside P2"
            if cfg.tau_gain_normalization
            else "none"
        ),
        "warmup": cfg.warmup,
        "warmup_epochs": WARMUP_EPOCHS if cfg.warmup else 0,
        "warmup_schedule": (
            "linear scale=min(1, epoch/10) applied to both P2 and A1"
            if cfg.warmup
            else "fixed"
        ),
        "p2_reduction": "mean over batch and neurons within each layer, then mean over layers",
        "a1_definition": "Exp0.1.2 mean normalized hidden firing activity over valid neuron-steps, then mean over layers",
        "diagnostic_note": (
            "no_reg still reports the original Exp0.1.2 P2/A1 values for observation, "
            "but both optimization weights are zero"
            if condition == "no_reg"
            else "regularization values are part of the optimization loss according to this condition"
        ),
    }


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


def _hidden_alphas(model: exp012.RegularizedMultiTauHierarchySNN) -> tuple[torch.Tensor, ...]:
    return tuple(getattr(model, name) for name in model._alpha_names)


def regularization_terms(
    trajectory: dict[str, object],
    lengths: torch.Tensor,
    alphas: tuple[torch.Tensor, ...],
    condition: str,
    tau: float = REG_TAU,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Return P2/A1 plus per-layer P2 under one Exp0.1.4 condition.

    `original_reg`, `warmup`, and diagnostic-only `no_reg` reproduce the
    Exp0.1.2 P2/A1 definitions. `temporal_norm` divides the cumulative
    endpoint by the valid exponentially weighted mass. `tau_gain_norm`
    multiplies each pre-reset membrane by (1-alpha_i) only on the P2 penalty
    path. `all_fixes` combines both normalizations; warmup is handled by the
    epoch-dependent lambda schedule, not by this function.
    """
    cfg = _condition_config(condition)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")

    spikes = trajectory.get("hidden_spikes")
    pre_reset = trajectory.get("hidden_pre_reset")
    if not isinstance(spikes, tuple) or not isinstance(pre_reset, tuple):
        raise TypeError("Expected hidden_spikes and hidden_pre_reset tuples")
    if len(spikes) != len(pre_reset) or not spikes:
        raise ValueError("Hidden trace tuples must be non-empty and aligned")
    if len(alphas) != len(pre_reset):
        raise ValueError("alpha tuple must match hidden layer count")

    p2_layers: list[torch.Tensor] = []
    a1_layers: list[torch.Tensor] = []
    for layer_spikes, layer_pre, alpha in zip(spikes, pre_reset, alphas, strict=True):
        if not isinstance(layer_spikes, torch.Tensor) or not isinstance(layer_pre, torch.Tensor):
            raise TypeError("Hidden traces must contain tensors")
        if layer_spikes.shape != layer_pre.shape:
            raise ValueError("Hidden spike and membrane traces must have equal shapes")
        batch, n_steps, width = layer_pre.shape
        if alpha.ndim != 1 or alpha.shape[0] != width:
            raise ValueError("alpha vector must have one value per hidden neuron")

        valid = exp01.exp50.valid_mask(lengths, n_steps)
        if valid.shape[0] != batch:
            raise ValueError("lengths batch dimension disagrees with trajectory")

        penalty_membrane = layer_pre
        if cfg.tau_gain_normalization:
            gain_scale = (1.0 - alpha).to(layer_pre.dtype).view(1, 1, width)
            penalty_membrane = layer_pre * gain_scale

        cumulative = torch.zeros(batch, width, device=layer_pre.device, dtype=layer_pre.dtype)
        ew_mass = torch.zeros(batch, device=layer_pre.device, dtype=layer_pre.dtype)
        for step in range(n_steps):
            valid_step = valid[:, step]
            updated = float(tau) * cumulative + penalty_membrane[:, step]
            cumulative = torch.where(valid_step.unsqueeze(-1), updated, cumulative)
            if cfg.temporal_normalization:
                updated_mass = float(tau) * ew_mass + 1.0
                ew_mass = torch.where(valid_step, updated_mass, ew_mass)

        if cfg.temporal_normalization:
            cumulative = cumulative / ew_mass.clamp_min(1e-12).unsqueeze(-1)
        p2_layers.append(cumulative.square().mean())

        valid_f = valid.to(layer_spikes.dtype).unsqueeze(-1)
        denominator = valid_f.sum() * float(width)
        if float(denominator.detach().item()) <= 0.0:
            raise ValueError("Regularization requires at least one valid timestep")
        a1_layers.append((layer_spikes * valid_f).sum() / denominator)

    return torch.stack(p2_layers).mean(), torch.stack(a1_layers).mean(), tuple(p2_layers)


def _grad_norm(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> float:
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    squared = 0.0
    for grad in grads:
        if grad is not None:
            squared += float(grad.detach().square().sum().item())
    return math.sqrt(max(squared, 0.0))


def _gradient_diagnostics(
    task_loss: torch.Tensor,
    p2: torch.Tensor,
    a1: torch.Tensor,
    lambda_p2: float,
    lambda_a1: float,
    model: torch.nn.Module,
) -> dict[str, float]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    task_norm = _grad_norm(task_loss, parameters)
    p2_raw_norm = _grad_norm(p2, parameters)
    a1_raw_norm = _grad_norm(a1, parameters)
    weighted_p2 = float(lambda_p2) * p2
    weighted_a1 = float(lambda_a1) * a1
    weighted_p2_norm = _grad_norm(weighted_p2, parameters)
    weighted_a1_norm = _grad_norm(weighted_a1, parameters)
    reg_norm = _grad_norm(weighted_p2 + weighted_a1, parameters)
    denominator = max(task_norm, 1e-12)
    return {
        "first_batch_task_grad_norm": task_norm,
        "first_batch_p2_grad_norm_raw": p2_raw_norm,
        "first_batch_a1_grad_norm_raw": a1_raw_norm,
        "first_batch_weighted_p2_grad_norm": weighted_p2_norm,
        "first_batch_weighted_a1_grad_norm": weighted_a1_norm,
        "first_batch_reg_grad_norm": reg_norm,
        "first_batch_p2_to_task_grad_ratio": weighted_p2_norm / denominator,
        "first_batch_a1_to_task_grad_ratio": weighted_a1_norm / denominator,
        "first_batch_reg_to_task_grad_ratio": reg_norm / denominator,
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

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp01.exp3.seed_all(paired_seed(spec, "model_init"))
    model = _new_model(spec, data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loaders(data, spec, config.batch_size, train_shuffle=True)["train"]
    val_loader = make_loaders(data, spec, config.batch_size, train_shuffle=False)["val"]

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        scale = _lambda_scale(spec.condition, epoch)
        lambda_p2 = REG_LAMBDA_P2 * scale
        lambda_a1 = REG_LAMBDA_A1 * scale
        task_sum = 0.0
        p2_sum = 0.0
        a1_sum = 0.0
        weighted_p2_sum = 0.0
        weighted_a1_sum = 0.0
        total_sum = 0.0
        p2_component_sums: list[float] | None = None
        n_total = 0
        gradient_row = {
            "first_batch_task_grad_norm": np.nan,
            "first_batch_p2_grad_norm_raw": np.nan,
            "first_batch_a1_grad_norm_raw": np.nan,
            "first_batch_weighted_p2_grad_norm": np.nan,
            "first_batch_weighted_a1_grad_norm": np.nan,
            "first_batch_reg_grad_norm": np.nan,
            "first_batch_p2_to_task_grad_ratio": np.nan,
            "first_batch_a1_to_task_grad_ratio": np.nan,
            "first_batch_reg_to_task_grad_ratio": np.nan,
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
            p2, a1, p2_components = regularization_terms(
                trajectory,
                lengths,
                _hidden_alphas(model),
                spec.condition,
            )
            weighted_p2 = float(lambda_p2) * p2
            weighted_a1 = float(lambda_a1) * a1
            total_loss = task_loss + weighted_p2 + weighted_a1

            if batch_index == 0 and epoch in GRAD_DIAGNOSTIC_EPOCHS:
                gradient_row = _gradient_diagnostics(
                    task_loss,
                    p2,
                    a1,
                    lambda_p2,
                    lambda_a1,
                    model,
                )

            total_loss.backward()
            optimizer.step()

            n = len(y)
            n_total += n
            task_sum += float(task_loss.item()) * n
            p2_sum += float(p2.item()) * n
            a1_sum += float(a1.item()) * n
            weighted_p2_sum += float(weighted_p2.item()) * n
            weighted_a1_sum += float(weighted_a1.item()) * n
            total_sum += float(total_loss.item()) * n
            if p2_component_sums is None:
                p2_component_sums = [0.0] * len(p2_components)
            for index, component in enumerate(p2_components):
                p2_component_sums[index] += float(component.item()) * n

        if n_total == 0 or p2_component_sums is None:
            raise RuntimeError("Training loader produced no batches")

        val_metrics = exp01._direct_metrics(model, val_loader, device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        task_mean = task_sum / n_total
        p2_mean = p2_sum / n_total
        a1_mean = a1_sum / n_total
        weighted_p2_mean = weighted_p2_sum / n_total
        weighted_a1_mean = weighted_a1_sum / n_total
        row: dict[str, float | int] = {
            "epoch": epoch,
            "lambda_scale": scale,
            "lambda_p2": lambda_p2,
            "lambda_a1": lambda_a1,
            "train_task_loss": task_mean,
            "train_p2": p2_mean,
            "train_a1": a1_mean,
            "train_weighted_p2": weighted_p2_mean,
            "train_weighted_a1": weighted_a1_mean,
            "train_total_loss": total_sum / n_total,
            "train_reg_to_task_ratio": (
                weighted_p2_mean + weighted_a1_mean
            ) / max(task_mean, 1e-12),
            "val_native_ba": val_ba,
            "val_native_loss": val_loss,
            **gradient_row,
        }
        for index, value in enumerate(p2_component_sums, start=1):
            row[f"train_p2_hidden_{index}"] = value / n_total
        history.append(row)

        improved = val_ba > best_val_ba + 1e-12 or (
            abs(val_ba - best_val_ba) <= 1e-12 and val_loss < best_val_loss
        )
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "condition": _condition_manifest(spec.condition),
            "best_epoch": best_epoch,
            "best_val_native_ba": best_val_ba,
            "best_val_native_loss": best_val_loss,
            "model_state_dict": best_state,
            "labels": data.labels,
            "split": data.split,
            "architecture": exp01.architecture_manifest(
                base_spec(spec), data.fs, len(data.labels)
            ),
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
        raise FileNotFoundError(f"Missing Exp0.1.4 checkpoint: {path}")
    device = torch.device(config.device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong experiment id in {path}")
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong protocol version in {path}")
    if checkpoint.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint identity mismatch for {spec.key}")
    if checkpoint.get("condition") != _condition_manifest(spec.condition):
        raise ValueError(f"Condition contract mismatch for {spec.key}")
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
        "condition": _condition_manifest(spec.condition),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_native_ba": float(checkpoint["best_val_native_ba"]),
        "best_val_native_loss": float(checkpoint["best_val_native_loss"]),
        "architecture": exp01.architecture_manifest(
            base_spec(spec), data.fs, len(data.labels)
        ),
        "primary_readout": "output_whole_count",
        "metrics": metrics,
        "native_metrics": metrics,
        "last_hidden_events_per_neuron_second": event_rates,
        "hidden_diagnostics": diagnostics,
        "provenance": {
            "base_experiment_id": BASE_EXPERIMENT_ID,
            "base_protocol_version": BASE_PROTOCOL_VERSION,
            "paired_base_key": base_spec(spec).key,
            "split_seed": int(exp01.exp3.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "input_contract": "identical to Exp0.1 direct SNN: Raw64 30-channel unsigned weighted events",
            "training_contract": (
                "binary Exp0.1 direct-SNN architecture, optimizer, LR, batch size, input, "
                "objective implementations, paired seeds, readout, and checkpoint selection are preserved; "
                "training budget is fixed at 30 epochs and only the regularization condition varies"
            ),
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
    if not isinstance(metrics, dict) or not isinstance(diagnostics, dict):
        raise TypeError("Malformed evaluation payload")
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
        "condition": spec.condition,
        "architecture": spec.architecture,
        "objective": spec.objective,
        "variant": VARIANT,
        "hidden_cap": HIDDEN_CAP,
        "output_cap": OUTPUT_CAP,
        "seed": spec.seed,
        "epochs": EPOCHS,
        "primary_readout": "output_whole_count",
        "best_epoch": int(payload["best_epoch"]),
        "train_ba": float(train["balanced_accuracy"]),
        "val_ba": float(val["balanced_accuracy"]),
        "test_ba": float(test["balanced_accuracy"]),
        "test_accuracy": float(test["accuracy"]),
        "test_macro_f1": float(test["macro_f1"]),
        "native_test_ba": float(test["balanced_accuracy"]),
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
        "test_mean_abs_pre_reset_membrane": float(
            np.mean([float(layer["mean_abs_pre_reset_membrane"]) for layer in layers])
        ),
        "test_last_hidden_mean_abs_pre_reset_membrane": float(
            last["mean_abs_pre_reset_membrane"]
        ),
    }


def _load_frozen_exp01_binary_context(repo_root: Path) -> pd.DataFrame:
    path = base_results_dir(repo_root) / "runs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen Exp0.1 runs.csv: {path}")
    runs = pd.read_csv(path)
    context = runs[
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
        context[["architecture", "objective", "seed"]]
        .itertuples(index=False, name=None)
    )
    if len(context) != 30 or observed != expected:
        raise ValueError("Frozen Exp0.1 binary direct context must contain the expected 30 runs")
    context["context_label"] = "frozen_exp0.1_100epoch"
    return context


def _flatten_agg_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    history_frames: list[pd.DataFrame] = []
    for spec in run_specs():
        evaluation_file = evaluation_path(root, spec)
        history_file = history_path(root, spec)
        if not evaluation_file.exists():
            raise FileNotFoundError(f"Missing required Exp0.1.4 evaluation: {evaluation_file}")
        if not history_file.exists():
            raise FileNotFoundError(f"Missing required Exp0.1.4 history: {history_file}")
        payload = json.loads(evaluation_file.read_text(encoding="utf-8"))
        rows.append(_row_from_payload(spec, payload))
        history = pd.read_csv(history_file)
        if len(history) != EPOCHS:
            raise ValueError(f"Expected {EPOCHS} history rows for {spec.key}, found {len(history)}")
        history.insert(0, "seed", spec.seed)
        history.insert(0, "objective", spec.objective)
        history.insert(0, "architecture", spec.architecture)
        history.insert(0, "condition", spec.condition)
        history_frames.append(history)

    root.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs_file = root / "runs.csv"
    runs.to_csv(runs_file, index=False)

    history_long = pd.concat(history_frames, ignore_index=True, sort=False)
    history_file = root / "history_long.csv"
    history_long.to_csv(history_file, index=False)

    metric_columns = [
        "train_ba",
        "val_ba",
        "test_ba",
        "test_accuracy",
        "test_macro_f1",
        "test_last_hidden_events_per_neuron_second",
        "test_mean_dead_neuron_fraction",
        "test_last_hidden_dead_neuron_fraction",
        "test_mean_active_step_fraction",
        "test_last_hidden_active_step_fraction",
    ]
    summary = (
        runs.groupby(["condition", "architecture", "objective"], dropna=False)[metric_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary = _flatten_agg_columns(summary)
    summary_file = root / "summary.csv"
    summary.to_csv(summary_file, index=False)

    indexed = runs.set_index(["condition", "architecture", "objective", "seed"])
    paired_rows: list[dict[str, object]] = []
    for spec in run_specs():
        if spec.condition == "no_reg":
            continue
        candidate = indexed.loc[(spec.condition, spec.architecture, spec.objective, spec.seed)]
        baseline = indexed.loc[("no_reg", spec.architecture, spec.objective, spec.seed)]
        paired_rows.append(
            {
                "condition": spec.condition,
                "architecture": spec.architecture,
                "objective": spec.objective,
                "seed": spec.seed,
                "no_reg_test_ba": float(baseline["test_ba"]),
                "condition_test_ba": float(candidate["test_ba"]),
                "delta_test_ba_vs_no_reg": float(candidate["test_ba"] - baseline["test_ba"]),
                "no_reg_last_hidden_events_per_neuron_second": float(
                    baseline["test_last_hidden_events_per_neuron_second"]
                ),
                "condition_last_hidden_events_per_neuron_second": float(
                    candidate["test_last_hidden_events_per_neuron_second"]
                ),
                "delta_last_hidden_events_per_neuron_second_vs_no_reg": float(
                    candidate["test_last_hidden_events_per_neuron_second"]
                    - baseline["test_last_hidden_events_per_neuron_second"]
                ),
                "no_reg_last_hidden_dead_neuron_fraction": float(
                    baseline["test_last_hidden_dead_neuron_fraction"]
                ),
                "condition_last_hidden_dead_neuron_fraction": float(
                    candidate["test_last_hidden_dead_neuron_fraction"]
                ),
                "delta_last_hidden_dead_neuron_fraction_vs_no_reg": float(
                    candidate["test_last_hidden_dead_neuron_fraction"]
                    - baseline["test_last_hidden_dead_neuron_fraction"]
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired_file = root / "paired_condition_effects.csv"
    paired.to_csv(paired_file, index=False)

    paired_summary = (
        paired.groupby(["condition", "architecture", "objective"], dropna=False)[
            [
                "delta_test_ba_vs_no_reg",
                "delta_last_hidden_events_per_neuron_second_vs_no_reg",
                "delta_last_hidden_dead_neuron_fraction_vs_no_reg",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    paired_summary = _flatten_agg_columns(paired_summary)
    paired_summary_file = root / "paired_condition_summary.csv"
    paired_summary.to_csv(paired_summary_file, index=False)

    gradient_columns = [
        "first_batch_task_grad_norm",
        "first_batch_p2_grad_norm_raw",
        "first_batch_a1_grad_norm_raw",
        "first_batch_weighted_p2_grad_norm",
        "first_batch_weighted_a1_grad_norm",
        "first_batch_reg_grad_norm",
        "first_batch_p2_to_task_grad_ratio",
        "first_batch_a1_to_task_grad_ratio",
        "first_batch_reg_to_task_grad_ratio",
        "train_reg_to_task_ratio",
        "train_p2",
        "train_a1",
        "val_native_ba",
    ]
    gradient_rows = history_long[
        history_long["epoch"].isin(GRAD_DIAGNOSTIC_EPOCHS)
    ].copy()
    gradient_summary = (
        gradient_rows.groupby(
            ["condition", "architecture", "objective", "epoch"], dropna=False
        )[gradient_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    gradient_summary = _flatten_agg_columns(gradient_summary)
    gradient_file = root / "gradient_diagnostics_summary.csv"
    gradient_summary.to_csv(gradient_file, index=False)

    context = _load_frozen_exp01_binary_context(repo_root)
    context_file = root / "frozen_exp01_binary_context.csv"
    context.to_csv(context_file, index=False)
    context_summary = (
        context.groupby(["architecture", "objective"], dropna=False)["test_ba"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    context_summary["system"] = "frozen_exp0.1_100epoch"
    context_summary_file = root / "frozen_exp01_summary.csv"
    context_summary.to_csv(context_summary_file, index=False)

    comparison_rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        comparison_rows.append(
            {
                "system": f"exp0.1.4_30epoch:{row.condition}",
                "condition": row.condition,
                "architecture": row.architecture,
                "objective": row.objective,
                "mean_test_ba": row.test_ba_mean,
                "sd_test_ba": row.test_ba_std,
                "n": row.test_ba_count,
            }
        )
    for row in context_summary.itertuples(index=False):
        comparison_rows.append(
            {
                "system": row.system,
                "condition": "historical_context",
                "architecture": row.architecture,
                "objective": row.objective,
                "mean_test_ba": row.mean,
                "sd_test_ba": row.std,
                "n": row.count,
            }
        )
    comparison_file = root / "comparison_summary.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_file, index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Which proposed fix prevents the Exp0.1.2/0.1.3 regularization collapse, and does the "
            "combined all-fixes condition preserve useful SNN classification/firing dynamics?"
        ),
        "new_training_run_count": len(run_specs()),
        "training_budget_epochs": EPOCHS,
        "run_matrix": {
            "conditions": list(CONDITIONS),
            "architectures": list(DIRECT_ARCHITECTURES),
            "objectives": list(OBJECTIVES),
            "seeds": list(SEEDS),
            "variant": VARIANT,
            "hidden_cap": HIDDEN_CAP,
            "total_runs": len(run_specs()),
        },
        "conditions": {
            condition: _condition_manifest(condition) for condition in CONDITIONS
        },
        "preserved_contracts": {
            "base_experiment_id": BASE_EXPERIMENT_ID,
            "base_protocol_version": BASE_PROTOCOL_VERSION,
            "hidden_width": HIDDEN_WIDTH,
            "output_cap": OUTPUT_CAP,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "checkpoint_selection": "validation Output WholeCount BA; tie-break normalized WholeCount CE",
            "paired_seed_policy": "condition is excluded from model-init and loader seeds",
            "snn_forward_dynamics": "unchanged Exp0.1 binary direct-SNN dynamics; normalization applies only on P2 penalty path",
        },
        "direct_architectures": {
            key: [list(shifts) for shifts in value]
            for key, value in DIRECT_ARCHITECTURES.items()
        },
        "gradient_diagnostics": {
            "epochs": list(GRAD_DIAGNOSTIC_EPOCHS),
            "batch": "first training batch only",
            "purpose": "measure weighted regularizer gradient competition against task gradient without excessive CPU overhead",
        },
        "historical_context": "frozen Exp0.1 binary 100-epoch direct-SNN results are copied for context only, not paired as equal-budget controls",
        "files": {
            "runs": runs_file.name,
            "summary": summary_file.name,
            "history_long": history_file.name,
            "paired_condition_effects": paired_file.name,
            "paired_condition_summary": paired_summary_file.name,
            "gradient_diagnostics_summary": gradient_file.name,
            "frozen_exp01_binary_context": context_file.name,
            "frozen_exp01_summary": context_summary_file.name,
            "comparison_summary": comparison_file.name,
        },
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    return {
        "runs": runs_file,
        "summary": summary_file,
        "history_long": history_file,
        "paired_condition_effects": paired_file,
        "paired_condition_summary": paired_summary_file,
        "gradient_diagnostics_summary": gradient_file,
        "frozen_exp01_binary_context": context_file,
        "frozen_exp01_summary": context_summary_file,
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
    run_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("finalize")
    return parser


def _config_from_args(args: argparse.Namespace, repo_root: Path) -> Config:
    return Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=getattr(args, "device", "cpu"),
        epochs=EPOCHS,
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
        f"condition={spec.condition} epochs={config.epochs}"
    )


if __name__ == "__main__":
    main()
