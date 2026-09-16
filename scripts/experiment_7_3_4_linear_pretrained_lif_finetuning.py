from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from scripts import experiment_7_3_training_strategy_decomposition as exp73


EXPERIMENT_ID = "experiment_7_3_4_linear_pretrained_lif_finetuning"
PROTOCOL_VERSION = "linear_pretrained_lif_finetuning_v1"
ARCHITECTURE = exp73.ARCHITECTURE
SEEDS = exp73.SEEDS
BACKBONE_OBJECTIVE = "wcce"
OBJECTIVE = "wcce"
INIT_SOURCES = ("a2", "b6", "random")
LIF_BETA = exp73.LIF_BETA
THRESHOLD = exp73.THRESHOLD
OUTPUT_CAP = exp73.OUTPUT_CAP
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
REPRO_TOL = 1e-12

DIRECT_METHODS = {
    "a2": "A2_linearW_direct_lif",
    "b6": "B6_linearW_direct_lif",
}
FINETUNE_METHODS = {
    "a2": "A2_linearW_lif_finetune",
    "b6": "B6_linearW_lif_finetune",
}
RANDOM_METHOD = "randomW_lif_train"


@dataclass(frozen=True)
class RunSpec:
    seed: int
    init_source: str

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__{self.init_source}_init_lif_wcce__seed{self.seed}"

    @property
    def backbone(self) -> exp73.BackboneSpec:
        return exp73.BackboneSpec(self.seed, BACKBONE_OBJECTIVE)


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp73.exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(seed, init_source) for seed in SEEDS for init_source in INIT_SOURCES]


def validate_spec(spec: RunSpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    if spec.init_source not in INIT_SOURCES:
        raise ValueError(spec.init_source)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _exp73_config(config: Config) -> exp73.Config:
    return exp73.Config(
        repo_root=config.repo_root,
        results_dir=exp73.results_dir(config.repo_root),
        device=config.device,
        batch_size=config.batch_size,
        threads=config.threads,
        max_epochs=config.max_epochs,
    )


def _source_paths(config: Config, spec: RunSpec) -> tuple[Path, Path, str]:
    root = exp73.results_dir(config.repo_root)
    if spec.init_source == "a2":
        source = exp73.E2ESpec(spec.seed, "linear", "wcce")
        return (
            exp73._path(root, "e2e_checkpoints", source.key, ".pt"),
            exp73._path(root, "e2e_evaluations", source.key, ".json"),
            source.method,
        )
    if spec.init_source == "b6":
        source = exp73.Stage2Spec(spec.seed, "wcce", "linear", "wcce")
        return (
            exp73._path(root, "stage2_checkpoints", source.key, ".pt"),
            exp73._path(root, "stage2_evaluations", source.key, ".json"),
            source.method,
        )
    raise ValueError(spec.init_source)


def _b8_evaluation_path(config: Config, seed: int) -> Path:
    source = exp73.Stage2Spec(seed, "wcce", "lif", "wcce")
    return exp73._path(
        exp73.results_dir(config.repo_root), "stage2_evaluations", source.key, ".json"
    )


def _load_pretrained_weight(
    config: Config, spec: RunSpec
) -> tuple[torch.Tensor, dict[str, Any]]:
    checkpoint_path, evaluation_path, source_method = _source_paths(config, spec)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not evaluation_path.exists():
        raise FileNotFoundError(evaluation_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state = checkpoint.get("model_state_dict", {})
    if "output_linear.weight" not in state:
        raise KeyError(f"output_linear.weight missing from {checkpoint_path}")
    weight = state["output_linear.weight"].detach().cpu().to(torch.float32).clone()
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    return weight, {
        "source_method": source_method,
        "source_checkpoint": str(checkpoint_path.relative_to(config.repo_root)),
        "source_evaluation": str(evaluation_path.relative_to(config.repo_root)),
        "source_best_epoch": int(checkpoint["best_epoch"]),
        "exp73_analog_test_ba": float(
            evaluation["cross_evaluation"]["test"]["analog"]["balanced_accuracy"]
        ),
        "exp73_lif_test_ba": float(
            evaluation["cross_evaluation"]["test"]["lif_beta05"]["balanced_accuracy"]
        ),
    }


def _load_cache(config: Config, spec: RunSpec) -> dict[str, Any]:
    cache = exp73._load_stage2_cache(spec.backbone, _exp73_config(config))
    meta = cache["metadata"]
    if meta.get("source_method") != "A2_e2e_linear_wcce":
        raise RuntimeError(
            f"{spec.key}: expected A2 WCCE frozen backbone, got {meta.get('source_method')}"
        )
    if not bool(meta.get("frozen_l1_l2")):
        raise RuntimeError(f"{spec.key}: Exp7.3 cache is not marked frozen")
    return cache


def _make_model(
    config: Config,
    spec: RunSpec,
    n_classes: int,
) -> tuple[exp73.Stage2Head, torch.Tensor, dict[str, Any]]:
    exp73.exp3.seed_all(exp73._stage2_pair_seed(spec.seed, "model_init"))
    model = exp73.Stage2Head("lif", n_classes).to(config.device)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable != ["output_linear.weight"]:
        raise RuntimeError(f"Unexpected trainable parameters: {trainable}")

    source: dict[str, Any]
    if spec.init_source in DIRECT_METHODS:
        weight, source = _load_pretrained_weight(config, spec)
        if tuple(weight.shape) != tuple(model.output_linear.weight.shape):
            raise RuntimeError(
                f"{spec.key}: source W shape {tuple(weight.shape)} != "
                f"head shape {tuple(model.output_linear.weight.shape)}"
            )
        with torch.no_grad():
            model.output_linear.weight.copy_(weight.to(config.device))
    else:
        b8_path = _b8_evaluation_path(config, spec.seed)
        if not b8_path.exists():
            raise FileNotFoundError(b8_path)
        b8 = json.loads(b8_path.read_text(encoding="utf-8"))
        source = {
            "source_method": "B8_wcbackbone_lif_wcce",
            "source_checkpoint": None,
            "source_evaluation": str(b8_path.relative_to(config.repo_root)),
            "source_best_epoch": int(b8["best_epoch"]),
            "exp73_b8_native_test_ba": float(
                b8["native_metrics"]["test"]["balanced_accuracy"]
            ),
            "exp73_b8_final_lif_test_ba": float(
                b8["cross_evaluation"]["test"]["lif_beta05"]["balanced_accuracy"]
            ),
        }

    initial_weight = model.output_linear.weight.detach().cpu().clone()
    return model, initial_weight, source


def _set_weight(model: exp73.Stage2Head, weight: torch.Tensor) -> None:
    with torch.no_grad():
        model.output_linear.weight.copy_(weight.to(model.output_linear.weight.device))


def _copy_weight(model: exp73.Stage2Head) -> torch.Tensor:
    return model.output_linear.weight.detach().cpu().clone()


def _weight_drift(weight: torch.Tensor, initial: torch.Tensor) -> dict[str, float]:
    w = weight.detach().cpu().double()
    w0 = initial.detach().cpu().double()
    base_norm = float(torch.linalg.vector_norm(w0))
    cur_norm = float(torch.linalg.vector_norm(w))
    delta_norm = float(torch.linalg.vector_norm(w - w0))
    rel_fro = delta_norm / max(base_norm, 1e-12)
    norm_ratio = cur_norm / max(base_norm, 1e-12)

    numerator = (w * w0).sum(dim=1)
    denominator = torch.linalg.vector_norm(w, dim=1) * torch.linalg.vector_norm(w0, dim=1)
    cosine = torch.where(
        denominator > 1e-12,
        numerator / denominator.clamp_min(1e-12),
        torch.ones_like(denominator),
    )
    return {
        "relative_frobenius_drift": float(rel_fro),
        "weight_norm_ratio": float(norm_ratio),
        "mean_class_cosine": float(cosine.mean()),
        "min_class_cosine": float(cosine.min()),
    }


def _evaluate_head(
    model: exp73.Stage2Head,
    loader: Iterable,
    device: torch.device,
) -> dict[str, Any]:
    ys: list[np.ndarray] = []
    lif_scores_all: list[np.ndarray] = []
    linear_scores_all: list[np.ndarray] = []
    lif_loss_sum = 0.0
    linear_loss_sum = 0.0
    n_total = 0
    total_output_spikes = 0.0
    total_valid_neuron_steps = 0
    sample_total_spikes: list[np.ndarray] = []
    correct_counts: list[np.ndarray] = []
    max_incorrect_counts: list[np.ndarray] = []
    margins: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for l2, y, lengths in loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            tr = model.forward_trajectory(l2)
            lif_loss, lif_scores = exp73._objective_loss_scores(
                tr["output_spikes"], lengths, y, OBJECTIVE
            )
            linear_loss, linear_scores = exp73._objective_loss_scores(
                tr["evidence"], lengths, y, OBJECTIVE
            )

            spikes = tr["output_spikes"]
            valid = exp73._valid_mask(lengths, spikes.shape[1]).to(spikes.dtype)
            counts = (spikes * valid.unsqueeze(-1)).sum(dim=1)
            correct = counts.gather(1, y[:, None]).squeeze(1)
            wrong = counts.clone()
            wrong.scatter_(1, y[:, None], float("-inf"))
            max_wrong = wrong.max(dim=1).values

            ys.append(y.cpu().numpy())
            lif_scores_all.append(lif_scores.cpu().numpy())
            linear_scores_all.append(linear_scores.cpu().numpy())
            lif_loss_sum += float(lif_loss) * len(y)
            linear_loss_sum += float(linear_loss) * len(y)
            n_total += len(y)
            total_output_spikes += float(counts.sum())
            total_valid_neuron_steps += int(lengths.sum()) * counts.shape[1]
            sample_total_spikes.append(counts.sum(dim=1).cpu().numpy())
            correct_counts.append(correct.cpu().numpy())
            max_incorrect_counts.append(max_wrong.cpu().numpy())
            margins.append((correct - max_wrong).cpu().numpy())

    y_np = np.concatenate(ys)
    lif_scores_np = np.concatenate(lif_scores_all)
    linear_scores_np = np.concatenate(linear_scores_all)
    lif_metrics = exp73._metrics(y_np, lif_scores_np)
    linear_metrics = exp73._metrics(y_np, linear_scores_np)
    lif_metrics["objective_loss"] = lif_loss_sum / max(n_total, 1)
    linear_metrics["objective_loss"] = linear_loss_sum / max(n_total, 1)

    totals = np.concatenate(sample_total_spikes)
    correct_np = np.concatenate(correct_counts)
    max_wrong_np = np.concatenate(max_incorrect_counts)
    margin_np = np.concatenate(margins)
    diagnostics = {
        "mean_total_output_spikes_per_sample": float(totals.mean()),
        "mean_output_spikes_per_neuron_per_sample": float(
            totals.mean() / lif_scores_np.shape[1]
        ),
        "silent_sample_fraction": float((totals == 0).mean()),
        "output_spike_fraction_per_valid_neuron_step": float(
            total_output_spikes / max(total_valid_neuron_steps, 1)
        ),
        "mean_correct_class_spike_count": float(correct_np.mean()),
        "mean_max_incorrect_class_spike_count": float(max_wrong_np.mean()),
        "mean_spike_count_margin": float(margin_np.mean()),
    }
    return {"lif": lif_metrics, "linear": linear_metrics, "spikes": diagnostics}


def _history_row(
    epoch: int,
    optimization_train_loss: float,
    train_eval: dict[str, Any],
    val_eval: dict[str, Any],
    drift: dict[str, float],
) -> dict[str, float]:
    return {
        "epoch": float(epoch),
        "optimization_train_loss": float(optimization_train_loss),
        "train_lif_ba": float(train_eval["lif"]["balanced_accuracy"]),
        "val_lif_ba": float(val_eval["lif"]["balanced_accuracy"]),
        "train_lif_loss": float(train_eval["lif"]["objective_loss"]),
        "val_lif_loss": float(val_eval["lif"]["objective_loss"]),
        "train_linear_ba": float(train_eval["linear"]["balanced_accuracy"]),
        "val_linear_ba": float(val_eval["linear"]["balanced_accuracy"]),
        "train_linear_loss": float(train_eval["linear"]["objective_loss"]),
        "val_linear_loss": float(val_eval["linear"]["objective_loss"]),
        "weight_relative_frobenius_drift": drift["relative_frobenius_drift"],
        "weight_norm_ratio": drift["weight_norm_ratio"],
        "weight_mean_class_cosine": drift["mean_class_cosine"],
        "weight_min_class_cosine": drift["min_class_cosine"],
        "val_mean_total_output_spikes_per_sample": float(
            val_eval["spikes"]["mean_total_output_spikes_per_sample"]
        ),
        "val_silent_sample_fraction": float(val_eval["spikes"]["silent_sample_fraction"]),
        "val_output_spike_fraction": float(
            val_eval["spikes"]["output_spike_fraction_per_valid_neuron_step"]
        ),
        "val_mean_spike_count_margin": float(val_eval["spikes"]["mean_spike_count_margin"]),
    }


def _evaluate_states(
    model: exp73.Stage2Head,
    states: dict[str, torch.Tensor],
    loaders: dict[str, Iterable],
    device: torch.device,
    initial_weight: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    metrics: dict[str, Any] = {}
    drifts: dict[str, dict[str, float]] = {}
    for state_name, weight in states.items():
        _set_weight(model, weight)
        metrics[state_name] = {
            split: _evaluate_head(model, loader, device)
            for split, loader in loaders.items()
        }
        drifts[state_name] = _weight_drift(weight, initial_weight)
    return metrics, drifts


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    evaluation_path = config.results_dir / "evaluations" / f"{spec.key}.json"
    checkpoint_path = config.results_dir / "checkpoints" / f"{spec.key}.pt"
    history_path = config.results_dir / "histories" / f"{spec.key}.csv"
    if (
        evaluation_path.exists()
        and checkpoint_path.exists()
        and history_path.exists()
        and not force
    ):
        return json.loads(evaluation_path.read_text(encoding="utf-8"))

    cache = _load_cache(config, spec)
    n_classes = int(np.max(cache["train"][1])) + 1
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model, initial_weight, source = _make_model(config, spec, n_classes)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp73.exp72.LR,
        weight_decay=exp73.exp72.WEIGHT_DECAY,
    )
    train_loader = exp73._cached_loaders(
        cache, spec.seed, config.batch_size, True
    )["train"]
    eval_loaders = exp73._cached_loaders(cache, spec.seed, config.batch_size, False)

    train0 = _evaluate_head(model, eval_loaders["train"], device)
    val0 = _evaluate_head(model, eval_loaders["val"], device)
    history = [
        _history_row(
            0,
            float("nan"),
            train0,
            val0,
            _weight_drift(initial_weight, initial_weight),
        )
    ]

    best_ba_state: torch.Tensor | None = None
    best_ba_epoch = -1
    best_ba = -1.0
    best_ba_loss = float("inf")
    best_loss_state: torch.Tensor | None = None
    best_loss_epoch = -1
    best_loss = float("inf")
    best_loss_ba = -1.0
    stopped_epoch = config.max_epochs

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_total = 0
        for l2, y, lengths in train_loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(l2)
            loss, _ = exp73._objective_loss_scores(
                trajectory["output_spikes"], lengths, y, OBJECTIVE
            )
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_eval = _evaluate_head(model, eval_loaders["train"], device)
        val_eval = _evaluate_head(model, eval_loaders["val"], device)
        history.append(
            _history_row(
                epoch,
                train_loss_sum / max(n_total, 1),
                train_eval,
                val_eval,
                _weight_drift(_copy_weight(model), initial_weight),
            )
        )

        val_lif = val_eval["lif"]
        if exp73._checkpoint_improved(val_lif, best_ba, best_ba_loss):
            best_ba = float(val_lif["balanced_accuracy"])
            best_ba_loss = float(val_lif["objective_loss"])
            best_ba_epoch = epoch
            best_ba_state = _copy_weight(model)

        val_loss = float(val_lif["objective_loss"])
        val_ba = float(val_lif["balanced_accuracy"])
        if val_loss < best_loss - 1e-12 or (
            abs(val_loss - best_loss) <= 1e-12 and val_ba > best_loss_ba
        ):
            best_loss = val_loss
            best_loss_ba = val_ba
            best_loss_epoch = epoch
            best_loss_state = _copy_weight(model)

        if (
            epoch >= MIN_EPOCHS
            and best_ba_epoch > 0
            and epoch - best_ba_epoch >= PATIENCE
        ):
            stopped_epoch = epoch
            break

    if best_ba_state is None or best_loss_state is None:
        raise RuntimeError(f"No trained checkpoint selected for {spec.key}")

    final_state = _copy_weight(model)
    states = {
        "epoch0": initial_weight,
        "best_val_ba": best_ba_state,
        "best_val_loss": best_loss_state,
        "final": final_state,
    }
    state_metrics, state_drifts = _evaluate_states(
        model, states, eval_loaders, device, initial_weight
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "source": source,
            "best_val_ba_epoch": best_ba_epoch,
            "best_val_loss_epoch": best_loss_epoch,
            "stopped_epoch": stopped_epoch,
            "weights": states,
            "trainable_parameters": int(model.output_linear.weight.numel()),
            "selection_rule": (
                "primary best checkpoint uses validation LIF balanced accuracy with "
                "validation LIF WCCE loss as tiebreak; epoch0 is not eligible"
            ),
        },
        checkpoint_path,
    )

    source_checks: dict[str, float | bool] = {}
    if spec.init_source in DIRECT_METHODS:
        epoch0_lif = float(
            state_metrics["epoch0"]["test"]["lif"]["balanced_accuracy"]
        )
        epoch0_linear = float(
            state_metrics["epoch0"]["test"]["linear"]["balanced_accuracy"]
        )
        source_checks = {
            "epoch0_lif_minus_exp73_lif_pp": 100.0
            * (epoch0_lif - float(source["exp73_lif_test_ba"])),
            "epoch0_linear_minus_exp73_analog_pp": 100.0
            * (epoch0_linear - float(source["exp73_analog_test_ba"])),
        }
    else:
        selected = float(
            state_metrics["best_val_ba"]["test"]["lif"]["balanced_accuracy"]
        )
        b8 = float(source["exp73_b8_native_test_ba"])
        source_checks = {
            "scratch_best_lif_minus_exp73_b8_pp": 100.0 * (selected - b8),
            "scratch_reproduces_exp73_b8": abs(selected - b8) <= REPRO_TOL,
        }

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "architecture": ARCHITECTURE,
            "backbone_source": "A2_e2e_linear_wcce",
            "frozen_l1_l2": True,
            "frozen_backbone_realization": "reuse Exp7.3 WCCE L2 cache",
            "objective": OBJECTIVE,
            "bias": False,
            "lif_beta": LIF_BETA,
            "threshold": THRESHOLD,
            "output_cap": OUTPUT_CAP,
            "only_trainable_parameter": "output_linear.weight",
            "trainable_parameters": int(model.output_linear.weight.numel()),
            "paired_model_init_seed": exp73._stage2_pair_seed(spec.seed, "model_init"),
            "paired_loader_order": True,
            "primary_checkpoint": "best_val_ba",
            "epoch0_not_eligible_for_finetune_selection": True,
            "test_evaluated_after_training_and_checkpoint_selection": True,
        },
        "source": source,
        "best_val_ba_epoch": best_ba_epoch,
        "best_val_loss_epoch": best_loss_epoch,
        "stopped_epoch": stopped_epoch,
        "state_metrics": state_metrics,
        "state_weight_drift": state_drifts,
        "source_checks": source_checks,
    }
    _save_json(evaluation_path, payload)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return payload


def _primary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    init_source = str(spec["init_source"])
    seed = int(spec["seed"])
    rows: list[dict[str, Any]] = []

    def row(method: str, state_name: str, role: str) -> dict[str, Any]:
        metrics = payload["state_metrics"][state_name]
        drift = payload["state_weight_drift"][state_name]
        return {
            "method": method,
            "init_source": init_source,
            "checkpoint_role": role,
            "seed": seed,
            "checkpoint_epoch": 0
            if state_name == "epoch0"
            else int(
                payload["best_val_ba_epoch"]
                if state_name == "best_val_ba"
                else payload["best_val_loss_epoch"]
            ),
            "val_lif_ba": float(metrics["val"]["lif"]["balanced_accuracy"]),
            "test_lif_ba": float(metrics["test"]["lif"]["balanced_accuracy"]),
            "val_linear_ba": float(metrics["val"]["linear"]["balanced_accuracy"]),
            "test_linear_ba": float(metrics["test"]["linear"]["balanced_accuracy"]),
            "test_mean_total_output_spikes_per_sample": float(
                metrics["test"]["spikes"]["mean_total_output_spikes_per_sample"]
            ),
            "test_silent_sample_fraction": float(
                metrics["test"]["spikes"]["silent_sample_fraction"]
            ),
            "weight_relative_frobenius_drift": float(
                drift["relative_frobenius_drift"]
            ),
            "weight_norm_ratio": float(drift["weight_norm_ratio"]),
            "weight_mean_class_cosine": float(drift["mean_class_cosine"]),
        }

    if init_source in DIRECT_METHODS:
        rows.append(row(DIRECT_METHODS[init_source], "epoch0", "direct_transfer"))
        rows.append(
            row(FINETUNE_METHODS[init_source], "best_val_ba", "best_val_ba")
        )
    else:
        rows.append(row(RANDOM_METHOD, "best_val_ba", "best_val_ba"))
    return rows


def _paired_contrast(
    runs: pd.DataFrame, name: str, left: str, right: str
) -> list[dict[str, Any]]:
    left_df = runs[runs.method == left][["seed", "test_lif_ba"]]
    right_df = runs[runs.method == right][["seed", "test_lif_ba"]]
    merged = left_df.merge(right_df, on="seed", suffixes=("_left", "_right"))
    if set(merged["seed"].astype(int)) != set(SEEDS):
        raise RuntimeError(f"Incomplete paired contrast {name}")
    return [
        {
            "contrast": name,
            "seed": int(row.seed),
            "left_method": left,
            "right_method": right,
            "delta_pp": 100.0
            * (float(row.test_lif_ba_left) - float(row.test_lif_ba_right)),
        }
        for row in merged.itertuples(index=False)
    ]


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    missing: list[Path] = []
    for spec in run_specs():
        path = config.results_dir / "evaluations" / f"{spec.key}.json"
        if path.exists():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} Exp7.3.4 evaluations:\n{preview}")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(
        [row for payload in payloads for row in _primary_rows(payload)]
    ).sort_values(["method", "seed"])
    if len(runs) != 15:
        raise RuntimeError(f"Expected 15 primary method/seed rows, got {len(runs)}")
    runs.to_csv(config.results_dir / "method_runs.csv", index=False)

    numeric = [
        "checkpoint_epoch",
        "val_lif_ba",
        "test_lif_ba",
        "val_linear_ba",
        "test_linear_ba",
        "test_mean_total_output_spikes_per_sample",
        "test_silent_sample_fraction",
        "weight_relative_frobenius_drift",
        "weight_norm_ratio",
        "weight_mean_class_cosine",
    ]
    summary = runs.groupby(
        ["method", "init_source", "checkpoint_role"], sort=False
    )[numeric].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.reset_index().to_csv(config.results_dir / "method_summary.csv", index=False)

    checkpoint_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    for payload in payloads:
        spec = payload["spec"]
        seed = int(spec["seed"])
        init_source = str(spec["init_source"])
        for state_name, split_metrics in payload["state_metrics"].items():
            drift = payload["state_weight_drift"][state_name]
            checkpoint_rows.append(
                {
                    "seed": seed,
                    "init_source": init_source,
                    "state": state_name,
                    "val_lif_ba": float(split_metrics["val"]["lif"]["balanced_accuracy"]),
                    "test_lif_ba": float(split_metrics["test"]["lif"]["balanced_accuracy"]),
                    "val_linear_ba": float(split_metrics["val"]["linear"]["balanced_accuracy"]),
                    "test_linear_ba": float(split_metrics["test"]["linear"]["balanced_accuracy"]),
                    "weight_relative_frobenius_drift": float(drift["relative_frobenius_drift"]),
                    "weight_norm_ratio": float(drift["weight_norm_ratio"]),
                    "weight_mean_class_cosine": float(drift["mean_class_cosine"]),
                }
            )
        source_row = {
            "seed": seed,
            "init_source": init_source,
            "source_method": payload["source"]["source_method"],
        }
        source_row.update(payload["source_checks"])
        for key in (
            "exp73_analog_test_ba",
            "exp73_lif_test_ba",
            "exp73_b8_native_test_ba",
            "exp73_b8_final_lif_test_ba",
        ):
            if key in payload["source"]:
                source_row[key] = payload["source"][key]
        source_rows.append(source_row)

        history_path = config.results_dir / "histories" / (
            f"{ARCHITECTURE}__{init_source}_init_lif_wcce__seed{seed}.csv"
        )
        history = pd.read_csv(history_path)
        history.insert(0, "seed", seed)
        history.insert(1, "init_source", init_source)
        histories.append(history)

    pd.DataFrame(checkpoint_rows).sort_values(
        ["init_source", "seed", "state"]
    ).to_csv(config.results_dir / "checkpoint_diagnostics.csv", index=False)
    source_df = pd.DataFrame(source_rows).sort_values(["init_source", "seed"])
    source_df.to_csv(config.results_dir / "source_reproduction_checks.csv", index=False)

    history_runs = pd.concat(histories, ignore_index=True)
    history_runs.to_csv(config.results_dir / "history_runs.csv", index=False)
    history_numeric = [
        column
        for column in history_runs.columns
        if column not in {"seed", "init_source", "epoch"}
    ]
    history_summary = history_runs.groupby(
        ["init_source", "epoch"], sort=False
    )[history_numeric].agg(["mean", "std", "count"])
    history_summary.columns = [f"{name}_{stat}" for name, stat in history_summary.columns]
    history_summary.reset_index().to_csv(
        config.results_dir / "history_summary.csv", index=False
    )

    contrast_defs = (
        (
            "A2_finetune_minus_direct",
            FINETUNE_METHODS["a2"],
            DIRECT_METHODS["a2"],
        ),
        (
            "B6_finetune_minus_direct",
            FINETUNE_METHODS["b6"],
            DIRECT_METHODS["b6"],
        ),
        (
            "A2_finetune_minus_random",
            FINETUNE_METHODS["a2"],
            RANDOM_METHOD,
        ),
        (
            "B6_finetune_minus_random",
            FINETUNE_METHODS["b6"],
            RANDOM_METHOD,
        ),
        (
            "B6_direct_minus_A2_direct",
            DIRECT_METHODS["b6"],
            DIRECT_METHODS["a2"],
        ),
        (
            "B6_finetune_minus_A2_finetune",
            FINETUNE_METHODS["b6"],
            FINETUNE_METHODS["a2"],
        ),
    )
    contrast_rows = [
        row
        for name, left, right in contrast_defs
        for row in _paired_contrast(runs, name, left, right)
    ]
    contrast_runs = pd.DataFrame(contrast_rows)
    contrast_runs.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_summary = (
        contrast_runs.groupby(["contrast", "left_method", "right_method"])["delta_pp"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "delta_pp_count",
                "mean": "delta_pp_mean",
                "std": "delta_pp_std",
            }
        )
    )
    contrast_summary.to_csv(config.results_dir / "contrast_summary.csv", index=False)

    random_checks = source_df[source_df.init_source == "random"]
    max_b8_delta = float(
        random_checks["scratch_best_lif_minus_exp73_b8_pp"].abs().max()
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "seeds": list(SEEDS),
        "backbone_objective": BACKBONE_OBJECTIVE,
        "objective": OBJECTIVE,
        "init_sources": list(INIT_SOURCES),
        "logical_training_runs": len(run_specs()),
        "primary_method_seed_rows": len(runs),
        "lif_beta": LIF_BETA,
        "threshold": THRESHOLD,
        "output_cap": OUTPUT_CAP,
        "primary_metric": "test_lif_ba from the best-validation-LIF-BA checkpoint",
        "selection_rule": (
            "epoch 0 is a fixed direct-transfer reference and is excluded from fine-tune "
            "selection; epochs >=1 use validation LIF BA with validation LIF WCCE loss tiebreak"
        ),
        "frozen_backbone_contract": (
            "All runs reuse the exact Exp7.3 WCCE L2 cache sourced from A2; no L1/L2 "
            "parameter is instantiated as trainable."
        ),
        "random_control_contract": (
            "Random initialization reuses Exp7.3 Stage-2 model-init seed, loader order, "
            "optimizer, LIF dynamics, and WCCE objective and should reproduce B8."
        ),
        "max_abs_random_control_vs_exp73_b8_delta_pp": max_b8_delta,
        "notebook_contract": "analysis-only; reads finalized CSV/JSON artifacts",
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp7.3.4 Linear-pretrained LIF head fine-tuning"
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp73.exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    config = Config(
        repo_root=root,
        results_dir=results_dir(root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        max_epochs=args.max_epochs,
    )
    if args.command == "run":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        run_one(specs[args.array_task_id], config, force=args.force)
    elif args.command == "finalize":
        finalize(config)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
