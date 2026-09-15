from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_2_5_output_synaptic_alpha as exp725
from scripts import experiment_7_2_6_output_readout_loss_shaping as exp726


EXPERIMENT_ID = "experiment_7_2_6_1_sum_vs_mean_ce"
PROTOCOL_VERSION = "sum_vs_mean_ce_v1"
ARCHITECTURE = exp726.ARCHITECTURE
REGULARIZATION = exp72.TASK_ONLY
SEEDS = exp726.SEEDS
MAX_EPOCHS = exp726.MAX_EPOCHS
MIN_EPOCHS = exp726.MIN_EPOCHS
PATIENCE = exp726.PATIENCE

SUM_REUSE = "sum_ce_reuse_c0"
MEAN_NEW = "mean_ce"
CONDITIONS = (SUM_REUSE, MEAN_NEW)


@dataclass(frozen=True)
class RunSpec:
    seed: int

    @property
    def source(self) -> exp726.SourceSpec:
        return exp726.SourceSpec(REGULARIZATION, self.seed)

    @property
    def c0_spec(self) -> exp726.E2ESpec:
        return exp726.E2ESpec(REGULARIZATION, self.seed, exp726.E2E_ANALOG)

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__{REGULARIZATION}__seed{self.seed}__{MEAN_NEW}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp726.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def specs() -> list[RunSpec]:
    return [RunSpec(seed) for seed in SEEDS]


def validate_spec(spec: RunSpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, spec: RunSpec, suffix: str) -> Path:
    return root / kind / f"{spec.key}{suffix}"


def _base_config(config: Config) -> exp726.Config:
    return exp726.Config(
        config.repo_root,
        exp726.results_dir(config.repo_root),
        config.device,
        config.batch_size,
        config.threads,
        config.max_epochs,
    )


def _base_eval_path(config: Config, spec: RunSpec) -> Path:
    return exp726._e2e_path(exp726.results_dir(config.repo_root), "evaluations", spec.c0_spec, ".json")


def _base_probe_path(config: Config, spec: RunSpec) -> Path:
    return exp726._e2e_path(exp726.results_dir(config.repo_root), "probe_evaluations", spec.c0_spec, ".json")


def _base_history_path(config: Config, spec: RunSpec) -> Path:
    return exp726._e2e_path(exp726.results_dir(config.repo_root), "histories", spec.c0_spec, ".csv")


def _require_base_artifacts(config: Config, spec: RunSpec, require_probe: bool = False) -> None:
    required = [_base_eval_path(config, spec), exp726._e2e_path(exp726.results_dir(config.repo_root), "checkpoints", spec.c0_spec, ".pt")]
    if require_probe:
        required.append(_base_probe_path(config, spec))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Exp7.2.6.1 reuses the task-only Exp7.2.6 C0 sum-CE run as its paired baseline. "
            f"Missing artifacts for seed {spec.seed}: {missing}. Run/finalize Exp7.2.6 first."
        )


def _aggregate_evidence(evidence: torch.Tensor, lengths: torch.Tensor, mode: str) -> torch.Tensor:
    summed = exp726._valid_sum(evidence, lengths)
    if mode == "sum":
        return summed
    if mode == "mean":
        denom = lengths.clamp_min(1).to(evidence.dtype).unsqueeze(1)
        return summed / denom
    raise ValueError(mode)


def _aggregate_l2(l2: torch.Tensor, lengths: torch.Tensor, mode: str) -> torch.Tensor:
    summed = exp726._valid_sum(l2, lengths)
    if mode == "sum":
        return summed
    if mode == "mean":
        denom = lengths.clamp_min(1).to(l2.dtype).unsqueeze(1)
        return summed / denom
    raise ValueError(mode)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _dual_eval(
    model: exp726.PairedShapingSNN,
    loader: Iterable,
    device: torch.device,
) -> dict[str, Any]:
    labels: list[np.ndarray] = []
    lengths_out: list[np.ndarray] = []
    sum_logits_out: list[np.ndarray] = []
    mean_logits_out: list[np.ndarray] = []
    sum_grad_out: list[np.ndarray] = []
    mean_grad_out: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            tr = model.forward_trajectory(X)
            l2 = tr["hidden_spikes"][-1]
            sum_feature = _aggregate_l2(l2, lengths, "sum")
            mean_feature = _aggregate_l2(l2, lengths, "mean")
            sum_logits = model.output_linear(sum_feature)
            mean_logits = model.output_linear(mean_feature)

            onehot = F.one_hot(y, num_classes=sum_logits.shape[1]).to(sum_logits.dtype)
            sum_delta = F.softmax(sum_logits, dim=1) - onehot
            mean_delta = F.softmax(mean_logits, dim=1) - onehot
            sum_grad = torch.linalg.vector_norm(sum_delta, dim=1) * torch.linalg.vector_norm(sum_feature, dim=1)
            mean_grad = torch.linalg.vector_norm(mean_delta, dim=1) * torch.linalg.vector_norm(mean_feature, dim=1)

            labels.append(y.cpu().numpy())
            lengths_out.append(lengths.cpu().numpy())
            sum_logits_out.append(sum_logits.cpu().numpy())
            mean_logits_out.append(mean_logits.cpu().numpy())
            sum_grad_out.append(sum_grad.cpu().numpy())
            mean_grad_out.append(mean_grad.cpu().numpy())

    y = np.concatenate(labels)
    lengths_np = np.concatenate(lengths_out).astype(np.float64)
    sum_logits_np = np.concatenate(sum_logits_out)
    mean_logits_np = np.concatenate(mean_logits_out)
    sum_grad_np = np.concatenate(sum_grad_out)
    mean_grad_np = np.concatenate(mean_grad_out)

    sum_pred = sum_logits_np.argmax(1)
    mean_pred = mean_logits_np.argmax(1)
    sum_norm = np.linalg.norm(sum_logits_np, axis=1)
    mean_norm = np.linalg.norm(mean_logits_np, axis=1)
    relation_error = np.max(np.abs(sum_logits_np - mean_logits_np * lengths_np[:, None]))

    def metrics(logits: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        tensor_logits = torch.from_numpy(logits)
        tensor_y = torch.from_numpy(y)
        return {
            **exp72._metrics(y, pred),
            "objective_loss": float(F.cross_entropy(tensor_logits, tensor_y)),
        }

    return {
        "sum_metrics": metrics(sum_logits_np, sum_pred),
        "mean_metrics": metrics(mean_logits_np, mean_pred),
        "prediction_equal": bool(np.array_equal(sum_pred, mean_pred)),
        "max_sum_equals_T_times_mean_error": float(relation_error),
        "sum_diagnostics": {
            "mean_logit_l2": float(sum_norm.mean()),
            "std_logit_l2": float(sum_norm.std()),
            "corr_length_logit_l2": _corr(lengths_np, sum_norm),
            "mean_output_w_grad_fro": float(sum_grad_np.mean()),
            "std_output_w_grad_fro": float(sum_grad_np.std()),
            "corr_length_output_w_grad_fro": _corr(lengths_np, sum_grad_np),
        },
        "mean_diagnostics": {
            "mean_logit_l2": float(mean_norm.mean()),
            "std_logit_l2": float(mean_norm.std()),
            "corr_length_logit_l2": _corr(lengths_np, mean_norm),
            "mean_output_w_grad_fro": float(mean_grad_np.mean()),
            "std_output_w_grad_fro": float(mean_grad_np.std()),
            "corr_length_output_w_grad_fro": _corr(lengths_np, mean_grad_np),
        },
    }


def _evaluate_mean(
    model: exp726.PairedShapingSNN,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    return _dual_eval(model, loader, device)["mean_metrics"]


def _load_base_model(spec: RunSpec, data: exp3.Data, config: Config):
    _require_base_artifacts(config, spec)
    return exp726._load_condition_model(spec.c0_spec, data, _base_config(config))


def _load_mean_model(spec: RunSpec, data: exp3.Data, config: Config):
    checkpoint = _path(config.results_dir, "checkpoints", spec, ".pt")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location=config.device, weights_only=False)
    model = exp726.PairedShapingSNN(exp726.E2E_ANALOG, len(data.labels), data.fs).to(config.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def run_mean(spec: RunSpec, data: exp3.Data, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    eval_path = _path(config.results_dir, "evaluations", spec, ".json")
    checkpoint_path = _path(config.results_dir, "checkpoints", spec, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    _require_base_artifacts(config, spec)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)

    # Exact pair with Exp7.2.6 C0: same model class, model-init seed, loader seed,
    # optimizer, learning rate, weight decay, and checkpoint selection. The only
    # training change is sum logits -> mean logits before CE.
    exp3.seed_all(exp726._reference_seed(spec.source, "model_init"))
    model = exp726.PairedShapingSNN(exp726.E2E_ANALOG, len(data.labels), data.fs).to(device)
    if model.output_linear.bias is not None:
        raise RuntimeError("Exp7.2.6.1 requires the same bias-free output projection as Exp7.2.6 C0")

    optimizer = torch.optim.Adam(model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY)
    train_loader = exp726._e2e_loaders(data, spec.c0_spec, config.batch_size, True)["train"]
    eval_loaders = exp726._e2e_loaders(data, spec.c0_spec, config.batch_size, False)

    initial_val = _dual_eval(model, eval_loaders["val"], device)
    if not initial_val["prediction_equal"]:
        raise RuntimeError("sum/mean prediction equivalence failed at initialization")

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            tr = model.forward_trajectory(X)
            logits = _aggregate_evidence(tr["output_evidence"], lengths, "mean")
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            n += len(y)

        train_metrics = _evaluate_mean(model, eval_loaders["train"], device)
        val_dual = _dual_eval(model, eval_loaders["val"], device)
        val_metrics = val_dual["mean_metrics"]
        if not val_dual["prediction_equal"]:
            raise RuntimeError(f"sum/mean readout predictions diverged at epoch {epoch}")

        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": loss_sum / max(n, 1),
                "val_mean_loss": float(val_metrics["objective_loss"]),
                "val_sum_loss": float(val_dual["sum_metrics"]["objective_loss"]),
                "val_mean_logit_l2": float(val_dual["mean_diagnostics"]["mean_logit_l2"]),
                "val_sum_logit_l2": float(val_dual["sum_diagnostics"]["mean_logit_l2"]),
                "val_mean_grad_w_fro": float(val_dual["mean_diagnostics"]["mean_output_w_grad_fro"]),
                "val_sum_grad_w_fro": float(val_dual["sum_diagnostics"]["mean_output_w_grad_fro"]),
                "val_mean_corr_length_logit_l2": float(val_dual["mean_diagnostics"]["corr_length_logit_l2"]),
                "val_sum_corr_length_logit_l2": float(val_dual["sum_diagnostics"]["corr_length_logit_l2"]),
                "val_mean_corr_length_grad_w_fro": float(val_dual["mean_diagnostics"]["corr_length_output_w_grad_fro"]),
                "val_sum_corr_length_grad_w_fro": float(val_dual["sum_diagnostics"]["corr_length_output_w_grad_fro"]),
            }
        )

        improved = val_metrics["balanced_accuracy"] > best_ba + 1e-12 or (
            abs(val_metrics["balanced_accuracy"] - best_ba) <= 1e-12
            and val_metrics["objective_loss"] < best_loss
        )
        if improved:
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE:
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for seed {spec.seed}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "condition": MEAN_NEW,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "model_state_dict": best_state,
            "paired_exp726_c0_spec": asdict(spec.c0_spec),
            "training_difference_from_exp726_c0": "valid mean logits before CE instead of valid sum logits before CE",
        },
        checkpoint_path,
    )

    model.load_state_dict(best_state, strict=True)
    final_dual = {split: _dual_eval(model, loader, device) for split, loader in eval_loaders.items()}
    for split, result in final_dual.items():
        if not result["prediction_equal"]:
            raise RuntimeError(f"final sum/mean prediction equivalence failed on {split}")

    base_model, _ = _load_base_model(spec, data, config)
    base_dual = {split: _dual_eval(base_model, loader, device) for split, loader in eval_loaders.items()}
    for split, result in base_dual.items():
        if not result["prediction_equal"]:
            raise RuntimeError(f"Exp7.2.6 C0 sum/mean prediction equivalence failed on {split}")

    history_path = _path(config.results_dir, "histories", spec, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "condition": MEAN_NEW,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "metrics": {split: result["mean_metrics"] for split, result in final_dual.items()},
        "initial_val_scaling_diagnostics": initial_val,
        "final_mean_model_dual_readout": final_dual,
        "exp726_sum_model_dual_readout": base_dual,
        "checkpoint_selection": "highest validation BA under mean-CE; mean-CE validation loss tie-break",
    }
    _save_json(eval_path, payload)
    return payload


def run_probe(spec: RunSpec, data: exp3.Data, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    _require_base_artifacts(config, spec, require_probe=True)
    path = _path(config.results_dir, "probe_evaluations", spec, ".json")
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    device = torch.device(config.device)
    model, checkpoint = _load_mean_model(spec, data, config)
    loaders = exp726._e2e_loaders(data, spec.c0_spec, config.batch_size, False)
    splits = {
        split: exp726._extract_l2_generic(model, loader, device)
        for split, loader in loaders.items()
    }
    ref_spec = spec.source.exp725_spec
    standardized_wc = exp725._fit_e2e_probe(exp726.PROBE_STD_WC, splits, ref_spec, data.bin_steps)
    fixed250 = exp725._fit_e2e_probe(exp726.PROBE_F250, splits, ref_spec, data.bin_steps)
    matched_wc = exp726._fit_matched_count_probe(splits, spec.c0_spec, len(data.labels))

    eval_payload = json.loads(_path(config.results_dir, "evaluations", spec, ".json").read_text(encoding="utf-8"))
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "condition": MEAN_NEW,
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "native_metrics": eval_payload["metrics"],
        "probes": {
            exp726.PROBE_MATCHED_WC: matched_wc,
            exp726.PROBE_STD_WC: standardized_wc,
            exp726.PROBE_F250: fixed250,
        },
    }
    _save_json(path, payload)
    return payload


def _aggregate(frame: pd.DataFrame, groups: list[str], values: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["n"] = len(group)
        for value in values:
            row[f"{value}_mean"] = float(group[value].mean())
            row[f"{value}_std"] = float(group[value].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def finalize(repo_root: Path) -> dict[str, Any]:
    root = results_dir(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    config = Config(repo_root, root)
    missing: list[str] = []
    native_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []

    for spec in specs():
        mean_eval_path = _path(root, "evaluations", spec, ".json")
        mean_probe_path = _path(root, "probe_evaluations", spec, ".json")
        base_eval_path = _base_eval_path(config, spec)
        base_probe_path = _base_probe_path(config, spec)
        base_history_path = _base_history_path(config, spec)
        mean_history_path = _path(root, "histories", spec, ".csv")
        for path in (mean_eval_path, mean_probe_path, base_eval_path, base_probe_path, base_history_path, mean_history_path):
            if not path.exists():
                missing.append(str(path))
        if any(not path.exists() for path in (mean_eval_path, mean_probe_path, base_eval_path, base_probe_path)):
            continue

        mean_eval = json.loads(mean_eval_path.read_text(encoding="utf-8"))
        mean_probe = json.loads(mean_probe_path.read_text(encoding="utf-8"))
        base_eval = json.loads(base_eval_path.read_text(encoding="utf-8"))
        base_probe = json.loads(base_probe_path.read_text(encoding="utf-8"))

        native_by_condition = {
            SUM_REUSE: base_eval["metrics"],
            MEAN_NEW: mean_eval["metrics"],
        }
        for condition, split_metrics in native_by_condition.items():
            for split, metrics in split_metrics.items():
                native_rows.append(
                    {
                        "architecture": ARCHITECTURE,
                        "regularization": REGULARIZATION,
                        "seed": spec.seed,
                        "condition": condition,
                        "split": split,
                        "accuracy": float(metrics["accuracy"]),
                        "balanced_accuracy": float(metrics["balanced_accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "objective_loss": float(metrics["objective_loss"]),
                    }
                )

        probes_by_condition = {
            SUM_REUSE: base_probe["probes"],
            MEAN_NEW: mean_probe["probes"],
        }
        for condition, probes in probes_by_condition.items():
            for probe_name, probe in probes.items():
                for split, metrics in probe["metrics"].items():
                    probe_rows.append(
                        {
                            "architecture": ARCHITECTURE,
                            "regularization": REGULARIZATION,
                            "seed": spec.seed,
                            "condition": condition,
                            "probe": probe_name,
                            "split": split,
                            "accuracy": float(metrics["accuracy"]),
                            "balanced_accuracy": float(metrics["balanced_accuracy"]),
                            "macro_f1": float(metrics["macro_f1"]),
                        }
                    )

        native_test_sum = float(base_eval["metrics"]["test"]["balanced_accuracy"])
        native_test_mean = float(mean_eval["metrics"]["test"]["balanced_accuracy"])
        delta_rows.append(
            {
                "seed": spec.seed,
                "metric": "native_test_ba",
                "mean_minus_sum": native_test_mean - native_test_sum,
                "sum_value": native_test_sum,
                "mean_value": native_test_mean,
            }
        )
        for probe_name in (exp726.PROBE_MATCHED_WC, exp726.PROBE_STD_WC, exp726.PROBE_F250):
            sum_value = float(base_probe["probes"][probe_name]["metrics"]["test"]["balanced_accuracy"])
            mean_value = float(mean_probe["probes"][probe_name]["metrics"]["test"]["balanced_accuracy"])
            delta_rows.append(
                {
                    "seed": spec.seed,
                    "metric": f"{probe_name}_test_ba",
                    "mean_minus_sum": mean_value - sum_value,
                    "sum_value": sum_value,
                    "mean_value": mean_value,
                }
            )

        dual_sources = {
            "initial_shared_weights": {"val": mean_eval["initial_val_scaling_diagnostics"]},
            "trained_mean_model": mean_eval["final_mean_model_dual_readout"],
            "trained_sum_model": mean_eval["exp726_sum_model_dual_readout"],
        }
        for model_state, splits in dual_sources.items():
            for split, dual in splits.items():
                equivalence_rows.append(
                    {
                        "seed": spec.seed,
                        "model_state": model_state,
                        "split": split,
                        "prediction_equal": bool(dual["prediction_equal"]),
                        "max_relation_error": float(dual["max_sum_equals_T_times_mean_error"]),
                        "sum_ba": float(dual["sum_metrics"]["balanced_accuracy"]),
                        "mean_ba": float(dual["mean_metrics"]["balanced_accuracy"]),
                        "sum_ce": float(dual["sum_metrics"]["objective_loss"]),
                        "mean_ce": float(dual["mean_metrics"]["objective_loss"]),
                    }
                )
                for mode in ("sum", "mean"):
                    diag = dual[f"{mode}_diagnostics"]
                    diagnostic_rows.append(
                        {
                            "seed": spec.seed,
                            "model_state": model_state,
                            "split": split,
                            "readout_scale": mode,
                            **{key: float(value) for key, value in diag.items()},
                        }
                    )

        if base_history_path.exists():
            frame = pd.read_csv(base_history_path).copy()
            frame["condition"] = SUM_REUSE
            frame["seed"] = spec.seed
            if "val_loss" not in frame.columns and "val_objective_loss" in frame.columns:
                frame["val_loss"] = frame["val_objective_loss"]
            history_frames.append(frame)
        if mean_history_path.exists():
            frame = pd.read_csv(mean_history_path).copy()
            frame["condition"] = MEAN_NEW
            frame["seed"] = spec.seed
            frame["val_loss"] = frame["val_mean_loss"]
            history_frames.append(frame)

    if missing:
        raise RuntimeError(f"Missing {len(missing)} Exp7.2.6.1/base artifacts; first={missing[:8]}")

    native = pd.DataFrame(native_rows)
    native.to_csv(root / "native_performance_runs.csv", index=False)
    _aggregate(native, ["condition", "split"], ["accuracy", "balanced_accuracy", "macro_f1", "objective_loss"]).to_csv(
        root / "native_performance_summary.csv", index=False
    )

    probes = pd.DataFrame(probe_rows)
    probes.to_csv(root / "l2_probe_runs.csv", index=False)
    _aggregate(probes, ["condition", "probe", "split"], ["accuracy", "balanced_accuracy", "macro_f1"]).to_csv(
        root / "l2_probe_summary.csv", index=False
    )

    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(root / "paired_delta_runs.csv", index=False)
    _aggregate(deltas, ["metric"], ["mean_minus_sum", "sum_value", "mean_value"]).to_csv(
        root / "paired_delta_summary.csv", index=False
    )

    equivalence = pd.DataFrame(equivalence_rows)
    equivalence.to_csv(root / "readout_equivalence_runs.csv", index=False)
    _aggregate(
        equivalence.assign(prediction_equal=equivalence.prediction_equal.astype(float)),
        ["model_state", "split"],
        ["prediction_equal", "max_relation_error", "sum_ba", "mean_ba", "sum_ce", "mean_ce"],
    ).to_csv(root / "readout_equivalence_summary.csv", index=False)

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(root / "scaling_diagnostics_runs.csv", index=False)
    _aggregate(
        diagnostics,
        ["model_state", "split", "readout_scale"],
        [
            "mean_logit_l2",
            "std_logit_l2",
            "corr_length_logit_l2",
            "mean_output_w_grad_fro",
            "std_output_w_grad_fro",
            "corr_length_output_w_grad_fro",
        ],
    ).to_csv(root / "scaling_diagnostics_summary.csv", index=False)

    history = pd.concat(history_frames, ignore_index=True)
    history.to_csv(root / "training_history_runs.csv", index=False)
    history_values = [column for column in ("train_ba", "val_ba", "train_loss", "val_loss") if column in history.columns]
    _aggregate(history, ["condition", "epoch"], history_values).to_csv(root / "training_history_summary.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "scientific_question": "Does valid-length normalization before CE explain the Exp7.2.4 A2 vs Exp7.2.6 C0 gap?",
        "architecture": ARCHITECTURE,
        "regularization": REGULARIZATION,
        "seeds": list(SEEDS),
        "conditions": {
            SUM_REUSE: "Exact existing Exp7.2.6 C0: bias-free analog output, valid sum logits before CE",
            MEAN_NEW: "Exact paired rerun with the sole training change: valid mean logits before CE",
        },
        "new_training_runs": len(specs()),
        "reused_sum_runs": len(specs()),
        "pairing": "same model class, bias=False, model-init seed stream, loader seed stream, Adam/LR/weight decay, max/min epochs and patience",
        "checkpoint_selection": "highest validation BA under the condition's training objective; validation objective loss tie-break",
        "readout_sanity": "For a fixed bias-free model, sum logits = valid_length * mean logits, so argmax predictions must match exactly",
        "notebook_inputs": [
            "native_performance_summary.csv",
            "l2_probe_summary.csv",
            "paired_delta_summary.csv",
            "readout_equivalence_summary.csv",
            "scaling_diagnostics_summary.csv",
            "training_history_summary.csv",
        ],
    }
    _save_json(root / "manifest.json", manifest)
    return manifest


def _select(index: int | None, seed: int | None) -> RunSpec:
    if index is not None:
        all_specs = specs()
        if not 0 <= index < len(all_specs):
            raise ValueError(index)
        return all_specs[index]
    if seed is None:
        raise ValueError("Specify --array-task-id or --seed")
    return RunSpec(seed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)

    for command in ("run-mean", "run-probe"):
        child = sub.add_parser(command)
        child.add_argument("--array-task-id", type=int)
        child.add_argument("--seed", type=int, choices=SEEDS)
        child.add_argument("--force", action="store_true")

    sub.add_parser("list")
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = find_repo_root(args.repo_root)
    config = Config(repo_root, results_dir(repo_root), args.device, args.batch_size, args.threads, args.max_epochs)

    if args.command == "list":
        for index, spec in enumerate(specs()):
            print(index, spec.key)
        return
    if args.command == "finalize":
        print(json.dumps(finalize(repo_root), indent=2))
        return

    spec = _select(args.array_task_id, args.seed)
    data = exp72.prepare_data(repo_root)
    if args.command == "run-mean":
        payload = run_mean(spec, data, config, args.force)
        print(json.dumps({"spec": payload["spec"], "best_epoch": payload["best_epoch"]}, indent=2))
        return
    if args.command == "run-probe":
        payload = run_probe(spec, data, config, args.force)
        print(json.dumps({"spec": payload["spec"], "checkpoint_best_epoch": payload["checkpoint_best_epoch"]}, indent=2))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
