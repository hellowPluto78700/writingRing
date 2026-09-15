from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_2_5_output_synaptic_alpha as exp725
from scripts import experiment_7_2_6_output_readout_loss_shaping as exp726
from scripts import experiment_7_2_6_1_sum_vs_mean_ce as exp7261


EXPERIMENT_ID = "experiment_7_2_6_2_mean_ce_bias"
PROTOCOL_VERSION = "mean_ce_bias_v1"
ARCHITECTURE = exp726.ARCHITECTURE
REGULARIZATION = exp72.TASK_ONLY
SEEDS = exp726.SEEDS
MAX_EPOCHS = exp726.MAX_EPOCHS
MIN_EPOCHS = exp726.MIN_EPOCHS
PATIENCE = exp726.PATIENCE

NO_BIAS_REUSE = "mean_ce_bias0_reuse"
BIAS_TRAINED_WITH_BIAS = "mean_ce_bias1_with_bias"
BIAS_TRAINED_ZERO_BIAS = "mean_ce_bias1_zero_bias"
NATIVE_CONDITIONS = (NO_BIAS_REUSE, BIAS_TRAINED_WITH_BIAS, BIAS_TRAINED_ZERO_BIAS)
PROBE_CONDITIONS = (NO_BIAS_REUSE, BIAS_TRAINED_WITH_BIAS)


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
    def baseline_spec(self) -> exp7261.RunSpec:
        return exp7261.RunSpec(self.seed)

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__{REGULARIZATION}__seed{self.seed}__mean_ce_bias1"


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


def _baseline_root(repo_root: Path) -> Path:
    return exp7261.results_dir(repo_root)


def _baseline_path(repo_root: Path, kind: str, spec: RunSpec, suffix: str) -> Path:
    return exp7261._path(_baseline_root(repo_root), kind, spec.baseline_spec, suffix)


def _require_baseline(repo_root: Path, spec: RunSpec, require_probe: bool = False) -> None:
    required = [
        _baseline_path(repo_root, "evaluations", spec, ".json"),
        _baseline_path(repo_root, "checkpoints", spec, ".pt"),
        _baseline_path(repo_root, "histories", spec, ".csv"),
    ]
    if require_probe:
        required.append(_baseline_path(repo_root, "probe_evaluations", spec, ".json"))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Exp7.2.6.2 reuses Exp7.2.6.1 task-only Mean-CE bias-free runs. "
            f"Missing baseline artifacts for seed {spec.seed}: {missing}. Run/finalize Exp7.2.6.1 first."
        )


def _build_bias_model(
    spec: RunSpec,
    n_classes: int,
    fs: float,
    device: torch.device,
) -> exp726.PairedShapingSNN:
    """Create exact Exp7.2.6.1 initialization, then add a zero-initialized bias."""
    exp3.seed_all(exp726._reference_seed(spec.source, "model_init"))
    model = exp726.PairedShapingSNN(exp726.E2E_ANALOG, n_classes, fs).to(device)
    if model.output_linear.bias is not None:
        raise RuntimeError("Expected Exp7.2.6.1 reference projection to be bias-free")
    reference_weight = model.output_linear.weight.detach().clone()
    biased = nn.Linear(exp726.HIDDEN_WIDTH, n_classes, bias=True).to(device=device, dtype=reference_weight.dtype)
    with torch.no_grad():
        biased.weight.copy_(reference_weight)
        biased.bias.zero_()
    model.output_linear = biased
    return model


def _mean_logits(model: exp726.PairedShapingSNN, tr: dict[str, Any], lengths: torch.Tensor) -> torch.Tensor:
    return exp7261._aggregate_evidence(tr["output_evidence"], lengths, "mean")


def _metrics_from_logits(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    pred = logits.argmax(1)
    return {
        **exp72._metrics(y, pred),
        "objective_loss": float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y))),
    }


def _evaluate_bias_model(
    model: exp726.PairedShapingSNN,
    loader: Iterable,
    device: torch.device,
) -> dict[str, Any]:
    if model.output_linear.bias is None:
        raise RuntimeError("Bias-model evaluation requires output_linear.bias")

    labels: list[np.ndarray] = []
    lengths_out: list[np.ndarray] = []
    mean_logits_out: list[np.ndarray] = []
    sum_logits_out: list[np.ndarray] = []
    zero_logits_out: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            tr = model.forward_trajectory(X)
            mean_logits = exp7261._aggregate_evidence(tr["output_evidence"], lengths, "mean")
            sum_logits = exp7261._aggregate_evidence(tr["output_evidence"], lengths, "sum")
            zero_logits = mean_logits - model.output_linear.bias.unsqueeze(0)
            labels.append(y.cpu().numpy())
            lengths_out.append(lengths.cpu().numpy())
            mean_logits_out.append(mean_logits.cpu().numpy())
            sum_logits_out.append(sum_logits.cpu().numpy())
            zero_logits_out.append(zero_logits.cpu().numpy())

    y = np.concatenate(labels)
    lengths_np = np.concatenate(lengths_out).astype(np.float64)
    with_logits = np.concatenate(mean_logits_out)
    sum_logits_np = np.concatenate(sum_logits_out)
    zero_logits = np.concatenate(zero_logits_out)
    with_pred = with_logits.argmax(1)
    zero_pred = zero_logits.argmax(1)
    with_correct = with_pred == y
    zero_correct = zero_pred == y
    relation_error = float(np.max(np.abs(sum_logits_np - with_logits * lengths_np[:, None])))
    bias = model.output_linear.bias.detach().cpu().numpy().astype(np.float64)
    zero_norm = np.linalg.norm(zero_logits, axis=1)

    return {
        "with_bias": _metrics_from_logits(y, with_logits),
        "zero_bias": _metrics_from_logits(y, zero_logits),
        "sum_mean_prediction_equal": bool(np.array_equal(sum_logits_np.argmax(1), with_pred)),
        "max_sum_equals_T_times_mean_error": relation_error,
        "prediction_change_fraction": float(np.mean(with_pred != zero_pred)),
        "zero_wrong_to_with_correct_fraction": float(np.mean((~zero_correct) & with_correct)),
        "zero_correct_to_with_wrong_fraction": float(np.mean(zero_correct & (~with_correct))),
        "bias_diagnostics": {
            "bias_l2": float(np.linalg.norm(bias)),
            "bias_abs_mean": float(np.abs(bias).mean()),
            "bias_std": float(bias.std()),
            "bias_range": float(bias.max() - bias.min()),
            "mean_zero_bias_logit_l2": float(zero_norm.mean()),
            "bias_l2_to_mean_zero_logit_l2": float(np.linalg.norm(bias) / max(float(zero_norm.mean()), 1e-12)),
        },
        "bias_vector": bias.tolist(),
    }


def _evaluate_with_bias(model, loader, device: torch.device) -> dict[str, float]:
    return _evaluate_bias_model(model, loader, device)["with_bias"]


def _load_bias_model(spec: RunSpec, data: exp3.Data, config: Config):
    checkpoint = _path(config.results_dir, "checkpoints", spec, ".pt")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location=config.device, weights_only=False)
    model = _build_bias_model(spec, len(data.labels), data.fs, torch.device(config.device))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def run_bias(spec: RunSpec, data: exp3.Data, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    _require_baseline(config.repo_root, spec)
    eval_path = _path(config.results_dir, "evaluations", spec, ".json")
    checkpoint_path = _path(config.results_dir, "checkpoints", spec, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _build_bias_model(spec, len(data.labels), data.fs, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY)
    train_loader = exp726._e2e_loaders(data, spec.c0_spec, config.batch_size, True)["train"]
    eval_loaders = exp726._e2e_loaders(data, spec.c0_spec, config.batch_size, False)

    initial_eval = _evaluate_bias_model(model, eval_loaders["val"], device)
    if initial_eval["bias_diagnostics"]["bias_l2"] != 0.0:
        raise RuntimeError("Bias must be initialized to exactly zero")
    if not initial_eval["sum_mean_prediction_equal"]:
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
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            tr = model.forward_trajectory(X)
            logits = _mean_logits(model, tr, lengths)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            n += len(y)

        train_metrics = _evaluate_with_bias(model, eval_loaders["train"], device)
        val_eval = _evaluate_bias_model(model, eval_loaders["val"], device)
        val_metrics = val_eval["with_bias"]
        if not val_eval["sum_mean_prediction_equal"]:
            raise RuntimeError(f"sum/mean prediction equivalence failed at epoch {epoch}")
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": loss_sum / max(n, 1),
                "val_loss": float(val_metrics["objective_loss"]),
                "val_zero_bias_ba": float(val_eval["zero_bias"]["balanced_accuracy"]),
                "val_zero_bias_loss": float(val_eval["zero_bias"]["objective_loss"]),
                "val_direct_bias_delta_ba": float(val_metrics["balanced_accuracy"] - val_eval["zero_bias"]["balanced_accuracy"]),
                "val_prediction_change_fraction": float(val_eval["prediction_change_fraction"]),
                "bias_l2": float(val_eval["bias_diagnostics"]["bias_l2"]),
                "bias_abs_mean": float(val_eval["bias_diagnostics"]["bias_abs_mean"]),
                "bias_range": float(val_eval["bias_diagnostics"]["bias_range"]),
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
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "model_state_dict": best_state,
            "paired_exp7261_spec": asdict(spec.baseline_spec),
            "pairing": "same hidden/output-weight initialization and loader stream; added output bias initialized to zero",
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state, strict=True)
    final_eval = {split: _evaluate_bias_model(model, loader, device) for split, loader in eval_loaders.items()}
    for split, result in final_eval.items():
        if not result["sum_mean_prediction_equal"]:
            raise RuntimeError(f"final sum/mean prediction equivalence failed on {split}")

    history_path = _path(config.results_dir, "histories", spec, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "initial_val": initial_eval,
        "final": final_eval,
        "metrics": {split: result["with_bias"] for split, result in final_eval.items()},
        "zero_bias_metrics": {split: result["zero_bias"] for split, result in final_eval.items()},
        "checkpoint_selection": "highest validation BA with bias under Mean-CE; validation Mean-CE loss tie-break",
        "class_labels": [str(label) for label in data.labels],
    }
    _save_json(eval_path, payload)
    return payload


def run_probe(spec: RunSpec, data: exp3.Data, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    _require_baseline(config.repo_root, spec, require_probe=True)
    path = _path(config.results_dir, "probe_evaluations", spec, ".json")
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    device = torch.device(config.device)
    model, checkpoint = _load_bias_model(spec, data, config)
    loaders = exp726._e2e_loaders(data, spec.c0_spec, config.batch_size, False)
    splits = {split: exp726._extract_l2_generic(model, loader, device) for split, loader in loaders.items()}
    ref_spec = spec.source.exp725_spec
    standardized_wc = exp725._fit_e2e_probe(exp726.PROBE_STD_WC, splits, ref_spec, data.bin_steps)
    fixed250 = exp725._fit_e2e_probe(exp726.PROBE_F250, splits, ref_spec, data.bin_steps)
    matched_wc = exp726._fit_matched_count_probe(splits, spec.c0_spec, len(data.labels))
    eval_payload = json.loads(_path(config.results_dir, "evaluations", spec, ".json").read_text(encoding="utf-8"))
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
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
    missing: list[str] = []
    native_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    bias_value_rows: list[dict[str, Any]] = []
    probe_delta_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []

    for spec in specs():
        new_eval_path = _path(root, "evaluations", spec, ".json")
        new_probe_path = _path(root, "probe_evaluations", spec, ".json")
        new_hist_path = _path(root, "histories", spec, ".csv")
        base_eval_path = _baseline_path(repo_root, "evaluations", spec, ".json")
        base_probe_path = _baseline_path(repo_root, "probe_evaluations", spec, ".json")
        base_hist_path = _baseline_path(repo_root, "histories", spec, ".csv")
        for path in (new_eval_path, new_probe_path, new_hist_path, base_eval_path, base_probe_path, base_hist_path):
            if not path.exists():
                missing.append(str(path))
        if any(not p.exists() for p in (new_eval_path, new_probe_path, base_eval_path, base_probe_path)):
            continue

        new_eval = json.loads(new_eval_path.read_text(encoding="utf-8"))
        new_probe = json.loads(new_probe_path.read_text(encoding="utf-8"))
        base_eval = json.loads(base_eval_path.read_text(encoding="utf-8"))
        base_probe = json.loads(base_probe_path.read_text(encoding="utf-8"))

        for split in ("train", "val", "test"):
            conditions = {
                NO_BIAS_REUSE: base_eval["metrics"][split],
                BIAS_TRAINED_WITH_BIAS: new_eval["metrics"][split],
                BIAS_TRAINED_ZERO_BIAS: new_eval["zero_bias_metrics"][split],
            }
            for condition, metrics in conditions.items():
                native_rows.append({
                    "architecture": ARCHITECTURE,
                    "regularization": REGULARIZATION,
                    "seed": spec.seed,
                    "condition": condition,
                    "split": split,
                    "accuracy": float(metrics["accuracy"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "objective_loss": float(metrics["objective_loss"]),
                })

            no_bias = float(conditions[NO_BIAS_REUSE]["balanced_accuracy"])
            with_bias = float(conditions[BIAS_TRAINED_WITH_BIAS]["balanced_accuracy"])
            zero_bias = float(conditions[BIAS_TRAINED_ZERO_BIAS]["balanced_accuracy"])
            total = with_bias - no_bias
            direct = with_bias - zero_bias
            training = zero_bias - no_bias
            decomposition_rows.append({
                "seed": spec.seed,
                "split": split,
                "no_bias_ba": no_bias,
                "bias_trained_with_bias_ba": with_bias,
                "bias_trained_zero_bias_ba": zero_bias,
                "total_bias_effect": total,
                "direct_readout_effect": direct,
                "e2e_training_effect": training,
                "decomposition_error": total - direct - training,
            })

            detail = new_eval["final"][split]
            diag = detail["bias_diagnostics"]
            bias_rows.append({
                "seed": spec.seed,
                "split": split,
                "prediction_change_fraction": float(detail["prediction_change_fraction"]),
                "zero_wrong_to_with_correct_fraction": float(detail["zero_wrong_to_with_correct_fraction"]),
                "zero_correct_to_with_wrong_fraction": float(detail["zero_correct_to_with_wrong_fraction"]),
                "sum_mean_prediction_equal": bool(detail["sum_mean_prediction_equal"]),
                "max_sum_equals_T_times_mean_error": float(detail["max_sum_equals_T_times_mean_error"]),
                **{key: float(value) for key, value in diag.items()},
            })

        test_detail = new_eval["final"]["test"]
        labels = new_eval["class_labels"]
        for index, value in enumerate(test_detail["bias_vector"]):
            bias_value_rows.append({"seed": spec.seed, "class_index": index, "class_label": labels[index], "bias": float(value)})

        for condition, probes in ((NO_BIAS_REUSE, base_probe["probes"]), (BIAS_TRAINED_WITH_BIAS, new_probe["probes"])):
            for probe_name, probe in probes.items():
                for split, metrics in probe["metrics"].items():
                    probe_rows.append({
                        "architecture": ARCHITECTURE,
                        "regularization": REGULARIZATION,
                        "seed": spec.seed,
                        "condition": condition,
                        "probe": probe_name,
                        "split": split,
                        "accuracy": float(metrics["accuracy"]),
                        "balanced_accuracy": float(metrics["balanced_accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                    })
        for probe_name in (exp726.PROBE_MATCHED_WC, exp726.PROBE_STD_WC, exp726.PROBE_F250):
            base_ba = float(base_probe["probes"][probe_name]["metrics"]["test"]["balanced_accuracy"])
            bias_ba = float(new_probe["probes"][probe_name]["metrics"]["test"]["balanced_accuracy"])
            probe_delta_rows.append({
                "seed": spec.seed,
                "probe": probe_name,
                "no_bias_test_ba": base_ba,
                "bias_trained_test_ba": bias_ba,
                "bias_trained_minus_no_bias": bias_ba - base_ba,
            })

        if base_hist_path.exists():
            frame = pd.read_csv(base_hist_path).copy()
            frame["condition"] = NO_BIAS_REUSE
            frame["seed"] = spec.seed
            history_frames.append(frame)
        if new_hist_path.exists():
            frame = pd.read_csv(new_hist_path).copy()
            frame["condition"] = BIAS_TRAINED_WITH_BIAS
            frame["seed"] = spec.seed
            history_frames.append(frame)

    if missing:
        raise RuntimeError(f"Missing {len(missing)} Exp7.2.6.2/baseline artifacts; first={missing[:8]}")

    native = pd.DataFrame(native_rows)
    native.to_csv(root / "native_performance_runs.csv", index=False)
    _aggregate(native, ["condition", "split"], ["accuracy", "balanced_accuracy", "macro_f1", "objective_loss"]).to_csv(root / "native_performance_summary.csv", index=False)

    decomposition = pd.DataFrame(decomposition_rows)
    decomposition.to_csv(root / "bias_effect_decomposition_runs.csv", index=False)
    _aggregate(decomposition, ["split"], ["no_bias_ba", "bias_trained_with_bias_ba", "bias_trained_zero_bias_ba", "total_bias_effect", "direct_readout_effect", "e2e_training_effect", "decomposition_error"]).to_csv(root / "bias_effect_decomposition_summary.csv", index=False)

    probes = pd.DataFrame(probe_rows)
    probes.to_csv(root / "l2_probe_runs.csv", index=False)
    _aggregate(probes, ["condition", "probe", "split"], ["accuracy", "balanced_accuracy", "macro_f1"]).to_csv(root / "l2_probe_summary.csv", index=False)

    probe_delta = pd.DataFrame(probe_delta_rows)
    probe_delta.to_csv(root / "l2_probe_delta_runs.csv", index=False)
    _aggregate(probe_delta, ["probe"], ["no_bias_test_ba", "bias_trained_test_ba", "bias_trained_minus_no_bias"]).to_csv(root / "l2_probe_delta_summary.csv", index=False)

    bias_diag = pd.DataFrame(bias_rows)
    bias_diag.to_csv(root / "bias_diagnostics_runs.csv", index=False)
    _aggregate(
        bias_diag,
        ["split"],
        [
            "prediction_change_fraction",
            "zero_wrong_to_with_correct_fraction",
            "zero_correct_to_with_wrong_fraction",
            "bias_l2",
            "bias_abs_mean",
            "bias_std",
            "bias_range",
            "mean_zero_bias_logit_l2",
            "bias_l2_to_mean_zero_logit_l2",
        ],
    ).to_csv(root / "bias_diagnostics_summary.csv", index=False)

    pd.DataFrame(bias_value_rows).to_csv(root / "bias_values.csv", index=False)

    histories = pd.concat(history_frames, ignore_index=True)
    histories.to_csv(root / "training_history_runs.csv", index=False)
    history_values = [c for c in ("train_ba", "val_ba", "train_loss", "val_loss", "val_zero_bias_ba", "val_direct_bias_delta_ba", "bias_l2") if c in histories.columns]
    _aggregate(histories, ["condition", "epoch"], history_values).to_csv(root / "training_history_summary.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "regularization": REGULARIZATION,
        "seeds": list(SEEDS),
        "new_training_runs": len(specs()),
        "baseline": "Exp7.2.6.1 Mean-CE bias=False, reused without retraining",
        "new_condition": "Mean-CE bias=True; output W copied from exact paired bias-free initialization; bias initialized to zero",
        "checkpoint_selection": "highest validation BA with bias; validation Mean-CE loss tie-break",
        "primary_decomposition": "total = direct readout + E2E training effect, using bias-zeroed inference on the bias-trained model",
        "notebook_inputs": [
            "native_performance_summary.csv",
            "bias_effect_decomposition_summary.csv",
            "l2_probe_summary.csv",
            "l2_probe_delta_summary.csv",
            "bias_diagnostics_summary.csv",
            "bias_values.csv",
            "training_history_summary.csv",
        ],
    }
    _save_json(root / "manifest.json", manifest)
    return manifest


def _select_spec(index: int | None, seed: int | None) -> RunSpec:
    grid = specs()
    if index is not None:
        if not 0 <= index < len(grid):
            raise ValueError(index)
        return grid[index]
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
    for name in ("run-bias", "run-probe"):
        p = sub.add_parser(name)
        p.add_argument("--array-task-id", type=int)
        p.add_argument("--seed", type=int, choices=SEEDS)
        p.add_argument("--force", action="store_true")
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
    data = exp72.prepare_data(repo_root)
    spec = _select_spec(args.array_task_id, args.seed)
    if args.command == "run-bias":
        payload = run_bias(spec, data, config, args.force)
        print(json.dumps({"spec": payload["spec"], "best_epoch": payload["best_epoch"]}, indent=2))
        return
    if args.command == "run-probe":
        payload = run_probe(spec, data, config, args.force)
        print(json.dumps({"spec": payload["spec"], "best_epoch": payload["checkpoint_best_epoch"]}, indent=2))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
