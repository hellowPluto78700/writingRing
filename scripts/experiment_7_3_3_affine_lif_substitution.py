from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from scripts import experiment_7_2_6_output_readout_loss_shaping as exp726
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_7_3_2_affine_probe_bridge as exp732


EXPERIMENT_ID = "experiment_7_3_3_affine_lif_substitution"
PROTOCOL_VERSION = "affine_lif_substitution_v1"
ARCHITECTURE = exp73.ARCHITECTURE
SEEDS = exp73.SEEDS
BACKBONE_OBJECTIVES = ("tsce", "wcce")
SOURCE_CASE = "P7_wholecount_scale_center_bias"
LIF_BETA = exp73.LIF_BETA
IF_BETA = 1.0
THRESHOLD = exp73.THRESHOLD
OUTPUT_CAP = exp73.OUTPUT_CAP
MAX_ITER = exp732.MAX_ITER
REPRO_TOL = 1e-12
CHARGE_TOL = 1e-9

METHODS = (
    "analog_affine",
    "analog_w_only",
    "lif_beta05_w_only",
    "lif_beta05_bias_start",
    "lif_beta05_bias_end",
    "if_beta1_bias_start_count",
    "if_beta1_bias_start_charge",
    "if_beta1_bias_end_count",
    "if_beta1_bias_end_charge",
)


@dataclass(frozen=True)
class RunSpec:
    seed: int
    backbone_objective: str

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__backbone_{self.backbone_objective}__seed{self.seed}"

    @property
    def p7_spec(self) -> exp732.RunSpec:
        return exp732.RunSpec(self.seed, self.backbone_objective, SOURCE_CASE)


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(seed, backbone_objective)
        for seed in SEEDS
        for backbone_objective in BACKBONE_OBJECTIVES
    ]


def validate_spec(spec: RunSpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    if spec.backbone_objective not in BACKBONE_OBJECTIVES:
        raise ValueError(spec.backbone_objective)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    pred = scores.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def _source_p7_path(config: Config, spec: RunSpec) -> Path:
    return exp732.results_dir(config.repo_root) / "evaluations" / f"{spec.p7_spec.key}.json"


def _load_source_p7(config: Config, spec: RunSpec) -> dict[str, Any]:
    path = _source_p7_path(config, spec)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Exp7.3.2 P7 source for {spec.key}: {path}. Run/finalize Exp7.3.2 first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != exp732.EXPERIMENT_ID:
        raise ValueError(f"Wrong source experiment in {path}")
    if payload.get("protocol_version") != exp732.PROTOCOL_VERSION:
        raise ValueError(f"Wrong source protocol in {path}")
    if payload.get("case", {}).get("key") != SOURCE_CASE:
        raise ValueError(f"Wrong source case in {path}")
    return payload


def _reconstruct_p7(
    config: Config, spec: RunSpec, source: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    cache = exp732._load_cache(config.repo_root, spec.p7_spec)
    features = {
        split: exp732._aggregate_features(cache[split], "wholecount")
        for split in ("train", "val", "test")
    }
    scaler = StandardScaler(with_mean=True, with_std=True).fit(features["train"][0])
    transformed = {
        split: (scaler.transform(x), y)
        for split, (x, y) in features.items()
    }
    selected_C = float(source["selected_C"])
    classifier = LogisticRegression(
        C=selected_C,
        max_iter=MAX_ITER,
        solver="lbfgs",
        random_state=exp732._classifier_seed(spec.p7_spec),
        fit_intercept=True,
    ).fit(transformed["train"][0], transformed["train"][1])

    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    coef = np.asarray(classifier.coef_, dtype=np.float64)
    explicit_intercept = np.asarray(classifier.intercept_, dtype=np.float64)
    weight_raw = coef / scale[None, :]
    effective_intercept = explicit_intercept - weight_raw @ mean

    reconstruction_metrics: dict[str, Any] = {}
    max_ba_delta = 0.0
    max_prediction_mismatch = 0
    max_score_error = 0.0
    for split, (raw_x, y) in features.items():
        scaled_x = transformed[split][0]
        sklearn_scores = classifier.decision_function(scaled_x)
        raw_scores = raw_x @ weight_raw.T + effective_intercept[None, :]
        max_score_error = max(max_score_error, float(np.max(np.abs(sklearn_scores - raw_scores))))
        sklearn_pred = classifier.predict(scaled_x)
        raw_pred = raw_scores.argmax(axis=1)
        mismatch = int(np.sum(sklearn_pred != raw_pred))
        max_prediction_mismatch = max(max_prediction_mismatch, mismatch)
        metrics = _metrics(y, raw_scores)
        source_metrics = source["metrics"][split]
        ba_delta = abs(
            float(metrics["balanced_accuracy"])
            - float(source_metrics["balanced_accuracy"])
        )
        max_ba_delta = max(max_ba_delta, ba_delta)
        reconstruction_metrics[split] = {
            "metrics": metrics,
            "source_balanced_accuracy": float(source_metrics["balanced_accuracy"]),
            "abs_ba_delta": ba_delta,
            "prediction_mismatch_count": mismatch,
        }

    if max_ba_delta > REPRO_TOL or max_prediction_mismatch != 0:
        raise RuntimeError(
            f"P7 parameter reconstruction failed for {spec.key}: "
            f"max_ba_delta={max_ba_delta}, prediction_mismatch={max_prediction_mismatch}"
        )

    reconstruction = {
        "selected_C": selected_C,
        "max_abs_score_error": max_score_error,
        "max_abs_ba_delta": max_ba_delta,
        "max_prediction_mismatch_count": max_prediction_mismatch,
        "metrics": reconstruction_metrics,
        "weight_l2": float(np.linalg.norm(weight_raw)),
        "effective_intercept_l2": float(np.linalg.norm(effective_intercept)),
        "effective_intercept_span": float(
            effective_intercept.max() - effective_intercept.min()
        ),
    }
    return cache, weight_raw, effective_intercept, reconstruction


def _whole_count(l2: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return exp726._whole_count_np(l2, lengths).astype(np.float64, copy=False)


def _analog_affine_scores(
    l2: np.ndarray, lengths: np.ndarray, weight: np.ndarray, intercept: np.ndarray
) -> np.ndarray:
    return _whole_count(l2, lengths) @ weight.T + intercept[None, :]


def _simulate_affine_unipolar(
    l2: np.ndarray,
    lengths: np.ndarray,
    weight: np.ndarray,
    intercept: np.ndarray,
    *,
    beta: float,
    cap: int,
    bias_pulse: str,
) -> dict[str, Any]:
    if bias_pulse not in {"none", "start", "end"}:
        raise ValueError(bias_pulse)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(beta)
    if cap < 1:
        raise ValueError(cap)

    n, steps, _ = l2.shape
    k = weight.shape[0]
    mem = np.zeros((n, k), dtype=np.float64)
    counts = np.zeros((n, k), dtype=np.float64)
    input_sum = np.zeros((n, k), dtype=np.float64)
    leak_sum = np.zeros((n, k), dtype=np.float64)
    total_spikes = 0.0
    valid_positions = 0
    negative_evidence = 0
    bias_injection_samples = 0

    for t in range(steps):
        current = l2[:, t].astype(np.float64) @ weight.T
        vt = t < lengths
        if bias_pulse == "start":
            inject = (t == 0) & (lengths > 0)
        elif bias_pulse == "end":
            inject = (lengths > 0) & (t == (lengths - 1))
        else:
            inject = np.zeros(n, dtype=bool)
        if np.any(inject):
            current[inject] += intercept[None, :]
            bias_injection_samples += int(inject.sum())

        pre = beta * mem + current
        spikes = np.floor(np.maximum(pre, 0.0) / THRESHOLD)
        spikes = np.minimum(spikes, float(cap))
        spikes[~vt] = 0.0
        new_mem = pre - THRESHOLD * spikes

        input_sum[vt] += current[vt]
        counts += spikes
        negative_evidence += int((current[vt] < 0.0).sum())
        total_spikes += float(spikes[vt].sum())
        valid_positions += int(pre[vt].size)
        mem[vt] = new_mem[vt]

        survives = t < (lengths - 1)
        if np.any(survives):
            leak_sum[survives] += (1.0 - beta) * mem[survives]

    residual = mem.copy()
    spike_charge = THRESHOLD * counts
    identity_error = input_sum - (spike_charge + leak_sum + residual)
    if beta == 1.0:
        charge_scores = spike_charge + residual
    else:
        charge_scores = spike_charge + leak_sum + residual

    valid_steps = max(int(lengths.sum()), 1)
    diagnostics = {
        "beta": float(beta),
        "bias_pulse": bias_pulse,
        "bias_injection_samples": int(bias_injection_samples),
        "mean_total_output_spikes_per_sample": float(counts.sum(axis=1).mean()),
        "silent_sample_fraction": float((counts.sum(axis=1) == 0).mean()),
        "silent_output_neuron_fraction": float((counts == 0).mean()),
        "output_spikes_per_neuron_second": float(
            total_spikes * 64.0 / max(k * valid_steps, 1)
        ),
        "negative_evidence_fraction": negative_evidence / max(valid_positions, 1),
        "mean_abs_final_membrane": float(np.abs(residual).mean()),
        "max_abs_charge_identity_error": float(np.abs(identity_error).max()),
    }
    return {
        "counts": counts,
        "input_sum": input_sum,
        "spike_charge": spike_charge,
        "leak": leak_sum,
        "residual": residual,
        "charge_scores": charge_scores,
        "diagnostics": diagnostics,
    }


def _evaluate_split(
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    weight: np.ndarray,
    intercept: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    l2, y, lengths = split
    analog_affine = _analog_affine_scores(l2, lengths, weight, intercept)
    analog_w_only = _whole_count(l2, lengths) @ weight.T

    lif_w_only = _simulate_affine_unipolar(
        l2,
        lengths,
        weight,
        np.zeros_like(intercept),
        beta=LIF_BETA,
        cap=OUTPUT_CAP,
        bias_pulse="none",
    )
    lif_start = _simulate_affine_unipolar(
        l2,
        lengths,
        weight,
        intercept,
        beta=LIF_BETA,
        cap=OUTPUT_CAP,
        bias_pulse="start",
    )
    lif_end = _simulate_affine_unipolar(
        l2,
        lengths,
        weight,
        intercept,
        beta=LIF_BETA,
        cap=OUTPUT_CAP,
        bias_pulse="end",
    )
    if_start = _simulate_affine_unipolar(
        l2,
        lengths,
        weight,
        intercept,
        beta=IF_BETA,
        cap=OUTPUT_CAP,
        bias_pulse="start",
    )
    if_end = _simulate_affine_unipolar(
        l2,
        lengths,
        weight,
        intercept,
        beta=IF_BETA,
        cap=OUTPUT_CAP,
        bias_pulse="end",
    )

    start_charge_error = float(np.max(np.abs(if_start["charge_scores"] - analog_affine)))
    end_charge_error = float(np.max(np.abs(if_end["charge_scores"] - analog_affine)))
    if start_charge_error > CHARGE_TOL or end_charge_error > CHARGE_TOL:
        raise RuntimeError(
            "beta=1 charge-preserving control failed: "
            f"start={start_charge_error}, end={end_charge_error}"
        )

    scores = {
        "analog_affine": analog_affine,
        "analog_w_only": analog_w_only,
        "lif_beta05_w_only": lif_w_only["counts"],
        "lif_beta05_bias_start": lif_start["counts"],
        "lif_beta05_bias_end": lif_end["counts"],
        "if_beta1_bias_start_count": if_start["counts"],
        "if_beta1_bias_start_charge": if_start["charge_scores"],
        "if_beta1_bias_end_count": if_end["counts"],
        "if_beta1_bias_end_charge": if_end["charge_scores"],
    }
    metrics = {method: _metrics(y, values) for method, values in scores.items()}
    diagnostics = {
        "lif_beta05_w_only": lif_w_only["diagnostics"],
        "lif_beta05_bias_start": lif_start["diagnostics"],
        "lif_beta05_bias_end": lif_end["diagnostics"],
        "if_beta1_bias_start": if_start["diagnostics"],
        "if_beta1_bias_end": if_end["diagnostics"],
        "if_beta1_start_charge_vs_analog_max_abs_error": start_charge_error,
        "if_beta1_end_charge_vs_analog_max_abs_error": end_charge_error,
    }
    return metrics, diagnostics


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    path = config.results_dir / "evaluations" / f"{spec.key}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    source = _load_source_p7(config, spec)
    cache, weight, intercept, reconstruction = _reconstruct_p7(config, spec, source)

    split_metrics: dict[str, Any] = {}
    split_diagnostics: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        metrics, diagnostics = _evaluate_split(cache[split], weight, intercept)
        split_metrics[split] = metrics
        split_diagnostics[split] = diagnostics

    source_test_ba = float(source["metrics"]["test"]["balanced_accuracy"])
    reconstructed_test_ba = float(
        split_metrics["test"]["analog_affine"]["balanced_accuracy"]
    )
    if abs(source_test_ba - reconstructed_test_ba) > REPRO_TOL:
        raise RuntimeError(
            f"Analog P7 reproduction mismatch for {spec.key}: "
            f"source={source_test_ba}, reconstructed={reconstructed_test_ba}"
        )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "architecture": ARCHITECTURE,
            "frozen_l1_l2": True,
            "source_experiment": exp732.EXPERIMENT_ID,
            "source_protocol": exp732.PROTOCOL_VERSION,
            "source_case": SOURCE_CASE,
            "source_selected_C": float(source["selected_C"]),
            "parameter_recovery": (
                "refit exact Exp7.3.2 P7 formulation at its already-selected C because "
                "Exp7.3.2 did not persist coef_/intercept_; no new hyperparameter selection"
            ),
            "same_affine_parameters_across_readouts": True,
            "no_backbone_retraining": True,
            "no_head_hyperparameter_reselection": True,
            "no_gain_calibration": True,
            "no_threshold_sweep": True,
            "no_beta_sweep": True,
            "lif_beta": LIF_BETA,
            "if_beta": IF_BETA,
            "threshold": THRESHOLD,
            "output_cap": OUTPUT_CAP,
            "bias_pulses": ["start", "end"],
        },
        "source_p7": {
            "path": str(_source_p7_path(config, spec).relative_to(config.repo_root)),
            "selected_C": float(source["selected_C"]),
            "test_balanced_accuracy": source_test_ba,
            "source_method": source["contract"]["source_exp73_method"],
            "source_best_epoch": int(source["contract"]["source_exp73_best_epoch"]),
        },
        "parameter_reconstruction": reconstruction,
        "parameters": {
            "weight_shape": list(weight.shape),
            "weight_l2": float(np.linalg.norm(weight)),
            "effective_intercept_l2": float(np.linalg.norm(intercept)),
            "effective_intercept_span": float(intercept.max() - intercept.min()),
        },
        "metrics": split_metrics,
        "diagnostics": split_diagnostics,
    }
    _save_json(path, payload)
    return payload


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    source_ba = float(payload["source_p7"]["test_balanced_accuracy"])
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        test = payload["metrics"]["test"][method]
        val = payload["metrics"]["val"][method]
        train = payload["metrics"]["train"][method]
        test_ba = float(test["balanced_accuracy"])
        rows.append(
            {
                "seed": int(spec["seed"]),
                "backbone_objective": spec["backbone_objective"],
                "method": method,
                "train_ba": float(train["balanced_accuracy"]),
                "val_ba": float(val["balanced_accuracy"]),
                "test_ba": test_ba,
                "test_accuracy": float(test["accuracy"]),
                "test_macro_f1": float(test["macro_f1"]),
                "source_p7_test_ba": source_ba,
                "delta_vs_affine_p7_pp": 100.0 * (test_ba - source_ba),
                "selected_C": float(payload["source_p7"]["selected_C"]),
                "weight_l2": float(payload["parameters"]["weight_l2"]),
                "effective_intercept_l2": float(
                    payload["parameters"]["effective_intercept_l2"]
                ),
            }
        )
    return rows


def _paired_contrast(
    runs: pd.DataFrame, name: str, left: str, right: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backbone in BACKBONE_OBJECTIVES:
        left_df = runs[(runs.backbone_objective == backbone) & (runs.method == left)]
        right_df = runs[(runs.backbone_objective == backbone) & (runs.method == right)]
        merged = left_df[["seed", "test_ba"]].merge(
            right_df[["seed", "test_ba"]], on="seed", suffixes=("_left", "_right")
        )
        for _, row in merged.iterrows():
            rows.append(
                {
                    "contrast": name,
                    "backbone_objective": backbone,
                    "seed": int(row["seed"]),
                    "left_method": left,
                    "right_method": right,
                    "delta_pp": 100.0
                    * (float(row["test_ba_left"]) - float(row["test_ba_right"])),
                }
            )
    return rows


def finalize(config: Config) -> None:
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
        raise FileNotFoundError(f"Missing {len(missing)} Exp7.3.3 evaluations:\n{preview}")

    runs = pd.DataFrame([row for payload in payloads for row in _rows(payload)]).sort_values(
        ["backbone_objective", "method", "seed"]
    )
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "method_runs.csv", index=False)

    numeric = [
        "train_ba",
        "val_ba",
        "test_ba",
        "test_accuracy",
        "test_macro_f1",
        "source_p7_test_ba",
        "delta_vs_affine_p7_pp",
        "selected_C",
        "weight_l2",
        "effective_intercept_l2",
    ]
    grouped = runs.groupby(["backbone_objective", "method"])[numeric].agg(["mean", "std"])
    grouped.columns = [f"{name}_{stat}" for name, stat in grouped.columns]
    grouped.reset_index().to_csv(config.results_dir / "method_summary.csv", index=False)

    contrast_defs = (
        ("analog_w_only_minus_affine", "analog_w_only", "analog_affine"),
        ("lif_w_only_minus_affine", "lif_beta05_w_only", "analog_affine"),
        ("lif_start_minus_affine", "lif_beta05_bias_start", "analog_affine"),
        ("lif_end_minus_affine", "lif_beta05_bias_end", "analog_affine"),
        ("lif_start_minus_w_only", "lif_beta05_bias_start", "lif_beta05_w_only"),
        ("lif_end_minus_w_only", "lif_beta05_bias_end", "lif_beta05_w_only"),
        ("lif_end_minus_start", "lif_beta05_bias_end", "lif_beta05_bias_start"),
        ("if_start_count_minus_affine", "if_beta1_bias_start_count", "analog_affine"),
        ("if_end_count_minus_affine", "if_beta1_bias_end_count", "analog_affine"),
        ("if_start_charge_minus_affine", "if_beta1_bias_start_charge", "analog_affine"),
        ("if_end_charge_minus_affine", "if_beta1_bias_end_charge", "analog_affine"),
    )
    contrasts: list[dict[str, Any]] = []
    for name, left, right in contrast_defs:
        contrasts.extend(_paired_contrast(runs, name, left, right))
    contrast_runs = pd.DataFrame(contrasts)
    contrast_runs.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_summary = (
        contrast_runs.groupby(["contrast", "backbone_objective"])["delta_pp"]
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

    repro_rows = []
    for payload in payloads:
        spec = payload["spec"]
        repro = payload["parameter_reconstruction"]
        repro_rows.append(
            {
                "seed": int(spec["seed"]),
                "backbone_objective": spec["backbone_objective"],
                "selected_C": float(payload["source_p7"]["selected_C"]),
                "source_p7_test_ba": float(payload["source_p7"]["test_balanced_accuracy"]),
                "reconstructed_p7_test_ba": float(
                    payload["metrics"]["test"]["analog_affine"]["balanced_accuracy"]
                ),
                "max_abs_ba_delta": float(repro["max_abs_ba_delta"]),
                "max_prediction_mismatch_count": int(
                    repro["max_prediction_mismatch_count"]
                ),
                "max_abs_score_error": float(repro["max_abs_score_error"]),
                "if_start_charge_max_abs_error": float(
                    payload["diagnostics"]["test"][
                        "if_beta1_start_charge_vs_analog_max_abs_error"
                    ]
                ),
                "if_end_charge_max_abs_error": float(
                    payload["diagnostics"]["test"][
                        "if_beta1_end_charge_vs_analog_max_abs_error"
                    ]
                ),
            }
        )
    pd.DataFrame(repro_rows).sort_values(
        ["backbone_objective", "seed"]
    ).to_csv(config.results_dir / "p7_reproduction_and_charge_checks.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "seeds": list(SEEDS),
        "backbone_objectives": list(BACKBONE_OBJECTIVES),
        "source_case": SOURCE_CASE,
        "methods": list(METHODS),
        "logical_runs": len(run_specs()),
        "lif_beta": LIF_BETA,
        "if_beta": IF_BETA,
        "threshold": THRESHOLD,
        "output_cap": OUTPUT_CAP,
        "notes": (
            "P7 affine parameters are reconstructed at the already-selected Exp7.3.2 C. "
            "All readouts use the same recovered W and effective sequence-level intercept. "
            "No gain, threshold, beta, or parameter re-optimization is performed."
        ),
    }
    _save_json(config.results_dir / "manifest.json", manifest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=find_repo_root())
    parser.add_argument("--threads", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--force", action="store_true")

    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    config = Config(repo_root=repo_root, results_dir=results_dir(repo_root), threads=args.threads)
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
