from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_0_4_l2_width_representation_capacity as source_backbone
from scripts import experiment_3_0_6_causal_temporal_decoding as exp306


EXPERIMENT_ID = "experiment_3_0_7_online_early_decision"
PROTOCOL_VERSION = "calibrated_commit_v1"

SEEDS = base.SEEDS
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
STABILITY_M = (1, 2, 3)
EXPECTED_RUNS = len(SEEDS)
TEMPERATURE_GRID = tuple(np.logspace(-1.0, 1.0, 161).tolist())


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    batch_size: int = base.BATCH_SIZE
    device: str = "cpu"
    resume: bool = True
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "snn").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate writingRing repository root")


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def selected_decoder(repo_root: Path) -> tuple[str, str]:
    path = exp306.results_dir(repo_root) / "selected_causal_decoder.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment 3.0.6 must be finalized before 3.0.7: missing {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("experiment_id") != exp306.EXPERIMENT_ID:
        raise ValueError(f"Unexpected 3.0.6 selection artifact: {path}")
    decoder = str(payload.get("selected_decoder"))
    objective = str(payload.get("source_objective"))
    if decoder not in exp306.DECODERS:
        raise ValueError(f"Unknown selected decoder: {decoder}")
    if objective not in base.OBJECTIVES:
        raise ValueError(f"Unknown source objective: {objective}")
    return decoder, objective


def run_path(root: Path, seed: int) -> Path:
    return root / "runs" / f"seed{seed}.json"


def _load_linear_model(
    repo_root: Path,
    decoder: str,
    seed: int,
) -> dict[str, np.ndarray]:
    path = exp306.model_path(exp306.results_dir(repo_root), decoder, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing Experiment 3.0.6 decoder model: {path}")
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _logits(
    features: np.ndarray,
    model: dict[str, np.ndarray],
) -> np.ndarray:
    scale = np.where(model["scaler_scale"] == 0.0, 1.0, model["scaler_scale"])
    z = (features.astype(np.float64) - model["scaler_mean"]) / scale
    return z @ model["coef"].T + model["intercept"]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits.astype(np.float64) / float(temperature)
    z -= z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = _softmax(logits, temperature)
    rows = np.arange(len(labels))
    return float(-np.log(np.clip(probs[rows, labels], 1e-12, 1.0)).mean())


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best: tuple[float, float] | None = None
    for temperature in TEMPERATURE_GRID:
        loss = _nll(logits, labels, float(temperature))
        if best is None or loss < best[0] - 1e-12:
            best = (loss, float(temperature))
    if best is None:
        raise RuntimeError("Temperature calibration failed")
    return best[1], best[0]


def _fixed_bin_totals(
    values: torch.Tensor,
    lengths: torch.Tensor,
    bin_steps: int,
) -> torch.Tensor:
    return base.fixed_counts(values, lengths, bin_steps).sum(dim=2)


def _extract_stream_partition(
    model: source_backbone.L2WidthNet,
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    split: str,
    objective: str,
    seed: int,
    config: Config,
    bin_steps: int,
) -> dict[str, np.ndarray]:
    loader = base.loader(
        X,
        y,
        lengths,
        config.batch_size,
        False,
        base.dseed(seed, objective, split, "exp307_loader"),
    )
    device = torch.device(config.device)
    y_parts: list[np.ndarray] = []
    length_parts: list[np.ndarray] = []
    l2_count_parts: list[np.ndarray] = []
    input_event_parts: list[np.ndarray] = []
    l1_spike_parts: list[np.ndarray] = []
    l2_spike_parts: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for Xb, yb, lb in loader:
            Xd = Xb.to(device)
            ld = lb.to(device)
            layers = model.layer_features(Xd)
            l1 = layers["L1"]
            l2 = layers["L2"]
            l2_counts = base.fixed_counts(l2, ld, bin_steps)
            input_events = _fixed_bin_totals((Xd != 0).to(Xd.dtype), ld, bin_steps)
            l1_spikes = _fixed_bin_totals(l1, ld, bin_steps)
            l2_spikes = _fixed_bin_totals(l2, ld, bin_steps)

            y_parts.append(yb.numpy())
            length_parts.append(lb.numpy())
            l2_count_parts.append(l2_counts.cpu().numpy())
            input_event_parts.append(input_events.cpu().numpy())
            l1_spike_parts.append(l1_spikes.cpu().numpy())
            l2_spike_parts.append(l2_spikes.cpu().numpy())

    return {
        "y": np.concatenate(y_parts),
        "lengths": np.concatenate(length_parts),
        "counts": np.concatenate(l2_count_parts),
        "input_events": np.concatenate(input_event_parts),
        "l1_spikes": np.concatenate(l1_spike_parts),
        "l2_spikes": np.concatenate(l2_spike_parts),
    }


def _sample_logits(
    counts: np.ndarray,
    decoder: str,
    model: dict[str, np.ndarray],
    n_valid_bins: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    one = counts[None, ...]
    for prefix_index in range(n_valid_bins):
        feature = exp306._feature_at_prefix(one, decoder, prefix_index)
        rows.append(_logits(feature, model)[0])
    return np.stack(rows, axis=0)


def _calibration_rows(
    partition: dict[str, np.ndarray],
    decoder: str,
    model: dict[str, np.ndarray],
    bin_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = partition["counts"]
    labels = partition["y"]
    valid_bins = exp306._valid_bins(partition["lengths"], bin_steps, counts.shape[1])
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for index, B in enumerate(valid_bins.astype(int)):
        logits_parts.append(_sample_logits(counts[index], decoder, model, B))
        label_parts.append(np.full(B, labels[index], dtype=np.int64))
    return np.concatenate(logits_parts), np.concatenate(label_parts)


def _policy_prediction(
    logits: np.ndarray,
    true_length: int,
    bin_steps: int,
    fs: float,
    temperature: float,
    threshold: float,
    stability: int,
) -> dict[str, object]:
    probs = _softmax(logits, temperature)
    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    duration_ms = float(true_length * 1000.0 / fs)

    # Only full checkpoints strictly before the known gesture offset can early-exit.
    regular_count = min(len(logits), max(0, (true_length - 1) // bin_steps))
    commit_index: int | None = None
    for index in range(regular_count):
        start = index - stability + 1
        if start < 0:
            continue
        labels = predicted[start : index + 1]
        confidences = confidence[start : index + 1]
        if np.all(labels == labels[-1]) and np.all(confidences >= threshold):
            commit_index = index
            break

    forced = commit_index is None
    if forced:
        decision_index = len(logits) - 1
        decision_ms = duration_ms
    else:
        decision_index = int(commit_index)
        decision_ms = float((decision_index + 1) * bin_steps * 1000.0 / fs)

    return {
        "prediction": int(predicted[decision_index]),
        "decision_index": int(decision_index),
        "decision_ms": decision_ms,
        "forced": bool(forced),
        "decision_confidence": float(confidence[decision_index]),
    }


def _evaluate_policy(
    partition: dict[str, np.ndarray],
    decoder: str,
    model: dict[str, np.ndarray],
    bin_steps: int,
    fs: float,
    temperature: float,
    threshold: float,
    stability: int,
) -> dict[str, float | int]:
    counts = partition["counts"]
    labels = partition["y"]
    lengths = partition["lengths"].astype(int)
    valid_bins = exp306._valid_bins(lengths, bin_steps, counts.shape[1])

    predictions = np.empty(len(labels), dtype=np.int64)
    latencies = np.empty(len(labels), dtype=np.float64)
    forced = np.zeros(len(labels), dtype=bool)
    spike_cost = np.empty(len(labels), dtype=np.float64)
    synops_cost = np.empty(len(labels), dtype=np.float64)

    for index, B in enumerate(valid_bins.astype(int)):
        logits = _sample_logits(counts[index], decoder, model, B)
        decision = _policy_prediction(
            logits,
            int(lengths[index]),
            bin_steps,
            fs,
            temperature,
            threshold,
            stability,
        )
        predictions[index] = int(decision["prediction"])
        latencies[index] = float(decision["decision_ms"])
        forced[index] = bool(decision["forced"])
        decision_bin = int(decision["decision_index"])
        n_used = decision_bin + 1
        input_events = float(partition["input_events"][index, :n_used].sum())
        l1_spikes = float(partition["l1_spikes"][index, :n_used].sum())
        l2_spikes = float(partition["l2_spikes"][index, :n_used].sum())
        spike_cost[index] = input_events + l1_spikes + l2_spikes
        synops_cost[index] = (
            source_backbone.L1_WIDTH * input_events
            + SOURCE_WIDTH * l1_spikes
            + len(base.LABELS) * l2_spikes
        )

    correct = predictions == labels
    early = ~forced
    premature_wrong = early & (~correct)
    metrics = {
        "n_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mean_decision_latency_ms": float(latencies.mean()),
        "median_decision_latency_ms": float(np.median(latencies)),
        "p_latency_le_500ms": float(np.mean(latencies <= 500.0)),
        "p_latency_le_750ms": float(np.mean(latencies <= 750.0)),
        "p_latency_le_1000ms": float(np.mean(latencies <= 1000.0)),
        "p_latency_le_1500ms": float(np.mean(latencies <= 1500.0)),
        "early_exit_coverage": float(np.mean(early)),
        "forced_decision_rate": float(np.mean(forced)),
        "premature_wrong_decision_rate": float(np.mean(premature_wrong)),
        "conditional_early_exit_accuracy": (
            float(np.mean(correct[early])) if np.any(early) else float("nan")
        ),
        "mean_spike_activity_cost": float(spike_cost.mean()),
        "median_spike_activity_cost": float(np.median(spike_cost)),
        "mean_synops_proxy": float(synops_cost.mean()),
        "median_synops_proxy": float(np.median(synops_cost)),
        "mean_spikes_per_correct_gesture": (
            float(spike_cost[correct].mean()) if np.any(correct) else float("nan")
        ),
        "mean_synops_per_correct_gesture": (
            float(synops_cost[correct].mean()) if np.any(correct) else float("nan")
        ),
    }
    return metrics


def _fixed_time_curve(
    partition: dict[str, np.ndarray],
    decoder: str,
    model: dict[str, np.ndarray],
    bin_steps: int,
    fs: float,
) -> list[dict[str, float | int]]:
    counts = partition["counts"]
    labels = partition["y"]
    lengths = partition["lengths"]
    valid_bins = exp306._valid_bins(lengths, bin_steps, counts.shape[1])
    rows: list[dict[str, float | int]] = []
    for prefix_index in range(counts.shape[1]):
        effective = np.minimum(prefix_index, valid_bins - 1)
        pred = np.empty(len(labels), dtype=np.int64)
        for index in range(len(labels)):
            feature = exp306._feature_at_prefix(
                counts[index : index + 1], decoder, int(effective[index])
            )
            pred[index] = int(_logits(feature, model)[0].argmax())
        rows.append(
            {
                "checkpoint_index": int(prefix_index),
                "elapsed_ms": float((prefix_index + 1) * bin_steps * 1000.0 / fs),
                "n_active": int(np.sum(valid_bins > prefix_index)),
                "accuracy": float(accuracy_score(labels, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
                "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
            }
        )
    return rows


def run_one(seed: int, data: base.Data, config: Config) -> dict[str, object]:
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}")
    out_path = run_path(config.results_dir, seed)
    if config.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    decoder, objective = selected_decoder(config.repo_root)
    linear_model = _load_linear_model(config.repo_root, decoder, seed)
    torch.set_num_threads(config.threads)
    snn_model, _ = source_backbone.load_model_for_evaluation(
        config.repo_root,
        source_backbone.results_dir(config.repo_root),
        SOURCE_WIDTH,
        objective,
        seed,
        data,
        torch.device(config.device),
    )
    raw_partitions = {
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    partitions = {
        split: _extract_stream_partition(
            snn_model,
            *values,
            split,
            objective,
            seed,
            config,
            data.bin_steps,
        )
        for split, values in raw_partitions.items()
    }

    calibration_logits, calibration_labels = _calibration_rows(
        partitions["val"], decoder, linear_model, data.bin_steps
    )
    temperature, calibration_nll = _fit_temperature(
        calibration_logits, calibration_labels
    )

    policy_rows: list[dict[str, object]] = []
    for split in ("val", "test"):
        for threshold in THRESHOLDS:
            for stability in STABILITY_M:
                metrics = _evaluate_policy(
                    partitions[split],
                    decoder,
                    linear_model,
                    data.bin_steps,
                    data.fs,
                    temperature,
                    threshold,
                    stability,
                )
                policy_rows.append(
                    {
                        "split": split,
                        "threshold": threshold,
                        "stability_m": stability,
                        "persistence_ms": (stability - 1) * CHECKPOINT_MS,
                        **metrics,
                    }
                )

    result: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "decoder": decoder,
        "source_objective": objective,
        "temperature": temperature,
        "validation_calibration_nll": calibration_nll,
        "thresholds": THRESHOLDS,
        "stability_m": STABILITY_M,
        "policy_results": policy_rows,
        "val_fixed_time_curve": _fixed_time_curve(
            partitions["val"], decoder, linear_model, data.bin_steps, data.fs
        ),
        "test_fixed_time_curve": _fixed_time_curve(
            partitions["test"], decoder, linear_model, data.bin_steps, data.fs
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=True)
    return result


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    policy_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        path = run_path(root, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing Experiment 3.0.7 run artifact: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        calibration_rows.append(
            {
                "seed": seed,
                "decoder": payload["decoder"],
                "source_objective": payload["source_objective"],
                "temperature": payload["temperature"],
                "validation_calibration_nll": payload["validation_calibration_nll"],
            }
        )
        for row in payload["policy_results"]:
            policy_rows.append({"seed": seed, **row})
        for split in ("val", "test"):
            for row in payload[f"{split}_fixed_time_curve"]:
                curve_rows.append({"seed": seed, "split": split, **row})

    policies = pd.DataFrame(policy_rows)
    curves = pd.DataFrame(curve_rows)
    calibration = pd.DataFrame(calibration_rows)

    test = policies[policies.split == "test"].copy()
    summary = (
        test.groupby(["threshold", "stability_m", "persistence_ms"], sort=True)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            sd_balanced_accuracy=("balanced_accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_latency_ms=("mean_decision_latency_ms", "mean"),
            median_latency_ms=("median_decision_latency_ms", "mean"),
            mean_early_exit_coverage=("early_exit_coverage", "mean"),
            mean_forced_decision_rate=("forced_decision_rate", "mean"),
            mean_premature_wrong_rate=("premature_wrong_decision_rate", "mean"),
            mean_spike_activity_cost=("mean_spike_activity_cost", "mean"),
            mean_synops_proxy=("mean_synops_proxy", "mean"),
        )
        .reset_index()
    )

    # Validation-only operating-point choice: maximize BA, then minimize latency.
    val = policies[policies.split == "val"].copy()
    val_summary = (
        val.groupby(["threshold", "stability_m"], sort=True)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_latency_ms=("mean_decision_latency_ms", "mean"),
            mean_forced_decision_rate=("forced_decision_rate", "mean"),
        )
        .reset_index()
        .sort_values(
            ["mean_balanced_accuracy", "mean_latency_ms", "threshold", "stability_m"],
            ascending=[False, True, True, True],
            ignore_index=True,
        )
    )
    selected = val_summary.iloc[0].to_dict()

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "policies": root / "experiment_3_0_7_policy_results.csv",
        "fixed_time_curves": root / "experiment_3_0_7_fixed_time_curves.csv",
        "calibration": root / "experiment_3_0_7_calibration.csv",
        "summary": root / "experiment_3_0_7_policy_summary.csv",
        "validation_selection": root / "experiment_3_0_7_validation_selection.csv",
        "selected_operating_point": root / "selected_operating_point.json",
    }
    policies.to_csv(outputs["policies"], index=False)
    curves.to_csv(outputs["fixed_time_curves"], index=False)
    calibration.to_csv(outputs["calibration"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    val_summary.to_csv(outputs["validation_selection"], index=False)
    with outputs["selected_operating_point"].open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "selection_split": "validation",
                "selection_rule": "maximize mean validation BA, then minimize mean decision latency",
                "threshold": float(selected["threshold"]),
                "stability_m": int(selected["stability_m"]),
                "mean_validation_balanced_accuracy": float(selected["mean_balanced_accuracy"]),
                "mean_validation_latency_ms": float(selected["mean_latency_ms"]),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 3.0.7 online early decision")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--batch-size", type=int, default=base.BATCH_SIZE)
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = find_repo_root()
    root = results_dir(repo_root)
    if args.command == "run-one":
        task_id = int(args.array_task_id)
        if task_id < 0 or task_id >= EXPECTED_RUNS:
            raise ValueError(f"array task id {task_id} outside 0..{EXPECTED_RUNS - 1}")
        seed = SEEDS[task_id]
        data = base.prepare_data(repo_root)
        result = run_one(
            seed,
            data,
            Config(
                repo_root=repo_root,
                results_dir=root,
                batch_size=args.batch_size,
                device=args.device,
                resume=not args.force,
            ),
        )
        print(
            f"completed seed={seed} decoder={result['decoder']} "
            f"temperature={result['temperature']:.6f}"
        )
    else:
        for name, path in finalize_experiment(repo_root).items():
            print(f"{name}: {path}")


SOURCE_WIDTH = 128
CHECKPOINT_MS = base.FIXED_MS


if __name__ == "__main__":
    main()
