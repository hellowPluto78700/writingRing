from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_0_4_l2_width_representation_capacity as source_backbone
from scripts import experiment_3_0_5_frozen_representation_accessibility as exp305


EXPERIMENT_ID = "experiment_3_0_6_causal_temporal_decoding"
PROTOCOL_VERSION = "frozen_causal_v1"

DECODERS = (
    "current250",
    "cumulative250",
    "prefix250_uniform",
    "prefix250_weighted",
)
SEEDS = base.SEEDS
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
EXPECTED_RUNS = len(DECODERS) * len(SEEDS)

SOURCE_WIDTH = 128
CHECKPOINT_MS = base.FIXED_MS


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


def run_specs() -> list[tuple[str, int]]:
    return [(decoder, seed) for decoder in DECODERS for seed in SEEDS]


def selected_objective(repo_root: Path) -> str:
    path = exp305.selected_backbone_path(exp305.results_dir(repo_root))
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment 3.0.5 must be finalized before 3.0.6: missing {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("experiment_id") != exp305.EXPERIMENT_ID:
        raise ValueError(f"Unexpected 3.0.5 selection artifact: {path}")
    objective = str(payload.get("selected_objective"))
    if objective not in base.OBJECTIVES:
        raise ValueError(f"Unknown selected objective in {path}: {objective}")
    return objective


def run_path(root: Path, decoder: str, seed: int) -> Path:
    return root / "runs" / f"{decoder}__seed{seed}.json"


def model_path(root: Path, decoder: str, seed: int) -> Path:
    return root / "models" / f"{decoder}__seed{seed}.npz"


def _extract_partition(
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
        base.dseed(seed, objective, split, "exp306_loader"),
    )
    device = torch.device(config.device)
    y_parts: list[np.ndarray] = []
    length_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for Xb, yb, lb in loader:
            Xb = Xb.to(device)
            lb_device = lb.to(device)
            spikes = model.layer_features(Xb)["L2"]
            counts = base.fixed_counts(spikes, lb_device, bin_steps)
            y_parts.append(yb.numpy())
            length_parts.append(lb.numpy())
            count_parts.append(counts.cpu().numpy())
    return {
        "y": np.concatenate(y_parts),
        "lengths": np.concatenate(length_parts),
        "counts": np.concatenate(count_parts),
    }


def _valid_bins(lengths: np.ndarray, bin_steps: int, n_bins: int) -> np.ndarray:
    return np.minimum(
        np.maximum(1, np.ceil(lengths.astype(np.float64) / float(bin_steps)).astype(int)),
        n_bins,
    )


def _prefix_feature(counts: np.ndarray, prefix_index: int) -> np.ndarray:
    masked = np.zeros_like(counts)
    masked[:, : prefix_index + 1] = counts[:, : prefix_index + 1]
    return masked.reshape(len(masked), -1)


def _feature_at_prefix(
    counts: np.ndarray,
    decoder: str,
    prefix_index: int,
) -> np.ndarray:
    if decoder == "current250":
        return counts[:, prefix_index]
    if decoder == "cumulative250":
        return counts[:, : prefix_index + 1].sum(axis=1)
    if decoder in ("prefix250_uniform", "prefix250_weighted"):
        return _prefix_feature(counts, prefix_index)
    raise ValueError(f"Unknown decoder: {decoder}")


def _stack_training_rows(
    partition: dict[str, np.ndarray],
    decoder: str,
    bin_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = partition["counts"]
    labels = partition["y"]
    valid_bins = _valid_bins(partition["lengths"], bin_steps, counts.shape[1])
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    weight_rows: list[float] = []

    for sample_index in range(len(labels)):
        B = int(valid_bins[sample_index])
        raw_weights = np.arange(1, B + 1, dtype=np.float64)
        if decoder != "prefix250_weighted":
            raw_weights = np.ones(B, dtype=np.float64)
        normalized = raw_weights / raw_weights.sum()
        one = counts[sample_index : sample_index + 1]
        for prefix_index in range(B):
            X_rows.append(_feature_at_prefix(one, decoder, prefix_index)[0])
            y_rows.append(int(labels[sample_index]))
            weight_rows.append(float(normalized[prefix_index]))

    return (
        np.asarray(X_rows, dtype=np.float64),
        np.asarray(y_rows, dtype=np.int64),
        np.asarray(weight_rows, dtype=np.float64),
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _cohort_curve(
    partition: dict[str, np.ndarray],
    decoder: str,
    bin_steps: int,
    fs: float,
    scaler: StandardScaler,
    classifier: LogisticRegression,
) -> list[dict[str, float | int]]:
    counts = partition["counts"]
    labels = partition["y"]
    valid_bins = _valid_bins(partition["lengths"], bin_steps, counts.shape[1])
    rows: list[dict[str, float | int]] = []

    for prefix_index in range(counts.shape[1]):
        effective_index = np.minimum(prefix_index, valid_bins - 1)
        predictions = np.empty(len(labels), dtype=np.int64)
        for index in range(len(labels)):
            feature = _feature_at_prefix(
                counts[index : index + 1],
                decoder,
                int(effective_index[index]),
            )
            predictions[index] = int(classifier.predict(scaler.transform(feature))[0])
        metric = _metrics(labels, predictions)
        rows.append(
            {
                "checkpoint_index": int(prefix_index),
                "elapsed_ms": float((prefix_index + 1) * bin_steps * 1000.0 / fs),
                "n_active": int(np.sum(valid_bins > prefix_index)),
                **metric,
            }
        )
    return rows


def _curve_auc(curve: list[dict[str, float | int]]) -> float:
    return float(np.mean([float(row["balanced_accuracy"]) for row in curve]))


def _sustained_time(
    curve: list[dict[str, float | int]],
    fraction: float,
) -> float:
    final_ba = float(curve[-1]["balanced_accuracy"])
    threshold = fraction * final_ba
    values = [float(row["balanced_accuracy"]) for row in curve]
    for index in range(len(values)):
        if all(value + 1e-12 >= threshold for value in values[index:]):
            return float(curve[index]["elapsed_ms"])
    return float(curve[-1]["elapsed_ms"])


def _native_timestep_curve(
    repo_root: Path,
    data: base.Data,
    seed: int,
    config: Config,
    split: str,
) -> list[dict[str, float | int]]:
    model, _ = source_backbone.load_model_for_evaluation(
        repo_root,
        source_backbone.results_dir(repo_root),
        SOURCE_WIDTH,
        "timestep_ce",
        seed,
        data,
        torch.device(config.device),
    )
    partition = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }[split]
    loader = base.loader(
        *partition,
        config.batch_size,
        False,
        base.dseed(seed, split, "exp306_native_loader"),
    )
    bin_steps = data.bin_steps
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_lengths: list[np.ndarray] = []
    device = torch.device(config.device)
    with torch.no_grad():
        for Xb, yb, lb in loader:
            spikes = model.layer_features(Xb.to(device))["L2"]
            logits_t = model.head(spikes).cpu().numpy()
            all_logits.append(logits_t)
            all_labels.append(yb.numpy())
            all_lengths.append(lb.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    lengths = np.concatenate(all_lengths)
    n_bins = int(math.ceil(data.T / bin_steps))
    rows: list[dict[str, float | int]] = []
    for bin_index in range(n_bins):
        cutoff = min((bin_index + 1) * bin_steps, data.T)
        effective = np.minimum(lengths, cutoff)
        predictions = np.empty(len(labels), dtype=np.int64)
        for index, length in enumerate(effective.astype(int)):
            predictions[index] = int(logits[index, : max(length, 1)].mean(axis=0).argmax())
        rows.append(
            {
                "checkpoint_index": bin_index,
                "elapsed_ms": float(cutoff * 1000.0 / data.fs),
                "n_active": int(np.sum(lengths > bin_index * bin_steps)),
                **_metrics(labels, predictions),
            }
        )
    return rows


def run_one(decoder: str, seed: int, data: base.Data, config: Config) -> dict[str, object]:
    if decoder not in DECODERS:
        raise ValueError(f"Unknown decoder: {decoder}")
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}")
    out_path = run_path(config.results_dir, decoder, seed)
    if config.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    objective = selected_objective(config.repo_root)
    torch.set_num_threads(config.threads)
    model, _ = source_backbone.load_model_for_evaluation(
        config.repo_root,
        source_backbone.results_dir(config.repo_root),
        SOURCE_WIDTH,
        objective,
        seed,
        data,
        torch.device(config.device),
    )
    partitions_raw = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    partitions = {
        split: _extract_partition(
            model,
            *values,
            split,
            objective,
            seed,
            config,
            data.bin_steps,
        )
        for split, values in partitions_raw.items()
    }

    train_x, train_y, train_w = _stack_training_rows(partitions["train"], decoder, data.bin_steps)
    scaler = StandardScaler().fit(train_x)
    train_z = scaler.transform(train_x)

    best: tuple[float, float, LogisticRegression, list[dict[str, float | int]]] | None = None
    for C in C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=5000,
            solver="lbfgs",
            random_state=base.dseed(seed, objective, decoder, C),
        ).fit(train_z, train_y, sample_weight=train_w)
        val_curve = _cohort_curve(
            partitions["val"], decoder, data.bin_steps, data.fs, scaler, classifier
        )
        val_auc = _curve_auc(val_curve)
        if best is None or val_auc > best[0] + 1e-12:
            best = (val_auc, float(C), classifier, val_curve)
    if best is None:
        raise RuntimeError("No causal decoder candidate selected")

    val_auc, selected_C, classifier, val_curve = best
    test_curve = _cohort_curve(
        partitions["test"], decoder, data.bin_steps, data.fs, scaler, classifier
    )
    result: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "decoder": decoder,
        "seed": int(seed),
        "source_objective": objective,
        "source_width": SOURCE_WIDTH,
        "fixed_bin_ms": CHECKPOINT_MS,
        "fixed_bin_steps": int(data.bin_steps),
        "selected_C": selected_C,
        "val_early_recognition_auc": val_auc,
        "test_early_recognition_auc": _curve_auc(test_curve),
        "test_final_balanced_accuracy": float(test_curve[-1]["balanced_accuracy"]),
        "test_sustained_t90_ms": _sustained_time(test_curve, 0.90),
        "test_sustained_t95_ms": _sustained_time(test_curve, 0.95),
        "val_curve": val_curve,
        "test_curve": test_curve,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    # Persist only numeric parameters, avoiding an opaque pickle artifact.
    mp = model_path(config.results_dir, decoder, seed)
    mp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        mp,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classes=classifier.classes_,
        coef=classifier.coef_,
        intercept=classifier.intercept_,
        selected_C=np.asarray([selected_C], dtype=np.float64),
    )
    return result


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    objective = selected_objective(repo_root)
    data = base.prepare_data(repo_root)
    config = Config(repo_root=repo_root, results_dir=root)

    for decoder, seed in run_specs():
        path = run_path(root, decoder, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing 3.0.6 run artifact: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "decoder": decoder,
                "seed": seed,
                "source_objective": payload["source_objective"],
                "selected_C": payload["selected_C"],
                "val_early_recognition_auc": payload["val_early_recognition_auc"],
                "test_early_recognition_auc": payload["test_early_recognition_auc"],
                "test_final_balanced_accuracy": payload["test_final_balanced_accuracy"],
                "test_sustained_t90_ms": payload["test_sustained_t90_ms"],
                "test_sustained_t95_ms": payload["test_sustained_t95_ms"],
            }
        )
        for split in ("val", "test"):
            for point in payload[f"{split}_curve"]:
                curve_rows.append(
                    {"decoder": decoder, "seed": seed, "split": split, **point}
                )

    # Reused native timestep baseline: one curve per seed, evaluated only here.
    for seed in SEEDS:
        for split in ("val", "test"):
            curve = _native_timestep_curve(repo_root, data, seed, config, split)
            for point in curve:
                curve_rows.append(
                    {"decoder": "native_timestep", "seed": seed, "split": split, **point}
                )
        val_curve = _native_timestep_curve(repo_root, data, seed, config, "val")
        test_curve = _native_timestep_curve(repo_root, data, seed, config, "test")
        rows.append(
            {
                "decoder": "native_timestep",
                "seed": seed,
                "source_objective": "timestep_ce",
                "selected_C": np.nan,
                "val_early_recognition_auc": _curve_auc(val_curve),
                "test_early_recognition_auc": _curve_auc(test_curve),
                "test_final_balanced_accuracy": float(test_curve[-1]["balanced_accuracy"]),
                "test_sustained_t90_ms": _sustained_time(test_curve, 0.90),
                "test_sustained_t95_ms": _sustained_time(test_curve, 0.95),
            }
        )

    runs = pd.DataFrame(rows)
    curves = pd.DataFrame(curve_rows)
    summary = (
        runs.groupby("decoder", sort=True)
        .agg(
            mean_val_early_recognition_auc=("val_early_recognition_auc", "mean"),
            sd_val_early_recognition_auc=("val_early_recognition_auc", "std"),
            mean_test_early_recognition_auc=("test_early_recognition_auc", "mean"),
            sd_test_early_recognition_auc=("test_early_recognition_auc", "std"),
            mean_test_final_ba=("test_final_balanced_accuracy", "mean"),
            sd_test_final_ba=("test_final_balanced_accuracy", "std"),
            mean_t90_ms=("test_sustained_t90_ms", "mean"),
            mean_t95_ms=("test_sustained_t95_ms", "mean"),
        )
        .reset_index()
    )
    causal_only = summary[summary.decoder.isin(DECODERS)].copy()
    causal_only = causal_only.sort_values(
        ["mean_val_early_recognition_auc", "mean_test_final_ba", "decoder"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    if causal_only.empty:
        raise RuntimeError("No causal decoder available for 3.0.6 selection")
    winner = str(causal_only.iloc[0]["decoder"])

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "runs": root / "experiment_3_0_6_runs.csv",
        "curves": root / "experiment_3_0_6_curves.csv",
        "summary": root / "experiment_3_0_6_summary.csv",
        "selection": root / "selected_causal_decoder.json",
    }
    runs.to_csv(outputs["runs"], index=False)
    curves.to_csv(outputs["curves"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    with outputs["selection"].open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "selected_decoder": winner,
                "source_objective": objective,
                "selection_metric": "mean validation Early Recognition AUC",
                "tie_breaker": "mean test final BA only for deterministic ordering; not for model selection",
                "seeds": SEEDS,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 3.0.6 causal temporal decoding")
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
        specs = run_specs()
        task_id = int(args.array_task_id)
        if task_id < 0 or task_id >= len(specs):
            raise ValueError(f"array task id {task_id} outside 0..{len(specs) - 1}")
        decoder, seed = specs[task_id]
        data = base.prepare_data(repo_root)
        result = run_one(
            decoder,
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
            f"completed decoder={decoder} seed={seed} "
            f"val_auc={result['val_early_recognition_auc']:.6f}"
        )
    else:
        for name, path in finalize_experiment(repo_root).items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
