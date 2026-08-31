from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiment_1_3_10_stacked_bin_snn_ablation import (
    BIN_MS,
    EVENT_CHANNEL_COUNT,
    EXPERIMENT_ID,
    GLOBAL_PADDED_LENGTH,
    PROTOCOL_VERSION,
    SPLIT_SEEDS,
    atomic_csv_dump,
    atomic_json_dump,
    build_global_padded_events,
    classification_metrics,
    derive_seed,
    find_repo_root,
    label_tag,
    load_cohort,
    make_user_split,
    parse_labels,
    results_dir,
    sample_hash,
    split_users,
)

BASELINE_ID = "fixed250_count_linear"
LOGREG_MAX_ITER = 5000


def build_fixed250_count_features(cohort, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | str]]:
    """Match Experiment 1.3.4 count+Linear semantics on the Exp 1.3.10 cohort."""
    bin_steps = max(1, int(np.rint(BIN_MS * cohort.fs / 1000.0)))
    n_bins = int(np.ceil(GLOBAL_PADDED_LENGTH / bin_steps))
    padded_steps = n_bins * bin_steps

    events = build_global_padded_events(cohort, frame)
    if padded_steps > GLOBAL_PADDED_LENGTH:
        events = np.pad(events, ((0, 0), (0, padded_steps - GLOBAL_PADDED_LENGTH), (0, 0)))

    n, _, channels = events.shape
    counts = events.reshape(n, n_bins, bin_steps, channels).sum(axis=2)
    features = counts.reshape(n, n_bins * channels).astype(np.float64, copy=False)
    labels = frame.label_idx.to_numpy(dtype=np.int64, copy=True)
    meta: dict[str, int | float | str] = {
        "name": BASELINE_ID,
        "bin_ms_requested": float(BIN_MS),
        "bin_steps": int(bin_steps),
        "bin_ms_actual": float(1000.0 * bin_steps / cohort.fs),
        "n_bins": int(n_bins),
        "event_channels": int(EVENT_CHANNEL_COUNT),
        "feature_dim": int(n_bins * channels),
        "aggregation": "per-channel sum within each fixed 250 ms bin, then flatten bins in absolute order",
    }
    return features, labels, meta


def _evaluate(model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = model.predict(x)
    return classification_metrics(y, pred)


def baseline_path(repo_root: Path, labels: tuple[str, ...], split_seed: int) -> Path:
    return results_dir(repo_root, labels) / "baselines" / f"{BASELINE_ID}_split{split_seed}.json"


def run_one_split(repo_root: Path, labels: tuple[str, ...], split_seed: int, overwrite: bool = False) -> dict[str, object]:
    output_path = baseline_path(repo_root, labels, split_seed)
    if output_path.exists() and not overwrite:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        print(f"Baseline exists; skipping: {output_path}")
        return payload

    cohort = load_cohort(repo_root, labels)
    parts = make_user_split(cohort.manifest, split_seed)

    x_train, y_train, representation = build_fixed250_count_features(cohort, parts["train"])
    x_val, y_val, _ = build_fixed250_count_features(cohort, parts["val"])
    x_test, y_test, _ = build_fixed250_count_features(cohort, parts["test"])

    # Match Experiment 1.3.4: fit feature standardization on train only.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    model_seed = derive_seed(split_seed, "count_linear")
    model = LogisticRegression(
        max_iter=LOGREG_MAX_ITER,
        random_state=model_seed,
        solver="lbfgs",
    )
    model.fit(x_train_scaled, y_train)

    metrics = {
        "train": _evaluate(model, x_train_scaled, y_train),
        "val": _evaluate(model, x_val_scaled, y_val),
        "test": _evaluate(model, x_test_scaled, y_test),
    }

    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "baseline_id": BASELINE_ID,
        "split_seed": int(split_seed),
        "labels": list(cohort.labels),
        "num_classes": len(cohort.labels),
        "sampling_rate_hz": float(cohort.fs),
        "representation": representation,
        "preprocessing": "StandardScaler fit on training features only",
        "classifier": {
            "type": "sklearn.linear_model.LogisticRegression",
            "solver": "lbfgs",
            "max_iter": LOGREG_MAX_ITER,
            "random_state": int(model_seed),
        },
        "sample_hash": sample_hash(parts),
        **split_users(parts),
        "split_sizes": {name: int(len(frame)) for name, frame in parts.items()},
        "metrics": metrics,
    }
    atomic_json_dump(payload, output_path)
    print(
        f"Fixed250+Linear split={split_seed} | "
        f"val BA={metrics['val']['balanced_accuracy']:.4f} | "
        f"test BA={metrics['test']['balanced_accuracy']:.4f}"
    )
    print(f"Saved: {output_path}")
    return payload


def flatten(payload: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "baseline_id": payload["baseline_id"],
        "split_seed": payload["split_seed"],
        "num_classes": payload["num_classes"],
        "sample_hash": payload["sample_hash"],
    }
    metrics = payload["metrics"]
    for split_name in ("train", "val", "test"):
        for metric in ("balanced_accuracy", "accuracy", "macro_f1"):
            row[f"{split_name}_{metric}"] = metrics[split_name][metric]
    return row


def write_summary(repo_root: Path, labels: tuple[str, ...], payloads: list[dict[str, object]]) -> None:
    root = results_dir(repo_root, labels)
    frame = pd.DataFrame(flatten(payload) for payload in payloads).sort_values("split_seed").reset_index(drop=True)
    atomic_csv_dump(frame, root / "fixed250_linear_runs.csv")

    summary = pd.DataFrame(
        [
            {
                "baseline_id": BASELINE_ID,
                "n_splits": int(frame.split_seed.nunique()),
                "mean_val_ba": float(frame.val_balanced_accuracy.mean()),
                "sd_val_ba": float(frame.val_balanced_accuracy.std(ddof=1)),
                "mean_test_ba": float(frame.test_balanced_accuracy.mean()),
                "sd_test_ba": float(frame.test_balanced_accuracy.std(ddof=1)),
                "mean_test_accuracy": float(frame.test_accuracy.mean()),
                "sd_test_accuracy": float(frame.test_accuracy.std(ddof=1)),
                "mean_test_macro_f1": float(frame.test_macro_f1.mean()),
                "sd_test_macro_f1": float(frame.test_macro_f1.std(ddof=1)),
            }
        ]
    )
    atomic_csv_dump(summary, root / "fixed250_linear_summary.csv")
    print(summary.to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired Fixed250 count + Linear baseline for Experiment 1.3.10")
    parser.add_argument(
        "--labels",
        type=str,
        default="A,B,C,D,E,X,G,H,I,J,K,L",
        help="Comma-separated labels; must match the SNN experiment label set",
    )
    parser.add_argument("--split-seed", type=int, default=None, choices=SPLIT_SEEDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    labels = parse_labels(args.labels)
    repo_root = find_repo_root()

    split_seeds = (args.split_seed,) if args.split_seed is not None else SPLIT_SEEDS
    payloads = [run_one_split(repo_root, labels, split_seed, overwrite=args.overwrite) for split_seed in split_seeds]

    if args.split_seed is None:
        write_summary(repo_root, labels, payloads)


if __name__ == "__main__":
    main()
