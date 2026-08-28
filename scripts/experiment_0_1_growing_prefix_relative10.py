from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts import experiment_3_2_nonlinear_temporal_interaction as exp32


EXPERIMENT_ID = "experiment_0_1_growing_prefix_relative10"
PROTOCOL_VERSION = "frozen_full_linear_v1"
REFERENCE_EXPERIMENT = exp32.REFERENCE_EXPERIMENT

SPLIT_SEEDS = exp32.SPLIT_SEEDS
EVENT_CHANNEL_COUNT = exp32.EVENT_CHANNEL_COUNT
EXPECTED_SAMPLING_RATE_HZ = exp32.EXPECTED_SAMPLING_RATE_HZ
GLOBAL_PADDED_LENGTH = exp32.GLOBAL_PADDED_LENGTH
RELATIVE_N_BINS = exp32.RELATIVE_N_BINS
LOGREG_MAX_ITER = exp32.LOGREG_MAX_ITER
PREFIX_STEP_SAMPLES = 10
EXPECTED_RUNS = len(SPLIT_SEEDS)
BASELINE_PARITY_TOL = 1e-10


@dataclass(frozen=True)
class RunSpec:
    split_seed: int

    @property
    def key(self) -> str:
        return f"split{self.split_seed}"


def find_repo_root(start: Path | None = None) -> Path:
    return exp32.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(seed) for seed in SPLIT_SEEDS]


def fit_full_scaler(train_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_full, dtype=np.float64)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - mean[None, :]) / std[None, :]


def relative10_from_sequence(sequence: np.ndarray) -> np.ndarray:
    x = np.asarray(sequence, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != EVENT_CHANNEL_COUNT:
        raise ValueError(f"Expected [T, {EVENT_CHANNEL_COUNT}] sequence")
    if len(x) < RELATIVE_N_BINS:
        raise ValueError("Relative10 requires at least 10 observed samples")
    chunks = np.array_split(x, RELATIVE_N_BINS, axis=0)
    return np.concatenate([chunk.sum(axis=0) for chunk in chunks]).astype(np.float64)


def full_relative10_features(
    cohort: exp32.Cohort,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    counts, _, labels, _ = exp32._relative_representation(cohort, frame, RELATIVE_N_BINS)
    return counts.reshape(len(counts), -1).astype(np.float64), labels


def sample_event_sequence(
    cohort: exp32.Cohort,
    row: object,
    observed_samples: int,
) -> np.ndarray:
    package = cohort.packages[int(getattr(row, "package_index"))]
    valid_length = min(int(getattr(row, "valid_length")), GLOBAL_PADDED_LENGTH)
    if observed_samples > valid_length:
        raise ValueError("observed prefix exceeds the reference valid length")
    return np.asarray(
        package.padded_spike_imu[
            int(getattr(row, "segment_index")), :observed_samples, :EVENT_CHANNEL_COUNT
        ],
        dtype=np.float32,
    )


def growing_prefix_ticks(frame: pd.DataFrame) -> list[int]:
    max_length = min(int(frame.valid_length.max()), GLOBAL_PADDED_LENGTH)
    return list(range(PREFIX_STEP_SAMPLES, max_length + 1, PREFIX_STEP_SAMPLES))


def evaluate_growing_prefix_curve(
    model: LogisticRegression,
    cohort: exp32.Cohort,
    frame: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    cohort_name: str,
) -> list[dict[str, object]]:
    labels = frame.label_idx.to_numpy(dtype=np.int64, copy=True)
    lengths = np.minimum(
        frame.valid_length.to_numpy(dtype=np.int64, copy=True), GLOBAL_PADDED_LENGTH
    )
    rows: list[dict[str, object]] = []
    for observed_samples in growing_prefix_ticks(frame):
        active = lengths >= observed_samples
        active_indices = np.flatnonzero(active)
        if len(active_indices) == 0:
            continue
        features = np.stack(
            [
                relative10_from_sequence(
                    sample_event_sequence(
                        cohort,
                        frame.iloc[int(index)],
                        observed_samples,
                    )
                )
                for index in active_indices
            ]
        )
        pred = model.predict(apply_scaler(features, mean, std))
        metrics = exp32._classification_metrics(labels[active], pred)
        rows.append(
            {
                "cohort": cohort_name,
                "observed_samples": int(observed_samples),
                "observed_time_ms": float(
                    observed_samples * 1000.0 / EXPECTED_SAMPLING_RATE_HZ
                ),
                "samples_per_relative_bin": float(observed_samples / RELATIVE_N_BINS),
                "n_samples": int(active.sum()),
                "coverage": float(active.mean()),
                **metrics,
            }
        )
    return rows


def _artifact_path(root: Path, spec: RunSpec) -> Path:
    return root / "runs" / f"{spec.key}.json"


def _model_path(root: Path, spec: RunSpec) -> Path:
    return root / "models" / f"{spec.key}.npz"


def save_model(
    path: Path,
    model: LogisticRegression,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coef=model.coef_,
        intercept=model.intercept_,
        classes=model.classes_,
        scaler_mean=mean,
        scaler_std=std,
    )


def run_one(
    spec: RunSpec,
    cohort: exp32.Cohort,
    root: Path,
    force: bool = False,
) -> dict[str, object]:
    out = _artifact_path(root, spec)
    if out.exists() and not force:
        payload = json.loads(out.read_text(encoding="utf-8"))
        identity = (payload.get("protocol_version"), int(payload.get("split_seed")))
        expected = (PROTOCOL_VERSION, spec.split_seed)
        if identity != expected:
            raise ValueError(f"Cached run identity mismatch: {identity} != {expected}")
        return payload

    parts = exp32.make_user_split(cohort.manifest, spec.split_seed)
    train_x, train_y = full_relative10_features(cohort, parts["train"])
    val_x, val_y = full_relative10_features(cohort, parts["val"])
    test_x, test_y = full_relative10_features(cohort, parts["test"])

    mean, std = fit_full_scaler(train_x)
    seed = exp32.derive_seed(spec.split_seed, "relative_10bin", "linear")
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=LOGREG_MAX_ITER,
        random_state=seed,
    )
    model.fit(apply_scaler(train_x, mean, std), train_y)

    full_metrics: dict[str, dict[str, float]] = {}
    for name, x, y in (
        ("train", train_x, train_y),
        ("val", val_x, val_y),
        ("test", test_x, test_y),
    ):
        full_metrics[name] = exp32._classification_metrics(
            y, model.predict(apply_scaler(x, mean, std))
        )

    curves: list[dict[str, object]] = []
    for name, frame in parts.items():
        curves.extend(
            evaluate_growing_prefix_curve(model, cohort, frame, mean, std, name)
        )

    model_path = _model_path(root, spec)
    save_model(model_path, model, mean, std)
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "split_seed": spec.split_seed,
        **{k: list(v) for k, v in exp32._split_users(parts).items()},
        "sample_hash": exp32._sample_hash(parts),
        "sampling_rate_hz": float(cohort.fs),
        "event_channel_count": EVENT_CHANNEL_COUNT,
        "relative_bins": RELATIVE_N_BINS,
        "prefix_step_samples": PREFIX_STEP_SAMPLES,
        "prefix_step_ms": float(PREFIX_STEP_SAMPLES * 1000.0 / cohort.fs),
        "feature_dim": int(train_x.shape[1]),
        "solver": "lbfgs",
        "max_iter": LOGREG_MAX_ITER,
        "train_metrics": full_metrics["train"],
        "val_metrics": full_metrics["val"],
        "test_metrics": full_metrics["test"],
        "model_path": str(model_path.relative_to(root)),
        "prefix_curves": curves,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _load_reference(repo_root: Path) -> pd.DataFrame:
    path = (
        repo_root
        / "notebooks"
        / "artifacts"
        / REFERENCE_EXPERIMENT
        / "experiment_1_3_3_results.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing reference baseline: {path}")
    frame = pd.read_csv(path)
    return frame[(frame.model == "linear") & (frame.representation == "relative_10bin")].copy()


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    payloads: list[dict[str, object]] = []
    for spec in run_specs():
        path = _artifact_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing Experiment 0.1 run artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"Protocol mismatch in {path}")
        payloads.append(payload)

    if len(payloads) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} runs, got {len(payloads)}")
    if {int(p["split_seed"]) for p in payloads} != set(SPLIT_SEEDS):
        raise ValueError("Split seed coverage mismatch")

    result_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for payload in payloads:
        row: dict[str, object] = {"split_seed": int(payload["split_seed"])}
        for cohort_name in ("train", "val", "test"):
            metrics = payload[f"{cohort_name}_metrics"]
            for metric_name, value in metrics.items():
                row[f"{cohort_name}_{metric_name}"] = float(value)
        result_rows.append(row)
        for curve in payload["prefix_curves"]:
            curve_rows.append({"split_seed": int(payload["split_seed"]), **curve})

    results = pd.DataFrame(result_rows).sort_values("split_seed").reset_index(drop=True)
    curves = pd.DataFrame(curve_rows).sort_values(
        ["cohort", "observed_samples", "split_seed"]
    ).reset_index(drop=True)

    summary = (
        results[["test_balanced_accuracy", "test_accuracy", "test_macro_f1"]]
        .agg(["mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )
    prefix_summary = (
        curves.groupby(["cohort", "observed_samples", "observed_time_ms"], as_index=False)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            sd_balanced_accuracy=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_coverage=("coverage", "mean"),
            mean_n_samples=("n_samples", "mean"),
        )
    )

    reference = _load_reference(repo_root)
    parity_rows: list[dict[str, object]] = []
    for row in results.itertuples(index=False):
        match = reference[reference.split_seed == row.split_seed]
        if len(match) != 1:
            raise ValueError(f"Missing Relative10 reference for split {row.split_seed}")
        reference_ba = float(match.iloc[0].test_balanced_accuracy)
        difference = float(row.test_balanced_accuracy - reference_ba)
        parity_rows.append(
            {
                "split_seed": int(row.split_seed),
                "exp01_test_ba": float(row.test_balanced_accuracy),
                "reference_test_ba": reference_ba,
                "difference": difference,
                "abs_difference": abs(difference),
            }
        )
    parity = pd.DataFrame(parity_rows)
    if float(parity.abs_difference.max()) > BASELINE_PARITY_TOL:
        raise ValueError(
            f"Full Relative10 parity drift exceeds {BASELINE_PARITY_TOL}: "
            f"{parity.abs_difference.max()}"
        )

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results": root / "experiment_0_1_results.csv",
        "prefix_curves": root / "experiment_0_1_prefix_curves.csv",
        "summary": root / "experiment_0_1_summary.csv",
        "prefix_summary": root / "experiment_0_1_prefix_summary.csv",
        "baseline_parity": root / "experiment_0_1_baseline_parity.csv",
        "provenance": root / "provenance.json",
    }
    results.to_csv(outputs["results"], index=False)
    curves.to_csv(outputs["prefix_curves"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    prefix_summary.to_csv(outputs["prefix_summary"], index=False)
    parity.to_csv(outputs["baseline_parity"], index=False)
    outputs["provenance"].write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "split_seeds": list(SPLIT_SEEDS),
                "expected_runs": EXPECTED_RUNS,
                "sampling_rate_hz": EXPECTED_SAMPLING_RATE_HZ,
                "event_channel_count": EVENT_CHANNEL_COUNT,
                "relative_bins": RELATIVE_N_BINS,
                "prefix_step_samples": PREFIX_STEP_SAMPLES,
                "prefix_step_ms": PREFIX_STEP_SAMPLES * 1000.0 / EXPECTED_SAMPLING_RATE_HZ,
                "training": "Linear and per-feature z-score scaler fit on complete Relative10 training gestures only",
                "evaluation": "At every 10 observed samples, recompute Relative10 over the entire currently observed prefix and apply the frozen full-gesture scaler and Linear model",
                "curve_semantics": "active gestures only; n_samples and coverage are reported at every tick",
                "future_data": "No sample after the current prefix is used to construct the growing-prefix representation",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 0.1 frozen full-Relative10 Linear on growing prefixes"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
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
        if not 0 <= task_id < len(specs):
            raise ValueError(f"array task id {task_id} outside 0..{len(specs)-1}")
        spec = specs[task_id]
        payload = run_one(spec, exp32.load_cohort(repo_root), root, force=args.force)
        print(
            f"completed {spec.key} "
            f"full_test_ba={payload['test_metrics']['balanced_accuracy']:.6f}"
        )
    else:
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
