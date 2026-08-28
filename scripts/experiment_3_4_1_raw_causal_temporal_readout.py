from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts import experiment_3_2_nonlinear_temporal_interaction as exp32


EXPERIMENT_ID = "experiment_3_4_1_raw_causal_temporal_readout"
PROTOCOL_VERSION = "prefix_linear_v1"
REFERENCE_EXPERIMENT = exp32.REFERENCE_EXPERIMENT

SPLIT_SEEDS = exp32.SPLIT_SEEDS
EVENT_CHANNEL_COUNT = exp32.EVENT_CHANNEL_COUNT
EXPECTED_SAMPLING_RATE_HZ = exp32.EXPECTED_SAMPLING_RATE_HZ
GLOBAL_PADDED_LENGTH = exp32.GLOBAL_PADDED_LENGTH
RELATIVE_N_BINS = exp32.RELATIVE_N_BINS
LOGREG_MAX_ITER = exp32.LOGREG_MAX_ITER

METHODS = (
    ("relative10", "whole"),
    ("fixed250", "whole"),
    ("fixed500", "whole"),
    ("fixed250", "prefix"),
    ("fixed500", "prefix"),
)
EXPECTED_RUNS = len(METHODS) * len(SPLIT_SEEDS)
PREFIX_WEIGHT = 0.5
FINAL_WEIGHT = 0.5
WHOLE_PARITY_TOL = 1e-10


@dataclass(frozen=True)
class RunSpec:
    representation: str
    supervision: str
    split_seed: int

    @property
    def key(self) -> str:
        return f"{self.representation}__{self.supervision}__split{self.split_seed}"


def find_repo_root(start: Path | None = None) -> Path:
    return exp32.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(rep, supervision, seed) for rep, supervision in METHODS for seed in SPLIT_SEEDS]


def fit_whole_scaler(train_whole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_whole, dtype=np.float64)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - mean[None, :]) / std[None, :]


def flatten_whole(counts: np.ndarray) -> np.ndarray:
    return np.asarray(counts).reshape(len(counts), -1)


def valid_bin_counts(mask: np.ndarray) -> np.ndarray:
    counts = np.asarray(mask, dtype=bool).sum(axis=1).astype(np.int64)
    if np.any(counts < 1):
        raise ValueError("Every fixed-duration sample must have at least one valid bin")
    return counts


def raw_prefix_feature(counts_one: np.ndarray, prefix_index: int) -> np.ndarray:
    counts = np.asarray(counts_one)
    if counts.ndim != 2:
        raise ValueError("counts_one must be [bins, channels]")
    if prefix_index < 1 or prefix_index > counts.shape[0]:
        raise ValueError("prefix_index outside representation")
    out = np.zeros_like(counts)
    out[:prefix_index] = counts[:prefix_index]
    return out.reshape(-1)


def build_prefix_examples(
    counts: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if counts.shape[:2] != mask.shape:
        raise ValueError("counts/mask shape mismatch")
    k_per_sample = valid_bin_counts(mask)
    features: list[np.ndarray] = []
    targets: list[int] = []
    weights: list[float] = []
    source_rows: list[int] = []
    prefix_indices: list[int] = []
    for row, k_i in enumerate(k_per_sample.tolist()):
        for k in range(1, k_i + 1):
            features.append(raw_prefix_feature(counts[row], k))
            targets.append(int(labels[row]))
            source_rows.append(row)
            prefix_indices.append(k)
            if k_i == 1:
                weights.append(1.0)
            elif k == k_i:
                weights.append(FINAL_WEIGHT)
            else:
                weights.append(PREFIX_WEIGHT / float(k_i - 1))
    return (
        np.stack(features).astype(np.float64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(source_rows, dtype=np.int64),
        np.asarray(prefix_indices, dtype=np.int64),
    )


def fit_linear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", max_iter=LOGREG_MAX_ITER, random_state=seed)
    model.fit(train_x, train_y, sample_weight=sample_weight)
    return model


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return exp32._classification_metrics(y_true, y_pred)


def evaluate_final(
    model: LogisticRegression,
    whole_raw: np.ndarray,
    labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float]:
    x = apply_scaler(flatten_whole(whole_raw), mean, std)
    return _metrics(labels, model.predict(x))


def _endpoint_prefix_raw(counts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    k = valid_bin_counts(mask)
    rows = [raw_prefix_feature(counts[i], int(k[i])) for i in range(len(counts))]
    return np.stack(rows)


def evaluate_prefix_curve(
    model: LogisticRegression,
    counts: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    sampling_rate_hz: float,
    samples_per_bin: int,
    cohort_name: str,
) -> list[dict[str, object]]:
    k_per_sample = valid_bin_counts(mask)
    n_total = len(labels)
    rows: list[dict[str, object]] = []
    endpoint_raw = _endpoint_prefix_raw(counts, mask)
    endpoint_pred = model.predict(apply_scaler(endpoint_raw, mean, std))
    for k in range(1, counts.shape[1] + 1):
        active = k_per_sample >= k
        if np.any(active):
            x_active = np.stack([raw_prefix_feature(counts[i], k) for i in np.flatnonzero(active)])
            pred_active = model.predict(apply_scaler(x_active, mean, std))
            metric_active = _metrics(labels[active], pred_active)
            rows.append({
                "cohort": cohort_name,
                "curve_type": "active",
                "prefix_index": k,
                "time_ms": float(k * samples_per_bin * 1000.0 / sampling_rate_hz),
                "n_samples": int(active.sum()),
                "coverage": float(active.mean()),
                **metric_active,
            })

        system_pred = endpoint_pred.copy()
        for i in np.flatnonzero(active):
            x_i = raw_prefix_feature(counts[i], k)[None, :]
            system_pred[i] = model.predict(apply_scaler(x_i, mean, std))[0]
        metric_system = _metrics(labels, system_pred)
        rows.append({
            "cohort": cohort_name,
            "curve_type": "endpoint_aware",
            "prefix_index": k,
            "time_ms": float(k * samples_per_bin * 1000.0 / sampling_rate_hz),
            "n_samples": int(n_total),
            "coverage": 1.0,
            **metric_system,
        })
    return rows


def _representation_parts(cohort: exp32.Cohort, parts: dict[str, pd.DataFrame], representation: str):
    return {name: exp32.build_representation(cohort, frame, representation) for name, frame in parts.items()}


def _model_path(root: Path, spec: RunSpec) -> Path:
    return root / "models" / f"{spec.key}.npz"


def _artifact_path(root: Path, spec: RunSpec) -> Path:
    return root / "runs" / f"{spec.key}.json"


def save_model(path: Path, model: LogisticRegression, mean: np.ndarray, std: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coef=model.coef_,
        intercept=model.intercept_,
        classes=model.classes_,
        scaler_mean=mean,
        scaler_std=std,
    )


def run_one(spec: RunSpec, cohort: exp32.Cohort, root: Path, force: bool = False) -> dict[str, object]:
    out = _artifact_path(root, spec)
    if out.exists() and not force:
        payload = json.loads(out.read_text(encoding="utf-8"))
        identity = (payload.get("protocol_version"), payload.get("representation"), payload.get("supervision"), payload.get("split_seed"))
        expected = (PROTOCOL_VERSION, spec.representation, spec.supervision, spec.split_seed)
        if identity != expected:
            raise ValueError(f"Cached run identity mismatch: {identity} != {expected}")
        return payload

    parts = exp32.make_user_split(cohort.manifest, spec.split_seed)
    built = _representation_parts(cohort, parts, spec.representation)
    train_counts, train_mask, train_y, meta = built["train"]
    val_counts, val_mask, val_y, val_meta = built["val"]
    test_counts, test_mask, test_y, test_meta = built["test"]
    if meta != val_meta or meta != test_meta:
        raise ValueError("Representation metadata mismatch across splits")

    train_whole = flatten_whole(train_counts)
    val_whole = flatten_whole(val_counts)
    test_whole = flatten_whole(test_counts)
    mean, std = fit_whole_scaler(train_whole)
    linear_seed = exp32.derive_seed(spec.split_seed, exp32._reference_representation_name(spec.representation), "linear")

    n_prefix_rows = len(train_y)
    prefix_weight_sum = float(len(train_y))
    if spec.supervision == "whole":
        model = fit_linear(apply_scaler(train_whole, mean, std), train_y, linear_seed)
    elif spec.supervision == "prefix":
        if spec.representation == "relative10":
            raise ValueError("Relative10 prefix training is not part of Experiment 3.4.1")
        prefix_x_raw, prefix_y, weights, source_rows, prefix_indices = build_prefix_examples(train_counts, train_mask, train_y)
        per_sample = np.bincount(source_rows, weights=weights, minlength=len(train_y))
        if not np.allclose(per_sample, 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError("Prefix sample weights do not sum to one per original gesture")
        model = fit_linear(apply_scaler(prefix_x_raw, mean, std), prefix_y, linear_seed, sample_weight=weights)
        n_prefix_rows = int(len(prefix_y))
        prefix_weight_sum = float(weights.sum())
    else:
        raise ValueError(spec.supervision)

    metrics = {
        "train": evaluate_final(model, train_counts, train_y, mean, std),
        "val": evaluate_final(model, val_counts, val_y, mean, std),
        "test": evaluate_final(model, test_counts, test_y, mean, std),
    }

    curves: list[dict[str, object]] = []
    if spec.representation.startswith("fixed"):
        samples_per_bin = int(meta["samples_per_bin"])
        for cohort_name, (counts, mask, labels) in {
            "train": (train_counts, train_mask, train_y),
            "val": (val_counts, val_mask, val_y),
            "test": (test_counts, test_mask, test_y),
        }.items():
            curves.extend(evaluate_prefix_curve(model, counts, mask, labels, mean, std, cohort.fs, samples_per_bin, cohort_name))

    model_path = _model_path(root, spec)
    save_model(model_path, model, mean, std)
    split_users = exp32._split_users(parts)
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "method": f"{spec.representation}_{spec.supervision}",
        "representation": spec.representation,
        "supervision": spec.supervision,
        "split_seed": spec.split_seed,
        **{k: list(v) for k, v in split_users.items()},
        "sample_hash": exp32._sample_hash(parts),
        "sampling_rate_hz": float(cohort.fs),
        "event_channel_count": EVENT_CHANNEL_COUNT,
        "requested_bin_ms": meta["requested_duration_ms"],
        "samples_per_bin": meta["samples_per_bin"],
        "n_bins": int(train_counts.shape[1]),
        "feature_dim": int(train_whole.shape[1]),
        "solver": "lbfgs",
        "max_iter": LOGREG_MAX_ITER,
        "train_metrics": metrics["train"],
        "val_metrics": metrics["val"],
        "test_metrics": metrics["test"],
        "n_train_original": int(len(train_y)),
        "n_train_prefix_rows": n_prefix_rows,
        "prefix_weight_sum": prefix_weight_sum,
        "model_path": str(model_path.relative_to(root)),
        "prefix_curves": curves,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _load_reference(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "notebooks" / "artifacts" / REFERENCE_EXPERIMENT / "experiment_1_3_3_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing reference baseline: {path}")
    frame = pd.read_csv(path)
    return frame[frame.model == "linear"].copy()


def _validate_payloads(payloads: list[dict[str, object]]) -> None:
    if len(payloads) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} runs, got {len(payloads)}")
    ids = {(p["representation"], p["supervision"], int(p["split_seed"])) for p in payloads}
    if len(ids) != EXPECTED_RUNS:
        raise ValueError("Duplicate run identities")
    for seed in SPLIT_SEEDS:
        group = [p for p in payloads if int(p["split_seed"]) == seed]
        if len(group) != len(METHODS):
            raise ValueError(f"Incomplete paired group for split {seed}")
        hashes = {p["sample_hash"] for p in group}
        users = {(tuple(p["train_users"]), tuple(p["val_users"]), tuple(p["test_users"])) for p in group}
        if len(hashes) != 1 or len(users) != 1:
            raise ValueError(f"Split/sample parity failed for split {seed}")


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    payloads: list[dict[str, object]] = []
    for spec in run_specs():
        path = _artifact_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing Experiment 3.4.1 run artifact: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    _validate_payloads(payloads)

    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for p in payloads:
        row = {
            "representation": p["representation"],
            "supervision": p["supervision"],
            "split_seed": p["split_seed"],
        }
        for cohort in ("train", "val", "test"):
            for metric in ("balanced_accuracy", "accuracy", "macro_f1"):
                row[f"{cohort}_{metric}"] = p[f"{cohort}_metrics"][metric]
        rows.append(row)
        for curve in p["prefix_curves"]:
            curve_rows.append({
                "split_seed": p["split_seed"],
                "representation": p["representation"],
                "training_mode": p["supervision"],
                **curve,
            })
    results = pd.DataFrame(rows)
    curves = pd.DataFrame(curve_rows)

    summary_rows = []
    for (rep, sup), g in results.groupby(["representation", "supervision"], sort=False):
        summary_rows.append({
            "representation": rep,
            "supervision": sup,
            "causal": "No" if rep == "relative10" else ("Yes" if sup == "prefix" else "final-only"),
            "n_splits": len(g),
            "mean_test_ba": float(g.test_balanced_accuracy.mean()),
            "sd_test_ba": float(g.test_balanced_accuracy.std(ddof=1)),
            "mean_val_ba": float(g.val_balanced_accuracy.mean()),
            "mean_test_accuracy": float(g.test_accuracy.mean()),
            "mean_test_macro_f1": float(g.test_macro_f1.mean()),
        })
    summary = pd.DataFrame(summary_rows)

    paired_rows = []
    for seed in SPLIT_SEEDS:
        q = results[results.split_seed == seed].set_index(["representation", "supervision"])
        w250 = float(q.loc[("fixed250", "whole"), "test_balanced_accuracy"])
        p250 = float(q.loc[("fixed250", "prefix"), "test_balanced_accuracy"])
        w500 = float(q.loc[("fixed500", "whole"), "test_balanced_accuracy"])
        p500 = float(q.loc[("fixed500", "prefix"), "test_balanced_accuracy"])
        rel = float(q.loc[("relative10", "whole"), "test_balanced_accuracy"])
        paired_rows.append({
            "split_seed": seed,
            "fixed250_whole": w250,
            "fixed250_prefix": p250,
            "delta250": p250 - w250,
            "fixed500_whole": w500,
            "fixed500_prefix": p500,
            "delta500": p500 - w500,
            "relative10_whole": rel,
            "gap250": rel - p250,
            "gap500": rel - p500,
        })
    paired = pd.DataFrame(paired_rows)

    reference = _load_reference(repo_root)
    parity_rows = []
    for row in results[results.supervision == "whole"].itertuples(index=False):
        ref_name = exp32._reference_representation_name(row.representation)
        match = reference[(reference.representation == ref_name) & (reference.split_seed == row.split_seed)]
        if len(match) != 1:
            raise ValueError(f"Missing reference {ref_name}/split{row.split_seed}")
        ref_ba = float(match.iloc[0].test_balanced_accuracy)
        diff = float(row.test_balanced_accuracy - ref_ba)
        parity_rows.append({"representation": row.representation, "split_seed": row.split_seed, "exp341_test_ba": row.test_balanced_accuracy, "reference_test_ba": ref_ba, "difference": diff, "abs_difference": abs(diff)})
    parity = pd.DataFrame(parity_rows)
    if float(parity.abs_difference.max()) > WHOLE_PARITY_TOL:
        raise ValueError(f"Whole Linear parity drift exceeds {WHOLE_PARITY_TOL}: {parity.abs_difference.max()}")

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results": root / "experiment_3_4_1_results.csv",
        "summary": root / "experiment_3_4_1_summary.csv",
        "prefix_curves": root / "experiment_3_4_1_prefix_curves.csv",
        "paired_deltas": root / "experiment_3_4_1_paired_deltas.csv",
        "baseline_parity": root / "experiment_3_4_1_baseline_parity.csv",
        "provenance": root / "provenance.json",
    }
    results.to_csv(outputs["results"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    curves.to_csv(outputs["prefix_curves"], index=False)
    paired.to_csv(outputs["paired_deltas"], index=False)
    parity.to_csv(outputs["baseline_parity"], index=False)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "split_seeds": list(SPLIT_SEEDS),
        "methods": [list(m) for m in METHODS],
        "expected_runs": EXPECTED_RUNS,
        "sampling_rate_hz": EXPECTED_SAMPLING_RATE_HZ,
        "event_channel_count": EVENT_CHANNEL_COUNT,
        "padded_length": GLOBAL_PADDED_LENGTH,
        "prefix_weight": PREFIX_WEIGHT,
        "final_weight": FINAL_WEIGHT,
        "normalization": "whole-training samples fit one per-feature z-score scaler shared by Whole and Prefix for each fixed duration",
        "prefix_order": "zero future raw bins -> flatten -> apply whole-training scaler",
        "relative10_role": "offline phase-normalized oracle/reference",
        "model_selection": "predefined methods; future Fixed250-vs-Fixed500 choice must use mean validation BA, not test BA",
        "primary_support_rule": "mean delta250 > 0 and at least 4/5 split deltas positive",
    }
    outputs["provenance"].write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 3.4.1 raw causal temporal Linear readout")
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
        print(f"completed {spec.key} test_ba={payload['test_metrics']['balanced_accuracy']:.6f}")
    else:
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
