from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_0_4_l2_width_representation_capacity as probe_utils
from scripts import experiment_3_0_5_frozen_representation_accessibility as source_selection


EXPERIMENT_ID = "experiment_3_1_raw_vs_snn_representation_value"
PROTOCOL_VERSION = "matched_repr_v1"

SOURCE_EXPERIMENT_ID = source_selection.SOURCE_EXPERIMENT_ID
SOURCE_PROTOCOL_VERSION = source_selection.SOURCE_PROTOCOL_VERSION
SOURCE_ARCHITECTURE = source_selection.SOURCE_ARCHITECTURE
SOURCE_WIDTH = source_selection.SOURCE_WIDTH
SOURCE_SHIFTS = source_selection.SOURCE_SHIFTS

SEEDS = base.SEEDS
PCA_DIM = probe_utils.PCA_DIM
REPRESENTATIONS = ("raw", "snn_l2", "raw_plus_snn")
PROBE_TYPES = (
    "full_count",
    "fixed250_ordered",
    "fixed250_shuffled",
    "fixed250_pca128",
    "relative10_ordered",
    "relative10_shuffled",
    "relative10_pca128",
)
PRIMARY_PROBES = (
    "fixed250_ordered",
    "fixed250_pca128",
    "relative10_ordered",
    "relative10_pca128",
)
EXPECTED_RUNS = len(SEEDS)
CONTROL_SHUFFLE_SEED = 31001
PROBE_RANDOM_SEED = 31002


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = base.BATCH_SIZE
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


def run_specs() -> list[int]:
    return [int(seed) for seed in SEEDS]


def evaluation_path(root: Path, seed: int) -> Path:
    return root / "evaluations" / f"seed{seed}.json"


def selected_backbone_path(repo_root: Path) -> Path:
    return source_selection.selected_backbone_path(source_selection.results_dir(repo_root))


def _validate_selection_payload(payload: dict[str, object]) -> str:
    if payload.get("experiment_id") != source_selection.EXPERIMENT_ID:
        raise ValueError("Experiment 3.1 requires an Experiment 3.0.5 selection artifact")
    if payload.get("protocol_version") != source_selection.PROTOCOL_VERSION:
        raise ValueError("Unexpected Experiment 3.0.5 selection protocol")
    if str(payload.get("source_architecture")) != SOURCE_ARCHITECTURE:
        raise ValueError("Selected backbone architecture does not match Experiment 3.1")
    if int(payload.get("source_width", -1)) != SOURCE_WIDTH:
        raise ValueError("Selected backbone width does not match Experiment 3.1")
    objective = str(payload.get("selected_objective"))
    if objective not in base.OBJECTIVES:
        raise ValueError(f"Unexpected selected objective: {objective}")
    return objective


def selected_source_objective(repo_root: Path) -> str:
    path = selected_backbone_path(repo_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Experiment 3.0.5 selected backbone artifact: {path}. "
            "Run/finalize Experiment 3.0.5 first."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid selection artifact: {path}")
    return _validate_selection_payload(payload)


def _shared_fixed_shuffle(
    raw_counts: np.ndarray,
    snn_counts: np.ndarray,
    lengths: np.ndarray,
    bin_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if raw_counts.shape[:2] != snn_counts.shape[:2]:
        raise ValueError("Raw and SNN fixed-bin cohorts are not aligned")
    raw_out = np.array(raw_counts, copy=True)
    snn_out = np.array(snn_counts, copy=True)
    rng = np.random.default_rng(seed)
    for index, length in enumerate(lengths.astype(int)):
        n_valid = max(1, int(np.ceil(length / bin_steps)))
        n_valid = min(n_valid, raw_counts.shape[1])
        order = rng.permutation(n_valid)
        raw_out[index, :n_valid] = raw_counts[index, order]
        snn_out[index, :n_valid] = snn_counts[index, order]
    return raw_out, snn_out


def _shared_relative_shuffle(
    raw_counts: np.ndarray,
    snn_counts: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if raw_counts.shape[:2] != snn_counts.shape[:2]:
        raise ValueError("Raw and SNN relative-bin cohorts are not aligned")
    raw_out = np.array(raw_counts, copy=True)
    snn_out = np.array(snn_counts, copy=True)
    rng = np.random.default_rng(seed)
    for index in range(len(raw_out)):
        order = rng.permutation(raw_counts.shape[1])
        raw_out[index] = raw_counts[index, order]
        snn_out[index] = snn_counts[index, order]
    return raw_out, snn_out


def _partition_features(
    model: probe_utils.L2WidthNet,
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    split_name: str,
    config: Config,
    bin_steps: int,
) -> dict[str, object]:
    loader = base.loader(
        X,
        y,
        lengths,
        config.batch_size,
        False,
        base.dseed(CONTROL_SHUFFLE_SEED, split_name, "exp31_loader"),
    )
    device = torch.device(config.device)

    y_parts: list[np.ndarray] = []
    length_parts: list[np.ndarray] = []
    raw_full_parts: list[np.ndarray] = []
    raw_fixed_parts: list[np.ndarray] = []
    raw_relative_parts: list[np.ndarray] = []
    snn_full_parts: list[np.ndarray] = []
    snn_fixed_parts: list[np.ndarray] = []
    snn_relative_parts: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for Xb, yb, lb in loader:
            Xb = Xb.to(device)
            lb_device = lb.to(device)
            l2_spikes = model.layer_features(Xb)["L2"]

            raw_full = probe_utils._masked_count(Xb, lb_device)
            raw_fixed = base.fixed_counts(Xb, lb_device, bin_steps)
            raw_relative = base.relative_counts(Xb, lb_device, base.N_REL)
            snn_full = probe_utils._masked_count(l2_spikes, lb_device)
            snn_fixed = base.fixed_counts(l2_spikes, lb_device, bin_steps)
            snn_relative = base.relative_counts(l2_spikes, lb_device, base.N_REL)

            y_parts.append(yb.numpy())
            length_parts.append(lb.numpy())
            raw_full_parts.append(raw_full.cpu().numpy())
            raw_fixed_parts.append(raw_fixed.cpu().numpy())
            raw_relative_parts.append(raw_relative.cpu().numpy())
            snn_full_parts.append(snn_full.cpu().numpy())
            snn_fixed_parts.append(snn_fixed.cpu().numpy())
            snn_relative_parts.append(snn_relative.cpu().numpy())

    y_all = np.concatenate(y_parts)
    lengths_all = np.concatenate(length_parts)
    raw_full = np.concatenate(raw_full_parts)
    raw_fixed = np.concatenate(raw_fixed_parts)
    raw_relative = np.concatenate(raw_relative_parts)
    snn_full = np.concatenate(snn_full_parts)
    snn_fixed = np.concatenate(snn_fixed_parts)
    snn_relative = np.concatenate(snn_relative_parts)

    fixed_seed = base.dseed(CONTROL_SHUFFLE_SEED, split_name, "fixed250")
    relative_seed = base.dseed(CONTROL_SHUFFLE_SEED, split_name, "relative10")
    raw_fixed_shuffled, snn_fixed_shuffled = _shared_fixed_shuffle(
        raw_fixed,
        snn_fixed,
        lengths_all,
        bin_steps,
        fixed_seed,
    )
    raw_relative_shuffled, snn_relative_shuffled = _shared_relative_shuffle(
        raw_relative,
        snn_relative,
        relative_seed,
    )

    raw = {
        "full_count": raw_full,
        "fixed250_ordered": raw_fixed.reshape(len(raw_fixed), -1),
        "fixed250_shuffled": raw_fixed_shuffled.reshape(len(raw_fixed_shuffled), -1),
        "relative10_ordered": raw_relative.reshape(len(raw_relative), -1),
        "relative10_shuffled": raw_relative_shuffled.reshape(len(raw_relative_shuffled), -1),
    }
    snn_l2 = {
        "full_count": snn_full,
        "fixed250_ordered": snn_fixed.reshape(len(snn_fixed), -1),
        "fixed250_shuffled": snn_fixed_shuffled.reshape(len(snn_fixed_shuffled), -1),
        "relative10_ordered": snn_relative.reshape(len(snn_relative), -1),
        "relative10_shuffled": snn_relative_shuffled.reshape(len(snn_relative_shuffled), -1),
    }
    return {
        "y": y_all,
        "lengths": lengths_all,
        "raw": raw,
        "snn_l2": snn_l2,
    }


def _source_key(probe_type: str) -> str:
    if probe_type == "fixed250_pca128":
        return "fixed250_ordered"
    if probe_type == "relative10_pca128":
        return "relative10_ordered"
    return probe_type


def _compose_features(
    partition: dict[str, object],
    representation: str,
    probe_type: str,
) -> np.ndarray:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    key = _source_key(probe_type)
    raw = partition["raw"]
    snn_l2 = partition["snn_l2"]
    if not isinstance(raw, dict) or not isinstance(snn_l2, dict):
        raise TypeError("Malformed partition feature dictionaries")
    raw_x = np.asarray(raw[key])
    snn_x = np.asarray(snn_l2[key])
    if len(raw_x) != len(snn_x):
        raise ValueError("Raw and SNN cohorts have different sample counts")
    if representation == "raw":
        return raw_x
    if representation == "snn_l2":
        return snn_x
    return np.concatenate([raw_x, snn_x], axis=1)


def _probe_seed(representation: str, probe_type: str) -> int:
    return base.dseed(PROBE_RANDOM_SEED, representation, probe_type)


def run_evaluation_one(
    seed: int,
    data: base.Data,
    config: Config,
) -> dict[str, object]:
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}")

    objective = selected_source_objective(config.repo_root)
    out_path = evaluation_path(config.results_dir, seed)
    if config.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        expected = (PROTOCOL_VERSION, objective, seed)
        actual = (
            str(cached.get("protocol_version")),
            str(cached.get("source_objective")),
            int(cached.get("seed")),
        )
        if actual != expected:
            raise ValueError(f"Evaluation identity mismatch in {out_path}: {actual} != {expected}")
        return cached

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model, checkpoint_payload = probe_utils.load_model_for_evaluation(
        config.repo_root,
        probe_utils.results_dir(config.repo_root),
        SOURCE_WIDTH,
        objective,
        seed,
        data,
        device,
    )
    source_selection._validate_source_checkpoint(checkpoint_payload, objective, seed)

    partitions = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    extracted = {
        split: _partition_features(
            model,
            *partition,
            split,
            config,
            data.bin_steps,
        )
        for split, partition in partitions.items()
    }

    probe_rows: list[dict[str, object]] = []
    for representation in REPRESENTATIONS:
        for probe_type in PROBE_TYPES:
            train_x = _compose_features(extracted["train"], representation, probe_type)
            val_x = _compose_features(extracted["val"], representation, probe_type)
            test_x = _compose_features(extracted["test"], representation, probe_type)
            if probe_type.endswith("_pca128"):
                metrics = probe_utils._pca_probe_metrics(
                    train_x,
                    np.asarray(extracted["train"]["y"]),
                    val_x,
                    np.asarray(extracted["val"]["y"]),
                    test_x,
                    np.asarray(extracted["test"]["y"]),
                    _probe_seed(representation, probe_type),
                )
            else:
                metrics = probe_utils._raw_probe_metrics(
                    train_x,
                    np.asarray(extracted["train"]["y"]),
                    val_x,
                    np.asarray(extracted["val"]["y"]),
                    test_x,
                    np.asarray(extracted["test"]["y"]),
                    _probe_seed(representation, probe_type),
                )
            probe_rows.append(
                {
                    "representation": representation,
                    "probe_type": probe_type,
                    **metrics,
                }
            )

    duration_metrics = probe_utils._raw_probe_metrics(
        (np.asarray(extracted["train"]["lengths"], dtype=np.float64) / data.fs)[:, None],
        np.asarray(extracted["train"]["y"]),
        (np.asarray(extracted["val"]["lengths"], dtype=np.float64) / data.fs)[:, None],
        np.asarray(extracted["val"]["y"]),
        (np.asarray(extracted["test"]["lengths"], dtype=np.float64) / data.fs)[:, None],
        np.asarray(extracted["test"]["y"]),
        _probe_seed("duration_only", "duration_only"),
    )
    probe_rows.append(
        {
            "representation": "duration_only",
            "probe_type": "duration_only",
            **duration_metrics,
        }
    )

    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_selection_experiment_id": source_selection.EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_protocol_version": SOURCE_PROTOCOL_VERSION,
        "source_architecture": SOURCE_ARCHITECTURE,
        "source_width": SOURCE_WIDTH,
        "source_shifts": SOURCE_SHIFTS,
        "source_objective": objective,
        "seed": int(seed),
        "split_seed": int(base.SPLIT_SEED),
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
        "sampling_rate_hz": float(data.fs),
        "event_channels": int(base.EVENT_CHANNELS),
        "fixed_bin_ms": float(base.FIXED_MS),
        "fixed_bin_steps": int(data.bin_steps),
        "relative_bins": int(base.N_REL),
        "pca_dim": int(PCA_DIM),
        "control_shuffle_seed": int(CONTROL_SHUFFLE_SEED),
        "probe_random_seed": int(PROBE_RANDOM_SEED),
        "probes": probe_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def _aggregate_mean_sd(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=True))
        row["n_runs"] = int(len(group))
        for column in value_cols:
            values = group[column].astype(float)
            row[f"mean_{column}"] = float(values.mean())
            row[f"sd_{column}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _assert_repeated_baseline_is_identical(frame: pd.DataFrame) -> None:
    baseline = frame[frame.representation.isin(("raw", "duration_only"))]
    numeric_cols = (
        "input_feature_dim",
        "feature_dim",
        "probe_C",
        "probe_val_balanced_accuracy",
        "probe_test_balanced_accuracy",
        "probe_test_macro_f1",
        "probe_test_accuracy",
    )
    for (representation, probe_type), group in baseline.groupby(
        ["representation", "probe_type"], sort=True
    ):
        if len(group) != EXPECTED_RUNS:
            raise ValueError(
                f"Expected {EXPECTED_RUNS} repeated {representation}/{probe_type} baselines, "
                f"got {len(group)}"
            )
        first = group.iloc[0]
        for column in numeric_cols:
            values = group[column].astype(float).to_numpy()
            if not np.allclose(values, float(first[column]), rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"Repeated baseline mismatch for {representation}/{probe_type}/{column}: "
                    f"{values.tolist()}"
                )


def _paired_delta_rows(probes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        seed_frame = probes[probes.seed == seed]
        indexed = seed_frame.set_index(["representation", "probe_type"])
        for probe_type in PROBE_TYPES:
            raw = indexed.loc[("raw", probe_type)]
            snn = indexed.loc[("snn_l2", probe_type)]
            fusion = indexed.loc[("raw_plus_snn", probe_type)]
            rows.append(
                {
                    "seed": int(seed),
                    "probe_type": probe_type,
                    "raw_test_ba": float(raw["probe_test_balanced_accuracy"]),
                    "snn_test_ba": float(snn["probe_test_balanced_accuracy"]),
                    "fusion_test_ba": float(fusion["probe_test_balanced_accuracy"]),
                    "snn_minus_raw_test_ba": float(
                        snn["probe_test_balanced_accuracy"]
                        - raw["probe_test_balanced_accuracy"]
                    ),
                    "fusion_minus_raw_test_ba": float(
                        fusion["probe_test_balanced_accuracy"]
                        - raw["probe_test_balanced_accuracy"]
                    ),
                    "fusion_minus_snn_test_ba": float(
                        fusion["probe_test_balanced_accuracy"]
                        - snn["probe_test_balanced_accuracy"]
                    ),
                    "snn_minus_raw_val_ba": float(
                        snn["probe_val_balanced_accuracy"]
                        - raw["probe_val_balanced_accuracy"]
                    ),
                    "fusion_minus_raw_val_ba": float(
                        fusion["probe_val_balanced_accuracy"]
                        - raw["probe_val_balanced_accuracy"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _deduplicate_baselines(probes: pd.DataFrame) -> pd.DataFrame:
    repeated = probes[~probes.representation.isin(("raw", "duration_only"))]
    baseline = (
        probes[probes.representation.isin(("raw", "duration_only"))]
        .sort_values(["representation", "probe_type", "seed"])
        .drop_duplicates(["representation", "probe_type"], keep="first")
    )
    return pd.concat([baseline, repeated], ignore_index=True)


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    source_objectives: set[str] = set()

    for seed in SEEDS:
        path = evaluation_path(root, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing Experiment 3.1 evaluation: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = (PROTOCOL_VERSION, seed)
        actual = (str(payload.get("protocol_version")), int(payload.get("seed")))
        if actual != expected:
            raise ValueError(f"Evaluation identity mismatch in {path}: {actual} != {expected}")
        source_objectives.add(str(payload.get("source_objective")))
        for probe in payload["probes"]:
            rows.append({"seed": int(seed), **probe})

    if len(source_objectives) != 1:
        raise ValueError(f"Experiment 3.1 mixed source objectives: {sorted(source_objectives)}")
    source_objective = next(iter(source_objectives))

    probes = pd.DataFrame(rows)
    _assert_repeated_baseline_is_identical(probes)
    deltas = _paired_delta_rows(probes)

    summary_input = _deduplicate_baselines(probes)
    probe_summary = _aggregate_mean_sd(
        summary_input,
        ["representation", "probe_type"],
        [
            "probe_val_balanced_accuracy",
            "probe_test_balanced_accuracy",
            "probe_test_macro_f1",
            "probe_test_accuracy",
        ],
    )
    delta_summary = _aggregate_mean_sd(
        deltas,
        ["probe_type"],
        [
            "snn_minus_raw_test_ba",
            "fusion_minus_raw_test_ba",
            "fusion_minus_snn_test_ba",
            "snn_minus_raw_val_ba",
            "fusion_minus_raw_val_ba",
        ],
    )

    primary_summary = probe_summary[probe_summary.probe_type.isin(PRIMARY_PROBES)].copy()
    primary = primary_summary.pivot(
        index="probe_type",
        columns="representation",
        values="mean_probe_test_balanced_accuracy",
    ).reset_index()
    primary = primary.rename(
        columns={
            "raw": "raw_mean_test_ba",
            "snn_l2": "snn_mean_test_ba",
            "raw_plus_snn": "fusion_mean_test_ba",
        }
    )
    primary = primary.merge(
        delta_summary[
            [
                "probe_type",
                "mean_snn_minus_raw_test_ba",
                "sd_snn_minus_raw_test_ba",
                "mean_fusion_minus_raw_test_ba",
                "sd_fusion_minus_raw_test_ba",
                "mean_fusion_minus_snn_test_ba",
                "sd_fusion_minus_snn_test_ba",
            ]
        ],
        on="probe_type",
        how="left",
    ).sort_values("probe_type", ignore_index=True)

    def primary_row(probe_type: str) -> dict[str, object]:
        row = primary[primary.probe_type == probe_type]
        if len(row) != 1:
            raise RuntimeError(f"Missing primary comparison for {probe_type}")
        record = row.iloc[0]
        return {
            key: (float(value) if key != "probe_type" else str(value))
            for key, value in record.to_dict().items()
        }

    conclusion = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_objective": source_objective,
        "split_seed": int(base.SPLIT_SEED),
        "primary_question": (
            "Does the frozen SNN add discriminative value beyond matched raw 30-channel "
            "event representations under the same split and linear-probe protocol?"
        ),
        "fixed250_ordered": primary_row("fixed250_ordered"),
        "fixed250_pca128": primary_row("fixed250_pca128"),
        "relative10_ordered": primary_row("relative10_ordered"),
        "relative10_pca128": primary_row("relative10_pca128"),
    }

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "probe_results": root / "experiment_3_1_probe_results.csv",
        "probe_summary": root / "experiment_3_1_probe_summary.csv",
        "paired_deltas": root / "experiment_3_1_paired_deltas.csv",
        "delta_summary": root / "experiment_3_1_delta_summary.csv",
        "primary_comparison": root / "experiment_3_1_primary_comparison.csv",
        "conclusion": root / "experiment_3_1_conclusion.json",
    }
    probes.to_csv(outputs["probe_results"], index=False)
    probe_summary.to_csv(outputs["probe_summary"], index=False)
    deltas.to_csv(outputs["paired_deltas"], index=False)
    delta_summary.to_csv(outputs["delta_summary"], index=False)
    primary.to_csv(outputs["primary_comparison"], index=False)
    with outputs["conclusion"].open("w", encoding="utf-8") as handle:
        json.dump(conclusion, handle, indent=2, sort_keys=True)
    return outputs


def _run_cli(args: argparse.Namespace) -> None:
    repo_root = find_repo_root()
    root = results_dir(repo_root)
    if args.command == "run-one":
        specs = run_specs()
        task_id = int(args.array_task_id)
        if task_id < 0 or task_id >= len(specs):
            raise ValueError(f"array task id {task_id} outside 0..{len(specs) - 1}")
        seed = specs[task_id]
        data = base.prepare_data(repo_root)
        config = Config(
            repo_root=repo_root,
            results_dir=root,
            device=args.device,
            batch_size=args.batch_size,
            resume=not args.force,
            threads=1,
        )
        result = run_evaluation_one(seed, data, config)
        print(
            f"completed seed={seed} source_objective={result['source_objective']} "
            f"probes={len(result['probes'])}"
        )
    elif args.command == "finalize":
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
    else:
        raise ValueError(args.command)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 3.1 matched raw vs frozen-SNN representation value"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--batch-size", type=int, default=base.BATCH_SIZE)
    run.add_argument("--force", action="store_true")
    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    _run_cli(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
