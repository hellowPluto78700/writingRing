from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_0_3_l3_bottleneck_ablation as source_backbone
from scripts import experiment_3_0_4_l2_width_representation_capacity as probe_utils


EXPERIMENT_ID = "experiment_3_0_5_frozen_representation_accessibility"
PROTOCOL_VERSION = "frozen_access_v1"

SOURCE_EXPERIMENT_ID = source_backbone.EXPERIMENT_ID
SOURCE_PROTOCOL_VERSION = source_backbone.PROTOCOL_VERSION
SOURCE_ARCHITECTURE = "B"
SOURCE_WIDTH = 128
SOURCE_SHIFTS = (2, 3, 4)

OBJECTIVES = base.OBJECTIVES
SEEDS = base.SEEDS
PCA_DIM = 128
PROBE_TYPES = (
    "full_count",
    "fixed250_ordered",
    "fixed250_shuffled",
    "fixed250_pca128",
    "relative10_ordered",
    "relative10_shuffled",
    "relative10_pca128",
    "duration_only",
)
SELECTION_PROBES = (
    "full_count",
    "fixed250_pca128",
    "relative10_pca128",
)
EXPECTED_EVAL_RUNS = len(OBJECTIVES) * len(SEEDS)


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


def eval_specs() -> list[tuple[str, int]]:
    return [(objective, seed) for objective in OBJECTIVES for seed in SEEDS]


def evaluation_path(root: Path, objective: str, seed: int) -> Path:
    return root / "evaluations" / f"{objective}__seed{seed}.json"


def selected_backbone_path(root: Path) -> Path:
    return root / "selected_backbone.json"


def _shuffle_fixed_bins(
    counts: np.ndarray,
    lengths: np.ndarray,
    bin_steps: int,
    seed: int,
) -> np.ndarray:
    out = np.array(counts, copy=True)
    rng = np.random.default_rng(seed)
    for index, length in enumerate(lengths.astype(int)):
        n_valid = max(1, int(np.ceil(length / bin_steps)))
        n_valid = min(n_valid, counts.shape[1])
        order = rng.permutation(n_valid)
        out[index, :n_valid] = counts[index, order]
    return out


def _shuffle_relative_bins(counts: np.ndarray, seed: int) -> np.ndarray:
    out = np.array(counts, copy=True)
    rng = np.random.default_rng(seed)
    for index in range(len(out)):
        out[index] = counts[index, rng.permutation(counts.shape[1])]
    return out


def _partition_features(
    model: probe_utils.L2WidthNet,
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    split_name: str,
    objective: str,
    seed: int,
    config: Config,
    fs: float,
    bin_steps: int,
) -> dict[str, np.ndarray]:
    loader = base.loader(
        X,
        y,
        lengths,
        config.batch_size,
        False,
        base.dseed(seed, objective, split_name, "exp305_loader"),
    )
    device = torch.device(config.device)
    y_parts: list[np.ndarray] = []
    length_parts: list[np.ndarray] = []
    full_parts: list[np.ndarray] = []
    fixed_parts: list[np.ndarray] = []
    relative_parts: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for Xb, yb, lb in loader:
            Xb = Xb.to(device)
            lb_device = lb.to(device)
            spikes = model.layer_features(Xb)["L2"]
            full = probe_utils._masked_count(spikes, lb_device)
            fixed = base.fixed_counts(spikes, lb_device, bin_steps)
            relative = base.relative_counts(spikes, lb_device, base.N_REL)

            y_parts.append(yb.numpy())
            length_parts.append(lb.numpy())
            full_parts.append(full.cpu().numpy())
            fixed_parts.append(fixed.cpu().numpy())
            relative_parts.append(relative.cpu().numpy())

    y_all = np.concatenate(y_parts)
    lengths_all = np.concatenate(length_parts)
    full_all = np.concatenate(full_parts)
    fixed_all = np.concatenate(fixed_parts)
    relative_all = np.concatenate(relative_parts)

    fixed_shuffle_seed = base.dseed(seed, objective, split_name, "fixed250_shuffle")
    relative_shuffle_seed = base.dseed(seed, objective, split_name, "relative10_shuffle")
    fixed_shuffled = _shuffle_fixed_bins(
        fixed_all,
        lengths_all,
        bin_steps,
        fixed_shuffle_seed,
    )
    relative_shuffled = _shuffle_relative_bins(relative_all, relative_shuffle_seed)

    return {
        "y": y_all,
        "lengths": lengths_all,
        "full_count": full_all,
        "fixed250_ordered": fixed_all.reshape(len(fixed_all), -1),
        "fixed250_shuffled": fixed_shuffled.reshape(len(fixed_shuffled), -1),
        "relative10_ordered": relative_all.reshape(len(relative_all), -1),
        "relative10_shuffled": relative_shuffled.reshape(len(relative_shuffled), -1),
        "duration_only": (lengths_all.astype(np.float64) / float(fs))[:, None],
    }


def _validate_source_checkpoint(
    payload: dict[str, object],
    objective: str,
    seed: int,
) -> None:
    if payload.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("Unexpected source experiment id")
    if payload.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise ValueError("Unexpected source protocol version")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("Source checkpoint has no result dict")
    if (
        str(result.get("architecture")) != SOURCE_ARCHITECTURE
        or str(result.get("objective")) != objective
        or int(result.get("seed")) != seed
    ):
        raise ValueError("Source checkpoint identity mismatch")


def run_evaluation_one(
    objective: str,
    seed: int,
    data: base.Data,
    config: Config,
) -> dict[str, object]:
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}")

    out_path = evaluation_path(config.results_dir, objective, seed)
    if config.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        expected = (PROTOCOL_VERSION, objective, seed)
        actual = (
            str(cached.get("protocol_version")),
            str(cached.get("objective")),
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
    _validate_source_checkpoint(checkpoint_payload, objective, seed)

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
            objective,
            seed,
            config,
            data.fs,
            data.bin_steps,
        )
        for split, partition in partitions.items()
    }

    probes: list[dict[str, object]] = []
    for probe_type in PROBE_TYPES:
        if probe_type == "fixed250_pca128":
            source_key = "fixed250_ordered"
            metrics = probe_utils._pca_probe_metrics(
                extracted["train"][source_key],
                extracted["train"]["y"],
                extracted["val"][source_key],
                extracted["val"]["y"],
                extracted["test"][source_key],
                extracted["test"]["y"],
                base.dseed(seed, objective, probe_type),
            )
        elif probe_type == "relative10_pca128":
            source_key = "relative10_ordered"
            metrics = probe_utils._pca_probe_metrics(
                extracted["train"][source_key],
                extracted["train"]["y"],
                extracted["val"][source_key],
                extracted["val"]["y"],
                extracted["test"][source_key],
                extracted["test"]["y"],
                base.dseed(seed, objective, probe_type),
            )
        else:
            metrics = probe_utils._raw_probe_metrics(
                extracted["train"][probe_type],
                extracted["train"]["y"],
                extracted["val"][probe_type],
                extracted["val"]["y"],
                extracted["test"][probe_type],
                extracted["test"]["y"],
                base.dseed(seed, objective, probe_type),
            )
        probes.append({"probe_type": probe_type, **metrics})

    source_result = checkpoint_payload["result"]
    native = {
        key: float(source_result[key])
        for key in (
            "train_accuracy",
            "train_balanced_accuracy",
            "train_macro_f1",
            "val_accuracy",
            "val_balanced_accuracy",
            "val_macro_f1",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
        )
        if key in source_result
    }
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_protocol_version": SOURCE_PROTOCOL_VERSION,
        "source_architecture": SOURCE_ARCHITECTURE,
        "source_width": SOURCE_WIDTH,
        "source_shifts": SOURCE_SHIFTS,
        "objective": objective,
        "seed": int(seed),
        "split_seed": base.SPLIT_SEED,
        "sampling_rate_hz": float(data.fs),
        "fixed_bin_ms": float(base.FIXED_MS),
        "fixed_bin_steps": int(data.bin_steps),
        "relative_bins": int(base.N_REL),
        "pca_dim": PCA_DIM,
        "native": native,
        "probes": probes,
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
        for column in value_cols:
            values = group[column].astype(float)
            row[f"mean_{column}"] = float(values.mean())
            row[f"sd_{column}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    probe_rows: list[dict[str, object]] = []
    native_rows: list[dict[str, object]] = []

    for objective, seed in eval_specs():
        path = evaluation_path(root, objective, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing Experiment 3.0.5 evaluation: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = (PROTOCOL_VERSION, objective, seed)
        actual = (
            str(payload.get("protocol_version")),
            str(payload.get("objective")),
            int(payload.get("seed")),
        )
        if actual != expected:
            raise ValueError(f"Evaluation identity mismatch in {path}: {actual} != {expected}")

        for probe in payload["probes"]:
            probe_rows.append({"objective": objective, "seed": seed, **probe})
        native_rows.append({"objective": objective, "seed": seed, **payload["native"]})

    probes = pd.DataFrame(probe_rows)
    native = pd.DataFrame(native_rows)

    derived_rows: list[dict[str, object]] = []
    for (objective, seed), group in probes.groupby(["objective", "seed"], sort=True):
        by_probe = group.set_index("probe_type")
        ordered_fixed = float(by_probe.loc["fixed250_ordered", "probe_test_balanced_accuracy"])
        shuffled_fixed = float(by_probe.loc["fixed250_shuffled", "probe_test_balanced_accuracy"])
        ordered_rel = float(by_probe.loc["relative10_ordered", "probe_test_balanced_accuracy"])
        shuffled_rel = float(by_probe.loc["relative10_shuffled", "probe_test_balanced_accuracy"])
        best_probe_ba = float(group["probe_test_balanced_accuracy"].max())
        native_ba = float(
            native.loc[
                (native.objective == objective) & (native.seed == seed),
                "test_balanced_accuracy",
            ].iloc[0]
        )
        duration_ba = float(by_probe.loc["duration_only", "probe_test_balanced_accuracy"])
        derived_rows.append(
            {
                "objective": objective,
                "seed": int(seed),
                "absolute_time_position_gain_ba": ordered_fixed - shuffled_fixed,
                "relative_phase_position_gain_ba": ordered_rel - shuffled_rel,
                "best_frozen_probe_ba": best_probe_ba,
                "native_test_ba": native_ba,
                "posthoc_linear_accessibility_gap_ba": best_probe_ba - native_ba,
                "duration_only_ba": duration_ba,
            }
        )
    derived = pd.DataFrame(derived_rows)

    selection_source = probes[probes.probe_type.isin(SELECTION_PROBES)].copy()
    selection = _aggregate_mean_sd(
        selection_source,
        ["objective"],
        ["probe_val_balanced_accuracy"],
    ).sort_values(
        ["mean_probe_val_balanced_accuracy", "objective"],
        ascending=[False, True],
        ignore_index=True,
    )
    if selection.empty:
        raise RuntimeError("No objective available for Experiment 3.0.5 selection")
    selected_objective = str(selection.iloc[0]["objective"])

    probe_summary = _aggregate_mean_sd(
        probes,
        ["objective", "probe_type"],
        ["probe_val_balanced_accuracy", "probe_test_balanced_accuracy", "probe_test_macro_f1"],
    )
    derived_summary = _aggregate_mean_sd(
        derived,
        ["objective"],
        [
            "absolute_time_position_gain_ba",
            "relative_phase_position_gain_ba",
            "posthoc_linear_accessibility_gap_ba",
            "duration_only_ba",
        ],
    )

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "probe_results": root / "experiment_3_0_5_probe_results.csv",
        "native_results": root / "experiment_3_0_5_native_results.csv",
        "derived_results": root / "experiment_3_0_5_derived_results.csv",
        "probe_summary": root / "experiment_3_0_5_probe_summary.csv",
        "derived_summary": root / "experiment_3_0_5_derived_summary.csv",
        "objective_selection": root / "experiment_3_0_5_objective_selection.csv",
        "selected_backbone": selected_backbone_path(root),
    }
    probes.to_csv(outputs["probe_results"], index=False)
    native.to_csv(outputs["native_results"], index=False)
    derived.to_csv(outputs["derived_results"], index=False)
    probe_summary.to_csv(outputs["probe_summary"], index=False)
    derived_summary.to_csv(outputs["derived_summary"], index=False)
    selection.to_csv(outputs["objective_selection"], index=False)
    with outputs["selected_backbone"].open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "selected_objective": selected_objective,
                "selection_metric": "mean validation BA across common dimension-matched probes",
                "selection_probes": SELECTION_PROBES,
                "source_experiment_id": SOURCE_EXPERIMENT_ID,
                "source_architecture": SOURCE_ARCHITECTURE,
                "source_width": SOURCE_WIDTH,
                "seeds": SEEDS,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return outputs


def _run_cli(args: argparse.Namespace) -> None:
    repo_root = find_repo_root()
    data = base.prepare_data(repo_root)
    root = results_dir(repo_root)
    if args.command == "run-one":
        specs = eval_specs()
        task_id = int(args.array_task_id)
        if task_id < 0 or task_id >= len(specs):
            raise ValueError(f"array task id {task_id} outside 0..{len(specs) - 1}")
        objective, seed = specs[task_id]
        config = Config(
            repo_root=repo_root,
            results_dir=root,
            device=args.device,
            batch_size=args.batch_size,
            resume=not args.force,
            threads=1,
        )
        result = run_evaluation_one(objective, seed, data, config)
        print(
            f"completed objective={objective} seed={seed} probes={len(result['probes'])}"
        )
    elif args.command == "finalize":
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
    else:
        raise ValueError(args.command)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 3.0.5 frozen representation accessibility")
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
