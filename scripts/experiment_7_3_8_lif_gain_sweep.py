from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from scripts import experiment_7_3_7_wholecount_bias_transfer as exp737
from scripts import experiment_7_3_training_strategy_decomposition as exp73


EXPERIMENT_ID = "experiment_7_3_8_lif_gain_sweep"
PROTOCOL_VERSION = "lif_gain_sweep_v1"
ARCHITECTURE = exp73.ARCHITECTURE
SEEDS = exp73.SEEDS
BACKBONE_OBJECTIVE = "wcce"
GAIN_GRID = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
WEIGHT_SOURCES = ("affine_w", "no_bias_w")
LINEAR_METHODS = (
    "A0_linear_no_bias_W0",
    "A1_linear_affine_W1_b1",
    "A2_linear_hybrid_W0_b1",
    "A3_linear_affine_W1_bias_off",
)
LIF_METHODS = (
    "B0_affineW_lif_selected_gain",
    "B1_affineW_lif_continuous_count_bias",
    "B2_affineW_lif_integer_count_bias",
    "C0_noBiasW_lif_selected_gain",
    "C1_noBiasW_lif_continuous_count_bias",
    "C2_noBiasW_lif_integer_count_bias",
)
METHODS = LINEAR_METHODS + LIF_METHODS


@dataclass(frozen=True)
class RunSpec:
    seed: int

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp73.exp72.BATCH_SIZE
    threads: int = 1
    calibration_max_epochs: int = exp737.CALIBRATION_MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(seed) for seed in SEEDS]


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _exp737_config(config: Config) -> exp737.Config:
    return exp737.Config(
        repo_root=config.repo_root,
        results_dir=exp737.results_dir(config.repo_root),
        device=config.device,
        batch_size=config.batch_size,
        threads=config.threads,
        calibration_max_epochs=config.calibration_max_epochs,
    )


def _load_cache(config: Config, spec: RunSpec) -> dict[str, Any]:
    return exp737._load_cache(_exp737_config(config), exp737.RunSpec(spec.seed))


def _count_diagnostics(counts: np.ndarray, lengths: np.ndarray) -> dict[str, float]:
    total = counts.sum(axis=1)
    active = (counts > 0).sum(axis=1)
    denom = np.maximum(lengths.astype(np.float64), 1.0)
    peak_rate = counts.max(axis=1) / denom
    return {
        "zero_output_fraction": float(np.mean(total == 0)),
        "mean_total_output_spikes": float(np.mean(total)),
        "mean_active_output_classes": float(np.mean(active)),
        "mean_per_class_count_variance": float(np.var(counts, axis=0).mean()),
        "mean_peak_class_spikes_per_timestep": float(np.mean(peak_rate)),
    }


def _linear_references(
    features: dict[str, tuple[np.ndarray, np.ndarray]],
    no_bias: dict[str, Any],
    affine: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    hybrid_metrics = exp737._evaluate_raw_linear(
        features, no_bias["raw_weight"], affine["raw_bias"]
    )
    affine_w_only_metrics = exp737._evaluate_raw_linear(
        features,
        affine["raw_weight"],
        np.zeros_like(affine["raw_bias"]),
    )
    return {
        "A0_linear_no_bias_W0": {
            "metrics": no_bias["metrics"],
            "weight_l2": no_bias["weight_l2"],
            "bias_l2": 0.0,
            "bias_span": 0.0,
            "selected_C": no_bias["selected_C"],
        },
        "A1_linear_affine_W1_b1": {
            "metrics": affine["metrics"],
            "weight_l2": affine["weight_l2"],
            "bias_l2": affine["bias_l2"],
            "bias_span": affine["bias_span"],
            "selected_C": affine["selected_C"],
        },
        "A2_linear_hybrid_W0_b1": {
            "metrics": hybrid_metrics,
            "weight_l2": no_bias["weight_l2"],
            "bias_l2": affine["bias_l2"],
            "bias_span": affine["bias_span"],
            "selected_C": np.nan,
        },
        "A3_linear_affine_W1_bias_off": {
            "metrics": affine_w_only_metrics,
            "weight_l2": affine["weight_l2"],
            "bias_l2": 0.0,
            "bias_span": 0.0,
            "selected_C": np.nan,
        },
    }


def _run_gain_sweep(
    cache: dict[str, Any],
    raw_weight: np.ndarray,
    source: str,
    config: Config,
) -> tuple[list[dict[str, Any]], dict[float, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    device = torch.device(config.device)
    rows: list[dict[str, Any]] = []
    counts_by_gain: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for gain in GAIN_GRID:
        split_counts = {
            split: exp737._lif_counts(
                cache[split],
                gain * raw_weight,
                config.batch_size,
                device,
            )
            for split in ("train", "val", "test")
        }
        counts_by_gain[gain] = split_counts
        for split, (counts, y) in split_counts.items():
            metrics = exp737._metrics(y, counts)
            diagnostics = _count_diagnostics(counts, cache[split][2])
            rows.append(
                {
                    "weight_source": source,
                    "gain": float(gain),
                    "split": split,
                    "accuracy": float(metrics["accuracy"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    **diagnostics,
                }
            )
    return rows, counts_by_gain


def _select_gain(rows: list[dict[str, Any]], source: str) -> float:
    candidates = [
        row for row in rows if row["weight_source"] == source and row["split"] == "val"
    ]
    if not candidates:
        raise RuntimeError(f"No validation gain candidates for {source}")
    best = candidates[0]
    for row in candidates[1:]:
        if row["balanced_accuracy"] > best["balanced_accuracy"] + 1e-12:
            best = row
    return float(best["gain"])


def _evaluate_selected_branch(
    counts: dict[str, tuple[np.ndarray, np.ndarray]],
    calibration: dict[str, Any],
    prefix: str,
) -> dict[str, dict[str, dict[str, float]]]:
    method_names = (
        f"{prefix}0_raw",
        f"{prefix}1_continuous",
        f"{prefix}2_integer",
    )
    result: dict[str, dict[str, dict[str, float]]] = {}
    q_centered = np.asarray(calibration["q_centered"], dtype=np.float64)
    q_integer = np.asarray(calibration["q_integer"], dtype=np.float64)
    for split, (x, y) in counts.items():
        result[split] = {
            method_names[0]: exp737._metrics(y, x),
            method_names[1]: exp737._metrics(y, x + q_centered[None, :]),
            method_names[2]: exp737._metrics(y, x + q_integer[None, :]),
        }
    return result


def _branch_method_names(source: str) -> tuple[str, str, str]:
    if source == "affine_w":
        return LIF_METHODS[0:3]
    if source == "no_bias_w":
        return LIF_METHODS[3:6]
    raise ValueError(source)


def _evaluate_branch_methods(
    counts: dict[str, tuple[np.ndarray, np.ndarray]],
    calibration: dict[str, Any],
    source: str,
) -> dict[str, dict[str, dict[str, float]]]:
    names = _branch_method_names(source)
    q_centered = np.asarray(calibration["q_centered"], dtype=np.float64)
    q_integer = np.asarray(calibration["q_integer"], dtype=np.float64)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for split, (x, y) in counts.items():
        result[split] = {
            names[0]: exp737._metrics(y, x),
            names[1]: exp737._metrics(y, x + q_centered[None, :]),
            names[2]: exp737._metrics(y, x + q_integer[None, :]),
        }
    return result


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    evaluation_path = config.results_dir / "evaluations" / f"{spec.key}.json"
    checkpoint_path = config.results_dir / "checkpoints" / f"{spec.key}.npz"
    if evaluation_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(evaluation_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    cache = _load_cache(config, spec)
    features = {
        split: exp737._wholecount_features(cache[split])
        for split in ("train", "val", "test")
    }
    scaler = StandardScaler(with_mean=False, with_std=True).fit(features["train"][0])
    no_bias = exp737._fit_linear_case(features, scaler, spec.seed, fit_intercept=False)
    affine = exp737._fit_linear_case(features, scaler, spec.seed, fit_intercept=True)
    linear = _linear_references(features, no_bias, affine)

    weight_map = {
        "affine_w": affine["raw_weight"],
        "no_bias_w": no_bias["raw_weight"],
    }
    all_sweep_rows: list[dict[str, Any]] = []
    branch_payload: dict[str, Any] = {}
    checkpoint_arrays: dict[str, np.ndarray] = {
        "linear_no_bias_raw_weight": no_bias["raw_weight"],
        "linear_affine_raw_weight": affine["raw_weight"],
        "linear_affine_raw_bias": affine["raw_bias"],
        "wholecount_scale": np.asarray(scaler.scale_, dtype=np.float64),
    }
    calibration_histories: list[dict[str, Any]] = []

    for source in WEIGHT_SOURCES:
        sweep_rows, counts_by_gain = _run_gain_sweep(
            cache, weight_map[source], source, config
        )
        all_sweep_rows.extend(sweep_rows)
        selected_gain = _select_gain(sweep_rows, source)
        selected_counts = counts_by_gain[selected_gain]
        calibration = exp737._fit_count_bias(
            selected_counts,
            spec.seed,
            device,
            config.calibration_max_epochs,
        )
        metrics = _evaluate_branch_methods(selected_counts, calibration, source)
        for row in calibration["history"]:
            calibration_histories.append({"weight_source": source, **row})
        checkpoint_arrays[f"{source}_selected_gain"] = np.asarray(
            [selected_gain], dtype=np.float64
        )
        checkpoint_arrays[f"{source}_count_bias_centered"] = np.asarray(
            calibration["q_centered"], dtype=np.float64
        )
        checkpoint_arrays[f"{source}_count_bias_nonnegative"] = np.asarray(
            calibration["q_nonnegative"], dtype=np.float64
        )
        checkpoint_arrays[f"{source}_count_bias_integer"] = np.asarray(
            calibration["q_integer"], dtype=np.int64
        )
        branch_payload[source] = {
            "selected_gain": selected_gain,
            "weight_l2_before_gain": float(np.linalg.norm(weight_map[source])),
            "weight_l2_after_gain": float(
                np.linalg.norm(selected_gain * weight_map[source])
            ),
            "metrics": metrics,
            "count_bias": {
                "best_epoch": calibration["best_epoch"],
                "stopped_epoch": calibration["stopped_epoch"],
                "best_val_ba": calibration["best_val_ba"],
                "best_val_loss": calibration["best_val_loss"],
                "ce_only_gain": calibration["gain"],
                "q_centered": np.asarray(calibration["q_centered"]).tolist(),
                "q_nonnegative": np.asarray(calibration["q_nonnegative"]).tolist(),
                "q_integer": np.asarray(calibration["q_integer"]).tolist(),
            },
        }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(checkpoint_path, **checkpoint_arrays)
    sweep_path = config.results_dir / "gain_sweeps" / f"{spec.key}.csv"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_sweep_rows).to_csv(sweep_path, index=False)
    history_path = config.results_dir / "calibration_histories" / f"{spec.key}.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(calibration_histories).to_csv(history_path, index=False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "architecture": ARCHITECTURE,
            "backbone_source": "A2_e2e_linear_wcce",
            "frozen_l1_l2": True,
            "linear_feature": "valid whole-count Z=sum_t z_t",
            "linear_scaler": "train-only per-feature std, no centering; folded into raw W",
            "linear_solver": "same Exp7.3.7 P4/P5-compatible LogisticRegression recipe",
            "hybrid_case": "exact W0 from no-bias Linear plus exact b1 from affine Linear; no retraining or rescaling",
            "gain_grid": list(GAIN_GRID),
            "gain_selection": "highest validation balanced accuracy; smallest gain wins ties; test untouched",
            "lif_beta": exp737.LIF_BETA,
            "threshold": exp737.THRESHOLD,
            "output_cap": exp737.OUTPUT_CAP,
            "lif_input_bias": "none for both weight sources",
            "lif_weight_training": "none; only a positive scalar current gain multiplies frozen Linear W",
            "post_gain_calibration": "fit only 12D count-space q plus CE-only positive scalar gain",
        },
        "linear": linear,
        "branches": branch_payload,
        "gain_sweep": all_sweep_rows,
    }
    _save_json(evaluation_path, payload)
    return payload


def _method_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(payload["spec"]["seed"])
    rows: list[dict[str, Any]] = []
    for method in LINEAR_METHODS:
        result = payload["linear"][method]
        rows.append(
            {
                "method": method,
                "seed": seed,
                "train_ba": float(result["metrics"]["train"]["balanced_accuracy"]),
                "val_ba": float(result["metrics"]["val"]["balanced_accuracy"]),
                "test_ba": float(result["metrics"]["test"]["balanced_accuracy"]),
                "selected_gain": np.nan,
                "weight_l2": float(result["weight_l2"]),
                "bias_l2": float(result["bias_l2"]),
            }
        )
    for source in WEIGHT_SOURCES:
        branch = payload["branches"][source]
        names = _branch_method_names(source)
        for idx, method in enumerate(names):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "train_ba": float(branch["metrics"]["train"][method]["balanced_accuracy"]),
                    "val_ba": float(branch["metrics"]["val"][method]["balanced_accuracy"]),
                    "test_ba": float(branch["metrics"]["test"][method]["balanced_accuracy"]),
                    "selected_gain": float(branch["selected_gain"]),
                    "weight_l2": float(branch["weight_l2_after_gain"]),
                    "bias_l2": (
                        float(np.linalg.norm(branch["count_bias"]["q_centered"]))
                        if idx == 1
                        else 0.0
                    ),
                }
            )
    return rows


def _contrast_row(payload: dict[str, Any]) -> dict[str, Any]:
    seed = int(payload["spec"]["seed"])
    lin = payload["linear"]
    a0 = float(lin["A0_linear_no_bias_W0"]["metrics"]["test"]["balanced_accuracy"])
    a1 = float(lin["A1_linear_affine_W1_b1"]["metrics"]["test"]["balanced_accuracy"])
    a2 = float(lin["A2_linear_hybrid_W0_b1"]["metrics"]["test"]["balanced_accuracy"])
    a3 = float(lin["A3_linear_affine_W1_bias_off"]["metrics"]["test"]["balanced_accuracy"])
    aw = payload["branches"]["affine_w"]
    nw = payload["branches"]["no_bias_w"]
    aw_names = _branch_method_names("affine_w")
    nw_names = _branch_method_names("no_bias_w")
    aw0 = float(aw["metrics"]["test"][aw_names[0]]["balanced_accuracy"])
    aw1 = float(aw["metrics"]["test"][aw_names[1]]["balanced_accuracy"])
    nw0 = float(nw["metrics"]["test"][nw_names[0]]["balanced_accuracy"])
    nw1 = float(nw["metrics"]["test"][nw_names[1]]["balanced_accuracy"])
    return {
        "seed": seed,
        "affine_vs_no_bias_linear_pp": 100.0 * (a1 - a0),
        "hybrid_b1_effect_on_W0_pp": 100.0 * (a2 - a0),
        "same_W1_bias_effect_pp": 100.0 * (a1 - a3),
        "affineW_gain_lif_gap_to_affine_linear_pp": 100.0 * (a1 - aw0),
        "affineW_count_bias_recovery_pp": 100.0 * (aw1 - aw0),
        "noBiasW_gain_lif_gap_to_no_bias_linear_pp": 100.0 * (a0 - nw0),
        "noBiasW_count_bias_recovery_pp": 100.0 * (nw1 - nw0),
        "selected_gain_affineW": float(aw["selected_gain"]),
        "selected_gain_noBiasW": float(nw["selected_gain"]),
    }


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    missing: list[Path] = []
    for spec in run_specs():
        path = config.results_dir / "evaluations" / f"{spec.key}.json"
        if path.exists():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing Exp7.3.8 evaluations:\n" + "\n".join(str(path) for path in missing)
        )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    method_runs = pd.DataFrame(
        [row for payload in payloads for row in _method_rows(payload)]
    ).sort_values(["method", "seed"])
    method_runs.to_csv(config.results_dir / "method_runs.csv", index=False)
    numeric = [c for c in method_runs.columns if c not in {"method", "seed"}]
    method_summary = method_runs.groupby("method", sort=False)[numeric].agg(
        ["mean", "std", "count"]
    )
    method_summary.columns = [f"{name}_{stat}" for name, stat in method_summary.columns]
    method_summary.reset_index().to_csv(
        config.results_dir / "method_summary.csv", index=False
    )

    sweep_frames: list[pd.DataFrame] = []
    for spec in run_specs():
        path = config.results_dir / "gain_sweeps" / f"{spec.key}.csv"
        df = pd.read_csv(path)
        df.insert(0, "seed", spec.seed)
        sweep_frames.append(df)
    sweep_runs = pd.concat(sweep_frames, ignore_index=True)
    sweep_runs.to_csv(config.results_dir / "gain_sweep_runs.csv", index=False)
    sweep_numeric = [
        c
        for c in sweep_runs.columns
        if c not in {"seed", "weight_source", "gain", "split"}
    ]
    sweep_summary = sweep_runs.groupby(
        ["weight_source", "gain", "split"], sort=True
    )[sweep_numeric].agg(["mean", "std", "count"])
    sweep_summary.columns = [f"{name}_{stat}" for name, stat in sweep_summary.columns]
    sweep_summary.reset_index().to_csv(
        config.results_dir / "gain_sweep_summary.csv", index=False
    )

    contrast_runs = pd.DataFrame([_contrast_row(p) for p in payloads]).sort_values("seed")
    contrast_runs.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_summary = contrast_runs.drop(columns=["seed"]).agg(
        ["mean", "std", "count"]
    ).T.reset_index()
    contrast_summary.columns = ["contrast", "mean", "std", "count"]
    contrast_summary.to_csv(config.results_dir / "contrast_summary.csv", index=False)

    bias_rows: list[dict[str, Any]] = []
    for payload in payloads:
        seed = int(payload["spec"]["seed"])
        for source in WEIGHT_SOURCES:
            bias = payload["branches"][source]["count_bias"]
            q = bias["q_centered"]
            q_nonnegative = bias["q_nonnegative"]
            q_integer = bias["q_integer"]
            for class_index in range(len(q)):
                bias_rows.append(
                    {
                        "seed": seed,
                        "weight_source": source,
                        "class_index": class_index,
                        "q_centered": float(q[class_index]),
                        "q_nonnegative": float(q_nonnegative[class_index]),
                        "q_integer": int(q_integer[class_index]),
                    }
                )
    bias_runs = pd.DataFrame(bias_rows).sort_values(
        ["weight_source", "class_index", "seed"]
    )
    bias_runs.to_csv(config.results_dir / "count_bias_runs.csv", index=False)
    bias_summary = bias_runs.groupby(["weight_source", "class_index"])[
        ["q_centered", "q_nonnegative", "q_integer"]
    ].agg(["mean", "std", "count"])
    bias_summary.columns = [f"{name}_{stat}" for name, stat in bias_summary.columns]
    bias_summary.reset_index().to_csv(
        config.results_dir / "count_bias_summary.csv", index=False
    )

    history_frames: list[pd.DataFrame] = []
    for spec in run_specs():
        path = config.results_dir / "calibration_histories" / f"{spec.key}.csv"
        df = pd.read_csv(path)
        df.insert(0, "seed", spec.seed)
        history_frames.append(df)
    history_runs = pd.concat(history_frames, ignore_index=True)
    history_runs.to_csv(config.results_dir / "calibration_history_runs.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "seeds": list(SEEDS),
        "gain_grid": list(GAIN_GRID),
        "weight_sources": list(WEIGHT_SOURCES),
        "methods": list(METHODS),
        "logical_training_runs": len(SEEDS),
        "frozen_backbone": "Exp7.3 A2_e2e_linear_wcce L2 cache",
        "hybrid_control": "W0 from no-bias Linear + b1 from affine Linear, exact reuse with no retraining/rescaling",
        "gain_selection": "per seed and weight source on validation BA; smallest gain wins ties",
        "lif_weight_training": "none",
        "count_bias_training": "after gain selection, only q plus CE-only positive scalar gain",
        "multi_cpu_contract": "one CPU per seed task; each task evaluates both weight sources and full gain sweep; afterok finalizer aggregates only",
        "notebook_contract": "analysis-only; method-level and gain-level aggregation only",
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp7.3.8 LIF current-gain sweep")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp73.exp72.BATCH_SIZE)
    parser.add_argument(
        "--calibration-max-epochs",
        type=int,
        default=exp737.CALIBRATION_MAX_EPOCHS,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    config = Config(
        repo_root=root,
        results_dir=results_dir(root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        calibration_max_epochs=args.calibration_max_epochs,
    )
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
