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

from scripts import experiment_3_0_2_hidden_multitau_architectures as exp302
from scripts import experiment_7_3_training_strategy_decomposition as exp73


EXPERIMENT_ID = "experiment_7_3_2_affine_probe_bridge"
PROTOCOL_VERSION = "affine_probe_bridge_v1"
ARCHITECTURE = exp73.ARCHITECTURE
SEEDS = exp73.SEEDS
BACKBONE_OBJECTIVES = ("tsce", "wcce")
MAX_ITER = 5000
C_GRID = tuple(float(v) for v in exp302.PROBE_C_GRID)


@dataclass(frozen=True)
class CaseDef:
    key: str
    aggregation: str
    center: bool
    bias: bool


CASES = (
    CaseDef("P0_mean_scale_no_center_no_bias", "mean", False, False),
    CaseDef("P1_mean_scale_no_center_bias", "mean", False, True),
    CaseDef("P2_mean_scale_center_no_bias", "mean", True, False),
    CaseDef("P3_mean_scale_center_bias", "mean", True, True),
    CaseDef("P4_wholecount_scale_no_center_no_bias", "wholecount", False, False),
    CaseDef("P5_wholecount_scale_no_center_bias", "wholecount", False, True),
    CaseDef("P6_wholecount_scale_center_no_bias", "wholecount", True, False),
    CaseDef("P7_wholecount_scale_center_bias", "wholecount", True, True),
)
CASE_BY_KEY = {case.key: case for case in CASES}


@dataclass(frozen=True)
class RunSpec:
    seed: int
    backbone_objective: str
    case_key: str

    @property
    def case(self) -> CaseDef:
        return CASE_BY_KEY[self.case_key]

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__backbone_{self.backbone_objective}__seed{self.seed}__{self.case_key}"


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
        RunSpec(seed, backbone_objective, case.key)
        for seed in SEEDS
        for backbone_objective in BACKBONE_OBJECTIVES
        for case in CASES
    ]


def validate_spec(spec: RunSpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    if spec.backbone_objective not in BACKBONE_OBJECTIVES:
        raise ValueError(spec.backbone_objective)
    if spec.case_key not in CASE_BY_KEY:
        raise ValueError(spec.case_key)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _exp73_cache_paths(repo_root: Path, spec: RunSpec) -> tuple[Path, Path]:
    backbone = exp73.BackboneSpec(spec.seed, spec.backbone_objective)
    root = exp73.results_dir(repo_root) / "stage2_l2_cache"
    return root / f"{backbone.key}.npz", root / f"{backbone.key}.json"


def _load_cache(repo_root: Path, spec: RunSpec) -> dict[str, Any]:
    npz_path, meta_path = _exp73_cache_paths(repo_root, spec)
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Exp7.3 frozen-L2 cache missing for {spec.key}: {npz_path} / {meta_path}"
        )
    with np.load(npz_path, allow_pickle=False) as z:
        result: dict[str, Any] = {
            split: (
                z[f"{split}_l2"].copy(),
                z[f"{split}_y"].copy(),
                z[f"{split}_lengths"].copy(),
            )
            for split in ("train", "val", "test")
        }
    result["metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return result


def _valid_mask(lengths: np.ndarray, n_steps: int) -> np.ndarray:
    return np.arange(n_steps, dtype=np.int64)[None, :] < lengths[:, None]


def _aggregate_features(
    split: tuple[np.ndarray, np.ndarray, np.ndarray], aggregation: str
) -> tuple[np.ndarray, np.ndarray]:
    l2, y, lengths = split
    x = l2.astype(np.float64, copy=False)
    mask = _valid_mask(lengths, x.shape[1]).astype(np.float64)[..., None]
    summed = (x * mask).sum(axis=1)
    if aggregation == "wholecount":
        features = summed
    elif aggregation == "mean":
        features = summed / np.maximum(lengths, 1).astype(np.float64)[:, None]
    else:
        raise ValueError(aggregation)
    return features, y.astype(np.int64, copy=False)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _classifier_seed(spec: RunSpec) -> int:
    if spec.case_key == "P7_wholecount_scale_center_bias":
        return int(exp73._probe_seed(spec.seed, "l2_wholecount_linear"))
    return int(exp73._probe_seed(spec.seed, f"exp7_3_2_{spec.case_key}"))


def _effective_affine(
    scaler: StandardScaler, classifier: LogisticRegression
) -> dict[str, Any]:
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    coef = np.asarray(classifier.coef_, dtype=np.float64)
    raw_weight = coef / scale[None, :]
    explicit_intercept = np.asarray(classifier.intercept_, dtype=np.float64)
    if scaler.with_mean:
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        centering_offset = -(coef * (mean / scale)[None, :]).sum(axis=1)
    else:
        centering_offset = np.zeros(coef.shape[0], dtype=np.float64)
    effective_intercept = explicit_intercept + centering_offset
    return {
        "weight_l2": float(np.linalg.norm(raw_weight)),
        "explicit_intercept_l2": float(np.linalg.norm(explicit_intercept)),
        "centering_offset_l2": float(np.linalg.norm(centering_offset)),
        "effective_intercept_l2": float(np.linalg.norm(effective_intercept)),
        "effective_intercept_span": float(effective_intercept.max() - effective_intercept.min()),
    }


def _fit_case(spec: RunSpec, cache: dict[str, Any]) -> dict[str, Any]:
    case = spec.case
    features = {
        split: _aggregate_features(cache[split], case.aggregation)
        for split in ("train", "val", "test")
    }
    scaler = StandardScaler(with_mean=case.center, with_std=True).fit(features["train"][0])
    transformed = {
        split: (scaler.transform(x), y)
        for split, (x, y) in features.items()
    }

    seed = _classifier_seed(spec)
    candidates: list[dict[str, Any]] = []
    best: tuple[float, float, LogisticRegression] | None = None
    for C in C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=MAX_ITER,
            solver="lbfgs",
            random_state=seed,
            fit_intercept=case.bias,
        ).fit(transformed["train"][0], transformed["train"][1])
        val_pred = classifier.predict(transformed["val"][0])
        val_ba = float(balanced_accuracy_score(transformed["val"][1], val_pred))
        candidates.append(
            {
                "C": float(C),
                "val_balanced_accuracy": val_ba,
                "n_iter_max": int(np.asarray(classifier.n_iter_).max()),
            }
        )
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError(f"No LogisticRegression candidate selected for {spec.key}")

    selected_val_ba, selected_C, classifier = best
    split_metrics = {
        split: _metrics(y, classifier.predict(x))
        for split, (x, y) in transformed.items()
    }
    source = cache["metadata"]
    legacy = source["representation_probes"]["l2_wholecount_linear"]
    legacy_test_ba = float(legacy["metrics"]["test"]["balanced_accuracy"])
    legacy_C = float(legacy["probe_C"])
    test_ba = float(split_metrics["test"]["balanced_accuracy"])
    affine = _effective_affine(scaler, classifier)

    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "case": asdict(case),
        "contract": {
            "architecture": ARCHITECTURE,
            "frozen_l1_l2": True,
            "source_exp73_method": source["source_method"],
            "source_exp73_best_epoch": source["source_best_epoch"],
            "feature_aggregation": case.aggregation,
            "scaler_with_std": True,
            "scaler_with_mean": case.center,
            "fit_intercept": case.bias,
            "solver": "lbfgs",
            "max_iter": MAX_ITER,
            "C_grid": list(C_GRID),
            "selection": "highest validation balanced accuracy; first C in grid wins ties",
            "test_untouched_until_reporting": True,
        },
        "selected_C": selected_C,
        "selected_val_balanced_accuracy": selected_val_ba,
        "candidate_validation": candidates,
        "metrics": split_metrics,
        "affine": affine,
        "scaler": {
            "scale_min": float(np.min(scaler.scale_)),
            "scale_max": float(np.max(scaler.scale_)),
            "mean_l2": float(np.linalg.norm(scaler.mean_)) if case.center else 0.0,
        },
        "legacy_l2_wholecount_probe": {
            "probe_C": legacy_C,
            "test_balanced_accuracy": legacy_test_ba,
        },
        "test_ba_minus_legacy_pp": 100.0 * (test_ba - legacy_test_ba),
        "p7_reproduction_check": case.key == "P7_wholecount_scale_center_bias",
        "p7_abs_test_ba_delta": (
            abs(test_ba - legacy_test_ba)
            if case.key == "P7_wholecount_scale_center_bias"
            else None
        ),
    }


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    path = config.results_dir / "evaluations" / f"{spec.key}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    cache = _load_cache(config.repo_root, spec)
    payload = _fit_case(spec, cache)
    _save_json(path, payload)
    return payload


def _row(payload: dict[str, Any]) -> dict[str, Any]:
    spec = payload["spec"]
    case = payload["case"]
    return {
        "seed": int(spec["seed"]),
        "backbone_objective": spec["backbone_objective"],
        "case": case["key"],
        "aggregation": case["aggregation"],
        "center": bool(case["center"]),
        "bias": bool(case["bias"]),
        "selected_C": float(payload["selected_C"]),
        "train_ba": float(payload["metrics"]["train"]["balanced_accuracy"]),
        "val_ba": float(payload["metrics"]["val"]["balanced_accuracy"]),
        "test_ba": float(payload["metrics"]["test"]["balanced_accuracy"]),
        "legacy_l2_wholecount_probe_test_ba": float(
            payload["legacy_l2_wholecount_probe"]["test_balanced_accuracy"]
        ),
        "test_ba_minus_legacy_pp": float(payload["test_ba_minus_legacy_pp"]),
        "weight_l2": float(payload["affine"]["weight_l2"]),
        "explicit_intercept_l2": float(payload["affine"]["explicit_intercept_l2"]),
        "centering_offset_l2": float(payload["affine"]["centering_offset_l2"]),
        "effective_intercept_l2": float(payload["affine"]["effective_intercept_l2"]),
        "effective_intercept_span": float(payload["affine"]["effective_intercept_span"]),
        "p7_abs_test_ba_delta": payload["p7_abs_test_ba_delta"],
    }


def _paired_contrast(
    runs: pd.DataFrame, name: str, left: str, right: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backbone in BACKBONE_OBJECTIVES:
        left_df = runs[(runs.backbone_objective == backbone) & (runs.case == left)]
        right_df = runs[(runs.backbone_objective == backbone) & (runs.case == right)]
        merged = left_df[["seed", "test_ba"]].merge(
            right_df[["seed", "test_ba"]], on="seed", suffixes=("_left", "_right")
        )
        for _, row in merged.iterrows():
            rows.append(
                {
                    "contrast": name,
                    "backbone_objective": backbone,
                    "seed": int(row["seed"]),
                    "left_case": left,
                    "right_case": right,
                    "delta_pp": 100.0 * (float(row["test_ba_left"]) - float(row["test_ba_right"])),
                }
            )
    return rows


def finalize(config: Config) -> None:
    payloads: list[dict[str, Any]] = []
    missing: list[Path] = []
    for spec in run_specs():
        path = config.results_dir / "evaluations" / f"{spec.key}.json"
        if not path.exists():
            missing.append(path)
        else:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} Exp7.3.2 evaluations:\n{preview}")

    runs = pd.DataFrame([_row(payload) for payload in payloads]).sort_values(
        ["backbone_objective", "case", "seed"]
    )
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "method_runs.csv", index=False)

    numeric_cols = [
        "selected_C",
        "train_ba",
        "val_ba",
        "test_ba",
        "legacy_l2_wholecount_probe_test_ba",
        "test_ba_minus_legacy_pp",
        "weight_l2",
        "explicit_intercept_l2",
        "centering_offset_l2",
        "effective_intercept_l2",
        "effective_intercept_span",
        "p7_abs_test_ba_delta",
    ]
    grouped = runs.groupby(
        ["backbone_objective", "case", "aggregation", "center", "bias"], dropna=False
    )[numeric_cols].agg(["mean", "std"])
    grouped.columns = [f"{name}_{stat}" for name, stat in grouped.columns]
    grouped.reset_index().to_csv(config.results_dir / "method_summary.csv", index=False)

    contrasts: list[dict[str, Any]] = []
    contrast_defs = (
        ("mean_center_effect_no_bias", "P2_mean_scale_center_no_bias", "P0_mean_scale_no_center_no_bias"),
        ("mean_center_effect_with_bias", "P3_mean_scale_center_bias", "P1_mean_scale_no_center_bias"),
        ("wholecount_center_effect_no_bias", "P6_wholecount_scale_center_no_bias", "P4_wholecount_scale_no_center_no_bias"),
        ("wholecount_center_effect_with_bias", "P7_wholecount_scale_center_bias", "P5_wholecount_scale_no_center_bias"),
        ("mean_bias_effect_no_center", "P1_mean_scale_no_center_bias", "P0_mean_scale_no_center_no_bias"),
        ("mean_bias_effect_centered", "P3_mean_scale_center_bias", "P2_mean_scale_center_no_bias"),
        ("wholecount_bias_effect_no_center", "P5_wholecount_scale_no_center_bias", "P4_wholecount_scale_no_center_no_bias"),
        ("wholecount_bias_effect_centered", "P7_wholecount_scale_center_bias", "P6_wholecount_scale_center_no_bias"),
        ("aggregation_effect_no_center_no_bias", "P4_wholecount_scale_no_center_no_bias", "P0_mean_scale_no_center_no_bias"),
        ("aggregation_effect_no_center_bias", "P5_wholecount_scale_no_center_bias", "P1_mean_scale_no_center_bias"),
        ("aggregation_effect_center_no_bias", "P6_wholecount_scale_center_no_bias", "P2_mean_scale_center_no_bias"),
        ("aggregation_effect_center_bias", "P7_wholecount_scale_center_bias", "P3_mean_scale_center_bias"),
        ("full_bridge_P7_minus_P0", "P7_wholecount_scale_center_bias", "P0_mean_scale_no_center_no_bias"),
    )
    for name, left, right in contrast_defs:
        contrasts.extend(_paired_contrast(runs, name, left, right))
    contrast_runs = pd.DataFrame(contrasts)
    contrast_runs.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_summary = (
        contrast_runs.groupby(["contrast", "backbone_objective"])["delta_pp"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "delta_pp_count", "mean": "delta_pp_mean", "std": "delta_pp_std"})
    )
    contrast_summary.to_csv(config.results_dir / "contrast_summary.csv", index=False)

    p7 = runs[runs.case == "P7_wholecount_scale_center_bias"][
        [
            "seed",
            "backbone_objective",
            "selected_C",
            "test_ba",
            "legacy_l2_wholecount_probe_test_ba",
            "p7_abs_test_ba_delta",
        ]
    ].copy()
    p7.to_csv(config.results_dir / "p7_legacy_reproduction.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "seeds": list(SEEDS),
        "backbone_objectives": list(BACKBONE_OBJECTIVES),
        "cases": [asdict(case) for case in CASES],
        "C_grid": list(C_GRID),
        "solver": "lbfgs",
        "max_iter": MAX_ITER,
        "logical_rows": len(run_specs()),
        "slurm_layout": "48 independent one-core tasks, one per backbone/seed/case; each task performs the C sweep and writes one evaluation",
        "legacy_reproduction_case": "P7_wholecount_scale_center_bias",
        "legacy_reproduction_note": "P7 intentionally matches Exp7.3 whole-count StandardScaler+intercept probe semantics and probe seed.",
    }
    _save_json(config.results_dir / "manifest.json", manifest)


def _config(args: argparse.Namespace) -> Config:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    return Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        threads=int(args.threads),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exp7.3.2 affine/centering probe bridge")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--array-task-id", type=int, required=True)
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _config(args)
    if args.command == "run":
        specs = run_specs()
        task_id = int(args.array_task_id)
        if task_id < 0 or task_id >= len(specs):
            raise IndexError(f"array task id {task_id} outside [0, {len(specs) - 1}]")
        payload = run_one(specs[task_id], config, force=bool(args.force))
        print(json.dumps({"key": specs[task_id].key, "test_ba": payload["metrics"]["test"]["balanced_accuracy"]}))
        return
    if args.command == "finalize":
        finalize(config)
        print(config.results_dir)
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
