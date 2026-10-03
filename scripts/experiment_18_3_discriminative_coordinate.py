#!/usr/bin/env python3
"""Exp18.3: neuron-level decomposition of high-rate discriminative coordinates.

This experiment is artifact-only. It never trains or modifies an SNN. It reads
finalized Exp18/Exp18.2/CoreBenchmark checkpoints and L2 spike traces to ask why
high-rate neurons become the preferred coordinates for the native bias-free
accumulator.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core_benchmark_v1.protocol import Protocol, SPLITS, paired_seed
from core_benchmark_v1.storage import file_hash, load_torch, save_json
from core_benchmark_v1.training import metrics
from scripts import analyze_experiment_16_2_persistent_neurons as pdiag
from scripts import experiment_18_membrane_history as exp18
from scripts import experiment_18_2_loss_geometry as exp182


EXPERIMENT_ID = "experiment_18_3_discriminative_coordinate"
PROTOCOL_VERSION = "discriminative_coordinate_v1"
SEEDS = (11, 23, 37)
CASES = ("I_WCCE", "I_MWCCE", "U_WCCE", "U_MWCCE")
CARRIERS = {
    "I_WCCE": "I",
    "I_MWCCE": "I",
    "U_WCCE": "U",
    "U_MWCCE": "U",
}
LOSSES = {
    "I_WCCE": "WCCE",
    "I_MWCCE": "MWCCE",
    "U_WCCE": "WCCE",
    "U_MWCCE": "MWCCE",
}
GROUP_FRACTION = 0.30
RANDOM_REPLICATES = 20
EXP18_RESULTS_REL = Path(
    "notebooks/artifacts/experiment_18_membrane_history/membrane_history_v1"
)
EXP18_2_RESULTS_REL = Path(
    "notebooks/artifacts/experiment_18_2_loss_geometry/loss_geometry_v1"
)


@dataclass(frozen=True)
class ExpSpec:
    case: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.case}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    core_results_dir: Path
    exp18_results_dir: Path
    exp18_2_results_dir: Path


def find_repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in (path, *path.parents):
        if (parent / "AGENTS.md").exists() and (parent / "core_benchmark_v1").exists():
            return parent
    raise FileNotFoundError("Repository root not found")


def default_results_dir(root: Path) -> Path:
    return root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def config_from_args(args: argparse.Namespace) -> Config:
    root = find_repo_root()
    return Config(
        repo_root=root,
        results_dir=(args.results or default_results_dir(root)).resolve(),
        core_results_dir=(args.core_results or root / exp18.exp16.CORE_RESULTS_REL).resolve(),
        exp18_results_dir=(args.exp18_results or root / EXP18_RESULTS_REL).resolve(),
        exp18_2_results_dir=(args.exp18_2_results or root / EXP18_2_RESULTS_REL).resolve(),
    )


def specs() -> list[ExpSpec]:
    return [ExpSpec(case, seed) for seed in SEEDS for case in CASES]


def _source_hash() -> str:
    return file_hash(Path(__file__).resolve())


def _dependency_hashes() -> dict[str, str]:
    return {
        "persistent_diagnostic_source_hash": file_hash(Path(pdiag.__file__).resolve()),
        "exp18_source_hash": file_hash(Path(exp18.__file__).resolve()),
        "exp18_2_source_hash": file_hash(Path(exp182.__file__).resolve()),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _core(config: Config) -> tuple[Protocol, dict[str, Any], dict[str, np.ndarray]]:
    p, lock, arrays, _ = exp182._core(
        exp182.Config(
            repo_root=config.repo_root,
            results_dir=config.exp18_2_results_dir,
            core_results_dir=config.core_results_dir,
            exp18_results_dir=config.exp18_results_dir,
        )
    )
    if tuple(p.seeds) != SEEDS or p.width != 128:
        raise ValueError("Exp18.3 requires the locked Exp18/CoreBenchmark 128-neuron contract")
    return p, lock, arrays


def _source_dir(config: Config, spec: ExpSpec, p: Protocol) -> Path:
    if spec.case == "I_WCCE":
        return config.core_results_dir / "runs" / exp18._o0_run(spec.seed, p).key
    if spec.case == "U_WCCE":
        return config.exp18_results_dir / "runs" / f"U_NORMAL__seed{spec.seed}"
    if spec.case in ("I_MWCCE", "U_MWCCE"):
        return config.exp18_2_results_dir / "runs" / f"{spec.case}__seed{spec.seed}"
    raise ValueError(spec.case)


def _protocol_lock(config: Config) -> dict[str, Any]:
    path = config.results_dir / "protocol.lock.json"
    if not path.is_file():
        raise FileNotFoundError("Run Exp18.3 prepare before diagnostics")
    lock = _read_json(path)
    if lock.get("source_hash") != _source_hash():
        raise ValueError("Exp18.3 source changed after prepare")
    if lock.get("dependency_hashes") != _dependency_hashes():
        raise ValueError("Exp18.3 dependency source changed after prepare")
    return lock


def _run_dir(config: Config, spec: ExpSpec) -> Path:
    return config.results_dir / "runs" / spec.key


def _expected_provenance(config: Config, spec: ExpSpec, core_identity: str) -> dict[str, Any]:
    lock = _protocol_lock(config)
    ref = lock["references"][spec.case][str(spec.seed)]
    return {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "source_hash": lock["source_hash"],
        "dependency_hashes": lock["dependency_hashes"],
        "core_identity": core_identity,
        "case": spec.case,
        "seed": spec.seed,
        "carrier": CARRIERS[spec.case],
        "loss": LOSSES[spec.case],
        "source_checkpoint_hash": ref["checkpoint.pt_hash"],
        "source_traces_hash": ref["traces.npz_hash"],
    }


def _validate_complete(directory: Path, expected: dict[str, Any]) -> bool:
    path = directory / "complete.json"
    if not path.is_file():
        return False
    payload = _read_json(path)
    if payload.get("status") != "PASS":
        raise ValueError(f"Non-PASS completion marker: {path}")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Completion provenance mismatch {directory.name}: {key}")
    for name, digest in payload.get("files", {}).items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"Changed Exp18.3 artifact: {directory / name}")
    return True


def _load_source(
    config: Config,
    spec: ExpSpec,
    p: Protocol,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    directory = _source_dir(config, spec, p)
    traces_path = directory / "traces.npz"
    checkpoint_path = directory / "checkpoint.pt"
    with np.load(traces_path, allow_pickle=False) as payload:
        occupancy = {}
        for split in SPLITS:
            spikes = np.asarray(payload[f"{split}__L2__spike"], dtype=np.uint8)
            counts = pdiag._whole_count(spikes, arrays[f"{split}_lengths"])
            occupancy[split] = counts / arrays[f"{split}_lengths"][:, None].astype(np.float64)
    state = load_torch(checkpoint_path)["model_state_dict"]
    weight = state["head.weight"].detach().cpu().numpy().astype(np.float64)
    if weight.shape != (len(p.labels), p.width):
        raise ValueError(f"Unexpected head shape {weight.shape} in {checkpoint_path}")
    return occupancy, weight


def _class_profiles(values: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    rows = []
    for label in range(n_classes):
        selected = y == label
        if not np.any(selected):
            raise ValueError(f"Missing class {label} in profile split")
        rows.append(values[selected].mean(axis=0))
    return np.stack(rows, axis=0)


def _centered_column_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Expected matched [class, neuron] matrices")
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    numer = (ac * bc).sum(axis=0)
    denom = np.sqrt((ac * ac).sum(axis=0) * (bc * bc).sum(axis=0))
    out = np.full(a.shape[1], np.nan, dtype=np.float64)
    valid = denom > 0
    out[valid] = numer[valid] / denom[valid]
    return out


def _aligned_signal(profile: np.ndarray, weight: np.ndarray) -> np.ndarray:
    pc = profile - profile.mean(axis=0, keepdims=True)
    wc = weight - weight.mean(axis=0, keepdims=True)
    return (pc * wc).sum(axis=0)


def _cross_entropy(scores: np.ndarray, y: np.ndarray) -> float:
    shifted = scores - scores.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1)) + scores.max(axis=1)
    chosen = scores[np.arange(len(y)), y.astype(np.int64)]
    return float(np.mean(logsumexp - chosen))


def _eval(scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    out = metrics(y, scores.argmax(axis=1))
    out["ce"] = _cross_entropy(scores, y)
    return out


def _balanced_column_mean(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mean(
        np.stack([values[y == label].mean(axis=0) for label in np.unique(y)], axis=0),
        axis=0,
    )


def _margin_contribution(
    occupancy: np.ndarray,
    scores: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    true = y.astype(np.int64)
    competitor_scores = scores.copy()
    competitor_scores[np.arange(len(y)), true] = -np.inf
    competitor = competitor_scores.argmax(axis=1)
    weight_diff = weight[true] - weight[competitor]
    contribution = occupancy * weight_diff
    return _balanced_column_mean(contribution, y)


def _single_neuron_ablation(
    occupancy: dict[str, np.ndarray],
    weight: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    train_mean = occupancy["train"].mean(axis=0)
    baseline: dict[str, dict[str, float]] = {}
    results: dict[str, dict[str, np.ndarray]] = {}
    n_neurons = weight.shape[1]
    for split in SPLITS:
        x = occupancy[split]
        y = arrays[f"{split}_y"]
        base_scores = x @ weight.T
        base = _eval(base_scores, y)
        baseline[split] = base
        mean_delta_ce = np.zeros(n_neurons, dtype=np.float64)
        mean_drop_ba = np.zeros(n_neurons, dtype=np.float64)
        zero_delta_ce = np.zeros(n_neurons, dtype=np.float64)
        zero_drop_ba = np.zeros(n_neurons, dtype=np.float64)
        for neuron in range(n_neurons):
            mean_scores = base_scores + (
                train_mean[neuron] - x[:, neuron]
            )[:, None] * weight[:, neuron][None, :]
            zero_scores = base_scores - x[:, neuron, None] * weight[:, neuron][None, :]
            mean_eval = _eval(mean_scores, y)
            zero_eval = _eval(zero_scores, y)
            mean_delta_ce[neuron] = mean_eval["ce"] - base["ce"]
            mean_drop_ba[neuron] = 100.0 * (base["ba"] - mean_eval["ba"])
            zero_delta_ce[neuron] = zero_eval["ce"] - base["ce"]
            zero_drop_ba[neuron] = 100.0 * (base["ba"] - zero_eval["ba"])
        results[split] = {
            "mean_replacement_delta_ce": mean_delta_ce,
            "mean_replacement_drop_ba_pp": mean_drop_ba,
            "zero_delta_ce": zero_delta_ce,
            "zero_drop_ba_pp": zero_drop_ba,
        }
    return baseline, results


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xr = pd.Series(x[mask]).rank(method="average").to_numpy()
    yr = pd.Series(y[mask]).rank(method="average").to_numpy()
    if float(np.std(xr)) == 0.0 or float(np.std(yr)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _standardized_ols(
    predictors: dict[str, np.ndarray],
    target: np.ndarray,
) -> dict[str, float]:
    names = list(predictors)
    columns = [np.asarray(predictors[name], dtype=np.float64) for name in names]
    mask = np.isfinite(target)
    for col in columns:
        mask &= np.isfinite(col)
    if int(mask.sum()) <= len(names) + 2:
        return {"n": int(mask.sum()), "r2": float("nan"), **{f"beta_{n}": float("nan") for n in names}}
    x = np.column_stack([col[mask] for col in columns])
    y = np.asarray(target[mask], dtype=np.float64)
    x_std = x.std(axis=0)
    keep = x_std > 0
    xz = np.zeros_like(x)
    xz[:, keep] = (x[:, keep] - x[:, keep].mean(axis=0)) / x_std[keep]
    yz_std = float(y.std())
    if yz_std <= 0:
        return {"n": int(mask.sum()), "r2": float("nan"), **{f"beta_{n}": float("nan") for n in names}}
    yz = (y - y.mean()) / yz_std
    design = np.column_stack([np.ones(len(xz)), xz])
    coef, *_ = np.linalg.lstsq(design, yz, rcond=None)
    pred = design @ coef
    denom = float(((yz - yz.mean()) ** 2).sum())
    r2 = 1.0 - float(((yz - pred) ** 2).sum()) / denom if denom > 0 else float("nan")
    out = {"n": int(mask.sum()), "r2": r2}
    for idx, name in enumerate(names):
        out[f"beta_{name}"] = float(coef[idx + 1]) if keep[idx] else float("nan")
    return out


def _ranking_indices(values: np.ndarray, fraction: float, *, descending: bool = True) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=-np.inf if descending else np.inf)
    order = np.argsort(-clean if descending else clean, kind="stable")
    k = max(1, int(math.ceil(fraction * len(order))))
    return np.sort(order[:k])


def _random_indices(width: int, k: int, spec: ExpSpec, replicate: int) -> np.ndarray:
    rng = np.random.default_rng(
        paired_seed(spec.seed, f"exp18.3:{spec.case}:random30:rep{replicate}")
    )
    return np.sort(rng.choice(width, size=k, replace=False))


def _group_ablation_rows(
    spec: ExpSpec,
    occupancy: dict[str, np.ndarray],
    weight: np.ndarray,
    arrays: dict[str, np.ndarray],
    rankings: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    width = weight.shape[1]
    k = max(1, int(math.ceil(GROUP_FRACTION * width)))
    groups: list[tuple[str, int, np.ndarray]] = [
        (name, -1, indices) for name, indices in rankings.items()
    ]
    groups.extend(
        ("random30", rep, _random_indices(width, k, spec, rep))
        for rep in range(RANDOM_REPLICATES)
    )
    train_mean = occupancy["train"].mean(axis=0)
    rows: list[dict[str, Any]] = []
    for family, replicate, selected in groups:
        selected_text = ";".join(str(int(v)) for v in selected.tolist())
        for split in SPLITS:
            x = occupancy[split]
            y = arrays[f"{split}_y"]
            base_scores = x @ weight.T
            base = _eval(base_scores, y)
            mean_scores = base_scores + (
                train_mean[selected][None, :] - x[:, selected]
            ) @ weight[:, selected].T
            zero_scores = base_scores - x[:, selected] @ weight[:, selected].T
            for method, scores in (
                ("mean_replacement", mean_scores),
                ("zero", zero_scores),
            ):
                cur = _eval(scores, y)
                rows.append({
                    "case": spec.case,
                    "seed": spec.seed,
                    "carrier": CARRIERS[spec.case],
                    "loss": LOSSES[spec.case],
                    "family": family,
                    "replicate": replicate,
                    "fraction_removed": GROUP_FRACTION,
                    "n_removed": len(selected),
                    "method": method,
                    "split": split,
                    "baseline_ba": base["ba"],
                    "ablated_ba": cur["ba"],
                    "delta_ba_pp": 100.0 * (base["ba"] - cur["ba"]),
                    "baseline_ce": base["ce"],
                    "ablated_ce": cur["ce"],
                    "delta_ce": cur["ce"] - base["ce"],
                    "selected_neurons": selected_text,
                })
    return rows


def _analyze(config: Config, spec: ExpSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    p, core_lock, arrays = _core(config)
    expected = _expected_provenance(config, spec, core_lock["identity"])
    occupancy, weight = _load_source(config, spec, p, arrays)

    train_profile = _class_profiles(occupancy["train"], arrays["train_y"], len(p.labels))
    val_profile = _class_profiles(occupancy["val"], arrays["val_y"], len(p.labels))
    test_profile = _class_profiles(occupancy["test"], arrays["test_y"], len(p.labels))
    weight_centered = weight - weight.mean(axis=0, keepdims=True)

    train_eta = pdiag._eta_squared(occupancy["train"], arrays["train_y"])
    val_eta = pdiag._eta_squared(occupancy["val"], arrays["val_y"])
    test_eta = pdiag._eta_squared(occupancy["test"], arrays["test_y"])
    train_alignment = _centered_column_cosine(train_profile, weight)
    val_alignment = _centered_column_cosine(val_profile, weight)
    test_alignment = _centered_column_cosine(test_profile, weight)
    train_aligned_signal = _aligned_signal(train_profile, weight)
    test_aligned_signal = _aligned_signal(test_profile, weight)
    profile_transfer = _centered_column_cosine(train_profile, test_profile)

    train_scores = occupancy["train"] @ weight.T
    val_scores = occupancy["val"] @ weight.T
    test_scores = occupancy["test"] @ weight.T
    train_margin = _margin_contribution(
        occupancy["train"], train_scores, arrays["train_y"], weight
    )
    val_margin = _margin_contribution(
        occupancy["val"], val_scores, arrays["val_y"], weight
    )
    test_margin = _margin_contribution(
        occupancy["test"], test_scores, arrays["test_y"], weight
    )

    baseline, ablation = _single_neuron_ablation(occupancy, weight, arrays)

    train_rate = p.fs * occupancy["train"].mean(axis=0)
    val_rate = p.fs * occupancy["val"].mean(axis=0)
    test_rate = p.fs * occupancy["test"].mean(axis=0)
    head_norm = np.linalg.norm(weight, axis=0)
    head_contrast_norm = np.linalg.norm(weight_centered, axis=0)
    class_profile_norm = np.linalg.norm(
        train_profile - train_profile.mean(axis=0, keepdims=True), axis=0
    )

    records = []
    for neuron in range(p.width):
        records.append({
            "case": spec.case,
            "seed": spec.seed,
            "carrier": CARRIERS[spec.case],
            "loss": LOSSES[spec.case],
            "neuron": neuron,
            "shift": (2, 3, 4)[0 if neuron < 43 else 1 if neuron < 86 else 2],
            "train_rate_hz": float(train_rate[neuron]),
            "val_rate_hz": float(val_rate[neuron]),
            "test_rate_hz": float(test_rate[neuron]),
            "train_class_eta2": float(train_eta[neuron]),
            "val_class_eta2": float(val_eta[neuron]),
            "test_class_eta2": float(test_eta[neuron]),
            "head_weight_norm": float(head_norm[neuron]),
            "head_contrast_norm": float(head_contrast_norm[neuron]),
            "train_class_profile_norm": float(class_profile_norm[neuron]),
            "train_readout_alignment_cos": float(train_alignment[neuron]),
            "val_readout_alignment_cos": float(val_alignment[neuron]),
            "test_readout_alignment_cos": float(test_alignment[neuron]),
            "train_aligned_signal": float(train_aligned_signal[neuron]),
            "test_aligned_signal": float(test_aligned_signal[neuron]),
            "class_profile_train_test_cos": float(profile_transfer[neuron]),
            "train_margin_contribution": float(train_margin[neuron]),
            "val_margin_contribution": float(val_margin[neuron]),
            "test_margin_contribution": float(test_margin[neuron]),
            "train_mean_replacement_delta_ce": float(
                ablation["train"]["mean_replacement_delta_ce"][neuron]
            ),
            "val_mean_replacement_delta_ce": float(
                ablation["val"]["mean_replacement_delta_ce"][neuron]
            ),
            "test_mean_replacement_delta_ce": float(
                ablation["test"]["mean_replacement_delta_ce"][neuron]
            ),
            "train_mean_replacement_drop_ba_pp": float(
                ablation["train"]["mean_replacement_drop_ba_pp"][neuron]
            ),
            "val_mean_replacement_drop_ba_pp": float(
                ablation["val"]["mean_replacement_drop_ba_pp"][neuron]
            ),
            "test_mean_replacement_drop_ba_pp": float(
                ablation["test"]["mean_replacement_drop_ba_pp"][neuron]
            ),
            "train_zero_delta_ce": float(ablation["train"]["zero_delta_ce"][neuron]),
            "val_zero_delta_ce": float(ablation["val"]["zero_delta_ce"][neuron]),
            "test_zero_delta_ce": float(ablation["test"]["zero_delta_ce"][neuron]),
            "train_zero_drop_ba_pp": float(ablation["train"]["zero_drop_ba_pp"][neuron]),
            "val_zero_drop_ba_pp": float(ablation["val"]["zero_drop_ba_pp"][neuron]),
            "test_zero_drop_ba_pp": float(ablation["test"]["zero_drop_ba_pp"][neuron]),
        })
    neurons = pd.DataFrame(records)

    high_rate = _ranking_indices(train_rate, GROUP_FRACTION)
    low_rate = _ranking_indices(train_rate, GROUP_FRACTION, descending=False)
    high_selectivity = _ranking_indices(train_eta, GROUP_FRACTION)
    high_alignment = _ranking_indices(train_aligned_signal, GROUP_FRACTION)
    high_margin = _ranking_indices(train_margin, GROUP_FRACTION)
    group_rows = _group_ablation_rows(
        spec,
        occupancy,
        weight,
        arrays,
        {
            "high_rate30": high_rate,
            "low_rate30": low_rate,
            "high_selectivity30": high_selectivity,
            "high_alignment30": high_alignment,
            "high_margin30": high_margin,
        },
    )
    groups = pd.DataFrame(group_rows)

    corr_targets = {
        "train_class_eta2": train_eta,
        "test_class_eta2": test_eta,
        "head_contrast_norm": head_contrast_norm,
        "train_readout_alignment_cos": train_alignment,
        "test_readout_alignment_cos": test_alignment,
        "class_profile_train_test_cos": profile_transfer,
        "train_margin_contribution": train_margin,
        "test_margin_contribution": test_margin,
        "test_mean_replacement_delta_ce": ablation["test"]["mean_replacement_delta_ce"],
        "test_mean_replacement_drop_ba_pp": ablation["test"]["mean_replacement_drop_ba_pp"],
    }
    correlations = {
        f"spearman_train_rate_vs_{name}": _spearman(train_rate, value)
        for name, value in corr_targets.items()
    }

    high25 = _ranking_indices(train_rate, 0.25)
    low25 = _ranking_indices(train_rate, 0.25, descending=False)
    summary: dict[str, Any] = {
        **expected,
        "status": "PASS",
        "n_neurons": p.width,
        "group_fraction": GROUP_FRACTION,
        "random_replicates": RANDOM_REPLICATES,
        **{f"{split}_native_ba": baseline[split]["ba"] for split in SPLITS},
        **{f"{split}_native_ce": baseline[split]["ce"] for split in SPLITS},
        "mean_train_rate_hz": float(np.mean(train_rate)),
        "mean_test_rate_hz": float(np.mean(test_rate)),
        "p95_train_rate_hz": float(np.quantile(train_rate, 0.95)),
        "fraction_train_rate_ge_10hz": float(np.mean(train_rate >= 10.0)),
        "mean_train_class_eta2": float(np.nanmean(train_eta)),
        "mean_test_class_eta2": float(np.nanmean(test_eta)),
        "mean_train_alignment_cos": float(np.nanmean(train_alignment)),
        "mean_test_alignment_cos": float(np.nanmean(test_alignment)),
        "mean_profile_transfer_cos": float(np.nanmean(profile_transfer)),
        "mean_test_margin_contribution": float(np.nanmean(test_margin)),
        "mean_test_mean_replacement_delta_ce": float(
            np.nanmean(ablation["test"]["mean_replacement_delta_ce"])
        ),
        "high25_train_class_eta2": float(np.nanmean(train_eta[high25])),
        "low25_train_class_eta2": float(np.nanmean(train_eta[low25])),
        "high25_test_class_eta2": float(np.nanmean(test_eta[high25])),
        "low25_test_class_eta2": float(np.nanmean(test_eta[low25])),
        "high25_test_alignment_cos": float(np.nanmean(test_alignment[high25])),
        "low25_test_alignment_cos": float(np.nanmean(test_alignment[low25])),
        "high25_test_margin_contribution": float(np.nanmean(test_margin[high25])),
        "low25_test_margin_contribution": float(np.nanmean(test_margin[low25])),
        "high25_test_mean_replacement_delta_ce": float(
            np.nanmean(ablation["test"]["mean_replacement_delta_ce"][high25])
        ),
        "low25_test_mean_replacement_delta_ce": float(
            np.nanmean(ablation["test"]["mean_replacement_delta_ce"][low25])
        ),
        "correlations": correlations,
        "utility_ols": _standardized_ols(
            {
                "log1p_rate": np.log1p(train_rate),
                "train_class_eta2": train_eta,
                "train_alignment": train_alignment,
                "head_contrast_norm": head_contrast_norm,
                "profile_transfer": profile_transfer,
            },
            ablation["test"]["mean_replacement_delta_ce"],
        ),
    }

    random_test = groups[
        (groups["family"] == "random30")
        & (groups["method"] == "mean_replacement")
        & (groups["split"] == "test")
    ]
    high_test = groups[
        (groups["family"] == "high_rate30")
        & (groups["method"] == "mean_replacement")
        & (groups["split"] == "test")
    ].iloc[0]
    summary["high_rate30_test_mean_replace_delta_ce"] = float(high_test["delta_ce"])
    summary["random30_test_mean_replace_delta_ce_mean"] = float(random_test["delta_ce"].mean())
    summary["random30_test_mean_replace_delta_ce_p95"] = float(
        np.quantile(random_test["delta_ce"], 0.95)
    )
    summary["high_rate30_test_mean_replace_drop_ba_pp"] = float(high_test["delta_ba_pp"])
    summary["random30_test_mean_replace_drop_ba_pp_mean"] = float(
        random_test["delta_ba_pp"].mean()
    )
    return neurons, groups, summary


def prepare(config: Config) -> dict[str, Any]:
    p, core_lock, _ = _core(config)
    exp18_2_aggregate = config.exp18_2_results_dir / "aggregate.json"
    if not exp18_2_aggregate.is_file():
        raise FileNotFoundError(exp18_2_aggregate)
    if _read_json(exp18_2_aggregate).get("status") != "PASS":
        raise ValueError("Exp18.2 aggregate is not PASS")

    references: dict[str, dict[str, Any]] = {case: {} for case in CASES}
    required = ("checkpoint.pt", "traces.npz", "native.json")
    for spec in specs():
        directory = _source_dir(config, spec, p)
        for name in required:
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        if spec.case in ("I_MWCCE", "U_MWCCE"):
            complete = _read_json(directory / "complete.json")
            if complete.get("status") != "PASS":
                raise ValueError(f"Exp18.2 source run is not PASS: {spec.key}")
        references[spec.case][str(spec.seed)] = {
            "directory": str(directory),
            **{f"{name}_hash": file_hash(directory / name) for name in required},
        }

    payload = {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "source_hash": _source_hash(),
        "dependency_hashes": _dependency_hashes(),
        "core_identity": core_lock["identity"],
        "core_dataset_hash": core_lock["dataset_hash"],
        "core_protocol_hash": core_lock["protocol_hash"],
        "exp18_2_aggregate_hash": file_hash(exp18_2_aggregate),
        "seeds": list(SEEDS),
        "cases": list(CASES),
        "carriers": CARRIERS,
        "losses": LOSSES,
        "group_fraction": GROUP_FRACTION,
        "random_replicates": RANDOM_REPLICATES,
        "references": references,
        "recommended_concurrency": 12,
    }
    config.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.results_dir / "protocol.lock.json"
    if path.is_file():
        previous = _read_json(path)
        if previous != payload:
            raise ValueError("Existing Exp18.3 protocol lock differs")
        return previous
    save_json(path, payload)
    return payload


def diagnose(config: Config, spec: ExpSpec) -> dict[str, Any]:
    p, core_lock, _ = _core(config)
    expected = _expected_provenance(config, spec, core_lock["identity"])
    directory = _run_dir(config, spec)
    directory.mkdir(parents=True, exist_ok=True)
    if _validate_complete(directory, expected):
        return {"status": "already_complete", "run": spec.key}

    lock = _protocol_lock(config)
    source = _source_dir(config, spec, p)
    ref = lock["references"][spec.case][str(spec.seed)]
    for name in ("checkpoint.pt", "traces.npz", "native.json"):
        if file_hash(source / name) != ref[f"{name}_hash"]:
            raise ValueError(f"Changed source artifact: {source / name}")

    neurons, groups, summary = _analyze(config, spec)
    neurons.to_csv(directory / "neurons.csv", index=False)
    groups.to_csv(directory / "group_ablation.csv", index=False)
    save_json(directory / "summary.json", summary)
    files = ("neurons.csv", "group_ablation.csv", "summary.json")
    save_json(directory / "complete.json", {
        **expected,
        "status": "PASS",
        "files": {name: file_hash(directory / name) for name in files},
    })
    return {
        "status": "PASS",
        "run": spec.key,
        "test_native_ba": summary["test_native_ba"],
        "mean_train_rate_hz": summary["mean_train_rate_hz"],
    }


def smoke(config: Config) -> dict[str, Any]:
    p, core_lock, arrays = _core(config)
    rows = []
    for case in CASES:
        spec = ExpSpec(case, SEEDS[0])
        occupancy, weight = _load_source(config, spec, p, arrays)
        train_profile = _class_profiles(
            occupancy["train"], arrays["train_y"], len(p.labels)
        )
        alignment = _centered_column_cosine(train_profile, weight)
        scores = occupancy["train"] @ weight.T
        margin = _margin_contribution(
            occupancy["train"], scores, arrays["train_y"], weight
        )
        baseline, ablation = _single_neuron_ablation(occupancy, weight, arrays)
        if weight.shape != (len(p.labels), p.width):
            raise AssertionError(case)
        if len(alignment) != p.width or len(margin) != p.width:
            raise AssertionError(f"{case}: neuron diagnostic shape mismatch")
        if not np.isfinite(baseline["train"]["ba"]):
            raise FloatingPointError(f"{case}: nonfinite baseline")
        if len(ablation["test"]["mean_replacement_delta_ce"]) != p.width:
            raise AssertionError(f"{case}: ablation shape mismatch")
        rows.append({
            "case": case,
            "train_ba": baseline["train"]["ba"],
            "mean_train_rate_hz": float(p.fs * occupancy["train"].mean()),
            "finite_alignment_fraction": float(np.isfinite(alignment).mean()),
        })
    return {
        "status": "PASS",
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "core_identity": core_lock["identity"],
        "cases": rows,
    }


def finalize(config: Config) -> dict[str, Any]:
    p, core_lock, _ = _core(config)
    lock = _protocol_lock(config)
    neuron_frames = []
    group_frames = []
    summaries = []
    for spec in specs():
        directory = _run_dir(config, spec)
        expected = _expected_provenance(config, spec, core_lock["identity"])
        if not _validate_complete(directory, expected):
            raise FileNotFoundError(f"Missing Exp18.3 complete run: {spec.key}")
        source = _source_dir(config, spec, p)
        ref = lock["references"][spec.case][str(spec.seed)]
        for name in ("checkpoint.pt", "traces.npz", "native.json"):
            if file_hash(source / name) != ref[f"{name}_hash"]:
                raise ValueError(f"Changed source artifact: {source / name}")
        neuron_frames.append(pd.read_csv(directory / "neurons.csv"))
        group_frames.append(pd.read_csv(directory / "group_ablation.csv"))
        summaries.append(_read_json(directory / "summary.json"))

    neurons = pd.concat(neuron_frames, ignore_index=True)
    groups = pd.concat(group_frames, ignore_index=True)
    aggregate_dir = config.results_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    neurons.to_csv(aggregate_dir / "neurons.csv", index=False)
    groups.to_csv(aggregate_dir / "group_ablation.csv", index=False)

    flat_summary_rows = []
    correlation_rows = []
    ols_rows = []
    for row in summaries:
        flat = {
            key: value
            for key, value in row.items()
            if key not in ("correlations", "utility_ols", "dependency_hashes")
            and isinstance(value, (str, int, float, bool))
        }
        flat_summary_rows.append(flat)
        correlation_rows.append({
            "case": row["case"],
            "seed": row["seed"],
            "carrier": row["carrier"],
            "loss": row["loss"],
            **row["correlations"],
        })
        ols_rows.append({
            "case": row["case"],
            "seed": row["seed"],
            "carrier": row["carrier"],
            "loss": row["loss"],
            **row["utility_ols"],
        })

    summary_frame = pd.DataFrame(flat_summary_rows)
    corr_frame = pd.DataFrame(correlation_rows)
    ols_frame = pd.DataFrame(ols_rows)
    summary_frame.to_csv(aggregate_dir / "run_summary.csv", index=False)
    corr_frame.to_csv(aggregate_dir / "correlations.csv", index=False)
    ols_frame.to_csv(aggregate_dir / "utility_ols.csv", index=False)

    numeric = [
        "test_native_ba",
        "mean_train_rate_hz",
        "p95_train_rate_hz",
        "fraction_train_rate_ge_10hz",
        "mean_train_class_eta2",
        "mean_test_class_eta2",
        "mean_train_alignment_cos",
        "mean_test_alignment_cos",
        "mean_profile_transfer_cos",
        "mean_test_margin_contribution",
        "mean_test_mean_replacement_delta_ce",
        "high25_test_class_eta2",
        "low25_test_class_eta2",
        "high25_test_alignment_cos",
        "low25_test_alignment_cos",
        "high25_test_margin_contribution",
        "low25_test_margin_contribution",
        "high25_test_mean_replacement_delta_ce",
        "low25_test_mean_replacement_delta_ce",
        "high_rate30_test_mean_replace_delta_ce",
        "random30_test_mean_replace_delta_ce_mean",
    ]
    case_rows = []
    for case in CASES:
        selected = summary_frame[summary_frame["case"] == case]
        out: dict[str, Any] = {
            "case": case,
            "carrier": CARRIERS[case],
            "loss": LOSSES[case],
        }
        for name in numeric:
            vals = selected[name].astype(float)
            out[f"{name}_mean"] = float(vals.mean())
            out[f"{name}_sd"] = float(vals.std(ddof=1))
        case_rows.append(out)
    case_frame = pd.DataFrame(case_rows)
    case_frame.to_csv(aggregate_dir / "case_summary.csv", index=False)

    paired_rows = []
    for carrier in ("I", "U"):
        wcce = f"{carrier}_WCCE"
        mwcce = f"{carrier}_MWCCE"
        for seed in SEEDS:
            a = summary_frame[
                (summary_frame["case"] == wcce) & (summary_frame["seed"] == seed)
            ].iloc[0]
            b = summary_frame[
                (summary_frame["case"] == mwcce) & (summary_frame["seed"] == seed)
            ].iloc[0]
            row: dict[str, Any] = {"carrier": carrier, "seed": seed}
            for name in numeric:
                row[f"delta_MWCCE_minus_WCCE__{name}"] = float(b[name] - a[name])
            paired_rows.append(row)
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(aggregate_dir / "paired_loss_contrasts.csv", index=False)

    group_summary = (
        groups.groupby(["case", "family", "method", "split"], dropna=False)
        .agg(
            delta_ba_pp_mean=("delta_ba_pp", "mean"),
            delta_ba_pp_sd=("delta_ba_pp", "std"),
            delta_ce_mean=("delta_ce", "mean"),
            delta_ce_sd=("delta_ce", "std"),
            n_rows=("delta_ce", "size"),
        )
        .reset_index()
    )
    group_summary.to_csv(aggregate_dir / "group_summary.csv", index=False)

    payload = {
        "status": "PASS",
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "core_identity": core_lock["identity"],
        "n_runs": len(summaries),
        "n_neuron_rows": len(neurons),
        "n_group_rows": len(groups),
        "files": {
            name: file_hash(aggregate_dir / name)
            for name in (
                "neurons.csv",
                "group_ablation.csv",
                "run_summary.csv",
                "correlations.csv",
                "utility_ols.csv",
                "case_summary.csv",
                "paired_loss_contrasts.csv",
                "group_summary.csv",
            )
        },
    }
    save_json(config.results_dir / "aggregate.json", payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    parser.add_argument("--exp18-results", type=Path)
    parser.add_argument("--exp18-2-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    sub.add_parser("prepare")
    diagnose_p = sub.add_parser("diagnose")
    diagnose_p.add_argument("--case", choices=CASES, required=True)
    diagnose_p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    sub.add_parser("finalize")
    args = parser.parse_args(argv)
    config = config_from_args(args)

    if args.command == "smoke":
        result = smoke(config)
    elif args.command == "prepare":
        result = prepare(config)
    elif args.command == "diagnose":
        result = diagnose(config, ExpSpec(args.case, args.seed))
    else:
        result = finalize(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
