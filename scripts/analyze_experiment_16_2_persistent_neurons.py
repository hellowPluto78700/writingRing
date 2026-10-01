#!/usr/bin/env python3
"""Exp16.2 persistent-neuron contribution diagnostic.

This analysis is artifact-only. It reads saved Exp16.2 spike traces and the
locked CoreBenchmark dataset; it never retrains the SNN.

Questions answered:
1. Which L1/L2 neurons fire persistently?
2. Are high-firing L2 neurons causally used by the WholeCount probe?
3. Is their information unique, or can a retrained probe recover after removal?
4. Are they mainly encoding duration?
5. Is their utility transferable from train to validation/test users?

Primary outputs live under:
  <exp16.2 results>/persistent_neuron_diagnostic/
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_benchmark_v1.probes import fit_probe
from core_benchmark_v1.protocol import Protocol, SPLITS
from core_benchmark_v1.training import metrics
from scripts import experiment_16_2_matched_budget_selective_write as exp16_2
from scripts import experiment_16_prefix_supervised_selective_memory as exp16


DEFAULT_CASES = ("C0", "GZ0", "S90", "G90", "S70", "G70", "S50", "G50")
TOP_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
OCCUPANCY_THRESHOLDS = (0.50, 0.80, 0.95)


def _parse_cases(value: str) -> tuple[str, ...]:
    cases = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(cases) - set(DEFAULT_CASES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown case(s): {unknown}; valid cases: {DEFAULT_CASES}"
        )
    if not cases:
        raise argparse.ArgumentTypeError("At least one case is required")
    return cases


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from exc
    unknown = sorted(set(seeds) - set(exp16_2.FORMAL_SEEDS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown seed(s): {unknown}; valid seeds: {exp16_2.FORMAL_SEEDS}"
        )
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def _config(
    results_dir: Path | None,
    core_results_dir: Path | None,
) -> exp16_2.Config:
    root = exp16_2.find_repo_root()
    return exp16_2.Config(
        repo_root=root,
        results_dir=(
            results_dir.resolve()
            if results_dir is not None
            else exp16_2.default_results_dir(root)
        ),
        core_results_dir=(
            core_results_dir.resolve()
            if core_results_dir is not None
            else (root / exp16.CORE_RESULTS_REL).resolve()
        ),
    )


def _load_layer_traces(
    result_root: Path,
    case: str,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    path = result_root / "runs" / f"{case}__seed{seed}" / "traces.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as payload:
        for split in SPLITS:
            out[split] = {
                "L1": np.asarray(payload[f"{split}__L1__spike"], dtype=np.uint8),
                "L2": np.asarray(payload[f"{split}__L2__spike"], dtype=np.uint8),
            }
    return out


def _whole_count(spikes: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    if spikes.ndim != 3 or len(spikes) != len(lengths):
        raise ValueError("Expected spikes=[sample,time,neuron] aligned with lengths")
    mask = np.arange(spikes.shape[1])[None, :] < lengths[:, None]
    return (spikes * mask[:, :, None]).sum(axis=1).astype(np.float64)


def _occupancy(spikes: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return _whole_count(spikes, lengths) / lengths[:, None].astype(np.float64)


def _longest_runs(spikes: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Return longest consecutive spike run for every sample/neuron."""
    if spikes.ndim != 3 or len(spikes) != len(lengths):
        raise ValueError("Expected spikes=[sample,time,neuron] aligned with lengths")
    n_samples, _, n_neurons = spikes.shape
    result = np.zeros((n_samples, n_neurons), dtype=np.int32)
    for i, length in enumerate(lengths.astype(int).tolist()):
        current = np.zeros(n_neurons, dtype=np.int32)
        best = np.zeros(n_neurons, dtype=np.int32)
        for row in spikes[i, :length]:
            current = (current + 1) * row.astype(np.int32)
            best = np.maximum(best, current)
        result[i] = best
    return result


def _safe_column_corr(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pearson correlation of every values column with target."""
    x = values.astype(np.float64)
    y = target.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean()
    denom = np.sqrt((x * x).sum(axis=0) * float((y * y).sum()))
    numer = (x * y[:, None]).sum(axis=0)
    result = np.zeros(x.shape[1], dtype=np.float64)
    valid = denom > 0
    result[valid] = numer[valid] / denom[valid]
    result[~valid] = np.nan
    return result


def _eta_squared(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """One-way eta-squared for every feature column."""
    x = values.astype(np.float64)
    global_mean = x.mean(axis=0)
    ss_total = ((x - global_mean) ** 2).sum(axis=0)
    ss_between = np.zeros(x.shape[1], dtype=np.float64)
    for group in np.unique(groups):
        chosen = groups == group
        if not np.any(chosen):
            continue
        mean = x[chosen].mean(axis=0)
        ss_between += int(chosen.sum()) * (mean - global_mean) ** 2
    out = np.zeros_like(ss_total)
    valid = ss_total > 0
    out[valid] = ss_between[valid] / ss_total[valid]
    out[~valid] = np.nan
    return out


def _balanced_column_mean(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    pieces = [values[y == label].mean(axis=0) for label in np.unique(y)]
    return np.mean(np.stack(pieces, axis=0), axis=0)


def _scores(model: Any, x_scaled: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(x_scaled), dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    return scores


def _cross_entropy_from_scores(
    scores: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
) -> float:
    class_to_col = {int(label): idx for idx, label in enumerate(classes.tolist())}
    cols = np.asarray([class_to_col[int(label)] for label in y], dtype=np.int64)
    shifted = scores - scores.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1)) + scores.max(axis=1)
    chosen = scores[np.arange(len(y)), cols]
    return float(np.mean(logsumexp - chosen))


def _predictions(scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return classes[np.argmax(scores, axis=1)]


def _eval_scores(
    scores: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    result = metrics(y, _predictions(scores, classes))
    result["ce"] = _cross_entropy_from_scores(scores, y, classes)
    return result


def _fit_wholecount_probe(
    features: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    p: Protocol,
) -> tuple[Any, Any, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    scaler, model, _ = fit_probe(
        features["train"],
        arrays["train_y"],
        features["val"],
        arrays["val_y"],
        "no_bias",
        p,
    )
    scaled = {
        split: scaler.transform(features[split].astype(np.float64))
        for split in SPLITS
    }
    score_map = {split: _scores(model, scaled[split]) for split in SPLITS}
    baseline = {
        split: _eval_scores(score_map[split], arrays[f"{split}_y"], model.classes_)
        for split in SPLITS
    }
    return scaler, model, scaled, score_map, baseline


def _margin_contribution(
    x_scaled: np.ndarray,
    scores: np.ndarray,
    y: np.ndarray,
    model: Any,
) -> np.ndarray:
    classes = np.asarray(model.classes_)
    coef = np.asarray(model.coef_, dtype=np.float64)
    class_to_col = {int(label): idx for idx, label in enumerate(classes.tolist())}
    true_cols = np.asarray([class_to_col[int(label)] for label in y], dtype=np.int64)
    competitor_scores = scores.copy()
    competitor_scores[np.arange(len(y)), true_cols] = -np.inf
    competitor_cols = competitor_scores.argmax(axis=1)
    weight_diff = coef[true_cols] - coef[competitor_cols]
    contribution = x_scaled * weight_diff
    return _balanced_column_mean(contribution, y)


def _single_neuron_frozen_ablation(
    scaled: dict[str, np.ndarray],
    score_map: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    scaler: Any,
    model: Any,
    baseline: dict[str, dict[str, float]],
) -> dict[str, dict[str, np.ndarray]]:
    """Mean-replacement ablation with the fitted probe frozen."""
    coef = np.asarray(model.coef_, dtype=np.float64)
    train_mean = features["train"].mean(axis=0)
    replacement_scaled = train_mean / np.asarray(scaler.scale_, dtype=np.float64)
    n_neurons = scaled["train"].shape[1]
    out: dict[str, dict[str, np.ndarray]] = {}

    for split in SPLITS:
        y = arrays[f"{split}_y"]
        ce_delta = np.zeros(n_neurons, dtype=np.float64)
        ba_drop = np.zeros(n_neurons, dtype=np.float64)
        x = scaled[split]
        base_scores = score_map[split]
        for neuron in range(n_neurons):
            dx = replacement_scaled[neuron] - x[:, neuron]
            ablated_scores = base_scores + dx[:, None] * coef[:, neuron][None, :]
            evaluated = _eval_scores(ablated_scores, y, model.classes_)
            ce_delta[neuron] = evaluated["ce"] - baseline[split]["ce"]
            ba_drop[neuron] = 100.0 * (baseline[split]["ba"] - evaluated["ba"])
        out[split] = {
            "delta_ce": ce_delta,
            "delta_ba_pp": ba_drop,
        }
    return out


def _group_frozen_ablation(
    selected: np.ndarray,
    split: str,
    scaled: dict[str, np.ndarray],
    score_map: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    scaler: Any,
    model: Any,
) -> dict[str, float]:
    train_mean = features["train"].mean(axis=0)
    replacement_scaled = train_mean / np.asarray(scaler.scale_, dtype=np.float64)
    x = scaled[split]
    coef = np.asarray(model.coef_, dtype=np.float64)
    dx = replacement_scaled[selected][None, :] - x[:, selected]
    scores = score_map[split] + dx @ coef[:, selected].T
    return _eval_scores(scores, arrays[f"{split}_y"], model.classes_)


def _fit_reduced_probe(
    selected: np.ndarray,
    features: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    p: Protocol,
) -> dict[str, dict[str, float]]:
    keep = np.ones(features["train"].shape[1], dtype=bool)
    keep[selected] = False
    reduced = {split: features[split][:, keep] for split in SPLITS}
    scaler, model, _, score_map, result = _fit_wholecount_probe(reduced, arrays, p)
    del scaler, model, score_map
    return result


def _fit_rate_probe(
    features: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    p: Protocol,
) -> dict[str, dict[str, float]]:
    rate = {
        split: features[split] / arrays[f"{split}_lengths"][:, None].astype(np.float64)
        for split in SPLITS
    }
    _, _, _, _, result = _fit_wholecount_probe(rate, arrays, p)
    return result


def _fit_duration_probe(
    arrays: dict[str, np.ndarray],
    p: Protocol,
) -> dict[str, dict[str, float]]:
    features = {
        split: arrays[f"{split}_lengths"].astype(np.float64)[:, None]
        for split in SPLITS
    }
    scaler, model, _ = fit_probe(
        features["train"],
        arrays["train_y"],
        features["val"],
        arrays["val_y"],
        "affine",
        p,
    )
    result: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        x = scaler.transform(features[split])
        scores = _scores(model, x)
        result[split] = _eval_scores(scores, arrays[f"{split}_y"], model.classes_)
    return result


def _layer_activity_rows(
    case: str,
    seed: int,
    layer: str,
    traces: dict[str, dict[str, np.ndarray]],
    arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_split: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        lengths = arrays[f"{split}_lengths"]
        spikes = traces[split][layer]
        counts = _whole_count(spikes, lengths)
        occupancy = counts / lengths[:, None]
        longest = _longest_runs(spikes, lengths)
        per_split[split] = {
            "counts": counts,
            "occupancy": occupancy,
            "longest": longest,
        }

    n_neurons = traces["train"][layer].shape[2]
    rows: list[dict[str, Any]] = []
    for neuron in range(n_neurons):
        row: dict[str, Any] = {
            "case": case,
            "seed": seed,
            "layer": layer,
            "neuron": neuron,
        }
        for split in SPLITS:
            occ = per_split[split]["occupancy"][:, neuron]
            longest = per_split[split]["longest"][:, neuron]
            counts = per_split[split]["counts"][:, neuron]
            lengths = arrays[f"{split}_lengths"].astype(np.float64)
            row.update(
                {
                    f"{split}_mean_occupancy": float(occ.mean()),
                    f"{split}_median_occupancy": float(np.median(occ)),
                    f"{split}_p95_occupancy": float(np.quantile(occ, 0.95)),
                    f"{split}_mean_longest_run": float(longest.mean()),
                    f"{split}_p95_longest_run": float(np.quantile(longest, 0.95)),
                    f"{split}_max_longest_run": int(longest.max()),
                    f"{split}_duration_corr": float(
                        _safe_column_corr(counts[:, None], lengths)[0]
                    ),
                }
            )
            for threshold in OCCUPANCY_THRESHOLDS:
                tag = int(round(100 * threshold))
                row[f"{split}_p_sample_occ_gt_{tag}"] = float((occ > threshold).mean())
        rows.append(row)

    summary: dict[str, Any] = {
        "case": case,
        "seed": seed,
        "layer": layer,
        "n_neurons": n_neurons,
    }
    for split in SPLITS:
        mean_per_neuron = per_split[split]["occupancy"].mean(axis=0)
        longest_mean = per_split[split]["longest"].mean(axis=0)
        summary.update(
            {
                f"{split}_mean_activity": float(mean_per_neuron.mean()),
                f"{split}_median_neuron_occupancy": float(np.median(mean_per_neuron)),
                f"{split}_p90_neuron_occupancy": float(np.quantile(mean_per_neuron, 0.90)),
                f"{split}_max_neuron_occupancy": float(mean_per_neuron.max()),
                f"{split}_mean_neuron_longest_run": float(longest_mean.mean()),
                f"{split}_p90_neuron_longest_run": float(np.quantile(longest_mean, 0.90)),
            }
        )
        for threshold in OCCUPANCY_THRESHOLDS:
            tag = int(round(100 * threshold))
            summary[f"{split}_fraction_neurons_mean_occ_gt_{tag}"] = float(
                (mean_per_neuron > threshold).mean()
            )
    return rows, summary


def _l2_utility_rows(
    case: str,
    seed: int,
    traces: dict[str, dict[str, np.ndarray]],
    arrays: dict[str, np.ndarray],
    p: Protocol,
    activity_frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    features = {
        split: _whole_count(traces[split]["L2"], arrays[f"{split}_lengths"])
        for split in SPLITS
    }
    scaler, model, scaled, score_map, baseline = _fit_wholecount_probe(features, arrays, p)
    frozen = _single_neuron_frozen_ablation(
        scaled, score_map, features, arrays, scaler, model, baseline
    )

    margin = {
        split: _margin_contribution(
            scaled[split],
            score_map[split],
            arrays[f"{split}_y"],
            model,
        )
        for split in SPLITS
    }
    class_eta = {
        split: _eta_squared(features[split], arrays[f"{split}_y"])
        for split in SPLITS
    }
    user_eta = {
        split: _eta_squared(features[split], arrays[f"{split}_users"])
        for split in SPLITS
    }

    l2_activity = activity_frame[
        (activity_frame["case"] == case)
        & (activity_frame["seed"] == seed)
        & (activity_frame["layer"] == "L2")
    ].sort_values("neuron")

    utility_rows: list[dict[str, Any]] = []
    for neuron, (_, activity) in enumerate(l2_activity.iterrows()):
        row = activity.to_dict()
        for split in SPLITS:
            row.update(
                {
                    f"{split}_margin_contribution": float(margin[split][neuron]),
                    f"{split}_frozen_delta_ce": float(frozen[split]["delta_ce"][neuron]),
                    f"{split}_frozen_delta_ba_pp": float(frozen[split]["delta_ba_pp"][neuron]),
                    f"{split}_class_eta2": float(class_eta[split][neuron]),
                    f"{split}_user_eta2": float(user_eta[split][neuron]),
                }
            )
        row["test_minus_train_delta_ce"] = (
            row["test_frozen_delta_ce"] - row["train_frozen_delta_ce"]
        )
        utility_rows.append(row)

    probe_rows: list[dict[str, Any]] = []
    rate = _fit_rate_probe(features, arrays, p)
    duration = _fit_duration_probe(arrays, p)
    for probe_name, result in (
        ("whole_count", baseline),
        ("rate_normalized", rate),
        ("duration_only", duration),
    ):
        row: dict[str, Any] = {"case": case, "seed": seed, "probe": probe_name}
        for split in SPLITS:
            row.update(
                {
                    f"{split}_ba": result[split]["ba"],
                    f"{split}_accuracy": result[split]["accuracy"],
                    f"{split}_ce": result[split]["ce"],
                }
            )
        row["train_test_gap"] = row["train_ba"] - row["test_ba"]
        probe_rows.append(row)

    train_occupancy = np.asarray(
        [row["train_mean_occupancy"] for row in utility_rows], dtype=np.float64
    )
    ranking = np.argsort(-train_occupancy, kind="stable")
    group_rows: list[dict[str, Any]] = []
    n_neurons = len(ranking)
    for fraction in TOP_FRACTIONS:
        k = max(1, int(math.ceil(fraction * n_neurons)))
        selected = np.sort(ranking[:k])
        selected_text = ";".join(str(int(v)) for v in selected.tolist())

        frozen_result = {
            split: _group_frozen_ablation(
                selected,
                split,
                scaled,
                score_map,
                features,
                arrays,
                scaler,
                model,
            )
            for split in SPLITS
        }
        retrained_result = _fit_reduced_probe(selected, features, arrays, p)

        for method, result in (
            ("frozen_mean_replacement", frozen_result),
            ("retrained_without_neurons", retrained_result),
        ):
            row = {
                "case": case,
                "seed": seed,
                "fraction_removed": fraction,
                "n_removed": k,
                "method": method,
                "selected_neurons": selected_text,
                "selected_train_mean_occupancy": float(train_occupancy[selected].mean()),
            }
            for split in SPLITS:
                row.update(
                    {
                        f"{split}_ba": result[split]["ba"],
                        f"{split}_ce": result[split]["ce"],
                        f"{split}_delta_ba_pp_vs_full": 100.0
                        * (baseline[split]["ba"] - result[split]["ba"]),
                        f"{split}_delta_ce_vs_full": result[split]["ce"]
                        - baseline[split]["ce"],
                    }
                )
            group_rows.append(row)

    return utility_rows, group_rows, probe_rows


def _plot_activity_distribution(
    case: str,
    seed: int,
    activity: pd.DataFrame,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for layer in ("L1", "L2"):
        values = activity[
            (activity["case"] == case)
            & (activity["seed"] == seed)
            & (activity["layer"] == layer)
        ]["test_mean_occupancy"].to_numpy()
        ax.hist(values, bins=np.linspace(0, 1, 21), alpha=0.45, label=layer)
    ax.set_xlabel("Mean valid-window firing occupancy per neuron")
    ax.set_ylabel("Neuron count")
    ax.set_title(f"{case} seed{seed}: L1/L2 occupancy distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_occupancy_utility(
    case: str,
    seed: int,
    utility: pd.DataFrame,
    output: Path,
) -> None:
    selected = utility[(utility["case"] == case) & (utility["seed"] == seed)]
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        selected["test_mean_occupancy"],
        selected["test_frozen_delta_ce"],
        c=selected["test_duration_corr"],
        s=30.0 + selected["test_p95_longest_run"].to_numpy() * 0.7,
    )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("L2 mean test occupancy")
    ax.set_ylabel("Frozen WholeCount ablation ΔCE (positive = useful)")
    ax.set_title(f"{case} seed{seed}: persistence vs WholeCount utility")
    fig.colorbar(scatter, ax=ax, label="count-duration correlation")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_train_test_utility(
    case: str,
    seed: int,
    utility: pd.DataFrame,
    output: Path,
) -> None:
    selected = utility[(utility["case"] == case) & (utility["seed"] == seed)]
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        selected["train_frozen_delta_ce"],
        selected["test_frozen_delta_ce"],
        c=selected["test_mean_occupancy"],
    )
    limits = np.asarray(
        [
            selected["train_frozen_delta_ce"].min(),
            selected["train_frozen_delta_ce"].max(),
            selected["test_frozen_delta_ce"].min(),
            selected["test_frozen_delta_ce"].max(),
        ],
        dtype=float,
    )
    lo, hi = float(limits.min()), float(limits.max())
    if math.isclose(lo, hi):
        lo, hi = lo - 1e-3, hi + 1e-3
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("Train frozen ablation ΔCE")
    ax.set_ylabel("Test frozen ablation ΔCE")
    ax.set_title(f"{case} seed{seed}: WholeCount utility transfer")
    fig.colorbar(scatter, ax=ax, label="test occupancy")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_group_ablation(
    case: str,
    seed: int,
    groups: pd.DataFrame,
    probe_summary: pd.DataFrame,
    output: Path,
) -> None:
    selected = groups[(groups["case"] == case) & (groups["seed"] == seed)]
    baseline = probe_summary[
        (probe_summary["case"] == case)
        & (probe_summary["seed"] == seed)
        & (probe_summary["probe"] == "whole_count")
    ].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("frozen_mean_replacement", "retrained_without_neurons"):
        rows = selected[selected["method"] == method].sort_values("fraction_removed")
        x = np.concatenate(([0.0], rows["fraction_removed"].to_numpy(dtype=float)))
        y = np.concatenate(
            (
                [100.0 * float(baseline["test_ba"])],
                100.0 * rows["test_ba"].to_numpy(dtype=float),
            )
        )
        ax.plot(x * 100.0, y, marker="o", label=method)
    ax.set_xlabel("Highest-train-occupancy neurons removed (%)")
    ax.set_ylabel("Test WholeCount BA (%)")
    ax.set_title(f"{case} seed{seed}: high-firing group ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _case_summary(
    layer_summary: pd.DataFrame,
    utility: pd.DataFrame,
    groups: pd.DataFrame,
    probes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in sorted(utility["case"].unique()):
        u = utility[utility["case"] == case]
        l2 = layer_summary[
            (layer_summary["case"] == case) & (layer_summary["layer"] == "L2")
        ]
        p = probes[probes["case"] == case]
        row: dict[str, Any] = {
            "case": case,
            "mean_l2_test_activity": float(l2["test_mean_activity"].mean()),
            "mean_l2_test_fraction_neurons_occ_gt_50": float(
                l2["test_fraction_neurons_mean_occ_gt_50"].mean()
            ),
            "mean_l2_test_fraction_neurons_occ_gt_80": float(
                l2["test_fraction_neurons_mean_occ_gt_80"].mean()
            ),
            "mean_l2_test_p90_neuron_occupancy": float(
                l2["test_p90_neuron_occupancy"].mean()
            ),
            "mean_test_frozen_delta_ce_highest_quartile_occ": float(
                u.assign(
                    occ_rank=u.groupby("seed")["test_mean_occupancy"].rank(pct=True)
                )
                .query("occ_rank >= 0.75")["test_frozen_delta_ce"]
                .mean()
            ),
        }
        for probe_name in ("whole_count", "rate_normalized", "duration_only"):
            pp = p[p["probe"] == probe_name]
            row[f"{probe_name}_test_ba"] = float(pp["test_ba"].mean())
        row["count_minus_rate_test_ba_pp"] = 100.0 * (
            row["whole_count_test_ba"] - row["rate_normalized_test_ba"]
        )

        for fraction in TOP_FRACTIONS:
            for method in ("frozen_mean_replacement", "retrained_without_neurons"):
                gg = groups[
                    (groups["case"] == case)
                    & np.isclose(groups["fraction_removed"], fraction)
                    & (groups["method"] == method)
                ]
                tag = int(round(100 * fraction))
                row[f"{method}_remove_top{tag}_test_delta_ba_pp"] = float(
                    gg["test_delta_ba_pp_vs_full"].mean()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(
    config: exp16_2.Config,
    *,
    cases: tuple[str, ...],
    seeds: tuple[int, ...],
    output_dir: Path | None,
) -> dict[str, Any]:
    p, lock, arrays = exp16._load_core(config)
    diagnostic_root = (
        output_dir.resolve()
        if output_dir is not None
        else config.results_dir / "persistent_neuron_diagnostic"
    )
    figure_root = diagnostic_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    activity_rows: list[dict[str, Any]] = []
    layer_summary_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for case in cases:
            print(f"Analyzing {case} seed{seed}", flush=True)
            traces = _load_layer_traces(config.results_dir, case, seed)

            run_activity_rows: list[dict[str, Any]] = []
            for layer in ("L1", "L2"):
                rows, summary = _layer_activity_rows(
                    case, seed, layer, traces, arrays
                )
                activity_rows.extend(rows)
                run_activity_rows.extend(rows)
                layer_summary_rows.append(summary)

            activity_frame = pd.DataFrame(run_activity_rows)
            u_rows, g_rows, p_rows = _l2_utility_rows(
                case, seed, traces, arrays, p, activity_frame
            )
            utility_rows.extend(u_rows)
            group_rows.extend(g_rows)
            probe_rows.extend(p_rows)

            all_activity = pd.DataFrame(activity_rows)
            all_utility = pd.DataFrame(utility_rows)
            all_groups = pd.DataFrame(group_rows)
            all_probes = pd.DataFrame(probe_rows)

            _plot_activity_distribution(
                case,
                seed,
                all_activity,
                figure_root / f"{case}__seed{seed}__activity_distribution.png",
            )
            _plot_occupancy_utility(
                case,
                seed,
                all_utility,
                figure_root / f"{case}__seed{seed}__occupancy_vs_utility.png",
            )
            _plot_train_test_utility(
                case,
                seed,
                all_utility,
                figure_root / f"{case}__seed{seed}__train_vs_test_utility.png",
            )
            _plot_group_ablation(
                case,
                seed,
                all_groups,
                all_probes,
                figure_root / f"{case}__seed{seed}__group_ablation.png",
            )

    activity_frame = pd.DataFrame(activity_rows)
    layer_summary_frame = pd.DataFrame(layer_summary_rows)
    utility_frame = pd.DataFrame(utility_rows)
    group_frame = pd.DataFrame(group_rows)
    probe_frame = pd.DataFrame(probe_rows)
    case_summary = _case_summary(
        layer_summary_frame, utility_frame, group_frame, probe_frame
    )

    activity_frame.to_csv(diagnostic_root / "layer_neuron_activity.csv", index=False)
    layer_summary_frame.to_csv(diagnostic_root / "layer_summary.csv", index=False)
    utility_frame.to_csv(diagnostic_root / "l2_neuron_utility.csv", index=False)
    group_frame.to_csv(diagnostic_root / "topk_group_ablation.csv", index=False)
    probe_frame.to_csv(diagnostic_root / "probe_controls.csv", index=False)
    case_summary.to_csv(diagnostic_root / "case_summary.csv", index=False)

    manifest = {
        "experiment_id": exp16_2.EXPERIMENT_ID,
        "protocol_version": exp16_2.PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "cases": list(cases),
        "seeds": list(seeds),
        "analysis_contract": {
            "snn_retrained": False,
            "primary_layer_for_utility": "L2",
            "primary_probe": "WholeCount no-bias",
            "single_neuron_ablation": (
                "frozen probe; replace one standardized feature by the "
                "training-set mean feature value"
            ),
            "topk_ranking": "descending train mean occupancy",
            "topk_fractions": list(TOP_FRACTIONS),
            "topk_controls": [
                "frozen mean replacement",
                "retrained WholeCount probe after feature removal",
            ],
            "duration_controls": [
                "count-duration correlation per neuron",
                "rate-normalized probe",
                "duration-only affine probe",
            ],
            "transfer_controls": [
                "train/val/test frozen ablation utility",
                "class eta-squared",
                "user eta-squared",
            ],
        },
        "outputs": {
            "layer_neuron_activity": "layer_neuron_activity.csv",
            "layer_summary": "layer_summary.csv",
            "l2_neuron_utility": "l2_neuron_utility.csv",
            "topk_group_ablation": "topk_group_ablation.csv",
            "probe_controls": "probe_controls.csv",
            "case_summary": "case_summary.csv",
            "figures": "figures/",
        },
    }
    (diagnostic_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--cases",
        type=_parse_cases,
        default=DEFAULT_CASES,
        help="Comma-separated Exp16.2 cases; default: all formal cases",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=tuple(exp16_2.FORMAL_SEEDS),
        help="Comma-separated seeds; default: 11,23,37",
    )
    args = parser.parse_args()

    config = _config(args.results, args.core_results)
    manifest = run_analysis(
        config,
        cases=args.cases,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )
    root = args.output_dir or (config.results_dir / "persistent_neuron_diagnostic")
    print(f"Persistent-neuron diagnostic written to {root.resolve()}")
    print(json.dumps(manifest["analysis_contract"], indent=2))


if __name__ == "__main__":
    main()
