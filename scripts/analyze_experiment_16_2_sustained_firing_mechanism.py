#!/usr/bin/env python3
"""Exp16.2 sustained-firing mechanism decomposition.

Artifact/checkpoint diagnostic only: this script does not retrain the SNN.

It decomposes L2 sustained firing into:
1. tau-group dependence;
2. raw L1->L2 drive u_t;
3. accumulated L2 synaptic current I_t;
4. post-reset membrane V_t / soft-reset residual;
5. zero-input free decay after the valid sequence;
6. inference-only dynamics counterfactuals with the *same learned drive*:
   - original unnormalized synapse + soft reset;
   - normalized synapse ((1-alpha) drive) + soft reset;
   - unnormalized synapse + hard reset;
   - normalized synapse + hard reset.

The learned network is feed-forward, and the Exp16.2 gate is z-only, so L2
counterfactual replay can hold the L1-derived drive fixed while changing only
L2 dynamics. This isolates whether the sustained tail comes primarily from
long-tau unnormalized current accumulation or from the reset rule.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from core_benchmark_v1.model import BenchmarkNet, lif_step
from core_benchmark_v1.protocol import Protocol
from scripts import experiment_16_2_matched_budget_selective_write as exp16_2
from scripts import experiment_16_prefix_supervised_selective_memory as exp16


DEFAULT_CASES = ("C0", "GZ0", "S90", "G90", "S70", "G70", "S50", "G50")
COUNTERFACTUALS = (
    "original",
    "normalized_synapse",
    "hard_reset",
    "normalized_hard_reset",
)
FREE_DECAY_MODES = ("network_zero_input", "l2_zero_drive")


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


def _spec_map(config: exp16_2.Config) -> dict[str, exp16_2.ExpSpec]:
    selected = exp16_2._selected_lambda_by_rho(config)
    return {spec.key: spec for spec in exp16_2.formal_specs(selected)}


def _load_model(
    config: exp16_2.Config,
    spec: exp16_2.ExpSpec,
    p: Protocol,
) -> torch.nn.Module:
    checkpoint = config.results_dir / "runs" / spec.key / "checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = exp16_2._make_model(spec, p)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def _gate_value(model: torch.nn.Module, s1: torch.Tensor) -> torch.Tensor:
    if not isinstance(model, exp16_2.MatchedBudgetNet):
        return s1.new_ones(s1.shape[0])
    if model.gate_kind == "static":
        if model.rho is None:
            raise ValueError("Static gate requires rho")
        return s1.new_full((s1.shape[0],), float(model.rho))
    return torch.sigmoid(model.gate_input(s1).squeeze(-1) + model.gate_bias)


def _stack_time(values: list[torch.Tensor], steps: int) -> torch.Tensor:
    out = torch.stack(values, dim=1)
    if out.shape[1] == steps:
        return out
    pad_shape = list(out.shape)
    pad_shape[1] = steps - out.shape[1]
    return torch.cat((out, out.new_zeros(pad_shape)), dim=1)


def _trace_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    lengths: torch.Tensor,
    p: Protocol,
    *,
    verify: bool,
) -> dict[str, Any]:
    """Exact forward replay with additional L2 mechanism traces."""
    if len(model.layers) != 2:
        raise ValueError("Mechanism diagnostic expects the two-layer Exp16.2 backbone")

    batch, steps, _ = x.shape
    syn0 = x.new_zeros(batch, p.width)
    syn1 = x.new_zeros(batch, p.width)
    mem0 = x.new_zeros(batch, p.width)
    mem1 = x.new_zeros(batch, p.width)

    s1_trace: list[torch.Tensor] = []
    s2_trace: list[torch.Tensor] = []
    raw_drive_trace: list[torch.Tensor] = []
    actual_drive_trace: list[torch.Tensor] = []
    gate_trace: list[torch.Tensor] = []
    current_trace: list[torch.Tensor] = []
    pre_trace: list[torch.Tensor] = []
    membrane_trace: list[torch.Tensor] = []

    max_valid = int(lengths.max().item())
    with torch.no_grad():
        for t in range(max_valid):
            active = (t < lengths)[:, None]
            cur = x[:, t] * active

            candidate0 = model.alpha_0 * syn0 + model.layers[0](cur)
            syn0 = torch.where(active, candidate0, syn0)
            s1, new_mem0, _ = lif_step(
                syn0,
                mem0,
                model.betas[0],
                p.threshold,
                p.surrogate_slope,
            )
            mem0 = torch.where(active, new_mem0, mem0)
            s1 = s1 * active

            raw_drive = model.layers[1](s1)
            gate = _gate_value(model, s1)
            actual_drive = gate[:, None] * raw_drive
            candidate1 = model.alpha_1 * syn1 + actual_drive
            syn1 = torch.where(active, candidate1, syn1)
            s2, new_mem1, pre1 = lif_step(
                syn1,
                mem1,
                model.betas[1],
                p.threshold,
                p.surrogate_slope,
            )
            mem1 = torch.where(active, new_mem1, mem1)
            s2 = s2 * active

            s1_trace.append(s1)
            s2_trace.append(s2)
            raw_drive_trace.append(raw_drive * active)
            actual_drive_trace.append(actual_drive * active)
            gate_trace.append(gate * active.squeeze(1))
            current_trace.append(syn1 * active)
            pre_trace.append(pre1 * active)
            membrane_trace.append(mem1 * active)

        result = {
            "s1": _stack_time(s1_trace, steps),
            "s2": _stack_time(s2_trace, steps),
            "raw_drive": _stack_time(raw_drive_trace, steps),
            "actual_drive": _stack_time(actual_drive_trace, steps),
            "gate": _stack_time(gate_trace, steps),
            "current": _stack_time(current_trace, steps),
            "pre": _stack_time(pre_trace, steps),
            "membrane": _stack_time(membrane_trace, steps),
            "end_syn0": syn0.clone(),
            "end_syn1": syn1.clone(),
            "end_mem0": mem0.clone(),
            "end_mem1": mem1.clone(),
        }

        if verify:
            official = model(x, lengths)
            if not torch.equal(result["s1"], official["spike"][0]):
                raise AssertionError("Instrumented L1 spikes differ from model.forward")
            if not torch.equal(result["s2"], official["spike"][1]):
                raise AssertionError("Instrumented L2 spikes differ from model.forward")
            if not torch.allclose(result["end_syn0"], official["final_syn"][0], atol=0, rtol=0):
                raise AssertionError("Instrumented L1 final synapse differs from model.forward")
            if not torch.allclose(result["end_syn1"], official["final_syn"][1], atol=0, rtol=0):
                raise AssertionError("Instrumented L2 final synapse differs from model.forward")
            if not torch.allclose(result["end_mem0"], official["final_mem"][0], atol=0, rtol=0):
                raise AssertionError("Instrumented L1 final membrane differs from model.forward")
            if not torch.allclose(result["end_mem1"], official["final_mem"][1], atol=0, rtol=0):
                raise AssertionError("Instrumented L2 final membrane differs from model.forward")

    return result


def _replay_l2(
    actual_drive: torch.Tensor,
    lengths: torch.Tensor,
    alpha: torch.Tensor,
    beta: float,
    threshold: float,
    slope: float,
    condition: str,
) -> dict[str, torch.Tensor]:
    if condition not in COUNTERFACTUALS:
        raise ValueError(condition)
    normalized = condition in {"normalized_synapse", "normalized_hard_reset"}
    hard_reset = condition in {"hard_reset", "normalized_hard_reset"}

    batch, steps, width = actual_drive.shape
    syn = actual_drive.new_zeros(batch, width)
    mem = actual_drive.new_zeros(batch, width)
    spikes: list[torch.Tensor] = []
    currents: list[torch.Tensor] = []
    membranes: list[torch.Tensor] = []

    with torch.no_grad():
        for t in range(int(lengths.max().item())):
            active = (t < lengths)[:, None]
            drive = actual_drive[:, t]
            if normalized:
                drive = (1.0 - alpha) * drive
            candidate = alpha * syn + drive
            syn = torch.where(active, candidate, syn)
            pre = beta * mem + syn
            spike = (pre >= threshold).to(pre.dtype) * active
            if hard_reset:
                post = torch.where(spike.bool(), torch.zeros_like(pre), pre)
            else:
                post = pre - threshold * spike
            mem = torch.where(active, post, mem)
            spikes.append(spike)
            currents.append(syn * active)
            membranes.append(mem * active)

    return {
        "spike": _stack_time(spikes, steps),
        "current": _stack_time(currents, steps),
        "membrane": _stack_time(membranes, steps),
    }


def _free_decay_network(
    model: torch.nn.Module,
    end_syn0: torch.Tensor,
    end_mem0: torch.Tensor,
    end_syn1: torch.Tensor,
    end_mem1: torch.Tensor,
    p: Protocol,
    tail_steps: int,
) -> torch.Tensor:
    """Continue the full network with zero sensor input and no valid mask."""
    syn0 = end_syn0.clone()
    mem0 = end_mem0.clone()
    syn1 = end_syn1.clone()
    mem1 = end_mem1.clone()
    batch = syn0.shape[0]
    zeros = syn0.new_zeros(batch, p.input_channels)
    spikes: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(tail_steps):
            syn0 = model.alpha_0 * syn0 + model.layers[0](zeros)
            s1, mem0, _ = lif_step(
                syn0,
                mem0,
                model.betas[0],
                p.threshold,
                p.surrogate_slope,
            )
            raw_drive = model.layers[1](s1)
            gate = _gate_value(model, s1)
            syn1 = model.alpha_1 * syn1 + gate[:, None] * raw_drive
            s2, mem1, _ = lif_step(
                syn1,
                mem1,
                model.betas[1],
                p.threshold,
                p.surrogate_slope,
            )
            spikes.append(s2)
    return torch.stack(spikes, dim=1)


def _free_decay_l2_zero_drive(
    end_syn1: torch.Tensor,
    end_mem1: torch.Tensor,
    alpha: torch.Tensor,
    beta: float,
    threshold: float,
    slope: float,
    tail_steps: int,
) -> torch.Tensor:
    """Continue L2 alone with its input drive clamped exactly to zero."""
    syn = end_syn1.clone()
    mem = end_mem1.clone()
    spikes: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(tail_steps):
            syn = alpha * syn
            spike, mem, _ = lif_step(syn, mem, beta, threshold, slope)
            spikes.append(spike)
    return torch.stack(spikes, dim=1)


def _valid_sample_mean(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    out = np.zeros((len(values), values.shape[2]), dtype=np.float64)
    for i, length in enumerate(lengths.astype(int).tolist()):
        out[i] = values[i, :length].mean(axis=0)
    return out


def _valid_sample_rms(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    out = np.zeros((len(values), values.shape[2]), dtype=np.float64)
    for i, length in enumerate(lengths.astype(int).tolist()):
        out[i] = np.sqrt(np.mean(np.square(values[i, :length]), axis=0))
    return out


def _valid_sample_fraction_positive(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    out = np.zeros((len(values), values.shape[2]), dtype=np.float64)
    for i, length in enumerate(lengths.astype(int).tolist()):
        out[i] = (values[i, :length] > 0).mean(axis=0)
    return out


def _valid_occupancy(spikes: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    out = np.zeros((len(spikes), spikes.shape[2]), dtype=np.float64)
    for i, length in enumerate(lengths.astype(int).tolist()):
        out[i] = spikes[i, :length].mean(axis=0)
    return out


def _longest_runs(spikes: np.ndarray, lengths: np.ndarray | None = None) -> np.ndarray:
    n_samples, steps, width = spikes.shape
    out = np.zeros((n_samples, width), dtype=np.int32)
    if lengths is None:
        lengths = np.full(n_samples, steps, dtype=np.int32)
    for i, length in enumerate(lengths.astype(int).tolist()):
        current = np.zeros(width, dtype=np.int32)
        best = np.zeros(width, dtype=np.int32)
        for row in spikes[i, :length]:
            current = (current + 1) * row.astype(np.int32)
            best = np.maximum(best, current)
        out[i] = best
    return out


def _conditional_post_spike_membrane(
    membrane: np.ndarray,
    spikes: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    result = np.full((len(spikes), spikes.shape[2]), np.nan, dtype=np.float64)
    for i, length in enumerate(lengths.astype(int).tolist()):
        s = spikes[i, :length].astype(bool)
        v = membrane[i, :length]
        counts = s.sum(axis=0)
        sums = (v * s).sum(axis=0)
        valid = counts > 0
        result[i, valid] = sums[valid] / counts[valid]
    return result


def _tail_last_spike_step(spikes: np.ndarray) -> np.ndarray:
    """Last tail spike index (1-based); zero means no tail spike."""
    n_samples, steps, width = spikes.shape
    out = np.zeros((n_samples, width), dtype=np.int32)
    for t in range(steps):
        out[spikes[:, t].astype(bool)] = t + 1
    return out


def _tau_table(alpha: np.ndarray, fs: float) -> pd.DataFrame:
    dt_ms = 1000.0 / fs
    unique = sorted(float(v) for v in np.unique(np.round(alpha, decimals=8)))
    rows = []
    labels = ("short", "medium", "long")
    for idx, value in enumerate(unique):
        tau_ms = -dt_ms / math.log(value)
        label = labels[idx] if len(unique) == 3 else f"group{idx}"
        rows.append(
            {
                "alpha": value,
                "tau_ms": tau_ms,
                "tau_group": label,
            }
        )
    return pd.DataFrame(rows)


def _neuron_group_metadata(model: torch.nn.Module, p: Protocol) -> pd.DataFrame:
    alpha = model.alpha_1.detach().cpu().numpy().astype(np.float64)
    table = _tau_table(alpha, p.fs)
    rows = []
    for neuron, value in enumerate(alpha.tolist()):
        matched = table.iloc[int(np.argmin(np.abs(table["alpha"].to_numpy() - value)))]
        rows.append(
            {
                "neuron": neuron,
                "alpha": float(value),
                "tau_ms": float(matched["tau_ms"]),
                "tau_group": str(matched["tau_group"]),
                "dc_gain": float(1.0 / (1.0 - value)),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_neuron_rows(
    case: str,
    seed: int,
    split: str,
    metadata: pd.DataFrame,
    sample_metrics: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for neuron in range(len(metadata)):
        row: dict[str, Any] = {
            "case": case,
            "seed": seed,
            "split": split,
            **metadata.iloc[neuron].to_dict(),
        }
        for name, values in sample_metrics.items():
            col = values[:, neuron]
            finite = col[np.isfinite(col)]
            row[f"mean_{name}"] = float(finite.mean()) if len(finite) else float("nan")
            row[f"median_{name}"] = float(np.median(finite)) if len(finite) else float("nan")
            row[f"p90_{name}"] = float(np.quantile(finite, 0.90)) if len(finite) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _counterfactual_rows(
    case: str,
    seed: int,
    split: str,
    metadata: pd.DataFrame,
    condition: str,
    occupancy: np.ndarray,
    longest: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for neuron in range(len(metadata)):
        rows.append(
            {
                "case": case,
                "seed": seed,
                "split": split,
                "condition": condition,
                **metadata.iloc[neuron].to_dict(),
                "mean_occupancy": float(occupancy[:, neuron].mean()),
                "p90_occupancy": float(np.quantile(occupancy[:, neuron], 0.90)),
                "mean_longest_run": float(longest[:, neuron].mean()),
                "p90_longest_run": float(np.quantile(longest[:, neuron], 0.90)),
            }
        )
    return pd.DataFrame(rows)


def _free_decay_rows(
    case: str,
    seed: int,
    split: str,
    metadata: pd.DataFrame,
    mode: str,
    spikes: np.ndarray,
    fs: float,
) -> pd.DataFrame:
    counts = spikes.sum(axis=1)
    occupancy = spikes.mean(axis=1)
    longest = _longest_runs(spikes)
    last_step = _tail_last_spike_step(spikes)
    thresholds_ms = (100, 250, 500)
    rows = []
    for neuron in range(len(metadata)):
        last_ms = last_step[:, neuron] * (1000.0 / fs)
        positive_last = last_ms[last_ms > 0]
        row: dict[str, Any] = {
            "case": case,
            "seed": seed,
            "split": split,
            "mode": mode,
            **metadata.iloc[neuron].to_dict(),
            "mean_tail_spike_count": float(counts[:, neuron].mean()),
            "mean_tail_occupancy": float(occupancy[:, neuron].mean()),
            "mean_tail_longest_run": float(longest[:, neuron].mean()),
            "fraction_samples_any_tail_spike": float((counts[:, neuron] > 0).mean()),
            "mean_last_spike_ms_given_any": (
                float(positive_last.mean()) if len(positive_last) else 0.0
            ),
            "p90_last_spike_ms_given_any": (
                float(np.quantile(positive_last, 0.90)) if len(positive_last) else 0.0
            ),
        }
        for threshold_ms in thresholds_ms:
            row[f"fraction_samples_spike_at_or_after_{threshold_ms}ms"] = float(
                (last_ms >= threshold_ms).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _group_summary(neurons: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        col
        for col in neurons.columns
        if col.startswith("mean_") or col.startswith("p90_")
    ]
    grouped = (
        neurons.groupby(["case", "seed", "split", "tau_group", "alpha", "tau_ms", "dc_gain"])[numeric]
        .mean()
        .reset_index()
    )
    grouped["fraction_neurons_mean_occupancy_gt_0p5"] = (
        neurons.assign(flag=neurons["mean_occupancy"] > 0.5)
        .groupby(["case", "seed", "split", "tau_group"])["flag"]
        .mean()
        .reset_index(drop=True)
    )
    return grouped


def _counterfactual_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    out = (
        frame.groupby(["case", "seed", "split", "condition", "tau_group"])[
            ["mean_occupancy", "mean_longest_run"]
        ]
        .mean()
        .reset_index()
    )
    fractions = (
        frame.assign(flag=frame["mean_occupancy"] > 0.5)
        .groupby(["case", "seed", "split", "condition", "tau_group"])["flag"]
        .mean()
        .reset_index(name="fraction_neurons_mean_occupancy_gt_0p5")
    )
    return out.merge(
        fractions,
        on=["case", "seed", "split", "condition", "tau_group"],
        how="left",
    )


def _free_decay_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "mean_tail_spike_count",
        "mean_tail_occupancy",
        "mean_tail_longest_run",
        "fraction_samples_any_tail_spike",
        "mean_last_spike_ms_given_any",
        "p90_last_spike_ms_given_any",
        "fraction_samples_spike_at_or_after_100ms",
        "fraction_samples_spike_at_or_after_250ms",
        "fraction_samples_spike_at_or_after_500ms",
    ]
    return (
        frame.groupby(["case", "seed", "split", "mode", "tau_group"])[value_cols]
        .mean()
        .reset_index()
    )


def _plot_tau_persistence(summary: pd.DataFrame, case: str, seed: int, path: Path) -> None:
    chosen = summary[(summary.case == case) & (summary.seed == seed)].copy()
    order = ["short", "medium", "long"]
    chosen["tau_group"] = pd.Categorical(chosen["tau_group"], categories=order, ordered=True)
    chosen = chosen.sort_values("tau_group")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(chosen["tau_group"].astype(str), chosen["mean_occupancy"])
    ax.set_ylabel("Mean L2 valid-window occupancy")
    ax.set_xlabel("L2 synaptic tau group")
    ax.set_title(f"{case} seed{seed}: sustained firing vs tau")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_drive_current(summary: pd.DataFrame, case: str, seed: int, path: Path) -> None:
    chosen = summary[(summary.case == case) & (summary.seed == seed)].copy()
    order = ["short", "medium", "long"]
    chosen["tau_group"] = pd.Categorical(chosen["tau_group"], categories=order, ordered=True)
    chosen = chosen.sort_values("tau_group")
    x = np.arange(len(chosen))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, chosen["mean_actual_drive"], marker="o", label="actual drive")
    ax.plot(x, chosen["mean_current"], marker="o", label="synaptic current")
    ax.plot(x, chosen["mean_predicted_dc_current"], marker="o", label="drive/(1-alpha)")
    ax.set_xticks(x, chosen["tau_group"].astype(str))
    ax.set_ylabel("Mean value")
    ax.set_xlabel("L2 synaptic tau group")
    ax.set_title(f"{case} seed{seed}: drive vs accumulated current")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_counterfactual(summary: pd.DataFrame, case: str, seed: int, path: Path) -> None:
    chosen = summary[(summary.case == case) & (summary.seed == seed)]
    order = ["short", "medium", "long"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order), dtype=float)
    for condition in COUNTERFACTUALS:
        rows = chosen[chosen.condition == condition].set_index("tau_group")
        y = [float(rows.loc[group, "mean_occupancy"]) for group in order]
        ax.plot(x, y, marker="o", label=condition)
    ax.set_xticks(x, order)
    ax.set_xlabel("L2 synaptic tau group")
    ax.set_ylabel("Mean valid-window occupancy")
    ax.set_title(f"{case} seed{seed}: inference-only dynamics counterfactual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_free_decay(summary: pd.DataFrame, case: str, seed: int, path: Path) -> None:
    chosen = summary[(summary.case == case) & (summary.seed == seed)]
    order = ["short", "medium", "long"]
    x = np.arange(len(order), dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in FREE_DECAY_MODES:
        rows = chosen[chosen["mode"] == mode].set_index("tau_group")
        y = [float(rows.loc[group, "fraction_samples_spike_at_or_after_250ms"]) for group in order]
        ax.plot(x, y, marker="o", label=mode)
    ax.set_xticks(x, order)
    ax.set_xlabel("L2 synaptic tau group")
    ax.set_ylabel("Fraction of samples with spike >=250 ms after valid end")
    ax.set_title(f"{case} seed{seed}: zero-input free decay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_analysis(
    config: exp16_2.Config,
    *,
    cases: tuple[str, ...],
    seeds: tuple[int, ...],
    split: str,
    tail_ms: float,
    output_dir: Path | None,
) -> dict[str, Any]:
    p, lock, arrays = exp16._load_core(config)
    specs = _spec_map(config)
    root = (
        output_dir.resolve()
        if output_dir is not None
        else config.results_dir / "sustained_firing_mechanism"
    )
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    tail_steps = max(1, int(math.ceil(tail_ms * p.fs / 1000.0)))

    neuron_frames: list[pd.DataFrame] = []
    cf_frames: list[pd.DataFrame] = []
    decay_frames: list[pd.DataFrame] = []

    for seed in seeds:
        for case in cases:
            key = f"{case}__seed{seed}"
            if key not in specs:
                raise KeyError(key)
            print(f"Analyzing {key}", flush=True)
            model = _load_model(config, specs[key], p)
            metadata = _neuron_group_metadata(model, p)

            per_batch_metrics: dict[str, list[np.ndarray]] = {
                "occupancy": [],
                "longest_run": [],
                "raw_drive": [],
                "raw_drive_rms": [],
                "raw_drive_positive_fraction": [],
                "actual_drive": [],
                "actual_drive_rms": [],
                "actual_drive_positive_fraction": [],
                "current": [],
                "current_above_threshold_fraction": [],
                "pre_reset": [],
                "membrane": [],
                "post_spike_membrane": [],
                "predicted_dc_current": [],
            }
            cf_accum: dict[str, dict[str, list[np.ndarray]]] = {
                condition: {"occupancy": [], "longest": []}
                for condition in COUNTERFACTUALS
            }
            decay_accum: dict[str, list[np.ndarray]] = {
                mode: [] for mode in FREE_DECAY_MODES
            }

            for batch_index, (x, _, lengths) in enumerate(
                exp16.loader(arrays, split, p, seed)
            ):
                trace = _trace_batch(
                    model,
                    x,
                    lengths,
                    p,
                    verify=(batch_index == 0),
                )
                lengths_np = lengths.numpy()
                spike_np = trace["s2"].numpy().astype(np.uint8)
                raw_np = trace["raw_drive"].numpy()
                drive_np = trace["actual_drive"].numpy()
                current_np = trace["current"].numpy()
                pre_np = trace["pre"].numpy()
                membrane_np = trace["membrane"].numpy()

                occupancy = _valid_occupancy(spike_np, lengths_np)
                longest = _longest_runs(spike_np, lengths_np)
                mean_drive = _valid_sample_mean(drive_np, lengths_np)
                alpha_np = model.alpha_1.detach().cpu().numpy()[None, :]
                predicted_dc = mean_drive / (1.0 - alpha_np)

                per_batch_metrics["occupancy"].append(occupancy)
                per_batch_metrics["longest_run"].append(longest.astype(np.float64))
                per_batch_metrics["raw_drive"].append(_valid_sample_mean(raw_np, lengths_np))
                per_batch_metrics["raw_drive_rms"].append(_valid_sample_rms(raw_np, lengths_np))
                per_batch_metrics["raw_drive_positive_fraction"].append(
                    _valid_sample_fraction_positive(raw_np, lengths_np)
                )
                per_batch_metrics["actual_drive"].append(mean_drive)
                per_batch_metrics["actual_drive_rms"].append(_valid_sample_rms(drive_np, lengths_np))
                per_batch_metrics["actual_drive_positive_fraction"].append(
                    _valid_sample_fraction_positive(drive_np, lengths_np)
                )
                per_batch_metrics["current"].append(_valid_sample_mean(current_np, lengths_np))
                per_batch_metrics["current_above_threshold_fraction"].append(
                    _valid_sample_mean((current_np > p.threshold).astype(np.float32), lengths_np)
                )
                per_batch_metrics["pre_reset"].append(_valid_sample_mean(pre_np, lengths_np))
                per_batch_metrics["membrane"].append(_valid_sample_mean(membrane_np, lengths_np))
                per_batch_metrics["post_spike_membrane"].append(
                    _conditional_post_spike_membrane(membrane_np, spike_np, lengths_np)
                )
                per_batch_metrics["predicted_dc_current"].append(predicted_dc)

                for condition in COUNTERFACTUALS:
                    replay = _replay_l2(
                        trace["actual_drive"],
                        lengths,
                        model.alpha_1,
                        model.betas[1],
                        p.threshold,
                        p.surrogate_slope,
                        condition,
                    )
                    replay_spike = replay["spike"].numpy().astype(np.uint8)
                    if condition == "original" and not np.array_equal(replay_spike, spike_np):
                        raise AssertionError(
                            f"{key}: original fixed-drive replay differs from traced L2"
                        )
                    cf_accum[condition]["occupancy"].append(
                        _valid_occupancy(replay_spike, lengths_np)
                    )
                    cf_accum[condition]["longest"].append(
                        _longest_runs(replay_spike, lengths_np).astype(np.float64)
                    )

                network_tail = _free_decay_network(
                    model,
                    trace["end_syn0"],
                    trace["end_mem0"],
                    trace["end_syn1"],
                    trace["end_mem1"],
                    p,
                    tail_steps,
                )
                l2_tail = _free_decay_l2_zero_drive(
                    trace["end_syn1"],
                    trace["end_mem1"],
                    model.alpha_1,
                    model.betas[1],
                    p.threshold,
                    p.surrogate_slope,
                    tail_steps,
                )
                decay_accum["network_zero_input"].append(
                    network_tail.numpy().astype(np.uint8)
                )
                decay_accum["l2_zero_drive"].append(
                    l2_tail.numpy().astype(np.uint8)
                )

            sample_metrics = {
                name: np.concatenate(parts, axis=0)
                for name, parts in per_batch_metrics.items()
            }
            neuron_frame = _aggregate_neuron_rows(
                case, seed, split, metadata, sample_metrics
            )
            neuron_frames.append(neuron_frame)

            for condition in COUNTERFACTUALS:
                cf_frames.append(
                    _counterfactual_rows(
                        case,
                        seed,
                        split,
                        metadata,
                        condition,
                        np.concatenate(cf_accum[condition]["occupancy"], axis=0),
                        np.concatenate(cf_accum[condition]["longest"], axis=0),
                    )
                )

            for mode in FREE_DECAY_MODES:
                decay_frames.append(
                    _free_decay_rows(
                        case,
                        seed,
                        split,
                        metadata,
                        mode,
                        np.concatenate(decay_accum[mode], axis=0),
                        p.fs,
                    )
                )

    neurons = pd.concat(neuron_frames, ignore_index=True)
    counterfactual = pd.concat(cf_frames, ignore_index=True)
    free_decay = pd.concat(decay_frames, ignore_index=True)

    tau_summary = _group_summary(neurons)
    cf_summary = _counterfactual_group_summary(counterfactual)
    decay_summary = _free_decay_group_summary(free_decay)

    root.mkdir(parents=True, exist_ok=True)
    neurons.to_csv(root / "l2_neuron_mechanism.csv", index=False)
    tau_summary.to_csv(root / "tau_group_summary.csv", index=False)
    counterfactual.to_csv(root / "counterfactual_neuron.csv", index=False)
    cf_summary.to_csv(root / "counterfactual_tau_summary.csv", index=False)
    free_decay.to_csv(root / "free_decay_neuron.csv", index=False)
    decay_summary.to_csv(root / "free_decay_tau_summary.csv", index=False)

    for seed in seeds:
        for case in cases:
            _plot_tau_persistence(
                tau_summary,
                case,
                seed,
                figures / f"{case}__seed{seed}__tau_persistence.png",
            )
            _plot_drive_current(
                tau_summary,
                case,
                seed,
                figures / f"{case}__seed{seed}__drive_current.png",
            )
            _plot_counterfactual(
                cf_summary,
                case,
                seed,
                figures / f"{case}__seed{seed}__counterfactual.png",
            )
            _plot_free_decay(
                decay_summary,
                case,
                seed,
                figures / f"{case}__seed{seed}__free_decay.png",
            )

    manifest = {
        "experiment_id": exp16_2.EXPERIMENT_ID,
        "protocol_version": exp16_2.PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "split": split,
        "cases": list(cases),
        "seeds": list(seeds),
        "tail_ms_requested": tail_ms,
        "tail_steps": tail_steps,
        "tail_ms_actual": tail_steps * 1000.0 / p.fs,
        "mechanism_contract": {
            "snn_retrained": False,
            "instrumented_forward_verified_against_model_forward": True,
            "l2_equation": "I_t = alpha I_{t-1} + g_t u_t",
            "soft_reset_equation": "V_t = beta V_{t-1} + I_t - threshold * spike_t",
            "normalized_counterfactual": (
                "I_t = alpha I_{t-1} + (1-alpha) g_t u_t; "
                "learned L1-derived drive otherwise fixed"
            ),
            "hard_reset_counterfactual": (
                "same learned L1-derived drive and synaptic current equation; "
                "set post-spike membrane to zero"
            ),
            "network_zero_input_tail": (
                "after each sample valid end, continue both L1 and L2 with sensor input=0"
            ),
            "l2_zero_drive_tail": (
                "after each sample valid end, clamp L2 external drive to exactly zero "
                "and evolve only its existing synaptic/membrane state"
            ),
        },
        "outputs": {
            "l2_neuron_mechanism": "l2_neuron_mechanism.csv",
            "tau_group_summary": "tau_group_summary.csv",
            "counterfactual_neuron": "counterfactual_neuron.csv",
            "counterfactual_tau_summary": "counterfactual_tau_summary.csv",
            "free_decay_neuron": "free_decay_neuron.csv",
            "free_decay_tau_summary": "free_decay_tau_summary.csv",
            "figures": "figures/",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
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
        help="Comma-separated cases; default: all Exp16.2 formal cases",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=tuple(exp16_2.FORMAL_SEEDS),
        help="Comma-separated formal seeds; default: 11,23,37",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split used for mechanism analysis; default: test",
    )
    parser.add_argument(
        "--tail-ms",
        type=float,
        default=1000.0,
        help="Zero-input free-decay continuation in milliseconds; default: 1000",
    )
    args = parser.parse_args()
    if args.tail_ms <= 0:
        parser.error("--tail-ms must be > 0")

    exp16_2.configure_cpu()
    config = _config(args.results, args.core_results)
    manifest = run_analysis(
        config,
        cases=args.cases,
        seeds=args.seeds,
        split=args.split,
        tail_ms=args.tail_ms,
        output_dir=args.output_dir,
    )
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else config.results_dir / "sustained_firing_mechanism"
    )
    print(f"Sustained-firing mechanism diagnostic written to {output_root}")
    print(json.dumps(manifest["mechanism_contract"], indent=2))


if __name__ == "__main__":
    main()
