#!/usr/bin/env python3
"""Exp17.2: diagnose history-input competition in the trained Exp17 SNN.

This is an artifact-only diagnostic extension. It never trains or modifies an
SNN. It replays the finalized Exp17 trajectory checkpoints and decomposes each
layer's factual pre-reset drive into

    M_t = beta * V_{t-1}          membrane carry
    H_t = alpha * I_{t-1}         synaptic-history drive
    N_t = W z_t                   current incoming drive

so that I_t = H_t + N_t and P_t = M_t + H_t + N_t.

The primary question is whether ordinary WCCE optimization increasingly makes
fire/not-fire decisions history-sufficient while reducing the instantaneous
leverage of current incoming evidence. A second, short-horizon intervention
zeros only the L2 incoming drive at one anchor and then restores factual future
L1 spikes, distinguishing instantaneous spike saturation from failure to write
new information into the persistent synaptic state.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from core_benchmark_v1.model import lif_step, valid_mask
from core_benchmark_v1.protocol import Protocol, SPLITS
from core_benchmark_v1.storage import load_torch, save_json
from scripts import experiment_16_prefix_supervised_selective_memory as exp16
from scripts import experiment_17_persistent_pathway_story as exp17


EXPERIMENT_ID = "experiment_17_2_history_input_competition"
PROTOCOL_VERSION = "history_input_competition_v1"
FORMAL_SEEDS = exp17.FORMAL_SEEDS
SOURCE_EXP17_REL = Path(
    "notebooks/artifacts/experiment_17_persistent_pathway_story/"
    "persistent_pathway_story_v1"
)
FIXED_GROUP_EPOCH = 20
PERSISTENT_QUARTILE = 0.25
RELATIVE_BINS = 10
EPS = 1e-8

# Instantaneous replay covers every finalized Exp17 trajectory snapshot and all
# locked splits. The more expensive short-horizon branch is intentionally
# restricted to a small longitudinal grid plus selected/stopped checkpoints.
HORIZON_BASE_EPOCHS = (0, 20, 40, 60, 80)
HORIZON_ROLE_EPOCHS = ("selected_best", "stopped")
HORIZON_STEPS = (1, 2, 4, 8, 16)
HORIZON_SPLITS = ("train", "test")
HORIZON_ANCHOR_STRIDE = 8


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    core_results_dir: Path
    exp17_results_dir: Path


def find_repo_root() -> Path:
    return exp17.find_repo_root()


def default_results_dir(root: Path) -> Path:
    return root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def config_from_args(args: argparse.Namespace) -> Config:
    root = find_repo_root()
    return Config(
        repo_root=root,
        results_dir=(args.results or default_results_dir(root)).resolve(),
        core_results_dir=(args.core_results or root / exp16.CORE_RESULTS_REL).resolve(),
        exp17_results_dir=(args.exp17_results or root / SOURCE_EXP17_REL).resolve(),
    )


def _core(config: Config) -> tuple[Protocol, dict[str, Any], dict[str, np.ndarray]]:
    return exp16._load_core(config)  # Config is deliberately duck-typed.


def _seed_source_dir(config: Config, seed: int) -> Path:
    return config.exp17_results_dir / "trajectory" / f"seed{seed}"


def _load_manifest(config: Config, seed: int) -> dict[str, Any]:
    path = _seed_source_dir(config, seed) / "snapshot_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["seed"]) != seed:
        raise ValueError(f"Exp17 manifest seed mismatch: expected {seed}, got {payload['seed']}")
    return payload


def _snapshot_path(config: Config, seed: int, epoch: int) -> Path:
    return exp17._snapshot_path(_seed_source_dir(config, seed), epoch)


def _load_snapshot_model(
    config: Config,
    seed: int,
    epoch: int,
    p: Protocol,
) -> torch.nn.Module:
    path = _snapshot_path(config, seed, epoch)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = load_torch(path)
    model = exp17._make_trajectory_model(seed, p)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def _stack_pad(values: list[torch.Tensor], steps: int) -> torch.Tensor:
    result = torch.stack(values, dim=1)
    return torch.nn.functional.pad(result, (0, 0, 0, steps - result.shape[1]))


def instrumented_replay(
    model: torch.nn.Module,
    x: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, Any]:
    """Replay BenchmarkNet exactly while exposing factual drive components.

    No state is changed relative to BenchmarkNet.forward(). Counterfactual
    tensors are computed from saved factual pre-step states and never feed back
    into the factual trajectory.
    """
    p = model.protocol
    batch, steps, channels = x.shape
    if channels != p.input_channels or lengths.shape != (batch,):
        raise ValueError("Invalid input/length geometry")
    if (lengths < 1).any() or (lengths > steps).any():
        raise ValueError("Invalid valid lengths")

    syn = [x.new_zeros(batch, p.width) for _ in model.layers]
    mem = [x.new_zeros(batch, p.width) for _ in model.layers]
    spikes: list[list[torch.Tensor]] = [[] for _ in model.layers]
    pre_reset: list[list[torch.Tensor]] = [[] for _ in model.layers]
    component_names = (
        "syn_prev",
        "mem_prev",
        "mem_carry",
        "syn_history",
        "new_drive",
        "syn_state",
        "mem_state",
    )
    components: list[dict[str, list[torch.Tensor]]] = [
        {name: [] for name in component_names} for _ in model.layers
    ]
    raw_active: list[torch.Tensor] = []
    raw_zero_l2_spike: list[torch.Tensor] = []
    evidence: list[torch.Tensor] = []
    auxiliary: list[torch.Tensor] = []

    max_steps = int(lengths.max().item())
    for t in range(max_steps):
        active = (t < lengths)[:, None]
        raw_cur = x[:, t] * active
        raw_active.append(((x[:, t].abs().sum(dim=1) > 0) & active[:, 0]))

        prev_syn = list(syn)
        prev_mem = list(mem)
        cur = raw_cur
        for i, linear in enumerate(model.layers):
            syn_prev = prev_syn[i]
            mem_prev = prev_mem[i]
            mem_carry = model.betas[i] * mem_prev
            syn_history = getattr(model, f"alpha_{i}") * syn_prev
            new_drive = linear(cur)
            candidate = syn_history + new_drive
            next_syn = torch.where(active, candidate, syn_prev)
            spike, next_mem_raw, pre = lif_step(
                next_syn,
                mem_prev,
                model.betas[i],
                p.threshold,
                p.surrogate_slope,
            )
            next_mem = torch.where(active, next_mem_raw, mem_prev)
            cur = spike * active

            spikes[i].append(cur)
            pre_reset[i].append(pre * active)
            components[i]["syn_prev"].append(syn_prev * active)
            components[i]["mem_prev"].append(mem_prev * active)
            components[i]["mem_carry"].append(mem_carry * active)
            components[i]["syn_history"].append(syn_history * active)
            components[i]["new_drive"].append(new_drive * active)
            components[i]["syn_state"].append(next_syn * active)
            components[i]["mem_state"].append(next_mem * active)
            syn[i], mem[i] = next_syn, next_mem

        evidence.append(model.head(cur))
        if model.auxiliary is not None:
            auxiliary.append(model.auxiliary(spikes[0][-1]))

        # Raw-current-input counterfactual for L2. Previous factual states are
        # held fixed, x_t is zeroed only for this timestep, and the resulting
        # L1 spike is propagated through the factual L2 weights.
        if len(model.layers) >= 2:
            syn1_cf = getattr(model, "alpha_0") * prev_syn[0]
            spike1_cf, _, _ = lif_step(
                syn1_cf,
                prev_mem[0],
                model.betas[0],
                p.threshold,
                p.surrogate_slope,
            )
            spike1_cf = spike1_cf * active
            drive2_cf = model.layers[1](spike1_cf)
            syn2_cf = getattr(model, "alpha_1") * prev_syn[1] + drive2_cf
            spike2_cf, _, _ = lif_step(
                syn2_cf,
                prev_mem[1],
                model.betas[1],
                p.threshold,
                p.surrogate_slope,
            )
            raw_zero_l2_spike.append(spike2_cf * active)

    out: dict[str, Any] = {
        "spike": tuple(_stack_pad(values, steps) for values in spikes),
        "pre_reset": tuple(_stack_pad(values, steps) for values in pre_reset),
        "evidence": _stack_pad(evidence, steps),
        "final_syn": tuple(syn),
        "final_mem": tuple(mem),
        "components": tuple(
            {name: _stack_pad(values[name], steps) for name in component_names}
            for values in components
        ),
        "raw_input_active": torch.nn.functional.pad(
            torch.stack(raw_active, dim=1),
            (0, steps - max_steps),
            value=False,
        ),
    }
    if raw_zero_l2_spike:
        out["raw_zero_l2_spike"] = _stack_pad(raw_zero_l2_spike, steps)
    if auxiliary:
        out["auxiliary"] = _stack_pad(auxiliary, steps)
    return out


def _decision_masks(
    full_spike: torch.Tensor,
    no_new_spike: torch.Tensor,
) -> dict[str, torch.Tensor]:
    full = full_spike.bool()
    no_new = no_new_spike.bool()
    return {
        "incoming_flip": full != no_new,
        "history_sufficient": full & no_new,
        "new_triggered": full & ~no_new,
        "new_suppressed": ~full & no_new,
    }


def _quartile_groups(values: np.ndarray) -> dict[int, str]:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    q = max(1, int(math.ceil(PERSISTENT_QUARTILE * len(order))))
    high = set(order[:q].tolist())
    low = set(order[-q:].tolist())
    return {
        neuron: (
            "high25"
            if neuron in high
            else "low25"
            if neuron in low
            else "middle50"
        )
        for neuron in range(len(values))
    }


def _tau_shift_labels(model: torch.nn.Module, layer: int) -> np.ndarray:
    shifts = tuple(int(value) for value in model.run.shifts[layer])
    q, r = divmod(model.protocol.width, len(shifts))
    labels = [
        shift
        for j, shift in enumerate(shifts)
        for _ in range(q + (j < r))
    ]
    return np.asarray(labels, dtype=np.int64)


def _new_layer_accumulator(width: int) -> dict[str, Any]:
    arrays = (
        "spike_count",
        "abs_mem_carry",
        "abs_syn_history",
        "abs_new_drive",
        "syn_dominance",
        "state_dominance",
        "incoming_flip",
        "history_sufficient",
        "new_triggered",
        "new_suppressed",
        "syn_history_flip",
        "mem_carry_flip",
        "history_margin",
        "full_margin",
        "raw_input_flip",
    )
    return {
        "valid_count": 0.0,
        "raw_active_count": 0.0,
        **{name: np.zeros(width, dtype=np.float64) for name in arrays},
    }


def _new_relative_accumulator() -> list[dict[str, float]]:
    return [
        {
            "valid_neuron_count": 0.0,
            "spike_count": 0.0,
            "incoming_flip": 0.0,
            "full_spike_count": 0.0,
            "history_sufficient": 0.0,
            "raw_active_neuron_count": 0.0,
            "raw_input_flip": 0.0,
        }
        for _ in range(RELATIVE_BINS)
    ]


def _relative_bin_masks(lengths: torch.Tensor, steps: int) -> list[torch.Tensor]:
    time = torch.arange(steps, device=lengths.device)[None, :]
    valid = time < lengths[:, None]
    denom = lengths[:, None].clamp_min(1)
    bins = torch.div(time * RELATIVE_BINS, denom, rounding_mode="floor")
    bins = bins.clamp(max=RELATIVE_BINS - 1)
    return [valid & (bins == index) for index in range(RELATIVE_BINS)]


def _accumulate_batch(
    replay: dict[str, Any],
    lengths: torch.Tensor,
    p: Protocol,
    accumulators: list[dict[str, Any]],
    relative: list[list[dict[str, float]]],
) -> None:
    steps = replay["spike"][0].shape[1]
    valid_bt = valid_mask(lengths, steps)
    bin_masks = _relative_bin_masks(lengths, steps)

    for layer, comp in enumerate(replay["components"]):
        full = replay["spike"][layer]
        valid = valid_bt[:, :, None]
        mem_carry = comp["mem_carry"]
        syn_history = comp["syn_history"]
        new_drive = comp["new_drive"]
        syn_state = comp["syn_state"]

        no_new = (mem_carry + syn_history >= p.threshold) & valid
        no_syn_history = (mem_carry + new_drive >= p.threshold) & valid
        no_mem_carry = (syn_history + new_drive >= p.threshold) & valid
        masks = _decision_masks(full, no_new)
        factual = full.bool() & valid

        acc = accumulators[layer]
        acc["valid_count"] += float(valid_bt.sum().item())
        acc["spike_count"] += (full * valid).sum((0, 1)).detach().cpu().numpy()
        acc["abs_mem_carry"] += (mem_carry.abs() * valid).sum((0, 1)).detach().cpu().numpy()
        acc["abs_syn_history"] += (syn_history.abs() * valid).sum((0, 1)).detach().cpu().numpy()
        acc["abs_new_drive"] += (new_drive.abs() * valid).sum((0, 1)).detach().cpu().numpy()

        syn_dom = syn_history.abs() / (syn_history.abs() + new_drive.abs() + EPS)
        state_dom = (mem_carry.abs() + syn_history.abs()) / (
            mem_carry.abs() + syn_history.abs() + new_drive.abs() + EPS
        )
        acc["syn_dominance"] += (syn_dom * valid).sum((0, 1)).detach().cpu().numpy()
        acc["state_dominance"] += (state_dom * valid).sum((0, 1)).detach().cpu().numpy()
        for name in ("incoming_flip", "history_sufficient", "new_triggered", "new_suppressed"):
            acc[name] += (masks[name] & valid).sum((0, 1)).detach().cpu().numpy()
        acc["syn_history_flip"] += (
            (full.bool() != no_syn_history) & valid
        ).sum((0, 1)).detach().cpu().numpy()
        acc["mem_carry_flip"] += (
            (full.bool() != no_mem_carry) & valid
        ).sum((0, 1)).detach().cpu().numpy()
        acc["history_margin"] += (
            (mem_carry + syn_history - p.threshold) * valid
        ).sum((0, 1)).detach().cpu().numpy()
        acc["full_margin"] += (
            (mem_carry + syn_state - syn_history - p.threshold) * valid
        ).sum((0, 1)).detach().cpu().numpy()
        # full_margin above simplifies to mem_carry + syn_state - threshold,
        # while retaining the explicit decomposition in the source expression.

        if layer == 1 and "raw_zero_l2_spike" in replay:
            raw_active = replay["raw_input_active"] & valid_bt
            raw_active_w = raw_active[:, :, None]
            raw_flip = (full.bool() != replay["raw_zero_l2_spike"].bool()) & raw_active_w
            acc["raw_active_count"] += float(raw_active.sum().item())
            acc["raw_input_flip"] += raw_flip.sum((0, 1)).detach().cpu().numpy()
        else:
            raw_active_w = None
            raw_flip = None

        for bin_index, bt_mask in enumerate(bin_masks):
            bin_valid = bt_mask[:, :, None]
            row = relative[layer][bin_index]
            n_valid = float(bin_valid.sum().item()) * p.width
            row["valid_neuron_count"] += n_valid
            row["spike_count"] += float((full * bin_valid).sum().item())
            row["incoming_flip"] += float((masks["incoming_flip"] & bin_valid).sum().item())
            row["full_spike_count"] += float((factual & bin_valid).sum().item())
            row["history_sufficient"] += float(
                (masks["history_sufficient"] & bin_valid).sum().item()
            )
            if layer == 1 and raw_active_w is not None and raw_flip is not None:
                active_bin = raw_active_w & bin_valid
                row["raw_active_neuron_count"] += float(active_bin.sum().item()) * p.width
                row["raw_input_flip"] += float((raw_flip & bin_valid).sum().item())


def _safe_divide(numerator: np.ndarray, denominator: float | np.ndarray) -> np.ndarray:
    denominator_array = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator_array,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator_array > 0,
    )


def _accumulator_rows(
    accumulator: dict[str, Any],
    *,
    seed: int,
    epoch: int,
    roles: list[str],
    split: str,
    layer: int,
    p: Protocol,
    tau_shifts: np.ndarray,
) -> list[dict[str, Any]]:
    valid = float(accumulator["valid_count"])
    spike_count = accumulator["spike_count"]
    raw_active = float(accumulator["raw_active_count"])
    width = len(spike_count)
    firing = _safe_divide(spike_count * p.fs, valid)
    full_spikes = spike_count
    rows: list[dict[str, Any]] = []
    for neuron in range(width):
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "snapshot_roles": ";".join(roles),
                "split": split,
                "layer": f"L{layer + 1}",
                "neuron": neuron,
                "tau_shift": int(tau_shifts[neuron]),
                "firing_rate_hz": float(firing[neuron]),
                "mean_abs_mem_carry": float(accumulator["abs_mem_carry"][neuron] / valid),
                "mean_abs_syn_history": float(accumulator["abs_syn_history"][neuron] / valid),
                "mean_abs_new_drive": float(accumulator["abs_new_drive"][neuron] / valid),
                "mean_syn_history_dominance": float(accumulator["syn_dominance"][neuron] / valid),
                "mean_state_dominance": float(accumulator["state_dominance"][neuron] / valid),
                "incoming_drive_flip_fraction": float(accumulator["incoming_flip"][neuron] / valid),
                "history_sufficient_spike_fraction": (
                    float(accumulator["history_sufficient"][neuron] / full_spikes[neuron])
                    if full_spikes[neuron] > 0
                    else math.nan
                ),
                "new_triggered_spike_fraction": (
                    float(accumulator["new_triggered"][neuron] / full_spikes[neuron])
                    if full_spikes[neuron] > 0
                    else math.nan
                ),
                "new_suppressed_fraction": float(accumulator["new_suppressed"][neuron] / valid),
                "syn_history_flip_fraction": float(accumulator["syn_history_flip"][neuron] / valid),
                "membrane_carry_flip_fraction": float(accumulator["mem_carry_flip"][neuron] / valid),
                "history_only_margin_mean": float(accumulator["history_margin"][neuron] / valid),
                "full_margin_mean": float(accumulator["full_margin"][neuron] / valid),
                "raw_input_flip_given_active": (
                    float(accumulator["raw_input_flip"][neuron] / raw_active)
                    if layer == 1 and raw_active > 0
                    else math.nan
                ),
                "raw_input_active_fraction": (
                    float(raw_active / valid) if layer == 1 and valid > 0 else math.nan
                ),
            }
        )
    return rows


def _relative_rows(
    relative: list[list[dict[str, float]]],
    *,
    seed: int,
    epoch: int,
    roles: list[str],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, bins in enumerate(relative):
        for index, values in enumerate(bins):
            valid = values["valid_neuron_count"]
            full_spikes = values["full_spike_count"]
            raw_valid = values["raw_active_neuron_count"]
            rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "snapshot_roles": ";".join(roles),
                    "split": split,
                    "layer": f"L{layer + 1}",
                    "relative_bin": index,
                    "firing_occupancy": (
                        values["spike_count"] / valid if valid > 0 else math.nan
                    ),
                    "incoming_drive_flip_fraction": (
                        values["incoming_flip"] / valid if valid > 0 else math.nan
                    ),
                    "history_sufficient_spike_fraction": (
                        values["history_sufficient"] / full_spikes
                        if full_spikes > 0
                        else math.nan
                    ),
                    "raw_input_flip_given_active": (
                        values["raw_input_flip"] / raw_valid
                        if raw_valid > 0
                        else math.nan
                    ),
                }
            )
    return rows


SUMMARY_METRICS = (
    "firing_rate_hz",
    "mean_abs_mem_carry",
    "mean_abs_syn_history",
    "mean_abs_new_drive",
    "mean_syn_history_dominance",
    "mean_state_dominance",
    "incoming_drive_flip_fraction",
    "history_sufficient_spike_fraction",
    "new_triggered_spike_fraction",
    "new_suppressed_fraction",
    "syn_history_flip_fraction",
    "membrane_carry_flip_fraction",
    "history_only_margin_mean",
    "full_margin_mean",
    "raw_input_flip_given_active",
)


def _snapshot_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["seed", "epoch", "snapshot_roles", "split", "layer"], sort=True
    ):
        row = dict(zip(("seed", "epoch", "snapshot_roles", "split", "layer"), keys))
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else math.nan
            row[f"{metric}_median"] = float(np.median(finite)) if len(finite) else math.nan
            row[f"{metric}_p25"] = float(np.quantile(finite, 0.25)) if len(finite) else math.nan
            row[f"{metric}_p75"] = float(np.quantile(finite, 0.75)) if len(finite) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _source_neuron_table(config: Config, seed: int) -> pd.DataFrame:
    path = _seed_source_dir(config, seed) / "trajectory_neurons.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "seed",
        "epoch",
        "neuron",
        "train_mean_occupancy",
        "head_weight_norm",
        "persistent_quartile",
        "train_class_eta2",
        "train_user_eta2",
        "val_class_eta2",
        "val_user_eta2",
        "test_class_eta2",
        "test_user_eta2",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Exp17 trajectory_neurons missing columns: {sorted(missing)}")
    return frame


def _fixed_epoch20_groups(source: pd.DataFrame) -> dict[int, str]:
    frame = source[source["epoch"] == FIXED_GROUP_EPOCH].sort_values("neuron")
    if len(frame) == 0:
        raise ValueError("Exp17 source lacks epoch-20 neuron diagnostics")
    mapping = _quartile_groups(frame["train_mean_occupancy"].to_numpy(dtype=np.float64))
    return {
        neuron: f"epoch20_{group}"
        for neuron, group in mapping.items()
    }


def _attach_l2_context(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    fixed_groups: dict[int, str],
) -> pd.DataFrame:
    result = frame.copy()
    result["dynamic_group"] = "not_applicable"
    result["epoch20_fixed_group"] = "not_applicable"
    result["source_head_weight_norm"] = np.nan
    result["source_class_eta2"] = np.nan
    result["source_user_eta2"] = np.nan
    result["source_persistent_quartile"] = "not_applicable"

    for epoch in sorted(result.loc[result["layer"] == "L2", "epoch"].unique()):
        train = result[
            (result["epoch"] == epoch)
            & (result["split"] == "train")
            & (result["layer"] == "L2")
        ].sort_values("neuron")
        groups = _quartile_groups(train["firing_rate_hz"].to_numpy(dtype=np.float64))
        mask = (result["epoch"] == epoch) & (result["layer"] == "L2")
        result.loc[mask, "dynamic_group"] = result.loc[mask, "neuron"].map(groups)
        result.loc[mask, "epoch20_fixed_group"] = result.loc[mask, "neuron"].map(fixed_groups)

    lookup = source.set_index(["epoch", "neuron"])
    for index, row in result[result["layer"] == "L2"].iterrows():
        key = (int(row["epoch"]), int(row["neuron"]))
        src = lookup.loc[key]
        split = str(row["split"])
        result.at[index, "source_head_weight_norm"] = float(src["head_weight_norm"])
        result.at[index, "source_class_eta2"] = float(src[f"{split}_class_eta2"])
        result.at[index, "source_user_eta2"] = float(src[f"{split}_user_eta2"])
        result.at[index, "source_persistent_quartile"] = str(src["persistent_quartile"])
    return result


def _exact_replay_audit(
    model: torch.nn.Module,
    x: torch.Tensor,
    lengths: torch.Tensor,
    replay: dict[str, Any],
) -> bool:
    reference = model(x, lengths)
    for key in ("evidence",):
        if not torch.equal(reference[key], replay[key]):
            return False
    for key in ("spike", "pre_reset", "final_syn", "final_mem"):
        if len(reference[key]) != len(replay[key]):
            return False
        if any(not torch.equal(a, b) for a, b in zip(reference[key], replay[key])):
            return False
    return True


def run_replay_task(config: Config, task_id: int) -> dict[str, Any]:
    if not 0 <= task_id < len(FORMAL_SEEDS):
        raise IndexError(task_id)
    seed = FORMAL_SEEDS[task_id]
    p, _, arrays = _core(config)
    manifest = _load_manifest(config, seed)
    source = _source_neuron_table(config, seed)
    fixed_groups = _fixed_epoch20_groups(source)

    directory = config.results_dir / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    neuron_path = directory / "neuron_dominance.csv"
    summary_path = directory / "snapshot_summary.csv"
    relative_path = directory / "relative_time_summary.csv"
    audit_path = directory / "audit.json"
    if all(path.exists() for path in (neuron_path, summary_path, relative_path, audit_path)):
        return {"status": "exists", "seed": seed}

    neuron_rows: list[dict[str, Any]] = []
    relative_rows: list[dict[str, Any]] = []
    exact_checks: list[dict[str, Any]] = []

    for snapshot in manifest["snapshots"]:
        epoch = int(snapshot["epoch"])
        roles = list(snapshot["roles"])
        model = _load_snapshot_model(config, seed, epoch, p)
        tau_shifts = [_tau_shift_labels(model, layer) for layer in range(len(model.layers))]

        for split in SPLITS:
            accumulators = [_new_layer_accumulator(p.width) for _ in model.layers]
            relative = [_new_relative_accumulator() for _ in model.layers]
            checked = False
            with torch.no_grad():
                for x, _, lengths in exp16.loader(arrays, split, p, seed):
                    replay = instrumented_replay(model, x, lengths)
                    if not checked:
                        exact = _exact_replay_audit(model, x, lengths, replay)
                        exact_checks.append(
                            {
                                "epoch": epoch,
                                "split": split,
                                "exact_factual_replay": exact,
                            }
                        )
                        if not exact:
                            raise AssertionError(
                                f"Instrumented replay diverged at seed{seed} epoch{epoch} {split}"
                            )
                        checked = True
                    _accumulate_batch(replay, lengths, p, accumulators, relative)

            for layer, accumulator in enumerate(accumulators):
                neuron_rows.extend(
                    _accumulator_rows(
                        accumulator,
                        seed=seed,
                        epoch=epoch,
                        roles=roles,
                        split=split,
                        layer=layer,
                        p=p,
                        tau_shifts=tau_shifts[layer],
                    )
                )
            relative_rows.extend(
                _relative_rows(
                    relative,
                    seed=seed,
                    epoch=epoch,
                    roles=roles,
                    split=split,
                )
            )
        print(f"Exp17.2 seed{seed}: replayed epoch {epoch}", flush=True)

    neuron_frame = _attach_l2_context(pd.DataFrame(neuron_rows), source, fixed_groups)
    neuron_frame.sort_values(["epoch", "split", "layer", "neuron"]).to_csv(
        neuron_path, index=False
    )
    _snapshot_summary(neuron_frame).sort_values(
        ["epoch", "split", "layer"]
    ).to_csv(summary_path, index=False)
    pd.DataFrame(relative_rows).sort_values(
        ["epoch", "split", "layer", "relative_bin"]
    ).to_csv(relative_path, index=False)

    epoch20_replay = neuron_frame[
        (neuron_frame["epoch"] == FIXED_GROUP_EPOCH)
        & (neuron_frame["split"] == "train")
        & (neuron_frame["layer"] == "L2")
    ].sort_values("neuron")
    epoch20_source = source[source["epoch"] == FIXED_GROUP_EPOCH].sort_values("neuron")
    max_occ_error = float(
        np.max(
            np.abs(
                epoch20_replay["firing_rate_hz"].to_numpy() / p.fs
                - epoch20_source["train_mean_occupancy"].to_numpy()
            )
        )
    )
    audit = {
        "seed": seed,
        "all_factual_replays_exact": all(row["exact_factual_replay"] for row in exact_checks),
        "exact_checks": exact_checks,
        "epoch20_max_abs_occupancy_error_vs_exp17": max_occ_error,
        "source_exp17_manifest": str(_seed_source_dir(config, seed) / "snapshot_manifest.json"),
        "no_parameter_or_checkpoint_mutation": True,
    }
    save_json(audit_path, audit)
    return {
        "status": "PASS",
        "seed": seed,
        "snapshots": len(manifest["snapshots"]),
        "epoch20_max_abs_occupancy_error": max_occ_error,
    }


def _horizon_epochs(manifest: dict[str, Any]) -> list[int]:
    available = {int(row["epoch"]): set(row["roles"]) for row in manifest["snapshots"]}
    selected = {epoch for epoch in HORIZON_BASE_EPOCHS if epoch in available}
    for epoch, roles in available.items():
        if any(role in roles for role in HORIZON_ROLE_EPOCHS):
            selected.add(epoch)
    return sorted(selected)


def _new_horizon_accumulator() -> dict[int, dict[str, float]]:
    return {
        step: {
            "eligible_samples": 0.0,
            "eligible_neuron_steps": 0.0,
            "spike_flip_count": 0.0,
            "abs_syn_state_delta": 0.0,
            "abs_mem_state_delta": 0.0,
            "current_evidence_delta_norm": 0.0,
            "cumulative_spike_count_delta_abs": 0.0,
            "cumulative_evidence_delta_norm": 0.0,
        }
        for step in HORIZON_STEPS
    }


def _accumulate_horizon_batch(
    model: torch.nn.Module,
    replay: dict[str, Any],
    lengths: torch.Tensor,
    accumulator: dict[int, dict[str, float]],
) -> None:
    if len(model.layers) < 2:
        raise ValueError("Exp17.2 horizon diagnostic requires L2")
    p = model.protocol
    l1_spikes = replay["spike"][0]
    l2_spikes = replay["spike"][1]
    l2 = replay["components"][1]
    steps = l2_spikes.shape[1]
    max_horizon = max(HORIZON_STEPS)
    alpha = getattr(model, "alpha_1")
    beta = model.betas[1]

    for anchor in range(0, steps, HORIZON_ANCHOR_STRIDE):
        anchor_active = (anchor < lengths) & (l1_spikes[:, anchor].sum(dim=1) > 0)
        if not bool(anchor_active.any()):
            continue
        cf_syn = l2["syn_prev"][:, anchor].clone()
        cf_mem = l2["mem_prev"][:, anchor].clone()
        cumulative_spike_delta = torch.zeros_like(cf_syn)
        cumulative_evidence_delta = torch.zeros(
            cf_syn.shape[0],
            model.head.out_features,
            dtype=cf_syn.dtype,
            device=cf_syn.device,
        )

        for offset in range(0, max_horizon + 1):
            t = anchor + offset
            if t >= steps:
                break
            active = anchor_active & (t < lengths)
            if not bool(active.any()):
                continue
            active_col = active[:, None]
            drive = (
                torch.zeros_like(l2["new_drive"][:, t])
                if offset == 0
                else l2["new_drive"][:, t]
            )
            candidate = alpha * cf_syn + drive
            spike_cf, next_mem_raw, _ = lif_step(
                candidate,
                cf_mem,
                beta,
                p.threshold,
                p.surrogate_slope,
            )
            next_syn = torch.where(active_col, candidate, cf_syn)
            next_mem = torch.where(active_col, next_mem_raw, cf_mem)
            spike_cf = spike_cf * active_col
            factual_spike = l2_spikes[:, t] * active_col
            spike_delta = factual_spike - spike_cf
            cumulative_spike_delta = cumulative_spike_delta + spike_delta
            evidence_delta = model.head(spike_delta)
            cumulative_evidence_delta = cumulative_evidence_delta + evidence_delta

            cf_syn, cf_mem = next_syn, next_mem
            if offset not in HORIZON_STEPS:
                continue

            target = accumulator[offset]
            n_samples = float(active.sum().item())
            target["eligible_samples"] += n_samples
            target["eligible_neuron_steps"] += n_samples * p.width
            target["spike_flip_count"] += float(
                ((factual_spike.bool() != spike_cf.bool()) & active_col).sum().item()
            )
            target["abs_syn_state_delta"] += float(
                ((l2["syn_state"][:, t] - cf_syn).abs() * active_col).sum().item()
            )
            target["abs_mem_state_delta"] += float(
                ((l2["mem_state"][:, t] - cf_mem).abs() * active_col).sum().item()
            )
            target["current_evidence_delta_norm"] += float(
                (evidence_delta.norm(dim=1) * active).sum().item()
            )
            target["cumulative_spike_count_delta_abs"] += float(
                (cumulative_spike_delta.abs() * active_col).sum().item()
            )
            target["cumulative_evidence_delta_norm"] += float(
                (cumulative_evidence_delta.norm(dim=1) * active).sum().item()
            )


def _horizon_rows(
    accumulator: dict[int, dict[str, float]],
    *,
    seed: int,
    epoch: int,
    roles: list[str],
    split: str,
    p: Protocol,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in HORIZON_STEPS:
        values = accumulator[step]
        samples = values["eligible_samples"]
        neuron_steps = values["eligible_neuron_steps"]
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "snapshot_roles": ";".join(roles),
                "split": split,
                "horizon_steps": step,
                "horizon_ms": step * 1000.0 / p.fs,
                "eligible_samples": int(samples),
                "spike_flip_fraction": (
                    values["spike_flip_count"] / neuron_steps
                    if neuron_steps > 0
                    else math.nan
                ),
                "mean_abs_syn_state_delta": (
                    values["abs_syn_state_delta"] / neuron_steps
                    if neuron_steps > 0
                    else math.nan
                ),
                "mean_abs_mem_state_delta": (
                    values["abs_mem_state_delta"] / neuron_steps
                    if neuron_steps > 0
                    else math.nan
                ),
                "mean_current_evidence_delta_norm": (
                    values["current_evidence_delta_norm"] / samples
                    if samples > 0
                    else math.nan
                ),
                "mean_abs_cumulative_spike_count_delta": (
                    values["cumulative_spike_count_delta_abs"] / neuron_steps
                    if neuron_steps > 0
                    else math.nan
                ),
                "mean_cumulative_evidence_delta_norm": (
                    values["cumulative_evidence_delta_norm"] / samples
                    if samples > 0
                    else math.nan
                ),
            }
        )
    return rows


def run_horizon_task(config: Config, task_id: int) -> dict[str, Any]:
    if not 0 <= task_id < len(FORMAL_SEEDS):
        raise IndexError(task_id)
    seed = FORMAL_SEEDS[task_id]
    p, _, arrays = _core(config)
    manifest = _load_manifest(config, seed)
    epochs = _horizon_epochs(manifest)
    role_map = {
        int(row["epoch"]): list(row["roles"]) for row in manifest["snapshots"]
    }
    directory = config.results_dir / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "short_horizon.csv"
    if output.exists():
        return {"status": "exists", "seed": seed}

    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        model = _load_snapshot_model(config, seed, epoch, p)
        for split in HORIZON_SPLITS:
            accumulator = _new_horizon_accumulator()
            with torch.no_grad():
                for x, _, lengths in exp16.loader(arrays, split, p, seed):
                    replay = instrumented_replay(model, x, lengths)
                    _accumulate_horizon_batch(model, replay, lengths, accumulator)
            rows.extend(
                _horizon_rows(
                    accumulator,
                    seed=seed,
                    epoch=epoch,
                    roles=role_map[epoch],
                    split=split,
                    p=p,
                )
            )
        print(f"Exp17.2 seed{seed}: horizon epoch {epoch}", flush=True)

    pd.DataFrame(rows).sort_values(
        ["epoch", "split", "horizon_steps"]
    ).to_csv(output, index=False)
    return {"status": "PASS", "seed": seed, "epochs": epochs}


def prepare(config: Config) -> dict[str, Any]:
    p, lock, _ = _core(config)
    if len(exp17._trajectory_spec(FORMAL_SEEDS[0]).case) == 0:
        raise AssertionError("Invalid Exp17 trajectory spec")
    source_protocol = config.exp17_results_dir / "protocol.json"
    if not source_protocol.exists():
        raise FileNotFoundError(source_protocol)

    source_manifests: dict[str, str] = {}
    for seed in FORMAL_SEEDS:
        manifest = _load_manifest(config, seed)
        source = _source_neuron_table(config, seed)
        if FIXED_GROUP_EPOCH not in set(source["epoch"].astype(int)):
            raise ValueError(f"seed{seed} lacks epoch-{FIXED_GROUP_EPOCH} source diagnostics")
        for snapshot in manifest["snapshots"]:
            path = _snapshot_path(config, seed, int(snapshot["epoch"]))
            if not path.exists():
                raise FileNotFoundError(path)
        source_manifests[str(seed)] = str(
            _seed_source_dir(config, seed) / "snapshot_manifest.json"
        )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment": exp17.EXPERIMENT_ID,
        "source_protocol": exp17.PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "core_dataset_hash": lock["dataset_hash"],
        "formal_seeds": list(FORMAL_SEEDS),
        "architecture": {
            "width": p.width,
            "fs": p.fs,
            "tau_mem_ms": p.tau_mem_ms,
            "threshold": p.threshold,
            "shifts": [list(v) for v in exp16.SHIFTS],
        },
        "instantaneous_replay": {
            "all_exp17_snapshots": True,
            "splits": list(SPLITS),
            "decomposition": {
                "membrane_carry": "beta * V[t-1]",
                "synaptic_history": "alpha * I[t-1]",
                "current_incoming_drive": "W * z[t]",
                "synaptic_state": "synaptic_history + current_incoming_drive",
                "pre_reset": "membrane_carry + synaptic_state",
            },
            "relative_bins": RELATIVE_BINS,
            "fixed_group_epoch": FIXED_GROUP_EPOCH,
        },
        "short_horizon": {
            "base_epochs": list(HORIZON_BASE_EPOCHS),
            "role_epochs": list(HORIZON_ROLE_EPOCHS),
            "splits": list(HORIZON_SPLITS),
            "steps": list(HORIZON_STEPS),
            "anchor_stride": HORIZON_ANCHOR_STRIDE,
            "intervention": (
                "zero L2 incoming drive only at anchor; preserve factual pre-anchor "
                "L2 state and restore factual future L1 spike drives"
            ),
        },
        "primary_hypothesis": (
            "WCCE training increasingly makes spike decisions history-sufficient "
            "while reducing current incoming-drive leverage."
        ),
        "interpretation_contract": {
            "history_dominance_support": (
                "firing rises while history-sufficient spike fraction rises and "
                "incoming-drive flip fraction falls consistently across seeds"
            ),
            "state_to_spike_bottleneck": (
                "instantaneous incoming-drive flips fall but short-horizon state/"
                "evidence effects remain substantial"
            ),
            "updateability_bottleneck": (
                "both instantaneous incoming-drive leverage and short-horizon "
                "state/evidence effects collapse"
            ),
            "hypothesis_not_supported": (
                "firing rises without reduced incoming-drive leverage"
            ),
        },
        "hard_contracts": {
            "artifact_only_no_training": True,
            "source_checkpoints_never_modified": True,
            "factual_replay_exactly_matches_BenchmarkNet": True,
            "padding_excluded": True,
            "epoch20_groups_use_train_occupancy_only": True,
            "test_never_selects_checkpoints_groups_or_hyperparameters": True,
            "core_benchmark_contract_unchanged": True,
        },
        "source_manifests": source_manifests,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["identity"] = hashlib.sha256(encoded).hexdigest()
    save_json(config.results_dir / "protocol.json", payload)
    return payload


def _read_seed_csvs(config: Config, filename: str) -> pd.DataFrame:
    paths = [config.results_dir / f"seed{seed}" / filename for seed in FORMAL_SEEDS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Exp17.2 outputs: " + ", ".join(missing))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _group_metric_summary(
    neurons: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    l2 = neurons[neurons["layer"] == "L2"].copy()
    metrics = [
        "firing_rate_hz",
        "mean_syn_history_dominance",
        "mean_state_dominance",
        "incoming_drive_flip_fraction",
        "history_sufficient_spike_fraction",
        "raw_input_flip_given_active",
        "source_class_eta2",
        "source_user_eta2",
    ]
    rows: list[dict[str, Any]] = []
    for keys, frame in l2.groupby(
        ["seed", "epoch", "snapshot_roles", "split", group_column], sort=True
    ):
        row = dict(
            zip(
                ("seed", "epoch", "snapshot_roles", "split", group_column),
                keys,
            )
        )
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else math.nan
            row[f"{metric}_std"] = float(finite.std(ddof=0)) if len(finite) else math.nan
        row["n_neurons"] = len(frame)
        rows.append(row)
    return pd.DataFrame(rows)


def _hypothesis_summary(neurons: pd.DataFrame) -> dict[str, Any]:
    result: list[dict[str, Any]] = []
    train_l2 = neurons[
        (neurons["split"] == "train") & (neurons["layer"] == "L2")
    ].copy()
    metrics = (
        "firing_rate_hz",
        "mean_syn_history_dominance",
        "incoming_drive_flip_fraction",
        "history_sufficient_spike_fraction",
        "raw_input_flip_given_active",
    )
    for seed in FORMAL_SEEDS:
        seed_frame = train_l2[train_l2["seed"] == seed]
        base = seed_frame[seed_frame["epoch"] == FIXED_GROUP_EPOCH].set_index("neuron")
        selected = seed_frame[
            seed_frame["snapshot_roles"].fillna("").str.contains("selected_best")
        ].set_index("neuron")
        if len(base) == 0 or len(selected) == 0:
            raise ValueError(f"seed{seed} missing epoch20 or selected-best diagnostics")
        selected_epoch = int(selected["epoch"].iloc[0])
        common = base.index.intersection(selected.index)
        base = base.loc[common]
        selected = selected.loc[common]
        row: dict[str, Any] = {
            "seed": seed,
            "from_epoch": FIXED_GROUP_EPOCH,
            "to_selected_best_epoch": selected_epoch,
        }
        for metric in metrics:
            row[f"delta_{metric}"] = float(selected[metric].mean() - base[metric].mean())

        delta_firing = selected["firing_rate_hz"] - base["firing_rate_hz"]
        delta_history_sufficient = (
            selected["history_sufficient_spike_fraction"]
            - base["history_sufficient_spike_fraction"]
        )
        delta_incoming_flip = (
            selected["incoming_drive_flip_fraction"]
            - base["incoming_drive_flip_fraction"]
        )
        row["spearman_delta_firing_vs_delta_history_sufficient"] = float(
            delta_firing.corr(delta_history_sufficient, method="spearman")
        )
        row["spearman_delta_firing_vs_negative_delta_incoming_flip"] = float(
            delta_firing.corr(-delta_incoming_flip, method="spearman")
        )

        low_mask = base["epoch20_fixed_group"] == "epoch20_low25"
        if bool(low_mask.any()):
            row["epoch20_low25_delta_firing_rate_hz"] = float(
                delta_firing[low_mask].mean()
            )
            row["epoch20_low25_delta_history_sufficient"] = float(
                delta_history_sufficient[low_mask].mean()
            )
            row["epoch20_low25_delta_incoming_flip"] = float(
                delta_incoming_flip[low_mask].mean()
            )
        result.append(row)

    return {
        "paired_epoch20_to_selected_best": result,
        "interpretation_rule": (
            "Treat these as mechanism diagnostics, not automatic causal proof. "
            "History-dominance support requires seed-consistent firing increase "
            "together with increased history-sufficient spikes and reduced "
            "incoming-drive flip fraction."
        ),
    }


def finalize(config: Config) -> dict[str, Any]:
    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)

    neurons = _read_seed_csvs(config, "neuron_dominance.csv")
    snapshots = _read_seed_csvs(config, "snapshot_summary.csv")
    relative = _read_seed_csvs(config, "relative_time_summary.csv")
    horizon = _read_seed_csvs(config, "short_horizon.csv")

    neurons.sort_values(["seed", "epoch", "split", "layer", "neuron"]).to_csv(
        aggregate / "neuron_dominance.csv", index=False
    )
    snapshots.sort_values(["seed", "epoch", "split", "layer"]).to_csv(
        aggregate / "trajectory_summary.csv", index=False
    )
    relative.sort_values(
        ["seed", "epoch", "split", "layer", "relative_bin"]
    ).to_csv(aggregate / "relative_time_summary.csv", index=False)
    horizon.sort_values(
        ["seed", "epoch", "split", "horizon_steps"]
    ).to_csv(aggregate / "short_horizon_summary.csv", index=False)

    _group_metric_summary(neurons, "epoch20_fixed_group").to_csv(
        aggregate / "epoch20_fixed_group.csv", index=False
    )
    _group_metric_summary(neurons, "dynamic_group").to_csv(
        aggregate / "dynamic_group_summary.csv", index=False
    )

    tau_rows: list[dict[str, Any]] = []
    l2 = neurons[neurons["layer"] == "L2"]
    for keys, frame in l2.groupby(
        ["seed", "epoch", "snapshot_roles", "split", "tau_shift"], sort=True
    ):
        row = dict(
            zip(
                ("seed", "epoch", "snapshot_roles", "split", "tau_shift"),
                keys,
            )
        )
        for metric in (
            "firing_rate_hz",
            "mean_syn_history_dominance",
            "incoming_drive_flip_fraction",
            "history_sufficient_spike_fraction",
            "raw_input_flip_given_active",
        ):
            row[f"{metric}_mean"] = float(frame[metric].mean())
        row["n_neurons"] = len(frame)
        tau_rows.append(row)
    pd.DataFrame(tau_rows).to_csv(aggregate / "tau_group_summary.csv", index=False)

    hypothesis = _hypothesis_summary(neurons)
    save_json(aggregate / "hypothesis_summary.json", hypothesis)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "neuron_rows": len(neurons),
        "trajectory_rows": len(snapshots),
        "relative_time_rows": len(relative),
        "short_horizon_rows": len(horizon),
        "outputs": [
            "neuron_dominance.csv",
            "trajectory_summary.csv",
            "relative_time_summary.csv",
            "short_horizon_summary.csv",
            "epoch20_fixed_group.csv",
            "dynamic_group_summary.csv",
            "tau_group_summary.csv",
            "hypothesis_summary.json",
        ],
    }
    save_json(aggregate / "manifest.json", manifest)
    return {"status": "PASS", **manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    parser.add_argument("--exp17-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    replay = sub.add_parser("replay")
    replay.add_argument("--task-id", type=int, required=True)
    horizon = sub.add_parser("horizon")
    horizon.add_argument("--task-id", type=int, required=True)
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    if args.command == "prepare":
        result = prepare(config)
    elif args.command == "replay":
        result = run_replay_task(config, args.task_id)
    elif args.command == "horizon":
        result = run_horizon_task(config, args.task_id)
    elif args.command == "finalize":
        result = finalize(config)
    else:
        raise ValueError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
