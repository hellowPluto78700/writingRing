from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import pandas as pd
import torch

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_5_2_2_frozen_local_multitau_syn as exp522


DIAGNOSTIC_ID = "experiment_5_2_2_single_segment_dynamics"
DIAGNOSTIC_VERSION = "checkpoint_single_segment_normal_vs_reset_v1"
DEFAULT_INTERVENTION = "reset567"
SELECTIONS = (
    "auto",
    "first",
    "normal_wrong_reset_correct",
    "normal_correct_reset_wrong",
    "both_correct",
    "both_wrong",
    "prediction_changed",
)


@dataclass(frozen=True)
class SelectedSegment:
    split: str
    index: int
    selection_reason: str
    true_class: int
    normal_prediction: int
    intervention_prediction: int


def diagnostic_root(repo_root: Path) -> Path:
    return exp522.results_dir(repo_root) / "single_segment_dynamics"


def _partition(
    data: base.Data,
    cache: dict[str, np.ndarray],
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    partitions = exp522._partitions(data, cache)
    if split not in partitions:
        raise ValueError(f"Unknown split: {split}")
    return partitions[split]


def _as_model_input(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def trace_segment(
    model: exp522.SynapticPhaseDecoder,
    l2_segment: np.ndarray | torch.Tensor,
    valid_length: int,
    reset_shifts: tuple[int, ...] = (),
) -> dict[str, np.ndarray]:
    """Run one valid segment and record the full internal trajectory.

    This reproduces Exp5.2.2 forward semantics exactly, while retaining
    quantities that the training/evaluation trajectory does not persist:
    input drive, pre-reset membrane, post-reset membrane, and output state.
    """
    device = model.alpha.device
    x = torch.as_tensor(l2_segment, dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[2] != exp522.LOCAL_WIDTH:
        raise ValueError(
            f"Expected one [T,{exp522.LOCAL_WIDTH}] L2 segment, got {tuple(x.shape)}"
        )
    length = int(valid_length)
    if length <= 0 or length > x.shape[1]:
        raise ValueError(f"Invalid valid_length={length} for padded T={x.shape[1]}")
    x = x[:, :length]

    syn = torch.zeros(1, exp522.TEMPORAL_WIDTH, device=device, dtype=x.dtype)
    mem = torch.zeros_like(syn)
    output_mem = torch.zeros(1, model.n_classes, device=device, dtype=x.dtype)
    reset_mask = model._reset_mask(reset_shifts, device)

    input_drive_parts: list[torch.Tensor] = []
    syn_parts: list[torch.Tensor] = []
    pre_mem_parts: list[torch.Tensor] = []
    spike_parts: list[torch.Tensor] = []
    post_mem_parts: list[torch.Tensor] = []
    pre_lif_parts: list[torch.Tensor] = []
    output_spike_parts: list[torch.Tensor] = []
    output_pre_mem_parts: list[torch.Tensor] = []
    output_post_mem_parts: list[torch.Tensor] = []

    with torch.no_grad():
        for timestep in range(length):
            if (
                timestep > 0
                and timestep % model.bin_steps == 0
                and bool(reset_mask.any())
            ):
                mask = reset_mask.unsqueeze(0)
                syn = syn.masked_fill(mask, 0.0)
                mem = mem.masked_fill(mask, 0.0)

            input_drive = model.input_hidden(x[:, timestep])
            syn = model.alpha * syn + input_drive
            hidden_spike, mem, hidden_pre_mem = model.hidden_lif(syn, mem)
            pre_lif = model.output_linear(hidden_spike)

            input_drive_parts.append(input_drive)
            syn_parts.append(syn)
            pre_mem_parts.append(hidden_pre_mem)
            spike_parts.append(hidden_spike)
            post_mem_parts.append(mem)
            pre_lif_parts.append(pre_lif)

            if model.output_lif is not None:
                output_spike, output_mem, output_pre_mem = model.output_lif(
                    pre_lif, output_mem
                )
                output_spike_parts.append(output_spike)
                output_pre_mem_parts.append(output_pre_mem)
                output_post_mem_parts.append(output_mem)

    def _stack(parts: list[torch.Tensor]) -> np.ndarray:
        return torch.stack(parts, dim=1).squeeze(0).cpu().numpy()

    trace = {
        "l2_spikes": x.squeeze(0).cpu().numpy(),
        "hidden_input_drive": _stack(input_drive_parts),
        "hidden_synaptic": _stack(syn_parts),
        "hidden_pre_reset_membrane": _stack(pre_mem_parts),
        "hidden_spikes": _stack(spike_parts),
        "hidden_post_reset_membrane": _stack(post_mem_parts),
        "pre_lif_logits": _stack(pre_lif_parts),
    }
    if output_spike_parts:
        trace["output_spikes"] = _stack(output_spike_parts)
        trace["output_pre_reset_membrane"] = _stack(output_pre_mem_parts)
        trace["output_post_reset_membrane"] = _stack(output_post_mem_parts)
    return trace


def _native_logits_from_trace(
    model: exp522.SynapticPhaseDecoder,
    trace: dict[str, np.ndarray],
) -> np.ndarray:
    if model.readout == "hidden_count_linear":
        return np.asarray(trace["pre_lif_logits"], dtype=np.float64).sum(axis=0)
    output = trace.get("output_spikes")
    if output is None:
        raise RuntimeError("output_lif trace is missing output_spikes")
    return np.asarray(output, dtype=np.float64).sum(axis=0)


def _predict_partition(
    model: exp522.SynapticPhaseDecoder,
    X: np.ndarray,
    lengths: np.ndarray,
    reset_shifts: tuple[int, ...],
    batch_size: int,
) -> np.ndarray:
    device = model.alpha.device
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            xb = _as_model_input(X[start:stop], device)
            lb = torch.as_tensor(
                lengths[start:stop], dtype=torch.long, device=device
            )
            trajectory = model.forward_trajectory(xb, reset_shifts=reset_shifts)
            logits = model.native_logits(trajectory, lb)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def select_segment(
    y: np.ndarray,
    normal_prediction: np.ndarray,
    intervention_prediction: np.ndarray,
    split: str,
    selection: str,
    sample_index: int | None,
) -> SelectedSegment:
    if len(y) != len(normal_prediction) or len(y) != len(intervention_prediction):
        raise ValueError("Prediction and label lengths do not match")
    if sample_index is not None:
        index = int(sample_index)
        if not 0 <= index < len(y):
            raise IndexError(f"sample_index={index} outside 0..{len(y)-1}")
        reason = "explicit_sample_index"
    else:
        correct_normal = normal_prediction == y
        correct_intervention = intervention_prediction == y
        changed = normal_prediction != intervention_prediction
        masks = {
            "first": np.ones(len(y), dtype=bool),
            "normal_wrong_reset_correct": (~correct_normal) & correct_intervention,
            "normal_correct_reset_wrong": correct_normal & (~correct_intervention),
            "both_correct": correct_normal & correct_intervention,
            "both_wrong": (~correct_normal) & (~correct_intervention),
            "prediction_changed": changed,
        }
        if selection == "auto":
            priorities = (
                "normal_wrong_reset_correct",
                "prediction_changed",
                "both_wrong",
                "both_correct",
                "first",
            )
            chosen_name = next(
                name for name in priorities if np.flatnonzero(masks[name]).size
            )
            index = int(np.flatnonzero(masks[chosen_name])[0])
            reason = chosen_name
        else:
            if selection not in masks:
                raise ValueError(f"Unknown selection: {selection}")
            candidates = np.flatnonzero(masks[selection])
            if candidates.size == 0:
                raise RuntimeError(
                    f"No {split} sample satisfies selection={selection}; "
                    "use --selection auto/first or --sample-index."
                )
            index = int(candidates[0])
            reason = selection

    return SelectedSegment(
        split=split,
        index=index,
        selection_reason=reason,
        true_class=int(y[index]),
        normal_prediction=int(normal_prediction[index]),
        intervention_prediction=int(intervention_prediction[index]),
    )


def _run_lengths(spikes: np.ndarray) -> list[int]:
    values = np.asarray(spikes > 0.5, dtype=np.int8)
    runs: list[int] = []
    current = 0
    for value in values:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _fraction_spikes_in_long_runs(runs: list[int], minimum: int) -> float:
    total = sum(runs)
    if total == 0:
        return 0.0
    return float(sum(run for run in runs if run >= minimum) / total)


def neuron_metrics(
    trace: dict[str, np.ndarray],
    profile: str,
    condition: str,
    fs: float,
) -> pd.DataFrame:
    spikes = np.asarray(trace["hidden_spikes"])
    syn = np.asarray(trace["hidden_synaptic"])
    pre_mem = np.asarray(trace["hidden_pre_reset_membrane"])
    post_mem = np.asarray(trace["hidden_post_reset_membrane"])
    rows: list[dict[str, object]] = []
    slices = exp522.profile_group_slices(profile)

    shift_by_neuron = np.empty(exp522.TEMPORAL_WIDTH, dtype=np.int64)
    for shift, group in slices.items():
        shift_by_neuron[group] = shift

    for neuron in range(exp522.TEMPORAL_WIDTH):
        neuron_spikes = spikes[:, neuron] > 0.5
        spike_indices = np.flatnonzero(neuron_spikes)
        runs = _run_lengths(neuron_spikes)
        if spike_indices.size:
            backlog_fraction = float(
                np.mean(post_mem[spike_indices, neuron] > exp522.THRESHOLD)
            )
            first_spike = int(spike_indices[0])
        else:
            backlog_fraction = float("nan")
            first_spike = -1
        rows.append(
            {
                "condition": condition,
                "neuron": neuron,
                "shift": int(shift_by_neuron[neuron]),
                "firing_rate_hz": float(neuron_spikes.mean() * fs),
                "spike_count": int(neuron_spikes.sum()),
                "first_spike_timestep": first_spike,
                "n_firing_runs": len(runs),
                "mean_run_length": float(np.mean(runs)) if runs else 0.0,
                "max_run_length": max(runs, default=0),
                "fraction_spikes_in_runs_ge4": _fraction_spikes_in_long_runs(runs, 4),
                "fraction_spikes_in_runs_ge8": _fraction_spikes_in_long_runs(runs, 8),
                "post_reset_above_threshold_given_spike": backlog_fraction,
                "endpoint_synaptic": float(syn[-1, neuron]),
                "endpoint_pre_reset_membrane": float(pre_mem[-1, neuron]),
                "endpoint_post_reset_membrane": float(post_mem[-1, neuron]),
            }
        )
    return pd.DataFrame(rows)


def group_metrics(neurons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (condition, shift), group in neurons.groupby(["condition", "shift"], sort=True):
        rows.append(
            {
                "condition": condition,
                "shift": int(shift),
                "n_neurons": int(len(group)),
                "mean_firing_rate_hz": float(group["firing_rate_hz"].mean()),
                "mean_max_run_length": float(group["max_run_length"].mean()),
                "max_run_length": int(group["max_run_length"].max()),
                "fraction_neurons_run_ge4": float(np.mean(group["max_run_length"] >= 4)),
                "fraction_neurons_run_ge8": float(np.mean(group["max_run_length"] >= 8)),
                "mean_fraction_spikes_in_runs_ge4": float(
                    group["fraction_spikes_in_runs_ge4"].mean()
                ),
                "mean_fraction_spikes_in_runs_ge8": float(
                    group["fraction_spikes_in_runs_ge8"].mean()
                ),
                "mean_backlog_fraction_when_spiking": float(
                    group["post_reset_above_threshold_given_spike"].mean(skipna=True)
                ),
                "mean_endpoint_abs_synaptic": float(
                    group["endpoint_synaptic"].abs().mean()
                ),
                "mean_endpoint_abs_post_membrane": float(
                    group["endpoint_post_reset_membrane"].abs().mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _group_boundaries(profile: str) -> tuple[list[int], list[float], list[str]]:
    boundaries: list[int] = []
    centers: list[float] = []
    labels: list[str] = []
    for shift, group in exp522.profile_group_slices(profile).items():
        if group.start:
            boundaries.append(group.start)
        centers.append((group.start + group.stop - 1) / 2.0)
        labels.append(f"s{shift} ({group.stop - group.start})")
    return boundaries, centers, labels


def _decorate_neuron_axis(ax: plt.Axes, profile: str) -> None:
    boundaries, centers, labels = _group_boundaries(profile)
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="0.6", linewidth=0.8)
    ax.set_yticks(centers)
    ax.set_yticklabels(labels)
    ax.set_ylabel("L3 neuron group")


def _draw_reset_boundaries(ax: plt.Axes, length: int, bin_steps: int) -> None:
    for timestep in range(bin_steps, length, bin_steps):
        ax.axvline(timestep - 0.5, color="0.75", linestyle=":", linewidth=0.7)


def plot_l3_raster(
    traces: dict[str, dict[str, np.ndarray]],
    profile: str,
    bin_steps: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(len(traces), 1, figsize=(12, 7), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)
    for ax, (condition, trace) in zip(axes_array, traces.items(), strict=True):
        spikes = np.asarray(trace["hidden_spikes"]) > 0.5
        timesteps, neurons = np.nonzero(spikes)
        ax.scatter(timesteps, neurons, s=5, c="black", marker=".")
        _decorate_neuron_axis(ax, profile)
        _draw_reset_boundaries(ax, len(spikes), bin_steps)
        ax.set_ylim(exp522.TEMPORAL_WIDTH - 0.5, -0.5)
        ax.set_title(condition)
    axes_array[-1].set_xlabel("Timestep")
    fig.suptitle("L3 spike raster")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _symlog_norm(arrays: list[np.ndarray]) -> SymLogNorm:
    maximum = max(float(np.max(np.abs(array))) for array in arrays)
    maximum = max(maximum, exp522.THRESHOLD * 2.0)
    return SymLogNorm(
        linthresh=exp522.THRESHOLD,
        linscale=1.0,
        vmin=-maximum,
        vmax=maximum,
        base=10.0,
    )


def plot_l3_state_heatmap(
    traces: dict[str, dict[str, np.ndarray]],
    profile: str,
    key: str,
    title: str,
    colorbar_label: str,
    bin_steps: int,
    path: Path,
) -> None:
    arrays = [np.asarray(trace[key]).T for trace in traces.values()]
    norm = _symlog_norm(arrays)
    fig, axes = plt.subplots(len(traces), 1, figsize=(12, 7), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)
    image = None
    for ax, (condition, trace) in zip(axes_array, traces.items(), strict=True):
        values = np.asarray(trace[key]).T
        image = ax.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap="coolwarm",
            norm=norm,
        )
        _decorate_neuron_axis(ax, profile)
        _draw_reset_boundaries(ax, values.shape[1], bin_steps)
        ax.set_title(condition)
    axes_array[-1].set_xlabel("Timestep")
    fig.suptitle(title)
    if image is not None:
        fig.colorbar(image, ax=list(axes_array), shrink=0.85, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_group_firing(
    traces: dict[str, dict[str, np.ndarray]],
    profile: str,
    bin_steps: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(len(traces), 1, figsize=(12, 7), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)
    slices = exp522.profile_group_slices(profile)
    for ax, (condition, trace) in zip(axes_array, traces.items(), strict=True):
        spikes = np.asarray(trace["hidden_spikes"])
        for shift, group in slices.items():
            ax.plot(spikes[:, group].mean(axis=1), label=f"s{shift}")
        _draw_reset_boundaries(ax, len(spikes), bin_steps)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Firing fraction")
        ax.set_title(condition)
        ax.legend(ncol=min(6, len(slices)), loc="upper right")
    axes_array[-1].set_xlabel("Timestep")
    fig.suptitle("L3 group firing fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_readout_activity(
    traces: dict[str, dict[str, np.ndarray]],
    model: exp522.SynapticPhaseDecoder,
    labels: list[object],
    true_class: int,
    bin_steps: int,
    path: Path,
) -> None:
    if model.readout == "hidden_count_linear":
        fig, axes = plt.subplots(
            len(traces), 1, figsize=(12, 7), sharex=True, sharey=True
        )
        axes_array = np.atleast_1d(axes)
        for ax, (condition, trace) in zip(axes_array, traces.items(), strict=True):
            cumulative_logits = np.cumsum(
                np.asarray(trace["pre_lif_logits"]), axis=0
            )
            for class_index in range(cumulative_logits.shape[1]):
                linewidth = 2.5 if class_index == true_class else 1.0
                ax.plot(
                    cumulative_logits[:, class_index],
                    linewidth=linewidth,
                    label=str(labels[class_index]),
                )
            _draw_reset_boundaries(ax, len(cumulative_logits), bin_steps)
            ax.set_ylabel("Cumulative logit")
            ax.set_title(condition)
        axes_array[-1].set_xlabel("Timestep")
        axes_array[0].legend(
            ncol=6, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0)
        )
        fig.suptitle("Shared Linear readout: cumulative class evidence")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(
        3, len(traces), figsize=(6 * len(traces), 10), sharex="col"
    )
    axes_array = np.asarray(axes)
    if axes_array.ndim == 1:
        axes_array = axes_array[:, None]
    for column, (condition, trace) in enumerate(traces.items()):
        output_spikes = np.asarray(trace["output_spikes"]) > 0.5
        timesteps, neurons = np.nonzero(output_spikes)
        axes_array[0, column].scatter(timesteps, neurons, s=10, c="black", marker=".")
        axes_array[0, column].set_yticks(range(len(labels)))
        axes_array[0, column].set_yticklabels([str(label) for label in labels])
        axes_array[0, column].set_ylabel("Output neuron")
        axes_array[0, column].set_title(condition)

        output_membrane = np.asarray(trace["output_post_reset_membrane"])
        for class_index in range(output_membrane.shape[1]):
            linewidth = 2.5 if class_index == true_class else 1.0
            axes_array[1, column].plot(
                output_membrane[:, class_index],
                linewidth=linewidth,
                label=str(labels[class_index]),
            )
        axes_array[1, column].axhline(
            exp522.THRESHOLD, color="0.5", linestyle="--", linewidth=0.8
        )
        axes_array[1, column].set_ylabel("Output post-reset U")

        cumulative = np.cumsum(output_spikes.astype(np.float32), axis=0)
        for class_index in range(cumulative.shape[1]):
            linewidth = 2.5 if class_index == true_class else 1.0
            axes_array[2, column].plot(
                cumulative[:, class_index],
                linewidth=linewidth,
                label=str(labels[class_index]),
            )
        axes_array[2, column].set_ylabel("Cumulative output spikes")
        axes_array[2, column].set_xlabel("Timestep")
        for row in range(3):
            _draw_reset_boundaries(axes_array[row, column], len(output_spikes), bin_steps)
    axes_array[1, 0].legend(
        ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0)
    )
    fig.suptitle("Output-LIF activity")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _label(labels: list[object], index: int) -> str:
    return str(labels[int(index)])


def run_diagnostic(
    *,
    repo_root: Path,
    profile: str,
    readout: str,
    seed: int,
    split: str,
    intervention: str,
    selection: str,
    sample_index: int | None,
    device_name: str,
    batch_size: int,
    threads: int,
) -> Path:
    if intervention == "normal":
        raise ValueError("Choose a non-normal intervention for paired visualization")
    if selection not in SELECTIONS:
        raise ValueError(f"Unknown selection={selection}; choices={SELECTIONS}")

    repo_root = repo_root.resolve()
    config = exp522.Config(
        repo_root=repo_root,
        results_dir=exp522.results_dir(repo_root),
        device=device_name,
        batch_size=batch_size,
        threads=threads,
    )
    torch.set_num_threads(threads)
    data = base.prepare_data(repo_root)
    spec = exp522.RunSpec(profile=profile, readout=readout, seed=seed)
    cache = exp522.load_source_cache(seed, data, config)
    model, checkpoint_payload = exp522.load_model(spec, data, config)
    X, y, lengths = _partition(data, cache, split)

    reset_shifts = exp522.effective_reset_shifts(profile, intervention)
    normal_prediction = _predict_partition(
        model, X, lengths, (), batch_size=batch_size
    )
    intervention_prediction = _predict_partition(
        model, X, lengths, reset_shifts, batch_size=batch_size
    )
    selected = select_segment(
        y,
        normal_prediction,
        intervention_prediction,
        split,
        selection,
        sample_index,
    )

    index = selected.index
    normal_trace = trace_segment(model, X[index], int(lengths[index]), ())
    intervention_trace = trace_segment(
        model, X[index], int(lengths[index]), reset_shifts
    )
    traces = {
        "normal": normal_trace,
        intervention: intervention_trace,
    }

    output_dir = (
        diagnostic_root(repo_root)
        / spec.key
        / f"{split}_sample{index:04d}__normal_vs_{intervention}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_dir / "trace_normal.npz", **normal_trace)
    np.savez_compressed(
        output_dir / f"trace_{intervention}.npz", **intervention_trace
    )

    neuron_table = pd.concat(
        [
            neuron_metrics(normal_trace, profile, "normal", data.fs),
            neuron_metrics(
                intervention_trace, profile, intervention, data.fs
            ),
        ],
        ignore_index=True,
    )
    neuron_table.to_csv(output_dir / "neuron_metrics.csv", index=False)
    group_table = group_metrics(neuron_table)
    group_table.to_csv(output_dir / "group_metrics.csv", index=False)

    figure_paths = {
        "l3_raster": output_dir / "01_l3_spike_raster.png",
        "l3_synaptic": output_dir / "02_l3_synaptic_state_heatmap.png",
        "l3_pre_membrane": output_dir / "03_l3_pre_reset_membrane_heatmap.png",
        "l3_post_membrane": output_dir / "04_l3_post_reset_membrane_heatmap.png",
        "l3_group_firing": output_dir / "05_l3_group_firing.png",
        "readout_activity": output_dir / "06_readout_activity.png",
    }
    plot_l3_raster(traces, profile, data.bin_steps, figure_paths["l3_raster"])
    plot_l3_state_heatmap(
        traces,
        profile,
        "hidden_synaptic",
        "L3 signed synaptic state I",
        "I",
        data.bin_steps,
        figure_paths["l3_synaptic"],
    )
    plot_l3_state_heatmap(
        traces,
        profile,
        "hidden_pre_reset_membrane",
        "L3 pre-reset membrane U^-",
        "U^-",
        data.bin_steps,
        figure_paths["l3_pre_membrane"],
    )
    plot_l3_state_heatmap(
        traces,
        profile,
        "hidden_post_reset_membrane",
        "L3 post-reset membrane U",
        "U",
        data.bin_steps,
        figure_paths["l3_post_membrane"],
    )
    plot_group_firing(
        traces, profile, data.bin_steps, figure_paths["l3_group_firing"]
    )
    plot_readout_activity(
        traces,
        model,
        list(data.labels),
        selected.true_class,
        data.bin_steps,
        figure_paths["readout_activity"],
    )

    normal_logits = _native_logits_from_trace(model, normal_trace)
    intervention_logits = _native_logits_from_trace(model, intervention_trace)
    summary = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "experiment_id": exp522.EXPERIMENT_ID,
        "protocol_version": exp522.PROTOCOL_VERSION,
        "checkpoint": str(
            exp522.checkpoint_path(config.results_dir, spec).relative_to(repo_root)
        ),
        "best_epoch": int(checkpoint_payload["best_epoch"]),
        "profile": profile,
        "readout": readout,
        "seed": seed,
        "split": split,
        "sample_index": index,
        "selection_reason": selected.selection_reason,
        "valid_length_timesteps": int(lengths[index]),
        "sampling_rate_hz": float(data.fs),
        "intervention": intervention,
        "effective_reset_shifts": list(reset_shifts),
        "reset_period_timesteps": int(data.bin_steps),
        "reset_period_ms": float(exp522.FIXED250_MS),
        "true_class_index": selected.true_class,
        "true_label": _label(list(data.labels), selected.true_class),
        "normal_prediction_index": int(np.argmax(normal_logits)),
        "normal_prediction_label": _label(
            list(data.labels), int(np.argmax(normal_logits))
        ),
        "intervention_prediction_index": int(np.argmax(intervention_logits)),
        "intervention_prediction_label": _label(
            list(data.labels), int(np.argmax(intervention_logits))
        ),
        "normal_logits": normal_logits.tolist(),
        "intervention_logits": intervention_logits.tolist(),
        "figures": {
            name: path.name for name, path in figure_paths.items()
        },
        "tables": {
            "neuron_metrics": "neuron_metrics.csv",
            "group_metrics": "group_metrics.csv",
        },
        "traces": {
            "normal": "trace_normal.npz",
            intervention: f"trace_{intervention}.npz",
        },
        "interpretation_contract": (
            "checkpoint-only visualization; no training, no weight update, "
            "no tail extension; all traces stop at the original valid endpoint"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one existing Exp5.2.2 checkpoint on one valid segment "
            "under normal state and one inference-time reset intervention."
        )
    )
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--profile", choices=tuple(exp522.PROFILE_SHIFTS), default="s234567")
    parser.add_argument("--readout", choices=exp522.READOUTS, default="hidden_count_linear")
    parser.add_argument("--seed", type=int, choices=exp522.SEEDS, default=11)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--intervention",
        choices=tuple(name for name in exp522.INTERVENTIONS if name != "normal"),
        default=DEFAULT_INTERVENTION,
    )
    parser.add_argument("--selection", choices=SELECTIONS, default="auto")
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=exp522.BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=1)
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else exp522.find_repo_root()
    )
    output_dir = run_diagnostic(
        repo_root=repo_root,
        profile=args.profile,
        readout=args.readout,
        seed=args.seed,
        split=args.split,
        intervention=args.intervention,
        selection=args.selection,
        sample_index=args.sample_index,
        device_name=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
