"""Full-recording Matplotlib verification for aligned Board-event segments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from writingring.board_event_segmentation import BoardEventSegmentedSample
from writingring.segmentation import SegmentedSample, SegmentLabel


class SegmentationVerificationError(ValueError):
    """Raised when final segmentation cannot be verified safely."""


@dataclass(frozen=True, slots=True)
class SegmentationVerificationConfig:
    """Full-recording panel and style configuration for either boundary mode."""

    panel_duration_s: float = 10.0
    output_dpi: int = 200
    valid_event_linewidth: float = 1.5
    transient_event_linewidth: float = 0.6
    valid_event_alpha: float = 0.80
    transient_event_alpha: float = 0.35
    segment_alpha: float = 0.10
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class SegmentationVerificationResult:
    """Written PNG facts for one complete Ring recording."""

    output_path: Path
    panel_count: int
    recording_duration_s: float
    displayed_event_count: int
    displayed_label_count: int
    displayed_segment_count: int


def build_segmentation_verification_path(
    output_root: Path,
    *,
    dataset_id: int,
) -> Path:
    """Return the aligned-mode PNG path within an already scoped output root."""

    if isinstance(dataset_id, bool) or not isinstance(dataset_id, int) or dataset_id < 0:
        raise SegmentationVerificationError("dataset_id must be a nonnegative integer")
    return Path(output_root) / f"{dataset_id}_ring_0_segmentation_verification.png"


def create_segmentation_verification_figure(
    *,
    ring_dataframe: pd.DataFrame,
    ring_timestamps_us: np.ndarray,
    transient_score: np.ndarray,
    aligned_board_events: pd.DataFrame | None,
    labels: Sequence[SegmentLabel],
    label_skip_reasons: Sequence[str | None],
    segmented_samples: Sequence[SegmentedSample | BoardEventSegmentedSample],
    output_path: Path,
    config: SegmentationVerificationConfig = SegmentationVerificationConfig(),
    user: str,
    action: str,
    dataset_id: int,
    boundary_mode: str,
    alignment_offset_us: float | None,
) -> SegmentationVerificationResult:
    """Render all recording panels from caller-supplied final boundaries only."""

    _validate_config(config)
    timestamps = _timestamps(ring_timestamps_us)
    score = _score(transient_score, sample_count=len(timestamps))
    if not isinstance(ring_dataframe, pd.DataFrame) or len(ring_dataframe) != len(timestamps):
        raise SegmentationVerificationError(
            "ring_dataframe must have one row per Ring timestamp"
        )
    if boundary_mode not in {"label", "aligned_board_events"}:
        raise SegmentationVerificationError(
            "boundary_mode must be label or aligned_board_events"
        )
    aligned_mode = boundary_mode == "aligned_board_events"
    overlay_events = aligned_board_events is not None
    if (aligned_mode or overlay_events) and (
        alignment_offset_us is None or not math.isfinite(float(alignment_offset_us))
    ):
        raise SegmentationVerificationError("aligned mode requires a finite alignment offset")
    events = _events(aligned_board_events, required=aligned_mode)
    markers = _labels(labels)
    if len(label_skip_reasons) != len(markers):
        raise SegmentationVerificationError(
            "label_skip_reasons must match the timestamp label count"
        )
    samples = tuple(segmented_samples)
    _validate_samples(samples, timestamps, boundary_mode=boundary_mode)
    output = Path(output_path)
    _validate_output_path(output, overwrite=config.overwrite)

    ring_start = float(timestamps[0])
    elapsed = (timestamps - ring_start) / 1_000_000.0
    duration = float(elapsed[-1])
    panel_count = max(1, math.ceil(duration / config.panel_duration_s))
    figure, axes = plt.subplots(
        panel_count,
        1,
        figsize=(16.0, 3.0 * panel_count),
        sharey=True,
        layout="constrained",
    )
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    y_max = float(np.quantile(score, 0.995) * 1.1)
    if not math.isfinite(y_max) or y_max <= 0.0:
        y_max = max(float(np.max(score)), 1.0)
    label_elapsed = np.asarray(
        [(marker.timestamp_us - ring_start) / 1_000_000.0 for marker in markers],
        dtype=np.float64,
    )
    event_elapsed = (
        events["aligned_event_timestamp_us"].to_numpy(dtype=np.float64) - ring_start
    ) / 1_000_000.0
    for panel_index, axis in enumerate(axes_array):
        panel_start = panel_index * config.panel_duration_s
        panel_stop = min((panel_index + 1) * config.panel_duration_s, duration)
        final_panel = panel_index == panel_count - 1
        mask = _panel_mask(elapsed, panel_start, panel_stop, final_panel)
        axis.plot(
            elapsed[mask],
            score[mask],
            color="tab:blue",
            linewidth=0.8,
            label="Ring transient score" if panel_index == 0 else None,
        )
        _draw_segments(
            axis,
            samples,
            timestamps_us=timestamps,
            ring_start_us=ring_start,
            panel_start_s=panel_start,
            panel_stop_s=panel_stop,
            alpha=config.segment_alpha,
            boundary_mode=boundary_mode,
        )
        _draw_events(
            axis,
            events,
            event_elapsed,
            panel_start_s=panel_start,
            panel_stop_s=panel_stop,
            final_panel=final_panel,
            first_panel=panel_index == 0,
            config=config,
        )
        _draw_labels(
            axis,
            markers,
            label_elapsed,
            label_skip_reasons,
            panel_start_s=panel_start,
            panel_stop_s=panel_stop,
            final_panel=final_panel,
            first_panel=panel_index == 0,
        )
        axis.set_xlim(panel_start, panel_stop if panel_stop > panel_start else panel_start + 1.0)
        axis.set_ylim(0.0, y_max)
        axis.set_title(f"{panel_start:g}\N{EN DASH}{panel_stop:g} s")
        axis.grid(True, alpha=0.25)
        if panel_index == len(axes_array) - 1:
            axis.set_xlabel("Elapsed Ring time from recording start (s)")
    axes_array[0].set_ylabel("Transient score")
    axes_array[0].legend(
        handles=_fixed_legend_handles(config, include_board_events=aligned_mode or overlay_events),
        loc="upper right",
        fontsize=8,
    )
    figure.suptitle(
        (
            "Aligned Board-Event-Guided IMU Segmentation Verification"
            if aligned_mode
            else "Timestamp-Label IMU Segmentation Verification"
        )
        + "\n"
        + f"{user} / action {action} / dataset {dataset_id}"
        + (
            f"; offset = {float(alignment_offset_us):.6f} us"
            if aligned_mode or overlay_events
            else ""
        )
    )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=config.output_dpi)
    except OSError as error:
        raise SegmentationVerificationError(
            f"could not save segmentation verification image: {output}: {error}"
        ) from error
    finally:
        plt.close(figure)
    if not output.is_file() or output.stat().st_size == 0:
        raise SegmentationVerificationError(
            f"segmentation verification image was not written: {output}"
        )
    return SegmentationVerificationResult(
        output_path=output,
        panel_count=panel_count,
        recording_duration_s=duration,
        displayed_event_count=len(events),
        displayed_label_count=len(markers),
        displayed_segment_count=len(samples),
    )


def _draw_segments(
    axis: plt.Axes,
    samples: Sequence[SegmentedSample | BoardEventSegmentedSample],
    *,
    timestamps_us: np.ndarray,
    ring_start_us: float,
    panel_start_s: float,
    panel_stop_s: float,
    alpha: float,
    boundary_mode: str,
) -> None:
    for segment_index, sample in enumerate(samples):
        start_us, stop_us = _sample_window_us(
            sample, timestamps_us=timestamps_us, boundary_mode=boundary_mode
        )
        start = (start_us - ring_start_us) / 1_000_000.0
        stop = (stop_us - ring_start_us) / 1_000_000.0
        visible_start = max(start, panel_start_s)
        visible_stop = min(stop, panel_stop_s)
        if visible_start >= visible_stop:
            continue
        axis.axvspan(visible_start, visible_stop, color="tab:green", alpha=alpha)
        if panel_start_s <= start < panel_stop_s:
            axis.text(start, 0.98, f"{segment_index:03d}:{sample.label}", transform=axis.get_xaxis_transform(), va="top", fontsize=7, color="0.25")


def _draw_events(
    axis: plt.Axes,
    events: pd.DataFrame,
    elapsed: np.ndarray,
    *,
    panel_start_s: float,
    panel_stop_s: float,
    final_panel: bool,
    first_panel: bool,
    config: SegmentationVerificationConfig,
) -> None:
    for row, event_elapsed in zip(events.itertuples(index=False), elapsed, strict=True):
        if not _in_panel(event_elapsed, panel_start_s, panel_stop_s, final_panel):
            continue
        if bool(row.incomplete_touch):
            continue
        transient = bool(row.transient)
        if not bool(row.valid_touch) and not transient:
            continue
        is_press = row.event_type == "press"
        axis.axvline(
            event_elapsed,
            color="tab:red" if is_press else "tab:orange",
            linestyle="--",
            linewidth=(config.transient_event_linewidth if transient else config.valid_event_linewidth),
            alpha=(config.transient_event_alpha if transient else config.valid_event_alpha),
            label=(
                ("Transient Board press" if is_press else "Transient Board lift")
                if transient
                else ("Valid Board press" if is_press else "Valid Board lift")
            ) if first_panel else None,
        )


def _draw_labels(
    axis: plt.Axes,
    labels: Sequence[SegmentLabel],
    elapsed: np.ndarray,
    skip_reasons: Sequence[str | None],
    *,
    panel_start_s: float,
    panel_stop_s: float,
    final_panel: bool,
    first_panel: bool,
) -> None:
    for marker, marker_elapsed, reason in zip(labels, elapsed, skip_reasons, strict=True):
        if not _in_panel(marker_elapsed, panel_start_s, panel_stop_s, final_panel):
            continue
        skipped = reason is not None
        axis.axvline(
            marker_elapsed,
            color="tab:red" if skipped else "0.45",
            linestyle="-" if skipped else "--",
            linewidth=1.4 if skipped else 0.8,
            alpha=0.90 if skipped else 0.65,
            label=("Skipped label" if skipped else "Timestamp label") if first_panel else None,
        )
        if skipped:
            axis.text(marker_elapsed, 0.88, f"{marker.label} [skipped:{reason}]", transform=axis.get_xaxis_transform(), rotation=90, va="top", fontsize=6, color="tab:red")


def _validate_config(config: SegmentationVerificationConfig) -> None:
    if not isinstance(config, SegmentationVerificationConfig):
        raise SegmentationVerificationError("config must be SegmentationVerificationConfig")
    if _finite_float(config.panel_duration_s, name="panel_duration_s") <= 0.0:
        raise SegmentationVerificationError("panel_duration_s must be positive")
    if isinstance(config.output_dpi, bool) or not isinstance(config.output_dpi, int) or config.output_dpi <= 0:
        raise SegmentationVerificationError("output_dpi must be positive")
    valid_width = _finite_float(config.valid_event_linewidth, name="valid_event_linewidth")
    transient_width = _finite_float(config.transient_event_linewidth, name="transient_event_linewidth")
    if valid_width <= transient_width:
        raise SegmentationVerificationError("valid_event_linewidth must exceed transient_event_linewidth")
    for value, name in ((config.valid_event_alpha, "valid_event_alpha"), (config.transient_event_alpha, "transient_event_alpha")):
        if not 0.0 < _finite_float(value, name=name) <= 1.0:
            raise SegmentationVerificationError(f"{name} must be in (0, 1]")
    if not 0.0 < _finite_float(config.segment_alpha, name="segment_alpha") <= 0.12:
        raise SegmentationVerificationError("segment_alpha must be in (0, 0.12]")
    if not isinstance(config.overwrite, bool):
        raise SegmentationVerificationError("overwrite must be a boolean")


def _timestamps(values: np.ndarray) -> np.ndarray:
    try:
        timestamps = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SegmentationVerificationError("ring_timestamps_us must be numeric") from error
    if timestamps.ndim != 1 or len(timestamps) == 0 or not np.isfinite(timestamps).all():
        raise SegmentationVerificationError("ring_timestamps_us must be nonempty and finite")
    if np.any(np.diff(timestamps) < 0.0):
        raise SegmentationVerificationError("ring timestamps must be nondecreasing")
    return timestamps.copy()


def _score(values: np.ndarray, *, sample_count: int) -> np.ndarray:
    try:
        score = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SegmentationVerificationError("transient_score must be numeric") from error
    if score.ndim != 1 or len(score) != sample_count or not np.isfinite(score).all():
        raise SegmentationVerificationError("transient_score must be finite and match Ring timestamps")
    return score.copy()


def _events(value: pd.DataFrame | None, *, required: bool) -> pd.DataFrame:
    if value is None and not required:
        return pd.DataFrame(
            columns=(
                "event_type",
                "valid_touch",
                "transient",
                "incomplete_touch",
                "aligned_event_timestamp_us",
            )
        )
    if not isinstance(value, pd.DataFrame):
        raise SegmentationVerificationError("aligned_board_events is required in aligned mode")
    required = {"event_type", "valid_touch", "transient", "incomplete_touch", "aligned_event_timestamp_us"}
    missing = sorted(required - set(value.columns))
    if missing:
        raise SegmentationVerificationError("aligned_board_events is missing: " + ", ".join(missing))
    result = value.copy(deep=True)
    try:
        times = result["aligned_event_timestamp_us"].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SegmentationVerificationError("aligned Board timestamps must be numeric") from error
    if not np.isfinite(times).all():
        raise SegmentationVerificationError("aligned Board timestamps must be finite")
    return result


def _labels(values: Sequence[SegmentLabel]) -> tuple[SegmentLabel, ...]:
    labels = tuple(values)
    if not labels:
        raise SegmentationVerificationError("at least one label is required")
    previous: float | None = None
    for label in labels:
        if not isinstance(label, SegmentLabel) or not math.isfinite(label.timestamp_us):
            raise SegmentationVerificationError("labels must be finite SegmentLabel values")
        if previous is not None and label.timestamp_us <= previous:
            raise SegmentationVerificationError("labels must be strictly increasing")
        previous = label.timestamp_us
    return labels


def _validate_samples(
    samples: Sequence[SegmentedSample | BoardEventSegmentedSample],
    timestamps: np.ndarray,
    *,
    boundary_mode: str,
) -> None:
    previous_end: float | None = None
    for sample in samples:
        if boundary_mode == "aligned_board_events" and not isinstance(
            sample, BoardEventSegmentedSample
        ):
            raise SegmentationVerificationError(
                "aligned mode samples must be BoardEventSegmentedSample values"
            )
        if boundary_mode == "label" and not isinstance(sample, SegmentedSample):
            raise SegmentationVerificationError(
                "label mode samples must be SegmentedSample values"
            )
        start, end = _sample_window_us(
            sample, timestamps_us=timestamps, boundary_mode=boundary_mode
        )
        if not start < end:
            raise SegmentationVerificationError("final segmentation window is empty")
        if previous_end is not None and previous_end > start:
            raise SegmentationVerificationError("final segmentation windows overlap")
        if not 0 <= sample.start_sample_index < sample.stop_sample_index_exclusive <= len(timestamps):
            raise SegmentationVerificationError("segment sample indices are outside Ring range")
        previous_end = end


def _sample_window_us(
    sample: SegmentedSample | BoardEventSegmentedSample,
    *,
    timestamps_us: np.ndarray,
    boundary_mode: str,
) -> tuple[float, float]:
    if boundary_mode == "aligned_board_events":
        assert isinstance(sample, BoardEventSegmentedSample)
        return sample.final_start_timestamp_us, sample.final_end_timestamp_us
    assert isinstance(sample, SegmentedSample)
    return (
        sample.label_timestamp_us,
        (
            float(np.nextafter(timestamps_us[-1], np.inf))
            if sample.next_label_timestamp_us is None
            else sample.next_label_timestamp_us
        ),
    )


def _fixed_legend_handles(
    config: SegmentationVerificationConfig,
    *,
    include_board_events: bool,
) -> list[Line2D | Patch]:
    handles: list[Line2D | Patch] = [
        Line2D([], [], color="tab:blue", linewidth=0.8, label="Ring transient score"),
    ]
    if include_board_events:
        handles.extend(
            (
                Line2D([], [], color="tab:red", linestyle="--", linewidth=config.valid_event_linewidth, alpha=config.valid_event_alpha, label="Valid Board press"),
                Line2D([], [], color="tab:orange", linestyle="--", linewidth=config.valid_event_linewidth, alpha=config.valid_event_alpha, label="Valid Board lift"),
                Line2D([], [], color="tab:red", linestyle="--", linewidth=config.transient_event_linewidth, alpha=config.transient_event_alpha, label="Transient Board press"),
                Line2D([], [], color="tab:orange", linestyle="--", linewidth=config.transient_event_linewidth, alpha=config.transient_event_alpha, label="Transient Board lift"),
            )
        )
    handles.extend(
        (
            Line2D([], [], color="0.45", linestyle="--", linewidth=0.8, alpha=0.65, label="Timestamp label"),
            Line2D([], [], color="tab:red", linestyle="-", linewidth=1.4, alpha=0.90, label="Skipped label"),
            Patch(facecolor="tab:green", alpha=config.segment_alpha, label="Exported segment"),
        )
    )
    return handles


def _validate_output_path(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SegmentationVerificationError(f"verification output already exists: {path}")
    if path.exists() and not path.is_file():
        raise SegmentationVerificationError(f"verification output is not a regular file: {path}")


def _panel_mask(values: np.ndarray, start: float, stop: float, final_panel: bool) -> np.ndarray:
    return (values >= start) & ((values <= stop) if final_panel else (values < stop))


def _in_panel(value: float, start: float, stop: float, final_panel: bool) -> bool:
    return start <= value <= stop if final_panel else start <= value < stop


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise SegmentationVerificationError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SegmentationVerificationError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise SegmentationVerificationError(f"{name} must be finite")
    return number
