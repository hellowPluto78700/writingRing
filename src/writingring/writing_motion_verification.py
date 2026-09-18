"""Verification plotting for writing-vs-airborne motion ablations.

The visual contract intentionally mirrors segmentation_verification.py:
the original Ring transient score is the base signal; Board press/lift events,
timestamp labels, and exported segment geometry are overlaid on the same
10-second panels. This module adds explicit final-segment start/end markers
and darker spans for the retained writing/contact intervals.

This code visualizes already-published segmentation/mask geometry. It never
re-runs Board segmentation policy and never changes touch ownership.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from writingring.segmentation_verification import (
    SegmentationVerificationConfig,
    _draw_events,
    _in_panel,
    _panel_mask,
)


class WritingMotionVerificationError(ValueError):
    """Raised when writing-motion verification inputs are inconsistent."""


@dataclass(frozen=True, slots=True)
class WritingMotionVerificationResult:
    output_path: Path
    panel_count: int
    recording_duration_s: float
    displayed_event_count: int
    displayed_label_count: int
    displayed_segment_count: int
    displayed_writing_interval_count: int


def create_writing_motion_verification_figure(
    *,
    canonical_timestamps_us: np.ndarray,
    display_timestamps_us: np.ndarray,
    transient_score: np.ndarray,
    board_events: pd.DataFrame,
    segment_rows: Sequence[Mapping[str, Any]],
    writing_interval_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    user: str,
    action: str,
    dataset_id: int,
    config: SegmentationVerificationConfig = SegmentationVerificationConfig(),
) -> WritingMotionVerificationResult:
    """Render one full-recording writing-mask verification figure."""

    canonical = _timestamps(canonical_timestamps_us, name="canonical_timestamps_us")
    display = _timestamps(display_timestamps_us, name="display_timestamps_us")
    if len(display) != len(canonical):
        raise WritingMotionVerificationError(
            "display_timestamps_us must match canonical sample count"
        )
    score = _score(transient_score, len(canonical))
    events = _event_rows(board_events, dataset_id=dataset_id)
    labels = _dataset_segment_rows(segment_rows, dataset_id=dataset_id)
    exported = [row for row in labels if _is_exported(row)]
    intervals = [
        dict(row)
        for row in writing_interval_rows
        if _required_int(row.get("dataset_id"), "dataset_id") == dataset_id
    ]
    retained = [row for row in intervals if _as_bool(row.get("retained_in_segment", True))]
    if not labels:
        raise WritingMotionVerificationError(
            f"dataset {dataset_id} has no source label rows"
        )

    output = Path(output_path)
    if output.exists() and not config.overwrite:
        raise WritingMotionVerificationError(
            f"verification output already exists: {output}"
        )

    display_start = float(display[0])
    elapsed = (display - display_start) / 1_000_000.0
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

    event_elapsed = (
        events["aligned_event_timestamp_us"].to_numpy(dtype=np.float64)
        - display_start
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
        )
        _draw_exported_segments(
            axis,
            exported,
            display_timestamps_us=display,
            display_start_us=display_start,
            panel_start_s=panel_start,
            panel_stop_s=panel_stop,
            final_panel=final_panel,
            alpha=config.segment_alpha,
        )
        _draw_writing_intervals(
            axis,
            retained,
            exported,
            display_timestamps_us=display,
            display_start_us=display_start,
            panel_start_s=panel_start,
            panel_stop_s=panel_stop,
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
        _draw_timestamp_labels(
            axis,
            labels,
            canonical_timestamps_us=canonical,
            display_timestamps_us=display,
            display_start_us=display_start,
            panel_start_s=panel_start,
            panel_stop_s=panel_stop,
            final_panel=final_panel,
        )
        axis.set_xlim(
            panel_start,
            panel_stop if panel_stop > panel_start else panel_start + 1.0,
        )
        axis.set_ylim(0.0, y_max)
        axis.set_title(f"{panel_start:g}\N{EN DASH}{panel_stop:g} s")
        axis.grid(True, alpha=0.25)
        if panel_index == len(axes_array) - 1:
            axis.set_xlabel("Elapsed Ring time from recording start (s)")

    axes_array[0].set_ylabel("Transient score")
    axes_array[0].legend(
        handles=_legend_handles(config),
        loc="upper right",
        fontsize=8,
    )
    clipped = sum(
        _as_bool(row.get("segment_boundary_clipped_start", False))
        or _as_bool(row.get("segment_boundary_clipped_end", False))
        for row in intervals
    )
    figure.suptitle(
        "Writing-Motion Mask Verification\n"
        f"{user} / action {action} / dataset {dataset_id}; "
        f"segments={len(exported)}, retained writing intervals={len(retained)}, "
        f"segment-clipped intervals={clipped}\n"
        "Source transient score + source Board events/labels/segments; "
        "dark spans are published retained writing/contact intervals"
    )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=config.output_dpi)
    except OSError as error:
        raise WritingMotionVerificationError(
            f"could not save writing-motion verification image: {output}: {error}"
        ) from error
    finally:
        plt.close(figure)
    if not output.is_file() or output.stat().st_size == 0:
        raise WritingMotionVerificationError(
            f"writing-motion verification image was not written: {output}"
        )
    return WritingMotionVerificationResult(
        output_path=output,
        panel_count=panel_count,
        recording_duration_s=duration,
        displayed_event_count=len(events),
        displayed_label_count=len(labels),
        displayed_segment_count=len(exported),
        displayed_writing_interval_count=len(retained),
    )


def _draw_exported_segments(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    *,
    display_timestamps_us: np.ndarray,
    display_start_us: float,
    panel_start_s: float,
    panel_stop_s: float,
    final_panel: bool,
    alpha: float,
) -> None:
    for row in rows:
        segment_index = _required_int(row.get("segment_index"), "segment_index")
        start_index = _required_int(row.get("start_sample_index"), "start_sample_index")
        stop_index = _required_int(
            row.get("stop_sample_index_exclusive"),
            "stop_sample_index_exclusive",
        )
        start_s, stop_s = _index_window_seconds(
            start_index,
            stop_index,
            display_timestamps_us=display_timestamps_us,
            display_start_us=display_start_us,
        )
        visible_start = max(start_s, panel_start_s)
        visible_stop = min(stop_s, panel_stop_s)
        if visible_start < visible_stop:
            axis.axvspan(
                visible_start,
                visible_stop,
                color="tab:green",
                alpha=alpha,
            )
        label = str(row.get("label", ""))
        if _in_panel(start_s, panel_start_s, panel_stop_s, final_panel):
            axis.axvline(
                start_s,
                color="tab:green",
                linestyle="-",
                linewidth=1.0,
                alpha=0.75,
            )
            axis.text(
                start_s,
                0.98,
                f"{segment_index:03d}:{label} start",
                transform=axis.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=6,
                color="0.25",
            )
        if _in_panel(stop_s, panel_start_s, panel_stop_s, final_panel):
            axis.axvline(
                stop_s,
                color="tab:green",
                linestyle=":",
                linewidth=1.0,
                alpha=0.75,
            )
            axis.text(
                stop_s,
                0.78,
                f"{segment_index:03d}:{label} end",
                transform=axis.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="left",
                fontsize=6,
                color="0.25",
            )


def _draw_writing_intervals(
    axis: plt.Axes,
    intervals: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    *,
    display_timestamps_us: np.ndarray,
    display_start_us: float,
    panel_start_s: float,
    panel_stop_s: float,
) -> None:
    by_segment = {
        _required_int(row.get("segment_index"), "segment_index"): row
        for row in segments
    }
    for row in intervals:
        segment_index = _required_int(row.get("segment_index"), "segment_index")
        segment = by_segment.get(segment_index)
        if segment is None:
            raise WritingMotionVerificationError(
                f"writing interval references missing exported segment {segment_index}"
            )
        local_press = _optional_int(row.get("press_segment_local_index"))
        local_lift = _optional_int(row.get("lift_segment_local_index"))
        if local_press is None or local_lift is None:
            continue
        start_index = (
            _required_int(segment.get("start_sample_index"), "start_sample_index")
            + local_press
        )
        lift_index = (
            _required_int(segment.get("start_sample_index"), "start_sample_index")
            + local_lift
        )
        if not 0 <= start_index <= lift_index < len(display_timestamps_us):
            raise WritingMotionVerificationError(
                "retained writing interval indices are outside recording range"
            )
        start_s, stop_s = _index_window_seconds(
            start_index,
            lift_index + 1,
            display_timestamps_us=display_timestamps_us,
            display_start_us=display_start_us,
        )
        visible_start = max(start_s, panel_start_s)
        visible_stop = min(stop_s, panel_stop_s)
        if visible_start < visible_stop:
            axis.axvspan(
                visible_start,
                visible_stop,
                color="tab:green",
                alpha=0.30,
            )


def _draw_timestamp_labels(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    *,
    canonical_timestamps_us: np.ndarray,
    display_timestamps_us: np.ndarray,
    display_start_us: float,
    panel_start_s: float,
    panel_stop_s: float,
    final_panel: bool,
) -> None:
    for row in rows:
        timestamp = _finite_float(row.get("label_timestamp_us"), "label_timestamp_us")
        index = min(
            int(np.searchsorted(canonical_timestamps_us, timestamp, side="left")),
            len(display_timestamps_us) - 1,
        )
        elapsed = (float(display_timestamps_us[index]) - display_start_us) / 1_000_000.0
        if not _in_panel(elapsed, panel_start_s, panel_stop_s, final_panel):
            continue
        skip_reason = _optional_text(row.get("skip_reason"))
        skipped = not _is_exported(row)
        axis.axvline(
            elapsed,
            color="tab:red" if skipped else "0.45",
            linestyle="-" if skipped else "--",
            linewidth=1.4 if skipped else 0.8,
            alpha=0.90 if skipped else 0.65,
        )
        label = str(row.get("label", ""))
        source_index = _optional_int(row.get("source_label_index"))
        prefix = "" if source_index is None else f"{source_index:03d}:"
        label_text = (
            f"{prefix}{label} [skipped:{skip_reason or 'not_exported'}]"
            if skipped
            else f"{prefix}{label}"
        )
        axis.text(
            elapsed,
            0.88 if skipped else 0.70,
            label_text,
            transform=axis.get_xaxis_transform(),
            rotation=90,
            va="top",
            fontsize=6,
            color="tab:red" if skipped else "0.35",
        )


def _index_window_seconds(
    start_index: int,
    stop_index_exclusive: int,
    *,
    display_timestamps_us: np.ndarray,
    display_start_us: float,
) -> tuple[float, float]:
    if not 0 <= start_index < stop_index_exclusive <= len(display_timestamps_us):
        raise WritingMotionVerificationError(
            "sample window is outside display timestamp range"
        )
    start_us = float(display_timestamps_us[start_index])
    stop_us = (
        float(display_timestamps_us[stop_index_exclusive])
        if stop_index_exclusive < len(display_timestamps_us)
        else float(np.nextafter(display_timestamps_us[-1], np.inf))
    )
    return (
        (start_us - display_start_us) / 1_000_000.0,
        (stop_us - display_start_us) / 1_000_000.0,
    )


def _legend_handles(
    config: SegmentationVerificationConfig,
) -> list[Line2D | Patch]:
    return [
        Line2D([], [], color="tab:blue", linewidth=0.8, label="Ring transient score"),
        Line2D(
            [], [], color="tab:red", linestyle="--",
            linewidth=config.valid_event_linewidth,
            alpha=config.valid_event_alpha,
            label="Valid Board press",
        ),
        Line2D(
            [], [], color="tab:orange", linestyle="--",
            linewidth=config.valid_event_linewidth,
            alpha=config.valid_event_alpha,
            label="Valid Board lift",
        ),
        Line2D(
            [], [], color="tab:red", linestyle="--",
            linewidth=config.transient_event_linewidth,
            alpha=config.transient_event_alpha,
            label="Transient Board press",
        ),
        Line2D(
            [], [], color="tab:orange", linestyle="--",
            linewidth=config.transient_event_linewidth,
            alpha=config.transient_event_alpha,
            label="Transient Board lift",
        ),
        Line2D(
            [], [], color="0.45", linestyle="--", linewidth=0.8, alpha=0.65,
            label="Timestamp label",
        ),
        Line2D(
            [], [], color="tab:red", linestyle="-", linewidth=1.4, alpha=0.90,
            label="Skipped label",
        ),
        Patch(facecolor="tab:green", alpha=config.segment_alpha, label="Exported segment"),
        Patch(facecolor="tab:green", alpha=0.30, label="Retained writing/contact"),
        Line2D(
            [], [], color="tab:green", linestyle="-", linewidth=1.0, alpha=0.75,
            label="Final segment start",
        ),
        Line2D(
            [], [], color="tab:green", linestyle=":", linewidth=1.0, alpha=0.75,
            label="Final segment end",
        ),
    ]


def _event_rows(value: pd.DataFrame, *, dataset_id: int) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise WritingMotionVerificationError("board_events must be a DataFrame")
    required = {
        "dataset_id",
        "event_type",
        "valid_touch",
        "transient",
        "incomplete_touch",
        "aligned_event_timestamp_us",
    }
    missing = sorted(required - set(value.columns))
    if missing:
        raise WritingMotionVerificationError(
            "board_events missing columns: " + ", ".join(missing)
        )
    rows = value.loc[
        pd.to_numeric(value["dataset_id"], errors="coerce") == dataset_id
    ].copy()
    if rows.empty:
        raise WritingMotionVerificationError(
            f"dataset {dataset_id} has no Board-event rows"
        )
    times = rows["aligned_event_timestamp_us"].to_numpy(dtype=np.float64)
    if not np.isfinite(times).all():
        raise WritingMotionVerificationError("Board-event timestamps must be finite")
    return rows


def _dataset_segment_rows(
    rows: Sequence[Mapping[str, Any]], *, dataset_id: int
) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in rows
        if _required_int(row.get("dataset_id"), "dataset_id") == dataset_id
    ]
    selected.sort(
        key=lambda row: _required_int(row.get("source_label_index"), "source_label_index")
    )
    return selected


def _is_exported(row: Mapping[str, Any]) -> bool:
    segment_index = _optional_int(row.get("segment_index"))
    raw = row.get("exported", segment_index is not None)
    return segment_index is not None and _as_bool(raw)


def _timestamps(values: np.ndarray, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise WritingMotionVerificationError(f"{name} must be numeric") from error
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise WritingMotionVerificationError(f"{name} must be finite and nonempty")
    if np.any(np.diff(array) < 0.0):
        raise WritingMotionVerificationError(f"{name} must be nondecreasing")
    return array.copy()


def _score(values: np.ndarray, sample_count: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise WritingMotionVerificationError("transient_score must be numeric") from error
    if array.shape != (sample_count,) or not np.isfinite(array).all():
        raise WritingMotionVerificationError(
            "transient_score must be finite and match recording sample count"
        )
    return array.copy()


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise WritingMotionVerificationError(
            f"expected integer-like value, got {value!r}"
        ) from error
    if not math.isfinite(numeric) or numeric != int(numeric):
        raise WritingMotionVerificationError(
            f"expected integer-like value, got {value!r}"
        )
    return int(numeric)


def _required_int(value: object, name: str) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise WritingMotionVerificationError(f"{name} is required")
    return parsed


def _finite_float(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise WritingMotionVerificationError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise WritingMotionVerificationError(f"{name} must be finite")
    return number


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    token = str(value).strip().lower()
    if token in {"true", "1"}:
        return True
    if token in {"false", "0", ""}:
        return False
    raise WritingMotionVerificationError(f"expected boolean-like value, got {value!r}")
