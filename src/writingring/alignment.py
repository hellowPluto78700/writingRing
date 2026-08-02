"""Group-level Ring candidate analysis around Board press events.

The functions in this module operate on caller-supplied tables and timestamp
arrays.  They never reorder, repair, or modify source data.  Absolute
timestamps are used only for lookup; interpolation is performed on local
relative-time values to retain useful precision for large microsecond clocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence
import warnings

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


GROUP_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "board_event_index",
    "board_timestamp_raw",
    "target_imu_timestamp",
    "center_imu_sample_index",
    "window_start_sample_index",
    "window_stop_sample_index_exclusive",
    "nearest_timestamp_error_us",
    "left_coverage_complete",
    "right_coverage_complete",
    "source_sample_count",
)
SKIPPED_EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "board_event_index",
    "board_timestamp_raw",
    "target_imu_timestamp",
    "source_sample_count",
    "reason",
)


class AlignmentAnalysisError(ValueError):
    """Raised when alignment analysis inputs cannot be used safely."""


class AlignmentPlotError(AlignmentAnalysisError):
    """Raised when a group alignment figure cannot be produced or saved."""


@dataclass(frozen=True, slots=True)
class GroupCandidateWindows:
    """Common-grid Ring windows associated with accepted Board presses."""

    relative_time_s: np.ndarray
    signal_matrices: dict[str, np.ndarray]
    transient_score_matrix: np.ndarray
    metadata: pd.DataFrame
    skipped_events: pd.DataFrame
    signal_columns: tuple[str, ...]

    @property
    def event_count(self) -> int:
        """Return the number of accepted Board events."""

        return int(self.transient_score_matrix.shape[0])

    @property
    def time_count(self) -> int:
        """Return the number of common relative-time samples."""

        return int(self.relative_time_s.size)


@dataclass(frozen=True, slots=True)
class ConsensusShift:
    """Candidate constant-offset refinement from a group consensus feature."""

    consensus_shift_s: float
    candidate_refined_offset_us: float
    consensus_transient_score: np.ndarray
    search_range_s: tuple[float, float]
    contributing_event_count: int


def nearest_sorted_index(
    sorted_values: np.ndarray,
    target: float,
) -> int:
    """Return the nearest index in a finite, strictly increasing 1-D array.

    Targets outside the data range resolve to the first or last sample.  An
    exact midpoint is resolved to the lower index for deterministic behavior.
    """

    values = _validated_strictly_increasing(sorted_values)
    target_value = _finite_float(target, name="target timestamp")
    return _nearest_sorted_index_validated(values, target_value)


def robust_normalize_rows(values: np.ndarray) -> np.ndarray:
    """Median/MAD-normalize each matrix row while preserving NaN positions.

    A row with no finite values remains all-NaN.  If a row's MAD is zero, its
    finite positions are set to zero so division by zero cannot manufacture
    infinities.  The input array is never modified.
    """

    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentAnalysisError(
            "row normalization input must be a numeric matrix"
        ) from error
    if matrix.ndim != 2:
        raise AlignmentAnalysisError(
            f"row normalization input must be two-dimensional, got {matrix.shape}"
        )

    normalized = np.full(matrix.shape, np.nan, dtype=np.float64)
    for row_index, row in enumerate(matrix):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        median = float(np.median(row[finite]))
        centered = row[finite] - median
        mad = float(np.median(np.abs(centered)))
        scale = 1.4826 * mad
        if scale > 0.0 and np.isfinite(scale):
            normalized[row_index, finite] = centered / scale
        else:
            normalized[row_index, finite] = 0.0
    return normalized


def build_group_candidate_windows(
    press_events: pd.DataFrame,
    ring_dataframe: pd.DataFrame,
    ring_timestamp_reconstructed: np.ndarray,
    transient_score: np.ndarray,
    *,
    signal_columns: Sequence[str],
    coarse_offset_us: float,
    radius_samples: int,
    imu_sampling_rate_hz: float,
) -> GroupCandidateWindows:
    """Interpolate every usable Board-press Ring window onto one time grid.

    Window bounds are timestamp-based.  Samples outside available Ring
    coverage remain NaN rather than being extrapolated.  Events with fewer
    than two source samples in the requested interval are returned in the
    skipped-event table instead of being silently discarded.
    """

    if not isinstance(press_events, pd.DataFrame):
        raise AlignmentAnalysisError("press_events must be a pandas DataFrame")
    required_event_columns = {"event_index", "frame_timestamp_raw"}
    missing_event_columns = required_event_columns - set(press_events.columns)
    if missing_event_columns:
        raise AlignmentAnalysisError(
            "press_events is missing column(s): "
            + ", ".join(sorted(missing_event_columns))
        )
    if press_events.empty:
        raise AlignmentAnalysisError("press_events must contain at least one event")
    if not isinstance(ring_dataframe, pd.DataFrame) or ring_dataframe.empty:
        raise AlignmentAnalysisError("ring_dataframe must contain Ring samples")

    columns = tuple(signal_columns)
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise AlignmentAnalysisError("signal_columns must contain nonempty names")
    if len(set(columns)) != len(columns):
        raise AlignmentAnalysisError("signal_columns must not contain duplicates")
    missing_signals = tuple(column for column in columns if column not in ring_dataframe)
    if missing_signals:
        raise AlignmentAnalysisError(
            "ring_dataframe is missing signal column(s): " + ", ".join(missing_signals)
        )

    timestamps = _validated_strictly_increasing(ring_timestamp_reconstructed)
    score = _validated_vector(transient_score, name="transient_score")
    sample_count = len(ring_dataframe)
    if timestamps.size != sample_count or score.size != sample_count:
        raise AlignmentAnalysisError(
            "ring_dataframe, ring_timestamp_reconstructed, and transient_score "
            f"must have equal lengths, got {sample_count}, {timestamps.size}, {score.size}"
        )
    if isinstance(radius_samples, bool) or not isinstance(radius_samples, (int, np.integer)):
        raise AlignmentAnalysisError("radius_samples must be a positive integer")
    radius = int(radius_samples)
    if radius <= 0:
        raise AlignmentAnalysisError("radius_samples must be a positive integer")
    sampling_rate = _finite_float(
        imu_sampling_rate_hz,
        name="IMU sampling rate",
        positive=True,
    )
    offset = _finite_float(coarse_offset_us, name="coarse offset")

    try:
        signal_values = ring_dataframe.loc[:, list(columns)].to_numpy(
            dtype=np.float64,
            copy=True,
        )
    except (TypeError, ValueError) as error:
        raise AlignmentAnalysisError(
            "Ring signal columns must contain numeric values"
        ) from error

    relative_time_s = (
        np.arange(-radius, radius + 1, dtype=np.float64) / sampling_rate
    )
    left_delta_us = float(relative_time_s[0] * 1_000_000.0)
    right_delta_us = float(relative_time_s[-1] * 1_000_000.0)
    matrix_rows: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    score_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, int | float | bool]] = []
    skipped_rows: list[dict[str, int | float | str]] = []

    for row_number, event in press_events.iterrows():
        event_index = _integer_event_value(event["event_index"], row_number=row_number)
        board_timestamp = _integer_timestamp_value(
            event["frame_timestamp_raw"],
            row_number=row_number,
        )
        target_timestamp = float(board_timestamp) + offset
        window_left = target_timestamp + left_delta_us
        window_right = target_timestamp + right_delta_us
        start = int(np.searchsorted(timestamps, window_left, side="left"))
        stop = int(np.searchsorted(timestamps, window_right, side="right"))
        source_window_count = stop - start
        if source_window_count < 2:
            skipped_rows.append(
                {
                    "board_event_index": event_index,
                    "board_timestamp_raw": board_timestamp,
                    "target_imu_timestamp": target_timestamp,
                    "source_sample_count": source_window_count,
                    "reason": "candidate window contains fewer than two Ring samples",
                }
            )
            continue

        center_index = _nearest_sorted_index_validated(timestamps, target_timestamp)
        local_time_s = (timestamps[start:stop] - target_timestamp) / 1_000_000.0
        for column_index, column in enumerate(columns):
            matrix_rows[column].append(
                np.interp(
                    relative_time_s,
                    local_time_s,
                    signal_values[start:stop, column_index],
                    left=np.nan,
                    right=np.nan,
                )
            )
        score_rows.append(
            np.interp(
                relative_time_s,
                local_time_s,
                score[start:stop],
                left=np.nan,
                right=np.nan,
            )
        )
        metadata_rows.append(
            {
                "board_event_index": event_index,
                "board_timestamp_raw": board_timestamp,
                "target_imu_timestamp": target_timestamp,
                "center_imu_sample_index": center_index,
                "window_start_sample_index": start,
                "window_stop_sample_index_exclusive": stop,
                "nearest_timestamp_error_us": float(
                    timestamps[center_index] - target_timestamp
                ),
                "left_coverage_complete": bool(timestamps[0] <= window_left),
                "right_coverage_complete": bool(timestamps[-1] >= window_right),
                "source_sample_count": source_window_count,
            }
        )

    time_count = relative_time_s.size
    signal_matrices = {
        column: _stack_rows(rows, time_count=time_count)
        for column, rows in matrix_rows.items()
    }
    transient_matrix = _stack_rows(score_rows, time_count=time_count)
    metadata = pd.DataFrame(metadata_rows, columns=GROUP_METADATA_COLUMNS)
    skipped_events = pd.DataFrame(skipped_rows, columns=SKIPPED_EVENT_COLUMNS)
    return GroupCandidateWindows(
        relative_time_s=relative_time_s,
        signal_matrices=signal_matrices,
        transient_score_matrix=transient_matrix,
        metadata=metadata,
        skipped_events=skipped_events,
        signal_columns=columns,
    )


def estimate_consensus_shift(
    relative_time_s: np.ndarray,
    transient_score_matrix: np.ndarray,
    *,
    coarse_offset_us: float,
    search_range_s: tuple[float, float] = (-0.30, 0.30),
) -> ConsensusShift:
    """Estimate a candidate offset correction from median normalized scores."""

    times = _validated_strictly_increasing(relative_time_s)
    try:
        scores = np.asarray(transient_score_matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentAnalysisError(
            "transient_score_matrix must be numeric"
        ) from error
    if scores.ndim != 2 or scores.shape[0] == 0:
        raise AlignmentAnalysisError(
            "transient_score_matrix must contain at least one event row"
        )
    if scores.shape[1] != times.size:
        raise AlignmentAnalysisError(
            "transient_score_matrix time dimension must match relative_time_s"
        )
    if not isinstance(search_range_s, tuple) or len(search_range_s) != 2:
        raise AlignmentAnalysisError("search_range_s must be a (minimum, maximum) tuple")
    search_min = _finite_float(search_range_s[0], name="search range minimum")
    search_max = _finite_float(search_range_s[1], name="search range maximum")
    if search_min > search_max:
        raise AlignmentAnalysisError(
            "search range minimum must be <= search range maximum"
        )
    offset = _finite_float(coarse_offset_us, name="coarse offset")

    normalized = robust_normalize_rows(scores)
    consensus = _nanmedian_columns(normalized)
    search_mask = (times >= search_min) & (times <= search_max)
    usable = search_mask & np.isfinite(consensus)
    if not np.any(usable):
        raise AlignmentAnalysisError(
            "consensus search range contains no finite group values"
        )
    candidate_indices = np.flatnonzero(usable)
    consensus_index = int(candidate_indices[np.argmax(consensus[usable])])
    shift_s = float(times[consensus_index])
    return ConsensusShift(
        consensus_shift_s=shift_s,
        candidate_refined_offset_us=offset + shift_s * 1_000_000.0,
        consensus_transient_score=consensus,
        search_range_s=(search_min, search_max),
        contributing_event_count=int(scores.shape[0]),
    )


def plot_group_candidate_overlay(
    windows: GroupCandidateWindows,
    *,
    normalized: bool = True,
    output_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot all event windows, their median, and IQR in seven panels."""

    _validate_group_windows(windows)
    matrices = [windows.signal_matrices[column] for column in windows.signal_columns]
    matrices.append(windows.transient_score_matrix)
    labels = [*windows.signal_columns, "transient_score"]

    figure, axes = plt.subplots(
        len(matrices),
        1,
        figsize=(14, 13),
        sharex=True,
        layout="constrained",
    )
    try:
        for axis, values, label in zip(axes, matrices, labels, strict=True):
            plotted = robust_normalize_rows(values) if normalized else values
            for row in plotted:
                axis.plot(
                    windows.relative_time_s,
                    row,
                    color="0.45",
                    linewidth=0.55,
                    alpha=0.20,
                )
            median, lower, upper = _nan_summary_columns(plotted)
            axis.fill_between(
                windows.relative_time_s,
                lower,
                upper,
                color="tab:blue",
                alpha=0.22,
                label="cross-event IQR",
            )
            axis.plot(
                windows.relative_time_s,
                median,
                color="tab:blue",
                linewidth=1.8,
                label="cross-event median",
            )
            axis.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel(
            "Relative time (s); zero = Board press timestamp + coarse_offset_us"
        )
        mode = "per-event robust-normalized" if normalized else "raw values"
        figure.suptitle(
            f"Group Board-press Ring candidates ({mode}; n={windows.event_count})"
        )
        _save_and_show(figure, output_path=output_path, show=show)
    except Exception:
        plt.close(figure)
        raise
    return figure


def plot_group_candidate_heatmap(
    windows: GroupCandidateWindows,
    *,
    signal_name: str = "transient_score",
    normalized: bool = True,
    output_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot one event-by-relative-time group matrix as a heatmap."""

    _validate_group_windows(windows)
    if signal_name == "transient_score":
        values = windows.transient_score_matrix
    elif signal_name in windows.signal_matrices:
        values = windows.signal_matrices[signal_name]
    else:
        raise AlignmentPlotError(f"unknown group heatmap signal: {signal_name}")
    plotted = robust_normalize_rows(values) if normalized else values

    figure, axis = plt.subplots(figsize=(14, 7), layout="constrained")
    try:
        finite_values = plotted[np.isfinite(plotted)]
        if finite_values.size == 0:
            raise AlignmentPlotError("group heatmap contains no finite values")
        image_options: dict[str, object] = {}
        color_map = "viridis"
        if normalized:
            limit = float(np.percentile(np.abs(finite_values), 98.0))
            if limit > 0.0:
                image_options.update(vmin=-limit, vmax=limit)
            color_map = "coolwarm"
        image = axis.imshow(
            np.ma.masked_invalid(plotted),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(
                float(windows.relative_time_s[0]),
                float(windows.relative_time_s[-1]),
                -0.5,
                windows.event_count - 0.5,
            ),
            cmap=color_map,
            **image_options,
        )
        event_indices = windows.metadata["board_event_index"].to_numpy(dtype=int)
        axis.set_yticks(np.arange(windows.event_count), labels=event_indices)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        axis.set_xlabel(
            "Relative time (s); zero = Board press timestamp + coarse_offset_us"
        )
        axis.set_ylabel("Board event index")
        mode = "per-event robust-normalized" if normalized else "raw"
        axis.set_title(f"Group candidate heatmap: {signal_name} ({mode})")
        figure.colorbar(image, ax=axis, label=f"{signal_name} ({mode})")
        _save_and_show(figure, output_path=output_path, show=show)
    except Exception:
        plt.close(figure)
        raise
    return figure


def _validated_strictly_increasing(values: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentAnalysisError("timestamps must be numeric") from error
    if array.ndim != 1:
        raise AlignmentAnalysisError(
            f"timestamps must be one-dimensional, got {array.shape}"
        )
    if array.size == 0:
        raise AlignmentAnalysisError("timestamps must be nonempty")
    if not np.isfinite(array).all():
        raise AlignmentAnalysisError("timestamps must contain only finite values")
    if not np.all(np.diff(array) > 0.0):
        raise AlignmentAnalysisError("timestamps must be strictly increasing")
    return np.array(array, dtype=np.float64, copy=True)


def _validated_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentAnalysisError(f"{name} must be numeric") from error
    if array.ndim != 1:
        raise AlignmentAnalysisError(f"{name} must be one-dimensional")
    return np.array(array, dtype=np.float64, copy=True)


def _nearest_sorted_index_validated(values: np.ndarray, target: float) -> int:
    insertion = int(np.searchsorted(values, target, side="left"))
    if insertion <= 0:
        return 0
    if insertion >= values.size:
        return int(values.size - 1)
    lower = insertion - 1
    if target - values[lower] <= values[insertion] - target:
        return lower
    return insertion


def _finite_float(value: object, *, name: str, positive: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AlignmentAnalysisError(f"{name} must be a finite number") from error
    if not np.isfinite(converted):
        raise AlignmentAnalysisError(f"{name} must be finite")
    if positive and converted <= 0.0:
        raise AlignmentAnalysisError(f"{name} must be positive")
    return converted


def _integer_event_value(value: object, *, row_number: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AlignmentAnalysisError(
            f"press event row {row_number!r} has a non-integer event_index"
        )
    return int(value)


def _integer_timestamp_value(value: object, *, row_number: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AlignmentAnalysisError(
            f"press event row {row_number!r} has a non-integer frame_timestamp_raw"
        )
    return int(value)


def _stack_rows(rows: list[np.ndarray], *, time_count: int) -> np.ndarray:
    if not rows:
        return np.empty((0, time_count), dtype=np.float64)
    return np.stack(rows).astype(np.float64, copy=False)


def _nanmedian_columns(values: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=0)


def _nan_summary_columns(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        lower = np.nanpercentile(values, 25.0, axis=0)
        upper = np.nanpercentile(values, 75.0, axis=0)
    return median, lower, upper


def _validate_group_windows(windows: GroupCandidateWindows) -> None:
    if not isinstance(windows, GroupCandidateWindows):
        raise AlignmentPlotError("group plotting requires GroupCandidateWindows")
    if windows.event_count == 0:
        raise AlignmentPlotError("group candidate result has no accepted events")
    if windows.metadata.shape[0] != windows.event_count:
        raise AlignmentPlotError("group metadata row count does not match event count")
    if tuple(windows.signal_matrices) != windows.signal_columns:
        raise AlignmentPlotError("group signal matrices do not match signal_columns")
    expected_shape = (windows.event_count, windows.time_count)
    if windows.transient_score_matrix.shape != expected_shape:
        raise AlignmentPlotError("transient-score matrix has an inconsistent shape")
    if any(matrix.shape != expected_shape for matrix in windows.signal_matrices.values()):
        raise AlignmentPlotError("one or more signal matrices have inconsistent shapes")


def _save_and_show(
    figure: Figure,
    *,
    output_path: str | Path | None,
    show: bool,
) -> None:
    if output_path is not None:
        try:
            path = Path(output_path)
        except TypeError as error:
            raise AlignmentPlotError(
                f"invalid alignment output path: {output_path!r}"
            ) from error
        if path.exists() and not path.is_file():
            raise AlignmentPlotError(
                f"alignment output path is not a regular file path: {path}"
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=160)
        except (OSError, ValueError) as error:
            raise AlignmentPlotError(
                f"cannot save alignment figure to {path}: {error}"
            ) from error
    if show:
        plt.show()
