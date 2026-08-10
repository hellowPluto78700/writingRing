"""Automatic constant-offset alignment of Board events to Ring transients.

Board press/lift labels remain Board-derived.  Ring peaks are unlabeled motion
transients and are never interpreted here as semantic press or lift classes.
All functions preserve caller-provided row and timestamp order and never
modify raw timestamp columns in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


_MatchScore = tuple[int, int, float, float]


BOARD_EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "event_type",
    "global_frame_index",
    "frame_timestamp_raw",
    "chunk_index",
    "contact_count",
    "paired_touch_index",
    "duration_frames",
    "transient",
    "valid_touch",
    "incomplete_touch",
)
TOUCH_PAIR_COLUMNS: Final[tuple[str, ...]] = (
    "paired_touch_index",
    "press_event_index",
    "lift_event_index",
    "press_global_frame_index",
    "lift_global_frame_index",
    "press_timestamp_raw",
    "lift_timestamp_raw",
    "duration_frames",
    "duration_s",
    "transient",
    "valid_touch",
    "incomplete_touch",
)
PEAK_REGION_COLUMNS: Final[tuple[str, ...]] = (
    "peak_index",
    "peak_timestamp",
    "peak_elapsed_s",
    "left_index",
    "right_index",
    "left_timestamp",
    "right_timestamp",
    "width_s",
    "height",
    "prominence",
)
MATCH_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "event_type",
    "paired_touch_index",
    "board_timestamp_raw",
    "shifted_event_timestamp",
    "matched",
    "matched_peak_index",
    "matched_peak_timestamp",
    "peak_left_timestamp",
    "peak_right_timestamp",
    "normalized_peak_distance",
    "residual_us",
    "peak_height",
    "peak_prominence",
)


class EventAlignmentError(ValueError):
    """Raised when event alignment inputs or results are unusable."""


class AlignmentFailureError(EventAlignmentError):
    """Raised when no candidate meets minimum alignment confidence."""


class InitialIntervalNoUsablePairError(EventAlignmentError):
    """Raised when a backward jump leaves no valid pair in the initial prefix.

    The initial Board interval is selected by row order, not by timestamp
    order.  This exception carries the boundary diagnostics needed by callers
    that may later classify this one precise condition separately from other
    alignment failures.
    """

    reason: Final[str] = "initial_interval_no_usable_pair"

    def __init__(
        self,
        *,
        previous_global_frame_index: int,
        next_global_frame_index: int,
        previous_timestamp_raw: object,
        next_timestamp_raw: object,
        prefix_boundary_position: int,
        last_pre_jump_global_frame_index: int,
        total_global_valid_pair_count: int,
        usable_prefix_valid_pair_count: int,
    ) -> None:
        self.previous_global_frame_index = previous_global_frame_index
        self.next_global_frame_index = next_global_frame_index
        self.previous_timestamp_raw = previous_timestamp_raw
        self.next_timestamp_raw = next_timestamp_raw
        self.prefix_boundary_position = prefix_boundary_position
        self.last_pre_jump_global_frame_index = last_pre_jump_global_frame_index
        self.total_global_valid_pair_count = total_global_valid_pair_count
        self.usable_prefix_valid_pair_count = usable_prefix_valid_pair_count

        # These explicit aliases keep the distinction between the jump's
        # adjacent frames and the exclusive positional boundary unambiguous.
        self.previous_frame_timestamp_raw = previous_timestamp_raw
        self.next_frame_timestamp_raw = next_timestamp_raw
        self.exclusive_prefix_boundary_position = prefix_boundary_position
        self.inclusive_last_pre_jump_global_frame_index = (
            last_pre_jump_global_frame_index
        )
        self.diagnostics: dict[str, object] = {
            "previous_global_frame_index": previous_global_frame_index,
            "next_global_frame_index": next_global_frame_index,
            "previous_timestamp_raw": previous_timestamp_raw,
            "next_timestamp_raw": next_timestamp_raw,
            "prefix_boundary_position": prefix_boundary_position,
            "last_pre_jump_global_frame_index": last_pre_jump_global_frame_index,
            "total_global_valid_pair_count": total_global_valid_pair_count,
            "usable_prefix_valid_pair_count": usable_prefix_valid_pair_count,
        }
        super().__init__(
            "the initial Board timestamp interval contains no usable valid "
            f"press/lift pair (boundary position {prefix_boundary_position}, "
            f"global valid pairs {total_global_valid_pair_count}, "
            f"usable prefix pairs {usable_prefix_valid_pair_count})"
        )


@dataclass(frozen=True, slots=True)
class BoardEventDetection:
    """All detected Board transitions and their press/lift touch pairs."""

    events: pd.DataFrame
    touch_pairs: pd.DataFrame
    minimum_duration_frames: int


@dataclass(frozen=True, slots=True)
class BoardIntervalSelection:
    """Dynamically selected Board interval and its diagnostics."""

    frames: pd.DataFrame
    contacts: pd.DataFrame
    events: pd.DataFrame
    touch_pairs: pd.DataFrame
    metadata: dict[str, object]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PeakDetectionConfig:
    """Robust transient-region detection parameters."""

    smoothing_window_samples: int = 5
    prominence_window_samples: int = 80
    prominence_mad_multiplier: float = 4.0
    minimum_prominence: float = 1.0
    boundary_prominence_fraction: float = 0.5
    merge_gap_samples: int = 4


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    """Candidate generation, ranking, and confidence parameters."""

    offset_search_range_us: tuple[float, float] | None = None
    offset_cluster_width_us: float = 20_000.0
    maximum_offset_candidates: int = 200
    minimum_event_coverage_ratio: float = 0.40
    minimum_valid_touch_pairs: int = 5
    ambiguity_distance_margin: float = 0.03
    press_lift_coverage_difference_warning: float = 0.20
    residual_trend_warning_ms: float = 10.0
    outside_ring_fraction_warning: float = 0.20
    touch_duration_difference_warning_ms: float = 150.0


@dataclass(frozen=True, slots=True)
class SequenceAlignmentResult:
    """Best monotonic event-to-peak alignment and ranked candidates."""

    success: bool
    best_offset_us: float
    second_best_offset_us: float | None
    candidates: pd.DataFrame
    event_matches: pd.DataFrame
    touch_pair_matches: pd.DataFrame
    report: dict[str, object]
    warnings: tuple[str, ...]

    @property
    def work_axis_offset_us(self) -> float:
        """Return the offset selected in the strict alignment work axis."""

        return self.best_offset_us


@dataclass(frozen=True, slots=True)
class EventVisibility:
    """Counts of Board event types inside and outside a plotted Ring range."""

    press_inside: int
    press_outside: int
    lift_inside: int
    lift_outside: int


def detect_board_events(
    board_frames: pd.DataFrame,
    *,
    minimum_duration_frames: int = 3,
) -> BoardEventDetection:
    """Detect ordered Board press/lift transitions without inventing event zero."""

    frames = _validated_board_frames(board_frames)
    minimum_duration = _positive_integer(
        minimum_duration_frames,
        name="minimum touch duration frames",
    )
    occupied = frames["contact_count"].to_numpy(dtype=np.int64) > 0
    transition = np.diff(
        occupied.astype(np.int8),
        prepend=occupied[0].astype(np.int8),
    )
    local_indices = np.flatnonzero(transition != 0)

    raw_rows: list[dict[str, object]] = []
    for event_index, local_index in enumerate(local_indices):
        frame = frames.iloc[int(local_index)]
        raw_rows.append(
            {
                "event_index": event_index,
                "event_type": "press" if transition[local_index] == 1 else "lift",
                "global_frame_index": int(frame["global_frame_index"]),
                "frame_timestamp_raw": int(frame["frame_timestamp_raw"]),
                "chunk_index": int(frame["chunk_index"]),
                "contact_count": int(frame["contact_count"]),
                "paired_touch_index": pd.NA,
                "duration_frames": pd.NA,
                "transient": False,
                "valid_touch": False,
                "incomplete_touch": False,
            }
        )

    touch_rows: list[dict[str, object]] = []
    touch_index = 0
    event_position = 0
    while event_position < len(raw_rows):
        event = raw_rows[event_position]
        if event["event_type"] != "press":
            event_position += 1
            continue
        lift_position: int | None = None
        next_position = event_position + 1
        while next_position < len(raw_rows):
            next_type = raw_rows[next_position]["event_type"]
            if next_type == "press":
                break
            if next_type == "lift":
                lift_position = next_position
                break
            next_position += 1

        press_global = int(event["global_frame_index"])
        press_timestamp = int(event["frame_timestamp_raw"])
        event["paired_touch_index"] = touch_index
        event["incomplete_touch"] = lift_position is None
        if lift_position is None:
            touch_rows.append(
                {
                    "paired_touch_index": touch_index,
                    "press_event_index": int(event["event_index"]),
                    "lift_event_index": pd.NA,
                    "press_global_frame_index": press_global,
                    "lift_global_frame_index": pd.NA,
                    "press_timestamp_raw": press_timestamp,
                    "lift_timestamp_raw": pd.NA,
                    "duration_frames": pd.NA,
                    "duration_s": np.nan,
                    "transient": False,
                    "valid_touch": False,
                    "incomplete_touch": True,
                }
            )
        else:
            lift = raw_rows[lift_position]
            lift_global = int(lift["global_frame_index"])
            lift_timestamp = int(lift["frame_timestamp_raw"])
            duration_frames = lift_global - press_global
            transient = duration_frames < minimum_duration
            valid = not transient
            for paired_event in (event, lift):
                paired_event["paired_touch_index"] = touch_index
                paired_event["duration_frames"] = duration_frames
                paired_event["transient"] = transient
                paired_event["valid_touch"] = valid
                paired_event["incomplete_touch"] = False
            touch_rows.append(
                {
                    "paired_touch_index": touch_index,
                    "press_event_index": int(event["event_index"]),
                    "lift_event_index": int(lift["event_index"]),
                    "press_global_frame_index": press_global,
                    "lift_global_frame_index": lift_global,
                    "press_timestamp_raw": press_timestamp,
                    "lift_timestamp_raw": lift_timestamp,
                    "duration_frames": duration_frames,
                    "duration_s": (lift_timestamp - press_timestamp) / 1_000_000.0,
                    "transient": transient,
                    "valid_touch": valid,
                    "incomplete_touch": False,
                }
            )
        touch_index += 1
        event_position += 1

    events = pd.DataFrame(raw_rows, columns=BOARD_EVENT_COLUMNS)
    touch_pairs = pd.DataFrame(touch_rows, columns=TOUCH_PAIR_COLUMNS)
    for column in ("paired_touch_index", "duration_frames"):
        if column in events:
            events[column] = events[column].astype("Int64")
    for column in (
        "lift_event_index",
        "lift_global_frame_index",
        "lift_timestamp_raw",
        "duration_frames",
    ):
        if column in touch_pairs:
            touch_pairs[column] = touch_pairs[column].astype("Int64")
    return BoardEventDetection(
        events=events,
        touch_pairs=touch_pairs,
        minimum_duration_frames=minimum_duration,
    )


def select_board_interval_from_presses(
    board_frames: pd.DataFrame,
    board_contacts: pd.DataFrame,
    detection: BoardEventDetection,
    *,
    target_valid_press_count: int = 20,
    seconds_before_first_valid_touch: float = 3.0,
    seconds_after_press: float = 3.0,
) -> BoardIntervalSelection:
    """Select a pre-roll before the first valid touch through the target press."""

    frames = _validated_board_frames(board_frames)
    if not isinstance(board_contacts, pd.DataFrame):
        raise EventAlignmentError("board_contacts must be a pandas DataFrame")
    if "frame_timestamp_raw" not in board_contacts:
        raise EventAlignmentError(
            "board_contacts is missing frame_timestamp_raw"
        )
    target_count = _positive_integer(
        target_valid_press_count,
        name="target valid press count",
    )
    before_seconds = _finite_float(
        seconds_before_first_valid_touch,
        name="seconds before first valid touch",
    )
    if before_seconds < 0.0:
        raise EventAlignmentError(
            "seconds before first valid touch must be nonnegative"
        )
    after_seconds = _finite_float(seconds_after_press, name="seconds after press")
    if after_seconds < 0.0:
        raise EventAlignmentError("seconds after press must be nonnegative")
    if not isinstance(detection, BoardEventDetection):
        raise EventAlignmentError("detection must be BoardEventDetection")

    frame_global_ids = _validated_global_frame_identities(
        frames["global_frame_index"],
        name="Board frame global identities",
        allow_missing=False,
        require_unique=True,
    )
    frame_identity_set = set(frame_global_ids.tolist())
    _validate_interval_detection(detection, frame_identity_set)

    try:
        frame_timestamps = frames["frame_timestamp_raw"].to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise EventAlignmentError("Board frame timestamps must be numeric") from error
    backward_positions = np.flatnonzero(np.diff(frame_timestamps) < 0.0) + 1
    usable_stop = int(backward_positions[0]) if len(backward_positions) else len(frames)
    usable_frames = frames.iloc[:usable_stop]
    first_available_timestamp = int(usable_frames["frame_timestamp_raw"].iloc[0])
    available_end = int(usable_frames["frame_timestamp_raw"].iloc[-1])
    maximum_global_frame = int(frame_global_ids[usable_stop - 1])
    prefix_frame_identity_set = set(frame_global_ids[:usable_stop].tolist())
    warnings_output: list[str] = []
    if len(backward_positions):
        warning = (
            "Board timestamp backward jump encountered after global frame "
            f"{maximum_global_frame}; dynamic interval selection uses the initial "
            "nondecreasing segment and reports the older tail without repair."
        )
        warnings_output.append(warning)

    globally_valid_pairs = detection.touch_pairs.loc[
        detection.touch_pairs["valid_touch"].astype(bool)
    ].copy()
    global_valid_pair_count = len(globally_valid_pairs)
    usable_pair_mask = _pair_endpoint_prefix_mask(
        globally_valid_pairs,
        prefix_frame_identity_set,
    )
    valid_pairs = globally_valid_pairs.loc[usable_pair_mask].copy()
    if len(backward_positions) and global_valid_pair_count > 0 and valid_pairs.empty:
        jump_position = int(backward_positions[0])
        previous_frame = frames.iloc[jump_position - 1]
        next_frame = frames.iloc[jump_position]
        raise InitialIntervalNoUsablePairError(
            previous_global_frame_index=int(frame_global_ids[jump_position - 1]),
            next_global_frame_index=int(frame_global_ids[jump_position]),
            previous_timestamp_raw=_python_scalar(
                previous_frame["frame_timestamp_raw"]
            ),
            next_timestamp_raw=_python_scalar(next_frame["frame_timestamp_raw"]),
            prefix_boundary_position=jump_position,
            last_pre_jump_global_frame_index=int(frame_global_ids[jump_position - 1]),
            total_global_valid_pair_count=global_valid_pair_count,
            usable_prefix_valid_pair_count=len(valid_pairs),
        )
    if valid_pairs.empty:
        raise EventAlignmentError(
            "no valid Board press/lift pair is available for interval selection"
        )
    first_valid_press_timestamp = int(valid_pairs.iloc[0]["press_timestamp_raw"])
    desired_start = int(
        round(first_valid_press_timestamp - before_seconds * 1_000_000.0)
    )
    board_start = max(desired_start, first_available_timestamp)
    start_was_clamped = board_start != desired_start
    selected_anchor_position = min(target_count, len(valid_pairs)) - 1
    boundary_pair = valid_pairs.iloc[selected_anchor_position]
    boundary_press_timestamp = int(boundary_pair["press_timestamp_raw"])
    desired_end = int(round(boundary_press_timestamp + after_seconds * 1_000_000.0))
    actual_end = min(desired_end, available_end)
    end_was_clamped = actual_end != desired_end

    # Select the positional initial interval before any timestamp filtering.
    # A stale post-jump tail can have timestamps that overlap this window.
    prefix_frames = frames.iloc[:usable_stop]
    frame_mask = prefix_frames["frame_timestamp_raw"].between(
        board_start,
        actual_end,
        inclusive="both",
    )
    selected_frames = prefix_frames.loc[frame_mask].copy()

    if "global_frame_index" in board_contacts:
        contact_global_ids = _validated_global_frame_identities(
            board_contacts["global_frame_index"],
            name="Board contact global identities",
            allow_missing=False,
            require_unique=False,
        )
        if not set(contact_global_ids.astype(np.int64).tolist()).issubset(
            frame_identity_set
        ):
            raise EventAlignmentError(
                "Board contacts contain global frame identities absent from "
                "Board frames"
            )
        contact_prefix_mask = np.isin(
            contact_global_ids,
            np.asarray(tuple(prefix_frame_identity_set), dtype=np.int64),
        )
    else:
        # Small test fixtures historically provide timestamp-only contacts.
        # Without frame identity there is no safe positional prefix to apply.
        contact_prefix_mask = np.ones(len(board_contacts), dtype=bool)
    contact_mask = (
        pd.Series(contact_prefix_mask, index=board_contacts.index)
        & board_contacts["frame_timestamp_raw"].between(
            board_start,
            actual_end,
            inclusive="both",
        )
    )
    prefix_events = detection.events.loc[
        _event_frame_prefix_mask(
            detection.events,
            prefix_frame_identity_set,
        )
    ]
    event_mask = prefix_events["frame_timestamp_raw"].between(
        board_start,
        actual_end,
        inclusive="both",
    )
    selected_contacts = board_contacts.loc[contact_mask].copy()
    selected_events = prefix_events.loc[event_mask].copy()
    selected_pair_indices = selected_events["paired_touch_index"].dropna().astype(int)
    selected_pairs = detection.touch_pairs.loc[
        detection.touch_pairs["paired_touch_index"].isin(selected_pair_indices)
        & _pair_endpoint_prefix_mask(
            detection.touch_pairs,
            prefix_frame_identity_set,
        )
    ].copy()
    if not selected_pairs.empty:
        selected_pairs["press_in_selected_interval"] = selected_pairs[
            "press_timestamp_raw"
        ].between(board_start, actual_end, inclusive="both")
        selected_pairs["lift_in_selected_interval"] = selected_pairs[
            "lift_timestamp_raw"
        ].between(board_start, actual_end, inclusive="both")

    selected_presses = selected_events.loc[
        selected_events["event_type"] == "press"
    ]
    last_selected_press = selected_presses.iloc[-1]
    last_selected_touch_index = int(last_selected_press["paired_touch_index"])
    last_selected_pair = detection.touch_pairs.loc[
        detection.touch_pairs["paired_touch_index"] == last_selected_touch_index
    ].iloc[0]
    last_selected_lift_timestamp = last_selected_pair["lift_timestamp_raw"]
    last_selected_lift_in_interval = bool(
        pd.notna(last_selected_lift_timestamp)
        and int(last_selected_lift_timestamp) <= actual_end
    )
    if not last_selected_lift_in_interval:
        warnings_output.append(
            "The last selected press has no paired lift "
            "inside the exact selected interval."
        )
    valid_press_selected = selected_events.loc[
        (selected_events["event_type"] == "press")
        & selected_events["valid_touch"].astype(bool)
    ]
    metadata: dict[str, object] = {
        "board_start_timestamp": board_start,
        "desired_board_start_timestamp": desired_start,
        "first_available_board_timestamp": first_available_timestamp,
        "first_valid_touch_press_timestamp": first_valid_press_timestamp,
        "seconds_before_first_valid_touch": before_seconds,
        "start_was_clamped": start_was_clamped,
        "desired_board_end_timestamp": desired_end,
        "actual_board_end_timestamp": actual_end,
        "valid_press_count_available": int(len(valid_pairs)),
        "valid_press_count_selected": int(len(valid_press_selected)),
        "selected_event_count": int(len(selected_events)),
        "selected_press_count": int(
            (selected_events["event_type"] == "press").sum()
        ),
        "selected_lift_count": int(
            (selected_events["event_type"] == "lift").sum()
        ),
        "end_was_clamped": end_was_clamped,
        "selected_interval_duration_s": (actual_end - board_start) / 1_000_000.0,
        "target_valid_press_count": target_count,
        "seconds_after_target_press": after_seconds,
        "boundary_press_event_index": int(boundary_pair["press_event_index"]),
        "boundary_press_lift_in_interval": bool(
            pd.notna(boundary_pair["lift_timestamp_raw"])
            and int(boundary_pair["lift_timestamp_raw"]) <= actual_end
        ),
        "last_selected_press_event_index": int(last_selected_press["event_index"]),
        "last_selected_press_lift_in_interval": last_selected_lift_in_interval,
        "backward_jump_count_in_full_board_order": int(len(backward_positions)),
    }
    return BoardIntervalSelection(
        frames=selected_frames,
        contacts=selected_contacts,
        events=selected_events,
        touch_pairs=selected_pairs,
        metadata=metadata,
        warnings=tuple(warnings_output),
    )


def compute_transient_score(
    ring_dataframe: pd.DataFrame,
    *,
    signal_columns: Sequence[str],
) -> np.ndarray:
    """Return a robust first-difference norm across the requested channels."""

    if not isinstance(ring_dataframe, pd.DataFrame) or ring_dataframe.empty:
        raise EventAlignmentError("ring_dataframe must contain samples")
    columns = tuple(signal_columns)
    missing = tuple(column for column in columns if column not in ring_dataframe)
    if not columns or missing:
        raise EventAlignmentError(
            "missing transient signal columns: " + ", ".join(missing)
        )
    try:
        signals = ring_dataframe.loc[:, list(columns)].to_numpy(
            dtype=np.float64,
            copy=True,
        )
    except (TypeError, ValueError) as error:
        raise EventAlignmentError("Ring transient signals must be numeric") from error
    return compute_transient_score_array(signals)


def compute_transient_score_array(values: np.ndarray) -> np.ndarray:
    """Return the same robust transient score for an arbitrary feature matrix."""

    try:
        signals = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise EventAlignmentError("transient feature values must be numeric") from error
    if signals.ndim != 2 or signals.shape[0] == 0 or signals.shape[1] == 0:
        raise EventAlignmentError("transient feature values must have nonempty shape (N, C)")
    if not np.isfinite(signals).all():
        raise EventAlignmentError("transient feature values must be finite")
    differences = np.diff(signals, axis=0, prepend=signals[[0]])
    median = np.median(differences, axis=0)
    mad = np.median(np.abs(differences - median), axis=0)
    normalized = (differences - median) / (1.4826 * mad + 1e-12)
    return np.sqrt(np.sum(normalized**2, axis=1))


def smooth_transient_score(
    transient_score: np.ndarray,
    *,
    window_samples: int,
) -> np.ndarray:
    """Return an edge-padded centered moving average of a transient score."""

    score = _validated_finite_vector(transient_score, name="transient score")
    window = _positive_integer(window_samples, name="smoothing window samples")
    if window % 2 == 0:
        raise EventAlignmentError("smoothing window samples must be odd")
    return _moving_average(score, window)


def detect_transient_peak_regions(
    transient_score: np.ndarray,
    ring_timestamps: np.ndarray,
    *,
    config: PeakDetectionConfig = PeakDetectionConfig(),
) -> pd.DataFrame:
    """Detect prominence-filtered, bounded, and optionally merged transients."""

    score = _validated_finite_vector(transient_score, name="transient score")
    timestamps = _validated_strict_timestamps(ring_timestamps)
    if score.size != timestamps.size:
        raise EventAlignmentError(
            "transient score and Ring timestamps must have equal lengths"
        )
    if score.size < 3:
        raise EventAlignmentError("at least three Ring samples are required")
    _validate_peak_config(config)
    smoothed = _moving_average(score, config.smoothing_window_samples)
    local_peaks = np.flatnonzero(
        (smoothed[1:-1] >= smoothed[:-2])
        & (smoothed[1:-1] > smoothed[2:])
    ) + 1
    baseline_mad = float(np.median(np.abs(smoothed - np.median(smoothed))))
    robust_scale = 1.4826 * baseline_mad
    prominence_threshold = max(
        float(config.minimum_prominence),
        float(config.prominence_mad_multiplier) * robust_scale,
    )
    raw_regions: list[dict[str, int | float]] = []
    window = int(config.prominence_window_samples)
    for peak in local_peaks:
        left_limit = max(0, int(peak) - window)
        right_limit = min(score.size - 1, int(peak) + window)
        left_min = float(np.min(smoothed[left_limit : int(peak) + 1]))
        right_min = float(np.min(smoothed[int(peak) : right_limit + 1]))
        prominence = float(smoothed[peak] - max(left_min, right_min))
        if prominence < prominence_threshold:
            continue
        boundary_level = float(
            smoothed[peak]
            - config.boundary_prominence_fraction * prominence
        )
        left = int(peak)
        while left > left_limit and smoothed[left - 1] >= boundary_level:
            left -= 1
        right = int(peak)
        while right < right_limit and smoothed[right + 1] >= boundary_level:
            right += 1
        raw_regions.append(
            {
                "peak_index": int(peak),
                "left_index": left,
                "right_index": right,
                "height": float(score[peak]),
                "prominence": prominence,
            }
        )
    if not raw_regions:
        raise EventAlignmentError(
            "no transient peak region passed the configured prominence threshold"
        )

    merged: list[dict[str, int | float]] = []
    for region in raw_regions:
        if (
            merged
            and int(region["left_index"])
            - int(merged[-1]["right_index"])
            <= config.merge_gap_samples
        ):
            previous = merged[-1]
            previous["right_index"] = max(
                int(previous["right_index"]),
                int(region["right_index"]),
            )
            if (
                float(region["prominence"]),
                float(region["height"]),
            ) > (
                float(previous["prominence"]),
                float(previous["height"]),
            ):
                previous["peak_index"] = int(region["peak_index"])
                previous["height"] = float(region["height"])
                previous["prominence"] = float(region["prominence"])
        else:
            merged.append(dict(region))

    rows: list[dict[str, int | float]] = []
    start_timestamp = float(timestamps[0])
    for region in merged:
        peak = int(region["peak_index"])
        left = int(region["left_index"])
        right = int(region["right_index"])
        rows.append(
            {
                "peak_index": peak,
                "peak_timestamp": float(timestamps[peak]),
                "peak_elapsed_s": (float(timestamps[peak]) - start_timestamp)
                / 1_000_000.0,
                "left_index": left,
                "right_index": right,
                "left_timestamp": float(timestamps[left]),
                "right_timestamp": float(timestamps[right]),
                "width_s": (float(timestamps[right]) - float(timestamps[left]))
                / 1_000_000.0,
                "height": float(region["height"]),
                "prominence": float(region["prominence"]),
            }
        )
    return pd.DataFrame(rows, columns=PEAK_REGION_COLUMNS)


def normalized_peak_region_distance(
    event_timestamp: float,
    *,
    peak_timestamp: float,
    left_timestamp: float,
    right_timestamp: float,
    epsilon_us: float = 1e-9,
) -> float:
    """Return center-relative distance; values above one are outside region."""

    event = _finite_float(event_timestamp, name="event timestamp")
    peak = _finite_float(peak_timestamp, name="peak timestamp")
    left = _finite_float(left_timestamp, name="peak left timestamp")
    right = _finite_float(right_timestamp, name="peak right timestamp")
    if not left <= peak <= right:
        raise EventAlignmentError("peak timestamps must satisfy left <= peak <= right")
    epsilon = _finite_float(epsilon_us, name="distance epsilon")
    if epsilon <= 0.0:
        raise EventAlignmentError("distance epsilon must be positive")
    if event <= peak:
        return (peak - event) / max(peak - left, epsilon)
    return (event - peak) / max(right - peak, epsilon)


def align_events_to_transient_peaks(
    board_events: pd.DataFrame,
    touch_pairs: pd.DataFrame,
    peak_regions: pd.DataFrame,
    *,
    ring_timestamp_range: tuple[float, float],
    config: AlignmentConfig = AlignmentConfig(),
) -> SequenceAlignmentResult:
    """Rank clustered offsets using monotonic one-to-one sequence matching."""

    events = _valid_alignment_events(board_events)
    pairs = _valid_touch_pairs(touch_pairs)
    peaks = _validated_peak_regions(peak_regions)
    _validate_alignment_config(config)
    ring_start = _finite_float(ring_timestamp_range[0], name="Ring start timestamp")
    ring_end = _finite_float(ring_timestamp_range[1], name="Ring end timestamp")
    if ring_start >= ring_end:
        raise EventAlignmentError("Ring timestamp range must be increasing")

    candidate_table = _generate_offset_candidates(
        events,
        peaks,
        ring_start=ring_start,
        ring_end=ring_end,
        config=config,
    )
    evaluations: list[dict[str, object]] = []
    matches_by_candidate: dict[int, pd.DataFrame] = {}
    touch_matches_by_candidate: dict[int, pd.DataFrame] = {}
    for candidate in candidate_table.itertuples(index=False):
        matches = match_shifted_events_to_peaks(
            events,
            peaks,
            offset_us=float(candidate.offset_us),
        )
        touch_matches = build_touch_pair_match_diagnostics(pairs, matches)
        summary = _candidate_summary(
            matches,
            touch_matches,
            events=events,
            pairs=pairs,
            offset_us=float(candidate.offset_us),
            vote_count=int(candidate.vote_count),
            candidate_id=int(candidate.candidate_id),
            ring_start=ring_start,
            ring_end=ring_end,
            total_peak_count=len(peaks),
        )
        evaluations.append(summary)
        matches_by_candidate[int(candidate.candidate_id)] = matches
        touch_matches_by_candidate[int(candidate.candidate_id)] = touch_matches
    ranked = pd.DataFrame(evaluations)
    ranked = ranked.sort_values(
        by=[
            "matched_event_count",
            "fully_matched_touch_pair_count",
            "median_normalized_peak_distance",
            "total_normalized_peak_distance",
            "mean_log1p_peak_prominence",
            "absolute_offset_us",
            "offset_us",
        ],
        ascending=[False, False, True, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    best_row = ranked.iloc[0]
    second_row = ranked.iloc[1] if len(ranked) > 1 else None
    best_id = int(best_row["candidate_id"])
    best_matches = matches_by_candidate[best_id]
    best_touch_matches = touch_matches_by_candidate[best_id]
    margin = _candidate_margin(best_row, second_row, events=len(events), pairs=len(pairs))
    report, warnings_output, success = _build_alignment_report(
        best_row,
        second_row,
        best_matches,
        best_touch_matches,
        total_events=len(events),
        total_pairs=len(pairs),
        margin=margin,
        ring_start=ring_start,
        ring_end=ring_end,
        config=config,
    )
    ranked["selected"] = ranked["rank"] == 1
    return SequenceAlignmentResult(
        success=success,
        best_offset_us=float(best_row["offset_us"]),
        second_best_offset_us=(
            None if second_row is None else float(second_row["offset_us"])
        ),
        candidates=ranked,
        event_matches=best_matches,
        touch_pair_matches=best_touch_matches,
        report=report,
        warnings=tuple(warnings_output),
    )


def match_shifted_events_to_peaks(
    board_events: pd.DataFrame,
    peak_regions: pd.DataFrame,
    *,
    offset_us: float,
) -> pd.DataFrame:
    """Match events monotonically with coverage-first, pair-aware objectives."""

    events = _valid_alignment_events(board_events)
    peaks = _validated_peak_regions(peak_regions)
    offset = _finite_float(offset_us, name="offset")
    event_count = len(events)
    peak_count = len(peaks)
    event_types = events["event_type"].to_numpy(dtype=str)
    pair_indices = events["paired_touch_index"].to_numpy(dtype=np.int64)
    shifted_times = (
        events["frame_timestamp_raw"].to_numpy(dtype=np.float64) + offset
    )
    peak_times = peaks["peak_timestamp"].to_numpy(dtype=np.float64)
    left_times = peaks["left_timestamp"].to_numpy(dtype=np.float64)
    right_times = peaks["right_timestamp"].to_numpy(dtype=np.float64)
    shifted_grid = shifted_times[:, np.newaxis]
    peak_grid = peak_times[np.newaxis, :]
    coverage = (
        (shifted_grid >= left_times[np.newaxis, :])
        & (shifted_grid <= right_times[np.newaxis, :])
    )
    left_scale = np.maximum(peak_times - left_times, 1e-9)
    right_scale = np.maximum(right_times - peak_times, 1e-9)
    distances = np.where(
        shifted_grid <= peak_grid,
        (peak_grid - shifted_grid) / left_scale[np.newaxis, :],
        (shifted_grid - peak_grid) / right_scale[np.newaxis, :],
    )
    log_prominences = np.log1p(
        peaks["prominence"].to_numpy(dtype=np.float64)
    )
    scores: list[list[list[_MatchScore | None]]] = [
        [[None, None] for _ in range(peak_count + 1)]
        for _ in range(event_count + 1)
    ]
    predecessor: list[list[list[tuple[int, int, int, int] | None]]] = [
        [[None, None] for _ in range(peak_count + 1)]
        for _ in range(event_count + 1)
    ]
    scores[0][0][0] = (0, 0, 0.0, 0.0)

    def update_state(
        next_event: int,
        next_peak: int,
        next_press_matched: int,
        candidate_score: _MatchScore,
        *,
        previous_event: int,
        previous_peak: int,
        previous_press_matched: int,
        action: int,
    ) -> None:
        current_score = scores[next_event][next_peak][next_press_matched]
        current_pointer = predecessor[next_event][next_peak][next_press_matched]
        current_action = -1 if current_pointer is None else current_pointer[3]
        if (
            current_score is None
            or candidate_score > current_score
            or (candidate_score == current_score and action > current_action)
        ):
            scores[next_event][next_peak][next_press_matched] = candidate_score
            predecessor[next_event][next_peak][next_press_matched] = (
                previous_event,
                previous_peak,
                previous_press_matched,
                action,
            )

    # Actions: 1 skips an event, 2 skips a peak, 3 matches both.  The second
    # state bit records whether the immediately open touch's press was matched.
    for event_position in range(event_count + 1):
        for peak_position in range(peak_count + 1):
            for press_matched in (0, 1):
                score = scores[event_position][peak_position][press_matched]
                if score is None:
                    continue
                if event_position < event_count:
                    update_state(
                        event_position + 1,
                        peak_position,
                        0,
                        score,
                        previous_event=event_position,
                        previous_peak=peak_position,
                        previous_press_matched=press_matched,
                        action=1,
                    )
                if peak_position < peak_count:
                    update_state(
                        event_position,
                        peak_position + 1,
                        press_matched,
                        score,
                        previous_event=event_position,
                        previous_peak=peak_position,
                        previous_press_matched=press_matched,
                        action=2,
                    )
                if event_position >= event_count or peak_position >= peak_count:
                    continue
                if not coverage[event_position, peak_position]:
                    continue
                distance = float(distances[event_position, peak_position])
                event_type = event_types[event_position]
                same_pair_as_previous_press = bool(
                    event_type == "lift"
                    and event_position > 0
                    and event_types[event_position - 1] == "press"
                    and pair_indices[event_position - 1]
                    == pair_indices[event_position]
                )
                completed_pair = int(
                    same_pair_as_previous_press and press_matched == 1
                )
                next_press_matched = int(event_type == "press")
                matched_score: _MatchScore = (
                    score[0] + 1,
                    score[1] + completed_pair,
                    score[2] - distance,
                    score[3] + float(log_prominences[peak_position]),
                )
                update_state(
                    event_position + 1,
                    peak_position + 1,
                    next_press_matched,
                    matched_score,
                    previous_event=event_position,
                    previous_peak=peak_position,
                    previous_press_matched=press_matched,
                    action=3,
                )

    matched_positions: dict[int, int] = {}
    event_position = event_count
    peak_position = peak_count
    final_scores = scores[event_position][peak_position]
    press_matched = max(
        (state for state in (0, 1) if final_scores[state] is not None),
        key=lambda state: final_scores[state],
    )
    while event_position > 0 or peak_position > 0:
        pointer = predecessor[event_position][peak_position][press_matched]
        if pointer is None:
            break
        previous_event, previous_peak, previous_press_matched, action = pointer
        if action == 3:
            matched_positions[previous_event] = previous_peak
        event_position = previous_event
        peak_position = previous_peak
        press_matched = previous_press_matched

    rows: list[dict[str, object]] = []
    for position, event in events.reset_index(drop=True).iterrows():
        shifted = float(event["frame_timestamp_raw"]) + offset
        row: dict[str, object] = {
            "event_index": int(event["event_index"]),
            "event_type": str(event["event_type"]),
            "paired_touch_index": int(event["paired_touch_index"]),
            "board_timestamp_raw": int(event["frame_timestamp_raw"]),
            "shifted_event_timestamp": shifted,
            "matched": position in matched_positions,
            "matched_peak_index": pd.NA,
            "matched_peak_timestamp": np.nan,
            "peak_left_timestamp": np.nan,
            "peak_right_timestamp": np.nan,
            "normalized_peak_distance": np.nan,
            "residual_us": np.nan,
            "peak_height": np.nan,
            "peak_prominence": np.nan,
        }
        if position in matched_positions:
            peak = peaks.iloc[matched_positions[position]]
            peak_timestamp = float(peak["peak_timestamp"])
            row.update(
                {
                    "matched_peak_index": int(peak["peak_index"]),
                    "matched_peak_timestamp": peak_timestamp,
                    "peak_left_timestamp": float(peak["left_timestamp"]),
                    "peak_right_timestamp": float(peak["right_timestamp"]),
                    "normalized_peak_distance": normalized_peak_region_distance(
                        shifted,
                        peak_timestamp=peak_timestamp,
                        left_timestamp=float(peak["left_timestamp"]),
                        right_timestamp=float(peak["right_timestamp"]),
                    ),
                    "residual_us": peak_timestamp - shifted,
                    "peak_height": float(peak["height"]),
                    "peak_prominence": float(peak["prominence"]),
                }
            )
        rows.append(row)
    matches = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    matches["matched_peak_index"] = matches["matched_peak_index"].astype("Int64")
    return matches


def build_touch_pair_match_diagnostics(
    touch_pairs: pd.DataFrame,
    event_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Join matched press/lift peaks and diagnose pair-level inconsistencies."""

    pairs = _valid_touch_pairs(touch_pairs)
    if not isinstance(event_matches, pd.DataFrame):
        raise EventAlignmentError("event_matches must be a DataFrame")
    match_lookup = event_matches.set_index("event_index")
    rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        press_event_index = int(pair.press_event_index)
        lift_event_index = int(pair.lift_event_index)
        press_in_alignment = press_event_index in match_lookup.index
        lift_in_alignment = lift_event_index in match_lookup.index
        press = (
            match_lookup.loc[press_event_index] if press_in_alignment else None
        )
        lift = match_lookup.loc[lift_event_index] if lift_in_alignment else None
        press_matched = bool(press["matched"]) if press is not None else False
        lift_matched = bool(lift["matched"]) if lift is not None else False
        both = press_matched and lift_matched
        peak_duration_ms = (
            (float(lift["matched_peak_timestamp"]) - float(press["matched_peak_timestamp"]))
            / 1_000.0
            if both
            else np.nan
        )
        board_duration_ms = (
            int(pair.lift_timestamp_raw) - int(pair.press_timestamp_raw)
        ) / 1_000.0
        rows.append(
            {
                "paired_touch_index": int(pair.paired_touch_index),
                "press_event_index": press_event_index,
                "lift_event_index": lift_event_index,
                "press_timestamp_raw": int(pair.press_timestamp_raw),
                "lift_timestamp_raw": int(pair.lift_timestamp_raw),
                "touch_duration_ms": board_duration_ms,
                "press_in_alignment": press_in_alignment,
                "lift_in_alignment": lift_in_alignment,
                "matched_press_peak_index": (
                    press["matched_peak_index"] if press is not None else pd.NA
                ),
                "matched_lift_peak_index": (
                    lift["matched_peak_index"] if lift is not None else pd.NA
                ),
                "matched_press_peak_timestamp": (
                    press["matched_peak_timestamp"] if press is not None else np.nan
                ),
                "matched_lift_peak_timestamp": (
                    lift["matched_peak_timestamp"] if lift is not None else np.nan
                ),
                "press_residual_us": (
                    press["residual_us"] if press is not None else np.nan
                ),
                "lift_residual_us": (
                    lift["residual_us"] if lift is not None else np.nan
                ),
                "press_matched": press_matched,
                "lift_matched": lift_matched,
                "both_events_matched": both,
                "matched_peak_duration_ms": peak_duration_ms,
                "duration_difference_ms": (
                    peak_duration_ms - board_duration_ms if both else np.nan
                ),
                "reversed_peak_order": bool(both and peak_duration_ms < 0.0),
            }
        )
    return pd.DataFrame(rows)


def plot_imu_with_board_events(
    ring_dataframe: pd.DataFrame,
    ring_timestamps: np.ndarray,
    transient_score: np.ndarray,
    board_events: pd.DataFrame,
    *,
    signal_columns: Sequence[str],
    offset_us: float = 0.0,
    duration_s: float = 30.0,
    title: str,
    output_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, EventVisibility]:
    """Plot seven Ring rows with Board events on the Ring-start time base."""

    timestamps, score, signals, columns, events, duration, offset = _plot_inputs(
        ring_dataframe,
        ring_timestamps,
        transient_score,
        board_events,
        signal_columns=signal_columns,
        duration_s=duration_s,
        offset_us=offset_us,
    )
    elapsed = (timestamps - timestamps[0]) / 1_000_000.0
    visible = elapsed <= duration
    figure, axes = plt.subplots(
        7,
        1,
        figsize=(15, 13),
        sharex=True,
        layout="constrained",
    )
    try:
        for axis_index, (axis, column) in enumerate(zip(axes[:6], columns, strict=True)):
            axis.plot(elapsed[visible], signals[visible, axis_index], linewidth=0.8)
            axis.set_ylabel(column)
            axis.grid(True, alpha=0.25)
        axes[6].plot(elapsed[visible], score[visible], color="black", linewidth=0.8)
        axes[6].set_ylabel("transient\nscore")
        axes[6].grid(True, alpha=0.25)
        event_elapsed, visibility = _event_elapsed_and_visibility(
            events,
            ring_start=float(timestamps[0]),
            offset_us=offset,
            duration_s=duration,
        )
        for axis in axes:
            _draw_event_lines(axis, events, event_elapsed)
            axis.set_xlim(0.0, duration)
        axes[-1].set_xlabel("Elapsed time from Ring reconstructed start (s)")
        figure.suptitle(title)
        _save_figure(figure, output_path=output_path, show=show)
    except Exception:
        plt.close(figure)
        raise
    return figure, visibility


def plot_transient_with_event_raster(
    ring_timestamps: np.ndarray,
    transient_score: np.ndarray,
    board_events: pd.DataFrame,
    *,
    offset_us: float = 0.0,
    duration_s: float = 30.0,
    title: str,
    output_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, EventVisibility]:
    """Plot Ring transient score above a discrete press/lift event raster."""

    timestamps = _validated_strict_timestamps(ring_timestamps)
    score = _validated_finite_vector(transient_score, name="transient score")
    if timestamps.size != score.size:
        raise EventAlignmentError("Ring timestamps and transient score lengths differ")
    events = _validated_plot_events(board_events)
    duration = _finite_float(duration_s, name="plot duration")
    offset = _finite_float(offset_us, name="plot offset")
    elapsed = (timestamps - timestamps[0]) / 1_000_000.0
    visible = elapsed <= duration
    event_elapsed, visibility = _event_elapsed_and_visibility(
        events,
        ring_start=float(timestamps[0]),
        offset_us=offset,
        duration_s=duration,
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(15, 7),
        sharex=True,
        gridspec_kw={"height_ratios": (4, 1)},
        layout="constrained",
    )
    try:
        axes[0].plot(elapsed[visible], score[visible], color="black", linewidth=0.8)
        axes[0].set_ylabel("Transient score")
        axes[0].grid(True, alpha=0.25)
        for event_type, y_value, color, linestyle in (
            ("press", 1.0, "tab:red", "-"),
            ("lift", 0.0, "tab:orange", "--"),
        ):
            positions = event_elapsed[events["event_type"].to_numpy() == event_type]
            on_screen = positions[(positions >= 0.0) & (positions <= duration)]
            axes[1].vlines(
                on_screen,
                y_value - 0.25,
                y_value + 0.25,
                color=color,
                linestyle=linestyle,
                linewidth=1.2,
                label=event_type,
            )
        axes[1].set_yticks((0.0, 1.0), labels=("lift", "press"))
        axes[1].set_ylim(-0.6, 1.6)
        axes[1].set_ylabel("Board event")
        axes[1].set_xlabel("Elapsed time from Ring reconstructed start (s)")
        axes[1].set_xlim(0.0, duration)
        axes[1].grid(True, axis="x", alpha=0.25)
        axes[1].legend(loc="upper right")
        figure.suptitle(title)
        _save_figure(figure, output_path=output_path, show=show)
    except Exception:
        plt.close(figure)
        raise
    return figure, visibility


def plot_alignment_residuals(
    event_matches: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot matched residuals and a diagnostic, unapplied least-squares trend."""

    matched = event_matches.loc[event_matches["matched"].astype(bool)].copy()
    if len(matched) < 2:
        raise EventAlignmentError("at least two matched events are required")
    elapsed = (
        matched["board_timestamp_raw"].to_numpy(dtype=np.float64)
        - float(matched["board_timestamp_raw"].iloc[0])
    ) / 1_000_000.0
    residual_ms = matched["residual_us"].to_numpy(dtype=np.float64) / 1_000.0
    slope, intercept = np.polyfit(elapsed, residual_ms, 1)
    figure, axis = plt.subplots(figsize=(11, 5), layout="constrained")
    try:
        for event_type, color, marker in (
            ("press", "tab:red", "o"),
            ("lift", "tab:orange", "s"),
        ):
            mask = matched["event_type"].to_numpy() == event_type
            axis.scatter(
                elapsed[mask],
                residual_ms[mask],
                color=color,
                marker=marker,
                label=event_type,
            )
        axis.plot(
            elapsed,
            slope * elapsed + intercept,
            color="tab:blue",
            linestyle="--",
            label="diagnostic linear trend (not applied)",
        )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_xlabel("Elapsed Board event time (s)")
        axis.set_ylabel("Peak-center residual (ms)")
        axis.set_title("Constant-offset residuals by Board event type")
        axis.grid(True, alpha=0.3)
        axis.legend()
        _save_figure(figure, output_path=output_path, show=show)
    except Exception:
        plt.close(figure)
        raise
    return figure


def _validate_interval_detection(
    detection: BoardEventDetection,
    frame_identity_set: set[int],
) -> None:
    """Validate identity-bearing detection tables used by interval selection."""

    events = detection.events
    pairs = detection.touch_pairs
    if not isinstance(events, pd.DataFrame):
        raise EventAlignmentError("Board events must be a DataFrame")
    if not isinstance(pairs, pd.DataFrame):
        raise EventAlignmentError("Board touch pairs must be a DataFrame")

    event_required = {
        "event_index",
        "event_type",
        "global_frame_index",
        "frame_timestamp_raw",
        "paired_touch_index",
        "valid_touch",
    }
    missing_events = event_required - set(events.columns)
    if missing_events:
        raise EventAlignmentError(
            "Board events are missing column(s): "
            + ", ".join(sorted(missing_events))
        )
    pair_required = {
        "paired_touch_index",
        "press_event_index",
        "lift_event_index",
        "press_global_frame_index",
        "lift_global_frame_index",
        "press_timestamp_raw",
        "lift_timestamp_raw",
        "valid_touch",
    }
    missing_pairs = pair_required - set(pairs.columns)
    if missing_pairs:
        raise EventAlignmentError(
            "Board touch pairs are missing column(s): "
            + ", ".join(sorted(missing_pairs))
        )

    event_ids = _validated_integer_column(
        events["event_index"],
        name="Board event identities",
        allow_missing=False,
        require_unique=True,
    )
    event_frame_ids = _validated_global_frame_identities(
        events["global_frame_index"],
        name="Board event global identities",
        allow_missing=False,
        require_unique=True,
    )
    if not set(event_frame_ids.tolist()).issubset(frame_identity_set):
        raise EventAlignmentError(
            "Board events contain global frame identities absent from Board frames"
        )
    if not events["event_type"].isin(("press", "lift")).all():
        raise EventAlignmentError("Board event types must be press or lift")
    _validated_finite_column(
        events["frame_timestamp_raw"],
        name="Board event timestamps",
        allow_missing=False,
    )
    event_pair_ids = _validated_integer_column(
        events["paired_touch_index"],
        name="Board event pair identities",
        allow_missing=True,
        require_unique=False,
    )

    pair_ids = _validated_integer_column(
        pairs["paired_touch_index"],
        name="Board touch-pair identities",
        allow_missing=False,
        require_unique=True,
    )
    press_event_ids = _validated_integer_column(
        pairs["press_event_index"],
        name="Board press event identities",
        allow_missing=True,
        require_unique=False,
    )
    lift_event_ids = _validated_integer_column(
        pairs["lift_event_index"],
        name="Board lift event identities",
        allow_missing=True,
        require_unique=False,
    )
    press_frame_ids = _validated_global_frame_identities(
        pairs["press_global_frame_index"],
        name="Board press global identities",
        allow_missing=True,
        require_unique=False,
    )
    lift_frame_ids = _validated_global_frame_identities(
        pairs["lift_global_frame_index"],
        name="Board lift global identities",
        allow_missing=True,
        require_unique=False,
    )
    nonmissing_press_ids = press_frame_ids[~np.isnan(press_frame_ids)]
    nonmissing_lift_ids = lift_frame_ids[~np.isnan(lift_frame_ids)]
    if not set(nonmissing_press_ids.astype(np.int64).tolist()).issubset(
        frame_identity_set
    ) or not set(nonmissing_lift_ids.astype(np.int64).tolist()).issubset(
        frame_identity_set
    ):
        raise EventAlignmentError(
            "Board touch pairs contain global frame identities absent from Board frames"
        )
    combined_pair_frame_ids = np.concatenate(
        (nonmissing_press_ids, nonmissing_lift_ids)
    )
    if len(combined_pair_frame_ids) != len(np.unique(combined_pair_frame_ids)):
        raise EventAlignmentError(
            "Board touch-pair frame identities must be unique"
        )
    _validated_finite_column(
        pairs["press_timestamp_raw"],
        name="Board press timestamps",
        allow_missing=False,
    )
    _validated_finite_column(
        pairs["lift_timestamp_raw"],
        name="Board lift timestamps",
        allow_missing=True,
    )
    try:
        valid_touch = pairs["valid_touch"].to_numpy(dtype=bool)
        events["valid_touch"].to_numpy(dtype=bool)
    except (TypeError, ValueError) as error:
        raise EventAlignmentError("Board valid-touch flags must be boolean") from error

    press_timestamps_missing = pairs["press_timestamp_raw"].isna().to_numpy(
        dtype=bool
    )
    lift_timestamps_missing = pairs["lift_timestamp_raw"].isna().to_numpy(
        dtype=bool
    )
    if np.any(press_timestamps_missing[valid_touch]) or np.any(
        lift_timestamps_missing[valid_touch]
    ):
        raise EventAlignmentError(
            "valid Board touch pairs must have press and lift timestamps"
        )
    event_valid_touch = events["valid_touch"].to_numpy(dtype=bool)
    if np.any(np.isnan(event_pair_ids[event_valid_touch])):
        raise EventAlignmentError(
            "valid Board events must reference touch-pair identities"
        )

    pair_index_set = set(pair_ids.astype(np.int64).tolist())
    nonmissing_event_pair_ids = event_pair_ids[~np.isnan(event_pair_ids)]
    if not set(nonmissing_event_pair_ids.astype(np.int64).tolist()).issubset(
        pair_index_set
    ):
        raise EventAlignmentError(
            "Board events reference touch-pair identities absent from pairs"
        )
    event_index_set = set(event_ids.astype(np.int64).tolist())
    valid_pair_mask = valid_touch
    if np.any(np.isnan(press_event_ids[valid_pair_mask])) or np.any(
        np.isnan(lift_event_ids[valid_pair_mask])
    ):
        raise EventAlignmentError(
            "valid Board touch pairs must reference press and lift events"
        )
    if not set(press_event_ids[valid_pair_mask].astype(np.int64).tolist()).issubset(
        event_index_set
    ) or not set(lift_event_ids[valid_pair_mask].astype(np.int64).tolist()).issubset(
        event_index_set
    ):
        raise EventAlignmentError(
            "valid Board touch pairs reference unknown event identities"
        )
    if np.any(np.isnan(press_frame_ids[valid_pair_mask])) or np.any(
        np.isnan(lift_frame_ids[valid_pair_mask])
    ):
        raise EventAlignmentError(
            "valid Board touch pairs must have press and lift frame identities"
        )


def _event_frame_prefix_mask(
    events: pd.DataFrame,
    prefix_frame_identity_set: set[int],
) -> pd.Series:
    event_ids = _validated_global_frame_identities(
        events["global_frame_index"],
        name="Board event global identities",
        allow_missing=False,
        require_unique=False,
    )
    return pd.Series(
        np.isin(
            event_ids,
            np.asarray(tuple(prefix_frame_identity_set), dtype=np.int64),
        ),
        index=events.index,
    )


def _pair_endpoint_prefix_mask(
    pairs: pd.DataFrame,
    prefix_frame_identity_set: set[int],
) -> pd.Series:
    press_ids = _validated_global_frame_identities(
        pairs["press_global_frame_index"],
        name="Board press global identities",
        allow_missing=True,
        require_unique=False,
    )
    lift_ids = _validated_global_frame_identities(
        pairs["lift_global_frame_index"],
        name="Board lift global identities",
        allow_missing=True,
        require_unique=False,
    )
    prefix_ids = np.asarray(tuple(prefix_frame_identity_set), dtype=np.int64)
    return pd.Series(
        (~np.isnan(press_ids))
        & (~np.isnan(lift_ids))
        & np.isin(press_ids, prefix_ids)
        & np.isin(lift_ids, prefix_ids),
        index=pairs.index,
    )


def _validated_global_frame_identities(
    values: pd.Series,
    *,
    name: str,
    allow_missing: bool,
    require_unique: bool,
) -> np.ndarray:
    return _validated_integer_column(
        values,
        name=name,
        allow_missing=allow_missing,
        require_unique=require_unique,
        nonnegative=True,
    )


def _validated_integer_column(
    values: pd.Series,
    *,
    name: str,
    allow_missing: bool,
    require_unique: bool,
    nonnegative: bool = False,
) -> np.ndarray:
    if not isinstance(values, pd.Series):
        raise EventAlignmentError(f"{name} must be a pandas Series")
    missing = values.isna().to_numpy(dtype=bool)
    try:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise EventAlignmentError(f"{name} must be numeric") from error
    invalid_numeric = np.isnan(numeric) & ~missing
    if invalid_numeric.any():
        raise EventAlignmentError(f"{name} must be numeric")
    if not np.isfinite(numeric[~missing]).all():
        raise EventAlignmentError(f"{name} must contain finite values")
    if not allow_missing and missing.any():
        raise EventAlignmentError(f"{name} must not contain missing values")
    present = ~missing
    if not np.all(np.equal(numeric[present], np.floor(numeric[present]))):
        raise EventAlignmentError(f"{name} must contain integer values")
    if nonnegative and np.any(numeric[present] < 0):
        raise EventAlignmentError(f"{name} must be nonnegative")
    normalized = np.array(numeric, copy=True)
    normalized[present] = np.floor(normalized[present])
    if require_unique and len(normalized[present]) != len(
        np.unique(normalized[present])
    ):
        raise EventAlignmentError(f"{name} must be unique")
    return normalized


def _validated_finite_column(
    values: pd.Series,
    *,
    name: str,
    allow_missing: bool,
) -> None:
    if not isinstance(values, pd.Series):
        raise EventAlignmentError(f"{name} must be a pandas Series")
    missing = values.isna().to_numpy(dtype=bool)
    try:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise EventAlignmentError(f"{name} must be numeric") from error
    if not allow_missing and missing.any():
        raise EventAlignmentError(f"{name} must not contain missing values")
    if np.isnan(numeric[~missing]).any() or not np.isfinite(numeric[~missing]).all():
        raise EventAlignmentError(f"{name} must contain finite values")


def _python_scalar(value: object) -> object:
    """Convert NumPy scalars in diagnostics to ordinary Python scalars."""

    return value.item() if isinstance(value, np.generic) else value


def _validated_board_frames(frames: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frames, pd.DataFrame) or frames.empty:
        raise EventAlignmentError("Board frames must be a nonempty DataFrame")
    required = {
        "global_frame_index",
        "frame_timestamp_raw",
        "chunk_index",
        "contact_count",
    }
    missing = required - set(frames.columns)
    if missing:
        raise EventAlignmentError(
            "Board frames are missing column(s): " + ", ".join(sorted(missing))
        )
    try:
        timestamps = frames["frame_timestamp_raw"].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise EventAlignmentError("Board frame timestamps must be numeric") from error
    if not np.isfinite(timestamps).all():
        raise EventAlignmentError("Board frame timestamps must be finite")
    _validated_global_frame_identities(
        frames["global_frame_index"],
        name="Board frame global identities",
        allow_missing=False,
        require_unique=True,
    )
    return frames


def _validated_strict_timestamps(values: np.ndarray) -> np.ndarray:
    """Validate a strictly increasing internal alignment/plotting time axis."""

    array = _validated_finite_vector(values, name="timestamps")
    if not np.all(np.diff(array) > 0.0):
        raise EventAlignmentError("timestamps must be strictly increasing")
    return array


def _validated_finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise EventAlignmentError(f"{name} must be numeric") from error
    if array.ndim != 1 or array.size == 0:
        raise EventAlignmentError(f"{name} must be a nonempty 1-D array")
    if not np.isfinite(array).all():
        raise EventAlignmentError(f"{name} must contain only finite values")
    return np.array(array, copy=True)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise EventAlignmentError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise EventAlignmentError(f"{name} must be a positive integer")
    return converted


def _finite_float(value: object, *, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise EventAlignmentError(f"{name} must be finite") from error
    if not np.isfinite(converted):
        raise EventAlignmentError(f"{name} must be finite")
    return converted


def _validate_peak_config(config: PeakDetectionConfig) -> None:
    if not isinstance(config, PeakDetectionConfig):
        raise EventAlignmentError("peak config must be PeakDetectionConfig")
    smoothing = _positive_integer(
        config.smoothing_window_samples,
        name="smoothing window samples",
    )
    if smoothing % 2 == 0:
        raise EventAlignmentError("smoothing window samples must be odd")
    _positive_integer(
        config.prominence_window_samples,
        name="prominence window samples",
    )
    if config.prominence_mad_multiplier < 0.0 or config.minimum_prominence < 0.0:
        raise EventAlignmentError("prominence thresholds must be nonnegative")
    if not 0.0 < config.boundary_prominence_fraction <= 1.0:
        raise EventAlignmentError(
            "boundary prominence fraction must satisfy 0 < fraction <= 1"
        )
    if config.merge_gap_samples < 0:
        raise EventAlignmentError("merge gap samples must be nonnegative")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return np.array(values, copy=True)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.full(window, 1.0 / window)
    return np.convolve(padded, kernel, mode="valid")


def _valid_alignment_events(events: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(events, pd.DataFrame):
        raise EventAlignmentError("Board events must be a DataFrame")
    required = {
        "event_index",
        "event_type",
        "frame_timestamp_raw",
        "paired_touch_index",
        "valid_touch",
    }
    missing = required - set(events.columns)
    if missing:
        raise EventAlignmentError(
            "Board events are missing column(s): " + ", ".join(sorted(missing))
        )
    valid = events.loc[events["valid_touch"].astype(bool)].copy()
    if valid.empty:
        raise EventAlignmentError("no valid Board press/lift events are available")
    if not valid["event_type"].isin(("press", "lift")).all():
        raise EventAlignmentError("Board event types must be press or lift")
    timestamps = valid["frame_timestamp_raw"].to_numpy(dtype=np.float64)
    if not np.all(np.diff(timestamps) >= 0.0):
        raise EventAlignmentError("valid Board events must preserve chronological order")
    return valid.reset_index(drop=True)


def _valid_touch_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(pairs, pd.DataFrame):
        raise EventAlignmentError("touch_pairs must be a DataFrame")
    required = {
        "paired_touch_index",
        "press_event_index",
        "lift_event_index",
        "press_timestamp_raw",
        "lift_timestamp_raw",
        "valid_touch",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise EventAlignmentError(
            "touch pairs are missing column(s): " + ", ".join(sorted(missing))
        )
    valid = pairs.loc[pairs["valid_touch"].astype(bool)].copy()
    if valid.empty:
        raise EventAlignmentError("no valid Board touch pairs are available")
    return valid.reset_index(drop=True)


def _validated_peak_regions(peaks: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(peaks, pd.DataFrame) or peaks.empty:
        raise EventAlignmentError("peak regions must be a nonempty DataFrame")
    missing = set(PEAK_REGION_COLUMNS) - set(peaks.columns)
    if missing:
        raise EventAlignmentError(
            "peak regions are missing column(s): " + ", ".join(sorted(missing))
        )
    ordered = peaks.reset_index(drop=True).copy()
    peak_times = ordered["peak_timestamp"].to_numpy(dtype=float)
    if not np.all(np.diff(peak_times) > 0.0):
        raise EventAlignmentError("peak regions must be strictly chronological")
    if not (
        (ordered["left_timestamp"] <= ordered["peak_timestamp"])
        & (ordered["peak_timestamp"] <= ordered["right_timestamp"])
    ).all():
        raise EventAlignmentError("each peak center must lie inside its region")
    return ordered


def _validate_alignment_config(config: AlignmentConfig) -> None:
    if not isinstance(config, AlignmentConfig):
        raise EventAlignmentError("alignment config must be AlignmentConfig")
    if config.offset_cluster_width_us <= 0.0:
        raise EventAlignmentError("offset cluster width must be positive")
    _positive_integer(
        config.maximum_offset_candidates,
        name="maximum offset candidates",
    )
    for name, value in (
        ("minimum event coverage", config.minimum_event_coverage_ratio),
        (
            "press/lift coverage difference warning",
            config.press_lift_coverage_difference_warning,
        ),
        ("outside Ring fraction warning", config.outside_ring_fraction_warning),
    ):
        if not 0.0 <= value <= 1.0:
            raise EventAlignmentError(f"{name} must be between zero and one")
    if config.minimum_valid_touch_pairs < 1:
        raise EventAlignmentError("minimum valid touch pairs must be positive")
    if config.touch_duration_difference_warning_ms < 0.0:
        raise EventAlignmentError(
            "touch duration difference warning must be nonnegative"
        )


def _generate_offset_candidates(
    events: pd.DataFrame,
    peaks: pd.DataFrame,
    *,
    ring_start: float,
    ring_end: float,
    config: AlignmentConfig,
) -> pd.DataFrame:
    board_times = events["frame_timestamp_raw"].to_numpy(dtype=np.float64)
    peak_times = peaks["peak_timestamp"].to_numpy(dtype=np.float64)
    if config.offset_search_range_us is None:
        search_min = ring_start - float(board_times[-1])
        search_max = ring_end - float(board_times[0])
    else:
        search_min = _finite_float(
            config.offset_search_range_us[0],
            name="offset search minimum",
        )
        search_max = _finite_float(
            config.offset_search_range_us[1],
            name="offset search maximum",
        )
    if search_min > search_max:
        raise EventAlignmentError("offset search minimum exceeds maximum")
    differences = (peak_times[:, np.newaxis] - board_times[np.newaxis, :]).ravel()
    differences = differences[
        (differences >= search_min) & (differences <= search_max)
    ]
    if differences.size == 0:
        raise AlignmentFailureError(
            "no peak-to-event offset falls inside the configured search range"
        )
    cluster_width = float(config.offset_cluster_width_us)
    cluster_ids = np.floor((differences - search_min) / cluster_width).astype(np.int64)
    rows: list[dict[str, int | float]] = []
    for cluster_id in np.unique(cluster_ids):
        cluster = differences[cluster_ids == cluster_id]
        rows.append(
            {
                "offset_us": float(np.median(cluster)),
                "vote_count": int(cluster.size),
                "cluster_spread_us": float(np.ptp(cluster)),
            }
        )
    rows.sort(key=lambda row: (-int(row["vote_count"]), abs(float(row["offset_us"]))))
    rows = rows[: config.maximum_offset_candidates]
    for candidate_id, row in enumerate(rows):
        row["candidate_id"] = candidate_id
    return pd.DataFrame(rows)[
        ["candidate_id", "offset_us", "vote_count", "cluster_spread_us"]
    ]


def _candidate_summary(
    matches: pd.DataFrame,
    touch_matches: pd.DataFrame,
    *,
    events: pd.DataFrame,
    pairs: pd.DataFrame,
    offset_us: float,
    vote_count: int,
    candidate_id: int,
    ring_start: float,
    ring_end: float,
    total_peak_count: int,
) -> dict[str, object]:
    matched = matches.loc[matches["matched"].astype(bool)]
    press = matches["event_type"] == "press"
    lift = matches["event_type"] == "lift"
    outside = ~matches["shifted_event_timestamp"].between(ring_start, ring_end)
    distances = matched["normalized_peak_distance"].to_numpy(dtype=float)
    prominences = matched["peak_prominence"].to_numpy(dtype=float)
    return {
        "candidate_id": candidate_id,
        "offset_us": offset_us,
        "absolute_offset_us": abs(offset_us),
        "vote_count": vote_count,
        "matched_event_count": int(len(matched)),
        "matched_press_count": int((matches.loc[press, "matched"]).sum()),
        "matched_lift_count": int((matches.loc[lift, "matched"]).sum()),
        "fully_matched_touch_pair_count": int(
            touch_matches["both_events_matched"].sum()
        ),
        "median_normalized_peak_distance": (
            float(np.median(distances)) if distances.size else np.inf
        ),
        "maximum_normalized_peak_distance": (
            float(np.max(distances)) if distances.size else np.inf
        ),
        "total_normalized_peak_distance": (
            float(np.sum(distances)) if distances.size else np.inf
        ),
        "mean_log1p_peak_prominence": (
            float(np.mean(np.log1p(prominences))) if prominences.size else 0.0
        ),
        "outside_ring_event_count": int(outside.sum()),
        "matched_peak_count": int(len(matched)),
        "total_peak_count": int(total_peak_count),
        "unmatched_peak_count": int(total_peak_count - len(matched)),
        "total_event_count": int(len(events)),
        "total_touch_pair_count": int(len(pairs)),
    }


def _candidate_margin(
    best: pd.Series,
    second: pd.Series | None,
    *,
    events: int,
    pairs: int,
) -> float:
    if second is None:
        return float("inf")

    def encoded(row: pd.Series) -> float:
        distance = min(float(row["median_normalized_peak_distance"]), 1.0)
        prominence = min(float(row["mean_log1p_peak_prominence"]), 20.0) / 20.0
        return (
            float(row["matched_event_count"])
            + float(row["fully_matched_touch_pair_count"]) / (pairs + 1.0)
            + (1.0 - distance) / ((pairs + 1.0) * (events + 1.0))
            + prominence / ((pairs + 1.0) * (events + 1.0) * 100.0)
        )

    return encoded(best) - encoded(second)


def _build_alignment_report(
    best: pd.Series,
    second: pd.Series | None,
    matches: pd.DataFrame,
    touch_matches: pd.DataFrame,
    *,
    total_events: int,
    total_pairs: int,
    margin: float,
    ring_start: float,
    ring_end: float,
    config: AlignmentConfig,
) -> tuple[dict[str, object], list[str], bool]:
    matched = matches.loc[matches["matched"].astype(bool)].copy()
    press_mask = matches["event_type"] == "press"
    lift_mask = matches["event_type"] == "lift"
    total_press = int(press_mask.sum())
    total_lift = int(lift_mask.sum())
    matched_press = int(matches.loc[press_mask, "matched"].sum())
    matched_lift = int(matches.loc[lift_mask, "matched"].sum())
    full_pairs = int(touch_matches["both_events_matched"].sum())
    press_only_pairs = int(
        (touch_matches["press_matched"] & ~touch_matches["lift_matched"]).sum()
    )
    lift_only_pairs = int(
        (~touch_matches["press_matched"] & touch_matches["lift_matched"]).sum()
    )
    reversed_pairs = int(touch_matches["reversed_peak_order"].sum())
    both_pair_rows = touch_matches.loc[touch_matches["both_events_matched"]]
    incompatible_duration_pairs = int(
        (
            both_pair_rows["duration_difference_ms"].abs()
            > config.touch_duration_difference_warning_ms
        ).sum()
    )
    event_coverage = len(matched) / total_events
    press_coverage = matched_press / total_press if total_press else None
    lift_coverage = matched_lift / total_lift if total_lift else None
    pair_objective_enabled = total_press > 0 and total_lift > 0
    pair_coverage = full_pairs / total_pairs if pair_objective_enabled else None
    residual_ms = np.abs(matched["residual_us"].to_numpy(dtype=float)) / 1_000.0
    distances = matched["normalized_peak_distance"].to_numpy(dtype=float)
    shifted = matches["shifted_event_timestamp"].to_numpy(dtype=float)
    outside_fraction = float(np.mean((shifted < ring_start) | (shifted > ring_end)))
    trend_change_ms = 0.0
    trend_slope_ms_per_s = 0.0
    if len(matched) >= 2:
        elapsed = (
            matched["board_timestamp_raw"].to_numpy(dtype=float)
            - float(matched["board_timestamp_raw"].iloc[0])
        ) / 1_000_000.0
        trend_slope_ms_per_s, _ = np.polyfit(
            elapsed,
            matched["residual_us"].to_numpy(dtype=float) / 1_000.0,
            1,
        )
        trend_change_ms = float(abs(trend_slope_ms_per_s * np.ptp(elapsed)))

    warnings_output: list[str] = []
    if event_coverage < config.minimum_event_coverage_ratio:
        warnings_output.append(
            f"Event coverage {event_coverage:.3f} is below configured minimum "
            f"{config.minimum_event_coverage_ratio:.3f}."
        )
    if total_pairs < config.minimum_valid_touch_pairs:
        warnings_output.append(
            f"Only {total_pairs} valid touch pairs are available; configured minimum "
            f"is {config.minimum_valid_touch_pairs}."
        )
    nearly_tied = False
    if second is not None:
        structurally_tied = (
            int(best["matched_event_count"]) == int(second["matched_event_count"])
            and int(best["fully_matched_touch_pair_count"])
            == int(second["fully_matched_touch_pair_count"])
        )
        distance_margin = (
            float(second["median_normalized_peak_distance"])
            - float(best["median_normalized_peak_distance"])
        )
        nearly_tied = structurally_tied and (
            distance_margin <= config.ambiguity_distance_margin
        )
        if nearly_tied:
            warnings_output.append(
                "Best and second-best offsets are nearly tied in coverage, pair "
                "completeness, and peak-center distance."
            )
    if (
        press_coverage is not None
        and lift_coverage is not None
        and abs(press_coverage - lift_coverage)
        > config.press_lift_coverage_difference_warning
    ):
        warnings_output.append(
            "Press and lift coverage differ by more than the configured threshold."
        )
    if trend_change_ms > config.residual_trend_warning_ms:
        warnings_output.append(
            f"Residual trend changes by about {trend_change_ms:.3f} ms; possible "
            "clock drift is reported but no affine correction is applied."
        )
    if outside_fraction > config.outside_ring_fraction_warning:
        warnings_output.append(
            f"{outside_fraction:.3f} of shifted events lie outside Ring coverage."
        )
    if reversed_pairs:
        warnings_output.append(
            f"{reversed_pairs} matched touch pair(s) have reversed Ring peak order."
        )
    if incompatible_duration_pairs:
        warnings_output.append(
            f"{incompatible_duration_pairs} fully matched touch pair(s) differ from "
            "Board touch duration by more than the configured threshold."
        )
    success = (
        event_coverage >= config.minimum_event_coverage_ratio
        and total_pairs >= config.minimum_valid_touch_pairs
    )
    matching_objective = ["maximize matched event count"]
    if pair_objective_enabled:
        matching_objective.append(
            "maximize fully matched press/lift touch-pair count"
        )
    matching_objective.extend(
        [
            "minimize normalized distance to peak centers",
            "weakly maximize log1p peak prominence",
            "minimize absolute offset as final candidate tie-breaker",
        ]
    )
    report: dict[str, object] = {
        "alignment_model": "constant_offset",
        "matched_event_types": [
            event_type
            for event_type in ("press", "lift")
            if bool((matches["event_type"] == event_type).any())
        ],
        "matching_objective": matching_objective,
        "pair_completeness_objective_enabled": pair_objective_enabled,
        "pair_aware_dynamic_programming": pair_objective_enabled,
        "structural_success": True,
        "alignment_success": success,
        "best_offset_us": float(best["offset_us"]),
        "work_axis_offset_us": float(best["offset_us"]),
        "second_best_offset_us": (
            None if second is None else float(second["offset_us"])
        ),
        "matched_event_count": int(len(matched)),
        "total_valid_event_count": total_events,
        "matched_press_count": matched_press,
        "total_valid_press_count": total_press,
        "matched_lift_count": matched_lift,
        "total_valid_lift_count": total_lift,
        "fully_matched_touch_pair_count": full_pairs,
        "total_valid_touch_pair_count": total_pairs,
        "event_coverage_ratio": event_coverage,
        "press_coverage_ratio": press_coverage,
        "lift_coverage_ratio": lift_coverage,
        "touch_pair_coverage_ratio": pair_coverage,
        "press_only_touch_pair_count": press_only_pairs,
        "lift_only_touch_pair_count": lift_only_pairs,
        "reversed_peak_order_touch_pair_count": reversed_pairs,
        "duration_incompatible_touch_pair_count": incompatible_duration_pairs,
        "matched_peak_count": int(best["matched_peak_count"]),
        "total_peak_count": int(best["total_peak_count"]),
        "unmatched_peak_count": int(best["unmatched_peak_count"]),
        "median_normalized_peak_distance": float(np.median(distances)),
        "maximum_normalized_peak_distance": float(np.max(distances)),
        "median_absolute_time_residual_ms": float(np.median(residual_ms)),
        "maximum_absolute_time_residual_ms": float(np.max(residual_ms)),
        "best_vs_second_best_margin": margin,
        "best_vs_second_best_nearly_tied": nearly_tied,
        "residual_trend_slope_ms_per_s": float(trend_slope_ms_per_s),
        "residual_trend_change_ms": trend_change_ms,
        "outside_ring_event_fraction": outside_fraction,
        "ring_peaks_are_unlabeled_motion_transients": True,
        "board_event_labels_source": "Board occupancy transitions",
        "affine_clock_correction_applied": False,
        "warnings": warnings_output,
    }
    return report, warnings_output, success


def _plot_inputs(
    ring_dataframe: pd.DataFrame,
    ring_timestamps: np.ndarray,
    transient_score: np.ndarray,
    board_events: pd.DataFrame,
    *,
    signal_columns: Sequence[str],
    duration_s: float,
    offset_us: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], pd.DataFrame, float, float]:
    timestamps = _validated_strict_timestamps(ring_timestamps)
    score = _validated_finite_vector(transient_score, name="transient score")
    columns = tuple(signal_columns)
    if not isinstance(ring_dataframe, pd.DataFrame) or ring_dataframe.empty:
        raise EventAlignmentError("ring_dataframe must contain samples")
    missing = tuple(column for column in columns if column not in ring_dataframe)
    if len(columns) != 6 or missing:
        raise EventAlignmentError("six configured Ring signal columns are required")
    signals = ring_dataframe.loc[:, list(columns)].to_numpy(dtype=float, copy=True)
    if len(signals) != len(timestamps) or len(score) != len(timestamps):
        raise EventAlignmentError("Ring plotting inputs must have equal lengths")
    events = _validated_plot_events(board_events)
    duration = _finite_float(duration_s, name="plot duration")
    if duration <= 0.0:
        raise EventAlignmentError("plot duration must be positive")
    offset = _finite_float(offset_us, name="plot offset")
    return timestamps, score, signals, columns, events, duration, offset


def _validated_plot_events(events: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(events, pd.DataFrame):
        raise EventAlignmentError("Board plotting events must be a DataFrame")
    required = {"event_type", "frame_timestamp_raw"}
    missing = required - set(events.columns)
    if missing:
        raise EventAlignmentError("Board plotting events are missing required columns")
    if not events["event_type"].isin(("press", "lift")).all():
        raise EventAlignmentError("Board plotting event types must be press or lift")
    return events.reset_index(drop=True).copy()


def _event_elapsed_and_visibility(
    events: pd.DataFrame,
    *,
    ring_start: float,
    offset_us: float,
    duration_s: float,
) -> tuple[np.ndarray, EventVisibility]:
    elapsed = (
        events["frame_timestamp_raw"].to_numpy(dtype=np.float64)
        + offset_us
        - ring_start
    ) / 1_000_000.0
    on_screen = (elapsed >= 0.0) & (elapsed <= duration_s)
    press = events["event_type"].to_numpy() == "press"
    lift = ~press
    return elapsed, EventVisibility(
        press_inside=int(np.sum(press & on_screen)),
        press_outside=int(np.sum(press & ~on_screen)),
        lift_inside=int(np.sum(lift & on_screen)),
        lift_outside=int(np.sum(lift & ~on_screen)),
    )


def _draw_event_lines(
    axis: plt.Axes,
    events: pd.DataFrame,
    event_elapsed: np.ndarray,
) -> None:
    for event_type, color, linestyle in (
        ("press", "tab:red", "-"),
        ("lift", "tab:orange", "--"),
    ):
        mask = events["event_type"].to_numpy() == event_type
        for line_index, position in enumerate(event_elapsed[mask]):
            axis.axvline(
                position,
                color=color,
                linestyle=linestyle,
                alpha=0.55,
                linewidth=0.8,
                label=event_type if line_index == 0 and axis is axis.figure.axes[0] else None,
            )
    if axis is axis.figure.axes[0]:
        axis.legend(loc="upper right")


def _save_figure(
    figure: Figure,
    *,
    output_path: str | Path | None,
    show: bool,
) -> None:
    if output_path is not None:
        path = Path(output_path)
        if path.exists() and not path.is_file():
            raise EventAlignmentError(f"plot output path is not a file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160)
    if show:
        plt.show()
