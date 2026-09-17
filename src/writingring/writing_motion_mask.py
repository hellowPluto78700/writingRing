"""Writing/contact masks derived from completed aligned-Board segmentation.

This module deliberately does not re-run Board segmentation policy. It consumes
already-audited `board_events.csv` rows, reconstructs the exact alignment
boundary time axis used by segmentation, and converts reliable valid touch pairs
into recording- and segment-level writing masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from writingring.alignment_io import build_offset_domain_timestamps


class WritingMotionMaskError(ValueError):
    """Raised when source segmentation artifacts cannot define a safe mask."""


@dataclass(frozen=True, slots=True)
class WritingInterval:
    dataset_id: int
    paired_touch_index: int
    press_timestamp_us: float
    lift_timestamp_us: float
    press_recording_sample_index: int
    lift_recording_sample_index: int
    assigned_segment_index: int | None
    assigned_label_index: int | None
    crossing_resolution: str
    collapsed_to_single_sample: bool

    @property
    def duration_samples(self) -> int:
        return self.lift_recording_sample_index - self.press_recording_sample_index + 1


_REQUIRED_EVENT_COLUMNS = {
    "dataset_id",
    "event_type",
    "paired_touch_index",
    "aligned_event_timestamp_us",
    "alignment_offset_domain",
    "valid_touch",
    "transient",
    "incomplete_touch",
    "assigned_label_index",
    "assigned_segment_index",
    "crossing_resolution",
}


def _as_bool(value: object, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"true", "1"}:
        return True
    if token in {"false", "0", ""}:
        return False
    raise WritingMotionMaskError(f"{field} must be boolean-like; got {value!r}")


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or numeric != int(numeric):
        raise WritingMotionMaskError(f"expected integer-like value, got {value!r}")
    return int(numeric)


def _required_int(value: object, *, field: str) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise WritingMotionMaskError(f"{field} is required")
    return parsed


def _finite_float(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WritingMotionMaskError(f"{field} must be numeric") from exc
    if not np.isfinite(number):
        raise WritingMotionMaskError(f"{field} must be finite")
    return number


def build_boundary_timestamps(
    canonical_timestamps_us: np.ndarray,
    *,
    offset_domain: str,
    sampling_rate_hz: float,
) -> np.ndarray:
    """Recreate the source segmentation's Board-boundary lookup axis."""

    try:
        return build_offset_domain_timestamps(
            canonical_timestamps_us,
            offset_domain=offset_domain,
            input_kind="spike-imu",
            feature_sampling_rate_hz=sampling_rate_hz,
        )
    except Exception as exc:
        raise WritingMotionMaskError(str(exc)) from exc


def _sample_index(boundary_timestamps_us: np.ndarray, timestamp_us: float) -> int:
    index = int(np.searchsorted(boundary_timestamps_us, timestamp_us, side="left"))
    if index < 0 or index >= len(boundary_timestamps_us):
        raise WritingMotionMaskError(
            f"Board event timestamp {timestamp_us} maps outside recording sample range"
        )
    return index


def build_recording_writing_mask(
    events: pd.DataFrame,
    *,
    dataset_id: int,
    canonical_timestamps_us: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, tuple[WritingInterval, ...], np.ndarray]:
    """Return physical-contact mask, intervals, and the exact boundary time axis.

    The recording mask keeps every complete, valid, non-transient touch pair,
    including a pair that was not assigned to a published classification
    segment. This preserves real contact motion in the continuous encoder state.
    Segment-level masks later restrict final exported samples to pairs explicitly
    owned by each published segment.
    """

    if not isinstance(events, pd.DataFrame):
        raise WritingMotionMaskError("events must be a pandas DataFrame")
    missing = sorted(_REQUIRED_EVENT_COLUMNS - set(events.columns))
    if missing:
        raise WritingMotionMaskError("board events missing columns: " + ", ".join(missing))
    rows = events.loc[pd.to_numeric(events["dataset_id"], errors="coerce") == dataset_id].copy()
    if rows.empty:
        raise WritingMotionMaskError(f"dataset {dataset_id} has no board-event audit rows")
    domains = {str(value) for value in rows["alignment_offset_domain"].dropna().unique()}
    if len(domains) != 1:
        raise WritingMotionMaskError(
            f"dataset {dataset_id} must have exactly one alignment offset domain; got {sorted(domains)}"
        )
    canonical = np.asarray(canonical_timestamps_us, dtype=np.float64)
    if canonical.ndim != 1 or len(canonical) < 2 or not np.isfinite(canonical).all():
        raise WritingMotionMaskError("canonical timestamps must be a finite vector")
    boundary = build_boundary_timestamps(
        canonical,
        offset_domain=next(iter(domains)),
        sampling_rate_hz=sampling_rate_hz,
    )
    mask = np.zeros(len(canonical), dtype=np.bool_)
    intervals: list[WritingInterval] = []

    pair_ids = sorted(
        {
            _required_int(value, field="paired_touch_index")
            for value in rows["paired_touch_index"]
            if not pd.isna(value)
        }
    )
    for pair_id in pair_ids:
        pair = rows[pd.to_numeric(rows["paired_touch_index"], errors="coerce") == pair_id]
        if pair.empty:
            continue
        reliable_rows = [
            row for row in pair.to_dict(orient="records")
            if _as_bool(row["valid_touch"], field="valid_touch")
            and not _as_bool(row["transient"], field="transient")
            and not _as_bool(row["incomplete_touch"], field="incomplete_touch")
        ]
        if not reliable_rows:
            continue
        press_rows = [row for row in reliable_rows if str(row["event_type"]) == "press"]
        lift_rows = [row for row in reliable_rows if str(row["event_type"]) == "lift"]
        if len(press_rows) != 1 or len(lift_rows) != 1:
            raise WritingMotionMaskError(
                f"dataset {dataset_id} pair {pair_id} must have one reliable press and one reliable lift"
            )
        press_row, lift_row = press_rows[0], lift_rows[0]
        press_us = _finite_float(press_row["aligned_event_timestamp_us"], field="press timestamp")
        lift_us = _finite_float(lift_row["aligned_event_timestamp_us"], field="lift timestamp")
        if lift_us <= press_us:
            raise WritingMotionMaskError(
                f"dataset {dataset_id} pair {pair_id} has non-positive touch duration"
            )
        press_index = _sample_index(boundary, press_us)
        lift_index = _sample_index(boundary, lift_us)
        if lift_index < press_index:
            raise WritingMotionMaskError(
                f"dataset {dataset_id} pair {pair_id} maps lift before press"
            )
        mask[press_index : lift_index + 1] = True

        press_segment = _optional_int(press_row["assigned_segment_index"])
        lift_segment = _optional_int(lift_row["assigned_segment_index"])
        if press_segment != lift_segment:
            raise WritingMotionMaskError(
                f"dataset {dataset_id} pair {pair_id} endpoints disagree on assigned segment: "
                f"{press_segment} vs {lift_segment}"
            )
        press_label = _optional_int(press_row["assigned_label_index"])
        lift_label = _optional_int(lift_row["assigned_label_index"])
        if press_label != lift_label:
            raise WritingMotionMaskError(
                f"dataset {dataset_id} pair {pair_id} endpoints disagree on assigned label"
            )
        resolutions = {
            str(value) for value in pair["crossing_resolution"].dropna().unique()
        }
        if len(resolutions) > 1:
            raise WritingMotionMaskError(
                f"dataset {dataset_id} pair {pair_id} has inconsistent crossing resolution"
            )
        intervals.append(
            WritingInterval(
                dataset_id=dataset_id,
                paired_touch_index=pair_id,
                press_timestamp_us=press_us,
                lift_timestamp_us=lift_us,
                press_recording_sample_index=press_index,
                lift_recording_sample_index=lift_index,
                assigned_segment_index=press_segment,
                assigned_label_index=press_label,
                crossing_resolution=(next(iter(resolutions)) if resolutions else "not_crossing"),
                collapsed_to_single_sample=press_index == lift_index,
            )
        )
    if not intervals:
        raise WritingMotionMaskError(f"dataset {dataset_id} has no reliable valid touch pairs")
    return mask, tuple(intervals), boundary


def build_segment_masks(
    *,
    segment_rows: Sequence[Mapping[str, Any]],
    intervals_by_dataset: Mapping[int, Sequence[WritingInterval]],
    segment_lengths: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build concatenated writing mask using only touch pairs owned by each segment."""

    lengths = np.asarray(segment_lengths)
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer):
        raise WritingMotionMaskError("segment_lengths must be a 1-D integer array")
    total = int(np.sum(lengths, dtype=np.int64))
    concatenated = np.zeros(total, dtype=np.bool_)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    interval_rows: list[dict[str, Any]] = []

    exported = [
        row for row in segment_rows
        if str(row.get("segment_index", "")).strip() != ""
        and str(row.get("exported", "true")).strip().lower() not in {"false", "0"}
    ]
    exported.sort(key=lambda row: _required_int(row["segment_index"], field="segment_index"))
    if len(exported) != len(lengths):
        raise WritingMotionMaskError(
            f"exported segment row count {len(exported)} != segment_lengths count {len(lengths)}"
        )
    for expected, row in enumerate(exported):
        segment_index = _required_int(row["segment_index"], field="segment_index")
        if segment_index != expected:
            raise WritingMotionMaskError("segment indices must be contiguous from zero")
        dataset_id = _required_int(row["dataset_id"], field="dataset_id")
        start = _required_int(row["start_sample_index"], field="start_sample_index")
        stop = _required_int(row["stop_sample_index_exclusive"], field="stop_sample_index_exclusive")
        if stop - start != int(lengths[segment_index]):
            raise WritingMotionMaskError(
                f"segment {segment_index} source slice length disagrees with segment_lengths"
            )
        local_mask = np.zeros(stop - start, dtype=np.bool_)
        stroke_index = 0
        for interval in intervals_by_dataset.get(dataset_id, ()):
            if interval.assigned_segment_index != segment_index:
                continue
            if not (
                start <= interval.press_recording_sample_index
                <= interval.lift_recording_sample_index < stop
            ):
                raise WritingMotionMaskError(
                    f"segment {segment_index} owns touch pair {interval.paired_touch_index} "
                    "whose mapped samples fall outside the published segment"
                )
            local_press = interval.press_recording_sample_index - start
            local_lift = interval.lift_recording_sample_index - start
            local_mask[local_press : local_lift + 1] = True
            interval_rows.append(
                {
                    "dataset_id": dataset_id,
                    "segment_index": segment_index,
                    "label": str(row.get("label", "")),
                    "source_label_index": _optional_int(row.get("source_label_index")),
                    "paired_touch_index": interval.paired_touch_index,
                    "stroke_index": stroke_index,
                    "press_timestamp_us": interval.press_timestamp_us,
                    "lift_timestamp_us": interval.lift_timestamp_us,
                    "press_recording_sample_index": interval.press_recording_sample_index,
                    "lift_recording_sample_index": interval.lift_recording_sample_index,
                    "press_segment_local_index": local_press,
                    "lift_segment_local_index": local_lift,
                    "duration_samples": interval.duration_samples,
                    "collapsed_to_single_sample": interval.collapsed_to_single_sample,
                    "crossing_resolution": interval.crossing_resolution,
                }
            )
            stroke_index += 1
        if not np.any(local_mask):
            raise WritingMotionMaskError(
                f"published segment {segment_index} has zero writing samples"
            )
        concatenated[int(offsets[segment_index]) : int(offsets[segment_index + 1])] = local_mask
    return concatenated, interval_rows


def apply_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Hard-zero every feature channel outside `mask`."""

    array = np.asarray(values)
    state = np.asarray(mask, dtype=np.bool_)
    if array.ndim != 2 or state.ndim != 1 or len(array) != len(state):
        raise WritingMotionMaskError("feature values and mask have incompatible shapes")
    output = array.copy()
    output[~state, :] = 0
    return output


def pad_writing_mask(
    writing_mask: np.ndarray,
    segment_lengths: np.ndarray,
    *,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad mask using the same overflow rule as `segment_padding`."""

    state = np.asarray(writing_mask, dtype=np.bool_)
    lengths = np.asarray(segment_lengths)
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer):
        raise WritingMotionMaskError("segment_lengths must be a 1-D integer array")
    if int(np.sum(lengths, dtype=np.int64)) != len(state):
        raise WritingMotionMaskError("writing mask length does not match segment lengths")
    if not isinstance(target_length, int) or target_length <= 0:
        raise WritingMotionMaskError("target_length must be positive")
    retained = np.flatnonzero(lengths <= target_length)
    output = np.zeros((len(retained), target_length), dtype=np.bool_)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    for out_index, source_index in enumerate(retained):
        start, stop = int(offsets[source_index]), int(offsets[source_index + 1])
        output[out_index, : int(lengths[source_index])] = state[start:stop]
    return output, retained.astype(np.int64, copy=False)


def write_intervals_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)
