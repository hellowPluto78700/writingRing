from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from writingring.writing_motion_mask import (
    WritingInterval,
    WritingMotionMaskError,
    apply_mask,
    build_recording_writing_mask,
    build_segment_masks,
    pad_writing_mask,
)


def _events() -> pd.DataFrame:
    rows = []
    for pair_id, (press, lift, segment) in enumerate(((200.0, 400.0, 0), (700.0, 800.0, 0))):
        for kind, timestamp in (("press", press), ("lift", lift)):
            rows.append(
                {
                    "dataset_id": 0,
                    "event_type": kind,
                    "paired_touch_index": pair_id,
                    "aligned_event_timestamp_us": timestamp,
                    "alignment_offset_domain": "canonical_timestamp",
                    "valid_touch": True,
                    "transient": False,
                    "incomplete_touch": False,
                    "assigned_label_index": 0,
                    "assigned_segment_index": segment,
                    "crossing_resolution": "not_crossing",
                }
            )
    return pd.DataFrame(rows)


def test_multiple_strokes_preserve_air_gap() -> None:
    timestamps = np.arange(0.0, 1000.0, 100.0)
    mask, intervals, _ = build_recording_writing_mask(
        _events(), dataset_id=0, canonical_timestamps_us=timestamps, sampling_rate_hz=10_000.0
    )
    assert len(intervals) == 2
    np.testing.assert_array_equal(
        mask,
        [False, False, True, True, True, False, False, True, True, False],
    )


def test_short_touch_can_collapse_to_one_sample() -> None:
    events = _events().iloc[:2].copy()
    events.loc[events["event_type"] == "press", "aligned_event_timestamp_us"] = 201.0
    events.loc[events["event_type"] == "lift", "aligned_event_timestamp_us"] = 250.0
    timestamps = np.array([0.0, 200.0, 300.0, 400.0])
    mask, intervals, _ = build_recording_writing_mask(
        events, dataset_id=0, canonical_timestamps_us=timestamps, sampling_rate_hz=10_000.0
    )
    assert intervals[0].collapsed_to_single_sample
    assert np.count_nonzero(mask) == 1


def test_segment_mask_only_uses_owned_intervals() -> None:
    intervals = {
        0: (
            WritingInterval(0, 0, 100, 200, 1, 2, 0, 0, "not_crossing", False),
            WritingInterval(0, 1, 300, 400, 3, 4, None, None, "rejected", False),
        )
    }
    mask, rows = build_segment_masks(
        segment_rows=[{
            "segment_index": "0", "dataset_id": "0", "start_sample_index": "0",
            "stop_sample_index_exclusive": "6", "label": "a", "source_label_index": "0",
        }],
        intervals_by_dataset=intervals,
        segment_lengths=np.array([6], dtype=np.int32),
    )
    np.testing.assert_array_equal(mask, [False, True, True, False, False, False])
    assert len(rows) == 1


def test_owned_interval_outside_segment_hard_fails() -> None:
    intervals = {0: (WritingInterval(0, 0, 100, 200, 0, 5, 0, 0, "not_crossing", False),)}
    with pytest.raises(WritingMotionMaskError, match="outside the published segment"):
        build_segment_masks(
            segment_rows=[{
                "segment_index": "0", "dataset_id": "0", "start_sample_index": "1",
                "stop_sample_index_exclusive": "5", "label": "a",
            }],
            intervals_by_dataset=intervals,
            segment_lengths=np.array([4], dtype=np.int32),
        )


def test_mask_and_padding_zero_airborne_samples() -> None:
    values = np.arange(18, dtype=np.float32).reshape(6, 3)
    mask = np.array([False, True, True, False, True, False])
    masked = apply_mask(values, mask)
    assert np.all(masked[~mask] == 0)
    np.testing.assert_array_equal(masked[mask], values[mask])
    padded, retained = pad_writing_mask(mask, np.array([3, 3]), target_length=4)
    np.testing.assert_array_equal(retained, [0, 1])
    assert padded.shape == (2, 4)
    assert not np.any(padded[:, 3])
