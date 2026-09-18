from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from writingring.segmentation_verification import SegmentationVerificationConfig
from writingring.writing_motion_verification import (
    _index_window_seconds,
    _legend_handles,
    create_writing_motion_verification_figure,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": 0,
                "event_type": "press",
                "valid_touch": True,
                "transient": False,
                "incomplete_touch": False,
                "aligned_event_timestamp_us": 1_000_000.0,
            },
            {
                "dataset_id": 0,
                "event_type": "lift",
                "valid_touch": True,
                "transient": False,
                "incomplete_touch": False,
                "aligned_event_timestamp_us": 1_500_000.0,
            },
            {
                "dataset_id": 0,
                "event_type": "press",
                "valid_touch": False,
                "transient": True,
                "incomplete_touch": False,
                "aligned_event_timestamp_us": 11_000_000.0,
            },
            {
                "dataset_id": 0,
                "event_type": "lift",
                "valid_touch": False,
                "transient": True,
                "incomplete_touch": False,
                "aligned_event_timestamp_us": 11_100_000.0,
            },
        ]
    )


def _segments() -> list[dict[str, object]]:
    return [
        {
            "dataset_id": 0,
            "source_label_index": 0,
            "segment_index": 0,
            "exported": True,
            "skip_reason": "",
            "label": "a",
            "label_timestamp_us": 500_000.0,
            "start_sample_index": 5,
            "stop_sample_index_exclusive": 20,
        },
        {
            "dataset_id": 0,
            "source_label_index": 1,
            "segment_index": "",
            "exported": False,
            "skip_reason": "no_complete_touch_pair",
            "label": "b",
            "label_timestamp_us": 12_000_000.0,
            "start_sample_index": "",
            "stop_sample_index_exclusive": "",
        },
    ]


def _intervals() -> list[dict[str, object]]:
    return [
        {
            "dataset_id": 0,
            "segment_index": 0,
            "paired_touch_index": 4,
            "press_segment_local_index": 5,
            "lift_segment_local_index": 10,
            "retained_in_segment": True,
            "segment_boundary_clipped_start": False,
            "segment_boundary_clipped_end": False,
        }
    ]


def test_writing_motion_verification_matches_segmentation_panel_contract(
    tmp_path: Path,
) -> None:
    timestamps = np.arange(0.0, 25_100_000.0, 100_000.0)
    output = tmp_path / "writing.png"
    result = create_writing_motion_verification_figure(
        canonical_timestamps_us=timestamps,
        display_timestamps_us=timestamps,
        transient_score=np.linspace(0.0, 5.0, len(timestamps)),
        board_events=_events(),
        segment_rows=_segments(),
        writing_interval_rows=_intervals(),
        output_path=output,
        user="user_0",
        action="0",
        dataset_id=0,
        config=SegmentationVerificationConfig(output_dpi=80),
    )
    assert result.panel_count == 3
    assert result.displayed_event_count == 4
    assert result.displayed_label_count == 2
    assert result.displayed_segment_count == 1
    assert result.displayed_writing_interval_count == 1
    assert output.is_file() and output.stat().st_size > 0


def test_writing_span_uses_inclusive_lift_sample() -> None:
    timestamps = np.arange(0.0, 1_000_000.0, 100_000.0)
    start_s, stop_s = _index_window_seconds(
        2,
        5,
        display_timestamps_us=timestamps,
        display_start_us=0.0,
    )
    assert start_s == 0.2
    assert stop_s == 0.5


def test_writing_motion_legend_extends_source_verification_semantics() -> None:
    labels = [
        handle.get_label()
        for handle in _legend_handles(SegmentationVerificationConfig())
    ]
    assert labels[:8] == [
        "Ring transient score",
        "Valid Board press",
        "Valid Board lift",
        "Transient Board press",
        "Transient Board lift",
        "Timestamp label",
        "Skipped label",
        "Exported segment",
    ]
    assert labels[8:] == [
        "Retained writing/contact",
        "Final segment start",
        "Final segment end",
    ]
