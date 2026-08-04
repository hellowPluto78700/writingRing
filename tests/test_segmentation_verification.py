from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from writingring.board_event_segmentation import BoardEventSegmentedSample
from writingring.segmentation import SegmentLabel
from writingring.segmentation_verification import (
    SegmentationVerificationConfig,
    SegmentationVerificationError,
    _in_panel,
    _fixed_legend_handles,
    create_segmentation_verification_figure,
)


def _sample(start: float, stop: float) -> BoardEventSegmentedSample:
    count = int((stop - start) / 100_000)
    return BoardEventSegmentedSample(
        imu=np.zeros((count, 6), dtype=np.float32),
        board_event_targets=np.zeros((count, 4), dtype=np.bool_),
        label="a",
        source_label_index=0,
        sample_count=count,
        start_sample_index=int(start / 100_000),
        stop_sample_index_exclusive=int(stop / 100_000),
        label_timestamp_us=start,
        next_label_timestamp_us=None,
        first_press_timestamp_us=start + 100_000,
        last_lift_timestamp_us=stop - 100_000,
        provisional_start_timestamp_us=start,
        provisional_end_timestamp_us=stop,
        final_start_timestamp_us=start,
        final_end_timestamp_us=stop,
        valid_touch_pair_count=1,
        transient_touch_pair_count=0,
        incomplete_touch_pair_count=0,
        crossing_touch_pair_count=0,
        accepted_crossing_touch_pair_count=0,
        rejected_crossing_touch_pair_count=0,
        boundary_source="aligned_board_events",
        boundary_collision_resolved=False,
    )


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_type": ["press", "lift"],
            "valid_touch": [True, True],
            "transient": [False, False],
            "incomplete_touch": [False, False],
            "aligned_event_timestamp_us": [1_000_000.0, 1_500_000.0],
        }
    )


def test_full_recording_verification_uses_dynamic_panels_and_writes_png(tmp_path: Path) -> None:
    timestamps = np.arange(0, 25_100_000, 100_000, dtype=np.float64)
    output = tmp_path / "verification.png"

    result = create_segmentation_verification_figure(
        ring_dataframe=pd.DataFrame({"sample": np.arange(len(timestamps))}),
        ring_timestamps_us=timestamps,
        transient_score=np.linspace(0.0, 1.0, len(timestamps)),
        aligned_board_events=_events(),
        labels=(SegmentLabel(500_000.0, "a", 1), SegmentLabel(20_000_000.0, "b", 2)),
        label_skip_reasons=(None, "no_complete_touch_pair"),
        segmented_samples=(_sample(500_000.0, 2_000_000.0),),
        output_path=output,
        user="user_0",
        action="0",
        dataset_id=0,
        boundary_mode="aligned_board_events",
        alignment_offset_us=0.0,
    )

    assert result.panel_count == 3
    assert result.output_path == output
    assert output.is_file() and output.stat().st_size > 0


@pytest.mark.parametrize(
    ("duration_us", "expected_panels"),
    [
        (7_000_000, 1),
        (20_000_000, 2),
        (20_000_001, 3),
        (53_000_000, 6),
    ],
)
def test_verification_panel_count_matches_full_recording_duration(
    tmp_path: Path,
    duration_us: int,
    expected_panels: int,
) -> None:
    timestamps = np.array([0.0, float(duration_us)])
    result = create_segmentation_verification_figure(
        ring_dataframe=pd.DataFrame({"sample": [0, 1]}),
        ring_timestamps_us=timestamps,
        transient_score=np.ones(2),
        aligned_board_events=_events(),
        labels=(SegmentLabel(0.0, "a", 1),),
        label_skip_reasons=(None,),
        segmented_samples=(),
        output_path=tmp_path / f"{duration_us}.png",
        user="user_0",
        action="0",
        dataset_id=0,
        boundary_mode="aligned_board_events",
        alignment_offset_us=0.0,
    )
    assert result.panel_count == expected_panels


def test_fixed_aligned_legend_has_all_eight_semantic_entries() -> None:
    handles = _fixed_legend_handles(
        SegmentationVerificationConfig(), include_board_events=True
    )
    assert [handle.get_label() for handle in handles] == [
        "Ring transient score",
        "Valid Board press",
        "Valid Board lift",
        "Transient Board press",
        "Transient Board lift",
        "Timestamp label",
        "Skipped label",
        "Exported segment",
    ]


def test_panel_boundary_marker_belongs_only_to_following_panel() -> None:
    assert not _in_panel(20.0, 10.0, 20.0, False)
    assert _in_panel(20.0, 20.0, 30.0, False)


def test_verification_rejects_overlapping_final_windows(tmp_path: Path) -> None:
    timestamps = np.arange(0, 3_100_000, 100_000, dtype=np.float64)
    with pytest.raises(SegmentationVerificationError, match="overlap"):
        create_segmentation_verification_figure(
            ring_dataframe=pd.DataFrame({"sample": np.arange(len(timestamps))}),
            ring_timestamps_us=timestamps,
            transient_score=np.ones(len(timestamps)),
            aligned_board_events=_events(),
            labels=(SegmentLabel(0.0, "a", 1),),
            label_skip_reasons=(None,),
            segmented_samples=(_sample(500_000.0, 2_000_000.0), _sample(1_500_000.0, 2_500_000.0)),
            output_path=tmp_path / "bad.png",
            user="user_0",
            action="0",
            dataset_id=0,
            boundary_mode="aligned_board_events",
            alignment_offset_us=0.0,
        )


def test_verification_config_rejects_large_segment_alpha(tmp_path: Path) -> None:
    timestamps = np.arange(0, 1_100_000, 100_000, dtype=np.float64)
    with pytest.raises(SegmentationVerificationError, match="segment_alpha"):
        create_segmentation_verification_figure(
            ring_dataframe=pd.DataFrame({"sample": np.arange(len(timestamps))}),
            ring_timestamps_us=timestamps,
            transient_score=np.ones(len(timestamps)),
            aligned_board_events=_events(),
            labels=(SegmentLabel(0.0, "a", 1),),
            label_skip_reasons=(None,),
            segmented_samples=(),
            output_path=tmp_path / "bad-alpha.png",
            config=SegmentationVerificationConfig(segment_alpha=0.2),
            user="user_0",
            action="0",
            dataset_id=0,
            boundary_mode="aligned_board_events",
            alignment_offset_us=0.0,
        )
