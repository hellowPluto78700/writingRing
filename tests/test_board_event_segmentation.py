from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import writingring.board_event_segmentation as board_event_segmentation
from writingring.alignment_io import AlignmentOffset, write_alignment_offset_txt
from writingring.board_event_segmentation import (
    ALIGNED_BOARD_EVENT_COLUMNS,
    ALIGNED_TOUCH_PAIR_COLUMNS,
    BoardEventSegmentationConfig,
    BoardEventSegmentationError,
    align_board_event_tables,
    prepare_complete_board_events,
    read_recording_alignment_offset,
    segment_recording_by_aligned_board_events,
    segment_user_action_by_aligned_board_events,
)
from writingring.discovery import Recording
from writingring.segmentation import SegmentLabel, segment_user_action
from writingring.segmentation_verification import SegmentationVerificationError
from writingring.gravity import GravityRemovalConfig


def _ring(*, end_us: int = 15_000_000) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(0, end_us + 1, 100_000, dtype=np.float64)
    imu = np.column_stack([timestamps + axis for axis in range(6)])
    return imu, timestamps


def _labels(*values: tuple[float, str]) -> tuple[SegmentLabel, ...]:
    return tuple(
        SegmentLabel(timestamp_us=timestamp, label=label, source_line_number=index)
        for index, (timestamp, label) in enumerate(values, start=1)
    )


def _aligned_tables(
    pairs: list[tuple[float, float | None, bool]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    event_index = 0
    for pair_index, (press, lift, transient) in enumerate(pairs):
        incomplete = lift is None
        valid = not transient and not incomplete
        event_rows.append(
            {
                "event_index": event_index,
                "event_type": "press",
                "global_frame_index": event_index,
                "frame_timestamp_raw": press,
                "paired_touch_index": pair_index,
                "duration_frames": 4 if not incomplete else pd.NA,
                "transient": transient,
                "valid_touch": valid,
                "incomplete_touch": incomplete,
                "crossing_touch": False,
                "aligned_event_timestamp_us": press,
            }
        )
        press_event_index = event_index
        event_index += 1
        if lift is not None:
            event_rows.append(
                {
                    "event_index": event_index,
                    "event_type": "lift",
                    "global_frame_index": event_index,
                    "frame_timestamp_raw": lift,
                    "paired_touch_index": pair_index,
                    "duration_frames": 4,
                    "transient": transient,
                    "valid_touch": valid,
                    "incomplete_touch": False,
                    "crossing_touch": False,
                    "aligned_event_timestamp_us": lift,
                }
            )
        pair_rows.append(
            {
                "paired_touch_index": pair_index,
                "press_event_index": press_event_index,
                "lift_event_index": event_index if lift is not None else pd.NA,
                "press_global_frame_index": press_event_index,
                "lift_global_frame_index": event_index if lift is not None else pd.NA,
                "press_timestamp_raw": press,
                "lift_timestamp_raw": lift if lift is not None else np.nan,
                "duration_frames": 4 if lift is not None else pd.NA,
                "duration_s": 0.0 if lift is None else (lift - press) / 1_000_000.0,
                "transient": transient,
                "valid_touch": valid,
                "incomplete_touch": incomplete,
                "crossing_touch": False,
                "aligned_press_us": press,
                "aligned_lift_us": lift if lift is not None else np.nan,
            }
        )
        if lift is not None:
            event_index += 1
    return (
        pd.DataFrame(event_rows, columns=ALIGNED_BOARD_EVENT_COLUMNS),
        pd.DataFrame(pair_rows, columns=ALIGNED_TOUCH_PAIR_COLUMNS),
    )


def test_prepare_board_events_trims_stale_tail_and_classifies_independent_lift() -> None:
    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(6),
            "frame_timestamp_raw": [0, 1_000_000, 1_100_000, 1_200_000, 1_300_000, 5],
            "chunk_index": 0,
            "contact_count": [0, 1, 1, 1, 0, 0],
        }
    )
    contacts = pd.DataFrame({"global_frame_index": np.arange(6)})

    prepared = prepare_complete_board_events(frames, contacts)

    assert len(prepared.frames) == 5
    assert prepared.stale_tail_start_frame_index == 5
    assert prepared.stale_tail_frame_count == 1
    assert prepared.touch_pairs.iloc[0]["valid_touch"]
    assert not frames.equals(prepared.frames)

    independent_lift_frames = frames.iloc[:3].copy()
    independent_lift_frames["contact_count"] = [1, 1, 0]
    independent = prepare_complete_board_events(independent_lift_frames, contacts.iloc[:3])
    assert bool(independent.events.iloc[0]["incomplete_touch"])


def test_raw_board_assisted_imu_bypasses_gravity_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 20.0],
        ],
        dtype=np.float64,
    )
    ring_path = tmp_path / "0_ring_0.bin"
    raw.tofile(ring_path)
    recording = Recording(
        user="user_0", action="0", dataset_id=0, ring_0_path=ring_path,
        ring_1_path=None, timestamp_path=None, board_chunk_paths=(),
        missing_chunk_indices=(), warnings=(),
    )

    def unexpected_gravity(*args: object, **kwargs: object) -> object:
        raise AssertionError("raw Board-assisted mode must not remove gravity")

    monkeypatch.setattr(
        "writingring.imu_preprocessing.process_ring_gravity", unexpected_gravity
    )
    _, imu, timestamps = board_event_segmentation._load_gravity_removed_ring(
        recording,
        gravity_config=GravityRemovalConfig(gravity_removal_method="raw"),
    )

    np.testing.assert_allclose(imu[:, :3], raw[:, :3] / 9.80665)
    np.testing.assert_array_equal(imu[:, 3:6], raw[:, :3])
    np.testing.assert_array_equal(imu[:, 6:9], raw[:, 3:6])
    np.testing.assert_array_equal(timestamps, raw[:, 6])


def test_alignment_is_identity_checked_and_applied_once(tmp_path: Path) -> None:
    events, pairs = _aligned_tables([(10_000_000.0, 11_000_000.0, False)])
    events = events.drop(
        columns=[
            "aligned_event_timestamp_us",
            "crosses_next_label_timestamp",
            "crossing_resolution",
            "assigned_label_index",
        ]
    )
    pairs = pairs.drop(
        columns=[
            "aligned_press_us",
            "aligned_lift_us",
            "crosses_next_label_timestamp",
            "crossing_resolution",
            "assigned_label_index",
        ]
    )
    offset = AlignmentOffset(
        user="user_0", action="0", dataset_id=7, ring_stream="ring_0",
        offset_us=500_000.0, alignment_model="constant_offset", alignment_success=True,
        event_coverage_ratio=1.0, matched_event_count=2, total_valid_event_count=2,
    )
    path = write_alignment_offset_txt(offset, output_path=tmp_path / "offset.txt")
    recording = Recording(
        user="user_0", action="0", dataset_id=7, ring_0_path=tmp_path / "7_ring_0.bin",
        ring_1_path=None, timestamp_path=None, board_chunk_paths=(),
        missing_chunk_indices=(), warnings=(),
    )

    loaded = read_recording_alignment_offset(path, recording=recording)
    aligned_events, aligned_pairs = align_board_event_tables(events, pairs, offset=loaded)

    assert aligned_events["aligned_event_timestamp_us"].tolist() == [10_500_000.0, 11_500_000.0]
    assert aligned_pairs["aligned_press_us"].tolist() == [10_500_000.0]
    with pytest.raises(BoardEventSegmentationError, match="twice"):
        align_board_event_tables(aligned_events, aligned_pairs, offset=loaded)


def test_single_touch_generates_context_window_and_event_targets() -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables([(10_500_000.0, 11_500_000.0, False)])

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "a"), (13_000_000.0, "b")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    assert [sample.label for sample in result.samples] == ["a"]
    sample = result.samples[0]
    assert sample.final_start_timestamp_us == 10_300_000.0
    assert sample.final_end_timestamp_us == 11_700_000.0
    assert sample.imu.shape == (14, 6)
    assert sample.board_event_targets.shape == (14, 4)
    assert sample.board_event_targets[2, 0]
    assert sample.board_event_targets[12, 1]
    assert {skipped.skip_reason for skipped in result.skipped} == {"no_complete_touch_pair"}


def test_work_axis_boundary_lookup_slices_canonical_feature_indices() -> None:
    canonical = np.array(
        [0.0, 900_000.0, 1_100_000.0, 1_600_000.0, 2_500_000.0, 3_000_000.0]
    )
    work = np.array(
        [0.0, 1_000_000.0, 1_250_000.0, 1_750_000.0, 2_500_000.0, 3_000_000.0]
    )
    imu = np.column_stack([np.arange(len(canonical), dtype=np.float64)] * 2)
    events, pairs = _aligned_tables([(1_200_000.0, 1_800_000.0, False)])

    result = segment_recording_by_aligned_board_events(
        feature_values=imu,
        timestamps_us=canonical,
        labels=_labels((1_000_000.0, "a"), (3_000_000.0, "b")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
        boundary_timestamps_us=work,
        config=BoardEventSegmentationConfig(
            pre_press_context_us=0.0,
            post_lift_context_us=0.0,
        ),
    )

    assert len(result.samples) == 1
    sample = result.samples[0]
    assert (sample.start_sample_index, sample.stop_sample_index_exclusive) == (2, 4)
    np.testing.assert_array_equal(sample.imu, imu[2:4])
    assert sample.final_start_timestamp_us == pytest.approx(1_100_000.0)
    assert sample.final_end_timestamp_us == pytest.approx(2_500_000.0)
    assert sample.boundary_start_timestamp_us == pytest.approx(1_200_000.0)
    assert sample.boundary_end_timestamp_us == pytest.approx(1_800_000.0)


def test_multiple_touches_and_transient_events_are_preserved_in_targets() -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables(
        [
            (10_500_000.0, 10_900_000.0, False),
            (11_000_000.0, 11_100_000.0, True),
            (12_000_000.0, 12_500_000.0, False),
        ]
    )

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "a"), (13_000_000.0, "b")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    sample = result.samples[0]
    assert (sample.final_start_timestamp_us, sample.final_end_timestamp_us) == (10_300_000.0, 12_700_000.0)
    assert sample.valid_touch_pair_count == 2
    assert sample.transient_touch_pair_count == 1
    assert sample.board_event_targets[:, 2].sum() == 1
    assert sample.board_event_targets[:, 3].sum() == 1


def test_collision_resolution_uses_next_label_as_hard_boundary() -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables(
        [(10_500_000.0, 12_900_000.0, False), (13_100_000.0, 13_500_000.0, False)]
    )

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "a"), (13_000_000.0, "b")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    assert [sample.label for sample in result.samples] == ["a", "b"]
    first, second = result.samples
    assert first.final_end_timestamp_us == 13_000_000.0
    assert second.final_start_timestamp_us == 13_000_000.0
    assert first.boundary_collision_resolved and second.boundary_collision_resolved
    assert (first.first_press_paired_touch_index, first.last_lift_paired_touch_index) == (0, 0)
    assert (second.first_press_paired_touch_index, second.last_lift_paired_touch_index) == (1, 1)


def test_crossing_touch_and_label_skip_rules_export_no_segment() -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables([(12_500_000.0, 13_100_000.0, False)])

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "a"), (13_000_000.0, "b")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    assert result.samples == ()
    assert result.skipped[0].skip_reason == "touch_crosses_next_label"
    assert result.skipped[0].crossing_touch_pair_count == 1
    assert result.aligned_touch_pairs.iloc[0]["crossing_touch"]

    close = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "a"), (10_050_000.0, "b")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )
    assert {item.skip_reason for item in close.skipped} >= {"adjacent_interval_lt_0.1s"}


def test_crossing_events_are_never_targets_when_neighbor_has_a_segment() -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables(
        [
            (10_500_000.0, 11_000_000.0, False),
            (12_800_000.0, 13_100_000.0, False),
            (13_200_000.0, 13_500_000.0, False),
        ]
    )

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "a"), (13_000_000.0, "b")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    assert [sample.label for sample in result.samples] == ["a", "b"]
    first, second = result.samples
    assert first.crossing_touch_pair_count == 1
    assert second.board_event_targets[1, 1] == 0  # crossing lift at 13.1 s
    assert second.board_event_targets[:, 0].sum() == 1
    assert second.board_event_targets[:, 1].sum() == 1


def test_accepted_crossing_is_owned_by_press_label_and_writes_targets() -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables(
        [
            (12_800_000.0, 13_200_000.0, False),
            (13_600_000.0, 13_900_000.0, False),
        ]
    )

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "a"), (13_000_000.0, "b")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    first, second = result.samples
    assert [sample.label for sample in result.samples] == ["a", "b"]
    assert first.board_event_targets[:, 0].sum() == 1
    assert first.board_event_targets[:, 1].sum() == 1
    assert (first.first_press_paired_touch_index, first.last_lift_paired_touch_index) == (0, 0)
    crossing = result.aligned_touch_pairs.iloc[0]
    assert bool(crossing["crosses_next_label_timestamp"])
    assert crossing["crossing_resolution"] == "accepted_before_next_press"
    assert crossing["assigned_label_index"] == 0
    assert second.board_event_targets[:, :2].sum() == 2


def test_rejected_crossing_overlapping_next_press_has_no_target_or_assignment() -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables(
        [
            (12_800_000.0, 13_200_000.0, False),
            (13_100_000.0, 13_500_000.0, False),
        ]
    )

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "a"), (13_000_000.0, "b")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    assert [sample.label for sample in result.samples] == ["b"]
    rejected = result.aligned_touch_pairs.iloc[0]
    assert rejected["crossing_resolution"] == "rejected_overlaps_next_press"
    assert pd.isna(rejected["assigned_label_index"])
    assert result.samples[0].board_event_targets[:, :2].sum() == 2


def test_normal_and_accepted_crossing_use_last_lift_and_midpoint_context_split() -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables(
        [
            (10_500_000.0, 11_000_000.0, False),
            (12_800_000.0, 13_300_000.0, False),
            (13_500_000.0, 13_800_000.0, False),
        ]
    )

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "a"), (13_000_000.0, "b")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    first, second = result.samples
    assert first.last_lift_timestamp_us == 13_300_000.0
    assert first.last_lift_paired_touch_index == 1
    assert first.board_event_targets[:, :2].sum() == 4
    assert first.final_end_timestamp_us == 13_400_000.0
    assert second.final_start_timestamp_us == 13_400_000.0
    assert first.final_end_timestamp_us <= second.final_start_timestamp_us
    assert first.boundary_collision_resolved and second.boundary_collision_resolved


@pytest.mark.parametrize(
    ("pairs_input", "reason", "field"),
    [
        ([(10_500_000.0, 10_600_000.0, True)], "only_transient_touches", "transient_touch_pair_count"),
        ([(10_500_000.0, None, False)], "incomplete_touch", "incomplete_touch_pair_count"),
    ],
)
def test_skipped_labels_retain_their_touch_classification_counts(
    pairs_input: list[tuple[float, float | None, bool]],
    reason: str,
    field: str,
) -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables(pairs_input)
    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "a"), (13_000_000.0, "b")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    skipped = next(item for item in result.skipped if item.source_label_index == 0)
    assert skipped.skip_reason == reason
    assert getattr(skipped, field) == 1


@pytest.mark.parametrize("lift", [10_400_000.0, 10_500_000.0])
def test_invalid_complete_touch_pair_order_is_rejected(lift: float) -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables([(10_500_000.0, lift, False)])

    with pytest.raises(BoardEventSegmentationError, match="press_timestamp < lift_timestamp"):
        segment_recording_by_aligned_board_events(
            ring_imu=imu,
            ring_timestamps_us=timestamps,
            labels=_labels((10_000_000.0, "a")),
            aligned_board_events=events,
            aligned_touch_pairs=pairs,
        )


def test_config_rejects_unimplemented_fallback_policy() -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables([])
    with pytest.raises(BoardEventSegmentationError, match="missing_event_policy"):
        segment_recording_by_aligned_board_events(
            ring_imu=imu,
            ring_timestamps_us=timestamps,
            labels=_labels((10_000_000.0, "a")),
            aligned_board_events=events,
            aligned_touch_pairs=pairs,
            config=BoardEventSegmentationConfig(missing_event_policy="fallback-to-label"),
        )


def test_user_action_aggregation_publishes_arrays_audit_and_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    action_dir = data_root / "user_0" / "0"
    action_dir.mkdir(parents=True)
    timestamps = 1_000_000.0 + np.arange(700) * 10_000.0
    rows = np.column_stack(
        (
            np.zeros((len(timestamps), 2)),
            np.full(len(timestamps), 9.8),
            np.zeros((len(timestamps), 3)),
            timestamps,
        )
    )
    rows.astype(np.float64).tofile(action_dir / "0_ring_0.bin")
    (action_dir / "0_timestamp.txt").write_text("1000000 a\n3000000 b\n", encoding="utf-8")
    offset_root = tmp_path / "offsets"
    offset_path = offset_root / "user_0" / "action_0" / "0_ring_board_offset.txt"
    write_alignment_offset_txt(
        AlignmentOffset(
            user="user_0", action="0", dataset_id=0, ring_stream="ring_0",
            offset_us=0.0, alignment_model="constant_offset", alignment_success=True,
            event_coverage_ratio=1.0, matched_event_count=2, total_valid_event_count=2,
        ),
        output_path=offset_path,
    )
    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(10),
            "frame_timestamp_raw": [
                1_000_000, 1_500_000, 1_600_000, 1_700_000, 1_800_000,
                1_900_000, 2_000_000, 2_100_000, 2_200_000, 2_300_000,
            ],
            "chunk_index": 0,
            "contact_count": [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        }
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda recording: type("Board", (), {"frames": frames, "contacts": pd.DataFrame({"global_frame_index": np.arange(10)})})(),
    )

    result = segment_user_action_by_aligned_board_events(
        data_root=data_root,
        user="user_0",
        action="0",
        output_root=tmp_path / "outputs",
        alignment_offset_root=offset_root,
    )

    base = tmp_path / "outputs" / "user_0" / "action_0"
    assert result.raw_imu.shape[1] == 9
    assert result.summary["channel_count"] == 9
    assert result.board_event_targets.shape == (len(result.raw_imu), 4)
    assert result.segment_offsets.shape == (len(result.labels) + 1,)
    assert result.summary["verification_image_count"] == 1
    assert result.summary["recordings_without_verification"] == []
    assert (base / "0_ring_0_segmentation_verification.png").is_file()
    assert result.output_paths.board_events_csv_path.is_file()
    assert len(result.manifest) == 2
    assert result.manifest["exported"].tolist() == [True, False]
    assert result.board_events["event_target_channel"].notna().sum() == 4
    boundary_events = result.board_events.loc[
        result.board_events["used_for_segment_boundary"]
    ]
    assert boundary_events["boundary_role"].tolist() == ["start", "end"]
    middle_events = result.board_events.loc[
        result.board_events["event_target_channel"].notna()
        & ~result.board_events["used_for_segment_boundary"]
    ]
    assert middle_events["boundary_role"].tolist() == ["none", "none"]
    skipped = result.manifest.loc[~result.manifest["exported"]].iloc[0]
    assert skipped["valid_touch_pair_count"] == 0
    label_result = segment_user_action(
        data_root=data_root,
        user="user_0",
        action="0",
        output_root=tmp_path / "label_outputs",
    )
    assert label_result.summary["boundary_mode"] == "label"
    assert label_result.output_paths.raw_imu_path.is_file()
    assert result.output_paths.raw_imu_path.is_file()


def test_user_action_aligned_mode_rejects_missing_offset_without_publishing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    action_dir = data_root / "user_0" / "0"
    action_dir.mkdir(parents=True)
    np.zeros((10, 7), dtype=np.float64).tofile(action_dir / "0_ring_0.bin")
    (action_dir / "0_timestamp.txt").write_text("0 a\n", encoding="utf-8")

    with pytest.raises(BoardEventSegmentationError, match="alignment offset is required"):
        segment_user_action_by_aligned_board_events(
            data_root=data_root,
            user="user_0",
            action="0",
            output_root=tmp_path / "outputs",
            alignment_offset_root=tmp_path / "offsets",
        )
    assert not (tmp_path / "outputs" / "user_0" / "action_0").exists()


def test_user_action_missing_board_preserves_existing_aligned_output(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    action_dir = data_root / "user_0" / "0"
    action_dir.mkdir(parents=True)
    timestamps = 1_000_000.0 + np.arange(700) * 10_000.0
    imu = np.column_stack(
        (
            np.zeros((len(timestamps), 2)),
            np.full(len(timestamps), 9.8),
            np.zeros((len(timestamps), 3)),
        )
    )
    np.column_stack((imu, timestamps)).astype(np.float64).tofile(
        action_dir / "0_ring_0.bin"
    )
    (action_dir / "0_timestamp.txt").write_text("1000000 a\n", encoding="utf-8")
    offset_root = tmp_path / "offsets"
    write_alignment_offset_txt(
        AlignmentOffset(
            user="user_0", action="0", dataset_id=0, ring_stream="ring_0",
            offset_us=0.0, alignment_model="constant_offset", alignment_success=True,
            event_coverage_ratio=1.0, matched_event_count=2, total_valid_event_count=2,
        ),
        output_path=offset_root / "user_0" / "action_0" / "0_ring_board_offset.txt",
    )
    existing = tmp_path / "outputs" / "user_0" / "action_0"
    existing.mkdir(parents=True)
    sentinel = existing / "known-good.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(BoardEventSegmentationError, match="could not load Board data"):
        segment_user_action_by_aligned_board_events(
            data_root=data_root,
            user="user_0",
            action="0",
            output_root=tmp_path / "outputs",
            alignment_offset_root=offset_root,
            overwrite=True,
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not list(existing.glob("*_rawIMU.npy"))


def test_verification_failure_preserves_existing_aligned_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    action_dir = data_root / "user_0" / "0"
    action_dir.mkdir(parents=True)
    timestamps = 1_000_000.0 + np.arange(700) * 10_000.0
    rows = np.column_stack(
        (
            np.zeros((len(timestamps), 2)),
            np.full(len(timestamps), 9.8),
            np.zeros((len(timestamps), 3)),
            timestamps,
        )
    )
    rows.astype(np.float64).tofile(action_dir / "0_ring_0.bin")
    (action_dir / "0_timestamp.txt").write_text("1000000 a\n3000000 b\n", encoding="utf-8")
    offset_root = tmp_path / "offsets"
    write_alignment_offset_txt(
        AlignmentOffset(
            user="user_0", action="0", dataset_id=0, ring_stream="ring_0",
            offset_us=0.0, alignment_model="constant_offset", alignment_success=True,
            event_coverage_ratio=1.0, matched_event_count=2, total_valid_event_count=2,
        ),
        output_path=offset_root / "user_0" / "action_0" / "0_ring_board_offset.txt",
    )
    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(5),
            "frame_timestamp_raw": [1_000_000, 1_500_000, 1_600_000, 1_700_000, 2_000_000],
            "chunk_index": 0,
            "contact_count": [0, 1, 1, 1, 0],
        }
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda recording: type("Board", (), {"frames": frames, "contacts": pd.DataFrame({"global_frame_index": np.arange(5)})})(),
    )

    def fail_verification(**kwargs: object) -> None:
        raise SegmentationVerificationError("simulated PNG write failure")

    monkeypatch.setattr(
        "writingring.segmentation_verification.create_segmentation_verification_figure",
        fail_verification,
    )
    existing = tmp_path / "outputs" / "user_0" / "action_0"
    existing.mkdir(parents=True)
    sentinel = existing / "known-good.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(BoardEventSegmentationError, match="segmentation verification failed"):
        segment_user_action_by_aligned_board_events(
            data_root=data_root,
            user="user_0",
            action="0",
            output_root=tmp_path / "outputs",
            alignment_offset_root=offset_root,
            overwrite=True,
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not list(existing.glob("*_rawIMU.npy"))
