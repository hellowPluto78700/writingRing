from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import writingring.board_event_segmentation as board_event_segmentation
from writingring.alignment_io import (
    AlignmentOffset,
    AlignmentSkipArtifact,
    build_alignment_input_provenance,
    build_alignment_outcome_paths,
    build_board_chunk_provenance,
    make_alignment_skip_artifact,
    publish_alignment_success,
    publish_alignment_skip,
    write_alignment_offset_txt,
)
from writingring.board_event_segmentation import (
    ALIGNED_BOARD_EVENT_COLUMNS,
    ALIGNED_TOUCH_PAIR_COLUMNS,
    BoardEventSegmentationConfig,
    BoardEventSegmentationError,
    BoardEventUserActionSegmentationErrorResult,
    align_board_event_tables,
    prepare_complete_board_events,
    read_recording_alignment_offset,
    segment_recording_by_aligned_board_events,
    segment_user_action_by_aligned_board_events,
)
from writingring.discovery import Recording, discover_recordings
from writingring.recording_features import load_raw_ring_features
from writingring.segmentation import SegmentLabel, segment_user_action
from writingring.segmentation_verification import SegmentationVerificationError
from writingring.gravity import GravityRemovalConfig
from writingring.event_alignment import InitialIntervalNoUsablePairError
from writingring.preprocessing_io import sha256_file


def _ring(*, end_us: int = 15_000_000) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(0, end_us + 1, 100_000, dtype=np.float64)
    imu = np.column_stack([timestamps + axis for axis in range(6)])
    return imu, timestamps


def _board_validation(*, empty_leading_chunk_indices: tuple[int, ...] = ()) -> SimpleNamespace:
    """Return the validation metadata required by the production Board contract."""

    return SimpleNamespace(
        empty_leading_chunk_indices=empty_leading_chunk_indices,
    )


def _labels(*values: tuple[float, str]) -> tuple[SegmentLabel, ...]:
    return tuple(
        SegmentLabel(timestamp_us=timestamp, label=label, source_line_number=index)
        for index, (timestamp, label) in enumerate(values, start=1)
    )


def _publish_raw_success_outcome(
    data_root: Path,
    offset_root: Path,
    board_path: Path,
    *,
    dataset_id: int = 0,
    offset: AlignmentOffset | None = None,
) -> None:
    """Publish the complete current outcome family for a raw fixture."""

    recording = next(
        item for item in discover_recordings(data_root) if item.dataset_id == dataset_id
    )
    feature = load_raw_ring_features(recording)
    input_provenance = build_alignment_input_provenance(feature)
    board_provenance = build_board_chunk_provenance((board_path,))
    paths = build_alignment_outcome_paths(
        offset_root,
        offset_root.parent / "verification",
        offset_root.parent / "reports",
        user=recording.user,
        action=recording.action,
        dataset_id=recording.dataset_id,
    )
    verification_source = offset_root.parent / f"{dataset_id}-alignment-source.png"
    verification_source.write_bytes(b"alignment verification")
    publish_alignment_success(
        paths,
        offset=offset
        or AlignmentOffset(
            user=recording.user,
            action=recording.action,
            dataset_id=recording.dataset_id,
            ring_stream="ring_0",
            offset_us=0.0,
            alignment_model="constant_offset",
            alignment_success=True,
            event_coverage_ratio=1.0,
            matched_event_count=2,
            total_valid_event_count=2,
        ),
        report={
            "recording": {
                "user": recording.user,
                "action": recording.action,
                "dataset_id": recording.dataset_id,
            },
            "input_provenance": input_provenance,
            "board_provenance": board_provenance,
        },
        verification_path=verification_source,
    )


def _publish_raw_skipped_outcome(
    data_root: Path,
    offset_root: Path,
    board_path: Path,
    *,
    dataset_id: int,
) -> None:
    """Publish one strict provenance-valid SKIPPED outcome for a fixture."""

    recording = next(
        item for item in discover_recordings(data_root) if item.dataset_id == dataset_id
    )
    feature = load_raw_ring_features(recording)
    input_provenance = build_alignment_input_provenance(feature)
    board_provenance = build_board_chunk_provenance((board_path,))
    paths = build_alignment_outcome_paths(
        offset_root,
        offset_root.parent / "verification",
        offset_root.parent / "reports",
        user=recording.user,
        action=recording.action,
        dataset_id=recording.dataset_id,
    )
    error = InitialIntervalNoUsablePairError(
        previous_global_frame_index=0,
        next_global_frame_index=1,
        previous_timestamp_raw=1_000_000,
        next_timestamp_raw=999_000,
        prefix_boundary_position=1,
        last_pre_jump_global_frame_index=0,
        total_global_valid_pair_count=1,
        usable_prefix_valid_pair_count=0,
    )
    skip = make_alignment_skip_artifact(
        error,
        user=recording.user,
        action=recording.action,
        dataset_id=recording.dataset_id,
        input_provenance=input_provenance,
        board_provenance=board_provenance,
    )
    publish_alignment_skip(
        paths,
        skip_artifact=skip,
        report={
            "recording": {
                "user": recording.user,
                "action": recording.action,
                "dataset_id": recording.dataset_id,
            },
            "input_provenance": input_provenance,
            "board_provenance": board_provenance,
        },
    )


def _publish_raw_event_coverage_skipped_outcome(
    data_root: Path,
    offset_root: Path,
    board_path: Path,
    *,
    dataset_id: int,
) -> None:
    """Publish a valid confidence-derived skip without using the factory."""

    recording = next(
        item for item in discover_recordings(data_root) if item.dataset_id == dataset_id
    )
    feature = load_raw_ring_features(recording)
    input_provenance = build_alignment_input_provenance(feature)
    board_provenance = build_board_chunk_provenance((board_path,))
    paths = build_alignment_outcome_paths(
        offset_root,
        offset_root.parent / "verification",
        offset_root.parent / "reports",
        user=recording.user,
        action=recording.action,
        dataset_id=recording.dataset_id,
    )
    skip = AlignmentSkipArtifact(
        recording={
            "user": recording.user,
            "action": recording.action,
            "dataset_id": recording.dataset_id,
        },
        diagnostics={
            "matched_event_count": 3,
            "total_valid_event_count": 10,
            "event_coverage_ratio": 0.3,
            "minimum_event_coverage_ratio": 0.75,
            "matched_press_count": 2,
            "total_valid_press_count": 4,
            "press_coverage_ratio": 0.5,
            "matched_lift_count": 1,
            "total_valid_lift_count": 6,
            "lift_coverage_ratio": 1 / 6,
            "fully_matched_touch_pair_count": 1,
            "total_valid_touch_pair_count": 5,
            "minimum_valid_touch_pairs": 5,
            "best_offset_us": -238_451.75,
            "best_vs_second_best_nearly_tied": False,
            "failed_confidence_checks": ["minimum_event_coverage_ratio"],
        },
        input_provenance=input_provenance,
        board_provenance=board_provenance,
        reason="insufficient_event_coverage",
    )
    publish_alignment_skip(
        paths,
        skip_artifact=skip,
        report={
            "recording": {
                "user": recording.user,
                "action": recording.action,
                "dataset_id": recording.dataset_id,
            },
            "input_provenance": input_provenance,
            "board_provenance": board_provenance,
        },
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


def test_board_label_gap_over_five_seconds_exports_short_final_windows() -> None:
    imu, timestamps = _ring(end_us=25_000_000)
    events, pairs = _aligned_tables(
        [
            (10_500_000.0, 11_000_000.0, False),
            (20_500_000.0, 21_000_000.0, False),
        ]
    )

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "a"), (20_000_000.0, "b")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    assert result.samples
    assert [sample.label for sample in result.samples] == ["a", "b"]
    assert (
        result.samples[0].next_label_timestamp_us
        - result.samples[0].label_timestamp_us
        > 5_000_000.0
    )
    assert all(
        sample.final_end_timestamp_us - sample.final_start_timestamp_us
        <= 5_000_000.0
        for sample in result.samples
    )


def test_board_final_window_longer_than_five_seconds_is_skipped() -> None:
    imu, timestamps = _ring(end_us=25_000_000)
    events, pairs = _aligned_tables([(10_500_000.0, 16_000_000.0, False)])

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "too_long"), (20_000_000.0, "next")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    assert result.samples == ()
    first_skip = next(item for item in result.skipped if item.source_label_index == 0)
    assert first_skip.skip_reason == "final_segment_duration_gt_5s"


def test_empty_leading_board_recovery_selects_most_prominent_peak_in_strict_range() -> None:
    timestamps = np.arange(8_000_000.0, 20_000_001.0, 100_000.0)
    imu = np.column_stack([timestamps + axis for axis in range(6)])
    transient_score = np.zeros(len(timestamps), dtype=np.float64)
    # The detector reports peaks near 9.7, 11.7, 12.7, and 14.2 s.  Only the
    # two middle peaks are strictly between the label and first Board frame.
    transient_score[[15, 35, 45, 60]] = [100.0, 20.0, 50.0, 100.0]
    events, pairs = _aligned_tables([(13_500_000.0, 14_000_000.0, False)])

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "recovered")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
        transient_score=transient_score,
        leading_board_chunk_empty=True,
        first_aligned_board_frame_us=13_000_000.0,
    )

    assert [sample.label for sample in result.samples] == ["recovered"]
    sample = result.samples[0]
    assert sample.boundary_start_timestamp_us == 12_500_000.0
    assert sample.final_start_timestamp_us == 12_500_000.0
    assert sample.final_end_timestamp_us - sample.final_start_timestamp_us <= 5_000_000.0


def test_empty_leading_board_recovery_has_no_fallback_without_alignment_grade_peak() -> None:
    timestamps = np.arange(8_000_000.0, 20_000_001.0, 100_000.0)
    imu = np.column_stack([timestamps + axis for axis in range(6)])
    events, pairs = _aligned_tables([(13_500_000.0, 14_000_000.0, False)])

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels((10_000_000.0, "recovery_failed")),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
        transient_score=np.zeros(len(timestamps), dtype=np.float64),
        leading_board_chunk_empty=True,
        first_aligned_board_frame_us=13_000_000.0,
    )

    assert result.samples == ()
    assert result.skipped[0].skip_reason == "first_segment_preboard_strong_peak_not_found"


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


def test_historical_crossing_touch_without_carry_in_exports_no_segment() -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables([(12_400_000.0, 13_100_000.0, False)])

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


def test_carry_in_press_is_owned_by_following_label_and_keeps_nominal_start() -> None:
    imu, timestamps = _ring(end_us=20_000_000)
    events, pairs = _aligned_tables(
        [
            (10_500_000.0, 11_000_000.0, False),
            (12_800_000.0, 13_200_000.0, False),
        ]
    )

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels(
            (10_000_000.0, "a"),
            (13_000_000.0, "b"),
            (17_000_000.0, "c"),
        ),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    assert [sample.label for sample in result.samples] == ["a", "b"]
    carry = result.samples[1]
    assert carry.source_label_index == 1
    assert carry.first_press_paired_touch_index == 1
    assert carry.final_start_timestamp_us == 12_600_000.0
    assert carry.boundary_collision_resolved is False
    carry_pair = result.aligned_touch_pairs.iloc[1]
    assert carry_pair["crossing_resolution"] == "carried_into_next_label"
    assert carry_pair["assigned_label_index"] == 1


def test_carry_in_start_uses_midpoint_when_nominal_context_enters_previous_window() -> None:
    imu, timestamps = _ring(end_us=20_000_000)
    events, pairs = _aligned_tables(
        [
            (11_500_000.0, 12_550_000.0, False),
            (12_800_000.0, 13_200_000.0, False),
        ]
    )

    result = segment_recording_by_aligned_board_events(
        ring_imu=imu,
        ring_timestamps_us=timestamps,
        labels=_labels(
            (10_000_000.0, "a"),
            (13_000_000.0, "b"),
            (17_000_000.0, "c"),
        ),
        aligned_board_events=events,
        aligned_touch_pairs=pairs,
    )

    first, carry = result.samples
    assert first.boundary_end_timestamp_us == 12_750_000.0
    assert carry.boundary_start_timestamp_us == pytest.approx(12_775_000.0)
    assert carry.final_start_timestamp_us == 12_800_000.0
    assert carry.boundary_collision_resolved
    assert carry.first_press_timestamp_us == 12_800_000.0


def test_crossing_touch_after_preceding_lift_does_not_trigger_carry_in() -> None:
    imu, timestamps = _ring(end_us=20_000_000)
    events, pairs = _aligned_tables(
        [
            (10_500_000.0, 12_800_000.0, False),
            (12_500_000.0, 13_100_000.0, False),
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

    assert [sample.label for sample in result.samples] == ["a", "b"]
    crossing = result.aligned_touch_pairs.iloc[1]
    assert crossing["crossing_resolution"] == "accepted_before_next_press"
    assert crossing["assigned_label_index"] == 0


def test_press_older_than_carry_in_lookback_keeps_historical_crossing_ownership() -> None:
    imu, timestamps = _ring(end_us=20_000_000)
    events, pairs = _aligned_tables(
        [
            (12_700_000.0, 13_100_000.0, False),
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

    assert [sample.label for sample in result.samples] == ["a", "b"]
    crossing = result.aligned_touch_pairs.iloc[0]
    assert crossing["crossing_resolution"] == "accepted_before_next_press"
    assert crossing["assigned_label_index"] == 0


def test_crossing_events_are_never_targets_when_neighbor_has_a_segment() -> None:
    imu, timestamps = _ring()
    events, pairs = _aligned_tables(
        [
            (10_500_000.0, 11_000_000.0, False),
            (12_400_000.0, 13_100_000.0, False),
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
            (12_400_000.0, 13_200_000.0, False),
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
            (12_400_000.0, 13_300_000.0, False),
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
    board_path = tmp_path / "0_board_0.gz"
    board_path.write_bytes(b"board provenance")
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
        lambda recording: type(
            "Board",
            (),
            {
                "frames": frames,
                "contacts": pd.DataFrame({"global_frame_index": np.arange(10)}),
                "chunk_paths": (board_path,),
                "validation": _board_validation(),
            },
        )(),
    )
    _publish_raw_success_outcome(data_root, offset_root, board_path)

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
    assert result.summary["source_recording_count"] == 1
    assert result.summary["processed_recording_count"] == 1
    assert result.summary["skipped_recording_count"] == 0
    assert result.summary["recording_skips"] == []
    dependency = result.summary["alignment_outcome_dependency"]
    assert dependency["source_recording_ids"] == [
        {"user": "user_0", "action": "0", "dataset_id": 0}
    ]
    assert dependency["outcomes_by_status"]["SKIPPED"] == []
    success_dependency = dependency["outcomes_by_status"]["SUCCESS"]
    assert success_dependency[0]["identity"] == dependency["source_recording_ids"][0]
    report_path = build_alignment_outcome_paths(
        offset_root,
        offset_root.parent / "verification",
        offset_root.parent / "reports",
        user="user_0",
        action="0",
        dataset_id=0,
    ).report_path
    assert success_dependency[0]["report_sha256"] == sha256_file(report_path)
    assert len(success_dependency[0]["report_sha256"]) == 64
    assert all(
        character in "0123456789abcdef"
        for character in success_dependency[0]["report_sha256"]
    )
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
    assert "alignment_outcome_dependency" not in label_result.summary
    assert label_result.output_paths.raw_imu_path.is_file()
    assert result.output_paths.raw_imu_path.is_file()


def test_user_action_quarantines_one_successful_alignment_recording_and_retains_dependency(
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
    board_paths = {dataset_id: tmp_path / f"{dataset_id}_board_0.gz" for dataset_id in range(3)}
    offset_root = tmp_path / "offsets"
    for dataset_id, board_path in board_paths.items():
        rows.astype(np.float64).tofile(action_dir / f"{dataset_id}_ring_0.bin")
        board_path.write_bytes(b"board provenance")
        _publish_raw_success_outcome(
            data_root, offset_root, board_path, dataset_id=dataset_id
        )
    for dataset_id in range(3):
        (action_dir / f"{dataset_id}_timestamp.txt").write_text(
            "1000000 a\n3000000 b\n", encoding="utf-8"
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
        lambda recording: type(
            "Board",
            (),
            {
                "frames": frames,
                "contacts": pd.DataFrame({"global_frame_index": np.arange(10)}),
                "chunk_paths": (board_paths[recording.dataset_id],),
                "validation": _board_validation(),
            },
        )(),
    )
    original = board_event_segmentation.segment_recording_by_aligned_board_events
    calls = 0

    def fail_only_second_recording(**kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise BoardEventSegmentationError("simulated local boundary failure")
        return original(**kwargs)

    monkeypatch.setattr(
        board_event_segmentation,
        "segment_recording_by_aligned_board_events",
        fail_only_second_recording,
    )

    result = segment_user_action_by_aligned_board_events(
        data_root=data_root,
        user="user_0",
        action="0",
        output_root=tmp_path / "outputs",
        alignment_offset_root=offset_root,
        config=BoardEventSegmentationConfig(recording_error_policy="skip"),
    )

    assert not isinstance(result, BoardEventUserActionSegmentationErrorResult)
    assert result.summary["source_recording_count"] == 3
    assert result.summary["processed_recording_count"] == 2
    assert result.summary["skipped_recording_count"] == 0
    assert result.summary["segmentation_error_recording_count"] == 1
    assert result.summary["segmentation_errors"] == [
        {
            "identity": {"user": "user_0", "action": "0", "dataset_id": 1},
            "alignment_status": "SUCCESS",
            "stage": "board_event_segmentation",
            "error_type": "BoardEventSegmentationError",
            "message": "simulated local boundary failure",
        }
    ]
    assert sorted(result.manifest["dataset_id"].unique().tolist()) == [0, 2]
    dependency = result.summary["alignment_outcome_dependency"]
    assert [entry["identity"]["dataset_id"] for entry in dependency["outcomes_by_status"]["SUCCESS"]] == [0, 1, 2]
    assert dependency["outcomes_by_status"]["SKIPPED"] == []
    assert result.recording_error_report_path is not None
    report = json.loads(result.recording_error_report_path.read_text(encoding="utf-8"))
    assert report["terminal_state"] == "completed_with_recording_errors"
    assert report["segmentation_errors"] == result.summary["segmentation_errors"]


def test_user_action_all_local_segmentation_errors_publish_report_without_empty_package(
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
    (action_dir / "0_timestamp.txt").write_text(
        "1000000 a\n3000000 b\n", encoding="utf-8"
    )
    offset_root = tmp_path / "offsets"
    board_path = tmp_path / "0_board_0.gz"
    board_path.write_bytes(b"board provenance")
    _publish_raw_success_outcome(data_root, offset_root, board_path)
    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(10),
            "frame_timestamp_raw": np.arange(10, dtype=float),
            "chunk_index": 0,
            "contact_count": [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        }
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda _recording: type(
            "Board",
            (),
            {
                "frames": frames,
                "contacts": pd.DataFrame({"global_frame_index": np.arange(10)}),
                "chunk_paths": (board_path,),
                "validation": _board_validation(),
            },
        )(),
    )
    monkeypatch.setattr(
        board_event_segmentation,
        "segment_recording_by_aligned_board_events",
        lambda **_kwargs: (_ for _ in ()).throw(
            BoardEventSegmentationError("simulated local boundary failure")
        ),
    )

    result = segment_user_action_by_aligned_board_events(
        data_root=data_root,
        user="user_0",
        action="0",
        output_root=tmp_path / "outputs",
        alignment_offset_root=offset_root,
        config=BoardEventSegmentationConfig(recording_error_policy="skip"),
    )

    assert isinstance(result, BoardEventUserActionSegmentationErrorResult)
    assert result.summary["terminal_state"] == "all_recordings_error"
    assert result.summary["processed_recording_count"] == 0
    assert result.summary["segmentation_error_recording_count"] == 1
    assert not (tmp_path / "outputs" / "user_0" / "action_0").exists()
    assert result.recording_error_report_path.is_file()


def test_user_action_mixed_success_and_skipped_excludes_recording_skip_from_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    action_dir = data_root / "user_0" / "0"
    action_dir.mkdir(parents=True)
    for dataset_id in (0, 1):
        timestamps = 1_000_000.0 + np.arange(700) * 10_000.0
        rows = np.column_stack(
            (
                np.zeros((len(timestamps), 2)),
                np.full(len(timestamps), 9.8),
                np.zeros((len(timestamps), 3)),
                timestamps,
            )
        )
        rows.astype(np.float64).tofile(action_dir / f"{dataset_id}_ring_0.bin")
    (action_dir / "0_timestamp.txt").write_text(
        "1000000 a\n3000000 b\n", encoding="utf-8"
    )
    offset_root = tmp_path / "offsets"
    board_paths = {
        dataset_id: tmp_path / f"{dataset_id}_board_0.gz"
        for dataset_id in (0, 1)
    }
    for path in board_paths.values():
        path.write_bytes(b"board provenance")
    _publish_raw_success_outcome(
        data_root, offset_root, board_paths[0], dataset_id=0
    )
    _publish_raw_skipped_outcome(
        data_root, offset_root, board_paths[1], dataset_id=1
    )
    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(10),
            "frame_timestamp_raw": [
                1_000_000,
                1_500_000,
                1_600_000,
                1_700_000,
                1_800_000,
                1_900_000,
                2_000_000,
                2_100_000,
                2_200_000,
                2_300_000,
            ],
            "chunk_index": 0,
            "contact_count": [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        }
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda recording: type(
            "Board",
            (),
            {
                "frames": frames,
                "contacts": pd.DataFrame({"global_frame_index": np.arange(10)}),
                "chunk_paths": (board_paths[recording.dataset_id],),
                "validation": _board_validation(),
            },
        )(),
    )

    result = segment_user_action_by_aligned_board_events(
        data_root=data_root,
        user="user_0",
        action="0",
        output_root=tmp_path / "outputs",
        alignment_offset_root=offset_root,
    )

    assert result.summary["source_recording_count"] == 2
    assert result.summary["processed_recording_count"] == 1
    assert result.summary["skipped_recording_count"] == 1
    assert result.summary["skipped_segment_count"] == 1
    assert result.summary["recording_skips"] == [
        {
            "identity": {"user": "user_0", "action": "0", "dataset_id": 1},
            "reason": "initial_interval_no_usable_pair",
            "diagnostics": {
                "previous_global_frame_index": 0,
                "next_global_frame_index": 1,
                "previous_timestamp_raw": 1_000_000.0,
                "next_timestamp_raw": 999_000.0,
                "prefix_boundary_position": 1,
                "last_pre_jump_global_frame_index": 0,
                "total_global_valid_pair_count": 1,
                "usable_prefix_valid_pair_count": 0,
            },
        }
    ]
    dependency = result.summary["alignment_outcome_dependency"]
    assert dependency["source_recording_ids"] == [
        {"user": "user_0", "action": "0", "dataset_id": 0},
        {"user": "user_0", "action": "0", "dataset_id": 1},
    ]
    assert [
        entry["identity"]["dataset_id"]
        for entry in dependency["outcomes_by_status"]["SUCCESS"]
    ] == [0]
    assert [
        entry["identity"]["dataset_id"]
        for entry in dependency["outcomes_by_status"]["SKIPPED"]
    ] == [1]
    skipped_dependency = dependency["outcomes_by_status"]["SKIPPED"][0]
    assert skipped_dependency["reason"] == "initial_interval_no_usable_pair"
    for entry in (
        dependency["outcomes_by_status"]["SUCCESS"]
        + dependency["outcomes_by_status"]["SKIPPED"]
    ):
        report_path = build_alignment_outcome_paths(
            offset_root,
            offset_root.parent / "verification",
            offset_root.parent / "reports",
            user="user_0",
            action="0",
            dataset_id=entry["identity"]["dataset_id"],
        ).report_path
        assert entry["report_sha256"] == sha256_file(report_path)
        assert len(entry["report_sha256"]) == 64
        assert all(character in "0123456789abcdef" for character in entry["report_sha256"])
    assert {
        entry["identity"]["dataset_id"]
        for status_entries in dependency["outcomes_by_status"].values()
        for entry in status_entries
    } == {identity["dataset_id"] for identity in dependency["source_recording_ids"]}
    assert result.summary["skipped_recording_count"] == len(
        dependency["outcomes_by_status"]["SKIPPED"]
    )
    assert result.summary["skipped_segment_count"] == 1
    assert set(result.manifest["dataset_id"]) == {0}
    assert set(result.board_events["dataset_id"]) == {0}
    assert result.summary["verification_image_count"] == 1


def test_user_action_mixed_success_and_event_coverage_skipped_processes_only_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    action_dir = data_root / "user_0" / "0"
    action_dir.mkdir(parents=True)
    for dataset_id in (0, 1):
        timestamps = 1_000_000.0 + np.arange(700) * 10_000.0
        rows = np.column_stack(
            (
                np.zeros((len(timestamps), 2)),
                np.full(len(timestamps), 9.8),
                np.zeros((len(timestamps), 3)),
                timestamps,
            )
        )
        rows.astype(np.float64).tofile(action_dir / f"{dataset_id}_ring_0.bin")
    (action_dir / "0_timestamp.txt").write_text(
        "1000000 a\n3000000 b\n", encoding="utf-8"
    )
    offset_root = tmp_path / "offsets"
    board_paths = {
        dataset_id: tmp_path / f"{dataset_id}_board_0.gz"
        for dataset_id in (0, 1)
    }
    for path in board_paths.values():
        path.write_bytes(b"board provenance")
    _publish_raw_success_outcome(
        data_root, offset_root, board_paths[0], dataset_id=0
    )
    _publish_raw_event_coverage_skipped_outcome(
        data_root, offset_root, board_paths[1], dataset_id=1
    )
    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(10),
            "frame_timestamp_raw": [
                1_000_000,
                1_500_000,
                1_600_000,
                1_700_000,
                1_800_000,
                1_900_000,
                2_000_000,
                2_100_000,
                2_200_000,
                2_300_000,
            ],
            "chunk_index": 0,
            "contact_count": [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        }
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda recording: type(
            "Board",
            (),
            {
                "frames": frames,
                "contacts": pd.DataFrame({"global_frame_index": np.arange(10)}),
                "chunk_paths": (board_paths[recording.dataset_id],),
                "validation": _board_validation(),
            },
        )(),
    )

    result = segment_user_action_by_aligned_board_events(
        data_root=data_root,
        user="user_0",
        action="0",
        output_root=tmp_path / "outputs",
        alignment_offset_root=offset_root,
    )

    assert result.summary["processed_recording_count"] == 1
    assert result.summary["skipped_recording_count"] == 1
    assert result.summary["recording_skips"] == [
        {
            "identity": {"user": "user_0", "action": "0", "dataset_id": 1},
            "reason": "insufficient_event_coverage",
            "diagnostics": {
                "matched_event_count": 3,
                "total_valid_event_count": 10,
                "event_coverage_ratio": 0.3,
                "minimum_event_coverage_ratio": 0.75,
                "matched_press_count": 2,
                "total_valid_press_count": 4,
                "press_coverage_ratio": 0.5,
                "matched_lift_count": 1,
                "total_valid_lift_count": 6,
                "lift_coverage_ratio": 1 / 6,
                "fully_matched_touch_pair_count": 1,
                "total_valid_touch_pair_count": 5,
                "minimum_valid_touch_pairs": 5,
                "best_offset_us": -238_451.75,
                "best_vs_second_best_nearly_tied": False,
                "failed_confidence_checks": ["minimum_event_coverage_ratio"],
            },
        }
    ]
    assert set(result.manifest["dataset_id"]) == {0}
    assert set(result.board_events["dataset_id"]) == {0}


def test_user_action_stale_skipped_outcome_fails_before_labels(
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
    board_path = tmp_path / "0_board_0.gz"
    board_path.write_bytes(b"board provenance")
    _publish_raw_skipped_outcome(
        data_root, tmp_path / "offsets", board_path, dataset_id=0
    )
    board_path.write_bytes(b"stale board provenance")
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda _recording: type(
            "Board",
            (),
            {
                "frames": pd.DataFrame(),
                "contacts": pd.DataFrame(),
                "chunk_paths": (board_path,),
                "validation": _board_validation(),
            },
        )(),
    )

    with pytest.raises(BoardEventSegmentationError, match="alignment outcome validation failed"):
        segment_user_action_by_aligned_board_events(
            data_root=data_root,
            user="user_0",
            action="0",
            output_root=tmp_path / "outputs",
            alignment_offset_root=tmp_path / "offsets",
        )
    assert not (tmp_path / "outputs").exists()


def test_user_action_all_skipped_fails_before_empty_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    action_dir = data_root / "user_0" / "0"
    action_dir.mkdir(parents=True)
    for dataset_id in (0, 1):
        timestamps = 1_000_000.0 + np.arange(700) * 10_000.0
        rows = np.column_stack(
            (
                np.zeros((len(timestamps), 2)),
                np.full(len(timestamps), 9.8),
                np.zeros((len(timestamps), 3)),
                timestamps,
            )
        )
        rows.astype(np.float64).tofile(action_dir / f"{dataset_id}_ring_0.bin")
    offset_root = tmp_path / "offsets"
    board_paths = {
        dataset_id: tmp_path / f"{dataset_id}_board_0.gz"
        for dataset_id in (0, 1)
    }
    for dataset_id, path in board_paths.items():
        path.write_bytes(b"board provenance")
        _publish_raw_skipped_outcome(
            data_root, offset_root, path, dataset_id=dataset_id
        )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda recording: type(
            "Board",
            (),
            {
                "frames": pd.DataFrame(),
                "contacts": pd.DataFrame(),
                "chunk_paths": (board_paths[recording.dataset_id],),
                "validation": _board_validation(),
            },
        )(),
    )

    with pytest.raises(BoardEventSegmentationError, match="at least one SUCCESS"):
        segment_user_action_by_aligned_board_events(
            data_root=data_root,
            user="user_0",
            action="0",
            output_root=tmp_path / "outputs",
            alignment_offset_root=offset_root,
        )
    assert not (tmp_path / "outputs").exists()


def test_user_action_all_event_coverage_skips_fail_before_aggregation_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    action_dir = data_root / "user_0" / "0"
    action_dir.mkdir(parents=True)
    for dataset_id in (0, 1):
        timestamps = 1_000_000.0 + np.arange(700) * 10_000.0
        rows = np.column_stack(
            (
                np.zeros((len(timestamps), 2)),
                np.full(len(timestamps), 9.8),
                np.zeros((len(timestamps), 3)),
                timestamps,
            )
        )
        rows.astype(np.float64).tofile(action_dir / f"{dataset_id}_ring_0.bin")
    offset_root = tmp_path / "offsets"
    board_paths = {
        dataset_id: tmp_path / f"{dataset_id}_board_0.gz"
        for dataset_id in (0, 1)
    }
    for dataset_id, path in board_paths.items():
        path.write_bytes(b"board provenance")
        _publish_raw_event_coverage_skipped_outcome(
            data_root, offset_root, path, dataset_id=dataset_id
        )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda recording: type(
            "Board",
            (),
            {
                "frames": pd.DataFrame(),
                "contacts": pd.DataFrame(),
                "chunk_paths": (board_paths[recording.dataset_id],),
                "validation": _board_validation(),
            },
        )(),
    )

    with pytest.raises(BoardEventSegmentationError, match="at least one SUCCESS"):
        segment_user_action_by_aligned_board_events(
            data_root=data_root,
            user="user_0",
            action="0",
            output_root=tmp_path / "outputs",
            alignment_offset_root=offset_root,
        )
    assert not (tmp_path / "outputs").exists()


def test_user_action_aligned_mode_rejects_missing_outcome_without_publishing(
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
    (action_dir / "0_timestamp.txt").write_text("0 a\n", encoding="utf-8")
    board_path = tmp_path / "0_board_0.gz"
    board_path.write_bytes(b"board provenance")
    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(4),
            "frame_timestamp_raw": [1_000_000, 1_100_000, 1_200_000, 1_300_000],
            "chunk_index": 0,
            "contact_count": [0, 0, 0, 0],
        }
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda _recording: type(
            "Board",
            (),
            {
                "frames": frames,
                "contacts": pd.DataFrame({"global_frame_index": np.arange(4)}),
                "chunk_paths": (board_path,),
                "validation": _board_validation(),
            },
        )(),
    )

    with pytest.raises(BoardEventSegmentationError, match="alignment outcome validation failed"):
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
            config=BoardEventSegmentationConfig(recording_error_policy="skip"),
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
    board_path = tmp_path / "0_board_0.gz"
    board_path.write_bytes(b"board provenance")
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
        lambda recording: type(
            "Board",
            (),
            {
                "frames": frames,
                "contacts": pd.DataFrame({"global_frame_index": np.arange(5)}),
                "chunk_paths": (board_path,),
                "validation": _board_validation(),
            },
        )(),
    )
    _publish_raw_success_outcome(data_root, offset_root, board_path)

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
            config=BoardEventSegmentationConfig(recording_error_policy="skip"),
            overwrite=True,
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not list(existing.glob("*_rawIMU.npy"))
