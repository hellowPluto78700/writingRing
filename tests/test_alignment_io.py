from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from writingring.alignment_io import (
    ALIGNMENT_OUTCOME_SCHEMA_VERSION,
    ALIGNMENT_SKIP_SCHEMA_VERSION,
    AlignmentInputProvenance,
    AlignmentOffset,
    AlignmentOffsetProjection,
    AlignmentOffsetExportError,
    AlignmentOutcomeError,
    AlignmentOutcomeStatus,
    AlignmentSkipArtifact,
    alignment_timestamp_metadata,
    apply_board_to_ring_offset,
    build_alignment_input_provenance,
    build_alignment_time_axes,
    build_alignment_offset_path,
    build_alignment_outcome_paths,
    build_board_chunk_provenance,
    build_alignment_timestamps,
    extract_alignment_offset,
    make_alignment_skip_artifact,
    project_alignment_offset_to_canonical,
    publish_alignment_outcome,
    publish_alignment_skip,
    publish_alignment_success,
    read_alignment_offset_txt,
    read_alignment_outcome_report,
    read_alignment_skip_artifact,
    sha256_array,
    validate_alignment_outcome,
    write_alignment_offset_txt,
    write_alignment_skip_artifact,
)
from writingring.event_alignment import (
    AlignmentConfig,
    InitialIntervalNoUsablePairError,
    SequenceAlignmentResult,
    align_events_to_transient_peaks,
    compute_transient_score_array,
    detect_board_events,
    detect_transient_peak_regions,
)


def _result(*, success: bool = True, offset_us: float = -238_451.75) -> SequenceAlignmentResult:
    return SequenceAlignmentResult(
        success=success,
        best_offset_us=offset_us,
        second_best_offset_us=None,
        candidates=pd.DataFrame(),
        event_matches=pd.DataFrame(),
        touch_pair_matches=pd.DataFrame(),
        report={
            "alignment_model": "constant_offset",
            "event_coverage_ratio": 0.875,
            "matched_event_count": 35,
            "total_valid_event_count": 40,
        },
        warnings=(),
    )


def test_offset_export_round_trip_and_mapping_direction(tmp_path: Path) -> None:
    offset = extract_alignment_offset(
        _result(), user="user_0", action="0", dataset_id=0
    )
    path = build_alignment_offset_path(
        tmp_path / "offsets", user="user_0", action="0", dataset_id=0
    )

    written = write_alignment_offset_txt(offset, output_path=path)
    loaded = read_alignment_offset_txt(
        written,
        expected_user="user_0",
        expected_action="0",
        expected_dataset_id=0,
    )

    assert path.name == "0_ring_board_offset.txt"
    assert path.parent == tmp_path / "offsets" / "user_0" / "action_0"
    assert loaded == offset
    assert loaded.offset_ms == pytest.approx(-238.45175)
    assert "ring_timestamp_us = board_timestamp_us + offset_us" in path.read_text()
    np.testing.assert_allclose(
        apply_board_to_ring_offset(np.array([1_000_000.0]), offset_us=250_000.0),
        [1_250_000.0],
    )


def test_offset_export_rejects_failure_nonfinite_identity_mismatch_and_overwrite(
    tmp_path: Path,
) -> None:
    with pytest.raises(AlignmentOffsetExportError, match="did not succeed"):
        extract_alignment_offset(_result(success=False), user="user_0", action="0", dataset_id=0)
    with pytest.raises(AlignmentOffsetExportError, match="finite"):
        extract_alignment_offset(_result(offset_us=float("nan")), user="user_0", action="0", dataset_id=0)
    offset = extract_alignment_offset(_result(), user="user_0", action="0", dataset_id=0)
    path = tmp_path / "offset.txt"
    write_alignment_offset_txt(offset, output_path=path)
    with pytest.raises(AlignmentOffsetExportError, match="already exists"):
        write_alignment_offset_txt(offset, output_path=path)
    write_alignment_offset_txt(offset, output_path=path, overwrite=True)
    with pytest.raises(AlignmentOffsetExportError, match="identity"):
        read_alignment_offset_txt(
            path,
            expected_user="user_9",
            expected_action="0",
            expected_dataset_id=0,
        )


def test_alignment_timestamp_axis_reconstructs_duplicates_without_mutation() -> None:
    canonical = np.array(
        [1_000_000.0, 1_005_000.0, 1_005_000.0, 1_015_000.0, 1_020_000.0],
        dtype=np.float64,
    )
    original = canonical.copy()

    alignment = build_alignment_timestamps(canonical, sampling_rate_hz=200.0)
    assert alignment.shape == canonical.shape
    assert np.all(np.diff(alignment) > 0.0)
    np.testing.assert_array_equal(canonical, original)
    metadata = alignment_timestamp_metadata(
        canonical,
        alignment,
        sampling_rate_hz=200.0,
    )
    assert metadata == {
        "ordering": "nondecreasing",
        "duplicate_step_count": 1,
        "strategy": "endpoint_reconstruction",
        "sampling_rate_hz": 200.0,
        "sample_count": 5,
        "canonical_start_us": 1_000_000.0,
        "canonical_stop_us": 1_020_000.0,
        "canonical_timestamps_modified": False,
    }
    np.testing.assert_array_equal(
        build_alignment_timestamps(canonical),
        np.linspace(canonical[0], canonical[-1], len(canonical), dtype=np.float64),
    )


def test_alignment_timestamp_axis_preserves_strict_input_and_rejects_backward() -> None:
    canonical = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    alignment = build_alignment_timestamps(canonical, sampling_rate_hz=200.0)
    assert alignment is not canonical
    np.testing.assert_array_equal(alignment, canonical)
    with pytest.raises(AlignmentOffsetExportError, match="nondecreasing"):
        build_alignment_timestamps(np.array([1.0, 3.0, 2.0]))


def test_source_aware_raw_axis_reconstructs_irregular_timestamps() -> None:
    canonical = np.array([1_000_000.0, 1_005_100.0, 1_010_000.0, 1_015_100.0])
    original = canonical.copy()
    axes = build_alignment_time_axes(
        canonical,
        input_kind="raw-ring",
        sampling_rate_hz=200.0,
    )

    expected = np.linspace(
        canonical[0], canonical[-1], len(canonical), dtype=np.float64
    )
    np.testing.assert_array_equal(axes.work_timestamps_us, expected)
    assert axes.strategy == "endpoint_reconstruction"
    assert axes.sampling_rate_hz == pytest.approx(
        (len(canonical) - 1) * 1_000_000.0 / (canonical[-1] - canonical[0])
    )
    assert axes.feature_sampling_rate_hz == pytest.approx(200.0)
    assert not np.array_equal(axes.work_timestamps_us, canonical)
    np.testing.assert_array_equal(canonical, original)


def test_source_aware_uniform_axes_share_endpoint_reconstruction() -> None:
    uniform = np.array([1_000_000.0, 1_005_000.0, 1_010_000.0, 1_015_000.0])
    raw_axes = build_alignment_time_axes(
        uniform,
        input_kind="raw-ring",
        sampling_rate_hz=200.0,
    )
    spike_axes = build_alignment_time_axes(
        uniform,
        input_kind="spike-imu",
        sampling_rate_hz=123.0,
    )
    np.testing.assert_allclose(raw_axes.work_timestamps_us, uniform)
    np.testing.assert_array_equal(spike_axes.work_timestamps_us, uniform)
    assert raw_axes.strategy == "endpoint_reconstruction"
    assert spike_axes.strategy == "endpoint_reconstruction"
    assert spike_axes.sampling_rate_hz == pytest.approx(200.0)
    assert spike_axes.feature_sampling_rate_hz == pytest.approx(123.0)


def test_spike_axis_infers_sampling_rate_without_metadata_rate() -> None:
    canonical = np.array([0.0, 5_000.0, 10_000.0, 15_000.0])
    axes = build_alignment_time_axes(
        canonical,
        input_kind="spike-imu",
        sampling_rate_hz=None,
    )

    assert axes.sampling_rate_hz == pytest.approx(200.0)
    assert axes.feature_sampling_rate_hz is None
    assert axes.strategy == "endpoint_reconstruction"
    np.testing.assert_array_equal(axes.work_timestamps_us, canonical)
    assert np.all(np.diff(axes.work_timestamps_us) > 0.0)


def test_spike_duplicate_axis_infers_and_reconstructs_without_mutation() -> None:
    canonical = np.array([0.0, 5_000.0, 5_000.0, 15_000.0])
    original = canonical.copy()
    axes = build_alignment_time_axes(
        canonical,
        input_kind="spike-imu",
        sampling_rate_hz=333.0,
    )

    assert axes.sampling_rate_hz == pytest.approx(200.0)
    assert axes.feature_sampling_rate_hz == pytest.approx(333.0)
    assert axes.strategy == "endpoint_reconstruction"
    assert np.all(np.diff(axes.work_timestamps_us) > 0.0)
    np.testing.assert_array_equal(canonical, original)
    np.testing.assert_array_equal(
        axes.work_timestamps_us,
        np.linspace(0.0, 15_000.0, 4, dtype=np.float64),
    )


def test_spike_axis_without_rate_rejects_zero_duration_without_assertion() -> None:
    with pytest.raises(
        AlignmentOffsetExportError,
        match="timestamps must have positive duration for alignment",
    ):
        build_alignment_time_axes(
            np.array([1_000.0, 1_000.0, 1_000.0]),
            input_kind="spike-imu",
            sampling_rate_hz=None,
        )


def test_raw_and_spike_alignment_results_are_representation_invariant() -> None:
    sample_count = 400
    increments = np.full(sample_count, 5_000.0, dtype=np.float64)
    increments[20] = 0.0
    increments[100] = 5_100.0
    canonical = 1_000_000.0 + np.cumsum(increments)
    values = np.zeros((sample_count, 6), dtype=np.float64)
    for center in (60, 150, 240, 330):
        values[center - 1 : center + 2, 0] = [1.0, 4.0, 1.0]

    raw_axes = build_alignment_time_axes(
        canonical,
        input_kind="raw-ring",
        sampling_rate_hz=200.0,
    )
    spike_axes = build_alignment_time_axes(
        canonical,
        input_kind="spike-imu",
        sampling_rate_hz=123.0,
    )
    np.testing.assert_array_equal(
        raw_axes.work_timestamps_us,
        spike_axes.work_timestamps_us,
    )

    raw_score = compute_transient_score_array(values)
    spike_score = compute_transient_score_array(values)
    np.testing.assert_array_equal(raw_score, spike_score)
    raw_peaks = detect_transient_peak_regions(raw_score, raw_axes.work_timestamps_us)
    spike_peaks = detect_transient_peak_regions(
        spike_score, spike_axes.work_timestamps_us
    )
    pd.testing.assert_frame_equal(raw_peaks, spike_peaks)

    board_frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(10),
            "frame_timestamp_raw": np.arange(10, dtype=np.int64) * 100_000,
            "chunk_index": np.zeros(10, dtype=np.int64),
            "contact_count": [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        }
    )
    detection = detect_board_events(board_frames)
    selected_peak_times = raw_peaks["peak_timestamp"].to_numpy(dtype=np.float64)
    board_events = detection.events.copy()
    board_events["frame_timestamp_raw"] = selected_peak_times
    touch_pairs = detection.touch_pairs.copy()
    touch_pairs["press_timestamp_raw"] = selected_peak_times[[0, 2]]
    touch_pairs["lift_timestamp_raw"] = selected_peak_times[[1, 3]]
    config = AlignmentConfig(
        offset_search_range_us=(-10_000.0, 10_000.0),
        minimum_event_coverage_ratio=0.5,
        minimum_valid_touch_pairs=1,
    )
    raw_result = align_events_to_transient_peaks(
        board_events,
        touch_pairs,
        raw_peaks,
        ring_timestamp_range=(
            float(raw_axes.work_timestamps_us[0]),
            float(raw_axes.work_timestamps_us[-1]),
        ),
        config=config,
    )
    spike_result = align_events_to_transient_peaks(
        board_events,
        touch_pairs,
        spike_peaks,
        ring_timestamp_range=(
            float(spike_axes.work_timestamps_us[0]),
            float(spike_axes.work_timestamps_us[-1]),
        ),
        config=config,
    )
    assert raw_result.success == spike_result.success
    assert raw_result.best_offset_us == spike_result.best_offset_us
    pd.testing.assert_frame_equal(raw_result.candidates, spike_result.candidates)
    pd.testing.assert_frame_equal(raw_result.event_matches, spike_result.event_matches)
    pd.testing.assert_frame_equal(
        raw_result.touch_pair_matches,
        spike_result.touch_pair_matches,
    )
    assert raw_result.report == spike_result.report

    raw_projection = project_alignment_offset_to_canonical(
        raw_result,
        raw_axes.canonical_timestamps_us,
        raw_axes.work_timestamps_us,
        sampling_rate_hz=raw_axes.alignment_sampling_rate_hz,
    )
    spike_projection = project_alignment_offset_to_canonical(
        spike_result,
        spike_axes.canonical_timestamps_us,
        spike_axes.work_timestamps_us,
        sampling_rate_hz=spike_axes.alignment_sampling_rate_hz,
    )
    assert raw_projection == spike_projection

    raw_offset = extract_alignment_offset(
        replace(
            raw_result,
            report=raw_result.report
            | {"alignment_time_axis": raw_axes.metadata},
        ),
        user="user_0",
        action="0",
        dataset_id=0,
        projection=raw_projection,
    )
    spike_offset = extract_alignment_offset(
        replace(
            spike_result,
            report=spike_result.report
            | {"alignment_time_axis": spike_axes.metadata},
        ),
        user="user_0",
        action="0",
        dataset_id=0,
        projection=spike_projection,
    )
    assert raw_offset.work_axis_offset_us == spike_offset.work_axis_offset_us
    assert raw_offset.offset_us == spike_offset.offset_us


def test_work_offset_is_projected_to_canonical_timestamp_domain() -> None:
    canonical = np.array([0.0, 5_000.0, 5_000.0, 10_000.0, 15_000.0])
    axes = build_alignment_time_axes(
        canonical,
        input_kind="spike-imu",
        sampling_rate_hz=200.0,
    )
    result = replace(
        _result(offset_us=123.0),
        event_matches=pd.DataFrame(
            {
                "matched": [True, True, True],
                "matched_peak_index": [2, 3, 4],
            }
        ),
    )

    projection = project_alignment_offset_to_canonical(
        result,
        axes.canonical_timestamps_us,
        axes.work_timestamps_us,
        sampling_rate_hz=axes.sampling_rate_hz,
    )

    assert isinstance(projection, AlignmentOffsetProjection)
    assert projection.work_axis_offset_us == pytest.approx(123.0)
    assert projection.projection_delta_us == pytest.approx(-1_250.0)
    assert projection.canonical_offset_us == pytest.approx(-1_127.0)
    assert projection.projection_delta_mad_us == pytest.approx(1_250.0)
    assert projection.contributing_match_count == 3
    board_time = axes.work_timestamps_us[3] - projection.work_axis_offset_us
    np.testing.assert_allclose(
        apply_board_to_ring_offset(
            np.asarray([board_time]), offset_us=projection.canonical_offset_us
        ),
        [axes.canonical_timestamps_us[3]],
    )


def test_projection_rejects_nonconstant_canonical_mapping() -> None:
    canonical = np.array([0.0, 5_000.0, 5_000.0, 5_000.0, 10_000.0])
    axes = build_alignment_time_axes(
        canonical,
        input_kind="spike-imu",
        sampling_rate_hz=400.0,
    )
    result = replace(
        _result(offset_us=0.0),
        event_matches=pd.DataFrame(
            {
                "matched": [True, True],
                "matched_peak_index": [1, 3],
            }
        ),
    )
    with pytest.raises(AlignmentOffsetExportError, match="not representable"):
        project_alignment_offset_to_canonical(
            result,
            axes.canonical_timestamps_us,
            axes.work_timestamps_us,
            sampling_rate_hz=axes.sampling_rate_hz,
        )


def test_work_axis_alignment_can_export_when_canonical_projection_fails(
    tmp_path: Path,
) -> None:
    canonical = np.array([0.0, 5_000.0, 5_000.0, 5_000.0, 5_000.0, 15_000.0])
    axes = build_alignment_time_axes(
        canonical,
        input_kind="spike-imu",
        sampling_rate_hz=200.0,
    )
    result = replace(
        _result(offset_us=102_127.25),
        event_matches=pd.DataFrame(
            {
                "matched": [True, True],
                "matched_peak_index": [1, 4],
            }
        ),
        report=_result().report
        | {
            "offset_domain": "alignment_work_axis",
            "alignment_time_axis": axes.metadata,
            "timestamp_source": {
                "sha256": sha256_array(canonical),
                "ordering": "nondecreasing",
                "duplicate_step_count": axes.duplicate_step_count,
            },
        },
    )
    with pytest.raises(AlignmentOffsetExportError, match="not representable") as error_info:
        project_alignment_offset_to_canonical(
            result,
            canonical,
            axes.work_timestamps_us,
            sampling_rate_hz=axes.alignment_sampling_rate_hz,
        )
    diagnostics = error_info.value.projection_diagnostics
    offset = extract_alignment_offset(
        result,
        user="user_2",
        action="0",
        dataset_id=3,
        projection_diagnostics=diagnostics,
    )

    assert offset.alignment_success is True
    assert offset.offset_domain == "alignment_work_axis"
    assert offset.offset_us == pytest.approx(102_127.25)
    assert offset.work_axis_offset_us == pytest.approx(102_127.25)
    assert offset.canonical_offset_us is None
    assert offset.canonical_offset_projection_success is False
    assert offset.projection_delta_range_us is not None
    written = write_alignment_offset_txt(
        offset, output_path=tmp_path / "work-axis-offset.txt"
    )
    loaded = read_alignment_offset_txt(written)
    assert loaded == offset


def test_alignment_offset_round_trip_preserves_timestamp_axis_provenance(
    tmp_path: Path,
) -> None:
    canonical = np.array([1_000_000.0, 1_005_000.0, 1_005_000.0], dtype=np.float64)
    alignment = build_alignment_timestamps(canonical, sampling_rate_hz=200.0)
    base = extract_alignment_offset(
        _result(), user="user_0", action="0", dataset_id=0
    )
    from dataclasses import replace

    offset = replace(
        base,
        timestamp_source_sha256=sha256_array(canonical),
        timestamp_source_ordering="nondecreasing",
        timestamp_source_duplicate_step_count=1,
        feature_sampling_rate_hz=200.0,
        alignment_time_axis_strategy="endpoint_reconstruction",
        alignment_time_axis_sampling_rate_hz=400.0,
        alignment_time_axis_sample_count=len(alignment),
        canonical_timestamps_modified=False,
    )
    path = write_alignment_offset_txt(offset, output_path=tmp_path / "offset.txt")
    loaded = read_alignment_offset_txt(path)
    assert loaded == offset


def test_projected_offset_round_trip_preserves_canonical_offset(tmp_path: Path) -> None:
    canonical = np.array([0.0, 5_000.0, 5_000.0, 10_000.0, 15_000.0])
    axes = build_alignment_time_axes(
        canonical,
        input_kind="spike-imu",
        sampling_rate_hz=200.0,
    )
    result = replace(
        _result(offset_us=123.0),
        event_matches=pd.DataFrame(
            {
                "matched": [True, True, True],
                "matched_peak_index": [2, 3, 4],
            }
        ),
    )
    projection = project_alignment_offset_to_canonical(
        result,
        axes.canonical_timestamps_us,
        axes.work_timestamps_us,
        sampling_rate_hz=axes.sampling_rate_hz,
    )
    offset = extract_alignment_offset(
        result,
        user="user_0",
        action="0",
        dataset_id=0,
        projection=projection,
    )
    metadata = axes.metadata
    offset = replace(
        offset,
        timestamp_source_sha256=sha256_array(canonical),
        timestamp_source_ordering=str(metadata["ordering"]),
        timestamp_source_duplicate_step_count=int(metadata["duplicate_step_count"]),
        alignment_time_axis_strategy=str(metadata["strategy"]),
        alignment_time_axis_sampling_rate_hz=float(metadata["sampling_rate_hz"]),
        alignment_time_axis_sample_count=int(metadata["sample_count"]),
        canonical_timestamps_modified=False,
    )
    path = write_alignment_offset_txt(offset, output_path=tmp_path / "projected.txt")
    loaded = read_alignment_offset_txt(path)
    assert loaded == offset
    assert loaded.offset_us == pytest.approx(-1_127.0)
    assert loaded.work_axis_offset_us == pytest.approx(123.0)


def _outcome_fixture(tmp_path: Path) -> tuple[
    object,
    ...,
]:
    """Create one real ordered Board chunk and a current raw provenance set."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    board_path = tmp_path / "0_board_0.gz"
    board_path.write_bytes(b"Board chunk provenance")
    board = build_board_chunk_provenance((board_path,))
    input_provenance = AlignmentInputProvenance(
        input_kind="raw-ring",
        user="user_0",
        action="0",
        dataset_id=0,
        timestamp_sha256="a" * 64,
        ring_0_sha256="b" * 64,
    )
    paths = build_alignment_outcome_paths(
        tmp_path / "offsets",
        tmp_path / "verification",
        tmp_path / "reports",
        user="user_0",
        action="0",
        dataset_id=0,
    )
    offset = extract_alignment_offset(
        _result(), user="user_0", action="0", dataset_id=0
    )
    verification_source = tmp_path / "source.png"
    verification_source.write_bytes(b"PNG verification")
    report = {
        "recording": input_provenance.recording,
        "input_provenance": input_provenance.to_dict(),
        "board_provenance": [chunk.to_dict() for chunk in board],
        "canonical_offset_projection_success": False,
    }
    return paths, input_provenance, board, offset, verification_source, report


def _publish_fixture(tmp_path: Path):
    paths, input_provenance, board, offset, verification_source, report = (
        _outcome_fixture(tmp_path)
    )
    publish_alignment_success(
        paths,
        offset=offset,
        report=report,
        verification_path=verification_source,
    )
    return paths, input_provenance, board, offset, verification_source, report


def _skip_fixture(tmp_path: Path):
    paths, input_provenance, board, offset, verification_source, _report = (
        _outcome_fixture(tmp_path)
    )
    error = InitialIntervalNoUsablePairError(
        previous_global_frame_index=10,
        next_global_frame_index=11,
        previous_timestamp_raw=1_000,
        next_timestamp_raw=900,
        prefix_boundary_position=11,
        last_pre_jump_global_frame_index=10,
        total_global_valid_pair_count=2,
        usable_prefix_valid_pair_count=0,
    )
    skip = make_alignment_skip_artifact(
        error,
        user="user_0",
        action="0",
        dataset_id=0,
        input_provenance=input_provenance,
        board_provenance=board,
    )
    report = {
        "recording": input_provenance.recording,
        "input_provenance": input_provenance.to_dict(),
        "board_provenance": [chunk.to_dict() for chunk in board],
    }
    return paths, input_provenance, board, skip, report


def _confidence_skip_fixture(tmp_path: Path, *, reason: str):
    paths, input_provenance, board, _offset, _source, _report = _outcome_fixture(
        tmp_path
    )
    if reason == "insufficient_valid_touch_pairs":
        diagnostics = {
            "total_valid_touch_pair_count": 3,
            "minimum_valid_touch_pairs": 5,
            "matched_event_count": 8,
            "total_valid_event_count": 10,
            "event_coverage_ratio": 0.8,
            "minimum_event_coverage_ratio": 0.75,
            "failed_confidence_checks": ["minimum_valid_touch_pairs"],
        }
    elif reason == "insufficient_event_coverage":
        diagnostics = {
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
        }
    else:
        raise AssertionError(f"unsupported test reason: {reason}")
    skip = AlignmentSkipArtifact(
        recording=input_provenance.recording,
        diagnostics=diagnostics,
        input_provenance=input_provenance,
        board_provenance=board,
        reason=reason,
    )
    report = {
        "recording": input_provenance.recording,
        "input_provenance": input_provenance.to_dict(),
        "board_provenance": [chunk.to_dict() for chunk in board],
    }
    return paths, input_provenance, board, skip, report


def test_alignment_outcome_strict_schema_and_provenance_aliases(
    tmp_path: Path,
) -> None:
    paths, input_provenance, board, _offset, _source, _report = _outcome_fixture(tmp_path)
    assert input_provenance.to_dict()["ring_0_sha256"] == "b" * 64
    assert [item["chunk_index"] for item in (chunk.to_dict() for chunk in board)] == [0]
    spike_provenance = build_alignment_input_provenance(
        {
            "input_kind": "spike-imu",
            "recording": input_provenance.recording,
            "feature_schema": "spike_schema_v1",
            "sampling_rate_hz": 200.0,
            "feature_values_sha256": "c" * 64,
            "feature_metadata_sha256": "d" * 64,
            "canonical_timestamps_sha256": "e" * 64,
        }
    )
    assert spike_provenance.to_dict()["feature_schema"] == "spike_schema_v1"
    assert spike_provenance.to_dict()["sampling_rate_hz"] == pytest.approx(200.0)

    error = InitialIntervalNoUsablePairError(
        previous_global_frame_index=0,
        next_global_frame_index=1,
        previous_timestamp_raw=10,
        next_timestamp_raw=9,
        prefix_boundary_position=1,
        last_pre_jump_global_frame_index=0,
        total_global_valid_pair_count=1,
        usable_prefix_valid_pair_count=0,
    )
    artifact = make_alignment_skip_artifact(
        error,
        user="user_0",
        action="0",
        dataset_id=0,
        input_provenance=input_provenance,
        board_provenance=board,
    )
    assert artifact.to_dict()["alignment_skip_schema_version"] == 1
    assert set(artifact.to_dict()["diagnostics"]) == {
        "previous_global_frame_index",
        "next_global_frame_index",
        "previous_timestamp_raw",
        "next_timestamp_raw",
        "prefix_boundary_position",
        "last_pre_jump_global_frame_index",
        "total_global_valid_pair_count",
        "usable_prefix_valid_pair_count",
    }
    with pytest.raises(AlignmentOutcomeError):
        replace(artifact, alignment_skip_schema_version=True)
    with pytest.raises(AlignmentOutcomeError):
        replace(artifact, alignment_skip_schema_version=1.0)

    conflicting = input_provenance.to_dict()
    conflicting["canonical_timestamps_sha256"] = "c" * 64
    with pytest.raises(AlignmentOutcomeError, match="conflicting timestamp"):
        build_alignment_input_provenance(conflicting)

    assert paths.skip_json_path.name == "0_ring_board_skip.json"


@pytest.mark.parametrize(
    "reason",
    ["insufficient_valid_touch_pairs", "insufficient_event_coverage"],
)
def test_confidence_skip_reasons_validate_and_publish(
    tmp_path: Path,
    reason: str,
) -> None:
    paths, input_provenance, board, skip, report = _confidence_skip_fixture(
        tmp_path,
        reason=reason,
    )

    payload = skip.to_dict()
    assert payload["alignment_skip_schema_version"] == ALIGNMENT_SKIP_SCHEMA_VERSION
    assert payload["reason"] == reason
    assert AlignmentSkipArtifact.from_dict(payload).to_dict() == payload

    publish_alignment_skip(
        paths,
        skip_artifact=skip,
        report=report,
    )
    outcome = validate_alignment_outcome(
        paths,
        expected_recording=input_provenance.recording,
        expected_input_provenance=input_provenance,
        expected_board_provenance=board,
    )
    assert outcome.status is AlignmentOutcomeStatus.SKIPPED
    assert not paths.offset_txt_path.exists()
    assert not paths.verification_png_path.exists()


@pytest.mark.parametrize(
    ("reason", "field", "value"),
    [
        (
            "insufficient_valid_touch_pairs",
            "total_valid_touch_pair_count",
            5,
        ),
        (
            "insufficient_event_coverage",
            "event_coverage_ratio",
            0.75,
        ),
        (
            "insufficient_event_coverage",
            "event_coverage_ratio",
            0.3001,
        ),
        (
            "insufficient_event_coverage",
            "press_coverage_ratio",
            0.51,
        ),
        (
            "insufficient_event_coverage",
            "lift_coverage_ratio",
            0.2,
        ),
        (
            "insufficient_event_coverage",
            "total_valid_touch_pair_count",
            4,
        ),
    ],
)
def test_confidence_skip_diagnostics_reject_incoherent_semantics(
    tmp_path: Path,
    reason: str,
    field: str,
    value: object,
) -> None:
    _paths, _input, _board, skip, _report = _confidence_skip_fixture(
        tmp_path,
        reason=reason,
    )
    payload = skip.to_dict()
    payload["diagnostics"][field] = value  # type: ignore[index]

    with pytest.raises(AlignmentOutcomeError):
        AlignmentSkipArtifact.from_dict(payload)


@pytest.mark.parametrize(
    ("reason", "diagnostics"),
    [
        ("unknown", {}),
        ("insufficient_valid_touch_pairs", {"total_valid_touch_pair_count": 1}),
        (
            "insufficient_event_coverage",
            {"total_valid_touch_pair_count": 5},
        ),
        ("insufficient_event_coverage", []),
    ],
)
def test_confidence_skip_diagnostics_reject_unknown_missing_and_wrong_schema(
    tmp_path: Path,
    reason: str,
    diagnostics: object,
) -> None:
    _paths, _input, _board, skip, _report = _confidence_skip_fixture(
        tmp_path,
        reason="insufficient_valid_touch_pairs",
    )
    payload = skip.to_dict()
    payload["reason"] = reason
    payload["diagnostics"] = diagnostics

    with pytest.raises(AlignmentOutcomeError):
        AlignmentSkipArtifact.from_dict(payload)


@pytest.mark.parametrize(
    ("reason", "field", "value"),
    [
        (
            "insufficient_valid_touch_pairs",
            "total_valid_touch_pair_count",
            True,
        ),
        (
            "insufficient_valid_touch_pairs",
            "minimum_valid_touch_pairs",
            5.0,
        ),
        (
            "insufficient_event_coverage",
            "event_coverage_ratio",
            float("nan"),
        ),
        (
            "insufficient_event_coverage",
            "best_offset_us",
            "-238451.75",
        ),
        (
            "insufficient_event_coverage",
            "failed_confidence_checks",
            ["minimum_event_coverage_ratio", "minimum_event_coverage_ratio"],
        ),
        (
            "insufficient_valid_touch_pairs",
            "failed_confidence_checks",
            ["minimum_event_coverage_ratio", "minimum_valid_touch_pairs"],
        ),
    ],
)
def test_confidence_skip_diagnostics_reject_invalid_numeric_and_check_lists(
    tmp_path: Path,
    reason: str,
    field: str,
    value: object,
) -> None:
    _paths, _input, _board, skip, _report = _confidence_skip_fixture(
        tmp_path,
        reason=reason,
    )
    payload = skip.to_dict()
    payload["diagnostics"][field] = value  # type: ignore[index]

    with pytest.raises(AlignmentOutcomeError):
        AlignmentSkipArtifact.from_dict(payload)


@pytest.mark.parametrize("schema_value", [True, 1.0, 1.5, "1"])
def test_alignment_report_schema_version_requires_literal_integer(
    tmp_path: Path,
    schema_value: object,
) -> None:
    paths, input_provenance, board, _offset, _source, _report = _publish_fixture(tmp_path)
    report = json.loads(paths.report_path.read_text(encoding="utf-8"))
    report["alignment_outcome_schema_version"] = schema_value
    paths.report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(AlignmentOutcomeError):
        validate_alignment_outcome(
            paths,
            expected_recording=input_provenance.recording,
            expected_input_provenance=input_provenance,
            expected_board_provenance=board,
        )


def test_alignment_outcome_validator_rejects_stale_partial_malformed_and_digest_files(
    tmp_path: Path,
) -> None:
    paths, input_provenance, board, _offset, _source, _report = _publish_fixture(
        tmp_path / "stale"
    )
    paths.verification_png_path.write_bytes(b"changed")
    with pytest.raises(AlignmentOutcomeError, match="digest"):
        validate_alignment_outcome(
            paths,
            expected_recording=input_provenance.recording,
            expected_input_provenance=input_provenance,
            expected_board_provenance=board,
        )

    paths, input_provenance, board, _offset, _source, _report = _publish_fixture(
        tmp_path / "missing"
    )
    paths.report_path.unlink()
    with pytest.raises(AlignmentOutcomeError):
        validate_alignment_outcome(
            paths,
            expected_recording=input_provenance.recording,
            expected_input_provenance=input_provenance,
            expected_board_provenance=board,
        )

    paths, input_provenance, board, _offset, _source, _report = _publish_fixture(
        tmp_path / "partial"
    )
    paths.verification_png_path.unlink()
    with pytest.raises(AlignmentOutcomeError):
        validate_alignment_outcome(
            paths,
            expected_recording=input_provenance.recording,
            expected_input_provenance=input_provenance,
            expected_board_provenance=board,
        )

    paths, input_provenance, board, _offset, _source, _report = _publish_fixture(
        tmp_path / "malformed"
    )
    paths.report_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(AlignmentOutcomeError):
        validate_alignment_outcome(
            paths,
            expected_recording=input_provenance.recording,
            expected_input_provenance=input_provenance,
            expected_board_provenance=board,
        )

    paths, input_provenance, board, _offset, _source, _report = _publish_fixture(
        tmp_path / "board"
    )
    board[0].path.write_bytes(b"stale Board bytes")
    current_board = build_board_chunk_provenance((board[0].path,))
    with pytest.raises(AlignmentOutcomeError, match="Board provenance"):
        validate_alignment_outcome(
            paths,
            expected_recording=input_provenance.recording,
            expected_input_provenance=input_provenance,
            expected_board_provenance=current_board,
        )


def test_alignment_outcome_validator_rejects_status_mismatch_and_conflict(
    tmp_path: Path,
) -> None:
    paths, input_provenance, board, _offset, _source, _report = _publish_fixture(tmp_path)
    report = json.loads(paths.report_path.read_text(encoding="utf-8"))
    report["alignment_success"] = False
    paths.report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(AlignmentOutcomeError):
        validate_alignment_outcome(
            paths,
            expected_recording=input_provenance.recording,
            expected_input_provenance=input_provenance,
            expected_board_provenance=board,
        )

    paths, input_provenance, board, skip, report = _skip_fixture(tmp_path / "conflict")
    write_alignment_skip_artifact(skip, paths.skip_json_path)
    report["alignment_outcome_schema_version"] = ALIGNMENT_OUTCOME_SCHEMA_VERSION
    report["alignment_status"] = AlignmentOutcomeStatus.SUCCESS.value
    report["alignment_success"] = True
    report["work_axis_alignment_success"] = True
    report["outcome_artifacts"] = []
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(AlignmentOutcomeError, match="conflicts"):
        validate_alignment_outcome(
            paths,
            expected_recording=input_provenance.recording,
            expected_input_provenance=input_provenance,
            expected_board_provenance=board,
        )


def test_alignment_success_without_canonical_projection_is_completed(
    tmp_path: Path,
) -> None:
    paths, input_provenance, board, _offset, source, report = _outcome_fixture(tmp_path)
    offset = AlignmentOffset(
        user="user_0",
        action="0",
        dataset_id=0,
        ring_stream="ring_0",
        offset_us=123.0,
        alignment_model="constant_offset",
        alignment_success=True,
        event_coverage_ratio=1.0,
        matched_event_count=2,
        total_valid_event_count=2,
        offset_schema_version=2,
        offset_domain="alignment_work_axis",
        canonical_offset_projection_success=False,
        work_axis_offset_us=123.0,
    )
    publish_alignment_success(
        paths,
        offset=offset,
        report=report,
        verification_path=source,
    )
    outcome = validate_alignment_outcome(
        paths,
        expected_recording=input_provenance.recording,
        expected_input_provenance=input_provenance,
        expected_board_provenance=board,
    )
    assert outcome.status is AlignmentOutcomeStatus.SUCCESS


def test_completed_publication_requires_current_feature_and_board_provenance(
    tmp_path: Path,
) -> None:
    paths, input_provenance, board, offset, source, report = _outcome_fixture(tmp_path)
    with pytest.raises(AlignmentOutcomeError, match="input provenance"):
        publish_alignment_success(
            paths,
            offset=offset,
            report={"recording": input_provenance.recording},
            verification_path=source,
        )
    with pytest.raises(AlignmentOutcomeError, match="Board provenance"):
        publish_alignment_success(
            paths,
            offset=offset,
            report={
                "recording": input_provenance.recording,
                "input_provenance": input_provenance.to_dict(),
            },
            verification_path=source,
        )
    skip_paths, _input, _board, skip, skip_report = _skip_fixture(tmp_path / "skip")
    with pytest.raises(AlignmentOutcomeError, match="input provenance"):
        publish_alignment_skip(
            skip_paths,
            skip_artifact=skip,
            report={"recording": skip_report["recording"]},
        )


def test_alignment_success_skip_transition_requires_gate_and_cleans_opposite(
    tmp_path: Path,
) -> None:
    paths, input_provenance, board, offset, source, report = _publish_fixture(tmp_path)
    _skip_paths, _input, _board, skip, skip_report = _skip_fixture(tmp_path / "unused")
    with pytest.raises(AlignmentOutcomeError, match="transition"):
        publish_alignment_skip(
            paths,
            skip_artifact=skip,
            report=skip_report,
        )
    assert paths.offset_txt_path.is_file()
    publish_alignment_skip(
        paths,
        skip_artifact=skip,
        report=skip_report,
        overwrite_outcome=True,
    )
    assert not paths.offset_txt_path.exists()
    assert paths.skip_json_path.is_file()
    validate_alignment_outcome(
        paths,
        expected_recording=input_provenance.recording,
        expected_input_provenance=input_provenance,
        expected_board_provenance=board,
    )

    publish_alignment_success(
        paths,
        offset=offset,
        report=report,
        verification_path=source,
        overwrite_outcome=True,
    )
    assert paths.offset_txt_path.is_file()
    assert not paths.skip_json_path.exists()


def test_failed_to_success_uses_normal_artifact_overwrite_flags(
    tmp_path: Path,
) -> None:
    paths, input_provenance, board, offset, source, report = _outcome_fixture(tmp_path)
    publish_alignment_outcome(
        paths,
        status=AlignmentOutcomeStatus.FAILED,
        report=report,
    )

    paths.offset_txt_path.parent.mkdir(parents=True, exist_ok=True)
    paths.offset_txt_path.write_bytes(b"stale offset")
    paths.verification_png_path.parent.mkdir(parents=True, exist_ok=True)
    paths.verification_png_path.write_bytes(b"stale verification")

    with pytest.raises(AlignmentOutcomeError, match="offset output"):
        publish_alignment_success(
            paths,
            offset=offset,
            report=report,
            verification_path=source,
            overwrite_outcome=True,
            overwrite_report=True,
        )

    # FAILED is report-only and does not require completed-state authorization;
    # each target is still protected by its ordinary overwrite flag.
    publish_alignment_success(
        paths,
        offset=offset,
        report=report,
        verification_path=source,
        overwrite_offset=True,
        overwrite_verification=True,
        overwrite_report=True,
    )
    outcome = validate_alignment_outcome(
        paths,
        expected_recording=input_provenance.recording,
        expected_input_provenance=input_provenance,
        expected_board_provenance=board,
    )
    assert outcome.status is AlignmentOutcomeStatus.SUCCESS


def test_failed_to_skip_uses_skip_and_report_overwrite_flags(
    tmp_path: Path,
) -> None:
    paths, input_provenance, board, skip, report = _skip_fixture(tmp_path)
    publish_alignment_outcome(
        paths,
        status=AlignmentOutcomeStatus.FAILED,
        report=report,
    )
    write_alignment_skip_artifact(skip, paths.skip_json_path)

    with pytest.raises(AlignmentOutcomeError, match="skip output"):
        publish_alignment_skip(
            paths,
            skip_artifact=skip,
            report=report,
            overwrite_outcome=True,
            overwrite_report=True,
        )

    publish_alignment_skip(
        paths,
        skip_artifact=skip,
        report=report,
        overwrite_offset=True,
        overwrite_report=True,
    )
    outcome = validate_alignment_outcome(
        paths,
        expected_recording=input_provenance.recording,
        expected_input_provenance=input_provenance,
        expected_board_provenance=board,
    )
    assert outcome.status is AlignmentOutcomeStatus.SKIPPED


def test_failed_success_cleans_stale_skip_without_outcome_authorization(
    tmp_path: Path,
) -> None:
    paths, input_provenance, board, offset, source, report = _outcome_fixture(tmp_path)
    publish_alignment_outcome(
        paths,
        status=AlignmentOutcomeStatus.FAILED,
        report=report,
    )
    _skip_paths, _skip_input, _skip_board, skip, _skip_report = _skip_fixture(
        tmp_path / "stale-skip"
    )
    write_alignment_skip_artifact(skip, paths.skip_json_path)

    publish_alignment_success(
        paths,
        offset=offset,
        report=report,
        verification_path=source,
        overwrite_offset=True,
        overwrite_verification=True,
        overwrite_report=True,
    )
    assert not paths.skip_json_path.exists()
    validate_alignment_outcome(
        paths,
        expected_recording=input_provenance.recording,
        expected_input_provenance=input_provenance,
        expected_board_provenance=board,
    )


def test_failed_skip_cleans_stale_success_without_outcome_authorization(
    tmp_path: Path,
) -> None:
    paths, input_provenance, board, skip, report = _skip_fixture(tmp_path)
    publish_alignment_outcome(
        paths,
        status=AlignmentOutcomeStatus.FAILED,
        report=report,
    )
    paths.offset_txt_path.parent.mkdir(parents=True, exist_ok=True)
    paths.offset_txt_path.write_bytes(b"stale offset")
    paths.verification_png_path.parent.mkdir(parents=True, exist_ok=True)
    paths.verification_png_path.write_bytes(b"stale verification")

    publish_alignment_skip(
        paths,
        skip_artifact=skip,
        report=report,
        overwrite_offset=True,
        overwrite_report=True,
    )
    assert not paths.offset_txt_path.exists()
    assert not paths.verification_png_path.exists()
    validate_alignment_outcome(
        paths,
        expected_recording=input_provenance.recording,
        expected_input_provenance=input_provenance,
        expected_board_provenance=board,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_global_valid_pair_count", 0),
        ("usable_prefix_valid_pair_count", 1),
        ("next_timestamp_raw", 1_001),
        ("prefix_boundary_position", 0),
        ("last_pre_jump_global_frame_index", 9),
        ("previous_global_frame_index", 9),
        ("next_global_frame_index", 12),
        ("previous_global_frame_index", "10"),
    ],
)
def test_skip_diagnostics_require_strict_coherent_predicates(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    _paths, _input, _board, skip, _report = _skip_fixture(tmp_path)
    payload = skip.to_dict()
    payload["diagnostics"][field] = value  # type: ignore[index]

    with pytest.raises(AlignmentOutcomeError):
        AlignmentSkipArtifact.from_dict(payload)


def test_public_outcome_readers_normalize_missing_and_wrong_manifest_types(
    tmp_path: Path,
) -> None:
    paths, input_provenance, board, _offset, _source, _report = _publish_fixture(
        tmp_path
    )
    report = json.loads(paths.report_path.read_text(encoding="utf-8"))
    report.pop("outcome_artifacts")
    paths.report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(AlignmentOutcomeError):
        validate_alignment_outcome(
            paths,
            expected_recording=input_provenance.recording,
            expected_input_provenance=input_provenance,
            expected_board_provenance=board,
        )
    with pytest.raises(AlignmentOutcomeError):
        read_alignment_outcome_report(paths.report_path)

    report["outcome_artifacts"] = {"filename": "not-a-list"}
    paths.report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(AlignmentOutcomeError):
        validate_alignment_outcome(
            paths,
            expected_recording=input_provenance.recording,
            expected_input_provenance=input_provenance,
            expected_board_provenance=board,
        )


def test_public_skip_reader_normalizes_wrong_json_field_types(
    tmp_path: Path,
) -> None:
    paths, _input, _board, skip, _report = _skip_fixture(tmp_path)
    payload = skip.to_dict()
    payload["diagnostics"]["previous_global_frame_index"] = "10"  # type: ignore[index]
    paths.skip_json_path.parent.mkdir(parents=True, exist_ok=True)
    paths.skip_json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AlignmentOutcomeError):
        read_alignment_skip_artifact(paths.skip_json_path)


def test_public_skip_reader_normalizes_oversized_numeric_diagnostics(
    tmp_path: Path,
) -> None:
    paths, _input, _board, skip, _report = _skip_fixture(tmp_path)
    payload = skip.to_dict()
    payload["diagnostics"]["previous_timestamp_raw"] = 10**1000  # type: ignore[index]
    paths.skip_json_path.parent.mkdir(parents=True, exist_ok=True)
    paths.skip_json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AlignmentOutcomeError):
        read_alignment_skip_artifact(paths.skip_json_path)


def test_alignment_publication_writes_report_last_and_exports_public_api(
    tmp_path: Path,
) -> None:
    import writingring

    assert "AlignmentOutcome" in writingring.__all__
    assert "validate_alignment_outcome" in writingring.__all__
    paths, _input, _board, _offset, source, report = _outcome_fixture(tmp_path)
    destinations: list[Path] = []
    import writingring.alignment_io as alignment_io

    original_replace = alignment_io.os.replace

    def record_replace(source_path: str | bytes | Path, destination: str | bytes | Path) -> None:
        destinations.append(Path(destination))
        original_replace(source_path, destination)

    with patch.object(alignment_io.os, "replace", side_effect=record_replace):
        publish_alignment_success(
            paths,
            offset=_offset,
            report=report,
            verification_path=source,
        )
    assert destinations[-1] == paths.report_path
