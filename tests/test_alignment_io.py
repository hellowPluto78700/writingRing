from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from writingring.alignment_io import (
    AlignmentOffsetProjection,
    AlignmentOffsetExportError,
    alignment_timestamp_metadata,
    apply_board_to_ring_offset,
    build_alignment_time_axes,
    build_alignment_offset_path,
    build_alignment_timestamps,
    extract_alignment_offset,
    project_alignment_offset_to_canonical,
    read_alignment_offset_txt,
    sha256_array,
    write_alignment_offset_txt,
)
from writingring.event_alignment import (
    AlignmentConfig,
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
