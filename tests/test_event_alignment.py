from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from writingring.event_alignment import (
    AlignmentConfig,
    EventAlignmentError,
    InitialIntervalNoUsablePairError,
    PeakDetectionConfig,
    align_events_to_transient_peaks,
    compute_transient_score,
    detect_board_events,
    detect_transient_peak_regions,
    match_shifted_events_to_peaks,
    normalized_peak_region_distance,
    plot_alignment_residuals,
    plot_imu_with_board_events,
    plot_transient_with_event_raster,
    select_board_interval_from_presses,
    smooth_transient_score,
)


def _frames(
    occupied: list[bool],
    *,
    timestamps: list[int] | None = None,
) -> pd.DataFrame:
    if timestamps is None:
        timestamps = [index * 100_000 for index in range(len(occupied))]
    return pd.DataFrame(
        {
            "global_frame_index": np.arange(len(occupied)),
            "frame_timestamp_raw": timestamps,
            "chunk_index": np.zeros(len(occupied), dtype=int),
            "contact_count": np.asarray(occupied, dtype=int),
        }
    )


def _contacts(timestamps: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"frame_timestamp_raw": timestamps, "force": 1.0})


def _contacts_with_global_ids(timestamps: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_frame_index": np.arange(len(timestamps)),
            "frame_timestamp_raw": timestamps,
            "force": 1.0,
        }
    )


def test_board_event_detection_pairs_press_and_lift_in_order() -> None:
    frames = _frames([False, True, True, True, False, False, True, True, True, False])

    result = detect_board_events(frames)

    assert result.events["event_type"].tolist() == ["press", "lift", "press", "lift"]
    assert result.events["event_index"].tolist() == [0, 1, 2, 3]
    assert result.events["paired_touch_index"].tolist() == [0, 0, 1, 1]
    assert result.touch_pairs["duration_frames"].tolist() == [3, 3]
    assert result.touch_pairs["valid_touch"].tolist() == [True, True]


def test_board_event_detection_does_not_invent_initial_press() -> None:
    frames = _frames([True, True, False, False, True, True, True, False])

    result = detect_board_events(frames)

    assert result.events.iloc[0]["event_type"] == "lift"
    assert pd.isna(result.events.iloc[0]["paired_touch_index"])
    assert result.events["event_type"].tolist().count("press") == 1


def test_board_event_detection_preserves_unmatched_final_press() -> None:
    result = detect_board_events(_frames([False, True, True, True]))

    assert result.events["event_type"].tolist() == ["press"]
    assert bool(result.events.iloc[0]["incomplete_touch"])
    assert pd.isna(result.touch_pairs.iloc[0]["lift_event_index"])
    assert not bool(result.touch_pairs.iloc[0]["valid_touch"])


def test_board_event_detection_flags_short_contact() -> None:
    result = detect_board_events(_frames([False, True, True, False]))

    assert result.touch_pairs.iloc[0]["duration_frames"] == 2
    assert bool(result.touch_pairs.iloc[0]["transient"])
    assert not bool(result.touch_pairs.iloc[0]["valid_touch"])


def _many_touch_frames(count: int, *, trailing_frames: int = 40) -> pd.DataFrame:
    occupied = [False]
    for _ in range(count):
        occupied.extend([True, True, True, False, False])
    occupied.extend([False] * trailing_frames)
    return _frames(occupied)


def test_dynamic_interval_uses_twentieth_press_plus_three_seconds() -> None:
    frames = _many_touch_frames(22)
    detection = detect_board_events(frames)
    valid_pairs = detection.touch_pairs.loc[detection.touch_pairs["valid_touch"]]
    twentieth_press = int(valid_pairs.iloc[19]["press_timestamp_raw"])

    selected = select_board_interval_from_presses(
        frames,
        _contacts(frames["frame_timestamp_raw"].tolist()),
        detection,
    )

    assert selected.metadata["desired_board_end_timestamp"] == twentieth_press + 3_000_000
    assert selected.metadata["actual_board_end_timestamp"] == twentieth_press + 3_000_000
    assert not selected.metadata["end_was_clamped"]
    assert selected.metadata["valid_press_count_available"] == 22
    assert selected.metadata["valid_press_count_selected"] >= 20


def test_dynamic_interval_starts_three_seconds_before_first_valid_touch() -> None:
    timestamps = [0, 3_000_000, 3_100_000, 4_900_000, 5_000_000,
                  5_100_000, 5_200_000, 5_300_000, 9_000_000]
    frames = _frames(
        [False, True, False, False, True, True, True, False, False],
        timestamps=timestamps,
    )
    detection = detect_board_events(frames)

    selected = select_board_interval_from_presses(
        frames,
        _contacts(timestamps),
        detection,
        target_valid_press_count=1,
    )

    assert selected.metadata["first_valid_touch_press_timestamp"] == 5_000_000
    assert selected.metadata["desired_board_start_timestamp"] == 2_000_000
    assert selected.metadata["board_start_timestamp"] == 2_000_000
    assert not selected.metadata["start_was_clamped"]
    assert selected.events["event_index"].tolist() == [0, 1, 2, 3]
    assert selected.events.iloc[:2]["transient"].all()


def test_dynamic_interval_clamps_touch_preroll_to_first_frame() -> None:
    timestamps = [0, 1_000_000, 1_100_000, 1_200_000, 1_300_000, 5_000_000]
    frames = _frames(
        [False, True, True, True, False, False],
        timestamps=timestamps,
    )
    detection = detect_board_events(frames)

    selected = select_board_interval_from_presses(
        frames,
        _contacts(timestamps),
        detection,
        target_valid_press_count=1,
    )

    assert selected.metadata["desired_board_start_timestamp"] == -2_000_000
    assert selected.metadata["board_start_timestamp"] == 0
    assert selected.metadata["start_was_clamped"]


def test_transient_score_uses_requested_acceleration_columns_and_smooths() -> None:
    dataframe = pd.DataFrame(
        {
            "linear_acc_x": [0.0, 0.0, 9.0, 0.0, 0.0],
            "linear_acc_y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "linear_acc_z": [0.0, 0.0, 0.0, 0.0, 0.0],
            "gyro_x": [0.0, 100.0, -100.0, 50.0, 0.0],
        }
    )

    score = compute_transient_score(
        dataframe,
        signal_columns=("linear_acc_x", "linear_acc_y", "linear_acc_z"),
    )
    changed_gyro = dataframe.assign(gyro_x=dataframe["gyro_x"] * 1_000.0)
    score_with_changed_gyro = compute_transient_score(
        changed_gyro,
        signal_columns=("linear_acc_x", "linear_acc_y", "linear_acc_z"),
    )
    smoothed = smooth_transient_score(score, window_samples=3)

    np.testing.assert_array_equal(score_with_changed_gyro, score)
    assert smoothed.shape == score.shape
    assert np.isfinite(smoothed).all()
    assert smoothed.max() < score.max()


def test_transient_score_includes_gyroscope_when_requested() -> None:
    dataframe = pd.DataFrame(
        {
            "linear_acc_x": np.zeros(6),
            "linear_acc_y": np.zeros(6),
            "linear_acc_z": np.zeros(6),
            "gyro_x": [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
            "gyro_y": np.zeros(6),
            "gyro_z": np.zeros(6),
        }
    )

    acceleration_only = compute_transient_score(
        dataframe,
        signal_columns=("linear_acc_x", "linear_acc_y", "linear_acc_z"),
    )
    six_axis = compute_transient_score(
        dataframe,
        signal_columns=tuple(dataframe.columns),
    )

    assert np.all(acceleration_only == 0.0)
    assert six_axis.max() > 0.0


def test_dynamic_interval_uses_last_press_and_clamps() -> None:
    frames = _many_touch_frames(3, trailing_frames=1)
    detection = detect_board_events(frames)
    last_press = int(
        detection.touch_pairs.loc[
            detection.touch_pairs["valid_touch"]
        ].iloc[-1]["press_timestamp_raw"]
    )

    selected = select_board_interval_from_presses(
        frames,
        _contacts(frames["frame_timestamp_raw"].tolist()),
        detection,
    )

    assert selected.metadata["desired_board_end_timestamp"] == last_press + 3_000_000
    assert selected.metadata["actual_board_end_timestamp"] == int(
        frames["frame_timestamp_raw"].iloc[-1]
    )
    assert selected.metadata["end_was_clamped"]


def test_dynamic_interval_rejects_no_valid_press() -> None:
    frames = _frames([False, False, False])
    detection = detect_board_events(frames)

    with pytest.raises(EventAlignmentError, match="no valid"):
        select_board_interval_from_presses(
            frames,
            _contacts(frames["frame_timestamp_raw"].tolist()),
            detection,
        )


def test_dynamic_interval_reports_lift_beyond_exact_end() -> None:
    timestamps = [0, 100_000, 200_000, 4_000_000, 4_100_000]
    frames = _frames([False, True, True, True, False], timestamps=timestamps)
    detection = detect_board_events(frames)

    selected = select_board_interval_from_presses(
        frames,
        _contacts(timestamps),
        detection,
    )

    assert selected.metadata["desired_board_end_timestamp"] == 3_100_000
    assert not selected.metadata["boundary_press_lift_in_interval"]
    assert any("no paired lift" in warning for warning in selected.warnings)


def test_dynamic_interval_reports_post_jump_only_pairs_with_structured_error() -> None:
    timestamps = [
        0,
        100_000,
        200_000,
        300_000,
        100_000,
        200_000,
        300_000,
        400_000,
        500_000,
    ]
    frames = _frames(
        [False, True, True, False, False, True, True, True, False],
        timestamps=timestamps,
    )
    detection = detect_board_events(frames)

    with pytest.raises(InitialIntervalNoUsablePairError) as raised:
        select_board_interval_from_presses(
            frames,
            _contacts_with_global_ids(timestamps),
            detection,
        )

    error = raised.value
    assert error.previous_global_frame_index == 3
    assert error.next_global_frame_index == 4
    assert error.previous_timestamp_raw == 300_000
    assert error.next_timestamp_raw == 100_000
    assert error.prefix_boundary_position == 4
    assert error.last_pre_jump_global_frame_index == 3
    assert error.total_global_valid_pair_count == 1
    assert error.usable_prefix_valid_pair_count == 0
    assert error.diagnostics["prefix_boundary_position"] == 4


def test_dynamic_interval_rejects_cross_boundary_pair_as_unusable() -> None:
    timestamps = [0, 100_000, 200_000, 300_000, 100_000, 200_000]
    frames = _frames(
        [False, True, True, True, True, False],
        timestamps=timestamps,
    )
    detection = detect_board_events(frames)

    with pytest.raises(InitialIntervalNoUsablePairError):
        select_board_interval_from_presses(
            frames,
            _contacts_with_global_ids(timestamps),
            detection,
        )


def test_dynamic_interval_prefixes_mixed_pairs_and_identity_contacts() -> None:
    timestamps = [
        0,
        1_000_000,
        1_100_000,
        1_200_000,
        1_300_000,
        100_000,
        200_000,
        300_000,
        400_000,
        500_000,
    ]
    frames = _frames(
        [False, True, True, True, False, True, True, True, False, False],
        timestamps=timestamps,
    )
    detection = detect_board_events(frames)

    selected = select_board_interval_from_presses(
        frames,
        _contacts_with_global_ids(timestamps),
        detection,
        target_valid_press_count=1,
    )

    assert selected.frames["global_frame_index"].tolist() == [0, 1, 2, 3, 4]
    assert selected.contacts["global_frame_index"].tolist() == [0, 1, 2, 3, 4]
    assert selected.events["global_frame_index"].tolist() == [1, 4]
    assert selected.touch_pairs["paired_touch_index"].tolist() == [0]
    assert selected.metadata["valid_press_count_available"] == 1


def test_dynamic_interval_excludes_cross_boundary_pair_events_from_alignment() -> None:
    timestamps = [
        0,
        1_000_000,
        1_100_000,
        1_200_000,
        1_300_000,
        1_400_000,
        100_000,
        200_000,
        300_000,
        400_000,
    ]
    frames = _frames(
        [False, True, True, True, False, True, True, True, True, False],
        timestamps=timestamps,
    )
    detection = detect_board_events(frames)

    selected = select_board_interval_from_presses(
        frames,
        _contacts_with_global_ids(timestamps),
        detection,
        target_valid_press_count=1,
    )

    assert selected.touch_pairs["paired_touch_index"].tolist() == [0]
    selected_event_ids = set(selected.events["event_index"].tolist())
    crossboundary_pair = detection.touch_pairs.loc[
        detection.touch_pairs["paired_touch_index"] == 1
    ].iloc[0]
    crossboundary_event_ids = {
        int(crossboundary_pair["press_event_index"]),
        int(crossboundary_pair["lift_event_index"]),
    }
    assert crossboundary_event_ids.isdisjoint(selected_event_ids)
    matching_valid_events = selected.events.loc[
        selected.events["valid_touch"].astype(bool)
    ]
    assert matching_valid_events["paired_touch_index"].tolist() == [0, 0]
    assert crossboundary_event_ids.isdisjoint(
        set(matching_valid_events["event_index"].tolist())
    )


def test_peak_detection_returns_ordered_region() -> None:
    score = np.zeros(101)
    score[48:53] = [1.0, 3.0, 8.0, 3.0, 1.0]
    timestamps = np.arange(101, dtype=float) * 5_000.0

    peaks = detect_transient_peak_regions(
        score,
        timestamps,
        config=PeakDetectionConfig(
            smoothing_window_samples=1,
            prominence_window_samples=20,
            prominence_mad_multiplier=0.0,
            minimum_prominence=1.0,
            boundary_prominence_fraction=0.75,
            merge_gap_samples=0,
        ),
    )

    peak = peaks.iloc[0]
    assert peak["left_index"] < peak["peak_index"] < peak["right_index"]
    assert peak["left_timestamp"] <= peak["peak_timestamp"] <= peak["right_timestamp"]
    assert peak["prominence"] > 0.0


def test_peak_detection_merges_near_duplicate_peaks() -> None:
    score = np.zeros(60)
    score[20] = 8.0
    score[23] = 7.0
    timestamps = np.arange(60, dtype=float) * 5_000.0

    peaks = detect_transient_peak_regions(
        score,
        timestamps,
        config=PeakDetectionConfig(
            smoothing_window_samples=1,
            prominence_window_samples=10,
            prominence_mad_multiplier=0.0,
            minimum_prominence=1.0,
            boundary_prominence_fraction=0.5,
            merge_gap_samples=3,
        ),
    )

    assert len(peaks) == 1
    assert peaks.iloc[0]["peak_index"] == 20


def test_normalized_distance_covers_region_and_prefers_center() -> None:
    center = normalized_peak_region_distance(
        100.0,
        peak_timestamp=100.0,
        left_timestamp=80.0,
        right_timestamp=140.0,
    )
    boundary = normalized_peak_region_distance(
        80.0,
        peak_timestamp=100.0,
        left_timestamp=80.0,
        right_timestamp=140.0,
    )
    outside = normalized_peak_region_distance(
        70.0,
        peak_timestamp=100.0,
        left_timestamp=80.0,
        right_timestamp=140.0,
    )

    assert center == 0.0
    assert boundary == 1.0
    assert outside > 1.0


def _manual_peaks(
    centers: list[float],
    *,
    half_width: float = 25_000.0,
    prominences: list[float] | None = None,
) -> pd.DataFrame:
    if prominences is None:
        prominences = [5.0] * len(centers)
    return pd.DataFrame(
        {
            "peak_index": np.arange(len(centers)),
            "peak_timestamp": centers,
            "peak_elapsed_s": np.asarray(centers) / 1_000_000.0,
            "left_index": np.arange(len(centers)),
            "right_index": np.arange(len(centers)),
            "left_timestamp": np.asarray(centers) - half_width,
            "right_timestamp": np.asarray(centers) + half_width,
            "width_s": 2.0 * half_width / 1_000_000.0,
            "height": np.asarray(prominences) * 10.0,
            "prominence": prominences,
        }
    )


def _two_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = _frames(
        [False, True, True, True, False, False, True, True, True, False],
        timestamps=[0, 1_000_000, 1_100_000, 1_200_000, 2_000_000,
                    2_500_000, 3_000_000, 3_100_000, 3_200_000, 4_000_000],
    )
    detection = detect_board_events(frames)
    return detection.events, detection.touch_pairs


def test_sequence_matching_is_monotonic_unique_and_allows_unmatched() -> None:
    events, _ = _two_pairs()
    peaks = _manual_peaks([1_500_000.0, 2_500_000.0, 4_500_000.0])

    matches = match_shifted_events_to_peaks(events, peaks, offset_us=500_000.0)
    matched = matches.loc[matches["matched"]]

    assert matched["matched_peak_index"].is_unique
    assert matched["event_index"].is_unique
    assert matched["matched_peak_timestamp"].is_monotonic_increasing
    assert set(matched["event_type"]) == {"press", "lift"}
    assert len(matched) == 3
    assert len(matches) == 4


def test_pair_aware_matching_prefers_complete_pair_before_center_distance() -> None:
    events, _ = _two_pairs()
    peaks = pd.DataFrame(
        {
            "peak_index": [0, 1],
            "peak_timestamp": [2_000_000.0, 3_000_000.0],
            "peak_elapsed_s": [0.0, 1.0],
            "left_index": [0, 1],
            "right_index": [0, 1],
            "left_timestamp": [900_000.0, 1_900_000.0],
            "right_timestamp": [2_100_000.0, 3_100_000.0],
            "width_s": [1.2, 1.2],
            "height": [1.0, 1.0],
            "prominence": [1.0, 1.0],
        }
    )

    matches = match_shifted_events_to_peaks(events, peaks, offset_us=0.0)
    matched = matches.loc[matches["matched"]]

    assert matched["event_index"].tolist() == [0, 1]
    assert matched["paired_touch_index"].nunique() == 1


def test_matching_coverage_dominates_peak_center_distance() -> None:
    events, _ = _two_pairs()
    peaks = pd.DataFrame(
        {
            "peak_index": [0, 1],
            "peak_timestamp": [1_900_000.0, 3_100_000.0],
            "peak_elapsed_s": [0.0, 1.2],
            "left_index": [0, 1],
            "right_index": [0, 1],
            "left_timestamp": [1_000_000.0, 2_000_000.0],
            "right_timestamp": [2_000_000.0, 3_200_000.0],
            "width_s": [1.0, 1.2],
            "height": [1.0, 1.0],
            "prominence": [1.0, 1.0],
        }
    )

    matches = match_shifted_events_to_peaks(events, peaks, offset_us=0.0)

    assert matches["matched"].sum() == 2
    assert matches.loc[matches["matched"], "event_index"].tolist() == [0, 1]


def test_matching_center_distance_dominates_peak_prominence() -> None:
    events, _ = _two_pairs()
    press = events.iloc[[0]].copy()
    peaks = pd.DataFrame(
        {
            "peak_index": [0, 1],
            "peak_timestamp": [1_000_000.0, 1_050_000.0],
            "peak_elapsed_s": [0.0, 0.05],
            "left_index": [0, 1],
            "right_index": [0, 1],
            "left_timestamp": [900_000.0, 900_000.0],
            "right_timestamp": [1_100_000.0, 1_100_000.0],
            "width_s": [0.2, 0.2],
            "height": [1.0, 1_000_000.0],
            "prominence": [1.0, 1_000_000.0],
        }
    )

    matches = match_shifted_events_to_peaks(press, peaks, offset_us=0.0)

    assert matches.iloc[0]["matched_peak_index"] == 0
    assert matches.iloc[0]["normalized_peak_distance"] == 0.0


def test_candidate_ranking_prefers_complete_pair_when_event_count_ties() -> None:
    events, pairs = _two_pairs()
    # +0.5 s matches the first complete pair; +4.0 s matches two presses.
    peaks = _manual_peaks([1_500_000.0, 2_500_000.0, 5_000_000.0, 7_000_000.0])

    result = align_events_to_transient_peaks(
        events,
        pairs,
        peaks,
        ring_timestamp_range=(0.0, 8_000_000.0),
        config=AlignmentConfig(
            offset_search_range_us=(0.0, 5_000_000.0),
            offset_cluster_width_us=10_000.0,
            minimum_event_coverage_ratio=0.25,
            minimum_valid_touch_pairs=1,
        ),
    )

    assert result.best_offset_us == pytest.approx(500_000.0)
    assert result.report["fully_matched_touch_pair_count"] >= 1
    assert result.report["pair_aware_dynamic_programming"]
    assert result.report["matching_objective"][:2] == [
        "maximize matched event count",
        "maximize fully matched press/lift touch-pair count",
    ]


def test_press_only_alignment_disables_pair_and_lift_coverage_objectives() -> None:
    events, pairs = _two_pairs()
    presses = events.loc[events["event_type"] == "press"].copy()
    peaks = _manual_peaks([1_500_000.0, 3_500_000.0])

    result = align_events_to_transient_peaks(
        presses,
        pairs,
        peaks,
        ring_timestamp_range=(0.0, 5_000_000.0),
        config=AlignmentConfig(
            offset_search_range_us=(0.0, 1_000_000.0),
            offset_cluster_width_us=10_000.0,
            minimum_event_coverage_ratio=0.5,
            minimum_valid_touch_pairs=1,
        ),
    )

    assert result.success
    assert result.report["matched_event_types"] == ["press"]
    assert not result.report["pair_completeness_objective_enabled"]
    assert result.report["lift_coverage_ratio"] is None
    assert result.report["touch_pair_coverage_ratio"] is None
    assert result.report["matching_objective"][1].startswith("minimize normalized")


def test_offset_recovery_is_stable_to_amplitude_and_extra_peaks() -> None:
    events, pairs = _two_pairs()
    true_centers = [1_500_000.0, 2_500_000.0, 3_500_000.0, 4_500_000.0]
    peaks = _manual_peaks(
        [700_000.0, *true_centers, 5_700_000.0],
        prominences=[1000.0, 1.0, 80.0, 2.0, 300.0, 900.0],
    )

    result = align_events_to_transient_peaks(
        events,
        pairs,
        peaks,
        ring_timestamp_range=(0.0, 6_000_000.0),
        config=AlignmentConfig(
            offset_search_range_us=(-1_000_000.0, 2_000_000.0),
            offset_cluster_width_us=20_000.0,
            minimum_event_coverage_ratio=0.75,
            minimum_valid_touch_pairs=1,
        ),
    )

    assert result.success
    assert result.best_offset_us == pytest.approx(500_000.0, abs=5_000.0)
    assert result.report["matched_event_count"] == 4


def test_alignment_reports_ambiguity_for_equal_sequences() -> None:
    events, pairs = _two_pairs()
    base = [1_500_000.0, 2_500_000.0, 3_500_000.0, 4_500_000.0]
    peaks = _manual_peaks([*base, *[value + 5_000_000.0 for value in base]])

    result = align_events_to_transient_peaks(
        events,
        pairs,
        peaks,
        ring_timestamp_range=(0.0, 10_000_000.0),
        config=AlignmentConfig(
            offset_search_range_us=(0.0, 6_000_000.0),
            offset_cluster_width_us=10_000.0,
            minimum_event_coverage_ratio=0.5,
            minimum_valid_touch_pairs=1,
        ),
    )

    assert result.report["best_vs_second_best_nearly_tied"]
    assert any("nearly tied" in warning for warning in result.warnings)


def test_low_coverage_returns_failed_alignment_without_forcing_all_events() -> None:
    events, pairs = _two_pairs()
    peaks = _manual_peaks([1_500_000.0])

    result = align_events_to_transient_peaks(
        events,
        pairs,
        peaks,
        ring_timestamp_range=(0.0, 5_000_000.0),
        config=AlignmentConfig(
            offset_search_range_us=(0.0, 1_000_000.0),
            minimum_event_coverage_ratio=0.75,
            minimum_valid_touch_pairs=1,
        ),
    )

    assert not result.success
    assert result.report["matched_event_count"] == 1
    assert result.event_matches["matched"].sum() == 1


def test_residual_trend_is_reported_without_affine_correction() -> None:
    events, pairs = _two_pairs()
    peaks = _manual_peaks(
        [1_500_000.0, 2_510_000.0, 3_520_000.0, 4_530_000.0],
        half_width=50_000.0,
    )

    result = align_events_to_transient_peaks(
        events,
        pairs,
        peaks,
        ring_timestamp_range=(0.0, 5_000_000.0),
        config=AlignmentConfig(
            offset_search_range_us=(0.0, 1_000_000.0),
            offset_cluster_width_us=40_000.0,
            minimum_event_coverage_ratio=0.75,
            minimum_valid_touch_pairs=1,
            residual_trend_warning_ms=5.0,
        ),
    )

    assert not result.report["affine_clock_correction_applied"]
    assert result.report["residual_trend_change_ms"] > 5.0
    assert any("clock drift" in warning for warning in result.warnings)


def test_before_after_plotting_uses_ring_start_and_distinguishes_events(
    tmp_path: Path,
) -> None:
    timestamps = 10_000_000.0 + np.arange(100) * 10_000.0
    signals = np.column_stack([np.sin(np.arange(100) / 10.0 + i) for i in range(6)])
    ring_dataframe = pd.DataFrame(
        signals,
        columns=("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"),
    )
    score = np.linalg.norm(signals, axis=1)
    events = pd.DataFrame(
        {
            "event_type": ["press", "lift"],
            "frame_timestamp_raw": [9_900_000, 10_200_000],
        }
    )
    imu_path = tmp_path / "before_imu.png"
    raster_path = tmp_path / "before_raster.png"

    imu, visibility = plot_imu_with_board_events(
        ring_dataframe,
        timestamps,
        score,
        events,
        signal_columns=ring_dataframe.columns,
        duration_s=0.5,
        title="Before",
        output_path=imu_path,
    )
    raster, raster_visibility = plot_transient_with_event_raster(
        timestamps,
        score,
        events,
        duration_s=0.5,
        title="Before",
        output_path=raster_path,
    )

    assert len(imu.axes) == 7
    assert visibility.press_outside == 1
    assert visibility.lift_inside == 1
    assert raster_visibility == visibility
    assert len(raster.axes) == 2
    assert raster.axes[1].collections
    assert imu_path.is_file() and imu_path.stat().st_size > 0
    assert raster_path.is_file() and raster_path.stat().st_size > 0
    plt.close(imu)
    plt.close(raster)


def test_residual_plot_is_created(tmp_path: Path) -> None:
    events, _ = _two_pairs()
    peaks = _manual_peaks([1_500_000.0, 2_500_000.0, 3_500_000.0, 4_500_000.0])
    matches = match_shifted_events_to_peaks(events, peaks, offset_us=500_000.0)
    output = tmp_path / "residuals.png"

    figure = plot_alignment_residuals(matches, output_path=output)

    assert output.is_file() and output.stat().st_size > 0
    assert len(figure.axes[0].collections) == 2
    plt.close(figure)
