from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from writingring.alignment import (
    AlignmentAnalysisError,
    build_group_candidate_windows,
    estimate_consensus_shift,
    nearest_sorted_index,
    plot_group_candidate_heatmap,
    plot_group_candidate_overlay,
    robust_normalize_rows,
)


def test_nearest_sorted_index_exact_and_midpoint() -> None:
    timestamps = np.asarray([10.0, 20.0, 30.0])

    assert nearest_sorted_index(timestamps, 20.0) == 1
    assert nearest_sorted_index(timestamps, 25.0) == 1
    assert nearest_sorted_index(timestamps, 26.0) == 2


def test_nearest_sorted_index_clamps_outside_range() -> None:
    timestamps = np.asarray([10.0, 20.0, 30.0])

    assert nearest_sorted_index(timestamps, -100.0) == 0
    assert nearest_sorted_index(timestamps, 100.0) == 2


@pytest.mark.parametrize(
    "timestamps, message",
    [
        (np.asarray([]), "nonempty"),
        (np.asarray([1.0, 1.0]), "strictly increasing"),
        (np.asarray([1.0, 0.0]), "strictly increasing"),
        (np.asarray([1.0, np.nan]), "finite"),
    ],
)
def test_nearest_sorted_index_rejects_invalid_timestamps(
    timestamps: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(AlignmentAnalysisError, match=message):
        nearest_sorted_index(timestamps, 1.0)


def _group_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    timestamps = np.asarray(
        [0, 800_000, 1_900_000, 2_100_000, 3_050_000, 4_000_000],
        dtype=np.float64,
    )
    base = timestamps / 1_000_000.0
    ring_dataframe = pd.DataFrame(
        {
            "acc_x": base,
            "acc_y": base + 1.0,
            "acc_z": base + 2.0,
            "gyr_x": base + 3.0,
            "gyr_y": base + 4.0,
            "gyr_z": base + 5.0,
        }
    )
    transient_score = base * 2.0
    press_events = pd.DataFrame(
        {
            "event_index": [4, 9, 12],
            "frame_timestamp_raw": [2_000_000, 3_000_000, 10_000_000],
        }
    )
    return press_events, ring_dataframe, timestamps, transient_score


def _build_group():
    press_events, ring_dataframe, timestamps, transient_score = _group_inputs()
    return build_group_candidate_windows(
        press_events,
        ring_dataframe,
        timestamps,
        transient_score,
        signal_columns=("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"),
        coarse_offset_us=0.0,
        radius_samples=1,
        imu_sampling_rate_hz=1.0,
    )


def test_group_windows_interpolate_irregular_timestamps_and_report_metadata() -> None:
    result = _build_group()

    np.testing.assert_allclose(result.relative_time_s, [-1.0, 0.0, 1.0])
    assert result.event_count == 2
    assert result.time_count == 3
    assert result.transient_score_matrix.shape == (2, 3)
    assert all(matrix.shape == (2, 3) for matrix in result.signal_matrices.values())
    np.testing.assert_allclose(result.signal_matrices["acc_x"][:, 1], [2.0, 3.0])
    assert np.isnan(result.signal_matrices["acc_x"][0, [0, 2]]).all()
    assert np.isnan(result.signal_matrices["acc_x"][1, 0])
    assert result.signal_matrices["acc_x"][1, 2] == pytest.approx(4.0)

    assert result.metadata["board_event_index"].tolist() == [4, 9]
    assert result.metadata["center_imu_sample_index"].tolist() == [2, 4]
    assert result.metadata["window_start_sample_index"].tolist() == [2, 3]
    assert result.metadata["window_stop_sample_index_exclusive"].tolist() == [4, 6]
    np.testing.assert_allclose(
        result.metadata["nearest_timestamp_error_us"],
        [-100_000.0, 50_000.0],
    )
    assert result.skipped_events["board_event_index"].tolist() == [12]
    assert "fewer than two" in result.skipped_events.iloc[0]["reason"]


def test_group_windows_reject_empty_or_nonmonotonic_inputs() -> None:
    _, ring_dataframe, timestamps, transient_score = _group_inputs()
    empty_events = pd.DataFrame(columns=("event_index", "frame_timestamp_raw"))
    with pytest.raises(AlignmentAnalysisError, match="at least one event"):
        build_group_candidate_windows(
            empty_events,
            ring_dataframe,
            timestamps,
            transient_score,
            signal_columns=("acc_x",),
            coarse_offset_us=0.0,
            radius_samples=1,
            imu_sampling_rate_hz=1.0,
        )

    events = pd.DataFrame(
        {"event_index": [0], "frame_timestamp_raw": [2_000_000]}
    )
    invalid_timestamps = timestamps.copy()
    invalid_timestamps[3] = invalid_timestamps[2]
    with pytest.raises(AlignmentAnalysisError, match="strictly increasing"):
        build_group_candidate_windows(
            events,
            ring_dataframe,
            invalid_timestamps,
            transient_score,
            signal_columns=("acc_x",),
            coarse_offset_us=0.0,
            radius_samples=1,
            imu_sampling_rate_hz=1.0,
        )


def test_robust_normalize_rows_handles_varying_constant_and_nan_rows() -> None:
    values = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
            [np.nan, 2.0, 4.0],
            [np.nan, np.nan, np.nan],
        ]
    )

    normalized = robust_normalize_rows(values)

    assert np.median(normalized[0]) == pytest.approx(0.0)
    np.testing.assert_array_equal(normalized[1], np.zeros(3))
    assert np.isnan(normalized[2, 0])
    assert np.median(normalized[2, 1:]) == pytest.approx(0.0)
    assert np.isnan(normalized[3]).all()
    assert not np.shares_memory(normalized, values)


def test_consensus_shift_recovers_common_feature_despite_outlier() -> None:
    relative_time_s = np.linspace(-0.5, 0.5, 101)
    background = 0.10 * np.sin(2.0 * np.pi * 3.0 * relative_time_s)
    common = background + 3.0 * np.exp(-((relative_time_s - 0.2) / 0.025) ** 2)
    rows = np.stack(
        [
            common,
            2.0 * common + 3.0,
            0.5 * common - 2.0,
            1.5 * common + 0.2,
            background
            + 20.0 * np.exp(-((relative_time_s + 0.3) / 0.02) ** 2),
        ]
    )

    result = estimate_consensus_shift(
        relative_time_s,
        rows,
        coarse_offset_us=1_000.0,
        search_range_s=(-0.30, 0.30),
    )

    assert result.consensus_shift_s == pytest.approx(0.2)
    assert result.candidate_refined_offset_us == pytest.approx(201_000.0)
    assert result.contributing_event_count == 5


def test_consensus_shift_enforces_search_range() -> None:
    relative_time_s = np.linspace(-0.5, 0.5, 101)
    common = (
        0.10 * np.sin(2.0 * np.pi * 3.0 * relative_time_s)
        + 3.0 * np.exp(-((relative_time_s - 0.2) / 0.025) ** 2)
    )
    rows = np.stack([common, common, common])

    result = estimate_consensus_shift(
        relative_time_s,
        rows,
        coarse_offset_us=0.0,
        search_range_s=(-0.10, 0.10),
    )

    assert -0.10 <= result.consensus_shift_s <= 0.10
    assert result.consensus_shift_s != pytest.approx(0.2)


def test_group_plots_have_expected_panels_and_nonempty_outputs(
    tmp_path: Path,
) -> None:
    result = _build_group()
    overlay_path = tmp_path / "group" / "overlay.png"
    heatmap_path = tmp_path / "group" / "heatmap.png"

    overlay = plot_group_candidate_overlay(
        result,
        normalized=True,
        output_path=overlay_path,
        show=False,
    )
    heatmap = plot_group_candidate_heatmap(
        result,
        output_path=heatmap_path,
        show=False,
    )

    assert len(overlay.axes) == 7
    assert all(axis.lines for axis in overlay.axes)
    assert all(axis.collections for axis in overlay.axes)
    assert len(heatmap.axes) == 2
    assert heatmap.axes[0].get_yticklabels()[0].get_text() == "4"
    assert overlay_path.is_file() and overlay_path.stat().st_size > 0
    assert heatmap_path.is_file() and heatmap_path.stat().st_size > 0
    plt.close(overlay)
    plt.close(heatmap)
