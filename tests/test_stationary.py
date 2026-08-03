from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from writingring.gravity import (
    GravityCalibrationError,
    GravityRemovalConfig,
    process_ring_gravity,
)
from writingring.ring_loader import load_ring
from writingring.stationary import (
    EXPECTED_GRAVITY_M_S2,
    StationarySearchConfig,
    StationarySearchError,
    find_stationary_interval,
)


def _imu(
    sample_count: int,
    *,
    gravity: np.ndarray | None = None,
    gyro: np.ndarray | None = None,
) -> np.ndarray:
    acceleration = np.tile(
        np.array([0.0, 0.0, EXPECTED_GRAVITY_M_S2])
        if gravity is None
        else gravity,
        (sample_count, 1),
    )
    gyroscope = np.tile(
        np.zeros(3) if gyro is None else gyro,
        (sample_count, 1),
    )
    return np.column_stack((acceleration, gyroscope))


@pytest.mark.parametrize("start", [0, 200, 400])
def test_finds_valid_stationary_interval_at_recording_positions(start: int) -> None:
    values = _imu(600, gyro=np.array([0.2, 0.0, 0.0]))
    values[start : start + 200, 3:] = 0.0

    result = find_stationary_interval(values)

    assert result.passed
    assert result.start_index == start
    assert result.stop_index == start + 20
    assert result.duration_s == pytest.approx(0.10)
    assert result.valid_window_count >= 1


def test_tilted_stationary_sensor_uses_body_frame_gravity_direction() -> None:
    gravity = np.array([3.0, -4.0, np.sqrt(EXPECTED_GRAVITY_M_S2**2 - 25.0)])
    result = find_stationary_interval(_imu(200, gravity=gravity))

    assert result.passed
    assert result.estimated_gravity_m_s2 == pytest.approx(EXPECTED_GRAVITY_M_S2)
    np.testing.assert_allclose(
        result.gravity_direction_body,
        gravity / np.linalg.norm(gravity),
    )
    np.testing.assert_allclose(result.estimated_gyro_bias_rad_s, 0.0)


def test_constant_rotation_is_rejected() -> None:
    result = find_stationary_interval(
        _imu(200, gyro=np.array([0.2, 0.0, 0.0]))
    )

    assert not result.passed
    assert "gyro_median_rad_s" in result.failed_checks
    assert "gyro_p95_rad_s" in result.failed_checks
    assert result.valid_window_count == 0


def test_no_valid_interval_returns_lowest_score_candidate() -> None:
    values = _imu(400, gyro=np.array([0.2, 0.0, 0.0]))
    values[100:300, :3] += [0.8, 0.0, 0.0]

    result = find_stationary_interval(values)

    assert not result.passed
    assert result.evaluated_window_count == 39
    assert result.score >= 0.0
    assert result.failed_checks


def test_score_first_search_can_choose_best_nonpassing_window() -> None:
    values = _imu(40)
    values[:20, 3] = 0.055
    values[20:, 2] = EXPECTED_GRAVITY_M_S2 + 0.4
    values[20:, 3] = 0.04
    base = StationarySearchConfig(
        stationary_duration_s=0.1,
        stride_duration_s=0.1,
    )

    passing_first = find_stationary_interval(values, config=base)
    score_first = find_stationary_interval(
        values,
        config=StationarySearchConfig(
            stationary_duration_s=0.1,
            stride_duration_s=0.1,
            prefer_passing_window=False,
        ),
    )

    assert passing_first.start_index == 20
    assert passing_first.passed
    assert score_first.start_index == 0
    assert not score_first.passed
    assert score_first.score < passing_first.score


@pytest.mark.parametrize(
    "values, message",
    [
        (np.zeros((200, 5)), "shape"),
        (np.full((200, 6), np.nan), "finite"),
        (np.zeros((19, 6)), "fewer than"),
    ],
)
def test_search_rejects_malformed_or_short_input(
    values: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(StationarySearchError, match=message):
        find_stationary_interval(values)


def test_manual_calibration_overrides_automatic_search(tmp_path: Path) -> None:
    imu = _imu(250)
    timestamps = 1_000_000.0 + np.arange(250) * 5_000.0
    rows = np.column_stack((imu, timestamps))
    path = tmp_path / "0_ring_0.bin"
    rows.astype(np.float64).tofile(path)
    ring = load_ring(path)
    config = GravityRemovalConfig(
        sampling_rate_hz=200.0,
        calibration_start_sample=20,
        calibration_stop_sample=220,
    )

    result = process_ring_gravity(
        ring,
        config=config,
        stationary_search_config=StationarySearchConfig(
            expected_gravity_m_s2=1.0,
        ),
    )

    assert result.calibration.start_sample == 20
    assert result.calibration.stop_sample == 220
    assert result.calibration.stationary_search is None


def test_automatic_calibration_uses_candidate_and_strict_failure_is_typed(
    tmp_path: Path,
) -> None:
    imu = _imu(400, gyro=np.array([0.2, 0.0, 0.0]))
    imu[100:300, 3:] = 0.0
    timestamps = 1_000_000.0 + np.arange(400) * 5_000.0
    path = tmp_path / "0_ring_0.bin"
    np.column_stack((imu, timestamps)).astype(np.float64).tofile(path)
    ring = load_ring(path)

    result = process_ring_gravity(ring, config=GravityRemovalConfig())
    assert result.calibration.stationary_search is not None
    assert result.calibration.stationary_search.passed
    assert result.calibration.start_sample == 100

    rotating = _imu(400, gyro=np.array([0.2, 0.0, 0.0]))
    rotating_path = tmp_path / "1_ring_0.bin"
    np.column_stack((rotating, timestamps)).astype(np.float64).tofile(rotating_path)
    rotating_ring = load_ring(rotating_path)
    with pytest.raises(GravityCalibrationError, match="automatic stationary"):
        process_ring_gravity(rotating_ring, config=GravityRemovalConfig())
