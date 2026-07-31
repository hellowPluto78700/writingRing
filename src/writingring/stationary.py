"""Pure NumPy stationary-interval detection for six-axis IMU recordings.

The detector expects acceleration in m/s^2 and gyroscope values in rad/s.
It evaluates fixed-duration windows over the entire recording and returns the
lowest-scoring valid window, or the lowest-scoring candidate when none pass.
Finding a candidate is evidence-based processing assistance; it does not
prove physical stationarity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np


EXPECTED_GRAVITY_M_S2: Final[float] = 9.80665
_IMU_COLUMN_COUNT: Final[int] = 6
_VECTOR_SIZE: Final[int] = 3
_MAD_SCALE: Final[float] = 1.4826


class StationarySearchError(ValueError):
    """Raised when stationary-interval search inputs are invalid."""


@dataclass(frozen=True, slots=True)
class StationarySearchConfig:
    """Thresholds and sampling assumptions for stationary-window search."""

    sampling_rate_hz: float = 200.0
    stationary_duration_s: float = 0.10
    stride_duration_s: float = 0.05
    expected_gravity_m_s2: float = EXPECTED_GRAVITY_M_S2
    max_gravity_error_m_s2: float = 0.50
    max_acc_norm_robust_std_m_s2: float = 0.15
    max_acc_axis_robust_std_m_s2: float = 0.25
    max_gyro_median_rad_s: float = 0.05
    max_gyro_p95_rad_s: float = 0.10
    max_gyro_robust_std_rad_s: float = 0.02


@dataclass(frozen=True, slots=True)
class StationarySearchResult:
    """Metrics and selected interval from a complete stationary search."""

    start_index: int
    stop_index: int
    duration_s: float
    passed: bool
    score: float
    estimated_gravity_m_s2: float
    gravity_direction_body: tuple[float, float, float]
    estimated_gyro_bias_rad_s: tuple[float, float, float]
    gravity_error_m_s2: float
    acc_norm_robust_std_m_s2: float
    acc_axis_robust_std_m_s2: float
    gyro_median_rad_s: float
    gyro_p95_rad_s: float
    gyro_robust_std_rad_s: float
    failed_checks: tuple[str, ...]
    evaluated_window_count: int
    valid_window_count: int


def find_stationary_interval(
    imu: np.ndarray,
    *,
    config: StationarySearchConfig = StationarySearchConfig(),
) -> StationarySearchResult:
    """Find the best fixed-duration stationary candidate in an ``(N, 6)`` IMU.

    The six columns must be acceleration x/y/z in m/s^2 followed by gyroscope
    x/y/z in rad/s. Every complete stride-aligned window is evaluated, along
    with a final end-aligned window when the stride would otherwise omit it.
    """

    values = _validate_imu(imu)
    validated_config = _validate_config(config)
    window_size = _sample_count(
        validated_config.stationary_duration_s,
        sampling_rate_hz=validated_config.sampling_rate_hz,
        name="stationary_duration_s",
    )
    stride_size = _sample_count(
        validated_config.stride_duration_s,
        sampling_rate_hz=validated_config.sampling_rate_hz,
        name="stride_duration_s",
    )
    if values.shape[0] < window_size:
        raise StationarySearchError(
            f"IMU has {values.shape[0]} sample(s), fewer than one "
            f"{window_size}-sample stationary window"
        )

    starts = list(range(0, values.shape[0] - window_size + 1, stride_size))
    final_start = values.shape[0] - window_size
    if starts[-1] != final_start:
        starts.append(final_start)

    candidates = tuple(
        _evaluate_window(
            values[start : start + window_size],
            start_index=start,
            stop_index=start + window_size,
            config=validated_config,
        )
        for start in starts
    )
    valid_candidates = tuple(candidate for candidate in candidates if candidate.passed)
    selected = min(
        valid_candidates if valid_candidates else candidates,
        key=lambda candidate: (candidate.score, candidate.start_index),
    )
    return StationarySearchResult(
        **{
            field: getattr(selected, field)
            for field in _RESULT_FIELDS
        },
        evaluated_window_count=len(candidates),
        valid_window_count=len(valid_candidates),
    )


@dataclass(frozen=True, slots=True)
class _StationaryCandidate:
    start_index: int
    stop_index: int
    duration_s: float
    passed: bool
    score: float
    estimated_gravity_m_s2: float
    gravity_direction_body: tuple[float, float, float]
    estimated_gyro_bias_rad_s: tuple[float, float, float]
    gravity_error_m_s2: float
    acc_norm_robust_std_m_s2: float
    acc_axis_robust_std_m_s2: float
    gyro_median_rad_s: float
    gyro_p95_rad_s: float
    gyro_robust_std_rad_s: float
    failed_checks: tuple[str, ...]


_RESULT_FIELDS: Final[tuple[str, ...]] = tuple(
    field
    for field in StationarySearchResult.__dataclass_fields__
    if field not in {"evaluated_window_count", "valid_window_count"}
)


def _evaluate_window(
    window: np.ndarray,
    *,
    start_index: int,
    stop_index: int,
    config: StationarySearchConfig,
) -> _StationaryCandidate:
    acceleration = window[:, :_VECTOR_SIZE]
    gyroscope = window[:, _VECTOR_SIZE:]
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    gyroscope_norm = np.linalg.norm(gyroscope, axis=1)
    estimated_gravity = float(np.median(acceleration_norm))
    gravity_error = abs(estimated_gravity - config.expected_gravity_m_s2)
    acc_norm_robust_std = _robust_standard_deviation(acceleration_norm)
    acc_axis_robust_std = float(
        np.linalg.norm(
            [_robust_standard_deviation(acceleration[:, axis]) for axis in range(3)]
        )
    )
    gyro_median = float(np.median(gyroscope_norm))
    gyro_p95 = float(np.percentile(gyroscope_norm, 95.0))
    gyro_robust_std = _robust_standard_deviation(gyroscope_norm)
    failed_checks = _failed_checks(
        gravity_error=gravity_error,
        acc_norm_robust_std=acc_norm_robust_std,
        acc_axis_robust_std=acc_axis_robust_std,
        gyro_median=gyro_median,
        gyro_p95=gyro_p95,
        gyro_robust_std=gyro_robust_std,
        config=config,
    )
    score = (
        gravity_error / config.max_gravity_error_m_s2
        + acc_norm_robust_std / config.max_acc_norm_robust_std_m_s2
        + acc_axis_robust_std / config.max_acc_axis_robust_std_m_s2
        + gyro_median / config.max_gyro_median_rad_s
        + gyro_p95 / config.max_gyro_p95_rad_s
        + gyro_robust_std / config.max_gyro_robust_std_rad_s
    )
    acceleration_center = np.median(acceleration, axis=0)
    center_norm = float(np.linalg.norm(acceleration_center))
    if center_norm > 0.0:
        gravity_direction = acceleration_center / center_norm
    else:
        gravity_direction = np.zeros(_VECTOR_SIZE, dtype=np.float64)
    gyro_bias = np.median(gyroscope, axis=0)
    return _StationaryCandidate(
        start_index=start_index,
        stop_index=stop_index,
        duration_s=(stop_index - start_index) / config.sampling_rate_hz,
        passed=not failed_checks,
        score=float(score),
        estimated_gravity_m_s2=estimated_gravity,
        gravity_direction_body=tuple(float(value) for value in gravity_direction),
        estimated_gyro_bias_rad_s=tuple(float(value) for value in gyro_bias),
        gravity_error_m_s2=gravity_error,
        acc_norm_robust_std_m_s2=acc_norm_robust_std,
        acc_axis_robust_std_m_s2=acc_axis_robust_std,
        gyro_median_rad_s=gyro_median,
        gyro_p95_rad_s=gyro_p95,
        gyro_robust_std_rad_s=gyro_robust_std,
        failed_checks=failed_checks,
    )


def _failed_checks(
    *,
    gravity_error: float,
    acc_norm_robust_std: float,
    acc_axis_robust_std: float,
    gyro_median: float,
    gyro_p95: float,
    gyro_robust_std: float,
    config: StationarySearchConfig,
) -> tuple[str, ...]:
    checks = (
        ("gravity_error_m_s2", gravity_error, config.max_gravity_error_m_s2),
        (
            "acc_norm_robust_std_m_s2",
            acc_norm_robust_std,
            config.max_acc_norm_robust_std_m_s2,
        ),
        (
            "acc_axis_robust_std_m_s2",
            acc_axis_robust_std,
            config.max_acc_axis_robust_std_m_s2,
        ),
        ("gyro_median_rad_s", gyro_median, config.max_gyro_median_rad_s),
        ("gyro_p95_rad_s", gyro_p95, config.max_gyro_p95_rad_s),
        (
            "gyro_robust_std_rad_s",
            gyro_robust_std,
            config.max_gyro_robust_std_rad_s,
        ),
    )
    return tuple(name for name, value, maximum in checks if value > maximum)


def _validate_imu(imu: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(imu, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise StationarySearchError(
            "IMU must be convertible to a numeric (N, 6) array"
        ) from error
    if values.ndim != 2 or values.shape[1] != _IMU_COLUMN_COUNT:
        raise StationarySearchError(
            f"IMU must have shape (N, 6), got {values.shape}"
        )
    if values.shape[0] == 0:
        raise StationarySearchError("IMU must contain at least one sample")
    if not np.isfinite(values).all():
        raise StationarySearchError("IMU must contain only finite values")
    return values


def _validate_config(config: StationarySearchConfig) -> StationarySearchConfig:
    if not isinstance(config, StationarySearchConfig):
        raise StationarySearchError("config must be a StationarySearchConfig object")
    for name in (
        "sampling_rate_hz",
        "stationary_duration_s",
        "stride_duration_s",
        "expected_gravity_m_s2",
        "max_gravity_error_m_s2",
        "max_acc_norm_robust_std_m_s2",
        "max_acc_axis_robust_std_m_s2",
        "max_gyro_median_rad_s",
        "max_gyro_p95_rad_s",
        "max_gyro_robust_std_rad_s",
    ):
        value = getattr(config, name)
        if isinstance(value, bool):
            raise StationarySearchError(f"{name} must be a positive finite number")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise StationarySearchError(
                f"{name} must be a positive finite number"
            ) from error
        if not math.isfinite(number) or number <= 0.0:
            raise StationarySearchError(f"{name} must be a positive finite number")
    return config


def _sample_count(
    duration_s: float,
    *,
    sampling_rate_hz: float,
    name: str,
) -> int:
    count = int(round(duration_s * sampling_rate_hz))
    if count < 1:
        raise StationarySearchError(
            f"{name} and sampling_rate_hz must produce at least one sample"
        )
    return count


def _robust_standard_deviation(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(_MAD_SCALE * np.median(np.abs(values - median)))
