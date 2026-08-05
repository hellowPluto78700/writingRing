"""Canonical nine-channel Ring IMU preprocessing output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from writingring.gravity import GravityRemovalConfig, GravityRemovalError, GravityRemovalResult, process_ring_gravity
from writingring.ring_loader import RingData
from writingring.xylo_gravity import XyloGravityError, XyloGravityResult, xylo_rotate_and_remove_gravity


STANDARD_GRAVITY_M_S2: Final[float] = 9.80665
RING_SOURCE_IMU_COLUMNS: Final[tuple[str, ...]] = (
    "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z",
)
PREPROCESSED_IMU_COLUMNS: Final[tuple[str, ...]] = (
    "acceleration_x_g", "acceleration_y_g", "acceleration_z_g",
    "acceleration_x", "acceleration_y", "acceleration_z",
    "gyro_x", "gyro_y", "gyro_z",
)
IMU_PREPROCESSING_METHODS: Final[tuple[str, ...]] = (
    "raw", "low-pass", "madgwick", "xylo-rotate-and-remove-gravity",
)


class IMUPreprocessingError(ValueError):
    """Raised when the common nine-channel contract cannot be met."""


@dataclass(frozen=True, slots=True)
class IMUPreprocessingResult:
    """One fixed-schema result with matching acceleration representations."""

    imu: np.ndarray
    acceleration_g: np.ndarray
    acceleration_m_s2: np.ndarray
    gyroscope_rad_s: np.ndarray
    method: str
    acceleration_semantics: str
    input_acceleration_unit: str
    acceleration_g_unit: str
    acceleration_m_s2_unit: str
    gyroscope_unit: str
    standard_gravity_m_s2: float
    gravity_result: GravityRemovalResult | None
    xylo_result: XyloGravityResult | None


def preprocess_ring_imu(
    ring: RingData,
    *,
    config: GravityRemovalConfig,
) -> IMUPreprocessingResult:
    """Preprocess one loaded Ring recording into the immutable nine-channel schema."""

    if not isinstance(ring, RingData):
        raise IMUPreprocessingError("ring must be RingData")
    if not isinstance(config, GravityRemovalConfig):
        raise IMUPreprocessingError("config must be GravityRemovalConfig")
    method = config.gravity_removal_method
    if method not in IMU_PREPROCESSING_METHODS:
        raise IMUPreprocessingError(
            "gravity_removal_method must be one of: " + ", ".join(IMU_PREPROCESSING_METHODS)
        )
    source = ring.dataframe.loc[:, list(RING_SOURCE_IMU_COLUMNS)].to_numpy(
        dtype=np.float64, copy=True
    )
    raw_acceleration = source[:, :3]
    raw_gyroscope = source[:, 3:]
    gravity_result: GravityRemovalResult | None = None
    xylo_result: XyloGravityResult | None = None
    if method == "raw":
        acceleration_m_s2 = raw_acceleration
        acceleration_semantics = "measured_acceleration_with_gravity"
        gyroscope = raw_gyroscope
    elif method == "xylo-rotate-and-remove-gravity":
        try:
            xylo_result = xylo_rotate_and_remove_gravity(
                raw_acceleration / STANDARD_GRAVITY_M_S2,
                sampling_rate_hz=config.sampling_rate_hz,
                config=config.xylo,
            )
        except XyloGravityError as error:
            raise IMUPreprocessingError(str(error)) from error
        acceleration_g = xylo_result.linear_acceleration_g
        acceleration_m_s2 = acceleration_g * STANDARD_GRAVITY_M_S2
        acceleration_semantics = "xylo_gravity_removed_acceleration"
        gyroscope = raw_gyroscope
    else:
        try:
            gravity_result = process_ring_gravity(ring, config=config)
        except GravityRemovalError as error:
            raise IMUPreprocessingError(str(error)) from error
        acceleration_m_s2 = gravity_result.linear_acceleration_body
        acceleration_semantics = "gravity_removed_linear_acceleration"
        gyroscope = gravity_result.angular_velocity_body_rad_s

    if method != "xylo-rotate-and-remove-gravity":
        acceleration_g = acceleration_m_s2 / STANDARD_GRAVITY_M_S2
    acceleration_g, acceleration_m_s2, gyroscope = _validated_outputs(
        acceleration_g, acceleration_m_s2, gyroscope
    )
    imu = np.column_stack((acceleration_g, acceleration_m_s2, gyroscope))
    for value in (imu, acceleration_g, acceleration_m_s2, gyroscope):
        value.setflags(write=False)
    return IMUPreprocessingResult(
        imu=imu,
        acceleration_g=acceleration_g,
        acceleration_m_s2=acceleration_m_s2,
        gyroscope_rad_s=gyroscope,
        method=method,
        acceleration_semantics=acceleration_semantics,
        input_acceleration_unit="m/s^2",
        acceleration_g_unit="g",
        acceleration_m_s2_unit="m/s^2",
        gyroscope_unit="rad/s",
        standard_gravity_m_s2=STANDARD_GRAVITY_M_S2,
        gravity_result=gravity_result,
        xylo_result=xylo_result,
    )


def _validated_outputs(
    acceleration_g: np.ndarray,
    acceleration_m_s2: np.ndarray,
    gyroscope_rad_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = tuple(np.asarray(value, dtype=np.float64).copy() for value in (
        acceleration_g, acceleration_m_s2, gyroscope_rad_s
    ))
    if any(value.ndim != 2 or value.shape[1] != 3 for value in values):
        raise IMUPreprocessingError("all IMU output components must have shape (N, 3)")
    if len({len(value) for value in values}) != 1:
        raise IMUPreprocessingError("all IMU output components must have the same sample count")
    acceleration_g_value, acceleration_m_s2_value, gyroscope_value = values
    if not np.isfinite(acceleration_g_value).all():
        raise IMUPreprocessingError("g-domain acceleration contains non-finite values")
    if not np.isfinite(acceleration_m_s2_value).all():
        raise IMUPreprocessingError("m/s^2 acceleration contains non-finite values")
    if not np.isfinite(gyroscope_value).all():
        raise IMUPreprocessingError("gyroscope contains non-finite values")
    if not np.allclose(
        acceleration_m_s2_value,
        acceleration_g_value * STANDARD_GRAVITY_M_S2,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise IMUPreprocessingError("g and m/s^2 acceleration channels are inconsistent")
    return values
