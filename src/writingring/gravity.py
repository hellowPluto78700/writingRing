"""Offline gravity-contribution estimation for WritingRing Ring IMU data.

The estimator preserves the raw Ring data and operates on explicitly
configured working axes and units.  It estimates the stationary acceleration
contribution represented in those axes, not a ground-truth physical gravity
vector. The default Madgwick result is anchored in a caller-selected
stationary calibration interval and propagated both forward and backward, so
it is noncausal. The alternative Butterworth low-pass result is causal. These
methods are intended for offline analysis rather than real-time control.

Column-vector convention
------------------------
For angular velocity ``omega_b`` expressed in a rotating body frame, a
world-fixed vector represented in that frame follows::

    d(g_b) / dt = -omega_b x g_b

Forward propagation therefore applies ``R(-omega_b * dt)``.  Backward
propagation applies its inverse.  Accelerometer correction is confidence
gated to reduce contamination by dynamic linear acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Final, Sequence

import numpy as np
import pandas as pd

from writingring.ring_loader import RingData
from writingring.stationary import (
    StationarySearchConfig,
    StationarySearchResult,
    find_stationary_interval,
)
from writingring.xylo_gravity import XyloGravityConfig


IDENTITY_AXIS_TRANSFORM: Final[
    tuple[tuple[float, float, float], ...]
] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
GRAVITY_DERIVED_COLUMNS: Final[tuple[str, ...]] = (
    "acc_body_x",
    "acc_body_y",
    "acc_body_z",
    "angular_velocity_body_x_rad_s",
    "angular_velocity_body_y_rad_s",
    "angular_velocity_body_z_rad_s",
    "gravity_body_x",
    "gravity_body_y",
    "gravity_body_z",
    "linear_acc_body_x",
    "linear_acc_body_y",
    "linear_acc_body_z",
    "gravity_correction_used",
    "gravity_correction_confidence",
)
_ACCELERATION_COLUMNS: Final[tuple[str, ...]] = ("acc_x", "acc_y", "acc_z")
_GYROSCOPE_COLUMNS: Final[tuple[str, ...]] = ("gyr_x", "gyr_y", "gyr_z")
_VECTOR_SIZE: Final[int] = 3
_ROTATION_TOLERANCE: Final[float] = 1e-12
GRAVITY_REMOVAL_METHODS: Final[tuple[str, ...]] = ("madgwick", "low-pass")


class GravityRemovalError(ValueError):
    """Base error for invalid gravity-removal inputs or configuration."""


class InvalidGravityInputError(GravityRemovalError):
    """Raised when acceleration or gyroscope arrays are malformed."""


class InvalidGravityConfigError(GravityRemovalError):
    """Raised when gravity-removal configuration is invalid."""


class GravityCalibrationError(GravityRemovalError):
    """Raised when calibration cannot support the requested strict result."""


@dataclass(frozen=True, slots=True)
class GravityRemovalConfig:
    """Explicit processing, calibration, and confidence-gate configuration."""

    sampling_rate_hz: float = 200.0
    gyro_scale_to_rad_s: float = 1.0
    calibration_start_sample: int | None = None
    calibration_stop_sample: int | None = None
    acceleration_scale_to_working_units: float = 1.0
    acceleration_unit_label: str = "m/s^2"
    axis_transform: tuple[tuple[float, float, float], ...] = (
        IDENTITY_AXIS_TRANSFORM
    )
    gravity_removal_method: str = "madgwick"
    madgwick_beta: float = 0.1
    low_pass_cutoff_hz: float = 0.2
    correction_time_constant_s: float = 1.0
    acceleration_gate_relative_tolerance: float = 0.15
    angular_rate_gate_rad_s: float | None = None
    calibration_min_samples: int = 20
    calibration_acceleration_norm_mad_relative_max: float = 0.03
    calibration_gyro_norm_median_max_rad_s: float = 0.1
    calibration_gyro_norm_mad_max_rad_s: float = 0.02
    strict_calibration: bool = True
    profile_name: str = "explicit"
    xylo: XyloGravityConfig = XyloGravityConfig()


@dataclass(frozen=True, slots=True)
class GravityCalibration:
    """Validated stationary calibration and its provenance."""

    config: GravityRemovalConfig
    start_sample: int
    stop_sample: int
    anchor_sample: int
    sample_count: int
    gravity_magnitude: float
    stationary_direction_body: tuple[float, float, float]
    gyro_bias_rad_s: tuple[float, float, float]
    acceleration_norm_median: float
    acceleration_norm_mad: float
    acceleration_norm_mad_relative: float
    gyro_norm_median_rad_s: float
    gyro_norm_mad_rad_s: float
    passed: bool
    gravity_magnitude_source: str
    gyro_bias_source: str
    warnings: tuple[str, ...]
    stationary_search: StationarySearchResult | None = None


@dataclass(frozen=True, slots=True)
class GravityRemovalDiagnostics:
    """Aggregate facts describing one completed offline estimate."""

    sample_count: int
    correction_used_count: int
    correction_rejected_count: int
    correction_used_fraction: float
    outputs_finite: bool
    gravity_magnitude_minimum: float
    gravity_magnitude_maximum: float
    calibration_residual_norm_median: float
    calibration_residual_norm_maximum: float
    nominal_sampling_rate_hz: float
    fixed_dt_seconds: float
    noncausal_bidirectional: bool
    provisional: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GravityRemovalResult:
    """Read-only derived arrays plus complete configuration and diagnostics."""

    acceleration_body: np.ndarray
    angular_velocity_body_rad_s: np.ndarray
    gravity_body: np.ndarray
    linear_acceleration_body: np.ndarray
    correction_used: np.ndarray
    correction_confidence: np.ndarray
    config: GravityRemovalConfig
    calibration: GravityCalibration
    diagnostics: GravityRemovalDiagnostics

    @property
    def sample_count(self) -> int:
        """Return the number of derived samples."""

        return int(self.acceleration_body.shape[0])

    def to_dataframe(
        self,
        *,
        index: pd.Index | Sequence[object] | None = None,
    ) -> pd.DataFrame:
        """Return a new DataFrame containing only explicit derived columns."""

        if index is None:
            dataframe_index: pd.Index | Sequence[object] = pd.RangeIndex(
                self.sample_count,
                name="sample_index",
            )
        else:
            if len(index) != self.sample_count:
                raise InvalidGravityInputError(
                    "DataFrame index length must match gravity result sample "
                    f"count {self.sample_count}, got {len(index)}"
                )
            dataframe_index = index.copy() if isinstance(index, pd.Index) else index

        values: dict[str, np.ndarray] = {}
        for prefix, array in (
            ("acc_body", self.acceleration_body),
            ("angular_velocity_body", self.angular_velocity_body_rad_s),
            ("gravity_body", self.gravity_body),
            ("linear_acc_body", self.linear_acceleration_body),
        ):
            suffix = "_rad_s" if prefix == "angular_velocity_body" else ""
            for axis_index, axis_name in enumerate(("x", "y", "z")):
                values[f"{prefix}_{axis_name}{suffix}"] = array[:, axis_index].copy()
        values["gravity_correction_used"] = self.correction_used.copy()
        values["gravity_correction_confidence"] = (
            self.correction_confidence.copy()
        )
        return pd.DataFrame(values, index=dataframe_index)


def upstream_suggested_config(
    *,
    sampling_rate_hz: float,
    calibration_start_sample: int | None = None,
    calibration_stop_sample: int | None = None,
    strict_calibration: bool = True,
    **overrides: object,
) -> GravityRemovalConfig:
    """Return an opt-in configuration based on unconfirmed upstream hints.

    This profile assumes raw acceleration divided by 9.8 is expressed in
    ``g``, raw gyroscope values are radians/second, and y/z signs should be
    flipped.  The assumptions are not confirmed format metadata.
    """

    values: dict[str, object] = {
        "sampling_rate_hz": sampling_rate_hz,
        "gyro_scale_to_rad_s": 1.0,
        "calibration_start_sample": calibration_start_sample,
        "calibration_stop_sample": calibration_stop_sample,
        "acceleration_scale_to_working_units": 1.0 / 9.8,
        "acceleration_unit_label": "g (assumption)",
        "axis_transform": (
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        ),
        "strict_calibration": strict_calibration,
        "profile_name": "upstream_suggested",
    }
    values.update(overrides)
    return GravityRemovalConfig(**values)  # type: ignore[arg-type]


def calibrate_gravity_removal(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    *,
    config: GravityRemovalConfig,
    gravity_magnitude: float | None = None,
    gyro_bias_rad_s: Sequence[float] | None = None,
) -> GravityCalibration:
    """Calibrate from an explicitly selected stationary sample interval."""

    validated_config, transform = _validate_config(config)
    acceleration_values, gyroscope_values = _validated_input_pair(
        acceleration,
        gyroscope,
    )
    start, stop = _validate_calibration_bounds(
        validated_config,
        sample_count=acceleration_values.shape[0],
    )
    acceleration_body, angular_velocity = _configure_inputs(
        acceleration_values,
        gyroscope_values,
        config=validated_config,
        transform=transform,
    )
    calibration_acceleration = acceleration_body[start:stop]
    calibration_gyro = angular_velocity[start:stop]

    acceleration_norm = np.linalg.norm(calibration_acceleration, axis=1)
    gyro_norm = np.linalg.norm(calibration_gyro, axis=1)
    acceleration_norm_median = float(np.median(acceleration_norm))
    acceleration_norm_mad = _median_absolute_deviation(acceleration_norm)
    acceleration_norm_mad_relative = (
        acceleration_norm_mad / acceleration_norm_median
        if acceleration_norm_median > 0.0
        else math.inf
    )
    gyro_norm_median = float(np.median(gyro_norm))
    gyro_norm_mad = _median_absolute_deviation(gyro_norm)

    median_acceleration = np.median(calibration_acceleration, axis=0)
    median_acceleration_norm = float(np.linalg.norm(median_acceleration))
    if (
        not np.isfinite(median_acceleration_norm)
        or median_acceleration_norm <= 0.0
    ):
        raise GravityCalibrationError(
            "calibration acceleration direction has zero or non-finite norm"
        )
    stationary_direction = median_acceleration / median_acceleration_norm

    magnitude_source = "measured"
    if gravity_magnitude is None:
        calibrated_magnitude = acceleration_norm_median
    else:
        calibrated_magnitude = _positive_finite_float(
            gravity_magnitude,
            name="caller-supplied gravity magnitude",
        )
        magnitude_source = "caller supplied"

    bias_source = "measured"
    if gyro_bias_rad_s is None:
        calibrated_bias = np.median(calibration_gyro, axis=0)
    else:
        calibrated_bias = _finite_vector(
            gyro_bias_rad_s,
            name="caller-supplied gyro bias",
        )
        bias_source = "caller supplied"

    failures: list[str] = []
    if (
        acceleration_norm_mad_relative
        > validated_config.calibration_acceleration_norm_mad_relative_max
    ):
        failures.append(
            "calibration acceleration-norm relative MAD "
            f"{acceleration_norm_mad_relative:.6g} exceeds "
            f"{validated_config.calibration_acceleration_norm_mad_relative_max:.6g}"
        )
    if (
        gyro_norm_median
        > validated_config.calibration_gyro_norm_median_max_rad_s
    ):
        failures.append(
            f"calibration gyro-norm median {gyro_norm_median:.6g} rad/s "
            "exceeds "
            f"{validated_config.calibration_gyro_norm_median_max_rad_s:.6g} rad/s"
        )
    if (
        gyro_norm_mad
        > validated_config.calibration_gyro_norm_mad_max_rad_s
    ):
        failures.append(
            f"calibration gyro-norm MAD {gyro_norm_mad:.6g} rad/s exceeds "
            f"{validated_config.calibration_gyro_norm_mad_max_rad_s:.6g} rad/s"
        )

    passed = not failures
    warnings = list(failures)
    if not passed and validated_config.strict_calibration:
        raise GravityCalibrationError(
            "stationary calibration failed: " + "; ".join(failures)
        )
    if not passed:
        warnings.insert(
            0,
            "calibration failed stationary checks; result is provisional",
        )
    warnings.append(
        "nominal sampling rate is an explicit processing assumption"
    )
    if validated_config.profile_name != "explicit":
        warnings.append(
            f"{validated_config.profile_name} unit/axis profile is assumed, "
            "not confirmed upstream"
        )

    return GravityCalibration(
        config=validated_config,
        start_sample=start,
        stop_sample=stop,
        anchor_sample=(start + stop - 1) // 2,
        sample_count=stop - start,
        gravity_magnitude=calibrated_magnitude,
        stationary_direction_body=tuple(
            float(value) for value in stationary_direction
        ),
        gyro_bias_rad_s=tuple(float(value) for value in calibrated_bias),
        acceleration_norm_median=acceleration_norm_median,
        acceleration_norm_mad=acceleration_norm_mad,
        acceleration_norm_mad_relative=acceleration_norm_mad_relative,
        gyro_norm_median_rad_s=gyro_norm_median,
        gyro_norm_mad_rad_s=gyro_norm_mad,
        passed=passed,
        gravity_magnitude_source=magnitude_source,
        gyro_bias_source=bias_source,
        warnings=tuple(warnings),
    )


def auto_calibrate_gravity_removal(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    *,
    gravity_config: GravityRemovalConfig,
    search_config: StationarySearchConfig | None = None,
    gravity_magnitude: float | None = None,
    gyro_bias_rad_s: Sequence[float] | None = None,
) -> GravityCalibration:
    """Find a stationary interval, then apply the existing calibration checks.

    Automatic search is defined for acceleration in m/s^2 and gyroscope data
    in rad/s. Non-default acceleration scaling profiles must instead supply a
    manual interval so their physical-unit assumptions remain explicit.
    """

    validated_config, transform = _validate_config(gravity_config)
    _require_automatic_search_units(validated_config)
    acceleration_values, gyroscope_values = _validated_input_pair(
        acceleration,
        gyroscope,
    )
    effective_search_config = (
        StationarySearchConfig(
            sampling_rate_hz=validated_config.sampling_rate_hz
        )
        if search_config is None
        else search_config
    )
    if not math.isclose(
        effective_search_config.sampling_rate_hz,
        validated_config.sampling_rate_hz,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise GravityCalibrationError(
            "stationary-search sampling_rate_hz must match gravity "
            "sampling_rate_hz"
        )
    acceleration_body, angular_velocity = _configure_inputs(
        acceleration_values,
        gyroscope_values,
        config=validated_config,
        transform=transform,
    )
    search_result = find_stationary_interval(
        np.column_stack((acceleration_body, angular_velocity)),
        config=effective_search_config,
    )
    selected_config = replace(
        validated_config,
        calibration_start_sample=search_result.start_index,
        calibration_stop_sample=search_result.stop_index,
    )
    if not search_result.passed and selected_config.strict_calibration:
        failed_checks = ", ".join(search_result.failed_checks)
        raise GravityCalibrationError(
            "automatic stationary calibration found no passing interval; "
            f"best candidate {search_result.start_index}:"
            f"{search_result.stop_index} failed {failed_checks}"
        )
    calibration = calibrate_gravity_removal(
        acceleration_values,
        gyroscope_values,
        config=selected_config,
        gravity_magnitude=gravity_magnitude,
        gyro_bias_rad_s=gyro_bias_rad_s,
    )
    warnings = list(calibration.warnings)
    if not search_result.passed:
        warnings.insert(
            0,
            "automatic stationary search found no passing interval; "
            "best candidate is provisional (failed: "
            + ", ".join(search_result.failed_checks)
            + ")",
        )
    else:
        warnings.append(
            "calibration interval selected by automatic stationary search"
        )
    return replace(
        calibration,
        passed=calibration.passed and search_result.passed,
        warnings=tuple(dict.fromkeys(warnings)),
        stationary_search=search_result,
    )


def remove_gravity_in_body_frame(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    *,
    config: GravityRemovalConfig,
    calibration: GravityCalibration,
) -> GravityRemovalResult:
    """Estimate and subtract the stationary contribution in configured axes."""

    validated_config, transform = _validate_config(config)
    if not isinstance(calibration, GravityCalibration):
        raise GravityCalibrationError(
            "calibration must be a GravityCalibration object"
        )
    if calibration.config != validated_config:
        raise GravityCalibrationError(
            "calibration configuration does not match processing configuration"
        )
    acceleration_values, gyroscope_values = _validated_input_pair(
        acceleration,
        gyroscope,
    )
    sample_count = int(acceleration_values.shape[0])
    _validate_calibration_bounds(validated_config, sample_count=sample_count)
    if not 0 <= calibration.anchor_sample < sample_count:
        raise GravityCalibrationError(
            f"calibration anchor {calibration.anchor_sample} is outside "
            f"the {sample_count}-sample input"
        )

    acceleration_body, angular_velocity = _configure_inputs(
        acceleration_values,
        gyroscope_values,
        config=validated_config,
        transform=transform,
    )
    gyro_bias = _finite_vector(
        calibration.gyro_bias_rad_s,
        name="calibration gyro bias",
    )
    angular_velocity = angular_velocity - gyro_bias
    gravity_magnitude = _positive_finite_float(
        calibration.gravity_magnitude,
        name="calibration gravity magnitude",
    )
    initial_direction = _finite_vector(
        calibration.stationary_direction_body,
        name="calibration stationary direction",
    )
    initial_direction_norm = float(np.linalg.norm(initial_direction))
    if not np.isclose(initial_direction_norm, 1.0, atol=1e-8, rtol=1e-8):
        raise GravityCalibrationError(
            "calibration stationary direction must have unit norm"
        )

    dt = 1.0 / validated_config.sampling_rate_hz
    if validated_config.gravity_removal_method == "madgwick":
        correction_confidence = _correction_confidence(
            acceleration_body,
            angular_velocity,
            gravity_magnitude=gravity_magnitude,
            config=validated_config,
        )
        gravity_body = _estimate_gravity_madgwick(
            acceleration_body,
            angular_velocity,
            gravity_magnitude=gravity_magnitude,
            initial_direction=initial_direction,
            anchor=calibration.anchor_sample,
            dt=dt,
            beta=validated_config.madgwick_beta,
            correction_confidence=correction_confidence,
        )
        correction_used = correction_confidence > 0.0
        correction_used[calibration.anchor_sample] = False
        noncausal_bidirectional = True
    else:
        gravity_body = _butterworth_low_pass_sos(
            acceleration_body,
            sampling_rate_hz=validated_config.sampling_rate_hz,
            cutoff_hz=validated_config.low_pass_cutoff_hz,
        )
        correction_confidence = np.ones(sample_count, dtype=np.float64)
        correction_used = np.ones(sample_count, dtype=np.bool_)
        noncausal_bidirectional = False

    linear_acceleration = acceleration_body - gravity_body
    gravity_norms = np.linalg.norm(gravity_body, axis=1)
    calibration_residual_norms = np.linalg.norm(
        linear_acceleration[
            calibration.start_sample : calibration.stop_sample
        ],
        axis=1,
    )
    outputs_finite = bool(
        np.isfinite(gravity_body).all()
        and np.isfinite(linear_acceleration).all()
        and np.isfinite(correction_confidence).all()
    )
    if not outputs_finite:
        raise GravityRemovalError(
            "gravity removal produced non-finite derived values"
        )

    warnings = list(calibration.warnings)
    if noncausal_bidirectional:
        warnings.append(
            "Madgwick offline estimate is noncausal: propagated forward and "
            "backward from the calibration anchor"
        )
    else:
        warnings.append(
            "gravity contribution is the causal second-order Butterworth "
            "low-pass output of each acceleration axis"
        )
    if not calibration.passed:
        warnings.append(
            "stationary calibration was not certified; derived values are "
            "provisional"
        )
    if validated_config.acceleration_unit_label != "raw acceleration units":
        warnings.append(
            "acceleration unit label is caller configured, not file metadata"
        )

    used_count = int(np.count_nonzero(correction_used))
    diagnostics = GravityRemovalDiagnostics(
        sample_count=sample_count,
        correction_used_count=used_count,
        correction_rejected_count=sample_count - used_count,
        correction_used_fraction=used_count / sample_count,
        outputs_finite=outputs_finite,
        gravity_magnitude_minimum=float(np.min(gravity_norms)),
        gravity_magnitude_maximum=float(np.max(gravity_norms)),
        calibration_residual_norm_median=float(
            np.median(calibration_residual_norms)
        ),
        calibration_residual_norm_maximum=float(
            np.max(calibration_residual_norms)
        ),
        nominal_sampling_rate_hz=validated_config.sampling_rate_hz,
        fixed_dt_seconds=dt,
        noncausal_bidirectional=noncausal_bidirectional,
        provisional=not calibration.passed,
        warnings=tuple(dict.fromkeys(warnings)),
    )

    result_arrays = (
        acceleration_body,
        angular_velocity,
        gravity_body,
        linear_acceleration,
        correction_used,
        correction_confidence,
    )
    for array in result_arrays:
        array.setflags(write=False)
    return GravityRemovalResult(
        acceleration_body=acceleration_body,
        angular_velocity_body_rad_s=angular_velocity,
        gravity_body=gravity_body,
        linear_acceleration_body=linear_acceleration,
        correction_used=correction_used,
        correction_confidence=correction_confidence,
        config=validated_config,
        calibration=calibration,
        diagnostics=diagnostics,
    )


def process_ring_gravity(
    ring_data: RingData,
    *,
    config: GravityRemovalConfig,
    calibration: GravityCalibration | None = None,
    gravity_magnitude: float | None = None,
    gyro_bias_rad_s: Sequence[float] | None = None,
    stationary_search_config: StationarySearchConfig | None = None,
) -> GravityRemovalResult:
    """Calibrate and process a loaded primary Ring stream without mutation."""

    if not isinstance(ring_data, RingData):
        raise InvalidGravityInputError(
            "process_ring_gravity requires a RingData object"
        )
    missing = tuple(
        column
        for column in (*_ACCELERATION_COLUMNS, *_GYROSCOPE_COLUMNS)
        if column not in ring_data.dataframe
    )
    if missing:
        raise InvalidGravityInputError(
            "RingData is missing required IMU column(s): "
            + ", ".join(missing)
        )
    acceleration = ring_data.dataframe.loc[
        :, list(_ACCELERATION_COLUMNS)
    ].to_numpy(copy=True)
    gyroscope = ring_data.dataframe.loc[
        :, list(_GYROSCOPE_COLUMNS)
    ].to_numpy(copy=True)
    validated_config, _ = _validate_config(config)
    effective_calibration = calibration
    if effective_calibration is None:
        if _has_manual_calibration_bounds(validated_config):
            effective_calibration = calibrate_gravity_removal(
                acceleration,
                gyroscope,
                config=validated_config,
                gravity_magnitude=gravity_magnitude,
                gyro_bias_rad_s=gyro_bias_rad_s,
            )
        else:
            effective_calibration = auto_calibrate_gravity_removal(
                acceleration,
                gyroscope,
                gravity_config=validated_config,
                search_config=stationary_search_config,
                gravity_magnitude=gravity_magnitude,
                gyro_bias_rad_s=gyro_bias_rad_s,
            )
    elif gravity_magnitude is not None or gyro_bias_rad_s is not None:
        raise GravityCalibrationError(
            "gravity_magnitude and gyro_bias_rad_s cannot be supplied with "
            "an existing calibration"
        )
    return remove_gravity_in_body_frame(
        acceleration,
        gyroscope,
        config=effective_calibration.config,
        calibration=effective_calibration,
    )


def _validated_input_pair(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    acceleration_values = _validated_samples(
        acceleration,
        name="acceleration",
    )
    gyroscope_values = _validated_samples(gyroscope, name="gyroscope")
    if acceleration_values.shape[0] != gyroscope_values.shape[0]:
        raise InvalidGravityInputError(
            "acceleration and gyroscope sample counts must match, got "
            f"{acceleration_values.shape[0]} and {gyroscope_values.shape[0]}"
        )
    return acceleration_values, gyroscope_values


def _validated_samples(values: np.ndarray, *, name: str) -> np.ndarray:
    try:
        samples = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidGravityInputError(
            f"{name} must be convertible to a numeric (N, 3) array"
        ) from error
    if samples.ndim != 2 or samples.shape[1] != _VECTOR_SIZE:
        raise InvalidGravityInputError(
            f"{name} must have shape (N, 3), got {samples.shape}"
        )
    if samples.shape[0] == 0:
        raise InvalidGravityInputError(f"{name} must contain at least one sample")
    if not np.isfinite(samples).all():
        raise InvalidGravityInputError(
            f"{name} must contain only finite values"
        )
    return samples.copy()


def _validate_config(
    config: GravityRemovalConfig,
) -> tuple[GravityRemovalConfig, np.ndarray]:
    if not isinstance(config, GravityRemovalConfig):
        raise InvalidGravityConfigError(
            "config must be a GravityRemovalConfig object"
        )
    _positive_finite_float(config.sampling_rate_hz, name="sampling_rate_hz")
    _positive_finite_float(
        config.acceleration_scale_to_working_units,
        name="acceleration_scale_to_working_units",
    )
    _positive_finite_float(
        config.gyro_scale_to_rad_s,
        name="gyro_scale_to_rad_s",
    )
    if config.gravity_removal_method not in GRAVITY_REMOVAL_METHODS:
        raise InvalidGravityConfigError(
            "gravity_removal_method must be one of "
            + ", ".join(GRAVITY_REMOVAL_METHODS)
        )
    _nonnegative_finite_float(config.madgwick_beta, name="madgwick_beta")
    cutoff_hz = _positive_finite_float(
        config.low_pass_cutoff_hz,
        name="low_pass_cutoff_hz",
    )
    if cutoff_hz >= config.sampling_rate_hz / 2.0:
        raise InvalidGravityConfigError(
            "low_pass_cutoff_hz must be below the Nyquist frequency "
            f"({config.sampling_rate_hz / 2.0:g} Hz)"
        )
    _positive_finite_float(
        config.correction_time_constant_s,
        name="correction_time_constant_s",
    )
    _positive_finite_float(
        config.acceleration_gate_relative_tolerance,
        name="acceleration_gate_relative_tolerance",
    )
    if config.angular_rate_gate_rad_s is not None:
        _positive_finite_float(
            config.angular_rate_gate_rad_s,
            name="angular_rate_gate_rad_s",
        )
    if (
        isinstance(config.calibration_min_samples, bool)
        or not isinstance(config.calibration_min_samples, int)
        or config.calibration_min_samples < 1
    ):
        raise InvalidGravityConfigError(
            "calibration_min_samples must be a positive integer"
        )
    _nonnegative_finite_float(
        config.calibration_acceleration_norm_mad_relative_max,
        name="calibration_acceleration_norm_mad_relative_max",
    )
    _nonnegative_finite_float(
        config.calibration_gyro_norm_median_max_rad_s,
        name="calibration_gyro_norm_median_max_rad_s",
    )
    _nonnegative_finite_float(
        config.calibration_gyro_norm_mad_max_rad_s,
        name="calibration_gyro_norm_mad_max_rad_s",
    )
    if not isinstance(config.strict_calibration, bool):
        raise InvalidGravityConfigError(
            "strict_calibration must be a boolean"
        )
    if (
        not isinstance(config.acceleration_unit_label, str)
        or not config.acceleration_unit_label.strip()
    ):
        raise InvalidGravityConfigError(
            "acceleration_unit_label must be a nonempty string"
        )
    if not isinstance(config.profile_name, str) or not config.profile_name.strip():
        raise InvalidGravityConfigError(
            "profile_name must be a nonempty string"
        )
    try:
        transform = np.asarray(config.axis_transform, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidGravityConfigError(
            "axis_transform must be a numeric 3 x 3 matrix"
        ) from error
    if transform.shape != (_VECTOR_SIZE, _VECTOR_SIZE):
        raise InvalidGravityConfigError(
            f"axis_transform must have shape (3, 3), got {transform.shape}"
        )
    if not np.isfinite(transform).all():
        raise InvalidGravityConfigError(
            "axis_transform must contain only finite values"
        )
    if not np.allclose(
        transform @ transform.T,
        np.eye(_VECTOR_SIZE),
        atol=1e-8,
        rtol=1e-8,
    ):
        raise InvalidGravityConfigError(
            "axis_transform must be orthonormal"
        )
    determinant = float(np.linalg.det(transform))
    if not np.isclose(determinant, 1.0, atol=1e-8, rtol=1e-8):
        raise InvalidGravityConfigError(
            "axis_transform must be right-handed with determinant +1"
        )
    normalized_transform = tuple(
        tuple(float(value) for value in row) for row in transform
    )
    normalized_config = replace(config, axis_transform=normalized_transform)
    return normalized_config, transform


def _validate_calibration_bounds(
    config: GravityRemovalConfig,
    *,
    sample_count: int,
) -> tuple[int, int]:
    start = config.calibration_start_sample
    stop = config.calibration_stop_sample
    if start is None and stop is None:
        raise GravityCalibrationError(
            "manual calibration bounds are unavailable; use automatic "
            "calibration or supply both bounds"
        )
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
    ):
        raise InvalidGravityConfigError(
            "calibration sample bounds must be integers"
        )
    if not 0 <= start < stop <= sample_count:
        raise InvalidGravityConfigError(
            "calibration bounds must satisfy "
            f"0 <= start < stop <= {sample_count}, got {start}:{stop}"
        )
    if stop - start < config.calibration_min_samples:
        raise GravityCalibrationError(
            f"calibration interval has {stop - start} sample(s), fewer than "
            f"calibration_min_samples={config.calibration_min_samples}"
        )
    return start, stop


def _has_manual_calibration_bounds(config: GravityRemovalConfig) -> bool:
    start = config.calibration_start_sample
    stop = config.calibration_stop_sample
    if (start is None) != (stop is None):
        raise InvalidGravityConfigError(
            "calibration_start_sample and calibration_stop_sample must be "
            "provided together"
        )
    return start is not None and stop is not None


def _require_automatic_search_units(config: GravityRemovalConfig) -> None:
    if not math.isclose(
        config.acceleration_scale_to_working_units,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise GravityCalibrationError(
            "automatic stationary calibration requires acceleration in m/s^2 "
            "without scaling; supply manual calibration bounds for this profile"
        )


def _configure_inputs(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    *,
    config: GravityRemovalConfig,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    acceleration_body = (
        acceleration
        * config.acceleration_scale_to_working_units
    ) @ transform.T
    angular_velocity = (
        gyroscope * config.gyro_scale_to_rad_s
    ) @ transform.T
    return acceleration_body, angular_velocity


def _correction_confidence(
    acceleration: np.ndarray,
    angular_velocity: np.ndarray,
    *,
    gravity_magnitude: float,
    config: GravityRemovalConfig,
) -> np.ndarray:
    acceleration_norms = np.linalg.norm(acceleration, axis=1)
    relative_error = (
        np.abs(acceleration_norms - gravity_magnitude) / gravity_magnitude
    )
    acceleration_confidence = np.clip(
        1.0
        - relative_error / config.acceleration_gate_relative_tolerance,
        0.0,
        1.0,
    )
    if config.angular_rate_gate_rad_s is None:
        return acceleration_confidence
    angular_rate = np.linalg.norm(angular_velocity, axis=1)
    angular_confidence = np.clip(
        1.0 - angular_rate / config.angular_rate_gate_rad_s,
        0.0,
        1.0,
    )
    return acceleration_confidence * angular_confidence


def _estimate_gravity_madgwick(
    acceleration: np.ndarray,
    angular_velocity: np.ndarray,
    *,
    gravity_magnitude: float,
    initial_direction: np.ndarray,
    anchor: int,
    dt: float,
    beta: float,
    correction_confidence: np.ndarray,
) -> np.ndarray:
    """Run the IMU-only Madgwick update in both directions from an anchor."""

    sample_count = acceleration.shape[0]
    quaternions = np.empty((sample_count, 4), dtype=np.float64)
    quaternions[anchor] = _quaternion_from_gravity_direction(initial_direction)

    for index in range(anchor, sample_count - 1):
        interval_omega = (
            angular_velocity[index] + angular_velocity[index + 1]
        ) * 0.5
        quaternions[index + 1] = _madgwick_imu_step(
            quaternions[index],
            interval_omega,
            acceleration[index + 1],
            dt=dt,
            beta=beta * float(correction_confidence[index + 1]),
        )

    for index in range(anchor, 0, -1):
        interval_omega = -(
            angular_velocity[index - 1] + angular_velocity[index]
        ) * 0.5
        quaternions[index - 1] = _madgwick_imu_step(
            quaternions[index],
            interval_omega,
            acceleration[index - 1],
            dt=dt,
            beta=beta * float(correction_confidence[index - 1]),
        )

    q0 = quaternions[:, 0]
    q1 = quaternions[:, 1]
    q2 = quaternions[:, 2]
    q3 = quaternions[:, 3]
    gravity_unit = np.column_stack(
        (
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q0 * q1 + q2 * q3),
            1.0 - 2.0 * (q1 * q1 + q2 * q2),
        )
    )
    return gravity_unit * gravity_magnitude


def _quaternion_from_gravity_direction(direction: np.ndarray) -> np.ndarray:
    """Return a zero-yaw scalar-first quaternion matching body-frame gravity."""

    x, y, z = (float(value) for value in direction)
    pitch = math.atan2(-x, math.hypot(y, z))
    roll = math.atan2(y, z)
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    quaternion = np.array(
        [
            math.cos(half_roll) * math.cos(half_pitch),
            math.sin(half_roll) * math.cos(half_pitch),
            math.cos(half_roll) * math.sin(half_pitch),
            -math.sin(half_roll) * math.sin(half_pitch),
        ],
        dtype=np.float64,
    )
    return quaternion / np.linalg.norm(quaternion)


def _madgwick_imu_step(
    quaternion: np.ndarray,
    angular_velocity: np.ndarray,
    acceleration: np.ndarray,
    *,
    dt: float,
    beta: float,
) -> np.ndarray:
    """Advance one 6-DoF Madgwick IMU update using scalar-first quaternions."""

    q0, q1, q2, q3 = quaternion
    angular_speed = float(np.linalg.norm(angular_velocity))
    if angular_speed > _ROTATION_TOLERANCE:
        half_angle = 0.5 * angular_speed * dt
        delta_vector = (
            angular_velocity / angular_speed * math.sin(half_angle)
        )
        d0 = math.cos(half_angle)
        d1, d2, d3 = delta_vector
        updated = np.array(
            [
                q0 * d0 - q1 * d1 - q2 * d2 - q3 * d3,
                q0 * d1 + q1 * d0 + q2 * d3 - q3 * d2,
                q0 * d2 - q1 * d3 + q2 * d0 + q3 * d1,
                q0 * d3 + q1 * d2 - q2 * d1 + q3 * d0,
            ],
            dtype=np.float64,
        )
    else:
        updated = quaternion.copy()

    acceleration_norm = float(np.linalg.norm(acceleration))
    if beta > 0.0 and acceleration_norm > _ROTATION_TOLERANCE:
        ax, ay, az = acceleration / acceleration_norm
        residual = np.array(
            [
                2.0 * (q1 * q3 - q0 * q2) - ax,
                2.0 * (q0 * q1 + q2 * q3) - ay,
                1.0 - 2.0 * (q1 * q1 + q2 * q2) - az,
            ],
            dtype=np.float64,
        )
        jacobian = np.array(
            [
                [-2.0 * q2, 2.0 * q3, -2.0 * q0, 2.0 * q1],
                [2.0 * q1, 2.0 * q0, 2.0 * q3, 2.0 * q2],
                [0.0, -4.0 * q1, -4.0 * q2, 0.0],
            ],
            dtype=np.float64,
        )
        gradient = jacobian.T @ residual
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > _ROTATION_TOLERANCE:
            updated -= beta * dt * gradient / gradient_norm

    updated_norm = float(np.linalg.norm(updated))
    if not math.isfinite(updated_norm) or updated_norm <= _ROTATION_TOLERANCE:
        raise GravityRemovalError("Madgwick update produced an invalid quaternion")
    return updated / updated_norm


def _butterworth_low_pass_sos(
    acceleration: np.ndarray,
    *,
    sampling_rate_hz: float,
    cutoff_hz: float,
) -> np.ndarray:
    """Apply one causal second-order Butterworth low-pass SOS per axis."""

    warped = math.tan(math.pi * cutoff_hz / sampling_rate_hz)
    normalization = 1.0 / (1.0 + math.sqrt(2.0) * warped + warped * warped)
    b0 = warped * warped * normalization
    b1 = 2.0 * b0
    b2 = b0
    a1 = 2.0 * (warped * warped - 1.0) * normalization
    a2 = (1.0 - math.sqrt(2.0) * warped + warped * warped) * normalization

    filtered = np.empty_like(acceleration)
    filtered[0] = acceleration[0]
    x1 = acceleration[0].copy()
    x2 = acceleration[0].copy()
    y1 = acceleration[0].copy()
    y2 = acceleration[0].copy()
    for index in range(1, acceleration.shape[0]):
        current = acceleration[index]
        output = (
            b0 * current + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        )
        filtered[index] = output
        x2, x1 = x1, current
        y2, y1 = y1, output
    return filtered


def _correct_gravity(
    predicted: np.ndarray,
    acceleration: np.ndarray,
    *,
    gravity_magnitude: float,
    confidence: float,
    base_weight: float,
) -> np.ndarray:
    predicted_unit = _normalized(predicted, fallback=None)
    if confidence <= 0.0:
        return predicted_unit * gravity_magnitude
    acceleration_unit = _normalized(acceleration, fallback=predicted_unit)
    weight = confidence * base_weight
    blended = (1.0 - weight) * predicted_unit + weight * acceleration_unit
    return _normalized(blended, fallback=predicted_unit) * gravity_magnitude


def _rotate_vector_rodrigues(
    vector: np.ndarray,
    rotation_vector: np.ndarray,
) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < _ROTATION_TOLERANCE:
        return vector.copy()
    axis = rotation_vector / angle
    return (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * float(np.dot(axis, vector)) * (1.0 - math.cos(angle))
    )


def _normalized(
    vector: np.ndarray,
    *,
    fallback: np.ndarray | None,
) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if np.isfinite(norm) and norm > _ROTATION_TOLERANCE:
        return vector / norm
    if fallback is not None:
        return fallback.copy()
    raise GravityRemovalError("cannot normalize a zero or non-finite vector")


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise GravityCalibrationError(
            f"{name} must be a finite three-value vector"
        ) from error
    if vector.shape != (_VECTOR_SIZE,) or not np.isfinite(vector).all():
        raise GravityCalibrationError(
            f"{name} must be a finite three-value vector"
        )
    return vector


def _median_absolute_deviation(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _positive_finite_float(value: object, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number <= 0.0:
        raise InvalidGravityConfigError(f"{name} must be greater than zero")
    return number


def _nonnegative_finite_float(value: object, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number < 0.0:
        raise InvalidGravityConfigError(f"{name} must be nonnegative")
    return number


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise InvalidGravityConfigError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise InvalidGravityConfigError(
            f"{name} must be a finite number"
        ) from error
    if not math.isfinite(number):
        raise InvalidGravityConfigError(f"{name} must be finite")
    return number
