from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from writingring.gravity import (
    GRAVITY_DERIVED_COLUMNS,
    GravityCalibrationError,
    GravityRemovalConfig,
    InvalidGravityConfigError,
    InvalidGravityInputError,
    _rotate_vector_rodrigues,
    calibrate_gravity_removal,
    process_ring_gravity,
    remove_gravity_in_body_frame,
    upstream_suggested_config,
)
from writingring.ring_loader import load_ring


def _config(
    *,
    sample_count: int = 200,
    calibration_start: int = 0,
    calibration_stop: int = 50,
    **overrides: object,
) -> GravityRemovalConfig:
    values: dict[str, object] = {
        "sampling_rate_hz": 100.0,
        "gyro_scale_to_rad_s": 1.0,
        "calibration_start_sample": calibration_start,
        "calibration_stop_sample": calibration_stop,
        "calibration_min_samples": 20,
        "correction_time_constant_s": 1.0,
    }
    values.update(overrides)
    return GravityRemovalConfig(**values)  # type: ignore[arg-type]


def _stationary(
    sample_count: int = 200,
    *,
    gravity: np.ndarray | None = None,
    gyro_bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gravity_vector = (
        np.array([0.0, 0.0, 9.8])
        if gravity is None
        else np.asarray(gravity, dtype=np.float64)
    )
    bias = (
        np.zeros(3)
        if gyro_bias is None
        else np.asarray(gyro_bias, dtype=np.float64)
    )
    return (
        np.tile(gravity_vector, (sample_count, 1)),
        np.tile(bias, (sample_count, 1)),
    )


@pytest.mark.parametrize(
    "gravity",
    [
        np.array([0.0, 0.0, 9.8]),
        np.array([9.8, 0.0, 0.0]),
        np.array([0.0, -9.8, 0.0]),
        np.array([3.0, 4.0, np.sqrt(9.8**2 - 25.0)]),
    ],
)
def test_stationary_orientations_produce_zero_linear_acceleration(
    gravity: np.ndarray,
) -> None:
    acceleration, gyroscope = _stationary(gravity=gravity)
    config = _config()
    calibration = calibrate_gravity_removal(
        acceleration,
        gyroscope,
        config=config,
    )

    result = remove_gravity_in_body_frame(
        acceleration,
        gyroscope,
        config=config,
        calibration=calibration,
    )

    assert calibration.passed
    np.testing.assert_allclose(result.gravity_body, acceleration, atol=1e-12)
    np.testing.assert_allclose(result.linear_acceleration_body, 0.0, atol=1e-12)
    assert result.diagnostics.calibration_residual_norm_maximum < 1e-12


def test_rodrigues_identity_and_quarter_turn() -> None:
    vector = np.array([0.0, 0.0, 1.0])

    np.testing.assert_array_equal(
        _rotate_vector_rodrigues(vector, np.zeros(3)),
        vector,
    )
    np.testing.assert_allclose(
        _rotate_vector_rodrigues(
            vector,
            np.array([np.pi / 2.0, 0.0, 0.0]),
        ),
        [0.0, -1.0, 0.0],
        atol=1e-12,
    )


def test_known_rotation_tracks_body_frame_gravity_with_correct_sign() -> None:
    sample_count = 250
    calibration_stop = 50
    dt = 0.01
    gyroscope = np.zeros((sample_count, 3))
    gyroscope[calibration_stop:, 0] = np.pi / 2.0
    gravity = np.empty((sample_count, 3))
    gravity[0] = [0.0, 0.0, 9.8]
    for index in range(sample_count - 1):
        interval_omega = (gyroscope[index] + gyroscope[index + 1]) * 0.5
        gravity[index + 1] = _rotate_vector_rodrigues(
            gravity[index],
            -interval_omega * dt,
        )
    config = _config(
        sample_count=sample_count,
        calibration_stop=calibration_stop,
        angular_rate_gate_rad_s=0.5,
    )
    calibration = calibrate_gravity_removal(
        gravity,
        gyroscope,
        config=config,
    )

    result = remove_gravity_in_body_frame(
        gravity,
        gyroscope,
        config=config,
        calibration=calibration,
    )

    np.testing.assert_allclose(result.gravity_body, gravity, atol=1e-10)
    np.testing.assert_allclose(
        result.linear_acceleration_body,
        0.0,
        atol=1e-10,
    )
    assert result.gravity_body[150, 1] > 9.7


def test_known_linear_acceleration_is_retained_when_gate_rejects_it() -> None:
    acceleration, gyroscope = _stationary()
    acceleration[80:140, 0] += 20.0
    config = _config(acceleration_gate_relative_tolerance=0.05)
    calibration = calibrate_gravity_removal(
        acceleration,
        gyroscope,
        config=config,
    )

    result = remove_gravity_in_body_frame(
        acceleration,
        gyroscope,
        config=config,
        calibration=calibration,
    )

    np.testing.assert_allclose(
        result.linear_acceleration_body[80:140],
        np.tile([20.0, 0.0, 0.0], (60, 1)),
        atol=1e-12,
    )
    assert not result.correction_used[80:140].any()


def test_low_pass_method_uses_default_cutoff_and_attenuates_fast_motion() -> None:
    sample_count = 1000
    sampling_rate_hz = 100.0
    time = np.arange(sample_count) / sampling_rate_hz
    acceleration = np.column_stack(
        (
            np.sin(2.0 * np.pi * 5.0 * time),
            np.zeros(sample_count),
            np.full(sample_count, 9.8),
        )
    )
    gyroscope = np.zeros_like(acceleration)
    config = _config(
        sample_count=sample_count,
        gravity_removal_method="low-pass",
    )
    calibration = calibrate_gravity_removal(
        acceleration,
        gyroscope,
        config=config,
    )

    result = remove_gravity_in_body_frame(
        acceleration,
        gyroscope,
        config=config,
        calibration=calibration,
    )

    assert result.config.low_pass_cutoff_hz == pytest.approx(0.2)
    assert not result.diagnostics.noncausal_bidirectional
    np.testing.assert_allclose(result.gravity_body[:, 2], 9.8, atol=1e-10)
    assert np.std(result.gravity_body[500:, 0]) < 0.01
    assert np.std(result.linear_acceleration_body[500:, 0]) > 0.70


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gravity_removal_method": "unknown"}, "gravity_removal_method"),
        ({"madgwick_beta": -0.1}, "madgwick_beta"),
        ({"low_pass_cutoff_hz": 50.0}, "Nyquist"),
    ],
)
def test_method_settings_are_validated(
    overrides: dict[str, object],
    message: str,
) -> None:
    acceleration, gyroscope = _stationary()

    with pytest.raises(InvalidGravityConfigError, match=message):
        calibrate_gravity_removal(
            acceleration,
            gyroscope,
            config=_config(**overrides),
        )


def test_calibration_estimates_and_removes_constant_gyro_bias() -> None:
    bias = np.array([0.01, -0.02, 0.03])
    acceleration, gyroscope = _stationary(gyro_bias=bias)
    config = _config()
    calibration = calibrate_gravity_removal(
        acceleration,
        gyroscope,
        config=config,
    )

    result = remove_gravity_in_body_frame(
        acceleration,
        gyroscope,
        config=config,
        calibration=calibration,
    )

    np.testing.assert_allclose(calibration.gyro_bias_rad_s, bias)
    np.testing.assert_allclose(result.angular_velocity_body_rad_s, 0.0)
    np.testing.assert_allclose(result.linear_acceleration_body, 0.0)


def test_bidirectional_anchor_supports_late_calibration() -> None:
    acceleration, gyroscope = _stationary(sample_count=240)
    config = _config(
        sample_count=240,
        calibration_start=100,
        calibration_stop=160,
    )
    calibration = calibrate_gravity_removal(
        acceleration,
        gyroscope,
        config=config,
    )

    result = remove_gravity_in_body_frame(
        acceleration,
        gyroscope,
        config=config,
        calibration=calibration,
    )

    assert calibration.anchor_sample == 129
    assert result.diagnostics.noncausal_bidirectional
    np.testing.assert_allclose(result.linear_acceleration_body, 0.0)


def test_strict_calibration_rejects_motion_and_provisional_records_warning() -> None:
    acceleration, gyroscope = _stationary()
    gyroscope[:50, 0] = 1.0
    strict = _config()

    with pytest.raises(GravityCalibrationError, match="stationary"):
        calibrate_gravity_removal(acceleration, gyroscope, config=strict)

    provisional = replace(strict, strict_calibration=False)
    calibration = calibrate_gravity_removal(
        acceleration,
        gyroscope,
        config=provisional,
    )
    result = remove_gravity_in_body_frame(
        acceleration,
        gyroscope,
        config=provisional,
        calibration=calibration,
    )

    assert not calibration.passed
    assert result.diagnostics.provisional
    assert any("provisional" in warning for warning in result.diagnostics.warnings)


@pytest.mark.parametrize(
    "transform",
    [
        ((1.0, 0.0), (0.0, 1.0)),
        ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ],
)
def test_invalid_axis_transform_is_rejected(
    transform: tuple[tuple[float, ...], ...],
) -> None:
    acceleration, gyroscope = _stationary()
    config = _config(axis_transform=transform)

    with pytest.raises(InvalidGravityConfigError, match="axis_transform"):
        calibrate_gravity_removal(acceleration, gyroscope, config=config)


def test_malformed_inputs_and_calibration_bounds_are_rejected() -> None:
    acceleration, gyroscope = _stationary()
    config = _config()

    with pytest.raises(InvalidGravityInputError, match="shape"):
        calibrate_gravity_removal(
            acceleration[:, 0],
            gyroscope,
            config=config,
        )
    with pytest.raises(InvalidGravityInputError, match="finite"):
        malformed = acceleration.copy()
        malformed[0, 0] = np.nan
        calibrate_gravity_removal(malformed, gyroscope, config=config)
    with pytest.raises(InvalidGravityInputError, match="counts"):
        calibrate_gravity_removal(
            acceleration[:-1],
            gyroscope,
            config=config,
        )
    with pytest.raises(InvalidGravityConfigError, match="bounds"):
        calibrate_gravity_removal(
            acceleration,
            gyroscope,
            config=replace(config, calibration_stop_sample=500),
        )


def test_result_is_immutable_and_dataframe_has_explicit_columns() -> None:
    acceleration, gyroscope = _stationary()
    before_acceleration = acceleration.copy()
    before_gyroscope = gyroscope.copy()
    config = _config()
    calibration = calibrate_gravity_removal(
        acceleration,
        gyroscope,
        config=config,
    )
    result = remove_gravity_in_body_frame(
        acceleration,
        gyroscope,
        config=config,
        calibration=calibration,
    )

    np.testing.assert_array_equal(acceleration, before_acceleration)
    np.testing.assert_array_equal(gyroscope, before_gyroscope)
    for array in (
        result.acceleration_body,
        result.angular_velocity_body_rad_s,
        result.gravity_body,
        result.linear_acceleration_body,
        result.correction_used,
        result.correction_confidence,
    ):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        result.gravity_body[0, 0] = 1.0

    dataframe = result.to_dataframe()
    assert tuple(dataframe.columns) == GRAVITY_DERIVED_COLUMNS
    assert dataframe.index.name == "sample_index"


def test_ring_adapter_does_not_mutate_loaded_dataframe(tmp_path: Path) -> None:
    acceleration, gyroscope = _stationary()
    timestamps = 1_000_000.0 + np.arange(200) * 10_000.0
    rows = np.column_stack((acceleration, gyroscope, timestamps))
    path = tmp_path / "0_ring_0.bin"
    rows.astype(np.float64).tofile(path)
    ring = load_ring(path)
    before = ring.dataframe.copy(deep=True)
    config = _config()

    result = process_ring_gravity(ring, config=config)
    derived = result.to_dataframe(index=ring.dataframe.index)

    pd.testing.assert_frame_equal(ring.dataframe, before)
    assert derived.index.equals(ring.dataframe.index)
    assert result.sample_count == len(ring.dataframe)


def test_upstream_profile_is_explicitly_labeled_as_assumed() -> None:
    config = upstream_suggested_config(
        sampling_rate_hz=200.0,
        calibration_start_sample=0,
        calibration_stop_sample=50,
    )

    assert config.acceleration_scale_to_working_units == pytest.approx(1 / 9.8)
    assert config.gyro_scale_to_rad_s == 1.0
    assert config.axis_transform == (
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    assert "assumption" in config.acceleration_unit_label
    assert config.profile_name == "upstream_suggested"
