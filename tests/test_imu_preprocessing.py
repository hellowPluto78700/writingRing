from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from writingring.gravity import GravityRemovalConfig
from writingring.imu_preprocessing import (
    PREPROCESSED_IMU_COLUMNS,
    STANDARD_GRAVITY_M_S2,
    preprocess_ring_imu,
)
from writingring.ring_loader import load_ring
from writingring.xylo_gravity import XyloGravityResult
from writingring.xylo_gravity import xylo_rotate_and_remove_gravity


def _ring(tmp_path: Path) -> object:
    rows = np.array(
        [[0.0, 0.0, STANDARD_GRAVITY_M_S2, 1.0, 2.0, 3.0, 100.0]],
        dtype=np.float64,
    )
    path = tmp_path / "0_ring_0.bin"
    rows.tofile(path)
    return load_ring(path)


def _assert_schema(result: object) -> None:
    assert result.imu.shape == (1, 9)
    np.testing.assert_array_equal(result.imu[:, :3], result.acceleration_g)
    np.testing.assert_array_equal(result.imu[:, 3:6], result.acceleration_m_s2)
    np.testing.assert_array_equal(result.imu[:, 6:], result.gyroscope_rad_s)
    np.testing.assert_allclose(
        result.acceleration_m_s2,
        result.acceleration_g * STANDARD_GRAVITY_M_S2,
        rtol=1e-6,
        atol=1e-7,
    )


def test_raw_preprocessing_keeps_measured_acceleration_in_both_units(tmp_path: Path) -> None:
    result = preprocess_ring_imu(
        _ring(tmp_path), config=GravityRemovalConfig(gravity_removal_method="raw")
    )

    _assert_schema(result)
    assert tuple(PREPROCESSED_IMU_COLUMNS) == (
        "acceleration_x_g", "acceleration_y_g", "acceleration_z_g",
        "acceleration_x", "acceleration_y", "acceleration_z",
        "gyro_x", "gyro_y", "gyro_z",
    )
    np.testing.assert_allclose(result.acceleration_g, [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(
        result.acceleration_m_s2, [[0.0, 0.0, STANDARD_GRAVITY_M_S2]]
    )
    assert result.acceleration_semantics == "measured_acceleration_with_gravity"


@pytest.mark.parametrize("method", ["low-pass", "madgwick"])
def test_gravity_methods_derive_g_from_their_linear_m_s2_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    expected = np.array([[STANDARD_GRAVITY_M_S2, 0.0, -STANDARD_GRAVITY_M_S2]])

    def fake_gravity(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            linear_acceleration_body=expected,
            angular_velocity_body_rad_s=np.array([[4.0, 5.0, 6.0]]),
        )

    monkeypatch.setattr("writingring.imu_preprocessing.process_ring_gravity", fake_gravity)
    result = preprocess_ring_imu(
        _ring(tmp_path), config=GravityRemovalConfig(gravity_removal_method=method)
    )

    _assert_schema(result)
    np.testing.assert_allclose(result.acceleration_g, [[1.0, 0.0, -1.0]])
    assert result.acceleration_semantics == "gravity_removed_linear_acceleration"


def test_xylo_uses_g_domain_and_derives_m_s2_from_its_final_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, np.ndarray] = {}
    output = np.array([[0.5, -0.25, 0.0]])

    def fake_xylo(acceleration_g: np.ndarray, **kwargs: object) -> XyloGravityResult:
        captured["input"] = acceleration_g.copy()
        frozen = output.copy()
        frozen.setflags(write=False)
        return XyloGravityResult(frozen, frozen, frozen, frozen, kwargs["config"])

    monkeypatch.setattr("writingring.imu_preprocessing.xylo_rotate_and_remove_gravity", fake_xylo)
    result = preprocess_ring_imu(
        _ring(tmp_path),
        config=GravityRemovalConfig(gravity_removal_method="xylo-rotate-and-remove-gravity"),
    )

    _assert_schema(result)
    np.testing.assert_allclose(captured["input"], [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(result.acceleration_g, output)
    np.testing.assert_allclose(
        result.acceleration_m_s2, [[4.903325, -2.4516625, 0.0]]
    )
    assert result.acceleration_semantics == "xylo_gravity_removed_acceleration"


def test_xylo_quantizer_input_is_g_domain_normalized_and_clipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, np.ndarray] = {}

    def fake_rotation(values: np.ndarray, *, config: object) -> np.ndarray:
        captured["normalized"] = values.copy()
        return values * (2 ** 15)

    monkeypatch.setattr("writingring.xylo_gravity._run_xylo_rotation", fake_rotation)
    result = xylo_rotate_and_remove_gravity(
        np.array([[0.0, 0.0, 9.80665]], dtype=np.float64),
        sampling_rate_hz=200.0,
    )

    np.testing.assert_allclose(captured["normalized"], [[0.0, 0.0, 1.0 - 1e-7]])
    np.testing.assert_allclose(result.rotation_removed_acceleration_g, [[0.0, 0.0, 2.0 - 2e-7]])
