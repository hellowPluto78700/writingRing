from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from writingring.imu_preprocessing import PREPROCESSED_IMU_COLUMNS, STANDARD_GRAVITY_M_S2
from writingring.preprocessing_io import (
    PREPROCESSED_IMU_UNITS,
    PreprocessingIOError,
    load_preprocessed_imu,
    load_preprocessing_summary,
    sha256_file,
    validate_preprocessed_imu,
    validate_timestamp_source_provenance,
)


def _imu(sample_count: int = 4) -> np.ndarray:
    acceleration_g = np.arange(sample_count * 3, dtype=np.float64).reshape(sample_count, 3) / 10.0
    gyro = np.arange(sample_count * 3, dtype=np.float64).reshape(sample_count, 3)
    return np.column_stack((acceleration_g, acceleration_g * STANDARD_GRAVITY_M_S2, gyro))


def _write_summary(
    path: Path,
    *,
    imu_path: Path,
    sample_count: int,
    method: str = "low-pass",
    sampling_rate_hz: float = 200.0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "preprocessed_imu",
                "sample_count": sample_count,
                "channel_count": 9,
                "channel_names": list(PREPROCESSED_IMU_COLUMNS),
                "sampling_rate_hz": sampling_rate_hz,
                "standard_gravity_m_s2": STANDARD_GRAVITY_M_S2,
                "acceleration_semantics": (
                    "measured_acceleration_with_gravity"
                    if method == "raw"
                    else "gravity_removed_linear_acceleration"
                ),
                "gravity_removal_method": method,
                "gravity_removed": method != "raw",
                "units": list(PREPROCESSED_IMU_UNITS),
                "recording": {"user": "user_a", "action": "letters", "data_id": 3},
                "source_file": str(imu_path.resolve()),
                "source_file_sha256": sha256_file(imu_path),
            }
        ),
        encoding="utf-8",
    )


def test_validate_preprocessed_imu_enforces_the_dual_acceleration_contract() -> None:
    values = _imu()
    validated = validate_preprocessed_imu(values)

    np.testing.assert_array_equal(validated, values)
    invalid = values.copy()
    invalid[0, 3] = 1.0
    with pytest.raises(PreprocessingIOError, match=r"g and m/s\^2 acceleration channels"):
        validate_preprocessed_imu(invalid)


@pytest.mark.parametrize(
    "values, message",
    [
        (np.ones((4, 3)), r"shape \(N, 9\)"),
        (np.full((4, 9), np.nan), "finite"),
        (np.ones((4, 9), dtype=object), "numeric"),
    ],
)
def test_validator_rejects_structural_invalid_arrays(
    values: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(PreprocessingIOError, match=message):
        validate_preprocessed_imu(values)


def test_loader_cross_checks_summary_provenance_and_rejects_raw_by_default(
    tmp_path: Path,
) -> None:
    imu_path = tmp_path / "3_preprocessedIMU.npy"
    summary_path = tmp_path / "3_preprocessing.json"
    np.save(imu_path, _imu(), allow_pickle=False)
    _write_summary(summary_path, imu_path=imu_path, sample_count=4, method="raw")

    with pytest.raises(PreprocessingIOError, match="gravity-included"):
        load_preprocessed_imu(imu_path)
    loaded = load_preprocessed_imu(imu_path, allow_gravity_included=True)
    assert loaded.imu.flags.writeable is False
    assert loaded.summary is not None
    assert loaded.summary.gravity_removal_method == "raw"

    summary_path.write_text(
        summary_path.read_text(encoding="utf-8").replace('"sample_count": 4', '"sample_count": 5'),
        encoding="utf-8",
    )
    with pytest.raises(PreprocessingIOError, match="sample_count"):
        load_preprocessed_imu(imu_path, allow_gravity_included=True)


def test_loader_rejects_source_hash_and_sampling_mismatches(tmp_path: Path) -> None:
    imu_path = tmp_path / "3_preprocessedIMU.npy"
    summary_path = tmp_path / "3_preprocessing.json"
    np.save(imu_path, _imu(), allow_pickle=False)
    _write_summary(summary_path, imu_path=imu_path, sample_count=4, sampling_rate_hz=100.0)

    with pytest.raises(PreprocessingIOError, match="sampling_rate_hz"):
        load_preprocessed_imu(imu_path, expected_sampling_rate_hz=200.0)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["source_file_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PreprocessingIOError, match="source_file_sha256"):
        load_preprocessed_imu(imu_path, allow_gravity_included=True)


def test_timestamp_source_provenance_rejects_same_shape_replacement(
    tmp_path: Path,
) -> None:
    imu_path = tmp_path / "3_preprocessedIMU.npy"
    summary_path = tmp_path / "3_preprocessing.json"
    timestamps_path = tmp_path / "3_timestamps_us.npy"
    np.save(imu_path, _imu(), allow_pickle=False)
    timestamps = np.arange(4, dtype=np.float64) * 5_000.0
    np.save(timestamps_path, timestamps, allow_pickle=False)
    _write_summary(summary_path, imu_path=imu_path, sample_count=4)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["timestamps_path"] = str(timestamps_path.resolve())
    payload["timestamps_sha256"] = sha256_file(timestamps_path)
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    summary = load_preprocessing_summary(summary_path)

    assert validate_timestamp_source_provenance(timestamps_path, summary) == sha256_file(
        timestamps_path
    )
    np.save(timestamps_path, timestamps + 1_000_000.0, allow_pickle=False)
    with pytest.raises(PreprocessingIOError, match="timestamp source SHA-256"):
        validate_timestamp_source_provenance(timestamps_path, summary)


@pytest.mark.parametrize("field", (
    "sample_count",
    "channel_names",
    "sampling_rate_hz",
    "gravity_removal_method",
    "gravity_removed",
    "acceleration_semantics",
    "units",
))
@pytest.mark.parametrize("replacement", ["missing", None])
def test_preprocessed_summary_requires_explicit_metadata(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    imu_path = tmp_path / "3_preprocessedIMU.npy"
    summary_path = tmp_path / "3_preprocessing.json"
    np.save(imu_path, _imu(), allow_pickle=False)
    _write_summary(summary_path, imu_path=imu_path, sample_count=4)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if replacement == "missing":
        payload.pop(field)
    else:
        payload[field] = replacement
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreprocessingIOError, match=field):
        load_preprocessed_imu(imu_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_count", "4"),
        ("channel_names", "not-a-list"),
        ("sampling_rate_hz", "200"),
        ("gravity_removal_method", 1),
        ("gravity_removed", "true"),
        ("acceleration_semantics", ["gravity_removed_linear_acceleration"]),
        ("units", {"gyro_x": "degree/s"}),
    ],
)
def test_preprocessed_summary_rejects_metadata_type_errors(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    imu_path = tmp_path / "3_preprocessedIMU.npy"
    summary_path = tmp_path / "3_preprocessing.json"
    np.save(imu_path, _imu(), allow_pickle=False)
    _write_summary(summary_path, imu_path=imu_path, sample_count=4)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload[field] = value
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreprocessingIOError, match=field):
        load_preprocessed_imu(imu_path)


@pytest.mark.parametrize(
    "units",
    [
        ["g", "g", "g", "m/s^2", "m/s^2", "m/s^2", "degree/s", "rad/s", "rad/s"],
        ["m/s^2", "g", "g", "g", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s"],
        list(PREPROCESSED_IMU_UNITS[:-1]),
    ],
)
def test_preprocessed_summary_requires_exact_nine_channel_units(
    tmp_path: Path,
    units: list[str],
) -> None:
    imu_path = tmp_path / "3_preprocessedIMU.npy"
    summary_path = tmp_path / "3_preprocessing.json"
    np.save(imu_path, _imu(), allow_pickle=False)
    _write_summary(summary_path, imu_path=imu_path, sample_count=4)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["units"] = units
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreprocessingIOError, match="units"):
        load_preprocessed_imu(imu_path)


def test_allow_gravity_included_cannot_bypass_missing_gravity_status(
    tmp_path: Path,
) -> None:
    imu_path = tmp_path / "3_preprocessedIMU.npy"
    summary_path = tmp_path / "3_preprocessing.json"
    np.save(imu_path, _imu(), allow_pickle=False)
    _write_summary(summary_path, imu_path=imu_path, sample_count=4, method="raw")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload.pop("gravity_removed")
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreprocessingIOError, match="gravity_removed"):
        load_preprocessed_imu(imu_path, allow_gravity_included=True)


def test_summary_parser_exposes_new_metadata_and_rejects_wrong_standard_gravity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "channel_names": list(PREPROCESSED_IMU_COLUMNS),
                "standard_gravity_m_s2": 9.8,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PreprocessingIOError, match="standard_gravity"):
        load_preprocessing_summary(path)
