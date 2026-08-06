from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import preprocess_ring_imu
from writingring.imu_preprocessing import PREPROCESSED_IMU_COLUMNS, STANDARD_GRAVITY_M_S2
from writingring.preprocessing_io import PREPROCESSED_IMU_UNITS


def _data_root(tmp_path: Path) -> Path:
    action = tmp_path / "data" / "user_a" / "letters"
    action.mkdir(parents=True)
    sample_count = 12
    acceleration = np.column_stack(
        (
            np.sin(np.arange(sample_count) / 3.0),
            np.cos(np.arange(sample_count) / 4.0),
            np.full(sample_count, STANDARD_GRAVITY_M_S2),
        )
    )
    gyro = np.zeros((sample_count, 3), dtype=np.float64)
    timestamps = 1_000_000.0 + np.arange(sample_count) * 5_000.0
    np.column_stack((acceleration, gyro, timestamps)).tofile(action / "3_ring_0.bin")
    (action / "3_ring_1.bin").write_bytes(b"must not be opened")
    return action.parents[1]


def test_cli_exports_one_complete_recording_and_summary(tmp_path: Path, capsys) -> None:
    data_root = _data_root(tmp_path)
    output_root = tmp_path / "preprocessed"

    assert preprocess_ring_imu.main(
        [
            "--data-root", str(data_root),
            "--output-root", str(output_root),
            "--user", "user_a", "--action", "letters", "--dataset-id", "3",
            "--gravity-removal-method", "raw",
        ]
    ) == 0

    directory = output_root / "user_a" / "letters" / "3"
    imu_path = directory / "3_preprocessedIMU.npy"
    timestamps_path = directory / "3_timestamps_us.npy"
    summary_path = directory / "3_preprocessing.json"
    values = np.load(imu_path, allow_pickle=False)
    timestamps = np.load(timestamps_path, allow_pickle=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert values.shape == (12, 9)
    np.testing.assert_array_equal(timestamps, 1_000_000.0 + np.arange(12) * 5_000.0)
    assert summary["sample_count"] == 12
    assert summary["channel_names"] == list(PREPROCESSED_IMU_COLUMNS)
    assert summary["units"] == list(PREPROCESSED_IMU_UNITS)
    assert summary["gravity_removal_method"] == "raw"
    assert summary["recording"] == {"user": "user_a", "action": "letters", "data_id": 3}
    assert "source_file_sha256" in summary
    assert summary["timestamps_path"] == str(timestamps_path.resolve())
    assert summary["timestamps_sha256"]
    assert "Exported 1 complete recording(s)" in capsys.readouterr().out


def test_cli_requires_complete_recording_selector(tmp_path: Path, capsys) -> None:
    assert preprocess_ring_imu.main(
        [
            "--data-root", str(tmp_path / "missing"),
            "--user", "user_a",
        ]
    ) == 2
    assert "supplied together" in capsys.readouterr().err
