from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import encode_spikes
from writingring.gravity import GravityRemovalConfig
from writingring.imu_preprocessing import (
    PREPROCESSED_IMU_COLUMNS,
    STANDARD_GRAVITY_M_S2,
    preprocess_ring_imu,
)
from writingring.preprocessing_io import PREPROCESSED_IMU_UNITS, sha256_file
from writingring.ring_loader import load_ring
from writingring.spike_encoding.contracts import SpikeEncodingSequenceResult
from writingring.spike_encoding.registry import register_encoder


class _BatchEncoder:
    name = "batch-contract-dummy"
    representation = "signed_dummy"

    def __init__(self, instances: list["_BatchEncoder"]) -> None:
        self.instances = instances
        self.reset_count = 0
        instances.append(self)

    @property
    def output_channel_names(self) -> tuple[str, ...]:
        return ("event",)

    def reset(self) -> None:
        self.reset_count += 1

    def encode_sequence(self, acceleration_g: np.ndarray) -> SpikeEncodingSequenceResult:
        return SpikeEncodingSequenceResult(
            values=acceleration_g[:, :1],
            channel_names=self.output_channel_names,
            representation=self.representation,
            diagnostics={},
        )


def _physical_imu(sample_count: int = 80) -> np.ndarray:
    acceleration_g = np.column_stack(
        (
            np.sin(np.arange(sample_count) / 5.0),
            np.cos(np.arange(sample_count) / 7.0),
            np.sin(np.arange(sample_count) / 9.0),
        )
    )
    gyro = np.zeros((sample_count, 3), dtype=np.float64)
    return np.column_stack((acceleration_g, acceleration_g * 9.80665, gyro))


def _write_artifact(directory: Path, data_id: int) -> Path:
    directory.mkdir(parents=True)
    imu_path = directory / f"{data_id}_preprocessedIMU.npy"
    summary_path = directory / f"{data_id}_preprocessing.json"
    values = _physical_imu()
    np.save(imu_path, values, allow_pickle=False)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "preprocessed_imu",
                "sample_count": len(values),
                "channel_count": 9,
                "channel_names": list(PREPROCESSED_IMU_COLUMNS),
                "sampling_rate_hz": 200.0,
                "standard_gravity_m_s2": 9.80665,
                "acceleration_semantics": "gravity_removed_linear_acceleration",
                "gravity_removal_method": "low-pass",
                "gravity_removed": True,
                "units": list(PREPROCESSED_IMU_UNITS),
                "recording": {"user": "user_a", "action": "letters", "data_id": data_id},
                "source_file": str(imu_path.resolve()),
                "source_file_sha256": __import__("hashlib").sha256(imu_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return imu_path


@pytest.mark.parametrize(
    ("method", "semantics"),
    [
        ("low-pass", "gravity_removed_linear_acceleration"),
        ("madgwick", "gravity_removed_linear_acceleration"),
        ("xylo-rotate-and-remove-gravity", "xylo_gravity_removed_acceleration"),
    ],
)
def test_preprocess_methods_are_valid_spike_producer_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    semantics: str,
) -> None:
    expected = _physical_imu(80)
    ring_path = tmp_path / "0_ring_0.bin"
    rows = np.column_stack(
        (
            expected[:, 0:3] * STANDARD_GRAVITY_M_S2,
            expected[:, 6:9],
            1_000_000.0 + np.arange(80) * 5_000.0,
        )
    )
    rows.tofile(ring_path)

    if method in {"low-pass", "madgwick"}:
        def fake_gravity(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                linear_acceleration_body=expected[:, 3:6],
                angular_velocity_body_rad_s=expected[:, 6:9],
            )

        monkeypatch.setattr("writingring.imu_preprocessing.process_ring_gravity", fake_gravity)
    else:
        def fake_xylo(acceleration_g: np.ndarray, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(linear_acceleration_g=np.array(acceleration_g, copy=True))

        monkeypatch.setattr(
            "writingring.imu_preprocessing.xylo_rotate_and_remove_gravity",
            fake_xylo,
        )

    result = preprocess_ring_imu(
        load_ring(ring_path),
        config=GravityRemovalConfig(gravity_removal_method=method),
    )
    source = tmp_path / "0_preprocessedIMU.npy"
    np.save(source, result.imu, allow_pickle=False)
    summary = tmp_path / "0_preprocessing.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "preprocessed_imu",
                "sample_count": 80,
                "channel_count": 9,
                "channel_names": list(PREPROCESSED_IMU_COLUMNS),
                "sampling_rate_hz": 200.0,
                "standard_gravity_m_s2": STANDARD_GRAVITY_M_S2,
                "acceleration_semantics": semantics,
                "gravity_removal_method": method,
                "gravity_removed": True,
                "units": list(PREPROCESSED_IMU_UNITS),
                "source_file": str(source.resolve()),
                "source_file_sha256": sha256_file(source),
            }
        ),
        encoding="utf-8",
    )
    assert result.imu.shape == (80, 9)
    np.testing.assert_allclose(
        result.imu[:, 3:6], result.imu[:, :3] * STANDARD_GRAVITY_M_S2
    )
    settings = tmp_path / "custom_wavelet.json"
    settings.write_text(json.dumps({"sampling_rate_hz": 200.0}), encoding="utf-8")
    assert encode_spikes.main(
        [
            "--input-imu", str(source),
            "--input-summary", str(summary),
            "--encoder", "custom-wavelet",
            "--encoder-settings", str(settings),
            "--output-root", str(tmp_path / "spikes"),
        ]
    ) == 0
    output = tmp_path / "spikes" / "custom-wavelet" / "0"
    events = np.load(output / "0_spikeEvents.npy", allow_pickle=False)
    spike_imu = np.load(output / "0_spikeIMU.npy", allow_pickle=False)
    assert events.shape == (80, 15)
    assert spike_imu.shape == (80, 21)
    np.testing.assert_array_equal(spike_imu[:, :15], events)
    np.testing.assert_array_equal(spike_imu[:, 15:], result.imu[:, 3:9])


def test_raw_preprocessing_is_rejected_by_default_and_requires_opt_in(
    tmp_path: Path,
    capsys,
) -> None:
    expected = _physical_imu(80)
    ring_path = tmp_path / "0_ring_0.bin"
    np.column_stack(
        (
            expected[:, :3] * STANDARD_GRAVITY_M_S2,
            expected[:, 6:9],
            1_000_000.0 + np.arange(80) * 5_000.0,
        )
    ).tofile(ring_path)
    result = preprocess_ring_imu(
        load_ring(ring_path),
        config=GravityRemovalConfig(gravity_removal_method="raw"),
    )
    source = tmp_path / "0_preprocessedIMU.npy"
    np.save(source, result.imu, allow_pickle=False)
    summary = tmp_path / "0_preprocessing.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "preprocessed_imu",
                "sample_count": len(result.imu),
                "channel_count": 9,
                "channel_names": list(PREPROCESSED_IMU_COLUMNS),
                "sampling_rate_hz": 200.0,
                "standard_gravity_m_s2": STANDARD_GRAVITY_M_S2,
                "acceleration_semantics": "measured_acceleration_with_gravity",
                "gravity_removal_method": "raw",
                "gravity_removed": False,
                "units": list(PREPROCESSED_IMU_UNITS),
                "source_file": str(source.resolve()),
                "source_file_sha256": sha256_file(source),
            }
        ),
        encoding="utf-8",
    )
    settings = tmp_path / "custom_wavelet.json"
    settings.write_text(json.dumps({"sampling_rate_hz": 200.0}), encoding="utf-8")
    arguments = [
        "--input-imu", str(source),
        "--input-summary", str(summary),
        "--encoder", "custom-wavelet",
        "--encoder-settings", str(settings),
        "--output-root", str(tmp_path / "spikes"),
    ]

    assert encode_spikes.main(arguments) == 2
    assert "gravity-included" in capsys.readouterr().err
    assert encode_spikes.main(arguments + ["--allow-gravity-included"]) == 0

    output = tmp_path / "spikes" / "custom-wavelet" / "0"
    spikes = np.load(output / "0_spikeEvents.npy", allow_pickle=False)
    spike_imu = np.load(output / "0_spikeIMU.npy", allow_pickle=False)
    assert spikes.shape == (80, 15)
    assert spike_imu.shape == (80, 21)
    np.testing.assert_array_equal(spike_imu[:, 15:], result.imu[:, 3:9])


def test_batch_encoding_preserves_relative_paths_and_resets_per_recording(
    tmp_path: Path,
    capsys,
) -> None:
    input_root = tmp_path / "preprocessed"
    _write_artifact(input_root / "user_a" / "letters" / "3", 3)
    _write_artifact(input_root / "user_a" / "letters" / "10", 10)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"sampling_rate_hz": 200.0}), encoding="utf-8")
    instances: list[_BatchEncoder] = []
    register_encoder(
        "batch-contract-dummy",
        lambda _settings: _BatchEncoder(instances),
        replace=True,
    )

    assert encode_spikes.main(
        [
            "--input-root", str(input_root),
            "--output-root", str(tmp_path / "spikes"),
            "--encoder", "batch-contract-dummy",
            "--encoder-settings", str(settings),
        ]
    ) == 0

    assert len(instances) == 2
    assert [instance.reset_count for instance in instances] == [1, 1]
    for data_id in (3, 10):
        output = tmp_path / "spikes" / "batch-contract-dummy" / "user_a" / "letters" / str(data_id)
        assert (output / "spikes.npy").is_file()
        assert (output / "metadata.json").is_file()
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["source"]["recording"]["data_id"] == data_id
        assert metadata["sequence_processing"]["sequence_count"] == 1
    assert "Published 2 batch spike encoding(s)" in capsys.readouterr().out
