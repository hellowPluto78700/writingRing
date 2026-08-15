from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from writingring.discovery import Recording, discover_recordings
from writingring.event_alignment import compute_transient_score_array
from writingring.imu_preprocessing import PREPROCESSED_IMU_COLUMNS
from writingring.preprocessing_io import PREPROCESSED_IMU_UNITS, sha256_file
from writingring.recording_features import (
    SPIKE_IMU_FEATURE_SCHEMA,
    SPIKE_IMU_TRANSIENT_CHANNEL_NAMES,
    RecordingFeatureError,
    load_raw_ring_features,
    load_spike_imu_features,
)


def _recording(tmp_path: Path, *, sample_count: int = 12) -> tuple[Path, Recording]:
    data_root = tmp_path / "data"
    action_dir = data_root / "user_0" / "letters"
    action_dir.mkdir(parents=True)
    timestamps = 1_000_000.0 + np.arange(sample_count, dtype=np.float64) * 5_000.0
    rows = np.column_stack(
        (
            np.zeros((sample_count, 2)),
            np.full(sample_count, 9.80665),
            np.zeros((sample_count, 3)),
            timestamps,
        )
    )
    rows.tofile(action_dir / "0_ring_0.bin")
    (action_dir / "0_timestamp.txt").write_text(
        "1000000 first\n1030000 last\n",
        encoding="utf-8",
    )
    return data_root, discover_recordings(data_root)[0]


def _spike_artifact(
    tmp_path: Path,
    recording: Recording,
    *,
    sample_count: int = 12,
    timestamps: np.ndarray | None = None,
) -> tuple[Path, Path, np.ndarray]:
    spike_root = tmp_path / "spike-root"
    directory = spike_root / recording.user / recording.action / str(recording.dataset_id)
    directory.mkdir(parents=True)
    values = np.arange(sample_count * 21, dtype=np.float32).reshape(sample_count, 21)
    timestamps = (
        np.asarray(timestamps, dtype=np.float64)
        if timestamps is not None
        else 1_000_000.0 + np.arange(sample_count, dtype=np.float64) * 5_000.0
    )
    values_path = directory / "spikeIMU.npy"
    timestamps_path = directory / "0_timestamps_us.npy"
    metadata_path = directory / "metadata.json"
    np.save(values_path, values, allow_pickle=False)
    np.save(timestamps_path, timestamps, allow_pickle=False)
    channel_names = [f"event_{index}" for index in range(15)] + list(
        SPIKE_IMU_TRANSIENT_CHANNEL_NAMES
    )
    metadata = {
        "schema_version": 3,
        "metadata_schema": "spike_encoding_v3",
        "recording": {
            "user": recording.user,
            "action": recording.action,
            "data_id": recording.dataset_id,
        },
        "sampling_rate_hz": 200.0,
        "encoder": {
            "spike_encoder": {"schema": "custom_wavelet_encoder_spec_v1", "frequencies_hz": [0.5, 1.0, 2.0, 4.0, 8.0]},
            "spike_encoder_spec_sha256": hashlib.sha256(json.dumps({"schema": "custom_wavelet_encoder_spec_v1", "frequencies_hz": [0.5, 1.0, 2.0, 4.0, 8.0]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        },
        "timestamps_path": str(timestamps_path.resolve()),
        "timestamps_sha256": sha256_file(timestamps_path),
        "timestamp_unit": "microseconds",
        "spike_imu": {
            "schema": SPIKE_IMU_FEATURE_SCHEMA,
            "sample_count": sample_count,
            "channel_count": 21,
            "channel_names": channel_names,
            "units": ["event"] * 15 + list(PREPROCESSED_IMU_UNITS[3:]),
            "sha256": sha256_file(values_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return spike_root, metadata_path, values


def test_raw_loader_keeps_one_feature_and_timestamp_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, recording = _recording(tmp_path)
    timestamps = np.arange(12, dtype=np.float64) * 5_000.0 + 1_000_000.0
    values = np.arange(108, dtype=np.float64).reshape(12, 9)
    values[:, 3:6] = values[:, :3] * 9.80665
    fake_ring = SimpleNamespace(
        dataframe=pd.DataFrame({"timestamp": timestamps}),
    )
    monkeypatch.setattr("writingring.recording_features.load_ring", lambda _: fake_ring)
    monkeypatch.setattr(
        "writingring.recording_features.preprocess_ring_imu",
        lambda *_args, **_kwargs: SimpleNamespace(imu=values),
    )

    loaded = load_raw_ring_features(recording)

    assert loaded.input_kind == "raw-ring"
    assert loaded.channel_names == PREPROCESSED_IMU_COLUMNS
    assert loaded.units == PREPROCESSED_IMU_UNITS
    assert loaded.transient_channel_indices == (3, 4, 5, 6, 7, 8)
    np.testing.assert_array_equal(loaded.values, values)
    np.testing.assert_array_equal(loaded.timestamps_us, timestamps)
    assert loaded.values.flags.writeable is False
    assert loaded.timestamps_us.flags.writeable is False


def test_spike_loader_validates_identity_hashes_and_accepts_duplicate_timestamps(
    tmp_path: Path,
) -> None:
    _, recording = _recording(tmp_path)
    timestamps = np.array(
        [1_000_000, 1_005_000, 1_005_000, 1_015_000] + [1_020_000] * 8,
        dtype=np.float64,
    )
    spike_root, metadata_path, values = _spike_artifact(
        tmp_path,
        recording,
        timestamps=timestamps,
    )

    loaded = load_spike_imu_features(recording, spike_root=spike_root)
    assert loaded.encoder_spec_sha256
    assert isinstance(loaded.encoder_spec, dict) or loaded.encoder_spec is not None
    with pytest.raises(TypeError):
        loaded.encoder_spec["frequencies_hz"] = [99.0]  # type: ignore[index]

    assert loaded.input_kind == "spike-imu"
    assert loaded.feature_schema == SPIKE_IMU_FEATURE_SCHEMA
    assert loaded.values.shape == (12, 21)
    assert loaded.transient_channel_indices == (15, 16, 17, 18, 19, 20)
    assert loaded.transient_channel_names == SPIKE_IMU_TRANSIENT_CHANNEL_NAMES
    np.testing.assert_array_equal(loaded.values, values)
    np.testing.assert_array_equal(loaded.timestamps_us, timestamps)
    assert loaded.metadata_path == metadata_path
    assert loaded.metadata_sha256 == hashlib.sha256(metadata_path.read_bytes()).hexdigest()

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["spike_imu"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RecordingFeatureError, match="hash"):
        load_spike_imu_features(recording, spike_root=spike_root)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload["recording"].update({"data_id": 99}),
        lambda payload: payload["spike_imu"].update({"schema": "wrong"}),
        lambda payload: payload["spike_imu"].update(
            {"units": ["event"] * 15 + ["degree/s"] * 3 + ["rad/s"] * 3}
        ),
        lambda payload: payload.pop("timestamps_sha256"),
    ],
)
def test_spike_loader_rejects_incomplete_or_misdeclared_metadata(
    tmp_path: Path,
    mutator,
) -> None:
    _, recording = _recording(tmp_path)
    spike_root, metadata_path, _ = _spike_artifact(tmp_path, recording)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutator(payload)
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RecordingFeatureError):
        load_spike_imu_features(recording, spike_root=spike_root)


def test_spike_transient_score_ignores_event_channels() -> None:
    values = np.zeros((8, 21), dtype=np.float64)
    values[:, 15:] = np.arange(48, dtype=np.float64).reshape(8, 6)
    changed = values.copy()
    changed[:, :15] = np.random.default_rng(7).normal(size=(8, 15))

    np.testing.assert_array_equal(
        compute_transient_score_array(values[:, 15:]),
        compute_transient_score_array(changed[:, 15:]),
    )
