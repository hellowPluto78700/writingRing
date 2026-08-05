from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import encode_spikes
from writingring.spike_encoding.contracts import SpikeEncodingSequenceResult
from writingring.spike_encoding.registry import ENCODER_REGISTRY, register_encoder


class _CliDummyEncoder:
    name = "phase1-cli-dummy"
    representation = "dummy"

    @property
    def output_channel_names(self) -> tuple[str, ...]:
        return ("event",)

    def reset(self) -> None:
        return None

    def encode_sequence(self, acceleration_g: np.ndarray) -> SpikeEncodingSequenceResult:
        return SpikeEncodingSequenceResult(
            values=acceleration_g[:, :1],
            channel_names=self.output_channel_names,
            representation=self.representation,
            diagnostics={},
        )


class _ResetCountingCustomWavelet:
    name = "custom-wavelet"
    representation = "signed_sparse_wavelet_extrema"
    reset_calls = 0
    padding_samples_each_side = 30

    @property
    def output_channel_names(self) -> tuple[str, ...]:
        return tuple(f"event_{index}" for index in range(15))

    def reset(self) -> None:
        type(self).reset_calls += 1

    def encode_sequence(self, acceleration_g: np.ndarray) -> SpikeEncodingSequenceResult:
        return SpikeEncodingSequenceResult(
            values=np.zeros((len(acceleration_g), 15), dtype=np.float32),
            channel_names=self.output_channel_names,
            representation=self.representation,
            diagnostics={},
        )


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    imu = tmp_path / "input_rawIMU.npy"
    settings = tmp_path / "dummy.json"
    offsets = tmp_path / "offsets.npy"
    np.save(imu, np.ones((3, 9), dtype=np.float32), allow_pickle=False)
    settings.write_text(json.dumps({"sampling_rate_hz": 200.0}), encoding="utf-8")
    np.save(offsets, np.array([0, 1, 3], dtype=np.int64), allow_pickle=False)
    return imu, settings, offsets


def test_cli_dispatches_a_registered_dummy_encoder(tmp_path: Path, capsys) -> None:
    register_encoder("phase1-cli-dummy", lambda settings: _CliDummyEncoder(), replace=True)
    imu, settings, offsets = _paths(tmp_path)

    assert encode_spikes.main(
        [
            "--input-imu", str(imu),
            "--encoder", "phase1-cli-dummy",
            "--encoder-settings", str(settings),
            "--sequence-mode", "offsets",
            "--sequence-offsets", str(offsets),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "Encoded 3 samples into 1 channels across 2 sequence(s)." in output
    assert "Published spike encoding" in output
    assert (imu.parent / "phase1-cli-dummy" / "input" / "input_spikeEvents.npy").is_file()


def test_cli_rejects_offsets_mode_without_offsets(tmp_path: Path, capsys) -> None:
    imu, settings, _ = _paths(tmp_path)

    assert encode_spikes.main(
        [
            "--input-imu", str(imu),
            "--encoder", "missing",
            "--encoder-settings", str(settings),
            "--sequence-mode", "offsets",
        ]
    ) == 2
    assert "requires --sequence-offsets" in capsys.readouterr().err


def test_cli_publishes_custom_wavelet_with_summary_and_dynamic_channels(
    tmp_path: Path,
    capsys,
) -> None:
    imu = tmp_path / "recording_rawIMU.npy"
    settings = tmp_path / "custom.json"
    summary = tmp_path / "recording_segmentation_summary.json"
    samples = np.zeros((80, 9), dtype=np.float32)
    samples[:, 0] = np.sin(np.arange(80) / 3.0)
    samples[:, 1] = -samples[:, 0]
    np.save(imu, samples, allow_pickle=False)
    settings.write_text(
        json.dumps({"sampling_rate_hz": 200.0}),
        encoding="utf-8",
    )
    labels = tmp_path / "recording_labels.npy"
    segment_offsets = tmp_path / "recording_segment_offsets.npy"
    segment_lengths = tmp_path / "recording_segment_lengths.npy"
    timestamps = tmp_path / "recording_timestamps.npy"
    manifest = tmp_path / "recording_manifest.json"
    np.save(labels, np.array(["a", "b"]), allow_pickle=False)
    np.save(segment_offsets, np.array([0, 40, 80], dtype=np.int64), allow_pickle=False)
    np.save(segment_lengths, np.array([40, 40], dtype=np.int64), allow_pickle=False)
    np.save(timestamps, np.arange(80, dtype=np.float64) / 200.0, allow_pickle=False)
    manifest.write_text("{}", encoding="utf-8")
    summary.write_text(
        json.dumps(
            {
                "channel_count": 9,
                "channel_names": [
                    "acceleration_x_g", "acceleration_y_g", "acceleration_z_g",
                    "acceleration_x", "acceleration_y", "acceleration_z",
                    "gyro_x", "gyro_y", "gyro_z",
                ],
                "gravity_removal": {"method": "low-pass", "sampling_rate_hz": 200.0},
                "acceleration_semantics": "gravity_removed_linear_acceleration",
            }
        ),
        encoding="utf-8",
    )
    assert encode_spikes.main(
        [
            "--input-imu", str(imu),
            "--input-summary", str(summary),
            "--encoder", "custom-wavelet",
            "--encoder-settings", str(settings),
        ]
    ) == 0

    output_directory = tmp_path / "custom-wavelet" / "recording"
    events = np.load(output_directory / "recording_spikeEvents.npy", allow_pickle=False)
    spike_imu = np.load(output_directory / "recording_spikeIMU.npy", allow_pickle=False)
    published = json.loads((output_directory / "recording_spike_encoding_summary.json").read_text(encoding="utf-8"))
    assert events.shape == (80, 15)
    assert spike_imu.shape == (80, 21)
    np.testing.assert_array_equal(spike_imu[:, :15], events)
    np.testing.assert_array_equal(spike_imu[:, 15:], samples[:, 3:])
    assert events.dtype == np.float32
    assert published["source"]["gravity_removal_method"] == "low-pass"
    assert published["settings"]["wavelet_widths_samples"] == [400, 200, 100, 50, 25]
    assert published["output"]["channel_names"][-1] == "event_z_8_hz"
    assert published["output"]["channel_order"] == "axis_major_frequency_minor"
    assert published["sequence_processing"]["state_reset_boundary"] == "recording"
    assert published["sequence_processing"]["offset_semantics"] == "recording"
    references = published["source"]["metadata_references"]
    assert references["labels_path"] == str(labels.resolve())
    assert references["segment_offsets_path"] == str(segment_offsets.resolve())
    assert references["segment_lengths_path"] == str(segment_lengths.resolve())
    assert references["timestamp_source_path"] == str(timestamps.resolve())
    assert references["segments_manifest_path"] == str(manifest.resolve())
    assert published["source"]["metadata_usage"] == {
        "used_by_encoder": False,
        "used_as_reset_boundaries": False,
        "reused_for_downstream_alignment": True,
        "indices_shifted": False,
    }
    np.testing.assert_array_equal(
        spike_imu[40:80],
        spike_imu[np.load(segment_offsets, allow_pickle=False)[1]:],
    )
    assert "Published spike encoding" in capsys.readouterr().out


@pytest.mark.parametrize(
    "option",
    [
        "--sequence-offsets",
        "--labels-path",
        "--segment-offsets-path",
        "--segment-lengths-path",
    ],
)
def test_cli_rejects_custom_wavelet_legacy_sidecars(
    tmp_path: Path,
    capsys,
    option: str,
) -> None:
    imu, settings, offsets = _paths(tmp_path)

    assert encode_spikes.main(
        [
            "--input-imu", str(imu),
            "--encoder", "custom-wavelet",
            "--encoder-settings", str(settings),
            option, str(offsets),
        ]
    ) == 2

    assert option in capsys.readouterr().err


@pytest.mark.parametrize(
    "option",
    [
        "--labels-path",
        "--segment-offsets-path",
        "--segment-lengths-path",
        "--segments-manifest-path",
    ],
)
def test_cli_rejects_missing_explicit_generic_sidecar(
    tmp_path: Path,
    capsys,
    option: str,
) -> None:
    register_encoder("phase1-cli-dummy", lambda settings: _CliDummyEncoder(), replace=True)
    imu, settings, _ = _paths(tmp_path)
    missing = tmp_path / "missing_labels.npy"

    assert encode_spikes.main(
        [
            "--input-imu", str(imu),
            "--encoder", "phase1-cli-dummy",
            "--encoder-settings", str(settings),
            option, str(missing),
        ]
    ) == 2

    assert "source metadata path is not a regular file" in capsys.readouterr().err


def test_cli_runs_custom_wavelet_as_exactly_one_recording(tmp_path: Path, capsys) -> None:
    imu, settings, _ = _paths(tmp_path)
    original_factory = ENCODER_REGISTRY["custom-wavelet"]
    _ResetCountingCustomWavelet.reset_calls = 0
    register_encoder(
        "custom-wavelet", lambda settings: _ResetCountingCustomWavelet(), replace=True
    )
    try:
        assert encode_spikes.main(
            [
                "--input-imu", str(imu),
                "--encoder", "custom-wavelet",
                "--encoder-settings", str(settings),
            ]
        ) == 0
    finally:
        register_encoder("custom-wavelet", original_factory, replace=True)

    assert _ResetCountingCustomWavelet.reset_calls == 1
    assert "across 1 recording(s)" in capsys.readouterr().out


def test_cli_validates_optional_timestamps_without_encoding_with_them(
    tmp_path: Path,
    capsys,
) -> None:
    imu, settings, _ = _paths(tmp_path)
    timestamps = tmp_path / "timestamps.npy"
    np.save(timestamps, np.array([0.0, 0.005, 0.01]), allow_pickle=False)

    assert encode_spikes.main(
        [
            "--input-imu", str(imu),
            "--encoder", "custom-wavelet",
            "--encoder-settings", str(settings),
            "--timestamps-path", str(timestamps),
        ]
    ) == 0
    assert "Traceback" not in capsys.readouterr().err


def test_cli_rejects_missing_explicit_timestamp_or_summary(tmp_path: Path, capsys) -> None:
    imu, settings, _ = _paths(tmp_path)

    assert encode_spikes.main(
        [
            "--input-imu", str(imu),
            "--encoder", "custom-wavelet",
            "--encoder-settings", str(settings),
            "--timestamps-path", str(tmp_path / "missing_timestamps.npy"),
        ]
    ) == 2
    assert "timestamps file is not a regular file" in capsys.readouterr().err

    assert encode_spikes.main(
        [
            "--input-imu", str(imu),
            "--input-summary", str(tmp_path / "missing_summary.json"),
            "--encoder", "custom-wavelet",
            "--encoder-settings", str(settings),
        ]
    ) == 2
    assert "input summary is not a regular file" in capsys.readouterr().err


def test_cli_rejects_untrusted_source_summary_without_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    imu, settings, _ = _paths(tmp_path)
    source_summary = tmp_path / "invalid_summary.json"
    source_summary.write_text(
        json.dumps(
            {
                "gravity_removal": {"method": "unknown", "sampling_rate_hz": 200.0},
                "acceleration_semantics": "unknown",
            }
        ),
        encoding="utf-8",
    )

    assert encode_spikes.main(
        [
            "--input-imu", str(imu),
            "--input-summary", str(source_summary),
            "--encoder", "missing",
            "--encoder-settings", str(settings),
        ]
    ) == 2

    error = capsys.readouterr().err
    assert "must be one of" in error
    assert "Traceback" not in error
