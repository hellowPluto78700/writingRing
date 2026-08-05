from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import encode_spikes
from writingring.spike_encoding.contracts import SpikeEncodingSequenceResult
from writingring.spike_encoding.registry import register_encoder


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
    offsets = tmp_path / "offsets.npy"
    samples = np.zeros((80, 9), dtype=np.float32)
    samples[:, 0] = np.sin(np.arange(80) / 3.0)
    samples[:, 1] = -samples[:, 0]
    np.save(imu, samples, allow_pickle=False)
    settings.write_text(
        json.dumps({"frequencies_hz": [2.0, 4.0], "sampling_rate_hz": 200.0}),
        encoding="utf-8",
    )
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
    np.save(offsets, np.array([0, 30, 80], dtype=np.int64), allow_pickle=False)

    assert encode_spikes.main(
        [
            "--input-imu", str(imu),
            "--input-summary", str(summary),
            "--encoder", "custom-wavelet",
            "--encoder-settings", str(settings),
            "--sequence-offsets", str(offsets),
        ]
    ) == 0

    output_directory = tmp_path / "custom-wavelet" / "recording"
    events = np.load(output_directory / "recording_spikeEvents.npy", allow_pickle=False)
    published = json.loads((output_directory / "recording_spike_encoding_summary.json").read_text(encoding="utf-8"))
    assert events.shape == (80, 6)
    assert events.dtype == np.float32
    assert published["source"]["gravity_removal_method"] == "low-pass"
    assert published["settings"]["wavelet_widths_samples"] == [100, 50]
    assert published["output"]["channel_names"][-1] == "event_z_4_hz"
    assert published["output"]["channel_order"] == "axis_major_frequency_minor"
    assert "Published spike encoding" in capsys.readouterr().out


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
