from __future__ import annotations

from pathlib import Path

import numpy as np

from writingring.spike_encoding.encoders.custom_wavelet import (
    CustomWaveletEncoder,
    CustomWaveletSettings,
)
from writingring.spike_encoding.io import load_spike_encoding_input
from writingring.spike_encoding.publication import (
    publish_spike_encoding,
    spike_encoding_output_paths,
)
from writingring.spike_encoding.runner import run_spike_encoder


def test_custom_wavelet_preserves_default_fifteen_channels_and_recording_reset() -> None:
    samples = np.column_stack(
        (
            np.sin(np.arange(100) / 5.0),
            -np.sin(np.arange(100) / 7.0),
            np.cos(np.arange(100) / 11.0),
        )
    )
    encoder = CustomWaveletEncoder()

    result = run_spike_encoder(
        encoder,
        acceleration_g=np.vstack((samples, samples)),
        sequence_offsets=np.array([0, 100, 200], dtype=np.int64),
        sequence_boundary_semantics="recording",
    )

    assert result.values.shape == (200, 15)
    assert len(result.channel_names) == 15
    assert result.channel_names[:5] == (
        "event_x_0p5_hz",
        "event_x_1_hz",
        "event_x_2_hz",
        "event_x_4_hz",
        "event_x_8_hz",
    )
    np.testing.assert_array_equal(result.values[:100], result.values[100:])
    assert result.summary["state_reset_boundary"] == "recording"


def test_spike_imu_replaces_only_g_channels_and_references_unchanged_sidecars(tmp_path: Path) -> None:
    raw_path = tmp_path / "recording_rawIMU.npy"
    raw = np.arange(90 * 9, dtype=np.float32).reshape(90, 9)
    raw[:, :3] = np.column_stack(
        (
            np.sin(np.arange(90) / 5.0),
            np.cos(np.arange(90) / 7.0),
            np.sin(np.arange(90) / 9.0),
        )
    )
    raw[:, 3:6] = raw[:, :3] * 9.80665
    raw_before = raw.copy()
    np.save(raw_path, raw, allow_pickle=False)
    labels = tmp_path / "recording_labels.npy"
    segment_offsets = tmp_path / "recording_segment_offsets.npy"
    segment_lengths = tmp_path / "recording_segment_lengths.npy"
    np.save(labels, np.array(["a", "b"]))
    np.save(segment_offsets, np.array([0, 40, 90], dtype=np.int64))
    np.save(segment_lengths, np.array([40, 50], dtype=np.int64))
    sidecars_before = {
        path: path.read_bytes() for path in (labels, segment_offsets, segment_lengths)
    }
    input_data = load_spike_encoding_input(raw_path)
    encoder = CustomWaveletEncoder(CustomWaveletSettings(output_dtype="float32"))
    output = run_spike_encoder(
        encoder,
        acceleration_g=input_data.acceleration_g,
        sequence_offsets=np.array([0, len(raw)], dtype=np.int64),
        sequence_boundary_semantics="recording",
    )
    paths = spike_encoding_output_paths(
        raw_imu_path=raw_path,
        encoder_name=encoder.name,
        output_stem=None,
        output_root=None,
    )

    summary = publish_spike_encoding(
        output=output,
        input_data=input_data,
        encoder=encoder,
        effective_settings={"sampling_rate_hz": 200.0},
        source_summary=None,
        sequence_mode="offsets",
        offsets_source="recording offsets",
        output_dtype="float32",
        paths=paths,
        overwrite=False,
        source_metadata_paths={
            "labels_path": labels,
            "segment_offsets_path": segment_offsets,
            "segment_lengths_path": segment_lengths,
        },
    )

    events = np.load(paths.spike_events_path, allow_pickle=False)
    spike_imu = np.load(paths.spike_imu_path, allow_pickle=False)
    recording_offsets = np.load(paths.recording_offsets_path, allow_pickle=False)
    assert events.shape == (90, 15)
    assert spike_imu.shape == (90, 21)
    np.testing.assert_array_equal(spike_imu[:, :15], events)
    np.testing.assert_array_equal(spike_imu[:, 15:], raw_before[:, 3:9])
    assert summary["spike_imu"]["schema"] == "signed_wavelet_events_plus_imu_v1"
    assert summary["sequence_processing"]["offset_semantics"] == "recording"
    assert summary["sequence_processing"]["offsets_artifact"] == paths.recording_offsets_path.name
    np.testing.assert_array_equal(recording_offsets, [0, 90])
    assert summary["settings"]["padding_mode"] == "reflect"
    assert summary["settings"]["iir_delay_compensated"] is False
    assert summary["alignment"]["segment_offsets_reused_without_shift"] is True
    np.testing.assert_array_equal(np.load(raw_path, allow_pickle=False), raw_before)
    for path, contents in sidecars_before.items():
        assert path.read_bytes() == contents
