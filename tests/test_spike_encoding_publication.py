from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from writingring.spike_encoding.contracts import SpikeEncodingError, SpikeEncodingSequenceResult
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


class _PublicationEncoder:
    name = "publication-dummy"
    representation = "signed_dummy_amplitudes"

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


class _OrderedPublicationEncoder(_PublicationEncoder):
    name = "ordered-publication-dummy"

    @property
    def output_metadata(self) -> dict[str, object]:
        return {"channel_order": "encoder_defined_order"}


def _completed_encoding(
    tmp_path: Path,
    *,
    encoder: _PublicationEncoder | None = None,
):
    raw_path = tmp_path / "user_0_action_0_rawIMU.npy"
    acceleration_g = np.arange(18, dtype=np.float32).reshape(6, 3) / 10.0
    raw = np.column_stack(
        (acceleration_g, acceleration_g * 9.80665, np.arange(18, dtype=np.float32).reshape(6, 3))
    )
    np.save(raw_path, raw, allow_pickle=False)
    input_data = load_spike_encoding_input(raw_path)
    encoder = encoder or _PublicationEncoder()
    output = run_spike_encoder(
        encoder,
        acceleration_g=input_data.acceleration_g,
        sequence_offsets=np.array([0, 2, 6], dtype=np.int64),
    )
    paths = spike_encoding_output_paths(
        raw_imu_path=raw_path,
        encoder_name=encoder.name,
        output_stem=None,
        output_root=None,
    )
    return raw, input_data, encoder, output, paths


def test_publication_writes_verified_independent_artifacts(tmp_path: Path) -> None:
    raw, input_data, encoder, output, paths = _completed_encoding(tmp_path)

    summary = publish_spike_encoding(
        output=output,
        input_data=input_data,
        encoder=encoder,
        effective_settings={"sampling_rate_hz": 200.0},
        source_summary=None,
        sequence_mode="offsets",
        offsets_source="provided-offsets.npy",
        output_dtype="float32",
        paths=paths,
        overwrite=False,
    )

    assert paths.output_directory == tmp_path / "publication-dummy" / "user_0_action_0"
    assert np.load(paths.spike_events_path, allow_pickle=False).dtype == np.float32
    assert not paths.spike_imu_path.exists()
    np.testing.assert_array_equal(np.load(paths.sequence_offsets_path, allow_pickle=False), [0, 2, 6])
    with paths.sequences_csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert json.loads(paths.summary_json_path.read_text(encoding="utf-8")) == summary
    assert summary["output"]["polarity_preserved"] is True
    assert "channel_order" not in summary["output"]
    assert summary["sequence_processing"]["state_reset_boundary"] == "sequence"
    assert summary["sequence_processing"]["offset_semantics"] == "sequence"
    assert summary["alignment"]["sample_count_preserved"] is True
    assert "spike_imu" not in summary
    np.testing.assert_array_equal(np.load(input_data.raw_imu_path, allow_pickle=False), raw)


def test_generic_publication_overwrites_compatible_offset_artifacts(tmp_path: Path) -> None:
    _, input_data, encoder, output, paths = _completed_encoding(tmp_path)
    arguments = {
        "output": output,
        "input_data": input_data,
        "encoder": encoder,
        "effective_settings": {"sampling_rate_hz": 200.0},
        "source_summary": None,
        "sequence_mode": "offsets",
        "offsets_source": "provided-offsets.npy",
        "output_dtype": "float32",
        "paths": paths,
        "overwrite": False,
    }
    publish_spike_encoding(**arguments)
    # The intermediate recording-named artifact is an owned legacy artifact,
    # so an overwrite can atomically migrate it back to sequence semantics.
    paths.sequence_offsets_path.rename(paths.recording_offsets_path)
    arguments["overwrite"] = True

    summary = publish_spike_encoding(**arguments)

    assert paths.sequence_offsets_path.is_file()
    assert not paths.recording_offsets_path.exists()
    assert summary["sequence_processing"]["offsets_artifact"] == paths.sequence_offsets_path.name


def test_publication_refuses_unrequested_or_unsafe_overwrite(tmp_path: Path) -> None:
    _, input_data, encoder, output, paths = _completed_encoding(tmp_path)
    arguments = {
        "output": output,
        "input_data": input_data,
        "encoder": encoder,
        "effective_settings": {"sampling_rate_hz": 200.0},
        "source_summary": None,
        "sequence_mode": "offsets",
        "offsets_source": "provided-offsets.npy",
        "output_dtype": "float32",
        "paths": paths,
        "overwrite": False,
    }
    publish_spike_encoding(**arguments)

    with pytest.raises(SpikeEncodingError, match="already exists"):
        publish_spike_encoding(**arguments)
    (paths.output_directory / "user_note.txt").write_text("keep", encoding="utf-8")
    arguments["overwrite"] = True
    with pytest.raises(SpikeEncodingError, match="non-spike-encoding"):
        publish_spike_encoding(**arguments)


def test_publication_preserves_declared_encoder_channel_order(tmp_path: Path) -> None:
    _, input_data, encoder, output, paths = _completed_encoding(
        tmp_path, encoder=_OrderedPublicationEncoder()
    )

    summary = publish_spike_encoding(
        output=output,
        input_data=input_data,
        encoder=encoder,
        effective_settings={"sampling_rate_hz": 200.0},
        source_summary=None,
        sequence_mode="offsets",
        offsets_source="provided-offsets.npy",
        output_dtype="float32",
        paths=paths,
        overwrite=False,
    )

    assert summary["output"]["channel_order"] == "encoder_defined_order"


def test_publication_records_rectified_representation_and_preserves_spike_imu_tail(
    tmp_path: Path,
) -> None:
    acceleration_g = np.column_stack(
        (
            np.sin(np.arange(160) / 3.0),
            np.zeros(160),
            -np.sin(np.arange(160) / 3.0),
        )
    ).astype(np.float32)
    raw = np.column_stack(
        (
            acceleration_g,
            acceleration_g * 9.80665,
            np.arange(160 * 3, dtype=np.float32).reshape(160, 3),
        )
    )

    def publish(root: Path, transform: str | None):
        root.mkdir()
        raw_path = root / "recording_rawIMU.npy"
        np.save(raw_path, raw, allow_pickle=False)
        input_data = load_spike_encoding_input(raw_path)
        encoder = CustomWaveletEncoder(
            CustomWaveletSettings(
                frequencies_hz=(0.5, 1.0, 2.0, 4.0, 8.0),
                post_encode_transform=transform,
                output_dtype="float32",
            )
        )
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
        )
        return raw, paths, summary

    signed_raw, signed_paths, signed_summary = publish(tmp_path / "signed", None)
    rectified_raw, rectified_paths, rectified_summary = publish(
        tmp_path / "rectified", "AbsRectify"
    )
    signed_events = np.load(signed_paths.spike_events_path, allow_pickle=False)
    rectified_events = np.load(rectified_paths.spike_events_path, allow_pickle=False)
    np.testing.assert_array_equal(rectified_events, np.abs(signed_events))
    np.testing.assert_array_equal(rectified_events != 0.0, signed_events != 0.0)
    np.testing.assert_array_equal(
        np.load(rectified_paths.spike_imu_path, allow_pickle=False)[:, 15:],
        rectified_raw[:, 3:9],
    )
    assert signed_raw.shape == rectified_raw.shape == (160, 9)
    assert signed_summary["encoder"]["representation"] == "signed_sparse_wavelet_extrema"
    assert signed_summary["encoder"]["post_encode_transform"] is None
    assert signed_summary["output"]["polarity_preserved"] is True
    assert signed_summary["spike_imu"]["event_representation"] == "signed"
    assert signed_summary["spike_imu"]["event_feature_schema"] == (
        "custom_wavelet_signed_events_v1"
    )
    assert rectified_summary["encoder"]["representation"] == (
        "abs_rectified_sparse_wavelet_extrema"
    )
    assert rectified_summary["encoder"]["event_representation"] == (
        "abs_rectified_sparse_wavelet_extrema"
    )
    assert rectified_summary["encoder"]["post_encode_transform"] == "AbsRectify"
    assert rectified_summary["output"]["polarity_preserved"] is False
    assert rectified_summary["spike_imu"]["schema"] == "signed_wavelet_events_plus_imu_v1"
    assert rectified_summary["spike_imu"]["event_representation"] == "unsigned"
    assert rectified_summary["spike_imu"]["event_feature_schema"] == (
        "custom_wavelet_abs_rectified_events_v1"
    )
    assert rectified_summary["spike_imu"]["post_encode_transform"] == "AbsRectify"
    assert rectified_summary["statistics"]["negative_event_count"] == 0


def test_publication_records_polarity_split_layout_and_preserves_imu_tail(
    tmp_path: Path,
) -> None:
    acceleration_g = np.column_stack(
        (
            np.sin(np.arange(160) / 3.0),
            np.zeros(160),
            -np.sin(np.arange(160) / 3.0),
        )
    ).astype(np.float32)
    raw = np.column_stack(
        (
            acceleration_g,
            acceleration_g * 9.80665,
            np.arange(160 * 3, dtype=np.float32).reshape(160, 3),
        )
    )
    raw_path = tmp_path / "recording_rawIMU.npy"
    np.save(raw_path, raw, allow_pickle=False)
    input_data = load_spike_encoding_input(raw_path)
    encoder = CustomWaveletEncoder(
        CustomWaveletSettings(post_encode_transform="PolaritySplitAbs", output_dtype="float32")
    )
    output = run_spike_encoder(
        encoder,
        acceleration_g=input_data.acceleration_g,
        sequence_offsets=np.array([0, len(raw)], dtype=np.int64),
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
        effective_settings={"sampling_rate_hz": 200.0, "post_encode_transform": "PolaritySplitAbs"},
        source_summary=None,
        sequence_mode="offsets",
        offsets_source="recording offsets",
        output_dtype="float32",
        paths=paths,
        overwrite=False,
    )

    events = np.load(paths.spike_events_path, allow_pickle=False)
    spike_imu = np.load(paths.spike_imu_path, allow_pickle=False)
    assert events.shape == (160, 30)
    assert spike_imu.shape == (160, 36)
    np.testing.assert_array_equal(spike_imu[:, :30], events)
    np.testing.assert_array_equal(spike_imu[:, 30:], raw[:, 3:9])
    assert summary["spike_imu"]["schema"] == "polarity_split_wavelet_events_plus_imu_v1"
    assert summary["spike_imu"]["channel_count"] == 36
    assert summary["spike_imu"]["event_channel_count"] == 30
    assert summary["spike_imu"]["event_representation"] == "unsigned"
    assert summary["spike_imu"]["event_feature_schema"] == (
        "custom_wavelet_polarity_split_abs_events_v1"
    )
    assert summary["spike_imu"]["channel_names"][:2] == ["event_x_0_pos", "event_x_0_neg_abs"]
    assert summary["output"]["polarity_preserved"] is True
