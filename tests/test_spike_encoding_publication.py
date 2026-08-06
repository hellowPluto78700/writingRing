from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from writingring.spike_encoding.contracts import SpikeEncodingError, SpikeEncodingSequenceResult
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
