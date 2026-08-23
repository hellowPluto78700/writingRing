from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.reconstruct_padded_spike_accel import (
    process_package,
    reconstruct_segment_events,
    reconstruction_kernels,
)
from scripts.reconstruction_schema import (
    POLARITY_SPLIT_EVENT_CONTRACT,
    POLARITY_SPLIT_WAVELET_SCHEMA,
    SIGNED_EVENT_CONTRACT,
    SIGNED_WAVELET_SCHEMA,
    extract_signed_event_channels,
    resolve_event_schema_contract,
)


def _encoder_spec() -> tuple[dict[str, object], str]:
    spec: dict[str, object] = {
        "schema": "custom_wavelet_encoder_spec_v1",
        "frequencies_hz": [1.0, 2.0, 4.0, 8.0, 16.0],
        "wavelet_widths_samples": [8, 4, 2, 1, 1],
    }
    encoded = json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return spec, hashlib.sha256(encoded).hexdigest()


def _polarity_split(signed_events: np.ndarray, imu: np.ndarray) -> np.ndarray:
    stored_events = np.empty(
        (*signed_events.shape[:-1], 30),
        dtype=signed_events.dtype,
    )
    stored_events[..., 0::2] = np.maximum(signed_events, 0.0)
    stored_events[..., 1::2] = np.maximum(-signed_events, 0.0)
    return np.concatenate((stored_events, imu), axis=-1)


def test_signed_decoder_preserves_existing_21_channel_layout() -> None:
    rng = np.random.default_rng(11)
    signed_events = rng.normal(size=(17, 15))
    imu = rng.normal(size=(17, 6))
    stored = np.concatenate((signed_events, imu), axis=1)

    decoded = extract_signed_event_channels(stored, SIGNED_EVENT_CONTRACT)

    np.testing.assert_array_equal(decoded, signed_events)


def test_polarity_split_decoder_recovers_signed_events_exactly() -> None:
    rng = np.random.default_rng(23)
    signed_events = rng.normal(size=(17, 15))
    imu = rng.normal(size=(17, 6))
    stored = _polarity_split(signed_events, imu)

    decoded = extract_signed_event_channels(stored, POLARITY_SPLIT_EVENT_CONTRACT)

    np.testing.assert_array_equal(decoded, signed_events)


def test_signed_and_polarity_split_reconstruction_are_equivalent() -> None:
    rng = np.random.default_rng(37)
    signed_events = rng.normal(size=(23, 15))
    imu = rng.normal(size=(23, 6))
    signed_stored = np.concatenate((signed_events, imu), axis=1)
    polarity_split_stored = _polarity_split(signed_events, imu)
    kernels = reconstruction_kernels(
        wavelet_widths_samples=(8, 4, 2, 1, 1),
        frequencies_hz=(1.0, 2.0, 4.0, 8.0, 16.0),
    )

    signed_reconstruction = reconstruct_segment_events(
        extract_signed_event_channels(signed_stored, SIGNED_EVENT_CONTRACT),
        kernels=kernels,
    )
    polarity_split_reconstruction = reconstruct_segment_events(
        extract_signed_event_channels(
            polarity_split_stored,
            POLARITY_SPLIT_EVENT_CONTRACT,
        ),
        kernels=kernels,
    )

    np.testing.assert_allclose(
        polarity_split_reconstruction,
        signed_reconstruction,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("source_schema", "source_channels", "padded_schema", "padded_channels"),
    (
        (SIGNED_WAVELET_SCHEMA, 21, SIGNED_WAVELET_SCHEMA, 21),
        (SIGNED_WAVELET_SCHEMA, 21, POLARITY_SPLIT_WAVELET_SCHEMA, 36),
        (POLARITY_SPLIT_WAVELET_SCHEMA, 36, POLARITY_SPLIT_WAVELET_SCHEMA, 36),
    ),
)
def test_padded_reconstruction_accepts_independent_source_and_padded_schemas(
    tmp_path: Path,
    source_schema: str,
    source_channels: int,
    padded_schema: str,
    padded_channels: int,
) -> None:
    rng = np.random.default_rng(53)
    valid_length = 23
    target_length = 28
    signed_events = rng.normal(size=(valid_length, 15))
    imu = rng.normal(size=(valid_length, 6))
    source_contract = resolve_event_schema_contract(source_schema, source_channels)
    padded_contract = resolve_event_schema_contract(padded_schema, padded_channels)

    def stored_values(contract) -> np.ndarray:
        if contract is SIGNED_EVENT_CONTRACT:
            return np.concatenate((signed_events, imu), axis=1)
        return _polarity_split(signed_events, imu)

    source_root = tmp_path / "segmentation"
    padded_root = tmp_path / "segmentation_padded"
    source_dir = source_root / "user_0" / "action_0"
    padded_dir = padded_root / "user_0" / "action_0"
    source_dir.mkdir(parents=True)
    padded_dir.mkdir(parents=True)
    stem = "user_0_action_0"
    encoder_spec, encoder_hash = _encoder_spec()

    source_values = stored_values(source_contract)
    padded_values = np.zeros((1, target_length, padded_channels), dtype=np.float64)
    padded_values[0, :valid_length] = stored_values(padded_contract)
    np.save(source_dir / f"{stem}_spikeIMU.npy", source_values)
    np.save(source_dir / f"{stem}_labels.npy", np.array(["A"], dtype=str))
    np.save(source_dir / f"{stem}_segment_offsets.npy", np.array([0, valid_length]))
    np.save(source_dir / f"{stem}_segment_lengths.npy", np.array([valid_length]))
    np.save(padded_dir / f"{stem}_paddedSpikeIMU.npy", padded_values)
    np.save(padded_dir / f"{stem}_labels.npy", np.array(["A"], dtype=str))
    np.save(padded_dir / f"{stem}_valid_lengths.npy", np.array([valid_length]))
    np.save(
        padded_dir / f"{stem}_valid_mask.npy",
        np.arange(target_length)[None, :] < valid_length,
    )

    source_summary = {
        "input_kind": "spike-imu",
        "feature_schema": source_schema,
        "channel_count": source_channels,
        "sampling_rate_hz": 64.0,
        "spike_encoder": encoder_spec,
        "spike_encoder_spec_sha256": encoder_hash,
    }
    (source_dir / f"{stem}_segmentation_summary.json").write_text(
        json.dumps(source_summary), encoding="utf-8"
    )
    padding_summary = {
        "input_kind": "spike-imu",
        "feature_schema": padded_schema,
        "channel_count": padded_channels,
        "target_length": target_length,
        "padding_side": "right",
        "overflow_policy": "skip",
    }
    (padded_dir / f"{stem}_padding_summary.json").write_text(
        json.dumps(padding_summary), encoding="utf-8"
    )
    (padded_dir / f"{stem}_padding_manifest.csv").write_text(
        "segment_index,output_segment_index,exported,original_length,target_length\n"
        f"0,0,true,{valid_length},{target_length}\n",
        encoding="utf-8",
    )

    process_package(
        dataset_root=tmp_path,
        padded_root=padded_root,
        user="user_0",
        action="0",
        source_dir=source_dir,
        stem=stem,
        sampling_rate_override=None,
        output_dtype="float64",
        overwrite=True,
    )

    output = np.load(
        padded_dir / f"{stem}_padded_reconstructed_accel_m_s2.npy",
        allow_pickle=False,
    )
    kernels = reconstruction_kernels(
        wavelet_widths_samples=(8, 4, 2, 1, 1),
        frequencies_hz=(1.0, 2.0, 4.0, 8.0, 16.0),
    )
    expected = reconstruct_segment_events(signed_events, kernels=kernels)
    np.testing.assert_allclose(output[0, :valid_length], expected)
    np.testing.assert_array_equal(output[0, valid_length:], 0.0)

    metadata = json.loads(
        (padded_dir / f"{stem}_padded_reconstructed_accel_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["source"]["input_feature_schema"] == source_schema
    assert metadata["source"]["stored_event_channel_count"] == (
        source_contract.stored_event_channel_count
    )
    assert metadata["padding"]["input_feature_schema"] == padded_schema
    assert metadata["padding"]["stored_channel_count"] == padded_channels
