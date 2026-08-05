from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from writingring.spike_encoding.contracts import SpikeEncodingError
from writingring.spike_encoding.io import (
    SUPPORTED_METHOD_SEMANTICS,
    load_encoder_settings,
    load_and_validate_timestamps,
    load_sequence_offsets,
    load_spike_encoding_input,
    load_spike_encoding_source_summary,
    resolve_sampling_rate_hz,
    single_array_offsets,
    validate_source_metadata_paths,
)


def _write_imu(path: Path, values: np.ndarray) -> None:
    np.save(path, values, allow_pickle=False)


def test_input_loader_reads_only_valid_nine_channel_finite_data(tmp_path: Path) -> None:
    source = tmp_path / "input_rawIMU.npy"
    values = np.arange(45, dtype=np.float32).reshape(5, 9)
    _write_imu(source, values)

    loaded = load_spike_encoding_input(source)

    assert loaded.raw_imu.flags.writeable is False
    assert loaded.acceleration_g.flags.writeable is False
    np.testing.assert_array_equal(loaded.acceleration_g, values[:, :3])
    np.testing.assert_array_equal(np.load(source, allow_pickle=False), values)


@pytest.mark.parametrize(
    "values, message",
    [
        (np.zeros((2, 6), dtype=np.float32), r"shape \(N, 9\)"),
        (np.array([[np.nan] * 9]), "finite"),
        (np.array([["x"] * 9]), "numeric"),
    ],
)
def test_input_loader_rejects_invalid_imu(values: np.ndarray, message: str, tmp_path: Path) -> None:
    source = tmp_path / "invalid.npy"
    _write_imu(source, values)

    with pytest.raises(SpikeEncodingError, match=message):
        load_spike_encoding_input(source)


def test_offsets_and_settings_are_strictly_validated(tmp_path: Path) -> None:
    offsets_path = tmp_path / "offsets.npy"
    np.save(offsets_path, np.array([0, 2, 5], dtype=np.int64), allow_pickle=False)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"example": 1}), encoding="utf-8")

    np.testing.assert_array_equal(
        load_sequence_offsets(offsets_path, sample_count=5), [0, 2, 5]
    )
    np.testing.assert_array_equal(single_array_offsets(sample_count=5), [0, 5])
    assert load_encoder_settings(settings_path) == {"example": 1}

    np.save(offsets_path, np.array([0, 2, 2, 5], dtype=np.int64), allow_pickle=False)
    with pytest.raises(SpikeEncodingError, match="strictly increasing"):
        load_sequence_offsets(offsets_path, sample_count=5)
    settings_path.write_text("[]", encoding="utf-8")
    with pytest.raises(SpikeEncodingError, match="must be an object"):
        load_encoder_settings(settings_path)


def test_source_metadata_paths_are_optional_read_only_references(tmp_path: Path) -> None:
    labels = tmp_path / "labels.npy"
    np.save(labels, np.array(["a"]))

    paths = validate_source_metadata_paths({"labels_path": labels})

    assert paths["labels_path"] == str(labels.resolve())
    assert paths["segment_offsets_path"] is None
    with pytest.raises(SpikeEncodingError, match="not a regular file"):
        validate_source_metadata_paths({"labels_path": tmp_path / "missing.npy"})


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        (np.array([0.0, 0.005]), r"shape \(3,\)"),
        (np.array([0.0, 0.005, 0.004]), "strictly increasing"),
        (np.array([0.0, 0.01, 0.02]), "inconsistent"),
    ],
)
def test_timestamp_validation_checks_row_alignment_and_cadence(
    tmp_path: Path,
    timestamps: np.ndarray,
    message: str,
) -> None:
    source = tmp_path / "timestamps.npy"
    np.save(source, timestamps, allow_pickle=False)

    with pytest.raises(SpikeEncodingError, match=message):
        load_and_validate_timestamps(
            source, sample_count=3, sampling_rate_hz=200.0
        )


def test_source_summary_resolves_or_rejects_sampling_rate_conflicts(tmp_path: Path) -> None:
    source = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "channel_count": 9,
                "channel_names": [
                    "acceleration_x_g", "acceleration_y_g", "acceleration_z_g",
                    "acceleration_x", "acceleration_y", "acceleration_z",
                    "gyro_x", "gyro_y", "gyro_z",
                ],
                "gravity_removal": {"method": "madgwick", "sampling_rate_hz": 200.0},
                "acceleration_semantics": "gravity_removed_linear_acceleration",
            }
        ),
        encoding="utf-8",
    )

    summary = load_spike_encoding_source_summary(source)
    assert summary.gravity_removal_method == "madgwick"
    assert resolve_sampling_rate_hz({}, summary)["sampling_rate_hz"] == 200.0
    with pytest.raises(SpikeEncodingError, match="conflicts"):
        resolve_sampling_rate_hz({"sampling_rate_hz": 64.0}, summary)
    with pytest.raises(SpikeEncodingError, match="required"):
        resolve_sampling_rate_hz({}, None)


@pytest.mark.parametrize("method, semantics", SUPPORTED_METHOD_SEMANTICS.items())
def test_source_summary_accepts_supported_method_semantics_pairs(
    tmp_path: Path,
    method: str,
    semantics: str,
) -> None:
    source = tmp_path / f"{method}.json"
    source.write_text(
        json.dumps(
            {
                "gravity_removal": {"method": method, "sampling_rate_hz": 200.0},
                "acceleration_semantics": semantics,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_spike_encoding_source_summary(source)

    assert loaded.gravity_removal_method == method
    assert loaded.acceleration_semantics == semantics


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"gravity_removal": {"method": "unknown"}, "acceleration_semantics": "unknown"},
            "must be one of",
        ),
        (
            {"gravity_removal": {"method": "raw"}, "acceleration_semantics": "gravity_removed_linear_acceleration"},
            "does not match",
        ),
        ({"gravity_removal": {"method": "raw"}}, "provided together"),
        ({"acceleration_semantics": "measured_acceleration_with_gravity"}, "provided together"),
    ],
)
def test_source_summary_rejects_incomplete_or_untrusted_semantics(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    source = tmp_path / "invalid_summary.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SpikeEncodingError, match=message):
        load_spike_encoding_source_summary(source)


def test_source_summary_allows_omitted_optional_semantic_provenance(tmp_path: Path) -> None:
    source = tmp_path / "partial_summary.json"
    source.write_text(json.dumps({"gravity_removal": {"sampling_rate_hz": 200.0}}), encoding="utf-8")

    loaded = load_spike_encoding_source_summary(source)

    assert loaded.gravity_removal_method is None
    assert loaded.acceleration_semantics is None
