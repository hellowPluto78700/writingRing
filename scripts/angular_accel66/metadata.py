from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import (
    DERIVATIVE_METHOD,
    OUTPUT_ENCODER_SCHEMA,
    OUTPUT_EVENT_SCHEMA,
    OUTPUT_SCHEMA,
    POLARITY_SPLIT_WAVELET_SPIKE_IMU_FEATURE_SCHEMA,
    RATE_HZ,
    SPIKE_IMU_TRANSIENT_CHANNEL_NAMES,
    BuildError,
    canonical_sha256,
    sha256_file,
)


def combined_names(source_names: Sequence[str], angular_names: Sequence[str]) -> list[str]:
    return (
        list(source_names[:30])
        + [f"angular_acceleration_{name}" for name in angular_names]
        + list(source_names[30:])
    )


def composite_spec(
    source_spec: Mapping[str, Any], source_hash: str, encoder: Any,
    event_names: Sequence[str],
) -> tuple[dict[str, Any], str]:
    spec = {
        "schema": OUTPUT_ENCODER_SCHEMA,
        "sampling_rate_hz": RATE_HZ,
        "event_channel_count": 60,
        "event_channel_names": list(event_names),
        "linear_acceleration_event_slice": [0, 30],
        "angular_acceleration_event_slice": [30, 60],
        "linear_acceleration_branch": {
            "spike_encoder": dict(source_spec),
            "spike_encoder_spec_sha256": source_hash,
        },
        "angular_acceleration_branch": {
            "derivative": {
                "schema": "angular_acceleration_derivative_v1",
                "method": DERIVATIVE_METHOD,
                "sampling_rate_hz": RATE_HZ,
                "sample_period_seconds": 1.0 / RATE_HZ,
                "edge_order": 2,
                "source_unit": "rad/s",
                "output_unit": "rad/s^2",
                "scope": "complete_recording_before_segmentation",
                "length_preserved": True,
            },
            "spike_encoder": encoder.canonical_encoder_spec,
            "spike_encoder_spec_sha256": encoder.canonical_encoder_spec_sha256,
        },
    }
    return spec, canonical_sha256(spec)


def resolve_timestamp_path(
    metadata: Mapping[str, Any], metadata_path: Path
) -> tuple[Path, str]:
    source = metadata.get("source") if isinstance(metadata.get("source"), Mapping) else {}
    refs = (
        source.get("metadata_references")
        if isinstance(source, Mapping)
        and isinstance(source.get("metadata_references"), Mapping)
        else {}
    )
    candidates = [
        metadata.get("timestamps_path"), metadata.get("timestamp_source_path"),
        source.get("timestamps_path") if isinstance(source, Mapping) else None,
        source.get("timestamp_source_path") if isinstance(source, Mapping) else None,
        refs.get("timestamp_source_path") if isinstance(refs, Mapping) else None,
    ]
    digest = metadata.get("timestamps_sha256")
    if not isinstance(digest, str) and isinstance(source, Mapping):
        digest = source.get("timestamps_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise BuildError("source metadata lacks timestamp SHA-256")
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = metadata_path.parent / path
        path = path.resolve()
        if path.is_file() and sha256_file(path) == digest:
            return path, digest
    raise BuildError(f"could not resolve verified canonical timestamps from {metadata_path}")


def patch_recording_metadata(
    metadata: Mapping[str, Any], *, metadata_path: Path, source_values_path: Path,
    output_dir: Path, values: Any, names: Sequence[str], units: Sequence[str],
    spec: Mapping[str, Any], spec_hash: str, source_spec: Mapping[str, Any],
    source_hash: str, encoder: Any,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(metadata))
    timestamp_path, timestamp_hash = resolve_timestamp_path(metadata, metadata_path)
    section = payload["spike_imu"]
    section.update({
        "schema": OUTPUT_SCHEMA, "channel_count": 66, "event_channel_count": 60,
        "event_representation": "unsigned", "event_feature_schema": OUTPUT_EVENT_SCHEMA,
        "channel_names": list(names), "units": list(units), "sample_count": len(values),
        "sha256": sha256_file(output_dir / "spikeIMU.npy"),
        "trailing_imu_channel_count": 6,
    })
    payload.update({
        "feature_schema": OUTPUT_SCHEMA, "channel_count": 66, "event_channel_count": 60,
        "event_representation": "unsigned", "event_feature_schema": OUTPUT_EVENT_SCHEMA,
        "sampling_rate_hz": RATE_HZ, "timestamps_path": str(timestamp_path),
        "timestamp_source_path": str(timestamp_path), "timestamps_sha256": timestamp_hash,
        "spike_imu_sha256": section["sha256"], "spike_encoder": dict(spec),
        "spike_encoder_spec_sha256": spec_hash,
    })
    encoder_section = payload.get("encoder") if isinstance(payload.get("encoder"), dict) else {}
    encoder_section.update({
        "name": "linear-angular-custom-wavelet", "spike_encoder": dict(spec),
        "spike_encoder_spec_sha256": spec_hash,
    })
    payload["encoder"] = encoder_section
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    output.update({
        "sample_count": len(values), "channel_count": 60,
        "channel_names": list(names[:60]), "dtype": values.dtype.name,
        "event_representation": "linear_and_angular_accel_polarity_split_sparse_wavelet_extrema",
        "post_encode_transform": "PolaritySplitAbs",
        "sha256": sha256_file(output_dir / "spikes.npy"),
    })
    payload["output"] = output
    payload["angular_acceleration"] = {
        "schema": "angular_acceleration_recording_v1", "method": DERIVATIVE_METHOD,
        "sampling_rate_hz": RATE_HZ, "unit": "rad/s^2",
        "artifact": "angularAcceleration.npy",
        "sha256": sha256_file(output_dir / "angularAcceleration.npy"),
        "event_channel_slice": [30, 60], "scope": "complete_recording_before_segmentation",
    }
    payload["derived_from"] = {
        "source_feature_schema": POLARITY_SPLIT_WAVELET_SPIKE_IMU_FEATURE_SCHEMA,
        "source_spike_imu_path": str(source_values_path.resolve()),
        "source_spike_imu_sha256": sha256_file(source_values_path),
        "source_metadata_path": str(metadata_path.resolve()),
        "source_metadata_sha256": sha256_file(metadata_path),
        "alignment_recomputed": False, "segmentation_geometry_recomputed": False,
    }
    payload["linear_acceleration_spike_encoder"] = dict(source_spec)
    payload["linear_acceleration_spike_encoder_spec_sha256"] = source_hash
    payload["angular_acceleration_spike_encoder"] = encoder.canonical_encoder_spec
    payload["angular_acceleration_spike_encoder_spec_sha256"] = (
        encoder.canonical_encoder_spec_sha256
    )
    return payload


def patch_contract(
    payload: dict[str, Any], *, names: Sequence[str], units: Sequence[str],
    spec: Mapping[str, Any], spec_hash: str, source_schema: str,
) -> None:
    payload.update({
        "source_feature_schema": source_schema, "feature_schema": OUTPUT_SCHEMA,
        "channel_count": 66, "event_representation": "unsigned",
        "event_feature_schema": OUTPUT_EVENT_SCHEMA, "event_channel_count": 60,
        "spike_encoder": dict(spec), "spike_encoder_spec_sha256": spec_hash,
        "encoder_spec": dict(spec), "encoder_spec_sha256": spec_hash,
        "channel_names": list(names), "units": list(units),
        "transient_channel_indices": list(range(60, 66)),
        "transient_channel_names": list(SPIKE_IMU_TRANSIENT_CHANNEL_NAMES),
        "event_channel_slice": [0, 60], "acceleration_m_s2_channel_slice": [60, 63],
        "gyroscope_channel_slice": [63, 66],
    })
