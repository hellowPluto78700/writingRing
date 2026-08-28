from __future__ import annotations

from pathlib import Path
import json
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .common import (
    OUTPUT_EVENT_SCHEMA,
    OUTPUT_SCHEMA,
    POLARITY_SPLIT_WAVELET_SPIKE_IMU_FEATURE_SCHEMA,
    BuildError,
    angular_acceleration_from_gyro,
    angular_encoder,
    combine_event_branches,
    load_json,
    publish_dir,
    read_csv,
    sha256_file,
    spike_root,
    validate_source_recording,
    write_csv,
    write_json,
)
from .metadata import combined_names, composite_spec, patch_contract, patch_recording_metadata


def build_recordings(
    source_root: Path, output_root: Path, user: str, *, overwrite: bool
) -> tuple[dict[int, Path], dict[str, Any], str, list[str], list[str]]:
    src_root, dst_root = spike_root(source_root), spike_root(output_root)
    found: list[tuple[Path, Path]] = []
    for metadata_path in sorted(src_root.rglob("metadata.json")):
        relative = metadata_path.parent.relative_to(src_root)
        if len(relative.parts) >= 3 and relative.parts[-3] == user:
            found.append((relative, metadata_path.parent))
    if not found:
        raise BuildError(f"no recordings found for {user}")

    built: dict[int, Path] = {}
    reference_spec: dict[str, Any] | None = None
    reference_hash: str | None = None
    reference_names: list[str] | None = None
    reference_units: list[str] | None = None
    for relative, source_dir in found:
        dataset_id = int(relative.parts[-1])
        source_values, metadata, source_spec, source_hash, source_names, source_units = (
            validate_source_recording(source_dir)
        )
        source = np.asarray(source_values)
        derivative = angular_acceleration_from_gyro(source[:, -3:])
        encoder = angular_encoder(source.dtype.name)
        encoded = encoder.encode_sequence(derivative)
        angular_events = np.asarray(encoded.values)
        combined = combine_event_branches(source, angular_events)
        names = combined_names(source_names, encoded.channel_names)
        units = list(source_units[:30]) + ["event"] * 30 + list(source_units[30:])
        spec, digest = composite_spec(source_spec, source_hash, encoder, names[:60])
        if reference_hash is None:
            reference_spec, reference_hash = spec, digest
            reference_names, reference_units = names, units
        elif digest != reference_hash:
            raise BuildError(
                f"{user} recordings do not share one composite encoder identity"
            )

        destination = dst_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.angular66.", dir=destination.parent)
        )
        published = False
        try:
            shutil.copytree(source_dir, staging, dirs_exist_ok=True)
            np.save(staging / "spikeIMU.npy", combined, allow_pickle=False)
            np.save(staging / "spikes.npy", combined[:, :60], allow_pickle=False)
            np.save(
                staging / "angularAcceleration.npy",
                derivative.astype(source.dtype, copy=False),
                allow_pickle=False,
            )
            patched = patch_recording_metadata(
                metadata,
                metadata_path=source_dir / "metadata.json",
                source_values_path=source_dir / "spikeIMU.npy",
                output_dir=staging,
                values=combined,
                names=names,
                units=units,
                spec=spec,
                spec_hash=digest,
                source_spec=source_spec,
                source_hash=source_hash,
                encoder=encoder,
            )
            write_json(staging / "metadata.json", patched)
            staged = np.load(staging / "spikeIMU.npy", allow_pickle=False)
            if not np.array_equal(staged[:, :30], source[:, :30]):
                raise BuildError("source linear-acceleration events changed")
            if not np.array_equal(staged[:, 60:], source[:, 30:]):
                raise BuildError("source trailing IMU channels changed")
            publish_dir(staging, destination, overwrite=overwrite)
            published = True
        finally:
            if not published and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        built[dataset_id] = destination / "spikeIMU.npy"

    assert reference_spec is not None and reference_hash is not None
    assert reference_names is not None and reference_units is not None
    return built, reference_spec, reference_hash, reference_names, reference_units


def _exported_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    result: list[Mapping[str, str]] = []
    for row in rows:
        if "exported" in row and row.get("exported", "").strip().lower() not in {"true", "1"}:
            continue
        if row.get("segment_index", "").strip() == "":
            continue
        result.append(row)
    return result


def build_segmentation(
    source_root: Path, output_root: Path, user: str, action: str,
    recordings: Mapping[int, Path], spec: Mapping[str, Any], spec_hash: str,
    names: Sequence[str], units: Sequence[str], *, overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    source = source_root / "segmentation" / user / f"action_{action}"
    destination = output_root / "segmentation" / user / f"action_{action}"
    stem = f"{user}_action_{action}"
    manifest_path = source / f"{stem}_segments.csv"
    summary_path = source / f"{stem}_segmentation_summary.json"
    fields, rows = read_csv(manifest_path)
    selected = sorted(_exported_rows(rows), key=lambda row: int(row["segment_index"]))
    pieces: list[np.ndarray] = []
    for expected_index, row in enumerate(selected):
        if int(row["segment_index"]) != expected_index:
            raise BuildError(f"non-contiguous segment indices for {user}")
        dataset_id = int(row["dataset_id"])
        if dataset_id not in recordings:
            raise BuildError(f"segmentation references missing dataset {dataset_id}")
        start = int(row["start_sample_index"])
        stop = int(row["stop_sample_index_exclusive"])
        values = np.load(recordings[dataset_id], allow_pickle=False, mmap_mode="r")
        if not 0 <= start < stop <= len(values):
            raise BuildError("reused segment slice is outside the derived recording")
        pieces.append(np.asarray(values[start:stop]))
    if not pieces:
        raise BuildError(f"no exported segments found for {user}")
    combined = np.concatenate(pieces, axis=0)
    source_segmented = np.load(
        source / f"{stem}_spikeIMU.npy", allow_pickle=False, mmap_mode="r"
    )
    if source_segmented.shape != (len(combined), 36):
        raise BuildError("source segmentation shape is incompatible")
    if not np.array_equal(combined[:, :30], source_segmented[:, :30]):
        raise BuildError("reused segmentation changed source event values")
    if not np.array_equal(combined[:, 60:], source_segmented[:, 30:]):
        raise BuildError("reused segmentation changed source trailing IMU")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.angular66.", dir=destination.parent)
    )
    published = False
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        np.save(staging / f"{stem}_spikeIMU.npy", combined, allow_pickle=False)
        patched_rows: list[dict[str, str]] = []
        for row in rows:
            item = dict(row)
            for key, value in {
                "feature_schema": OUTPUT_SCHEMA,
                "event_representation": "unsigned",
                "event_feature_schema": OUTPUT_EVENT_SCHEMA,
                "event_channel_count": "60",
                "channel_count": "66",
                "channel_schema": OUTPUT_SCHEMA,
                "channel_names": json.dumps(list(names), separators=(",", ":")),
                "units": json.dumps(list(units), separators=(",", ":")),
                "transient_channel_indices": json.dumps(list(range(60, 66))),
            }.items():
                if key in item:
                    item[key] = value
            dataset_token = item.get("dataset_id", "").strip()
            if dataset_token:
                dataset_id = int(float(dataset_token))
                recording_path = recordings.get(dataset_id)
                if recording_path is None:
                    raise BuildError(
                        f"segmentation manifest references missing dataset {dataset_id}"
                    )
                recording_meta = recording_path.parent / "metadata.json"
                for key, value in {
                    "feature_values_path": str(recording_path.resolve()),
                    "feature_values_sha256": sha256_file(recording_path),
                    "feature_metadata_sha256": sha256_file(recording_meta),
                }.items():
                    if key in item:
                        item[key] = value
            patched_rows.append(item)
        write_csv(staging / manifest_path.name, fields, patched_rows)
        summary = load_json(summary_path)
        source_schema = str(summary.get("feature_schema", ""))
        if source_schema != POLARITY_SPLIT_WAVELET_SPIKE_IMU_FEATURE_SCHEMA:
            raise BuildError("source segmentation is not the expected 36-channel schema")
        patch_contract(
            summary, names=names, units=units, spec=spec, spec_hash=spec_hash,
            source_schema=source_schema,
        )
        summary.update({
            "source_segmentation_directory": str(source.resolve()),
            "segmentation_geometry_reused": True,
            "segmentation_boundaries_recomputed": False,
        })
        write_json(staging / summary_path.name, summary)
        for suffix in ("labels", "segment_offsets", "segment_lengths"):
            old = source / f"{stem}_{suffix}.npy"
            new = staging / f"{stem}_{suffix}.npy"
            if not np.array_equal(np.load(old, allow_pickle=False), np.load(new, allow_pickle=False)):
                raise BuildError(f"segmentation geometry changed: {suffix}")
        targets = source / f"{stem}_board_event_targets.npy"
        if targets.exists() and not np.array_equal(
            np.load(targets, allow_pickle=False),
            np.load(staging / targets.name, allow_pickle=False),
        ):
            raise BuildError("Board-event targets changed")
        publish_dir(staging, destination, overwrite=overwrite)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return destination, summary


def _exact_one(root: Path, pattern: str) -> Path:
    found = sorted(root.glob(pattern))
    if len(found) != 1:
        raise BuildError(f"expected exactly one {pattern!r} under {root}; found {len(found)}")
    return found[0]


def build_padding(
    source_root: Path, output_root: Path, user: str, action: str,
    segmentation_dir: Path, segmentation_summary: Mapping[str, Any], *, overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    source = source_root / "segmentation_padded" / user / f"action_{action}"
    destination = output_root / "segmentation_padded" / user / f"action_{action}"
    padded_path = _exact_one(source, "*_paddedSpikeIMU.npy")
    lengths_path = _exact_one(source, "*_valid_lengths.npy")
    mask_path = _exact_one(source, "*_valid_mask.npy")
    labels_path = _exact_one(source, "*_labels.npy")
    summary_path = _exact_one(source, "*_padding_summary.json")
    manifest_path = _exact_one(source, "*_padding_manifest.csv")
    old = np.load(padded_path, allow_pickle=False, mmap_mode="r")
    valid_lengths = np.load(lengths_path, allow_pickle=False)
    valid_mask = np.load(mask_path, allow_pickle=False)
    if old.ndim != 3 or old.shape[2] != 36:
        raise BuildError("source padded SpikeIMU must have shape (S, T, 36)")
    if valid_lengths.shape != (len(old),) or valid_mask.shape != old.shape[:2]:
        raise BuildError("source padded geometry is incompatible")
    padding_summary = load_json(summary_path)
    try:
        padding_value = old.dtype.type(float(padding_summary["padding_value"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise BuildError("source padding summary has invalid padding_value") from exc

    stem = f"{user}_action_{action}"
    segmented = np.load(
        segmentation_dir / f"{stem}_spikeIMU.npy", allow_pickle=False, mmap_mode="r"
    )
    offsets = np.load(segmentation_dir / f"{stem}_segment_offsets.npy", allow_pickle=False)
    segment_lengths = np.load(
        segmentation_dir / f"{stem}_segment_lengths.npy", allow_pickle=False
    )
    fields, rows = read_csv(manifest_path)
    output = np.full(old.shape[:2] + (66,), padding_value, dtype=old.dtype)
    output[:, :, :30] = old[:, :, :30]
    output[:, :, 60:] = old[:, :, 30:]
    exported_count = 0
    for row in rows:
        if row.get("exported", "").strip().lower() not in {"true", "1"}:
            continue
        segment_index = int(row["segment_index"])
        output_index = int(row["output_segment_index"])
        length = int(valid_lengths[output_index])
        if length != int(segment_lengths[segment_index]):
            raise BuildError("padding valid length disagrees with reused segment length")
        start, stop = int(offsets[segment_index]), int(offsets[segment_index + 1])
        output[output_index, :length, 30:60] = segmented[start:stop, 30:60]
        exported_count += 1
    if exported_count != len(output):
        raise BuildError("padding manifest exported-row count is inconsistent")
    if np.any(output[:, :, 30:60][~valid_mask] != padding_value):
        raise BuildError("angular-event padding changed source padding geometry")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.angular66.", dir=destination.parent)
    )
    published = False
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        np.save(staging / padded_path.name, output, allow_pickle=False)
        patched_rows: list[dict[str, str]] = []
        for row in rows:
            item = dict(row)
            for key, value in {
                "feature_schema": OUTPUT_SCHEMA,
                "event_representation": "unsigned",
                "event_feature_schema": OUTPUT_EVENT_SCHEMA,
                "event_channel_count": "60",
                "channel_count": "66",
                "source_input_directory": str(segmentation_dir.resolve()),
            }.items():
                if key in item:
                    item[key] = value
            patched_rows.append(item)
        write_csv(staging / manifest_path.name, fields, patched_rows)
        source_schema = str(padding_summary.get("feature_schema", ""))
        if source_schema != POLARITY_SPLIT_WAVELET_SPIKE_IMU_FEATURE_SCHEMA:
            raise BuildError("source padding summary is not the expected 36-channel schema")
        patch_contract(
            padding_summary,
            names=segmentation_summary["channel_names"],
            units=segmentation_summary["units"],
            spec=segmentation_summary["spike_encoder"],
            spec_hash=segmentation_summary["spike_encoder_spec_sha256"],
            source_schema=source_schema,
        )
        padding_summary.update({
            "source_padding_directory": str(source.resolve()),
            "padding_geometry_reused": True,
            "padding_target_recomputed": False,
        })
        write_json(staging / summary_path.name, padding_summary)
        for geometry in (lengths_path, mask_path, labels_path):
            if not np.array_equal(
                np.load(geometry, allow_pickle=False),
                np.load(staging / geometry.name, allow_pickle=False),
            ):
                raise BuildError(f"padding geometry changed: {geometry.name}")
        publish_dir(staging, destination, overwrite=overwrite)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return destination, padding_summary
