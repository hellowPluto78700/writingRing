#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TRAILING_IMU_CHANNEL_COUNT = 6


class ReuseError(RuntimeError):
    pass


def refresh_alignment_outcome_manifests(
    dest_alignment: Path,
) -> dict[str, int]:
    """Refresh alignment outcome artifact digests after provenance rebinding.

    patch_alignment_tree() changes TXT/JSON alignment artifacts because the
    feature values/metadata hashes change with the new wavelet encoder.

    The alignment outcome report contains SHA-256 digests of the corresponding
    offset/skip/verification artifacts. Those manifest digests therefore need
    to be recomputed after rebinding.
    """

    reports_root = dest_alignment / "reports"
    offsets_root = dest_alignment / "offsets"
    verification_root = dest_alignment / "verification"

    if not reports_root.is_dir():
        raise ReuseError(
            f"alignment reports directory is missing: {reports_root}"
        )

    report_count = 0
    artifact_count = 0

    for report_path in sorted(reports_root.rglob("*.json")):
        report = load_json(report_path)

        entries = report.get("outcome_artifacts")

        # Only completed outcome reports have an artifact manifest.
        if entries is None:
            continue

        if not isinstance(entries, list):
            raise ReuseError(
                f"alignment report outcome_artifacts must be a list: "
                f"{report_path}"
            )

        try:
            relative_parent = report_path.relative_to(reports_root).parent
        except ValueError as exc:
            raise ReuseError(
                f"alignment report is outside reports root: {report_path}"
            ) from exc

        candidate_dirs = (
            offsets_root / relative_parent,
            verification_root / relative_parent,
        )

        for entry in entries:
            if not isinstance(entry, dict):
                raise ReuseError(
                    f"invalid outcome_artifacts entry: {report_path}"
                )

            filename = entry.get("filename")

            if not isinstance(filename, str) or not filename:
                raise ReuseError(
                    f"alignment artifact manifest is missing filename: "
                    f"{report_path}"
                )

            candidates = [
                directory / filename
                for directory in candidate_dirs
                if (directory / filename).is_file()
            ]

            if len(candidates) != 1:
                raise ReuseError(
                    f"expected exactly one manifested alignment artifact "
                    f"{filename!r} for {report_path}, got "
                    f"{[str(path) for path in candidates]}"
                )

            artifact_path = candidates[0]

            entry["sha256"] = sha256_file(artifact_path)
            artifact_count += 1

        temporary_path = report_path.with_name(
            f".{report_path.name}.rebind.tmp"
        )

        temporary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        os.replace(temporary_path, report_path)

        report_count += 1

    if report_count == 0:
        raise ReuseError(
            f"no alignment outcome reports were refreshed under "
            f"{reports_root}"
        )

    return {
        "reports_refreshed": report_count,
        "artifact_digests_refreshed": artifact_count,
    }


@dataclass(frozen=True)
class Record:
    user: str
    action_token: str
    dataset_id: str
    directory: Path
    values_path: Path
    metadata_path: Path
    metadata: dict[str, Any]

    @property
    def key(self) -> tuple[str, str]:
        return self.user, self.dataset_id


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReuseError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReuseError(f"JSON root must be an object: {path}")
    return value


def recursive_values(obj: Any, key_name: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == key_name:
                found.append(value)
            found.extend(recursive_values(value, key_name))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(recursive_values(value, key_name))
    return found


def first_recursive(obj: Any, names: Iterable[str]) -> Any:
    for name in names:
        values = recursive_values(obj, name)
        if values:
            return values[0]
    return None


def action_matches(token: str, action: str) -> bool:
    return token in {action, f"action_{action}"}


def spike_root(combination_root: Path) -> Path:
    return combination_root / "spikeEncoding" / "custom-wavelet"


def discover_records(combination_root: Path, action: str) -> dict[tuple[str, str], Record]:
    root = spike_root(combination_root)
    if not root.is_dir():
        raise ReuseError(f"missing spike root: {root}")
    result: dict[tuple[str, str], Record] = {}
    for metadata_path in sorted(root.rglob("metadata.json")):
        directory = metadata_path.parent
        try:
            rel = directory.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) < 3:
            continue
        user, action_token, dataset_id = rel.parts[-3:]
        if not action_matches(action_token, action):
            continue
        values_path = directory / "spikeIMU.npy"
        if not values_path.is_file():
            raise ReuseError(f"missing spikeIMU.npy beside {metadata_path}")
        metadata = load_json(metadata_path)
        key = (user, dataset_id)
        if key in result:
            raise ReuseError(f"duplicate SpikeIMU record key: {key}")
        result[key] = Record(
            user=user,
            action_token=action_token,
            dataset_id=dataset_id,
            directory=directory,
            values_path=values_path,
            metadata_path=metadata_path,
            metadata=metadata,
        )
    if not result:
        raise ReuseError(f"no custom-wavelet records found for action {action} under {root}")
    return result


def encoder_spec(meta: dict[str, Any]) -> dict[str, Any]:
    spec = meta.get("spike_encoder")
    if not isinstance(spec, dict):
        raise ReuseError("metadata is missing top-level spike_encoder object")
    digest = meta.get("spike_encoder_spec_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ReuseError("metadata is missing a valid spike_encoder_spec_sha256")
    return spec


def encoder_hash(meta: dict[str, Any]) -> str:
    encoder_spec(meta)
    return str(meta["spike_encoder_spec_sha256"])


def spike_layout(meta: dict[str, Any]) -> tuple[int, int]:
    """Return validated total/event channel counts for signed or polarity split output."""
    spike = meta.get("spike_imu")
    if not isinstance(spike, dict):
        raise ReuseError("metadata is missing top-level spike_imu object")
    channels = spike.get("channel_count")
    event_channels = spike.get("event_channel_count")
    if (
        not isinstance(channels, int)
        or not isinstance(event_channels, int)
        or channels <= TRAILING_IMU_CHANNEL_COUNT
        or event_channels + TRAILING_IMU_CHANNEL_COUNT != channels
    ):
        raise ReuseError("metadata has an invalid SpikeIMU channel layout")
    return channels, event_channels


def frequencies(meta: dict[str, Any]) -> list[float]:
    spec = encoder_spec(meta)
    value = spec.get("frequencies_hz")
    if not isinstance(value, list) or len(value) != 5:
        raise ReuseError("spike_encoder.frequencies_hz must contain five values")
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise ReuseError("spike_encoder.frequencies_hz must be numeric") from exc


def strip_frequency_scale_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            lowered = key.lower()
            if "frequenc" in lowered or "width" in lowered:
                continue
            out[key] = strip_frequency_scale_fields(child)
        return out
    if isinstance(value, list):
        return [strip_frequency_scale_fields(x) for x in value]
    return value


def timestamp_hash(meta: dict[str, Any]) -> str | None:
    value = first_recursive(meta, ("timestamps_sha256", "timestamp_sha256"))
    return value if isinstance(value, str) else None


def check_uniform_source_records(records: dict[tuple[str, str], Record]) -> tuple[str, list[float]]:
    first = next(iter(records.values()))
    base_hash = encoder_hash(first.metadata)
    base_freq = frequencies(first.metadata)
    for record in records.values():
        if encoder_hash(record.metadata) != base_hash:
            raise ReuseError(
                f"source action mixes encoder identities: {record.user}/{record.dataset_id}"
            )
        if frequencies(record.metadata) != base_freq:
            raise ReuseError(
                f"source action mixes frequency lists: {record.user}/{record.dataset_id}"
            )
        channels, _ = spike_layout(record.metadata)
        values = np.load(record.values_path, allow_pickle=False, mmap_mode="r")
        if values.ndim != 2 or values.shape[1] != channels or values.shape[0] == 0:
            raise ReuseError(f"invalid SpikeIMU shape {values.shape}: {record.values_path}")
    return base_hash, base_freq


def find_alignment_report(root: Path, record: Record, action: str) -> Path:
    reports_root = root / "alignment" / "reports"
    candidates = []
    for path in reports_root.rglob("*.json") if reports_root.is_dir() else []:
        text = path.as_posix()
        if record.user in path.parts and record.dataset_id in path.name and (
            f"action_{action}" in path.parts or action in path.parts or f"action_{action}" in text
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise ReuseError(
            f"expected one alignment report for {record.user}/{record.dataset_id}, got {len(candidates)}"
        )
    return candidates[0]


def validate_alignment_reusable(root: Path, records: dict[tuple[str, str], Record], action: str) -> None:
    offsets_root = root / "alignment" / "offsets"
    reports_root = root / "alignment" / "reports"
    if not offsets_root.is_dir() or not reports_root.is_dir():
        raise ReuseError("aligned-board reuse requires alignment/offsets and alignment/reports")
    for record in records.values():
        report_path = find_alignment_report(root, record, action)
        report = load_json(report_path)
        used_values = recursive_values(report, "spike_event_channels_used")
        if not used_values or any(value is not False for value in used_values):
            raise ReuseError(
                f"alignment report does not prove spike_event_channels_used=false: {report_path}"
            )
        channel_values = recursive_values(report, "transient_channel_indices")
        normalized = []
        for value in channel_values:
            if isinstance(value, list):
                try:
                    normalized.append([int(x) for x in value])
                except (TypeError, ValueError):
                    pass
        channels, _ = spike_layout(record.metadata)
        expected_transient = list(range(channels - TRAILING_IMU_CHANNEL_COUNT, channels))
        if expected_transient not in normalized:
            raise ReuseError(
                f"alignment report does not prove transient channels {expected_transient}: {report_path}"
            )


def source_segmentation_dir(root: Path, user: str, action: str) -> Path:
    return root / "segmentation" / user / f"action_{action}"


def require_source_segmentation(root: Path, records: dict[tuple[str, str], Record], action: str) -> None:
    users = sorted({record.user for record in records.values()})
    for user in users:
        directory = source_segmentation_dir(root, user, action)
        stem = f"{user}_action_{action}"
        for suffix in (
            "_spikeIMU.npy",
            "_labels.npy",
            "_segment_offsets.npy",
            "_segment_lengths.npy",
            "_segments.csv",
            "_segmentation_summary.json",
        ):
            path = directory / f"{stem}{suffix}"
            if not path.is_file():
                raise ReuseError(f"missing reusable segmentation artifact: {path}")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = args.source_root.resolve()
    if not root.is_dir():
        raise ReuseError(f"source root is not a directory: {root}")
    preprocessed = root / "preprocessedIMU"
    if not preprocessed.is_dir() or not any(preprocessed.rglob("*_preprocessedIMU.npy")):
        raise ReuseError(f"source root has no reusable preprocessedIMU: {preprocessed}")
    records = discover_records(root, args.action)
    source_hash, source_freq = check_uniform_source_records(records)
    require_source_segmentation(root, records, args.action)
    if args.boundary_mode == "aligned-board-events":
        validate_alignment_reusable(root, records, args.action)
    return {
        "status": "PASS",
        "source_root": str(root),
        "record_count": len(records),
        "user_count": len({r.user for r in records.values()}),
        "source_encoder_hash": source_hash,
        "source_frequencies_hz": source_freq,
        "boundary_mode": args.boundary_mode,
        "alignment_reusable": args.boundary_mode == "aligned-board-events",
        "segmentation_geometry_reusable": True,
        "preprocessed_imu_reusable": True,
        "source_has_padding": (root / "segmentation_padded").is_dir(),
    }


def compare_new_encoding(
    source_records: dict[tuple[str, str], Record],
    dest_records: dict[tuple[str, str], Record],
    requested_freq: list[float],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
    if set(source_records) != set(dest_records):
        missing = sorted(set(source_records) - set(dest_records))
        extra = sorted(set(dest_records) - set(source_records))
        raise ReuseError(f"new encoding record set differs; missing={missing}, extra={extra}")
    replacements: dict[str, tuple[str, str]] = {}
    rows: list[dict[str, Any]] = []
    for key in sorted(source_records):
        old = source_records[key]
        new = dest_records[key]
        if frequencies(new.metadata) != requested_freq:
            raise ReuseError(
                f"new frequencies mismatch for {new.user}/{new.dataset_id}: {frequencies(new.metadata)}"
            )
        if strip_frequency_scale_fields(encoder_spec(old.metadata)) != strip_frequency_scale_fields(
            encoder_spec(new.metadata)
        ):
            raise ReuseError(
                f"new encoder changed non-frequency semantics for {new.user}/{new.dataset_id}"
            )
        old_ts = timestamp_hash(old.metadata)
        new_ts = timestamp_hash(new.metadata)
        if old_ts is None or new_ts is None or old_ts != new_ts:
            raise ReuseError(
                f"timestamp provenance changed for {new.user}/{new.dataset_id}: {old_ts} -> {new_ts}"
            )
        old_values = np.load(old.values_path, allow_pickle=False, mmap_mode="r")
        new_values = np.load(new.values_path, allow_pickle=False, mmap_mode="r")
        old_channels, old_events = spike_layout(old.metadata)
        new_channels, new_events = spike_layout(new.metadata)
        if old_values.ndim != 2 or new_values.ndim != 2 or old_values.shape[0] != new_values.shape[0]:
            raise ReuseError(
                f"SpikeIMU row count changed for {new.user}/{new.dataset_id}: {old_values.shape} -> {new_values.shape}"
            )
        if old_values.shape[1] != old_channels or new_values.shape[1] != new_channels:
            raise ReuseError(f"unexpected SpikeIMU layout: {old_values.shape} -> {new_values.shape}")
        if not np.array_equal(old_values[:, old_events:], new_values[:, new_events:]):
            raise ReuseError(
                f"trailing IMU channels changed for {new.user}/{new.dataset_id}; alignment reuse is unsafe"
            )
        old_values_hash = sha256_file(old.values_path)
        new_values_hash = sha256_file(new.values_path)
        old_meta_hash = sha256_file(old.metadata_path)
        new_meta_hash = sha256_file(new.metadata_path)
        for old_hash, new_hash, label in (
            (old_values_hash, new_values_hash, "values"),
            (old_meta_hash, new_meta_hash, "metadata"),
        ):
            if old_hash in replacements and replacements[old_hash][0] != new_hash:
                raise ReuseError(f"ambiguous {label} hash replacement for {old_hash}")
            replacements[old_hash] = (new_hash, f"{new.user}/{new.dataset_id}:{label}")
        rows.append(
            {
                "user": new.user,
                "dataset_id": new.dataset_id,
                "old_encoder_hash": encoder_hash(old.metadata),
                "new_encoder_hash": encoder_hash(new.metadata),
                "old_values_sha256": old_values_hash,
                "new_values_sha256": new_values_hash,
                "old_metadata_sha256": old_meta_hash,
                "new_metadata_sha256": new_meta_hash,
                "timestamp_sha256": new_ts,
                "trailing_imu_equal": True,
            }
        )
    return rows, replacements


def patch_alignment_tree(dest_alignment: Path, replacements: dict[str, tuple[str, str]]) -> dict[str, int]:
    counts = {old: 0 for old in replacements}
    for path in sorted(dest_alignment.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".json", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        changed = text
        for old_hash, (new_hash, _label) in replacements.items():
            occurrences = changed.count(old_hash)
            if occurrences:
                changed = changed.replace(old_hash, new_hash)
                counts[old_hash] += occurrences
        if changed != text:
            path.write_text(changed, encoding="utf-8")
    return counts


def rebind_alignment(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()

    source_records = discover_records(source_root, args.action)
    dest_records = discover_records(dest_root, args.action)

    requested = [float(x) for x in args.frequencies]

    rows, replacements = compare_new_encoding(
        source_records,
        dest_records,
        requested,
    )

    source_alignment = source_root / "alignment"
    dest_alignment = dest_root / "alignment"

    if not source_alignment.is_dir():
        raise ReuseError(
            f"missing source alignment directory: {source_alignment}"
        )

    if dest_alignment.exists():
        if not args.overwrite:
            raise ReuseError(
                f"destination alignment already exists: {dest_alignment}"
            )

        shutil.rmtree(dest_alignment)

    # Start from the completed source alignment result.
    shutil.copytree(source_alignment, dest_alignment)

    # Replace the old complete SpikeIMU / metadata provenance hashes with
    # hashes for the newly encoded wavelet variant.
    counts = patch_alignment_tree(
        dest_alignment,
        replacements,
    )

    unused = [
        label
        for old, (new, label) in replacements.items()
        if old != new and counts[old] == 0
    ]

    if unused:
        raise ReuseError(
            "alignment provenance did not contain expected source hashes for: "
            + ", ".join(unused)
        )

    # Ensure stale old provenance is completely gone.
    for old_hash, (new_hash, _label) in replacements.items():
        if old_hash == new_hash:
            continue

        for path in dest_alignment.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in {".txt", ".json", ".csv"}
            ):
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue

                if old_hash in text:
                    raise ReuseError(
                        f"old provenance hash remains after rebind: {path}"
                    )

    # IMPORTANT:
    #
    # patch_alignment_tree() changes alignment offset TXT / skip JSON
    # contents. Completed alignment reports contain SHA-256 manifests for
    # those files. Refresh those artifact digests before segmentation loads
    # the alignment outcome.
    manifest_refresh = refresh_alignment_outcome_manifests(
        dest_alignment
    )

    report = {
        "status": "PASS",
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "requested_frequencies_hz": requested,
        "recordings": rows,
        "alignment_source": str(source_alignment),
        "alignment_destination": str(dest_alignment),
        "alignment_recomputed": False,
        "alignment_provenance_rebound": True,
        "alignment_manifest_refresh": manifest_refresh,
        "replacement_occurrences": {
            replacements[k][1]: counts[k]
            for k in counts
        },
    }

    out = dest_root / "wavelet_alignment_reuse.json"

    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return report


def compare_npy(a: Path, b: Path, label: str) -> dict[str, Any]:
    if not a.is_file() or not b.is_file():
        raise ReuseError(f"missing {label}: {a} or {b}")
    left = np.load(a, allow_pickle=False)
    right = np.load(b, allow_pickle=False)
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ReuseError(f"{label} differs: {a} vs {b}")
    return {"shape": list(left.shape), "equal": True}


def compare_segment_geometry(source_root: Path, dest_root: Path, action: str, users: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for user in users:
        src = source_root / "segmentation" / user / f"action_{action}"
        dst = dest_root / "segmentation" / user / f"action_{action}"
        stem = f"{user}_action_{action}"
        checks = {
            "labels": compare_npy(src / f"{stem}_labels.npy", dst / f"{stem}_labels.npy", "labels"),
            "offsets": compare_npy(
                src / f"{stem}_segment_offsets.npy",
                dst / f"{stem}_segment_offsets.npy",
                "segment offsets",
            ),
            "lengths": compare_npy(
                src / f"{stem}_segment_lengths.npy",
                dst / f"{stem}_segment_lengths.npy",
                "segment lengths",
            ),
        }
        src_targets = src / f"{stem}_board_event_targets.npy"
        dst_targets = dst / f"{stem}_board_event_targets.npy"
        if src_targets.exists() or dst_targets.exists():
            checks["board_event_targets"] = compare_npy(src_targets, dst_targets, "board event targets")
        src_values = np.load(src / f"{stem}_spikeIMU.npy", allow_pickle=False, mmap_mode="r")
        dst_values = np.load(dst / f"{stem}_spikeIMU.npy", allow_pickle=False, mmap_mode="r")
        src_summary = load_json(src / f"{stem}_segmentation_summary.json")
        src_channels = src_summary.get("channel_count")
        dst_channels = load_json(dst / f"{stem}_segmentation_summary.json").get("channel_count")
        if not isinstance(src_channels, int) or not isinstance(dst_channels, int):
            raise ReuseError(f"segmented SpikeIMU schema is missing for {user}")
        if src_values.ndim != 2 or dst_values.ndim != 2 or src_values.shape[0] != dst_values.shape[0]:
            raise ReuseError(f"segmented SpikeIMU row count changed for {user}")
        if src_values.shape[1] != src_channels or dst_values.shape[1] != dst_channels:
            raise ReuseError(f"segmented SpikeIMU layout mismatch for {user}")
        if not np.array_equal(src_values[:, -TRAILING_IMU_CHANNEL_COUNT:], dst_values[:, -TRAILING_IMU_CHANNEL_COUNT:]):
            raise ReuseError(f"segmented trailing IMU channels changed for {user}")
        summary = load_json(dst / f"{stem}_segmentation_summary.json")
        if not isinstance(summary.get("spike_encoder_spec_sha256"), str):
            raise ReuseError(f"destination segmentation summary lacks encoder identity: {user}")
        checks["trailing_imu_equal"] = True
        checks["dest_encoder_hash"] = summary["spike_encoder_spec_sha256"]
        result[user] = checks
    return result


def find_padding_summary(root: Path) -> Path | None:
    padded = root / "segmentation_padded"
    if not padded.is_dir():
        return None
    preferred = padded / "padding_dataset_summary.json"
    if preferred.is_file():
        return preferred
    candidates = []
    for path in padded.glob("*.json"):
        try:
            payload = load_json(path)
        except ReuseError:
            continue
        if "target_length" in payload and "segment_count" in payload:
            candidates.append(path)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ReuseError(f"multiple possible padding root summaries: {candidates}")
    return None


def source_transform(args: argparse.Namespace) -> str:
    records = discover_records(args.source_root.resolve(), args.action)
    check_uniform_source_records(records)
    values = set()
    for record in records.values():
        transform = first_recursive(record.metadata, ("post_encode_transform",))
        if transform is None:
            values.add("none")
        elif transform in {"AbsRectify", "PolaritySplitAbs"}:
            values.add("AbsRectify")
        else:
            raise ReuseError(f"unsupported source post_encode_transform: {transform!r}")
    if len(values) != 1:
        raise ReuseError(f"source action mixes post-encode transforms: {sorted(values)}")
    return next(iter(values))


def check_encoding(args: argparse.Namespace) -> dict[str, Any]:
    source_records = discover_records(args.source_root.resolve(), args.action)
    dest_records = discover_records(args.dest_root.resolve(), args.action)
    requested = [float(x) for x in args.frequencies]
    rows, _ = compare_new_encoding(source_records, dest_records, requested)
    hashes = sorted({encoder_hash(r.metadata) for r in dest_records.values()})
    if len(hashes) != 1:
        raise ReuseError(f"destination mixes encoder identities: {hashes}")
    return {
        "status": "PASS",
        "requested_frequencies_hz": requested,
        "dest_encoder_hash": hashes[0],
        "recordings": rows,
    }


def padding_target(args: argparse.Namespace) -> int | None:
    summary_path = find_padding_summary(args.source_root.resolve())
    if summary_path is None:
        return None
    summary = load_json(summary_path)
    try:
        target = int(summary["target_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReuseError(f"invalid target_length in {summary_path}") from exc
    if target <= 0:
        raise ReuseError(f"invalid target_length in {summary_path}")
    return target


def compare_padding_geometry(source_root: Path, dest_root: Path, action: str, users: list[str]) -> dict[str, Any]:
    source_summary = find_padding_summary(source_root)
    if source_summary is None:
        return {"source_has_padding": False, "checked": False}
    dest_summary = find_padding_summary(dest_root)
    if dest_summary is None:
        raise ReuseError("source has padded data but destination padding summary is missing")
    source_root_meta = load_json(source_summary)
    dest_root_meta = load_json(dest_summary)
    if int(source_root_meta["target_length"]) != int(dest_root_meta["target_length"]):
        raise ReuseError("padding target length changed")
    per_user: dict[str, Any] = {}
    for user in users:
        src_dir = source_root / "segmentation_padded" / user / f"action_{action}"
        dst_dir = dest_root / "segmentation_padded" / user / f"action_{action}"
        if not src_dir.is_dir():
            continue
        if not dst_dir.is_dir():
            raise ReuseError(f"destination padded package missing for {user}")
        checks: dict[str, Any] = {}
        for token in ("labels", "valid_lengths", "valid_mask"):
            src_matches = sorted(src_dir.glob(f"*_{token}.npy"))
            dst_matches = sorted(dst_dir.glob(f"*_{token}.npy"))
            if len(src_matches) != 1 or len(dst_matches) != 1:
                raise ReuseError(f"expected one padded {token} artifact for {user}")
            checks[token] = compare_npy(src_matches[0], dst_matches[0], f"padded {token}")
        per_user[user] = checks
    return {
        "source_has_padding": True,
        "checked": True,
        "target_length": int(source_root_meta["target_length"]),
        "users": per_user,
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()
    requested = [float(x) for x in args.frequencies]
    source_records = discover_records(source_root, args.action)
    dest_records = discover_records(dest_root, args.action)
    encoding_rows, _ = compare_new_encoding(source_records, dest_records, requested)
    users = sorted({record.user for record in source_records.values()})
    segment_checks = compare_segment_geometry(source_root, dest_root, args.action, users)
    padding_checks = compare_padding_geometry(source_root, dest_root, args.action, users)
    if args.boundary_mode == "aligned-board-events":
        rebind_report = dest_root / "wavelet_alignment_reuse.json"
        if not rebind_report.is_file():
            raise ReuseError("missing alignment reuse report")
        payload = load_json(rebind_report)
        if payload.get("alignment_recomputed") is not False or payload.get(
            "alignment_provenance_rebound"
        ) is not True:
            raise ReuseError("alignment reuse report does not prove reuse/rebind")
    source_freq = frequencies(next(iter(source_records.values())).metadata)
    source_hash = encoder_hash(next(iter(source_records.values())).metadata)
    new_hashes = sorted({encoder_hash(r.metadata) for r in dest_records.values()})
    if len(new_hashes) != 1:
        raise ReuseError(f"destination mixes encoder identities: {new_hashes}")
    if requested != source_freq and new_hashes[0] == source_hash:
        raise ReuseError("frequency change did not change encoder identity")
    report = {
        "status": "PASS",
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "action": args.action,
        "boundary_mode": args.boundary_mode,
        "source_frequencies_hz": source_freq,
        "requested_frequencies_hz": requested,
        "source_encoder_hash": source_hash,
        "dest_encoder_hash": new_hashes[0],
        "recordings": encoding_rows,
        "segmentation_geometry": segment_checks,
        "padding_geometry": padding_checks,
        "contracts": {
            "preprocessed_reused": True,
            "alignment_offset_recomputed": False if args.boundary_mode == "aligned-board-events" else None,
            "alignment_input_trailing_imu_channels_equal": True,
            "segment_geometry_equal": True,
            "destination_metadata_republished_for_new_encoder": True,
        },
    }
    output_path = dest_root / "wavelet_variant_validation.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and safely reuse invariant pipeline artifacts across wavelet-frequency variants."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--action", required=True)
    p.add_argument("--boundary-mode", choices=("label", "aligned-board-events"), required=True)

    p = sub.add_parser("rebind-alignment")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--dest-root", type=Path, required=True)
    p.add_argument("--action", required=True)
    p.add_argument("--frequencies", type=float, nargs=5, required=True)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("padding-target")
    p.add_argument("--source-root", type=Path, required=True)

    p = sub.add_parser("source-transform")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--action", required=True)

    p = sub.add_parser("check-encoding")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--dest-root", type=Path, required=True)
    p.add_argument("--action", required=True)
    p.add_argument("--frequencies", type=float, nargs=5, required=True)

    p = sub.add_parser("verify")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--dest-root", type=Path, required=True)
    p.add_argument("--action", required=True)
    p.add_argument("--boundary-mode", choices=("label", "aligned-board-events"), required=True)
    p.add_argument("--frequencies", type=float, nargs=5, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "preflight":
            result = preflight(args)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "rebind-alignment":
            result = rebind_alignment(args)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "padding-target":
            target = padding_target(args)
            print("NONE" if target is None else target)
            return 0
        if args.command == "source-transform":
            print(source_transform(args))
            return 0
        if args.command == "check-encoding":
            result = check_encoding(args)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "verify":
            result = verify(args)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        raise ReuseError(f"unsupported command: {args.command}")
    except (ReuseError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
