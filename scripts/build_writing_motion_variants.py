#!/usr/bin/env python3
"""Build writing-only motion ablation datasets from aligned-Board segmentation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from writingring.imu_preprocessing import STANDARD_GRAVITY_M_S2
from writingring.spike_encoding.encoders.custom_wavelet import (
    CustomWaveletEncoder,
    CustomWaveletSettings,
)
from writingring.writing_motion_mask import (
    WritingMotionMaskError,
    apply_mask,
    build_recording_writing_mask,
    build_segment_masks,
    pad_writing_mask,
    write_intervals_csv,
)


class BuildError(RuntimeError):
    pass


SOURCE_CHANNEL_COUNT = 36
EVENT_CHANNEL_COUNT = 30


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BuildError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise BuildError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _integral(value: object, *, field: str) -> int:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise BuildError(f"{field} must be integer-like; got {value!r}") from exc
    if not math.isfinite(numeric) or numeric != int(numeric):
        raise BuildError(f"{field} must be integer-like; got {value!r}")
    return int(numeric)


def _user_sort_key(user: str) -> tuple[str, int, str]:
    prefix, sep, suffix = user.rpartition("_")
    return (prefix, int(suffix), user) if sep and suffix.isdigit() else (user, -1, user)


def _spike_root(source_root: Path) -> Path:
    return source_root / "spikeEncoding" / "custom-wavelet"


def _recording_dirs(source_root: Path, user: str | None = None) -> list[tuple[str, str, int, Path]]:
    root = _spike_root(source_root)
    if not root.is_dir():
        raise BuildError(f"missing source spike root: {root}")
    found: list[tuple[str, str, int, Path]] = []
    for metadata in sorted(root.rglob("metadata.json")):
        relative = metadata.parent.relative_to(root)
        if len(relative.parts) < 3:
            continue
        rec_user, action_token, dataset_token = relative.parts[-3:]
        if user is not None and rec_user != user:
            continue
        action = action_token.removeprefix("action_")
        dataset_id = _integral(dataset_token, field="dataset id")
        found.append((rec_user, action, dataset_id, metadata.parent))
    return found


def _completed_segmentation_packages(source_root: Path) -> list[tuple[str, str, Path]]:
    root = source_root / "segmentation"
    if not root.is_dir():
        raise BuildError(f"missing completed source segmentation root: {root}")
    found: list[tuple[str, str, Path]] = []
    for lengths_path in sorted(root.glob("*/action_*/*_segment_lengths.npy")):
        directory = lengths_path.parent
        user = directory.parent.name
        action = directory.name.removeprefix("action_")
        stem = f"{user}_action_{action}"
        if lengths_path.name != f"{stem}_segment_lengths.npy":
            continue
        found.append((user, action, directory))
    if not found:
        raise BuildError(f"no completed segmentation packages found below {root}")
    return found


def list_users(source_root: Path) -> tuple[str, ...]:
    users = {item[0] for item in _completed_segmentation_packages(source_root)}
    return tuple(sorted(users, key=_user_sort_key))


def infer_action(source_root: Path) -> str:
    actions = {item[1] for item in _completed_segmentation_packages(source_root)}
    if len(actions) != 1:
        raise BuildError(f"expected exactly one completed segmentation action, got {sorted(actions)}")
    return next(iter(actions))


def _timestamps_from_metadata(metadata: Mapping[str, Any], metadata_path: Path, sample_count: int) -> np.ndarray:
    candidate = metadata.get("timestamps_path") or metadata.get("timestamp_source_path")
    source = metadata.get("source")
    if candidate is None and isinstance(source, Mapping):
        candidate = source.get("timestamps_path") or source.get("timestamp_source_path")
    if not isinstance(candidate, str) or not candidate:
        raise BuildError(f"metadata has no canonical timestamp path: {metadata_path}")
    path = Path(candidate)
    if not path.is_absolute():
        path = (metadata_path.parent / path).resolve()
    if not path.is_file():
        raise BuildError(f"canonical timestamp artifact does not exist: {path}")
    timestamps = np.load(path, allow_pickle=False)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.shape != (sample_count,) or not np.isfinite(timestamps).all():
        raise BuildError(f"invalid canonical timestamps: {path}")
    expected = metadata.get("timestamps_sha256") or metadata.get("timestamp_sha256")
    if isinstance(expected, str) and _sha256(path) != expected:
        raise BuildError(f"canonical timestamp hash mismatch: {path}")
    return timestamps


def _source_spec(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], str, CustomWaveletEncoder]:
    encoder = metadata.get("encoder")
    if not isinstance(encoder, Mapping):
        raise BuildError("source metadata lacks encoder section")
    spec = encoder.get("spike_encoder")
    digest = encoder.get("spike_encoder_spec_sha256")
    if not isinstance(spec, Mapping) or not isinstance(digest, str):
        raise BuildError("source metadata lacks canonical encoder identity")
    settings_meta = metadata.get("settings")
    output_dtype = "float32"
    if isinstance(settings_meta, Mapping) and settings_meta.get("output_dtype") in {"float32", "float64"}:
        output_dtype = str(settings_meta["output_dtype"])
    payload = {
        "wavelet_name": spec.get("wavelet_name", "acceleration"),
        "frequencies_hz": tuple(spec.get("frequencies_hz", ())),
        "sampling_rate_hz": spec.get("sampling_rate_hz"),
        "prony_denominator_order": spec.get("prony_denominator_order", 2),
        "prony_numerator_order": spec.get("prony_numerator_order", 2),
        "max_filter_time_s": spec.get("max_filter_time_s", 0.3),
        "max_filter_frequency_decades": spec.get("max_filter_frequency_decades", 0.5),
        "output_dtype": output_dtype,
        "post_encode_transform": spec.get("post_encode_transform"),
    }
    encoder_obj = CustomWaveletEncoder(CustomWaveletSettings.from_mapping(payload))
    if encoder_obj.canonical_encoder_spec_sha256 != digest:
        raise BuildError("reconstructed encoder identity does not match source")
    return dict(spec), digest, encoder_obj


def _validate_recording(directory: Path) -> tuple[np.ndarray, dict[str, Any], np.ndarray, CustomWaveletEncoder]:
    values_path = directory / "spikeIMU.npy"
    metadata_path = directory / "metadata.json"
    if not values_path.is_file() or not metadata_path.is_file():
        raise BuildError(f"incomplete source recording: {directory}")
    values = np.load(values_path, allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != SOURCE_CHANNEL_COUNT or not np.isfinite(values).all():
        raise BuildError(f"source recording must be finite (N,36): {values_path}")
    metadata = _load_json(metadata_path)
    section = metadata.get("spike_imu")
    if not isinstance(section, Mapping):
        raise BuildError(f"source recording metadata lacks spike_imu: {metadata_path}")
    if section.get("channel_count") != SOURCE_CHANNEL_COUNT or section.get("event_channel_count") != EVENT_CHANNEL_COUNT:
        raise BuildError("source must be canonical 30-event + 6-IMU SpikeIMU")
    if section.get("schema") != "polarity_split_wavelet_events_plus_imu_v1":
        raise BuildError("source must use polarity_split_wavelet_events_plus_imu_v1")
    if section.get("event_representation") != "unsigned":
        raise BuildError("source event representation must be unsigned")
    rate = float(metadata.get("sampling_rate_hz", 0.0))
    if not math.isclose(rate, 64.0, rel_tol=0.0, abs_tol=1e-12):
        raise BuildError("writing-motion builder currently requires the 64 Hz source pipeline")
    _, _, encoder = _source_spec(metadata)
    timestamps = _timestamps_from_metadata(metadata, metadata_path, len(values))
    return values, metadata, timestamps, encoder


def _patch_recording_metadata(
    metadata: Mapping[str, Any], *, values: np.ndarray, branch: str,
    source_values_path: Path, writing_mask_path: Path,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(metadata))
    section = payload.get("spike_imu")
    if not isinstance(section, dict):
        raise BuildError("spike_imu metadata disappeared while patching")
    section["sample_count"] = int(len(values))
    section["sha256"] = "pending"
    section["spike_imu_sha256"] = "pending"
    payload["writing_motion_variant"] = {
        "schema_version": 1,
        "branch": branch,
        "airborne_motion_zeroed": True,
        "all_motion_channels_zeroed_outside_recording_contact_mask": True,
        "source_spike_imu_path": str(source_values_path.resolve()),
        "source_spike_imu_sha256": _sha256(source_values_path),
        "recording_writing_mask_path": str(writing_mask_path.resolve()),
        "alignment_recomputed": False,
        "segmentation_geometry_reused": True,
    }
    return payload


def _save_recording_artifact(
    destination: Path, *, values: np.ndarray, metadata: Mapping[str, Any],
    writing_mask: np.ndarray, branch: str, source_values_path: Path,
    overwrite: bool,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise BuildError(f"destination exists: {destination}; use --overwrite")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.writing-motion.", dir=destination.parent))
    try:
        np.save(staging / "spikeIMU.npy", values, allow_pickle=False)
        np.save(staging / "spikes.npy", values[:, :EVENT_CHANNEL_COUNT], allow_pickle=False)
        np.save(staging / "writing_mask.npy", writing_mask, allow_pickle=False)
        patched = _patch_recording_metadata(
            metadata,
            values=values,
            branch=branch,
            source_values_path=source_values_path,
            writing_mask_path=destination / "writing_mask.npy",
        )
        digest = _sha256(staging / "spikeIMU.npy")
        patched["spike_imu"]["sha256"] = digest
        patched["spike_imu"]["spike_imu_sha256"] = digest
        _write_json(staging / "metadata.json", patched)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "spikeIMU.npy"


def _plot_verification(
    path: Path, *, user: str, action: str, dataset_id: int,
    boundary_timestamps_us: np.ndarray, source_values: np.ndarray, writing_mask: np.ndarray,
    event_rows: pd.DataFrame, segment_rows: Sequence[Mapping[str, str]], dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    time_s = (boundary_timestamps_us - boundary_timestamps_us[0]) / 1_000_000.0
    original = source_values[:, 30:33]
    masked = original.copy()
    masked[~writing_mask] = 0
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    for axis_index, label in enumerate(("x", "y", "z")):
        axes[0].plot(time_s, original[:, axis_index], linewidth=0.8, label=f"acc_{label}")
        axes[1].plot(time_s, masked[:, axis_index], linewidth=0.8, label=f"acc_{label}")
    axes[0].set_ylabel("Original accel\n(m/s²)")
    axes[1].set_ylabel("Masked accel\n(m/s²)")
    axes[0].legend(ncol=3, loc="upper right")
    axes[1].legend(ncol=3, loc="upper right")
    axes[2].step(time_s, writing_mask.astype(np.int8), where="post")
    axes[2].set_ylabel("Writing mask")
    axes[2].set_ylim(-0.1, 1.1)
    axes[3].set_ylabel("Segments")
    axes[3].set_xlabel("Recording elapsed time (s)")
    axes[3].set_ylim(0.0, 1.0)
    for row in event_rows.to_dict(orient="records"):
        event_time = (float(row["aligned_event_timestamp_us"]) - boundary_timestamps_us[0]) / 1_000_000.0
        style = "-" if str(row["event_type"]) == "press" else "--"
        for axis in axes[:3]:
            axis.axvline(event_time, linewidth=0.7, linestyle=style, alpha=0.45)
    for row in segment_rows:
        if str(row.get("segment_index", "")).strip() == "" or _integral(row["dataset_id"], field="dataset_id") != dataset_id:
            continue
        start = _integral(row["start_sample_index"], field="start_sample_index")
        stop = _integral(row["stop_sample_index_exclusive"], field="stop_sample_index_exclusive")
        left, right = time_s[start], time_s[stop - 1]
        axes[3].axvspan(left, right, alpha=0.2)
        axes[3].text((left + right) / 2.0, 0.5, str(row.get("label", "")), ha="center", va="center", fontsize=8)
    kept = int(np.count_nonzero(writing_mask))
    fig.suptitle(
        f"{user} action {action} dataset {dataset_id} — writing-only mask verification "
        f"({kept}/{len(writing_mask)} samples, {kept/len(writing_mask):.1%} kept)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _patch_segmentation_summary(summary: Mapping[str, Any], *, branch: str, mask_path: Path, intervals_path: Path) -> dict[str, Any]:
    payload = json.loads(json.dumps(summary))
    payload["writing_motion_variant"] = {
        "schema_version": 1,
        "branch": branch,
        "airborne_motion_zeroed": True,
        "writing_mask_path": str(mask_path.resolve()),
        "writing_intervals_path": str(intervals_path.resolve()),
        "press_lift_endpoints_inclusive": True,
        "segmentation_geometry_reused": True,
        "segmentation_boundaries_recomputed": False,
        "alignment_recomputed": False,
    }
    return payload


def _publish_segment_branch(
    source_dir: Path, destination: Path, *, branch_values: np.ndarray,
    writing_mask: np.ndarray, interval_rows: Sequence[Mapping[str, Any]],
    user: str, action: str, branch: str, overwrite: bool,
) -> None:
    if destination.exists() and not overwrite:
        raise BuildError(f"destination exists: {destination}; use --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.writing-motion.", dir=destination.parent))
    try:
        shutil.copytree(source_dir, staging, dirs_exist_ok=True)
        stem = f"{user}_action_{action}"
        np.save(staging / f"{stem}_spikeIMU.npy", branch_values, allow_pickle=False)
        np.save(staging / f"{stem}_writing_mask.npy", writing_mask, allow_pickle=False)
        intervals_path = staging / f"{stem}_writing_intervals.csv"
        write_intervals_csv(intervals_path, interval_rows)
        summary_path = staging / f"{stem}_segmentation_summary.json"
        summary = _load_json(summary_path)
        patched = _patch_segmentation_summary(
            summary,
            branch=branch,
            mask_path=destination / f"{stem}_writing_mask.npy",
            intervals_path=destination / intervals_path.name,
        )
        _write_json(summary_path, patched)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build_padded_branch(
    source_padded_dir: Path, destination: Path, *, segmented_values: np.ndarray,
    writing_mask: np.ndarray, segment_lengths: np.ndarray,
    user: str, action: str, branch: str, overwrite: bool,
) -> dict[str, Any]:
    stem = f"{user}_action_{action}"
    summary_path = source_padded_dir / f"{stem}_padding_summary.json"
    source_summary = _load_json(summary_path)
    target = int(source_summary["target_length"])
    padded_mask, retained = pad_writing_mask(writing_mask, segment_lengths, target_length=target)
    offsets = np.concatenate(([0], np.cumsum(segment_lengths, dtype=np.int64)))
    padded = np.zeros((len(retained), target, segmented_values.shape[1]), dtype=segmented_values.dtype)
    for out_index, source_index in enumerate(retained):
        start, stop = int(offsets[source_index]), int(offsets[source_index + 1])
        length = int(segment_lengths[source_index])
        padded[out_index, :length] = segmented_values[start:stop]
    source_valid_lengths = np.load(source_padded_dir / f"{stem}_valid_lengths.npy", allow_pickle=False)
    source_valid_mask = np.load(source_padded_dir / f"{stem}_valid_mask.npy", allow_pickle=False)
    if not np.array_equal(source_valid_lengths, segment_lengths[retained].astype(source_valid_lengths.dtype, copy=False)):
        raise BuildError("derived retained segment geometry differs from source padding")
    if source_valid_mask.shape != padded_mask.shape:
        raise BuildError("source valid_mask shape differs from derived padded writing mask")
    if np.any(padded_mask & ~source_valid_mask.astype(bool)):
        raise BuildError("padded writing mask extends into source padding")
    if destination.exists() and not overwrite:
        raise BuildError(f"destination exists: {destination}; use --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.writing-motion.", dir=destination.parent))
    try:
        shutil.copytree(source_padded_dir, staging, dirs_exist_ok=True)
        np.save(staging / f"{stem}_paddedSpikeIMU.npy", padded, allow_pickle=False)
        np.save(staging / f"{stem}_padded_writing_mask.npy", padded_mask, allow_pickle=False)
        summary = _load_json(staging / f"{stem}_padding_summary.json")
        summary["writing_motion_variant"] = {
            "schema_version": 1,
            "branch": branch,
            "airborne_motion_zeroed": True,
            "padded_writing_mask_filename": f"{stem}_padded_writing_mask.npy",
            "padding_geometry_reused": True,
        }
        _write_json(staging / f"{stem}_padding_summary.json", summary)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"target_length": target, "retained_segment_count": int(len(retained))}


def build_user(
    source_root: Path, output_root: Path, user: str, *, overwrite: bool,
    verification: bool, verification_dpi: int,
) -> dict[str, Any]:
    action = infer_action(source_root)
    if user not in list_users(source_root):
        raise BuildError(f"unknown user: {user}")
    source_seg = source_root / "segmentation" / user / f"action_{action}"
    source_pad = source_root / "segmentation_padded" / user / f"action_{action}"
    stem = f"{user}_action_{action}"
    if not source_seg.is_dir() or not source_pad.is_dir():
        raise BuildError(f"source user package is incomplete: {user}")
    segment_rows = _read_csv(source_seg / f"{stem}_segments.csv")
    event_df = pd.read_csv(source_seg / f"{stem}_board_events.csv")
    segment_lengths = np.load(source_seg / f"{stem}_segment_lengths.npy", allow_pickle=False)
    source_segmented = np.load(source_seg / f"{stem}_spikeIMU.npy", allow_pickle=False)
    if source_segmented.ndim != 2 or source_segmented.shape[1] != SOURCE_CHANNEL_COUNT:
        raise BuildError("source segmented SpikeIMU must have 36 channels")
    if int(np.sum(segment_lengths, dtype=np.int64)) != len(source_segmented):
        raise BuildError("source segment lengths do not sum to segmented row count")

    exported_dataset_ids = sorted(
        {
            _integral(row["dataset_id"], field="dataset_id")
            for row in segment_rows
            if str(row.get("segment_index", "")).strip() != ""
            and str(row.get("exported", "true")).strip().lower() not in {"false", "0"}
        }
    )
    if not exported_dataset_ids:
        raise BuildError(f"completed source segmentation has no exported recordings for {user}")
    recording_index: dict[int, Path] = {}
    for rec_user, rec_action, dataset_id, directory in _recording_dirs(source_root, user):
        if rec_action != action:
            continue
        if dataset_id in recording_index:
            raise BuildError(f"duplicate source recording dataset id {dataset_id} for {user}")
        recording_index[dataset_id] = directory
    missing_recordings = [dataset_id for dataset_id in exported_dataset_ids if dataset_id not in recording_index]
    if missing_recordings:
        raise BuildError(
            f"source segmentation references missing SpikeIMU recordings for {user}: {missing_recordings}"
        )
    recordings = [
        (user, action, dataset_id, recording_index[dataset_id])
        for dataset_id in exported_dataset_ids
    ]
    d1_recordings: dict[int, np.ndarray] = {}
    d2_recordings: dict[int, np.ndarray] = {}
    intervals_by_dataset: dict[int, Sequence[Any]] = {}
    writing_stats: list[dict[str, Any]] = []

    for rec_user, rec_action, dataset_id, directory in recordings:
        source_values, metadata, timestamps, encoder = _validate_recording(directory)
        rate = float(metadata["sampling_rate_hz"])
        recording_mask, intervals, boundary_timestamps = build_recording_writing_mask(
            event_df,
            dataset_id=dataset_id,
            canonical_timestamps_us=timestamps,
            sampling_rate_hz=rate,
        )
        intervals_by_dataset[dataset_id] = intervals
        d1 = apply_mask(source_values, recording_mask)

        acceleration_g = source_values[:, 30:33].astype(np.float64) / STANDARD_GRAVITY_M_S2
        masked_acceleration_g = acceleration_g.copy()
        masked_acceleration_g[~recording_mask] = 0.0
        encoded = encoder.encode_sequence(masked_acceleration_g)
        events = np.asarray(encoded.values)
        if events.shape != (len(source_values), EVENT_CHANNEL_COUNT):
            raise BuildError("re-encoded event matrix has unexpected shape")
        d2 = np.empty_like(source_values)
        d2[:, :EVENT_CHANNEL_COUNT] = events.astype(source_values.dtype, copy=False)
        d2[:, EVENT_CHANNEL_COUNT:] = source_values[:, EVENT_CHANNEL_COUNT:]
        d2 = apply_mask(d2, recording_mask)

        relative = directory.relative_to(_spike_root(source_root))
        for branch, values in (("postencode_mask", d1), ("masked_accel_reencode", d2)):
            record_dest = output_root / branch / "recordings" / relative
            _save_recording_artifact(
                record_dest,
                values=values,
                metadata=metadata,
                writing_mask=recording_mask,
                branch=branch,
                source_values_path=directory / "spikeIMU.npy",
                overwrite=overwrite,
            )
        if verification:
            verification_path = (
                output_root / "masked_accel_reencode" / "masking" / "verification"
                / user / f"action_{action}" / f"{dataset_id}_masked_accel_verification.png"
            )
            _plot_verification(
                verification_path,
                user=user,
                action=action,
                dataset_id=dataset_id,
                boundary_timestamps_us=boundary_timestamps,
                source_values=source_values,
                writing_mask=recording_mask,
                event_rows=event_df[pd.to_numeric(event_df["dataset_id"], errors="coerce") == dataset_id],
                segment_rows=segment_rows,
                dpi=verification_dpi,
            )
            _write_json(
                verification_path.with_suffix(".json"),
                {
                    "user": user,
                    "action": action,
                    "dataset_id": dataset_id,
                    "sample_count": len(recording_mask),
                    "sampling_rate_hz": rate,
                    "writing_interval_count": len(intervals),
                    "recording_boundary_clipped_interval_count": int(
                        sum(
                            interval.recording_boundary_clipped_start
                            or interval.recording_boundary_clipped_end
                            for interval in intervals
                        )
                    ),
                    "writing_sample_count": int(np.count_nonzero(recording_mask)),
                    "writing_fraction": float(np.mean(recording_mask)),
                    "reposition_sample_count": int(np.count_nonzero(~recording_mask)),
                    "reposition_fraction": float(np.mean(~recording_mask)),
                },
            )
        writing_stats.append(
            {
                "dataset_id": dataset_id,
                "sample_count": len(recording_mask),
                "writing_samples": int(np.count_nonzero(recording_mask)),
                "writing_fraction": float(np.mean(recording_mask)),
                "touch_pair_count": len(intervals),
                "recording_boundary_clipped_touch_pair_count": int(
                    sum(
                        interval.recording_boundary_clipped_start
                        or interval.recording_boundary_clipped_end
                        for interval in intervals
                    )
                ),
            }
        )
        d1_recordings[dataset_id] = d1
        d2_recordings[dataset_id] = d2

    segment_mask, interval_rows = build_segment_masks(
        segment_rows=segment_rows,
        intervals_by_dataset=intervals_by_dataset,
        segment_lengths=segment_lengths,
    )
    offsets = np.concatenate(([0], np.cumsum(segment_lengths, dtype=np.int64)))
    zero_writing_segment_count = sum(
        not np.any(segment_mask[int(offsets[index]) : int(offsets[index + 1])])
        for index in range(len(segment_lengths))
    )
    segment_boundary_clipped_interval_count = sum(
        bool(row.get("segment_boundary_clipped_start"))
        or bool(row.get("segment_boundary_clipped_end"))
        for row in interval_rows
    )
    no_segment_overlap_interval_count = sum(
        not bool(row.get("retained_in_segment"))
        for row in interval_rows
    )
    recording_boundary_clipped_interval_count = sum(
        interval.recording_boundary_clipped_start
        or interval.recording_boundary_clipped_end
        for intervals in intervals_by_dataset.values()
        for interval in intervals
    )
    d1_pieces: list[np.ndarray] = []
    d2_pieces: list[np.ndarray] = []
    exported_rows = [row for row in segment_rows if str(row.get("segment_index", "")).strip() != ""]
    exported_rows.sort(key=lambda row: _integral(row["segment_index"], field="segment_index"))
    for segment_index, row in enumerate(exported_rows):
        dataset_id = _integral(row["dataset_id"], field="dataset_id")
        start = _integral(row["start_sample_index"], field="start_sample_index")
        stop = _integral(row["stop_sample_index_exclusive"], field="stop_sample_index_exclusive")
        mask_start, mask_stop = int(offsets[segment_index]), int(offsets[segment_index + 1])
        local_mask = segment_mask[mask_start:mask_stop]
        d1_piece = apply_mask(d1_recordings[dataset_id][start:stop], local_mask)
        d2_piece = apply_mask(d2_recordings[dataset_id][start:stop], local_mask)
        d1_pieces.append(d1_piece)
        d2_pieces.append(d2_piece)
    d1_segmented = np.concatenate(d1_pieces, axis=0)
    d2_segmented = np.concatenate(d2_pieces, axis=0)
    if d1_segmented.shape != source_segmented.shape or d2_segmented.shape != source_segmented.shape:
        raise BuildError("derived segmented shapes differ from source")
    if not np.array_equal(d1_segmented[segment_mask], source_segmented[segment_mask]):
        raise BuildError("D1 changed source values inside writing mask")
    if np.any(d1_segmented[~segment_mask]) or np.any(d2_segmented[~segment_mask]):
        raise BuildError("derived dataset contains nonzero airborne samples")

    for branch, values in (("postencode_mask", d1_segmented), ("masked_accel_reencode", d2_segmented)):
        seg_dest = output_root / branch / "segmentation" / user / f"action_{action}"
        _publish_segment_branch(
            source_seg,
            seg_dest,
            branch_values=values,
            writing_mask=segment_mask,
            interval_rows=interval_rows,
            user=user,
            action=action,
            branch=branch,
            overwrite=overwrite,
        )
        _build_padded_branch(
            source_pad,
            output_root / branch / "segmentation_padded" / user / f"action_{action}",
            segmented_values=values,
            writing_mask=segment_mask,
            segment_lengths=segment_lengths,
            user=user,
            action=action,
            branch=branch,
            overwrite=overwrite,
        )

    annotation_root = output_root / "original_reference" / "annotations" / user / f"action_{action}"
    annotation_root.mkdir(parents=True, exist_ok=True)
    np.save(annotation_root / f"{stem}_writing_mask.npy", segment_mask, allow_pickle=False)
    write_intervals_csv(annotation_root / f"{stem}_writing_intervals.csv", interval_rows)
    _write_json(
        annotation_root / f"{stem}_annotation_summary.json",
        {
            "source_root": str(source_root.resolve()),
            "user": user,
            "action": action,
            "segment_count": int(len(segment_lengths)),
            "writing_samples": int(np.count_nonzero(segment_mask)),
            "total_segment_samples": int(len(segment_mask)),
            "writing_fraction": float(np.mean(segment_mask)),
            "writing_interval_count": int(len(interval_rows)),
            "recording_boundary_clipped_interval_count": int(
                recording_boundary_clipped_interval_count
            ),
            "segment_boundary_clipped_interval_count": int(
                segment_boundary_clipped_interval_count
            ),
            "no_segment_overlap_interval_count": int(
                no_segment_overlap_interval_count
            ),
            "zero_writing_segment_count": int(zero_writing_segment_count),
        },
    )
    report = {
        "status": "PASS",
        "schema_version": 1,
        "user": user,
        "action": action,
        "source_root": str(source_root.resolve()),
        "segment_count": int(len(segment_lengths)),
        "writing_sample_count": int(np.count_nonzero(segment_mask)),
        "segment_sample_count": int(len(segment_mask)),
        "writing_fraction": float(np.mean(segment_mask)),
        "interval_count": int(len(interval_rows)),
        "recording_boundary_clipped_interval_count": int(
            recording_boundary_clipped_interval_count
        ),
        "segment_boundary_clipped_interval_count": int(
            segment_boundary_clipped_interval_count
        ),
        "no_segment_overlap_interval_count": int(
            no_segment_overlap_interval_count
        ),
        "zero_writing_segment_count": int(zero_writing_segment_count),
        "recordings": writing_stats,
        "branches": ["original_reference", "postencode_mask", "masked_accel_reencode"],
        "geometry_reused": True,
        "verification_enabled": verification,
    }
    _write_json(output_root / "user_reports" / f"{user}.json", report)
    return report


def _copy_root_padding_metadata(source_root: Path, branch_root: Path, *, branch: str, overwrite: bool) -> None:
    source = source_root / "segmentation_padded"
    branch_root.mkdir(parents=True, exist_ok=True)
    for name in ("padding_dataset_manifest.csv",):
        src = source / name
        if src.is_file():
            dst = branch_root / name
            if dst.exists() and not overwrite:
                raise BuildError(f"destination exists: {dst}; use --overwrite")
            shutil.copy2(src, dst)
    src_summary = source / "padding_dataset_summary.json"
    if src_summary.is_file():
        summary = _load_json(src_summary)
        summary["input_root"] = str((branch_root.parent / "segmentation").resolve())
        summary["output_root"] = str(branch_root.resolve())
        summary["writing_motion_variant"] = {
            "schema_version": 1,
            "branch": branch,
            "padding_geometry_reused": True,
        }
        _write_json(branch_root / "padding_dataset_summary.json", summary)


def finalize(source_root: Path, output_root: Path, *, overwrite: bool) -> dict[str, Any]:
    users = list_users(source_root)
    action = infer_action(source_root)
    reports = []
    for user in users:
        path = output_root / "user_reports" / f"{user}.json"
        if not path.is_file():
            raise BuildError(f"missing user worker report: {path}")
        report = _load_json(path)
        if report.get("status") != "PASS":
            raise BuildError(f"user worker did not PASS: {path}")
        reports.append(report)
    for branch in ("postencode_mask", "masked_accel_reencode"):
        _copy_root_padding_metadata(
            source_root,
            output_root / branch / "segmentation_padded",
            branch=branch,
            overwrite=overwrite,
        )
    source_summary = source_root / "segmentation_padded" / "padding_dataset_summary.json"
    source_digest = _sha256(source_summary) if source_summary.is_file() else None
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "source_root": str(source_root.resolve()),
        "action": action,
        "users": list(users),
        "user_count": len(users),
        "branches": {
            "D0_original": {
                "kind": "reference",
                "root": str(source_root.resolve()),
                "definition": "Encoder(a)",
            },
            "D1_postencode_mask": {
                "kind": "derived",
                "root": str((output_root / "postencode_mask").resolve()),
                "definition": "m * Encoder(a)",
            },
            "D2_masked_accel_reencode": {
                "kind": "derived",
                "root": str((output_root / "masked_accel_reencode").resolve()),
                "definition": "m * Encoder(m * a)",
            },
        },
        "source_padding_summary_sha256": source_digest,
        "geometry_reused": True,
        "alignment_recomputed": False,
        "segmentation_recomputed": False,
        "press_lift_endpoints_inclusive": True,
        "airborne_definition": "valid segment samples outside assigned complete valid non-transient press-lift unions",
        "recording_reencode_mask_definition": "all complete valid non-transient physical Board touch pairs",
        "user_reports": reports,
    }
    _write_json(output_root / "writing_motion_ablation_manifest.json", manifest)
    _write_json(
        output_root / "original_reference" / "dataset_reference.json",
        {
            "source_root": str(source_root.resolve()),
            "definition": "D0 = Encoder(a)",
            "features_copied": False,
            "annotations_root": str((output_root / "original_reference" / "annotations").resolve()),
        },
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list-users")
    p.add_argument("--source-root", type=Path, required=True)
    p = sub.add_parser("build-user")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-verification", action="store_true")
    p.add_argument("--verification-dpi", type=int, default=200)
    p = sub.add_parser("finalize")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "list-users":
            for user in list_users(args.source_root.resolve()):
                print(user)
            return 0
        if args.command == "build-user":
            if args.verification_dpi <= 0:
                raise BuildError("verification DPI must be positive")
            report = build_user(
                args.source_root.resolve(),
                args.output_root.resolve(),
                args.user,
                overwrite=args.overwrite,
                verification=not args.no_verification,
                verification_dpi=args.verification_dpi,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "finalize":
            report = finalize(
                args.source_root.resolve(),
                args.output_root.resolve(),
                overwrite=args.overwrite,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        raise BuildError(f"unsupported command: {args.command}")
    except (BuildError, WritingMotionMaskError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
