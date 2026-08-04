"""Variable-length timestamp-label segmentation for primary Ring IMU streams.

Each label starts one half-open interval ending immediately before the Ring
sample selected for the following label.  The exporter stores all variable-
length segments in one contiguous numeric array plus offsets, avoiding padded
data and object arrays that would require pickle to load.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Final, Sequence

import numpy as np
import pandas as pd

from writingring.discovery import DiscoveryError, Recording, discover_recordings
from writingring.gravity import (
    GRAVITY_REMOVAL_METHODS,
    GravityRemovalConfig,
    GravityRemovalError,
    process_ring_gravity,
)
from writingring.ring_loader import RingLoadError, load_ring


IMU_CHANNEL_COLUMNS: Final[tuple[str, ...]] = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
)
_TIMESTAMP_COLUMN: Final[str] = "timestamp"
_WRONG_LABEL: Final[str] = "wrong"
_DEFAULT_MINIMUM_LABEL_INTERVAL_US: Final[float] = 100_000.0
_DEFAULT_MAXIMUM_SEGMENT_DURATION_US: Final[float] = 5_000_000.0
_DEFAULT_GRAVITY_REMOVAL_METHOD: Final[str] = "low-pass"


class SegmentationError(ValueError):
    """Raised when labels, Ring inputs, or segmentation exports are invalid."""


class SegmentLabelParseError(SegmentationError):
    """Raised when a timestamp-label text file cannot be parsed safely."""


class SegmentationOutputError(SegmentationError):
    """Raised when segmentation outputs cannot be published safely."""


@dataclass(frozen=True, slots=True)
class SegmentLabel:
    """One source marker defining the start of a labelled IMU segment."""

    timestamp_us: float
    label: str
    source_line_number: int


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    """Output dtype, endpoint, and invalid-label policy for segments."""

    output_dtype: str = "float32"
    include_last_label: bool = True
    minimum_label_interval_us: float = _DEFAULT_MINIMUM_LABEL_INTERVAL_US
    maximum_segment_duration_us: float = _DEFAULT_MAXIMUM_SEGMENT_DURATION_US


@dataclass(frozen=True, slots=True)
class SegmentedSample:
    """One unpadded IMU segment plus its source-boundary metadata."""

    imu: np.ndarray
    label: str
    source_label_index: int
    sample_count: int
    start_sample_index: int
    stop_sample_index_exclusive: int
    label_timestamp_us: float
    next_label_timestamp_us: float | None


@dataclass(frozen=True, slots=True)
class SegmentationOutputPaths:
    """Deterministic output locations for one user/action aggregation."""

    raw_imu_path: Path
    labels_path: Path
    segment_offsets_path: Path
    segment_lengths_path: Path
    segments_csv_path: Path
    summary_json_path: Path


@dataclass(frozen=True, slots=True)
class UserActionSegmentationResult:
    """Contiguous IMU values and boundaries for one user/action export."""

    user: str
    action: str
    raw_imu: np.ndarray
    labels: np.ndarray
    segment_offsets: np.ndarray
    segment_lengths: np.ndarray
    manifest: pd.DataFrame
    summary: dict[str, object]
    output_paths: SegmentationOutputPaths


def load_timestamp_labels(path: Path) -> tuple[SegmentLabel, ...]:
    """Parse strictly increasing ``timestamp label`` markers from UTF-8 text."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise SegmentLabelParseError(
            f"could not read timestamp label file {source}: {error}"
        ) from error
    labels: list[SegmentLabel] = []
    previous_timestamp: float | None = None
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not fields[1].strip():
            raise SegmentLabelParseError(
                f"{source}:{line_number}: expected timestamp and nonempty label"
            )
        try:
            timestamp = _finite_float(
                fields[0], name=f"{source}:{line_number} timestamp"
            )
        except SegmentationError as error:
            raise SegmentLabelParseError(str(error)) from error
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise SegmentLabelParseError(
                f"{source}:{line_number}: timestamps must be strictly increasing"
            )
        labels.append(
            SegmentLabel(
                timestamp_us=timestamp,
                label=fields[1].strip(),
                source_line_number=line_number,
            )
        )
        previous_timestamp = timestamp
    if not labels:
        raise SegmentLabelParseError(f"{source}: no timestamp labels were found")
    return tuple(labels)


def segment_recording_by_labels(
    *,
    ring_imu: np.ndarray,
    ring_timestamps_us: np.ndarray,
    labels: Sequence[SegmentLabel],
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[SegmentedSample, ...]:
    """Split one Ring stream at labels without padding or truncation."""

    dtype = _validated_config(config)
    imu, timestamps = _validated_ring_inputs(ring_imu, ring_timestamps_us)
    markers = _validated_labels(labels)
    _validate_label_range(markers, timestamps)
    skip_reasons = label_start_skip_reasons(
        markers,
        ring_end_timestamp_us=float(timestamps[-1]),
        config=config,
    )
    stop_label_index = len(markers) if config.include_last_label else len(markers) - 1
    if stop_label_index == 0:
        raise SegmentationError("excluding the last label leaves no segments")

    samples: list[SegmentedSample] = []
    for label_index, marker in enumerate(markers[:stop_label_index]):
        if skip_reasons[label_index] is not None:
            continue
        next_marker = (
            markers[label_index + 1]
            if label_index + 1 < len(markers)
            else None
        )
        start = int(np.searchsorted(timestamps, marker.timestamp_us, side="left"))
        stop = (
            int(np.searchsorted(timestamps, next_marker.timestamp_us, side="left"))
            if next_marker is not None
            else len(timestamps)
        )
        if start >= len(timestamps):
            raise SegmentationError(
                f"label at {marker.timestamp_us:g} has no Ring sample at or after it"
            )
        if stop < start:
            raise SegmentationError(
                f"segment stop precedes start for label at {marker.timestamp_us:g}"
            )
        segment = np.asarray(imu[start:stop], dtype=dtype).copy()
        if len(segment) == 0:
            raise SegmentationError(
                f"label at {marker.timestamp_us:g} defines an empty IMU segment"
            )
        segment.setflags(write=False)
        samples.append(
            SegmentedSample(
                imu=segment,
                label=marker.label,
                source_label_index=label_index,
                sample_count=len(segment),
                start_sample_index=start,
                stop_sample_index_exclusive=stop,
                label_timestamp_us=marker.timestamp_us,
                next_label_timestamp_us=(
                    None if next_marker is None else next_marker.timestamp_us
                ),
            )
        )
    if not samples:
        raise SegmentationError("no valid label starts remain after interval filtering")
    return tuple(samples)


def build_segmentation_output_paths(
    output_root: Path,
    *,
    user: str,
    action: str,
) -> SegmentationOutputPaths:
    """Return deterministic paths without creating or overwriting anything."""

    _validate_identity(user=user, action=action)
    base = Path(output_root) / user / f"action_{action}"
    stem = f"{user}_action_{action}"
    return SegmentationOutputPaths(
        raw_imu_path=base / f"{stem}_rawIMU.npy",
        labels_path=base / f"{stem}_labels.npy",
        segment_offsets_path=base / f"{stem}_segment_offsets.npy",
        segment_lengths_path=base / f"{stem}_segment_lengths.npy",
        segments_csv_path=base / f"{stem}_segments.csv",
        summary_json_path=base / f"{stem}_segmentation_summary.json",
    )


def segment_user_action(
    *,
    data_root: Path,
    user: str,
    action: str,
    output_root: Path,
    config: SegmentationConfig = SegmentationConfig(),
    gravity_config: GravityRemovalConfig | None = None,
    overwrite: bool = False,
) -> UserActionSegmentationResult:
    """Remove gravity, then aggregate variable-length segments and save them."""

    dtype = _validated_config(config)
    effective_gravity_config = _effective_gravity_config(gravity_config)
    _validate_identity(user=user, action=action)
    if not isinstance(overwrite, bool):
        raise SegmentationError("overwrite must be a boolean")
    try:
        recordings = discover_recordings(data_root)
    except DiscoveryError as error:
        raise SegmentationError(str(error)) from error
    selected = sorted(
        (
            recording
            for recording in recordings
            if recording.user == user and recording.action == action
        ),
        key=lambda recording: recording.dataset_id,
    )
    if not selected:
        raise SegmentationError(
            f"no recordings found for user={user!r}, action={action!r}"
        )

    all_samples: list[SegmentedSample] = []
    manifest_rows: list[dict[str, object]] = []
    skipped_label_counts: dict[str, int] = {}
    source_label_count = 0
    for recording in selected:
        ring_imu, timestamps = _recording_ring_arrays(
            recording, gravity_config=effective_gravity_config
        )
        if recording.timestamp_path is None:
            raise SegmentationError(
                f"dataset {recording.dataset_id} is missing its timestamp label file"
            )
        markers = load_timestamp_labels(recording.timestamp_path)
        source_label_count += len(markers)
        for reason in label_start_skip_reasons(
            markers,
            ring_end_timestamp_us=float(timestamps[-1]),
            config=config,
        ):
            if reason is not None:
                skipped_label_counts[reason] = skipped_label_counts.get(reason, 0) + 1
        samples = segment_recording_by_labels(
            ring_imu=ring_imu,
            ring_timestamps_us=timestamps,
            labels=markers,
            config=config,
        )
        for sample in samples:
            all_samples.append(sample)
            manifest_rows.append(
                _manifest_row(
                    segment_index=len(all_samples) - 1,
                    user=user,
                    action=action,
                    recording=recording,
                    label_index=sample.source_label_index,
                    sample=sample,
                    timestamps=timestamps,
                    gravity_config=effective_gravity_config,
                )
            )

    raw_imu = np.concatenate([sample.imu for sample in all_samples], axis=0)
    labels_array = np.asarray([sample.label for sample in all_samples], dtype=np.str_)
    segment_lengths = np.asarray(
        [sample.sample_count for sample in all_samples], dtype=np.int32
    )
    segment_offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(segment_lengths, dtype=np.int64))
    )
    manifest = pd.DataFrame(manifest_rows)
    summary = _summary(
        user=user,
        action=action,
        recordings=selected,
        samples=all_samples,
        dtype=dtype,
        source_label_count=source_label_count,
        skipped_label_counts=skipped_label_counts,
        minimum_label_interval_us=config.minimum_label_interval_us,
        maximum_segment_duration_us=config.maximum_segment_duration_us,
        gravity_config=effective_gravity_config,
    )
    paths = build_segmentation_output_paths(output_root, user=user, action=action)
    _write_outputs(
        paths,
        raw_imu=raw_imu,
        labels=labels_array,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        manifest=manifest,
        summary=summary,
        overwrite=overwrite,
    )
    for array in (raw_imu, labels_array, segment_offsets, segment_lengths):
        array.setflags(write=False)
    return UserActionSegmentationResult(
        user=user,
        action=action,
        raw_imu=raw_imu,
        labels=labels_array,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        manifest=manifest,
        summary=summary,
        output_paths=paths,
    )


def _recording_ring_arrays(
    recording: Recording,
    *,
    gravity_config: GravityRemovalConfig,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        ring = load_ring(recording)
        gravity_result = process_ring_gravity(ring, config=gravity_config)
    except (RingLoadError, GravityRemovalError) as error:
        raise SegmentationError(
            "could not load and remove gravity from dataset "
            f"{recording.dataset_id} primary Ring: {error}"
        ) from error
    imu = np.column_stack(
        (
            gravity_result.linear_acceleration_body,
            gravity_result.angular_velocity_body_rad_s,
        )
    )
    timestamps = ring.dataframe[_TIMESTAMP_COLUMN].to_numpy(copy=True)
    return _validated_ring_inputs(imu, timestamps)


def _manifest_row(
    *,
    segment_index: int,
    user: str,
    action: str,
    recording: Recording,
    label_index: int,
    sample: SegmentedSample,
    timestamps: np.ndarray,
    gravity_config: GravityRemovalConfig,
) -> dict[str, object]:
    stop = sample.stop_sample_index_exclusive
    return {
        "segment_index": segment_index,
        "user": user,
        "action": action,
        "dataset_id": recording.dataset_id,
        "label_index": label_index,
        "label": sample.label,
        "label_timestamp_us": sample.label_timestamp_us,
        "next_label_timestamp_us": sample.next_label_timestamp_us,
        "start_sample_index": sample.start_sample_index,
        "stop_sample_index_exclusive": stop,
        "first_sample_timestamp_us": float(timestamps[sample.start_sample_index]),
        "last_sample_timestamp_us": float(timestamps[stop - 1]),
        "sample_count": sample.sample_count,
        "gravity_removal_method": gravity_config.gravity_removal_method,
        "gravity_low_pass_cutoff_hz": gravity_config.low_pass_cutoff_hz,
        "gravity_madgwick_beta": gravity_config.madgwick_beta,
        "is_last_label": sample.next_label_timestamp_us is None,
        "segment_end_source": (
            "ring_recording_end"
            if sample.next_label_timestamp_us is None
            else "next_label_timestamp"
        ),
        "ring_source_path": str(recording.ring_0_path),
        "label_source_path": str(recording.timestamp_path),
    }


def _summary(
    *,
    user: str,
    action: str,
    recordings: Sequence[Recording],
    samples: Sequence[SegmentedSample],
    dtype: np.dtype,
    source_label_count: int,
    skipped_label_counts: dict[str, int],
    minimum_label_interval_us: float,
    maximum_segment_duration_us: float,
    gravity_config: GravityRemovalConfig,
) -> dict[str, object]:
    lengths = np.asarray([sample.sample_count for sample in samples], dtype=np.int64)
    return {
        "boundary_mode": "label",
        "user": user,
        "action": action,
        "recording_count": len(recordings),
        "dataset_ids": [recording.dataset_id for recording in recordings],
        "segment_count": len(samples),
        "source_label_count": source_label_count,
        "skipped_label_count": int(sum(skipped_label_counts.values())),
        "skipped_label_counts_by_reason": dict(sorted(skipped_label_counts.items())),
        "minimum_label_interval_us": minimum_label_interval_us,
        "maximum_segment_duration_us": maximum_segment_duration_us,
        "gravity_removal": {
            "method": gravity_config.gravity_removal_method,
            "sampling_rate_hz": gravity_config.sampling_rate_hz,
            "low_pass_cutoff_hz": gravity_config.low_pass_cutoff_hz,
            "madgwick_beta": gravity_config.madgwick_beta,
            "strict_calibration": gravity_config.strict_calibration,
        },
        "total_imu_sample_count": int(np.sum(lengths)),
        "channel_count": len(IMU_CHANNEL_COLUMNS),
        "minimum_segment_length": int(np.min(lengths)),
        "maximum_segment_length": int(np.max(lengths)),
        "median_segment_length": float(np.median(lengths)),
        "output_dtype": dtype.name,
        "time_mapping": "segment_i = [label_i_timestamp, label_(i+1)_timestamp)",
        "padding_or_truncation": "disabled",
        "storage": "rawIMU is contiguous; segment_offsets delimit each segment",
    }


def _write_outputs(
    paths: SegmentationOutputPaths,
    *,
    raw_imu: np.ndarray,
    labels: np.ndarray,
    segment_offsets: np.ndarray,
    segment_lengths: np.ndarray,
    manifest: pd.DataFrame,
    summary: dict[str, object],
    overwrite: bool,
) -> None:
    all_paths = [
        paths.raw_imu_path,
        paths.labels_path,
        paths.segment_offsets_path,
        paths.segment_lengths_path,
        paths.segments_csv_path,
        paths.summary_json_path,
        *_legacy_output_paths(paths),
    ]
    existing = [path for path in all_paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise SegmentationOutputError(
            "segmentation output already exists; use overwrite=True: " + joined
        )
    paths.raw_imu_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(paths.raw_imu_path, raw_imu)
    _atomic_save_npy(paths.labels_path, labels)
    _atomic_save_npy(paths.segment_offsets_path, segment_offsets)
    _atomic_save_npy(paths.segment_lengths_path, segment_lengths)
    _atomic_write_text(paths.segments_csv_path, manifest.to_csv(index=False))
    _atomic_write_text(
        paths.summary_json_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    _remove_legacy_outputs(paths, overwrite=overwrite)


def _legacy_output_paths(paths: SegmentationOutputPaths) -> tuple[Path, Path]:
    stem = paths.raw_imu_path.name.removesuffix("_rawIMU.npy")
    return (
        paths.raw_imu_path.with_name(f"{stem}_valid_lengths.npy"),
        paths.raw_imu_path.with_name(f"{stem}_valid_mask.npy"),
    )


def _remove_legacy_outputs(paths: SegmentationOutputPaths, *, overwrite: bool) -> None:
    """Remove only replaced fixed-length sidecars after a successful rewrite."""

    for path in _legacy_output_paths(paths):
        if not path.exists():
            continue
        if not overwrite:
            raise SegmentationOutputError(
                "segmentation output already exists; use overwrite=True: " + str(path)
            )
        try:
            path.unlink()
        except OSError as error:
            raise SegmentationOutputError(
                f"could not remove replaced fixed-length output {path}: {error}"
            ) from error


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            np.save(file, values, allow_pickle=False)
        os.replace(temporary, path)
    except OSError as error:
        raise SegmentationOutputError(f"could not write {path}: {error}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write(text)
        os.replace(temporary, path)
    except OSError as error:
        raise SegmentationOutputError(f"could not write {path}: {error}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def _validated_config(config: SegmentationConfig) -> np.dtype:
    if not isinstance(config, SegmentationConfig):
        raise SegmentationError("config must be a SegmentationConfig")
    if not isinstance(config.include_last_label, bool):
        raise SegmentationError("include_last_label must be a boolean")
    minimum_interval = _finite_float(
        config.minimum_label_interval_us, name="minimum_label_interval_us"
    )
    if minimum_interval < 0.0:
        raise SegmentationError("minimum_label_interval_us must be nonnegative")
    maximum_duration = _finite_float(
        config.maximum_segment_duration_us, name="maximum_segment_duration_us"
    )
    if maximum_duration <= 0.0:
        raise SegmentationError("maximum_segment_duration_us must be positive")
    if minimum_interval > maximum_duration:
        raise SegmentationError(
            "minimum_label_interval_us must not exceed maximum_segment_duration_us"
        )
    if config.output_dtype not in {"float32", "float64"}:
        raise SegmentationError("output_dtype must be float32 or float64")
    return np.dtype(config.output_dtype)


def _effective_gravity_config(
    gravity_config: GravityRemovalConfig | None,
) -> GravityRemovalConfig:
    """Return the low-pass default while keeping gravity settings explicit."""

    if gravity_config is None:
        return GravityRemovalConfig(
            gravity_removal_method=_DEFAULT_GRAVITY_REMOVAL_METHOD,
            # Low-pass does not use the calibration estimate itself. The
            # existing reusable API still computes it for diagnostics, so a
            # failed stationary check must remain provisional rather than
            # prevent the default preprocessing path from exporting data.
            strict_calibration=False,
        )
    if not isinstance(gravity_config, GravityRemovalConfig):
        raise SegmentationError("gravity_config must be a GravityRemovalConfig")
    if gravity_config.gravity_removal_method not in GRAVITY_REMOVAL_METHODS:
        choices = ", ".join(GRAVITY_REMOVAL_METHODS)
        raise SegmentationError(f"gravity_removal_method must be one of: {choices}")
    return gravity_config


def _validated_ring_inputs(
    ring_imu: np.ndarray,
    ring_timestamps_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        imu = np.asarray(ring_imu, dtype=np.float64)
        timestamps = np.asarray(ring_timestamps_us, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SegmentationError("Ring IMU and timestamps must be numeric") from error
    if imu.ndim != 2 or imu.shape[1] != len(IMU_CHANNEL_COLUMNS) or len(imu) == 0:
        raise SegmentationError("Ring IMU must have nonempty shape (M, 6)")
    if timestamps.ndim != 1 or len(timestamps) != len(imu):
        raise SegmentationError("Ring timestamps must be a length-M vector")
    if not np.isfinite(imu).all() or not np.isfinite(timestamps).all():
        raise SegmentationError("Ring IMU and timestamps must be finite")
    if np.any(np.diff(timestamps) < 0.0):
        raise SegmentationError("Ring timestamps must be nondecreasing")
    return imu.copy(), timestamps.copy()


def _validated_labels(labels: Sequence[SegmentLabel]) -> tuple[SegmentLabel, ...]:
    values = tuple(labels)
    if not values:
        raise SegmentationError("at least one timestamp label is required")
    previous: float | None = None
    for label in values:
        if not isinstance(label, SegmentLabel):
            raise SegmentationError("labels must contain SegmentLabel values")
        timestamp = _finite_float(label.timestamp_us, name="label timestamp")
        if not isinstance(label.label, str) or not label.label:
            raise SegmentationError("label text must be nonempty")
        if previous is not None and timestamp <= previous:
            raise SegmentationError("label timestamps must be strictly increasing")
        previous = timestamp
    return values


def _validate_label_range(labels: Sequence[SegmentLabel], timestamps: np.ndarray) -> None:
    start = float(timestamps[0])
    stop = float(timestamps[-1])
    for label in labels:
        if label.timestamp_us < start or label.timestamp_us > stop:
            raise SegmentationError(
                f"label timestamp {label.timestamp_us:g} is outside Ring range {start:g}:{stop:g}"
            )


def _label_start_skip_reasons(
    labels: Sequence[SegmentLabel],
    *,
    ring_end_timestamp_us: float,
    config: SegmentationConfig,
) -> tuple[str | None, ...]:
    """Classify source markers that remain boundaries but not starts."""

    reasons: list[str | None] = [None] * len(labels)
    for index, marker in enumerate(labels):
        if marker.label.strip().casefold() == _WRONG_LABEL:
            reasons[index] = "label_is_wrong"

    for index in range(len(labels) - 1):
        interval = labels[index + 1].timestamp_us - labels[index].timestamp_us
        if interval < config.minimum_label_interval_us:
            _set_skip_reason(reasons, index, "adjacent_interval_lt_0.1s")
            _set_skip_reason(reasons, index + 1, "adjacent_interval_lt_0.1s")
        elif interval > config.maximum_segment_duration_us:
            _set_skip_reason(reasons, index, "segment_duration_gt_5s")

    final_interval = ring_end_timestamp_us - labels[-1].timestamp_us
    if final_interval > config.maximum_segment_duration_us:
        _set_skip_reason(reasons, len(labels) - 1, "final_segment_duration_gt_5s")
    return tuple(reasons)


def label_start_skip_reasons(
    labels: Sequence[SegmentLabel],
    *,
    ring_end_timestamp_us: float,
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[str | None, ...]:
    """Return the current label-only validity decisions without segmenting.

    Board-guided segmentation uses this public helper so both boundary modes
    share identical ``wrong``, too-close, and too-long label-start rules.
    Labels remain interval boundaries even when their returned reason is not
    ``None``.
    """

    _validated_config(config)
    markers = _validated_labels(labels)
    end = _finite_float(ring_end_timestamp_us, name="ring_end_timestamp_us")
    if end < markers[-1].timestamp_us:
        raise SegmentationError("ring_end_timestamp_us precedes the final label")
    return _label_start_skip_reasons(
        markers,
        ring_end_timestamp_us=end,
        config=config,
    )


def _set_skip_reason(reasons: list[str | None], index: int, reason: str) -> None:
    """Keep an explicit ``wrong`` label reason when multiple rules apply."""

    if reasons[index] is None:
        reasons[index] = reason


def _validate_identity(*, user: str, action: str) -> None:
    for name, value in (("user", user), ("action", action)):
        if (
            not isinstance(value, str)
            or not value
            or value.strip() != value
            or "/" in value
            or "\\" in value
        ):
            raise SegmentationError(f"{name} must be a nonempty safe path component")


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise SegmentationError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SegmentationError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise SegmentationError(f"{name} must be finite")
    return number
