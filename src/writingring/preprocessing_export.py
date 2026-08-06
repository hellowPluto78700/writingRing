"""Export complete, file-backed Ring IMU preprocessing artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np

from writingring.discovery import DiscoveryError, Recording, discover_recordings
from writingring.gravity import GravityRemovalConfig
from writingring.imu_preprocessing import (
    IMUPreprocessingError,
    PREPROCESSED_IMU_COLUMNS,
    STANDARD_GRAVITY_M_S2,
    preprocess_ring_imu,
)
from writingring.preprocessing_io import (
    PREPROCESSING_SCHEMA_VERSION,
    PREPROCESSED_IMU_ARTIFACT_TYPE,
    PREPROCESSED_IMU_UNITS,
    PreprocessingIOError,
    load_preprocessed_imu,
    load_preprocessing_summary,
    sha256_file,
    validate_preprocessed_imu,
)
from writingring.ring_loader import RingLoadError, load_ring


class PreprocessingExportError(ValueError):
    """Raised when a preprocessing artifact cannot be safely exported."""


@dataclass(frozen=True, slots=True)
class PreprocessingOutputPaths:
    """The canonical feature, timestamp, and summary files for one recording."""

    output_directory: Path
    imu_path: Path
    timestamps_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class PreprocessingExportResult:
    """One published preprocessing artifact and its validated summary."""

    recording: Recording
    imu: np.ndarray
    summary: dict[str, object]
    output_paths: PreprocessingOutputPaths


def build_preprocessing_output_paths(
    output_root: Path,
    *,
    recording: Recording,
) -> PreprocessingOutputPaths:
    """Return the stable ``user/action/data_id`` artifact namespace."""

    _validate_identity(recording)
    directory = Path(output_root) / recording.user / recording.action / str(recording.dataset_id)
    stem = str(recording.dataset_id)
    return PreprocessingOutputPaths(
        output_directory=directory,
        imu_path=directory / f"{stem}_preprocessedIMU.npy",
        timestamps_path=directory / f"{stem}_timestamps_us.npy",
        summary_path=directory / f"{stem}_preprocessing.json",
    )


def export_recording_preprocessing(
    recording: Recording,
    *,
    output_root: Path,
    gravity_config: GravityRemovalConfig,
    output_dtype: str = "float32",
    overwrite: bool = False,
) -> PreprocessingExportResult:
    """Preprocess and atomically publish one complete Ring recording."""

    _validate_identity(recording)
    dtype = _validated_output_dtype(output_dtype)
    if not isinstance(gravity_config, GravityRemovalConfig):
        raise PreprocessingExportError("gravity_config must be a GravityRemovalConfig")
    if not isinstance(overwrite, bool):
        raise PreprocessingExportError("overwrite must be a boolean")
    try:
        ring = load_ring(recording)
        result = preprocess_ring_imu(ring, config=gravity_config)
    except (RingLoadError, IMUPreprocessingError) as error:
        raise PreprocessingExportError(
            f"could not preprocess {recording.user}/{recording.action}/"
            f"{recording.dataset_id}: {error}"
        ) from error
    values = np.asarray(result.imu, dtype=dtype).copy()
    timestamps = _validated_timestamps(
        ring.dataframe["timestamp"].to_numpy(dtype=np.float64, copy=True),
        sample_count=len(values),
    )
    try:
        validate_preprocessed_imu(values)
    except PreprocessingIOError as error:
        raise PreprocessingExportError(str(error)) from error
    paths = build_preprocessing_output_paths(output_root, recording=recording)
    summary = _build_summary(
        recording=recording,
        result=result,
        gravity_config=gravity_config,
        values=values,
        timestamps=timestamps,
        output_dtype=dtype,
        output_path=paths.imu_path,
        timestamps_path=paths.timestamps_path,
    )
    _publish_artifact(
        paths,
        values=values,
        timestamps=timestamps,
        summary=summary,
        overwrite=overwrite,
    )
    try:
        published = load_preprocessed_imu(
            paths.imu_path,
            summary_path=paths.summary_path,
            allow_gravity_included=True,
        )
    except PreprocessingIOError as error:
        raise PreprocessingExportError(
            f"published preprocessing artifact failed verification: {error}"
        ) from error
    published_values = np.array(published.imu, copy=True)
    published_values.setflags(write=False)
    return PreprocessingExportResult(
        recording=recording,
        imu=published_values,
        summary=summary,
        output_paths=paths,
    )


def export_discovered_preprocessing(
    *,
    data_root: Path,
    output_root: Path,
    gravity_config: GravityRemovalConfig,
    output_dtype: str = "float32",
    overwrite: bool = False,
    user: str | None = None,
    action: str | None = None,
    dataset_id: int | None = None,
) -> tuple[PreprocessingExportResult, ...]:
    """Export all or one exact discovered recording in deterministic order."""

    if (user is None) != (action is None) or (user is None) != (dataset_id is None):
        raise PreprocessingExportError(
            "user, action, and dataset_id must be supplied together or omitted together"
        )
    try:
        recordings = discover_recordings(data_root)
    except DiscoveryError as error:
        raise PreprocessingExportError(str(error)) from error
    selected = sorted(
        (
            recording
            for recording in recordings
            if user is None
            or (
                recording.user == user
                and recording.action == action
                and recording.dataset_id == dataset_id
            )
        ),
        key=lambda recording: (recording.user, recording.action, recording.dataset_id),
    )
    if not selected:
        selector = "all recordings" if user is None else f"{user}/{action}/{dataset_id}"
        raise PreprocessingExportError(f"no recordings found for {selector}")
    return tuple(
        export_recording_preprocessing(
            recording,
            output_root=output_root,
            gravity_config=gravity_config,
            output_dtype=output_dtype,
            overwrite=overwrite,
        )
        for recording in selected
    )


def _build_summary(
    *,
    recording: Recording,
    result: object,
    gravity_config: GravityRemovalConfig,
    values: np.ndarray,
    timestamps: np.ndarray,
    output_dtype: np.dtype,
    output_path: Path,
    timestamps_path: Path,
) -> dict[str, object]:
    method = str(result.method)
    gravity_removed = method != "raw"
    source_ring = recording.ring_0_path.resolve()
    identity = {
        "user": recording.user,
        "action": recording.action,
        "data_id": recording.dataset_id,
    }
    summary: dict[str, object] = {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "artifact_type": PREPROCESSED_IMU_ARTIFACT_TYPE,
        "recording": identity,
        "user": recording.user,
        "action": recording.action,
        "data_id": recording.dataset_id,
        "sample_count": len(values),
        "channel_count": len(PREPROCESSED_IMU_COLUMNS),
        "channel_names": list(PREPROCESSED_IMU_COLUMNS),
        "sampling_rate_hz": float(gravity_config.sampling_rate_hz),
        "standard_gravity_m_s2": STANDARD_GRAVITY_M_S2,
        "acceleration_semantics": result.acceleration_semantics,
        "gravity_removal_method": method,
        "gravity_removed": gravity_removed,
        "units": list(PREPROCESSED_IMU_UNITS),
        "source_file": str(output_path.resolve()),
        "source_imu_path": str(output_path.resolve()),
        "timestamps_path": str(timestamps_path.resolve()),
        "timestamp_source_path": str(timestamps_path.resolve()),
        "timestamp_unit": "microseconds",
        "source": {
            "ring_0_path": str(source_ring),
            "ring_0_sha256": sha256_file(source_ring),
        },
        "gravity_removal": {
            "method": method,
            "sampling_rate_hz": float(gravity_config.sampling_rate_hz),
            "gravity_removed": gravity_removed,
        },
        "storage": {
            "dtype": output_dtype.name,
            "row_semantics": "one complete recording; no segmentation or resampling",
        },
        "provenance": {
            "recording_identity": identity,
            "ring_0_path": str(source_ring),
            "ring_0_sha256": sha256_file(source_ring),
        },
    }
    # The artifact hash is filled after the NPY has been staged.  Publication
    # updates this same dictionary before writing the JSON sidecar.
    summary["source_file_sha256"] = "pending"
    summary["timestamps_sha256"] = "pending"
    summary["timestamp_sha256"] = "pending"
    return summary


def _publish_artifact(
    paths: PreprocessingOutputPaths,
    *,
    values: np.ndarray,
    timestamps: np.ndarray,
    summary: dict[str, object],
    overwrite: bool,
) -> None:
    destination = paths.output_directory
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_dir():
        raise PreprocessingExportError(f"preprocessing output is not a directory: {destination}")
    if destination.exists() and not overwrite:
        raise PreprocessingExportError(
            f"preprocessing output already exists; use --overwrite: {destination}"
        )
    if destination.exists():
        owned = {
            paths.imu_path.name,
            paths.timestamps_path.name,
            paths.summary_path.name,
        }
        unexpected = sorted(path.name for path in destination.iterdir() if path.name not in owned)
        if unexpected:
            raise PreprocessingExportError(
                "refusing to overwrite non-preprocessing files: " + ", ".join(unexpected)
            )
    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        assert staging is not None
        staged_imu = staging / paths.imu_path.name
        staged_timestamps = staging / paths.timestamps_path.name
        staged_summary = staging / paths.summary_path.name
        with staged_imu.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        with staged_timestamps.open("wb") as stream:
            np.save(stream, timestamps, allow_pickle=False)
        summary["source_file_sha256"] = sha256_file(staged_imu)
        summary["source_imu_sha256"] = summary["source_file_sha256"]
        summary["timestamps_sha256"] = sha256_file(staged_timestamps)
        summary["timestamp_sha256"] = summary["timestamps_sha256"]
        staged_summary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _verify_staged(
            staged_imu,
            staged_timestamps,
            staged_summary,
            values=values,
            timestamps=timestamps,
            summary=summary,
        )
        backup = destination.with_name(f".{destination.name}.backup")
        if backup.exists():
            raise PreprocessingExportError(f"cannot publish while backup exists: {backup}")
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except OSError:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        staging = None
    except OSError as error:
        raise PreprocessingExportError(f"could not publish preprocessing artifact: {error}") from error
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def _verify_staged(
    imu_path: Path,
    timestamps_path: Path,
    summary_path: Path,
    *,
    values: np.ndarray,
    timestamps: np.ndarray,
    summary: dict[str, object],
) -> None:
    stored = np.load(imu_path, allow_pickle=False)
    if stored.shape != values.shape or stored.dtype != values.dtype:
        raise PreprocessingExportError("staged preprocessing NPY failed verification")
    validate_preprocessed_imu(stored)
    stored_timestamps = np.load(timestamps_path, allow_pickle=False)
    if stored_timestamps.shape != timestamps.shape or not np.array_equal(
        stored_timestamps, timestamps
    ):
        raise PreprocessingExportError("staged canonical timestamps failed verification")
    _validated_timestamps(stored_timestamps, sample_count=len(values))
    try:
        loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreprocessingExportError("staged preprocessing summary failed verification") from error
    if loaded_summary != summary:
        raise PreprocessingExportError("staged preprocessing summary changed during verification")
    try:
        load_preprocessing_summary(summary_path)
    except PreprocessingIOError as error:
        raise PreprocessingExportError(
            f"staged preprocessing summary failed contract validation: {error}"
        ) from error


def _validated_output_dtype(value: str) -> np.dtype:
    if value not in {"float32", "float64"}:
        raise PreprocessingExportError("output_dtype must be float32 or float64")
    return np.dtype(value)


def _validated_timestamps(values: np.ndarray, *, sample_count: int) -> np.ndarray:
    try:
        timestamps = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PreprocessingExportError("Ring timestamps must be numeric") from error
    if timestamps.ndim != 1 or len(timestamps) != sample_count or len(timestamps) == 0:
        raise PreprocessingExportError(
            "canonical timestamps must be a nonempty vector aligned with preprocessing rows"
        )
    if not np.isfinite(timestamps).all():
        raise PreprocessingExportError("canonical timestamps must be finite")
    if np.any(np.diff(timestamps) < 0.0):
        raise PreprocessingExportError("canonical timestamps must be nondecreasing")
    return np.array(timestamps, copy=True)


def _validate_identity(recording: Recording) -> None:
    if not isinstance(recording.user, str) or not recording.user:
        raise PreprocessingExportError("recording user must be nonempty")
    if not isinstance(recording.action, str) or not recording.action:
        raise PreprocessingExportError("recording action must be nonempty")
    if (
        isinstance(recording.dataset_id, bool)
        or not isinstance(recording.dataset_id, int)
        or recording.dataset_id < 0
    ):
        raise PreprocessingExportError("recording dataset ID must be a nonnegative integer")
