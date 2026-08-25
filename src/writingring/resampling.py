"""Complete-recording IMU resampling between preprocessing and encoding."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Final

import numpy as np
from scipy import signal

from writingring.imu_preprocessing import PREPROCESSED_IMU_COLUMNS, STANDARD_GRAVITY_M_S2
from writingring.preprocessing_io import (
    PREPROCESSING_SCHEMA_VERSION,
    PREPROCESSED_IMU_UNITS,
    RESAMPLED_IMU_ARTIFACT_TYPE,
    PreprocessingIOError,
    load_preprocessed_imu,
    sha256_file,
    validate_preprocessed_imu,
    validate_timestamp_source_provenance,
)


class ResamplingError(ValueError):
    """Raised when a complete-recording resampling artifact is invalid."""


@dataclass(frozen=True, slots=True)
class ResamplingOutputPaths:
    output_directory: Path
    imu_path: Path
    timestamps_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class ResamplingResult:
    paths: ResamplingOutputPaths
    imu: np.ndarray
    timestamps: np.ndarray
    summary: dict[str, object]


TIMESTAMP_UNIT: Final[str] = "microseconds"
POLYPHASE_WINDOW: Final[tuple[str, float]] = ("kaiser", 5.0)
POLYPHASE_PADTYPE: Final[str] = "line"
MAX_RATE_RATIO_DENOMINATOR: Final[int] = 1_000_000


def resampling_output_paths(output_root: Path, relative_recording: Path) -> ResamplingOutputPaths:
    relative = Path(relative_recording)
    if relative.is_absolute() or len(relative.parts) < 3 or any(part in {"", ".", ".."} for part in relative.parts):
        raise ResamplingError("relative_recording must be a safe user/action/data_id path")
    stem = relative.name
    directory = Path(output_root) / relative
    return ResamplingOutputPaths(
        output_directory=directory,
        imu_path=directory / f"{stem}_resampledIMU.npy",
        timestamps_path=directory / f"{stem}_resampled_timestamps_us.npy",
        summary_path=directory / f"{stem}_resampling.json",
    )


def resample_recording(
    *,
    input_imu_path: Path,
    input_summary_path: Path,
    input_timestamps_path: Path,
    output_root: Path,
    relative_recording: Path,
    target_rate_hz: float,
    overwrite: bool = False,
) -> ResamplingResult:
    """Resample a validated nine-channel recording without changing its source."""

    try:
        artifact = load_preprocessed_imu(
            input_imu_path,
            summary_path=input_summary_path,
            allow_gravity_included=True,
        )
    except PreprocessingIOError as error:
        raise ResamplingError(str(error)) from error
    if artifact.summary is None or artifact.summary.sampling_rate_hz is None:
        raise ResamplingError("resampling requires complete preprocessing metadata")
    summary = artifact.summary
    if summary.payload.get("timestamp_unit") != TIMESTAMP_UNIT:
        raise ResamplingError("resampling requires preprocessing timestamp_unit='microseconds'")
    try:
        source_timestamps = np.asarray(np.load(input_timestamps_path, allow_pickle=False), dtype=np.float64)
    except (OSError, ValueError) as error:
        raise ResamplingError(f"could not load source timestamps: {error}") from error
    _validate_source_timestamps(source_timestamps, sample_count=len(artifact.imu))
    try:
        source_timestamp_hash = validate_timestamp_source_provenance(input_timestamps_path, summary)
    except PreprocessingIOError as error:
        raise ResamplingError(str(error)) from error
    source_rate_hz = _finite_positive(summary.sampling_rate_hz, name="source sampling_rate_hz")
    target_rate_hz = _finite_positive(target_rate_hz, name="target_rate_hz")
    if target_rate_hz >= source_rate_hz:
        raise ResamplingError("target_rate_hz must be lower than source sampling_rate_hz; upsampling is unsupported")

    rate_up, rate_down = _polyphase_rate_ratio(source_rate_hz, target_rate_hz)
    values, timestamps = _resample_values(
        artifact.imu,
        source_timestamps,
        source_rate_hz=source_rate_hz,
        target_rate_hz=target_rate_hz,
        rate_up=rate_up,
        rate_down=rate_down,
    )
    paths = resampling_output_paths(output_root, relative_recording)
    output_summary = _build_summary(
        paths=paths,
        values=values,
        timestamps=timestamps,
        source_imu_path=Path(input_imu_path),
        source_summary_path=Path(input_summary_path),
        source_timestamps_path=Path(input_timestamps_path),
        source_imu_hash=sha256_file(Path(input_imu_path)),
        source_summary_hash=sha256_file(Path(input_summary_path)),
        source_timestamp_hash=source_timestamp_hash,
        source_summary=summary.payload,
        source_rate_hz=source_rate_hz,
        target_rate_hz=target_rate_hz,
        rate_up=rate_up,
        rate_down=rate_down,
    )
    _publish(paths, values=values, timestamps=timestamps, summary=output_summary, overwrite=overwrite)
    try:
        verified = load_preprocessed_imu(paths.imu_path, summary_path=paths.summary_path, allow_gravity_included=True)
    except PreprocessingIOError as error:
        raise ResamplingError(f"published resampling artifact failed validation: {error}") from error
    if len(verified.imu) != len(timestamps):
        raise ResamplingError("published resampling artifact row count changed")
    values.setflags(write=False)
    timestamps.setflags(write=False)
    return ResamplingResult(paths=paths, imu=values, timestamps=timestamps, summary=output_summary)


def _polyphase_rate_ratio(source_rate_hz: float, target_rate_hz: float) -> tuple[int, int]:
    ratio = (
        Fraction(str(target_rate_hz)) / Fraction(str(source_rate_hz))
    ).limit_denominator(MAX_RATE_RATIO_DENOMINATOR)
    if ratio.numerator <= 0 or ratio.denominator <= 0:
        raise ResamplingError("could not derive a positive polyphase rate ratio")
    represented_rate = source_rate_hz * ratio.numerator / ratio.denominator
    tolerance = max(1e-12, abs(target_rate_hz) * 1e-12)
    if not math.isclose(represented_rate, target_rate_hz, rel_tol=1e-12, abs_tol=tolerance):
        raise ResamplingError("target sampling rate cannot be represented accurately for polyphase resampling")
    return ratio.numerator, ratio.denominator


def _resample_values(
    source: np.ndarray,
    timestamps_us: np.ndarray,
    *,
    source_rate_hz: float,
    target_rate_hz: float,
    rate_up: int,
    rate_down: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(source) < 2 or timestamps_us[-1] <= timestamps_us[0]:
        raise ResamplingError("resampling requires at least two samples with a positive timestamp span")
    duration_seconds = (timestamps_us[-1] - timestamps_us[0]) / 1_000_000.0
    target_count = int(math.floor(duration_seconds * target_rate_hz + 1e-12)) + 1
    if target_count < 2:
        raise ResamplingError("recording duration is shorter than one target sampling interval")
    target_offsets_us = np.arange(target_count, dtype=np.float64) * (1_000_000.0 / target_rate_hz)
    target_timestamps = timestamps_us[0] + target_offsets_us
    target_timestamps = target_timestamps[target_timestamps <= timestamps_us[-1]]
    if len(target_timestamps) < 2 or np.any(np.diff(target_timestamps) <= 0.0):
        raise ResamplingError("target timestamps are not strictly increasing")

    # Canonical timestamps remain immutable provenance and may contain duplicates.
    # The continuous values are sampled on the source artifact's declared nominal
    # sampling grid; resample_poly performs anti-alias FIR filtering and rational
    # rate conversion in one operation. The timestamp-domain output contract is
    # still the uniform target grid derived above.
    continuous = np.asarray(source[:, 3:9], dtype=np.float64)
    output_continuous = signal.resample_poly(
        continuous,
        up=rate_up,
        down=rate_down,
        axis=0,
        window=POLYPHASE_WINDOW,
        padtype=POLYPHASE_PADTYPE,
    )
    if len(output_continuous) < len(target_timestamps):
        raise ResamplingError(
            "polyphase resampling produced fewer samples than the target timestamp grid"
        )
    output_continuous = output_continuous[: len(target_timestamps)]

    output = np.empty((len(target_timestamps), 9), dtype=np.float64)
    output[:, 3:9] = output_continuous
    output[:, :3] = output_continuous[:, :3] / STANDARD_GRAVITY_M_S2
    try:
        validate_preprocessed_imu(output)
    except PreprocessingIOError as error:
        raise ResamplingError(str(error)) from error
    return output.astype(source.dtype, copy=False), target_timestamps


def _build_summary(
    *,
    paths: ResamplingOutputPaths,
    values: np.ndarray,
    timestamps: np.ndarray,
    source_imu_path: Path,
    source_summary_path: Path,
    source_timestamps_path: Path,
    source_imu_hash: str,
    source_summary_hash: str,
    source_timestamp_hash: str,
    source_summary: dict[str, object],
    source_rate_hz: float,
    target_rate_hz: float,
    rate_up: int,
    rate_down: int,
) -> dict[str, object]:
    recording = source_summary.get("recording")
    if not isinstance(recording, dict):
        raise ResamplingError("preprocessing metadata must declare recording identity")
    return {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "artifact_type": RESAMPLED_IMU_ARTIFACT_TYPE,
        "recording": dict(recording),
        "sample_count": len(values),
        "channel_count": 9,
        "channel_names": list(PREPROCESSED_IMU_COLUMNS),
        "sampling_rate_hz": target_rate_hz,
        "standard_gravity_m_s2": STANDARD_GRAVITY_M_S2,
        "acceleration_semantics": source_summary["acceleration_semantics"],
        "gravity_removal_method": source_summary["gravity_removal_method"],
        "gravity_removed": source_summary["gravity_removed"],
        "units": list(PREPROCESSED_IMU_UNITS),
        "source_file": str(paths.imu_path.resolve()),
        "source_imu_path": str(paths.imu_path.resolve()),
        "timestamps_path": str(paths.timestamps_path.resolve()),
        "timestamp_source_path": str(paths.timestamps_path.resolve()),
        "timestamp_unit": TIMESTAMP_UNIT,
        "resampling": {
            "schema": "complete_recording_resampling_v1",
            "source_imu_path": str(source_imu_path.resolve()),
            "source_imu_sha256": source_imu_hash,
            "source_summary_path": str(source_summary_path.resolve()),
            "source_summary_sha256": source_summary_hash,
            "source_timestamps_path": str(source_timestamps_path.resolve()),
            "source_timestamps_sha256": source_timestamp_hash,
            "source_sampling_rate_hz": source_rate_hz,
            "target_sampling_rate_hz": target_rate_hz,
            "source_timestamp_axis": "canonical_non_decreasing",
            "source_value_time_axis": "nominal_uniform_from_declared_sampling_rate",
            "target_timestamp_grid": "uniform_from_first_source_timestamp_without_extrapolation",
            "resampling_method": "scipy.signal.resample_poly",
            "rate_conversion": {"up": rate_up, "down": rate_down},
            "anti_alias_filter": {
                "family": "polyphase_fir",
                "implementation": "scipy.signal.resample_poly",
                "window": POLYPHASE_WINDOW[0],
                "kaiser_beta": POLYPHASE_WINDOW[1],
                "padtype": POLYPHASE_PADTYPE,
            },
            "resampled_columns": ["acceleration_m_s2", "gyro"],
            "acceleration_g_recomputed_from_m_s2": True,
        },
    }


def _publish(paths: ResamplingOutputPaths, *, values: np.ndarray, timestamps: np.ndarray, summary: dict[str, object], overwrite: bool) -> None:
    destination = paths.output_directory
    if destination.exists() and not overwrite:
        raise ResamplingError(f"resampling output already exists: {destination}; use --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        staged_imu = staging / paths.imu_path.name
        staged_timestamps = staging / paths.timestamps_path.name
        staged_summary = staging / paths.summary_path.name
        np.save(staged_imu, values, allow_pickle=False)
        np.save(staged_timestamps, timestamps, allow_pickle=False)
        summary["source_file_sha256"] = sha256_file(staged_imu)
        summary["source_imu_sha256"] = summary["source_file_sha256"]
        summary["timestamps_sha256"] = sha256_file(staged_timestamps)
        summary["timestamp_sha256"] = summary["timestamps_sha256"]
        staged_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        backup = destination.with_name(f".{destination.name}.backup")
        if backup.exists():
            raise ResamplingError(f"cannot publish while backup exists: {backup}")
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
    except OSError as error:
        raise ResamplingError(f"could not publish resampling artifact: {error}") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _validate_source_timestamps(values: np.ndarray, *, sample_count: int) -> None:
    if values.ndim != 1 or len(values) != sample_count or not np.isfinite(values).all():
        raise ResamplingError("source timestamps must be a finite vector aligned with source IMU rows")
    if np.any(np.diff(values) < 0.0):
        raise ResamplingError("source timestamps must be nondecreasing")


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ResamplingError(f"{name} must be a finite positive number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ResamplingError(f"{name} must be a finite positive number") from error
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ResamplingError(f"{name} must be a finite positive number")
    return numeric
