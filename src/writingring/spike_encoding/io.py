"""Read-only input and sequence-boundary validation for spike encoding."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from collections.abc import Mapping
from typing import Final

import numpy as np

from writingring.imu_preprocessing import PREPROCESSED_IMU_COLUMNS
from writingring.preprocessing_io import (
    PreprocessedIMUSummary,
    PreprocessingIOError,
    load_preprocessed_imu,
    load_preprocessing_summary,
    validate_preprocessing_summary,
)
from writingring.spike_encoding.contracts import SpikeEncodingError


_ACCELERATION_G_COLUMNS: Final[tuple[str, str, str]] = PREPROCESSED_IMU_COLUMNS[:3]
SUPPORTED_METHOD_SEMANTICS: Final[dict[str, str]] = {
    "raw": "measured_acceleration_with_gravity",
    "low-pass": "gravity_removed_linear_acceleration",
    "madgwick": "gravity_removed_linear_acceleration",
    "xylo-rotate-and-remove-gravity": "xylo_gravity_removed_acceleration",
    "xylo": "xylo_gravity_removed_acceleration",
}
SOURCE_METADATA_PATH_KEYS: Final[tuple[str, ...]] = (
    "labels_path",
    "segment_offsets_path",
    "segment_lengths_path",
    "timestamp_source_path",
    "segments_manifest_path",
)


@dataclass(frozen=True, slots=True)
class SpikeEncodingInput:
    """Validated immutable source values for one complete recording."""

    source_imu_path: Path
    preprocessed_imu: np.ndarray
    acceleration_g: np.ndarray

    @property
    def raw_imu_path(self) -> Path:
        """Deprecated compatibility alias for the preprocessed source path."""

        return self.source_imu_path

    @property
    def raw_imu(self) -> np.ndarray:
        """Deprecated compatibility alias for the preprocessed source array."""

        return self.preprocessed_imu


SpikeEncodingSourceSummary = PreprocessedIMUSummary


def load_spike_encoding_input(
    path: Path,
    *,
    summary_path: Path | None = None,
    allow_gravity_included: bool = False,
    expected_sampling_rate_hz: float | None = None,
    expected_recording: Mapping[str, object] | None = None,
) -> SpikeEncodingInput:
    """Load one complete nine-channel preprocessed recording.

    The physical dual-acceleration-unit contract is always checked.  When a
    summary is supplied (or colocated for the new artifact naming scheme),
    its provenance and gravity-removal semantics are checked as well.
    """

    try:
        artifact = load_preprocessed_imu(
            path,
            summary_path=summary_path,
            allow_gravity_included=allow_gravity_included,
            expected_sampling_rate_hz=expected_sampling_rate_hz,
            expected_recording=expected_recording,
        )
    except PreprocessingIOError as error:
        raise SpikeEncodingError(str(error)) from error
    return SpikeEncodingInput(
        source_imu_path=artifact.path,
        preprocessed_imu=artifact.imu,
        acceleration_g=artifact.acceleration_g,
    )


def load_sequence_offsets(path: Path, *, sample_count: int) -> np.ndarray:
    """Load offsets that partition all samples into nonempty sequences."""

    source = Path(path)
    if not source.is_file():
        raise SpikeEncodingError(f"sequence offsets file is not a regular file: {source}")
    try:
        offsets = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SpikeEncodingError(f"could not load sequence offsets {source}: {error}") from error
    return validate_sequence_offsets(offsets, sample_count=sample_count)


def single_array_offsets(*, sample_count: int) -> np.ndarray:
    """Return the explicit single-sequence partition for ``sample_count`` rows."""

    return validate_sequence_offsets(np.array([0, sample_count]), sample_count=sample_count)


def load_and_validate_timestamps(
    path: Path,
    *,
    sample_count: int,
    sampling_rate_hz: float,
) -> np.ndarray:
    """Load optional numeric timestamps and validate their recording alignment.

    Timestamps are intentionally not passed to an encoder.  They only confirm
    that the source rows can be reused by a later segmentation stage.
    """

    source = Path(path)
    if not source.is_file():
        raise SpikeEncodingError(f"timestamps file is not a regular file: {source}")
    try:
        timestamps = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SpikeEncodingError(f"could not load timestamps {source}: {error}") from error
    try:
        values = np.asarray(timestamps, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SpikeEncodingError("timestamps must be numeric") from error
    if values.ndim != 1 or len(values) != sample_count:
        raise SpikeEncodingError(
            f"timestamps must have shape ({sample_count},) to match input IMU rows"
        )
    if not np.isfinite(values).all():
        raise SpikeEncodingError("timestamps must contain only finite values")
    intervals = np.diff(values)
    if len(intervals) and np.any(intervals <= 0.0):
        raise SpikeEncodingError("timestamps must be strictly increasing")
    expected_interval = 1.0 / _optional_finite_positive(
        sampling_rate_hz, name="sampling_rate_hz"
    )
    if len(intervals) and not math.isclose(
        float(np.median(intervals)), expected_interval, rel_tol=0.05, abs_tol=0.0
    ):
        raise SpikeEncodingError(
            "timestamp sampling interval is inconsistent with sampling_rate_hz"
        )
    values.setflags(write=False)
    return values


def validate_source_metadata_paths(
    paths: Mapping[str, Path | None] | None,
) -> dict[str, str | None]:
    """Validate optional, read-only source sidecars for summary provenance."""

    if paths is None:
        return {key: None for key in SOURCE_METADATA_PATH_KEYS}
    unknown = sorted(set(paths) - set(SOURCE_METADATA_PATH_KEYS))
    if unknown:
        raise SpikeEncodingError(
            "unknown source metadata path keys: " + ", ".join(unknown)
        )
    validated: dict[str, str | None] = {}
    for key in SOURCE_METADATA_PATH_KEYS:
        path = paths.get(key)
        if path is None:
            validated[key] = None
            continue
        source = Path(path)
        if not source.is_file():
            raise SpikeEncodingError(
                f"source metadata path is not a regular file: {source}"
            )
        validated[key] = str(source.resolve())
    return validated


def validate_sequence_offsets(offsets: np.ndarray, *, sample_count: int) -> np.ndarray:
    """Validate an integer, complete, strictly increasing sequence partition."""

    if not isinstance(sample_count, int) or sample_count <= 0:
        raise SpikeEncodingError("sample_count must be a positive integer")
    if not isinstance(offsets, np.ndarray) or offsets.ndim != 1 or len(offsets) < 2:
        raise SpikeEncodingError("sequence offsets must be a one-dimensional array with at least two values")
    if not np.issubdtype(offsets.dtype, np.integer):
        raise SpikeEncodingError("sequence offsets must have an integer dtype")
    values = offsets.astype(np.int64, copy=True)
    if int(values[0]) != 0 or int(values[-1]) != sample_count:
        raise SpikeEncodingError(
            f"sequence offsets must start at 0 and end at sample_count={sample_count}"
        )
    if np.any(np.diff(values) <= 0):
        raise SpikeEncodingError("sequence offsets must be strictly increasing")
    values.setflags(write=False)
    return values


def load_encoder_settings(path: Path) -> dict[str, object]:
    """Load one encoder-private JSON object without interpreting its fields."""

    source = Path(path)
    if not source.is_file():
        raise SpikeEncodingError(f"encoder settings file is not a regular file: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SpikeEncodingError(f"could not parse encoder settings {source}: {error}") from error
    if not isinstance(payload, dict):
        raise SpikeEncodingError("encoder settings JSON must be an object")
    return dict(payload)


def load_spike_encoding_source_summary(path: Path) -> SpikeEncodingSourceSummary:
    """Read source-independent preprocessing metadata.

    The old function name is retained for callers that still have a
    segmentation summary.  The parser accepts that legacy shape while using
    the same method, channel, and provenance checks as new summaries.
    """

    try:
        return load_preprocessing_summary(path)
    except PreprocessingIOError as error:
        message = str(error).replace("preprocessing summary", "input summary")
        raise SpikeEncodingError(message) from error


def resolve_sampling_rate_hz(
    settings: dict[str, object],
    source_summary: SpikeEncodingSourceSummary | None,
) -> dict[str, object]:
    """Apply the documented settings/summary sampling-rate precedence strictly."""

    resolved = dict(settings)
    settings_value = _optional_finite_positive(
        resolved.get("sampling_rate_hz"), name="settings sampling_rate_hz"
    )
    summary_value = None if source_summary is None else source_summary.sampling_rate_hz
    if settings_value is None and summary_value is None:
        raise SpikeEncodingError(
            "sampling_rate_hz is required in encoder settings or input summary"
        )
    if settings_value is not None and summary_value is not None and not math.isclose(
        settings_value, summary_value, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SpikeEncodingError(
            "settings sampling_rate_hz conflicts with input summary sampling_rate_hz"
        )
    resolved["sampling_rate_hz"] = settings_value if settings_value is not None else summary_value
    return resolved


def _validate_summary_schema(payload: dict[str, object]) -> None:
    channel_count = payload.get("channel_count")
    if channel_count is not None and channel_count != len(PREPROCESSED_IMU_COLUMNS):
        raise SpikeEncodingError("input summary channel_count must be 9")
    channel_names = payload.get("channel_names")
    if channel_names is not None:
        if not isinstance(channel_names, list) or tuple(channel_names) != PREPROCESSED_IMU_COLUMNS:
            raise SpikeEncodingError("input summary channel_names do not match the nine-channel IMU schema")
    standard_gravity = payload.get("standard_gravity_m_s2")
    if standard_gravity is not None:
        _optional_finite_positive(standard_gravity, name="input summary standard_gravity_m_s2")


def _validate_method_semantics(method: object, semantics: object) -> None:
    if (method is None) != (semantics is None):
        raise SpikeEncodingError(
            "input summary gravity removal method and acceleration_semantics must be provided together"
        )
    if method is None:
        return
    if not isinstance(method, str) or method not in SUPPORTED_METHOD_SEMANTICS:
        choices = ", ".join(SUPPORTED_METHOD_SEMANTICS)
        raise SpikeEncodingError(
            f"input summary gravity removal method must be one of: {choices}"
        )
    if not isinstance(semantics, str):
        raise SpikeEncodingError("input summary acceleration_semantics must be a string")
    expected = SUPPORTED_METHOD_SEMANTICS[method]
    if semantics != expected:
        raise SpikeEncodingError(
            "input summary acceleration_semantics does not match gravity removal method "
            f"{method!r}; expected {expected!r}"
        )


def _optional_finite_positive(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SpikeEncodingError(f"{name} must be a finite positive number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise SpikeEncodingError(f"{name} must be a finite positive number") from error
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise SpikeEncodingError(f"{name} must be a finite positive number")
    return numeric
