"""Unified feature/timestamp inputs for recording-level consumers.

The segmentation and alignment consumers operate on one immutable feature
matrix plus the canonical Ring timestamp vector.  Raw Ring input is still
preprocessed in memory for backwards compatibility; SpikeIMU input is loaded
from the published ``spikeIMU.npy`` artifact and never preprocessed again.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from writingring.discovery import Recording
from writingring.gravity import GravityRemovalConfig
from writingring.imu_preprocessing import (
    PREPROCESSED_IMU_COLUMNS,
    preprocess_ring_imu,
)
from writingring.preprocessing_io import PREPROCESSED_IMU_UNITS, sha256_file
from writingring.ring_loader import RingLoadError, load_ring


RAW_RING_FEATURE_SCHEMA = "dual_acceleration_units_v1"
SPIKE_IMU_FEATURE_SCHEMA = "signed_wavelet_events_plus_imu_v1"
SPIKE_IMU_CHANNEL_COUNT = 21
SPIKE_IMU_TRANSIENT_CHANNEL_INDICES = tuple(range(15, 21))
SPIKE_IMU_TRANSIENT_CHANNEL_NAMES = (
    "acceleration_x_m_s2",
    "acceleration_y_m_s2",
    "acceleration_z_m_s2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
)
SPIKE_IMU_UNITS = ("event",) * 15 + ("m/s^2",) * 3 + ("rad/s",) * 3


class RecordingFeatureError(ValueError):
    """Raised when a raw or SpikeIMU feature artifact is not trustworthy."""


@dataclass(frozen=True, slots=True)
class RecordingFeatureInput:
    """Validated feature values and their canonical timestamp provenance."""

    values: np.ndarray
    timestamps_us: np.ndarray

    user: str
    action: str
    dataset_id: int

    input_kind: str
    feature_schema: str
    channel_names: tuple[str, ...]
    units: tuple[str, ...]

    transient_channel_indices: tuple[int, ...]
    transient_channel_names: tuple[str, ...]

    values_path: Path
    metadata_path: Path | None
    timestamps_path: Path | None

    values_sha256: str
    metadata_sha256: str | None
    timestamps_sha256: str

    sampling_rate_hz: float

    @property
    def sample_count(self) -> int:
        """Return the row count shared by values and timestamps."""

        return int(self.values.shape[0])

    @property
    def channel_count(self) -> int:
        """Return the feature channel count."""

        return int(self.values.shape[1])


def load_raw_ring_features(
    recording: Recording,
    *,
    gravity_config: GravityRemovalConfig | None = None,
) -> RecordingFeatureInput:
    """Load one Ring recording and apply the existing preprocessing path."""

    _validate_recording(recording)
    effective_config = gravity_config or GravityRemovalConfig(
        gravity_removal_method="low-pass",
        strict_calibration=False,
    )
    try:
        ring = load_ring(recording)
        result = preprocess_ring_imu(ring, config=effective_config)
    except (RingLoadError, ValueError) as error:
        raise RecordingFeatureError(
            f"could not load and preprocess Ring recording {recording.dataset_id}: {error}"
        ) from error
    timestamps = ring.dataframe["timestamp"].to_numpy(dtype=np.float64, copy=True)
    values = np.asarray(result.imu, dtype=np.float64).copy()
    _validate_feature_arrays(values, timestamps, name="raw Ring features")
    sampling_rate_hz = _sampling_rate_from_config(effective_config)
    return _feature_input(
        values=values,
        timestamps_us=timestamps,
        recording=recording,
        input_kind="raw-ring",
        feature_schema=RAW_RING_FEATURE_SCHEMA,
        channel_names=PREPROCESSED_IMU_COLUMNS,
        units=PREPROCESSED_IMU_UNITS,
        transient_channel_indices=tuple(range(3, 9)),
        transient_channel_names=PREPROCESSED_IMU_COLUMNS[3:],
        values_path=recording.ring_0_path,
        metadata_path=None,
        timestamps_path=None,
        values_sha256=sha256_file(recording.ring_0_path),
        metadata_sha256=None,
        timestamps_sha256=_sha256_array(timestamps),
        sampling_rate_hz=sampling_rate_hz,
    )


def load_spike_imu_features(
    recording: Recording,
    *,
    spike_root: Path,
    expected_sampling_rate_hz: float | None = None,
) -> RecordingFeatureInput:
    """Load and validate one published canonical SpikeIMU recording."""

    _validate_recording(recording)
    root = Path(spike_root)
    directory = root / recording.user / recording.action / str(recording.dataset_id)
    values_path = directory / "spikeIMU.npy"
    metadata_path = directory / "metadata.json"
    if not values_path.is_file():
        raise RecordingFeatureError(f"SpikeIMU artifact is not a regular file: {values_path}")
    if not metadata_path.is_file():
        raise RecordingFeatureError(f"SpikeIMU metadata is not a regular file: {metadata_path}")
    metadata = _load_metadata(metadata_path)
    _validate_spike_metadata_identity(metadata, recording)
    spike_section = metadata.get("spike_imu")
    if not isinstance(spike_section, dict):
        raise RecordingFeatureError("SpikeIMU metadata must contain a spike_imu object")
    if spike_section.get("schema") != SPIKE_IMU_FEATURE_SCHEMA:
        raise RecordingFeatureError(
            f"SpikeIMU metadata schema must be {SPIKE_IMU_FEATURE_SCHEMA!r}"
        )
    if spike_section.get("channel_count") != SPIKE_IMU_CHANNEL_COUNT:
        raise RecordingFeatureError("SpikeIMU metadata channel_count must be 21")
    try:
        values = np.load(values_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RecordingFeatureError(f"could not load SpikeIMU artifact {values_path}: {error}") from error
    values = np.asarray(values)
    _validate_feature_arrays(
        values,
        np.zeros(len(values), dtype=np.float64),
        name="SpikeIMU features",
        validate_timestamps=False,
        expected_channel_count=SPIKE_IMU_CHANNEL_COUNT,
    )
    expected_sample_count = _metadata_sample_count(spike_section, metadata)
    if expected_sample_count != len(values):
        raise RecordingFeatureError("SpikeIMU metadata sample_count does not match spikeIMU.npy")
    expected_values_hash = _first_metadata_value(
        spike_section,
        metadata,
        "sha256",
        "spike_imu_sha256",
        "values_sha256",
    )
    if expected_values_hash is None:
        raise RecordingFeatureError(
            "SpikeIMU metadata must declare the spikeIMU SHA-256 hash"
        )
    values_hash = sha256_file(values_path)
    if expected_values_hash is not None and values_hash != expected_values_hash:
        raise RecordingFeatureError("SpikeIMU metadata hash does not match spikeIMU.npy")

    channel_names = _spike_channel_names(spike_section)
    units = _spike_units(spike_section, channel_count=len(channel_names))
    timestamps_path = _resolve_timestamp_path(metadata, metadata_path=metadata_path)
    if timestamps_path is None:
        raise RecordingFeatureError(
            "SpikeIMU metadata does not identify a canonical timestamp artifact"
        )
    timestamps = _load_timestamps(timestamps_path, sample_count=len(values))
    expected_timestamp_hash = _first_metadata_value(
        metadata,
        spike_section,
        "timestamps_sha256",
        "timestamp_sha256",
    )
    if expected_timestamp_hash is None:
        raise RecordingFeatureError(
            "SpikeIMU metadata must declare the canonical timestamp SHA-256 hash"
        )
    timestamps_hash = sha256_file(timestamps_path)
    if expected_timestamp_hash is not None and timestamps_hash != expected_timestamp_hash:
        raise RecordingFeatureError(
            "SpikeIMU metadata timestamp hash does not match the canonical timestamp artifact"
        )
    timestamp_unit = _metadata_timestamp_unit(metadata)
    if timestamp_unit != "microseconds":
        raise RecordingFeatureError(
            "SpikeIMU metadata timestamp_unit must be 'microseconds'"
        )
    sampling_rate_hz = _metadata_sampling_rate(metadata)
    if expected_sampling_rate_hz is not None:
        expected_rate = _finite_positive(
            expected_sampling_rate_hz,
            name="expected sampling_rate_hz",
        )
        if not math.isclose(
            sampling_rate_hz,
            expected_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RecordingFeatureError(
                "SpikeIMU metadata sampling_rate_hz does not match the requested rate"
            )
    return _feature_input(
        values=values,
        timestamps_us=timestamps,
        recording=recording,
        input_kind="spike-imu",
        feature_schema=SPIKE_IMU_FEATURE_SCHEMA,
        channel_names=channel_names,
        units=units,
        transient_channel_indices=SPIKE_IMU_TRANSIENT_CHANNEL_INDICES,
        transient_channel_names=SPIKE_IMU_TRANSIENT_CHANNEL_NAMES,
        values_path=values_path,
        metadata_path=metadata_path,
        timestamps_path=timestamps_path,
        values_sha256=values_hash,
        metadata_sha256=sha256_file(metadata_path),
        timestamps_sha256=timestamps_hash,
        sampling_rate_hz=sampling_rate_hz,
    )


def load_recording_features(
    recording: Recording,
    *,
    input_kind: str = "raw-ring",
    gravity_config: GravityRemovalConfig | None = None,
    spike_root: Path | None = None,
    expected_sampling_rate_hz: float | None = None,
) -> RecordingFeatureInput:
    """Dispatch to the selected raw-ring or SpikeIMU feature loader."""

    if input_kind == "raw-ring":
        if spike_root is not None:
            raise RecordingFeatureError("spike_root is only valid for input_kind='spike-imu'")
        if expected_sampling_rate_hz is not None:
            raise RecordingFeatureError(
                "expected_sampling_rate_hz is only valid for input_kind='spike-imu'"
            )
        return load_raw_ring_features(recording, gravity_config=gravity_config)
    if input_kind == "spike-imu":
        if spike_root is None:
            raise RecordingFeatureError("spike_root is required for input_kind='spike-imu'")
        if gravity_config is not None:
            raise RecordingFeatureError(
                "gravity preprocessing settings are not valid for input_kind='spike-imu'"
            )
        return load_spike_imu_features(
            recording,
            spike_root=spike_root,
            expected_sampling_rate_hz=expected_sampling_rate_hz,
        )
    raise RecordingFeatureError("input_kind must be 'raw-ring' or 'spike-imu'")


def compute_transient_score_array(values: np.ndarray) -> np.ndarray:
    """Compute the robust first-difference transient score for feature columns."""

    try:
        signals = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RecordingFeatureError("transient feature values must be numeric") from error
    if signals.ndim != 2 or signals.shape[0] == 0 or signals.shape[1] == 0:
        raise RecordingFeatureError("transient feature values must have nonempty shape (N, C)")
    if not np.isfinite(signals).all():
        raise RecordingFeatureError("transient feature values must be finite")
    differences = np.diff(signals, axis=0, prepend=signals[[0]])
    median = np.median(differences, axis=0)
    mad = np.median(np.abs(differences - median), axis=0)
    normalized = (differences - median) / (1.4826 * mad + 1e-12)
    return np.sqrt(np.sum(normalized**2, axis=1))


def _feature_input(
    *,
    values: np.ndarray,
    timestamps_us: np.ndarray,
    recording: Recording,
    input_kind: str,
    feature_schema: str,
    channel_names: tuple[str, ...],
    units: tuple[str, ...],
    transient_channel_indices: tuple[int, ...],
    transient_channel_names: tuple[str, ...],
    values_path: Path,
    metadata_path: Path | None,
    timestamps_path: Path | None,
    values_sha256: str,
    metadata_sha256: str | None,
    timestamps_sha256: str,
    sampling_rate_hz: float,
) -> RecordingFeatureInput:
    _validate_feature_arrays(values, timestamps_us, name="recording features")
    if len(channel_names) != values.shape[1] or len(units) != values.shape[1]:
        raise RecordingFeatureError("feature channel names and units must match values columns")
    if len(transient_channel_indices) != len(transient_channel_names):
        raise RecordingFeatureError("transient channel indices and names must have equal length")
    for index in transient_channel_indices:
        if index < 0 or index >= values.shape[1]:
            raise RecordingFeatureError("transient channel index is outside feature columns")
    rate = _finite_positive(sampling_rate_hz, name="sampling_rate_hz")
    frozen_values = np.array(values, copy=True)
    frozen_timestamps = np.array(timestamps_us, dtype=np.float64, copy=True)
    frozen_values.setflags(write=False)
    frozen_timestamps.setflags(write=False)
    return RecordingFeatureInput(
        values=frozen_values,
        timestamps_us=frozen_timestamps,
        user=recording.user,
        action=recording.action,
        dataset_id=recording.dataset_id,
        input_kind=input_kind,
        feature_schema=feature_schema,
        channel_names=tuple(channel_names),
        units=tuple(units),
        transient_channel_indices=tuple(transient_channel_indices),
        transient_channel_names=tuple(transient_channel_names),
        values_path=Path(values_path),
        metadata_path=None if metadata_path is None else Path(metadata_path),
        timestamps_path=None if timestamps_path is None else Path(timestamps_path),
        values_sha256=values_sha256,
        metadata_sha256=metadata_sha256,
        timestamps_sha256=timestamps_sha256,
        sampling_rate_hz=rate,
    )


def _validate_feature_arrays(
    values: np.ndarray,
    timestamps_us: np.ndarray,
    *,
    name: str,
    validate_timestamps: bool = True,
    expected_channel_count: int | None = None,
) -> None:
    if not isinstance(values, np.ndarray):
        raise RecordingFeatureError(f"{name} must be an ndarray")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise RecordingFeatureError(f"{name} must have nonempty shape (N, C)")
    if expected_channel_count is not None and values.shape[1] != expected_channel_count:
        raise RecordingFeatureError(
            f"{name} must have shape (N, {expected_channel_count})"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise RecordingFeatureError(f"{name} must be numeric")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RecordingFeatureError(f"{name} must be numeric") from error
    if not np.isfinite(numeric).all():
        raise RecordingFeatureError(f"{name} must be finite")
    if not validate_timestamps:
        return
    try:
        timestamps = np.asarray(timestamps_us, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RecordingFeatureError("timestamps_us must be numeric") from error
    if timestamps.ndim != 1 or len(timestamps) != len(values):
        raise RecordingFeatureError("timestamps_us must have one value per feature row")
    if not np.isfinite(timestamps).all():
        raise RecordingFeatureError("timestamps_us must be finite")
    if np.any(np.diff(timestamps) < 0.0):
        raise RecordingFeatureError("timestamps_us must be nondecreasing")


def _load_timestamps(path: Path, *, sample_count: int) -> np.ndarray:
    try:
        timestamps = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RecordingFeatureError(f"could not load canonical timestamps {path}: {error}") from error
    try:
        values = np.asarray(timestamps, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RecordingFeatureError(f"canonical timestamps must be numeric: {path}") from error
    _validate_feature_arrays(
        np.zeros((sample_count, 1), dtype=np.float64),
        values,
        name="canonical timestamp feature placeholder",
    )
    return np.array(values, copy=True)


def _load_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecordingFeatureError(f"could not parse SpikeIMU metadata {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RecordingFeatureError("SpikeIMU metadata must be a JSON object")
    return payload


def _validate_spike_metadata_identity(metadata: Mapping[str, object], recording: Recording) -> None:
    source = metadata.get("recording")
    if source is None and isinstance(metadata.get("source"), dict):
        source = metadata["source"].get("recording")  # type: ignore[index]
    if not isinstance(source, dict):
        raise RecordingFeatureError("SpikeIMU metadata must declare recording identity")
    expected = {"user": recording.user, "action": recording.action, "data_id": recording.dataset_id}
    for key, value in expected.items():
        if source.get(key) != value:
            raise RecordingFeatureError(f"SpikeIMU metadata recording.{key} does not match input recording")


def _metadata_sample_count(section: Mapping[str, object], metadata: Mapping[str, object]) -> int:
    value = section.get("sample_count")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecordingFeatureError("SpikeIMU metadata sample_count must be a positive integer")
    input_section = metadata.get("input")
    if isinstance(input_section, dict) and "sample_count" in input_section:
        input_count = input_section["sample_count"]
        if input_count != value:
            raise RecordingFeatureError(
                "SpikeIMU metadata input.sample_count conflicts with spike_imu.sample_count"
            )
    return value


def _spike_channel_names(section: Mapping[str, object]) -> tuple[str, ...]:
    value = section.get("channel_names")
    if not isinstance(value, list) or len(value) != SPIKE_IMU_CHANNEL_COUNT or not all(
        isinstance(name, str) and name for name in value
    ):
        raise RecordingFeatureError("SpikeIMU metadata channel_names must contain 21 names")
    names = tuple(value)
    if names[15:] != SPIKE_IMU_TRANSIENT_CHANNEL_NAMES:
        raise RecordingFeatureError("SpikeIMU metadata trailing channel_names do not match the IMU schema")
    return names


def _spike_units(section: Mapping[str, object], *, channel_count: int) -> tuple[str, ...]:
    value = section.get("units")
    if value is None:
        raise RecordingFeatureError("SpikeIMU metadata units are required")
    if not isinstance(value, list) or len(value) != channel_count or not all(
        isinstance(unit, str) and unit for unit in value
    ):
        raise RecordingFeatureError("SpikeIMU metadata units must contain 21 values")
    units = tuple(value)
    if units[15:] != SPIKE_IMU_UNITS[15:]:
        raise RecordingFeatureError(
            "SpikeIMU metadata trailing units do not match the IMU schema"
        )
    return units


def _resolve_timestamp_path(
    metadata: Mapping[str, object],
    *,
    metadata_path: Path,
) -> Path | None:
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    references = source.get("metadata_references") if isinstance(source, dict) else {}
    candidates: list[object] = [
        metadata.get("timestamps_path"),
        metadata.get("timestamp_source_path"),
        source.get("timestamps_path") if isinstance(source, dict) else None,
        source.get("timestamp_source_path") if isinstance(source, dict) else None,
        references.get("timestamp_source_path") if isinstance(references, dict) else None,
    ]
    if not any(isinstance(value, str) and value for value in candidates):
        raise RecordingFeatureError(
            "SpikeIMU metadata must identify a canonical timestamp artifact"
        )
    for value in candidates:
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = metadata_path.parent / path
        if path.is_file():
            return path
        raise RecordingFeatureError(f"canonical timestamp artifact is not a regular file: {path}")
    return None


def _metadata_sampling_rate(metadata: Mapping[str, object]) -> float:
    value = metadata.get("sampling_rate_hz")
    if value is None:
        raise RecordingFeatureError(
            "SpikeIMU metadata sampling_rate_hz is required"
        )
    return _finite_positive(value, name="SpikeIMU metadata sampling_rate_hz")


def _metadata_timestamp_unit(metadata: Mapping[str, object]) -> str | None:
    value = metadata.get("timestamp_unit")
    if value is None and isinstance(metadata.get("source"), dict):
        value = metadata["source"].get("timestamp_unit")  # type: ignore[index]
    if value is not None and not isinstance(value, str):
        raise RecordingFeatureError("SpikeIMU metadata timestamp_unit must be a string")
    return value


def _first_metadata_value(
    first: Mapping[str, object],
    second: Mapping[str, object],
    *keys: str,
) -> str | None:
    for key in keys:
        value = first.get(key)
        if value is None:
            value = second.get(key)
        if value is not None:
            if not isinstance(value, str) or len(value) != 64:
                raise RecordingFeatureError(f"metadata {key} must be a SHA-256 hex digest")
            lowered = value.lower()
            if any(character not in "0123456789abcdef" for character in lowered):
                raise RecordingFeatureError(f"metadata {key} must be a SHA-256 hex digest")
            return lowered
    return None


def _sampling_rate_from_config(config: GravityRemovalConfig) -> float:
    return _finite_positive(config.sampling_rate_hz, name="sampling_rate_hz")


def _validate_recording(recording: Recording) -> None:
    if not isinstance(recording, Recording):
        raise RecordingFeatureError("recording must be a Recording")


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise RecordingFeatureError(f"{name} must be a finite positive number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RecordingFeatureError(f"{name} must be a finite positive number") from error
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise RecordingFeatureError(f"{name} must be a finite positive number")
    return numeric


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()
