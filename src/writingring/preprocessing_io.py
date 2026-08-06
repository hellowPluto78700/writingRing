"""File-level I/O contracts for complete preprocessed Ring IMU recordings.

The in-memory :func:`writingring.imu_preprocessing.preprocess_ring_imu`
function already produces the canonical nine-channel tensor.  This module
defines the durable handoff used when that tensor is written to disk and
consumed by a later spike-encoding process.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from collections.abc import Mapping
from typing import Final

import numpy as np

from writingring.imu_preprocessing import (
    IMU_PREPROCESSING_METHODS,
    PREPROCESSED_IMU_COLUMNS,
    STANDARD_GRAVITY_M_S2,
)


PREPROCESSING_SCHEMA_VERSION: Final[int] = 1
PREPROCESSED_IMU_ARTIFACT_TYPE: Final[str] = "preprocessed_imu"
DEFAULT_ACCELERATION_RTOL: Final[float] = 1e-6
DEFAULT_ACCELERATION_ATOL: Final[float] = 1e-7
PREPROCESSED_IMU_UNITS: Final[tuple[str, ...]] = (
    "g", "g", "g",
    "m/s^2", "m/s^2", "m/s^2",
    "rad/s", "rad/s", "rad/s",
)
REQUIRED_PREPROCESSING_METADATA_FIELDS: Final[tuple[str, ...]] = (
    "sample_count",
    "channel_names",
    "sampling_rate_hz",
    "gravity_removal_method",
    "gravity_removed",
    "acceleration_semantics",
    "units",
)

_METHOD_SEMANTICS: Final[dict[str, str]] = {
    "raw": "measured_acceleration_with_gravity",
    "low-pass": "gravity_removed_linear_acceleration",
    "madgwick": "gravity_removed_linear_acceleration",
    "xylo-rotate-and-remove-gravity": "xylo_gravity_removed_acceleration",
    # ``xylo`` appeared in early handoff notes; accept it as a metadata alias
    # while the executable preprocessing method retains its full name.
    "xylo": "xylo_gravity_removed_acceleration",
}


class PreprocessingIOError(ValueError):
    """Raised when a preprocessed IMU artifact cannot satisfy its contract."""


@dataclass(frozen=True, slots=True)
class PreprocessedIMUSummary:
    """Validated metadata associated with one preprocessed IMU artifact."""

    path: Path
    payload: dict[str, object]
    schema_version: int | None
    sample_count: int | None
    channel_names: tuple[str, ...] | None
    sampling_rate_hz: float | None
    standard_gravity_m_s2: float | None
    acceleration_semantics: str | None
    gravity_removal_method: str | None
    gravity_removed: bool | None
    units: tuple[str, ...] | None
    source_file: Path | None
    source_file_sha256: str | None
    recording: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class PreprocessedIMUArtifact:
    """Validated complete recording and its optional source summary."""

    path: Path
    imu: np.ndarray
    acceleration_g: np.ndarray
    summary: PreprocessedIMUSummary | None

    @property
    def shape(self) -> tuple[int, int]:
        """Expose the underlying array shape for array-oriented callers."""

        return self.imu.shape

    @property
    def dtype(self) -> np.dtype:
        """Expose the underlying array dtype for array-oriented callers."""

        return self.imu.dtype

    def __len__(self) -> int:
        return len(self.imu)

    def __getitem__(self, item: object) -> object:
        return self.imu[item]  # type: ignore[index]

    def __array__(self, dtype: object = None) -> np.ndarray:
        return np.asarray(self.imu, dtype=dtype)


def validate_preprocessed_imu(
    values: np.ndarray,
    *,
    standard_gravity_m_s2: float = STANDARD_GRAVITY_M_S2,
    rtol: float = DEFAULT_ACCELERATION_RTOL,
    atol: float = DEFAULT_ACCELERATION_ATOL,
) -> np.ndarray:
    """Validate and copy the canonical complete-recording nine-channel tensor.

    The first three columns are acceleration in ``g``; columns three through
    five are the same signal in ``m/s²``; the final three columns are gyro
    values.  The returned array is writable so callers can choose their own
    read-only boundary after copying it into a dataclass.
    """

    if not isinstance(values, np.ndarray):
        raise PreprocessingIOError("preprocessed IMU must be an ndarray")
    if values.ndim != 2 or values.shape[1] != len(PREPROCESSED_IMU_COLUMNS) or len(values) == 0:
        raise PreprocessingIOError("preprocessed IMU must have nonempty shape (N, 9)")
    if not np.issubdtype(values.dtype, np.number):
        raise PreprocessingIOError("preprocessed IMU dtype must be numeric")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PreprocessingIOError("preprocessed IMU must be numeric") from error
    if not np.isfinite(numeric).all():
        raise PreprocessingIOError("preprocessed IMU must contain only finite values")
    gravity = _finite_positive(standard_gravity_m_s2, name="standard_gravity_m_s2")
    with np.errstate(invalid="ignore", over="ignore"):
        acceleration_m_s2 = numeric[:, 3:6]
        expected_m_s2 = numeric[:, :3] * gravity
    if not np.allclose(acceleration_m_s2, expected_m_s2, rtol=rtol, atol=atol):
        raise PreprocessingIOError(
            "g and m/s^2 acceleration channels are inconsistent; expected "
            "acceleration_m_s2 = acceleration_g * standard_gravity_m_s2"
        )
    return np.array(values, copy=True)


def load_preprocessed_imu(
    path: Path,
    *,
    summary_path: Path | None = None,
    allow_gravity_included: bool = False,
    expected_sampling_rate_hz: float | None = None,
    expected_recording: Mapping[str, object] | None = None,
) -> PreprocessedIMUArtifact:
    """Load one complete preprocessed IMU artifact without modifying it.

    A summary is optional for backwards compatibility with older segmentation
    exports.  New ``*_preprocessedIMU.npy`` files automatically use the
    colocated ``*_preprocessing.json`` when it exists.
    """

    source = Path(path)
    if not source.is_file():
        raise PreprocessingIOError(f"preprocessed IMU file is not a regular file: {source}")
    try:
        loaded = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PreprocessingIOError(f"could not load preprocessed IMU {source}: {error}") from error
    values = validate_preprocessed_imu(loaded)

    resolved_summary_path = Path(summary_path) if summary_path is not None else _default_summary_path(source)
    summary: PreprocessedIMUSummary | None = None
    if resolved_summary_path is not None and resolved_summary_path.is_file():
        summary = load_preprocessing_summary(resolved_summary_path)
        validate_preprocessing_summary(
            summary,
            input_path=source,
            sample_count=len(values),
            expected_sampling_rate_hz=expected_sampling_rate_hz,
            expected_recording=expected_recording,
            allow_gravity_included=allow_gravity_included,
            require_complete=(
                _is_preprocessed_imu_artifact_path(source) or allow_gravity_included
            ),
        )
    elif summary_path is not None:
        raise PreprocessingIOError(
            f"preprocessing summary is not a regular file: {resolved_summary_path}"
        )
    elif source.name.endswith("_preprocessedIMU.npy"):
        raise PreprocessingIOError(
            "complete preprocessed IMU artifacts require a colocated preprocessing summary"
        )
    if allow_gravity_included and summary is None:
        raise PreprocessingIOError(
            "--allow-gravity-included requires a preprocessing summary explicitly declaring "
            "raw gravity_removed=false"
        )
    if summary is not None and summary.standard_gravity_m_s2 is not None:
        if not math.isclose(
            summary.standard_gravity_m_s2,
            STANDARD_GRAVITY_M_S2,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise PreprocessingIOError(
                "preprocessing summary standard_gravity_m_s2 does not match "
                f"{STANDARD_GRAVITY_M_S2}"
            )
    values.setflags(write=False)
    acceleration_g = np.array(values[:, :3], copy=True)
    acceleration_g.setflags(write=False)
    return PreprocessedIMUArtifact(
        path=source,
        imu=values,
        acceleration_g=acceleration_g,
        summary=summary,
    )


def load_preprocessing_summary(path: Path) -> PreprocessedIMUSummary:
    """Read and validate one source-independent preprocessing summary."""

    source = Path(path)
    if not source.is_file():
        raise PreprocessingIOError(f"preprocessing summary is not a regular file: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreprocessingIOError(f"could not parse preprocessing summary {source}: {error}") from error
    if not isinstance(payload, dict):
        raise PreprocessingIOError("preprocessing summary JSON must be an object")
    return _parse_summary(
        source,
        payload,
        require_complete=_is_preprocessing_summary_path(source),
    )


def validate_preprocessing_summary(
    summary: PreprocessedIMUSummary,
    *,
    input_path: Path | None = None,
    sample_count: int | None = None,
    expected_sampling_rate_hz: float | None = None,
    expected_recording: Mapping[str, object] | None = None,
    allow_gravity_included: bool = False,
    require_complete: bool = False,
) -> None:
    """Cross-check summary provenance against the artifact being consumed."""

    if not isinstance(summary, PreprocessedIMUSummary):
        raise PreprocessingIOError("summary must be a PreprocessedIMUSummary")
    if require_complete or _summary_declares_preprocessed_artifact(summary.payload):
        _validate_required_preprocessing_metadata(summary)
    if summary.sample_count is not None and sample_count is not None:
        if summary.sample_count != sample_count:
            raise PreprocessingIOError(
                "preprocessing summary sample_count does not match input IMU rows"
            )
    if summary.channel_names is not None and summary.channel_names != PREPROCESSED_IMU_COLUMNS:
        raise PreprocessingIOError("preprocessing summary channel_names do not match the nine-channel IMU schema")
    if summary.source_file is not None and input_path is not None:
        if summary.source_file.resolve() != Path(input_path).resolve():
            raise PreprocessingIOError(
                "preprocessing summary source_file does not match the input IMU"
            )
    if summary.source_file_sha256 is not None and input_path is not None:
        actual_hash = sha256_file(Path(input_path))
        if actual_hash != summary.source_file_sha256:
            raise PreprocessingIOError(
                "preprocessing summary source_file_sha256 does not match the input IMU"
            )
    if expected_sampling_rate_hz is not None and summary.sampling_rate_hz is not None:
        expected = _finite_positive(expected_sampling_rate_hz, name="expected_sampling_rate_hz")
        if not math.isclose(
            expected,
            summary.sampling_rate_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise PreprocessingIOError(
                "preprocessing summary sampling_rate_hz does not match encoder settings"
            )
    if expected_recording is not None and summary.recording is not None:
        for key, expected_value in expected_recording.items():
            if key in summary.recording and summary.recording[key] != expected_value:
                raise PreprocessingIOError(
                    f"preprocessing summary recording.{key} does not match input recording"
                )
    if not allow_gravity_included and _summary_includes_gravity(summary):
        raise PreprocessingIOError(
            "gravity-included preprocessing is not allowed; use "
            "--allow-gravity-included when measured acceleration is intentional"
        )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    source = Path(path)
    if not source.is_file():
        raise PreprocessingIOError(f"cannot hash a non-file path: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PreprocessingIOError(f"could not hash {source}: {error}") from error
    return digest.hexdigest()


def _parse_summary(
    path: Path,
    payload: dict[str, object],
    *,
    require_complete: bool = False,
) -> PreprocessedIMUSummary:
    schema_version = _optional_positive_int(payload.get("schema_version"), name="schema_version")
    artifact_type = payload.get("artifact_type")
    if artifact_type is not None and artifact_type != PREPROCESSED_IMU_ARTIFACT_TYPE:
        raise PreprocessingIOError(
            f"preprocessing summary artifact_type must be {PREPROCESSED_IMU_ARTIFACT_TYPE!r}"
        )

    channel_count = payload.get("channel_count")
    if channel_count is not None and channel_count != len(PREPROCESSED_IMU_COLUMNS):
        raise PreprocessingIOError("preprocessing summary channel_count must be 9")
    channel_names_value = payload.get("channel_names")
    channel_names: tuple[str, ...] | None = None
    if channel_names_value is not None:
        if not isinstance(channel_names_value, list) or not all(
            isinstance(value, str) for value in channel_names_value
        ):
            raise PreprocessingIOError("preprocessing summary channel_names must be a string list")
        channel_names = tuple(channel_names_value)
        if channel_names != PREPROCESSED_IMU_COLUMNS:
            raise PreprocessingIOError("preprocessing summary channel_names do not match the nine-channel IMU schema")

    gravity_payload = payload.get("gravity_removal")
    if gravity_payload is not None and not isinstance(gravity_payload, dict):
        raise PreprocessingIOError("preprocessing summary gravity_removal must be an object")
    gravity = gravity_payload if isinstance(gravity_payload, dict) else {}
    root_method = payload.get("gravity_removal_method")
    nested_method = gravity.get("method")
    if root_method is not None and nested_method is not None and root_method != nested_method:
        raise PreprocessingIOError(
            "preprocessing summary gravity_removal_method conflicts between nested and root fields"
        )
    method = nested_method if nested_method is not None else root_method
    supported_methods = (*IMU_PREPROCESSING_METHODS, "xylo")
    if method is not None and (not isinstance(method, str) or method not in supported_methods):
        choices = ", ".join(supported_methods)
        raise PreprocessingIOError(
            f"preprocessing summary gravity_removal_method must be one of: {choices}"
        )

    semantics = payload.get("acceleration_semantics")
    if semantics is not None and not isinstance(semantics, str):
        raise PreprocessingIOError("preprocessing summary acceleration_semantics must be a string")
    _validate_method_semantics(method, semantics)

    sampling_rate = _resolve_numeric_summary_field(
        payload,
        gravity,
        "sampling_rate_hz",
        name="preprocessing summary sampling_rate_hz",
    )
    standard_gravity = _resolve_numeric_summary_field(
        payload,
        gravity,
        "standard_gravity_m_s2",
        name="preprocessing summary standard_gravity_m_s2",
    )
    if standard_gravity is not None and not math.isclose(
        standard_gravity,
        STANDARD_GRAVITY_M_S2,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PreprocessingIOError(
            "preprocessing summary standard_gravity_m_s2 does not match "
            f"{STANDARD_GRAVITY_M_S2}"
        )
    declared_preprocessed_artifact = (
        require_complete or _summary_declares_preprocessed_artifact(payload)
    )
    units = _parse_units(payload.get("units"), required=declared_preprocessed_artifact)
    sample_count = _optional_positive_int(
        payload.get("sample_count"), name="sample_count"
    )

    root_gravity_removed = payload.get("gravity_removed")
    nested_gravity_removed = gravity.get("gravity_removed")
    for value in (root_gravity_removed, nested_gravity_removed):
        if value is not None and not isinstance(value, bool):
            raise PreprocessingIOError("preprocessing summary gravity_removed must be a boolean")
    if (
        root_gravity_removed is not None
        and nested_gravity_removed is not None
        and root_gravity_removed != nested_gravity_removed
    ):
        raise PreprocessingIOError(
            "preprocessing summary gravity_removed conflicts between nested and root fields"
        )
    gravity_removed = root_gravity_removed
    if gravity_removed is None:
        gravity_removed = nested_gravity_removed
    if gravity_removed is not None and not isinstance(gravity_removed, bool):
        raise PreprocessingIOError("preprocessing summary gravity_removed must be a boolean")
    if method is not None:
        inferred_removed = method != "raw"
        if gravity_removed is not None and gravity_removed != inferred_removed:
            raise PreprocessingIOError(
                "preprocessing summary gravity_removed conflicts with gravity removal method"
            )
        if gravity_removed is None:
            gravity_removed = inferred_removed

    source_file, source_hash = _summary_source(path, payload)
    if source_hash is not None and (
        len(source_hash) != 64 or any(character not in "0123456789abcdef" for character in source_hash)
    ):
        raise PreprocessingIOError("preprocessing summary source_file_sha256 must be a SHA-256 hex digest")
    recording = _summary_recording(payload)
    summary = PreprocessedIMUSummary(
        path=path,
        payload=dict(payload),
        schema_version=schema_version,
        sample_count=sample_count,
        channel_names=channel_names,
        sampling_rate_hz=sampling_rate,
        standard_gravity_m_s2=standard_gravity,
        acceleration_semantics=semantics,
        gravity_removal_method=method,
        gravity_removed=gravity_removed,
        units=units,
        source_file=source_file,
        source_file_sha256=source_hash,
        recording=recording,
    )
    if declared_preprocessed_artifact:
        _validate_required_preprocessing_metadata(summary)
    return summary


def _summary_source(
    summary_path: Path,
    payload: Mapping[str, object],
) -> tuple[Path | None, str | None]:
    nested = payload.get("source")
    source = nested if isinstance(nested, dict) else {}
    source_value = _first_value(
        payload,
        source,
        "source_file",
        "source_imu_path",
        "preprocessed_imu_path",
        "artifact_path",
        "imu_path",
    )
    source_file: Path | None = None
    if source_value is not None:
        if not isinstance(source_value, str) or not source_value:
            raise PreprocessingIOError("preprocessing summary source_file must be a nonempty string")
        source_file = Path(source_value)
        if not source_file.is_absolute():
            source_file = summary_path.parent / source_file
    hash_value = _first_value(
        payload,
        source,
        "source_file_sha256",
        "source_imu_sha256",
        "artifact_sha256",
        "sha256",
    )
    source_hash = None if hash_value is None else str(hash_value).lower()
    return source_file, source_hash


def _summary_recording(payload: Mapping[str, object]) -> dict[str, object] | None:
    value = payload.get("recording")
    if value is None:
        value = payload.get("identity")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PreprocessingIOError("preprocessing summary recording must be an object")
    result = dict(value)
    for key in ("user", "action", "data_id"):
        if key in result and not isinstance(result[key], (str, int)):
            raise PreprocessingIOError(f"preprocessing summary recording.{key} is invalid")
    return result


def _summary_includes_gravity(summary: PreprocessedIMUSummary) -> bool:
    if summary.gravity_removed is False:
        return True
    if summary.gravity_removal_method == "raw":
        return True
    if summary.acceleration_semantics == "measured_acceleration_with_gravity":
        return True
    return False


def _validate_required_preprocessing_metadata(
    summary: PreprocessedIMUSummary,
) -> None:
    """Require the explicit metadata needed to hand off a new artifact."""

    missing = [
        field
        for field in REQUIRED_PREPROCESSING_METADATA_FIELDS
        if field not in summary.payload or summary.payload[field] is None
    ]
    if missing:
        raise PreprocessingIOError(
            "preprocessing summary is missing required metadata: " + ", ".join(missing)
        )
    if summary.sample_count is None:
        raise PreprocessingIOError("preprocessing summary sample_count must be a positive integer")
    if summary.channel_names is None:
        raise PreprocessingIOError("preprocessing summary channel_names is required")
    if summary.sampling_rate_hz is None:
        raise PreprocessingIOError("preprocessing summary sampling_rate_hz is required")
    if summary.gravity_removal_method is None:
        raise PreprocessingIOError("preprocessing summary gravity_removal_method is required")
    if summary.acceleration_semantics is None:
        raise PreprocessingIOError("preprocessing summary acceleration_semantics is required")
    if not isinstance(summary.payload["gravity_removed"], bool):
        raise PreprocessingIOError("preprocessing summary gravity_removed must be a boolean")
    if summary.gravity_removed is None:
        raise PreprocessingIOError("preprocessing summary gravity_removed is required")
    if summary.units != PREPROCESSED_IMU_UNITS:
        raise PreprocessingIOError(
            "preprocessing summary units must exactly match the nine-channel canonical units"
        )


def _parse_units(value: object, *, required: bool) -> tuple[str, ...] | None:
    if value is None:
        if required:
            raise PreprocessingIOError(
                "preprocessing summary units must be a nine-element string list"
            )
        return None
    if not required:
        return None
    if not isinstance(value, list) or not all(isinstance(unit, str) for unit in value):
        raise PreprocessingIOError(
            "preprocessing summary units must be a nine-element string list"
        )
    units = tuple(value)
    if units != PREPROCESSED_IMU_UNITS:
        raise PreprocessingIOError(
            "preprocessing summary units must exactly match the nine-channel canonical units"
        )
    return units


def _summary_declares_preprocessed_artifact(payload: Mapping[str, object]) -> bool:
    return (
        payload.get("artifact_type") == PREPROCESSED_IMU_ARTIFACT_TYPE
        or payload.get("schema_version") == PREPROCESSING_SCHEMA_VERSION
    )


def _is_preprocessed_imu_artifact_path(path: Path) -> bool:
    return path.name.endswith("_preprocessedIMU.npy")


def _is_preprocessing_summary_path(path: Path) -> bool:
    return path.name.endswith("_preprocessing.json")


def _validate_method_semantics(method: object, semantics: object) -> None:
    if method is None and semantics is None:
        return
    if method is None or semantics is None:
        raise PreprocessingIOError(
            "preprocessing summary gravity_removal_method and acceleration_semantics "
            "must be provided together"
        )
    if method != "raw" and semantics == "gravity_removed_body_frame":
        return
    expected = _METHOD_SEMANTICS[str(method)]
    if semantics != expected:
        raise PreprocessingIOError(
            "preprocessing summary acceleration_semantics does not match gravity_removal_method "
            f"{method!r}; expected {expected!r}"
        )


def _default_summary_path(path: Path) -> Path | None:
    name = path.name
    if name.endswith("_preprocessedIMU.npy"):
        return path.with_name(name.removesuffix("_preprocessedIMU.npy") + "_preprocessing.json")
    return None


def _first_value(
    first: Mapping[str, object],
    second: Mapping[str, object],
    *keys: str,
) -> object | None:
    for key in keys:
        if key in first:
            return first[key]
        if key in second:
            return second[key]
    return None


def _resolve_numeric_summary_field(
    root: Mapping[str, object],
    nested: Mapping[str, object],
    key: str,
    *,
    name: str,
) -> float | None:
    root_value = _optional_finite_positive(root.get(key), name=name) if key in root else None
    nested_value = _optional_finite_positive(nested.get(key), name=name) if key in nested else None
    if root_value is not None and nested_value is not None and not math.isclose(
        root_value,
        nested_value,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PreprocessingIOError(f"preprocessing summary {key} conflicts between nested and root fields")
    return root_value if root_value is not None else nested_value


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PreprocessingIOError(f"{name} must be a finite positive number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise PreprocessingIOError(f"{name} must be a finite positive number") from error
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise PreprocessingIOError(f"{name} must be a finite positive number")
    return numeric


def _optional_finite_positive(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_positive(value, name=name)


def _optional_positive_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreprocessingIOError(f"{name} must be a positive integer")
    return value
