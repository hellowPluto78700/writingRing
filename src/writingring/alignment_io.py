"""Safe export and application of constant Ring--Board time offsets.

The only supported mapping is ``ring_timestamp_us = board_timestamp_us +
offset_us``.  Keeping this convention in one module prevents sign changes
between matching, saved offsets, and verification plots.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Final

import numpy as np

from writingring.event_alignment import SequenceAlignmentResult


ALIGNMENT_MODEL: Final[str] = "constant_offset"
TIMESTAMP_UNIT: Final[str] = "microseconds"
TIME_MAPPING: Final[str] = "ring_timestamp_us = board_timestamp_us + offset_us"


class AlignmentOffsetExportError(ValueError):
    """Raised when a constant offset cannot be safely exported or consumed."""


@dataclass(frozen=True, slots=True)
class AlignmentOffset:
    """A successful, recording-specific constant Board-to-Ring offset."""

    user: str
    action: str
    dataset_id: int
    ring_stream: str
    offset_us: float
    alignment_model: str
    alignment_success: bool
    event_coverage_ratio: float | None
    matched_event_count: int | None
    total_valid_event_count: int | None
    alignment_signal_source: str | None = None
    feature_schema: str | None = None
    feature_values_sha256: str | None = None
    feature_metadata_sha256: str | None = None
    timestamp_sha256: str | None = None
    transient_channel_indices: tuple[int, ...] | None = None
    spike_event_channels_used: bool | None = None

    @property
    def offset_ms(self) -> float:
        """Return the offset in milliseconds for display only."""

        return self.offset_us / 1_000.0


def extract_alignment_offset(
    result: SequenceAlignmentResult,
    *,
    user: str,
    action: str,
    dataset_id: int,
    ring_stream: str = "ring_0",
) -> AlignmentOffset:
    """Extract one exportable offset only from a successful alignment result."""

    if not isinstance(result, SequenceAlignmentResult):
        raise AlignmentOffsetExportError(
            "result must be a SequenceAlignmentResult"
        )
    _validate_identity(user=user, action=action, dataset_id=dataset_id)
    if ring_stream != "ring_0":
        raise AlignmentOffsetExportError("ring_stream must be 'ring_0'")
    if not result.success:
        raise AlignmentOffsetExportError(
            "alignment did not succeed; no offset file was written"
        )
    offset_us = _finite_float(result.best_offset_us, name="best_offset_us")
    report = result.report
    if not isinstance(report, dict):
        raise AlignmentOffsetExportError("alignment result report must be a dictionary")
    model = report.get("alignment_model", ALIGNMENT_MODEL)
    if model != ALIGNMENT_MODEL:
        raise AlignmentOffsetExportError(
            "alignment result must use the constant_offset model"
        )
    return AlignmentOffset(
        user=user,
        action=action,
        dataset_id=dataset_id,
        ring_stream=ring_stream,
        offset_us=offset_us,
        alignment_model=ALIGNMENT_MODEL,
        alignment_success=True,
        event_coverage_ratio=_optional_finite_float(
            report.get("event_coverage_ratio"),
            name="event_coverage_ratio",
        ),
        matched_event_count=_optional_nonnegative_int(
            report.get("matched_event_count"),
            name="matched_event_count",
        ),
        total_valid_event_count=_optional_nonnegative_int(
            report.get("total_valid_event_count"),
            name="total_valid_event_count",
        ),
        alignment_signal_source=_optional_nonempty_string(
            report.get("alignment_signal_source"),
            name="alignment_signal_source",
        ),
        feature_schema=_optional_nonempty_string(
            report.get("feature_schema"),
            name="feature_schema",
        ),
        feature_values_sha256=_optional_sha256(
            report.get("feature_values_sha256"),
            name="feature_values_sha256",
        ),
        feature_metadata_sha256=_optional_sha256(
            report.get("feature_metadata_sha256"),
            name="feature_metadata_sha256",
        ),
        timestamp_sha256=_optional_sha256(
            report.get("timestamp_sha256"),
            name="timestamp_sha256",
        ),
        transient_channel_indices=_optional_channel_indices(
            report.get("transient_channel_indices")
        ),
        spike_event_channels_used=_optional_bool(
            report.get("spike_event_channels_used"),
            name="spike_event_channels_used",
        ),
    )


def build_alignment_offset_path(
    output_root: Path,
    *,
    user: str,
    action: str,
    dataset_id: int,
) -> Path:
    """Return the unique TXT path for a recording without writing it."""

    _validate_identity(user=user, action=action, dataset_id=dataset_id)
    return Path(output_root) / user / f"action_{action}" / (
        f"{dataset_id}_ring_board_offset.txt"
    )


def write_alignment_offset_txt(
    offset: AlignmentOffset,
    *,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Atomically write an explicitly directed and unit-labelled TXT export."""

    _validate_offset(offset)
    path = Path(output_path)
    if path.exists():
        if not path.is_file():
            raise AlignmentOffsetExportError(
                f"offset output path is not a regular file: {path}"
            )
        if not overwrite:
            raise AlignmentOffsetExportError(
                f"offset output already exists; use overwrite=True: {path}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _format_offset(offset)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    except OSError as error:
        raise AlignmentOffsetExportError(
            f"could not write alignment offset TXT: {path}: {error}"
        ) from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
    return path


def read_alignment_offset_txt(
    path: Path,
    *,
    expected_user: str | None = None,
    expected_action: str | None = None,
    expected_dataset_id: int | None = None,
) -> AlignmentOffset:
    """Read and validate a previously exported successful constant offset."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise AlignmentOffsetExportError(
            f"could not read alignment offset TXT: {source}: {error}"
        ) from error
    fields: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        if "=" not in line:
            raise AlignmentOffsetExportError(
                f"invalid offset TXT line {line_number}: expected key=value"
            )
        key, value = line.split("=", maxsplit=1)
        if not key or key in fields:
            raise AlignmentOffsetExportError(
                f"invalid or duplicate offset TXT key on line {line_number}"
            )
        fields[key] = value
    required = {
        "user",
        "action",
        "dataset_id",
        "ring_stream",
        "alignment_model",
        "timestamp_unit",
        "time_mapping",
        "offset_us",
        "alignment_success",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise AlignmentOffsetExportError(
            "offset TXT is missing required field(s): " + ", ".join(missing)
        )
    if fields["timestamp_unit"] != TIMESTAMP_UNIT:
        raise AlignmentOffsetExportError("offset TXT timestamp_unit must be microseconds")
    if fields["time_mapping"] != TIME_MAPPING:
        raise AlignmentOffsetExportError("offset TXT has an unsupported time_mapping")
    if fields["alignment_model"] != ALIGNMENT_MODEL:
        raise AlignmentOffsetExportError(
            "offset TXT alignment_model must be constant_offset"
        )
    if fields["alignment_success"] != "true":
        raise AlignmentOffsetExportError("offset TXT alignment_success must be true")
    offset = AlignmentOffset(
        user=fields["user"],
        action=fields["action"],
        dataset_id=_nonnegative_int(fields["dataset_id"], name="dataset_id"),
        ring_stream=fields["ring_stream"],
        offset_us=_finite_float(fields["offset_us"], name="offset_us"),
        alignment_model=fields["alignment_model"],
        alignment_success=True,
        event_coverage_ratio=_optional_finite_float(
            fields.get("event_coverage_ratio"), name="event_coverage_ratio"
        ),
        matched_event_count=_optional_nonnegative_int(
            fields.get("matched_event_count"), name="matched_event_count"
        ),
        total_valid_event_count=_optional_nonnegative_int(
            fields.get("total_valid_event_count"),
            name="total_valid_event_count",
        ),
        alignment_signal_source=_optional_nonempty_string(
            fields.get("alignment_signal_source"),
            name="alignment_signal_source",
        ),
        feature_schema=_optional_nonempty_string(
            fields.get("feature_schema"), name="feature_schema"
        ),
        feature_values_sha256=_optional_sha256(
            fields.get("feature_values_sha256"),
            name="feature_values_sha256",
        ),
        feature_metadata_sha256=_optional_sha256(
            fields.get("feature_metadata_sha256"),
            name="feature_metadata_sha256",
        ),
        timestamp_sha256=_optional_sha256(
            fields.get("timestamp_sha256"),
            name="timestamp_sha256",
        ),
        transient_channel_indices=_optional_channel_indices(
            fields.get("transient_channel_indices")
        ),
        spike_event_channels_used=_optional_bool(
            fields.get("spike_event_channels_used"),
            name="spike_event_channels_used",
        ),
    )
    _validate_offset(offset)
    expected = (expected_user, expected_action, expected_dataset_id)
    supplied = tuple(value is not None for value in expected)
    if any(supplied) and not all(supplied):
        raise AlignmentOffsetExportError(
            "expected_user, expected_action, and expected_dataset_id must be supplied together"
        )
    if all(supplied):
        _validate_identity(
            user=expected_user or "",
            action=expected_action or "",
            dataset_id=expected_dataset_id if expected_dataset_id is not None else -1,
        )
        if (offset.user, offset.action, offset.dataset_id) != expected:
            raise AlignmentOffsetExportError(
                "offset TXT recording identity does not match the requested recording"
            )
    return offset


def apply_board_to_ring_offset(
    board_timestamps_us: np.ndarray,
    *,
    offset_us: float,
) -> np.ndarray:
    """Return derived Ring-axis timestamps using ``ring = board + offset``."""

    try:
        timestamps = np.asarray(board_timestamps_us, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentOffsetExportError(
            "board_timestamps_us must be numeric"
        ) from error
    if not np.isfinite(timestamps).all():
        raise AlignmentOffsetExportError("board_timestamps_us must be finite")
    return timestamps.copy() + _finite_float(offset_us, name="offset_us")


def validate_alignment_feature_provenance(
    offset: AlignmentOffset,
    feature_input: object,
) -> None:
    """Require an alignment offset to describe the exact feature input.

    Legacy offsets intentionally remain readable for raw-ring workflows.  A
    SpikeIMU overlay, however, must prove that its offset was produced from
    the same recording, feature schema, values, metadata, timestamps, and
    transient-channel contract as the feature matrix being rendered.
    """

    from writingring.recording_features import RecordingFeatureInput

    if not isinstance(offset, AlignmentOffset):
        raise AlignmentOffsetExportError("offset must be an AlignmentOffset")
    if not isinstance(feature_input, RecordingFeatureInput):
        raise AlignmentOffsetExportError(
            "feature_input must be a RecordingFeatureInput"
        )
    _validate_offset(offset)
    identity = (offset.user, offset.action, offset.dataset_id)
    expected_identity = (
        feature_input.user,
        feature_input.action,
        feature_input.dataset_id,
    )
    if identity != expected_identity:
        raise AlignmentOffsetExportError(
            "alignment offset recording identity does not match feature input"
        )
    if offset.alignment_signal_source != feature_input.input_kind:
        raise AlignmentOffsetExportError(
            "alignment offset signal source does not match feature input"
        )
    if offset.feature_schema != feature_input.feature_schema:
        raise AlignmentOffsetExportError(
            "alignment offset feature schema does not match feature input"
        )
    if offset.feature_values_sha256 != feature_input.values_sha256:
        raise AlignmentOffsetExportError(
            "alignment offset feature values SHA-256 does not match feature input"
        )
    if offset.feature_metadata_sha256 != feature_input.metadata_sha256:
        raise AlignmentOffsetExportError(
            "alignment offset feature metadata SHA-256 does not match feature input"
        )
    if offset.timestamp_sha256 != feature_input.timestamps_sha256:
        raise AlignmentOffsetExportError(
            "alignment offset timestamp SHA-256 does not match feature input"
        )
    if offset.transient_channel_indices != feature_input.transient_channel_indices:
        raise AlignmentOffsetExportError(
            "alignment offset transient channel indices do not match feature input"
        )
    if feature_input.input_kind == "spike-imu" and offset.spike_event_channels_used is not False:
        raise AlignmentOffsetExportError(
            "SpikeIMU alignment offset must declare spike_event_channels_used=false"
        )


def _format_offset(offset: AlignmentOffset) -> str:
    coverage = (
        "" if offset.event_coverage_ratio is None else f"{offset.event_coverage_ratio:.6f}"
    )
    matched = "" if offset.matched_event_count is None else str(offset.matched_event_count)
    total = (
        ""
        if offset.total_valid_event_count is None
        else str(offset.total_valid_event_count)
    )
    provenance = ""
    if offset.alignment_signal_source is not None:
        assert offset.feature_schema is not None
        assert offset.feature_values_sha256 is not None
        assert offset.feature_metadata_sha256 is not None
        assert offset.timestamp_sha256 is not None
        assert offset.transient_channel_indices is not None
        assert offset.spike_event_channels_used is not None
        provenance = (
            f"alignment_signal_source={offset.alignment_signal_source}\n"
            f"feature_schema={offset.feature_schema}\n"
            f"feature_values_sha256={offset.feature_values_sha256}\n"
            f"feature_metadata_sha256={offset.feature_metadata_sha256}\n"
            f"timestamp_sha256={offset.timestamp_sha256}\n"
            "transient_channel_indices="
            f"{','.join(str(index) for index in offset.transient_channel_indices)}\n"
            "spike_event_channels_used="
            f"{'true' if offset.spike_event_channels_used else 'false'}\n"
        )
    return (
        f"user={offset.user}\n"
        f"action={offset.action}\n"
        f"dataset_id={offset.dataset_id}\n"
        f"ring_stream={offset.ring_stream}\n"
        f"alignment_model={ALIGNMENT_MODEL}\n"
        f"timestamp_unit={TIMESTAMP_UNIT}\n"
        f"time_mapping={TIME_MAPPING}\n"
        f"offset_us={offset.offset_us:.6f}\n"
        f"offset_ms={offset.offset_ms:.6f}\n"
        "alignment_success=true\n"
        f"event_coverage_ratio={coverage}\n"
        f"matched_event_count={matched}\n"
        f"total_valid_event_count={total}\n"
        f"{provenance}"
    )


def _validate_offset(offset: AlignmentOffset) -> None:
    if not isinstance(offset, AlignmentOffset):
        raise AlignmentOffsetExportError("offset must be an AlignmentOffset")
    _validate_identity(
        user=offset.user,
        action=offset.action,
        dataset_id=offset.dataset_id,
    )
    if offset.ring_stream != "ring_0":
        raise AlignmentOffsetExportError("ring_stream must be 'ring_0'")
    if offset.alignment_model != ALIGNMENT_MODEL:
        raise AlignmentOffsetExportError("alignment_model must be constant_offset")
    if not offset.alignment_success:
        raise AlignmentOffsetExportError(
            "alignment did not succeed; no offset file was written"
        )
    _finite_float(offset.offset_us, name="offset_us")
    coverage = _optional_finite_float(
        offset.event_coverage_ratio, name="event_coverage_ratio"
    )
    if coverage is not None and not 0.0 <= coverage <= 1.0:
        raise AlignmentOffsetExportError(
            "event_coverage_ratio must be between zero and one"
        )
    _optional_nonnegative_int(offset.matched_event_count, name="matched_event_count")
    _optional_nonnegative_int(
        offset.total_valid_event_count, name="total_valid_event_count"
    )
    provenance = (
        offset.alignment_signal_source,
        offset.feature_schema,
        offset.feature_values_sha256,
        offset.feature_metadata_sha256,
        offset.timestamp_sha256,
        offset.transient_channel_indices,
        offset.spike_event_channels_used,
    )
    if any(value is not None for value in provenance):
        if any(value is None for value in provenance):
            raise AlignmentOffsetExportError(
                "alignment feature provenance is incomplete"
            )
        _optional_nonempty_string(
            offset.alignment_signal_source,
            name="alignment_signal_source",
        )
        _optional_nonempty_string(offset.feature_schema, name="feature_schema")
        _sha256_digest(
            offset.feature_values_sha256,
            name="feature_values_sha256",
        )
        _sha256_digest(
            offset.feature_metadata_sha256,
            name="feature_metadata_sha256",
        )
        _sha256_digest(offset.timestamp_sha256, name="timestamp_sha256")
        _channel_indices(offset.transient_channel_indices)
        if not isinstance(offset.spike_event_channels_used, bool):
            raise AlignmentOffsetExportError(
                "spike_event_channels_used must be a boolean"
            )


def _validate_identity(*, user: str, action: str, dataset_id: int) -> None:
    for name, value in (("user", user), ("action", action)):
        if (
            not isinstance(value, str)
            or not value
            or value.strip() != value
            or "/" in value
            or "\\" in value
        ):
            raise AlignmentOffsetExportError(
                f"{name} must be a nonempty safe path component"
            )
    _nonnegative_int(dataset_id, name="dataset_id")


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise AlignmentOffsetExportError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AlignmentOffsetExportError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise AlignmentOffsetExportError(f"{name} must be finite")
    return number


def _optional_finite_float(value: object, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    return _finite_float(value, name=name)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise AlignmentOffsetExportError(f"{name} must be a nonnegative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise AlignmentOffsetExportError(
            f"{name} must be a nonnegative integer"
        ) from error
    if str(value).strip() not in {str(number), f"+{number}"} or number < 0:
        raise AlignmentOffsetExportError(f"{name} must be a nonnegative integer")
    return number


def _optional_nonnegative_int(value: object, *, name: str) -> int | None:
    if value is None or value == "":
        return None
    return _nonnegative_int(value, name=name)


def _optional_nonempty_string(value: object, *, name: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value or value.strip() != value:
        raise AlignmentOffsetExportError(f"{name} must be a nonempty string")
    return value


def _sha256_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AlignmentOffsetExportError(f"{name} must be a SHA-256 hex digest")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise AlignmentOffsetExportError(f"{name} must be a SHA-256 hex digest")
    return value.lower()


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None or value == "":
        return None
    return _sha256_digest(value, name=name)


def _channel_indices(value: object) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise AlignmentOffsetExportError(
            "transient_channel_indices must be a nonempty integer list"
        )
    if not value:
        raise AlignmentOffsetExportError(
            "transient_channel_indices must be a nonempty integer list"
        )
    indices = tuple(_nonnegative_int(item, name="transient_channel_indices") for item in value)
    if len(set(indices)) != len(indices):
        raise AlignmentOffsetExportError(
            "transient_channel_indices must not contain duplicates"
        )
    return indices


def _optional_channel_indices(value: object) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = value.split(",")
        if any(not part.strip() for part in parts):
            raise AlignmentOffsetExportError(
                "transient_channel_indices must be comma-separated integers"
            )
        value = tuple(part.strip() for part in parts)
    return _channel_indices(value)


def _optional_bool(value: object, *, name: str) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise AlignmentOffsetExportError(f"{name} must be a boolean")
