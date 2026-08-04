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
