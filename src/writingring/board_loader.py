"""Load and validate numerically ordered WritingRing board chunks.

The public :func:`load_board` API accepts a discovered
:class:`~writingring.discovery.Recording`, a single board path, or an explicit
sequence of board paths. Paths are consumed exactly as supplied and must
already be in nondecreasing numeric chunk-index order. Timestamps are never
used to reorder, split, repair, or discard chunks.

Before deserialization, the loader imports the official pickle classes from
``core.sensel_lib.frame_data``. Each payload is then loaded with the upstream
author's ``compress_pickle.load(path)`` operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Final, Sequence

import compress_pickle
import numpy as np
import pandas as pd

from writingring.discovery import (
    FileKind,
    Recording,
    parse_recording_filename,
)


FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "dataset_id",
    "chunk_index",
    "frame_index_in_chunk",
    "global_frame_index",
    "frame_timestamp_raw",
    "contact_count",
    "force_array_shape",
    "force_array_dtype",
)
CONTACT_COLUMNS: Final[tuple[str, ...]] = (
    "dataset_id",
    "chunk_index",
    "frame_index_in_chunk",
    "global_frame_index",
    "frame_timestamp_raw",
    "contact_index",
    "contact_id",
    "state",
    "x",
    "y_raw",
    "y_display",
    "force",
    "area",
    "major",
    "minor",
    "delta_x",
    "delta_y",
    "delta_force",
    "delta_area",
    "label",
    "contact_frame_id",
)
EXPECTED_FRAME_ATTRIBUTES: Final[tuple[str, ...]] = (
    "force_array",
    "timestamp",
    "contacts",
)
EXPECTED_CONTACT_ATTRIBUTES: Final[tuple[str, ...]] = (
    "id",
    "state",
    "x",
    "y",
    "area",
    "force",
    "major",
    "minor",
    "delta_x",
    "delta_y",
    "delta_force",
    "delta_area",
    "label",
    "frame_id",
)
BACKWARD_POSITION_SAMPLE_LIMIT: Final[int] = 20


class BoardLoadError(Exception):
    """Base exception for board inputs that cannot be loaded safely."""


class BoardClassImportError(BoardLoadError):
    """Raised when the official pickle classes are unavailable."""


class BoardPathError(BoardLoadError):
    """Raised for invalid paths, membership, or numeric path ordering."""


class MissingBoardChunksError(BoardPathError):
    """Raised when no board chunk paths were supplied."""


class BoardChunkError(BoardLoadError):
    """Base error for a chunk that could not produce safe frame data."""

    def __init__(self, message: str, chunk_report: BoardChunkReport) -> None:
        super().__init__(message)
        self.chunk_report = chunk_report


class EmptyBoardChunkFileError(BoardChunkError):
    """Raised for a zero-byte chunk file, distinct from a serialized ``[]``."""


class BoardDeserializationError(BoardChunkError):
    """Raised when ``compress_pickle.load`` fails."""


class UnexpectedBoardContainerError(BoardChunkError):
    """Raised when a decoded top-level value is not a list."""


class UnexpectedBoardObjectError(BoardChunkError):
    """Raised when a frame, contacts container, or contact has the wrong type."""


class MissingBoardAttributeError(BoardChunkError):
    """Raised when a frame or contact lacks a required confirmed attribute."""


@dataclass(frozen=True, slots=True)
class FiniteRange:
    """Finite extrema plus the number of finite and non-finite values."""

    minimum: float | None
    maximum: float | None
    finite_count: int
    nonfinite_count: int


@dataclass(frozen=True, slots=True)
class BoardChunkReport:
    """Deserialization and validation status for one chunk path."""

    dataset_id: int
    chunk_index: int
    source_path: Path
    file_size_bytes: int | None
    deserialized: bool
    frame_count: int | None
    contact_count: int | None
    frames_without_contacts: int | None
    first_frame_timestamp: float | None
    last_frame_timestamp: float | None
    empty: bool | None
    timestamps_finite: bool | None
    timestamps_nondecreasing: bool | None
    timestamps_strictly_increasing: bool | None
    duplicate_timestamp_steps: int | None
    backward_timestamp_steps: int | None
    backward_timestamp_step_positions: tuple[int, ...]
    backward_positions_truncated: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChunkBoundaryReport:
    """Timestamp relationship between adjacent supplied chunk paths."""

    previous_chunk_index: int
    next_chunk_index: int
    consecutive_numeric_indices: bool
    previous_last_timestamp: float | None
    next_first_timestamp: float | None
    timestamp_delta: float | None
    backward: bool | None
    warning: str | None


@dataclass(frozen=True, slots=True)
class BoardValidationReport:
    """Recording-level validation aggregated without changing chunk order."""

    dataset_id: int
    chunk_count: int
    chunk_indices: tuple[int, ...]
    missing_chunk_indices: tuple[int, ...]
    duplicate_chunk_indices: tuple[int, ...]
    empty_chunk_indices: tuple[int, ...]
    empty_leading_chunk_indices: tuple[int, ...]
    frame_count: int
    contact_count: int
    frames_without_contacts: int
    first_frame_timestamp_in_numeric_order: float | None
    last_frame_timestamp_in_numeric_order: float | None
    within_chunk_backward_steps: int
    within_chunk_backward_locations: tuple[tuple[int, int], ...]
    chunk_boundaries: tuple[ChunkBoundaryReport, ...]
    cross_chunk_backward_boundaries: tuple[ChunkBoundaryReport, ...]
    x_range: FiniteRange
    y_raw_range: FiniteRange
    force_range: FiniteRange
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BoardData:
    """Loaded frame/contact tables and complete chunk validation metadata."""

    source_recording: Recording | None
    chunk_paths: tuple[Path, ...]
    chunk_reports: tuple[BoardChunkReport, ...]
    frames: pd.DataFrame
    contacts: pd.DataFrame
    validation: BoardValidationReport
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _OfficialTypes:
    frame: type
    contact: type


def load_board(
    source: Recording | str | Path | Sequence[str | Path],
) -> BoardData:
    """Load board chunks in their already supplied numeric filename order.

    A ``Recording`` contributes ``recording.board_chunk_paths`` exactly as
    discovery stored them. Explicit sequences are also preserved exactly.
    Paths from different dataset IDs or paths in decreasing numeric order are
    rejected before deserialization.

    Serialized empty lists are retained as successful empty chunks. Files or
    objects that cannot be decoded safely raise structured exceptions; chunk
    failures are never skipped.
    """

    recording, paths, dataset_id, chunk_indices = _resolve_source(source)
    official_types = _load_official_types()

    frame_rows: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    chunk_reports: list[BoardChunkReport] = []
    x_values: list[float] = []
    y_values: list[float] = []
    force_values: list[float] = []
    global_frame_index = 0

    for path, chunk_index in zip(paths, chunk_indices, strict=True):
        report, new_frame_rows, new_contact_rows = _load_chunk(
            path,
            dataset_id=dataset_id,
            chunk_index=chunk_index,
            global_frame_index_start=global_frame_index,
            official_types=official_types,
        )
        chunk_reports.append(report)
        frame_rows.extend(new_frame_rows)
        contact_rows.extend(new_contact_rows)
        global_frame_index += report.frame_count or 0
        x_values.extend(float(row["x"]) for row in new_contact_rows)
        y_values.extend(float(row["y_raw"]) for row in new_contact_rows)
        force_values.extend(float(row["force"]) for row in new_contact_rows)

    frames = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
    contacts = pd.DataFrame(contact_rows, columns=CONTACT_COLUMNS)
    validation = _build_recording_report(
        dataset_id=dataset_id,
        chunk_indices=chunk_indices,
        chunk_reports=tuple(chunk_reports),
        x_values=x_values,
        y_values=y_values,
        force_values=force_values,
    )
    return BoardData(
        source_recording=recording,
        chunk_paths=paths,
        chunk_reports=tuple(chunk_reports),
        frames=frames,
        contacts=contacts,
        validation=validation,
        warnings=validation.warnings,
    )


def _resolve_source(
    source: Recording | str | Path | Sequence[str | Path],
) -> tuple[Recording | None, tuple[Path, ...], int, tuple[int, ...]]:
    if isinstance(source, Recording):
        recording: Recording | None = source
        paths = tuple(source.board_chunk_paths)
        expected_dataset_id: int | None = source.dataset_id
    elif isinstance(source, (str, Path)):
        recording = None
        paths = (Path(source),)
        expected_dataset_id = None
    else:
        recording = None
        paths = tuple(Path(path) for path in source)
        expected_dataset_id = None

    if not paths:
        raise MissingBoardChunksError("no board chunk paths were supplied")

    dataset_ids: list[int] = []
    chunk_indices: list[int] = []
    for path in paths:
        parsed = parse_recording_filename(path.name)
        if (
            parsed is None
            or parsed.kind is not FileKind.BOARD_CHUNK
            or parsed.chunk_index is None
        ):
            raise BoardPathError(f"not a recognized board chunk path: {path}")
        dataset_ids.append(parsed.dataset_id)
        chunk_indices.append(parsed.chunk_index)

    dataset_id = dataset_ids[0]
    if any(candidate != dataset_id for candidate in dataset_ids):
        raise BoardPathError(
            "board paths contain more than one dataset ID: "
            + ", ".join(str(value) for value in dataset_ids)
        )
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise BoardPathError(
            f"board dataset {dataset_id} does not match recording dataset "
            f"{expected_dataset_id}"
        )
    if any(
        next_index < current_index
        for current_index, next_index in zip(
            chunk_indices,
            chunk_indices[1:],
        )
    ):
        raise BoardPathError(
            "board paths are not in nondecreasing numeric chunk-index order: "
            + ", ".join(str(index) for index in chunk_indices)
        )

    for path in paths:
        if not path.exists():
            raise BoardPathError(f"board chunk does not exist: {path}")
        if not path.is_file():
            raise BoardPathError(
                f"board chunk path is not a regular file: {path}"
            )

    return recording, paths, dataset_id, tuple(chunk_indices)


def _load_official_types() -> _OfficialTypes:
    try:
        module: ModuleType = import_module("core.sensel_lib.frame_data")
        frame_type = getattr(module, "FrameData")
        contact_type = getattr(module, "ContactData")
    except (ImportError, AttributeError) as error:
        raise BoardClassImportError(
            "official pickle classes are not importable from "
            "core.sensel_lib.frame_data"
        ) from error
    if not isinstance(frame_type, type) or not isinstance(contact_type, type):
        raise BoardClassImportError(
            "core.sensel_lib.frame_data does not expose FrameData and "
            "ContactData classes"
        )
    return _OfficialTypes(frame=frame_type, contact=contact_type)


def _load_chunk(
    path: Path,
    *,
    dataset_id: int,
    chunk_index: int,
    global_frame_index_start: int,
    official_types: _OfficialTypes,
) -> tuple[
    BoardChunkReport,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    file_size = path.stat().st_size
    if file_size == 0:
        message = f"board chunk file is empty: {path}"
        report = _failed_chunk_report(
            dataset_id,
            chunk_index,
            path,
            file_size,
            message,
        )
        raise EmptyBoardChunkFileError(message, report)

    try:
        loaded = compress_pickle.load(path)
    except Exception as error:
        message = f"failed to deserialize board chunk {path}: {error}"
        report = _failed_chunk_report(
            dataset_id,
            chunk_index,
            path,
            file_size,
            message,
        )
        raise BoardDeserializationError(message, report) from error

    if not isinstance(loaded, list):
        message = (
            f"board chunk top-level object must be list, got "
            f"{type(loaded).__module__}.{type(loaded).__qualname__}: {path}"
        )
        report = _failed_chunk_report(
            dataset_id,
            chunk_index,
            path,
            file_size,
            message,
            deserialized=True,
        )
        raise UnexpectedBoardContainerError(message, report)

    frame_rows: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    timestamps: list[float] = []
    frames_without_contacts = 0
    for frame_index, frame in enumerate(loaded):
        context = f"{path}, frame {frame_index}"
        if not isinstance(frame, official_types.frame):
            message = (
                f"unexpected frame type at {context}: "
                f"{type(frame).__module__}.{type(frame).__qualname__}"
            )
            report = _failed_chunk_report(
                dataset_id,
                chunk_index,
                path,
                file_size,
                message,
                deserialized=True,
            )
            raise UnexpectedBoardObjectError(message, report)
        _require_attributes(
            frame,
            EXPECTED_FRAME_ATTRIBUTES,
            context=context,
            dataset_id=dataset_id,
            chunk_index=chunk_index,
            path=path,
            file_size=file_size,
        )
        if not isinstance(frame.contacts, list):
            message = (
                f"contacts container must be list at {context}, got "
                f"{type(frame.contacts).__module__}."
                f"{type(frame.contacts).__qualname__}"
            )
            report = _failed_chunk_report(
                dataset_id,
                chunk_index,
                path,
                file_size,
                message,
                deserialized=True,
            )
            raise UnexpectedBoardObjectError(message, report)

        timestamp_numeric = _numeric_value(
            frame.timestamp,
            name="frame timestamp",
            context=context,
            dataset_id=dataset_id,
            chunk_index=chunk_index,
            path=path,
            file_size=file_size,
        )
        timestamps.append(timestamp_numeric)
        global_frame_index = global_frame_index_start + frame_index
        frame_rows.append(
            {
                "dataset_id": dataset_id,
                "chunk_index": chunk_index,
                "frame_index_in_chunk": frame_index,
                "global_frame_index": global_frame_index,
                "frame_timestamp_raw": frame.timestamp,
                "contact_count": len(frame.contacts),
                "force_array_shape": getattr(frame.force_array, "shape", None),
                "force_array_dtype": str(
                    getattr(frame.force_array, "dtype", type(frame.force_array))
                ),
            }
        )
        if not frame.contacts:
            frames_without_contacts += 1

        for contact_index, contact in enumerate(frame.contacts):
            contact_context = (
                f"{path}, frame {frame_index}, contact {contact_index}"
            )
            if not isinstance(contact, official_types.contact):
                message = (
                    f"unexpected contact type at {contact_context}: "
                    f"{type(contact).__module__}.{type(contact).__qualname__}"
                )
                report = _failed_chunk_report(
                    dataset_id,
                    chunk_index,
                    path,
                    file_size,
                    message,
                    deserialized=True,
                )
                raise UnexpectedBoardObjectError(message, report)
            _require_attributes(
                contact,
                EXPECTED_CONTACT_ATTRIBUTES,
                context=contact_context,
                dataset_id=dataset_id,
                chunk_index=chunk_index,
                path=path,
                file_size=file_size,
            )
            x = _numeric_value(
                contact.x,
                name="contact x",
                context=contact_context,
                dataset_id=dataset_id,
                chunk_index=chunk_index,
                path=path,
                file_size=file_size,
            )
            y_raw = _numeric_value(
                contact.y,
                name="contact y",
                context=contact_context,
                dataset_id=dataset_id,
                chunk_index=chunk_index,
                path=path,
                file_size=file_size,
            )
            force = _numeric_value(
                contact.force,
                name="contact force",
                context=contact_context,
                dataset_id=dataset_id,
                chunk_index=chunk_index,
                path=path,
                file_size=file_size,
            )
            contact_rows.append(
                {
                    "dataset_id": dataset_id,
                    "chunk_index": chunk_index,
                    "frame_index_in_chunk": frame_index,
                    "global_frame_index": global_frame_index,
                    "frame_timestamp_raw": frame.timestamp,
                    "contact_index": contact_index,
                    "contact_id": contact.id,
                    "state": contact.state,
                    "x": x,
                    "y_raw": y_raw,
                    "y_display": 1.0 - y_raw,
                    "force": force,
                    "area": contact.area,
                    "major": contact.major,
                    "minor": contact.minor,
                    "delta_x": contact.delta_x,
                    "delta_y": contact.delta_y,
                    "delta_force": contact.delta_force,
                    "delta_area": contact.delta_area,
                    "label": contact.label,
                    "contact_frame_id": contact.frame_id,
                }
            )

    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    if timestamp_array.size:
        differences = np.diff(timestamp_array)
        timestamps_finite = bool(np.isfinite(timestamp_array).all())
        duplicate_steps = int(np.count_nonzero(differences == 0))
        backward_array = np.flatnonzero(differences < 0) + 1
        backward_steps = int(backward_array.size)
        backward_positions = tuple(
            int(value)
            for value in backward_array[:BACKWARD_POSITION_SAMPLE_LIMIT]
        )
        timestamps_nondecreasing = timestamps_finite and bool(
            np.all(differences >= 0)
        )
        timestamps_strictly_increasing = timestamps_finite and bool(
            np.all(differences > 0)
        )
        first_timestamp = float(timestamp_array[0])
        last_timestamp = float(timestamp_array[-1])
    else:
        timestamps_finite = None
        duplicate_steps = None
        backward_steps = None
        backward_positions = ()
        timestamps_nondecreasing = None
        timestamps_strictly_increasing = None
        first_timestamp = None
        last_timestamp = None

    warnings: list[str] = []
    if not loaded:
        warnings.append("chunk is a successfully deserialized empty list")
    if frames_without_contacts:
        warnings.append(
            f"{frames_without_contacts} frame(s) contain no contacts"
        )
    if timestamps_finite is False:
        warnings.append("chunk contains non-finite frame timestamps")
    if duplicate_steps:
        warnings.append(
            f"{duplicate_steps} duplicate frame timestamp step(s)"
        )
    if backward_steps:
        warnings.append(
            f"{backward_steps} backward frame timestamp step(s)"
        )
    coordinate_warnings = _nonfinite_contact_warnings(contact_rows)
    warnings.extend(coordinate_warnings)

    report = BoardChunkReport(
        dataset_id=dataset_id,
        chunk_index=chunk_index,
        source_path=path,
        file_size_bytes=file_size,
        deserialized=True,
        frame_count=len(loaded),
        contact_count=len(contact_rows),
        frames_without_contacts=frames_without_contacts,
        first_frame_timestamp=first_timestamp,
        last_frame_timestamp=last_timestamp,
        empty=not loaded,
        timestamps_finite=timestamps_finite,
        timestamps_nondecreasing=timestamps_nondecreasing,
        timestamps_strictly_increasing=timestamps_strictly_increasing,
        duplicate_timestamp_steps=duplicate_steps,
        backward_timestamp_steps=backward_steps,
        backward_timestamp_step_positions=backward_positions,
        backward_positions_truncated=(
            (backward_steps or 0) > BACKWARD_POSITION_SAMPLE_LIMIT
        ),
        warnings=tuple(warnings),
        errors=(),
    )
    return report, frame_rows, contact_rows


def _require_attributes(
    value: object,
    names: tuple[str, ...],
    *,
    context: str,
    dataset_id: int,
    chunk_index: int,
    path: Path,
    file_size: int,
) -> None:
    missing = tuple(name for name in names if not hasattr(value, name))
    if not missing:
        return
    message = f"missing expected attribute(s) {', '.join(missing)} at {context}"
    report = _failed_chunk_report(
        dataset_id,
        chunk_index,
        path,
        file_size,
        message,
        deserialized=True,
    )
    raise MissingBoardAttributeError(message, report)


def _numeric_value(
    value: object,
    *,
    name: str,
    context: str,
    dataset_id: int,
    chunk_index: int,
    path: Path,
    file_size: int,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as error:
        message = f"{name} is not numeric at {context}: {value!r}"
        report = _failed_chunk_report(
            dataset_id,
            chunk_index,
            path,
            file_size,
            message,
            deserialized=True,
        )
        raise UnexpectedBoardObjectError(message, report) from error


def _failed_chunk_report(
    dataset_id: int,
    chunk_index: int,
    path: Path,
    file_size: int | None,
    error: str,
    *,
    deserialized: bool = False,
) -> BoardChunkReport:
    return BoardChunkReport(
        dataset_id=dataset_id,
        chunk_index=chunk_index,
        source_path=path,
        file_size_bytes=file_size,
        deserialized=deserialized,
        frame_count=None,
        contact_count=None,
        frames_without_contacts=None,
        first_frame_timestamp=None,
        last_frame_timestamp=None,
        empty=None,
        timestamps_finite=None,
        timestamps_nondecreasing=None,
        timestamps_strictly_increasing=None,
        duplicate_timestamp_steps=None,
        backward_timestamp_steps=None,
        backward_timestamp_step_positions=(),
        backward_positions_truncated=False,
        warnings=(),
        errors=(error,),
    )


def _build_recording_report(
    *,
    dataset_id: int,
    chunk_indices: tuple[int, ...],
    chunk_reports: tuple[BoardChunkReport, ...],
    x_values: list[float],
    y_values: list[float],
    force_values: list[float],
) -> BoardValidationReport:
    unique_indices = set(chunk_indices)
    missing_indices = tuple(
        index
        for index in range(0, max(unique_indices) + 1)
        if index not in unique_indices
    )
    duplicate_indices = tuple(
        sorted(
            index
            for index in unique_indices
            if chunk_indices.count(index) > 1
        )
    )
    empty_indices = tuple(
        report.chunk_index for report in chunk_reports if report.empty
    )
    empty_leading: list[int] = []
    for report in chunk_reports:
        if report.empty:
            empty_leading.append(report.chunk_index)
        else:
            break

    boundaries = tuple(
        _build_boundary(previous, next_report)
        for previous, next_report in zip(
            chunk_reports,
            chunk_reports[1:],
        )
    )
    backward_boundaries = tuple(
        boundary for boundary in boundaries if boundary.backward
    )
    within_locations = tuple(
        (report.chunk_index, position)
        for report in chunk_reports
        for position in report.backward_timestamp_step_positions
    )
    total_frames = sum(report.frame_count or 0 for report in chunk_reports)
    total_contacts = sum(report.contact_count or 0 for report in chunk_reports)
    frames_without_contacts = sum(
        report.frames_without_contacts or 0 for report in chunk_reports
    )
    nonempty_reports = tuple(
        report for report in chunk_reports if not report.empty
    )
    first_timestamp = (
        nonempty_reports[0].first_frame_timestamp
        if nonempty_reports
        else None
    )
    last_timestamp = (
        nonempty_reports[-1].last_frame_timestamp
        if nonempty_reports
        else None
    )
    x_range = _finite_range(x_values)
    y_range = _finite_range(y_values)
    force_range = _finite_range(force_values)

    warnings: list[str] = []
    if missing_indices:
        warnings.append(
            "missing board chunk indices: "
            + ", ".join(str(index) for index in missing_indices)
        )
    if duplicate_indices:
        warnings.append(
            "duplicate board chunk indices retained: "
            + ", ".join(str(index) for index in duplicate_indices)
        )
    if empty_indices:
        warnings.append(
            "empty board chunks retained: "
            + ", ".join(str(index) for index in empty_indices)
        )
    if empty_leading:
        warnings.append(
            "empty leading board chunks retained: "
            + ", ".join(str(index) for index in empty_leading)
        )
    if frames_without_contacts:
        warnings.append(
            f"{frames_without_contacts} frame(s) contain no contacts"
        )
    within_backward_steps = sum(
        report.backward_timestamp_steps or 0 for report in chunk_reports
    )
    if within_backward_steps:
        warnings.append(
            f"{within_backward_steps} within-chunk backward timestamp step(s)"
        )
    for boundary in backward_boundaries:
        warnings.append(
            f"backward timestamp boundary {boundary.previous_chunk_index}->"
            f"{boundary.next_chunk_index}: delta={boundary.timestamp_delta}"
        )
    for name, value_range in (
        ("x", x_range),
        ("y_raw", y_range),
        ("force", force_range),
    ):
        if value_range.nonfinite_count:
            warnings.append(
                f"{value_range.nonfinite_count} non-finite {name} value(s)"
            )

    return BoardValidationReport(
        dataset_id=dataset_id,
        chunk_count=len(chunk_reports),
        chunk_indices=chunk_indices,
        missing_chunk_indices=missing_indices,
        duplicate_chunk_indices=duplicate_indices,
        empty_chunk_indices=empty_indices,
        empty_leading_chunk_indices=tuple(empty_leading),
        frame_count=total_frames,
        contact_count=total_contacts,
        frames_without_contacts=frames_without_contacts,
        first_frame_timestamp_in_numeric_order=first_timestamp,
        last_frame_timestamp_in_numeric_order=last_timestamp,
        within_chunk_backward_steps=within_backward_steps,
        within_chunk_backward_locations=within_locations,
        chunk_boundaries=boundaries,
        cross_chunk_backward_boundaries=backward_boundaries,
        x_range=x_range,
        y_raw_range=y_range,
        force_range=force_range,
        warnings=tuple(warnings),
    )


def _build_boundary(
    previous: BoardChunkReport,
    next_report: BoardChunkReport,
) -> ChunkBoundaryReport:
    consecutive = next_report.chunk_index == previous.chunk_index + 1
    previous_timestamp = previous.last_frame_timestamp
    next_timestamp = next_report.first_frame_timestamp
    delta: float | None = None
    backward: bool | None = None
    warning: str | None = None
    if previous_timestamp is None or next_timestamp is None:
        warning = "timestamp continuity cannot be assessed across an empty chunk"
    elif not (
        np.isfinite(previous_timestamp) and np.isfinite(next_timestamp)
    ):
        warning = "timestamp continuity cannot be assessed with non-finite values"
    else:
        delta = float(next_timestamp - previous_timestamp)
        backward = delta < 0
        if backward:
            warning = "next chunk begins before the previous chunk ends"
        elif not consecutive:
            warning = "chunk indices are not consecutive"

    return ChunkBoundaryReport(
        previous_chunk_index=previous.chunk_index,
        next_chunk_index=next_report.chunk_index,
        consecutive_numeric_indices=consecutive,
        previous_last_timestamp=previous_timestamp,
        next_first_timestamp=next_timestamp,
        timestamp_delta=delta,
        backward=backward,
        warning=warning,
    )


def _finite_range(values: list[float]) -> FiniteRange:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return FiniteRange(
        minimum=float(np.min(finite)) if finite.size else None,
        maximum=float(np.max(finite)) if finite.size else None,
        finite_count=int(finite.size),
        nonfinite_count=int(array.size - finite.size),
    )


def _nonfinite_contact_warnings(
    contact_rows: list[dict[str, object]],
) -> tuple[str, ...]:
    warnings: list[str] = []
    for name in ("x", "y_raw", "force"):
        count = sum(
            not np.isfinite(float(row[name])) for row in contact_rows
        )
        if count:
            warnings.append(f"{count} non-finite {name} value(s)")
    return tuple(warnings)
