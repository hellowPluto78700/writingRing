"""Build text and JSON-safe summaries from loaded WritingRing results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from writingring.board_loader import BoardData, FiniteRange
from writingring.discovery import Recording
from writingring.ring_loader import RingData


def build_recording_summary(
    recording: Recording,
    ring_data: RingData,
    board_data: BoardData,
) -> dict[str, Any]:
    """Return the common inspection information used by both CLI tools."""

    ring = ring_data.validation
    board = board_data.validation
    chunk_range = (
        [board.chunk_indices[0], board.chunk_indices[-1]]
        if board.chunk_indices
        else None
    )
    return {
        "recording": {
            "user": recording.user,
            "action": recording.action,
            "dataset_id": recording.dataset_id,
            "primary_ring_path": recording.ring_0_path,
            "ring_1_metadata_path": recording.ring_1_path,
            "timestamp_marker_path": recording.timestamp_path,
            "board_chunk_count": len(recording.board_chunk_paths),
            "board_chunk_paths": recording.board_chunk_paths,
            "board_chunk_indices": recording.board_chunk_indices,
            "discovery_warnings": recording.warnings,
        },
        "ring": {
            "source_path": ring_data.source_path,
            "raw_dataframe_shape": ring_data.raw_shape,
            "sample_count": ring.sample_count,
            "raw_timestamp_start": ring.raw_timestamp_start,
            "raw_timestamp_end": ring.raw_timestamp_end,
            "inferred_duration_s": ring.inferred_duration_s,
            "inferred_sampling_rate_hz": ring.inferred_sampling_rate_hz,
            "duplicate_timestamp_steps": ring.duplicate_timestamp_steps,
            "backward_timestamp_steps": ring.backward_timestamp_steps,
            "nan_counts_by_column": ring.nan_counts_by_column,
            "positive_infinity_counts_by_column": (
                ring.positive_infinity_counts_by_column
            ),
            "negative_infinity_counts_by_column": (
                ring.negative_infinity_counts_by_column
            ),
            "inferred_timestamp_unit": ring.inferred_timestamp_unit,
            "warnings": ring_data.warnings,
        },
        "board": {
            "source_paths": board_data.chunk_paths,
            "chunk_count": board.chunk_count,
            "chunk_indices": board.chunk_indices,
            "chunk_index_range": chunk_range,
            "missing_chunk_indices": board.missing_chunk_indices,
            "empty_chunk_indices": board.empty_chunk_indices,
            "frame_count": board.frame_count,
            "contact_count": board.contact_count,
            "contact_free_frame_count": board.frames_without_contacts,
            "first_timestamp_in_numeric_file_order": (
                board.first_frame_timestamp_in_numeric_order
            ),
            "last_timestamp_in_numeric_file_order": (
                board.last_frame_timestamp_in_numeric_order
            ),
            "within_chunk_backward_steps": board.within_chunk_backward_steps,
            "cross_chunk_backward_boundaries": (
                board.cross_chunk_backward_boundaries
            ),
            "x_range": _range_summary(board.x_range),
            "raw_y_range": _range_summary(board.y_raw_range),
            "force_range": _range_summary(board.force_range),
            "warnings": board_data.warnings,
        },
        "notes": {
            "physical_units": (
                "Signal, coordinate, and force physical units remain "
                "undocumented; values are reported as raw/stored values."
            ),
            "ring_timestamp_interpretation": (
                "Ring relative seconds use an inferred microsecond "
                "interpretation that is not a confirmed upstream contract."
            ),
            "synchronization": (
                "Ring and Board clocks are not assumed synchronized; no "
                "synchronization was performed."
            ),
        },
    }


def format_recording_summary(summary: Mapping[str, Any]) -> str:
    """Format a common summary as readable inspection text."""

    recording = summary["recording"]
    ring = summary["ring"]
    board = summary["board"]
    boundaries = board["cross_chunk_backward_boundaries"]
    boundary_text = (
        "; ".join(
            f"{boundary.previous_chunk_index}->{boundary.next_chunk_index} "
            f"(delta={boundary.timestamp_delta})"
            for boundary in boundaries
        )
        or "none"
    )
    chunk_range = board["chunk_index_range"]
    range_text = (
        f"{chunk_range[0]}-{chunk_range[1]}"
        if chunk_range is not None
        else "none"
    )

    lines = [
        "Recording",
        f"  user: {recording['user']}",
        f"  action: {recording['action']}",
        f"  dataset ID: {recording['dataset_id']}",
        f"  primary Ring path: {recording['primary_ring_path']}",
        f"  Ring 1 metadata path: {_optional(recording['ring_1_metadata_path'])}",
        (
            "  timestamp marker path: "
            f"{_optional(recording['timestamp_marker_path'])}"
        ),
        f"  Board chunk count: {recording['board_chunk_count']}",
        "  Board chunk paths:",
        *(
            f"    - {path}"
            for path in recording["board_chunk_paths"]
        ),
        "",
        "Ring",
        f"  raw DataFrame shape: {tuple(ring['raw_dataframe_shape'])}",
        f"  sample count: {ring['sample_count']}",
        f"  raw timestamp start: {ring['raw_timestamp_start']}",
        f"  raw timestamp end: {ring['raw_timestamp_end']}",
        f"  inferred duration (s): {_optional(ring['inferred_duration_s'])}",
        (
            "  inferred sampling rate (Hz): "
            f"{_optional(ring['inferred_sampling_rate_hz'])}"
        ),
        f"  duplicate timestamp steps: {ring['duplicate_timestamp_steps']}",
        f"  backward timestamp steps: {ring['backward_timestamp_steps']}",
        f"  NaN counts: {_format_counts(ring['nan_counts_by_column'])}",
        (
            "  positive infinity counts: "
            f"{_format_counts(ring['positive_infinity_counts_by_column'])}"
        ),
        (
            "  negative infinity counts: "
            f"{_format_counts(ring['negative_infinity_counts_by_column'])}"
        ),
        f"  warnings: {_format_warnings(ring['warnings'])}",
        "",
        "Board",
        f"  chunk count: {board['chunk_count']}",
        f"  chunk-index range: {range_text}",
        (
            "  missing chunk indices: "
            f"{_format_indices(board['missing_chunk_indices'])}"
        ),
        (
            "  empty chunk indices: "
            f"{_format_indices(board['empty_chunk_indices'])}"
        ),
        f"  frame count: {board['frame_count']}",
        f"  contact count: {board['contact_count']}",
        f"  contact-free frame count: {board['contact_free_frame_count']}",
        (
            "  first timestamp in numeric file order: "
            f"{_optional(board['first_timestamp_in_numeric_file_order'])}"
        ),
        (
            "  last timestamp in numeric file order: "
            f"{_optional(board['last_timestamp_in_numeric_file_order'])}"
        ),
        (
            "  within-chunk backward steps: "
            f"{board['within_chunk_backward_steps']}"
        ),
        f"  cross-chunk backward boundaries: {boundary_text}",
        f"  x range: {_format_range(board['x_range'])}",
        f"  raw-y range: {_format_range(board['raw_y_range'])}",
        f"  force range: {_format_range(board['force_range'])}",
        f"  warnings: {_format_warnings(board['warnings'])}",
    ]
    return "\n".join(lines)


def to_jsonable(value: Any) -> Any:
    """Recursively convert project values into strict JSON-compatible values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence):
        return [to_jsonable(item) for item in value]
    raise TypeError(
        f"value of type {type(value).__module__}.{type(value).__qualname__} "
        "is not JSON serializable"
    )


def write_json(path: str | Path, value: Any) -> Path:
    """Write strict, indented JSON and create missing parent directories."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        to_jsonable(value),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    output_path.write_text(serialized + "\n", encoding="utf-8")
    return output_path


def _range_summary(value_range: FiniteRange) -> dict[str, int | float | None]:
    return {
        "minimum": value_range.minimum,
        "maximum": value_range.maximum,
        "finite_count": value_range.finite_count,
        "nonfinite_count": value_range.nonfinite_count,
    }


def _optional(value: object | None) -> str:
    return "none" if value is None else str(value)


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _format_indices(indices: Sequence[int]) -> str:
    return ", ".join(str(index) for index in indices) or "none"


def _format_warnings(warnings: Sequence[str]) -> str:
    return "; ".join(warnings) or "none"


def _format_range(summary: Mapping[str, int | float | None]) -> str:
    return (
        f"{_optional(summary['minimum'])} to "
        f"{_optional(summary['maximum'])} "
        f"(finite={summary['finite_count']}, "
        f"nonfinite={summary['nonfinite_count']})"
    )
