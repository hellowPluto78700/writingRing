"""Load and validate WritingRing ``ring_0`` binary files.

The public :func:`load_ring` API accepts either a discovered
:class:`~writingring.discovery.Recording` or an explicit ``ring_0`` path. A
``Recording`` always resolves to its ``ring_0_path``; ``ring_1`` paths are
rejected before any payload is read.

The upstream binary behavior is preserved exactly: native-endian float64
values are read with ``numpy.fromfile`` and reshaped to ``(-1, 7)``. Signal
units remain unspecified. A relative-time column is exposed only as an
explicitly inferred microsecond interpretation of the raw timestamp values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from writingring.discovery import (
    FileKind,
    Recording,
    parse_recording_filename,
)


RING_COLUMNS: Final[tuple[str, ...]] = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "timestamp",
)
RELATIVE_TIME_COLUMN: Final[str] = "relative_time_inferred_s"
VALUES_PER_SAMPLE: Final[int] = len(RING_COLUMNS)
FLOAT64_SIZE_BYTES: Final[int] = np.dtype(np.float64).itemsize
BACKWARD_POSITION_SAMPLE_LIMIT: Final[int] = 20
INFERRED_TIMESTAMP_UNIT: Final[str] = (
    "microseconds (observed/inferred, not confirmed upstream)"
)


class RingLoadError(Exception):
    """Base exception for ring files that cannot be loaded safely."""


class RingPathError(RingLoadError):
    """Raised for a missing, non-file, or non-ring_0 source path."""


class EmptyRingFileError(RingLoadError):
    """Raised when a ring file has no payload."""


class MalformedRingFileError(RingLoadError):
    """Raised when a ring payload cannot contain complete seven-value rows."""


@dataclass(frozen=True, slots=True)
class ChannelStatistics:
    """Finite-value statistics and non-finite counts for one raw column.

    Standard deviation is the population standard deviation (``ddof=0``)
    over finite values only. Extrema and moments are ``None`` when a column
    contains no finite values.
    """

    finite_count: int
    nan_count: int
    positive_infinity_count: int
    negative_infinity_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    standard_deviation: float | None


@dataclass(frozen=True, slots=True)
class TimestampInterpretation:
    """Metadata that keeps inferred timestamp semantics explicit."""

    raw_timestamp_column: str
    raw_timestamp_unit_confirmed: bool
    inferred_timestamp_unit: str
    inference_basis: str
    seconds_per_raw_unit_inferred: float
    relative_time_column: str | None
    synthetic_time_axis_used: bool
    nominal_sampling_rate_hz: float | None


@dataclass(frozen=True, slots=True)
class RingValidationReport:
    """Structured validation results for one successfully decoded ring file."""

    sample_count: int
    value_count: int
    file_size_bytes: int
    expected_sample_count_from_file_size: int
    raw_columns_present: bool
    raw_timestamp_start: float
    raw_timestamp_end: float
    raw_timestamp_duration: float
    timestamps_finite: bool
    timestamps_nondecreasing: bool
    timestamps_strictly_increasing: bool
    duplicate_timestamp_steps: int
    backward_timestamp_steps: int
    backward_timestamp_step_positions: tuple[int, ...]
    backward_positions_truncated: bool
    inferred_timestamp_unit: str
    inferred_duration_s: float | None
    inferred_sampling_rate_hz: float | None
    nan_counts_by_column: dict[str, int]
    positive_infinity_counts_by_column: dict[str, int]
    negative_infinity_counts_by_column: dict[str, int]
    per_channel_statistics: dict[str, ChannelStatistics]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RingData:
    """Loaded raw ring values plus validation and interpretation metadata."""

    source_path: Path
    dataframe: pd.DataFrame
    validation: RingValidationReport
    timestamp_interpretation: TimestampInterpretation
    warnings: tuple[str, ...]

    @property
    def raw_shape(self) -> tuple[int, int]:
        """Shape of only the seven confirmed upstream columns."""

        return self.dataframe.loc[:, list(RING_COLUMNS)].shape


def load_ring(source: Recording | str | Path) -> RingData:
    """Load one primary ring stream from a recording or explicit path.

    The explicit path must match ``*_ring_0.bin``. Passing a
    :class:`Recording` always selects ``recording.ring_0_path`` and never
    consults ``ring_1_path``.

    Raises:
        RingPathError: The path is missing, not a regular file, or is not a
            recognized ``ring_0`` filename.
        EmptyRingFileError: The file is empty.
        MalformedRingFileError: The byte or value count cannot form complete
            float64 seven-column samples.
        RingLoadError: NumPy cannot read the otherwise valid-looking path.
    """

    path = _resolve_ring_0_path(source)
    _validate_path(path)

    try:
        file_size_bytes = path.stat().st_size
    except OSError as error:
        raise RingPathError(f"cannot stat ring file {path}: {error}") from error

    if file_size_bytes == 0:
        raise EmptyRingFileError(f"ring file is empty: {path}")
    if file_size_bytes % FLOAT64_SIZE_BYTES != 0:
        raise MalformedRingFileError(
            f"ring file byte length {file_size_bytes} is not divisible by "
            f"float64 size {FLOAT64_SIZE_BYTES}: {path}"
        )

    expected_value_count = file_size_bytes // FLOAT64_SIZE_BYTES
    if expected_value_count % VALUES_PER_SAMPLE != 0:
        raise MalformedRingFileError(
            f"ring file contains {expected_value_count} float64 values; "
            f"the count is not divisible by {VALUES_PER_SAMPLE}: {path}"
        )

    try:
        raw = np.fromfile(path, dtype=np.float64)
    except OSError as error:
        raise RingLoadError(f"cannot read ring file {path}: {error}") from error

    if raw.size != expected_value_count:
        raise MalformedRingFileError(
            f"ring file changed or was read incompletely: expected "
            f"{expected_value_count} values, decoded {raw.size}: {path}"
        )

    values = raw.reshape(-1, VALUES_PER_SAMPLE)
    dataframe = pd.DataFrame(values, columns=RING_COLUMNS)
    dataframe.index = pd.RangeIndex(len(dataframe), name="sample_index")

    validation, relative_time = _build_validation_report(
        values,
        file_size_bytes=file_size_bytes,
    )
    if relative_time is not None:
        dataframe[RELATIVE_TIME_COLUMN] = relative_time

    interpretation = TimestampInterpretation(
        raw_timestamp_column="timestamp",
        raw_timestamp_unit_confirmed=False,
        inferred_timestamp_unit=INFERRED_TIMESTAMP_UNIT,
        inference_basis=(
            "The supplied sample aligns with board/text timestamps and yields "
            "about 201 Hz when interpreted as microseconds; the ring writer "
            "and an upstream unit contract are not supplied."
        ),
        seconds_per_raw_unit_inferred=1e-6,
        relative_time_column=(
            RELATIVE_TIME_COLUMN if relative_time is not None else None
        ),
        synthetic_time_axis_used=False,
        nominal_sampling_rate_hz=None,
    )
    return RingData(
        source_path=path,
        dataframe=dataframe,
        validation=validation,
        timestamp_interpretation=interpretation,
        warnings=validation.warnings,
    )


def _resolve_ring_0_path(source: Recording | str | Path) -> Path:
    path = source.ring_0_path if isinstance(source, Recording) else Path(source)
    parsed = parse_recording_filename(path.name)
    if parsed is None or parsed.kind is not FileKind.RING_0:
        raise RingPathError(
            f"only *_ring_0.bin may be loaded as primary ring input: {path}"
        )
    return path


def _validate_path(path: Path) -> None:
    if not path.exists():
        raise RingPathError(f"ring file does not exist: {path}")
    if not path.is_file():
        raise RingPathError(f"ring path is not a regular file: {path}")


def _build_validation_report(
    values: np.ndarray,
    *,
    file_size_bytes: int,
) -> tuple[RingValidationReport, np.ndarray | None]:
    sample_count, column_count = values.shape
    value_count = values.size
    statistics = {
        name: _channel_statistics(values[:, index])
        for index, name in enumerate(RING_COLUMNS)
    }
    nan_counts = {
        name: stats.nan_count for name, stats in statistics.items()
    }
    positive_infinity_counts = {
        name: stats.positive_infinity_count
        for name, stats in statistics.items()
    }
    negative_infinity_counts = {
        name: stats.negative_infinity_count
        for name, stats in statistics.items()
    }

    timestamps = values[:, -1]
    timestamps_finite = bool(np.isfinite(timestamps).all())
    with np.errstate(invalid="ignore", over="ignore"):
        timestamp_differences = np.diff(timestamps)
        raw_timestamp_duration = float(timestamps[-1] - timestamps[0])

    duplicate_steps = int(np.count_nonzero(timestamp_differences == 0))
    backward_positions_array = np.flatnonzero(timestamp_differences < 0) + 1
    backward_steps = int(backward_positions_array.size)
    backward_positions = tuple(
        int(position)
        for position in backward_positions_array[:BACKWARD_POSITION_SAMPLE_LIMIT]
    )
    timestamps_nondecreasing = timestamps_finite and bool(
        np.all(timestamp_differences >= 0)
    )
    timestamps_strictly_increasing = timestamps_finite and bool(
        np.all(timestamp_differences > 0)
    )

    warnings = _build_warnings(
        statistics=statistics,
        sample_count=sample_count,
        timestamps_finite=timestamps_finite,
        duplicate_steps=duplicate_steps,
        backward_steps=backward_steps,
        backward_positions=backward_positions,
        timestamps_nondecreasing=timestamps_nondecreasing,
        raw_timestamp_duration=raw_timestamp_duration,
    )

    inferred_duration_s: float | None = None
    inferred_sampling_rate_hz: float | None = None
    if (
        sample_count >= 2
        and timestamps_nondecreasing
        and np.isfinite(raw_timestamp_duration)
        and raw_timestamp_duration > 0
    ):
        inferred_duration_s = raw_timestamp_duration * 1e-6
        inferred_sampling_rate_hz = (
            (sample_count - 1) / inferred_duration_s
        )

    relative_time: np.ndarray | None = None
    if timestamps_finite:
        relative_time = (timestamps - timestamps[0]) * 1e-6

    report = RingValidationReport(
        sample_count=sample_count,
        value_count=value_count,
        file_size_bytes=file_size_bytes,
        expected_sample_count_from_file_size=(
            file_size_bytes // (FLOAT64_SIZE_BYTES * VALUES_PER_SAMPLE)
        ),
        raw_columns_present=(
            column_count == VALUES_PER_SAMPLE
            and len(RING_COLUMNS) == VALUES_PER_SAMPLE
        ),
        raw_timestamp_start=float(timestamps[0]),
        raw_timestamp_end=float(timestamps[-1]),
        raw_timestamp_duration=raw_timestamp_duration,
        timestamps_finite=timestamps_finite,
        timestamps_nondecreasing=timestamps_nondecreasing,
        timestamps_strictly_increasing=timestamps_strictly_increasing,
        duplicate_timestamp_steps=duplicate_steps,
        backward_timestamp_steps=backward_steps,
        backward_timestamp_step_positions=backward_positions,
        backward_positions_truncated=(
            backward_steps > BACKWARD_POSITION_SAMPLE_LIMIT
        ),
        inferred_timestamp_unit=INFERRED_TIMESTAMP_UNIT,
        inferred_duration_s=inferred_duration_s,
        inferred_sampling_rate_hz=inferred_sampling_rate_hz,
        nan_counts_by_column=nan_counts,
        positive_infinity_counts_by_column=positive_infinity_counts,
        negative_infinity_counts_by_column=negative_infinity_counts,
        per_channel_statistics=statistics,
        warnings=warnings,
    )
    return report, relative_time


def _channel_statistics(values: np.ndarray) -> ChannelStatistics:
    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        mean = float(np.mean(finite_values))
        standard_deviation = float(np.std(finite_values, ddof=0))
    else:
        minimum = None
        maximum = None
        mean = None
        standard_deviation = None

    return ChannelStatistics(
        finite_count=int(finite_values.size),
        nan_count=int(np.count_nonzero(np.isnan(values))),
        positive_infinity_count=int(np.count_nonzero(np.isposinf(values))),
        negative_infinity_count=int(np.count_nonzero(np.isneginf(values))),
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        standard_deviation=standard_deviation,
    )


def _build_warnings(
    *,
    statistics: dict[str, ChannelStatistics],
    sample_count: int,
    timestamps_finite: bool,
    duplicate_steps: int,
    backward_steps: int,
    backward_positions: tuple[int, ...],
    timestamps_nondecreasing: bool,
    raw_timestamp_duration: float,
) -> tuple[str, ...]:
    warnings: list[str] = []
    for name, stats in statistics.items():
        if (
            stats.nan_count
            or stats.positive_infinity_count
            or stats.negative_infinity_count
        ):
            warnings.append(
                f"non-finite values in {name}: NaN={stats.nan_count}, "
                f"+inf={stats.positive_infinity_count}, "
                f"-inf={stats.negative_infinity_count}"
            )

    if not timestamps_finite:
        warnings.append("timestamp column contains non-finite values")
    if duplicate_steps:
        warnings.append(
            f"{duplicate_steps} duplicate timestamp step(s) observed"
        )
    if backward_steps:
        formatted_positions = ", ".join(
            str(position) for position in backward_positions
        )
        suffix = (
            ", ..." if backward_steps > len(backward_positions) else ""
        )
        warnings.append(
            f"{backward_steps} backward timestamp step(s) observed at sample "
            f"indices {formatted_positions}{suffix}"
        )

    if sample_count < 2:
        warnings.append(
            "inferred sampling rate unavailable: fewer than two samples"
        )
    elif not timestamps_nondecreasing:
        warnings.append(
            "inferred sampling rate unavailable: timestamps are not finite "
            "and nondecreasing"
        )
    elif not np.isfinite(raw_timestamp_duration) or raw_timestamp_duration <= 0:
        warnings.append(
            "inferred sampling rate unavailable: raw timestamp duration is "
            "not finite and positive"
        )

    return tuple(warnings)

