"""Safe export and application of constant Ring--Board time offsets.

The only supported mapping is ``ring_timestamp_us = board_timestamp_us +
offset_us``.  Keeping this convention in one module prevents sign changes
between matching, saved offsets, and verification plots.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Final, Sequence

import numpy as np

from writingring.event_alignment import SequenceAlignmentResult


ALIGNMENT_MODEL: Final[str] = "constant_offset"
TIMESTAMP_UNIT: Final[str] = "microseconds"
TIME_MAPPING: Final[str] = "ring_timestamp_us = board_timestamp_us + offset_us"


class AlignmentOffsetExportError(ValueError):
    """Raised when a constant offset cannot be safely exported or consumed."""


def sha256_array(values: np.ndarray) -> str:
    """Hash a numeric NumPy array's contiguous bytes without mutating it."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentOffsetExportError("array to hash must be numeric") from error
    if array.ndim == 0 or not np.isfinite(array).all():
        raise AlignmentOffsetExportError("array to hash must be finite and non-scalar")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class AlignmentTimeAxes:
    """Canonical and strictly increasing work axes for one alignment input."""

    canonical_timestamps_us: np.ndarray
    work_timestamps_us: np.ndarray
    input_kind: str
    sampling_rate_hz: float
    strategy: str
    duplicate_step_count: int

    @property
    def alignment_timestamps_us(self) -> np.ndarray:
        """Return the work axis under the explicit alignment terminology."""

        return self.work_timestamps_us

    @property
    def metadata(self) -> dict[str, object]:
        """Return report-safe metadata without hashing the synthetic work axis."""

        return {
            "ordering": "nondecreasing",
            "duplicate_step_count": self.duplicate_step_count,
            "strategy": self.strategy,
            "sampling_rate_hz": self.sampling_rate_hz,
            "sample_count": int(self.canonical_timestamps_us.size),
            "canonical_timestamps_modified": False,
        }


@dataclass(frozen=True, slots=True)
class AlignmentOffsetProjection:
    """Projection of a work-axis offset into the canonical timestamp domain."""

    work_axis_offset_us: float
    canonical_offset_us: float
    projection_delta_us: float
    projection_delta_median_us: float
    projection_delta_mad_us: float
    projection_delta_min_us: float
    projection_delta_max_us: float
    projection_delta_range_us: float
    contributing_match_count: int
    warnings: tuple[str, ...] = ()

    @property
    def projection_delta_ptp_us(self) -> float:
        """Return the delta range using the common peak-to-peak spelling."""

        return self.projection_delta_range_us


def build_alignment_time_axes(
    canonical_timestamps_us: np.ndarray,
    *,
    input_kind: str,
    sampling_rate_hz: float | None = None,
) -> AlignmentTimeAxes:
    """Build canonical and source-aware strictly increasing alignment axes.

    The canonical vector is copied and never rewritten.  Raw Ring alignment
    intentionally retains the historical endpoint reconstruction for every
    nondecreasing timestamp stream, including strictly increasing streams
    with sampling jitter.  SpikeIMU keeps a strict canonical vector as its
    work axis and reconstructs only when duplicate rows require it.
    """

    canonical = _validated_timestamp_vector(canonical_timestamps_us)
    if input_kind not in {"raw-ring", "spike-imu"}:
        raise AlignmentOffsetExportError(
            "input_kind must be 'raw-ring' or 'spike-imu'"
        )
    duplicate_step_count = int(np.count_nonzero(np.diff(canonical) == 0.0))
    has_duplicates = duplicate_step_count > 0
    supplied_rate = None
    if sampling_rate_hz is not None:
        supplied_rate = _finite_float(sampling_rate_hz, name="sampling_rate_hz")
        if supplied_rate <= 0.0:
            raise AlignmentOffsetExportError("sampling_rate_hz must be positive")

    duration_us = float(canonical[-1] - canonical[0])
    endpoint_rate: float | None = None
    if duration_us > 0.0:
        endpoint_rate = (canonical.size - 1) * 1_000_000.0 / duration_us

    if input_kind == "raw-ring":
        if endpoint_rate is None:
            raise AlignmentOffsetExportError(
                "timestamps must have positive duration for alignment"
            )
        rate = endpoint_rate
        strategy = "raw_endpoint_reconstruction"
        work = canonical[0] + (
            np.arange(canonical.size, dtype=np.float64) * 1_000_000.0 / rate
        )
    else:
        if supplied_rate is None:
            if endpoint_rate is None:
                raise AlignmentOffsetExportError(
                    "sampling_rate_hz was not supplied and cannot be inferred "
                    "from SpikeIMU timestamp endpoints"
                )
            rate = endpoint_rate
        else:
            rate = supplied_rate
        if has_duplicates:
            strategy = "strict_reconstruction"
            work = canonical[0] + (
                np.arange(canonical.size, dtype=np.float64) * 1_000_000.0 / rate
            )
        else:
            strategy = "canonical_strict_copy"
            work = canonical.copy()

    if work.ndim != 1 or not np.isfinite(work).all() or not np.all(np.diff(work) > 0.0):
        raise AlignmentOffsetExportError(
            "alignment work timestamps must be finite and strictly increasing"
        )
    return AlignmentTimeAxes(
        canonical_timestamps_us=canonical,
        work_timestamps_us=work,
        input_kind=input_kind,
        sampling_rate_hz=float(rate),
        strategy=strategy,
        duplicate_step_count=duplicate_step_count,
    )


def build_alignment_timestamps(
    canonical_timestamps_us: np.ndarray,
    sampling_rate_hz: float | None = None,
    *,
    input_kind: str = "spike-imu",
) -> np.ndarray:
    """Compatibility wrapper for building one source-aware work axis."""

    return build_alignment_time_axes(
        canonical_timestamps_us,
        input_kind=input_kind,
        sampling_rate_hz=sampling_rate_hz,
    ).work_timestamps_us.copy()


def alignment_timestamp_metadata(
    canonical_timestamps_us: np.ndarray,
    alignment_timestamps_us: np.ndarray,
    sampling_rate_hz: float | None = None,
    *,
    input_kind: str = "spike-imu",
) -> dict[str, object]:
    """Validate and describe the canonical/work-axis relationship."""

    axes = build_alignment_time_axes(
        canonical_timestamps_us,
        input_kind=input_kind,
        sampling_rate_hz=sampling_rate_hz,
    )
    try:
        alignment = np.asarray(alignment_timestamps_us, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentOffsetExportError(
            "alignment_timestamps_us must be numeric"
        ) from error
    if (
        alignment.ndim != 1
        or alignment.size != axes.canonical_timestamps_us.size
        or not np.isfinite(alignment).all()
        or not np.all(np.diff(alignment) > 0.0)
    ):
        raise AlignmentOffsetExportError(
            "alignment_timestamps_us must be a finite, strictly increasing vector "
            "with the canonical sample count"
        )
    if not np.array_equal(alignment, axes.work_timestamps_us):
        raise AlignmentOffsetExportError(
            "alignment_timestamps_us does not match the canonical work-axis strategy"
        )
    return dict(axes.metadata)


def _validated_timestamp_vector(canonical_timestamps_us: np.ndarray) -> np.ndarray:
    try:
        canonical = np.asarray(canonical_timestamps_us, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentOffsetExportError(
            "canonical_timestamps_us must be numeric"
        ) from error
    if canonical.ndim != 1 or canonical.size < 2 or not np.isfinite(canonical).all():
        raise AlignmentOffsetExportError(
            "canonical_timestamps_us must be a finite vector with at least two samples"
        )
    differences = np.diff(canonical)
    if np.any(differences < 0.0):
        raise AlignmentOffsetExportError(
            "canonical_timestamps_us must be nondecreasing"
        )
    return canonical.copy()


def project_alignment_offset_to_canonical(
    result: SequenceAlignmentResult,
    canonical_timestamps_us: np.ndarray,
    alignment_work_timestamps_us: np.ndarray,
    *,
    matched_peak_indices: Sequence[int] | np.ndarray | None = None,
    sampling_rate_hz: float | None = None,
    warning_threshold_samples: float = 0.5,
    failure_threshold_samples: float = 1.5,
) -> AlignmentOffsetProjection:
    """Project a work-axis alignment offset into canonical timestamp space.

    Matching residuals stay on the work axis.  Only the coordinate difference
    ``canonical[k] - work[k]`` for selected peak rows is projected, so residual
    matching error is not silently absorbed into the exported Board-to-Ring
    offset.
    """

    if not isinstance(result, SequenceAlignmentResult):
        raise AlignmentOffsetExportError(
            "result must be a SequenceAlignmentResult"
        )
    work_offset = _finite_float(result.best_offset_us, name="work_axis_offset_us")
    canonical = _validated_timestamp_vector(canonical_timestamps_us)
    try:
        work = np.asarray(alignment_work_timestamps_us, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentOffsetExportError(
            "alignment_work_timestamps_us must be numeric"
        ) from error
    if (
        work.ndim != 1
        or work.size != canonical.size
        or not np.isfinite(work).all()
        or not np.all(np.diff(work) > 0.0)
    ):
        raise AlignmentOffsetExportError(
            "alignment_work_timestamps_us must be finite, strictly increasing, "
            "and match canonical sample count"
        )
    if matched_peak_indices is None:
        matches = result.event_matches
        if not hasattr(matches, "columns") or not {
            "matched",
            "matched_peak_index",
        }.issubset(matches.columns):
            raise AlignmentOffsetExportError(
                "successful alignment result is missing matched peak indices"
            )
        matched = matches.loc[matches["matched"].astype(bool), "matched_peak_index"]
        matched_peak_indices = matched.to_numpy()
    indices = _validated_match_indices(matched_peak_indices, sample_count=canonical.size)
    deltas = canonical[indices] - work[indices]
    median = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - median)))
    minimum = float(np.min(deltas))
    maximum = float(np.max(deltas))
    delta_range = maximum - minimum
    rate = (
        _finite_float(sampling_rate_hz, name="sampling_rate_hz")
        if sampling_rate_hz is not None
        else (work.size - 1) * 1_000_000.0 / float(work[-1] - work[0])
    )
    if rate <= 0.0:
        raise AlignmentOffsetExportError("sampling_rate_hz must be positive")
    sample_interval_us = 1_000_000.0 / rate
    warning_threshold = _finite_float(
        warning_threshold_samples, name="warning_threshold_samples"
    )
    failure_threshold = _finite_float(
        failure_threshold_samples, name="failure_threshold_samples"
    )
    if warning_threshold < 0.0 or failure_threshold <= warning_threshold:
        raise AlignmentOffsetExportError(
            "projection quality thresholds must be nonnegative and ordered"
        )
    warnings: list[str] = []
    if delta_range > failure_threshold * sample_interval_us:
        raise AlignmentOffsetExportError(
            "canonical offset projection is not representable by one constant "
            f"offset: delta range {delta_range:.3f} us exceeds "
            f"{failure_threshold:.3f} sampling intervals"
        )
    if delta_range > warning_threshold * sample_interval_us:
        warnings.append(
            "canonical/work timestamp projection delta range is larger than "
            f"{warning_threshold:.3f} sampling intervals"
        )
    return AlignmentOffsetProjection(
        work_axis_offset_us=work_offset,
        canonical_offset_us=work_offset + median,
        projection_delta_us=median,
        projection_delta_median_us=median,
        projection_delta_mad_us=mad,
        projection_delta_min_us=minimum,
        projection_delta_max_us=maximum,
        projection_delta_range_us=delta_range,
        contributing_match_count=int(indices.size),
        warnings=tuple(warnings),
    )


def _validated_match_indices(
    values: Sequence[int] | np.ndarray,
    *,
    sample_count: int,
) -> np.ndarray:
    try:
        indices_float = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentOffsetExportError(
            "matched_peak_indices must be numeric"
        ) from error
    if (
        indices_float.ndim != 1
        or indices_float.size == 0
        or not np.isfinite(indices_float).all()
        or not np.all(indices_float == np.floor(indices_float))
    ):
        raise AlignmentOffsetExportError(
            "matched_peak_indices must be a nonempty integer vector"
        )
    indices = indices_float.astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= sample_count):
        raise AlignmentOffsetExportError(
            "matched_peak_indices are outside the canonical timestamp range"
        )
    return indices


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
    timestamp_source_sha256: str | None = None
    timestamp_source_ordering: str | None = None
    timestamp_source_duplicate_step_count: int | None = None
    alignment_time_axis_strategy: str | None = None
    alignment_time_axis_sampling_rate_hz: float | None = None
    alignment_time_axis_sample_count: int | None = None
    canonical_timestamps_modified: bool | None = None
    work_axis_offset_us: float | None = None
    projection_delta_us: float | None = None
    projection_delta_median_us: float | None = None
    projection_delta_mad_us: float | None = None
    projection_delta_min_us: float | None = None
    projection_delta_max_us: float | None = None
    projection_delta_range_us: float | None = None
    projection_contributing_match_count: int | None = None

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
    projection: AlignmentOffsetProjection | None = None,
) -> AlignmentOffset:
    """Extract an offset, using a canonical projection when one is supplied.

    The legacy no-projection path remains available for results that do not
    carry a source-specific work-axis contract.  Source-aware results must
    provide ``projection`` so ``AlignmentOffset.offset_us`` remains safe for
    downstream canonical timestamp segmentation.
    """

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
    work_axis_offset_us = _finite_float(
        result.best_offset_us, name="work_axis_offset_us"
    )
    report = result.report
    if not isinstance(report, dict):
        raise AlignmentOffsetExportError("alignment result report must be a dictionary")
    model = report.get("alignment_model", ALIGNMENT_MODEL)
    if model != ALIGNMENT_MODEL:
        raise AlignmentOffsetExportError(
            "alignment result must use the constant_offset model"
        )
    if projection is None:
        if "alignment_time_axis" in report or "canonical_offset_projection" in report:
            raise AlignmentOffsetExportError(
                "source-aware alignment results require a canonical offset projection"
            )
        offset_us = work_axis_offset_us
        projection_fields: dict[str, object] = {}
    else:
        _validate_projection(projection)
        if not math.isclose(
            projection.work_axis_offset_us,
            work_axis_offset_us,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise AlignmentOffsetExportError(
                "canonical offset projection does not match alignment result"
            )
        offset_us = projection.canonical_offset_us
        projection_fields = {
            "work_axis_offset_us": projection.work_axis_offset_us,
            "projection_delta_us": projection.projection_delta_us,
            "projection_delta_median_us": projection.projection_delta_median_us,
            "projection_delta_mad_us": projection.projection_delta_mad_us,
            "projection_delta_min_us": projection.projection_delta_min_us,
            "projection_delta_max_us": projection.projection_delta_max_us,
            "projection_delta_range_us": projection.projection_delta_range_us,
            "projection_contributing_match_count": (
                projection.contributing_match_count
            ),
        }
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
        timestamp_source_sha256=_report_timestamp_source_value(
            report, "sha256"
        ),
        timestamp_source_ordering=_report_timestamp_source_value(
            report, "ordering"
        ),
        timestamp_source_duplicate_step_count=_report_timestamp_source_int(
            report, "duplicate_step_count"
        ),
        alignment_time_axis_strategy=_report_alignment_axis_value(
            report, "strategy"
        ),
        alignment_time_axis_sampling_rate_hz=_report_alignment_axis_float(
            report, "sampling_rate_hz"
        ),
        alignment_time_axis_sample_count=_report_alignment_axis_int(
            report, "sample_count"
        ),
        canonical_timestamps_modified=_report_alignment_axis_bool(
            report, "canonical_timestamps_modified"
        ),
        **projection_fields,
    )


def _validate_projection(projection: AlignmentOffsetProjection) -> None:
    if not isinstance(projection, AlignmentOffsetProjection):
        raise AlignmentOffsetExportError(
            "projection must be an AlignmentOffsetProjection"
        )
    work_offset = _finite_float(
        projection.work_axis_offset_us, name="work_axis_offset_us"
    )
    canonical_offset = _finite_float(
        projection.canonical_offset_us, name="canonical_offset_us"
    )
    delta = _finite_float(projection.projection_delta_us, name="projection_delta_us")
    median = _finite_float(
        projection.projection_delta_median_us,
        name="projection_delta_median_us",
    )
    mad = _finite_float(projection.projection_delta_mad_us, name="projection_delta_mad_us")
    minimum = _finite_float(
        projection.projection_delta_min_us, name="projection_delta_min_us"
    )
    maximum = _finite_float(
        projection.projection_delta_max_us, name="projection_delta_max_us"
    )
    delta_range = _finite_float(
        projection.projection_delta_range_us,
        name="projection_delta_range_us",
    )
    if not math.isclose(canonical_offset, work_offset + delta, abs_tol=1e-6):
        raise AlignmentOffsetExportError(
            "canonical_offset_us must equal work_axis_offset_us plus projection_delta_us"
        )
    if not math.isclose(delta, median, abs_tol=1e-6):
        raise AlignmentOffsetExportError(
            "projection_delta_us must equal projection_delta_median_us"
        )
    if mad < 0.0 or minimum > maximum or delta_range < 0.0:
        raise AlignmentOffsetExportError("projection diagnostics are inconsistent")
    if not math.isclose(delta_range, maximum - minimum, abs_tol=1e-6):
        raise AlignmentOffsetExportError(
            "projection_delta_range_us must equal max minus min"
        )
    count = _nonnegative_int(
        projection.contributing_match_count,
        name="projection_contributing_match_count",
    )
    if count < 1:
        raise AlignmentOffsetExportError(
            "projection_contributing_match_count must be positive"
        )
    if not isinstance(projection.warnings, tuple) or any(
        not isinstance(warning, str) or not warning for warning in projection.warnings
    ):
        raise AlignmentOffsetExportError("projection warnings must be strings")


def _report_timestamp_source_value(
    report: dict[str, object], key: str
) -> str | None:
    source = report.get("timestamp_source")
    if not isinstance(source, dict):
        return None
    value = source.get(key)
    if key == "sha256":
        return _optional_sha256(value, name="timestamp_source_sha256")
    return _optional_nonempty_string(value, name=f"timestamp_source_{key}")


def _report_timestamp_source_int(
    report: dict[str, object], key: str
) -> int | None:
    source = report.get("timestamp_source")
    if not isinstance(source, dict):
        return None
    return _optional_nonnegative_int(
        source.get(key), name=f"timestamp_source_{key}"
    )


def _alignment_report_section(report: dict[str, object]) -> dict[str, object] | None:
    value = report.get("alignment_time_axis")
    return value if isinstance(value, dict) else None


def _report_alignment_axis_value(
    report: dict[str, object], key: str
) -> str | None:
    section = _alignment_report_section(report)
    if section is None:
        return None
    return _optional_nonempty_string(
        section.get(key), name=f"alignment_time_axis_{key}"
    )


def _report_alignment_axis_float(
    report: dict[str, object], key: str
) -> float | None:
    section = _alignment_report_section(report)
    if section is None:
        return None
    return _optional_finite_float(
        section.get(key), name=f"alignment_time_axis_{key}"
    )


def _report_alignment_axis_int(
    report: dict[str, object], key: str
) -> int | None:
    section = _alignment_report_section(report)
    if section is None:
        return None
    return _optional_nonnegative_int(
        section.get(key), name=f"alignment_time_axis_{key}"
    )


def _report_alignment_axis_bool(
    report: dict[str, object], key: str
) -> bool | None:
    section = _alignment_report_section(report)
    if section is None:
        return None
    return _optional_bool(section.get(key), name=f"alignment_time_axis_{key}")


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
        timestamp_source_sha256=_optional_sha256(
            fields.get("timestamp_source_sha256"),
            name="timestamp_source_sha256",
        ),
        timestamp_source_ordering=_optional_nonempty_string(
            fields.get("timestamp_source_ordering"),
            name="timestamp_source_ordering",
        ),
        timestamp_source_duplicate_step_count=_optional_nonnegative_int(
            fields.get("timestamp_source_duplicate_step_count"),
            name="timestamp_source_duplicate_step_count",
        ),
        alignment_time_axis_strategy=_optional_nonempty_string(
            fields.get("alignment_time_axis_strategy"),
            name="alignment_time_axis_strategy",
        ),
        alignment_time_axis_sampling_rate_hz=_optional_finite_float(
            fields.get("alignment_time_axis_sampling_rate_hz"),
            name="alignment_time_axis_sampling_rate_hz",
        ),
        alignment_time_axis_sample_count=_optional_nonnegative_int(
            fields.get("alignment_time_axis_sample_count"),
            name="alignment_time_axis_sample_count",
        ),
        canonical_timestamps_modified=_optional_bool(
            fields.get("canonical_timestamps_modified"),
            name="canonical_timestamps_modified",
        ),
        work_axis_offset_us=_optional_finite_float(
            fields.get("work_axis_offset_us"), name="work_axis_offset_us"
        ),
        projection_delta_us=_optional_finite_float(
            fields.get("projection_delta_us"), name="projection_delta_us"
        ),
        projection_delta_median_us=_optional_finite_float(
            fields.get("projection_delta_median_us"),
            name="projection_delta_median_us",
        ),
        projection_delta_mad_us=_optional_finite_float(
            fields.get("projection_delta_mad_us"),
            name="projection_delta_mad_us",
        ),
        projection_delta_min_us=_optional_finite_float(
            fields.get("projection_delta_min_us"),
            name="projection_delta_min_us",
        ),
        projection_delta_max_us=_optional_finite_float(
            fields.get("projection_delta_max_us"),
            name="projection_delta_max_us",
        ),
        projection_delta_range_us=_optional_finite_float(
            fields.get("projection_delta_range_us"),
            name="projection_delta_range_us",
        ),
        projection_contributing_match_count=_optional_nonnegative_int(
            fields.get("projection_contributing_match_count"),
            name="projection_contributing_match_count",
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
    timestamp_provenance = ""
    if offset.timestamp_source_sha256 is not None:
        assert offset.timestamp_source_ordering is not None
        assert offset.timestamp_source_duplicate_step_count is not None
        assert offset.alignment_time_axis_strategy is not None
        assert offset.alignment_time_axis_sampling_rate_hz is not None
        assert offset.alignment_time_axis_sample_count is not None
        assert offset.canonical_timestamps_modified is not None
        timestamp_provenance = (
            f"timestamp_source_sha256={offset.timestamp_source_sha256}\n"
            f"timestamp_source_ordering={offset.timestamp_source_ordering}\n"
            "timestamp_source_duplicate_step_count="
            f"{offset.timestamp_source_duplicate_step_count}\n"
            "alignment_time_axis_strategy="
            f"{offset.alignment_time_axis_strategy}\n"
            "alignment_time_axis_sampling_rate_hz="
            f"{offset.alignment_time_axis_sampling_rate_hz:.12g}\n"
            "alignment_time_axis_sample_count="
            f"{offset.alignment_time_axis_sample_count}\n"
            "canonical_timestamps_modified="
            f"{'true' if offset.canonical_timestamps_modified else 'false'}\n"
        )
    projection = ""
    if offset.work_axis_offset_us is not None:
        assert offset.projection_delta_us is not None
        assert offset.projection_delta_median_us is not None
        assert offset.projection_delta_mad_us is not None
        assert offset.projection_delta_min_us is not None
        assert offset.projection_delta_max_us is not None
        assert offset.projection_delta_range_us is not None
        assert offset.projection_contributing_match_count is not None
        projection = (
            f"work_axis_offset_us={offset.work_axis_offset_us:.6f}\n"
            f"projection_delta_us={offset.projection_delta_us:.6f}\n"
            "projection_delta_median_us="
            f"{offset.projection_delta_median_us:.6f}\n"
            "projection_delta_mad_us="
            f"{offset.projection_delta_mad_us:.6f}\n"
            "projection_delta_min_us="
            f"{offset.projection_delta_min_us:.6f}\n"
            "projection_delta_max_us="
            f"{offset.projection_delta_max_us:.6f}\n"
            "projection_delta_range_us="
            f"{offset.projection_delta_range_us:.6f}\n"
            "projection_contributing_match_count="
            f"{offset.projection_contributing_match_count}\n"
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
        f"{timestamp_provenance}"
        f"{projection}"
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
    timestamp_provenance = (
        offset.timestamp_source_sha256,
        offset.timestamp_source_ordering,
        offset.timestamp_source_duplicate_step_count,
        offset.alignment_time_axis_strategy,
        offset.alignment_time_axis_sampling_rate_hz,
        offset.alignment_time_axis_sample_count,
        offset.canonical_timestamps_modified,
    )
    if any(value is not None for value in timestamp_provenance):
        if any(value is None for value in timestamp_provenance):
            raise AlignmentOffsetExportError(
                "timestamp source provenance is incomplete"
            )
        _sha256_digest(
            offset.timestamp_source_sha256,
            name="timestamp_source_sha256",
        )
        if offset.timestamp_source_ordering != "nondecreasing":
            raise AlignmentOffsetExportError(
                "timestamp_source_ordering must be nondecreasing"
            )
        _nonnegative_int(
            offset.timestamp_source_duplicate_step_count,
            name="timestamp_source_duplicate_step_count",
        )
        if offset.alignment_time_axis_strategy not in {
            "canonical_strict_copy",
            "strict_reconstruction",
            "raw_endpoint_reconstruction",
        }:
            raise AlignmentOffsetExportError(
                "alignment_time_axis_strategy is unsupported"
            )
        sampling_rate = _finite_float(
            offset.alignment_time_axis_sampling_rate_hz,
            name="alignment_time_axis_sampling_rate_hz",
        )
        if sampling_rate <= 0.0:
            raise AlignmentOffsetExportError(
                "alignment_time_axis_sampling_rate_hz must be positive"
            )
        sample_count = _nonnegative_int(
            offset.alignment_time_axis_sample_count,
            name="alignment_time_axis_sample_count",
        )
        if sample_count < 3:
            raise AlignmentOffsetExportError(
                "alignment_time_axis_sample_count must be at least three"
            )
        if offset.canonical_timestamps_modified is not False:
            raise AlignmentOffsetExportError(
                "canonical_timestamps_modified must be false"
            )
    projection = (
        offset.work_axis_offset_us,
        offset.projection_delta_us,
        offset.projection_delta_median_us,
        offset.projection_delta_mad_us,
        offset.projection_delta_min_us,
        offset.projection_delta_max_us,
        offset.projection_delta_range_us,
        offset.projection_contributing_match_count,
    )
    if any(value is not None for value in projection):
        if any(value is None for value in projection):
            raise AlignmentOffsetExportError(
                "canonical offset projection is incomplete"
            )
        _validate_projection(
            AlignmentOffsetProjection(
                work_axis_offset_us=offset.work_axis_offset_us,  # type: ignore[arg-type]
                canonical_offset_us=offset.offset_us,
                projection_delta_us=offset.projection_delta_us,  # type: ignore[arg-type]
                projection_delta_median_us=offset.projection_delta_median_us,  # type: ignore[arg-type]
                projection_delta_mad_us=offset.projection_delta_mad_us,  # type: ignore[arg-type]
                projection_delta_min_us=offset.projection_delta_min_us,  # type: ignore[arg-type]
                projection_delta_max_us=offset.projection_delta_max_us,  # type: ignore[arg-type]
                projection_delta_range_us=offset.projection_delta_range_us,  # type: ignore[arg-type]
                contributing_match_count=offset.projection_contributing_match_count,  # type: ignore[arg-type]
            )
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
