"""Safe export and application of constant Ring--Board time offsets.

The only supported mapping is ``ring_timestamp_us = board_timestamp_us +
offset_us``.  Keeping this convention in one module prevents sign changes
between matching, saved offsets, and verification plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Final, Any

import numpy as np

from writingring.event_alignment import SequenceAlignmentResult


ALIGNMENT_MODEL: Final[str] = "constant_offset"
TIMESTAMP_UNIT: Final[str] = "microseconds"
TIME_MAPPING: Final[str] = "ring_timestamp_us = board_timestamp_us + offset_us"
ALIGNMENT_OFFSET_SCHEMA_VERSION: Final[int] = 2
ALIGNMENT_SKIP_SCHEMA_VERSION: Final[int] = 1
ALIGNMENT_OUTCOME_SCHEMA_VERSION: Final[int] = 1
ALIGNMENT_SKIP_REASON: Final[str] = "initial_interval_no_usable_pair"
ALIGNMENT_WORK_AXIS_DOMAIN: Final[str] = "alignment_work_axis"
CANONICAL_TIMESTAMP_DOMAIN: Final[str] = "canonical_timestamp"
_OFFSET_DOMAINS: Final[frozenset[str]] = frozenset(
    {ALIGNMENT_WORK_AXIS_DOMAIN, CANONICAL_TIMESTAMP_DOMAIN}
)


class AlignmentOffsetExportError(ValueError):
    """Raised when a constant offset cannot be safely exported or consumed."""


class AlignmentOutcomeStatus(str, Enum):
    """The two completed alignment states and the non-completed report state."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


# ``AlignmentStatus`` is a short compatibility spelling for callers that use
# the status terminology from the outcome contract.
AlignmentStatus = AlignmentOutcomeStatus


class AlignmentOutcomeError(AlignmentOffsetExportError):
    """Raised when an alignment outcome is absent, stale, or inconsistent."""


def _normalize_outcome_errors(function: Any) -> Any:
    """Normalize malformed public outcome inputs to ``AlignmentOutcomeError``.

    The low-level offset helpers intentionally retain their historical
    ``AlignmentOffsetExportError`` contract.  Outcome and skip readers sit on
    top of those helpers, so malformed JSON/provenance fields must cross this
    public boundary as the more specific outcome error instead of leaking a
    ``KeyError``, a built-in parsing error, or the base offset exception.
    """

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except AlignmentOutcomeError:
            raise
        except (
            AlignmentOffsetExportError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            raise AlignmentOutcomeError(
                f"malformed alignment outcome: {error}"
            ) from error

    return wrapped


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
    # ``sampling_rate_hz`` is retained for API compatibility, but explicitly
    # means the rate of the synthetic alignment work axis.  It must never be
    # populated from SpikeIMU feature metadata.
    sampling_rate_hz: float
    strategy: str
    duplicate_step_count: int
    feature_sampling_rate_hz: float | None = None

    @property
    def alignment_timestamps_us(self) -> np.ndarray:
        """Return the work axis under the explicit alignment terminology."""

        return self.work_timestamps_us

    @property
    def alignment_sampling_rate_hz(self) -> float:
        """Return the endpoint-derived sampling rate of the work axis."""

        return self.sampling_rate_hz

    @property
    def metadata(self) -> dict[str, object]:
        """Return report-safe metadata without hashing the synthetic work axis."""

        return {
            "ordering": "nondecreasing",
            "duplicate_step_count": self.duplicate_step_count,
            "strategy": self.strategy,
            "sampling_rate_hz": self.sampling_rate_hz,
            "sample_count": int(self.canonical_timestamps_us.size),
            "canonical_start_us": float(self.canonical_timestamps_us[0]),
            "canonical_stop_us": float(self.canonical_timestamps_us[-1]),
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
    """Build one representation-invariant strictly increasing alignment axis.

    ``sampling_rate_hz`` is accepted as feature metadata for validation and
    provenance only.  The alignment work axis is always reconstructed from
    the canonical timestamp endpoints so raw Ring and SpikeIMU inputs with
    the same canonical timestamps cannot acquire different time coordinates.
    """

    canonical = _validated_timestamp_vector(canonical_timestamps_us)
    if input_kind not in {"raw-ring", "spike-imu"}:
        raise AlignmentOffsetExportError(
            "input_kind must be 'raw-ring' or 'spike-imu'"
        )
    duplicate_step_count = int(np.count_nonzero(np.diff(canonical) == 0.0))
    feature_rate = None
    if sampling_rate_hz is not None:
        feature_rate = _finite_float(sampling_rate_hz, name="sampling_rate_hz")
        if feature_rate <= 0.0:
            raise AlignmentOffsetExportError("sampling_rate_hz must be positive")

    duration_us = float(canonical[-1] - canonical[0])
    if duration_us <= 0.0:
        raise AlignmentOffsetExportError(
            "timestamps must have positive duration for alignment"
        )

    endpoint_rate = (canonical.size - 1) * 1_000_000.0 / duration_us
    work = np.linspace(
        float(canonical[0]),
        float(canonical[-1]),
        canonical.size,
        dtype=np.float64,
    )
    strategy = "endpoint_reconstruction"

    if work.ndim != 1 or not np.isfinite(work).all() or not np.all(np.diff(work) > 0.0):
        raise AlignmentOffsetExportError(
            "alignment work timestamps must be finite and strictly increasing"
        )
    return AlignmentTimeAxes(
        canonical_timestamps_us=canonical,
        work_timestamps_us=work,
        input_kind=input_kind,
        sampling_rate_hz=float(endpoint_rate),
        strategy=strategy,
        duplicate_step_count=duplicate_step_count,
        feature_sampling_rate_hz=feature_rate,
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


def build_offset_domain_timestamps(
    canonical_timestamps_us: np.ndarray,
    *,
    offset_domain: str,
    input_kind: str,
    feature_sampling_rate_hz: float | None = None,
) -> np.ndarray:
    """Return the timestamp vector used to locate boundaries for an offset.

    New alignment artifacts use the endpoint-reconstructed work axis for
    matching and Board boundary lookup.  Legacy artifacts explicitly (or by
    schema inference) use canonical timestamps.  The canonical vector is
    always returned unchanged as the feature slicing/provenance vector.
    """

    canonical = _validated_timestamp_vector(canonical_timestamps_us)
    if offset_domain == CANONICAL_TIMESTAMP_DOMAIN:
        return canonical
    if offset_domain != ALIGNMENT_WORK_AXIS_DOMAIN:
        raise AlignmentOffsetExportError(
            "offset_domain must be alignment_work_axis or canonical_timestamp"
        )
    return build_alignment_time_axes(
        canonical,
        input_kind=input_kind,
        sampling_rate_hz=feature_sampling_rate_hz,
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
    projection = AlignmentOffsetProjection(
        work_axis_offset_us=work_offset,
        canonical_offset_us=work_offset + median,
        projection_delta_us=median,
        projection_delta_median_us=median,
        projection_delta_mad_us=mad,
        projection_delta_min_us=minimum,
        projection_delta_max_us=maximum,
        projection_delta_range_us=delta_range,
        contributing_match_count=int(indices.size),
        warnings=(),
    )
    if delta_range > failure_threshold * sample_interval_us:
        error = AlignmentOffsetExportError(
            "canonical offset projection is not representable by one constant "
            f"offset: delta range {delta_range:.3f} us exceeds "
            f"{failure_threshold:.3f} sampling intervals"
        )
        setattr(error, "projection_diagnostics", projection)
        raise error
    if delta_range > warning_threshold * sample_interval_us:
        warnings.append(
            "canonical/work timestamp projection delta range is larger than "
            f"{warning_threshold:.3f} sampling intervals"
        )
    return AlignmentOffsetProjection(
        work_axis_offset_us=projection.work_axis_offset_us,
        canonical_offset_us=projection.canonical_offset_us,
        projection_delta_us=projection.projection_delta_us,
        projection_delta_median_us=projection.projection_delta_median_us,
        projection_delta_mad_us=projection.projection_delta_mad_us,
        projection_delta_min_us=projection.projection_delta_min_us,
        projection_delta_max_us=projection.projection_delta_max_us,
        projection_delta_range_us=projection.projection_delta_range_us,
        contributing_match_count=projection.contributing_match_count,
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
    offset_schema_version: int | None = None
    offset_domain: str = CANONICAL_TIMESTAMP_DOMAIN
    canonical_offset_us: float | None = None
    canonical_offset_projection_success: bool | None = None
    alignment_signal_source: str | None = None
    feature_schema: str | None = None
    feature_values_sha256: str | None = None
    feature_metadata_sha256: str | None = None
    feature_sampling_rate_hz: float | None = None
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

    @property
    def boundary_offset_us(self) -> float:
        """Return the Board-to-Ring offset in the artifact's declared domain."""

        if self.offset_domain == ALIGNMENT_WORK_AXIS_DOMAIN:
            if self.work_axis_offset_us is None:
                raise AlignmentOffsetExportError(
                    "alignment_work_axis offset is missing work_axis_offset_us"
                )
            return self.work_axis_offset_us
        return self.offset_us


def extract_alignment_offset(
    result: SequenceAlignmentResult,
    *,
    user: str,
    action: str,
    dataset_id: int,
    ring_stream: str = "ring_0",
    projection: AlignmentOffsetProjection | None = None,
    projection_diagnostics: AlignmentOffsetProjection | None = None,
) -> AlignmentOffset:
    """Extract a successful alignment with the work-axis offset as primary.

    Results carrying an alignment-time-axis report are the schema-v2 source
    aware form.  Their ``offset_us`` always remains the work-axis offset;
    canonical projection is optional diagnostic provenance.  Results without
    that report retain the old canonical-domain export behavior.
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
    source_aware = (
        report.get("offset_domain") == ALIGNMENT_WORK_AXIS_DOMAIN
        or "alignment_time_axis" in report
    )
    if projection is None:
        projection_fields: dict[str, object] = {}
        if projection_diagnostics is not None:
            _validate_projection(projection_diagnostics)
            if not math.isclose(
                projection_diagnostics.work_axis_offset_us,
                work_axis_offset_us,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise AlignmentOffsetExportError(
                    "projection diagnostics do not match alignment result"
                )
            projection_fields = {
                "work_axis_offset_us": projection_diagnostics.work_axis_offset_us,
                "projection_delta_us": projection_diagnostics.projection_delta_us,
                "projection_delta_median_us": projection_diagnostics.projection_delta_median_us,
                "projection_delta_mad_us": projection_diagnostics.projection_delta_mad_us,
                "projection_delta_min_us": projection_diagnostics.projection_delta_min_us,
                "projection_delta_max_us": projection_diagnostics.projection_delta_max_us,
                "projection_delta_range_us": projection_diagnostics.projection_delta_range_us,
                "projection_contributing_match_count": (
                    projection_diagnostics.contributing_match_count
                ),
            }
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
    if source_aware:
        offset_us = work_axis_offset_us
        offset_domain = ALIGNMENT_WORK_AXIS_DOMAIN
        offset_schema_version: int | None = ALIGNMENT_OFFSET_SCHEMA_VERSION
        canonical_offset_us = (
            None if projection is None else projection.canonical_offset_us
        )
        canonical_projection_success: bool | None = projection is not None
        projection_fields.setdefault("work_axis_offset_us", work_axis_offset_us)
    else:
        # Preserve the old API for callers that construct a bare alignment
        # result without source-aware axis metadata.
        offset_us = (
            work_axis_offset_us
            if projection is None
            else projection.canonical_offset_us
        )
        offset_domain = CANONICAL_TIMESTAMP_DOMAIN
        offset_schema_version = None
        canonical_offset_us = None
        canonical_projection_success = None
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
        offset_schema_version=offset_schema_version,
        offset_domain=offset_domain,
        canonical_offset_us=canonical_offset_us,
        canonical_offset_projection_success=canonical_projection_success,
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
        feature_sampling_rate_hz=_optional_finite_float(
            report.get("feature_sampling_rate_hz"),
            name="feature_sampling_rate_hz",
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
    schema_version = _optional_nonnegative_int(
        fields.get("offset_schema_version"), name="offset_schema_version"
    )
    if schema_version is not None and schema_version != ALIGNMENT_OFFSET_SCHEMA_VERSION:
        raise AlignmentOffsetExportError(
            "offset TXT has an unsupported offset_schema_version"
        )
    if schema_version is None:
        # Files predating the explicit domain field used canonical timestamps.
        offset_domain = CANONICAL_TIMESTAMP_DOMAIN
        canonical_offset_us = None
        canonical_projection_success = None
    else:
        if "offset_domain" not in fields:
            raise AlignmentOffsetExportError(
                "schema-v2 offset TXT must declare offset_domain"
            )
        offset_domain = _offset_domain(fields["offset_domain"])
        canonical_offset_us = _optional_finite_float(
            fields.get("canonical_offset_us"), name="canonical_offset_us"
        )
        canonical_projection_success = _optional_bool(
            fields.get("canonical_offset_projection_success"),
            name="canonical_offset_projection_success",
        )
        if canonical_projection_success is None:
            raise AlignmentOffsetExportError(
                "schema-v2 offset TXT must declare canonical_offset_projection_success"
            )
    offset = AlignmentOffset(
        user=fields["user"],
        action=fields["action"],
        dataset_id=_nonnegative_int(fields["dataset_id"], name="dataset_id"),
        ring_stream=fields["ring_stream"],
        offset_us=_finite_float(fields["offset_us"], name="offset_us"),
        alignment_model=fields["alignment_model"],
        alignment_success=True,
        offset_schema_version=schema_version,
        offset_domain=offset_domain,
        canonical_offset_us=canonical_offset_us,
        canonical_offset_projection_success=canonical_projection_success,
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
        feature_sampling_rate_hz=_optional_finite_float(
            fields.get("feature_sampling_rate_hz"),
            name="feature_sampling_rate_hz",
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
    if (
        offset.feature_sampling_rate_hz is not None
        and not math.isclose(
            offset.feature_sampling_rate_hz,
            feature_input.sampling_rate_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise AlignmentOffsetExportError(
            "alignment offset feature sampling rate does not match feature input"
        )
    if offset.timestamp_sha256 != feature_input.timestamps_sha256:
        raise AlignmentOffsetExportError(
            "alignment offset timestamp SHA-256 does not match feature input"
        )
    if (
        offset.timestamp_source_sha256 is not None
        and offset.timestamp_source_sha256 != feature_input.timestamps_sha256
    ):
        raise AlignmentOffsetExportError(
            "alignment offset timestamp source SHA-256 does not match feature input"
        )
    if (
        offset.alignment_time_axis_sample_count is not None
        and offset.alignment_time_axis_sample_count != feature_input.sample_count
    ):
        raise AlignmentOffsetExportError(
            "alignment offset alignment-axis sample count does not match feature input"
        )
    if offset.alignment_time_axis_strategy is not None:
        if offset.alignment_time_axis_strategy != "endpoint_reconstruction":
            raise AlignmentOffsetExportError(
                "alignment offset alignment-axis strategy is unsupported"
            )
        expected_axes = build_alignment_time_axes(
            feature_input.timestamps_us,
            input_kind=feature_input.input_kind,
            sampling_rate_hz=feature_input.sampling_rate_hz,
        )
        if offset.alignment_time_axis_sampling_rate_hz is None or not math.isclose(
            offset.alignment_time_axis_sampling_rate_hz,
            expected_axes.alignment_sampling_rate_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AlignmentOffsetExportError(
                "alignment offset alignment-axis sampling rate does not match feature timestamps"
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
    feature_sampling_rate_line = (
        ""
        if offset.feature_sampling_rate_hz is None
        else f"feature_sampling_rate_hz={offset.feature_sampling_rate_hz:.17g}\n"
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
            f"{offset.alignment_time_axis_sampling_rate_hz:.17g}\n"
            "alignment_time_axis_sample_count="
            f"{offset.alignment_time_axis_sample_count}\n"
            "canonical_timestamps_modified="
            f"{'true' if offset.canonical_timestamps_modified else 'false'}\n"
        )
    schema = ""
    if offset.offset_schema_version is not None:
        assert offset.canonical_offset_projection_success is not None
        canonical_offset = (
            ""
            if offset.canonical_offset_us is None
            else f"{offset.canonical_offset_us:.17g}"
        )
        schema = (
            f"offset_schema_version={offset.offset_schema_version}\n"
            f"offset_domain={offset.offset_domain}\n"
            f"canonical_offset_us={canonical_offset}\n"
            "canonical_offset_projection_success="
            f"{'true' if offset.canonical_offset_projection_success else 'false'}\n"
        )
    projection = ""
    if offset.work_axis_offset_us is not None:
        work_axis_line = f"work_axis_offset_us={offset.work_axis_offset_us:.17g}\n"
        diagnostics = (
            offset.projection_delta_us,
            offset.projection_delta_median_us,
            offset.projection_delta_mad_us,
            offset.projection_delta_min_us,
            offset.projection_delta_max_us,
            offset.projection_delta_range_us,
            offset.projection_contributing_match_count,
        )
        if any(value is not None for value in diagnostics):
            if any(value is None for value in diagnostics):
                raise AlignmentOffsetExportError(
                    "canonical offset projection is incomplete"
                )
            projection = (
                work_axis_line
                + f"projection_delta_us={offset.projection_delta_us:.6f}\n"
                + "projection_delta_median_us="
                + f"{offset.projection_delta_median_us:.6f}\n"
                + "projection_delta_mad_us="
                + f"{offset.projection_delta_mad_us:.6f}\n"
                + "projection_delta_min_us="
                + f"{offset.projection_delta_min_us:.6f}\n"
                + "projection_delta_max_us="
                + f"{offset.projection_delta_max_us:.6f}\n"
                + "projection_delta_range_us="
                + f"{offset.projection_delta_range_us:.6f}\n"
                + "projection_contributing_match_count="
                + f"{offset.projection_contributing_match_count}\n"
            )
        else:
            projection = work_axis_line
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
        f"{feature_sampling_rate_line}"
        f"{schema}"
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
    if offset.offset_schema_version is not None:
        schema_version = _nonnegative_int(
            offset.offset_schema_version, name="offset_schema_version"
        )
        if schema_version != ALIGNMENT_OFFSET_SCHEMA_VERSION:
            raise AlignmentOffsetExportError(
                "offset_schema_version is unsupported"
            )
        domain = _offset_domain(offset.offset_domain)
        if offset.canonical_offset_projection_success is None:
            raise AlignmentOffsetExportError(
                "schema-v2 offset must declare canonical_offset_projection_success"
            )
        if not isinstance(offset.canonical_offset_projection_success, bool):
            raise AlignmentOffsetExportError(
                "canonical_offset_projection_success must be a boolean"
            )
        if domain == ALIGNMENT_WORK_AXIS_DOMAIN:
            if offset.work_axis_offset_us is None:
                raise AlignmentOffsetExportError(
                    "alignment_work_axis offset requires work_axis_offset_us"
                )
            work_offset = _finite_float(
                offset.work_axis_offset_us, name="work_axis_offset_us"
            )
            if not math.isclose(work_offset, offset.offset_us, abs_tol=1e-6):
                raise AlignmentOffsetExportError(
                    "alignment_work_axis offset_us must equal work_axis_offset_us"
                )
            if offset.canonical_offset_projection_success:
                if offset.canonical_offset_us is None:
                    raise AlignmentOffsetExportError(
                        "successful canonical projection requires canonical_offset_us"
                    )
            elif offset.canonical_offset_us is not None:
                raise AlignmentOffsetExportError(
                    "failed canonical projection must not publish canonical_offset_us"
                )
        elif offset.canonical_offset_us is not None and not math.isclose(
            offset.canonical_offset_us, offset.offset_us, abs_tol=1e-6
        ):
            raise AlignmentOffsetExportError(
                "canonical_timestamp canonical_offset_us must equal offset_us"
            )
    else:
        _offset_domain(offset.offset_domain)
    feature_sampling_rate = _optional_finite_float(
        offset.feature_sampling_rate_hz,
        name="feature_sampling_rate_hz",
    )
    if feature_sampling_rate is not None and feature_sampling_rate <= 0.0:
        raise AlignmentOffsetExportError("feature_sampling_rate_hz must be positive")
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
        allowed_strategies = {"endpoint_reconstruction"}
        if offset.offset_schema_version is None:
            allowed_strategies.update(
                {"canonical_strict_copy", "strict_reconstruction", "raw_endpoint_reconstruction"}
            )
        if offset.alignment_time_axis_strategy not in allowed_strategies:
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
    projection_diagnostics = (
        offset.projection_delta_us,
        offset.projection_delta_median_us,
        offset.projection_delta_mad_us,
        offset.projection_delta_min_us,
        offset.projection_delta_max_us,
        offset.projection_delta_range_us,
        offset.projection_contributing_match_count,
    )
    if any(value is not None for value in projection_diagnostics):
        if offset.work_axis_offset_us is None or any(
            value is None for value in projection_diagnostics
        ):
            raise AlignmentOffsetExportError(
                "canonical offset projection is incomplete"
            )
        canonical_offset = (
            offset.canonical_offset_us
            if offset.offset_schema_version is not None
            and offset.canonical_offset_us is not None
            else (
                offset.work_axis_offset_us + offset.projection_delta_us
                if offset.offset_schema_version is not None
                and offset.work_axis_offset_us is not None
                and offset.projection_delta_us is not None
                else offset.offset_us
            )
        )
        _validate_projection(
            AlignmentOffsetProjection(
                work_axis_offset_us=offset.work_axis_offset_us,  # type: ignore[arg-type]
                canonical_offset_us=canonical_offset,
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
    except (TypeError, ValueError, OverflowError) as error:
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


def _offset_domain(value: object) -> str:
    if not isinstance(value, str) or value not in _OFFSET_DOMAINS:
        raise AlignmentOffsetExportError(
            "offset_domain must be alignment_work_axis or canonical_timestamp"
        )
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


# ---------------------------------------------------------------------------
# Completed alignment outcomes
# ---------------------------------------------------------------------------

_SKIP_DIAGNOSTIC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "previous_global_frame_index",
        "next_global_frame_index",
        "previous_timestamp_raw",
        "next_timestamp_raw",
        "prefix_boundary_position",
        "last_pre_jump_global_frame_index",
        "total_global_valid_pair_count",
        "usable_prefix_valid_pair_count",
    }
)


@dataclass(frozen=True, slots=True)
class AlignmentInputProvenance:
    """The current feature input identity and content hashes for one run."""

    input_kind: str
    user: str
    action: str
    dataset_id: int
    feature_schema: str | None = None
    sampling_rate_hz: float | None = None
    feature_values_sha256: str | None = None
    feature_metadata_sha256: str | None = None
    timestamp_sha256: str | None = None
    ring_0_sha256: str | None = None

    @property
    def canonical_timestamps_sha256(self) -> str | None:
        """Return the canonical timestamp-array digest under its long name."""

        return self.timestamp_sha256

    @property
    def values_sha256(self) -> str | None:
        """Compatibility alias for the existing feature provenance spelling."""

        return self.feature_values_sha256

    @property
    def metadata_sha256(self) -> str | None:
        """Compatibility alias for the existing feature provenance spelling."""

        return self.feature_metadata_sha256

    @property
    def recording(self) -> dict[str, object]:
        return {
            "user": self.user,
            "action": self.action,
            "dataset_id": self.dataset_id,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation used by outcome artifacts."""

        _validate_input_provenance(self)
        payload: dict[str, object] = {
            "input_kind": self.input_kind,
            "recording": self.recording,
        }
        if self.input_kind == "raw-ring":
            payload.update(
                {
                    "ring_0_sha256": self.ring_0_sha256,
                    "timestamp_sha256": self.timestamp_sha256,
                    "canonical_timestamps_sha256": self.timestamp_sha256,
                }
            )
        else:
            payload.update(
                {
                    "feature_schema": self.feature_schema,
                    "sampling_rate_hz": self.sampling_rate_hz,
                    "feature_values_sha256": self.feature_values_sha256,
                    "feature_metadata_sha256": self.feature_metadata_sha256,
                    "timestamp_sha256": self.timestamp_sha256,
                    "canonical_timestamps_sha256": self.timestamp_sha256,
                }
            )
        return payload


# The longer spelling is useful to downstream callers and keeps the public
# contract discoverable without making callers depend on the implementation's
# historical ``feature`` terminology.
AlignmentFeatureProvenance = AlignmentInputProvenance
CurrentAlignmentInputProvenance = AlignmentInputProvenance


@dataclass(frozen=True, slots=True)
class BoardChunkProvenance:
    """Digest and numeric identity of one Board chunk in loader order."""

    chunk_index: int
    sha256: str
    path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        _nonnegative_int(self.chunk_index, name="board chunk_index")
        digest = _strict_sha256(self.sha256, name="board chunk sha256")
        return {"chunk_index": self.chunk_index, "sha256": digest}


@dataclass(frozen=True, slots=True)
class AlignmentOutcomePaths:
    """Stable sibling paths for TXT/skip, verification, and final report."""

    offset_txt_path: Path
    skip_json_path: Path
    verification_png_path: Path
    report_path: Path

    @property
    def offset_path(self) -> Path:
        return self.offset_txt_path

    @property
    def skip_path(self) -> Path:
        return self.skip_json_path

    @property
    def verification_path(self) -> Path:
        return self.verification_png_path

    @property
    def offset_txt(self) -> Path:
        return self.offset_txt_path

    @property
    def skip_json(self) -> Path:
        return self.skip_json_path

    @property
    def verification_png(self) -> Path:
        return self.verification_png_path

    @property
    def report(self) -> Path:
        return self.report_path


AlignmentOutputPaths = AlignmentOutcomePaths
AlignmentOutcomeValidationError = AlignmentOutcomeError
AlignmentProvenance = AlignmentInputProvenance


@dataclass(frozen=True, slots=True)
class AlignmentSkipArtifact:
    """Validated representation of the only publishable alignment skip."""

    recording: Mapping[str, object]
    diagnostics: Mapping[str, object]
    input_provenance: AlignmentInputProvenance | Mapping[str, object]
    board_provenance: Sequence[BoardChunkProvenance | Mapping[str, object]]
    reason: str = ALIGNMENT_SKIP_REASON
    alignment_skip_schema_version: int = ALIGNMENT_SKIP_SCHEMA_VERSION
    artifact_kind: str = "alignment_skip"

    def __post_init__(self) -> None:
        _literal_schema_version(
            self.alignment_skip_schema_version,
            name="alignment_skip_schema_version",
        )

    @property
    def schema_version(self) -> int:
        return self.alignment_skip_schema_version

    @_normalize_outcome_errors
    def to_dict(self) -> dict[str, object]:
        recording = _validated_recording_mapping(self.recording)
        diagnostics = _validate_skip_diagnostics(self.diagnostics)
        input_payload = _coerce_input_provenance(self.input_provenance).to_dict()
        board_payload = _coerce_board_provenance(self.board_provenance)
        _literal_schema_version(
            self.alignment_skip_schema_version,
            name="alignment_skip_schema_version",
        )
        if self.artifact_kind != "alignment_skip":
            raise AlignmentOutcomeError("alignment skip artifact_kind is invalid")
        if self.reason != ALIGNMENT_SKIP_REASON:
            raise AlignmentOutcomeError("alignment skip reason is unsupported")
        return {
            "alignment_skip_schema_version": self.alignment_skip_schema_version,
            "artifact_kind": self.artifact_kind,
            "recording": recording,
            "reason": self.reason,
            "diagnostics": diagnostics,
            "input_provenance": input_payload,
            "board_provenance": board_payload,
        }

    @classmethod
    @_normalize_outcome_errors
    def from_dict(cls, payload: Mapping[str, object]) -> "AlignmentSkipArtifact":
        if not isinstance(payload, Mapping):
            raise AlignmentOutcomeError("alignment skip artifact must be a JSON object")
        required = {
            "alignment_skip_schema_version",
            "artifact_kind",
            "recording",
            "reason",
            "diagnostics",
            "input_provenance",
            "board_provenance",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise AlignmentOutcomeError(
                "alignment skip artifact is missing required field(s): "
                + ", ".join(missing)
            )
        return cls(
            recording=_validated_recording_mapping(payload["recording"]),
            diagnostics=_validate_skip_diagnostics(payload["diagnostics"]),
            input_provenance=_coerce_input_provenance(payload["input_provenance"]),
            board_provenance=_coerce_board_provenance(payload["board_provenance"]),
            reason=payload["reason"],  # type: ignore[arg-type]
            alignment_skip_schema_version=_literal_schema_version(
                payload["alignment_skip_schema_version"],
                name="alignment_skip_schema_version",
            ),
            artifact_kind=payload["artifact_kind"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class AlignmentOutcome:
    """A completed SUCCESS or SKIPPED artifact set and its final report."""

    status: AlignmentOutcomeStatus
    recording: Mapping[str, object]
    paths: AlignmentOutcomePaths
    report: Mapping[str, object]

    @property
    def alignment_status(self) -> str:
        return self.status.value

    @property
    def status_name(self) -> str:
        """Return the terminal-state spelling used in task summaries."""

        return self.status.name

    @property
    def is_success(self) -> bool:
        return self.status is AlignmentOutcomeStatus.SUCCESS

    @property
    def is_skipped(self) -> bool:
        return self.status is AlignmentOutcomeStatus.SKIPPED


def build_alignment_input_provenance(
    feature_input: object | None = None,
    *,
    input_kind: str | None = None,
    user: str | None = None,
    action: str | None = None,
    dataset_id: int | None = None,
    ring_0_sha256: str | None = None,
    ring_0_path: Path | None = None,
    canonical_timestamps_sha256: str | None = None,
    canonical_timestamps_us: np.ndarray | None = None,
    timestamp_sha256: str | None = None,
    feature_schema: str | None = None,
    sampling_rate_hz: float | None = None,
    feature_values_sha256: str | None = None,
    feature_metadata_sha256: str | None = None,
) -> AlignmentInputProvenance:
    """Build current provenance from a feature input or explicit run values."""

    if feature_input is not None:
        if isinstance(feature_input, AlignmentInputProvenance):
            return _coerce_input_provenance(feature_input)
        if isinstance(feature_input, Mapping):
            return _coerce_input_provenance(feature_input)
        required_attrs = (
            "input_kind",
            "user",
            "action",
            "dataset_id",
            "feature_schema",
            "sampling_rate_hz",
            "values_sha256",
            "metadata_sha256",
            "timestamps_sha256",
        )
        if not all(hasattr(feature_input, name) for name in required_attrs):
            raise AlignmentOutcomeError(
                "feature_input must expose the RecordingFeatureInput provenance fields"
            )
        return _coerce_input_provenance(
            {
                "input_kind": feature_input.input_kind,
                "recording": {
                    "user": feature_input.user,
                    "action": feature_input.action,
                    "dataset_id": feature_input.dataset_id,
                },
                "feature_schema": feature_input.feature_schema,
                "sampling_rate_hz": feature_input.sampling_rate_hz,
                "feature_values_sha256": feature_input.values_sha256,
                "feature_metadata_sha256": feature_input.metadata_sha256,
                "timestamp_sha256": feature_input.timestamps_sha256,
                "ring_0_sha256": (
                    feature_input.values_sha256
                    if feature_input.input_kind == "raw-ring"
                    else None
                ),
            }
        )
    if input_kind is None or user is None or action is None or dataset_id is None:
        raise AlignmentOutcomeError(
            "input_kind, user, action, and dataset_id are required for provenance"
        )
    if ring_0_sha256 is None and ring_0_path is not None:
        ring_0_sha256 = _sha256_regular_file(Path(ring_0_path), name="ring_0")
    timestamp_digest = timestamp_sha256 or canonical_timestamps_sha256
    if timestamp_digest is None and canonical_timestamps_us is not None:
        timestamp_digest = sha256_array(canonical_timestamps_us)
    if timestamp_digest is None:
        raise AlignmentOutcomeError(
            "canonical timestamp SHA-256 is required for input provenance"
        )
    return _coerce_input_provenance(
        {
            "input_kind": input_kind,
            "recording": {
                "user": user,
                "action": action,
                "dataset_id": dataset_id,
            },
            "feature_schema": feature_schema,
            "sampling_rate_hz": sampling_rate_hz,
            "feature_values_sha256": feature_values_sha256,
            "feature_metadata_sha256": feature_metadata_sha256,
            "timestamp_sha256": timestamp_digest,
            "ring_0_sha256": ring_0_sha256,
        }
    )


current_alignment_input_provenance = build_alignment_input_provenance


def build_board_chunk_provenance(
    source: object,
) -> tuple[BoardChunkProvenance, ...]:
    """Hash Board files in their existing numeric chunk order."""

    paths: tuple[Path, ...]
    reports: Sequence[object] | None = None
    if hasattr(source, "chunk_paths"):
        paths = tuple(Path(path) for path in source.chunk_paths)
        reports = getattr(source, "chunk_reports", None)
    elif isinstance(source, (str, Path)):
        paths = (Path(source),)
    else:
        try:
            paths = tuple(Path(path) for path in source)  # type: ignore[arg-type]
        except TypeError as error:
            raise AlignmentOutcomeError(
                "Board provenance source must be BoardData or a path sequence"
            ) from error
    if not paths:
        raise AlignmentOutcomeError("Board provenance must contain at least one chunk")
    values: list[BoardChunkProvenance] = []
    for position, path in enumerate(paths):
        chunk_index: int | None = None
        if reports is not None and position < len(reports):
            report = reports[position]
            candidate = getattr(report, "chunk_index", None)
            if candidate is not None:
                chunk_index = _nonnegative_int(candidate, name="board chunk_index")
        if chunk_index is None:
            try:
                from writingring.discovery import parse_recording_filename

                parsed = parse_recording_filename(path.name)
            except (ImportError, ValueError) as error:
                raise AlignmentOutcomeError(
                    f"could not parse Board chunk identity: {path}"
                ) from error
            chunk_index = (
                None if parsed is None else getattr(parsed, "chunk_index", None)
            )
            if chunk_index is None:
                raise AlignmentOutcomeError(
                    f"Board chunk path has no numeric chunk index: {path}"
                )
            chunk_index = _nonnegative_int(chunk_index, name="board chunk_index")
        digest = _sha256_regular_file(path, name="Board chunk")
        values.append(BoardChunkProvenance(chunk_index, digest, path))
    indices = [item.chunk_index for item in values]
    if any(next_index < index for index, next_index in zip(indices, indices[1:])):
        raise AlignmentOutcomeError(
            "Board provenance chunks are not in numeric chunk-index order"
        )
    return tuple(values)


build_board_provenance = build_board_chunk_provenance


def build_alignment_skip_path(
    offset_root: Path,
    *,
    user: str,
    action: str,
    dataset_id: int,
) -> Path:
    """Return the skip JSON sibling of the success offset TXT."""

    _validate_identity(user=user, action=action, dataset_id=dataset_id)
    return Path(offset_root) / user / f"action_{action}" / (
        f"{dataset_id}_ring_board_skip.json"
    )


def build_alignment_outcome_paths(
    offset_root: Path,
    verification_root: Path,
    report_root: Path,
    *,
    user: str,
    action: str,
    dataset_id: int,
) -> AlignmentOutcomePaths:
    """Return the complete sibling artifact namespace for one recording."""

    offset = build_alignment_offset_path(
        offset_root, user=user, action=action, dataset_id=dataset_id
    )
    return AlignmentOutcomePaths(
        offset_txt_path=offset,
        skip_json_path=build_alignment_skip_path(
            offset_root, user=user, action=action, dataset_id=dataset_id
        ),
        verification_png_path=(
            Path(verification_root)
            / user
            / f"action_{action}"
            / f"{dataset_id}_alignment_verification.png"
        ),
        report_path=_build_alignment_report_path(
            report_root, user=user, action=action, dataset_id=dataset_id
        ),
    )


def _build_alignment_report_path(
    report_root: Path,
    *,
    user: str,
    action: str,
    dataset_id: int,
) -> Path:
    _validate_identity(user=user, action=action, dataset_id=dataset_id)
    return Path(report_root) / user / f"action_{action}" / (
        f"{dataset_id}_alignment_report.json"
    )


@_normalize_outcome_errors
def make_alignment_skip_artifact(
    error: object,
    *,
    user: str,
    action: str,
    dataset_id: int,
    input_provenance: object,
    board_provenance: object,
) -> AlignmentSkipArtifact:
    """Convert T1's typed diagnostic into the strict skip artifact model."""

    from writingring.event_alignment import InitialIntervalNoUsablePairError

    if not isinstance(error, InitialIntervalNoUsablePairError):
        raise AlignmentOutcomeError(
            "only InitialIntervalNoUsablePairError can create an alignment skip"
        )
    _validate_identity(user=user, action=action, dataset_id=dataset_id)
    return AlignmentSkipArtifact(
        recording={"user": user, "action": action, "dataset_id": dataset_id},
        diagnostics=dict(error.diagnostics),
        input_provenance=_coerce_input_provenance(input_provenance),
        board_provenance=_coerce_board_provenance(board_provenance),
    )


build_alignment_skip_artifact = make_alignment_skip_artifact


@_normalize_outcome_errors
def write_alignment_skip_artifact(
    artifact: AlignmentSkipArtifact | Mapping[str, object],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one strict, validated skip JSON artifact."""

    value = (
        artifact
        if isinstance(artifact, AlignmentSkipArtifact)
        else AlignmentSkipArtifact.from_dict(artifact)
    )
    payload = value.to_dict()
    path = Path(output_path)
    _preflight_single_target(path, overwrite=overwrite, label="skip artifact")
    _atomic_write_json(path, payload)
    return path


write_alignment_skip_json = write_alignment_skip_artifact


@_normalize_outcome_errors
def read_alignment_skip_artifact(
    path: Path,
    *,
    expected_user: str | None = None,
    expected_action: str | None = None,
    expected_dataset_id: int | None = None,
    expected_input_provenance: object | None = None,
    expected_board_provenance: object | None = None,
) -> AlignmentSkipArtifact:
    """Read the strict skip artifact and optionally verify current provenance."""

    payload = _read_strict_json(Path(path), label="alignment skip artifact")
    artifact = AlignmentSkipArtifact.from_dict(payload)
    artifact_payload = artifact.to_dict()
    identity = _expected_identity(
        expected_user=expected_user,
        expected_action=expected_action,
        expected_dataset_id=expected_dataset_id,
    )
    if identity is not None and artifact_payload["recording"] != identity:
        raise AlignmentOutcomeError(
            "alignment skip recording identity does not match the requested recording"
        )
    if expected_input_provenance is not None:
        if artifact_payload["input_provenance"] != _coerce_input_provenance(
            expected_input_provenance
        ).to_dict():
            raise AlignmentOutcomeError("alignment skip input provenance is stale")
    if expected_board_provenance is not None:
        if artifact_payload["board_provenance"] != _coerce_board_provenance(
            expected_board_provenance
        ):
            raise AlignmentOutcomeError("alignment skip Board provenance is stale")
    return artifact


validate_alignment_skip_artifact = read_alignment_skip_artifact


@_normalize_outcome_errors
def validate_alignment_outcome(
    paths_or_offset_root: AlignmentOutcomePaths | Path,
    verification_root: Path | None = None,
    report_root: Path | None = None,
    *,
    expected_user: str | None = None,
    expected_action: str | None = None,
    expected_dataset_id: int | None = None,
    expected_recording: Mapping[str, object] | None = None,
    input_provenance: object | None = None,
    expected_input_provenance: object | None = None,
    board_provenance: object | None = None,
    expected_board_provenance: object | None = None,
) -> AlignmentOutcome:
    """Validate one complete SUCCESS or SKIPPED outcome.

    The report is authoritative: a TXT/PNG or skip JSON without a matching
    v1 report manifest is never returned as a completed outcome.
    """

    paths = _coerce_outcome_paths(
        paths_or_offset_root,
        verification_root=verification_root,
        report_root=report_root,
        expected_user=expected_user,
        expected_action=expected_action,
        expected_dataset_id=expected_dataset_id,
        expected_recording=expected_recording,
    )
    identity = _expected_identity(
        expected_user=expected_user,
        expected_action=expected_action,
        expected_dataset_id=expected_dataset_id,
        expected_recording=expected_recording,
    )
    if identity is None:
        raise AlignmentOutcomeError(
            "expected recording identity is required for outcome validation"
        )
    current_input = expected_input_provenance or input_provenance
    current_board = expected_board_provenance or board_provenance
    if current_input is None or current_board is None:
        raise AlignmentOutcomeError(
            "current input and Board provenance are required for outcome validation"
        )
    expected_input = _coerce_input_provenance(current_input).to_dict()
    if expected_input.get("recording") != identity:
        raise AlignmentOutcomeError(
            "current input provenance identity does not match expected recording"
        )
    expected_board = _coerce_board_provenance(current_board)

    report = _read_strict_json(paths.report_path, label="alignment outcome report")
    _validate_report_header(report, identity=identity)
    status = _coerce_status(report["alignment_status"])
    if status is AlignmentOutcomeStatus.FAILED:
        raise AlignmentOutcomeError("failed alignment reports are not completed outcomes")
    if report.get("input_provenance") != expected_input:
        raise AlignmentOutcomeError("alignment report input provenance is stale")
    if report.get("board_provenance") != expected_board:
        raise AlignmentOutcomeError("alignment report Board provenance is stale")

    # ``get`` is intentional: a missing manifest is a malformed outcome, not
    # an implementation-level ``KeyError``.
    entries = _validate_manifest(report.get("outcome_artifacts"))
    if status is AlignmentOutcomeStatus.SUCCESS:
        if report.get("alignment_success") is not True:
            raise AlignmentOutcomeError(
                "status-success report must declare alignment_success=true"
            )
        if report.get("work_axis_alignment_success") is not True:
            raise AlignmentOutcomeError(
                "status-success report must declare work_axis_alignment_success=true"
            )
        if paths.skip_json_path.exists():
            raise AlignmentOutcomeError(
                "success outcome conflicts with an alignment skip artifact"
            )
        expected_names = {
            paths.offset_txt_path.name,
            paths.verification_png_path.name,
        }
        if {entry["filename"] for entry in entries} != expected_names:
            raise AlignmentOutcomeError(
                "status-success report must manifest exactly offset TXT and verification PNG"
            )
        _validate_manifest_file(
            paths.offset_txt_path,
            entries,
            label="alignment offset TXT",
        )
        offset = read_alignment_offset_txt(
            paths.offset_txt_path,
            expected_user=identity["user"],
            expected_action=identity["action"],
            expected_dataset_id=identity["dataset_id"],
        )
        _validate_offset_input_provenance(offset, expected_input)
        if not paths.verification_png_path.is_file():
            raise AlignmentOutcomeError(
                "status-success verification PNG is missing or not a regular file"
            )
        if paths.verification_png_path.stat().st_size <= 0:
            raise AlignmentOutcomeError("status-success verification PNG is empty")
        _validate_manifest_file(
            paths.verification_png_path,
            entries,
            label="alignment verification PNG",
        )
    else:
        if report.get("alignment_success") is not False:
            raise AlignmentOutcomeError(
                "status-skipped report must declare alignment_success=false"
            )
        if report.get("work_axis_alignment_success") is not False:
            raise AlignmentOutcomeError(
                "status-skipped report must declare work_axis_alignment_success=false"
            )
        if paths.offset_txt_path.exists() or paths.verification_png_path.exists():
            raise AlignmentOutcomeError(
                "skipped outcome conflicts with success artifacts"
            )
        if [entry["filename"] for entry in entries] != [paths.skip_json_path.name]:
            raise AlignmentOutcomeError(
                "status-skipped report must manifest exactly one skip JSON"
            )
        _validate_manifest_file(
            paths.skip_json_path,
            entries,
            label="alignment skip JSON",
        )
        skip = read_alignment_skip_artifact(
            paths.skip_json_path,
            expected_user=identity["user"],
            expected_action=identity["action"],
            expected_dataset_id=identity["dataset_id"],
            expected_input_provenance=expected_input,
            expected_board_provenance=expected_board,
        )
        if skip.reason != ALIGNMENT_SKIP_REASON:
            raise AlignmentOutcomeError("alignment skip reason is unsupported")
    return AlignmentOutcome(
        status=status,
        recording=identity,
        paths=paths,
        report=report,
    )


validate_completed_alignment_outcome = validate_alignment_outcome
validate_completed_alignment = validate_alignment_outcome
validate_alignment_completion = validate_alignment_outcome
read_alignment_outcome = validate_alignment_outcome


@_normalize_outcome_errors
def read_alignment_outcome_report(path: Path) -> dict[str, object]:
    """Read a v1 report manifest without treating it as completion."""

    payload = _read_strict_json(Path(path), label="alignment outcome report")
    _validate_manifest(payload.get("outcome_artifacts"))
    return payload


@_normalize_outcome_errors
def publish_alignment_outcome(
    paths: AlignmentOutcomePaths,
    *,
    status: AlignmentOutcomeStatus | str,
    report: Mapping[str, object],
    offset: AlignmentOffset | None = None,
    skip_artifact: AlignmentSkipArtifact | Mapping[str, object] | None = None,
    verification_path: Path | None = None,
    overwrite_offset: bool = False,
    overwrite_skip: bool | None = None,
    overwrite_verification: bool = False,
    overwrite_report: bool = False,
    overwrite_outcome: bool = False,
) -> AlignmentOutcome:
    """Preflight, stage, and atomically publish one alignment outcome.

    Individual artifacts are replaced before the strict report manifest.  A
    success/skip transition additionally requires ``overwrite_outcome`` and
    removes obsolete opposite-state files before publishing the new report.
    """

    if not isinstance(paths, AlignmentOutcomePaths):
        raise AlignmentOutcomeError("paths must be AlignmentOutcomePaths")
    outcome_status = _coerce_status(status)
    if not isinstance(report, Mapping):
        raise AlignmentOutcomeError("alignment outcome report must be a mapping")
    if not isinstance(overwrite_outcome, bool):
        raise AlignmentOutcomeError("overwrite_outcome must be a boolean")
    if overwrite_skip is not None:
        if not isinstance(overwrite_skip, bool):
            raise AlignmentOutcomeError("overwrite_skip must be a boolean")
        overwrite_offset = overwrite_skip
    report_base = dict(report)
    completed = outcome_status in {
        AlignmentOutcomeStatus.SUCCESS,
        AlignmentOutcomeStatus.SKIPPED,
    }
    if completed and not report_base.get("input_provenance"):
        raise AlignmentOutcomeError(
            "completed alignment outcome requires current input provenance"
        )
    if "input_provenance" in report_base:
        report_base["input_provenance"] = _coerce_input_provenance(
            report_base["input_provenance"]
        ).to_dict()
    if completed and not report_base.get("board_provenance"):
        raise AlignmentOutcomeError(
            "completed alignment outcome requires current Board provenance"
        )
    if "board_provenance" in report_base:
        report_base["board_provenance"] = _coerce_board_provenance(
            report_base["board_provenance"]
        )
    identity = _report_identity(report_base)
    if identity is None:
        if offset is not None:
            identity = {
                "user": offset.user,
                "action": offset.action,
                "dataset_id": offset.dataset_id,
            }
            report_base["recording"] = identity
        elif skip_artifact is not None:
            candidate = (
                skip_artifact
                if isinstance(skip_artifact, AlignmentSkipArtifact)
                else AlignmentSkipArtifact.from_dict(skip_artifact)
            )
            identity = _validated_recording_mapping(candidate.recording)
            report_base["recording"] = identity
        else:
            raise AlignmentOutcomeError(
                "outcome report must declare recording identity"
            )
    else:
        identity = _validated_recording_mapping(identity)

    if completed:
        if report_base["input_provenance"]["recording"] != identity:  # type: ignore[index]
            raise AlignmentOutcomeError(
                "completed outcome input provenance identity does not match report"
            )

    skip_value: AlignmentSkipArtifact | None = None
    if skip_artifact is not None:
        skip_value = (
            skip_artifact
            if isinstance(skip_artifact, AlignmentSkipArtifact)
            else AlignmentSkipArtifact.from_dict(skip_artifact)
        )
    if outcome_status is AlignmentOutcomeStatus.SUCCESS:
        if offset is None or not isinstance(offset, AlignmentOffset):
            raise AlignmentOutcomeError("SUCCESS publication requires an AlignmentOffset")
        _validate_offset(offset)
        if (offset.user, offset.action, offset.dataset_id) != (
            identity["user"],
            identity["action"],
            identity["dataset_id"],
        ):
            raise AlignmentOutcomeError("offset identity does not match outcome report")
        if skip_value is not None or verification_path is None:
            if skip_value is not None:
                raise AlignmentOutcomeError("SUCCESS cannot include a skip artifact")
            raise AlignmentOutcomeError("SUCCESS publication requires verification PNG")
    elif outcome_status is AlignmentOutcomeStatus.SKIPPED:
        if skip_value is None:
            raise AlignmentOutcomeError("SKIPPED publication requires a skip artifact")
        skip_payload = skip_value.to_dict()
        if skip_payload["recording"] != identity:
            raise AlignmentOutcomeError("skip artifact identity does not match outcome report")
        if (
            skip_payload["input_provenance"] != report_base.get("input_provenance")
            or skip_payload["board_provenance"] != report_base.get("board_provenance")
        ):
            raise AlignmentOutcomeError(
                "skip artifact provenance does not match the current outcome provenance"
            )
        if offset is not None or verification_path is not None:
            raise AlignmentOutcomeError("SKIPPED cannot include success artifacts")
    else:
        if offset is not None or skip_value is not None or verification_path is not None:
            raise AlignmentOutcomeError("FAILED publication cannot include artifacts")

    old_state = _existing_outcome_state(paths)
    requested_state = outcome_status
    completed_states = {
        AlignmentOutcomeStatus.SUCCESS,
        AlignmentOutcomeStatus.SKIPPED,
    }
    # FAILED is a report-only state, not a completed outcome transition.  It
    # therefore never grants the cross-artifact overwrite authorization that a
    # SUCCESS <-> SKIPPED replacement receives.
    transition = (
        old_state in completed_states
        and requested_state in completed_states
        and old_state is not requested_state
    )
    opposite_present = _opposite_artifact_present(paths, requested_state)
    # A FAILED report is not a completed outcome.  If it coexists with stale
    # opposite-state artifacts, those artifacts are cleaned as part of the
    # requested publication; they do not create a completed-state transition
    # authorization requirement.
    outcome_transition_required = transition or (
        opposite_present and old_state is not AlignmentOutcomeStatus.FAILED
    )
    if outcome_transition_required and not overwrite_outcome:
        raise AlignmentOutcomeError(
            "success/skip outcome transition requires overwrite_outcome=True"
        )
    transition_authorized = bool(
        overwrite_outcome
        and (
            transition
            or (opposite_present and old_state is not AlignmentOutcomeStatus.FAILED)
        )
    )
    remove_obsolete_authorized = bool(
        transition_authorized
        or (old_state is AlignmentOutcomeStatus.FAILED and opposite_present)
    )
    _preflight_outcome_targets(
        paths,
        status=outcome_status,
        overwrite_offset=overwrite_offset,
        overwrite_verification=overwrite_verification,
        overwrite_report=overwrite_report,
        transition_authorized=transition_authorized,
    )

    staged: list[tuple[Path, Path]] = []
    try:
        manifest: list[dict[str, str]] = []
        if outcome_status is AlignmentOutcomeStatus.SUCCESS:
            offset_stage = _stage_offset(offset, paths.offset_txt_path.parent)
            staged.append((offset_stage, paths.offset_txt_path))
            manifest.append(
                {
                    "filename": paths.offset_txt_path.name,
                    "sha256": _sha256_regular_file(offset_stage, name="offset TXT"),
                }
            )
            verification_source = Path(verification_path or "")
            if not verification_source.is_file() or verification_source.stat().st_size <= 0:
                raise AlignmentOutcomeError(
                    "verification PNG must be a nonempty regular file"
                )
            verification_stage = _stage_file(
                verification_source, paths.verification_png_path.parent
            )
            staged.append((verification_stage, paths.verification_png_path))
            manifest.append(
                {
                    "filename": paths.verification_png_path.name,
                    "sha256": _sha256_regular_file(
                        verification_stage, name="verification PNG"
                    ),
                }
            )
            report_base.setdefault("alignment_success", True)
            report_base.setdefault("work_axis_alignment_success", True)
        elif outcome_status is AlignmentOutcomeStatus.SKIPPED:
            skip_stage = _stage_json(skip_value.to_dict(), paths.skip_json_path.parent)
            staged.append((skip_stage, paths.skip_json_path))
            manifest.append(
                {
                    "filename": paths.skip_json_path.name,
                    "sha256": _sha256_regular_file(skip_stage, name="skip JSON"),
                }
            )
            report_base.setdefault("alignment_success", False)
            report_base.setdefault("work_axis_alignment_success", False)
        else:
            report_base["alignment_success"] = False
            report_base["work_axis_alignment_success"] = False

        report_base["alignment_outcome_schema_version"] = ALIGNMENT_OUTCOME_SCHEMA_VERSION
        report_base["alignment_status"] = outcome_status.value
        report_base["outcome_artifacts"] = manifest
        report_payload = _jsonable(report_base)
        _validate_report_header(report_payload, identity=identity)
        report_stage = _stage_json(report_payload, paths.report_path.parent)
        try:
            for temporary, destination in tuple(staged):
                os.replace(temporary, destination)
                staged.remove((temporary, destination))
            if remove_obsolete_authorized:
                _remove_obsolete_artifacts(paths, outcome_status)
            # The report is intentionally replaced last.  Missing or stale
            # report/artifact combinations fail the authoritative validator.
            os.replace(report_stage, paths.report_path)
            report_stage = None  # type: ignore[assignment]
        finally:
            if report_stage is not None and report_stage.exists():
                report_stage.unlink(missing_ok=True)
    except (OSError, AlignmentOffsetExportError) as error:
        raise AlignmentOutcomeError(f"could not publish alignment outcome: {error}") from error
    finally:
        for temporary, _destination in staged:
            temporary.unlink(missing_ok=True)

    return AlignmentOutcome(
        status=outcome_status,
        recording=identity,
        paths=paths,
        report=report_payload,
    )


@_normalize_outcome_errors
def publish_alignment_success(
    paths: AlignmentOutcomePaths,
    *,
    offset: AlignmentOffset,
    report: Mapping[str, object],
    verification_path: Path,
    overwrite_offset: bool = False,
    overwrite_verification: bool = False,
    overwrite_report: bool = False,
    overwrite_outcome: bool = False,
) -> AlignmentOutcome:
    return publish_alignment_outcome(
        paths,
        status=AlignmentOutcomeStatus.SUCCESS,
        report=report,
        offset=offset,
        verification_path=verification_path,
        overwrite_offset=overwrite_offset,
        overwrite_verification=overwrite_verification,
        overwrite_report=overwrite_report,
        overwrite_outcome=overwrite_outcome,
    )


@_normalize_outcome_errors
def publish_alignment_skip(
    paths: AlignmentOutcomePaths,
    *,
    skip_artifact: AlignmentSkipArtifact,
    report: Mapping[str, object],
    overwrite_offset: bool = False,
    overwrite_skip: bool | None = None,
    overwrite_report: bool = False,
    overwrite_outcome: bool = False,
) -> AlignmentOutcome:
    return publish_alignment_outcome(
        paths,
        status=AlignmentOutcomeStatus.SKIPPED,
        report=report,
        skip_artifact=skip_artifact,
        overwrite_offset=overwrite_offset,
        overwrite_skip=overwrite_skip,
        overwrite_report=overwrite_report,
        overwrite_outcome=overwrite_outcome,
    )


@_normalize_outcome_errors
def write_alignment_outcome_report(
    path: Path,
    report: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> Path:
    """Write a strict report manifest for callers that stage artifacts first."""

    if not isinstance(report, Mapping):
        raise AlignmentOutcomeError("alignment outcome report must be a mapping")
    payload = _jsonable(dict(report))
    identity = _report_identity(payload)
    if identity is None:
        raise AlignmentOutcomeError("alignment outcome report must declare recording identity")
    _validate_report_header(payload, identity=identity)
    _validate_manifest(payload.get("outcome_artifacts"))
    _preflight_single_target(Path(path), overwrite=overwrite, label="alignment report")
    _atomic_write_json(Path(path), payload)
    return Path(path)


write_alignment_report = write_alignment_outcome_report


def _coerce_status(value: object) -> AlignmentOutcomeStatus:
    if isinstance(value, AlignmentOutcomeStatus):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {item.value for item in AlignmentOutcomeStatus}:
            return AlignmentOutcomeStatus(lowered)
        if value.upper() in AlignmentOutcomeStatus.__members__:
            return AlignmentOutcomeStatus[value.upper()]
    raise AlignmentOutcomeError(
        "alignment_status must be success, skipped, or failed"
    )


def _coerce_outcome_paths(
    value: AlignmentOutcomePaths | Path,
    *,
    verification_root: Path | None,
    report_root: Path | None,
    expected_user: str | None,
    expected_action: str | None,
    expected_dataset_id: int | None,
    expected_recording: Mapping[str, object] | None,
) -> AlignmentOutcomePaths:
    if isinstance(value, AlignmentOutcomePaths):
        return value
    if not isinstance(value, (str, Path)):
        raise AlignmentOutcomeError(
            "outcome roots must be AlignmentOutcomePaths or a path"
        )
    identity = _expected_identity(
        expected_user=expected_user,
        expected_action=expected_action,
        expected_dataset_id=expected_dataset_id,
        expected_recording=expected_recording,
    )
    if identity is None or verification_root is None or report_root is None:
        raise AlignmentOutcomeError(
            "outcome roots, verification root, report root, and identity are required"
        )
    return build_alignment_outcome_paths(
        Path(value),
        verification_root,
        report_root,
        user=str(identity["user"]),
        action=str(identity["action"]),
        dataset_id=int(identity["dataset_id"]),
    )


def _coerce_input_provenance(value: object) -> AlignmentInputProvenance:
    if isinstance(value, AlignmentInputProvenance):
        _validate_input_provenance(value)
        return value
    if not isinstance(value, Mapping):
        if all(
            hasattr(value, name)
            for name in (
                "input_kind",
                "user",
                "action",
                "dataset_id",
                "feature_schema",
                "sampling_rate_hz",
                "values_sha256",
                "metadata_sha256",
                "timestamps_sha256",
            )
        ):
            return build_alignment_input_provenance(value)
        raise AlignmentOutcomeError("input provenance must be a mapping or feature input")
    recording = value.get("recording")
    if not isinstance(recording, Mapping):
        recording = value
    user = recording.get("user")
    action = recording.get("action")
    dataset_id = recording.get("dataset_id", recording.get("data_id"))
    input_kind = value.get("input_kind")
    if not isinstance(input_kind, str):
        raise AlignmentOutcomeError("input provenance input_kind is required")
    timestamp = _coalesce_aliases(
        value,
        (
            "timestamp_sha256",
            "canonical_timestamps_sha256",
            "canonical_timestamp_sha256",
            "timestamp_array_sha256",
            "timestamps_sha256",
        ),
        name="timestamp SHA-256",
    )
    feature_values = _coalesce_aliases(
        value,
        ("feature_values_sha256", "values_sha256"),
        name="feature values SHA-256",
    )
    feature_metadata = _coalesce_aliases(
        value,
        ("feature_metadata_sha256", "metadata_sha256"),
        name="feature metadata SHA-256",
    )
    ring_digest = _coalesce_aliases(
        value,
        ("ring_0_sha256", "ring_0_content_sha256"),
        name="ring_0 SHA-256",
    )
    if input_kind == "raw-ring" and ring_digest is None:
        ring_digest = feature_values
    result = AlignmentInputProvenance(
        input_kind=input_kind,
        user=user,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
        dataset_id=dataset_id,  # type: ignore[arg-type]
        feature_schema=value.get("feature_schema"),  # type: ignore[arg-type]
        sampling_rate_hz=value.get(
            "sampling_rate_hz", value.get("feature_sampling_rate_hz")
        ),  # type: ignore[arg-type]
        feature_values_sha256=feature_values,  # type: ignore[arg-type]
        feature_metadata_sha256=feature_metadata,  # type: ignore[arg-type]
        timestamp_sha256=timestamp,  # type: ignore[arg-type]
        ring_0_sha256=ring_digest,  # type: ignore[arg-type]
    )
    _validate_input_provenance(result)
    return result


def _validate_input_provenance(value: AlignmentInputProvenance) -> None:
    if value.input_kind not in {"raw-ring", "spike-imu"}:
        raise AlignmentOutcomeError("input provenance input_kind is unsupported")
    _validate_identity(user=value.user, action=value.action, dataset_id=value.dataset_id)
    _strict_sha256(value.timestamp_sha256, name="timestamp_sha256")
    if value.input_kind == "raw-ring":
        _strict_sha256(value.ring_0_sha256, name="ring_0_sha256")
        if any(
            field is not None
            for field in (
                value.feature_schema,
                value.feature_values_sha256,
                value.feature_metadata_sha256,
            )
        ) and value.feature_values_sha256 != value.ring_0_sha256:
            raise AlignmentOutcomeError(
                "raw-ring feature values hash must identify ring_0 content"
            )
    else:
        if not isinstance(value.feature_schema, str) or not value.feature_schema:
            raise AlignmentOutcomeError("SpikeIMU feature_schema is required")
        rate = _finite_float(value.sampling_rate_hz, name="sampling_rate_hz")
        if rate <= 0.0:
            raise AlignmentOutcomeError("sampling_rate_hz must be positive")
        _strict_sha256(value.feature_values_sha256, name="feature_values_sha256")
        _strict_sha256(value.feature_metadata_sha256, name="feature_metadata_sha256")


def _coalesce_aliases(
    value: Mapping[str, object],
    names: Sequence[str],
    *,
    name: str,
) -> object | None:
    present = [value[key] for key in names if key in value]
    if not present:
        return None
    first = present[0]
    if any(item != first for item in present[1:]):
        raise AlignmentOutcomeError(f"conflicting {name} aliases")
    return first


def _coerce_board_provenance(value: object) -> list[dict[str, object]]:
    if isinstance(value, (str, Path)) or hasattr(value, "chunk_paths"):
        items: Sequence[object] = build_board_chunk_provenance(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if value and all(isinstance(item, (str, Path)) for item in value):
            items = build_board_chunk_provenance(value)
        else:
            items = value
    else:
        raise AlignmentOutcomeError("Board provenance must be an ordered sequence")
    result: list[dict[str, object]] = []
    for item in items:
        if isinstance(item, BoardChunkProvenance):
            result.append(item.to_dict())
        elif isinstance(item, Mapping):
            if set(item) - {"chunk_index", "sha256", "path"}:
                raise AlignmentOutcomeError("Board provenance entry has unknown fields")
            result.append(
                {
                    "chunk_index": _nonnegative_int(
                        item.get("chunk_index"), name="board chunk_index"
                    ),
                    "sha256": _strict_sha256(
                        item.get("sha256"), name="board chunk sha256"
                    ),
                }
            )
        else:
            raise AlignmentOutcomeError("Board provenance entries must be objects")
    if not result:
        raise AlignmentOutcomeError("Board provenance must contain at least one chunk")
    indices = [int(item["chunk_index"]) for item in result]
    if any(next_index < index for index, next_index in zip(indices, indices[1:])):
        raise AlignmentOutcomeError(
            "Board provenance chunks are not in numeric chunk-index order"
        )
    return result


def _validated_recording_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AlignmentOutcomeError("recording identity must be an object")
    dataset_id = value.get("dataset_id", value.get("data_id"))
    _validate_identity(
        user=value.get("user"),  # type: ignore[arg-type]
        action=value.get("action"),  # type: ignore[arg-type]
        dataset_id=dataset_id,  # type: ignore[arg-type]
    )
    return {
        "user": value["user"],
        "action": value["action"],
        "dataset_id": _nonnegative_int(dataset_id, name="dataset_id"),
    }


def _expected_identity(
    *,
    expected_user: str | None = None,
    expected_action: str | None = None,
    expected_dataset_id: int | None = None,
    expected_recording: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    supplied = (expected_user, expected_action, expected_dataset_id)
    if expected_recording is not None:
        if any(value is not None for value in supplied):
            raise AlignmentOutcomeError(
                "expected_recording cannot be combined with individual identity fields"
            )
        return _validated_recording_mapping(expected_recording)
    if all(value is None for value in supplied):
        return None
    if any(value is None for value in supplied):
        raise AlignmentOutcomeError(
            "expected_user, expected_action, and expected_dataset_id must be supplied together"
        )
    return _validated_recording_mapping(
        {
            "user": expected_user,
            "action": expected_action,
            "dataset_id": expected_dataset_id,
        }
    )


def _report_identity(report: Mapping[str, object]) -> dict[str, object] | None:
    value = report.get("recording")
    if value is None:
        return None
    return _validated_recording_mapping(value)


def _validate_skip_diagnostics(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AlignmentOutcomeError("alignment skip diagnostics must be an object")
    if set(value) != _SKIP_DIAGNOSTIC_KEYS:
        missing = sorted(_SKIP_DIAGNOSTIC_KEYS - set(value))
        extra = sorted(set(value) - _SKIP_DIAGNOSTIC_KEYS)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unknown " + ", ".join(extra))
        raise AlignmentOutcomeError(
            "alignment skip diagnostics keys are not exact (" + "; ".join(details) + ")"
        )
    result: dict[str, object] = {}
    integer_keys = {
        "previous_global_frame_index",
        "next_global_frame_index",
        "prefix_boundary_position",
        "last_pre_jump_global_frame_index",
        "total_global_valid_pair_count",
        "usable_prefix_valid_pair_count",
    }
    for key, item in value.items():
        if key in integer_keys:
            if isinstance(item, (bool, np.bool_)) or not isinstance(
                item, (int, np.integer)
            ):
                raise AlignmentOutcomeError(
                    f"diagnostics.{key} must be a nonnegative integer"
                )
            result[key] = _nonnegative_int(item, name=f"diagnostics.{key}")
        else:
            if isinstance(item, (bool, np.bool_)) or not isinstance(
                item, (int, float, np.integer, np.floating)
            ):
                raise AlignmentOutcomeError(
                    f"diagnostics.{key} must be a finite number"
                )
            result[key] = _finite_float(item, name=f"diagnostics.{key}")

    if result["total_global_valid_pair_count"] <= 0:
        raise AlignmentOutcomeError(
            "alignment skip diagnostics require a positive total global valid-pair count"
        )
    if result["usable_prefix_valid_pair_count"] != 0:
        raise AlignmentOutcomeError(
            "alignment skip diagnostics require zero usable-prefix valid pairs"
        )
    if not (
        result["next_timestamp_raw"] < result["previous_timestamp_raw"]
    ):
        raise AlignmentOutcomeError(
            "alignment skip diagnostics require a raw backward timestamp jump"
        )
    prefix_boundary = result["prefix_boundary_position"]
    if prefix_boundary <= 0:
        raise AlignmentOutcomeError(
            "alignment skip diagnostics require a positive prefix boundary"
        )
    previous_global = result["previous_global_frame_index"]
    next_global = result["next_global_frame_index"]
    if result["last_pre_jump_global_frame_index"] != previous_global:
        raise AlignmentOutcomeError(
            "alignment skip diagnostics last-pre frame must equal previous frame"
        )
    if previous_global != prefix_boundary - 1 or next_global != prefix_boundary:
        raise AlignmentOutcomeError(
            "alignment skip diagnostics frame identities must be contiguous at the prefix boundary"
        )
    return result


def _validate_report_header(
    report: Mapping[str, object],
    *,
    identity: Mapping[str, object],
) -> None:
    if not isinstance(report, Mapping):
        raise AlignmentOutcomeError("alignment outcome report must be an object")
    _literal_schema_version(
        report.get("alignment_outcome_schema_version"),
        name="alignment_outcome_schema_version",
    )
    status = _coerce_status(report.get("alignment_status"))
    reported_identity = _report_identity(report)
    if reported_identity != dict(identity):
        raise AlignmentOutcomeError("alignment outcome report identity is stale")
    if status is AlignmentOutcomeStatus.SUCCESS:
        for key in ("alignment_success", "work_axis_alignment_success"):
            if not isinstance(report.get(key), bool):
                raise AlignmentOutcomeError(
                    f"status-success report {key} must be a boolean"
                )
            if report.get(key) is not True:
                raise AlignmentOutcomeError(
                    f"status-success report {key} must agree with alignment_status"
                )
    elif status is AlignmentOutcomeStatus.SKIPPED:
        for key in ("alignment_success", "work_axis_alignment_success"):
            if not isinstance(report.get(key), bool):
                raise AlignmentOutcomeError(
                    f"status-skipped report {key} must be a boolean"
                )
            if report.get(key) is not False:
                raise AlignmentOutcomeError(
                    f"status-skipped report {key} must agree with alignment_status"
                )


def _validate_manifest(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise AlignmentOutcomeError("outcome_artifacts must be a list")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"filename", "sha256"}:
            raise AlignmentOutcomeError(
                "each outcome_artifacts entry must contain filename and sha256 only"
            )
        filename = item.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in {".", ".."}
        ):
            raise AlignmentOutcomeError("outcome artifact filename must be a base name")
        if filename in seen:
            raise AlignmentOutcomeError("outcome artifact filenames must be unique")
        digest = item.get("sha256")
        if not isinstance(digest, str) or digest != digest.lower():
            raise AlignmentOutcomeError("outcome artifact SHA-256 must be lowercase")
        _strict_sha256(digest, name="outcome artifact sha256")
        seen.add(filename)
        result.append({"filename": filename, "sha256": digest})
    return result


def _validate_manifest_file(
    path: Path,
    entries: Sequence[Mapping[str, str]],
    *,
    label: str,
) -> None:
    if not path.is_file():
        raise AlignmentOutcomeError(f"{label} is missing or not a regular file")
    digest = _sha256_regular_file(path, name=label)
    matches = [entry for entry in entries if entry.get("filename") == path.name]
    if len(matches) != 1 or matches[0].get("sha256") != digest:
        raise AlignmentOutcomeError(f"{label} digest does not match the report manifest")


def _validate_offset_input_provenance(
    offset: AlignmentOffset,
    input_payload: Mapping[str, object],
) -> None:
    """Check source-aware offset fields when the success TXT carries them."""

    input_kind = input_payload.get("input_kind")
    if input_kind != "spike-imu":
        return
    # The v1 outcome report is the authoritative provenance carrier. Legacy
    # success TXT files remain readable and may not carry source-aware fields;
    # when those fields are present, however, they must agree with the
    # current SpikeIMU input.
    if offset.alignment_signal_source is None:
        return
    if offset.alignment_signal_source != "spike-imu":
        raise AlignmentOutcomeError(
            "SpikeIMU success offset is missing feature provenance"
        )
    if offset.feature_schema != input_payload.get("feature_schema"):
        raise AlignmentOutcomeError("success offset feature schema is stale")
    if offset.feature_values_sha256 != input_payload.get("feature_values_sha256"):
        raise AlignmentOutcomeError("success offset feature values hash is stale")
    if offset.feature_metadata_sha256 != input_payload.get("feature_metadata_sha256"):
        raise AlignmentOutcomeError("success offset feature metadata hash is stale")
    if offset.timestamp_sha256 != input_payload.get("timestamp_sha256"):
        raise AlignmentOutcomeError("success offset timestamp hash is stale")
    expected_rate = input_payload.get("sampling_rate_hz")
    if (
        offset.feature_sampling_rate_hz is not None
        and not math.isclose(
            offset.feature_sampling_rate_hz,
            _finite_float(expected_rate, name="sampling_rate_hz"),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise AlignmentOutcomeError("success offset sampling rate is stale")


def _existing_outcome_state(
    paths: AlignmentOutcomePaths,
) -> AlignmentOutcomeStatus | None:
    if paths.report_path.is_file():
        try:
            payload = _read_strict_json(paths.report_path, label="existing alignment report")
            return _coerce_status(payload.get("alignment_status"))
        except AlignmentOffsetExportError:
            pass
    if paths.skip_json_path.exists() and not paths.offset_txt_path.exists():
        return AlignmentOutcomeStatus.SKIPPED
    if (paths.offset_txt_path.exists() or paths.verification_png_path.exists()) and not paths.skip_json_path.exists():
        return AlignmentOutcomeStatus.SUCCESS
    return None


def _opposite_artifact_present(
    paths: AlignmentOutcomePaths,
    requested: AlignmentOutcomeStatus,
) -> bool:
    if requested is AlignmentOutcomeStatus.SUCCESS:
        return paths.skip_json_path.exists()
    if requested is AlignmentOutcomeStatus.SKIPPED:
        return paths.offset_txt_path.exists() or paths.verification_png_path.exists()
    return (
        paths.offset_txt_path.exists()
        or paths.skip_json_path.exists()
        or paths.verification_png_path.exists()
    )


def _preflight_outcome_targets(
    paths: AlignmentOutcomePaths,
    *,
    status: AlignmentOutcomeStatus,
    overwrite_offset: bool,
    overwrite_verification: bool,
    overwrite_report: bool,
    transition_authorized: bool,
) -> None:
    for path in (
        paths.offset_txt_path,
        paths.skip_json_path,
        paths.verification_png_path,
        paths.report_path,
    ):
        if path.exists() and not path.is_file():
            raise AlignmentOutcomeError(
                f"alignment outcome target is not a regular file: {path}"
            )
    if status is AlignmentOutcomeStatus.SUCCESS:
        if paths.offset_txt_path.exists() and not (
            overwrite_offset or transition_authorized
        ):
            raise AlignmentOutcomeError(
                "offset output already exists; use overwrite_offset=True"
            )
        if paths.verification_png_path.exists() and not (
            overwrite_verification or transition_authorized
        ):
            raise AlignmentOutcomeError(
                "verification output already exists; use overwrite_verification=True"
            )
    elif status is AlignmentOutcomeStatus.SKIPPED:
        if paths.skip_json_path.exists() and not (
            overwrite_offset or transition_authorized
        ):
            raise AlignmentOutcomeError(
                "skip output already exists; use overwrite_offset=True"
            )
    if paths.report_path.exists() and not (overwrite_report or transition_authorized):
        raise AlignmentOutcomeError(
            "alignment report already exists; use overwrite_report=True"
        )


def _remove_obsolete_artifacts(
    paths: AlignmentOutcomePaths,
    status: AlignmentOutcomeStatus,
) -> None:
    obsolete = (
        (paths.skip_json_path,)
        if status is AlignmentOutcomeStatus.SUCCESS
        else (
            paths.offset_txt_path,
            paths.verification_png_path,
        )
        if status is AlignmentOutcomeStatus.SKIPPED
        else (
            paths.offset_txt_path,
            paths.skip_json_path,
            paths.verification_png_path,
        )
    )
    for path in obsolete:
        if path.exists():
            if not path.is_file():
                raise AlignmentOutcomeError(
                    f"obsolete alignment artifact is not a regular file: {path}"
                )
            path.unlink()


def _preflight_single_target(path: Path, *, overwrite: bool, label: str) -> None:
    if path.exists():
        if not path.is_file():
            raise AlignmentOutcomeError(f"{label} path is not a regular file: {path}")
        if not overwrite:
            raise AlignmentOutcomeError(f"{label} already exists; enable overwrite: {path}")


def _stage_offset(offset: AlignmentOffset, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=".alignment-offset-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        path = Path(stream.name)
    try:
        write_alignment_offset_txt(offset, output_path=path, overwrite=True)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _stage_file(source: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=directory,
        prefix=".alignment-artifact-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        destination = Path(stream.name)
        with Path(source).open("rb") as input_stream:
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                stream.write(block)
    return destination


def _stage_json(payload: Mapping[str, object], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    text = _strict_json_text(payload)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=".alignment-json-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(text)
        return Path(stream.name)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = _stage_json(payload, path.parent)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_json_text(payload: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            _jsonable(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise AlignmentOutcomeError(
            f"alignment outcome JSON is not strict: {error}"
        ) from error


def _read_strict_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AlignmentOutcomeError(f"could not read {label}: {path}: {error}") from error
    try:
        payload = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AlignmentOutcomeError(f"{label} is not strict JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AlignmentOutcomeError(f"{label} must be a JSON object: {path}")
    return payload


def _jsonable(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise AlignmentOutcomeError("alignment outcome JSON cannot contain non-finite values")
    return value


def _strict_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise AlignmentOutcomeError(f"{name} must be a lowercase SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise AlignmentOutcomeError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _literal_schema_version(value: object, *, name: str) -> int:
    """Accept only the literal JSON integer required by the v1 contracts."""

    if type(value) is not int or value != 1:
        raise AlignmentOutcomeError(f"{name} must be the literal integer 1")
    return value


def _sha256_regular_file(path: Path, *, name: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise AlignmentOutcomeError(f"{name} is not a regular file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AlignmentOutcomeError(f"could not hash {name}: {source}: {error}") from error
    return digest.hexdigest()
