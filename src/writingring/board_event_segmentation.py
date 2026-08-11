"""Aligned Board-event-guided Ring IMU segmentation and publication.

The recording-level API accepts already loaded immutable-source inputs.  The
separate user/action API performs discovery, required alignment validation,
verification plotting, and transactional output publication.  Label-only
segmentation remains isolated in :mod:`writingring.segmentation`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Final, Sequence

import numpy as np
import pandas as pd

from writingring.alignment_io import (
    ALIGNMENT_WORK_AXIS_DOMAIN,
    AlignmentOffset,
    AlignmentOffsetExportError,
    AlignmentOutcomeError,
    AlignmentOutcome,
    apply_board_to_ring_offset,
    build_alignment_input_provenance,
    build_board_chunk_provenance,
    build_offset_domain_timestamps,
    read_alignment_skip_artifact,
    read_alignment_offset_txt,
    validate_alignment_outcome,
    validate_alignment_feature_provenance,
)
from writingring.discovery import Recording
from writingring.board_loader import BoardLoadError, load_board
from writingring.event_alignment import (
    compute_transient_score,
    compute_transient_score_array,
    detect_board_events,
)
from writingring.gravity import (
    GravityRemovalConfig,
    GravityRemovalError,
    process_ring_gravity,
)
from writingring.imu_preprocessing import (
    IMU_PREPROCESSING_METHODS,
    IMUPreprocessingError,
    PREPROCESSED_IMU_COLUMNS,
    STANDARD_GRAVITY_M_S2,
    preprocess_ring_imu,
)
from writingring.ring_loader import RingLoadError, load_ring
from writingring.recording_features import (
    RecordingFeatureError,
    RecordingFeatureInput,
    load_recording_features,
    validate_common_feature_sampling_rate,
)
from writingring.preprocessing_io import sha256_file
from writingring.segmentation import (
    SegmentLabel,
    SegmentLabelParseError,
    SegmentationConfig,
    label_start_skip_reasons,
    load_timestamp_labels,
)


ALIGNED_BOARD_EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "event_type",
    "global_frame_index",
    "frame_timestamp_raw",
    "paired_touch_index",
    "duration_frames",
    "transient",
    "valid_touch",
    "incomplete_touch",
    "crossing_touch",
    "crosses_next_label_timestamp",
    "crossing_resolution",
    "assigned_label_index",
    "aligned_event_timestamp_us",
)
ALIGNED_TOUCH_PAIR_COLUMNS: Final[tuple[str, ...]] = (
    "paired_touch_index",
    "press_event_index",
    "lift_event_index",
    "press_global_frame_index",
    "lift_global_frame_index",
    "press_timestamp_raw",
    "lift_timestamp_raw",
    "duration_frames",
    "duration_s",
    "transient",
    "valid_touch",
    "incomplete_touch",
    "crossing_touch",
    "crosses_next_label_timestamp",
    "crossing_resolution",
    "assigned_label_index",
    "aligned_press_us",
    "aligned_lift_us",
)
_TARGET_CHANNELS: Final[dict[tuple[str, bool], int]] = {
    ("press", False): 0,
    ("lift", False): 1,
    ("press", True): 2,
    ("lift", True): 3,
}
class BoardEventSegmentationError(ValueError):
    """Raised when Board events cannot safely define aligned segments."""


@dataclass(frozen=True, slots=True)
class BoardEventSegmentationConfig:
    """Independent boundary and event policies for aligned Board mode."""

    output_dtype: str = "float32"
    include_last_label: bool = True
    pre_press_context_us: float = 200_000.0
    post_lift_context_us: float = 200_000.0
    missing_event_policy: str = "skip"
    crossing_touch_policy: str = "accept_until_next_press"
    minimum_label_interval_us: float = 100_000.0
    maximum_segment_duration_us: float = 5_000_000.0
    require_successful_alignment: bool = True
    label_time_domain: str = "ring"
    minimum_duration_frames: int = 3
    recording_error_policy: str = "error"


@dataclass(frozen=True, slots=True)
class BoardEventPreparation:
    """Complete non-stale Board data and normalized transition tables."""

    frames: pd.DataFrame
    contacts: pd.DataFrame
    events: pd.DataFrame
    touch_pairs: pd.DataFrame
    stale_tail_start_frame_index: int | None
    stale_tail_frame_count: int


@dataclass(frozen=True, slots=True)
class BoardEventSegmentedSample:
    """One final Board-guided variable-length IMU segment and its targets."""

    imu: np.ndarray
    board_event_targets: np.ndarray
    label: str
    source_label_index: int
    sample_count: int
    start_sample_index: int
    stop_sample_index_exclusive: int
    label_timestamp_us: float
    next_label_timestamp_us: float | None
    first_press_timestamp_us: float
    last_lift_timestamp_us: float
    provisional_start_timestamp_us: float
    provisional_end_timestamp_us: float
    final_start_timestamp_us: float
    final_end_timestamp_us: float
    valid_touch_pair_count: int
    transient_touch_pair_count: int
    incomplete_touch_pair_count: int
    crossing_touch_pair_count: int
    accepted_crossing_touch_pair_count: int
    rejected_crossing_touch_pair_count: int
    boundary_source: str
    boundary_collision_resolved: bool
    first_press_paired_touch_index: int | None = None
    last_lift_paired_touch_index: int | None = None
    boundary_start_timestamp_us: float | None = None
    boundary_end_timestamp_us: float | None = None


@dataclass(frozen=True, slots=True)
class SkippedBoardEventSegment:
    """One source label that remains auditable but exports no IMU data."""

    source_label_index: int
    label: str
    label_timestamp_us: float
    next_label_timestamp_us: float | None
    skip_reason: str
    valid_touch_pair_count: int = 0
    transient_touch_pair_count: int = 0
    incomplete_touch_pair_count: int = 0
    crossing_touch_pair_count: int = 0
    accepted_crossing_touch_pair_count: int = 0
    rejected_crossing_touch_pair_count: int = 0


@dataclass(frozen=True, slots=True)
class LabelIntervalAnalysis:
    """Immutable classification facts for one source label interval."""

    eligible_pairs: pd.DataFrame
    valid_touch_pair_count: int
    transient_touch_pair_count: int
    incomplete_touch_pair_count: int
    crossing_touch_pair_count: int
    accepted_crossing_touch_pair_count: int
    rejected_crossing_touch_pair_count: int


@dataclass(frozen=True, slots=True)
class BoardEventSegmentationResult:
    """All recording-level Board-guided samples, skips, and event tables."""

    samples: tuple[BoardEventSegmentedSample, ...]
    skipped: tuple[SkippedBoardEventSegment, ...]
    aligned_board_events: pd.DataFrame
    aligned_touch_pairs: pd.DataFrame
    label_skip_reasons: tuple[str | None, ...]
    interval_analyses: tuple[LabelIntervalAnalysis, ...]


@dataclass(frozen=True, slots=True)
class BoardEventSegmentationOutputPaths:
    """Method-isolated artifacts for one aligned user/action aggregation."""

    output_directory: Path
    raw_imu_path: Path
    labels_path: Path
    segment_offsets_path: Path
    segment_lengths_path: Path
    board_event_targets_path: Path
    segments_csv_path: Path
    board_events_csv_path: Path
    summary_json_path: Path


@dataclass(frozen=True, slots=True)
class BoardEventUserActionSegmentationResult:
    """Published contiguous aligned-mode arrays, audit tables, and paths."""

    user: str
    action: str
    raw_imu: np.ndarray
    labels: np.ndarray
    segment_offsets: np.ndarray
    segment_lengths: np.ndarray
    board_event_targets: np.ndarray
    manifest: pd.DataFrame
    board_events: pd.DataFrame
    summary: dict[str, object]
    output_paths: BoardEventSegmentationOutputPaths
    recording_error_report_path: Path | None = None


@dataclass(frozen=True, slots=True)
class BoardEventUserActionSegmentationErrorResult:
    """A completed report-only user/action with no exportable segment package."""

    user: str
    action: str
    summary: dict[str, object]
    recording_error_report_path: Path


@dataclass(slots=True)
class _CandidateWindow:
    source_label_index: int
    label: SegmentLabel
    next_label: SegmentLabel | None
    first_press_us: float
    last_lift_us: float
    provisional_start_us: float
    provisional_end_us: float
    valid_pair_count: int
    transient_pair_count: int
    incomplete_pair_count: int
    crossing_pair_count: int
    accepted_crossing_pair_count: int
    rejected_crossing_pair_count: int
    final_start_us: float
    final_end_us: float
    collision_resolved: bool = False


def prepare_complete_board_events(
    board_frames: pd.DataFrame,
    board_contacts: pd.DataFrame,
    *,
    minimum_duration_frames: int = 3,
) -> BoardEventPreparation:
    """Trim the first backward-time stale tail, then detect all transitions."""

    frames, contacts, stale_start = trim_board_stale_tail(board_frames, board_contacts)
    detection = detect_board_events(
        frames, minimum_duration_frames=minimum_duration_frames
    )
    events, touch_pairs = _standardize_board_detection(
        detection.events, detection.touch_pairs
    )
    return BoardEventPreparation(
        frames=frames,
        contacts=contacts,
        events=events,
        touch_pairs=touch_pairs,
        stale_tail_start_frame_index=stale_start,
        stale_tail_frame_count=len(board_frames) - len(frames),
    )


def trim_board_stale_tail(
    board_frames: pd.DataFrame,
    board_contacts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int | None]:
    """Keep the initial nondecreasing Board timestamp run without mutation."""

    frames = _validated_board_frames(board_frames)
    contacts = _validated_board_contacts(board_contacts)
    timestamps = frames["frame_timestamp_raw"].to_numpy(dtype=np.float64)
    backward = np.flatnonzero(np.diff(timestamps) < 0.0)
    stop = int(backward[0] + 1) if len(backward) else len(frames)
    trimmed_frames = frames.iloc[:stop].copy(deep=True).reset_index(drop=True)
    if trimmed_frames.empty:
        raise BoardEventSegmentationError("Board stale-tail trim left no frames")
    last_global_frame = int(trimmed_frames["global_frame_index"].iloc[-1])
    trimmed_contacts = contacts.loc[
        contacts["global_frame_index"].astype(np.int64) <= last_global_frame
    ].copy(deep=True).reset_index(drop=True)
    return trimmed_frames, trimmed_contacts, (stop if stop < len(frames) else None)


def read_recording_alignment_offset(
    path: Path,
    *,
    recording: Recording,
) -> AlignmentOffset:
    """Read an already estimated offset while enforcing recording identity."""

    if not isinstance(recording, Recording):
        raise BoardEventSegmentationError("recording must be a Recording")
    try:
        return read_alignment_offset_txt(
            path,
            expected_user=recording.user,
            expected_action=recording.action,
            expected_dataset_id=recording.dataset_id,
        )
    except AlignmentOffsetExportError as error:
        raise BoardEventSegmentationError(
            f"invalid alignment offset for dataset {recording.dataset_id}: {error}"
        ) from error


def align_board_event_tables(
    events: pd.DataFrame,
    touch_pairs: pd.DataFrame,
    *,
    offset: AlignmentOffset,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a validated Board-to-Ring offset exactly once to derived tables."""

    if not isinstance(offset, AlignmentOffset) or not offset.alignment_success:
        raise BoardEventSegmentationError("a successful AlignmentOffset is required")
    raw_events, raw_pairs = _validated_standard_event_tables(events, touch_pairs)
    if "aligned_event_timestamp_us" in events or "aligned_press_us" in touch_pairs:
        raise BoardEventSegmentationError("alignment offset must not be applied twice")
    try:
        boundary_offset_us = offset.boundary_offset_us
        aligned_events = raw_events.copy(deep=True)
        aligned_events["aligned_event_timestamp_us"] = apply_board_to_ring_offset(
            aligned_events["frame_timestamp_raw"].to_numpy(dtype=np.float64),
            offset_us=boundary_offset_us,
        )
        aligned_pairs = raw_pairs.copy(deep=True)
        aligned_pairs["aligned_press_us"] = apply_board_to_ring_offset(
            aligned_pairs["press_timestamp_raw"].to_numpy(dtype=np.float64),
            offset_us=boundary_offset_us,
        )
        lift_raw = aligned_pairs["lift_timestamp_raw"].to_numpy(dtype=np.float64)
        aligned_pairs["aligned_lift_us"] = np.where(
            np.isfinite(lift_raw), lift_raw + boundary_offset_us, np.nan
        )
    except (AlignmentOffsetExportError, TypeError, ValueError) as error:
        raise BoardEventSegmentationError(f"could not align Board events: {error}") from error
    return aligned_events, aligned_pairs


def segment_recording_by_aligned_board_events(
    *,
    ring_imu: np.ndarray | None = None,
    ring_timestamps_us: np.ndarray | None = None,
    feature_values: np.ndarray | None = None,
    timestamps_us: np.ndarray | None = None,
    labels: Sequence[SegmentLabel],
    aligned_board_events: pd.DataFrame,
    aligned_touch_pairs: pd.DataFrame,
    config: BoardEventSegmentationConfig = BoardEventSegmentationConfig(),
    boundary_timestamps_us: np.ndarray | None = None,
) -> BoardEventSegmentationResult:
    """Build segments with boundary lookup separated from feature slicing.

    ``timestamps_us`` remains the canonical feature timestamp vector.  When
    ``boundary_timestamps_us`` is supplied, all Board/label boundary searches
    use that work-axis vector while the returned IMU rows and their canonical
    timestamp metadata are selected by sample index.
    """

    dtype = _validated_config(config)
    imu_source, timestamp_source = _resolve_feature_arguments(
        feature_values=feature_values,
        timestamps_us=timestamps_us,
        ring_imu=ring_imu,
        ring_timestamps_us=ring_timestamps_us,
    )
    imu, timestamps = _validated_ring_inputs(imu_source, timestamp_source)
    boundary_timestamps = _validated_boundary_timestamps(
        boundary_timestamps_us, canonical_timestamps_us=timestamps
    )
    markers = _validated_labels(labels)
    _validate_label_range(markers, boundary_timestamps)
    events, touch_pairs = _validated_aligned_event_tables(
        aligned_board_events, aligned_touch_pairs
    )
    events, touch_pairs = _classify_crossing_touches(
        events, touch_pairs, markers, policy=config.crossing_touch_policy
    )
    label_config = SegmentationConfig(
        output_dtype=config.output_dtype,
        include_last_label=config.include_last_label,
        minimum_label_interval_us=config.minimum_label_interval_us,
        maximum_segment_duration_us=config.maximum_segment_duration_us,
    )
    label_reasons = label_start_skip_reasons(
        markers,
        ring_end_timestamp_us=float(boundary_timestamps[-1]),
        config=label_config,
    )
    analyses = tuple(
        _analyze_label_interval(
            source_label_index=index,
            label=label,
            next_label=(markers[index + 1] if index + 1 < len(markers) else None),
            ring_end_us=float(boundary_timestamps[-1]),
            events=events,
            touch_pairs=touch_pairs,
        )
        for index, label in enumerate(markers)
    )
    candidates: dict[int, _CandidateWindow] = {}
    skipped: dict[int, SkippedBoardEventSegment] = {}
    selected_count = len(markers) if config.include_last_label else len(markers) - 1
    for index, label in enumerate(markers):
        next_label = markers[index + 1] if index + 1 < len(markers) else None
        analysis = analyses[index]
        if index >= selected_count:
            skipped[index] = _skipped(
                label, index, next_label, "excluded_last_label", analysis=analysis
            )
            continue
        if label_reasons[index] is not None:
            skipped[index] = _skipped(
                label,
                index,
                next_label,
                label_reasons[index] or "invalid_label",
                analysis=analysis,
            )
            continue
        if analysis.eligible_pairs.empty:
            skipped[index] = _skipped(
                label,
                index,
                next_label,
                _missing_event_reason(
                    analysis.transient_touch_pair_count,
                    analysis.incomplete_touch_pair_count,
                    analysis.rejected_crossing_touch_pair_count,
                ),
                analysis=analysis,
            )
            continue
        eligible = analysis.eligible_pairs
        first_press = float(eligible.iloc[0]["aligned_press_us"])
        last_lift = float(eligible.iloc[-1]["aligned_lift_us"])
        candidates[index] = _CandidateWindow(
            source_label_index=index,
            label=label,
            next_label=next_label,
            first_press_us=first_press,
            last_lift_us=last_lift,
            provisional_start_us=max(float(boundary_timestamps[0]), first_press - config.pre_press_context_us),
            provisional_end_us=min(float(boundary_timestamps[-1]), last_lift + config.post_lift_context_us),
            valid_pair_count=len(eligible),
            transient_pair_count=analysis.transient_touch_pair_count,
            incomplete_pair_count=analysis.incomplete_touch_pair_count,
            crossing_pair_count=analysis.crossing_touch_pair_count,
            accepted_crossing_pair_count=analysis.accepted_crossing_touch_pair_count,
            rejected_crossing_pair_count=analysis.rejected_crossing_touch_pair_count,
            final_start_us=max(float(boundary_timestamps[0]), first_press - config.pre_press_context_us),
            final_end_us=min(float(boundary_timestamps[-1]), last_lift + config.post_lift_context_us),
        )

    _resolve_window_collisions(candidates)
    _resolve_remaining_overlaps(candidates)
    samples: list[BoardEventSegmentedSample] = []
    for index in sorted(candidates):
        candidate = candidates[index]
        eligible = analyses[index].eligible_pairs
        if candidate.final_start_us >= candidate.final_end_us:
            skipped[index] = _skipped(
                candidate.label,
                index,
                candidate.next_label,
                "final_window_empty_after_collision",
                analysis=analyses[index],
            )
            continue
        start = int(
            np.searchsorted(boundary_timestamps, candidate.final_start_us, side="left")
        )
        stop = int(
            np.searchsorted(boundary_timestamps, candidate.final_end_us, side="left")
        )
        if not 0 <= start < stop <= len(imu):
            skipped[index] = _skipped(
                candidate.label,
                index,
                candidate.next_label,
                "final_window_has_no_ring_samples",
                analysis=analyses[index],
            )
            continue
        segment = np.asarray(imu[start:stop], dtype=dtype).copy()
        boundary_segment_times = boundary_timestamps[start:stop]
        targets = _event_targets_for_segment(
            events,
            segment_timestamps_us=boundary_segment_times,
            final_start_us=candidate.final_start_us,
            final_end_us=candidate.final_end_us,
            source_label_index=index,
        )
        segment.setflags(write=False)
        targets.setflags(write=False)
        samples.append(
            BoardEventSegmentedSample(
                imu=segment,
                board_event_targets=targets,
                label=candidate.label.label,
                source_label_index=index,
                sample_count=len(segment),
                start_sample_index=start,
                stop_sample_index_exclusive=stop,
                label_timestamp_us=_canonical_point_for_boundary(
                    candidate.label.timestamp_us,
                    canonical_timestamps_us=timestamps,
                    boundary_timestamps_us=boundary_timestamps,
                ),
                next_label_timestamp_us=(
                    None
                    if candidate.next_label is None
                    else _canonical_point_for_boundary(
                        candidate.next_label.timestamp_us,
                        canonical_timestamps_us=timestamps,
                        boundary_timestamps_us=boundary_timestamps,
                    )
                ),
                first_press_timestamp_us=_canonical_point_for_boundary(
                    candidate.first_press_us,
                    canonical_timestamps_us=timestamps,
                    boundary_timestamps_us=boundary_timestamps,
                ),
                last_lift_timestamp_us=_canonical_point_for_boundary(
                    candidate.last_lift_us,
                    canonical_timestamps_us=timestamps,
                    boundary_timestamps_us=boundary_timestamps,
                ),
                provisional_start_timestamp_us=_canonical_start_for_boundary(
                    candidate.provisional_start_us,
                    canonical_timestamps_us=timestamps,
                    boundary_timestamps_us=boundary_timestamps,
                ),
                provisional_end_timestamp_us=_canonical_end_for_boundary(
                    candidate.provisional_end_us,
                    canonical_timestamps_us=timestamps,
                    boundary_timestamps_us=boundary_timestamps,
                ),
                final_start_timestamp_us=_canonical_start_for_boundary(
                    candidate.final_start_us,
                    canonical_timestamps_us=timestamps,
                    boundary_timestamps_us=boundary_timestamps,
                ),
                final_end_timestamp_us=_canonical_end_for_boundary(
                    candidate.final_end_us,
                    canonical_timestamps_us=timestamps,
                    boundary_timestamps_us=boundary_timestamps,
                ),
                valid_touch_pair_count=candidate.valid_pair_count,
                transient_touch_pair_count=candidate.transient_pair_count,
                incomplete_touch_pair_count=candidate.incomplete_pair_count,
                crossing_touch_pair_count=candidate.crossing_pair_count,
                accepted_crossing_touch_pair_count=candidate.accepted_crossing_pair_count,
                rejected_crossing_touch_pair_count=candidate.rejected_crossing_pair_count,
                boundary_source="aligned_board_events",
                boundary_collision_resolved=candidate.collision_resolved,
                first_press_paired_touch_index=_optional_int(
                    eligible.iloc[0]["paired_touch_index"]
                ),
                last_lift_paired_touch_index=_optional_int(
                    eligible.iloc[-1]["paired_touch_index"]
                ),
                boundary_start_timestamp_us=candidate.final_start_us,
                boundary_end_timestamp_us=candidate.final_end_us,
            )
        )
    _validate_final_samples(samples)
    return BoardEventSegmentationResult(
        samples=tuple(samples),
        skipped=tuple(skipped[index] for index in sorted(skipped)),
        aligned_board_events=events,
        aligned_touch_pairs=touch_pairs,
        label_skip_reasons=label_reasons,
        interval_analyses=analyses,
    )


def build_board_event_segmentation_output_paths(
    output_root: Path,
    *,
    user: str,
    action: str,
    input_kind: str = "raw-ring",
) -> BoardEventSegmentationOutputPaths:
    """Return aligned-mode paths without publishing anything."""

    _validate_identity(user=user, action=action)
    if input_kind not in {"raw-ring", "spike-imu"}:
        raise BoardEventSegmentationError(
            "input_kind must be 'raw-ring' or 'spike-imu'"
        )
    base = Path(output_root) / user / f"action_{action}"
    stem = f"{user}_action_{action}"
    feature_suffix = "spikeIMU" if input_kind == "spike-imu" else "rawIMU"
    return BoardEventSegmentationOutputPaths(
        output_directory=base,
        raw_imu_path=base / f"{stem}_{feature_suffix}.npy",
        labels_path=base / f"{stem}_labels.npy",
        segment_offsets_path=base / f"{stem}_segment_offsets.npy",
        segment_lengths_path=base / f"{stem}_segment_lengths.npy",
        board_event_targets_path=base / f"{stem}_board_event_targets.npy",
        segments_csv_path=base / f"{stem}_segments.csv",
        board_events_csv_path=base / f"{stem}_board_events.csv",
        summary_json_path=base / f"{stem}_segmentation_summary.json",
    )


def segment_user_action_by_aligned_board_events(
    *,
    data_root: Path,
    user: str,
    action: str,
    output_root: Path,
    alignment_offset_root: Path,
    config: BoardEventSegmentationConfig = BoardEventSegmentationConfig(),
    verification_config: object | None = None,
    gravity_config: GravityRemovalConfig | None = None,
    input_kind: str = "raw-ring",
    spike_root: Path | None = None,
    expected_sampling_rate_hz: float | None = None,
    overwrite: bool = False,
) -> (
    BoardEventUserActionSegmentationResult
    | BoardEventUserActionSegmentationErrorResult
):
    """Load, align, verify, aggregate, and transactionally publish one action."""

    _validated_config(config)
    _validate_identity(user=user, action=action)
    if input_kind not in {"raw-ring", "spike-imu"}:
        raise BoardEventSegmentationError(
            "input_kind must be 'raw-ring' or 'spike-imu'"
        )
    if input_kind == "spike-imu" and spike_root is None:
        raise BoardEventSegmentationError(
            "spike_root is required for input_kind='spike-imu'"
        )
    if input_kind == "raw-ring" and spike_root is not None:
        raise BoardEventSegmentationError(
            "spike_root is only valid for input_kind='spike-imu'"
        )
    if input_kind == "raw-ring" and expected_sampling_rate_hz is not None:
        raise BoardEventSegmentationError(
            "expected_sampling_rate_hz is only valid for input_kind='spike-imu'"
        )
    if input_kind == "spike-imu" and gravity_config is not None:
        raise BoardEventSegmentationError(
            "gravity preprocessing settings are not valid for input_kind='spike-imu'"
        )
    if not isinstance(overwrite, bool):
        raise BoardEventSegmentationError("overwrite must be a boolean")
    from writingring.discovery import DiscoveryError, discover_recordings
    from writingring.segmentation_verification import (
        SegmentationVerificationConfig,
        SegmentationVerificationError,
        build_segmentation_verification_path,
        create_segmentation_verification_figure,
    )

    effective_verification = (
        SegmentationVerificationConfig()
        if verification_config is None
        else verification_config
    )
    if not isinstance(effective_verification, SegmentationVerificationConfig):
        raise BoardEventSegmentationError(
            "verification_config must be SegmentationVerificationConfig"
        )
    effective_gravity = (
        _effective_gravity_config(gravity_config)
        if input_kind == "raw-ring"
        else None
    )
    try:
        recordings = discover_recordings(data_root)
    except DiscoveryError as error:
        raise BoardEventSegmentationError(str(error)) from error
    selected = sorted(
        (
            recording
            for recording in recordings
            if recording.user == user and recording.action == action
        ),
        key=lambda recording: recording.dataset_id,
    )
    if not selected:
        raise BoardEventSegmentationError(
            f"no recordings found for user={user!r}, action={action!r}"
        )
    paths = build_board_event_segmentation_output_paths(
        output_root, user=user, action=action, input_kind=input_kind
    )
    if paths.output_directory.exists() and not overwrite:
        raise BoardEventSegmentationError(
            "aligned-board-events output already exists; use overwrite=True: "
            + str(paths.output_directory)
        )
    preloaded_feature_inputs: dict[int, RecordingFeatureInput] = {}
    if input_kind == "spike-imu":
        for recording in selected:
            feature_input = _load_alignment_feature_input(
                recording,
                input_kind=input_kind,
                gravity_config=effective_gravity,
                spike_root=spike_root,
                # The caller-requested rate is a SUCCESS-only condition in
                # aligned mode.  Load and validate every feature artifact
                # before classifying its alignment outcome so a valid
                # SKIPPED recording cannot fail this check prematurely.
                expected_sampling_rate_hz=None,
            )
            preloaded_feature_inputs[recording.dataset_id] = feature_input
    output_parent = paths.output_directory.parent
    output_root_directory = output_parent.parent
    output_parent_existed = output_parent.exists()
    output_root_existed = output_root_directory.exists()
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{paths.output_directory.name}.",
            dir=paths.output_directory.parent,
        )
    )
    try:
        recording_artifacts = _build_aligned_recording_artifacts(
            selected,
            staging_directory=staging,
            alignment_offset_root=Path(alignment_offset_root),
            config=config,
            verification_config=effective_verification,
            gravity_config=effective_gravity,
            input_kind=input_kind,
            spike_root=spike_root,
            expected_sampling_rate_hz=expected_sampling_rate_hz,
            preloaded_feature_inputs=(
                preloaded_feature_inputs if input_kind == "spike-imu" else None
            ),
            build_verification_path=build_segmentation_verification_path,
            create_verification_figure=create_segmentation_verification_figure,
            verification_error=SegmentationVerificationError,
        )
        if (
            not recording_artifacts.processed
            and recording_artifacts.segmentation_errors
        ):
            _remove_existing_aligned_output(
                paths.output_directory, overwrite=overwrite
            )
            error_report_path = _write_recording_error_report(
                output_root=Path(output_root),
                user=user,
                action=action,
                report=_recording_error_report(
                    recording_artifacts,
                    input_kind=input_kind,
                    terminal_state="all_recordings_error",
                ),
            )
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            return BoardEventUserActionSegmentationErrorResult(
                user=user,
                action=action,
                summary=_terminal_recording_error_summary(
                    recording_artifacts,
                    input_kind=input_kind,
                    alignment_offset_root=Path(alignment_offset_root),
                    config=config,
                ),
                recording_error_report_path=error_report_path,
            )
        result = _aggregate_and_write_aligned_outputs(
            recording_artifacts,
            user=user,
            action=action,
            output_directory=staging,
            alignment_offset_root=Path(alignment_offset_root),
            config=config,
            gravity_config=effective_gravity,
            input_kind=input_kind,
            published_output_directory=paths.output_directory,
        )
        _publish_staging_directory(
            staging, final_directory=paths.output_directory, overwrite=overwrite
        )
        published_paths = build_board_event_segmentation_output_paths(
            output_root,
            user=user,
            action=action,
            input_kind=input_kind,
        )
        error_report_path = _update_recording_error_report(
            output_root=Path(output_root),
            user=user,
            action=action,
            recording_artifacts=recording_artifacts,
            input_kind=input_kind,
            terminal_state="completed_with_recording_errors",
        )
        return replace(
            result,
            output_paths=published_paths,
            recording_error_report_path=error_report_path,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        for directory, existed in (
            (output_parent, output_parent_existed),
            (output_root_directory, output_root_existed),
        ):
            if existed or not directory.exists():
                continue
            try:
                directory.rmdir()
            except OSError:
                pass
        raise


@dataclass(frozen=True, slots=True)
class _RecordingArtifacts:
    recording: Recording
    feature_input: RecordingFeatureInput
    timestamps_us: np.ndarray
    boundary_timestamps_us: np.ndarray
    result: BoardEventSegmentationResult
    labels: tuple[SegmentLabel, ...]
    offset: AlignmentOffset
    offset_path: Path
    verification_path: Path
    outcome: AlignmentOutcome


@dataclass(frozen=True, slots=True)
class _SkippedRecordingArtifacts:
    """One provenance-valid recording-level alignment skip."""

    recording: Recording
    reason: str
    diagnostics: Mapping[str, object]
    outcome: AlignmentOutcome


@dataclass(frozen=True, slots=True)
class _SegmentationErrorRecordingArtifacts:
    """A validated alignment success that failed an allowed local segment step."""

    recording: Recording
    stage: str
    error_type: str
    message: str
    outcome: AlignmentOutcome


@dataclass(frozen=True, slots=True)
class _AlignedRecordingArtifacts:
    """All selected recordings partitioned by validated alignment outcome."""

    processed: tuple[_RecordingArtifacts, ...]
    skipped: tuple[_SkippedRecordingArtifacts, ...]
    segmentation_errors: tuple[_SegmentationErrorRecordingArtifacts, ...]
    source_recordings: tuple[Recording, ...]
    source_recording_count: int


class _RecordingLocalSegmentationFailure(Exception):
    """Wrap one explicitly permitted recording-local segmentation failure."""

    def __init__(self, *, stage: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


def _prepare_success_recording_segmentation(
    *,
    recording: Recording,
    feature_values: np.ndarray,
    timestamps_us: np.ndarray,
    boundary_timestamps_us: np.ndarray,
    offset: AlignmentOffset,
    board_frames: pd.DataFrame,
    board_contacts: pd.DataFrame,
    config: BoardEventSegmentationConfig,
) -> tuple[tuple[SegmentLabel, ...], BoardEventSegmentationResult, tuple[str | None, ...]]:
    """Run the narrowly quarantineable portion of a SUCCESS recording."""

    if recording.timestamp_path is None:
        raise _RecordingLocalSegmentationFailure(
            stage="timestamp_labels",
            cause=BoardEventSegmentationError(
                f"dataset {recording.dataset_id} is missing its timestamp label file"
            ),
        )
    try:
        source_labels = load_timestamp_labels(recording.timestamp_path)
        boundary_labels = _labels_in_boundary_domain(
            source_labels,
            canonical_timestamps_us=timestamps_us,
            boundary_timestamps_us=boundary_timestamps_us,
            offset_us=offset.boundary_offset_us,
            label_time_domain=config.label_time_domain,
        )
    except (SegmentLabelParseError, BoardEventSegmentationError) as error:
        raise _RecordingLocalSegmentationFailure(
            stage="timestamp_labels", cause=error
        ) from error
    try:
        prepared = prepare_complete_board_events(
            board_frames,
            board_contacts,
            minimum_duration_frames=config.minimum_duration_frames,
        )
    except BoardEventSegmentationError as error:
        raise _RecordingLocalSegmentationFailure(
            stage="board_event_preparation", cause=error
        ) from error
    try:
        events, pairs = align_board_event_tables(
            prepared.events, prepared.touch_pairs, offset=offset
        )
    except BoardEventSegmentationError as error:
        raise _RecordingLocalSegmentationFailure(
            stage="board_event_alignment", cause=error
        ) from error
    try:
        recording_result = segment_recording_by_aligned_board_events(
            feature_values=feature_values,
            timestamps_us=timestamps_us,
            labels=boundary_labels,
            aligned_board_events=events,
            aligned_touch_pairs=pairs,
            config=config,
            boundary_timestamps_us=boundary_timestamps_us,
        )
        reasons = _combined_label_skip_reasons(source_labels, recording_result)
    except BoardEventSegmentationError as error:
        raise _RecordingLocalSegmentationFailure(
            stage="board_event_segmentation", cause=error
        ) from error
    return source_labels, recording_result, reasons


def _is_quarantinable_recording_failure(error: Exception) -> bool:
    """Reject filesystem and alignment-artifact failures hidden by wrappers."""

    current: BaseException | None = error
    while current is not None:
        if isinstance(current, (OSError, AlignmentOffsetExportError, AlignmentOutcomeError)):
            return False
        current = current.__cause__
    return isinstance(error, (SegmentLabelParseError, BoardEventSegmentationError))


def _build_aligned_recording_artifacts(
    recordings: Sequence[Recording],
    *,
    staging_directory: Path,
    alignment_offset_root: Path,
    config: BoardEventSegmentationConfig,
    verification_config: object,
    gravity_config: GravityRemovalConfig | None,
    input_kind: str,
    spike_root: Path | None,
    expected_sampling_rate_hz: float | None,
    preloaded_feature_inputs: Mapping[int, RecordingFeatureInput] | None,
    build_verification_path: object,
    create_verification_figure: object,
    verification_error: type[Exception],
) -> _AlignedRecordingArtifacts:
    """Validate outcomes, then prepare only SUCCESS recordings for export.

    Feature and Board inputs are loaded before the public alignment outcome
    validator so both completed states are checked against current
    provenance.  A validated SKIPPED recording is retained only as an audit
    detail; it never enters label, event, segment, verification, or aggregate
    processing.
    """

    ordered_recordings = tuple(
        sorted(recordings, key=lambda recording: recording.dataset_id)
    )
    artifacts: list[_RecordingArtifacts] = []
    skipped_recordings: list[_SkippedRecordingArtifacts] = []
    segmentation_errors: list[_SegmentationErrorRecordingArtifacts] = []
    report_root = Path(alignment_offset_root).parent / "reports"
    verification_root = Path(alignment_offset_root).parent / "verification"
    for recording in ordered_recordings:
        if preloaded_feature_inputs is None:
            feature_input = _load_alignment_feature_input(
                recording,
                input_kind=input_kind,
                gravity_config=gravity_config,
                spike_root=spike_root,
                expected_sampling_rate_hz=(
                    None if input_kind == "spike-imu" else expected_sampling_rate_hz
                ),
            )
        else:
            try:
                feature_input = preloaded_feature_inputs[recording.dataset_id]
            except KeyError as error:
                raise BoardEventSegmentationError(
                    f"preloaded feature input is missing dataset {recording.dataset_id}"
                ) from error
        ring_imu = feature_input.values
        timestamps = feature_input.timestamps_us

        try:
            board = load_board(recording)
        except BoardLoadError as error:
            raise BoardEventSegmentationError(
                f"could not load Board data for dataset {recording.dataset_id}: {error}"
            ) from error
        try:
            input_provenance = build_alignment_input_provenance(feature_input)
            board_provenance = build_board_chunk_provenance(board)
            outcome = validate_alignment_outcome(
                alignment_offset_root,
                verification_root,
                report_root,
                expected_recording={
                    "user": recording.user,
                    "action": recording.action,
                    "dataset_id": recording.dataset_id,
                },
                expected_input_provenance=input_provenance,
                expected_board_provenance=board_provenance,
            )
        except AlignmentOutcomeError as error:
            raise BoardEventSegmentationError(
                f"alignment outcome validation failed for dataset "
                f"{recording.dataset_id}: {error}"
            ) from error

        if outcome.is_skipped:
            try:
                skip = read_alignment_skip_artifact(
                    outcome.paths.skip_json_path,
                    expected_user=recording.user,
                    expected_action=recording.action,
                    expected_dataset_id=recording.dataset_id,
                    expected_input_provenance=input_provenance,
                    expected_board_provenance=board_provenance,
                )
            except AlignmentOutcomeError as error:
                raise BoardEventSegmentationError(
                    f"alignment skip validation failed for dataset "
                    f"{recording.dataset_id}: {error}"
                ) from error
            skipped_recordings.append(
                _SkippedRecordingArtifacts(
                    recording=recording,
                    reason=skip.reason,
                    diagnostics=dict(skip.diagnostics),
                    outcome=outcome,
                )
            )
            continue
        if not outcome.is_success:
            raise BoardEventSegmentationError(
                f"alignment outcome for dataset {recording.dataset_id} is not SUCCESS"
            )
        if input_kind == "spike-imu":
            _validate_requested_spike_sampling_rate(
                feature_input,
                expected_sampling_rate_hz=expected_sampling_rate_hz,
            )

        offset_path = outcome.paths.offset_txt_path
        offset = read_recording_alignment_offset(offset_path, recording=recording)
        if config.require_successful_alignment and not offset.alignment_success:
            raise BoardEventSegmentationError(
                f"alignment was not successful for dataset {recording.dataset_id}"
            )
        if input_kind == "spike-imu":
            try:
                validate_alignment_feature_provenance(offset, feature_input)
            except AlignmentOffsetExportError as error:
                raise BoardEventSegmentationError(
                    f"alignment feature provenance failed for dataset "
                    f"{recording.dataset_id}: {error}"
                ) from error
        try:
            boundary_timestamps = build_offset_domain_timestamps(
                timestamps,
                offset_domain=offset.offset_domain,
                input_kind=feature_input.input_kind,
                feature_sampling_rate_hz=feature_input.sampling_rate_hz,
            )
        except AlignmentOffsetExportError as error:
            raise BoardEventSegmentationError(
                f"could not build alignment boundary axis for dataset "
                f"{recording.dataset_id}: {error}"
            ) from error
        try:
            source_labels, recording_result, reasons = (
                _prepare_success_recording_segmentation(
                    recording=recording,
                    feature_values=ring_imu,
                    timestamps_us=timestamps,
                    boundary_timestamps_us=boundary_timestamps,
                    offset=offset,
                    board_frames=board.frames,
                    board_contacts=board.contacts,
                    config=config,
                )
            )
        except _RecordingLocalSegmentationFailure as failure:
            if (
                config.recording_error_policy != "skip"
                or not _is_quarantinable_recording_failure(failure.cause)
            ):
                raise BoardEventSegmentationError(
                    "Board-event segmentation failed for dataset "
                    f"{recording.dataset_id} at {failure.stage}: {failure.cause}"
                ) from failure.cause
            segmentation_errors.append(
                _SegmentationErrorRecordingArtifacts(
                    recording=recording,
                    stage=failure.stage,
                    error_type=type(failure.cause).__name__,
                    message=str(failure.cause),
                    outcome=outcome,
                )
            )
            continue
        verification_path = build_verification_path(
            staging_directory,
            dataset_id=recording.dataset_id,
            input_kind=input_kind,
        )
        try:
            create_verification_figure(
                ring_dataframe=None,
                feature_values=feature_input.values,
                ring_timestamps_us=timestamps,
                transient_score=compute_transient_score_array(
                    feature_input.values[:, feature_input.transient_channel_indices]
                ),
                aligned_board_events=recording_result.aligned_board_events,
                labels=source_labels,
                label_skip_reasons=reasons,
                segmented_samples=recording_result.samples,
                output_path=verification_path,
                config=replace(verification_config, overwrite=True),
                user=recording.user,
                action=recording.action,
                dataset_id=recording.dataset_id,
                boundary_mode="aligned_board_events",
                alignment_offset_us=offset.boundary_offset_us,
                alignment_timestamps_us=boundary_timestamps,
                alignment_time_axis_strategy=(
                    (
                        offset.alignment_time_axis_strategy
                        if offset.offset_domain == ALIGNMENT_WORK_AXIS_DOMAIN
                        else "canonical_timestamp"
                    )
                ),
                canonical_offset_projection_success=(
                    offset.canonical_offset_projection_success
                ),
            )
        except verification_error as error:
            raise BoardEventSegmentationError(
                f"segmentation verification failed for dataset {recording.dataset_id}: {error}"
            ) from error
        if not verification_path.is_file() or verification_path.stat().st_size == 0:
            raise BoardEventSegmentationError(
                f"segmentation verification image was not written for dataset {recording.dataset_id}"
            )
        artifacts.append(
            _RecordingArtifacts(
                recording=recording,
                feature_input=feature_input,
                timestamps_us=timestamps,
                boundary_timestamps_us=boundary_timestamps,
                result=recording_result,
                labels=source_labels,
                offset=offset,
                offset_path=offset_path,
                verification_path=verification_path,
                outcome=outcome,
            )
        )
    return _AlignedRecordingArtifacts(
        processed=tuple(artifacts),
        skipped=tuple(skipped_recordings),
        segmentation_errors=tuple(segmentation_errors),
        source_recordings=ordered_recordings,
        source_recording_count=len(ordered_recordings),
    )


def build_recording_error_report_path(
    output_root: Path,
    *,
    user: str,
    action: str,
) -> Path:
    """Return the report-only state path without making a segment package."""

    _validate_identity(user=user, action=action)
    stem = f"{user}_action_{action}"
    return (
        Path(output_root)
        / "recording_errors"
        / user
        / f"action_{action}"
        / f"{stem}_segmentation_recording_errors.json"
    )


def _recording_identity(recording: Recording) -> dict[str, object]:
    return {
        "user": recording.user,
        "action": recording.action,
        "dataset_id": recording.dataset_id,
    }


def _segmentation_error_entries(
    artifacts: _AlignedRecordingArtifacts,
) -> list[dict[str, object]]:
    return [
        {
            "identity": _recording_identity(error.recording),
            "alignment_status": "SUCCESS",
            "stage": error.stage,
            "error_type": error.error_type,
            "message": error.message,
        }
        for error in artifacts.segmentation_errors
    ]


def _alignment_outcome_dependency(
    artifacts: _AlignedRecordingArtifacts,
) -> dict[str, object]:
    successful = sorted(
        [
            (artifact.recording, artifact.outcome)
            for artifact in artifacts.processed
        ]
        + [
            (error.recording, error.outcome)
            for error in artifacts.segmentation_errors
        ],
        key=lambda item: item[0].dataset_id,
    )
    return {
        "source_recording_ids": [
            _recording_identity(recording)
            for recording in artifacts.source_recordings
        ],
        "outcomes_by_status": {
            "SUCCESS": [
                {
                    "identity": _recording_identity(recording),
                    "report_sha256": sha256_file(outcome.paths.report_path),
                }
                for recording, outcome in successful
            ],
            "SKIPPED": [
                {
                    "identity": _recording_identity(skipped.recording),
                    "reason": skipped.reason,
                    "report_sha256": sha256_file(
                        skipped.outcome.paths.report_path
                    ),
                }
                for skipped in artifacts.skipped
            ],
        },
    }


def _recording_error_report(
    artifacts: _AlignedRecordingArtifacts,
    *,
    input_kind: str,
    terminal_state: str,
) -> dict[str, object]:
    if terminal_state not in {
        "completed_with_recording_errors",
        "all_recordings_error",
    }:
        raise BoardEventSegmentationError("invalid recording-error terminal state")
    if not artifacts.segmentation_errors:
        raise BoardEventSegmentationError(
            "recording-error report requires at least one segmentation error"
        )
    if terminal_state == "all_recordings_error" and artifacts.processed:
        raise BoardEventSegmentationError(
            "all_recordings_error cannot include processed recordings"
        )
    return {
        "schema_version": 1,
        "terminal_state": terminal_state,
        "input_kind": input_kind,
        "boundary_mode": "aligned_board_events",
        "source_recording_count": artifacts.source_recording_count,
        "processed_recording_count": len(artifacts.processed),
        "skipped_recording_count": len(artifacts.skipped),
        "alignment_skipped_recording_count": len(artifacts.skipped),
        "segmentation_error_recording_count": len(artifacts.segmentation_errors),
        "segmentation_errors": _segmentation_error_entries(artifacts),
        "alignment_outcome_dependency": _alignment_outcome_dependency(artifacts),
    }


def _terminal_recording_error_summary(
    artifacts: _AlignedRecordingArtifacts,
    *,
    input_kind: str,
    alignment_offset_root: Path,
    config: BoardEventSegmentationConfig,
) -> dict[str, object]:
    report = _recording_error_report(
        artifacts,
        input_kind=input_kind,
        terminal_state="all_recordings_error",
    )
    return {
        **report,
        "alignment_required": config.require_successful_alignment,
        "alignment_offset_root": str(alignment_offset_root),
        "recording_error_policy": config.recording_error_policy,
        "recording_error_policy": config.recording_error_policy,
        "published_segment_package": False,
    }


def _write_recording_error_report(
    *,
    output_root: Path,
    user: str,
    action: str,
    report: Mapping[str, object],
) -> Path:
    path = build_recording_error_report_path(output_root, user=user, action=action)
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise BoardEventSegmentationError(
            f"could not publish recording-error report: {path}: {error}"
        ) from error
    return path


def _clear_recording_error_report(
    output_root: Path,
    *,
    user: str,
    action: str,
) -> None:
    path = build_recording_error_report_path(output_root, user=user, action=action)
    if not path.exists():
        return
    if not path.is_file():
        raise BoardEventSegmentationError(
            f"recording-error report is not a regular file: {path}"
        )
    try:
        path.unlink()
    except OSError as error:
        raise BoardEventSegmentationError(
            f"could not remove stale recording-error report: {path}: {error}"
        ) from error


def _update_recording_error_report(
    *,
    output_root: Path,
    user: str,
    action: str,
    recording_artifacts: _AlignedRecordingArtifacts,
    input_kind: str,
    terminal_state: str,
) -> Path | None:
    if not recording_artifacts.segmentation_errors:
        _clear_recording_error_report(output_root, user=user, action=action)
        return None
    return _write_recording_error_report(
        output_root=output_root,
        user=user,
        action=action,
        report=_recording_error_report(
            recording_artifacts,
            input_kind=input_kind,
            terminal_state=terminal_state,
        ),
    )


def _remove_existing_aligned_output(directory: Path, *, overwrite: bool) -> None:
    if not directory.exists():
        return
    if not overwrite:
        raise BoardEventSegmentationError(
            "aligned-board-events output already exists; use overwrite=True: "
            + str(directory)
        )
    if not directory.is_dir():
        raise BoardEventSegmentationError(
            f"aligned-board-events output is not a directory: {directory}"
        )
    try:
        shutil.rmtree(directory)
    except OSError as error:
        raise BoardEventSegmentationError(
            f"could not remove replaced aligned-board-events output: {directory}: {error}"
        ) from error


def _aggregate_and_write_aligned_outputs(
    recording_artifacts: _AlignedRecordingArtifacts,
    *,
    user: str,
    action: str,
    output_directory: Path,
    alignment_offset_root: Path,
    config: BoardEventSegmentationConfig,
    gravity_config: GravityRemovalConfig | None,
    input_kind: str,
    published_output_directory: Path,
) -> BoardEventUserActionSegmentationResult:
    if not recording_artifacts.processed:
        raise BoardEventSegmentationError(
            "aligned-board-events segmentation requires at least one SUCCESS "
            "alignment outcome; all selected recordings were SKIPPED"
        )
    artifacts = recording_artifacts.processed
    all_samples: list[BoardEventSegmentedSample] = []
    manifest_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    segment_index = 0
    for artifact in artifacts:
        sample_by_label = {sample.source_label_index: sample for sample in artifact.result.samples}
        skipped_by_label = {sample.source_label_index: sample for sample in artifact.result.skipped}
        for label_index, label in enumerate(artifact.labels):
            sample = sample_by_label.get(label_index)
            skipped = skipped_by_label.get(label_index)
            manifest_rows.append(
                _manifest_row(
                    artifact,
                    label_index=label_index,
                    label=label,
                    sample=sample,
                    skipped=skipped,
                    segment_index=segment_index if sample is not None else None,
                    published_verification_path=(
                        published_output_directory / artifact.verification_path.name
                    ),
                    gravity_config=gravity_config,
                )
            )
            if sample is not None:
                all_samples.append(sample)
                segment_index += 1
        event_rows.extend(
            _board_event_rows(artifact, global_segment_start=segment_index - len(artifact.result.samples))
        )
    channel_count = artifacts[0].feature_input.channel_count
    raw_imu = (
        np.concatenate([sample.imu for sample in all_samples], axis=0)
        if all_samples
        else np.empty((0, channel_count), dtype=np.dtype(config.output_dtype))
    )
    targets = (
        np.concatenate([sample.board_event_targets for sample in all_samples], axis=0)
        if all_samples
        else np.empty((0, 4), dtype=np.bool_)
    )
    labels = np.asarray([sample.label for sample in all_samples], dtype=np.str_)
    lengths = np.asarray([sample.sample_count for sample in all_samples], dtype=np.int32)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)))
    manifest = pd.DataFrame(manifest_rows)
    events = pd.DataFrame(event_rows)
    _validate_aggregate_arrays(
        raw_imu,
        labels,
        offsets,
        lengths,
        targets,
        channel_count=channel_count,
    )
    summary = _aligned_summary(
        recording_artifacts,
        samples=all_samples,
        output_dtype=np.dtype(config.output_dtype),
        alignment_offset_root=alignment_offset_root,
        config=config,
        gravity_config=gravity_config,
        input_kind=input_kind,
    )
    stem = f"{user}_action_{action}"
    paths = _staging_output_paths(
        output_directory,
        stem=stem,
        input_kind=input_kind,
    )
    _save_npy(paths.raw_imu_path, raw_imu)
    _save_npy(paths.labels_path, labels)
    _save_npy(paths.segment_offsets_path, offsets)
    _save_npy(paths.segment_lengths_path, lengths)
    _save_npy(paths.board_event_targets_path, targets)
    paths.segments_csv_path.write_text(manifest.to_csv(index=False), encoding="utf-8")
    paths.board_events_csv_path.write_text(events.to_csv(index=False), encoding="utf-8")
    paths.summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for array in (raw_imu, labels, offsets, lengths, targets):
        array.setflags(write=False)
    return BoardEventUserActionSegmentationResult(
        user=user,
        action=action,
        raw_imu=raw_imu,
        labels=labels,
        segment_offsets=offsets,
        segment_lengths=lengths,
        board_event_targets=targets,
        manifest=manifest,
        board_events=events,
        summary=summary,
        output_paths=paths,
    )


def _load_alignment_feature_input(
    recording: Recording,
    *,
    input_kind: str,
    gravity_config: GravityRemovalConfig | None,
    spike_root: Path | None,
    expected_sampling_rate_hz: float | None,
) -> RecordingFeatureInput:
    """Load the exact feature/timestamp source used by aligned segmentation."""

    try:
        return load_recording_features(
            recording,
            input_kind=input_kind,
            gravity_config=gravity_config,
            spike_root=spike_root,
            expected_sampling_rate_hz=expected_sampling_rate_hz,
        )
    except RecordingFeatureError as error:
        raise BoardEventSegmentationError(
            f"could not load alignment feature input for dataset "
            f"{recording.dataset_id}: {error}"
        ) from error


def _validate_requested_spike_sampling_rate(
    feature_input: RecordingFeatureInput,
    *,
    expected_sampling_rate_hz: float | None,
) -> None:
    """Apply the caller-rate check to a validated SUCCESS feature input."""

    if expected_sampling_rate_hz is None:
        return
    if isinstance(expected_sampling_rate_hz, bool):
        raise BoardEventSegmentationError(
            "could not load alignment feature input for dataset "
            f"{feature_input.dataset_id}: expected sampling_rate_hz must be a "
            "finite positive number"
        )
    try:
        expected_rate = float(expected_sampling_rate_hz)
    except (TypeError, ValueError) as error:
        raise BoardEventSegmentationError(
            "could not load alignment feature input for dataset "
            f"{feature_input.dataset_id}: expected sampling_rate_hz must be a "
            "finite positive number"
        ) from error
    if not math.isfinite(expected_rate) or expected_rate <= 0.0:
        raise BoardEventSegmentationError(
            "could not load alignment feature input for dataset "
            f"{feature_input.dataset_id}: expected sampling_rate_hz must be a "
            "finite positive number"
        )
    if not math.isclose(
        feature_input.sampling_rate_hz,
        expected_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise BoardEventSegmentationError(
            "could not load alignment feature input for dataset "
            f"{feature_input.dataset_id}: SpikeIMU metadata sampling_rate_hz "
            "does not match the requested rate"
        )


def _load_gravity_removed_ring(
    recording: Recording, *, gravity_config: GravityRemovalConfig
) -> tuple[object, np.ndarray, np.ndarray]:
    """Load raw IMU unchanged or apply the configured gravity preprocessing."""

    try:
        ring = load_ring(recording)
        imu = preprocess_ring_imu(ring, config=gravity_config).imu
    except (RingLoadError, GravityRemovalError, IMUPreprocessingError) as error:
        raise BoardEventSegmentationError(
            f"could not load and preprocess Ring IMU for dataset {recording.dataset_id}: {error}"
        ) from error
    timestamps = ring.dataframe["timestamp"].to_numpy(copy=True)
    return ring, *_validated_ring_inputs(imu, timestamps)


def _labels_in_boundary_domain(
    labels: Sequence[SegmentLabel],
    *,
    canonical_timestamps_us: np.ndarray,
    boundary_timestamps_us: np.ndarray,
    offset_us: float,
    label_time_domain: str,
) -> tuple[SegmentLabel, ...]:
    if label_time_domain not in {"ring", "board"}:
        raise BoardEventSegmentationError("label_time_domain must be ring or board")
    if label_time_domain == "ring":
        if np.array_equal(canonical_timestamps_us, boundary_timestamps_us):
            values = [label.timestamp_us for label in labels]
        else:
            values = [
                float(
                    boundary_timestamps_us[
                        min(
                            int(
                                np.searchsorted(
                                    canonical_timestamps_us,
                                    label.timestamp_us,
                                    side="left",
                                )
                            ),
                            len(boundary_timestamps_us) - 1,
                        )
                    ]
                )
                for label in labels
            ]
    else:
        values = [label.timestamp_us + offset_us for label in labels]
    return tuple(
        SegmentLabel(
            timestamp_us=timestamp,
            label=label.label,
            source_line_number=label.source_line_number,
        )
        for label, timestamp in zip(labels, values, strict=True)
    )


def _labels_in_ring_domain(
    labels: Sequence[SegmentLabel], *, offset_us: float, label_time_domain: str
) -> tuple[SegmentLabel, ...]:
    """Compatibility wrapper for callers that do not have a work axis."""

    if label_time_domain == "ring":
        return tuple(labels)
    if label_time_domain != "board":
        raise BoardEventSegmentationError("label_time_domain must be ring or board")
    return tuple(
        SegmentLabel(
            timestamp_us=label.timestamp_us + offset_us,
            label=label.label,
            source_line_number=label.source_line_number,
        )
        for label in labels
    )


def _combined_label_skip_reasons(
    labels: Sequence[SegmentLabel], result: BoardEventSegmentationResult
) -> tuple[str | None, ...]:
    reasons = list(result.label_skip_reasons)
    for skipped in result.skipped:
        reasons[skipped.source_label_index] = skipped.skip_reason
    if len(reasons) != len(labels):
        raise BoardEventSegmentationError("recording result has incompatible label skip reasons")
    return tuple(reasons)


def _manifest_row(
    artifact: _RecordingArtifacts,
    *,
    label_index: int,
    label: SegmentLabel,
    sample: BoardEventSegmentedSample | None,
    skipped: SkippedBoardEventSegment | None,
    segment_index: int | None,
    published_verification_path: Path,
    gravity_config: GravityRemovalConfig | None,
) -> dict[str, object]:
    next_label = (
        artifact.labels[label_index + 1]
        if label_index + 1 < len(artifact.labels)
        else None
    )
    feature = artifact.feature_input
    is_raw = feature.input_kind == "raw-ring"
    gravity_method = (
        None if gravity_config is None else gravity_config.gravity_removal_method
    )
    base: dict[str, object] = {
        "segment_index": segment_index,
        "exported": sample is not None,
        "skip_reason": None if sample is not None else (skipped.skip_reason if skipped else "not_processed"),
        "user": artifact.recording.user,
        "action": artifact.recording.action,
        "dataset_id": artifact.recording.dataset_id,
        "source_label_index": label_index,
        "label": label.label,
        "label_timestamp_us": label.timestamp_us,
        "next_label_timestamp_us": None if next_label is None else next_label.timestamp_us,
        "alignment_offset_us": artifact.offset.offset_us,
        "work_axis_offset_us": artifact.offset.work_axis_offset_us,
        "canonical_offset_us": artifact.offset.canonical_offset_us,
        "alignment_offset_domain": artifact.offset.offset_domain,
        "canonical_offset_projection_success": artifact.offset.canonical_offset_projection_success,
        "alignment_event_coverage_ratio": artifact.offset.event_coverage_ratio,
        "ring_source_path": str(artifact.recording.ring_0_path),
        "feature_values_path": str(feature.values_path),
        "label_source_path": str(artifact.recording.timestamp_path),
        "timestamp_source_path": (
            None if feature.timestamps_path is None else str(feature.timestamps_path)
        ),
        "offset_source_path": str(artifact.offset_path),
        "segmentation_verification_path": str(published_verification_path),
        "input_kind": feature.input_kind,
        "feature_schema": feature.feature_schema,
        "channel_count": feature.channel_count,
        "channel_schema": feature.feature_schema,
        "channel_names": list(feature.channel_names),
        "units": list(feature.units),
        "transient_channel_indices": list(feature.transient_channel_indices),
        "feature_values_sha256": feature.values_sha256,
        "feature_metadata_sha256": feature.metadata_sha256,
        "timestamps_sha256": feature.timestamps_sha256,
        "acceleration_semantics": (
            _acceleration_semantics(gravity_method) if is_raw and gravity_method else None
        ),
        "acceleration_g_unit": "g" if is_raw else None,
        "acceleration_m_s2_unit": "m/s^2",
        "gyroscope_unit": "rad/s",
        "standard_gravity_m_s2": STANDARD_GRAVITY_M_S2 if is_raw else None,
        "gravity_removal_method": gravity_method,
    }
    if sample is None:
        base.update(
            {
                "first_press_timestamp_us": None,
                "last_lift_timestamp_us": None,
                "provisional_start_timestamp_us": None,
                "provisional_end_timestamp_us": None,
                "final_start_timestamp_us": None,
                "final_end_timestamp_us": None,
                "start_sample_index": None,
                "stop_sample_index_exclusive": None,
                "sample_count": None,
                "valid_touch_pair_count": (
                    0 if skipped is None else skipped.valid_touch_pair_count
                ),
                "transient_touch_pair_count": (
                    0 if skipped is None else skipped.transient_touch_pair_count
                ),
                "incomplete_touch_pair_count": (
                    0 if skipped is None else skipped.incomplete_touch_pair_count
                ),
                "crossing_touch_pair_count": (
                    0 if skipped is None else skipped.crossing_touch_pair_count
                ),
                "accepted_crossing_touch_pair_count": (
                    0 if skipped is None else skipped.accepted_crossing_touch_pair_count
                ),
                "rejected_crossing_touch_pair_count": (
                    0 if skipped is None else skipped.rejected_crossing_touch_pair_count
                ),
                "boundary_source": None,
                "boundary_collision_resolved": False,
            }
        )
        return base
    base.update(
        {
            "first_press_timestamp_us": sample.first_press_timestamp_us,
            "last_lift_timestamp_us": sample.last_lift_timestamp_us,
            "provisional_start_timestamp_us": sample.provisional_start_timestamp_us,
            "provisional_end_timestamp_us": sample.provisional_end_timestamp_us,
            "final_start_timestamp_us": sample.final_start_timestamp_us,
            "final_end_timestamp_us": sample.final_end_timestamp_us,
            "start_sample_index": sample.start_sample_index,
            "stop_sample_index_exclusive": sample.stop_sample_index_exclusive,
            "sample_count": sample.sample_count,
            "valid_touch_pair_count": sample.valid_touch_pair_count,
            "transient_touch_pair_count": sample.transient_touch_pair_count,
            "incomplete_touch_pair_count": sample.incomplete_touch_pair_count,
            "crossing_touch_pair_count": sample.crossing_touch_pair_count,
            "accepted_crossing_touch_pair_count": sample.accepted_crossing_touch_pair_count,
            "rejected_crossing_touch_pair_count": sample.rejected_crossing_touch_pair_count,
            "boundary_source": sample.boundary_source,
            "boundary_collision_resolved": sample.boundary_collision_resolved,
        }
    )
    return base


def _board_event_rows(
    artifact: _RecordingArtifacts, *, global_segment_start: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    samples = artifact.result.samples
    boundary_start = float(artifact.boundary_timestamps_us[0])
    for event in artifact.result.aligned_board_events.itertuples(index=False):
        timestamp = float(event.aligned_event_timestamp_us)
        channel = _event_target_channel(event)
        assigned: int | None = None
        boundary_role = "none"
        boundary_segment_index: int | None = None
        if channel is not None and bool(event.valid_touch):
            for local_segment_index, sample in enumerate(samples):
                if (
                    str(event.event_type) == "press"
                    and _same_touch_index(
                        event.paired_touch_index,
                        sample.first_press_paired_touch_index,
                    )
                ):
                    boundary_role = "start"
                    boundary_segment_index = global_segment_start + local_segment_index
                    break
                if (
                    str(event.event_type) == "lift"
                    and _same_touch_index(
                        event.paired_touch_index,
                        sample.last_lift_paired_touch_index,
                    )
                ):
                    boundary_role = "end"
                    boundary_segment_index = global_segment_start + local_segment_index
                    break
        if channel is not None:
            for local_segment_index, sample in enumerate(samples):
                if _optional_int(event.assigned_label_index) != sample.source_label_index:
                    continue
                boundary_start_us = (
                    sample.boundary_start_timestamp_us
                    if sample.boundary_start_timestamp_us is not None
                    else sample.final_start_timestamp_us
                )
                boundary_end_us = (
                    sample.boundary_end_timestamp_us
                    if sample.boundary_end_timestamp_us is not None
                    else sample.final_end_timestamp_us
                )
                if not boundary_start_us <= timestamp < boundary_end_us:
                    continue
                segment_times = artifact.boundary_timestamps_us[
                    sample.start_sample_index : sample.stop_sample_index_exclusive
                ]
                local_index = int(np.searchsorted(segment_times, timestamp, side="left"))
                if local_index < sample.sample_count and sample.board_event_targets[local_index, channel]:
                    assigned = global_segment_start + local_segment_index
                break
        if assigned is None and boundary_segment_index is not None:
            assigned = boundary_segment_index
        rows.append(
            {
                "user": artifact.recording.user,
                "action": artifact.recording.action,
                "dataset_id": artifact.recording.dataset_id,
                "event_index": event.event_index,
                "event_type": event.event_type,
                "paired_touch_index": event.paired_touch_index,
                "frame_timestamp_raw": event.frame_timestamp_raw,
                "aligned_event_timestamp_us": timestamp,
                "aligned_event_elapsed_s": (timestamp - boundary_start) / 1_000_000.0,
                "alignment_offset_domain": artifact.offset.offset_domain,
                "valid_touch": bool(event.valid_touch),
                "transient": bool(event.transient),
                "incomplete_touch": bool(event.incomplete_touch),
                "crossing_touch": bool(event.crossing_touch),
                "crosses_next_label_timestamp": bool(event.crosses_next_label_timestamp),
                "crossing_resolution": event.crossing_resolution,
                "assigned_label_index": _optional_int(event.assigned_label_index),
                "duration_frames": event.duration_frames,
                "source_global_frame_index": event.global_frame_index,
                "used_for_segment_boundary": boundary_role != "none",
                "boundary_role": boundary_role,
                "assigned_segment_index": assigned,
                "event_target_channel": channel if assigned is not None else None,
            }
        )
    return rows


def _event_target_channel(event: object) -> int | None:
    if bool(event.incomplete_touch) or _optional_int(event.assigned_label_index) is None:
        return None
    if not bool(event.valid_touch) and not bool(event.transient):
        return None
    return _TARGET_CHANNELS[(str(event.event_type), bool(event.transient))]


def _aligned_summary(
    recording_artifacts: _AlignedRecordingArtifacts,
    *,
    samples: Sequence[BoardEventSegmentedSample],
    output_dtype: np.dtype,
    alignment_offset_root: Path,
    config: BoardEventSegmentationConfig,
    gravity_config: GravityRemovalConfig | None,
    input_kind: str,
) -> dict[str, object]:
    artifacts = recording_artifacts.processed
    if not artifacts:
        raise BoardEventSegmentationError("aligned summary requires at least one recording")
    feature_inputs = [artifact.feature_input for artifact in artifacts]
    first_feature = feature_inputs[0]
    if any(
        feature.input_kind != input_kind
        or feature.feature_schema != first_feature.feature_schema
        or feature.channel_count != first_feature.channel_count
        or feature.channel_names != first_feature.channel_names
        or feature.units != first_feature.units
        for feature in feature_inputs
    ):
        raise BoardEventSegmentationError(
            "aligned recordings do not share one feature input schema"
        )
    if input_kind == "spike-imu":
        try:
            common_sampling_rate_hz = validate_common_feature_sampling_rate(
                feature_inputs
            )
        except RecordingFeatureError as error:
            raise BoardEventSegmentationError(str(error)) from error
    else:
        common_sampling_rate_hz = first_feature.sampling_rate_hz
    skip_counts: dict[str, int] = {}
    boundary_counts: dict[str, int] = {}
    for artifact in artifacts:
        for skipped in artifact.result.skipped:
            skip_counts[skipped.skip_reason] = skip_counts.get(skipped.skip_reason, 0) + 1
        for sample in artifact.result.samples:
            boundary_counts[sample.boundary_source] = boundary_counts.get(sample.boundary_source, 0) + 1
    pairs = [artifact.result.aligned_touch_pairs for artifact in artifacts]
    pair_table = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    verification_paths = [artifact.verification_path for artifact in artifacts]
    raw_gravity = gravity_config is not None and input_kind == "raw-ring"
    feature_provenance = [
        {
            "dataset_id": artifact.recording.dataset_id,
            "values_path": str(artifact.feature_input.values_path),
            "values_sha256": artifact.feature_input.values_sha256,
            "metadata_path": (
                None
                if artifact.feature_input.metadata_path is None
                else str(artifact.feature_input.metadata_path)
            ),
            "metadata_sha256": artifact.feature_input.metadata_sha256,
            "timestamps_sha256": artifact.feature_input.timestamps_sha256,
            "timestamps_path": (
                None
                if artifact.feature_input.timestamps_path is None
                else str(artifact.feature_input.timestamps_path)
            ),
            "sampling_rate_hz": artifact.feature_input.sampling_rate_hz,
        }
        for artifact in artifacts
    ]
    alignment_outcome_dependency = _alignment_outcome_dependency(
        recording_artifacts
    )
    return {
        "input_kind": input_kind,
        "boundary_mode": "aligned_board_events",
        "boundary_source": "aligned_board_events",
        "time_mapping": "ring_timestamp_us = board_timestamp_us + offset_us",
        "alignment_offset_domain": (
            artifacts[0].offset.offset_domain
            if len({artifact.offset.offset_domain for artifact in artifacts}) == 1
            else "mixed"
        ),
        "alignment_time_axis": {
            "strategy": artifacts[0].offset.alignment_time_axis_strategy,
            "sampling_rate_hz": artifacts[0].offset.alignment_time_axis_sampling_rate_hz,
            "sample_count": artifacts[0].offset.alignment_time_axis_sample_count,
        },
        "canonical_offset_projection_success": all(
            artifact.offset.canonical_offset_projection_success is not False
            for artifact in artifacts
        ),
        "padding_or_truncation": "disabled",
        "alignment_required": config.require_successful_alignment,
        "alignment_offset_root": str(alignment_offset_root),
        "pre_press_context_us": config.pre_press_context_us,
        "post_lift_context_us": config.post_lift_context_us,
        "source_label_count": int(sum(len(artifact.labels) for artifact in artifacts)),
        "exported_segment_count": len(samples),
        "skipped_segment_count": int(sum(len(artifact.result.skipped) for artifact in artifacts)),
        "skipped_segment_counts_by_reason": dict(sorted(skip_counts.items())),
        "valid_touch_pair_count": int(pair_table["valid_touch"].astype(bool).sum()) if not pair_table.empty else 0,
        "transient_touch_pair_count": int(pair_table["transient"].astype(bool).sum()) if not pair_table.empty else 0,
        "incomplete_touch_pair_count": int(pair_table["incomplete_touch"].astype(bool).sum()) if not pair_table.empty else 0,
        "crossing_touch_pair_count": int(pair_table["crossing_touch"].astype(bool).sum()) if not pair_table.empty else 0,
        "accepted_crossing_touch_pair_count": int(
            (pair_table["crossing_resolution"] == "accepted_before_next_press").sum()
        ) if not pair_table.empty else 0,
        "rejected_crossing_touch_pair_count": int(
            pair_table["crossing_resolution"].isin(
                ("rejected_overlaps_next_press", "rejected_next_press_missing")
            ).sum()
        ) if not pair_table.empty else 0,
        "crossing_touch_policy": config.crossing_touch_policy,
        "board_event_target_channels": ["valid_press", "valid_lift", "transient_press", "transient_lift"],
        "boundary_source_counts": dict(sorted(boundary_counts.items())),
        "recording_count": len(artifacts),
        "source_recording_count": recording_artifacts.source_recording_count,
        "processed_recording_count": len(artifacts),
        "skipped_recording_count": len(recording_artifacts.skipped),
        "alignment_skipped_recording_count": len(recording_artifacts.skipped),
        "segmentation_error_recording_count": len(
            recording_artifacts.segmentation_errors
        ),
        "segmentation_errors": _segmentation_error_entries(recording_artifacts),
        "recording_skips": [
            {
                "identity": {
                    "user": skipped.recording.user,
                    "action": skipped.recording.action,
                    "dataset_id": skipped.recording.dataset_id,
                },
                "reason": skipped.reason,
                "diagnostics": dict(skipped.diagnostics),
            }
            for skipped in recording_artifacts.skipped
        ],
        "alignment_outcome_dependency": alignment_outcome_dependency,
        "verification_image_count": len(verification_paths),
        "recordings_without_verification": [
            artifact.recording.dataset_id
            for artifact in artifacts
            if not artifact.verification_path.is_file() or artifact.verification_path.stat().st_size == 0
        ],
        "output_dtype": output_dtype.name,
        "output_schema_version": 4,
        "feature_schema": first_feature.feature_schema,
        "channel_count": first_feature.channel_count,
        "channel_names": list(first_feature.channel_names),
        "channel_units": list(first_feature.units),
        "sampling_rate_hz": common_sampling_rate_hz,
        "feature_sampling_rate_hz": common_sampling_rate_hz,
        "sample_count_by_recording": {
            str(feature.dataset_id): feature.sample_count for feature in feature_inputs
        },
        "transient_channel_indices": list(first_feature.transient_channel_indices),
        "transient_channel_names": list(first_feature.transient_channel_names),
        "event_channel_slice": [0, 15] if input_kind == "spike-imu" else None,
        "acceleration_m_s2_channel_slice": [15, 18] if input_kind == "spike-imu" else [3, 6],
        "gyroscope_channel_slice": [18, 21] if input_kind == "spike-imu" else [6, 9],
        "feature_values_sha256": [
            feature.values_sha256 for feature in feature_inputs
        ],
        "feature_metadata_sha256": [
            feature.metadata_sha256 for feature in feature_inputs
        ],
        "timestamps_sha256": [
            feature.timestamps_sha256 for feature in feature_inputs
        ],
        "feature_provenance": feature_provenance,
        "timestamps": {
            "unit": "microseconds",
            "paths": sorted(
                {
                    str(feature.timestamps_path)
                    for feature in feature_inputs
                    if feature.timestamps_path is not None
                }
            ),
            "sha256": sorted({feature.timestamps_sha256 for feature in feature_inputs}),
        },
        "alignment_input_hash_match_verified": (
            input_kind == "spike-imu"
            and all(artifact.offset.feature_values_sha256 == artifact.feature_input.values_sha256
                    and artifact.offset.feature_metadata_sha256 == artifact.feature_input.metadata_sha256
                    and artifact.offset.timestamp_sha256 == artifact.feature_input.timestamps_sha256
                    for artifact in artifacts)
        ),
        "board_event_targets_present": True,
        "units": {
            "channel_units": list(first_feature.units),
            "acceleration_x_g": "g" if input_kind == "raw-ring" else None,
            "acceleration_y_g": "g" if input_kind == "raw-ring" else None,
            "acceleration_z_g": "g" if input_kind == "raw-ring" else None,
            "acceleration_x": "m/s^2",
            "acceleration_y": "m/s^2",
            "acceleration_z": "m/s^2",
            "gyro_x": "rad/s",
            "gyro_y": "rad/s",
            "gyro_z": "rad/s",
        },
        "standard_gravity_m_s2": STANDARD_GRAVITY_M_S2 if raw_gravity else None,
        "acceleration_semantics": (
            _acceleration_semantics(gravity_config.gravity_removal_method)
            if raw_gravity
            else None
        ),
        "gravity_removal": {
            "method": (
                gravity_config.gravity_removal_method
                if raw_gravity
                else "upstream_spike_imu"
            ),
            "sampling_rate_hz": (
                None
                if not raw_gravity or gravity_config.gravity_removal_method == "raw"
                else gravity_config.sampling_rate_hz
            ),
            "low_pass_cutoff_hz": (
                None
                if not raw_gravity or gravity_config.gravity_removal_method == "raw"
                else gravity_config.low_pass_cutoff_hz
            ),
            "madgwick_beta": (
                None
                if not raw_gravity or gravity_config.gravity_removal_method == "raw"
                else gravity_config.madgwick_beta
            ),
        },
    }


def _staging_output_paths(
    output_directory: Path,
    *,
    stem: str,
    input_kind: str = "raw-ring",
) -> BoardEventSegmentationOutputPaths:
    if input_kind not in {"raw-ring", "spike-imu"}:
        raise BoardEventSegmentationError(
            "input_kind must be 'raw-ring' or 'spike-imu'"
        )
    feature_suffix = "spikeIMU" if input_kind == "spike-imu" else "rawIMU"
    return BoardEventSegmentationOutputPaths(
        output_directory=output_directory,
        raw_imu_path=output_directory / f"{stem}_{feature_suffix}.npy",
        labels_path=output_directory / f"{stem}_labels.npy",
        segment_offsets_path=output_directory / f"{stem}_segment_offsets.npy",
        segment_lengths_path=output_directory / f"{stem}_segment_lengths.npy",
        board_event_targets_path=output_directory / f"{stem}_board_event_targets.npy",
        segments_csv_path=output_directory / f"{stem}_segments.csv",
        board_events_csv_path=output_directory / f"{stem}_board_events.csv",
        summary_json_path=output_directory / f"{stem}_segmentation_summary.json",
    )


def _save_npy(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)


def _validate_aggregate_arrays(
    raw_imu: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    targets: np.ndarray,
    *,
    channel_count: int | None = None,
) -> None:
    expected_channels = (
        len(PREPROCESSED_IMU_COLUMNS) if channel_count is None else channel_count
    )
    if (
        raw_imu.ndim != 2
        or raw_imu.shape[1] != expected_channels
        or targets.shape != (len(raw_imu), 4)
    ):
        raise BoardEventSegmentationError("aligned aggregate arrays have invalid channel shapes")
    if len(labels) != len(lengths) or len(offsets) != len(labels) + 1:
        raise BoardEventSegmentationError("aligned aggregate arrays have inconsistent segment counts")
    if not np.array_equal(np.diff(offsets), lengths) or int(offsets[-1]) != len(raw_imu):
        raise BoardEventSegmentationError("aligned aggregate offsets do not delimit raw IMU")


def _publish_staging_directory(
    staging: Path, *, final_directory: Path, overwrite: bool
) -> None:
    backup: Path | None = None
    try:
        if final_directory.exists():
            if not overwrite:
                raise BoardEventSegmentationError(
                    "aligned-board-events output already exists; use overwrite=True: "
                    + str(final_directory)
                )
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{final_directory.name}.backup.",
                    dir=final_directory.parent,
                )
            )
            backup.rmdir()
            os.replace(final_directory, backup)
        os.replace(staging, final_directory)
    except OSError as error:
        if backup is not None and backup.exists() and not final_directory.exists():
            os.replace(backup, final_directory)
        raise BoardEventSegmentationError(
            f"could not publish aligned-board-events outputs: {error}"
        ) from error
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _effective_gravity_config(
    gravity_config: GravityRemovalConfig | None,
) -> GravityRemovalConfig:
    if gravity_config is None:
        return GravityRemovalConfig(
            gravity_removal_method="low-pass", strict_calibration=False
        )
    if not isinstance(gravity_config, GravityRemovalConfig):
        raise BoardEventSegmentationError("gravity_config must be GravityRemovalConfig")
    if gravity_config.gravity_removal_method not in IMU_PREPROCESSING_METHODS:
        raise BoardEventSegmentationError(
            "gravity_removal_method must be one of: "
            + ", ".join(IMU_PREPROCESSING_METHODS)
        )
    return gravity_config


def _acceleration_semantics(method: str) -> str:
    if method == "raw":
        return "measured_acceleration_with_gravity"
    if method == "xylo-rotate-and-remove-gravity":
        return "xylo_gravity_removed_acceleration"
    return "gravity_removed_linear_acceleration"


def _analyze_label_interval(
    *,
    source_label_index: int,
    label: SegmentLabel,
    next_label: SegmentLabel | None,
    ring_end_us: float,
    events: pd.DataFrame,
    touch_pairs: pd.DataFrame,
) -> LabelIntervalAnalysis:
    press = touch_pairs["aligned_press_us"]
    lift = touch_pairs["aligned_lift_us"]
    in_interval = press >= label.timestamp_us
    if next_label is None:
        in_interval &= press <= ring_end_us
    else:
        in_interval &= press < next_label.timestamp_us
    complete = ~touch_pairs["incomplete_touch"].astype(bool) & np.isfinite(lift)
    ordered = lift > press
    owned = (
        touch_pairs["assigned_label_index"].astype("Int64") == source_label_index
    ).fillna(False)
    eligible_mask = (
        owned
        & complete
        & ordered
        & touch_pairs["valid_touch"].astype(bool)
        & ~touch_pairs["transient"].astype(bool)
    )
    transient_mask = owned & complete & ordered & touch_pairs["transient"].astype(bool)
    incomplete_mask = in_interval & touch_pairs["incomplete_touch"].astype(bool)
    event_interval = events["aligned_event_timestamp_us"] >= label.timestamp_us
    if next_label is not None:
        event_interval &= events["aligned_event_timestamp_us"] < next_label.timestamp_us
    else:
        event_interval &= events["aligned_event_timestamp_us"] <= ring_end_us
    incomplete_event_count = int(
        np.count_nonzero(event_interval & events["incomplete_touch"].astype(bool))
    )
    return LabelIntervalAnalysis(
        eligible_pairs=touch_pairs.loc[eligible_mask]
        .sort_values("aligned_press_us")
        .copy(deep=True),
        valid_touch_pair_count=int(np.count_nonzero(eligible_mask)),
        transient_touch_pair_count=int(np.count_nonzero(transient_mask)),
        incomplete_touch_pair_count=max(
            int(np.count_nonzero(incomplete_mask)), incomplete_event_count
        ),
        crossing_touch_pair_count=int(
            np.count_nonzero(in_interval & touch_pairs["crosses_next_label_timestamp"].astype(bool))
        ),
        accepted_crossing_touch_pair_count=int(
            np.count_nonzero(
                in_interval
                & (touch_pairs["crossing_resolution"] == "accepted_before_next_press")
            )
        ),
        rejected_crossing_touch_pair_count=int(
            np.count_nonzero(
                in_interval
                & touch_pairs["crossing_resolution"].isin(
                    ("rejected_overlaps_next_press", "rejected_next_press_missing")
                )
            )
        ),
    )


def _resolve_window_collisions(candidates: dict[int, _CandidateWindow]) -> None:
    """Split overlapping adjacent contexts between their touch cores."""

    ordered = [candidates[index] for index in sorted(candidates)]
    for current, following in zip(ordered, ordered[1:], strict=False):
        _resolve_pair_overlap(current, following)


def _resolve_remaining_overlaps(candidates: dict[int, _CandidateWindow]) -> None:
    previous: _CandidateWindow | None = None
    for candidate in (candidates[index] for index in sorted(candidates)):
        if previous is not None and previous.final_end_us > candidate.final_start_us:
            _resolve_pair_overlap(previous, candidate)
        previous = candidate


def _resolve_pair_overlap(current: _CandidateWindow, following: _CandidateWindow) -> None:
    """Keep both event cores and split an overlapping context at its midpoint."""

    if current.final_end_us <= following.final_start_us:
        return
    if current.last_lift_us >= following.first_press_us:
        raise BoardEventSegmentationError(
            "event-guided segment cores overlap; crossing touch was not safely classified"
        )
    split_us = (current.last_lift_us + following.first_press_us) / 2.0
    current.final_end_us = min(current.final_end_us, split_us)
    following.final_start_us = max(following.final_start_us, split_us)
    current.collision_resolved = True
    following.collision_resolved = True


def _event_targets_for_segment(
    events: pd.DataFrame,
    *,
    segment_timestamps_us: np.ndarray,
    final_start_us: float,
    final_end_us: float,
    source_label_index: int,
) -> np.ndarray:
    targets = np.zeros((len(segment_timestamps_us), 4), dtype=np.bool_)
    for event in events.itertuples(index=False):
        timestamp = float(event.aligned_event_timestamp_us)
        if not final_start_us <= timestamp < final_end_us:
            continue
        if bool(event.incomplete_touch) or _optional_int(event.assigned_label_index) != source_label_index:
            continue
        transient = bool(event.transient)
        if not bool(event.valid_touch) and not transient:
            continue
        channel = _TARGET_CHANNELS[(str(event.event_type), transient)]
        local_index = int(np.searchsorted(segment_timestamps_us, timestamp, side="left"))
        if local_index < len(segment_timestamps_us):
            targets[local_index, channel] = True
    return targets


def _missing_event_reason(
    transient_count: int, incomplete_count: int, crossing_count: int
) -> str:
    if crossing_count:
        return "touch_crosses_next_label"
    if transient_count:
        return "only_transient_touches"
    if incomplete_count:
        return "incomplete_touch"
    return "no_complete_touch_pair"


def _skipped(
    label: SegmentLabel,
    index: int,
    next_label: SegmentLabel | None,
    reason: str,
    *,
    analysis: LabelIntervalAnalysis | None = None,
) -> SkippedBoardEventSegment:
    return SkippedBoardEventSegment(
        source_label_index=index,
        label=label.label,
        label_timestamp_us=label.timestamp_us,
        next_label_timestamp_us=(None if next_label is None else next_label.timestamp_us),
        skip_reason=reason,
        valid_touch_pair_count=(
            0 if analysis is None else analysis.valid_touch_pair_count
        ),
        transient_touch_pair_count=(
            0 if analysis is None else analysis.transient_touch_pair_count
        ),
        incomplete_touch_pair_count=(
            0 if analysis is None else analysis.incomplete_touch_pair_count
        ),
        crossing_touch_pair_count=(
            0 if analysis is None else analysis.crossing_touch_pair_count
        ),
        accepted_crossing_touch_pair_count=(
            0 if analysis is None else analysis.accepted_crossing_touch_pair_count
        ),
        rejected_crossing_touch_pair_count=(
            0 if analysis is None else analysis.rejected_crossing_touch_pair_count
        ),
    )


def _standardize_board_detection(
    events: pd.DataFrame, touch_pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_table = events.copy(deep=True)
    pair_table = touch_pairs.copy(deep=True)
    unpaired_lift = (event_table["event_type"] == "lift") & event_table[
        "paired_touch_index"
    ].isna()
    event_table.loc[unpaired_lift, "incomplete_touch"] = True
    event_table["crossing_touch"] = False
    pair_table["crossing_touch"] = False
    for table in (event_table, pair_table):
        table["crosses_next_label_timestamp"] = False
        table["crossing_resolution"] = "not_crossing"
        table["assigned_label_index"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
    return _validated_standard_event_tables(event_table, pair_table)


def _classify_crossing_touches(
    events: pd.DataFrame,
    touch_pairs: pd.DataFrame,
    labels: Sequence[SegmentLabel],
    *,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify crossing pairs from their press interval and next valid press.

    A crossing pair is owned by its press label only when its lift is strictly
    before the next label's first valid press.  This prevents one physical
    touch from becoming a target in both neighboring segments.
    """

    event_table = events.copy(deep=True)
    pair_table = touch_pairs.copy(deep=True)
    boundaries = np.asarray([label.timestamp_us for label in labels], dtype=np.float64)
    press = pair_table["aligned_press_us"].to_numpy(dtype=np.float64)
    lift = pair_table["aligned_lift_us"].to_numpy(dtype=np.float64)
    press_label_indices = np.searchsorted(boundaries, press, side="right") - 1
    complete_ordered = (
        ~pair_table["incomplete_touch"].astype(bool).to_numpy()
        & np.isfinite(lift)
        & (lift > press)
    )
    valid_non_transient = (
        complete_ordered
        & pair_table["valid_touch"].astype(bool).to_numpy()
        & ~pair_table["transient"].astype(bool).to_numpy()
    )
    first_valid_press: dict[int, float] = {}
    for index in range(len(labels)):
        candidates = press[(press_label_indices == index) & valid_non_transient]
        if len(candidates):
            first_valid_press[index] = float(np.min(candidates))

    crosses = np.zeros(len(pair_table), dtype=bool)
    resolutions = np.full(len(pair_table), "not_crossing", dtype=object)
    assigned = pd.Series(pd.NA, index=pair_table.index, dtype="Int64")
    for row_index, press_label_index in enumerate(press_label_indices):
        if press_label_index < 0 or press_label_index >= len(labels):
            continue
        if not complete_ordered[row_index]:
            continue
        next_label_index = press_label_index + 1
        if next_label_index >= len(labels) or lift[row_index] < boundaries[next_label_index]:
            assigned.iloc[row_index] = int(press_label_index)
            continue
        crosses[row_index] = True
        next_press = first_valid_press.get(next_label_index)
        if next_press is None:
            resolutions[row_index] = "rejected_next_press_missing"
        elif lift[row_index] < next_press:
            resolutions[row_index] = "accepted_before_next_press"
            if policy == "accept_until_next_press":
                assigned.iloc[row_index] = int(press_label_index)
        else:
            resolutions[row_index] = "rejected_overlaps_next_press"
    pair_table["crosses_next_label_timestamp"] = crosses
    pair_table["crossing_touch"] = crosses
    pair_table["crossing_resolution"] = resolutions
    pair_table["assigned_label_index"] = assigned
    pair_details = pair_table.set_index("paired_touch_index")
    event_table["crosses_next_label_timestamp"] = [
        False if pd.isna(pair_index) else bool(pair_details.loc[pair_index, "crosses_next_label_timestamp"])
        for pair_index in event_table["paired_touch_index"]
    ]
    event_table["crossing_touch"] = event_table["crosses_next_label_timestamp"].astype(bool)
    event_table["crossing_resolution"] = [
        "not_crossing" if pd.isna(pair_index) else str(pair_details.loc[pair_index, "crossing_resolution"])
        for pair_index in event_table["paired_touch_index"]
    ]
    event_table["assigned_label_index"] = pd.Series(
        [
            pd.NA if pd.isna(pair_index) else pair_details.loc[pair_index, "assigned_label_index"]
            for pair_index in event_table["paired_touch_index"]
        ],
        index=event_table.index,
        dtype="Int64",
    )
    return event_table, pair_table


def _validated_standard_event_tables(
    events: pd.DataFrame, touch_pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events, touch_pairs = _with_crossing_classification_columns(events, touch_pairs)
    _require_dataframe(events, name="Board events")
    _require_dataframe(touch_pairs, name="Board touch pairs")
    required_events = set(ALIGNED_BOARD_EVENT_COLUMNS) - {"aligned_event_timestamp_us"}
    required_pairs = set(ALIGNED_TOUCH_PAIR_COLUMNS) - {
        "aligned_press_us",
        "aligned_lift_us",
    }
    _require_columns(events, required_events, name="Board events")
    _require_columns(touch_pairs, required_pairs, name="Board touch pairs")
    event_table = events.loc[:, sorted(required_events)].copy(deep=True)
    pair_table = touch_pairs.loc[:, sorted(required_pairs)].copy(deep=True)
    _validate_event_types(event_table)
    _finite_series(event_table["frame_timestamp_raw"], name="frame_timestamp_raw")
    _finite_series(pair_table["press_timestamp_raw"], name="press_timestamp_raw")
    lift = pd.to_numeric(pair_table["lift_timestamp_raw"], errors="coerce")
    complete = ~pair_table["incomplete_touch"].astype(bool)
    if not np.isfinite(lift[complete]).all():
        raise BoardEventSegmentationError("complete touch pairs require finite lift timestamps")
    return event_table, pair_table


def _validated_aligned_event_tables(
    events: pd.DataFrame, touch_pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events, touch_pairs = _with_crossing_classification_columns(events, touch_pairs)
    required_events = set(ALIGNED_BOARD_EVENT_COLUMNS)
    required_pairs = set(ALIGNED_TOUCH_PAIR_COLUMNS)
    _require_dataframe(events, name="aligned Board events")
    _require_dataframe(touch_pairs, name="aligned Board touch pairs")
    _require_columns(events, required_events, name="aligned Board events")
    _require_columns(touch_pairs, required_pairs, name="aligned Board touch pairs")
    event_table = events.loc[:, list(ALIGNED_BOARD_EVENT_COLUMNS)].copy(deep=True)
    pair_table = touch_pairs.loc[:, list(ALIGNED_TOUCH_PAIR_COLUMNS)].copy(deep=True)
    _validate_event_types(event_table)
    _finite_series(event_table["aligned_event_timestamp_us"], name="aligned event timestamp")
    _finite_series(pair_table["aligned_press_us"], name="aligned press timestamp")
    complete = ~pair_table["incomplete_touch"].astype(bool)
    if not np.isfinite(pair_table.loc[complete, "aligned_lift_us"].to_numpy(dtype=float)).all():
        raise BoardEventSegmentationError("complete touch pairs require finite aligned lift timestamps")
    if (
        pair_table.loc[complete, "aligned_lift_us"].to_numpy(dtype=np.float64)
        <= pair_table.loc[complete, "aligned_press_us"].to_numpy(dtype=np.float64)
    ).any():
        raise BoardEventSegmentationError(
            "aligned touch pair must satisfy press_timestamp < lift_timestamp"
        )
    return event_table, pair_table


def _with_crossing_classification_columns(
    events: pd.DataFrame, touch_pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accept pre-policy event tables while normalizing new audit columns."""

    _require_dataframe(events, name="Board events")
    _require_dataframe(touch_pairs, name="Board touch pairs")
    event_table = events.copy(deep=True)
    pair_table = touch_pairs.copy(deep=True)
    for table in (event_table, pair_table):
        if "crossing_touch" not in table:
            table["crossing_touch"] = False
        if "crosses_next_label_timestamp" not in table:
            table["crosses_next_label_timestamp"] = False
        if "crossing_resolution" not in table:
            table["crossing_resolution"] = "not_crossing"
        if "assigned_label_index" not in table:
            table["assigned_label_index"] = pd.Series(
                pd.NA, index=table.index, dtype="Int64"
            )
    return event_table, pair_table


def _validated_config(config: BoardEventSegmentationConfig) -> np.dtype:
    if not isinstance(config, BoardEventSegmentationConfig):
        raise BoardEventSegmentationError("config must be BoardEventSegmentationConfig")
    if config.output_dtype not in {"float32", "float64"}:
        raise BoardEventSegmentationError("output_dtype must be float32 or float64")
    if not isinstance(config.include_last_label, bool):
        raise BoardEventSegmentationError("include_last_label must be a boolean")
    for value, name in (
        (config.pre_press_context_us, "pre_press_context_us"),
        (config.post_lift_context_us, "post_lift_context_us"),
    ):
        if _finite_float(value, name=name) < 0.0:
            raise BoardEventSegmentationError(f"{name} must be nonnegative")
    minimum = _finite_float(config.minimum_label_interval_us, name="minimum_label_interval_us")
    maximum = _finite_float(config.maximum_segment_duration_us, name="maximum_segment_duration_us")
    if minimum <= 0.0 or maximum <= 0.0 or minimum > maximum:
        raise BoardEventSegmentationError("label interval limits must be positive and ordered")
    if config.missing_event_policy != "skip":
        raise BoardEventSegmentationError("only missing_event_policy='skip' is implemented")
    if config.crossing_touch_policy not in {"accept_until_next_press", "skip"}:
        raise BoardEventSegmentationError(
            "crossing_touch_policy must be 'accept_until_next_press' or 'skip'"
        )
    if config.label_time_domain not in {"ring", "board"}:
        raise BoardEventSegmentationError("label_time_domain must be 'ring' or 'board'")
    if not isinstance(config.require_successful_alignment, bool):
        raise BoardEventSegmentationError("require_successful_alignment must be a boolean")
    if not isinstance(config.minimum_duration_frames, int) or config.minimum_duration_frames < 1:
        raise BoardEventSegmentationError("minimum_duration_frames must be a positive integer")
    if config.recording_error_policy not in {"error", "skip"}:
        raise BoardEventSegmentationError(
            "recording_error_policy must be 'error' or 'skip'"
        )
    return np.dtype(config.output_dtype)


def _validated_ring_inputs(
    ring_imu: np.ndarray, ring_timestamps_us: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    try:
        imu = np.asarray(ring_imu, dtype=np.float64)
        timestamps = np.asarray(ring_timestamps_us, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise BoardEventSegmentationError("Ring IMU and timestamps must be numeric") from error
    if imu.ndim != 2 or imu.shape[1] == 0 or len(imu) == 0:
        raise BoardEventSegmentationError(
            "feature values must have nonempty shape (M, C)"
        )
    if timestamps.ndim != 1 or len(timestamps) != len(imu):
        raise BoardEventSegmentationError("Ring timestamps must be a length-M vector")
    if not np.isfinite(imu).all() or not np.isfinite(timestamps).all():
        raise BoardEventSegmentationError("Ring IMU and timestamps must be finite")
    if np.any(np.diff(timestamps) < 0.0):
        raise BoardEventSegmentationError("Ring timestamps must be nondecreasing")
    return imu.copy(), timestamps.copy()


def _resolve_feature_arguments(
    *,
    feature_values: np.ndarray | None,
    timestamps_us: np.ndarray | None,
    ring_imu: np.ndarray | None,
    ring_timestamps_us: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Accept generic feature names while retaining the raw-ring API aliases."""

    if feature_values is not None and ring_imu is not None:
        raise BoardEventSegmentationError(
            "feature_values and ring_imu cannot both be supplied"
        )
    if timestamps_us is not None and ring_timestamps_us is not None:
        raise BoardEventSegmentationError(
            "timestamps_us and ring_timestamps_us cannot both be supplied"
        )
    values = feature_values if feature_values is not None else ring_imu
    timestamps = timestamps_us if timestamps_us is not None else ring_timestamps_us
    if values is None or timestamps is None:
        raise BoardEventSegmentationError(
            "feature_values and timestamps_us are required"
        )
    return values, timestamps


def _validated_labels(labels: Sequence[SegmentLabel]) -> tuple[SegmentLabel, ...]:
    markers = tuple(labels)
    if not markers:
        raise BoardEventSegmentationError("at least one timestamp label is required")
    previous: float | None = None
    for marker in markers:
        if not isinstance(marker, SegmentLabel):
            raise BoardEventSegmentationError("labels must contain SegmentLabel values")
        timestamp = _finite_float(marker.timestamp_us, name="label timestamp")
        if not marker.label:
            raise BoardEventSegmentationError("label text must be nonempty")
        if previous is not None and timestamp <= previous:
            raise BoardEventSegmentationError("label timestamps must be strictly increasing")
        previous = timestamp
    return markers


def _validate_label_range(labels: Sequence[SegmentLabel], timestamps: np.ndarray) -> None:
    for marker in labels:
        if marker.timestamp_us < timestamps[0] or marker.timestamp_us > timestamps[-1]:
            raise BoardEventSegmentationError("label timestamp is outside the Ring range")


def _validated_boundary_timestamps(
    values: np.ndarray | None,
    *,
    canonical_timestamps_us: np.ndarray,
) -> np.ndarray:
    if values is None:
        return canonical_timestamps_us.copy()
    try:
        boundary = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise BoardEventSegmentationError(
            "boundary_timestamps_us must be numeric"
        ) from error
    if (
        boundary.ndim != 1
        or boundary.size != canonical_timestamps_us.size
        or not np.isfinite(boundary).all()
        or not np.all(np.diff(boundary) >= 0.0)
    ):
        raise BoardEventSegmentationError(
            "boundary_timestamps_us must be finite and nondecreasing, "
            "and match the canonical sample count"
        )
    return boundary.copy()


def _canonical_point_for_boundary(
    timestamp_us: float,
    *,
    canonical_timestamps_us: np.ndarray,
    boundary_timestamps_us: np.ndarray,
) -> float:
    index = int(np.searchsorted(boundary_timestamps_us, timestamp_us, side="left"))
    index = min(max(index, 0), len(canonical_timestamps_us) - 1)
    return float(canonical_timestamps_us[index])


def _canonical_start_for_boundary(
    timestamp_us: float,
    *,
    canonical_timestamps_us: np.ndarray,
    boundary_timestamps_us: np.ndarray,
) -> float:
    return _canonical_point_for_boundary(
        timestamp_us,
        canonical_timestamps_us=canonical_timestamps_us,
        boundary_timestamps_us=boundary_timestamps_us,
    )


def _canonical_end_for_boundary(
    timestamp_us: float,
    *,
    canonical_timestamps_us: np.ndarray,
    boundary_timestamps_us: np.ndarray,
) -> float:
    index = int(np.searchsorted(boundary_timestamps_us, timestamp_us, side="left"))
    if index >= len(canonical_timestamps_us):
        return float(canonical_timestamps_us[-1])
    return float(canonical_timestamps_us[index])


def _validated_board_frames(frames: pd.DataFrame) -> pd.DataFrame:
    _require_dataframe(frames, name="Board frames")
    _require_columns(
        frames,
        {"global_frame_index", "frame_timestamp_raw", "chunk_index", "contact_count"},
        name="Board frames",
    )
    if frames.empty:
        raise BoardEventSegmentationError("Board frames must not be empty")
    result = frames.copy(deep=True)
    _finite_series(result["frame_timestamp_raw"], name="frame_timestamp_raw")
    return result


def _validated_board_contacts(contacts: pd.DataFrame) -> pd.DataFrame:
    _require_dataframe(contacts, name="Board contacts")
    _require_columns(contacts, {"global_frame_index"}, name="Board contacts")
    return contacts.copy(deep=True)


def _validate_final_samples(samples: Sequence[BoardEventSegmentedSample]) -> None:
    previous_stop: int | None = None
    for sample in samples:
        if not 0 <= sample.start_sample_index < sample.stop_sample_index_exclusive:
            raise BoardEventSegmentationError("final segmentation window is empty")
        if previous_stop is not None and previous_stop > sample.start_sample_index:
            raise BoardEventSegmentationError("final segmentation windows overlap")
        previous_stop = sample.stop_sample_index_exclusive


def _require_dataframe(value: object, *, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise BoardEventSegmentationError(f"{name} must be a pandas DataFrame")


def _require_columns(dataframe: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise BoardEventSegmentationError(f"{name} is missing column(s): {', '.join(missing)}")


def _validate_event_types(events: pd.DataFrame) -> None:
    if not events["event_type"].isin(("press", "lift")).all():
        raise BoardEventSegmentationError("Board event_type values must be press or lift")


def _finite_series(values: pd.Series, *, name: str) -> None:
    try:
        numeric = values.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise BoardEventSegmentationError(f"{name} must be numeric") from error
    if not np.isfinite(numeric).all():
        raise BoardEventSegmentationError(f"{name} must be finite")


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise BoardEventSegmentationError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise BoardEventSegmentationError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise BoardEventSegmentationError(f"{name} must be finite")
    return number


def _optional_int(value: object) -> int | None:
    """Return an optional integer touch identifier without coercing missing data."""

    if pd.isna(value):
        return None
    return int(value)


def _same_touch_index(value: object, expected: int | None) -> bool:
    """Compare a pandas nullable touch ID to an optional stored identifier."""

    return expected is not None and not pd.isna(value) and int(value) == expected


def _validate_identity(*, user: str, action: str) -> None:
    for name, value in (("user", user), ("action", action)):
        if (
            not isinstance(value, str)
            or not value
            or value.strip() != value
            or "/" in value
            or "\\" in value
        ):
            raise BoardEventSegmentationError(
                f"{name} must be a nonempty safe path component"
            )
