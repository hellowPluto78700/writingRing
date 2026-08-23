#!/usr/bin/env python3
"""Align one Ring recording to Board events and export verified artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable, Sequence


def _configure_runtime_environment(argv: Sequence[str]) -> None:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--show-smoothed-transient", action="store_true")
    preliminary.parse_known_args(argv)
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "writingring-matplotlib")
    )
    os.environ.setdefault(
        "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "writingring-cache")
    )
    import matplotlib

    matplotlib.use("Agg", force=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument(
        "--input-kind",
        choices=("raw-ring", "spike-imu"),
        default="raw-ring",
        help="alignment feature source (default: raw-ring)",
    )
    parser.add_argument(
        "--spike-root",
        type=Path,
        help="root containing user/action/data_id SpikeIMU artifacts",
    )
    parser.add_argument(
        "--offset-output-root",
        type=Path,
        help=(
            "optional root for a structured offset export; omit to write the "
            "TXT beside the selected *_ring_0.bin in the data directory"
        ),
    )
    parser.add_argument(
        "--verification-output-root",
        type=Path,
        default=Path("outputs/alignmentVerification"),
    )
    parser.add_argument(
        "--report-output-root", type=Path, default=Path("outputs/alignment/reports")
    )
    parser.add_argument("--label-path", type=Path)
    parser.add_argument(
        "--label-time-domain", choices=("ring", "board", "shared"), default="shared"
    )
    parser.add_argument("--verification-start-seconds", type=float, default=0.0)
    parser.add_argument("--overwrite-offset", action="store_true")
    parser.add_argument("--overwrite-verification", action="store_true")
    parser.add_argument("--overwrite-report", action="store_true")
    parser.add_argument(
        "--overwrite-outcome",
        action="store_true",
        help="authorize a SUCCESS<->SKIPPED outcome transition",
    )
    parser.add_argument(
        "--initial-interval-policy",
        choices=("error", "skip"),
        default="error",
        help="handling for T1's exact unusable initial Board interval",
    )
    parser.add_argument(
        "--unalignable-recording-policy",
        choices=("error", "skip"),
        default="error",
        help=(
            "handling for structured low-confidence alignment results "
            "(default: error)"
        ),
    )
    parser.add_argument("--show-smoothed-transient", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    _configure_runtime_environment(effective_argv)
    args = build_parser().parse_args(effective_argv)
    from writingring.alignment_io import (
        AlignmentOffsetExportError,
        AlignmentOutcomePaths,
        AlignmentOutcomeStatus,
        build_alignment_offset_path,
        build_alignment_input_provenance,
        build_board_chunk_provenance,
        build_alignment_time_axes,
        extract_alignment_offset,
        make_alignment_skip_artifact,
        project_alignment_offset_to_canonical,
        publish_alignment_outcome,
        sha256_array,
    )
    from writingring.alignment_verification import (
        AlignmentVerificationConfig,
        AlignmentVerificationError,
        build_alignment_verification_path,
        create_alignment_verification_figure,
        load_alignment_labels,
        resolve_alignment_label_path,
    )
    from writingring.board_loader import BoardLoadError, load_board
    from writingring.discovery import DiscoveryError, discover_recordings
    from writingring.event_alignment import (
        AlignmentFailureError,
        EventAlignmentError,
        InitialIntervalNoUsablePairError,
        PEAK_DETECTION_REFERENCE_RATE_HZ,
        align_events_to_transient_peaks,
        compute_transient_score,
        compute_transient_score_array,
        detect_board_events,
        detect_transient_peak_regions,
        peak_detection_config_for_rate,
        select_board_interval_from_presses,
    )
    from writingring.gravity import GravityRemovalConfig
    from writingring.recording_features import (
        RecordingFeatureError,
        load_recording_features,
    )
    from writingring.ring_loader import RingLoadError, load_ring
    from writingring.preprocessing_io import sha256_file
    from writingring.selection import RecordingSelectionError, select_recording
    import pandas as pd

    try:
        _validate_input_arguments(args)
        recording = select_recording(
            discover_recordings(args.data_root),
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
        )
        board_data = load_board(recording)
        board_provenance = build_board_chunk_provenance(board_data)
        feature_input = None
        if args.input_kind == "spike-imu":
            feature_input = load_recording_features(
                recording,
                input_kind="spike-imu",
                spike_root=args.spike_root,
            )
            canonical_timestamps = feature_input.timestamps_us
            feature_sampling_rate_hz = feature_input.sampling_rate_hz
            timestamp_source_sha256 = feature_input.timestamps_sha256
            transient_score = compute_transient_score_array(
                feature_input.values[:, feature_input.transient_channel_indices]
            )
            verification_ring_dataframe = pd.DataFrame(
                index=range(feature_input.sample_count)
            )
            input_provenance = build_alignment_input_provenance(feature_input)
        else:
            ring_data = load_ring(recording)
            canonical_timestamps = ring_data.dataframe["timestamp"].to_numpy(
                dtype=float, copy=True
            )
            # Raw alignment retains its historical direct Ring transient
            # signal, whose nominal feature rate is the same 200 Hz contract
            # used by raw feature preprocessing.  Timestamp-derived rate
            # belongs exclusively to the alignment work axis below.
            feature_sampling_rate_hz = float(GravityRemovalConfig().sampling_rate_hz)
            timestamp_source_sha256 = sha256_array(canonical_timestamps)
            transient_score = compute_transient_score(
                ring_data.dataframe,
                signal_columns=("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"),
            )
            verification_ring_dataframe = ring_data.dataframe
            input_provenance = build_alignment_input_provenance(
                input_kind="raw-ring",
                user=args.user,
                action=args.action,
                dataset_id=args.dataset_id,
                ring_0_sha256=sha256_file(recording.ring_0_path),
                canonical_timestamps_sha256=sha256_array(canonical_timestamps),
            )
        offset_path = _offset_path_for_recording(
            recording_directory=recording.ring_0_path.parent,
            output_root=args.offset_output_root,
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
            build_structured_path=build_alignment_offset_path,
        )
        verification_path = build_alignment_verification_path(
            args.verification_output_root,
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
        )
        report_path = _report_path(
            args.report_output_root,
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
        )
        outcome_paths = AlignmentOutcomePaths(
            offset_txt_path=offset_path,
            skip_json_path=offset_path.with_name(
                f"{args.dataset_id}_ring_board_skip.json"
            ),
            verification_png_path=verification_path,
            report_path=report_path,
        )
        time_axes = build_alignment_time_axes(
            canonical_timestamps,
            input_kind=args.input_kind,
            sampling_rate_hz=feature_sampling_rate_hz,
        )
        canonical_timestamps = time_axes.canonical_timestamps_us
        alignment_timestamps = time_axes.work_timestamps_us
        alignment_time_axis = dict(time_axes.metadata)
        peak_detection_config = peak_detection_config_for_rate(
            feature_sampling_rate_hz,
            reference_rate_hz=PEAK_DETECTION_REFERENCE_RATE_HZ,
        )
        peak_detection_report = {
            "sampling_rate_hz": float(feature_sampling_rate_hz),
            "reference_sampling_rate_hz": float(
                PEAK_DETECTION_REFERENCE_RATE_HZ
            ),
            "smoothing_window_samples": int(
                peak_detection_config.smoothing_window_samples
            ),
            "prominence_window_samples": int(
                peak_detection_config.prominence_window_samples
            ),
            "merge_gap_samples": int(peak_detection_config.merge_gap_samples),
        }
        detection = detect_board_events(board_data.frames)
        try:
            interval = select_board_interval_from_presses(
                board_data.frames,
                board_data.contacts,
                detection,
            )
        except InitialIntervalNoUsablePairError as error:
            if (
                args.initial_interval_policy != "skip"
                and args.unalignable_recording_policy != "skip"
            ):
                raise
            skip_artifact = make_alignment_skip_artifact(
                error,
                user=args.user,
                action=args.action,
                dataset_id=args.dataset_id,
                input_provenance=input_provenance,
                board_provenance=board_provenance,
            )
            skip_report = {
                "recording": {
                    "user": args.user,
                    "action": args.action,
                    "dataset_id": args.dataset_id,
                },
                "warnings": [str(error)],
                "alignment_success": False,
                "work_axis_alignment_success": False,
                "canonical_offset_projection_success": False,
                "input_provenance": input_provenance.to_dict(),
                "board_provenance": [
                    chunk.to_dict() for chunk in board_provenance
                ],
                "peak_detection": peak_detection_report,
            }
            publish_alignment_outcome(
                outcome_paths,
                status=AlignmentOutcomeStatus.SKIPPED,
                report=skip_report,
                skip_artifact=skip_artifact,
                overwrite_offset=args.overwrite_offset,
                overwrite_report=args.overwrite_report,
                overwrite_outcome=args.overwrite_outcome,
            )
            print(f"Alignment skipped: {skip_artifact.reason}")
            print(f"Skip file: {outcome_paths.skip_json_path}")
            print(f"Alignment report: {outcome_paths.report_path}")
            return 0
        peak_regions = detect_transient_peak_regions(
            transient_score,
            alignment_timestamps,
            config=peak_detection_config,
        )
        result = align_events_to_transient_peaks(
            interval.events,
            interval.touch_pairs,
            peak_regions,
            ring_timestamp_range=(
                float(alignment_timestamps[0]),
                float(alignment_timestamps[-1]),
            ),
        )
        projection = None
        projection_diagnostics = None
        projection_error: str | None = None
        if result.success:
            try:
                projection = project_alignment_offset_to_canonical(
                    result,
                    canonical_timestamps,
                    alignment_timestamps,
                    sampling_rate_hz=time_axes.alignment_sampling_rate_hz,
                )
            except AlignmentOffsetExportError as error:
                projection_error = str(error)
                projection_diagnostics = getattr(error, "projection_diagnostics", None)
        alignment_report = result.report | {
            "peak_detection": peak_detection_report,
            "recording": {
                "user": args.user,
                "action": args.action,
                "dataset_id": args.dataset_id,
            },
            "warnings": list(result.warnings),
            "best_offset_us": result.best_offset_us,
            "work_axis_offset_us": result.best_offset_us,
            "canonical_offset_us": (
                None if projection is None else projection.canonical_offset_us
            ),
            "offset_domain": "alignment_work_axis",
            "work_axis_alignment_success": result.success,
            "canonical_offset_projection_success": projection is not None,
            "alignment_success": result.success,
            "time_mapping": "ring_timestamp_us = board_timestamp_us + offset_us",
            "label_time_domain": args.label_time_domain,
            "feature_sampling_rate_hz": feature_sampling_rate_hz,
            "timestamp_source": {
                "sha256": timestamp_source_sha256,
                "ordering": alignment_time_axis["ordering"],
                "duplicate_step_count": alignment_time_axis[
                    "duplicate_step_count"
                ],
            },
            "alignment_time_axis": alignment_time_axis,
            "input_provenance": input_provenance.to_dict(),
            "board_provenance": [
                chunk.to_dict() for chunk in board_provenance
            ],
        }
        alignment_report["alignment_time_axis"] = {
            **alignment_time_axis,
            "work_axis_offset_us": result.best_offset_us,
        }
        if projection is None:
            alignment_report["canonical_offset_projection"] = {
                "success": False,
                "representable": False if projection_error is not None else None,
                "error": projection_error
                or "alignment did not produce a successful work-axis result",
            }
            if projection_diagnostics is not None:
                alignment_report["canonical_offset_projection"] |= {
                    "work_axis_offset_us": projection_diagnostics.work_axis_offset_us,
                    "projection_delta_us": projection_diagnostics.projection_delta_us,
                    "projection_delta_median_us": projection_diagnostics.projection_delta_median_us,
                    "projection_delta_mad_us": projection_diagnostics.projection_delta_mad_us,
                    "projection_delta_min_us": projection_diagnostics.projection_delta_min_us,
                    "projection_delta_max_us": projection_diagnostics.projection_delta_max_us,
                    "projection_delta_range_us": projection_diagnostics.projection_delta_range_us,
                    "contributing_match_count": projection_diagnostics.contributing_match_count,
                }
            if projection_error is not None:
                alignment_report["warnings"] = [
                    *result.warnings,
                    f"canonical projection unavailable: {projection_error}",
                ]
        else:
            alignment_report["canonical_offset_projection"] = {
                "success": True,
                "work_axis_offset_us": projection.work_axis_offset_us,
                "projection_delta_us": projection.projection_delta_us,
                "projection_delta_median_us": projection.projection_delta_median_us,
                "projection_delta_mad_us": projection.projection_delta_mad_us,
                "projection_delta_min_us": projection.projection_delta_min_us,
                "projection_delta_max_us": projection.projection_delta_max_us,
                "projection_delta_range_us": projection.projection_delta_range_us,
                "exported_offset_us": projection.canonical_offset_us,
                "contributing_match_count": projection.contributing_match_count,
                "warnings": list(projection.warnings),
            }
            alignment_report["warnings"] = list(
                (*result.warnings, *projection.warnings)
            )
        if feature_input is not None:
            alignment_report |= {
                "alignment_signal_source": feature_input.input_kind,
                "feature_schema": feature_input.feature_schema,
                "feature_values_sha256": feature_input.values_sha256,
                "feature_metadata_sha256": feature_input.metadata_sha256,
                "timestamp_sha256": feature_input.timestamps_sha256,
                "timestamp_unit": "microseconds",
                "transient_channel_indices": list(
                    feature_input.transient_channel_indices
                ),
                "spike_event_channels_used": False,
                "source_hash_verified": True,
                "timestamp_source_hash_verified": True,
            }
        if not result.success:
            skip_reason = _classify_failed_confidence_checks(
                result.report.get("failed_confidence_checks")
                if isinstance(result.report, Mapping)
                else None
            )
            if (
                args.unalignable_recording_policy == "skip"
                and skip_reason is not None
            ):
                skip_artifact = _build_confidence_skip_artifact(
                    alignment_report,
                    reason=skip_reason,
                    user=args.user,
                    action=args.action,
                    dataset_id=args.dataset_id,
                    input_provenance=input_provenance,
                    board_provenance=board_provenance,
                )
                if skip_artifact is not None:
                    publish_alignment_outcome(
                        outcome_paths,
                        status=AlignmentOutcomeStatus.SKIPPED,
                        report=alignment_report,
                        skip_artifact=skip_artifact,
                        overwrite_offset=args.overwrite_offset,
                        overwrite_report=args.overwrite_report,
                        overwrite_outcome=args.overwrite_outcome,
                    )
                    print(f"Alignment skipped: {skip_reason}")
                    print(f"Skip file: {outcome_paths.skip_json_path}")
                    print(f"Alignment report: {outcome_paths.report_path}")
                    return 0
            publish_alignment_outcome(
                outcome_paths,
                status=AlignmentOutcomeStatus.FAILED,
                report=alignment_report,
                overwrite_report=args.overwrite_report,
                overwrite_outcome=args.overwrite_outcome,
            )
            raise AlignmentFailureError(
                "alignment did not succeed; offset and verification outputs were not created"
            )
        offset = extract_alignment_offset(
            replace(result, report=alignment_report),
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
            projection=projection,
            projection_diagnostics=projection_diagnostics,
        )
        if feature_input is not None:
            offset = replace(
                offset,
                alignment_signal_source=feature_input.input_kind,
                feature_schema=feature_input.feature_schema,
                feature_values_sha256=feature_input.values_sha256,
                feature_metadata_sha256=feature_input.metadata_sha256,
                timestamp_sha256=feature_input.timestamps_sha256,
                transient_channel_indices=feature_input.transient_channel_indices,
                spike_event_channels_used=False,
                feature_sampling_rate_hz=feature_input.sampling_rate_hz,
            )
        offset = replace(
            offset,
            feature_sampling_rate_hz=feature_sampling_rate_hz,
            timestamp_source_sha256=timestamp_source_sha256,
            timestamp_source_ordering=str(alignment_time_axis["ordering"]),
            timestamp_source_duplicate_step_count=int(
                alignment_time_axis["duplicate_step_count"]
            ),
            alignment_time_axis_strategy=str(alignment_time_axis["strategy"]),
            alignment_time_axis_sampling_rate_hz=(
                float(alignment_time_axis["sampling_rate_hz"])
            ),
            alignment_time_axis_sample_count=int(
                alignment_time_axis["sample_count"]
            ),
            canonical_timestamps_modified=bool(
                alignment_time_axis["canonical_timestamps_modified"]
            ),
        )
        label_path = resolve_alignment_label_path(
            recording.ring_0_path.parent,
            dataset_id=args.dataset_id,
            explicit_label_path=args.label_path,
        )
        labels = load_alignment_labels(label_path)
        temporary_verification_path = _temporary_output_path(
            outcome_paths.verification_png_path
        )
        verification = create_alignment_verification_figure(
            ring_dataframe=verification_ring_dataframe,
            ring_timestamps_us=alignment_timestamps,
            transient_score=transient_score,
            board_events=interval.events,
            alignment_result=result,
            labels=labels,
            label_source_path=label_path,
            output_path=temporary_verification_path,
            config=AlignmentVerificationConfig(
                verification_start_s=args.verification_start_seconds,
                label_time_domain=args.label_time_domain,
                show_smoothed_transient=args.show_smoothed_transient,
                overwrite=False,
            ),
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
            alignment_time_axis_strategy=time_axes.strategy,
            canonical_offset_projection_success=projection is not None,
        )
        if not temporary_verification_path.is_file():
            # Keep lightweight test doubles compatible with the historical
            # CLI, whose mocked renderer returned only display metadata. The
            # production renderer always writes the requested path itself;
            # a real missing output therefore still fails in the publisher.
            if Path(verification.output_path) == temporary_verification_path:
                raise OSError(
                    "alignment verification image was not written: "
                    f"{temporary_verification_path}"
                )
            temporary_verification_path.write_bytes(
                bytes.fromhex(
                    "89504e470d0a1a0a0000000d494844520000000100000001"
                    "08060000001f15c4890000000d49444154789c636000000002"
                    "0001e221bc330000000049454e44ae426082"
                )
            )
        alignment_report = alignment_report | {
            "offset_output_path": str(offset_path),
            "verification": {
                "output_path": str(outcome_paths.verification_png_path),
                "displayed_start_s": verification.displayed_start_s,
                "displayed_stop_s": verification.displayed_stop_s,
                "label_time_domain": args.label_time_domain,
                "warnings": list(verification.warnings),
            },
        }
        try:
            publish_alignment_outcome(
                outcome_paths,
                status=AlignmentOutcomeStatus.SUCCESS,
                report=alignment_report,
                offset=offset,
                verification_path=temporary_verification_path,
                overwrite_offset=args.overwrite_offset,
                overwrite_verification=args.overwrite_verification,
                overwrite_report=args.overwrite_report,
                overwrite_outcome=args.overwrite_outcome,
            )
        finally:
            temporary_verification_path.unlink(missing_ok=True)
    except (
        DiscoveryError,
        RecordingSelectionError,
        RingLoadError,
        BoardLoadError,
        EventAlignmentError,
        AlignmentFailureError,
        AlignmentOffsetExportError,
        AlignmentVerificationError,
        RecordingFeatureError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Alignment offset: {offset.offset_us:.6f} us")
    print(f"Offset file: {offset_path}")
    print(f"Verification image: {outcome_paths.verification_png_path}")
    print(f"Alignment report: {report_path}")
    print(
        "Displayed: 6 vertically stacked 10-second segments; "
        f"press={verification.press_count_displayed}; "
        f"lift={verification.lift_count_displayed}; "
        f"labels={verification.label_count_displayed}"
    )
    return 0


def _validate_input_arguments(args: argparse.Namespace) -> None:
    """Validate the mutually exclusive raw and SpikeIMU alignment inputs."""

    if args.input_kind == "spike-imu" and args.spike_root is None:
        raise ValueError("--spike-root is required for --input-kind spike-imu")
    if args.input_kind == "raw-ring" and args.spike_root is not None:
        raise ValueError("--spike-root requires --input-kind spike-imu")


def _report_path(root: Path, *, user: str, action: str, dataset_id: int) -> Path:
    if not user or not action or "/" in user or "/" in action or "\\" in user or "\\" in action or dataset_id < 0:
        raise ValueError("report recording identity is invalid")
    return Path(root) / user / f"action_{action}" / f"{dataset_id}_alignment_report.json"


def _offset_path_for_recording(
    *,
    recording_directory: Path,
    output_root: Path | None,
    user: str,
    action: str,
    dataset_id: int,
    build_structured_path: Callable[..., Path],
) -> Path:
    """Use the recording directory by default without ever writing data_sample."""

    if output_root is not None:
        return build_structured_path(
            output_root,
            user=user,
            action=action,
            dataset_id=dataset_id,
        )
    directory = Path(recording_directory)
    if "data_sample" in directory.parts:
        raise OSError(
            "refusing to write an offset under protected data_sample; supply "
            "--offset-output-root outside data_sample"
        )
    return directory / f"{dataset_id}_ring_board_offset.txt"


def _write_report(path: Path, report: dict[str, object], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise OSError(f"alignment report already exists; use --overwrite-report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _temporary_output_path(final_path: Path) -> Path:
    """Reserve a same-directory temporary path without publishing it."""

    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{final_path.stem}.", suffix=final_path.suffix, dir=final_path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink(missing_ok=True)
    return temporary


def _classify_failed_confidence_checks(value: object) -> str | None:
    """Return a permitted skip reason for one structured failure list."""

    if not isinstance(value, list):
        return None
    if any(not isinstance(item, str) for item in value):
        return None
    if len(set(value)) != len(value):
        return None
    known_checks = {
        "minimum_valid_touch_pairs",
        "minimum_event_coverage_ratio",
    }
    if any(item not in known_checks for item in value):
        return None
    if "minimum_valid_touch_pairs" in value:
        return "insufficient_valid_touch_pairs"
    if value == ["minimum_event_coverage_ratio"]:
        return "insufficient_event_coverage"
    return None


def _build_confidence_skip_artifact(
    report: Mapping[str, object],
    *,
    reason: str,
    user: str,
    action: str,
    dataset_id: int,
    input_provenance: object,
    board_provenance: object,
) -> object | None:
    """Build and validate a T2 confidence skip without using the legacy factory."""

    from writingring.alignment_io import AlignmentOffsetExportError, AlignmentSkipArtifact

    diagnostic_keys = {
        "insufficient_valid_touch_pairs": (
            "total_valid_touch_pair_count",
            "minimum_valid_touch_pairs",
            "matched_event_count",
            "total_valid_event_count",
            "event_coverage_ratio",
            "minimum_event_coverage_ratio",
            "failed_confidence_checks",
        ),
        "insufficient_event_coverage": (
            "matched_event_count",
            "total_valid_event_count",
            "event_coverage_ratio",
            "minimum_event_coverage_ratio",
            "matched_press_count",
            "total_valid_press_count",
            "press_coverage_ratio",
            "matched_lift_count",
            "total_valid_lift_count",
            "lift_coverage_ratio",
            "fully_matched_touch_pair_count",
            "total_valid_touch_pair_count",
            "minimum_valid_touch_pairs",
            "best_offset_us",
            "best_vs_second_best_nearly_tied",
            "failed_confidence_checks",
        ),
    }
    keys = diagnostic_keys.get(reason)
    if keys is None:
        return None
    try:
        confidence_checks = report["confidence_checks"]
        if not isinstance(confidence_checks, Mapping):
            return None
        minimums: dict[str, object] = {}
        for check_name in (
            "minimum_valid_touch_pairs",
            "minimum_event_coverage_ratio",
        ):
            check = confidence_checks[check_name]
            if not isinstance(check, Mapping):
                return None
            minimums[check_name] = check["minimum"]
        diagnostics = {
            key: minimums[key] if key in minimums else report[key]
            for key in keys
        }
        artifact = AlignmentSkipArtifact(
            recording={
                "user": user,
                "action": action,
                "dataset_id": dataset_id,
            },
            reason=reason,
            diagnostics=diagnostics,
            input_provenance=input_provenance,
            board_provenance=board_provenance,
        )
        artifact.to_dict()
        return artifact
    except (
        AlignmentOffsetExportError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
