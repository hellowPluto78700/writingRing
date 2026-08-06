#!/usr/bin/env python3
"""Align one Ring recording to Board events and export verified artifacts."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--show-smoothed-transient", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    _configure_runtime_environment(effective_argv)
    args = build_parser().parse_args(effective_argv)
    from writingring.alignment_io import (
        AlignmentOffsetExportError,
        build_alignment_offset_path,
        build_alignment_time_axes,
        extract_alignment_offset,
        project_alignment_offset_to_canonical,
        sha256_array,
        write_alignment_offset_txt,
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
        align_events_to_transient_peaks,
        compute_transient_score,
        compute_transient_score_array,
        detect_board_events,
        detect_transient_peak_regions,
        select_board_interval_from_presses,
    )
    from writingring.recording_features import (
        RecordingFeatureError,
        load_recording_features,
    )
    from writingring.ring_loader import RingLoadError, load_ring
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
        feature_input = None
        if args.input_kind == "spike-imu":
            feature_input = load_recording_features(
                recording,
                input_kind="spike-imu",
                spike_root=args.spike_root,
            )
            canonical_timestamps = feature_input.timestamps_us
            alignment_sampling_rate_hz = feature_input.sampling_rate_hz
            timestamp_source_sha256 = feature_input.timestamps_sha256
            transient_score = compute_transient_score_array(
                feature_input.values[:, feature_input.transient_channel_indices]
            )
            verification_ring_dataframe = pd.DataFrame(
                index=range(feature_input.sample_count)
            )
        else:
            ring_data = load_ring(recording)
            canonical_timestamps = ring_data.dataframe["timestamp"].to_numpy(
                dtype=float, copy=True
            )
            alignment_sampling_rate_hz = ring_data.validation.inferred_sampling_rate_hz
            timestamp_source_sha256 = sha256_array(canonical_timestamps)
            transient_score = compute_transient_score(
                ring_data.dataframe,
                signal_columns=("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"),
            )
            verification_ring_dataframe = ring_data.dataframe
        time_axes = build_alignment_time_axes(
            canonical_timestamps,
            input_kind=args.input_kind,
            sampling_rate_hz=alignment_sampling_rate_hz,
        )
        canonical_timestamps = time_axes.canonical_timestamps_us
        alignment_timestamps = time_axes.work_timestamps_us
        alignment_time_axis = dict(time_axes.metadata)
        detection = detect_board_events(board_data.frames)
        interval = select_board_interval_from_presses(
            board_data.frames,
            board_data.contacts,
            detection,
        )
        peak_regions = detect_transient_peak_regions(
            transient_score, alignment_timestamps
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
        projection_error: str | None = None
        if result.success:
            try:
                projection = project_alignment_offset_to_canonical(
                    result,
                    canonical_timestamps,
                    alignment_timestamps,
                    sampling_rate_hz=time_axes.sampling_rate_hz,
                    # Raw Ring's legacy endpoint axis can span timestamp
                    # quantisation/duplicate runs.  Keep the projection
                    # diagnostic warning at half a sample, but use the
                    # empirically validated raw-recording failure guard so
                    # existing recordings are not rejected before Board
                    # segmentation can consume their canonical offset.
                    failure_threshold_samples=(
                        20.0 if args.input_kind == "raw-ring" else 1.5
                    ),
                )
            except AlignmentOffsetExportError as error:
                projection_error = str(error)
        report_path = _report_path(
            args.report_output_root,
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
        )
        alignment_report = result.report | {
            "recording": {
                "user": args.user,
                "action": args.action,
                "dataset_id": args.dataset_id,
            },
            "warnings": list(result.warnings),
            "best_offset_us": (
                result.best_offset_us
                if projection is None
                else projection.canonical_offset_us
            ),
            "work_axis_offset_us": result.best_offset_us,
            "alignment_success": result.success and projection is not None,
            "time_mapping": "ring_timestamp_us = board_timestamp_us + offset_us",
            "label_time_domain": args.label_time_domain,
            "timestamp_source": {
                "sha256": timestamp_source_sha256,
                "ordering": alignment_time_axis["ordering"],
                "duplicate_step_count": alignment_time_axis[
                    "duplicate_step_count"
                ],
            },
            "alignment_time_axis": alignment_time_axis,
        }
        alignment_report["alignment_time_axis"] = {
            **alignment_time_axis,
            "work_axis_offset_us": result.best_offset_us,
        }
        if projection is None:
            alignment_report["canonical_offset_projection"] = {
                "success": False,
                "error": projection_error
                or "alignment did not produce a successful work-axis result",
            }
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
        _write_report(report_path, alignment_report, overwrite=args.overwrite_report)
        if not result.success or projection is None:
            raise AlignmentFailureError(
                projection_error
                or "alignment did not succeed; offset and verification outputs were not created"
            )
        offset = extract_alignment_offset(
            result,
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
            projection=projection,
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
            )
        offset = replace(
            offset,
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
        offset_path = _offset_path_for_recording(
            recording_directory=recording.ring_0_path.parent,
            output_root=args.offset_output_root,
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
            build_structured_path=build_alignment_offset_path,
        )
        write_alignment_offset_txt(
            offset, output_path=offset_path, overwrite=args.overwrite_offset
        )
        label_path = resolve_alignment_label_path(
            recording.ring_0_path.parent,
            dataset_id=args.dataset_id,
            explicit_label_path=args.label_path,
        )
        labels = load_alignment_labels(label_path)
        verification_path = build_alignment_verification_path(
            args.verification_output_root,
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
        )
        verification = create_alignment_verification_figure(
            ring_dataframe=verification_ring_dataframe,
            ring_timestamps_us=alignment_timestamps,
            transient_score=transient_score,
            board_events=interval.events,
            alignment_result=result,
            labels=labels,
            label_source_path=label_path,
            output_path=verification_path,
            config=AlignmentVerificationConfig(
                verification_start_s=args.verification_start_seconds,
                label_time_domain=args.label_time_domain,
                show_smoothed_transient=args.show_smoothed_transient,
                overwrite=args.overwrite_verification,
            ),
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
        )
        _write_report(
            report_path,
            alignment_report
            | {
                "offset_output_path": str(offset_path),
                "verification": {
                    "output_path": str(verification.output_path),
                    "displayed_start_s": verification.displayed_start_s,
                    "displayed_stop_s": verification.displayed_stop_s,
                    "label_time_domain": args.label_time_domain,
                    "warnings": list(verification.warnings),
                },
            },
            overwrite=True,
        )
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
    print(f"Verification image: {verification.output_path}")
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


if __name__ == "__main__":
    raise SystemExit(main())
