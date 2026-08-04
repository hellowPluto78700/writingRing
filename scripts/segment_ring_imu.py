#!/usr/bin/env python3
"""Remove Ring gravity, then export label or aligned-Board IMU segments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

import numpy as np


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "writingring-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "writingring-cache")
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without loading any recordings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--boundary-mode",
        choices=("label", "aligned-board-events"),
        default="label",
        help="segment boundaries: timestamp labels (default) or aligned Board events",
    )
    parser.add_argument(
        "--output-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--exclude-last-label",
        action="store_true",
        help="omit each recording's final label instead of extending it to Ring end",
    )
    parser.add_argument("--sampling-rate", type=float, default=200.0)
    parser.add_argument(
        "--gravity-removal-method",
        choices=("raw", "low-pass", "madgwick"),
        default="low-pass",
        help=(
            "IMU preprocessing before segmentation: raw (no gravity removal), "
            "low-pass, or Madgwick (default: low-pass)"
        ),
    )
    parser.add_argument(
        "--low-pass-cutoff-hz",
        type=float,
        default=0.2,
        help="second-order Butterworth cutoff in Hz (default: 0.2)",
    )
    parser.add_argument(
        "--madgwick-beta",
        type=float,
        default=0.1,
        help="Madgwick sensor-fusion gain (default: 0.1)",
    )
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="allow gravity output when stationary-calibration checks fail",
    )
    parser.add_argument(
        "--alignment-offset-root",
        type=Path,
        help="root containing validated Ring--Board offset TXT files (aligned mode only)",
    )
    parser.add_argument(
        "--pre-press-context-seconds",
        type=float,
        help="context retained before the first valid Board press (aligned mode; default: 0.2)",
    )
    parser.add_argument(
        "--post-lift-context-seconds",
        type=float,
        help="context retained after the last valid Board lift (aligned mode; default: 0.2)",
    )
    parser.add_argument(
        "--missing-event-policy",
        choices=("skip",),
        help="policy for a label interval without a complete valid touch (aligned mode; default: skip)",
    )
    parser.add_argument(
        "--crossing-touch-policy",
        choices=("accept_until_next_press", "skip"),
        help=(
            "crossing-touch policy (aligned mode; default: accept until the "
            "next label's first valid press)"
        ),
    )
    parser.add_argument(
        "--verification-panel-seconds",
        type=float,
        help="full-recording verification panel duration (aligned mode; default: 10)",
    )
    parser.add_argument(
        "--verification-dpi",
        type=int,
        help="verification PNG resolution (aligned mode; default: 200)",
    )
    parser.add_argument(
        "--overwrite-verification",
        action="store_true",
        help="allow replacement of an existing aligned verification output set",
    )
    parser.add_argument(
        "--write-label-verification",
        action="store_true",
        help="write full-recording verification PNGs for label mode",
    )
    parser.add_argument(
        "--overlay-aligned-board-events",
        action="store_true",
        help="overlay aligned Board events in label verification only",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace the complete output set"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run segmentation and report the deterministic output paths."""

    args = build_parser().parse_args(argv)
    try:
        _validate_mode_arguments(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    from writingring.gravity import GravityRemovalConfig

    gravity_config = GravityRemovalConfig(
        sampling_rate_hz=args.sampling_rate,
        gravity_removal_method=args.gravity_removal_method,
        low_pass_cutoff_hz=args.low_pass_cutoff_hz,
        madgwick_beta=args.madgwick_beta,
        strict_calibration=(
            args.gravity_removal_method == "madgwick" and not args.provisional
        ),
    )
    output_root = (
        args.output_root
        if args.output_root is not None
        else _default_output_root(
            args.gravity_removal_method,
            boundary_mode=args.boundary_mode,
        )
    )
    if args.boundary_mode == "label":
        from writingring.board_loader import BoardLoadError
        from writingring.segmentation import SegmentationConfig, segment_user_action

        try:
            label_config = SegmentationConfig(
                output_dtype=args.output_dtype,
                include_last_label=not args.exclude_last_label,
            )
            result = segment_user_action(
                data_root=args.data_root,
                user=args.user,
                action=args.action,
                output_root=output_root,
                config=label_config,
                gravity_config=gravity_config,
                overwrite=args.overwrite or args.overwrite_verification,
            )
        except (OSError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(
            "Exported "
            f"{result.summary['segment_count']} variable-length segments with "
            f"{result.raw_imu.shape[0]} total IMU samples."
        )
        if args.write_label_verification:
            try:
                verification_count = _write_label_verifications(
                    data_root=args.data_root,
                    user=args.user,
                    action=args.action,
                    output_directory=result.output_paths.raw_imu_path.parent,
                    gravity_config=gravity_config,
                    segmentation_config=label_config,
                    panel_duration_s=_verification_panel_seconds(args),
                    output_dpi=_verification_dpi(args),
                    overwrite=args.overwrite or args.overwrite_verification,
                    overlay_aligned_board_events=args.overlay_aligned_board_events,
                    alignment_offset_root=args.alignment_offset_root,
                )
            except (OSError, ValueError, BoardLoadError) as error:
                print(f"error: {error}", file=sys.stderr)
                return 2
            print(f"Label verification images: {verification_count}")
        _print_common_outputs(result.output_paths, args.gravity_removal_method)
        return 0

    from writingring.board_event_segmentation import (
        BoardEventSegmentationConfig,
        segment_user_action_by_aligned_board_events,
    )
    from writingring.segmentation_verification import SegmentationVerificationConfig

    try:
        result = segment_user_action_by_aligned_board_events(
            data_root=args.data_root,
            user=args.user,
            action=args.action,
            output_root=output_root,
            alignment_offset_root=args.alignment_offset_root,
            config=BoardEventSegmentationConfig(
                output_dtype=args.output_dtype,
                include_last_label=not args.exclude_last_label,
                pre_press_context_us=_seconds_to_us(args.pre_press_context_seconds),
                post_lift_context_us=_seconds_to_us(args.post_lift_context_seconds),
                missing_event_policy=args.missing_event_policy or "skip",
                crossing_touch_policy=(
                    args.crossing_touch_policy or "accept_until_next_press"
                ),
            ),
            verification_config=SegmentationVerificationConfig(
                panel_duration_s=(
                    _verification_panel_seconds(args)
                ),
                output_dpi=(
                    _verification_dpi(args)
                ),
                overwrite=args.overwrite_verification,
            ),
            gravity_config=gravity_config,
            overwrite=args.overwrite or args.overwrite_verification,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        "Exported "
        f"{result.summary['exported_segment_count']} Board-guided variable-length "
        f"segments with {result.raw_imu.shape[0]} total IMU samples."
    )
    _print_common_outputs(result.output_paths, args.gravity_removal_method)
    print(f"Board event targets: {result.output_paths.board_event_targets_path}")
    print(f"Board event audit: {result.output_paths.board_events_csv_path}")
    print(
        "Verification images: "
        f"{result.summary['verification_image_count']} / "
        f"{result.summary['recording_count']}"
    )
    return 0


def _validate_mode_arguments(args: argparse.Namespace) -> None:
    """Reject mode-specific options instead of silently ignoring them."""

    aligned_values = (
        args.pre_press_context_seconds,
        args.post_lift_context_seconds,
        args.missing_event_policy,
        args.crossing_touch_policy,
    )
    if args.boundary_mode == "label":
        if args.gravity_removal_method == "raw":
            raise ValueError(
                "--gravity-removal-method raw is supported only with "
                "--boundary-mode aligned-board-events"
            )
        if any(value is not None for value in aligned_values):
            raise ValueError(
                "aligned-board-events options require "
                "--boundary-mode aligned-board-events"
            )
        if args.overlay_aligned_board_events and not args.write_label_verification:
            raise ValueError(
                "--overlay-aligned-board-events requires --write-label-verification"
            )
        if args.overlay_aligned_board_events and args.alignment_offset_root is None:
            raise ValueError(
                "--overlay-aligned-board-events requires --alignment-offset-root"
            )
        if (
            args.alignment_offset_root is not None
            and not args.overlay_aligned_board_events
        ):
            raise ValueError(
                "--alignment-offset-root in label mode requires "
                "--overlay-aligned-board-events"
            )
        if (
            (args.verification_panel_seconds is not None or args.verification_dpi is not None)
            and not args.write_label_verification
        ):
            raise ValueError(
                "verification panel options in label mode require "
                "--write-label-verification"
            )
        if args.overwrite_verification and not args.write_label_verification:
            raise ValueError(
                "--overwrite-verification in label mode requires "
                "--write-label-verification"
            )
        return
    if args.write_label_verification or args.overlay_aligned_board_events:
        raise ValueError(
            "label verification options require --boundary-mode label"
        )
    if args.alignment_offset_root is None:
        raise ValueError(
            "--alignment-offset-root is required for "
            "--boundary-mode aligned-board-events"
        )


def _seconds_to_us(value: float | None) -> float:
    """Convert an optional CLI duration to the stored timestamp unit."""

    seconds = 0.2 if value is None else value
    return seconds * 1_000_000.0


def _verification_panel_seconds(args: argparse.Namespace) -> float:
    """Return the default or explicitly supplied panel duration."""

    return 10.0 if args.verification_panel_seconds is None else args.verification_panel_seconds


def _verification_dpi(args: argparse.Namespace) -> int:
    """Return the default or explicitly supplied verification resolution."""

    return 200 if args.verification_dpi is None else args.verification_dpi


def _write_label_verifications(
    *,
    data_root: Path,
    user: str,
    action: str,
    output_directory: Path,
    gravity_config: object,
    segmentation_config: object,
    panel_duration_s: float,
    output_dpi: int,
    overwrite: bool,
    overlay_aligned_board_events: bool,
    alignment_offset_root: Path | None,
) -> int:
    """Render optional label-mode figures without changing label boundaries."""

    from writingring.alignment_io import build_alignment_offset_path
    from writingring.board_event_segmentation import (
        align_board_event_tables,
        prepare_complete_board_events,
        read_recording_alignment_offset,
    )
    from writingring.discovery import discover_recordings
    from writingring.event_alignment import compute_transient_score
    from writingring.gravity import process_ring_gravity
    from writingring.board_loader import load_board
    from writingring.ring_loader import load_ring
    from writingring.segmentation import (
        SegmentationConfig,
        label_start_skip_reasons,
        load_timestamp_labels,
        segment_recording_by_labels,
    )
    from writingring.segmentation_verification import (
        SegmentationVerificationConfig,
        build_segmentation_verification_path,
        create_segmentation_verification_figure,
    )

    recordings = sorted(
        (
            recording
            for recording in discover_recordings(data_root)
            if recording.user == user and recording.action == action
        ),
        key=lambda recording: recording.dataset_id,
    )
    if not recordings:
        raise ValueError(f"no recordings found for user={user!r}, action={action!r}")
    if not isinstance(segmentation_config, SegmentationConfig):
        raise ValueError("segmentation_config must be SegmentationConfig")
    verification = SegmentationVerificationConfig(
        panel_duration_s=panel_duration_s,
        output_dpi=output_dpi,
        overwrite=overwrite,
    )
    for recording in recordings:
        if recording.timestamp_path is None:
            raise ValueError(f"dataset {recording.dataset_id} is missing timestamp labels")
        ring = load_ring(recording)
        gravity = process_ring_gravity(ring, config=gravity_config)
        timestamps = ring.dataframe["timestamp"].to_numpy(copy=True)
        imu = np.column_stack(
            (gravity.linear_acceleration_body, gravity.angular_velocity_body_rad_s)
        )
        labels = load_timestamp_labels(recording.timestamp_path)
        samples = segment_recording_by_labels(
            ring_imu=imu,
            ring_timestamps_us=timestamps,
            labels=labels,
            config=segmentation_config,
        )
        reasons = label_start_skip_reasons(
            labels,
            ring_end_timestamp_us=float(timestamps[-1]),
            config=segmentation_config,
        )
        events = None
        offset_us = None
        if overlay_aligned_board_events:
            assert alignment_offset_root is not None
            offset = read_recording_alignment_offset(
                build_alignment_offset_path(
                    alignment_offset_root,
                    user=recording.user,
                    action=recording.action,
                    dataset_id=recording.dataset_id,
                ),
                recording=recording,
            )
            board = load_board(recording)
            prepared = prepare_complete_board_events(board.frames, board.contacts)
            events, _ = align_board_event_tables(
                prepared.events, prepared.touch_pairs, offset=offset
            )
            offset_us = offset.offset_us
        create_segmentation_verification_figure(
            ring_dataframe=ring.dataframe,
            ring_timestamps_us=timestamps,
            transient_score=compute_transient_score(
                ring.dataframe,
                signal_columns=("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"),
            ),
            aligned_board_events=events,
            labels=labels,
            label_skip_reasons=reasons,
            segmented_samples=samples,
            output_path=build_segmentation_verification_path(
                output_directory, dataset_id=recording.dataset_id
            ),
            config=verification,
            user=recording.user,
            action=recording.action,
            dataset_id=recording.dataset_id,
            boundary_mode="label",
            alignment_offset_us=offset_us,
        )
    return len(recordings)


def _print_common_outputs(output_paths: object, gravity_method: str) -> None:
    """Print common user/action output paths for either result type."""

    print(f"Gravity removal: {gravity_method}")
    print(f"Raw IMU: {output_paths.raw_imu_path}")
    print(f"Labels: {output_paths.labels_path}")
    print(f"Segment offsets: {output_paths.segment_offsets_path}")
    print(f"Segment lengths: {output_paths.segment_lengths_path}")
    print(f"Manifest: {output_paths.segments_csv_path}")
    print(f"Summary: {output_paths.summary_json_path}")


def _default_output_root(method: str, *, boundary_mode: str = "label") -> Path:
    """Return the method- and boundary-specific default output root."""

    if boundary_mode == "aligned-board-events":
        names = {
            "raw": "boardAssistSegmentedIMU_RawIMU",
            "low-pass": "boardAssistSegmentedIMU_LowPassFilterin",
            "madgwick": "boardAssistSegmentedIMU_Madgwick",
        }
    else:
        names = {
            "low-pass": "segmentedIMU_LowPassFiltering",
            "madgwick": "segmentedIMU_Madgwick",
        }
    return Path("outputs") / names[method]


if __name__ == "__main__":
    raise SystemExit(main())
