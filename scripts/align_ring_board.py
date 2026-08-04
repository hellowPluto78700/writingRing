#!/usr/bin/env python3
"""Align one Ring recording to Board events and export verified artifacts."""

from __future__ import annotations

import argparse
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
        extract_alignment_offset,
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
        detect_board_events,
        detect_transient_peak_regions,
        select_board_interval_from_presses,
    )
    from writingring.ring_loader import RingLoadError, load_ring
    from writingring.selection import RecordingSelectionError, select_recording

    try:
        recording = select_recording(
            discover_recordings(args.data_root),
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
        )
        ring_data = load_ring(recording)
        board_data = load_board(recording)
        timestamps = _reconstructed_ring_timestamps(ring_data.dataframe["timestamp"].to_numpy())
        transient_score = compute_transient_score(
            ring_data.dataframe,
            signal_columns=("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"),
        )
        detection = detect_board_events(board_data.frames)
        interval = select_board_interval_from_presses(
            board_data.frames,
            board_data.contacts,
            detection,
        )
        peak_regions = detect_transient_peak_regions(transient_score, timestamps)
        result = align_events_to_transient_peaks(
            interval.events,
            interval.touch_pairs,
            peak_regions,
            ring_timestamp_range=(float(timestamps[0]), float(timestamps[-1])),
        )
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
            "best_offset_us": result.best_offset_us,
            "alignment_success": result.success,
            "time_mapping": "ring_timestamp_us = board_timestamp_us + offset_us",
            "label_time_domain": args.label_time_domain,
        }
        _write_report(report_path, alignment_report, overwrite=args.overwrite_report)
        if not result.success:
            raise AlignmentFailureError(
                "alignment did not succeed; offset and verification outputs were not created"
            )
        offset = extract_alignment_offset(
            result, user=args.user, action=args.action, dataset_id=args.dataset_id
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
            ring_dataframe=ring_data.dataframe,
            ring_timestamps_us=timestamps,
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


def _reconstructed_ring_timestamps(raw_timestamps: object) -> object:
    import numpy as np

    timestamps = np.asarray(raw_timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 3 or not np.isfinite(timestamps).all():
        raise ValueError("Ring timestamps must be a finite vector with at least three samples")
    duration_s = (timestamps[-1] - timestamps[0]) / 1_000_000.0
    if duration_s <= 0.0:
        raise ValueError("Ring timestamp endpoint duration must be positive")
    rate_hz = (len(timestamps) - 1) / duration_s
    return timestamps[0] + np.arange(len(timestamps), dtype=np.float64) * 1_000_000.0 / rate_hz


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
