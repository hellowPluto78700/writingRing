#!/usr/bin/env python3
"""Generate the standard plots and JSON summary for one recording.

The four expected output filenames may be replaced when the caller explicitly
reuses an output directory. Other files in that directory are left untouched.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from typing import Any, Sequence


def _configure_runtime_environment(argv: Sequence[str]) -> bool:
    """Select Agg before importing pyplot when the effective mode is no-show."""

    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    preliminary_args, _ = preliminary.parse_known_args(argv)
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "writingring-matplotlib"),
    )
    os.environ.setdefault(
        "XDG_CACHE_HOME",
        str(Path(tempfile.gettempdir()) / "writingring-cache"),
    )
    if not preliminary_args.show:
        import matplotlib

        matplotlib.use("Agg", force=True)
    return bool(preliminary_args.show)


def _load_runtime() -> SimpleNamespace:
    """Import plotting-dependent components only after backend selection."""

    import matplotlib.pyplot as plt

    from writingring.board_loader import BoardLoadError, load_board
    from writingring.discovery import DiscoveryError, discover_recordings
    from writingring.inspection import build_recording_summary, write_json
    from writingring.plotting import (
        BOARD_FRAME_INDEX,
        BOARD_RAW_TIMESTAMP,
        RING_INFERRED_TIME,
        RING_SAMPLE_INDEX,
        PlottingError,
        plot_board_force_over_time,
        plot_ring_imu,
        plot_touch_trajectory,
    )
    from writingring.ring_loader import RingLoadError, load_ring
    from writingring.selection import (
        RecordingSelectionError,
        select_recording,
    )

    return SimpleNamespace(
        plt=plt,
        discover_recordings=discover_recordings,
        select_recording=select_recording,
        load_ring=load_ring,
        load_board=load_board,
        build_recording_summary=build_recording_summary,
        write_json=write_json,
        plot_ring_imu=plot_ring_imu,
        plot_touch_trajectory=plot_touch_trajectory,
        plot_board_force_over_time=plot_board_force_over_time,
        ring_time_choices=(RING_SAMPLE_INDEX, RING_INFERRED_TIME),
        board_time_choices=(BOARD_FRAME_INDEX, BOARD_RAW_TIMESTAMP),
        ring_time_default=RING_INFERRED_TIME,
        board_time_default=BOARD_FRAME_INDEX,
        expected_errors=(
            DiscoveryError,
            RecordingSelectionError,
            RingLoadError,
            BoardLoadError,
            PlottingError,
        ),
    )


def build_parser(runtime: SimpleNamespace) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The four expected output files may be overwritten in the "
            "explicitly selected output directory; unrelated files are kept."
        ),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ring-time-axis",
        choices=runtime.ring_time_choices,
        default=runtime.ring_time_default,
    )
    parser.add_argument(
        "--board-time-axis",
        choices=runtime.board_time_choices,
        default=runtime.board_time_default,
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    _configure_runtime_environment(effective_argv)
    runtime = _load_runtime()
    args = build_parser(runtime).parse_args(effective_argv)
    figures: list[Any] = []

    try:
        output_dir = args.output_dir
        if output_dir.exists() and not output_dir.is_dir():
            raise OSError(f"output path is not a directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        recording = runtime.select_recording(
            runtime.discover_recordings(args.data_root),
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
        )
        ring_data = runtime.load_ring(recording)
        board_data = runtime.load_board(recording)

        output_paths = {
            "ring_imu": output_dir / "ring_imu.png",
            "touch_trajectory": output_dir / "touch_trajectory.png",
            "board_force_time": output_dir / "board_force_time.png",
            "summary": output_dir / "summary.json",
        }
        figures.append(
            runtime.plot_ring_imu(
                ring_data,
                time_axis=args.ring_time_axis,
                output_path=output_paths["ring_imu"],
                show=args.show,
            )
        )
        figures.append(
            runtime.plot_touch_trajectory(
                board_data,
                output_path=output_paths["touch_trajectory"],
                show=args.show,
            )
        )
        figures.append(
            runtime.plot_board_force_over_time(
                board_data,
                time_axis=args.board_time_axis,
                output_path=output_paths["board_force_time"],
                show=args.show,
            )
        )

        summary = runtime.build_recording_summary(
            recording,
            ring_data,
            board_data,
        )
        summary["plotting"] = {
            "ring_time_axis": args.ring_time_axis,
            "board_time_axis": args.board_time_axis,
            "show": args.show,
            "generated_output_paths": output_paths,
            "overwrite_policy": (
                "Only ring_imu.png, touch_trajectory.png, "
                "board_force_time.png, and summary.json may be replaced in "
                "this explicitly selected output directory."
            ),
        }
        runtime.write_json(output_paths["summary"], summary)

        for name in (
            "ring_imu",
            "touch_trajectory",
            "board_force_time",
            "summary",
        ):
            print(f"{name}: {output_paths[name]}")
    except (
        *runtime.expected_errors,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        if not args.show:
            for figure in figures:
                runtime.plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
