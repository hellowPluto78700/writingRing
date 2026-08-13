#!/usr/bin/env python3
"""Plot unsynchronized Ring transient score, raw labels, and raw Board transitions.

The plot intentionally performs no Ring<->Board synchronization.

Rules:
- Ring transient score uses raw/canonical Ring timestamps.
- Labels use their original timestamp.txt timestamps.
- Board press/lift uses raw frame_timestamp_raw directly.
- No alignment offset is read or applied.
- No Board press/lift pairing is performed.
- No Board transient/valid-touch classification is performed.
- Every plotted timestamp is converted to seconds relative to the raw Ring
  start timestamp only for readability:

      seconds = (timestamp_us - ring_start_us) / 1_000_000

  The same origin subtraction is applied to Ring, labels, and Board, so their
  original unsynchronized relative offsets are preserved.
- The figure is split into fixed-duration rows, default 10 seconds per row.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

import numpy as np


def _configure_matplotlib() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "writingring-matplotlib"),
    )
    os.environ.setdefault(
        "XDG_CACHE_HOME",
        str(Path(tempfile.gettempdir()) / "writingring-cache"),
    )
    import matplotlib

    matplotlib.use("Agg", force=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--action", default="0")
    parser.add_argument(
        "--dataset-id",
        "--recording",
        dest="dataset_id",
        type=int,
        required=True,
        help="recording dataset ID",
    )
    parser.add_argument(
        "--input-kind",
        choices=("raw-ring", "spike-imu"),
        default="raw-ring",
        help=(
            "source used only to compute the transient score; "
            "timestamps remain canonical/raw Ring timestamps"
        ),
    )
    parser.add_argument(
        "--spike-root",
        type=Path,
        help="required only for --input-kind spike-imu",
    )
    parser.add_argument(
        "--panel-seconds",
        type=float,
        default=10.0,
        help="duration of each row in seconds (default: 10)",
    )
    parser.add_argument(
        "--xlim-source",
        choices=("union", "ring"),
        default="union",
        help=(
            "union: include raw Board/label timestamps outside the Ring range; "
            "ring: restrict rows to the Ring range"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output PNG; default: "
            "outputs/diagnostics/unsynced_timeline/"
            "<user>_action_<action>_<dataset>_unsynced_10s.png"
        ),
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.dataset_id < 0:
        raise ValueError("--dataset-id/--recording must be nonnegative")
    if not math.isfinite(args.panel_seconds) or args.panel_seconds <= 0.0:
        raise ValueError("--panel-seconds must be a finite positive number")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.input_kind == "spike-imu" and args.spike_root is None:
        raise ValueError("--spike-root is required for --input-kind spike-imu")


def _raw_board_transitions(board_frames):
    """Detect raw occupancy transitions only; do not pair or classify touches."""
    import pandas as pd

    required = {"frame_timestamp_raw", "contact_count", "global_frame_index"}
    missing = sorted(required - set(board_frames.columns))
    if missing:
        raise ValueError(
            "Board frames are missing required columns: " + ", ".join(missing)
        )

    if board_frames.empty:
        return pd.DataFrame(
            columns=(
                "event_index",
                "event_type",
                "global_frame_index",
                "frame_timestamp_raw",
            )
        )

    contact_count = board_frames["contact_count"].to_numpy(dtype=np.int64)
    occupied = contact_count > 0
    occupied_i8 = occupied.astype(np.int8)

    # No synthetic event at the first frame.
    transition = np.diff(occupied_i8, prepend=occupied_i8[0])
    positions = np.flatnonzero(transition != 0)

    rows = []
    for event_index, position in enumerate(positions):
        frame = board_frames.iloc[int(position)]
        rows.append(
            {
                "event_index": event_index,
                "event_type": "press" if transition[position] == 1 else "lift",
                "global_frame_index": int(frame["global_frame_index"]),
                "frame_timestamp_raw": float(frame["frame_timestamp_raw"]),
            }
        )

    return pd.DataFrame(rows)


def _seconds_from_ring_start(values_us, ring_start_us: float) -> np.ndarray:
    values = np.asarray(values_us, dtype=np.float64)
    return (values - ring_start_us) / 1_000_000.0


def _panel_limits(
    *,
    ring_seconds: np.ndarray,
    label_seconds: np.ndarray,
    board_seconds: np.ndarray,
    panel_seconds: float,
    source: str,
) -> tuple[float, int]:
    if source == "ring":
        minimum = float(ring_seconds[0])
        maximum = float(ring_seconds[-1])
    else:
        parts = [ring_seconds]
        if label_seconds.size:
            parts.append(label_seconds)
        if board_seconds.size:
            parts.append(board_seconds)
        all_times = np.concatenate(parts)
        minimum = float(np.min(all_times))
        maximum = float(np.max(all_times))

    start = math.floor(minimum / panel_seconds) * panel_seconds
    stop = math.ceil(maximum / panel_seconds) * panel_seconds
    if stop <= start:
        stop = start + panel_seconds

    panel_count = max(1, int(math.ceil((stop - start) / panel_seconds)))
    return start, panel_count


def _load_transient_source(recording, args):
    from writingring.event_alignment import (
        compute_transient_score,
        compute_transient_score_array,
    )

    if args.input_kind == "spike-imu":
        from writingring.recording_features import load_recording_features

        feature = load_recording_features(
            recording,
            input_kind="spike-imu",
            spike_root=args.spike_root,
        )
        timestamps_us = np.asarray(feature.timestamps_us, dtype=np.float64)
        transient_score = compute_transient_score_array(
            feature.values[:, feature.transient_channel_indices]
        )
        source_description = (
            "SpikeIMU transient channels "
            f"{tuple(feature.transient_channel_indices)}"
        )
        return timestamps_us, transient_score, source_description

    from writingring.ring_loader import load_ring

    ring = load_ring(recording)
    timestamps_us = ring.dataframe["timestamp"].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    transient_score = compute_transient_score(
        ring.dataframe,
        signal_columns=(
            "acc_x",
            "acc_y",
            "acc_z",
            "gyr_x",
            "gyr_y",
            "gyr_z",
        ),
    )
    return timestamps_us, transient_score, "raw Ring six-axis transient"


def _default_output(args: argparse.Namespace) -> Path:
    return (
        Path("outputs")
        / "diagnostics"
        / "unsynced_timeline"
        / (
            f"{args.user}_action_{args.action}_{args.dataset_id}"
            "_unsynced_10s.png"
        )
    )


def _inside_panel(
    values: np.ndarray,
    start: float,
    stop: float,
    *,
    final_panel: bool,
) -> np.ndarray:
    if final_panel:
        return (values >= start) & (values <= stop)
    return (values >= start) & (values < stop)


def _plot(
    *,
    args: argparse.Namespace,
    ring_seconds: np.ndarray,
    transient_score: np.ndarray,
    labels,
    label_seconds: np.ndarray,
    board_events,
    board_seconds: np.ndarray,
    ring_start_us: float,
    transient_source: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panel_start, panel_count = _panel_limits(
        ring_seconds=ring_seconds,
        label_seconds=label_seconds,
        board_seconds=board_seconds,
        panel_seconds=args.panel_seconds,
        source=args.xlim_source,
    )

    figure, axes = plt.subplots(
        panel_count,
        1,
        figsize=(16.0, max(3.0, 2.8 * panel_count)),
        squeeze=False,
        layout="constrained",
    )
    axes = axes[:, 0]

    y_max = float(np.quantile(transient_score, 0.995) * 1.10)
    if not math.isfinite(y_max) or y_max <= 0.0:
        y_max = max(float(np.max(transient_score)), 1.0)

    event_types = (
        board_events["event_type"].to_numpy(dtype=object)
        if len(board_events)
        else np.empty(0, dtype=object)
    )

    for panel_index, axis in enumerate(axes):
        start = panel_start + panel_index * args.panel_seconds
        stop = start + args.panel_seconds
        final_panel = panel_index == panel_count - 1

        ring_mask = _inside_panel(
            ring_seconds, start, stop, final_panel=final_panel
        )
        if np.any(ring_mask):
            axis.plot(
                ring_seconds[ring_mask],
                transient_score[ring_mask],
                color="0.55",      
                linewidth=0.5,     
                alpha=0.55,        
            )

        label_mask = _inside_panel(
            label_seconds, start, stop, final_panel=final_panel
        )
        for label_index in np.flatnonzero(label_mask):
            x = float(label_seconds[label_index])
            axis.axvline(
                x,
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
            )
            axis.text(
                x,
                y_max * 0.98,
                str(labels[int(label_index)].label),
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=7,
                clip_on=True,
            )

        event_mask = _inside_panel(
            board_seconds, start, stop, final_panel=final_panel
        )
        for event_index in np.flatnonzero(event_mask):
            event_type = str(event_types[event_index])

            if event_type == "press":
                color = "red"
                linestyle = "-."
            else:  # lift
                color = "gold"   # 或 "yellow"
                linestyle = ":"

            axis.axvline(
                float(board_seconds[event_index]),
                color=color,
                linestyle=linestyle,
                linewidth=1.2,
                alpha=0.85,
            )

        axis.set_xlim(start, stop)
        axis.set_ylim(0.0, y_max)
        axis.grid(True, alpha=0.25)
        axis.set_title(f"{start:g}–{stop:g} s", fontsize=10)
        axis.set_ylabel("Transient score")

    axes[-1].set_xlabel(
        "Raw time relative to Ring start "
        "(seconds; no offset / no synchronization)"
    )

    legend_handles = [
        Line2D(
            [], [],
            color="0.55",
            linestyle="-",
            linewidth=0.5,
            alpha=0.55,
            label="Ring transient score",
        ),
        Line2D(
            [], [],
            color="C0",
            linestyle="--",
            linewidth=1.0,
            label="raw label timestamp",
        ),
        Line2D(
            [], [],
            color="red",
            linestyle="-.",
            linewidth=1.2,
            label="raw Board press",
        ),
        Line2D(
            [], [],
            color="gold",
            linestyle=":",
            linewidth=1.2,
            label="raw Board lift",
        ),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=8)

    figure.suptitle(
        "Unsynchronized Ring / Labels / Board Events\n"
        f"{args.user} / action {args.action} / dataset {args.dataset_id}\n"
        f"plot origin = raw Ring start {ring_start_us:.0f} us; "
        f"transient source = {transient_source}",
        fontsize=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    _configure_matplotlib()
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    try:
        _validate_args(args)

        from writingring.board_loader import load_board
        from writingring.discovery import discover_recordings
        from writingring.segmentation import load_timestamp_labels
        from writingring.selection import select_recording

        recording = select_recording(
            discover_recordings(args.data_root),
            user=args.user,
            action=str(args.action),
            dataset_id=args.dataset_id,
        )

        if recording.timestamp_path is None:
            raise ValueError(
                f"dataset {args.dataset_id} has no timestamp label file"
            )

        ring_timestamps_us, transient_score, transient_source = (
            _load_transient_source(recording, args)
        )
        if ring_timestamps_us.ndim != 1 or len(ring_timestamps_us) == 0:
            raise ValueError("Ring timestamp vector must be nonempty")
        if len(transient_score) != len(ring_timestamps_us):
            raise ValueError(
                "transient score length does not match Ring timestamps"
            )

        labels = load_timestamp_labels(recording.timestamp_path)
        label_timestamps_us = np.asarray(
            [label.timestamp_us for label in labels],
            dtype=np.float64,
        )

        board = load_board(recording)
        board_events = _raw_board_transitions(board.frames)
        board_timestamps_us = (
            board_events["frame_timestamp_raw"].to_numpy(dtype=np.float64)
            if len(board_events)
            else np.empty(0, dtype=np.float64)
        )

        ring_start_us = float(ring_timestamps_us[0])

        # One common display-origin shift only; no synchronization.
        ring_seconds = _seconds_from_ring_start(
            ring_timestamps_us, ring_start_us
        )
        label_seconds = _seconds_from_ring_start(
            label_timestamps_us, ring_start_us
        )
        board_seconds = _seconds_from_ring_start(
            board_timestamps_us, ring_start_us
        )

        output_path = args.output or _default_output(args)
        _plot(
            args=args,
            ring_seconds=ring_seconds,
            transient_score=np.asarray(transient_score, dtype=np.float64),
            labels=labels,
            label_seconds=label_seconds,
            board_events=board_events,
            board_seconds=board_seconds,
            ring_start_us=ring_start_us,
            transient_source=transient_source,
            output_path=output_path,
        )

        press_count = (
            int((board_events["event_type"] == "press").sum())
            if len(board_events)
            else 0
        )
        lift_count = (
            int((board_events["event_type"] == "lift").sum())
            if len(board_events)
            else 0
        )

        board_frame_ts = board.frames["frame_timestamp_raw"].to_numpy(
            dtype=np.float64
        )
        backward_jump_count = int(
            np.count_nonzero(np.diff(board_frame_ts) < 0.0)
        )

        print(
            "Ring raw timestamp range: "
            f"{ring_timestamps_us[0]:.0f} .. {ring_timestamps_us[-1]:.0f} us "
            f"= {ring_seconds[0]:.3f} .. {ring_seconds[-1]:.3f} s "
            "relative to Ring start"
        )
        print(
            f"Label count: {len(labels)}; "
            f"relative range: "
            f"{label_seconds.min():.3f} .. {label_seconds.max():.3f} s"
        )
        if len(board_events):
            print(
                f"Board raw transitions: press={press_count}, lift={lift_count}; "
                f"relative range: "
                f"{board_seconds.min():.3f} .. {board_seconds.max():.3f} s"
            )
        else:
            print("Board raw transitions: press=0, lift=0")

        print(f"Board raw timestamp backward jumps: {backward_jump_count}")
        print("Alignment offset applied: NO")
        print("Board pairing/transient classification: NO")
        print(f"Saved plot: {output_path}")
        return 0

    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
