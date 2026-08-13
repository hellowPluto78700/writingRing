#!/usr/bin/env python3
"""Plot an unsynchronized SpikeIMU / raw-label / raw-Board timeline.

This diagnostic intentionally performs NO Ring<->Board synchronization.

Transient-score contract
------------------------
The transient score uses the exact same source/API as Board-assisted
segmentation verification:

    feature_input = load_recording_features(..., input_kind="spike-imu", ...)
    compute_transient_score_array(
        feature_input.values[:, feature_input.transient_channel_indices]
    )

For the current SpikeIMU schema, transient_channel_indices are 15:21.

Timeline contract
-----------------
- SpikeIMU uses its canonical Ring timestamps.
- timestamp.txt labels use their original timestamps.
- Board press/lift timestamps use raw frame_timestamp_raw.
- No alignment offset is read or applied.
- No alignment work axis is built.
- Board transitions are detected only from contact_count occupancy changes:
      0 -> >0 : press
      >0 -> 0 : lift
- No press/lift pairing is performed.
- No transient/valid_touch Board classification is performed.
- For readable x-axis values only, all timestamps use the same Ring-start
  display origin:

      seconds = (timestamp_us - ring_start_us) / 1_000_000

  This common translation is NOT synchronization and preserves the original
  relative offsets between Ring, labels, and Board.
- Plot rows are 10 seconds by default.
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


DEFAULT_SPIKE_ROOT = Path(
    "outputs/action0_pipeline/low-pass/"
    "aligned-board-events/spikeEncoding/custom-wavelet"
)


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
        "--spike-root",
        type=Path,
        default=DEFAULT_SPIKE_ROOT,
        help=(
            "root containing published SpikeIMU artifacts; default: "
            f"{DEFAULT_SPIKE_ROOT}"
        ),
    )
    parser.add_argument(
        "--panel-seconds",
        type=float,
        default=10.0,
        help="duration of each plot row in seconds (default: 10)",
    )
    parser.add_argument(
        "--xlim-source",
        choices=("union", "ring"),
        default="union",
        help=(
            "union: include raw label/Board timestamps outside Ring range; "
            "ring: restrict panels to canonical SpikeIMU/Ring range"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output PNG path; default: "
            "outputs/diagnostics/unsynced_spikeimu_timeline/"
            "<user>_action_<action>_<dataset>_unsynced_spikeimu_10s.png"
        ),
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.dataset_id < 0:
        raise ValueError("--dataset-id/--recording must be nonnegative")
    if not math.isfinite(args.panel_seconds) or args.panel_seconds <= 0.0:
        raise ValueError("--panel-seconds must be finite and positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")


def _raw_board_transitions(board_frames):
    """Return only raw occupancy transitions, without touch pairing."""
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

    # Do not invent an event at frame zero if the recording starts occupied.
    transition = np.diff(occupied_i8, prepend=occupied_i8[0])
    positions = np.flatnonzero(transition != 0)

    rows = []
    for event_index, position in enumerate(positions):
        frame = board_frames.iloc[int(position)]
        rows.append(
            {
                "event_index": event_index,
                "event_type": (
                    "press" if transition[int(position)] == 1 else "lift"
                ),
                "global_frame_index": int(frame["global_frame_index"]),
                "frame_timestamp_raw": float(frame["frame_timestamp_raw"]),
            }
        )

    return pd.DataFrame(rows)


def _seconds_from_ring_start(
    timestamps_us: np.ndarray,
    ring_start_us: float,
) -> np.ndarray:
    values = np.asarray(timestamps_us, dtype=np.float64)
    return (values - ring_start_us) / 1_000_000.0


def _panel_layout(
    *,
    ring_seconds: np.ndarray,
    label_seconds: np.ndarray,
    board_seconds: np.ndarray,
    panel_seconds: float,
    xlim_source: str,
) -> tuple[float, int]:
    if xlim_source == "ring":
        minimum = float(ring_seconds[0])
        maximum = float(ring_seconds[-1])
    else:
        parts = [ring_seconds]
        if label_seconds.size:
            parts.append(label_seconds)
        if board_seconds.size:
            parts.append(board_seconds)
        all_seconds = np.concatenate(parts)
        minimum = float(np.min(all_seconds))
        maximum = float(np.max(all_seconds))

    first_panel_start = math.floor(minimum / panel_seconds) * panel_seconds
    final_panel_stop = math.ceil(maximum / panel_seconds) * panel_seconds
    if final_panel_stop <= first_panel_start:
        final_panel_stop = first_panel_start + panel_seconds

    panel_count = max(
        1,
        int(math.ceil(
            (final_panel_stop - first_panel_start) / panel_seconds
        )),
    )
    return first_panel_start, panel_count


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


def _default_output(args: argparse.Namespace) -> Path:
    return (
        Path("outputs")
        / "diagnostics"
        / "unsynced_spikeimu_timeline"
        / (
            f"{args.user}_action_{args.action}_{args.dataset_id}"
            "_unsynced_spikeimu_10s.png"
        )
    )


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
    transient_channel_indices: tuple[int, ...],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    first_panel_start, panel_count = _panel_layout(
        ring_seconds=ring_seconds,
        label_seconds=label_seconds,
        board_seconds=board_seconds,
        panel_seconds=args.panel_seconds,
        xlim_source=args.xlim_source,
    )

    figure, axes = plt.subplots(
        panel_count,
        1,
        figsize=(16.0, max(3.0, 2.8 * panel_count)),
        squeeze=False,
        layout="constrained",
    )
    axes = axes[:, 0]

    # Same style of robust global display ceiling used by verification plots.
    y_max = float(np.quantile(transient_score, 0.995) * 1.10)
    if not math.isfinite(y_max) or y_max <= 0.0:
        y_max = max(float(np.max(transient_score)), 1.0)

    event_types = (
        board_events["event_type"].to_numpy(dtype=object)
        if len(board_events)
        else np.empty(0, dtype=object)
    )

    for panel_index, axis in enumerate(axes):
        panel_start = first_panel_start + panel_index * args.panel_seconds
        panel_stop = panel_start + args.panel_seconds
        final_panel = panel_index == panel_count - 1

        ring_mask = _inside_panel(
            ring_seconds,
            panel_start,
            panel_stop,
            final_panel=final_panel,
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
            label_seconds,
            panel_start,
            panel_stop,
            final_panel=final_panel,
        )
        for label_index in np.flatnonzero(label_mask):
            x = float(label_seconds[int(label_index)])
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

        board_mask = _inside_panel(
            board_seconds,
            panel_start,
            panel_stop,
            final_panel=final_panel,
        )
        for board_index in np.flatnonzero(board_mask):
            event_type = str(event_types[int(board_index)])
            color = "red" if event_type == "press" else "gold"
            linestyle = "-." if event_type == "press" else ":"
            axis.axvline(
                float(board_seconds[int(board_index)]),
                color=color,
                linestyle=linestyle,
                linewidth=1.2,
                alpha=0.85,
            )

        axis.set_xlim(panel_start, panel_stop)
        axis.set_ylim(0.0, y_max)
        axis.grid(True, alpha=0.25)
        axis.set_title(
            f"{panel_start:g}-{panel_stop:g} s",
            fontsize=10,
        )
        axis.set_ylabel("Transient score")

    axes[-1].set_xlabel(
        "Raw time relative to Ring start "
        "(seconds; no offset / no synchronization)"
    )

    axes[0].legend(
        handles=[
            Line2D(
                [], [], color="0.55", linestyle="-", linewidth=0.5,
                alpha=0.55, label="SpikeIMU transient score"
            ),
            Line2D(
                [], [], linestyle="--", linewidth=1.0,
                label="raw label timestamp"
            ),
            Line2D(
                [], [], color="red", linestyle="-.", linewidth=1.2,
                label="raw Board press"
            ),
            Line2D(
                [], [], color="gold", linestyle=":", linewidth=1.2,
                label="raw Board lift"
            ),
        ],
        loc="upper right",
        fontsize=8,
    )

    figure.suptitle(
        "Unsynchronized SpikeIMU / Labels / Board Events\n"
        f"{args.user} / action {args.action} / dataset {args.dataset_id}\n"
        f"Ring origin={ring_start_us:.0f} us; "
        f"transient channels={transient_channel_indices}; "
        "NO Board<->Ring offset",
        fontsize=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    _configure_matplotlib()
    args = build_parser().parse_args(
        sys.argv[1:] if argv is None else argv
    )

    try:
        _validate_args(args)

        from writingring.board_loader import load_board
        from writingring.discovery import discover_recordings
        from writingring.event_alignment import compute_transient_score_array
        from writingring.recording_features import load_recording_features
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

        # Exact same SpikeIMU transient-score source/API as Board-assisted
        # segmentation verification.
        feature_input = load_recording_features(
            recording,
            input_kind="spike-imu",
            spike_root=args.spike_root,
        )
        transient_score = compute_transient_score_array(
            feature_input.values[
                :, feature_input.transient_channel_indices
            ]
        )
        ring_timestamps_us = np.asarray(
            feature_input.timestamps_us,
            dtype=np.float64,
        )

        if len(transient_score) != len(ring_timestamps_us):
            raise ValueError(
                "transient score length does not match SpikeIMU timestamps"
            )

        labels = load_timestamp_labels(recording.timestamp_path)
        label_timestamps_us = np.asarray(
            [label.timestamp_us for label in labels],
            dtype=np.float64,
        )

        board = load_board(recording)

        # Full raw Board frame order: no stale-tail trim, no pairing,
        # no transient/valid-touch classification.
        board_events = _raw_board_transitions(board.frames)
        board_timestamps_us = (
            board_events["frame_timestamp_raw"].to_numpy(
                dtype=np.float64
            )
            if len(board_events)
            else np.empty(0, dtype=np.float64)
        )

        ring_start_us = float(ring_timestamps_us[0])

        # One common display-origin conversion only. This is NOT alignment.
        ring_seconds = _seconds_from_ring_start(
            ring_timestamps_us,
            ring_start_us,
        )
        label_seconds = _seconds_from_ring_start(
            label_timestamps_us,
            ring_start_us,
        )
        board_seconds = _seconds_from_ring_start(
            board_timestamps_us,
            ring_start_us,
        )

        output_path = args.output or _default_output(args)
        _plot(
            args=args,
            ring_seconds=ring_seconds,
            transient_score=np.asarray(
                transient_score,
                dtype=np.float64,
            ),
            labels=labels,
            label_seconds=label_seconds,
            board_events=board_events,
            board_seconds=board_seconds,
            ring_start_us=ring_start_us,
            transient_channel_indices=tuple(
                feature_input.transient_channel_indices
            ),
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

        board_frame_timestamps = board.frames[
            "frame_timestamp_raw"
        ].to_numpy(dtype=np.float64)
        backward_jump_count = int(
            np.count_nonzero(
                np.diff(board_frame_timestamps) < 0.0
            )
        )

        print(
            "SpikeIMU transient-score API: "
            "compute_transient_score_array("
            "feature_input.values[:, "
            "feature_input.transient_channel_indices])"
        )
        print(
            "SpikeIMU transient channels: "
            f"{tuple(feature_input.transient_channel_indices)}"
        )
        print(
            "Canonical Ring timestamp range: "
            f"{ring_timestamps_us[0]:.0f} .. "
            f"{ring_timestamps_us[-1]:.0f} us "
            f"= {ring_seconds[0]:.3f} .. "
            f"{ring_seconds[-1]:.3f} s relative to Ring start"
        )

        if label_seconds.size:
            print(
                f"Raw labels: count={len(labels)}, "
                f"relative range="
                f"{label_seconds.min():.3f} .. "
                f"{label_seconds.max():.3f} s"
            )

        if len(board_events):
            print(
                f"Raw Board transitions: "
                f"press={press_count}, lift={lift_count}, "
                f"relative range="
                f"{board_seconds.min():.3f} .. "
                f"{board_seconds.max():.3f} s"
            )
        else:
            print("Raw Board transitions: press=0, lift=0")

        print(
            f"Raw Board timestamp backward jumps: "
            f"{backward_jump_count}"
        )
        print("Alignment offset applied: NO")
        print("Alignment work axis used: NO")
        print("Board pairing used: NO")
        print("Board transient classification used: NO")
        print(f"Saved plot: {output_path}")
        return 0

    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
