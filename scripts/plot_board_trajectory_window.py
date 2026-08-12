#!/usr/bin/env python3
"""Plot WritingRing Board contact trajectories for a selected recording window.

Example:
    python scripts/plot_board_trajectory_window.py \
        --data-root data \
        --user user_20 \
        --recording 3 \
        --start-s 6 \
        --end-s 8

By default, --start-s/--end-s are elapsed seconds from the Ring recording
start, matching the elapsed-time convention used by segmentation verification.
Use --time-origin board to measure seconds from the first loaded Board frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from writingring.board_loader import BoardLoadError, load_board
from writingring.discovery import DiscoveryError, Recording, discover_recordings
from writingring.ring_loader import RingLoadError, load_ring


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--user",
        required=True,
        help="User directory, e.g. user_20. A bare integer such as 20 is also accepted.",
    )
    parser.add_argument(
        "--action",
        default="0",
        help="Action directory (default: 0).",
    )
    parser.add_argument(
        "--recording",
        "--dataset-id",
        dest="dataset_id",
        type=int,
        required=True,
        help="Recording/dataset ID, i.e. the integer prefix in <id>_ring_0.bin.",
    )
    parser.add_argument(
        "--start-s",
        type=float,
        required=True,
        help="Window start in elapsed seconds.",
    )
    parser.add_argument(
        "--end-s",
        type=float,
        required=True,
        help="Window end in elapsed seconds; the selected interval is [start, end).",
    )
    parser.add_argument(
        "--time-origin",
        choices=("ring", "board"),
        default="ring",
        help=(
            "Interpret elapsed seconds from the Ring start (default) or from "
            "the first loaded Board frame."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. If omitted, a deterministic path under outputs/board_trajectory is used.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive window after saving (requires a GUI backend; normally omit on servers).",
    )
    return parser


def normalize_user(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("--user must not be empty")
    return value if value.startswith("user_") else f"user_{value}"


def resolve_recording(
    data_root: Path, *, user: str, action: str, dataset_id: int
) -> Recording:
    recordings = discover_recordings(data_root)
    matches = [
        recording
        for recording in recordings
        if recording.user == user
        and recording.action == str(action)
        and recording.dataset_id == dataset_id
    ]
    if not matches:
        raise ValueError(
            f"recording not found: user={user}, action={action}, dataset_id={dataset_id}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"recording identity is ambiguous: user={user}, action={action}, dataset_id={dataset_id}"
        )
    return matches[0]


def first_finite_timestamp(values: pd.Series, *, name: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    if not len(finite):
        raise ValueError(f"{name} has no finite timestamps")
    return float(finite[0])


def resolve_origin_us(recording: Recording, board_frames: pd.DataFrame, time_origin: str) -> float:
    if time_origin == "ring":
        ring = load_ring(recording)
        timestamps = ring.dataframe["timestamp"].to_numpy(dtype=np.float64)
        if not len(timestamps) or not np.isfinite(timestamps[0]):
            raise ValueError("Ring recording has no finite start timestamp")
        return float(timestamps[0])

    if board_frames.empty:
        raise ValueError("Board recording has no frames")
    return first_finite_timestamp(
        board_frames["frame_timestamp_raw"], name="Board recording"
    )


def split_contact_tracks(contacts: pd.DataFrame) -> list[pd.DataFrame]:
    """Split contact IDs into individual strokes without joining reused IDs."""

    if contacts.empty:
        return []

    ordered = contacts.sort_values(
        ["global_frame_index", "contact_index"], kind="stable"
    ).copy()

    tracks: list[pd.DataFrame] = []
    active: dict[int, list[int]] = {}
    previous_frame: dict[int, int] = {}

    for row_index, row in ordered.iterrows():
        contact_id = int(row["contact_id"])
        state = int(row["state"])
        frame_index = int(row["global_frame_index"])

        should_restart = False
        if contact_id in active:
            if state == 1:
                should_restart = True
            elif frame_index - previous_frame[contact_id] > 5:
                should_restart = True

        if should_restart:
            indices = active.pop(contact_id)
            if indices:
                tracks.append(ordered.loc[indices].copy())

        active.setdefault(contact_id, []).append(row_index)
        previous_frame[contact_id] = frame_index

        if state == 3:
            indices = active.pop(contact_id)
            tracks.append(ordered.loc[indices].copy())
            previous_frame.pop(contact_id, None)

    for indices in active.values():
        if indices:
            tracks.append(ordered.loc[indices].copy())

    tracks.sort(key=lambda table: int(table["global_frame_index"].iloc[0]))
    return tracks


def marker_sizes_from_force(force: np.ndarray) -> np.ndarray:
    finite = np.isfinite(force)
    if not finite.any():
        return np.full(len(force), 24.0)
    positive = np.maximum(force[finite], 0.0)
    scale = float(np.quantile(positive, 0.95)) if len(positive) else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return np.full(len(force), 24.0)
    normalized = np.clip(np.maximum(force, 0.0) / scale, 0.0, 1.0)
    normalized[~finite] = 0.0
    return 18.0 + 70.0 * normalized


def default_output_path(
    *, user: str, action: str, dataset_id: int, start_s: float, end_s: float, time_origin: str
) -> Path:
    def token(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    return (
        Path("outputs")
        / "board_trajectory"
        / user
        / f"action_{action}"
        / (
            f"dataset_{dataset_id}_{time_origin}_"
            f"{token(start_s)}s_{token(end_s)}s.png"
        )
    )


def plot_window(
    *,
    recording: Recording,
    board_frames: pd.DataFrame,
    board_contacts: pd.DataFrame,
    origin_us: float,
    start_s: float,
    end_s: float,
    time_origin: str,
    output_path: Path,
    dpi: int,
) -> tuple[int, int, int]:
    window_start_us = origin_us + start_s * 1_000_000.0
    window_end_us = origin_us + end_s * 1_000_000.0

    frame_times = pd.to_numeric(
        board_frames["frame_timestamp_raw"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    frame_mask = (
        np.isfinite(frame_times)
        & (frame_times >= window_start_us)
        & (frame_times < window_end_us)
    )
    selected_frames = board_frames.loc[frame_mask].copy()

    contact_times = pd.to_numeric(
        board_contacts["frame_timestamp_raw"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    contact_mask = (
        np.isfinite(contact_times)
        & (contact_times >= window_start_us)
        & (contact_times < window_end_us)
    )
    selected_contacts = board_contacts.loc[contact_mask].copy()

    if not selected_contacts.empty:
        selected_contacts["elapsed_s"] = (
            pd.to_numeric(selected_contacts["frame_timestamp_raw"], errors="coerce")
            - origin_us
        ) / 1_000_000.0

    tracks = split_contact_tracks(selected_contacts)

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    # The writer normalizes x by 230 and y by 130. Keep normalized coordinates,
    # but give the plotting box the corresponding board-like width/height ratio.
    ax.set_box_aspect(130.0 / 230.0)
    ax.set_xlabel("Board x (normalized)")
    ax.set_ylabel("Board y_display = 1 - y_raw (normalized)")
    ax.grid(True, alpha=0.25)

    scatter = None
    if selected_contacts.empty:
        ax.text(
            0.5,
            0.5,
            "No Board contacts in selected window",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        elapsed = selected_contacts["elapsed_s"].to_numpy(dtype=np.float64)
        force = pd.to_numeric(selected_contacts["force"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        scatter = ax.scatter(
            selected_contacts["x"].to_numpy(dtype=np.float64),
            selected_contacts["y_display"].to_numpy(dtype=np.float64),
            c=elapsed,
            s=marker_sizes_from_force(force),
            alpha=0.65,
        )

        for track_index, track in enumerate(tracks):
            x = track["x"].to_numpy(dtype=np.float64)
            y = track["y_display"].to_numpy(dtype=np.float64)
            if not len(x):
                continue
            ax.plot(x, y, linewidth=1.25, alpha=0.8)
            contact_id = int(track["contact_id"].iloc[0])
            track_start_s = float(track["elapsed_s"].iloc[0])
            ax.text(
                float(x[0]),
                float(y[0]),
                f"id={contact_id} t={track_start_s:.3f}s",
                fontsize=7,
            )
            if len(x) >= 2:
                ax.annotate(
                    "",
                    xy=(float(x[-1]), float(y[-1])),
                    xytext=(float(x[-2]), float(y[-2])),
                    arrowprops={"arrowstyle": "->", "linewidth": 1.0},
                )

        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label(
            "Elapsed time from Ring start (s)"
            if time_origin == "ring"
            else "Elapsed time from Board start (s)"
        )

    ax.set_title(
        f"Board contact trajectory | {recording.user} | action {recording.action} | "
        f"dataset {recording.dataset_id}\n"
        f"window [{start_s:g}, {end_s:g}) s from {time_origin} start | "
        f"frames={len(selected_frames)}, contact samples={len(selected_contacts)}, tracks={len(tracks)}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return len(selected_frames), len(selected_contacts), len(tracks)


def main() -> int:
    args = build_parser().parse_args()

    try:
        if not np.isfinite(args.start_s) or not np.isfinite(args.end_s):
            raise ValueError("--start-s and --end-s must be finite")
        if args.start_s < 0.0:
            raise ValueError("--start-s must be nonnegative")
        if args.end_s <= args.start_s:
            raise ValueError("--end-s must be greater than --start-s")
        if args.dpi <= 0:
            raise ValueError("--dpi must be positive")

        user = normalize_user(args.user)
        recording = resolve_recording(
            args.data_root,
            user=user,
            action=str(args.action),
            dataset_id=args.dataset_id,
        )
        if not recording.board_chunk_paths:
            raise ValueError("selected recording has no Board chunks")

        board = load_board(recording)
        origin_us = resolve_origin_us(recording, board.frames, args.time_origin)
        output_path = args.output or default_output_path(
            user=user,
            action=str(args.action),
            dataset_id=args.dataset_id,
            start_s=args.start_s,
            end_s=args.end_s,
            time_origin=args.time_origin,
        )

        frame_count, contact_count, track_count = plot_window(
            recording=recording,
            board_frames=board.frames,
            board_contacts=board.contacts,
            origin_us=origin_us,
            start_s=args.start_s,
            end_s=args.end_s,
            time_origin=args.time_origin,
            output_path=output_path,
            dpi=args.dpi,
        )

        print(f"Recording: {recording.user}/action_{recording.action}/dataset_{recording.dataset_id}")
        print(f"Time origin: {args.time_origin}")
        print(f"Window: [{args.start_s:g}, {args.end_s:g}) s")
        print(f"Board frames in window: {frame_count}")
        print(f"Board contact samples in window: {contact_count}")
        print(f"Trajectory tracks: {track_count}")
        print(f"Output: {output_path}")
        for warning in board.warnings:
            print(f"Board warning: {warning}", file=sys.stderr)

        if args.show:
            print(
                "Note: the script uses the non-interactive Agg backend for reliable server output; "
                "open the saved PNG to view it."
            )
        return 0

    except (DiscoveryError, BoardLoadError, RingLoadError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
