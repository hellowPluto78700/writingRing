#!/usr/bin/env python3
"""Inspect one discovered WritingRing recording and its validation reports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "writingring-matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "writingring-cache"),
)

from writingring.board_loader import BoardLoadError, load_board
from writingring.discovery import DiscoveryError, discover_recordings
from writingring.inspection import (
    build_recording_summary,
    format_recording_summary,
    write_json,
)
from writingring.ring_loader import RingLoadError, load_ring
from writingring.selection import RecordingSelectionError, select_recording


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional path for the same inspection information as JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        recording = select_recording(
            discover_recordings(args.data_root),
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
        )
        ring_data = load_ring(recording)
        board_data = load_board(recording)
        summary = build_recording_summary(recording, ring_data, board_data)
        print(format_recording_summary(summary))
        if args.json_output is not None:
            output_path = write_json(args.json_output, summary)
            print(f"\nJSON summary: {output_path}")
    except (
        DiscoveryError,
        RecordingSelectionError,
        RingLoadError,
        BoardLoadError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
