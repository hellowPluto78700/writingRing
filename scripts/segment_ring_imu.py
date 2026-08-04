#!/usr/bin/env python3
"""Export fixed-size Ring IMU segments defined by timestamp label markers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence


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
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/segmentedIMU")
    )
    parser.add_argument(
        "--output-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--exclude-last-label",
        action="store_true",
        help="omit each recording's final label instead of extending it to Ring end",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace the complete output set"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run segmentation and report the deterministic output paths."""

    args = build_parser().parse_args(argv)
    from writingring.segmentation import SegmentationConfig, segment_user_action

    config = SegmentationConfig(
        output_dtype=args.output_dtype,
        include_last_label=not args.exclude_last_label,
    )
    try:
        result = segment_user_action(
            data_root=args.data_root,
            user=args.user,
            action=args.action,
            output_root=args.output_root,
            config=config,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        "Exported "
        f"{result.summary['segment_count']} variable-length segments with "
        f"{result.raw_imu.shape[0]} total IMU samples."
    )
    print(f"Raw IMU: {result.output_paths.raw_imu_path}")
    print(f"Labels: {result.output_paths.labels_path}")
    print(f"Segment offsets: {result.output_paths.segment_offsets_path}")
    print(f"Segment lengths: {result.output_paths.segment_lengths_path}")
    print(f"Manifest: {result.output_paths.segments_csv_path}")
    print(f"Summary: {result.output_paths.summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
