#!/usr/bin/env python3
"""Create a fixed-length SpikeIMU dataset, skipping segments over the target."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "writingring-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "writingring-cache"))


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed-length padding CLI parser."""

    parser = argparse.ArgumentParser(
        description=__doc__ + " Input must be a SpikeIMU segmentation root."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--target-length", type=int)
    source.add_argument("--analysis-report", type=Path)
    parser.add_argument(
        "--recommendation", choices=("pure-padding", "p99", "balanced"),
        default="pure-padding", help="report recommendation to use (default: pure-padding)",
    )
    parser.add_argument("--sampling-rate", type=float, default=200.0)
    parser.add_argument("--padding-value", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Preflight all input packages, then publish one fixed-length root."""

    args = build_parser().parse_args(argv)
    from writingring.segment_padding import (
        SegmentPaddingError,
        publish_padded_root,
        resolve_target_length,
        validate_segmented_root,
    )

    try:
        datasets = validate_segmented_root(args.input_root)
        target = resolve_target_length(
            target_length=args.target_length, analysis_report=args.analysis_report,
            recommendation=args.recommendation, datasets=datasets, input_root=args.input_root,
        )
        output_root = args.output_root or args.input_root.with_name(
            f"{args.input_root.name}_padded_{target}"
        )
        summary = publish_padded_root(
            datasets, input_root=args.input_root, output_root=output_root,
            target_length=target, sampling_rate_hz=args.sampling_rate,
            padding_value=args.padding_value, overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"Published {summary['segment_count']} padded segments in "
        f"{summary['processed_user_action_count']} user/action packages at target length {target}; "
        f"skipped {summary['skipped_segment_count']} overlong segments."
    )
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
