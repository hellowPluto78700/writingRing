#!/usr/bin/env python3
"""Analyze completed variable-length SpikeIMU segmentation exports globally."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "writingring-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "writingring-cache"))
matplotlib.use("Agg")


def build_parser() -> argparse.ArgumentParser:
    """Build the analysis CLI parser."""

    parser = argparse.ArgumentParser(
        description=__doc__ + " Input must be a SpikeIMU segmentation root."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sampling-rate", type=float, default=200.0)
    parser.add_argument("--candidate-lengths", type=int, nargs="+")
    parser.add_argument("--minimum-coverage", type=float, default=0.99)
    parser.add_argument("--round-to", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a completed segmentation root and write its global report."""

    args = build_parser().parse_args(argv)
    from writingring.segment_padding import (
        DEFAULT_CANDIDATE_LENGTHS,
        SegmentPaddingError,
        analyze_segment_lengths,
        validate_segmented_root,
        write_segment_length_analysis,
    )

    output_dir = args.output_dir or args.input_root / "padding_analysis"
    try:
        datasets = validate_segmented_root(args.input_root)
        analysis = analyze_segment_lengths(
            datasets,
            sampling_rate_hz=args.sampling_rate,
            candidate_lengths=(args.candidate_lengths or DEFAULT_CANDIDATE_LENGTHS),
            minimum_coverage=args.minimum_coverage,
            round_to=args.round_to,
        )
        paths = write_segment_length_analysis(
            analysis, input_root=args.input_root, output_dir=output_dir,
            datasets=datasets, overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    stats = analysis["length_statistics"]
    recommendations = analysis["recommendations"]
    pure = recommendations["pure_padding"]
    print(
        f"Analyzed {analysis['segment_count']} segments in {analysis['user_action_count']} user/action packages. "
        f"Range: {stats['minimum']}–{stats['maximum']} samples."
    )
    print(f"Pure-padding target: {pure['target_length']} samples ({pure['target_length'] / args.sampling_rate:.3f} seconds).")
    print(f"Analysis report: {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
