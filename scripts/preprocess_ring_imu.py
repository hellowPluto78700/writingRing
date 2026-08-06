#!/usr/bin/env python3
"""Export complete gravity-preprocessed Ring IMU recordings."""

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
    """Build the preprocessing export command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        "--input-root",
        dest="data_root",
        type=Path,
        required=True,
        help="raw WritingRing root containing user/action/ring_0 files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/preprocessedIMU"),
        help="complete-recording artifact root (default: outputs/preprocessedIMU)",
    )
    parser.add_argument("--user")
    parser.add_argument("--action")
    parser.add_argument("--dataset-id", type=int)
    parser.add_argument("--output-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--sampling-rate", type=float, default=200.0)
    parser.add_argument(
        "--gravity-removal-method",
        choices=("raw", "low-pass", "madgwick", "xylo-rotate-and-remove-gravity"),
        default="low-pass",
        help="preprocessing method (default: low-pass)",
    )
    parser.add_argument(
        "--low-pass-cutoff-hz",
        type=float,
        default=0.2,
        help="causal low-pass cutoff in Hz (default: 0.2)",
    )
    parser.add_argument(
        "--madgwick-beta",
        type=float,
        default=0.1,
        help="Madgwick fusion gain (default: 0.1)",
    )
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="allow Madgwick output when stationary calibration checks fail",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Export one selected recording or all discovered recordings."""

    args = build_parser().parse_args(argv)
    from writingring.gravity import GravityRemovalConfig
    from writingring.preprocessing_export import (
        PreprocessingExportError,
        export_discovered_preprocessing,
    )

    selectors = (args.user, args.action, args.dataset_id)
    if any(value is not None for value in selectors) and not all(
        value is not None for value in selectors
    ):
        print(
            "error: --user, --action, and --dataset-id must be supplied together",
            file=sys.stderr,
        )
        return 2
    try:
        config = GravityRemovalConfig(
            sampling_rate_hz=args.sampling_rate,
            gravity_removal_method=args.gravity_removal_method,
            low_pass_cutoff_hz=args.low_pass_cutoff_hz,
            madgwick_beta=args.madgwick_beta,
            strict_calibration=(args.gravity_removal_method == "madgwick" and not args.provisional),
        )
        results = export_discovered_preprocessing(
            data_root=args.data_root,
            output_root=args.output_root,
            gravity_config=config,
            output_dtype=args.output_dtype,
            overwrite=args.overwrite,
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
        )
    except (OSError, ValueError, PreprocessingExportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for result in results:
        print(
            f"Exported {result.recording.user}/{result.recording.action}/"
            f"{result.recording.dataset_id}: {len(result.imu)} samples"
        )
        print(f"  IMU: {result.output_paths.imu_path}")
        print(f"  Timestamps: {result.output_paths.timestamps_path}")
        print(f"  Summary: {result.output_paths.summary_path}")
    print(
        f"Exported {len(results)} complete recording(s) with "
        f"gravity method {args.gravity_removal_method}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
