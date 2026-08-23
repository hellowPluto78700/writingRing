#!/usr/bin/env python3
"""Resample complete preprocessed IMU recordings before spike encoding."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-rate-hz", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from writingring.resampling import ResamplingError, resample_recording

    paths = sorted(args.input_root.rglob("*_preprocessedIMU.npy"))
    if not paths:
        print("error: input root contains no preprocessed IMU artifacts", file=sys.stderr)
        return 2
    try:
        for path in paths:
            relative = path.relative_to(args.input_root).parent
            stem = path.name.removesuffix("_preprocessedIMU.npy")
            summary = path.with_name(f"{stem}_preprocessing.json")
            timestamps = path.with_name(f"{stem}_timestamps_us.npy")
            result = resample_recording(
                input_imu_path=path,
                input_summary_path=summary,
                input_timestamps_path=timestamps,
                output_root=args.output_root,
                relative_recording=relative,
                target_rate_hz=args.target_rate_hz,
                overwrite=args.overwrite,
            )
            print(f"Resampled {path.relative_to(args.input_root)} -> {len(result.imu)} rows")
    except (OSError, ValueError, ResamplingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
