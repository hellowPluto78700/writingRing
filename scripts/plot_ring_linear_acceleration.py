#!/usr/bin/env python3
"""Estimate and plot body-frame gravity and linear acceleration for one Ring."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from typing import Sequence


def _configure_runtime_environment(argv: Sequence[str]) -> bool:
    """Select Agg before importing pyplot when --no-show is present."""

    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--no-show", action="store_true")
    preliminary_args, _ = preliminary.parse_known_args(argv)
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "writingring-matplotlib"),
    )
    os.environ.setdefault(
        "XDG_CACHE_HOME",
        str(Path(tempfile.gettempdir()) / "writingring-cache"),
    )
    if preliminary_args.no_show:
        import matplotlib

        matplotlib.use("Agg", force=True)
    return not bool(preliminary_args.no_show)


def _load_runtime() -> SimpleNamespace:
    """Load pyplot-dependent project components after backend selection."""

    import matplotlib.pyplot as plt

    from writingring.discovery import DiscoveryError, discover_recordings
    from writingring.gravity import (
        GravityRemovalConfig,
        GravityRemovalError,
        process_ring_gravity,
        upstream_suggested_config,
    )
    from writingring.plotting import (
        PlottingError,
        RING_INFERRED_TIME,
        RING_SAMPLE_INDEX,
        plot_ring_gravity_removal,
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
        gravity_config=GravityRemovalConfig,
        upstream_suggested_config=upstream_suggested_config,
        process_ring_gravity=process_ring_gravity,
        plot_ring_gravity_removal=plot_ring_gravity_removal,
        ring_time_axes=(RING_SAMPLE_INDEX, RING_INFERRED_TIME),
        expected_errors=(
            DiscoveryError,
            RecordingSelectionError,
            RingLoadError,
            GravityRemovalError,
            PlottingError,
            ValueError,
        ),
    )


def build_parser(runtime: SimpleNamespace) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-rate", type=float, default=200.0)
    parser.add_argument("--calibration-start", type=int, required=True)
    parser.add_argument("--calibration-stop", type=int, required=True)
    parser.add_argument(
        "--profile",
        choices=("explicit", "upstream_suggested"),
        default="explicit",
    )
    parser.add_argument("--gyro-scale-to-rad-s", type=float)
    parser.add_argument("--acceleration-scale", type=float, default=1.0)
    parser.add_argument(
        "--acceleration-unit-label",
        default="raw acceleration units",
    )
    parser.add_argument(
        "--axis-transform",
        type=float,
        nargs=9,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
    )
    parser.add_argument(
        "--correction-time-constant",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--acceleration-gate-tolerance",
        type=float,
        default=0.15,
    )
    parser.add_argument("--angular-rate-gate", type=float)
    parser.add_argument("--calibration-min-samples", type=int, default=20)
    parser.add_argument(
        "--calibration-acceleration-mad-relative-max",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--calibration-gyro-median-max",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--calibration-gyro-mad-max",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="continue with warnings when stationary calibration checks fail",
    )
    parser.add_argument(
        "--time-axis",
        choices=runtime.ring_time_axes,
        default="inferred_time",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    show = _configure_runtime_environment(effective_argv)
    runtime = _load_runtime()
    args = build_parser(runtime).parse_args(effective_argv)
    figure = None

    try:
        _validate_output(args.output, overwrite=args.overwrite)
        config = _build_config(args, runtime)
        recording = runtime.select_recording(
            runtime.discover_recordings(args.data_root),
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
        )
        ring_data = runtime.load_ring(recording)
        result = runtime.process_ring_gravity(ring_data, config=config)
        figure = runtime.plot_ring_gravity_removal(
            ring_data,
            result,
            time_axis=args.time_axis,
            output_path=args.output,
            show=show,
        )
        _print_summary(args.output, result)
    except (*runtime.expected_errors, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        if not show and figure is not None:
            runtime.plt.close(figure)
    return 0


def _build_config(
    args: argparse.Namespace,
    runtime: SimpleNamespace,
) -> object:
    transform = _axis_transform(args.axis_transform)
    common = {
        "sampling_rate_hz": args.sampling_rate,
        "calibration_start_sample": args.calibration_start,
        "calibration_stop_sample": args.calibration_stop,
        "correction_time_constant_s": args.correction_time_constant,
        "acceleration_gate_relative_tolerance": (
            args.acceleration_gate_tolerance
        ),
        "angular_rate_gate_rad_s": args.angular_rate_gate,
        "calibration_min_samples": args.calibration_min_samples,
        "calibration_acceleration_norm_mad_relative_max": (
            args.calibration_acceleration_mad_relative_max
        ),
        "calibration_gyro_norm_median_max_rad_s": (
            args.calibration_gyro_median_max
        ),
        "calibration_gyro_norm_mad_max_rad_s": (
            args.calibration_gyro_mad_max
        ),
        "strict_calibration": not args.provisional,
    }
    if args.profile == "upstream_suggested":
        overrides = dict(common)
        if args.gyro_scale_to_rad_s is not None:
            overrides["gyro_scale_to_rad_s"] = args.gyro_scale_to_rad_s
        if args.axis_transform is not None:
            overrides["axis_transform"] = transform
        if args.acceleration_scale != 1.0:
            overrides["acceleration_scale_to_working_units"] = (
                args.acceleration_scale
            )
        if args.acceleration_unit_label != "raw acceleration units":
            overrides["acceleration_unit_label"] = (
                args.acceleration_unit_label
            )
        return runtime.upstream_suggested_config(**overrides)

    if args.gyro_scale_to_rad_s is None:
        raise ValueError(
            "--gyro-scale-to-rad-s is required with --profile explicit"
        )
    return runtime.gravity_config(
        **common,
        gyro_scale_to_rad_s=args.gyro_scale_to_rad_s,
        acceleration_scale_to_working_units=args.acceleration_scale,
        acceleration_unit_label=args.acceleration_unit_label,
        axis_transform=transform,
        profile_name="explicit",
    )


def _axis_transform(
    flat_values: Sequence[float] | None,
) -> tuple[tuple[float, float, float], ...]:
    if flat_values is None:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    return (
        tuple(float(value) for value in flat_values[0:3]),
        tuple(float(value) for value in flat_values[3:6]),
        tuple(float(value) for value in flat_values[6:9]),
    )


def _validate_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not path.is_file():
            raise OSError(f"output path is not a regular file: {path}")
        if not overwrite:
            raise OSError(
                f"output file already exists; use --overwrite: {path}"
            )


def _print_summary(output: Path, result: object) -> None:
    calibration = result.calibration
    diagnostics = result.diagnostics
    print(f"Saved: {output}")
    print(f"Samples: {diagnostics.sample_count}")
    print(
        "Calibration: "
        f"{calibration.start_sample}:{calibration.stop_sample} "
        f"({'passed' if calibration.passed else 'PROVISIONAL'})"
    )
    print(f"Calibration anchor: {calibration.anchor_sample}")
    print(
        f"Gravity magnitude: {calibration.gravity_magnitude:g} "
        f"{result.config.acceleration_unit_label}"
    )
    print(
        "Nominal sampling rate: "
        f"{diagnostics.nominal_sampling_rate_hz:g} Hz (assumed)"
    )
    print(f"Profile: {result.config.profile_name}")
    print(
        "Correction accepted: "
        f"{diagnostics.correction_used_count}/{diagnostics.sample_count} "
        f"({diagnostics.correction_used_fraction * 100:.1f}%)"
    )
    print(
        "Calibration residual median/max: "
        f"{diagnostics.calibration_residual_norm_median:g}/"
        f"{diagnostics.calibration_residual_norm_maximum:g} "
        f"{result.config.acceleration_unit_label}"
    )
    for warning in diagnostics.warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    raise SystemExit(main())
