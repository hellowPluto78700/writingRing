#!/usr/bin/env python3
"""Plot windowed acceleration PSD overlays for one primary Ring recording."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from typing import Any, Sequence


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
    """Import pyplot-dependent components only after backend selection."""

    import matplotlib.pyplot as plt

    from writingring.discovery import DiscoveryError, discover_recordings
    from writingring.gravity import (
        GravityRemovalConfig,
        GravityRemovalError,
        process_ring_gravity,
        upstream_suggested_config,
    )
    from writingring.ring_loader import RingLoadError, load_ring
    from writingring.selection import (
        RecordingSelectionError,
        select_recording,
    )
    from writingring.spectral import (
        ACCELERATION_COLUMNS,
        SUPPORTED_AGGREGATES,
        SpectralAnalysisError,
        compute_frequency_support,
        compute_windowed_psd,
        plot_acceleration_psd_overlay,
        plot_ring_acceleration_frequency_support,
        plot_ring_acceleration_psd_overlay,
    )
    from writingring.stationary import StationarySearchConfig

    return SimpleNamespace(
        plt=plt,
        discover_recordings=discover_recordings,
        select_recording=select_recording,
        load_ring=load_ring,
        gravity_config=GravityRemovalConfig,
        upstream_suggested_config=upstream_suggested_config,
        stationary_search_config=StationarySearchConfig,
        process_ring_gravity=process_ring_gravity,
        acceleration_columns=ACCELERATION_COLUMNS,
        compute_frequency_support=compute_frequency_support,
        compute_windowed_psd=compute_windowed_psd,
        plot_acceleration_psd_overlay=plot_acceleration_psd_overlay,
        plot_ring_acceleration_frequency_support=(
            plot_ring_acceleration_frequency_support
        ),
        plot_ring_acceleration_psd_overlay=(
            plot_ring_acceleration_psd_overlay
        ),
        aggregate_choices=SUPPORTED_AGGREGATES,
        expected_errors=(
            DiscoveryError,
            RecordingSelectionError,
            RingLoadError,
            GravityRemovalError,
            SpectralAnalysisError,
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
    parser.add_argument(
        "--plot-mode",
        choices=("psd", "support"),
        default="psd",
    )
    parser.add_argument("--sampling-rate", type=float, default=200.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--aggregate",
        choices=runtime.aggregate_choices,
        default="mean",
    )
    parser.add_argument("--frequency-min", type=float)
    parser.add_argument("--frequency-max", type=float)
    parser.add_argument("--high-power-quantile", type=float, default=0.90)
    parser.add_argument("--minimum-support", type=float, default=20.0)
    parser.add_argument("--frequency-smoothing-bins", type=int, default=3)
    parser.add_argument("--top-k-labels", type=int, default=5)
    parser.add_argument(
        "--remove-gravity",
        action="store_true",
        help="analyze gravity-removed body-frame linear acceleration",
    )
    _add_gravity_arguments(parser)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _add_gravity_arguments(parser: argparse.ArgumentParser) -> None:
    """Add gravity-removal controls matching the linear-acceleration CLI."""

    parser.add_argument("--calibration-start", type=int)
    parser.add_argument("--calibration-stop", type=int)
    parser.add_argument(
        "--auto-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="automatically select a stationary interval when manual bounds are absent",
    )
    parser.add_argument("--stationary-duration-s", type=float, default=0.10)
    parser.add_argument("--stationary-stride-s", type=float, default=0.05)
    parser.add_argument("--expected-gravity", type=float, default=9.80665)
    parser.add_argument(
        "--profile",
        choices=("explicit", "upstream_suggested"),
        default="explicit",
    )
    parser.add_argument("--gyro-scale-to-rad-s", type=float, default=1.0)
    parser.add_argument("--acceleration-scale", type=float, default=1.0)
    parser.add_argument("--acceleration-unit-label", default="m/s^2")
    parser.add_argument(
        "--axis-transform",
        type=float,
        nargs=9,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
    )
    parser.add_argument("--correction-time-constant", type=float, default=1.0)
    parser.add_argument("--acceleration-gate-tolerance", type=float, default=0.15)
    parser.add_argument("--angular-rate-gate", type=float)
    parser.add_argument("--calibration-min-samples", type=int, default=20)
    parser.add_argument(
        "--calibration-acceleration-mad-relative-max",
        type=float,
        default=0.03,
    )
    parser.add_argument("--calibration-gyro-median-max", type=float, default=0.1)
    parser.add_argument("--calibration-gyro-mad-max", type=float, default=0.02)
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="continue with warnings when stationary calibration checks fail",
    )


def _build_gravity_config(
    args: argparse.Namespace,
    runtime: SimpleNamespace,
) -> object:
    transform = _axis_transform(args.axis_transform)
    common = {
        "sampling_rate_hz": args.sampling_rate,
        "calibration_start_sample": args.calibration_start,
        "calibration_stop_sample": args.calibration_stop,
        "correction_time_constant_s": args.correction_time_constant,
        "acceleration_gate_relative_tolerance": args.acceleration_gate_tolerance,
        "angular_rate_gate_rad_s": args.angular_rate_gate,
        "calibration_min_samples": args.calibration_min_samples,
        "calibration_acceleration_norm_mad_relative_max": (
            args.calibration_acceleration_mad_relative_max
        ),
        "calibration_gyro_norm_median_max_rad_s": args.calibration_gyro_median_max,
        "calibration_gyro_norm_mad_max_rad_s": args.calibration_gyro_mad_max,
        "strict_calibration": not args.provisional,
    }
    if args.profile == "upstream_suggested":
        overrides = dict(common)
        if args.gyro_scale_to_rad_s != 1.0:
            overrides["gyro_scale_to_rad_s"] = args.gyro_scale_to_rad_s
        if args.axis_transform is not None:
            overrides["axis_transform"] = transform
        if args.acceleration_scale != 1.0:
            overrides["acceleration_scale_to_working_units"] = (
                args.acceleration_scale
            )
        if args.acceleration_unit_label != "m/s^2":
            overrides["acceleration_unit_label"] = args.acceleration_unit_label
        return runtime.upstream_suggested_config(**overrides)
    return runtime.gravity_config(
        **common,
        gyro_scale_to_rad_s=args.gyro_scale_to_rad_s,
        acceleration_scale_to_working_units=args.acceleration_scale,
        acceleration_unit_label=args.acceleration_unit_label,
        axis_transform=transform,
        profile_name="explicit",
    )


def _build_stationary_search_config(
    args: argparse.Namespace,
    runtime: SimpleNamespace,
) -> object:
    return runtime.stationary_search_config(
        sampling_rate_hz=args.sampling_rate,
        stationary_duration_s=args.stationary_duration_s,
        stride_duration_s=args.stationary_stride_s,
        expected_gravity_m_s2=args.expected_gravity,
    )


def _axis_transform(
    flat_values: Sequence[float] | None,
) -> tuple[tuple[float, float, float], ...]:
    if flat_values is None:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return (
        tuple(float(value) for value in flat_values[0:3]),
        tuple(float(value) for value in flat_values[3:6]),
        tuple(float(value) for value in flat_values[6:9]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    show = _configure_runtime_environment(effective_argv)
    runtime = _load_runtime()
    args = build_parser(runtime).parse_args(effective_argv)
    figure = None
    frequency_min = (
        args.frequency_min
        if args.frequency_min is not None
        else (1.0 if args.plot_mode == "support" else 0.0)
    )
    frequency_max = (
        args.frequency_max
        if args.frequency_max is not None
        else (100.0 if args.plot_mode == "support" else 30.0)
    )

    try:
        if args.output.exists():
            if not args.output.is_file():
                raise OSError(
                    f"output path is not a regular file: {args.output}"
                )
            if not args.overwrite:
                raise OSError(
                    f"output file already exists; use --overwrite: {args.output}"
                )

        recording = runtime.select_recording(
            runtime.discover_recordings(args.data_root),
            user=args.user,
            action=args.action,
            dataset_id=args.dataset_id,
        )
        ring_data = runtime.load_ring(recording)
        gravity_result = None
        source_label = "Acceleration"
        psd_unit_label = "raw acceleration units²/Hz"
        acceleration_by_axis = {
            column: ring_data.dataframe[column].to_numpy(copy=True)
            for column in runtime.acceleration_columns
        }
        if args.remove_gravity:
            if (
                args.calibration_start is None
                and args.calibration_stop is None
                and not args.auto_calibration
            ):
                raise ValueError(
                    "automatic calibration is disabled; supply both "
                    "--calibration-start and --calibration-stop"
                )
            gravity_result = runtime.process_ring_gravity(
                ring_data,
                config=_build_gravity_config(args, runtime),
                stationary_search_config=_build_stationary_search_config(
                    args,
                    runtime,
                ),
            )
            acceleration_by_axis = {
                column: gravity_result.linear_acceleration_body[:, axis_index].copy()
                for axis_index, column in enumerate(runtime.acceleration_columns)
            }
            source_label = "Linear acceleration"
            psd_unit_label = (
                f"({gravity_result.config.acceleration_unit_label})²/Hz"
            )
        if args.plot_mode == "support":
            windowed_by_axis = {
                column: runtime.compute_windowed_psd(
                    acceleration_by_axis[column],
                    sampling_rate_hz=args.sampling_rate,
                    window_seconds=args.window_seconds,
                    overlap_ratio=args.overlap,
                )
                for column in runtime.acceleration_columns
            }
            supports = {
                column: runtime.compute_frequency_support(
                    windowed_by_axis[column],
                    frequency_min_hz=frequency_min,
                    frequency_max_hz=frequency_max,
                    high_power_quantile=args.high_power_quantile,
                    smoothing_bins=args.frequency_smoothing_bins,
                )
                for column in runtime.acceleration_columns
            }
            metadata = windowed_by_axis[runtime.acceleration_columns[0]]
            figure = runtime.plot_ring_acceleration_frequency_support(
                supports,
                identity=(
                    f"{args.user} / action {args.action} / "
                    f"dataset {args.dataset_id}"
                ),
                source_label=source_label,
                sampling_rate_hz=args.sampling_rate,
                window_seconds=args.window_seconds,
                overlap_ratio=args.overlap,
                minimum_support_percent=args.minimum_support,
                top_k_labels=args.top_k_labels,
                output_path=args.output,
                show=show,
            )
            _print_support_summary(
                args=args,
                ring_data=ring_data,
                metadata=metadata,
                supports=supports,
                frequency_min=frequency_min,
                frequency_max=frequency_max,
                source_label=source_label,
                gravity_result=gravity_result,
            )
        else:
            metadata = runtime.compute_windowed_psd(
                acceleration_by_axis["acc_x"],
                sampling_rate_hz=args.sampling_rate,
                window_seconds=args.window_seconds,
                overlap_ratio=args.overlap,
            )
            if gravity_result is None:
                figure = runtime.plot_ring_acceleration_psd_overlay(
                    ring_data,
                    sampling_rate_hz=args.sampling_rate,
                    window_seconds=args.window_seconds,
                    overlap_ratio=args.overlap,
                    aggregate=args.aggregate,
                    frequency_min_hz=frequency_min,
                    frequency_max_hz=frequency_max,
                    output_path=args.output,
                    show=show,
                )
            else:
                figure = runtime.plot_acceleration_psd_overlay(
                    gravity_result.linear_acceleration_body,
                    identity=(
                        f" — user={args.user}, action={args.action}, "
                        f"dataset={args.dataset_id}"
                    ),
                    source_label=source_label,
                    psd_unit_label=psd_unit_label,
                    sampling_rate_hz=args.sampling_rate,
                    window_seconds=args.window_seconds,
                    overlap_ratio=args.overlap,
                    aggregate=args.aggregate,
                    frequency_min_hz=frequency_min,
                    frequency_max_hz=frequency_max,
                    output_path=args.output,
                    show=show,
                )
            _print_psd_summary(
                args=args,
                ring_data=ring_data,
                metadata=metadata,
                frequency_min=frequency_min,
                frequency_max=frequency_max,
                source_label=source_label,
                gravity_result=gravity_result,
            )
    except (*runtime.expected_errors, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        if not show and figure is not None:
            runtime.plt.close(figure)
    return 0


def _print_support_summary(
    *,
    args: argparse.Namespace,
    ring_data: Any,
    metadata: Any,
    supports: dict[str, Any],
    frequency_min: float,
    frequency_max: float,
    source_label: str,
    gravity_result: Any | None,
) -> None:
    print(f"Samples: {ring_data.validation.sample_count}")
    print(f"Analysis input: {source_label.lower()}")
    print(f"Windows: {metadata.window_count}")
    print(f"Window size: {metadata.window_size_samples} samples")
    print(f"Hop size: {metadata.hop_size_samples} samples")
    print(f"Frequency resolution: {metadata.frequency_resolution_hz:g} Hz")
    print(f"Analyzed range: {frequency_min:g}–{frequency_max:g} Hz")
    print(f"High-power quantile: {args.high_power_quantile:g}")
    print(f"Minimum highlighted support: {args.minimum_support:g}%")
    print(f"Output path: {args.output}")
    _print_gravity_summary(gravity_result)
    titles = {
        "acc_x": f"{source_label} X",
        "acc_y": f"{source_label} Y",
        "acc_z": f"{source_label} Z",
    }
    for column, support in supports.items():
        print(f"\n{titles[column]}:")
        ranked = sorted(
            range(support.frequencies_hz.size),
            key=lambda index: (
                -support.support_percent[index],
                -support.supporting_strength[index],
                support.frequencies_hz[index],
            ),
        )
        for index in ranked[: args.top_k_labels]:
            print(
                f"  {support.frequencies_hz[index]:g} Hz: "
                f"support={support.support_percent[index]:.2f}%, "
                f"strength={support.supporting_strength[index]:.6f}"
            )


def _print_psd_summary(
    *,
    args: argparse.Namespace,
    ring_data: Any,
    metadata: Any,
    frequency_min: float,
    frequency_max: float,
    source_label: str,
    gravity_result: Any | None,
) -> None:
    print(f"Saved: {args.output}")
    print(f"Samples: {ring_data.validation.sample_count}")
    print(f"Analysis input: {source_label.lower()}")
    print(f"Nominal sampling rate: {args.sampling_rate:g} Hz")
    print(f"Window duration: {args.window_seconds:g} s")
    print(f"Window size: {metadata.window_size_samples} samples")
    print(f"Overlap: {args.overlap * 100:g}%")
    print(f"Hop size: {metadata.hop_size_samples} samples")
    print(f"Windows: {metadata.window_count}")
    print(f"Frequency resolution: {metadata.frequency_resolution_hz:g} Hz")
    print(f"Displayed range: {frequency_min:g}–{frequency_max:g} Hz")
    print(f"Aggregate: {args.aggregate}")
    _print_gravity_summary(gravity_result)


def _print_gravity_summary(gravity_result: Any | None) -> None:
    if gravity_result is None:
        return
    calibration = gravity_result.calibration
    diagnostics = gravity_result.diagnostics
    print(
        "Calibration: "
        f"{calibration.start_sample}:{calibration.stop_sample} "
        f"({'passed' if calibration.passed else 'PROVISIONAL'})"
    )
    search = calibration.stationary_search
    if search is None:
        print("Calibration selection: manual interval")
    else:
        print(
            "Automatic stationary search: "
            f"{search.start_index}:{search.stop_index} "
            f"({search.duration_s:g} s; score={search.score:.4g}; "
            f"{'passed' if search.passed else 'FAILED'})"
        )
        print(
            "Stationary gravity / gyro bias: "
            f"{search.estimated_gravity_m_s2:.6g} m/s^2 / "
            f"{search.estimated_gyro_bias_rad_s} rad/s"
        )
        if search.failed_checks:
            print("Failed stationary checks: " + ", ".join(search.failed_checks))
    print(f"Profile: {gravity_result.config.profile_name}")
    print(
        "Correction accepted: "
        f"{diagnostics.correction_used_count}/{diagnostics.sample_count} "
        f"({diagnostics.correction_used_fraction * 100:.1f}%)"
    )
    for warning in diagnostics.warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    raise SystemExit(main())
