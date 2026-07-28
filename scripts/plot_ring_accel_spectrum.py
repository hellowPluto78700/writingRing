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
        plot_ring_acceleration_frequency_support,
        plot_ring_acceleration_psd_overlay,
    )

    return SimpleNamespace(
        plt=plt,
        discover_recordings=discover_recordings,
        select_recording=select_recording,
        load_ring=load_ring,
        acceleration_columns=ACCELERATION_COLUMNS,
        compute_frequency_support=compute_frequency_support,
        compute_windowed_psd=compute_windowed_psd,
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
            SpectralAnalysisError,
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
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


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
        if args.plot_mode == "support":
            windowed_by_axis = {
                column: runtime.compute_windowed_psd(
                    ring_data.dataframe[column].to_numpy(copy=True),
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
            )
        else:
            metadata = runtime.compute_windowed_psd(
                ring_data.dataframe["acc_x"].to_numpy(copy=True),
                sampling_rate_hz=args.sampling_rate,
                window_seconds=args.window_seconds,
                overlap_ratio=args.overlap,
            )
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
            print(f"Saved: {args.output}")
            print(f"Samples: {ring_data.validation.sample_count}")
            print(f"Nominal sampling rate: {args.sampling_rate:g} Hz")
            print(f"Window duration: {args.window_seconds:g} s")
            print(f"Window size: {metadata.window_size_samples} samples")
            print(f"Overlap: {args.overlap * 100:g}%")
            print(f"Hop size: {metadata.hop_size_samples} samples")
            print(f"Windows: {metadata.window_count}")
            print(
                "Frequency resolution: "
                f"{metadata.frequency_resolution_hz:g} Hz"
            )
            print(
                f"Displayed range: {frequency_min:g}–"
                f"{frequency_max:g} Hz"
            )
            print(f"Aggregate: {args.aggregate}")
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
) -> None:
    print(f"Samples: {ring_data.validation.sample_count}")
    print(f"Windows: {metadata.window_count}")
    print(f"Window size: {metadata.window_size_samples} samples")
    print(f"Hop size: {metadata.hop_size_samples} samples")
    print(f"Frequency resolution: {metadata.frequency_resolution_hz:g} Hz")
    print(f"Analyzed range: {frequency_min:g}–{frequency_max:g} Hz")
    print(f"High-power quantile: {args.high_power_quantile:g}")
    print(f"Minimum highlighted support: {args.minimum_support:g}%")
    print(f"Output path: {args.output}")
    titles = {
        "acc_x": "Acceleration X",
        "acc_y": "Acceleration Y",
        "acc_z": "Acceleration Z",
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


if __name__ == "__main__":
    raise SystemExit(main())
