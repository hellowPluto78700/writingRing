#!/usr/bin/env python3
"""Run and publish a registered spike encoder over preprocessed Ring IMU data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the generic spike-encoding command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-imu", type=Path, required=True)
    parser.add_argument("--input-summary", type=Path)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--encoder-settings", type=Path, required=True)
    parser.add_argument("--sequence-mode", choices=("offsets", "single-array"))
    parser.add_argument("--sequence-offsets", type=Path)
    parser.add_argument(
        "--recording-offsets",
        type=Path,
        help="integer [0, ..., N] recording partition for encoders that support it",
    )
    parser.add_argument(
        "--boundary-padding-mode",
        choices=("reflect",),
        default="reflect",
        help="Custom Wavelet recording-edge context policy (default: reflect)",
    )
    parser.add_argument(
        "--event-index-semantics",
        choices=("occurrence",),
        default="occurrence",
        help="Custom Wavelet output index convention (default: occurrence)",
    )
    parser.add_argument("--labels-path", type=Path)
    parser.add_argument("--segment-offsets-path", type=Path)
    parser.add_argument("--segment-lengths-path", type=Path)
    parser.add_argument(
        "--timestamps-path",
        "--timestamp-source-path",
        dest="timestamps_path",
        type=Path,
        help="optional NPY timestamps used only to validate row alignment",
    )
    parser.add_argument("--segments-manifest-path", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="must match --input-imu's parent; output is stored under that input root",
    )
    parser.add_argument("--output-stem")
    parser.add_argument("--output-dtype", choices=("float32", "float64"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate, run, and atomically publish one registered encoder result."""

    args = build_parser().parse_args(argv)
    from writingring.spike_encoding import (
        SpikeEncodingError,
        create_encoder,
        load_encoder_settings,
        load_sequence_offsets,
        load_spike_encoding_input,
        load_spike_encoding_source_summary,
        load_and_validate_timestamps,
        publish_spike_encoding,
        resolve_sampling_rate_hz,
        run_spike_encoder,
        single_array_offsets,
        spike_encoding_output_paths,
    )

    try:
        _validate_custom_wavelet_arguments(args)
        input_data = load_spike_encoding_input(args.input_imu)
        settings = load_encoder_settings(args.encoder_settings)
        if args.output_dtype is not None:
            settings["output_dtype"] = args.output_dtype
        source_summary = (
            load_spike_encoding_source_summary(args.input_summary)
            if args.input_summary is not None
            else None
        )
        settings = resolve_sampling_rate_hz(settings, source_summary)
        if args.timestamps_path is not None:
            load_and_validate_timestamps(
                args.timestamps_path,
                sample_count=len(input_data.raw_imu),
                sampling_rate_hz=float(settings["sampling_rate_hz"]),
            )
        mode, offsets_path, boundary_semantics = _resolved_boundaries(args)
        offsets = (
            load_sequence_offsets(offsets_path, sample_count=len(input_data.acceleration_g))
            if offsets_path is not None
            else single_array_offsets(sample_count=len(input_data.acceleration_g))
        )
        encoder = create_encoder(args.encoder, settings=settings)
        if args.encoder == "custom-wavelet" and len(encoder.output_channel_names) != 15:
            raise ValueError(
                "custom-wavelet must produce exactly 15 event channels "
                "(three axes by five frequencies)"
            )
        result = run_spike_encoder(
            encoder,
            acceleration_g=input_data.acceleration_g,
            sequence_offsets=offsets,
            sequence_boundary_semantics=boundary_semantics,
        )
        output_dtype = _resolved_output_dtype(
            args, settings, encoder, result.values.dtype.name
        )
        paths = spike_encoding_output_paths(
            raw_imu_path=input_data.raw_imu_path,
            encoder_name=result.encoder_name,
            output_stem=args.output_stem,
            output_root=args.output_root,
        )
        summary = publish_spike_encoding(
            output=result,
            input_data=input_data,
            encoder=encoder,
            effective_settings=settings,
            source_summary=source_summary,
            sequence_mode=mode,
            offsets_source=(
                str(offsets_path.resolve())
                if offsets_path is not None
                else "single-array"
            ),
            output_dtype=output_dtype,
            paths=paths,
            overwrite=args.overwrite,
            source_metadata_paths=_source_metadata_paths(args, input_data.raw_imu_path),
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.encoder == "custom-wavelet":
        padding = getattr(encoder, "padding_samples_each_side")
        print(
            f"Encoded {result.summary['input_sample_count']} rows into "
            f"{result.summary['channel_count']} signed spike channels across "
            f"{result.summary['sequence_count']} recording(s)."
        )
        print(
            f"Published {result.summary['input_sample_count']} × "
            f"{summary['spike_imu']['channel_count']} aligned spike IMU."
        )
        print(f"Padding {padding} samples per recording; event indices align to extrema occurrences.")
        print("Labels, timestamps, and segment offsets are unchanged.")
    else:
        print(
            f"Encoded {result.summary['input_sample_count']} samples into "
            f"{result.summary['channel_count']} channels across "
            f"{result.summary['sequence_count']} sequence(s)."
        )
    print(f"Published spike encoding: {paths.output_directory}")
    print(f"Summary: {paths.summary_json_path}")
    return 0


def _resolved_sequence_mode(args: argparse.Namespace) -> str:
    """Apply the explicit-offset default without guessing source continuity."""

    if args.sequence_mode == "offsets" and args.sequence_offsets is None:
        raise ValueError("--sequence-mode offsets requires --sequence-offsets")
    if args.sequence_mode == "single-array" and args.sequence_offsets is not None:
        raise ValueError("--sequence-offsets requires --sequence-mode offsets")
    if args.sequence_mode is not None:
        return args.sequence_mode
    return "offsets" if args.sequence_offsets is not None else "single-array"


def _resolved_boundaries(args: argparse.Namespace) -> tuple[str, Path | None, str]:
    """Resolve legacy generic sequences and explicit recording boundaries safely."""

    if args.recording_offsets is not None:
        if args.sequence_offsets is not None or args.sequence_mode is not None:
            raise ValueError(
                "--recording-offsets cannot be combined with --sequence-offsets or --sequence-mode"
            )
        return "offsets", args.recording_offsets, "recording"
    mode = _resolved_sequence_mode(args)
    if args.encoder == "custom-wavelet":
        return "single-array", None, "recording"
    return mode, args.sequence_offsets, "sequence"


def _source_metadata_paths(args: argparse.Namespace, raw_imu_path: Path) -> dict[str, Path]:
    """Reference supplied or colocated immutable sidecars without editing them."""

    stem = raw_imu_path.name.removesuffix("_rawIMU.npy").removesuffix(".npy")
    explicit = {
        "labels_path": args.labels_path,
        "segment_offsets_path": args.segment_offsets_path,
        "segment_lengths_path": args.segment_lengths_path,
        "timestamp_source_path": args.timestamps_path,
        "segments_manifest_path": args.segments_manifest_path,
    }
    # Explicit paths deliberately remain in the result even when invalid so
    # publication can reject typos.  Colocated sidecars are only advisory.
    candidates = {
        "labels_path": (raw_imu_path.parent / f"{stem}_labels.npy",),
        "segment_offsets_path": (raw_imu_path.parent / f"{stem}_segment_offsets.npy",),
        "segment_lengths_path": (raw_imu_path.parent / f"{stem}_segment_lengths.npy",),
        "timestamp_source_path": (
            raw_imu_path.parent / f"{stem}_timestamps.npy",
            raw_imu_path.parent / f"{stem}_timestamp.npy",
        ),
        "segments_manifest_path": (
            raw_imu_path.parent / f"{stem}_segments.csv",
            raw_imu_path.parent / f"{stem}_manifest.json",
        ),
    }
    resolved: dict[str, Path] = {}
    for key, candidate_paths in candidates.items():
        explicit_path = explicit[key]
        if explicit_path is not None:
            resolved[key] = explicit_path
        else:
            candidate = next((path for path in candidate_paths if path.is_file()), None)
            if candidate is not None:
                resolved[key] = candidate
    return resolved


def _validate_custom_wavelet_arguments(args: argparse.Namespace) -> None:
    """Keep Custom Wavelet strictly recording-level and metadata-independent."""

    if args.encoder != "custom-wavelet":
        return
    disallowed = {
        "--sequence-mode": args.sequence_mode,
        "--sequence-offsets": args.sequence_offsets,
        "--recording-offsets": args.recording_offsets,
        "--labels-path": args.labels_path,
        "--segment-offsets-path": args.segment_offsets_path,
        "--segment-lengths-path": args.segment_lengths_path,
        "--segments-manifest-path": args.segments_manifest_path,
    }
    supplied = next((option for option, value in disallowed.items() if value is not None), None)
    if supplied is not None:
        raise ValueError(
            f"{supplied} is not supported by custom-wavelet; "
            "one input IMU file is exactly one recording"
        )


def _resolved_output_dtype(
    args: argparse.Namespace,
    settings: dict[str, object],
    encoder: object,
    fallback: str,
) -> str:
    """Use the explicit CLI override, encoder setting, or validated result dtype."""

    metadata = getattr(encoder, "encoding_metadata", None)
    metadata_dtype = metadata.get("output_dtype") if isinstance(metadata, dict) else None
    candidate = args.output_dtype or settings.get("output_dtype") or metadata_dtype or fallback
    if candidate not in {"float32", "float64"}:
        raise ValueError("output dtype must be float32 or float64")
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())
