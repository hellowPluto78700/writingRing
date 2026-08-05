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
        publish_spike_encoding,
        resolve_sampling_rate_hz,
        run_spike_encoder,
        single_array_offsets,
        spike_encoding_output_paths,
    )

    try:
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
        mode = _resolved_sequence_mode(args)
        offsets = (
            load_sequence_offsets(args.sequence_offsets, sample_count=len(input_data.acceleration_g))
            if mode == "offsets"
            else single_array_offsets(sample_count=len(input_data.acceleration_g))
        )
        encoder = create_encoder(args.encoder, settings=settings)
        result = run_spike_encoder(
            encoder,
            acceleration_g=input_data.acceleration_g,
            sequence_offsets=offsets,
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
                str(args.sequence_offsets.resolve())
                if args.sequence_offsets is not None
                else "single-array"
            ),
            output_dtype=output_dtype,
            paths=paths,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
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
