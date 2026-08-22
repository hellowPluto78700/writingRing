#!/usr/bin/env python3
"""Run a registered spike encoder over complete preprocessed IMU recordings."""

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
    """Build the single-recording and input-root batch command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-imu", type=Path, help="one complete recording .npy")
    inputs.add_argument(
        "--input-root",
        type=Path,
        help="root to scan recursively for complete recording .npy files",
    )
    parser.add_argument(
        "--pattern",
        default="*_preprocessedIMU.npy",
        help="batch glob pattern relative to --input-root (default: *_preprocessedIMU.npy)",
    )
    parser.add_argument("--input-summary", type=Path)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--encoder-settings", type=Path, required=True)
    parser.add_argument(
        "--encoder-frequencies-hz",
        nargs=5,
        type=float,
        metavar=("F0", "F1", "F2", "F3", "F4"),
        help="override the five Custom Wavelet frequency bands (Hz)",
    )
    parser.add_argument(
        "--post-encode-transform",
        choices=("none", "AbsRectify", "PolaritySplitAbs"),
        default=None,
        help="override the encoder settings post-encode transform",
    )
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
        help="output root; batch mode mirrors the input relative recording paths",
    )
    parser.add_argument("--output-stem", help="legacy single-file output stem")
    parser.add_argument("--output-dtype", choices=("float32", "float64"))
    parser.add_argument(
        "--allow-gravity-included",
        action="store_true",
        help="allow summaries that explicitly declare measured acceleration with gravity",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate, encode, and atomically publish one or many recordings."""

    args = build_parser().parse_args(argv)
    from writingring.spike_encoding import SpikeEncodingError

    try:
        _validate_custom_wavelet_arguments(args)
        if args.input_root is not None:
            _validate_batch_arguments(args)
            summaries = _run_batch(args)
            print(f"Published {len(summaries)} batch spike encoding(s).")
            return 0
        summary = _run_one(
            args,
            input_path=args.input_imu,
            summary_path=_resolve_summary_path(args.input_imu, args.input_summary),
            recording_relative_path=None,
            canonical=False,
        )
    except (OSError, ValueError, SpikeEncodingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    _print_result(summary)
    return 0


def _run_batch(args: argparse.Namespace) -> list[dict[str, object]]:
    """Encode each discovered file with a fresh encoder and one recording boundary."""

    from writingring.spike_encoding import SpikeEncodingError

    root = Path(args.input_root)
    if not root.is_dir():
        raise SpikeEncodingError(f"input root is not a directory: {root}")
    input_paths = sorted(
        (path for path in root.rglob(args.pattern) if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not input_paths:
        raise SpikeEncodingError(f"input root contains no files matching pattern {args.pattern!r}: {root}")
    results: list[dict[str, object]] = []
    for input_path in input_paths:
        relative_path = _batch_recording_relative_path(root, input_path)
        summary = _run_one(
            args,
            input_path=input_path,
            summary_path=_resolve_summary_path(input_path, None),
            recording_relative_path=relative_path,
            expected_recording=_recording_identity(relative_path),
            canonical=True,
        )
        results.append(summary)
        input_section = summary.get("input", {})
        sample_count = input_section.get("sample_count") if isinstance(input_section, dict) else "?"
        print(
            f"Encoded {sample_count} rows from "
            f"{input_path.relative_to(root)} -> {summary['published_directory']}"
        )
    return results


def _run_one(
    args: argparse.Namespace,
    *,
    input_path: Path,
    summary_path: Path | None,
    recording_relative_path: Path | None,
    canonical: bool,
    expected_recording: dict[str, object] | None = None,
) -> dict[str, object]:
    from writingring.spike_encoding import (
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
    from writingring.preprocessing_io import (
        sha256_file,
        validate_timestamp_source_provenance,
    )

    settings = load_encoder_settings(args.encoder_settings)
    if args.encoder_frequencies_hz is not None:
        settings["frequencies_hz"] = list(args.encoder_frequencies_hz)
    if args.post_encode_transform is not None:
        settings["post_encode_transform"] = (
            None
            if args.post_encode_transform == "none"
            else args.post_encode_transform
        )
    if args.output_dtype is not None:
        settings["output_dtype"] = args.output_dtype
    source_summary = (
        load_spike_encoding_source_summary(summary_path)
        if summary_path is not None
        else None
    )
    settings = resolve_sampling_rate_hz(settings, source_summary)
    input_data = load_spike_encoding_input(
        input_path,
        summary_path=summary_path,
        allow_gravity_included=args.allow_gravity_included,
        expected_sampling_rate_hz=float(settings["sampling_rate_hz"]),
        expected_recording=expected_recording,
    )
    timestamp_path = _resolve_timestamp_path(
        args,
        input_path=input_path,
        source_summary=source_summary,
    )
    if timestamp_path is not None:
        load_and_validate_timestamps(
            timestamp_path,
            sample_count=len(input_data.preprocessed_imu),
            sampling_rate_hz=float(settings["sampling_rate_hz"]),
            check_sampling_interval=(
                _timestamp_unit(source_summary) != "microseconds"
            ),
        )
    timestamp_source_hash = None
    timestamp_source_hash_verified = False
    if timestamp_path is not None:
        timestamp_source_hash = sha256_file(timestamp_path)
        if source_summary is not None and (
            source_summary.timestamps_path is not None
            or source_summary.timestamps_sha256 is not None
        ):
            timestamp_source_hash = validate_timestamp_source_provenance(
                timestamp_path,
                source_summary,
            )
            timestamp_source_hash_verified = True
    mode, offsets_path, boundary_semantics = _resolved_boundaries(args)
    offsets = (
        load_sequence_offsets(offsets_path, sample_count=len(input_data.acceleration_g))
        if offsets_path is not None
        else single_array_offsets(sample_count=len(input_data.acceleration_g))
    )
    encoder = create_encoder(args.encoder, settings=settings)
    expected_event_channels = (
        30
        if settings.get("post_encode_transform") == "PolaritySplitAbs"
        else 15
    )
    if (
        args.encoder == "custom-wavelet"
        and len(encoder.output_channel_names) != expected_event_channels
    ):
        raise ValueError(
            "custom-wavelet must produce exactly "
            f"{expected_event_channels} event channels for the selected transform"
        )
    result = run_spike_encoder(
        encoder,
        acceleration_g=input_data.acceleration_g,
        sequence_offsets=offsets,
        sequence_boundary_semantics=boundary_semantics,
    )
    output_dtype = _resolved_output_dtype(args, settings, encoder, result.values.dtype.name)
    paths = spike_encoding_output_paths(
        source_imu_path=input_data.source_imu_path,
        encoder_name=result.encoder_name,
        output_stem=args.output_stem,
        output_root=args.output_root,
        recording_relative_path=recording_relative_path,
        recording_id=(None if recording_relative_path is None else recording_relative_path.name),
        canonical=canonical,
    )
    use_legacy_sidecars = not (
        canonical
        or (
            args.encoder == "custom-wavelet"
            and input_data.source_imu_path.name.endswith("_preprocessedIMU.npy")
        )
    )
    summary = publish_spike_encoding(
        output=result,
        input_data=input_data,
        encoder=encoder,
        effective_settings=settings,
        source_summary=source_summary,
        sequence_mode=mode,
        offsets_source=(
            str(offsets_path.resolve()) if offsets_path is not None else "single-array"
        ),
        output_dtype=output_dtype,
        paths=paths,
        overwrite=args.overwrite,
        source_metadata_paths=_source_metadata_paths(
            args,
            input_data.source_imu_path,
            timestamp_path=timestamp_path,
            include_legacy_sidecars=use_legacy_sidecars,
        ),
        allow_gravity_included=args.allow_gravity_included,
        timestamp_source_hash=timestamp_source_hash,
        timestamp_source_hash_verified=timestamp_source_hash_verified,
    )
    summary["published_directory"] = str(paths.output_directory.resolve())
    return summary


def _print_result(summary: dict[str, object]) -> None:
    encoder = summary.get("encoder", {})
    encoder_name = encoder.get("name") if isinstance(encoder, dict) else None
    if encoder_name == "custom-wavelet":
        output = summary.get("output", {})
        spike_imu = summary.get("spike_imu", {})
        channel_count = spike_imu.get("channel_count") if isinstance(spike_imu, dict) else None
        representation = (
            output.get("event_representation")
            if isinstance(output, dict)
            else None
        )
        if not isinstance(representation, str):
            representation = (
                encoder.get("representation")
                if isinstance(encoder, dict)
                else None
            )
        if not isinstance(representation, str):
            representation = "spike"
        print(
            f"Encoded {summary['input']['sample_count'] if isinstance(summary.get('input'), dict) else summary['statistics']['sample_count']} rows into "
            f"{output.get('channel_count', '?') if isinstance(output, dict) else '?'} {representation} spike channels across "
            f"{summary['sequence_processing']['sequence_count']} recording(s)."
        )
        if channel_count is not None:
            print(f"Published {summary['input']['sample_count']} × {channel_count} aligned spike IMU.")
    else:
        output = summary.get("output", {})
        print(
            f"Encoded {summary['input']['sample_count'] if isinstance(summary.get('input'), dict) else '?'} samples into "
            f"{output.get('channel_count', '?') if isinstance(output, dict) else '?'} channels across "
            f"{summary['sequence_processing']['sequence_count']} sequence(s)."
        )
    print(f"Published spike encoding: {summary['published_directory']}")


def _resolve_summary_path(input_path: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    name = input_path.name
    candidates: list[Path] = []
    if name.endswith("_preprocessedIMU.npy"):
        stem = name.removesuffix("_preprocessedIMU.npy")
        candidates.append(input_path.with_name(f"{stem}_preprocessing.json"))
    if name.endswith("_rawIMU.npy"):
        stem = name.removesuffix("_rawIMU.npy")
        candidates.append(input_path.with_name(f"{stem}_segmentation_summary.json"))
    candidates.append(input_path.with_suffix(".json"))
    return next((path for path in candidates if path.is_file()), None)


def _batch_recording_relative_path(root: Path, input_path: Path) -> Path:
    relative = input_path.relative_to(root)
    parent = relative.parent
    stem = _derived_stem(input_path)
    # The exporter uses user/action/data_id/<data_id>_preprocessedIMU.npy,
    # so that parent already is the desired recording namespace.  For a
    # flatter user/action/file layout, include the file stem to avoid clashes.
    if len(parent.parts) < 3:
        return parent / stem if parent.parts else Path(stem)
    return parent


def _recording_identity(relative_path: Path) -> dict[str, object] | None:
    if len(relative_path.parts) < 3:
        return None
    user, action, data_id = relative_path.parts[-3:]
    parsed_data_id: object = int(data_id) if data_id.isdigit() else data_id
    return {"user": user, "action": action, "data_id": parsed_data_id}


def _derived_stem(path: Path) -> str:
    name = path.name
    for suffix in ("_preprocessedIMU.npy", "_rawIMU.npy", ".npy"):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return path.stem


def _validate_batch_arguments(args: argparse.Namespace) -> None:
    if args.input_summary is not None:
        raise ValueError("--input-summary is only supported with --input-imu")
    if args.output_stem is not None:
        raise ValueError("--output-stem is only supported with --input-imu")
    if any(
        value is not None
        for value in (
            args.sequence_mode,
            args.sequence_offsets,
            args.recording_offsets,
            args.labels_path,
            args.segment_offsets_path,
            args.segment_lengths_path,
            args.timestamps_path,
            args.segments_manifest_path,
        )
    ):
        raise ValueError(
            "batch mode accepts complete recordings only; sequence and segmentation sidecars "
            "must not be supplied"
        )


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
    """Resolve legacy generic sequences and complete-recording boundaries safely."""

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


def _resolve_timestamp_path(
    args: argparse.Namespace,
    *,
    input_path: Path,
    source_summary: object | None,
) -> Path | None:
    """Resolve the canonical timestamp sidecar before publishing metadata."""

    if args.timestamps_path is not None:
        return Path(args.timestamps_path)
    summary_path = getattr(source_summary, "timestamps_path", None)
    if summary_path is not None:
        return Path(summary_path)
    if input_path.name.endswith("_preprocessedIMU.npy"):
        stem = input_path.name.removesuffix("_preprocessedIMU.npy")
        candidates = (
            input_path.with_name(f"{stem}_timestamps_us.npy"),
            input_path.with_name(f"{stem}_timestamps.npy"),
            input_path.with_name(f"{stem}_timestamp.npy"),
        )
        return next((path for path in candidates if path.is_file()), None)
    return None


def _timestamp_unit(source_summary: object | None) -> str | None:
    payload = getattr(source_summary, "payload", None)
    if not isinstance(payload, dict):
        return None
    value = payload.get("timestamp_unit")
    return value if isinstance(value, str) else None


def _source_metadata_paths(
    args: argparse.Namespace,
    source_imu_path: Path,
    *,
    timestamp_path: Path | None = None,
    include_legacy_sidecars: bool = True,
) -> dict[str, Path]:
    """Reference canonical timestamps and optional immutable legacy sidecars."""

    stem = _derived_stem(source_imu_path)
    explicit = {
        "labels_path": args.labels_path,
        "segment_offsets_path": args.segment_offsets_path,
        "segment_lengths_path": args.segment_lengths_path,
        "timestamp_source_path": timestamp_path,
        "segments_manifest_path": args.segments_manifest_path,
    }
    candidates = {
        "labels_path": (source_imu_path.parent / f"{stem}_labels.npy",),
        "segment_offsets_path": (source_imu_path.parent / f"{stem}_segment_offsets.npy",),
        "segment_lengths_path": (source_imu_path.parent / f"{stem}_segment_lengths.npy",),
        "timestamp_source_path": (
            source_imu_path.parent / f"{stem}_timestamps_us.npy",
            source_imu_path.parent / f"{stem}_timestamps.npy",
            source_imu_path.parent / f"{stem}_timestamp.npy",
        ),
        "segments_manifest_path": (
            source_imu_path.parent / f"{stem}_segments.csv",
            source_imu_path.parent / f"{stem}_manifest.json",
        ),
    }
    resolved: dict[str, Path] = {}
    for key, candidate_paths in candidates.items():
        if not include_legacy_sidecars and key != "timestamp_source_path":
            continue
        explicit_path = explicit[key]
        if explicit_path is not None:
            resolved[key] = explicit_path
        else:
            candidate = next((path for path in candidate_paths if path.is_file()), None)
            if candidate is not None:
                resolved[key] = candidate
    return resolved


def _validate_custom_wavelet_arguments(args: argparse.Namespace) -> None:
    """Keep Custom Wavelet recording-level and free of boundary sidecars."""

    if args.post_encode_transform is not None and args.encoder != "custom-wavelet":
        raise ValueError(
            "--post-encode-transform is only supported with encoder 'custom-wavelet'"
        )
    if args.encoder != "custom-wavelet":
        if args.encoder_frequencies_hz is not None:
            raise ValueError(
                "--encoder-frequencies-hz is only supported with encoder 'custom-wavelet'"
            )
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
