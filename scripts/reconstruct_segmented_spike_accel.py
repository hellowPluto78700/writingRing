#!/usr/bin/env python3
"""Reconstruct acceleration from segmented WritingRing SpikeIMU artifacts.

Given a dataset root such as
    outputs/action0_rectified/low-pass/aligned-board-events
and one or more users, this script discovers variable-length SpikeIMU
segmentation packages under ``<root>/segmentation/<user>/action_*``.

For every discovered user/action package it:
  1. validates the aggregate ``*_spikeIMU.npy`` and authoritative segment
     offsets/lengths;
  2. decodes each supported SpikeIMU event schema to canonical signed (T, 15)
     events, then reconstructs x/y/z acceleration independently inside each
     segment using the Custom Wavelet reconstruction used by the comparison
     notebook;
  3. concatenates the per-segment reconstructions back into one row-aligned
     ``(N, 3)`` array; and
  4. writes that derived array next to the source SpikeIMU, plus JSON metadata
     and a CSV segment manifest.

The reconstruction is intentionally NOT appended to SpikeIMU.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import numpy as np
from scipy import signal

try:
    from scripts.reconstruction_schema import (
        CANONICAL_SIGNED_EVENT_CHANNEL_COUNT,
        EventSchemaContract,
        SIGNED_WAVELET_SCHEMA,
        event_contract_metadata,
        extract_signed_event_channels,
        resolve_event_schema_contract,
        validate_spike_imu_array,
    )
except ModuleNotFoundError:  # Direct ``python scripts/<entrypoint>.py`` execution.
    from reconstruction_schema import (  # type: ignore[no-redef]
        CANONICAL_SIGNED_EVENT_CHANNEL_COUNT,
        EventSchemaContract,
        SIGNED_WAVELET_SCHEMA,
        event_contract_metadata,
        extract_signed_event_channels,
        resolve_event_schema_contract,
        validate_spike_imu_array,
    )

SPIKE_SCHEMA = SIGNED_WAVELET_SCHEMA
EVENT_CHANNEL_COUNT = CANONICAL_SIGNED_EVENT_CHANNEL_COUNT
EVENTS_PER_AXIS = 5
AXIS_NAMES = ("x", "y", "z")
DEFAULT_SCALE_DIVISOR = 2.5
STANDARD_GRAVITY_M_S2 = 9.80665
OUTPUT_SUFFIX = "_reconstructed_accel_m_s2.npy"
METADATA_SUFFIX = "_reconstructed_accel_metadata.json"
MANIFEST_SUFFIX = "_reconstructed_accel_segments.csv"


class ReconstructionError(ValueError):
    """Raised when a segmentation package or reconstruction is invalid."""


@dataclass(frozen=True)
class SegmentationPackage:
    user: str
    action: str
    directory: Path
    prefix: str
    spike_path: Path
    labels_path: Path
    offsets_path: Path
    lengths_path: Path
    summary_path: Path


@dataclass(frozen=True)
class LoadedPackage:
    package: SegmentationPackage
    spike_imu: np.ndarray
    labels: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    summary: dict[str, object]
    schema_contract: EventSchemaContract
    sampling_rate_hz: float
    sampling_rate_source: str
    encoder_spec: dict[str, object]
    encoder_spec_sha256: str
    frequencies_hz: tuple[float, ...]
    wavelet_widths_samples: tuple[int, ...]


@dataclass(frozen=True)
class ReconstructionPaths:
    values: Path
    metadata: Path
    manifest: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct per-segment acceleration from schema-aware "
            "variable-length WritingRing SpikeIMU segmentation outputs."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=str,
        help=(
            "Dataset root, for example "
            "outputs/action0_rectified/low-pass/aligned-board-events. "
            "A path that already points at segmentation/ is also accepted."
        ),
    )
    parser.add_argument(
        "--users",
        nargs="+",
        required=True,
        help=(
            "One or more users, e.g. --users user_0 user_3. Numeric values "
            "such as 0 3 are normalized to user_0 user_3. Use --users all "
            "to process every user_* directory that exists."
        ),
    )
    parser.add_argument(
        "--sampling-rate-hz",
        type=float,
        default=None,
        help=(
            "Optional explicit reconstruction sampling rate. By default each "
            "user/action package must provide top-level sampling_rate_hz in "
            "its segmentation summary."
        ),
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Stored dtype for reconstructed acceleration (default: float64).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace already-complete reconstruction outputs.",
    )
    return parser.parse_args(argv)


def normalize_path(text: str | os.PathLike[str]) -> Path:
    return Path(str(text).replace("\\", "/")).expanduser().resolve()


def normalize_user_name(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ReconstructionError("user names must be nonempty")
    if text.isdigit():
        return f"user_{text}"
    return text


def resolve_segmentation_root(dataset_root: Path) -> Path:
    root = Path(dataset_root)
    nested = root / "segmentation"
    if nested.is_dir():
        return nested
    if root.is_dir() and root.name == "segmentation":
        return root
    if root.is_dir() and any(p.is_dir() and p.name.startswith("user_") for p in root.iterdir()):
        return root
    raise FileNotFoundError(
        "Could not find variable-length segmentation directory. Expected "
        f"{nested} or a path that already points at segmentation/: {root}"
    )


def natural_key(text: str) -> tuple[object, ...]:
    parts: list[object] = []
    token = ""
    numeric = False
    for char in text:
        if char.isdigit() == numeric:
            token += char
            continue
        if token:
            parts.append(int(token) if numeric else token.lower())
        token = char
        numeric = char.isdigit()
    if token:
        parts.append(int(token) if numeric else token.lower())
    return tuple(parts)


def resolve_users(segmentation_root: Path, requested: Sequence[str]) -> tuple[list[str], list[str]]:
    if not requested:
        raise ReconstructionError("at least one user is required")
    normalized = [normalize_user_name(value) for value in requested]
    if "all" in normalized:
        if len(normalized) != 1:
            raise ReconstructionError("--users all cannot be mixed with explicit user names")
        users = sorted(
            [p.name for p in segmentation_root.iterdir() if p.is_dir() and p.name.startswith("user_")],
            key=natural_key,
        )
        if not users:
            raise FileNotFoundError(f"No user_* directories found under {segmentation_root}")
        return users, []

    seen: set[str] = set()
    existing: list[str] = []
    missing: list[str] = []
    for user in normalized:
        if user in seen:
            continue
        seen.add(user)
        if (segmentation_root / user).is_dir():
            existing.append(user)
        else:
            missing.append(user)
    if not existing:
        raise FileNotFoundError(
            "None of the requested users exist under the segmentation root: "
            + ", ".join(normalized)
        )
    return existing, missing


def discover_packages(segmentation_root: Path, users: Sequence[str]) -> list[SegmentationPackage]:
    packages: list[SegmentationPackage] = []
    for user in users:
        user_dir = segmentation_root / user
        action_dirs = sorted(
            [p for p in user_dir.iterdir() if p.is_dir() and p.name.startswith("action_")],
            key=lambda p: natural_key(p.name),
        )
        for action_dir in action_dirs:
            spike_paths = sorted(action_dir.glob(f"{user}_action_*_spikeIMU.npy"), key=lambda p: natural_key(p.name))
            for spike_path in spike_paths:
                name = spike_path.name
                prefix = name[: -len("_spikeIMU.npy")]
                action = action_dir.name.removeprefix("action_")
                expected_prefix = f"{user}_action_{action}"
                if prefix != expected_prefix:
                    # Avoid accidentally processing unrelated copied artifacts.
                    continue
                package = SegmentationPackage(
                    user=user,
                    action=action,
                    directory=action_dir,
                    prefix=prefix,
                    spike_path=spike_path,
                    labels_path=action_dir / f"{prefix}_labels.npy",
                    offsets_path=action_dir / f"{prefix}_segment_offsets.npy",
                    lengths_path=action_dir / f"{prefix}_segment_lengths.npy",
                    summary_path=action_dir / f"{prefix}_segmentation_summary.json",
                )
                missing = [
                    path.name
                    for path in (
                        package.labels_path,
                        package.offsets_path,
                        package.lengths_path,
                        package.summary_path,
                    )
                    if not path.is_file()
                ]
                if missing:
                    raise FileNotFoundError(
                        f"Incomplete SpikeIMU segmentation package in {action_dir}: "
                        + ", ".join(missing)
                    )
                packages.append(package)
    return packages


def load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReconstructionError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReconstructionError(f"Expected a JSON object: {path}")
    return value


def load_encoder_contract(summary: dict[str, object], *, path: Path) -> tuple[dict[str, object], str, tuple[float, ...], tuple[int, ...]]:
    spec = summary.get("spike_encoder")
    digest = summary.get("spike_encoder_spec_sha256")
    if not isinstance(spec, dict) or not isinstance(digest, str) or len(digest) != 64:
        raise ReconstructionError(f"{path}: missing spike_encoder identity and spec hash")
    try:
        encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ReconstructionError(f"{path}: spike_encoder is not canonical JSON") from error
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise ReconstructionError(f"{path}: spike_encoder_spec_sha256 does not match spike_encoder")
    frequencies = spec.get("frequencies_hz")
    widths = spec.get("wavelet_widths_samples")
    if not isinstance(frequencies, list) or len(frequencies) != EVENTS_PER_AXIS:
        raise ReconstructionError(f"{path}: spike_encoder frequencies_hz must contain exactly five values")
    if not isinstance(widths, list) or len(widths) != EVENTS_PER_AXIS:
        raise ReconstructionError(f"{path}: spike_encoder wavelet_widths_samples must contain exactly five values")
    try:
        parsed_frequencies = tuple(float(value) for value in frequencies)
        parsed_widths = tuple(int(value) for value in widths)
    except (TypeError, ValueError) as error:
        raise ReconstructionError(f"{path}: invalid spike_encoder frequencies or widths") from error
    if any(not math.isfinite(value) or value <= 0 for value in parsed_frequencies):
        raise ReconstructionError(f"{path}: spike_encoder frequencies must be finite and positive")
    if any(isinstance(value, bool) or int(value) != value or value <= 0 for value in widths):
        raise ReconstructionError(f"{path}: spike_encoder widths must be positive integers")
    return spec, digest, parsed_frequencies, parsed_widths


def load_package(
    package: SegmentationPackage,
    *,
    sampling_rate_override: float | None,
) -> LoadedPackage:
    try:
        spike_imu = np.load(package.spike_path, allow_pickle=False)
        labels = np.load(package.labels_path, allow_pickle=False).astype(str)
        offsets = np.load(package.offsets_path, allow_pickle=False).astype(np.int64)
        lengths = np.load(package.lengths_path, allow_pickle=False).astype(np.int64)
    except (OSError, ValueError) as error:
        raise ReconstructionError(f"Could not load package {package.prefix}: {error}") from error

    summary = load_json(package.summary_path)

    spike_imu = np.asarray(spike_imu)
    observed_channel_count = spike_imu.shape[-1] if spike_imu.ndim >= 1 else None
    if summary.get("input_kind") not in (None, "spike-imu"):
        raise ReconstructionError(
            f"{package.summary_path}: expected input_kind='spike-imu', "
            f"got {summary.get('input_kind')!r}"
        )
    try:
        schema_contract = resolve_event_schema_contract(
            summary.get("feature_schema"),
            summary.get("channel_count", observed_channel_count),
            context=str(package.summary_path),
        )
        validate_spike_imu_array(
            spike_imu,
            schema_contract,
            context=str(package.spike_path),
        )
    except ValueError as error:
        raise ReconstructionError(str(error)) from error
    if spike_imu.ndim != 2:
        raise ReconstructionError(
            f"{package.spike_path}: expected shape (N, channels), got {spike_imu.shape}"
        )
    if len(spike_imu) == 0:
        raise ReconstructionError(f"{package.spike_path}: SpikeIMU aggregate is empty")
    if not np.isfinite(spike_imu).all():
        raise ReconstructionError(f"{package.spike_path}: SpikeIMU contains non-finite values")

    if labels.ndim != 1:
        raise ReconstructionError(f"{package.labels_path}: labels must be one-dimensional")
    if offsets.ndim != 1 or len(offsets) != len(labels) + 1:
        raise ReconstructionError(
            f"{package.offsets_path}: offsets must have segment_count + 1 entries"
        )
    if lengths.ndim != 1 or lengths.shape != labels.shape:
        raise ReconstructionError(
            f"{package.lengths_path}: lengths must have one entry per segment"
        )
    if offsets[0] != 0 or offsets[-1] != len(spike_imu):
        raise ReconstructionError(
            f"{package.offsets_path}: offsets do not span all SpikeIMU rows"
        )
    if np.any(np.diff(offsets) <= 0):
        raise ReconstructionError(
            f"{package.offsets_path}: every exported segment must have positive length"
        )
    if not np.array_equal(np.diff(offsets), lengths):
        raise ReconstructionError(
            f"{package.lengths_path}: segment_lengths must equal diff(segment_offsets)"
        )

    if sampling_rate_override is not None:
        sampling_rate_hz = finite_positive(sampling_rate_override, name="--sampling-rate-hz")
        sampling_rate_source = "cli_override"
    else:
        sampling_rate_hz = finite_positive(
            summary.get("sampling_rate_hz"),
            name=f"{package.summary_path.name} sampling_rate_hz",
        )
        sampling_rate_source = "segmentation_summary"

    encoder_spec, encoder_hash, frequencies_hz, wavelet_widths_samples = load_encoder_contract(
        summary, path=package.summary_path
    )

    return LoadedPackage(
        package=package,
        spike_imu=spike_imu,
        labels=labels,
        offsets=offsets,
        lengths=lengths,
        summary=summary,
        schema_contract=schema_contract,
        sampling_rate_hz=sampling_rate_hz,
        sampling_rate_source=sampling_rate_source,
        encoder_spec=encoder_spec,
        encoder_spec_sha256=encoder_hash,
        frequencies_hz=frequencies_hz,
        wavelet_widths_samples=wavelet_widths_samples,
    )


def finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ReconstructionError(f"{name} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ReconstructionError(f"{name} must be a positive finite number") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ReconstructionError(f"{name} must be a positive finite number")
    return number


def acceleration_wavelet_numpy(M: int, s: float) -> np.ndarray:
    """NumPy equivalent of vendor accelerationWavelet(M, s)."""
    M = int(M)
    s = float(s)
    if M <= 0 or not math.isfinite(s) or s <= 0.0:
        raise ReconstructionError(f"Invalid acceleration wavelet parameters M={M}, s={s}")
    x = (np.arange(M, dtype=np.float64) - (M - 1) / 2.0) / s
    support = (x > -0.5) & (x < 0.5)
    wavelet = support * (29.0 / 4.0) * x * (4.0 * x**2 - 1.0)
    return np.sqrt(1.0 / s) * wavelet


def reconstruction_kernels(
    *,
    wavelet_widths_samples: Sequence[int],
    frequencies_hz: Sequence[float] | None = None,
) -> tuple[np.ndarray, ...]:
    widths = np.asarray(wavelet_widths_samples, dtype=np.int64)
    if widths.shape != (EVENTS_PER_AXIS,) or np.any(widths <= 0):
        raise ReconstructionError("Exactly five positive wavelet widths are required")
    frequencies = tuple(frequencies_hz) if frequencies_hz is not None else tuple(widths)
    if len(frequencies) != EVENTS_PER_AXIS:
        raise ReconstructionError("Exactly five reconstruction frequencies are required")
    # Same adaptation as the notebook: use the full vendor-style 2*max(width)
    # kernel even for short label segments, and keep only the band si/2 delay
    # placement because current SpikeIMU events are already occurrence-aligned.
    max_kernel_length = int(2 * int(widths.max()))

    kernels: list[np.ndarray] = []
    for frequency_hz, width in zip(frequencies, widths, strict=True):
        impulse_index = max_kernel_length // 2 - int(width) // 2
        if impulse_index < 0 or impulse_index >= max_kernel_length:
            raise ReconstructionError(f"Kernel placement failed for band {frequency_hz:g}")
        impulse = signal.unit_impulse(max_kernel_length, impulse_index)
        wavelet = acceleration_wavelet_numpy(int(width), int(width))[::-1]
        kernels.append(signal.convolve(impulse, wavelet, mode="same"))
    return tuple(kernels)


def reconstruct_segment_events(
    segment_events: np.ndarray,
    *,
    kernels: Sequence[np.ndarray],
    scale_divisor: float = DEFAULT_SCALE_DIVISOR,
) -> np.ndarray:
    events = np.asarray(segment_events, dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != EVENT_CHANNEL_COUNT or len(events) == 0:
        raise ReconstructionError(
            f"Expected nonempty event matrix shape (T, 15), got {events.shape}"
        )
    if not np.isfinite(events).all():
        raise ReconstructionError("Segment event matrix contains non-finite values")
    if len(kernels) != EVENTS_PER_AXIS:
        raise ReconstructionError("Exactly five reconstruction kernels are required")
    divisor = finite_positive(scale_divisor, name="reconstruction scale divisor")

    events_3d = events.reshape(len(events), 3, EVENTS_PER_AXIS)
    reconstructed_g = np.zeros((len(events), 3), dtype=np.float64)
    for axis_index in range(3):
        for band_index, kernel in enumerate(kernels):
            reconstructed_g[:, axis_index] += signal.convolve(
                events_3d[:, axis_index, band_index],
                np.asarray(kernel, dtype=np.float64),
                mode="same",
            ) / divisor
    return reconstructed_g * STANDARD_GRAVITY_M_S2


def reconstruct_package(loaded: LoadedPackage) -> np.ndarray:
    kernels = reconstruction_kernels(
        wavelet_widths_samples=loaded.wavelet_widths_samples,
        frequencies_hz=loaded.frequencies_hz,
    )
    reconstructed = np.zeros((len(loaded.spike_imu), 3), dtype=np.float64)
    for segment_index in range(len(loaded.labels)):
        start = int(loaded.offsets[segment_index])
        stop = int(loaded.offsets[segment_index + 1])
        signed_events = extract_signed_event_channels(
            loaded.spike_imu[start:stop],
            loaded.schema_contract,
            context=f"{loaded.package.summary_path} segment {segment_index}",
        )
        reconstructed[start:stop] = reconstruct_segment_events(signed_events, kernels=kernels)
    if not np.isfinite(reconstructed).all():
        raise ReconstructionError(
            f"Non-finite reconstruction produced for {loaded.package.prefix}"
        )
    return reconstructed


def output_paths(package: SegmentationPackage) -> ReconstructionPaths:
    return ReconstructionPaths(
        values=package.directory / f"{package.prefix}{OUTPUT_SUFFIX}",
        metadata=package.directory / f"{package.prefix}{METADATA_SUFFIX}",
        manifest=package.directory / f"{package.prefix}{MANIFEST_SUFFIX}",
    )


def check_existing_outputs(
    paths: ReconstructionPaths,
    package: SegmentationPackage,
    *,
    overwrite: bool,
    output_dtype: str,
    sampling_rate_override: float | None,
) -> str:
    existing = [path for path in (paths.values, paths.metadata, paths.manifest) if path.exists()]
    if not existing:
        return "write"
    for path in existing:
        if not path.is_file():
            raise ReconstructionError(f"Output path exists but is not a file: {path}")
    if overwrite:
        return "write"
    if len(existing) != 3:
        raise ReconstructionError(
            "Partial reconstruction outputs already exist; use --overwrite or clean them first: "
            + ", ".join(str(path) for path in existing)
        )

    metadata = load_json(paths.metadata)
    source = metadata.get("source")
    output = metadata.get("output")
    reconstruction = metadata.get("reconstruction")
    if not isinstance(source, dict) or not isinstance(output, dict) or not isinstance(reconstruction, dict):
        raise ReconstructionError(
            f"Existing reconstruction metadata is malformed; use --overwrite: {paths.metadata}"
        )

    expected_source_hashes = {
        "spike_imu_sha256": sha256_file(package.spike_path),
        "labels_sha256": sha256_file(package.labels_path),
        "segment_offsets_sha256": sha256_file(package.offsets_path),
        "segment_lengths_sha256": sha256_file(package.lengths_path),
        "segmentation_summary_sha256": sha256_file(package.summary_path),
    }
    stale_fields = [
        key for key, expected in expected_source_hashes.items() if source.get(key) != expected
    ]
    if stale_fields:
        raise ReconstructionError(
            "Existing reconstruction is stale relative to current segmentation inputs "
            f"({', '.join(stale_fields)}); rerun with --overwrite"
        )

    if output.get("sha256") != sha256_file(paths.values):
        raise ReconstructionError(
            f"Existing reconstructed acceleration hash does not match metadata; use --overwrite: {paths.values}"
        )
    if output.get("segment_manifest_sha256") != sha256_file(paths.manifest):
        raise ReconstructionError(
            f"Existing reconstruction manifest hash does not match metadata; use --overwrite: {paths.manifest}"
        )
    if output.get("dtype") != output_dtype:
        raise ReconstructionError(
            f"Existing reconstruction dtype is {output.get('dtype')!r}, requested {output_dtype!r}; "
            "rerun with --overwrite"
        )
    if sampling_rate_override is not None:
        requested_rate = finite_positive(sampling_rate_override, name="--sampling-rate-hz")
        existing_rate = finite_positive(
            reconstruction.get("sampling_rate_hz"),
            name="existing reconstruction sampling_rate_hz",
        )
        if not math.isclose(existing_rate, requested_rate, rel_tol=0.0, abs_tol=1e-12):
            raise ReconstructionError(
                f"Existing reconstruction uses {existing_rate:g} Hz but {requested_rate:g} Hz was requested; "
                "rerun with --overwrite"
            )
    return "skip"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp.npy", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        np.save(temp_path, values, allow_pickle=False)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def manifest_text(loaded: LoadedPackage) -> str:
    from io import StringIO

    output = StringIO()
    fieldnames = (
        "segment_index",
        "label",
        "start_row",
        "stop_row_exclusive",
        "length_samples",
        "duration_s",
        "event_nonzero_count",
        "event_l1_sum",
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for segment_index, label in enumerate(loaded.labels.tolist()):
        start = int(loaded.offsets[segment_index])
        stop = int(loaded.offsets[segment_index + 1])
        events = extract_signed_event_channels(
            loaded.spike_imu[start:stop],
            loaded.schema_contract,
            context=f"{loaded.package.summary_path} segment {segment_index}",
        ).astype(np.float64, copy=False)
        writer.writerow(
            {
                "segment_index": segment_index,
                "label": str(label),
                "start_row": start,
                "stop_row_exclusive": stop,
                "length_samples": stop - start,
                "duration_s": format((stop - start) / loaded.sampling_rate_hz, ".12g"),
                "event_nonzero_count": int(np.count_nonzero(events)),
                "event_l1_sum": format(float(np.abs(events).sum()), ".12g"),
            }
        )
    return output.getvalue()


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def metadata_object(
    loaded: LoadedPackage,
    paths: ReconstructionPaths,
    reconstructed: np.ndarray,
    *,
    dataset_root: Path,
    output_dtype: str,
) -> dict[str, object]:
    package = loaded.package
    return {
        "schema_version": 1,
        "artifact_type": "segmentwise_custom_wavelet_acceleration_reconstruction",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "user": package.user,
            "action": package.action,
            "package_prefix": package.prefix,
        },
        "source": {
            **event_contract_metadata(loaded.schema_contract),
            "dataset_root": str(dataset_root),
            "spike_imu_path": relative_or_absolute(package.spike_path, dataset_root),
            "spike_imu_sha256": sha256_file(package.spike_path),
            "labels_path": relative_or_absolute(package.labels_path, dataset_root),
            "labels_sha256": sha256_file(package.labels_path),
            "segment_offsets_path": relative_or_absolute(package.offsets_path, dataset_root),
            "segment_offsets_sha256": sha256_file(package.offsets_path),
            "segment_lengths_path": relative_or_absolute(package.lengths_path, dataset_root),
            "segment_lengths_sha256": sha256_file(package.lengths_path),
            "segmentation_summary_path": relative_or_absolute(package.summary_path, dataset_root),
            "segmentation_summary_sha256": sha256_file(package.summary_path),
            "input_kind": loaded.summary.get("input_kind"),
            "feature_schema": loaded.summary.get("feature_schema"),
            "spike_imu_shape": [int(v) for v in loaded.spike_imu.shape],
            "event_channel_slice": [
                0,
                loaded.schema_contract.stored_event_channel_count,
            ],
            "published_event_values_used_as_is": True,
        },
        "segmentation": {
            "segment_count": int(len(loaded.labels)),
            "row_count": int(len(loaded.spike_imu)),
            "segmentwise_reconstruction": True,
            "convolution_crosses_segment_boundaries": False,
            "outside_segment_context": "zero",
        },
        "reconstruction": {
            **event_contract_metadata(loaded.schema_contract),
            "method": "custom_wavelet_event_convolution_sum",
            "wavelet": "accelerationWavelet",
            "frequencies_hz": list(loaded.frequencies_hz),
            "wavelet_widths_samples": list(loaded.wavelet_widths_samples),
            "spike_encoder_spec_sha256": loaded.encoder_spec_sha256,
            "source_spike_encoder_spec_sha256": loaded.encoder_spec_sha256,
            "sampling_rate_hz": float(loaded.sampling_rate_hz),
            "sampling_rate_source": loaded.sampling_rate_source,
            "scale_divisor": float(DEFAULT_SCALE_DIVISOR),
            "intermediate_acceleration_unit": "g",
            "standard_gravity_m_s2": float(STANDARD_GRAVITY_M_S2),
            "output_acceleration_unit": "m/s^2",
            "band_delay_compensation": "si/2 impulse placement",
            "max_filter_confirmation_delay_compensation": False,
            "occurrence_alignment_note": (
                "Current WritingRing SpikeIMU event rows are treated as occurrence-aligned; "
                "the vendor max-filter confirmation delay is not applied a second time."
            ),
            "rectification_note": (
                "Reconstruction uses the published event amplitudes exactly as stored. If the "
                "source artifact was AbsRectify-transformed, lost event polarity cannot be recovered."
            ),
        },
        "output": {
            "path": relative_or_absolute(paths.values, dataset_root),
            "sha256": sha256_file(paths.values),
            "shape": [int(v) for v in reconstructed.shape],
            "dtype": output_dtype,
            "channel_names": [
                "reconstructed_acceleration_x_m_s2",
                "reconstructed_acceleration_y_m_s2",
                "reconstructed_acceleration_z_m_s2",
            ],
            "units": ["m/s^2", "m/s^2", "m/s^2"],
            "row_alignment": "one row per source SpikeIMU row",
            "appended_to_spike_imu": False,
            "segment_manifest_path": relative_or_absolute(paths.manifest, dataset_root),
            "segment_manifest_sha256": sha256_file(paths.manifest),
        },
    }


def publish_package(
    loaded: LoadedPackage,
    reconstructed: np.ndarray,
    *,
    dataset_root: Path,
    output_dtype: str,
    overwrite: bool,
) -> tuple[str, ReconstructionPaths]:
    paths = output_paths(loaded.package)
    mode = check_existing_outputs(
        paths,
        loaded.package,
        overwrite=overwrite,
        output_dtype=output_dtype,
        sampling_rate_override=(
            loaded.sampling_rate_hz if loaded.sampling_rate_source == "cli_override" else None
        ),
    )
    if mode == "skip":
        return "skipped-existing", paths

    dtype = np.float32 if output_dtype == "float32" else np.float64
    stored = np.asarray(reconstructed, dtype=dtype)
    if stored.shape != (len(loaded.spike_imu), 3):
        raise ReconstructionError(
            f"Output shape mismatch for {loaded.package.prefix}: {stored.shape}"
        )

    atomic_save_npy(paths.values, stored)
    try:
        manifest = manifest_text(loaded)
        atomic_write_text(paths.manifest, manifest)
        metadata = metadata_object(
            loaded,
            paths,
            stored,
            dataset_root=dataset_root,
            output_dtype=output_dtype,
        )
        atomic_write_text(paths.metadata, json.dumps(metadata, indent=2, sort_keys=False) + "\n")
    except Exception:
        # Do not leave a newly-created partial publication behind.
        paths.values.unlink(missing_ok=True)
        paths.manifest.unlink(missing_ok=True)
        paths.metadata.unlink(missing_ok=True)
        raise

    return "written", paths


def run(args: argparse.Namespace) -> int:
    dataset_root = normalize_path(args.dataset_root)
    segmentation_root = resolve_segmentation_root(dataset_root)
    users, missing_users = resolve_users(segmentation_root, args.users)

    print(f"Dataset root:      {dataset_root}")
    print(f"Segmentation root: {segmentation_root}")
    print(f"Users selected:    {', '.join(users)}")
    for user in missing_users:
        print(f"WARNING: requested user does not exist and will be skipped: {user}")

    packages = discover_packages(segmentation_root, users)
    if not packages:
        raise FileNotFoundError(
            "No complete *_spikeIMU.npy segmentation packages were discovered for the selected users"
        )

    written = 0
    skipped = 0
    failed = 0
    print(f"Packages found:    {len(packages)}")

    for package in packages:
        print(f"\n[{package.user} / action {package.action}] {package.directory}")
        try:
            loaded = load_package(
                package,
                sampling_rate_override=args.sampling_rate_hz,
            )
            paths = output_paths(package)
            mode = check_existing_outputs(
                paths,
                package,
                overwrite=bool(args.overwrite),
                output_dtype=args.output_dtype,
                sampling_rate_override=args.sampling_rate_hz,
            )
            if mode == "skip":
                skipped += 1
                print(f"  SKIP complete output already exists: {paths.values.name}")
                continue

            print(
                f"  source: {loaded.spike_imu.shape[0]} rows, "
                f"{len(loaded.labels)} segments, {loaded.sampling_rate_hz:g} Hz"
            )
            reconstructed = reconstruct_package(loaded)
            status, paths = publish_package(
                loaded,
                reconstructed,
                dataset_root=dataset_root,
                output_dtype=args.output_dtype,
                overwrite=bool(args.overwrite),
            )
            if status == "written":
                written += 1
                print(f"  WROTE {paths.values.name} {tuple(reconstructed.shape)}")
                print(f"  WROTE {paths.metadata.name}")
                print(f"  WROTE {paths.manifest.name}")
            else:
                skipped += 1
                print(f"  SKIP {paths.values.name}")
        except Exception as error:
            failed += 1
            print(f"  ERROR: {error}")

    print("\nSummary")
    print(f"  written: {written}")
    print(f"  skipped: {skipped}")
    print(f"  failed:  {failed}")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ReconstructionError, OSError) as error:
        print(f"ERROR: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
