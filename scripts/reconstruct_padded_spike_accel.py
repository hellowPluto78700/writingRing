#!/usr/bin/env python3
"""Reconstruct and right-pad acceleration for every padded WritingRing segment.

Input
-----
Give a dataset combination root, for example:

    outputs/action0_rectified/low-pass/aligned-board-events

The script automatically scans every user/action package under:

    <dataset_root>/segmentation/

and requires the matching canonical padded package under:

    <dataset_root>/segmentation_padded/

For each retained padded segment, reconstruction is performed from the original
UNPADDED SpikeIMU event channels after schema-specific decoding to canonical
signed (T, 15) events, independently inside that segment.  The resulting
(T, 3) acceleration is then right-padded to the exact target length used by the
matching ``*_paddedSpikeIMU.npy`` package.  Convolution never sees padding
samples and never crosses segment boundaries.

Output
------
Next to each matching ``*_paddedSpikeIMU.npy`` this script writes:

    <stem>_padded_reconstructed_accel_m_s2.npy   # (S, target_length, 3)
    <stem>_padded_reconstructed_accel_metadata.json

The reconstruction is NOT appended to SpikeIMU.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
from scipy import signal

try:
    from scripts.reconstruction_schema import (
        CANONICAL_SIGNED_EVENT_CHANNEL_COUNT,
        SIGNED_WAVELET_SCHEMA,
        event_contract_metadata,
        extract_signed_event_channels,
        resolve_event_schema_contract,
        validate_spike_imu_array,
    )
except ModuleNotFoundError:  # Direct ``python scripts/<entrypoint>.py`` execution.
    from reconstruction_schema import (  # type: ignore[no-redef]
        CANONICAL_SIGNED_EVENT_CHANNEL_COUNT,
        SIGNED_WAVELET_SCHEMA,
        event_contract_metadata,
        extract_signed_event_channels,
        resolve_event_schema_contract,
        validate_spike_imu_array,
    )

SPIKE_SCHEMA = SIGNED_WAVELET_SCHEMA
EVENT_CHANNEL_COUNT = CANONICAL_SIGNED_EVENT_CHANNEL_COUNT
EVENTS_PER_AXIS = 5
DEFAULT_SCALE_DIVISOR = 2.5
STANDARD_GRAVITY_M_S2 = 9.80665
OUTPUT_SUFFIX = "_padded_reconstructed_accel_m_s2.npy"
METADATA_SUFFIX = "_padded_reconstructed_accel_metadata.json"


class ReconstructionError(ValueError):
    """Raised when source, padded package, or reconstruction is invalid."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        help=(
            "Combination root containing segmentation/ and segmentation_padded/, "
            "e.g. outputs/action0_rectified/low-pass/aligned-board-events"
        ),
    )
    parser.add_argument(
        "--sampling-rate-hz",
        type=float,
        default=None,
        help=(
            "Optional reconstruction sampling-rate override. Otherwise use each "
            "variable-length segmentation summary's sampling_rate_hz."
        ),
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing padded reconstruction outputs.",
    )
    return parser.parse_args(argv)


def normalize_path(value: str | os.PathLike[str]) -> Path:
    return Path(str(value).replace("\\", "/")).expanduser().resolve()


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


def load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReconstructionError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReconstructionError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npy", dir=path.parent
    )
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
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def natural_key(text: str) -> tuple[object, ...]:
    parts: list[object] = []
    token = ""
    numeric = False
    for char in text:
        current_numeric = char.isdigit()
        if token and current_numeric != numeric:
            parts.append(int(token) if numeric else token.lower())
            token = ""
        token += char
        numeric = current_numeric
    if token:
        parts.append(int(token) if numeric else token.lower())
    return tuple(parts)


def acceleration_wavelet_numpy(M: int, s: float) -> np.ndarray:
    """NumPy equivalent of vendor accelerationWavelet(M, s)."""
    M = int(M)
    s = float(s)
    if M <= 0 or not math.isfinite(s) or s <= 0.0:
        raise ReconstructionError(f"Invalid wavelet parameters M={M}, s={s}")
    x = (np.arange(M, dtype=np.float64) - (M - 1) / 2.0) / s
    support = (x > -0.5) & (x < 0.5)
    wavelet = support * (29.0 / 4.0) * x * (4.0 * x**2 - 1.0)
    return np.sqrt(1.0 / s) * wavelet


def reconstruction_kernels(
    wavelet_widths_samples: Sequence[int],
    frequencies_hz: Sequence[float] | None = None,
) -> tuple[np.ndarray, ...]:
    widths = np.asarray(wavelet_widths_samples, dtype=np.int64)
    if widths.shape != (EVENTS_PER_AXIS,) or np.any(widths <= 0):
        raise ReconstructionError("Exactly five positive wavelet widths are required")
    frequencies = tuple(frequencies_hz) if frequencies_hz is not None else tuple(widths)
    if len(frequencies) != EVENTS_PER_AXIS:
        raise ReconstructionError("Exactly five reconstruction frequencies are required")
    max_kernel_length = 2 * int(widths.max())
    kernels: list[np.ndarray] = []
    for frequency_hz, width in zip(frequencies, widths, strict=True):
        impulse_index = max_kernel_length // 2 - int(width) // 2
        if not 0 <= impulse_index < max_kernel_length:
            raise ReconstructionError(
                f"Kernel placement failed for {frequency_hz:g} Hz"
            )
        impulse = signal.unit_impulse(max_kernel_length, impulse_index)
        wavelet = acceleration_wavelet_numpy(int(width), int(width))[::-1]
        kernels.append(signal.convolve(impulse, wavelet, mode="same"))
    return tuple(kernels)


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


def reconstruct_segment_events(
    segment_events: np.ndarray,
    *,
    kernels: Sequence[np.ndarray],
) -> np.ndarray:
    events = np.asarray(segment_events, dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != EVENT_CHANNEL_COUNT or len(events) == 0:
        raise ReconstructionError(
            f"Expected nonempty segment event matrix (T, 15), got {events.shape}"
        )
    if not np.isfinite(events).all():
        raise ReconstructionError("Segment events contain non-finite values")

    events_3d = events.reshape(len(events), 3, EVENTS_PER_AXIS)
    reconstructed_g = np.zeros((len(events), 3), dtype=np.float64)
    for axis_index in range(3):
        for band_index, kernel in enumerate(kernels):
            reconstructed_g[:, axis_index] += signal.convolve(
                events_3d[:, axis_index, band_index],
                np.asarray(kernel, dtype=np.float64),
                mode="same",
            ) / DEFAULT_SCALE_DIVISOR
    return reconstructed_g * STANDARD_GRAVITY_M_S2


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ReconstructionError(f"Invalid boolean value in padding manifest: {value!r}")


def parse_manifest_int(value: object, *, field: str, path: Path) -> int:
    """Parse an integer-valued CSV field, accepting pandas-style ``0.0``."""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        raise ReconstructionError(
            f"{path}: required integer field {field!r} is empty: {value!r}"
        )
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError as error:
        raise ReconstructionError(
            f"{path}: field {field!r} must be integer-valued, got {value!r}"
        ) from error
    if not math.isfinite(number) or not number.is_integer():
        raise ReconstructionError(
            f"{path}: field {field!r} must be integer-valued, got {value!r}"
        )
    return int(number)


def load_padding_manifest(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ReconstructionError(f"Could not read padding manifest {path}: {error}") from error
    if not rows:
        raise ReconstructionError(f"Padding manifest has no rows: {path}")
    required = {"segment_index", "output_segment_index", "exported", "original_length", "target_length"}
    missing = required.difference(rows[0])
    if missing:
        raise ReconstructionError(
            f"Padding manifest {path} missing columns: {sorted(missing)}"
        )
    return rows


def discover_source_packages(segmentation_root: Path) -> list[tuple[str, str, Path, str]]:
    packages: list[tuple[str, str, Path, str]] = []
    users = sorted(
        [p for p in segmentation_root.iterdir() if p.is_dir() and p.name.startswith("user_")],
        key=lambda p: natural_key(p.name),
    )
    if not users:
        raise ReconstructionError(f"No user_* directories found under {segmentation_root}")

    for user_dir in users:
        actions = sorted(
            [p for p in user_dir.iterdir() if p.is_dir() and p.name.startswith("action_")],
            key=lambda p: natural_key(p.name),
        )
        for action_dir in actions:
            action = action_dir.name.removeprefix("action_")
            stem = f"{user_dir.name}_action_{action}"
            if (action_dir / f"{stem}_spikeIMU.npy").is_file():
                packages.append((user_dir.name, action, action_dir, stem))
    if not packages:
        raise ReconstructionError(f"No SpikeIMU packages found below {segmentation_root}")
    return packages


def require_files(paths: Sequence[Path], *, context: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ReconstructionError(f"Missing required file(s) for {context}: " + ", ".join(missing))


def process_package(
    *,
    dataset_root: Path,
    padded_root: Path,
    user: str,
    action: str,
    source_dir: Path,
    stem: str,
    sampling_rate_override: float | None,
    output_dtype: str,
    overwrite: bool,
) -> str:
    spike_path = source_dir / f"{stem}_spikeIMU.npy"
    labels_path = source_dir / f"{stem}_labels.npy"
    offsets_path = source_dir / f"{stem}_segment_offsets.npy"
    lengths_path = source_dir / f"{stem}_segment_lengths.npy"
    source_summary_path = source_dir / f"{stem}_segmentation_summary.json"

    padded_dir = padded_root / user / f"action_{action}"
    padded_spike_path = padded_dir / f"{stem}_paddedSpikeIMU.npy"
    padded_labels_path = padded_dir / f"{stem}_labels.npy"
    valid_lengths_path = padded_dir / f"{stem}_valid_lengths.npy"
    valid_mask_path = padded_dir / f"{stem}_valid_mask.npy"
    padding_manifest_path = padded_dir / f"{stem}_padding_manifest.csv"
    padding_summary_path = padded_dir / f"{stem}_padding_summary.json"

    require_files(
        [
            spike_path, labels_path, offsets_path, lengths_path, source_summary_path,
            padded_spike_path, padded_labels_path, valid_lengths_path, valid_mask_path,
            padding_manifest_path, padding_summary_path,
        ],
        context=f"{user}/action_{action}",
    )

    spike_imu = np.load(spike_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False).astype(str)
    offsets = np.load(offsets_path, allow_pickle=False).astype(np.int64)
    lengths = np.load(lengths_path, allow_pickle=False).astype(np.int64)
    padded_spike = np.load(padded_spike_path, allow_pickle=False)
    padded_labels = np.load(padded_labels_path, allow_pickle=False).astype(str)
    valid_lengths = np.load(valid_lengths_path, allow_pickle=False).astype(np.int64)
    valid_mask = np.load(valid_mask_path, allow_pickle=False)
    source_summary = load_json(source_summary_path)
    padding_summary = load_json(padding_summary_path)
    manifest_rows = load_padding_manifest(padding_manifest_path)
    encoder_spec, encoder_spec_sha256, frequencies_hz, wavelet_widths_samples = load_encoder_contract(
        source_summary, path=source_summary_path
    )

    if source_summary.get("input_kind") not in (None, "spike-imu"):
        raise ReconstructionError(f"{source_summary_path}: not a SpikeIMU segmentation summary")
    try:
        source_contract = resolve_event_schema_contract(
            source_summary.get("feature_schema"),
            source_summary.get("channel_count", spike_imu.shape[-1]),
            context=str(source_summary_path),
        )
        padded_contract = resolve_event_schema_contract(
            padding_summary.get("feature_schema"),
            padding_summary.get("channel_count", padded_spike.shape[-1]),
            context=str(padding_summary_path),
        )
        validate_spike_imu_array(
            spike_imu,
            source_contract,
            context=str(spike_path),
        )
        validate_spike_imu_array(
            padded_spike,
            padded_contract,
            context=str(padded_spike_path),
        )
    except ValueError as error:
        raise ReconstructionError(str(error)) from error

    if spike_imu.ndim != 2:
        raise ReconstructionError(f"{spike_path}: expected (N, channels), got {spike_imu.shape}")
    if labels.ndim != 1 or lengths.ndim != 1 or labels.shape != lengths.shape:
        raise ReconstructionError(f"{stem}: source labels/lengths shapes are inconsistent")
    if offsets.ndim != 1 or len(offsets) != len(labels) + 1:
        raise ReconstructionError(f"{offsets_path}: expected segment_count+1 offsets")
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(spike_imu):
        raise ReconstructionError(f"{offsets_path}: offsets do not span SpikeIMU")
    if not np.array_equal(np.diff(offsets), lengths):
        raise ReconstructionError(f"{lengths_path}: lengths != diff(offsets)")

    if padded_spike.ndim != 3:
        raise ReconstructionError(
            f"{padded_spike_path}: expected (S,target,channels), got {padded_spike.shape}"
        )
    retained_count, target_length, _ = padded_spike.shape
    if padded_labels.shape != (retained_count,):
        raise ReconstructionError(f"{padded_labels_path}: wrong shape {padded_labels.shape}")
    if valid_lengths.shape != (retained_count,):
        raise ReconstructionError(f"{valid_lengths_path}: wrong shape {valid_lengths.shape}")
    if valid_mask.shape != (retained_count, target_length):
        raise ReconstructionError(f"{valid_mask_path}: wrong shape {valid_mask.shape}")
    if valid_mask.dtype != np.bool_:
        valid_mask = valid_mask.astype(np.bool_)

    summary_target = padding_summary.get("target_length")
    if int(summary_target) != target_length:
        raise ReconstructionError(
            f"{padding_summary_path}: target_length={summary_target!r} does not match padded array {target_length}"
        )
    if padding_summary.get("padding_side") != "right":
        raise ReconstructionError(f"{padding_summary_path}: expected right padding")
    if padding_summary.get("overflow_policy") != "skip":
        raise ReconstructionError(f"{padding_summary_path}: expected overflow_policy='skip'")

    exported_rows: list[tuple[int, int, int]] = []
    for row in manifest_rows:
        if not parse_bool(row["exported"]):
            continue
        source_index = parse_manifest_int(
            row["segment_index"], field="segment_index", path=padding_manifest_path
        )
        output_index = parse_manifest_int(
            row["output_segment_index"], field="output_segment_index", path=padding_manifest_path
        )
        original_length = parse_manifest_int(
            row["original_length"], field="original_length", path=padding_manifest_path
        )
        row_target = parse_manifest_int(
            row["target_length"], field="target_length", path=padding_manifest_path
        )
        if row_target != target_length:
            raise ReconstructionError(f"{padding_manifest_path}: inconsistent target_length")
        exported_rows.append((source_index, output_index, original_length))

    if len(exported_rows) != retained_count:
        raise ReconstructionError(
            f"{padding_manifest_path}: exported row count {len(exported_rows)} != padded segment count {retained_count}"
        )
    exported_rows.sort(key=lambda item: item[1])
    if [item[1] for item in exported_rows] != list(range(retained_count)):
        raise ReconstructionError(f"{padding_manifest_path}: output_segment_index is not contiguous")

    sampling_rate_hz = (
        finite_positive(sampling_rate_override, name="--sampling-rate-hz")
        if sampling_rate_override is not None
        else finite_positive(source_summary.get("sampling_rate_hz"), name="sampling_rate_hz")
    )
    kernels = reconstruction_kernels(
        wavelet_widths_samples=wavelet_widths_samples,
        frequencies_hz=frequencies_hz,
    )

    reconstructed = np.zeros((retained_count, target_length, 3), dtype=np.float64)
    for source_index, output_index, original_length in exported_rows:
        if not 0 <= source_index < len(lengths):
            raise ReconstructionError(f"{padding_manifest_path}: invalid segment_index {source_index}")
        expected_length = int(lengths[source_index])
        if expected_length != original_length:
            raise ReconstructionError(
                f"{padding_manifest_path}: source segment {source_index} length mismatch"
            )
        if original_length > target_length:
            raise ReconstructionError(
                f"{padding_manifest_path}: exported overlong segment {source_index}"
            )
        if int(valid_lengths[output_index]) != original_length:
            raise ReconstructionError(
                f"{valid_lengths_path}: output segment {output_index} length mismatch"
            )
        expected_mask = np.arange(target_length) < original_length
        if not np.array_equal(valid_mask[output_index], expected_mask):
            raise ReconstructionError(
                f"{valid_mask_path}: output segment {output_index} is not canonical right padding"
            )
        if str(padded_labels[output_index]) != str(labels[source_index]):
            raise ReconstructionError(
                f"{padded_labels_path}: label mismatch for output segment {output_index}"
            )

        start = int(offsets[source_index])
        stop = int(offsets[source_index + 1])
        signed_events = extract_signed_event_channels(
            spike_imu[start:stop],
            source_contract,
            context=f"{source_summary_path} segment {source_index}",
        )
        rec = reconstruct_segment_events(signed_events, kernels=kernels)
        reconstructed[output_index, :original_length] = rec
        # [original_length:] intentionally remains exact zero padding.

    if not np.isfinite(reconstructed).all():
        raise ReconstructionError(f"{stem}: reconstruction contains non-finite values")
    if retained_count and np.any(reconstructed[~valid_mask] != 0.0):
        raise ReconstructionError(f"{stem}: padded reconstruction region is not zero")

    output_path = padded_dir / f"{stem}{OUTPUT_SUFFIX}"
    metadata_path = padded_dir / f"{stem}{METADATA_SUFFIX}"
    if (output_path.exists() or metadata_path.exists()) and not overwrite:
        raise ReconstructionError(
            f"Output already exists for {stem}; rerun with --overwrite: {output_path}"
        )

    dtype = np.float32 if output_dtype == "float32" else np.float64
    stored = reconstructed.astype(dtype, copy=False)
    atomic_save_npy(output_path, stored)
    try:
        metadata = {
            "schema_version": 1,
            "artifact_type": "padded_segmentwise_custom_wavelet_acceleration_reconstruction",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": {"user": user, "action": action, "package_prefix": stem},
            "source": {
                **event_contract_metadata(source_contract),
                "variable_spike_imu": str(spike_path.relative_to(dataset_root)),
                "variable_spike_imu_sha256": sha256_file(spike_path),
                "segment_offsets": str(offsets_path.relative_to(dataset_root)),
                "segment_offsets_sha256": sha256_file(offsets_path),
                "segment_lengths": str(lengths_path.relative_to(dataset_root)),
                "segment_lengths_sha256": sha256_file(lengths_path),
                "segmentation_summary": str(source_summary_path.relative_to(dataset_root)),
                "segmentation_summary_sha256": sha256_file(source_summary_path),
                "padded_spike_imu": str(padded_spike_path.relative_to(dataset_root)),
                "padded_spike_imu_sha256": sha256_file(padded_spike_path),
                "valid_lengths": str(valid_lengths_path.relative_to(dataset_root)),
                "valid_lengths_sha256": sha256_file(valid_lengths_path),
                "valid_mask": str(valid_mask_path.relative_to(dataset_root)),
                "valid_mask_sha256": sha256_file(valid_mask_path),
                "padding_manifest": str(padding_manifest_path.relative_to(dataset_root)),
                "padding_manifest_sha256": sha256_file(padding_manifest_path),
                "padding_summary": str(padding_summary_path.relative_to(dataset_root)),
                "padding_summary_sha256": sha256_file(padding_summary_path),
            },
            "reconstruction": {
                "input_event_slice": [0, source_contract.stored_event_channel_count],
                **event_contract_metadata(source_contract),
                "method": "custom_wavelet_event_convolution_sum",
                "wavelet": "accelerationWavelet",
                "frequencies_hz": list(frequencies_hz),
                "wavelet_widths_samples": list(wavelet_widths_samples),
                "spike_encoder_spec_sha256": encoder_spec_sha256,
                "source_spike_encoder_spec_sha256": encoder_spec_sha256,
                "sampling_rate_hz": sampling_rate_hz,
                "scale_divisor": DEFAULT_SCALE_DIVISOR,
                "standard_gravity_m_s2": STANDARD_GRAVITY_M_S2,
                "output_unit": "m/s^2",
                "segmentwise_before_padding": True,
                "convolution_crosses_segment_boundaries": False,
                "convolution_sees_padding": False,
                "max_filter_confirmation_delay_compensation": False,
            },
            "padding": {
                "source": "existing WritingRing segmentation_padded package",
                "input_feature_schema": padded_contract.feature_schema,
                "stored_channel_count": padded_contract.total_channel_count,
                "target_length": target_length,
                "padding_side": "right",
                "padding_value_m_s2": 0.0,
                "overflow_policy": "follow padding_manifest; skipped source segments remain absent",
                "retained_segment_count": retained_count,
                "valid_mask_reused_for_alignment": True,
            },
            "output": {
                "path": str(output_path.relative_to(dataset_root)),
                "sha256": sha256_file(output_path),
                "shape": [int(v) for v in stored.shape],
                "dtype": output_dtype,
                "channel_names": [
                    "reconstructed_acceleration_x_m_s2",
                    "reconstructed_acceleration_y_m_s2",
                    "reconstructed_acceleration_z_m_s2",
                ],
                "units": ["m/s^2", "m/s^2", "m/s^2"],
                "alignment": "same segment axis, time axis, labels, valid_lengths, and valid_mask as paddedSpikeIMU",
                "appended_to_spike_imu": False,
            },
            "rectification_note": (
                "Reconstruction uses event amplitudes exactly as stored. If source events were "
                "AbsRectify-transformed, lost event polarity cannot be recovered."
            ),
        }
        atomic_write_text(metadata_path, json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    except Exception:
        output_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise

    return f"{output_path.name} {tuple(stored.shape)}"


def run(args: argparse.Namespace) -> int:
    dataset_root = normalize_path(args.dataset_root)
    segmentation_root = dataset_root / "segmentation"
    padded_root = dataset_root / "segmentation_padded"
    if not segmentation_root.is_dir():
        raise ReconstructionError(f"Missing variable-length segmentation root: {segmentation_root}")
    if not padded_root.is_dir():
        raise ReconstructionError(
            f"Missing padded root: {padded_root}. Run the repository padding pipeline first."
        )

    packages = discover_source_packages(segmentation_root)
    print(f"Dataset root:      {dataset_root}")
    print(f"Segmentation root: {segmentation_root}")
    print(f"Padded root:       {padded_root}")
    print(f"Packages found:    {len(packages)}")
    print("Users:             automatic scan of every existing user_*")

    written = 0
    failed = 0
    for user, action, source_dir, stem in packages:
        print(f"\n[{user} / action {action}]")
        try:
            result = process_package(
                dataset_root=dataset_root,
                padded_root=padded_root,
                user=user,
                action=action,
                source_dir=source_dir,
                stem=stem,
                sampling_rate_override=args.sampling_rate_hz,
                output_dtype=args.output_dtype,
                overwrite=bool(args.overwrite),
            )
            written += 1
            print(f"  WROTE {result}")
        except Exception as error:
            failed += 1
            print(f"  ERROR: {error}")

    print("\nSummary")
    print(f"  written: {written}")
    print(f"  failed:  {failed}")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, ReconstructionError) as error:
        print(f"ERROR: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
