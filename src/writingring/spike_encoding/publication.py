"""Validated, atomic publication of independent spike-encoding artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Mapping

import numpy as np

from writingring.imu_preprocessing import PREPROCESSED_IMU_COLUMNS
from writingring.spike_encoding.contracts import (
    SpikeEncoder,
    SpikeEncodingError,
    SpikeEncodingOutput,
)
from writingring.spike_encoding.io import (
    SpikeEncodingInput,
    SpikeEncodingSourceSummary,
    validate_sequence_offsets,
)


_SEQUENCE_FIELDS = (
    "sequence_index",
    "start_offset",
    "stop_offset_exclusive",
    "sample_count",
    "nonzero_event_count",
    "positive_event_count",
    "negative_event_count",
    "event_density",
)


class SpikeEncodingPublishError(SpikeEncodingError):
    """Raised when a completed encoding cannot be safely published."""


@dataclass(frozen=True, slots=True)
class SpikeEncodingOutputPaths:
    """The four owned artifacts of one encoder run."""

    output_directory: Path
    spike_events_path: Path
    sequence_offsets_path: Path
    sequences_csv_path: Path
    summary_json_path: Path


def spike_encoding_output_paths(
    *,
    raw_imu_path: Path,
    encoder_name: str,
    output_stem: str | None,
    output_root: Path | None,
) -> SpikeEncodingOutputPaths:
    """Derive the documented input-root-local namespace and owned filenames."""

    source = Path(raw_imu_path)
    source_root = source.parent.resolve()
    if output_root is not None and Path(output_root).resolve() != source_root:
        raise SpikeEncodingPublishError(
            "--output-root must match the input IMU parent directory; "
            "spike-encoding outputs are stored beside the input root"
        )
    name = _safe_component(encoder_name, name="encoder name")
    stem = _safe_component(output_stem or _derived_stem(source), name="output stem")
    directory = source_root / name / stem
    return SpikeEncodingOutputPaths(
        output_directory=directory,
        spike_events_path=directory / f"{stem}_spikeEvents.npy",
        sequence_offsets_path=directory / f"{stem}_spike_sequence_offsets.npy",
        sequences_csv_path=directory / f"{stem}_spike_sequences.csv",
        summary_json_path=directory / f"{stem}_spike_encoding_summary.json",
    )


def publish_spike_encoding(
    *,
    output: SpikeEncodingOutput,
    input_data: SpikeEncodingInput,
    encoder: SpikeEncoder,
    effective_settings: Mapping[str, object],
    source_summary: SpikeEncodingSourceSummary | None,
    sequence_mode: str,
    offsets_source: str,
    output_dtype: str,
    paths: SpikeEncodingOutputPaths,
    overwrite: bool,
) -> dict[str, object]:
    """Stage, verify, and atomically publish one complete encoding result."""

    values = _validated_values(
        output.values,
        sample_count=len(input_data.acceleration_g),
        channel_count=len(output.channel_names),
        output_dtype=output_dtype,
    )
    offsets = validate_sequence_offsets(output.sequence_offsets, sample_count=len(values))
    statistics = _validated_statistics(output.sequence_statistics, sequence_count=len(offsets) - 1)
    if sequence_mode not in {"offsets", "single-array"}:
        raise SpikeEncodingPublishError("sequence_mode must be 'offsets' or 'single-array'")
    if not isinstance(offsets_source, str) or not offsets_source:
        raise SpikeEncodingPublishError("offsets_source must be a nonempty string")
    summary = _build_summary(
        output=output,
        input_data=input_data,
        encoder=encoder,
        effective_settings=effective_settings,
        source_summary=source_summary,
        sequence_mode=sequence_mode,
        offsets_source=offsets_source,
        values=values,
    )
    _publish_staged(paths, values=values, offsets=offsets, statistics=statistics, summary=summary, overwrite=overwrite)
    return summary


def _validated_values(
    values: np.ndarray,
    *,
    sample_count: int,
    channel_count: int,
    output_dtype: str,
) -> np.ndarray:
    try:
        dtype = np.dtype(output_dtype)
    except TypeError as error:
        raise SpikeEncodingPublishError("output dtype must be float32 or float64") from error
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise SpikeEncodingPublishError("output dtype must be float32 or float64")
    try:
        stored = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise SpikeEncodingPublishError("encoder output cannot be stored as the requested dtype") from error
    if stored.ndim != 2 or stored.shape != (sample_count, channel_count):
        raise SpikeEncodingPublishError(
            f"encoder output must have shape ({sample_count}, {channel_count}) before publication"
        )
    if not np.isfinite(stored).all():
        raise SpikeEncodingPublishError("encoder output is not finite in the requested dtype")
    return stored.copy()


def _validated_statistics(
    rows: tuple[dict[str, object], ...],
    *,
    sequence_count: int,
) -> tuple[dict[str, object], ...]:
    if len(rows) != sequence_count:
        raise SpikeEncodingPublishError("sequence statistics count does not match sequence offsets")
    validated: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(_SEQUENCE_FIELDS):
            raise SpikeEncodingPublishError("sequence statistics do not match the CSV schema")
        if row["sequence_index"] != index:
            raise SpikeEncodingPublishError("sequence statistics have non-contiguous sequence indices")
        validated.append(dict(row))
    return tuple(validated)


def _build_summary(
    *,
    output: SpikeEncodingOutput,
    input_data: SpikeEncodingInput,
    encoder: SpikeEncoder,
    effective_settings: Mapping[str, object],
    source_summary: SpikeEncodingSourceSummary | None,
    sequence_mode: str,
    offsets_source: str,
    values: np.ndarray,
) -> dict[str, object]:
    metadata = getattr(encoder, "encoding_metadata", None)
    settings = dict(metadata) if isinstance(metadata, Mapping) else dict(effective_settings)
    settings.setdefault("sampling_rate_hz", effective_settings["sampling_rate_hz"])
    nonzero = int(np.count_nonzero(values))
    output_metadata = _validated_output_metadata(getattr(encoder, "output_metadata", None))
    output_section: dict[str, object] = {
        "sample_count": len(values),
        "channel_count": values.shape[1],
        "channel_names": list(output.channel_names),
        "dtype": values.dtype.name,
        "binary": output.representation == "binary_spike_train",
        "polarity_preserved": "signed" in output.representation,
        "amplitude_preserved": output.representation != "binary_spike_train",
    }
    if "channel_order" in output_metadata:
        output_section["channel_order"] = output_metadata["channel_order"]
    summary = {
        "schema_version": 1,
        "encoder": {"name": output.encoder_name, "representation": output.representation},
        "source": {
            "raw_imu_path": str(input_data.raw_imu_path.resolve()),
            "summary_path": None if source_summary is None else str(source_summary.path.resolve()),
            "gravity_removal_method": None if source_summary is None else source_summary.gravity_removal_method,
            "acceleration_semantics": None if source_summary is None else source_summary.acceleration_semantics,
        },
        "input": {
            "sample_count": len(input_data.acceleration_g),
            "channel_indices": [0, 1, 2],
            "channel_names": list(PREPROCESSED_IMU_COLUMNS[:3]),
            "unit": "g",
            "sampling_rate_hz": effective_settings["sampling_rate_hz"],
        },
        "settings": settings,
        "sequence_processing": {
            "mode": sequence_mode,
            "sequence_count": len(output.sequence_statistics),
            "state_reset_boundary": output.summary["state_reset_boundary"],
            "offsets_source": offsets_source,
        },
        "output": output_section,
        "statistics": {
            "nonzero_event_count": nonzero,
            "positive_event_count": int(np.count_nonzero(values > 0.0)),
            "negative_event_count": int(np.count_nonzero(values < 0.0)),
            "event_density": nonzero / values.size,
        },
    }
    try:
        json.dumps(summary, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SpikeEncodingPublishError("spike-encoding summary is not JSON serializable") from error
    return summary


def _validated_output_metadata(metadata: object) -> dict[str, object]:
    """Accept only explicitly declared, well-formed generic output metadata."""

    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise SpikeEncodingPublishError("encoder output_metadata must be a mapping")
    channel_order = metadata.get("channel_order")
    if channel_order is None:
        return {}
    if not isinstance(channel_order, str) or not channel_order:
        raise SpikeEncodingPublishError("encoder output_metadata channel_order must be a nonempty string")
    return {"channel_order": channel_order}


def _publish_staged(
    paths: SpikeEncodingOutputPaths,
    *,
    values: np.ndarray,
    offsets: np.ndarray,
    statistics: tuple[dict[str, object], ...],
    summary: dict[str, object],
    overwrite: bool,
) -> None:
    destination = paths.output_directory
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise SpikeEncodingPublishError(
            f"spike-encoding output already exists; use --overwrite: {destination}"
        )
    if destination.exists():
        _validate_owned_destination(paths)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        staged = _staging_paths(paths, staging)
        _save_npy(staged.spike_events_path, values)
        _save_npy(staged.sequence_offsets_path, offsets)
        staged.sequences_csv_path.write_text(_csv_text(statistics), encoding="utf-8")
        staged.summary_json_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _verify_staged(staged, values=values, offsets=offsets, sequence_count=len(statistics))
        _publish_directory(staging, destination=destination, overwrite=overwrite)
    except OSError as error:
        raise SpikeEncodingPublishError(f"could not publish spike encoding: {error}") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _validate_owned_destination(paths: SpikeEncodingOutputPaths) -> None:
    owned = {
        paths.spike_events_path.name,
        paths.sequence_offsets_path.name,
        paths.sequences_csv_path.name,
        paths.summary_json_path.name,
    }
    unexpected = sorted(path.name for path in paths.output_directory.iterdir() if path.name not in owned)
    if unexpected:
        raise SpikeEncodingPublishError(
            "refusing to overwrite non-spike-encoding files in output directory: "
            + ", ".join(unexpected)
        )


def _staging_paths(paths: SpikeEncodingOutputPaths, staging: Path) -> SpikeEncodingOutputPaths:
    return SpikeEncodingOutputPaths(
        output_directory=staging,
        spike_events_path=staging / paths.spike_events_path.name,
        sequence_offsets_path=staging / paths.sequence_offsets_path.name,
        sequences_csv_path=staging / paths.sequences_csv_path.name,
        summary_json_path=staging / paths.summary_json_path.name,
    )


def _save_npy(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)


def _csv_text(rows: tuple[dict[str, object], ...]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_SEQUENCE_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _verify_staged(
    paths: SpikeEncodingOutputPaths,
    *,
    values: np.ndarray,
    offsets: np.ndarray,
    sequence_count: int,
) -> None:
    stored_values = np.load(paths.spike_events_path, allow_pickle=False)
    stored_offsets = np.load(paths.sequence_offsets_path, allow_pickle=False)
    if stored_values.shape != values.shape or stored_values.dtype != values.dtype or not np.isfinite(stored_values).all():
        raise SpikeEncodingPublishError("staged spikeEvents NPY failed verification")
    if not np.array_equal(stored_offsets, offsets):
        raise SpikeEncodingPublishError("staged sequence offsets NPY failed verification")
    with paths.sequences_csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) != sequence_count or tuple(rows[0]) != _SEQUENCE_FIELDS:
        raise SpikeEncodingPublishError("staged sequence CSV failed verification")
    try:
        loaded_summary = json.loads(paths.summary_json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SpikeEncodingPublishError("staged summary JSON failed verification") from error
    if not isinstance(loaded_summary, dict):
        raise SpikeEncodingPublishError("staged summary JSON failed verification")


def _publish_directory(staging: Path, *, destination: Path, overwrite: bool) -> None:
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        raise SpikeEncodingPublishError(f"cannot publish while backup exists: {backup}")
    try:
        if destination.exists():
            if not overwrite:
                raise SpikeEncodingPublishError(
                    f"spike-encoding output already exists; use --overwrite: {destination}"
                )
            os.replace(destination, backup)
        os.replace(staging, destination)
    except OSError:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _derived_stem(raw_imu_path: Path) -> str:
    name = raw_imu_path.name
    return name.removesuffix("_rawIMU.npy").removesuffix(".npy")


def _safe_component(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise SpikeEncodingPublishError(f"{name} must be a nonempty path component")
    if Path(value).name != value:
        raise SpikeEncodingPublishError(f"{name} must be a single path component")
    return value
