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
    validate_source_metadata_paths,
    validate_sequence_offsets,
)
from writingring.preprocessing_io import (
    PreprocessingIOError,
    sha256_file,
    validate_preprocessing_summary,
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
    """Known artifacts, including both compatible offset-file spellings."""

    output_directory: Path
    spike_events_path: Path
    spike_imu_path: Path
    sequence_offsets_path: Path
    recording_offsets_path: Path
    sequences_csv_path: Path
    summary_json_path: Path


def spike_encoding_output_paths(
    *,
    raw_imu_path: Path | None = None,
    encoder_name: str = "",
    output_stem: str | None = None,
    output_root: Path | None = None,
    source_imu_path: Path | None = None,
    recording_relative_path: Path | str | None = None,
    recording_id: str | None = None,
    canonical: bool = False,
) -> SpikeEncodingOutputPaths:
    """Derive output paths for one encoding.

    The legacy single-file call remains input-root-local and uses the original
    descriptive filenames.  Batch callers provide ``recording_relative_path``
    and request canonical filenames so an output root can mirror an arbitrary
    preprocessed-recording tree.
    """

    source_value = source_imu_path if source_imu_path is not None else raw_imu_path
    if source_value is None:
        raise SpikeEncodingPublishError("source_imu_path is required")
    source = Path(source_value)
    name = _safe_component(encoder_name, name="encoder name")
    stem = _safe_component(output_stem or _derived_stem(source), name="output stem")
    if recording_relative_path is not None:
        relative = _safe_relative_path(recording_relative_path, name="recording relative path")
        base = Path(output_root) if output_root is not None else source.parent
        directory = base / name / relative
        if recording_id is not None:
            _safe_component(recording_id, name="recording id")
        canonical = True
    else:
        base = Path(output_root) if output_root is not None else source.parent
        directory = base / name / stem
        if recording_id is not None:
            _safe_component(recording_id, name="recording id")
    if canonical:
        events_name = "spikes.npy"
        spike_imu_name = "spikeIMU.npy"
        sequence_offsets_name = "sequence_offsets.npy"
        recording_offsets_name = "recording_offsets.npy"
        sequences_name = "sequences.csv"
        summary_name = "metadata.json"
    else:
        events_name = f"{stem}_spikeEvents.npy"
        spike_imu_name = f"{stem}_spikeIMU.npy"
        sequence_offsets_name = f"{stem}_spike_sequence_offsets.npy"
        recording_offsets_name = f"{stem}_spike_recording_offsets.npy"
        sequences_name = f"{stem}_spike_sequences.csv"
        summary_name = f"{stem}_spike_encoding_summary.json"
    return SpikeEncodingOutputPaths(
        output_directory=directory,
        spike_events_path=directory / events_name,
        spike_imu_path=directory / spike_imu_name,
        sequence_offsets_path=directory / sequence_offsets_name,
        recording_offsets_path=directory / recording_offsets_name,
        sequences_csv_path=directory / sequences_name,
        summary_json_path=directory / summary_name,
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
    source_metadata_paths: Mapping[str, Path | None] | None = None,
    allow_gravity_included: bool = False,
) -> dict[str, object]:
    """Stage, verify, and atomically publish one complete encoding result."""

    values = _validated_values(
        output.values,
        sample_count=len(input_data.acceleration_g),
        channel_count=len(output.channel_names),
        output_dtype=output_dtype,
    )
    if source_summary is not None:
        try:
            validate_preprocessing_summary(
                source_summary,
                input_path=input_data.source_imu_path,
                sample_count=len(input_data.preprocessed_imu),
                expected_sampling_rate_hz=float(effective_settings["sampling_rate_hz"]),
                allow_gravity_included=allow_gravity_included,
            )
        except (KeyError, TypeError, ValueError, PreprocessingIOError) as error:
            raise SpikeEncodingPublishError(str(error)) from error
    offsets = validate_sequence_offsets(output.sequence_offsets, sample_count=len(values))
    statistics = _validated_statistics(output.sequence_statistics, sequence_count=len(offsets) - 1)
    offset_semantics = _offset_semantics(output)
    offsets_path = _offsets_path(paths, offset_semantics=offset_semantics)
    spike_imu = (
        _build_spike_imu(values, input_data.preprocessed_imu)
        if _publishes_signed_wavelet_spike_imu(output, input_data.preprocessed_imu, values)
        else None
    )
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
        source_metadata_paths=source_metadata_paths,
        values=values,
        spike_imu=spike_imu,
        offset_semantics=offset_semantics,
        offsets_artifact=offsets_path.name,
    )
    _publish_staged(
        paths,
        values=values,
        spike_imu=spike_imu,
        offsets=offsets,
        offsets_path=offsets_path,
        statistics=statistics,
        summary=summary,
        overwrite=overwrite,
    )
    return summary


def _offset_semantics(output: SpikeEncodingOutput) -> str:
    """Keep generic sequence partitions distinct from Custom Wavelet recordings."""

    return "recording" if output.encoder_name == "custom-wavelet" else "sequence"


def _offsets_path(
    paths: SpikeEncodingOutputPaths,
    *,
    offset_semantics: str,
) -> Path:
    if offset_semantics == "recording":
        return paths.recording_offsets_path
    if offset_semantics == "sequence":
        return paths.sequence_offsets_path
    raise SpikeEncodingPublishError("offset semantics must be 'recording' or 'sequence'")


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


def _publishes_signed_wavelet_spike_imu(
    output: SpikeEncodingOutput,
    raw_imu: np.ndarray,
    values: np.ndarray,
) -> bool:
    """Return whether this output satisfies the only signed-wavelet IMU schema."""

    return (
        output.encoder_name == "custom-wavelet"
        and values.shape[1] == 15
        and raw_imu.ndim == 2
        and raw_imu.shape == (len(values), len(PREPROCESSED_IMU_COLUMNS))
    )


def _build_spike_imu(values: np.ndarray, raw_imu: np.ndarray) -> np.ndarray:
    """Replace the g-domain channels with encoded events without mutating source data."""

    source = np.asarray(raw_imu)
    if source.ndim != 2 or source.shape != (len(values), len(PREPROCESSED_IMU_COLUMNS)):
        raise SpikeEncodingPublishError("input raw IMU shape changed before publication")
    spike_imu = np.column_stack((values, source[:, 3:9]))
    expected_shape = (len(values), values.shape[1] + 6)
    if spike_imu.shape != expected_shape or not np.isfinite(spike_imu).all():
        raise SpikeEncodingPublishError("spike IMU has an invalid shape or non-finite values")
    if not np.array_equal(spike_imu[:, values.shape[1] :], source[:, 3:9]):
        raise SpikeEncodingPublishError("spike IMU did not preserve source m/s² and gyro channels")
    return spike_imu


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
    source_metadata_paths: Mapping[str, Path] | None,
    values: np.ndarray,
    spike_imu: np.ndarray | None,
    offset_semantics: str,
    offsets_artifact: str,
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
    try:
        metadata_references = validate_source_metadata_paths(source_metadata_paths)
    except SpikeEncodingError as error:
        raise SpikeEncodingPublishError(str(error)) from error
    source_hash = (
        source_summary.source_file_sha256
        if source_summary is not None and source_summary.source_file_sha256 is not None
        else sha256_file(input_data.source_imu_path)
    )
    source_recording = (
        None if source_summary is None or source_summary.recording is None
        else dict(source_summary.recording)
    )
    has_metadata_references = any(value is not None for value in metadata_references.values())
    source_section: dict[str, object] = {
        "source_imu_path": str(input_data.source_imu_path.resolve()),
        "preprocessed_imu_path": str(input_data.source_imu_path.resolve()),
        "raw_imu_path": str(input_data.source_imu_path.resolve()),
        "source_file_sha256": source_hash,
        "summary_path": None if source_summary is None else str(source_summary.path.resolve()),
        "source_summary_path": None if source_summary is None else str(source_summary.path.resolve()),
        "gravity_removal_method": None if source_summary is None else source_summary.gravity_removal_method,
        "acceleration_semantics": None if source_summary is None else source_summary.acceleration_semantics,
        "metadata_references": metadata_references,
        "recording": source_recording,
    }
    summary = {
        "schema_version": 3,
        "metadata_schema": "spike_encoding_v3",
        "encoder": {"name": output.encoder_name, "representation": output.representation},
        "source": source_section,
        "source_imu_path": str(input_data.source_imu_path.resolve()),
        "source_file_sha256": source_hash,
        "recording": source_recording,
        "input_channel_names": list(PREPROCESSED_IMU_COLUMNS),
        "encoder_input_channel_names": list(PREPROCESSED_IMU_COLUMNS[:3]),
        "gravity_removal_method": (
            None if source_summary is None else source_summary.gravity_removal_method
        ),
        "sampling_rate_hz": effective_settings["sampling_rate_hz"],
        "input": {
            "sample_count": len(input_data.acceleration_g),
            "channel_count": len(PREPROCESSED_IMU_COLUMNS),
            "channel_indices": [0, 1, 2],
            "channel_names": list(PREPROCESSED_IMU_COLUMNS),
            "encoder_input_channel_names": list(PREPROCESSED_IMU_COLUMNS[:3]),
            "encoder_input_unit": "g",
            "unit": "g",
            "sampling_rate_hz": effective_settings["sampling_rate_hz"],
        },
        "settings": settings,
        "sequence_processing": {
            "mode": sequence_mode,
            "offset_semantics": offset_semantics,
            "offsets_artifact": offsets_artifact,
            "sequence_count": len(output.sequence_statistics),
            "sequence_boundary_semantics": output.summary["sequence_boundary_semantics"],
            "state_reset_boundary": output.summary["state_reset_boundary"],
            "offsets_source": offsets_source,
        },
        "output": output_section,
        "alignment": {
            "row_aligned_with_source_imu": True,
            "row_aligned_with_source_raw_imu": True,
            "sample_count_preserved": True,
            "timestamps_reused_without_shift": metadata_references["timestamp_source_path"] is not None,
            "labels_reused_without_shift": metadata_references["labels_path"] is not None,
            "segment_offsets_reused_without_shift": (
                metadata_references["segment_offsets_path"] is not None
            ),
        },
        "statistics": {
            "nonzero_event_count": nonzero,
            "positive_event_count": int(np.count_nonzero(values > 0.0)),
            "negative_event_count": int(np.count_nonzero(values < 0.0)),
            "event_density": nonzero / values.size,
        },
    }
    if spike_imu is not None:
        trailing_channel_names = [
            "acceleration_x_m_s2",
            "acceleration_y_m_s2",
            "acceleration_z_m_s2",
            "gyro_x_rad_s",
            "gyro_y_rad_s",
            "gyro_z_rad_s",
        ]
        summary["spike_imu"] = {
            "schema": "signed_wavelet_events_plus_imu_v1",
            "sample_count": len(spike_imu),
            "channel_count": spike_imu.shape[1],
            "channel_names": list(output.channel_names) + trailing_channel_names,
            "trailing_channel_names": trailing_channel_names,
            "source_channel_names": list(PREPROCESSED_IMU_COLUMNS[3:]),
            "dtype": spike_imu.dtype.name,
            "source_imu_columns": [3, 4, 5, 6, 7, 8],
            "source_raw_imu_columns": [3, 4, 5, 6, 7, 8],
        }
    if output.encoder_name == "custom-wavelet":
        summary["source"]["metadata_usage"] = {
            "used_by_encoder": False,
            "used_as_reset_boundaries": False,
            "reused_for_downstream_alignment": has_metadata_references,
            "indices_shifted": False,
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
    spike_imu: np.ndarray | None,
    offsets: np.ndarray,
    offsets_path: Path,
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
        if spike_imu is not None:
            _save_npy(staged.spike_imu_path, spike_imu)
        _save_npy(_staged_offset_path(staged, offsets_path), offsets)
        staged.sequences_csv_path.write_text(_csv_text(statistics), encoding="utf-8")
        staged.summary_json_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _verify_staged(
            staged,
            values=values,
            spike_imu=spike_imu,
            offsets=offsets,
            offsets_path=_staged_offset_path(staged, offsets_path),
            sequence_count=len(statistics),
        )
        _publish_directory(staging, destination=destination, overwrite=overwrite)
    except OSError as error:
        raise SpikeEncodingPublishError(f"could not publish spike encoding: {error}") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _validate_owned_destination(paths: SpikeEncodingOutputPaths) -> None:
    owned = {
        paths.spike_events_path.name,
        paths.spike_imu_path.name,
        paths.sequence_offsets_path.name,
        paths.recording_offsets_path.name,
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
        spike_imu_path=staging / paths.spike_imu_path.name,
        sequence_offsets_path=staging / paths.sequence_offsets_path.name,
        recording_offsets_path=staging / paths.recording_offsets_path.name,
        sequences_csv_path=staging / paths.sequences_csv_path.name,
        summary_json_path=staging / paths.summary_json_path.name,
    )


def _staged_offset_path(staged: SpikeEncodingOutputPaths, offsets_path: Path) -> Path:
    """Map an output-directory offset path into the staging directory."""

    if offsets_path.name == staged.recording_offsets_path.name:
        return staged.recording_offsets_path
    if offsets_path.name == staged.sequence_offsets_path.name:
        return staged.sequence_offsets_path
    raise SpikeEncodingPublishError("offset artifact does not belong to output paths")


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
    spike_imu: np.ndarray | None,
    offsets: np.ndarray,
    offsets_path: Path,
    sequence_count: int,
) -> None:
    stored_values = np.load(paths.spike_events_path, allow_pickle=False)
    stored_offsets = np.load(offsets_path, allow_pickle=False)
    if stored_values.shape != values.shape or stored_values.dtype != values.dtype or not np.isfinite(stored_values).all():
        raise SpikeEncodingPublishError("staged spikeEvents NPY failed verification")
    if spike_imu is not None:
        stored_spike_imu = np.load(paths.spike_imu_path, allow_pickle=False)
        if (
            stored_spike_imu.shape != spike_imu.shape
            or stored_spike_imu.dtype != spike_imu.dtype
            or not np.array_equal(stored_spike_imu, spike_imu)
        ):
            raise SpikeEncodingPublishError("staged spikeIMU NPY failed verification")
    if not np.array_equal(stored_offsets, offsets):
        raise SpikeEncodingPublishError("staged offset NPY failed verification")
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


def _derived_stem(source_imu_path: Path) -> str:
    name = source_imu_path.name
    for suffix in ("_preprocessedIMU.npy", "_rawIMU.npy", ".npy"):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return source_imu_path.stem


def _safe_component(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise SpikeEncodingPublishError(f"{name} must be a nonempty path component")
    if Path(value).name != value:
        raise SpikeEncodingPublishError(f"{name} must be a single path component")
    return value


def _safe_relative_path(value: Path | str, *, name: str) -> Path:
    """Validate a batch recording namespace without allowing path escape."""

    relative = Path(value)
    if relative.is_absolute() or not relative.parts:
        raise SpikeEncodingPublishError(f"{name} must be a nonempty relative path")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise SpikeEncodingPublishError(f"{name} must not contain empty or parent components")
    for part in relative.parts:
        _safe_component(part, name=name)
    return relative
