from __future__ import annotations

"""Unified raw/reconstruction datasets for acceleration CNN experiments.

This module preserves the current trainer/embedding batch contract:

    {
        "x": Tensor[3, T],
        "label": scalar long Tensor,
        "valid_mask": Tensor[T] bool,
        "valid_length": scalar long Tensor,
        "sample_id": str,
        ...
    }

Raw acceleration comes from ``paddedSpikeIMU[..., 15:18]``. Reconstruction
comes from the aligned standalone ``*_padded_reconstructed_accel_m_s2.npy``.
All normalization statistics are computed from valid time points only, and the
invalid right-padding region is reset to exact zero after normalization.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from snn.action0_dataset import (
        Action0DatasetError,
        PADDING_DATASET_SUMMARY_FILENAME,
        SPIKE_IMU_CHANNEL_COUNT,
        load_padding_dataset_metadata as _repository_load_padding_dataset_metadata,
    )
except (ImportError, ModuleNotFoundError):
    class Action0DatasetError(ValueError):
        pass

    PADDING_DATASET_SUMMARY_FILENAME = "padding_dataset_summary.json"
    SPIKE_IMU_CHANNEL_COUNT = 21
    _repository_load_padding_dataset_metadata = None

try:
    from writingring.segment_padding import SPIKE_IMU_FEATURE_SCHEMA
except (ImportError, ModuleNotFoundError):
    SPIKE_IMU_FEATURE_SCHEMA = "signed_wavelet_events_plus_imu_v1"


ACCELERATION_SLICE = slice(15, 18)
ACCELERATION_CHANNEL_NAMES = (
    "acceleration_x_m_s2",
    "acceleration_y_m_s2",
    "acceleration_z_m_s2",
)
RECONSTRUCTION_SUFFIX = "_padded_reconstructed_accel_m_s2.npy"
RECONSTRUCTION_METADATA_SUFFIX = "_padded_reconstructed_accel_metadata.json"

DatasetSource = Literal["raw", "reconstruction", "mixed"]
NormalizationSource = Literal["raw", "reconstruction", "mixed"]

__all__ = [
    "Action0DatasetError",
    "ACCELERATION_SLICE",
    "ACCELERATION_CHANNEL_NAMES",
    "RECONSTRUCTION_SUFFIX",
    "RECONSTRUCTION_METADATA_SUFFIX",
    "NormalizationStats",
    "AccelerationProducerMetadata",
    "AccelerationPackage",
    "LoadedAccelerationData",
    "CohortIdentity",
    "SplitAssignment",
    "AccelerationSegmentDataset",
    "MixedAccelerationDataset",
    "natural_key",
    "normalize_label_list",
    "normalize_user_list",
    "normalize_user_name",
    "restrict_manifest_to_cohort",
    "resolve_padded_root",
    "normalize_dataset_roots",
    "load_acceleration_data",
    "build_cohort_identity",
    "parse_checkpoint_cohort_identity",
    "validate_checkpoint_cohort_identity",
    "automatic_user_split",
    "prepare_user_disjoint_splits",
    "fit_acceleration_normalization",
    "build_dataset",
    "make_loader",
    "build_split_loaders",
]


def natural_key(text: str) -> tuple[object, ...]:
    parts: list[object] = []
    token = ""
    numeric = False
    for char in str(text):
        current_numeric = char.isdigit()
        if token and current_numeric != numeric:
            parts.append(int(token) if numeric else token.lower())
            token = ""
        token += char
        numeric = current_numeric
    if token:
        parts.append(int(token) if numeric else token.lower())
    return tuple(parts)


def load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Action0DatasetError(f"Could not read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise Action0DatasetError(f"Expected JSON object: {path}")
    return value


def parse_bool_value(value: object, *, context: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise Action0DatasetError(f"{context}: expected boolean, got {value!r}")


def parse_manifest_int(value: object, *, field: str, path: Path) -> int:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        raise Action0DatasetError(f"{path}: required integer field {field!r} is empty")
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError as error:
        raise Action0DatasetError(
            f"{path}: field {field!r} must be integer-valued, got {value!r}"
        ) from error
    if not math.isfinite(number) or not number.is_integer():
        raise Action0DatasetError(
            f"{path}: field {field!r} must be integer-valued, got {value!r}"
        )
    return int(number)


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray
    valid_time_points: int
    fitted_on: str

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError("Normalization mean/std must each have shape (3,)")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise FloatingPointError("Normalization mean/std contain non-finite values")
        if np.any(std <= 0):
            raise ValueError("Normalization std must be strictly positive")
        if self.valid_time_points < 0:
            raise ValueError("valid_time_points must be non-negative")
        if not str(self.fitted_on).strip():
            raise ValueError("fitted_on must be non-empty")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "valid_time_points": int(self.valid_time_points),
            "fitted_on": self.fitted_on,
        }


@dataclass(frozen=True)
class AccelerationProducerMetadata:
    input_kind: str | None
    feature_schema: str | None
    channel_count: int
    target_length: int
    sampling_rate_hz: float
    padding_side: str | None
    spike_encoder_spec: dict[str, object] | None
    spike_encoder_spec_sha256: str | None
    raw: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_kind": self.input_kind,
            "feature_schema": self.feature_schema,
            "channel_count": self.channel_count,
            "target_length": self.target_length,
            "sampling_rate_hz": self.sampling_rate_hz,
            "padding_side": self.padding_side,
            "spike_encoder": self.spike_encoder_spec,
            "spike_encoder_spec_sha256": self.spike_encoder_spec_sha256,
            "raw": self.raw,
        }


@dataclass(slots=True)
class AccelerationPackage:
    user: str
    action: str
    stem: str
    directory: Path
    source_padded_root: Path
    padded_spike_imu_path: Path
    labels_path: Path
    valid_lengths_path: Path
    valid_mask_path: Path
    padding_manifest_path: Path
    padding_summary_path: Path
    reconstructed_acceleration_path: Path | None
    reconstruction_metadata_path: Path | None
    padded_spike_imu: np.ndarray
    reconstructed_acceleration: np.ndarray | None
    labels: np.ndarray
    valid_lengths: np.ndarray
    valid_mask: np.ndarray

    @property
    def segment_count(self) -> int:
        return int(self.padded_spike_imu.shape[0])

    @property
    def padded_length(self) -> int:
        return int(self.padded_spike_imu.shape[1])

    @property
    def has_reconstruction(self) -> bool:
        return self.reconstructed_acceleration is not None


@dataclass(frozen=True)
class LoadedAccelerationData:
    """Validated logical dataset assembled from one or more padded roots.

    ``padded_root`` and ``producer_metadata`` retain the single-root surface
    used by existing callers.  The plural fields are authoritative for new
    multi-root provenance and compatibility checks.
    """

    padded_root: Path
    producer_metadata: AccelerationProducerMetadata
    padded_roots: tuple[Path, ...]
    producer_metadatas: tuple[AccelerationProducerMetadata, ...]
    root_arguments: tuple[str, ...]
    packages: tuple[AccelerationPackage, ...]
    sample_manifest: pd.DataFrame

    @property
    def selected_actions(self) -> tuple[str, ...]:
        return tuple(sorted({package.action for package in self.packages}, key=natural_key))


@dataclass(frozen=True)
class CohortIdentity:
    """Path-independent identity of the logical cohort authorized by A."""

    selected_actions: tuple[str, ...]
    selected_sample_count: int
    canonical_sample_id_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_actions": list(self.selected_actions),
            "selected_sample_count": self.selected_sample_count,
            "canonical_sample_id_digest": self.canonical_sample_id_digest,
        }


@dataclass(frozen=True)
class SplitAssignment:
    sample_manifest: pd.DataFrame
    train_users: tuple[str, ...]
    val_users: tuple[str, ...]
    test_users: tuple[str, ...]
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    split_summary: pd.DataFrame
    label_split_counts: pd.DataFrame


def resolve_padded_root(
    root: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    root = Path(root).expanduser()
    if not root.is_absolute() and repository_root is not None:
        root = Path(repository_root).expanduser() / root
    root = root.resolve()

    direct_summary = root / PADDING_DATASET_SUMMARY_FILENAME
    if direct_summary.is_file():
        return root

    child = root / "segmentation_padded"
    if (child / PADDING_DATASET_SUMMARY_FILENAME).is_file():
        return child

    candidates = sorted(root.rglob(PADDING_DATASET_SUMMARY_FILENAME)) if root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0].parent
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {PADDING_DATASET_SUMMARY_FILENAME!r} below {root}"
        )
    candidate_text = "\n".join(f"  - {path.parent}" for path in candidates)
    raise Action0DatasetError(
        "ROOT resolves to multiple padded datasets; select one exact root:\n"
        + candidate_text
    )


def _root_arguments(root: str | Path | Sequence[str | Path]) -> tuple[str | Path, ...]:
    if isinstance(root, (str, Path)):
        return (root,)
    if not isinstance(root, Sequence):
        raise TypeError("dataset roots must be a path or a non-string sequence of paths")
    roots = tuple(root)
    if not roots:
        raise ValueError("At least one dataset root must be selected")
    if any(not isinstance(value, (str, Path)) for value in roots):
        raise TypeError("dataset root selections must contain only str or pathlib.Path values")
    return roots


def normalize_dataset_roots(
    root: str | Path | Sequence[str | Path],
    *,
    repository_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Resolve one or more roots and reject duplicate physical datasets."""

    requested = _root_arguments(root)
    resolved = tuple(
        resolve_padded_root(value, repository_root=repository_root)
        for value in requested
    )
    seen: set[Path] = set()
    duplicates: list[Path] = []
    for padded_root in resolved:
        if padded_root in seen:
            duplicates.append(padded_root)
        seen.add(padded_root)
    if duplicates:
        duplicate_text = ", ".join(str(path) for path in duplicates)
        raise Action0DatasetError(
            "Duplicate dataset roots resolve to the same padded dataset: "
            f"{duplicate_text}"
        )
    return resolved


def _load_producer_metadata(padded_root: Path) -> AccelerationProducerMetadata:
    summary_path = padded_root / PADDING_DATASET_SUMMARY_FILENAME
    raw = load_json_object(summary_path)

    if _repository_load_padding_dataset_metadata is not None:
        metadata = _repository_load_padding_dataset_metadata(padded_root)
        input_kind = getattr(metadata, "input_kind", raw.get("input_kind"))
        feature_schema = getattr(metadata, "feature_schema", raw.get("feature_schema"))
        channel_count = int(getattr(metadata, "channel_count", raw.get("channel_count")))
        target_length = int(getattr(metadata, "target_length", raw.get("target_length")))
        sampling_rate_hz = float(
            getattr(metadata, "sampling_rate_hz", raw.get("sampling_rate_hz"))
        )
        padding_side = getattr(metadata, "padding_side", raw.get("padding_side"))
    else:
        input_kind = raw.get("input_kind")
        feature_schema = raw.get("feature_schema")
        try:
            channel_count = int(raw["channel_count"])
            target_length = int(raw["target_length"])
            sampling_rate_hz = float(raw["sampling_rate_hz"])
        except (KeyError, TypeError, ValueError) as error:
            raise Action0DatasetError(
                f"{summary_path}: missing/invalid channel_count, target_length, or sampling_rate_hz"
            ) from error
        padding_side = raw.get("padding_side")

    return AccelerationProducerMetadata(
        input_kind=None if input_kind is None else str(input_kind),
        feature_schema=None if feature_schema is None else str(feature_schema),
        channel_count=channel_count,
        target_length=target_length,
        sampling_rate_hz=sampling_rate_hz,
        padding_side=None if padding_side is None else str(padding_side),
        spike_encoder_spec=(
            raw.get("spike_encoder")
            if isinstance(raw.get("spike_encoder"), dict)
            else None
        ),
        spike_encoder_spec_sha256=(
            str(raw.get("spike_encoder_spec_sha256"))
            if raw.get("spike_encoder_spec_sha256") is not None
            else None
        ),
        raw=raw,
    )


def _validate_producer_metadata(metadata: AccelerationProducerMetadata) -> None:
    if metadata.input_kind not in (None, "spike-imu"):
        raise Action0DatasetError(
            f"Expected input_kind='spike-imu', got {metadata.input_kind!r}"
        )
    if metadata.feature_schema not in (None, SPIKE_IMU_FEATURE_SCHEMA):
        raise Action0DatasetError(
            f"Expected feature_schema={SPIKE_IMU_FEATURE_SCHEMA!r}, "
            f"got {metadata.feature_schema!r}"
        )
    if metadata.channel_count != SPIKE_IMU_CHANNEL_COUNT:
        raise Action0DatasetError(
            f"Expected {SPIKE_IMU_CHANNEL_COUNT} channels, got {metadata.channel_count}"
        )
    if metadata.target_length <= 0:
        raise Action0DatasetError("target_length must be positive")
    if not math.isfinite(metadata.sampling_rate_hz) or metadata.sampling_rate_hz <= 0:
        raise Action0DatasetError("sampling_rate_hz must be positive and finite")
    if metadata.padding_side not in (None, "right"):
        raise Action0DatasetError("This evaluation requires canonical right padding")


def _validate_compatible_producer_metadata(
    padded_roots: Sequence[Path],
    metadatas: Sequence[AccelerationProducerMetadata],
) -> None:
    """Reject selected roots that cannot share one fixed-shape batch contract."""

    if len(padded_roots) != len(metadatas) or not padded_roots:
        raise AssertionError("Each selected padded root must have one metadata record")
    baseline = metadatas[0]
    fields = (
        "input_kind",
        "feature_schema",
        "channel_count",
        "target_length",
        "sampling_rate_hz",
        "padding_side",
    )
    mismatches: list[str] = []
    for padded_root, metadata in zip(padded_roots[1:], metadatas[1:], strict=True):
        differences = [
            f"{field}={getattr(metadata, field)!r} (expected {getattr(baseline, field)!r})"
            for field in fields
            if getattr(metadata, field) != getattr(baseline, field)
        ]
        if differences:
            mismatches.append(f"{padded_root}: " + "; ".join(differences))
    if mismatches:
        raise Action0DatasetError(
            "Incompatible producer metadata across selected padded roots. "
            f"Baseline {padded_roots[0]}: "
            + " | ".join(mismatches)
        )
    if len(padded_roots) > 1:
        missing = [
            f"{root} (action roots selected together)"
            for root, metadata in zip(padded_roots, metadatas, strict=True)
            if not metadata.spike_encoder_spec_sha256
            or not metadata.spike_encoder_spec
        ]
        if missing:
            raise Action0DatasetError(
                "Missing spike encoder identity for multi-root combination; "
                "regenerate the affected padded roots before combining: "
                + ", ".join(missing)
            )
        if any(
            metadata.spike_encoder_spec_sha256 != baseline.spike_encoder_spec_sha256
            for metadata in metadatas[1:]
        ):
            details = []
            for root, metadata in zip(padded_roots, metadatas, strict=True):
                differences = {
                    key: (baseline.spike_encoder_spec.get(key), metadata.spike_encoder_spec.get(key))
                    for key in sorted(set(baseline.spike_encoder_spec) | set(metadata.spike_encoder_spec))
                    if baseline.spike_encoder_spec.get(key) != metadata.spike_encoder_spec.get(key)
                }
                details.append(
                    f"{root}: hash={metadata.spike_encoder_spec_sha256!r}, "
                    f"spec_differences={differences or 'none'}"
                )
            raise Action0DatasetError(
                "Incompatible spike encoder identity across selected roots; "
                f"expected hash={baseline.spike_encoder_spec_sha256!r}. "
                + " | ".join(details)
            )


def discover_padded_spike_paths(padded_root: Path) -> tuple[Path, ...]:
    paths = sorted(
        padded_root.rglob("*_paddedSpikeIMU.npy"),
        key=lambda path: tuple(
            natural_key(part) for part in path.relative_to(padded_root).parts
        ),
    )
    if not paths:
        raise Action0DatasetError(
            f"No *_paddedSpikeIMU.npy files found below {padded_root}"
        )
    return tuple(paths)


def assert_finite_memmap(
    values: np.ndarray,
    *,
    path: Path,
    chunk_segments: int = 256,
) -> None:
    if not np.issubdtype(values.dtype, np.number):
        raise Action0DatasetError(f"Expected numeric array: {path}; dtype={values.dtype}")
    for start in range(0, len(values), chunk_segments):
        stop = min(start + chunk_segments, len(values))
        if not np.isfinite(values[start:stop]).all():
            raise Action0DatasetError(
                f"Non-finite values found in {path}, segments {start}:{stop}"
            )


def _load_manifest_rows(
    path: Path,
    *,
    segment_count: int,
    padded_length: int,
    labels: np.ndarray,
    valid_lengths: np.ndarray,
) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    required_columns = {
        "segment_index",
        "output_segment_index",
        "exported",
        "original_length",
        "target_length",
    }
    missing = required_columns.difference(manifest.columns)
    if missing:
        raise Action0DatasetError(f"{path}: missing columns {sorted(missing)}")

    exported_mask = manifest["exported"].map(
        lambda value: parse_bool_value(value, context=str(path))
    )
    exported = manifest.loc[exported_mask].copy()
    if len(exported) != segment_count:
        raise Action0DatasetError(
            f"{path}: exported rows={len(exported)} != padded segments={segment_count}"
        )

    for field in (
        "segment_index",
        "output_segment_index",
        "original_length",
        "target_length",
    ):
        exported[f"_{field}"] = [
            parse_manifest_int(value, field=field, path=path)
            for value in exported[field]
        ]

    exported = exported.sort_values(
        "_output_segment_index", kind="stable"
    ).reset_index(drop=True)
    if exported["_output_segment_index"].tolist() != list(range(segment_count)):
        raise Action0DatasetError(
            f"{path}: output_segment_index must be contiguous 0..S-1"
        )
    if not np.array_equal(
        exported["_original_length"].to_numpy(np.int64), valid_lengths
    ):
        raise Action0DatasetError(
            f"{path}: original_length does not match valid_lengths"
        )
    if set(exported["_target_length"].tolist()) != {padded_length}:
        raise Action0DatasetError(f"{path}: inconsistent target_length")
    if "label" in exported.columns:
        manifest_labels = exported["label"].astype(str).to_numpy()
        if not np.array_equal(manifest_labels, labels):
            raise Action0DatasetError(f"{path}: exported labels do not match labels.npy")
    return exported


def _load_and_validate_package(
    padded_path: Path,
    *,
    padded_root: Path,
    root_target_length: int,
    require_reconstruction: bool,
) -> tuple[AccelerationPackage, pd.DataFrame]:
    action_dir = padded_path.parent
    user_dir = action_dir.parent
    if not user_dir.name.startswith("user_") or not action_dir.name.startswith("action_"):
        raise Action0DatasetError(
            f"Expected <root>/user_*/action_* hierarchy, got {padded_path}"
        )

    user = user_dir.name
    action = action_dir.name.removeprefix("action_")
    stem = padded_path.name.removesuffix("_paddedSpikeIMU.npy")
    expected_stem = f"{user}_action_{action}"
    if stem != expected_stem:
        raise Action0DatasetError(
            f"Package stem mismatch: expected {expected_stem!r}, got {stem!r}"
        )

    labels_path = action_dir / f"{stem}_labels.npy"
    valid_lengths_path = action_dir / f"{stem}_valid_lengths.npy"
    valid_mask_path = action_dir / f"{stem}_valid_mask.npy"
    padding_manifest_path = action_dir / f"{stem}_padding_manifest.csv"
    padding_summary_path = action_dir / f"{stem}_padding_summary.json"
    reconstruction_path = action_dir / f"{stem}{RECONSTRUCTION_SUFFIX}"
    reconstruction_metadata_path = action_dir / f"{stem}{RECONSTRUCTION_METADATA_SUFFIX}"

    required = (
        labels_path,
        valid_lengths_path,
        valid_mask_path,
        padding_manifest_path,
        padding_summary_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required padded package files for {user}/action_{action}: "
            + ", ".join(missing)
        )

    if require_reconstruction:
        recon_missing = [
            str(path)
            for path in (reconstruction_path, reconstruction_metadata_path)
            if not path.is_file()
        ]
        if recon_missing:
            raise FileNotFoundError(
                f"Missing reconstruction files for {user}/action_{action}: "
                + ", ".join(recon_missing)
            )

    padded_spike = np.load(padded_path, allow_pickle=False, mmap_mode="r")
    labels = np.load(labels_path, allow_pickle=False).astype(str)
    valid_lengths_raw = np.load(valid_lengths_path, allow_pickle=False)
    if not np.issubdtype(valid_lengths_raw.dtype, np.integer):
        raise Action0DatasetError(
            f"{valid_lengths_path}: valid_lengths must have integer dtype"
        )
    valid_lengths = valid_lengths_raw.astype(np.int64, copy=False)
    valid_mask = np.load(valid_mask_path, allow_pickle=False, mmap_mode="r")

    if padded_spike.ndim != 3 or padded_spike.shape[2] != SPIKE_IMU_CHANNEL_COUNT:
        raise Action0DatasetError(
            f"{padded_path}: expected (S,T,{SPIKE_IMU_CHANNEL_COUNT}), got {padded_spike.shape}"
        )
    segment_count, padded_length, _ = map(int, padded_spike.shape)
    if segment_count <= 0 or padded_length <= 0:
        raise Action0DatasetError(f"{padded_path}: padded array must be non-empty")
    if padded_length != int(root_target_length):
        raise Action0DatasetError(
            f"{padded_path}: T={padded_length} != root target_length={root_target_length}"
        )
    assert_finite_memmap(padded_spike, path=padded_path)

    if labels.shape != (segment_count,):
        raise Action0DatasetError(f"{labels_path}: wrong shape {labels.shape}")
    if valid_lengths.shape != (segment_count,):
        raise Action0DatasetError(f"{valid_lengths_path}: wrong shape {valid_lengths.shape}")
    if valid_mask.shape != (segment_count, padded_length):
        raise Action0DatasetError(f"{valid_mask_path}: wrong shape {valid_mask.shape}")
    if valid_mask.dtype != np.dtype(np.bool_):
        raise Action0DatasetError(f"{valid_mask_path}: valid_mask must have bool dtype")
    if np.any(valid_lengths <= 0) or np.any(valid_lengths > padded_length):
        raise Action0DatasetError(
            f"{valid_lengths_path}: valid lengths must lie in [1,{padded_length}]"
        )

    time_index = np.arange(padded_length, dtype=np.int64)[None, :]
    for start in range(0, segment_count, 1024):
        stop = min(start + 1024, segment_count)
        expected_mask = time_index < valid_lengths[start:stop, None]
        if not np.array_equal(valid_mask[start:stop], expected_mask):
            raise Action0DatasetError(
                f"{valid_mask_path}: mask is not canonical contiguous right padding"
            )

    padding_summary = load_json_object(padding_summary_path)
    if int(padding_summary.get("target_length")) != padded_length:
        raise Action0DatasetError(f"{padding_summary_path}: target_length mismatch")
    if padding_summary.get("padding_side") != "right":
        raise Action0DatasetError(f"{padding_summary_path}: expected padding_side='right'")
    if padding_summary.get("overflow_policy") != "skip":
        raise Action0DatasetError(f"{padding_summary_path}: expected overflow_policy='skip'")
    if padding_summary.get("input_kind") not in (None, "spike-imu"):
        raise Action0DatasetError(f"{padding_summary_path}: expected spike-imu package")
    if padding_summary.get("feature_schema") not in (None, SPIKE_IMU_FEATURE_SCHEMA):
        raise Action0DatasetError(f"{padding_summary_path}: unexpected feature_schema")

    exported = _load_manifest_rows(
        padding_manifest_path,
        segment_count=segment_count,
        padded_length=padded_length,
        labels=labels,
        valid_lengths=valid_lengths,
    )

    reconstructed: np.ndarray | None = None
    reconstruction_path_out: Path | None = None
    reconstruction_metadata_out: Path | None = None
    if reconstruction_path.is_file() or reconstruction_metadata_path.is_file():
        if not reconstruction_path.is_file() or not reconstruction_metadata_path.is_file():
            raise Action0DatasetError(
                f"Incomplete reconstruction artifact pair in {action_dir}"
            )
        reconstructed = np.load(
            reconstruction_path, allow_pickle=False, mmap_mode="r"
        )
        if reconstructed.shape != (segment_count, padded_length, 3):
            raise Action0DatasetError(
                f"{reconstruction_path}: expected {(segment_count, padded_length, 3)}, "
                f"got {reconstructed.shape}"
            )
        assert_finite_memmap(reconstructed, path=reconstruction_path)
        for start in range(0, segment_count, 256):
            stop = min(start + 256, segment_count)
            mask = np.asarray(valid_mask[start:stop], dtype=np.bool_)
            values = np.asarray(reconstructed[start:stop])
            if np.any(values[~mask] != 0.0):
                raise Action0DatasetError(
                    f"{reconstruction_path}: padded reconstruction region is not exact zero"
                )

        reconstruction_metadata = load_json_object(reconstruction_metadata_path)
        if reconstruction_metadata.get("artifact_type") != (
            "padded_segmentwise_custom_wavelet_acceleration_reconstruction"
        ):
            raise Action0DatasetError(
                f"{reconstruction_metadata_path}: unexpected artifact_type"
            )
        output_meta = reconstruction_metadata.get("output")
        if not isinstance(output_meta, dict):
            raise Action0DatasetError(
                f"{reconstruction_metadata_path}: missing output metadata"
            )
        if output_meta.get("shape") != [segment_count, padded_length, 3]:
            raise Action0DatasetError(
                f"{reconstruction_metadata_path}: reconstruction shape metadata mismatch"
            )
        if output_meta.get("units") != ["m/s^2", "m/s^2", "m/s^2"]:
            raise Action0DatasetError(
                f"{reconstruction_metadata_path}: expected m/s^2 output units"
            )
        reconstruction_path_out = reconstruction_path
        reconstruction_metadata_out = reconstruction_metadata_path

    provenance_rows: list[dict[str, object]] = []
    for output_index, row in exported.iterrows():
        provenance_rows.append(
            {
                "sample_id": (
                    f"{user}/action_{action}/{stem}/segment_{output_index:06d}"
                ),
                "user": user,
                "action": action,
                "stem": stem,
                "segment_index": int(output_index),
                "source_segment_index": int(row["_segment_index"]),
                "dataset_id": row.get("dataset_id", ""),
                "label": str(labels[output_index]),
                "valid_length": int(valid_lengths[output_index]),
                "package_relative_dir": str(action_dir.relative_to(padded_root)),
                "source_padded_root": str(padded_root),
                "has_reconstruction": reconstructed is not None,
            }
        )

    package = AccelerationPackage(
        user=user,
        action=action,
        stem=stem,
        directory=action_dir,
        source_padded_root=padded_root,
        padded_spike_imu_path=padded_path,
        labels_path=labels_path,
        valid_lengths_path=valid_lengths_path,
        valid_mask_path=valid_mask_path,
        padding_manifest_path=padding_manifest_path,
        padding_summary_path=padding_summary_path,
        reconstructed_acceleration_path=reconstruction_path_out,
        reconstruction_metadata_path=reconstruction_metadata_out,
        padded_spike_imu=padded_spike,
        reconstructed_acceleration=reconstructed,
        labels=labels,
        valid_lengths=valid_lengths,
        valid_mask=valid_mask,
    )
    return package, pd.DataFrame(provenance_rows)


def load_acceleration_data(
    root: str | Path | Sequence[str | Path],
    *,
    repository_root: str | Path | None = None,
    require_reconstruction: bool = False,
) -> LoadedAccelerationData:
    root_arguments = _root_arguments(root)
    padded_roots = normalize_dataset_roots(
        root_arguments,
        repository_root=repository_root,
    )
    producer_metadatas = tuple(
        _load_producer_metadata(padded_root) for padded_root in padded_roots
    )
    for producer_metadata in producer_metadatas:
        _validate_producer_metadata(producer_metadata)
    _validate_compatible_producer_metadata(padded_roots, producer_metadatas)

    packages: list[AccelerationPackage] = []
    manifest_parts: list[pd.DataFrame] = []
    seen_identity: set[tuple[str, str]] = set()

    for root_index, (padded_root, producer_metadata) in enumerate(
        zip(padded_roots, producer_metadatas, strict=True)
    ):
        for padded_path in discover_padded_spike_paths(padded_root):
            package, rows = _load_and_validate_package(
                padded_path,
                padded_root=padded_root,
                root_target_length=producer_metadata.target_length,
                require_reconstruction=require_reconstruction,
            )
            identity = (package.user, package.action)
            if identity in seen_identity:
                raise Action0DatasetError(
                    "Duplicate package identity across selected roots: "
                    f"{identity}"
                )
            seen_identity.add(identity)
            rows["source_root_index"] = root_index
            rows["package_index"] = len(packages)
            packages.append(package)
            manifest_parts.append(rows)

    sample_manifest = pd.concat(manifest_parts, ignore_index=True)
    if sample_manifest["sample_id"].duplicated().any():
        raise Action0DatasetError("Duplicate sample IDs found")
    if require_reconstruction and not sample_manifest["has_reconstruction"].all():
        raise Action0DatasetError("At least one package lacks reconstruction")

    return LoadedAccelerationData(
        padded_root=padded_roots[0],
        producer_metadata=producer_metadatas[0],
        padded_roots=padded_roots,
        producer_metadatas=producer_metadatas,
        root_arguments=tuple(str(value) for value in root_arguments),
        packages=tuple(packages),
        sample_manifest=sample_manifest,
    )


def build_cohort_identity(
    sample_manifest: pd.DataFrame,
    *,
    selected_actions: Sequence[object],
) -> CohortIdentity:
    """Return a deterministic, path-independent A cohort identity.

    The digest contains only canonical logical sample IDs.  Root paths remain
    provenance, never cohort identity, so relocating an unchanged dataset does
    not invalidate its A checkpoint.
    """

    if "sample_id" not in sample_manifest.columns:
        raise ValueError("sample_manifest is missing the sample_id column")
    sample_ids = [str(value) for value in sample_manifest["sample_id"]]
    if not sample_ids:
        raise ValueError("Cannot build a cohort identity for an empty sample manifest")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Cannot build a cohort identity with duplicate sample IDs")
    actions = tuple(sorted({str(value) for value in selected_actions}, key=natural_key))
    if not actions:
        raise ValueError("Cohort identity requires at least one selected action")
    canonical_ids = sorted(sample_ids)
    digest = hashlib.sha256(
        json.dumps(canonical_ids, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return CohortIdentity(
        selected_actions=actions,
        selected_sample_count=len(canonical_ids),
        canonical_sample_id_digest=digest,
    )


def parse_checkpoint_cohort_identity(
    checkpoint: Mapping[str, object],
) -> CohortIdentity | None:
    """Read modern A cohort metadata, leaving legacy single-root checkpoints usable."""

    value = checkpoint.get("cohort_identity")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Checkpoint cohort_identity must be a mapping")
    required = {
        "selected_actions",
        "selected_sample_count",
        "canonical_sample_id_digest",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(
            "Checkpoint cohort_identity is incomplete; missing "
            f"{sorted(missing)}"
        )
    actions_value = value["selected_actions"]
    if isinstance(actions_value, (str, bytes, bytearray)) or not isinstance(
        actions_value, Sequence
    ):
        raise ValueError("Checkpoint cohort_identity.selected_actions must be a sequence")
    actions = tuple(sorted({str(item) for item in actions_value}, key=natural_key))
    if not actions or len(actions) != len(actions_value):
        raise ValueError(
            "Checkpoint cohort_identity.selected_actions must be non-empty and unique"
        )
    try:
        count = int(value["selected_sample_count"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Checkpoint cohort_identity.selected_sample_count must be an integer"
        ) from error
    digest = str(value["canonical_sample_id_digest"])
    if count <= 0 or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Checkpoint cohort_identity has an invalid sample digest")
    return CohortIdentity(
        selected_actions=actions,
        selected_sample_count=count,
        canonical_sample_id_digest=digest,
    )


def validate_checkpoint_cohort_identity(
    checkpoint: Mapping[str, object],
    sample_manifest: pd.DataFrame,
    *,
    selected_actions: Sequence[object],
    selected_root_count: int,
) -> CohortIdentity | None:
    """Validate a B/C/D candidate against A's exact action/sample cohort.

    Legacy checkpoints have no path-independent sample identity.  They stay
    available only for one-root compatibility; selecting two roots requires a
    newly generated Experiment A checkpoint.
    """

    persisted = parse_checkpoint_cohort_identity(checkpoint)
    if persisted is None:
        if selected_root_count > 1:
            raise ValueError(
                "Legacy Experiment A checkpoint cannot authorize a multi-root run. "
                "Regenerate Experiment A for the selected dataset roots."
            )
        return None

    candidate = build_cohort_identity(
        sample_manifest,
        selected_actions=selected_actions,
    )
    mismatches: list[str] = []
    if candidate.selected_actions != persisted.selected_actions:
        mismatches.append(
            "selected actions "
            f"{list(candidate.selected_actions)} != {list(persisted.selected_actions)}"
        )
    if candidate.selected_sample_count != persisted.selected_sample_count:
        mismatches.append(
            "selected sample count "
            f"{candidate.selected_sample_count} != {persisted.selected_sample_count}"
        )
    if candidate.canonical_sample_id_digest != persisted.canonical_sample_id_digest:
        mismatches.append("canonical sample-ID digest differs")
    if mismatches:
        raise ValueError(
            "Selected dataset does not match the authoritative Experiment A cohort: "
            + "; ".join(mismatches)
            + ". Regenerate Experiment A for this action/sample cohort."
        )
    return persisted


def normalize_user_name(value: object) -> str:
    text = str(value)
    if text.startswith("user_"):
        return text
    if text.isdecimal():
        return f"user_{text}"
    return text


def normalize_label_list(values: Sequence[object]) -> list[str]:
    """Canonicalize an ordered label selection and reject duplicates."""
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("included_labels must be a non-string label sequence")
    if any(value is None for value in values):
        raise ValueError("included_labels must not contain null labels")
    normalized = [str(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(
            "included_labels contains duplicate canonical labels: "
            f"{normalized}"
        )
    return normalized


def normalize_user_list(values: Sequence[object]) -> list[str]:
    normalized = [normalize_user_name(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"User list contains duplicates: {normalized}")
    return normalized


def restrict_manifest_to_cohort(
    sample_manifest: pd.DataFrame,
    *,
    split_users: Sequence[object],
    class_to_idx: Mapping[object, object],
) -> pd.DataFrame:
    """Retain exactly the users and labels authorized by an A checkpoint.

    Dataset/package loading and validation happen before this manifest-level
    operation.  Filtering preserves source row order and all sample identity
    columns, while making missing required users or labels explicit instead of
    silently intersecting the checkpoint cohort with the available data.
    """
    required_columns = {"user", "label"}
    missing = required_columns.difference(sample_manifest.columns)
    if missing:
        raise ValueError(f"sample_manifest is missing columns: {sorted(missing)}")

    normalized_users = normalize_user_list(split_users)
    required_users = set(normalized_users)
    if not required_users:
        raise ValueError("A checkpoint split-user union must not be empty")

    if not isinstance(class_to_idx, Mapping):
        raise TypeError("A checkpoint class_to_idx must be a mapping")
    required_labels = {str(label) for label in class_to_idx}
    if not required_labels:
        raise ValueError("A checkpoint class_to_idx must not be empty")

    manifest = sample_manifest.copy().reset_index(drop=True)
    manifest_users = manifest["user"].astype(str).map(normalize_user_name)
    manifest_labels = manifest["label"].astype(str)
    user_mask = manifest_users.isin(required_users)
    user_manifest_labels = set(manifest_labels.loc[user_mask])
    missing_labels = sorted(required_labels.difference(user_manifest_labels))
    if missing_labels:
        raise ValueError(
            "A checkpoint class_to_idx labels are missing after user cohort "
            f"restriction: {missing_labels}"
        )

    selected_mask = user_mask & manifest_labels.isin(required_labels)
    surviving_users = set(manifest_users.loc[selected_mask])
    missing_users = sorted(required_users.difference(surviving_users), key=natural_key)
    if missing_users:
        raise ValueError(
            "A checkpoint split users have no surviving rows after label "
            f"restriction: {missing_users}"
        )

    return manifest.loc[selected_mask].reset_index(drop=True)


def _validate_user_splits(
    train_users: Sequence[str],
    val_users: Sequence[str],
    test_users: Sequence[str],
) -> None:
    sets = {
        "train": set(train_users),
        "val": set(val_users),
        "test": set(test_users),
    }
    if any(not value for value in sets.values()):
        raise ValueError("train/val/test user splits must all be non-empty")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(f"User leakage between {left} and {right}: {sorted(overlap)}")


def automatic_user_split(
    users: Sequence[str],
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 12345,
) -> tuple[list[str], list[str], list[str]]:
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction and val_fraction must be positive and sum to < 1")

    ordered = sorted(set(users), key=natural_key)
    if len(ordered) < 3:
        raise ValueError("At least three users are required for train/val/test splits")

    rng = np.random.default_rng(seed)
    shuffled = [ordered[index] for index in rng.permutation(len(ordered))]
    n_users = len(shuffled)
    n_train = max(1, min(n_users - 2, int(round(n_users * train_fraction))))
    n_val = max(1, min(n_users - n_train - 1, int(round(n_users * val_fraction))))
    n_test = n_users - n_train - n_val
    if n_test < 1:
        n_val -= 1
        n_test += 1
    if min(n_train, n_val, n_test) < 1:
        raise AssertionError("Internal split calculation produced an empty split")

    train = sorted(shuffled[:n_train], key=natural_key)
    val = sorted(shuffled[n_train : n_train + n_val], key=natural_key)
    test = sorted(shuffled[n_train + n_val :], key=natural_key)
    return train, val, test


def prepare_user_disjoint_splits(
    sample_manifest: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 12345,
    explicit_train_users: Sequence[object] | None = None,
    explicit_val_users: Sequence[object] | None = None,
    explicit_test_users: Sequence[object] | None = None,
    excluded_users: Sequence[object] = (),
    included_labels: Sequence[object] | None = None,
    require_all_users_assigned: bool = True,
    require_all_labels_in_all_splits: bool = True,
    class_to_idx: dict[str, int] | None = None,
) -> SplitAssignment:
    manifest = sample_manifest.copy().reset_index(drop=True)
    required_columns = {"sample_id", "user", "label"}
    missing = required_columns.difference(manifest.columns)
    if missing:
        raise ValueError(f"sample_manifest is missing columns: {sorted(missing)}")

    normalized_excluded_users = normalize_user_list(excluded_users)
    excluded_user_set = set(normalized_excluded_users)
    normalized_included_labels = (
        None if included_labels is None else normalize_label_list(included_labels)
    )
    explicit = (explicit_train_users, explicit_val_users, explicit_test_users)
    normalized_explicit = tuple(
        None if value is None else normalize_user_list(value or []) for value in explicit
    )
    for split_name, users in zip(("train", "val", "test"), normalized_explicit):
        if users is None:
            continue
        overlap = sorted(excluded_user_set.intersection(users), key=natural_key)
        if overlap:
            raise ValueError(
                f"Explicit {split_name} users include excluded users: {overlap}"
            )

    if excluded_user_set:
        normalized_manifest_users = manifest["user"].astype(str).map(normalize_user_name)
        manifest = manifest.loc[
            ~normalized_manifest_users.isin(excluded_user_set)
        ].reset_index(drop=True)

    if normalized_included_labels is not None:
        available_labels = set(manifest["label"].astype(str))
        missing_labels = [
            label
            for label in normalized_included_labels
            if label not in available_labels
        ]
        if missing_labels:
            raise ValueError(
                "Requested labels are absent after user exclusion: "
                f"{missing_labels}"
            )
        if not normalized_included_labels:
            raise ValueError(
                "included_labels must contain at least one label when provided"
            )
        manifest = manifest.loc[
            manifest["label"].astype(str).isin(set(normalized_included_labels))
        ].reset_index(drop=True)

    if all(value is None for value in normalized_explicit):
        train_users, val_users, test_users = automatic_user_split(
            manifest["user"].astype(str).unique().tolist(),
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            seed=seed,
        )
    elif all(value is not None for value in normalized_explicit):
        train_users, val_users, test_users = normalized_explicit
    else:
        raise ValueError(
            "Set all three explicit train/val/test user lists, or leave all three None"
        )

    _validate_user_splits(train_users, val_users, test_users)

    available_users = set(manifest["user"].astype(str).unique())
    assigned_users = set(train_users) | set(val_users) | set(test_users)
    missing_users = sorted(assigned_users - available_users, key=natural_key)
    unused_users = sorted(available_users - assigned_users, key=natural_key)
    if missing_users:
        raise FileNotFoundError(f"Requested users are absent: {missing_users}")
    if require_all_users_assigned and unused_users:
        raise ValueError(f"Users were not assigned to a split: {unused_users}")

    split_by_user = {
        **{user: "train" for user in train_users},
        **{user: "val" for user in val_users},
        **{user: "test" for user in test_users},
    }
    manifest["split"] = manifest["user"].astype(str).map(split_by_user)
    if require_all_users_assigned and manifest["split"].isna().any():
        raise AssertionError("Every sample must inherit a user-level split")
    manifest = manifest.loc[manifest["split"].notna()].reset_index(drop=True)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_ids = set(manifest.loc[manifest["split"] == left, "sample_id"])
        right_ids = set(manifest.loc[manifest["split"] == right, "sample_id"])
        if left_ids & right_ids:
            raise AssertionError(f"Sample leakage between {left} and {right}")

    all_labels = sorted(manifest["label"].astype(str).unique().tolist())
    if class_to_idx is None:
        class_to_idx = {label: index for index, label in enumerate(all_labels)}
    else:
        class_to_idx = {str(label): int(index) for label, index in class_to_idx.items()}
        if set(class_to_idx) != set(all_labels):
            raise ValueError(
                "Provided class_to_idx label set differs from dataset labels"
            )
        if sorted(class_to_idx.values()) != list(range(len(class_to_idx))):
            raise ValueError("class_to_idx values must be contiguous 0..C-1")

    idx_to_class = {index: label for label, index in class_to_idx.items()}
    manifest["label_idx"] = manifest["label"].astype(str).map(class_to_idx).astype(np.int64)

    split_summary = (
        manifest.groupby("split")
        .agg(
            users=("user", "nunique"),
            samples=("sample_id", "size"),
            labels=("label", "nunique"),
        )
        .reindex(["train", "val", "test"])
    )
    label_split_counts = pd.crosstab(
        manifest["label"], manifest["split"]
    ).reindex(index=all_labels, columns=["train", "val", "test"], fill_value=0)

    if require_all_labels_in_all_splits and (label_split_counts == 0).any().any():
        missing_table = label_split_counts[label_split_counts.eq(0).any(axis=1)]
        raise ValueError(
            "At least one label is absent from a split:\n"
            + missing_table.to_string()
        )

    assigned_user_names = set(train_users) | set(val_users) | set(test_users)
    if excluded_user_set.intersection(assigned_user_names):
        raise AssertionError("Excluded users must not be assigned to a split")
    eligible_manifest_users = manifest["user"].astype(str).map(normalize_user_name)
    if eligible_manifest_users.isin(excluded_user_set).any():
        raise AssertionError("Excluded users must not remain in the sample manifest")

    return SplitAssignment(
        sample_manifest=manifest,
        train_users=tuple(train_users),
        val_users=tuple(val_users),
        test_users=tuple(test_users),
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        split_summary=split_summary,
        label_split_counts=label_split_counts,
    )


def _source_values(
    package: AccelerationPackage,
    segment_indices: np.ndarray,
    source: Literal["raw", "reconstruction"],
) -> np.ndarray:
    if source == "raw":
        return np.asarray(
            package.padded_spike_imu[segment_indices, :, ACCELERATION_SLICE],
            dtype=np.float64,
        )
    if source == "reconstruction":
        if package.reconstructed_acceleration is None:
            raise FileNotFoundError(
                f"Package {package.user}/action_{package.action} has no reconstruction"
            )
        return np.asarray(
            package.reconstructed_acceleration[segment_indices],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported source: {source}")


def fit_acceleration_normalization(
    packages: Sequence[AccelerationPackage],
    sample_manifest: pd.DataFrame,
    *,
    split: str = "train",
    source: NormalizationSource = "raw",
    chunk_segments: int = 256,
) -> NormalizationStats:
    """Fit channel mean/std from valid time points only.

    ``source='mixed'`` gives raw and reconstruction equal sample weight by
    accumulating both domains for each selected segment.
    """
    if source not in {"raw", "reconstruction", "mixed"}:
        raise ValueError(f"Unsupported normalization source: {source!r}")
    if chunk_segments <= 0:
        raise ValueError("chunk_segments must be positive")

    selected = sample_manifest.loc[sample_manifest["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"Split {split!r} contains no samples")

    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    valid_time_points = 0
    source_list = ("raw", "reconstruction") if source == "mixed" else (source,)

    for package_index, rows in selected.groupby("package_index", sort=False):
        package = packages[int(package_index)]
        segment_indices = rows["segment_index"].to_numpy(np.int64)
        for start in range(0, len(segment_indices), chunk_segments):
            indices = segment_indices[start : start + chunk_segments]
            mask = np.asarray(package.valid_mask[indices], dtype=np.bool_)
            for actual_source in source_list:
                acceleration = _source_values(package, indices, actual_source)
                valid_values = acceleration[mask]
                if valid_values.ndim != 2 or valid_values.shape[1] != 3:
                    raise AssertionError(
                        "Masked acceleration must have shape (valid_time_points, 3)"
                    )
                total += valid_values.sum(axis=0)
                total_sq += np.square(valid_values).sum(axis=0)
                valid_time_points += int(valid_values.shape[0])

    if valid_time_points <= 0:
        raise ValueError("Selected split contains no valid acceleration samples")

    mean = total / valid_time_points
    variance = total_sq / valid_time_points - np.square(mean)
    variance = np.maximum(variance, 1e-12)
    std = np.sqrt(variance)

    return NormalizationStats(
        mean=mean,
        std=std,
        valid_time_points=valid_time_points,
        fitted_on=f"{source}:{split}",
    )


class AccelerationSegmentDataset(Dataset[dict[str, object]]):
    """Single-domain raw or reconstruction dataset."""

    def __init__(
        self,
        packages: Sequence[AccelerationPackage],
        sample_manifest: pd.DataFrame,
        *,
        split: str,
        source: Literal["raw", "reconstruction"],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        if source not in {"raw", "reconstruction"}:
            raise ValueError("source must be 'raw' or 'reconstruction'")
        rows = sample_manifest.loc[sample_manifest["split"] == split].copy().reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"Split {split!r} contains no samples")

        self.packages = tuple(packages)
        self.split = split
        self.source = source
        self.package_indices = rows["package_index"].to_numpy(np.int64)
        self.segment_indices = rows["segment_index"].to_numpy(np.int64)
        self.labels = rows["label_idx"].to_numpy(np.int64)
        self.valid_lengths = rows["valid_length"].to_numpy(np.int64)
        self.sample_ids = rows["sample_id"].astype(str).tolist()
        self.users = rows["user"].astype(str).tolist()
        self.actions = rows["action"].astype(str).tolist()
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 3)
        self.padded_length = int(self.packages[int(self.package_indices[0])].padded_length)

        if np.any(self.std <= 0) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Normalization mean/std are invalid")
        if source == "reconstruction":
            missing = sorted(
                {
                    self.packages[int(index)].user
                    for index in np.unique(self.package_indices)
                    if self.packages[int(index)].reconstructed_acceleration is None
                }
            )
            if missing:
                raise FileNotFoundError(
                    f"Reconstruction requested but missing for users: {missing}"
                )

    def __len__(self) -> int:
        return len(self.labels)

    def _load_acceleration(
        self,
        package: AccelerationPackage,
        segment_index: int,
    ) -> np.ndarray:
        if self.source == "raw":
            return np.array(
                package.padded_spike_imu[segment_index, :, ACCELERATION_SLICE],
                dtype=np.float32,
                copy=True,
            )
        if package.reconstructed_acceleration is None:
            raise FileNotFoundError(
                f"Missing reconstruction for {package.user}/action_{package.action}"
            )
        return np.array(
            package.reconstructed_acceleration[segment_index],
            dtype=np.float32,
            copy=True,
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        package = self.packages[int(self.package_indices[index])]
        segment_index = int(self.segment_indices[index])
        acceleration = self._load_acceleration(package, segment_index)
        mask = np.array(package.valid_mask[segment_index], dtype=np.bool_, copy=True)
        valid_length = int(self.valid_lengths[index])

        if acceleration.shape != (self.padded_length, 3):
            raise AssertionError(f"Unexpected acceleration shape: {acceleration.shape}")
        if mask.shape != (self.padded_length,):
            raise AssertionError(f"Unexpected valid_mask shape: {mask.shape}")
        if int(mask.sum()) != valid_length:
            raise AssertionError("valid_mask.sum() must equal valid_length")

        acceleration = (acceleration - self.mean) / self.std
        acceleration[~mask] = 0.0
        if not np.isfinite(acceleration).all():
            raise FloatingPointError(
                f"Non-finite normalized sample: {self.sample_ids[index]}"
            )
        if np.any(acceleration[~mask] != 0.0):
            raise AssertionError("Normalized padding must be exact zero")

        return {
            "x": torch.from_numpy(np.ascontiguousarray(acceleration.T)),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "valid_mask": torch.from_numpy(mask),
            "valid_length": torch.tensor(valid_length, dtype=torch.long),
            "sample_id": self.sample_ids[index],
            "user": self.users[index],
            "action": self.actions[index],
            "source": self.source,
        }


class MixedAccelerationDataset(Dataset[dict[str, object]]):
    """D2 dataset: each logical sample appears once as raw and once as recon."""

    def __init__(
        self,
        packages: Sequence[AccelerationPackage],
        sample_manifest: pd.DataFrame,
        *,
        split: str,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.raw_dataset = AccelerationSegmentDataset(
            packages,
            sample_manifest,
            split=split,
            source="raw",
            mean=mean,
            std=std,
        )
        self.recon_dataset = AccelerationSegmentDataset(
            packages,
            sample_manifest,
            split=split,
            source="reconstruction",
            mean=mean,
            std=std,
        )
        if len(self.raw_dataset) != len(self.recon_dataset):
            raise AssertionError("Raw/reconstruction dataset lengths must match")
        if self.raw_dataset.sample_ids != self.recon_dataset.sample_ids:
            raise AssertionError("Raw/reconstruction sample order must match")

        self.split = split
        self.source = "mixed"
        self.padded_length = self.raw_dataset.padded_length
        self.labels = np.repeat(self.raw_dataset.labels, 2)
        self.valid_lengths = np.repeat(self.raw_dataset.valid_lengths, 2)
        self.users = [user for user in self.raw_dataset.users for _ in range(2)]
        self.actions = [action for action in self.raw_dataset.actions for _ in range(2)]
        self.sample_ids = [
            f"{sample_id}::{source}"
            for sample_id in self.raw_dataset.sample_ids
            for source in ("raw", "reconstruction")
        ]

    def __len__(self) -> int:
        return 2 * len(self.raw_dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        base_index = index // 2
        use_reconstruction = bool(index % 2)
        item = dict(
            self.recon_dataset[base_index]
            if use_reconstruction
            else self.raw_dataset[base_index]
        )
        source = "reconstruction" if use_reconstruction else "raw"
        item["sample_id"] = f"{item['sample_id']}::{source}"
        item["source"] = source
        return item


def build_dataset(
    packages: Sequence[AccelerationPackage],
    sample_manifest: pd.DataFrame,
    *,
    split: str,
    source: DatasetSource,
    normalization: NormalizationStats,
) -> Dataset[dict[str, object]]:
    if source == "mixed":
        return MixedAccelerationDataset(
            packages,
            sample_manifest,
            split=split,
            mean=normalization.mean,
            std=normalization.std,
        )
    if source in {"raw", "reconstruction"}:
        return AccelerationSegmentDataset(
            packages,
            sample_manifest,
            split=split,
            source=source,
            mean=normalization.mean,
            std=normalization.std,
        )
    raise ValueError(f"Unsupported dataset source: {source!r}")


def make_loader(
    dataset: Dataset[dict[str, object]],
    *,
    batch_size: int = 128,
    shuffle: bool = False,
    seed: int = 12345,
    num_workers: int = 0,
    use_gpu: bool = True,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
) -> DataLoader[dict[str, object]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    if pin_memory is None:
        pin_memory = bool(use_gpu and torch.cuda.is_available())
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0")

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def build_split_loaders(
    packages: Sequence[AccelerationPackage],
    sample_manifest: pd.DataFrame,
    *,
    train_source: DatasetSource,
    val_source: DatasetSource,
    test_source: DatasetSource,
    normalization: NormalizationStats,
    batch_size: int = 128,
    num_workers: int = 0,
    use_gpu: bool = True,
    seed: int = 12345,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
) -> dict[str, DataLoader[dict[str, object]]]:
    """Build the loaders expected by training.py and embedding.py."""
    train_dataset = build_dataset(
        packages,
        sample_manifest,
        split="train",
        source=train_source,
        normalization=normalization,
    )
    val_dataset = build_dataset(
        packages,
        sample_manifest,
        split="val",
        source=val_source,
        normalization=normalization,
    )
    test_dataset = build_dataset(
        packages,
        sample_manifest,
        split="test",
        source=test_source,
        normalization=normalization,
    )

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        use_gpu=use_gpu,
        seed=seed,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return {
        "train": make_loader(train_dataset, shuffle=True, **kwargs),
        "train_eval": make_loader(train_dataset, shuffle=False, **kwargs),
        "val": make_loader(val_dataset, shuffle=False, **kwargs),
        "test": make_loader(test_dataset, shuffle=False, **kwargs),
    }
