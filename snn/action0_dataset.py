"""Dataset support for label-segmented, padded Action0 SpikeIMU exports.

The producer writes variable-length segments under ``segmentation`` and writes
their fixed-length training representation under ``segmentation_padded``.  This
module consumes only that final padded representation; it never repads or
re-encodes signals.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from writingring.recording_features import (
    SPIKE_IMU_TRAILING_CHANNEL_COUNT,
)
from writingring.segment_padding import SPIKE_IMU_FEATURE_SCHEMA


DATASET_VARIANT_DIRS: Final[dict[str, str]] = {
    "lowpass": "low-pass",
    "raw": "raw",
    "madgwick": "madgwick",
    "xylo": "xylo",
}
BOUNDARY: Final[str] = "label"
PADDED_SEGMENTATION_DIRECTORY: Final[str] = "segmentation_padded"
PADDING_DATASET_SUMMARY_FILENAME: Final[str] = "padding_dataset_summary.json"
SPIKE_IMU_CHANNEL_COUNT: Final[int] = 21
INPUT_CHANNEL_COUNT: Final[int] = 15

_DIAGNOSTIC_COUNT_FIELDS: Final[tuple[str, ...]] = (
    "processed_user_action_count",
    "source_segment_count",
    "segment_count",
    "skipped_segment_count",
    "board_assisted_package_count",
    "label_only_package_count",
    "failed_package_count",
)


class Action0DatasetError(ValueError):
    """Raised when a padded Action0 producer artifact violates its contract."""


@dataclass(frozen=True, slots=True)
class PaddingDatasetMetadata:
    """Validated root-level metadata published by the padding producer."""

    input_kind: str
    feature_schema: str
    channel_count: int
    target_length: int
    sampling_rate_hz: float
    padding_side: str
    event_representation: str | None = None
    event_feature_schema: str | None = None
    event_channel_count: int | None = None
    spike_encoder_spec_sha256: str | None = None
    processed_user_action_count: int | None = None
    source_segment_count: int | None = None
    segment_count: int | None = None
    skipped_segment_count: int | None = None
    board_assisted_package_count: int | None = None
    label_only_package_count: int | None = None
    failed_package_count: int | None = None

    @property
    def diagnostic_counts(self) -> dict[str, int]:
        """Return optional producer counters without treating them as dataset identity."""

        return {
            name: value
            for name in _DIAGNOSTIC_COUNT_FIELDS
            if (value := getattr(self, name)) is not None
        }

@dataclass(frozen=True, slots=True)
class _PaddedPackage:
    """Validated arrays for one user/action padded export."""

    user: str
    action: str
    stem: str
    padded_spike_imu: np.ndarray
    labels: np.ndarray
    label_indices: np.ndarray
    valid_mask: np.ndarray
    valid_lengths: np.ndarray


def resolve_dataset_root(pipeline_root: Path, dataset_variant: str) -> Path:
    """Return the fixed-label root for one supported Action0 variant."""

    try:
        variant_directory = DATASET_VARIANT_DIRS[dataset_variant]
    except KeyError as error:
        supported = ", ".join(DATASET_VARIANT_DIRS)
        raise ValueError(
            f"unsupported dataset variant {dataset_variant!r}; expected one of: {supported}"
        ) from error
    return Path(pipeline_root) / variant_directory / BOUNDARY


def resolve_segmentation_root(dataset_root: Path) -> Path:
    """Return the producer's final fixed-length segment directory.

    ``segment_padding.py`` writes padded SpikeIMU arrays and masks here.  The
    sibling ``segmentation`` directory is the pre-padding continuous matrix and
    is intentionally not a training input.
    """

    return Path(dataset_root) / PADDED_SEGMENTATION_DIRECTORY


def load_padding_dataset_metadata(segmentation_root: Path) -> PaddingDatasetMetadata:
    """Load and validate the producer's root-level padded dataset contract.

    The summary is intentionally the only source for producer-wide target and
    sampling metadata.  Absolute source/output roots are not interpreted by
    the trainer, so a package can be moved after publication.
    """

    summary_path = Path(segmentation_root) / PADDING_DATASET_SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(
            "missing padded dataset summary: "
            f"{summary_path}. Run the Action0 padding producer first."
        )
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Action0DatasetError(
            f"could not read padded dataset summary {summary_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise Action0DatasetError(
            f"padded dataset summary must be a JSON object: {summary_path}"
        )

    input_kind = _require_exact_string(
        payload, "input_kind", "spike-imu", summary_path
    )
    feature_schema = _require_nonempty_string(payload, "feature_schema", summary_path)
    channel_count = _require_exact_int(
        payload,
        "channel_count",
        summary_path,
    )
    if channel_count <= SPIKE_IMU_TRAILING_CHANNEL_COUNT:
        raise Action0DatasetError(
            f"padded dataset summary {summary_path} requires channel_count to exceed "
            f"{SPIKE_IMU_TRAILING_CHANNEL_COUNT}; got {channel_count}"
        )
    event_representation = _optional_nonempty_string(payload, "event_representation")
    event_feature_schema = _optional_nonempty_string(payload, "event_feature_schema")
    event_channel_count = payload.get("event_channel_count")
    if event_channel_count is not None and (
        not isinstance(event_channel_count, int)
        or isinstance(event_channel_count, bool)
        or event_channel_count != channel_count - SPIKE_IMU_TRAILING_CHANNEL_COUNT
    ):
        raise Action0DatasetError(
            f"padded dataset summary {summary_path} requires event_channel_count="
            f"{channel_count - SPIKE_IMU_TRAILING_CHANNEL_COUNT}; got {event_channel_count!r}"
        )
    spike_encoder_spec_sha256 = _optional_sha256(payload, "spike_encoder_spec_sha256")
    target_length = _require_positive_int(payload, "target_length", summary_path)
    sampling_rate_hz = _require_positive_finite_float(
        payload, "sampling_rate_hz", summary_path
    )
    padding_side = _require_exact_string(payload, "padding_side", "right", summary_path)

    diagnostics: dict[str, int | None] = {}
    for name in _DIAGNOSTIC_COUNT_FIELDS:
        if name in payload:
            diagnostics[name] = _require_nonnegative_int(payload, name, summary_path)
        else:
            diagnostics[name] = None
    return PaddingDatasetMetadata(
        input_kind=input_kind,
        feature_schema=feature_schema,
        channel_count=channel_count,
        event_representation=event_representation,
        event_feature_schema=event_feature_schema,
        event_channel_count=event_channel_count,
        spike_encoder_spec_sha256=spike_encoder_spec_sha256,
        target_length=target_length,
        sampling_rate_hz=sampling_rate_hz,
        padding_side=padding_side,
        **diagnostics,
    )


def _require_exact_string(
    payload: Mapping[str, object],
    name: str,
    expected: str,
    summary_path: Path,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or value != expected:
        raise Action0DatasetError(
            f"padded dataset summary {summary_path} requires {name}={expected!r}; "
            f"got {value!r}"
        )
    return value


def _require_nonempty_string(
    payload: Mapping[str, object],
    name: str,
    summary_path: Path,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise Action0DatasetError(
            f"padded dataset summary {summary_path} requires non-empty {name}; got {value!r}"
        )
    return value


def _require_exact_int(
    payload: Mapping[str, object],
    name: str,
    summary_path: Path,
    *,
    expected: int | None = None,
) -> int:
    value = payload.get(name)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or (expected is not None and value != expected)
    ):
        requirement = "an integer" if expected is None else repr(expected)
        raise Action0DatasetError(
            f"padded dataset summary {summary_path} requires {name}={requirement}; "
            f"got {value!r}"
        )
    return value


def _optional_nonempty_string(payload: Mapping[str, object], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise Action0DatasetError(
            f"padded dataset summary requires non-empty {name}; got {value!r}"
        )
    return value


def _optional_sha256(payload: Mapping[str, object], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise Action0DatasetError(
            f"padded dataset summary requires a SHA-256 {name}; got {value!r}"
        )
    return value


def _require_positive_int(
    payload: Mapping[str, object], name: str, summary_path: Path
) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Action0DatasetError(
            f"padded dataset summary {summary_path} requires {name} to be a positive integer; "
            f"got {value!r}"
        )
    return value


def _require_nonnegative_int(
    payload: Mapping[str, object], name: str, summary_path: Path
) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Action0DatasetError(
            f"padded dataset summary {summary_path} requires diagnostic {name} "
            f"to be a non-negative integer; got {value!r}"
        )
    return value


def _require_positive_finite_float(
    payload: Mapping[str, object], name: str, summary_path: Path
) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Action0DatasetError(
            f"padded dataset summary {summary_path} requires {name} to be a "
            f"finite positive number; got {value!r}"
        )
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise Action0DatasetError(
            f"padded dataset summary {summary_path} requires {name} to be a "
            f"finite positive number; got {value!r}"
        ) from error
    if not math.isfinite(converted) or converted <= 0:
        raise Action0DatasetError(
            f"padded dataset summary {summary_path} requires {name} to be a "
            f"finite positive number; got {value!r}"
        )
    return converted


def discover_class_to_idx(segmentation_root: Path) -> dict[str, int]:
    """Build a deterministic mapping from every label in a selected variant."""

    packages = _discover_package_paths(Path(segmentation_root), users=None)
    labels: set[str] = set()
    for _, _, stem, directory in packages:
        label_path = directory / f"{stem}_labels.npy"
        values = _load_npy(label_path, description="labels")
        if values.ndim != 1:
            raise Action0DatasetError(
                f"labels must be one-dimensional: {label_path} has shape {tuple(values.shape)}"
            )
        if len(values) == 0:
            raise Action0DatasetError(f"labels must contain at least one segment: {label_path}")
        labels.update(str(value) for value in values)
    if not labels:
        raise Action0DatasetError(f"no labels found below padded segmentation root: {segmentation_root}")
    return {label: index for index, label in enumerate(sorted(labels))}


class Action0SegmentDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """One fixed-length Action0 segment per item.

    Each item is ``(x, label, valid_mask)`` with shapes ``(T, C_event)``, ``()``,
    and ``(T,)`` respectively.  The source packages are memory mapped but the
    returned tensors are writable copies, which keeps DataLoader workers and
    autograd isolated from the NPY files.
    """

    def __init__(
        self,
        segmentation_root: Path,
        users: Sequence[str],
        class_to_idx: Mapping[str, int],
    ) -> None:
        self.segmentation_root = Path(segmentation_root)
        self.users = _validate_users(users)
        self.class_to_idx = _validate_class_mapping(class_to_idx)
        self.producer_metadata = load_padding_dataset_metadata(self.segmentation_root)
        _validate_unsigned_event_contract(self.producer_metadata)
        self._input_channel_count = int(self.producer_metadata.event_channel_count)

        package_paths = _discover_package_paths(self.segmentation_root, users=self.users)
        packages: list[_PaddedPackage] = []
        index: list[tuple[int, int]] = []
        padded_length: int | None = None
        for user, action, stem, directory in package_paths:
            package = _load_and_validate_package(
                user=user,
                action=action,
                stem=stem,
                directory=directory,
                class_to_idx=self.class_to_idx,
                expected_channel_count=self.producer_metadata.channel_count,
            )
            current_length = int(package.padded_spike_imu.shape[1])
            if padded_length is None:
                padded_length = current_length
            elif current_length != padded_length:
                raise Action0DatasetError(
                    "all padded packages must use one across-dataset time length; "
                    f"expected {padded_length}, got {current_length} in {directory}"
                )
            package_index = len(packages)
            packages.append(package)
            index.extend((package_index, segment_index) for segment_index in range(len(package.labels)))

        if not packages or padded_length is None or not index:
            raise Action0DatasetError(
                f"no padded Action0 segments found for users {list(self.users)!r} in "
                f"{self.segmentation_root}"
            )
        self._packages = tuple(packages)
        self._index = tuple(index)
        self._padded_length = padded_length

    @property
    def padded_length(self) -> int:
        """The common producer-defined padded segment length."""

        return self._padded_length

    @property
    def input_channel_count(self) -> int:
        """Return the full event-channel count supplied to the SNN."""

        return self._input_channel_count

    @property
    def class_distribution(self) -> dict[str, int]:
        """Return this split's original-label frequencies in stable order."""

        counts: Counter[str] = Counter()
        for package in self._packages:
            counts.update(str(value) for value in package.labels)
        return {label: counts[label] for label in sorted(counts)}

    @property
    def valid_fraction(self) -> float:
        """Return the fraction of producer samples marked valid in this split."""

        valid = sum(int(package.valid_mask.sum()) for package in self._packages)
        total = len(self) * self._padded_length
        return valid / total

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        package_index, segment_index = self._index[index]
        package = self._packages[package_index]
        features = np.array(
            package.padded_spike_imu[segment_index, :, : self._input_channel_count],
            dtype=np.float32,
            copy=True,
        )
        if features.shape != (self._padded_length, self._input_channel_count):
            raise AssertionError(
                "validated padded package changed shape after initialization: "
                f"expected {(self._padded_length, self._input_channel_count)}, got {features.shape}"
            )
        if float(np.min(features)) < 0.0:
            raise Action0DatasetError(
                "padded SpikeIMU event channels must be nonnegative for "
                "event_representation='unsigned'"
            )
        valid_mask = np.array(package.valid_mask[segment_index], dtype=np.bool_, copy=True)
        label = int(package.label_indices[segment_index])
        return (
            torch.from_numpy(features),
            torch.tensor(label, dtype=torch.long),
            torch.from_numpy(valid_mask),
        )


def _validate_unsigned_event_contract(metadata: PaddingDatasetMetadata) -> None:
    if metadata.event_representation != "unsigned":
        raise Action0DatasetError(
            "Action0 SNN requires event_representation='unsigned'; "
            f"got {metadata.event_representation!r}"
        )
    if not metadata.event_feature_schema:
        raise Action0DatasetError(
            "Action0 SNN requires a non-empty event_feature_schema"
        )
    if metadata.event_channel_count != (
        metadata.channel_count - SPIKE_IMU_TRAILING_CHANNEL_COUNT
    ):
        raise Action0DatasetError(
            "Action0 SNN requires event_channel_count to equal channel_count minus 6"
        )
    if not metadata.spike_encoder_spec_sha256:
        raise Action0DatasetError(
            "Action0 SNN requires spike_encoder_spec_sha256"
        )


def _validate_users(users: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(user) for user in users)
    if not normalized:
        raise ValueError("at least one user is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"users must not contain duplicates: {list(normalized)!r}")
    return normalized


def _validate_class_mapping(class_to_idx: Mapping[str, int]) -> dict[str, int]:
    mapping = {str(label): int(index) for label, index in class_to_idx.items()}
    if not mapping:
        raise ValueError("class_to_idx must not be empty")
    expected = list(range(len(mapping)))
    if sorted(mapping.values()) != expected:
        raise ValueError(
            "class_to_idx values must be a contiguous zero-based range; "
            f"got {sorted(mapping.values())}"
        )
    return mapping


def _discover_package_paths(
    segmentation_root: Path,
    *,
    users: Sequence[str] | None,
) -> tuple[tuple[str, str, str, Path], ...]:
    if not segmentation_root.is_dir():
        raise FileNotFoundError(
            "padded segmentation root does not exist: "
            f"{segmentation_root}. Run the Action0 padding producer first."
        )

    if users is None:
        user_directories = sorted(
            (path for path in segmentation_root.iterdir() if path.is_dir()),
            key=lambda path: _natural_name_key(path.name),
        )
    else:
        user_directories = []
        for user in users:
            directory = segmentation_root / user
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"requested user {user!r} is missing from padded segmentation root: "
                    f"{directory}"
                )
            user_directories.append(directory)

    found: list[tuple[str, str, str, Path]] = []
    for user_directory in user_directories:
        action_directory = user_directory / "action_0"
        if not action_directory.is_dir():
            if users is not None:
                raise FileNotFoundError(
                    f"requested user {user_directory.name!r} has no Action0 padded package: "
                    f"{action_directory}"
                )
            continue
        matches = sorted(action_directory.glob("*_paddedSpikeIMU.npy"))
        if len(matches) != 1:
            raise Action0DatasetError(
                f"expected exactly one padded SpikeIMU file in {action_directory}, found {len(matches)}"
            )
        padded_path = matches[0]
        stem = padded_path.name.removesuffix("_paddedSpikeIMU.npy")
        expected_stem = f"{user_directory.name}_action_0"
        if stem != expected_stem:
            raise Action0DatasetError(
                f"padded SpikeIMU filename must use producer stem {expected_stem!r}, got {padded_path.name!r}"
            )
        found.append((user_directory.name, "0", stem, action_directory))

    if not found:
        raise Action0DatasetError(f"no Action0 padded packages found below {segmentation_root}")
    return tuple(found)


def _load_and_validate_package(
    *,
    user: str,
    action: str,
    stem: str,
    directory: Path,
    class_to_idx: Mapping[str, int],
    expected_channel_count: int,
) -> _PaddedPackage:
    padded_path = directory / f"{stem}_paddedSpikeIMU.npy"
    labels_path = directory / f"{stem}_labels.npy"
    mask_path = directory / f"{stem}_valid_mask.npy"
    lengths_path = directory / f"{stem}_valid_lengths.npy"

    padded_spike_imu = _load_npy(padded_path, description="padded SpikeIMU", mmap=True)
    labels = _load_npy(labels_path, description="labels")
    valid_mask = _load_npy(mask_path, description="valid mask", mmap=True)
    valid_lengths = _load_npy(lengths_path, description="valid lengths")

    if (
        padded_spike_imu.ndim != 3
        or padded_spike_imu.shape[2] != expected_channel_count
    ):
        raise Action0DatasetError(
            f"padded SpikeIMU must have shape (segment_count, T_pad, {expected_channel_count}): "
            f"{padded_path} has shape {tuple(padded_spike_imu.shape)}"
        )
    if padded_spike_imu.shape[0] == 0 or padded_spike_imu.shape[1] == 0:
        raise Action0DatasetError(f"padded SpikeIMU must contain segments and time steps: {padded_path}")
    if not np.issubdtype(padded_spike_imu.dtype, np.number) or not np.isfinite(padded_spike_imu).all():
        raise Action0DatasetError(f"padded SpikeIMU values must be finite numeric values: {padded_path}")

    segment_count, padded_length, _ = padded_spike_imu.shape
    if labels.ndim != 1 or len(labels) != segment_count:
        raise Action0DatasetError(
            f"labels must have shape ({segment_count},): {labels_path} has shape {tuple(labels.shape)}"
        )
    if valid_mask.ndim != 2 or valid_mask.shape != (segment_count, padded_length):
        raise Action0DatasetError(
            f"valid mask must have shape ({segment_count}, {padded_length}): "
            f"{mask_path} has shape {tuple(valid_mask.shape)}"
        )
    if valid_mask.dtype != np.dtype(np.bool_):
        raise Action0DatasetError(f"valid mask must have bool dtype: {mask_path} has dtype {valid_mask.dtype}")
    if valid_lengths.ndim != 1 or len(valid_lengths) != segment_count:
        raise Action0DatasetError(
            f"valid lengths must have shape ({segment_count},): "
            f"{lengths_path} has shape {tuple(valid_lengths.shape)}"
        )
    if not np.issubdtype(valid_lengths.dtype, np.integer):
        raise Action0DatasetError(
            f"valid lengths must have integer dtype: {lengths_path} has dtype {valid_lengths.dtype}"
        )
    if np.any(valid_lengths <= 0) or np.any(valid_lengths > padded_length):
        raise Action0DatasetError(
            f"valid lengths must lie in [1, {padded_length}]: {lengths_path}"
        )

    expected_mask = np.arange(padded_length)[None, :] < valid_lengths[:, None]
    if not np.array_equal(valid_mask, expected_mask):
        raise Action0DatasetError(
            "valid-mask True positions must be one contiguous right-padded prefix matching "
            f"valid lengths: {mask_path}"
        )

    label_indices = np.empty(segment_count, dtype=np.int64)
    for segment_index, value in enumerate(labels):
        label = str(value)
        try:
            label_indices[segment_index] = class_to_idx[label]
        except KeyError as error:
            raise Action0DatasetError(
                f"label {label!r} in {labels_path} is absent from the global class mapping"
            ) from error
    return _PaddedPackage(
        user=user,
        action=action,
        stem=stem,
        padded_spike_imu=padded_spike_imu,
        labels=labels,
        label_indices=label_indices,
        valid_mask=valid_mask,
        valid_lengths=valid_lengths,
    )


def _load_npy(path: Path, *, description: str, mmap: bool = False) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description} producer artifact: {path}")
    try:
        values = np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)
    except (OSError, ValueError) as error:
        raise Action0DatasetError(f"could not load {description} artifact {path}: {error}") from error
    if not isinstance(values, np.ndarray):
        raise Action0DatasetError(f"expected ndarray in {description} artifact: {path}")
    return values


def _natural_name_key(value: str) -> tuple[int, int | str]:
    prefix, separator, suffix = value.rpartition("_")
    if separator and suffix.isdecimal():
        return (0, int(suffix))
    return (1, value)
