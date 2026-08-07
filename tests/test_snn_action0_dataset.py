from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from snn.action0_dataset import (
    INPUT_CHANNEL_COUNT,
    Action0SegmentDataset,
    discover_class_to_idx,
    resolve_dataset_root,
    resolve_segmentation_root,
)


def _write_padded_package(
    segmentation_root: Path,
    *,
    user: str,
    labels: list[str],
    padded_length: int = 4,
) -> np.ndarray:
    directory = segmentation_root / user / "action_0"
    directory.mkdir(parents=True)
    stem = f"{user}_action_0"
    segment_count = len(labels)
    values = np.zeros((segment_count, padded_length, 21), dtype=np.float32)
    values[:, :, :15] = np.arange(
        segment_count * padded_length * 15, dtype=np.float32
    ).reshape(segment_count, padded_length, 15)
    values[:, :, 15:] = 10_000 + np.arange(
        segment_count * padded_length * 6, dtype=np.float32
    ).reshape(segment_count, padded_length, 6)
    valid_lengths = np.asarray(
        [padded_length - (index % 2) for index in range(segment_count)],
        dtype=np.int32,
    )
    valid_mask = np.arange(padded_length)[None, :] < valid_lengths[:, None]
    np.save(directory / f"{stem}_paddedSpikeIMU.npy", values, allow_pickle=False)
    np.save(directory / f"{stem}_labels.npy", np.asarray(labels), allow_pickle=False)
    np.save(directory / f"{stem}_valid_lengths.npy", valid_lengths, allow_pickle=False)
    np.save(directory / f"{stem}_valid_mask.npy", valid_mask, allow_pickle=False)
    return values


def test_variant_resolution_is_fixed_to_label_padded_output(tmp_path: Path) -> None:
    pipeline_root = tmp_path / "action0_pipeline"
    expected_directories = {
        "lowpass": "low-pass",
        "raw": "raw",
        "madgwick": "madgwick",
        "xylo": "xylo",
    }

    for variant, directory in expected_directories.items():
        dataset_root = resolve_dataset_root(pipeline_root, variant)
        assert dataset_root == pipeline_root / directory / "label"
        assert resolve_segmentation_root(dataset_root) == dataset_root / "segmentation_padded"


def test_dataset_returns_only_the_first_fifteen_spike_channels(tmp_path: Path) -> None:
    segmentation_root = tmp_path / "low-pass" / "label" / "segmentation_padded"
    source = _write_padded_package(segmentation_root, user="user_0", labels=["A", "B"])
    _write_padded_package(segmentation_root, user="user_1", labels=["B", "A"])
    class_to_idx = discover_class_to_idx(segmentation_root)

    dataset = Action0SegmentDataset(segmentation_root, ["user_0"], class_to_idx)
    features, label, valid_mask = dataset[0]

    assert features.dtype == torch.float32
    assert features.shape == (4, INPUT_CHANNEL_COUNT)
    torch.testing.assert_close(features, torch.from_numpy(source[0, :, :15]))
    assert not torch.any(features >= 10_000)
    assert label.dtype == torch.long
    assert label.item() == class_to_idx["A"]
    assert valid_mask.dtype == torch.bool
    assert valid_mask.shape == (4,)
    assert valid_mask.tolist() == [True, True, True, True]
    assert dataset.padded_length == 4
    assert dataset.class_distribution == {"A": 1, "B": 1}
