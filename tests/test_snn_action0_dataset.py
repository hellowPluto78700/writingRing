from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from snn.action0_dataset import (
    INPUT_CHANNEL_COUNT,
    SPIKE_IMU_FEATURE_SCHEMA,
    Action0DatasetError,
    Action0SegmentDataset,
    discover_class_to_idx,
    load_padding_dataset_metadata,
    resolve_dataset_root,
    resolve_segmentation_root,
)


def _write_padded_package(
    segmentation_root: Path,
    *,
    user: str,
    labels: list[str],
    padded_length: int = 4,
    channel_count: int = 21,
) -> np.ndarray:
    directory = segmentation_root / user / "action_0"
    directory.mkdir(parents=True)
    stem = f"{user}_action_0"
    segment_count = len(labels)
    event_channel_count = channel_count - 6
    values = np.zeros((segment_count, padded_length, channel_count), dtype=np.float32)
    values[:, :, :event_channel_count] = np.arange(
        segment_count * padded_length * event_channel_count, dtype=np.float32
    ).reshape(segment_count, padded_length, event_channel_count)
    values[:, :, event_channel_count:] = 10_000 + np.arange(
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


def _write_metadata(
    segmentation_root: Path,
    *,
    target_length: object = 4,
    sampling_rate_hz: object = 200.0,
    channel_count: int = 21,
    **overrides: object,
) -> None:
    payload: dict[str, object] = {
        "input_kind": "spike-imu",
        "feature_schema": (
            SPIKE_IMU_FEATURE_SCHEMA
            if channel_count == 21
            else "polarity_split_wavelet_events_plus_imu_v1"
        ),
        "channel_count": channel_count,
        "event_representation": "unsigned",
        "event_feature_schema": (
            "custom_wavelet_polarity_split_abs_events_v1"
            if channel_count == 36
            else "custom_wavelet_abs_rectified_events_v1"
        ),
        "event_channel_count": channel_count - 6,
        "spike_encoder_spec_sha256": "0" * 64,
        "target_length": target_length,
        "sampling_rate_hz": sampling_rate_hz,
        "padding_side": "right",
    }
    payload.update(overrides)
    segmentation_root.mkdir(parents=True, exist_ok=True)
    (segmentation_root / "padding_dataset_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


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
    _write_metadata(segmentation_root)
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


def test_dataset_keeps_all_thirty_polarity_split_event_channels(tmp_path: Path) -> None:
    segmentation_root = tmp_path / "segmentation_padded"
    source = _write_padded_package(
        segmentation_root,
        user="user_0",
        labels=["A", "B"],
        channel_count=36,
    )
    _write_padded_package(
        segmentation_root,
        user="user_1",
        labels=["B", "A"],
        channel_count=36,
    )
    _write_metadata(segmentation_root, channel_count=36)
    class_to_idx = discover_class_to_idx(segmentation_root)

    dataset = Action0SegmentDataset(segmentation_root, ["user_0"], class_to_idx)
    features, _, _ = dataset[0]

    assert dataset.input_channel_count == 30
    assert features.shape == (4, 30)
    torch.testing.assert_close(features, torch.from_numpy(source[0, :, :30]))
    assert not torch.any(features >= 10_000)


def test_dataset_rejects_signed_event_metadata(tmp_path: Path) -> None:
    segmentation_root = tmp_path / "segmentation_padded"
    _write_padded_package(segmentation_root, user="user_0", labels=["A", "B"])
    _write_metadata(segmentation_root, event_representation="signed")

    with pytest.raises(Action0DatasetError, match="event_representation='unsigned'"):
        Action0SegmentDataset(segmentation_root, ["user_0"], {"A": 0, "B": 1})


def test_dataset_rejects_negative_values_declared_unsigned(tmp_path: Path) -> None:
    segmentation_root = tmp_path / "segmentation_padded"
    _write_padded_package(segmentation_root, user="user_0", labels=["A", "B"])
    _write_metadata(segmentation_root)
    padded_path = (
        segmentation_root / "user_0" / "action_0" / "user_0_action_0_paddedSpikeIMU.npy"
    )
    values = np.load(padded_path, allow_pickle=False)
    values[0, 0, 0] = -1.0
    np.save(padded_path, values, allow_pickle=False)
    dataset = Action0SegmentDataset(segmentation_root, ["user_0"], {"A": 0, "B": 1})

    with pytest.raises(Action0DatasetError, match="must be nonnegative"):
        _ = dataset[0]


def test_valid_padding_metadata_loads_and_keeps_diagnostic_counts(tmp_path: Path) -> None:
    segmentation_root = tmp_path / "segmentation_padded"
    _write_metadata(
        segmentation_root,
        processed_user_action_count=3,
        source_segment_count=12,
        segment_count=10,
        skipped_segment_count=2,
        board_assisted_package_count=1,
        label_only_package_count=2,
        failed_package_count=0,
        input_root="/relocated/source",
        output_root="/relocated/output",
    )

    metadata = load_padding_dataset_metadata(segmentation_root)

    assert metadata.target_length == 4
    assert metadata.sampling_rate_hz == 200.0
    assert metadata.diagnostic_counts == {
        "processed_user_action_count": 3,
        "source_segment_count": 12,
        "segment_count": 10,
        "skipped_segment_count": 2,
        "board_assisted_package_count": 1,
        "label_only_package_count": 2,
        "failed_package_count": 0,
    }


def test_missing_padding_metadata_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="padding_dataset_summary.json"):
        load_padding_dataset_metadata(tmp_path / "segmentation_padded")


def test_malformed_padding_metadata_fails(tmp_path: Path) -> None:
    segmentation_root = tmp_path / "segmentation_padded"
    segmentation_root.mkdir()
    (segmentation_root / "padding_dataset_summary.json").write_text(
        "not-json", encoding="utf-8"
    )

    with pytest.raises(Action0DatasetError, match="could not read padded dataset summary"):
        load_padding_dataset_metadata(segmentation_root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_kind", "raw-ring", "input_kind"),
        ("event_channel_count", 14, "event_channel_count"),
        ("event_feature_schema", "", "event_feature_schema"),
        ("spike_encoder_spec_sha256", "not-a-hash", "spike_encoder_spec_sha256"),
        ("target_length", 0, "target_length"),
        ("target_length", 4.0, "target_length"),
        ("sampling_rate_hz", 0.0, "sampling_rate_hz"),
        ("sampling_rate_hz", "200", "sampling_rate_hz"),
        ("padding_side", "left", "padding_side"),
    ],
)
def test_padding_metadata_contract_fields_are_validated(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    segmentation_root = tmp_path / "segmentation_padded"
    _write_metadata(segmentation_root, **{field: value})

    with pytest.raises(Action0DatasetError, match=message):
        load_padding_dataset_metadata(segmentation_root)
