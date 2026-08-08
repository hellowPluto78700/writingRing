from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("snntorch")
import torch

import snn.train_action0 as train_action0
from snn.train_action0 import main
from snn.action0_dataset import Action0DatasetError, SPIKE_IMU_FEATURE_SCHEMA


def _write_package(root: Path, user: str, *, padded_length: int = 4) -> None:
    directory = root / "low-pass" / "label" / "segmentation_padded" / user / "action_0"
    directory.mkdir(parents=True)
    stem = f"{user}_action_0"
    values = np.zeros((2, padded_length, 21), dtype=np.float32)
    values[0, :, :15] = 1.0
    values[1, :, :15] = 0.5
    valid_lengths = np.asarray([padded_length, padded_length - 1], dtype=np.int32)
    valid_mask = np.arange(padded_length)[None, :] < valid_lengths[:, None]
    np.save(directory / f"{stem}_paddedSpikeIMU.npy", values, allow_pickle=False)
    np.save(directory / f"{stem}_labels.npy", np.asarray(["A", "B"]), allow_pickle=False)
    np.save(directory / f"{stem}_valid_lengths.npy", valid_lengths, allow_pickle=False)
    np.save(directory / f"{stem}_valid_mask.npy", valid_mask, allow_pickle=False)


def _write_metadata(
    root: Path, *, target_length: int = 4, sampling_rate_hz: float = 200.0
) -> None:
    directory = root / "low-pass" / "label" / "segmentation_padded"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_kind": "spike-imu",
        "feature_schema": SPIKE_IMU_FEATURE_SCHEMA,
        "channel_count": 21,
        "target_length": target_length,
        "sampling_rate_hz": sampling_rate_hz,
        "padding_side": "right",
    }
    (directory / "padding_dataset_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_dry_run_completes_data_forward_loss_backward_and_step(tmp_path: Path) -> None:
    for user in ("user_0", "user_1", "user_2"):
        _write_package(tmp_path, user)
    _write_metadata(tmp_path)

    result = main(
        [
            "--pipeline_root",
            str(tmp_path),
            "--dataset_variant",
            "lowpass",
            "--sample_freq",
            "200",
            "--train_users",
            "user_0",
            "--val_users",
            "user_1",
            "--test_users",
            "user_2",
            "--batch_size",
            "1",
            "--dry_run",
        ]
    )

    assert result["mode"] == "dry_run"
    assert result["class_to_idx"] == {"A": 0, "B": 1}
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["loss"] >= 0
    assert 0 <= metrics["zero_output_spike_fraction"] <= 1


def test_dry_run_reuses_inspected_batch_for_one_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for user in ("user_0", "user_1", "user_2"):
        _write_package(tmp_path, user)
    _write_metadata(tmp_path)

    observed: dict[str, object] = {}
    original_inspect = train_action0._inspect_first_batch

    def capture_inspect(
        inputs: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor
    ) -> None:
        observed["inspected"] = (
            inputs.detach().clone(),
            labels.detach().clone(),
            valid_mask.detach().clone(),
        )
        original_inspect(inputs, labels, valid_mask)

    monkeypatch.setattr(train_action0, "_inspect_first_batch", capture_inspect)

    original_run_epoch = train_action0.run_epoch

    def capture_run_epoch(model: torch.nn.Module, dataloader: object, *args: object, **kwargs: object) -> dict[str, float]:
        batches = list(dataloader)  # type: ignore[arg-type]
        observed["optimized_batches"] = batches
        observed["expected_num_classes"] = kwargs["expected_num_classes"]
        return original_run_epoch(model, batches, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(train_action0, "run_epoch", capture_run_epoch)

    optimizer_steps = 0
    original_step = torch.optim.Adam.step

    def count_step(optimizer: torch.optim.Adam, *args: object, **kwargs: object) -> None:
        nonlocal optimizer_steps
        optimizer_steps += 1
        original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", count_step)

    main(
        [
            "--pipeline_root",
            str(tmp_path),
            "--dataset_variant",
            "lowpass",
            "--sample_freq",
            "200",
            "--train_users",
            "user_0",
            "--val_users",
            "user_1",
            "--test_users",
            "user_2",
            "--batch_size",
            "1",
            "--dry_run",
        ]
    )

    optimized_batches = observed["optimized_batches"]
    assert isinstance(optimized_batches, list)
    assert len(optimized_batches) == 1
    inspected = observed["inspected"]
    assert isinstance(inspected, tuple)
    optimized = optimized_batches[0]
    assert isinstance(optimized, tuple)
    for inspected_tensor, optimized_tensor in zip(inspected, optimized):
        torch.testing.assert_close(inspected_tensor, optimized_tensor)
    assert observed["expected_num_classes"] == 2
    assert optimizer_steps == 1


def test_one_epoch_runs_train_validation_and_test(tmp_path: Path) -> None:
    for user in ("user_0", "user_1", "user_2"):
        _write_package(tmp_path, user)
    _write_metadata(tmp_path)

    result = main(
        [
            "--pipeline_root",
            str(tmp_path),
            "--dataset_variant",
            "lowpass",
            "--sample_freq",
            "200",
            "--train_users",
            "user_0",
            "--val_users",
            "user_1",
            "--test_users",
            "user_2",
            "--batch_size",
            "1",
            "--num_epochs",
            "1",
            "--max_train_batches",
            "1",
        ]
    )

    assert result["mode"] == "train"
    assert result["best_epoch"] == 0
    test_metrics = result["test_metrics"]
    assert isinstance(test_metrics, dict)
    assert test_metrics["loss"] >= 0


def test_sample_frequency_mismatch_fails_before_training(tmp_path: Path) -> None:
    for user in ("user_0", "user_1", "user_2"):
        _write_package(tmp_path, user)
    _write_metadata(tmp_path, sampling_rate_hz=200.0)

    with pytest.raises(Action0DatasetError, match="sample_freq.*sampling_rate_hz"):
        _run_main(tmp_path, sample_freq=201.0)


def test_package_padded_length_must_match_producer_target(tmp_path: Path) -> None:
    _write_package(tmp_path, "user_0")
    _write_package(tmp_path, "user_1", padded_length=5)
    _write_package(tmp_path, "user_2")
    _write_metadata(tmp_path, target_length=4)

    with pytest.raises(Action0DatasetError, match="padded lengths.*target_length"):
        _run_main(tmp_path)


def test_cross_split_padded_length_mismatch_fails(tmp_path: Path) -> None:
    _write_package(tmp_path, "user_0")
    _write_package(tmp_path, "user_1")
    _write_package(tmp_path, "user_2", padded_length=5)
    _write_metadata(tmp_path, target_length=4)

    with pytest.raises(Action0DatasetError, match="padded lengths.*target_length"):
        _run_main(tmp_path)


def _run_main(tmp_path: Path, *, sample_freq: float = 200.0) -> dict[str, object]:
    return main(
        [
            "--pipeline_root",
            str(tmp_path),
            "--dataset_variant",
            "lowpass",
            "--sample_freq",
            str(sample_freq),
            "--train_users",
            "user_0",
            "--val_users",
            "user_1",
            "--test_users",
            "user_2",
            "--batch_size",
            "1",
            "--dry_run",
        ]
    )
