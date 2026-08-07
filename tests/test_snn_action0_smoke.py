from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("snntorch")

from snn.train_action0 import main


def _write_package(root: Path, user: str) -> None:
    directory = root / "low-pass" / "label" / "segmentation_padded" / user / "action_0"
    directory.mkdir(parents=True)
    stem = f"{user}_action_0"
    values = np.zeros((2, 4, 21), dtype=np.float32)
    values[0, :, :15] = 1.0
    values[1, :, :15] = 0.5
    valid_lengths = np.asarray([4, 3], dtype=np.int32)
    valid_mask = np.arange(4)[None, :] < valid_lengths[:, None]
    np.save(directory / f"{stem}_paddedSpikeIMU.npy", values, allow_pickle=False)
    np.save(directory / f"{stem}_labels.npy", np.asarray(["A", "B"]), allow_pickle=False)
    np.save(directory / f"{stem}_valid_lengths.npy", valid_lengths, allow_pickle=False)
    np.save(directory / f"{stem}_valid_mask.npy", valid_mask, allow_pickle=False)


def test_dry_run_completes_data_forward_loss_backward_and_step(tmp_path: Path) -> None:
    for user in ("user_0", "user_1", "user_2"):
        _write_package(tmp_path, user)

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


def test_one_epoch_runs_train_validation_and_test(tmp_path: Path) -> None:
    for user in ("user_0", "user_1", "user_2"):
        _write_package(tmp_path, user)

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
