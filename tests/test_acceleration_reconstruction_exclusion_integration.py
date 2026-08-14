from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import scripts.run_experiment_b as runner_b
import scripts.run_experiment_c as runner_c
import scripts.run_experiment_d as runner_d
import scripts.run_experiment_a as runner_a
from snn.accel_reconstruction_eval.config import UserSplitConfig, experiment_a_config
from snn.accel_reconstruction_eval.datasets import NormalizationStats
from snn.accel_reconstruction_eval.embedding import EmbeddingBundle
from snn.accel_reconstruction_eval.io import (
    load_checkpoint,
    save_embedding_bundle as write_embedding_bundle,
)


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for user in ("user_0", "user_1", "user_2", "user_3"):
        package_index = int(user.removeprefix("user_"))
        for segment_index, label in enumerate(("a", "b", "unselected")):
            rows.append(
                {
                    "sample_id": f"{user}/sample_{label}",
                    "user": user,
                    "label": label,
                    "package_index": package_index,
                    "segment_index": segment_index,
                }
            )
    return pd.DataFrame(rows)


def _embedding_bundle(sample_manifest: pd.DataFrame) -> EmbeddingBundle:
    sample_ids = sample_manifest["sample_id"].astype(str).to_numpy()
    labels = sample_manifest["label_idx"].to_numpy(np.int64)
    count = len(sample_ids)
    h = np.ones((count, 2), dtype=np.float32)
    z = h / np.linalg.norm(h, axis=1, keepdims=True)
    logits = np.zeros((count, 2), dtype=np.float32)
    return EmbeddingBundle(
        h=h,
        z=z,
        y=labels,
        cnn_pred=np.zeros(count, dtype=np.int64),
        logits=logits,
        sample_id=sample_ids,
    )


def test_a_checkpoint_and_bcd_split_preparation_preserve_selected_cohort_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_manifest = _manifest()
    data = SimpleNamespace(
        padded_root=tmp_path / "padded",
        producer_metadata=SimpleNamespace(to_dict=lambda: {"producer": "test"}),
        packages=(),
        sample_manifest=source_manifest,
    )
    loader_manifests: list[pd.DataFrame] = []
    embedding_sample_ids: dict[str, set[str]] = {}

    class FakeModel(torch.nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def architecture_config(self) -> dict[str, object]:
            return {"class_name": "FakeModel", "num_classes": self.num_classes}

    monkeypatch.setattr(runner_a, "load_acceleration_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(runner_a, "MaskAwareAccelerationCNN", FakeModel)
    monkeypatch.setattr(
        runner_a,
        "fit_acceleration_normalization",
        lambda *args, **kwargs: NormalizationStats(
            mean=np.zeros(3),
            std=np.ones(3),
            valid_time_points=1,
            fitted_on="raw:train",
        ),
    )

    def build_loaders(*args, **kwargs):
        split_manifest = args[1].copy()
        loader_manifests.append(split_manifest)
        return {
            "train": ("train", split_manifest),
            "train_eval": ("train", split_manifest),
            "val": ("val", split_manifest),
            "test": ("test", split_manifest),
        }

    monkeypatch.setattr(runner_a, "build_split_loaders", build_loaders)
    monkeypatch.setattr(runner_a, "build_cross_entropy", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner_a, "build_adam_optimizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner_a,
        "fit_cnn",
        lambda model, **kwargs: (
            pd.DataFrame({"epoch": [1]}),
            model.state_dict(),
            1,
            0.75,
        ),
    )
    monkeypatch.setattr(
        runner_a,
        "evaluate_cnn_splits",
        lambda *args, **kwargs: pd.DataFrame({"split": ["test"]}),
    )

    def extract_embeddings(*args, **kwargs):
        loaders = args[1]
        return {
            split_name: _embedding_bundle(
                split_manifest.loc[split_manifest["split"] == source_split]
            )
            for split_name, (source_split, split_manifest) in loaders.items()
        }

    monkeypatch.setattr(runner_a, "extract_embedding_splits", extract_embeddings)
    monkeypatch.setattr(
        runner_a,
        "evaluate_representation",
        lambda *args, **kwargs: SimpleNamespace(
            summary=pd.DataFrame({"metric": ["test"]})
        ),
    )

    def save_embeddings(path, bundle):
        embedding_sample_ids[Path(path).name] = set(bundle["sample_id"].astype(str))
        return write_embedding_bundle(path, bundle)

    monkeypatch.setattr(runner_a, "save_embedding_bundle", save_embeddings)
    monkeypatch.setattr(
        runner_a,
        "save_representation_evaluation",
        lambda output_dir, evaluation: {},
    )

    config = replace(
        experiment_a_config(output_dir=tmp_path, random_seed=7),
        split=UserSplitConfig(
            explicit_train_users=("0",),
            explicit_val_users=("user_1",),
            explicit_test_users=("user_2",),
            excluded_users=("3",),
            included_labels=("a", "b"),
        ),
    )
    result_a = runner_a.run_experiment_a(
        root=tmp_path / "source",
        repository_root=tmp_path,
        output_dir=tmp_path / "a_artifacts",
        config=config,
        device="cpu",
    )

    checkpoint = load_checkpoint(result_a.checkpoint_path)
    expected_labels = {"a", "b"}
    expected_excluded_ids = {
        f"user_3/sample_{label}"
        for label in ("a", "b", "unselected")
    }
    expected_eligible_ids = {
        f"user_{user}/sample_{label}"
        for user in range(3)
        for label in expected_labels
    }
    expected_eligible_identity = {
        (user, segment_index)
        for user in range(3)
        for segment_index, label in enumerate(("a", "b", "unselected"))
        if label in expected_labels
    }

    assert result_a.artifact_paths["checkpoint"].is_file()
    assert result_a.artifact_paths["provenance"].is_file()
    assert result_a.artifact_paths["cohort"].is_file()
    assert checkpoint["excluded_users"] == ["user_3"]
    assert checkpoint["eligible_users"] == ["user_0", "user_1", "user_2"]
    assert checkpoint["class_to_idx"] == {"a": 0, "b": 1}
    assert len(loader_manifests) == 1
    assert not set(loader_manifests[0]["sample_id"]).intersection(expected_excluded_ids)
    assert set(loader_manifests[0]["sample_id"]) == expected_eligible_ids
    assert set(loader_manifests[0]["label"]) == expected_labels
    assert set(
        zip(loader_manifests[0]["package_index"], loader_manifests[0]["segment_index"])
    ) == expected_eligible_identity
    assert embedding_sample_ids
    assert all(not ids.intersection(expected_excluded_ids) for ids in embedding_sample_ids.values())
    assert all(ids.issubset(expected_eligible_ids) for ids in embedding_sample_ids.values())

    for runner, config_factory, kwargs in (
        (runner_b, runner_b.experiment_b_config, {"checkpoint": checkpoint}),
        (runner_c, runner_c.experiment_c_config, {"reference_checkpoint": checkpoint}),
        (
            runner_d,
            lambda: runner_d.experiment_d_config(test_source="raw"),
            {"reference_checkpoint": checkpoint},
        ),
    ):
        split = runner._prepare_split(
            sample_manifest=source_manifest,
            config=config_factory(),
            **kwargs,
        )
        assert split.train_users == ("user_0",)
        assert split.val_users == ("user_1",)
        assert split.test_users == ("user_2",)
        assert set(split.sample_manifest["sample_id"]) == expected_eligible_ids
        assert set(split.sample_manifest["label"]) == expected_labels
        assert split.class_to_idx == checkpoint["class_to_idx"]
        assert set(
            zip(
                split.sample_manifest["package_index"],
                split.sample_manifest["segment_index"],
            )
        ) == expected_eligible_identity
        assert not set(split.sample_manifest["sample_id"]).intersection(
            expected_excluded_ids
        )
