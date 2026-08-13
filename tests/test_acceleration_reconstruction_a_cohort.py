from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import scripts.run_experiment_a as runner
from snn.accel_reconstruction_eval.config import UserSplitConfig, experiment_a_config
from snn.accel_reconstruction_eval.datasets import NormalizationStats
from snn.accel_reconstruction_eval.io import load_checkpoint


def _manifest(users: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [
                f"{user}/sample_{label}"
                for user in users
                for label in ("a", "b")
            ],
            "user": [user for user in users for _ in ("a", "b")],
            "label": [label for _ in users for label in ("a", "b")],
        }
    )


def test_cohort_artifacts_capture_source_cohort_and_post_split_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_manifest = _manifest([f"user_{index}" for index in range(1, 7)])
    data = SimpleNamespace(
        padded_root=tmp_path / "padded",
        producer_metadata=SimpleNamespace(to_dict=lambda: {"producer": "test"}),
        packages=(),
        sample_manifest=source_manifest,
    )

    class FakeModel(torch.nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def architecture_config(self) -> dict[str, object]:
            return {"class_name": "FakeModel", "num_classes": self.num_classes}

    monkeypatch.setattr(
        runner,
        "load_acceleration_data",
        lambda *args, **kwargs: data,
    )
    monkeypatch.setattr(runner, "MaskAwareAccelerationCNN", FakeModel)
    monkeypatch.setattr(
        runner,
        "fit_acceleration_normalization",
        lambda *args, **kwargs: NormalizationStats(
            mean=np.zeros(3),
            std=np.ones(3),
            valid_time_points=1,
            fitted_on="raw:train",
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_split_loaders",
        lambda *args, **kwargs: {
            "train": object(),
            "train_eval": object(),
            "val": object(),
            "test": object(),
        },
    )
    monkeypatch.setattr(runner, "build_cross_entropy", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "build_adam_optimizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner,
        "fit_cnn",
        lambda model, **kwargs: (
            pd.DataFrame({"epoch": [1]}),
            model.state_dict(),
            1,
            0.75,
        ),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_cnn_splits",
        lambda *args, **kwargs: pd.DataFrame({"split": ["test"]}),
    )
    monkeypatch.setattr(
        runner,
        "extract_embedding_splits",
        lambda *args, **kwargs: {
            "train": {},
            "val": {},
            "test": {},
        },
    )
    monkeypatch.setattr(
        runner,
        "evaluate_representation",
        lambda *args, **kwargs: SimpleNamespace(
            summary=pd.DataFrame({"metric": ["test"]})
        ),
    )
    monkeypatch.setattr(
        runner,
        "save_embedding_bundle",
        lambda path, bundle: Path(path),
    )
    monkeypatch.setattr(
        runner,
        "save_representation_evaluation",
        lambda output_dir, evaluation: {},
    )

    config = replace(
        experiment_a_config(output_dir=tmp_path, random_seed=7),
        split=UserSplitConfig(
            explicit_train_users=("user_1",),
            explicit_val_users=("user_3",),
            explicit_test_users=("user_4",),
            require_all_users_assigned=False,
            excluded_users=("2", "user_6"),
        ),
    )

    result = runner.run_experiment_a(
        root=tmp_path / "source",
        repository_root=tmp_path,
        output_dir=tmp_path / "artifacts",
        config=config,
        device="cpu",
    )

    expected_cohort = {
        "excluded_users": ["user_2", "user_6"],
        "eligible_users": ["user_1", "user_3", "user_4", "user_5"],
    }
    checkpoint = load_checkpoint(result.checkpoint_path)
    assert {
        key: checkpoint[key]
        for key in ("excluded_users", "eligible_users")
    } == expected_cohort

    provenance = json.loads(
        (result.output_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert {
        key: provenance[key]
        for key in ("excluded_users", "eligible_users")
    } == expected_cohort
    assert {
        key: provenance[key]
        for key in ("train_users", "val_users", "test_users", "class_to_idx")
    }
    assert result.artifact_paths["cohort"] == result.output_dir / "cohort.json"

    cohort = json.loads(
        (result.output_dir / "cohort.json").read_text(encoding="utf-8")
    )
    assert cohort == expected_cohort
    assert set(cohort) == {"excluded_users", "eligible_users"}

    saved_manifest = pd.read_csv(result.output_dir / "sample_manifest.csv")
    assert set(saved_manifest["user"]) == {"user_1", "user_3", "user_4"}
    assert not set(saved_manifest["user"]).intersection(
        expected_cohort["excluded_users"]
    )
    assert "user_5" not in set(saved_manifest["user"])


def test_cohort_payload_normalizes_aliases_and_sorts_empty_exclusion_cohort() -> None:
    payload = runner._build_cohort_payload(
        pd.DataFrame(
            {
                "user": ["user_10", "2", "user_1", "research participant"],
            }
        ),
        excluded_users=(),
    )

    assert payload == {
        "excluded_users": [],
        "eligible_users": [
            "research participant",
            "user_1",
            "user_2",
            "user_10",
        ],
    }
