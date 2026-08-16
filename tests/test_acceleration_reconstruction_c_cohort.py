from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.run_experiment_c as runner
from snn.accel_reconstruction_eval.config import UserSplitConfig, experiment_c_config
from snn.accel_reconstruction_eval.datasets import NormalizationStats
from snn.accel_reconstruction_eval.io import load_checkpoint as read_checkpoint


def _manifest(
    users: list[str],
    labels: tuple[str, ...] = ("a", "b"),
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for user in users:
        rows.extend(
            {
                "sample_id": f"{user}/sample_{label}",
                "user": user,
                "label": label,
            }
            for label in labels
        )
    return pd.DataFrame(rows)


def _checkpoint(
    *,
    eligible: list[str],
    excluded: list[str] | None = None,
    train: tuple[str, ...] = ("user_0",),
    val: tuple[str, ...] = ("user_1",),
    test: tuple[str, ...] = ("user_2",),
    require_all_users_assigned: bool = True,
    modern: bool = True,
    class_to_idx: dict[str, int] | None = None,
    random_seed: int = 12345,
) -> dict[str, object]:
    checkpoint: dict[str, object] = {
        "random_seed": random_seed,
        "train_users": list(train),
        "val_users": list(val),
        "test_users": list(test),
        "class_to_idx": {"a": 0, "b": 1}
        if class_to_idx is None
        else class_to_idx,
        "experiment_config": {
            "random_seed": random_seed,
            "split": {
                "require_all_users_assigned": require_all_users_assigned,
            }
        },
    }
    if modern:
        checkpoint["excluded_users"] = [] if excluded is None else excluded
        checkpoint["eligible_users"] = eligible
    return checkpoint


def _config(*, split: UserSplitConfig | None = None):
    return replace(experiment_c_config(), split=split or UserSplitConfig())


def test_modern_reference_inherits_exclusion_mapping_and_saved_assignment_policy() -> None:
    checkpoint = _checkpoint(
        eligible=[f"user_{index}" for index in range(5)],
        excluded=["5"],
        train=("0",),
        val=("user_1",),
        test=("user_2",),
        require_all_users_assigned=False,
    )
    manifest = _manifest([f"user_{index}" for index in range(6)])

    contract = runner._resolve_cohort_contract(manifest, checkpoint)
    result = runner._prepare_split(
        sample_manifest=manifest,
        config=_config(),
        reference_checkpoint=checkpoint,
        cohort=contract,
    )

    assert contract.source == "checkpoint"
    assert contract.excluded_users == ("user_5",)
    assert contract.require_all_users_assigned is False
    assert result.train_users == ("user_0",)
    assert result.val_users == ("user_1",)
    assert result.test_users == ("user_2",)
    assert result.class_to_idx == {"a": 0, "b": 1}
    assert set(result.sample_manifest["user"]) == {
        "user_0",
        "user_1",
        "user_2",
    }
    assert "user_5" not in set(result.sample_manifest["user"])


@pytest.mark.parametrize(
    "split",
    [
        UserSplitConfig(excluded_users=("user_5",)),
        UserSplitConfig(
            explicit_train_users=("user_0",),
            explicit_val_users=("user_1",),
            explicit_test_users=("user_2",),
        ),
        UserSplitConfig(included_labels=("a",)),
    ],
)
def test_local_c_cohort_selection_conflicts_with_reference(
    split: UserSplitConfig,
) -> None:
    with pytest.raises(ValueError, match="Experiment C local"):
        runner._prepare_split(
            sample_manifest=_manifest([f"user_{index}" for index in range(6)]),
            config=_config(split=split),
            reference_checkpoint=_checkpoint(
                eligible=[f"user_{index}" for index in range(5)],
                excluded=["user_5"],
            ),
        )


def test_modern_reference_filters_added_dataset_users() -> None:
    checkpoint = _checkpoint(
        eligible=[f"user_{index}" for index in range(4)],
        excluded=["user_4"],
    )
    result = runner._prepare_split(
        sample_manifest=_manifest(
            [f"user_{index}" for index in range(5)] + ["user_9"]
        ),
        config=_config(),
        reference_checkpoint=checkpoint,
    )
    assert set(result.sample_manifest["user"]) == {
        f"user_{index}" for index in range(3)
    }


def test_modern_reference_rejects_missing_split_user() -> None:
    checkpoint = _checkpoint(
        eligible=[f"user_{index}" for index in range(4)],
        excluded=["user_4"],
    )
    with pytest.raises(ValueError, match="no surviving rows"):
        runner._prepare_split(
            sample_manifest=_manifest(["user_0", "user_1", "user_3"]),
            config=_config(),
            reference_checkpoint=checkpoint,
        )


def test_reference_filter_keeps_only_checkpoint_labels() -> None:
    checkpoint = _checkpoint(
        eligible=["user_0", "user_1", "user_2"],
        class_to_idx={"a": 0, "b": 1},
    )
    result = runner._prepare_split(
        sample_manifest=_manifest(
            ["user_0", "user_1", "user_2", "user_9"],
            labels=("a", "b", "unselected"),
        ),
        config=_config(),
        reference_checkpoint=checkpoint,
    )
    assert set(result.sample_manifest["label"]) == {"a", "b"}
    assert set(result.sample_manifest["user"]) == {
        "user_0",
        "user_1",
        "user_2",
    }


def test_modern_reference_rejects_split_and_class_drift() -> None:
    checkpoint = _checkpoint(
        eligible=[f"user_{index}" for index in range(3)],
        train=("user_3",),
        class_to_idx={"a": 0, "other": 1},
    )

    with pytest.raises(ValueError, match="not eligible"):
        runner._resolve_cohort_contract(
            _manifest([f"user_{index}" for index in range(3)]),
            checkpoint,
        )

    checkpoint["train_users"] = ["user_0"]
    with pytest.raises(ValueError, match="class_to_idx labels are missing"):
        runner._prepare_split(
            sample_manifest=_manifest([f"user_{index}" for index in range(3)]),
            config=_config(),
            reference_checkpoint=checkpoint,
        )


def test_legacy_reference_uses_exact_no_exclusion_fallback() -> None:
    users = [f"user_{index}" for index in range(3)]
    checkpoint = _checkpoint(
        eligible=users,
        train=("0",),
        val=("user_1",),
        test=("user_2",),
        modern=False,
    )

    contract = runner._resolve_cohort_contract(_manifest(users), checkpoint)
    assert contract.source == "legacy_no_exclusion_fallback"
    assert contract.excluded_users == ()
    assert contract.eligible_users == tuple(users)

    result = runner._prepare_split(
        sample_manifest=_manifest(users + ["user_9"]),
        config=_config(),
        reference_checkpoint=checkpoint,
    )
    assert set(result.sample_manifest["user"]) == set(users)


@pytest.mark.parametrize(
    "update, message",
    [
        ({"eligible_users": ["user_0"]}, "both excluded_users and eligible_users"),
        ({"excluded_users": "user_3"}, "non-string user sequence"),
        ({"excluded_users": ["3", "user_3"]}, "duplicates"),
    ],
)
def test_malformed_reference_cohort_fields_are_rejected(
    update: dict[str, object],
    message: str,
) -> None:
    checkpoint = _checkpoint(
        eligible=["user_0", "user_1", "user_2"],
        excluded=["user_3"],
        random_seed=37,
    )
    if message.startswith("both"):
        checkpoint.pop("excluded_users")
    checkpoint.update(update)

    with pytest.raises(ValueError, match=message):
        runner._resolve_cohort_contract(
            _manifest(["user_0", "user_1", "user_2", "user_3"]),
            checkpoint,
        )


def _patch_c_execution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    data: SimpleNamespace,
) -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.weight = torch.nn.Parameter(torch.tensor([3.0]))

        def architecture_config(self) -> dict[str, object]:
            return {"class_name": "FakeModel", "num_classes": self.num_classes}

    monkeypatch.setattr(runner, "load_acceleration_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(runner, "MaskAwareAccelerationCNN", FakeModel)
    monkeypatch.setattr(
        runner,
        "fit_acceleration_normalization",
        lambda *args, **kwargs: NormalizationStats(
            mean=np.zeros(3),
            std=np.ones(3),
            valid_time_points=1,
            fitted_on="reconstruction:train",
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
        lambda *args, **kwargs: {"train": {}, "val": {}, "test": {}},
    )
    monkeypatch.setattr(
        runner,
        "evaluate_representation",
        lambda *args, **kwargs: SimpleNamespace(
            summary=pd.DataFrame({"metric": [1.0]})
        ),
    )
    monkeypatch.setattr(
        runner,
        "save_embedding_bundle",
        lambda path, bundle: Path(path),
    )
    monkeypatch.setattr(runner, "save_representation_evaluation", lambda *args, **kwargs: {})


def test_modern_run_persists_cohort_identity_and_reconstruction_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    users = [f"user_{index}" for index in range(4)]
    checkpoint = _checkpoint(
        eligible=["user_0", "user_1", "user_2"],
        excluded=["user_3"],
        random_seed=37,
    )
    checkpoint.update(
        {
            "artifact_type": "acceleration_cnn_experiment_checkpoint",
            "schema_version": 2,
            "model_state_dict": {"weight": torch.tensor([99.0])},
            "model_config": {"class_name": "ReferenceModel"},
            "normalization_mean": [9.0, 9.0, 9.0],
            "normalization_std": [2.0, 2.0, 2.0],
        }
    )
    reference_path = tmp_path / "reference.pt"
    reference_path.write_bytes(b"reference")
    data = SimpleNamespace(
        padded_root=tmp_path / "padded",
        packages=(),
        sample_manifest=_manifest(users),
        producer_metadata=SimpleNamespace(to_dict=lambda: {"producer": "test"}),
    )
    _patch_c_execution(monkeypatch, data=data)
    monkeypatch.setattr(runner, "load_checkpoint", lambda *args, **kwargs: checkpoint)

    result = runner.run_experiment_c(
        root=tmp_path / "source",
        repository_root=tmp_path,
        output_dir=tmp_path / "out",
        reference_checkpoint=reference_path,
        config=None,
        device="cpu",
    )

    saved = read_checkpoint(result.checkpoint_path)
    expected_identity = {
        "checkpoint": reference_path.resolve(),
        "sha256": hashlib.sha256(b"reference").hexdigest(),
        "artifact_type": "acceleration_cnn_experiment_checkpoint",
        "schema_version": 2,
    }
    assert saved["excluded_users"] == ["user_3"]
    assert saved["eligible_users"] == ["user_0", "user_1", "user_2"]
    assert saved["cohort_source"] == "checkpoint"
    assert saved["reference_identity"] == expected_identity
    assert saved["normalization_fitted_on"] == "reconstruction:train"
    assert saved["random_seed"] == 37
    assert saved["model_state_dict"]["weight"].item() == 3.0

    provenance = json.loads(
        (result.output_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["excluded_users"] == ["user_3"]
    assert provenance["eligible_users"] == ["user_0", "user_1", "user_2"]
    assert provenance["cohort_source"] == "checkpoint"
    assert provenance["reference_identity"] == {
        **expected_identity,
        "checkpoint": str(reference_path.resolve()),
    }
    assert result.normalization.fitted_on == "reconstruction:train"


def test_allow_new_split_records_standalone_local_cohort_without_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    users = [f"user_{index}" for index in range(4)]
    data = SimpleNamespace(
        padded_root=tmp_path / "padded",
        packages=(),
        sample_manifest=_manifest(users),
        producer_metadata=SimpleNamespace(to_dict=lambda: {"producer": "test"}),
    )
    _patch_c_execution(monkeypatch, data=data)

    missing_reference = tmp_path / "missing-reference.pt"
    config = _config(
        split=UserSplitConfig(
            explicit_train_users=("0",),
            explicit_val_users=("user_1",),
            explicit_test_users=("user_2",),
            excluded_users=("3",),
        )
    )
    result = runner.run_experiment_c(
        root=tmp_path / "source",
        repository_root=tmp_path,
        output_dir=tmp_path / "out",
        reference_checkpoint=missing_reference,
        config=config,
        device="cpu",
        allow_new_split=True,
    )

    checkpoint = read_checkpoint(result.checkpoint_path)
    assert checkpoint["cohort_source"] == "standalone_no_reference"
    assert checkpoint["excluded_users"] == ["user_3"]
    assert checkpoint["eligible_users"] == ["user_0", "user_1", "user_2"]
    assert checkpoint["reference_identity"] is None

    provenance = json.loads(
        (result.output_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["cohort_source"] == "standalone_no_reference"
    assert provenance["reference_identity"] is None
    assert provenance["split_reference_only"] is False
    assert set(provenance["train_users"]) == {"user_0"}
    assert set(provenance["val_users"]) == {"user_1"}
    assert set(provenance["test_users"]) == {"user_2"}
