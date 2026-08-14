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

import scripts.run_experiment_b as runner
from snn.accel_reconstruction_eval.config import UserSplitConfig, experiment_b_config
from snn.accel_reconstruction_eval.datasets import NormalizationStats


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
) -> dict[str, object]:
    checkpoint: dict[str, object] = {
        "train_users": list(train),
        "val_users": list(val),
        "test_users": list(test),
        "class_to_idx": {"a": 0, "b": 1}
        if class_to_idx is None
        else class_to_idx,
        "experiment_config": {
            "split": {
                "require_all_users_assigned": require_all_users_assigned,
            }
        },
    }
    if modern:
        checkpoint["excluded_users"] = [] if excluded is None else excluded
        checkpoint["eligible_users"] = eligible
    return checkpoint


def _b_config(*, split: UserSplitConfig | None = None):
    return replace(experiment_b_config(), split=split or UserSplitConfig())


def test_modern_checkpoint_inherits_exclusion_split_and_class_mapping() -> None:
    checkpoint = _checkpoint(
        eligible=["user_0", "user_1", "user_2", "user_3"],
        excluded=["4"],
        train=("0",),
        val=("user_1",),
        test=("user_2", "user_3"),
    )

    result = runner._prepare_split(
        sample_manifest=_manifest([f"user_{index}" for index in range(5)]),
        config=_b_config(),
        checkpoint=checkpoint,
    )

    assert result.train_users == ("user_0",)
    assert result.val_users == ("user_1",)
    assert result.test_users == ("user_2", "user_3")
    assert result.class_to_idx == {"a": 0, "b": 1}
    assert set(result.sample_manifest["user"]) == {
        "user_0",
        "user_1",
        "user_2",
        "user_3",
    }
    assert "user_4" not in set(result.sample_manifest["user"])


@pytest.mark.parametrize(
    "split",
    [
        UserSplitConfig(excluded_users=("user_9",)),
        UserSplitConfig(
            explicit_train_users=("user_0",),
            explicit_val_users=("user_1",),
            explicit_test_users=("user_2",),
        ),
        UserSplitConfig(included_labels=("a",)),
    ],
)
def test_local_b_cohort_selection_is_rejected(split: UserSplitConfig) -> None:
    with pytest.raises(
        ValueError,
        match="local (excluded_users|explicit user lists|included_labels)",
    ):
        runner._prepare_split(
            sample_manifest=_manifest([f"user_{index}" for index in range(4)]),
            config=_b_config(split=split),
            checkpoint=_checkpoint(
                eligible=[f"user_{index}" for index in range(4)],
                train=("user_0",),
                val=("user_1",),
                test=("user_2", "user_3"),
            ),
        )


def test_modern_checkpoint_filters_added_dataset_users() -> None:
    checkpoint = _checkpoint(
        eligible=["user_0", "user_1", "user_2", "user_3"],
        excluded=["user_4"],
        train=("user_0",),
        val=("user_1",),
        test=("user_2", "user_3"),
    )
    result = runner._prepare_split(
        sample_manifest=_manifest(
            ["user_0", "user_1", "user_2", "user_3", "user_9"]
        ),
        config=_b_config(),
        checkpoint=checkpoint,
    )
    assert set(result.sample_manifest["user"]) == {
        "user_0",
        "user_1",
        "user_2",
        "user_3",
    }


def test_modern_checkpoint_rejects_missing_split_user() -> None:
    checkpoint = _checkpoint(
        eligible=["user_0", "user_1", "user_2", "user_3"],
        excluded=["user_4"],
        train=("user_0",),
        val=("user_1",),
        test=("user_2", "user_3"),
    )
    with pytest.raises(ValueError, match="no surviving rows"):
        runner._prepare_split(
            sample_manifest=_manifest(["user_0", "user_1", "user_2"]),
            config=_b_config(),
            checkpoint=checkpoint,
        )


def test_modern_checkpoint_rejects_excluded_split_leakage() -> None:
    checkpoint = _checkpoint(
        eligible=["user_0", "user_1", "user_2"],
        excluded=["user_3"],
        train=("user_3",),
        val=("user_1",),
        test=("user_2",),
    )

    with pytest.raises(ValueError, match="not eligible"):
        runner._resolve_cohort_contract(_manifest([f"user_{i}" for i in range(4)]), checkpoint)


def test_class_label_mismatch_is_rejected_after_cohort_validation() -> None:
    checkpoint = _checkpoint(
        eligible=["user_0", "user_1", "user_2"],
        train=("user_0",),
        val=("user_1",),
        test=("user_2",),
        class_to_idx={"a": 0, "other": 1},
    )

    with pytest.raises(ValueError, match="class_to_idx labels are missing"):
        runner._prepare_split(
            sample_manifest=_manifest(["user_0", "user_1", "user_2"]),
            config=_b_config(),
            checkpoint=checkpoint,
        )


def test_checkpoint_filter_keeps_only_authorized_labels() -> None:
    checkpoint = _checkpoint(
        eligible=["user_0", "user_1", "user_2"],
        class_to_idx={"a": 0, "b": 1},
    )
    result = runner._prepare_split(
        sample_manifest=_manifest(
            ["user_0", "user_1", "user_2", "user_9"],
            labels=("a", "b", "unselected"),
        ),
        config=_b_config(),
        checkpoint=checkpoint,
    )
    assert set(result.sample_manifest["label"]) == {"a", "b"}
    assert set(result.sample_manifest["user"]) == {
        "user_0",
        "user_1",
        "user_2",
    }


def test_checkpoint_filter_rejects_label_only_available_outside_split_users() -> None:
    checkpoint = _checkpoint(
        eligible=["user_0", "user_1", "user_2"],
        class_to_idx={"a": 0, "c": 1},
    )
    manifest = pd.concat(
        [
            _manifest(["user_0", "user_1", "user_2"], labels=("a",)),
            _manifest(["user_9"], labels=("c",)),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="class_to_idx labels are missing"):
        runner._prepare_split(
            sample_manifest=manifest,
            config=_b_config(),
            checkpoint=checkpoint,
        )


def test_saved_a_assignment_policy_allows_eligible_but_unassigned_users() -> None:
    checkpoint = _checkpoint(
        eligible=[f"user_{index}" for index in range(5)],
        train=("user_0",),
        val=("user_1",),
        test=("user_2",),
        require_all_users_assigned=False,
    )
    manifest = _manifest([f"user_{index}" for index in range(5)])

    contract = runner._resolve_cohort_contract(manifest, checkpoint)
    result = runner._prepare_split(
        sample_manifest=manifest,
        config=_b_config(),
        checkpoint=checkpoint,
        cohort=contract,
    )

    assert contract.require_all_users_assigned is False
    assert contract.eligible_users == tuple(f"user_{i}" for i in range(5))
    assert set(result.sample_manifest["user"]) == {"user_0", "user_1", "user_2"}


def test_empty_modern_exclusion_is_inherited_without_filtering() -> None:
    users = [f"user_{index}" for index in range(3)]
    checkpoint = _checkpoint(
        eligible=users,
        excluded=[],
        train=("user_0",),
        val=("user_1",),
        test=("user_2",),
    )

    contract = runner._resolve_cohort_contract(_manifest(users), checkpoint)
    result = runner._prepare_split(
        sample_manifest=_manifest(users),
        config=_b_config(),
        checkpoint=checkpoint,
        cohort=contract,
    )

    assert contract.excluded_users == ()
    assert set(result.sample_manifest["user"]) == set(users)


def test_legacy_checkpoint_uses_explicit_no_exclusion_fallback() -> None:
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
        config=_b_config(),
        checkpoint=checkpoint,
    )
    assert set(result.sample_manifest["user"]) == set(users)


@pytest.mark.parametrize(
    "checkpoint_update, message",
    [
        ({"eligible_users": ["user_0"]}, "both excluded_users and eligible_users"),
        ({"excluded_users": "user_0"}, "non-string user sequence"),
        ({"excluded_users": ["1", "user_1"]}, "duplicates"),
    ],
)
def test_malformed_checkpoint_cohort_fields_are_rejected(
    checkpoint_update: dict[str, object],
    message: str,
) -> None:
    checkpoint = _checkpoint(
        eligible=["user_0", "user_1", "user_2"],
        excluded=["user_3"],
        train=("user_0",),
        val=("user_1",),
        test=("user_2",),
    )
    if message.startswith("both"):
        checkpoint.pop("excluded_users")
    checkpoint.update(checkpoint_update)

    with pytest.raises(ValueError, match=message):
        runner._resolve_cohort_contract(
            _manifest(["user_0", "user_1", "user_2", "user_3"]),
            checkpoint,
        )


def test_b_provenance_records_effective_checkpoint_cohort_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    users = [f"user_{index}" for index in range(4)]
    checkpoint = _checkpoint(
        eligible=["user_0", "user_1", "user_2"],
        excluded=["user_3"],
        train=("user_0",),
        val=("user_1",),
        test=("user_2",),
    )
    checkpoint.update(
        {
            "artifact_type": "acceleration_cnn_checkpoint",
            "schema_version": 2,
            "model_config": {"class_name": "FakeModel"},
            "normalization_mean": [0.0, 0.0, 0.0],
            "normalization_std": [1.0, 1.0, 1.0],
            "model_state_dict": {},
        }
    )
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    data = SimpleNamespace(
        padded_root=tmp_path / "padded",
        packages=(),
        sample_manifest=_manifest(users),
        producer_metadata=SimpleNamespace(to_dict=lambda: {"producer": "test"}),
    )

    class FakeModel(torch.nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.num_classes = num_classes

    monkeypatch.setattr(runner, "load_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(runner, "load_acceleration_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(runner, "MaskAwareAccelerationCNN", FakeModel)
    monkeypatch.setattr(
        runner,
        "normalization_from_checkpoint",
        lambda value: NormalizationStats(
            mean=np.zeros(3),
            std=np.ones(3),
            valid_time_points=1,
            fitted_on="raw:train",
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_split_loaders",
        lambda *args, **kwargs: {name: object() for name in ("train_eval", "val", "test")},
    )
    monkeypatch.setattr(runner, "build_cross_entropy", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner,
        "restore_model_from_checkpoint",
        lambda model, *args, **kwargs: (
            [parameter.requires_grad_(False) for parameter in model.parameters()],
            model.eval(),
        )[1],
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
            name: {}
            for name in (
                "train_raw",
                "val_raw",
                "test_raw",
                "test_reconstruction",
            )
        },
    )
    evaluation = SimpleNamespace(summary=pd.DataFrame({"metric": [1.0]}))
    monkeypatch.setattr(runner, "evaluate_representation", lambda *args, **kwargs: evaluation)
    monkeypatch.setattr(
        runner,
        "evaluate_paired_preservation",
        lambda *args, **kwargs: SimpleNamespace(
            summary=pd.DataFrame({"metric": [1.0]}),
            transitions=pd.DataFrame({"transition": ["same"]}),
        ),
    )
    monkeypatch.setattr(
        runner,
        "compare_representation_results",
        lambda *args, **kwargs: pd.DataFrame({"metric": [1.0]}),
    )
    monkeypatch.setattr(runner, "save_embedding_bundle", lambda path, bundle: Path(path))
    monkeypatch.setattr(runner, "save_representation_evaluation", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "save_paired_preservation", lambda *args, **kwargs: {})

    result = runner.run_experiment_b(
        root=tmp_path / "source",
        repository_root=tmp_path,
        output_dir=tmp_path / "out",
        baseline_checkpoint=baseline,
        config=_b_config(),
        device="cpu",
    )

    provenance = json.loads(
        (result.output_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["excluded_users"] == ["user_3"]
    assert provenance["eligible_users"] == ["user_0", "user_1", "user_2"]
    assert provenance["train_users"] == ["user_0"]
    assert provenance["val_users"] == ["user_1"]
    assert provenance["test_users"] == ["user_2"]
    assert provenance["class_to_idx"] == {"a": 0, "b": 1}
    assert provenance["cohort_source"] == "checkpoint"
    assert provenance["baseline_identity"]["sha256"] == hashlib.sha256(
        b"baseline"
    ).hexdigest()
