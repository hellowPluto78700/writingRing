from __future__ import annotations

"""Run Experiment C: reconstruction-trained CNN -> reconstruction test.

Protocol
--------
1. Load the padded dataset and standalone reconstructed acceleration.
2. Reuse Experiment A's user split and class mapping when a reference
   checkpoint is supplied. The checkpoint is used for split/class metadata
   only; no model weights or normalization statistics are reused.
3. Fit acceleration normalization on reconstruction TRAIN valid samples only.
4. Train a new MaskAwareAccelerationCNN from scratch on reconstruction train.
5. Select the best epoch on reconstruction validation balanced accuracy.
6. Evaluate reconstruction test classification and the shared representation
   protocol (kNN, retrieval, prototype, linear probe, geometry, clustering).
7. Save a self-describing Experiment C checkpoint and all reusable artifacts.
"""

import argparse
import hashlib
import random
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from snn.accel_reconstruction_eval import (
    ExperimentConfig,
    MaskAwareAccelerationCNN,
    NormalizationStats,
    SplitAssignment,
    build_adam_optimizer,
    build_cross_entropy,
    build_experiment_checkpoint,
    build_split_loaders,
    evaluate_cnn_splits,
    evaluate_representation,
    experiment_c_config,
    extract_embedding_splits,
    fit_acceleration_normalization,
    fit_cnn,
    load_acceleration_data,
    load_checkpoint,
    prepare_user_disjoint_splits,
    save_checkpoint,
    save_embedding_bundle,
    save_provenance,
    save_representation_evaluation,
)
from snn.accel_reconstruction_eval.datasets import (
    natural_key,
    normalize_user_list,
    normalize_user_name,
)


DEFAULT_DATASET_ROOT = Path(
    "outputs/action0_rectified/low-pass/aligned-board-events"
)
DEFAULT_REFERENCE_CHECKPOINT = Path(
    "notebooks/artifacts/acceleration_cnn_representation/"
    "best_acceleration_cnn.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "notebooks/artifacts/experiment_C_reconstruction_trained"
)


@dataclass(frozen=True)
class ExperimentCRunResult:
    """Lightweight result returned to notebooks and programmatic callers."""

    output_dir: Path
    checkpoint_path: Path
    summary: pd.DataFrame
    classification_splits: pd.DataFrame
    training_history: pd.DataFrame
    split_summary: pd.DataFrame
    label_split_counts: pd.DataFrame
    normalization: NormalizationStats
    artifact_paths: Mapping[str, Path]


@dataclass(frozen=True)
class _CCohortContract:
    """Validated cohort authority used by Experiment C."""

    excluded_users: tuple[str, ...]
    eligible_users: tuple[str, ...]
    train_users: tuple[str, ...]
    val_users: tuple[str, ...]
    test_users: tuple[str, ...]
    class_to_idx: dict[str, int]
    require_all_users_assigned: bool
    source: str


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(path: str | Path, repository_root: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = repository_root / value
    return value.resolve()


def _select_device(config: ExperimentConfig, requested: str | None) -> torch.device:
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return device
    return torch.device(
        "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
    )


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_checkpoint_user_list(
    value: object,
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, SequenceABC
    ):
        raise ValueError(
            f"Checkpoint {field_name} must be a non-string user sequence"
        )
    if any(item is None for item in value):
        raise ValueError(f"Checkpoint {field_name} contains a null user")
    try:
        normalized = normalize_user_list(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid checkpoint {field_name}: {exc}") from exc
    return tuple(normalized)


def _checkpoint_user_field(
    checkpoint: Mapping[str, object],
    field_name: str,
) -> tuple[str, ...]:
    if field_name not in checkpoint:
        raise ValueError(f"Checkpoint is missing {field_name}")
    return _canonical_checkpoint_user_list(
        checkpoint[field_name],
        field_name=field_name,
    )


def _checkpoint_class_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, MappingABC):
        raise ValueError("Checkpoint class_to_idx must be a mapping")
    try:
        class_to_idx = {str(label): int(index) for label, index in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint class_to_idx must map labels to integers") from exc
    if not class_to_idx:
        raise ValueError("Checkpoint class_to_idx must not be empty")
    if sorted(class_to_idx.values()) != list(range(len(class_to_idx))):
        raise ValueError("Checkpoint class_to_idx values must be contiguous 0..C-1")
    return class_to_idx


def _source_user_cohort(sample_manifest: pd.DataFrame) -> set[str]:
    if "user" not in sample_manifest.columns:
        raise ValueError("sample_manifest is missing the user column")
    return {normalize_user_name(value) for value in sample_manifest["user"]}


def _checkpoint_assignment_policy(
    checkpoint: Mapping[str, object],
    *,
    modern: bool,
) -> bool:
    experiment_config = checkpoint.get("experiment_config")
    if not isinstance(experiment_config, MappingABC):
        if modern:
            raise ValueError(
                "Modern checkpoint must save experiment_config.split"
            )
        return True

    split_config = experiment_config.get("split")
    if not isinstance(split_config, MappingABC):
        if modern:
            raise ValueError(
                "Modern checkpoint must save experiment_config.split"
            )
        return True
    if "require_all_users_assigned" not in split_config:
        if modern:
            raise ValueError(
                "Modern checkpoint must save split.require_all_users_assigned"
            )
        return True
    value = split_config["require_all_users_assigned"]
    if not isinstance(value, bool):
        raise ValueError(
            "Checkpoint split.require_all_users_assigned must be a boolean"
        )
    return value


def _validate_local_cohort_config(config: ExperimentConfig) -> None:
    split_config = config.split
    if split_config.excluded_users:
        raise ValueError(
            "Experiment C local excluded_users must be empty; "
            "the Experiment A checkpoint is authoritative"
        )
    if any(
        value is not None
        for value in (
            split_config.explicit_train_users,
            split_config.explicit_val_users,
            split_config.explicit_test_users,
        )
    ):
        raise ValueError(
            "Experiment C local explicit user lists must all be None; "
            "the Experiment A checkpoint is authoritative"
        )


def _resolve_cohort_contract(
    sample_manifest: pd.DataFrame,
    checkpoint: Mapping[str, object],
) -> _CCohortContract:
    cohort_fields = (
        "excluded_users" in checkpoint,
        "eligible_users" in checkpoint,
    )
    if any(cohort_fields) and not all(cohort_fields):
        raise ValueError(
            "Checkpoint cohort contract must contain both excluded_users and "
            "eligible_users"
        )
    modern = all(cohort_fields)

    split_users = tuple(
        _checkpoint_user_field(checkpoint, field)
        for field in ("train_users", "val_users", "test_users")
    )
    train_users, val_users, test_users = split_users
    split_sets = {
        "train": set(train_users),
        "val": set(val_users),
        "test": set(test_users),
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_sets[left].intersection(split_sets[right])
        if overlap:
            raise ValueError(
                f"Checkpoint user leakage between {left} and {right}: "
                f"{sorted(overlap, key=natural_key)}"
            )

    if modern:
        excluded_users = _canonical_checkpoint_user_list(
            checkpoint["excluded_users"],
            field_name="excluded_users",
        )
        eligible_users = _canonical_checkpoint_user_list(
            checkpoint["eligible_users"],
            field_name="eligible_users",
        )
        excluded_set = set(excluded_users)
        eligible_set = set(eligible_users)
        if excluded_set.intersection(eligible_set):
            raise ValueError(
                "Checkpoint excluded_users and eligible_users overlap"
            )
        assigned_users = set().union(*split_sets.values())
        split_outside_eligible = assigned_users.difference(eligible_set)
        if split_outside_eligible:
            raise ValueError(
                "Checkpoint split users are not eligible: "
                f"{sorted(split_outside_eligible, key=natural_key)}"
            )

        source_users = _source_user_cohort(sample_manifest)
        expected_users = eligible_set.union(excluded_set)
        if source_users != expected_users:
            missing = sorted(
                expected_users.difference(source_users),
                key=natural_key,
            )
            additional = sorted(
                source_users.difference(expected_users),
                key=natural_key,
            )
            raise ValueError(
                "Source dataset user cohort does not match checkpoint cohort; "
                f"missing={missing}, additional={additional}"
            )
        source = "checkpoint"
    else:
        excluded_users = ()
        assigned_users = set().union(*split_sets.values())
        source_users = _source_user_cohort(sample_manifest)
        if source_users != assigned_users:
            missing = sorted(
                assigned_users.difference(source_users),
                key=natural_key,
            )
            additional = sorted(
                source_users.difference(assigned_users),
                key=natural_key,
            )
            raise ValueError(
                "Legacy checkpoint fallback requires the dataset user cohort "
                "to equal the checkpoint split-user union; "
                f"missing={missing}, additional={additional}"
            )
        eligible_users = tuple(sorted(assigned_users, key=natural_key))
        source = "legacy_no_exclusion_fallback"

    class_to_idx = _checkpoint_class_mapping(checkpoint.get("class_to_idx"))
    return _CCohortContract(
        excluded_users=excluded_users,
        eligible_users=eligible_users,
        train_users=train_users,
        val_users=val_users,
        test_users=test_users,
        class_to_idx=class_to_idx,
        require_all_users_assigned=_checkpoint_assignment_policy(
            checkpoint,
            modern=modern,
        ),
        source=source,
    )


def _build_standalone_cohort(
    sample_manifest: pd.DataFrame,
    config: ExperimentConfig,
    split: SplitAssignment,
) -> _CCohortContract:
    excluded_users = tuple(config.split.excluded_users)
    excluded_set = set(excluded_users)
    eligible_users = tuple(
        sorted(
            _source_user_cohort(sample_manifest).difference(excluded_set),
            key=natural_key,
        )
    )
    return _CCohortContract(
        excluded_users=excluded_users,
        eligible_users=eligible_users,
        train_users=split.train_users,
        val_users=split.val_users,
        test_users=split.test_users,
        class_to_idx=dict(split.class_to_idx),
        require_all_users_assigned=config.split.require_all_users_assigned,
        source="standalone_no_reference",
    )


def _prepare_split(
    *,
    sample_manifest: pd.DataFrame,
    config: ExperimentConfig,
    reference_checkpoint: Mapping[str, object] | None,
    cohort: _CCohortContract | None = None,
) -> SplitAssignment:
    split_config = config.split

    if reference_checkpoint is not None:
        _validate_local_cohort_config(config)
        if cohort is None:
            cohort = _resolve_cohort_contract(
                sample_manifest,
                reference_checkpoint,
            )
        return prepare_user_disjoint_splits(
            sample_manifest,
            seed=config.random_seed,
            explicit_train_users=cohort.train_users,
            explicit_val_users=cohort.val_users,
            explicit_test_users=cohort.test_users,
            excluded_users=cohort.excluded_users,
            require_all_users_assigned=cohort.require_all_users_assigned,
            require_all_labels_in_all_splits=(
                split_config.require_all_labels_in_all_splits
            ),
            class_to_idx=dict(cohort.class_to_idx),
        )

    return prepare_user_disjoint_splits(
        sample_manifest,
        train_fraction=split_config.train_fraction,
        val_fraction=split_config.val_fraction,
        seed=config.random_seed,
        explicit_train_users=split_config.explicit_train_users,
        explicit_val_users=split_config.explicit_val_users,
        explicit_test_users=split_config.explicit_test_users,
        excluded_users=split_config.excluded_users,
        require_all_users_assigned=split_config.require_all_users_assigned,
        require_all_labels_in_all_splits=(
            split_config.require_all_labels_in_all_splits
        ),
    )


def run_experiment_c(
    *,
    root: str | Path = DEFAULT_DATASET_ROOT,
    repository_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    reference_checkpoint: str | Path | None = DEFAULT_REFERENCE_CHECKPOINT,
    config: ExperimentConfig | None = None,
    device: str | None = None,
    allow_new_split: bool = False,
) -> ExperimentCRunResult:
    """Execute Experiment C end-to-end and save all reusable artifacts.

    Parameters
    ----------
    reference_checkpoint:
        Experiment A checkpoint used only to recover train/val/test users and
        ``class_to_idx``. No A weights or normalization are loaded into C.
        Set to ``None`` only together with ``allow_new_split=True`` if a new
        split is intentionally desired.
    allow_new_split:
        Safety valve for standalone use. The default is False because A/B/C/D
        should normally share the same user split.
    """
    repository_root = Path(repository_root).expanduser().resolve()
    output_dir = _resolve_path(output_dir, repository_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = experiment_c_config(
            output_dir=output_dir,
            random_seed=12345,
        )
    else:
        config = replace(config, output_dir=output_dir)
        config.validate()

    if not config.training_enabled:
        raise ValueError("Experiment C requires training_enabled=True")
    if (
        config.train_source != "reconstruction"
        or config.val_source != "reconstruction"
        or config.test_source != "reconstruction"
        or config.normalization_source != "reconstruction"
    ):
        raise ValueError(
            "Experiment C must use reconstruction for train/val/test and "
            "reconstruction-train normalization"
        )

    _set_global_seed(config.random_seed)
    torch_device = _select_device(config, device)

    reference_path: Path | None = None
    reference: dict[str, object] | None = None
    if reference_checkpoint is not None:
        reference_path = _resolve_path(reference_checkpoint, repository_root)
        if reference_path.is_file():
            reference = load_checkpoint(reference_path, map_location="cpu")
        elif not allow_new_split:
            raise FileNotFoundError(
                "Reference Experiment A checkpoint was not found: "
                f"{reference_path}. Supply the correct checkpoint or pass "
                "allow_new_split=True only if a new split is intentional."
            )
    elif not allow_new_split:
        raise ValueError(
            "Experiment C should reuse the A split. Supply reference_checkpoint, "
            "or set allow_new_split=True only if a new split is intentional."
        )

    data = load_acceleration_data(
        root,
        repository_root=repository_root,
        require_reconstruction=True,
    )
    reference_identity: dict[str, object] | None = None
    if reference is not None:
        if reference_path is None:
            raise AssertionError("A loaded reference checkpoint needs a path")
        reference_identity = {
            "checkpoint": reference_path,
            "sha256": _sha256_file(reference_path),
            "artifact_type": reference.get("artifact_type"),
            "schema_version": reference.get("schema_version"),
        }

    cohort: _CCohortContract | None = None
    if reference is not None:
        cohort = _resolve_cohort_contract(data.sample_manifest, reference)
    split = _prepare_split(
        sample_manifest=data.sample_manifest,
        config=config,
        reference_checkpoint=reference,
        cohort=cohort,
    )
    if cohort is None:
        cohort = _build_standalone_cohort(
            data.sample_manifest,
            config,
            split,
        )

    normalization = fit_acceleration_normalization(
        data.packages,
        split.sample_manifest,
        split="train",
        source="reconstruction",
        chunk_segments=config.normalization_chunk_segments,
    )
    if normalization.fitted_on != "reconstruction:train":
        raise AssertionError(
            "Experiment C normalization must be fitted on reconstruction:train; "
            f"got {normalization.fitted_on!r}"
        )

    loaders = build_split_loaders(
        data.packages,
        split.sample_manifest,
        train_source=config.train_source,
        val_source=config.val_source,
        test_source=config.test_source,
        normalization=normalization,
        batch_size=config.loader.batch_size,
        num_workers=config.loader.num_workers,
        use_gpu=(torch_device.type == "cuda"),
        seed=config.random_seed,
        pin_memory=config.loader.pin_memory,
        persistent_workers=config.loader.persistent_workers,
    )

    num_classes = len(split.class_to_idx)
    model = MaskAwareAccelerationCNN(num_classes=num_classes).to(torch_device)

    train_labels = split.sample_manifest.loc[
        split.sample_manifest["split"] == "train", "label_idx"
    ].to_numpy(np.int64)
    criterion = build_cross_entropy(
        device=torch_device,
        train_labels=train_labels,
        num_classes=num_classes,
        use_class_weights=config.training.use_class_weights,
    )
    optimizer = build_adam_optimizer(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    history, best_state, best_epoch, best_val_ba = fit_cnn(
        model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        device=torch_device,
        num_epochs=config.training.num_epochs,
        patience=config.training.early_stopping_patience,
        max_train_batches=config.training.max_train_batches,
        grad_clip_norm=config.training.grad_clip_norm,
        verbose=config.evaluation.verbose,
    )
    model.load_state_dict(best_state, strict=True)

    classification_splits = evaluate_cnn_splits(
        model,
        {
            "train": loaders["train_eval"],
            "val": loaders["val"],
            "test": loaders["test"],
        },
        criterion,
        device=torch_device,
    )

    embeddings = extract_embedding_splits(
        model,
        {
            "train": loaders["train_eval"],
            "val": loaders["val"],
            "test": loaders["test"],
        },
        device=torch_device,
    )

    evaluation = evaluate_representation(
        train_bundle=embeddings["train"],
        val_bundle=embeddings["val"],
        test_bundle=embeddings["test"],
        num_classes=num_classes,
        device=torch_device,
        idx_to_class=split.idx_to_class,
        config=config.evaluation,
    )

    checkpoint = build_experiment_checkpoint(
        model=model,
        class_to_idx=split.class_to_idx,
        normalization=normalization,
        train_users=split.train_users,
        val_users=split.val_users,
        test_users=split.test_users,
        best_epoch=best_epoch,
        best_val_balanced_accuracy=best_val_ba,
        experiment_config=config,
        producer_metadata=data.producer_metadata,
        extra={
            "split_reference_checkpoint": (
                None if reference_path is None else str(reference_path)
            ),
            "split_reference_only": reference is not None,
            "reference_identity": reference_identity,
            "excluded_users": list(cohort.excluded_users),
            "eligible_users": list(cohort.eligible_users),
            "cohort_source": cohort.source,
            "experiment_protocol": (
                "reconstruction_train_to_reconstruction_test"
            ),
        },
    )

    artifact_paths: dict[str, Path] = {}
    checkpoint_path = save_checkpoint(
        output_dir / "best_acceleration_cnn.pt",
        checkpoint,
    )
    artifact_paths["checkpoint"] = checkpoint_path

    history_path = output_dir / "training_history.csv"
    history.to_csv(history_path, index=False)
    artifact_paths["training_history"] = history_path

    classification_path = output_dir / "classification_splits.csv"
    classification_splits.to_csv(classification_path)
    artifact_paths["classification_splits"] = classification_path

    manifest_path = output_dir / "sample_manifest.csv"
    split.sample_manifest.to_csv(manifest_path, index=False)
    artifact_paths["sample_manifest"] = manifest_path

    split_summary_path = output_dir / "split_summary.csv"
    split.split_summary.to_csv(split_summary_path)
    artifact_paths["split_summary"] = split_summary_path

    label_counts_path = output_dir / "label_split_counts.csv"
    split.label_split_counts.to_csv(label_counts_path)
    artifact_paths["label_split_counts"] = label_counts_path

    for split_name, bundle in embeddings.items():
        path = save_embedding_bundle(
            output_dir / f"embeddings_{split_name}.npz",
            bundle,
        )
        artifact_paths[f"embeddings_{split_name}"] = path

    evaluation_paths = save_representation_evaluation(
        output_dir,
        evaluation,
    )
    artifact_paths.update(
        {f"evaluation_{name}": path for name, path in evaluation_paths.items()}
    )

    provenance_path = save_provenance(
        output_dir / "provenance.json",
        {
            "experiment": "C_reconstruction",
            "protocol": {
                "train_source": "reconstruction",
                "val_source": "reconstruction",
                "test_source": "reconstruction",
                "normalization_fitted_on": normalization.fitted_on,
                "model_initialization": "from_scratch",
                "reference_checkpoint_role": (
                    "split_and_class_mapping_only" if reference is not None else None
                ),
            },
            "repository_root": repository_root,
            "dataset_root_argument": str(root),
            "resolved_padded_root": data.padded_root,
            "output_dir": output_dir,
            "reference_checkpoint": reference_path,
            "split_reference_only": reference is not None,
            "reference_identity": reference_identity,
            "device": torch_device,
            "config": config.to_dict(),
            "normalization": normalization.to_dict(),
            "excluded_users": cohort.excluded_users,
            "eligible_users": cohort.eligible_users,
            "train_users": split.train_users,
            "val_users": split.val_users,
            "test_users": split.test_users,
            "class_to_idx": split.class_to_idx,
            "cohort_source": cohort.source,
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_ba,
            "producer_metadata": data.producer_metadata.to_dict(),
        },
    )
    artifact_paths["provenance"] = provenance_path

    return ExperimentCRunResult(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        summary=evaluation.summary,
        classification_splits=classification_splits,
        training_history=history,
        split_summary=split.split_summary,
        label_split_counts=split.label_split_counts,
        normalization=normalization,
        artifact_paths=artifact_paths,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment C: train a new acceleration CNN on reconstructed "
            "acceleration and evaluate on reconstructed held-out-user test data."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=DEFAULT_REFERENCE_CHECKPOINT,
        help=(
            "Experiment A checkpoint used only for train/val/test users and "
            "class_to_idx."
        ),
    )
    parser.add_argument(
        "--allow-new-split",
        action="store_true",
        help="Allow a newly generated user split if the A checkpoint is unavailable.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    repository_root = args.repository_root.expanduser().resolve()

    config = experiment_c_config(
        output_dir=args.output_dir,
        random_seed=12345,
    )
    config = replace(
        config,
        use_gpu=not args.cpu,
        loader=replace(
            config.loader,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ),
        training=replace(
            config.training,
            num_epochs=args.epochs,
            early_stopping_patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_class_weights=args.use_class_weights,
            grad_clip_norm=args.grad_clip_norm,
            max_train_batches=args.max_train_batches,
        ),
        evaluation=replace(
            config.evaluation,
            verbose=not args.quiet,
        ),
    )
    config.validate()

    result = run_experiment_c(
        root=args.root,
        repository_root=repository_root,
        output_dir=args.output_dir,
        reference_checkpoint=args.reference_checkpoint,
        config=config,
        device="cpu" if args.cpu else None,
        allow_new_split=args.allow_new_split,
    )

    print("\nExperiment C complete")
    print(f"Output directory: {result.output_dir}")
    print(f"Checkpoint:       {result.checkpoint_path}")
    print("\nSummary:")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
