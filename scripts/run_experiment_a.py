from __future__ import annotations

"""Run Experiment A: raw-acceleration CNN baseline on held-out users.

Protocol
--------
1. Load canonical padded SpikeIMU packages.
2. Build a user-disjoint train/validation/test split.
3. Fit acceleration normalization on RAW TRAIN valid samples only.
4. Train a new MaskAwareAccelerationCNN from scratch on raw acceleration
   channels 15:18.
5. Select the best epoch using validation balanced accuracy.
6. Evaluate raw held-out test classification and the shared representation
   protocol (kNN, retrieval, prototype, linear probe, geometry, clustering).
7. Save a self-describing Experiment A checkpoint and standardized artifacts.

This runner is intended to replace the implementation-heavy Experiment A
notebook. The notebook should become a thin configuration/presentation layer.
"""

import argparse
import random
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
    build_adam_optimizer,
    build_cross_entropy,
    build_experiment_checkpoint,
    build_split_loaders,
    evaluate_cnn_splits,
    evaluate_representation,
    experiment_a_config,
    extract_embedding_splits,
    fit_acceleration_normalization,
    fit_cnn,
    load_acceleration_data,
    prepare_user_disjoint_splits,
    save_checkpoint,
    save_embedding_bundle,
    save_provenance,
    save_representation_evaluation,
)


DEFAULT_DATASET_ROOT = Path(
    "outputs/action0_rectified/low-pass/aligned-board-events"
)
DEFAULT_OUTPUT_DIR = Path(
    "notebooks/artifacts/acceleration_cnn_representation"
)


@dataclass(frozen=True)
class ExperimentARunResult:
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
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False"
            )
        return device
    return torch.device(
        "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
    )


def _prepare_split(
    *,
    sample_manifest: pd.DataFrame,
    config: ExperimentConfig,
):
    split_config = config.split
    return prepare_user_disjoint_splits(
        sample_manifest,
        train_fraction=split_config.train_fraction,
        val_fraction=split_config.val_fraction,
        seed=config.random_seed,
        explicit_train_users=split_config.explicit_train_users,
        explicit_val_users=split_config.explicit_val_users,
        explicit_test_users=split_config.explicit_test_users,
        require_all_users_assigned=split_config.require_all_users_assigned,
        require_all_labels_in_all_splits=(
            split_config.require_all_labels_in_all_splits
        ),
    )


def run_experiment_a(
    *,
    root: str | Path = DEFAULT_DATASET_ROOT,
    repository_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: ExperimentConfig | None = None,
    device: str | None = None,
) -> ExperimentARunResult:
    """Execute Experiment A end-to-end and save standardized artifacts.

    Experiment A is the authoritative raw-acceleration baseline. Its checkpoint
    supplies the user split, class mapping, model state, and raw-train
    normalization used by strict Experiment B; Experiments C/D should reuse its
    split/class mapping for cross-condition comparability.
    """
    repository_root = Path(repository_root).expanduser().resolve()
    output_dir = _resolve_path(output_dir, repository_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = experiment_a_config(
            output_dir=output_dir,
            random_seed=12345,
        )
    else:
        config = replace(config, output_dir=output_dir)
        config.validate()

    if not config.training_enabled:
        raise ValueError("Experiment A requires training_enabled=True")
    if (
        config.train_source != "raw"
        or config.val_source != "raw"
        or config.test_source != "raw"
        or config.normalization_source != "raw"
    ):
        raise ValueError(
            "Experiment A must use raw acceleration for train/val/test and "
            "raw-train normalization"
        )

    _set_global_seed(config.random_seed)
    torch_device = _select_device(config, device)

    data = load_acceleration_data(
        root,
        repository_root=repository_root,
        require_reconstruction=False,
    )
    split = _prepare_split(
        sample_manifest=data.sample_manifest,
        config=config,
    )

    normalization = fit_acceleration_normalization(
        data.packages,
        split.sample_manifest,
        split="train",
        source="raw",
        chunk_segments=config.normalization_chunk_segments,
    )
    if normalization.fitted_on != "raw:train":
        raise AssertionError(
            "Experiment A normalization must be fitted on raw:train; "
            f"got {normalization.fitted_on!r}"
        )

    loaders = build_split_loaders(
        data.packages,
        split.sample_manifest,
        train_source="raw",
        val_source="raw",
        test_source="raw",
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
            "experiment_protocol": "raw_train_to_raw_test",
            "baseline_role": "authoritative_experiment_a_reference",
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
            "experiment": "A_raw",
            "protocol": {
                "train_source": "raw",
                "val_source": "raw",
                "test_source": "raw",
                "normalization_fitted_on": normalization.fitted_on,
                "model_initialization": "from_scratch",
                "checkpoint_role": (
                    "authoritative split/class/normalization/model reference "
                    "for Experiment B"
                ),
            },
            "repository_root": repository_root,
            "dataset_root_argument": str(root),
            "resolved_padded_root": data.padded_root,
            "output_dir": output_dir,
            "device": torch_device,
            "config": config.to_dict(),
            "normalization": normalization.to_dict(),
            "train_users": split.train_users,
            "val_users": split.val_users,
            "test_users": split.test_users,
            "class_to_idx": split.class_to_idx,
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_ba,
            "producer_metadata": data.producer_metadata.to_dict(),
        },
    )
    artifact_paths["provenance"] = provenance_path

    return ExperimentARunResult(
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
            "Experiment A: train the acceleration CNN on raw paddedSpikeIMU "
            "channels 15:18 and evaluate on held-out raw-user test data."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=12345)
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

    config = experiment_a_config(
        output_dir=args.output_dir,
        random_seed=args.seed,
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
            random_seed=args.seed,
            verbose=not args.quiet,
        ),
    )
    config.validate()

    result = run_experiment_a(
        root=args.root,
        repository_root=repository_root,
        output_dir=args.output_dir,
        config=config,
        device="cpu" if args.cpu else None,
    )

    print(f"Output directory: {result.output_dir}")
    print(f"Checkpoint:       {result.checkpoint_path}")
    print("Summary:")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
