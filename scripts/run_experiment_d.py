from __future__ import annotations

"""Experiment D runner: mixed raw+reconstruction training with dual-domain tests.

Protocol
--------
1. Recover the exact train/validation/test users and class mapping from the
   Experiment A checkpoint (metadata only).
2. Fit normalization on the mixed training domain: every training segment is
   represented once as raw acceleration and once as reconstructed acceleration.
3. Train a new MaskAwareAccelerationCNN from scratch on the mixed training set.
4. Select the best epoch on mixed validation balanced accuracy.
5. Evaluate the same frozen best model on both held-out-user domains:
      - raw test
      - reconstruction test
6. Reuse the same mixed-train / mixed-validation embedding references for kNN,
   prototypes, and linear-probe training in both test evaluations.
7. Save paired raw/reconstruction preservation diagnostics for the same test
   samples.

The Experiment A checkpoint is never used for model weights or normalization.
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
    compare_representation_results,
    evaluate_cnn_splits,
    evaluate_paired_preservation,
    evaluate_representation,
    experiment_d_config,
    extract_embedding_splits,
    fit_acceleration_normalization,
    fit_cnn,
    load_acceleration_data,
    load_checkpoint,
    prepare_user_disjoint_splits,
    save_checkpoint,
    save_embedding_bundle,
    save_paired_preservation,
    save_provenance,
    save_representation_evaluation,
)


DEFAULT_DATASET_ROOT = Path(
    "outputs/action0_rectified/low-pass/aligned-board-events"
)
DEFAULT_REFERENCE_CHECKPOINT = Path(
    "notebooks/artifacts/acceleration_cnn_representation/"
    "best_acceleration_cnn.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "notebooks/artifacts/experiment_D_mixed_training"
)


@dataclass(frozen=True)
class ExperimentDRunResult:
    """Lightweight result returned to notebooks and programmatic callers."""

    output_dir: Path
    checkpoint_path: Path
    raw_summary: pd.DataFrame
    reconstruction_summary: pd.DataFrame
    domain_comparison: pd.DataFrame
    paired_summary: pd.DataFrame
    paired_transitions: pd.DataFrame
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
    reference_checkpoint: Mapping[str, object] | None,
):
    split_config = config.split

    if reference_checkpoint is not None:
        return prepare_user_disjoint_splits(
            sample_manifest,
            seed=config.random_seed,
            explicit_train_users=reference_checkpoint["train_users"],
            explicit_val_users=reference_checkpoint["val_users"],
            explicit_test_users=reference_checkpoint["test_users"],
            require_all_users_assigned=split_config.require_all_users_assigned,
            require_all_labels_in_all_splits=(
                split_config.require_all_labels_in_all_splits
            ),
            class_to_idx=dict(reference_checkpoint["class_to_idx"]),
        )

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


def run_experiment_d(
    *,
    root: str | Path = DEFAULT_DATASET_ROOT,
    repository_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    reference_checkpoint: str | Path | None = DEFAULT_REFERENCE_CHECKPOINT,
    config: ExperimentConfig | None = None,
    device: str | None = None,
    allow_new_split: bool = False,
) -> ExperimentDRunResult:
    """Execute Experiment D end-to-end and save dual-domain artifacts.

    ``config.test_source`` is not used to choose a single test domain. Experiment
    D always evaluates both raw and reconstruction test data using the same best
    mixed-trained model. The field remains part of ExperimentConfig for
    compatibility with ``experiment_d_config`` and is stored as the nominal
    configuration value in the checkpoint/provenance.

    Parameters
    ----------
    reference_checkpoint:
        Experiment A checkpoint used only to recover train/val/test users and
        ``class_to_idx``. No A weights or normalization are reused.
    allow_new_split:
        Safety valve for standalone use. Keep False for strict A/B/C/D
        comparability.
    """
    repository_root = Path(repository_root).expanduser().resolve()
    output_dir = _resolve_path(output_dir, repository_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = experiment_d_config(
            test_source="raw",
            output_dir=output_dir,
            random_seed=12345,
        )
    else:
        config = replace(config, output_dir=output_dir)
        config.validate()

    if not config.training_enabled:
        raise ValueError("Experiment D requires training_enabled=True")
    if config.train_source != "mixed" or config.val_source != "mixed":
        raise ValueError("Experiment D must use mixed train and mixed validation")
    if config.normalization_source != "mixed":
        raise ValueError("Experiment D normalization must be fitted on mixed train")
    if config.test_source not in {"raw", "reconstruction"}:
        raise ValueError("Experiment D nominal test_source must be raw or reconstruction")

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
            "Experiment D should reuse the A split. Supply reference_checkpoint, "
            "or set allow_new_split=True only if a new split is intentional."
        )

    data = load_acceleration_data(
        root,
        repository_root=repository_root,
        require_reconstruction=True,
    )
    split = _prepare_split(
        sample_manifest=data.sample_manifest,
        config=config,
        reference_checkpoint=reference,
    )

    normalization = fit_acceleration_normalization(
        data.packages,
        split.sample_manifest,
        split="train",
        source="mixed",
        chunk_segments=config.normalization_chunk_segments,
    )
    if normalization.fitted_on != "mixed:train":
        raise AssertionError(
            "Experiment D normalization must be fitted on mixed:train; "
            f"got {normalization.fitted_on!r}"
        )

    # Build one loader set for each test domain. Train and validation are mixed
    # in both; only the first set is used for optimization so shuffle state is
    # unambiguous.
    raw_loaders = build_split_loaders(
        data.packages,
        split.sample_manifest,
        train_source="mixed",
        val_source="mixed",
        test_source="raw",
        normalization=normalization,
        batch_size=config.loader.batch_size,
        num_workers=config.loader.num_workers,
        use_gpu=(torch_device.type == "cuda"),
        seed=config.random_seed,
        pin_memory=config.loader.pin_memory,
        persistent_workers=config.loader.persistent_workers,
    )
    reconstruction_loaders = build_split_loaders(
        data.packages,
        split.sample_manifest,
        train_source="mixed",
        val_source="mixed",
        test_source="reconstruction",
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

    # Mixed duplicates every training example once per domain. Class-frequency
    # ratios are unchanged, so weights computed from the one-row-per-segment
    # manifest are identical to weights from the duplicated mixed dataset.
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
        train_loader=raw_loaders["train"],
        val_loader=raw_loaders["val"],
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
            "train_mixed": raw_loaders["train_eval"],
            "val_mixed": raw_loaders["val"],
            "test_raw": raw_loaders["test"],
            "test_reconstruction": reconstruction_loaders["test"],
        },
        criterion,
        device=torch_device,
    )

    embeddings = extract_embedding_splits(
        model,
        {
            "train_mixed": raw_loaders["train_eval"],
            "val_mixed": raw_loaders["val"],
            "test_raw": raw_loaders["test"],
            "test_reconstruction": reconstruction_loaders["test"],
        },
        device=torch_device,
    )

    raw_evaluation = evaluate_representation(
        train_bundle=embeddings["train_mixed"],
        val_bundle=embeddings["val_mixed"],
        test_bundle=embeddings["test_raw"],
        num_classes=num_classes,
        device=torch_device,
        idx_to_class=split.idx_to_class,
        config=config.evaluation,
    )
    reconstruction_evaluation = evaluate_representation(
        train_bundle=embeddings["train_mixed"],
        val_bundle=embeddings["val_mixed"],
        test_bundle=embeddings["test_reconstruction"],
        num_classes=num_classes,
        device=torch_device,
        idx_to_class=split.idx_to_class,
        config=config.evaluation,
    )

    paired = evaluate_paired_preservation(
        embeddings["test_raw"],
        embeddings["test_reconstruction"],
        idx_to_class=split.idx_to_class,
    )

    domain_comparison = compare_representation_results(
        {
            "D_mixed_to_raw": raw_evaluation,
            "D_mixed_to_reconstruction": reconstruction_evaluation,
        }
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
            "experiment_protocol": "mixed_train_dual_domain_test",
            "evaluation_test_sources": ["raw", "reconstruction"],
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

    embedding_names = {
        "train_mixed": "embeddings_train_mixed.npz",
        "val_mixed": "embeddings_val_mixed.npz",
        "test_raw": "embeddings_test_raw.npz",
        "test_reconstruction": "embeddings_test_reconstruction.npz",
    }
    for split_name, filename in embedding_names.items():
        path = save_embedding_bundle(
            output_dir / filename,
            embeddings[split_name],
        )
        artifact_paths[f"embeddings_{split_name}"] = path

    raw_dir = output_dir / "raw_test"
    raw_paths = save_representation_evaluation(
        raw_dir,
        raw_evaluation,
    )
    artifact_paths.update(
        {f"raw_{name}": path for name, path in raw_paths.items()}
    )

    reconstruction_dir = output_dir / "reconstruction_test"
    reconstruction_paths = save_representation_evaluation(
        reconstruction_dir,
        reconstruction_evaluation,
    )
    artifact_paths.update(
        {
            f"reconstruction_{name}": path
            for name, path in reconstruction_paths.items()
        }
    )

    paired_dir = output_dir / "paired_preservation"
    paired_paths = save_paired_preservation(
        paired_dir,
        paired,
        prefix="paired",
    )
    artifact_paths.update(
        {f"paired_{name}": path for name, path in paired_paths.items()}
    )

    comparison_path = output_dir / "domain_comparison.csv"
    domain_comparison.to_csv(comparison_path)
    artifact_paths["domain_comparison"] = comparison_path

    # One-row condition summary is convenient for downstream comparison code.
    condition_summary = pd.concat(
        [
            raw_evaluation.summary.assign(condition="D_mixed_to_raw"),
            reconstruction_evaluation.summary.assign(
                condition="D_mixed_to_reconstruction"
            ),
        ],
        ignore_index=True,
    )
    condition_cols = ["condition"] + [
        column for column in condition_summary.columns if column != "condition"
    ]
    condition_summary = condition_summary[condition_cols]
    condition_summary_path = output_dir / "condition_summaries.csv"
    condition_summary.to_csv(condition_summary_path, index=False)
    artifact_paths["condition_summaries"] = condition_summary_path

    provenance_path = save_provenance(
        output_dir / "provenance.json",
        {
            "experiment": "D_mixed_training",
            "protocol": {
                "train_source": "mixed",
                "val_source": "mixed",
                "test_sources": ["raw", "reconstruction"],
                "normalization_fitted_on": normalization.fitted_on,
                "model_initialization": "from_scratch",
                "same_model_for_both_test_domains": True,
                "reference_checkpoint_role": (
                    "split_and_class_mapping_only" if reference is not None else None
                ),
                "evaluation_reference_domain": "mixed_train_and_mixed_validation",
            },
            "repository_root": repository_root,
            "dataset_root_argument": str(root),
            "resolved_padded_root": data.padded_root,
            "output_dir": output_dir,
            "reference_checkpoint": reference_path,
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

    return ExperimentDRunResult(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        raw_summary=raw_evaluation.summary,
        reconstruction_summary=reconstruction_evaluation.summary,
        domain_comparison=domain_comparison,
        paired_summary=paired.summary,
        paired_transitions=paired.transitions,
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
            "Experiment D: train one new acceleration CNN on duplicated raw + "
            "reconstruction samples, then evaluate the same model on raw and "
            "reconstructed held-out-user test data."
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
        help=(
            "Allow a newly generated user split if the A checkpoint is unavailable."
        ),
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

    config = experiment_d_config(
        test_source="raw",
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

    result = run_experiment_d(
        root=args.root,
        repository_root=repository_root,
        output_dir=args.output_dir,
        reference_checkpoint=args.reference_checkpoint,
        config=config,
        device="cpu" if args.cpu else None,
        allow_new_split=args.allow_new_split,
    )

    print("\nExperiment D complete")
    print(f"Output directory: {result.output_dir}")
    print(f"Checkpoint:       {result.checkpoint_path}")
    print("\nRaw test summary:")
    print(result.raw_summary.to_string(index=False))
    print("\nReconstruction test summary:")
    print(result.reconstruction_summary.to_string(index=False))
    print("\nPaired preservation summary:")
    print(result.paired_summary.to_string(index=False))


if __name__ == "__main__":
    main()
