from __future__ import annotations

"""Run Experiment B: reconstructed acceleration through the frozen A CNN.

Protocol
--------
1. Load the authoritative Experiment A checkpoint.
2. Reuse its exact user split, class mapping, model weights, and raw-train
   acceleration normalization.
3. Load canonical padded SpikeIMU packages plus aligned reconstructed
   acceleration artifacts.
4. Restore MaskAwareAccelerationCNN with strict=True and freeze it. No model
   training or fine-tuning is allowed in Experiment B.
5. Build raw train/raw validation/raw test and reconstruction test loaders using
   the checkpoint normalization. No reconstruction-domain normalization is fit.
6. Evaluate two held-out test views with the same frozen feature extractor:
      - raw test: baseline/reference consistency check
      - reconstruction test: Experiment B result
   Retrieval, prototypes, kNN K-selection, and linear probe all remain tied to
   raw train/raw validation embeddings for both evaluations.
7. Run paired raw/reconstruction preservation analysis on matched test samples.
8. Save standardized Experiment B artifacts and provenance.

The root ``summary.csv`` is the Experiment B reconstruction-test result. Raw
reference evaluation artifacts are stored below ``raw_reference/``.
"""

import argparse
import hashlib
import random
from collections.abc import Mapping as MappingABC
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
    build_cross_entropy,
    build_split_loaders,
    compare_representation_results,
    evaluate_cnn_splits,
    evaluate_paired_preservation,
    evaluate_representation,
    experiment_b_config,
    extract_embedding_splits,
    load_acceleration_data,
    load_checkpoint,
    normalization_from_checkpoint,
    prepare_user_disjoint_splits,
    restore_model_from_checkpoint,
    save_embedding_bundle,
    save_paired_preservation,
    save_provenance,
    save_representation_evaluation,
)


DEFAULT_DATASET_ROOT = Path(
    "outputs/action0_rectified/low-pass/aligned-board-events"
)
DEFAULT_BASELINE_CHECKPOINT = Path(
    "notebooks/artifacts/acceleration_cnn_representation/"
    "best_acceleration_cnn.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "notebooks/artifacts/experiment_B_reconstruction_frozen_cnn"
)


@dataclass(frozen=True)
class ExperimentBRunResult:
    """Lightweight result returned to notebooks and programmatic callers."""

    output_dir: Path
    baseline_checkpoint_path: Path
    baseline_checkpoint_sha256: str
    summary: pd.DataFrame
    raw_reference_summary: pd.DataFrame
    domain_comparison: pd.DataFrame
    paired_summary: pd.DataFrame
    paired_transitions: pd.DataFrame
    classification_splits: pd.DataFrame
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


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def _checkpoint_uses_class_weights(checkpoint: Mapping[str, object]) -> bool:
    config = checkpoint.get("experiment_config")
    if not isinstance(config, MappingABC):
        return False
    training = config.get("training")
    if not isinstance(training, MappingABC):
        return False
    return bool(training.get("use_class_weights", False))


def _prepare_split(
    *,
    sample_manifest: pd.DataFrame,
    config: ExperimentConfig,
    checkpoint: Mapping[str, object],
):
    split_config = config.split
    return prepare_user_disjoint_splits(
        sample_manifest,
        seed=config.random_seed,
        explicit_train_users=checkpoint["train_users"],
        explicit_val_users=checkpoint["val_users"],
        explicit_test_users=checkpoint["test_users"],
        require_all_users_assigned=split_config.require_all_users_assigned,
        require_all_labels_in_all_splits=(
            split_config.require_all_labels_in_all_splits
        ),
        class_to_idx=dict(checkpoint["class_to_idx"]),
    )


def run_experiment_b(
    *,
    root: str | Path = DEFAULT_DATASET_ROOT,
    repository_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
    config: ExperimentConfig | None = None,
    device: str | None = None,
) -> ExperimentBRunResult:
    """Execute strict Experiment B and save standardized artifacts.

    Experiment B is evaluation-only. The Experiment A checkpoint is
    authoritative for:

    - model weights,
    - class_to_idx,
    - train/validation/test users, and
    - raw-training normalization.

    Reconstruction data is never used for CNN training, normalization fitting,
    kNN K selection, prototype construction, or linear-probe training.
    """
    repository_root = Path(repository_root).expanduser().resolve()
    output_dir = _resolve_path(output_dir, repository_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = _resolve_path(baseline_checkpoint, repository_root)
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Experiment A checkpoint was not found: {baseline_path}"
        )
    baseline_sha256 = _sha256_file(baseline_path)

    if config is None:
        config = experiment_b_config(
            baseline_checkpoint=baseline_path,
            output_dir=output_dir,
            random_seed=12345,
        )
    else:
        config = replace(
            config,
            baseline_checkpoint=baseline_path,
            output_dir=output_dir,
        )
        config.validate()

    if config.training_enabled:
        raise ValueError("Experiment B requires training_enabled=False")
    if (
        config.train_source != "raw"
        or config.val_source != "raw"
        or config.test_source != "reconstruction"
        or config.normalization_source != "checkpoint"
    ):
        raise ValueError(
            "Experiment B must use raw train/validation references, "
            "reconstruction test queries, and checkpoint normalization"
        )

    _set_global_seed(config.random_seed)
    torch_device = _select_device(config, device)

    checkpoint = load_checkpoint(baseline_path, map_location="cpu")
    normalization = normalization_from_checkpoint(checkpoint)

    data = load_acceleration_data(
        root,
        repository_root=repository_root,
        require_reconstruction=True,
    )
    split = _prepare_split(
        sample_manifest=data.sample_manifest,
        config=config,
        checkpoint=checkpoint,
    )

    # Build matched loader sets. Train/validation are raw in both; only the test
    # domain differs. No fitting or adaptation is performed here.
    raw_loaders = build_split_loaders(
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
    reconstruction_loaders = build_split_loaders(
        data.packages,
        split.sample_manifest,
        train_source="raw",
        val_source="raw",
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
    restore_model_from_checkpoint(
        model,
        checkpoint,
        strict=True,
        device=torch_device,
        freeze=True,
    )

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Experiment B model must remain frozen")
    if model.training:
        raise AssertionError("Experiment B frozen model must be in eval mode")

    train_labels = split.sample_manifest.loc[
        split.sample_manifest["split"] == "train", "label_idx"
    ].to_numpy(np.int64)
    criterion = build_cross_entropy(
        device=torch_device,
        train_labels=train_labels,
        num_classes=num_classes,
        use_class_weights=_checkpoint_uses_class_weights(checkpoint),
    )

    classification_splits = evaluate_cnn_splits(
        model,
        {
            "train_raw": raw_loaders["train_eval"],
            "val_raw": raw_loaders["val"],
            "test_raw": raw_loaders["test"],
            "test_reconstruction": reconstruction_loaders["test"],
        },
        criterion,
        device=torch_device,
    )

    embeddings = extract_embedding_splits(
        model,
        {
            "train_raw": raw_loaders["train_eval"],
            "val_raw": raw_loaders["val"],
            "test_raw": raw_loaders["test"],
            "test_reconstruction": reconstruction_loaders["test"],
        },
        device=torch_device,
    )

    raw_reference_evaluation = evaluate_representation(
        train_bundle=embeddings["train_raw"],
        val_bundle=embeddings["val_raw"],
        test_bundle=embeddings["test_raw"],
        num_classes=num_classes,
        device=torch_device,
        idx_to_class=split.idx_to_class,
        config=config.evaluation,
    )
    reconstruction_evaluation = evaluate_representation(
        train_bundle=embeddings["train_raw"],
        val_bundle=embeddings["val_raw"],
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
            "A_raw_reference": raw_reference_evaluation,
            "B_raw_to_reconstruction": reconstruction_evaluation,
        }
    )

    artifact_paths: dict[str, Path] = {}

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
        "train_raw": "embeddings_train_raw.npz",
        "val_raw": "embeddings_val_raw.npz",
        "test_raw": "embeddings_test_raw.npz",
        "test_reconstruction": "embeddings_test_reconstruction.npz",
    }
    for split_name, filename in embedding_names.items():
        path = save_embedding_bundle(output_dir / filename, embeddings[split_name])
        artifact_paths[f"embeddings_{split_name}"] = path

    # Root evaluation artifacts are the actual Experiment B result.
    reconstruction_paths = save_representation_evaluation(
        output_dir,
        reconstruction_evaluation,
    )
    artifact_paths.update(
        {
            f"reconstruction_{name}": path
            for name, path in reconstruction_paths.items()
        }
    )

    raw_reference_dir = output_dir / "raw_reference"
    raw_reference_paths = save_representation_evaluation(
        raw_reference_dir,
        raw_reference_evaluation,
    )
    artifact_paths.update(
        {f"raw_reference_{name}": path for name, path in raw_reference_paths.items()}
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

    provenance_path = save_provenance(
        output_dir / "provenance.json",
        {
            "experiment": "B_frozen_raw_to_reconstruction",
            "protocol": {
                "training_enabled": False,
                "train_reference_source": "raw",
                "validation_reference_source": "raw",
                "test_query_source": "reconstruction",
                "normalization_source": "experiment_a_checkpoint",
                "model_weights_source": "experiment_a_checkpoint",
                "model_frozen": True,
                "checkpoint_restore_strict": True,
                "raw_reference_test_also_evaluated": True,
                "paired_raw_reconstruction_test_analysis": True,
            },
            "repository_root": repository_root,
            "dataset_root_argument": str(root),
            "resolved_padded_root": data.padded_root,
            "output_dir": output_dir,
            "baseline_checkpoint": baseline_path,
            "baseline_checkpoint_sha256": baseline_sha256,
            "baseline_checkpoint_artifact_type": checkpoint.get("artifact_type"),
            "baseline_checkpoint_schema_version": checkpoint.get("schema_version"),
            "baseline_model_config": checkpoint.get("model_config"),
            "device": torch_device,
            "config": config.to_dict(),
            "normalization": normalization.to_dict(),
            "train_users": split.train_users,
            "val_users": split.val_users,
            "test_users": split.test_users,
            "class_to_idx": split.class_to_idx,
            "producer_metadata": data.producer_metadata.to_dict(),
        },
    )
    artifact_paths["provenance"] = provenance_path

    return ExperimentBRunResult(
        output_dir=output_dir,
        baseline_checkpoint_path=baseline_path,
        baseline_checkpoint_sha256=baseline_sha256,
        summary=reconstruction_evaluation.summary,
        raw_reference_summary=raw_reference_evaluation.summary,
        domain_comparison=domain_comparison,
        paired_summary=paired.summary,
        paired_transitions=paired.transitions,
        classification_splits=classification_splits,
        split_summary=split.split_summary,
        label_split_counts=split.label_split_counts,
        normalization=normalization,
        artifact_paths=artifact_paths,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment B: evaluate reconstructed acceleration with the frozen "
            "Experiment A CNN and raw-training reference representation."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=DEFAULT_BASELINE_CHECKPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    repository_root = args.repository_root.expanduser().resolve()

    config = experiment_b_config(
        baseline_checkpoint=args.baseline_checkpoint,
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
        evaluation=replace(
            config.evaluation,
            random_seed=args.seed,
            verbose=not args.quiet,
        ),
    )
    config.validate()

    result = run_experiment_b(
        root=args.root,
        repository_root=repository_root,
        output_dir=args.output_dir,
        baseline_checkpoint=args.baseline_checkpoint,
        config=config,
        device="cpu" if args.cpu else None,
    )

    print(f"Output directory:       {result.output_dir}")
    print(f"A baseline checkpoint:  {result.baseline_checkpoint_path}")
    print(f"A checkpoint SHA-256:   {result.baseline_checkpoint_sha256}")
    print("Experiment B summary:")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
