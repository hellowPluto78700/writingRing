from __future__ import annotations

"""Checkpoint and artifact I/O for acceleration CNN experiments.

The checkpoint schema keeps the keys used by the existing raw baseline notebook
(``model_state_dict``, ``model_config``, ``class_to_idx``, normalization and user
splits), while adding experiment-source/provenance fields for A/B/C/D.
"""

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from .config import ExperimentConfig
    from .datasets import (
        ACCELERATION_CHANNEL_NAMES,
        ACCELERATION_SLICE,
        NormalizationStats,
    )
    from .embedding import EmbeddingBundle
    from .evaluation import (
        PairedPreservationResult,
        RepresentationEvaluationResult,
    )
except ImportError:  # direct-module use in notebooks/tests
    from config import ExperimentConfig
    from datasets import (
        ACCELERATION_CHANNEL_NAMES,
        ACCELERATION_SLICE,
        NormalizationStats,
    )
    from embedding import EmbeddingBundle
    from evaluation import (
        PairedPreservationResult,
        RepresentationEvaluationResult,
    )


CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_ARTIFACT_TYPE = "acceleration_cnn_experiment_checkpoint"
LEGACY_CHECKPOINT_ARTIFACT_TYPE = "acceleration_cnn_representation_checkpoint"

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_ARTIFACT_TYPE",
    "json_safe",
    "build_experiment_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    "restore_model_from_checkpoint",
    "normalization_from_checkpoint",
    "save_embedding_bundle",
    "load_embedding_bundle",
    "save_representation_evaluation",
    "save_paired_preservation",
    "save_provenance",
    "load_summary_csv",
]


def json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.pt", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.save(value, temp_path)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_to_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.csv", dir=path.parent, text=True
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        frame.to_csv(temp_path, index=index)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _normalization_fields(
    normalization: NormalizationStats | Mapping[str, object],
) -> tuple[list[float], list[float], int, str]:
    if isinstance(normalization, NormalizationStats):
        return (
            normalization.mean.tolist(),
            normalization.std.tolist(),
            int(normalization.valid_time_points),
            normalization.fitted_on,
        )

    mean = np.asarray(normalization["mean"], dtype=np.float64)
    std = np.asarray(normalization["std"], dtype=np.float64)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Normalization mean/std must each have shape (3,)")
    valid_time_points = int(normalization.get("valid_time_points", 0))
    fitted_on = str(normalization.get("fitted_on", "unknown"))
    return mean.tolist(), std.tolist(), valid_time_points, fitted_on


def build_experiment_checkpoint(
    *,
    model: nn.Module,
    class_to_idx: Mapping[str, int],
    normalization: NormalizationStats | Mapping[str, object],
    train_users: Sequence[str],
    val_users: Sequence[str],
    test_users: Sequence[str],
    best_epoch: int,
    best_val_balanced_accuracy: float,
    experiment_config: ExperimentConfig | Mapping[str, object] | None = None,
    producer_metadata: object | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a checkpoint compatible with both the old baseline and new runs."""
    mean, std, valid_time_points, fitted_on = _normalization_fields(normalization)
    class_to_idx = {str(key): int(value) for key, value in class_to_idx.items()}
    if sorted(class_to_idx.values()) != list(range(len(class_to_idx))):
        raise ValueError("class_to_idx values must be contiguous 0..C-1")

    if hasattr(model, "architecture_config"):
        model_config = model.architecture_config()
    else:
        model_config = {"class_name": type(model).__name__}

    if experiment_config is None:
        config_dict: dict[str, object] = {}
    elif isinstance(experiment_config, ExperimentConfig):
        experiment_config.validate()
        config_dict = experiment_config.to_dict()
    else:
        config_dict = dict(experiment_config)

    checkpoint: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": CHECKPOINT_ARTIFACT_TYPE,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": model_config,
        "class_to_idx": class_to_idx,
        "acceleration_slice": [ACCELERATION_SLICE.start, ACCELERATION_SLICE.stop],
        "acceleration_channel_names": list(ACCELERATION_CHANNEL_NAMES),
        "normalization_mean": mean,
        "normalization_std": std,
        "normalization_valid_time_points": valid_time_points,
        "normalization_fitted_on": fitted_on,
        "train_users": [str(value) for value in train_users],
        "val_users": [str(value) for value in val_users],
        "test_users": [str(value) for value in test_users],
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy": float(best_val_balanced_accuracy),
        "experiment_config": config_dict,
    }

    for key in ("name", "train_source", "val_source", "test_source", "random_seed"):
        if key in config_dict:
            checkpoint[
                "experiment_name" if key == "name" else key
            ] = config_dict[key]

    if producer_metadata is not None:
        if hasattr(producer_metadata, "to_dict"):
            checkpoint["producer_metadata"] = producer_metadata.to_dict()
        elif is_dataclass(producer_metadata):
            checkpoint["producer_metadata"] = asdict(producer_metadata)
        elif isinstance(producer_metadata, Mapping):
            checkpoint["producer_metadata"] = dict(producer_metadata)
        else:
            checkpoint["producer_metadata"] = json_safe(producer_metadata)

    if extra:
        reserved = set(checkpoint)
        overlap = reserved.intersection(extra)
        if overlap:
            raise KeyError(f"extra attempts to overwrite checkpoint keys: {sorted(overlap)}")
        checkpoint.update(dict(extra))

    return checkpoint


def save_checkpoint(path: str | Path, checkpoint: Mapping[str, object]) -> Path:
    path = Path(path)
    _atomic_torch_save(path, dict(checkpoint))
    return path


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")

    required = {
        "model_state_dict",
        "model_config",
        "class_to_idx",
        "normalization_mean",
        "normalization_std",
        "train_users",
        "val_users",
        "test_users",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"Checkpoint is missing keys: {missing}")

    artifact_type = checkpoint.get("artifact_type")
    if artifact_type not in {
        None,
        CHECKPOINT_ARTIFACT_TYPE,
        LEGACY_CHECKPOINT_ARTIFACT_TYPE,
    }:
        raise ValueError(f"Unexpected checkpoint artifact_type={artifact_type!r}")

    class_to_idx = {str(k): int(v) for k, v in dict(checkpoint["class_to_idx"]).items()}
    if sorted(class_to_idx.values()) != list(range(len(class_to_idx))):
        raise ValueError("Checkpoint class_to_idx values must be contiguous 0..C-1")
    checkpoint["class_to_idx"] = class_to_idx
    return checkpoint


def restore_model_from_checkpoint(
    model: nn.Module,
    checkpoint: Mapping[str, object],
    *,
    strict: bool = True,
    device: torch.device | None = None,
    freeze: bool = False,
) -> nn.Module:
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if device is not None:
        model.to(device)
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
    return model


def normalization_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> NormalizationStats:
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float64)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float64)
    return NormalizationStats(
        mean=mean,
        std=std,
        valid_time_points=int(checkpoint.get("normalization_valid_time_points", 0)),
        fitted_on=str(checkpoint.get("normalization_fitted_on", "checkpoint/raw_train")),
    )


def save_embedding_bundle(path: str | Path, bundle: EmbeddingBundle) -> Path:
    path = Path(path)
    _atomic_save_npz(
        path,
        h=np.asarray(bundle["h"]),
        z=np.asarray(bundle["z"]),
        y=np.asarray(bundle["y"]),
        cnn_pred=np.asarray(bundle["cnn_pred"]),
        logits=np.asarray(bundle["logits"]),
        sample_id=np.asarray(bundle["sample_id"]).astype(str),
    )
    return path


def load_embedding_bundle(path: str | Path) -> EmbeddingBundle:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        required = {"h", "z", "y", "cnn_pred", "logits", "sample_id"}
        missing = sorted(required.difference(values.files))
        if missing:
            raise KeyError(f"Embedding archive is missing arrays: {missing}")
        return EmbeddingBundle(
            h=values["h"],
            z=values["z"],
            y=values["y"].astype(np.int64, copy=False),
            cnn_pred=values["cnn_pred"].astype(np.int64, copy=False),
            logits=values["logits"],
            sample_id=values["sample_id"].astype(str),
        )


def save_representation_evaluation(
    output_dir: str | Path,
    result: RepresentationEvaluationResult,
    *,
    prefix: str = "",
) -> dict[str, Path]:
    """Save all reusable tables/arrays from ``evaluate_representation``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""

    paths = {
        "summary": output_dir / f"{stem}summary.csv",
        "knn_validation": output_dir / f"{stem}knn_validation.csv",
        "same_label": output_dir / f"{stem}same_label_at_k.csv",
        "retrieval": output_dir / f"{stem}retrieval_summary.csv",
        "geometry": output_dir / f"{stem}geometry_summary.csv",
        "intra_by_class": output_dir / f"{stem}intra_by_class.csv",
        "linear_probe_history": output_dir / f"{stem}linear_probe_history.csv",
        "unsupervised": output_dir / f"{stem}unsupervised_summary.csv",
        "arrays": output_dir / f"{stem}evaluation_arrays.npz",
        "linear_probe": output_dir / f"{stem}linear_probe.pt",
    }

    _atomic_to_csv(result.summary, paths["summary"], index=False)
    _atomic_to_csv(result.knn_validation_table, paths["knn_validation"], index=False)
    _atomic_to_csv(result.same_label_table, paths["same_label"], index=False)
    _atomic_to_csv(result.retrieval_summary, paths["retrieval"], index=False)
    _atomic_to_csv(result.geometry_summary, paths["geometry"], index=False)
    _atomic_to_csv(result.intra_by_class, paths["intra_by_class"], index=False)
    _atomic_to_csv(result.linear_probe_history, paths["linear_probe_history"], index=False)
    _atomic_to_csv(result.unsupervised_summary, paths["unsupervised"], index=False)

    _atomic_save_npz(
        paths["arrays"],
        knn_prediction=result.knn_prediction,
        average_precision=result.average_precision,
        prototype_prediction=result.prototype_prediction,
        kmeans_assignment=result.kmeans_assignment,
        kmeans_centers=result.kmeans_centers,
        silhouette_values=result.silhouette_values,
        silhouette_indices=result.silhouette_indices,
        linear_probe_feature_mean=result.linear_probe_feature_mean,
        linear_probe_feature_std=result.linear_probe_feature_std,
    )
    _atomic_torch_save(
        paths["linear_probe"],
        {
            "state_dict": result.linear_probe_state_dict,
            "best_epoch": result.linear_probe_best_epoch,
            "best_val_balanced_accuracy": result.linear_probe_best_val_balanced_accuracy,
            "feature_mean": result.linear_probe_feature_mean,
            "feature_std": result.linear_probe_feature_std,
        },
    )
    return paths


def save_paired_preservation(
    output_dir: str | Path,
    result: PairedPreservationResult,
    *,
    prefix: str = "paired",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / f"{prefix}_summary.csv",
        "transitions": output_dir / f"{prefix}_transitions.csv",
        "per_class": output_dir / f"{prefix}_per_class.csv",
        "arrays": output_dir / f"{prefix}_arrays.npz",
    }
    _atomic_to_csv(result.summary, paths["summary"], index=False)
    _atomic_to_csv(result.transitions, paths["transitions"], index=False)
    _atomic_to_csv(result.per_class, paths["per_class"], index=False)
    _atomic_save_npz(
        paths["arrays"],
        cosine_similarity=result.cosine_similarity,
        cosine_distance=result.cosine_distance,
    )
    return paths


def save_provenance(path: str | Path, payload: Mapping[str, object]) -> Path:
    path = Path(path)
    text = json.dumps(json_safe(dict(payload)), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(path, text)
    return path


def load_summary_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)
