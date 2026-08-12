from __future__ import annotations

"""Embedding extraction and feature preprocessing for acceleration CNNs.

The current notebook interface is preserved:

    bundle = extract_embeddings(model, loader, device=device)
    bundle["h"]
    bundle["z"]
    bundle["y"]
    bundle["cnn_pred"]
    bundle["logits"]
    bundle["sample_id"]

``EmbeddingBundle`` is a typed Mapping, so existing dictionary-style notebook
code continues to work while new modules can use attribute access.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


__all__ = [
    "EmbeddingBundle",
    "l2_normalize_embeddings",
    "extract_embeddings",
    "extract_embedding_splits",
    "standardize_feature_splits",
    "apply_feature_standardization",
    "validate_paired_bundles",
    "paired_embedding_cosine",
]


@dataclass(frozen=True)
class EmbeddingBundle(Mapping[str, np.ndarray]):
    """Container for one split of frozen-CNN outputs and representations."""

    h: np.ndarray
    z: np.ndarray
    y: np.ndarray
    cnn_pred: np.ndarray
    logits: np.ndarray
    sample_id: np.ndarray

    _KEYS = ("h", "z", "y", "cnn_pred", "logits", "sample_id")

    def __post_init__(self) -> None:
        h = np.asarray(self.h)
        z = np.asarray(self.z)
        y = np.asarray(self.y)
        pred = np.asarray(self.cnn_pred)
        logits = np.asarray(self.logits)
        sample_id = np.asarray(self.sample_id)

        if h.ndim != 2:
            raise ValueError(f"h must be 2-D, got {h.shape}")
        if z.shape != h.shape:
            raise ValueError(f"z must match h shape; h={h.shape}, z={z.shape}")

        n = len(h)
        if y.shape != (n,):
            raise ValueError(f"y must have shape {(n,)}, got {y.shape}")
        if pred.shape != (n,):
            raise ValueError(
                f"cnn_pred must have shape {(n,)}, got {pred.shape}"
            )
        if logits.ndim != 2 or logits.shape[0] != n:
            raise ValueError(
                f"logits must have shape (N,C), got {logits.shape}"
            )
        if sample_id.shape != (n,):
            raise ValueError(
                f"sample_id must have shape {(n,)}, got {sample_id.shape}"
            )

        if not np.isfinite(h).all():
            raise FloatingPointError("h contains non-finite values")
        if not np.isfinite(z).all():
            raise FloatingPointError("z contains non-finite values")
        if not np.isfinite(logits).all():
            raise FloatingPointError("logits contains non-finite values")

        if n:
            norms = np.linalg.norm(z, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
                raise ValueError("Every row of z must be L2-normalized")

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    @property
    def sample_count(self) -> int:
        return int(len(self.y))

    @property
    def embedding_dim(self) -> int:
        return int(self.h.shape[1])

    @property
    def num_logit_classes(self) -> int:
        return int(self.logits.shape[1])

    def as_dict(self) -> dict[str, np.ndarray]:
        return {key: self[key] for key in self._KEYS}


def l2_normalize_embeddings(
    h: np.ndarray,
    *,
    eps: float = 0.0,
) -> np.ndarray:
    """L2-normalize embedding rows using the baseline notebook convention."""
    values = np.asarray(h, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"h must be 2-D, got {values.shape}")
    if len(values) == 0:
        raise ValueError("Cannot normalize an empty embedding matrix")
    if not np.isfinite(values).all():
        raise FloatingPointError("Embedding matrix contains non-finite values")

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if eps > 0:
        norms = np.maximum(norms, float(eps))
    elif np.any(norms <= 0):
        raise FloatingPointError(
            "Every embedding must have a finite positive L2 norm"
        )

    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise FloatingPointError(
            "Every embedding must have a finite positive L2 norm"
        )

    z = values / norms
    if not np.allclose(
        np.linalg.norm(z, axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise AssertionError("L2-normalized embeddings must have norm 1")
    return z.astype(np.float32, copy=False)


@torch.inference_mode()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader[dict[str, object]],
    *,
    device: torch.device,
) -> EmbeddingBundle:
    """Extract pre-classifier ``h`` and cosine-normalized ``z``.

    Expected batch keys are the same as the current CNN notebook:
    ``x``, ``label``, ``valid_mask``, and ``sample_id``.
    """
    model.eval()

    h_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    sample_ids: list[str] = []

    for batch in loader:
        try:
            x = batch["x"]
            labels = batch["label"]
            valid_mask = batch["valid_mask"]
            batch_sample_ids = batch["sample_id"]
        except KeyError as error:
            raise KeyError(
                "Each embedding batch must contain 'x', 'label', "
                "'valid_mask', and 'sample_id'"
            ) from error

        if not isinstance(x, torch.Tensor):
            raise TypeError("batch['x'] must be a torch.Tensor")
        if not isinstance(labels, torch.Tensor):
            raise TypeError("batch['label'] must be a torch.Tensor")
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("batch['valid_mask'] must be a torch.Tensor")

        x = x.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        valid_mask = valid_mask.to(
            device=device,
            dtype=torch.bool,
            non_blocking=True,
        )
        labels = labels.to(dtype=torch.long)

        logits, h = model(
            x,
            valid_mask=valid_mask,
        )

        h_parts.append(
            h.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        y_parts.append(
            labels.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        pred_parts.append(
            logits.argmax(dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        logit_parts.append(
            logits.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        sample_ids.extend(str(value) for value in batch_sample_ids)

    if not h_parts:
        raise ValueError("DataLoader produced no samples")

    h_all = np.concatenate(h_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    pred_all = np.concatenate(pred_parts, axis=0)
    logits_all = np.concatenate(logit_parts, axis=0)
    z_all = l2_normalize_embeddings(h_all)

    return EmbeddingBundle(
        h=h_all,
        z=z_all,
        y=y_all,
        cnn_pred=pred_all,
        logits=logits_all,
        sample_id=np.asarray(sample_ids, dtype=str),
    )


def extract_embedding_splits(
    model: nn.Module,
    loaders: Mapping[str, DataLoader[dict[str, object]]],
    *,
    device: torch.device,
) -> dict[str, EmbeddingBundle]:
    """Extract embeddings for named train/val/test-style loaders."""
    if not loaders:
        raise ValueError("loaders must contain at least one split")
    return {
        str(split_name): extract_embeddings(
            model,
            loader,
            device=device,
        )
        for split_name, loader in loaders.items()
    }


def standardize_feature_splits(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Standardize probe features using train statistics only.

    This is the same transform used by the current linear-probe notebook.
    """
    train = np.asarray(train_x)
    val = np.asarray(val_x)
    test = np.asarray(test_x)

    if train.ndim != 2 or val.ndim != 2 or test.ndim != 2:
        raise ValueError("train_x, val_x, and test_x must all be 2-D")
    if train.shape[1] != val.shape[1] or train.shape[1] != test.shape[1]:
        raise ValueError("Feature dimensions must match across splits")
    if len(train) == 0:
        raise ValueError("train_x must be non-empty")

    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)

    return (
        apply_feature_standardization(train, mean, std),
        apply_feature_standardization(val, mean, std),
        apply_feature_standardization(test, mean, std),
        mean,
        std,
    )


def apply_feature_standardization(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply previously fitted per-dimension feature standardization."""
    x = np.asarray(values)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"values must be 2-D, got {x.shape}")
    if mean.shape != (x.shape[1],) or std.shape != (x.shape[1],):
        raise ValueError(
            "mean/std must match feature dimension; "
            f"values={x.shape}, mean={mean.shape}, std={std.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("mean/std contain non-finite values")
    if np.any(std <= 0):
        raise ValueError("std must be strictly positive")

    transformed = (x - mean) / std
    if not np.isfinite(transformed).all():
        raise FloatingPointError(
            "Standardized feature matrix contains non-finite values"
        )
    return transformed.astype(np.float32)


def _bundle_field(
    bundle: EmbeddingBundle | Mapping[str, Any],
    key: str,
) -> np.ndarray:
    try:
        return np.asarray(bundle[key])
    except KeyError as error:
        raise KeyError(f"Embedding bundle is missing {key!r}") from error


def validate_paired_bundles(
    reference: EmbeddingBundle | Mapping[str, Any],
    query: EmbeddingBundle | Mapping[str, Any],
    *,
    require_labels: bool = True,
    require_sample_ids: bool = True,
) -> None:
    """Validate one-to-one raw/reconstruction style bundle alignment."""
    reference_z = _bundle_field(reference, "z")
    query_z = _bundle_field(query, "z")

    if reference_z.shape != query_z.shape:
        raise ValueError(
            "Paired embedding shapes differ: "
            f"{reference_z.shape} vs {query_z.shape}"
        )

    if require_labels:
        reference_y = _bundle_field(reference, "y")
        query_y = _bundle_field(query, "y")
        if not np.array_equal(reference_y, query_y):
            raise ValueError("Paired embedding labels are not identical")

    if require_sample_ids:
        reference_ids = _bundle_field(reference, "sample_id").astype(str)
        query_ids = _bundle_field(query, "sample_id").astype(str)
        if not np.array_equal(reference_ids, query_ids):
            raise ValueError("Paired embedding sample_id order is not identical")


def paired_embedding_cosine(
    reference: EmbeddingBundle | Mapping[str, Any],
    query: EmbeddingBundle | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sample cosine similarity and cosine distance for paired z."""
    validate_paired_bundles(reference, query)

    reference_z = np.asarray(reference["z"], dtype=np.float64)
    query_z = np.asarray(query["z"], dtype=np.float64)

    similarity = np.sum(reference_z * query_z, axis=1)
    similarity = np.clip(similarity, -1.0, 1.0)
    distance = 1.0 - similarity
    return similarity, distance
