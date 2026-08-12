from __future__ import annotations

"""Training utilities for the current acceleration CNN interface.

The functions in this module expect DataLoader batches with at least:

    batch["x"]          -> (B, 3, T)
    batch["label"]      -> (B,)
    batch["valid_mask"] -> (B, T)

Additional fields such as ``valid_length`` and ``sample_id`` are allowed and
ignored by the trainer.

The core API remains compatible with the current notebook:

    run_cnn_epoch(...)
    fit_cnn(...)

Two notebook globals have deliberately become explicit optional arguments:

- ``grad_clip_norm`` replaces the old implicit ``GRAD_CLIP_NORM``.
- ``max_train_batches`` replaces the old implicit ``MAX_TRAIN_BATCHES``.

Both default to ``None``, so the existing call pattern remains valid.
"""

import copy
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


try:
    # Prefer the repository implementation when this module lives inside
    # WritingRing, so reported numbers match existing notebooks exactly.
    from snn.action0_engine import (
        _classification_metrics as _repository_classification_metrics,
    )
except (ImportError, ModuleNotFoundError):
    _repository_classification_metrics = None


__all__ = [
    "classification_metrics",
    "compute_balanced_class_weights",
    "build_cross_entropy",
    "build_adam_optimizer",
    "run_cnn_epoch",
    "fit_cnn",
    "evaluate_cnn_splits",
]


def _fallback_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Dependency-light fallback for accuracy, balanced accuracy, and macro F1.

    In the WritingRing repository, ``snn.action0_engine._classification_metrics``
    is preferred automatically. This fallback mainly makes the module portable
    and testable outside the repository.
    """
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)

    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true/y_pred shape mismatch: {true.shape} vs {pred.shape}"
        )
    if len(true) == 0:
        raise ValueError("Cannot compute classification metrics on zero samples")

    accuracy = float(np.mean(true == pred))
    classes = np.unique(true)

    recalls: list[float] = []
    f1_values: list[float] = []

    for class_index in classes:
        true_positive = int(
            np.count_nonzero(
                (true == class_index) & (pred == class_index)
            )
        )
        false_negative = int(
            np.count_nonzero(
                (true == class_index) & (pred != class_index)
            )
        )
        false_positive = int(
            np.count_nonzero(
                (true != class_index) & (pred == class_index)
            )
        )

        recall_denominator = true_positive + false_negative
        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )
        recalls.append(float(recall))

        precision_denominator = true_positive + false_positive
        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
        denominator = precision + recall
        f1 = (
            2.0 * precision * recall / denominator
            if denominator > 0
            else 0.0
        )
        f1_values.append(float(f1))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
    }


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Compute the classification metrics used by the current experiments."""
    if _repository_classification_metrics is not None:
        metrics = _repository_classification_metrics(
            list(y_true),
            list(y_pred),
        )
        return {
            str(key): float(value)
            for key, value in metrics.items()
        }
    return _fallback_classification_metrics(y_true, y_pred)


def compute_balanced_class_weights(
    train_labels: Sequence[int] | np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    """Return inverse-frequency class weights with mean weight approximately 1.

    The formula matches the current notebook:

        weight_c = N / (C * N_c)

    where N is the number of training samples, C is the class count, and N_c
    is the number of training samples belonging to class c.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    labels = np.asarray(train_labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError(
            "train_labels must be a non-empty one-dimensional array"
        )
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(
            f"train_labels must lie in [0, {num_classes - 1}]"
        )

    counts = np.bincount(
        labels,
        minlength=num_classes,
    ).astype(np.float64)

    if np.any(counts <= 0):
        missing = np.flatnonzero(counts <= 0).tolist()
        raise ValueError(
            "Cannot build balanced class weights because at least one "
            f"training class has zero samples: {missing}"
        )

    return counts.sum() / (num_classes * counts)


def build_cross_entropy(
    *,
    device: torch.device,
    train_labels: Sequence[int] | np.ndarray | None = None,
    num_classes: int | None = None,
    use_class_weights: bool = False,
) -> nn.CrossEntropyLoss:
    """Build the baseline cross-entropy criterion.

    With ``use_class_weights=False`` this is exactly the unweighted criterion
    used by the current baseline notebook.

    With ``use_class_weights=True``, ``train_labels`` and ``num_classes`` are
    required and the same inverse-frequency weighting formula as the notebook
    is used.
    """
    weight_tensor: torch.Tensor | None = None

    if use_class_weights:
        if train_labels is None or num_classes is None:
            raise ValueError(
                "train_labels and num_classes are required when "
                "use_class_weights=True"
            )
        weights = compute_balanced_class_weights(
            train_labels,
            num_classes=num_classes,
        )
        weight_tensor = torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        )

    return nn.CrossEntropyLoss(weight=weight_tensor)


def build_adam_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.0,
) -> torch.optim.Adam:
    """Build the Adam optimizer used by the current CNN baseline."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    return torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def run_cnn_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, object]],
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    """Run one training or evaluation epoch.

    Training mode is selected by supplying ``optimizer``. If ``optimizer`` is
    ``None``, the model runs in evaluation mode without gradient tracking.

    Parameters
    ----------
    max_batches:
        Optional smoke-test limit. ``None`` uses the full DataLoader.
    grad_clip_norm:
        Optional max norm for gradient clipping. ``None`` disables clipping.
    """
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive or None")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive or None")

    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = 0.0
    sample_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.set_grad_enabled(is_train):
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            try:
                x = batch["x"]
                labels = batch["label"]
                valid_mask = batch["valid_mask"]
            except KeyError as error:
                raise KeyError(
                    "Each batch must contain 'x', 'label', and 'valid_mask'"
                ) from error

            if not isinstance(x, torch.Tensor):
                raise TypeError("batch['x'] must be a torch.Tensor")
            if not isinstance(labels, torch.Tensor):
                raise TypeError("batch['label'] must be a torch.Tensor")
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError(
                    "batch['valid_mask'] must be a torch.Tensor"
                )

            x = x.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            labels = labels.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            valid_mask = valid_mask.to(
                device=device,
                dtype=torch.bool,
                non_blocking=True,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits, _ = model(
                x,
                valid_mask=valid_mask,
            )
            loss = criterion(
                logits,
                labels,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CNN loss")

            if is_train:
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(grad_clip_norm),
                    )
                optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = int(labels.shape[0])

            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size
            y_true.extend(
                labels.detach().cpu().tolist()
            )
            y_pred.extend(
                predictions.detach().cpu().tolist()
            )

    if sample_count == 0:
        raise ValueError("DataLoader produced no samples")

    metrics = classification_metrics(
        y_true,
        y_pred,
    )
    metrics["loss"] = loss_sum / sample_count
    return metrics


def fit_cnn(
    model: nn.Module,
    *,
    train_loader: DataLoader[dict[str, object]],
    val_loader: DataLoader[dict[str, object]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int | None,
    max_train_batches: int | None = None,
    grad_clip_norm: float | None = None,
    selection_metric: str = "balanced_accuracy",
    min_improvement: float = 1e-12,
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    dict[str, torch.Tensor],
    int,
    float,
]:
    """Fit the CNN and keep the best validation state.

    The first seven parameters preserve the existing notebook interface.
    Additional keyword-only options remove former notebook-global dependencies
    while keeping their old behavior by default.

    Returns
    -------
    history:
        DataFrame with one train and one validation row per completed epoch.
    best_state:
        Deep-copied state dict from the best validation epoch.
    best_epoch:
        1-based epoch index.
    best_metric:
        Best validation value of ``selection_metric``.
    """
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if patience is not None and patience <= 0:
        raise ValueError("patience must be positive or None")
    if max_train_batches is not None and max_train_batches <= 0:
        raise ValueError(
            "max_train_batches must be positive or None"
        )
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError(
            "grad_clip_norm must be positive or None"
        )
    if min_improvement < 0:
        raise ValueError("min_improvement must be non-negative")
    if not selection_metric:
        raise ValueError("selection_metric must be non-empty")

    history_rows: list[
        dict[str, float | int | str]
    ] = []

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_metric = -math.inf
    epochs_without_improvement = 0

    for epoch in range(
        1,
        num_epochs + 1,
    ):
        train_metrics = run_cnn_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            max_batches=max_train_batches,
            grad_clip_norm=grad_clip_norm,
        )

        val_metrics = run_cnn_epoch(
            model,
            val_loader,
            criterion,
            device=device,
        )

        if selection_metric not in val_metrics:
            raise KeyError(
                f"selection_metric={selection_metric!r} is absent from "
                f"validation metrics {sorted(val_metrics)}"
            )

        history_rows.extend(
            [
                {
                    "epoch": epoch,
                    "split": "train",
                    **train_metrics,
                },
                {
                    "epoch": epoch,
                    "split": "val",
                    **val_metrics,
                },
            ]
        )

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss={train_metrics['loss']:.5f}, "
                f"BA={train_metrics['balanced_accuracy']:.5f} | "
                f"val loss={val_metrics['loss']:.5f}, "
                f"BA={val_metrics['balanced_accuracy']:.5f}"
            )

        current = float(
            val_metrics[selection_metric]
        )

        if current > best_metric + min_improvement:
            best_metric = current
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            patience is not None
            and epochs_without_improvement >= patience
        ):
            if verbose:
                print(
                    f"Early stopping after {epoch} epochs; "
                    f"best epoch={best_epoch}"
                )
            break

    if best_state is None:
        raise RuntimeError(
            "Training completed without a best validation state"
        )

    history = pd.DataFrame(
        history_rows
    )
    return (
        history,
        best_state,
        best_epoch,
        best_metric,
    )


def evaluate_cnn_splits(
    model: nn.Module,
    loaders: Mapping[
        str,
        DataLoader[dict[str, object]],
    ],
    criterion: nn.Module,
    *,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate a trained model on named splits using the same epoch metrics.

    Example
    -------
    ``loaders={"train": train_eval_loader, "val": val_loader, "test": test_loader}``
    """
    if not loaders:
        raise ValueError("loaders must contain at least one split")

    split_metrics = {
        str(split_name): run_cnn_epoch(
            model,
            loader,
            criterion,
            device=device,
        )
        for split_name, loader in loaders.items()
    }

    frame = pd.DataFrame(
        split_metrics
    ).T
    frame.index.name = "split"
    return frame
