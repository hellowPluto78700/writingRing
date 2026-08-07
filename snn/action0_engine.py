"""Training and evaluation loop for masked Action0 segment classification."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from .action0_losses import MaskedCrossEntropySpkReg


def masked_spike_counts(output: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Sum output spikes over valid time steps only."""

    if output.ndim != 3:
        raise ValueError(f"output must have shape (batch, time, classes), got {tuple(output.shape)}")
    if valid_mask.shape != output.shape[:2]:
        raise ValueError(
            "valid_mask must have shape (batch, time); "
            f"expected {tuple(output.shape[:2])}, got {tuple(valid_mask.shape)}"
        )
    mask = valid_mask.unsqueeze(-1).to(device=output.device, dtype=output.dtype)
    return (output * mask).sum(dim=1)


def normalized_spike_scores(spike_counts: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Convert segment spike counts into finite class scores, including all-zero output."""

    if spike_counts.ndim != 2:
        raise ValueError(
            f"spike_counts must have shape (batch, classes), got {tuple(spike_counts.shape)}"
        )
    if spike_counts.shape[1] == 0:
        raise ValueError("spike_counts must include at least one class")
    if eps <= 0:
        raise ValueError("eps must be positive")
    return (spike_counts + eps) / (
        spike_counts.sum(dim=1, keepdim=True) + eps * spike_counts.shape[1]
    )


def run_epoch(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: MaskedCrossEntropySpkReg,
    optimizer: Optimizer | None = None,
    *,
    split: str = "train",
    device: torch.device | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run a train, validation, or test epoch without counting padded positions."""

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    is_train = split == "train"
    if is_train and optimizer is None:
        raise ValueError("optimizer is required for a training epoch")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    if device is None:
        device = torch.device("cpu")

    model.train(is_train)
    sample_count = 0
    running_loss = 0.0
    total_output_spikes = 0.0
    total_valid_spikes = 0.0
    zero_output_segments = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.set_grad_enabled(is_train):
        for batch_index, (inputs, labels, valid_mask) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            valid_mask = valid_mask.to(device, dtype=torch.bool)
            if inputs.ndim != 3:
                raise ValueError(f"inputs must have shape (batch, time, channels), got {tuple(inputs.shape)}")
            if labels.shape != (inputs.shape[0],):
                raise ValueError(
                    f"labels must have shape {(inputs.shape[0],)}, got {tuple(labels.shape)}"
                )
            if valid_mask.shape != inputs.shape[:2]:
                raise ValueError(
                    "valid_mask must have shape (batch, time); "
                    f"expected {tuple(inputs.shape[:2])}, got {tuple(valid_mask.shape)}"
                )
            if not valid_mask.any(dim=1).all():
                raise ValueError("every segment must contain at least one valid time step")

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad()

            output = model(inputs, valid_mask=valid_mask)
            if output.ndim != 3 or output.shape[:2] != inputs.shape[:2]:
                raise AssertionError(
                    "model output must have shape (batch, time, classes); "
                    f"inputs={tuple(inputs.shape)}, output={tuple(output.shape)}"
                )
            if output.shape[2] <= 0:
                raise AssertionError("model output must include at least one class")

            spikes = getattr(model, "spkTotal", None)
            spike_neuron_count = getattr(model, "spike_neuron_count", None)
            loss = criterion(
                output,
                labels,
                valid_mask,
                spikes=spikes,
                spike_neuron_count=spike_neuron_count,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {split} loss")
            if is_train:
                loss.backward()
                optimizer.step()

            spike_counts = masked_spike_counts(output, valid_mask)
            scores = normalized_spike_scores(spike_counts)
            predictions = spike_counts.argmax(dim=1)
            batch_size = inputs.shape[0]
            if spikes is None:
                raise AssertionError("Action0 model must expose spkTotal after forward")

            sample_count += batch_size
            running_loss += float(loss.detach()) * batch_size
            total_output_spikes += float(spike_counts.detach().sum())
            total_valid_spikes += float(spikes.detach())
            zero_output_segments += int((spike_counts.sum(dim=1) == 0).sum())
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())
            del scores

    if sample_count == 0:
        raise ValueError(f"{split} dataloader produced no batches")
    metrics = _classification_metrics(y_true, y_pred)
    metrics.update(
        {
            "loss": running_loss / sample_count,
            "mean_output_spikes": total_output_spikes / sample_count,
            "mean_total_spikes": total_valid_spikes / sample_count,
            "zero_output_spike_fraction": zero_output_segments / sample_count,
            "total_output_spikes": total_output_spikes,
            "total_valid_spikes": total_valid_spikes,
        }
    )
    return metrics


def _classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("metrics require equally sized, non-empty predictions and targets")
    accuracy = sum(target == prediction for target, prediction in zip(y_true, y_pred)) / len(y_true)
    classes = sorted(set(y_true) | set(y_pred))
    recalls: list[float] = []
    f1_values: list[float] = []
    weighted_f1_total = 0.0
    for class_index in classes:
        true_positive = sum(
            target == class_index and prediction == class_index
            for target, prediction in zip(y_true, y_pred)
        )
        false_positive = sum(
            target != class_index and prediction == class_index
            for target, prediction in zip(y_true, y_pred)
        )
        false_negative = sum(
            target == class_index and prediction != class_index
            for target, prediction in zip(y_true, y_pred)
        )
        support = true_positive + false_negative
        precision_denominator = true_positive + false_positive
        recall = true_positive / support if support else 0.0
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        f1_denominator = precision + recall
        f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0
        if support:
            recalls.append(recall)
            weighted_f1_total += f1 * support
        f1_values.append(f1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_values) / len(f1_values),
        "weighted_f1": weighted_f1_total / len(y_true),
    }
