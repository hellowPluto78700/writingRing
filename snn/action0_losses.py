"""Mask-aware loss functions for fixed-length Action0 segments."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn


class MaskedCrossEntropySpkReg(nn.Module):
    """Cross entropy over valid steps, with optional valid-neuron-time regularization."""

    def __init__(self, spike_regularization: float = 0.0) -> None:
        super().__init__()
        if not math.isfinite(spike_regularization) or spike_regularization < 0:
            raise ValueError("spike_regularization must be a finite non-negative value")
        self.spike_regularization = float(spike_regularization)

    def forward(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
        spikes: torch.Tensor | None = None,
        *,
        spike_neuron_count: int | None = None,
    ) -> torch.Tensor:
        """Return masked classification loss plus optional spike regularization.

        ``labels`` is repeated across valid time steps only logically: invalid
        per-step losses are multiplied by zero before the mean is formed.
        """

        if output.ndim != 3:
            raise ValueError(f"output must have shape (batch, time, classes), got {tuple(output.shape)}")
        if labels.ndim != 1 or labels.shape[0] != output.shape[0]:
            raise ValueError(
                "labels must have shape (batch,); "
                f"expected {(output.shape[0],)}, got {tuple(labels.shape)}"
            )
        if valid_mask.shape != output.shape[:2]:
            raise ValueError(
                "valid_mask must have shape (batch, time); "
                f"expected {tuple(output.shape[:2])}, got {tuple(valid_mask.shape)}"
            )

        batch_size, time_steps, class_count = output.shape
        targets = labels[:, None].expand(batch_size, time_steps)
        loss_per_step = functional.cross_entropy(
            output.reshape(batch_size * time_steps, class_count),
            targets.reshape(batch_size * time_steps),
            reduction="none",
        ).reshape(batch_size, time_steps)
        mask = valid_mask.to(device=loss_per_step.device, dtype=loss_per_step.dtype)
        classification_loss = (loss_per_step * mask).sum() / mask.sum().clamp_min(1)

        if self.spike_regularization == 0:
            return classification_loss
        if spikes is None:
            raise ValueError("spikes are required when spike_regularization is non-zero")
        if spike_neuron_count is None or spike_neuron_count <= 0:
            raise ValueError(
                "a positive spike_neuron_count is required when spike_regularization is non-zero"
            )
        valid_neuron_time = mask.sum().clamp_min(1) * spike_neuron_count
        return classification_loss + self.spike_regularization * (spikes / valid_neuron_time)
