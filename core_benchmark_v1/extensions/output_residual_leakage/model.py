"""Spike-limited output with endpoint residual correction."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn.functional as F
from core_benchmark_v1.model import BinarySpike, valid_mask


def residual_leakage_output(
    evidence: torch.Tensor,
    lengths: torch.Tensor,
    *,
    beta: float,
    threshold: float,
    slope: float,
) -> dict[str, torch.Tensor]:
    if evidence.ndim != 3 or lengths.shape != (evidence.shape[0],):
        raise ValueError("Unaligned evidence/lengths")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    batch, steps, classes = evidence.shape
    membrane = evidence.new_zeros(batch, classes)
    spikes: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    pre_reset: list[torch.Tensor] = []
    trajectory: list[torch.Tensor] = []
    for t in range(steps):
        active = (t < lengths)[:, None]
        pre = beta * membrane + evidence[:, t]
        spike = BinarySpike.apply(pre, threshold, slope) * active
        new_membrane = pre - threshold * spike
        membrane = torch.where(active, new_membrane, membrane)
        spikes.append(spike)
        residuals.append(membrane * active)
        pre_reset.append(pre * active)
        trajectory.append(threshold * spike)
    spike_tensor = torch.stack(spikes, dim=1)
    residual_tensor = torch.stack(residuals, dim=1)
    pre_tensor = torch.stack(pre_reset, dim=1)
    trajectory_tensor = torch.stack(trajectory, dim=1)
    final_index = (lengths - 1).clamp_min(0)
    final_residual = residual_tensor[torch.arange(batch, device=lengths.device), final_index]
    endpoint = torch.zeros_like(trajectory_tensor)
    endpoint[torch.arange(batch, device=lengths.device), final_index] = final_residual
    trajectory_tensor = trajectory_tensor + endpoint
    mask = valid_mask(lengths, steps).unsqueeze(-1)
    spike_score = (threshold * spike_tensor * mask).sum(1)
    total_score = (trajectory_tensor * mask).sum(1)
    analog_score = (evidence * mask).sum(1)
    leakage_drift = analog_score - total_score
    return {
        "spike": spike_tensor,
        "residual": residual_tensor,
        "pre_reset": pre_tensor,
        "trajectory": trajectory_tensor,
        "final_residual": final_residual,
        "spike_score": spike_score,
        "total_score": total_score,
        "analog_score": analog_score,
        "leakage_drift": leakage_drift,
    }


def extension_loss(output: dict[str, torch.Tensor], lengths: torch.Tensor, y: torch.Tensor, objective: str) -> torch.Tensor:
    trajectory = output["trajectory"]
    if objective == "wcce":
        return F.cross_entropy(output["total_score"] / lengths[:, None], y)
    if objective == "tsce":
        targets = y[:, None].expand(-1, trajectory.shape[1])
        losses = F.cross_entropy(trajectory.transpose(1, 2), targets, reduction="none")
        mask = valid_mask(lengths, trajectory.shape[1])
        return ((losses * mask).sum(1) / lengths).mean()
    raise ValueError(objective)