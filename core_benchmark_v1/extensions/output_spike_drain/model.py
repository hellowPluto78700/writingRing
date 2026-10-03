"""Positive-only cap-1 output LIF with output-only full drain after valid time."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from core_benchmark_v1.model import lif_step, valid_mask, valid_sum


def valid_output(
    evidence: torch.Tensor,
    lengths: torch.Tensor,
    *,
    beta: float,
    threshold: float,
    slope: float,
) -> dict[str, torch.Tensor]:
    """Run the exact Core cap-1 LIF semantics during each sample's valid window."""
    if evidence.ndim != 3 or lengths.shape != (evidence.shape[0],):
        raise ValueError("Unaligned evidence/lengths")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    batch, steps, classes = evidence.shape
    membrane = evidence.new_zeros(batch, classes)
    spikes: list[torch.Tensor] = []
    pre_reset: list[torch.Tensor] = []
    for t in range(steps):
        active = (t < lengths)[:, None]
        spike, new_membrane, pre = lif_step(
            evidence[:, t], membrane, beta, threshold, slope
        )
        membrane = torch.where(active, new_membrane, membrane)
        spikes.append(spike * active)
        pre_reset.append(pre * active)
    spike_tensor = torch.stack(spikes, dim=1)
    return {
        "spike": spike_tensor,
        "pre_reset": torch.stack(pre_reset, dim=1),
        "final_membrane": membrane,
        "valid_count": valid_sum(spike_tensor, lengths),
    }


def required_drain_steps(
    final_membrane: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Integer serialization depth for positive supra-threshold backlog only."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return torch.floor(torch.clamp_min(final_membrane.detach(), 0.0) / threshold).to(torch.int64)


def drain_output_state(
    final_membrane: torch.Tensor,
    *,
    threshold: float,
    slope: float,
    max_steps: int,
) -> dict[str, torch.Tensor]:
    """Serialize positive endpoint backlog with beta_drain=1 and zero new input."""
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    required = required_drain_steps(final_membrane, threshold=threshold)
    k_required = int(required.max().item()) if required.numel() else 0
    cap_hit = k_required > max_steps
    steps = min(k_required, max_steps)
    membrane = final_membrane
    count = torch.zeros_like(membrane)
    for _ in range(steps):
        spike, membrane, _ = lif_step(
            torch.zeros_like(membrane), membrane, 1.0, threshold, slope
        )
        count = count + spike
    return {
        "drain_count": count,
        "residual_after_drain": membrane,
        "required_steps_per_unit": required,
        "k_required": final_membrane.new_tensor(float(k_required)),
        "cap_hit": final_membrane.new_tensor(float(cap_hit)),
    }


def drained_spike_output(
    evidence: torch.Tensor,
    lengths: torch.Tensor,
    *,
    beta: float,
    threshold: float,
    slope: float,
    max_drain_steps: int,
) -> dict[str, torch.Tensor]:
    valid = valid_output(
        evidence, lengths, beta=beta, threshold=threshold, slope=slope
    )
    drain = drain_output_state(
        valid["final_membrane"],
        threshold=threshold,
        slope=slope,
        max_steps=max_drain_steps,
    )
    total_count = valid["valid_count"] + drain["drain_count"]
    return {
        **valid,
        **drain,
        "total_count": total_count,
        "valid_mean_logits": valid["valid_count"] / lengths[:, None],
        "drained_mean_logits": total_count / lengths[:, None],
    }


def drain_loss(output: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    """Exact Core E2E logit scaling, replacing valid count by fully-drained count."""
    return F.cross_entropy(output["drained_mean_logits"], y)
