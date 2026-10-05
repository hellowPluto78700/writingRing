"""Spike-only output trajectories and Core-style sample-balanced TSCE."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from core_benchmark_v1.model import valid_mask
from core_benchmark_v1.extensions.output_spike_drain.model import (
    drain_output_state,
    valid_output,
)


def spike_tsce_output(
    evidence: torch.Tensor,
    lengths: torch.Tensor,
    *,
    beta: float,
    threshold: float,
    slope: float,
    max_drain_steps: int,
    include_drain: bool = True,
) -> dict[str, torch.Tensor]:
    """Run cap-1 output dynamics and optionally the established endpoint drain.

    Valid TSCE uses the binary output spike emitted at each sensory timestep.
    Drain TSCE adds the complete positive endpoint drain count to the sample's
    final valid timestep. Drain iterations are serialization time, not new
    sensory timesteps and therefore never enlarge the TSCE denominator.
    """
    valid = valid_output(
        evidence,
        lengths,
        beta=beta,
        threshold=threshold,
        slope=slope,
    )
    spike = valid["spike"]
    if include_drain:
        drain = drain_output_state(
            valid["final_membrane"],
            threshold=threshold,
            slope=slope,
            max_steps=max_drain_steps,
        )
        drain_count = drain["drain_count"]
    else:
        drain_count = torch.zeros_like(valid["valid_count"])
        drain = {
            "drain_count": drain_count,
            "residual_after_drain": valid["final_membrane"],
            "required_steps_per_unit": torch.zeros_like(
                valid["final_membrane"], dtype=torch.int64
            ),
            "k_required": valid["final_membrane"].new_zeros(()),
            "cap_hit": valid["final_membrane"].new_zeros(()),
        }

    endpoint = torch.zeros_like(spike)
    final_index = (lengths - 1).clamp_min(0)
    endpoint[torch.arange(spike.shape[0], device=spike.device), final_index] = drain_count
    total_count = valid["valid_count"] + drain_count
    return {
        **valid,
        **drain,
        "total_count": total_count,
        "valid_mean_logits": valid["valid_count"] / lengths[:, None],
        "drained_mean_logits": total_count / lengths[:, None],
        "valid_trajectory": spike,
        "drained_trajectory": spike + endpoint,
    }


def spike_tsce_loss(
    output: dict[str, torch.Tensor],
    lengths: torch.Tensor,
    y: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Core O1 temporal reduction applied to spike-only output logits."""
    if mode == "valid":
        trajectory = output["valid_trajectory"]
    elif mode == "drain":
        trajectory = output["drained_trajectory"]
    else:
        raise ValueError(mode)
    targets = y[:, None].expand(-1, trajectory.shape[1])
    losses = F.cross_entropy(trajectory.transpose(1, 2), targets, reduction="none")
    mask = valid_mask(lengths, trajectory.shape[1])
    return ((losses * mask).sum(1) / lengths).mean()


def native_count(output: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "valid":
        return output["valid_count"]
    if mode == "drain":
        return output["total_count"]
    raise ValueError(mode)
