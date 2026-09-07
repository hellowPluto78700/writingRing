from __future__ import annotations

import torch

from scripts import experiment_5_3_2_when_representation as _impl


# Re-export the experiment contract while patching the trajectory routine used by
# all Slurm entrypoints below. GitHub's contents API is whole-file only; keeping
# this correction isolated avoids duplicating the 1,300-line experiment module.
globals().update(
    {
        name: value
        for name, value in vars(_impl).items()
        if not name.startswith("__")
    }
)


def _safe_forward_trajectory(
    self: _impl.WhenBranchNet,
    x: torch.Tensor,
    lengths: torch.Tensor,
    reset_state_each_step: bool = False,
) -> _impl.WhenTrajectory:
    """Run the WHEN SNN without mutating previously recorded trajectory states.

    The reset intervention must remove carried state before the current input is
    processed. Fresh zero tensors are assigned rather than calling ``zero_()``
    because the latter aliases tensors already appended to the trajectory.
    """

    if torch.any(lengths <= 0) or torch.any(lengths > x.shape[1]):
        raise ValueError("Invalid sequence lengths for WHEN trajectory")
    synaptic, membrane, previous_spike = self._initial_state(x)
    synaptic_parts: list[torch.Tensor] = []
    membrane_parts: list[torch.Tensor] = []
    spike_parts: list[torch.Tensor] = []
    for timestep in range(x.shape[1]):
        if reset_state_each_step:
            synaptic = torch.zeros_like(synaptic)
            membrane = torch.zeros_like(membrane)
            previous_spike = torch.zeros_like(previous_spike)
        next_synaptic, next_membrane, next_spike = self._step(
            x[:, timestep], synaptic, membrane, previous_spike
        )
        valid = (timestep < lengths).unsqueeze(1)
        synaptic = torch.where(valid, next_synaptic, synaptic)
        membrane = torch.where(valid, next_membrane, membrane)
        spike = next_spike * valid.to(next_spike.dtype)
        synaptic_parts.append(synaptic)
        membrane_parts.append(membrane)
        spike_parts.append(spike)
        previous_spike = spike
    return _impl.WhenTrajectory(
        synaptic=torch.stack(synaptic_parts, dim=1),
        membranes=torch.stack(membrane_parts, dim=1),
        spikes=torch.stack(spike_parts, dim=1),
    )


_impl.WhenBranchNet.forward_trajectory = _safe_forward_trajectory
WhenBranchNet = _impl.WhenBranchNet


def main() -> None:
    _impl.main()


if __name__ == "__main__":
    main()
