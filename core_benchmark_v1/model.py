"""Paired multi-tau SNN with unnormalized synapses and a bias-free accumulator."""
from __future__ import annotations
import math
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from .protocol import Protocol, Run, paired_seed


class BinarySpike(torch.autograd.Function):
    """Same cap=1 forward/surrogate derivative as historical MacroMultiSpikeLIF."""
    @staticmethod
    def forward(ctx: Any, v: torch.Tensor, threshold: float, slope: float) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.threshold, ctx.slope = threshold, slope
        return (v >= threshold).to(v.dtype)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (v,) = ctx.saved_tensors
        distance = (v - ctx.threshold) / ctx.threshold
        derivative = 1 / (1 + ctx.slope * distance.abs()).square() / ctx.threshold
        return grad * derivative, None, None


def lif_step(current: torch.Tensor, membrane: torch.Tensor, beta: float, threshold: float, slope: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = beta * membrane + current
    spike = BinarySpike.apply(pre, threshold, slope)
    return spike, pre - threshold * spike, pre


def valid_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.arange(steps, device=lengths.device)[None, :] < lengths[:, None]


def valid_sum(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return (values * valid_mask(lengths, values.shape[1]).unsqueeze(-1)).sum(1)


def mean_logits(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return valid_sum(values, lengths) / lengths[:, None]


def sequence_loss(values: torch.Tensor, lengths: torch.Tensor, y: torch.Tensor, objective: str) -> torch.Tensor:
    if objective == 'wcce':
        return F.cross_entropy(mean_logits(values, lengths), y)
    if objective == 'tsce':
        # Equal weight per sample, not extra weight for longer gestures.
        targets = y[:, None].expand(-1, values.shape[1])
        loss = F.cross_entropy(values.transpose(1, 2), targets, reduction='none')
        return ((loss * valid_mask(lengths, values.shape[1])).sum(1) / lengths).mean()
    raise ValueError(objective)


def objective_loss(trajectory: dict[str, Any], lengths: torch.Tensor, y: torch.Tensor, run: Run, p: Protocol) -> torch.Tensor:
    main = sequence_loss(trajectory['evidence'], lengths, y, 'tsce' if run.objective == 'tsce' else 'wcce')
    if run.objective in ('wcce_l1_wcce', 'wcce_l1_tsce'):
        kind = 'tsce' if run.objective.endswith('tsce') else 'wcce'
        main = main + p.auxiliary_weight * sequence_loss(trajectory['auxiliary'], lengths, y, kind)
    return main


class BenchmarkNet(nn.Module):
    def __init__(self, run: Run, p: Protocol) -> None:
        super().__init__()
        p.validate()
        self.run, self.protocol = run, p
        self.layers = nn.ModuleList(nn.Linear(p.input_channels if i == 0 else p.width, p.width, bias=False) for i in range(len(run.shifts)))
        self.head = nn.Linear(p.width, len(p.labels), bias=False)
        self.auxiliary = nn.Linear(p.width, len(p.labels), bias=False) if run.objective.startswith('wcce_l1') else None
        self.betas: list[float] = []
        for i, shifts in enumerate(run.shifts):
            q, r = divmod(p.width, len(shifts))
            values = [1 - 2.0 ** (-shift) for j, shift in enumerate(shifts) for _ in range(q + (j < r))]
            self.register_buffer(f'alpha_{i}', torch.tensor(values, dtype=torch.float32))
            tau = run.l1_tau_mem_ms if i == 0 and run.l1_tau_mem_ms is not None else p.tau_mem_ms
            self.betas.append(math.exp(-(1000 / p.fs) / tau))
        # Each common parameter has its own RNG stream; extra layers cannot shift it.
        for name, parameter in self.named_parameters():
            generator = torch.Generator().manual_seed(paired_seed(run.seed, f'init:{name}'))
            bound = 1 / math.sqrt(parameter.shape[1])
            with torch.no_grad():
                parameter.uniform_(-bound, bound, generator=generator)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, *, reset_at: torch.Tensor | None = None, reset_layers: tuple[int, ...] = ()) -> dict[str, Any]:
        p = self.protocol
        batch, steps, channels = x.shape
        if channels != p.input_channels or lengths.shape != (batch,) or (lengths < 1).any() or (lengths > steps).any():
            raise ValueError('Invalid input/valid-length geometry')
        if reset_at is not None and reset_at.shape != lengths.shape:
            raise ValueError('Reset boundaries must be per sample')
        syn = [x.new_zeros(batch, p.width) for _ in self.layers]
        mem = [x.new_zeros(batch, p.width) for _ in self.layers]
        spikes: list[list[torch.Tensor]] = [[] for _ in self.layers]
        pre_reset: list[list[torch.Tensor]] = [[] for _ in self.layers]
        evidence, auxiliary = [], []
        for t in range(int(lengths.max().item())):
            active = (t < lengths)[:, None]
            cur = x[:, t] * active
            for i, linear in enumerate(self.layers):
                if reset_at is not None and i in reset_layers:
                    retain = (reset_at != t)[:, None]
                    syn[i], mem[i] = syn[i] * retain, mem[i] * retain
                # Mandatory equation: there is NO (1-alpha) input-drive factor.
                candidate = getattr(self, f'alpha_{i}') * syn[i] + linear(cur)
                syn[i] = torch.where(active, candidate, syn[i])
                spike, new_mem, pre = lif_step(syn[i], mem[i], self.betas[i], p.threshold, p.surrogate_slope)
                mem[i] = torch.where(active, new_mem, mem[i])
                cur = spike * active
                spikes[i].append(cur)
                pre_reset[i].append(pre * active)
            evidence.append(self.head(cur))
            if self.auxiliary is not None:
                auxiliary.append(self.auxiliary(spikes[0][-1]))
        def stack(values: list[torch.Tensor]) -> torch.Tensor:
            result = torch.stack(values, dim=1)
            return F.pad(result, (0, 0, 0, steps - result.shape[1]))
        out: dict[str, Any] = {'spike': tuple(stack(s) for s in spikes),
                               'pre_reset': tuple(stack(s) for s in pre_reset),
                               'evidence': stack(evidence), 'final_syn': tuple(syn), 'final_mem': tuple(mem)}
        if auxiliary:
            out['auxiliary'] = stack(auxiliary)
        return out


def spike_readout(evidence: torch.Tensor, lengths: torch.Tensor, beta: float, threshold: float, slope: float) -> torch.Tensor:
    # alpha_out=0: no additional synaptic filter, signed membrane without clamp.
    membrane = evidence.new_zeros(evidence.shape[0], evidence.shape[-1])
    spikes = []
    for t in range(evidence.shape[1]):
        active = (t < lengths)[:, None]
        spike, new_mem, _ = lif_step(evidence[:, t], membrane, beta, threshold, slope)
        membrane = torch.where(active, new_mem, membrane)
        spikes.append(spike * active)
    return torch.stack(spikes, dim=1)
