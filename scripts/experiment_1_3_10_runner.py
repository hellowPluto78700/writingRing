from __future__ import annotations

"""Experiment 1.3.10 entry point with bounded trainable synaptic dynamics.

The core implementation lives in experiment_1_3_10_stacked_bin_snn_ablation.py.
This runner defines the full architecture grid and replaces unconstrained
``learn_alpha=True`` dynamics with a physically bounded tau_syn
parameterization for the trainable-dynamics condition.
"""

import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate

import experiment_1_3_10_stacked_bin_snn_ablation as experiment


PROTOCOL_VERSION = "stacked250_v2_bounded_tau_syn"
TAU_SYN_MIN_MS = 20.0
TAU_SYN_MAX_MS = 300.0

ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "1h128": (128,),
    "1h256": (256,),
    "1h512": (512,),
    "1h1024": (1024,),
    "2h128": (128, 128),
}


def _tau_to_raw(tau_ms: torch.Tensor) -> torch.Tensor:
    """Inverse-logit initialization for tau in the bounded open interval."""
    p = (tau_ms - TAU_SYN_MIN_MS) / (TAU_SYN_MAX_MS - TAU_SYN_MIN_MS)
    p = p.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(p / (1.0 - p))


def _alpha_to_tau_ms(alpha: torch.Tensor, sampling_rate_hz: float) -> torch.Tensor:
    dt_ms = 1000.0 / float(sampling_rate_hz)
    return -dt_ms / torch.log(alpha.clamp(1e-6, 1.0 - 1e-6))


class BoundedTauSynaptic(snn.Synaptic):
    """snnTorch Synaptic cell with sigmoid-bounded trainable tau_syn.

    The state equations, reset semantics, beta handling, threshold handling,
    and surrogate gradient remain snnTorch 0.9.4 compatible. Only the source
    of alpha changes: alpha is derived differentiably from a bounded physical
    tau_syn rather than optimized directly.
    """

    def __init__(
        self,
        tau_syn_ms_init: torch.Tensor | float,
        sampling_rate_hz: float,
        beta: torch.Tensor | float,
        threshold: torch.Tensor | float,
        *,
        spike_grad,
        reset_mechanism: str,
        learn_tau_syn: bool,
        learn_beta: bool,
        learn_threshold: bool,
    ) -> None:
        tau_init = torch.as_tensor(tau_syn_ms_init, dtype=torch.float32)
        if torch.any(tau_init <= TAU_SYN_MIN_MS) or torch.any(tau_init >= TAU_SYN_MAX_MS):
            raise ValueError(
                f"Initial tau_syn must lie strictly inside "
                f"[{TAU_SYN_MIN_MS}, {TAU_SYN_MAX_MS}] ms"
            )

        self.sampling_rate_hz = float(sampling_rate_hz)
        self.dt_ms = 1000.0 / self.sampling_rate_hz
        alpha_init = torch.exp(-self.dt_ms / tau_init)

        # Keep snnTorch's alpha buffer only for state-dict compatibility.
        # The forward dynamics below use effective_alpha(), not this buffer.
        super().__init__(
            alpha=alpha_init,
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset_mechanism,
            learn_alpha=False,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )

        raw_tau = _tau_to_raw(tau_init)
        if learn_tau_syn:
            self.raw_tau_syn = nn.Parameter(raw_tau)
        else:
            self.register_buffer("raw_tau_syn", raw_tau)

    def tau_syn_ms(self) -> torch.Tensor:
        return TAU_SYN_MIN_MS + (TAU_SYN_MAX_MS - TAU_SYN_MIN_MS) * torch.sigmoid(
            self.raw_tau_syn
        )

    def effective_alpha(self) -> torch.Tensor:
        return torch.exp(-self.dt_ms / self.tau_syn_ms())

    def _base_state_function(self, input_: torch.Tensor):
        syn = self.effective_alpha() * self.syn + input_
        mem = self.beta.clamp(0, 1) * self.mem + syn
        return syn, mem

    def _base_state_reset_zero(self, input_: torch.Tensor):
        syn = self.effective_alpha() * self.syn + input_
        mem = self.beta.clamp(0, 1) * self.mem + syn
        return torch.zeros_like(syn), mem


class BoundedStackedBinSNN(experiment.StackedBinSNN):
    """Original Exp 1.3.10 network with bounded trainable tau_syn cells."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        num_classes: int,
        sampling_rate_hz: float,
        trainable_dynamics: bool,
    ) -> None:
        # Build exactly the original fixed-dynamics network first so Linear
        # weight initialization and all non-dynamics semantics stay paired.
        super().__init__(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            num_classes=num_classes,
            sampling_rate_hz=sampling_rate_hz,
            trainable_dynamics=False,
        )
        self.trainable_dynamics = bool(trainable_dynamics)
        if not self.trainable_dynamics:
            return

        spike_grad = surrogate.fast_sigmoid(slope=experiment.SURROGATE_SLOPE)
        beta0 = experiment.tau_ms_to_decay(experiment.TAU_MEM_MS, sampling_rate_hz)

        bounded_hidden = nn.ModuleList()
        for width in self.hidden_sizes:
            alpha0 = experiment.build_alpha_vector(width)
            tau0 = _alpha_to_tau_ms(alpha0, sampling_rate_hz)
            bounded_hidden.append(
                BoundedTauSynaptic(
                    tau_syn_ms_init=tau0,
                    sampling_rate_hz=sampling_rate_hz,
                    beta=torch.full((width,), beta0, dtype=torch.float32),
                    threshold=torch.full(
                        (width,), experiment.THRESHOLD, dtype=torch.float32
                    ),
                    spike_grad=spike_grad,
                    reset_mechanism=experiment.RESET_MECHANISM,
                    learn_tau_syn=True,
                    learn_beta=True,
                    learn_threshold=True,
                )
            )
        self.hidden_lifs = bounded_hidden

        # Preserve class symmetry at the output: one shared trainable tau_syn,
        # beta, and threshold scalar for every output class neuron.
        self.lif_out = BoundedTauSynaptic(
            tau_syn_ms_init=torch.tensor(
                experiment.OUTPUT_TAU_SYN_MS, dtype=torch.float32
            ),
            sampling_rate_hz=sampling_rate_hz,
            beta=beta0,
            threshold=experiment.THRESHOLD,
            spike_grad=spike_grad,
            reset_mechanism=experiment.RESET_MECHANISM,
            learn_tau_syn=True,
            learn_beta=True,
            learn_threshold=True,
        )


def _lif_tau_syn_ms(lif, sampling_rate_hz: float):
    if isinstance(lif, BoundedTauSynaptic):
        return experiment._tensor_numpy(lif.tau_syn_ms())
    return experiment.decay_to_tau_ms(
        experiment._tensor_numpy(lif.alpha), sampling_rate_hz
    )


def dynamics_summary(model: BoundedStackedBinSNN) -> dict[str, object]:
    layers: list[dict[str, object]] = []
    for index, lif in enumerate(model.hidden_lifs, start=1):
        beta = experiment._tensor_numpy(lif.beta)
        threshold = experiment._tensor_numpy(lif.threshold)
        layers.append(
            {
                "layer": f"hidden_{index}",
                "tau_syn_ms": experiment._summary(
                    _lif_tau_syn_ms(lif, model.sampling_rate_hz)
                ),
                "tau_mem_ms": experiment._summary(
                    experiment.decay_to_tau_ms(beta, model.sampling_rate_hz)
                ),
                "threshold": experiment._summary(threshold),
            }
        )

    output_beta = experiment._tensor_numpy(model.lif_out.beta)
    output_threshold = experiment._tensor_numpy(model.lif_out.threshold)
    return {
        "tau_syn_parameterization": (
            f"tau_ms={TAU_SYN_MIN_MS}+"
            f"{TAU_SYN_MAX_MS - TAU_SYN_MIN_MS}*sigmoid(raw_tau_syn)"
        ),
        "tau_syn_bounds_ms": {
            "min": TAU_SYN_MIN_MS,
            "max": TAU_SYN_MAX_MS,
        },
        "hidden": layers,
        "output": {
            "tau_syn_ms": experiment._summary(
                _lif_tau_syn_ms(model.lif_out, model.sampling_rate_hz)
            ),
            "tau_mem_ms": experiment._summary(
                experiment.decay_to_tau_ms(output_beta, model.sampling_rate_hz)
            ),
            "threshold": experiment._summary(output_threshold),
        },
    }


# Patch the core experiment at import time. The existing run mapping, training
# loop, checkpoint selection, evaluation, and artifact finalization are reused.
experiment.PROTOCOL_VERSION = PROTOCOL_VERSION
experiment.ARCHITECTURES = ARCHITECTURES
experiment.StackedBinSNN = BoundedStackedBinSNN
experiment.dynamics_summary = dynamics_summary
experiment.EXPECTED_RUNS = (
    len(experiment.SPLIT_SEEDS)
    * len(ARCHITECTURES)
    * len(experiment.OBJECTIVES)
    * len(experiment.TRAIN_REGIMES)
)


if __name__ == "__main__":
    experiment.main()
