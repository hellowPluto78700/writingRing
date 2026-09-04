from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from snntorch import surrogate

from scripts import experiment_5_2_frozen_local_tauR_sweep as exp52


base = exp52.base
probe_utils = exp52.probe_utils

EXPERIMENT_ID = "experiment_5_3_synaptic_contextual_evidence"
PROTOCOL_VERSION = "frozen_local_synaptic_history_context_v1"
SEEDS = (11, 23, 37, 53, 71)
LOCAL_WIDTH = 128
TEMPORAL_WIDTH = 128
N_REL = 10
THRESHOLD = base.THRESHOLD
RESET = base.RESET
SURROGATE_SLOPE = base.SURROGATE_SLOPE
EPOCHS = base.EPOCHS
BATCH_SIZE = base.BATCH_SIZE
LR = 1e-3
WEIGHT_DECAY = 0.0
LOGIT_GAIN = 5.0
TAIL_SECONDS = 2.0
SHUFFLE_REPLICATES = 5
EPS = 1e-8

CLASSIFICATION_PROBES = (
    "hidden_whole_count",
    "hidden_fixed250_ordered",
    "hidden_relative10_ordered",
    "hidden_uend",
)
PHASE_PROBES = (
    "phase_contextual",
    "phase_synaptic_state",
    "phase_membrane",
    "phase_spike",
    "phase_gate",
)


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    beta: float | None
    shifts_syn: tuple[int, ...] = ()


CONDITIONS = (
    Condition("direct", "direct", None),
    Condition("lif_tau242", "lif", 0.9375),
    Condition("lif_beta050", "lif", 0.50),
    Condition("syn_single_s4", "syn", 0.50, (4,)),
    Condition("syn_single_s5", "syn", 0.50, (5,)),
    Condition("syn_single_s6", "syn", 0.50, (6,)),
    Condition("syn_multi_s23456", "syn", 0.50, (2, 3, 4, 5, 6)),
    Condition("syn_multi_s456", "syn", 0.50, (4, 5, 6)),
    Condition("syn_multi_s56", "syn", 0.50, (5, 6)),
    Condition("rsnn_beta050", "rsnn", 0.50),
    Condition("factorized_rsnn_beta050", "factorized", 0.50),
)
CONDITION_BY_NAME = {condition.name: condition for condition in CONDITIONS}
CONDITION_NAMES = tuple(condition.name for condition in CONDITIONS)


@dataclass(frozen=True)
class RunSpec:
    condition: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.condition}__seed{self.seed}"

    @property
    def definition(self) -> Condition:
        try:
            return CONDITION_BY_NAME[self.condition]
        except KeyError as error:
            raise ValueError(f"Unknown Exp5.3 condition: {self.condition}") from error


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


@dataclass
class Trajectory:
    contextual: torch.Tensor
    evidence: torch.Tensor
    spikes: torch.Tensor
    membranes: torch.Tensor
    synaptic: torch.Tensor
    gates: torch.Tensor


def find_repo_root(start: Path | None = None) -> Path:
    return exp52.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_results_dir(repo_root: Path) -> Path:
    return exp52.results_dir(repo_root)


def local_phase_path(root: Path, seed: int) -> Path:
    return root / "local_phase" / f"seed{seed}.json"


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def decay_from_shift(shift: int) -> float:
    return exp52.beta_from_shift(shift)


def tau_ms_from_decay(decay: float, fs: float) -> float:
    if decay == 1.0:
        return math.inf
    if not 0.0 < decay < 1.0:
        raise ValueError(f"Decay must be in (0, 1], got {decay}")
    return -(1000.0 / float(fs)) / math.log(float(decay))


def tau_ms_from_shift(shift: int, fs: float) -> float:
    return tau_ms_from_decay(decay_from_shift(shift), fs)


def run_specs() -> list[RunSpec]:
    return [RunSpec(condition.name, seed) for condition in CONDITIONS for seed in SEEDS]


def _loader(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _partitions(
    data: base.Data,
    cache: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        "train": (cache["Xtrain"], data.ytr, data.ltr),
        "val": (cache["Xval"], data.yva, data.lva),
        "test": (cache["Xtest"], data.yte, data.lte),
    }


def _make_loaders(
    data: base.Data,
    cache: dict[str, np.ndarray],
    spec: RunSpec,
    config: Config,
    train_shuffle: bool,
) -> dict[str, DataLoader]:
    return {
        split: _loader(
            *partition,
            config.batch_size,
            train_shuffle if split == "train" else False,
            base.dseed(spec.seed, "exp5_3", split, "loader"),
        )
        for split, partition in _partitions(data, cache).items()
    }


def _masked_sum(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = base.mask(lengths, sequence.shape[1]).to(sequence.dtype).unsqueeze(-1)
    return (sequence * valid).sum(dim=1)


def _valid_endpoint(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return exp52._valid_endpoint(sequence, lengths)


def _relative_bin_means(
    sequence: torch.Tensor,
    lengths: torch.Tensor,
    n_bins: int = N_REL,
) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(f"Expected (batch, time, features), got {tuple(sequence.shape)}")
    if torch.any(lengths <= 0) or torch.any(lengths > sequence.shape[1]):
        raise ValueError("Relative-bin means require valid lengths inside the sequence")
    samples: list[torch.Tensor] = []
    for sample_index in range(sequence.shape[0]):
        length = int(lengths[sample_index].item())
        bins: list[torch.Tensor] = []
        for bin_index in range(n_bins):
            start = int(math.floor(bin_index * length / n_bins))
            stop = int(math.floor((bin_index + 1) * length / n_bins))
            if stop <= start:
                start = min(start, length - 1)
                stop = start + 1
            bins.append(sequence[sample_index, start:stop].mean(dim=0))
        samples.append(torch.stack(bins, dim=0))
    return torch.stack(samples, dim=0)


def _phase_targets(batch_size: int) -> np.ndarray:
    return np.tile(np.arange(N_REL, dtype=np.int64), int(batch_size))


def _shuffle_valid_prefix(X: np.ndarray, lengths: np.ndarray, seed: int) -> np.ndarray:
    shuffled = np.array(X, copy=True)
    generator = np.random.default_rng(seed)
    for sample_index, raw_length in enumerate(lengths):
        length = int(raw_length)
        if length > 1:
            shuffled[sample_index, :length] = X[
                sample_index, generator.permutation(length)
            ]
    return shuffled


def _shuffle_cache(
    data: base.Data,
    cache: dict[str, np.ndarray],
    seed: int,
    replicate: int,
) -> dict[str, np.ndarray]:
    lengths_by_split = {"train": data.ltr, "val": data.lva, "test": data.lte}
    return {
        f"X{split}": _shuffle_valid_prefix(
            cache[f"X{split}"],
            lengths,
            base.dseed(seed, "exp5_3", "temporal_shuffle", replicate, split),
        )
        for split, lengths in lengths_by_split.items()
    }


def _alpha_layout(
    shifts: tuple[int, ...],
    width: int = TEMPORAL_WIDTH,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if not shifts:
        return torch.zeros(width, dtype=torch.float32), ()
    assigned = tuple(shifts[index % len(shifts)] for index in range(width))
    counts = tuple(assigned.count(shift) for shift in shifts)
    alpha = torch.tensor(
        [decay_from_shift(shift) for shift in assigned], dtype=torch.float32
    )
    return alpha, counts


class ContextualEvidenceNet(nn.Module):
    """Causal contextualizer plus a signed non-leaky evidence accumulator."""

    def __init__(self, condition: str, n_classes: int, fs: float) -> None:
        super().__init__()
        if condition not in CONDITION_BY_NAME:
            raise ValueError(f"Unknown condition: {condition}")
        self.condition = condition
        self.definition = CONDITION_BY_NAME[condition]
        self.family = self.definition.family
        self.fs = float(fs)
        self.hidden_width = TEMPORAL_WIDTH
        self.n_classes = int(n_classes)
        self.beta = float(self.definition.beta or 0.0)
        self.spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

        self.input_projection = (
            None
            if self.family == "direct"
            else nn.Linear(LOCAL_WIDTH, TEMPORAL_WIDTH, bias=False)
        )
        self.recurrent = (
            nn.Linear(TEMPORAL_WIDTH, TEMPORAL_WIDTH, bias=False)
            if self.family in {"rsnn", "factorized"}
            else None
        )
        self.gate_projection = (
            nn.Linear(TEMPORAL_WIDTH, LOCAL_WIDTH, bias=True)
            if self.family == "factorized"
            else None
        )
        self.evidence_head = nn.Linear(LOCAL_WIDTH, n_classes, bias=False)

        alpha, counts = _alpha_layout(self.definition.shifts_syn)
        self.register_buffer("alpha_vector", alpha)
        self.alpha_counts = counts

    @staticmethod
    def _reset_membrane(membrane: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        if RESET == "subtract":
            return membrane - spike * THRESHOLD
        if RESET == "zero":
            return membrane * (1.0 - spike)
        if RESET == "none":
            return membrane
        raise ValueError(f"Unsupported reset mechanism: {RESET}")

    def _initial_state(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (x.shape[0], TEMPORAL_WIDTH)
        synaptic = torch.zeros(shape, dtype=x.dtype, device=x.device)
        membrane = torch.zeros_like(synaptic)
        spike = torch.zeros_like(synaptic)
        return synaptic, membrane, spike

    def _context_step(
        self,
        local_t: torch.Tensor,
        synaptic: torch.Tensor,
        membrane: torch.Tensor,
        previous_spike: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.input_projection is None:
            raise RuntimeError("Direct condition has no contextual state step")
        current = self.input_projection(local_t)
        if self.recurrent is not None:
            current = current + self.recurrent(previous_spike)
        if self.family == "syn":
            synaptic = self.alpha_vector.to(current) * synaptic + current
            drive = synaptic
        else:
            synaptic = torch.zeros_like(synaptic)
            drive = current
        membrane = self.beta * membrane + drive
        spike = self.spike_grad(membrane - THRESHOLD)
        membrane = self._reset_membrane(membrane, spike)
        return spike, synaptic, membrane

    def _step_outputs(
        self,
        local_t: torch.Tensor,
        synaptic: torch.Tensor,
        membrane: torch.Tensor,
        previous_spike: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.family == "direct":
            contextual = local_t
            spike = local_t
            synaptic = torch.zeros_like(local_t)
            membrane = local_t
            gate = torch.ones_like(local_t)
        else:
            spike, synaptic, membrane = self._context_step(
                local_t, synaptic, membrane, previous_spike
            )
            if self.family == "factorized":
                if self.gate_projection is None:
                    raise RuntimeError("Factorized condition requires a gate")
                gate = 2.0 * torch.sigmoid(self.gate_projection(membrane))
                contextual = gate * local_t
            else:
                gate = torch.ones_like(local_t)
                contextual = spike
        evidence = self.evidence_head(contextual)
        return contextual, evidence, spike, synaptic, membrane, gate

    def forward_accumulator(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        reset_state_each_step: bool = False,
    ) -> torch.Tensor:
        if torch.any(lengths <= 0) or torch.any(lengths > x.shape[1]):
            raise ValueError("Invalid sequence lengths for accumulator")
        synaptic, membrane, previous_spike = self._initial_state(x)
        accumulator = torch.zeros(
            x.shape[0], self.n_classes, dtype=x.dtype, device=x.device
        )
        for timestep in range(x.shape[1]):
            if reset_state_each_step and self.family != "direct":
                synaptic.zero_()
                membrane.zero_()
                previous_spike.zero_()
            _, evidence, spike, synaptic, membrane, _ = self._step_outputs(
                x[:, timestep], synaptic, membrane, previous_spike
            )
            valid = (timestep < lengths).to(evidence.dtype).unsqueeze(1)
            accumulator = accumulator + valid * evidence
            previous_spike = spike
        return accumulator

    @staticmethod
    def normalized_logits(
        accumulator: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return LOGIT_GAIN * accumulator / lengths.to(accumulator.dtype).unsqueeze(1)

    def loss_logits(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        y: torch.Tensor,
        reset_state_each_step: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        accumulator = self.forward_accumulator(x, lengths, reset_state_each_step)
        logits = self.normalized_logits(accumulator, lengths)
        return F.cross_entropy(logits, y), logits, accumulator

    def forward_trajectory(
        self,
        x: torch.Tensor,
        reset_state_each_step: bool = False,
    ) -> Trajectory:
        synaptic, membrane, previous_spike = self._initial_state(x)
        contextual_parts: list[torch.Tensor] = []
        evidence_parts: list[torch.Tensor] = []
        spike_parts: list[torch.Tensor] = []
        membrane_parts: list[torch.Tensor] = []
        synaptic_parts: list[torch.Tensor] = []
        gate_parts: list[torch.Tensor] = []
        for timestep in range(x.shape[1]):
            if reset_state_each_step and self.family != "direct":
                synaptic.zero_()
                membrane.zero_()
                previous_spike.zero_()
            contextual, evidence, spike, synaptic, membrane, gate = self._step_outputs(
                x[:, timestep], synaptic, membrane, previous_spike
            )
            contextual_parts.append(contextual)
            evidence_parts.append(evidence)
            spike_parts.append(spike)
            membrane_parts.append(membrane)
            synaptic_parts.append(synaptic)
            gate_parts.append(gate)
            previous_spike = spike
        return Trajectory(
            contextual=torch.stack(contextual_parts, dim=1),
            evidence=torch.stack(evidence_parts, dim=1),
            spikes=torch.stack(spike_parts, dim=1),
            membranes=torch.stack(membrane_parts, dim=1),
            synaptic=torch.stack(synaptic_parts, dim=1),
            gates=torch.stack(gate_parts, dim=1),
        )

    def tail_event_count(
        self,
        endpoint_synaptic: torch.Tensor,
        endpoint_membrane: torch.Tensor,
        endpoint_spike: torch.Tensor,
        n_tail_steps: int,
    ) -> float:
        if self.family == "direct":
            return 0.0
        synaptic = endpoint_synaptic
        membrane = endpoint_membrane
        previous_spike = endpoint_spike
        local_t = torch.zeros(
            endpoint_spike.shape[0],
            LOCAL_WIDTH,
            dtype=endpoint_spike.dtype,
            device=endpoint_spike.device,
        )
        total = 0.0
        for _ in range(n_tail_steps):
            spike, synaptic, membrane = self._context_step(
                local_t, synaptic, membrane, previous_spike
            )
            total += float(spike.sum().item())
            previous_spike = spike
        return total


def _initialize_model(
    spec: RunSpec,
    n_classes: int,
    fs: float,
    device: torch.device,
) -> ContextualEvidenceNet:
    base.seed_all(base.dseed(spec.seed, "exp5_3", "constructor"))
    model = ContextualEvidenceNet(spec.condition, n_classes, fs).to(device)
    if model.input_projection is not None:
        base.seed_all(base.dseed(spec.seed, "exp5_3", "input_projection"))
        model.input_projection.reset_parameters()
    if model.recurrent is not None:
        base.seed_all(base.dseed(spec.seed, "exp5_3", "recurrent"))
        model.recurrent.reset_parameters()
    if model.gate_projection is not None:
        base.seed_all(base.dseed(spec.seed, "exp5_3", "gate_projection"))
        model.gate_projection.reset_parameters()
        if model.gate_projection.bias is not None:
            nn.init.zeros_(model.gate_projection.bias)
    base.seed_all(base.dseed(spec.seed, "exp5_3", "evidence_head"))
    model.evidence_head.reset_parameters()
    return model


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return base.metrics(y_true, y_pred)


def evaluate_native(
    model: ContextualEvidenceNet,
    loader: DataLoader,
    device: torch.device,
    fs: float,
    reset_state_each_step: bool = False,
    diagnostics: bool = False,
) -> dict[str, float]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    valid_events = 0.0
    valid_seconds = 0.0
    tail_events = 0.0
    accumulator_norm_sum = 0.0
    evidence_abs_sum = 0.0
    valid_steps = 0.0
    tail_steps = int(round(TAIL_SECONDS * float(fs)))

    with torch.no_grad():
        for local, y, lengths in loader:
            local = local.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            if diagnostics:
                trajectory = model.forward_trajectory(local, reset_state_each_step)
                accumulator = _masked_sum(trajectory.evidence, lengths)
                logits = model.normalized_logits(accumulator, lengths)
                loss = F.cross_entropy(logits, y)
                valid = base.mask(lengths, trajectory.spikes.shape[1]).to(
                    trajectory.spikes.dtype
                ).unsqueeze(-1)
                valid_events += float((trajectory.spikes * valid).sum().item())
                valid_seconds += float(lengths.sum().item()) / float(fs)
                valid_steps += float(lengths.sum().item())
                evidence_abs_sum += float(
                    (trajectory.evidence.abs() * valid).sum().item()
                )
                tail_events += model.tail_event_count(
                    _valid_endpoint(trajectory.synaptic, lengths),
                    _valid_endpoint(trajectory.membranes, lengths),
                    _valid_endpoint(trajectory.spikes, lengths),
                    tail_steps,
                )
            else:
                loss, logits, accumulator = model.loss_logits(
                    local, lengths, y, reset_state_each_step
                )
            n = len(y)
            n_total += n
            loss_sum += float(loss.item()) * n
            accumulator_norm_sum += float(
                torch.linalg.vector_norm(accumulator, dim=1).sum().item()
            )
            y_true_parts.append(y.cpu().numpy())
            y_pred_parts.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    out = _classification_metrics(y_true, y_pred)
    out["loss"] = loss_sum / max(n_total, 1)
    out["accumulator_l2_mean"] = accumulator_norm_sum / max(n_total, 1)
    if diagnostics:
        out["hidden_events_per_neuron_second"] = valid_events / max(
            valid_seconds * TEMPORAL_WIDTH, EPS
        )
        out["hidden_tail_event_fraction"] = tail_events / max(
            valid_events + tail_events, EPS
        )
        out["hidden_tail_events_per_neuron_second"] = tail_events / max(
            n_total * TAIL_SECONDS * TEMPORAL_WIDTH, EPS
        )
        out["evidence_abs_per_valid_step"] = evidence_abs_sum / max(
            valid_steps * model.n_classes, EPS
        )
    return out


def _probe_metrics(
    features: dict[str, dict[str, np.ndarray]],
    probe_name: str,
    seed: int,
    namespace: str,
) -> dict[str, object]:
    return probe_utils._raw_probe_metrics(
        features["train"][probe_name],
        features["train"]["y"],
        features["val"][probe_name],
        features["val"]["y"],
        features["test"][probe_name],
        features["test"]["y"],
        base.dseed(seed, "exp5_3", namespace, probe_name, "linear_probe"),
    )


def collect_features(
    model: ContextualEvidenceNet,
    data: base.Data,
    cache: dict[str, np.ndarray],
    spec: RunSpec,
    config: Config,
    reset_state_each_step: bool = False,
    full: bool = True,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
]:
    classification: dict[str, dict[str, np.ndarray]] = {}
    phase: dict[str, dict[str, np.ndarray]] = {}
    loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)
    device = torch.device(config.device)
    model.eval()
    with torch.no_grad():
        for split, loader in loaders.items():
            y_parts: list[np.ndarray] = []
            whole_parts: list[np.ndarray] = []
            fixed_parts: list[np.ndarray] = []
            relative_parts: list[np.ndarray] = []
            uend_parts: list[np.ndarray] = []
            phase_parts: dict[str, list[np.ndarray]] = {
                probe: [] for probe in PHASE_PROBES
            }
            phase_y_parts: list[np.ndarray] = []
            for local, y, lengths in loader:
                local = local.to(device)
                lengths_device = lengths.to(device)
                trajectory = model.forward_trajectory(local, reset_state_each_step)
                whole_parts.append(
                    _masked_sum(trajectory.contextual, lengths_device).cpu().numpy()
                )
                if full:
                    fixed_parts.append(
                        base.fixed_counts(
                            trajectory.contextual, lengths_device, data.bin_steps
                        ).flatten(start_dim=1).cpu().numpy()
                    )
                    relative_parts.append(
                        base.relative_counts(
                            trajectory.contextual, lengths_device, N_REL
                        ).flatten(start_dim=1).cpu().numpy()
                    )
                    uend_parts.append(
                        _valid_endpoint(
                            trajectory.membranes, lengths_device
                        ).cpu().numpy()
                    )
                    phase_sequences = {
                        "phase_contextual": trajectory.contextual,
                        "phase_synaptic_state": trajectory.synaptic,
                        "phase_membrane": trajectory.membranes,
                        "phase_spike": trajectory.spikes,
                        "phase_gate": trajectory.gates,
                    }
                    for probe_name, sequence in phase_sequences.items():
                        means = _relative_bin_means(sequence, lengths_device, N_REL)
                        phase_parts[probe_name].append(
                            means.flatten(start_dim=0, end_dim=1).cpu().numpy()
                        )
                    phase_y_parts.append(_phase_targets(len(y)))
                y_parts.append(y.numpy())
            class_split: dict[str, np.ndarray] = {
                "y": np.concatenate(y_parts),
                "hidden_whole_count": np.concatenate(whole_parts),
            }
            if full:
                class_split.update(
                    {
                        "hidden_fixed250_ordered": np.concatenate(fixed_parts),
                        "hidden_relative10_ordered": np.concatenate(relative_parts),
                        "hidden_uend": np.concatenate(uend_parts),
                    }
                )
                phase[split] = {
                    "y": np.concatenate(phase_y_parts),
                    **{
                        name: np.concatenate(parts)
                        for name, parts in phase_parts.items()
                    },
                }
            classification[split] = class_split
    return classification, phase


def _local_phase_features(
    data: base.Data,
    cache: dict[str, np.ndarray],
    config: Config,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    device = torch.device(config.device)
    for split, (X, _, lengths) in _partitions(data, cache).items():
        loader = _loader(
            X,
            np.zeros(len(X), dtype=np.int64),
            lengths,
            config.batch_size,
            False,
            base.dseed(seed, "exp5_3", "local_phase", split),
        )
        feature_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        with torch.no_grad():
            for local, _, batch_lengths in loader:
                local = local.to(device)
                batch_lengths = batch_lengths.to(device)
                means = _relative_bin_means(local, batch_lengths, N_REL)
                feature_parts.append(
                    means.flatten(start_dim=0, end_dim=1).cpu().numpy()
                )
                y_parts.append(_phase_targets(len(local)))
        out[split] = {
            "y": np.concatenate(y_parts),
            "phase_local": np.concatenate(feature_parts),
        }
    return out


def _source_config(config: Config) -> exp52.Config:
    return exp52.Config(
        repo_root=config.repo_root,
        results_dir=source_results_dir(config.repo_root),
        device=config.device,
        epochs=exp52.EPOCHS,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def prepare_local_seed(
    seed: int,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Unknown Exp5.3 seed: {seed}")
    source_config = _source_config(config)
    exp52.prepare_local_seed(seed, data, source_config, force=force)
    cache = exp52.load_local_cache(seed, data, source_config)
    destination = local_phase_path(config.results_dir, seed)
    if destination.exists() and not force:
        return destination
    features = _local_phase_features(data, cache, config, seed)
    metrics = _probe_metrics(features, "phase_local", seed, "local_phase")
    _save_json(
        destination,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "source_experiment_id": exp52.EXPERIMENT_ID,
            "source_protocol_version": exp52.PROTOCOL_VERSION,
            "source_cache": str(
                exp52.local_cache_path(source_config.results_dir, seed).relative_to(
                    config.repo_root
                )
            ),
            "probe_type": "phase_local",
            **metrics,
        },
    )
    return destination


def load_local_cache(
    seed: int,
    data: base.Data,
    config: Config,
) -> dict[str, np.ndarray]:
    return exp52.load_local_cache(seed, data, _source_config(config))


def parameter_counts(model: ContextualEvidenceNet) -> dict[str, int]:
    counts = {
        "input_projection": 0,
        "recurrent": 0,
        "gate_projection": 0,
        "evidence_head": int(
            sum(parameter.numel() for parameter in model.evidence_head.parameters())
        ),
    }
    if model.input_projection is not None:
        counts["input_projection"] = int(
            sum(parameter.numel() for parameter in model.input_projection.parameters())
        )
    if model.recurrent is not None:
        counts["recurrent"] = int(
            sum(parameter.numel() for parameter in model.recurrent.parameters())
        )
    if model.gate_projection is not None:
        counts["gate_projection"] = int(
            sum(parameter.numel() for parameter in model.gate_projection.parameters())
        )
    counts["trainable_total"] = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    return counts


def provenance(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    model: ContextualEvidenceNet,
) -> dict[str, object]:
    definition = spec.definition
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": spec.seed,
        "split_seed": base.SPLIT_SEED,
        "condition": definition.name,
        "architecture_family": definition.family,
        "beta": definition.beta,
        "tau_mem_ms": (
            tau_ms_from_decay(definition.beta, data.fs)
            if definition.beta is not None
            else None
        ),
        "shift_syn": definition.shifts_syn,
        "alpha_syn": tuple(decay_from_shift(shift) for shift in definition.shifts_syn),
        "tau_syn_ms": tuple(
            tau_ms_from_shift(shift, data.fs) for shift in definition.shifts_syn
        ),
        "alpha_neuron_counts": model.alpha_counts,
        "temporal_width": TEMPORAL_WIDTH,
        "local_width": LOCAL_WIDTH,
        "local_source_experiment": exp52.EXPERIMENT_ID,
        "local_source_protocol": exp52.PROTOCOL_VERSION,
        "local_source_frozen": True,
        "sampling_rate_hz": float(data.fs),
        "objective": "final_accumulated_ce",
        "objective_formula": "CE(5 * sum_valid(e_t) / T_valid, y)",
        "evidence_accumulator": (
            "12-D signed non-leaky non-spiking sum; no evidence-head bias"
        ),
        "checkpoint_selection": (
            "max validation native accumulator BA; tie-break validation CE"
        ),
        "threshold": THRESHOLD,
        "reset": RESET,
        "logit_gain_training_only": LOGIT_GAIN,
        "shuffle_replicates": SHUFFLE_REPLICATES,
        "tail_seconds": TAIL_SECONDS,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "parameter_counts": parameter_counts(model),
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
    }


def train_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    cache = load_local_cache(spec.seed, data, config)
    train_loaders = _make_loaders(data, cache, spec, config, train_shuffle=True)
    eval_loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _initialize_model(spec, len(data.labels), data.fs, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_ba = -np.inf
    best_val_loss = np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for local, y, lengths in train_loaders["train"]:
            local = local.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, logits, _ = model.loss_logits(local, lengths, y)
            loss.backward()
            optimizer.step()
            n = len(y)
            train_n += n
            train_loss_sum += float(loss.item()) * n
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = _classification_metrics(
            np.concatenate(train_true), np.concatenate(train_pred)
        )
        val_metrics = evaluate_native(
            model, eval_loaders["val"], device, data.fs, diagnostics=False
        )
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_n, 1),
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_loss": val_loss,
                "val_balanced_accuracy": val_ba,
            }
        )
        improved = val_ba > best_val_ba + 1e-12 or (
            abs(val_ba - best_val_ba) <= 1e-12
            and val_loss < best_val_loss - 1e-12
        )
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    model.load_state_dict(best_state)
    final_native = {
        split: evaluate_native(model, loader, device, data.fs, diagnostics=True)
        for split, loader in eval_loaders.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "provenance": provenance(spec, data, config, model),
            "result": {
                "best_epoch": best_epoch,
                "best_val_balanced_accuracy": best_val_ba,
                "best_val_loss": best_val_loss,
                "native": final_native,
            },
            "state_dict": best_state,
        },
        destination,
    )
    hpath = history_path(config.results_dir, spec)
    hpath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hpath, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[ContextualEvidenceNet, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.3 checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise ValueError(f"Wrong Exp5.3 checkpoint identity: {path}")
    if payload.get("spec") != asdict(spec):
        raise ValueError(f"Checkpoint spec mismatch: {path}")
    model = ContextualEvidenceNet(spec.condition, len(data.labels), data.fs).to(
        torch.device(config.device)
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _native_all_splits(
    model: ContextualEvidenceNet,
    loaders: dict[str, DataLoader],
    device: torch.device,
    fs: float,
    reset_state_each_step: bool,
) -> dict[str, dict[str, float]]:
    return {
        split: evaluate_native(
            model,
            loader,
            device,
            fs,
            reset_state_each_step=reset_state_each_step,
            diagnostics=True,
        )
        for split, loader in loaders.items()
    }


def evaluate_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    cache = load_local_cache(spec.seed, data, config)
    model, checkpoint = load_model(spec, data, config)
    device = torch.device(config.device)
    loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)
    native = _native_all_splits(model, loaders, device, data.fs, False)

    normal_features, phase_features = collect_features(
        model, data, cache, spec, config, reset_state_each_step=False, full=True
    )
    probes = [
        {
            "probe_type": probe,
            **_probe_metrics(normal_features, probe, spec.seed, "ordered"),
        }
        for probe in CLASSIFICATION_PROBES
    ]
    phase_probes = [
        {
            "probe_type": probe,
            **_probe_metrics(phase_features, probe, spec.seed, "phase"),
        }
        for probe in PHASE_PROBES
    ]

    reset_native = _native_all_splits(model, loaders, device, data.fs, True)
    reset_features, _ = collect_features(
        model, data, cache, spec, config, reset_state_each_step=True, full=False
    )
    reset_probe = {
        "probe_type": "hidden_whole_count",
        **_probe_metrics(
            reset_features,
            "hidden_whole_count",
            spec.seed,
            "state_reset",
        ),
    }
    ablations: list[dict[str, object]] = [
        {
            "ablation": "state_reset",
            "replicate": 0,
            "native": reset_native,
            "hidden_whole_count_probe": reset_probe,
        }
    ]

    for replicate in range(SHUFFLE_REPLICATES):
        shuffled_cache = _shuffle_cache(data, cache, spec.seed, replicate)
        shuffled_loaders = _make_loaders(
            data, shuffled_cache, spec, config, train_shuffle=False
        )
        shuffled_native = _native_all_splits(
            model, shuffled_loaders, device, data.fs, False
        )
        shuffled_features, _ = collect_features(
            model,
            data,
            shuffled_cache,
            spec,
            config,
            reset_state_each_step=False,
            full=False,
        )
        shuffled_probe = {
            "probe_type": "hidden_whole_count",
            **_probe_metrics(
                shuffled_features,
                "hidden_whole_count",
                spec.seed,
                f"temporal_shuffle_{replicate}",
            ),
        }
        ablations.append(
            {
                "ablation": "temporal_shuffle",
                "replicate": replicate,
                "native": shuffled_native,
                "hidden_whole_count_probe": shuffled_probe,
            }
        )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "provenance": checkpoint["provenance"],
        "best_epoch": checkpoint["result"]["best_epoch"],
        "native": native,
        "probes": probes,
        "phase_probes": phase_probes,
        "ablations": ablations,
    }
    _save_json(destination, payload)
    return payload


def run_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    train_one(spec, data, config, force=force)
    return evaluate_one(spec, data, config, force=force)


def _condition_row(condition: Condition, fs: float) -> dict[str, object]:
    return {
        "condition": condition.name,
        "architecture_family": condition.family,
        "beta": condition.beta,
        "tau_mem_ms": (
            tau_ms_from_decay(condition.beta, fs)
            if condition.beta is not None
            else np.nan
        ),
        "shift_syn": json.dumps(condition.shifts_syn),
        "tau_syn_ms": json.dumps(
            [tau_ms_from_shift(shift, fs) for shift in condition.shifts_syn]
        ),
    }


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    history_parts: list[pd.DataFrame] = []

    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp5.3 evaluation: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("spec") != asdict(spec)
        ):
            raise ValueError(f"Evaluation identity mismatch: {path}")
        condition = spec.definition
        fs = float(payload["provenance"]["sampling_rate_hz"])
        row: dict[str, object] = {
            **_condition_row(condition, fs),
            "seed": spec.seed,
            "best_epoch": payload["best_epoch"],
            "trainable_parameters": payload["provenance"]["parameter_counts"][
                "trainable_total"
            ],
        }
        for split in ("train", "val", "test"):
            for metric, value in payload["native"][split].items():
                row[f"native_{split}_{metric}"] = value
        rows.append(row)

        for probe in payload["probes"]:
            probe_rows.append(
                {
                    **_condition_row(condition, fs),
                    "seed": spec.seed,
                    "probe_type": probe["probe_type"],
                    "probe_C": probe["probe_C"],
                    "val_ba": probe["probe_val_balanced_accuracy"],
                    "test_ba": probe["probe_test_balanced_accuracy"],
                    "test_accuracy": probe["probe_test_accuracy"],
                    "test_macro_f1": probe["probe_test_macro_f1"],
                }
            )
        for probe in payload["phase_probes"]:
            phase_rows.append(
                {
                    **_condition_row(condition, fs),
                    "seed": spec.seed,
                    "probe_type": probe["probe_type"],
                    "probe_C": probe["probe_C"],
                    "val_ba": probe["probe_val_balanced_accuracy"],
                    "test_ba": probe["probe_test_balanced_accuracy"],
                    "test_accuracy": probe["probe_test_accuracy"],
                    "test_macro_f1": probe["probe_test_macro_f1"],
                }
            )

        ordered_probe = next(
            probe
            for probe in payload["probes"]
            if probe["probe_type"] == "hidden_whole_count"
        )
        ablation_rows.append(
            {
                **_condition_row(condition, fs),
                "seed": spec.seed,
                "ablation": "ordered",
                "replicate": 0,
                "native_val_ba": payload["native"]["val"]["balanced_accuracy"],
                "native_test_ba": payload["native"]["test"]["balanced_accuracy"],
                "hidden_whole_count_val_ba": ordered_probe[
                    "probe_val_balanced_accuracy"
                ],
                "hidden_whole_count_test_ba": ordered_probe[
                    "probe_test_balanced_accuracy"
                ],
            }
        )
        for ablation in payload["ablations"]:
            probe = ablation["hidden_whole_count_probe"]
            ablation_rows.append(
                {
                    **_condition_row(condition, fs),
                    "seed": spec.seed,
                    "ablation": ablation["ablation"],
                    "replicate": ablation["replicate"],
                    "native_val_ba": ablation["native"]["val"][
                        "balanced_accuracy"
                    ],
                    "native_test_ba": ablation["native"]["test"][
                        "balanced_accuracy"
                    ],
                    "hidden_whole_count_val_ba": probe[
                        "probe_val_balanced_accuracy"
                    ],
                    "hidden_whole_count_test_ba": probe[
                        "probe_test_balanced_accuracy"
                    ],
                }
            )

        hpath = history_path(root, spec)
        if not hpath.exists():
            raise FileNotFoundError(f"Missing Exp5.3 history: {hpath}")
        history = pd.read_csv(hpath)
        history.insert(0, "seed", spec.seed)
        history.insert(0, "architecture_family", condition.family)
        history.insert(0, "condition", condition.name)
        history_parts.append(history)

    local_rows: list[dict[str, object]] = []
    source_root = source_results_dir(repo_root)
    for seed in SEEDS:
        reference_path = exp52.local_reference_path(source_root, seed)
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing Exp5.2 local reference: {reference_path}")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference.get("protocol_version") != exp52.PROTOCOL_VERSION:
            raise ValueError(f"Wrong Exp5.2 local reference identity: {reference_path}")
        for probe in reference["probes"]:
            local_rows.append(
                {
                    "seed": seed,
                    "probe_type": probe["probe_type"],
                    "probe_C": probe["probe_C"],
                    "val_ba": probe["probe_val_balanced_accuracy"],
                    "test_ba": probe["probe_test_balanced_accuracy"],
                    "test_accuracy": probe["probe_test_accuracy"],
                    "test_macro_f1": probe["probe_test_macro_f1"],
                }
            )
        phase_path = local_phase_path(root, seed)
        if not phase_path.exists():
            raise FileNotFoundError(f"Missing Exp5.3 local phase probe: {phase_path}")
        phase = json.loads(phase_path.read_text(encoding="utf-8"))
        if phase.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"Wrong local phase identity: {phase_path}")
        local_rows.append(
            {
                "seed": seed,
                "probe_type": "phase_local",
                "probe_C": phase["probe_C"],
                "val_ba": phase["probe_val_balanced_accuracy"],
                "test_ba": phase["probe_test_balanced_accuracy"],
                "test_accuracy": phase["probe_test_accuracy"],
                "test_macro_f1": phase["probe_test_macro_f1"],
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "runs": root / "runs.csv",
        "histories": root / "histories.csv",
        "probe_runs": root / "probe_runs.csv",
        "phase_probe_runs": root / "phase_probe_runs.csv",
        "ablation_runs": root / "ablation_runs.csv",
        "local_reference": root / "local_reference.csv",
        "manifest": root / "manifest.json",
    }
    pd.DataFrame(rows).to_csv(outputs["runs"], index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(
        outputs["histories"], index=False
    )
    pd.DataFrame(probe_rows).to_csv(outputs["probe_runs"], index=False)
    pd.DataFrame(phase_rows).to_csv(outputs["phase_probe_runs"], index=False)
    pd.DataFrame(ablation_rows).to_csv(outputs["ablation_runs"], index=False)
    pd.DataFrame(local_rows).to_csv(outputs["local_reference"], index=False)
    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "expected_runs": len(run_specs()),
            "conditions": [asdict(condition) for condition in CONDITIONS],
            "seeds": SEEDS,
            "source_experiment_id": exp52.EXPERIMENT_ID,
            "source_protocol_version": exp52.PROTOCOL_VERSION,
            "primary_objective": "final_accumulated_ce",
            "primary_metric": (
                "native validation/test balanced accuracy from signed non-leaky accumulator"
            ),
            "selection_policy": (
                "checkpoint by validation BA; notebook selects best single and multi "
                "condition by mean validation BA only"
            ),
            "aggregation_policy": (
                "finalizer only concatenates completed artifacts; notebook computes "
                "summaries, paired effects, validation selections, and plots"
            ),
            "shuffle_replicates": SHUFFLE_REPLICATES,
            "files": {
                name: path.name for name, path in outputs.items() if name != "manifest"
            },
        },
    )
    return outputs


def _config_from_args(args: argparse.Namespace) -> Config:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    return Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threads=args.threads,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 5.3 frozen-local synaptic history contextualization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo-root", type=str, default=None)
        subparser.add_argument("--device", type=str, default="cpu")
        subparser.add_argument("--threads", type=int, default=1)
        subparser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        subparser.add_argument("--epochs", type=int, default=EPOCHS)
        subparser.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare-local")
    common(prepare)
    prepare.add_argument("--array-task-id", type=int, required=True)

    run = subparsers.add_parser("run-one")
    common(run)
    run.add_argument("--array-task-id", type=int, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--repo-root", type=str, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "finalize":
        repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return

    config = _config_from_args(args)
    data = base.prepare_data(config.repo_root)
    if args.command == "prepare-local":
        task_id = int(args.array_task_id)
        if not 0 <= task_id < len(SEEDS):
            raise IndexError(f"prepare-local task {task_id} outside 0..{len(SEEDS)-1}")
        print(prepare_local_seed(SEEDS[task_id], data, config, force=args.force))
        return
    if args.command == "run-one":
        specs = run_specs()
        task_id = int(args.array_task_id)
        if not 0 <= task_id < len(specs):
            raise IndexError(f"run-one task {task_id} outside 0..{len(specs)-1}")
        payload = run_one(specs[task_id], data, config, force=args.force)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
