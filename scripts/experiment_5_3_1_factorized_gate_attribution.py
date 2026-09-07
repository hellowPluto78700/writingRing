from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from snntorch import surrogate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from scripts import experiment_5_2_frozen_local_tauR_sweep as exp52
from scripts import experiment_5_3_synaptic_contextual_evidence as exp53


base = exp53.base
probe_utils = exp53.probe_utils

EXPERIMENT_ID = "experiment_5_3_1_factorized_gate_attribution"
PROTOCOL_VERSION = "factorized_gate_causal_attribution_v1"
SEEDS = exp53.SEEDS
LOCAL_WIDTH = exp53.LOCAL_WIDTH
CONTEXT_WIDTH = exp53.TEMPORAL_WIDTH
N_REL = exp53.N_REL
THRESHOLD = exp53.THRESHOLD
RESET = exp53.RESET
SURROGATE_SLOPE = exp53.SURROGATE_SLOPE
EPOCHS = exp53.EPOCHS
BATCH_SIZE = exp53.BATCH_SIZE
LR = exp53.LR
WEIGHT_DECAY = exp53.WEIGHT_DECAY
LOGIT_GAIN = exp53.LOGIT_GAIN
SHUFFLE_REPLICATES = exp53.SHUFFLE_REPLICATES
EPS = exp53.EPS

CLASSIFICATION_PROBES = (
    "hidden_whole_count",
    "hidden_fixed250_ordered",
    "hidden_relative10_ordered",
)
PHASE_PROBES = (
    "phase_contextual",
    "phase_membrane",
    "phase_gate",
)


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    beta: float
    recurrent: bool
    stateful: bool


CONDITIONS = (
    Condition("factorized_ff_gate", "ff_gate", 0.0, False, False),
    Condition("factorized_lif22", "lif_gate", 0.50, False, True),
    Condition("factorized_lif242", "lif_gate", 0.9375, False, True),
    Condition("factorized_rsnn22", "rsnn_gate", 0.50, True, True),
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
            raise ValueError(f"Unknown Exp5.3.1 condition: {self.condition}") from error


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
    gates: torch.Tensor


def find_repo_root(start: Path | None = None) -> Path:
    return exp53.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_results_dir(repo_root: Path) -> Path:
    return exp52.results_dir(repo_root)


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_specs() -> list[RunSpec]:
    return [RunSpec(condition.name, seed) for condition in CONDITIONS for seed in SEEDS]


def tau_ms_from_beta(beta: float, fs: float) -> float:
    if beta == 0.0:
        return 0.0
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    return -(1000.0 / float(fs)) / math.log(float(beta))


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
            base.dseed(spec.seed, "exp5_3_1", split, "loader"),
        )
        for split, partition in _partitions(data, cache).items()
    }


def _masked_sum(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = base.mask(lengths, sequence.shape[1]).to(sequence.dtype).unsqueeze(-1)
    return (sequence * valid).sum(dim=1)


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
            base.dseed(seed, "exp5_3_1", "temporal_shuffle", replicate, split),
        )
        for split, lengths in lengths_by_split.items()
    }


class FactorizedGateNet(nn.Module):
    """Preserve local WHAT and vary only the mechanism that generates its gate."""

    def __init__(self, condition: str, n_classes: int, fs: float) -> None:
        super().__init__()
        if condition not in CONDITION_BY_NAME:
            raise ValueError(f"Unknown Exp5.3.1 condition: {condition}")
        self.condition = condition
        self.definition = CONDITION_BY_NAME[condition]
        self.family = self.definition.family
        self.beta = float(self.definition.beta)
        self.fs = float(fs)
        self.n_classes = int(n_classes)
        self.spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

        self.input_projection = nn.Linear(LOCAL_WIDTH, CONTEXT_WIDTH, bias=False)
        self.recurrent = (
            nn.Linear(CONTEXT_WIDTH, CONTEXT_WIDTH, bias=False)
            if self.definition.recurrent
            else None
        )
        self.gate_projection = nn.Linear(CONTEXT_WIDTH, LOCAL_WIDTH, bias=True)
        self.evidence_head = nn.Linear(LOCAL_WIDTH, n_classes, bias=False)

    @staticmethod
    def _reset_membrane(membrane: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        if RESET == "subtract":
            return membrane - spike * THRESHOLD
        if RESET == "zero":
            return membrane * (1.0 - spike)
        if RESET == "none":
            return membrane
        raise ValueError(f"Unsupported reset mechanism: {RESET}")

    def _initial_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (x.shape[0], CONTEXT_WIDTH)
        membrane = torch.zeros(shape, dtype=x.dtype, device=x.device)
        spike = torch.zeros_like(membrane)
        return membrane, spike

    def _step(
        self,
        local_t: torch.Tensor,
        membrane: torch.Tensor,
        previous_spike: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        drive = self.input_projection(local_t)
        if self.recurrent is not None:
            drive = drive + self.recurrent(previous_spike)
        membrane = self.beta * membrane + drive
        spike = self.spike_grad(membrane - THRESHOLD)
        membrane = self._reset_membrane(membrane, spike)
        gate = 2.0 * torch.sigmoid(self.gate_projection(membrane))
        contextual = gate * local_t
        evidence = self.evidence_head(contextual)
        return contextual, evidence, spike, membrane, gate

    def forward_trajectory(
        self,
        x: torch.Tensor,
        reset_state_each_step: bool = False,
    ) -> Trajectory:
        membrane, previous_spike = self._initial_state(x)
        contextual_parts: list[torch.Tensor] = []
        evidence_parts: list[torch.Tensor] = []
        spike_parts: list[torch.Tensor] = []
        membrane_parts: list[torch.Tensor] = []
        gate_parts: list[torch.Tensor] = []
        for timestep in range(x.shape[1]):
            if reset_state_each_step:
                membrane.zero_()
                previous_spike.zero_()
            contextual, evidence, spike, membrane, gate = self._step(
                x[:, timestep], membrane, previous_spike
            )
            contextual_parts.append(contextual)
            evidence_parts.append(evidence)
            spike_parts.append(spike)
            membrane_parts.append(membrane)
            gate_parts.append(gate)
            previous_spike = spike
        return Trajectory(
            contextual=torch.stack(contextual_parts, dim=1),
            evidence=torch.stack(evidence_parts, dim=1),
            spikes=torch.stack(spike_parts, dim=1),
            membranes=torch.stack(membrane_parts, dim=1),
            gates=torch.stack(gate_parts, dim=1),
        )

    def forward_accumulator(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        reset_state_each_step: bool = False,
    ) -> torch.Tensor:
        if torch.any(lengths <= 0) or torch.any(lengths > x.shape[1]):
            raise ValueError("Invalid sequence lengths for Exp5.3.1 accumulator")
        trajectory = self.forward_trajectory(x, reset_state_each_step)
        return _masked_sum(trajectory.evidence, lengths)

    @staticmethod
    def normalized_logits(
        accumulator: torch.Tensor,
        lengths: torch.Tensor,
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


def _initialize_model(
    spec: RunSpec,
    n_classes: int,
    fs: float,
    device: torch.device,
) -> FactorizedGateNet:
    base.seed_all(base.dseed(spec.seed, "exp5_3_1", "constructor"))
    model = FactorizedGateNet(spec.condition, n_classes, fs).to(device)

    base.seed_all(base.dseed(spec.seed, "exp5_3_1", "input_projection"))
    model.input_projection.reset_parameters()
    base.seed_all(base.dseed(spec.seed, "exp5_3_1", "gate_projection"))
    model.gate_projection.reset_parameters()
    if model.gate_projection.bias is not None:
        nn.init.zeros_(model.gate_projection.bias)
    base.seed_all(base.dseed(spec.seed, "exp5_3_1", "evidence_head"))
    model.evidence_head.reset_parameters()
    if model.recurrent is not None:
        base.seed_all(base.dseed(spec.seed, "exp5_3_1", "recurrent"))
        model.recurrent.reset_parameters()
    return model


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
        raise ValueError(f"Unknown Exp5.3.1 seed: {seed}")
    source_config = _source_config(config)
    exp52.prepare_local_seed(seed, data, source_config, force=force)
    cache = exp52.load_local_cache(seed, data, source_config)
    for split in ("train", "val", "test"):
        expected = len(getattr(data, {"train": "ytr", "val": "yva", "test": "yte"}[split]))
        if cache[f"X{split}"].shape[0] != expected:
            raise ValueError(f"Frozen local cache sample mismatch for seed {seed} split {split}")
    return exp52.local_cache_path(source_config.results_dir, seed)


def load_local_cache(
    seed: int,
    data: base.Data,
    config: Config,
) -> dict[str, np.ndarray]:
    return exp52.load_local_cache(seed, data, _source_config(config))


def evaluate_native(
    model: FactorizedGateNet,
    loader: DataLoader,
    device: torch.device,
    reset_state_each_step: bool = False,
) -> dict[str, float]:
    model.eval()
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    evidence_abs_sum = 0.0
    valid_steps = 0.0
    with torch.no_grad():
        for local, y, lengths in loader:
            local = local.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(local, reset_state_each_step)
            accumulator = _masked_sum(trajectory.evidence, lengths)
            logits = model.normalized_logits(accumulator, lengths)
            loss = F.cross_entropy(logits, y)
            valid = base.mask(lengths, local.shape[1]).to(local.dtype).unsqueeze(-1)
            evidence_abs_sum += float((trajectory.evidence.abs() * valid).sum().item())
            valid_steps += float(lengths.sum().item())
            n = len(y)
            n_total += n
            loss_sum += float(loss.item()) * n
            true_parts.append(y.cpu().numpy())
            pred_parts.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(pred_parts)
    out = base.metrics(y_true, y_pred)
    out["loss"] = loss_sum / max(n_total, 1)
    out["evidence_abs_per_valid_step"] = evidence_abs_sum / max(
        valid_steps * model.n_classes, EPS
    )
    return out


def collect_features(
    model: FactorizedGateNet,
    data: base.Data,
    cache: dict[str, np.ndarray],
    spec: RunSpec,
    config: Config,
    reset_state_each_step: bool = False,
    full: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
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
            phase_parts: dict[str, list[np.ndarray]] = {name: [] for name in PHASE_PROBES}
            phase_y_parts: list[np.ndarray] = []
            for local, y, lengths in loader:
                local = local.to(device)
                lengths_device = lengths.to(device)
                trajectory = model.forward_trajectory(local, reset_state_each_step)
                whole_parts.append(_masked_sum(trajectory.contextual, lengths_device).cpu().numpy())
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
                    phase_sequences = {
                        "phase_contextual": trajectory.contextual,
                        "phase_membrane": trajectory.membranes,
                        "phase_gate": trajectory.gates,
                    }
                    for probe_name, sequence in phase_sequences.items():
                        means = exp53._relative_bin_means(sequence, lengths_device, N_REL)
                        phase_parts[probe_name].append(
                            means.flatten(start_dim=0, end_dim=1).cpu().numpy()
                        )
                    phase_y_parts.append(exp53._phase_targets(len(y)))
                y_parts.append(y.numpy())
            split_features: dict[str, np.ndarray] = {
                "y": np.concatenate(y_parts),
                "hidden_whole_count": np.concatenate(whole_parts),
            }
            if full:
                split_features.update(
                    {
                        "hidden_fixed250_ordered": np.concatenate(fixed_parts),
                        "hidden_relative10_ordered": np.concatenate(relative_parts),
                    }
                )
                phase[split] = {
                    "y": np.concatenate(phase_y_parts),
                    **{name: np.concatenate(parts) for name, parts in phase_parts.items()},
                }
            classification[split] = split_features
    return classification, phase


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
        base.dseed(seed, "exp5_3_1", namespace, probe_name, "linear_probe"),
    )


def _fit_ordered_transfer_probe(
    features: dict[str, dict[str, np.ndarray]],
    seed: int,
) -> tuple[StandardScaler, LogisticRegression, dict[str, float | int]]:
    train_x = features["train"]["hidden_whole_count"]
    train_y = features["train"]["y"]
    val_x = features["val"]["hidden_whole_count"]
    val_y = features["val"]["y"]
    test_x = features["test"]["hidden_whole_count"]
    test_y = features["test"]["y"]
    scaler = StandardScaler().fit(train_x)
    train_z = scaler.transform(train_x)
    val_z = scaler.transform(val_x)
    best: tuple[float, float, LogisticRegression] | None = None
    for C in probe_utils.PROBE_C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=5000,
            solver="lbfgs",
            random_state=base.dseed(seed, "exp5_3_1", "ordered_transfer_probe"),
        ).fit(train_z, train_y)
        val_ba = float(balanced_accuracy_score(val_y, classifier.predict(val_z)))
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError("No ordered transfer probe candidate selected")
    val_ba, C, classifier = best
    pred = classifier.predict(scaler.transform(test_x))
    metrics: dict[str, float | int] = {
        "feature_dim": int(train_x.shape[1]),
        "probe_C": C,
        "probe_val_balanced_accuracy": val_ba,
        "probe_test_balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
        "probe_test_accuracy": float(accuracy_score(test_y, pred)),
        "probe_test_macro_f1": float(
            f1_score(test_y, pred, average="macro", zero_division=0)
        ),
    }
    return scaler, classifier, metrics


def _apply_transfer_probe(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    features: dict[str, dict[str, np.ndarray]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for split in ("val", "test"):
        x = features[split]["hidden_whole_count"]
        y = features[split]["y"]
        pred = classifier.predict(scaler.transform(x))
        out[f"{split}_balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
        out[f"{split}_accuracy"] = float(accuracy_score(y, pred))
        out[f"{split}_macro_f1"] = float(
            f1_score(y, pred, average="macro", zero_division=0)
        )
    return out


def history_sensitivity(
    model: FactorizedGateNet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    gate_abs = 0.0
    gate_den = 0.0
    active_gate_abs = 0.0
    active_den = 0.0
    evidence_abs = 0.0
    evidence_den = 0.0
    accumulator_l1 = 0.0
    n_total = 0
    with torch.no_grad():
        for local, _, lengths in loader:
            local = local.to(device)
            lengths = lengths.to(device)
            normal = model.forward_trajectory(local, reset_state_each_step=False)
            reset = model.forward_trajectory(local, reset_state_each_step=True)
            valid = base.mask(lengths, local.shape[1]).to(local.dtype).unsqueeze(-1)
            gate_delta = (normal.gates - reset.gates).abs()
            gate_abs += float((gate_delta * valid).sum().item())
            gate_den += float(valid.sum().item()) * LOCAL_WIDTH
            active = local * valid
            active_gate_abs += float((gate_delta * active).sum().item())
            active_den += float(active.sum().item())
            evidence_delta = (normal.evidence - reset.evidence).abs()
            evidence_abs += float((evidence_delta * valid).sum().item())
            evidence_den += float(valid.sum().item()) * model.n_classes
            normal_acc = _masked_sum(normal.evidence, lengths)
            reset_acc = _masked_sum(reset.evidence, lengths)
            accumulator_l1 += float((normal_acc - reset_acc).abs().sum(dim=1).sum().item())
            n_total += len(local)
    return {
        "gate_history_mae": gate_abs / max(gate_den, EPS),
        "active_feature_gate_history_mae": active_gate_abs / max(active_den, EPS),
        "evidence_history_mae": evidence_abs / max(evidence_den, EPS),
        "accumulator_history_l1_mean": accumulator_l1 / max(n_total, 1),
    }


def parameter_counts(model: FactorizedGateNet) -> dict[str, int]:
    out = {
        "input_projection": int(sum(p.numel() for p in model.input_projection.parameters())),
        "gate_projection": int(sum(p.numel() for p in model.gate_projection.parameters())),
        "evidence_head": int(sum(p.numel() for p in model.evidence_head.parameters())),
        "recurrent": 0,
    }
    if model.recurrent is not None:
        out["recurrent"] = int(sum(p.numel() for p in model.recurrent.parameters()))
    out["trainable_total"] = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    return out


def provenance(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    model: FactorizedGateNet,
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
        "tau_mem_ms": tau_ms_from_beta(definition.beta, data.fs),
        "recurrent": definition.recurrent,
        "stateful": definition.stateful,
        "local_width": LOCAL_WIDTH,
        "context_width": CONTEXT_WIDTH,
        "local_source_experiment": exp52.EXPERIMENT_ID,
        "local_source_protocol": exp52.PROTOCOL_VERSION,
        "local_source_frozen": True,
        "sampling_rate_hz": float(data.fs),
        "objective": "final_accumulated_ce",
        "objective_formula": "CE(5 * sum_valid(e_t) / T_valid, y)",
        "gate_formula": "g_t = 2*sigmoid(W_g u_t + b_g); contextual_t = g_t * z_t",
        "evidence_accumulator": "signed non-leaky non-spiking sum; evidence-head bias=False",
        "checkpoint_selection": "max native validation BA; tie-break validation CE",
        "threshold": THRESHOLD,
        "reset": RESET,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "shuffle_replicates": SHUFFLE_REPLICATES,
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

        train_metrics = base.metrics(np.concatenate(train_true), np.concatenate(train_pred))
        val_metrics = evaluate_native(model, eval_loaders["val"], device)
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
            abs(val_ba - best_val_ba) <= 1e-12 and val_loss < best_val_loss - 1e-12
        )
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    model.load_state_dict(best_state)
    native = {
        split: evaluate_native(model, loader, device)
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
                "native": native,
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
) -> tuple[FactorizedGateNet, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.3.1 checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong Exp5.3.1 checkpoint identity: {path}")
    if payload.get("spec") != asdict(spec):
        raise ValueError(f"Checkpoint spec mismatch: {path}")
    model = FactorizedGateNet(spec.condition, len(data.labels), data.fs).to(
        torch.device(config.device)
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _native_all_splits(
    model: FactorizedGateNet,
    loaders: dict[str, DataLoader],
    device: torch.device,
    reset_state_each_step: bool,
) -> dict[str, dict[str, float]]:
    return {
        split: evaluate_native(
            model, loader, device, reset_state_each_step=reset_state_each_step
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

    native = _native_all_splits(model, loaders, device, False)
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
    scaler, classifier, ordered_transfer = _fit_ordered_transfer_probe(
        normal_features, spec.seed
    )

    reset_native = _native_all_splits(model, loaders, device, True)
    reset_features, _ = collect_features(
        model, data, cache, spec, config, reset_state_each_step=True, full=False
    )
    reset_refit = _probe_metrics(
        reset_features, "hidden_whole_count", spec.seed, "state_reset_refit"
    )
    reset_transfer = _apply_transfer_probe(scaler, classifier, reset_features)

    ablations: list[dict[str, object]] = [
        {
            "ablation": "state_reset",
            "replicate": 0,
            "native": reset_native,
            "hidden_whole_count_refit_probe": reset_refit,
            "hidden_whole_count_transfer_probe": reset_transfer,
        }
    ]

    for replicate in range(SHUFFLE_REPLICATES):
        shuffled_cache = _shuffle_cache(data, cache, spec.seed, replicate)
        shuffled_loaders = _make_loaders(
            data, shuffled_cache, spec, config, train_shuffle=False
        )
        shuffled_native = _native_all_splits(model, shuffled_loaders, device, False)
        shuffled_features, _ = collect_features(
            model,
            data,
            shuffled_cache,
            spec,
            config,
            reset_state_each_step=False,
            full=False,
        )
        shuffled_refit = _probe_metrics(
            shuffled_features,
            "hidden_whole_count",
            spec.seed,
            f"temporal_shuffle_{replicate}_refit",
        )
        shuffled_transfer = _apply_transfer_probe(scaler, classifier, shuffled_features)
        ablations.append(
            {
                "ablation": "temporal_shuffle",
                "replicate": replicate,
                "native": shuffled_native,
                "hidden_whole_count_refit_probe": shuffled_refit,
                "hidden_whole_count_transfer_probe": shuffled_transfer,
            }
        )

    sensitivity = {
        split: history_sensitivity(model, loader, device)
        for split, loader in loaders.items()
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "provenance": checkpoint["provenance"],
        "best_epoch": checkpoint["result"]["best_epoch"],
        "native": native,
        "probes": probes,
        "phase_probes": phase_probes,
        "ordered_transfer_probe": ordered_transfer,
        "ablations": ablations,
        "history_sensitivity": sensitivity,
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
        "tau_mem_ms": tau_ms_from_beta(condition.beta, fs),
        "recurrent": condition.recurrent,
        "stateful": condition.stateful,
    }


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    history_parts: list[pd.DataFrame] = []

    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp5.3.1 evaluation: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != PROTOCOL_VERSION or payload.get("spec") != asdict(spec):
            raise ValueError(f"Evaluation identity mismatch: {path}")
        condition = spec.definition
        fs = float(payload["provenance"]["sampling_rate_hz"])
        common = {**_condition_row(condition, fs), "seed": spec.seed}
        row: dict[str, object] = {
            **common,
            "best_epoch": payload["best_epoch"],
            "trainable_parameters": payload["provenance"]["parameter_counts"]["trainable_total"],
        }
        for split in ("train", "val", "test"):
            for metric, value in payload["native"][split].items():
                row[f"native_{split}_{metric}"] = value
        rows.append(row)

        for probe in payload["probes"]:
            probe_rows.append(
                {
                    **common,
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
                    **common,
                    "probe_type": probe["probe_type"],
                    "probe_C": probe["probe_C"],
                    "val_ba": probe["probe_val_balanced_accuracy"],
                    "test_ba": probe["probe_test_balanced_accuracy"],
                    "test_accuracy": probe["probe_test_accuracy"],
                    "test_macro_f1": probe["probe_test_macro_f1"],
                }
            )

        ordered_probe = next(
            probe for probe in payload["probes"] if probe["probe_type"] == "hidden_whole_count"
        )
        ordered_transfer = payload["ordered_transfer_probe"]
        ablation_rows.append(
            {
                **common,
                "ablation": "ordered",
                "replicate": 0,
                "native_val_ba": payload["native"]["val"]["balanced_accuracy"],
                "native_test_ba": payload["native"]["test"]["balanced_accuracy"],
                "hidden_whole_count_refit_val_ba": ordered_probe["probe_val_balanced_accuracy"],
                "hidden_whole_count_refit_test_ba": ordered_probe["probe_test_balanced_accuracy"],
                "hidden_whole_count_transfer_val_ba": ordered_transfer["probe_val_balanced_accuracy"],
                "hidden_whole_count_transfer_test_ba": ordered_transfer["probe_test_balanced_accuracy"],
            }
        )
        for ablation in payload["ablations"]:
            refit = ablation["hidden_whole_count_refit_probe"]
            transfer = ablation["hidden_whole_count_transfer_probe"]
            ablation_rows.append(
                {
                    **common,
                    "ablation": ablation["ablation"],
                    "replicate": ablation["replicate"],
                    "native_val_ba": ablation["native"]["val"]["balanced_accuracy"],
                    "native_test_ba": ablation["native"]["test"]["balanced_accuracy"],
                    "hidden_whole_count_refit_val_ba": refit["probe_val_balanced_accuracy"],
                    "hidden_whole_count_refit_test_ba": refit["probe_test_balanced_accuracy"],
                    "hidden_whole_count_transfer_val_ba": transfer["val_balanced_accuracy"],
                    "hidden_whole_count_transfer_test_ba": transfer["test_balanced_accuracy"],
                }
            )
        for split, metrics in payload["history_sensitivity"].items():
            sensitivity_rows.append({**common, "split": split, **metrics})

        hpath = history_path(root, spec)
        if not hpath.exists():
            raise FileNotFoundError(f"Missing Exp5.3.1 history: {hpath}")
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

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "runs": root / "runs.csv",
        "histories": root / "histories.csv",
        "probe_runs": root / "probe_runs.csv",
        "phase_probe_runs": root / "phase_probe_runs.csv",
        "ablation_runs": root / "ablation_runs.csv",
        "history_sensitivity_runs": root / "history_sensitivity_runs.csv",
        "local_reference": root / "local_reference.csv",
        "manifest": root / "manifest.json",
    }
    pd.DataFrame(rows).to_csv(outputs["runs"], index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(outputs["histories"], index=False)
    pd.DataFrame(probe_rows).to_csv(outputs["probe_runs"], index=False)
    pd.DataFrame(phase_rows).to_csv(outputs["phase_probe_runs"], index=False)
    pd.DataFrame(ablation_rows).to_csv(outputs["ablation_runs"], index=False)
    pd.DataFrame(sensitivity_rows).to_csv(outputs["history_sensitivity_runs"], index=False)
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
            "primary_representation_metric": "HiddenWholeCount - LocalWholeCount",
            "preservation_metric": "HiddenRelative10 as ordered-trajectory ceiling only",
            "causal_metrics": [
                "trained factorized FF vs stateful conditions",
                "ordered vs state_reset HiddenWholeCount",
                "ordered vs temporal_shuffle HiddenWholeCount",
                "ordered-trained transfer probe",
                "gate/evidence history sensitivity",
            ],
            "checkpoint_selection": "native validation BA only; tie-break validation CE",
            "aggregation_policy": "finalizer only aggregates completed run artifacts; notebook performs summaries and plots",
            "shuffle_replicates": SHUFFLE_REPLICATES,
            "files": {name: path.name for name, path in outputs.items() if name != "manifest"},
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
        description="Experiment 5.3.1 factorized gate causal attribution"
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
