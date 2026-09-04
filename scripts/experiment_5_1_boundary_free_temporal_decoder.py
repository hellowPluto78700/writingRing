from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import snntorch as snn
from snntorch import surrogate

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_0_4_l2_width_representation_capacity as probe_utils
from scripts import experiment_5_0_1_exp3_analog_head_control as exp501


EXPERIMENT_ID = "experiment_5_1_boundary_free_temporal_decoder"
PROTOCOL_VERSION = "frozen_local_leaky_interface_rsnn_v1"

LOCAL_SOURCE_CONDITION = exp501.EXP3_EXACT
SEEDS = exp501.SEEDS
INTERFACES = ("raw_l2", "leaky242")
ARCHITECTURES = ("ff", "rsnn")
TEMPORAL_TAU_MEM_MS = (125.0, 250.0, 500.0, 1000.0)
TEMPORAL_WIDTH = 128
LOCAL_WIDTH = 128

AGG_SHIFT = 4
AGG_ALPHA = 1.0 - 2.0 ** (-AGG_SHIFT)
THRESHOLD = base.THRESHOLD
SURROGATE_SLOPE = base.SURROGATE_SLOPE
RESET = base.RESET

EPOCHS = 100
BATCH_SIZE = base.BATCH_SIZE
LR = 1e-3
WEIGHT_DECAY = 0.0
EPS = 1e-8

TEMPORAL_PROBES = (
    "hidden_whole_count",
    "hidden_fixed250_ordered",
    "hidden_relative10_ordered",
    "hidden_uend",
)
INTERFACE_PROBES = (
    "interface_valid_sum",
    "interface_fixed250_ordered",
    "interface_relative10_ordered",
    "interface_endpoint",
)


@dataclass(frozen=True)
class RunSpec:
    interface: str
    architecture: str
    tau_mem_ms: float
    seed: int

    @property
    def key(self) -> str:
        tau_tag = str(int(self.tau_mem_ms))
        return f"{self.interface}__{self.architecture}__tau{tau_tag}ms__seed{self.seed}"


@dataclass(frozen=True)
class InterfaceProbeSpec:
    interface: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.interface}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "snn").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate writingRing repository root")


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def local_cache_path(root: Path, seed: int) -> Path:
    return root / "frozen_l2" / f"exp3_exact_timestep__seed{seed}.npz"


def local_cache_meta_path(root: Path, seed: int) -> Path:
    return root / "frozen_l2" / f"exp3_exact_timestep__seed{seed}.json"


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def interface_probe_path(root: Path, spec: InterfaceProbeSpec) -> Path:
    return root / "interface_probes" / f"{spec.key}.json"


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(interface, architecture, tau_mem_ms, seed)
        for interface in INTERFACES
        for architecture in ARCHITECTURES
        for tau_mem_ms in TEMPORAL_TAU_MEM_MS
        for seed in SEEDS
    ]


def interface_probe_specs() -> list[InterfaceProbeSpec]:
    return [InterfaceProbeSpec(interface, seed) for interface in INTERFACES for seed in SEEDS]


def aggregator_tau_ms(fs: float) -> float:
    dt_ms = 1000.0 / float(fs)
    return -dt_ms / math.log(AGG_ALPHA)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _local_source_root(repo_root: Path) -> Path:
    return exp501.results_dir(repo_root)


def prepare_local_seed(seed: int, config: Config, force: bool = False) -> Path:
    """Prepare one exact Exp5.0.1/Exp3 frozen L2 spike cache.

    Existing Exp5.0.1 checkpoints are reused. On a fresh clone, the exact
    Exp3 control is trained once for this seed, then frozen L2 trajectories are
    cached as uint8 so the 48 temporal-decoder runs never repeat local-SNN work.
    """
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}")
    destination = local_cache_path(config.results_dir, seed)
    meta_path = local_cache_meta_path(config.results_dir, seed)
    if destination.exists() and meta_path.exists() and not force:
        return destination

    torch.set_num_threads(config.threads)
    data = base.prepare_data(config.repo_root)
    source_spec = exp501.RunSpec(LOCAL_SOURCE_CONDITION, seed)
    source_config = exp501.Config(
        repo_root=config.repo_root,
        results_dir=_local_source_root(config.repo_root),
        device=config.device,
        epochs=exp501.EPOCHS,
        batch_size=config.batch_size,
        threads=config.threads,
    )
    exp501.train_one(source_spec, data, source_config, force=False)
    model, payload = exp501.load_model(source_spec, data, source_config)
    device = torch.device(config.device)

    arrays: dict[str, np.ndarray] = {}
    partitions = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    model.eval()
    with torch.no_grad():
        for split, (X, y, lengths) in partitions.items():
            loader = base.loader(
                X,
                y,
                lengths,
                config.batch_size,
                False,
                base.dseed(seed, "exp5_1", split, "local_cache"),
            )
            parts: list[np.ndarray] = []
            for Xb, _, _ in loader:
                Xb = Xb.to(device)
                spikes = model.layer_features(Xb)["L2"]  # type: ignore[attr-defined]
                parts.append(spikes.detach().cpu().numpy().astype(np.uint8))
            arrays[f"X{split}"] = np.concatenate(parts, axis=0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    source_checkpoint = exp501.checkpoint_path(_local_source_root(config.repo_root), source_spec)
    _save_json(
        meta_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "source_experiment_id": exp501.EXPERIMENT_ID,
            "source_protocol_version": exp501.PROTOCOL_VERSION,
            "source_condition": LOCAL_SOURCE_CONDITION,
            "source_checkpoint": str(source_checkpoint.relative_to(config.repo_root)),
            "source_best_epoch": payload["result"]["best_epoch"],
            "split_seed": base.SPLIT_SEED,
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "sampling_rate_hz": float(data.fs),
            "local_width": LOCAL_WIDTH,
            "l1_shifts": (2, 3, 4),
            "l2_shifts": (2, 3, 4),
            "cache_dtype": "uint8 binary L2 spikes",
        },
    )
    return destination


def load_local_cache(seed: int, data: base.Data, config: Config) -> dict[str, np.ndarray]:
    path = local_cache_path(config.results_dir, seed)
    meta_path = local_cache_meta_path(config.results_dir, seed)
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing frozen L2 cache for seed {seed}: run prepare-local first"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = (PROTOCOL_VERSION, seed, LOCAL_SOURCE_CONDITION)
    actual = (
        str(meta.get("protocol_version")),
        int(meta.get("seed")),
        str(meta.get("source_condition")),
    )
    if actual != expected:
        raise ValueError(f"Frozen L2 cache identity mismatch: {actual} != {expected}")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].astype(np.float32) for name in loaded.files}
    expected_shapes = {
        "Xtrain": (len(data.ytr), data.T, LOCAL_WIDTH),
        "Xval": (len(data.yva), data.T, LOCAL_WIDTH),
        "Xtest": (len(data.yte), data.T, LOCAL_WIDTH),
    }
    for name, shape in expected_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError(f"Frozen L2 cache {name} shape mismatch: {arrays.get(name, None)}")
    return arrays


def interface_sequence(l2_spikes: torch.Tensor, interface: str) -> torch.Tensor:
    if interface == "raw_l2":
        return l2_spikes
    if interface != "leaky242":
        raise ValueError(f"Unknown interface: {interface}")
    state = torch.zeros_like(l2_spikes[:, 0])
    outputs: list[torch.Tensor] = []
    for timestep in range(l2_spikes.shape[1]):
        state = AGG_ALPHA * state + l2_spikes[:, timestep]
        outputs.append(state)
    return torch.stack(outputs, dim=1)


class TemporalAnalogDecoder(nn.Module):
    """FF/RSNN temporal decoder with an analog whole-sequence evidence head."""

    def __init__(
        self,
        architecture: str,
        tau_mem_ms: float,
        n_classes: int,
        fs: float,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        if tau_mem_ms not in TEMPORAL_TAU_MEM_MS:
            raise ValueError(f"Unsupported tau_mem_ms: {tau_mem_ms}")
        self.architecture = architecture
        self.tau_mem_ms = float(tau_mem_ms)
        self.hidden_width = TEMPORAL_WIDTH
        dt_ms = 1000.0 / float(fs)
        beta = math.exp(-dt_ms / self.tau_mem_ms)
        spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

        self.input_hidden = nn.Linear(LOCAL_WIDTH, TEMPORAL_WIDTH, bias=False)
        self.recurrent = (
            nn.Linear(TEMPORAL_WIDTH, TEMPORAL_WIDTH, bias=False)
            if architecture == "rsnn"
            else None
        )
        self.hidden_lif = snn.Leaky(
            beta=beta,
            threshold=THRESHOLD,
            spike_grad=spike_grad,
            reset_mechanism=RESET,
        )
        self.head = nn.Linear(TEMPORAL_WIDTH, n_classes, bias=True)

    def forward_trajectory(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_steps, _ = x.shape
        hidden_mem = torch.zeros(
            batch, self.hidden_width, device=x.device, dtype=x.dtype
        )
        prev_spike = torch.zeros_like(hidden_mem)
        spike_parts: list[torch.Tensor] = []
        mem_parts: list[torch.Tensor] = []
        logit_parts: list[torch.Tensor] = []
        for timestep in range(n_steps):
            current = self.input_hidden(x[:, timestep])
            if self.recurrent is not None:
                current = current + self.recurrent(prev_spike)
            spike, hidden_mem = self.hidden_lif(current, hidden_mem)
            logits = self.head(spike)
            spike_parts.append(spike)
            mem_parts.append(hidden_mem)
            logit_parts.append(logits)
            prev_spike = spike
        return (
            torch.stack(spike_parts, dim=1),
            torch.stack(mem_parts, dim=1),
            torch.stack(logit_parts, dim=1),
        )

    @staticmethod
    def sequence_logits(logits_t: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        valid = base.mask(lengths, logits_t.shape[1]).to(logits_t.dtype).unsqueeze(-1)
        return (logits_t * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def loss_logits(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spikes, membranes, logits_t = self.forward_trajectory(x)
        logits = self.sequence_logits(logits_t, lengths)
        return F.cross_entropy(logits, y), logits, spikes, membranes


def _initialize_temporal_model(
    spec: RunSpec,
    n_classes: int,
    fs: float,
    device: torch.device,
) -> TemporalAnalogDecoder:
    # Reset modules independently so input/head initialization is paired across
    # raw/leaky interfaces, FF/RSNN, and tau values for a given seed.
    base.seed_all(base.dseed(spec.seed, "exp5_1", "constructor"))
    model = TemporalAnalogDecoder(
        spec.architecture,
        spec.tau_mem_ms,
        n_classes,
        fs,
    ).to(device)
    base.seed_all(base.dseed(spec.seed, "exp5_1", "input_hidden"))
    model.input_hidden.reset_parameters()
    if model.recurrent is not None:
        base.seed_all(base.dseed(spec.seed, "exp5_1", "recurrent"))
        model.recurrent.reset_parameters()
    base.seed_all(base.dseed(spec.seed, "exp5_1", "analog_head"))
    model.head.reset_parameters()
    return model


def _loader(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def _partitions(data: base.Data, cache: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
            base.dseed(spec.seed, "exp5_1", split, "loader"),
        )
        for split, partition in _partitions(data, cache).items()
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return base.metrics(y_true, y_pred)


def _masked_sum(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = base.mask(lengths, sequence.shape[1]).to(sequence.dtype).unsqueeze(-1)
    return (sequence * valid).sum(dim=1)


def _valid_endpoint(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    index = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[index, lengths.clamp_min(1) - 1]


def _event_diagnostics(
    spikes: torch.Tensor,
    lengths: torch.Tensor,
    fs: float,
) -> tuple[float, float]:
    valid = base.mask(lengths, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
    valid_events = float((spikes * valid).sum().item())
    full_events = float(spikes.sum().item())
    valid_seconds = float(lengths.sum().item()) / float(fs)
    rate = valid_events / max(valid_seconds * spikes.shape[2], EPS)
    tail_fraction = (full_events - valid_events) / max(full_events, EPS)
    return rate, tail_fraction


def evaluate_native(
    model: TemporalAnalogDecoder,
    loader: DataLoader,
    interface: str,
    device: torch.device,
    fs: float,
) -> dict[str, float]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    valid_events = 0.0
    full_events = 0.0
    valid_seconds = 0.0
    with torch.no_grad():
        for l2, y, lengths in loader:
            l2 = l2.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            temporal_input = interface_sequence(l2, interface)
            loss, logits, spikes, _ = model.loss_logits(temporal_input, lengths, y)
            n = len(y)
            loss_sum += float(loss.item()) * n
            n_total += n
            y_true_parts.append(y.cpu().numpy())
            y_pred_parts.append(logits.argmax(dim=1).cpu().numpy())
            valid = base.mask(lengths, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
            valid_events += float((spikes * valid).sum().item())
            full_events += float(spikes.sum().item())
            valid_seconds += float(lengths.sum().item()) / float(fs)
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    out = _classification_metrics(y_true, y_pred)
    out["loss"] = loss_sum / max(n_total, 1)
    out["hidden_events_per_neuron_second"] = valid_events / max(valid_seconds * TEMPORAL_WIDTH, EPS)
    out["hidden_tail_event_fraction"] = (full_events - valid_events) / max(full_events, EPS)
    return out


def _probe_metrics(
    features: dict[str, dict[str, np.ndarray]],
    probe_name: str,
    seed: int,
) -> dict[str, object]:
    return probe_utils._raw_probe_metrics(
        features["train"][probe_name],
        features["train"]["y"],
        features["val"][probe_name],
        features["val"]["y"],
        features["test"][probe_name],
        features["test"]["y"],
        base.dseed(seed, "exp5_1", probe_name, "linear_probe"),
    )


def collect_interface_features(
    data: base.Data,
    cache: dict[str, np.ndarray],
    interface: str,
    config: Config,
) -> dict[str, dict[str, np.ndarray]]:
    device = torch.device(config.device)
    out: dict[str, dict[str, np.ndarray]] = {}
    for split, (X, y, lengths) in _partitions(data, cache).items():
        loader = _loader(
            X,
            y,
            lengths,
            config.batch_size,
            False,
            base.dseed(0, "exp5_1", split, "interface_probe_loader"),
        )
        y_parts: list[np.ndarray] = []
        sum_parts: list[np.ndarray] = []
        fixed_parts: list[np.ndarray] = []
        relative_parts: list[np.ndarray] = []
        endpoint_parts: list[np.ndarray] = []
        with torch.no_grad():
            for l2, yb, lb in loader:
                l2 = l2.to(device)
                lb_device = lb.to(device)
                seq = interface_sequence(l2, interface)
                sum_parts.append(_masked_sum(seq, lb_device).cpu().numpy())
                fixed_parts.append(base.fixed_counts(seq, lb_device, data.bin_steps).flatten(start_dim=1).cpu().numpy())
                relative_parts.append(base.relative_counts(seq, lb_device, base.N_REL).flatten(start_dim=1).cpu().numpy())
                endpoint_parts.append(_valid_endpoint(seq, lb_device).cpu().numpy())
                y_parts.append(yb.numpy())
        out[split] = {
            "y": np.concatenate(y_parts),
            "interface_valid_sum": np.concatenate(sum_parts),
            "interface_fixed250_ordered": np.concatenate(fixed_parts),
            "interface_relative10_ordered": np.concatenate(relative_parts),
            "interface_endpoint": np.concatenate(endpoint_parts),
        }
    return out


def run_interface_probe(
    spec: InterfaceProbeSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    destination = interface_probe_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    cache = load_local_cache(spec.seed, data, config)
    extracted = collect_interface_features(data, cache, spec.interface, config)
    probes = [
        {"probe_type": probe, **_probe_metrics(extracted, probe, spec.seed)}
        for probe in INTERFACE_PROBES
    ]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "interface": spec.interface,
        "seed": spec.seed,
        "aggregator_alpha": AGG_ALPHA if spec.interface == "leaky242" else None,
        "aggregator_tau_ms": aggregator_tau_ms(data.fs) if spec.interface == "leaky242" else None,
        "probes": probes,
    }
    _save_json(destination, payload)
    return payload


def collect_temporal_features(
    model: TemporalAnalogDecoder,
    data: base.Data,
    cache: dict[str, np.ndarray],
    spec: RunSpec,
    config: Config,
) -> dict[str, dict[str, np.ndarray]]:
    device = torch.device(config.device)
    out: dict[str, dict[str, np.ndarray]] = {}
    loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)
    model.eval()
    with torch.no_grad():
        for split, loader in loaders.items():
            y_parts: list[np.ndarray] = []
            whole_parts: list[np.ndarray] = []
            fixed_parts: list[np.ndarray] = []
            relative_parts: list[np.ndarray] = []
            uend_parts: list[np.ndarray] = []
            for l2, y, lengths in loader:
                l2 = l2.to(device)
                lengths_device = lengths.to(device)
                temporal_input = interface_sequence(l2, spec.interface)
                spikes, membranes, _ = model.forward_trajectory(temporal_input)
                whole_parts.append(_masked_sum(spikes, lengths_device).cpu().numpy())
                fixed_parts.append(base.fixed_counts(spikes, lengths_device, data.bin_steps).flatten(start_dim=1).cpu().numpy())
                relative_parts.append(base.relative_counts(spikes, lengths_device, base.N_REL).flatten(start_dim=1).cpu().numpy())
                uend_parts.append(_valid_endpoint(membranes, lengths_device).cpu().numpy())
                y_parts.append(y.numpy())
            out[split] = {
                "y": np.concatenate(y_parts),
                "hidden_whole_count": np.concatenate(whole_parts),
                "hidden_fixed250_ordered": np.concatenate(fixed_parts),
                "hidden_relative10_ordered": np.concatenate(relative_parts),
                "hidden_uend": np.concatenate(uend_parts),
            }
    return out


def parameter_counts(spec: RunSpec, n_classes: int) -> dict[str, int]:
    input_hidden = LOCAL_WIDTH * TEMPORAL_WIDTH
    recurrent = TEMPORAL_WIDTH * TEMPORAL_WIDTH if spec.architecture == "rsnn" else 0
    head = TEMPORAL_WIDTH * n_classes + n_classes
    return {
        "input_hidden": input_hidden,
        "recurrent": recurrent,
        "analog_head": head,
        "trainable_total": input_hidden + recurrent + head,
        "aggregator_trainable": 0,
    }


def provenance(spec: RunSpec, data: base.Data, config: Config) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": spec.seed,
        "split_seed": base.SPLIT_SEED,
        "interface": spec.interface,
        "architecture": spec.architecture,
        "temporal_tau_mem_ms": spec.tau_mem_ms,
        "temporal_width": TEMPORAL_WIDTH,
        "local_source_experiment": exp501.EXPERIMENT_ID,
        "local_source_protocol": exp501.PROTOCOL_VERSION,
        "local_source_condition": LOCAL_SOURCE_CONDITION,
        "local_source_frozen": True,
        "local_width": LOCAL_WIDTH,
        "local_shifts": (2, 3, 4),
        "sampling_rate_hz": float(data.fs),
        "raw_dt_ms": 1000.0 / float(data.fs),
        "aggregator": (
            "identity-preserving leaky accumulator; no weights/threshold/reset"
            if spec.interface == "leaky242"
            else "none; raw frozen L2 spikes"
        ),
        "aggregator_alpha": AGG_ALPHA if spec.interface == "leaky242" else None,
        "aggregator_tau_ms": aggregator_tau_ms(data.fs) if spec.interface == "leaky242" else None,
        "aggregator_dc_mass_samples": 1.0 / (1.0 - AGG_ALPHA) if spec.interface == "leaky242" else None,
        "temporal_update_rate": "raw 64Hz; no temporal downsampling",
        "head": "Linear(128,12,bias=True); no output LIF",
        "objective": "CE(valid-mean analog class evidence, y)",
        "checkpoint_selection": "max validation native BA; tie-break validation CE",
        "threshold": THRESHOLD,
        "reset": RESET,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "parameter_counts": parameter_counts(spec, len(data.labels)),
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
    loaders = _make_loaders(data, cache, spec, config, train_shuffle=True)
    eval_loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _initialize_temporal_model(spec, len(data.labels), data.fs, device)
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
        for l2, y, lengths in loaders["train"]:
            l2 = l2.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            with torch.no_grad():
                temporal_input = interface_sequence(l2, spec.interface)
            optimizer.zero_grad(set_to_none=True)
            loss, logits, _, _ = model.loss_logits(temporal_input, lengths, y)
            loss.backward()
            optimizer.step()
            n = len(y)
            train_n += n
            train_loss_sum += float(loss.item()) * n
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = _classification_metrics(np.concatenate(train_true), np.concatenate(train_pred))
        val_metrics = evaluate_native(model, eval_loaders["val"], spec.interface, device, data.fs)
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
    final = {
        split: evaluate_native(model, loader, spec.interface, device, data.fs)
        for split, loader in eval_loaders.items()
    }
    result = {
        "spec": spec.__dict__,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_ba,
        "best_val_loss": best_val_loss,
        "native": final,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "provenance": provenance(spec, data, config),
            "result": result,
            "state_dict": best_state,
        },
        destination,
    )
    history_file = history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False)
    return destination


def load_model(spec: RunSpec, data: base.Data, config: Config) -> tuple[TemporalAnalogDecoder, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.1 checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong Exp5.1 checkpoint identity: {path}")
    if payload.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint spec mismatch: {path}")
    model = TemporalAnalogDecoder(spec.architecture, spec.tau_mem_ms, len(data.labels), data.fs).to(torch.device(config.device))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


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
    eval_loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)
    native = {
        split: evaluate_native(model, loader, spec.interface, device, data.fs)
        for split, loader in eval_loaders.items()
    }
    extracted = collect_temporal_features(model, data, cache, spec, config)
    probes = [
        {"probe_type": probe, **_probe_metrics(extracted, probe, spec.seed)}
        for probe in TEMPORAL_PROBES
    ]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "provenance": checkpoint["provenance"],
        "best_epoch": checkpoint["result"]["best_epoch"],
        "native": native,
        "probes": probes,
    }
    _save_json(destination, payload)
    return payload


def run_one(spec: RunSpec, data: base.Data, config: Config, force: bool = False) -> dict[str, object]:
    train_one(spec, data, config, force=force)
    return evaluate_one(spec, data, config, force=force)


def _probe_value(payload: dict[str, object], probe_type: str, key: str) -> float:
    for probe in payload["probes"]:  # type: ignore[index]
        if probe["probe_type"] == probe_type:
            return float(probe[key])
    raise KeyError(probe_type)


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    history_parts: list[pd.DataFrame] = []
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp5.1 evaluation: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != PROTOCOL_VERSION or payload.get("spec") != spec.__dict__:
            raise ValueError(f"Evaluation identity mismatch: {path}")
        row: dict[str, object] = {
            "interface": spec.interface,
            "architecture": spec.architecture,
            "tau_mem_ms": spec.tau_mem_ms,
            "seed": spec.seed,
            "best_epoch": payload["best_epoch"],
        }
        for split in ("train", "val", "test"):
            native = payload["native"][split]
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "loss",
                "hidden_events_per_neuron_second",
                "hidden_tail_event_fraction",
            ):
                row[f"native_{split}_{metric}"] = native[metric]
        for probe in TEMPORAL_PROBES:
            row[f"{probe}_val_ba"] = _probe_value(payload, probe, "probe_val_balanced_accuracy")
            row[f"{probe}_test_ba"] = _probe_value(payload, probe, "probe_test_balanced_accuracy")
            row[f"{probe}_C"] = _probe_value(payload, probe, "probe_C")
        rows.append(row)
        hpath = history_path(root, spec)
        if not hpath.exists():
            raise FileNotFoundError(f"Missing Exp5.1 history: {hpath}")
        history = pd.read_csv(hpath)
        history.insert(0, "seed", spec.seed)
        history.insert(0, "tau_mem_ms", spec.tau_mem_ms)
        history.insert(0, "architecture", spec.architecture)
        history.insert(0, "interface", spec.interface)
        history_parts.append(history)

    interface_rows: list[dict[str, object]] = []
    for spec in interface_probe_specs():
        path = interface_probe_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp5.1 interface probe: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for probe in payload["probes"]:
            interface_rows.append(
                {
                    "interface": spec.interface,
                    "seed": spec.seed,
                    "probe_type": probe["probe_type"],
                    "probe_C": probe["probe_C"],
                    "val_ba": probe["probe_val_balanced_accuracy"],
                    "test_ba": probe["probe_test_balanced_accuracy"],
                }
            )

    reference_path = _local_source_root(repo_root) / "runs.csv"
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing committed Exp5.0.1 reference table: {reference_path}")
    reference = pd.read_csv(reference_path)
    reference = reference[reference["condition"] == LOCAL_SOURCE_CONDITION].copy()
    if set(reference["seed"].astype(int)) != set(SEEDS):
        raise ValueError("Exp5.0.1 local reference seeds do not match Exp5.1 seeds")

    runs_path = root / "runs.csv"
    interface_path = root / "interface_probes.csv"
    histories_path = root / "histories.csv"
    reference_out = root / "local_reference.csv"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(runs_path, index=False)
    pd.DataFrame(interface_rows).to_csv(interface_path, index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(histories_path, index=False)
    reference.to_csv(reference_out, index=False)
    manifest_path = root / "manifest.json"
    _save_json(
        manifest_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "expected_decoder_runs": len(run_specs()),
            "expected_interface_probe_runs": len(interface_probe_specs()),
            "expected_local_cache_seeds": len(SEEDS),
            "interfaces": INTERFACES,
            "architectures": ARCHITECTURES,
            "temporal_tau_mem_ms": TEMPORAL_TAU_MEM_MS,
            "aggregator_shift": AGG_SHIFT,
            "aggregator_alpha": AGG_ALPHA,
            "aggregator_tau_ms_at_64hz": aggregator_tau_ms(64.0),
            "aggregator_dc_mass_samples": 1.0 / (1.0 - AGG_ALPHA),
            "local_source_condition": LOCAL_SOURCE_CONDITION,
            "seeds": SEEDS,
            "aggregation_policy": "finalizer concatenates only; notebook computes summaries, selection, paired effects, and plots",
            "files": {
                "runs": runs_path.name,
                "interface_probes": interface_path.name,
                "histories": histories_path.name,
                "local_reference": reference_out.name,
            },
        },
    )
    return {
        "runs": runs_path,
        "interface_probes": interface_path,
        "histories": histories_path,
        "local_reference": reference_out,
        "manifest": manifest_path,
    }


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
    parser = argparse.ArgumentParser(description="Experiment 5.1 boundary-free temporal decoder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--repo-root", type=str, default=None)
        sub.add_argument("--device", type=str, default="cpu")
        sub.add_argument("--threads", type=int, default=1)
        sub.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        sub.add_argument("--epochs", type=int, default=EPOCHS)
        sub.add_argument("--force", action="store_true")

    prep = subparsers.add_parser("prepare-local")
    common(prep)
    prep.add_argument("--array-task-id", type=int, required=True)

    interface = subparsers.add_parser("probe-interface")
    common(interface)
    interface.add_argument("--array-task-id", type=int, required=True)

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
        path = prepare_local_seed(SEEDS[task_id], config, force=args.force)
        print(path)
        return
    if args.command == "probe-interface":
        specs = interface_probe_specs()
        task_id = int(args.array_task_id)
        if not 0 <= task_id < len(specs):
            raise IndexError(f"probe-interface task {task_id} outside 0..{len(specs)-1}")
        payload = run_interface_probe(specs[task_id], data, config, force=args.force)
        print(json.dumps(payload, indent=2, sort_keys=True))
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
