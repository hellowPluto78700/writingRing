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


EXPERIMENT_ID = "experiment_5_2_frozen_local_tauR_sweep"
PROTOCOL_VERSION = "frozen_exp3_l2_endpoint_tauR_v1"
LOCAL_SOURCE_CONDITION = exp501.EXP3_EXACT
SEEDS = (11, 23, 37, 53, 71)
ARCHITECTURES = ("ff", "rsnn")
SHIFT_MEM_R = (3, 4, 5, 6, 7)
TEMPORAL_WIDTH = 128
LOCAL_WIDTH = 128
THRESHOLD = base.THRESHOLD
SURROGATE_SLOPE = base.SURROGATE_SLOPE
RESET = base.RESET
EPOCHS = base.EPOCHS
BATCH_SIZE = base.BATCH_SIZE
LR = 1e-3
WEIGHT_DECAY = 0.0
TAIL_SECONDS = 2.0
EPS = 1e-8

TEMPORAL_PROBES = (
    "hidden_whole_count",
    "hidden_fixed250_ordered",
    "hidden_relative10_ordered",
    "hidden_uend",
)
LOCAL_REFERENCE_PROBES = (
    "local_whole_count",
    "local_fixed250_ordered",
    "local_relative10_ordered",
)


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    shift_mem_r: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.architecture}__shift{self.shift_mem_r}__seed{self.seed}"


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


def local_source_root(root: Path) -> Path:
    return root / "local_source_exp501"


def local_cache_path(root: Path, seed: int) -> Path:
    return root / "frozen_l2" / f"exp3_exact_timestep__seed{seed}.npz"


def local_cache_meta_path(root: Path, seed: int) -> Path:
    return root / "frozen_l2" / f"exp3_exact_timestep__seed{seed}.json"


def local_reference_path(root: Path, seed: int) -> Path:
    return root / "local_references" / f"seed{seed}.json"


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def beta_from_shift(shift_mem_r: int) -> float:
    if shift_mem_r <= 0:
        raise ValueError("shift_mem_r must be positive")
    return 1.0 - 2.0 ** (-int(shift_mem_r))


def tau_ms_from_shift(shift_mem_r: int, fs: float) -> float:
    beta = beta_from_shift(shift_mem_r)
    dt_ms = 1000.0 / float(fs)
    return -dt_ms / math.log(beta)


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(architecture, shift_mem_r, seed)
        for architecture in ARCHITECTURES
        for shift_mem_r in SHIFT_MEM_R
        for seed in SEEDS
    ]


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
            base.dseed(spec.seed, "exp5_2", split, "loader"),
        )
        for split, partition in _partitions(data, cache).items()
    }


def _masked_sum(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = base.mask(lengths, sequence.shape[1]).to(sequence.dtype).unsqueeze(-1)
    return (sequence * valid).sum(dim=1)


def _valid_endpoint(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if torch.any(lengths <= 0):
        raise ValueError("All valid lengths must be positive")
    if torch.any(lengths > sequence.shape[1]):
        raise ValueError("A valid length exceeds the available trajectory")
    batch_index = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[batch_index, lengths - 1]


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
        base.dseed(seed, "exp5_2", probe_name, "linear_probe"),
    )


def _source_config(config: Config) -> exp501.Config:
    return exp501.Config(
        repo_root=config.repo_root,
        results_dir=local_source_root(config.results_dir),
        device=config.device,
        epochs=exp501.EPOCHS,
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
        raise ValueError(f"Unknown Exp5.2 seed: {seed}")
    destination = local_cache_path(config.results_dir, seed)
    meta_path = local_cache_meta_path(config.results_dir, seed)
    reference_path = local_reference_path(config.results_dir, seed)
    if destination.exists() and meta_path.exists() and reference_path.exists() and not force:
        return destination

    torch.set_num_threads(config.threads)
    source_spec = exp501.RunSpec(LOCAL_SOURCE_CONDITION, seed)
    source_config = _source_config(config)
    exp501.train_one(source_spec, data, source_config, force=force)
    model, source_payload = exp501.load_model(source_spec, data, source_config)
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
                base.dseed(seed, "exp5_2", split, "local_cache"),
            )
            parts: list[np.ndarray] = []
            for Xb, _, _ in loader:
                spikes = model.layer_features(Xb.to(device))["L2"]  # type: ignore[attr-defined]
                parts.append(spikes.detach().cpu().numpy().astype(np.uint8))
            arrays[f"X{split}"] = np.concatenate(parts, axis=0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    source_checkpoint = exp501.checkpoint_path(source_config.results_dir, source_spec)
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
            "source_best_epoch": source_payload["result"]["best_epoch"],
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

    cache = load_local_cache(seed, data, config)
    reference = collect_local_reference_features(data, cache, seed, config)
    probes = [
        {"probe_type": name, **_probe_metrics(reference, name, seed)}
        for name in LOCAL_REFERENCE_PROBES
    ]
    _save_json(
        reference_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "probes": probes,
        },
    )
    return destination


def load_local_cache(
    seed: int,
    data: base.Data,
    config: Config,
) -> dict[str, np.ndarray]:
    path = local_cache_path(config.results_dir, seed)
    meta_path = local_cache_meta_path(config.results_dir, seed)
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing Exp5.2 frozen L2 cache for seed {seed}; run prepare-local first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_identity = (PROTOCOL_VERSION, seed, LOCAL_SOURCE_CONDITION)
    actual_identity = (
        str(meta.get("protocol_version")),
        int(meta.get("seed")),
        str(meta.get("source_condition")),
    )
    if actual_identity != expected_identity:
        raise ValueError(f"Frozen L2 cache identity mismatch: {actual_identity} != {expected_identity}")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].astype(np.float32) for name in loaded.files}
    expected_shapes = {
        "Xtrain": (len(data.ytr), data.T, LOCAL_WIDTH),
        "Xval": (len(data.yva), data.T, LOCAL_WIDTH),
        "Xtest": (len(data.yte), data.T, LOCAL_WIDTH),
    }
    for name, shape in expected_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            actual = arrays[name].shape if name in arrays else None
            raise ValueError(f"Frozen L2 cache {name} shape mismatch: {actual} != {shape}")
    return arrays


def collect_local_reference_features(
    data: base.Data,
    cache: dict[str, np.ndarray],
    seed: int,
    config: Config,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    device = torch.device(config.device)
    for split, (X, y, lengths) in _partitions(data, cache).items():
        loader = _loader(
            X,
            y,
            lengths,
            config.batch_size,
            False,
            base.dseed(seed, "exp5_2", split, "local_reference_loader"),
        )
        y_parts: list[np.ndarray] = []
        whole_parts: list[np.ndarray] = []
        fixed_parts: list[np.ndarray] = []
        relative_parts: list[np.ndarray] = []
        with torch.no_grad():
            for l2, yb, lb in loader:
                l2 = l2.to(device)
                lb_device = lb.to(device)
                whole_parts.append(_masked_sum(l2, lb_device).cpu().numpy())
                fixed_parts.append(
                    base.fixed_counts(l2, lb_device, data.bin_steps).flatten(start_dim=1).cpu().numpy()
                )
                relative_parts.append(
                    base.relative_counts(l2, lb_device, base.N_REL).flatten(start_dim=1).cpu().numpy()
                )
                y_parts.append(yb.numpy())
        out[split] = {
            "y": np.concatenate(y_parts),
            "local_whole_count": np.concatenate(whole_parts),
            "local_fixed250_ordered": np.concatenate(fixed_parts),
            "local_relative10_ordered": np.concatenate(relative_parts),
        }
    return out


class EndpointMemoryDecoder(nn.Module):
    """One-layer FF/RSNN temporal decoder trained directly on hidden Uend."""

    def __init__(
        self,
        architecture: str,
        shift_mem_r: int,
        n_classes: int,
        fs: float,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        if shift_mem_r not in SHIFT_MEM_R:
            raise ValueError(f"Unsupported shift_mem_r: {shift_mem_r}")
        self.architecture = architecture
        self.shift_mem_r = int(shift_mem_r)
        self.hidden_width = TEMPORAL_WIDTH
        self.fs = float(fs)
        self.beta = beta_from_shift(self.shift_mem_r)
        self.tau_mem_ms = tau_ms_from_shift(self.shift_mem_r, self.fs)
        spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

        self.input_hidden = nn.Linear(LOCAL_WIDTH, TEMPORAL_WIDTH, bias=False)
        self.recurrent = (
            nn.Linear(TEMPORAL_WIDTH, TEMPORAL_WIDTH, bias=False)
            if architecture == "rsnn"
            else None
        )
        self.hidden_lif = snn.Leaky(
            beta=self.beta,
            threshold=THRESHOLD,
            spike_grad=spike_grad,
            reset_mechanism=RESET,
        )
        self.endpoint_head = nn.Linear(TEMPORAL_WIDTH, n_classes, bias=True)

    def forward_trajectory(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_steps, _ = x.shape
        hidden_mem = torch.zeros(batch, self.hidden_width, device=x.device, dtype=x.dtype)
        prev_spike = torch.zeros_like(hidden_mem)
        spike_parts: list[torch.Tensor] = []
        mem_parts: list[torch.Tensor] = []
        for timestep in range(n_steps):
            current = self.input_hidden(x[:, timestep])
            if self.recurrent is not None:
                current = current + self.recurrent(prev_spike)
            spike, hidden_mem = self.hidden_lif(current, hidden_mem)
            spike_parts.append(spike)
            mem_parts.append(hidden_mem)
            prev_spike = spike
        return torch.stack(spike_parts, dim=1), torch.stack(mem_parts, dim=1)

    def endpoint_logits(self, membranes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.endpoint_head(_valid_endpoint(membranes, lengths))

    def loss_logits(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spikes, membranes = self.forward_trajectory(x)
        logits = self.endpoint_logits(membranes, lengths)
        return F.cross_entropy(logits, y), logits, spikes, membranes

    def tail_event_count(
        self,
        endpoint_spike: torch.Tensor,
        endpoint_membrane: torch.Tensor,
        n_tail_steps: int,
    ) -> float:
        prev_spike = endpoint_spike
        hidden_mem = endpoint_membrane
        tail_events = 0.0
        for _ in range(n_tail_steps):
            current = torch.zeros_like(hidden_mem)
            if self.recurrent is not None:
                current = current + self.recurrent(prev_spike)
            spike, hidden_mem = self.hidden_lif(current, hidden_mem)
            tail_events += float(spike.sum().item())
            prev_spike = spike
        return tail_events


def _initialize_model(
    spec: RunSpec,
    n_classes: int,
    fs: float,
    device: torch.device,
) -> EndpointMemoryDecoder:
    base.seed_all(base.dseed(spec.seed, "exp5_2", "constructor"))
    model = EndpointMemoryDecoder(
        spec.architecture,
        spec.shift_mem_r,
        n_classes,
        fs,
    ).to(device)
    base.seed_all(base.dseed(spec.seed, "exp5_2", "input_hidden"))
    model.input_hidden.reset_parameters()
    if model.recurrent is not None:
        base.seed_all(base.dseed(spec.seed, "exp5_2", "recurrent"))
        model.recurrent.reset_parameters()
    base.seed_all(base.dseed(spec.seed, "exp5_2", "endpoint_head"))
    model.endpoint_head.reset_parameters()
    return model


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return base.metrics(y_true, y_pred)


def evaluate_native(
    model: EndpointMemoryDecoder,
    loader: DataLoader,
    device: torch.device,
    fs: float,
) -> dict[str, float]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    valid_events = 0.0
    valid_seconds = 0.0
    tail_events = 0.0
    endpoint_norm_sum = 0.0
    tail_steps = int(round(TAIL_SECONDS * float(fs)))

    with torch.no_grad():
        for l2, y, lengths in loader:
            l2 = l2.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            loss, logits, spikes, membranes = model.loss_logits(l2, lengths, y)
            n = len(y)
            loss_sum += float(loss.item()) * n
            n_total += n
            y_true_parts.append(y.cpu().numpy())
            y_pred_parts.append(logits.argmax(dim=1).cpu().numpy())

            valid = base.mask(lengths, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
            valid_events += float((spikes * valid).sum().item())
            valid_seconds += float(lengths.sum().item()) / float(fs)
            endpoint_spike = _valid_endpoint(spikes, lengths)
            endpoint_membrane = _valid_endpoint(membranes, lengths)
            endpoint_norm_sum += float(torch.linalg.vector_norm(endpoint_membrane, dim=1).sum().item())
            tail_events += model.tail_event_count(endpoint_spike, endpoint_membrane, tail_steps)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    out = _classification_metrics(y_true, y_pred)
    out["loss"] = loss_sum / max(n_total, 1)
    out["hidden_events_per_neuron_second"] = valid_events / max(valid_seconds * TEMPORAL_WIDTH, EPS)
    out["hidden_tail_event_fraction"] = tail_events / max(valid_events + tail_events, EPS)
    out["hidden_tail_events_per_neuron_second"] = tail_events / max(
        n_total * TAIL_SECONDS * TEMPORAL_WIDTH, EPS
    )
    out["endpoint_membrane_l2_mean"] = endpoint_norm_sum / max(n_total, 1)
    return out


def collect_temporal_features(
    model: EndpointMemoryDecoder,
    data: base.Data,
    cache: dict[str, np.ndarray],
    spec: RunSpec,
    config: Config,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
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
            for l2, y, lengths in loader:
                l2 = l2.to(device)
                lengths_device = lengths.to(device)
                spikes, membranes = model.forward_trajectory(l2)
                whole_parts.append(_masked_sum(spikes, lengths_device).cpu().numpy())
                fixed_parts.append(
                    base.fixed_counts(spikes, lengths_device, data.bin_steps).flatten(start_dim=1).cpu().numpy()
                )
                relative_parts.append(
                    base.relative_counts(spikes, lengths_device, base.N_REL).flatten(start_dim=1).cpu().numpy()
                )
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
    endpoint_head = TEMPORAL_WIDTH * n_classes + n_classes
    return {
        "input_hidden": input_hidden,
        "recurrent": recurrent,
        "endpoint_head": endpoint_head,
        "trainable_total": input_hidden + recurrent + endpoint_head,
    }


def provenance(spec: RunSpec, data: base.Data, config: Config) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": spec.seed,
        "split_seed": base.SPLIT_SEED,
        "architecture": spec.architecture,
        "shift_mem_r": spec.shift_mem_r,
        "beta_r": beta_from_shift(spec.shift_mem_r),
        "tau_mem_r_ms": tau_ms_from_shift(spec.shift_mem_r, data.fs),
        "temporal_width": TEMPORAL_WIDTH,
        "local_source_experiment": exp501.EXPERIMENT_ID,
        "local_source_protocol": exp501.PROTOCOL_VERSION,
        "local_source_condition": LOCAL_SOURCE_CONDITION,
        "local_source_frozen": True,
        "local_width": LOCAL_WIDTH,
        "local_l1_shifts": (2, 3, 4),
        "local_l2_shifts": (2, 3, 4),
        "sampling_rate_hz": float(data.fs),
        "temporal_update_rate": "raw 64Hz frozen L2 trajectory; no pooling/compressor/downsampling",
        "separate_temporal_synaptic_state": False,
        "head": "Linear(hidden Uend 128 -> classes, bias=True); no output LIF",
        "objective": "endpoint_ce = CE(Linear(U_hidden[valid_end]), y)",
        "checkpoint_selection": "max validation native endpoint BA; tie-break validation CE",
        "threshold": THRESHOLD,
        "reset": RESET,
        "tail_seconds": TAIL_SECONDS,
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
        for l2, y, lengths in train_loaders["train"]:
            l2 = l2.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, logits, _, _ = model.loss_logits(l2, lengths, y)
            loss.backward()
            optimizer.step()
            n = len(y)
            train_n += n
            train_loss_sum += float(loss.item()) * n
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = _classification_metrics(np.concatenate(train_true), np.concatenate(train_pred))
        val_metrics = evaluate_native(model, eval_loaders["val"], device, data.fs)
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
    final_native = {
        split: evaluate_native(model, loader, device, data.fs)
        for split, loader in eval_loaders.items()
    }
    result = {
        "spec": spec.__dict__,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_ba,
        "best_val_loss": best_val_loss,
        "native": final_native,
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
    hpath = history_path(config.results_dir, spec)
    hpath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hpath, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[EndpointMemoryDecoder, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.2 checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong Exp5.2 checkpoint identity: {path}")
    if payload.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint spec mismatch: {path}")
    model = EndpointMemoryDecoder(
        spec.architecture,
        spec.shift_mem_r,
        len(data.labels),
        data.fs,
    ).to(torch.device(config.device))
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
        split: evaluate_native(model, loader, device, data.fs)
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


def run_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    train_one(spec, data, config, force=force)
    return evaluate_one(spec, data, config, force=force)


def _probe_value(payload: dict[str, object], probe_type: str, key: str) -> float:
    probes = payload["probes"]  # type: ignore[index]
    for probe in probes:
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
            raise FileNotFoundError(f"Missing Exp5.2 evaluation: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != PROTOCOL_VERSION or payload.get("spec") != spec.__dict__:
            raise ValueError(f"Evaluation identity mismatch: {path}")
        row: dict[str, object] = {
            "architecture": spec.architecture,
            "shift_mem_r": spec.shift_mem_r,
            "tau_mem_r_ms": tau_ms_from_shift(spec.shift_mem_r, 64.0),
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
                "hidden_tail_events_per_neuron_second",
                "endpoint_membrane_l2_mean",
            ):
                row[f"native_{split}_{metric}"] = native[metric]
        for probe in TEMPORAL_PROBES:
            row[f"{probe}_val_ba"] = _probe_value(payload, probe, "probe_val_balanced_accuracy")
            row[f"{probe}_test_ba"] = _probe_value(payload, probe, "probe_test_balanced_accuracy")
            row[f"{probe}_C"] = _probe_value(payload, probe, "probe_C")
        rows.append(row)

        hpath = history_path(root, spec)
        if not hpath.exists():
            raise FileNotFoundError(f"Missing Exp5.2 history: {hpath}")
        history = pd.read_csv(hpath)
        history.insert(0, "seed", spec.seed)
        history.insert(0, "tau_mem_r_ms", tau_ms_from_shift(spec.shift_mem_r, 64.0))
        history.insert(0, "shift_mem_r", spec.shift_mem_r)
        history.insert(0, "architecture", spec.architecture)
        history_parts.append(history)

    reference_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        path = local_reference_path(root, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp5.2 local reference: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != PROTOCOL_VERSION or int(payload.get("seed")) != seed:
            raise ValueError(f"Local reference identity mismatch: {path}")
        for probe in payload["probes"]:
            reference_rows.append(
                {
                    "seed": seed,
                    "probe_type": probe["probe_type"],
                    "probe_C": probe["probe_C"],
                    "val_ba": probe["probe_val_balanced_accuracy"],
                    "test_ba": probe["probe_test_balanced_accuracy"],
                }
            )

    root.mkdir(parents=True, exist_ok=True)
    runs_path = root / "runs.csv"
    histories_path = root / "histories.csv"
    local_reference_out = root / "local_reference.csv"
    pd.DataFrame(rows).to_csv(runs_path, index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(histories_path, index=False)
    pd.DataFrame(reference_rows).to_csv(local_reference_out, index=False)

    manifest_path = root / "manifest.json"
    _save_json(
        manifest_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "expected_decoder_runs": len(run_specs()),
            "expected_local_cache_seeds": len(SEEDS),
            "architectures": ARCHITECTURES,
            "shift_mem_r": SHIFT_MEM_R,
            "tau_mem_r_ms_at_64hz": [tau_ms_from_shift(shift, 64.0) for shift in SHIFT_MEM_R],
            "seeds": SEEDS,
            "primary_objective": "endpoint_ce",
            "primary_metric": "native validation/test balanced accuracy from hidden Uend endpoint head",
            "selection_policy": "notebook selects tau by mean validation native BA only; test BA never selects",
            "aggregation_policy": "finalizer concatenates per-run/per-seed artifacts only; notebook computes summaries, paired effects, timing recovery, selected condition, and plots",
            "tail_seconds": TAIL_SECONDS,
            "files": {
                "runs": runs_path.name,
                "histories": histories_path.name,
                "local_reference": local_reference_out.name,
            },
        },
    )
    return {
        "runs": runs_path,
        "histories": histories_path,
        "local_reference": local_reference_out,
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
    parser = argparse.ArgumentParser(description="Experiment 5.2 frozen-local FF/RSNN tauR sweep")
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
        path = prepare_local_seed(SEEDS[task_id], data, config, force=args.force)
        print(path)
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
