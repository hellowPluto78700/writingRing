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
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_5_2_frozen_local_tauR_sweep as exp52


EXPERIMENT_ID = "experiment_5_2_1_ff_multitau_objectives"
PROTOCOL_VERSION = "frozen_exp3_l2_ff_multitau_objectives_v1"
SOURCE_EXPERIMENT_ID = exp52.EXPERIMENT_ID
SOURCE_PROTOCOL_VERSION = exp52.PROTOCOL_VERSION
SEEDS = exp52.SEEDS
MEMORY_MODES = ("single_shift7", "multitau_4567")
OBJECTIVES = ("endpoint_ce", "endpoint_fixed250_aux_ce")
SINGLE_SHIFT = 7
MULTI_SHIFTS = (4, 5, 6, 7)
TEMPORAL_WIDTH = exp52.TEMPORAL_WIDTH
LOCAL_WIDTH = exp52.LOCAL_WIDTH
THRESHOLD = exp52.THRESHOLD
SURROGATE_SLOPE = exp52.SURROGATE_SLOPE
RESET = exp52.RESET
EPOCHS = exp52.EPOCHS
BATCH_SIZE = exp52.BATCH_SIZE
LR = exp52.LR
WEIGHT_DECAY = exp52.WEIGHT_DECAY
TAIL_SECONDS = exp52.TAIL_SECONDS
AUX_LAMBDA = 1.0
EPS = 1e-8

TEMPORAL_PROBES = exp52.TEMPORAL_PROBES
LOCAL_REFERENCE_PROBES = exp52.LOCAL_REFERENCE_PROBES


@dataclass(frozen=True)
class RunSpec:
    memory_mode: str
    objective: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.memory_mode}__{self.objective}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp52.find_repo_root(start)


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
    return [
        RunSpec(memory_mode, objective, seed)
        for memory_mode in MEMORY_MODES
        for objective in OBJECTIVES
        for seed in SEEDS
    ]


def memory_shifts(memory_mode: str) -> tuple[int, ...]:
    if memory_mode == "single_shift7":
        return (SINGLE_SHIFT,)
    if memory_mode == "multitau_4567":
        return MULTI_SHIFTS
    raise ValueError(f"Unknown memory mode: {memory_mode}")


def memory_group_slices(memory_mode: str) -> dict[int, slice]:
    shifts = memory_shifts(memory_mode)
    if TEMPORAL_WIDTH % len(shifts) != 0:
        raise ValueError("TEMPORAL_WIDTH must divide evenly across memory groups")
    group_width = TEMPORAL_WIDTH // len(shifts)
    return {
        shift: slice(index * group_width, (index + 1) * group_width)
        for index, shift in enumerate(shifts)
    }


def beta_vector(memory_mode: str) -> torch.Tensor:
    values = torch.empty(TEMPORAL_WIDTH, dtype=torch.float32)
    for shift, group in memory_group_slices(memory_mode).items():
        values[group] = exp52.beta_from_shift(shift)
    return values


def tau_profile_ms(memory_mode: str, fs: float) -> tuple[float, ...]:
    return tuple(exp52.tau_ms_from_shift(shift, fs) for shift in memory_shifts(memory_mode))


def _source_config(config: Config) -> exp52.Config:
    return exp52.Config(
        repo_root=config.repo_root,
        results_dir=source_results_dir(config.repo_root),
        device=config.device,
        epochs=exp52.EPOCHS,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def load_source_cache(seed: int, data: base.Data, config: Config) -> dict[str, np.ndarray]:
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}")
    return exp52.load_local_cache(seed, data, _source_config(config))


def _loader(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return exp52._loader(X, y, lengths, batch_size, shuffle, seed)


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
            base.dseed(spec.seed, "exp5_2_1", split, "loader"),
        )
        for split, partition in _partitions(data, cache).items()
    }


class FFTemporalDecoder(nn.Module):
    """One-layer FF temporal state model with single- or multi-tau membrane memory."""

    def __init__(
        self,
        memory_mode: str,
        objective: str,
        n_classes: int,
        T: int,
        fs: float,
        bin_steps: int,
    ) -> None:
        super().__init__()
        if memory_mode not in MEMORY_MODES:
            raise ValueError(f"Unknown memory mode: {memory_mode}")
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective: {objective}")
        self.memory_mode = memory_mode
        self.objective = objective
        self.hidden_width = TEMPORAL_WIDTH
        self.T = int(T)
        self.fs = float(fs)
        self.bin_steps = int(bin_steps)
        self.n_fixed_bins = int(math.ceil(self.T / self.bin_steps))
        self.beta = beta_vector(memory_mode)
        spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

        self.input_hidden = nn.Linear(LOCAL_WIDTH, TEMPORAL_WIDTH, bias=False)
        self.hidden_lif = snn.Leaky(
            beta=self.beta,
            threshold=THRESHOLD,
            spike_grad=spike_grad,
            reset_mechanism=RESET,
        )
        self.endpoint_head = nn.Linear(TEMPORAL_WIDTH, n_classes, bias=True)
        self.aux_head = (
            nn.Linear(self.n_fixed_bins * TEMPORAL_WIDTH, n_classes, bias=True)
            if objective == "endpoint_fixed250_aux_ce"
            else None
        )

    def forward_trajectory(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_steps, _ = x.shape
        hidden_mem = torch.zeros(batch, self.hidden_width, device=x.device, dtype=x.dtype)
        spike_parts: list[torch.Tensor] = []
        mem_parts: list[torch.Tensor] = []
        for timestep in range(n_steps):
            current = self.input_hidden(x[:, timestep])
            spike, hidden_mem = self.hidden_lif(current, hidden_mem)
            spike_parts.append(spike)
            mem_parts.append(hidden_mem)
        return torch.stack(spike_parts, dim=1), torch.stack(mem_parts, dim=1)

    def endpoint_logits(self, membranes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.endpoint_head(exp52._valid_endpoint(membranes, lengths))

    def auxiliary_logits(self, spikes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor | None:
        if self.aux_head is None:
            return None
        features = base.fixed_counts(spikes, lengths, self.bin_steps).flatten(start_dim=1)
        if features.shape[1] != self.aux_head.in_features:
            raise ValueError(
                f"Fixed250 feature size {features.shape[1]} != aux head input {self.aux_head.in_features}"
            )
        return self.aux_head(features)

    def loss_logits(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        spikes, membranes = self.forward_trajectory(x)
        endpoint_logits = self.endpoint_logits(membranes, lengths)
        endpoint_ce = F.cross_entropy(endpoint_logits, y)
        aux_logits = self.auxiliary_logits(spikes, lengths)
        if aux_logits is None:
            aux_ce = None
            total_loss = endpoint_ce
        else:
            aux_ce = F.cross_entropy(aux_logits, y)
            total_loss = endpoint_ce + AUX_LAMBDA * aux_ce
        return total_loss, endpoint_ce, aux_ce, endpoint_logits, spikes, membranes

    def tail_event_count(
        self,
        endpoint_membrane: torch.Tensor,
        n_tail_steps: int,
    ) -> float:
        hidden_mem = endpoint_membrane
        tail_events = 0.0
        for _ in range(n_tail_steps):
            current = torch.zeros_like(hidden_mem)
            spike, hidden_mem = self.hidden_lif(current, hidden_mem)
            tail_events += float(spike.sum().item())
        return tail_events


def _initialize_model(
    spec: RunSpec,
    data: base.Data,
    device: torch.device,
) -> FFTemporalDecoder:
    base.seed_all(base.dseed(spec.seed, "exp5_2_1", "constructor"))
    model = FFTemporalDecoder(
        spec.memory_mode,
        spec.objective,
        len(data.labels),
        data.T,
        data.fs,
        data.bin_steps,
    ).to(device)
    # Keep input and deployed endpoint-head initializations paired across all four conditions.
    base.seed_all(base.dseed(spec.seed, "exp5_2_1", "input_hidden"))
    model.input_hidden.reset_parameters()
    base.seed_all(base.dseed(spec.seed, "exp5_2_1", "endpoint_head"))
    model.endpoint_head.reset_parameters()
    if model.aux_head is not None:
        base.seed_all(base.dseed(spec.seed, "exp5_2_1", "aux_head"))
        model.aux_head.reset_parameters()
    return model


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return base.metrics(y_true, y_pred)


def evaluate_native(
    model: FFTemporalDecoder,
    loader: DataLoader,
    device: torch.device,
    fs: float,
) -> dict[str, object]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    total_loss_sum = 0.0
    endpoint_loss_sum = 0.0
    aux_loss_sum = 0.0
    aux_n = 0
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
            total_loss, endpoint_ce, aux_ce, logits, spikes, membranes = model.loss_logits(
                l2, lengths, y
            )
            n = len(y)
            n_total += n
            total_loss_sum += float(total_loss.item()) * n
            endpoint_loss_sum += float(endpoint_ce.item()) * n
            if aux_ce is not None:
                aux_loss_sum += float(aux_ce.item()) * n
                aux_n += n
            y_true_parts.append(y.cpu().numpy())
            y_pred_parts.append(logits.argmax(dim=1).cpu().numpy())

            valid = base.mask(lengths, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
            valid_events += float((spikes * valid).sum().item())
            valid_seconds += float(lengths.sum().item()) / float(fs)
            endpoint_membrane = exp52._valid_endpoint(membranes, lengths)
            endpoint_norm_sum += float(
                torch.linalg.vector_norm(endpoint_membrane, dim=1).sum().item()
            )
            tail_events += model.tail_event_count(endpoint_membrane, tail_steps)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    out: dict[str, object] = _classification_metrics(y_true, y_pred)
    out["loss"] = total_loss_sum / max(n_total, 1)
    out["endpoint_loss"] = endpoint_loss_sum / max(n_total, 1)
    out["aux_loss"] = aux_loss_sum / aux_n if aux_n else None
    out["hidden_events_per_neuron_second"] = valid_events / max(
        valid_seconds * TEMPORAL_WIDTH, EPS
    )
    out["hidden_tail_event_fraction"] = tail_events / max(valid_events + tail_events, EPS)
    out["hidden_tail_events_per_neuron_second"] = tail_events / max(
        n_total * TAIL_SECONDS * TEMPORAL_WIDTH, EPS
    )
    out["endpoint_membrane_l2_mean"] = endpoint_norm_sum / max(n_total, 1)
    return out


def collect_temporal_features(
    model: FFTemporalDecoder,
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
                whole_parts.append(exp52._masked_sum(spikes, lengths_device).cpu().numpy())
                fixed_parts.append(
                    base.fixed_counts(spikes, lengths_device, data.bin_steps)
                    .flatten(start_dim=1)
                    .cpu()
                    .numpy()
                )
                relative_parts.append(
                    base.relative_counts(spikes, lengths_device, base.N_REL)
                    .flatten(start_dim=1)
                    .cpu()
                    .numpy()
                )
                uend_parts.append(exp52._valid_endpoint(membranes, lengths_device).cpu().numpy())
                y_parts.append(y.numpy())
            out[split] = {
                "y": np.concatenate(y_parts),
                "hidden_whole_count": np.concatenate(whole_parts),
                "hidden_fixed250_ordered": np.concatenate(fixed_parts),
                "hidden_relative10_ordered": np.concatenate(relative_parts),
                "hidden_uend": np.concatenate(uend_parts),
            }
    return out


def parameter_counts(spec: RunSpec, data: base.Data) -> dict[str, int]:
    n_classes = len(data.labels)
    input_hidden = LOCAL_WIDTH * TEMPORAL_WIDTH
    endpoint_head = TEMPORAL_WIDTH * n_classes + n_classes
    n_fixed_bins = int(math.ceil(data.T / data.bin_steps))
    aux_head = (
        n_fixed_bins * TEMPORAL_WIDTH * n_classes + n_classes
        if spec.objective == "endpoint_fixed250_aux_ce"
        else 0
    )
    return {
        "input_hidden": input_hidden,
        "endpoint_head": endpoint_head,
        "aux_head_training_only": aux_head,
        "deployed_trainable_total": input_hidden + endpoint_head,
        "training_trainable_total": input_hidden + endpoint_head + aux_head,
    }


def provenance(spec: RunSpec, data: base.Data, config: Config) -> dict[str, object]:
    slices = memory_group_slices(spec.memory_mode)
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": spec.seed,
        "split_seed": base.SPLIT_SEED,
        "architecture": "ff",
        "memory_mode": spec.memory_mode,
        "memory_shifts": memory_shifts(spec.memory_mode),
        "memory_group_widths": {
            str(shift): int(group.stop - group.start) for shift, group in slices.items()
        },
        "tau_mem_ms": tau_profile_ms(spec.memory_mode, data.fs),
        "objective": spec.objective,
        "aux_lambda": AUX_LAMBDA if spec.objective == "endpoint_fixed250_aux_ce" else 0.0,
        "auxiliary_supervision": (
            "training-only hidden Fixed250 ordered spike-count CE; auxiliary head is discarded at deployment"
            if spec.objective == "endpoint_fixed250_aux_ce"
            else "none"
        ),
        "deployed_readout": "Linear(hidden Uend 128 -> classes, bias=True); no output LIF",
        "primary_loss": "endpoint_ce = CE(Linear(U_hidden[valid_end]), y)",
        "checkpoint_selection": "max validation native endpoint BA; tie-break validation endpoint CE",
        "source_experiment": SOURCE_EXPERIMENT_ID,
        "source_protocol": SOURCE_PROTOCOL_VERSION,
        "source_results_dir": str(source_results_dir(config.repo_root).relative_to(config.repo_root)),
        "source_cache_reused": True,
        "source_local_frozen": True,
        "local_width": LOCAL_WIDTH,
        "sampling_rate_hz": float(data.fs),
        "temporal_update_rate": "raw 64Hz frozen L2 trajectory; no pooling/compressor/downsampling",
        "threshold": THRESHOLD,
        "reset": RESET,
        "tail_seconds": TAIL_SECONDS,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "parameter_counts": parameter_counts(spec, data),
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
    cache = load_source_cache(spec.seed, data, config)
    train_loaders = _make_loaders(data, cache, spec, config, train_shuffle=True)
    eval_loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _initialize_model(spec, data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_ba = -np.inf
    best_val_endpoint_loss = np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int | None]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_total_sum = 0.0
        train_endpoint_sum = 0.0
        train_aux_sum = 0.0
        train_aux_n = 0
        train_n = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for l2, y, lengths in train_loaders["train"]:
            l2 = l2.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            total_loss, endpoint_ce, aux_ce, logits, _, _ = model.loss_logits(l2, lengths, y)
            total_loss.backward()
            optimizer.step()
            n = len(y)
            train_n += n
            train_total_sum += float(total_loss.item()) * n
            train_endpoint_sum += float(endpoint_ce.item()) * n
            if aux_ce is not None:
                train_aux_sum += float(aux_ce.item()) * n
                train_aux_n += n
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = _classification_metrics(np.concatenate(train_true), np.concatenate(train_pred))
        val_metrics = evaluate_native(model, eval_loaders["val"], device, data.fs)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_endpoint_loss = float(val_metrics["endpoint_loss"])
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_total_sum / max(train_n, 1),
                "train_endpoint_loss": train_endpoint_sum / max(train_n, 1),
                "train_aux_loss": train_aux_sum / train_aux_n if train_aux_n else None,
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_total_loss": val_metrics["loss"],
                "val_endpoint_loss": val_endpoint_loss,
                "val_aux_loss": val_metrics["aux_loss"],
                "val_balanced_accuracy": val_ba,
            }
        )
        improved = val_ba > best_val_ba + 1e-12 or (
            abs(val_ba - best_val_ba) <= 1e-12
            and val_endpoint_loss < best_val_endpoint_loss - 1e-12
        )
        if improved:
            best_val_ba = val_ba
            best_val_endpoint_loss = val_endpoint_loss
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
        "best_val_endpoint_loss": best_val_endpoint_loss,
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
) -> tuple[FFTemporalDecoder, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.2.1 checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong Exp5.2.1 checkpoint identity: {path}")
    if payload.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint spec mismatch: {path}")
    model = FFTemporalDecoder(
        spec.memory_mode,
        spec.objective,
        len(data.labels),
        data.T,
        data.fs,
        data.bin_steps,
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
    cache = load_source_cache(spec.seed, data, config)
    model, checkpoint = load_model(spec, data, config)
    device = torch.device(config.device)
    eval_loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)
    native = {
        split: evaluate_native(model, loader, device, data.fs)
        for split, loader in eval_loaders.items()
    }
    extracted = collect_temporal_features(model, data, cache, spec, config)
    probes = [
        {"probe_type": probe, **exp52._probe_metrics(extracted, probe, spec.seed)}
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
            raise FileNotFoundError(f"Missing Exp5.2.1 evaluation: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != PROTOCOL_VERSION or payload.get("spec") != spec.__dict__:
            raise ValueError(f"Evaluation identity mismatch: {path}")
        row: dict[str, object] = {
            "memory_mode": spec.memory_mode,
            "objective": spec.objective,
            "seed": spec.seed,
            "tau_profile_ms": ",".join(f"{value:.3f}" for value in tau_profile_ms(spec.memory_mode, 64.0)),
            "best_epoch": payload["best_epoch"],
        }
        for split in ("train", "val", "test"):
            native = payload["native"][split]
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "loss",
                "endpoint_loss",
                "aux_loss",
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
            raise FileNotFoundError(f"Missing Exp5.2.1 history: {hpath}")
        history = pd.read_csv(hpath)
        history.insert(0, "seed", spec.seed)
        history.insert(0, "objective", spec.objective)
        history.insert(0, "memory_mode", spec.memory_mode)
        history_parts.append(history)

    source_root = source_results_dir(repo_root)
    reference_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        path = exp52.local_reference_path(source_root, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp5.2 source local reference: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != SOURCE_PROTOCOL_VERSION or int(payload.get("seed")) != seed:
            raise ValueError(f"Source local reference identity mismatch: {path}")
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
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "source_protocol_version": SOURCE_PROTOCOL_VERSION,
            "source_cache_policy": "reuse committed Exp5.2 frozen L2 caches; fail on missing/mismatched source",
            "expected_runs": len(run_specs()),
            "architecture": "ff",
            "memory_modes": MEMORY_MODES,
            "objectives": OBJECTIVES,
            "single_shift": SINGLE_SHIFT,
            "multi_shifts": MULTI_SHIFTS,
            "seeds": SEEDS,
            "aux_lambda": AUX_LAMBDA,
            "deployed_readout": "hidden Uend -> Linear -> class logits",
            "primary_metric": "native validation/test balanced accuracy from deployed Uend endpoint head",
            "checkpoint_selection": "per-run max validation native endpoint BA; tie-break validation endpoint CE",
            "analysis_policy": "notebook is analysis-only and computes paired memory-mode, objective, interaction, representation, and head-utilization effects",
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
    parser = argparse.ArgumentParser(
        description="Experiment 5.2.1 FF single-vs-multi-tau x endpoint-objective factorial"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-one")
    run.add_argument("--repo-root", type=str, default=None)
    run.add_argument("--device", type=str, default="cpu")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run.add_argument("--epochs", type=int, default=EPOCHS)
    run.add_argument("--force", action="store_true")
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
    specs = run_specs()
    task_id = int(args.array_task_id)
    if not 0 <= task_id < len(specs):
        raise IndexError(f"run-one task {task_id} outside 0..{len(specs)-1}")
    payload = run_one(specs[task_id], data, config, force=args.force)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
