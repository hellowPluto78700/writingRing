from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_2_3_objective_temporal_dynamics as exp723

EXPERIMENT_ID = "experiment_7_2_5_output_synaptic_alpha"
PROTOCOL_VERSION = "output_synaptic_alpha_v1"

SOURCE_FAMILIES = (exp723.S1, exp723.S2, exp723.S3, exp723.S4)
ARCHITECTURES = {name: exp723.ARCHITECTURES[name] for name in exp723.ARCHITECTURE_ORDER}
ARCHITECTURE_ORDER = tuple(ARCHITECTURES)
REGULARIZATIONS = exp723.REGULARIZATIONS
SEEDS = exp723.SEEDS
OBJECTIVES = ("whole_count_ce", "timestep_ce")
OUTPUT_ALPHAS = (0.0, 0.5)
OUTPUT_BETA = 0.5
OUTPUT_CAP = 1
HIDDEN_WIDTH = exp723.HIDDEN_WIDTH
MAX_EPOCHS = 100
MIN_EPOCHS = 20
PATIENCE = 30

VALID_WC_PROBE = "l2_wholecount_linear"
VALID_F250_PROBE = "l2_fixed250_linear"
PROBE_SOURCES = (VALID_WC_PROBE, VALID_F250_PROBE)

FAMILY_LABELS = {
    exp723.S1: "S1 spike WC, beta=.5",
    exp723.S2: "S2 spike TSCE, beta=.5",
    exp723.S3: "S3 spike WC, beta=1",
    exp723.S4: "S4 spike TSCE, beta=1",
}
MATCHED_SOURCE_BY_OBJECTIVE = {
    "whole_count_ce": exp723.S1,
    "timestep_ce": exp723.S2,
}


@dataclass(frozen=True)
class SourceSpec:
    architecture: str
    source_family: str
    regularization: str
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.architecture}__{self.source_family}__"
            f"{self.regularization}__seed{self.seed}"
        )


@dataclass(frozen=True)
class FrozenHeadSpec:
    architecture: str
    source_family: str
    regularization: str
    seed: int
    objective: str
    output_alpha: float

    @property
    def source(self) -> SourceSpec:
        return SourceSpec(self.architecture, self.source_family, self.regularization, self.seed)

    @property
    def key(self) -> str:
        alpha = "a00" if self.output_alpha == 0.0 else "a05"
        obj = "wc" if self.objective == "whole_count_ce" else "tsce"
        return f"{self.source.key}__{obj}__{alpha}__b05"


@dataclass(frozen=True)
class E2ESpec:
    architecture: str
    regularization: str
    seed: int
    objective: str
    output_alpha: float

    @property
    def key(self) -> str:
        alpha = "a00" if self.output_alpha == 0.0 else "a05"
        obj = "wc" if self.objective == "whole_count_ce" else "tsce"
        return f"{self.architecture}__{self.regularization}__seed{self.seed}__{obj}__{alpha}__b05"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp72.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_specs() -> list[SourceSpec]:
    return [
        SourceSpec(a, f, r, s)
        for a in ARCHITECTURE_ORDER
        for f in SOURCE_FAMILIES
        for r in REGULARIZATIONS
        for s in SEEDS
    ]


def frozen_head_specs() -> list[FrozenHeadSpec]:
    return [
        FrozenHeadSpec(src.architecture, src.source_family, src.regularization, src.seed, objective, alpha)
        for src in source_specs()
        for objective in OBJECTIVES
        for alpha in OUTPUT_ALPHAS
    ]


def e2e_specs() -> list[E2ESpec]:
    return [
        E2ESpec(a, r, s, objective, alpha)
        for a in ARCHITECTURE_ORDER
        for r in REGULARIZATIONS
        for s in SEEDS
        for objective in OBJECTIVES
        for alpha in OUTPUT_ALPHAS
    ]


def validate_source_spec(spec: SourceSpec) -> None:
    if spec.architecture not in ARCHITECTURES:
        raise ValueError(spec.architecture)
    if spec.source_family not in SOURCE_FAMILIES:
        raise ValueError(spec.source_family)
    if spec.regularization not in REGULARIZATIONS:
        raise ValueError(spec.regularization)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def validate_frozen_spec(spec: FrozenHeadSpec) -> None:
    validate_source_spec(spec.source)
    if spec.objective not in OBJECTIVES:
        raise ValueError(spec.objective)
    if spec.output_alpha not in OUTPUT_ALPHAS:
        raise ValueError(spec.output_alpha)


def validate_e2e_spec(spec: E2ESpec) -> None:
    if spec.architecture not in ARCHITECTURES:
        raise ValueError(spec.architecture)
    if spec.regularization not in REGULARIZATIONS:
        raise ValueError(spec.regularization)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    if spec.objective not in OBJECTIVES:
        raise ValueError(spec.objective)
    if spec.output_alpha not in OUTPUT_ALPHAS:
        raise ValueError(spec.output_alpha)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _source_path(root: Path, kind: str, spec: SourceSpec, suffix: str) -> Path:
    return root / kind / spec.source_family / f"{spec.key}{suffix}"


def _frozen_path(root: Path, kind: str, spec: FrozenHeadSpec, suffix: str) -> Path:
    return root / kind / "frozen_head" / spec.source_family / f"{spec.key}{suffix}"


def _e2e_path(root: Path, kind: str, spec: E2ESpec, suffix: str) -> Path:
    return root / kind / "e2e" / spec.objective / f"{spec.key}{suffix}"


def _source_exp723_spec(spec: SourceSpec) -> exp723.RunSpec:
    return exp723.RunSpec(spec.architecture, spec.source_family, spec.regularization, spec.seed)


def frozen_pair_seed(spec: FrozenHeadSpec, role: str) -> int:
    """Pair alpha=0 and alpha=.5 by excluding alpha from every random seed."""
    return exp3.dseed(
        spec.seed,
        EXPERIMENT_ID,
        "frozen",
        spec.architecture,
        spec.source_family,
        spec.regularization,
        spec.objective,
        role,
    )


def e2e_pair_seed(spec: E2ESpec, role: str) -> int:
    """Pair alpha=0 and alpha=.5 by excluding alpha from every random seed."""
    return exp3.dseed(
        spec.seed,
        EXPERIMENT_ID,
        "e2e",
        spec.architecture,
        spec.regularization,
        spec.objective,
        role,
    )


def _base_exp723_config(config: Config) -> exp723.Config:
    return exp723.Config(
        repo_root=config.repo_root,
        results_dir=exp723.results_dir(config.repo_root),
        device=config.device,
        batch_size=config.batch_size,
        threads=config.threads,
        max_epochs=exp723.MAX_EPOCHS,
    )


def _assert_binary_l2(array: np.ndarray, name: str) -> None:
    if not np.all((array == 0) | (array == 1)):
        bad = array[(array != 0) & (array != 1)]
        raise RuntimeError(f"{name} is not binary; first values={bad[:8]}")


def prepare_frozen_cache(spec: SourceSpec, data: exp3.Data, config: Config, force: bool = False) -> Path:
    validate_source_spec(spec)
    destination = _source_path(config.results_dir, "frozen_l2_cache", spec, ".npz")
    metadata_path = _source_path(config.results_dir, "frozen_l2_cache", spec, ".json")
    if destination.exists() and metadata_path.exists() and not force:
        return destination

    torch.set_num_threads(config.threads)
    source_spec = _source_exp723_spec(spec)
    model, checkpoint_payload, reused = exp723._load_any_model(source_spec, data, _base_exp723_config(config))
    loaders = exp723._loaders(data, source_spec, config.batch_size, False)
    splits = {
        name: exp723._extract_l2(model, loader, torch.device(config.device))
        for name, loader in loaders.items()
    }

    payload: dict[str, np.ndarray] = {}
    split_meta: dict[str, Any] = {}
    for name, (l2, labels, lengths) in splits.items():
        arr = l2.numpy()
        _assert_binary_l2(arr, f"{spec.key}/{name}")
        compact = arr.astype(np.uint8, copy=False)
        payload[f"{name}_l2"] = compact
        payload[f"{name}_y"] = labels.astype(np.int64, copy=False)
        payload[f"{name}_lengths"] = lengths.astype(np.int64, copy=False)
        split_meta[name] = {
            "shape": list(compact.shape),
            "n_samples": int(len(labels)),
            "mean_firing_fraction_full": float(compact.mean()),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    _save_json(metadata_path, {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment_id": exp723.EXPERIMENT_ID,
        "source_protocol_version": exp723.PROTOCOL_VERSION,
        "source_spec": asdict(spec),
        "source_checkpoint_best_epoch": int(checkpoint_payload["best_epoch"]),
        "source_reuses_exp7_2_checkpoint": bool(reused),
        "storage_dtype": "uint8",
        "splits": split_meta,
    })
    return destination


def _load_cache(spec: SourceSpec, config: Config) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    path = _source_path(config.results_dir, "frozen_l2_cache", spec, ".npz")
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen L2 cache: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {
            name: (
                z[f"{name}_l2"].copy(),
                z[f"{name}_y"].copy(),
                z[f"{name}_lengths"].copy(),
            )
            for name in ("train", "val", "test")
        }


def _cached_loader(
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    l2, labels, lengths = split
    dataset = TensorDataset(
        torch.from_numpy(l2),
        torch.from_numpy(labels),
        torch.from_numpy(lengths),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def _cached_loaders(
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    spec: FrozenHeadSpec,
    batch_size: int,
    shuffle_train: bool,
) -> dict[str, DataLoader]:
    return {
        name: _cached_loader(
            split,
            batch_size,
            shuffle_train if name == "train" else False,
            frozen_pair_seed(spec, f"{name}_loader"),
        )
        for name, split in cache.items()
    }


class OutputSynapticHead(nn.Module):
    def __init__(self, n_classes: int, output_alpha: float) -> None:
        super().__init__()
        if output_alpha not in OUTPUT_ALPHAS:
            raise ValueError(output_alpha)
        self.n_classes = int(n_classes)
        self.output_alpha = float(output_alpha)
        self.output_linear = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=False)
        self.output_lif = exp401.MacroMultiSpikeLIF(
            beta=OUTPUT_BETA,
            threshold=exp72.THRESHOLD,
            max_spikes_per_dt=OUTPUT_CAP,
            surrogate_slope=exp72.SURROGATE_SLOPE,
        )

    def forward(self, l2: torch.Tensor) -> torch.Tensor:
        if l2.ndim != 3 or l2.shape[-1] != HIDDEN_WIDTH:
            raise ValueError(f"Expected [B,T,{HIDDEN_WIDTH}], got {tuple(l2.shape)}")
        batch, steps, _ = l2.shape
        syn = torch.zeros(batch, self.n_classes, device=l2.device, dtype=l2.dtype)
        mem = torch.zeros_like(syn)
        spikes: list[torch.Tensor] = []
        for t in range(steps):
            syn = self.output_alpha * syn + self.output_linear(l2[:, t])
            out, mem, _ = self.output_lif(syn, mem)
            spikes.append(out)
        return torch.stack(spikes, dim=1)


def _output_objective(spikes: torch.Tensor, lengths: torch.Tensor, y: torch.Tensor, objective: str, fs: float) -> torch.Tensor:
    return exp50.objective_loss(spikes, lengths, y, objective, OUTPUT_CAP, fs)


def _full_logits(spikes: torch.Tensor) -> torch.Tensor:
    return (spikes / float(OUTPUT_CAP)).sum(dim=1)


def _tail_totals(spikes: torch.Tensor, lengths: torch.Tensor) -> dict[str, float]:
    valid = exp50.valid_mask(lengths, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
    valid_spikes = float((spikes * valid).sum().detach().cpu())
    tail_spikes = float((spikes * (1.0 - valid)).sum().detach().cpu())
    valid_steps = int(lengths.sum().detach().cpu())
    tail_steps = int((spikes.shape[1] - lengths).sum().detach().cpu())
    return {
        "valid_spikes": valid_spikes,
        "tail_spikes": tail_spikes,
        "valid_steps": float(valid_steps),
        "tail_steps": float(tail_steps),
    }


def _finish_tail(totals: dict[str, float], width: int, fs: float) -> dict[str, float]:
    valid_spikes = totals["valid_spikes"]
    tail_spikes = totals["tail_spikes"]
    total = valid_spikes + tail_spikes
    valid_seconds = totals["valid_steps"] / float(fs)
    tail_seconds = totals["tail_steps"] / float(fs)
    return {
        **totals,
        "tail_fraction": tail_spikes / total if total > 0 else 0.0,
        "valid_spikes_per_neuron_second": valid_spikes / (width * valid_seconds) if valid_seconds > 0 else 0.0,
        "tail_spikes_per_neuron_second": tail_spikes / (width * tail_seconds) if tail_seconds > 0 else 0.0,
    }


def _evaluate_output_model(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    objective: str,
    fs: float,
    forward_fn,
) -> dict[str, Any]:
    labels: list[np.ndarray] = []
    valid_preds: list[np.ndarray] = []
    full_preds: list[np.ndarray] = []
    objective_sum = 0.0
    n_total = 0
    tail = {"valid_spikes": 0.0, "tail_spikes": 0.0, "valid_steps": 0.0, "tail_steps": 0.0}
    output_width: int | None = None
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            spikes = forward_fn(model, X)
            output_width = int(spikes.shape[-1])
            task = _output_objective(spikes, lengths, y, objective, fs)
            valid_logits = exp50.deployment_logits(spikes, lengths, OUTPUT_CAP)
            full_logits = _full_logits(spikes)
            labels.append(y.cpu().numpy())
            valid_preds.append(valid_logits.argmax(1).cpu().numpy())
            full_preds.append(full_logits.argmax(1).cpu().numpy())
            objective_sum += float(task) * len(y)
            n_total += len(y)
            batch_tail = _tail_totals(spikes, lengths)
            for key in tail:
                tail[key] += batch_tail[key]
    y_true = np.concatenate(labels)
    valid_metrics = exp72._metrics(y_true, np.concatenate(valid_preds))
    full_metrics = exp72._metrics(y_true, np.concatenate(full_preds))
    return {
        "valid": valid_metrics,
        "full": full_metrics,
        "objective_loss": objective_sum / max(n_total, 1),
        "tail": _finish_tail(tail, output_width or 1, fs),
    }


def _frozen_forward(model: OutputSynapticHead, l2: torch.Tensor) -> torch.Tensor:
    return model(l2)


def _early_stop(epoch: int, best_epoch: int) -> bool:
    return epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE


def _plot_history(rows: list[dict[str, float]], path: Path, title: str) -> None:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame.epoch, frame.train_ba, label="train valid WC BA")
    ax.plot(frame.epoch, frame.val_ba, label="val valid WC BA")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced accuracy")
    ax2 = ax.twinx()
    ax2.plot(frame.epoch, frame.train_objective_loss, linestyle="--", label="train objective")
    ax2.plot(frame.epoch, frame.val_objective_loss, linestyle=":", label="val objective")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_frozen_head(spec: FrozenHeadSpec, data: exp3.Data, config: Config, force: bool = False) -> dict[str, Any]:
    validate_frozen_spec(spec)
    evaluation_path = _frozen_path(config.results_dir, "evaluations", spec, ".json")
    checkpoint_path = _frozen_path(config.results_dir, "checkpoints", spec, ".pt")
    if evaluation_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(evaluation_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    cache = _load_cache(spec.source, config)
    device = torch.device(config.device)
    exp3.seed_all(frozen_pair_seed(spec, "model_init"))
    model = OutputSynapticHead(len(data.labels), spec.output_alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY)
    train_loader = _cached_loaders(cache, spec, config.batch_size, True)["train"]
    eval_loaders = _cached_loaders(cache, spec, config.batch_size, False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    rows: list[dict[str, float]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_objective_sum = 0.0
        n_total = 0
        for l2, y, lengths in train_loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            spikes = model(l2)
            loss = _output_objective(spikes, lengths, y, spec.objective, data.fs)
            loss.backward()
            optimizer.step()
            train_objective_sum += float(loss.detach()) * len(y)
            n_total += len(y)
        train_metrics = _evaluate_output_model(model, eval_loaders["train"], device, spec.objective, data.fs, _frozen_forward)
        val_metrics = _evaluate_output_model(model, eval_loaders["val"], device, spec.objective, data.fs, _frozen_forward)
        train_objective = train_objective_sum / max(n_total, 1)
        rows.append({
            "epoch": float(epoch),
            "train_ba": float(train_metrics["valid"]["balanced_accuracy"]),
            "val_ba": float(val_metrics["valid"]["balanced_accuracy"]),
            "train_objective_loss": float(train_objective),
            "val_objective_loss": float(val_metrics["objective_loss"]),
        })
        improved = val_metrics["valid"]["balanced_accuracy"] > best_ba + 1e-12 or (
            abs(val_metrics["valid"]["balanced_accuracy"] - best_ba) <= 1e-12
            and val_metrics["objective_loss"] < best_loss
        )
        if improved:
            best_ba = float(val_metrics["valid"]["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if _early_stop(epoch, best_epoch):
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No frozen-head checkpoint selected for {spec.key}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "block": "frozen_head",
        "spec": asdict(spec),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_ba": best_ba,
        "best_val_objective_loss": best_loss,
        "model_state_dict": best_state,
        "trainable_parameters": HIDDEN_WIDTH * len(data.labels),
    }, checkpoint_path)
    model.load_state_dict(best_state, strict=True)
    final_metrics = {
        name: _evaluate_output_model(model, loader, device, spec.objective, data.fs, _frozen_forward)
        for name, loader in eval_loaders.items()
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "block": "frozen_head",
        "spec": asdict(spec),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "metrics": final_metrics,
    }
    _save_json(evaluation_path, payload)
    history_path = _frozen_path(config.results_dir, "histories", spec, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(history_path, index=False)
    _plot_history(rows, _frozen_path(config.results_dir, "training_curves", spec, ".png"), spec.key)
    return payload


class PairedE2ESNN(nn.Module):
    def __init__(self, spec: E2ESpec, n_classes: int, fs: float) -> None:
        super().__init__()
        validate_e2e_spec(spec)
        self.spec = spec
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        shifts = ARCHITECTURES[spec.architecture]
        self.hidden_linears = nn.ModuleList([
            nn.Linear(exp72.EXPECTED_CHANNELS, HIDDEN_WIDTH, bias=False),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False),
        ])
        beta_hidden = math.exp(-(1000.0 / fs) / exp72.TAU_MEM_MS)
        self.hidden_lifs = nn.ModuleList([
            exp401.MacroMultiSpikeLIF(
                beta=beta_hidden,
                threshold=exp72.THRESHOLD,
                max_spikes_per_dt=1,
                surrogate_slope=exp72.SURROGATE_SLOPE,
            )
            for _ in range(2)
        ])
        self.register_buffer("alpha_0", exp50.alpha_vector(HIDDEN_WIDTH, shifts[0]))
        self.register_buffer("alpha_1", exp50.alpha_vector(HIDDEN_WIDTH, shifts[1]))
        self.output_linear = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=False)
        self.output_lif = exp401.MacroMultiSpikeLIF(
            beta=OUTPUT_BETA,
            threshold=exp72.THRESHOLD,
            max_spikes_per_dt=OUTPUT_CAP,
            surrogate_slope=exp72.SURROGATE_SLOPE,
        )
        self.output_alpha = float(spec.output_alpha)

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != exp72.EXPECTED_CHANNELS:
            raise ValueError(channels)
        syn = [torch.zeros(batch, HIDDEN_WIDTH, device=x.device, dtype=x.dtype) for _ in range(2)]
        mem = [torch.zeros_like(syn[0]), torch.zeros_like(syn[1])]
        hidden: list[list[torch.Tensor]] = [[], []]
        out_syn = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)
        out_mem = torch.zeros_like(out_syn)
        output_spikes: list[torch.Tensor] = []
        pre_output_evidence: list[torch.Tensor] = []
        output_synaptic_state: list[torch.Tensor] = []
        for t in range(steps):
            cur = x[:, t]
            for li in range(2):
                alpha = getattr(self, f"alpha_{li}")
                syn[li] = alpha * syn[li] + self.hidden_linears[li](cur)
                spk, mem[li], _ = self.hidden_lifs[li](syn[li], mem[li])
                hidden[li].append(spk)
                cur = spk
            evidence = self.output_linear(cur)
            out_syn = self.output_alpha * out_syn + evidence
            out_spk, out_mem, _ = self.output_lif(out_syn, out_mem)
            pre_output_evidence.append(evidence)
            output_synaptic_state.append(out_syn)
            output_spikes.append(out_spk)
        return {
            "hidden_spikes": tuple(torch.stack(v, dim=1) for v in hidden),
            "pre_output_evidence": torch.stack(pre_output_evidence, dim=1),
            "output_synaptic_state": torch.stack(output_synaptic_state, dim=1),
            "output_spikes": torch.stack(output_spikes, dim=1),
        }


def _e2e_loaders(data: exp3.Data, spec: E2ESpec, batch_size: int, shuffle_train: bool) -> dict[str, Any]:
    parts = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    return {
        name: exp3.loader(
            X,
            y,
            lengths,
            batch_size,
            shuffle_train if name == "train" else False,
            e2e_pair_seed(spec, f"{name}_loader"),
        )
        for name, (X, y, lengths) in parts.items()
    }


def _e2e_forward(model: PairedE2ESNN, X: torch.Tensor) -> torch.Tensor:
    return model.forward_trajectory(X)["output_spikes"]


def _e2e_regularizer(
    spec: E2ESpec,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
    calibration: dict[str, Any],
    epoch: int,
) -> torch.Tensor:
    if spec.regularization == exp72.TASK_ONLY:
        return torch.zeros((), device=lengths.device)
    rate, _, _, persist = exp72.regularization_terms(trajectory["hidden_spikes"], lengths)
    scale = exp72.warmup_scale(epoch)
    return scale * (
        float(calibration["lambda_rate"]) * rate
        + float(calibration["lambda_persist"]) * persist
    )


def _calibrate_e2e(spec: E2ESpec, model: PairedE2ESNN, data: exp3.Data, config: Config) -> dict[str, Any]:
    if spec.regularization == exp72.TASK_ONLY:
        return {"calibrated": False, "lambda_rate": 0.0, "lambda_persist": 0.0}
    loader = _e2e_loaders(data, spec, config.batch_size, True)["train"]
    params = tuple(layer.weight for layer in model.hidden_linears)
    rate_ratios: list[float] = []
    persist_ratios: list[float] = []
    device = torch.device(config.device)
    for index, (X, y, lengths) in enumerate(loader):
        if index >= exp72.CALIBRATION_BATCHES:
            break
        X, y, lengths = X.to(device), y.to(device), lengths.to(device)
        tr = model.forward_trajectory(X)
        task = _output_objective(tr["output_spikes"], lengths, y, spec.objective, data.fs)
        rate, _, _, persist = exp72.regularization_terms(tr["hidden_spikes"], lengths)
        task_grad = exp72._grad_norm(task, params, True)
        rate_grad = exp72._grad_norm(rate, params, True)
        persist_grad = exp72._grad_norm(persist, params, False)
        if task_grad > 1e-12:
            rate_ratios.append(rate_grad / task_grad)
            persist_ratios.append(persist_grad / task_grad)
    if not rate_ratios:
        raise RuntimeError(f"No calibration batches for {spec.key}")
    rr = float(np.median(rate_ratios))
    pr = float(np.median(persist_ratios))
    return {
        "calibrated": True,
        "lambda_rate": exp72.TARGET_RATE_GRAD_RATIO / rr if rr > 1e-12 else 1.0,
        "lambda_persist": exp72.TARGET_PERSIST_GRAD_RATIO / pr if pr > 1e-12 else 1.0,
        "raw_rate_to_task": rr,
        "raw_persist_to_task": pr,
        "parameter_scope": "both hidden input matrices only",
        "output_regularized": False,
    }


def _extract_e2e_l2(model: PairedE2ESNN, loader, device: torch.device):
    xs: list[torch.Tensor] = []
    ys: list[np.ndarray] = []
    lengths_out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            tr = model.forward_trajectory(X.to(device))
            xs.append(tr["hidden_spikes"][-1].cpu())
            ys.append(y.numpy())
            lengths_out.append(lengths.numpy())
    return torch.cat(xs, dim=0), np.concatenate(ys), np.concatenate(lengths_out)


def _fit_e2e_probe(
    source: str,
    splits: dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]],
    spec: E2ESpec,
    bin_steps: int,
) -> dict[str, Any]:
    features = {
        name: (exp723._probe_features(l2, lengths, bin_steps, source), y)
        for name, (l2, y, lengths) in splits.items()
    }
    probe = exp01._fit_linear_probe(
        features["train"][0], features["train"][1],
        features["val"][0], features["val"][1],
        features["test"][0], features["test"][1],
        e2e_pair_seed(spec, f"probe_{source}"),
    )
    return {
        "source": source,
        "feature_dim": int(probe["feature_dim"]),
        "probe_C": float(probe["probe_C"]),
        "metrics": {split: probe[split] for split in ("train", "val", "test")},
    }


def run_e2e(spec: E2ESpec, data: exp3.Data, config: Config, force: bool = False) -> dict[str, Any]:
    validate_e2e_spec(spec)
    evaluation_path = _e2e_path(config.results_dir, "evaluations", spec, ".json")
    checkpoint_path = _e2e_path(config.results_dir, "checkpoints", spec, ".pt")
    if evaluation_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(evaluation_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp3.seed_all(e2e_pair_seed(spec, "model_init"))
    model = PairedE2ESNN(spec, len(data.labels), data.fs).to(device)
    calibration = _calibrate_e2e(spec, model, data, config)
    _save_json(_e2e_path(config.results_dir, "calibrations", spec, ".json"), calibration)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY)
    train_loader = _e2e_loaders(data, spec, config.batch_size, True)["train"]
    eval_loaders = _e2e_loaders(data, spec, config.batch_size, False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    rows: list[dict[str, float]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        task_sum = 0.0
        reg_sum = 0.0
        total_sum = 0.0
        n_total = 0
        for X, y, lengths in train_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            tr = model.forward_trajectory(X)
            task = _output_objective(tr["output_spikes"], lengths, y, spec.objective, data.fs)
            reg = _e2e_regularizer(spec, tr, lengths, calibration, epoch)
            total = task + reg
            total.backward()
            optimizer.step()
            task_sum += float(task.detach()) * len(y)
            reg_sum += float(reg.detach()) * len(y)
            total_sum += float(total.detach()) * len(y)
            n_total += len(y)
        train_metrics = _evaluate_output_model(model, eval_loaders["train"], device, spec.objective, data.fs, _e2e_forward)
        val_metrics = _evaluate_output_model(model, eval_loaders["val"], device, spec.objective, data.fs, _e2e_forward)
        rows.append({
            "epoch": float(epoch),
            "train_ba": float(train_metrics["valid"]["balanced_accuracy"]),
            "val_ba": float(val_metrics["valid"]["balanced_accuracy"]),
            "train_objective_loss": task_sum / max(n_total, 1),
            "train_reg_loss": reg_sum / max(n_total, 1),
            "train_total_loss": total_sum / max(n_total, 1),
            "val_objective_loss": float(val_metrics["objective_loss"]),
        })
        improved = val_metrics["valid"]["balanced_accuracy"] > best_ba + 1e-12 or (
            abs(val_metrics["valid"]["balanced_accuracy"] - best_ba) <= 1e-12
            and val_metrics["objective_loss"] < best_loss
        )
        if improved:
            best_ba = float(val_metrics["valid"]["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if _early_stop(epoch, best_epoch):
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No E2E checkpoint selected for {spec.key}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "block": "e2e",
        "spec": asdict(spec),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_ba": best_ba,
        "best_val_objective_loss": best_loss,
        "model_state_dict": best_state,
        "calibration": calibration,
    }, checkpoint_path)
    model.load_state_dict(best_state, strict=True)
    final_metrics = {
        name: _evaluate_output_model(model, loader, device, spec.objective, data.fs, _e2e_forward)
        for name, loader in eval_loaders.items()
    }
    l2_splits = {name: _extract_e2e_l2(model, loader, device) for name, loader in eval_loaders.items()}
    probes = {source: _fit_e2e_probe(source, l2_splits, spec, data.bin_steps) for source in PROBE_SOURCES}
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "block": "e2e",
        "spec": asdict(spec),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "calibration": calibration,
        "metrics": final_metrics,
        "probes": probes,
    }
    _save_json(evaluation_path, payload)
    history_path = _e2e_path(config.results_dir, "histories", spec, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(history_path, index=False)
    _plot_history(rows, _e2e_path(config.results_dir, "training_curves", spec, ".png"), spec.key)
    return payload


def _aggregate(df: pd.DataFrame, groups: list[str], values: list[str]) -> pd.DataFrame:
    grouped = df.groupby(groups, dropna=False)[values]
    return pd.concat([
        grouped.size().rename("n"),
        grouped.mean().add_suffix("_mean"),
        grouped.std(ddof=1).fillna(0).add_suffix("_std"),
    ], axis=1).reset_index()


def _performance_rows(payload: dict[str, Any], block: str) -> list[dict[str, Any]]:
    spec = payload["spec"]
    if block == "frozen_head":
        base = {
            "block": block,
            "architecture": spec["architecture"],
            "source_family": spec["source_family"],
            "regularization": spec["regularization"],
            "seed": int(spec["seed"]),
            "objective": spec["objective"],
            "output_alpha": float(spec["output_alpha"]),
            "output_beta": OUTPUT_BETA,
        }
    else:
        base = {
            "block": block,
            "architecture": spec["architecture"],
            "regularization": spec["regularization"],
            "seed": int(spec["seed"]),
            "objective": spec["objective"],
            "output_alpha": float(spec["output_alpha"]),
            "output_beta": OUTPUT_BETA,
        }
    rows: list[dict[str, Any]] = []
    for split, result in payload["metrics"].items():
        for scope in ("valid", "full"):
            metrics = result[scope]
            rows.append({
                **base,
                "split": split,
                "scope": scope,
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "objective_loss": float(result["objective_loss"]),
            })
    return rows


def _tail_rows(payload: dict[str, Any], block: str) -> list[dict[str, Any]]:
    spec = payload["spec"]
    base = {
        "block": block,
        "architecture": spec["architecture"],
        "regularization": spec["regularization"],
        "seed": int(spec["seed"]),
        "objective": spec["objective"],
        "output_alpha": float(spec["output_alpha"]),
        "output_beta": OUTPUT_BETA,
    }
    if block == "frozen_head":
        base["source_family"] = spec["source_family"]
    return [{**base, "split": split, **result["tail"]} for split, result in payload["metrics"].items()]


def _probe_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    base = {
        "architecture": spec["architecture"],
        "regularization": spec["regularization"],
        "seed": int(spec["seed"]),
        "objective": spec["objective"],
        "output_alpha": float(spec["output_alpha"]),
        "output_beta": OUTPUT_BETA,
    }
    rows: list[dict[str, Any]] = []
    for source, probe in payload["probes"].items():
        for split, metrics in probe["metrics"].items():
            rows.append({
                **base,
                "source": source,
                "split": split,
                "feature_dim": int(probe["feature_dim"]),
                "probe_C": float(probe["probe_C"]),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
            })
    return rows


def _load_exp723_runs(repo_root: Path) -> pd.DataFrame:
    path = exp723.results_dir(repo_root) / "performance_runs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Exp7.2.5 requires finalized Exp7.2.3 runs: {path}")
    frame = pd.read_csv(path)
    expected = {
        "architecture", "family", "regularization", "seed", "source", "split",
        "accuracy", "balanced_accuracy", "macro_f1",
    }
    missing = expected.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Exp7.2.3 performance_runs.csv missing {sorted(missing)}")
    return frame


def _paired_alpha_deltas(performance: pd.DataFrame, extra_keys: list[str], prefix: str) -> pd.DataFrame:
    test = performance[(performance["split"] == "test") & (performance["scope"] == "valid")]
    keys = ["architecture", "regularization", "seed", "objective", *extra_keys]
    rows: list[dict[str, Any]] = []
    for key, group in test.groupby(keys, dropna=False):
        indexed = group.set_index("output_alpha")
        if 0.0 not in indexed.index or 0.5 not in indexed.index:
            raise RuntimeError(f"Missing paired alpha condition for {key}")
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(keys, key_values, strict=True))
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            rows.append({
                "contrast": f"{prefix}_alpha05_minus_alpha00",
                **base,
                "metric": metric,
                "delta": float(indexed.loc[0.5, metric] - indexed.loc[0.0, metric]),
            })
    return pd.DataFrame(rows)


def _e2e_probe_deltas(probes: pd.DataFrame) -> pd.DataFrame:
    test = probes[probes["split"] == "test"]
    keys = ["architecture", "regularization", "seed", "objective", "source"]
    rows: list[dict[str, Any]] = []
    for key, group in test.groupby(keys, dropna=False):
        indexed = group.set_index("output_alpha")
        if 0.0 not in indexed.index or 0.5 not in indexed.index:
            raise RuntimeError(f"Missing paired E2E probe alpha condition for {key}")
        base = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            rows.append({
                "contrast": "e2e_probe_alpha05_minus_alpha00",
                **base,
                "metric": metric,
                "delta": float(indexed.loc[0.5, metric] - indexed.loc[0.0, metric]),
            })
    return pd.DataFrame(rows)


def _historical_mean(old: pd.DataFrame, regularization: str, family: str, source: str) -> float:
    selected = old[
        (old["regularization"] == regularization)
        & (old["family"] == family)
        & (old["source"] == source)
        & (old["split"] == "test")
    ]
    expected_n = len(ARCHITECTURE_ORDER) * len(SEEDS)
    if len(selected) != expected_n:
        raise RuntimeError(f"Expected {expected_n} historical rows for {regularization}/{family}/{source}, got {len(selected)}")
    return float(selected["balanced_accuracy"].mean())


def _select_mean(frame: pd.DataFrame, filters: dict[str, Any], column: str = "balanced_accuracy") -> float:
    selected = frame
    for key, value in filters.items():
        selected = selected[selected[key] == value]
    if selected.empty:
        raise RuntimeError(f"No rows for filters={filters}")
    return float(selected[column].mean())


def _frozen_report(old: pd.DataFrame, frozen_perf: pd.DataFrame, regularization: str) -> pd.DataFrame:
    test = frozen_perf[
        (frozen_perf["regularization"] == regularization)
        & (frozen_perf["split"] == "test")
    ]
    rows: list[dict[str, Any]] = []
    for family in SOURCE_FAMILIES:
        wc0 = _select_mean(test, {"source_family": family, "objective": "whole_count_ce", "output_alpha": 0.0, "scope": "valid"})
        wc5 = _select_mean(test, {"source_family": family, "objective": "whole_count_ce", "output_alpha": 0.5, "scope": "valid"})
        wc0_full = _select_mean(test, {"source_family": family, "objective": "whole_count_ce", "output_alpha": 0.0, "scope": "full"})
        wc5_full = _select_mean(test, {"source_family": family, "objective": "whole_count_ce", "output_alpha": 0.5, "scope": "full"})
        ts0 = _select_mean(test, {"source_family": family, "objective": "timestep_ce", "output_alpha": 0.0, "scope": "valid"})
        ts5 = _select_mean(test, {"source_family": family, "objective": "timestep_ce", "output_alpha": 0.5, "scope": "valid"})
        ts0_full = _select_mean(test, {"source_family": family, "objective": "timestep_ce", "output_alpha": 0.0, "scope": "full"})
        ts5_full = _select_mean(test, {"source_family": family, "objective": "timestep_ce", "output_alpha": 0.5, "scope": "full"})
        rows.append({
            "source_family": family,
            "source_family_label": FAMILY_LABELS[family],
            "regularization": regularization,
            "native_ba": _historical_mean(old, regularization, family, "native"),
            "l2_wc_linear_ba": _historical_mean(old, regularization, family, VALID_WC_PROBE),
            "wc_refit_alpha00_ba": wc0,
            "wc_refit_alpha05_ba": wc5,
            "wc_alpha_gain_pp": 100.0 * (wc5 - wc0),
            "wc_alpha00_full_minus_valid_pp": 100.0 * (wc0_full - wc0),
            "wc_alpha05_full_minus_valid_pp": 100.0 * (wc5_full - wc5),
            "tsce_refit_alpha00_ba": ts0,
            "tsce_refit_alpha05_ba": ts5,
            "tsce_alpha_gain_pp": 100.0 * (ts5 - ts0),
            "tsce_alpha00_full_minus_valid_pp": 100.0 * (ts0_full - ts0),
            "tsce_alpha05_full_minus_valid_pp": 100.0 * (ts5_full - ts5),
        })
    return pd.DataFrame(rows)


def _e2e_report(e2e_perf: pd.DataFrame, probes: pd.DataFrame, regularization: str, objective: str) -> pd.DataFrame:
    perf = e2e_perf[
        (e2e_perf["regularization"] == regularization)
        & (e2e_perf["objective"] == objective)
        & (e2e_perf["split"] == "test")
    ]
    probe = probes[
        (probes["regularization"] == regularization)
        & (probes["objective"] == objective)
        & (probes["split"] == "test")
    ]
    rows: list[dict[str, Any]] = []
    for alpha in OUTPUT_ALPHAS:
        native_valid = _select_mean(perf, {"output_alpha": alpha, "scope": "valid"})
        native_full = _select_mean(perf, {"output_alpha": alpha, "scope": "full"})
        wc = _select_mean(probe, {"output_alpha": alpha, "source": VALID_WC_PROBE})
        f250 = _select_mean(probe, {"output_alpha": alpha, "source": VALID_F250_PROBE})
        rows.append({
            "regularization": regularization,
            "objective": objective,
            "output_alpha": alpha,
            "output_beta": OUTPUT_BETA,
            "native_valid_ba": native_valid,
            "l2_wc_probe_ba": wc,
            "l2_f250_probe_ba": f250,
            "f250_minus_wc_pp": 100.0 * (f250 - wc),
            "native_full_ba": native_full,
            "full_minus_valid_pp": 100.0 * (native_full - native_valid),
        })
    return pd.DataFrame(rows)


def _mechanism_rows(frozen_deltas: pd.DataFrame, e2e_deltas: pd.DataFrame) -> pd.DataFrame:
    frozen = frozen_deltas[
        (frozen_deltas["metric"] == "balanced_accuracy")
    ].copy()
    e2e = e2e_deltas[e2e_deltas["metric"] == "balanced_accuracy"].copy()
    rows: list[dict[str, Any]] = []
    for objective, source_family in MATCHED_SOURCE_BY_OBJECTIVE.items():
        f = frozen[(frozen["objective"] == objective) & (frozen["source_family"] == source_family)]
        e = e2e[e2e["objective"] == objective]
        merged = e.merge(
            f[["architecture", "regularization", "seed", "delta"]].rename(columns={"delta": "frozen_alpha_gain"}),
            on=["architecture", "regularization", "seed"],
            how="inner",
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            rows.append({
                "architecture": row.architecture,
                "regularization": row.regularization,
                "seed": int(row.seed),
                "objective": objective,
                "matched_frozen_source": source_family,
                "e2e_alpha_gain": float(row.delta),
                "frozen_alpha_gain": float(row.frozen_alpha_gain),
                "shaping_diagnostic": float(row.delta - row.frozen_alpha_gain),
            })
    return pd.DataFrame(rows)


def finalize(repo_root: Path) -> dict[str, Any]:
    root = results_dir(repo_root)
    frozen_perf_rows: list[dict[str, Any]] = []
    frozen_tail_rows: list[dict[str, Any]] = []
    e2e_perf_rows: list[dict[str, Any]] = []
    e2e_tail_rows: list[dict[str, Any]] = []
    e2e_probe_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for spec in frozen_head_specs():
        path = _frozen_path(root, "evaluations", spec, ".json")
        if not path.exists():
            missing.append(f"frozen:{spec.key}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        frozen_perf_rows.extend(_performance_rows(payload, "frozen_head"))
        frozen_tail_rows.extend(_tail_rows(payload, "frozen_head"))

    for spec in e2e_specs():
        path = _e2e_path(root, "evaluations", spec, ".json")
        if not path.exists():
            missing.append(f"e2e:{spec.key}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        e2e_perf_rows.extend(_performance_rows(payload, "e2e"))
        e2e_tail_rows.extend(_tail_rows(payload, "e2e"))
        e2e_probe_rows.extend(_probe_rows(payload))

    if missing:
        raise RuntimeError(f"Missing {len(missing)} Exp7.2.5 evaluations; first={missing[:8]}")

    frozen_perf = pd.DataFrame(frozen_perf_rows)
    frozen_perf.to_csv(root / "frozen_head_performance_runs.csv", index=False)
    _aggregate(
        frozen_perf,
        ["architecture", "source_family", "regularization", "objective", "output_alpha", "output_beta", "split", "scope"],
        ["accuracy", "balanced_accuracy", "macro_f1", "objective_loss"],
    ).to_csv(root / "frozen_head_performance_summary.csv", index=False)

    frozen_tail = pd.DataFrame(frozen_tail_rows)
    frozen_tail.to_csv(root / "frozen_head_tail_runs.csv", index=False)
    _aggregate(
        frozen_tail,
        ["architecture", "source_family", "regularization", "objective", "output_alpha", "output_beta", "split"],
        ["tail_fraction", "valid_spikes_per_neuron_second", "tail_spikes_per_neuron_second"],
    ).to_csv(root / "frozen_head_tail_summary.csv", index=False)

    e2e_perf = pd.DataFrame(e2e_perf_rows)
    e2e_perf.to_csv(root / "e2e_performance_runs.csv", index=False)
    _aggregate(
        e2e_perf,
        ["architecture", "regularization", "objective", "output_alpha", "output_beta", "split", "scope"],
        ["accuracy", "balanced_accuracy", "macro_f1", "objective_loss"],
    ).to_csv(root / "e2e_performance_summary.csv", index=False)

    e2e_tail = pd.DataFrame(e2e_tail_rows)
    e2e_tail.to_csv(root / "e2e_tail_runs.csv", index=False)
    _aggregate(
        e2e_tail,
        ["architecture", "regularization", "objective", "output_alpha", "output_beta", "split"],
        ["tail_fraction", "valid_spikes_per_neuron_second", "tail_spikes_per_neuron_second"],
    ).to_csv(root / "e2e_tail_summary.csv", index=False)

    probes = pd.DataFrame(e2e_probe_rows)
    probes.to_csv(root / "e2e_l2_probe_runs.csv", index=False)
    _aggregate(
        probes,
        ["architecture", "regularization", "objective", "output_alpha", "output_beta", "source", "split"],
        ["accuracy", "balanced_accuracy", "macro_f1"],
    ).to_csv(root / "e2e_l2_probe_summary.csv", index=False)

    frozen_delta = _paired_alpha_deltas(frozen_perf, ["source_family"], "frozen")
    frozen_delta.to_csv(root / "frozen_alpha_delta_runs.csv", index=False)
    _aggregate(
        frozen_delta,
        ["contrast", "architecture", "source_family", "regularization", "objective", "metric"],
        ["delta"],
    ).to_csv(root / "frozen_alpha_delta_summary.csv", index=False)

    e2e_delta = _paired_alpha_deltas(e2e_perf, [], "e2e")
    e2e_delta.to_csv(root / "e2e_alpha_delta_runs.csv", index=False)
    _aggregate(
        e2e_delta,
        ["contrast", "architecture", "regularization", "objective", "metric"],
        ["delta"],
    ).to_csv(root / "e2e_alpha_delta_summary.csv", index=False)

    probe_delta = _e2e_probe_deltas(probes)
    probe_delta.to_csv(root / "e2e_probe_alpha_delta_runs.csv", index=False)
    _aggregate(
        probe_delta,
        ["contrast", "architecture", "regularization", "objective", "source", "metric"],
        ["delta"],
    ).to_csv(root / "e2e_probe_alpha_delta_summary.csv", index=False)

    mechanism = _mechanism_rows(frozen_delta, e2e_delta)
    mechanism.to_csv(root / "mechanism_alpha_gain_runs.csv", index=False)
    _aggregate(
        mechanism,
        ["architecture", "regularization", "objective", "matched_frozen_source"],
        ["e2e_alpha_gain", "frozen_alpha_gain", "shaping_diagnostic"],
    ).to_csv(root / "mechanism_alpha_gain_summary.csv", index=False)

    old = _load_exp723_runs(repo_root)
    report_files: list[str] = []
    for regularization in REGULARIZATIONS:
        frozen_report = _frozen_report(old, frozen_perf, regularization)
        frozen_name = f"report_frozen_{regularization}.csv"
        frozen_report.to_csv(root / frozen_name, index=False)
        report_files.append(frozen_name)
        for objective in OBJECTIVES:
            e2e_report = _e2e_report(e2e_perf, probes, regularization, objective)
            short = "wc" if objective == "whole_count_ce" else "tsce"
            e2e_name = f"report_e2e_{regularization}_{short}.csv"
            e2e_report.to_csv(root / e2e_name, index=False)
            report_files.append(e2e_name)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment_id": exp723.EXPERIMENT_ID,
        "source_families": list(SOURCE_FAMILIES),
        "architectures": list(ARCHITECTURE_ORDER),
        "regularizations": list(REGULARIZATIONS),
        "seeds": list(SEEDS),
        "objectives": list(OBJECTIVES),
        "output_alphas": list(OUTPUT_ALPHAS),
        "output_beta": OUTPUT_BETA,
        "source_cache_tasks": len(source_specs()),
        "frozen_head_training_runs": len(frozen_head_specs()),
        "e2e_training_runs": len(e2e_specs()),
        "primary_readout": "valid-length output whole-count balanced accuracy",
        "checkpoint_selection": "validation valid-length whole-count BA; objective loss tie-break",
        "pairing_contract": "alpha=0 and alpha=0.5 share model initialization and loader-order seeds",
        "notebook_inputs": [
            "frozen_head_performance_summary.csv",
            "e2e_performance_summary.csv",
            "e2e_l2_probe_summary.csv",
            "frozen_alpha_delta_summary.csv",
            "e2e_alpha_delta_summary.csv",
            "e2e_probe_alpha_delta_summary.csv",
            "frozen_head_tail_summary.csv",
            "e2e_tail_summary.csv",
            "mechanism_alpha_gain_summary.csv",
            *report_files,
        ],
    }
    _save_json(root / "manifest.json", manifest)
    return manifest


def _select_source(index: int | None, args: argparse.Namespace) -> SourceSpec:
    specs = source_specs()
    if index is not None:
        if not 0 <= index < len(specs):
            raise ValueError(index)
        return specs[index]
    if None in (args.architecture, args.source_family, args.regularization, args.seed):
        raise ValueError("Specify --array-task-id or all source fields")
    return SourceSpec(args.architecture, args.source_family, args.regularization, args.seed)


def _select_frozen(index: int | None, args: argparse.Namespace) -> FrozenHeadSpec:
    specs = frozen_head_specs()
    if index is not None:
        if not 0 <= index < len(specs):
            raise ValueError(index)
        return specs[index]
    if None in (args.architecture, args.source_family, args.regularization, args.seed, args.objective, args.output_alpha):
        raise ValueError("Specify --array-task-id or all frozen-head fields")
    return FrozenHeadSpec(
        args.architecture,
        args.source_family,
        args.regularization,
        args.seed,
        args.objective,
        args.output_alpha,
    )


def _select_e2e(index: int | None, args: argparse.Namespace) -> E2ESpec:
    specs = e2e_specs()
    if index is not None:
        if not 0 <= index < len(specs):
            raise ValueError(index)
        return specs[index]
    if None in (args.architecture, args.regularization, args.seed, args.objective, args.output_alpha):
        raise ValueError("Specify --array-task-id or all E2E fields")
    return E2ESpec(args.architecture, args.regularization, args.seed, args.objective, args.output_alpha)


def _add_common_spec_args(parser: argparse.ArgumentParser, include_source: bool) -> None:
    parser.add_argument("--array-task-id", type=int)
    parser.add_argument("--architecture", choices=ARCHITECTURE_ORDER)
    if include_source:
        parser.add_argument("--source-family", choices=SOURCE_FAMILIES)
    parser.add_argument("--regularization", choices=REGULARIZATIONS)
    parser.add_argument("--seed", type=int, choices=SEEDS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-frozen")
    _add_common_spec_args(prep, include_source=True)
    prep.add_argument("--force", action="store_true")

    frozen = sub.add_parser("run-frozen")
    _add_common_spec_args(frozen, include_source=True)
    frozen.add_argument("--objective", choices=OBJECTIVES)
    frozen.add_argument("--output-alpha", type=float, choices=OUTPUT_ALPHAS)
    frozen.add_argument("--force", action="store_true")

    e2e = sub.add_parser("run-e2e")
    _add_common_spec_args(e2e, include_source=False)
    e2e.add_argument("--objective", choices=OBJECTIVES)
    e2e.add_argument("--output-alpha", type=float, choices=OUTPUT_ALPHAS)
    e2e.add_argument("--force", action="store_true")

    sub.add_parser("list-source")
    sub.add_parser("list-frozen")
    sub.add_parser("list-e2e")
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = find_repo_root(args.repo_root)
    config = Config(repo_root, results_dir(repo_root), args.device, args.batch_size, args.threads, args.max_epochs)

    if args.command == "list-source":
        for index, spec in enumerate(source_specs()):
            print(index, spec.key)
        return
    if args.command == "list-frozen":
        for index, spec in enumerate(frozen_head_specs()):
            print(index, spec.key)
        return
    if args.command == "list-e2e":
        for index, spec in enumerate(e2e_specs()):
            print(index, spec.key)
        return
    if args.command == "finalize":
        print(json.dumps(finalize(repo_root), indent=2))
        return

    data = exp72.prepare_data(repo_root)
    if args.command == "prepare-frozen":
        spec = _select_source(args.array_task_id, args)
        path = prepare_frozen_cache(spec, data, config, args.force)
        print(json.dumps({"source_spec": asdict(spec), "cache": str(path)}, indent=2))
        return
    if args.command == "run-frozen":
        spec = _select_frozen(args.array_task_id, args)
        payload = run_frozen_head(spec, data, config, args.force)
        print(json.dumps({"spec": payload["spec"], "best_epoch": payload["best_epoch"]}, indent=2))
        return
    if args.command == "run-e2e":
        spec = _select_e2e(args.array_task_id, args)
        payload = run_e2e(spec, data, config, args.force)
        print(json.dumps({"spec": payload["spec"], "best_epoch": payload["best_epoch"]}, indent=2))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
