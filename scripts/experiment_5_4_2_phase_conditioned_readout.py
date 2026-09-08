from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_5_4_1_constrained_conjunction_residual as exp541

exp54 = exp541.exp54
exp5323 = exp54.exp5323
parent = exp54.parent
base = exp541.base

EXPERIMENT_ID = "experiment_5_4_2_phase_conditioned_readout"
PROTOCOL_VERSION = "phase_conditioned_readout_v1"
SEEDS = exp541.SEEDS
WHAT_WIDTH = exp541.WHAT_WIDTH
WHEN_WIDTH = exp541.WHEN_WIDTH
N_CLASSES = 12
SCREEN_BANKS = 4
SCREEN_RANK = 4
BANK_SWEEP = (2, 4, 8)
RANK_SWEEP = (2, 4, 8)
MEM_STD_FLOOR = 1e-6
SHUFFLE_REPLICATES = 5
RELATIVE_PHASE_BINS = 10
PREFIX_FRACTIONS = tuple(i / 10.0 for i in range(1, 11))
EPOCHS = exp541.EPOCHS
BATCH_SIZE = exp541.BATCH_SIZE
LR = exp541.LR
WEIGHT_DECAY = exp541.WEIGHT_DECAY

WHAT_ONLY = "what_only"
ADDITIVE_SPIKE = "additive_spike"
ADDITIVE_MEM = "additive_mem"
BILINEAR_SPIKE = "bilinear_spike"
BILINEAR_MEM = "bilinear_mem"
BANK_SPIKE = "bank_spike"
BANK_MEM = "bank_mem"
BANK_MEAN_WHEN = "bank_mean_when"
BANK_ELAPSED = "bank_elapsed"
BANK_RELPHASE_ORACLE = "bank_relphase_oracle"
SCREEN_CONDITIONS = (
    WHAT_ONLY,
    ADDITIVE_SPIKE,
    ADDITIVE_MEM,
    BILINEAR_SPIKE,
    BILINEAR_MEM,
    BANK_SPIKE,
    BANK_MEM,
    BANK_MEAN_WHEN,
    BANK_ELAPSED,
    BANK_RELPHASE_ORACLE,
)
CANDIDATE_CONDITIONS = (
    BILINEAR_SPIKE,
    BILINEAR_MEM,
    BANK_SPIKE,
    BANK_MEM,
)
FINAL_ABLATIONS = (
    "ordered",
    "mean_when",
    "when_zero",
    "when_shuffle",
    "when_circular_shift",
    "reset_when",
    "residual_what_zero",
)
SOURCE_WHEN_CONDITION = exp5323.FF128_CONDITION


@dataclass(frozen=True)
class RunSpec:
    stage: str
    condition: str
    seed: int
    n_banks: int = SCREEN_BANKS
    rank: int = SCREEN_RANK

    @property
    def mechanism(self) -> str:
        if self.condition == WHAT_ONLY:
            return "base"
        if self.condition.startswith("additive_"):
            return "additive"
        if self.condition.startswith("bilinear_"):
            return "bilinear"
        if self.condition.startswith("bank_"):
            return "bank"
        raise ValueError(self.condition)

    @property
    def context_source(self) -> str:
        return {
            WHAT_ONLY: "none",
            ADDITIVE_SPIKE: "spike",
            ADDITIVE_MEM: "membrane",
            BILINEAR_SPIKE: "spike",
            BILINEAR_MEM: "membrane",
            BANK_SPIKE: "spike",
            BANK_MEM: "membrane",
            BANK_MEAN_WHEN: "membrane_mean",
            BANK_ELAPSED: "elapsed",
            BANK_RELPHASE_ORACLE: "relative_phase_oracle",
        }[self.condition]

    @property
    def key(self) -> str:
        cap = (
            f"k{self.n_banks}_r{self.rank}"
            if self.mechanism == "bank"
            else f"r{self.rank}"
            if self.mechanism == "bilinear"
            else "fixed"
        )
        return f"{self.condition}__seed{self.seed}__{cap}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


@dataclass
class ReadoutTrajectory:
    base_logits: torch.Tensor
    residual_logits: torch.Tensor
    base_evidence: torch.Tensor
    residual_evidence: torch.Tensor
    gate_probabilities: torch.Tensor | None
    centered_gates: torch.Tensor | None


def find_repo_root(start: Path | None = None) -> Path:
    return exp541.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_cache_path(root: Path, seed: int) -> Path:
    return root / "source_cache" / f"seed{seed}.npz"


def source_meta_path(root: Path, seed: int) -> Path:
    return root / "source_cache" / f"seed{seed}.json"


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / f"{spec.stage}_checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / f"{spec.stage}_evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / f"{spec.stage}_histories" / f"{spec.key}.csv"


def final_evaluation_path(root: Path, seed: int) -> Path:
    return root / "final_evaluations" / f"seed{seed}.json"


def screen_selection_path(root: Path) -> Path:
    return root / "screen_selection.json"


def selection_path(root: Path) -> Path:
    return root / "selection.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def screen_specs() -> list[RunSpec]:
    return [RunSpec("screen", condition, seed) for condition in SCREEN_CONDITIONS for seed in SEEDS]


def _validate_spec(spec: RunSpec) -> None:
    if spec.seed not in SEEDS or spec.condition not in SCREEN_CONDITIONS:
        raise ValueError(f"Invalid run spec: {spec}")
    if spec.mechanism == "bank" and spec.n_banks <= 1:
        raise ValueError("bank requires K > 1")
    if spec.mechanism in ("bank", "bilinear") and spec.rank <= 0:
        raise ValueError("rank must be positive")


def _exp541_config(config: Config) -> exp541.Config:
    return exp541.Config(
        repo_root=config.repo_root,
        results_dir=exp541.results_dir(config.repo_root),
        device=config.device,
        epochs=exp541.EPOCHS,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _exp54_config(config: Config) -> exp54.Config:
    return exp54.Config(
        repo_root=config.repo_root,
        results_dir=exp54.results_dir(config.repo_root),
        device=config.device,
        epochs=exp54.EPOCHS,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _exp5323_config(config: Config) -> exp5323.Config:
    return exp5323.Config(
        repo_root=config.repo_root,
        results_dir=exp5323.results_dir(config.repo_root),
        device=config.device,
        epochs=exp5323.EPOCHS,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _source_spec(seed: int) -> exp5323.RunSpec:
    return exp5323.RunSpec(SOURCE_WHEN_CONDITION, seed)


def _base_spec(seed: int) -> exp541.RunSpec:
    return exp541.RunSpec(seed)


def _labels_lengths(data: base.Data, split: str) -> tuple[np.ndarray, np.ndarray]:
    return {
        "train": (data.ytr, data.ltr),
        "val": (data.yva, data.lva),
        "test": (data.yte, data.lte),
    }[split]


def _valid_channel_stats(values: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.concatenate([values[i, : int(length)] for i, length in enumerate(lengths)], axis=0).astype(np.float64)
    mean = valid.mean(axis=0)
    std = valid.std(axis=0)
    std = np.where(std < MEM_STD_FLOOR, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _scale_membrane(values: np.ndarray, lengths: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = ((values.astype(np.float32) - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    for i, length in enumerate(lengths):
        out[i, int(length) :] = 0.0
    return out


def prepare_source_seed(seed: int, data: base.Data, config: Config, force: bool = False) -> Path:
    destination = source_cache_path(config.results_dir, seed)
    meta_destination = source_meta_path(config.results_dir, seed)
    if destination.exists() and meta_destination.exists() and not force:
        load_source_cache(seed, data, config)
        return destination

    base_config = _exp541_config(config)
    exp541.prepare_inputs_seed(seed, data, base_config, force=False)
    exp541.train_base(_base_spec(seed), data, base_config, force=False)

    fusion_config = _exp54_config(config)
    exp54.prepare_fusion_seed(seed, data, fusion_config, force=False)
    fusion_arrays, _ = exp54.load_fusion_cache(seed, data, fusion_config)

    when_config = _exp5323_config(config)
    source_model, source_payload = exp5323.load_model(_source_spec(seed), data, when_config)
    source_model.eval()
    local_cache = parent.load_local_cache(seed, data, exp5323._parent_config(when_config))
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    arrays: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for split, (what_np, lengths_np) in parent._partitions(data, local_cache).items():
            loader = DataLoader(
                TensorDataset(torch.tensor(what_np, dtype=torch.float32), torch.tensor(lengths_np, dtype=torch.long)),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
            )
            osp, rsp, omp, rmp = [], [], [], []
            for what, lengths in loader:
                what, lengths = what.to(device), lengths.to(device)
                ordered = source_model.forward_trajectory(what, lengths, reset_state_each_step=False)
                reset = source_model.forward_trajectory(what, lengths, reset_state_each_step=True)
                osp.append(ordered.spikes.cpu().numpy().astype(np.uint8))
                rsp.append(reset.spikes.cpu().numpy().astype(np.uint8))
                omp.append(ordered.membranes.cpu().numpy().astype(np.float32))
                rmp.append(reset.membranes.cpu().numpy().astype(np.float32))
            ordered_spikes = np.concatenate(osp)
            reset_spikes = np.concatenate(rsp)
            if not np.array_equal(ordered_spikes, fusion_arrays[f"when_ordered_{split}"]):
                raise ValueError(f"ordered WHEN mismatch for {split}")
            if not np.array_equal(reset_spikes, fusion_arrays[f"when_reset_{split}"]):
                raise ValueError(f"reset WHEN mismatch for {split}")
            if not np.array_equal(np.asarray(what_np, dtype=np.uint8), fusion_arrays[f"what_{split}"]):
                raise ValueError(f"WHAT mismatch for {split}")
            arrays[f"what_{split}"] = np.asarray(what_np, dtype=np.uint8)
            arrays[f"when_spike_ordered_{split}"] = ordered_spikes
            arrays[f"when_spike_reset_{split}"] = reset_spikes
            arrays[f"when_mem_ordered_{split}"] = np.concatenate(omp)
            arrays[f"when_mem_reset_{split}"] = np.concatenate(rmp)

    mem_mean, mem_std = _valid_channel_stats(arrays["when_mem_ordered_train"], data.ltr)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    _save_json(
        meta_destination,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "source_when_experiment": exp5323.EXPERIMENT_ID,
            "source_when_protocol": exp5323.PROTOCOL_VERSION,
            "source_when_condition": SOURCE_WHEN_CONDITION,
            "source_when_best_epoch": source_payload["result"]["best_epoch"],
            "source_base_experiment": exp541.EXPERIMENT_ID,
            "source_base_protocol": exp541.PROTOCOL_VERSION,
            "membrane_scaler": {
                "fit_split": "train",
                "fit_scope": "valid timesteps only; padding excluded",
                "mean": mem_mean.tolist(),
                "std": mem_std.tolist(),
            },
            "train_max_elapsed_seconds": float((int(np.max(data.ltr)) - 1) / float(data.fs)),
            "split_seed": base.SPLIT_SEED,
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "sampling_rate_hz": float(data.fs),
        },
    )
    load_source_cache(seed, data, config)
    return destination


def load_source_cache(seed: int, data: base.Data, config: Config) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path, meta_path = source_cache_path(config.results_dir, seed), source_meta_path(config.results_dir, seed)
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing source cache for seed {seed}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("experiment_id") != EXPERIMENT_ID or meta.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("source cache identity mismatch")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    for split in ("train", "val", "test"):
        labels, _ = _labels_lengths(data, split)
        for prefix, width in (("what", WHAT_WIDTH), ("when_spike_ordered", WHEN_WIDTH), ("when_spike_reset", WHEN_WIDTH), ("when_mem_ordered", WHEN_WIDTH), ("when_mem_reset", WHEN_WIDTH)):
            key = f"{prefix}_{split}"
            if arrays[key].shape != (len(labels), data.T, width):
                raise ValueError(f"cache shape mismatch: {key}")
    return arrays, meta


def _loader(arrays: dict[str, np.ndarray], data: base.Data, split: str, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    labels, lengths = _labels_lengths(data, split)
    return DataLoader(
        TensorDataset(
            torch.tensor(arrays[f"what_{split}"], dtype=torch.float32),
            torch.tensor(arrays[f"when_spike_ordered_{split}"], dtype=torch.float32),
            torch.tensor(arrays[f"when_spike_reset_{split}"], dtype=torch.float32),
            torch.tensor(arrays[f"when_mem_ordered_{split}"], dtype=torch.float32),
            torch.tensor(arrays[f"when_mem_reset_{split}"], dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def _make_loaders(arrays: dict[str, np.ndarray], data: base.Data, spec: RunSpec, config: Config, train_shuffle: bool, splits: tuple[str, ...] = ("train", "val")) -> dict[str, DataLoader]:
    return {
        split: _loader(
            arrays,
            data,
            split,
            config.batch_size,
            train_shuffle if split == "train" else False,
            base.dseed(spec.seed, EXPERIMENT_ID, spec.stage, spec.condition, spec.n_banks, spec.rank, split),
        )
        for split in splits
    }


def relative_phase_oracle_context(lengths: torch.Tensor, steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    centers = torch.linspace(0.0, 1.0, WHEN_WIDTH, device=device, dtype=dtype)
    sigma = 1.5 / max(WHEN_WIDTH - 1, 1)
    time = torch.arange(steps, device=device, dtype=dtype).unsqueeze(0)
    denom = (lengths.to(dtype) - 1).clamp(min=1).unsqueeze(1)
    phase = (time / denom).clamp(0, 1)
    basis = torch.exp(-0.5 * ((phase.unsqueeze(-1) - centers.view(1, 1, -1)) / sigma) ** 2)
    return basis * base.mask(lengths, steps).to(dtype).unsqueeze(-1)


def _scale_membrane_tensor(membrane: torch.Tensor, lengths: torch.Tensor, meta: dict[str, object]) -> torch.Tensor:
    scaler = meta["membrane_scaler"]
    mean = torch.tensor(scaler["mean"], dtype=membrane.dtype, device=membrane.device)
    std = torch.tensor(scaler["std"], dtype=membrane.dtype, device=membrane.device)
    scaled = (membrane - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    return scaled * base.mask(lengths, membrane.shape[1]).to(membrane.dtype).unsqueeze(-1)


def _mean_valid_context(context: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = base.mask(lengths, context.shape[1]).to(context.dtype).unsqueeze(-1)
    mean = (context * valid).sum(1) / lengths.to(context.dtype).unsqueeze(1)
    return mean.unsqueeze(1).expand_as(context) * valid


def _native_context(spec: RunSpec, spike: torch.Tensor, membrane: torch.Tensor, lengths: torch.Tensor, meta: dict[str, object], data: base.Data) -> torch.Tensor:
    if spec.context_source == "spike":
        return spike
    mem = _scale_membrane_tensor(membrane, lengths, meta)
    if spec.context_source == "membrane":
        return mem
    if spec.context_source == "membrane_mean":
        return _mean_valid_context(mem, lengths)
    if spec.context_source == "elapsed":
        return exp54.elapsed_context(spike.shape[0], spike.shape[1], data.fs, float(meta["train_max_elapsed_seconds"]), spike.device, spike.dtype)
    if spec.context_source == "relative_phase_oracle":
        return relative_phase_oracle_context(lengths, spike.shape[1], spike.device, spike.dtype)
    return torch.zeros_like(spike)


class PhaseConditionedReadout(nn.Module):
    def __init__(self, base_model: exp541.DirectWhatBase, n_classes: int, mechanism: str, n_banks: int = SCREEN_BANKS, rank: int = SCREEN_RANK) -> None:
        super().__init__()
        self.base_model = base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.mechanism = mechanism
        self.n_banks = n_banks
        self.rank = rank
        if mechanism == "additive":
            self.additive = nn.Linear(WHEN_WIDTH, n_classes, bias=False)
            nn.init.zeros_(self.additive.weight)
        elif mechanism == "bilinear":
            self.what_factor = nn.Linear(WHAT_WIDTH, rank, bias=False)
            self.when_factor = nn.Linear(WHEN_WIDTH, rank, bias=False)
            self.bilinear_output = nn.Linear(rank, n_classes, bias=False)
            self.alpha = nn.Parameter(torch.tensor(0.0))
        elif mechanism == "bank":
            self.gate = nn.Linear(WHEN_WIDTH, n_banks, bias=False)
            self.bank_u = nn.Parameter(torch.empty(n_banks, n_classes, rank))
            self.bank_v = nn.Parameter(torch.empty(n_banks, WHAT_WIDTH, rank))
            nn.init.xavier_uniform_(self.bank_u)
            nn.init.xavier_uniform_(self.bank_v)
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(mechanism)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, what: torch.Tensor, context: torch.Tensor, lengths: torch.Tensor, *, zero_residual_what: bool = False, return_trajectory: bool = False):
        with torch.no_grad():
            base_logits, base_evidence = self.base_model(what, lengths, return_evidence=True)
        valid = base.mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
        residual_what = torch.zeros_like(what) if zero_residual_what else what
        probs = centered = None
        if self.mechanism == "additive":
            residual_evidence = self.additive(context) * valid
        elif self.mechanism == "bilinear":
            q = self.what_factor(residual_what) * self.when_factor(context)
            residual_evidence = self.alpha * self.bilinear_output(q) * valid
        else:
            probs = torch.softmax(self.gate(context), dim=-1)
            centered = probs - 1.0 / self.n_banks
            projected = torch.einsum("bti,kir->btkr", residual_what, self.bank_v)
            bank_evidence = torch.einsum("btkr,kcr->btkc", projected, self.bank_u)
            residual_evidence = self.alpha * (centered.unsqueeze(-1) * bank_evidence).sum(2) * valid
        residual_logits = residual_evidence.sum(1)
        logits = base_logits + residual_logits
        trajectory = None
        if return_trajectory:
            trajectory = ReadoutTrajectory(base_logits, residual_logits, base_evidence, residual_evidence, probs, centered)
        return logits, trajectory


def parameter_counts(model: PhaseConditionedReadout) -> dict[str, int]:
    return {
        "frozen_base": int(sum(p.numel() for p in model.base_model.parameters())),
        "trainable_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "stored_total": int(sum(p.numel() for p in model.parameters())),
    }


def _initialize_model(spec: RunSpec, data: base.Data, config: Config) -> tuple[PhaseConditionedReadout, dict[str, object]]:
    base_model, payload = exp541.load_base(_base_spec(spec.seed), data, _exp541_config(config))
    base.seed_all(base.dseed(spec.seed, EXPERIMENT_ID, spec.stage, spec.condition, spec.n_banks, spec.rank))
    return PhaseConditionedReadout(base_model, len(data.labels), spec.mechanism, spec.n_banks, spec.rank).to(config.device), payload


def _classification_metrics(true: np.ndarray, pred: np.ndarray, loss: float) -> dict[str, float | int]:
    return {
        "loss": float(loss),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro")),
        "n_samples": int(len(true)),
    }


def _same_source_context(spec: RunSpec, spike_o: torch.Tensor, spike_r: torch.Tensor, mem_o: torch.Tensor, mem_r: torch.Tensor, lengths: torch.Tensor, meta: dict[str, object], reset: bool = False) -> torch.Tensor:
    if spec.context_source == "spike":
        return spike_r if reset else spike_o
    raw = mem_r if reset else mem_o
    return _scale_membrane_tensor(raw, lengths, meta)


def _ablation_context(spec: RunSpec, ablation: str, spike_o: torch.Tensor, spike_r: torch.Tensor, mem_o: torch.Tensor, mem_r: torch.Tensor, lengths: torch.Tensor, meta: dict[str, object], replicate: int, batch_index: int) -> tuple[torch.Tensor, bool]:
    ordered = _same_source_context(spec, spike_o, spike_r, mem_o, mem_r, lengths, meta)
    if ablation == "ordered":
        return ordered, False
    if ablation == "mean_when":
        return _mean_valid_context(ordered, lengths), False
    if ablation == "when_zero":
        return torch.zeros_like(ordered), False
    if ablation == "when_shuffle":
        return exp54._shuffle_valid_context(ordered, lengths, base.dseed(spec.seed, EXPERIMENT_ID, spec.key, ablation, replicate, batch_index)), False
    if ablation == "when_circular_shift":
        return exp54._circular_shift_valid_context(ordered, lengths), False
    if ablation == "reset_when":
        return _same_source_context(spec, spike_o, spike_r, mem_o, mem_r, lengths, meta, reset=True), False
    if ablation == "residual_what_zero":
        return ordered, True
    raise ValueError(ablation)


def _evaluate(model: PhaseConditionedReadout, loader: DataLoader, spec: RunSpec, data: base.Data, meta: dict[str, object], device: torch.device, *, ablation: str = "ordered", replicate: int = 0, diagnostics: bool = False) -> dict[str, Any]:
    model.eval()
    ys, ps, bps, losses, ratios = [], [], [], [], []
    gate_entropy, gate_diffs = [], []
    utilization = np.zeros(model.n_banks if model.mechanism == "bank" else 0, dtype=np.int64)
    phase_sum = np.zeros((RELATIVE_PHASE_BINS, model.n_banks), dtype=np.float64) if model.mechanism == "bank" else None
    phase_n = np.zeros(RELATIVE_PHASE_BINS, dtype=np.int64) if model.mechanism == "bank" else None
    prefix_records: dict[float, list[tuple[int, int, int]]] = {rho: [] for rho in PREFIX_FRACTIONS}
    ordered_spikes = reset_spikes = valid_spike_denom = 0.0
    max_abs = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            what, spike_o, spike_r, mem_o, mem_r, labels, lengths = [x.to(device) for x in batch]
            if spec.condition in CANDIDATE_CONDITIONS:
                context, zero_what = _ablation_context(spec, ablation, spike_o, spike_r, mem_o, mem_r, lengths, meta, replicate, batch_index)
            else:
                context, zero_what = _native_context(spec, spike_o, mem_o, lengths, meta, data), False
            logits, tr = model(what, context, lengths, zero_residual_what=zero_what, return_trajectory=True)
            loss = F.cross_entropy(logits, labels)
            pred, base_pred = logits.argmax(1), tr.base_logits.argmax(1)
            ys.append(labels.cpu().numpy()); ps.append(pred.cpu().numpy()); bps.append(base_pred.cpu().numpy())
            losses.append((float(loss.item()), len(labels)))
            ratio = torch.linalg.vector_norm(tr.residual_logits, dim=1) / (torch.linalg.vector_norm(tr.base_logits, dim=1) + 1e-8)
            ratios.extend(ratio.cpu().tolist())
            max_abs = max(max_abs, float((logits - tr.base_logits).abs().max().item()))
            if diagnostics:
                valid = base.mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
                ordered_spikes += float((spike_o * valid).sum().item())
                reset_spikes += float((spike_r * valid).sum().item())
                valid_spike_denom += float(valid.sum().item()) * WHEN_WIDTH
                for row, length_t in enumerate(lengths.cpu()):
                    length = int(length_t.item())
                    bc = tr.base_evidence[row, :length].cumsum(0)
                    rc = tr.residual_evidence[row, :length].cumsum(0)
                    for rho in PREFIX_FRACTIONS:
                        endpoint = min(length - 1, max(0, int(math.ceil(rho * length)) - 1))
                        prefix_records[rho].append((int(labels[row]), int((bc[endpoint] + model.base_model.class_bias).argmax()), int((bc[endpoint] + rc[endpoint] + model.base_model.class_bias).argmax())))
                if model.mechanism == "bank":
                    valid_bool = base.mask(lengths, what.shape[1])
                    vp = tr.gate_probabilities[valid_bool]
                    gate_entropy.extend((-(vp * torch.log(vp.clamp_min(1e-12))).sum(1)).cpu().tolist())
                    utilization += np.bincount(vp.argmax(1).cpu().numpy(), minlength=model.n_banks)
                    for row, length_t in enumerate(lengths.cpu()):
                        length = int(length_t.item())
                        if length > 1:
                            gate_diffs.extend(torch.linalg.vector_norm(tr.centered_gates[row, 1:length] - tr.centered_gates[row, : length - 1], dim=1).cpu().tolist())
                        for t in range(length):
                            b = min(RELATIVE_PHASE_BINS - 1, int((t / max(length - 1, 1)) * RELATIVE_PHASE_BINS))
                            phase_sum[b] += tr.centered_gates[row, t].cpu().numpy(); phase_n[b] += 1
    true, pred, base_pred = np.concatenate(ys), np.concatenate(ps), np.concatenate(bps)
    out = _classification_metrics(true, pred, sum(v * n for v, n in losses) / sum(n for _, n in losses))
    r = np.asarray(ratios)
    out["residual_ratio"] = {"mean": float(r.mean()), "median": float(np.median(r)), "p95": float(np.percentile(r, 95)), "max": float(r.max())}
    out["max_abs_logit_delta_vs_base"] = max_abs
    if diagnostics:
        bc, cc = base_pred == true, pred == true
        out["rescue_harm"] = {"rescued": int((~bc & cc).sum()), "harmed": int((bc & ~cc).sum()), "retained_correct": int((bc & cc).sum()), "retained_error": int((~bc & ~cc).sum())}
        out["prefix"] = []
        for rho, records in prefix_records.items():
            arr = np.asarray(records)
            out["prefix"].append({"progress": rho, "base_balanced_accuracy": float(balanced_accuracy_score(arr[:, 0], arr[:, 1])), "combined_balanced_accuracy": float(balanced_accuracy_score(arr[:, 0], arr[:, 2]))})
        out["when_source_activity"] = {"ordered_firing_fraction": ordered_spikes / max(valid_spike_denom, 1.0), "reset_firing_fraction": reset_spikes / max(valid_spike_denom, 1.0)}
        if model.mechanism == "bank":
            out["gate"] = {
                "entropy_mean": float(np.mean(gate_entropy)),
                "temporal_variation_mean": float(np.mean(gate_diffs)) if gate_diffs else 0.0,
                "utilization": (utilization / max(utilization.sum(), 1)).tolist(),
                "relative_phase_centered_gate": [(phase_sum[i] / max(phase_n[i], 1)).tolist() for i in range(RELATIVE_PHASE_BINS)],
            }
        else:
            out["gate"] = None
    return out


def _evaluate_base(model: exp541.DirectWhatBase, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval(); ys, ps, losses = [], [], []
    with torch.no_grad():
        for what, _so, _sr, _mo, _mr, labels, lengths in loader:
            what, labels, lengths = what.to(device), labels.to(device), lengths.to(device)
            logits, _ = model(what, lengths); loss = F.cross_entropy(logits, labels)
            ys.append(labels.cpu().numpy()); ps.append(logits.argmax(1).cpu().numpy()); losses.append((float(loss.item()), len(labels)))
    true, pred = np.concatenate(ys), np.concatenate(ps)
    return _classification_metrics(true, pred, sum(v * n for v, n in losses) / sum(n for _, n in losses))


def train_one(spec: RunSpec, data: base.Data, config: Config, force: bool = False) -> Path:
    _validate_spec(spec)
    if spec.condition == WHAT_ONLY:
        raise ValueError("WHAT-only is frozen")
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    arrays, meta = load_source_cache(spec.seed, data, config)
    train_loaders = _make_loaders(arrays, data, spec, config, True, splits=("train", "val"))
    eval_loaders = _make_loaders(arrays, data, spec, config, False, splits=("train", "val"))
    device = torch.device(config.device); torch.set_num_threads(config.threads)
    model, base_payload = _initialize_model(spec, data, config)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    epoch0_train = _evaluate(model, eval_loaders["train"], spec, data, meta, device)
    epoch0_val = _evaluate(model, eval_loaders["val"], spec, data, meta, device)
    best_ba, best_loss, best_epoch = float(epoch0_val["balanced_accuracy"]), float(epoch0_val["loss"]), 0
    best_state = copy.deepcopy(model.state_dict())
    history = [{"epoch": 0, "train_loss": epoch0_train["loss"], "train_balanced_accuracy": epoch0_train["balanced_accuracy"], "val_loss": best_loss, "val_balanced_accuracy": best_ba}]
    for epoch in range(1, config.epochs + 1):
        model.train(); ys, ps, loss_sum, n_total = [], [], 0.0, 0
        for what, spike_o, _spike_r, mem_o, _mem_r, labels, lengths in train_loaders["train"]:
            what, spike_o, mem_o, labels, lengths = [x.to(device) for x in (what, spike_o, mem_o, labels, lengths)]
            context = _native_context(spec, spike_o, mem_o, lengths, meta, data)
            optimizer.zero_grad(set_to_none=True); logits, _ = model(what, context, lengths)
            loss = F.cross_entropy(logits, labels); loss.backward(); optimizer.step()
            n = len(labels); loss_sum += float(loss.item()) * n; n_total += n
            ys.append(labels.cpu().numpy()); ps.append(logits.detach().argmax(1).cpu().numpy())
        val = _evaluate(model, eval_loaders["val"], spec, data, meta, device)
        val_ba, val_loss = float(val["balanced_accuracy"]), float(val["loss"])
        history.append({"epoch": epoch, "train_loss": loss_sum / n_total, "train_balanced_accuracy": float(balanced_accuracy_score(np.concatenate(ys), np.concatenate(ps))), "val_loss": val_loss, "val_balanced_accuracy": val_ba})
        if val_ba > best_ba + 1e-12 or (abs(val_ba - best_ba) <= 1e-12 and val_loss < best_loss):
            best_ba, best_loss, best_epoch, best_state = val_ba, val_loss, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    val_ablations = []
    if spec.condition in CANDIDATE_CONDITIONS:
        for ablation in ("ordered", "mean_when", "when_zero", "when_circular_shift", "reset_when", "residual_what_zero"):
            val_ablations.append({"ablation": ablation, "replicate": 0, "val": _evaluate(model, eval_loaders["val"], spec, data, meta, device, ablation=ablation)})
        for replicate in range(SHUFFLE_REPLICATES):
            val_ablations.append({"ablation": "when_shuffle", "replicate": replicate, "val": _evaluate(model, eval_loaders["val"], spec, data, meta, device, ablation="when_shuffle", replicate=replicate)})
        base_val = float(base_payload["result"]["native"]["val"]["balanced_accuracy"])
        for item in val_ablations:
            if item["ablation"] in ("when_zero", "residual_what_zero"):
                if float(item["val"]["max_abs_logit_delta_vs_base"]) > 1e-6 or abs(float(item["val"]["balanced_accuracy"]) - base_val) > 1e-12:
                    raise RuntimeError("zero-correction architecture contract failed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION, "stage": spec.stage, "spec": asdict(spec), "result": {"best_epoch": best_epoch, "best_val_balanced_accuracy": best_ba, "best_val_loss": best_loss, "val_ablations": val_ablations}, "state_dict": best_state}, destination)
    hp = history_path(config.results_dir, spec); hp.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(history).to_csv(hp, index=False)
    return destination


def load_model(spec: RunSpec, data: base.Data, config: Config) -> tuple[PhaseConditionedReadout, dict[str, object]]:
    payload = torch.load(checkpoint_path(config.results_dir, spec), map_location=config.device, weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("spec") != asdict(spec):
        raise ValueError("checkpoint identity mismatch")
    model, _ = _initialize_model(spec, data, config); model.load_state_dict(payload["state_dict"]); model.eval()
    return model, payload


def evaluate_validation_one(spec: RunSpec, data: base.Data, config: Config, force: bool = False) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    arrays, meta = load_source_cache(spec.seed, data, config)
    loaders = _make_loaders(arrays, data, spec, config, False, splits=("train", "val"))
    device = torch.device(config.device)
    base_model, base_payload = exp541.load_base(_base_spec(spec.seed), data, _exp541_config(config))
    base_native = {split: _evaluate_base(base_model, loaders[split], device) for split in ("train", "val")}
    if spec.condition == WHAT_ONLY:
        payload = {"experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION, "stage": spec.stage, "spec": asdict(spec), "parameter_counts": {"trainable_total": 0}, "best_epoch": base_payload["result"]["best_epoch"], "base_native": base_native, "native": base_native, "val_ablations": [], "test_evaluated": False}
    else:
        model, checkpoint = load_model(spec, data, config)
        native = {split: _evaluate(model, loaders[split], spec, data, meta, device) for split in ("train", "val")}
        payload = {"experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION, "stage": spec.stage, "spec": asdict(spec), "parameter_counts": parameter_counts(model), "best_epoch": checkpoint["result"]["best_epoch"], "base_native": base_native, "native": native, "val_ablations": checkpoint["result"]["val_ablations"], "test_evaluated": False}
    _save_json(destination, payload); return payload


def run_validation_one(spec: RunSpec, data: base.Data, config: Config, force: bool = False) -> dict[str, object]:
    if spec.condition != WHAT_ONLY:
        train_one(spec, data, config, force)
    return evaluate_validation_one(spec, data, config, force)


def _load_eval(root: Path, spec: RunSpec) -> dict[str, object]:
    payload = json.loads(evaluation_path(root, spec).read_text(encoding="utf-8"))
    if payload.get("test_evaluated") is not False:
        raise ValueError("selection stage artifact contains test")
    return payload


def _ablation_mean(payload: dict[str, object], ablation: str) -> float:
    vals = [float(item["val"]["balanced_accuracy"]) for item in payload["val_ablations"] if item["ablation"] == ablation]
    return float(np.mean(vals))


def _sem(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def finalize_screen(repo_root: Path) -> dict[str, Path]:
    root, specs = results_dir(repo_root), screen_specs()
    if len(specs) != 50 or len({s.key for s in specs}) != 50:
        raise RuntimeError("screen must be 10 conditions x 5 seeds")
    payloads = {(s.condition, s.seed): _load_eval(root, s) for s in specs}
    base_val = {seed: float(payloads[(WHAT_ONLY, seed)]["native"]["val"]["balanced_accuracy"]) for seed in SEEDS}
    rows, paired = [], []
    for s in specs:
        p, val = payloads[(s.condition, s.seed)], payloads[(s.condition, s.seed)]["native"]["val"]
        delta = float(val["balanced_accuracy"]) - base_val[s.seed]
        rows.append({"condition": s.condition, "mechanism": s.mechanism, "context_source": s.context_source, "seed": s.seed, "n_banks": s.n_banks if s.mechanism == "bank" else None, "rank": s.rank if s.mechanism in ("bank", "bilinear") else None, "trainable_parameter_count": p["parameter_counts"]["trainable_total"], "best_epoch": p["best_epoch"], "val_balanced_accuracy": val["balanced_accuracy"], "delta_val_balanced_accuracy_vs_what": delta, "test_evaluated": False})
        if s.condition != WHAT_ONLY:
            paired.append({"condition": s.condition, "seed": s.seed, "delta_val_balanced_accuracy_vs_what": delta, "candidate": s.condition in CANDIDATE_CONDITIONS})
    elapsed_mean = float(np.mean([payloads[(BANK_ELAPSED, seed)]["native"]["val"]["balanced_accuracy"] for seed in SEEDS]))
    summaries = []
    for condition in CANDIDATE_CONDITIONS:
        deltas = np.asarray([float(payloads[(condition, seed)]["native"]["val"]["balanced_accuracy"]) - base_val[seed] for seed in SEEDS])
        ordered = np.asarray([float(payloads[(condition, seed)]["native"]["val"]["balanced_accuracy"]) for seed in SEEDS])
        mean_ctx = np.asarray([_ablation_mean(payloads[(condition, seed)], "mean_when") for seed in SEEDS])
        shuffle = np.asarray([_ablation_mean(payloads[(condition, seed)], "when_shuffle") for seed in SEEDS])
        circular = np.asarray([_ablation_mean(payloads[(condition, seed)], "when_circular_shift") for seed in SEEDS])
        trainable = int(payloads[(condition, SEEDS[0])]["parameter_counts"]["trainable_total"])
        item = {"condition": condition, "mechanism": RunSpec("screen", condition, SEEDS[0]).mechanism, "context_source": RunSpec("screen", condition, SEEDS[0]).context_source, "mean_paired_val_delta": float(deltas.mean()), "sem_paired_val_delta": _sem(deltas), "nonnegative_seed_count": int((deltas >= -1e-12).sum()), "mean_ordered_val_ba": float(ordered.mean()), "mean_same_source_mean_context_val_ba": float(mean_ctx.mean()), "mean_shuffle_val_ba": float(shuffle.mean()), "mean_circular_val_ba": float(circular.mean()), "elapsed_control_mean_val_ba": elapsed_mean, "trainable_parameter_count": trainable}
        item["eligible"] = bool(item["mean_paired_val_delta"] > 0 and item["nonnegative_seed_count"] >= 4 and ordered.mean() > mean_ctx.mean() and ordered.mean() > shuffle.mean() and ordered.mean() > circular.mean() and ordered.mean() > elapsed_mean)
        summaries.append(item)
    eligible = [x for x in summaries if x["eligible"]]
    selected = None
    if eligible:
        best = max(eligible, key=lambda x: x["mean_paired_val_delta"]); cutoff = best["mean_paired_val_delta"] - best["sem_paired_val_delta"]
        selected = min([x for x in eligible if x["mean_paired_val_delta"] >= cutoff], key=lambda x: (x["trainable_parameter_count"], -x["mean_paired_val_delta"]))
    outputs = {"screen_runs": root / "screen_runs.csv", "screen_paired_deltas": root / "screen_paired_deltas.csv", "screen_selection": screen_selection_path(root)}
    root.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(outputs["screen_runs"], index=False); pd.DataFrame(paired).to_csv(outputs["screen_paired_deltas"], index=False)
    _save_json(outputs["screen_selection"], {"experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION, "stage": "screen", "candidate_summary": summaries, "selected_condition": None if selected is None else selected["condition"], "selected_mechanism": None if selected is None else selected["mechanism"], "selected_context_source": None if selected is None else selected["context_source"], "selection_metric": "mean paired validation BA delta vs frozen WHAT base", "only_validation_selected": True, "test_not_used_for_selection": True, "status": "selected" if selected else "no_supported_candidate"})
    return outputs


def _load_screen_selection(root: Path) -> dict[str, object]:
    p = json.loads(screen_selection_path(root).read_text(encoding="utf-8"))
    if p.get("only_validation_selected") is not True or p.get("test_not_used_for_selection") is not True:
        raise ValueError("invalid screen selection")
    return p


def refine_specs(root: Path) -> list[RunSpec]:
    selection = _load_screen_selection(root); condition = selection.get("selected_condition")
    if condition is None:
        raise RuntimeError("screen found no supported candidate; test remains unopened")
    probe = RunSpec("refine", str(condition), SEEDS[0])
    if probe.mechanism == "bank":
        return [RunSpec("refine", str(condition), seed, k, r) for k in BANK_SWEEP for r in RANK_SWEEP for seed in SEEDS]
    return [RunSpec("refine", str(condition), seed, SCREEN_BANKS, r) for r in RANK_SWEEP for seed in SEEDS]


def finalize_refine(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root); screen = _load_screen_selection(root); specs = refine_specs(root)
    expected = 45 if screen["selected_mechanism"] == "bank" else 15
    if len(specs) != expected:
        raise RuntimeError("unexpected refine mapping")
    payloads = {(s.n_banks, s.rank, s.seed): _load_eval(root, s) for s in specs}
    base_val = {seed: float(next(p for (k, r, sd), p in payloads.items() if sd == seed)["base_native"]["val"]["balanced_accuracy"]) for seed in SEEDS}
    elapsed = float(next(x for x in screen["candidate_summary"] if x["condition"] == screen["selected_condition"])["elapsed_control_mean_val_ba"])
    rows, summaries = [], []
    for s in specs:
        p = payloads[(s.n_banks, s.rank, s.seed)]; val = p["native"]["val"]; delta = float(val["balanced_accuracy"]) - base_val[s.seed]
        rows.append({"condition": s.condition, "mechanism": s.mechanism, "context_source": s.context_source, "seed": s.seed, "n_banks": s.n_banks if s.mechanism == "bank" else None, "rank": s.rank, "trainable_parameter_count": p["parameter_counts"]["trainable_total"], "best_epoch": p["best_epoch"], "val_balanced_accuracy": val["balanced_accuracy"], "delta_val_balanced_accuracy_vs_what": delta, "test_evaluated": False})
    for k, r in sorted({(s.n_banks, s.rank) for s in specs}):
        pp = [payloads[(k, r, seed)] for seed in SEEDS]
        deltas = np.asarray([float(p["native"]["val"]["balanced_accuracy"]) - base_val[seed] for p, seed in zip(pp, SEEDS)])
        ordered = np.asarray([float(p["native"]["val"]["balanced_accuracy"]) for p in pp]); mean_ctx = np.asarray([_ablation_mean(p, "mean_when") for p in pp]); shuffle = np.asarray([_ablation_mean(p, "when_shuffle") for p in pp]); circular = np.asarray([_ablation_mean(p, "when_circular_shift") for p in pp])
        item = {"condition": specs[0].condition, "n_banks": k if specs[0].mechanism == "bank" else None, "rank": r, "mean_paired_val_delta": float(deltas.mean()), "sem_paired_val_delta": _sem(deltas), "nonnegative_seed_count": int((deltas >= -1e-12).sum()), "trainable_parameter_count": int(pp[0]["parameter_counts"]["trainable_total"]), "eligible": bool(deltas.mean() > 0 and (deltas >= -1e-12).sum() >= 4 and ordered.mean() > mean_ctx.mean() and ordered.mean() > shuffle.mean() and ordered.mean() > circular.mean() and ordered.mean() > elapsed)}
        summaries.append(item)
    eligible = [x for x in summaries if x["eligible"]]
    if not eligible:
        raise RuntimeError("refine found no supported capacity; test remains unopened")
    best = max(eligible, key=lambda x: x["mean_paired_val_delta"]); cutoff = best["mean_paired_val_delta"] - best["sem_paired_val_delta"]
    selected = min([x for x in eligible if x["mean_paired_val_delta"] >= cutoff], key=lambda x: (x["trainable_parameter_count"], -x["mean_paired_val_delta"]))
    outputs = {"refine_runs": root / "refine_runs.csv", "selection": selection_path(root)}; pd.DataFrame(rows).to_csv(outputs["refine_runs"], index=False)
    _save_json(outputs["selection"], {"experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION, "selected_condition": selected["condition"], "selected_mechanism": screen["selected_mechanism"], "selected_context_source": screen["selected_context_source"], "selected_n_banks": selected["n_banks"], "selected_rank": selected["rank"], "selection_metric": "mean paired validation BA delta vs frozen WHAT base", "refine_summary": summaries, "only_validation_selected": True, "test_not_used_for_selection": True})
    return outputs


def _load_selection(root: Path) -> dict[str, object]:
    p = json.loads(selection_path(root).read_text(encoding="utf-8"))
    if p.get("only_validation_selected") is not True or p.get("test_not_used_for_selection") is not True:
        raise ValueError("invalid locked selection")
    return p


def selected_refine_spec(root: Path, seed: int) -> RunSpec:
    s = _load_selection(root)
    return RunSpec("refine", str(s["selected_condition"]), seed, int(s["selected_n_banks"] or SCREEN_BANKS), int(s["selected_rank"]))


def evaluate_final_seed(seed: int, data: base.Data, config: Config, force: bool = False) -> dict[str, object]:
    destination = final_evaluation_path(config.results_dir, seed)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    selection = _load_selection(config.results_dir); spec = selected_refine_spec(config.results_dir, seed)
    arrays, meta = load_source_cache(seed, data, config)
    test_loader = _make_loaders(arrays, data, spec, config, False, splits=("test",))["test"]
    model, checkpoint = load_model(spec, data, config); device = torch.device(config.device)
    base_model, _ = exp541.load_base(_base_spec(seed), data, _exp541_config(config)); base_test = _evaluate_base(base_model, test_loader, device)
    ordered = _evaluate(model, test_loader, spec, data, meta, device, diagnostics=True)
    ablations = [{"ablation": "ordered", "replicate": 0, "test": ordered}]
    for ablation in ("mean_when", "when_zero", "when_circular_shift", "reset_when", "residual_what_zero"):
        metrics = _evaluate(model, test_loader, spec, data, meta, device, ablation=ablation)
        if ablation in ("when_zero", "residual_what_zero") and (float(metrics["max_abs_logit_delta_vs_base"]) > 1e-6 or abs(float(metrics["balanced_accuracy"]) - float(base_test["balanced_accuracy"])) > 1e-12):
            raise RuntimeError("final zero-correction contract failed")
        ablations.append({"ablation": ablation, "replicate": 0, "test": metrics})
    for replicate in range(SHUFFLE_REPLICATES):
        ablations.append({"ablation": "when_shuffle", "replicate": replicate, "test": _evaluate(model, test_loader, spec, data, meta, device, ablation="when_shuffle", replicate=replicate)})
    payload = {"experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION, "selection": selection, "spec": asdict(spec), "seed": seed, "best_epoch": checkpoint["result"]["best_epoch"], "parameter_counts": parameter_counts(model), "base_test": base_test, "ordered_test": ordered, "ablations": ablations, "test_evaluated": True}
    _save_json(destination, payload); return payload


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root); selection = _load_selection(root)
    final_rows, paired_rows, ablation_rows, gate_rows, residual_rows, rescue_rows, prefix_rows = [], [], [], [], [], [], []
    for seed in SEEDS:
        p = json.loads(final_evaluation_path(root, seed).read_text(encoding="utf-8")); base_test, ordered = p["base_test"], p["ordered_test"]
        delta = float(ordered["balanced_accuracy"]) - float(base_test["balanced_accuracy"])
        final_rows.append({"condition": selection["selected_condition"], "mechanism": selection["selected_mechanism"], "context_source": selection["selected_context_source"], "seed": seed, "n_banks": selection["selected_n_banks"], "rank": selection["selected_rank"], "best_epoch": p["best_epoch"], "trainable_parameter_count": p["parameter_counts"]["trainable_total"], "base_test_balanced_accuracy": base_test["balanced_accuracy"], "test_balanced_accuracy": ordered["balanced_accuracy"], "delta_test_ba_vs_base": delta, "test_accuracy": ordered["accuracy"], "test_macro_f1": ordered["macro_f1"], "test_loss": ordered["loss"]})
        paired_rows.append({"seed": seed, "comparison": f"{selection['selected_condition']}_minus_what_only", "delta_test_balanced_accuracy": delta, "delta_test_macro_f1": float(ordered["macro_f1"]) - float(base_test["macro_f1"]), "delta_test_loss": float(ordered["loss"]) - float(base_test["loss"])})
        for item in p["ablations"]:
            m = item["test"]; ablation_rows.append({"seed": seed, "ablation": item["ablation"], "replicate": item["replicate"], "test_balanced_accuracy": m["balanced_accuracy"], "test_accuracy": m["accuracy"], "test_macro_f1": m["macro_f1"], "test_loss": m["loss"], "max_abs_logit_delta_vs_base": m["max_abs_logit_delta_vs_base"]})
        residual_rows.append({"seed": seed, **ordered["residual_ratio"]}); rescue_rows.append({"seed": seed, **ordered["rescue_harm"]})
        for row in ordered["prefix"]:
            prefix_rows.extend([{"seed": seed, "prefix_fraction": row["progress"], "model": "what_only", "balanced_accuracy": row["base_balanced_accuracy"]}, {"seed": seed, "prefix_fraction": row["progress"], "model": str(selection["selected_condition"]), "balanced_accuracy": row["combined_balanced_accuracy"]}])
        a = ordered["when_source_activity"]
        gate_rows.extend([{"seed": seed, "metric": "when_ordered_firing_fraction", "bank": None, "phase_bin": None, "value": a["ordered_firing_fraction"]}, {"seed": seed, "metric": "when_reset_firing_fraction", "bank": None, "phase_bin": None, "value": a["reset_firing_fraction"]}])
        gate = ordered.get("gate")
        if gate:
            gate_rows.extend([{"seed": seed, "metric": "gate_entropy_mean", "bank": None, "phase_bin": None, "value": gate["entropy_mean"]}, {"seed": seed, "metric": "gate_temporal_variation_mean", "bank": None, "phase_bin": None, "value": gate["temporal_variation_mean"]}])
            for bank, value in enumerate(gate["utilization"]): gate_rows.append({"seed": seed, "metric": "bank_utilization", "bank": bank, "phase_bin": None, "value": value})
            for phase_bin, values in enumerate(gate["relative_phase_centered_gate"]):
                for bank, value in enumerate(values): gate_rows.append({"seed": seed, "metric": "centered_gate_by_relative_phase", "bank": bank, "phase_bin": phase_bin, "value": value})
    outputs = {name: root / filename for name, filename in {"screen_runs": "screen_runs.csv", "screen_paired_deltas": "screen_paired_deltas.csv", "refine_runs": "refine_runs.csv", "final_runs": "final_runs.csv", "final_paired_deltas": "final_paired_deltas.csv", "ablation_runs": "ablation_runs.csv", "gate_activity": "gate_activity.csv", "residual_metrics": "residual_metrics.csv", "rescue_harm": "rescue_harm.csv", "prefix_ba": "prefix_ba.csv", "selection": "selection.json", "manifest": "manifest.json"}.items()}
    for required in (outputs["screen_runs"], outputs["screen_paired_deltas"], outputs["refine_runs"], outputs["selection"]):
        if not required.exists(): raise FileNotFoundError(f"Finalizer will not regenerate missing upstream artifact: {required}")
    for name, rows in (("final_runs", final_rows), ("final_paired_deltas", paired_rows), ("ablation_runs", ablation_rows), ("gate_activity", gate_rows), ("residual_metrics", residual_rows), ("rescue_harm", rescue_rows), ("prefix_ba", prefix_rows)):
        pd.DataFrame(rows).to_csv(outputs[name], index=False)
    _save_json(outputs["manifest"], {"experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION, "seeds": list(SEEDS), "screen_conditions": list(SCREEN_CONDITIONS), "candidate_conditions": list(CANDIDATE_CONDITIONS), "selection": selection, "only_validation_selected": True, "test_not_used_for_selection": True, "final_test_opened_only_after_selection_json": True, "final_ablations": list(FINAL_ABLATIONS), "aggregation_policy": "finalizer aggregates existing artifacts only"})
    return outputs


def _config_from_args(args: argparse.Namespace) -> Config:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    return Config(repo_root, results_dir(repo_root), args.device, args.epochs, args.batch_size, args.threads)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 5.4.2 phase-conditioned readout mechanism search"); sub = parser.add_subparsers(dest="command", required=True)
    def common(p):
        p.add_argument("--repo-root", default=None); p.add_argument("--device", default="cpu"); p.add_argument("--threads", type=int, default=1); p.add_argument("--batch-size", type=int, default=BATCH_SIZE); p.add_argument("--epochs", type=int, default=EPOCHS); p.add_argument("--force", action="store_true")
    for name in ("prepare-source", "screen-run-one", "refine-run-one", "final-run-one"):
        p = sub.add_parser(name); common(p); p.add_argument("--array-task-id", type=int, required=True)
    for name in ("screen-finalize", "refine-task-count", "refine-finalize", "finalize"):
        p = sub.add_parser(name); p.add_argument("--repo-root", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command in ("screen-finalize", "refine-task-count", "refine-finalize", "finalize"):
        root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
        if args.command == "screen-finalize": outputs = finalize_screen(root)
        elif args.command == "refine-task-count": print(len(refine_specs(results_dir(root)))); return
        elif args.command == "refine-finalize": outputs = finalize_refine(root)
        else: outputs = finalize_experiment(root)
        for name, path in outputs.items(): print(f"{name}: {path}")
        return
    config = _config_from_args(args); data = base.prepare_data(config.repo_root); task = int(args.array_task_id)
    if args.command == "prepare-source": print(prepare_source_seed(SEEDS[task], data, config, args.force)); return
    if args.command == "screen-run-one": print(json.dumps(run_validation_one(screen_specs()[task], data, config, args.force), indent=2)); return
    if args.command == "refine-run-one": print(json.dumps(run_validation_one(refine_specs(config.results_dir)[task], data, config, args.force), indent=2)); return
    if args.command == "final-run-one": print(json.dumps(evaluate_final_seed(SEEDS[task], data, config, args.force), indent=2)); return


if __name__ == "__main__":
    main()
