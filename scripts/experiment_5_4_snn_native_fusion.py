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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from snntorch import surrogate

from scripts import experiment_5_3_2_3_ff_before_rsnn as exp5323


parent = exp5323.parent
base = exp5323.base
exp52 = exp5323.exp52

EXPERIMENT_ID = "experiment_5_4_snn_native_fusion"
PROTOCOL_VERSION = "snn_native_what_when_fusion_v1"
SEEDS = exp5323.SEEDS
WHAT_WIDTH = exp5323.LOCAL_WIDTH
WHEN_WIDTH = exp5323.RSNN_WIDTH
FUSION_WIDTH = 128
SHORT_SHIFT = exp5323.SHORT_SHIFT
THRESHOLD = exp5323.THRESHOLD
RESET = exp5323.RESET
SURROGATE_SLOPE = exp5323.SURROGATE_SLOPE
EPOCHS = exp5323.EPOCHS
BATCH_SIZE = exp5323.BATCH_SIZE
LR = exp5323.LR
WEIGHT_DECAY = exp5323.WEIGHT_DECAY
SHUFFLE_REPLICATES = 5

WHAT_ONLY = "what_only_lif"
WHEN_ONLY = "when_only_lif"
WHAT_ELAPSED = "what_elapsed_lif"
WHAT_RESETWHEN = "what_resetwhen_lif"
WHAT_WHEN_LINEAR = "what_when_linear"
WHAT_WHEN_FUSION = "what_when_fusion_lif"
CONDITIONS = (
    WHAT_ONLY,
    WHEN_ONLY,
    WHAT_ELAPSED,
    WHAT_RESETWHEN,
    WHAT_WHEN_LINEAR,
    WHAT_WHEN_FUSION,
)
MAIN_CONDITION = WHAT_WHEN_FUSION
LINEAR_CONDITION = WHAT_WHEN_LINEAR
SOURCE_WHEN_CONDITION = exp5323.FF128_CONDITION


@dataclass(frozen=True)
class RunSpec:
    condition: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.condition}__seed{self.seed}"

    @property
    def fusion_mode(self) -> str:
        return "linear" if self.condition == LINEAR_CONDITION else "lif"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp5323.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def fusion_cache_path(root: Path, seed: int) -> Path:
    return root / "fusion_inputs" / f"seed{seed}.npz"


def fusion_cache_meta_path(root: Path, seed: int) -> Path:
    return root / "fusion_inputs" / f"seed{seed}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_specs() -> list[RunSpec]:
    return [RunSpec(condition, seed) for condition in CONDITIONS for seed in SEEDS]


def _validate_spec(spec: RunSpec) -> None:
    if spec.condition not in CONDITIONS:
        raise ValueError(f"Unknown Exp5.4 condition: {spec.condition}")
    if spec.seed not in SEEDS:
        raise ValueError(f"Unknown Exp5.4 seed: {spec.seed}")


def _source_config(config: Config) -> exp5323.Config:
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


def _labels_lengths(data: base.Data, split: str) -> tuple[np.ndarray, np.ndarray]:
    if split == "train":
        return data.ytr, data.ltr
    if split == "val":
        return data.yva, data.lva
    if split == "test":
        return data.yte, data.lte
    raise ValueError(f"Unknown split: {split}")


def _cache_key(prefix: str, split: str) -> str:
    return f"{prefix}_{split}"


def prepare_fusion_seed(
    seed: int,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Unknown Exp5.4 seed: {seed}")
    destination = fusion_cache_path(config.results_dir, seed)
    meta_path = fusion_cache_meta_path(config.results_dir, seed)
    if destination.exists() and meta_path.exists() and not force:
        load_fusion_cache(seed, data, config)
        return destination

    source_config = _source_config(config)
    exp5323.prepare_local_seed(seed, data, source_config, force=False)
    source_model, source_payload = exp5323.load_model(_source_spec(seed), data, source_config)
    source_model.eval()
    local_cache = parent.load_local_cache(
        seed,
        data,
        exp5323._parent_config(source_config),
    )

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    arrays: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for split, (what_np, lengths_np) in parent._partitions(data, local_cache).items():
            loader = DataLoader(
                TensorDataset(
                    torch.tensor(what_np, dtype=torch.float32),
                    torch.tensor(lengths_np, dtype=torch.long),
                ),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
            )
            ordered_parts: list[np.ndarray] = []
            reset_parts: list[np.ndarray] = []
            for what, lengths in loader:
                what = what.to(device)
                lengths = lengths.to(device)
                ordered = source_model.forward_trajectory(
                    what,
                    lengths,
                    reset_state_each_step=False,
                ).spikes
                reset = source_model.forward_trajectory(
                    what,
                    lengths,
                    reset_state_each_step=True,
                ).spikes
                ordered_parts.append(ordered.cpu().numpy().astype(np.uint8))
                reset_parts.append(reset.cpu().numpy().astype(np.uint8))

            arrays[_cache_key("what", split)] = np.asarray(what_np, dtype=np.uint8)
            arrays[_cache_key("when_ordered", split)] = np.concatenate(ordered_parts, axis=0)
            arrays[_cache_key("when_reset", split)] = np.concatenate(reset_parts, axis=0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    train_max_elapsed_seconds = float((int(np.max(data.ltr)) - 1) / float(data.fs))
    source_checkpoint = exp5323.checkpoint_path(source_config.results_dir, _source_spec(seed))
    _save_json(
        meta_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "source_experiment_id": exp5323.EXPERIMENT_ID,
            "source_protocol_version": exp5323.PROTOCOL_VERSION,
            "source_condition": SOURCE_WHEN_CONDITION,
            "source_checkpoint": str(source_checkpoint.relative_to(config.repo_root)),
            "source_best_epoch": source_payload["result"]["best_epoch"],
            "what_width": WHAT_WIDTH,
            "when_width": WHEN_WIDTH,
            "what_dtype": "uint8 binary frozen Local-SNN L2 spikes",
            "when_dtype": "uint8 binary frozen FF-SNN128 -> RSNN64 output spikes",
            "ordered_when": "normal causal frozen WHEN trajectory",
            "reset_when": "FF and RSNN state reset before every timestep",
            "train_max_elapsed_seconds": train_max_elapsed_seconds,
            "elapsed_basis_policy": (
                "64 fixed RBFs over absolute t/fs, centers from 0 to train-set max elapsed; "
                "test elapsed is clipped to that train-derived range; final duration is never used"
            ),
            "split_seed": base.SPLIT_SEED,
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "sampling_rate_hz": float(data.fs),
        },
    )
    load_fusion_cache(seed, data, config)
    return destination


def load_fusion_cache(
    seed: int,
    data: base.Data,
    config: Config,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = fusion_cache_path(config.results_dir, seed)
    meta_path = fusion_cache_meta_path(config.results_dir, seed)
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing Exp5.4 fusion cache for seed {seed}; run prepare-fusion first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("experiment_id") != EXPERIMENT_ID
        or meta.get("protocol_version") != PROTOCOL_VERSION
        or meta.get("seed") != seed
        or meta.get("source_experiment_id") != exp5323.EXPERIMENT_ID
        or meta.get("source_protocol_version") != exp5323.PROTOCOL_VERSION
        or meta.get("source_condition") != SOURCE_WHEN_CONDITION
    ):
        raise ValueError(f"Exp5.4 fusion cache identity mismatch: {meta_path}")

    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    for split in ("train", "val", "test"):
        labels, lengths = _labels_lengths(data, split)
        expected_what = (len(labels), data.T, WHAT_WIDTH)
        expected_when = (len(labels), data.T, WHEN_WIDTH)
        for prefix, shape in (
            ("what", expected_what),
            ("when_ordered", expected_when),
            ("when_reset", expected_when),
        ):
            key = _cache_key(prefix, split)
            if key not in arrays or arrays[key].shape != shape:
                actual = arrays[key].shape if key in arrays else None
                raise ValueError(f"Fusion cache {key} shape mismatch: {actual} != {shape}")
        if np.any(lengths <= 0) or np.any(lengths > data.T):
            raise ValueError(f"Invalid valid lengths in {split}")
    return arrays, meta


def _loader(
    arrays: dict[str, np.ndarray],
    data: base.Data,
    split: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    labels, lengths = _labels_lengths(data, split)
    dataset = TensorDataset(
        torch.tensor(arrays[_cache_key("what", split)], dtype=torch.float32),
        torch.tensor(arrays[_cache_key("when_ordered", split)], dtype=torch.float32),
        torch.tensor(arrays[_cache_key("when_reset", split)], dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
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


def _loader_seed(seed: int, split: str) -> int:
    return base.dseed(seed, EXPERIMENT_ID, split, "loader")


def _make_loaders(
    arrays: dict[str, np.ndarray],
    data: base.Data,
    spec: RunSpec,
    config: Config,
    train_shuffle: bool,
) -> dict[str, DataLoader]:
    return {
        split: _loader(
            arrays,
            data,
            split,
            config.batch_size,
            train_shuffle if split == "train" else False,
            _loader_seed(spec.seed, split),
        )
        for split in ("train", "val", "test")
    }


def elapsed_context(
    batch_size: int,
    steps: int,
    fs: float,
    train_max_elapsed_seconds: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if train_max_elapsed_seconds <= 0.0:
        raise ValueError("train_max_elapsed_seconds must be positive")
    elapsed = torch.arange(steps, device=device, dtype=dtype) / float(fs)
    elapsed = elapsed.clamp(max=float(train_max_elapsed_seconds))
    centers = torch.linspace(
        0.0,
        float(train_max_elapsed_seconds),
        WHEN_WIDTH,
        device=device,
        dtype=dtype,
    )
    spacing = float(train_max_elapsed_seconds) / max(WHEN_WIDTH - 1, 1)
    sigma = max(1.5 * spacing, 1.0 / float(fs))
    basis = torch.exp(-0.5 * ((elapsed[:, None] - centers[None, :]) / sigma) ** 2)
    return basis.unsqueeze(0).expand(batch_size, -1, -1)


class FusionClassifier(nn.Module):
    """Fixed-synapse WHAT/WHEN fusion with either LIF or linear interaction control."""

    def __init__(self, n_classes: int, fs: float, fusion_mode: str) -> None:
        super().__init__()
        if fusion_mode not in ("lif", "linear"):
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        self.fusion_mode = fusion_mode
        self.what_projection = nn.Linear(WHAT_WIDTH, FUSION_WIDTH, bias=False)
        self.when_projection = nn.Linear(WHEN_WIDTH, FUSION_WIDTH, bias=False)
        self.output_projection = nn.Linear(FUSION_WIDTH, self.n_classes, bias=False)
        self.class_bias = nn.Parameter(torch.zeros(self.n_classes))
        decay = parent.decay_from_shift(SHORT_SHIFT)
        self.register_buffer("alpha_vector", torch.full((FUSION_WIDTH,), decay, dtype=torch.float32))
        self.register_buffer("beta_vector", torch.full((FUSION_WIDTH,), decay, dtype=torch.float32))
        self.spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

    @staticmethod
    def _reset_membrane(membrane: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        return exp5323.FrontEndWhenNet._reset_membrane(membrane, spike)

    def forward(
        self,
        what: torch.Tensor,
        context: torch.Tensor,
        lengths: torch.Tensor,
        return_fusion_trajectory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if what.ndim != 3 or context.ndim != 3:
            raise ValueError("WHAT and context must have shape [batch, time, channels]")
        if what.shape[:2] != context.shape[:2]:
            raise ValueError("WHAT/context batch-time dimensions must match")
        if what.shape[-1] != WHAT_WIDTH or context.shape[-1] != WHEN_WIDTH:
            raise ValueError("Unexpected WHAT/WHEN width")
        if torch.any(lengths <= 0) or torch.any(lengths > what.shape[1]):
            raise ValueError("Invalid valid lengths")

        batch = what.shape[0]
        synaptic = torch.zeros((batch, FUSION_WIDTH), dtype=what.dtype, device=what.device)
        membrane = torch.zeros_like(synaptic)
        accumulator = torch.zeros((batch, self.n_classes), dtype=what.dtype, device=what.device)
        trajectory: list[torch.Tensor] = []

        for timestep in range(what.shape[1]):
            valid = (timestep < lengths).unsqueeze(1)
            current = self.what_projection(what[:, timestep]) + self.when_projection(context[:, timestep])
            if self.fusion_mode == "linear":
                fusion_t = current * valid.to(current.dtype)
            else:
                next_synaptic = self.alpha_vector.to(current) * synaptic + current
                next_membrane = self.beta_vector.to(current) * membrane + next_synaptic
                next_spike = self.spike_grad(next_membrane - THRESHOLD)
                next_membrane = self._reset_membrane(next_membrane, next_spike)
                synaptic = torch.where(valid, next_synaptic, synaptic)
                membrane = torch.where(valid, next_membrane, membrane)
                fusion_t = next_spike * valid.to(next_spike.dtype)

            evidence_t = self.output_projection(fusion_t)
            accumulator = accumulator + evidence_t
            if return_fusion_trajectory:
                trajectory.append(fusion_t)

        logits = accumulator + self.class_bias
        fusion_trajectory = torch.stack(trajectory, dim=1) if return_fusion_trajectory else None
        return logits, fusion_trajectory


def _initialize_model(spec: RunSpec, data: base.Data, device: torch.device) -> FusionClassifier:
    _validate_spec(spec)
    base.seed_all(base.dseed(spec.seed, EXPERIMENT_ID, "paired_fusion", "constructor"))
    model = FusionClassifier(len(data.labels), data.fs, spec.fusion_mode).to(device)
    for name, module in (
        ("what_projection", model.what_projection),
        ("when_projection", model.when_projection),
        ("output_projection", model.output_projection),
    ):
        base.seed_all(base.dseed(spec.seed, EXPERIMENT_ID, "paired_fusion", name))
        module.reset_parameters()
    with torch.no_grad():
        model.class_bias.zero_()
    return model


def parameter_counts(model: FusionClassifier) -> dict[str, int]:
    return {
        "what_projection": int(sum(p.numel() for p in model.what_projection.parameters())),
        "when_projection": int(sum(p.numel() for p in model.when_projection.parameters())),
        "output_projection": int(sum(p.numel() for p in model.output_projection.parameters())),
        "class_bias": int(model.class_bias.numel()),
        "trainable_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def _select_inputs(
    spec: RunSpec,
    what: torch.Tensor,
    when_ordered: torch.Tensor,
    when_reset: torch.Tensor,
    fs: float,
    train_max_elapsed_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.condition == WHAT_ONLY:
        return what, torch.zeros_like(when_ordered)
    if spec.condition == WHEN_ONLY:
        return torch.zeros_like(what), when_ordered
    if spec.condition == WHAT_ELAPSED:
        return what, elapsed_context(
            what.shape[0],
            what.shape[1],
            fs,
            train_max_elapsed_seconds,
            what.device,
            what.dtype,
        )
    if spec.condition == WHAT_RESETWHEN:
        return what, when_reset
    if spec.condition in (WHAT_WHEN_LINEAR, WHAT_WHEN_FUSION):
        return what, when_ordered
    raise ValueError(f"Unknown Exp5.4 condition: {spec.condition}")


def _shuffle_valid_context(
    context: torch.Tensor,
    lengths: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    out = context.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for row, length_tensor in enumerate(lengths.detach().cpu()):
        length = int(length_tensor.item())
        if length <= 1:
            continue
        order = torch.randperm(length, generator=generator)
        out[row, :length] = context[row, order.to(context.device)]
    return out


def _circular_shift_valid_context(context: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    out = context.clone()
    for row, length_tensor in enumerate(lengths.detach().cpu()):
        length = int(length_tensor.item())
        if length <= 1:
            continue
        shift = max(1, length // 3)
        out[row, :length] = torch.roll(context[row, :length], shifts=shift, dims=0)
    return out


def _evaluate(
    model: FusionClassifier,
    loader: DataLoader,
    spec: RunSpec,
    device: torch.device,
    fs: float,
    train_max_elapsed_seconds: float,
    context_ablation: str = "ordered",
    replicate: int = 0,
) -> dict[str, float | int | None]:
    if context_ablation not in ("ordered", "when_zero", "when_shuffle", "when_circular_shift"):
        raise ValueError(f"Unknown context ablation: {context_ablation}")
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    spike_sum = 0.0
    valid_neuron_steps = 0.0

    with torch.no_grad():
        for batch_index, (what, when_ordered, when_reset, labels, lengths) in enumerate(loader):
            what = what.to(device)
            when_ordered = when_ordered.to(device)
            when_reset = when_reset.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            selected_what, context = _select_inputs(
                spec,
                what,
                when_ordered,
                when_reset,
                fs,
                train_max_elapsed_seconds,
            )
            if context_ablation != "ordered":
                if spec.condition != MAIN_CONDITION:
                    raise ValueError("Context ablations are defined only for the trained main fusion model")
                if context_ablation == "when_zero":
                    context = torch.zeros_like(context)
                elif context_ablation == "when_shuffle":
                    context = _shuffle_valid_context(
                        context,
                        lengths,
                        base.dseed(spec.seed, EXPERIMENT_ID, context_ablation, replicate, batch_index),
                    )
                elif context_ablation == "when_circular_shift":
                    context = _circular_shift_valid_context(context, lengths)

            logits, fusion = model(
                selected_what,
                context,
                lengths,
                return_fusion_trajectory=model.fusion_mode == "lif",
            )
            loss = F.cross_entropy(logits, labels)
            pred = logits.argmax(dim=1)
            n = len(labels)
            loss_sum += float(loss.item()) * n
            n_total += n
            y_true.append(labels.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            if fusion is not None:
                valid = base.mask(lengths, fusion.shape[1]).to(fusion.dtype).unsqueeze(-1)
                spike_sum += float((fusion * valid).sum().item())
                valid_neuron_steps += float(valid.sum().item()) * FUSION_WIDTH

    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    firing_rate = spike_sum / valid_neuron_steps if valid_neuron_steps > 0 else None
    return {
        "loss": loss_sum / max(n_total, 1),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro")),
        "n_samples": int(n_total),
        "fusion_firing_fraction": firing_rate,
    }


def _provenance(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    meta: dict[str, object],
    model: FusionClassifier,
) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "condition": spec.condition,
        "seed": spec.seed,
        "split_seed": base.SPLIT_SEED,
        "fusion_mode": spec.fusion_mode,
        "what_width": WHAT_WIDTH,
        "when_width": WHEN_WIDTH,
        "fusion_width": FUSION_WIDTH,
        "source_when_experiment": exp5323.EXPERIMENT_ID,
        "source_when_protocol": exp5323.PROTOCOL_VERSION,
        "source_when_condition": SOURCE_WHEN_CONDITION,
        "source_when_checkpoint": meta["source_checkpoint"],
        "what_frozen": True,
        "when_frozen": True,
        "fusion_recurrent": False,
        "fusion_shift_syn": SHORT_SHIFT,
        "fusion_shift_mem": SHORT_SHIFT,
        "fusion_tau_syn_ms": parent.tau_ms_from_shift(SHORT_SHIFT, data.fs),
        "fusion_tau_mem_ms": parent.tau_ms_from_shift(SHORT_SHIFT, data.fs),
        "threshold": THRESHOLD,
        "reset": RESET,
        "surrogate_slope": SURROGATE_SLOPE,
        "output_projection_bias": False,
        "class_bias_policy": "single class bias added once after whole-sequence accumulation",
        "objective": "final whole-sequence cross entropy only",
        "readout": "sum per-timestep class evidence over valid timesteps; no temporal pooling before fusion",
        "elapsed_baseline": meta["elapsed_basis_policy"],
        "causality_contract": (
            "final duration is never a model/fusion/context input; valid length is used only to stop padded "
            "state/evidence updates and to choose the endpoint"
        ),
        "checkpoint_selection": "maximum validation balanced accuracy; tie lower validation cross entropy",
        "initialization": (
            "all six conditions share paired per-seed initial weights for W_what, W_when, W_out and class bias"
        ),
        "minibatch_order": "paired across all conditions within seed",
        "parameter_counts": parameter_counts(model),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "sampling_rate_hz": float(data.fs),
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
    _validate_spec(spec)
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    arrays, meta = load_fusion_cache(spec.seed, data, config)
    loaders_train = _make_loaders(arrays, data, spec, config, train_shuffle=True)
    loaders_eval = _make_loaders(arrays, data, spec, config, train_shuffle=False)
    train_max_elapsed_seconds = float(meta["train_max_elapsed_seconds"])

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _initialize_model(spec, data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val_ba = -math.inf
    best_val_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        loss_sum = 0.0
        n_total = 0
        for what, when_ordered, when_reset, labels, lengths in loaders_train["train"]:
            what = what.to(device)
            when_ordered = when_ordered.to(device)
            when_reset = when_reset.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            selected_what, context = _select_inputs(
                spec,
                what,
                when_ordered,
                when_reset,
                data.fs,
                train_max_elapsed_seconds,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(selected_what, context, lengths)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            pred = logits.detach().argmax(dim=1)
            n = len(labels)
            loss_sum += float(loss.item()) * n
            n_total += n
            y_true.append(labels.detach().cpu().numpy())
            y_pred.append(pred.cpu().numpy())

        train_true = np.concatenate(y_true)
        train_pred = np.concatenate(y_pred)
        train_loss = loss_sum / max(n_total, 1)
        train_ba = float(balanced_accuracy_score(train_true, train_pred))
        val_metrics = _evaluate(
            model,
            loaders_eval["val"],
            spec,
            device,
            data.fs,
            train_max_elapsed_seconds,
        )
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_balanced_accuracy": train_ba,
                "val_loss": val_loss,
                "val_balanced_accuracy": val_ba,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
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
        split: _evaluate(
            model,
            loader,
            spec,
            device,
            data.fs,
            train_max_elapsed_seconds,
        )
        for split, loader in loaders_eval.items()
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "provenance": _provenance(spec, data, config, meta, model),
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
) -> tuple[FusionClassifier, dict[str, object]]:
    _validate_spec(spec)
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.4 checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("spec") != asdict(spec)
    ):
        raise ValueError(f"Wrong Exp5.4 checkpoint identity: {path}")
    model = FusionClassifier(len(data.labels), data.fs, spec.fusion_mode).to(torch.device(config.device))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def evaluate_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    _validate_spec(spec)
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    arrays, meta = load_fusion_cache(spec.seed, data, config)
    loaders = _make_loaders(arrays, data, spec, config, train_shuffle=False)
    train_max_elapsed_seconds = float(meta["train_max_elapsed_seconds"])
    model, checkpoint = load_model(spec, data, config)
    device = torch.device(config.device)
    native = {
        split: _evaluate(
            model,
            loader,
            spec,
            device,
            data.fs,
            train_max_elapsed_seconds,
        )
        for split, loader in loaders.items()
    }

    ablations: list[dict[str, object]] = []
    if spec.condition == MAIN_CONDITION:
        ablations.append(
            {
                "ablation": "ordered",
                "replicate": 0,
                "test": native["test"],
            }
        )
        ablations.append(
            {
                "ablation": "when_zero",
                "replicate": 0,
                "test": _evaluate(
                    model,
                    loaders["test"],
                    spec,
                    device,
                    data.fs,
                    train_max_elapsed_seconds,
                    context_ablation="when_zero",
                ),
            }
        )
        for replicate in range(SHUFFLE_REPLICATES):
            ablations.append(
                {
                    "ablation": "when_shuffle",
                    "replicate": replicate,
                    "test": _evaluate(
                        model,
                        loaders["test"],
                        spec,
                        device,
                        data.fs,
                        train_max_elapsed_seconds,
                        context_ablation="when_shuffle",
                        replicate=replicate,
                    ),
                }
            )
        ablations.append(
            {
                "ablation": "when_circular_shift",
                "replicate": 0,
                "test": _evaluate(
                    model,
                    loaders["test"],
                    spec,
                    device,
                    data.fs,
                    train_max_elapsed_seconds,
                    context_ablation="when_circular_shift",
                ),
            }
        )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "condition": spec.condition,
        "fusion_mode": spec.fusion_mode,
        "seed": spec.seed,
        "parameter_counts": parameter_counts(model),
        "best_epoch": checkpoint["result"]["best_epoch"],
        "provenance": checkpoint["provenance"],
        "native": native,
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


def _load_eval(root: Path, spec: RunSpec) -> dict[str, object]:
    path = evaluation_path(root, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.4 evaluation: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("spec") != asdict(spec)
    ):
        raise ValueError(f"Exp5.4 evaluation identity mismatch: {path}")
    return payload


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    specs = run_specs()
    if len(specs) != 30 or len({spec.key for spec in specs}) != 30:
        raise RuntimeError("Exp5.4 must contain exactly 30 unique condition-seed runs")
    for condition in CONDITIONS:
        if sum(spec.condition == condition for spec in specs) != len(SEEDS):
            raise RuntimeError(f"Condition {condition} must have exactly five seeds")

    run_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    activity_rows: list[dict[str, object]] = []
    history_parts: list[pd.DataFrame] = []
    for spec in specs:
        payload = _load_eval(root, spec)
        row: dict[str, object] = {
            "condition": spec.condition,
            "fusion_mode": spec.fusion_mode,
            "seed": spec.seed,
            "parameter_count": payload["parameter_counts"]["trainable_total"],
            "best_epoch": payload["best_epoch"],
        }
        for split in ("train", "val", "test"):
            metrics = payload["native"][split]
            for key in ("loss", "balanced_accuracy", "accuracy", "macro_f1"):
                row[f"{split}_{key}"] = metrics[key]
            activity_rows.append(
                {
                    "condition": spec.condition,
                    "fusion_mode": spec.fusion_mode,
                    "seed": spec.seed,
                    "split": split,
                    "fusion_firing_fraction": metrics["fusion_firing_fraction"],
                }
            )
        run_rows.append(row)

        hpath = history_path(root, spec)
        if not hpath.exists():
            raise FileNotFoundError(f"Missing history: {hpath}")
        history = pd.read_csv(hpath)
        history.insert(0, "seed", spec.seed)
        history.insert(0, "condition", spec.condition)
        history_parts.append(history)
        for ablation in payload["ablations"]:
            test = ablation["test"]
            ablation_rows.append(
                {
                    "condition": spec.condition,
                    "seed": spec.seed,
                    "ablation": ablation["ablation"],
                    "replicate": ablation["replicate"],
                    "test_balanced_accuracy": test["balanced_accuracy"],
                    "test_accuracy": test["accuracy"],
                    "test_macro_f1": test["macro_f1"],
                    "test_loss": test["loss"],
                    "fusion_firing_fraction": test["fusion_firing_fraction"],
                }
            )

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "runs": root / "runs.csv",
        "histories": root / "histories.csv",
        "ablation_runs": root / "ablation_runs.csv",
        "activity_runs": root / "activity_runs.csv",
        "paired_deltas": root / "paired_deltas.csv",
        "manifest": root / "manifest.json",
    }
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(outputs["runs"], index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(outputs["histories"], index=False)
    pd.DataFrame(ablation_rows).to_csv(outputs["ablation_runs"], index=False)
    pd.DataFrame(activity_rows).to_csv(outputs["activity_runs"], index=False)

    main = runs_df[runs_df["condition"] == MAIN_CONDITION].set_index("seed")
    delta_rows: list[dict[str, object]] = []
    for comparator in CONDITIONS:
        if comparator == MAIN_CONDITION:
            continue
        control = runs_df[runs_df["condition"] == comparator].set_index("seed")
        for seed in SEEDS:
            delta_rows.append(
                {
                    "comparison": f"{MAIN_CONDITION}_minus_{comparator}",
                    "comparator": comparator,
                    "seed": seed,
                    "delta_test_balanced_accuracy": (
                        main.loc[seed, "test_balanced_accuracy"] - control.loc[seed, "test_balanced_accuracy"]
                    ),
                    "delta_test_macro_f1": main.loc[seed, "test_macro_f1"] - control.loc[seed, "test_macro_f1"],
                    "delta_test_loss": main.loc[seed, "test_loss"] - control.loc[seed, "test_loss"],
                }
            )
    pd.DataFrame(delta_rows).to_csv(outputs["paired_deltas"], index=False)

    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "scientific_question": (
                "whether frozen history-dependent WHEN spikes improve final letter classification by "
                "contextualizing frozen WHAT spikes in a fixed-synapse Fusion LIF layer"
            ),
            "conditions": list(CONDITIONS),
            "main_condition": MAIN_CONDITION,
            "seeds": list(SEEDS),
            "expected_train_runs": len(specs),
            "run_order": (
                "condition-major, five seeds per condition: what_only_lif, when_only_lif, "
                "what_elapsed_lif, what_resetwhen_lif, what_when_linear, what_when_fusion_lif"
            ),
            "architecture_contract": (
                "frozen WHAT128 spikes + frozen context64 -> fixed W_what/W_when -> short-memory non-recurrent "
                "Fusion128 -> fixed W_out -> per-timestep 12D evidence -> non-leaky whole-sequence accumulator"
            ),
            "source_when": f"{exp5323.EXPERIMENT_ID}/{exp5323.PROTOCOL_VERSION}/{SOURCE_WHEN_CONDITION}",
            "fusion_width": FUSION_WIDTH,
            "fusion_recurrent": False,
            "fusion_short_shift": SHORT_SHIFT,
            "objective": "final whole-sequence CE only",
            "output_bias_contract": "W_out has no bias; one class bias is added once after accumulation",
            "causality_contract": (
                "elapsed baseline uses absolute t/fs only; final duration is not an input to any condition; "
                "valid length only masks padded updates and selects the endpoint"
            ),
            "primary_comparisons": [
                f"{MAIN_CONDITION} vs {WHAT_ONLY}: does WHEN add task value beyond WHAT?",
                f"{MAIN_CONDITION} vs {WHAT_ELAPSED}: does learned WHEN beat a strong causal clock?",
                f"{MAIN_CONDITION} vs {WHAT_RESETWHEN}: does ordered temporal history matter?",
                f"{MAIN_CONDITION} vs {WHAT_WHEN_LINEAR}: does Fusion LIF threshold/state interaction matter?",
                f"{WHEN_ONLY}: how much letter identity remains in WHEN alone?",
            ],
            "test_time_main_ablations": ["when_zero", "when_shuffle", "when_circular_shift"],
            "winner_logic": (
                "the WHEN fusion hypothesis is strongest when the main condition beats WHAT-only, elapsed-time, "
                "reset-WHEN and parameter-matched linear fusion, and aligned WHEN degrades when zeroed/shuffled/shifted"
            ),
            "aggregation_policy": "finalizer aggregates existing 30 run artifacts only; it never trains or regenerates caches",
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
    parser = argparse.ArgumentParser(description="Experiment 5.4 SNN-native causal WHAT-WHEN fusion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo-root", type=str, default=None)
        subparser.add_argument("--device", type=str, default="cpu")
        subparser.add_argument("--threads", type=int, default=1)
        subparser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        subparser.add_argument("--epochs", type=int, default=EPOCHS)
        subparser.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare-fusion")
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
    if args.command == "prepare-fusion":
        task_id = int(args.array_task_id)
        if not 0 <= task_id < len(SEEDS):
            raise IndexError(f"prepare-fusion task {task_id} outside 0..{len(SEEDS)-1}")
        print(prepare_fusion_seed(SEEDS[task_id], data, config, force=args.force))
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
