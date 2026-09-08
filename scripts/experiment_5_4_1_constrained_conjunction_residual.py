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
from torch.utils.data import DataLoader
from snntorch import surrogate

from scripts import experiment_5_4_snn_native_fusion as exp54


base = exp54.base

EXPERIMENT_ID = "experiment_5_4_1_constrained_conjunction_residual"
PROTOCOL_VERSION = "direct_what_conjunction_residual_v2"
SEEDS = exp54.SEEDS
WHAT_WIDTH = exp54.WHAT_WIDTH
WHEN_WIDTH = exp54.WHEN_WIDTH
CONTEXT_WIDTH = 128
SELECTION_THRESHOLD = exp54.THRESHOLD
CONJUNCTION_THRESHOLD = 1.0
CONJUNCTION_GAIN = 0.75
SELECTOR_INIT_MAX = 0.05
SURROGATE_SLOPE = exp54.SURROGATE_SLOPE
EPOCHS = exp54.EPOCHS
BATCH_SIZE = exp54.BATCH_SIZE
LR = exp54.LR
WEIGHT_DECAY = exp54.WEIGHT_DECAY
SHUFFLE_REPLICATES = 5

BASE_CONDITION = "direct_what_wholecount"
CONDITION = "direct_what_plus_constrained_when_residual"
ABLATIONS = (
    "ordered",
    "reset_when",
    "when_zero",
    "when_shuffle",
    "when_circular_shift",
    "residual_what_zero",
)


@dataclass(frozen=True)
class RunSpec:
    seed: int

    @property
    def condition(self) -> str:
        return CONDITION

    @property
    def key(self) -> str:
        return f"{CONDITION}__seed{self.seed}"

    @property
    def base_key(self) -> str:
        return f"{BASE_CONDITION}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


@dataclass
class ResidualTrajectory:
    what_selector_spikes: torch.Tensor
    when_selector_spikes: torch.Tensor
    conjunction_spikes: torch.Tensor
    residual_logits: torch.Tensor


def find_repo_root(start: Path | None = None) -> Path:
    return exp54.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_reference_path(root: Path, seed: int) -> Path:
    return root / "source_references" / f"seed{seed}.json"


def base_checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "base_checkpoints" / f"{spec.base_key}.pt"


def base_history_path(root: Path, spec: RunSpec) -> Path:
    return root / "base_histories" / f"{spec.base_key}.csv"


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
    return [RunSpec(seed) for seed in SEEDS]


def _validate_spec(spec: RunSpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(f"Unknown Exp5.4.1 seed: {spec.seed}")


def _source_config(config: Config) -> exp54.Config:
    return exp54.Config(
        repo_root=config.repo_root,
        results_dir=exp54.results_dir(config.repo_root),
        device=config.device,
        epochs=exp54.EPOCHS,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def prepare_inputs_seed(
    seed: int,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Unknown Exp5.4.1 seed: {seed}")
    destination = source_reference_path(config.results_dir, seed)
    if destination.exists() and not force:
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if (
            payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("seed") != seed
        ):
            raise ValueError(f"Wrong Exp5.4.1 source reference identity: {destination}")
        return destination

    source_config = _source_config(config)
    exp54.prepare_fusion_seed(seed, data, source_config, force=False)
    arrays, cache_meta = exp54.load_fusion_cache(seed, data, source_config)
    del arrays
    source_cache = exp54.fusion_cache_path(source_config.results_dir, seed)
    source_cache_meta = exp54.fusion_cache_meta_path(source_config.results_dir, seed)
    _save_json(
        destination,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "source_experiment_id": exp54.EXPERIMENT_ID,
            "source_protocol_version": exp54.PROTOCOL_VERSION,
            "source_fusion_cache": str(source_cache.relative_to(config.repo_root)),
            "source_fusion_cache_meta": str(source_cache_meta.relative_to(config.repo_root)),
            "source_when_checkpoint": cache_meta["source_checkpoint"],
            "source_when_condition": cache_meta["source_condition"],
            "what_definition": "uint8 binary frozen Local-SNN L2 spikes; used directly with no extra Fusion-LIF",
            "when_definition": "uint8 binary frozen FF-SNN128 -> RSNN64 output spikes",
            "what_width": WHAT_WIDTH,
            "when_width": WHEN_WIDTH,
            "split_seed": base.SPLIT_SEED,
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "sampling_rate_hz": float(data.fs),
        },
    )
    return destination


def _source_reference(config: Config, seed: int) -> dict[str, object]:
    path = source_reference_path(config.results_dir, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.4.1 source reference: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("seed") != seed
        or payload.get("source_experiment_id") != exp54.EXPERIMENT_ID
        or payload.get("source_protocol_version") != exp54.PROTOCOL_VERSION
    ):
        raise ValueError(f"Exp5.4.1 source reference identity mismatch: {path}")
    return payload


def _make_loaders(
    arrays: dict[str, np.ndarray],
    data: base.Data,
    spec: RunSpec,
    config: Config,
    train_shuffle: bool,
    stage: str,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        loaders[split] = exp54._loader(
            arrays,
            data,
            split,
            config.batch_size,
            train_shuffle if split == "train" else False,
            base.dseed(spec.seed, EXPERIMENT_ID, stage, split, "loader"),
        )
    return loaders


class DirectWhatBase(nn.Module):
    """Direct fixed-synapse readout of frozen WHAT L2 spikes with no extra SNN layer."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.output_projection = nn.Linear(WHAT_WIDTH, self.n_classes, bias=False)
        self.class_bias = nn.Parameter(torch.zeros(self.n_classes))

    def forward(
        self,
        what: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_evidence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if what.ndim != 3 or what.shape[-1] != WHAT_WIDTH:
            raise ValueError("WHAT must have shape [batch, time, 128]")
        if torch.any(lengths <= 0) or torch.any(lengths > what.shape[1]):
            raise ValueError("Invalid valid lengths")
        valid = base.mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
        evidence = self.output_projection(what) * valid
        logits = evidence.sum(dim=1) + self.class_bias
        return logits, evidence if return_evidence else None


def _initialize_base(spec: RunSpec, data: base.Data, device: torch.device) -> DirectWhatBase:
    _validate_spec(spec)
    base.seed_all(base.dseed(spec.seed, EXPERIMENT_ID, "direct_what_base", "constructor"))
    model = DirectWhatBase(len(data.labels)).to(device)
    base.seed_all(base.dseed(spec.seed, EXPERIMENT_ID, "direct_what_base", "output_projection"))
    model.output_projection.reset_parameters()
    with torch.no_grad():
        model.class_bias.zero_()
    return model


def base_parameter_count(model: DirectWhatBase) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _classification_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    loss: float,
    n_samples: int,
) -> dict[str, float | int]:
    return {
        "loss": float(loss),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro")),
        "n_samples": int(n_samples),
    }


def _evaluate_base(
    model: DirectWhatBase,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for what, _when_ordered, _when_reset, labels, lengths in loader:
            what = what.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            logits, _ = model(what, lengths)
            loss = F.cross_entropy(logits, labels)
            pred = logits.argmax(dim=1)
            n = len(labels)
            loss_sum += float(loss.item()) * n
            n_total += n
            y_true.append(labels.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    return _classification_metrics(true, pred, loss_sum / max(n_total, 1), n_total)


def _is_better(val_ba: float, val_loss: float, best_ba: float, best_loss: float) -> bool:
    return val_ba > best_ba + 1e-12 or (
        abs(val_ba - best_ba) <= 1e-12 and val_loss < best_loss - 1e-12
    )


def train_base(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    _validate_spec(spec)
    destination = base_checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    _source_reference(config, spec.seed)
    source_config = _source_config(config)
    arrays, _ = exp54.load_fusion_cache(spec.seed, data, source_config)
    train_loaders = _make_loaders(arrays, data, spec, config, train_shuffle=True, stage="base")
    eval_loaders = _make_loaders(arrays, data, spec, config, train_shuffle=False, stage="base_eval")

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _initialize_base(spec, data, device)
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
        for what, _when_ordered, _when_reset, labels, lengths in train_loaders["train"]:
            what = what.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(what, lengths)
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
        val_metrics = _evaluate_base(model, eval_loaders["val"], device)
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
        if _is_better(val_ba, val_loss, best_val_ba, best_val_loss):
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"No direct WHAT base checkpoint selected for seed {spec.seed}")
    model.load_state_dict(best_state, strict=True)
    native = {
        split: _evaluate_base(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "stage": BASE_CONDITION,
            "spec": asdict(spec),
            "provenance": {
                "seed": spec.seed,
                "split_seed": base.SPLIT_SEED,
                "what_source": "frozen Local-SNN L2 spikes from Exp5.4 fusion cache",
                "architecture": "WHAT128 -> bias-free Linear128x12 -> valid-timestep evidence sum -> one final class bias",
                "extra_snn_between_what_and_base": False,
                "objective": "final whole-sequence cross entropy only",
                "causality_contract": "final duration is never a model input; valid length only masks padding and selects endpoint",
                "parameter_count": base_parameter_count(model),
            },
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
    hpath = base_history_path(config.results_dir, spec)
    hpath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hpath, index=False)
    return destination


def load_base(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[DirectWhatBase, dict[str, object]]:
    _validate_spec(spec)
    path = base_checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.4.1 direct WHAT base checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("stage") != BASE_CONDITION
        or payload.get("spec") != asdict(spec)
    ):
        raise ValueError(f"Wrong Exp5.4.1 base checkpoint identity: {path}")
    model = DirectWhatBase(len(data.labels)).to(torch.device(config.device))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


class ResidualConjunctionClassifier(nn.Module):
    """Frozen direct WHAT evidence plus a structurally conjunctive WHAT x WHEN correction."""

    def __init__(self, base_model: DirectWhatBase, n_classes: int) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.base_model = base_model
        self.base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.what_selector = nn.Linear(WHAT_WIDTH, CONTEXT_WIDTH, bias=False)
        self.when_selector = nn.Linear(WHEN_WIDTH, CONTEXT_WIDTH, bias=False)
        self.residual_output = nn.Linear(CONTEXT_WIDTH, self.n_classes, bias=False)
        self.spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

    def train(self, mode: bool = True) -> ResidualConjunctionClassifier:
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(
        self,
        what: torch.Tensor,
        when_context: torch.Tensor,
        lengths: torch.Tensor,
        *,
        zero_residual_what: bool = False,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, ResidualTrajectory | None]:
        if what.ndim != 3 or when_context.ndim != 3:
            raise ValueError("WHAT and WHEN must have shape [batch, time, channels]")
        if what.shape[:2] != when_context.shape[:2]:
            raise ValueError("WHAT/WHEN batch-time dimensions must match")
        if what.shape[-1] != WHAT_WIDTH or when_context.shape[-1] != WHEN_WIDTH:
            raise ValueError("Unexpected WHAT/WHEN width")
        if torch.any(lengths <= 0) or torch.any(lengths > what.shape[1]):
            raise ValueError("Invalid valid lengths")

        with torch.no_grad():
            base_logits, _ = self.base_model(what, lengths)
        residual_what = torch.zeros_like(what) if zero_residual_what else what
        what_selected = self.spike_grad(self.what_selector(residual_what) - SELECTION_THRESHOLD)
        when_selected = self.spike_grad(self.when_selector(when_context) - SELECTION_THRESHOLD)
        conjunction_current = CONJUNCTION_GAIN * (what_selected + when_selected)
        conjunction = self.spike_grad(conjunction_current - CONJUNCTION_THRESHOLD)

        valid = base.mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
        what_selected = what_selected * valid
        when_selected = when_selected * valid
        conjunction = conjunction * valid
        residual_step = self.residual_output(conjunction) * valid
        residual_logits = residual_step.sum(dim=1)
        logits = base_logits + residual_logits
        trajectory = None
        if return_trajectory:
            trajectory = ResidualTrajectory(
                what_selector_spikes=what_selected,
                when_selector_spikes=when_selected,
                conjunction_spikes=conjunction,
                residual_logits=residual_logits,
            )
        return logits, trajectory


def _initialize_residual(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[ResidualConjunctionClassifier, dict[str, object]]:
    base_model, base_payload = load_base(spec, data, config)
    model = ResidualConjunctionClassifier(base_model, len(data.labels)).to(torch.device(config.device))
    base.seed_all(base.dseed(spec.seed, EXPERIMENT_ID, "what_selector"))
    nn.init.uniform_(model.what_selector.weight, a=0.0, b=SELECTOR_INIT_MAX)
    base.seed_all(base.dseed(spec.seed, EXPERIMENT_ID, "when_selector"))
    nn.init.uniform_(model.when_selector.weight, a=0.0, b=SELECTOR_INIT_MAX)
    with torch.no_grad():
        model.residual_output.weight.zero_()
    return model, base_payload


def parameter_counts(model: ResidualConjunctionClassifier) -> dict[str, int]:
    return {
        "frozen_base": int(sum(p.numel() for p in model.base_model.parameters())),
        "what_selector": int(sum(p.numel() for p in model.what_selector.parameters())),
        "when_selector": int(sum(p.numel() for p in model.when_selector.parameters())),
        "residual_output": int(sum(p.numel() for p in model.residual_output.parameters())),
        "trainable_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "stored_total": int(sum(p.numel() for p in model.parameters())),
    }


def _shuffle_valid_context(context: torch.Tensor, lengths: torch.Tensor, seed: int) -> torch.Tensor:
    return exp54._shuffle_valid_context(context, lengths, seed)


def _circular_shift_valid_context(context: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return exp54._circular_shift_valid_context(context, lengths)


def _select_ablation(
    ablation: str,
    when_ordered: torch.Tensor,
    when_reset: torch.Tensor,
    lengths: torch.Tensor,
    seed: int,
    replicate: int,
    batch_index: int,
) -> tuple[torch.Tensor, bool]:
    if ablation == "ordered":
        return when_ordered, False
    if ablation == "reset_when":
        return when_reset, False
    if ablation == "when_zero":
        return torch.zeros_like(when_ordered), False
    if ablation == "when_shuffle":
        return (
            _shuffle_valid_context(
                when_ordered,
                lengths,
                base.dseed(seed, EXPERIMENT_ID, ablation, replicate, batch_index),
            ),
            False,
        )
    if ablation == "when_circular_shift":
        return _circular_shift_valid_context(when_ordered, lengths), False
    if ablation == "residual_what_zero":
        return when_ordered, True
    raise ValueError(f"Unknown Exp5.4.1 ablation: {ablation}")


def _evaluate_residual(
    model: ResidualConjunctionClassifier,
    loader: DataLoader,
    spec: RunSpec,
    device: torch.device,
    *,
    ablation: str = "ordered",
    replicate: int = 0,
) -> dict[str, float | int]:
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown Exp5.4.1 ablation: {ablation}")
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    what_selector_sum = 0.0
    when_selector_sum = 0.0
    conjunction_sum = 0.0
    valid_context_steps = 0.0
    residual_abs_sum = 0.0
    residual_abs_max = 0.0

    with torch.no_grad():
        for batch_index, (what, when_ordered, when_reset, labels, lengths) in enumerate(loader):
            what = what.to(device)
            when_ordered = when_ordered.to(device)
            when_reset = when_reset.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            context, zero_residual_what = _select_ablation(
                ablation,
                when_ordered,
                when_reset,
                lengths,
                spec.seed,
                replicate,
                batch_index,
            )
            logits, trajectory = model(
                what,
                context,
                lengths,
                zero_residual_what=zero_residual_what,
                return_trajectory=True,
            )
            if trajectory is None:
                raise RuntimeError("Missing residual trajectory during evaluation")
            loss = F.cross_entropy(logits, labels)
            pred = logits.argmax(dim=1)
            n = len(labels)
            loss_sum += float(loss.item()) * n
            n_total += n
            y_true.append(labels.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            valid = base.mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
            valid_context_steps += float(valid.sum().item()) * CONTEXT_WIDTH
            what_selector_sum += float((trajectory.what_selector_spikes * valid).sum().item())
            when_selector_sum += float((trajectory.when_selector_spikes * valid).sum().item())
            conjunction_sum += float((trajectory.conjunction_spikes * valid).sum().item())
            residual_abs_sum += float(trajectory.residual_logits.abs().sum().item())
            residual_abs_max = max(residual_abs_max, float(trajectory.residual_logits.abs().max().item()))

    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    metrics = _classification_metrics(true, pred, loss_sum / max(n_total, 1), n_total)
    denom = max(valid_context_steps, 1.0)
    metrics.update(
        {
            "what_selector_firing_fraction": what_selector_sum / denom,
            "when_selector_firing_fraction": when_selector_sum / denom,
            "conjunction_firing_fraction": conjunction_sum / denom,
            "residual_mean_abs_logit": residual_abs_sum / max(n_total * model.n_classes, 1),
            "residual_abs_max": residual_abs_max,
        }
    )
    return metrics


def _residual_provenance(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    model: ResidualConjunctionClassifier,
    source_reference: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "condition": CONDITION,
        "seed": spec.seed,
        "split_seed": base.SPLIT_SEED,
        "what_source": source_reference["what_definition"],
        "when_source": source_reference["when_definition"],
        "base_architecture": "direct frozen WHAT128 -> bias-free Linear12 -> valid evidence sum -> one class bias",
        "extra_fusion_lif_between_what_and_base": False,
        "base_frozen_during_residual_training": True,
        "residual_what_input": "the same frozen WHAT L2 spikes used by the direct base; no learned intermediate WHAT representation",
        "when_frozen": True,
        "context_width": CONTEXT_WIDTH,
        "selection_threshold": SELECTION_THRESHOLD,
        "conjunction_threshold": CONJUNCTION_THRESHOLD,
        "conjunction_gain_per_side": CONJUNCTION_GAIN,
        "structural_conjunction": (
            "WHAT-selector and WHEN-selector spikes feed a fixed two-input threshold; either side alone contributes "
            "0.75 < 1.0, while both contribute 1.5 > 1.0"
        ),
        "context_temporal_state": False,
        "residual_output_bias": False,
        "residual_output_initialization": "exact zero; epoch-0 logits equal the frozen direct WHAT base",
        "second_class_bias": False,
        "objective": "final whole-sequence cross entropy only",
        "readout": "frozen direct WHAT base logits + sum of per-timestep conjunction residual evidence",
        "checkpoint_selection": "epoch 0 is legal; maximize validation BA, tie lower validation CE",
        "causality_contract": "final duration is never a model/context input; valid length only masks padding and selects endpoint",
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


def train_residual(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    _validate_spec(spec)
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    source_reference = _source_reference(config, spec.seed)
    source_config = _source_config(config)
    arrays, _ = exp54.load_fusion_cache(spec.seed, data, source_config)
    train_loaders = _make_loaders(arrays, data, spec, config, train_shuffle=True, stage="residual")
    eval_loaders = _make_loaders(arrays, data, spec, config, train_shuffle=False, stage="residual_eval")
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model, base_payload = _initialize_residual(spec, data, config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    epoch0_train = _evaluate_residual(model, eval_loaders["train"], spec, device)
    epoch0_val = _evaluate_residual(model, eval_loaders["val"], spec, device)
    best_val_ba = float(epoch0_val["balanced_accuracy"])
    best_val_loss = float(epoch0_val["loss"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int]] = [
        {
            "epoch": 0,
            "train_loss": float(epoch0_train["loss"]),
            "train_balanced_accuracy": float(epoch0_train["balanced_accuracy"]),
            "val_loss": best_val_loss,
            "val_balanced_accuracy": best_val_ba,
            "val_accuracy": float(epoch0_val["accuracy"]),
            "val_macro_f1": float(epoch0_val["macro_f1"]),
            "conjunction_firing_fraction": float(epoch0_val["conjunction_firing_fraction"]),
            "residual_mean_abs_logit": float(epoch0_val["residual_mean_abs_logit"]),
        }
    ]

    for epoch in range(1, config.epochs + 1):
        model.train()
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        loss_sum = 0.0
        n_total = 0
        for what, when_ordered, _when_reset, labels, lengths in train_loaders["train"]:
            what = what.to(device)
            when_ordered = when_ordered.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(what, when_ordered, lengths)
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
        val_metrics = _evaluate_residual(model, eval_loaders["val"], spec, device)
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
                "conjunction_firing_fraction": float(val_metrics["conjunction_firing_fraction"]),
                "residual_mean_abs_logit": float(val_metrics["residual_mean_abs_logit"]),
            }
        )
        if _is_better(val_ba, val_loss, best_val_ba, best_val_loss):
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state, strict=True)
    native = {
        split: _evaluate_residual(model, loader, spec, device)
        for split, loader in eval_loaders.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "stage": CONDITION,
            "spec": asdict(spec),
            "provenance": _residual_provenance(spec, data, config, model, source_reference),
            "base_result": base_payload["result"],
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


def load_residual(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[ResidualConjunctionClassifier, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.4.1 residual checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("stage") != CONDITION
        or payload.get("spec") != asdict(spec)
    ):
        raise ValueError(f"Wrong Exp5.4.1 residual checkpoint identity: {path}")
    model, _ = _initialize_residual(spec, data, config)
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
    source_config = _source_config(config)
    arrays, _ = exp54.load_fusion_cache(spec.seed, data, source_config)
    loaders = _make_loaders(arrays, data, spec, config, train_shuffle=False, stage="final_eval")
    model, checkpoint = load_residual(spec, data, config)
    base_model, base_payload = load_base(spec, data, config)
    device = torch.device(config.device)
    native = {
        split: _evaluate_residual(model, loader, spec, device)
        for split, loader in loaders.items()
    }
    base_native = {
        split: _evaluate_base(base_model, loader, device)
        for split, loader in loaders.items()
    }

    ablations: list[dict[str, object]] = [
        {"ablation": "ordered", "replicate": 0, "test": native["test"]},
        {
            "ablation": "reset_when",
            "replicate": 0,
            "test": _evaluate_residual(model, loaders["test"], spec, device, ablation="reset_when"),
        },
        {
            "ablation": "when_zero",
            "replicate": 0,
            "test": _evaluate_residual(model, loaders["test"], spec, device, ablation="when_zero"),
        },
    ]
    for replicate in range(SHUFFLE_REPLICATES):
        ablations.append(
            {
                "ablation": "when_shuffle",
                "replicate": replicate,
                "test": _evaluate_residual(
                    model,
                    loaders["test"],
                    spec,
                    device,
                    ablation="when_shuffle",
                    replicate=replicate,
                ),
            }
        )
    ablations.extend(
        [
            {
                "ablation": "when_circular_shift",
                "replicate": 0,
                "test": _evaluate_residual(
                    model,
                    loaders["test"],
                    spec,
                    device,
                    ablation="when_circular_shift",
                ),
            },
            {
                "ablation": "residual_what_zero",
                "replicate": 0,
                "test": _evaluate_residual(
                    model,
                    loaders["test"],
                    spec,
                    device,
                    ablation="residual_what_zero",
                ),
            },
        ]
    )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "condition": CONDITION,
        "seed": spec.seed,
        "parameter_counts": parameter_counts(model),
        "best_epoch": checkpoint["result"]["best_epoch"],
        "base_best_epoch": base_payload["result"]["best_epoch"],
        "provenance": checkpoint["provenance"],
        "base_native": base_native,
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
    train_residual(spec, data, config, force=force)
    return evaluate_one(spec, data, config, force=force)


def _load_eval(root: Path, spec: RunSpec) -> dict[str, object]:
    path = evaluation_path(root, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.4.1 evaluation: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("spec") != asdict(spec)
    ):
        raise ValueError(f"Exp5.4.1 evaluation identity mismatch: {path}")
    return payload


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    specs = run_specs()
    if len(specs) != 5 or len({spec.key for spec in specs}) != 5:
        raise RuntimeError("Exp5.4.1 must contain exactly five unique seed runs")

    base_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    activity_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    base_history_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []

    for spec in specs:
        payload = _load_eval(root, spec)
        base_metrics = payload["base_native"]["test"]
        residual_metrics = payload["native"]["test"]
        base_rows.append(
            {
                "condition": BASE_CONDITION,
                "seed": spec.seed,
                "best_epoch": payload["base_best_epoch"],
                "parameter_count": payload["parameter_counts"]["frozen_base"],
                "test_balanced_accuracy": base_metrics["balanced_accuracy"],
                "test_accuracy": base_metrics["accuracy"],
                "test_macro_f1": base_metrics["macro_f1"],
                "test_loss": base_metrics["loss"],
            }
        )
        row: dict[str, object] = {
            "condition": CONDITION,
            "seed": spec.seed,
            "best_epoch": payload["best_epoch"],
            "base_best_epoch": payload["base_best_epoch"],
            "trainable_parameter_count": payload["parameter_counts"]["trainable_total"],
            "frozen_base_parameter_count": payload["parameter_counts"]["frozen_base"],
        }
        for split in ("train", "val", "test"):
            metrics = payload["native"][split]
            for key in ("loss", "balanced_accuracy", "accuracy", "macro_f1"):
                row[f"{split}_{key}"] = metrics[key]
            for key in (
                "what_selector_firing_fraction",
                "when_selector_firing_fraction",
                "conjunction_firing_fraction",
                "residual_mean_abs_logit",
                "residual_abs_max",
            ):
                activity_rows.append(
                    {
                        "condition": CONDITION,
                        "seed": spec.seed,
                        "split": split,
                        "metric": key,
                        "value": metrics[key],
                    }
                )
        row["base_test_balanced_accuracy"] = base_metrics["balanced_accuracy"]
        row["delta_test_ba_vs_base"] = float(residual_metrics["balanced_accuracy"]) - float(base_metrics["balanced_accuracy"])
        run_rows.append(row)
        paired_rows.append(
            {
                "comparison": f"{CONDITION}_minus_{BASE_CONDITION}",
                "seed": spec.seed,
                "delta_test_balanced_accuracy": float(residual_metrics["balanced_accuracy"]) - float(base_metrics["balanced_accuracy"]),
                "delta_test_macro_f1": float(residual_metrics["macro_f1"]) - float(base_metrics["macro_f1"]),
                "delta_test_loss": float(residual_metrics["loss"]) - float(base_metrics["loss"]),
            }
        )
        for ablation in payload["ablations"]:
            metrics = ablation["test"]
            ablation_rows.append(
                {
                    "condition": CONDITION,
                    "seed": spec.seed,
                    "ablation": ablation["ablation"],
                    "replicate": ablation["replicate"],
                    "test_balanced_accuracy": metrics["balanced_accuracy"],
                    "test_accuracy": metrics["accuracy"],
                    "test_macro_f1": metrics["macro_f1"],
                    "test_loss": metrics["loss"],
                    "what_selector_firing_fraction": metrics["what_selector_firing_fraction"],
                    "when_selector_firing_fraction": metrics["when_selector_firing_fraction"],
                    "conjunction_firing_fraction": metrics["conjunction_firing_fraction"],
                    "residual_mean_abs_logit": metrics["residual_mean_abs_logit"],
                    "residual_abs_max": metrics["residual_abs_max"],
                }
            )
        bhpath = base_history_path(root, spec)
        hpath = history_path(root, spec)
        if not bhpath.exists() or not hpath.exists():
            raise FileNotFoundError(f"Missing Exp5.4.1 history for seed {spec.seed}")
        base_history = pd.read_csv(bhpath)
        base_history.insert(0, "seed", spec.seed)
        base_history_parts.append(base_history)
        history = pd.read_csv(hpath)
        history.insert(0, "seed", spec.seed)
        history_parts.append(history)

    outputs = {
        "base_runs": root / "base_runs.csv",
        "base_histories": root / "base_histories.csv",
        "runs": root / "runs.csv",
        "histories": root / "histories.csv",
        "ablation_runs": root / "ablation_runs.csv",
        "activity_runs": root / "activity_runs.csv",
        "paired_deltas": root / "paired_deltas.csv",
        "manifest": root / "manifest.json",
    }
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(base_rows).to_csv(outputs["base_runs"], index=False)
    pd.concat(base_history_parts, ignore_index=True).to_csv(outputs["base_histories"], index=False)
    pd.DataFrame(run_rows).to_csv(outputs["runs"], index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(outputs["histories"], index=False)
    pd.DataFrame(ablation_rows).to_csv(outputs["ablation_runs"], index=False)
    pd.DataFrame(activity_rows).to_csv(outputs["activity_runs"], index=False)
    pd.DataFrame(paired_rows).to_csv(outputs["paired_deltas"], index=False)
    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "scientific_question": (
                "whether frozen WHEN spikes add class information to the original WHAT L2 spike representation when "
                "WHEN can act only through an instantaneous structural WHAT x WHEN conjunction residual"
            ),
            "seeds": list(SEEDS),
            "base_condition": BASE_CONDITION,
            "residual_condition": CONDITION,
            "expected_base_runs": 5,
            "expected_residual_runs": 5,
            "architecture_contract": (
                "WHAT128 -> direct bias-free Linear12 -> non-leaky valid-time accumulator + one class bias; in parallel, "
                "WHAT128 -> selector128 AND frozen WHEN64 -> selector128 -> conjunction128 -> bias-free residual Linear12 -> accumulator"
            ),
            "extra_fusion_lif_between_what_and_base": False,
            "baseline_preservation": "freeze trained direct WHAT base; zero-initialize residual output; epoch 0 exactly equals direct WHAT base",
            "shortcut_prevention": "residual evidence is structurally zero if either residual WHAT input or WHEN input is zero",
            "context_temporal_state": False,
            "ablations": list(ABLATIONS),
            "shuffle_replicates": SHUFFLE_REPLICATES,
            "winner_logic": (
                "primary evidence is positive paired test BA vs direct WHAT base plus ordered > reset/shuffle/shift; "
                "when_zero and residual_what_zero must exactly recover the direct WHAT base"
            ),
            "aggregation_policy": "finalizer only aggregates existing base/residual artifacts; no training or cache regeneration",
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
    parser = argparse.ArgumentParser(description="Experiment 5.4.1 direct WHAT plus constrained WHATxWHEN residual")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo-root", type=str, default=None)
        subparser.add_argument("--device", type=str, default="cpu")
        subparser.add_argument("--threads", type=int, default=1)
        subparser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        subparser.add_argument("--epochs", type=int, default=EPOCHS)
        subparser.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare-inputs")
    common(prepare)
    prepare.add_argument("--array-task-id", type=int, required=True)

    train_base_parser = subparsers.add_parser("train-base")
    common(train_base_parser)
    train_base_parser.add_argument("--array-task-id", type=int, required=True)

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
    task_id = int(args.array_task_id)
    if not 0 <= task_id < len(SEEDS):
        raise IndexError(f"Exp5.4.1 array task {task_id} outside 0..{len(SEEDS)-1}")
    spec = RunSpec(SEEDS[task_id])
    if args.command == "prepare-inputs":
        print(prepare_inputs_seed(spec.seed, data, config, force=args.force))
        return
    if args.command == "train-base":
        print(train_base(spec, data, config, force=args.force))
        return
    if args.command == "run-one":
        payload = run_one(spec, data, config, force=args.force)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
