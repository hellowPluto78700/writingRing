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
exp5323 = exp54.exp5323

EXPERIMENT_ID = "experiment_5_4_1_constrained_conjunction_residual"
PROTOCOL_VERSION = "constrained_conjunction_residual_v1"
SEEDS = exp54.SEEDS
WHAT_WIDTH = exp54.WHAT_WIDTH
WHEN_WIDTH = exp54.WHEN_WIDTH
BASE_FUSION_WIDTH = exp54.FUSION_WIDTH
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

CONDITION = "constrained_conjunction_residual"
SOURCE_BASE_CONDITION = exp54.WHAT_ONLY
SOURCE_COMPARATORS = (
    exp54.WHAT_ONLY,
    exp54.WHAT_ELAPSED,
    exp54.WHAT_RESETWHEN,
    exp54.WHAT_WHEN_LINEAR,
    exp54.WHAT_WHEN_FUSION,
)
ABLATIONS = (
    "ordered",
    "reset_when",
    "when_zero",
    "when_shuffle",
    "when_circular_shift",
    "base_context_zero",
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
    base_selector_spikes: torch.Tensor
    when_selector_spikes: torch.Tensor
    conjunction_spikes: torch.Tensor
    residual_logits: torch.Tensor


def find_repo_root(start: Path | None = None) -> Path:
    return exp54.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def source_reference_path(root: Path, seed: int) -> Path:
    return root / "source_references" / f"seed{seed}.json"


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


def _source_spec(condition: str, seed: int) -> exp54.RunSpec:
    return exp54.RunSpec(condition, seed)


def _base_spec(seed: int) -> exp54.RunSpec:
    return _source_spec(SOURCE_BASE_CONDITION, seed)


def _source_eval(config: Config, condition: str, seed: int) -> dict[str, object]:
    source_root = exp54.results_dir(config.repo_root)
    return exp54._load_eval(source_root, _source_spec(condition, seed))


def prepare_source_seed(
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
    arrays, cache_meta = exp54.load_fusion_cache(seed, data, source_config)
    del arrays
    base_model, base_payload = exp54.load_model(_base_spec(seed), data, source_config)
    del base_model
    base_eval = _source_eval(config, SOURCE_BASE_CONDITION, seed)
    for condition in SOURCE_COMPARATORS:
        _source_eval(config, condition, seed)

    source_checkpoint = exp54.checkpoint_path(source_config.results_dir, _base_spec(seed))
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
            "source_base_condition": SOURCE_BASE_CONDITION,
            "source_base_checkpoint": str(source_checkpoint.relative_to(config.repo_root)),
            "source_base_best_epoch": base_payload["result"]["best_epoch"],
            "source_base_test_balanced_accuracy": base_eval["native"]["test"]["balanced_accuracy"],
            "source_fusion_cache": str(source_cache.relative_to(config.repo_root)),
            "source_fusion_cache_meta": str(source_cache_meta.relative_to(config.repo_root)),
            "source_when_checkpoint": cache_meta["source_checkpoint"],
            "source_when_condition": cache_meta["source_condition"],
            "what_width": WHAT_WIDTH,
            "when_width": WHEN_WIDTH,
            "base_fusion_width": BASE_FUSION_WIDTH,
            "split_seed": base.SPLIT_SEED,
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
        },
    )
    return destination


class ResidualConjunctionClassifier(nn.Module):
    """Frozen WHAT classifier plus a structurally conjunctive WHEN residual path."""

    def __init__(
        self,
        base_model: exp54.FusionClassifier,
        n_classes: int,
    ) -> None:
        super().__init__()
        if base_model.fusion_mode != "lif":
            raise ValueError("Exp5.4.1 requires the Exp5.4 WHAT-only LIF base model")
        self.n_classes = int(n_classes)
        self.base_model = base_model
        self.base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

        self.base_selector = nn.Linear(BASE_FUSION_WIDTH, CONTEXT_WIDTH, bias=False)
        self.when_selector = nn.Linear(WHEN_WIDTH, CONTEXT_WIDTH, bias=False)
        self.residual_output = nn.Linear(CONTEXT_WIDTH, self.n_classes, bias=False)
        self.spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

    def train(self, mode: bool = True) -> ResidualConjunctionClassifier:
        super().train(mode)
        self.base_model.eval()
        return self

    def residual_parameters(self) -> list[nn.Parameter]:
        return [
            *self.base_selector.parameters(),
            *self.when_selector.parameters(),
            *self.residual_output.parameters(),
        ]

    def forward(
        self,
        what: torch.Tensor,
        when_context: torch.Tensor,
        lengths: torch.Tensor,
        *,
        zero_base_context: bool = False,
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

        zero_when = torch.zeros_like(when_context)
        with torch.no_grad():
            base_logits, base_fusion = self.base_model(
                what,
                zero_when,
                lengths,
                return_fusion_trajectory=True,
            )
        if base_fusion is None:
            raise RuntimeError("Frozen WHAT base did not return its Fusion spike trajectory")
        base_context = torch.zeros_like(base_fusion) if zero_base_context else base_fusion

        base_selected = self.spike_grad(self.base_selector(base_context) - SELECTION_THRESHOLD)
        when_selected = self.spike_grad(self.when_selector(when_context) - SELECTION_THRESHOLD)
        conjunction_current = CONJUNCTION_GAIN * (base_selected + when_selected)
        conjunction = self.spike_grad(conjunction_current - CONJUNCTION_THRESHOLD)

        valid = base.mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
        base_selected = base_selected * valid
        when_selected = when_selected * valid
        conjunction = conjunction * valid
        residual_step = self.residual_output(conjunction) * valid
        residual_logits = residual_step.sum(dim=1)
        logits = base_logits + residual_logits

        trajectory = None
        if return_trajectory:
            trajectory = ResidualTrajectory(
                base_selector_spikes=base_selected,
                when_selector_spikes=when_selected,
                conjunction_spikes=conjunction,
                residual_logits=residual_logits,
            )
        return logits, trajectory


def _initialize_model(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[ResidualConjunctionClassifier, dict[str, object]]:
    _validate_spec(spec)
    source_config = _source_config(config)
    base_model, base_payload = exp54.load_model(_base_spec(spec.seed), data, source_config)
    model = ResidualConjunctionClassifier(base_model, len(data.labels)).to(torch.device(config.device))

    base.seed_all(base.dseed(spec.seed, EXPERIMENT_ID, "base_selector"))
    nn.init.uniform_(model.base_selector.weight, a=0.0, b=SELECTOR_INIT_MAX)
    base.seed_all(base.dseed(spec.seed, EXPERIMENT_ID, "when_selector"))
    nn.init.uniform_(model.when_selector.weight, a=0.0, b=SELECTOR_INIT_MAX)
    with torch.no_grad():
        model.residual_output.weight.zero_()
    return model, base_payload


def parameter_counts(model: ResidualConjunctionClassifier) -> dict[str, int]:
    frozen_base = int(sum(p.numel() for p in model.base_model.parameters()))
    return {
        "frozen_base": frozen_base,
        "base_selector": int(sum(p.numel() for p in model.base_selector.parameters())),
        "when_selector": int(sum(p.numel() for p in model.when_selector.parameters())),
        "residual_output": int(sum(p.numel() for p in model.residual_output.parameters())),
        "trainable_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "stored_total": int(sum(p.numel() for p in model.parameters())),
    }


def _loaders(
    arrays: dict[str, np.ndarray],
    data: base.Data,
    spec: RunSpec,
    config: Config,
    train_shuffle: bool,
) -> dict[str, DataLoader]:
    source_spec = _base_spec(spec.seed)
    source_config = _source_config(config)
    return exp54._make_loaders(
        arrays,
        data,
        source_spec,
        source_config,
        train_shuffle=train_shuffle,
    )


def _shuffle_valid_context(
    context: torch.Tensor,
    lengths: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    return exp54._shuffle_valid_context(context, lengths, seed)


def _circular_shift_valid_context(
    context: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    return exp54._circular_shift_valid_context(context, lengths)


def _select_when(
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
    if ablation == "base_context_zero":
        return when_ordered, True
    raise ValueError(f"Unknown Exp5.4.1 ablation: {ablation}")


def _evaluate(
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
    base_selector_sum = 0.0
    when_selector_sum = 0.0
    conjunction_sum = 0.0
    valid_context_steps = 0.0
    residual_abs_sum = 0.0

    with torch.no_grad():
        for batch_index, (what, when_ordered, when_reset, labels, lengths) in enumerate(loader):
            what = what.to(device)
            when_ordered = when_ordered.to(device)
            when_reset = when_reset.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            context, zero_base_context = _select_when(
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
                zero_base_context=zero_base_context,
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
            base_selector_sum += float((trajectory.base_selector_spikes * valid).sum().item())
            when_selector_sum += float((trajectory.when_selector_spikes * valid).sum().item())
            conjunction_sum += float((trajectory.conjunction_spikes * valid).sum().item())
            residual_abs_sum += float(trajectory.residual_logits.abs().sum().item())

    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    denom = max(valid_context_steps, 1.0)
    return {
        "loss": loss_sum / max(n_total, 1),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro")),
        "n_samples": int(n_total),
        "base_selector_firing_fraction": base_selector_sum / denom,
        "when_selector_firing_fraction": when_selector_sum / denom,
        "conjunction_firing_fraction": conjunction_sum / denom,
        "residual_mean_abs_logit": residual_abs_sum / max(n_total * model.n_classes, 1),
    }


def _is_better(val_ba: float, val_loss: float, best_ba: float, best_loss: float) -> bool:
    return val_ba > best_ba + 1e-12 or (
        abs(val_ba - best_ba) <= 1e-12 and val_loss < best_loss - 1e-12
    )


def _provenance(
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
        "source_experiment_id": exp54.EXPERIMENT_ID,
        "source_protocol_version": exp54.PROTOCOL_VERSION,
        "source_base_condition": SOURCE_BASE_CONDITION,
        "source_base_checkpoint": source_reference["source_base_checkpoint"],
        "source_when_checkpoint": source_reference["source_when_checkpoint"],
        "base_frozen": True,
        "base_context": "frozen Exp5.4 WHAT-only Fusion128 spikes; raw WHAT is not visible to the residual path",
        "when_frozen": True,
        "when_context": "frozen Exp5.3.2.3 FF-SNN128 -> RSNN64 output spikes",
        "context_width": CONTEXT_WIDTH,
        "selection_threshold": SELECTION_THRESHOLD,
        "conjunction_threshold": CONJUNCTION_THRESHOLD,
        "conjunction_gain_per_side": CONJUNCTION_GAIN,
        "structural_conjunction": (
            "base-selector and WHEN-selector spikes feed a fixed two-input threshold; either side alone "
            "contributes 0.75 < threshold 1.0, while both contribute 1.5 > threshold"
        ),
        "context_temporal_state": False,
        "residual_output_bias": False,
        "residual_output_initialization": "exact zero; epoch-0 logits equal the frozen base classifier",
        "second_class_bias": False,
        "objective": "final whole-sequence cross entropy only",
        "readout": "frozen base logits + sum of per-timestep residual class evidence over valid timesteps",
        "checkpoint_selection": (
            "epoch 0 frozen base is a legal candidate; maximize validation balanced accuracy, tie lower validation CE"
        ),
        "causality_contract": (
            "final duration is never a model/context input; valid length only masks padded timesteps and selects endpoint"
        ),
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
        or payload.get("source_base_condition") != SOURCE_BASE_CONDITION
    ):
        raise ValueError(f"Exp5.4.1 source reference identity mismatch: {path}")
    return payload


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

    source_reference = _source_reference(config, spec.seed)
    source_config = _source_config(config)
    arrays, _ = exp54.load_fusion_cache(spec.seed, data, source_config)
    train_loaders = _loaders(arrays, data, spec, config, train_shuffle=True)
    eval_loaders = _loaders(arrays, data, spec, config, train_shuffle=False)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model, source_base_payload = _initialize_model(spec, data, config)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    # Epoch 0 is exactly the frozen WHAT-only classifier and is a legal checkpoint.
    epoch0_train = _evaluate(model, eval_loaders["train"], spec, device)
    epoch0_val = _evaluate(model, eval_loaders["val"], spec, device)
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
        val_metrics = _evaluate(model, eval_loaders["val"], spec, device)
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
        split: _evaluate(model, loader, spec, device)
        for split, loader in eval_loaders.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "provenance": _provenance(spec, data, config, model, source_reference),
            "source_base_result": source_base_payload["result"],
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
) -> tuple[ResidualConjunctionClassifier, dict[str, object]]:
    _validate_spec(spec)
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.4.1 checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("spec") != asdict(spec)
    ):
        raise ValueError(f"Wrong Exp5.4.1 checkpoint identity: {path}")
    model, _ = _initialize_model(spec, data, config)
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

    source_config = _source_config(config)
    arrays, _ = exp54.load_fusion_cache(spec.seed, data, source_config)
    loaders = _loaders(arrays, data, spec, config, train_shuffle=False)
    model, checkpoint = load_model(spec, data, config)
    device = torch.device(config.device)
    native = {
        split: _evaluate(model, loader, spec, device)
        for split, loader in loaders.items()
    }

    ablations: list[dict[str, object]] = [
        {"ablation": "ordered", "replicate": 0, "test": native["test"]},
        {
            "ablation": "reset_when",
            "replicate": 0,
            "test": _evaluate(model, loaders["test"], spec, device, ablation="reset_when"),
        },
        {
            "ablation": "when_zero",
            "replicate": 0,
            "test": _evaluate(model, loaders["test"], spec, device, ablation="when_zero"),
        },
    ]
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
                "test": _evaluate(
                    model,
                    loaders["test"],
                    spec,
                    device,
                    ablation="when_circular_shift",
                ),
            },
            {
                "ablation": "base_context_zero",
                "replicate": 0,
                "test": _evaluate(
                    model,
                    loaders["test"],
                    spec,
                    device,
                    ablation="base_context_zero",
                ),
            },
        ]
    )

    source_rows = {
        condition: _source_eval(config, condition, spec.seed)["native"]["test"]
        for condition in SOURCE_COMPARATORS
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "condition": CONDITION,
        "seed": spec.seed,
        "parameter_counts": parameter_counts(model),
        "best_epoch": checkpoint["result"]["best_epoch"],
        "provenance": checkpoint["provenance"],
        "native": native,
        "ablations": ablations,
        "source_test_metrics": source_rows,
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

    run_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    activity_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    history_parts: list[pd.DataFrame] = []

    for spec in specs:
        payload = _load_eval(root, spec)
        row: dict[str, object] = {
            "condition": CONDITION,
            "seed": spec.seed,
            "best_epoch": payload["best_epoch"],
            "trainable_parameter_count": payload["parameter_counts"]["trainable_total"],
            "frozen_base_parameter_count": payload["parameter_counts"]["frozen_base"],
        }
        for split in ("train", "val", "test"):
            metrics = payload["native"][split]
            for key in ("loss", "balanced_accuracy", "accuracy", "macro_f1"):
                row[f"{split}_{key}"] = metrics[key]
            for key in (
                "base_selector_firing_fraction",
                "when_selector_firing_fraction",
                "conjunction_firing_fraction",
                "residual_mean_abs_logit",
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
        base_test_ba = float(payload["source_test_metrics"][SOURCE_BASE_CONDITION]["balanced_accuracy"])
        row["source_base_test_balanced_accuracy"] = base_test_ba
        row["delta_test_ba_vs_base"] = float(payload["native"]["test"]["balanced_accuracy"]) - base_test_ba
        run_rows.append(row)

        for condition, metrics in payload["source_test_metrics"].items():
            source_rows.append(
                {
                    "condition": condition,
                    "seed": spec.seed,
                    "test_balanced_accuracy": metrics["balanced_accuracy"],
                    "test_accuracy": metrics["accuracy"],
                    "test_macro_f1": metrics["macro_f1"],
                    "test_loss": metrics["loss"],
                }
            )
            paired_rows.append(
                {
                    "comparison": f"{CONDITION}_minus_{condition}",
                    "comparator": condition,
                    "seed": spec.seed,
                    "delta_test_balanced_accuracy": float(payload["native"]["test"]["balanced_accuracy"])
                    - float(metrics["balanced_accuracy"]),
                    "delta_test_macro_f1": float(payload["native"]["test"]["macro_f1"])
                    - float(metrics["macro_f1"]),
                    "delta_test_loss": float(payload["native"]["test"]["loss"])
                    - float(metrics["loss"]),
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
                    "base_selector_firing_fraction": metrics["base_selector_firing_fraction"],
                    "when_selector_firing_fraction": metrics["when_selector_firing_fraction"],
                    "conjunction_firing_fraction": metrics["conjunction_firing_fraction"],
                    "residual_mean_abs_logit": metrics["residual_mean_abs_logit"],
                }
            )

        hpath = history_path(root, spec)
        if not hpath.exists():
            raise FileNotFoundError(f"Missing Exp5.4.1 history: {hpath}")
        history = pd.read_csv(hpath)
        history.insert(0, "seed", spec.seed)
        history_parts.append(history)

    outputs = {
        "runs": root / "runs.csv",
        "histories": root / "histories.csv",
        "ablation_runs": root / "ablation_runs.csv",
        "activity_runs": root / "activity_runs.csv",
        "source_runs": root / "source_runs.csv",
        "paired_deltas": root / "paired_deltas.csv",
        "manifest": root / "manifest.json",
    }
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(outputs["runs"], index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(outputs["histories"], index=False)
    pd.DataFrame(ablation_rows).to_csv(outputs["ablation_runs"], index=False)
    pd.DataFrame(activity_rows).to_csv(outputs["activity_runs"], index=False)
    pd.DataFrame(source_rows).to_csv(outputs["source_runs"], index=False)
    pd.DataFrame(paired_rows).to_csv(outputs["paired_deltas"], index=False)

    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "scientific_question": (
                "whether frozen WHEN spikes provide incremental class value when they can only act through a "
                "structurally conjunctive residual correction on frozen WHAT-only Fusion spikes"
            ),
            "condition": CONDITION,
            "seeds": list(SEEDS),
            "expected_train_runs": len(specs),
            "source_base": f"{exp54.EXPERIMENT_ID}/{exp54.PROTOCOL_VERSION}/{SOURCE_BASE_CONDITION}",
            "source_comparators": list(SOURCE_COMPARATORS),
            "architecture_contract": (
                "frozen Exp5.4 WHAT-only logits + [frozen base Fusion128 spikes -> selector spikes] AND "
                "[frozen WHEN64 spikes -> selector spikes] -> conjunction128 -> bias-free residual 12D evidence -> sum"
            ),
            "baseline_preservation": (
                "residual output is zero-initialized; epoch 0 is exactly the source WHAT-only classifier and is a legal checkpoint"
            ),
            "shortcut_prevention": (
                "the residual path never reads raw WHAT and cannot emit conjunction spikes if either base Fusion spikes or WHEN spikes are zero"
            ),
            "context_temporal_state": False,
            "ablations": list(ABLATIONS),
            "shuffle_replicates": SHUFFLE_REPLICATES,
            "winner_logic": (
                "primary evidence is positive paired test BA vs source WHAT-only together with aligned > reset/shuffle/shift, "
                "while when_zero and base_context_zero remain exactly at the frozen base"
            ),
            "aggregation_policy": "finalizer aggregates five existing run artifacts only; it never trains or regenerates source caches",
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
        description="Experiment 5.4.1 constrained WHATxWHEN residual conjunction"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo-root", type=str, default=None)
        subparser.add_argument("--device", type=str, default="cpu")
        subparser.add_argument("--threads", type=int, default=1)
        subparser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        subparser.add_argument("--epochs", type=int, default=EPOCHS)
        subparser.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare-source")
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
    task_id = int(args.array_task_id)
    if not 0 <= task_id < len(SEEDS):
        raise IndexError(f"Exp5.4.1 array task {task_id} outside 0..{len(SEEDS)-1}")
    spec = RunSpec(SEEDS[task_id])
    if args.command == "prepare-source":
        print(prepare_source_seed(spec.seed, data, config, force=args.force))
        return
    if args.command == "run-one":
        payload = run_one(spec, data, config, force=args.force)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
