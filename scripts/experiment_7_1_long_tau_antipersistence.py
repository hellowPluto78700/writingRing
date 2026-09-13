from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from scripts import experiment_3_0_1_single_tau_objective_comparison as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50
from scripts import experiment_7_0_hierarchical_context_snn as exp70


EXPERIMENT_ID = "experiment_7_1_long_tau_antipersistence"
PROTOCOL_VERSION = "long_tau_antipersistence_v1"

NO_REG = "no_reg"
ALL_LOSS_MASKED = "all_loss_masked"
WHOLECOUNT_ONLY_MASKED = "wholecount_only_masked"
CONDITIONS = (NO_REG, ALL_LOSS_MASKED, WHOLECOUNT_ONLY_MASKED)
TRAIN_SEEDS = (11, 23, 37)

SHORT_SHIFT = 4
MIDDLE_SHIFT = 5
LONG_SHIFT = 6
HIERARCHICAL_SHIFTS = (SHORT_SHIFT, MIDDLE_SHIFT, LONG_SHIFT)
HIDDEN_WIDTH = 128
HIDDEN_CAP = 1
OUTPUT_CAP = 1

MAX_EPOCHS = 100
MIN_EPOCHS = 20
PATIENCE = 30
WARMUP_EPOCHS = 10
BATCH_SIZE = exp3.BATCH_SIZE
LR = exp3.LR
WEIGHT_DECAY = 0.0

CALIBRATION_BATCHES = 5
TARGET_RATE_GRAD_RATIO = 0.025
TARGET_PERSIST_GRAD_RATIO = 0.05

PERSIST_SHORT_WINDOW = 3
PERSIST_SHORT_ALLOWED = 2
PERSIST_LONG_WINDOW = 8
PERSIST_LONG_ALLOWED = 4
PERSIST_LONG_WEIGHT = 0.5

TAU_MEM_MS = exp70.TAU_MEM_MS
OUTPUT_TAU_MEM_MS = exp70.OUTPUT_TAU_MEM_MS
THRESHOLD = exp70.THRESHOLD
SURROGATE_SLOPE = exp70.SURROGATE_SLOPE
EXPECTED_FS = exp70.EXPECTED_FS
EXPECTED_CHANNELS = exp70.EXPECTED_CHANNELS
EXPECTED_STEPS = exp70.EXPECTED_STEPS
EXPECTED_SPLIT_SEED = exp70.EXPECTED_SPLIT_SEED


@dataclass(frozen=True)
class RunSpec:
    condition: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.condition}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    max_epochs: int = MAX_EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp70.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(condition, seed) for condition in CONDITIONS for seed in TRAIN_SEEDS]


def validate_spec(spec: RunSpec) -> None:
    if spec.condition not in CONDITIONS:
        raise ValueError(f"Unknown Exp7.1 condition: {spec.condition}")
    if spec.seed not in TRAIN_SEEDS:
        raise ValueError(f"Unknown Exp7.1 seed: {spec.seed}")


def paired_seed(seed: int, role: str) -> int:
    """Condition-independent stream so R0/R1/R2 are paired within each seed."""
    return exp3.dseed(seed, "exp7_1_long_tau_antipersistence", role)


def prepare_data(repo_root: Path) -> exp3.Data:
    data = exp70.prepare_data(repo_root)
    if not np.isclose(float(data.fs), EXPECTED_FS):
        raise ValueError(f"Exp7.1 requires {EXPECTED_FS:g} Hz")
    if int(exp3.SPLIT_SEED) != EXPECTED_SPLIT_SEED:
        raise ValueError("Exp7.1 split-seed contract changed")
    if int(data.Xtr.shape[1]) != EXPECTED_STEPS:
        raise ValueError(f"Exp7.1 requires {EXPECTED_STEPS} padded timesteps")
    return data


def _partitions(data: exp3.Data) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }


def make_loaders(
    data: exp3.Data,
    seed: int,
    batch_size: int,
    train_shuffle: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    return {
        split: exp3.loader(
            X,
            y,
            lengths,
            batch_size,
            train_shuffle if split == "train" else False,
            paired_seed(seed, f"{split}_loader"),
        )
        for split, (X, y, lengths) in _partitions(data).items()
    }


def calibration_loader(
    data: exp3.Data,
    seed: int,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    return exp3.loader(
        data.Xtr,
        data.ytr,
        data.ltr,
        batch_size,
        True,
        paired_seed(seed, "calibration_loader"),
    )


class SerialLongTauSNN(nn.Module):
    """Raw30 -> S(s4) -> M(s5) -> L(s6) -> short-tau output LIF."""

    def __init__(self, n_classes: int, fs: float) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.fs = float(fs)

        self.short_linear = nn.Linear(EXPECTED_CHANNELS, HIDDEN_WIDTH, bias=False)
        self.middle_linear = nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False)
        self.long_linear = nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False)
        self.output_linear = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=False)

        hidden_beta = math.exp(-(1000.0 / self.fs) / float(TAU_MEM_MS))
        output_beta = math.exp(-(1000.0 / self.fs) / float(OUTPUT_TAU_MEM_MS))
        self.short_lif = exp401.MacroMultiSpikeLIF(
            beta=hidden_beta,
            threshold=THRESHOLD,
            max_spikes_per_dt=HIDDEN_CAP,
            surrogate_slope=SURROGATE_SLOPE,
        )
        self.middle_lif = exp401.MacroMultiSpikeLIF(
            beta=hidden_beta,
            threshold=THRESHOLD,
            max_spikes_per_dt=HIDDEN_CAP,
            surrogate_slope=SURROGATE_SLOPE,
        )
        self.long_lif = exp401.MacroMultiSpikeLIF(
            beta=hidden_beta,
            threshold=THRESHOLD,
            max_spikes_per_dt=HIDDEN_CAP,
            surrogate_slope=SURROGATE_SLOPE,
        )
        self.output_lif = exp401.MacroMultiSpikeLIF(
            beta=output_beta,
            threshold=THRESHOLD,
            max_spikes_per_dt=OUTPUT_CAP,
            surrogate_slope=SURROGATE_SLOPE,
        )

        self.short_alpha = exp70.exp60.decay_from_shift(SHORT_SHIFT)
        self.middle_alpha = exp70.exp60.decay_from_shift(MIDDLE_SHIFT)
        self.long_alpha = exp70.exp60.decay_from_shift(LONG_SHIFT)

    def forward_trajectory(
        self,
        x: torch.Tensor,
        capture_states: bool = False,
    ) -> dict[str, Any]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,C], got {tuple(x.shape)}")
        batch, n_steps, channels = x.shape
        if channels != EXPECTED_CHANNELS:
            raise ValueError(f"Expected {EXPECTED_CHANNELS} channels, got {channels}")

        short_syn = torch.zeros(batch, HIDDEN_WIDTH, device=x.device, dtype=x.dtype)
        middle_syn = torch.zeros_like(short_syn)
        long_syn = torch.zeros_like(short_syn)
        short_mem = torch.zeros_like(short_syn)
        middle_mem = torch.zeros_like(short_syn)
        long_mem = torch.zeros_like(short_syn)
        output_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)

        long_spikes: list[torch.Tensor] = []
        output_spikes: list[torch.Tensor] = []
        captured_spikes: dict[str, list[torch.Tensor]] = {
            "short": [],
            "middle": [],
            "long": [],
        }
        long_currents: list[torch.Tensor] = []
        long_membranes: list[torch.Tensor] = []

        for step in range(n_steps):
            short_syn = self.short_alpha * short_syn + self.short_linear(x[:, step])
            short_spike, short_mem, _ = self.short_lif(short_syn, short_mem)

            middle_syn = self.middle_alpha * middle_syn + self.middle_linear(short_spike)
            middle_spike, middle_mem, _ = self.middle_lif(middle_syn, middle_mem)

            long_syn = self.long_alpha * long_syn + self.long_linear(middle_spike)
            long_spike, long_mem, _ = self.long_lif(long_syn, long_mem)

            output_spike, output_mem, _ = self.output_lif(
                self.output_linear(long_spike), output_mem
            )
            long_spikes.append(long_spike)
            output_spikes.append(output_spike)

            if capture_states:
                captured_spikes["short"].append(short_spike)
                captured_spikes["middle"].append(middle_spike)
                captured_spikes["long"].append(long_spike)
                long_currents.append(long_syn)
                long_membranes.append(long_mem)

        payload: dict[str, Any] = {
            "long_spikes": torch.stack(long_spikes, dim=1),
            "output_spikes": torch.stack(output_spikes, dim=1),
        }
        if capture_states:
            payload["hidden_spikes"] = {
                name: torch.stack(values, dim=1)
                for name, values in captured_spikes.items()
            }
            payload["long_current"] = torch.stack(long_currents, dim=1)
            payload["long_membrane"] = torch.stack(long_membranes, dim=1)
        return payload


def wholecount_loss(
    output_spikes: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    logits = exp50.deployment_ce_logits(output_spikes, lengths, OUTPUT_CAP)
    return F.cross_entropy(logits, y)


def long_rate_loss(
    long_spikes: torch.Tensor,
    lengths: torch.Tensor,
    condition: str,
) -> torch.Tensor:
    if condition == WHOLECOUNT_ONLY_MASKED:
        return long_spikes.mean()
    if condition == ALL_LOSS_MASKED:
        valid = exp50.valid_mask(lengths, long_spikes.shape[1]).to(long_spikes.dtype)
        numerator = (long_spikes * valid.unsqueeze(-1)).sum()
        denominator = valid.sum() * long_spikes.shape[2]
        return numerator / denominator.clamp_min(1.0)
    raise ValueError(f"Rate loss is undefined for condition {condition}")


def persistence_loss(
    long_spikes: torch.Tensor,
    lengths: torch.Tensor,
    condition: str,
    window: int,
    allowed: int,
) -> torch.Tensor:
    if window < 1 or window > long_spikes.shape[1]:
        raise ValueError("Invalid persistence window")
    if allowed < 0 or allowed >= window:
        raise ValueError("allowed must satisfy 0 <= allowed < window")

    windows = long_spikes.unfold(dimension=1, size=window, step=1)
    counts = windows.sum(dim=-1)
    penalty = F.relu(counts - float(allowed)).square()
    if condition == WHOLECOUNT_ONLY_MASKED:
        return penalty.mean()
    if condition == ALL_LOSS_MASKED:
        starts = torch.arange(
            penalty.shape[1], device=lengths.device, dtype=lengths.dtype
        ).unsqueeze(0)
        valid_windows = starts + int(window) <= lengths.unsqueeze(1)
        mask = valid_windows.to(penalty.dtype).unsqueeze(-1)
        numerator = (penalty * mask).sum()
        denominator = mask.sum() * penalty.shape[2]
        return numerator / denominator.clamp_min(1.0)
    raise ValueError(f"Persistence loss is undefined for condition {condition}")


def long_regularization_terms(
    long_spikes: torch.Tensor,
    lengths: torch.Tensor,
    condition: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if condition == NO_REG:
        zero = long_spikes.sum() * 0.0
        return zero, zero, zero, zero
    rate = long_rate_loss(long_spikes, lengths, condition)
    p32 = persistence_loss(
        long_spikes,
        lengths,
        condition,
        PERSIST_SHORT_WINDOW,
        PERSIST_SHORT_ALLOWED,
    )
    p84 = persistence_loss(
        long_spikes,
        lengths,
        condition,
        PERSIST_LONG_WINDOW,
        PERSIST_LONG_ALLOWED,
    )
    persist = p32 + PERSIST_LONG_WEIGHT * p84
    return rate, p32, p84, persist


def warmup_scale(epoch: int) -> float:
    if epoch < 1:
        raise ValueError("epoch must be >= 1")
    return min(1.0, float(epoch) / float(WARMUP_EPOCHS))


def _gradient_norm(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    retain_graph: bool,
) -> float:
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    square_sum = 0.0
    for grad in grads:
        if grad is not None:
            square_sum += float(grad.detach().square().sum().item())
    return math.sqrt(max(square_sum, 0.0))


def _calibrated_lambda(target_ratio: float, median_ratio: float) -> float:
    if not math.isfinite(median_ratio) or median_ratio <= 1e-12:
        raise RuntimeError(
            "Regularizer gradient is zero/non-finite during calibration; "
            "cannot calibrate a meaningful Exp7.1 strength"
        )
    value = float(target_ratio) / float(median_ratio)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("Calibrated lambda is invalid")
    return value


def calibrate_strength(
    spec: RunSpec,
    model: SerialLongTauSNN,
    data: exp3.Data,
    config: Config,
) -> dict[str, object]:
    if spec.condition == NO_REG:
        return {
            "condition": spec.condition,
            "calibrated": False,
            "lambda_rate": 0.0,
            "lambda_persist": 0.0,
            "target_rate_grad_ratio": 0.0,
            "target_persist_grad_ratio": 0.0,
            "parameter_scope": "long_linear.weight",
            "batches": [],
        }

    device = torch.device(config.device)
    loader = calibration_loader(data, spec.seed, config.batch_size)
    params = (model.long_linear.weight,)
    rows: list[dict[str, float | int]] = []
    model.eval()

    for batch_index, (X, y, lengths) in enumerate(loader):
        if batch_index >= CALIBRATION_BATCHES:
            break
        X = X.to(device)
        y = y.to(device)
        lengths = lengths.to(device)
        trajectory = model.forward_trajectory(X)
        task = wholecount_loss(trajectory["output_spikes"], lengths, y)
        rate, p32, p84, persist = long_regularization_terms(
            trajectory["long_spikes"], lengths, spec.condition
        )
        task_norm = _gradient_norm(task, params, retain_graph=True)
        rate_norm = _gradient_norm(rate, params, retain_graph=True)
        persist_norm = _gradient_norm(persist, params, retain_graph=False)
        if task_norm <= 1e-12:
            raise RuntimeError("Task gradient is zero during Exp7.1 calibration")
        rows.append(
            {
                "batch_index": batch_index,
                "task_loss": float(task.detach().item()),
                "rate_loss": float(rate.detach().item()),
                "p32_loss": float(p32.detach().item()),
                "p84_loss": float(p84.detach().item()),
                "persist_loss": float(persist.detach().item()),
                "task_long_grad_norm": task_norm,
                "rate_long_grad_norm": rate_norm,
                "persist_long_grad_norm": persist_norm,
                "rate_to_task_grad_ratio": rate_norm / task_norm,
                "persist_to_task_grad_ratio": persist_norm / task_norm,
            }
        )

    if len(rows) != CALIBRATION_BATCHES:
        raise RuntimeError(
            f"Calibration expected {CALIBRATION_BATCHES} batches, got {len(rows)}"
        )
    rate_ratios = np.asarray([row["rate_to_task_grad_ratio"] for row in rows], dtype=float)
    persist_ratios = np.asarray(
        [row["persist_to_task_grad_ratio"] for row in rows], dtype=float
    )
    if not np.all(np.isfinite(rate_ratios)) or not np.all(np.isfinite(persist_ratios)):
        raise RuntimeError("Non-finite gradient ratio during Exp7.1 calibration")
    median_rate = float(np.median(rate_ratios))
    median_persist = float(np.median(persist_ratios))
    lambda_rate = _calibrated_lambda(TARGET_RATE_GRAD_RATIO, median_rate)
    lambda_persist = _calibrated_lambda(TARGET_PERSIST_GRAD_RATIO, median_persist)
    return {
        "condition": spec.condition,
        "calibrated": True,
        "calibration_batches": CALIBRATION_BATCHES,
        "parameter_scope": "long_linear.weight only",
        "target_rate_grad_ratio": TARGET_RATE_GRAD_RATIO,
        "target_persist_grad_ratio": TARGET_PERSIST_GRAD_RATIO,
        "median_rate_to_task_grad_ratio": median_rate,
        "median_persist_to_task_grad_ratio": median_persist,
        "lambda_rate": lambda_rate,
        "lambda_persist": lambda_persist,
        "achieved_rate_grad_ratio": lambda_rate * median_rate,
        "achieved_persist_grad_ratio": lambda_persist * median_persist,
        "calibration_loader_seed": paired_seed(spec.seed, "calibration_loader"),
        "batches": rows,
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return exp70._classification_metrics(y_true, y_pred)


def evaluate_loader(
    model: SerialLongTauSNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            loss = wholecount_loss(trajectory["output_spikes"], lengths, y)
            logits = exp50.deployment_logits(
                trajectory["output_spikes"], lengths, OUTPUT_CAP
            )
            n = len(y)
            n_total += n
            loss_sum += float(loss.item()) * n
            labels.append(y.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    metrics = _classification_metrics(np.concatenate(labels), np.concatenate(predictions))
    metrics["loss"] = loss_sum / max(n_total, 1)
    return metrics


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def calibration_path(root: Path, spec: RunSpec) -> Path:
    return root / "calibrations" / f"{spec.key}.json"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def activity_path(root: Path, spec: RunSpec) -> Path:
    return root / "activities" / f"{spec.key}.npz"


def raster_path(root: Path, spec: RunSpec, layer: str) -> Path:
    return root / "rasters" / f"{spec.key}__{layer}.png"


def training_curve_path(root: Path, spec: RunSpec) -> Path:
    return root / "training_curves" / f"{spec.key}.png"


def current_trace_path(root: Path, spec: RunSpec) -> Path:
    return root / "current_traces" / f"{spec.key}__long_current.png"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _plot_training_history(
    history: pd.DataFrame,
    destination: Path,
    title: str,
    best_epoch: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(history["epoch"], history["train_task_loss"], label="train task")
    axes[0].plot(history["epoch"], history["val_task_loss"], label="val task")
    axes[0].plot(history["epoch"], history["train_total_loss"], label="train total")
    axes[0].axvline(best_epoch, linestyle="--", linewidth=1, label="best")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Task / total loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_rate_loss"], label="rate")
    axes[1].plot(history["epoch"], history["train_persist_loss"], label="persistence")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Raw regularizer")
    axes[1].set_title("Long-layer regularizers")
    axes[1].legend()

    axes[2].plot(history["epoch"], history["train_ba"], label="train BA")
    axes[2].plot(history["epoch"], history["val_ba"], label="val BA")
    axes[2].axvline(best_epoch, linestyle="--", linewidth=1, label="best")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Balanced accuracy")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Balanced accuracy")
    axes[2].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _weight_stats(model: SerialLongTauSNN) -> dict[str, float]:
    weight = model.long_linear.weight.detach()
    positive = weight[weight > 0]
    negative = weight[weight < 0]
    return {
        "long_weight_fro": float(torch.linalg.vector_norm(weight).item()),
        "long_weight_mean_abs": float(weight.abs().mean().item()),
        "long_weight_positive_mean": (
            float(positive.mean().item()) if positive.numel() else 0.0
        ),
        "long_weight_negative_abs_mean": (
            float(negative.abs().mean().item()) if negative.numel() else 0.0
        ),
    }


def _early_stop_triggered(epoch: int, best_epoch: int) -> bool:
    if epoch < MIN_EPOCHS + PATIENCE:
        return False
    reference = max(int(best_epoch), MIN_EPOCHS)
    return epoch - reference >= PATIENCE


def architecture_manifest(data: exp3.Data) -> dict[str, object]:
    return {
        "input": {
            "channels": EXPECTED_CHANNELS,
            "sampling_rate_hz": float(data.fs),
            "timesteps": EXPECTED_STEPS,
            "representation": "Raw64 30-channel event spike train",
        },
        "hidden_layers": [
            {
                "name": "short",
                "width": HIDDEN_WIDTH,
                "shift_syn": SHORT_SHIFT,
                "tau_syn_ms": exp70.tau_ms_from_shift(SHORT_SHIFT, data.fs),
            },
            {
                "name": "middle",
                "width": HIDDEN_WIDTH,
                "shift_syn": MIDDLE_SHIFT,
                "tau_syn_ms": exp70.tau_ms_from_shift(MIDDLE_SHIFT, data.fs),
            },
            {
                "name": "long",
                "width": HIDDEN_WIDTH,
                "shift_syn": LONG_SHIFT,
                "tau_syn_ms": exp70.tau_ms_from_shift(LONG_SHIFT, data.fs),
            },
        ],
        "hidden_tau_mem_ms": float(TAU_MEM_MS),
        "hidden_binary_cap": HIDDEN_CAP,
        "synaptic_update": "I_t = alpha*I_{t-1} + W*x_t",
        "fusion": "strict serial: output reads long spikes only; no short/middle bypass",
        "output": {
            "neurons": len(data.labels),
            "type": "short-tau binary LIF",
            "tau_mem_ms": float(OUTPUT_TAU_MEM_MS),
            "threshold": float(THRESHOLD),
            "cap": OUTPUT_CAP,
            "readout": "valid-length whole count",
        },
    }


def train_one(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
    force: bool,
) -> Path:
    validate_spec(spec)
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    if config.max_epochs != MAX_EPOCHS:
        raise ValueError(f"Exp7.1 max_epochs is frozen at {MAX_EPOCHS}")

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp3.seed_all(paired_seed(spec.seed, "model_init"))
    model = SerialLongTauSNN(len(data.labels), data.fs).to(device)

    calibration = calibrate_strength(spec, model, data, config)
    _save_json(calibration_path(config.results_dir, spec), calibration)
    lambda_rate = float(calibration["lambda_rate"])
    lambda_persist = float(calibration["lambda_persist"])

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loaders(data, spec.seed, config.batch_size, True)["train"]
    eval_loaders = make_loaders(data, spec.seed, config.batch_size, False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_loss = float("inf")
    stopped_epoch = MAX_EPOCHS
    stop_reason = "max_epochs"
    rows: list[dict[str, float | int | bool]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        scale = warmup_scale(epoch)
        task_sum = 0.0
        rate_sum = 0.0
        p32_sum = 0.0
        p84_sum = 0.0
        persist_sum = 0.0
        weighted_rate_sum = 0.0
        weighted_persist_sum = 0.0
        total_sum = 0.0
        n_total = 0

        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            task = wholecount_loss(trajectory["output_spikes"], lengths, y)
            rate, p32, p84, persist = long_regularization_terms(
                trajectory["long_spikes"], lengths, spec.condition
            )
            weighted_rate = scale * lambda_rate * rate
            weighted_persist = scale * lambda_persist * persist
            total = task + weighted_rate + weighted_persist
            total.backward()
            optimizer.step()

            n = len(y)
            n_total += n
            task_sum += float(task.detach().item()) * n
            rate_sum += float(rate.detach().item()) * n
            p32_sum += float(p32.detach().item()) * n
            p84_sum += float(p84.detach().item()) * n
            persist_sum += float(persist.detach().item()) * n
            weighted_rate_sum += float(weighted_rate.detach().item()) * n
            weighted_persist_sum += float(weighted_persist.detach().item()) * n
            total_sum += float(total.detach().item()) * n

        if n_total == 0:
            raise RuntimeError("Training loader produced no samples")

        train_metrics = evaluate_loader(model, eval_loaders["train"], device)
        val_metrics = evaluate_loader(model, eval_loaders["val"], device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        improved = val_ba > best_val_ba + 1e-12 or (
            abs(val_ba - best_val_ba) <= 1e-12 and val_loss < best_val_loss
        )
        if improved:
            best_epoch = epoch
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        weight_stats = _weight_stats(model)
        patience_counter = (
            max(0, epoch - max(best_epoch, MIN_EPOCHS)) if epoch >= MIN_EPOCHS else 0
        )
        rows.append(
            {
                "epoch": epoch,
                "warmup_scale": scale,
                "lambda_rate": lambda_rate,
                "lambda_persist": lambda_persist,
                "train_task_loss": task_sum / n_total,
                "train_rate_loss": rate_sum / n_total,
                "train_p32_loss": p32_sum / n_total,
                "train_p84_loss": p84_sum / n_total,
                "train_persist_loss": persist_sum / n_total,
                "train_weighted_rate_loss": weighted_rate_sum / n_total,
                "train_weighted_persist_loss": weighted_persist_sum / n_total,
                "train_total_loss": total_sum / n_total,
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": val_ba,
                "val_task_loss": val_loss,
                "best_so_far": bool(improved),
                "patience_counter": patience_counter,
                **weight_stats,
            }
        )

        if _early_stop_triggered(epoch, best_epoch):
            stopped_epoch = epoch
            stop_reason = "patience_30_after_min20_gate"
            break

    if best_state is None or best_epoch < 1:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "calibration": calibration,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "stop_reason": stop_reason,
            "best_val_ba": best_val_ba,
            "best_val_loss": best_val_loss,
            "model_state_dict": best_state,
            "labels": data.labels,
            "split": data.split,
            "architecture": architecture_manifest(data),
        },
        destination,
    )
    history = pd.DataFrame(rows)
    hpath = history_path(config.results_dir, spec)
    hpath.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(hpath, index=False)
    _plot_training_history(
        history,
        training_curve_path(config.results_dir, spec),
        spec.key,
        best_epoch,
    )
    return destination


def load_model(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
) -> tuple[SerialLongTauSNN, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(path)
    device = torch.device(config.device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong experiment id in {path}")
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong protocol version in {path}")
    if checkpoint.get("spec") != asdict(spec):
        raise ValueError(f"Checkpoint identity mismatch for {spec.key}")
    model = SerialLongTauSNN(len(data.labels), data.fs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _run_lengths(binary: np.ndarray) -> np.ndarray:
    x = np.asarray(binary, dtype=np.int8).reshape(-1)
    if x.size == 0:
        return np.empty(0, dtype=np.int64)
    padded = np.concatenate(([0], x, [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return (ends - starts).astype(np.int64, copy=False)


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def long_dynamics_metrics(
    model: SerialLongTauSNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    full_spikes = 0.0
    valid_spikes = 0.0
    padding_spikes = 0.0
    full_steps = 0.0
    valid_steps = 0.0
    padding_steps = 0.0
    current_sq_full = 0.0
    current_sq_valid = 0.0
    current_sq_padding = 0.0
    full_runs: list[float] = []
    valid_runs: list[float] = []
    full_max_per_neuron: list[float] = []
    valid_max_per_neuron: list[float] = []

    with torch.no_grad():
        for X, _, lengths in loader:
            X = X.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X, capture_states=True)
            spikes = trajectory["long_spikes"]
            current = trajectory["long_current"]
            B, T, N = spikes.shape
            valid = exp50.valid_mask(lengths, T).to(spikes.dtype).unsqueeze(-1)
            padding = 1.0 - valid

            full_spikes += float(spikes.sum().item())
            valid_spikes += float((spikes * valid).sum().item())
            padding_spikes += float((spikes * padding).sum().item())
            full_steps += float(B * T * N)
            valid_steps += float(valid.sum().item() * N)
            padding_steps += float(padding.sum().item() * N)

            current_sq_full += float(current.square().sum().item())
            current_sq_valid += float((current.square() * valid).sum().item())
            current_sq_padding += float((current.square() * padding).sum().item())

            spikes_np = spikes.cpu().numpy()
            lengths_np = lengths.cpu().numpy().astype(int)
            for b in range(B):
                valid_length = int(lengths_np[b])
                for n in range(N):
                    fr = _run_lengths(spikes_np[b, :, n])
                    vr = _run_lengths(spikes_np[b, :valid_length, n])
                    full_runs.extend(fr.astype(float).tolist())
                    valid_runs.extend(vr.astype(float).tolist())
                    full_max_per_neuron.append(float(fr.max()) if fr.size else 0.0)
                    valid_max_per_neuron.append(float(vr.max()) if vr.size else 0.0)

    return {
        "long_full_firing_fraction": full_spikes / max(full_steps, 1.0),
        "long_valid_firing_fraction": valid_spikes / max(valid_steps, 1.0),
        "long_padding_firing_fraction": padding_spikes / max(padding_steps, 1.0),
        "long_current_rms_full": math.sqrt(current_sq_full / max(full_steps, 1.0)),
        "long_current_rms_valid": math.sqrt(current_sq_valid / max(valid_steps, 1.0)),
        "long_current_rms_padding": math.sqrt(
            current_sq_padding / max(padding_steps, 1.0)
        ),
        "long_mean_run_length_full": _safe_mean(full_runs),
        "long_mean_run_length_valid": _safe_mean(valid_runs),
        "long_mean_max_run_per_neuron_full": _safe_mean(full_max_per_neuron),
        "long_mean_max_run_per_neuron_valid": _safe_mean(valid_max_per_neuron),
        "long_p95_max_run_per_neuron_full": _safe_percentile(full_max_per_neuron, 95.0),
        "long_p95_max_run_per_neuron_valid": _safe_percentile(valid_max_per_neuron, 95.0),
        "long_global_max_run_full": max(full_max_per_neuron, default=0.0),
        "long_global_max_run_valid": max(valid_max_per_neuron, default=0.0),
    }


def _visualization_sample(data: exp3.Data) -> tuple[int, int]:
    lengths = np.asarray(data.lva, dtype=np.int64)
    median = float(np.median(lengths))
    index = int(np.argmin(np.abs(lengths.astype(np.float64) - median)))
    return index, int(lengths[index])


def _plot_raster(
    spikes: np.ndarray,
    destination: Path,
    valid_length: int,
    title: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(spikes.T, aspect="auto", interpolation="nearest", origin="lower")
    ax.axvline(valid_length, linestyle="--", linewidth=1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_fixed_sample_activity(
    spec: RunSpec,
    model: SerialLongTauSNN,
    data: exp3.Data,
    config: Config,
) -> None:
    device = torch.device(config.device)
    index, valid_length = _visualization_sample(data)
    sample = torch.tensor(data.Xva[index:index + 1], dtype=torch.float32, device=device)
    sample[:, valid_length:] = 0.0
    with torch.no_grad():
        trajectory = model.forward_trajectory(sample, capture_states=True)

    hidden = trajectory["hidden_spikes"]
    output = trajectory["output_spikes"][0].cpu().numpy()
    long_current = trajectory["long_current"][0].cpu().numpy()
    long_membrane = trajectory["long_membrane"][0].cpu().numpy()
    payload: dict[str, np.ndarray] = {
        "input_spikes": sample[0].cpu().numpy(),
        "output_spikes": output,
        "long_current": long_current,
        "long_membrane": long_membrane,
        "valid_length": np.asarray(valid_length, dtype=np.int64),
        "label": np.asarray(int(data.yva[index]), dtype=np.int64),
    }
    for layer in ("short", "middle", "long"):
        spikes = hidden[layer][0].cpu().numpy()
        payload[f"{layer}_spikes"] = spikes
        _plot_raster(
            spikes,
            raster_path(config.results_dir, spec, layer),
            valid_length,
            f"{spec.key} - {layer}",
        )
    _plot_raster(
        output,
        raster_path(config.results_dir, spec, "output"),
        valid_length,
        f"{spec.key} - output",
    )

    cpath = current_trace_path(config.results_dir, spec)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(np.arange(long_current.shape[0]), np.mean(np.abs(long_current), axis=1))
    ax.axvline(valid_length, linestyle="--", linewidth=1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean |I_long|")
    ax.set_title(f"{spec.key} - long synaptic current")
    fig.tight_layout()
    fig.savefig(cpath, dpi=160, bbox_inches="tight")
    plt.close(fig)

    apath = activity_path(config.results_dir, spec)
    apath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(apath, **payload)


def evaluate_one(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    model, checkpoint = load_model(spec, data, config)
    device = torch.device(config.device)
    loaders = make_loaders(data, spec.seed, config.batch_size, False)
    metrics = {split: evaluate_loader(model, loader, device) for split, loader in loaders.items()}
    dynamics = {
        split: long_dynamics_metrics(model, loader, device)
        for split, loader in loaders.items()
    }
    weights = _weight_stats(model)
    save_fixed_sample_activity(spec, model, data, config)
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "metrics": metrics,
        "dynamics": dynamics,
        "weight_stats": weights,
        "calibration": checkpoint["calibration"],
        "best_epoch": int(checkpoint["best_epoch"]),
        "stopped_epoch": int(checkpoint["stopped_epoch"]),
        "stop_reason": checkpoint["stop_reason"],
        "best_val_ba": float(checkpoint["best_val_ba"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "architecture": checkpoint["architecture"],
        "provenance": {
            "split_seed": int(exp3.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "checkpoint_selection": "highest validation BA; lower validation WholeCount CE tie-break",
            "regularizer_target": "long hidden spikes only; gradients remain end-to-end",
        },
    }
    _save_json(destination, payload)
    return payload


def run_one(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    train_one(spec, data, config, force)
    return evaluate_one(spec, data, config, force)


def _row_from_evaluation(payload: dict[str, object]) -> dict[str, object]:
    spec = payload["spec"]
    metrics = payload["metrics"]
    dynamics = payload["dynamics"]
    calibration = payload["calibration"]
    weight_stats = payload["weight_stats"]
    test_dyn = dynamics["test"]
    return {
        "condition": spec["condition"],
        "seed": int(spec["seed"]),
        "best_epoch": int(payload["best_epoch"]),
        "stopped_epoch": int(payload["stopped_epoch"]),
        "train_ba": float(metrics["train"]["balanced_accuracy"]),
        "val_ba": float(metrics["val"]["balanced_accuracy"]),
        "test_ba": float(metrics["test"]["balanced_accuracy"]),
        "test_accuracy": float(metrics["test"]["accuracy"]),
        "test_macro_f1": float(metrics["test"]["macro_f1"]),
        "lambda_rate": float(calibration["lambda_rate"]),
        "lambda_persist": float(calibration["lambda_persist"]),
        **{key: float(value) for key, value in test_dyn.items()},
        **{key: float(value) for key, value in weight_stats.items()},
    }


def _summary_plot(summary: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = summary.set_index("condition").reindex(CONDITIONS).reset_index()
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(
        x,
        frame["test_ba_mean"],
        yerr=frame["test_ba_std"].fillna(0.0),
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(frame["condition"], rotation=20, ha="right")
    ax.set_ylabel("Test balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Exp7.1 classification")
    fig.tight_layout()
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _dynamics_plot(dynamics: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = dynamics.set_index("condition").reindex(CONDITIONS).reset_index()
    x = np.arange(len(frame))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, frame["long_valid_firing_fraction_mean"], width, label="valid")
    ax.bar(x, frame["long_full_firing_fraction_mean"], width, label="full")
    ax.bar(x + width, frame["long_padding_firing_fraction_mean"], width, label="padding")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["condition"], rotation=20, ha="right")
    ax.set_ylabel("Long-layer firing fraction")
    ax.set_title("Exp7.1 long-layer activity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)


def finalize(repo_root: Path) -> dict[str, object]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_row_from_evaluation(payload))
        calibration = payload["calibration"]
        calibration_rows.append(
            {
                "condition": spec.condition,
                "seed": spec.seed,
                "lambda_rate": float(calibration["lambda_rate"]),
                "lambda_persist": float(calibration["lambda_persist"]),
                "target_rate_grad_ratio": float(calibration["target_rate_grad_ratio"]),
                "target_persist_grad_ratio": float(calibration["target_persist_grad_ratio"]),
                "achieved_rate_grad_ratio": float(
                    calibration.get("achieved_rate_grad_ratio", 0.0)
                ),
                "achieved_persist_grad_ratio": float(
                    calibration.get("achieved_persist_grad_ratio", 0.0)
                ),
            }
        )

    runs = pd.DataFrame(rows)
    runs.to_csv(root / "runs.csv", index=False)
    summary = (
        runs.groupby("condition")
        .agg(
            n_runs=("test_ba", "count"),
            train_ba_mean=("train_ba", "mean"),
            train_ba_std=("train_ba", "std"),
            val_ba_mean=("val_ba", "mean"),
            val_ba_std=("val_ba", "std"),
            test_ba_mean=("test_ba", "mean"),
            test_ba_std=("test_ba", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_macro_f1_mean=("test_macro_f1", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            stopped_epoch_mean=("stopped_epoch", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(root / "summary.csv", index=False)

    dynamics_columns = [
        "long_full_firing_fraction",
        "long_valid_firing_fraction",
        "long_padding_firing_fraction",
        "long_current_rms_full",
        "long_current_rms_valid",
        "long_current_rms_padding",
        "long_mean_run_length_full",
        "long_mean_run_length_valid",
        "long_mean_max_run_per_neuron_full",
        "long_mean_max_run_per_neuron_valid",
        "long_p95_max_run_per_neuron_full",
        "long_p95_max_run_per_neuron_valid",
        "long_global_max_run_full",
        "long_global_max_run_valid",
        "long_weight_fro",
        "long_weight_mean_abs",
        "long_weight_positive_mean",
        "long_weight_negative_abs_mean",
    ]
    agg_spec: dict[str, tuple[str, str]] = {}
    for column in dynamics_columns:
        agg_spec[f"{column}_mean"] = (column, "mean")
        agg_spec[f"{column}_std"] = (column, "std")
    dynamics_summary = runs.groupby("condition").agg(**agg_spec).reset_index()
    dynamics_summary.to_csv(root / "dynamics_summary.csv", index=False)

    contrast_defs = {
        "all_loss_masked_minus_no_reg": (ALL_LOSS_MASKED, NO_REG),
        "wholecount_only_masked_minus_no_reg": (WHOLECOUNT_ONLY_MASKED, NO_REG),
        "wholecount_only_masked_minus_all_loss_masked": (
            WHOLECOUNT_ONLY_MASKED,
            ALL_LOSS_MASKED,
        ),
    }
    contrast_metrics = [
        "test_ba",
        "long_full_firing_fraction",
        "long_padding_firing_fraction",
        "long_mean_max_run_per_neuron_full",
        "long_p95_max_run_per_neuron_full",
        "long_current_rms_full",
        "long_weight_fro",
    ]
    paired_rows: list[dict[str, object]] = []
    indexed = runs.set_index(["condition", "seed"])
    for seed in TRAIN_SEEDS:
        for contrast, (lhs, rhs) in contrast_defs.items():
            row: dict[str, object] = {"seed": seed, "contrast": contrast}
            for metric in contrast_metrics:
                row[f"{metric}_delta"] = float(
                    indexed.loc[(lhs, seed), metric] - indexed.loc[(rhs, seed), metric]
                )
            paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(root / "paired_condition_deltas.csv", index=False)

    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_summary = (
        calibration_frame.groupby("condition")
        .agg(
            lambda_rate_mean=("lambda_rate", "mean"),
            lambda_rate_std=("lambda_rate", "std"),
            lambda_persist_mean=("lambda_persist", "mean"),
            lambda_persist_std=("lambda_persist", "std"),
            achieved_rate_grad_ratio_mean=("achieved_rate_grad_ratio", "mean"),
            achieved_persist_grad_ratio_mean=("achieved_persist_grad_ratio", "mean"),
        )
        .reset_index()
    )
    calibration_summary.to_csv(root / "calibration_summary.csv", index=False)

    plots = root / "summary_plots"
    _summary_plot(summary, plots / "test_ba_by_condition.png")
    _dynamics_plot(dynamics_summary, plots / "long_firing_by_condition.png")

    data = prepare_data(repo_root)
    vis_index, vis_length = _visualization_sample(data)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": "Can long-layer-specific rate and anti-persistence regularization suppress pathological sustained firing in a serial s4/s5/s6 end-to-end SNN without killing useful long activity?",
        "conditions": list(CONDITIONS),
        "seeds": list(TRAIN_SEEDS),
        "run_count": len(run_specs()),
        "backbone": "Raw30 -> S(s4) -> M(s5) -> L(s6) -> short-tau output LIF -> valid WholeCount",
        "objective": "L_WC + warmup * [lambda_r L_rate^L + lambda_p (L_3,2^L + 0.5 L_8,4^L)]",
        "mask_policy": {
            NO_REG: "WholeCount valid; no regularizer",
            ALL_LOSS_MASKED: "WholeCount, rate and persistence all use valid region; persistence windows must lie completely within valid length",
            WHOLECOUNT_ONLY_MASKED: "WholeCount uses valid region; rate and persistence use the full 256-step trajectory including valid/padding boundary windows",
        },
        "gradient_calibration": {
            "batches": CALIBRATION_BATCHES,
            "parameter_scope": "long_linear.weight only",
            "rate_target": TARGET_RATE_GRAD_RATIO,
            "persistence_target": TARGET_PERSIST_GRAD_RATIO,
            "calibrated_separately_per regularized condition and seed": True,
        },
        "training": {
            "max_epochs": MAX_EPOCHS,
            "minimum_gate_epochs": MIN_EPOCHS,
            "patience_after_gate": PATIENCE,
            "earliest_early_stop_epoch": MIN_EPOCHS + PATIENCE,
            "regularizer_warmup_epochs": WARMUP_EPOCHS,
        },
        "fixed_visualization_sample": {
            "validation_index": vis_index,
            "valid_length": vis_length,
        },
        "notebook_policy": "aggregate-only: notebook consumes summary.csv, dynamics_summary.csv, paired_condition_deltas.csv and calibration_summary.csv; no per-run histories/evaluations/rasters",
    }
    _save_json(root / "manifest.json", manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 7.1 long-tau anti-persistence")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-runs")
    run_parser = sub.add_parser("run-one")
    run_parser.add_argument("--array-task-id", type=int, required=True)
    run_parser.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    repo_root = find_repo_root(args.repo_root)
    config = Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size),
        threads=int(args.threads),
    )
    if args.command == "list-runs":
        for index, spec in enumerate(run_specs()):
            print(index, spec.key)
        return
    if args.command == "finalize":
        print(json.dumps(finalize(repo_root), indent=2))
        return

    data = prepare_data(repo_root)
    if args.command == "run-one":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        payload = run_one(specs[args.array_task_id], data, config, force=args.force)
        print(json.dumps(payload, indent=2))
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
