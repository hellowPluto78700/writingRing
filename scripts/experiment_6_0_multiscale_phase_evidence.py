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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_3_0_1_single_tau_objective_comparison as exp3
from scripts import experiment_3_0_2_hidden_multitau_architectures as exp302
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50


EXPERIMENT_ID = "experiment_6_0_multiscale_phase_evidence"
PROTOCOL_VERSION = "single_hidden_multiscale_phase_evidence_v1"

OBJECTIVES = (
    "whole_count",
    "whole_count_hce",
    "whole_count_contextual_gain",
)
MEM_SHIFTS = (1, 2, 3)
TRAIN_SEEDS = (11, 23, 37)
SYN_SHIFTS = (4, 5, 6)
GROUP_COUNTS = (43, 43, 42)
HIDDEN_WIDTH = sum(GROUP_COUNTS)
OUTPUT_CAP = 1
HIDDEN_CAP = 1

EPOCHS = 100
BATCH_SIZE = exp3.BATCH_SIZE
LR = exp3.LR
WEIGHT_DECAY = 0.0
LAMBDA_HCE = 1.0
LAMBDA_CG = 1.0
FIXED250_MS = 250.0
VIS_STEPS = 256
EXPECTED_FS = 64.0
EXPECTED_CHANNELS = 30
EXPECTED_SPLIT_SEED = 12345
OUTPUT_TAU_MEM_MS = exp3.TAU_MEM_MS
THRESHOLD = exp3.THRESHOLD
SURROGATE_SLOPE = exp3.SURROGATE_SLOPE


@dataclass(frozen=True)
class RunSpec:
    objective: str
    mem_shift: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.objective}__mem{self.mem_shift}__seed{self.seed}"


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


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(objective=objective, mem_shift=mem_shift, seed=seed)
        for objective in OBJECTIVES
        for mem_shift in MEM_SHIFTS
        for seed in TRAIN_SEEDS
    ]


def _validate_spec(spec: RunSpec) -> None:
    if spec.objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {spec.objective}")
    if spec.mem_shift not in MEM_SHIFTS:
        raise ValueError(f"Unknown hidden membrane shift: {spec.mem_shift}")
    if spec.seed not in TRAIN_SEEDS:
        raise ValueError(f"Unknown training seed: {spec.seed}")


def paired_seed(spec: RunSpec, role: str) -> int:
    """Pair model initialization and loader order across objectives/mem shifts."""
    return exp3.dseed(spec.seed, "exp6_0_paired", role)


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def learning_curve_path(root: Path, spec: RunSpec) -> Path:
    return root / "learning_curves" / f"{spec.key}.png"


def component_curve_path(root: Path, spec: RunSpec) -> Path:
    return root / "loss_components" / f"{spec.key}.png"


def activity_path(root: Path, spec: RunSpec, reversed_order: bool = False) -> Path:
    suffix = "__reversed" if reversed_order else ""
    return root / "activities" / f"{spec.key}{suffix}.npz"


def raster_path(root: Path, spec: RunSpec, reversed_order: bool = False) -> Path:
    suffix = "__reversed" if reversed_order else ""
    return root / "rasters" / f"{spec.key}{suffix}.png"


def baseline_path(root: Path) -> Path:
    return root / "baseline_fixed250_linear.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def decay_from_shift(shift: int) -> float:
    if shift <= 0:
        raise ValueError("shift must be positive")
    return float(1.0 - 2.0 ** (-int(shift)))


def tau_ms_from_shift(shift: int, fs: float) -> float:
    decay = decay_from_shift(shift)
    return float(-(1000.0 / float(fs)) / math.log(decay))


def group_slices() -> tuple[slice, slice, slice]:
    c4, c5, c6 = GROUP_COUNTS
    return (
        slice(0, c4),
        slice(c4, c4 + c5),
        slice(c4 + c5, c4 + c5 + c6),
    )


def syn_alpha_vector() -> torch.Tensor:
    values: list[float] = []
    for shift, count in zip(SYN_SHIFTS, GROUP_COUNTS, strict=True):
        values.extend([decay_from_shift(shift)] * count)
    return torch.tensor(values, dtype=torch.float32)


def fixed250_steps(fs: float) -> int:
    return int(np.rint(FIXED250_MS * float(fs) / 1000.0))


def prepare_data(repo_root: Path) -> exp3.Data:
    data = exp3.prepare_data(repo_root)
    if abs(float(data.fs) - EXPECTED_FS) > 1e-12:
        raise ValueError(f"Exp6.0 requires {EXPECTED_FS:g} Hz data, got {data.fs}")
    if exp3.EVENT_CHANNELS != EXPECTED_CHANNELS:
        raise ValueError(
            f"Exp6.0 requires {EXPECTED_CHANNELS} raw event channels, "
            f"repo contract reports {exp3.EVENT_CHANNELS}"
        )
    if int(EXPECTED_SPLIT_SEED) != int(exp3.SPLIT_SEED):
        raise ValueError("Exp6.0 split-seed contract changed")
    if fixed250_steps(data.fs) != 16:
        raise ValueError("At 64 Hz Exp6.0 requires 250 ms == 16 timesteps")
    if data.Xtr.shape[1] != VIS_STEPS:
        raise ValueError(
            f"Exp6.0 requires globally padded {VIS_STEPS}-timestep inputs, "
            f"got {data.Xtr.shape[1]}"
        )
    return data


def _partitions(
    data: exp3.Data,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }


def _loader(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def make_loaders(
    data: exp3.Data,
    spec: RunSpec,
    config: Config,
    train_shuffle: bool,
) -> dict[str, DataLoader]:
    return {
        split: _loader(
            X,
            y,
            lengths,
            config.batch_size,
            train_shuffle if split == "train" else False,
            paired_seed(spec, f"{split}_loader"),
        )
        for split, (X, y, lengths) in _partitions(data).items()
    }


class SingleHiddenMultiTauSNN(nn.Module):
    """Raw30 -> 128 heterogeneous normalized-synapse LIF -> K binary LIF."""

    def __init__(self, n_classes: int, fs: float, mem_shift: int) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        self.mem_shift = int(mem_shift)
        self.input_hidden = nn.Linear(EXPECTED_CHANNELS, HIDDEN_WIDTH, bias=False)
        self.output_linear = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=False)

        hidden_beta = decay_from_shift(self.mem_shift)
        output_beta = math.exp(-(1000.0 / self.fs) / float(OUTPUT_TAU_MEM_MS))
        self.hidden_lif = exp401.MacroMultiSpikeLIF(
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
        self.register_buffer("syn_alpha", syn_alpha_vector())

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,C] input, got {tuple(x.shape)}")
        batch, n_steps, channels = x.shape
        if channels != EXPECTED_CHANNELS:
            raise ValueError(f"Expected {EXPECTED_CHANNELS} channels, got {channels}")

        syn = torch.zeros(batch, HIDDEN_WIDTH, device=x.device, dtype=x.dtype)
        hidden_mem = torch.zeros_like(syn)
        output_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)

        hidden_spikes: list[torch.Tensor] = []
        output_spikes: list[torch.Tensor] = []
        pre_output_evidence: list[torch.Tensor] = []

        alpha = self.syn_alpha.to(device=x.device, dtype=x.dtype)
        one_minus_alpha = 1.0 - alpha
        for step in range(n_steps):
            projected = self.input_hidden(x[:, step])
            syn = alpha * syn + one_minus_alpha * projected
            hidden_spike, hidden_mem, _ = self.hidden_lif(syn, hidden_mem)
            evidence = self.output_linear(hidden_spike)
            output_spike, output_mem, _ = self.output_lif(evidence, output_mem)
            hidden_spikes.append(hidden_spike)
            pre_output_evidence.append(evidence)
            output_spikes.append(output_spike)

        return {
            "hidden_spikes": torch.stack(hidden_spikes, dim=1),
            "pre_output_evidence": torch.stack(pre_output_evidence, dim=1),
            "output_spikes": torch.stack(output_spikes, dim=1),
        }

    def group_evidence(self, hidden_spikes: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if hidden_spikes.shape[-1] != HIDDEN_WIDTH:
            raise ValueError("Hidden width mismatch")
        groups: list[torch.Tensor] = []
        for sl in group_slices():
            groups.append(
                F.linear(hidden_spikes[..., sl], self.output_linear.weight[:, sl])
            )
        return tuple(groups)


def valid_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.arange(steps, device=lengths.device)[None, :] < lengths[:, None]


def class_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("class_margin expects [B,K] logits")
    correct = logits.gather(1, y[:, None]).squeeze(1)
    blocked = F.one_hot(y, num_classes=logits.shape[1]).bool()
    other = logits.masked_fill(blocked, -torch.inf)
    return correct - torch.logsumexp(other, dim=1)


def reverse_complete_bins(
    x: torch.Tensor,
    lengths: torch.Tensor,
    bin_steps: int,
) -> torch.Tensor:
    """Reverse complete valid 250-ms bins; preserve within-bin order/remainder/padding."""
    if bin_steps <= 0:
        raise ValueError("bin_steps must be positive")
    out = x.clone()
    for index in range(x.shape[0]):
        valid = int(lengths[index].item())
        if valid < 0 or valid > x.shape[1]:
            raise ValueError("Invalid sequence length")
        n_full = valid // bin_steps
        if n_full <= 1:
            continue
        full_steps = n_full * bin_steps
        bins = x[index, :full_steps].reshape(n_full, bin_steps, x.shape[-1])
        out[index, :full_steps] = bins.flip(0).reshape(full_steps, x.shape[-1])
    return out


def _whole_count_ce(
    output_spikes: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    logits = exp50.deployment_ce_logits(output_spikes, lengths, OUTPUT_CAP)
    return F.cross_entropy(logits, y)


def _hce_loss(
    original_evidence: torch.Tensor,
    reversed_evidence: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    orig = exp50.valid_mean(original_evidence, lengths)
    rev = exp50.valid_mean(reversed_evidence, lengths)
    orig_margin = class_margin(orig, y)
    rev_margin = class_margin(rev, y)
    loss = F.softplus(rev_margin - orig_margin).mean()
    return loss, orig_margin.mean(), rev_margin.mean()


def _contextual_gain_loss(
    model: SingleHiddenMultiTauSNN,
    hidden_spikes: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
    bin_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e4, e5, e6 = model.group_evidence(hidden_spikes)
    mid_gains: list[torch.Tensor] = []
    long_gains: list[torch.Tensor] = []

    for index in range(hidden_spikes.shape[0]):
        valid = int(lengths[index].item())
        n_full = valid // bin_steps
        if n_full <= 0:
            continue
        full_steps = n_full * bin_steps
        label = y[index].expand(n_full)

        b4 = e4[index, :full_steps].reshape(n_full, bin_steps, -1).sum(dim=1)
        b5 = e5[index, :full_steps].reshape(n_full, bin_steps, -1).sum(dim=1)
        b6 = e6[index, :full_steps].reshape(n_full, bin_steps, -1).sum(dim=1)

        m4 = class_margin(b4, label)
        m45 = class_margin(b4 + b5, label)
        m456 = class_margin(b4 + b5 + b6, label)
        mid_gains.append((m45 - m4).mean())
        long_gains.append((m456 - m45).mean())

    if not mid_gains:
        zero = hidden_spikes.sum() * 0.0
        return zero, zero.detach(), zero.detach()

    mid_gain = torch.stack(mid_gains).mean()
    long_gain = torch.stack(long_gains).mean()
    loss = F.softplus(-mid_gain) + F.softplus(-long_gain)
    return loss, mid_gain, long_gain


def objective_components(
    model: SingleHiddenMultiTauSNN,
    X: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    objective: str,
    bin_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    trajectory = model.forward_trajectory(X)
    whole_ce = _whole_count_ce(trajectory["output_spikes"], lengths, y)
    zero = whole_ce.detach() * 0.0
    components: dict[str, torch.Tensor] = {
        "wholecount_ce": whole_ce,
        "hce_loss": zero,
        "cg_loss": zero,
        "orig_margin": torch.tensor(float("nan"), device=X.device),
        "rev_margin": torch.tensor(float("nan"), device=X.device),
        "mid_gain": torch.tensor(float("nan"), device=X.device),
        "long_gain": torch.tensor(float("nan"), device=X.device),
    }

    if objective == "whole_count":
        total = whole_ce
    elif objective == "whole_count_hce":
        reversed_X = reverse_complete_bins(X, lengths, bin_steps)
        reversed_trajectory = model.forward_trajectory(reversed_X)
        hce, orig_margin, rev_margin = _hce_loss(
            trajectory["pre_output_evidence"],
            reversed_trajectory["pre_output_evidence"],
            lengths,
            y,
        )
        components.update(
            {
                "hce_loss": hce,
                "orig_margin": orig_margin,
                "rev_margin": rev_margin,
            }
        )
        total = whole_ce + LAMBDA_HCE * hce
    elif objective == "whole_count_contextual_gain":
        cg, mid_gain, long_gain = _contextual_gain_loss(
            model,
            trajectory["hidden_spikes"],
            lengths,
            y,
            bin_steps,
        )
        components.update(
            {
                "cg_loss": cg,
                "mid_gain": mid_gain,
                "long_gain": long_gain,
            }
        )
        total = whole_ce + LAMBDA_CG * cg
    else:
        raise ValueError(f"Unknown objective: {objective}")
    components["total_loss"] = total
    return total, components, trajectory


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _mean_or_nan(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def evaluate_loader(
    model: SingleHiddenMultiTauSNN,
    loader: DataLoader,
    device: torch.device,
    objective: str,
    bin_steps: int,
) -> dict[str, float]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    sums: dict[str, float] = {
        "total_loss": 0.0,
        "wholecount_ce": 0.0,
        "hce_loss": 0.0,
        "cg_loss": 0.0,
    }
    extras: dict[str, list[float]] = {
        "orig_margin": [],
        "rev_margin": [],
        "mid_gain": [],
        "long_gain": [],
    }
    n_total = 0

    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            _, components, trajectory = objective_components(
                model, X, y, lengths, objective, bin_steps
            )
            logits = exp50.deployment_logits(
                trajectory["output_spikes"], lengths, OUTPUT_CAP
            )
            n = len(y)
            n_total += n
            for name in sums:
                sums[name] += float(components[name].item()) * n
            for name in extras:
                value = float(components[name].item())
                if np.isfinite(value):
                    extras[name].append(value)
            labels.append(y.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())

    metrics = _classification_metrics(
        np.concatenate(labels),
        np.concatenate(predictions),
    )
    for name, value in sums.items():
        metrics[name] = value / max(n_total, 1)
    for name, values in extras.items():
        metrics[name] = _mean_or_nan(values)
    return metrics


def _new_model(spec: RunSpec, data: exp3.Data) -> SingleHiddenMultiTauSNN:
    _validate_spec(spec)
    return SingleHiddenMultiTauSNN(
        n_classes=len(data.labels),
        fs=data.fs,
        mem_shift=spec.mem_shift,
    )


def architecture_manifest(
    spec: RunSpec,
    data: exp3.Data,
) -> dict[str, object]:
    return {
        "input": {
            "channels": EXPECTED_CHANNELS,
            "sampling_rate_hz": float(data.fs),
            "representation": "Raw64 event spike train; no pre-SNN pooling/scaling",
        },
        "hidden": {
            "width": HIDDEN_WIDTH,
            "binary_spike": True,
            "synaptic_update": "I_t = alpha*I_{t-1} + (1-alpha)*W*x_t",
            "syn_shift_groups": list(SYN_SHIFTS),
            "group_neurons": list(GROUP_COUNTS),
            "tau_syn_ms": [
                tau_ms_from_shift(shift, data.fs) for shift in SYN_SHIFTS
            ],
            "mem_shift": spec.mem_shift,
            "tau_mem_ms": tau_ms_from_shift(spec.mem_shift, data.fs),
            "threshold": THRESHOLD,
        },
        "output": {
            "neurons": len(data.labels),
            "binary_spike": True,
            "tau_mem_ms": float(OUTPUT_TAU_MEM_MS),
            "readout": "valid-length whole count",
        },
        "objective": spec.objective,
        "fixed250_steps": fixed250_steps(data.fs),
    }


def _plot_history(history: pd.DataFrame, root: Path, spec: RunSpec) -> None:
    path = learning_curve_path(root, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_total_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_total_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss")
    axes[0].set_title(spec.key)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_ba"], label="train")
    axes[1].plot(history["epoch"], history["val_ba"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    component_path = component_curve_path(root, spec)
    component_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["epoch"], history["train_wholecount_ce"], label="train whole-count CE")
    ax.plot(history["epoch"], history["val_wholecount_ce"], label="val whole-count CE")
    if spec.objective == "whole_count_hce":
        ax.plot(history["epoch"], history["train_hce_loss"], label="train HCE")
        ax.plot(history["epoch"], history["val_hce_loss"], label="val HCE")
    elif spec.objective == "whole_count_contextual_gain":
        ax.plot(history["epoch"], history["train_cg_loss"], label="train CG")
        ax.plot(history["epoch"], history["val_cg_loss"], label="val CG")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss component")
    ax.set_title(spec.key)
    ax.legend()
    fig.tight_layout()
    fig.savefig(component_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_one(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
    force: bool,
) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp3.seed_all(paired_seed(spec, "model_init"))
    model = _new_model(spec, data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loaders(data, spec, config, train_shuffle=True)["train"]
    eval_loaders = make_loaders(data, spec, config, train_shuffle=False)
    bin_steps = fixed250_steps(data.fs)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_whole_ce = float("inf")
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = objective_components(
                model, X, y, lengths, spec.objective, bin_steps
            )
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_loader(
            model, eval_loaders["train"], device, spec.objective, bin_steps
        )
        val_metrics = evaluate_loader(
            model, eval_loaders["val"], device, spec.objective, bin_steps
        )
        rows.append(
            {
                "epoch": epoch,
                "train_total_loss": train_metrics["total_loss"],
                "val_total_loss": val_metrics["total_loss"],
                "train_wholecount_ce": train_metrics["wholecount_ce"],
                "val_wholecount_ce": val_metrics["wholecount_ce"],
                "train_hce_loss": train_metrics["hce_loss"],
                "val_hce_loss": val_metrics["hce_loss"],
                "train_cg_loss": train_metrics["cg_loss"],
                "val_cg_loss": val_metrics["cg_loss"],
                "train_orig_margin": train_metrics["orig_margin"],
                "val_orig_margin": val_metrics["orig_margin"],
                "train_rev_margin": train_metrics["rev_margin"],
                "val_rev_margin": val_metrics["rev_margin"],
                "train_mid_gain": train_metrics["mid_gain"],
                "val_mid_gain": val_metrics["mid_gain"],
                "train_long_gain": train_metrics["long_gain"],
                "val_long_gain": val_metrics["long_gain"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_ba": train_metrics["balanced_accuracy"],
                "val_ba": val_metrics["balanced_accuracy"],
            }
        )

        val_ba = float(val_metrics["balanced_accuracy"])
        val_whole_ce = float(val_metrics["wholecount_ce"])
        improved = val_ba > best_val_ba + 1e-12 or (
            abs(val_ba - best_val_ba) <= 1e-12
            and val_whole_ce < best_val_whole_ce
        )
        if improved:
            best_epoch = epoch
            best_val_ba = val_ba
            best_val_whole_ce = val_whole_ce
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(f"No best checkpoint selected for {spec.key}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_ba,
            "best_val_wholecount_ce": best_val_whole_ce,
            "model_state_dict": best_state,
            "labels": [str(value) for value in data.labels],
            "split": data.split,
            "architecture": architecture_manifest(spec, data),
        },
        destination,
    )

    history = pd.DataFrame(rows)
    hist_path = history_path(config.results_dir, spec)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(hist_path, index=False)
    _plot_history(history, config.results_dir, spec)
    return destination


def load_model(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
) -> tuple[SingleHiddenMultiTauSNN, dict[str, Any]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp6.0 checkpoint: {path}")
    device = torch.device(config.device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong experiment id in {path}")
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong protocol version in {path}")
    if checkpoint.get("spec") != asdict(spec):
        raise ValueError(f"Checkpoint spec mismatch for {spec.key}")
    model = _new_model(spec, data).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def visualization_sample_index(data: exp3.Data) -> int:
    lengths = np.asarray(data.lva, dtype=np.int64)
    if lengths.size == 0:
        raise ValueError("Validation split is empty")
    median = float(np.median(lengths))
    distances = np.abs(lengths.astype(np.float64) - median)
    return int(np.flatnonzero(distances == distances.min())[0])


def _fire_rate(spikes: np.ndarray) -> float:
    return float(spikes.mean()) if spikes.size else float("nan")


def _activity_stats(hidden: np.ndarray, output: np.ndarray, valid_length: int) -> dict[str, float]:
    valid_length = int(np.clip(valid_length, 0, hidden.shape[0]))
    s4, s5, s6 = group_slices()
    groups = {"s4": s4, "s5": s5, "s6": s6}
    stats: dict[str, float] = {}
    for name, sl in groups.items():
        stats[f"hidden_{name}_fr_valid"] = _fire_rate(hidden[:valid_length, sl])
        stats[f"hidden_{name}_fr_tail"] = _fire_rate(hidden[valid_length:, sl])
    stats["output_fr_valid"] = _fire_rate(output[:valid_length])
    stats["output_fr_tail"] = _fire_rate(output[valid_length:])
    return stats


def _plot_raster(
    hidden: np.ndarray,
    output: np.ndarray,
    valid_length: int,
    labels: list[str],
    path: Path,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ht, hn = np.where(hidden > 0)
    axes[0].scatter(ht, hn, s=5, marker=".")
    axes[0].axhline(GROUP_COUNTS[0] - 0.5, linewidth=0.8)
    axes[0].axhline(GROUP_COUNTS[0] + GROUP_COUNTS[1] - 0.5, linewidth=0.8)
    axes[0].axvline(valid_length, linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Hidden neuron")
    axes[0].set_ylim(-1, HIDDEN_WIDTH)
    axes[0].set_title(title)

    ot, on = np.where(output > 0)
    axes[1].scatter(ot, on, s=12, marker=".")
    axes[1].axvline(valid_length, linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Output neuron")
    axes[1].set_xlabel("Timestep")
    axes[1].set_xlim(0, VIS_STEPS - 1)
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels(labels)
    axes[1].set_ylim(-1, len(labels))

    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_activity(
    model: SingleHiddenMultiTauSNN,
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
    checkpoint: dict[str, Any],
    reversed_order: bool,
) -> dict[str, object]:
    index = visualization_sample_index(data)
    true_label_index = int(data.yva[index])
    labels = [str(value) for value in data.labels]
    valid_length = min(int(data.lva[index]), VIS_STEPS)

    x_np = np.zeros((1, VIS_STEPS, EXPECTED_CHANNELS), dtype=np.float32)
    available = min(VIS_STEPS, data.Xva.shape[1])
    x_np[:, :available] = data.Xva[index : index + 1, :available]
    x_np[:, valid_length:] = 0.0
    x = torch.tensor(x_np, dtype=torch.float32, device=torch.device(config.device))
    lengths = torch.tensor([valid_length], dtype=torch.long, device=x.device)
    if reversed_order:
        x = reverse_complete_bins(x, lengths, fixed250_steps(data.fs))

    with torch.no_grad():
        trajectory = model.forward_trajectory(x)
        logits = exp50.deployment_logits(
            trajectory["output_spikes"], lengths, OUTPUT_CAP
        )
    hidden = trajectory["hidden_spikes"][0].cpu().numpy()
    output = trajectory["output_spikes"][0].cpu().numpy()
    pred_index = int(logits.argmax(dim=1).item())

    npz_path = activity_path(config.results_dir, spec, reversed_order)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        input_spikes=x[0].detach().cpu().numpy(),
        hidden_spikes=hidden,
        output_spikes=output,
        valid_length=np.asarray(valid_length, dtype=np.int64),
        true_label=np.asarray(true_label_index, dtype=np.int64),
        pred_label=np.asarray(pred_index, dtype=np.int64),
        hidden_syn_shifts=np.asarray(
            [shift for shift, count in zip(SYN_SHIFTS, GROUP_COUNTS, strict=True) for _ in range(count)],
            dtype=np.int64,
        ),
        hidden_mem_shift=np.asarray(spec.mem_shift, dtype=np.int64),
        validation_index=np.asarray(index, dtype=np.int64),
        best_epoch=np.asarray(int(checkpoint["best_epoch"]), dtype=np.int64),
    )

    order_name = "reversed bins" if reversed_order else "original order"
    title = (
        f"{spec.key} | {order_name} | "
        f"true={labels[true_label_index]} pred={labels[pred_index]} "
        f"valid={valid_length} best_epoch={checkpoint['best_epoch']}"
    )
    _plot_raster(
        hidden,
        output,
        valid_length,
        labels,
        raster_path(config.results_dir, spec, reversed_order),
        title,
    )
    return {
        "validation_index": index,
        "valid_length": valid_length,
        "true_label_index": true_label_index,
        "true_label": labels[true_label_index],
        "pred_label_index": pred_index,
        "pred_label": labels[pred_index],
        "activity_npz": str(npz_path.relative_to(config.repo_root)),
        "raster_png": str(
            raster_path(config.results_dir, spec, reversed_order).relative_to(
                config.repo_root
            )
        ),
        "firing_rates": _activity_stats(hidden, output, valid_length),
    }


def evaluate_one(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model, checkpoint = load_model(spec, data, config)
    loaders = make_loaders(data, spec, config, train_shuffle=False)
    bin_steps = fixed250_steps(data.fs)
    split_metrics = {
        split: evaluate_loader(model, loader, device, spec.objective, bin_steps)
        for split, loader in loaders.items()
    }
    activity = _save_activity(model, spec, data, config, checkpoint, False)
    reversed_activity = (
        _save_activity(model, spec, data, config, checkpoint, True)
        if spec.objective == "whole_count_hce"
        else None
    )

    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "best_epoch": int(checkpoint["best_epoch"]),
        "architecture": architecture_manifest(spec, data),
        "metrics": split_metrics,
        "activity": activity,
        "reversed_activity": reversed_activity,
        "provenance": {
            "split_seed": int(exp3.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": [str(value) for value in data.labels],
            "checkpoint_selection": "highest validation balanced accuracy; tie-break lower validation whole-count CE",
            "test_policy": "test metrics evaluated only after loading the validation-selected checkpoint",
            "binary_spike_contract": "hidden cap=1; output cap=1",
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


def _raw_fixed250_features(
    X: np.ndarray,
    lengths: np.ndarray,
    data: exp3.Data,
) -> np.ndarray:
    tensor = torch.tensor(X, dtype=torch.float32)
    valid_lengths = torch.tensor(lengths, dtype=torch.long)
    features = exp3.fixed_counts(tensor, valid_lengths, fixed250_steps(data.fs))
    return features.flatten(start_dim=1).numpy()


def _fit_fixed250_linear(data: exp3.Data) -> dict[str, object]:
    partitions = _partitions(data)
    built = {
        split: (_raw_fixed250_features(X, lengths, data), y)
        for split, (X, y, lengths) in partitions.items()
    }
    scaler = StandardScaler().fit(built["train"][0])
    z = {
        split: scaler.transform(features)
        for split, (features, _) in built.items()
    }

    best: tuple[float, float, LogisticRegression] | None = None
    baseline_seed = exp3.dseed(exp3.SPLIT_SEED, "exp6_0_fixed250_linear")
    for C in exp302.PROBE_C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=5000,
            solver="lbfgs",
            random_state=baseline_seed,
        ).fit(z["train"], built["train"][1])
        val_ba = float(
            balanced_accuracy_score(
                built["val"][1],
                classifier.predict(z["val"]),
            )
        )
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError("No Fixed250 Linear candidate selected")

    _, selected_C, classifier = best
    metrics: dict[str, dict[str, float]] = {}
    for split in ("train", "val", "test"):
        y = built[split][1]
        pred = classifier.predict(z[split])
        metrics[split] = _classification_metrics(y, pred)

    return {
        "representation": "Raw64 -> ordered non-overlapping 250-ms channel counts -> flatten",
        "feature_dim": int(built["train"][0].shape[1]),
        "probe_C": selected_C,
        "metrics": metrics,
        "split_seed": int(exp3.SPLIT_SEED),
        "bin_steps": fixed250_steps(data.fs),
    }


def run_baseline(repo_root: Path, force: bool) -> dict[str, object]:
    root = results_dir(repo_root)
    destination = baseline_path(root)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    data = prepare_data(repo_root)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "baseline": _fit_fixed250_linear(data),
    }
    _save_json(destination, payload)
    return payload


def _row_from_evaluation(spec: RunSpec, payload: dict[str, object]) -> dict[str, object]:
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")
    activity = payload["activity"]
    if not isinstance(activity, dict):
        raise TypeError("activity must be a dict")
    firing = activity["firing_rates"]
    if not isinstance(firing, dict):
        raise TypeError("firing_rates must be a dict")
    train = metrics["train"]
    val = metrics["val"]
    test = metrics["test"]
    return {
        "objective": spec.objective,
        "mem_shift": spec.mem_shift,
        "tau_mem_ms": tau_ms_from_shift(spec.mem_shift, EXPECTED_FS),
        "seed": spec.seed,
        "best_epoch": int(payload["best_epoch"]),
        "train_loss": float(train["total_loss"]),
        "val_loss": float(val["total_loss"]),
        "train_ba": float(train["balanced_accuracy"]),
        "val_ba": float(val["balanced_accuracy"]),
        "test_ba": float(test["balanced_accuracy"]),
        "test_accuracy": float(test["accuracy"]),
        "test_macro_f1": float(test["macro_f1"]),
        "train_wholecount_ce": float(train["wholecount_ce"]),
        "val_wholecount_ce": float(val["wholecount_ce"]),
        "test_wholecount_ce": float(test["wholecount_ce"]),
        "test_hce_loss": float(test["hce_loss"]),
        "test_cg_loss": float(test["cg_loss"]),
        "test_orig_margin": float(test["orig_margin"]),
        "test_rev_margin": float(test["rev_margin"]),
        "test_mid_gain": float(test["mid_gain"]),
        "test_long_gain": float(test["long_gain"]),
        **{name: float(value) for name, value in firing.items()},
    }


def _summary_plots(
    runs: pd.DataFrame,
    baseline_test_ba: float,
    root: Path,
) -> None:
    plot_dir = root / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "whole_count": "WholeCount",
        "whole_count_hce": "WholeCount + HCE",
        "whole_count_contextual_gain": "WholeCount + Contextual Gain",
    }

    for metric, ylabel, filename in (
        ("test_ba", "Test balanced accuracy", "test_ba_vs_mem_shift.png"),
        ("val_ba", "Validation balanced accuracy", "val_ba_vs_mem_shift.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for objective in OBJECTIVES:
            subset = runs[runs["objective"] == objective]
            grouped = subset.groupby("mem_shift")[metric]
            means = grouped.mean().reindex(MEM_SHIFTS)
            stds = grouped.std(ddof=1).reindex(MEM_SHIFTS).fillna(0.0)
            ax.errorbar(
                MEM_SHIFTS,
                means.to_numpy(),
                yerr=stds.to_numpy(),
                marker="o",
                capsize=4,
                label=labels[objective],
            )
        if metric == "test_ba":
            ax.axhline(
                baseline_test_ba,
                linestyle="--",
                label="Raw Fixed250 + Linear",
            )
        ax.set_xticks(MEM_SHIFTS)
        ax.set_xlabel("Hidden membrane shift")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, shift_name in zip(axes, ("s4", "s5", "s6"), strict=True):
        column = f"hidden_{shift_name}_fr_valid"
        for objective in OBJECTIVES:
            subset = runs[runs["objective"] == objective]
            grouped = subset.groupby("mem_shift")[column]
            means = grouped.mean().reindex(MEM_SHIFTS)
            stds = grouped.std(ddof=1).reindex(MEM_SHIFTS).fillna(0.0)
            ax.errorbar(
                MEM_SHIFTS,
                means.to_numpy(),
                yerr=stds.to_numpy(),
                marker="o",
                capsize=4,
                label=labels[objective],
            )
        ax.set_title(f"{shift_name} hidden group")
        ax.set_xticks(MEM_SHIFTS)
        ax.set_xlabel("Hidden membrane shift")
    axes[0].set_ylabel("Valid-region spike fraction")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(
        plot_dir / "hidden_firing_rate_vs_mem_shift.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(fig)


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing required Exp6.0 evaluation: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_row_from_evaluation(spec, payload))

    baseline_file = baseline_path(root)
    if not baseline_file.exists():
        raise FileNotFoundError(f"Missing Fixed250 baseline artifact: {baseline_file}")
    baseline_payload = json.loads(baseline_file.read_text(encoding="utf-8"))
    baseline_test_ba = float(
        baseline_payload["baseline"]["metrics"]["test"]["balanced_accuracy"]
    )

    runs = pd.DataFrame(rows)
    runs_file = root / "runs.csv"
    root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(runs_file, index=False)

    summary = (
        runs.groupby(["objective", "mem_shift", "tau_mem_ms"])[
            [
                "val_ba",
                "test_ba",
                "test_accuracy",
                "test_macro_f1",
                "hidden_s4_fr_valid",
                "hidden_s5_fr_valid",
                "hidden_s6_fr_valid",
                "hidden_s4_fr_tail",
                "hidden_s5_fr_tail",
                "hidden_s6_fr_tail",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary_file = root / "summary.csv"
    summary.to_csv(summary_file, index=False)

    history_index = pd.DataFrame(
        [
            {
                "objective": spec.objective,
                "mem_shift": spec.mem_shift,
                "seed": spec.seed,
                "history_csv": str(
                    history_path(root, spec).relative_to(repo_root)
                ),
                "learning_curve_png": str(
                    learning_curve_path(root, spec).relative_to(repo_root)
                ),
                "raster_png": str(
                    raster_path(root, spec).relative_to(repo_root)
                ),
            }
            for spec in run_specs()
        ]
    )
    history_index_file = root / "history_index.csv"
    history_index.to_csv(history_index_file, index=False)

    first_payload_path = evaluation_path(root, run_specs()[0])
    first_payload = json.loads(first_payload_path.read_text(encoding="utf-8"))
    first_activity = first_payload["activity"]
    sample_manifest = {
        "validation_index": int(first_activity["validation_index"]),
        "valid_length": int(first_activity["valid_length"]),
        "label_index": int(first_activity["true_label_index"]),
        "label": str(first_activity["true_label"]),
        "selection_rule": "first validation sample whose valid length is closest to the validation median; chosen deterministically by each worker before inspecting model results",
        "visualization_steps": VIS_STEPS,
    }
    sample_file = root / "visualization_sample.json"
    _save_json(sample_file, sample_manifest)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "runs": len(run_specs()),
        "epochs": EPOCHS,
        "training_seeds": list(TRAIN_SEEDS),
        "user_split_seed": int(exp3.SPLIT_SEED),
        "objectives": list(OBJECTIVES),
        "hidden_mem_shifts": list(MEM_SHIFTS),
        "hidden_syn_shifts": list(SYN_SHIFTS),
        "hidden_group_counts": list(GROUP_COUNTS),
        "binary_spikes": True,
        "multi_cpu_policy": "27 independent one-core SNN tasks plus one one-core Fixed250 baseline; one artifact-only finalizer",
        "notebook_policy": "analysis-only; notebook reads finalized CSV/JSON/PNG artifacts and never trains",
        "baseline_test_ba": baseline_test_ba,
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    _summary_plots(runs, baseline_test_ba, root)

    return {
        "runs": runs_file,
        "summary": summary_file,
        "history_index": history_index_file,
        "visualization_sample": sample_file,
        "manifest": manifest_file,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 6.0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)

    sub.add_parser("baseline")
    sub.add_parser("finalize")
    sub.add_parser("list-runs")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = find_repo_root()
    root = results_dir(repo_root)
    config = Config(
        repo_root=repo_root,
        results_dir=root,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threads=args.threads,
    )

    if args.command == "list-runs":
        for index, spec in enumerate(run_specs()):
            print(index, spec.key)
        return
    if args.command == "baseline":
        payload = run_baseline(repo_root, args.force)
        print(json.dumps(payload, indent=2))
        return
    if args.command == "finalize":
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return
    if args.command == "run-one":
        specs = run_specs()
        task_id = int(args.array_task_id)
        if task_id < 0 or task_id >= len(specs):
            raise IndexError(f"array-task-id {task_id} outside [0, {len(specs) - 1}]")
        spec = specs[task_id]
        data = prepare_data(repo_root)
        payload = run_one(spec, data, config, args.force)
        print(
            json.dumps(
                {
                    "spec": asdict(spec),
                    "best_epoch": payload["best_epoch"],
                    "val_ba": payload["metrics"]["val"]["balanced_accuracy"],
                    "test_ba": payload["metrics"]["test"]["balanced_accuracy"],
                },
                indent=2,
            )
        )
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
