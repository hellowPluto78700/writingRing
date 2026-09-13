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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
import torch.nn.functional as F

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objective_comparison as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50
from scripts import experiment_6_0_multiscale_phase_evidence as exp60


EXPERIMENT_ID = "experiment_7_0_hierarchical_context_snn"
PROTOCOL_VERSION = "hierarchical_context_v1"

METHOD_WHAT_ONLY = "what_only"
METHOD_ALL_SKIP = "all_skip"
METHOD_ADDITIVE_CONTEXT = "additive_context"
METHOD_CONTEXT_GAIN = "context_gain"
METHODS = (
    METHOD_WHAT_ONLY,
    METHOD_ALL_SKIP,
    METHOD_ADDITIVE_CONTEXT,
    METHOD_CONTEXT_GAIN,
)

LEGACY = "legacy"
NORMALIZED = "normalized"
SYNAPSE_MODES = (LEGACY, NORMALIZED)
TRAIN_SEEDS = (11, 23, 37)

SHORT_SHIFT = 4
MIDDLE_SHIFT = 5
LONG_SHIFT = 6
HIERARCHICAL_SHIFTS = (SHORT_SHIFT, MIDDLE_SHIFT, LONG_SHIFT)
HIDDEN_WIDTH = 128
HIDDEN_CAP = 1
OUTPUT_CAP = 1
GAIN_LAMBDA = 1.0

LOCAL_SHIFTS: tuple[tuple[int, ...], ...] = ((2, 3, 4), (2, 3, 4))
LOCAL_WIDTH = exp01.HIDDEN_WIDTH
LOCAL_HIDDEN_CAP = 1

MAX_EPOCHS = 100
MIN_EPOCHS = 20
PATIENCE = 30
BATCH_SIZE = exp3.BATCH_SIZE
LR = exp3.LR
WEIGHT_DECAY = 0.0

TAU_MEM_MS = exp3.TAU_MEM_MS
OUTPUT_TAU_MEM_MS = exp60.OUTPUT_TAU_MEM_MS
THRESHOLD = exp3.THRESHOLD
SURROGATE_SLOPE = exp3.SURROGATE_SLOPE
EXPECTED_FS = 64.0
EXPECTED_CHANNELS = 30
EXPECTED_STEPS = 256
EXPECTED_SPLIT_SEED = 12345
FIXED250_MS = 250.0


@dataclass(frozen=True)
class RunSpec:
    method: str
    synapse_mode: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.method}__{self.synapse_mode}__seed{self.seed}"


@dataclass(frozen=True)
class LocalRunSpec:
    seed: int

    @property
    def key(self) -> str:
        return f"local_snn_fixed250_linear__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = MAX_EPOCHS
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
        RunSpec(method=method, synapse_mode=mode, seed=seed)
        for method in METHODS
        for mode in SYNAPSE_MODES
        for seed in TRAIN_SEEDS
    ]


def local_run_specs() -> list[LocalRunSpec]:
    return [LocalRunSpec(seed=seed) for seed in TRAIN_SEEDS]


def _validate_spec(spec: RunSpec) -> None:
    if spec.method not in METHODS:
        raise ValueError(f"Unknown Exp7.0 method: {spec.method}")
    if spec.synapse_mode not in SYNAPSE_MODES:
        raise ValueError(f"Unknown synapse mode: {spec.synapse_mode}")
    if spec.seed not in TRAIN_SEEDS:
        raise ValueError(f"Unknown training seed: {spec.seed}")


def paired_seed(seed: int, role: str) -> int:
    """Pair data order and common initialization streams across methods/modes."""
    return exp3.dseed(seed, "exp7_0_hierarchical_context", role)


def fixed250_steps(fs: float) -> int:
    return int(np.rint(FIXED250_MS * float(fs) / 1000.0))


def tau_ms_from_shift(shift: int, fs: float = EXPECTED_FS) -> float:
    return exp60.tau_ms_from_shift(shift, fs)


def synapse_step(
    state: torch.Tensor,
    drive: torch.Tensor,
    alpha: float,
    mode: str,
) -> torch.Tensor:
    if mode == LEGACY:
        return alpha * state + drive
    if mode == NORMALIZED:
        return alpha * state + (1.0 - alpha) * drive
    raise ValueError(f"Unknown synapse mode: {mode}")


def prepare_data(repo_root: Path) -> exp3.Data:
    data = exp60.prepare_data(repo_root)
    if not np.isclose(float(data.fs), EXPECTED_FS):
        raise ValueError(f"Exp7.0 requires {EXPECTED_FS:g} Hz, got {data.fs}")
    if int(exp3.SPLIT_SEED) != EXPECTED_SPLIT_SEED:
        raise ValueError("Exp7.0 split-seed contract changed")
    if data.Xtr.shape[1] != EXPECTED_STEPS:
        raise ValueError(f"Exp7.0 requires {EXPECTED_STEPS} padded timesteps")
    if fixed250_steps(data.fs) != 16:
        raise ValueError("At 64 Hz Exp7.0 requires 250 ms == 16 timesteps")
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


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


class HierarchicalContextSNN(nn.Module):
    """Raw30 -> short(s4) -> middle(s5) -> optional long(s6) -> short-tau output LIF."""

    def __init__(self, method: str, synapse_mode: str, n_classes: int, fs: float) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        if synapse_mode not in SYNAPSE_MODES:
            raise ValueError(f"Unknown synapse mode: {synapse_mode}")
        self.method = method
        self.synapse_mode = synapse_mode
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        self.has_long = method != METHOD_WHAT_ONLY

        self.short_linear = nn.Linear(EXPECTED_CHANNELS, HIDDEN_WIDTH, bias=False)
        self.middle_linear = nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False)
        self.long_linear = (
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False) if self.has_long else None
        )

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
        self.long_lif = (
            exp401.MacroMultiSpikeLIF(
                beta=hidden_beta,
                threshold=THRESHOLD,
                max_spikes_per_dt=HIDDEN_CAP,
                surrogate_slope=SURROGATE_SLOPE,
            )
            if self.has_long
            else None
        )
        self.output_lif = exp401.MacroMultiSpikeLIF(
            beta=output_beta,
            threshold=THRESHOLD,
            max_spikes_per_dt=OUTPUT_CAP,
            surrogate_slope=SURROGATE_SLOPE,
        )

        self.short_alpha = exp60.decay_from_shift(SHORT_SHIFT)
        self.middle_alpha = exp60.decay_from_shift(MIDDLE_SHIFT)
        self.long_alpha = exp60.decay_from_shift(LONG_SHIFT)

        self.what_head: nn.Linear | None = None
        self.context_head: nn.Linear | None = None
        self.gain_head: nn.Linear | None = None
        self.short_head: nn.Linear | None = None
        self.middle_head: nn.Linear | None = None
        self.long_head: nn.Linear | None = None

        if method == METHOD_ALL_SKIP:
            self.short_head = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=False)
            self.middle_head = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=False)
            self.long_head = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=False)
        else:
            self.what_head = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=False)
            if method == METHOD_ADDITIVE_CONTEXT:
                self.context_head = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=False)
            elif method == METHOD_CONTEXT_GAIN:
                self.gain_head = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=False)
                nn.init.zeros_(self.gain_head.weight)

    def _evidence(
        self,
        short_spike: torch.Tensor,
        middle_spike: torch.Tensor,
        long_spike: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.method == METHOD_WHAT_ONLY:
            if self.what_head is None:
                raise RuntimeError("Missing WHAT head")
            what = self.what_head(middle_spike)
            return what, what, None

        if long_spike is None:
            raise RuntimeError(f"{self.method} requires long-layer spikes")

        if self.method == METHOD_ALL_SKIP:
            if self.short_head is None or self.middle_head is None or self.long_head is None:
                raise RuntimeError("Missing all-skip heads")
            evidence = (
                self.short_head(short_spike)
                + self.middle_head(middle_spike)
                + self.long_head(long_spike)
            )
            return evidence, None, None

        if self.what_head is None:
            raise RuntimeError("Missing WHAT head")
        what = self.what_head(middle_spike)
        if self.method == METHOD_ADDITIVE_CONTEXT:
            if self.context_head is None:
                raise RuntimeError("Missing context head")
            return what + self.context_head(long_spike), what, None
        if self.method == METHOD_CONTEXT_GAIN:
            if self.gain_head is None:
                raise RuntimeError("Missing context-gain head")
            gain = 1.0 + GAIN_LAMBDA * torch.tanh(self.gain_head(long_spike))
            return gain * what, what, gain
        raise ValueError(f"Unknown method: {self.method}")

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
        long_syn = torch.zeros_like(short_syn) if self.has_long else None
        short_mem = torch.zeros_like(short_syn)
        middle_mem = torch.zeros_like(short_syn)
        long_mem = torch.zeros_like(short_syn) if self.has_long else None
        output_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)

        output_spikes: list[torch.Tensor] = []
        hidden_spikes: dict[str, list[torch.Tensor]] = {
            "short": [],
            "middle": [],
        }
        synaptic_currents: dict[str, list[torch.Tensor]] = {
            "short": [],
            "middle": [],
        }
        membrane_states: dict[str, list[torch.Tensor]] = {
            "short": [],
            "middle": [],
        }
        if self.has_long:
            hidden_spikes["long"] = []
            synaptic_currents["long"] = []
            membrane_states["long"] = []
        gains: list[torch.Tensor] = []
        what_evidence: list[torch.Tensor] = []
        class_evidence: list[torch.Tensor] = []

        for step in range(n_steps):
            short_drive = self.short_linear(x[:, step])
            short_syn = synapse_step(
                short_syn, short_drive, self.short_alpha, self.synapse_mode
            )
            short_spike, short_mem, _ = self.short_lif(short_syn, short_mem)

            middle_drive = self.middle_linear(short_spike)
            middle_syn = synapse_step(
                middle_syn, middle_drive, self.middle_alpha, self.synapse_mode
            )
            middle_spike, middle_mem, _ = self.middle_lif(middle_syn, middle_mem)

            long_spike: torch.Tensor | None = None
            if self.has_long:
                if self.long_linear is None or self.long_lif is None:
                    raise RuntimeError("Missing long layer")
                if long_syn is None or long_mem is None:
                    raise RuntimeError("Missing long state")
                long_drive = self.long_linear(middle_spike)
                long_syn = synapse_step(
                    long_syn, long_drive, self.long_alpha, self.synapse_mode
                )
                long_spike, long_mem, _ = self.long_lif(long_syn, long_mem)

            evidence, what, gain = self._evidence(short_spike, middle_spike, long_spike)
            output_spike, output_mem, _ = self.output_lif(evidence, output_mem)
            output_spikes.append(output_spike)

            if capture_states:
                hidden_spikes["short"].append(short_spike)
                hidden_spikes["middle"].append(middle_spike)
                synaptic_currents["short"].append(short_syn)
                synaptic_currents["middle"].append(middle_syn)
                membrane_states["short"].append(short_mem)
                membrane_states["middle"].append(middle_mem)
                if self.has_long:
                    if long_spike is None or long_syn is None or long_mem is None:
                        raise RuntimeError("Missing long trajectory state")
                    hidden_spikes["long"].append(long_spike)
                    synaptic_currents["long"].append(long_syn)
                    membrane_states["long"].append(long_mem)
                class_evidence.append(evidence)
                if what is not None:
                    what_evidence.append(what)
                if gain is not None:
                    gains.append(gain)

        payload: dict[str, Any] = {
            "output_spikes": torch.stack(output_spikes, dim=1),
        }
        if capture_states:
            payload["hidden_spikes"] = {
                name: torch.stack(values, dim=1) for name, values in hidden_spikes.items()
            }
            payload["synaptic_currents"] = {
                name: torch.stack(values, dim=1)
                for name, values in synaptic_currents.items()
            }
            payload["membrane_states"] = {
                name: torch.stack(values, dim=1)
                for name, values in membrane_states.items()
            }
            payload["class_evidence"] = torch.stack(class_evidence, dim=1)
            if what_evidence:
                payload["what_evidence"] = torch.stack(what_evidence, dim=1)
            if gains:
                payload["context_gain"] = torch.stack(gains, dim=1)
        return payload


def hierarchical_checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / "hierarchical" / f"{spec.key}.pt"


def hierarchical_history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / "hierarchical" / f"{spec.key}.csv"


def hierarchical_evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / "hierarchical" / f"{spec.key}.json"


def local_checkpoint_path(root: Path, spec: LocalRunSpec) -> Path:
    return root / "checkpoints" / "local_baseline" / f"{spec.key}.pt"


def local_history_path(root: Path, spec: LocalRunSpec) -> Path:
    return root / "histories" / "local_baseline" / f"{spec.key}.csv"


def local_evaluation_path(root: Path, spec: LocalRunSpec) -> Path:
    return root / "evaluations" / "local_baseline" / f"{spec.key}.json"


def raw_baseline_path(root: Path) -> Path:
    return root / "baseline_raw_fixed250_linear.json"


def training_curve_path(root: Path, family: str, key: str) -> Path:
    return root / "training_curves" / family / f"{key}.png"


def activity_path(root: Path, family: str, key: str) -> Path:
    return root / "activities" / family / f"{key}.npz"


def raster_path(root: Path, family: str, key: str, layer: str) -> Path:
    return root / "rasters" / family / f"{key}__{layer}.png"


def context_gain_path(root: Path, key: str) -> Path:
    return root / "context_gains" / f"{key}.png"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _whole_count_loss(
    output_spikes: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    logits = exp50.deployment_ce_logits(output_spikes, lengths, OUTPUT_CAP)
    return F.cross_entropy(logits, y)


def evaluate_hierarchical_loader(
    model: HierarchicalContextSNN,
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
            output_spikes = trajectory["output_spikes"]
            loss = _whole_count_loss(output_spikes, lengths, y)
            logits = exp50.deployment_logits(output_spikes, lengths, OUTPUT_CAP)
            n = len(y)
            n_total += n
            loss_sum += float(loss.item()) * n
            labels.append(y.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    metrics = _classification_metrics(np.concatenate(labels), np.concatenate(predictions))
    metrics["loss"] = loss_sum / max(n_total, 1)
    return metrics


def _plot_training_history(
    history: pd.DataFrame,
    destination: Path,
    title: str,
    best_epoch: int,
    stopped_epoch: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].axvline(best_epoch, linestyle="--", linewidth=1, label="best")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_ba"], label="train")
    axes[1].plot(history["epoch"], history["val_ba"], label="val")
    axes[1].axvline(best_epoch, linestyle="--", linewidth=1, label="best")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Balanced accuracy")
    axes[1].legend()
    fig.suptitle(f"{title} (stopped epoch {stopped_epoch})")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _is_improved(
    val_ba: float,
    val_loss: float,
    best_val_ba: float,
    best_val_loss: float,
) -> bool:
    return val_ba > best_val_ba + 1e-12 or (
        abs(val_ba - best_val_ba) <= 1e-12 and val_loss < best_val_loss
    )


def train_hierarchical(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
    force: bool,
) -> Path:
    _validate_spec(spec)
    destination = hierarchical_checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp3.seed_all(paired_seed(spec.seed, "model_init"))
    model = HierarchicalContextSNN(
        method=spec.method,
        synapse_mode=spec.synapse_mode,
        n_classes=len(data.labels),
        fs=data.fs,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loaders(data, spec.seed, config.batch_size, train_shuffle=True)["train"]
    eval_loaders = make_loaders(data, spec.seed, config.batch_size, train_shuffle=False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_loss = float("inf")
    rows: list[dict[str, float | int | bool]] = []
    stopped_epoch = config.epochs

    for epoch in range(1, config.epochs + 1):
        model.train()
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            loss = _whole_count_loss(trajectory["output_spikes"], lengths, y)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_hierarchical_loader(
            model, eval_loaders["train"], device
        )
        val_metrics = evaluate_hierarchical_loader(model, eval_loaders["val"], device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        improved = _is_improved(val_ba, val_loss, best_val_ba, best_val_loss)
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        patience_counter = max(0, epoch - best_epoch) if best_epoch > 0 else epoch
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "val_loss": val_loss,
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": val_ba,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_so_far": bool(improved),
                "patience_counter": int(patience_counter),
            }
        )
        if epoch >= MIN_EPOCHS and patience_counter >= PATIENCE:
            stopped_epoch = epoch
            break

    if best_state is None or best_epoch < 1:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_val_ba,
            "best_val_loss": best_val_loss,
            "model_state_dict": best_state,
            "labels": data.labels,
            "split": data.split,
            "architecture": architecture_manifest(spec, data),
        },
        destination,
    )
    history = pd.DataFrame(rows)
    history_file = hierarchical_history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_file, index=False)
    _plot_training_history(
        history,
        training_curve_path(config.results_dir, "hierarchical", spec.key),
        spec.key,
        best_epoch,
        stopped_epoch,
    )
    return destination


def architecture_manifest(spec: RunSpec, data: exp3.Data) -> dict[str, object]:
    layers = [
        {"name": "short", "width": HIDDEN_WIDTH, "shift_syn": SHORT_SHIFT},
        {"name": "middle", "width": HIDDEN_WIDTH, "shift_syn": MIDDLE_SHIFT},
    ]
    if spec.method != METHOD_WHAT_ONLY:
        layers.append({"name": "long", "width": HIDDEN_WIDTH, "shift_syn": LONG_SHIFT})
    for layer in layers:
        layer["tau_syn_ms"] = tau_ms_from_shift(int(layer["shift_syn"]), data.fs)
        layer["tau_mem_ms"] = float(TAU_MEM_MS)
        layer["binary_spike"] = True
        layer["cap"] = HIDDEN_CAP
    return {
        "input": {
            "channels": EXPECTED_CHANNELS,
            "sampling_rate_hz": float(data.fs),
            "timesteps": int(data.T),
            "representation": "Raw64 30-channel event spike train",
        },
        "hidden_layers": layers,
        "synapse_mode": spec.synapse_mode,
        "synaptic_update": (
            "I_t = alpha*I_{t-1} + W*x_t"
            if spec.synapse_mode == LEGACY
            else "I_t = alpha*I_{t-1} + (1-alpha)*W*x_t"
        ),
        "method": spec.method,
        "output": {
            "neurons": len(data.labels),
            "type": "short-tau binary LIF",
            "tau_mem_ms": float(OUTPUT_TAU_MEM_MS),
            "threshold": float(THRESHOLD),
            "cap": OUTPUT_CAP,
            "readout": "valid-length whole count",
        },
        "training": {
            "objective": "whole_count_ce",
            "max_epochs": MAX_EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "patience": PATIENCE,
        },
    }


def load_hierarchical(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
) -> tuple[HierarchicalContextSNN, dict[str, object]]:
    path = hierarchical_checkpoint_path(config.results_dir, spec)
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
    model = HierarchicalContextSNN(
        method=spec.method,
        synapse_mode=spec.synapse_mode,
        n_classes=len(data.labels),
        fs=data.fs,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _visualization_sample(data: exp3.Data) -> tuple[int, int]:
    lengths = np.asarray(data.lva, dtype=np.int64)
    median = float(np.median(lengths))
    index = int(np.argmin(np.abs(lengths.astype(np.float64) - median)))
    return index, int(lengths[index])


def _plot_raster(spikes: np.ndarray, destination: Path, valid_length: int, title: str) -> None:
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


def _activity_stats(spikes: np.ndarray, valid_length: int, fs: float) -> dict[str, float]:
    valid = spikes[:valid_length]
    tail = spikes[valid_length:]
    width = spikes.shape[1]
    return {
        "valid_firing_fraction": float(valid.mean()) if valid.size else 0.0,
        "zero_tail_firing_fraction": float(tail.mean()) if tail.size else 0.0,
        "valid_spikes_per_neuron_second": float(valid.sum() * fs / max(valid_length * width, 1)),
    }


def save_hierarchical_activity(
    spec: RunSpec,
    model: HierarchicalContextSNN,
    data: exp3.Data,
    config: Config,
) -> dict[str, dict[str, float]]:
    device = torch.device(config.device)
    index, valid_length = _visualization_sample(data)
    sample = torch.tensor(data.Xva[index:index + 1], dtype=torch.float32, device=device)
    sample[:, valid_length:] = 0.0
    with torch.no_grad():
        trajectory = model.forward_trajectory(sample, capture_states=True)

    payload: dict[str, np.ndarray] = {
        "input_spikes": sample[0].cpu().numpy(),
        "output_spikes": trajectory["output_spikes"][0].cpu().numpy(),
        "valid_length": np.asarray(valid_length, dtype=np.int64),
        "label": np.asarray(int(data.yva[index]), dtype=np.int64),
    }
    hidden = trajectory["hidden_spikes"]
    currents = trajectory["synaptic_currents"]
    membranes = trajectory["membrane_states"]
    summaries: dict[str, dict[str, float]] = {}
    for layer in hidden:
        spikes = hidden[layer][0].cpu().numpy()
        payload[f"{layer}_spikes"] = spikes
        payload[f"{layer}_current"] = currents[layer][0].cpu().numpy()
        payload[f"{layer}_membrane"] = membranes[layer][0].cpu().numpy()
        summaries[layer] = _activity_stats(spikes, valid_length, data.fs)
        _plot_raster(
            spikes,
            raster_path(config.results_dir, "hierarchical", spec.key, layer),
            valid_length,
            f"{spec.key} — {layer}",
        )

    output = payload["output_spikes"]
    summaries["output"] = _activity_stats(output, valid_length, data.fs)
    _plot_raster(
        output,
        raster_path(config.results_dir, "hierarchical", spec.key, "output"),
        valid_length,
        f"{spec.key} — output",
    )
    payload["class_evidence"] = trajectory["class_evidence"][0].cpu().numpy()
    if "what_evidence" in trajectory:
        payload["what_evidence"] = trajectory["what_evidence"][0].cpu().numpy()
    if "context_gain" in trajectory:
        gains = trajectory["context_gain"][0].cpu().numpy()
        payload["context_gain"] = gains
        gain_destination = context_gain_path(config.results_dir, spec.key)
        gain_destination.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 5))
        for class_index in range(gains.shape[1]):
            ax.plot(np.arange(gains.shape[0]), gains[:, class_index], linewidth=1)
        ax.axvline(valid_length, linestyle="--", linewidth=1)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Class-specific gain")
        ax.set_title(spec.key)
        fig.tight_layout()
        fig.savefig(gain_destination, dpi=160, bbox_inches="tight")
        plt.close(fig)

    destination = activity_path(config.results_dir, "hierarchical", spec.key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return summaries


def evaluate_hierarchical(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    destination = hierarchical_evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    model, checkpoint = load_hierarchical(spec, data, config)
    loaders = make_loaders(data, spec.seed, config.batch_size, train_shuffle=False)
    metrics = {
        split: evaluate_hierarchical_loader(model, loader, torch.device(config.device))
        for split, loader in loaders.items()
    }
    activity_summary = save_hierarchical_activity(spec, model, data, config)
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "family": "hierarchical_snn",
        "spec": asdict(spec),
        "best_epoch": int(checkpoint["best_epoch"]),
        "stopped_epoch": int(checkpoint["stopped_epoch"]),
        "best_val_ba": float(checkpoint["best_val_ba"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "metrics": metrics,
        "activity_summary": activity_summary,
        "architecture": architecture_manifest(spec, data),
        "provenance": {
            "split_seed": int(exp3.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "checkpoint_selection": "validation balanced accuracy; lower validation whole-count CE tie-break",
        },
    }
    _save_json(destination, payload)
    return payload


def run_hierarchical_one(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    train_hierarchical(spec, data, config, force)
    return evaluate_hierarchical(spec, data, config, force)


def _new_local_model(data: exp3.Data) -> exp01.MultiTauHierarchySNN:
    return exp01.MultiTauHierarchySNN(
        layer_shifts=LOCAL_SHIFTS,
        n_classes=len(data.labels),
        fs=data.fs,
        hidden_cap=LOCAL_HIDDEN_CAP,
        output_spiking=False,
    )


def evaluate_local_native_loader(
    model: exp01.MultiTauHierarchySNN,
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
            logits_t = trajectory["analog_logits"]
            if not isinstance(logits_t, torch.Tensor):
                raise TypeError("Expected analog logits")
            segment_logits = exp50.valid_mean(logits_t, lengths)
            loss = F.cross_entropy(segment_logits, y)
            n = len(y)
            n_total += n
            loss_sum += float(loss.item()) * n
            labels.append(y.cpu().numpy())
            predictions.append(segment_logits.argmax(dim=1).cpu().numpy())
    metrics = _classification_metrics(np.concatenate(labels), np.concatenate(predictions))
    metrics["loss"] = loss_sum / max(n_total, 1)
    return metrics


def train_local_baseline(
    spec: LocalRunSpec,
    data: exp3.Data,
    config: Config,
    force: bool,
) -> Path:
    destination = local_checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp3.seed_all(paired_seed(spec.seed, "model_init"))
    model = _new_local_model(data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loaders(data, spec.seed, config.batch_size, train_shuffle=True)["train"]
    eval_loaders = make_loaders(data, spec.seed, config.batch_size, train_shuffle=False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_loss = float("inf")
    rows: list[dict[str, float | int | bool]] = []
    stopped_epoch = config.epochs

    for epoch in range(1, config.epochs + 1):
        model.train()
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            logits_t = trajectory["analog_logits"]
            if not isinstance(logits_t, torch.Tensor):
                raise TypeError("Expected analog logits")
            valid = exp50.valid_mask(lengths, logits_t.shape[1])
            targets = y[:, None].expand(len(y), logits_t.shape[1])
            loss = F.cross_entropy(logits_t[valid], targets[valid])
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_local_native_loader(
            model, eval_loaders["train"], device
        )
        val_metrics = evaluate_local_native_loader(model, eval_loaders["val"], device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        improved = _is_improved(val_ba, val_loss, best_val_ba, best_val_loss)
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        patience_counter = max(0, epoch - best_epoch) if best_epoch > 0 else epoch
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "val_loss": val_loss,
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": val_ba,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_so_far": bool(improved),
                "patience_counter": int(patience_counter),
            }
        )
        if epoch >= MIN_EPOCHS and patience_counter >= PATIENCE:
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "family": "local_snn_fixed250_linear",
            "spec": asdict(spec),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_val_ba,
            "best_val_loss": best_val_loss,
            "model_state_dict": best_state,
            "labels": data.labels,
            "split": data.split,
        },
        destination,
    )
    history = pd.DataFrame(rows)
    history_file = local_history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_file, index=False)
    _plot_training_history(
        history,
        training_curve_path(config.results_dir, "local_baseline", spec.key),
        spec.key,
        best_epoch,
        stopped_epoch,
    )
    return destination


def load_local_baseline(
    spec: LocalRunSpec,
    data: exp3.Data,
    config: Config,
) -> tuple[exp01.MultiTauHierarchySNN, dict[str, object]]:
    path = local_checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(path)
    device = torch.device(config.device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong experiment id in {path}")
    if checkpoint.get("spec") != asdict(spec):
        raise ValueError(f"Checkpoint identity mismatch for {spec.key}")
    model = _new_local_model(data).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _extract_local_fixed250(
    model: exp01.MultiTauHierarchySNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    bin_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            hidden = trajectory["hidden_spikes"]
            if not isinstance(hidden, tuple):
                raise TypeError("Expected hidden spike tuple")
            features = exp3.fixed_counts(hidden[-1], lengths, bin_steps).flatten(start_dim=1)
            matrices.append(features.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(matrices), np.concatenate(labels)


def save_local_activity(
    spec: LocalRunSpec,
    model: exp01.MultiTauHierarchySNN,
    data: exp3.Data,
    config: Config,
) -> dict[str, dict[str, float]]:
    device = torch.device(config.device)
    index, valid_length = _visualization_sample(data)
    sample = torch.tensor(data.Xva[index:index + 1], dtype=torch.float32, device=device)
    sample[:, valid_length:] = 0.0
    with torch.no_grad():
        trajectory = model.forward_trajectory(sample)
    hidden = trajectory["hidden_spikes"]
    if not isinstance(hidden, tuple):
        raise TypeError("Expected hidden spike tuple")
    payload: dict[str, np.ndarray] = {
        "input_spikes": sample[0].cpu().numpy(),
        "valid_length": np.asarray(valid_length, dtype=np.int64),
        "label": np.asarray(int(data.yva[index]), dtype=np.int64),
    }
    summaries: dict[str, dict[str, float]] = {}
    for layer_index, spikes_tensor in enumerate(hidden, start=1):
        layer = f"L{layer_index}"
        spikes = spikes_tensor[0].cpu().numpy()
        payload[f"{layer}_spikes"] = spikes
        summaries[layer] = _activity_stats(spikes, valid_length, data.fs)
        _plot_raster(
            spikes,
            raster_path(config.results_dir, "local_baseline", spec.key, layer),
            valid_length,
            f"{spec.key} — {layer}",
        )
    destination = activity_path(config.results_dir, "local_baseline", spec.key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return summaries


def evaluate_local_baseline(
    spec: LocalRunSpec,
    data: exp3.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    destination = local_evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    model, checkpoint = load_local_baseline(spec, data, config)
    device = torch.device(config.device)
    loaders = make_loaders(data, spec.seed, config.batch_size, train_shuffle=False)
    native = {
        split: evaluate_local_native_loader(model, loader, device)
        for split, loader in loaders.items()
    }
    extracted = {
        split: _extract_local_fixed250(model, loader, device, data.bin_steps)
        for split, loader in loaders.items()
    }
    probe = exp01._fit_linear_probe(
        extracted["train"][0],
        extracted["train"][1],
        extracted["val"][0],
        extracted["val"][1],
        extracted["test"][0],
        extracted["test"][1],
        exp3.dseed(spec.seed, "exp7_0_local_fixed250_probe"),
    )
    activity_summary = save_local_activity(spec, model, data, config)
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "family": "local_snn_fixed250_linear",
        "spec": asdict(spec),
        "best_epoch": int(checkpoint["best_epoch"]),
        "stopped_epoch": int(checkpoint["stopped_epoch"]),
        "best_val_native_ba": float(checkpoint["best_val_ba"]),
        "best_val_native_loss": float(checkpoint["best_val_loss"]),
        "native_metrics": native,
        "fixed250_probe": probe,
        "metrics": {split: probe[split] for split in ("train", "val", "test")},
        "activity_summary": activity_summary,
        "architecture": {
            "hidden_layers": [
                {
                    "layer": 1,
                    "width": LOCAL_WIDTH,
                    "shifts": list(LOCAL_SHIFTS[0]),
                    "tau_syn_ms": [tau_ms_from_shift(shift, data.fs) for shift in LOCAL_SHIFTS[0]],
                },
                {
                    "layer": 2,
                    "width": LOCAL_WIDTH,
                    "shifts": list(LOCAL_SHIFTS[1]),
                    "tau_syn_ms": [tau_ms_from_shift(shift, data.fs) for shift in LOCAL_SHIFTS[1]],
                },
            ],
            "tau_mem_ms": float(TAU_MEM_MS),
            "hidden_cap": LOCAL_HIDDEN_CAP,
            "training_head": "temporary analog timestep classifier",
            "deployment_readout": "L2 ordered Fixed250 counts + validation-selected LogisticRegression",
        },
        "provenance": {
            "source_architecture": "Exp0.1 local_234x2 binary baseline",
            "split_seed": int(exp3.SPLIT_SEED),
            "checkpoint_selection": "validation native analog-head BA; lower native CE tie-break",
        },
    }
    _save_json(destination, payload)
    return payload


def run_local_one(
    spec: LocalRunSpec,
    data: exp3.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    train_local_baseline(spec, data, config, force)
    return evaluate_local_baseline(spec, data, config, force)


def _raw_fixed250_features(
    X: np.ndarray,
    lengths: np.ndarray,
    bin_steps: int,
) -> np.ndarray:
    tensor = torch.tensor(X, dtype=torch.float32)
    valid_lengths = torch.tensor(lengths, dtype=torch.long)
    return exp3.fixed_counts(tensor, valid_lengths, bin_steps).flatten(start_dim=1).numpy()


def run_raw_baseline(repo_root: Path, force: bool = False) -> dict[str, object]:
    root = results_dir(repo_root)
    destination = raw_baseline_path(root)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    data = prepare_data(repo_root)
    built = {
        split: (_raw_fixed250_features(X, lengths, data.bin_steps), y)
        for split, (X, y, lengths) in _partitions(data).items()
    }
    probe = exp01._fit_linear_probe(
        built["train"][0],
        built["train"][1],
        built["val"][0],
        built["val"][1],
        built["test"][0],
        built["test"][1],
        exp3.dseed(EXPECTED_SPLIT_SEED, "exp7_0_raw_fixed250_linear"),
    )
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "family": "raw_fixed250_linear",
        "metrics": {split: probe[split] for split in ("train", "val", "test")},
        "feature_dim": int(probe["feature_dim"]),
        "probe_C": float(probe["probe_C"]),
        "split_seed": int(exp3.SPLIT_SEED),
        "protocol": "Raw64 -> ordered 250-ms channel counts -> flatten -> train-only StandardScaler -> validation-selected LogisticRegression",
    }
    _save_json(destination, payload)
    return payload


def _evaluation_to_row(payload: dict[str, object]) -> dict[str, object]:
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")
    spec = payload.get("spec", {})
    if not isinstance(spec, dict):
        spec = {}
    family = str(payload["family"])
    if family == "hierarchical_snn":
        method = str(spec["method"])
        mode = str(spec["synapse_mode"])
        seed: float | int = int(spec["seed"])
    elif family == "local_snn_fixed250_linear":
        method = "local_snn_fixed250_linear"
        mode = "legacy_local"
        seed = int(spec["seed"])
    else:
        raise ValueError(f"Unexpected run family: {family}")
    train = metrics["train"]
    val = metrics["val"]
    test = metrics["test"]
    return {
        "family": family,
        "method": method,
        "synapse_mode": mode,
        "seed": seed,
        "best_epoch": payload.get("best_epoch", np.nan),
        "stopped_epoch": payload.get("stopped_epoch", np.nan),
        "train_ba": float(train["balanced_accuracy"]),
        "val_ba": float(val["balanced_accuracy"]),
        "test_ba": float(test["balanced_accuracy"]),
        "test_accuracy": float(test["accuracy"]),
        "test_macro_f1": float(test["macro_f1"]),
    }


def _summary_plot(summary: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = summary.copy()
    labels = [f"{m}\n{s}" for m, s in zip(frame["method"], frame["synapse_mode"], strict=True)]
    x = np.arange(len(frame))
    y = frame["test_ba_mean"].to_numpy(dtype=float)
    err = frame["test_ba_std"].fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x, y, yerr=err, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Test balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Exp7.0 method-level comparison (mean +/- SD)")
    fig.tight_layout()
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _paired_plot(paired: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    grouped = (
        paired.groupby("method")["normalized_minus_legacy_test_ba"]
        .agg(["mean", "std"])
        .reindex(METHODS)
    )
    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x, grouped["mean"], yerr=grouped["std"].fillna(0.0), capsize=3)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=25, ha="right")
    ax.set_ylabel("Normalized - legacy test BA")
    ax.set_title("Paired neuron-dynamics effect")
    fig.tight_layout()
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)


def finalize(repo_root: Path) -> dict[str, object]:
    root = results_dir(repo_root)
    raw_path = raw_baseline_path(root)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = [
        {
            "family": "raw_fixed250_linear",
            "method": "raw_fixed250_linear",
            "synapse_mode": "reference",
            "seed": np.nan,
            "best_epoch": np.nan,
            "stopped_epoch": np.nan,
            "train_ba": float(raw["metrics"]["train"]["balanced_accuracy"]),
            "val_ba": float(raw["metrics"]["val"]["balanced_accuracy"]),
            "test_ba": float(raw["metrics"]["test"]["balanced_accuracy"]),
            "test_accuracy": float(raw["metrics"]["test"]["accuracy"]),
            "test_macro_f1": float(raw["metrics"]["test"]["macro_f1"]),
        }
    ]
    firing_rows: list[dict[str, object]] = []

    for spec in local_run_specs():
        path = local_evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_evaluation_to_row(payload))
        for layer, values in payload["activity_summary"].items():
            firing_rows.append(
                {
                    "method": "local_snn_fixed250_linear",
                    "synapse_mode": "legacy_local",
                    "seed": spec.seed,
                    "layer": layer,
                    **values,
                }
            )

    for spec in run_specs():
        path = hierarchical_evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_evaluation_to_row(payload))
        for layer, values in payload["activity_summary"].items():
            firing_rows.append(
                {
                    "method": spec.method,
                    "synapse_mode": spec.synapse_mode,
                    "seed": spec.seed,
                    "layer": layer,
                    **values,
                }
            )

    runs = pd.DataFrame(rows)
    runs.to_csv(root / "runs.csv", index=False)
    summary = (
        runs.groupby(["method", "synapse_mode"], dropna=False)
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

    hierarchical = runs[runs["family"] == "hierarchical_snn"].copy()
    pivot = hierarchical.pivot_table(
        index=["method", "seed"], columns="synapse_mode", values="test_ba"
    ).reset_index()
    if LEGACY not in pivot or NORMALIZED not in pivot:
        raise RuntimeError("Incomplete legacy/normalized pairing")
    paired = pivot[["method", "seed", LEGACY, NORMALIZED]].copy()
    paired["normalized_minus_legacy_test_ba"] = paired[NORMALIZED] - paired[LEGACY]
    paired.to_csv(root / "paired_neuron_deltas.csv", index=False)

    architecture = hierarchical.pivot_table(
        index=["synapse_mode", "seed"], columns="method", values="test_ba"
    ).reset_index()
    required = set(METHODS)
    if not required.issubset(architecture.columns):
        raise RuntimeError("Incomplete method matrix for architecture deltas")
    delta_rows: list[dict[str, object]] = []
    contrasts = {
        "all_skip_minus_what": (METHOD_ALL_SKIP, METHOD_WHAT_ONLY),
        "additive_context_minus_what": (METHOD_ADDITIVE_CONTEXT, METHOD_WHAT_ONLY),
        "context_gain_minus_what": (METHOD_CONTEXT_GAIN, METHOD_WHAT_ONLY),
        "context_gain_minus_additive": (METHOD_CONTEXT_GAIN, METHOD_ADDITIVE_CONTEXT),
    }
    for _, row in architecture.iterrows():
        for contrast, (lhs, rhs) in contrasts.items():
            delta_rows.append(
                {
                    "synapse_mode": row["synapse_mode"],
                    "seed": int(row["seed"]),
                    "contrast": contrast,
                    "test_ba_delta": float(row[lhs] - row[rhs]),
                }
            )
    method_deltas = pd.DataFrame(delta_rows)
    method_deltas.to_csv(root / "method_deltas.csv", index=False)

    firing = pd.DataFrame(firing_rows)
    firing.to_csv(root / "firing_runs.csv", index=False)
    firing_summary = (
        firing.groupby(["method", "synapse_mode", "layer"])
        .agg(
            n_runs=("valid_firing_fraction", "count"),
            valid_firing_fraction_mean=("valid_firing_fraction", "mean"),
            valid_firing_fraction_std=("valid_firing_fraction", "std"),
            zero_tail_firing_fraction_mean=("zero_tail_firing_fraction", "mean"),
            zero_tail_firing_fraction_std=("zero_tail_firing_fraction", "std"),
            valid_spikes_per_neuron_second_mean=("valid_spikes_per_neuron_second", "mean"),
        )
        .reset_index()
    )
    firing_summary.to_csv(root / "firing_summary.csv", index=False)

    plots = root / "summary_plots"
    _summary_plot(summary, plots / "test_ba_by_method.png")
    _paired_plot(paired, plots / "paired_neuron_delta.png")

    visualization_index, visualization_valid = _visualization_sample(prepare_data(repo_root))
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "raw_baseline_jobs": 1,
        "local_snn_runs": len(local_run_specs()),
        "hierarchical_snn_runs": len(run_specs()),
        "total_gradient_training_runs": len(local_run_specs()) + len(run_specs()),
        "max_epochs": MAX_EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "patience": PATIENCE,
        "visualization_sample": {
            "val_index": visualization_index,
            "valid_length": visualization_valid,
        },
        "notebook_policy": "analysis-only; consume method-level aggregate artifacts, not per-run histories/evaluations",
        "aggregate_artifacts": [
            "summary.csv",
            "paired_neuron_deltas.csv",
            "method_deltas.csv",
            "firing_summary.csv",
        ],
    }
    _save_json(root / "manifest.json", manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 7.0 hierarchical context SNN")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-runs")
    sub.add_parser("list-local-runs")

    run_one_parser = sub.add_parser("run-one")
    run_one_parser.add_argument("--array-task-id", type=int, required=True)
    run_one_parser.add_argument("--force", action="store_true")

    local_parser = sub.add_parser("run-local")
    local_parser.add_argument("--array-task-id", type=int, required=True)
    local_parser.add_argument("--force", action="store_true")

    baseline_parser = sub.add_parser("baseline")
    baseline_parser.add_argument("--force", action="store_true")

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
        epochs=min(int(args.epochs), MAX_EPOCHS),
        batch_size=int(args.batch_size),
        threads=int(args.threads),
    )

    if args.command == "list-runs":
        for index, spec in enumerate(run_specs()):
            print(index, spec.key)
        return
    if args.command == "list-local-runs":
        for index, spec in enumerate(local_run_specs()):
            print(index, spec.key)
        return
    if args.command == "baseline":
        print(json.dumps(run_raw_baseline(repo_root, force=args.force), indent=2))
        return
    if args.command == "finalize":
        print(json.dumps(finalize(repo_root), indent=2))
        return

    data = prepare_data(repo_root)
    if args.command == "run-one":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        payload = run_hierarchical_one(
            specs[args.array_task_id], data, config, force=args.force
        )
        print(json.dumps(payload, indent=2))
        return
    if args.command == "run-local":
        specs = local_run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        payload = run_local_one(specs[args.array_task_id], data, config, force=args.force)
        print(json.dumps(payload, indent=2))
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
