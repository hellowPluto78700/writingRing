from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_0_1_2_regularized_general_comparison as exp012
from scripts import experiment_0_1_4_parallel_regularization_ablation as exp014
from scripts import experiment_0_1_5_gradient_calibrated_all_fixes as exp015


EXPERIMENT_ID = "experiment_0_2_endpoint_tail_regularization"
PROTOCOL_VERSION = "endpoint_tail_regularization_v1"
BASE_EXPERIMENT_ID = exp01.EXPERIMENT_ID
BASE_PROTOCOL_VERSION = exp01.PROTOCOL_VERSION

DIRECT_FAMILY = exp01.DIRECT_FAMILY
DIRECT_ARCHITECTURES = exp01.DIRECT_ARCHITECTURES
OBJECTIVES = ("whole_count_ce", "timestep_ce")
PROFILES = ("a1", "a1_p2", "tail", "a1_tail")
FROZEN_PROFILE = "none"
SEEDS = (11, 23, 37)
VARIANT = "binary"
HIDDEN_CAP = 1
OUTPUT_CAP = 1
HIDDEN_WIDTH = exp01.HIDDEN_WIDTH

EPOCHS = 50
BATCH_SIZE = exp01.BATCH_SIZE
LR = exp01.LR
WEIGHT_DECAY = exp01.WEIGHT_DECAY
TARGET_GRAD_RATIO = 0.05
CALIBRATION_BATCHES = 5
WARMUP_EPOCHS = 10
BASE_LAMBDA_A1 = 0.1
BASE_LAMBDA_P2 = 0.01
BASE_LAMBDA_TAIL = 0.1
REG_TAU = exp014.REG_TAU

TAIL_HORIZON_MS = 600.0
TAIL_STAGE_MS = (200.0, 400.0, 600.0)
TAIL_STAGE_WEIGHTS = (1.0, 2.0, 4.0)
TAIL_WEIGHT_SUM = float(sum(TAIL_STAGE_WEIGHTS))
RASTER_SAMPLE_INDEX = 4


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    objective: str
    profile: str
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{DIRECT_FAMILY}__{self.architecture}__{self.objective}__"
            f"{VARIANT}__hcap{HIDDEN_CAP}__{self.profile}__seed{self.seed}"
        )


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def base_results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / BASE_EXPERIMENT_ID / BASE_PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(architecture, objective, profile, seed)
        for architecture in DIRECT_ARCHITECTURES
        for objective in OBJECTIVES
        for profile in PROFILES
        for seed in SEEDS
    ]


def frozen_specs() -> list[RunSpec]:
    return [
        RunSpec(architecture, objective, FROZEN_PROFILE, seed)
        for architecture in DIRECT_ARCHITECTURES
        for objective in OBJECTIVES
        for seed in SEEDS
    ]


def base_spec(spec: RunSpec) -> exp01.RunSpec:
    return exp01.RunSpec(
        DIRECT_FAMILY,
        spec.architecture,
        spec.objective,
        VARIANT,
        HIDDEN_CAP,
        spec.seed,
    )


def paired_seed(spec: RunSpec, role: str) -> int:
    return exp01.paired_seed(base_spec(spec), role)


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def calibration_path(root: Path, spec: RunSpec) -> Path:
    return root / "calibrations" / f"{spec.key}.json"


def raster_path(root: Path, spec: RunSpec) -> Path:
    return root / "firing_patterns" / spec.architecture / spec.objective / spec.profile / f"seed{spec.seed}.png"


def trace_path(root: Path, spec: RunSpec) -> Path:
    return root / "firing_patterns" / spec.architecture / spec.objective / spec.profile / f"seed{spec.seed}.npz"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_data(repo_root: Path) -> exp01.exp3.Data:
    data = exp01.prepare_data(repo_root)
    if exp01.exp3.SPLIT_SEED != 12345:
        raise ValueError(f"Exp0.2 requires split seed 12345, got {exp01.exp3.SPLIT_SEED}")
    return data


def make_loaders(
    data: exp01.exp3.Data,
    spec: RunSpec,
    batch_size: int,
    train_shuffle: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    return exp01.make_loaders(data, base_spec(spec), batch_size, train_shuffle)


class DiagnosticMultiTauHierarchySNN(exp012.RegularizedMultiTauHierarchySNN):
    """Exp0.1 binary direct SNN with spikes, synaptic state, pre-reset and membrane traces."""

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, object]:
        batch, n_steps, channels = x.shape
        if channels != exp01.exp3.EVENT_CHANNELS:
            raise ValueError(f"Expected {exp01.exp3.EVENT_CHANNELS} input channels, got {channels}")

        syn_states = [
            torch.zeros(batch, HIDDEN_WIDTH, device=x.device, dtype=x.dtype)
            for _ in self.hidden_linears
        ]
        mem_states = [torch.zeros_like(value) for value in syn_states]
        layer_spikes: list[list[torch.Tensor]] = [[] for _ in self.hidden_linears]
        layer_synaptic: list[list[torch.Tensor]] = [[] for _ in self.hidden_linears]
        layer_pre_reset: list[list[torch.Tensor]] = [[] for _ in self.hidden_linears]
        layer_membrane: list[list[torch.Tensor]] = [[] for _ in self.hidden_linears]
        output_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)
        output_spikes: list[torch.Tensor] = []
        output_membrane: list[torch.Tensor] = []

        for step in range(n_steps):
            current = x[:, step]
            for index, (linear, lif) in enumerate(zip(self.hidden_linears, self.hidden_lifs, strict=True)):
                alpha = getattr(self, self._alpha_names[index])
                syn_states[index] = alpha * syn_states[index] + linear(current)
                spike, mem_states[index], pre_reset = lif(syn_states[index], mem_states[index])
                layer_synaptic[index].append(syn_states[index])
                layer_pre_reset[index].append(pre_reset)
                layer_spikes[index].append(spike)
                layer_membrane[index].append(mem_states[index])
                current = spike

            if self.output_linear is None or self.output_lif is None:
                raise RuntimeError("Exp0.2 requires a spiking output head")
            spike_out, output_mem, _ = self.output_lif(self.output_linear(current), output_mem)
            output_spikes.append(spike_out)
            output_membrane.append(output_mem)

        return {
            "hidden_spikes": tuple(torch.stack(parts, dim=1) for parts in layer_spikes),
            "hidden_synaptic": tuple(torch.stack(parts, dim=1) for parts in layer_synaptic),
            "hidden_pre_reset": tuple(torch.stack(parts, dim=1) for parts in layer_pre_reset),
            "hidden_membrane": tuple(torch.stack(parts, dim=1) for parts in layer_membrane),
            "output_spikes": torch.stack(output_spikes, dim=1),
            "output_membrane": torch.stack(output_membrane, dim=1),
        }


def _new_model(spec: RunSpec, data: exp01.exp3.Data) -> DiagnosticMultiTauHierarchySNN:
    return DiagnosticMultiTauHierarchySNN(
        layer_shifts=DIRECT_ARCHITECTURES[spec.architecture],
        n_classes=len(data.labels),
        fs=data.fs,
        hidden_cap=HIDDEN_CAP,
    )


def _hidden_alphas(model: DiagnosticMultiTauHierarchySNN) -> tuple[torch.Tensor, ...]:
    return tuple(getattr(model, name) for name in model._alpha_names)


def _hidden_parameters(model: DiagnosticMultiTauHierarchySNN) -> tuple[torch.nn.Parameter, ...]:
    params = tuple(
        parameter
        for linear in model.hidden_linears
        for parameter in linear.parameters()
        if parameter.requires_grad
    )
    if not params:
        raise RuntimeError("No trainable hidden parameters")
    return params


def _steps(ms: float, fs: float) -> int:
    return int(np.rint(float(ms) * float(fs) / 1000.0))


def tail_horizon_steps(fs: float) -> int:
    value = _steps(TAIL_HORIZON_MS, fs)
    if value < 1:
        raise ValueError("Tail horizon must contain at least one timestep")
    return value


def tail_stage_boundaries(fs: float) -> tuple[int, int, int]:
    values = tuple(_steps(ms, fs) for ms in TAIL_STAGE_MS)
    if not (0 < values[0] < values[1] < values[2]):
        raise ValueError(f"Invalid tail stage boundaries at fs={fs}: {values}")
    return values


def endpoint_rollout_input(X: torch.Tensor, lengths: torch.Tensor, fs: float) -> torch.Tensor:
    """Return valid input prefixes followed by a fixed zero-input endpoint-relative rollout."""
    if X.ndim != 3 or lengths.ndim != 1 or X.shape[0] != lengths.shape[0]:
        raise ValueError("Expected X [B,T,C] and lengths [B]")
    if torch.any(lengths <= 0) or torch.any(lengths > X.shape[1]):
        raise ValueError("lengths must lie inside the source tensor")
    horizon = tail_horizon_steps(fs)
    n_steps = int(lengths.max().item()) + horizon
    rollout = torch.zeros(X.shape[0], n_steps, X.shape[2], device=X.device, dtype=X.dtype)
    copy_steps = min(X.shape[1], n_steps)
    rollout[:, :copy_steps] = X[:, :copy_steps]
    valid = torch.arange(n_steps, device=X.device).unsqueeze(0) < lengths.unsqueeze(1)
    return rollout * valid.to(X.dtype).unsqueeze(-1)


def tail_stage_masks(lengths: torch.Tensor, n_steps: int, fs: float) -> tuple[torch.Tensor, ...]:
    b1, b2, b3 = tail_stage_boundaries(fs)
    offsets = ((0, b1), (b1, b2), (b2, b3))
    t = torch.arange(n_steps, device=lengths.device).unsqueeze(0)
    masks: list[torch.Tensor] = []
    for start, stop in offsets:
        masks.append((t >= (lengths + start).unsqueeze(1)) & (t < (lengths + stop).unsqueeze(1)))
    return tuple(masks)


def _mean_activity(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or mask.ndim != 2 or values.shape[:2] != mask.shape:
        raise ValueError("values/mask shape mismatch")
    denominator = mask.to(values.dtype).sum() * float(values.shape[-1])
    if float(denominator.detach().item()) <= 0.0:
        raise ValueError("Activity mask is empty")
    return (values * mask.to(values.dtype).unsqueeze(-1)).sum() / denominator


def tail_activity_terms(
    trajectory: dict[str, object],
    lengths: torch.Tensor,
    fs: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    spikes = trajectory.get("hidden_spikes")
    if not isinstance(spikes, tuple) or not spikes:
        raise TypeError("Expected non-empty hidden_spikes tuple")
    n_steps = spikes[0].shape[1]
    masks = tail_stage_masks(lengths, n_steps, fs)
    stage_values: list[torch.Tensor] = []
    for mask in masks:
        per_layer = [_mean_activity(layer, mask) for layer in spikes]
        stage_values.append(torch.stack(per_layer).mean())
    weighted = sum(weight * value for weight, value in zip(TAIL_STAGE_WEIGHTS, stage_values, strict=True))
    return weighted / TAIL_WEIGHT_SUM, (stage_values[0], stage_values[1], stage_values[2])


def valid_regularization_terms(
    trajectory: dict[str, object],
    lengths: torch.Tensor,
    model: DiagnosticMultiTauHierarchySNN,
) -> tuple[torch.Tensor, torch.Tensor]:
    p2, a1, _ = exp014.regularization_terms(
        trajectory,
        lengths,
        _hidden_alphas(model),
        condition="all_fixes",
        tau=REG_TAU,
    )
    return p2, a1


def base_regularizer(
    profile: str,
    p2: torch.Tensor,
    a1: torch.Tensor,
    tail: torch.Tensor,
) -> torch.Tensor:
    if profile == "a1":
        return BASE_LAMBDA_A1 * a1
    if profile == "a1_p2":
        return BASE_LAMBDA_A1 * a1 + BASE_LAMBDA_P2 * p2
    if profile == "tail":
        return BASE_LAMBDA_TAIL * tail
    if profile == "a1_tail":
        return BASE_LAMBDA_A1 * a1 + BASE_LAMBDA_TAIL * tail
    raise ValueError(f"Unknown regularization profile: {profile}")


def warmup_scale(epoch: int) -> float:
    if epoch < 1:
        raise ValueError("epoch must be >= 1")
    return min(1.0, float(epoch) / float(WARMUP_EPOCHS))


def calibration_loader(
    data: exp01.exp3.Data,
    spec: RunSpec,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    return exp01.exp3.loader(
        data.Xtr,
        data.ytr,
        data.ltr,
        batch_size,
        True,
        paired_seed(spec, "exp0_2_calibration_loader"),
    )


def _task_and_reg(
    spec: RunSpec,
    model: DiagnosticMultiTauHierarchySNN,
    X: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    fs: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    needs_tail = spec.profile in {"tail", "a1_tail"}
    model_input = endpoint_rollout_input(X, lengths, fs) if needs_tail else X
    trajectory = model.forward_trajectory(model_input)
    output_spikes = trajectory["output_spikes"]
    if not isinstance(output_spikes, torch.Tensor):
        raise TypeError("Expected output_spikes tensor")
    task = exp01.exp50.objective_loss(
        output_spikes,
        lengths,
        y,
        spec.objective,
        OUTPUT_CAP,
        fs,
    )
    zero = task.new_zeros(())
    p2 = zero
    a1 = zero
    tail = zero
    if spec.profile in {"a1", "a1_p2", "a1_tail"}:
        p2_all, a1 = valid_regularization_terms(trajectory, lengths, model)
        if spec.profile == "a1_p2":
            p2 = p2_all
    if needs_tail:
        tail, _ = tail_activity_terms(trajectory, lengths, fs)
    reg0 = base_regularizer(spec.profile, p2, a1, tail)
    return task, reg0, {"p2": p2, "a1": a1, "tail": tail}


def calibrate_strength(
    spec: RunSpec,
    model: DiagnosticMultiTauHierarchySNN,
    data: exp01.exp3.Data,
    config: Config,
) -> dict[str, object]:
    params = _hidden_parameters(model)
    rows: list[dict[str, float | int]] = []
    model.eval()
    for batch_index, (X, y, lengths) in enumerate(calibration_loader(data, spec, config.batch_size)):
        if batch_index >= CALIBRATION_BATCHES:
            break
        X = X.to(config.device)
        y = y.to(config.device)
        lengths = lengths.to(config.device)
        task, reg0, terms = _task_and_reg(spec, model, X, y, lengths, data.fs)
        stats = exp015._gradient_pair_stats(task, reg0, params)
        rows.append(
            {
                "batch_index": batch_index,
                "task_loss": float(task.detach().item()),
                "base_regularizer": float(reg0.detach().item()),
                "p2": float(terms["p2"].detach().item()),
                "a1": float(terms["a1"].detach().item()),
                "tail": float(terms["tail"].detach().item()),
                **stats,
            }
        )
    if len(rows) != CALIBRATION_BATCHES:
        raise RuntimeError(f"Expected {CALIBRATION_BATCHES} calibration batches, got {len(rows)}")
    ratios = np.asarray([row["reg_to_task_hidden_grad_ratio"] for row in rows], dtype=float)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0.0):
        raise RuntimeError(f"Invalid calibration ratios for {spec.key}: {ratios}")
    median_ratio = float(np.median(ratios))
    kappa = exp015.calibrated_kappa(TARGET_GRAD_RATIO, median_ratio)
    return {
        "target_grad_ratio": TARGET_GRAD_RATIO,
        "calibration_batches": CALIBRATION_BATCHES,
        "calibration_ratio_median": median_ratio,
        "kappa": kappa,
        "parameter_scope": "hidden linear weights only; output head excluded",
        "loader_seed": paired_seed(spec, "exp0_2_calibration_loader"),
        "rows": rows,
    }


def _validation_metrics(
    model: DiagnosticMultiTauHierarchySNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    objective: str,
    fs: float,
) -> dict[str, float]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    whole_ce_sum = 0.0
    native_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            output = trajectory["output_spikes"]
            if not isinstance(output, torch.Tensor):
                raise TypeError("Expected output_spikes tensor")
            logits = exp01.exp50.deployment_logits(output, lengths, OUTPUT_CAP)
            ce_logits = exp01.exp50.deployment_ce_logits(output, lengths, OUTPUT_CAP)
            whole_ce = F.cross_entropy(ce_logits, y)
            native = exp01.exp50.objective_loss(output, lengths, y, objective, OUTPUT_CAP, fs)
            n = len(y)
            n_total += n
            whole_ce_sum += float(whole_ce.item()) * n
            native_sum += float(native.item()) * n
            labels.append(y.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    metrics = exp01._classification_metrics(np.concatenate(labels), np.concatenate(predictions))
    metrics["whole_count_loss"] = whole_ce_sum / max(n_total, 1)
    metrics["objective_loss"] = native_sum / max(n_total, 1)
    return metrics


def train_one(spec: RunSpec, data: exp01.exp3.Data, config: Config, force: bool) -> Path:
    if spec.profile not in PROFILES:
        raise ValueError("Frozen no-reg controls are evaluation-only")
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp01.exp3.seed_all(paired_seed(spec, "model_init"))
    model = _new_model(spec, data).to(device)
    calibration = calibrate_strength(spec, model, data, config)
    kappa = float(calibration["kappa"])
    _save_json(calibration_path(config.results_dir, spec), calibration)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loaders(data, spec, config.batch_size, train_shuffle=True)["train"]
    val_loader = make_loaders(data, spec, config.batch_size, train_shuffle=False)["val"]

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_whole_ce = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"task": 0.0, "reg0": 0.0, "p2": 0.0, "a1": 0.0, "tail": 0.0, "total": 0.0}
        n_total = 0
        scale = warmup_scale(epoch)
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            task, reg0, terms = _task_and_reg(spec, model, X, y, lengths, data.fs)
            total = task + scale * kappa * reg0
            total.backward()
            optimizer.step()
            n = len(y)
            n_total += n
            totals["task"] += float(task.detach().item()) * n
            totals["reg0"] += float(reg0.detach().item()) * n
            totals["p2"] += float(terms["p2"].detach().item()) * n
            totals["a1"] += float(terms["a1"].detach().item()) * n
            totals["tail"] += float(terms["tail"].detach().item()) * n
            totals["total"] += float(total.detach().item()) * n

        val = _validation_metrics(model, val_loader, device, spec.objective, data.fs)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "warmup_scale": scale,
            "kappa": kappa,
            "train_task_loss": totals["task"] / max(n_total, 1),
            "train_base_regularizer": totals["reg0"] / max(n_total, 1),
            "train_p2": totals["p2"] / max(n_total, 1),
            "train_a1": totals["a1"] / max(n_total, 1),
            "train_tail": totals["tail"] / max(n_total, 1),
            "train_total_loss": totals["total"] / max(n_total, 1),
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_whole_count_loss": val["whole_count_loss"],
            "val_objective_loss": val["objective_loss"],
        }
        history.append(row)
        val_ba = float(val["balanced_accuracy"])
        val_ce = float(val["whole_count_loss"])
        improved = val_ba > best_val_ba + 1e-12 or (
            abs(val_ba - best_val_ba) <= 1e-12 and val_ce < best_val_whole_ce
        )
        if improved:
            best_val_ba = val_ba
            best_val_whole_ce = val_ce
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_ba,
            "best_val_whole_count_loss": best_val_whole_ce,
            "calibration": calibration,
            "model_state_dict": best_state,
            "labels": data.labels,
            "split": data.split,
            "architecture": exp01.architecture_manifest(base_spec(spec), data.fs, len(data.labels)),
        },
        destination,
    )
    history_file = history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False)
    return destination


def _load_new_model(
    spec: RunSpec,
    data: exp01.exp3.Data,
    config: Config,
) -> tuple[DiagnosticMultiTauHierarchySNN, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp0.2 checkpoint: {path}")
    checkpoint = torch.load(path, map_location=config.device, weights_only=False)
    if checkpoint.get("experiment_id") != EXPERIMENT_ID or checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Checkpoint protocol mismatch: {path}")
    if checkpoint.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    model = _new_model(spec, data).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _load_frozen_model(
    spec: RunSpec,
    data: exp01.exp3.Data,
    config: Config,
) -> tuple[DiagnosticMultiTauHierarchySNN, dict[str, object], Path]:
    if spec.profile != FROZEN_PROFILE:
        raise ValueError("Expected frozen profile")
    base = base_spec(spec)
    path = exp01.checkpoint_path(base_results_dir(config.repo_root), base)
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen Exp0.1 checkpoint: {path}")
    checkpoint = torch.load(path, map_location=config.device, weights_only=False)
    if checkpoint.get("experiment_id") != BASE_EXPERIMENT_ID:
        raise ValueError(f"Wrong frozen experiment id: {path}")
    model = _new_model(spec, data).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint, path


def _last_spike_settling_ms(spikes: torch.Tensor, lengths: torch.Tensor, fs: float) -> np.ndarray:
    horizon = tail_horizon_steps(fs)
    result = np.zeros(spikes.shape[0], dtype=float)
    binary_any = spikes.detach().gt(0).any(dim=2)
    for i in range(spikes.shape[0]):
        start = int(lengths[i].item())
        tail = binary_any[i, start : start + horizon]
        indices = torch.nonzero(tail, as_tuple=False).flatten()
        result[i] = 0.0 if len(indices) == 0 else min(horizon, int(indices[-1].item()) + 1) * 1000.0 / fs
    return result


def _shift_slices(width: int, shifts: tuple[int, ...]) -> list[tuple[int, int, int]]:
    base, remainder = divmod(width, len(shifts))
    slices: list[tuple[int, int, int]] = []
    start = 0
    for index, shift in enumerate(shifts):
        count = base + (1 if index < remainder else 0)
        stop = start + count
        slices.append((shift, start, stop))
        start = stop
    if start != width:
        raise RuntimeError("Shift slices do not cover hidden width")
    return slices


def _tail_diagnostics_for_loader(
    model: DiagnosticMultiTauHierarchySNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    fs: float,
) -> dict[str, object]:
    stage_hidden_sum = np.zeros(3, dtype=float)
    stage_last_sum = np.zeros(3, dtype=float)
    stage_output_sum = np.zeros(3, dtype=float)
    stage_syn_sq_sum = np.zeros(3, dtype=float)
    stage_syn_count = np.zeros(3, dtype=float)
    valid_hidden_sum = 0.0
    valid_hidden_count = 0.0
    hidden_tail_spikes = 0.0
    output_tail_spikes = 0.0
    hidden_tail_neuron_steps = 0.0
    output_tail_neuron_steps = 0.0
    settle_hidden: list[float] = []
    settle_output: list[float] = []
    shift_sq: dict[int, float] = {}
    shift_count: dict[int, float] = {}
    n_samples = 0

    model.eval()
    with torch.no_grad():
        for X, _, lengths in loader:
            X = X.to(device)
            lengths = lengths.to(device)
            rollout = endpoint_rollout_input(X, lengths, fs)
            trajectory = model.forward_trajectory(rollout)
            hidden = trajectory["hidden_spikes"]
            synaptic = trajectory["hidden_synaptic"]
            output = trajectory["output_spikes"]
            if not isinstance(hidden, tuple) or not isinstance(synaptic, tuple) or not isinstance(output, torch.Tensor):
                raise TypeError("Malformed diagnostic trajectory")
            masks = tail_stage_masks(lengths, rollout.shape[1], fs)
            valid = exp01.exp50.valid_mask(lengths, rollout.shape[1])

            for layer in hidden:
                valid_hidden_sum += float((layer * valid.to(layer.dtype).unsqueeze(-1)).sum().item())
                valid_hidden_count += float(valid.sum().item() * layer.shape[-1])
            for j, mask in enumerate(masks):
                per_layer = [float(_mean_activity(layer, mask).item()) for layer in hidden]
                stage_hidden_sum[j] += float(np.mean(per_layer)) * len(X)
                stage_last_sum[j] += float(_mean_activity(hidden[-1], mask).item()) * len(X)
                stage_output_sum[j] += float(_mean_activity(output, mask).item()) * len(X)
                last_syn = synaptic[-1]
                mask3 = mask.to(last_syn.dtype).unsqueeze(-1)
                stage_syn_sq_sum[j] += float((last_syn.square() * mask3).sum().item())
                stage_syn_count[j] += float(mask.sum().item() * last_syn.shape[-1])

            tail_mask = torch.stack(masks, dim=0).any(dim=0)
            hidden_tail_spikes += float((hidden[-1] * tail_mask.to(hidden[-1].dtype).unsqueeze(-1)).sum().item())
            output_tail_spikes += float((output * tail_mask.to(output.dtype).unsqueeze(-1)).sum().item())
            hidden_tail_neuron_steps += float(tail_mask.sum().item() * hidden[-1].shape[-1])
            output_tail_neuron_steps += float(tail_mask.sum().item() * output.shape[-1])
            settle_hidden.extend(_last_spike_settling_ms(hidden[-1], lengths, fs).tolist())
            settle_output.extend(_last_spike_settling_ms(output, lengths, fs).tolist())

            last_syn = synaptic[-1]
            shifts = model.layer_shifts[-1]
            for shift, start, stop in _shift_slices(last_syn.shape[-1], shifts):
                values = last_syn[:, :, start:stop]
                for mask in masks:
                    mask3 = mask.to(values.dtype).unsqueeze(-1)
                    shift_sq[shift] = shift_sq.get(shift, 0.0) + float((values.square() * mask3).sum().item())
                    shift_count[shift] = shift_count.get(shift, 0.0) + float(mask.sum().item() * (stop - start))
            n_samples += len(X)

    if n_samples == 0:
        raise RuntimeError("Cannot evaluate empty loader")
    return {
        "valid_hidden_activity": valid_hidden_sum / max(valid_hidden_count, 1.0),
        "tail_stage_hidden_activity": (stage_hidden_sum / n_samples).tolist(),
        "tail_stage_final_hidden_activity": (stage_last_sum / n_samples).tolist(),
        "tail_stage_output_activity": (stage_output_sum / n_samples).tolist(),
        "tail_stage_final_hidden_syn_rms": [
            math.sqrt(stage_syn_sq_sum[j] / max(stage_syn_count[j], 1.0)) for j in range(3)
        ],
        "tail_area_final_hidden_spikes_per_neuron": hidden_tail_spikes / max(hidden_tail_neuron_steps, 1.0) * tail_horizon_steps(fs),
        "tail_area_output_spikes_per_neuron": output_tail_spikes / max(output_tail_neuron_steps, 1.0) * tail_horizon_steps(fs),
        "settling_ms_final_hidden_mean": float(np.mean(settle_hidden)),
        "settling_ms_final_hidden_median": float(np.median(settle_hidden)),
        "settling_ms_output_mean": float(np.mean(settle_output)),
        "settling_ms_output_median": float(np.median(settle_output)),
        "final_hidden_syn_tail_rms_by_shift": {
            str(shift): math.sqrt(shift_sq[shift] / max(shift_count[shift], 1.0))
            for shift in sorted(shift_sq)
        },
    }


def _raster_sample(
    data: exp01.exp3.Data,
    sample_index: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    X = torch.as_tensor(data.Xte[sample_index : sample_index + 1])
    y = int(data.yte[sample_index])
    length = int(data.lte[sample_index])
    lengths = torch.tensor([length], dtype=torch.long)
    return X, lengths, y, sample_index


def _save_raster_and_trace(
    spec: RunSpec,
    model: DiagnosticMultiTauHierarchySNN,
    data: exp01.exp3.Data,
    root: Path,
    device: torch.device,
) -> dict[str, object]:
    X, lengths, label, sample_index = _raster_sample(data, RASTER_SAMPLE_INDEX)
    X = X.to(device)
    lengths = lengths.to(device)
    rollout = endpoint_rollout_input(X, lengths, data.fs)
    with torch.no_grad():
        trajectory = model.forward_trajectory(rollout)
    hidden = trajectory["hidden_spikes"]
    synaptic = trajectory["hidden_synaptic"]
    membrane = trajectory["hidden_membrane"]
    output = trajectory["output_spikes"]
    output_membrane = trajectory["output_membrane"]
    if not all(isinstance(value, tuple) for value in (hidden, synaptic, membrane)) or not isinstance(output, torch.Tensor) or not isinstance(output_membrane, torch.Tensor):
        raise TypeError("Malformed raster trajectory")
    final_hidden = hidden[-1][0].cpu().numpy()
    final_syn = synaptic[-1][0].cpu().numpy()
    final_mem = membrane[-1][0].cpu().numpy()
    output_np = output[0].cpu().numpy()
    output_mem_np = output_membrane[0].cpu().numpy()
    endpoint = int(lengths.item())
    prediction = int(exp01.exp50.deployment_logits(output, lengths, OUTPUT_CAP).argmax(dim=1).item())

    trace_file = trace_path(root, spec)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trace_file,
        final_hidden_spikes=final_hidden,
        output_spikes=output_np,
        final_hidden_synaptic=final_syn,
        final_hidden_membrane=final_mem,
        output_membrane=output_mem_np,
        endpoint=np.asarray(endpoint),
        label=np.asarray(label),
        prediction=np.asarray(prediction),
        fs=np.asarray(data.fs),
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    ts, neurons = np.nonzero(final_hidden > 0.5)
    axes[0].scatter(ts, neurons, s=7, marker=".")
    axes[0].set_ylabel("Final hidden neuron")
    axes[0].set_title(f"{spec.architecture} | {spec.objective} | {spec.profile} | seed {spec.seed}")
    for shift, start, stop in _shift_slices(final_hidden.shape[1], model.layer_shifts[-1]):
        axes[0].axhline(stop - 0.5, linewidth=0.5, alpha=0.4)
        axes[0].text(0.995, (start + stop - 1) / 2, f"s{shift}", transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=8)

    ts_out, neurons_out = np.nonzero(output_np > 0.5)
    axes[1].scatter(ts_out, neurons_out, s=12, marker=".")
    axes[1].set_ylabel("Output neuron")
    axes[1].set_xlabel("Timestep")
    axes[1].set_yticks(np.arange(output_np.shape[1]))

    boundaries = (0,) + tail_stage_boundaries(data.fs)
    for axis in axes:
        axis.axvline(endpoint, linewidth=1.5, linestyle="--")
        for offset in boundaries[1:]:
            axis.axvline(endpoint + offset, linewidth=0.7, linestyle=":", alpha=0.6)
        axis.set_xlim(0, rollout.shape[1] - 1)
    fig_file = raster_path(root, spec)
    fig_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_file, dpi=160)
    plt.close(fig)
    return {
        "sample_index": sample_index,
        "label": label,
        "prediction": prediction,
        "endpoint_timestep": endpoint,
        "rollout_timesteps": int(rollout.shape[1]),
        "png": str(fig_file.relative_to(root)),
        "trace": str(trace_file.relative_to(root)),
    }


def evaluate_model(
    spec: RunSpec,
    model: DiagnosticMultiTauHierarchySNN,
    checkpoint: dict[str, object],
    data: exp01.exp3.Data,
    config: Config,
    source: str,
    checkpoint_source: str,
) -> dict[str, object]:
    loaders = make_loaders(data, spec, config.batch_size, train_shuffle=False)
    device = torch.device(config.device)
    classification = {
        split: _validation_metrics(model, loader, device, spec.objective, data.fs)
        for split, loader in loaders.items()
    }
    tail = {
        split: _tail_diagnostics_for_loader(model, loader, device, data.fs)
        for split, loader in loaders.items()
    }
    raster = _save_raster_and_trace(spec, model, data, config.results_dir, device)
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "source": source,
        "checkpoint_source": checkpoint_source,
        "epochs_trained": int(checkpoint.get("best_epoch", checkpoint.get("epochs_trained", EPOCHS))),
        "classification": classification,
        "tail_diagnostics": tail,
        "raster": raster,
    }


def evaluate_one(spec: RunSpec, data: exp01.exp3.Data, config: Config, force: bool) -> Path:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    if spec.profile == FROZEN_PROFILE:
        model, checkpoint, source_path = _load_frozen_model(spec, data, config)
        source = "frozen_exp0.1"
    else:
        model, checkpoint = _load_new_model(spec, data, config)
        source_path = checkpoint_path(config.results_dir, spec)
        source = "exp0.2"
    payload = evaluate_model(spec, model, checkpoint, data, config, source, str(source_path))
    _save_json(destination, payload)
    return destination


def run_one(spec: RunSpec, data: exp01.exp3.Data, config: Config, force: bool) -> Path:
    train_one(spec, data, config, force=force)
    return evaluate_one(spec, data, config, force=force)


def evaluate_frozen(data: exp01.exp3.Data, config: Config, force: bool) -> list[Path]:
    return [evaluate_one(spec, data, config, force=force) for spec in frozen_specs()]


def _flatten_evaluation(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = payload["spec"]
    test_cls = payload["classification"]["test"]
    test_tail = payload["tail_diagnostics"]["test"]
    row: dict[str, object] = {
        "architecture": spec["architecture"],
        "objective": spec["objective"],
        "profile": spec["profile"],
        "seed": spec["seed"],
        "source": payload["source"],
        "epochs_trained": payload["epochs_trained"],
        "test_accuracy": test_cls["accuracy"],
        "test_balanced_accuracy": test_cls["balanced_accuracy"],
        "test_macro_f1": test_cls["macro_f1"],
        "test_whole_count_loss": test_cls["whole_count_loss"],
        "test_objective_loss": test_cls["objective_loss"],
        "test_valid_hidden_activity": test_tail["valid_hidden_activity"],
        "test_tail_area_final_hidden": test_tail["tail_area_final_hidden_spikes_per_neuron"],
        "test_tail_area_output": test_tail["tail_area_output_spikes_per_neuron"],
        "test_settling_ms_final_hidden_mean": test_tail["settling_ms_final_hidden_mean"],
        "test_settling_ms_output_mean": test_tail["settling_ms_output_mean"],
        "raster_png": payload["raster"]["png"],
        "trace_npz": payload["raster"]["trace"],
    }
    for j, value in enumerate(test_tail["tail_stage_final_hidden_activity"], start=1):
        row[f"test_tail_stage{j}_final_hidden_activity"] = value
    for j, value in enumerate(test_tail["tail_stage_output_activity"], start=1):
        row[f"test_tail_stage{j}_output_activity"] = value
    for j, value in enumerate(test_tail["tail_stage_final_hidden_syn_rms"], start=1):
        row[f"test_tail_stage{j}_final_hidden_syn_rms"] = value
    for shift, value in test_tail["final_hidden_syn_tail_rms_by_shift"].items():
        row[f"test_final_hidden_syn_tail_rms_shift{shift}"] = value
    return row


def finalize_experiment(config: Config) -> dict[str, Path]:
    expected = run_specs() + frozen_specs()
    paths = [evaluation_path(config.results_dir, spec) for spec in expected]
    missing = [path for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} evaluation artifacts; finalizer never regenerates runs:\n{preview}")

    rows = [_flatten_evaluation(path) for path in paths]
    summary = pd.DataFrame(rows).sort_values(["architecture", "objective", "profile", "seed"])
    summary_path = config.results_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    numeric_cols = [
        "test_balanced_accuracy",
        "test_valid_hidden_activity",
        "test_tail_area_final_hidden",
        "test_tail_area_output",
        "test_settling_ms_final_hidden_mean",
        "test_settling_ms_output_mean",
    ]
    grouped = summary.groupby(["architecture", "objective", "profile"], as_index=False)[numeric_cols].agg(["mean", "std"])
    grouped.columns = ["_".join(filter(None, map(str, col))).rstrip("_") for col in grouped.columns.to_flat_index()]
    grouped_path = config.results_dir / "comparison_summary.csv"
    grouped.to_csv(grouped_path, index=False)

    baseline = summary[summary["profile"] == FROZEN_PROFILE][
        ["architecture", "objective", "seed", "test_balanced_accuracy", "test_tail_area_final_hidden", "test_settling_ms_final_hidden_mean"]
    ].rename(
        columns={
            "test_balanced_accuracy": "baseline_test_balanced_accuracy",
            "test_tail_area_final_hidden": "baseline_test_tail_area_final_hidden",
            "test_settling_ms_final_hidden_mean": "baseline_test_settling_ms_final_hidden_mean",
        }
    )
    paired = summary[summary["profile"] != FROZEN_PROFILE].merge(
        baseline, on=["architecture", "objective", "seed"], how="left", validate="many_to_one"
    )
    paired["delta_test_ba_vs_frozen_exp01"] = paired["test_balanced_accuracy"] - paired["baseline_test_balanced_accuracy"]
    paired["delta_tail_area_vs_frozen_exp01"] = paired["test_tail_area_final_hidden"] - paired["baseline_test_tail_area_final_hidden"]
    paired["delta_settling_ms_vs_frozen_exp01"] = paired["test_settling_ms_final_hidden_mean"] - paired["baseline_test_settling_ms_final_hidden_mean"]
    paired_path = config.results_dir / "paired_vs_frozen_exp01.csv"
    paired.to_csv(paired_path, index=False)

    calibration_rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = calibration_path(config.results_dir, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing calibration artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        calibration_rows.append(
            {
                "architecture": spec.architecture,
                "objective": spec.objective,
                "profile": spec.profile,
                "seed": spec.seed,
                "target_grad_ratio": payload["target_grad_ratio"],
                "calibration_ratio_median": payload["calibration_ratio_median"],
                "kappa": payload["kappa"],
            }
        )
    calibration_summary = config.results_dir / "calibration_summary.csv"
    pd.DataFrame(calibration_rows).to_csv(calibration_summary, index=False)

    history_frames: list[pd.DataFrame] = []
    for spec in run_specs():
        path = history_path(config.results_dir, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing history artifact: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "seed", spec.seed)
        frame.insert(0, "profile", spec.profile)
        frame.insert(0, "objective", spec.objective)
        frame.insert(0, "architecture", spec.architecture)
        history_frames.append(frame)
    history_long = config.results_dir / "history_long.csv"
    pd.concat(history_frames, ignore_index=True).to_csv(history_long, index=False)

    raster_index = summary[["architecture", "objective", "profile", "seed", "raster_png", "trace_npz"]].copy()
    raster_index_path = config.results_dir / "raster_index.csv"
    raster_index.to_csv(raster_index_path, index=False)

    manifest_path = config.results_dir / "manifest.json"
    _save_json(manifest_path, experiment_manifest())
    return {
        "summary": summary_path,
        "comparison_summary": grouped_path,
        "paired_vs_frozen": paired_path,
        "calibration_summary": calibration_summary,
        "history_long": history_long,
        "raster_index": raster_index_path,
        "manifest": manifest_path,
    }


def experiment_manifest() -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "new_run_count": len(run_specs()),
        "frozen_evaluation_count": len(frozen_specs()),
        "architectures": list(DIRECT_ARCHITECTURES),
        "objectives": list(OBJECTIVES),
        "profiles": [FROZEN_PROFILE, *PROFILES],
        "seeds": list(SEEDS),
        "epochs_new_runs": EPOCHS,
        "frozen_source": f"{BASE_EXPERIMENT_ID}/{BASE_PROTOCOL_VERSION}",
        "gradient_calibration": {
            "target_ratio": TARGET_GRAD_RATIO,
            "batches": CALIBRATION_BATCHES,
            "scope": "hidden linear weights only",
            "warmup_epochs": WARMUP_EPOCHS,
        },
        "regularization": {
            "a1": "0.1 * Exp0.1.5 all-fixes valid hidden activity",
            "a1_p2": "0.1*A1 + 0.01*Exp0.1.5 all-fixes P2",
            "tail": "0.1 * normalized endpoint-relative hidden spike tail",
            "a1_tail": "0.1*A1 + 0.1*tail",
        },
        "tail": {
            "horizon_ms": TAIL_HORIZON_MS,
            "stage_boundaries_ms": list(TAIL_STAGE_MS),
            "weights": list(TAIL_STAGE_WEIGHTS),
            "normalization": "divide weighted stage activity by sum(weights); mean neuron-step within layer, then mean layers",
            "classification_region": "t < valid_length only",
            "tail_region": "valid_length <= t < valid_length + horizon",
            "post_endpoint_input": "explicit zero input",
            "output_layer_regularized": False,
        },
        "raster": {
            "test_sample_index": RASTER_SAMPLE_INDEX,
            "panels": ["final_hidden_spikes", "output_spikes"],
            "x_axis": "valid segment plus 600-ms zero-input rollout",
            "final_hidden_order": "native neuron order grouped by configured synaptic shift",
        },
    }


def _config_from_args(args: argparse.Namespace) -> Config:
    repo_root = Path(args.repo_root).resolve()
    return Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        epochs=getattr(args, "epochs", EPOCHS),
        batch_size=getattr(args, "batch_size", BATCH_SIZE),
        threads=args.threads,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 0.2 endpoint-tail regularization")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--epochs", type=int, default=EPOCHS)
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run.add_argument("--force", action="store_true")

    frozen = sub.add_parser("eval-frozen")
    frozen.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    frozen.add_argument("--force", action="store_true")

    sub.add_parser("finalize")
    sub.add_parser("describe")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    if args.command == "describe":
        print(json.dumps(experiment_manifest(), indent=2, sort_keys=True))
        return
    if args.command == "finalize":
        outputs = finalize_experiment(config)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return
    data = prepare_data(config.repo_root)
    if args.command == "eval-frozen":
        for path in evaluate_frozen(data, config, force=args.force):
            print(path)
        return
    specs = run_specs()
    if not 0 <= args.array_task_id < len(specs):
        raise IndexError(f"array-task-id {args.array_task_id} outside [0,{len(specs) - 1}]")
    path = run_one(specs[args.array_task_id], data, config, force=args.force)
    print(path)


if __name__ == "__main__":
    main()
