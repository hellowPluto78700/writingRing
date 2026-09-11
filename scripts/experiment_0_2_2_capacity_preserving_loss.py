from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_0_2_endpoint_tail_regularization as exp02


EXPERIMENT_ID = "experiment_0_2_2_capacity_preserving_loss"
PROTOCOL_VERSION = "capacity_preserving_loss_v1"
ARCHITECTURES = ("mid_long", "short_mid_long")
OBJECTIVE = "whole_count_ce"
CONDITIONS = (
    "wc_only",
    "sat",
    "relative_tail",
    "sat_relative_tail",
    "sat_relative_tail_capacity",
)
SEEDS = (11, 23, 37)
EPOCHS = 50
WARMUP_EPOCHS = 5
REG_RAMP_END_EPOCH = 15
TARGET_GRAD_RATIO = 0.05
CALIBRATION_BATCHES = 5
CAPACITY_ETA = 0.70
SATURATION_WINDOW_MS = 250.0
SATURATION_QUANTILE = 0.95
RELATIVE_TAIL_QUANTILE = 0.90
LONG_SHIFTS = (6, 7)
HEALTHY_REFERENCE_ARCHITECTURE = "short_mid"
HEALTHY_REFERENCE_SHIFTS = (4, 5)
TAIL_STAGE_WEIGHTS = (1.0, 2.0, 4.0)
EPS = 1e-8
GRAD_EPS = 1e-12
BATCH_SIZE = exp02.BATCH_SIZE
LR = exp02.LR
WEIGHT_DECAY = exp02.WEIGHT_DECAY
HIDDEN_CAP = exp02.HIDDEN_CAP
OUTPUT_CAP = exp02.OUTPUT_CAP
HISTORY_COLUMNS = (
    "architecture",
    "condition",
    "seed",
    "epoch",
    "train_balanced_accuracy",
    "val_balanced_accuracy",
    "train_task_ce",
    "val_task_ce",
    "train_optim_task_loss",
    "train_optim_raw_reg_loss",
    "train_optim_weighted_reg_loss",
    "train_optim_total_loss",
    "train_sat_loss",
    "train_relative_tail_loss",
    "train_capacity_loss",
    "train_weighted_sat_loss",
    "train_weighted_relative_tail_loss",
    "train_weighted_capacity_loss",
    "regularizer_warmup_scale",
    "gradient_calibration_kappa",
    "train_long_valid_fr_hz",
    "train_long_tail_fr_hz",
    "train_long_tail_valid_ratio",
    "val_long_valid_fr_hz",
    "val_long_tail_fr_hz",
    "val_long_tail_valid_ratio",
    "output_weight_norm_s2",
    "output_weight_norm_s3",
    "output_weight_norm_s4",
    "output_weight_norm_s5",
    "output_weight_norm_s6",
    "output_weight_norm_s7",
    "output_weight_energy_long_fraction",
)


@dataclass(frozen=True)
class WarmupSpec:
    architecture: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.architecture}__seed{self.seed}"


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    condition: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.architecture}__{OBJECTIVE}__{self.condition}__seed{self.seed}"

    @property
    def warmup_spec(self) -> WarmupSpec:
        return WarmupSpec(self.architecture, self.seed)


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = BATCH_SIZE
    threads: int = 1


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def reference_path(root: Path) -> Path:
    return root / "reference" / "healthy_reference.json"


def warmup_path(root: Path, spec: WarmupSpec) -> Path:
    return root / "warmups" / f"{spec.key}.pt"


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def shift_history_path(root: Path, spec: RunSpec) -> Path:
    return root / "shift_histories" / f"{spec.key}.csv"


def calibration_path(root: Path, spec: RunSpec) -> Path:
    return root / "calibrations" / f"{spec.key}.json"


def plot_path(root: Path, spec: RunSpec) -> Path:
    return root / "plots" / spec.architecture / spec.condition / f"seed{spec.seed}.png"


def warmup_specs() -> list[WarmupSpec]:
    return [WarmupSpec(architecture, seed) for architecture in ARCHITECTURES for seed in SEEDS]


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(architecture, condition, seed)
        for architecture in ARCHITECTURES
        for condition in CONDITIONS
        for seed in SEEDS
    ]


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_data(repo_root: Path) -> exp01.exp3.Data:
    data = exp02.prepare_data(repo_root)
    if exp01.exp3.SPLIT_SEED != 12345:
        raise ValueError(f"Exp0.2.2 requires split seed 12345, got {exp01.exp3.SPLIT_SEED}")
    return data


def _model_spec(architecture: str, seed: int, profile: str = "tail") -> exp02.RunSpec:
    return exp02.RunSpec(architecture, OBJECTIVE, profile, seed)


def _new_model(architecture: str, seed: int, data: exp01.exp3.Data) -> exp02.DiagnosticMultiTauHierarchySNN:
    return exp02._new_model(_model_spec(architecture, seed), data)


def _paired_seed(architecture: str, seed: int, role: str) -> int:
    return exp02.paired_seed(_model_spec(architecture, seed), role)


def _partition(data: exp01.exp3.Data, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split == "train":
        return data.Xtr, data.ytr, data.ltr
    if split == "val":
        return data.Xva, data.yva, data.lva
    if split == "test":
        return data.Xte, data.yte, data.lte
    raise ValueError(split)


def _loader(
    data: exp01.exp3.Data,
    split: str,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    include_index: bool,
) -> DataLoader:
    X, y, lengths = _partition(data, split)
    tensors: list[torch.Tensor] = [
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
        torch.as_tensor(lengths, dtype=torch.long),
    ]
    if include_index:
        tensors.append(torch.arange(len(X), dtype=torch.long))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def _train_loader(data: exp01.exp3.Data, spec: WarmupSpec, epoch: int, batch_size: int) -> DataLoader:
    seed = _paired_seed(spec.architecture, spec.seed, f"exp0_2_2_train_epoch_{epoch}")
    return _loader(data, "train", batch_size, shuffle=True, seed=seed, include_index=True)


def _eval_loader(data: exp01.exp3.Data, split: str, spec: WarmupSpec, batch_size: int) -> DataLoader:
    seed = _paired_seed(spec.architecture, spec.seed, f"exp0_2_2_eval_{split}")
    return _loader(data, split, batch_size, shuffle=False, seed=seed, include_index=True)


def _shift_slices(architecture: str) -> dict[int, tuple[int, int]]:
    shifts = tuple(int(v) for v in exp02.DIRECT_ARCHITECTURES[architecture][-1])
    slices = exp02._shift_slices(exp02.HIDDEN_WIDTH, shifts)
    return {int(shift): (int(start), int(stop)) for shift, start, stop in slices}


def _selected_slice_indices(architecture: str, shifts: Iterable[int]) -> list[tuple[int, int, int]]:
    mapping = _shift_slices(architecture)
    selected: list[tuple[int, int, int]] = []
    for shift in shifts:
        if shift not in mapping:
            raise ValueError(f"Architecture {architecture} does not contain shift {shift}")
        start, stop = mapping[shift]
        selected.append((shift, start, stop))
    return selected


def regularizer_scale(epoch: int) -> float:
    if epoch <= WARMUP_EPOCHS:
        return 0.0
    if epoch >= REG_RAMP_END_EPOCH:
        return 1.0
    return float(epoch - WARMUP_EPOCHS) / float(REG_RAMP_END_EPOCH - WARMUP_EPOCHS)


def _task_loss(
    output_spikes: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
    fs: float,
) -> torch.Tensor:
    return exp01.exp50.objective_loss(
        output_spikes,
        lengths,
        y,
        OBJECTIVE,
        OUTPUT_CAP,
        fs,
    )


def _valid_mask(lengths: torch.Tensor, n_steps: int) -> torch.Tensor:
    return exp01.exp50.valid_mask(lengths, n_steps)


def _rate_per_sample(
    spikes: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    stop: int,
    fs: float,
) -> torch.Tensor:
    selected = spikes[:, :, start:stop]
    counts = (selected * mask.to(selected.dtype).unsqueeze(-1)).sum(dim=(1, 2))
    steps = mask.to(selected.dtype).sum(dim=1)
    denom = steps * float(stop - start)
    return torch.where(denom > 0, counts * float(fs) / denom.clamp_min(1.0), torch.zeros_like(counts))


def _pooled_rate(
    spikes: torch.Tensor,
    mask: torch.Tensor,
    slices: list[tuple[int, int, int]],
    fs: float,
) -> tuple[float, float, float]:
    spike_sum = 0.0
    neuron_steps = 0.0
    for _, start, stop in slices:
        selected = spikes[:, :, start:stop]
        mask_f = mask.to(selected.dtype).unsqueeze(-1)
        spike_sum += float((selected * mask_f).sum().item())
        neuron_steps += float(mask.sum().item()) * float(stop - start)
    hz = spike_sum * float(fs) / neuron_steps if neuron_steps > 0 else float("nan")
    return hz, spike_sum, neuron_steps


def saturation_loss(
    final_hidden_spikes: torch.Tensor,
    lengths: torch.Tensor,
    architecture: str,
    fs: float,
    rho_max: float,
) -> torch.Tensor:
    window = max(1, int(np.rint(SATURATION_WINDOW_MS * float(fs) / 1000.0)))
    if final_hidden_spikes.shape[1] < window:
        return final_hidden_spikes.new_zeros(())
    penalties: list[torch.Tensor] = []
    end_times = torch.arange(window - 1, final_hidden_spikes.shape[1], device=lengths.device).unsqueeze(0)
    valid_windows = end_times < lengths.unsqueeze(1)
    for _, start, stop in _selected_slice_indices(architecture, LONG_SHIFTS):
        selected = final_hidden_spikes[:, :, start:stop].transpose(1, 2)
        occupancy = F.avg_pool1d(selected, kernel_size=window, stride=1).transpose(1, 2)
        mask = valid_windows.unsqueeze(-1).expand_as(occupancy)
        if bool(mask.any()):
            penalties.append(F.relu(occupancy[mask] - float(rho_max)).pow(2).mean())
    if not penalties:
        return final_hidden_spikes.new_zeros(())
    return torch.stack(penalties).mean()


def relative_tail_loss(
    final_hidden_spikes: torch.Tensor,
    lengths: torch.Tensor,
    architecture: str,
    fs: float,
    gamma: tuple[float, float, float],
) -> torch.Tensor:
    valid = _valid_mask(lengths, final_hidden_spikes.shape[1])
    stages = exp02.tail_stage_masks(lengths, final_hidden_spikes.shape[1], fs)
    per_shift: list[torch.Tensor] = []
    for _, start, stop in _selected_slice_indices(architecture, LONG_SHIFTS):
        valid_rate = _rate_per_sample(final_hidden_spikes, valid, start, stop, fs)
        stage_losses: list[torch.Tensor] = []
        for weight, target, mask in zip(TAIL_STAGE_WEIGHTS, gamma, stages, strict=True):
            tail_rate = _rate_per_sample(final_hidden_spikes, mask, start, stop, fs)
            ratio = tail_rate / valid_rate.clamp_min(1e-4)
            stage_losses.append(float(weight) * F.relu(ratio - float(target)).pow(2).mean())
        per_shift.append(sum(stage_losses) / float(sum(TAIL_STAGE_WEIGHTS)))
    return torch.stack(per_shift).mean()


def capacity_floor_loss(
    final_hidden_spikes: torch.Tensor,
    lengths: torch.Tensor,
    sample_indices: torch.Tensor,
    architecture: str,
    fs: float,
    warmup_reference_hz: dict[int, torch.Tensor],
) -> torch.Tensor:
    valid = _valid_mask(lengths, final_hidden_spikes.shape[1])
    losses: list[torch.Tensor] = []
    for shift, start, stop in _selected_slice_indices(architecture, LONG_SHIFTS):
        current = _rate_per_sample(final_hidden_spikes, valid, start, stop, fs)
        reference_all = warmup_reference_hz[shift].to(current.device)
        reference = reference_all[sample_indices]
        active = reference > 1e-6
        if bool(active.any()):
            floor = CAPACITY_ETA * reference[active]
            normalized_deficit = F.relu(floor - current[active]) / reference[active].clamp_min(1e-4)
            losses.append(normalized_deficit.pow(2).mean())
    if not losses:
        return final_hidden_spikes.new_zeros(())
    return torch.stack(losses).mean()


def condition_components(condition: str) -> tuple[bool, bool, bool]:
    mapping = {
        "wc_only": (False, False, False),
        "sat": (True, False, False),
        "relative_tail": (False, True, False),
        "sat_relative_tail": (True, True, False),
        "sat_relative_tail_capacity": (True, True, True),
    }
    if condition not in mapping:
        raise ValueError(condition)
    return mapping[condition]


def _forward_losses(
    model: exp02.DiagnosticMultiTauHierarchySNN,
    X: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    sample_indices: torch.Tensor,
    spec: RunSpec,
    fs: float,
    reference: dict[str, object],
    warmup_reference_hz: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    use_sat, use_rel, use_cap = condition_components(spec.condition)
    needs_tail = use_rel
    model_input = exp02.endpoint_rollout_input(X, lengths, fs) if needs_tail else X
    trajectory = model.forward_trajectory(model_input)
    output = trajectory.get("output_spikes")
    hidden = trajectory.get("hidden_spikes")
    if not isinstance(output, torch.Tensor) or not isinstance(hidden, tuple) or not hidden:
        raise TypeError("Unexpected SNN trajectory")
    task = _task_loss(output, lengths, y, fs)
    final_hidden = hidden[-1]
    zero = task.new_zeros(())
    sat = saturation_loss(final_hidden, lengths, spec.architecture, fs, float(reference["rho_max"])) if use_sat else zero
    gamma_values = tuple(float(v) for v in reference["gamma"])
    rel = relative_tail_loss(final_hidden, lengths, spec.architecture, fs, gamma_values) if use_rel else zero
    cap = capacity_floor_loss(
        final_hidden,
        lengths,
        sample_indices,
        spec.architecture,
        fs,
        warmup_reference_hz,
    ) if use_cap else zero
    reg = sat + rel + cap
    return task, reg, {"sat": sat, "relative_tail": rel, "capacity": cap}


def _output_weight_stats(
    model: exp02.DiagnosticMultiTauHierarchySNN,
    architecture: str,
) -> dict[str, float]:
    if model.output_linear is None:
        raise RuntimeError("Missing output linear layer")
    weight = model.output_linear.weight.detach()
    per_neuron = torch.linalg.vector_norm(weight, dim=0)
    stats: dict[str, float] = {}
    mapping = _shift_slices(architecture)
    for shift in range(2, 8):
        if shift in mapping:
            start, stop = mapping[shift]
            stats[f"output_weight_norm_s{shift}"] = float(per_neuron[start:stop].mean().item())
        else:
            stats[f"output_weight_norm_s{shift}"] = float("nan")
    total_energy = float(weight.pow(2).sum().item())
    long_energy = 0.0
    for shift in LONG_SHIFTS:
        start, stop = mapping[shift]
        long_energy += float(weight[:, start:stop].pow(2).sum().item())
    stats["output_weight_energy_long_fraction"] = long_energy / total_energy if total_energy > 0 else float("nan")
    return stats


def evaluate_split(
    model: exp02.DiagnosticMultiTauHierarchySNN,
    data: exp01.exp3.Data,
    split: str,
    warmup_spec: WarmupSpec,
    config: Config,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    model.eval()
    device = torch.device(config.device)
    loader = _eval_loader(data, split, warmup_spec, config.batch_size)
    all_y: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    task_sum = 0.0
    n_total = 0
    shift_accum: dict[int, dict[str, float]] = {
        shift: {"valid_spikes": 0.0, "valid_neuron_steps": 0.0, "tail_spikes": 0.0, "tail_neuron_steps": 0.0}
        for shift in LONG_SHIFTS
    }
    stage_accum: dict[int, list[dict[str, float]]] = {
        shift: [{"spikes": 0.0, "neuron_steps": 0.0} for _ in range(3)] for shift in LONG_SHIFTS
    }
    with torch.no_grad():
        for batch in loader:
            X, y, lengths, _ = batch
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            rollout = exp02.endpoint_rollout_input(X, lengths, data.fs)
            trajectory = model.forward_trajectory(rollout)
            output = trajectory["output_spikes"]
            hidden = trajectory["hidden_spikes"]
            if not isinstance(output, torch.Tensor) or not isinstance(hidden, tuple):
                raise TypeError("Unexpected trajectory")
            task = _task_loss(output, lengths, y, data.fs)
            task_sum += float(task.item()) * len(X)
            n_total += len(X)
            valid_out = _valid_mask(lengths, output.shape[1]).to(output.dtype).unsqueeze(-1)
            counts = (output * valid_out).sum(dim=1)
            pred = counts.argmax(dim=1)
            all_y.append(y.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())

            final_hidden = hidden[-1]
            valid = _valid_mask(lengths, final_hidden.shape[1])
            stages = exp02.tail_stage_masks(lengths, final_hidden.shape[1], data.fs)
            for shift, start, stop in _selected_slice_indices(warmup_spec.architecture, LONG_SHIFTS):
                valid_hz, valid_spikes, valid_steps = _pooled_rate(final_hidden, valid, [(shift, start, stop)], data.fs)
                del valid_hz
                shift_accum[shift]["valid_spikes"] += valid_spikes
                shift_accum[shift]["valid_neuron_steps"] += valid_steps
                for index, mask in enumerate(stages):
                    _, spikes_sum, neuron_steps = _pooled_rate(final_hidden, mask, [(shift, start, stop)], data.fs)
                    stage_accum[shift][index]["spikes"] += spikes_sum
                    stage_accum[shift][index]["neuron_steps"] += neuron_steps
                    shift_accum[shift]["tail_spikes"] += spikes_sum
                    shift_accum[shift]["tail_neuron_steps"] += neuron_steps

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    shift_rows: list[dict[str, float | int | str]] = []
    pooled_valid_spikes = pooled_valid_steps = pooled_tail_spikes = pooled_tail_steps = 0.0
    for shift in LONG_SHIFTS:
        acc = shift_accum[shift]
        valid_rate = acc["valid_spikes"] * data.fs / acc["valid_neuron_steps"] if acc["valid_neuron_steps"] else float("nan")
        tail_rate = acc["tail_spikes"] * data.fs / acc["tail_neuron_steps"] if acc["tail_neuron_steps"] else float("nan")
        row: dict[str, float | int | str] = {
            "split": split,
            "shift": shift,
            "valid_firing_rate_hz": valid_rate,
            "tail_firing_rate_hz": tail_rate,
            "tail_valid_ratio": tail_rate / valid_rate if valid_rate > 0 else float("nan"),
        }
        for index in range(3):
            item = stage_accum[shift][index]
            row[f"tail_stage{index + 1}_firing_rate_hz"] = (
                item["spikes"] * data.fs / item["neuron_steps"] if item["neuron_steps"] else float("nan")
            )
        shift_rows.append(row)
        pooled_valid_spikes += acc["valid_spikes"]
        pooled_valid_steps += acc["valid_neuron_steps"]
        pooled_tail_spikes += acc["tail_spikes"]
        pooled_tail_steps += acc["tail_neuron_steps"]

    long_valid = pooled_valid_spikes * data.fs / pooled_valid_steps if pooled_valid_steps else float("nan")
    long_tail = pooled_tail_spikes * data.fs / pooled_tail_steps if pooled_tail_steps else float("nan")
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "task_ce": task_sum / max(n_total, 1),
        "long_valid_fr_hz": long_valid,
        "long_tail_fr_hz": long_tail,
        "long_tail_valid_ratio": long_tail / long_valid if long_valid > 0 else float("nan"),
    }
    metrics.update(_output_weight_stats(model, warmup_spec.architecture))
    return metrics, shift_rows


def _per_sample_long_reference(
    model: exp02.DiagnosticMultiTauHierarchySNN,
    data: exp01.exp3.Data,
    spec: WarmupSpec,
    config: Config,
) -> dict[int, torch.Tensor]:
    model.eval()
    device = torch.device(config.device)
    loader = _eval_loader(data, "train", spec, config.batch_size)
    result = {shift: torch.zeros(len(data.Xtr), dtype=torch.float32) for shift in LONG_SHIFTS}
    with torch.no_grad():
        for X, _, lengths, indices in loader:
            X = X.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            hidden = trajectory["hidden_spikes"]
            if not isinstance(hidden, tuple):
                raise TypeError("Unexpected hidden_spikes")
            final_hidden = hidden[-1]
            valid = _valid_mask(lengths, final_hidden.shape[1])
            for shift, start, stop in _selected_slice_indices(spec.architecture, LONG_SHIFTS):
                values = _rate_per_sample(final_hidden, valid, start, stop, data.fs).detach().cpu()
                result[shift][indices] = values
    return result


def _snapshot_rng() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def _restore_rng(state: dict[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]


def _seed_model(architecture: str, seed: int) -> None:
    value = _paired_seed(architecture, seed, "exp0_2_2_model_init")
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)


def _train_task_epoch(
    model: exp02.DiagnosticMultiTauHierarchySNN,
    optimizer: torch.optim.Optimizer,
    data: exp01.exp3.Data,
    spec: WarmupSpec,
    epoch: int,
    config: Config,
) -> dict[str, float]:
    model.train()
    total_task = 0.0
    n_batches = 0
    for X, y, lengths, _ in _train_loader(data, spec, epoch, config.batch_size):
        X = X.to(config.device)
        y = y.to(config.device)
        lengths = lengths.to(config.device)
        trajectory = model.forward_trajectory(X)
        output = trajectory["output_spikes"]
        if not isinstance(output, torch.Tensor):
            raise TypeError("Unexpected output_spikes")
        task = _task_loss(output, lengths, y, data.fs)
        optimizer.zero_grad(set_to_none=True)
        task.backward()
        optimizer.step()
        total_task += float(task.item())
        n_batches += 1
    mean_task = total_task / max(n_batches, 1)
    return {
        "train_optim_task_loss": mean_task,
        "train_optim_raw_reg_loss": 0.0,
        "train_optim_weighted_reg_loss": 0.0,
        "train_optim_total_loss": mean_task,
        "train_sat_loss": 0.0,
        "train_relative_tail_loss": 0.0,
        "train_capacity_loss": 0.0,
        "train_weighted_sat_loss": 0.0,
        "train_weighted_relative_tail_loss": 0.0,
        "train_weighted_capacity_loss": 0.0,
    }


def _history_row(
    spec: RunSpec,
    epoch: int,
    train_eval: dict[str, float],
    val_eval: dict[str, float],
    optim: dict[str, float],
    kappa: float,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "architecture": spec.architecture,
        "condition": spec.condition,
        "seed": spec.seed,
        "epoch": epoch,
        "train_balanced_accuracy": train_eval["balanced_accuracy"],
        "val_balanced_accuracy": val_eval["balanced_accuracy"],
        "train_task_ce": train_eval["task_ce"],
        "val_task_ce": val_eval["task_ce"],
        **optim,
        "regularizer_warmup_scale": regularizer_scale(epoch),
        "gradient_calibration_kappa": kappa,
        "train_long_valid_fr_hz": train_eval["long_valid_fr_hz"],
        "train_long_tail_fr_hz": train_eval["long_tail_fr_hz"],
        "train_long_tail_valid_ratio": train_eval["long_tail_valid_ratio"],
        "val_long_valid_fr_hz": val_eval["long_valid_fr_hz"],
        "val_long_tail_fr_hz": val_eval["long_tail_fr_hz"],
        "val_long_tail_valid_ratio": val_eval["long_tail_valid_ratio"],
    }
    for shift in range(2, 8):
        row[f"output_weight_norm_s{shift}"] = val_eval[f"output_weight_norm_s{shift}"]
    row["output_weight_energy_long_fraction"] = val_eval["output_weight_energy_long_fraction"]
    return row


def extract_healthy_reference(data: exp01.exp3.Data, config: Config, force: bool) -> Path:
    destination = reference_path(config.results_dir)
    if destination.exists() and not force:
        return destination
    occupancy_values: list[np.ndarray] = []
    ratios: list[list[float]] = [[], [], []]
    window = max(1, int(np.rint(SATURATION_WINDOW_MS * data.fs / 1000.0)))
    for seed in SEEDS:
        source_spec = exp02.RunSpec(HEALTHY_REFERENCE_ARCHITECTURE, OBJECTIVE, exp02.FROZEN_PROFILE, seed)
        source_config = exp02.Config(
            repo_root=config.repo_root,
            results_dir=exp02.results_dir(config.repo_root),
            device=config.device,
            batch_size=config.batch_size,
            threads=config.threads,
        )
        model, checkpoint, source_path = exp02._load_frozen_model(source_spec, data, source_config)
        del checkpoint
        model.eval()
        loader = _eval_loader(data, "train", WarmupSpec(HEALTHY_REFERENCE_ARCHITECTURE, seed), config.batch_size)
        with torch.no_grad():
            for X, _, lengths, _ in loader:
                X = X.to(config.device)
                lengths = lengths.to(config.device)
                rollout = exp02.endpoint_rollout_input(X, lengths, data.fs)
                trajectory = model.forward_trajectory(rollout)
                hidden = trajectory["hidden_spikes"]
                if not isinstance(hidden, tuple):
                    raise TypeError("Unexpected hidden_spikes")
                final_hidden = hidden[-1]
                valid = _valid_mask(lengths, final_hidden.shape[1])
                stages = exp02.tail_stage_masks(lengths, final_hidden.shape[1], data.fs)
                mapping = _shift_slices(HEALTHY_REFERENCE_ARCHITECTURE)
                selected_parts = [final_hidden[:, :, mapping[s][0]:mapping[s][1]] for s in HEALTHY_REFERENCE_SHIFTS]
                selected = torch.cat(selected_parts, dim=2)
                if selected.shape[1] >= window:
                    occ = F.avg_pool1d(selected.transpose(1, 2), kernel_size=window, stride=1).transpose(1, 2)
                    end_times = torch.arange(window - 1, selected.shape[1], device=lengths.device).unsqueeze(0)
                    mask = (end_times < lengths.unsqueeze(1)).unsqueeze(-1).expand_as(occ)
                    occupancy_values.append(occ[mask].detach().cpu().numpy())
                valid_rate_parts = []
                stage_rate_parts: list[list[torch.Tensor]] = [[], [], []]
                for shift in HEALTHY_REFERENCE_SHIFTS:
                    start, stop = mapping[shift]
                    valid_rate_parts.append(_rate_per_sample(final_hidden, valid, start, stop, data.fs))
                    for stage_index, stage_mask in enumerate(stages):
                        stage_rate_parts[stage_index].append(
                            _rate_per_sample(final_hidden, stage_mask, start, stop, data.fs)
                        )
                valid_rate = torch.stack(valid_rate_parts, dim=0).mean(dim=0)
                for stage_index in range(3):
                    tail_rate = torch.stack(stage_rate_parts[stage_index], dim=0).mean(dim=0)
                    good = valid_rate > 1e-4
                    if bool(good.any()):
                        ratios[stage_index].extend((tail_rate[good] / valid_rate[good]).detach().cpu().tolist())
        print(f"[exp0.2.2] healthy reference seed={seed} checkpoint={source_path}")
    if not occupancy_values or any(not values for values in ratios):
        raise RuntimeError("Healthy-reference extraction produced empty statistics")
    rho_max = float(np.quantile(np.concatenate(occupancy_values), SATURATION_QUANTILE))
    gamma = [float(np.quantile(np.asarray(values), RELATIVE_TAIL_QUANTILE)) for values in ratios]
    payload = {
        "source_experiment": exp02.BASE_EXPERIMENT_ID,
        "source_architecture": HEALTHY_REFERENCE_ARCHITECTURE,
        "source_objective": OBJECTIVE,
        "source_profile": exp02.FROZEN_PROFILE,
        "source_seeds": list(SEEDS),
        "source_split": "train",
        "healthy_shifts": list(HEALTHY_REFERENCE_SHIFTS),
        "saturation_window_ms": SATURATION_WINDOW_MS,
        "saturation_window_steps": window,
        "saturation_quantile": SATURATION_QUANTILE,
        "rho_max": rho_max,
        "relative_tail_quantile": RELATIVE_TAIL_QUANTILE,
        "gamma": gamma,
        "tail_stage_ms": list(exp02.TAIL_STAGE_MS),
        "tail_stage_weights": list(TAIL_STAGE_WEIGHTS),
        "fs_hz": float(data.fs),
        "test_data_used": False,
    }
    _save_json(destination, payload)
    print(f"[exp0.2.2] healthy reference -> {destination}: rho={rho_max:.6g}, gamma={gamma}")
    return destination


def _load_reference(root: Path) -> dict[str, object]:
    path = reference_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Missing healthy reference: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_warmup(data: exp01.exp3.Data, spec: WarmupSpec, config: Config, force: bool) -> Path:
    destination = warmup_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    _seed_model(spec.architecture, spec.seed)
    model = _new_model(spec.architecture, spec.seed, data).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, float | int | str]] = []
    shift_history: list[dict[str, float | int | str]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_ce = float("inf")
    base_run = RunSpec(spec.architecture, "wc_only", spec.seed)
    for epoch in range(1, WARMUP_EPOCHS + 1):
        optim = _train_task_epoch(model, optimizer, data, spec, epoch, config)
        train_eval, train_shift = evaluate_split(model, data, "train", spec, config)
        val_eval, val_shift = evaluate_split(model, data, "val", spec, config)
        history.append(_history_row(base_run, epoch, train_eval, val_eval, optim, 0.0))
        for split_name, rows in (("train", train_shift), ("val", val_shift)):
            for row in rows:
                shift_history.append({"architecture": spec.architecture, "condition": "wc_only", "seed": spec.seed, "epoch": epoch, "split": split_name, **row})
        val_ba = val_eval["balanced_accuracy"]
        val_ce = val_eval["task_ce"]
        if val_ba > best_val_ba + 1e-12 or (abs(val_ba - best_val_ba) <= 1e-12 and val_ce < best_val_ce):
            best_val_ba = val_ba
            best_val_ce = val_ce
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Warmup did not select a checkpoint")
    warmup_reference_hz = _per_sample_long_reference(model, data, spec, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "warmup_spec": spec.__dict__,
            "epoch": WARMUP_EPOCHS,
            "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _snapshot_rng(),
            "history": history,
            "shift_history": shift_history,
            "best_epoch_through_warmup": best_epoch,
            "best_val_ba_through_warmup": best_val_ba,
            "best_val_ce_through_warmup": best_val_ce,
            "best_state_through_warmup": best_state,
            "train_long_valid_rate_hz_by_shift": warmup_reference_hz,
        },
        destination,
    )
    print(f"[exp0.2.2] warmup {spec.key} -> {destination}")
    return destination


def _load_warmup(root: Path, spec: WarmupSpec) -> dict[str, object]:
    path = warmup_path(root, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing warmup checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Warmup identity mismatch: {path}")
    if payload.get("warmup_spec") != spec.__dict__:
        raise ValueError(f"Warmup spec mismatch: {path}")
    if int(payload.get("epoch", -1)) != WARMUP_EPOCHS:
        raise ValueError(f"Warmup epoch mismatch: {path}")
    return payload


def _long_incoming_weight(model: exp02.DiagnosticMultiTauHierarchySNN, architecture: str) -> tuple[torch.nn.Parameter, list[slice]]:
    parameter = model.hidden_linears[-1].weight
    slices = [slice(start, stop) for _, start, stop in _selected_slice_indices(architecture, LONG_SHIFTS)]
    return parameter, slices


def _sliced_grad_norm(grad: torch.Tensor | None, slices: list[slice]) -> float:
    if grad is None:
        return 0.0
    values = torch.cat([grad[slc].reshape(-1) for slc in slices])
    return float(torch.linalg.vector_norm(values).detach().item())


def calibrate_regularizer(
    model: exp02.DiagnosticMultiTauHierarchySNN,
    data: exp01.exp3.Data,
    spec: RunSpec,
    config: Config,
    reference: dict[str, object],
    warmup_reference_hz: dict[int, torch.Tensor],
) -> dict[str, object]:
    if spec.condition == "wc_only":
        return {"kappa": 0.0, "target_grad_ratio": TARGET_GRAD_RATIO, "ratios": [], "calibration_batches": 0}
    model.train()
    parameter, slices = _long_incoming_weight(model, spec.architecture)
    loader = _loader(
        data,
        "train",
        config.batch_size,
        shuffle=True,
        seed=_paired_seed(spec.architecture, spec.seed, f"exp0_2_2_calibration_{spec.condition}"),
        include_index=True,
    )
    ratios: list[float] = []
    task_norms: list[float] = []
    reg_norms: list[float] = []
    for batch_index, (X, y, lengths, indices) in enumerate(loader):
        if batch_index >= CALIBRATION_BATCHES:
            break
        X = X.to(config.device)
        y = y.to(config.device)
        lengths = lengths.to(config.device)
        indices = indices.to(config.device)
        task, reg, _ = _forward_losses(model, X, y, lengths, indices, spec, data.fs, reference, warmup_reference_hz)
        task_grad = torch.autograd.grad(task, parameter, retain_graph=True, allow_unused=True)[0]
        reg_grad = torch.autograd.grad(reg, parameter, retain_graph=False, allow_unused=True)[0]
        task_norm = _sliced_grad_norm(task_grad, slices)
        reg_norm = _sliced_grad_norm(reg_grad, slices)
        task_norms.append(task_norm)
        reg_norms.append(reg_norm)
        if task_norm > GRAD_EPS:
            ratios.append(reg_norm / task_norm)
    if not ratios:
        raise RuntimeError(f"No valid task gradients during calibration for {spec.key}")
    median_ratio = float(np.median(ratios))
    if not math.isfinite(median_ratio) or median_ratio <= GRAD_EPS:
        raise RuntimeError(f"Regularizer gradient is effectively zero for {spec.key}: median_ratio={median_ratio}")
    kappa = TARGET_GRAD_RATIO / median_ratio
    return {
        "kappa": kappa,
        "target_grad_ratio": TARGET_GRAD_RATIO,
        "median_raw_grad_ratio": median_ratio,
        "ratios": ratios,
        "task_grad_norms": task_norms,
        "reg_grad_norms": reg_norms,
        "calibration_batches": len(ratios),
        "parameter_scope": "final-hidden incoming-weight rows assigned to s6/s7",
    }


def _train_regularized_epoch(
    model: exp02.DiagnosticMultiTauHierarchySNN,
    optimizer: torch.optim.Optimizer,
    data: exp01.exp3.Data,
    spec: RunSpec,
    epoch: int,
    config: Config,
    reference: dict[str, object],
    warmup_reference_hz: dict[int, torch.Tensor],
    kappa: float,
) -> dict[str, float]:
    model.train()
    sums = {
        "task": 0.0,
        "raw_reg": 0.0,
        "weighted_reg": 0.0,
        "total": 0.0,
        "sat": 0.0,
        "relative_tail": 0.0,
        "capacity": 0.0,
        "weighted_sat": 0.0,
        "weighted_relative_tail": 0.0,
        "weighted_capacity": 0.0,
    }
    n_batches = 0
    scale = regularizer_scale(epoch)
    for X, y, lengths, indices in _train_loader(data, spec.warmup_spec, epoch, config.batch_size):
        X = X.to(config.device)
        y = y.to(config.device)
        lengths = lengths.to(config.device)
        indices = indices.to(config.device)
        task, reg, components = _forward_losses(
            model, X, y, lengths, indices, spec, data.fs, reference, warmup_reference_hz
        )
        weighted_reg = float(scale) * float(kappa) * reg
        total = task + weighted_reg
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        sums["task"] += float(task.item())
        sums["raw_reg"] += float(reg.item())
        sums["weighted_reg"] += float(weighted_reg.item())
        sums["total"] += float(total.item())
        for name in ("sat", "relative_tail", "capacity"):
            raw_value = float(components[name].item())
            sums[name] += raw_value
            sums[f"weighted_{name}"] += float(scale) * float(kappa) * raw_value
        n_batches += 1
    denom = max(n_batches, 1)
    return {
        "train_optim_task_loss": sums["task"] / denom,
        "train_optim_raw_reg_loss": sums["raw_reg"] / denom,
        "train_optim_weighted_reg_loss": sums["weighted_reg"] / denom,
        "train_optim_total_loss": sums["total"] / denom,
        "train_sat_loss": sums["sat"] / denom,
        "train_relative_tail_loss": sums["relative_tail"] / denom,
        "train_capacity_loss": sums["capacity"] / denom,
        "train_weighted_sat_loss": sums["weighted_sat"] / denom,
        "train_weighted_relative_tail_loss": sums["weighted_relative_tail"] / denom,
        "train_weighted_capacity_loss": sums["weighted_capacity"] / denom,
    }


def _copy_warmup_history(payload: dict[str, object], spec: RunSpec) -> list[dict[str, float | int | str]]:
    rows = payload["history"]
    if not isinstance(rows, list):
        raise TypeError("Warmup history is not a list")
    copied: list[dict[str, float | int | str]] = []
    for source in rows:
        row = dict(source)
        row["condition"] = spec.condition
        row["gradient_calibration_kappa"] = 0.0
        copied.append(row)
    return copied


def _copy_warmup_shift_history(payload: dict[str, object], spec: RunSpec) -> list[dict[str, float | int | str]]:
    rows = payload["shift_history"]
    if not isinstance(rows, list):
        raise TypeError("Warmup shift history is not a list")
    copied = []
    for source in rows:
        row = dict(source)
        row["condition"] = spec.condition
        copied.append(row)
    return copied


def _plot_training(history: pd.DataFrame, destination: Path, best_epoch: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    epoch = history["epoch"]

    axes[0, 0].plot(epoch, history["train_balanced_accuracy"], label="Train BA")
    axes[0, 0].plot(epoch, history["val_balanced_accuracy"], label="Val BA")
    axes[0, 0].set_ylabel("Balanced accuracy")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].legend()

    axes[0, 1].plot(epoch, history["train_task_ce"], label="Train WholeCount CE")
    axes[0, 1].plot(epoch, history["val_task_ce"], label="Val WholeCount CE")
    axes[0, 1].set_ylabel("Task CE")
    axes[0, 1].legend()

    axes[0, 2].plot(epoch, history["train_optim_task_loss"], label="Task")
    axes[0, 2].plot(epoch, history["train_optim_weighted_reg_loss"], label="Weighted regularizer")
    axes[0, 2].plot(epoch, history["train_optim_total_loss"], label="Total")
    for column, label in (
        ("train_weighted_sat_loss", "Sat component"),
        ("train_weighted_relative_tail_loss", "Relative-tail component"),
        ("train_weighted_capacity_loss", "Capacity component"),
    ):
        if column in history and float(history[column].abs().max()) > 0:
            axes[0, 2].plot(epoch, history[column], linewidth=0.9, alpha=0.65, label=label)
    axes[0, 2].set_ylabel("Optimization loss")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(epoch, history["train_long_valid_fr_hz"], label="Train long valid FR")
    axes[1, 0].plot(epoch, history["train_long_tail_fr_hz"], label="Train long tail FR")
    axes[1, 0].plot(epoch, history["val_long_valid_fr_hz"], linestyle="--", label="Val long valid FR")
    axes[1, 0].plot(epoch, history["val_long_tail_fr_hz"], linestyle="--", label="Val long tail FR")
    axes[1, 0].set_ylabel("Hz / neuron")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(epoch, history["train_long_tail_valid_ratio"], label="Train tail/valid")
    axes[1, 1].plot(epoch, history["val_long_tail_valid_ratio"], label="Val tail/valid")
    axes[1, 1].axhline(1.0, linewidth=0.8)
    axes[1, 1].set_ylabel("Tail / valid FR")
    axes[1, 1].legend()

    axes[1, 2].plot(epoch, history["output_weight_norm_s6"], label="s6 outgoing norm")
    axes[1, 2].plot(epoch, history["output_weight_norm_s7"], label="s7 outgoing norm")
    axes[1, 2].plot(epoch, history["output_weight_energy_long_fraction"], label="Long energy fraction")
    axes[1, 2].set_ylabel("Output utilization")
    axes[1, 2].legend(fontsize=8)

    for ax in axes.flat:
        ax.axvline(WARMUP_EPOCHS, linewidth=0.8, linestyle=":")
        ax.axvline(REG_RAMP_END_EPOCH, linewidth=0.8, linestyle=":")
        ax.axvline(best_epoch, linewidth=0.8, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{history['architecture'].iloc[0]} / {history['condition'].iloc[0]} / seed {history['seed'].iloc[0]}"
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_one(data: exp01.exp3.Data, spec: RunSpec, config: Config, force: bool) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    reference = _load_reference(config.results_dir)
    warmup = _load_warmup(config.results_dir, spec.warmup_spec)
    model = _new_model(spec.architecture, spec.seed, data).to(config.device)
    model.load_state_dict(warmup["model_state"])  # type: ignore[arg-type]
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(warmup["optimizer_state"])  # type: ignore[arg-type]
    _restore_rng(warmup["rng_state"])  # type: ignore[arg-type]
    warmup_reference_hz_raw = warmup["train_long_valid_rate_hz_by_shift"]
    if not isinstance(warmup_reference_hz_raw, dict):
        raise TypeError("Warmup firing-rate reference missing")
    warmup_reference_hz = {int(k): torch.as_tensor(v, dtype=torch.float32) for k, v in warmup_reference_hz_raw.items()}

    calibration = calibrate_regularizer(model, data, spec, config, reference, warmup_reference_hz)
    kappa = float(calibration["kappa"])
    _save_json(calibration_path(config.results_dir, spec), {"spec": spec.__dict__, **calibration})

    history = _copy_warmup_history(warmup, spec)
    shift_history = _copy_warmup_shift_history(warmup, spec)
    best_state = {name: value.detach().cpu().clone() for name, value in warmup["best_state_through_warmup"].items()}  # type: ignore[union-attr]
    best_epoch = int(warmup["best_epoch_through_warmup"])
    best_val_ba = float(warmup["best_val_ba_through_warmup"])
    best_val_ce = float(warmup["best_val_ce_through_warmup"])

    for epoch in range(WARMUP_EPOCHS + 1, EPOCHS + 1):
        optim = _train_regularized_epoch(
            model, optimizer, data, spec, epoch, config, reference, warmup_reference_hz, kappa
        )
        train_eval, train_shift = evaluate_split(model, data, "train", spec.warmup_spec, config)
        val_eval, val_shift = evaluate_split(model, data, "val", spec.warmup_spec, config)
        history.append(_history_row(spec, epoch, train_eval, val_eval, optim, kappa))
        for split_name, rows in (("train", train_shift), ("val", val_shift)):
            for row in rows:
                shift_history.append({"architecture": spec.architecture, "condition": spec.condition, "seed": spec.seed, "epoch": epoch, "split": split_name, **row})
        val_ba = val_eval["balanced_accuracy"]
        val_ce = val_eval["task_ce"]
        if val_ba > best_val_ba + 1e-12 or (abs(val_ba - best_val_ba) <= 1e-12 and val_ce < best_val_ce):
            best_val_ba = val_ba
            best_val_ce = val_ce
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    history_frame = pd.DataFrame(history)
    missing_columns = [column for column in HISTORY_COLUMNS if column not in history_frame.columns]
    if missing_columns:
        raise RuntimeError(f"History schema missing columns: {missing_columns}")
    if len(history_frame) != EPOCHS:
        raise RuntimeError(f"Expected {EPOCHS} history rows, got {len(history_frame)}")
    history_file = history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history_frame.to_csv(history_file, index=False)
    shift_file = shift_history_path(config.results_dir, spec)
    shift_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(shift_history).to_csv(shift_file, index=False)

    model.load_state_dict(best_state)
    test_eval, test_shift = evaluate_split(model, data, "test", spec.warmup_spec, config)
    evaluation = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_ba,
        "best_val_task_ce": best_val_ce,
        "test": test_eval,
        "test_shift": test_shift,
        "kappa": kappa,
        "reference": reference,
    }
    _save_json(evaluation_path(config.results_dir, spec), evaluation)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_ba,
            "best_val_task_ce": best_val_ce,
            "model_state": best_state,
            "training_budget_epochs": EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "kappa": kappa,
        },
        destination,
    )
    _plot_training(history_frame, plot_path(config.results_dir, spec), best_epoch)
    print(f"[exp0.2.2] run {spec.key} best_epoch={best_epoch} test_ba={test_eval['balanced_accuracy']:.4f}")
    return destination


def finalize(repo_root: Path) -> None:
    root = results_dir(repo_root)
    specs = run_specs()
    missing: list[str] = []
    for spec in specs:
        for path in (
            checkpoint_path(root, spec),
            evaluation_path(root, spec),
            history_path(root, spec),
            shift_history_path(root, spec),
            calibration_path(root, spec),
            plot_path(root, spec),
        ):
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise RuntimeError(f"Missing Exp0.2.2 artifacts: {missing[:10]}")

    summary_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    histories: list[pd.DataFrame] = []
    shift_histories: list[pd.DataFrame] = []
    for spec in specs:
        evaluation = json.loads(evaluation_path(root, spec).read_text(encoding="utf-8"))
        test = evaluation["test"]
        summary_rows.append(
            {
                "architecture": spec.architecture,
                "condition": spec.condition,
                "seed": spec.seed,
                "best_epoch": evaluation["best_epoch"],
                "best_val_balanced_accuracy": evaluation["best_val_balanced_accuracy"],
                "best_val_task_ce": evaluation["best_val_task_ce"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_task_ce": test["task_ce"],
                "test_long_valid_fr_hz": test["long_valid_fr_hz"],
                "test_long_tail_fr_hz": test["long_tail_fr_hz"],
                "test_long_tail_valid_ratio": test["long_tail_valid_ratio"],
                "output_weight_energy_long_fraction": test["output_weight_energy_long_fraction"],
                "output_weight_norm_s6": test["output_weight_norm_s6"],
                "output_weight_norm_s7": test["output_weight_norm_s7"],
            }
        )
        calibration = json.loads(calibration_path(root, spec).read_text(encoding="utf-8"))
        calibration_rows.append(
            {
                "architecture": spec.architecture,
                "condition": spec.condition,
                "seed": spec.seed,
                "kappa": calibration["kappa"],
                "median_raw_grad_ratio": calibration.get("median_raw_grad_ratio", 0.0),
                "calibration_batches": calibration["calibration_batches"],
            }
        )
        histories.append(pd.read_csv(history_path(root, spec)))
        shift_histories.append(pd.read_csv(shift_history_path(root, spec)))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "summary.csv", index=False)
    numeric = [
        "best_val_balanced_accuracy",
        "best_val_task_ce",
        "test_balanced_accuracy",
        "test_accuracy",
        "test_macro_f1",
        "test_task_ce",
        "test_long_valid_fr_hz",
        "test_long_tail_fr_hz",
        "test_long_tail_valid_ratio",
        "output_weight_energy_long_fraction",
        "output_weight_norm_s6",
        "output_weight_norm_s7",
    ]
    comparison = summary.groupby(["architecture", "condition"])[numeric].agg(["mean", "std"]).reset_index()
    comparison.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in comparison.columns
    ]
    comparison.to_csv(root / "comparison_summary.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(root / "calibration_summary.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(root / "history_long.csv", index=False)
    pd.concat(shift_histories, ignore_index=True).to_csv(root / "shift_history_long.csv", index=False)

    reference = _load_reference(root)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architectures": list(ARCHITECTURES),
        "objective": OBJECTIVE,
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "shared_warmup_epochs": WARMUP_EPOCHS,
        "regularizer_ramp_end_epoch": REG_RAMP_END_EPOCH,
        "target_grad_ratio": TARGET_GRAD_RATIO,
        "calibration_batches": CALIBRATION_BATCHES,
        "capacity_eta": CAPACITY_ETA,
        "long_shifts": list(LONG_SHIFTS),
        "healthy_reference": reference,
        "test_policy": "test evaluated once per run after validation-selected checkpoint is fixed",
        "execution": "6 one-core warmup tasks; 30 one-core main tasks; aggregation-only finalizer",
        "history_rows_per_run": EPOCHS,
    }
    _save_json(root / "manifest.json", manifest)
    print(f"[exp0.2.2] finalized {len(summary)} runs -> {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp0.2.2 capacity-preserving anti-persistent-firing loss")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extract-reference")
    warmup = sub.add_parser("run-warmup")
    warmup.add_argument("--array-task-id", type=int, required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config = Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
    )
    torch.set_num_threads(max(1, int(config.threads)))
    if args.command == "finalize":
        finalize(repo_root)
        return
    data = prepare_data(repo_root)
    if args.command == "extract-reference":
        extract_healthy_reference(data, config, args.force)
    elif args.command == "run-warmup":
        specs = warmup_specs()
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        run_warmup(data, specs[args.array_task_id], config, args.force)
    elif args.command == "run-one":
        specs = run_specs()
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        run_one(data, specs[args.array_task_id], config, args.force)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
