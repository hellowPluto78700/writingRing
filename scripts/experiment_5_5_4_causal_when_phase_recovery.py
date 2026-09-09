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
from snntorch import surrogate
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_5_3_2_2_when_width_sweep as exp5322
from scripts import experiment_5_5_non_snn_history_experts as exp55
from scripts import experiment_5_5_3_teacher_weight_routing_decomposition as exp553


base = exp55.base

EXPERIMENT_ID = "experiment_5_5_4_causal_when_phase_recovery"
PROTOCOL_VERSION = "causal_when_phase_recovery_v1"
SEEDS = exp55.SEEDS
WHAT_WIDTH = exp55.WHAT_WIDTH
N_CLASSES = exp55.N_CLASSES
N_PHASES = 10
STATE_WIDTH = 64
GRU = "gru64_progress"
RSNN = "rsnn64_progress"
ARCHITECTURES = (GRU, RSNN)
HARD = "hard"
LOCAL2 = "local2"
ROUTINGS = (HARD, LOCAL2)
ORACLE = "oracle"
EPOCHS = exp55.EPOCHS
PATIENCE = exp55.PATIENCE
BATCH_SIZE = exp55.BATCH_SIZE
LR = exp55.LR
WEIGHT_DECAY = exp55.WEIGHT_DECAY
GRAD_CLIP = exp55.GRAD_CLIP
PROGRESS_BETA = 0.1
EQUIVALENCE_ATOL = 1e-6
SHORT_SHIFT = exp5322.SHORT_SHIFT
THRESHOLD = exp5322.THRESHOLD
RESET = exp5322.RESET
SURROGATE_SLOPE = exp5322.SURROGATE_SLOPE


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    patience: int = PATIENCE
    batch_size: int = BATCH_SIZE
    threads: int = 1


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.architecture}__seed{self.seed}"


def find_repo_root(start: Path | None = None) -> Path:
    return exp55.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_validation_path(root: Path, seed: int) -> Path:
    return root / "source_validation" / f"seed{seed}.json"


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def trajectory_path(root: Path, spec: RunSpec) -> Path:
    return root / "trajectories" / f"{spec.key}__test.npz"


def routing_disagreement_path(root: Path, spec: RunSpec) -> Path:
    return root / "routing_disagreement" / f"{spec.key}.csv"


def margin_degradation_path(root: Path, spec: RunSpec) -> Path:
    return root / "margin_degradation" / f"{spec.key}.csv"


def phase_sensitivity_path(root: Path, seed: int) -> Path:
    return root / "phase_sensitivity" / f"seed{seed}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_specs() -> list[RunSpec]:
    return [RunSpec(architecture, seed) for architecture in ARCHITECTURES for seed in SEEDS]


def _validate_spec(spec: RunSpec) -> None:
    if spec.architecture not in ARCHITECTURES or spec.seed not in SEEDS:
        raise ValueError(f"Invalid Exp5.5.4 run spec: {spec}")


def _source_config(config: Config) -> exp55.Config:
    return exp55.Config(
        repo_root=config.repo_root,
        results_dir=exp55.results_dir(config.repo_root),
        device=config.device,
        epochs=config.epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _teacher_config(config: Config) -> exp553.Config:
    return exp553.Config(
        repo_root=config.repo_root,
        results_dir=exp553.results_dir(config.repo_root),
        device=config.device,
        epochs=config.epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _labels_lengths(data: base.Data, split: str) -> tuple[np.ndarray, np.ndarray]:
    return exp55._labels_lengths(data, split)


def _load_what(seed: int, data: base.Data, config: Config) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    # Deliberately load the existing frozen fusion cache. Exp5.5.4 never prepares or retrains WHAT.
    return exp55._load_what_arrays(seed, data, _source_config(config))


def _load_relative10_teacher(
    seed: int,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    weight_bank, bias, bin_steps, metadata = exp553.load_teacher(
        exp553.RELATIVE10,
        seed,
        _teacher_config(config),
    )
    if bin_steps != 0:
        raise ValueError("Relative10 teacher unexpectedly has fixed-duration bin_steps")
    if weight_bank.shape != (N_PHASES, N_CLASSES, WHAT_WIDTH):
        raise ValueError(f"Relative10 teacher shape changed: {weight_bank.shape}")
    return weight_bank, bias, metadata


def _loader_seed(seed: int, split: str) -> int:
    # Architecture-independent seed keeps the GRU/RSNN batch ordering paired.
    return base.dseed(seed, EXPERIMENT_ID, "paired_loader", split)


def _loader(
    what: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(what, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def _loaders(
    what: dict[str, np.ndarray],
    data: base.Data,
    seed: int,
    config: Config,
    splits: tuple[str, ...],
    train_shuffle: bool,
) -> dict[str, DataLoader]:
    out: dict[str, DataLoader] = {}
    for split in splits:
        labels, lengths = _labels_lengths(data, split)
        out[split] = _loader(
            what[split],
            labels,
            lengths,
            config.batch_size,
            train_shuffle if split == "train" else False,
            _loader_seed(seed, split),
        )
    return out


def sequence_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    time = torch.arange(steps, device=lengths.device).unsqueeze(0)
    return time < lengths.unsqueeze(1)


def r10_progress_targets(lengths: torch.Tensor, steps: int, dtype: torch.dtype) -> torch.Tensor:
    """R10-aligned target r_t=t/T; T is supervision/masking information only."""
    time = torch.arange(steps, device=lengths.device, dtype=dtype).unsqueeze(0)
    denom = lengths.clamp_min(1).to(dtype).unsqueeze(1)
    return time / denom


def r10_phase_targets(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    positions = torch.arange(steps, device=lengths.device).unsqueeze(0)
    return torch.div(
        positions * N_PHASES,
        lengths.clamp_min(1).unsqueeze(1),
        rounding_mode="floor",
    ).clamp(max=N_PHASES - 1)


class GRUProgressPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(WHAT_WIDTH, STATE_WIDTH, batch_first=True)
        self.progress_head = nn.Linear(STATE_WIDTH, 1)

    def forward_progress(self, what: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        hidden, _ = self.gru(what)
        return torch.sigmoid(self.progress_head(hidden).squeeze(-1))


class RSNNProgressPredictor(nn.Module):
    """Direct RSNN64 copied from the proven Exp5.3.2.x short-memory dynamics."""

    def __init__(self, fs: float) -> None:
        super().__init__()
        self.fs = float(fs)
        self.spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)
        self.input_projection = nn.Linear(WHAT_WIDTH, STATE_WIDTH, bias=False)
        self.recurrent = nn.Linear(STATE_WIDTH, STATE_WIDTH, bias=False)
        self.progress_head = nn.Linear(STATE_WIDTH, 1, bias=True)
        decay = exp5322.parent.decay_from_shift(SHORT_SHIFT)
        self.register_buffer("alpha_vector", torch.full((STATE_WIDTH,), decay, dtype=torch.float32))
        self.register_buffer("beta_vector", torch.full((STATE_WIDTH,), decay, dtype=torch.float32))

    @staticmethod
    def _reset_membrane(membrane: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        if RESET == "subtract":
            return membrane - spike * THRESHOLD
        if RESET == "zero":
            return membrane * (1.0 - spike)
        if RESET == "none":
            return membrane
        raise ValueError(f"Unsupported reset mechanism: {RESET}")

    def membrane_trajectory(self, what: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if torch.any(lengths <= 0) or torch.any(lengths > what.shape[1]):
            raise ValueError("Invalid sequence lengths for RSNN progress trajectory")
        shape = (what.shape[0], STATE_WIDTH)
        synaptic = torch.zeros(shape, dtype=what.dtype, device=what.device)
        membrane = torch.zeros_like(synaptic)
        previous_spike = torch.zeros_like(synaptic)
        membrane_parts: list[torch.Tensor] = []
        for timestep in range(what.shape[1]):
            current = self.input_projection(what[:, timestep]) + self.recurrent(previous_spike)
            next_synaptic = self.alpha_vector.to(current) * synaptic + current
            next_membrane = self.beta_vector.to(current) * membrane + next_synaptic
            next_spike = self.spike_grad(next_membrane - THRESHOLD)
            next_membrane = self._reset_membrane(next_membrane, next_spike)
            valid = (timestep < lengths).unsqueeze(1)
            synaptic = torch.where(valid, next_synaptic, synaptic)
            membrane = torch.where(valid, next_membrane, membrane)
            spike = next_spike * valid.to(next_spike.dtype)
            membrane_parts.append(membrane)
            previous_spike = spike
        return torch.stack(membrane_parts, dim=1)

    def forward_progress(self, what: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        membrane = self.membrane_trajectory(what, lengths)
        return torch.sigmoid(self.progress_head(membrane).squeeze(-1))


def initialize_model(spec: RunSpec, fs: float, device: torch.device) -> nn.Module:
    _validate_spec(spec)
    # Start both architecture families from the same run-level RNG seed.
    base.seed_all(base.dseed(spec.seed, EXPERIMENT_ID, "paired_constructor"))
    if spec.architecture == GRU:
        return GRUProgressPredictor().to(device)
    if spec.architecture == RSNN:
        return RSNNProgressPredictor(fs).to(device)
    raise ValueError(spec.architecture)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "trainable_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "stored_total": int(sum(p.numel() for p in model.parameters())),
    }


def sample_balanced_progress_loss(
    pred: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    target = r10_progress_targets(lengths, pred.shape[1], pred.dtype)
    valid = sequence_mask(lengths, pred.shape[1]).to(pred.dtype)
    raw = F.smooth_l1_loss(pred, target, reduction="none", beta=PROGRESS_BETA)
    per_sample = (raw * valid).sum(dim=1) / lengths.clamp_min(1).to(pred.dtype)
    return per_sample.mean()


def _smooth_l1_numpy(error: np.ndarray) -> np.ndarray:
    delta = np.abs(np.asarray(error, dtype=np.float64))
    return np.where(
        delta < PROGRESS_BETA,
        0.5 * delta * delta / PROGRESS_BETA,
        delta - 0.5 * PROGRESS_BETA,
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)


def _spearman(pred: np.ndarray, target: np.ndarray) -> float:
    x = _rankdata(pred)
    y = _rankdata(target)
    if len(x) <= 1 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    pred64 = np.asarray(pred, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    denom = float(np.sum((target64 - target64.mean()) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((target64 - pred64) ** 2) / denom)


def _target_progress_numpy(lengths: np.ndarray, steps: int) -> np.ndarray:
    time = np.arange(steps, dtype=np.float64)[None, :]
    return time / np.maximum(np.asarray(lengths, dtype=np.float64)[:, None], 1.0)


def _valid_mask_numpy(lengths: np.ndarray, steps: int) -> np.ndarray:
    return np.arange(steps, dtype=np.int64)[None, :] < np.asarray(lengths, dtype=np.int64)[:, None]


def _phase_from_progress(progress: np.ndarray) -> np.ndarray:
    return np.minimum(N_PHASES - 1, np.floor(N_PHASES * np.clip(progress, 0.0, 1.0)).astype(np.int64))


def progress_metrics(
    pred: np.ndarray,
    lengths: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
    pred64 = np.asarray(pred, dtype=np.float64)
    lengths64 = np.asarray(lengths, dtype=np.int64)
    target = _target_progress_numpy(lengths64, pred64.shape[1])
    valid = _valid_mask_numpy(lengths64, pred64.shape[1])
    true_bin = _phase_from_progress(target)
    pred_bin = _phase_from_progress(pred64)

    sample_mae: list[float] = []
    sample_smooth: list[float] = []
    sample_phase_acc: list[float] = []
    sample_bin_mae: list[float] = []
    sample_within1: list[float] = []
    progress_backtrack: list[float] = []
    phase_backtrack: list[float] = []
    for row, length_value in enumerate(lengths64):
        length = int(length_value)
        p = pred64[row, :length]
        t = target[row, :length]
        pb = pred_bin[row, :length]
        tb = true_bin[row, :length]
        sample_mae.append(float(np.abs(p - t).mean()))
        sample_smooth.append(float(_smooth_l1_numpy(p - t).mean()))
        sample_phase_acc.append(float(np.mean(pb == tb)))
        sample_bin_mae.append(float(np.abs(pb - tb).mean()))
        sample_within1.append(float(np.mean(np.abs(pb - tb) <= 1)))
        if length > 1:
            progress_backtrack.append(float(np.mean(np.diff(p) < 0.0)))
            phase_backtrack.append(float(np.mean(np.diff(pb) < 0)))
        else:
            progress_backtrack.append(0.0)
            phase_backtrack.append(0.0)

    flat_pred = pred64[valid]
    flat_target = target[valid]
    metrics: dict[str, float | int] = {
        "sample_balanced_progress_mae": float(np.mean(sample_mae)),
        "sample_balanced_smooth_l1": float(np.mean(sample_smooth)),
        "progress_r2": _r2(flat_pred, flat_target),
        "progress_spearman": _spearman(flat_pred, flat_target),
        "sample_balanced_phase10_accuracy": float(np.mean(sample_phase_acc)),
        "sample_balanced_mean_abs_bin_error": float(np.mean(sample_bin_mae)),
        "sample_balanced_within1_bin_accuracy": float(np.mean(sample_within1)),
        "sample_balanced_progress_backtrack_rate": float(np.mean(progress_backtrack)),
        "sample_balanced_phase_backtrack_rate": float(np.mean(phase_backtrack)),
        "n_samples": int(len(lengths64)),
        "n_valid_timesteps": int(valid.sum()),
    }
    return metrics, target, true_bin, pred_bin


def _predict_progress(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    progress_parts: list[np.ndarray] = []
    length_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    with torch.no_grad():
        for what, labels, lengths in loader:
            progress = model.forward_progress(what.to(device), lengths.to(device))  # type: ignore[attr-defined]
            progress_parts.append(progress.cpu().numpy())
            length_parts.append(lengths.numpy())
            label_parts.append(labels.numpy())
    return np.concatenate(progress_parts), np.concatenate(length_parts), np.concatenate(label_parts)


def evaluate_progress_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    pred, lengths, _ = _predict_progress(model, loader, device)
    metrics, _, _, _ = progress_metrics(pred, lengths)
    return metrics


def _checkpoint_better(candidate: dict[str, float | int], best: dict[str, float | int]) -> bool:
    cand_mae = float(candidate["sample_balanced_progress_mae"])
    best_mae = float(best["sample_balanced_progress_mae"])
    if cand_mae < best_mae - 1e-12:
        return True
    if abs(cand_mae - best_mae) > 1e-12:
        return False
    cand_loss = float(candidate["sample_balanced_smooth_l1"])
    best_loss = float(best["sample_balanced_smooth_l1"])
    if cand_loss < best_loss - 1e-12:
        return True
    if abs(cand_loss - best_loss) > 1e-12:
        return False
    return float(candidate["progress_spearman"]) > float(best["progress_spearman"]) + 1e-12


def _train_progress_model(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool,
) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    source = source_validation_path(config.results_dir, spec.seed)
    if not source.exists():
        raise FileNotFoundError(f"Run requires completed source validation: {source}")

    what, _ = _load_what(spec.seed, data, config)
    train_loader = _loaders(what, data, spec.seed, config, ("train",), True)["train"]
    eval_loaders = _loaders(what, data, spec.seed, config, ("train", "val"), False)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = initialize_model(spec, float(data.fs), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train0 = evaluate_progress_model(model, eval_loaders["train"], device)
    val0 = evaluate_progress_model(model, eval_loaders["val"], device)
    best_metrics = dict(val0)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    rows: list[dict[str, float | int]] = [
        {
            "epoch": 0,
            "train_progress_mae": float(train0["sample_balanced_progress_mae"]),
            "train_smooth_l1": float(train0["sample_balanced_smooth_l1"]),
            "train_spearman": float(train0["progress_spearman"]),
            "val_progress_mae": float(val0["sample_balanced_progress_mae"]),
            "val_smooth_l1": float(val0["sample_balanced_smooth_l1"]),
            "val_spearman": float(val0["progress_spearman"]),
        }
    ]
    stale = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, _yb, lengths in train_loader:
            xb = xb.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_progress(xb, lengths)  # type: ignore[attr-defined]
            loss = sample_balanced_progress_loss(pred, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            n = len(lengths)
            train_loss_sum += float(loss.detach().item()) * n
            train_count += n

        val = evaluate_progress_model(model, eval_loaders["val"], device)
        rows.append(
            {
                "epoch": epoch,
                "train_progress_mae": math.nan,
                "train_smooth_l1": train_loss_sum / max(train_count, 1),
                "train_spearman": math.nan,
                "val_progress_mae": float(val["sample_balanced_progress_mae"]),
                "val_smooth_l1": float(val["sample_balanced_smooth_l1"]),
                "val_spearman": float(val["progress_spearman"]),
            }
        )
        if _checkpoint_better(val, best_metrics):
            best_metrics = dict(val)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "state_width": STATE_WIDTH,
            "progress_target": "r_t=t/T on valid timesteps; T used only for supervision and masking",
            "loss": f"sample-balanced SmoothL1 beta={PROGRESS_BETA}",
            "best_epoch": best_epoch,
            "best_val_metrics": best_metrics,
            "stopped_after_epoch": int(rows[-1]["epoch"]),
            "state_dict": best_state,
            "classification_gradient_used": False,
            "test_used_for_checkpoint_selection": False,
        },
        destination,
    )
    hist = history_path(config.results_dir, spec)
    hist.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(hist, index=False)
    return destination


def _load_checkpoint(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[nn.Module, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("spec") != asdict(spec)
        or payload.get("classification_gradient_used") is not False
        or payload.get("test_used_for_checkpoint_selection") is not False
    ):
        raise ValueError(f"Exp5.5.4 checkpoint identity mismatch: {path}")
    model = initialize_model(spec, float(data.fs), torch.device(config.device))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def routing_probabilities_from_progress(
    progress: np.ndarray,
    lengths: np.ndarray,
    routing: str,
) -> np.ndarray:
    if routing not in ROUTINGS:
        raise ValueError(routing)
    values = np.asarray(progress, dtype=np.float64)
    lengths64 = np.asarray(lengths, dtype=np.int64)
    valid = _valid_mask_numpy(lengths64, values.shape[1])
    q = np.zeros((values.shape[0], values.shape[1], N_PHASES), dtype=np.float64)
    rows, cols = np.nonzero(valid)
    p = np.clip(values[rows, cols], 0.0, 1.0)
    if routing == HARD:
        bins = np.minimum(N_PHASES - 1, np.floor(N_PHASES * p).astype(np.int64))
        q[rows, cols, bins] = 1.0
        return q

    scaled = N_PHASES * p - 0.5
    low_edge = scaled <= 0.0
    high_edge = scaled >= float(N_PHASES - 1)
    middle = ~(low_edge | high_edge)
    q[rows[low_edge], cols[low_edge], 0] = 1.0
    q[rows[high_edge], cols[high_edge], N_PHASES - 1] = 1.0
    if np.any(middle):
        left = np.floor(scaled[middle]).astype(np.int64)
        frac = scaled[middle] - left
        mr = rows[middle]
        mc = cols[middle]
        q[mr, mc, left] = 1.0 - frac
        q[mr, mc, left + 1] = frac
    return q


def teacher_logits_from_progress(
    what: np.ndarray,
    lengths: np.ndarray,
    progress: np.ndarray,
    weight_bank: np.ndarray,
    bias: np.ndarray,
    routing: str,
) -> np.ndarray:
    q = routing_probabilities_from_progress(progress, lengths, routing)
    features = np.einsum("ntk,ntd->nkd", q, np.asarray(what, dtype=np.float64), optimize=True)
    return np.einsum(
        "nkd,kcd->nc",
        features,
        np.asarray(weight_bank, dtype=np.float64),
        optimize=True,
    ) + np.asarray(bias, dtype=np.float64)


def _oracle_progress(lengths: np.ndarray, steps: int) -> np.ndarray:
    return _target_progress_numpy(lengths, steps)


def _phase_sensitivity_rows(
    seed: int,
    what: np.ndarray,
    lengths: np.ndarray,
    weight_bank: np.ndarray,
) -> list[dict[str, float | int]]:
    target = _oracle_progress(lengths, what.shape[1])
    true_phase = _phase_from_progress(target)
    valid = _valid_mask_numpy(lengths, what.shape[1])
    rows: list[dict[str, float | int]] = []
    for phase in range(N_PHASES):
        sample_idx, time_idx = np.nonzero(valid & (true_phase == phase))
        local = np.asarray(what[sample_idx, time_idx], dtype=np.float64)
        for substitute in range(N_PHASES):
            if len(local) == 0:
                mean_l2 = math.nan
            else:
                evidence_true = local @ np.asarray(weight_bank[phase], dtype=np.float64).T
                evidence_sub = local @ np.asarray(weight_bank[substitute], dtype=np.float64).T
                mean_l2 = float(np.linalg.norm(evidence_true - evidence_sub, axis=1).mean())
            rows.append(
                {
                    "seed": seed,
                    "true_phase": phase,
                    "substitute_phase": substitute,
                    "n_timesteps": int(len(local)),
                    "mean_evidence_l2": mean_l2,
                }
            )
    return rows


def validate_source_seed(
    seed: int,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    if seed not in SEEDS:
        raise ValueError(seed)
    destination = source_validation_path(config.results_dir, seed)
    sensitivity = phase_sensitivity_path(config.results_dir, seed)
    if destination.exists() and sensitivity.exists() and not force:
        return destination

    teacher_npz = exp553.teacher_npz_path(exp553.results_dir(config.repo_root), exp553.RELATIVE10, seed)
    teacher_json = exp553.teacher_json_path(exp553.results_dir(config.repo_root), exp553.RELATIVE10, seed)
    frozen_json = exp553.frozen_evaluation_path(exp553.results_dir(config.repo_root), exp553.RELATIVE10, seed)
    missing = [path for path in (teacher_npz, teacher_json, frozen_json) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Exp5.5.4 never retrains the Relative10 teacher; missing: {missing}")

    what, source_meta = _load_what(seed, data, config)
    weight_bank, bias, teacher_meta = _load_relative10_teacher(seed, config)
    labels_by_split = {"train": data.ytr, "val": data.yva, "test": data.yte}
    lengths_by_split = {"train": data.ltr, "val": data.lva, "test": data.lte}
    oracle_metrics: dict[str, dict[str, object]] = {routing: {} for routing in ROUTINGS}
    equivalence: dict[str, object] = {}

    for split in ("train", "val", "test"):
        lengths = lengths_by_split[split]
        oracle = _oracle_progress(lengths, what[split].shape[1])
        hard_logits = teacher_logits_from_progress(
            what[split], lengths, oracle, weight_bank, bias, HARD
        )
        reference_logits = exp553._teacher_logits_numpy(
            what[split],
            lengths,
            weight_bank,
            bias,
            exp553.RELATIVE10,
            "hard",
            float(data.fs),
            0,
        )
        max_abs = float(np.max(np.abs(hard_logits - reference_logits)))
        predictions_identical = bool(np.array_equal(hard_logits.argmax(1), reference_logits.argmax(1)))
        if max_abs > EQUIVALENCE_ATOL or not predictions_identical:
            raise RuntimeError(
                f"Exp5.5.4 fatal oracle-hard gate failed for seed {seed} {split}: {max_abs:.3e}"
            )
        source_delta = float(teacher_meta["equivalence"][f"{split}_max_abs_offline_vs_hard_streaming"])
        source_identical = bool(teacher_meta["equivalence"][f"{split}_predictions_identical"])
        if source_delta > EQUIVALENCE_ATOL or not source_identical:
            raise RuntimeError(f"Exp5.5.3 source teacher equivalence is not valid for seed {seed} {split}")
        equivalence[f"{split}_max_abs_vs_exp553_hard"] = max_abs
        equivalence[f"{split}_predictions_identical_vs_exp553_hard"] = predictions_identical
        for routing in ROUTINGS:
            logits = teacher_logits_from_progress(
                what[split], lengths, oracle, weight_bank, bias, routing
            )
            oracle_metrics[routing][split] = exp55._metrics_from_logits(labels_by_split[split], logits)

    sensitivity.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        _phase_sensitivity_rows(seed, what["test"], data.lte, weight_bank)
    ).to_csv(sensitivity, index=False)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "split_seed": base.SPLIT_SEED,
        "source_experiment_id": exp553.EXPERIMENT_ID,
        "source_protocol_version": exp553.PROTOCOL_VERSION,
        "source_teacher": str(teacher_npz.relative_to(config.repo_root)),
        "source_what_definition": "frozen Local-SNN L2 WHAT128 from the Exp5.4/5.5 fusion cache",
        "source_cache_seed": int(source_meta.get("seed", seed)),
        "teacher_frozen": True,
        "what_frozen": True,
        "equivalence": equivalence,
        "oracle_classification": oracle_metrics,
        "phase_sensitivity": str(sensitivity.relative_to(config.repo_root)),
        "classification_gradient_used": False,
        "test_used_for_model_selection": False,
    }
    _save_json(destination, payload)
    return destination


def _routing_disagreement_rows(
    spec: RunSpec,
    labels: np.ndarray,
    lengths: np.ndarray,
    true_bin: np.ndarray,
    pred_bin: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for index, length_value in enumerate(lengths):
        length = int(length_value)
        true = true_bin[index, :length]
        pred = pred_bin[index, :length]
        abs_error = np.abs(pred - true)
        rows.append(
            {
                "architecture": spec.architecture,
                "seed": spec.seed,
                "sample_index": index,
                "label": int(labels[index]),
                "length": length,
                "routing_disagreement_rate": float(np.mean(pred != true)),
                "mean_abs_bin_error": float(abs_error.mean()),
                "within1_bin_accuracy": float(np.mean(abs_error <= 1)),
            }
        )
    return rows


def _margin_rows(
    spec: RunSpec,
    labels: np.ndarray,
    routing_error: pd.DataFrame,
    predicted_logits: dict[str, np.ndarray],
    oracle_logits: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    disagreement = routing_error.set_index("sample_index")["routing_disagreement_rate"]
    for routing in ROUTINGS:
        pred_margin = exp553._margin(predicted_logits[routing], labels)
        oracle_margin = exp553._margin(oracle_logits[routing], labels)
        for index in range(len(labels)):
            rows.append(
                {
                    "architecture": spec.architecture,
                    "routing": routing,
                    "seed": spec.seed,
                    "sample_index": index,
                    "label": int(labels[index]),
                    "routing_disagreement_rate": float(disagreement.loc[index]),
                    "oracle_true_class_margin": float(oracle_margin[index]),
                    "predicted_true_class_margin": float(pred_margin[index]),
                    "margin_delta_pred_minus_oracle": float(pred_margin[index] - oracle_margin[index]),
                }
            )
    return rows


def run_when_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    _validate_spec(spec)
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    _train_progress_model(spec, data, config, force=force)
    model, checkpoint = _load_checkpoint(spec, data, config)
    what, source_meta = _load_what(spec.seed, data, config)
    eval_loaders = _loaders(what, data, spec.seed, config, ("train", "val", "test"), False)
    device = torch.device(config.device)
    weight_bank, bias, _ = _load_relative10_teacher(spec.seed, config)

    progress_by_split: dict[str, np.ndarray] = {}
    targets_by_split: dict[str, np.ndarray] = {}
    true_bins_by_split: dict[str, np.ndarray] = {}
    pred_bins_by_split: dict[str, np.ndarray] = {}
    phase_metrics: dict[str, dict[str, float | int]] = {}
    classification: dict[str, dict[str, object]] = {routing: {} for routing in ROUTINGS}
    predicted_logits_by_split: dict[str, dict[str, np.ndarray]] = {}
    oracle_logits_by_split: dict[str, dict[str, np.ndarray]] = {}

    for split in ("train", "val", "test"):
        pred, lengths, labels = _predict_progress(model, eval_loaders[split], device)
        metrics, target, true_bin, pred_bin = progress_metrics(pred, lengths)
        progress_by_split[split] = pred
        targets_by_split[split] = target
        true_bins_by_split[split] = true_bin
        pred_bins_by_split[split] = pred_bin
        phase_metrics[split] = metrics
        predicted_logits_by_split[split] = {}
        oracle_logits_by_split[split] = {}
        oracle_progress = _oracle_progress(lengths, pred.shape[1])
        for routing in ROUTINGS:
            predicted_logits = teacher_logits_from_progress(
                what[split], lengths, pred, weight_bank, bias, routing
            )
            oracle_logits = teacher_logits_from_progress(
                what[split], lengths, oracle_progress, weight_bank, bias, routing
            )
            predicted_logits_by_split[split][routing] = predicted_logits
            oracle_logits_by_split[split][routing] = oracle_logits
            classification[routing][split] = exp55._metrics_from_logits(labels, predicted_logits)

    test_confusion = np.zeros((N_PHASES, N_PHASES), dtype=np.int64)
    valid_test = _valid_mask_numpy(data.lte, what["test"].shape[1])
    np.add.at(
        test_confusion,
        (true_bins_by_split["test"][valid_test], pred_bins_by_split["test"][valid_test]),
        1,
    )

    routing_rows = _routing_disagreement_rows(
        spec,
        data.yte,
        data.lte,
        true_bins_by_split["test"],
        pred_bins_by_split["test"],
    )
    routing_df = pd.DataFrame(routing_rows)
    routing_path = routing_disagreement_path(config.results_dir, spec)
    routing_path.parent.mkdir(parents=True, exist_ok=True)
    routing_df.to_csv(routing_path, index=False)

    margin_rows = _margin_rows(
        spec,
        data.yte,
        routing_df,
        predicted_logits_by_split["test"],
        oracle_logits_by_split["test"],
    )
    margin_path = margin_degradation_path(config.results_dir, spec)
    margin_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(margin_rows).to_csv(margin_path, index=False)

    trajectory = trajectory_path(config.results_dir, spec)
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trajectory,
        progress_pred=progress_by_split["test"],
        progress_target=targets_by_split["test"],
        true_phase=true_bins_by_split["test"],
        predicted_phase=pred_bins_by_split["test"],
        lengths=data.lte,
        labels=data.yte,
        hard_logits=predicted_logits_by_split["test"][HARD],
        local2_logits=predicted_logits_by_split["test"][LOCAL2],
        oracle_hard_logits=oracle_logits_by_split["test"][HARD],
        oracle_local2_logits=oracle_logits_by_split["test"][LOCAL2],
    )

    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "architecture": spec.architecture,
        "seed": spec.seed,
        "state_width": STATE_WIDTH,
        "parameter_counts": parameter_counts(model),
        "best_epoch": checkpoint["best_epoch"],
        "stopped_after_epoch": checkpoint["stopped_after_epoch"],
        "best_val_metrics": checkpoint["best_val_metrics"],
        "progress_metrics": phase_metrics,
        "classification": classification,
        "phase_confusion_test": test_confusion.tolist(),
        "source_cache_seed": int(source_meta.get("seed", spec.seed)),
        "progress_contract": "current causal state predicts r_t=t/T; T is never a model input",
        "training_objective": f"sample-balanced SmoothL1(beta={PROGRESS_BETA}) only",
        "checkpoint_selection": "minimum validation sample-balanced progress MAE; tie lower validation SmoothL1; tie higher validation Spearman",
        "teacher_contract": "Exp5.5.3 Relative10 W_k and bias frozen; no classifier retraining",
        "classification_gradient_used": False,
        "test_used_for_checkpoint_selection": False,
        "trajectory": str(trajectory.relative_to(config.repo_root)),
        "routing_disagreement": str(routing_path.relative_to(config.repo_root)),
        "margin_degradation": str(margin_path.relative_to(config.repo_root)),
    }
    _save_json(destination, payload)
    return payload


def _sem(values: np.ndarray) -> float:
    return 0.0 if len(values) <= 1 else float(values.std(ddof=1) / math.sqrt(len(values)))


def _run_row(
    when: str,
    routing: str,
    seed: int,
    metrics: dict[str, object],
    parameter_count: int,
) -> dict[str, object]:
    return {
        "when": when,
        "routing": routing,
        "seed": seed,
        "trainable_parameter_count": parameter_count,
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_loss": metrics["loss"],
    }


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    run_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    routing_frames: list[pd.DataFrame] = []
    margin_frames: list[pd.DataFrame] = []
    sensitivity_frames: list[pd.DataFrame] = []
    oracle_by_seed: dict[tuple[int, str], float] = {}

    for seed in SEEDS:
        source = source_validation_path(root, seed)
        sensitivity = phase_sensitivity_path(root, seed)
        if not source.exists() or not sensitivity.exists():
            raise FileNotFoundError(
                f"Finalizer will not regenerate missing Exp5.5.4 source artifacts for seed {seed}"
            )
        payload = json.loads(source.read_text(encoding="utf-8"))
        if (
            payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("teacher_frozen") is not True
            or payload.get("classification_gradient_used") is not False
        ):
            raise ValueError(f"Invalid source validation artifact: {source}")
        for routing in ROUTINGS:
            metrics = payload["oracle_classification"][routing]["test"]
            run_rows.append(_run_row(ORACLE, routing, seed, metrics, 0))
            oracle_by_seed[(seed, routing)] = float(metrics["balanced_accuracy"])
        phase_rows.append(
            {
                "when": ORACLE,
                "seed": seed,
                "split": "test",
                "sample_balanced_progress_mae": 0.0,
                "sample_balanced_smooth_l1": 0.0,
                "progress_r2": 1.0,
                "progress_spearman": 1.0,
                "sample_balanced_phase10_accuracy": 1.0,
                "sample_balanced_mean_abs_bin_error": 0.0,
                "sample_balanced_within1_bin_accuracy": 1.0,
                "sample_balanced_progress_backtrack_rate": 0.0,
                "sample_balanced_phase_backtrack_rate": 0.0,
            }
        )
        sensitivity_frames.append(pd.read_csv(sensitivity))

    for spec in run_specs():
        path = evaluation_path(root, spec)
        routing_path = routing_disagreement_path(root, spec)
        margin_path = margin_degradation_path(root, spec)
        if not path.exists() or not routing_path.exists() or not margin_path.exists():
            raise FileNotFoundError(f"Finalizer will not retrain or reevaluate missing run: {spec.key}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("spec") != asdict(spec)
            or payload.get("classification_gradient_used") is not False
            or payload.get("test_used_for_checkpoint_selection") is not False
        ):
            raise ValueError(f"Invalid Exp5.5.4 evaluation artifact: {path}")
        parameter_count = int(payload["parameter_counts"]["trainable_total"])
        for routing in ROUTINGS:
            metrics = payload["classification"][routing]["test"]
            run_rows.append(_run_row(spec.architecture, routing, spec.seed, metrics, parameter_count))
            predicted = float(metrics["balanced_accuracy"])
            oracle = oracle_by_seed[(spec.seed, routing)]
            paired_rows.append(
                {
                    "architecture": spec.architecture,
                    "routing": routing,
                    "seed": spec.seed,
                    "oracle_test_balanced_accuracy": oracle,
                    "predicted_test_balanced_accuracy": predicted,
                    "delta_pred_minus_oracle": predicted - oracle,
                    "gap_oracle_minus_pred": oracle - predicted,
                }
            )
        for split, metrics in payload["progress_metrics"].items():
            phase_rows.append({"when": spec.architecture, "seed": spec.seed, "split": split, **metrics})
        confusion = np.asarray(payload["phase_confusion_test"], dtype=np.int64)
        for true_phase in range(N_PHASES):
            denom = int(confusion[true_phase].sum())
            for pred_phase in range(N_PHASES):
                count = int(confusion[true_phase, pred_phase])
                confusion_rows.append(
                    {
                        "architecture": spec.architecture,
                        "seed": spec.seed,
                        "true_phase": true_phase,
                        "predicted_phase": pred_phase,
                        "count": count,
                        "row_fraction": float(count / denom) if denom else math.nan,
                    }
                )
        routing_frames.append(pd.read_csv(routing_path))
        margin_frames.append(pd.read_csv(margin_path))

    runs = pd.DataFrame(run_rows)
    phase = pd.DataFrame(phase_rows)
    paired = pd.DataFrame(paired_rows)
    confusion = pd.DataFrame(confusion_rows)
    routing = pd.concat(routing_frames, ignore_index=True)
    margins = pd.concat(margin_frames, ignore_index=True)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)

    summary_rows: list[dict[str, object]] = []
    for when in (ORACLE, GRU, RSNN):
        test_phase = phase[(phase["when"] == when) & (phase["split"] == "test")]
        row: dict[str, object] = {
            "when": when,
            "n_seeds": int(test_phase["seed"].nunique()),
            "progress_mae_mean": float(test_phase["sample_balanced_progress_mae"].mean()),
            "progress_mae_sd": float(test_phase["sample_balanced_progress_mae"].std(ddof=1)) if len(test_phase) > 1 else 0.0,
            "progress_spearman_mean": float(test_phase["progress_spearman"].mean()),
            "phase10_accuracy_mean": float(test_phase["sample_balanced_phase10_accuracy"].mean()),
            "within1_bin_accuracy_mean": float(test_phase["sample_balanced_within1_bin_accuracy"].mean()),
            "mean_abs_bin_error_mean": float(test_phase["sample_balanced_mean_abs_bin_error"].mean()),
        }
        for routing_name in ROUTINGS:
            values = runs[(runs["when"] == when) & (runs["routing"] == routing_name)][
                "test_balanced_accuracy"
            ].to_numpy(dtype=np.float64)
            row[f"{routing_name}_test_ba_mean"] = float(values.mean())
            row[f"{routing_name}_test_ba_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{routing_name}_test_ba_sem"] = _sem(values)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    outputs = {
        "runs": root / "runs.csv",
        "summary": root / "summary.csv",
        "paired_deltas": root / "paired_deltas.csv",
        "phase_metrics": root / "phase_metrics.csv",
        "phase_confusion": root / "phase_confusion.csv",
        "routing_disagreement": root / "routing_disagreement.csv",
        "margin_degradation": root / "margin_degradation.csv",
        "phase_sensitivity": root / "phase_sensitivity.csv",
        "manifest": root / "manifest.json",
    }
    root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(outputs["runs"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    paired.to_csv(outputs["paired_deltas"], index=False)
    phase.to_csv(outputs["phase_metrics"], index=False)
    confusion.to_csv(outputs["phase_confusion"], index=False)
    routing.to_csv(outputs["routing_disagreement"], index=False)
    margins.to_csv(outputs["margin_degradation"], index=False)
    sensitivity.to_csv(outputs["phase_sensitivity"], index=False)
    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seeds": list(SEEDS),
            "architectures": list(ARCHITECTURES),
            "state_width": STATE_WIDTH,
            "routing_conditions": ["oracle_hard", "oracle_local2", "gru64_hard", "gru64_local2", "rsnn64_hard", "rsnn64_local2"],
            "source_contract": "existing frozen WHAT128 and Exp5.5.3 Relative10 teacher only; source validation never retrains either",
            "progress_target": "r_t=t/T so floor(10*r_t) exactly matches Relative10 floor(10*t/T)",
            "training_objective": f"sample-balanced SmoothL1(beta={PROGRESS_BETA}) only; no classification CE or other regularizer",
            "selection_policy": "minimum validation sample-balanced progress MAE, then lower validation SmoothL1, then higher validation Spearman; test excluded",
            "architecture_comparison": "state-width-matched GRU64 versus direct RSNN64, not parameter-matched",
            "teacher_policy": "W_k and bias frozen for all downstream replay; no classification fitting",
            "multi_cpu_policy": "5 source-validation tasks -> 10 independent WHEN train/evaluate tasks -> one artifact-only finalizer",
            "notebook_policy": "analysis-only; reads finalized artifacts and never trains or regenerates missing artifacts",
        },
    )
    return outputs


def _config_from_args(args: argparse.Namespace) -> Config:
    root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    return Config(
        repo_root=root,
        results_dir=results_dir(root),
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        threads=args.threads,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 5.5.4 causal WHEN phase recovery")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repo-root", default=None)
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=1)
        command.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        command.add_argument("--epochs", type=int, default=EPOCHS)
        command.add_argument("--patience", type=int, default=PATIENCE)
        command.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate-source")
    add_common(validate)
    validate.add_argument("--array-task-id", type=int, required=True)

    run = subparsers.add_parser("run-when")
    add_common(run)
    run.add_argument("--array-task-id", type=int, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--repo-root", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "finalize":
        root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
        for name, path in finalize_experiment(root).items():
            print(f"{name}: {path}")
        return

    config = _config_from_args(args)
    data = base.prepare_data(config.repo_root)
    task = int(args.array_task_id)
    if args.command == "validate-source":
        if task < 0 or task >= len(SEEDS):
            raise IndexError(f"validate-source task {task} outside 0..{len(SEEDS)-1}")
        print(validate_source_seed(SEEDS[task], data, config, force=args.force))
        return
    if args.command == "run-when":
        specs = run_specs()
        if task < 0 or task >= len(specs):
            raise IndexError(f"run-when task {task} outside 0..{len(specs)-1}")
        print(json.dumps(run_when_one(specs[task], data, config, force=args.force), indent=2))
        return
    raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
