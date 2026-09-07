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
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from snntorch import surrogate
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from scripts import experiment_5_2_frozen_local_tauR_sweep as exp52
from scripts import experiment_5_3_synaptic_contextual_evidence as exp53


base = exp53.base

EXPERIMENT_ID = "experiment_5_3_2_when_representation"
PROTOCOL_VERSION = "supervised_causal_when_v1"
SEEDS = (11, 23, 37, 53, 71)
LOCAL_WIDTH = 128
WHEN_WIDTH = 128
N_PHASES = 10
SHORT_SHIFT = 1
THRESHOLD = exp53.THRESHOLD
RESET = exp53.RESET
SURROGATE_SLOPE = exp53.SURROGATE_SLOPE
EPOCHS = exp53.EPOCHS
BATCH_SIZE = exp53.BATCH_SIZE
LR = 1e-3
WEIGHT_DECAY = 0.0
SHUFFLE_REPLICATES = 5
PHASE_C_GRID = (0.01, 0.1, 1.0, 10.0)
PROGRESS_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0)
SPIKE_WINDOWS_SECONDS = (0.25, 0.50)
EPS = 1e-8


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    shifts_mem: tuple[int, ...]
    shifts_syn: tuple[int, ...]
    recurrent: bool = False


CONDITIONS = (
    Condition("mem_single_s4", "mem", (4,), (SHORT_SHIFT,)),
    Condition("mem_single_s5", "mem", (5,), (SHORT_SHIFT,)),
    Condition("mem_single_s6", "mem", (6,), (SHORT_SHIFT,)),
    Condition("mem_multi_s456", "mem", (4, 5, 6), (SHORT_SHIFT,)),
    Condition("syn_single_s4", "syn", (SHORT_SHIFT,), (4,)),
    Condition("syn_single_s5", "syn", (SHORT_SHIFT,), (5,)),
    Condition("syn_single_s6", "syn", (SHORT_SHIFT,), (6,)),
    Condition("syn_multi_s456", "syn", (SHORT_SHIFT,), (4, 5, 6)),
    Condition("rsnn_shortmem", "rsnn", (SHORT_SHIFT,), (SHORT_SHIFT,), True),
)
CONDITION_BY_NAME = {condition.name: condition for condition in CONDITIONS}
CONDITION_NAMES = tuple(condition.name for condition in CONDITIONS)


@dataclass(frozen=True)
class RunSpec:
    condition: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.condition}__seed{self.seed}"

    @property
    def definition(self) -> Condition:
        try:
            return CONDITION_BY_NAME[self.condition]
        except KeyError as error:
            raise ValueError(f"Unknown Exp5.3.2 condition: {self.condition}") from error


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


@dataclass
class WhenTrajectory:
    synaptic: torch.Tensor
    membranes: torch.Tensor
    spikes: torch.Tensor


def find_repo_root(start: Path | None = None) -> Path:
    return exp53.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_results_dir(repo_root: Path) -> Path:
    return exp52.results_dir(repo_root)


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def baseline_path(root: Path, seed: int) -> Path:
    return root / "baselines" / f"seed{seed}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_specs() -> list[RunSpec]:
    return [RunSpec(condition.name, seed) for condition in CONDITIONS for seed in SEEDS]


def decay_from_shift(shift: int) -> float:
    return exp53.decay_from_shift(shift)


def tau_ms_from_shift(shift: int, fs: float) -> float:
    return exp53.tau_ms_from_shift(shift, fs)


def _decay_layout(
    shifts: tuple[int, ...],
    width: int = WHEN_WIDTH,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if not shifts:
        raise ValueError("At least one shift is required")
    assigned = tuple(shifts[index % len(shifts)] for index in range(width))
    counts = tuple(assigned.count(shift) for shift in shifts)
    vector = torch.tensor(
        [decay_from_shift(shift) for shift in assigned], dtype=torch.float32
    )
    return vector, counts, assigned


def _loader(
    X: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
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


def _partitions(
    data: base.Data,
    cache: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "train": (cache["Xtrain"], data.ltr),
        "val": (cache["Xval"], data.lva),
        "test": (cache["Xtest"], data.lte),
    }


def _make_loaders(
    data: base.Data,
    cache: dict[str, np.ndarray],
    spec: RunSpec,
    config: Config,
    train_shuffle: bool,
) -> dict[str, DataLoader]:
    return {
        split: _loader(
            *partition,
            config.batch_size,
            train_shuffle if split == "train" else False,
            base.dseed(spec.seed, "exp5_3_2", split, "loader"),
        )
        for split, partition in _partitions(data, cache).items()
    }


def _phase_progress_targets(
    lengths: torch.Tensor,
    max_steps: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.any(lengths <= 0) or torch.any(lengths > max_steps):
        raise ValueError("Invalid lengths for WHEN targets")
    steps = torch.arange(max_steps, device=lengths.device, dtype=dtype).unsqueeze(0)
    denom = (lengths - 1).clamp_min(1).to(dtype).unsqueeze(1)
    progress = (steps / denom).clamp(0.0, 1.0)
    phase = torch.floor(progress * float(N_PHASES)).to(torch.long).clamp(0, N_PHASES - 1)
    valid = base.mask(lengths, max_steps)
    return phase, progress, valid


def _shuffle_valid_prefix(X: np.ndarray, lengths: np.ndarray, seed: int) -> np.ndarray:
    shuffled = np.array(X, copy=True)
    generator = np.random.default_rng(seed)
    for sample_index, raw_length in enumerate(lengths):
        length = int(raw_length)
        if length > 1:
            shuffled[sample_index, :length] = X[
                sample_index, generator.permutation(length)
            ]
    return shuffled


def _shuffle_cache(
    data: base.Data,
    cache: dict[str, np.ndarray],
    seed: int,
    replicate: int,
) -> dict[str, np.ndarray]:
    lengths_by_split = {"train": data.ltr, "val": data.lva, "test": data.lte}
    return {
        f"X{split}": _shuffle_valid_prefix(
            cache[f"X{split}"],
            lengths,
            base.dseed(seed, "exp5_3_2", "temporal_shuffle", replicate, split),
        )
        for split, lengths in lengths_by_split.items()
    }


class WhenBranchNet(nn.Module):
    """One-layer causal SNN trained only to represent relative WHEN."""

    def __init__(self, condition: str, fs: float) -> None:
        super().__init__()
        if condition not in CONDITION_BY_NAME:
            raise ValueError(f"Unknown Exp5.3.2 condition: {condition}")
        self.condition = condition
        self.definition = CONDITION_BY_NAME[condition]
        self.family = self.definition.family
        self.fs = float(fs)
        self.spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

        self.input_projection = nn.Linear(LOCAL_WIDTH, WHEN_WIDTH, bias=False)
        self.recurrent = (
            nn.Linear(WHEN_WIDTH, WHEN_WIDTH, bias=False)
            if self.definition.recurrent
            else None
        )
        self.phase_head = nn.Linear(WHEN_WIDTH, N_PHASES, bias=True)
        self.progress_head = nn.Linear(WHEN_WIDTH, 1, bias=True)

        beta, beta_counts, beta_assigned = _decay_layout(self.definition.shifts_mem)
        alpha, alpha_counts, alpha_assigned = _decay_layout(self.definition.shifts_syn)
        self.register_buffer("beta_vector", beta)
        self.register_buffer("alpha_vector", alpha)
        self.beta_counts = beta_counts
        self.alpha_counts = alpha_counts
        self.beta_assigned = beta_assigned
        self.alpha_assigned = alpha_assigned

    @staticmethod
    def _reset_membrane(membrane: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        if RESET == "subtract":
            return membrane - spike * THRESHOLD
        if RESET == "zero":
            return membrane * (1.0 - spike)
        if RESET == "none":
            return membrane
        raise ValueError(f"Unsupported reset mechanism: {RESET}")

    def _initial_state(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (x.shape[0], WHEN_WIDTH)
        synaptic = torch.zeros(shape, dtype=x.dtype, device=x.device)
        membrane = torch.zeros_like(synaptic)
        spike = torch.zeros_like(synaptic)
        return synaptic, membrane, spike

    def _step(
        self,
        local_t: torch.Tensor,
        synaptic: torch.Tensor,
        membrane: torch.Tensor,
        previous_spike: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = self.input_projection(local_t)
        if self.recurrent is not None:
            current = current + self.recurrent(previous_spike)
        synaptic = self.alpha_vector.to(current) * synaptic + current
        membrane = self.beta_vector.to(current) * membrane + synaptic
        spike = self.spike_grad(membrane - THRESHOLD)
        membrane = self._reset_membrane(membrane, spike)
        return synaptic, membrane, spike

    def forward_trajectory(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        reset_state_each_step: bool = False,
    ) -> WhenTrajectory:
        if torch.any(lengths <= 0) or torch.any(lengths > x.shape[1]):
            raise ValueError("Invalid sequence lengths for WHEN trajectory")
        synaptic, membrane, previous_spike = self._initial_state(x)
        synaptic_parts: list[torch.Tensor] = []
        membrane_parts: list[torch.Tensor] = []
        spike_parts: list[torch.Tensor] = []
        for timestep in range(x.shape[1]):
            if reset_state_each_step:
                synaptic.zero_()
                membrane.zero_()
                previous_spike.zero_()
            next_synaptic, next_membrane, next_spike = self._step(
                x[:, timestep], synaptic, membrane, previous_spike
            )
            valid = (timestep < lengths).unsqueeze(1)
            synaptic = torch.where(valid, next_synaptic, synaptic)
            membrane = torch.where(valid, next_membrane, membrane)
            spike = next_spike * valid.to(next_spike.dtype)
            synaptic_parts.append(synaptic)
            membrane_parts.append(membrane)
            spike_parts.append(spike)
            previous_spike = spike
        return WhenTrajectory(
            synaptic=torch.stack(synaptic_parts, dim=1),
            membranes=torch.stack(membrane_parts, dim=1),
            spikes=torch.stack(spike_parts, dim=1),
        )

    def heads_from_membrane(
        self, membranes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase_logits = self.phase_head(membranes)
        progress = torch.sigmoid(self.progress_head(membranes).squeeze(-1))
        return phase_logits, progress

    def loss_components(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        reset_state_each_step: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, WhenTrajectory]:
        trajectory = self.forward_trajectory(x, lengths, reset_state_each_step)
        phase_logits, progress_pred = self.heads_from_membrane(trajectory.membranes)
        phase_target, progress_target, valid = _phase_progress_targets(
            lengths, x.shape[1], x.dtype
        )
        batch_size, steps = phase_target.shape
        phase_step = F.cross_entropy(
            phase_logits.reshape(batch_size * steps, N_PHASES),
            phase_target.reshape(batch_size * steps),
            reduction="none",
        ).reshape(batch_size, steps)
        progress_step = F.smooth_l1_loss(
            progress_pred, progress_target, reduction="none"
        )
        valid_float = valid.to(x.dtype)
        denom = lengths.to(x.dtype)
        phase_loss = ((phase_step * valid_float).sum(dim=1) / denom).mean()
        progress_loss = ((progress_step * valid_float).sum(dim=1) / denom).mean()
        total_loss = phase_loss + progress_loss
        return (
            total_loss,
            phase_loss,
            progress_loss,
            phase_logits,
            progress_pred,
            trajectory,
        )

    def group_indices(self, state_kind: str) -> dict[str, list[int]]:
        if state_kind == "membrane":
            assigned = self.beta_assigned
            shifts = self.definition.shifts_mem
        elif state_kind == "synaptic":
            assigned = self.alpha_assigned
            shifts = self.definition.shifts_syn
        else:
            raise ValueError(f"Unknown state kind: {state_kind}")
        if len(shifts) <= 1:
            return {}
        return {
            f"{state_kind}_s{shift}": [
                index for index, assigned_shift in enumerate(assigned) if assigned_shift == shift
            ]
            for shift in shifts
        }


def _initialize_model(
    spec: RunSpec,
    fs: float,
    device: torch.device,
) -> WhenBranchNet:
    base.seed_all(base.dseed(spec.seed, "exp5_3_2", "constructor"))
    model = WhenBranchNet(spec.condition, fs).to(device)
    base.seed_all(base.dseed(spec.seed, "exp5_3_2", "input_projection"))
    model.input_projection.reset_parameters()
    base.seed_all(base.dseed(spec.seed, "exp5_3_2", "phase_head"))
    model.phase_head.reset_parameters()
    base.seed_all(base.dseed(spec.seed, "exp5_3_2", "progress_head"))
    model.progress_head.reset_parameters()
    if model.recurrent is not None:
        base.seed_all(base.dseed(spec.seed, "exp5_3_2", "recurrent"))
        model.recurrent.reset_parameters()
    return model


def _source_config(config: Config) -> exp52.Config:
    return exp52.Config(
        repo_root=config.repo_root,
        results_dir=source_results_dir(config.repo_root),
        device=config.device,
        epochs=exp52.EPOCHS,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def load_local_cache(
    seed: int,
    data: base.Data,
    config: Config,
) -> dict[str, np.ndarray]:
    return exp52.load_local_cache(seed, data, _source_config(config))


def _baseline_arrays(
    X: np.ndarray,
    lengths: np.ndarray,
    fs: float,
) -> dict[str, np.ndarray]:
    what_parts: list[np.ndarray] = []
    elapsed_parts: list[np.ndarray] = []
    phase_parts: list[np.ndarray] = []
    progress_parts: list[np.ndarray] = []
    for sample_index, raw_length in enumerate(lengths):
        length = int(raw_length)
        if length <= 0:
            raise ValueError("WHEN baseline requires positive valid length")
        progress = (
            np.zeros(1, dtype=np.float32)
            if length == 1
            else np.arange(length, dtype=np.float32) / float(length - 1)
        )
        phase = np.minimum(
            N_PHASES - 1,
            np.floor(progress * float(N_PHASES)).astype(np.int64),
        )
        what_parts.append(np.asarray(X[sample_index, :length], dtype=np.float32))
        elapsed_parts.append((np.arange(length, dtype=np.float32) / float(fs))[:, None])
        phase_parts.append(phase)
        progress_parts.append(progress)
    return {
        "what": np.concatenate(what_parts, axis=0),
        "elapsed": np.concatenate(elapsed_parts, axis=0),
        "phase": np.concatenate(phase_parts, axis=0),
        "progress": np.concatenate(progress_parts, axis=0),
    }


def _fit_phase_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
) -> dict[str, float | int]:
    scaler = StandardScaler().fit(train_x)
    train_z = scaler.transform(train_x)
    val_z = scaler.transform(val_x)
    test_z = scaler.transform(test_x)
    best: tuple[float, float, LogisticRegression] | None = None
    for C in PHASE_C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=3000,
            solver="lbfgs",
            random_state=seed,
        ).fit(train_z, train_y)
        val_pred = classifier.predict(val_z)
        val_ba = float(balanced_accuracy_score(val_y, val_pred))
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError("No phase probe candidate selected")
    val_ba, C, classifier = best
    test_pred = classifier.predict(test_z)
    return {
        "phase_probe_C": C,
        "phase_probe_val_ba": val_ba,
        "phase_probe_test_ba": float(balanced_accuracy_score(test_y, test_pred)),
        "phase_probe_test_accuracy": float(accuracy_score(test_y, test_pred)),
        "phase_probe_test_macro_f1": float(
            f1_score(test_y, test_pred, average="macro", zero_division=0)
        ),
    }


def _fit_progress_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float]:
    scaler = StandardScaler().fit(train_x)
    train_z = scaler.transform(train_x)
    val_z = scaler.transform(val_x)
    test_z = scaler.transform(test_x)
    best: tuple[float, float, Ridge] | None = None
    for alpha in PROGRESS_ALPHA_GRID:
        regressor = Ridge(alpha=alpha).fit(train_z, train_y)
        val_pred = np.clip(regressor.predict(val_z), 0.0, 1.0)
        val_mae = float(mean_absolute_error(val_y, val_pred))
        if best is None or val_mae < best[0] - 1e-12:
            best = (val_mae, float(alpha), regressor)
    if best is None:
        raise RuntimeError("No progress probe candidate selected")
    val_mae, alpha, regressor = best
    test_pred = np.clip(regressor.predict(test_z), 0.0, 1.0)
    return {
        "progress_probe_alpha": alpha,
        "progress_probe_val_mae": val_mae,
        "progress_probe_test_mae": float(mean_absolute_error(test_y, test_pred)),
        "progress_probe_test_r2": float(r2_score(test_y, test_pred)),
    }


def _fit_feature_probe(
    features: dict[str, dict[str, np.ndarray]],
    feature_name: str,
    seed: int,
    namespace: str,
) -> dict[str, object]:
    phase = _fit_phase_probe(
        features["train"][feature_name],
        features["train"]["phase"],
        features["val"][feature_name],
        features["val"]["phase"],
        features["test"][feature_name],
        features["test"]["phase"],
        base.dseed(seed, "exp5_3_2", namespace, feature_name, "phase_probe"),
    )
    progress = _fit_progress_probe(
        features["train"][feature_name],
        features["train"]["progress"],
        features["val"][feature_name],
        features["val"]["progress"],
        features["test"][feature_name],
        features["test"]["progress"],
    )
    return {
        "feature_type": feature_name,
        "feature_dim": int(features["train"][feature_name].shape[1]),
        **phase,
        **progress,
    }


def prepare_local_seed(
    seed: int,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Unknown Exp5.3.2 seed: {seed}")
    source_config = _source_config(config)
    exp52.prepare_local_seed(seed, data, source_config, force=force)
    cache = exp52.load_local_cache(seed, data, source_config)
    for split, (X, lengths) in _partitions(data, cache).items():
        if X.shape[0] != len(lengths):
            raise ValueError(f"Frozen local cache sample mismatch for seed {seed} split {split}")

    destination = baseline_path(config.results_dir, seed)
    if force or not destination.exists():
        arrays = {
            split: _baseline_arrays(X, lengths, data.fs)
            for split, (X, lengths) in _partitions(data, cache).items()
        }
        baselines = []
        for feature_name in ("what", "elapsed"):
            baselines.append(
                {
                    "baseline": "what_only" if feature_name == "what" else "elapsed_time_only",
                    **_fit_feature_probe(arrays, feature_name, seed, "baseline"),
                }
            )
        _save_json(
            destination,
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "sampling_rate_hz": float(data.fs),
                "elapsed_time_input": "t / fs only; final duration T is never an input",
                "what_input": "frozen Local-SNN L2 spike vector z_t only",
                "baselines": baselines,
            },
        )
    return exp52.local_cache_path(source_config.results_dir, seed)


def _trailing_count(spikes: torch.Tensor, window_steps: int) -> torch.Tensor:
    if window_steps <= 0:
        raise ValueError("window_steps must be positive")
    cumulative = torch.cumsum(spikes, dim=1)
    out = cumulative.clone()
    if spikes.shape[1] > window_steps:
        out[:, window_steps:] = cumulative[:, window_steps:] - cumulative[:, :-window_steps]
    return out


def _flatten_valid(sequence: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    return sequence[valid].detach().cpu().numpy()


def collect_features(
    model: WhenBranchNet,
    data: base.Data,
    cache: dict[str, np.ndarray],
    spec: RunSpec,
    config: Config,
    reset_state_each_step: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)
    device = torch.device(config.device)
    window250 = max(1, int(round(float(data.fs) * SPIKE_WINDOWS_SECONDS[0])))
    window500 = max(1, int(round(float(data.fs) * SPIKE_WINDOWS_SECONDS[1])))
    output: dict[str, dict[str, np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for split, loader in loaders.items():
            parts: dict[str, list[np.ndarray]] = {
                "phase": [],
                "progress": [],
                "membrane": [],
                "synaptic": [],
                "spike": [],
                "spike250": [],
                "spike500": [],
            }
            group_parts: dict[str, list[np.ndarray]] = {}
            for local, lengths in loader:
                local = local.to(device)
                lengths = lengths.to(device)
                trajectory = model.forward_trajectory(
                    local, lengths, reset_state_each_step=reset_state_each_step
                )
                phase, progress, valid = _phase_progress_targets(
                    lengths, local.shape[1], local.dtype
                )
                spike250 = _trailing_count(trajectory.spikes, window250)
                spike500 = _trailing_count(trajectory.spikes, window500)
                parts["phase"].append(phase[valid].cpu().numpy())
                parts["progress"].append(progress[valid].cpu().numpy())
                parts["membrane"].append(_flatten_valid(trajectory.membranes, valid))
                parts["synaptic"].append(_flatten_valid(trajectory.synaptic, valid))
                parts["spike"].append(_flatten_valid(trajectory.spikes, valid))
                parts["spike250"].append(_flatten_valid(spike250, valid))
                parts["spike500"].append(_flatten_valid(spike500, valid))
                for state_kind, sequence in (
                    ("membrane", trajectory.membranes),
                    ("synaptic", trajectory.synaptic),
                ):
                    for group_name, indices in model.group_indices(state_kind).items():
                        group_parts.setdefault(group_name, []).append(
                            _flatten_valid(sequence[:, :, indices], valid)
                        )
            split_output = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
            split_output.update(
                {name: np.concatenate(values, axis=0) for name, values in group_parts.items()}
            )
            output[split] = split_output
    return output


def _native_metrics_from_batch(
    phase_logits: torch.Tensor,
    progress_pred: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    phase_target, progress_target, valid = _phase_progress_targets(
        lengths, phase_logits.shape[1], progress_pred.dtype
    )
    phase_true = phase_target[valid].detach().cpu().numpy()
    phase_pred = phase_logits.argmax(dim=-1)[valid].detach().cpu().numpy()
    progress_true = progress_target[valid].detach().cpu().numpy()
    progress_hat = progress_pred[valid].detach().cpu().numpy()
    sample_abs = (
        ((progress_pred - progress_target).abs() * valid.to(progress_pred.dtype)).sum(dim=1)
        / lengths.to(progress_pred.dtype)
    )
    return phase_true, phase_pred, progress_true, progress_hat, float(sample_abs.sum().item())


def evaluate_native(
    model: WhenBranchNet,
    loader: DataLoader,
    device: torch.device,
    reset_state_each_step: bool = False,
) -> dict[str, float]:
    model.eval()
    phase_true_parts: list[np.ndarray] = []
    phase_pred_parts: list[np.ndarray] = []
    progress_true_parts: list[np.ndarray] = []
    progress_pred_parts: list[np.ndarray] = []
    total_loss_sum = 0.0
    phase_loss_sum = 0.0
    progress_loss_sum = 0.0
    sample_progress_abs_sum = 0.0
    n_samples = 0
    with torch.no_grad():
        for local, lengths in loader:
            local = local.to(device)
            lengths = lengths.to(device)
            total, phase_loss, progress_loss, phase_logits, progress_pred, _ = model.loss_components(
                local, lengths, reset_state_each_step=reset_state_each_step
            )
            phase_true, phase_pred, progress_true, progress_hat, sample_abs_sum = _native_metrics_from_batch(
                phase_logits, progress_pred, lengths
            )
            n = len(local)
            n_samples += n
            total_loss_sum += float(total.item()) * n
            phase_loss_sum += float(phase_loss.item()) * n
            progress_loss_sum += float(progress_loss.item()) * n
            sample_progress_abs_sum += sample_abs_sum
            phase_true_parts.append(phase_true)
            phase_pred_parts.append(phase_pred)
            progress_true_parts.append(progress_true)
            progress_pred_parts.append(progress_hat)
    phase_true = np.concatenate(phase_true_parts)
    phase_pred = np.concatenate(phase_pred_parts)
    progress_true = np.concatenate(progress_true_parts)
    progress_pred = np.concatenate(progress_pred_parts)
    return {
        "loss": total_loss_sum / max(n_samples, 1),
        "phase_ce": phase_loss_sum / max(n_samples, 1),
        "progress_loss": progress_loss_sum / max(n_samples, 1),
        "phase_balanced_accuracy": float(balanced_accuracy_score(phase_true, phase_pred)),
        "phase_accuracy": float(accuracy_score(phase_true, phase_pred)),
        "phase_macro_f1": float(f1_score(phase_true, phase_pred, average="macro", zero_division=0)),
        "progress_mae": float(mean_absolute_error(progress_true, progress_pred)),
        "progress_sample_balanced_mae": sample_progress_abs_sum / max(n_samples, 1),
        "progress_r2": float(r2_score(progress_true, progress_pred)),
    }


def parameter_counts(model: WhenBranchNet) -> dict[str, int]:
    return {
        "input_projection": int(sum(p.numel() for p in model.input_projection.parameters())),
        "recurrent": int(sum(p.numel() for p in model.recurrent.parameters())) if model.recurrent is not None else 0,
        "phase_head": int(sum(p.numel() for p in model.phase_head.parameters())),
        "progress_head": int(sum(p.numel() for p in model.progress_head.parameters())),
        "trainable_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def provenance(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    model: WhenBranchNet,
) -> dict[str, object]:
    definition = spec.definition
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": spec.seed,
        "split_seed": base.SPLIT_SEED,
        "condition": definition.name,
        "family": definition.family,
        "shifts_mem": list(definition.shifts_mem),
        "tau_mem_ms": [tau_ms_from_shift(shift, data.fs) for shift in definition.shifts_mem],
        "mem_group_counts": list(model.beta_counts),
        "shifts_syn": list(definition.shifts_syn),
        "tau_syn_ms": [tau_ms_from_shift(shift, data.fs) for shift in definition.shifts_syn],
        "syn_group_counts": list(model.alpha_counts),
        "recurrent": definition.recurrent,
        "local_width": LOCAL_WIDTH,
        "when_width": WHEN_WIDTH,
        "sampling_rate_hz": float(data.fs),
        "local_source_experiment": exp52.EXPERIMENT_ID,
        "local_source_protocol": exp52.PROTOCOL_VERSION,
        "local_source_frozen": True,
        "objective": "supervised_when",
        "objective_formula": "L_phase(U_t) + L_progress(U_t)",
        "phase_target": "q_t=min(9,floor(10*t/(T-1))); T used only to construct supervision",
        "progress_target": "p_t=t/(T-1); T used only to construct supervision",
        "loss_reduction": "mean over valid timesteps within each sample, then mean over samples",
        "checkpoint_selection": "max validation native U phase BA; tie lower progress MAE; tie lower total loss",
        "threshold": THRESHOLD,
        "reset": RESET,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "shuffle_replicates": SHUFFLE_REPLICATES,
        "spike_windows_seconds": list(SPIKE_WINDOWS_SECONDS),
        "parameter_counts": parameter_counts(model),
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
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    cache = load_local_cache(spec.seed, data, config)
    train_loaders = _make_loaders(data, cache, spec, config, train_shuffle=True)
    eval_loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _initialize_model(spec, data.fs, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_ba = -np.inf
    best_val_progress_mae = np.inf
    best_val_loss = np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_total_sum = 0.0
        train_phase_sum = 0.0
        train_progress_sum = 0.0
        train_n = 0
        train_phase_true: list[np.ndarray] = []
        train_phase_pred: list[np.ndarray] = []
        train_progress_abs_sum = 0.0
        for local, lengths in train_loaders["train"]:
            local = local.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            total, phase_loss, progress_loss, phase_logits, progress_pred, _ = model.loss_components(
                local, lengths
            )
            total.backward()
            optimizer.step()
            phase_true, phase_pred, _, _, sample_abs_sum = _native_metrics_from_batch(
                phase_logits.detach(), progress_pred.detach(), lengths
            )
            n = len(local)
            train_n += n
            train_total_sum += float(total.item()) * n
            train_phase_sum += float(phase_loss.item()) * n
            train_progress_sum += float(progress_loss.item()) * n
            train_progress_abs_sum += sample_abs_sum
            train_phase_true.append(phase_true)
            train_phase_pred.append(phase_pred)

        train_ba = float(
            balanced_accuracy_score(
                np.concatenate(train_phase_true), np.concatenate(train_phase_pred)
            )
        )
        val_metrics = evaluate_native(model, eval_loaders["val"], device)
        val_ba = float(val_metrics["phase_balanced_accuracy"])
        val_progress_mae = float(val_metrics["progress_sample_balanced_mae"])
        val_loss = float(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_total_sum / max(train_n, 1),
                "train_phase_ce": train_phase_sum / max(train_n, 1),
                "train_progress_loss": train_progress_sum / max(train_n, 1),
                "train_phase_balanced_accuracy": train_ba,
                "train_progress_sample_balanced_mae": train_progress_abs_sum / max(train_n, 1),
                "val_loss": val_loss,
                "val_phase_ce": val_metrics["phase_ce"],
                "val_progress_loss": val_metrics["progress_loss"],
                "val_phase_balanced_accuracy": val_ba,
                "val_progress_sample_balanced_mae": val_progress_mae,
            }
        )
        improved = (
            val_ba > best_val_ba + 1e-12
            or (
                abs(val_ba - best_val_ba) <= 1e-12
                and val_progress_mae < best_val_progress_mae - 1e-12
            )
            or (
                abs(val_ba - best_val_ba) <= 1e-12
                and abs(val_progress_mae - best_val_progress_mae) <= 1e-12
                and val_loss < best_val_loss - 1e-12
            )
        )
        if improved:
            best_val_ba = val_ba
            best_val_progress_mae = val_progress_mae
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    model.load_state_dict(best_state)
    native = {
        split: evaluate_native(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "provenance": provenance(spec, data, config, model),
            "result": {
                "best_epoch": best_epoch,
                "best_val_phase_balanced_accuracy": best_val_ba,
                "best_val_progress_mae": best_val_progress_mae,
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
) -> tuple[WhenBranchNet, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.3.2 checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong Exp5.3.2 checkpoint identity: {path}")
    if payload.get("spec") != asdict(spec):
        raise ValueError(f"Checkpoint spec mismatch: {path}")
    model = WhenBranchNet(spec.condition, data.fs).to(torch.device(config.device))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _all_native_splits(
    model: WhenBranchNet,
    loaders: dict[str, DataLoader],
    device: torch.device,
    reset_state_each_step: bool,
) -> dict[str, dict[str, float]]:
    return {
        split: evaluate_native(model, loader, device, reset_state_each_step)
        for split, loader in loaders.items()
    }


def _main_probe_rows(
    features: dict[str, dict[str, np.ndarray]],
    spec: RunSpec,
    namespace: str,
) -> list[dict[str, object]]:
    return [
        _fit_feature_probe(features, feature_name, spec.seed, namespace)
        for feature_name in ("membrane", "synaptic", "spike", "spike250", "spike500")
    ]


def _group_probe_rows(
    model: WhenBranchNet,
    features: dict[str, dict[str, np.ndarray]],
    spec: RunSpec,
) -> list[dict[str, object]]:
    names = list(model.group_indices("membrane")) + list(model.group_indices("synaptic"))
    return [
        {
            "group": name,
            **_fit_feature_probe(features, name, spec.seed, "group"),
        }
        for name in names
    ]


def evaluate_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    cache = load_local_cache(spec.seed, data, config)
    model, checkpoint = load_model(spec, data, config)
    device = torch.device(config.device)
    loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)

    native = _all_native_splits(model, loaders, device, False)
    ordered_features = collect_features(model, data, cache, spec, config, False)
    probes = _main_probe_rows(ordered_features, spec, "ordered")
    group_probes = _group_probe_rows(model, ordered_features, spec)

    reset_native = _all_native_splits(model, loaders, device, True)
    reset_features = collect_features(model, data, cache, spec, config, True)
    reset_membrane_probe = _fit_feature_probe(
        reset_features, "membrane", spec.seed, "state_reset"
    )
    ablations: list[dict[str, object]] = [
        {
            "ablation": "state_reset",
            "replicate": 0,
            "native": reset_native,
            "membrane_probe": reset_membrane_probe,
        }
    ]

    for replicate in range(SHUFFLE_REPLICATES):
        shuffled_cache = _shuffle_cache(data, cache, spec.seed, replicate)
        shuffled_loaders = _make_loaders(
            data, shuffled_cache, spec, config, train_shuffle=False
        )
        shuffled_native = _all_native_splits(model, shuffled_loaders, device, False)
        shuffled_features = collect_features(
            model, data, shuffled_cache, spec, config, False
        )
        shuffled_membrane_probe = _fit_feature_probe(
            shuffled_features,
            "membrane",
            spec.seed,
            f"temporal_shuffle_{replicate}",
        )
        ablations.append(
            {
                "ablation": "temporal_shuffle",
                "replicate": replicate,
                "native": shuffled_native,
                "membrane_probe": shuffled_membrane_probe,
            }
        )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "provenance": checkpoint["provenance"],
        "best_epoch": checkpoint["result"]["best_epoch"],
        "native": native,
        "probes": probes,
        "group_probes": group_probes,
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


def _condition_row(condition: Condition, fs: float) -> dict[str, object]:
    return {
        "condition": condition.name,
        "family": condition.family,
        "shifts_mem": json.dumps(list(condition.shifts_mem)),
        "tau_mem_ms": json.dumps([tau_ms_from_shift(shift, fs) for shift in condition.shifts_mem]),
        "shifts_syn": json.dumps(list(condition.shifts_syn)),
        "tau_syn_ms": json.dumps([tau_ms_from_shift(shift, fs) for shift in condition.shifts_syn]),
        "recurrent": condition.recurrent,
    }


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    local_rows: list[dict[str, object]] = []
    history_parts: list[pd.DataFrame] = []

    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp5.3.2 evaluation: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != PROTOCOL_VERSION or payload.get("spec") != asdict(spec):
            raise ValueError(f"Evaluation identity mismatch: {path}")
        fs = float(payload["provenance"]["sampling_rate_hz"])
        common = {**_condition_row(spec.definition, fs), "seed": spec.seed}
        row: dict[str, object] = {
            **common,
            "best_epoch": payload["best_epoch"],
            "trainable_parameters": payload["provenance"]["parameter_counts"]["trainable_total"],
        }
        for split in ("train", "val", "test"):
            for metric, value in payload["native"][split].items():
                row[f"native_{split}_{metric}"] = value
        rows.append(row)

        for probe in payload["probes"]:
            probe_rows.append({**common, **probe})
        for probe in payload["group_probes"]:
            group_rows.append({**common, **probe})

        ordered_membrane = next(
            probe for probe in payload["probes"] if probe["feature_type"] == "membrane"
        )
        ablation_rows.append(
            {
                **common,
                "ablation": "ordered",
                "replicate": 0,
                "native_val_phase_ba": payload["native"]["val"]["phase_balanced_accuracy"],
                "native_test_phase_ba": payload["native"]["test"]["phase_balanced_accuracy"],
                "native_test_progress_mae": payload["native"]["test"]["progress_sample_balanced_mae"],
                "probe_val_phase_ba": ordered_membrane["phase_probe_val_ba"],
                "probe_test_phase_ba": ordered_membrane["phase_probe_test_ba"],
                "probe_test_progress_mae": ordered_membrane["progress_probe_test_mae"],
            }
        )
        for ablation in payload["ablations"]:
            probe = ablation["membrane_probe"]
            ablation_rows.append(
                {
                    **common,
                    "ablation": ablation["ablation"],
                    "replicate": ablation["replicate"],
                    "native_val_phase_ba": ablation["native"]["val"]["phase_balanced_accuracy"],
                    "native_test_phase_ba": ablation["native"]["test"]["phase_balanced_accuracy"],
                    "native_test_progress_mae": ablation["native"]["test"]["progress_sample_balanced_mae"],
                    "probe_val_phase_ba": probe["phase_probe_val_ba"],
                    "probe_test_phase_ba": probe["phase_probe_test_ba"],
                    "probe_test_progress_mae": probe["progress_probe_test_mae"],
                }
            )

        hpath = history_path(root, spec)
        if not hpath.exists():
            raise FileNotFoundError(f"Missing Exp5.3.2 history: {hpath}")
        history = pd.read_csv(hpath)
        history.insert(0, "seed", spec.seed)
        history.insert(0, "family", spec.definition.family)
        history.insert(0, "condition", spec.condition)
        history_parts.append(history)

    for seed in SEEDS:
        bpath = baseline_path(root, seed)
        if not bpath.exists():
            raise FileNotFoundError(f"Missing Exp5.3.2 baseline: {bpath}")
        baseline = json.loads(bpath.read_text(encoding="utf-8"))
        if baseline.get("protocol_version") != PROTOCOL_VERSION or baseline.get("seed") != seed:
            raise ValueError(f"Baseline identity mismatch: {bpath}")
        for entry in baseline["baselines"]:
            baseline_rows.append({"seed": seed, **entry})

        reference_path = exp52.local_reference_path(source_results_dir(repo_root), seed)
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing Exp5.2 local reference: {reference_path}")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference.get("protocol_version") != exp52.PROTOCOL_VERSION:
            raise ValueError(f"Wrong Exp5.2 local reference identity: {reference_path}")
        for probe in reference["probes"]:
            local_rows.append(
                {
                    "seed": seed,
                    "probe_type": probe["probe_type"],
                    "val_ba": probe["probe_val_balanced_accuracy"],
                    "test_ba": probe["probe_test_balanced_accuracy"],
                    "test_accuracy": probe["probe_test_accuracy"],
                    "test_macro_f1": probe["probe_test_macro_f1"],
                }
            )

    outputs = {
        "runs": root / "runs.csv",
        "histories": root / "histories.csv",
        "probe_runs": root / "probe_runs.csv",
        "group_probe_runs": root / "group_probe_runs.csv",
        "ablation_runs": root / "ablation_runs.csv",
        "baseline_runs": root / "baseline_runs.csv",
        "local_reference": root / "local_reference.csv",
        "manifest": root / "manifest.json",
    }
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outputs["runs"], index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(outputs["histories"], index=False)
    pd.DataFrame(probe_rows).to_csv(outputs["probe_runs"], index=False)
    pd.DataFrame(group_rows).to_csv(outputs["group_probe_runs"], index=False)
    pd.DataFrame(ablation_rows).to_csv(outputs["ablation_runs"], index=False)
    pd.DataFrame(baseline_rows).to_csv(outputs["baseline_runs"], index=False)
    pd.DataFrame(local_rows).to_csv(outputs["local_reference"], index=False)
    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "expected_runs": len(run_specs()),
            "conditions": [asdict(condition) for condition in CONDITIONS],
            "seeds": list(SEEDS),
            "source_experiment_id": exp52.EXPERIMENT_ID,
            "source_protocol_version": exp52.PROTOCOL_VERSION,
            "primary_when_interface": "post-reset membrane U_t",
            "training_objective": "L_WHEN = L_phase(U_t) + L_progress(U_t)",
            "phase_supervision": "10-way relative progress, diagnostic target built from known training segment end",
            "progress_supervision": "continuous t/(T-1), target only; final duration is not an input",
            "primary_ranking": "high ordered U phase BA; positive ordered-reset history gain; low progress MAE; prefer spike-window accessibility",
            "causal_ablation": ["state_reset", "temporal_shuffle"],
            "spike_readouts": ["instantaneous spike", "trailing 250ms count", "trailing 500ms count"],
            "baselines": ["current WHAT z_t only", "elapsed time t/fs only"],
            "aggregation_policy": "finalizer only aggregates completed run/baseline artifacts; notebook is analysis-only",
            "checkpoint_selection": "max validation native U phase BA; tie lower progress MAE; tie lower total loss",
            "shuffle_replicates": SHUFFLE_REPLICATES,
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
        description="Experiment 5.3.2 supervised causal WHEN representation benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo-root", type=str, default=None)
        subparser.add_argument("--device", type=str, default="cpu")
        subparser.add_argument("--threads", type=int, default=1)
        subparser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        subparser.add_argument("--epochs", type=int, default=EPOCHS)
        subparser.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare-local")
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
    if args.command == "prepare-local":
        task_id = int(args.array_task_id)
        if not 0 <= task_id < len(SEEDS):
            raise IndexError(f"prepare-local task {task_id} outside 0..{len(SEEDS)-1}")
        print(prepare_local_seed(SEEDS[task_id], data, config, force=args.force))
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
