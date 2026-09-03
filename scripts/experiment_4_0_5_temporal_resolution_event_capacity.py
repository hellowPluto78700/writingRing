from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from scripts import experiment_4_0_fixed250_temporal_snn as exp40
from scripts import experiment_4_0_1_multispike_macro_lif as exp401


EXPERIMENT_ID = "experiment_4_0_5_temporal_resolution_event_capacity"
PROTOCOL_VERSION = "hierarchical_temporal_resolution_event_cap_v1"
INPUT_REPRESENTATIONS = ("fixed250", "repeat4", "raw64")
VARIANTS: tuple[tuple[str, int, int], ...] = (
    ("binary", 1, 1),
    ("multi_h", 31, 1),
    ("multi_ho", 31, 31),
)
SEEDS = exp40.SEEDS
EXPECTED_RUNS = len(INPUT_REPRESENTATIONS) * len(VARIANTS) * len(SEEDS)

LOCAL_WIDTH = 128
STATE_WIDTH = 128
STATE_TAU_MEM_MS = 250.0
OUTPUT_TAU_MEM_MS = 250.0
LOCAL_BETA = 0.0
REPEAT_FACTOR = 4
THRESHOLD = exp401.THRESHOLD
BATCH_SIZE = exp401.BATCH_SIZE
EPOCHS = exp401.EPOCHS
LR = exp401.LR
WEIGHT_DECAY = exp401.WEIGHT_DECAY
PROBE_MAX_ITER = exp40.LOGREG_MAX_ITER
EPS = exp40.EPS


@dataclass(frozen=True)
class RunSpec:
    representation: str
    variant: str
    hidden_cap: int
    output_cap: int
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.representation}__{self.variant}__"
            f"hcap{self.hidden_cap}__ocap{self.output_cap}__seed{self.seed}"
        )


@dataclass
class TemporalData:
    representation: str
    Xtr: np.ndarray
    ytr: np.ndarray
    vtr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    vva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    vte: np.ndarray
    channel_scale: np.ndarray
    labels: tuple[str, ...]
    split: dict[str, tuple[str, ...]]
    fs: float
    dt_ms: float
    n_steps: int
    information_contract: str


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    resume: bool = True
    threads: int = 1


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(representation, variant, hidden_cap, output_cap, seed)
        for representation in INPUT_REPRESENTATIONS
        for variant, hidden_cap, output_cap in VARIANTS
        for seed in SEEDS
    ]


def paired_seed(spec: RunSpec, role: str) -> int:
    """Pair model initialization and sample order across all conditions per seed."""
    return exp40.base.dseed(spec.seed, "exp4_0_5_paired", role)


def state_beta(dt_ms: float) -> float:
    return math.exp(-float(dt_ms) / STATE_TAU_MEM_MS)


def output_beta(dt_ms: float) -> float:
    return math.exp(-float(dt_ms) / OUTPUT_TAU_MEM_MS)


def repeat4_from_fixed(
    X: np.ndarray,
    valid_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand each Fixed250 vector into four equal-mass 62.5-ms microsteps.

    This transformation is deterministic and contains no sub-bin timing. The
    four microsteps sum exactly to the original Fixed250 vector, so input event
    mass is not multiplied when temporal opportunities are increased.
    """
    X = np.asarray(X, dtype=np.float32)
    valid_bins = np.asarray(valid_bins, dtype=np.int64)
    expanded = np.repeat(X / float(REPEAT_FACTOR), REPEAT_FACTOR, axis=1)
    valid_steps = valid_bins * REPEAT_FACTOR
    return expanded.astype(np.float32, copy=False), valid_steps


def _assert_mass_equivalence(
    fixed: np.ndarray,
    candidate: np.ndarray,
    name: str,
) -> None:
    fixed_mass = fixed.sum(axis=1, dtype=np.float64)
    candidate_mass = candidate.sum(axis=1, dtype=np.float64)
    if not np.allclose(fixed_mass, candidate_mass, rtol=2e-5, atol=2e-5):
        max_error = float(np.max(np.abs(fixed_mass - candidate_mass)))
        raise ValueError(f"{name} does not preserve Fixed250 total event mass; max error={max_error}")


def prepare_all_representations(repo_root: Path) -> dict[str, TemporalData]:
    """Build all three views from one raw split and one Fixed250-fitted scaler."""
    raw = exp40.base.prepare_data(repo_root)
    if raw.Xtr.shape[2] != exp40.EVENT_CHANNELS:
        raise ValueError(f"Expected {exp40.EVENT_CHANNELS} event channels")
    if not np.isclose(raw.fs, 64.0):
        raise ValueError(f"Expected 64 Hz raw events, got {raw.fs}")
    if raw.bin_steps != 16:
        raise ValueError(f"Expected 250 ms = 16 raw samples, got {raw.bin_steps}")

    fixed_parts: list[tuple[np.ndarray, np.ndarray]] = []
    for X, lengths in (
        (raw.Xtr, raw.ltr),
        (raw.Xva, raw.lva),
        (raw.Xte, raw.lte),
    ):
        fixed_parts.append(exp40.fixed250_channel_counts(X, lengths, raw.bin_steps))
    (Ftr, btr), (Fva, bva), (Fte, bte) = fixed_parts
    scale = exp40.fit_zero_preserving_channel_scale(Ftr, btr)
    Ftr = exp40.apply_channel_scale(Ftr, scale)
    Fva = exp40.apply_channel_scale(Fva, scale)
    Fte = exp40.apply_channel_scale(Fte, scale)

    R4tr, r4tr = repeat4_from_fixed(Ftr, btr)
    R4va, r4va = repeat4_from_fixed(Fva, bva)
    R4te, r4te = repeat4_from_fixed(Fte, bte)

    Xtr = exp40.apply_channel_scale(raw.Xtr, scale)
    Xva = exp40.apply_channel_scale(raw.Xva, scale)
    Xte = exp40.apply_channel_scale(raw.Xte, scale)
    _assert_mass_equivalence(Ftr, R4tr, "repeat4 train")
    _assert_mass_equivalence(Fva, R4va, "repeat4 val")
    _assert_mass_equivalence(Fte, R4te, "repeat4 test")
    _assert_mass_equivalence(Ftr, Xtr, "raw64 train")
    _assert_mass_equivalence(Fva, Xva, "raw64 val")
    _assert_mass_equivalence(Fte, Xte, "raw64 test")

    common = {
        "channel_scale": scale,
        "labels": raw.labels,
        "split": raw.split,
        "fs": float(raw.fs),
    }
    return {
        "fixed250": TemporalData(
            representation="fixed250",
            Xtr=Ftr,
            ytr=raw.ytr,
            vtr=btr,
            Xva=Fva,
            yva=raw.yva,
            vva=bva,
            Xte=Fte,
            yte=raw.yte,
            vte=bte,
            dt_ms=exp40.FIXED_MS,
            n_steps=Ftr.shape[1],
            information_contract=(
                "Fixed250 channel-wise weighted event sums; within-bin timing discarded"
            ),
            **common,
        ),
        "repeat4": TemporalData(
            representation="repeat4",
            Xtr=R4tr,
            ytr=raw.ytr,
            vtr=r4tr,
            Xva=R4va,
            yva=raw.yva,
            vva=r4va,
            Xte=R4te,
            yte=raw.yte,
            vte=r4te,
            dt_ms=exp40.FIXED_MS / REPEAT_FACTOR,
            n_steps=R4tr.shape[1],
            information_contract=(
                "same Fixed250 vectors and total event mass as fixed250, deterministically split "
                "into four equal 62.5-ms microsteps; no within-bin timing restored"
            ),
            **common,
        ),
        "raw64": TemporalData(
            representation="raw64",
            Xtr=Xtr,
            ytr=raw.ytr,
            vtr=raw.ltr,
            Xva=Xva,
            yva=raw.yva,
            vva=raw.lva,
            Xte=Xte,
            yte=raw.yte,
            vte=raw.lte,
            dt_ms=1000.0 / float(raw.fs),
            n_steps=Xtr.shape[1],
            information_contract=(
                "original 64-Hz 30-channel weighted event sequence; genuine within-Fixed250 timing retained"
            ),
            **common,
        ),
    }


def parameter_counts(n_classes: int) -> dict[str, int]:
    input_local = exp40.EVENT_CHANNELS * LOCAL_WIDTH
    local_state = LOCAL_WIDTH * STATE_WIDTH
    recurrent = STATE_WIDTH * STATE_WIDTH
    state_output = STATE_WIDTH * n_classes
    return {
        "input_local": int(input_local),
        "local_state": int(local_state),
        "recurrent": int(recurrent),
        "state_output": int(state_output),
        "total": int(input_local + local_state + recurrent + state_output),
    }


class HierarchicalTemporalDecoder(nn.Module):
    """Local multi-event encoder followed by a recurrent temporal memory layer."""

    def __init__(
        self,
        n_classes: int,
        dt_ms: float,
        hidden_cap: int,
        output_cap: int,
    ) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.dt_ms = float(dt_ms)
        self.hidden_cap = int(hidden_cap)
        self.output_cap = int(output_cap)
        self.input_local = nn.Linear(exp40.EVENT_CHANNELS, LOCAL_WIDTH, bias=False)
        self.local_lif = exp401.MacroMultiSpikeLIF(
            beta=LOCAL_BETA,
            threshold=THRESHOLD,
            max_spikes_per_dt=hidden_cap,
        )
        self.local_state = nn.Linear(LOCAL_WIDTH, STATE_WIDTH, bias=False)
        self.recurrent = nn.Linear(STATE_WIDTH, STATE_WIDTH, bias=False)
        self.state_lif = exp401.MacroMultiSpikeLIF(
            beta=state_beta(dt_ms),
            threshold=THRESHOLD,
            max_spikes_per_dt=hidden_cap,
        )
        self.state_output = nn.Linear(STATE_WIDTH, n_classes, bias=False)
        self.output_lif = exp401.MacroMultiSpikeLIF(
            beta=output_beta(dt_ms),
            threshold=THRESHOLD,
            max_spikes_per_dt=output_cap,
        )

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, n_steps, channels = x.shape
        if channels != exp40.EVENT_CHANNELS:
            raise ValueError(f"Expected {exp40.EVENT_CHANNELS} channels, got {channels}")
        local_mem = torch.zeros(batch, LOCAL_WIDTH, device=x.device, dtype=x.dtype)
        state_mem = torch.zeros(batch, STATE_WIDTH, device=x.device, dtype=x.dtype)
        prev_state_spikes = torch.zeros_like(state_mem)
        output_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)

        local_spikes_seq: list[torch.Tensor] = []
        local_pre_seq: list[torch.Tensor] = []
        state_spikes_seq: list[torch.Tensor] = []
        state_mems_seq: list[torch.Tensor] = []
        state_pre_seq: list[torch.Tensor] = []
        output_spikes_seq: list[torch.Tensor] = []
        output_mems_seq: list[torch.Tensor] = []
        output_pre_seq: list[torch.Tensor] = []

        for step in range(n_steps):
            local_current = self.input_local(x[:, step])
            local_spikes, local_mem, local_pre = self.local_lif(local_current, local_mem)
            state_current = self.local_state(local_spikes) + self.recurrent(prev_state_spikes)
            state_spikes, state_mem, state_pre = self.state_lif(state_current, state_mem)
            output_current = self.state_output(state_spikes)
            output_spikes, output_mem, output_pre = self.output_lif(output_current, output_mem)

            local_spikes_seq.append(local_spikes)
            local_pre_seq.append(local_pre)
            state_spikes_seq.append(state_spikes)
            state_mems_seq.append(state_mem)
            state_pre_seq.append(state_pre)
            output_spikes_seq.append(output_spikes)
            output_mems_seq.append(output_mem)
            output_pre_seq.append(output_pre)
            prev_state_spikes = state_spikes

        return {
            "local_spikes": torch.stack(local_spikes_seq, dim=1),
            "local_pre_reset": torch.stack(local_pre_seq, dim=1),
            "state_spikes": torch.stack(state_spikes_seq, dim=1),
            "state_membranes": torch.stack(state_mems_seq, dim=1),
            "state_pre_reset": torch.stack(state_pre_seq, dim=1),
            "output_spikes": torch.stack(output_spikes_seq, dim=1),
            "output_membranes": torch.stack(output_mems_seq, dim=1),
            "output_pre_reset": torch.stack(output_pre_seq, dim=1),
        }


def normalized_output_evidence(
    output_spikes: torch.Tensor,
    valid_steps: torch.Tensor,
    output_cap: int,
) -> torch.Tensor:
    return exp40.valid_whole_count(
        output_spikes / float(output_cap),
        valid_steps,
    )


def full_output_evidence(output_spikes: torch.Tensor, output_cap: int) -> torch.Tensor:
    return exp40.full_whole_count(output_spikes / float(output_cap))


def loader(
    X: np.ndarray,
    y: np.ndarray,
    valid_steps: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> torch.utils.data.DataLoader:
    return exp40.loader(X, y, valid_steps, batch_size, shuffle, seed)


def make_loaders(
    data: TemporalData,
    spec: RunSpec,
    batch_size: int,
    train_shuffle: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    parts = {
        "train": (data.Xtr, data.ytr, data.vtr),
        "val": (data.Xva, data.yva, data.vva),
        "test": (data.Xte, data.yte, data.vte),
    }
    return {
        split: loader(
            X,
            y,
            valid,
            batch_size,
            train_shuffle if split == "train" else False,
            paired_seed(spec, f"{split}_loader"),
        )
        for split, (X, y, valid) in parts.items()
    }


def _stats_per_second(stats: dict[str, float], dt_ms: float) -> dict[str, float]:
    out = dict(stats)
    out["mean_events_per_neuron_second"] = (
        float(stats["mean_events_per_neuron_step"]) * 1000.0 / float(dt_ms)
    )
    return out


def evaluate_model(
    model: HierarchicalTemporalDecoder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    valid_pred_parts: list[np.ndarray] = []
    full_pred_parts: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    local_stat_sums: dict[str, float] = {}
    state_stat_sums: dict[str, float] = {}
    output_stat_sums: dict[str, float] = {}
    stat_weight = 0
    state_tail_weighted = 0.0
    output_tail_weighted = 0.0
    valid_step_sum = 0.0

    with torch.no_grad():
        for X, y, valid_steps in data_loader:
            X = X.to(device)
            y = y.to(device)
            valid_steps = valid_steps.to(device)
            traj = model.forward_trajectory(X)
            valid_logits = normalized_output_evidence(
                traj["output_spikes"], valid_steps, model.output_cap
            )
            full_logits = full_output_evidence(traj["output_spikes"], model.output_cap)
            loss = F.cross_entropy(valid_logits, y)
            n = len(y)
            n_total += n
            loss_sum += float(loss.item()) * n
            valid_step_sum += float(valid_steps.sum().item())
            y_true_parts.append(y.cpu().numpy())
            valid_pred_parts.append(valid_logits.argmax(dim=1).cpu().numpy())
            full_pred_parts.append(full_logits.argmax(dim=1).cpu().numpy())

            stats_triplet = (
                (
                    local_stat_sums,
                    exp401._distribution_stats(
                        traj["local_spikes"],
                        traj["local_pre_reset"],
                        valid_steps,
                        model.hidden_cap,
                    ),
                ),
                (
                    state_stat_sums,
                    exp401._distribution_stats(
                        traj["state_spikes"],
                        traj["state_pre_reset"],
                        valid_steps,
                        model.hidden_cap,
                    ),
                ),
                (
                    output_stat_sums,
                    exp401._distribution_stats(
                        traj["output_spikes"],
                        traj["output_pre_reset"],
                        valid_steps,
                        model.output_cap,
                    ),
                ),
            )
            for target, source in stats_triplet:
                for key, value in source.items():
                    target[key] = target.get(key, 0.0) + float(value) * n
            state_tail_weighted += exp401._tail_fraction(
                traj["state_spikes"], valid_steps
            ) * n
            output_tail_weighted += exp401._tail_fraction(
                traj["output_spikes"], valid_steps
            ) * n
            stat_weight += n

    y_true = np.concatenate(y_true_parts)
    valid_pred = np.concatenate(valid_pred_parts)
    full_pred = np.concatenate(full_pred_parts)
    divisor = max(stat_weight, 1)

    def mean_stats(sums: dict[str, float]) -> dict[str, float]:
        base = {key: float(value / divisor) for key, value in sums.items()}
        return _stats_per_second(base, model.dt_ms)

    mean_valid_steps = valid_step_sum / max(n_total, 1)
    return {
        "valid_count_loss": float(loss_sum / max(n_total, 1)),
        "valid_count": exp40.metrics(y_true, valid_pred),
        "full_count": exp40.metrics(y_true, full_pred),
        "local_valid_stats": mean_stats(local_stat_sums),
        "state_valid_stats": mean_stats(state_stat_sums),
        "output_valid_stats": mean_stats(output_stat_sums),
        "state_tail_event_fraction": float(state_tail_weighted / divisor),
        "output_tail_event_fraction": float(output_tail_weighted / divisor),
        "mean_valid_steps": float(mean_valid_steps),
        "mean_valid_duration_s": float(mean_valid_steps * model.dt_ms / 1000.0),
        "mean_theoretical_output_count_levels": float(
            mean_valid_steps * model.output_cap + 1.0
        ),
    }


def extract_probe_features(
    model: HierarchicalTemporalDecoder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hidden WholeCount and endpoint membrane; neither exposes phase slots."""
    count_parts: list[np.ndarray] = []
    endpoint_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, valid_steps in data_loader:
            X = X.to(device)
            valid_steps = valid_steps.to(device)
            traj = model.forward_trajectory(X)
            hidden_count = exp40.valid_whole_count(traj["state_spikes"], valid_steps)
            endpoint = exp40.valid_final_membrane(traj["state_membranes"], valid_steps)
            count_parts.append(hidden_count.cpu().numpy())
            endpoint_parts.append(endpoint.cpu().numpy())
            label_parts.append(y.numpy())
    return (
        np.concatenate(count_parts),
        np.concatenate(endpoint_parts),
        np.concatenate(label_parts),
    )


def fit_linear_probes(
    model: HierarchicalTemporalDecoder,
    loaders: dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> dict[str, object]:
    extracted = {
        split: extract_probe_features(model, data_loader, device)
        for split, data_loader in loaders.items()
    }
    train_count, train_endpoint, ytr = extracted["train"]
    probes: dict[str, LogisticRegression] = {}
    for name, features in (
        ("hidden_whole_count", train_count),
        ("uend", train_endpoint),
    ):
        probe = LogisticRegression(
            max_iter=PROBE_MAX_ITER,
            class_weight="balanced",
            solver="lbfgs",
        )
        probe.fit(features, ytr)
        probes[name] = probe

    out: dict[str, object] = {
        "temporal_phase_access": False,
        "source_snn_frozen": True,
        "classifier": "balanced LogisticRegression(lbfgs)",
        "hidden_whole_count_feature_dim": STATE_WIDTH,
        "uend_feature_dim": STATE_WIDTH,
        "metrics": {},
    }
    for split, (hidden_count, endpoint, y) in extracted.items():
        out["metrics"][split] = {
            "hidden_whole_count_linear": exp40.metrics(
                y, probes["hidden_whole_count"].predict(hidden_count)
            ),
            "uend_linear": exp40.metrics(y, probes["uend"].predict(endpoint)),
        }
    return out


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def train_one(
    spec: RunSpec,
    data: TemporalData,
    root: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    threads: int,
    force: bool,
) -> Path:
    destination = checkpoint_path(root, spec)
    if destination.exists() and not force:
        return destination

    torch.set_num_threads(threads)
    exp40.base.seed_all(paired_seed(spec, "model_init"))
    model = HierarchicalTemporalDecoder(
        n_classes=len(data.labels),
        dt_ms=data.dt_ms,
        hidden_cap=spec.hidden_cap,
        output_cap=spec.output_cap,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    train_loader = make_loaders(data, spec, batch_size, train_shuffle=True)["train"]
    val_loader = make_loaders(data, spec, batch_size, train_shuffle=False)["val"]

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for X, y, valid_steps in train_loader:
            X = X.to(device)
            y = y.to(device)
            valid_steps = valid_steps.to(device)
            optimizer.zero_grad(set_to_none=True)
            traj = model.forward_trajectory(X)
            logits = normalized_output_evidence(
                traj["output_spikes"], valid_steps, model.output_cap
            )
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            n = len(y)
            train_loss_sum += float(loss.item()) * n
            train_n += n
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = exp40.metrics(
            np.concatenate(train_true), np.concatenate(train_pred)
        )
        val_metrics = evaluate_model(model, val_loader, device)
        val_ba = float(val_metrics["valid_count"]["balanced_accuracy"])
        val_loss = float(val_metrics["valid_count_loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_n, 1),
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_loss": val_loss,
                "val_balanced_accuracy": val_ba,
            }
        )
        improved = val_ba > best_val_ba + 1e-12 or (
            abs(val_ba - best_val_ba) <= 1e-12 and val_loss < best_val_loss
        )
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "dt_ms": data.dt_ms,
            "state_tau_mem_ms": STATE_TAU_MEM_MS,
            "output_tau_mem_ms": OUTPUT_TAU_MEM_MS,
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_ba,
            "best_val_loss": best_val_loss,
            "model_state_dict": best_state,
            "channel_scale": data.channel_scale,
            "labels": data.labels,
            "split": data.split,
        },
        destination,
    )
    history_file = history_path(root, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False)
    return destination


def load_best_model(
    spec: RunSpec,
    data: TemporalData,
    root: Path,
    device: torch.device,
) -> tuple[HierarchicalTemporalDecoder, dict[str, object]]:
    path = checkpoint_path(root, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint payload: {path}")
    stored_spec = checkpoint.get("spec")
    if stored_spec != spec.__dict__:
        raise ValueError(f"Checkpoint identity mismatch for {spec.key}: {stored_spec}")
    if not np.isclose(float(checkpoint.get("dt_ms", -1.0)), data.dt_ms):
        raise ValueError(f"Checkpoint dt mismatch for {spec.key}")
    model = HierarchicalTemporalDecoder(
        n_classes=len(data.labels),
        dt_ms=data.dt_ms,
        hidden_cap=spec.hidden_cap,
        output_cap=spec.output_cap,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_one(
    spec: RunSpec,
    data: TemporalData,
    root: Path,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, object]:
    destination = evaluation_path(root, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    model, checkpoint = load_best_model(spec, data, root, device)
    loaders = make_loaders(data, spec, batch_size, train_shuffle=False)
    metrics = {
        split: evaluate_model(model, data_loader, device)
        for split, data_loader in loaders.items()
    }
    probes = fit_linear_probes(model, loaders, device)
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "architecture": {
            "topology": "30 -> Local128(beta=0,no recurrence) -> RSNN128 -> 12",
            "parameter_counts": parameter_counts(len(data.labels)),
            "state_tau_mem_ms": STATE_TAU_MEM_MS,
            "output_tau_mem_ms": OUTPUT_TAU_MEM_MS,
            "dt_ms": data.dt_ms,
            "state_beta": state_beta(data.dt_ms),
            "output_beta": output_beta(data.dt_ms),
            "threshold": THRESHOLD,
        },
        "input": {
            "representation": data.representation,
            "information_contract": data.information_contract,
            "n_steps": data.n_steps,
            "dt_ms": data.dt_ms,
            "channel_scale_source": "training Fixed250 valid bins only; shared by all representations",
            "channel_scale": data.channel_scale.tolist(),
        },
        "checkpoint": {
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_val_balanced_accuracy": float(
                checkpoint["best_val_balanced_accuracy"]
            ),
            "best_val_loss": float(checkpoint["best_val_loss"]),
        },
        "metrics": metrics,
        "linear_probes": probes,
        "provenance": {
            "split_seed": int(exp40.base.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "event_channels": exp40.EVENT_CHANNELS,
            "readout_training_objective": "valid normalized output WholeCount CE",
            "probe_training": (
                "post-hoc on frozen best SNN; train partition only; no phase-slot access"
            ),
        },
    }
    _save_json(destination, payload)
    return payload


def run_one(
    spec: RunSpec,
    all_data: dict[str, TemporalData],
    config: Config,
    force: bool,
) -> dict[str, object]:
    data = all_data[spec.representation]
    device = torch.device(config.device)
    train_one(
        spec,
        data,
        config.results_dir,
        device,
        config.epochs,
        config.batch_size,
        config.threads,
        force,
    )
    return evaluate_one(
        spec,
        data,
        config.results_dir,
        device,
        config.batch_size,
        force,
    )


def _run_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing required evaluation artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        probes = payload["linear_probes"]["metrics"]
        for split in ("train", "val", "test"):
            metrics = payload["metrics"][split]
            hidden_probe = probes[split]["hidden_whole_count_linear"]
            uend_probe = probes[split]["uend_linear"]
            row: dict[str, object] = {
                "representation": spec.representation,
                "variant": spec.variant,
                "hidden_cap": spec.hidden_cap,
                "output_cap": spec.output_cap,
                "seed": spec.seed,
                "split": split,
                "dt_ms": float(payload["architecture"]["dt_ms"]),
                "state_beta": float(payload["architecture"]["state_beta"]),
                "output_beta": float(payload["architecture"]["output_beta"]),
                "valid_count_loss": float(metrics["valid_count_loss"]),
                "valid_count_accuracy": float(metrics["valid_count"]["accuracy"]),
                "valid_count_balanced_accuracy": float(
                    metrics["valid_count"]["balanced_accuracy"]
                ),
                "valid_count_macro_f1": float(metrics["valid_count"]["macro_f1"]),
                "full_count_balanced_accuracy": float(
                    metrics["full_count"]["balanced_accuracy"]
                ),
                "hidden_count_linear_balanced_accuracy": float(
                    hidden_probe["balanced_accuracy"]
                ),
                "hidden_count_linear_macro_f1": float(hidden_probe["macro_f1"]),
                "uend_linear_balanced_accuracy": float(
                    uend_probe["balanced_accuracy"]
                ),
                "uend_linear_macro_f1": float(uend_probe["macro_f1"]),
                "state_tail_event_fraction": float(
                    metrics["state_tail_event_fraction"]
                ),
                "output_tail_event_fraction": float(
                    metrics["output_tail_event_fraction"]
                ),
                "mean_valid_steps": float(metrics["mean_valid_steps"]),
                "mean_valid_duration_s": float(metrics["mean_valid_duration_s"]),
                "mean_theoretical_output_count_levels": float(
                    metrics["mean_theoretical_output_count_levels"]
                ),
            }
            for prefix in ("local", "state", "output"):
                stats = metrics[f"{prefix}_valid_stats"]
                for key, value in stats.items():
                    row[f"{prefix}_{key}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def _readout_columns() -> dict[str, str]:
    return {
        "output_whole_count": "valid_count_balanced_accuracy",
        "hidden_whole_count_linear": "hidden_count_linear_balanced_accuracy",
        "uend_linear": "uend_linear_balanced_accuracy",
    }


def _paired_effects(test: pd.DataFrame) -> pd.DataFrame:
    indexed = test.set_index(["representation", "variant", "seed"])
    rows: list[dict[str, object]] = []
    for variant, _, _ in VARIANTS:
        for seed in SEEDS:
            for readout, column in _readout_columns().items():
                fixed = float(indexed.loc[("fixed250", variant, seed), column])
                repeat = float(indexed.loc[("repeat4", variant, seed), column])
                raw = float(indexed.loc[("raw64", variant, seed), column])
                rows.extend(
                    [
                        {
                            "effect": "repeat4_minus_fixed250",
                            "variant": variant,
                            "seed": seed,
                            "readout": readout,
                            "delta_balanced_accuracy": repeat - fixed,
                        },
                        {
                            "effect": "raw64_minus_repeat4",
                            "variant": variant,
                            "seed": seed,
                            "readout": readout,
                            "delta_balanced_accuracy": raw - repeat,
                        },
                        {
                            "effect": "raw64_minus_fixed250",
                            "variant": variant,
                            "seed": seed,
                            "readout": readout,
                            "delta_balanced_accuracy": raw - fixed,
                        },
                    ]
                )
    for representation in INPUT_REPRESENTATIONS:
        for seed in SEEDS:
            for readout, column in _readout_columns().items():
                binary = float(indexed.loc[(representation, "binary", seed), column])
                multi_h = float(indexed.loc[(representation, "multi_h", seed), column])
                multi_ho = float(indexed.loc[(representation, "multi_ho", seed), column])
                rows.extend(
                    [
                        {
                            "effect": "multi_h_minus_binary",
                            "representation": representation,
                            "seed": seed,
                            "readout": readout,
                            "delta_balanced_accuracy": multi_h - binary,
                        },
                        {
                            "effect": "multi_ho_minus_binary",
                            "representation": representation,
                            "seed": seed,
                            "readout": readout,
                            "delta_balanced_accuracy": multi_ho - binary,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    runs = _run_rows(root)
    expected_rows = EXPECTED_RUNS * 3
    if len(runs) != expected_rows:
        raise ValueError(f"Expected {expected_rows} finalized rows, got {len(runs)}")
    runs_file = root / "runs.csv"
    runs.to_csv(runs_file, index=False)

    test = runs[runs["split"] == "test"].copy()
    summary_columns = [
        "valid_count_balanced_accuracy",
        "hidden_count_linear_balanced_accuracy",
        "uend_linear_balanced_accuracy",
        "state_tail_event_fraction",
        "output_tail_event_fraction",
        "state_mean_events_per_neuron_second",
        "output_mean_events_per_neuron_second",
        "state_fraction_at_cap",
        "output_fraction_at_cap",
    ]
    summary = (
        test.groupby(["representation", "variant", "hidden_cap", "output_cap"])[
            summary_columns
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    summary_file = root / "summary.csv"
    summary.to_csv(summary_file, index=False)

    effects = _paired_effects(test)
    effects_file = root / "paired_effects.csv"
    effects.to_csv(effects_file, index=False)
    effect_summary = (
        effects.groupby(
            [col for col in ("effect", "representation", "variant", "readout") if col in effects.columns],
            dropna=False,
        )["delta_balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    effect_summary_file = root / "paired_effects_summary.csv"
    effect_summary.to_csv(effect_summary_file, index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "expected_runs": EXPECTED_RUNS,
        "representations": INPUT_REPRESENTATIONS,
        "variants": [
            {"variant": variant, "hidden_cap": hidden, "output_cap": output}
            for variant, hidden, output in VARIANTS
        ],
        "seeds": SEEDS,
        "architecture": "30 -> Local128(beta=0) -> RSNN128(tau_mem=250ms physical) -> 12",
        "readouts": list(_readout_columns()),
        "primary_training_objective": "valid normalized output WholeCount CE",
        "files": {
            "runs": runs_file.name,
            "summary": summary_file.name,
            "paired_effects": effects_file.name,
            "paired_effects_summary": effect_summary_file.name,
        },
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    return {
        "runs": runs_file,
        "summary": summary_file,
        "paired_effects": effects_file,
        "paired_effects_summary": effect_summary_file,
        "manifest": manifest_file,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-one")
    run_parser.add_argument("--array-task-id", type=int, required=True)
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--threads", type=int, default=1)
    run_parser.add_argument("--epochs", type=int, default=EPOCHS)
    run_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = exp40.find_repo_root()
    root = results_dir(repo_root)
    if args.command == "finalize":
        paths = finalize_experiment(repo_root)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    specs = run_specs()
    if args.array_task_id < 0 or args.array_task_id >= len(specs):
        raise IndexError(
            f"array task {args.array_task_id} outside [0, {len(specs) - 1}]"
        )
    spec = specs[args.array_task_id]
    all_data = prepare_all_representations(repo_root)
    config = Config(
        repo_root=repo_root,
        results_dir=root,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threads=args.threads,
    )
    payload = run_one(spec, all_data, config, force=args.force)
    test = payload["metrics"]["test"]["valid_count"]["balanced_accuracy"]
    hcount = payload["linear_probes"]["metrics"]["test"][
        "hidden_whole_count_linear"
    ]["balanced_accuracy"]
    uend = payload["linear_probes"]["metrics"]["test"]["uend_linear"][
        "balanced_accuracy"
    ]
    print(
        f"completed {spec.key}: output_count_BA={test:.6f} "
        f"hidden_count_linear_BA={hcount:.6f} uend_linear_BA={uend:.6f}"
    )


if __name__ == "__main__":
    main()
