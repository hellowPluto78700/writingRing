from __future__ import annotations

import argparse
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts import experiment_3_0_1_single_tau_objective_comparison as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401


EXPERIMENT_ID = "experiment_5_0_local_evidence_objectives"
PROTOCOL_VERSION = "local_objective_x_event_capacity_v1"

OBJECTIVES = (
    "whole_count_ce",
    "timestep_ce",
    "fixed250_bin_ce",
    "fixed500_bin_ce",
    "relative10_bin_ce",
)
VARIANTS: tuple[tuple[str, int, int], ...] = (
    ("binary", 1, 1),
    ("multi_h", 31, 1),
    ("multi_ho", 31, 31),
)
SEEDS = (11, 23, 37, 53, 71)

L1_WIDTH = 128
L2_WIDTH = 128
L1_SHIFTS = (2, 3, 4)
L2_SHIFTS = (2, 3, 4)
TAU_MEM_MS = exp3.TAU_MEM_MS
THRESHOLD = exp3.THRESHOLD
SURROGATE_SLOPE = exp3.SURROGATE_SLOPE
RESET = "subtract"
FIXED250_MS = 250.0
FIXED500_MS = 500.0
RELATIVE_BINS = 10
LOGIT_GAIN = 5.0

BATCH_SIZE = exp3.BATCH_SIZE
EPOCHS = exp3.EPOCHS
LR = exp3.LR
WEIGHT_DECAY = 0.0
PROBE_MAX_ITER = 5000

READOUTS = (
    "output_whole_count",
    "l1_whole_count_linear",
    "l2_whole_count_linear",
    "l2_fixed250_ordered_linear",
    "l2_relative10_ordered_linear",
    "l2_uend_linear",
)


@dataclass(frozen=True)
class RunSpec:
    objective: str
    variant: str
    hidden_cap: int
    output_cap: int
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.objective}__{self.variant}__"
            f"hcap{self.hidden_cap}__ocap{self.output_cap}__seed{self.seed}"
        )


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


@dataclass
class FrozenFeatures:
    labels: np.ndarray
    output_count: np.ndarray
    l1_count: np.ndarray
    l2_count: np.ndarray
    l2_fixed250: np.ndarray
    l2_relative10: np.ndarray
    l2_uend: np.ndarray


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(objective, variant, hidden_cap, output_cap, seed)
        for objective in OBJECTIVES
        for variant, hidden_cap, output_cap in VARIANTS
        for seed in SEEDS
    ]


def paired_seed(spec: RunSpec, role: str) -> int:
    """Pair initialization and sample order across all 15 conditions per seed."""
    return exp3.dseed(spec.seed, "exp5_0_paired", role)


def _group_counts(width: int, shifts: tuple[int, ...]) -> tuple[int, ...]:
    base, remainder = divmod(width, len(shifts))
    return tuple(base + (1 if index < remainder else 0) for index in range(len(shifts)))


def alpha_vector(width: int, shifts: tuple[int, ...]) -> torch.Tensor:
    values: list[float] = []
    for shift, count in zip(shifts, _group_counts(width, shifts), strict=True):
        values.extend([exp3.alpha(shift)] * count)
    return torch.tensor(values, dtype=torch.float32)


def beta_value(fs: float) -> float:
    return math.exp(-(1000.0 / float(fs)) / TAU_MEM_MS)


def fixed_steps(ms: float, fs: float) -> int:
    return int(np.rint(float(ms) * float(fs) / 1000.0))


class LocalEvidenceSNN(nn.Module):
    """Exp3-style two-layer multi-tau local SNN with a 12-neuron spiking head.

    L1/L2 have explicit synaptic state with shifts (2,3,4). The membrane time
    constant stays short at the Exp3 value. Binary/Multi-H/Multi-HO share the
    exact same state equations and weights; only the event cap changes.
    """

    def __init__(
        self,
        n_classes: int,
        fs: float,
        hidden_cap: int,
        output_cap: int,
    ) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        self.dt_ms = 1000.0 / self.fs
        self.hidden_cap = int(hidden_cap)
        self.output_cap = int(output_cap)
        beta = beta_value(self.fs)

        self.f1 = nn.Linear(exp3.EVENT_CHANNELS, L1_WIDTH, bias=False)
        self.f2 = nn.Linear(L1_WIDTH, L2_WIDTH, bias=False)
        self.output_linear = nn.Linear(L2_WIDTH, self.n_classes, bias=False)
        self.l1_lif = exp401.MacroMultiSpikeLIF(
            beta=beta,
            threshold=THRESHOLD,
            max_spikes_per_dt=self.hidden_cap,
            surrogate_slope=SURROGATE_SLOPE,
        )
        self.l2_lif = exp401.MacroMultiSpikeLIF(
            beta=beta,
            threshold=THRESHOLD,
            max_spikes_per_dt=self.hidden_cap,
            surrogate_slope=SURROGATE_SLOPE,
        )
        self.output_lif = exp401.MacroMultiSpikeLIF(
            beta=beta,
            threshold=THRESHOLD,
            max_spikes_per_dt=self.output_cap,
            surrogate_slope=SURROGATE_SLOPE,
        )
        self.register_buffer("alpha1", alpha_vector(L1_WIDTH, L1_SHIFTS))
        self.register_buffer("alpha2", alpha_vector(L2_WIDTH, L2_SHIFTS))

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, n_steps, channels = x.shape
        if channels != exp3.EVENT_CHANNELS:
            raise ValueError(f"Expected {exp3.EVENT_CHANNELS} channels, got {channels}")

        syn1 = torch.zeros(batch, L1_WIDTH, device=x.device, dtype=x.dtype)
        mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, L2_WIDTH, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(syn2)
        output_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)

        l1_spikes: list[torch.Tensor] = []
        l1_mems: list[torch.Tensor] = []
        l2_spikes: list[torch.Tensor] = []
        l2_mems: list[torch.Tensor] = []
        output_spikes: list[torch.Tensor] = []
        output_mems: list[torch.Tensor] = []

        for step in range(n_steps):
            syn1 = self.alpha1 * syn1 + self.f1(x[:, step])
            s1, mem1, _ = self.l1_lif(syn1, mem1)
            syn2 = self.alpha2 * syn2 + self.f2(s1)
            s2, mem2, _ = self.l2_lif(syn2, mem2)
            so, output_mem, _ = self.output_lif(self.output_linear(s2), output_mem)

            l1_spikes.append(s1)
            l1_mems.append(mem1)
            l2_spikes.append(s2)
            l2_mems.append(mem2)
            output_spikes.append(so)
            output_mems.append(output_mem)

        return {
            "l1_spikes": torch.stack(l1_spikes, dim=1),
            "l1_membranes": torch.stack(l1_mems, dim=1),
            "l2_spikes": torch.stack(l2_spikes, dim=1),
            "l2_membranes": torch.stack(l2_mems, dim=1),
            "output_spikes": torch.stack(output_spikes, dim=1),
            "output_membranes": torch.stack(output_mems, dim=1),
        }


def valid_mask(lengths: torch.Tensor, n_steps: int) -> torch.Tensor:
    return torch.arange(n_steps, device=lengths.device)[None, :] < lengths[:, None]


def valid_sum(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = valid_mask(lengths, values.shape[1]).to(values.dtype).unsqueeze(-1)
    return (values * mask).sum(dim=1)


def valid_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return valid_sum(values, lengths) / lengths.clamp_min(1).to(values.dtype).unsqueeze(-1)


def deployment_logits(
    output_spikes: torch.Tensor,
    lengths: torch.Tensor,
    output_cap: int,
) -> torch.Tensor:
    return valid_sum(output_spikes / float(output_cap), lengths)


def deployment_ce_logits(
    output_spikes: torch.Tensor,
    lengths: torch.Tensor,
    output_cap: int,
) -> torch.Tensor:
    return LOGIT_GAIN * valid_mean(output_spikes / float(output_cap), lengths)


def fixed_bin_means_and_sizes(
    spikes: torch.Tensor,
    lengths: torch.Tensor,
    steps: int,
    cap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if steps < 1:
        raise ValueError("steps must be positive")
    batch, n_steps, dim = spikes.shape
    valid = valid_mask(lengths, n_steps).to(spikes.dtype)
    normalized = spikes / float(cap)
    n_bins = int(math.ceil(n_steps / steps))
    pad_steps = n_bins * steps - n_steps
    masked = normalized * valid.unsqueeze(-1)
    masked = F.pad(masked, (0, 0, 0, pad_steps))
    valid_padded = F.pad(valid, (0, pad_steps))
    sums = masked.reshape(batch, n_bins, steps, dim).sum(dim=2)
    sizes = valid_padded.reshape(batch, n_bins, steps).sum(dim=2)
    means = sums / sizes.clamp_min(1.0).unsqueeze(-1)
    return means, sizes


def relative_bin_means_and_sizes(
    spikes: torch.Tensor,
    lengths: torch.Tensor,
    n_bins: int,
    cap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, n_steps, dim = spikes.shape
    positions = torch.arange(n_steps, device=spikes.device)[None, :].expand(batch, n_steps)
    denominators = lengths.clamp_min(1)[:, None]
    valid = positions < denominators
    bin_index = torch.div(
        positions * n_bins,
        denominators,
        rounding_mode="floor",
    ).clamp(max=n_bins - 1)

    sums = torch.zeros(batch, n_bins, dim, device=spikes.device, dtype=spikes.dtype)
    sizes = torch.zeros(batch, n_bins, device=spikes.device, dtype=spikes.dtype)
    source = (spikes / float(cap)) * valid.to(spikes.dtype).unsqueeze(-1)
    sums.scatter_add_(1, bin_index.unsqueeze(-1).expand(-1, -1, dim), source)
    sizes.scatter_add_(1, bin_index, valid.to(spikes.dtype))
    means = sums / sizes.clamp_min(1.0).unsqueeze(-1)
    return means, sizes


def _weighted_bin_ce(
    means: torch.Tensor,
    sizes: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    batch, n_bins, n_classes = means.shape
    targets = y[:, None].expand(batch, n_bins)
    losses = F.cross_entropy(
        (LOGIT_GAIN * means).reshape(batch * n_bins, n_classes),
        targets.reshape(batch * n_bins),
        reduction="none",
    ).reshape(batch, n_bins)
    weights = sizes / lengths.clamp_min(1).to(sizes.dtype).unsqueeze(-1)
    return (losses * weights).sum(dim=1).mean()


def objective_loss(
    output_spikes: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
    objective: str,
    output_cap: int,
    fs: float,
) -> torch.Tensor:
    if objective == "whole_count_ce":
        return F.cross_entropy(
            deployment_ce_logits(output_spikes, lengths, output_cap),
            y,
        )
    if objective == "timestep_ce":
        batch, n_steps, n_classes = output_spikes.shape
        logits = LOGIT_GAIN * output_spikes / float(output_cap)
        valid = valid_mask(lengths, n_steps)
        targets = y[:, None].expand(batch, n_steps)
        return F.cross_entropy(logits[valid], targets[valid])
    if objective == "fixed250_bin_ce":
        means, sizes = fixed_bin_means_and_sizes(
            output_spikes,
            lengths,
            fixed_steps(FIXED250_MS, fs),
            output_cap,
        )
        return _weighted_bin_ce(means, sizes, lengths, y)
    if objective == "fixed500_bin_ce":
        means, sizes = fixed_bin_means_and_sizes(
            output_spikes,
            lengths,
            fixed_steps(FIXED500_MS, fs),
            output_cap,
        )
        return _weighted_bin_ce(means, sizes, lengths, y)
    if objective == "relative10_bin_ce":
        means, sizes = relative_bin_means_and_sizes(
            output_spikes,
            lengths,
            RELATIVE_BINS,
            output_cap,
        )
        return _weighted_bin_ce(means, sizes, lengths, y)
    raise ValueError(f"Unknown objective: {objective}")


def prepare_data(repo_root: Path) -> exp3.Data:
    return exp3.prepare_data(repo_root)


def make_loaders(
    data: exp3.Data,
    spec: RunSpec,
    batch_size: int,
    train_shuffle: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    parts = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    return {
        split: exp3.loader(
            X,
            y,
            lengths,
            batch_size,
            train_shuffle if split == "train" else False,
            paired_seed(spec, f"{split}_loader"),
        )
        for split, (X, y, lengths) in parts.items()
    }


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _scaled_linear_probe() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=PROBE_MAX_ITER,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def evaluate_deployment(
    model: LocalEvidenceSNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for X, y, lengths in data_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            traj = model.forward_trajectory(X)
            logits = deployment_logits(traj["output_spikes"], lengths, model.output_cap)
            ce_logits = deployment_ce_logits(traj["output_spikes"], lengths, model.output_cap)
            loss = F.cross_entropy(ce_logits, y)
            n = len(y)
            n_total += n
            loss_sum += float(loss.item()) * n
            labels.append(y.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    metrics = exp3.metrics(y_true, y_pred)
    metrics["loss"] = loss_sum / max(n_total, 1)
    return metrics


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
    model = LocalEvidenceSNN(
        n_classes=len(data.labels),
        fs=data.fs,
        hidden_cap=spec.hidden_cap,
        output_cap=spec.output_cap,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loaders = make_loaders(data, spec, config.batch_size, train_shuffle=True)
    train_loader = loaders["train"]
    val_loader = make_loaders(data, spec, config.batch_size, train_shuffle=False)["val"]

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        objective_loss_sum = 0.0
        train_n = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            traj = model.forward_trajectory(X)
            loss = objective_loss(
                traj["output_spikes"],
                lengths,
                y,
                spec.objective,
                model.output_cap,
                data.fs,
            )
            loss.backward()
            optimizer.step()

            logits = deployment_logits(traj["output_spikes"], lengths, model.output_cap)
            n = len(y)
            train_n += n
            objective_loss_sum += float(loss.item()) * n
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = exp3.metrics(np.concatenate(train_true), np.concatenate(train_pred))
        val_metrics = evaluate_deployment(model, val_loader, device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_objective_loss": objective_loss_sum / max(train_n, 1),
                "train_output_whole_count_ba": train_metrics["balanced_accuracy"],
                "val_output_whole_count_loss": val_loss,
                "val_output_whole_count_ba": val_ba,
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
            "best_epoch": best_epoch,
            "best_val_output_whole_count_ba": best_val_ba,
            "best_val_output_whole_count_loss": best_val_loss,
            "model_state_dict": best_state,
            "labels": data.labels,
            "split": data.split,
            "architecture": architecture_manifest(data.fs, len(data.labels)),
        },
        destination,
    )
    history_file = history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
) -> tuple[LocalEvidenceSNN, dict[str, object]]:
    device = torch.device(config.device)
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.0 checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong experiment id in {path}")
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong protocol version in {path}")
    if checkpoint.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint identity mismatch for {spec.key}")
    model = LocalEvidenceSNN(
        n_classes=len(data.labels),
        fs=data.fs,
        hidden_cap=spec.hidden_cap,
        output_cap=spec.output_cap,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _gather_endpoint(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    index = (lengths.clamp_min(1) - 1).to(torch.long)
    rows = torch.arange(len(values), device=values.device)
    return values[rows, index]


def _event_diagnostics(
    spikes: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
    fs: float,
) -> dict[str, float]:
    mask = valid_mask(lengths, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
    valid_events = float((spikes * mask).sum().item())
    tail_events = float((spikes * (1.0 - mask)).sum().item())
    valid_steps = float(lengths.sum().item())
    all_events = valid_events + tail_events
    return {
        "valid_events": valid_events,
        "tail_events": tail_events,
        "events_per_neuron_second": (
            valid_events * float(fs) / max(valid_steps * float(width), 1.0)
        ),
        "tail_event_fraction": tail_events / max(all_events, 1e-12),
    }


def extract_features(
    model: LocalEvidenceSNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    fs: float,
) -> tuple[FrozenFeatures, dict[str, dict[str, float]]]:
    model.eval()
    labels: list[np.ndarray] = []
    output_count: list[np.ndarray] = []
    l1_count: list[np.ndarray] = []
    l2_count: list[np.ndarray] = []
    l2_fixed250: list[np.ndarray] = []
    l2_relative10: list[np.ndarray] = []
    l2_uend: list[np.ndarray] = []
    diagnostics_sum = {
        "l1": {"valid_events": 0.0, "tail_events": 0.0, "valid_steps": 0.0},
        "l2": {"valid_events": 0.0, "tail_events": 0.0, "valid_steps": 0.0},
        "output": {"valid_events": 0.0, "tail_events": 0.0, "valid_steps": 0.0},
    }

    with torch.no_grad():
        for X, y, lengths in data_loader:
            X = X.to(device)
            lengths = lengths.to(device)
            traj = model.forward_trajectory(X)
            labels.append(y.numpy())
            output_count.append(valid_sum(traj["output_spikes"], lengths).cpu().numpy())
            l1_count.append(valid_sum(traj["l1_spikes"], lengths).cpu().numpy())
            l2_count.append(valid_sum(traj["l2_spikes"], lengths).cpu().numpy())
            l2_fixed250.append(
                exp3.fixed_counts(
                    traj["l2_spikes"], lengths, fixed_steps(FIXED250_MS, fs)
                ).flatten(start_dim=1).cpu().numpy()
            )
            l2_relative10.append(
                exp3.relative_counts(
                    traj["l2_spikes"], lengths, RELATIVE_BINS
                ).flatten(start_dim=1).cpu().numpy()
            )
            l2_uend.append(_gather_endpoint(traj["l2_membranes"], lengths).cpu().numpy())

            for name, key, width in (
                ("l1", "l1_spikes", L1_WIDTH),
                ("l2", "l2_spikes", L2_WIDTH),
                ("output", "output_spikes", model.n_classes),
            ):
                mask = valid_mask(lengths, traj[key].shape[1]).to(traj[key].dtype).unsqueeze(-1)
                diagnostics_sum[name]["valid_events"] += float((traj[key] * mask).sum().item())
                diagnostics_sum[name]["tail_events"] += float((traj[key] * (1.0 - mask)).sum().item())
                diagnostics_sum[name]["valid_steps"] += float(lengths.sum().item())

    diagnostics: dict[str, dict[str, float]] = {}
    for name, width in (("l1", L1_WIDTH), ("l2", L2_WIDTH), ("output", model.n_classes)):
        valid_events = diagnostics_sum[name]["valid_events"]
        tail_events = diagnostics_sum[name]["tail_events"]
        valid_steps = diagnostics_sum[name]["valid_steps"]
        diagnostics[name] = {
            "valid_events": valid_events,
            "tail_events": tail_events,
            "events_per_neuron_second": valid_events * fs / max(valid_steps * width, 1.0),
            "tail_event_fraction": tail_events / max(valid_events + tail_events, 1e-12),
        }

    return (
        FrozenFeatures(
            labels=np.concatenate(labels),
            output_count=np.concatenate(output_count),
            l1_count=np.concatenate(l1_count),
            l2_count=np.concatenate(l2_count),
            l2_fixed250=np.concatenate(l2_fixed250),
            l2_relative10=np.concatenate(l2_relative10),
            l2_uend=np.concatenate(l2_uend),
        ),
        diagnostics,
    )


def _feature_map(features: FrozenFeatures) -> dict[str, np.ndarray]:
    return {
        "l1_whole_count_linear": features.l1_count,
        "l2_whole_count_linear": features.l2_count,
        "l2_fixed250_ordered_linear": features.l2_fixed250,
        "l2_relative10_ordered_linear": features.l2_relative10,
        "l2_uend_linear": features.l2_uend,
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
    loaders = make_loaders(data, spec, config.batch_size, train_shuffle=False)
    extracted = {
        split: extract_features(model, loader, device, data.fs)
        for split, loader in loaders.items()
    }
    features = {split: value[0] for split, value in extracted.items()}
    diagnostics = {split: value[1] for split, value in extracted.items()}

    probes: dict[str, Pipeline] = {}
    train_map = _feature_map(features["train"])
    for readout, matrix in train_map.items():
        probe = _scaled_linear_probe()
        probe.fit(matrix, features["train"].labels)
        probes[readout] = probe

    metrics: dict[str, dict[str, object]] = {}
    for split, feature in features.items():
        split_metrics: dict[str, object] = {
            "output_whole_count": exp3.metrics(
                feature.labels, feature.output_count.argmax(axis=1)
            )
        }
        for readout, matrix in _feature_map(feature).items():
            split_metrics[readout] = exp3.metrics(
                feature.labels,
                probes[readout].predict(matrix),
            )
        metrics[split] = split_metrics

    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_output_whole_count_ba": float(
            checkpoint["best_val_output_whole_count_ba"]
        ),
        "best_val_output_whole_count_loss": float(
            checkpoint["best_val_output_whole_count_loss"]
        ),
        "architecture": architecture_manifest(data.fs, len(data.labels)),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "probe_protocol": {
            "classifier": "train-only StandardScaler + balanced LogisticRegression(lbfgs)",
            "l1_whole_count_linear": "128D whole-valid-gesture L1 spike counts; no temporal position",
            "l2_whole_count_linear": "128D whole-valid-gesture L2 spike counts; no temporal position",
            "l2_fixed250_ordered_linear": "ordered Fixed250 L2 spike-count bins flattened; absolute coarse time visible",
            "l2_relative10_ordered_linear": "ordered Relative10 L2 spike-count bins flattened; relative phase visible",
            "l2_uend_linear": "128D L2 membrane state at the single valid endpoint timestep",
        },
        "provenance": {
            "split_seed": int(exp3.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "input_contract": "Exp3 Raw64 30-channel unsigned weighted events; no extra Exp4 Fixed250-fitted scaling",
            "checkpoint_selection": "maximum validation Output WholeCount BA; tie-break minimum validation normalized WholeCount CE",
            "relative_endpoint_use": "valid length is used only to construct/mask training loss and evaluation; never input to the SNN",
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


def architecture_manifest(fs: float, n_classes: int) -> dict[str, object]:
    return {
        "input_channels": exp3.EVENT_CHANNELS,
        "sampling_rate_hz": float(fs),
        "l1_width": L1_WIDTH,
        "l1_shifts": list(L1_SHIFTS),
        "l1_group_neurons": list(_group_counts(L1_WIDTH, L1_SHIFTS)),
        "l1_tau_syn_ms": [exp3.tau_ms(shift, fs) for shift in L1_SHIFTS],
        "l2_width": L2_WIDTH,
        "l2_shifts": list(L2_SHIFTS),
        "l2_group_neurons": list(_group_counts(L2_WIDTH, L2_SHIFTS)),
        "l2_tau_syn_ms": [exp3.tau_ms(shift, fs) for shift in L2_SHIFTS],
        "tau_mem_ms": TAU_MEM_MS,
        "threshold": THRESHOLD,
        "reset": RESET,
        "output_neurons": int(n_classes),
        "output_synaptic_state": False,
        "output_tau_mem_ms": TAU_MEM_MS,
        "logit_gain_training_only": LOGIT_GAIN,
    }


def _metric_ba(metrics: dict[str, object], readout: str) -> float:
    return float(metrics[readout]["balanced_accuracy"])  # type: ignore[index]


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing required Exp5.0 artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for split, split_metrics in payload["metrics"].items():
            row: dict[str, object] = {
                "objective": spec.objective,
                "variant": spec.variant,
                "hidden_cap": spec.hidden_cap,
                "output_cap": spec.output_cap,
                "seed": spec.seed,
                "split": split,
                "best_epoch": int(payload["best_epoch"]),
            }
            for readout in READOUTS:
                row[f"{readout}_ba"] = _metric_ba(split_metrics, readout)
            for layer in ("l1", "l2", "output"):
                layer_diag = payload["diagnostics"][split][layer]
                row[f"{layer}_events_per_neuron_second"] = float(
                    layer_diag["events_per_neuron_second"]
                )
                row[f"{layer}_tail_event_fraction"] = float(
                    layer_diag["tail_event_fraction"]
                )
            rows.append(row)

    runs = pd.DataFrame(rows)
    runs_file = root / "runs.csv"
    runs.to_csv(runs_file, index=False)

    test = runs[runs["split"] == "test"].copy()
    metric_columns = [
        *[f"{readout}_ba" for readout in READOUTS],
        "l1_events_per_neuron_second",
        "l2_events_per_neuron_second",
        "output_events_per_neuron_second",
        "l1_tail_event_fraction",
        "l2_tail_event_fraction",
        "output_tail_event_fraction",
    ]
    summary = (
        test.groupby(["objective", "variant"])[metric_columns]
        .agg(["mean", "std", "count"])
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

    indexed = test.set_index(["objective", "variant", "seed"])
    loss_effect_rows: list[dict[str, object]] = []
    for objective in OBJECTIVES:
        if objective == "whole_count_ce":
            continue
        for variant, _, _ in VARIANTS:
            for seed in SEEDS:
                for readout in READOUTS:
                    column = f"{readout}_ba"
                    candidate = float(indexed.loc[(objective, variant, seed), column])
                    baseline = float(indexed.loc[("whole_count_ce", variant, seed), column])
                    loss_effect_rows.append(
                        {
                            "objective": objective,
                            "baseline_objective": "whole_count_ce",
                            "variant": variant,
                            "seed": seed,
                            "readout": readout,
                            "delta_ba": candidate - baseline,
                        }
                    )
    paired_loss = pd.DataFrame(loss_effect_rows)
    paired_loss_file = root / "paired_loss_effects.csv"
    paired_loss.to_csv(paired_loss_file, index=False)
    paired_loss_summary = (
        paired_loss.groupby(["objective", "variant", "readout"])["delta_ba"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    paired_loss_summary_file = root / "paired_loss_effects_summary.csv"
    paired_loss_summary.to_csv(paired_loss_summary_file, index=False)

    variant_pairs = (
        ("multi_h", "binary"),
        ("multi_ho", "binary"),
        ("multi_ho", "multi_h"),
    )
    variant_rows: list[dict[str, object]] = []
    for objective in OBJECTIVES:
        for candidate_variant, baseline_variant in variant_pairs:
            for seed in SEEDS:
                for readout in READOUTS:
                    column = f"{readout}_ba"
                    candidate = float(indexed.loc[(objective, candidate_variant, seed), column])
                    baseline = float(indexed.loc[(objective, baseline_variant, seed), column])
                    variant_rows.append(
                        {
                            "objective": objective,
                            "candidate_variant": candidate_variant,
                            "baseline_variant": baseline_variant,
                            "seed": seed,
                            "readout": readout,
                            "delta_ba": candidate - baseline,
                        }
                    )
    paired_variant = pd.DataFrame(variant_rows)
    paired_variant_file = root / "paired_variant_effects.csv"
    paired_variant.to_csv(paired_variant_file, index=False)
    paired_variant_summary = (
        paired_variant.groupby(
            ["objective", "candidate_variant", "baseline_variant", "readout"]
        )["delta_ba"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    paired_variant_summary_file = root / "paired_variant_effects_summary.csv"
    paired_variant_summary.to_csv(paired_variant_summary_file, index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "How do local-supervision temporal scale and spike/event capacity interact in an "
            "Exp3-style local-evidence SNN when final inference is always Output WholeCount?"
        ),
        "run_count": len(run_specs()),
        "objectives": list(OBJECTIVES),
        "variants": [
            {"name": name, "hidden_cap": hidden_cap, "output_cap": output_cap}
            for name, hidden_cap, output_cap in VARIANTS
        ],
        "seeds": list(SEEDS),
        "architecture": architecture_manifest(exp3.EXPECTED_FS, len(exp3.LABELS)),
        "primary_metric": "test Output WholeCount balanced accuracy",
        "readouts": list(READOUTS),
        "checkpoint_selection": "validation Output WholeCount BA for every training objective",
        "files": {
            "runs": runs_file.name,
            "summary": summary_file.name,
            "paired_loss_effects": paired_loss_file.name,
            "paired_loss_effects_summary": paired_loss_summary_file.name,
            "paired_variant_effects": paired_variant_file.name,
            "paired_variant_effects_summary": paired_variant_summary_file.name,
        },
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    return {
        "runs": runs_file,
        "summary": summary_file,
        "paired_loss_effects": paired_loss_file,
        "paired_loss_effects_summary": paired_loss_summary_file,
        "paired_variant_effects": paired_variant_file,
        "paired_variant_effects_summary": paired_variant_summary_file,
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

    eval_parser = subparsers.add_parser("eval-one")
    eval_parser.add_argument("--objective", choices=OBJECTIVES, required=True)
    eval_parser.add_argument("--variant", choices=tuple(item[0] for item in VARIANTS), required=True)
    eval_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--threads", type=int, default=1)
    eval_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    eval_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("finalize")
    return parser


def _config_from_args(args: argparse.Namespace, repo_root: Path) -> Config:
    return Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=getattr(args, "device", "cpu"),
        epochs=getattr(args, "epochs", EPOCHS),
        batch_size=getattr(args, "batch_size", BATCH_SIZE),
        threads=getattr(args, "threads", 1),
    )


def _variant_caps(name: str) -> tuple[int, int]:
    for variant, hidden_cap, output_cap in VARIANTS:
        if variant == name:
            return hidden_cap, output_cap
    raise ValueError(f"Unknown variant: {name}")


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = exp3.find_repo_root()
    if args.command == "finalize":
        for name, path in finalize_experiment(repo_root).items():
            print(f"{name}: {path}")
        return

    data = prepare_data(repo_root)
    config = _config_from_args(args, repo_root)
    if args.command == "run-one":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(
                f"array task {args.array_task_id} outside [0, {len(specs) - 1}]"
            )
        spec = specs[args.array_task_id]
        payload = run_one(spec, data, config, args.force)
    else:
        hidden_cap, output_cap = _variant_caps(args.variant)
        spec = RunSpec(args.objective, args.variant, hidden_cap, output_cap, args.seed)
        payload = evaluate_one(spec, data, config, args.force)

    test = payload["metrics"]["test"]
    print(
        f"completed {spec.key}: "
        f"output_count_BA={test['output_whole_count']['balanced_accuracy']:.6f} "
        f"l2_count_linear_BA={test['l2_whole_count_linear']['balanced_accuracy']:.6f} "
        f"l2_fixed250_linear_BA={test['l2_fixed250_ordered_linear']['balanced_accuracy']:.6f} "
        f"l2_relative10_linear_BA={test['l2_relative10_ordered_linear']['balanced_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
