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
import snntorch as snn
from snntorch import surrogate

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_0_3_l3_bottleneck_ablation as selected


EXPERIMENT_ID = "experiment_3_4_causal_fixed250_objectives"
PROTOCOL_VERSION = "causal_prefix_v1"

OBJECTIVES = (
    "timestep_ce",
    "whole_fixed250_ce",
    "causal_fixed250_prefix_ce",
)
SEEDS = base.SEEDS
EXPECTED_RUNS = len(OBJECTIVES) * len(SEEDS)

SOURCE_ARCHITECTURE = "B"
L1_WIDTH = selected.L1_WIDTH
L2_WIDTH = selected.L2_WIDTH
L1_SHIFTS = selected.L1_SHIFTS
L2_SHIFTS = selected.L2_SHIFTS
EVENT_CHANNELS = base.EVENT_CHANNELS
FIXED_MS = base.FIXED_MS
PREFIX_WEIGHT = 0.5
FINAL_WEIGHT = 0.5
EPOCHS = base.EPOCHS
BATCH_SIZE = base.BATCH_SIZE
LR = base.LR


@dataclass(frozen=True)
class RunSpec:
    objective: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.objective}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    resume: bool = True
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "snn").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate writingRing repository root")


def results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / EXPERIMENT_ID
        / PROTOCOL_VERSION
    )


def run_specs() -> list[RunSpec]:
    return [RunSpec(objective, seed) for objective in OBJECTIVES for seed in SEEDS]


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def run_artifact_path(root: Path, spec: RunSpec) -> Path:
    return root / "runs" / f"{spec.key}.json"


def _alpha_vector(width: int, shifts: tuple[int, ...]) -> torch.Tensor:
    values = torch.empty(width, dtype=torch.float32)
    for shift, sl in selected.balanced_group_slices(width, shifts).items():
        values[sl] = base.alpha(int(shift))
    return values


def valid_bin_counts(lengths: torch.Tensor, bin_steps: int, n_bins: int) -> torch.Tensor:
    """Number of non-empty Fixed250 bins, retaining the partial final bin."""
    return torch.div(
        lengths.clamp_min(1) + bin_steps - 1,
        bin_steps,
        rounding_mode="floor",
    ).clamp(min=1, max=n_bins)


class CausalFixed250Net(nn.Module):
    """Selected two-layer Phase-C SNN with objective-specific linear heads."""

    def __init__(
        self,
        objective: str,
        n_classes: int,
        T: int,
        fs: float,
        bin_steps: int,
    ) -> None:
        super().__init__()
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective: {objective}")
        self.objective = objective
        self.n_classes = int(n_classes)
        self.T = int(T)
        self.bin_steps = int(bin_steps)
        self.n_bins = int(math.ceil(T / bin_steps))

        beta = math.exp(-(1000.0 / fs) / base.TAU_MEM_MS)
        spike_grad = surrogate.fast_sigmoid(slope=base.SURROGATE_SLOPE)

        def make_lif(width: int, shifts: tuple[int, ...]) -> snn.Synaptic:
            return snn.Synaptic(
                alpha=_alpha_vector(width, shifts),
                beta=beta,
                threshold=base.THRESHOLD,
                spike_grad=spike_grad,
                reset_mechanism=base.RESET,
            )

        self.f1 = nn.Linear(EVENT_CHANNELS, L1_WIDTH, bias=False)
        self.l1 = make_lif(L1_WIDTH, L1_SHIFTS)
        self.f2 = nn.Linear(L1_WIDTH, L2_WIDTH, bias=False)
        self.l2 = make_lif(L2_WIDTH, L2_SHIFTS)

        if objective == "timestep_ce":
            head_dim = L2_WIDTH
        else:
            head_dim = L2_WIDTH * self.n_bins
        self.head = nn.Linear(head_dim, n_classes, bias=True)

    def layer_features(self, x: torch.Tensor) -> torch.Tensor:
        batch, T, _ = x.shape
        syn1 = torch.zeros(batch, L1_WIDTH, device=x.device, dtype=x.dtype)
        mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, L2_WIDTH, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(syn2)
        sequence: list[torch.Tensor] = []
        for timestep in range(T):
            s1, syn1, mem1 = self.l1(self.f1(x[:, timestep]), syn1, mem1)
            s2, syn2, mem2 = self.l2(self.f2(s1), syn2, mem2)
            sequence.append(s2)
        return torch.stack(sequence, dim=1)

    def fixed_counts(self, spikes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return base.fixed_counts(spikes, lengths, self.bin_steps)

    def fixed_prefix_logits_from_counts(self, counts: torch.Tensor) -> torch.Tensor:
        if self.objective == "timestep_ce":
            raise ValueError("Fixed-bin prefix logits require a Fixed250 objective")
        if counts.shape[1:] != (self.n_bins, L2_WIDTH):
            raise ValueError(
                f"Expected counts [B,{self.n_bins},{L2_WIDTH}], got {tuple(counts.shape)}"
            )
        weights = self.head.weight.view(self.n_classes, self.n_bins, L2_WIDTH)
        per_bin = torch.einsum("bkd,ckd->bkc", counts, weights)
        return per_bin.cumsum(dim=1) + self.head.bias[None, None, :]

    def fixed_prefix_logits(
        self,
        spikes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = self.fixed_counts(spikes, lengths)
        return self.fixed_prefix_logits_from_counts(counts), counts

    def loss_and_final_logits(
        self,
        spikes: torch.Tensor,
        lengths: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        batch, T, _ = spikes.shape
        if self.objective == "timestep_ce":
            logits_t = self.head(spikes)
            valid = base.mask(lengths, T)
            targets = y[:, None].expand(batch, T)
            loss = F.cross_entropy(logits_t[valid], targets[valid])
            weights = valid.to(logits_t.dtype).unsqueeze(-1)
            final_logits = (logits_t * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            return loss, final_logits, {
                "prefix_loss": float("nan"),
                "final_loss": float("nan"),
            }

        prefix_logits, counts = self.fixed_prefix_logits(spikes, lengths)
        if self.objective == "whole_fixed250_ce":
            logits = self.head(counts.flatten(start_dim=1))
            loss = F.cross_entropy(logits, y)
            return loss, logits, {
                "prefix_loss": float("nan"),
                "final_loss": float(loss.detach().item()),
            }

        valid_bins = valid_bin_counts(lengths, self.bin_steps, self.n_bins)
        final_index = valid_bins - 1
        batch_index = torch.arange(batch, device=spikes.device)
        final_logits = prefix_logits[batch_index, final_index]
        final_ce = F.cross_entropy(final_logits, y, reduction="none")

        targets = y[:, None].expand(batch, self.n_bins)
        prefix_ce = F.cross_entropy(
            prefix_logits.reshape(-1, self.n_classes),
            targets.reshape(-1),
            reduction="none",
        ).reshape(batch, self.n_bins)
        positions = torch.arange(self.n_bins, device=spikes.device)[None, :]
        prefix_mask = positions < final_index[:, None]
        prefix_count = prefix_mask.sum(dim=1)
        prefix_ce_per_sample = (
            (prefix_ce * prefix_mask.to(prefix_ce.dtype)).sum(dim=1)
            / prefix_count.clamp_min(1).to(prefix_ce.dtype)
        )
        has_prefix = prefix_count > 0
        loss_per_sample = torch.where(
            has_prefix,
            PREFIX_WEIGHT * prefix_ce_per_sample + FINAL_WEIGHT * final_ce,
            final_ce,
        )
        loss = loss_per_sample.mean()

        prefix_mean = (
            prefix_ce_per_sample[has_prefix].mean()
            if bool(has_prefix.any())
            else torch.full((), float("nan"), device=spikes.device)
        )
        return loss, final_logits, {
            "prefix_loss": float(prefix_mean.detach().item()),
            "final_loss": float(final_ce.mean().detach().item()),
        }

    def periodic_prefix_logits(
        self,
        spikes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Logits at 250, 500, ... ms without using future samples."""
        if self.objective != "timestep_ce":
            logits, _ = self.fixed_prefix_logits(spikes, lengths)
            return logits

        logits_t = self.head(spikes)
        cumulative = logits_t.cumsum(dim=1)
        outputs: list[torch.Tensor] = []
        for k in range(1, self.n_bins + 1):
            end = min(k * self.bin_steps, logits_t.shape[1])
            outputs.append(cumulative[:, end - 1] / float(end))
        return torch.stack(outputs, dim=1)

    def parameter_counts(self) -> dict[str, int]:
        head = sum(parameter.numel() for parameter in self.head.parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "head": int(head),
            "backbone": int(total - head),
            "total": int(total),
        }


def build_initialized_model(
    objective: str,
    n_classes: int,
    T: int,
    fs: float,
    bin_steps: int,
    seed: int,
) -> CausalFixed250Net:
    """Paired init: identical backbone across objectives; identical Fixed250 heads."""
    base.seed_all(base.dseed(seed, "exp34_shared_backbone_init"))
    model = CausalFixed250Net(objective, n_classes, T, fs, bin_steps)
    if objective == "timestep_ce":
        head_seed = base.dseed(seed, "exp34_timestep_head_init")
    else:
        head_seed = base.dseed(seed, "exp34_fixed250_head_init")
    base.seed_all(head_seed)
    model.head.reset_parameters()
    return model


def _metrics_with_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: float,
) -> dict[str, float]:
    out = base.metrics(y_true, y_pred)
    out["loss"] = float(loss)
    return out


def evaluate_final(
    model: CausalFixed250Net,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for X, y, lengths in data_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            spikes = model.layer_features(X)
            loss, logits, _ = model.loss_and_final_logits(spikes, lengths, y)
            n = len(y)
            loss_sum += float(loss.item()) * n
            n_total += n
            y_true_parts.append(y.cpu().numpy())
            y_pred_parts.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    return _metrics_with_loss(y_true, y_pred, loss_sum / max(n_total, 1))


def evaluate_prefix_curve(
    model: CausalFixed250Net,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    fs: float,
) -> list[dict[str, float | int]]:
    model.eval()
    labels: list[list[np.ndarray]] = [[] for _ in range(model.n_bins)]
    predictions: list[list[np.ndarray]] = [[] for _ in range(model.n_bins)]
    with torch.no_grad():
        for X, y, lengths in data_loader:
            X = X.to(device)
            lengths_device = lengths.to(device)
            spikes = model.layer_features(X)
            prefix_logits = model.periodic_prefix_logits(spikes, lengths_device)
            for index in range(model.n_bins):
                end_step = (index + 1) * model.bin_steps
                keep = lengths >= end_step
                if not bool(keep.any()):
                    continue
                labels[index].append(y[keep].numpy())
                predictions[index].append(
                    prefix_logits[keep.to(device), index]
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )

    rows: list[dict[str, float | int]] = []
    for index, (y_parts, pred_parts) in enumerate(zip(labels, predictions, strict=True)):
        if not y_parts:
            continue
        y_true = np.concatenate(y_parts)
        y_pred = np.concatenate(pred_parts)
        metric = base.metrics(y_true, y_pred)
        rows.append(
            {
                "prefix_bin": index + 1,
                "observed_ms": float((index + 1) * model.bin_steps * 1000.0 / fs),
                "n_samples": int(len(y_true)),
                "balanced_accuracy": float(metric["balanced_accuracy"]),
                "accuracy": float(metric["accuracy"]),
                "macro_f1": float(metric["macro_f1"]),
            }
        )
    return rows


def _provenance(spec: RunSpec, data: base.Data, config: Config) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "objective": spec.objective,
        "seed": int(spec.seed),
        "split_seed": int(base.SPLIT_SEED),
        "source_architecture": SOURCE_ARCHITECTURE,
        "l1_width": int(L1_WIDTH),
        "l2_width": int(L2_WIDTH),
        "l1_shifts": list(L1_SHIFTS),
        "l2_shifts": list(L2_SHIFTS),
        "sampling_rate_hz": float(data.fs),
        "padded_length": int(data.T),
        "fixed_bin_ms": float(FIXED_MS),
        "fixed_bin_steps": int(data.bin_steps),
        "n_bins": int(data.n_bins),
        "event_channels": int(EVENT_CHANNELS),
        "tau_mem_ms": float(base.TAU_MEM_MS),
        "threshold": float(base.THRESHOLD),
        "reset": base.RESET,
        "prefix_weight": float(PREFIX_WEIGHT),
        "final_weight": float(FINAL_WEIGHT),
        "partial_final_bin": "retained with samples beyond valid_length masked to zero",
        "checkpoint_selection": "highest validation final BA; tie lower validation objective loss; tie earlier epoch",
        "epochs": int(config.epochs),
        "batch_size": int(config.batch_size),
        "learning_rate": float(LR),
    }


def train_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[CausalFixed250Net, dict[str, object]]:
    if spec.objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {spec.objective}")
    if spec.seed not in SEEDS:
        raise ValueError(f"Unknown seed: {spec.seed}")

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model = build_initialized_model(
        spec.objective,
        len(data.labels),
        data.T,
        data.fs,
        data.bin_steps,
        spec.seed,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = base.loader(
        data.Xtr,
        data.ytr,
        data.ltr,
        config.batch_size,
        True,
        base.dseed(spec.seed, "exp34_train_loader"),
    )
    val_loader = base.loader(
        data.Xva,
        data.yva,
        data.lva,
        config.batch_size,
        False,
        base.dseed(spec.seed, "exp34_val_loader"),
    )

    best_val_ba = -np.inf
    best_val_loss = np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        y_true_parts: list[np.ndarray] = []
        y_pred_parts: list[np.ndarray] = []
        loss_sum = 0.0
        n_total = 0
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            spikes = model.layer_features(X)
            loss, final_logits, _ = model.loss_and_final_logits(spikes, lengths, y)
            loss.backward()
            optimizer.step()
            n = len(y)
            loss_sum += float(loss.item()) * n
            n_total += n
            y_true_parts.append(y.detach().cpu().numpy())
            y_pred_parts.append(final_logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = base.metrics(np.concatenate(y_true_parts), np.concatenate(y_pred_parts))
        train_loss = loss_sum / max(n_total, 1)
        val_metrics = evaluate_final(model, val_loader, device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
                "val_loss": val_loss,
                "val_balanced_accuracy": val_ba,
            }
        )
        improved = (
            val_ba > best_val_ba + 1e-12
            or (
                abs(val_ba - best_val_ba) <= 1e-12
                and val_loss < best_val_loss - 1e-12
            )
        )
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No best checkpoint selected")
    model.load_state_dict(best_state)
    result = {
        "best_epoch": int(best_epoch),
        "history": history,
    }
    return model, result


def run_one(spec: RunSpec, data: base.Data, config: Config) -> dict[str, object]:
    artifact = run_artifact_path(config.results_dir, spec)
    checkpoint = checkpoint_path(config.results_dir, spec)
    if config.resume and artifact.exists() and checkpoint.exists():
        with artifact.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        identity = (
            cached.get("experiment_id"),
            cached.get("protocol_version"),
            cached.get("objective"),
            cached.get("seed"),
        )
        expected = (EXPERIMENT_ID, PROTOCOL_VERSION, spec.objective, spec.seed)
        if identity != expected:
            raise ValueError(f"Cached run identity mismatch: {identity} != {expected}")
        return cached

    model, training = train_one(spec, data, config)
    device = torch.device(config.device)
    loaders = {
        "train": base.loader(data.Xtr, data.ytr, data.ltr, config.batch_size, False, 0),
        "val": base.loader(data.Xva, data.yva, data.lva, config.batch_size, False, 0),
        "test": base.loader(data.Xte, data.yte, data.lte, config.batch_size, False, 0),
    }
    final_metrics = {
        split: evaluate_final(model, loader, device)
        for split, loader in loaders.items()
    }
    prefix_curve = evaluate_prefix_curve(model, loaders["test"], device, data.fs)
    provenance = _provenance(spec, data, config)
    result: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "objective": spec.objective,
        "seed": int(spec.seed),
        "provenance": provenance,
        "parameter_counts": model.parameter_counts(),
        "best_epoch": training["best_epoch"],
        "history": training["history"],
        "final_metrics": final_metrics,
        "prefix_curve": prefix_curve,
    }

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "provenance": provenance,
            "state_dict": model.state_dict(),
            "result": result,
        },
        checkpoint,
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def _result_row(payload: dict[str, object]) -> dict[str, object]:
    metrics = payload["final_metrics"]
    params = payload["parameter_counts"]
    row: dict[str, object] = {
        "objective": payload["objective"],
        "seed": int(payload["seed"]),
        "best_epoch": int(payload["best_epoch"]),
        "head_parameters": int(params["head"]),
        "backbone_parameters": int(params["backbone"]),
        "total_parameters": int(params["total"]),
    }
    for split in ("train", "val", "test"):
        split_metrics = metrics[split]
        for name in ("loss", "accuracy", "balanced_accuracy", "macro_f1"):
            row[f"{split}_{name}"] = float(split_metrics[name])
    return row


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for objective, group in frame.groupby("objective", sort=False):
        rows.append(
            {
                "objective": objective,
                "n_seeds": int(len(group)),
                "mean_test_ba": float(group.test_balanced_accuracy.mean()),
                "sd_test_ba": float(group.test_balanced_accuracy.std(ddof=1)),
                "mean_test_accuracy": float(group.test_accuracy.mean()),
                "sd_test_accuracy": float(group.test_accuracy.std(ddof=1)),
                "mean_test_macro_f1": float(group.test_macro_f1.mean()),
                "sd_test_macro_f1": float(group.test_macro_f1.std(ddof=1)),
                "mean_best_epoch": float(group.best_epoch.mean()),
                "head_parameters": int(group.head_parameters.iloc[0]),
                "total_parameters": int(group.total_parameters.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _paired_final(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot(index="seed", columns="objective", values="test_balanced_accuracy")
    required = set(OBJECTIVES)
    if set(pivot.columns) != required:
        raise ValueError(f"Missing paired objectives: {set(pivot.columns)} != {required}")
    rows: list[dict[str, object]] = []
    for seed, record in pivot.iterrows():
        causal = float(record["causal_fixed250_prefix_ce"])
        whole = float(record["whole_fixed250_ce"])
        timestep = float(record["timestep_ce"])
        rows.append(
            {
                "seed": int(seed),
                "timestep_test_ba": timestep,
                "whole_fixed250_test_ba": whole,
                "causal_fixed250_test_ba": causal,
                "causal_minus_whole_ba": causal - whole,
                "causal_minus_timestep_ba": causal - timestep,
            }
        )
    return pd.DataFrame(rows)


def _prefix_summary(prefix: pd.DataFrame) -> pd.DataFrame:
    return (
        prefix.groupby(["objective", "observed_ms"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_ba=("balanced_accuracy", "mean"),
            sd_ba=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            min_n_samples=("n_samples", "min"),
        )
        .reset_index()
    )


def _streaming_delta(prefix: pd.DataFrame) -> float:
    selected_times = prefix[(prefix.observed_ms >= 500.0) & (prefix.observed_ms <= 1500.0)]
    if selected_times.empty:
        return float("nan")
    pivot = selected_times.pivot_table(
        index=["seed", "observed_ms"],
        columns="objective",
        values="balanced_accuracy",
        aggfunc="first",
    ).dropna(subset=["whole_fixed250_ce", "causal_fixed250_prefix_ce"])
    if pivot.empty:
        return float("nan")
    return float(
        (pivot["causal_fixed250_prefix_ce"] - pivot["whole_fixed250_ce"]).mean()
    )


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    result_rows: list[dict[str, object]] = []
    prefix_rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = run_artifact_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing Experiment 3.4 run artifact: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        identity = (
            payload.get("experiment_id"),
            payload.get("protocol_version"),
            payload.get("objective"),
            payload.get("seed"),
        )
        expected = (EXPERIMENT_ID, PROTOCOL_VERSION, spec.objective, spec.seed)
        if identity != expected:
            raise ValueError(f"Run identity mismatch in {path}: {identity} != {expected}")
        result_rows.append(_result_row(payload))
        for record in payload["prefix_curve"]:
            prefix_rows.append(
                {
                    "objective": spec.objective,
                    "seed": int(spec.seed),
                    **record,
                }
            )

    results = pd.DataFrame(result_rows)
    prefix = pd.DataFrame(prefix_rows)
    if len(results) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} runs, got {len(results)}")

    fixed = results[results.objective.isin(("whole_fixed250_ce", "causal_fixed250_prefix_ce"))]
    if fixed.groupby("objective").head_parameters.nunique().max() != 1:
        raise ValueError("Fixed250 head parameter count changed across seeds")
    fixed_counts = fixed.groupby("objective").head_parameters.first()
    if int(fixed_counts["whole_fixed250_ce"]) != int(fixed_counts["causal_fixed250_prefix_ce"]):
        raise ValueError("Whole250 and Causal250 must be parameter matched")

    summary = _summary(results)
    paired = _paired_final(results)
    prefix_summary = _prefix_summary(prefix)
    mean_delta = float(paired.causal_minus_whole_ba.mean())
    positive_seeds = int((paired.causal_minus_whole_ba > 0.0).sum())
    criterion_1 = mean_delta > 0.0
    criterion_2 = positive_seeds >= 2
    conclusion = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "primary_question": (
            "Does causal Fixed250 prefix supervision improve final balanced accuracy "
            "over parameter-matched whole-segment Fixed250 supervision?"
        ),
        "hypothesis": (
            "Progressive causal-prefix supervision will improve fixed-duration SNN "
            "classification by aligning training with streaming evidence accumulation."
        ),
        "validation_standard": {
            "criterion_1": "mean(Causal250 final test BA - Whole250 final test BA) > 0",
            "criterion_2": "Causal250 final test BA > Whole250 in at least 2 of 3 paired seeds",
            "decision_rule": "both criteria must pass",
        },
        "criterion_1_mean_final_ba_gain_positive": bool(criterion_1),
        "criterion_2_at_least_two_seeds_positive": bool(criterion_2),
        "primary_hypothesis_supported": bool(criterion_1 and criterion_2),
        "mean_causal_minus_whole_final_ba": mean_delta,
        "positive_final_ba_seeds": positive_seeds,
        "mean_streaming_ba_delta_500_to_1500ms": _streaming_delta(prefix),
        "partial_final_bin_policy": (
            "The final non-empty Fixed250 bin is retained even when partial; samples "
            "after valid_length are masked to zero. It contributes to final endpoint "
            "prediction only, not to the non-final prefix-loss average."
        ),
        "causality_guardrail": (
            "No relative-progress or final-duration normalization is used to form "
            "periodic prefix predictions."
        ),
    }

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results": root / "experiment_3_4_results.csv",
        "summary": root / "experiment_3_4_summary.csv",
        "paired": root / "experiment_3_4_paired_final.csv",
        "prefix_curve": root / "experiment_3_4_prefix_curve.csv",
        "prefix_summary": root / "experiment_3_4_prefix_summary.csv",
        "conclusion": root / "experiment_3_4_conclusion.json",
        "provenance": root / "provenance.json",
    }
    results.to_csv(outputs["results"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    paired.to_csv(outputs["paired"], index=False)
    prefix.to_csv(outputs["prefix_curve"], index=False)
    prefix_summary.to_csv(outputs["prefix_summary"], index=False)
    with outputs["conclusion"].open("w", encoding="utf-8") as handle:
        json.dump(conclusion, handle, indent=2, sort_keys=True)
    with outputs["provenance"].open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "objectives": list(OBJECTIVES),
                "seeds": list(SEEDS),
                "expected_runs": EXPECTED_RUNS,
                "split_seed": int(base.SPLIT_SEED),
                "source_architecture": SOURCE_ARCHITECTURE,
                "architecture": {
                    "input_channels": EVENT_CHANNELS,
                    "l1_width": L1_WIDTH,
                    "l1_shifts": list(L1_SHIFTS),
                    "l2_width": L2_WIDTH,
                    "l2_shifts": list(L2_SHIFTS),
                    "l3": None,
                },
                "fixed_bin_ms": FIXED_MS,
                "prefix_weight": PREFIX_WEIGHT,
                "final_weight": FINAL_WEIGHT,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return outputs


def _run_cli(args: argparse.Namespace) -> None:
    repo_root = find_repo_root()
    root = results_dir(repo_root)
    if args.command == "run-one":
        specs = run_specs()
        task_id = int(args.array_task_id)
        if task_id < 0 or task_id >= len(specs):
            raise ValueError(f"array task id {task_id} outside 0..{len(specs) - 1}")
        data = base.prepare_data(repo_root)
        config = Config(
            repo_root=repo_root,
            results_dir=root,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume=not args.force,
            threads=1,
        )
        spec = specs[task_id]
        result = run_one(spec, data, config)
        test_ba = result["final_metrics"]["test"]["balanced_accuracy"]
        print(f"completed {spec.key} test_BA={test_ba:.4f}")
        return
    if args.command == "finalize":
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return
    raise ValueError(args.command)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 3.4 causal Fixed250 objective comparison"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--epochs", type=int, default=EPOCHS)
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run.add_argument("--force", action="store_true")
    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    _run_cli(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
