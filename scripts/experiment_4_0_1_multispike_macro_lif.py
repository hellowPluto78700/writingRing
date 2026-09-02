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

from scripts import experiment_4_0_fixed250_temporal_snn as exp40


EXPERIMENT_ID = "experiment_4_0_1_multispike_macro_lif"
PROTOCOL_VERSION = "macro_lif_event_cap_factorial_v1"

SEEDS = exp40.SEEDS
ANCHORS = (
    ("ff", 128, 2000.0),
    ("rsnn", 128, 250.0),
)
VARIANTS: tuple[tuple[str, int, int], ...] = (
    ("binary", 1, 1),
    ("multi_h", 31, 1),
    ("multi_o", 1, 31),
    ("multi_ho", 31, 31),
)
EXPECTED_RUNS = len(ANCHORS) * len(VARIANTS) * len(SEEDS)

SURROGATE_SLOPE = exp40.SURROGATE_SLOPE
THRESHOLD = exp40.THRESHOLD
BATCH_SIZE = exp40.BATCH_SIZE
EPOCHS = exp40.EPOCHS
LR = exp40.LR
WEIGHT_DECAY = exp40.WEIGHT_DECAY
EPS = exp40.EPS


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    hidden_width: int
    tau_mem_ms: float
    variant: str
    hidden_cap: int
    output_cap: int
    seed: int

    @property
    def key(self) -> str:
        tau_tag = str(int(self.tau_mem_ms))
        return (
            f"{self.architecture}__h{self.hidden_width}__tau{tau_tag}ms__"
            f"{self.variant}__hcap{self.hidden_cap}__ocap{self.output_cap}__seed{self.seed}"
        )


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
    specs: list[RunSpec] = []
    for architecture, hidden_width, tau_mem_ms in ANCHORS:
        for variant, hidden_cap, output_cap in VARIANTS:
            for seed in SEEDS:
                specs.append(
                    RunSpec(
                        architecture=architecture,
                        hidden_width=hidden_width,
                        tau_mem_ms=tau_mem_ms,
                        variant=variant,
                        hidden_cap=hidden_cap,
                        output_cap=output_cap,
                        seed=seed,
                    )
                )
    return specs


def paired_seed(spec: RunSpec, role: str) -> int:
    """Use identical random streams across cap variants in each paired group."""
    return exp40.base.dseed(
        spec.seed,
        "exp4_0_1_paired",
        spec.architecture,
        spec.hidden_width,
        spec.tau_mem_ms,
        role,
    )


class _MultiThresholdSpike(torch.autograd.Function):
    """Integer event count in forward, multi-threshold surrogate in backward.

    Forward implements a capped subtractive-reset-compatible event count:

        s = clip(floor(max(v, 0) / threshold), 0, max_spikes_per_dt)

    Backward treats the count as a sum of threshold crossings and adds one
    fast-sigmoid surrogate derivative around each reachable threshold. The
    cap=1 and cap=31 cases therefore share exactly the same implementation;
    only the number of communicable threshold crossings changes.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        membrane: torch.Tensor,
        threshold: float,
        max_spikes_per_dt: int,
        slope: float,
    ) -> torch.Tensor:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if max_spikes_per_dt < 1:
            raise ValueError("max_spikes_per_dt must be >= 1")
        ctx.save_for_backward(membrane)
        ctx.threshold = float(threshold)
        ctx.max_spikes_per_dt = int(max_spikes_per_dt)
        ctx.slope = float(slope)
        positive = membrane.clamp_min(0.0)
        counts = torch.floor(positive / float(threshold))
        return counts.clamp(max=float(max_spikes_per_dt))

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        (membrane,) = ctx.saved_tensors
        threshold = float(ctx.threshold)
        slope = float(ctx.slope)
        max_spikes = int(ctx.max_spikes_per_dt)
        grad = torch.zeros_like(membrane)
        for crossing in range(1, max_spikes + 1):
            normalized_distance = (membrane - crossing * threshold) / threshold
            grad = grad + 1.0 / (1.0 + slope * normalized_distance.abs()).pow(2)
        grad = grad / threshold
        return grad_output * grad, None, None, None


def multi_threshold_spike(
    membrane: torch.Tensor,
    threshold: float,
    max_spikes_per_dt: int,
    slope: float = SURROGATE_SLOPE,
) -> torch.Tensor:
    return _MultiThresholdSpike.apply(
        membrane,
        float(threshold),
        int(max_spikes_per_dt),
        float(slope),
    )


class MacroMultiSpikeLIF(nn.Module):
    """Leaky macro-step LIF with capped multi-event communication.

    Every call performs the same update order for cap=1 and cap=31:

        pre_reset = beta * membrane + current
        spikes = capped multi-threshold event count(pre_reset)
        membrane = pre_reset - spikes * threshold

    This avoids comparing snnTorch's default reset timing against a custom
    multi-spike implementation. The experiment changes only the event cap.
    """

    def __init__(
        self,
        beta: float,
        threshold: float,
        max_spikes_per_dt: int,
        surrogate_slope: float = SURROGATE_SLOPE,
    ) -> None:
        super().__init__()
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if max_spikes_per_dt < 1:
            raise ValueError("max_spikes_per_dt must be >= 1")
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.max_spikes_per_dt = int(max_spikes_per_dt)
        self.surrogate_slope = float(surrogate_slope)

    def forward(
        self,
        current: torch.Tensor,
        membrane: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_reset = self.beta * membrane + current
        spikes = multi_threshold_spike(
            pre_reset,
            self.threshold,
            self.max_spikes_per_dt,
            self.surrogate_slope,
        )
        membrane = pre_reset - spikes * self.threshold
        return spikes, membrane, pre_reset


class MacroTemporalDecoder(nn.Module):
    """Exp4.0 topology with one controlled change: event cap per macro step."""

    def __init__(
        self,
        architecture: str,
        hidden_width: int,
        tau_mem_ms: float,
        n_classes: int,
        hidden_cap: int,
        output_cap: int,
    ) -> None:
        super().__init__()
        if architecture not in exp40.ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.architecture = architecture
        self.hidden_width = int(hidden_width)
        self.tau_mem_ms = float(tau_mem_ms)
        self.n_classes = int(n_classes)
        self.hidden_cap = int(hidden_cap)
        self.output_cap = int(output_cap)

        beta_hidden = math.exp(-exp40.FIXED_MS / self.tau_mem_ms)
        beta_output = math.exp(-exp40.FIXED_MS / exp40.OUTPUT_TAU_MEM_MS)
        self.input_hidden = nn.Linear(exp40.EVENT_CHANNELS, hidden_width, bias=False)
        self.recurrent = (
            nn.Linear(hidden_width, hidden_width, bias=False)
            if architecture == "rsnn"
            else None
        )
        self.hidden_lif = MacroMultiSpikeLIF(
            beta=beta_hidden,
            threshold=THRESHOLD,
            max_spikes_per_dt=hidden_cap,
        )
        self.hidden_output = nn.Linear(hidden_width, n_classes, bias=False)
        self.output_lif = MacroMultiSpikeLIF(
            beta=beta_output,
            threshold=THRESHOLD,
            max_spikes_per_dt=output_cap,
        )

    def forward_trajectory(
        self,
        x: torch.Tensor,
        collect_pre_reset: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        batch, n_bins, _ = x.shape
        hidden_mem = torch.zeros(
            batch, self.hidden_width, device=x.device, dtype=x.dtype
        )
        output_mem = torch.zeros(
            batch, self.n_classes, device=x.device, dtype=x.dtype
        )
        prev_hidden_spikes = torch.zeros_like(hidden_mem)
        hidden_spikes: list[torch.Tensor] = []
        output_spikes: list[torch.Tensor] = []
        output_mems: list[torch.Tensor] = []
        hidden_pre: list[torch.Tensor] = []
        output_pre: list[torch.Tensor] = []

        for b in range(n_bins):
            current = self.input_hidden(x[:, b])
            if self.recurrent is not None:
                current = current + self.recurrent(prev_hidden_spikes)
            h_spikes, hidden_mem, h_pre = self.hidden_lif(current, hidden_mem)
            output_current = self.hidden_output(h_spikes)
            o_spikes, output_mem, o_pre = self.output_lif(output_current, output_mem)
            hidden_spikes.append(h_spikes)
            output_spikes.append(o_spikes)
            output_mems.append(output_mem)
            if collect_pre_reset:
                hidden_pre.append(h_pre)
                output_pre.append(o_pre)
            prev_hidden_spikes = h_spikes

        return (
            torch.stack(output_spikes, dim=1),
            torch.stack(output_mems, dim=1),
            torch.stack(hidden_spikes, dim=1),
            torch.stack(hidden_pre, dim=1) if collect_pre_reset else None,
            torch.stack(output_pre, dim=1) if collect_pre_reset else None,
        )


def normalize_output_spikes(spikes: torch.Tensor, output_cap: int) -> torch.Tensor:
    if output_cap < 1:
        raise ValueError("output_cap must be >= 1")
    return spikes / float(output_cap)


def valid_evidence(
    output_spikes: torch.Tensor,
    valid_bins: torch.Tensor,
    output_cap: int,
) -> torch.Tensor:
    return exp40.valid_whole_count(
        normalize_output_spikes(output_spikes, output_cap),
        valid_bins,
    )


def full_evidence(output_spikes: torch.Tensor, output_cap: int) -> torch.Tensor:
    return exp40.full_whole_count(normalize_output_spikes(output_spikes, output_cap))


def _valid_mask(valid_bins: torch.Tensor, n_bins: int, width: int) -> torch.Tensor:
    return exp40.macro_mask(valid_bins, n_bins).unsqueeze(-1).expand(-1, -1, width)


def _distribution_stats(
    spikes: torch.Tensor,
    pre_reset: torch.Tensor,
    valid_bins: torch.Tensor,
    cap: int,
) -> dict[str, float]:
    mask = _valid_mask(valid_bins, spikes.shape[1], spikes.shape[2])
    values = spikes[mask]
    pre_values = pre_reset[mask]
    if values.numel() == 0:
        return {
            "mean_events_per_neuron_step": 0.0,
            "fraction_zero": 1.0,
            "fraction_one": 0.0,
            "fraction_gt1": 0.0,
            "fraction_ge4": 0.0,
            "fraction_at_cap": 0.0,
            "pre_reset_membrane_mean": 0.0,
            "pre_reset_membrane_max": 0.0,
        }
    return {
        "mean_events_per_neuron_step": float(values.float().mean().item()),
        "fraction_zero": float((values == 0).float().mean().item()),
        "fraction_one": float((values == 1).float().mean().item()),
        "fraction_gt1": float((values > 1).float().mean().item()),
        "fraction_ge4": float((values >= 4).float().mean().item()),
        "fraction_at_cap": float((values >= cap).float().mean().item()),
        "pre_reset_membrane_mean": float(pre_values.float().mean().item()),
        "pre_reset_membrane_max": float(pre_values.float().max().item()),
    }


def _tail_fraction(spikes: torch.Tensor, valid_bins: torch.Tensor) -> float:
    valid = exp40.valid_whole_count(spikes, valid_bins).sum()
    full = exp40.full_whole_count(spikes).sum()
    tail = full - valid
    return float((tail / full.clamp_min(EPS)).item())


def evaluate_model(
    model: MacroTemporalDecoder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    valid_pred_parts: list[np.ndarray] = []
    full_pred_parts: list[np.ndarray] = []
    valid_mem_pred_parts: list[np.ndarray] = []
    full_mem_pred_parts: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    hidden_stat_sums: dict[str, float] = {}
    output_stat_sums: dict[str, float] = {}
    stat_weight = 0
    hidden_tail_weighted = 0.0
    output_tail_weighted = 0.0

    with torch.no_grad():
        for X, y, valid_bins in data_loader:
            X = X.to(device)
            y = y.to(device)
            valid_bins = valid_bins.to(device)
            output_spikes, output_mems, hidden_spikes, hidden_pre, output_pre = (
                model.forward_trajectory(X, collect_pre_reset=True)
            )
            if hidden_pre is None or output_pre is None:
                raise RuntimeError("Evaluation requires pre-reset membrane diagnostics")
            valid_logits = valid_evidence(output_spikes, valid_bins, model.output_cap)
            full_logits = full_evidence(output_spikes, model.output_cap)
            valid_mem = exp40.valid_final_membrane(output_mems, valid_bins)
            full_mem = output_mems[:, -1]
            loss = F.cross_entropy(valid_logits, y)
            n = len(y)
            n_total += n
            loss_sum += float(loss.item()) * n
            y_true_parts.append(y.cpu().numpy())
            valid_pred_parts.append(valid_logits.argmax(dim=1).cpu().numpy())
            full_pred_parts.append(full_logits.argmax(dim=1).cpu().numpy())
            valid_mem_pred_parts.append(valid_mem.argmax(dim=1).cpu().numpy())
            full_mem_pred_parts.append(full_mem.argmax(dim=1).cpu().numpy())

            h_stats = _distribution_stats(
                hidden_spikes, hidden_pre, valid_bins, model.hidden_cap
            )
            o_stats = _distribution_stats(
                output_spikes, output_pre, valid_bins, model.output_cap
            )
            for key, value in h_stats.items():
                hidden_stat_sums[key] = hidden_stat_sums.get(key, 0.0) + value * n
            for key, value in o_stats.items():
                output_stat_sums[key] = output_stat_sums.get(key, 0.0) + value * n
            hidden_tail_weighted += _tail_fraction(hidden_spikes, valid_bins) * n
            output_tail_weighted += _tail_fraction(output_spikes, valid_bins) * n
            stat_weight += n

    y_true = np.concatenate(y_true_parts)
    result: dict[str, object] = {
        "valid_count_loss": float(loss_sum / max(n_total, 1)),
        "valid_count": exp40.metrics(y_true, np.concatenate(valid_pred_parts)),
        "full_count": exp40.metrics(y_true, np.concatenate(full_pred_parts)),
        "valid_membrane": exp40.metrics(
            y_true, np.concatenate(valid_mem_pred_parts)
        ),
        "full_membrane": exp40.metrics(y_true, np.concatenate(full_mem_pred_parts)),
        "hidden_valid_stats": {
            key: float(value / max(stat_weight, 1))
            for key, value in hidden_stat_sums.items()
        },
        "output_valid_stats": {
            key: float(value / max(stat_weight, 1))
            for key, value in output_stat_sums.items()
        },
        "hidden_tail_event_fraction": float(
            hidden_tail_weighted / max(stat_weight, 1)
        ),
        "output_tail_event_fraction": float(
            output_tail_weighted / max(stat_weight, 1)
        ),
    }
    return result


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _provenance(
    spec: RunSpec,
    data: exp40.BinnedData,
    config: Config,
) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_representation_experiment": exp40.EXPERIMENT_ID,
        "architecture": spec.architecture,
        "hidden_width": spec.hidden_width,
        "tau_mem_ms": spec.tau_mem_ms,
        "variant": spec.variant,
        "hidden_cap": spec.hidden_cap,
        "output_cap": spec.output_cap,
        "output_readout_normalization": "divide output spikes by output_cap before WholeCount CE",
        "seed": spec.seed,
        "split_seed": int(exp40.base.SPLIT_SEED),
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
        "event_channels": exp40.EVENT_CHANNELS,
        "fixed_bin_ms": exp40.FIXED_MS,
        "n_padded_bins": data.n_bins,
        "hidden_update": "pre=beta*mem+current; events=capped floor(max(pre,0)/threshold); mem=pre-events*threshold",
        "surrogate": "sum of fast-sigmoid derivatives around each reachable threshold crossing",
        "threshold": THRESHOLD,
        "output_tau_mem_ms": exp40.OUTPUT_TAU_MEM_MS,
        "state_dynamics": "all padded bins execute; zero input after endpoint; no state freeze",
        "primary_objective": "valid normalized output-event WholeCount CE",
        "paired_randomness": "same architecture/seed uses identical model-init and train-loader random streams across all cap variants",
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "channel_scale": data.channel_scale.tolist(),
        "parameter_counts": exp40.parameter_counts(
            spec.hidden_width,
            spec.architecture == "rsnn",
            len(data.labels),
        ),
    }


def run_one(
    spec: RunSpec,
    data: exp40.BinnedData,
    config: Config,
) -> dict[str, object]:
    path = evaluation_path(config.results_dir, spec)
    ckpt = checkpoint_path(config.results_dir, spec)
    if config.resume and path.exists() and ckpt.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp40.base.seed_all(paired_seed(spec, "model_init"))
    model = MacroTemporalDecoder(
        architecture=spec.architecture,
        hidden_width=spec.hidden_width,
        tau_mem_ms=spec.tau_mem_ms,
        n_classes=len(data.labels),
        hidden_cap=spec.hidden_cap,
        output_cap=spec.output_cap,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    partitions = [
        (data.Xtr, data.ytr, data.btr),
        (data.Xva, data.yva, data.bva),
        (data.Xte, data.yte, data.bte),
    ]
    train_loader = exp40.loader(
        *partitions[0],
        config.batch_size,
        True,
        paired_seed(spec, "train_loader"),
    )
    eval_loaders = [
        exp40.loader(
            *partition,
            config.batch_size,
            False,
            paired_seed(spec, f"eval_loader_{split}"),
        )
        for partition, split in zip(partitions, ("train", "val", "test"), strict=True)
    ]

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -np.inf
    best_val_loss = np.inf
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for X, y, valid_bins in train_loader:
            X = X.to(device)
            y = y.to(device)
            valid_bins = valid_bins.to(device)
            optimizer.zero_grad(set_to_none=True)
            output_spikes, _, _, _, _ = model.forward_trajectory(X)
            logits = valid_evidence(output_spikes, valid_bins, model.output_cap)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(y)
            train_n += len(y)

        val_metrics = evaluate_model(model, eval_loaders[1], device)
        val_ba = float(val_metrics["valid_count"]["balanced_accuracy"])
        val_loss = float(val_metrics["valid_count_loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss_sum / max(train_n, 1)),
                "val_valid_count_loss": val_loss,
                "val_valid_count_ba": val_ba,
            }
        )
        if (val_ba > best_val_ba) or (
            np.isclose(val_ba, best_val_ba) and val_loss < best_val_loss
        ):
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    model.load_state_dict(best_state)
    split_metrics = {
        split: evaluate_model(model, dl, device)
        for split, dl in zip(("train", "val", "test"), eval_loaders, strict=True)
    }
    provenance = _provenance(spec, data, config)

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "provenance": provenance,
            "best_epoch": best_epoch,
            "model_state_dict": best_state,
            "history": history,
        },
        ckpt,
    )
    payload: dict[str, object] = {
        "spec": spec.__dict__,
        "key": spec.key,
        "best_epoch": best_epoch,
        "provenance": provenance,
        "metrics": split_metrics,
        "history": history,
    }
    _save_json(path, payload)
    return payload


def _flatten_metrics(spec: RunSpec, payload: dict[str, object], split: str) -> dict[str, object]:
    metrics = payload["metrics"][split]
    row: dict[str, object] = {
        "architecture": spec.architecture,
        "hidden_width": spec.hidden_width,
        "tau_mem_ms": spec.tau_mem_ms,
        "variant": spec.variant,
        "hidden_cap": spec.hidden_cap,
        "output_cap": spec.output_cap,
        "seed": spec.seed,
        "split": split,
        "best_epoch": payload["best_epoch"],
        "parameter_count": payload["provenance"]["parameter_counts"]["total"],
        "hidden_tail_event_fraction": metrics["hidden_tail_event_fraction"],
        "output_tail_event_fraction": metrics["output_tail_event_fraction"],
    }
    for readout in ("valid_count", "full_count", "valid_membrane", "full_membrane"):
        for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
            row[f"{readout}_{metric_name}"] = metrics[readout][metric_name]
    for prefix, source in (
        ("hidden", metrics["hidden_valid_stats"]),
        ("output", metrics["output_valid_stats"]),
    ):
        for name, value in source.items():
            row[f"{prefix}_{name}"] = value
    return row


def _write_factorial_effects(test: pd.DataFrame, root: Path) -> None:
    rows: list[dict[str, object]] = []
    for (architecture, seed), group in test.groupby(["architecture", "seed"]):
        scores = {
            str(row.variant): float(row.valid_count_balanced_accuracy)
            for row in group.itertuples(index=False)
        }
        required = {"binary", "multi_h", "multi_o", "multi_ho"}
        if set(scores) != required:
            raise ValueError(
                f"Incomplete factorial group for {architecture} seed {seed}: {sorted(scores)}"
            )
        binary = scores["binary"]
        multi_h = scores["multi_h"]
        multi_o = scores["multi_o"]
        multi_ho = scores["multi_ho"]
        rows.append(
            {
                "architecture": architecture,
                "seed": int(seed),
                "binary": binary,
                "multi_h": multi_h,
                "multi_o": multi_o,
                "multi_ho": multi_ho,
                "hidden_effect_at_output1": multi_h - binary,
                "output_effect_at_hidden1": multi_o - binary,
                "hidden_effect_at_output31": multi_ho - multi_o,
                "output_effect_at_hidden31": multi_ho - multi_h,
                "interaction": multi_ho - multi_h - multi_o + binary,
            }
        )
    pd.DataFrame(rows).to_csv(root / "factorial_effects.csv", index=False)


def _write_binary_sanity(test: pd.DataFrame, config: Config) -> None:
    source_path = exp40.results_dir(config.repo_root) / "runs.csv"
    if not source_path.exists():
        return
    source = pd.read_csv(source_path)
    source = source[source["split"] == "test"].copy()
    custom = test[test["variant"] == "binary"].copy()
    rows: list[dict[str, object]] = []
    for row in custom.itertuples(index=False):
        match = source[
            (source["architecture"] == row.architecture)
            & (source["hidden_width"] == row.hidden_width)
            & np.isclose(source["tau_mem_ms"], row.tau_mem_ms)
            & (source["seed"] == row.seed)
        ]
        if len(match) != 1:
            continue
        original = float(match.iloc[0]["valid_count_balanced_accuracy"])
        custom_ba = float(row.valid_count_balanced_accuracy)
        rows.append(
            {
                "architecture": row.architecture,
                "hidden_width": int(row.hidden_width),
                "tau_mem_ms": float(row.tau_mem_ms),
                "seed": int(row.seed),
                "original_exp4_0_binary_ba": original,
                "custom_cap1_binary_ba": custom_ba,
                "custom_minus_original": custom_ba - original,
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(config.results_dir / "binary_sanity.csv", index=False)


def finalize(data: exp40.BinnedData, config: Config) -> None:
    specs = run_specs()
    missing = [
        str(evaluation_path(config.results_dir, spec))
        for spec in specs
        if not evaluation_path(config.results_dir, spec).exists()
    ]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Cannot finalize: {len(missing)} required artifacts are missing.\n{preview}"
        )

    rows: list[dict[str, object]] = []
    for spec in specs:
        payload = json.loads(
            evaluation_path(config.results_dir, spec).read_text(encoding="utf-8")
        )
        for split in ("train", "val", "test"):
            rows.append(_flatten_metrics(spec, payload, split))
    runs = pd.DataFrame(rows)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "runs.csv", index=False)

    test = runs[runs["split"] == "test"].copy()
    group_cols = [
        "architecture",
        "hidden_width",
        "tau_mem_ms",
        "variant",
        "hidden_cap",
        "output_cap",
    ]
    value_cols = [
        "valid_count_balanced_accuracy",
        "valid_count_macro_f1",
        "full_count_balanced_accuracy",
        "hidden_tail_event_fraction",
        "output_tail_event_fraction",
        "hidden_mean_events_per_neuron_step",
        "hidden_fraction_gt1",
        "hidden_fraction_at_cap",
        "output_mean_events_per_neuron_step",
        "output_fraction_gt1",
        "output_fraction_at_cap",
    ]
    summary = test.groupby(group_cols)[value_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(config.results_dir / "summary.csv", index=False)
    _write_factorial_effects(test, config.results_dir)
    _write_binary_sanity(test, config)

    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "expected_runs": EXPECTED_RUNS,
        "anchors": [
            {
                "architecture": architecture,
                "hidden_width": hidden_width,
                "tau_mem_ms": tau_mem_ms,
            }
            for architecture, hidden_width, tau_mem_ms in ANCHORS
        ],
        "variants": [
            {
                "name": name,
                "hidden_cap": hidden_cap,
                "output_cap": output_cap,
            }
            for name, hidden_cap, output_cap in VARIANTS
        ],
        "seeds": list(SEEDS),
        "split_seed": int(exp40.base.SPLIT_SEED),
        "fixed_bin_ms": exp40.FIXED_MS,
        "event_channels": exp40.EVENT_CHANNELS,
        "output_readout_normalization": "output event count divided by output_cap before CE",
        "custom_binary_control": "all four variants use the same MacroMultiSpikeLIF update/reset implementation",
        "paired_randomness": "same architecture/seed uses identical initialization and train-loader order across cap variants",
        "state_dynamics": "all padded bins execute; zero input after endpoint; no state freeze",
        "channel_scale": data.channel_scale.tolist(),
    }
    _save_json(config.results_dir / "provenance.json", provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 4.0.1 binary vs multi-spike macro-LIF ablation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--epochs", type=int, default=EPOCHS)
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--force-retrain", action="store_true")
    final = sub.add_parser("finalize")
    final.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = exp40.find_repo_root()
    root = results_dir(repo_root)
    data = exp40.prepare_binned_data(repo_root)
    if args.command == "run-one":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(
                f"array-task-id {args.array_task_id} outside [0,{len(specs)-1}]"
            )
        spec = specs[args.array_task_id]
        config = Config(
            repo_root=repo_root,
            results_dir=root,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume=not args.force_retrain,
            threads=args.threads,
        )
        result = run_one(spec, data, config)
        print(
            json.dumps(
                {"completed": spec.key, "best_epoch": result["best_epoch"]},
                indent=2,
            )
        )
    else:
        config = Config(repo_root=repo_root, results_dir=root, device=args.device)
        finalize(data, config)
        print(root / "summary.csv")


if __name__ == "__main__":
    main()
