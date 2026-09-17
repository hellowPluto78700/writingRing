from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73


EXPERIMENT_ID = "experiment_8_1_hidden_quantization_ablation"
PROTOCOL_VERSION = "hidden_quantization_ablation_v1"
SEEDS = exp73.SEEDS
L2_WIDTH = exp73.HIDDEN_WIDTH
SHIFTS = (2, 3, 4)
THRESHOLD = float(exp73.THRESHOLD)
THRESHOLD_MULTIPLIERS = (0.5, 1.0, 1.5)
WEIGHTED_CAP = 31
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
REGULARIZATION = exp73.REGULARIZATION
PROBE_STATES = ("pre_reset", "communication")
PROBE_AGGREGATIONS = ("whole_mean", "fixed250_ordered_mean")
LAYERS = ("l1", "l2")

METHOD_CONFIG: dict[str, dict[str, Any]] = {
    "binary128": {"l1_width": 128, "coding": "binary"},
    "weighted31_128": {"l1_width": 128, "coding": "weighted31"},
    "mt3_128": {"l1_width": 128, "coding": "hetero3"},
    "binary384": {"l1_width": 384, "coding": "binary"},
    "mt3_384": {"l1_width": 384, "coding": "hetero3"},
}
METHOD_ORDER = tuple(METHOD_CONFIG)
EXPECTED_RUNS = len(METHOD_ORDER) * len(SEEDS)


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.method}__a2_wcce__{REGULARIZATION}__seed{self.seed}"

    @property
    def l1_width(self) -> int:
        return int(METHOD_CONFIG[self.method]["l1_width"])

    @property
    def coding(self) -> str:
        return str(METHOD_CONFIG[self.method]["coding"])


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(method, seed) for method in METHOD_ORDER for seed in SEEDS]


def validate_spec(spec: RunSpec) -> None:
    if spec.method not in METHOD_CONFIG:
        raise ValueError(spec.method)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _valid_mask(lengths: torch.Tensor, n_steps: int) -> torch.Tensor:
    return torch.arange(n_steps, device=lengths.device)[None, :] < lengths[:, None]


def _valid_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _valid_mask(lengths, values.shape[1]).to(values.dtype).unsqueeze(-1)
    return (values * mask).sum(dim=1) / lengths.clamp_min(1).to(values.dtype).unsqueeze(1)


def _bank_widths(width: int) -> tuple[int, ...]:
    q, r = divmod(width, len(THRESHOLD_MULTIPLIERS))
    return tuple(q + (i < r) for i in range(len(THRESHOLD_MULTIPLIERS)))


def l1_threshold_bank_widths(spec: RunSpec) -> tuple[int, ...]:
    if spec.coding != "hetero3":
        return (spec.l1_width,)
    return _bank_widths(spec.l1_width)


def _l1_alpha_vector(spec: RunSpec) -> torch.Tensor:
    if spec.coding != "hetero3":
        return exp50.alpha_vector(spec.l1_width, SHIFTS)
    return torch.cat(
        [exp50.alpha_vector(width, SHIFTS) for width in l1_threshold_bank_widths(spec)]
    )


def _l1_threshold_vector(spec: RunSpec) -> torch.Tensor:
    if spec.coding != "hetero3":
        return torch.full((spec.l1_width,), THRESHOLD, dtype=torch.float32)
    parts = [
        torch.full((width,), THRESHOLD * multiplier, dtype=torch.float32)
        for width, multiplier in zip(
            l1_threshold_bank_widths(spec), THRESHOLD_MULTIPLIERS, strict=True
        )
    ]
    return torch.cat(parts)


def l1_groups(spec: RunSpec) -> list[dict[str, float | int]]:
    groups: list[dict[str, float | int]] = []
    start = 0
    if spec.coding == "hetero3":
        banks = zip(
            l1_threshold_bank_widths(spec), THRESHOLD_MULTIPLIERS, strict=True
        )
    else:
        banks = ((spec.l1_width, 1.0),)
    for bank_index, (bank_width, multiplier) in enumerate(banks):
        for local in exp72.shift_groups(SHIFTS, width=int(bank_width)):
            count = int(local["count"])
            groups.append(
                {
                    "bank": int(bank_index),
                    "threshold_multiplier": float(multiplier),
                    "shift": int(local["shift"]),
                    "start": start,
                    "stop": start + count,
                    "count": count,
                }
            )
            start += count
    if start != spec.l1_width:
        raise RuntimeError((spec.key, start, spec.l1_width))
    return groups


def l2_groups() -> list[dict[str, float | int]]:
    return [
        {
            "bank": 0,
            "threshold_multiplier": 1.0,
            "shift": int(group["shift"]),
            "start": int(group["start"]),
            "stop": int(group["stop"]),
            "count": int(group["count"]),
        }
        for group in exp72.shift_groups(SHIFTS, width=L2_WIDTH)
    ]


class _VectorThresholdSpike(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        membrane: torch.Tensor,
        thresholds: torch.Tensor,
        slope: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(membrane, thresholds)
        ctx.slope = float(slope)
        return (membrane >= thresholds).to(membrane.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        membrane, thresholds = ctx.saved_tensors
        slope = float(ctx.slope)
        normalized = (membrane - thresholds) / thresholds
        grad = 1.0 / (1.0 + slope * normalized.abs()).pow(2)
        grad = grad / thresholds
        return grad_output * grad, None, None


class VectorThresholdBinaryLIF(nn.Module):
    """Binary LIF with fixed per-neuron thresholds; thresholds never adapt."""

    def __init__(self, beta: float, thresholds: torch.Tensor, surrogate_slope: float) -> None:
        super().__init__()
        if thresholds.ndim != 1 or bool(torch.any(thresholds <= 0)):
            raise ValueError("thresholds must be a positive 1-D tensor")
        self.beta = float(beta)
        self.surrogate_slope = float(surrogate_slope)
        self.register_buffer("thresholds", thresholds.detach().clone().float())

    def forward(
        self, current: torch.Tensor, membrane: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_reset = self.beta * membrane + current
        spikes = _VectorThresholdSpike.apply(
            pre_reset, self.thresholds, self.surrogate_slope
        )
        post_reset = pre_reset - spikes * self.thresholds
        return spikes, post_reset, pre_reset


class Exp81Net(nn.Module):
    def __init__(self, spec: RunSpec, n_classes: int, fs: float) -> None:
        super().__init__()
        validate_spec(spec)
        self.spec = spec
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        self.hidden_linears = nn.ModuleList(
            [
                nn.Linear(exp72.EXPECTED_CHANNELS, spec.l1_width, bias=False),
                nn.Linear(spec.l1_width, L2_WIDTH, bias=False),
            ]
        )
        beta_hidden = math.exp(-(1000.0 / fs) / exp72.TAU_MEM_MS)
        if spec.coding == "hetero3":
            l1_lif: nn.Module = VectorThresholdBinaryLIF(
                beta_hidden,
                _l1_threshold_vector(spec),
                exp72.SURROGATE_SLOPE,
            )
        else:
            l1_lif = exp401.MacroMultiSpikeLIF(
                beta=beta_hidden,
                threshold=THRESHOLD,
                max_spikes_per_dt=WEIGHTED_CAP if spec.coding == "weighted31" else 1,
                surrogate_slope=exp72.SURROGATE_SLOPE,
            )
        self.l1_lif = l1_lif
        self.l2_lif = exp401.MacroMultiSpikeLIF(
            beta=beta_hidden,
            threshold=THRESHOLD,
            max_spikes_per_dt=1,
            surrogate_slope=exp72.SURROGATE_SLOPE,
        )
        self.register_buffer("alpha_0", _l1_alpha_vector(spec))
        self.register_buffer("alpha_1", exp50.alpha_vector(L2_WIDTH, SHIFTS))
        self.output_linear = nn.Linear(L2_WIDTH, n_classes, bias=False)

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != exp72.EXPECTED_CHANNELS:
            raise ValueError(channels)
        syn1 = torch.zeros(batch, self.spec.l1_width, device=x.device, dtype=x.dtype)
        mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, L2_WIDTH, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(syn2)
        l1_comm: list[torch.Tensor] = []
        l2_comm: list[torch.Tensor] = []
        l1_pre: list[torch.Tensor] = []
        l2_pre: list[torch.Tensor] = []
        evidence: list[torch.Tensor] = []
        for timestep in range(steps):
            syn1 = self.alpha_0 * syn1 + self.hidden_linears[0](x[:, timestep])
            out1, mem1, pre1 = self.l1_lif(syn1, mem1)
            syn2 = self.alpha_1 * syn2 + self.hidden_linears[1](out1)
            out2, mem2, pre2 = self.l2_lif(syn2, mem2)
            l1_comm.append(out1)
            l2_comm.append(out2)
            l1_pre.append(pre1)
            l2_pre.append(pre2)
            evidence.append(self.output_linear(out2))
        return {
            "hidden_communication": (
                torch.stack(l1_comm, dim=1),
                torch.stack(l2_comm, dim=1),
            ),
            "hidden_pre_reset": (
                torch.stack(l1_pre, dim=1),
                torch.stack(l2_pre, dim=1),
            ),
            "evidence": torch.stack(evidence, dim=1),
        }


def new_model(spec: RunSpec, data: exp3.Data) -> Exp81Net:
    return Exp81Net(spec, len(data.labels), data.fs)


def _evaluate_linear(
    model: Exp81Net, loader: Iterable, device: torch.device
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum, n_total = 0.0, 0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            trajectory = model.forward_trajectory(X)
            scores = _valid_mean(trajectory["evidence"], lengths)
            loss = F.cross_entropy(scores, y)
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)
    metrics = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    metrics["objective_loss"] = loss_sum / max(n_total, 1)
    return metrics


def _output_lif_spikes(evidence: torch.Tensor) -> torch.Tensor:
    batch, steps, n_classes = evidence.shape
    membrane = torch.zeros(batch, n_classes, device=evidence.device, dtype=evidence.dtype)
    lif = exp401.MacroMultiSpikeLIF(
        beta=0.5,
        threshold=THRESHOLD,
        max_spikes_per_dt=1,
        surrogate_slope=exp72.SURROGATE_SLOPE,
    ).to(evidence.device)
    spikes: list[torch.Tensor] = []
    for timestep in range(steps):
        spike, membrane, _ = lif(evidence[:, timestep], membrane)
        spikes.append(spike)
    return torch.stack(spikes, dim=1)


def _evaluate_lif_transfer(
    model: Exp81Net, loader: Iterable, device: torch.device
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X, lengths = X.to(device), lengths.to(device)
            trajectory = model.forward_trajectory(X)
            spikes = _output_lif_spikes(trajectory["evidence"])
            mask = _valid_mask(lengths, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
            scores = (spikes * mask).sum(dim=1)
            ys.append(y.numpy())
            preds.append(scores.argmax(1).cpu().numpy())
    return exp72._metrics(np.concatenate(ys), np.concatenate(preds))


def _whole_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return _valid_mean(values, lengths)


def _fixed250_ordered_mean(
    values: torch.Tensor, lengths: torch.Tensor, bin_steps: int
) -> torch.Tensor:
    batch, steps, width = values.shape
    n_bins = int(math.ceil(steps / bin_steps))
    padded_steps = n_bins * bin_steps
    mask = _valid_mask(lengths, steps).to(values.dtype)
    if padded_steps != steps:
        values = torch.cat(
            [
                values,
                torch.zeros(
                    batch,
                    padded_steps - steps,
                    width,
                    device=values.device,
                    dtype=values.dtype,
                ),
            ],
            dim=1,
        )
        mask = torch.cat(
            [
                mask,
                torch.zeros(
                    batch,
                    padded_steps - steps,
                    device=values.device,
                    dtype=values.dtype,
                ),
            ],
            dim=1,
        )
    sums = (values * mask.unsqueeze(-1)).reshape(
        batch, n_bins, bin_steps, width
    ).sum(dim=2)
    counts = mask.reshape(batch, n_bins, bin_steps).sum(dim=2).clamp_min(1.0)
    return (sums / counts.unsqueeze(-1)).flatten(start_dim=1)


def _probe_features(
    values: torch.Tensor,
    lengths: np.ndarray,
    aggregation: str,
    bin_steps: int,
) -> np.ndarray:
    length_tensor = torch.as_tensor(lengths, dtype=torch.long)
    if aggregation == "whole_mean":
        return _whole_mean(values, length_tensor).numpy()
    if aggregation == "fixed250_ordered_mean":
        return _fixed250_ordered_mean(values, length_tensor, bin_steps).numpy()
    raise ValueError(aggregation)


def _extract_state_splits(
    model: Exp81Net,
    loaders: dict[str, Iterable],
    device: torch.device,
) -> dict[str, dict[str, dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]]]]:
    output: dict[str, dict[str, dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]]]] = {
        layer: {state: {} for state in PROBE_STATES} for layer in LAYERS
    }
    model.eval()
    with torch.no_grad():
        for split, loader in loaders.items():
            chunks = {
                layer: {state: [] for state in PROBE_STATES} for layer in LAYERS
            }
            ys: list[np.ndarray] = []
            lengths_all: list[np.ndarray] = []
            for X, y, lengths in loader:
                trajectory = model.forward_trajectory(X.to(device))
                for index, layer in enumerate(LAYERS):
                    chunks[layer]["communication"].append(
                        trajectory["hidden_communication"][index].cpu()
                    )
                    chunks[layer]["pre_reset"].append(
                        trajectory["hidden_pre_reset"][index].cpu()
                    )
                ys.append(y.numpy())
                lengths_all.append(lengths.numpy())
            y_array = np.concatenate(ys)
            length_array = np.concatenate(lengths_all)
            for layer in LAYERS:
                for state in PROBE_STATES:
                    output[layer][state][split] = (
                        torch.cat(chunks[layer][state], dim=0),
                        y_array,
                        length_array,
                    )
    return output


def _fit_state_probes(
    splits: dict[str, dict[str, dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]]]],
    seed: int,
    bin_steps: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer in LAYERS:
        result[layer] = {}
        for state in PROBE_STATES:
            result[layer][state] = {}
            for aggregation in PROBE_AGGREGATIONS:
                features = {
                    split: (
                        _probe_features(values, lengths, aggregation, bin_steps),
                        y,
                    )
                    for split, (values, y, lengths) in splits[layer][state].items()
                }
                result[layer][state][aggregation] = exp01._fit_linear_probe(
                    features["train"][0],
                    features["train"][1],
                    features["val"][0],
                    features["val"][1],
                    features["test"][0],
                    features["test"][1],
                    seed=exp3.dseed(seed, EXPERIMENT_ID, layer, state, aggregation),
                )
    return result


def _communication_stats(
    values: torch.Tensor,
    lengths: np.ndarray,
    groups: list[dict[str, float | int]],
    cap: int,
) -> list[dict[str, float | int]]:
    length_tensor = torch.as_tensor(lengths, dtype=torch.long)
    mask = _valid_mask(length_tensor, values.shape[1]).to(values.dtype).unsqueeze(-1)
    rows: list[dict[str, float | int]] = []
    for group in groups:
        start, stop = int(group["start"]), int(group["stop"])
        chunk = values[:, :, start:stop]
        group_mask = mask.expand(-1, -1, stop - start)
        denominator = float(group_mask.sum().item())
        mean_value = float((chunk * group_mask).sum().item()) / max(denominator, 1.0)
        fraction_nonzero = float(((chunk > 0).to(chunk.dtype) * group_mask).sum().item()) / max(
            denominator, 1.0
        )
        fraction_gt1 = float(((chunk > 1).to(chunk.dtype) * group_mask).sum().item()) / max(
            denominator, 1.0
        )
        fraction_at_cap = float(
            ((chunk >= float(cap)).to(chunk.dtype) * group_mask).sum().item()
        ) / max(denominator, 1.0)
        rows.append(
            {
                **group,
                "cap": int(cap),
                "mean_value_per_neuron_step": mean_value,
                "fraction_nonzero": fraction_nonzero,
                "fraction_gt1": fraction_gt1,
                "fraction_at_cap": fraction_at_cap,
            }
        )
    return rows


def _collect_activity(
    spec: RunSpec,
    state_splits: dict[
        str, dict[str, dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]]]
    ],
) -> dict[str, Any]:
    l1_values, _, l1_lengths = state_splits["l1"]["communication"]["test"]
    l2_values, _, l2_lengths = state_splits["l2"]["communication"]["test"]
    l1_cap = WEIGHTED_CAP if spec.coding == "weighted31" else 1
    return {
        "l1": _communication_stats(l1_values, l1_lengths, l1_groups(spec), l1_cap),
        "l2": _communication_stats(l2_values, l2_lengths, l2_groups(), 1),
    }


def _probe_ba(
    probes: dict[str, Any], layer: str, state: str, aggregation: str
) -> float:
    probe = probes[layer][state][aggregation]
    if "metrics" in probe:
        return float(probe["metrics"]["test"]["balanced_accuracy"])
    if "test" in probe and isinstance(probe["test"], dict):
        return float(probe["test"]["balanced_accuracy"])
    return float(probe["test_balanced_accuracy"])


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    evaluation_path = _path(config.results_dir, "evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "checkpoints", spec.key, ".pt")
    if evaluation_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(evaluation_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)

    exp3.seed_all(exp73._e2e_pair_seed(spec.seed, "model_init"))
    model = new_model(spec, data).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY
    )
    train_loader = exp73._raw_loaders(
        data, spec.seed, config.batch_size, True
    )["train"]
    eval_loaders = exp73._raw_loaders(data, spec.seed, config.batch_size, False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch, best_ba, best_loss = -1, -1.0, float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum, n_total = 0.0, 0
        for X, y, lengths in train_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            scores = _valid_mean(trajectory["evidence"], lengths)
            loss = F.cross_entropy(scores, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate_linear(model, eval_loaders["train"], device)
        val_metrics = _evaluate_linear(model, eval_loaders["val"], device)
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )
        if exp73._checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE:
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "method_config": METHOD_CONFIG[spec.method],
            "threshold_multipliers": THRESHOLD_MULTIPLIERS,
            "weighted_cap": WEIGHTED_CAP,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )

    model.load_state_dict(best_state, strict=True)
    linear_metrics = {
        split: _evaluate_linear(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    lif_metrics = {
        split: _evaluate_lif_transfer(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    state_splits = _extract_state_splits(model, eval_loaders, device)
    probes = _fit_state_probes(state_splits, spec.seed, data.bin_steps)
    activity = _collect_activity(spec, state_splits)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "method_config": METHOD_CONFIG[spec.method],
        "l1_threshold_bank_widths": list(l1_threshold_bank_widths(spec)),
        "threshold_multipliers": list(THRESHOLD_MULTIPLIERS),
        "weighted_cap": WEIGHTED_CAP,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "linear_metrics": linear_metrics,
        "lif_transfer_metrics": lif_metrics,
        "representation_probes": probes,
        "activity": activity,
    }
    _save_json(evaluation_path, payload)
    history_path = _path(config.results_dir, "histories", spec.key, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return payload


def _method_row(payload: dict[str, Any]) -> dict[str, Any]:
    probes = payload["representation_probes"]
    row: dict[str, Any] = {
        "method": payload["spec"]["method"],
        "seed": payload["spec"]["seed"],
        "l1_width": payload["method_config"]["l1_width"],
        "coding": payload["method_config"]["coding"],
        "parameter_count": payload["parameter_count"],
        "linear_test_ba": payload["linear_metrics"]["test"]["balanced_accuracy"],
        "lif_test_ba": payload["lif_transfer_metrics"]["test"]["balanced_accuracy"],
        "best_epoch": payload["best_epoch"],
    }
    for layer in LAYERS:
        for state in PROBE_STATES:
            for aggregation in PROBE_AGGREGATIONS:
                short_aggregation = "whole" if aggregation == "whole_mean" else "fixed250"
                row[f"{layer}_{state}_{short_aggregation}_ba"] = _probe_ba(
                    probes, layer, state, aggregation
                )
    for layer in LAYERS:
        row[f"{layer}_quantization_gap_whole"] = (
            row[f"{layer}_pre_reset_whole_ba"]
            - row[f"{layer}_communication_whole_ba"]
        )
        row[f"{layer}_quantization_gap_fixed250"] = (
            row[f"{layer}_pre_reset_fixed250_ba"]
            - row[f"{layer}_communication_fixed250_ba"]
        )
    return row


def _activity_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, groups in payload["activity"].items():
        for group in groups:
            rows.append(
                {
                    "method": payload["spec"]["method"],
                    "seed": payload["spec"]["seed"],
                    "layer": layer,
                    **group,
                }
            )
    return rows


def _paired_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("weighted_vs_binary_128", "weighted31_128", "binary128"),
        ("mt_vs_binary_128", "mt3_128", "binary128"),
        ("mt_vs_weighted_128", "mt3_128", "weighted31_128"),
        ("width_binary_384_vs_128", "binary384", "binary128"),
        ("width_mt_384_vs_128", "mt3_384", "mt3_128"),
        ("mt_vs_binary_384", "mt3_384", "binary384"),
    )
    metrics = [
        "linear_test_ba",
        "lif_test_ba",
        "l1_pre_reset_fixed250_ba",
        "l1_communication_fixed250_ba",
        "l2_pre_reset_fixed250_ba",
        "l2_communication_fixed250_ba",
        "l1_quantization_gap_fixed250",
        "l2_quantization_gap_fixed250",
    ]
    rows: list[dict[str, Any]] = []
    for contrast, method_a, method_b in comparisons:
        for seed in SEEDS:
            a = runs[(runs.method == method_a) & (runs.seed == seed)].iloc[0]
            b = runs[(runs.method == method_b) & (runs.seed == seed)].iloc[0]
            row: dict[str, Any] = {
                "contrast": contrast,
                "method_a": method_a,
                "method_b": method_b,
                "seed": seed,
            }
            for metric in metrics:
                row[f"delta_{metric}"] = float(a[metric] - b[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for spec in run_specs():
        path = _path(config.results_dir, "evaluations", spec.key, ".json")
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp8.1 evaluation: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if len(payloads) != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} runs, got {len(payloads)}")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([_method_row(payload) for payload in payloads])
    runs.to_csv(config.results_dir / "method_runs.csv", index=False)
    metric_columns = [
        column
        for column in runs.columns
        if column not in {"method", "seed", "coding"}
    ]
    summary = (
        runs.groupby("method", sort=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(config.results_dir / "method_summary.csv", index=False)

    contrasts = _paired_contrasts(runs)
    contrasts.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_metrics = [column for column in contrasts if column.startswith("delta_")]
    contrast_summary = (
        contrasts.groupby(["contrast", "method_a", "method_b"], sort=False)[
            contrast_metrics
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    contrast_summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in contrast_summary.columns
    ]
    contrast_summary.to_csv(config.results_dir / "contrast_summary.csv", index=False)

    activity_runs = pd.DataFrame(
        [row for payload in payloads for row in _activity_rows(payload)]
    )
    activity_runs.to_csv(config.results_dir / "activity_runs.csv", index=False)
    activity_summary = (
        activity_runs.groupby(
            ["method", "layer", "threshold_multiplier", "shift"], sort=False
        )[
            [
                "mean_value_per_neuron_step",
                "fraction_nonzero",
                "fraction_gt1",
                "fraction_at_cap",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    activity_summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in activity_summary.columns
    ]
    activity_summary.to_csv(config.results_dir / "activity_summary.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "methods": METHOD_CONFIG,
        "method_order": list(METHOD_ORDER),
        "seeds": list(SEEDS),
        "counts": {
            "methods": len(METHOD_ORDER),
            "seeds": len(SEEDS),
            "parallel_runs": EXPECTED_RUNS,
        },
        "training": "Exp7.3 A2-compatible end-to-end Linear/WCCE",
        "hidden_shifts": list(SHIFTS),
        "l2_width": L2_WIDTH,
        "threshold": THRESHOLD,
        "threshold_multipliers": list(THRESHOLD_MULTIPLIERS),
        "weighted_hidden_cap": WEIGHTED_CAP,
        "scope": "L1 coding/width only; L2 remains 128-neuron binary (234)",
        "primary_metric": "linear_test_ba",
        "diagnostics": {
            "states": list(PROBE_STATES),
            "aggregations": list(PROBE_AGGREGATIONS),
        },
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp8.1 hidden quantization ablation")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-runs")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--array-task-id", type=int, required=True)
    run_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = find_repo_root(args.repo_root)
    config = Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        max_epochs=args.max_epochs,
    )
    specs = run_specs()
    if args.command == "list-runs":
        for index, spec in enumerate(specs):
            print(index, spec.key)
        return
    if args.command == "run":
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        payload = run_one(specs[args.array_task_id], config, force=args.force)
        print(
            json.dumps(
                {
                    "key": specs[args.array_task_id].key,
                    "test_ba": payload["linear_metrics"]["test"]["balanced_accuracy"],
                },
                indent=2,
            )
        )
        return
    if args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
