from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import json
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_0_2_hidden_multitau_architectures as prev


EXPERIMENT_ID = "experiment_3_0_3_l3_bottleneck_ablation"
PROTOCOL_VERSION = "l3_bottleneck_v1"
BASELINE_EXPERIMENT_ID = prev.EXPERIMENT_ID
BASELINE_PROTOCOL_VERSION = prev.PROTOCOL_VERSION
BASELINE_ARCHITECTURE = "C"

OBJECTIVES = base.OBJECTIVES
SEEDS = base.SEEDS
EPOCHS = base.EPOCHS
BATCH_SIZE = base.BATCH_SIZE
LR = base.LR
N_REL = base.N_REL
FIXED_MS = base.FIXED_MS
EVENT_CHANNELS = base.EVENT_CHANNELS

L1_WIDTH = 128
L2_WIDTH = 128
L1_SHIFTS = (2, 3, 4)
L2_SHIFTS = (2, 3, 4)

PROBE_TYPES = prev.PROBE_TYPES
PROBE_C_GRID = prev.PROBE_C_GRID
balanced_group_slices = prev.balanced_group_slices
aggregate_mean_sd = prev.aggregate_mean_sd


@dataclass(frozen=True)
class ArchitectureSpec:
    label: str
    l3_width: int | None
    l3_shifts: tuple[int, ...]
    description: str


# A is exactly Experiment 3.0.2 architecture C and is reused without retraining.
# B-E keep L1/L2 fixed at 128 neurons with shifts (2,3,4) and only alter L3.
ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "A": ArchitectureSpec(
        label="3.0.2-C baseline",
        l3_width=64,
        l3_shifts=(3,),
        description="Reuse 3.0.2-C: 64-neuron single-tau L3",
    ),
    "B": ArchitectureSpec(
        label="No-L3",
        l3_width=None,
        l3_shifts=(),
        description="Remove L3 and attach the task head directly to L2",
    ),
    "C": ArchitectureSpec(
        label="Width128",
        l3_width=128,
        l3_shifts=(3,),
        description="Keep single-tau shift3 L3 but remove 128->64 width compression",
    ),
    "D": ArchitectureSpec(
        label="MultiTau64",
        l3_width=64,
        l3_shifts=(2, 3, 4),
        description="Keep L3 width 64 but remove single-tau temporal compression",
    ),
    "E": ArchitectureSpec(
        label="MultiTau128",
        l3_width=128,
        l3_shifts=(2, 3, 4),
        description="Keep width and multi-tau family through the third spiking layer",
    ),
}

TRAIN_ARCHITECTURES = ("B", "C", "D", "E")
EVAL_ARCHITECTURES = ("A", "B", "C", "D", "E")
EXPECTED_TRAIN_RUNS = len(TRAIN_ARCHITECTURES) * len(OBJECTIVES) * len(SEEDS)
EXPECTED_EVAL_RUNS = len(EVAL_ARCHITECTURES) * len(OBJECTIVES) * len(SEEDS)


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


def architecture_spec(architecture: str) -> ArchitectureSpec:
    try:
        return ARCHITECTURES[architecture]
    except KeyError as exc:
        raise ValueError(f"Unknown architecture: {architecture}") from exc


def architecture_layers(
    architecture: str,
) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
    spec = architecture_spec(architecture)
    layers: list[tuple[str, int, tuple[int, ...]]] = [
        ("L1", L1_WIDTH, L1_SHIFTS),
        ("L2", L2_WIDTH, L2_SHIFTS),
    ]
    if spec.l3_width is not None:
        layers.append(("L3", int(spec.l3_width), spec.l3_shifts))
    return tuple(layers)


def layer_names(architecture: str) -> tuple[str, ...]:
    return tuple(layer for layer, _, _ in architecture_layers(architecture))


def last_layer_name(architecture: str) -> str:
    return layer_names(architecture)[-1]


def last_layer_width(architecture: str) -> int:
    return architecture_layers(architecture)[-1][1]


def train_specs() -> list[tuple[str, str, int]]:
    return [
        (architecture, objective, seed)
        for architecture in TRAIN_ARCHITECTURES
        for objective in OBJECTIVES
        for seed in SEEDS
    ]


def eval_specs() -> list[tuple[str, str, int]]:
    return [
        (architecture, objective, seed)
        for architecture in EVAL_ARCHITECTURES
        for objective in OBJECTIVES
        for seed in SEEDS
    ]


def architecture_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for architecture in EVAL_ARCHITECTURES:
        spec = architecture_spec(architecture)
        for layer, width, shifts in architecture_layers(architecture):
            slices = balanced_group_slices(width, shifts)
            for shift, sl in slices.items():
                rows.append(
                    {
                        "architecture": architecture,
                        "label": spec.label,
                        "description": spec.description,
                        "has_l3": spec.l3_width is not None,
                        "last_layer": last_layer_name(architecture),
                        "layer": layer,
                        "layer_width": width,
                        "shifts": ",".join(map(str, shifts)),
                        "shift": int(shift),
                        "group_neurons": int(sl.stop - sl.start),
                        "tau_syn_ms": base.tau_ms(int(shift), base.EXPECTED_FS),
                        "source_experiment": (
                            BASELINE_EXPERIMENT_ID if architecture == "A" else EXPERIMENT_ID
                        ),
                        "source_architecture": (
                            BASELINE_ARCHITECTURE if architecture == "A" else architecture
                        ),
                    }
                )
    return rows


def _alpha_vector(width: int, shifts: tuple[int, ...]) -> torch.Tensor:
    values = torch.empty(width, dtype=torch.float32)
    for shift, sl in balanced_group_slices(width, shifts).items():
        values[sl] = base.alpha(shift)
    return values


class L3AblationNet(nn.Module):
    def __init__(
        self,
        architecture: str,
        objective: str,
        n_classes: int,
        T: int,
        fs: float,
        bin_steps: int,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.spec = architecture_spec(architecture)
        self.layer_order = layer_names(architecture)
        self.layer_widths = {
            layer: width for layer, width, _ in architecture_layers(architecture)
        }
        self.layer_shifts = {
            layer: shifts for layer, _, shifts in architecture_layers(architecture)
        }
        self.layer_slices = {
            layer: balanced_group_slices(self.layer_widths[layer], self.layer_shifts[layer])
            for layer in self.layer_order
        }
        self.last_layer = last_layer_name(architecture)
        self.last_width = last_layer_width(architecture)

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

        # Keep construction order aligned with Experiment 3.0.2 so reused A and
        # paired f1/f2 initialization remain directly comparable.
        self.f1 = nn.Linear(EVENT_CHANNELS, L1_WIDTH, bias=False)
        self.l1 = make_lif(L1_WIDTH, L1_SHIFTS)
        self.f2 = nn.Linear(L1_WIDTH, L2_WIDTH, bias=False)
        self.l2 = make_lif(L2_WIDTH, L2_SHIFTS)

        if self.spec.l3_width is not None:
            l3_width = int(self.spec.l3_width)
            self.f3 = nn.Linear(L2_WIDTH, l3_width, bias=False)
            self.l3 = make_lif(l3_width, self.spec.l3_shifts)

        self.objective = objective
        self.T = T
        self.bin_steps = bin_steps
        if objective == "timestep_ce":
            head_dim = self.last_width
        elif objective == "relative10_sequence_ce":
            head_dim = self.last_width * N_REL
        elif objective == "fixed250_sequence_ce":
            head_dim = self.last_width * int(math.ceil(T / bin_steps))
        else:
            raise ValueError(f"Unknown objective: {objective}")
        self.head = nn.Linear(head_dim, n_classes, bias=True)

    def layer_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, T, _ = x.shape
        syn1 = torch.zeros(batch, L1_WIDTH, device=x.device, dtype=x.dtype)
        mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, L2_WIDTH, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(syn2)
        seq1: list[torch.Tensor] = []
        seq2: list[torch.Tensor] = []

        if self.spec.l3_width is not None:
            l3_width = int(self.spec.l3_width)
            syn3 = torch.zeros(batch, l3_width, device=x.device, dtype=x.dtype)
            mem3 = torch.zeros_like(syn3)
            seq3: list[torch.Tensor] = []

        for timestep in range(T):
            s1, syn1, mem1 = self.l1(self.f1(x[:, timestep]), syn1, mem1)
            s2, syn2, mem2 = self.l2(self.f2(s1), syn2, mem2)
            seq1.append(s1)
            seq2.append(s2)
            if self.spec.l3_width is not None:
                s3, syn3, mem3 = self.l3(self.f3(s2), syn3, mem3)
                seq3.append(s3)

        out = {
            "L1": torch.stack(seq1, dim=1),
            "L2": torch.stack(seq2, dim=1),
        }
        if self.spec.l3_width is not None:
            out["L3"] = torch.stack(seq3, dim=1)
        return out

    def loss_logits(
        self,
        spikes: torch.Tensor,
        lengths: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, T, _ = spikes.shape
        if self.objective == "timestep_ce":
            logits_t = self.head(spikes)
            valid = base.mask(lengths, T)
            targets = y[:, None].expand(batch, T)
            loss = F.cross_entropy(logits_t[valid], targets[valid])
            weights = valid.to(logits_t.dtype).unsqueeze(-1)
            segment_logits = (logits_t * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            return loss, segment_logits

        if self.objective == "relative10_sequence_ce":
            features = base.relative_counts(spikes, lengths)
        else:
            features = base.fixed_counts(spikes, lengths, self.bin_steps)
        logits = self.head(features.flatten(start_dim=1))
        return F.cross_entropy(logits, y), logits


def evaluate_native(
    model: L3AblationNet,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for X, y, lengths in data_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            layers = model.layer_features(X)
            spikes = layers[model.last_layer]
            loss, logits = model.loss_logits(spikes, lengths, y)
            losses.append(float(loss.item()) * len(y))
            y_true_parts.append(y.cpu().numpy())
            y_pred_parts.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    out = base.metrics(y_true, y_pred)
    out["loss"] = float(sum(losses) / len(y_true))
    return out


def training_checkpoint_path(root: Path, architecture: str, objective: str, seed: int) -> Path:
    return root / "checkpoints" / f"arch{architecture}__{objective}__seed{seed}.pt"


def evaluation_path(root: Path, architecture: str, objective: str, seed: int) -> Path:
    return root / "evaluations" / f"arch{architecture}__{objective}__seed{seed}.json"


def _architecture_provenance(architecture: str) -> dict[str, object]:
    spec = architecture_spec(architecture)
    return {
        "label": spec.label,
        "l1_width": L1_WIDTH,
        "l1_shifts": L1_SHIFTS,
        "l2_width": L2_WIDTH,
        "l2_shifts": L2_SHIFTS,
        "l3_width": spec.l3_width,
        "l3_shifts": spec.l3_shifts,
        "last_layer": last_layer_name(architecture),
        "last_width": last_layer_width(architecture),
    }


def _provenance(
    architecture: str,
    objective: str,
    seed: int,
    data: base.Data,
    config: Config,
) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": architecture,
        "architecture_spec": _architecture_provenance(architecture),
        "seed": int(seed),
        "objective": objective,
        "split_seed": base.SPLIT_SEED,
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
        "sampling_rate_hz": float(data.fs),
        "padded_length": int(data.T),
        "fixed_bin_steps": int(data.bin_steps),
        "tau_mem_ms": base.TAU_MEM_MS,
        "threshold": base.THRESHOLD,
        "surrogate_slope": base.SURROGATE_SLOPE,
        "reset": base.RESET,
        "epochs": int(config.epochs),
        "batch_size": int(config.batch_size),
        "learning_rate": LR,
        "relative_bins": N_REL,
        "fixed_bin_ms": FIXED_MS,
        "event_channels": EVENT_CHANNELS,
    }


def _validate_training_checkpoint(
    payload: dict[str, object],
    expected: dict[str, object],
    path: Path,
) -> None:
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong experiment id in {path}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong protocol version in {path}")
    if payload.get("provenance") != expected:
        raise ValueError(
            f"Checkpoint provenance mismatch for {path}. Use --force-retrain."
        )
    if not isinstance(payload.get("result"), dict):
        raise ValueError(f"Checkpoint has no result dict: {path}")


def run_train_one(
    architecture: str,
    objective: str,
    seed: int,
    data: base.Data,
    config: Config,
) -> dict[str, object]:
    if architecture not in TRAIN_ARCHITECTURES:
        raise ValueError(f"Architecture {architecture} is not trained in Experiment 3.0.3")
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}")

    checkpoint = training_checkpoint_path(config.results_dir, architecture, objective, seed)
    provenance = _provenance(architecture, objective, seed, data, config)
    if config.resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _validate_training_checkpoint(payload, provenance, checkpoint)
        return payload["result"]

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)

    # Paired initialization: all new architectures share identical f1/f2 weights
    # for the same master seed. Architectures with the same L3 width also share
    # the same initial f3 weights. A/D (64) are therefore directly paired at f3;
    # C/E (128) are paired at f3. Head initialization is paired within equal
    # final feature dimensions for each objective.
    base.seed_all(base.dseed(seed, "shared_backbone_init"))
    model = L3AblationNet(
        architecture,
        objective,
        len(data.labels),
        data.T,
        data.fs,
        data.bin_steps,
    ).to(device)
    base.seed_all(base.dseed(seed, objective, "head_init"))
    model.head.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    partitions = [
        (data.Xtr, data.ytr, data.ltr),
        (data.Xva, data.yva, data.lva),
        (data.Xte, data.yte, data.lte),
    ]
    train_loader = base.loader(
        *partitions[0],
        config.batch_size,
        True,
        base.dseed(seed, "train", "loader"),
    )
    eval_loaders = [
        base.loader(*partition, config.batch_size, False, base.dseed(seed, name, "loader"))
        for partition, name in zip(partitions, ("train", "val", "test"), strict=True)
    ]

    best_val_ba = -np.inf
    best_val_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        train_loss_sum = 0.0
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            layers = model.layer_features(X)
            spikes = layers[model.last_layer]
            loss, logits = model.loss_logits(spikes, lengths, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(y)
            train_true.append(y.cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = base.metrics(np.concatenate(train_true), np.concatenate(train_pred))
        val_metrics = evaluate_native(model, eval_loaders[1], device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / len(data.ytr),
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
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
        raise RuntimeError("No best checkpoint was selected")
    model.load_state_dict(best_state)

    final = [evaluate_native(model, data_loader, device) for data_loader in eval_loaders]
    result: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "architecture": architecture,
        "architecture_spec": _architecture_provenance(architecture),
        "objective": objective,
        "seed": int(seed),
        "best_epoch": best_epoch,
        "history": history,
    }
    for split, split_metrics in zip(("train", "val", "test"), final, strict=True):
        for key in ("loss", "accuracy", "balanced_accuracy", "macro_f1"):
            result[f"{split}_{key}"] = float(split_metrics[key])
    result["train_test_ba_gap"] = (
        float(result["train_balanced_accuracy"]) - float(result["test_balanced_accuracy"])
    )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "provenance": provenance,
            "result": result,
            "state_dict": best_state,
        },
        checkpoint,
    )
    return result


def _load_checkpoint(
    repo_root: Path,
    root: Path,
    architecture: str,
    objective: str,
    seed: int,
) -> dict[str, object]:
    if architecture == "A":
        path = prev.training_checkpoint_path(
            prev.results_dir(repo_root),
            BASELINE_ARCHITECTURE,
            objective,
            seed,
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing Experiment 3.0.2-C baseline checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("experiment_id") != BASELINE_EXPERIMENT_ID:
            raise ValueError(f"Wrong 3.0.2 experiment id in {path}")
        if payload.get("protocol_version") != BASELINE_PROTOCOL_VERSION:
            raise ValueError(f"Wrong 3.0.2 protocol version in {path}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"Missing 3.0.2 result in {path}")
        if (
            str(result.get("architecture")) != BASELINE_ARCHITECTURE
            or str(result.get("objective")) != objective
            or int(result.get("seed")) != seed
        ):
            raise ValueError(f"3.0.2-C baseline identity mismatch: {path}")
        return {"path": path, "payload": payload}

    path = training_checkpoint_path(root, architecture, objective, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing Experiment 3.0.3 checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong experiment id in {path}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong protocol version in {path}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"Missing result in {path}")
    if (
        str(result.get("architecture")) != architecture
        or str(result.get("objective")) != objective
        or int(result.get("seed")) != seed
    ):
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    return {"path": path, "payload": payload}


def load_model_for_evaluation(
    repo_root: Path,
    root: Path,
    architecture: str,
    objective: str,
    seed: int,
    data: base.Data,
    device: torch.device,
) -> tuple[L3AblationNet, dict[str, object]]:
    loaded = _load_checkpoint(repo_root, root, architecture, objective, seed)
    payload = loaded["payload"]
    model = L3AblationNet(
        architecture,
        objective,
        len(data.labels),
        data.T,
        data.fs,
        data.bin_steps,
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _masked_count(spikes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = base.mask(lengths, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
    return (spikes * valid).sum(dim=1)


def _partition_features(
    model: L3AblationNet,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, object]:
    y_parts: list[np.ndarray] = []
    whole_parts: dict[tuple[str, str], list[np.ndarray]] = {}
    subgroup_parts: dict[tuple[str, int, str], list[np.ndarray]] = {}
    whole_spikes: dict[str, float] = {layer: 0.0 for layer in model.layer_order}
    subgroup_spikes: dict[tuple[str, int], float] = {}
    valid_steps = 0.0

    with torch.no_grad():
        for X, y, lengths in data_loader:
            X = X.to(device)
            lengths = lengths.to(device)
            layers = model.layer_features(X)
            y_parts.append(y.numpy())
            valid_steps += float(lengths.sum().item())

            for layer in model.layer_order:
                spikes = layers[layer]
                full = _masked_count(spikes, lengths)
                fixed = base.fixed_counts(spikes, lengths, model.bin_steps).flatten(start_dim=1)
                whole_parts.setdefault((layer, "full_count"), []).append(full.cpu().numpy())
                whole_parts.setdefault((layer, "fixed250"), []).append(fixed.cpu().numpy())
                whole_spikes[layer] += float(full.sum().item())

                slices = model.layer_slices[layer]
                if len(slices) <= 1:
                    continue
                for shift, sl in slices.items():
                    subgroup = spikes[:, :, sl]
                    subgroup_full = _masked_count(subgroup, lengths)
                    subgroup_fixed = base.fixed_counts(
                        subgroup, lengths, model.bin_steps
                    ).flatten(start_dim=1)
                    subgroup_parts.setdefault(
                        (layer, int(shift), "full_count"), []
                    ).append(subgroup_full.cpu().numpy())
                    subgroup_parts.setdefault(
                        (layer, int(shift), "fixed250"), []
                    ).append(subgroup_fixed.cpu().numpy())
                    subgroup_spikes[(layer, int(shift))] = (
                        subgroup_spikes.get((layer, int(shift)), 0.0)
                        + float(subgroup_full.sum().item())
                    )

    y = np.concatenate(y_parts)
    whole = {key: np.concatenate(parts) for key, parts in whole_parts.items()}
    subgroup = {key: np.concatenate(parts) for key, parts in subgroup_parts.items()}
    whole_rates = {
        layer: whole_spikes[layer] / max(valid_steps * model.layer_widths[layer], 1.0)
        for layer in model.layer_order
    }
    subgroup_rates: dict[tuple[str, int], float] = {}
    for layer in model.layer_order:
        slices = model.layer_slices[layer]
        if len(slices) <= 1:
            continue
        for shift, sl in slices.items():
            width = sl.stop - sl.start
            subgroup_rates[(layer, int(shift))] = subgroup_spikes[(layer, int(shift))] / max(
                valid_steps * width, 1.0
            )
    return {
        "y": y,
        "whole": whole,
        "subgroup": subgroup,
        "whole_rates": whole_rates,
        "subgroup_rates": subgroup_rates,
    }


def run_evaluation_one(
    architecture: str,
    objective: str,
    seed: int,
    data: base.Data,
    config: Config,
) -> dict[str, object]:
    out_path = evaluation_path(config.results_dir, architecture, objective, seed)
    if config.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = (PROTOCOL_VERSION, architecture, objective, seed)
        actual = (
            str(payload.get("protocol_version")),
            str(payload.get("architecture")),
            str(payload.get("objective")),
            int(payload.get("seed")),
        )
        if actual != expected:
            raise ValueError(f"Evaluation identity mismatch in {out_path}: {actual} != {expected}")
        return payload

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model, checkpoint_payload = load_model_for_evaluation(
        config.repo_root,
        config.results_dir,
        architecture,
        objective,
        seed,
        data,
        device,
    )

    partitions = [
        (data.Xtr, data.ytr, data.ltr),
        (data.Xva, data.yva, data.lva),
        (data.Xte, data.yte, data.lte),
    ]
    loaders = [
        base.loader(*partition, config.batch_size, False, base.dseed(seed, split, "eval303"))
        for partition, split in zip(partitions, ("train", "val", "test"), strict=True)
    ]
    train, val, test = [
        _partition_features(model, data_loader, device)
        for data_loader in loaders
    ]

    layer_probe_rows: list[dict[str, object]] = []
    for layer in model.layer_order:
        for probe_type in PROBE_TYPES:
            key = (layer, probe_type)
            metrics = prev._probe_metrics(
                train["whole"][key], train["y"],
                val["whole"][key], val["y"],
                test["whole"][key], test["y"],
                base.dseed(seed, "probe303", architecture, objective, layer, probe_type),
            )
            layer_probe_rows.append(
                {
                    "layer": layer,
                    "probe_type": probe_type,
                    **metrics,
                }
            )

    subgroup_probe_rows: list[dict[str, object]] = []
    for key in sorted(train["subgroup"]):
        layer, shift, probe_type = key
        metrics = prev._probe_metrics(
            train["subgroup"][key], train["y"],
            val["subgroup"][key], val["y"],
            test["subgroup"][key], test["y"],
            base.dseed(seed, "subprobe303", architecture, objective, layer, shift, probe_type),
        )
        sl = model.layer_slices[layer][shift]
        subgroup_probe_rows.append(
            {
                "layer": layer,
                "shift": int(shift),
                "tau_syn_ms": base.tau_ms(int(shift), data.fs),
                "group_neurons": int(sl.stop - sl.start),
                "probe_type": probe_type,
                **metrics,
            }
        )

    firing_rows: list[dict[str, object]] = []
    for split, partition in zip(("train", "val", "test"), (train, val, test), strict=True):
        for layer in model.layer_order:
            firing_rows.append(
                {
                    "split": split,
                    "scope": "whole_layer",
                    "layer": layer,
                    "shift": -1,
                    "tau_syn_ms": None,
                    "neurons": int(model.layer_widths[layer]),
                    "firing_rate": float(partition["whole_rates"][layer]),
                }
            )
        for (layer, shift), rate in sorted(partition["subgroup_rates"].items()):
            sl = model.layer_slices[layer][shift]
            firing_rows.append(
                {
                    "split": split,
                    "scope": "tau_subgroup",
                    "layer": layer,
                    "shift": int(shift),
                    "tau_syn_ms": base.tau_ms(int(shift), data.fs),
                    "neurons": int(sl.stop - sl.start),
                    "firing_rate": float(rate),
                }
            )

    native_result = checkpoint_payload["result"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": architecture,
        "architecture_spec": _architecture_provenance(architecture),
        "objective": objective,
        "seed": int(seed),
        "last_layer": model.last_layer,
        "source_experiment": (
            BASELINE_EXPERIMENT_ID if architecture == "A" else EXPERIMENT_ID
        ),
        "source_protocol": (
            BASELINE_PROTOCOL_VERSION if architecture == "A" else PROTOCOL_VERSION
        ),
        "source_architecture": (
            BASELINE_ARCHITECTURE if architecture == "A" else architecture
        ),
        "native_best_epoch": int(native_result["best_epoch"]),
        "layer_probes": layer_probe_rows,
        "subgroup_probes": subgroup_probe_rows,
        "firing_rates": firing_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def native_result_and_history(
    repo_root: Path,
    root: Path,
    architecture: str,
    objective: str,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    loaded = _load_checkpoint(repo_root, root, architecture, objective, seed)
    result = loaded["payload"]["result"]
    native = {
        "architecture": architecture,
        "objective": objective,
        "seed": int(seed),
        "best_epoch": int(result["best_epoch"]),
        "last_layer": last_layer_name(architecture),
    }
    for key in (
        "train_loss",
        "train_accuracy",
        "train_balanced_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "test_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "train_test_ba_gap",
    ):
        native[key] = float(result[key])
    history = [
        {
            "architecture": architecture,
            "objective": objective,
            "seed": int(seed),
            **row,
        }
        for row in result["history"]
    ]
    return native, history
