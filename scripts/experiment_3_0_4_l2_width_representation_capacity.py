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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_0_2_hidden_multitau_architectures as probe_base
from scripts import experiment_3_0_3_l3_bottleneck_ablation as prev


EXPERIMENT_ID = "experiment_3_0_4_l2_width_representation_capacity"
PROTOCOL_VERSION = "l2_width_v1"

BASELINE_EXPERIMENT_ID = prev.EXPERIMENT_ID
BASELINE_PROTOCOL_VERSION = prev.PROTOCOL_VERSION
BASELINE_ARCHITECTURE = "B"
BASELINE_WIDTH = 128

OBJECTIVES = base.OBJECTIVES
SEEDS = base.SEEDS
EPOCHS = base.EPOCHS
BATCH_SIZE = base.BATCH_SIZE
LR = base.LR
N_REL = base.N_REL
FIXED_MS = base.FIXED_MS
EVENT_CHANNELS = base.EVENT_CHANNELS

L1_WIDTH = 128
L1_SHIFTS = (2, 3, 4)
L2_SHIFTS = (2, 3, 4)
L2_WIDTHS = (32, 64, 128, 256)
TRAIN_WIDTHS = (32, 64, 256)
EVAL_WIDTHS = L2_WIDTHS

PCA_DIM = 128
RAW_PROBE_TYPES = ("full_count", "fixed250")
LAYER_PROBE_TYPES = ("full_count", "fixed250", "fixed250_pca128")
SUBGROUP_PROBE_TYPES = RAW_PROBE_TYPES
PROBE_C_GRID = probe_base.PROBE_C_GRID
balanced_group_slices = probe_base.balanced_group_slices
aggregate_mean_sd = probe_base.aggregate_mean_sd

EXPECTED_TRAIN_RUNS = len(TRAIN_WIDTHS) * len(OBJECTIVES) * len(SEEDS)
EXPECTED_EVAL_RUNS = len(EVAL_WIDTHS) * len(OBJECTIVES) * len(SEEDS)


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


def train_specs() -> list[tuple[int, str, int]]:
    return [
        (width, objective, seed)
        for width in TRAIN_WIDTHS
        for objective in OBJECTIVES
        for seed in SEEDS
    ]


def eval_specs() -> list[tuple[int, str, int]]:
    return [
        (width, objective, seed)
        for width in EVAL_WIDTHS
        for objective in OBJECTIVES
        for seed in SEEDS
    ]


def _alpha_vector(width: int, shifts: tuple[int, ...]) -> torch.Tensor:
    values = torch.empty(width, dtype=torch.float32)
    for shift, sl in balanced_group_slices(width, shifts).items():
        values[sl] = base.alpha(int(shift))
    return values


def head_input_dim(width: int, objective: str, T: int, bin_steps: int) -> int:
    if objective == "timestep_ce":
        return int(width)
    if objective == "relative10_sequence_ce":
        return int(width * N_REL)
    if objective == "fixed250_sequence_ce":
        return int(width * math.ceil(T / bin_steps))
    raise ValueError(f"Unknown objective: {objective}")


def head_parameter_count(
    width: int,
    objective: str,
    T: int,
    bin_steps: int,
    n_classes: int,
) -> int:
    input_dim = head_input_dim(width, objective, T, bin_steps)
    return int(input_dim * n_classes + n_classes)


def configuration_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for width in EVAL_WIDTHS:
        for layer, layer_width, shifts in (
            ("L1", L1_WIDTH, L1_SHIFTS),
            ("L2", width, L2_SHIFTS),
        ):
            for shift, sl in balanced_group_slices(layer_width, shifts).items():
                rows.append(
                    {
                        "l2_width": int(width),
                        "layer": layer,
                        "layer_width": int(layer_width),
                        "shifts": ",".join(map(str, shifts)),
                        "shift": int(shift),
                        "group_neurons": int(sl.stop - sl.start),
                        "tau_syn_ms": base.tau_ms(int(shift), base.EXPECTED_FS),
                        "source_experiment": (
                            BASELINE_EXPERIMENT_ID
                            if width == BASELINE_WIDTH
                            else EXPERIMENT_ID
                        ),
                        "source_architecture": (
                            BASELINE_ARCHITECTURE
                            if width == BASELINE_WIDTH
                            else f"W{width}"
                        ),
                    }
                )
    return rows


class L2WidthNet(nn.Module):
    def __init__(
        self,
        l2_width: int,
        objective: str,
        n_classes: int,
        T: int,
        fs: float,
        bin_steps: int,
    ) -> None:
        super().__init__()
        if l2_width not in EVAL_WIDTHS:
            raise ValueError(f"Unsupported L2 width: {l2_width}")

        self.l2_width = int(l2_width)
        self.objective = objective
        self.T = int(T)
        self.bin_steps = int(bin_steps)
        self.last_layer = "L2"
        self.layer_order = ("L1", "L2")
        self.layer_widths = {"L1": L1_WIDTH, "L2": self.l2_width}
        self.layer_shifts = {"L1": L1_SHIFTS, "L2": L2_SHIFTS}
        self.layer_slices = {
            layer: balanced_group_slices(self.layer_widths[layer], self.layer_shifts[layer])
            for layer in self.layer_order
        }

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

        # Construction order matches Experiment 3.0.3-B through L2, so W128
        # can strict-load the reused checkpoint. Across widths, L1 starts from
        # the same seeded weights; L2 uses the same seed/fan-in with a different
        # output shape, so native width comparisons still require probe controls.
        self.f1 = nn.Linear(EVENT_CHANNELS, L1_WIDTH, bias=False)
        self.l1 = make_lif(L1_WIDTH, L1_SHIFTS)
        self.f2 = nn.Linear(L1_WIDTH, self.l2_width, bias=False)
        self.l2 = make_lif(self.l2_width, L2_SHIFTS)

        input_dim = head_input_dim(self.l2_width, objective, T, bin_steps)
        self.head = nn.Linear(input_dim, n_classes, bias=True)

    def layer_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, T, _ = x.shape
        syn1 = torch.zeros(batch, L1_WIDTH, device=x.device, dtype=x.dtype)
        mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, self.l2_width, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(syn2)

        seq1: list[torch.Tensor] = []
        seq2: list[torch.Tensor] = []
        for timestep in range(T):
            s1, syn1, mem1 = self.l1(self.f1(x[:, timestep]), syn1, mem1)
            s2, syn2, mem2 = self.l2(self.f2(s1), syn2, mem2)
            seq1.append(s1)
            seq2.append(s2)

        return {
            "L1": torch.stack(seq1, dim=1),
            "L2": torch.stack(seq2, dim=1),
        }

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
        elif self.objective == "fixed250_sequence_ce":
            features = base.fixed_counts(spikes, lengths, self.bin_steps)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        logits = self.head(features.flatten(start_dim=1))
        return F.cross_entropy(logits, y), logits


def evaluate_native(
    model: L2WidthNet,
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
            spikes = model.layer_features(X)["L2"]
            loss, logits = model.loss_logits(spikes, lengths, y)
            losses.append(float(loss.item()) * len(y))
            y_true_parts.append(y.cpu().numpy())
            y_pred_parts.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    out = base.metrics(y_true, y_pred)
    out["loss"] = float(sum(losses) / len(y_true))
    return out


def training_checkpoint_path(root: Path, width: int, objective: str, seed: int) -> Path:
    return root / "checkpoints" / f"width{width}__{objective}__seed{seed}.pt"


def evaluation_path(root: Path, width: int, objective: str, seed: int) -> Path:
    return root / "evaluations" / f"width{width}__{objective}__seed{seed}.json"


def _provenance(
    width: int,
    objective: str,
    seed: int,
    data: base.Data,
    config: Config,
) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "l2_width": int(width),
        "l1_width": L1_WIDTH,
        "l1_shifts": L1_SHIFTS,
        "l2_shifts": L2_SHIFTS,
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
        "pca_dim": PCA_DIM,
        "head_feature_dim": head_input_dim(width, objective, data.T, data.bin_steps),
        "head_parameter_count": head_parameter_count(
            width, objective, data.T, data.bin_steps, len(data.labels)
        ),
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
    width: int,
    objective: str,
    seed: int,
    data: base.Data,
    config: Config,
) -> dict[str, object]:
    if width not in TRAIN_WIDTHS:
        raise ValueError(f"L2 width {width} is not trained in Experiment 3.0.4")
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}")

    checkpoint = training_checkpoint_path(config.results_dir, width, objective, seed)
    provenance = _provenance(width, objective, seed, data, config)
    if config.resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _validate_training_checkpoint(payload, provenance, checkpoint)
        return payload["result"]

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)

    # W128 is reused from 3.0.3-B, which used this exact seed contract.
    # New widths keep the same f1 initialization and data-loader ordering.
    base.seed_all(base.dseed(seed, "shared_backbone_init"))
    model = L2WidthNet(
        width,
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
        base.loader(
            *partition,
            config.batch_size,
            False,
            base.dseed(seed, split, "loader"),
        )
        for partition, split in zip(
            partitions,
            ("train", "val", "test"),
            strict=True,
        )
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

            spikes = model.layer_features(X)["L2"]
            loss, logits = model.loss_logits(spikes, lengths, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(y)
            train_true.append(y.cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = base.metrics(
            np.concatenate(train_true),
            np.concatenate(train_pred),
        )
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

    final = [
        evaluate_native(model, data_loader, device)
        for data_loader in eval_loaders
    ]
    result: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "l2_width": int(width),
        "objective": objective,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "head_feature_dim": int(model.head.in_features),
        "head_parameter_count": int(
            model.head.weight.numel() + model.head.bias.numel()
        ),
        "history": history,
    }
    for split, split_metrics in zip(
        ("train", "val", "test"),
        final,
        strict=True,
    ):
        for key in ("loss", "accuracy", "balanced_accuracy", "macro_f1"):
            result[f"{split}_{key}"] = float(split_metrics[key])
    result["train_test_ba_gap"] = (
        float(result["train_balanced_accuracy"])
        - float(result["test_balanced_accuracy"])
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
    width: int,
    objective: str,
    seed: int,
) -> dict[str, object]:
    if width == BASELINE_WIDTH:
        path = prev.training_checkpoint_path(
            prev.results_dir(repo_root),
            BASELINE_ARCHITECTURE,
            objective,
            seed,
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Experiment 3.0.3-B W128 baseline checkpoint: {path}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("experiment_id") != BASELINE_EXPERIMENT_ID:
            raise ValueError(f"Wrong 3.0.3 experiment id in {path}")
        if payload.get("protocol_version") != BASELINE_PROTOCOL_VERSION:
            raise ValueError(f"Wrong 3.0.3 protocol version in {path}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"Missing 3.0.3 result in {path}")
        if (
            str(result.get("architecture")) != BASELINE_ARCHITECTURE
            or str(result.get("objective")) != objective
            or int(result.get("seed")) != seed
        ):
            raise ValueError(f"3.0.3-B baseline identity mismatch: {path}")
        return {"path": path, "payload": payload}

    path = training_checkpoint_path(root, width, objective, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing Experiment 3.0.4 checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong experiment id in {path}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong protocol version in {path}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"Missing result in {path}")
    if (
        int(result.get("l2_width")) != width
        or str(result.get("objective")) != objective
        or int(result.get("seed")) != seed
    ):
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    return {"path": path, "payload": payload}


def load_model_for_evaluation(
    repo_root: Path,
    root: Path,
    width: int,
    objective: str,
    seed: int,
    data: base.Data,
    device: torch.device,
) -> tuple[L2WidthNet, dict[str, object]]:
    loaded = _load_checkpoint(
        repo_root,
        root,
        width,
        objective,
        seed,
    )
    payload = loaded["payload"]
    model = L2WidthNet(
        width,
        objective,
        len(data.labels),
        data.T,
        data.fs,
        data.bin_steps,
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _masked_count(
    spikes: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    valid = base.mask(
        lengths,
        spikes.shape[1],
    ).to(spikes.dtype).unsqueeze(-1)
    return (spikes * valid).sum(dim=1)


def _partition_features(
    model: L2WidthNet,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, object]:
    y_parts: list[np.ndarray] = []
    whole_parts: dict[tuple[str, str], list[np.ndarray]] = {}
    subgroup_parts: dict[tuple[str, int, str], list[np.ndarray]] = {}
    whole_spikes = {layer: 0.0 for layer in model.layer_order}
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
                fixed = base.fixed_counts(
                    spikes,
                    lengths,
                    model.bin_steps,
                ).flatten(start_dim=1)

                whole_parts.setdefault(
                    (layer, "full_count"),
                    [],
                ).append(full.cpu().numpy())
                whole_parts.setdefault(
                    (layer, "fixed250"),
                    [],
                ).append(fixed.cpu().numpy())
                whole_spikes[layer] += float(full.sum().item())

                for shift, sl in model.layer_slices[layer].items():
                    subgroup = spikes[:, :, sl]
                    subgroup_full = _masked_count(subgroup, lengths)
                    subgroup_fixed = base.fixed_counts(
                        subgroup,
                        lengths,
                        model.bin_steps,
                    ).flatten(start_dim=1)

                    subgroup_parts.setdefault(
                        (layer, int(shift), "full_count"),
                        [],
                    ).append(subgroup_full.cpu().numpy())
                    subgroup_parts.setdefault(
                        (layer, int(shift), "fixed250"),
                        [],
                    ).append(subgroup_fixed.cpu().numpy())
                    subgroup_spikes[(layer, int(shift))] = (
                        subgroup_spikes.get((layer, int(shift)), 0.0)
                        + float(subgroup_full.sum().item())
                    )

    y = np.concatenate(y_parts)
    whole = {
        key: np.concatenate(parts)
        for key, parts in whole_parts.items()
    }
    subgroup = {
        key: np.concatenate(parts)
        for key, parts in subgroup_parts.items()
    }

    whole_rates = {
        layer: whole_spikes[layer]
        / max(valid_steps * model.layer_widths[layer], 1.0)
        for layer in model.layer_order
    }
    whole_total_spikes_per_timestep = {
        layer: whole_spikes[layer] / max(valid_steps, 1.0)
        for layer in model.layer_order
    }

    subgroup_rates: dict[tuple[str, int], float] = {}
    subgroup_total_spikes_per_timestep: dict[tuple[str, int], float] = {}
    for layer in model.layer_order:
        for shift, sl in model.layer_slices[layer].items():
            width = sl.stop - sl.start
            total = subgroup_spikes[(layer, int(shift))]
            subgroup_rates[(layer, int(shift))] = total / max(
                valid_steps * width,
                1.0,
            )
            subgroup_total_spikes_per_timestep[(layer, int(shift))] = (
                total / max(valid_steps, 1.0)
            )

    return {
        "y": y,
        "whole": whole,
        "subgroup": subgroup,
        "whole_rates": whole_rates,
        "whole_total_spikes_per_timestep": whole_total_spikes_per_timestep,
        "subgroup_rates": subgroup_rates,
        "subgroup_total_spikes_per_timestep": subgroup_total_spikes_per_timestep,
    }


def _raw_probe_metrics(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
) -> dict[str, float | int | None]:
    metrics = probe_base._probe_metrics(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        seed,
    )
    return {
        "input_feature_dim": int(train_x.shape[1]),
        "pca_explained_variance_ratio_sum": None,
        **metrics,
    }


def _pca_probe_metrics(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
) -> dict[str, float | int]:
    max_components = min(train_x.shape[0] - 1, train_x.shape[1])
    if PCA_DIM > max_components:
        raise ValueError(
            f"PCA_DIM={PCA_DIM} exceeds allowable components={max_components}"
        )

    input_scaler = StandardScaler().fit(train_x)
    train_scaled = input_scaler.transform(train_x)
    val_scaled = input_scaler.transform(val_x)
    test_scaled = input_scaler.transform(test_x)

    pca = PCA(
        n_components=PCA_DIM,
        svd_solver="randomized",
        random_state=seed,
    ).fit(train_scaled)
    train_pca = pca.transform(train_scaled)
    val_pca = pca.transform(val_scaled)
    test_pca = pca.transform(test_scaled)

    output_scaler = StandardScaler().fit(train_pca)
    train_z = output_scaler.transform(train_pca)
    val_z = output_scaler.transform(val_pca)
    test_z = output_scaler.transform(test_pca)

    best: tuple[float, float, LogisticRegression] | None = None
    for C in PROBE_C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=5000,
            solver="lbfgs",
            random_state=seed,
        ).fit(train_z, train_y)
        val_ba = float(
            balanced_accuracy_score(
                val_y,
                classifier.predict(val_z),
            )
        )
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)

    if best is None:
        raise RuntimeError("No PCA probe candidate was selected")

    val_ba, C, classifier = best
    pred = classifier.predict(test_z)
    return {
        "input_feature_dim": int(train_x.shape[1]),
        "feature_dim": int(PCA_DIM),
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "probe_C": C,
        "probe_val_balanced_accuracy": val_ba,
        "probe_test_balanced_accuracy": float(
            balanced_accuracy_score(test_y, pred)
        ),
        "probe_test_macro_f1": float(
            f1_score(
                test_y,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "probe_test_accuracy": float(
            accuracy_score(test_y, pred)
        ),
    }


def run_evaluation_one(
    width: int,
    objective: str,
    seed: int,
    data: base.Data,
    config: Config,
) -> dict[str, object]:
    if width not in EVAL_WIDTHS:
        raise ValueError(f"Unknown L2 width: {width}")
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}")

    out_path = evaluation_path(
        config.results_dir,
        width,
        objective,
        seed,
    )
    if config.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = (
            PROTOCOL_VERSION,
            width,
            objective,
            seed,
        )
        actual = (
            str(payload.get("protocol_version")),
            int(payload.get("l2_width")),
            str(payload.get("objective")),
            int(payload.get("seed")),
        )
        if actual != expected:
            raise ValueError(
                f"Evaluation identity mismatch in {out_path}: "
                f"{actual} != {expected}"
            )
        return payload

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model, checkpoint_payload = load_model_for_evaluation(
        config.repo_root,
        config.results_dir,
        width,
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
        base.loader(
            *partition,
            config.batch_size,
            False,
            base.dseed(seed, split, "eval304"),
        )
        for partition, split in zip(
            partitions,
            ("train", "val", "test"),
            strict=True,
        )
    ]
    train, val, test = [
        _partition_features(
            model,
            data_loader,
            device,
        )
        for data_loader in loaders
    ]

    layer_probe_rows: list[dict[str, object]] = []
    for layer in model.layer_order:
        for probe_type in RAW_PROBE_TYPES:
            key = (layer, probe_type)
            metrics = _raw_probe_metrics(
                train["whole"][key],
                train["y"],
                val["whole"][key],
                val["y"],
                test["whole"][key],
                test["y"],
                base.dseed(
                    seed,
                    "probe304",
                    width,
                    objective,
                    layer,
                    probe_type,
                ),
            )
            layer_probe_rows.append(
                {
                    "layer": layer,
                    "probe_type": probe_type,
                    **metrics,
                }
            )

        key = (layer, "fixed250")
        metrics = _pca_probe_metrics(
            train["whole"][key],
            train["y"],
            val["whole"][key],
            val["y"],
            test["whole"][key],
            test["y"],
            base.dseed(
                seed,
                "probe304",
                width,
                objective,
                layer,
                "fixed250_pca128",
            ),
        )
        layer_probe_rows.append(
            {
                "layer": layer,
                "probe_type": "fixed250_pca128",
                **metrics,
            }
        )

    subgroup_probe_rows: list[dict[str, object]] = []
    for key in sorted(train["subgroup"]):
        layer, shift, probe_type = key
        metrics = _raw_probe_metrics(
            train["subgroup"][key],
            train["y"],
            val["subgroup"][key],
            val["y"],
            test["subgroup"][key],
            test["y"],
            base.dseed(
                seed,
                "subprobe304",
                width,
                objective,
                layer,
                shift,
                probe_type,
            ),
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
    for split, partition in zip(
        ("train", "val", "test"),
        (train, val, test),
        strict=True,
    ):
        for layer in model.layer_order:
            firing_rows.append(
                {
                    "split": split,
                    "scope": "whole_layer",
                    "layer": layer,
                    "shift": -1,
                    "tau_syn_ms": None,
                    "neurons": int(model.layer_widths[layer]),
                    "firing_rate": float(
                        partition["whole_rates"][layer]
                    ),
                    "total_spikes_per_timestep": float(
                        partition["whole_total_spikes_per_timestep"][layer]
                    ),
                }
            )

        for key, rate in sorted(
            partition["subgroup_rates"].items()
        ):
            layer, shift = key
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
                    "total_spikes_per_timestep": float(
                        partition[
                            "subgroup_total_spikes_per_timestep"
                        ][key]
                    ),
                }
            )

    native_result = checkpoint_payload["result"]
    state_dict = checkpoint_payload["state_dict"]
    head_weight = state_dict["head.weight"]
    head_bias = state_dict["head.bias"]

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "l2_width": int(width),
        "objective": objective,
        "seed": int(seed),
        "last_layer": "L2",
        "pca_dim": PCA_DIM,
        "source_experiment": (
            BASELINE_EXPERIMENT_ID
            if width == BASELINE_WIDTH
            else EXPERIMENT_ID
        ),
        "source_protocol": (
            BASELINE_PROTOCOL_VERSION
            if width == BASELINE_WIDTH
            else PROTOCOL_VERSION
        ),
        "source_architecture": (
            BASELINE_ARCHITECTURE
            if width == BASELINE_WIDTH
            else f"W{width}"
        ),
        "native_best_epoch": int(native_result["best_epoch"]),
        "head_feature_dim": int(head_weight.shape[1]),
        "head_parameter_count": int(
            head_weight.numel() + head_bias.numel()
        ),
        "layer_probes": layer_probe_rows,
        "subgroup_probes": subgroup_probe_rows,
        "firing_rates": firing_rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )
    return payload


def native_result_and_history(
    repo_root: Path,
    root: Path,
    width: int,
    objective: str,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    loaded = _load_checkpoint(
        repo_root,
        root,
        width,
        objective,
        seed,
    )
    payload = loaded["payload"]
    result = payload["result"]
    state_dict = payload["state_dict"]
    head_weight = state_dict["head.weight"]
    head_bias = state_dict["head.bias"]

    native: dict[str, object] = {
        "l2_width": int(width),
        "objective": objective,
        "seed": int(seed),
        "best_epoch": int(result["best_epoch"]),
        "last_layer": "L2",
        "source_experiment": (
            BASELINE_EXPERIMENT_ID
            if width == BASELINE_WIDTH
            else EXPERIMENT_ID
        ),
        "head_feature_dim": int(head_weight.shape[1]),
        "head_parameter_count": int(
            head_weight.numel() + head_bias.numel()
        ),
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
            "l2_width": int(width),
            "objective": objective,
            "seed": int(seed),
            **row,
        }
        for row in result["history"]
    ]
    return native, history
