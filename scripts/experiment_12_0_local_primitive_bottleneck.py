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
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73


EXPERIMENT_ID = "experiment_12_0_local_primitive_bottleneck"
PROTOCOL_VERSION = "local_primitive_bottleneck_v1"
ARCHITECTURE = exp73.ARCHITECTURE
SHIFTS = exp73.SHIFTS
SEEDS = exp73.SEEDS
HIDDEN_WIDTH = exp73.HIDDEN_WIDTH
PRIMITIVE_WIDTH = 16
BETA = 15.0 / 16.0
EPS = 1e-8
TEMPERATURE = 1.0
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE

TEMPORAL_MODES = ("t0", "raw", "ema")
DENSE_METHODS = ("r0b_analog", "r1_what", "r2_main_frozen", "r2_main_e2e")
POSTHOC_SOURCES = ("r0b_analog", "r2_main_frozen", "r2_main_e2e")
SPARSITY_QUANTILES = (0.20, 0.40, 0.60, 0.80, 0.90)


def _zero_preserving_rms(values: torch.Tensor, dim: int) -> torch.Tensor:
    mean_square = values.square().mean(dim=dim)
    safe_root = torch.sqrt(mean_square.clamp_min(EPS))
    return torch.where(mean_square > EPS, safe_root, torch.zeros_like(mean_square))


def _assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite tensor detected: {name}")


def _assert_finite_trainable_state(model: nn.Module, *, gradients: bool) -> None:
    kind = "gradient" if gradients else "parameter"
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.grad if gradients else parameter
        if gradients and value is None:
            continue
        if value is not None and not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite {kind} detected: {name}")


@dataclass(frozen=True)
class DenseSpec:
    temporal_mode: str
    method: str
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{ARCHITECTURE}__{self.temporal_mode}__{self.method}"
            f"__seed{self.seed}"
        )


@dataclass(frozen=True)
class PosthocSpec:
    temporal_mode: str
    source_method: str
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{ARCHITECTURE}__{self.temporal_mode}__{self.source_method}"
            f"__seed{self.seed}"
        )


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
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / EXPERIMENT_ID
        / PROTOCOL_VERSION
    )


def dense_specs() -> list[DenseSpec]:
    return [
        DenseSpec(mode, method, seed)
        for mode in TEMPORAL_MODES
        for method in DENSE_METHODS
        for seed in SEEDS
    ]


def posthoc_specs() -> list[PosthocSpec]:
    return [
        PosthocSpec(mode, source, seed)
        for mode in TEMPORAL_MODES
        for source in POSTHOC_SOURCES
        for seed in SEEDS
    ]


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _validate_dense(spec: DenseSpec) -> None:
    if spec.temporal_mode not in TEMPORAL_MODES:
        raise ValueError(spec.temporal_mode)
    if spec.method not in DENSE_METHODS:
        raise ValueError(spec.method)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def _validate_posthoc(spec: PosthocSpec) -> None:
    if spec.temporal_mode not in TEMPORAL_MODES:
        raise ValueError(spec.temporal_mode)
    if spec.source_method not in POSTHOC_SOURCES:
        raise ValueError(spec.source_method)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def _a2_checkpoint_path(repo_root: Path, seed: int) -> Path:
    source_spec = exp73.E2ESpec(seed, "linear", "wcce")
    return _path(
        exp73.results_dir(repo_root),
        "e2e_checkpoints",
        source_spec.key,
        ".pt",
    )


class PrimitiveBottleneckNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        fs: float,
        temporal_mode: str,
        method: str,
    ) -> None:
        super().__init__()
        if temporal_mode not in TEMPORAL_MODES:
            raise ValueError(temporal_mode)
        if method not in DENSE_METHODS:
            raise ValueError(method)
        self.temporal_mode = temporal_mode
        self.method = method
        self.n_classes = int(n_classes)
        self.backbone = exp73.Exp73Net("linear", n_classes, fs)
        self.backbone.output_linear.requires_grad_(False)
        self.primitive_projection = nn.Linear(
            HIDDEN_WIDTH, PRIMITIVE_WIDTH, bias=False
        )
        self.classifier = nn.Linear(PRIMITIVE_WIDTH, n_classes, bias=False)

    def load_a2_backbone(self, payload: dict[str, Any]) -> None:
        self.backbone.load_state_dict(payload["model_state_dict"], strict=True)
        self.backbone.output_linear.requires_grad_(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(trainable and name != "output_linear.weight")

    def _temporal_integrate(self, z: torch.Tensor) -> torch.Tensor:
        if self.temporal_mode == "t0":
            return z
        state = torch.zeros_like(z[:, 0])
        out: list[torch.Tensor] = []
        injection = 1.0 if self.temporal_mode == "raw" else (1.0 - BETA)
        for t in range(z.shape[1]):
            state = BETA * state + injection * z[:, t]
            out.append(state)
        return torch.stack(out, dim=1)

    @staticmethod
    def primitive_components(
        h: torch.Tensor,
        projection: nn.Linear,
    ) -> dict[str, torch.Tensor]:
        s = projection(h)
        centered = s - s.mean(dim=-1, keepdim=True)
        spread = _zero_preserving_rms(centered, dim=-1)
        normalized = centered / spread.unsqueeze(-1).clamp_min(EPS)
        q = F.softmax(normalized / TEMPERATURE, dim=-1)
        activity = _zero_preserving_rms(h, dim=-1)
        top2 = torch.topk(normalized, k=2, dim=-1).values
        confidence = top2[..., 0] - top2[..., 1]
        winner = q.argmax(dim=-1)
        return {
            "s": s,
            "centered": centered,
            "spread": spread,
            "normalized": normalized,
            "q": q,
            "activity": activity,
            "confidence": confidence,
            "winner": winner,
        }

    def representation(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux = self.primitive_components(h, self.primitive_projection)
        if self.method == "r0b_analog":
            routed = aux["s"]
        elif self.method == "r1_what":
            routed = aux["q"]
        elif self.method in {"r2_main_frozen", "r2_main_e2e"}:
            routed = aux["activity"].unsqueeze(-1) * aux["q"]
        else:
            raise ValueError(self.method)
        return routed, aux

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        backbone_tr = self.backbone.forward_trajectory(x)
        z = backbone_tr["hidden_spikes"][-1]
        h = self._temporal_integrate(z)
        routed, aux = self.representation(h)
        evidence = self.classifier(routed)
        return {
            "hidden_spikes": backbone_tr["hidden_spikes"],
            "z": z,
            "h": h,
            "routed": routed,
            "evidence": evidence,
            "primitive": aux,
        }


def _valid_sum(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = exp73._valid_mask(lengths, values.shape[1]).to(values.dtype)
    return (values * valid.unsqueeze(-1)).sum(dim=1)


def _loss_scores(
    evidence: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = _valid_sum(evidence, lengths)
    return F.cross_entropy(scores, y), scores


def _evaluate(
    model: PrimitiveBottleneckNet,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            tr = model.forward_trajectory(X)
            loss, scores = _loss_scores(tr["evidence"], lengths, y)
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)
    out = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    out["objective_loss"] = loss_sum / max(n_total, 1)
    return out


def _diagnostics(
    model: PrimitiveBottleneckNet,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    chunks: dict[str, list[torch.Tensor]] = {
        "activity": [],
        "spread": [],
        "confidence": [],
        "entropy": [],
        "winner": [],
    }
    model.eval()
    with torch.no_grad():
        for X, _, lengths in loader:
            X = X.to(device)
            lengths = lengths.to(device)
            tr = model.forward_trajectory(X)
            valid = exp73._valid_mask(lengths, X.shape[1])
            p = tr["primitive"]
            q = p["q"]
            entropy = -(q * torch.log(q.clamp_min(EPS))).sum(dim=-1)
            chunks["activity"].append(p["activity"][valid].cpu())
            chunks["spread"].append(p["spread"][valid].cpu())
            chunks["confidence"].append(p["confidence"][valid].cpu())
            chunks["entropy"].append(entropy[valid].cpu())
            chunks["winner"].append(p["winner"][valid].cpu())

    out: dict[str, float] = {}
    for key in ("activity", "spread", "confidence", "entropy"):
        values = torch.cat(chunks[key])
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(unbiased=False))
    winner = torch.cat(chunks["winner"])
    counts = torch.bincount(winner, minlength=PRIMITIVE_WIDTH).float()
    probs = counts / counts.sum().clamp_min(1)
    nz = probs > 0
    util_entropy = -(probs[nz] * torch.log(probs[nz])).sum()
    out["primitive_effective_k"] = float(torch.exp(util_entropy))
    out["primitive_used_count"] = float(nz.sum())
    return out


def _load_model(
    spec: DenseSpec,
    config: Config,
) -> tuple[PrimitiveBottleneckNet, exp3.Data]:
    data = exp3.prepare_data(config.repo_root)
    checkpoint_path = _a2_checkpoint_path(config.repo_root, spec.seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing Exp7.3 A2 checkpoint required by Exp12.0: {checkpoint_path}"
        )
    payload = torch.load(
        checkpoint_path, map_location=config.device, weights_only=False
    )
    model = PrimitiveBottleneckNet(
        len(data.labels), data.fs, spec.temporal_mode, spec.method
    ).to(config.device)
    model.load_a2_backbone(payload)
    model.set_backbone_trainable(spec.method == "r2_main_e2e")
    return model, data


def run_dense(
    spec: DenseSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    _validate_dense(spec)
    eval_path = _path(config.results_dir, "dense_evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "dense_checkpoints", spec.key, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp3.seed_all(exp3.dseed(spec.seed, EXPERIMENT_ID, spec.temporal_mode, "dense_pair"))
    model, data = _load_model(spec, config)

    head_params = list(model.primitive_projection.parameters()) + list(
        model.classifier.parameters()
    )
    if spec.method == "r2_main_e2e":
        backbone_params = [
            parameter
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.Adam(
            [
                {"params": head_params, "lr": exp72.LR},
                {"params": backbone_params, "lr": 0.1 * exp72.LR},
            ],
            weight_decay=exp72.WEIGHT_DECAY,
        )
    else:
        optimizer = torch.optim.Adam(
            head_params, lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY
        )

    train_loader = exp73._raw_loaders(
        data, spec.seed, config.batch_size, shuffle_train=True
    )["train"]
    eval_loaders = exp73._raw_loaders(
        data, spec.seed, config.batch_size, shuffle_train=False
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_total = 0
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            tr = model.forward_trajectory(X)
            loss, _ = _loss_scores(tr["evidence"], lengths, y)
            _assert_finite_tensor("training loss", loss)
            loss.backward()
            _assert_finite_trainable_state(model, gradients=True)
            optimizer.step()
            _assert_finite_trainable_state(model, gradients=False)
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate(model, eval_loaders["train"], device)
        val_metrics = _evaluate(model, eval_loaders["val"], device)
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["objective_loss"])
        if not math.isfinite(val_ba) or not math.isfinite(val_loss):
            raise RuntimeError(
                f"Non-finite validation metric for {spec.key}: "
                f"balanced_accuracy={val_ba}, objective_loss={val_loss}"
            )
        if exp73._checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = val_ba
            best_loss = val_loss
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
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "model_state_dict": best_state,
            "source_a2_checkpoint": str(_a2_checkpoint_path(config.repo_root, spec.seed)),
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state, strict=True)
    native = {
        split: _evaluate(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    diagnostics = {
        split: _diagnostics(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    l2_splits = {
        split: exp73._extract_l2(model.backbone, loader, device)
        for split, loader in eval_loaders.items()
    }
    probes = exp73._fit_probes(l2_splits, spec.seed, data.bin_steps)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "architecture": ARCHITECTURE,
            "shifts": [list(value) for value in SHIFTS],
            "single_user_split": True,
            "random_seeds": list(SEEDS),
            "primitive_width": PRIMITIVE_WIDTH,
            "beta": BETA,
            "temperature": TEMPERATURE,
            "aggregation": "valid_timestep_sum",
            "classifier_bias": False,
            "primitive_projection_bias": False,
            "backbone_start": "Exp7.3 A2 pretrained checkpoint",
            "backbone_trainable": spec.method == "r2_main_e2e",
            "backbone_lr_ratio": 0.1 if spec.method == "r2_main_e2e" else 0.0,
        },
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native,
        "diagnostics": diagnostics,
        "representation_probes": probes,
    }
    _save_json(eval_path, payload)

    history_path = _path(config.results_dir, "dense_histories", spec.key, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return payload


def _load_dense_model(
    spec: DenseSpec,
    config: Config,
) -> tuple[PrimitiveBottleneckNet, exp3.Data]:
    model, data = _load_model(spec, config)
    checkpoint_path = _path(config.results_dir, "dense_checkpoints", spec.key, ".pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, data


def _extract_primitive_split(
    model: PrimitiveBottleneckNet,
    loader: Iterable,
    device: torch.device,
) -> dict[str, np.ndarray]:
    routed_all: list[np.ndarray] = []
    z_all: list[np.ndarray] = []
    h_all: list[np.ndarray] = []
    s_all: list[np.ndarray] = []
    normalized_all: list[np.ndarray] = []
    q_all: list[np.ndarray] = []
    activity_all: list[np.ndarray] = []
    confidence_all: list[np.ndarray] = []
    winner_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    lengths_all: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            tr = model.forward_trajectory(X.to(device))
            p = tr["primitive"]
            routed_all.append(tr["routed"].cpu().numpy())
            z_all.append(tr["z"].cpu().numpy())
            h_all.append(tr["h"].cpu().numpy())
            s_all.append(p["s"].cpu().numpy())
            normalized_all.append(p["normalized"].cpu().numpy())
            q_all.append(p["q"].cpu().numpy())
            activity_all.append(p["activity"].cpu().numpy())
            confidence_all.append(p["confidence"].cpu().numpy())
            winner_all.append(p["winner"].cpu().numpy())
            y_all.append(y.numpy())
            lengths_all.append(lengths.numpy())
    return {
        "routed": np.concatenate(routed_all),
        "z": np.concatenate(z_all),
        "h": np.concatenate(h_all),
        "s": np.concatenate(s_all),
        "normalized": np.concatenate(normalized_all),
        "q": np.concatenate(q_all),
        "activity": np.concatenate(activity_all),
        "confidence": np.concatenate(confidence_all),
        "winner": np.concatenate(winner_all),
        "y": np.concatenate(y_all),
        "lengths": np.concatenate(lengths_all),
    }


def _aggregate_masked(
    routed: np.ndarray,
    mask: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    valid = np.arange(routed.shape[1])[None, :] < lengths[:, None]
    active = valid & mask
    return (routed * active[..., None]).sum(axis=1)


def _same_primitive_margin(normalized: np.ndarray, k: int) -> np.ndarray:
    selected = normalized[..., k]
    others = np.delete(normalized, k, axis=-1)
    return selected - others.max(axis=-1)


def _peak_mask(
    normalized: np.ndarray,
    winner: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
) -> np.ndarray:
    n, steps, _ = normalized.shape
    out = np.zeros((n, steps), dtype=bool)
    for i in range(n):
        stop = int(lengths[i])
        for t in range(2, stop):
            candidate = t - 1
            k = int(winner[i, candidate])
            margins = _same_primitive_margin(normalized[i, :stop], k)
            center = float(margins[candidate])
            if center <= threshold:
                continue
            if center > float(margins[candidate - 1]) and center >= float(margins[t]):
                out[i, candidate] = True
    return out


def _fit_feature_probe(
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    return exp01._fit_linear_probe(
        features["train"],
        labels["train"],
        features["val"],
        labels["val"],
        features["test"],
        labels["test"],
        seed,
    )


def _probe_row(
    representation: str,
    quantile: float,
    threshold: float,
    probe: dict[str, Any],
    split_data: dict[str, dict[str, np.ndarray]],
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    test_mask = masks["test"]
    test_lengths = split_data["test"]["lengths"]
    valid = np.arange(test_mask.shape[1])[None, :] < test_lengths[:, None]
    active = test_mask & valid
    events = active.sum(axis=1)
    return {
        "representation": representation,
        "quantile": quantile,
        "threshold": threshold,
        "test_ba": float(probe["test"]["balanced_accuracy"]),
        "test_accuracy": float(probe["test"]["accuracy"]),
        "test_macro_f1": float(probe["test"]["macro_f1"]),
        "active_fraction_test": float(active.sum() / max(valid.sum(), 1)),
        "events_per_sample_test_mean": float(events.mean()),
        "zero_event_fraction_test": float(np.mean(events == 0)),
    }


def _all_true_mask(values: np.ndarray) -> np.ndarray:
    return np.ones(values.shape[:2], dtype=bool)


def _raw_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted / TEMPERATURE)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def _fixed_transform_row(
    name: str,
    arrays: dict[str, np.ndarray],
    split_data: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    masks = {split: _all_true_mask(values) for split, values in arrays.items()}
    features = {
        split: _aggregate_masked(
            arrays[split], masks[split], split_data[split]["lengths"]
        )
        for split in arrays
    }
    probe = _fit_feature_probe(features, labels, seed)
    return _probe_row(
        name,
        0.0,
        float("-inf"),
        probe,
        split_data,
        masks,
    )


def _magnitude_only_features(
    split_data: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for split, values in split_data.items():
        activity = values["activity"]
        lengths = values["lengths"]
        valid = np.arange(activity.shape[1])[None, :] < lengths[:, None]
        masked = np.where(valid, activity, 0.0)
        summed = masked.sum(axis=1)
        mean = summed / np.maximum(lengths, 1)
        max_value = np.where(valid, activity, -np.inf).max(axis=1)
        max_value[~np.isfinite(max_value)] = 0.0
        features[split] = np.stack((summed, mean, max_value), axis=1)
    return features


def _count_only_features(
    masks: dict[str, np.ndarray],
    split_data: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for split, mask in masks.items():
        values = split_data[split]
        lengths = values["lengths"]
        valid = np.arange(mask.shape[1])[None, :] < lengths[:, None]
        active = mask & valid
        counts = active.sum(axis=1).astype(np.float64)
        total_magnitude = (
            values["activity"] * active.astype(values["activity"].dtype)
        ).sum(axis=1)
        mean_interval = np.zeros(len(mask), dtype=np.float64)
        for i in range(len(mask)):
            positions = np.flatnonzero(active[i])
            if len(positions) >= 2:
                mean_interval[i] = float(np.diff(positions).mean())
        features[split] = np.stack(
            (
                counts,
                total_magnitude,
                lengths.astype(np.float64),
                mean_interval,
            ),
            axis=1,
        )
    return features


def run_posthoc(
    spec: PosthocSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    _validate_posthoc(spec)
    output_path = _path(config.results_dir, "posthoc", spec.key, ".json")
    csv_path = _path(config.results_dir, "posthoc", spec.key, ".csv")
    if output_path.exists() and csv_path.exists() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))

    source_spec = DenseSpec(spec.temporal_mode, spec.source_method, spec.seed)
    model, data = _load_dense_model(source_spec, config)
    device = torch.device(config.device)
    loaders = exp73._raw_loaders(
        data, spec.seed, config.batch_size, shuffle_train=False
    )
    split_data = {
        split: _extract_primitive_split(model, loader, device)
        for split, loader in loaders.items()
    }
    labels = {split: values["y"] for split, values in split_data.items()}
    rows: list[dict[str, Any]] = []

    if spec.source_method == "r0b_analog":
        fixed_arrays: dict[str, dict[str, np.ndarray]] = {
            "r0_direct_l2": {
                split: values["z"] for split, values in split_data.items()
            },
            "r0a_integrated_128d": {
                split: values["h"] for split, values in split_data.items()
            },
            "r0b_analog_16d": {
                split: values["s"] for split, values in split_data.items()
            },
            "r1_main_what_posthoc": {
                split: values["q"] for split, values in split_data.items()
            },
            "r2_main_activity_x_what_posthoc": {
                split: values["activity"][..., None] * values["q"]
                for split, values in split_data.items()
            },
            "r1_legacy_raw_softmax_posthoc": {
                split: _raw_softmax(values["s"])
                for split, values in split_data.items()
            },
            "r2_legacy_rms_x_raw_softmax_posthoc": {
                split: np.sqrt(np.mean(values["s"] ** 2, axis=-1))[..., None]
                * _raw_softmax(values["s"])
                for split, values in split_data.items()
            },
        }
        for name, arrays in fixed_arrays.items():
            rows.append(
                _fixed_transform_row(
                    name,
                    arrays,
                    split_data,
                    labels,
                    exp3.dseed(
                        spec.seed,
                        EXPERIMENT_ID,
                        spec.key,
                        "fixed_transform",
                        name,
                    ),
                )
            )
    else:
        dense_masks = {
            split: np.ones_like(values["confidence"], dtype=bool)
            for split, values in split_data.items()
        }
        dense_features = {
            split: _aggregate_masked(
                values["routed"], dense_masks[split], values["lengths"]
            )
            for split, values in split_data.items()
        }
        dense_probe = _fit_feature_probe(
            dense_features,
            labels,
            exp3.dseed(spec.seed, EXPERIMENT_ID, spec.key, "dense_probe"),
        )
        rows.append(
            _probe_row(
                "r2_dense",
                0.0,
                float("-inf"),
                dense_probe,
                split_data,
                dense_masks,
            )
        )

        hard_token = {
            split: values["activity"][..., None]
            * np.eye(PRIMITIVE_WIDTH, dtype=np.float32)[values["winner"]]
            for split, values in split_data.items()
        }
        rows.append(
            _fixed_transform_row(
                "r2_hard_token",
                hard_token,
                split_data,
                labels,
                exp3.dseed(spec.seed, EXPERIMENT_ID, spec.key, "hard_token"),
            )
        )

        magnitude_features = _magnitude_only_features(split_data)
        magnitude_probe = _fit_feature_probe(
            magnitude_features,
            labels,
            exp3.dseed(spec.seed, EXPERIMENT_ID, spec.key, "magnitude_only"),
        )
        rows.append(
            {
                "representation": "magnitude_only",
                "quantile": 0.0,
                "threshold": float("-inf"),
                "test_ba": float(
                    magnitude_probe["test"]["balanced_accuracy"]
                ),
                "test_accuracy": float(magnitude_probe["test"]["accuracy"]),
                "test_macro_f1": float(magnitude_probe["test"]["macro_f1"]),
                "active_fraction_test": 1.0,
                "events_per_sample_test_mean": float("nan"),
                "zero_event_fraction_test": 0.0,
            }
        )

        train = split_data["train"]
        train_valid = (
            np.arange(train["confidence"].shape[1])[None, :]
            < train["lengths"][:, None]
        )
        train_conf = train["confidence"][train_valid]
        for quantile in SPARSITY_QUANTILES:
            threshold = float(np.quantile(train_conf, quantile))

            gate_masks = {
                split: values["confidence"] > threshold
                for split, values in split_data.items()
            }
            gate_features = {
                split: _aggregate_masked(
                    values["routed"], gate_masks[split], values["lengths"]
                )
                for split, values in split_data.items()
            }
            gate_probe = _fit_feature_probe(
                gate_features,
                labels,
                exp3.dseed(
                    spec.seed, EXPERIMENT_ID, spec.key, "r3", str(quantile)
                ),
            )
            rows.append(
                _probe_row(
                    "r3_gate",
                    quantile,
                    threshold,
                    gate_probe,
                    split_data,
                    gate_masks,
                )
            )

            peak_masks = {
                split: _peak_mask(
                    values["normalized"],
                    values["winner"],
                    values["lengths"],
                    threshold,
                )
                for split, values in split_data.items()
            }
            peak_features = {
                split: _aggregate_masked(
                    values["routed"], peak_masks[split], values["lengths"]
                )
                for split, values in split_data.items()
            }
            peak_probe = _fit_feature_probe(
                peak_features,
                labels,
                exp3.dseed(
                    spec.seed, EXPERIMENT_ID, spec.key, "r4", str(quantile)
                ),
            )
            rows.append(
                _probe_row(
                    "r4_peak",
                    quantile,
                    threshold,
                    peak_probe,
                    split_data,
                    peak_masks,
                )
            )

            count_features = _count_only_features(peak_masks, split_data)
            count_probe = _fit_feature_probe(
                count_features,
                labels,
                exp3.dseed(
                    spec.seed,
                    EXPERIMENT_ID,
                    spec.key,
                    "r4_count_only",
                    str(quantile),
                ),
            )
            count_row = _probe_row(
                "r4_count_only",
                quantile,
                threshold,
                count_probe,
                split_data,
                peak_masks,
            )
            rows.append(count_row)

    frame = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "source_dense_checkpoint": source_spec.key,
        "threshold_source": (
            "training confidence quantiles"
            if spec.source_method != "r0b_analog"
            else "not applicable"
        ),
        "quantiles": list(SPARSITY_QUANTILES),
        "selection_training": (
            "posthoc only; encoder and primitive projection frozen"
        ),
        "rows": rows,
    }
    _save_json(output_path, payload)
    return payload

def _dense_row(payload: dict[str, Any]) -> dict[str, Any]:
    spec = payload["spec"]
    test = payload["native_metrics"]["test"]
    probes = payload["representation_probes"]
    diag = payload["diagnostics"]["test"]
    return {
        "temporal_mode": spec["temporal_mode"],
        "method": spec["method"],
        "seed": int(spec["seed"]),
        "best_epoch": int(payload["best_epoch"]),
        "test_ba": float(test["balanced_accuracy"]),
        "test_accuracy": float(test["accuracy"]),
        "test_macro_f1": float(test["macro_f1"]),
        "l2_wholecount_probe_test_ba": float(
            probes["l2_wholecount_linear"]["metrics"]["test"]["balanced_accuracy"]
        ),
        "l2_fixed250_probe_test_ba": float(
            probes["l2_fixed250_linear"]["metrics"]["test"]["balanced_accuracy"]
        ),
        "activity_mean": float(diag["activity_mean"]),
        "spread_mean": float(diag["spread_mean"]),
        "confidence_mean": float(diag["confidence_mean"]),
        "entropy_mean": float(diag["entropy_mean"]),
        "primitive_effective_k": float(diag["primitive_effective_k"]),
    }


def finalize(config: Config) -> dict[str, Any]:
    dense_rows = []
    for spec in dense_specs():
        path = _path(config.results_dir, "dense_evaluations", spec.key, ".json")
        if not path.exists():
            raise FileNotFoundError(path)
        dense_rows.append(_dense_row(json.loads(path.read_text(encoding="utf-8"))))
    dense = pd.DataFrame(dense_rows)
    dense.to_csv(config.results_dir / "dense_runs.csv", index=False)
    dense_summary = (
        dense.groupby(["temporal_mode", "method"], sort=False)
        .agg(
            test_ba_mean=("test_ba", "mean"),
            test_ba_std=("test_ba", "std"),
            l2_wholecount_probe_test_ba_mean=("l2_wholecount_probe_test_ba", "mean"),
            l2_fixed250_probe_test_ba_mean=("l2_fixed250_probe_test_ba", "mean"),
            primitive_effective_k_mean=("primitive_effective_k", "mean"),
        )
        .reset_index()
    )
    dense_summary.to_csv(config.results_dir / "dense_summary.csv", index=False)

    posthoc_frames = []
    for spec in posthoc_specs():
        path = _path(config.results_dir, "posthoc", spec.key, ".csv")
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "seed", spec.seed)
        frame.insert(0, "source_method", spec.source_method)
        frame.insert(0, "temporal_mode", spec.temporal_mode)
        posthoc_frames.append(frame)
    posthoc = pd.concat(posthoc_frames, ignore_index=True)
    posthoc.to_csv(config.results_dir / "posthoc_runs.csv", index=False)
    posthoc_summary = (
        posthoc.groupby(
            ["temporal_mode", "source_method", "representation", "quantile"],
            sort=False,
        )
        .agg(
            test_ba_mean=("test_ba", "mean"),
            test_ba_std=("test_ba", "std"),
            active_fraction_test_mean=("active_fraction_test", "mean"),
            events_per_sample_test_mean=("events_per_sample_test_mean", "mean"),
            zero_event_fraction_test_mean=("zero_event_fraction_test", "mean"),
        )
        .reset_index()
    )
    posthoc_summary.to_csv(config.results_dir / "posthoc_summary.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS",
        "architecture": ARCHITECTURE,
        "shifts": [list(value) for value in SHIFTS],
        "single_user_split": True,
        "seeds": list(SEEDS),
        "temporal_modes": list(TEMPORAL_MODES),
        "dense_methods": list(DENSE_METHODS),
        "posthoc_sources": list(POSTHOC_SOURCES),
        "primitive_width": PRIMITIVE_WIDTH,
        "beta": BETA,
        "tau_ms_at_64hz": float(-(1000.0 / 64.0) / math.log(BETA)),
        "counts": {
            "dense_jobs": len(dense_specs()),
            "posthoc_jobs": len(posthoc_specs()),
        },
        "notes": {
            "raw_vs_ema": (
                "RAW and EMA use the same beta and temporal kernel shape; they differ "
                "only in injection scale and are retained as an explicit empirical case."
            ),
            "r3_r4": (
                "R3/R4 are post-hoc on fixed R2 encoders; no surrogate-gradient "
                "selection training is used in this phase."
            ),
        },
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp12.0 local primitive bottleneck and sparse-event diagnosis"
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--force", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)
    dense = sub.add_parser("dense")
    dense.add_argument("--array-task-id", type=int, required=True)
    posthoc = sub.add_parser("posthoc")
    posthoc.add_argument("--array-task-id", type=int, required=True)
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    config = Config(
        repo_root=root,
        results_dir=results_dir(root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        max_epochs=args.max_epochs,
    )

    if args.command == "dense":
        specs = dense_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        run_dense(specs[args.array_task_id], config, force=args.force)
        return
    if args.command == "posthoc":
        specs = posthoc_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        run_posthoc(specs[args.array_task_id], config, force=args.force)
        return
    if args.command == "finalize":
        finalize(config)
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
