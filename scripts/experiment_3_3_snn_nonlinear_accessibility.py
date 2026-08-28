from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_0_4_l2_width_representation_capacity as probe_utils
from scripts import experiment_3_0_5_frozen_representation_accessibility as source_selection
from scripts import experiment_3_1_raw_vs_snn_representation_value as exp31
from scripts import experiment_3_2_nonlinear_temporal_interaction as exp32


EXPERIMENT_ID = "experiment_3_3_snn_nonlinear_accessibility"
PROTOCOL_VERSION = "matched_raw_snn_residual_v1"
REFERENCE_EXPERIMENT = exp31.EXPERIMENT_ID
REFERENCE_PROTOCOL_VERSION = exp31.PROTOCOL_VERSION

SNN_SEEDS = base.SEEDS
SOURCES = ("raw", "snn_l2")
TEMPORAL_REPRESENTATIONS = ("fixed250", "relative10")
RESIDUAL_DECODERS = exp32.RESIDUAL_DECODERS
PRIMARY_REPRESENTATION = "fixed250"
PRIMARY_DECODER = "local_transition"

MODEL_INIT_SEED = 3301
RESIDUAL_RANK = exp32.RESIDUAL_RANK
BATCH_SIZE = 64
MAX_EPOCHS = exp32.MAX_EPOCHS
MIN_EPOCHS = exp32.MIN_EPOCHS
EARLY_STOP_PATIENCE = exp32.EARLY_STOP_PATIENCE
LEARNING_RATE = exp32.LEARNING_RATE
WEIGHT_DECAY = exp32.WEIGHT_DECAY
GRAD_CLIP_NORM = exp32.GRAD_CLIP_NORM
LINEAR_REPRO_TOL = 1e-10

RAW_CHANNELS = base.EVENT_CHANNELS
SNN_L2_CHANNELS = exp31.SOURCE_WIDTH

# Raw is deterministic on the one fixed 3.1 split, so it is trained once per
# temporal representation / decoder. SNN-L2 is repeated across the three
# independently trained frozen source-SNN seeds.
EXPECTED_RAW_RUNS = len(TEMPORAL_REPRESENTATIONS) * len(RESIDUAL_DECODERS)
EXPECTED_SNN_RUNS = (
    len(SNN_SEEDS) * len(TEMPORAL_REPRESENTATIONS) * len(RESIDUAL_DECODERS)
)
EXPECTED_RUNS = EXPECTED_RAW_RUNS + EXPECTED_SNN_RUNS


@dataclass(frozen=True)
class RunSpec:
    source: str
    temporal_representation: str
    decoder: str
    snn_seed: int | None

    @property
    def key(self) -> str:
        seed_text = "shared" if self.snn_seed is None else f"seed{self.snn_seed}"
        return (
            f"{self.source}__{self.temporal_representation}__"
            f"{self.decoder}__{seed_text}"
        )


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    min_epochs: int = MIN_EPOCHS
    patience: int = EARLY_STOP_PATIENCE
    lr: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    grad_clip_norm: float = GRAD_CLIP_NORM
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
    specs: list[RunSpec] = []
    for representation in TEMPORAL_REPRESENTATIONS:
        for decoder in RESIDUAL_DECODERS:
            specs.append(RunSpec("raw", representation, decoder, None))
            for seed in SNN_SEEDS:
                specs.append(RunSpec("snn_l2", representation, decoder, int(seed)))
    return specs


def _validate_spec(spec: RunSpec) -> None:
    if spec.source not in SOURCES:
        raise ValueError(f"Unknown source: {spec.source}")
    if spec.temporal_representation not in TEMPORAL_REPRESENTATIONS:
        raise ValueError(
            f"Unknown temporal representation: {spec.temporal_representation}"
        )
    if spec.decoder not in RESIDUAL_DECODERS:
        raise ValueError(f"Unknown decoder: {spec.decoder}")
    if spec.source == "raw" and spec.snn_seed is not None:
        raise ValueError("Raw runs must use snn_seed=None")
    if spec.source == "snn_l2" and spec.snn_seed not in SNN_SEEDS:
        raise ValueError(f"SNN-L2 run requires one of {SNN_SEEDS}, got {spec.snn_seed}")


def _artifact_path(root: Path, spec: RunSpec) -> Path:
    return root / "runs" / f"{spec.key}.json"


def _checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def _probe_type(representation: str) -> str:
    return {
        "fixed250": "fixed250_ordered",
        "relative10": "relative10_ordered",
    }[representation]


def _linear_seed(source: str, representation: str) -> int:
    return exp31._probe_seed(source, _probe_type(representation))


def _model_seed(spec: RunSpec) -> int:
    # Deliberately independent of source and frozen SNN seed. This keeps the
    # residual optimization seed fixed so variation across SNN runs comes from
    # the frozen representation rather than a different readout initialization.
    return exp32.derive_seed(
        MODEL_INIT_SEED,
        spec.temporal_representation,
        spec.decoder,
    )


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def fit_linear_baseline(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
) -> dict[str, object]:
    """Reproduce the exact Experiment 3.1 ordered linear-probe protocol."""
    train_flat = np.asarray(train_x).reshape(len(train_x), -1)
    val_flat = np.asarray(val_x).reshape(len(val_x), -1)
    test_flat = np.asarray(test_x).reshape(len(test_x), -1)

    scaler = StandardScaler().fit(train_flat)
    train_z = scaler.transform(train_flat)
    val_z = scaler.transform(val_flat)
    test_z = scaler.transform(test_flat)

    best: tuple[float, float, LogisticRegression] | None = None
    for C in probe_utils.PROBE_C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=5000,
            solver="lbfgs",
            random_state=seed,
        ).fit(train_z, train_y)
        val_ba = float(
            balanced_accuracy_score(val_y, classifier.predict(val_z))
        )
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError("No linear probe candidate was selected")

    val_ba, C, classifier = best
    split_arrays = {
        "train": (train_z, train_y),
        "val": (val_z, val_y),
        "test": (test_z, test_y),
    }
    metrics: dict[str, dict[str, float]] = {}
    logits: dict[str, np.ndarray] = {}
    for split_name, (features, labels) in split_arrays.items():
        pred = classifier.predict(features)
        metrics[split_name] = _classification_metrics(labels, pred)
        if split_name == "val":
            metrics[split_name]["balanced_accuracy"] = val_ba
        logits[split_name] = np.asarray(
            classifier.decision_function(features),
            dtype=np.float32,
        )

    return {
        "model": classifier,
        "scaler": scaler,
        "C": C,
        "logits": logits,
        "metrics": metrics,
    }


def _mask_from_lengths(
    lengths: np.ndarray,
    n_bins: int,
    representation: str,
    bin_steps: int,
) -> np.ndarray:
    if representation == "relative10":
        return np.ones((len(lengths), n_bins), dtype=bool)
    valid_bins = np.ceil(np.asarray(lengths, dtype=np.float64) / bin_steps).astype(int)
    valid_bins = np.clip(valid_bins, 1, n_bins)
    return np.arange(n_bins)[None, :] < valid_bins[:, None]


def _raw_partition(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    representation: str,
    data: base.Data,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    lengths_t = torch.from_numpy(np.asarray(lengths, dtype=np.int64))
    if representation == "fixed250":
        counts_t = base.fixed_counts(x_t, lengths_t, data.bin_steps)
    elif representation == "relative10":
        counts_t = base.relative_counts(x_t, lengths_t, base.N_REL)
    else:
        raise ValueError(representation)
    counts = counts_t.numpy().astype(np.float32, copy=False)
    mask = _mask_from_lengths(lengths, counts.shape[1], representation, data.bin_steps)
    return counts, mask, np.asarray(y, dtype=np.int64)


def _snn_partition(
    model: probe_utils.L2WidthNet,
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    representation: str,
    split_name: str,
    data: base.Data,
    config: Config,
    snn_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = base.loader(
        X,
        y,
        lengths,
        config.batch_size,
        False,
        base.dseed(snn_seed, split_name, "exp33_loader"),
    )
    device = torch.device(config.device)

    count_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    length_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for Xb, yb, lb in loader:
            Xb = Xb.to(device)
            lb_device = lb.to(device)
            l2_spikes = model.layer_features(Xb)["L2"]
            if representation == "fixed250":
                counts_t = base.fixed_counts(
                    l2_spikes,
                    lb_device,
                    data.bin_steps,
                )
            elif representation == "relative10":
                counts_t = base.relative_counts(
                    l2_spikes,
                    lb_device,
                    base.N_REL,
                )
            else:
                raise ValueError(representation)
            count_parts.append(counts_t.cpu().numpy())
            y_parts.append(yb.numpy())
            length_parts.append(lb.numpy())

    counts = np.concatenate(count_parts).astype(np.float32, copy=False)
    y_all = np.concatenate(y_parts).astype(np.int64, copy=False)
    lengths_all = np.concatenate(length_parts).astype(np.int64, copy=False)
    mask = _mask_from_lengths(
        lengths_all,
        counts.shape[1],
        representation,
        data.bin_steps,
    )
    return counts, mask, y_all


def build_partitions(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    str,
]:
    """Build matched ordered [sample, temporal-bin, channel] features."""
    _validate_spec(spec)
    raw_partitions = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }

    source_objective = exp31.selected_source_objective(config.repo_root)
    if spec.source == "raw":
        built = {
            name: _raw_partition(
                X,
                y,
                lengths,
                spec.temporal_representation,
                data,
            )
            for name, (X, y, lengths) in raw_partitions.items()
        }
        return built, source_objective

    if spec.snn_seed is None:
        raise ValueError("SNN-L2 source requires snn_seed")
    device = torch.device(config.device)
    model, checkpoint_payload = probe_utils.load_model_for_evaluation(
        config.repo_root,
        probe_utils.results_dir(config.repo_root),
        exp31.SOURCE_WIDTH,
        source_objective,
        spec.snn_seed,
        data,
        device,
    )
    source_selection._validate_source_checkpoint(
        checkpoint_payload,
        source_objective,
        spec.snn_seed,
    )
    built = {
        name: _snn_partition(
            model,
            X,
            y,
            lengths,
            spec.temporal_representation,
            name,
            data,
            config,
            spec.snn_seed,
        )
        for name, (X, y, lengths) in raw_partitions.items()
    }
    return built, source_objective


class ResidualDecoder(nn.Module):
    def __init__(
        self,
        decoder: str,
        n_bins: int,
        n_channels: int,
        n_classes: int,
        rank: int = RESIDUAL_RANK,
    ) -> None:
        super().__init__()
        if decoder not in RESIDUAL_DECODERS:
            raise ValueError(f"Unknown decoder: {decoder}")
        self.decoder = decoder
        self.n_bins = int(n_bins)
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.rank = int(rank)

        if decoder in ("local", "local_transition"):
            self.local_projection = nn.Linear(n_channels, rank, bias=False)
            self.local_head = nn.Linear(n_bins * rank, n_classes, bias=False)
            nn.init.zeros_(self.local_head.weight)
        else:
            self.local_projection = None
            self.local_head = None

        if decoder in ("transition", "local_transition"):
            self.transition_p = nn.Linear(n_channels, rank, bias=False)
            self.transition_q = nn.Linear(n_channels, rank, bias=False)
            self.transition_head = nn.Linear(
                (n_bins - 1) * rank,
                n_classes,
                bias=False,
            )
            nn.init.zeros_(self.transition_head.weight)
        else:
            self.transition_p = None
            self.transition_q = None
            self.transition_head = None

    def residual_logits(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        if self.local_projection is not None and self.local_head is not None:
            local = F.gelu(self.local_projection(x))
            local = local * mask.unsqueeze(-1).to(local.dtype)
            pieces.append(self.local_head(local.flatten(start_dim=1)))

        if (
            self.transition_p is not None
            and self.transition_q is not None
            and self.transition_head is not None
        ):
            left = self.transition_p(x[:, :-1])
            right = self.transition_q(x[:, 1:])
            pair_mask = (
                mask[:, :-1] & mask[:, 1:]
            ).unsqueeze(-1).to(left.dtype)
            transition = (left * right) * pair_mask
            pieces.append(self.transition_head(transition.flatten(start_dim=1)))

        if not pieces:
            raise RuntimeError("Residual decoder has no active branch")
        return torch.stack(pieces, dim=0).sum(dim=0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        return base_logits + self.residual_logits(x, mask)


def _make_loader(
    x: np.ndarray,
    mask: np.ndarray,
    base_logits: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor(base_logits, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _evaluate_model(
    model: ResidualDecoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    with torch.no_grad():
        for x, mask, base_logits, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            base_logits = base_logits.to(device)
            y = y.to(device)
            logits = model(x, mask, base_logits)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            total_count += int(len(y))
            y_true.append(y.cpu().numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
    truth = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    return {
        "loss": total_loss / max(total_count, 1),
        **_classification_metrics(truth, pred),
    }


def _zero_residual_identity_error(
    model: ResidualDecoder,
    x: np.ndarray,
    mask: np.ndarray,
    base_logits: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    n = min(len(x), 16)
    with torch.no_grad():
        x_t = torch.tensor(x[:n], dtype=torch.float32, device=device)
        m_t = torch.tensor(mask[:n], dtype=torch.bool, device=device)
        b_t = torch.tensor(base_logits[:n], dtype=torch.float32, device=device)
        logits = model(x_t, m_t, b_t)
    return float(torch.max(torch.abs(logits - b_t)).item())


def train_residual(
    spec: RunSpec,
    train_x: np.ndarray,
    train_mask: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_mask: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_mask: np.ndarray,
    test_y: np.ndarray,
    base_logits: dict[str, np.ndarray],
    config: Config,
) -> tuple[ResidualDecoder, dict[str, object]]:
    effective_seed = _model_seed(spec)
    exp32.seed_everything(effective_seed)
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)

    scale = exp32.fit_channel_rms_scale(train_x, train_mask)
    scaled = {
        "train": exp32.apply_channel_scale(train_x, scale),
        "val": exp32.apply_channel_scale(val_x, scale),
        "test": exp32.apply_channel_scale(test_x, scale),
    }
    model = ResidualDecoder(
        spec.decoder,
        n_bins=train_x.shape[1],
        n_channels=train_x.shape[2],
        n_classes=len(np.unique(train_y)),
        rank=RESIDUAL_RANK,
    ).to(device)
    identity_error = _zero_residual_identity_error(
        model,
        scaled["val"],
        val_mask,
        base_logits["val"],
        device,
    )
    if identity_error > 1e-6:
        raise RuntimeError(
            f"Zero-residual identity check failed: {identity_error}"
        )

    train_loader = _make_loader(
        scaled["train"],
        train_mask,
        base_logits["train"],
        train_y,
        config.batch_size,
        True,
        exp32.derive_seed(effective_seed, "train_loader"),
    )
    train_eval_loader = _make_loader(
        scaled["train"],
        train_mask,
        base_logits["train"],
        train_y,
        config.batch_size,
        False,
        exp32.derive_seed(effective_seed, "train_eval_loader"),
    )
    val_loader = _make_loader(
        scaled["val"],
        val_mask,
        base_logits["val"],
        val_y,
        config.batch_size,
        False,
        exp32.derive_seed(effective_seed, "val_loader"),
    )
    test_loader = _make_loader(
        scaled["test"],
        test_mask,
        base_logits["test"],
        test_y,
        config.batch_size,
        False,
        exp32.derive_seed(effective_seed, "test_loader"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    epoch0_val = _evaluate_model(model, val_loader, device)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_ba = float(epoch0_val["balanced_accuracy"])
    best_val_loss = float(epoch0_val["loss"])
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = [
        {
            "epoch": 0,
            "train_loss": float("nan"),
            **{f"val_{key}": value for key, value in epoch0_val.items()},
        }
    ]

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        for x, mask, frozen_logits, y in train_loader:
            x = x.to(device)
            mask = mask.to(device)
            frozen_logits = frozen_logits.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask, frozen_logits)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip_norm,
            )
            optimizer.step()

            running_loss += float(loss.item()) * len(y)
            running_count += int(len(y))

        val_metrics = _evaluate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(running_count, 1),
                **{
                    f"val_{key}": value
                    for key, value in val_metrics.items()
                },
            }
        )

        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        improved = (
            val_ba > best_val_ba + 1e-12
            or (
                abs(val_ba - best_val_ba) <= 1e-12
                and val_loss < best_val_loss - 1e-12
            )
        )
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_ba = val_ba
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch >= config.min_epochs
            and epochs_without_improvement >= config.patience
        ):
            break

    model.load_state_dict(best_state)
    metrics = {
        "train": _evaluate_model(model, train_eval_loader, device),
        "val": _evaluate_model(model, val_loader, device),
        "test": _evaluate_model(model, test_loader, device),
    }
    return model, {
        "effective_model_seed": int(effective_seed),
        "identity_error_epoch0": identity_error,
        "best_epoch": int(best_epoch),
        "n_epochs_ran": int(history[-1]["epoch"]),
        "n_trainable_params": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "channel_rms_scale": scale.tolist(),
        "history": history,
        "metrics": metrics,
    }


def run_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> dict[str, object]:
    _validate_spec(spec)
    out_path = _artifact_path(config.results_dir, spec)
    if config.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        expected = (
            PROTOCOL_VERSION,
            spec.source,
            spec.temporal_representation,
            spec.decoder,
            spec.snn_seed,
        )
        actual = (
            str(cached.get("protocol_version")),
            str(cached.get("source")),
            str(cached.get("temporal_representation")),
            str(cached.get("decoder")),
            cached.get("snn_seed"),
        )
        if actual != expected:
            raise ValueError(
                f"Run identity mismatch in {out_path}: {actual} != {expected}"
            )
        return cached

    partitions, source_objective = build_partitions(spec, data, config)
    train_x, train_mask, train_y = partitions["train"]
    val_x, val_mask, val_y = partitions["val"]
    test_x, test_mask, test_y = partitions["test"]

    n_channels = train_x.shape[2]
    expected_channels = RAW_CHANNELS if spec.source == "raw" else SNN_L2_CHANNELS
    if n_channels != expected_channels:
        raise ValueError(
            f"{spec.source} expected {expected_channels} channels, got {n_channels}"
        )

    linear_seed = _linear_seed(spec.source, spec.temporal_representation)
    baseline = fit_linear_baseline(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        linear_seed,
    )
    model, training = train_residual(
        spec,
        train_x,
        train_mask,
        train_y,
        val_x,
        val_mask,
        val_y,
        test_x,
        test_mask,
        test_y,
        baseline["logits"],
        config,
    )

    linear_test = baseline["metrics"]["test"]
    residual_test = training["metrics"]["test"]
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "reference_protocol_version": REFERENCE_PROTOCOL_VERSION,
        "source": spec.source,
        "temporal_representation": spec.temporal_representation,
        "decoder": spec.decoder,
        "snn_seed": spec.snn_seed,
        "source_objective": source_objective,
        "split_seed": int(base.SPLIT_SEED),
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
        "sampling_rate_hz": float(data.fs),
        "fixed_bin_ms": float(base.FIXED_MS),
        "fixed_bin_steps": int(data.bin_steps),
        "relative_bins": int(base.N_REL),
        "input_channels": int(n_channels),
        "n_time_bins": int(train_x.shape[1]),
        "linear_feature_dim": int(np.prod(train_x.shape[1:])),
        "linear_seed": int(linear_seed),
        "linear_C": float(baseline["C"]),
        "linear_metrics": baseline["metrics"],
        "model_init_seed": int(MODEL_INIT_SEED),
        "effective_model_seed": int(training["effective_model_seed"]),
        "residual_rank": int(RESIDUAL_RANK),
        "best_epoch": int(training["best_epoch"]),
        "n_epochs_ran": int(training["n_epochs_ran"]),
        "n_trainable_params": int(training["n_trainable_params"]),
        "identity_error_epoch0": float(training["identity_error_epoch0"]),
        "channel_rms_scale": training["channel_rms_scale"],
        "residual_metrics": training["metrics"],
        "test_delta_vs_linear": float(
            residual_test["balanced_accuracy"]
            - linear_test["balanced_accuracy"]
        ),
        "val_delta_vs_linear": float(
            training["metrics"]["val"]["balanced_accuracy"]
            - baseline["metrics"]["val"]["balanced_accuracy"]
        ),
        "mask_policy": (
            "fixed250 masks invalid padded bins; relative10 uses all ten bins; "
            "mask is never a classifier feature"
        ),
        "residual_policy": (
            "frozen Experiment 3.1 Linear logits + zero-initialized nonlinear residual"
        ),
        "training_history": training["history"],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    checkpoint = _checkpoint_path(config.results_dir, spec)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": {
                "source": spec.source,
                "temporal_representation": spec.temporal_representation,
                "decoder": spec.decoder,
                "snn_seed": spec.snn_seed,
            },
            "source_objective": source_objective,
            "effective_model_seed": int(training["effective_model_seed"]),
            "best_epoch": int(training["best_epoch"]),
            "channel_rms_scale": training["channel_rms_scale"],
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    return payload


def _load_reference_probe_results(repo_root: Path) -> pd.DataFrame:
    path = (
        exp31.results_dir(repo_root)
        / "experiment_3_1_probe_results.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing finalized Experiment 3.1 probe results: {path}"
        )
    return pd.read_csv(path)


def _reference_linear_row(
    reference: pd.DataFrame,
    source: str,
    representation: str,
    snn_seed: int | None,
) -> pd.Series:
    probe_type = _probe_type(representation)
    rows = reference[
        (reference.representation == source)
        & (reference.probe_type == probe_type)
    ]
    if source == "snn_l2":
        rows = rows[rows.seed == snn_seed]
    else:
        rows = rows.sort_values("seed").head(1)
    if len(rows) != 1:
        raise ValueError(
            f"Expected one 3.1 reference row for {source}/{probe_type}/{snn_seed}, "
            f"got {len(rows)}"
        )
    return rows.iloc[0]


def _assert_linear_reproduction(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dedup = (
        frame.sort_values(
            ["source", "temporal_representation", "snn_seed", "decoder"],
            na_position="first",
        )
        .drop_duplicates(
            ["source", "temporal_representation", "snn_seed"],
            keep="first",
        )
        .reset_index(drop=True)
    )
    for row in dedup.itertuples(index=False):
        ref = _reference_linear_row(
            reference,
            str(row.source),
            str(row.temporal_representation),
            None if pd.isna(row.snn_seed) else int(row.snn_seed),
        )
        checks = {
            "linear_C": (float(row.linear_C), float(ref.probe_C)),
            "linear_val_ba": (
                float(row.linear_val_ba),
                float(ref.probe_val_balanced_accuracy),
            ),
            "linear_test_ba": (
                float(row.linear_test_ba),
                float(ref.probe_test_balanced_accuracy),
            ),
            "linear_test_macro_f1": (
                float(row.linear_test_macro_f1),
                float(ref.probe_test_macro_f1),
            ),
            "linear_test_accuracy": (
                float(row.linear_test_accuracy),
                float(ref.probe_test_accuracy),
            ),
        }
        for metric, (actual, expected) in checks.items():
            difference = actual - expected
            rows.append(
                {
                    "source": row.source,
                    "temporal_representation": row.temporal_representation,
                    "snn_seed": row.snn_seed,
                    "metric": metric,
                    "exp33_value": actual,
                    "exp31_value": expected,
                    "difference": difference,
                    "abs_difference": abs(difference),
                }
            )
            if abs(difference) > LINEAR_REPRO_TOL:
                raise ValueError(
                    "Experiment 3.3 failed to reproduce Experiment 3.1 "
                    f"{row.source}/{row.temporal_representation}/{row.snn_seed} "
                    f"{metric}: {actual} vs {expected}"
                )
    return pd.DataFrame(rows)


def _rows_from_payload(payload: dict[str, object]) -> dict[str, object]:
    linear = payload["linear_metrics"]
    residual = payload["residual_metrics"]
    if not isinstance(linear, dict) or not isinstance(residual, dict):
        raise TypeError("Malformed metrics in run payload")
    return {
        "source": payload["source"],
        "temporal_representation": payload["temporal_representation"],
        "decoder": payload["decoder"],
        "snn_seed": payload["snn_seed"],
        "source_objective": payload["source_objective"],
        "split_seed": payload["split_seed"],
        "input_channels": payload["input_channels"],
        "n_time_bins": payload["n_time_bins"],
        "linear_feature_dim": payload["linear_feature_dim"],
        "linear_C": payload["linear_C"],
        "best_epoch": payload["best_epoch"],
        "n_trainable_params": payload["n_trainable_params"],
        "identity_error_epoch0": payload["identity_error_epoch0"],
        "linear_train_ba": linear["train"]["balanced_accuracy"],
        "linear_val_ba": linear["val"]["balanced_accuracy"],
        "linear_test_ba": linear["test"]["balanced_accuracy"],
        "linear_test_accuracy": linear["test"]["accuracy"],
        "linear_test_macro_f1": linear["test"]["macro_f1"],
        "residual_train_ba": residual["train"]["balanced_accuracy"],
        "residual_val_ba": residual["val"]["balanced_accuracy"],
        "residual_test_ba": residual["test"]["balanced_accuracy"],
        "residual_test_accuracy": residual["test"]["accuracy"],
        "residual_test_macro_f1": residual["test"]["macro_f1"],
        "test_delta_vs_linear": payload["test_delta_vs_linear"],
        "val_delta_vs_linear": payload["val_delta_vs_linear"],
    }


def _assert_repeated_linear_identity(frame: pd.DataFrame) -> None:
    columns = (
        "linear_C",
        "linear_train_ba",
        "linear_val_ba",
        "linear_test_ba",
        "linear_test_accuracy",
        "linear_test_macro_f1",
    )
    for keys, group in frame.groupby(
        ["source", "temporal_representation", "snn_seed"],
        dropna=False,
        sort=True,
    ):
        first = group.iloc[0]
        for column in columns:
            values = group[column].astype(float).to_numpy()
            if not np.allclose(
                values,
                float(first[column]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Repeated Linear baseline mismatch for {keys}/{column}: "
                    f"{values.tolist()}"
                )


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["source", "temporal_representation", "decoder"],
        sort=True,
    ):
        source, representation, decoder = keys
        deltas = group.test_delta_vs_linear.astype(float)
        residual = group.residual_test_ba.astype(float)
        linear = group.linear_test_ba.astype(float)
        rows.append(
            {
                "source": source,
                "temporal_representation": representation,
                "decoder": decoder,
                "n_runs": int(len(group)),
                "mean_linear_test_ba": float(linear.mean()),
                "sd_linear_test_ba": (
                    float(linear.std(ddof=1)) if len(group) > 1 else 0.0
                ),
                "mean_residual_test_ba": float(residual.mean()),
                "sd_residual_test_ba": (
                    float(residual.std(ddof=1)) if len(group) > 1 else 0.0
                ),
                "mean_delta_vs_linear": float(deltas.mean()),
                "sd_delta_vs_linear": (
                    float(deltas.std(ddof=1)) if len(group) > 1 else 0.0
                ),
                "median_delta_vs_linear": float(deltas.median()),
                "improved_runs": int((deltas > 0.0).sum()),
                "n_trainable_params": int(group.n_trainable_params.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _paired_accessibility(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for representation in TEMPORAL_REPRESENTATIONS:
        for decoder in RESIDUAL_DECODERS:
            raw = frame[
                (frame.source == "raw")
                & (frame.temporal_representation == representation)
                & (frame.decoder == decoder)
            ]
            if len(raw) != 1:
                raise ValueError(
                    f"Expected one raw run for {representation}/{decoder}"
                )
            raw_row = raw.iloc[0]
            snn = frame[
                (frame.source == "snn_l2")
                & (frame.temporal_representation == representation)
                & (frame.decoder == decoder)
            ].sort_values("snn_seed")
            if len(snn) != len(SNN_SEEDS):
                raise ValueError(
                    f"Expected {len(SNN_SEEDS)} SNN runs for "
                    f"{representation}/{decoder}, got {len(snn)}"
                )
            for snn_row in snn.itertuples(index=False):
                rows.append(
                    {
                        "temporal_representation": representation,
                        "decoder": decoder,
                        "snn_seed": int(snn_row.snn_seed),
                        "raw_linear_test_ba": float(raw_row.linear_test_ba),
                        "snn_linear_test_ba": float(snn_row.linear_test_ba),
                        "snn_minus_raw_linear_test_ba": float(
                            snn_row.linear_test_ba - raw_row.linear_test_ba
                        ),
                        "raw_nonlinear_gain": float(
                            raw_row.test_delta_vs_linear
                        ),
                        "snn_nonlinear_gain": float(
                            snn_row.test_delta_vs_linear
                        ),
                        "snn_gain_minus_raw_gain": float(
                            snn_row.test_delta_vs_linear
                            - raw_row.test_delta_vs_linear
                        ),
                        "raw_residual_test_ba": float(
                            raw_row.residual_test_ba
                        ),
                        "snn_residual_test_ba": float(
                            snn_row.residual_test_ba
                        ),
                        "snn_minus_raw_residual_test_ba": float(
                            snn_row.residual_test_ba
                            - raw_row.residual_test_ba
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in paired.groupby(
        ["temporal_representation", "decoder"],
        sort=True,
    ):
        representation, decoder = keys
        linear_delta = group.snn_minus_raw_linear_test_ba.astype(float)
        gain_delta = group.snn_gain_minus_raw_gain.astype(float)
        snn_gain = group.snn_nonlinear_gain.astype(float)
        rows.append(
            {
                "temporal_representation": representation,
                "decoder": decoder,
                "n_snn_seeds": int(len(group)),
                "mean_snn_minus_raw_linear_test_ba": float(
                    linear_delta.mean()
                ),
                "sd_snn_minus_raw_linear_test_ba": float(
                    linear_delta.std(ddof=1)
                ),
                "raw_nonlinear_gain": float(
                    group.raw_nonlinear_gain.iloc[0]
                ),
                "mean_snn_nonlinear_gain": float(snn_gain.mean()),
                "sd_snn_nonlinear_gain": float(snn_gain.std(ddof=1)),
                "mean_snn_gain_minus_raw_gain": float(gain_delta.mean()),
                "sd_snn_gain_minus_raw_gain": float(
                    gain_delta.std(ddof=1)
                ),
                "snn_lower_gain_seeds": int((gain_delta < 0.0).sum()),
                "snn_linear_better_seeds": int((linear_delta > 0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _primary_conclusion(paired_summary: pd.DataFrame) -> dict[str, object]:
    row = paired_summary[
        (paired_summary.temporal_representation == PRIMARY_REPRESENTATION)
        & (paired_summary.decoder == PRIMARY_DECODER)
    ]
    if len(row) != 1:
        raise RuntimeError("Missing primary Fixed250 Local+Transition summary")
    record = row.iloc[0]
    linear_better = (
        float(record.mean_snn_minus_raw_linear_test_ba) > 0.0
    )
    raw_has_nonlinear_gain = float(record.raw_nonlinear_gain) > 0.0
    snn_gain_smaller = float(record.mean_snn_gain_minus_raw_gain) < 0.0
    consistency = int(record.snn_lower_gain_seeds) >= 2
    supported = (
        linear_better
        and raw_has_nonlinear_gain
        and snn_gain_smaller
        and consistency
    )
    return {
        "primary_representation": PRIMARY_REPRESENTATION,
        "primary_decoder": PRIMARY_DECODER,
        "hypothesis": (
            "Frozen SNN-L2 has already converted part of the nonlinear "
            "absolute-time temporal structure into linearly accessible features."
        ),
        "validation_standard": {
            "criterion_1": (
                "mean(SNN-L2 Linear BA - Raw Linear BA) > 0 on Fixed250"
            ),
            "criterion_2": (
                "Raw Fixed250 Local+Transition gain over its frozen Linear baseline > 0"
            ),
            "criterion_3": (
                "mean(SNN-L2 nonlinear gain - Raw nonlinear gain) < 0"
            ),
            "criterion_4": (
                "at least 2 of 3 frozen SNN seeds have smaller nonlinear gain than Raw"
            ),
            "decision_rule": "all four criteria must pass",
        },
        "criterion_1_linear_better": bool(linear_better),
        "criterion_2_raw_has_nonlinear_gain": bool(raw_has_nonlinear_gain),
        "criterion_3_snn_gain_smaller": bool(snn_gain_smaller),
        "criterion_4_consistent_across_seeds": bool(consistency),
        "primary_hypothesis_supported": bool(supported),
        "mean_snn_minus_raw_linear_test_ba": float(
            record.mean_snn_minus_raw_linear_test_ba
        ),
        "raw_nonlinear_gain": float(record.raw_nonlinear_gain),
        "mean_snn_nonlinear_gain": float(record.mean_snn_nonlinear_gain),
        "mean_snn_gain_minus_raw_gain": float(
            record.mean_snn_gain_minus_raw_gain
        ),
        "snn_lower_gain_seeds": int(record.snn_lower_gain_seeds),
        "n_snn_seeds": int(record.n_snn_seeds),
    }


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    objectives: set[str] = set()

    for spec in run_specs():
        path = _artifact_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Experiment 3.3 run artifact: {path}"
            )
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = (
            EXPERIMENT_ID,
            PROTOCOL_VERSION,
            spec.source,
            spec.temporal_representation,
            spec.decoder,
            spec.snn_seed,
        )
        actual = (
            str(payload.get("experiment_id")),
            str(payload.get("protocol_version")),
            str(payload.get("source")),
            str(payload.get("temporal_representation")),
            str(payload.get("decoder")),
            payload.get("snn_seed"),
        )
        if actual != expected:
            raise ValueError(
                f"Run identity mismatch in {path}: {actual} != {expected}"
            )
        objectives.add(str(payload.get("source_objective")))
        rows.append(_rows_from_payload(payload))

    if len(rows) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} rows, got {len(rows)}")
    if len(objectives) != 1:
        raise ValueError(f"Mixed source objectives: {sorted(objectives)}")

    frame = pd.DataFrame(rows)
    _assert_repeated_linear_identity(frame)
    reference = _load_reference_probe_results(repo_root)
    reproduction = _assert_linear_reproduction(frame, reference)
    summary = _summary(frame)
    paired = _paired_accessibility(frame)
    paired_summary = _paired_summary(paired)
    conclusion = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "source_objective": next(iter(objectives)),
        "split_seed": int(base.SPLIT_SEED),
        "linear_reproduction_max_abs_difference": float(
            reproduction.abs_difference.max()
        ),
        "primary_question": (
            "Has the frozen SNN-L2 already converted part of the nonlinear "
            "temporal structure in raw events into linearly accessible features?"
        ),
        **_primary_conclusion(paired_summary),
        "secondary_control": (
            "Relative10 is descriptive only and does not determine the primary "
            "Fixed250 hypothesis decision."
        ),
        "guardrail": (
            "SNN-L2 has 128 channels versus Raw's 30. A smaller SNN residual gain "
            "is conservative evidence for linearization; a larger SNN gain is "
            "capacity-confounded and should not by itself be interpreted as more "
            "intrinsic nonlinear information."
        ),
    }

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results": root / "experiment_3_3_results.csv",
        "summary": root / "experiment_3_3_summary.csv",
        "paired_accessibility": (
            root / "experiment_3_3_paired_accessibility.csv"
        ),
        "paired_summary": root / "experiment_3_3_paired_summary.csv",
        "linear_reproduction": (
            root / "experiment_3_3_linear_reproduction.csv"
        ),
        "conclusion": root / "experiment_3_3_conclusion.json",
        "provenance": root / "provenance.json",
    }
    frame.to_csv(outputs["results"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    paired.to_csv(outputs["paired_accessibility"], index=False)
    paired_summary.to_csv(outputs["paired_summary"], index=False)
    reproduction.to_csv(outputs["linear_reproduction"], index=False)
    with outputs["conclusion"].open("w", encoding="utf-8") as handle:
        json.dump(conclusion, handle, indent=2, sort_keys=True)
    with outputs["provenance"].open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "reference_experiment": REFERENCE_EXPERIMENT,
                "reference_protocol_version": REFERENCE_PROTOCOL_VERSION,
                "source_selection_experiment": source_selection.EXPERIMENT_ID,
                "source_architecture": exp31.SOURCE_ARCHITECTURE,
                "source_width": exp31.SOURCE_WIDTH,
                "source_shifts": exp31.SOURCE_SHIFTS,
                "snn_seeds": SNN_SEEDS,
                "split_seed": int(base.SPLIT_SEED),
                "sources": SOURCES,
                "temporal_representations": TEMPORAL_REPRESENTATIONS,
                "residual_decoders": RESIDUAL_DECODERS,
                "residual_rank": RESIDUAL_RANK,
                "model_init_seed": MODEL_INIT_SEED,
                "expected_runs": EXPECTED_RUNS,
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
            raise ValueError(
                f"array task id {task_id} outside 0..{len(specs) - 1}"
            )
        data = base.prepare_data(repo_root)
        config = Config(
            repo_root=repo_root,
            results_dir=root,
            device=args.device,
            batch_size=args.batch_size,
            resume=not args.force,
            threads=1,
        )
        spec = specs[task_id]
        result = run_one(spec, data, config)
        print(
            f"completed {spec.key} "
            f"linear_test_ba={result['linear_metrics']['test']['balanced_accuracy']:.4f} "
            f"residual_test_ba={result['residual_metrics']['test']['balanced_accuracy']:.4f}"
        )
        return

    if args.command == "finalize":
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return

    raise ValueError(args.command)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 3.3 matched Raw vs frozen SNN-L2 nonlinear accessibility"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run.add_argument("--force", action="store_true")

    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    _run_cli(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
