from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from snn.accel_reconstruction_eval.datasets import load_acceleration_data


EXPERIMENT_ID = "experiment_3_2_nonlinear_temporal_interaction"
PROTOCOL_VERSION = "low_rank_residual_v1"
REFERENCE_EXPERIMENT = "experiment_1_3_3_temporal_decoder_comparison"

SPLIT_SEEDS = (11, 23, 37, 53, 71)
N_TRAIN_USERS, N_VAL_USERS, N_TEST_USERS = 12, 4, 4
REPRESENTATIONS = ("relative10", "fixed250", "fixed500")
RESIDUAL_DECODERS = ("local", "transition", "local_transition")
EXPECTED_RUNS = len(SPLIT_SEEDS) * len(REPRESENTATIONS) * len(RESIDUAL_DECODERS)

MODEL_INIT_SEED = 2026
RESIDUAL_RANK = 16
BATCH_SIZE = 64
MAX_EPOCHS = 200
MIN_EPOCHS = 20
EARLY_STOP_PATIENCE = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
LOGREG_MAX_ITER = 5000
BASELINE_REPRO_TOL = 0.01

EVENT_CHANNEL_COUNT = 30
TOTAL_CHANNEL_COUNT = 36
EXPECTED_SAMPLING_RATE_HZ = 64.0
EXPECTED_EVENT_REPRESENTATION = "unsigned"
EXPECTED_EVENT_FEATURE_SCHEMA = "custom_wavelet_polarity_split_abs_events_v1"
INCLUDED_LABELS = ("A", "B", "C", "D", "E", "X", "G", "H", "I", "J", "K", "L")
GLOBAL_PADDED_LENGTH = 256
RELATIVE_N_BINS = 10
EPS = 1e-8


@dataclass(frozen=True)
class RunSpec:
    representation: str
    decoder: str
    split_seed: int

    @property
    def key(self) -> str:
        return f"{self.representation}__{self.decoder}__split{self.split_seed}"


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


@dataclass
class Cohort:
    manifest: pd.DataFrame
    packages: object
    fs: float
    labels: tuple[str, ...]


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "snn").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate writingRing repository root")


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def derive_seed(seed: int, *parts: object) -> int:
    text = "|".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(representation, decoder, split_seed)
        for representation in REPRESENTATIONS
        for decoder in RESIDUAL_DECODERS
        for split_seed in SPLIT_SEEDS
    ]


def _dataset_roots(repo_root: Path) -> list[Path]:
    return [
        repo_root / "outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded",
        repo_root / "outputs/action1_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded",
    ]


def load_cohort(repo_root: Path) -> Cohort:
    data = load_acceleration_data(
        _dataset_roots(repo_root),
        repository_root=repo_root,
        require_reconstruction=False,
    )
    fs_values = {float(metadata.sampling_rate_hz) for metadata in data.producer_metadatas}
    if len(fs_values) != 1:
        raise ValueError(f"Expected one shared sampling rate, got {fs_values}")
    fs = fs_values.pop()
    if not np.isclose(fs, EXPECTED_SAMPLING_RATE_HZ):
        raise ValueError(f"Expected {EXPECTED_SAMPLING_RATE_HZ} Hz, got {fs}")

    for root, metadata in zip(data.padded_roots, data.producer_metadatas, strict=True):
        raw = metadata.raw
        if raw.get("event_representation") != EXPECTED_EVENT_REPRESENTATION:
            raise ValueError(f"{root}: unexpected event representation")
        if raw.get("event_feature_schema") != EXPECTED_EVENT_FEATURE_SCHEMA:
            raise ValueError(f"{root}: unexpected event feature schema")
        if raw.get("event_channel_count") != EVENT_CHANNEL_COUNT:
            raise ValueError(f"{root}: expected {EVENT_CHANNEL_COUNT} event channels")
        if metadata.channel_count != TOTAL_CHANNEL_COUNT:
            raise ValueError(f"{root}: expected {TOTAL_CHANNEL_COUNT} total channels")

    rows: list[dict[str, object]] = []
    keep = set(INCLUDED_LABELS)
    for package_index, package in enumerate(data.packages):
        for segment_index, label in enumerate(package.labels.astype(str)):
            if label not in keep:
                continue
            rows.append(
                {
                    "package_index": package_index,
                    "segment_index": segment_index,
                    "user": str(package.user),
                    "action": str(package.action),
                    "label": str(label),
                    "valid_length": int(package.valid_lengths[segment_index]),
                    "package_padded_length": int(package.padded_spike_imu.shape[1]),
                    "sample_id": f"{package.user}/action_{package.action}/{segment_index}",
                }
            )
    manifest = pd.DataFrame(rows)
    labels = tuple(sorted(manifest.label.unique().tolist()))
    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    manifest["label_idx"] = manifest.label.map(class_to_idx).astype(int)
    if len(manifest) != 853 or manifest.user.nunique() != 20 or len(labels) != 12:
        raise ValueError(
            f"Unexpected cohort: samples={len(manifest)} users={manifest.user.nunique()} classes={len(labels)}"
        )
    if int(manifest.package_padded_length.max()) != GLOBAL_PADDED_LENGTH:
        raise ValueError(
            f"Expected global padded length {GLOBAL_PADDED_LENGTH}, got {manifest.package_padded_length.max()}"
        )
    return Cohort(manifest=manifest, packages=data.packages, fs=fs, labels=labels)


def make_user_split(manifest: pd.DataFrame, split_seed: int) -> dict[str, pd.DataFrame]:
    if split_seed not in SPLIT_SEEDS:
        raise ValueError(f"Unknown split seed: {split_seed}")
    users = sorted(manifest.user.unique().tolist())
    if len(users) != N_TRAIN_USERS + N_VAL_USERS + N_TEST_USERS:
        raise ValueError(f"Expected 20 users, got {len(users)}")
    rng = np.random.default_rng(derive_seed(split_seed, "user_split"))
    perm = np.asarray(users, dtype=object)
    rng.shuffle(perm)
    train_users = set(perm[:N_TRAIN_USERS].tolist())
    val_users = set(perm[N_TRAIN_USERS : N_TRAIN_USERS + N_VAL_USERS].tolist())
    test_users = set(perm[-N_TEST_USERS:].tolist())
    if train_users & val_users or train_users & test_users or val_users & test_users:
        raise RuntimeError("User-disjoint split construction failed")
    return {
        "train": manifest[manifest.user.isin(train_users)].reset_index(drop=True),
        "val": manifest[manifest.user.isin(val_users)].reset_index(drop=True),
        "test": manifest[manifest.user.isin(test_users)].reset_index(drop=True),
    }


def _split_users(parts: dict[str, pd.DataFrame]) -> dict[str, tuple[str, ...]]:
    return {
        f"{name}_users": tuple(sorted(frame.user.unique().tolist()))
        for name, frame in parts.items()
    }


def _sample_hash(parts: dict[str, pd.DataFrame]) -> str:
    payload = []
    for split_name in ("train", "val", "test"):
        payload.extend(f"{split_name}:{sample_id}" for sample_id in parts[split_name].sample_id.tolist())
    return hashlib.sha256("\n".join(payload).encode()).hexdigest()


def _build_global_padded_events(cohort: Cohort, frame: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(frame), GLOBAL_PADDED_LENGTH, EVENT_CHANNEL_COUNT), dtype=np.float32)
    for index, row in enumerate(frame.itertuples(index=False)):
        package = cohort.packages[int(row.package_index)]
        x = np.asarray(
            package.padded_spike_imu[int(row.segment_index), :, :EVENT_CHANNEL_COUNT],
            dtype=np.float32,
        )
        valid = min(int(row.valid_length), len(x), GLOBAL_PADDED_LENGTH)
        out[index, :valid] = x[:valid]
    return out


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return frame.label_idx.to_numpy(dtype=np.int64, copy=True)


def _lengths(frame: pd.DataFrame) -> np.ndarray:
    return np.minimum(
        frame.valid_length.to_numpy(dtype=np.int64, copy=True),
        GLOBAL_PADDED_LENGTH,
    )


def _fixed_representation(
    cohort: Cohort,
    frame: pd.DataFrame,
    requested_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    samples_per_bin = int(np.rint(requested_ms * cohort.fs / 1000.0))
    if samples_per_bin <= 0:
        raise ValueError(f"Invalid fixed duration {requested_ms} ms")
    events = _build_global_padded_events(cohort, frame)
    lengths = _lengths(frame)
    n, T, channels = events.shape
    n_bins = int(np.ceil(T / samples_per_bin))
    target = n_bins * samples_per_bin
    if target > T:
        events = np.pad(events, ((0, 0), (0, target - T), (0, 0)))
    counts = events.reshape(n, n_bins, samples_per_bin, channels).sum(axis=2).astype(np.float32)
    valid_bins = np.ceil(lengths / samples_per_bin).astype(int)
    mask = np.arange(n_bins)[None, :] < valid_bins[:, None]
    meta = {
        "requested_duration_ms": float(requested_ms),
        "samples_per_bin": int(samples_per_bin),
        "actual_duration_ms": float(1000.0 * samples_per_bin / cohort.fs),
        "n_time_bins": int(n_bins),
    }
    return counts, mask, _labels(frame), meta


def _relative_representation(
    cohort: Cohort,
    frame: pd.DataFrame,
    n_bins: int = RELATIVE_N_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    features: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        package = cohort.packages[int(row.package_index)]
        valid_length = min(int(row.valid_length), GLOBAL_PADDED_LENGTH)
        x = np.asarray(
            package.padded_spike_imu[int(row.segment_index), :valid_length, :EVENT_CHANNEL_COUNT],
            dtype=np.float32,
        )
        if n_bins > valid_length:
            raise ValueError("n_bins exceeds valid length")
        chunks = np.array_split(x, n_bins, axis=0)
        features.append(np.stack([chunk.sum(axis=0) for chunk in chunks]))
    counts = np.stack(features).astype(np.float32)
    mask = np.ones((len(counts), n_bins), dtype=bool)
    meta = {
        "requested_duration_ms": None,
        "samples_per_bin": None,
        "actual_duration_ms": None,
        "n_time_bins": int(n_bins),
    }
    return counts, mask, _labels(frame), meta


def build_representation(
    cohort: Cohort,
    frame: pd.DataFrame,
    representation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if representation == "relative10":
        return _relative_representation(cohort, frame, RELATIVE_N_BINS)
    if representation == "fixed250":
        return _fixed_representation(cohort, frame, 250.0)
    if representation == "fixed500":
        return _fixed_representation(cohort, frame, 500.0)
    raise ValueError(f"Unknown representation: {representation}")


def _standardize_from_train(
    train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    train = np.asarray(train, dtype=np.float64)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (
        (train - mean) / std,
        *[(np.asarray(array, dtype=np.float64) - mean) / std for array in others],
    )


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
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
    train_flat = np.asarray(train_x).reshape(len(train_x), -1)
    val_flat = np.asarray(val_x).reshape(len(val_x), -1)
    test_flat = np.asarray(test_x).reshape(len(test_x), -1)
    train_z, val_z, test_z = _standardize_from_train(train_flat, val_flat, test_flat)
    model = LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=seed, solver="lbfgs")
    model.fit(train_z, train_y)

    logits = {
        "train": np.asarray(model.decision_function(train_z), dtype=np.float32),
        "val": np.asarray(model.decision_function(val_z), dtype=np.float32),
        "test": np.asarray(model.decision_function(test_z), dtype=np.float32),
    }
    metrics: dict[str, dict[str, float]] = {}
    for split_name, x, y in (
        ("train", train_z, train_y),
        ("val", val_z, val_y),
        ("test", test_z, test_y),
    ):
        metrics[split_name] = _classification_metrics(y, model.predict(x))
    return {"model": model, "logits": logits, "metrics": metrics}


def fit_channel_rms_scale(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    mask64 = np.asarray(mask, dtype=np.float64)[..., None]
    denominator = max(float(mask64.sum()), 1.0)
    mean_square = (x64 * x64 * mask64).sum(axis=(0, 1)) / denominator
    scale = np.sqrt(np.maximum(mean_square, EPS))
    return scale.astype(np.float32)


def apply_channel_scale(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) / scale[None, None, :]).astype(np.float32)


class ResidualDecoder(nn.Module):
    def __init__(self, decoder: str, n_bins: int, n_classes: int, rank: int = RESIDUAL_RANK) -> None:
        super().__init__()
        if decoder not in RESIDUAL_DECODERS:
            raise ValueError(f"Unknown decoder: {decoder}")
        self.decoder = decoder
        self.n_bins = int(n_bins)
        self.rank = int(rank)
        self.n_classes = int(n_classes)

        if decoder in ("local", "local_transition"):
            self.local_projection = nn.Linear(EVENT_CHANNEL_COUNT, rank, bias=False)
            self.local_head = nn.Linear(n_bins * rank, n_classes, bias=False)
            nn.init.zeros_(self.local_head.weight)
        else:
            self.local_projection = None
            self.local_head = None

        if decoder in ("transition", "local_transition"):
            self.transition_p = nn.Linear(EVENT_CHANNEL_COUNT, rank, bias=False)
            self.transition_q = nn.Linear(EVENT_CHANNEL_COUNT, rank, bias=False)
            self.transition_head = nn.Linear((n_bins - 1) * rank, n_classes, bias=False)
            nn.init.zeros_(self.transition_head.weight)
        else:
            self.transition_p = None
            self.transition_q = None
            self.transition_head = None

    def residual_logits(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
            pair_mask = (mask[:, :-1] & mask[:, 1:]).unsqueeze(-1).to(left.dtype)
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


def _model_seed(spec: RunSpec) -> int:
    return derive_seed(MODEL_INIT_SEED, spec.representation, spec.decoder)


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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


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
    with torch.no_grad():
        x_t = torch.tensor(x[: min(len(x), 16)], dtype=torch.float32, device=device)
        m_t = torch.tensor(mask[: min(len(mask), 16)], dtype=torch.bool, device=device)
        b_t = torch.tensor(base_logits[: min(len(base_logits), 16)], dtype=torch.float32, device=device)
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
    seed_everything(effective_seed)
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)

    scale = fit_channel_rms_scale(train_x, train_mask)
    scaled = {
        "train": apply_channel_scale(train_x, scale),
        "val": apply_channel_scale(val_x, scale),
        "test": apply_channel_scale(test_x, scale),
    }
    model = ResidualDecoder(
        spec.decoder,
        n_bins=train_x.shape[1],
        n_classes=len(np.unique(train_y)),
        rank=RESIDUAL_RANK,
    ).to(device)
    identity_error = _zero_residual_identity_error(
        model, scaled["val"], val_mask, base_logits["val"], device
    )
    if identity_error > 1e-6:
        raise RuntimeError(f"Zero-residual identity check failed: {identity_error}")

    train_loader = _make_loader(
        scaled["train"], train_mask, base_logits["train"], train_y,
        config.batch_size, True, derive_seed(effective_seed, "train_loader"),
    )
    val_loader = _make_loader(
        scaled["val"], val_mask, base_logits["val"], val_y,
        config.batch_size, False, derive_seed(effective_seed, "val_loader"),
    )
    test_loader = _make_loader(
        scaled["test"], test_mask, base_logits["test"], test_y,
        config.batch_size, False, derive_seed(effective_seed, "test_loader"),
    )
    train_eval_loader = _make_loader(
        scaled["train"], train_mask, base_logits["train"], train_y,
        config.batch_size, False, derive_seed(effective_seed, "train_eval_loader"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    epoch0_val = _evaluate_model(model, val_loader, device)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_ba = float(epoch0_val["balanced_accuracy"])
    best_val_loss = float(epoch0_val["loss"])
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = [
        {"epoch": 0, "train_loss": float("nan"), **{f"val_{k}": v for k, v in epoch0_val.items()}}
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            running_loss += float(loss.item()) * len(y)
            running_count += int(len(y))

        val_metrics = _evaluate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(running_count, 1),
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        improved = (val_ba > best_val_ba + 1e-12) or (
            abs(val_ba - best_val_ba) <= 1e-12 and val_loss < best_val_loss - 1e-12
        )
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_ba = val_ba
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= config.min_epochs and epochs_without_improvement >= config.patience:
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
        "n_trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "channel_rms_scale": scale.tolist(),
        "history": history,
        "metrics": metrics,
    }


def _artifact_path(root: Path, spec: RunSpec) -> Path:
    return root / "runs" / f"{spec.key}.json"


def _checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def _reference_representation_name(representation: str) -> str:
    return {
        "relative10": "relative_10bin",
        "fixed250": "fixed_250ms",
        "fixed500": "fixed_500ms",
    }[representation]


def run_one(spec: RunSpec, cohort: Cohort, config: Config) -> dict[str, object]:
    out_path = _artifact_path(config.results_dir, spec)
    if config.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        expected = (PROTOCOL_VERSION, spec.representation, spec.decoder, spec.split_seed)
        actual = (
            str(cached.get("protocol_version")),
            str(cached.get("representation")),
            str(cached.get("decoder")),
            int(cached.get("split_seed")),
        )
        if actual != expected:
            raise ValueError(f"Run identity mismatch in {out_path}: {actual} != {expected}")
        return cached

    parts = make_user_split(cohort.manifest, spec.split_seed)
    built = {
        split_name: build_representation(cohort, frame, spec.representation)
        for split_name, frame in parts.items()
    }
    train_x, train_mask, train_y, meta = built["train"]
    val_x, val_mask, val_y, val_meta = built["val"]
    test_x, test_mask, test_y, test_meta = built["test"]
    if meta != val_meta or meta != test_meta:
        raise ValueError("Representation metadata differs across train/val/test")

    linear_seed = derive_seed(spec.split_seed, _reference_representation_name(spec.representation), "linear")
    baseline = fit_linear_baseline(train_x, train_y, val_x, val_y, test_x, test_y, linear_seed)

    model, training = train_residual(
        spec,
        train_x, train_mask, train_y,
        val_x, val_mask, val_y,
        test_x, test_mask, test_y,
        baseline["logits"],
        config,
    )
    split_users = _split_users(parts)
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "representation": spec.representation,
        "decoder": spec.decoder,
        "split_seed": int(spec.split_seed),
        "model_init_seed": int(MODEL_INIT_SEED),
        "effective_model_seed": int(training["effective_model_seed"]),
        "sampling_rate_hz": float(cohort.fs),
        "event_channels": EVENT_CHANNEL_COUNT,
        "residual_rank": RESIDUAL_RANK,
        "feature_dim": int(np.prod(train_x.shape[1:])),
        "n_time_bins": int(train_x.shape[1]),
        "requested_duration_ms": meta["requested_duration_ms"],
        "samples_per_bin": meta["samples_per_bin"],
        "actual_duration_ms": meta["actual_duration_ms"],
        "mask_policy": "local uses m_b; transition uses m_b*m_b+1; mask is never a classifier feature",
        "partial_final_bin_policy": "keep true event count in the final partial fixed-duration bin",
        "scaler_policy": "linear: train-only per-feature z-score matching 1.3.3; residual: train-only valid-bin channel RMS",
        "cohort_hash": _sample_hash(parts),
        **split_users,
        "n_train": int(len(parts["train"])),
        "n_val": int(len(parts["val"])),
        "n_test": int(len(parts["test"])),
        "linear_seed": int(linear_seed),
        "linear_metrics": baseline["metrics"],
        "best_epoch": int(training["best_epoch"]),
        "n_epochs_ran": int(training["n_epochs_ran"]),
        "identity_error_epoch0": float(training["identity_error_epoch0"]),
        "n_trainable_params": int(training["n_trainable_params"]),
        "residual_metrics": training["metrics"],
        "test_delta_vs_linear": float(
            training["metrics"]["test"]["balanced_accuracy"]
            - baseline["metrics"]["test"]["balanced_accuracy"]
        ),
        "val_delta_vs_linear": float(
            training["metrics"]["val"]["balanced_accuracy"]
            - baseline["metrics"]["val"]["balanced_accuracy"]
        ),
        "channel_rms_scale": training["channel_rms_scale"],
        "history": training["history"],
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
            "representation": spec.representation,
            "decoder": spec.decoder,
            "split_seed": spec.split_seed,
            "effective_model_seed": training["effective_model_seed"],
            "best_epoch": training["best_epoch"],
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    return payload


def _load_reference_linear_results(repo_root: Path) -> pd.DataFrame:
    path = (
        repo_root
        / "notebooks"
        / "artifacts"
        / REFERENCE_EXPERIMENT
        / "experiment_1_3_3_results.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing Experiment 1.3.3 reference results: {path}")
    frame = pd.read_csv(path)
    return frame[frame.model == "linear"].copy()


def _assert_complete_and_consistent(payloads: list[dict[str, object]]) -> None:
    if len(payloads) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} runs, got {len(payloads)}")
    keys = {
        (str(p["representation"]), str(p["decoder"]), int(p["split_seed"]))
        for p in payloads
    }
    if len(keys) != EXPECTED_RUNS:
        raise ValueError("Duplicate Experiment 3.2 run identities found")

    for representation in REPRESENTATIONS:
        for split_seed in SPLIT_SEEDS:
            group = [
                p for p in payloads
                if p["representation"] == representation and int(p["split_seed"]) == split_seed
            ]
            if len(group) != len(RESIDUAL_DECODERS):
                raise ValueError(f"Incomplete paired group for {representation}/split{split_seed}")
            hashes = {str(p["cohort_hash"]) for p in group}
            if len(hashes) != 1:
                raise ValueError(f"Cohort mismatch for {representation}/split{split_seed}")
            user_triplets = {
                (
                    tuple(p["train_users"]),
                    tuple(p["val_users"]),
                    tuple(p["test_users"]),
                )
                for p in group
            }
            if len(user_triplets) != 1:
                raise ValueError(f"User split mismatch for {representation}/split{split_seed}")
            baseline_values = np.asarray(
                [float(p["linear_metrics"]["test"]["balanced_accuracy"]) for p in group]
            )
            if not np.allclose(baseline_values, baseline_values[0], rtol=0.0, atol=1e-12):
                raise ValueError(f"Repeated Linear baseline mismatch for {representation}/split{split_seed}")


def _rows_from_payloads(payloads: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        linear = payload["linear_metrics"]
        residual = payload["residual_metrics"]
        rows.append(
            {
                "representation": payload["representation"],
                "decoder": payload["decoder"],
                "split_seed": payload["split_seed"],
                "best_epoch": payload["best_epoch"],
                "n_trainable_params": payload["n_trainable_params"],
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
        )
    return pd.DataFrame(rows)


def _baseline_reproduction(
    results: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    dedup = (
        results.sort_values(["representation", "split_seed", "decoder"])
        .drop_duplicates(["representation", "split_seed"], keep="first")
    )
    rows: list[dict[str, object]] = []
    for row in dedup.itertuples(index=False):
        ref_name = _reference_representation_name(str(row.representation))
        ref = reference[
            (reference.representation == ref_name)
            & (reference.split_seed == int(row.split_seed))
        ]
        if len(ref) != 1:
            raise ValueError(f"Missing 1.3.3 Linear reference for {ref_name}/split{row.split_seed}")
        ref_ba = float(ref.iloc[0].test_balanced_accuracy)
        difference = float(row.linear_test_ba - ref_ba)
        rows.append(
            {
                "representation": row.representation,
                "split_seed": int(row.split_seed),
                "exp32_linear_test_ba": float(row.linear_test_ba),
                "exp133_linear_test_ba": ref_ba,
                "difference": difference,
                "abs_difference": abs(difference),
            }
        )
    frame = pd.DataFrame(rows)
    worst = float(frame.abs_difference.max())
    if worst > BASELINE_REPRO_TOL:
        raise ValueError(
            f"Experiment 3.2 Linear baseline drift exceeds {BASELINE_REPRO_TOL:.3f}: max abs diff={worst:.6f}"
        )
    return frame


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (representation, decoder), group in results.groupby(["representation", "decoder"], sort=True):
        values = group.residual_test_ba.astype(float)
        deltas = group.test_delta_vs_linear.astype(float)
        rows.append(
            {
                "representation": representation,
                "decoder": decoder,
                "n_splits": int(len(group)),
                "mean_test_ba": float(values.mean()),
                "sd_test_ba": float(values.std(ddof=1)),
                "mean_test_macro_f1": float(group.residual_test_macro_f1.astype(float).mean()),
                "mean_delta_vs_linear": float(deltas.mean()),
                "sd_delta_vs_linear": float(deltas.std(ddof=1)),
                "median_delta_vs_linear": float(deltas.median()),
                "min_delta_vs_linear": float(deltas.min()),
                "max_delta_vs_linear": float(deltas.max()),
                "improved_splits": int((deltas > 0).sum()),
                "n_trainable_params": int(group.n_trainable_params.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for representation in REPRESENTATIONS:
        for split_seed in SPLIT_SEEDS:
            group = results[
                (results.representation == representation)
                & (results.split_seed == split_seed)
            ].set_index("decoder")
            local = float(group.loc["local", "residual_test_ba"])
            transition = float(group.loc["transition", "residual_test_ba"])
            combined = float(group.loc["local_transition", "residual_test_ba"])
            linear = float(group.loc["transition", "linear_test_ba"])
            rows.append(
                {
                    "representation": representation,
                    "split_seed": split_seed,
                    "linear_test_ba": linear,
                    "local_test_ba": local,
                    "transition_test_ba": transition,
                    "local_transition_test_ba": combined,
                    "local_minus_linear": local - linear,
                    "transition_minus_linear": transition - linear,
                    "local_transition_minus_linear": combined - linear,
                    "transition_minus_local": transition - local,
                    "local_transition_minus_transition": combined - transition,
                }
            )
    return pd.DataFrame(rows)


def _mechanism_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for representation, group in deltas.groupby("representation", sort=True):
        values = group.transition_minus_local.astype(float)
        rows.append(
            {
                "representation": representation,
                "mean_transition_minus_local": float(values.mean()),
                "sd_transition_minus_local": float(values.std(ddof=1)),
                "median_transition_minus_local": float(values.median()),
                "transition_better_splits": int((values > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    payloads: list[dict[str, object]] = []
    for spec in run_specs():
        path = _artifact_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing Experiment 3.2 run artifact: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = (PROTOCOL_VERSION, spec.representation, spec.decoder, spec.split_seed)
        actual = (
            str(payload.get("protocol_version")),
            str(payload.get("representation")),
            str(payload.get("decoder")),
            int(payload.get("split_seed")),
        )
        if actual != expected:
            raise ValueError(f"Run identity mismatch in {path}: {actual} != {expected}")
        payloads.append(payload)

    _assert_complete_and_consistent(payloads)
    results = _rows_from_payloads(payloads)
    reference = _load_reference_linear_results(repo_root)
    reproduction = _baseline_reproduction(results, reference)
    summary = _summary(results)
    deltas = _paired_deltas(results)
    mechanism = _mechanism_summary(deltas)

    baseline = (
        results.sort_values(["representation", "split_seed", "decoder"])
        .drop_duplicates(["representation", "split_seed"], keep="first")
    )
    baseline_summary = (
        baseline.groupby("representation", sort=True)
        .agg(
            n_splits=("split_seed", "size"),
            mean_linear_test_ba=("linear_test_ba", "mean"),
            sd_linear_test_ba=("linear_test_ba", "std"),
            mean_linear_test_macro_f1=("linear_test_macro_f1", "mean"),
        )
        .reset_index()
    )

    fixed250 = deltas[deltas.representation == "fixed250"]
    conclusion = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "primary_question": (
            "Do nonlinear interactions between adjacent temporal segments contain discriminative "
            "information beyond a position-aware Linear decoder?"
        ),
        "primary_representation": "fixed250",
        "fixed250_transition_mean_delta_vs_linear": float(fixed250.transition_minus_linear.mean()),
        "fixed250_transition_improved_splits": int((fixed250.transition_minus_linear > 0).sum()),
        "fixed250_transition_minus_local_mean": float(fixed250.transition_minus_local.mean()),
        "fixed250_transition_better_than_local_splits": int((fixed250.transition_minus_local > 0).sum()),
        "baseline_reproduction_max_abs_difference": float(reproduction.abs_difference.max()),
    }

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results": root / "experiment_3_2_results.csv",
        "paired_deltas": root / "experiment_3_2_paired_deltas.csv",
        "summary": root / "experiment_3_2_summary.csv",
        "mechanism_summary": root / "experiment_3_2_mechanism_summary.csv",
        "baseline_summary": root / "experiment_3_2_linear_baseline_summary.csv",
        "baseline_reproduction": root / "experiment_3_2_baseline_reproduction.csv",
        "parameter_counts": root / "experiment_3_2_parameter_counts.csv",
        "conclusion": root / "experiment_3_2_conclusion.json",
        "provenance": root / "provenance.json",
    }
    results.to_csv(outputs["results"], index=False)
    deltas.to_csv(outputs["paired_deltas"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    mechanism.to_csv(outputs["mechanism_summary"], index=False)
    baseline_summary.to_csv(outputs["baseline_summary"], index=False)
    reproduction.to_csv(outputs["baseline_reproduction"], index=False)
    (
        results[["representation", "decoder", "n_trainable_params"]]
        .drop_duplicates()
        .sort_values(["representation", "decoder"])
        .to_csv(outputs["parameter_counts"], index=False)
    )
    with outputs["conclusion"].open("w", encoding="utf-8") as handle:
        json.dump(conclusion, handle, indent=2, sort_keys=True)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "split_seeds": list(SPLIT_SEEDS),
        "representations": list(REPRESENTATIONS),
        "residual_decoders": list(RESIDUAL_DECODERS),
        "expected_runs": EXPECTED_RUNS,
        "model_init_seed": MODEL_INIT_SEED,
        "residual_rank": RESIDUAL_RANK,
        "sampling_rate_hz": EXPECTED_SAMPLING_RATE_HZ,
        "event_channels": EVENT_CHANNEL_COUNT,
        "linear_protocol": "1.3.3 train-only per-feature z-score + lbfgs LogisticRegression",
        "residual_scaling": "train-only valid-bin channel RMS",
        "selection_metric": "validation balanced accuracy; tie break validation CE then earlier epoch",
        "test_policy": "test evaluated only after best validation checkpoint is selected",
    }
    with outputs["provenance"].open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
    return outputs


def _run_cli(args: argparse.Namespace) -> None:
    repo_root = find_repo_root()
    root = results_dir(repo_root)
    if args.command == "run-one":
        specs = run_specs()
        task_id = int(args.array_task_id)
        if task_id < 0 or task_id >= len(specs):
            raise ValueError(f"array task id {task_id} outside 0..{len(specs) - 1}")
        spec = specs[task_id]
        cohort = load_cohort(repo_root)
        config = Config(
            repo_root=repo_root,
            results_dir=root,
            device=args.device,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            resume=not args.force,
            threads=1,
        )
        result = run_one(spec, cohort, config)
        print(
            f"completed {spec.key} best_epoch={result['best_epoch']} "
            f"test_ba={result['residual_metrics']['test']['balanced_accuracy']:.6f} "
            f"delta={result['test_delta_vs_linear']:+.6f}"
        )
    elif args.command == "finalize":
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
    else:
        raise ValueError(args.command)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 3.2 low-rank nonlinear temporal interaction probe")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    run.add_argument("--force", action="store_true")
    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    _run_cli(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
