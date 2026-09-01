from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from scripts import with_gyro_experiment_0_1_temporal_representation_probe as exp01


EXPERIMENT_ID = "withGyro_experiment_0_2_nonlinear_temporal_decoder_probe"
PROTOCOL_VERSION = "structured_residual_gru_v1"
REFERENCE_EXPERIMENT = exp01.EXPERIMENT_ID
REFERENCE_PROTOCOL = exp01.PROTOCOL_VERSION

SPLIT_SEEDS = exp01.SPLIT_SEEDS
CHANNEL_SETS = exp01.CHANNEL_SETS
REPRESENTATIONS = ("fixed250", "relative10")
RESIDUAL_DECODERS = ("local", "transition", "local_transition")
NEURAL_DECODERS = (*RESIDUAL_DECODERS, "gru")

MODEL_INIT_SEED = 2026
RESIDUAL_RANK = 16
GRU_HIDDEN_SIZE = 32
BATCH_SIZE = 64
MAX_EPOCHS = 200
MIN_EPOCHS = 20
EARLY_STOP_PATIENCE = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
LOGREG_MAX_ITER = exp01.LOGREG_MAX_ITER
BASELINE_REPRO_TOL = 0.01
EPS = 1e-8

EXPECTED_BASELINES = len(SPLIT_SEEDS) * len(CHANNEL_SETS) * len(REPRESENTATIONS)
EXPECTED_NEURAL_RUNS = EXPECTED_BASELINES * len(NEURAL_DECODERS)


@dataclass(frozen=True)
class BaselineSpec:
    channel_set: str
    representation: str
    split_seed: int

    @property
    def key(self) -> str:
        return f"{self.channel_set}__{self.representation}__split{self.split_seed}"


@dataclass(frozen=True)
class RunSpec:
    channel_set: str
    representation: str
    decoder: str
    split_seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.channel_set}__{self.representation}__{self.decoder}"
            f"__split{self.split_seed}"
        )

    @property
    def baseline_spec(self) -> BaselineSpec:
        return BaselineSpec(self.channel_set, self.representation, self.split_seed)


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
    return exp01.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / "withGyro"
        / "experiment_0_2_nonlinear_temporal_decoder_probe"
        / PROTOCOL_VERSION
    )


def baseline_specs() -> list[BaselineSpec]:
    return [
        BaselineSpec(channel_set, representation, split_seed)
        for channel_set in CHANNEL_SETS
        for representation in REPRESENTATIONS
        for split_seed in SPLIT_SEEDS
    ]


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(channel_set, representation, decoder, split_seed)
        for channel_set in CHANNEL_SETS
        for representation in REPRESENTATIONS
        for decoder in NEURAL_DECODERS
        for split_seed in SPLIT_SEEDS
    ]


def _representation_condition(representation: str) -> str:
    if representation == "fixed250":
        return "fixed_0250ms"
    if representation == "relative10":
        return "relative_10bin"
    raise ValueError(f"Unknown representation: {representation}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return frame.label_idx.to_numpy(dtype=np.int64, copy=True)


def _split_users(parts: Mapping[str, pd.DataFrame]) -> dict[str, list[str]]:
    return {
        f"{name}_users": sorted(frame.user.unique().tolist())
        for name, frame in parts.items()
    }


def _sample_hash(parts: Mapping[str, pd.DataFrame]) -> str:
    return exp01._sample_hash(parts)


def build_temporal_representation(
    cohort: exp01.Cohort,
    frame: pd.DataFrame,
    representation: str,
    channel_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    start, stop = exp01._channel_bounds(channel_set)
    channel_count = stop - start

    if representation == "fixed250":
        samples_per_bin = int(np.rint(250.0 * cohort.fs / 1000.0))
        if samples_per_bin != 16:
            raise ValueError(f"Expected 16 samples for Fixed250 at 64 Hz, got {samples_per_bin}")
        n_bins = int(np.ceil(cohort.padded_length / samples_per_bin))
        rows: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for row in frame.itertuples(index=False):
            values = np.zeros((cohort.padded_length, channel_count), dtype=np.float32)
            events = exp01._sample_channel_events(cohort, row, channel_set)
            valid_length = min(len(events), cohort.padded_length)
            values[:valid_length] = events[:valid_length]
            counts = values.reshape(n_bins, samples_per_bin, channel_count).sum(axis=1)
            valid_bins = int(np.ceil(valid_length / samples_per_bin))
            mask = np.arange(n_bins) < valid_bins
            rows.append(counts)
            masks.append(mask)
        x = np.stack(rows).astype(np.float32)
        mask = np.stack(masks).astype(bool)
        meta = {
            "representation": representation,
            "condition": _representation_condition(representation),
            "channel_set": channel_set,
            "channel_start": start,
            "channel_stop": stop,
            "channel_count": channel_count,
            "n_time_bins": n_bins,
            "requested_duration_ms": 250.0,
            "samples_per_bin": samples_per_bin,
            "actual_duration_ms": float(1000.0 * samples_per_bin / cohort.fs),
            "feature_dim": int(n_bins * channel_count),
        }
        return x, mask, _labels(frame), meta

    if representation == "relative10":
        n_bins = 10
        rows = []
        for row in frame.itertuples(index=False):
            events = exp01._sample_channel_events(cohort, row, channel_set)
            if len(events) < n_bins:
                raise ValueError(f"Relative10 exceeds valid length {len(events)}")
            chunks = np.array_split(events, n_bins, axis=0)
            rows.append(np.stack([chunk.sum(axis=0) for chunk in chunks], axis=0))
        x = np.stack(rows).astype(np.float32)
        mask = np.ones((len(x), n_bins), dtype=bool)
        meta = {
            "representation": representation,
            "condition": _representation_condition(representation),
            "channel_set": channel_set,
            "channel_start": start,
            "channel_stop": stop,
            "channel_count": channel_count,
            "n_time_bins": n_bins,
            "requested_duration_ms": None,
            "samples_per_bin": None,
            "actual_duration_ms": None,
            "feature_dim": int(n_bins * channel_count),
        }
        return x, mask, _labels(frame), meta

    raise ValueError(f"Unknown representation: {representation}")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _fit_linear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    train_flat = np.asarray(train_x, dtype=np.float64).reshape(len(train_x), -1)
    val_flat = np.asarray(val_x, dtype=np.float64).reshape(len(val_x), -1)
    test_flat = np.asarray(test_x, dtype=np.float64).reshape(len(test_x), -1)
    mean, std = exp01.fit_standardizer(train_flat)
    standardized = {
        "train": exp01.apply_standardizer(train_flat, mean, std),
        "val": exp01.apply_standardizer(val_flat, mean, std),
        "test": exp01.apply_standardizer(test_flat, mean, std),
    }
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=LOGREG_MAX_ITER,
        random_state=seed,
    )
    model.fit(standardized["train"], train_y)
    labels = {"train": train_y, "val": val_y, "test": test_y}
    logits: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for split_name in ("train", "val", "test"):
        x = standardized[split_name]
        y = labels[split_name]
        logits[split_name] = np.asarray(model.decision_function(x), dtype=np.float32)
        metrics[split_name] = classification_metrics(y, model.predict(x))
    n_params = int(model.coef_.size + model.intercept_.size)
    return {
        "logits": logits,
        "metrics": metrics,
        "n_trainable_params": n_params,
    }


def _baseline_json_path(root: Path, spec: BaselineSpec) -> Path:
    return root / "baselines" / f"{spec.key}.json"


def _baseline_logits_path(root: Path, spec: BaselineSpec) -> Path:
    return root / "baseline_logits" / f"{spec.key}.npz"


def _run_json_path(root: Path, spec: RunSpec) -> Path:
    return root / "runs" / f"{spec.key}.json"


def _checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def fit_one_baseline(
    spec: BaselineSpec,
    cohort: exp01.Cohort,
    root: Path,
    *,
    force: bool = False,
) -> dict[str, object]:
    meta_path = _baseline_json_path(root, spec)
    logits_path = _baseline_logits_path(root, spec)
    if meta_path.exists() and logits_path.exists() and not force:
        payload = _load_json(meta_path)
        expected = (PROTOCOL_VERSION, spec.channel_set, spec.representation, spec.split_seed)
        actual = (
            str(payload.get("protocol_version")),
            str(payload.get("channel_set")),
            str(payload.get("representation")),
            int(payload.get("split_seed", -1)),
        )
        if actual != expected:
            raise ValueError(f"Cached baseline identity mismatch: {actual} != {expected}")
        return payload

    parts = exp01.make_user_split(cohort.manifest, spec.split_seed)
    built = {
        name: build_temporal_representation(cohort, frame, spec.representation, spec.channel_set)
        for name, frame in parts.items()
    }
    train_x, train_mask, train_y, meta = built["train"]
    val_x, val_mask, val_y, val_meta = built["val"]
    test_x, test_mask, test_y, test_meta = built["test"]
    if meta != val_meta or meta != test_meta:
        raise ValueError("Representation metadata differs across train/val/test")

    condition = _representation_condition(spec.representation)
    linear_seed = exp01.derive_seed(spec.split_seed, condition, spec.channel_set, "classifier")
    fitted = _fit_linear(
        train_x, train_y, val_x, val_y, test_x, test_y, seed=linear_seed
    )
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "reference_protocol": REFERENCE_PROTOCOL,
        "baseline_key": spec.key,
        "channel_set": spec.channel_set,
        "representation": spec.representation,
        "split_seed": spec.split_seed,
        "linear_seed": int(linear_seed),
        "sample_hash": _sample_hash(parts),
        **_split_users(parts),
        "n_train": int(len(parts["train"])),
        "n_val": int(len(parts["val"])),
        "n_test": int(len(parts["test"])),
        "representation_meta": meta,
        "n_trainable_params": int(fitted["n_trainable_params"]),
        "metrics": fitted["metrics"],
        "scaler_policy": "train-only per-feature z-score matching withGyro Experiment 0.1",
        "raw_imu_excluded": True,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logits = fitted["logits"]
    np.savez_compressed(
        logits_path,
        train=np.asarray(logits["train"], dtype=np.float32),
        val=np.asarray(logits["val"], dtype=np.float32),
        test=np.asarray(logits["test"], dtype=np.float32),
    )
    return payload


def run_all_baselines(repo_root: Path, *, force: bool = False) -> None:
    cohort = exp01.load_cohort(repo_root)
    root = results_dir(repo_root)
    for index, spec in enumerate(baseline_specs(), start=1):
        payload = fit_one_baseline(spec, cohort, root, force=force)
        print(
            f"[baseline {index:02d}/{EXPECTED_BASELINES}] {spec.key} "
            f"test BA={payload['metrics']['test']['balanced_accuracy']:.4f}"
        )


def _load_baseline(root: Path, spec: BaselineSpec) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    meta_path = _baseline_json_path(root, spec)
    logits_path = _baseline_logits_path(root, spec)
    if not meta_path.is_file() or not logits_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen Linear baseline for {spec.key}; run the baseline stage first"
        )
    payload = _load_json(meta_path)
    expected = (PROTOCOL_VERSION, spec.channel_set, spec.representation, spec.split_seed)
    actual = (
        str(payload.get("protocol_version")),
        str(payload.get("channel_set")),
        str(payload.get("representation")),
        int(payload.get("split_seed", -1)),
    )
    if actual != expected:
        raise ValueError(f"Frozen baseline identity mismatch: {actual} != {expected}")
    with np.load(logits_path, allow_pickle=False) as archive:
        logits = {name: np.asarray(archive[name], dtype=np.float32) for name in ("train", "val", "test")}
    return payload, logits


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


def fit_channel_rms_scale(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    mask64 = np.asarray(mask, dtype=np.float64)[..., None]
    denominator = max(float(mask64.sum()), 1.0)
    mean_square = (x64 * x64 * mask64).sum(axis=(0, 1)) / denominator
    return np.sqrt(np.maximum(mean_square, EPS)).astype(np.float32)


def apply_channel_scale(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) / scale[None, None, :]).astype(np.float32)


class ResidualDecoder(nn.Module):
    def __init__(
        self,
        decoder: str,
        input_dim: int,
        n_bins: int,
        n_classes: int,
        rank: int = RESIDUAL_RANK,
    ) -> None:
        super().__init__()
        if decoder not in RESIDUAL_DECODERS:
            raise ValueError(f"Unknown residual decoder: {decoder}")
        self.decoder = decoder
        self.input_dim = int(input_dim)
        self.n_bins = int(n_bins)
        self.rank = int(rank)
        self.n_classes = int(n_classes)

        if decoder in ("local", "local_transition"):
            self.local_projection = nn.Linear(input_dim, rank, bias=False)
            self.local_head = nn.Linear(n_bins * rank, n_classes, bias=False)
            nn.init.zeros_(self.local_head.weight)
        else:
            self.local_projection = None
            self.local_head = None

        if decoder in ("transition", "local_transition"):
            self.transition_p = nn.Linear(input_dim, rank, bias=False)
            self.transition_q = nn.Linear(input_dim, rank, bias=False)
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
            transition = left * right * pair_mask
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


class GRUDecoder(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_size: int = GRU_HIDDEN_SIZE) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        del base_logits
        lengths = mask.sum(dim=1).to(dtype=torch.long)
        if torch.any(lengths <= 0):
            raise ValueError("GRU received a sequence with no valid bins")
        packed = pack_padded_sequence(
            x,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.head(hidden[-1])


def _model_seed(spec: RunSpec) -> int:
    return exp01.derive_seed(
        MODEL_INIT_SEED,
        spec.channel_set,
        spec.representation,
        spec.decoder,
    )


def _make_loader(
    x: np.ndarray,
    mask: np.ndarray,
    base_logits: np.ndarray,
    y: np.ndarray,
    *,
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
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    truth: list[np.ndarray] = []
    pred: list[np.ndarray] = []
    with torch.no_grad():
        for x, mask, base_logits, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            base_logits = base_logits.to(device)
            y = y.to(device)
            logits = model(x, mask, base_logits)
            total_loss += float(F.cross_entropy(logits, y, reduction="sum").item())
            total_count += int(len(y))
            truth.append(y.cpu().numpy())
            pred.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(truth)
    y_pred = np.concatenate(pred)
    return {
        "loss": total_loss / max(total_count, 1),
        **classification_metrics(y_true, y_pred),
    }


def _zero_residual_identity_error(
    model: ResidualDecoder,
    x: np.ndarray,
    mask: np.ndarray,
    base_logits: np.ndarray,
    device: torch.device,
) -> float:
    count = min(len(x), 16)
    with torch.no_grad():
        x_t = torch.tensor(x[:count], dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask[:count], dtype=torch.bool, device=device)
        base_t = torch.tensor(base_logits[:count], dtype=torch.float32, device=device)
        logits = model(x_t, mask_t, base_t)
    return float(torch.max(torch.abs(logits - base_t)).item())


def train_neural_decoder(
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
) -> tuple[nn.Module, dict[str, object]]:
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
    n_classes = len(np.unique(train_y))
    if spec.decoder in RESIDUAL_DECODERS:
        model: nn.Module = ResidualDecoder(
            spec.decoder,
            input_dim=train_x.shape[2],
            n_bins=train_x.shape[1],
            n_classes=n_classes,
            rank=RESIDUAL_RANK,
        )
    elif spec.decoder == "gru":
        model = GRUDecoder(
            input_dim=train_x.shape[2],
            n_classes=n_classes,
            hidden_size=GRU_HIDDEN_SIZE,
        )
    else:
        raise ValueError(f"Unknown neural decoder: {spec.decoder}")
    model = model.to(device)

    identity_error: float | None = None
    if isinstance(model, ResidualDecoder):
        identity_error = _zero_residual_identity_error(
            model, scaled["val"], val_mask, base_logits["val"], device
        )
        if identity_error > 1e-6:
            raise RuntimeError(f"Zero-residual identity check failed: {identity_error}")

    loaders = {
        "train": _make_loader(
            scaled["train"], train_mask, base_logits["train"], train_y,
            batch_size=config.batch_size,
            shuffle=True,
            seed=exp01.derive_seed(effective_seed, "train_loader"),
        ),
        "train_eval": _make_loader(
            scaled["train"], train_mask, base_logits["train"], train_y,
            batch_size=config.batch_size,
            shuffle=False,
            seed=exp01.derive_seed(effective_seed, "train_eval_loader"),
        ),
        "val": _make_loader(
            scaled["val"], val_mask, base_logits["val"], val_y,
            batch_size=config.batch_size,
            shuffle=False,
            seed=exp01.derive_seed(effective_seed, "val_loader"),
        ),
        "test": _make_loader(
            scaled["test"], test_mask, base_logits["test"], test_y,
            batch_size=config.batch_size,
            shuffle=False,
            seed=exp01.derive_seed(effective_seed, "test_loader"),
        ),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if isinstance(model, ResidualDecoder):
        epoch0 = _evaluate_model(model, loaders["val"], device)
        history.append(
            {"epoch": 0, "train_loss": float("nan"), **{f"val_{k}": v for k, v in epoch0.items()}}
        )
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        best_val_ba = float(epoch0["balanced_accuracy"])
        best_val_loss = float(epoch0["loss"])

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        for x, mask, frozen_logits, y in loaders["train"]:
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

        val_metrics = _evaluate_model(model, loaders["val"], device)
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

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint candidate")
    model.load_state_dict(best_state)
    metrics = {
        "train": _evaluate_model(model, loaders["train_eval"], device),
        "val": _evaluate_model(model, loaders["val"], device),
        "test": _evaluate_model(model, loaders["test"], device),
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


def run_one(spec: RunSpec, cohort: exp01.Cohort, config: Config) -> dict[str, object]:
    out_path = _run_json_path(config.results_dir, spec)
    checkpoint_path = _checkpoint_path(config.results_dir, spec)
    if config.resume and out_path.exists() and checkpoint_path.exists():
        payload = _load_json(out_path)
        expected = (
            PROTOCOL_VERSION,
            spec.channel_set,
            spec.representation,
            spec.decoder,
            spec.split_seed,
        )
        actual = (
            str(payload.get("protocol_version")),
            str(payload.get("channel_set")),
            str(payload.get("representation")),
            str(payload.get("decoder")),
            int(payload.get("split_seed", -1)),
        )
        if actual != expected:
            raise ValueError(f"Cached run identity mismatch: {actual} != {expected}")
        return payload

    baseline_meta, base_logits = _load_baseline(config.results_dir, spec.baseline_spec)
    parts = exp01.make_user_split(cohort.manifest, spec.split_seed)
    if _sample_hash(parts) != baseline_meta["sample_hash"]:
        raise ValueError(f"Cohort hash mismatch for frozen baseline {spec.baseline_spec.key}")
    built = {
        name: build_temporal_representation(cohort, frame, spec.representation, spec.channel_set)
        for name, frame in parts.items()
    }
    train_x, train_mask, train_y, meta = built["train"]
    val_x, val_mask, val_y, val_meta = built["val"]
    test_x, test_mask, test_y, test_meta = built["test"]
    if meta != val_meta or meta != test_meta or meta != baseline_meta["representation_meta"]:
        raise ValueError("Representation metadata differs from frozen baseline")
    for split_name, y in (("train", train_y), ("val", val_y), ("test", test_y)):
        if len(base_logits[split_name]) != len(y):
            raise ValueError(f"Frozen baseline logit count mismatch for {split_name}")

    model, training = train_neural_decoder(
        spec,
        train_x, train_mask, train_y,
        val_x, val_mask, val_y,
        test_x, test_mask, test_y,
        base_logits,
        config,
    )
    residual = spec.decoder in RESIDUAL_DECODERS
    test_ba = float(training["metrics"]["test"]["balanced_accuracy"])
    linear_test_ba = float(baseline_meta["metrics"]["test"]["balanced_accuracy"])
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "reference_protocol": REFERENCE_PROTOCOL,
        "run_key": spec.key,
        "channel_set": spec.channel_set,
        "representation": spec.representation,
        "decoder": spec.decoder,
        "decoder_family": "frozen_linear_residual" if residual else "standalone_recurrent",
        "split_seed": spec.split_seed,
        "model_init_seed": MODEL_INIT_SEED,
        "effective_model_seed": int(training["effective_model_seed"]),
        "sample_hash": baseline_meta["sample_hash"],
        **_split_users(parts),
        "representation_meta": meta,
        "residual_rank": RESIDUAL_RANK if residual else None,
        "gru_hidden_size": GRU_HIDDEN_SIZE if spec.decoder == "gru" else None,
        "mask_policy": (
            "Local uses m_b; Transition uses m_b*m_(b+1); GRU uses packed valid-bin lengths; "
            "mask is never an explicit classifier feature"
        ),
        "partial_final_bin_policy": "retain the true event count in the final partial Fixed250 bin",
        "scaler_policy": "train-only valid-bin per-channel RMS for neural decoder inputs",
        "linear_metrics": baseline_meta["metrics"],
        "metrics": training["metrics"],
        "best_epoch": int(training["best_epoch"]),
        "n_epochs_ran": int(training["n_epochs_ran"]),
        "n_trainable_params": int(training["n_trainable_params"]),
        "identity_error_epoch0": training["identity_error_epoch0"],
        "test_delta_vs_linear": test_ba - linear_test_ba,
        "val_delta_vs_linear": float(
            training["metrics"]["val"]["balanced_accuracy"]
            - baseline_meta["metrics"]["val"]["balanced_accuracy"]
        ),
        "channel_rms_scale": training["channel_rms_scale"],
        "history": training["history"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "channel_set": spec.channel_set,
            "representation": spec.representation,
            "decoder": spec.decoder,
            "split_seed": spec.split_seed,
            "effective_model_seed": training["effective_model_seed"],
            "best_epoch": training["best_epoch"],
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return payload


def _load_all_baselines(root: Path) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for spec in baseline_specs():
        path = _baseline_json_path(root, spec)
        if not path.is_file():
            raise FileNotFoundError(f"Missing baseline artifact: {path}")
        payload = _load_json(path)
        expected = (spec.channel_set, spec.representation, spec.split_seed)
        actual = (
            str(payload.get("channel_set")),
            str(payload.get("representation")),
            int(payload.get("split_seed", -1)),
        )
        if payload.get("protocol_version") != PROTOCOL_VERSION or actual != expected:
            raise ValueError(f"Baseline identity mismatch in {path}")
        payloads.append(payload)
    return payloads


def _load_all_runs(root: Path) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for spec in run_specs():
        path = _run_json_path(root, spec)
        checkpoint = _checkpoint_path(root, spec)
        if not path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"Missing neural run/checkpoint for {spec.key}")
        payload = _load_json(path)
        expected = (spec.channel_set, spec.representation, spec.decoder, spec.split_seed)
        actual = (
            str(payload.get("channel_set")),
            str(payload.get("representation")),
            str(payload.get("decoder")),
            int(payload.get("split_seed", -1)),
        )
        if payload.get("protocol_version") != PROTOCOL_VERSION or actual != expected:
            raise ValueError(f"Neural run identity mismatch in {path}")
        payloads.append(payload)
    return payloads


def _assert_paired_consistency(
    baselines: list[dict[str, object]],
    runs: list[dict[str, object]],
) -> None:
    baseline_map = {
        (str(p["channel_set"]), str(p["representation"]), int(p["split_seed"])): p
        for p in baselines
    }
    if len(baseline_map) != EXPECTED_BASELINES:
        raise ValueError("Duplicate or missing frozen Linear baselines")
    run_keys = {
        (
            str(p["channel_set"]),
            str(p["representation"]),
            str(p["decoder"]),
            int(p["split_seed"]),
        )
        for p in runs
    }
    if len(run_keys) != EXPECTED_NEURAL_RUNS:
        raise ValueError("Duplicate or missing neural run identities")
    for payload in runs:
        key = (
            str(payload["channel_set"]),
            str(payload["representation"]),
            int(payload["split_seed"]),
        )
        baseline = baseline_map[key]
        if payload["sample_hash"] != baseline["sample_hash"]:
            raise ValueError(f"Cohort mismatch for {key}")
        for split_name in ("train_users", "val_users", "test_users"):
            if payload[split_name] != baseline[split_name]:
                raise ValueError(f"User split mismatch for {key}/{split_name}")
        repeated = float(payload["linear_metrics"]["test"]["balanced_accuracy"])
        frozen = float(baseline["metrics"]["test"]["balanced_accuracy"])
        if not np.isclose(repeated, frozen, rtol=0.0, atol=1e-12):
            raise ValueError(f"Frozen Linear metric drift for {key}")


def _result_rows(
    baselines: list[dict[str, object]],
    runs: list[dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for payload in baselines:
        metrics = payload["metrics"]
        meta = payload["representation_meta"]
        rows.append(
            {
                "channel_set": payload["channel_set"],
                "representation": payload["representation"],
                "decoder": "linear",
                "decoder_family": "linear",
                "split_seed": payload["split_seed"],
                "best_epoch": np.nan,
                "n_trainable_params": payload["n_trainable_params"],
                "n_time_bins": meta["n_time_bins"],
                "channel_count": meta["channel_count"],
                "feature_dim": meta["feature_dim"],
                "train_ba": metrics["train"]["balanced_accuracy"],
                "val_ba": metrics["val"]["balanced_accuracy"],
                "test_ba": metrics["test"]["balanced_accuracy"],
                "test_accuracy": metrics["test"]["accuracy"],
                "test_macro_f1": metrics["test"]["macro_f1"],
                "test_delta_vs_linear": 0.0,
            }
        )
    for payload in runs:
        metrics = payload["metrics"]
        meta = payload["representation_meta"]
        rows.append(
            {
                "channel_set": payload["channel_set"],
                "representation": payload["representation"],
                "decoder": payload["decoder"],
                "decoder_family": payload["decoder_family"],
                "split_seed": payload["split_seed"],
                "best_epoch": payload["best_epoch"],
                "n_trainable_params": payload["n_trainable_params"],
                "n_time_bins": meta["n_time_bins"],
                "channel_count": meta["channel_count"],
                "feature_dim": meta["feature_dim"],
                "train_ba": metrics["train"]["balanced_accuracy"],
                "val_ba": metrics["val"]["balanced_accuracy"],
                "test_ba": metrics["test"]["balanced_accuracy"],
                "test_accuracy": metrics["test"]["accuracy"],
                "test_macro_f1": metrics["test"]["macro_f1"],
                "test_delta_vs_linear": payload["test_delta_vs_linear"],
            }
        )
    frame = pd.DataFrame(rows)
    expected = EXPECTED_BASELINES * (1 + len(NEURAL_DECODERS))
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} finalized rows, got {len(frame)}")
    return frame.sort_values(["representation", "channel_set", "decoder", "split_seed"]).reset_index(drop=True)


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["representation", "channel_set", "decoder"], as_index=False)
        .agg(
            n_splits=("split_seed", "nunique"),
            mean_test_ba=("test_ba", "mean"),
            sd_test_ba=("test_ba", "std"),
            mean_test_accuracy=("test_accuracy", "mean"),
            sd_test_accuracy=("test_accuracy", "std"),
            mean_test_macro_f1=("test_macro_f1", "mean"),
            sd_test_macro_f1=("test_macro_f1", "std"),
            mean_delta_vs_linear=("test_delta_vs_linear", "mean"),
            sd_delta_vs_linear=("test_delta_vs_linear", "std"),
            improved_splits=("test_delta_vs_linear", lambda s: int((s > 0).sum())),
        )
        .sort_values(["representation", "channel_set", "decoder"])
        .reset_index(drop=True)
    )


def _paired_decoder_deltas(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected_decoders = {"linear", *NEURAL_DECODERS}
    for (representation, channel_set, split_seed), group in results.groupby(
        ["representation", "channel_set", "split_seed"], sort=True
    ):
        values = group.set_index("decoder")["test_ba"].astype(float)
        if set(values.index) != expected_decoders:
            raise ValueError(f"Incomplete decoder group for {representation}/{channel_set}/split{split_seed}")
        linear = float(values["linear"])
        local = float(values["local"])
        transition = float(values["transition"])
        local_transition = float(values["local_transition"])
        gru = float(values["gru"])
        rows.append(
            {
                "representation": representation,
                "channel_set": channel_set,
                "split_seed": int(split_seed),
                "linear_test_ba": linear,
                "local_test_ba": local,
                "transition_test_ba": transition,
                "local_transition_test_ba": local_transition,
                "gru_test_ba": gru,
                "local_minus_linear": local - linear,
                "transition_minus_linear": transition - linear,
                "local_transition_minus_linear": local_transition - linear,
                "gru_minus_linear": gru - linear,
                "transition_minus_local": transition - local,
                "local_transition_minus_transition": local_transition - transition,
                "local_transition_minus_local": local_transition - local,
                "local_transition_minus_gru": local_transition - gru,
            }
        )
    return pd.DataFrame(rows)


def _sensor_deltas(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected_channels = set(CHANNEL_SETS)
    for (representation, decoder, split_seed), group in results.groupby(
        ["representation", "decoder", "split_seed"], sort=True
    ):
        values = group.set_index("channel_set")["test_ba"].astype(float)
        if set(values.index) != expected_channels:
            raise ValueError(f"Incomplete sensor group for {representation}/{decoder}/split{split_seed}")
        accel = float(values["accel30"])
        angular = float(values["angular30"])
        combined = float(values["combined60"])
        rows.append(
            {
                "representation": representation,
                "decoder": decoder,
                "split_seed": int(split_seed),
                "accel30_test_ba": accel,
                "angular30_test_ba": angular,
                "combined60_test_ba": combined,
                "combined_minus_accel": combined - accel,
                "combined_minus_angular": combined - angular,
                "angular_minus_accel": angular - accel,
            }
        )
    return pd.DataFrame(rows)


def _nonlinear_synergy(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for representation in REPRESENTATIONS:
        for split_seed in SPLIT_SEEDS:
            group = paired[
                (paired.representation == representation)
                & (paired.split_seed == split_seed)
            ].set_index("channel_set")
            for decoder, column in (
                ("local", "local_minus_linear"),
                ("transition", "transition_minus_linear"),
                ("local_transition", "local_transition_minus_linear"),
            ):
                accel_gain = float(group.loc["accel30", column])
                angular_gain = float(group.loc["angular30", column])
                combined_gain = float(group.loc["combined60", column])
                rows.append(
                    {
                        "representation": representation,
                        "decoder": decoder,
                        "split_seed": split_seed,
                        "accel30_nonlinear_gain": accel_gain,
                        "angular30_nonlinear_gain": angular_gain,
                        "combined60_nonlinear_gain": combined_gain,
                        "combined_gain_minus_angular_gain": combined_gain - angular_gain,
                        "combined_gain_minus_accel_gain": combined_gain - accel_gain,
                    }
                )
    return pd.DataFrame(rows)


def _baseline_reproduction(repo_root: Path, baselines: list[dict[str, object]]) -> pd.DataFrame:
    reference_path = exp01.results_dir(repo_root) / "experiment_0_1_results.csv"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing withGyro Experiment 0.1 finalized results: {reference_path}")
    reference = pd.read_csv(reference_path)
    reference = reference[
        (reference.classifier == "linear")
        & (reference.eval_split.isin(["val", "test"]))
        & (reference.condition.isin(["fixed_0250ms", "relative_10bin"]))
    ].copy()
    rows: list[dict[str, object]] = []
    for payload in baselines:
        condition = _representation_condition(str(payload["representation"]))
        for eval_split in ("val", "test"):
            ref = reference[
                (reference.channel_set == payload["channel_set"])
                & (reference.condition == condition)
                & (reference.split_seed == int(payload["split_seed"]))
                & (reference.eval_split == eval_split)
            ]
            if len(ref) != 1:
                raise ValueError(
                    f"Missing unique Experiment 0.1 reference for {payload['baseline_key']}/{eval_split}"
                )
            new_ba = float(payload["metrics"][eval_split]["balanced_accuracy"])
            ref_ba = float(ref.iloc[0].balanced_accuracy)
            rows.append(
                {
                    "channel_set": payload["channel_set"],
                    "representation": payload["representation"],
                    "split_seed": payload["split_seed"],
                    "eval_split": eval_split,
                    "exp02_linear_ba": new_ba,
                    "exp01_linear_ba": ref_ba,
                    "difference": new_ba - ref_ba,
                    "abs_difference": abs(new_ba - ref_ba),
                }
            )
    frame = pd.DataFrame(rows)
    worst = float(frame.abs_difference.max())
    if worst > BASELINE_REPRO_TOL:
        raise ValueError(
            f"Experiment 0.2 Linear baseline drift exceeds {BASELINE_REPRO_TOL:.3f}: "
            f"max abs diff={worst:.6f}"
        )
    return frame


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    baselines = _load_all_baselines(root)
    runs = _load_all_runs(root)
    _assert_paired_consistency(baselines, runs)

    results = _result_rows(baselines, runs)
    summary = _summary(results)
    paired = _paired_decoder_deltas(results)
    sensor = _sensor_deltas(results)
    synergy = _nonlinear_synergy(paired)
    reproduction = _baseline_reproduction(repo_root, baselines)
    parameter_counts = (
        results[["representation", "channel_set", "decoder", "n_trainable_params"]]
        .drop_duplicates()
        .sort_values(["representation", "channel_set", "decoder"])
        .reset_index(drop=True)
    )

    local_transition = paired.copy()
    conclusion = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "primary_question": (
            "Where does additional information from angular events and accel+angular fusion "
            "become accessible: linearly, through structured local/transition nonlinearities, "
            "or through a generic GRU?"
        ),
        "linear_baseline_reproduction_max_abs_difference": float(reproduction.abs_difference.max()),
        "fixed250_combined_local_transition_mean_delta_vs_linear": float(
            local_transition[
                (local_transition.representation == "fixed250")
                & (local_transition.channel_set == "combined60")
            ].local_transition_minus_linear.mean()
        ),
        "relative10_combined_local_transition_mean_delta_vs_linear": float(
            local_transition[
                (local_transition.representation == "relative10")
                & (local_transition.channel_set == "combined60")
            ].local_transition_minus_linear.mean()
        ),
        "fixed250_combined_local_transition_minus_gru_mean": float(
            local_transition[
                (local_transition.representation == "fixed250")
                & (local_transition.channel_set == "combined60")
            ].local_transition_minus_gru.mean()
        ),
        "relative10_combined_local_transition_minus_gru_mean": float(
            local_transition[
                (local_transition.representation == "relative10")
                & (local_transition.channel_set == "combined60")
            ].local_transition_minus_gru.mean()
        ),
    }

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results": root / "experiment_0_2_results.csv",
        "summary": root / "experiment_0_2_summary.csv",
        "paired_decoder_deltas": root / "experiment_0_2_paired_decoder_deltas.csv",
        "sensor_deltas": root / "experiment_0_2_sensor_deltas.csv",
        "nonlinear_synergy": root / "experiment_0_2_nonlinear_synergy.csv",
        "parameter_counts": root / "experiment_0_2_parameter_counts.csv",
        "linear_reproduction": root / "experiment_0_2_linear_reproduction.csv",
        "conclusion": root / "experiment_0_2_conclusion.json",
        "provenance": root / "provenance.json",
    }
    results.to_csv(outputs["results"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    paired.to_csv(outputs["paired_decoder_deltas"], index=False)
    sensor.to_csv(outputs["sensor_deltas"], index=False)
    synergy.to_csv(outputs["nonlinear_synergy"], index=False)
    parameter_counts.to_csv(outputs["parameter_counts"], index=False)
    reproduction.to_csv(outputs["linear_reproduction"], index=False)
    outputs["conclusion"].write_text(
        json.dumps(conclusion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "reference_protocol": REFERENCE_PROTOCOL,
        "dataset_root": str(exp01.dataset_root(repo_root).resolve()),
        "split_seeds": list(SPLIT_SEEDS),
        "channel_sets": {key: list(value) for key, value in CHANNEL_SETS.items()},
        "channel_set_semantics": {
            "accel30": "linear-acceleration polarity-split wavelet events",
            "angular30": "gyro-derived angular-acceleration polarity-split wavelet events",
            "combined60": "accel30 concatenated with angular30",
        },
        "raw_imu_excluded": True,
        "representations": list(REPRESENTATIONS),
        "representation_semantics": {
            "fixed250": "16-sample (250 ms at 64 Hz) event-count bins over the 256-sample padded timeline; final partial valid bin retained",
            "relative10": "np.array_split of each valid gesture prefix into 10 relative-progress chunks, followed by channel-wise event sums",
        },
        "decoders": ["linear", *NEURAL_DECODERS],
        "residual_rank": RESIDUAL_RANK,
        "gru_hidden_size": GRU_HIDDEN_SIZE,
        "residual_definition": "frozen Linear logits plus zero-initialized Local, Transition, or Local+Transition residual logits",
        "gru_definition": "one-layer unidirectional GRU over the same binned representation; last valid packed hidden state -> Linear class head",
        "linear_scaling": "train-only per-feature z-score matching Experiment 0.1",
        "neural_scaling": "train-only per-channel RMS over valid bins",
        "model_selection": "highest validation Balanced Accuracy; ties use lower validation CE; residual epoch 0 is a valid frozen-Linear checkpoint",
        "training": {
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "grad_clip_norm": GRAD_CLIP_NORM,
        },
        "expected_baselines": EXPECTED_BASELINES,
        "expected_neural_runs": EXPECTED_NEURAL_RUNS,
        "slurm_execution": (
            "one CPU baseline job fits all 30 frozen Linear baselines; afterok launches 120 one-core "
            "neural array tasks; afterok finalizer aggregates existing artifacts only"
        ),
    }
    outputs["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "WithGyro Experiment 0.2: accel30/angular30/combined60 nonlinear temporal decoder probe"
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)
    baselines = sub.add_parser("run-baselines")
    baselines.add_argument("--force", action="store_true")
    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    sub.add_parser("describe")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = find_repo_root()
    root = results_dir(repo_root)
    if args.command == "describe":
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "protocol_version": PROTOCOL_VERSION,
                    "split_seeds": list(SPLIT_SEEDS),
                    "channel_sets": {key: list(value) for key, value in CHANNEL_SETS.items()},
                    "representations": list(REPRESENTATIONS),
                    "neural_decoders": list(NEURAL_DECODERS),
                    "expected_baselines": EXPECTED_BASELINES,
                    "expected_neural_runs": EXPECTED_NEURAL_RUNS,
                    "array_task_range": [0, EXPECTED_NEURAL_RUNS - 1],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "run-baselines":
        run_all_baselines(repo_root, force=bool(args.force))
        return
    if args.command == "finalize":
        outputs = finalize_experiment(repo_root)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return

    specs = run_specs()
    task_id = int(args.array_task_id)
    if not 0 <= task_id < len(specs):
        raise SystemExit(f"array-task-id must be in [0, {len(specs) - 1}], got {task_id}")
    cohort = exp01.load_cohort(repo_root)
    config = Config(
        repo_root=repo_root,
        results_dir=root,
        resume=not bool(args.force),
    )
    payload = run_one(specs[task_id], cohort, config)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
