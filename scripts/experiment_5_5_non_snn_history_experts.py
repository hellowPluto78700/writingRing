from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_5_4_3_elapsed_readout_capacity as exp543


exp54 = exp543.exp54
base = exp543.base

EXPERIMENT_ID = "experiment_5_5_non_snn_history_experts"
PROTOCOL_VERSION = "non_snn_history_experts_v1"
SEEDS = exp543.SEEDS
WHAT_WIDTH = exp543.WHAT_WIDTH
N_CLASSES = 12
GRU_WIDTH = 64
N_STATES = 8
EPOCHS = 200
PATIENCE = 25
BATCH_SIZE = exp543.BATCH_SIZE
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
RBF_SIGMA_MULTIPLIER = 1.5
SHUFFLE_REPLICATES = 5
PROFILE_RELATIVE_BINS = 10
PROFILE_ABSOLUTE_SECONDS = 0.250
Q_PROBE_MAX_ITER = 3000
SAME_WHAT_MAX_POINTS = 4000
SAME_WHAT_NEIGHBORS = 16
SAME_WHAT_TOP_PAIRS = 100
SAME_WHAT_HIGH_SIMILARITY = 0.90

SHARED = "shared"
CLOCK = "clock"
ORACLE_PROGRESS = "oracle_progress"
RESET_GRU = "reset_gru"
ORDERED_GRU = "ordered_gru"
CONDITIONS = (SHARED, CLOCK, ORACLE_PROGRESS, RESET_GRU, ORDERED_GRU)
GRU_CONDITIONS = (RESET_GRU, ORDERED_GRU)
EXPERT_CONDITIONS = (CLOCK, ORACLE_PROGRESS, RESET_GRU, ORDERED_GRU)


@dataclass(frozen=True)
class RunSpec:
    condition: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.condition}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    patience: int = PATIENCE
    batch_size: int = BATCH_SIZE
    threads: int = 1


@dataclass
class Trajectory:
    q: torch.Tensor | None
    evidence: torch.Tensor


def find_repo_root(start: Path | None = None) -> Path:
    return exp543.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def reference_model_path(root: Path, seed: int) -> Path:
    return root / "references" / f"seed{seed}.npz"


def reference_path(root: Path, seed: int) -> Path:
    return root / "references" / f"seed{seed}.json"


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def trajectory_path(root: Path, spec: RunSpec) -> Path:
    return root / "trajectories" / f"{spec.key}__test.npz"


def same_what_path(root: Path, spec: RunSpec) -> Path:
    return root / "same_what" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_specs() -> list[RunSpec]:
    return [RunSpec(condition, seed) for condition in CONDITIONS for seed in SEEDS]


def _validate_spec(spec: RunSpec) -> None:
    if spec.condition not in CONDITIONS or spec.seed not in SEEDS:
        raise ValueError(f"Invalid Exp5.5 run spec: {spec}")


def _exp54_config(config: Config) -> exp54.Config:
    return exp54.Config(
        repo_root=config.repo_root,
        results_dir=exp54.results_dir(config.repo_root),
        device=config.device,
        epochs=exp54.EPOCHS,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _labels_lengths(data: base.Data, split: str) -> tuple[np.ndarray, np.ndarray]:
    return {
        "train": (data.ytr, data.ltr),
        "val": (data.yva, data.lva),
        "test": (data.yte, data.lte),
    }[split]


def _load_what_arrays(
    seed: int,
    data: base.Data,
    config: Config,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays, meta = exp54.load_fusion_cache(seed, data, _exp54_config(config))
    what = {
        split: np.asarray(arrays[f"what_{split}"], dtype=np.float32)
        for split in ("train", "val", "test")
    }
    for split, values in what.items():
        labels, _ = _labels_lengths(data, split)
        if values.shape != (len(labels), data.T, WHAT_WIDTH):
            raise ValueError(f"Exp5.5 WHAT cache shape mismatch for {split}: {values.shape}")
    return what, meta


def _metrics_from_logits(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    true = np.asarray(labels, dtype=np.int64)
    values = np.asarray(logits, dtype=np.float64)
    pred = values.argmax(axis=1)
    shifted = values - values.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -float(log_probs[np.arange(len(true)), true].mean())
    return {
        "loss": loss,
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro")),
        "n_samples": int(len(true)),
    }


def prepare_reference_seed(
    seed: int,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Unknown Exp5.5 seed: {seed}")
    destination = reference_path(config.results_dir, seed)
    model_destination = reference_model_path(config.results_dir, seed)
    if destination.exists() and model_destination.exists() and not force:
        load_reference(seed, config)
        return destination

    source_config = _exp54_config(config)
    exp54.prepare_fusion_seed(seed, data, source_config, force=False)
    what, source_meta = _load_what_arrays(seed, data, config)

    train_features, bin_steps, n_bins = exp543.fixed250_features(what["train"], data.ltr, data)
    val_features, val_steps, val_bins = exp543.fixed250_features(what["val"], data.lva, data)
    test_features, test_steps, test_bins = exp543.fixed250_features(what["test"], data.lte, data)
    if (bin_steps, n_bins) != (val_steps, val_bins) or (bin_steps, n_bins) != (test_steps, test_bins):
        raise RuntimeError("Exp5.5 Fixed250 layout changed across splits")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)
    classifier = LogisticRegression(
        max_iter=exp543.LOGREG_MAX_ITER,
        solver="lbfgs",
        random_state=base.dseed(seed, EXPERIMENT_ID, "fixed250_reference"),
    )
    classifier.fit(train_scaled, data.ytr)
    expected_classes = np.arange(len(data.labels), dtype=np.int64)
    if not np.array_equal(classifier.classes_, expected_classes):
        raise RuntimeError("Exp5.5 Fixed250 class order mismatch")

    effective_weight = classifier.coef_.astype(np.float64) / scaler.scale_[None, :]
    effective_bias = classifier.intercept_.astype(np.float64) - effective_weight @ scaler.mean_.astype(np.float64)
    weight_bins = effective_weight.reshape(len(data.labels), n_bins, WHAT_WIDTH).transpose(1, 0, 2)
    offline = {
        "train": classifier.decision_function(train_scaled),
        "val": classifier.decision_function(val_scaled),
        "test": classifier.decision_function(test_scaled),
    }
    features = {"train": train_features, "val": val_features, "test": test_features}
    lengths = {"train": data.ltr, "val": data.lva, "test": data.lte}
    labels = {"train": data.ytr, "val": data.yva, "test": data.yte}
    equivalence: dict[str, float | bool] = {}
    metrics: dict[str, object] = {}
    for split in ("train", "val", "test"):
        raw = features[split] @ effective_weight.T + effective_bias
        stream = exp543.streaming_fixed250_logits(
            what[split],
            lengths[split],
            weight_bins,
            effective_bias,
            bin_steps,
        )
        raw_delta = float(np.max(np.abs(offline[split] - raw)))
        stream_delta = float(np.max(np.abs(offline[split] - stream)))
        if raw_delta > 1e-6 or stream_delta > 1e-6:
            raise RuntimeError(
                f"Exp5.5 Fixed250 offline/streaming equivalence failed for {split}: "
                f"raw={raw_delta:.3e}, stream={stream_delta:.3e}"
            )
        if not np.array_equal(offline[split].argmax(1), stream.argmax(1)):
            raise RuntimeError(f"Exp5.5 Fixed250 predictions changed in streaming form for {split}")
        equivalence[f"{split}_max_abs_offline_vs_raw"] = raw_delta
        equivalence[f"{split}_max_abs_offline_vs_streaming"] = stream_delta
        equivalence[f"{split}_predictions_identical"] = True
        metrics[split] = _metrics_from_logits(labels[split], offline[split])

    model_destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_destination,
        weight_bins=weight_bins,
        bias=effective_bias,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classes=classifier.classes_,
        bin_steps=np.asarray([bin_steps], dtype=np.int64),
        n_bins=np.asarray([n_bins], dtype=np.int64),
    )
    _save_json(
        destination,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "source_experiment_id": exp54.EXPERIMENT_ID,
            "source_protocol_version": exp54.PROTOCOL_VERSION,
            "source_what_definition": "frozen Local-SNN L2 spikes from the Exp5.4 fusion cache",
            "split_seed": base.SPLIT_SEED,
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "sampling_rate_hz": float(data.fs),
            "train_max_elapsed_seconds": float(source_meta["train_max_elapsed_seconds"]),
            "fixed250": {
                "requested_seconds": exp543.FIXED250_SECONDS,
                "bin_steps": int(bin_steps),
                "actual_seconds": float(bin_steps / data.fs),
                "n_bins": int(n_bins),
                "feature_dim": int(n_bins * WHAT_WIDTH),
                "classifier": "StandardScaler(train only) + LogisticRegression(lbfgs)",
                "metrics": metrics,
                "equivalence": equivalence,
            },
            "test_used_for_model_selection": False,
        },
    )
    load_reference(seed, config)
    return destination


def load_reference(seed: int, config: Config) -> dict[str, object]:
    path = reference_path(config.results_dir, seed)
    model_path = reference_model_path(config.results_dir, seed)
    if not path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing Exp5.5 reference for seed {seed}; run prepare-reference first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("seed") != seed
        or payload.get("test_used_for_model_selection") is not False
    ):
        raise ValueError(f"Exp5.5 reference identity mismatch: {path}")
    return payload


def _loader(
    what: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(what, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def _make_loaders(
    what: dict[str, np.ndarray],
    data: base.Data,
    spec: RunSpec,
    config: Config,
    train_shuffle: bool,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, DataLoader]:
    output: dict[str, DataLoader] = {}
    for split in splits:
        labels, lengths = _labels_lengths(data, split)
        output[split] = _loader(
            what[split],
            labels,
            lengths,
            config.batch_size,
            train_shuffle if split == "train" else False,
            base.dseed(spec.seed, EXPERIMENT_ID, "loader", split),
        )
    return output


def sequence_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    time = torch.arange(steps, device=lengths.device).unsqueeze(0)
    return time < lengths.unsqueeze(1)


def _rbf_probabilities(values: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("RBF sigma must be positive")
    logits = -0.5 * ((values.unsqueeze(-1) - centers.view(1, 1, -1)) / sigma) ** 2
    return torch.softmax(logits, dim=-1)


class NonSNNHistoryExperts(nn.Module):
    """Non-spiking WHAT readout with either fixed or learned context-conditioned full experts."""

    def __init__(
        self,
        condition: str,
        n_classes: int = N_CLASSES,
        what_width: int = WHAT_WIDTH,
        gru_width: int = GRU_WIDTH,
        n_states: int = N_STATES,
    ) -> None:
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.condition = condition
        self.n_classes = int(n_classes)
        self.what_width = int(what_width)
        self.gru_width = int(gru_width)
        self.n_states = int(n_states)
        self.class_bias = nn.Parameter(torch.zeros(self.n_classes))
        if condition == SHARED:
            self.shared_weight = nn.Parameter(torch.empty(self.n_classes, self.what_width))
            nn.init.xavier_uniform_(self.shared_weight)
        else:
            self.expert_weight = nn.Parameter(torch.empty(self.n_states, self.n_classes, self.what_width))
            nn.init.xavier_uniform_(self.expert_weight)
            if condition in GRU_CONDITIONS:
                self.gru = nn.GRU(self.what_width, self.gru_width, batch_first=True)
                self.state_head = nn.Linear(self.gru_width, self.n_states)

    def context_probabilities(
        self,
        what: torch.Tensor,
        lengths: torch.Tensor,
        sampling_rate_hz: float,
        train_max_elapsed_seconds: float,
    ) -> torch.Tensor:
        if self.condition == SHARED:
            raise RuntimeError("Shared readout has no latent context probabilities")
        batch, steps, _ = what.shape
        valid = sequence_mask(lengths, steps).to(what.dtype).unsqueeze(-1)
        if self.condition == CLOCK:
            max_elapsed = max(float(train_max_elapsed_seconds), 1.0 / float(sampling_rate_hz))
            centers = torch.linspace(0.0, max_elapsed, self.n_states, device=what.device, dtype=what.dtype)
            sigma = RBF_SIGMA_MULTIPLIER * max_elapsed / max(self.n_states - 1, 1)
            elapsed = torch.arange(steps, device=what.device, dtype=what.dtype) / float(sampling_rate_hz)
            elapsed = elapsed.clamp(max=max_elapsed).view(1, steps).expand(batch, -1)
            q = _rbf_probabilities(elapsed, centers, sigma)
        elif self.condition == ORACLE_PROGRESS:
            centers = torch.linspace(0.0, 1.0, self.n_states, device=what.device, dtype=what.dtype)
            sigma = RBF_SIGMA_MULTIPLIER / max(self.n_states - 1, 1)
            time = torch.arange(steps, device=what.device, dtype=what.dtype).view(1, steps)
            denom = (lengths.to(what.dtype) - 1.0).clamp(min=1.0).unsqueeze(1)
            progress = (time / denom).clamp(0.0, 1.0)
            q = _rbf_probabilities(progress, centers, sigma)
        elif self.condition == RESET_GRU:
            flat = what.reshape(batch * steps, 1, self.what_width)
            h_after, _ = self.gru(flat)
            h_current = h_after[:, 0, :].reshape(batch, steps, self.gru_width)
            q = torch.softmax(self.state_head(h_current), dim=-1)
        elif self.condition == ORDERED_GRU:
            h_after, _ = self.gru(what)
            h_zero = torch.zeros(batch, 1, self.gru_width, device=what.device, dtype=what.dtype)
            h_previous = torch.cat((h_zero, h_after[:, :-1, :]), dim=1)
            q = torch.softmax(self.state_head(h_previous), dim=-1)
        else:
            raise ValueError(self.condition)
        return q * valid

    def evidence_from_q(
        self,
        what: torch.Tensor,
        q: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if self.condition == SHARED:
            raise RuntimeError("Shared readout does not consume q")
        valid = sequence_mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
        bank_evidence = torch.einsum("btd,kcd->btkc", what, self.expert_weight)
        return (q.unsqueeze(-1) * bank_evidence).sum(dim=2) * valid

    def logits_from_q(
        self,
        what: torch.Tensor,
        q: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.evidence_from_q(what, q, lengths).sum(dim=1) + self.class_bias

    def forward(
        self,
        what: torch.Tensor,
        lengths: torch.Tensor,
        sampling_rate_hz: float,
        train_max_elapsed_seconds: float,
        *,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, Trajectory | None]:
        valid = sequence_mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
        q: torch.Tensor | None = None
        if self.condition == SHARED:
            evidence = F.linear(what, self.shared_weight) * valid
        else:
            q = self.context_probabilities(
                what,
                lengths,
                sampling_rate_hz,
                train_max_elapsed_seconds,
            )
            evidence = self.evidence_from_q(what, q, lengths)
        logits = evidence.sum(dim=1) + self.class_bias
        trajectory = Trajectory(q=q, evidence=evidence) if return_trajectory else None
        return logits, trajectory


def parameter_counts(model: NonSNNHistoryExperts) -> dict[str, int]:
    return {
        "trainable_total": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "stored_total": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def _initialize_model(spec: RunSpec, data: base.Data, config: Config) -> NonSNNHistoryExperts:
    _validate_spec(spec)
    # Condition-independent constructor seed keeps expert initialization paired.
    base.seed_all(base.dseed(spec.seed, EXPERIMENT_ID, "paired_constructor"))
    return NonSNNHistoryExperts(
        condition=spec.condition,
        n_classes=len(data.labels),
        what_width=WHAT_WIDTH,
        gru_width=GRU_WIDTH,
        n_states=N_STATES,
    ).to(config.device)


def _evaluate(
    model: NonSNNHistoryExperts,
    loader: DataLoader,
    data: base.Data,
    reference: dict[str, object],
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for what, labels, lengths in loader:
            what = what.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            logits, _ = model(
                what,
                lengths,
                float(data.fs),
                float(reference["train_max_elapsed_seconds"]),
            )
            loss = F.cross_entropy(logits, labels)
            n = len(labels)
            loss_sum += float(loss.item()) * n
            count += n
            true_parts.append(labels.cpu().numpy())
            pred_parts.append(logits.argmax(dim=1).cpu().numpy())
    true = np.concatenate(true_parts)
    pred = np.concatenate(pred_parts)
    return {
        "loss": float(loss_sum / max(count, 1)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro")),
        "n_samples": int(count),
    }


def train_one(spec: RunSpec, data: base.Data, config: Config, force: bool = False) -> Path:
    _validate_spec(spec)
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    reference = load_reference(spec.seed, config)
    what, _ = _load_what_arrays(spec.seed, data, config)
    train_loaders = _make_loaders(what, data, spec, config, True, ("train",))
    eval_loaders = _make_loaders(what, data, spec, config, False, ("train", "val"))
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _initialize_model(spec, data, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    epoch0_train = _evaluate(model, eval_loaders["train"], data, reference, device)
    epoch0_val = _evaluate(model, eval_loaders["val"], data, reference, device)
    best_ba = float(epoch0_val["balanced_accuracy"])
    best_loss = float(epoch0_val["loss"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history = [
        {
            "epoch": 0,
            "train_loss": epoch0_train["loss"],
            "train_balanced_accuracy": epoch0_train["balanced_accuracy"],
            "val_loss": best_loss,
            "val_balanced_accuracy": best_ba,
        }
    ]

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb, lengths in train_loaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(
                xb,
                lengths,
                float(data.fs),
                float(reference["train_max_elapsed_seconds"]),
            )
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            n = len(yb)
            train_loss_sum += float(loss.item()) * n
            train_count += n
            train_true.append(yb.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        val_metrics = _evaluate(model, eval_loaders["val"], data, reference, device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_count, 1),
                "train_balanced_accuracy": float(
                    balanced_accuracy_score(np.concatenate(train_true), np.concatenate(train_pred))
                ),
                "val_loss": val_loss,
                "val_balanced_accuracy": val_ba,
            }
        )
        improved = val_ba > best_ba + 1e-12 or (
            abs(val_ba - best_ba) <= 1e-12 and val_loss < best_loss - 1e-12
        )
        if improved:
            best_ba = val_ba
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "model_seed": base.dseed(spec.seed, EXPERIMENT_ID, "paired_constructor"),
            "state_dict": best_state,
            "result": {
                "best_epoch": best_epoch,
                "best_val_balanced_accuracy": best_ba,
                "best_val_loss": best_loss,
                "stopped_after_epoch": int(history[-1]["epoch"]),
            },
        },
        destination,
    )
    hist_path = history_path(config.results_dir, spec)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hist_path, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[NonSNNHistoryExperts, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("spec") != asdict(spec)
    ):
        raise ValueError(f"Exp5.5 checkpoint identity mismatch: {path}")
    model = _initialize_model(spec, data, config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _collect_trajectory(
    model: NonSNNHistoryExperts,
    loader: DataLoader,
    data: base.Data,
    reference: dict[str, object],
    device: torch.device,
) -> dict[str, np.ndarray]:
    if model.condition == SHARED:
        raise ValueError("Shared condition has no q trajectory")
    what_parts: list[np.ndarray] = []
    q_parts: list[np.ndarray] = []
    evidence_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    length_parts: list[np.ndarray] = []
    with torch.no_grad():
        for what, labels, lengths in loader:
            what_device = what.to(device)
            lengths_device = lengths.to(device)
            _, trajectory = model(
                what_device,
                lengths_device,
                float(data.fs),
                float(reference["train_max_elapsed_seconds"]),
                return_trajectory=True,
            )
            if trajectory is None or trajectory.q is None:
                raise RuntimeError("Expected q trajectory")
            what_parts.append(what.numpy().astype(np.uint8))
            q_parts.append(trajectory.q.cpu().numpy().astype(np.float32))
            evidence_parts.append(trajectory.evidence.cpu().numpy().astype(np.float32))
            label_parts.append(labels.numpy().astype(np.int64))
            length_parts.append(lengths.numpy().astype(np.int64))
    return {
        "what": np.concatenate(what_parts),
        "q": np.concatenate(q_parts),
        "evidence": np.concatenate(evidence_parts),
        "labels": np.concatenate(label_parts),
        "lengths": np.concatenate(length_parts),
    }


def _state_diagnostics(
    q: np.ndarray,
    lengths: np.ndarray,
    sampling_rate_hz: float,
) -> dict[str, object]:
    valid_rows: list[np.ndarray] = []
    total_variation: list[float] = []
    transitions = 0
    transition_seconds = 0.0
    dwell_steps: list[int] = []
    rel_sum = np.zeros((PROFILE_RELATIVE_BINS, q.shape[-1]), dtype=np.float64)
    rel_count = np.zeros(PROFILE_RELATIVE_BINS, dtype=np.int64)
    absolute_bins = max(1, int(math.ceil(q.shape[1] / (PROFILE_ABSOLUTE_SECONDS * sampling_rate_hz))))
    abs_sum = np.zeros((absolute_bins, q.shape[-1]), dtype=np.float64)
    abs_count = np.zeros(absolute_bins, dtype=np.int64)

    for row, length_value in enumerate(lengths):
        length = int(length_value)
        values = np.asarray(q[row, :length], dtype=np.float64)
        if length <= 0:
            continue
        valid_rows.append(values)
        if length > 1:
            total_variation.extend(np.abs(values[1:] - values[:-1]).sum(axis=1).tolist())
            states = values.argmax(axis=1)
            transitions += int(np.count_nonzero(states[1:] != states[:-1]))
            transition_seconds += float((length - 1) / sampling_rate_hz)
            run_length = 1
            for index in range(1, length):
                if states[index] == states[index - 1]:
                    run_length += 1
                else:
                    dwell_steps.append(run_length)
                    run_length = 1
            dwell_steps.append(run_length)
        else:
            dwell_steps.append(1)
        denom = max(length - 1, 1)
        for timestep in range(length):
            relative = timestep / denom
            rbin = min(PROFILE_RELATIVE_BINS - 1, int(relative * PROFILE_RELATIVE_BINS))
            rel_sum[rbin] += values[timestep]
            rel_count[rbin] += 1
            abin = min(
                absolute_bins - 1,
                int((timestep / sampling_rate_hz) / PROFILE_ABSOLUTE_SECONDS),
            )
            abs_sum[abin] += values[timestep]
            abs_count[abin] += 1

    stacked = np.concatenate(valid_rows, axis=0)
    entropy = -(stacked * np.log(np.clip(stacked, 1e-12, None))).sum(axis=1)
    occupancy = stacked.mean(axis=0)
    rel_profile = [
        (rel_sum[index] / max(int(rel_count[index]), 1)).tolist()
        for index in range(PROFILE_RELATIVE_BINS)
    ]
    abs_profile = [
        (abs_sum[index] / max(int(abs_count[index]), 1)).tolist()
        for index in range(absolute_bins)
    ]
    return {
        "occupancy": occupancy.tolist(),
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(np.median(entropy)),
        "total_variation_l1_mean": float(np.mean(total_variation)) if total_variation else 0.0,
        "argmax_transitions_per_second": float(transitions / max(transition_seconds, 1e-12)),
        "mean_dwell_seconds": float(np.mean(dwell_steps) / sampling_rate_hz),
        "median_dwell_seconds": float(np.median(dwell_steps) / sampling_rate_hz),
        "relative_profile": rel_profile,
        "relative_profile_centers": [
            (index + 0.5) / PROFILE_RELATIVE_BINS for index in range(PROFILE_RELATIVE_BINS)
        ],
        "absolute_profile": abs_profile,
        "absolute_profile_centers_seconds": [
            (index + 0.5) * PROFILE_ABSOLUTE_SECONDS for index in range(absolute_bins)
        ],
    }


def _q_features(trajectory: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    q = trajectory["q"]
    lengths = trajectory["lengths"]
    mean_features = np.stack(
        [q[row, : int(length)].mean(axis=0) for row, length in enumerate(lengths)],
        axis=0,
    )
    final_features = np.stack(
        [q[row, max(int(length) - 1, 0)] for row, length in enumerate(lengths)],
        axis=0,
    )
    return {"mean_q": mean_features, "final_q": final_features}


def _q_only_probes(
    trajectories: dict[str, dict[str, np.ndarray]],
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    features = {split: _q_features(trajectory) for split, trajectory in trajectories.items()}
    for feature_name in ("mean_q", "final_q"):
        scaler = StandardScaler()
        train_x = scaler.fit_transform(features["train"][feature_name])
        classifier = LogisticRegression(
            max_iter=Q_PROBE_MAX_ITER,
            solver="lbfgs",
            random_state=base.dseed(seed, EXPERIMENT_ID, "q_probe", feature_name),
        )
        classifier.fit(train_x, trajectories["train"]["labels"])
        for split in ("train", "val", "test"):
            x = train_x if split == "train" else scaler.transform(features[split][feature_name])
            logits = classifier.decision_function(x)
            metric = _metrics_from_logits(trajectories[split]["labels"], logits)
            rows.append({"feature": feature_name, "split": split, **metric})
    return rows


def _shuffled_q_metrics(
    model: NonSNNHistoryExperts,
    test_trajectory: dict[str, np.ndarray],
    seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    what = test_trajectory["what"]
    q = test_trajectory["q"]
    labels = test_trajectory["labels"]
    lengths = test_trajectory["lengths"]
    rows: list[dict[str, object]] = []
    what_tensor = torch.tensor(what, dtype=torch.float32, device=device)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    for replicate in range(SHUFFLE_REPLICATES):
        rng = np.random.default_rng(base.dseed(seed, EXPERIMENT_ID, "q_shuffle", replicate))
        shuffled = q.copy()
        for row, length_value in enumerate(lengths):
            length = int(length_value)
            shuffled[row, :length] = q[row, rng.permutation(length)]
        q_tensor = torch.tensor(shuffled, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model.logits_from_q(what_tensor, q_tensor, lengths_tensor).cpu().numpy()
        rows.append({"replicate": replicate, **_metrics_from_logits(labels, logits)})
    return rows


def _same_what_pairs(
    trajectory: dict[str, np.ndarray],
    seed: int,
) -> pd.DataFrame:
    what = trajectory["what"]
    q = trajectory["q"]
    evidence = trajectory["evidence"]
    labels = trajectory["labels"]
    lengths = trajectory["lengths"]
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]] = []
    for sample_index, length_value in enumerate(lengths):
        length = int(length_value)
        for timestep in range(length):
            vector = what[sample_index, timestep].astype(np.float32)
            if float(np.linalg.norm(vector)) <= 0.0:
                continue
            rows.append(
                (
                    vector,
                    q[sample_index, timestep].astype(np.float32),
                    evidence[sample_index, timestep].astype(np.float32),
                    sample_index,
                    timestep,
                    int(labels[sample_index]),
                )
            )
    columns = [
        "seed",
        "sample_a",
        "timestep_a",
        "label_a",
        "sample_b",
        "timestep_b",
        "label_b",
        "what_cosine_similarity",
        "q_l1_distance",
        "evidence_l1_distance",
        "top_support_a",
        "top_support_b",
        "different_top_support",
        "high_similarity",
    ]
    if len(rows) < 2:
        return pd.DataFrame(columns=columns)
    rng = np.random.default_rng(base.dseed(seed, EXPERIMENT_ID, "same_what_sample"))
    if len(rows) > SAME_WHAT_MAX_POINTS:
        selection = np.sort(rng.choice(len(rows), size=SAME_WHAT_MAX_POINTS, replace=False))
        rows = [rows[index] for index in selection]
    vectors = np.stack([row[0] for row in rows])
    neighbors = NearestNeighbors(
        n_neighbors=min(SAME_WHAT_NEIGHBORS, len(rows)),
        metric="cosine",
        algorithm="brute",
    ).fit(vectors)
    distances, indices = neighbors.kneighbors(vectors)
    candidates: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for index, (distance_row, neighbor_row) in enumerate(zip(distances, indices)):
        _, q_a, evidence_a, sample_a, timestep_a, label_a = rows[index]
        for distance, neighbor_index in zip(distance_row[1:], neighbor_row[1:]):
            _, q_b, evidence_b, sample_b, timestep_b, label_b = rows[int(neighbor_index)]
            if sample_a == sample_b or label_a == label_b:
                continue
            pair_key = tuple(sorted((index, int(neighbor_index))))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            similarity = float(1.0 - distance)
            candidates.append(
                {
                    "seed": seed,
                    "sample_a": sample_a,
                    "timestep_a": timestep_a,
                    "label_a": label_a,
                    "sample_b": sample_b,
                    "timestep_b": timestep_b,
                    "label_b": label_b,
                    "what_cosine_similarity": similarity,
                    "q_l1_distance": float(np.abs(q_a - q_b).sum()),
                    "evidence_l1_distance": float(np.abs(evidence_a - evidence_b).sum()),
                    "top_support_a": int(np.argmax(evidence_a)),
                    "top_support_b": int(np.argmax(evidence_b)),
                    "different_top_support": bool(np.argmax(evidence_a) != np.argmax(evidence_b)),
                    "high_similarity": bool(similarity >= SAME_WHAT_HIGH_SIMILARITY),
                }
            )
            break
    candidates.sort(
        key=lambda row: (
            float(row["what_cosine_similarity"]),
            float(row["q_l1_distance"]),
            float(row["evidence_l1_distance"]),
        ),
        reverse=True,
    )
    return pd.DataFrame(candidates[:SAME_WHAT_TOP_PAIRS], columns=columns)


def evaluate_run_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    reference = load_reference(spec.seed, config)
    what, source_meta = _load_what_arrays(spec.seed, data, config)
    loaders = _make_loaders(what, data, spec, config, False)
    model, checkpoint = load_model(spec, data, config)
    device = torch.device(config.device)
    metrics = {
        split: _evaluate(model, loaders[split], data, reference, device)
        for split in ("train", "val", "test")
    }

    state_diagnostics: dict[str, object] | None = None
    q_only: list[dict[str, object]] = []
    shuffled_q: list[dict[str, object]] = []
    saved_trajectory: str | None = None
    same_what_csv: str | None = None
    if spec.condition != SHARED:
        test_trajectory = _collect_trajectory(model, loaders["test"], data, reference, device)
        state_diagnostics = _state_diagnostics(
            test_trajectory["q"],
            test_trajectory["lengths"],
            float(data.fs),
        )
        if spec.condition in GRU_CONDITIONS:
            trajectories = {
                split: _collect_trajectory(model, loaders[split], data, reference, device)
                for split in ("train", "val", "test")
            }
            q_only = _q_only_probes(trajectories, spec.seed)
            if spec.condition == ORDERED_GRU:
                test_trajectory = trajectories["test"]
        if spec.condition == ORDERED_GRU:
            shuffled_q = _shuffled_q_metrics(model, test_trajectory, spec.seed, device)
            path = trajectory_path(config.results_dir, spec)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **test_trajectory)
            saved_trajectory = str(path.relative_to(config.repo_root))
            pair_path = same_what_path(config.results_dir, spec)
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            _same_what_pairs(test_trajectory, spec.seed).to_csv(pair_path, index=False)
            same_what_csv = str(pair_path.relative_to(config.repo_root))

    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "seed": spec.seed,
        "model_seed": checkpoint["model_seed"],
        "best_epoch": checkpoint["result"]["best_epoch"],
        "stopped_after_epoch": checkpoint["result"]["stopped_after_epoch"],
        "parameter_counts": parameter_counts(model),
        "train": metrics["train"],
        "val": metrics["val"],
        "test": metrics["test"],
        "state_diagnostics": state_diagnostics,
        "q_only_probes": q_only,
        "shuffled_q_test": shuffled_q,
        "test_trajectory": saved_trajectory,
        "same_what_pairs": same_what_csv,
        "source": {
            "source_experiment_id": exp54.EXPERIMENT_ID,
            "source_protocol_version": exp54.PROTOCOL_VERSION,
            "source_cache_seed": source_meta.get("seed", spec.seed),
            "what_definition": "frozen Local-SNN L2 spikes",
            "sampling_rate_hz": float(data.fs),
            "split_seed": base.SPLIT_SEED,
        },
        "selection_policy": "best epoch selected by validation balanced accuracy, tie-broken by validation loss",
        "test_used_for_checkpoint_selection": False,
    }
    _save_json(destination, payload)
    return payload


def run_one(spec: RunSpec, data: base.Data, config: Config, force: bool = False) -> dict[str, object]:
    train_one(spec, data, config, force=force)
    return evaluate_run_one(spec, data, config, force=force)


def _sem(values: np.ndarray) -> float:
    return 0.0 if len(values) <= 1 else float(values.std(ddof=1) / math.sqrt(len(values)))


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    references: list[dict[str, object]] = []
    payloads: dict[tuple[str, int], dict[str, object]] = {}
    for seed in SEEDS:
        ref_path = reference_path(root, seed)
        if not ref_path.exists():
            raise FileNotFoundError(f"Finalizer will not regenerate missing reference: {ref_path}")
        reference = json.loads(ref_path.read_text(encoding="utf-8"))
        fixed = reference["fixed250"]["metrics"]
        references.append(
            {
                "seed": seed,
                "split_seed": reference["split_seed"],
                "fixed250_val_balanced_accuracy": fixed["val"]["balanced_accuracy"],
                "fixed250_test_balanced_accuracy": fixed["test"]["balanced_accuracy"],
                "fixed250_test_accuracy": fixed["test"]["accuracy"],
                "fixed250_test_macro_f1": fixed["test"]["macro_f1"],
                "streaming_max_abs_delta": reference["fixed250"]["equivalence"]["test_max_abs_offline_vs_streaming"],
            }
        )
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Finalizer will not retrain missing run: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("test_used_for_checkpoint_selection") is not False:
            raise ValueError(f"Invalid Exp5.5 evaluation artifact: {path}")
        payloads[(spec.condition, spec.seed)] = payload

    run_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    q_probe_rows: list[dict[str, object]] = []
    shuffle_rows: list[dict[str, object]] = []
    same_what_frames: list[pd.DataFrame] = []
    for spec in run_specs():
        payload = payloads[(spec.condition, spec.seed)]
        run_rows.append(
            {
                "condition": spec.condition,
                "seed": spec.seed,
                "model_seed": payload["model_seed"],
                "best_epoch": payload["best_epoch"],
                "stopped_after_epoch": payload["stopped_after_epoch"],
                "trainable_parameter_count": payload["parameter_counts"]["trainable_total"],
                "val_balanced_accuracy": payload["val"]["balanced_accuracy"],
                "val_accuracy": payload["val"]["accuracy"],
                "val_macro_f1": payload["val"]["macro_f1"],
                "val_loss": payload["val"]["loss"],
                "test_balanced_accuracy": payload["test"]["balanced_accuracy"],
                "test_accuracy": payload["test"]["accuracy"],
                "test_macro_f1": payload["test"]["macro_f1"],
                "test_loss": payload["test"]["loss"],
            }
        )
        diagnostic = payload.get("state_diagnostics")
        if diagnostic is not None:
            state_rows.extend(
                [
                    {
                        "condition": spec.condition,
                        "seed": spec.seed,
                        "metric": metric,
                        "state": None,
                        "value": diagnostic[metric],
                    }
                    for metric in (
                        "entropy_mean",
                        "entropy_median",
                        "total_variation_l1_mean",
                        "argmax_transitions_per_second",
                        "mean_dwell_seconds",
                        "median_dwell_seconds",
                    )
                ]
            )
            for state, value in enumerate(diagnostic["occupancy"]):
                state_rows.append(
                    {
                        "condition": spec.condition,
                        "seed": spec.seed,
                        "metric": "occupancy",
                        "state": state,
                        "value": value,
                    }
                )
            for center, probabilities in zip(
                diagnostic["relative_profile_centers"], diagnostic["relative_profile"]
            ):
                for state, value in enumerate(probabilities):
                    profile_rows.append(
                        {
                            "condition": spec.condition,
                            "seed": spec.seed,
                            "axis": "relative_progress",
                            "position": center,
                            "state": state,
                            "probability": value,
                        }
                    )
            for center, probabilities in zip(
                diagnostic["absolute_profile_centers_seconds"], diagnostic["absolute_profile"]
            ):
                for state, value in enumerate(probabilities):
                    profile_rows.append(
                        {
                            "condition": spec.condition,
                            "seed": spec.seed,
                            "axis": "elapsed_seconds",
                            "position": center,
                            "state": state,
                            "probability": value,
                        }
                    )
        for row in payload.get("q_only_probes", []):
            q_probe_rows.append({"condition": spec.condition, "seed": spec.seed, **row})
        for row in payload.get("shuffled_q_test", []):
            shuffle_rows.append({"condition": spec.condition, "seed": spec.seed, **row})
        pair_path = same_what_path(root, spec)
        if spec.condition == ORDERED_GRU:
            if not pair_path.exists():
                raise FileNotFoundError(f"Missing ordered same-WHAT diagnostic: {pair_path}")
            frame = pd.read_csv(pair_path)
            if not frame.empty:
                same_what_frames.append(frame)

    runs = pd.DataFrame(run_rows)
    summaries: list[dict[str, object]] = []
    for condition in CONDITIONS:
        group = runs[runs["condition"] == condition].sort_values("seed")
        test_ba = group["test_balanced_accuracy"].to_numpy(dtype=float)
        val_ba = group["val_balanced_accuracy"].to_numpy(dtype=float)
        summaries.append(
            {
                "condition": condition,
                "n_runs": int(len(group)),
                "trainable_parameter_count": int(group["trainable_parameter_count"].iloc[0]),
                "mean_val_balanced_accuracy": float(val_ba.mean()),
                "sd_val_balanced_accuracy": float(val_ba.std(ddof=1)),
                "mean_test_balanced_accuracy": float(test_ba.mean()),
                "sd_test_balanced_accuracy": float(test_ba.std(ddof=1)),
                "sem_test_balanced_accuracy": _sem(test_ba),
                "mean_test_accuracy": float(group["test_accuracy"].mean()),
                "mean_test_macro_f1": float(group["test_macro_f1"].mean()),
            }
        )

    comparisons = (
        (ORDERED_GRU, SHARED, "ordered_minus_shared"),
        (ORDERED_GRU, RESET_GRU, "ordered_minus_reset"),
        (ORDERED_GRU, CLOCK, "ordered_minus_clock"),
        (ORDERED_GRU, ORACLE_PROGRESS, "ordered_minus_oracle_progress"),
        (RESET_GRU, SHARED, "reset_minus_shared"),
        (CLOCK, SHARED, "clock_minus_shared"),
    )
    paired_rows: list[dict[str, object]] = []
    for left, right, name in comparisons:
        for seed in SEEDS:
            left_ba = float(payloads[(left, seed)]["test"]["balanced_accuracy"])
            right_ba = float(payloads[(right, seed)]["test"]["balanced_accuracy"])
            paired_rows.append(
                {
                    "seed": seed,
                    "comparison": name,
                    "left_condition": left,
                    "right_condition": right,
                    "delta_test_balanced_accuracy": left_ba - right_ba,
                }
            )
    for seed in SEEDS:
        ordered_ba = float(payloads[(ORDERED_GRU, seed)]["test"]["balanced_accuracy"])
        shuffle = payloads[(ORDERED_GRU, seed)]["shuffled_q_test"]
        shuffle_mean = float(np.mean([row["balanced_accuracy"] for row in shuffle]))
        paired_rows.append(
            {
                "seed": seed,
                "comparison": "ordered_minus_shuffled_q",
                "left_condition": ORDERED_GRU,
                "right_condition": "ordered_gru_shuffled_q",
                "delta_test_balanced_accuracy": ordered_ba - shuffle_mean,
            }
        )

    outputs = {
        "references": root / "references.csv",
        "runs": root / "runs.csv",
        "summary": root / "summary.csv",
        "paired_deltas": root / "paired_deltas.csv",
        "state_diagnostics": root / "state_diagnostics.csv",
        "state_profiles": root / "state_profiles.csv",
        "q_only_probes": root / "q_only_probes.csv",
        "shuffled_q": root / "shuffled_q.csv",
        "same_what_pairs": root / "same_what_pairs.csv",
        "manifest": root / "manifest.json",
    }
    pd.DataFrame(references).to_csv(outputs["references"], index=False)
    runs.to_csv(outputs["runs"], index=False)
    pd.DataFrame(summaries).to_csv(outputs["summary"], index=False)
    pd.DataFrame(paired_rows).to_csv(outputs["paired_deltas"], index=False)
    pd.DataFrame(state_rows).to_csv(outputs["state_diagnostics"], index=False)
    pd.DataFrame(profile_rows).to_csv(outputs["state_profiles"], index=False)
    pd.DataFrame(q_probe_rows).to_csv(outputs["q_only_probes"], index=False)
    pd.DataFrame(shuffle_rows).to_csv(outputs["shuffled_q"], index=False)
    same_what = pd.concat(same_what_frames, ignore_index=True) if same_what_frames else pd.DataFrame()
    same_what.to_csv(outputs["same_what_pairs"], index=False)
    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seeds": list(SEEDS),
            "conditions": list(CONDITIONS),
            "gru_width": GRU_WIDTH,
            "n_states": N_STATES,
            "loss": "whole-gesture cross entropy only",
            "ordered_state_definition": "q_t = softmax(G h_{t-1}); h_{t-1} contains WHAT_1..WHAT_{t-1}",
            "reset_state_definition": "q_t = softmax(G GRU(m_t, 0)); no cross-timestep history",
            "clock_definition": "8 smooth RBF probabilities over causal absolute elapsed time, clipped to train max elapsed",
            "oracle_progress_definition": "8 smooth RBF probabilities over t/(T-1); diagnostic only and non-deployable",
            "expert_definition": "K independent full 12x128 matrices with soft gating; no per-timestep expert bias",
            "accumulator": "valid-timestep evidence sum plus one final 12D class bias",
            "selection_policy": "all five conditions are predeclared; best epoch is validation-selected within each run; test does not select checkpoints or conditions",
            "multi_cpu_policy": "5 reference tasks -> 25 independent train/evaluate tasks -> one artifact-only finalizer",
            "notebook_policy": "analysis-only; reads finalized CSV/JSON artifacts and never trains or regenerates missing runs",
        },
    )
    return outputs


def _config_from_args(args: argparse.Namespace) -> Config:
    root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    return Config(
        repo_root=root,
        results_dir=results_dir(root),
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        threads=args.threads,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 5.5 non-SNN history-conditioned WHAT experts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repo-root", default=None)
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=1)
        command.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        command.add_argument("--epochs", type=int, default=EPOCHS)
        command.add_argument("--patience", type=int, default=PATIENCE)
        command.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare-reference")
    add_common(prepare)
    prepare.add_argument("--array-task-id", type=int, required=True)

    run = subparsers.add_parser("run-one")
    add_common(run)
    run.add_argument("--array-task-id", type=int, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--repo-root", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "finalize":
        root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
        for name, path in finalize_experiment(root).items():
            print(f"{name}: {path}")
        return

    config = _config_from_args(args)
    data = base.prepare_data(config.repo_root)
    task = int(args.array_task_id)
    if args.command == "prepare-reference":
        if task < 0 or task >= len(SEEDS):
            raise IndexError(f"prepare-reference array task {task} outside 0..{len(SEEDS) - 1}")
        print(prepare_reference_seed(SEEDS[task], data, config, force=args.force))
        return
    if args.command == "run-one":
        specs = run_specs()
        if task < 0 or task >= len(specs):
            raise IndexError(f"run-one array task {task} outside 0..{len(specs) - 1}")
        print(json.dumps(run_one(specs[task], data, config, force=args.force), indent=2))
        return
    raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
