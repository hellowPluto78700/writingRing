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

from scripts import experiment_5_5_non_snn_history_experts as exp55
from scripts import experiment_5_5_1_regularized_semantic_when as exp551


base = exp55.base
exp54 = exp55.exp54

EXPERIMENT_ID = "experiment_5_5_2_progress_anchored_history_residual"
PROTOCOL_VERSION = "progress_anchored_history_residual_v1"
SEEDS = exp55.SEEDS
WHAT_WIDTH = exp55.WHAT_WIDTH
N_CLASSES = exp55.N_CLASSES
GRU_WIDTH = exp55.GRU_WIDTH
N_STATES = exp55.N_STATES
EPOCHS = exp55.EPOCHS
PATIENCE = exp55.PATIENCE
BATCH_SIZE = exp55.BATCH_SIZE
LR = exp55.LR
WEIGHT_DECAY = exp55.WEIGHT_DECAY
GRAD_CLIP = exp55.GRAD_CLIP
ALPHAS = (0.25, 0.5, 1.0)
COORD_WIDTH = 64
SHUFFLE_REPLICATES = 5
SAME_WHAT_MAX_POINTS = 4000
SAME_WHAT_NEIGHBORS = 16
SAME_WHAT_TOP_PAIRS = 100
SAME_WHAT_SIMILARITY = 0.95
SAME_WHAT_NORM_RATIO_MAX = 1.10
ORACLE_COORD_DELTA = 0.05
CLOCK_COORD_DELTA_SECONDS = 0.125
RESIDUAL_PROBE_MAX_ITER = 3000
EPS = 1e-12

CLOCK = "clock"
ORACLE = "oracle_progress"
ANCHORS = (CLOCK, ORACLE)
COORD = "coordinate"
RESET = "reset"
LAG1 = "lag1"
ORDERED = "ordered"
RESIDUAL_MODES = (COORD, RESET, LAG1, ORDERED)

SAME_WHAT_COLUMNS = (
    "anchor", "seed", "sample_a", "timestep_a", "label_a", "sample_b", "timestep_b", "label_b",
    "what_cosine_similarity", "what_norm_ratio", "coordinate_a", "coordinate_b", "coordinate_abs_delta",
    "anchor_q_l1_distance", "delta_z_l1_distance", "q_l1_distance", "delta_w_pair_fro",
    "top_support_a", "top_support_b", "different_top_support", "anchor_margin_a", "residual_margin_a",
    "margin_delta_a", "anchor_margin_b", "residual_margin_b", "margin_delta_b",
)


@dataclass(frozen=True)
class ScreenSpec:
    residual_mode: str
    alpha: float
    seed: int

    @property
    def key(self) -> str:
        token = str(self.alpha).replace(".", "p")
        return f"{self.residual_mode}__a{token}__seed{self.seed}"


@dataclass(frozen=True)
class FinalSpec:
    anchor: str
    residual_mode: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.anchor}__{self.residual_mode}__seed{self.seed}"


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
class ResidualTrajectory:
    anchor_logits: torch.Tensor
    anchor_q: torch.Tensor
    delta_z: torch.Tensor
    q: torch.Tensor
    anchor_evidence: torch.Tensor
    evidence: torch.Tensor


def find_repo_root(start: Path | None = None) -> Path:
    return exp55.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_results_dir(repo_root: Path) -> Path:
    return exp55.results_dir(repo_root)


def source_551_results_dir(repo_root: Path) -> Path:
    return exp551.results_dir(repo_root)


def source_reference_path(root: Path, seed: int) -> Path:
    return root / "source_references" / f"seed{seed}.json"


def screen_checkpoint_path(root: Path, spec: ScreenSpec) -> Path:
    return root / "screen_checkpoints" / f"{spec.key}.pt"


def screen_history_path(root: Path, spec: ScreenSpec) -> Path:
    return root / "screen_histories" / f"{spec.key}.csv"


def screen_evaluation_path(root: Path, spec: ScreenSpec) -> Path:
    return root / "screen_evaluations" / f"{spec.key}.json"


def selection_path(root: Path) -> Path:
    return root / "selection.json"


def oracle_checkpoint_path(root: Path, spec: FinalSpec) -> Path:
    return root / "final_checkpoints" / f"{spec.key}.pt"


def oracle_history_path(root: Path, spec: FinalSpec) -> Path:
    return root / "final_histories" / f"{spec.key}.csv"


def final_evaluation_path(root: Path, spec: FinalSpec) -> Path:
    return root / "final_evaluations" / f"{spec.key}.json"


def trajectory_path(root: Path, spec: FinalSpec) -> Path:
    return root / "trajectories" / f"{spec.key}__test.npz"


def same_what_path(root: Path, spec: FinalSpec) -> Path:
    return root / "same_what" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def screen_specs() -> list[ScreenSpec]:
    return [ScreenSpec(mode, alpha, seed) for mode in RESIDUAL_MODES for alpha in ALPHAS for seed in SEEDS]


def final_mode_specs(anchor: str) -> list[FinalSpec]:
    if anchor not in ANCHORS:
        raise ValueError(anchor)
    return [FinalSpec(anchor, mode, seed) for mode in RESIDUAL_MODES for seed in SEEDS]


def _source_config(config: Config) -> exp55.Config:
    return exp55.Config(
        repo_root=config.repo_root,
        results_dir=source_results_dir(config.repo_root),
        device=config.device,
        epochs=config.epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _labels_lengths(data: Any, split: str) -> tuple[np.ndarray, np.ndarray]:
    return exp55._labels_lengths(data, split)


def _load_what_splits(
    seed: int,
    data: Any,
    config: Config,
    splits: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays, meta = exp54.load_fusion_cache(seed, data, exp55._exp54_config(_source_config(config)))
    output: dict[str, np.ndarray] = {}
    for split in splits:
        values = np.asarray(arrays[f"what_{split}"], dtype=np.float32)
        labels, _ = _labels_lengths(data, split)
        expected = (len(labels), data.T, WHAT_WIDTH)
        if values.shape != expected:
            raise ValueError(f"Exp5.5.2 WHAT cache shape mismatch for {split}: {values.shape} != {expected}")
        output[split] = values
    return output, meta


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
    data: Any,
    seed: int,
    config: Config,
    splits: tuple[str, ...],
    train_shuffle: bool,
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
            base.dseed(seed, EXPERIMENT_ID, "loader", split),
        )
    return output


def sequence_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return exp55.sequence_mask(lengths, steps)


def _source_condition(anchor: str) -> str:
    if anchor == CLOCK:
        return exp55.CLOCK
    if anchor == ORACLE:
        return exp55.ORACLE_PROGRESS
    raise ValueError(anchor)


def _source_eval(repo_root: Path, anchor: str, seed: int) -> dict[str, object]:
    spec = exp55.RunSpec(_source_condition(anchor), seed)
    path = exp55.evaluation_path(source_results_dir(repo_root), spec)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _source_anchor_model(
    anchor: str,
    seed: int,
    data: Any,
    config: Config,
) -> tuple[exp55.NonSNNHistoryExperts, dict[str, object]]:
    return exp55.load_model(
        exp55.RunSpec(_source_condition(anchor), seed),
        data,
        _source_config(config),
    )


class ProgressAnchoredResidual(nn.Module):
    """Frozen Exp5.5 anchor experts plus a bounded residual routing correction."""

    def __init__(
        self,
        *,
        anchor: str,
        residual_mode: str,
        alpha: float,
        expert_weight: torch.Tensor,
        class_bias: torch.Tensor,
        n_states: int = N_STATES,
        what_width: int = WHAT_WIDTH,
        gru_width: int = GRU_WIDTH,
    ) -> None:
        super().__init__()
        if anchor not in ANCHORS:
            raise ValueError(anchor)
        if residual_mode not in RESIDUAL_MODES:
            raise ValueError(residual_mode)
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if tuple(expert_weight.shape) != (n_states, class_bias.numel(), what_width):
            raise ValueError(f"Unexpected expert bank shape: {tuple(expert_weight.shape)}")
        self.anchor = anchor
        self.residual_mode = residual_mode
        self.alpha = float(alpha)
        self.n_states = int(n_states)
        self.what_width = int(what_width)
        self.gru_width = int(gru_width)

        self.expert_weight = nn.Parameter(expert_weight.detach().clone(), requires_grad=False)
        self.class_bias = nn.Parameter(class_bias.detach().clone(), requires_grad=False)
        if residual_mode == COORD:
            self.coord_hidden = nn.Linear(1, COORD_WIDTH)
            self.residual_head = nn.Linear(COORD_WIDTH, self.n_states)
        else:
            self.gru = nn.GRU(self.what_width, self.gru_width, batch_first=True)
            self.residual_head = nn.Linear(self.gru_width, self.n_states)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def anchor_coordinate(
        self,
        what: torch.Tensor,
        lengths: torch.Tensor,
        sampling_rate_hz: float,
        train_max_elapsed_seconds: float,
    ) -> torch.Tensor:
        batch, steps, _ = what.shape
        if self.anchor == CLOCK:
            max_elapsed = max(float(train_max_elapsed_seconds), 1.0 / float(sampling_rate_hz))
            elapsed = torch.arange(steps, device=what.device, dtype=what.dtype) / float(sampling_rate_hz)
            return elapsed.clamp(max=max_elapsed).view(1, steps, 1).expand(batch, -1, -1)
        time = torch.arange(steps, device=what.device, dtype=what.dtype).view(1, steps)
        denom = (lengths.to(what.dtype) - 1.0).clamp(min=1.0).unsqueeze(1)
        return (time / denom).clamp(0.0, 1.0).unsqueeze(-1)

    def anchor_routing_logits(
        self,
        what: torch.Tensor,
        lengths: torch.Tensor,
        sampling_rate_hz: float,
        train_max_elapsed_seconds: float,
    ) -> torch.Tensor:
        coordinate = self.anchor_coordinate(what, lengths, sampling_rate_hz, train_max_elapsed_seconds)
        if self.anchor == CLOCK:
            max_elapsed = max(float(train_max_elapsed_seconds), 1.0 / float(sampling_rate_hz))
            centers = torch.linspace(
                0.0,
                max_elapsed,
                self.n_states,
                device=what.device,
                dtype=what.dtype,
            )
            sigma = exp55.RBF_SIGMA_MULTIPLIER * max_elapsed / max(self.n_states - 1, 1)
        else:
            centers = torch.linspace(0.0, 1.0, self.n_states, device=what.device, dtype=what.dtype)
            sigma = exp55.RBF_SIGMA_MULTIPLIER / max(self.n_states - 1, 1)
        return -0.5 * ((coordinate - centers.view(1, 1, -1)) / sigma) ** 2

    def residual_features(self, what: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = what.shape
        if self.residual_mode == RESET:
            flat = what.reshape(batch * steps, 1, self.what_width)
            h_flat, _ = self.gru(flat)
            return h_flat[:, 0, :].reshape(batch, steps, self.gru_width)
        if self.residual_mode == LAG1:
            shifted = torch.zeros_like(what)
            if steps > 1:
                shifted[:, 1:, :] = what[:, :-1, :]
            flat = shifted.reshape(batch * steps, 1, self.what_width)
            h_flat, _ = self.gru(flat)
            h = h_flat[:, 0, :].reshape(batch, steps, self.gru_width)
            h = h.clone()
            h[:, 0, :] = 0.0
            return h
        if self.residual_mode == ORDERED:
            h_current, _ = self.gru(what)
            h_zero = torch.zeros(batch, 1, self.gru_width, device=what.device, dtype=what.dtype)
            return torch.cat((h_zero, h_current[:, :-1, :]), dim=1)
        raise RuntimeError("coordinate mode does not use GRU residual_features")

    def residual_logits(
        self,
        what: torch.Tensor,
        lengths: torch.Tensor,
        sampling_rate_hz: float,
        train_max_elapsed_seconds: float,
    ) -> torch.Tensor:
        if self.residual_mode == COORD:
            coordinate = self.anchor_coordinate(what, lengths, sampling_rate_hz, train_max_elapsed_seconds)
            raw = self.residual_head(F.gelu(self.coord_hidden(coordinate)))
        else:
            raw = self.residual_head(self.residual_features(what))
        delta = self.alpha * torch.tanh(raw)
        valid = sequence_mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
        delta = delta * valid
        if self.residual_mode in (LAG1, ORDERED) and what.shape[1] > 0:
            delta = delta.clone()
            delta[:, 0, :] = 0.0
        return delta

    def evidence_from_q(
        self,
        what: torch.Tensor,
        q: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
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
    ) -> tuple[torch.Tensor, ResidualTrajectory | None]:
        valid = sequence_mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
        anchor_logits = self.anchor_routing_logits(
            what, lengths, sampling_rate_hz, train_max_elapsed_seconds
        )
        anchor_q = torch.softmax(anchor_logits, dim=-1) * valid
        delta_z = self.residual_logits(what, lengths, sampling_rate_hz, train_max_elapsed_seconds)
        q = torch.softmax(anchor_logits + delta_z, dim=-1) * valid
        anchor_evidence = self.evidence_from_q(what, anchor_q, lengths)
        evidence = self.evidence_from_q(what, q, lengths)
        logits = evidence.sum(dim=1) + self.class_bias
        trajectory = None
        if return_trajectory:
            trajectory = ResidualTrajectory(
                anchor_logits=anchor_logits,
                anchor_q=anchor_q,
                delta_z=delta_z,
                q=q,
                anchor_evidence=anchor_evidence,
                evidence=evidence,
            )
        return logits, trajectory


def _initialize_model(
    *,
    anchor: str,
    residual_mode: str,
    alpha: float,
    seed: int,
    data: Any,
    config: Config,
) -> ProgressAnchoredResidual:
    source_model, _ = _source_anchor_model(anchor, seed, data, config)
    model_seed = base.dseed(seed, EXPERIMENT_ID, anchor, residual_mode, alpha, "residual_constructor")
    base.seed_all(model_seed)
    return ProgressAnchoredResidual(
        anchor=anchor,
        residual_mode=residual_mode,
        alpha=alpha,
        expert_weight=source_model.expert_weight.detach().cpu(),
        class_bias=source_model.class_bias.detach().cpu(),
        n_states=N_STATES,
        what_width=WHAT_WIDTH,
        gru_width=GRU_WIDTH,
    ).to(config.device)


def parameter_counts(model: ProgressAnchoredResidual) -> dict[str, int]:
    return {
        "trainable_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "stored_total": int(sum(p.numel() for p in model.parameters())),
        "frozen_expert_bank": int(model.expert_weight.numel()),
        "frozen_class_bias": int(model.class_bias.numel()),
    }


def _evaluate_classification(
    model: ProgressAnchoredResidual,
    loader: DataLoader,
    data: Any,
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


def _train_model(
    *,
    anchor: str,
    residual_mode: str,
    alpha: float,
    seed: int,
    data: Any,
    config: Config,
    checkpoint: Path,
    history: Path,
    force: bool,
) -> Path:
    if checkpoint.exists() and not force:
        return checkpoint
    load_source_reference(seed, config)
    what, _ = _load_what_splits(seed, data, config, ("train", "val"))
    train_loader = _make_loaders(what, data, seed, config, ("train",), True)["train"]
    eval_loaders = _make_loaders(what, data, seed, config, ("train", "val"), False)
    reference = exp55.load_reference(seed, _source_config(config))
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _initialize_model(
        anchor=anchor,
        residual_mode=residual_mode,
        alpha=alpha,
        seed=seed,
        data=data,
        config=config,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("Residual model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    train0 = _evaluate_classification(model, eval_loaders["train"], data, reference, device)
    val0 = _evaluate_classification(model, eval_loaders["val"], data, reference, device)
    best_ba = float(val0["balanced_accuracy"])
    best_loss = float(val0["loss"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    rows: list[dict[str, object]] = [
        {
            "epoch": 0,
            "train_loss": train0["loss"],
            "train_balanced_accuracy": train0["balanced_accuracy"],
            "val_loss": val0["loss"],
            "val_balanced_accuracy": val0["balanced_accuracy"],
        }
    ]

    for epoch in range(1, config.epochs + 1):
        model.train()
        true_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        loss_sum = 0.0
        count = 0
        for xb, yb, lengths in train_loader:
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
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            n = len(yb)
            count += n
            loss_sum += float(loss.detach().item()) * n
            true_parts.append(yb.detach().cpu().numpy())
            pred_parts.append(logits.detach().argmax(dim=1).cpu().numpy())

        val = _evaluate_classification(model, eval_loaders["val"], data, reference, device)
        val_ba = float(val["balanced_accuracy"])
        val_loss = float(val["loss"])
        rows.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(count, 1),
                "train_balanced_accuracy": float(
                    balanced_accuracy_score(np.concatenate(true_parts), np.concatenate(pred_parts))
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
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    source_spec = exp55.RunSpec(_source_condition(anchor), seed)
    source_checkpoint = exp55.checkpoint_path(source_results_dir(config.repo_root), source_spec)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "anchor": anchor,
            "residual_mode": residual_mode,
            "alpha": float(alpha),
            "seed": seed,
            "model_seed": base.dseed(seed, EXPERIMENT_ID, anchor, residual_mode, alpha, "residual_constructor"),
            "source_checkpoint": str(source_checkpoint.relative_to(config.repo_root)),
            "state_dict": best_state,
            "result": {
                "best_epoch": best_epoch,
                "best_val_balanced_accuracy": best_ba,
                "best_val_loss": best_loss,
                "stopped_after_epoch": int(rows[-1]["epoch"]),
            },
        },
        checkpoint,
    )
    history.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(history, index=False)
    return checkpoint


def _load_checkpoint(
    path: Path,
    *,
    anchor: str,
    residual_mode: str,
    alpha: float,
    seed: int,
    data: Any,
    config: Config,
) -> tuple[ProgressAnchoredResidual, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("anchor") != anchor
        or payload.get("residual_mode") != residual_mode
        or not math.isclose(float(payload.get("alpha", -1.0)), float(alpha), rel_tol=0.0, abs_tol=1e-12)
        or payload.get("seed") != seed
    ):
        raise ValueError(f"Exp5.5.2 checkpoint identity mismatch: {path}")
    model = _initialize_model(
        anchor=anchor,
        residual_mode=residual_mode,
        alpha=alpha,
        seed=seed,
        data=data,
        config=config,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def validate_source_seed(seed: int, data: Any, config: Config, force: bool = False) -> Path:
    if seed not in SEEDS:
        raise ValueError(seed)
    destination = source_reference_path(config.results_dir, seed)
    if destination.exists() and not force:
        load_source_reference(seed, config)
        return destination

    source_root = source_results_dir(config.repo_root)
    reference_path = exp55.reference_path(source_root, seed)
    source_paths: dict[str, str] = {}
    for anchor in ANCHORS:
        source_spec = exp55.RunSpec(_source_condition(anchor), seed)
        checkpoint = exp55.checkpoint_path(source_root, source_spec)
        evaluation = exp55.evaluation_path(source_root, source_spec)
        for path in (checkpoint, evaluation):
            if not path.exists():
                raise FileNotFoundError(path)
        source_paths[f"{anchor}_checkpoint"] = str(checkpoint.relative_to(config.repo_root))
        source_paths[f"{anchor}_evaluation"] = str(evaluation.relative_to(config.repo_root))
    reg_ordered = exp551.final_evaluation_path(source_551_results_dir(config.repo_root), seed)
    if not reg_ordered.exists():
        raise FileNotFoundError(reg_ordered)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    reference = exp55.load_reference(seed, _source_config(config))
    what, _ = _load_what_splits(seed, data, config, ("val",))
    loader = _make_loaders(what, data, seed, config, ("val",), False)["val"]
    device = torch.device(config.device)
    equivalence: dict[str, object] = {}
    for anchor in ANCHORS:
        source_model, _ = _source_anchor_model(anchor, seed, data, config)
        target = _initialize_model(
            anchor=anchor,
            residual_mode=ORDERED,
            alpha=1.0,
            seed=seed,
            data=data,
            config=config,
        )
        max_logits = 0.0
        max_q = 0.0
        predictions_identical = True
        with torch.no_grad():
            for xb, _, lengths in loader:
                xb = xb.to(device)
                lengths = lengths.to(device)
                source_logits, source_trajectory = source_model(
                    xb,
                    lengths,
                    float(data.fs),
                    float(reference["train_max_elapsed_seconds"]),
                    return_trajectory=True,
                )
                target_logits, target_trajectory = target(
                    xb,
                    lengths,
                    float(data.fs),
                    float(reference["train_max_elapsed_seconds"]),
                    return_trajectory=True,
                )
                if source_trajectory is None or source_trajectory.q is None or target_trajectory is None:
                    raise RuntimeError("Source equivalence requires q trajectories")
                max_logits = max(max_logits, float(torch.max(torch.abs(source_logits - target_logits)).item()))
                max_q = max(max_q, float(torch.max(torch.abs(source_trajectory.q - target_trajectory.q)).item()))
                predictions_identical = predictions_identical and bool(
                    torch.equal(source_logits.argmax(dim=1), target_logits.argmax(dim=1))
                )
        if max_logits > 1e-6 or max_q > 1e-6 or not predictions_identical:
            raise RuntimeError(
                f"Exp5.5.2 zero-residual source equivalence failed for {anchor}/seed{seed}: "
                f"logits={max_logits:.3e}, q={max_q:.3e}, pred={predictions_identical}"
            )
        equivalence[anchor] = {
            "val_max_abs_logits": max_logits,
            "val_max_abs_q": max_q,
            "val_predictions_identical": predictions_identical,
        }

    _save_json(
        destination,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "source_experiment_id": exp55.EXPERIMENT_ID,
            "source_protocol_version": exp55.PROTOCOL_VERSION,
            "source_reference": str(reference_path.relative_to(config.repo_root)),
            "source_regordered_5_5_1": str(reg_ordered.relative_to(config.repo_root)),
            "sources": source_paths,
            "split_seed": reference["split_seed"],
            "labels": reference["labels"],
            "train_users": reference["train_users"],
            "val_users": reference["val_users"],
            "test_users": reference["test_users"],
            "sampling_rate_hz": reference["sampling_rate_hz"],
            "equivalence": equivalence,
            "test_used_for_source_equivalence": False,
        },
    )
    return destination


def load_source_reference(seed: int, config: Config) -> dict[str, object]:
    path = source_reference_path(config.results_dir, seed)
    if not path.exists():
        raise FileNotFoundError(f"Run Exp5.5.2 source validation first: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("seed") != seed
        or payload.get("test_used_for_source_equivalence") is not False
    ):
        raise ValueError(f"Invalid Exp5.5.2 source reference: {path}")
    return payload


def train_screen_one(spec: ScreenSpec, data: Any, config: Config, force: bool = False) -> Path:
    if spec not in screen_specs():
        raise ValueError(spec)
    return _train_model(
        anchor=CLOCK,
        residual_mode=spec.residual_mode,
        alpha=spec.alpha,
        seed=spec.seed,
        data=data,
        config=config,
        checkpoint=screen_checkpoint_path(config.results_dir, spec),
        history=screen_history_path(config.results_dir, spec),
        force=force,
    )


def evaluate_screen_one(spec: ScreenSpec, data: Any, config: Config, force: bool = False) -> dict[str, object]:
    destination = screen_evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    load_source_reference(spec.seed, config)
    what, _ = _load_what_splits(spec.seed, data, config, ("train", "val"))
    loaders = _make_loaders(what, data, spec.seed, config, ("train", "val"), False)
    reference = exp55.load_reference(spec.seed, _source_config(config))
    model, checkpoint = _load_checkpoint(
        screen_checkpoint_path(config.results_dir, spec),
        anchor=CLOCK,
        residual_mode=spec.residual_mode,
        alpha=spec.alpha,
        seed=spec.seed,
        data=data,
        config=config,
    )
    device = torch.device(config.device)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "stage": "clock_screen",
        "spec": asdict(spec),
        "best_epoch": checkpoint["result"]["best_epoch"],
        "stopped_after_epoch": checkpoint["result"]["stopped_after_epoch"],
        "parameter_counts": parameter_counts(model),
        "train": _evaluate_classification(model, loaders["train"], data, reference, device),
        "val": _evaluate_classification(model, loaders["val"], data, reference, device),
        "test_evaluated": False,
        "loss": "whole-gesture classification CE only",
        "selection_policy": "best epoch by validation BA, tie-broken by validation CE",
    }
    _save_json(destination, payload)
    return payload


def run_screen_one(spec: ScreenSpec, data: Any, config: Config, force: bool = False) -> dict[str, object]:
    train_screen_one(spec, data, config, force=force)
    return evaluate_screen_one(spec, data, config, force=force)


def _sem(values: np.ndarray) -> float:
    return 0.0 if len(values) <= 1 else float(values.std(ddof=1) / math.sqrt(len(values)))


def select_screen(repo_root: Path) -> Path:
    root = results_dir(repo_root)
    run_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    selected_alpha_by_mode: dict[str, float] = {}
    rescue_supported_by_mode: dict[str, bool] = {}

    baselines = {
        seed: float(_source_eval(repo_root, CLOCK, seed)["val"]["balanced_accuracy"])
        for seed in SEEDS
    }
    for seed, value in baselines.items():
        run_rows.append(
            {
                "residual_mode": "anchor_only",
                "alpha": np.nan,
                "seed": seed,
                "val_balanced_accuracy": value,
                "val_loss": _source_eval(repo_root, CLOCK, seed)["val"]["loss"],
                "source": "Exp5.5 Clock",
            }
        )

    for spec in screen_specs():
        path = screen_evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Selector will not train missing screen run: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("test_evaluated") is not False:
            raise ValueError(f"Screen artifact touched test: {path}")
        val_ba = float(payload["val"]["balanced_accuracy"])
        run_rows.append(
            {
                "residual_mode": spec.residual_mode,
                "alpha": spec.alpha,
                "seed": spec.seed,
                "val_balanced_accuracy": val_ba,
                "val_loss": payload["val"]["loss"],
                "source": "Exp5.5.2 Clock screen",
            }
        )
        pair_rows.append(
            {
                "residual_mode": spec.residual_mode,
                "alpha": spec.alpha,
                "seed": spec.seed,
                "delta_val_balanced_accuracy_vs_clock": val_ba - baselines[spec.seed],
            }
        )

    runs = pd.DataFrame(run_rows)
    pairs = pd.DataFrame(pair_rows)
    for mode in RESIDUAL_MODES:
        mode_rows: list[dict[str, object]] = []
        for alpha in ALPHAS:
            group = runs[(runs["residual_mode"] == mode) & np.isclose(runs["alpha"], alpha)].sort_values("seed")
            pair_group = pairs[(pairs["residual_mode"] == mode) & np.isclose(pairs["alpha"], alpha)]
            vals = group["val_balanced_accuracy"].to_numpy(float)
            deltas = pair_group["delta_val_balanced_accuracy_vs_clock"].to_numpy(float)
            row = {
                "residual_mode": mode,
                "alpha": alpha,
                "mean_val_balanced_accuracy": float(vals.mean()),
                "sd_val_balanced_accuracy": float(vals.std(ddof=1)),
                "sem_val_balanced_accuracy": _sem(vals),
                "mean_delta_vs_clock": float(deltas.mean()),
                "nonnegative_seed_count": int(np.count_nonzero(deltas >= 0.0)),
                "eligible": bool(float(deltas.mean()) > 0.0 and np.count_nonzero(deltas >= 0.0) >= 4),
            }
            summary_rows.append(row)
            mode_rows.append(row)
        mode_frame = pd.DataFrame(mode_rows)
        eligible = mode_frame[mode_frame["eligible"]].copy()
        supported = not eligible.empty
        rescue_supported_by_mode[mode] = supported
        pool = eligible if supported else mode_frame
        best_index = pool["mean_val_balanced_accuracy"].idxmax()
        best = pool.loc[best_index]
        threshold = float(best["mean_val_balanced_accuracy"] - best["sem_val_balanced_accuracy"])
        within = pool[pool["mean_val_balanced_accuracy"] >= threshold].sort_values(
            ["alpha", "mean_val_balanced_accuracy"], ascending=[True, False]
        )
        selected_alpha_by_mode[mode] = float(within.iloc[0]["alpha"])

    root.mkdir(parents=True, exist_ok=True)
    screen_runs = root / "screen_runs.csv"
    screen_summary = root / "screen_summary.csv"
    screen_paired = root / "screen_paired_deltas.csv"
    runs.to_csv(screen_runs, index=False)
    pd.DataFrame(summary_rows).to_csv(screen_summary, index=False)
    pairs.to_csv(screen_paired, index=False)
    destination = selection_path(root)
    _save_json(
        destination,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "selected_alpha_by_mode": selected_alpha_by_mode,
            "clock_residual_rescue_supported_by_mode": rescue_supported_by_mode,
            "eligibility_rule": "per mode: mean paired Clock-val BA delta > 0 and >=4/5 seed deltas nonnegative",
            "selection_rule": "per mode: among eligible alphas, choose the smallest alpha within 1 SEM of the highest mean val BA; if none eligible, use the same 1-SEM/smallest-alpha fallback and mark rescue unsupported",
            "oracle_used_for_alpha_selection": False,
            "test_used_for_alpha_selection": False,
            "screen_runs": str(screen_runs.relative_to(repo_root)),
            "screen_summary": str(screen_summary.relative_to(repo_root)),
            "screen_paired_deltas": str(screen_paired.relative_to(repo_root)),
        },
    )
    return destination


def load_selection(config: Config) -> dict[str, object]:
    path = selection_path(config.results_dir)
    if not path.exists():
        raise FileNotFoundError("Exp5.5.2 selection.json is required before final stages")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_alpha_by_mode")
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("oracle_used_for_alpha_selection") is not False
        or payload.get("test_used_for_alpha_selection") is not False
        or not isinstance(selected, dict)
        or set(selected) != set(RESIDUAL_MODES)
    ):
        raise ValueError(f"Invalid Exp5.5.2 selection artifact: {path}")
    for mode in RESIDUAL_MODES:
        if float(selected[mode]) not in ALPHAS:
            raise ValueError(f"Invalid selected alpha for {mode}: {selected[mode]}")
    return payload


def _selected_alpha(config: Config, mode: str) -> float:
    return float(load_selection(config)["selected_alpha_by_mode"][mode])


def _collect_trajectory(
    model: ProgressAnchoredResidual,
    loader: DataLoader,
    data: Any,
    reference: dict[str, object],
    device: torch.device,
    *,
    include_what: bool,
) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {
        "anchor_logits": [],
        "anchor_q": [],
        "delta_z": [],
        "q": [],
        "anchor_evidence": [],
        "evidence": [],
        "labels": [],
        "lengths": [],
    }
    if include_what:
        parts["what"] = []
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
            if trajectory is None:
                raise RuntimeError("Expected residual trajectory")
            parts["anchor_logits"].append(trajectory.anchor_logits.cpu().numpy().astype(np.float32))
            parts["anchor_q"].append(trajectory.anchor_q.cpu().numpy().astype(np.float32))
            parts["delta_z"].append(trajectory.delta_z.cpu().numpy().astype(np.float32))
            parts["q"].append(trajectory.q.cpu().numpy().astype(np.float32))
            parts["anchor_evidence"].append(trajectory.anchor_evidence.cpu().numpy().astype(np.float32))
            parts["evidence"].append(trajectory.evidence.cpu().numpy().astype(np.float32))
            parts["labels"].append(labels.numpy().astype(np.int64))
            parts["lengths"].append(lengths.numpy().astype(np.int64))
            if include_what:
                parts["what"].append(what.numpy().astype(np.float32))
    return {name: np.concatenate(values) for name, values in parts.items()}


def _valid_flat(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.concatenate([values[row, : int(length)] for row, length in enumerate(lengths)], axis=0)


def residual_diagnostics(
    model: ProgressAnchoredResidual,
    trajectory: dict[str, np.ndarray],
) -> dict[str, float | int]:
    lengths = trajectory["lengths"]
    delta = _valid_flat(trajectory["delta_z"], lengths).astype(np.float64)
    q = _valid_flat(trajectory["q"], lengths).astype(np.float64)
    anchor_q = _valid_flat(trajectory["anchor_q"], lengths).astype(np.float64)
    evidence = _valid_flat(trajectory["evidence"], lengths).astype(np.float64)
    anchor_evidence = _valid_flat(trajectory["anchor_evidence"], lengths).astype(np.float64)
    dq = q - anchor_q

    experts = model.expert_weight.detach().cpu().numpy().astype(np.float64).reshape(N_STATES, -1)
    gram = experts @ experts.T
    delta_w_sq = np.einsum("ni,ij,nj->n", dq, gram, dq)
    anchor_w_sq = np.einsum("ni,ij,nj->n", anchor_q, gram, anchor_q)
    q_w_sq = np.einsum("ni,ij,nj->n", q, gram, q)
    cross = np.einsum("ni,ij,nj->n", anchor_q, gram, q)
    delta_w = np.sqrt(np.maximum(delta_w_sq, 0.0))
    anchor_w = np.sqrt(np.maximum(anchor_w_sq, 0.0))
    q_w = np.sqrt(np.maximum(q_w_sq, 0.0))
    r_w = delta_w / (anchor_w + EPS)
    weight_cos = cross / (anchor_w * q_w + EPS)

    delta_e = np.linalg.norm(evidence - anchor_evidence, axis=1)
    anchor_e = np.linalg.norm(anchor_evidence, axis=1)
    e_ratio = delta_e / (anchor_e + EPS)
    kl = np.sum(q * (np.log(np.clip(q, EPS, None)) - np.log(np.clip(anchor_q, EPS, None))), axis=1)
    l1 = np.abs(dq).sum(axis=1)
    delta_l2 = np.linalg.norm(delta, axis=1)
    saturation = np.mean(np.abs(delta) >= 0.9 * model.alpha)
    return {
        "n_valid_timesteps": int(len(delta)),
        "delta_z_l2_mean": float(delta_l2.mean()),
        "delta_z_l2_median": float(np.median(delta_l2)),
        "delta_z_saturation_fraction": float(saturation),
        "q_l1_mean": float(l1.mean()),
        "q_kl_to_anchor_mean": float(kl.mean()),
        "anchor_argmax_changed_fraction": float(np.mean(q.argmax(axis=1) != anchor_q.argmax(axis=1))),
        "relative_weight_correction_mean": float(r_w.mean()),
        "relative_weight_correction_median": float(np.median(r_w)),
        "anchor_vs_residual_weight_cosine_mean": float(weight_cos.mean()),
        "relative_evidence_correction_mean": float(e_ratio.mean()),
        "relative_evidence_correction_median": float(np.median(e_ratio)),
    }


def _residual_features(trajectory: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    delta = trajectory["delta_z"]
    lengths = trajectory["lengths"]
    mean_delta = np.stack(
        [delta[row, : int(length)].mean(axis=0) for row, length in enumerate(lengths)],
        axis=0,
    )
    final_delta = np.stack(
        [delta[row, max(int(length) - 1, 0)] for row, length in enumerate(lengths)],
        axis=0,
    )
    return {"mean_delta_z": mean_delta, "final_delta_z": final_delta}


def residual_only_probes(
    trajectories: dict[str, dict[str, np.ndarray]],
    seed: int,
    anchor: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    features = {split: _residual_features(trajectory) for split, trajectory in trajectories.items()}
    for feature_name in ("mean_delta_z", "final_delta_z"):
        scaler = StandardScaler()
        train_x = scaler.fit_transform(features["train"][feature_name])
        classifier = LogisticRegression(
            max_iter=RESIDUAL_PROBE_MAX_ITER,
            solver="lbfgs",
            random_state=base.dseed(seed, EXPERIMENT_ID, anchor, "residual_probe", feature_name),
        )
        classifier.fit(train_x, trajectories["train"]["labels"])
        for split in ("train", "val", "test"):
            x = train_x if split == "train" else scaler.transform(features[split][feature_name])
            logits = classifier.decision_function(x)
            metric = exp55._metrics_from_logits(trajectories[split]["labels"], logits)
            rows.append({"feature": feature_name, "split": split, **metric})
    return rows


def _q_from_anchor_and_delta(anchor_logits: np.ndarray, delta: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    values = anchor_logits.astype(np.float64) + delta.astype(np.float64)
    values = values - values.max(axis=-1, keepdims=True)
    q = np.exp(values)
    q /= q.sum(axis=-1, keepdims=True)
    for row, length in enumerate(lengths):
        q[row, int(length) :] = 0.0
    return q.astype(np.float32)


def residual_alignment_metrics(
    model: ProgressAnchoredResidual,
    trajectory: dict[str, np.ndarray],
    seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    if "what" not in trajectory:
        raise ValueError("Alignment diagnostics require WHAT")
    what = trajectory["what"]
    anchor_logits = trajectory["anchor_logits"]
    delta = trajectory["delta_z"]
    labels = trajectory["labels"]
    lengths = trajectory["lengths"]
    what_tensor = torch.tensor(what, dtype=torch.float32, device=device)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    rows: list[dict[str, object]] = []
    for ablation in ("circular_shift", "random_shuffle"):
        for replicate in range(SHUFFLE_REPLICATES):
            rng = np.random.default_rng(base.dseed(seed, EXPERIMENT_ID, model.anchor, ablation, replicate))
            shifted = delta.copy()
            for row, length_value in enumerate(lengths):
                length = int(length_value)
                if length <= 1:
                    continue
                if ablation == "circular_shift":
                    offset = int(rng.integers(1, length))
                    shifted[row, :length] = np.roll(delta[row, :length], shift=offset, axis=0)
                else:
                    shifted[row, :length] = delta[row, rng.permutation(length)]
            q = _q_from_anchor_and_delta(anchor_logits, shifted, lengths)
            with torch.no_grad():
                logits = model.logits_from_q(
                    what_tensor,
                    torch.tensor(q, dtype=torch.float32, device=device),
                    lengths_tensor,
                ).cpu().numpy()
            rows.append({"ablation": ablation, "replicate": replicate, **exp55._metrics_from_logits(labels, logits)})
    return rows


def clock_progress_alignment(trajectory: dict[str, np.ndarray]) -> dict[str, float | int]:
    q = trajectory["q"]
    anchor_q = trajectory["anchor_q"]
    lengths = trajectory["lengths"]
    centers = np.linspace(0.0, 1.0, N_STATES, dtype=np.float64)
    before: list[float] = []
    after: list[float] = []
    improved: list[bool] = []
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        if length <= 0:
            continue
        target = np.arange(length, dtype=np.float64) / max(length - 1, 1)
        anchor_hat = anchor_q[row, :length].astype(np.float64) @ centers
        residual_hat = q[row, :length].astype(np.float64) @ centers
        before_error = np.abs(anchor_hat - target)
        after_error = np.abs(residual_hat - target)
        before.extend(before_error.tolist())
        after.extend(after_error.tolist())
        improved.extend((after_error < before_error).tolist())
    return {
        "n_valid_timesteps": int(len(before)),
        "anchor_progress_mae": float(np.mean(before)),
        "residual_progress_mae": float(np.mean(after)),
        "mae_change_residual_minus_anchor": float(np.mean(after) - np.mean(before)),
        "fraction_timesteps_improved": float(np.mean(improved)),
        "progress_proxy": "routing center-of-mass over normalized expert index",
    }


def _classification_margin(evidence: np.ndarray, label: int) -> float:
    correct = float(evidence[label])
    mask = np.ones(len(evidence), dtype=bool)
    mask[label] = False
    return correct - float(np.max(evidence[mask]))


def same_what_same_coordinate(
    model: ProgressAnchoredResidual,
    trajectory: dict[str, np.ndarray],
    data: Any,
    seed: int,
) -> pd.DataFrame:
    if "what" not in trajectory:
        raise ValueError("same-WHAT diagnostic requires WHAT")
    what = trajectory["what"]
    q = trajectory["q"]
    anchor_q = trajectory["anchor_q"]
    delta = trajectory["delta_z"]
    evidence = trajectory["evidence"]
    anchor_evidence = trajectory["anchor_evidence"]
    labels = trajectory["labels"]
    lengths = trajectory["lengths"]
    points: list[tuple[np.ndarray, int, int, int, float, float]] = []
    for sample_index, length_value in enumerate(lengths):
        length = int(length_value)
        for timestep in range(length):
            vector = what[sample_index, timestep].astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0:
                continue
            coordinate = (
                float(timestep / max(length - 1, 1))
                if model.anchor == ORACLE
                else float(timestep / float(data.fs))
            )
            points.append((vector, sample_index, timestep, int(labels[sample_index]), coordinate, norm))
    columns = list(SAME_WHAT_COLUMNS)
    if len(points) < 2:
        return pd.DataFrame(columns=columns)
    rng = np.random.default_rng(base.dseed(seed, EXPERIMENT_ID, model.anchor, "same_what_sample"))
    if len(points) > SAME_WHAT_MAX_POINTS:
        selection = np.sort(rng.choice(len(points), size=SAME_WHAT_MAX_POINTS, replace=False))
        points = [points[index] for index in selection]
    vectors = np.stack([row[0] for row in points])
    neighbors = NearestNeighbors(
        n_neighbors=min(SAME_WHAT_NEIGHBORS, len(points)), metric="cosine", algorithm="brute"
    ).fit(vectors)
    distances, indices = neighbors.kneighbors(vectors)
    experts = model.expert_weight.detach().cpu().numpy().astype(np.float64).reshape(N_STATES, -1)
    coord_limit = ORACLE_COORD_DELTA if model.anchor == ORACLE else CLOCK_COORD_DELTA_SECONDS
    candidates: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for index, (distance_row, neighbor_row) in enumerate(zip(distances, indices)):
        _, sample_a, timestep_a, label_a, coord_a, norm_a = points[index]
        for distance, neighbor_index_value in zip(distance_row[1:], neighbor_row[1:]):
            neighbor_index = int(neighbor_index_value)
            _, sample_b, timestep_b, label_b, coord_b, norm_b = points[neighbor_index]
            if sample_a == sample_b or label_a == label_b:
                continue
            pair_key = tuple(sorted((index, neighbor_index)))
            if pair_key in seen:
                continue
            similarity = float(1.0 - distance)
            norm_ratio = max(norm_a, norm_b) / max(min(norm_a, norm_b), EPS)
            coord_delta = abs(coord_a - coord_b)
            if (
                similarity < SAME_WHAT_SIMILARITY
                or norm_ratio > SAME_WHAT_NORM_RATIO_MAX
                or coord_delta > coord_limit
            ):
                continue
            seen.add(pair_key)
            qa = q[sample_a, timestep_a].astype(np.float64)
            qb = q[sample_b, timestep_b].astype(np.float64)
            aa = anchor_q[sample_a, timestep_a].astype(np.float64)
            ab = anchor_q[sample_b, timestep_b].astype(np.float64)
            dwa = (qa - aa) @ experts
            dwb = (qb - ab) @ experts
            ea = evidence[sample_a, timestep_a].astype(np.float64)
            eb = evidence[sample_b, timestep_b].astype(np.float64)
            aea = anchor_evidence[sample_a, timestep_a].astype(np.float64)
            aeb = anchor_evidence[sample_b, timestep_b].astype(np.float64)
            anchor_margin_a = _classification_margin(aea, label_a)
            residual_margin_a = _classification_margin(ea, label_a)
            anchor_margin_b = _classification_margin(aeb, label_b)
            residual_margin_b = _classification_margin(eb, label_b)
            candidates.append(
                {
                    "anchor": model.anchor,
                    "seed": seed,
                    "sample_a": sample_a,
                    "timestep_a": timestep_a,
                    "label_a": label_a,
                    "sample_b": sample_b,
                    "timestep_b": timestep_b,
                    "label_b": label_b,
                    "what_cosine_similarity": similarity,
                    "what_norm_ratio": norm_ratio,
                    "coordinate_a": coord_a,
                    "coordinate_b": coord_b,
                    "coordinate_abs_delta": coord_delta,
                    "anchor_q_l1_distance": float(np.abs(aa - ab).sum()),
                    "delta_z_l1_distance": float(np.abs(delta[sample_a, timestep_a] - delta[sample_b, timestep_b]).sum()),
                    "q_l1_distance": float(np.abs(qa - qb).sum()),
                    "delta_w_pair_fro": float(np.linalg.norm(dwa - dwb)),
                    "top_support_a": int(np.argmax(ea)),
                    "top_support_b": int(np.argmax(eb)),
                    "different_top_support": bool(np.argmax(ea) != np.argmax(eb)),
                    "anchor_margin_a": anchor_margin_a,
                    "residual_margin_a": residual_margin_a,
                    "margin_delta_a": residual_margin_a - anchor_margin_a,
                    "anchor_margin_b": anchor_margin_b,
                    "residual_margin_b": residual_margin_b,
                    "margin_delta_b": residual_margin_b - anchor_margin_b,
                }
            )
            break
    candidates.sort(
        key=lambda row: (
            float(row["what_cosine_similarity"]),
            float(row["delta_z_l1_distance"]),
            float(row["delta_w_pair_fro"]),
        ),
        reverse=True,
    )
    return pd.DataFrame(candidates[:SAME_WHAT_TOP_PAIRS], columns=columns)


def _evaluate_final(
    spec: FinalSpec,
    checkpoint: Path,
    alpha: float,
    data: Any,
    config: Config,
    force: bool,
) -> dict[str, object]:
    destination = final_evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    load_source_reference(spec.seed, config)
    what, _ = _load_what_splits(spec.seed, data, config, ("train", "val", "test"))
    loaders = _make_loaders(what, data, spec.seed, config, ("train", "val", "test"), False)
    reference = exp55.load_reference(spec.seed, _source_config(config))
    model, checkpoint_payload = _load_checkpoint(
        checkpoint,
        anchor=spec.anchor,
        residual_mode=spec.residual_mode,
        alpha=alpha,
        seed=spec.seed,
        data=data,
        config=config,
    )
    device = torch.device(config.device)
    metrics = {
        split: _evaluate_classification(model, loaders[split], data, reference, device)
        for split in ("train", "val", "test")
    }
    include_all = spec.residual_mode == ORDERED
    trajectories = {
        split: _collect_trajectory(
            model,
            loaders[split],
            data,
            reference,
            device,
            include_what=(include_all and split == "test"),
        )
        for split in ("train", "val", "test")
    }
    diagnostics = {split: residual_diagnostics(model, trajectory) for split, trajectory in trajectories.items()}
    alignment: list[dict[str, object]] = []
    probes: list[dict[str, object]] = []
    progress_alignment: dict[str, object] | None = None
    trajectory_file: str | None = None
    same_file: str | None = None
    if spec.residual_mode == ORDERED:
        alignment = residual_alignment_metrics(model, trajectories["test"], spec.seed, device)
        probes = residual_only_probes(trajectories, spec.seed, spec.anchor)
        if spec.anchor == CLOCK:
            progress_alignment = clock_progress_alignment(trajectories["test"])
        path = trajectory_path(config.results_dir, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **trajectories["test"])
        trajectory_file = str(path.relative_to(config.repo_root))
        same = same_what_same_coordinate(model, trajectories["test"], data, spec.seed)
        same_path = same_what_path(config.results_dir, spec)
        same_path.parent.mkdir(parents=True, exist_ok=True)
        same.to_csv(same_path, index=False)
        same_file = str(same_path.relative_to(config.repo_root))

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "stage": "final",
        "spec": asdict(spec),
        "alpha": alpha,
        "best_epoch": checkpoint_payload["result"]["best_epoch"],
        "stopped_after_epoch": checkpoint_payload["result"]["stopped_after_epoch"],
        "source_checkpoint": checkpoint_payload["source_checkpoint"],
        "parameter_counts": parameter_counts(model),
        "metrics": metrics,
        "residual_diagnostics": diagnostics,
        "ordered_alignment_test": alignment,
        "residual_only_probes": probes,
        "clock_progress_alignment": progress_alignment,
        "trajectory": trajectory_file,
        "same_what_same_coordinate": same_file,
        "test_evaluated": True,
    }
    _save_json(destination, payload)
    return payload


def evaluate_clock_final_one(spec: FinalSpec, data: Any, config: Config, force: bool = False) -> dict[str, object]:
    if spec.anchor != CLOCK:
        raise ValueError(spec)
    alpha = _selected_alpha(config, spec.residual_mode)
    screen_spec = ScreenSpec(spec.residual_mode, alpha, spec.seed)
    checkpoint = screen_checkpoint_path(config.results_dir, screen_spec)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Selected Clock checkpoint must be reused from screen: {checkpoint}")
    return _evaluate_final(spec, checkpoint, alpha, data, config, force)


def train_oracle_final_one(spec: FinalSpec, data: Any, config: Config, force: bool = False) -> Path:
    if spec.anchor != ORACLE:
        raise ValueError(spec)
    alpha = _selected_alpha(config, spec.residual_mode)
    return _train_model(
        anchor=ORACLE,
        residual_mode=spec.residual_mode,
        alpha=alpha,
        seed=spec.seed,
        data=data,
        config=config,
        checkpoint=oracle_checkpoint_path(config.results_dir, spec),
        history=oracle_history_path(config.results_dir, spec),
        force=force,
    )


def run_oracle_final_one(spec: FinalSpec, data: Any, config: Config, force: bool = False) -> dict[str, object]:
    checkpoint = train_oracle_final_one(spec, data, config, force=force)
    alpha = _selected_alpha(config, spec.residual_mode)
    return _evaluate_final(spec, checkpoint, alpha, data, config, force)


def _final_source_551(seed: int, repo_root: Path) -> dict[str, object]:
    path = exp551.final_evaluation_path(source_551_results_dir(repo_root), seed)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    selection = json.loads(selection_path(root).read_text(encoding="utf-8"))
    if selection.get("test_used_for_alpha_selection") is not False:
        raise ValueError("Finalizer refuses a test-tuned selection artifact")

    payloads: dict[tuple[str, str, int], dict[str, object]] = {}
    for anchor in ANCHORS:
        for spec in final_mode_specs(anchor):
            path = final_evaluation_path(root, spec)
            if not path.exists():
                raise FileNotFoundError(f"Finalizer will not regenerate missing final artifact: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("test_evaluated") is not True:
                raise ValueError(path)
            payloads[(anchor, spec.residual_mode, spec.seed)] = payload

    run_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    progress_rows: list[dict[str, object]] = []
    same_frames: list[pd.DataFrame] = []

    for seed in SEEDS:
        baseline: dict[str, dict[str, object]] = {}
        for anchor in ANCHORS:
            source = _source_eval(repo_root, anchor, seed)
            baseline[anchor] = source
            run_rows.append(
                {
                    "condition": anchor,
                    "seed": seed,
                    "val_balanced_accuracy": source["val"]["balanced_accuracy"],
                    "test_balanced_accuracy": source["test"]["balanced_accuracy"],
                    "test_accuracy": source["test"]["accuracy"],
                    "test_macro_f1": source["test"]["macro_f1"],
                    "source": "Exp5.5 anchor",
                }
            )
        ref = json.loads(exp55.reference_path(source_results_dir(repo_root), seed).read_text(encoding="utf-8"))
        fixed = ref["fixed250"]["metrics"]
        run_rows.append(
            {
                "condition": "fixed250",
                "seed": seed,
                "val_balanced_accuracy": fixed["val"]["balanced_accuracy"],
                "test_balanced_accuracy": fixed["test"]["balanced_accuracy"],
                "test_accuracy": fixed["test"]["accuracy"],
                "test_macro_f1": fixed["test"]["macro_f1"],
                "source": "Exp5.5 Fixed250 reference",
            }
        )
        reg = _final_source_551(seed, repo_root)["ordered"]
        run_rows.append(
            {
                "condition": "regordered_5_5_1",
                "seed": seed,
                "val_balanced_accuracy": reg["val"]["balanced_accuracy"],
                "test_balanced_accuracy": reg["test"]["balanced_accuracy"],
                "test_accuracy": reg["test"]["accuracy"],
                "test_macro_f1": reg["test"]["macro_f1"],
                "source": "Exp5.5.1 selected RegOrdered",
            }
        )

        for anchor in ANCHORS:
            anchor_val = float(baseline[anchor]["val"]["balanced_accuracy"])
            anchor_test = float(baseline[anchor]["test"]["balanced_accuracy"])
            new_ba: dict[str, float] = {}
            for mode in RESIDUAL_MODES:
                payload = payloads[(anchor, mode, seed)]
                metrics = payload["metrics"]
                condition = f"{anchor}+{mode}"
                test_ba = float(metrics["test"]["balanced_accuracy"])
                new_ba[mode] = test_ba
                run_rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "val_balanced_accuracy": metrics["val"]["balanced_accuracy"],
                        "test_balanced_accuracy": test_ba,
                        "test_accuracy": metrics["test"]["accuracy"],
                        "test_macro_f1": metrics["test"]["macro_f1"],
                        "source": "Exp5.5.2 selected residual",
                    }
                )
                val_delta = float(metrics["val"]["balanced_accuracy"]) - anchor_val
                test_delta = test_ba - anchor_test
                gap_rows.append(
                    {
                        "anchor": anchor,
                        "residual_mode": mode,
                        "seed": seed,
                        "alpha": payload["alpha"],
                        "delta_val_balanced_accuracy": val_delta,
                        "delta_test_balanced_accuracy": test_delta,
                        "generalization_gap_val_minus_test_delta": val_delta - test_delta,
                    }
                )
                for split, diagnostic in payload["residual_diagnostics"].items():
                    diagnostic_rows.append(
                        {
                            "anchor": anchor,
                            "residual_mode": mode,
                            "seed": seed,
                            "alpha": payload["alpha"],
                            "split": split,
                            **diagnostic,
                        }
                    )
                if mode == ORDERED:
                    for row in payload["ordered_alignment_test"]:
                        alignment_rows.append({"anchor": anchor, "seed": seed, **row})
                    for row in payload["residual_only_probes"]:
                        probe_rows.append({"anchor": anchor, "seed": seed, **row})
                    if payload["clock_progress_alignment"] is not None:
                        progress_rows.append({"anchor": anchor, "seed": seed, **payload["clock_progress_alignment"]})
                    same_path_value = payload["same_what_same_coordinate"]
                    if same_path_value:
                        same_path = repo_root / str(same_path_value)
                        if not same_path.exists():
                            raise FileNotFoundError(same_path)
                        frame = pd.read_csv(same_path)
                        if not frame.empty:
                            same_frames.append(frame)

                paired_rows.append(
                    {
                        "seed": seed,
                        "anchor": anchor,
                        "comparison": f"{mode}_minus_anchor",
                        "delta_test_balanced_accuracy": test_ba - anchor_test,
                    }
                )

            ordered = new_ba[ORDERED]
            for control in (RESET, COORD, LAG1):
                paired_rows.append(
                    {
                        "seed": seed,
                        "anchor": anchor,
                        "comparison": f"ordered_minus_{control}",
                        "delta_test_balanced_accuracy": ordered - new_ba[control],
                    }
                )
            ordered_payload = payloads[(anchor, ORDERED, seed)]
            for ablation in ("circular_shift", "random_shuffle"):
                values = [
                    float(row["balanced_accuracy"])
                    for row in ordered_payload["ordered_alignment_test"]
                    if row["ablation"] == ablation
                ]
                if len(values) != SHUFFLE_REPLICATES:
                    raise ValueError(f"Expected {SHUFFLE_REPLICATES} {ablation} replicates")
                mean_value = float(np.mean(values))
                paired_rows.append(
                    {
                        "seed": seed,
                        "anchor": anchor,
                        "comparison": f"ordered_minus_{ablation}",
                        "delta_test_balanced_accuracy": ordered - mean_value,
                    }
                )

    runs = pd.DataFrame(run_rows)
    summary_rows: list[dict[str, object]] = []
    for condition, group in runs.groupby("condition", sort=False):
        values = group["test_balanced_accuracy"].to_numpy(float)
        summary_rows.append(
            {
                "condition": condition,
                "n_runs": int(len(group)),
                "mean_test_balanced_accuracy": float(values.mean()),
                "sd_test_balanced_accuracy": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "sem_test_balanced_accuracy": _sem(values),
                "mean_test_accuracy": float(group["test_accuracy"].mean()),
                "mean_test_macro_f1": float(group["test_macro_f1"].mean()),
            }
        )

    same = (
        pd.concat(same_frames, ignore_index=True)
        if same_frames
        else pd.DataFrame(columns=list(SAME_WHAT_COLUMNS))
    )
    outputs = {
        "final_runs": root / "final_runs.csv",
        "final_summary": root / "final_summary.csv",
        "final_paired_deltas": root / "final_paired_deltas.csv",
        "residual_diagnostics": root / "residual_diagnostics.csv",
        "residual_alignment": root / "residual_alignment.csv",
        "residual_only_probes": root / "residual_only_probes.csv",
        "generalization_gap": root / "generalization_gap.csv",
        "clock_progress_alignment": root / "clock_progress_alignment.csv",
        "same_what_same_coordinate": root / "same_what_same_coordinate.csv",
        "manifest": root / "manifest.json",
    }
    root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(outputs["final_runs"], index=False)
    pd.DataFrame(summary_rows).to_csv(outputs["final_summary"], index=False)
    pd.DataFrame(paired_rows).to_csv(outputs["final_paired_deltas"], index=False)
    pd.DataFrame(diagnostic_rows).to_csv(outputs["residual_diagnostics"], index=False)
    pd.DataFrame(alignment_rows).to_csv(outputs["residual_alignment"], index=False)
    pd.DataFrame(probe_rows).to_csv(outputs["residual_only_probes"], index=False)
    pd.DataFrame(gap_rows).to_csv(outputs["generalization_gap"], index=False)
    pd.DataFrame(progress_rows).to_csv(outputs["clock_progress_alignment"], index=False)
    same.to_csv(outputs["same_what_same_coordinate"], index=False)
    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seeds": list(SEEDS),
            "anchors": list(ANCHORS),
            "residual_modes": list(RESIDUAL_MODES),
            "alpha_screen": list(ALPHAS),
            "selected_alpha_by_mode": selection["selected_alpha_by_mode"],
            "clock_residual_rescue_supported_by_mode": selection["clock_residual_rescue_supported_by_mode"],
            "architecture": "frozen Exp5.5 Clock/Oracle expert bank + bounded residual routing logits",
            "routing_contract": "q_t = softmax(a_anchor_t + alpha*tanh(delta_t)); Ordered delta_t sees h_(t-1), Reset sees m_t, Lag1 sees only m_(t-1), Coordinate sees only anchor coordinate",
            "source_equivalence_contract": "zero residual projection must reproduce the paired Exp5.5 anchor logits/q/predictions within 1e-6",
            "loss": "whole-gesture classification CE only; no progress/confidence/sticky/diversity auxiliary",
            "selection_policy": "mode-specific alpha selected only on Clock validation; Oracle and test are excluded from selection",
            "multi_cpu_policy": "5 source validation -> 60 Clock screen tasks (max 50 concurrent) -> selector -> 20 Clock final evaluations + 20 Oracle train/evaluate tasks -> artifact-only finalizer",
            "notebook_policy": "analysis-only; consumes finalized CSV/JSON only and never trains/selects/regenerates",
            "note": "the five stochastic seeds share one fixed user-disjoint split and are not five independent split replicates",
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
    parser = argparse.ArgumentParser(description="Experiment 5.5.2 progress-anchored history-residual routing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repo-root", default=None)
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=1)
        command.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        command.add_argument("--epochs", type=int, default=EPOCHS)
        command.add_argument("--patience", type=int, default=PATIENCE)
        command.add_argument("--force", action="store_true")

    source = subparsers.add_parser("validate-source")
    add_common(source)
    source.add_argument("--array-task-id", type=int, required=True)

    screen = subparsers.add_parser("screen-one")
    add_common(screen)
    screen.add_argument("--array-task-id", type=int, required=True)

    select = subparsers.add_parser("select-screen")
    select.add_argument("--repo-root", default=None)

    clock_final = subparsers.add_parser("clock-final-one")
    add_common(clock_final)
    clock_final.add_argument("--array-task-id", type=int, required=True)

    oracle_final = subparsers.add_parser("oracle-final-one")
    add_common(oracle_final)
    oracle_final.add_argument("--array-task-id", type=int, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--repo-root", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "select-screen":
        root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
        print(select_screen(root))
        return
    if args.command == "finalize":
        root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
        for name, path in finalize_experiment(root).items():
            print(f"{name}: {path}")
        return

    config = _config_from_args(args)
    data = base.prepare_data(config.repo_root)
    task = int(args.array_task_id)
    if args.command == "validate-source":
        if task < 0 or task >= len(SEEDS):
            raise IndexError(task)
        print(validate_source_seed(SEEDS[task], data, config, force=args.force))
        return
    if args.command == "screen-one":
        specs = screen_specs()
        if task < 0 or task >= len(specs):
            raise IndexError(task)
        print(json.dumps(run_screen_one(specs[task], data, config, force=args.force), indent=2))
        return
    if args.command == "clock-final-one":
        specs = final_mode_specs(CLOCK)
        if task < 0 or task >= len(specs):
            raise IndexError(task)
        print(json.dumps(evaluate_clock_final_one(specs[task], data, config, force=args.force), indent=2))
        return
    if args.command == "oracle-final-one":
        specs = final_mode_specs(ORACLE)
        if task < 0 or task >= len(specs):
            raise IndexError(task)
        print(json.dumps(run_oracle_final_one(specs[task], data, config, force=args.force), indent=2))
        return
    raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
