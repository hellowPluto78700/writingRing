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
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_5_5_non_snn_history_experts as exp55


base = exp55.base
exp54 = exp55.exp54

EXPERIMENT_ID = "experiment_5_5_1_regularized_semantic_when"
PROTOCOL_VERSION = "regularized_semantic_when_v1"
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
PROGRESS_BETA = 0.1
PROGRESS_BINS = 10
MATCHED_PROGRESS_DELTA = 0.05
MATCHED_WHAT_SIMILARITY = 0.95
Q_PROGRESS_MAX_POINTS = 50_000

ORDERED = "ordered_gru"
RESET = "reset_gru"

R0 = "ce_only"
R1 = "progress"
R2 = "progress_sticky"
R3 = "progress_sticky_confidence"


@dataclass(frozen=True)
class Recipe:
    name: str
    lambda_progress: float
    lambda_sticky: float
    lambda_confidence: float
    complexity: int


RECIPES = {
    R0: Recipe(R0, 0.0, 0.0, 0.0, 0),
    R1: Recipe(R1, 0.3, 0.0, 0.0, 1),
    R2: Recipe(R2, 0.3, 0.1, 0.0, 2),
    R3: Recipe(R3, 0.3, 0.1, 0.02, 3),
}
SCREEN_RECIPES = (R1, R2, R3)


@dataclass(frozen=True)
class ScreenSpec:
    recipe: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.recipe}__seed{self.seed}"


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
    q: torch.Tensor
    evidence: torch.Tensor
    progress: torch.Tensor


def find_repo_root(start: Path | None = None) -> Path:
    return exp55.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_results_dir(repo_root: Path) -> Path:
    return exp55.results_dir(repo_root)


def screen_specs() -> list[ScreenSpec]:
    return [ScreenSpec(recipe, seed) for recipe in SCREEN_RECIPES for seed in SEEDS]


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


def reset_checkpoint_path(root: Path, seed: int) -> Path:
    return root / "reset_checkpoints" / f"seed{seed}.pt"


def reset_history_path(root: Path, seed: int) -> Path:
    return root / "reset_histories" / f"seed{seed}.csv"


def reset_evaluation_path(root: Path, seed: int) -> Path:
    return root / "reset_evaluations" / f"seed{seed}.json"


def final_evaluation_path(root: Path, seed: int) -> Path:
    return root / "final_evaluations" / f"seed{seed}.json"


def trajectory_path(root: Path, seed: int) -> Path:
    return root / "trajectories" / f"ordered__seed{seed}__test.npz"


def same_what_path(root: Path, seed: int) -> Path:
    return root / "same_what" / f"ordered__seed{seed}.csv"


def matched_progress_same_what_path(root: Path, seed: int) -> Path:
    return root / "same_what" / f"ordered__seed{seed}__matched_progress.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
            raise ValueError(f"Exp5.5.1 WHAT cache shape mismatch for {split}: {values.shape} != {expected}")
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
    loaders: dict[str, DataLoader] = {}
    for split in splits:
        labels, lengths = _labels_lengths(data, split)
        loaders[split] = _loader(
            what[split],
            labels,
            lengths,
            config.batch_size,
            train_shuffle if split == "train" else False,
            base.dseed(seed, EXPERIMENT_ID, "loader", split),
        )
    return loaders


def sequence_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    time = torch.arange(steps, device=lengths.device).unsqueeze(0)
    return time < lengths.unsqueeze(1)


def progress_targets(lengths: torch.Tensor, steps: int, dtype: torch.dtype) -> torch.Tensor:
    time = torch.arange(steps, device=lengths.device, dtype=dtype).unsqueeze(0)
    denom = (lengths.to(dtype) - 1.0).clamp(min=1.0).unsqueeze(1)
    return (time / denom).clamp(0.0, 1.0)


class RegularizedSemanticWhen(nn.Module):
    """Exp5.5 GRU/expert readout with an auxiliary progress head.

    The progress prediction never enters routing. Ordered routing remains strict-history:
    q_t = softmax(G h_(t-1)), while progress_t is predicted from h_t.
    """

    def __init__(
        self,
        mode: str,
        n_classes: int = N_CLASSES,
        what_width: int = WHAT_WIDTH,
        gru_width: int = GRU_WIDTH,
        n_states: int = N_STATES,
    ) -> None:
        super().__init__()
        if mode not in (ORDERED, RESET):
            raise ValueError(mode)
        self.mode = mode
        self.n_classes = int(n_classes)
        self.what_width = int(what_width)
        self.gru_width = int(gru_width)
        self.n_states = int(n_states)

        # Keep construction order identical to Exp5.5 before adding progress_head.
        self.class_bias = nn.Parameter(torch.zeros(self.n_classes))
        self.expert_weight = nn.Parameter(torch.empty(self.n_states, self.n_classes, self.what_width))
        nn.init.xavier_uniform_(self.expert_weight)
        self.gru = nn.GRU(self.what_width, self.gru_width, batch_first=True)
        self.state_head = nn.Linear(self.gru_width, self.n_states)
        self.progress_head = nn.Linear(self.gru_width, 1)

    def hidden_trajectories(self, what: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = what.shape
        if self.mode == ORDERED:
            h_current, _ = self.gru(what)
            h_zero = torch.zeros(batch, 1, self.gru_width, device=what.device, dtype=what.dtype)
            h_route = torch.cat((h_zero, h_current[:, :-1, :]), dim=1)
        else:
            flat = what.reshape(batch * steps, 1, self.what_width)
            h_flat, _ = self.gru(flat)
            h_current = h_flat[:, 0, :].reshape(batch, steps, self.gru_width)
            h_route = h_current
        return h_route, h_current

    def context_probabilities(
        self,
        what: torch.Tensor,
        lengths: torch.Tensor,
        sampling_rate_hz: float | None = None,
        train_max_elapsed_seconds: float | None = None,
    ) -> torch.Tensor:
        del sampling_rate_hz, train_max_elapsed_seconds
        h_route, _ = self.hidden_trajectories(what)
        q = torch.softmax(self.state_head(h_route), dim=-1)
        valid = sequence_mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
        return q * valid

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
        *,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, Trajectory | None]:
        h_route, h_current = self.hidden_trajectories(what)
        valid = sequence_mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
        q = torch.softmax(self.state_head(h_route), dim=-1) * valid
        progress = torch.sigmoid(self.progress_head(h_current)).squeeze(-1)
        evidence = self.evidence_from_q(what, q, lengths)
        logits = evidence.sum(dim=1) + self.class_bias
        trajectory = Trajectory(q=q, evidence=evidence, progress=progress) if return_trajectory else None
        return logits, trajectory


def parameter_counts(model: RegularizedSemanticWhen) -> dict[str, int]:
    return {
        "trainable_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "stored_total": int(sum(p.numel() for p in model.parameters())),
    }


def initialize_model(seed: int, mode: str, data: Any, config: Config) -> RegularizedSemanticWhen:
    # Reuse the Exp5.5 constructor seed so the shared core starts from the same initialization.
    model_seed = base.dseed(seed, exp55.EXPERIMENT_ID, "paired_constructor")
    base.seed_all(model_seed)
    return RegularizedSemanticWhen(
        mode=mode,
        n_classes=len(data.labels),
        what_width=WHAT_WIDTH,
        gru_width=GRU_WIDTH,
        n_states=N_STATES,
    ).to(config.device)


def loss_components(
    logits: torch.Tensor,
    labels: torch.Tensor,
    trajectory: Trajectory,
    lengths: torch.Tensor,
    recipe: Recipe,
) -> dict[str, torch.Tensor]:
    cls = F.cross_entropy(logits, labels)
    steps = trajectory.q.shape[1]
    valid = sequence_mask(lengths, steps)
    target = progress_targets(lengths, steps, trajectory.progress.dtype)
    progress_raw = F.smooth_l1_loss(
        trajectory.progress,
        target,
        reduction="none",
        beta=PROGRESS_BETA,
    )
    progress = (progress_raw * valid.to(progress_raw.dtype)).sum() / valid.sum().clamp(min=1)

    if steps > 1:
        pair_valid = sequence_mask((lengths - 1).clamp(min=0), steps - 1)
        sticky_raw = torch.abs(trajectory.q[:, 1:] - trajectory.q[:, :-1]).sum(dim=-1)
        sticky = (sticky_raw * pair_valid.to(sticky_raw.dtype)).sum() / pair_valid.sum().clamp(min=1)
    else:
        sticky = logits.new_zeros(())

    q_safe = trajectory.q.clamp(min=1e-12)
    entropy = -(trajectory.q * q_safe.log()).sum(dim=-1) / math.log(N_STATES)
    confidence = (entropy * valid.to(entropy.dtype)).sum() / valid.sum().clamp(min=1)

    weighted_progress = recipe.lambda_progress * progress
    weighted_sticky = recipe.lambda_sticky * sticky
    weighted_confidence = recipe.lambda_confidence * confidence
    total = cls + weighted_progress + weighted_sticky + weighted_confidence
    return {
        "classification": cls,
        "progress": progress,
        "sticky": sticky,
        "confidence": confidence,
        "weighted_progress": weighted_progress,
        "weighted_sticky": weighted_sticky,
        "weighted_confidence": weighted_confidence,
        "total": total,
    }


def _classification_metrics(labels: np.ndarray, pred: np.ndarray, loss: float) -> dict[str, float | int]:
    return {
        "loss": float(loss),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "n_samples": int(len(labels)),
    }


def evaluate_classification(
    model: RegularizedSemanticWhen,
    loader: DataLoader,
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
            logits, _ = model(what, lengths)
            loss = F.cross_entropy(logits, labels)
            n = len(labels)
            loss_sum += float(loss.item()) * n
            count += n
            true_parts.append(labels.cpu().numpy())
            pred_parts.append(logits.argmax(dim=1).cpu().numpy())
    true = np.concatenate(true_parts)
    pred = np.concatenate(pred_parts)
    return _classification_metrics(true, pred, loss_sum / max(count, 1))


def _train_model(
    *,
    seed: int,
    mode: str,
    recipe: Recipe,
    data: Any,
    config: Config,
    checkpoint: Path,
    history: Path,
    force: bool,
) -> Path:
    if checkpoint.exists() and not force:
        return checkpoint
    what, _ = _load_what_splits(seed, data, config, ("train", "val"))
    train_loader = _make_loaders(what, data, seed, config, ("train",), True)["train"]
    eval_loaders = _make_loaders(what, data, seed, config, ("train", "val"), False)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = initialize_model(seed, mode, data, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train0 = evaluate_classification(model, eval_loaders["train"], device)
    val0 = evaluate_classification(model, eval_loaders["val"], device)
    best_ba = float(val0["balanced_accuracy"])
    best_cls_loss = float(val0["loss"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    rows: list[dict[str, object]] = [
        {
            "epoch": 0,
            "train_classification_loss": train0["loss"],
            "train_balanced_accuracy": train0["balanced_accuracy"],
            "val_classification_loss": val0["loss"],
            "val_balanced_accuracy": val0["balanced_accuracy"],
            "train_progress_loss": np.nan,
            "train_sticky_loss": np.nan,
            "train_confidence_loss": np.nan,
            "train_weighted_progress_loss": np.nan,
            "train_weighted_sticky_loss": np.nan,
            "train_weighted_confidence_loss": np.nan,
            "train_total_loss": np.nan,
        }
    ]

    for epoch in range(1, config.epochs + 1):
        model.train()
        true_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        sums = {name: 0.0 for name in (
            "classification", "progress", "sticky", "confidence",
            "weighted_progress", "weighted_sticky", "weighted_confidence", "total"
        )}
        count = 0
        for xb, yb, lengths in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, trajectory = model(xb, lengths, return_trajectory=True)
            if trajectory is None:
                raise RuntimeError("Exp5.5.1 training requires trajectory")
            components = loss_components(logits, yb, trajectory, lengths, recipe)
            components["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            n = len(yb)
            count += n
            for name in sums:
                sums[name] += float(components[name].detach().item()) * n
            true_parts.append(yb.detach().cpu().numpy())
            pred_parts.append(logits.detach().argmax(dim=1).cpu().numpy())

        val = evaluate_classification(model, eval_loaders["val"], device)
        val_ba = float(val["balanced_accuracy"])
        val_cls_loss = float(val["loss"])
        rows.append(
            {
                "epoch": epoch,
                "train_classification_loss": sums["classification"] / max(count, 1),
                "train_balanced_accuracy": float(
                    balanced_accuracy_score(np.concatenate(true_parts), np.concatenate(pred_parts))
                ),
                "val_classification_loss": val_cls_loss,
                "val_balanced_accuracy": val_ba,
                "train_progress_loss": sums["progress"] / max(count, 1),
                "train_sticky_loss": sums["sticky"] / max(count, 1),
                "train_confidence_loss": sums["confidence"] / max(count, 1),
                "train_weighted_progress_loss": sums["weighted_progress"] / max(count, 1),
                "train_weighted_sticky_loss": sums["weighted_sticky"] / max(count, 1),
                "train_weighted_confidence_loss": sums["weighted_confidence"] / max(count, 1),
                "train_total_loss": sums["total"] / max(count, 1),
            }
        )
        improved = val_ba > best_ba + 1e-12 or (
            abs(val_ba - best_ba) <= 1e-12 and val_cls_loss < best_cls_loss - 1e-12
        )
        if improved:
            best_ba = val_ba
            best_cls_loss = val_cls_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "mode": mode,
            "recipe": asdict(recipe),
            "model_seed": base.dseed(seed, exp55.EXPERIMENT_ID, "paired_constructor"),
            "state_dict": best_state,
            "result": {
                "best_epoch": best_epoch,
                "best_val_balanced_accuracy": best_ba,
                "best_val_classification_loss": best_cls_loss,
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
    seed: int,
    mode: str,
    recipe: Recipe,
    data: Any,
    config: Config,
) -> tuple[RegularizedSemanticWhen, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("seed") != seed
        or payload.get("mode") != mode
        or payload.get("recipe") != asdict(recipe)
    ):
        raise ValueError(f"Exp5.5.1 checkpoint identity mismatch: {path}")
    model = initialize_model(seed, mode, data, config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def validate_source_seed(seed: int, data: Any, config: Config, force: bool = False) -> Path:
    if seed not in SEEDS:
        raise ValueError(seed)
    destination = source_reference_path(config.results_dir, seed)
    if destination.exists() and not force:
        return destination
    source_root = source_results_dir(config.repo_root)
    reference_path = exp55.reference_path(source_root, seed)
    required_evaluations = {
        condition: exp55.evaluation_path(source_root, exp55.RunSpec(condition, seed))
        for condition in (exp55.SHARED, exp55.CLOCK, exp55.ORACLE_PROGRESS, exp55.RESET_GRU, exp55.ORDERED_GRU)
    }
    missing = [path for path in (reference_path, *required_evaluations.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Exp5.5.1 requires finalized Exp5.5 artifacts; missing: {missing}")
    _, meta = _load_what_splits(seed, data, config, ("train", "val"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    _save_json(
        destination,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "source_experiment_id": exp55.EXPERIMENT_ID,
            "source_protocol_version": exp55.PROTOCOL_VERSION,
            "source_reference": str(reference_path.relative_to(config.repo_root)),
            "source_evaluations": {
                key: str(path.relative_to(config.repo_root)) for key, path in required_evaluations.items()
            },
            "source_cache_seed": meta.get("seed", seed),
            "what_definition": "frozen Local-SNN L2 spikes",
            "split_seed": reference["split_seed"],
            "sampling_rate_hz": reference["sampling_rate_hz"],
            "labels": reference["labels"],
            "train_users": reference["train_users"],
            "val_users": reference["val_users"],
            "test_users": reference["test_users"],
            "test_used_for_screen_selection": False,
        },
    )
    return destination


def train_screen_one(spec: ScreenSpec, data: Any, config: Config, force: bool = False) -> Path:
    if spec.recipe not in SCREEN_RECIPES or spec.seed not in SEEDS:
        raise ValueError(spec)
    recipe = RECIPES[spec.recipe]
    return _train_model(
        seed=spec.seed,
        mode=ORDERED,
        recipe=recipe,
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
    recipe = RECIPES[spec.recipe]
    what, _ = _load_what_splits(spec.seed, data, config, ("train", "val"))
    loaders = _make_loaders(what, data, spec.seed, config, ("train", "val"), False)
    model, checkpoint = _load_checkpoint(
        screen_checkpoint_path(config.results_dir, spec),
        seed=spec.seed,
        mode=ORDERED,
        recipe=recipe,
        data=data,
        config=config,
    )
    device = torch.device(config.device)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "stage": "screen",
        "spec": asdict(spec),
        "recipe": asdict(recipe),
        "model_seed": checkpoint["model_seed"],
        "best_epoch": checkpoint["result"]["best_epoch"],
        "stopped_after_epoch": checkpoint["result"]["stopped_after_epoch"],
        "parameter_counts": parameter_counts(model),
        "train": evaluate_classification(model, loaders["train"], device),
        "val": evaluate_classification(model, loaders["val"], device),
        "test_evaluated": False,
        "selection_policy": "best epoch by validation balanced accuracy, tie-broken by validation classification CE",
    }
    _save_json(destination, payload)
    return payload


def run_screen_one(spec: ScreenSpec, data: Any, config: Config, force: bool = False) -> dict[str, object]:
    train_screen_one(spec, data, config, force=force)
    return evaluate_screen_one(spec, data, config, force=force)


def _source_eval(repo_root: Path, condition: str, seed: int) -> dict[str, object]:
    path = exp55.evaluation_path(source_results_dir(repo_root), exp55.RunSpec(condition, seed))
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sem(values: np.ndarray) -> float:
    return 0.0 if len(values) <= 1 else float(values.std(ddof=1) / math.sqrt(len(values)))


def select_screen(repo_root: Path) -> Path:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    baseline: dict[int, float] = {}
    for seed in SEEDS:
        source = _source_eval(repo_root, exp55.ORDERED_GRU, seed)
        baseline[seed] = float(source["val"]["balanced_accuracy"])
        rows.append(
            {
                "recipe": R0,
                "seed": seed,
                "val_balanced_accuracy": baseline[seed],
                "val_classification_loss": source["val"]["loss"],
                "source": "Exp5.5 ordered_gru CE-only",
            }
        )
    for spec in screen_specs():
        path = screen_evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Screen selector will not train missing run: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("test_evaluated") is not False:
            raise ValueError(f"Screen artifact touched test: {path}")
        val_ba = float(payload["val"]["balanced_accuracy"])
        rows.append(
            {
                "recipe": spec.recipe,
                "seed": spec.seed,
                "val_balanced_accuracy": val_ba,
                "val_classification_loss": payload["val"]["loss"],
                "source": "Exp5.5.1 screen",
            }
        )
        paired.append(
            {
                "recipe": spec.recipe,
                "seed": spec.seed,
                "delta_val_balanced_accuracy_vs_ce_only": val_ba - baseline[spec.seed],
            }
        )

    runs = pd.DataFrame(rows)
    pair_df = pd.DataFrame(paired)
    summary_rows: list[dict[str, object]] = []
    for name in SCREEN_RECIPES:
        group = runs[runs["recipe"] == name].sort_values("seed")
        deltas = pair_df[pair_df["recipe"] == name]["delta_val_balanced_accuracy_vs_ce_only"].to_numpy(float)
        vals = group["val_balanced_accuracy"].to_numpy(float)
        summary_rows.append(
            {
                "recipe": name,
                "complexity": RECIPES[name].complexity,
                "mean_val_balanced_accuracy": float(vals.mean()),
                "sd_val_balanced_accuracy": float(vals.std(ddof=1)),
                "sem_val_balanced_accuracy": _sem(vals),
                "mean_delta_vs_ce_only": float(deltas.mean()),
                "nonnegative_seed_count": int(np.count_nonzero(deltas >= 0.0)),
                "eligible": bool(float(deltas.mean()) > 0.0 and np.count_nonzero(deltas >= 0.0) >= 4),
            }
        )
    summary = pd.DataFrame(summary_rows)
    eligible = summary[summary["eligible"]].copy()
    rescue_supported = not eligible.empty
    pool = eligible if rescue_supported else summary
    best_index = pool["mean_val_balanced_accuracy"].idxmax()
    best_row = pool.loc[best_index]
    threshold = float(best_row["mean_val_balanced_accuracy"] - best_row["sem_val_balanced_accuracy"])
    within = pool[pool["mean_val_balanced_accuracy"] >= threshold].copy()
    within = within.sort_values(["complexity", "mean_val_balanced_accuracy"], ascending=[True, False])
    selected = str(within.iloc[0]["recipe"])

    screen_runs_path = root / "screen_runs.csv"
    screen_summary_path = root / "screen_summary.csv"
    screen_paired_path = root / "screen_paired_deltas.csv"
    root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(screen_runs_path, index=False)
    summary.to_csv(screen_summary_path, index=False)
    pair_df.to_csv(screen_paired_path, index=False)
    destination = selection_path(root)
    _save_json(
        destination,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "selected_recipe": selected,
            "selected_recipe_parameters": asdict(RECIPES[selected]),
            "regularization_rescue_supported": rescue_supported,
            "eligibility_rule": "mean paired val BA delta > 0 and >=4/5 seed deltas nonnegative",
            "selection_rule": "among eligible recipes, choose lowest-complexity recipe within 1 SEM of the highest mean val BA; if none eligible, record validation-best fallback and mark rescue unsupported",
            "test_used_for_recipe_selection": False,
            "screen_runs": str(screen_runs_path.relative_to(repo_root)),
            "screen_summary": str(screen_summary_path.relative_to(repo_root)),
            "screen_paired_deltas": str(screen_paired_path.relative_to(repo_root)),
        },
    )
    return destination


def load_selection(config: Config) -> dict[str, object]:
    path = selection_path(config.results_dir)
    if not path.exists():
        raise FileNotFoundError("Exp5.5.1 selection.json is required before matched-reset/final stages")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("test_used_for_recipe_selection") is not False
        or payload.get("selected_recipe") not in SCREEN_RECIPES
    ):
        raise ValueError(f"Invalid Exp5.5.1 selection artifact: {path}")
    return payload


def train_reset_one(seed: int, data: Any, config: Config, force: bool = False) -> Path:
    selection = load_selection(config)
    recipe = RECIPES[str(selection["selected_recipe"])]
    return _train_model(
        seed=seed,
        mode=RESET,
        recipe=recipe,
        data=data,
        config=config,
        checkpoint=reset_checkpoint_path(config.results_dir, seed),
        history=reset_history_path(config.results_dir, seed),
        force=force,
    )


def evaluate_reset_validation_one(seed: int, data: Any, config: Config, force: bool = False) -> dict[str, object]:
    destination = reset_evaluation_path(config.results_dir, seed)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    selection = load_selection(config)
    recipe = RECIPES[str(selection["selected_recipe"])]
    what, _ = _load_what_splits(seed, data, config, ("train", "val"))
    loaders = _make_loaders(what, data, seed, config, ("train", "val"), False)
    model, checkpoint = _load_checkpoint(
        reset_checkpoint_path(config.results_dir, seed),
        seed=seed,
        mode=RESET,
        recipe=recipe,
        data=data,
        config=config,
    )
    device = torch.device(config.device)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "stage": "matched_reset",
        "seed": seed,
        "recipe": asdict(recipe),
        "model_seed": checkpoint["model_seed"],
        "best_epoch": checkpoint["result"]["best_epoch"],
        "stopped_after_epoch": checkpoint["result"]["stopped_after_epoch"],
        "parameter_counts": parameter_counts(model),
        "train": evaluate_classification(model, loaders["train"], device),
        "val": evaluate_classification(model, loaders["val"], device),
        "test_evaluated": False,
    }
    _save_json(destination, payload)
    return payload


def run_reset_one(seed: int, data: Any, config: Config, force: bool = False) -> dict[str, object]:
    train_reset_one(seed, data, config, force=force)
    return evaluate_reset_validation_one(seed, data, config, force=force)


def _collect_trajectory(
    model: RegularizedSemanticWhen,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    what_parts: list[np.ndarray] = []
    q_parts: list[np.ndarray] = []
    evidence_parts: list[np.ndarray] = []
    progress_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    length_parts: list[np.ndarray] = []
    with torch.no_grad():
        for what, labels, lengths in loader:
            what_device = what.to(device)
            lengths_device = lengths.to(device)
            _, trajectory = model(what_device, lengths_device, return_trajectory=True)
            if trajectory is None:
                raise RuntimeError("Expected Exp5.5.1 trajectory")
            what_parts.append(what.numpy().astype(np.uint8))
            q_parts.append(trajectory.q.cpu().numpy().astype(np.float32))
            evidence_parts.append(trajectory.evidence.cpu().numpy().astype(np.float32))
            progress_parts.append(trajectory.progress.cpu().numpy().astype(np.float32))
            label_parts.append(labels.numpy().astype(np.int64))
            length_parts.append(lengths.numpy().astype(np.int64))
    return {
        "what": np.concatenate(what_parts),
        "q": np.concatenate(q_parts),
        "evidence": np.concatenate(evidence_parts),
        "progress": np.concatenate(progress_parts),
        "labels": np.concatenate(label_parts),
        "lengths": np.concatenate(length_parts),
    }


def _flatten_progress(trajectory: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    q_parts: list[np.ndarray] = []
    for row, length_value in enumerate(trajectory["lengths"]):
        length = int(length_value)
        if length <= 0:
            continue
        denom = max(length - 1, 1)
        target = np.arange(length, dtype=np.float64) / denom
        pred_parts.append(trajectory["progress"][row, :length].astype(np.float64))
        target_parts.append(target)
        q_parts.append(trajectory["q"][row, :length].astype(np.float64))
    return np.concatenate(pred_parts), np.concatenate(target_parts), np.concatenate(q_parts)


def _rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def progress_metrics(trajectory: dict[str, np.ndarray]) -> dict[str, float | int]:
    pred, target, _ = _flatten_progress(trajectory)
    error = pred - target
    rmse = float(np.sqrt(np.mean(error**2)))
    pearson = float(np.corrcoef(pred, target)[0, 1]) if len(pred) > 1 else 0.0
    spearman = float(np.corrcoef(_rankdata(pred), _rankdata(target))[0, 1]) if len(pred) > 1 else 0.0
    pred_bin = np.minimum(PROGRESS_BINS - 1, np.floor(pred * PROGRESS_BINS).astype(int))
    target_bin = np.minimum(PROGRESS_BINS - 1, np.floor(target * PROGRESS_BINS).astype(int))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": rmse,
        "r2": float(r2_score(target, pred)),
        "pearson": pearson,
        "spearman": spearman,
        "coarse_10bin_accuracy": float(np.mean(pred_bin == target_bin)),
        "n_timesteps": int(len(pred)),
    }


def q_progress_probe(
    trajectories: dict[str, dict[str, np.ndarray]],
    seed: int,
) -> dict[str, float | int]:
    _, train_target, train_q = _flatten_progress(trajectories["train"])
    _, test_target, test_q = _flatten_progress(trajectories["test"])
    if len(train_target) > Q_PROGRESS_MAX_POINTS:
        rng = np.random.default_rng(base.dseed(seed, EXPERIMENT_ID, "q_progress_probe"))
        idx = np.sort(rng.choice(len(train_target), size=Q_PROGRESS_MAX_POINTS, replace=False))
        train_target = train_target[idx]
        train_q = train_q[idx]
    model = Ridge(alpha=1.0)
    model.fit(train_target[:, None], train_q)
    pred = model.predict(test_target[:, None])
    return {
        "test_r2_uniform_average": float(r2_score(test_q, pred, multioutput="uniform_average")),
        "test_q_mae": float(np.mean(np.abs(test_q - pred))),
        "n_train_timesteps": int(len(train_target)),
        "n_test_timesteps": int(len(test_target)),
        "probe": "Ridge(true_relative_progress -> q), train-fit/test-evaluated",
    }


def _matched_progress_same_what(
    base_frame: pd.DataFrame,
    trajectory: dict[str, np.ndarray],
) -> pd.DataFrame:
    columns = list(base_frame.columns) + ["progress_a", "progress_b", "progress_abs_delta"]
    if base_frame.empty:
        return pd.DataFrame(columns=columns)
    lengths = trajectory["lengths"]
    frame = base_frame.copy()
    progress_a: list[float] = []
    progress_b: list[float] = []
    for row in frame.itertuples(index=False):
        la = int(lengths[int(row.sample_a)])
        lb = int(lengths[int(row.sample_b)])
        progress_a.append(float(int(row.timestep_a) / max(la - 1, 1)))
        progress_b.append(float(int(row.timestep_b) / max(lb - 1, 1)))
    frame["progress_a"] = progress_a
    frame["progress_b"] = progress_b
    frame["progress_abs_delta"] = np.abs(frame["progress_a"] - frame["progress_b"])
    frame = frame[
        (frame["what_cosine_similarity"] >= MATCHED_WHAT_SIMILARITY)
        & (frame["progress_abs_delta"] <= MATCHED_PROGRESS_DELTA)
    ]
    return frame.reset_index(drop=True)


def _evaluate_final_mode(
    *,
    seed: int,
    mode: str,
    recipe: Recipe,
    checkpoint: Path,
    what: dict[str, np.ndarray],
    data: Any,
    config: Config,
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]], RegularizedSemanticWhen]:
    loaders = _make_loaders(what, data, seed, config, ("train", "val", "test"), False)
    model, checkpoint_payload = _load_checkpoint(
        checkpoint,
        seed=seed,
        mode=mode,
        recipe=recipe,
        data=data,
        config=config,
    )
    device = torch.device(config.device)
    trajectories = {
        split: _collect_trajectory(model, loaders[split], device)
        for split in ("train", "val", "test")
    }
    payload: dict[str, object] = {
        "mode": mode,
        "model_seed": checkpoint_payload["model_seed"],
        "best_epoch": checkpoint_payload["result"]["best_epoch"],
        "stopped_after_epoch": checkpoint_payload["result"]["stopped_after_epoch"],
        "parameter_counts": parameter_counts(model),
        "train": evaluate_classification(model, loaders["train"], device),
        "val": evaluate_classification(model, loaders["val"], device),
        "test": evaluate_classification(model, loaders["test"], device),
        "progress_metrics": {
            split: progress_metrics(trajectories[split]) for split in ("train", "val", "test")
        },
        "state_diagnostics": exp55._state_diagnostics(
            trajectories["test"]["q"], trajectories["test"]["lengths"], float(data.fs)
        ),
        "q_only_probes": exp55._q_only_probes(trajectories, seed),
        "q_progress_probe": q_progress_probe(trajectories, seed),
    }
    return payload, trajectories, model


def evaluate_final_one(seed: int, data: Any, config: Config, force: bool = False) -> dict[str, object]:
    destination = final_evaluation_path(config.results_dir, seed)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    selection = load_selection(config)
    recipe_name = str(selection["selected_recipe"])
    recipe = RECIPES[recipe_name]
    ordered_spec = ScreenSpec(recipe_name, seed)
    ordered_checkpoint = screen_checkpoint_path(config.results_dir, ordered_spec)
    reset_checkpoint = reset_checkpoint_path(config.results_dir, seed)
    if not reset_evaluation_path(config.results_dir, seed).exists():
        raise FileNotFoundError("Matched reset validation artifact must exist before final test")

    what, source_meta = _load_what_splits(seed, data, config, ("train", "val", "test"))
    ordered, ordered_traj, ordered_model = _evaluate_final_mode(
        seed=seed,
        mode=ORDERED,
        recipe=recipe,
        checkpoint=ordered_checkpoint,
        what=what,
        data=data,
        config=config,
    )
    reset, _, _ = _evaluate_final_mode(
        seed=seed,
        mode=RESET,
        recipe=recipe,
        checkpoint=reset_checkpoint,
        what=what,
        data=data,
        config=config,
    )

    test_trajectory = ordered_traj["test"]
    device = torch.device(config.device)
    shuffled = exp55._shuffled_q_metrics(ordered_model, test_trajectory, seed, device)
    traj_path = trajectory_path(config.results_dir, seed)
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(traj_path, **test_trajectory)
    same_frame = exp55._same_what_pairs(test_trajectory, seed)
    pair_path = same_what_path(config.results_dir, seed)
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    same_frame.to_csv(pair_path, index=False)
    matched_frame = _matched_progress_same_what(same_frame, test_trajectory)
    matched_path = matched_progress_same_what_path(config.results_dir, seed)
    matched_frame.to_csv(matched_path, index=False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "stage": "final_test",
        "seed": seed,
        "recipe": asdict(recipe),
        "selection": {
            "selected_recipe": recipe_name,
            "regularization_rescue_supported": selection["regularization_rescue_supported"],
            "test_used_for_recipe_selection": False,
        },
        "ordered": ordered,
        "reset": reset,
        "ordered_shuffled_q_test": shuffled,
        "test_trajectory": str(traj_path.relative_to(config.repo_root)),
        "same_what_pairs": str(pair_path.relative_to(config.repo_root)),
        "matched_progress_same_what": str(matched_path.relative_to(config.repo_root)),
        "source": {
            "source_experiment_id": exp55.EXPERIMENT_ID,
            "source_protocol_version": exp55.PROTOCOL_VERSION,
            "source_cache_seed": source_meta.get("seed", seed),
            "what_definition": "frozen Local-SNN L2 spikes",
            "split_seed": base.SPLIT_SEED,
            "sampling_rate_hz": float(data.fs),
        },
        "test_used_for_recipe_selection": False,
    }
    _save_json(destination, payload)
    return payload


def _flatten_state_diagnostics(condition: str, seed: int, diagnostic: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in (
        "entropy_mean", "entropy_median", "total_variation_l1_mean",
        "argmax_transitions_per_second", "mean_dwell_seconds", "median_dwell_seconds",
    ):
        rows.append({"condition": condition, "seed": seed, "metric": metric, "state": None, "value": diagnostic[metric]})
    for state, value in enumerate(diagnostic["occupancy"]):
        rows.append({"condition": condition, "seed": seed, "metric": "occupancy", "state": state, "value": value})
    return rows


def _flatten_state_profiles(condition: str, seed: int, diagnostic: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for axis, centers_key, profile_key in (
        ("relative_progress", "relative_profile_centers", "relative_profile"),
        ("elapsed_seconds", "absolute_profile_centers_seconds", "absolute_profile"),
    ):
        for center, probabilities in zip(diagnostic[centers_key], diagnostic[profile_key]):
            for state, probability in enumerate(probabilities):
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "axis": axis,
                        "position": center,
                        "state": state,
                        "probability": probability,
                    }
                )
    return rows


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    selection = json.loads(selection_path(root).read_text(encoding="utf-8")) if selection_path(root).exists() else None
    if selection is None:
        raise FileNotFoundError("Finalizer will not create missing selection.json")
    recipe_name = str(selection["selected_recipe"])
    payloads: dict[int, dict[str, object]] = {}
    for seed in SEEDS:
        path = final_evaluation_path(root, seed)
        if not path.exists():
            raise FileNotFoundError(f"Finalizer will not run/evaluate missing final seed: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("test_used_for_recipe_selection") is not False:
            raise ValueError(path)
        payloads[seed] = payload

    run_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    progress_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    q_only_rows: list[dict[str, object]] = []
    q_progress_rows: list[dict[str, object]] = []
    shuffle_rows: list[dict[str, object]] = []
    same_frames: list[pd.DataFrame] = []
    matched_frames: list[pd.DataFrame] = []

    for seed in SEEDS:
        source_ref = json.loads(exp55.reference_path(source_results_dir(repo_root), seed).read_text(encoding="utf-8"))
        baseline_conditions = (
            ("shared", exp55.SHARED),
            ("clock", exp55.CLOCK),
            ("oracle_progress", exp55.ORACLE_PROGRESS),
            ("ce_reset", exp55.RESET_GRU),
            ("ce_ordered", exp55.ORDERED_GRU),
        )
        baseline_ba: dict[str, float] = {}
        for label, condition in baseline_conditions:
            source = _source_eval(repo_root, condition, seed)
            ba = float(source["test"]["balanced_accuracy"])
            baseline_ba[label] = ba
            run_rows.append(
                {
                    "condition": label,
                    "seed": seed,
                    "val_balanced_accuracy": source["val"]["balanced_accuracy"],
                    "test_balanced_accuracy": ba,
                    "test_accuracy": source["test"]["accuracy"],
                    "test_macro_f1": source["test"]["macro_f1"],
                    "source": "Exp5.5",
                }
            )
        fixed = source_ref["fixed250"]["metrics"]
        fixed_ba = float(fixed["test"]["balanced_accuracy"])
        run_rows.append(
            {
                "condition": "fixed250",
                "seed": seed,
                "val_balanced_accuracy": fixed["val"]["balanced_accuracy"],
                "test_balanced_accuracy": fixed_ba,
                "test_accuracy": fixed["test"]["accuracy"],
                "test_macro_f1": fixed["test"]["macro_f1"],
                "source": "Exp5.5 Fixed250 reference",
            }
        )

        payload = payloads[seed]
        for label, block in (("reg_ordered", payload["ordered"]), ("reg_reset", payload["reset"])):
            run_rows.append(
                {
                    "condition": label,
                    "seed": seed,
                    "val_balanced_accuracy": block["val"]["balanced_accuracy"],
                    "test_balanced_accuracy": block["test"]["balanced_accuracy"],
                    "test_accuracy": block["test"]["accuracy"],
                    "test_macro_f1": block["test"]["macro_f1"],
                    "source": "Exp5.5.1 selected recipe",
                }
            )
            for split, metrics in block["progress_metrics"].items():
                progress_rows.append({"condition": label, "seed": seed, "split": split, **metrics})
            state_rows.extend(_flatten_state_diagnostics(label, seed, block["state_diagnostics"]))
            profile_rows.extend(_flatten_state_profiles(label, seed, block["state_diagnostics"]))
            for row in block["q_only_probes"]:
                q_only_rows.append({"condition": label, "seed": seed, **row})
            q_progress_rows.append({"condition": label, "seed": seed, **block["q_progress_probe"]})

        shuffled_mean = float(np.mean([row["balanced_accuracy"] for row in payload["ordered_shuffled_q_test"]]))
        run_rows.append(
            {
                "condition": "reg_ordered_shuffled_q",
                "seed": seed,
                "val_balanced_accuracy": np.nan,
                "test_balanced_accuracy": shuffled_mean,
                "test_accuracy": float(np.mean([row["accuracy"] for row in payload["ordered_shuffled_q_test"]])),
                "test_macro_f1": float(np.mean([row["macro_f1"] for row in payload["ordered_shuffled_q_test"]])),
                "source": "Exp5.5.1 within-gesture shuffled q",
            }
        )
        for row in payload["ordered_shuffled_q_test"]:
            shuffle_rows.append({"condition": "reg_ordered", "seed": seed, **row})

        reg_ordered_ba = float(payload["ordered"]["test"]["balanced_accuracy"])
        reg_reset_ba = float(payload["reset"]["test"]["balanced_accuracy"])
        comparisons = (
            ("reg_ordered_minus_ce_ordered", reg_ordered_ba, baseline_ba["ce_ordered"]),
            ("reg_ordered_minus_reg_reset", reg_ordered_ba, reg_reset_ba),
            ("reg_ordered_minus_clock", reg_ordered_ba, baseline_ba["clock"]),
            ("reg_ordered_minus_oracle_progress", reg_ordered_ba, baseline_ba["oracle_progress"]),
            ("reg_ordered_minus_shuffled_q", reg_ordered_ba, shuffled_mean),
            ("reg_ordered_minus_fixed250", reg_ordered_ba, fixed_ba),
        )
        for name, left, right in comparisons:
            paired_rows.append(
                {
                    "seed": seed,
                    "comparison": name,
                    "delta_test_balanced_accuracy": left - right,
                }
            )

        same_path = same_what_path(root, seed)
        matched_path = matched_progress_same_what_path(root, seed)
        if not same_path.exists() or not matched_path.exists():
            raise FileNotFoundError(f"Missing same-WHAT diagnostics for seed {seed}")
        same = pd.read_csv(same_path)
        matched = pd.read_csv(matched_path)
        if not same.empty:
            same_frames.append(same)
        if not matched.empty:
            matched_frames.append(matched)

    runs = pd.DataFrame(run_rows)
    summaries: list[dict[str, object]] = []
    for condition, group in runs.groupby("condition", sort=False):
        values = group["test_balanced_accuracy"].to_numpy(float)
        summaries.append(
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

    outputs = {
        "final_runs": root / "final_runs.csv",
        "final_summary": root / "final_summary.csv",
        "final_paired_deltas": root / "final_paired_deltas.csv",
        "progress_metrics": root / "progress_metrics.csv",
        "state_diagnostics": root / "state_diagnostics.csv",
        "state_profiles": root / "state_profiles.csv",
        "q_only_probes": root / "q_only_probes.csv",
        "q_progress_probe": root / "q_progress_probe.csv",
        "shuffled_q": root / "shuffled_q.csv",
        "same_what_pairs": root / "same_what_pairs.csv",
        "matched_progress_same_what": root / "matched_progress_same_what.csv",
        "manifest": root / "manifest.json",
    }
    root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(outputs["final_runs"], index=False)
    pd.DataFrame(summaries).to_csv(outputs["final_summary"], index=False)
    pd.DataFrame(paired_rows).to_csv(outputs["final_paired_deltas"], index=False)
    pd.DataFrame(progress_rows).to_csv(outputs["progress_metrics"], index=False)
    pd.DataFrame(state_rows).to_csv(outputs["state_diagnostics"], index=False)
    pd.DataFrame(profile_rows).to_csv(outputs["state_profiles"], index=False)
    pd.DataFrame(q_only_rows).to_csv(outputs["q_only_probes"], index=False)
    pd.DataFrame(q_progress_rows).to_csv(outputs["q_progress_probe"], index=False)
    pd.DataFrame(shuffle_rows).to_csv(outputs["shuffled_q"], index=False)
    same = pd.concat(same_frames, ignore_index=True) if same_frames else pd.DataFrame(columns=[
        "seed", "sample_a", "timestep_a", "label_a", "sample_b", "timestep_b", "label_b",
        "what_cosine_similarity", "q_l1_distance", "evidence_l1_distance", "top_support_a",
        "top_support_b", "different_top_support", "high_similarity"
    ])
    matched = pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame(columns=[
        *same.columns, "progress_a", "progress_b", "progress_abs_delta"
    ])
    same.to_csv(outputs["same_what_pairs"], index=False)
    matched.to_csv(outputs["matched_progress_same_what"], index=False)
    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seeds": list(SEEDS),
            "screen_recipes": [asdict(RECIPES[name]) for name in SCREEN_RECIPES],
            "selected_recipe": recipe_name,
            "regularization_rescue_supported": selection["regularization_rescue_supported"],
            "architecture": "frozen WHAT128 -> GRU64 -> strict-history semantic q8/full 12x128 experts + auxiliary progress head",
            "routing_contract": "q_t = softmax(G h_(t-1)) for ordered mode; predicted progress never enters q or expert weights",
            "progress_contract": "progress_t = sigmoid(P h_t), supervised by t/(T-1) on valid timesteps only",
            "loss": "whole-gesture CE + selected weak progress/sticky/confidence auxiliaries; no balance loss",
            "selection_policy": "validation-only nested recipe screen; test unavailable until selection.json is fixed",
            "multi_cpu_policy": "5 source-validation -> 15 ordered screen -> selector -> 5 matched reset -> 5 final test/diagnostic -> artifact-only finalizer",
            "notebook_policy": "analysis-only; finalized CSV/JSON input only; no training, Slurm, selection, or regeneration",
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
    parser = argparse.ArgumentParser(description="Experiment 5.5.1 regularized semantic WHEN")
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

    reset = subparsers.add_parser("reset-one")
    add_common(reset)
    reset.add_argument("--array-task-id", type=int, required=True)

    final = subparsers.add_parser("final-one")
    add_common(final)
    final.add_argument("--array-task-id", type=int, required=True)

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
    if args.command == "reset-one":
        if task < 0 or task >= len(SEEDS):
            raise IndexError(task)
        print(json.dumps(run_reset_one(SEEDS[task], data, config, force=args.force), indent=2))
        return
    if args.command == "final-one":
        if task < 0 or task >= len(SEEDS):
            raise IndexError(task)
        print(json.dumps(evaluate_final_one(SEEDS[task], data, config, force=args.force), indent=2))
        return
    raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
