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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F

from core_benchmark_v1.data import load_cache
from core_benchmark_v1.diagnostics import run_diagnostics
from core_benchmark_v1.model import BenchmarkNet, mean_logits, valid_mask
from core_benchmark_v1.probes import run_probes
from core_benchmark_v1.protocol import Protocol, Run, digest, paired_seed
from core_benchmark_v1.storage import file_hash, save_json, save_torch
from core_benchmark_v1.training import cpu_state, extract, metrics


EXPERIMENT_ID = "experiment_14_history_organization"
PROTOCOL_VERSION = "history_organization_v1"
CORE_RESULTS_REL = Path("core_benchmark_v1/results/main")
SEEDS = (11, 23, 37)
SHIFTS = ((2, 3, 4), (2, 3, 4))
LAMBDA_GRID = (0.01, 0.03, 0.1, 0.3)
PHASES = (0.25, 0.50, 0.75, 1.00)
HISTORY_MS = 250
HISTORY_EVAL_MS = (50, 100, 250, 500, 1000)
PROJECTION_HIDDEN = 64
PROJECTION_DIM = 32
CONTRASTIVE_TEMPERATURE = 0.1
WARMUP_EPOCHS = 10
RAMP_END_EPOCH = 30
CLASSES_PER_BATCH = 8
USERS_PER_CLASS = 4
SAMPLES_PER_USER = 4
EXPECTED_BATCH_SIZE = CLASSES_PER_BATCH * USERS_PER_CLASS * SAMPLES_PER_USER


@dataclass(frozen=True)
class LossSpec:
    case: str
    seed: int
    lambda_class: float = 0.0
    lambda_history: float = 0.0
    shuffled_auxiliary: bool = False
    phase: int = 1

    @property
    def key(self) -> str:
        parts = [self.case, f"seed{self.seed}"]
        if self.lambda_class:
            parts.append(f"lc{self.lambda_class:g}")
        if self.lambda_history:
            parts.append(f"lh{self.lambda_history:g}")
        return "__".join(parts).replace(".", "p")


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    core_results_dir: Path


def find_repo_root(start: Path | None = None) -> Path:
    path = (start or Path(__file__)).resolve()
    for parent in (path, *path.parents):
        if (parent / "AGENTS.md").exists() and (parent / "core_benchmark_v1").exists():
            return parent
    raise FileNotFoundError("Could not locate repository root")


def default_results_dir(root: Path) -> Path:
    return root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def config_from_args(args: argparse.Namespace) -> Config:
    root = find_repo_root()
    return Config(
        root,
        (args.results or default_results_dir(root)).resolve(),
        (args.core_results or root / CORE_RESULTS_REL).resolve(),
    )


def phase1_specs() -> list[LossSpec]:
    specs: list[LossSpec] = []
    for seed in SEEDS:
        specs.append(LossSpec("C0_wcce", seed))
        for value in LAMBDA_GRID:
            specs.append(LossSpec("C1_whole_cu", seed, lambda_class=value))
            specs.append(LossSpec("C2_phase_cu", seed, lambda_class=value))
            specs.append(LossSpec("C3_history_delta", seed, lambda_history=value))
        specs.append(LossSpec("C5_shuffled_phase_control", seed, lambda_class=0.1, shuffled_auxiliary=True))
    return specs


def phase2_specs(selection: dict[str, Any]) -> list[LossSpec]:
    lc = float(selection["C2_phase_cu"]["lambda"])
    lh = float(selection["C3_history_delta"]["lambda"])
    strengths = (("best", lc, lh), ("half", 0.5 * lc, 0.5 * lh))
    return [
        LossSpec(f"C4_combined_{name}", seed, lambda_class=a, lambda_history=b, phase=2)
        for seed in SEEDS
        for name, a, b in strengths
    ]


def _load_core(config: Config) -> tuple[Protocol, dict[str, Any], dict[str, np.ndarray]]:
    lock_path = config.core_results_dir / "protocol.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    locked_version = str(lock["protocol"].get("version"))
    if locked_version not in {"core_benchmark_v1.0", "core_benchmark_v1.1"}:
        raise ValueError(f"Unsupported CoreBenchmark artifact version: {locked_version}")
    protocol_payload = dict(lock["protocol"])
    protocol_payload["version"] = Protocol().version
    p = Protocol.from_dict(protocol_payload)
    if file_hash(config.core_results_dir / "dataset.npz") != lock["dataset_hash"]:
        raise ValueError("Frozen CoreBenchmark dataset hash changed")
    arrays = load_cache(config.core_results_dir, p, lock)
    if tuple(p.seeds) != SEEDS:
        raise ValueError(f"Core seeds changed: {p.seeds}")
    if (p.width, p.fs, p.input_channels, p.tau_mem_ms) != (128, 64.0, 30, 22.54):
        raise ValueError("Core geometry no longer matches Exp14 contract")
    expected = {
        "train": ["user_0", "user_1", "user_11", "user_12", "user_13", "user_14", "user_15",
                  "user_18", "user_19", "user_2", "user_20", "user_5", "user_7", "user_8"],
        "val": ["user_16", "user_4", "user_9"],
        "test": ["user_10", "user_3", "user_6"],
    }
    actual = {key: list(value) for key, value in lock["data_metadata"]["users"].items()}
    if actual != expected:
        raise ValueError(f"Core user split changed: {actual}")
    return p, lock, arrays


def prepare(config: Config) -> dict[str, Any]:
    p, lock, arrays = _load_core(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "core_protocol_hash": lock["protocol_hash"],
        "core_dataset_hash": lock["dataset_hash"],
        "seeds": list(SEEDS),
        "shifts": [list(v) for v in SHIFTS],
        "tau_mem_ms": p.tau_mem_ms,
        "fs": p.fs,
        "width": p.width,
        "history_ms": HISTORY_MS,
        "lambda_grid": list(LAMBDA_GRID),
        "phases": list(PHASES),
        "batch_contract": {
            "classes_per_batch": CLASSES_PER_BATCH,
            "users_per_class": USERS_PER_CLASS,
            "samples_per_user": SAMPLES_PER_USER,
            "batch_size": EXPECTED_BATCH_SIZE,
        },
        "users": lock["data_metadata"]["users"],
        "phase1_runs": [asdict(spec) | {"key": spec.key} for spec in phase1_specs()],
        "train_samples": int(len(arrays["train_y"])),
    }
    payload["identity"] = digest(payload)
    save_json(config.results_dir / "protocol.json", payload)
    return payload


class ProjectionHead(nn.Module):
    def __init__(self, seed: int, input_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, PROJECTION_HIDDEN),
            nn.ReLU(),
            nn.Linear(PROJECTION_HIDDEN, PROJECTION_DIM),
        )
        for name, parameter in self.named_parameters():
            with torch.no_grad():
                if parameter.ndim == 1:
                    parameter.zero_()
                else:
                    generator = torch.Generator().manual_seed(
                        paired_seed(seed, f"exp14:projection:{name}")
                    )
                    bound = 1.0 / math.sqrt(parameter.shape[1])
                    parameter.uniform_(-bound, bound, generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def _valid_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = valid_mask(lengths, values.shape[1]).to(values.dtype).unsqueeze(-1)
    return (values * mask).sum(1) / lengths.to(values.dtype).unsqueeze(1)


def _phase_vectors(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    result = []
    for phase in PHASES:
        index = torch.ceil(lengths.to(torch.float32) * phase).to(torch.long).sub(1).clamp_min(0)
        result.append(values[batch, index])
    return torch.stack(result, dim=1)


def cross_user_supcon(embeddings: torch.Tensor, labels: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
    logits = embeddings @ embeddings.T / CONTRASTIVE_TEMPERATURE
    n = len(labels)
    eye = torch.eye(n, dtype=torch.bool, device=labels.device)
    positive = (labels[:, None] == labels[None, :]) & (users[:, None] != users[None, :]) & ~eye
    valid_anchor = positive.any(dim=1)
    if not bool(valid_anchor.any()):
        raise ValueError("Batch contains no cross-user positive anchors")
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    per_anchor = -(log_prob * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return per_anchor[valid_anchor].mean()


def _structured_batches(arrays: dict[str, np.ndarray], seed: int, epoch: int) -> Iterable[np.ndarray]:
    labels = arrays["train_y"]
    users = arrays["train_users"]
    classes = np.unique(labels)
    rng = np.random.default_rng(paired_seed(seed, f"exp14:sampler:{epoch}"))
    groups: dict[int, dict[str, np.ndarray]] = {}
    for class_id in classes.tolist():
        groups[int(class_id)] = {}
        for user in sorted(set(users[labels == class_id].tolist())):
            indices = np.flatnonzero((labels == class_id) & (users == user))
            if len(indices):
                groups[int(class_id)][str(user)] = indices
        if len(groups[int(class_id)]) < USERS_PER_CLASS:
            raise ValueError(f"Class {class_id} has fewer than {USERS_PER_CLASS} train users")
    count = max(1, int(math.ceil(len(labels) / EXPECTED_BATCH_SIZE)))
    for _ in range(count):
        selected_classes = rng.choice(classes, size=CLASSES_PER_BATCH, replace=False)
        batch: list[int] = []
        for class_id in selected_classes.tolist():
            user_map = groups[int(class_id)]
            selected_users = rng.choice(np.asarray(sorted(user_map), dtype=object), size=USERS_PER_CLASS, replace=False)
            for user in selected_users.tolist():
                pool = user_map[str(user)]
                chosen = rng.choice(pool, size=SAMPLES_PER_USER, replace=len(pool) < SAMPLES_PER_USER)
                batch.extend(int(value) for value in chosen.tolist())
        indices = np.asarray(batch, dtype=np.int64)
        rng.shuffle(indices)
        yield indices


def _aux_weight(target: float, epoch: int) -> float:
    if target <= 0 or epoch <= WARMUP_EPOCHS:
        return 0.0
    if epoch >= RAMP_END_EPOCH:
        return float(target)
    return float(target) * (epoch - WARMUP_EPOCHS) / (RAMP_END_EPOCH - WARMUP_EPOCHS)


def _run(spec: LossSpec) -> Run:
    return Run(spec.case, spec.seed, "14_history_organization", shifts=SHIFTS, objective="wcce")


def _evaluate_validation(model: BenchmarkNet, arrays: dict[str, np.ndarray]) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(arrays["val_x"])
        y = torch.from_numpy(arrays["val_y"])
        lengths = torch.from_numpy(arrays["val_lengths"])
        scores = mean_logits(model(x, lengths)["evidence"], lengths)
        out = metrics(y.numpy(), scores.argmax(1).numpy())
        out["mean_logit_ce"] = float(F.cross_entropy(scores, y))
        return out


def _history_delta(
    model: BenchmarkNet,
    x: torch.Tensor,
    lengths: torch.Tensor,
    full_pre: torch.Tensor,
    history_steps: int,
    anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    if anchor is None:
        anchor = lengths - 1
    reset_at = (anchor - history_steps + 1).clamp_min(0)
    truncated = model(x, lengths, reset_at=reset_at, reset_layers=(0, 1))["pre_reset"][1]
    batch = torch.arange(len(lengths), device=x.device)
    return full_pre[batch, anchor] - truncated[batch, anchor]


def _history_delta_phases(
    model: BenchmarkNet,
    x: torch.Tensor,
    lengths: torch.Tensor,
    full_pre: torch.Tensor,
    history_steps: int,
) -> torch.Tensor:
    values = []
    for phase in PHASES:
        anchor = torch.ceil(lengths.to(torch.float32) * phase).to(torch.long).sub(1).clamp_min(0)
        values.append(_history_delta(model, x, lengths, full_pre, history_steps, anchor))
    return torch.stack(values, dim=1)


def _loss(
    model: BenchmarkNet,
    projector: ProjectionHead,
    x: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    users: torch.Tensor,
    spec: LossSpec,
    epoch: int,
    shuffle_seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    trajectory = model(x, lengths)
    wcce = F.cross_entropy(mean_logits(trajectory["evidence"], lengths), y)
    class_loss = wcce.new_tensor(0.0)
    history_loss = wcce.new_tensor(0.0)
    auxiliary_y = y
    if spec.shuffled_auxiliary:
        generator = torch.Generator().manual_seed(shuffle_seed)
        auxiliary_y = y[torch.randperm(len(y), generator=generator)]

    if spec.case == "C1_whole_cu":
        class_loss = cross_user_supcon(projector(_valid_mean(trajectory["pre_reset"][1], lengths)), auxiliary_y, users)
    elif spec.case in ("C2_phase_cu", "C5_shuffled_phase_control") or spec.case.startswith("C4_combined"):
        phase_values = _phase_vectors(trajectory["pre_reset"][1], lengths)
        class_loss = torch.stack([
            cross_user_supcon(projector(phase_values[:, index]), auxiliary_y, users)
            for index in range(phase_values.shape[1])
        ]).mean()

    if spec.case == "C3_history_delta" or spec.case.startswith("C4_combined"):
        steps = max(1, int(round(HISTORY_MS * model.protocol.fs / 1000.0)))
        delta = _history_delta(model, x, lengths, trajectory["pre_reset"][1], steps)
        history_loss = cross_user_supcon(projector(delta), y, users)

    wc = _aux_weight(spec.lambda_class, epoch)
    wh = _aux_weight(spec.lambda_history, epoch)
    total = wcce + wc * class_loss + wh * history_loss
    return total, {
        "wcce": float(wcce.detach()),
        "class_loss": float(class_loss.detach()),
        "history_loss": float(history_loss.detach()),
        "class_weight": wc,
        "history_weight": wh,
    }


def train_one(config: Config, spec: LossSpec) -> dict[str, Any]:
    p, core_lock, arrays = _load_core(config)
    directory = config.results_dir / "runs" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    if checkpoint_path.exists():
        return {"status": "exists", "key": spec.key}

    model = BenchmarkNet(_run(spec), p)
    projector = ProjectionHead(spec.seed, p.width)
    initial = cpu_state(model)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + (list(projector.parameters()) if spec.lambda_class or spec.lambda_history else []),
        lr=p.learning_rate,
        weight_decay=p.weight_decay,
    )
    user_names = sorted(set(arrays["train_users"].tolist()))
    user_map = {str(name): index for index, name in enumerate(user_names)}
    user_ids = np.asarray([user_map[str(value)] for value in arrays["train_users"].tolist()], dtype=np.int64)

    best = _evaluate_validation(model, arrays)
    best_state, best_projector, best_epoch = cpu_state(model), cpu_state(projector), 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, p.max_epochs + 1):
        model.train()
        projector.train()
        total = {"loss": 0.0, "wcce": 0.0, "class": 0.0, "history": 0.0, "n": 0}
        for batch_index, indices in enumerate(_structured_batches(arrays, spec.seed, epoch)):
            x = torch.from_numpy(arrays["train_x"][indices])
            y = torch.from_numpy(arrays["train_y"][indices])
            lengths = torch.from_numpy(arrays["train_lengths"][indices])
            users = torch.from_numpy(user_ids[indices])
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _loss(
                model, projector, x, y, lengths, users, spec, epoch,
                paired_seed(spec.seed, f"exp14:shuffle:{epoch}:{batch_index}"),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite loss")
            loss.backward()
            optimizer.step()
            n = len(indices)
            total["loss"] += float(loss.detach()) * n
            total["wcce"] += parts["wcce"] * n
            total["class"] += parts["class_loss"] * n
            total["history"] += parts["history_loss"] * n
            total["n"] += n
        val = _evaluate_validation(model, arrays)
        history.append({
            "epoch": epoch,
            "train_total_loss": total["loss"] / total["n"],
            "train_wcce": total["wcce"] / total["n"],
            "train_class_loss": total["class"] / total["n"],
            "train_history_loss": total["history"] / total["n"],
            "class_weight": _aux_weight(spec.lambda_class, epoch),
            "history_weight": _aux_weight(spec.lambda_history, epoch),
            "val_ba": val["ba"],
            "val_mean_logit_ce": val["mean_logit_ce"],
        })
        improved = val["ba"] > best["ba"] + 1e-12 or (
            abs(val["ba"] - best["ba"]) <= 1e-12 and val["mean_logit_ce"] < best["mean_logit_ce"] - 1e-12
        )
        if improved:
            best, best_state, best_projector, best_epoch = val, cpu_state(model), cpu_state(projector), epoch
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    model.load_state_dict(best_state)
    projector.load_state_dict(best_projector)
    save_torch(checkpoint_path, {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "core_identity": core_lock["identity"],
        "model_state_dict": best_state,
        "projection_state_dict": best_projector,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "best_val": best,
        "initial_model_state_dict": initial,
        "selection_rule": "held-out-user validation BA, then validation valid-mean-logit CE, then earliest epoch",
    })
    save_json(directory / "history.json", {"rows": history})
    evaluate_one(config, spec, model)
    return {"status": "PASS", "key": spec.key, "best_val_ba": best["ba"]}


def _load_model(config: Config, spec: LossSpec) -> BenchmarkNet:
    p, _, _ = _load_core(config)
    payload = torch.load(config.results_dir / "runs" / spec.key / "checkpoint.pt", map_location="cpu", weights_only=False)
    if payload["spec"] != asdict(spec):
        raise ValueError(f"Checkpoint spec mismatch: {spec.key}")
    model = BenchmarkNet(_run(spec), p)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def _resample(values: np.ndarray, lengths: np.ndarray, steps: int = 64) -> np.ndarray:
    out: list[np.ndarray] = []
    for value, valid in zip(values, lengths):
        valid = max(1, min(int(valid), value.shape[0]))
        source = np.asarray(value[:valid], dtype=np.float32)
        if valid == 1:
            out.append(np.repeat(source, steps, axis=0))
            continue
        old = np.linspace(0.0, 1.0, valid)
        new = np.linspace(0.0, 1.0, steps)
        target = np.empty((steps, source.shape[1]), dtype=np.float32)
        for dimension in range(source.shape[1]):
            target[:, dimension] = np.interp(new, old, source[:, dimension])
        out.append(target)
    return np.stack(out)


def _trajectory_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an, bn = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2)
    dot = np.sum(a * b, axis=2)
    denom = an * bn
    similarity = np.divide(dot, np.maximum(denom, 1e-12), out=np.zeros_like(dot), where=denom > 1e-12)
    distance = 1.0 - np.clip(similarity, -1.0, 1.0)
    distance[(an <= 1e-12) & (bn <= 1e-12)] = 0.0
    return distance.mean(axis=1)


def _geometry(traces: dict[str, np.ndarray], arrays: dict[str, np.ndarray], spec: LossSpec) -> list[dict[str, Any]]:
    rng = np.random.default_rng(paired_seed(spec.seed, "exp14:geometry"))
    rows: list[dict[str, Any]] = []
    for state in ("spike", "pre_reset"):
        for layer in ("L1", "L2"):
            for split in ("train", "test"):
                trajectories = _resample(traces[f"{split}__{layer}__{state}"], arrays[f"{split}_lengths"])
                y, users = arrays[f"{split}_y"], arrays[f"{split}_users"]
                same: list[float] = []
                different: list[float] = []
                for index in range(len(y)):
                    sc = np.flatnonzero((y == y[index]) & (users != users[index]))
                    dc = np.flatnonzero((y != y[index]) & (users != users[index]))
                    if len(sc):
                        j = int(rng.choice(sc))
                        same.append(float(_trajectory_distance(trajectories[index:index+1], trajectories[j:j+1])[0]))
                    if len(dc):
                        j = int(rng.choice(dc))
                        different.append(float(_trajectory_distance(trajectories[index:index+1], trajectories[j:j+1])[0]))
                d_sc, d_dc = float(np.mean(same)), float(np.mean(different))
                rows.append({
                    "case": spec.case, "seed": spec.seed, "state": state, "layer": layer, "split": split,
                    "same_class_cross_user_distance": d_sc,
                    "different_class_cross_user_distance": d_dc,
                    "cross_user_ratio": d_dc / max(d_sc, 1e-12),
                })
    return rows


def _retrieval(traces: dict[str, np.ndarray], arrays: dict[str, np.ndarray], spec: LossSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = np.unique(arrays["train_y"])
    for state in ("spike", "pre_reset"):
        for layer in ("L1", "L2"):
            train = _resample(traces[f"train__{layer}__{state}"], arrays["train_lengths"])
            test = _resample(traces[f"test__{layer}__{state}"], arrays["test_lengths"])
            prototypes = np.stack([train[arrays["train_y"] == label].mean(axis=0) for label in labels])
            pred = []
            for query in test:
                repeated = np.repeat(query[None], len(prototypes), axis=0)
                pred.append(int(labels[int(np.argmin(_trajectory_distance(repeated, prototypes)))]))
            rows.append({
                "case": spec.case, "seed": spec.seed, "state": state, "layer": layer,
                "retrieval_ba": float(balanced_accuracy_score(arrays["test_y"], np.asarray(pred))),
            })
    return rows


def _final_state(model: BenchmarkNet, arrays: dict[str, np.ndarray], split: str, history_ms: int | None) -> np.ndarray:
    x = torch.from_numpy(arrays[f"{split}_x"])
    lengths = torch.from_numpy(arrays[f"{split}_lengths"])
    if history_ms is None:
        trajectory = model(x, lengths)
    else:
        steps = max(1, int(round(history_ms * model.protocol.fs / 1000.0)))
        trajectory = model(x, lengths, reset_at=(lengths - steps).clamp_min(0), reset_layers=(0, 1))
    batch = torch.arange(len(lengths))
    return trajectory["pre_reset"][1][batch, lengths - 1].detach().numpy().astype(np.float64)


def _history_generalization(model: BenchmarkNet, arrays: dict[str, np.ndarray], spec: LossSpec) -> list[dict[str, Any]]:
    train_y, test_y = arrays["train_y"], arrays["test_y"]
    train_users = arrays["train_users"]
    users = sorted(set(train_users.tolist()))
    folds_by_user = {str(user): index % 3 for index, user in enumerate(users)}
    folds = np.asarray([folds_by_user[str(user)] for user in train_users], dtype=np.int64)

    def id_cv(x: np.ndarray) -> float:
        truth, pred = [], []
        for fold in range(3):
            train_mask, val_mask = folds != fold, folds == fold
            scaler = StandardScaler().fit(x[train_mask])
            clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=spec.seed)
            clf.fit(scaler.transform(x[train_mask]), train_y[train_mask])
            truth.append(train_y[val_mask])
            pred.append(clf.predict(scaler.transform(x[val_mask])))
        return float(balanced_accuracy_score(np.concatenate(truth), np.concatenate(pred)))

    durations: tuple[int | None, ...] = (*HISTORY_EVAL_MS, None)
    feature_cache = {
        (split, duration): _final_state(model, arrays, split, duration)
        for split in ("train", "test")
        for duration in durations
    }
    base_id = id_cv(feature_cache[("train", 50)])
    base_scaler = StandardScaler().fit(feature_cache[("train", 50)])
    base_clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=spec.seed)
    base_clf.fit(base_scaler.transform(feature_cache[("train", 50)]), train_y)
    base_ood = float(balanced_accuracy_score(
        test_y, base_clf.predict(base_scaler.transform(feature_cache[("test", 50)]))
    ))

    rows = []
    for duration in durations:
        train_x, test_x = feature_cache[("train", duration)], feature_cache[("test", duration)]
        id_ba = id_cv(train_x)
        scaler = StandardScaler().fit(train_x)
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=spec.seed)
        clf.fit(scaler.transform(train_x), train_y)
        ood_ba = float(balanced_accuracy_score(test_y, clf.predict(scaler.transform(test_x))))
        rows.append({
            "case": spec.case,
            "seed": spec.seed,
            "history_ms": "full" if duration is None else int(duration),
            "id_ba": id_ba,
            "ood_ba": ood_ba,
            "id_gain_pp_vs_50": 100.0 * (id_ba - base_id),
            "ood_gain_pp_vs_50": 100.0 * (ood_ba - base_ood),
            "excess_seen_gain_pp": 100.0 * ((id_ba - base_id) - (ood_ba - base_ood)),
        })
    return rows


def evaluate_one(config: Config, spec: LossSpec, model: BenchmarkNet | None = None) -> dict[str, Any]:
    p, _, arrays = _load_core(config)
    directory = config.results_dir / "runs" / spec.key
    model = model or _load_model(config, spec)
    model.eval()
    traces, native = extract(model, arrays, p, _run(spec))
    save_json(directory / "native.json", native)
    run_probes(directory, traces, arrays, _run(spec), p)
    diagnostic_protocol = Protocol.from_dict({
        **asdict(p),
        "diagnostic_cases": [spec.case],
        "history_ms": list(HISTORY_EVAL_MS),
    })
    run_diagnostics(directory, model, traces, arrays, _run(spec), diagnostic_protocol)
    pd.DataFrame(_geometry(traces, arrays, spec)).to_csv(directory / "cross_user_geometry.csv", index=False)
    pd.DataFrame(_retrieval(traces, arrays, spec)).to_csv(directory / "cross_user_retrieval.csv", index=False)
    pd.DataFrame(_history_generalization(model, arrays, spec)).to_csv(directory / "history_generalization.csv", index=False)
    return {"status": "PASS", "key": spec.key}


def select_phase1(config: Config) -> dict[str, Any]:
    rows = []
    for spec in phase1_specs():
        path = config.results_dir / "runs" / spec.key / "checkpoint.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.append({
            "case": spec.case,
            "seed": spec.seed,
            "lambda_class": spec.lambda_class,
            "lambda_history": spec.lambda_history,
            "val_ba": float(payload["best_val"]["ba"]),
            "best_epoch": int(payload["best_epoch"]),
            "key": spec.key,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(config.results_dir / "phase1_validation.csv", index=False)
    selection: dict[str, Any] = {}
    for case, field in (("C2_phase_cu", "lambda_class"), ("C3_history_delta", "lambda_history")):
        grouped = frame[frame.case == case].groupby(field)["val_ba"].agg(["mean", "std"]).reset_index()
        grouped = grouped.sort_values(["mean", field], ascending=[False, True])
        best = grouped.iloc[0]
        selection[case] = {
            "lambda": float(best[field]),
            "val_ba_mean": float(best["mean"]),
            "val_ba_std": float(best["std"]),
        }
    save_json(config.results_dir / "selection.json", selection)
    return selection


def finalize(config: Config) -> dict[str, Any]:
    selection = json.loads((config.results_dir / "selection.json").read_text())
    specs = phase1_specs() + phase2_specs(selection)
    native_rows, geometry, retrieval, history = [], [], [], []
    for spec in specs:
        directory = config.results_dir / "runs" / spec.key
        required = [
            "checkpoint.pt", "native.json", "cross_user_geometry.csv",
            "cross_user_retrieval.csv", "history_generalization.csv", "diagnostics.json",
        ]
        for name in required:
            if not (directory / name).exists():
                raise FileNotFoundError(directory / name)
        native = json.loads((directory / "native.json").read_text())
        native_rows.append({
            "case": spec.case, "seed": spec.seed,
            "lambda_class": spec.lambda_class, "lambda_history": spec.lambda_history,
            "train_ba": native["splits"]["train"]["ba"],
            "val_ba": native["splits"]["val"]["ba"],
            "test_ba": native["splits"]["test"]["ba"],
            "train_test_gap": native["train_test_gap"],
        })
        geometry.append(pd.read_csv(directory / "cross_user_geometry.csv"))
        retrieval.append(pd.read_csv(directory / "cross_user_retrieval.csv"))
        history.append(pd.read_csv(directory / "history_generalization.csv"))

    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    native_frame = pd.DataFrame(native_rows)
    native_frame.to_csv(aggregate / "native_runs.csv", index=False)
    pd.concat(geometry, ignore_index=True).to_csv(aggregate / "cross_user_geometry.csv", index=False)
    pd.concat(retrieval, ignore_index=True).to_csv(aggregate / "cross_user_retrieval.csv", index=False)
    pd.concat(history, ignore_index=True).to_csv(aggregate / "history_generalization.csv", index=False)
    native_frame.groupby(["case", "lambda_class", "lambda_history"])["test_ba"].agg(["mean", "std", "count"]).reset_index().to_csv(
        aggregate / "native_summary.csv", index=False
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "selection": selection,
        "run_count": len(specs),
        "phase1_count": len(phase1_specs()),
        "phase2_count": len(phase2_specs(selection)),
    }
    save_json(aggregate / "manifest.json", manifest)
    return manifest


def _spec_at(specs: list[LossSpec], task_id: int) -> LossSpec:
    if not 0 <= task_id < len(specs):
        raise ValueError(f"task-id must be in [0, {len(specs) - 1}]")
    return specs[task_id]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exp14 cross-user history-organization loss study")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    train1 = sub.add_parser("train-phase1")
    train1.add_argument("--task-id", type=int, required=True)
    sub.add_parser("select")
    train2 = sub.add_parser("train-phase2")
    train2.add_argument("--task-id", type=int, required=True)
    sub.add_parser("finalize")
    sub.add_parser("plan")
    args = parser.parse_args(argv)
    config = config_from_args(args)

    if args.command == "prepare":
        print(json.dumps(prepare(config), indent=2))
    elif args.command == "plan":
        print(json.dumps({
            "phase1": [asdict(spec) | {"key": spec.key} for spec in phase1_specs()],
            "phase1_count": len(phase1_specs()),
            "phase2_count": 6,
        }, indent=2))
    elif args.command == "train-phase1":
        print(json.dumps(train_one(config, _spec_at(phase1_specs(), args.task_id)), indent=2))
    elif args.command == "select":
        print(json.dumps(select_phase1(config), indent=2))
    elif args.command == "train-phase2":
        selection = json.loads((config.results_dir / "selection.json").read_text())
        print(json.dumps(train_one(config, _spec_at(phase2_specs(selection), args.task_id)), indent=2))
    elif args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))


if __name__ == "__main__":
    main()
