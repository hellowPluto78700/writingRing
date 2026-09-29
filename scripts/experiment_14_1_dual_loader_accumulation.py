from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from core_benchmark_v1.data import load_cache, loader
from core_benchmark_v1.diagnostics import run_diagnostics
from core_benchmark_v1.model import BenchmarkNet, mean_logits
from core_benchmark_v1.probes import fit_probe, run_probes, temporal_features
from core_benchmark_v1.protocol import Protocol, Run, digest, paired_seed
from core_benchmark_v1.storage import file_hash, save_json, save_torch
from core_benchmark_v1.training import cpu_state, evaluate_validation, extract, metrics
from scripts import experiment_14_history_organization as exp14


EXPERIMENT_ID = "experiment_14_1_dual_loader_accumulation"
PROTOCOL_VERSION = "dual_loader_accumulation_v1"
CORE_RESULTS_REL = Path("core_benchmark_v1/results/main")
SEEDS = (11, 23, 37)
SHIFTS = ((2, 3, 4), (2, 3, 4))
PHASE_LAMBDAS = (0.01, 0.03, 0.06, 0.10)
PREFIX_LAMBDAS = (0.10, 0.25, 0.50)
PHASES = (0.25, 0.50, 0.75, 1.00)
PREFIX_PHASES = (0.50, 0.75)
PREFIX_WEIGHTS = (0.5, 0.5)
WARMUP_EPOCHS = 10
RAMP_END_EPOCH = 30
VAL_NATIVE_TOL = 0.01
VAL_RELATIVE_TOL = 0.01
VAL_RETRIEVAL_TIE = 0.005
GRADIENT_DIAGNOSTIC_EPOCHS = (10, 20, 30, 50)
GRADIENT_DIAGNOSTIC_BATCHES = 4
TASK_BATCH_TRACE_COUNT = 3
AUX_CLASSES_PER_BATCH = 8
AUX_USERS_PER_CLASS = 8
AUX_SAMPLES_PER_USER = 1
AUX_BATCH_SIZE = AUX_CLASSES_PER_BATCH * AUX_USERS_PER_CLASS * AUX_SAMPLES_PER_USER


@dataclass(frozen=True)
class LossSpec:
    case: str
    seed: int
    lambda_phase: float = 0.0
    lambda_prefix: float = 0.0
    phase: int = 1

    @property
    def key(self) -> str:
        parts = [self.case, f"seed{self.seed}"]
        if self.lambda_phase:
            parts.append(f"lp{self.lambda_phase:g}")
        if self.lambda_prefix:
            parts.append(f"la{self.lambda_prefix:g}")
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
        specs.append(LossSpec("C0_dual_null", seed))
        specs.extend(LossSpec("P_phase_cu", seed, lambda_phase=value) for value in PHASE_LAMBDAS)
        specs.extend(LossSpec("A_prefix_wcce", seed, lambda_prefix=value) for value in PREFIX_LAMBDAS)
    return specs


def phase2_specs(selection: dict[str, Any]) -> list[LossSpec]:
    lp = float(selection["phase"]["lambda"])
    la = float(selection["prefix"]["lambda"])
    strengths = (
        ("J0_combined", lp, la),
        ("Jp_half_phase", 0.5 * lp, la),
        ("Ja_half_prefix", lp, 0.5 * la),
        ("Jb_half_both", 0.5 * lp, 0.5 * la),
    )
    return [
        LossSpec(case, seed, lambda_phase=phase_weight, lambda_prefix=prefix_weight, phase=2)
        for seed in SEEDS
        for case, phase_weight, prefix_weight in strengths
    ]


def final_eval_specs(selection: dict[str, Any]) -> list[LossSpec]:
    lp = float(selection["phase"]["lambda"])
    la = float(selection["prefix"]["lambda"])
    selected: list[LossSpec] = []
    for seed in SEEDS:
        selected.extend(
            (
                LossSpec("C0_dual_null", seed),
                LossSpec("P_phase_cu", seed, lambda_phase=lp),
                LossSpec("A_prefix_wcce", seed, lambda_prefix=la),
            )
        )
    selected.extend(phase2_specs(selection))
    if len({spec.key for spec in selected}) != len(selected):
        raise AssertionError("Final evaluation specs must be unique")
    return selected


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
    if (p.width, p.fs, p.input_channels, p.tau_mem_ms, p.batch_size) != (128, 64.0, 30, 22.54, 128):
        raise ValueError("Core geometry or task batch size no longer matches Exp14.1 contract")
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
        "phase_lambdas": list(PHASE_LAMBDAS),
        "prefix_lambdas": list(PREFIX_LAMBDAS),
        "phase_points": list(PHASES),
        "prefix_phases": list(PREFIX_PHASES),
        "prefix_weights": list(PREFIX_WEIGHTS),
        "gradient_diagnostic_batches": GRADIENT_DIAGNOSTIC_BATCHES,
        "task_batch_trace_count": TASK_BATCH_TRACE_COUNT,
        "warmup_epochs": WARMUP_EPOCHS,
        "ramp_end_epoch": RAMP_END_EPOCH,
        "task_loader": "CoreBenchmark shuffled DataLoader, batch_size=128",
        "aux_loader": {
            "classes_per_batch": AUX_CLASSES_PER_BATCH,
            "users_per_class": AUX_USERS_PER_CLASS,
            "samples_per_class_user": AUX_SAMPLES_PER_USER,
            "batch_size": AUX_BATCH_SIZE,
            "sampling_unit": "distinct user first, then one unique segment from that class-user cell",
        },
        "aux_sampler_validation": _validate_aux_sampler_schedule(arrays, p),
        "selection_tolerances": {
            "native_ba_absolute": VAL_NATIVE_TOL,
            "relative10_ba_absolute": VAL_RELATIVE_TOL,
            "retrieval_tie_absolute": VAL_RETRIEVAL_TIE,
        },
        "users": lock["data_metadata"]["users"],
        "phase1_runs": [asdict(spec) | {"key": spec.key} for spec in phase1_specs()],
        "phase1_count": len(phase1_specs()),
        "phase2_count": 12,
        "final_eval_count": 21,
        "train_samples": int(len(arrays["train_y"])),
    }
    payload["identity"] = digest(payload)
    save_json(config.results_dir / "protocol.json", payload)
    return payload


def _run(spec: LossSpec) -> Run:
    return Run(spec.case, spec.seed, "14_1_dual_loader_accumulation", shifts=SHIFTS, objective="wcce")


def _core_o0_run(seed: int) -> Run:
    return Run("O0", seed, "01_objective", objective="wcce")


def _aux_user_pools(arrays: dict[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    labels = arrays["train_y"]
    users = arrays["train_users"]
    pools: dict[int, dict[str, np.ndarray]] = {}
    for class_id in np.unique(labels).tolist():
        user_map: dict[str, np.ndarray] = {}
        for user in sorted(set(users[labels == class_id].tolist())):
            indices = np.flatnonzero((labels == class_id) & (users == user))
            if len(indices) > 0:
                user_map[str(user)] = indices
        pools[int(class_id)] = user_map
    return pools


def _aux_sampler_report(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    labels = arrays["train_y"]
    pools = _aux_user_pools(arrays)
    rows: list[dict[str, Any]] = []
    for class_id in sorted(pools):
        user_map = pools[class_id]
        counts = [len(indices) for indices in user_map.values()]
        row = {
            "class_id": int(class_id),
            "eligible_train_users": int(len(user_map)),
            "unique_train_samples": int((labels == class_id).sum()),
            "min_samples_per_eligible_user": int(min(counts)) if counts else 0,
            "max_samples_per_eligible_user": int(max(counts)) if counts else 0,
        }
        if len(user_map) < AUX_USERS_PER_CLASS:
            raise ValueError(
                f"class={class_id} has {len(user_map)} train users with >=1 sample; "
                f"need at least {AUX_USERS_PER_CLASS}"
            )
        rows.append(row)
    return rows


def _validated_aux_batches(
    arrays: dict[str, np.ndarray],
    seed: int,
    epoch: int,
    task_batch_size: int = 128,
) -> list[np.ndarray]:
    labels = arrays["train_y"]
    users = arrays["train_users"]
    classes = np.unique(labels)
    pools = _aux_user_pools(arrays)
    for class_id in classes.tolist():
        if len(pools[int(class_id)]) < AUX_USERS_PER_CLASS:
            raise ValueError(
                f"class={int(class_id)} has {len(pools[int(class_id)])} train users with >=1 sample; "
                f"need at least {AUX_USERS_PER_CLASS}"
            )

    rng = np.random.default_rng(paired_seed(seed, f"exp14_1:aux_sampler:{epoch}"))
    count = max(1, int(math.ceil(len(labels) / task_batch_size)))
    batches: list[np.ndarray] = []
    for _ in range(count):
        selected_classes = rng.choice(classes, size=AUX_CLASSES_PER_BATCH, replace=False)
        batch: list[int] = []
        for class_id in selected_classes.tolist():
            user_map = pools[int(class_id)]
            selected_users = rng.choice(
                np.asarray(sorted(user_map), dtype=object),
                size=AUX_USERS_PER_CLASS,
                replace=False,
            )
            for user in selected_users.tolist():
                pool = user_map[str(user)]
                chosen = int(rng.choice(pool))
                batch.append(chosen)
        indices = np.asarray(batch, dtype=np.int64)
        rng.shuffle(indices)

        if len(indices) != AUX_BATCH_SIZE:
            raise AssertionError(
                f"Auxiliary batch has {len(indices)} samples; expected {AUX_BATCH_SIZE}"
            )
        if len(np.unique(indices)) != len(indices):
            raise AssertionError("Auxiliary batch contains duplicate segment IDs")
        batch_y = labels[indices]
        batch_users = users[indices]
        unique_classes, class_counts = np.unique(batch_y, return_counts=True)
        if len(unique_classes) != AUX_CLASSES_PER_BATCH or not np.all(
            class_counts == AUX_USERS_PER_CLASS
        ):
            raise AssertionError("Auxiliary batch class geometry changed")
        for class_id in unique_classes.tolist():
            class_users = batch_users[batch_y == class_id]
            if len(np.unique(class_users)) != AUX_USERS_PER_CLASS:
                raise AssertionError(
                    f"class={int(class_id)} does not contain {AUX_USERS_PER_CLASS} distinct users"
                )
        for row, (label, user) in enumerate(zip(batch_y.tolist(), batch_users.tolist())):
            positive = (batch_y == label) & (batch_users != user)
            positive[row] = False
            if int(positive.sum()) != AUX_USERS_PER_CLASS - 1:
                raise AssertionError(
                    f"Auxiliary anchor expected {AUX_USERS_PER_CLASS - 1} cross-user positives; "
                    f"label={label} user={user} got {int(positive.sum())}"
                )
        batches.append(indices)
    return batches


def _validate_aux_sampler_schedule(
    arrays: dict[str, np.ndarray],
    p: Protocol,
) -> dict[str, Any]:
    report = _aux_sampler_report(arrays)
    checked = 0
    for seed in SEEDS:
        for epoch in range(0, p.max_epochs + 1):
            batches = _validated_aux_batches(
                arrays,
                seed,
                epoch,
                task_batch_size=p.batch_size,
            )
            expected_steps = max(1, int(math.ceil(len(arrays["train_y"]) / p.batch_size)))
            if len(batches) != expected_steps:
                raise AssertionError(
                    f"Auxiliary step count mismatch: {len(batches)} vs {expected_steps}"
                )
            checked += len(batches)
    return {
        "class_report": report,
        "validated_seed_epoch_pairs": int(len(SEEDS) * (p.max_epochs + 1)),
        "validated_aux_batches": int(checked),
    }


def _task_batch_trace(arrays: dict[str, np.ndarray], p: Protocol, seed: int) -> list[list[str]]:
    generator = torch.Generator().manual_seed(paired_seed(seed, "loader:train"))
    permutation = torch.randperm(len(arrays["train_y"]), generator=generator).tolist()
    rows: list[list[str]] = []
    for start in range(0, min(len(permutation), p.batch_size * TASK_BATCH_TRACE_COUNT), p.batch_size):
        indices = permutation[start:start + p.batch_size]
        rows.append([str(arrays["train_ids"][index]) for index in indices])
    return rows


def _needs_aux_batch(spec: LossSpec) -> bool:
    return spec.lambda_phase > 0.0


def _aux_weight(target: float, epoch: int) -> float:
    if target <= 0.0 or epoch <= WARMUP_EPOCHS:
        return 0.0
    if epoch >= RAMP_END_EPOCH:
        return float(target)
    return float(target) * (epoch - WARMUP_EPOCHS) / (RAMP_END_EPOCH - WARMUP_EPOCHS)


def _prefix_means(evidence: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if evidence.ndim != 3 or lengths.ndim != 1 or evidence.shape[0] != lengths.shape[0]:
        raise ValueError("Invalid evidence/length geometry")
    cumulative = evidence.cumsum(dim=1)
    batch = torch.arange(evidence.shape[0], device=evidence.device)
    means: list[torch.Tensor] = []
    for phase in PREFIX_PHASES:
        count = torch.ceil(lengths.to(torch.float32) * phase).to(torch.long).clamp_min(1)
        index = count.sub(1)
        summed = cumulative[batch, index]
        means.append(summed / count.to(evidence.dtype).unsqueeze(1))
    return means[0], means[1]


def prefix_wcce(evidence: torch.Tensor, lengths: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean50, mean75 = _prefix_means(evidence, lengths)
    return (
        PREFIX_WEIGHTS[0] * F.cross_entropy(mean50, y)
        + PREFIX_WEIGHTS[1] * F.cross_entropy(mean75, y)
    )


def _phase_cu_loss_from_tensors(
    model: BenchmarkNet,
    projector: exp14.ProjectionHead,
    x: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    users: torch.Tensor,
) -> torch.Tensor:
    trajectory = model(x, lengths)
    phase_values = exp14._phase_vectors(trajectory["pre_reset"][1], lengths)
    return torch.stack(
        [
            exp14.cross_user_supcon(projector(phase_values[:, index]), y, users)
            for index in range(phase_values.shape[1])
        ]
    ).mean()


def _user_ids(arrays: dict[str, np.ndarray]) -> np.ndarray:
    names = sorted(set(arrays["train_users"].tolist()))
    mapping = {str(name): index for index, name in enumerate(names)}
    return np.asarray([mapping[str(value)] for value in arrays["train_users"].tolist()], dtype=np.int64)


def _split_metrics(model: BenchmarkNet, arrays: dict[str, np.ndarray], p: Protocol, seed: int, split: str) -> dict[str, float]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for x, y, lengths in loader(arrays, split, p, seed):
            scores = mean_logits(model(x, lengths)["evidence"], lengths)
            targets.append(y.numpy())
            predictions.append(scores.argmax(1).numpy())
            losses.append(float(F.cross_entropy(scores, y, reduction="sum")))
    y_all = np.concatenate(targets)
    pred_all = np.concatenate(predictions)
    out = metrics(y_all, pred_all)
    out["mean_logit_ce"] = sum(losses) / len(y_all)
    return out


def _extract_splits(
    model: BenchmarkNet,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
    splits: tuple[str, ...],
) -> dict[str, np.ndarray]:
    model.eval()
    traces: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for split in splits:
            chunks: dict[str, list[np.ndarray]] = {"evidence": []}
            for layer in ("L1", "L2"):
                for state in ("spike", "pre_reset"):
                    chunks[f"{layer}__{state}"] = []
            for x, _, lengths in loader(arrays, split, p, seed):
                trajectory = model(x, lengths)
                chunks["evidence"].append(trajectory["evidence"].numpy().astype(np.float32))
                for li, layer in enumerate(("L1", "L2")):
                    chunks[f"{layer}__spike"].append(trajectory["spike"][li].numpy().astype(np.uint8))
                    chunks[f"{layer}__pre_reset"].append(trajectory["pre_reset"][li].numpy().astype(np.float32))
            for key, values in chunks.items():
                traces[f"{split}__{key}"] = np.concatenate(values, axis=0)
    return traces


def _prefix_evidence_vectors_numpy(
    evidence: np.ndarray,
    lengths: np.ndarray,
    phase: float,
) -> np.ndarray:
    count = np.ceil(lengths.astype(np.float64) * phase).astype(np.int64)
    count = np.maximum(count, 1)
    cumulative = np.cumsum(evidence, axis=1)
    vectors = cumulative[np.arange(len(lengths)), count - 1]
    return vectors / count[:, None]


def _native_prefix_diagnostics(
    evidence: np.ndarray,
    lengths: np.ndarray,
    y: np.ndarray,
    split: str,
) -> dict[str, Any]:
    phases = (0.50, 0.75, 1.00)
    vectors = {phase: _prefix_evidence_vectors_numpy(evidence, lengths, phase) for phase in phases}
    rows: list[dict[str, Any]] = []
    margins: dict[float, np.ndarray] = {}
    for phase in phases:
        vector = vectors[phase]
        prediction = vector.argmax(axis=1)
        correct = vector[np.arange(len(y)), y]
        masked = vector.copy()
        masked[np.arange(len(y)), y] = -np.inf
        margin = correct - masked.max(axis=1)
        margins[phase] = margin
        rows.append({
            "split": split,
            "phase": phase,
            **metrics(y, prediction),
            "mean_margin": float(np.mean(margin)),
            "median_margin": float(np.median(margin)),
        })

    def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        return np.divide(
            np.sum(left * right, axis=1),
            np.maximum(denom, 1e-12),
            out=np.zeros(len(left), dtype=np.float64),
            where=denom > 1e-12,
        )

    # True-class support relative to the mean competing logit is additive in time,
    # so segment signs and cancellation can be interpreted without changing competitors.
    class_count = evidence.shape[2]
    true = np.take_along_axis(evidence, y[:, None, None], axis=2).squeeze(2)
    other_mean = (evidence.sum(axis=2) - true) / max(class_count - 1, 1)
    support = true - other_mean
    segment_support = np.zeros((len(y), 3), dtype=np.float64)
    for index, length in enumerate(lengths.astype(int).tolist()):
        b50 = max(1, int(np.ceil(0.50 * length)))
        b75 = max(b50, int(np.ceil(0.75 * length)))
        segment_support[index, 0] = support[index, :b50].sum()
        segment_support[index, 1] = support[index, b50:b75].sum()
        segment_support[index, 2] = support[index, b75:length].sum()
    adjacent_product = segment_support[:, :-1] * segment_support[:, 1:]
    nonzero_adjacent = adjacent_product != 0
    reversal = adjacent_product < 0
    reversal_rate = float(reversal[nonzero_adjacent].mean()) if nonzero_adjacent.any() else 0.0
    cancellation = 1.0 - (
        np.abs(segment_support.sum(axis=1))
        / np.maximum(np.abs(segment_support).sum(axis=1), 1e-12)
    )
    summary = {
        "split": split,
        "cosine_50_75_mean": float(np.mean(cosine(vectors[0.50], vectors[0.75]))),
        "cosine_75_100_mean": float(np.mean(cosine(vectors[0.75], vectors[1.00]))),
        "monotonic_margin_fraction": float(
            ((margins[0.50] <= margins[0.75]) & (margins[0.75] <= margins[1.00])).mean()
        ),
        "support_sign_reversal_rate": reversal_rate,
        "support_cancellation_ratio_mean": float(np.mean(cancellation)),
        "support_cancellation_ratio_median": float(np.median(cancellation)),
    }
    return {"rows": rows, "summary": summary}


def _validation_retrieval(
    traces: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    state: str,
    layer: str = "L2",
) -> float:
    labels = np.unique(arrays["train_y"])
    train = exp14._resample(traces[f"train__{layer}__{state}"], arrays["train_lengths"])
    val = exp14._resample(traces[f"val__{layer}__{state}"], arrays["val_lengths"])
    prototypes = np.stack([train[arrays["train_y"] == label].mean(axis=0) for label in labels])
    prediction: list[int] = []
    for query in val:
        repeated = np.repeat(query[None], len(prototypes), axis=0)
        distance = exp14._trajectory_distance(repeated, prototypes)
        prediction.append(int(labels[int(np.argmin(distance))]))
    return float(metrics(arrays["val_y"], np.asarray(prediction))["ba"])


def _validation_probe(
    traces: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    p: Protocol,
    aggregation: str,
) -> dict[str, float]:
    features = {
        split: temporal_features(
            traces[f"{split}__L2__spike"],
            arrays[f"{split}_lengths"],
            arrays[f"{split}_ids"],
            aggregation,
            p,
        )
        for split in ("train", "val")
    }
    scaler, model, _ = fit_probe(
        features["train"],
        arrays["train_y"],
        features["val"],
        arrays["val_y"],
        "no_bias",
        p,
    )
    train_pred = model.predict(scaler.transform(features["train"].astype(np.float64)))
    val_pred = model.predict(scaler.transform(features["val"].astype(np.float64)))
    return {
        "train_ba": float(metrics(arrays["train_y"], train_pred)["ba"]),
        "val_ba": float(metrics(arrays["val_y"], val_pred)["ba"]),
        "C": float(model.C),
    }


def _validation_artifacts(
    config: Config,
    spec: LossSpec,
    model: BenchmarkNet,
    p: Protocol,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    traces = _extract_splits(model, arrays, p, spec.seed, ("train", "val"))
    native_train = _split_metrics(model, arrays, p, spec.seed, "train")
    native_val = _split_metrics(model, arrays, p, spec.seed, "val")
    retrieval_spike = _validation_retrieval(traces, arrays, "spike")
    retrieval_pre = _validation_retrieval(traces, arrays, "pre_reset")
    whole = _validation_probe(traces, arrays, p, "whole_count")
    relative = _validation_probe(traces, arrays, p, "relative10_ordered")
    fixed250 = _validation_probe(traces, arrays, p, "fixed250_ordered")
    val_prefix = _native_prefix_diagnostics(
        traces["val__evidence"], arrays["val_lengths"], arrays["val_y"], "val"
    )
    prefix_by_phase = {float(row["phase"]): row for row in val_prefix["rows"]}
    collapse_gap_pp = 100.0 * (relative["val_ba"] - whole["val_ba"])
    summary = {
        "case": spec.case,
        "seed": spec.seed,
        "lambda_phase": spec.lambda_phase,
        "lambda_prefix": spec.lambda_prefix,
        "native_train_ba": native_train["ba"],
        "native_val_ba": native_val["ba"],
        "native_val_ce": native_val["mean_logit_ce"],
        "val_retrieval_l2_spike": retrieval_spike,
        "val_retrieval_l2_pre_reset": retrieval_pre,
        "val_wholecount_no_bias_ba": whole["val_ba"],
        "val_relative10_no_bias_ba": relative["val_ba"],
        "val_fixed250_no_bias_ba": fixed250["val_ba"],
        "val_native_prefix50_ba": prefix_by_phase[0.50]["ba"],
        "val_native_prefix75_ba": prefix_by_phase[0.75]["ba"],
        "val_native_prefix100_ba": prefix_by_phase[1.00]["ba"],
        "val_collapse_gap_pp": collapse_gap_pp,
    }
    save_json(config.results_dir / "runs" / spec.key / "validation_summary.json", summary)
    return summary


def _gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.detach().reshape(-1)
    right_flat = right.detach().reshape(-1)
    denom = float(left_flat.norm() * right_flat.norm())
    if denom <= 1e-20:
        return float("nan")
    return float(torch.dot(left_flat, right_flat) / denom)


def _mean_gradient(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("Cannot average an empty gradient list")
    return torch.stack([value.detach() for value in values], dim=0).mean(dim=0)


def _gradient_diagnostic(
    model: BenchmarkNet,
    projector: exp14.ProjectionHead,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    spec: LossSpec,
    epoch: int,
    label: str,
) -> dict[str, Any]:
    was_training = model.training
    projector_was_training = projector.training
    model.eval()
    projector.eval()
    parameter = model.layers[1].weight

    task_wcce: list[torch.Tensor] = []
    task_prefix: list[torch.Tensor] = []
    for batch_index, (x, y, lengths) in enumerate(
        loader(arrays, "train", p, spec.seed, shuffle=True)
    ):
        if batch_index >= GRADIENT_DIAGNOSTIC_BATCHES:
            break
        trajectory = model(x, lengths)
        wcce = F.cross_entropy(mean_logits(trajectory["evidence"], lengths), y)
        prefix = prefix_wcce(trajectory["evidence"], lengths, y)
        task_wcce.append(torch.autograd.grad(wcce, parameter, retain_graph=True)[0])
        task_prefix.append(torch.autograd.grad(prefix, parameter)[0])

    phase_grads: list[torch.Tensor] = []
    if _needs_aux_batch(spec):
        user_ids = _user_ids(arrays)
        aux_batches = _validated_aux_batches(
            arrays,
            spec.seed,
            epoch=0,
            task_batch_size=p.batch_size,
        )
        for indices in aux_batches[:GRADIENT_DIAGNOSTIC_BATCHES]:
            x = torch.from_numpy(arrays["train_x"][indices])
            y = torch.from_numpy(arrays["train_y"][indices])
            lengths = torch.from_numpy(arrays["train_lengths"][indices])
            users = torch.from_numpy(user_ids[indices])
            phase = _phase_cu_loss_from_tensors(model, projector, x, y, lengths, users)
            phase_grads.append(torch.autograd.grad(phase, parameter)[0])

    g_w = _mean_gradient(task_wcce)
    g_a = _mean_gradient(task_prefix)
    g_p = _mean_gradient(phase_grads) if phase_grads else None
    wp = _aux_weight(spec.lambda_phase, epoch)
    wa = _aux_weight(spec.lambda_prefix, epoch)
    norm_w = float(g_w.norm())
    norm_p = float(g_p.norm()) if g_p is not None else float("nan")
    norm_a = float(g_a.norm())
    if was_training:
        model.train()
    if projector_was_training:
        projector.train()
    return {
        "case": spec.case,
        "seed": spec.seed,
        "lambda_phase": spec.lambda_phase,
        "lambda_prefix": spec.lambda_prefix,
        "epoch": epoch,
        "label": label,
        "diagnostic_task_batches": len(task_wcce),
        "diagnostic_aux_batches": len(phase_grads),
        "wcce_grad_norm_l2": norm_w,
        "phase_grad_norm_l2": norm_p,
        "prefix_grad_norm_l2": norm_a,
        "weighted_phase_over_wcce": (wp * norm_p / norm_w) if norm_w > 0 else float("nan"),
        "weighted_prefix_over_wcce": (wa * norm_a / norm_w) if norm_w > 0 else float("nan"),
        "cos_phase_prefix": _gradient_cosine(g_p, g_a) if g_p is not None else float("nan"),
        "cos_phase_wcce": _gradient_cosine(g_p, g_w) if g_p is not None else float("nan"),
        "cos_prefix_wcce": _gradient_cosine(g_a, g_w),
    }

def _verify_c0_reference(config: Config, spec: LossSpec, best_state: dict[str, torch.Tensor], best_epoch: int) -> None:
    if spec.case != "C0_dual_null":
        return
    path = config.core_results_dir / "runs" / f"O0__seed{spec.seed}" / "checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen Core O0 reference: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    reference = payload["model_state_dict"]
    if reference.keys() != best_state.keys():
        raise AssertionError("C0/Core O0 state keys differ")
    mismatched = [key for key in reference if not torch.equal(reference[key], best_state[key])]
    if mismatched:
        raise AssertionError(f"C0 failed bitwise Core O0 reproduction: {mismatched[:5]}")
    if int(payload["best_epoch"]) != int(best_epoch):
        raise AssertionError(
            f"C0/Core O0 best epoch differs: Exp14.1={best_epoch}, Core={payload['best_epoch']}"
        )


def train_one(config: Config, spec: LossSpec) -> dict[str, Any]:
    p, core_lock, arrays = _load_core(config)
    directory = config.results_dir / "runs" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    if checkpoint_path.exists() and (directory / "validation_summary.json").exists():
        return {"status": "exists", "key": spec.key}

    model = BenchmarkNet(_run(spec), p)
    projector = exp14.ProjectionHead(spec.seed, p.width)
    initial = cpu_state(model)
    params = list(model.parameters())
    if spec.lambda_phase > 0:
        params += list(projector.parameters())
    optimizer = torch.optim.Adam(params, lr=p.learning_rate, weight_decay=p.weight_decay)
    train_loader = loader(arrays, "train", p, spec.seed, shuffle=True)
    user_ids = _user_ids(arrays)
    save_json(
        directory / "task_batch_trace.json",
        {
            "source": "independent replay of the frozen Core RandomSampler seed",
            "first_epoch_batch_ids": _task_batch_trace(arrays, p, spec.seed),
        },
    )

    best = evaluate_validation(model, arrays, p, spec.seed)
    best_state = initial
    best_projector = cpu_state(projector)
    best_epoch = 0
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train_total_loss": None,
            "train_wcce": None,
            "train_prefix_loss": None,
            "train_phase_loss": None,
            "phase_weight": 0.0,
            "prefix_weight": 0.0,
            "val_ba": best["ba"],
            "val_mean_logit_ce": best["mean_logit_ce"],
        }
    ]
    gradient_rows: list[dict[str, Any]] = []

    for epoch in range(1, p.max_epochs + 1):
        model.train()
        projector.train()
        aux_batches = (
            _validated_aux_batches(arrays, spec.seed, epoch, task_batch_size=p.batch_size)
            if _needs_aux_batch(spec)
            else None
        )
        if aux_batches is not None and len(aux_batches) != len(train_loader):
            raise AssertionError(
                f"Task/aux step mismatch: {len(train_loader)} vs {len(aux_batches)}"
            )
        totals = {"loss": 0.0, "wcce": 0.0, "prefix": 0.0, "phase": 0.0, "n": 0}
        wp = _aux_weight(spec.lambda_phase, epoch)
        wa = _aux_weight(spec.lambda_prefix, epoch)
        for step, (x, y, lengths) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            trajectory = model(x, lengths)
            wcce = F.cross_entropy(mean_logits(trajectory["evidence"], lengths), y)
            total_loss = wcce
            prefix_loss = wcce.new_tensor(0.0)
            phase_loss = wcce.new_tensor(0.0)
            if wa > 0.0:
                prefix_loss = prefix_wcce(trajectory["evidence"], lengths, y)
                total_loss = total_loss + wa * prefix_loss
            if wp > 0.0:
                if aux_batches is None:
                    raise AssertionError("Phase-CU weight is nonzero without an auxiliary batch")
                aux_indices = aux_batches[step]
                aux_x = torch.from_numpy(arrays["train_x"][aux_indices])
                aux_y = torch.from_numpy(arrays["train_y"][aux_indices])
                aux_lengths = torch.from_numpy(arrays["train_lengths"][aux_indices])
                aux_users = torch.from_numpy(user_ids[aux_indices])
                phase_loss = _phase_cu_loss_from_tensors(
                    model, projector, aux_x, aux_y, aux_lengths, aux_users
                )
                total_loss = total_loss + wp * phase_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"{spec.key}: nonfinite loss at epoch {epoch}")
            total_loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in params
            ):
                raise FloatingPointError(f"{spec.key}: nonfinite gradient at epoch {epoch}")
            optimizer.step()
            n = len(y)
            totals["loss"] += float(total_loss.detach()) * n
            totals["wcce"] += float(wcce.detach()) * n
            totals["prefix"] += float(prefix_loss.detach()) * n
            totals["phase"] += float(phase_loss.detach()) * n
            totals["n"] += n

        val = evaluate_validation(model, arrays, p, spec.seed)
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": totals["loss"] / totals["n"],
                "train_wcce": totals["wcce"] / totals["n"],
                "train_prefix_loss": totals["prefix"] / totals["n"],
                "train_phase_loss": totals["phase"] / totals["n"],
                "phase_weight": wp,
                "prefix_weight": wa,
                "val_ba": val["ba"],
                "val_mean_logit_ce": val["mean_logit_ce"],
            }
        )
        improved = val["ba"] > best["ba"] + 1e-12 or (
            abs(val["ba"] - best["ba"]) <= 1e-12
            and val["mean_logit_ce"] < best["mean_logit_ce"] - 1e-12
        )
        if improved:
            best = val
            best_state = cpu_state(model)
            best_projector = cpu_state(projector)
            best_epoch = epoch

        if epoch in GRADIENT_DIAGNOSTIC_EPOCHS:
            gradient_rows.append(
                _gradient_diagnostic(model, projector, arrays, p, spec, epoch, f"epoch{epoch}")
            )
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    model.load_state_dict(best_state)
    projector.load_state_dict(best_projector)
    _verify_c0_reference(config, spec, best_state, best_epoch)
    gradient_rows.append(
        _gradient_diagnostic(model, projector, arrays, p, spec, best_epoch, "selected_checkpoint")
    )
    pd.DataFrame(gradient_rows).to_csv(directory / "gradient_diagnostics.csv", index=False)
    save_json(directory / "history.json", {"rows": history})
    save_torch(
        checkpoint_path,
        {
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
            "c0_core_o0_bitwise_match": spec.case == "C0_dual_null",
        },
    )
    validation = _validation_artifacts(config, spec, model, p, arrays)
    return {
        "status": "PASS",
        "key": spec.key,
        "best_val_ba": best["ba"],
        "validation": validation,
    }


def _load_model(config: Config, spec: LossSpec) -> BenchmarkNet:
    p, _, _ = _load_core(config)
    payload = torch.load(
        config.results_dir / "runs" / spec.key / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    if payload["spec"] != asdict(spec):
        raise ValueError(f"Checkpoint spec mismatch: {spec.key}")
    model = BenchmarkNet(_run(spec), p)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def _read_validation_summary(config: Config, spec: LossSpec) -> dict[str, Any]:
    path = config.results_dir / "runs" / spec.key / "validation_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def select_phase1(config: Config) -> dict[str, Any]:
    rows = [_read_validation_summary(config, spec) | {"key": spec.key} for spec in phase1_specs()]
    frame = pd.DataFrame(rows)
    frame.to_csv(config.results_dir / "phase1_validation.csv", index=False)
    c0 = frame[frame.case == "C0_dual_null"]
    if len(c0) != len(SEEDS):
        raise AssertionError("Missing C0 validation rows")
    c0_native = float(c0["native_val_ba"].mean())
    c0_relative = float(c0["val_relative10_no_bias_ba"].mean())

    phase = (
        frame[frame.case == "P_phase_cu"]
        .groupby("lambda_phase", as_index=False)
        .agg(
            native_val_ba=("native_val_ba", "mean"),
            retrieval=("val_retrieval_l2_spike", "mean"),
            retrieval_std=("val_retrieval_l2_spike", "std"),
        )
    )
    phase["eligible_native"] = phase["native_val_ba"] >= c0_native - VAL_NATIVE_TOL
    eligible_phase = phase[phase["eligible_native"]].copy()
    if eligible_phase.empty:
        raise RuntimeError("No phase lambda satisfies the native validation BA constraint")
    best_retrieval = float(eligible_phase["retrieval"].max())
    eligible_phase = eligible_phase[
        eligible_phase["retrieval"] >= best_retrieval - VAL_RETRIEVAL_TIE
    ].sort_values("lambda_phase")
    phase_choice = eligible_phase.iloc[0]

    prefix = (
        frame[frame.case == "A_prefix_wcce"]
        .groupby("lambda_prefix", as_index=False)
        .agg(
            native_val_ba=("native_val_ba", "mean"),
            wholecount=("val_wholecount_no_bias_ba", "mean"),
            relative10=("val_relative10_no_bias_ba", "mean"),
            collapse_gap_pp=("val_collapse_gap_pp", "mean"),
        )
    )
    prefix["eligible_native"] = prefix["native_val_ba"] >= c0_native - VAL_NATIVE_TOL
    prefix["eligible_relative10"] = (
        prefix["relative10"] >= c0_relative - VAL_RELATIVE_TOL
    )
    eligible_prefix = prefix[prefix["eligible_native"] & prefix["eligible_relative10"]].copy()
    if eligible_prefix.empty:
        raise RuntimeError("No prefix lambda satisfies native/relative10 validation constraints")
    best_whole = float(eligible_prefix["wholecount"].max())
    eligible_prefix = eligible_prefix[
        np.isclose(eligible_prefix["wholecount"], best_whole, rtol=0.0, atol=1e-12)
    ].sort_values(["collapse_gap_pp", "lambda_prefix"])
    prefix_choice = eligible_prefix.iloc[0]

    phase.to_csv(config.results_dir / "phase_selection_candidates.csv", index=False)
    prefix.to_csv(config.results_dir / "prefix_selection_candidates.csv", index=False)
    selection = {
        "c0_validation": {
            "native_ba_mean": c0_native,
            "relative10_no_bias_ba_mean": c0_relative,
        },
        "phase": {
            "lambda": float(phase_choice["lambda_phase"]),
            "native_val_ba_mean": float(phase_choice["native_val_ba"]),
            "retrieval_l2_spike_mean": float(phase_choice["retrieval"]),
            "rule": "native BA >= C0-1pp; maximize L2 spike retrieval; within 0.5pp choose smaller lambda",
        },
        "prefix": {
            "lambda": float(prefix_choice["lambda_prefix"]),
            "native_val_ba_mean": float(prefix_choice["native_val_ba"]),
            "wholecount_no_bias_ba_mean": float(prefix_choice["wholecount"]),
            "relative10_no_bias_ba_mean": float(prefix_choice["relative10"]),
            "collapse_gap_pp_mean": float(prefix_choice["collapse_gap_pp"]),
            "rule": "native BA >= C0-1pp and relative10 >= C0-1pp; maximize wholecount; tie by smaller collapse gap then lambda",
        },
    }
    save_json(config.results_dir / "selection.json", selection)
    return selection


def evaluate_final(config: Config, spec: LossSpec) -> dict[str, Any]:
    p, _, arrays = _load_core(config)
    directory = config.results_dir / "runs" / spec.key
    checkpoint = directory / "checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    done = directory / "final_evaluated.json"
    required = (
        "native.json",
        "probes.json",
        "diagnostics.json",
        "cross_user_geometry.csv",
        "cross_user_retrieval.csv",
        "history_generalization.csv",
    )
    if done.exists() and all((directory / name).exists() for name in required):
        return {"status": "exists", "key": spec.key}
    model = _load_model(config, spec)
    model.eval()
    traces, native = extract(model, arrays, p, _run(spec))
    save_json(directory / "native.json", native)
    run_probes(directory, traces, arrays, _run(spec), p)
    diagnostic_protocol = Protocol.from_dict(
        {
            **asdict(p),
            "diagnostic_cases": [spec.case],
            "history_ms": list(exp14.HISTORY_EVAL_MS),
        }
    )
    run_diagnostics(directory, model, traces, arrays, _run(spec), diagnostic_protocol)
    pd.DataFrame(exp14._geometry(traces, arrays, spec)).to_csv(
        directory / "cross_user_geometry.csv", index=False
    )
    pd.DataFrame(exp14._retrieval(traces, arrays, spec)).to_csv(
        directory / "cross_user_retrieval.csv", index=False
    )
    pd.DataFrame(exp14._history_generalization(model, arrays, spec)).to_csv(
        directory / "history_generalization.csv", index=False
    )
    prefix_payload = {
        split: _native_prefix_diagnostics(
            traces[f"{split}__evidence"],
            arrays[f"{split}_lengths"],
            arrays[f"{split}_y"],
            split,
        )
        for split in ("train", "val", "test")
    }
    save_json(directory / "prefix_evidence_diagnostics.json", prefix_payload)
    save_json(done, {"status": "PASS", "key": spec.key})
    return {"status": "PASS", "key": spec.key}


def _primary_probe_rows(directory: Path, spec: LossSpec) -> list[dict[str, Any]]:
    payload = json.loads((directory / "probes.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    wanted = {"whole_count", "fixed250_ordered", "relative10_ordered"}
    for row in payload["rows"]:
        if (
            row["layer"] == "L2"
            and row["state"] == "spike"
            and row["decoder"] == "no_bias"
            and row["aggregation"] in wanted
            and int(row["shuffle_seed"]) == -1
        ):
            rows.append(
                {
                    "case": spec.case,
                    "seed": spec.seed,
                    "lambda_phase": spec.lambda_phase,
                    "lambda_prefix": spec.lambda_prefix,
                    **row,
                }
            )
    if {row["aggregation"] for row in rows} != wanted:
        raise AssertionError(f"Missing primary probes for {spec.key}")
    return rows


def finalize(config: Config) -> dict[str, Any]:
    selection = json.loads((config.results_dir / "selection.json").read_text(encoding="utf-8"))
    specs = final_eval_specs(selection)
    native_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    collapse_rows: list[dict[str, Any]] = []
    geometry: list[pd.DataFrame] = []
    retrieval: list[pd.DataFrame] = []
    history: list[pd.DataFrame] = []
    gradient: list[pd.DataFrame] = []
    prefix_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    for spec in specs:
        directory = config.results_dir / "runs" / spec.key
        required = (
            "checkpoint.pt",
            "native.json",
            "probes.json",
            "cross_user_geometry.csv",
            "cross_user_retrieval.csv",
            "history_generalization.csv",
            "gradient_diagnostics.csv",
            "prefix_evidence_diagnostics.json",
            "final_evaluated.json",
        )
        for name in required:
            if not (directory / name).exists():
                raise FileNotFoundError(directory / name)
        native = json.loads((directory / "native.json").read_text(encoding="utf-8"))
        native_rows.append(
            {
                "case": spec.case,
                "seed": spec.seed,
                "lambda_phase": spec.lambda_phase,
                "lambda_prefix": spec.lambda_prefix,
                "train_ba": native["splits"]["train"]["ba"],
                "val_ba": native["splits"]["val"]["ba"],
                "test_ba": native["splits"]["test"]["ba"],
                "test_accuracy": native["splits"]["test"]["accuracy"],
                "test_macro_f1": native["splits"]["test"]["macro_f1"],
                "train_test_gap": native["train_test_gap"],
            }
        )
        current_probes = _primary_probe_rows(directory, spec)
        probe_rows.extend(current_probes)
        by_aggregation = {row["aggregation"]: row for row in current_probes}
        collapse_rows.append(
            {
                "case": spec.case,
                "seed": spec.seed,
                "lambda_phase": spec.lambda_phase,
                "lambda_prefix": spec.lambda_prefix,
                "val_wholecount_ba": by_aggregation["whole_count"]["val_ba"],
                "val_relative10_ba": by_aggregation["relative10_ordered"]["val_ba"],
                "val_collapse_gap_pp": 100.0
                * (
                    by_aggregation["relative10_ordered"]["val_ba"]
                    - by_aggregation["whole_count"]["val_ba"]
                ),
                "test_wholecount_ba": by_aggregation["whole_count"]["test_ba"],
                "test_relative10_ba": by_aggregation["relative10_ordered"]["test_ba"],
                "test_collapse_gap_pp": 100.0
                * (
                    by_aggregation["relative10_ordered"]["test_ba"]
                    - by_aggregation["whole_count"]["test_ba"]
                ),
            }
        )
        geometry.append(pd.read_csv(directory / "cross_user_geometry.csv"))
        retrieval.append(pd.read_csv(directory / "cross_user_retrieval.csv"))
        history.append(pd.read_csv(directory / "history_generalization.csv"))
        grad = pd.read_csv(directory / "gradient_diagnostics.csv")
        grad["run_key"] = spec.key
        gradient.append(grad)
        prefix_payload = json.loads(
            (directory / "prefix_evidence_diagnostics.json").read_text(encoding="utf-8")
        )
        for split, payload in prefix_payload.items():
            for row in payload["rows"]:
                prefix_rows.append({
                    "case": spec.case,
                    "seed": spec.seed,
                    "lambda_phase": spec.lambda_phase,
                    "lambda_prefix": spec.lambda_prefix,
                    **row,
                })
            evidence_rows.append({
                "case": spec.case,
                "seed": spec.seed,
                "lambda_phase": spec.lambda_phase,
                "lambda_prefix": spec.lambda_prefix,
                **payload["summary"],
            })

    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    native_frame = pd.DataFrame(native_rows)
    probe_frame = pd.DataFrame(probe_rows)
    collapse_frame = pd.DataFrame(collapse_rows)
    retrieval_frame = pd.concat(retrieval, ignore_index=True)
    native_frame.to_csv(aggregate / "native_runs.csv", index=False)
    probe_frame.to_csv(aggregate / "probe_summary.csv", index=False)
    collapse_frame.to_csv(aggregate / "collapse_gap.csv", index=False)
    pd.concat(geometry, ignore_index=True).to_csv(
        aggregate / "cross_user_geometry.csv", index=False
    )
    retrieval_frame.to_csv(aggregate / "cross_user_retrieval.csv", index=False)
    pd.concat(history, ignore_index=True).to_csv(
        aggregate / "history_generalization.csv", index=False
    )
    pd.concat(gradient, ignore_index=True).to_csv(
        aggregate / "gradient_diagnostics.csv", index=False
    )
    pd.DataFrame(prefix_rows).to_csv(
        aggregate / "native_prefix_metrics.csv", index=False
    )
    pd.DataFrame(evidence_rows).to_csv(
        aggregate / "evidence_organization_diagnostics.csv", index=False
    )
    native_summary = (
        native_frame.groupby(["case", "lambda_phase", "lambda_prefix"])["test_ba"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    native_summary.to_csv(aggregate / "native_summary.csv", index=False)
    collapse_summary = (
        collapse_frame.groupby(["case", "lambda_phase", "lambda_prefix"])[
            ["test_wholecount_ba", "test_relative10_ba", "test_collapse_gap_pp"]
        ]
        .mean()
        .reset_index()
    )
    collapse_summary.to_csv(aggregate / "collapse_gap_summary.csv", index=False)

    retrieval_l2_spike = retrieval_frame[
        (retrieval_frame["state"] == "spike") & (retrieval_frame["layer"] == "L2")
    ][["case", "seed", "retrieval_ba"]]
    paired = native_frame[["case", "seed", "test_ba"]].merge(
        collapse_frame[
            ["case", "seed", "test_wholecount_ba", "test_relative10_ba", "test_collapse_gap_pp"]
        ],
        on=["case", "seed"],
        how="inner",
    ).merge(retrieval_l2_spike, on=["case", "seed"], how="inner")
    baseline = paired[paired.case == "C0_dual_null"].set_index("seed")
    delta_rows: list[dict[str, Any]] = []
    for row in paired.itertuples(index=False):
        ref = baseline.loc[row.seed]
        delta_rows.append(
            {
                "case": row.case,
                "seed": row.seed,
                "native_test_ba_delta_pp": 100.0 * (row.test_ba - ref.test_ba),
                "retrieval_delta_pp": 100.0 * (row.retrieval_ba - ref.retrieval_ba),
                "wholecount_delta_pp": 100.0
                * (row.test_wholecount_ba - ref.test_wholecount_ba),
                "relative10_delta_pp": 100.0
                * (row.test_relative10_ba - ref.test_relative10_ba),
                "collapse_gap_delta_pp": row.test_collapse_gap_pp - ref.test_collapse_gap_pp,
            }
        )
    pd.DataFrame(delta_rows).to_csv(aggregate / "paired_delta_vs_c0.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "selection": selection,
        "phase1_count": len(phase1_specs()),
        "phase2_count": len(phase2_specs(selection)),
        "final_eval_count": len(specs),
        "final_cases": sorted({spec.case for spec in specs}),
    }
    save_json(aggregate / "manifest.json", manifest)
    return manifest


def _spec_at(specs: list[LossSpec], task_id: int) -> LossSpec:
    if not 0 <= task_id < len(specs):
        raise ValueError(f"task-id must be in [0, {len(specs) - 1}]")
    return specs[task_id]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Exp14.1 dual-loader phase alignment + Prefix-WCCE accumulation study"
    )
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    train1 = sub.add_parser("train-phase1")
    train1.add_argument("--task-id", type=int, required=True)
    sub.add_parser("select")
    train2 = sub.add_parser("train-phase2")
    train2.add_argument("--task-id", type=int, required=True)
    evaluate = sub.add_parser("eval-final")
    evaluate.add_argument("--task-id", type=int, required=True)
    sub.add_parser("finalize")
    sub.add_parser("plan")
    args = parser.parse_args(argv)
    config = config_from_args(args)

    if args.command == "prepare":
        print(json.dumps(prepare(config), indent=2))
    elif args.command == "plan":
        print(
            json.dumps(
                {
                    "phase1": [asdict(spec) | {"key": spec.key} for spec in phase1_specs()],
                    "phase1_count": len(phase1_specs()),
                    "phase2_count": 12,
                    "final_eval_count": 21,
                },
                indent=2,
            )
        )
    elif args.command == "train-phase1":
        print(json.dumps(train_one(config, _spec_at(phase1_specs(), args.task_id)), indent=2))
    elif args.command == "select":
        print(json.dumps(select_phase1(config), indent=2))
    elif args.command == "train-phase2":
        selection = json.loads((config.results_dir / "selection.json").read_text(encoding="utf-8"))
        print(
            json.dumps(
                train_one(config, _spec_at(phase2_specs(selection), args.task_id)),
                indent=2,
            )
        )
    elif args.command == "eval-final":
        selection = json.loads((config.results_dir / "selection.json").read_text(encoding="utf-8"))
        print(
            json.dumps(
                evaluate_final(config, _spec_at(final_eval_specs(selection), args.task_id)),
                indent=2,
            )
        )
    elif args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))


if __name__ == "__main__":
    main()
