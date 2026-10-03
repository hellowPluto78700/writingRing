#!/usr/bin/env python3
"""Exp18.2: factor history carrier and loss geometry under the locked Exp18 contract."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from core_benchmark_v1.data import loader
from core_benchmark_v1.model import mean_logits, valid_sum
from core_benchmark_v1.probes import run_probes, temporal_features
from core_benchmark_v1.protocol import Protocol, Run, SPLITS
from core_benchmark_v1.storage import file_hash, load_torch, save_json, save_npz, save_torch, state_hash
from core_benchmark_v1.training import cpu_state
from scripts import experiment_18_membrane_history as exp18
from scripts import experiment_18_1_layerwise_memory_carrier as exp181

EXPERIMENT_ID = "experiment_18_2_loss_geometry"
PROTOCOL_VERSION = "loss_geometry_v1"
SEEDS = (11, 23, 37)
FORMAL_CASES = ("I_NWCCE", "I_MWCCE", "U_NWCCE", "U_MWCCE")
REFERENCE_CASES = ("I_WCCE_REF", "U_WCCE_REF")
ALL_CASES = ("I_WCCE_REF", "I_NWCCE", "I_MWCCE", "U_WCCE_REF", "U_NWCCE", "U_MWCCE")
SHIFTS = ((2, 3, 4), (2, 3, 4))
CARRIERS = {
    "I_WCCE_REF": ("I", "I"),
    "I_NWCCE": ("I", "I"),
    "I_MWCCE": ("I", "I"),
    "U_WCCE_REF": ("U", "U"),
    "U_NWCCE": ("U", "U"),
    "U_MWCCE": ("U", "U"),
}
LOSS_KINDS = {
    "I_WCCE_REF": "wcce",
    "I_NWCCE": "nwcce",
    "I_MWCCE": "mwcce",
    "U_WCCE_REF": "wcce",
    "U_NWCCE": "nwcce",
    "U_MWCCE": "mwcce",
}
MARGIN_TARGET = 1.0
PRIMARY_AGGREGATIONS = (
    "whole_count",
    "fixed250_ordered",
    "fixed250_shuffled",
    "relative10_ordered",
    "relative10_shuffled",
)
RATE_QUANTILES = (0.50, 0.75, 0.90, 0.95)
EXP18_RESULTS_REL = Path("notebooks/artifacts/experiment_18_membrane_history/membrane_history_v1")


@dataclass(frozen=True)
class ExpSpec:
    case: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.case}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    core_results_dir: Path
    exp18_results_dir: Path


def find_repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in (path, *path.parents):
        if (parent / "AGENTS.md").exists() and (parent / "core_benchmark_v1").exists():
            return parent
    raise FileNotFoundError("Repository root not found")


def default_results_dir(root: Path) -> Path:
    return root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def config_from_args(args: argparse.Namespace) -> Config:
    root = find_repo_root()
    return Config(
        repo_root=root,
        results_dir=(args.results or default_results_dir(root)).resolve(),
        core_results_dir=(args.core_results or root / exp18.exp16.CORE_RESULTS_REL).resolve(),
        exp18_results_dir=(args.exp18_results or root / EXP18_RESULTS_REL).resolve(),
    )


def specs() -> list[ExpSpec]:
    return [ExpSpec(case, seed) for seed in SEEDS for case in FORMAL_CASES]


def _run(spec: ExpSpec) -> Run:
    return Run(spec.case, spec.seed, "18_2_loss_geometry", shifts=SHIFTS, objective="wcce")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hash() -> str:
    return file_hash(Path(__file__).resolve())


def _dependency_hashes() -> dict[str, str]:
    return {
        "exp18_source_hash": file_hash(Path(exp18.__file__).resolve()),
        "exp18_1_source_hash": file_hash(Path(exp181.__file__).resolve()),
    }


def _as_exp181_config(config: Config) -> exp181.Config:
    return exp181.Config(
        repo_root=config.repo_root,
        results_dir=config.results_dir,
        core_results_dir=config.core_results_dir,
        exp18_results_dir=config.exp18_results_dir,
    )


def _core(config: Config) -> tuple[Protocol, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    p, lock, arrays, exp18_refs = exp181._core(_as_exp181_config(config))
    if tuple(p.seeds) != SEEDS or p.width != 128 or tuple(tuple(v) for v in SHIFTS) != SHIFTS:
        raise ValueError("Exp18.2 requires the locked CoreBenchmark/Exp18 seed and architecture contract")
    if p.wcce_reduction != "valid_mean_logits" or p.native_bias:
        raise ValueError("Exp18.2 requires bias-free valid-mean native WCCE")
    return p, lock, arrays, exp18_refs


def _reference_dir(config: Config, case: str, seed: int, p: Protocol) -> Path:
    if case == "I_WCCE_REF":
        return config.core_results_dir / "runs" / exp18._o0_run(seed, p).key
    if case == "U_WCCE_REF":
        return config.exp18_results_dir / "runs" / f"U_NORMAL__seed{seed}"
    raise ValueError(case)


def _model(spec: ExpSpec, p: Protocol) -> exp181.HybridMemoryNet:
    return exp181.HybridMemoryNet(
        exp181.ExpSpec(spec.case, spec.seed),
        p,
        carriers=CARRIERS[spec.case],
    )


def _checkpoint_dir(config: Config, spec: ExpSpec) -> Path:
    return config.results_dir / "runs" / spec.key


def _protocol_lock(config: Config) -> dict[str, Any]:
    path = config.results_dir / "protocol.lock.json"
    if not path.is_file():
        raise FileNotFoundError("Run Exp18.2 prepare before formal training")
    lock = _read_json(path)
    if lock.get("source_hash") != _source_hash():
        raise ValueError("Exp18.2 source changed after prepare; use a new results directory or rerun prepare")
    if lock.get("dependency_hashes") != _dependency_hashes():
        raise ValueError("Exp18/Exp18.1 dependency source changed after Exp18.2 prepare")
    return lock


def _run_provenance(config: Config, spec: ExpSpec, core_identity: str) -> dict[str, Any]:
    lock = _protocol_lock(config)
    return {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "source_hash": lock["source_hash"],
        "dependency_hashes": lock["dependency_hashes"],
        "core_identity": core_identity,
        "case": spec.case,
        "seed": spec.seed,
        "carriers": list(CARRIERS[spec.case]),
        "loss_kind": LOSS_KINDS[spec.case],
    }


def _validate_checkpoint(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Checkpoint provenance mismatch for {key}: {payload.get(key)!r} != {value!r}")


def _validate_complete(directory: Path, expected: dict[str, Any]) -> bool:
    path = directory / "complete.json"
    if not path.is_file():
        return False
    complete = _read_json(path)
    if complete.get("status") != "PASS":
        raise ValueError(f"Incomplete run: {directory.name}")
    for key, value in expected.items():
        if complete.get(key) != value:
            raise ValueError(f"Completion provenance mismatch for {directory.name}: {key}")
    for name, digest in complete.get("files", {}).items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"Changed completed artifact: {directory / name}")
    return True


def normalized_counts(counts: torch.Tensor) -> torch.Tensor:
    """Exact L1 count normalization for nonzero samples; zero-count samples stay zero."""
    if counts.ndim != 2 or bool((counts < 0).any()):
        raise ValueError("Normalized-WCCE expects nonnegative [batch, neuron] spike counts")
    total = counts.sum(1, keepdim=True)
    denominator = torch.where(total > 0, total, torch.ones_like(total))
    return counts / denominator


def normalized_logits(
    model: exp181.HybridMemoryNet,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    counts = valid_sum(trajectory["spike"][-1], lengths)
    return float(gamma) * model.head(normalized_counts(counts))


def margin_wcce_loss(logits: torch.Tensor, y: torch.Tensor, margin_target: float = MARGIN_TARGET) -> torch.Tensor:
    if logits.ndim != 2 or y.shape != (logits.shape[0],):
        raise ValueError("Invalid margin-WCCE logits/target geometry")
    target = logits.gather(1, y[:, None]).squeeze(1)
    wrong = logits.masked_fill(F.one_hot(y, logits.shape[1]).bool(), -torch.inf).max(1).values
    margin = target - wrong
    return F.relu(float(margin_target) - margin).mean()


def training_loss(
    model: exp181.HybridMemoryNet,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
    y: torch.Tensor,
    loss_config: dict[str, Any],
) -> torch.Tensor:
    kind = loss_config["loss_kind"]
    if kind == "nwcce":
        return F.cross_entropy(normalized_logits(model, trajectory, lengths, float(loss_config["gamma"])), y)
    if kind == "mwcce":
        return margin_wcce_loss(mean_logits(trajectory["evidence"], lengths), y, float(loss_config["margin_target"]))
    if kind == "wcce":
        return F.cross_entropy(mean_logits(trajectory["evidence"], lengths), y)
    raise ValueError(kind)


@torch.no_grad()
def calibrate_normalized_gamma(
    model: exp181.HybridMemoryNet,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
) -> dict[str, float | str]:
    """Label-free epoch-0 logit-RMS matching on the locked training split."""
    model.eval()
    raw_ss = 0.0
    normalized_ss = 0.0
    n_logits = 0
    nonzero_samples = 0
    total_samples = 0
    for x, _y, lengths in loader(arrays, "train", p, seed, shuffle=False):
        trajectory = model(x, lengths)
        raw = mean_logits(trajectory["evidence"], lengths)
        counts = valid_sum(trajectory["spike"][-1], lengths)
        normalized = model.head(normalized_counts(counts))
        raw_ss += float(raw.square().sum())
        normalized_ss += float(normalized.square().sum())
        n_logits += raw.numel()
        nonzero_samples += int((counts.sum(1) > 0).sum())
        total_samples += len(x)
    raw_rms = math.sqrt(raw_ss / max(n_logits, 1))
    normalized_rms = math.sqrt(normalized_ss / max(n_logits, 1))
    if not math.isfinite(raw_rms) or not math.isfinite(normalized_rms) or raw_rms <= 0 or normalized_rms <= 0:
        raise FloatingPointError(
            f"Cannot calibrate normalized-WCCE scale: raw_rms={raw_rms}, normalized_rms={normalized_rms}"
        )
    gamma = raw_rms / normalized_rms
    if not math.isfinite(gamma) or gamma <= 0:
        raise FloatingPointError(f"Invalid normalized-WCCE gamma: {gamma}")
    return {
        "loss_kind": "nwcce",
        "gamma": gamma,
        "calibration": "epoch0_train_label_free_logit_rms",
        "raw_logit_rms": raw_rms,
        "normalized_logit_rms": normalized_rms,
        "nonzero_sample_fraction": nonzero_samples / max(total_samples, 1),
    }


def _make_loss_config(
    model: exp181.HybridMemoryNet,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    spec: ExpSpec,
) -> dict[str, Any]:
    kind = LOSS_KINDS[spec.case]
    if kind == "nwcce":
        return dict(calibrate_normalized_gamma(model, arrays, p, spec.seed))
    if kind == "mwcce":
        return {"loss_kind": "mwcce", "margin_target": MARGIN_TARGET}
    return {"loss_kind": "wcce"}


def _split_eval(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
    split: str,
) -> dict[str, float]:
    return exp18._split_eval(model, arrays, p, seed, split)


def train(config: Config, spec: ExpSpec) -> tuple[exp181.HybridMemoryNet, dict[str, Any]]:
    p, core_lock, arrays, _ = _core(config)
    expected = _run_provenance(config, spec, core_lock["identity"])
    directory = _checkpoint_dir(config, spec)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    loss_config_path = directory / "loss_config.json"

    if checkpoint.is_file():
        payload = load_torch(checkpoint)
        _validate_checkpoint(payload, expected)
        if not loss_config_path.is_file():
            raise FileNotFoundError(f"Missing loss config for resumable checkpoint: {loss_config_path}")
        loss_config = _read_json(loss_config_path)
        if payload.get("loss_config") != loss_config:
            raise ValueError("Checkpoint/loss-config mismatch")
        model = _model(spec, p)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return model, loss_config

    model = _model(spec, p)
    loss_config = _make_loss_config(model, arrays, p, spec)
    save_json(loss_config_path, loss_config)
    initial = cpu_state(model)
    save_torch(directory / "initial.pt", {
        **expected,
        "loss_config": loss_config,
        "model_state_dict": initial,
    })

    optimizer = torch.optim.Adam(model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
    best = _split_eval(model, arrays, p, spec.seed, "val")
    best_state, best_epoch = initial, 0
    history: list[dict[str, Any]] = [{
        "epoch": 0,
        "train_loss": None,
        "val_ba": best["ba"],
        "val_mean_logit_ce": best["mean_logit_ce"],
    }]
    train_loader = loader(arrays, "train", p, spec.seed, shuffle=True)
    stopped_epoch = 0

    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            trajectory = model(x, lengths)
            loss = training_loss(model, trajectory, lengths, y, loss_config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite loss")
            loss.backward()
            if any(param.grad is not None and not torch.isfinite(param.grad).all() for param in model.parameters()):
                raise FloatingPointError(f"{spec.key}: nonfinite gradient")
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)

        val = _split_eval(model, arrays, p, spec.seed, "val")
        history.append({
            "epoch": epoch,
            "train_loss": total / count,
            "val_ba": val["ba"],
            "val_mean_logit_ce": val["mean_logit_ce"],
        })
        if exp18._better(val, best):
            best, best_state, best_epoch = dict(val), cpu_state(model), epoch
        stopped_epoch = epoch
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    model.load_state_dict(best_state, strict=True)
    save_json(directory / "history.json", {"rows": history})
    save_torch(checkpoint, {
        **expected,
        "loss_config": loss_config,
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val": best,
        "selection_rule": "native validation BA, then native validation mean-logit CE, then earliest epoch",
    })
    return model, loss_config


def _group_bounds(width: int) -> list[tuple[int, int]]:
    q, r = divmod(width, 3)
    result = []
    start = 0
    for gi in range(3):
        stop = start + q + (gi < r)
        result.append((start, stop))
        start = stop
    return result


def _rate_summary(
    traces: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    p: Protocol,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_valid_steps = float(np.asarray(arrays[f"{split}_lengths"], dtype=np.float64).sum())
    for li in range(2):
        z = traces[f"{split}__L{li+1}__spike"].astype(np.float64)
        rates = p.fs * z.sum(axis=(0, 1)) / max(total_valid_steps, 1.0)
        for gi, (a, b) in enumerate(_group_bounds(p.width)):
            selected = rates[a:b]
            row: dict[str, Any] = {
                "split": split,
                "layer": f"L{li+1}",
                "shift": SHIFTS[li][gi],
                "mean_firing_hz": float(selected.mean()),
                "fraction_ge_10hz": float((selected >= 10.0).mean()),
                "fraction_ge_20hz": float((selected >= 20.0).mean()),
            }
            for q in RATE_QUANTILES:
                row[f"p{int(100*q)}_firing_hz"] = float(np.quantile(selected, q))
            rows.append(row)
    return rows


def _count_summary(
    traces: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for li in range(2):
        z = traces[f"{split}__L{li+1}__spike"].astype(np.float64)
        counts = z.sum(1)
        total = counts.sum(1)
        l2 = np.sqrt(np.square(counts).sum(1))
        rows.append({
            "split": split,
            "layer": f"L{li+1}",
            "mean_total_count": float(total.mean()),
            "mean_count_l2": float(l2.mean()),
            "p95_total_count": float(np.quantile(total, 0.95)),
        })
    return rows


def _parameter_norms_from_state(state: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "L1_weight_fro": float(state["layers.0.weight"].float().norm()),
        "L2_weight_fro": float(state["layers.1.weight"].float().norm()),
        "head_weight_fro": float(state["head.weight"].float().norm()),
    }


def _augment_activity(
    activity: dict[str, Any],
    traces: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    p: Protocol,
    model: exp181.HybridMemoryNet,
) -> dict[str, Any]:
    lookup = {
        (row["split"], row["layer"], row["shift"]): row
        for split in SPLITS
        for row in _rate_summary(traces, arrays, p, split)
    }
    for row in activity["rows"]:
        extra = lookup[(row["split"], row["layer"], row["shift"])]
        row.update({key: value for key, value in extra.items() if key not in ("split", "layer", "shift")})
    activity["count_summary"] = [
        row
        for split in SPLITS
        for row in _count_summary(traces, arrays, split)
    ]
    activity["parameter_norms"] = _parameter_norms_from_state(cpu_state(model))
    return activity


def evaluate(config: Config, spec: ExpSpec) -> dict[str, Any]:
    p, core_lock, arrays, _ = _core(config)
    expected = _run_provenance(config, spec, core_lock["identity"])
    directory = _checkpoint_dir(config, spec)
    payload = load_torch(directory / "checkpoint.pt")
    _validate_checkpoint(payload, expected)
    model = _model(spec, p)
    model.load_state_dict(payload["model_state_dict"], strict=True)

    checkpoint_hash = file_hash(directory / "checkpoint.pt")
    traces, native, activity = exp181.extract(
        model,
        arrays,
        p,
        exp181.ExpSpec(spec.case, spec.seed),
    )
    activity = _augment_activity(activity, traces, arrays, p, model)
    save_npz(directory / "traces.npz", traces)
    save_json(directory / "native.json", native)
    save_json(directory / "activity.json", activity)
    run_probes(directory, traces, arrays, _run(spec), p)
    if file_hash(directory / "checkpoint.pt") != checkpoint_hash:
        raise AssertionError("Evaluation modified the selected checkpoint")
    return native


def smoke(config: Config) -> dict[str, Any]:
    p, core_lock, arrays, _ = _core(config)
    batch = next(iter(loader(arrays, "train", p, SEEDS[0], shuffle=False)))
    x, y, lengths = batch
    x, y, lengths = x[: min(8, len(x))], y[: min(8, len(y))], lengths[: min(8, len(lengths))]

    gamma_by_carrier: dict[str, dict[str, Any]] = {}
    for carrier, case in (("I", "I_NWCCE"), ("U", "U_NWCCE")):
        model = _model(ExpSpec(case, SEEDS[0]), p)
        gamma_by_carrier[carrier] = calibrate_normalized_gamma(model, arrays, p, SEEDS[0])

    results = []
    for case in FORMAL_CASES:
        spec = ExpSpec(case, SEEDS[0])
        model = _model(spec, p)
        optimizer = torch.optim.Adam(model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
        loss_config = (
            gamma_by_carrier[CARRIERS[case][0]]
            if LOSS_KINDS[case] == "nwcce"
            else {"loss_kind": "mwcce", "margin_target": MARGIN_TARGET}
        )
        before = cpu_state(model)
        trajectory = model(x, lengths)
        loss = training_loss(model, trajectory, lengths, y, loss_config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss) or any(param.grad is not None and not torch.isfinite(param.grad).all() for param in model.parameters()):
            raise FloatingPointError(f"{case}: nonfinite smoke loss/gradient")
        optimizer.step()
        after = cpu_state(model)
        if state_hash(before) == state_hash(after):
            raise AssertionError(f"{case}: optimizer step did not change parameters")
        feature = temporal_features(
            trajectory["spike"][1].detach().numpy(),
            lengths.numpy(),
            arrays["train_ids"][: len(x)],
            "whole_count",
            p,
        )
        if feature.shape != (len(x), p.width):
            raise AssertionError(f"{case}: canonical probe feature path returned {feature.shape}")
        with tempfile.TemporaryDirectory(prefix="exp18_2_smoke_") as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_torch(path, {"model_state_dict": after})
            restored = _model(spec, p)
            restored.load_state_dict(load_torch(path)["model_state_dict"], strict=True)
            with torch.no_grad():
                logits = mean_logits(restored(x, lengths)["evidence"], lengths)
            if logits.shape != (len(x), len(p.labels)) or not torch.isfinite(logits).all():
                raise AssertionError(f"{case}: checkpoint/evaluation roundtrip failed")
        results.append({
            "case": case,
            "carriers": list(CARRIERS[case]),
            "loss_kind": LOSS_KINDS[case],
            "loss": float(loss.detach()),
        })

    return {
        "status": "PASS",
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "core_identity": core_lock["identity"],
        "gamma_calibration": gamma_by_carrier,
        "cases": results,
    }


def prepare(config: Config) -> dict[str, Any]:
    p, core_lock, _, exp18_refs = _core(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    references: dict[str, dict[str, Any]] = {}
    for case in REFERENCE_CASES:
        references[case] = {}
        for seed in SEEDS:
            directory = _reference_dir(config, case, seed, p)
            required = ("checkpoint.pt", "native.json", "probes.json", "traces.npz")
            for name in required:
                if not (directory / name).is_file():
                    raise FileNotFoundError(f"Missing reference artifact: {directory / name}")
            references[case][str(seed)] = {
                "directory": str(directory),
                **{f"{name}_hash": file_hash(directory / name) for name in required},
            }

    payload = {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "source_hash": _source_hash(),
        "dependency_hashes": _dependency_hashes(),
        "core_identity": core_lock["identity"],
        "core_dataset_hash": core_lock["dataset_hash"],
        "core_protocol_hash": core_lock["protocol_hash"],
        "exp18_protocol_hash": exp18_refs["protocol_hash"],
        "seeds": list(SEEDS),
        "formal_cases": list(FORMAL_CASES),
        "reference_cases": list(REFERENCE_CASES),
        "all_cases": list(ALL_CASES),
        "carriers": {case: list(CARRIERS[case]) for case in ALL_CASES},
        "loss_kinds": LOSS_KINDS,
        "margin_target": MARGIN_TARGET,
        "normalized_scale_calibration": "per-carrier-per-seed epoch0 train label-free logit RMS",
        "references": references,
        "recommended_concurrency": 12,
    }
    path = config.results_dir / "protocol.lock.json"
    if path.is_file():
        previous = _read_json(path)
        if previous != payload:
            raise ValueError("Existing Exp18.2 protocol lock differs; use a new results directory")
        return previous
    save_json(path, payload)
    return payload


def run_one(config: Config, spec: ExpSpec) -> dict[str, Any]:
    p, core_lock, _, _ = _core(config)
    expected = _run_provenance(config, spec, core_lock["identity"])
    directory = _checkpoint_dir(config, spec)
    if _validate_complete(directory, expected):
        return {"run": spec.key, "status": "already_complete"}

    train(config, spec)
    native = evaluate(config, spec)
    required = (
        "loss_config.json",
        "initial.pt",
        "checkpoint.pt",
        "history.json",
        "native.json",
        "activity.json",
        "traces.npz",
        "probes.json",
        "probe_search.json",
        "probe_decoders.npz",
        "probe_predictions.npz",
    )
    save_json(directory / "complete.json", {
        **expected,
        "status": "PASS",
        "files": {name: file_hash(directory / name) for name in required},
    })
    return {"run": spec.key, "status": "PASS", "test_ba": native["splits"]["test"]["ba"]}


def _primary_probe(directory: Path) -> dict[str, float]:
    rows = _read_json(directory / "probes.json")["rows"]
    result: dict[str, float] = {}
    for aggregation in PRIMARY_AGGREGATIONS:
        selected = [
            row for row in rows
            if row["layer"] == "L2"
            and row["state"] == "spike"
            and row["decoder"] == "no_bias"
            and row["aggregation"] == aggregation
        ]
        if not selected:
            raise ValueError(f"Missing L2 spike no-bias probe {aggregation}: {directory}")
        result[aggregation] = float(np.mean([row["test_ba"] for row in selected]))
    return result


def _load_traces(directory: Path) -> dict[str, np.ndarray]:
    with np.load(directory / "traces.npz", allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _checkpoint_state(directory: Path) -> dict[str, torch.Tensor]:
    payload = load_torch(directory / "checkpoint.pt")
    return payload["model_state_dict"]


def _row_from_directory(
    directory: Path,
    case: str,
    seed: int,
    arrays: dict[str, np.ndarray],
    p: Protocol,
) -> dict[str, Any]:
    native = _read_json(directory / "native.json")
    probe = _primary_probe(directory)
    traces = _load_traces(directory)
    rates = _rate_summary(traces, arrays, p, "test")
    l2_rates = np.concatenate([
        p.fs
        * traces["test__L2__spike"][:, :, a:b].astype(np.float64).sum(axis=(0, 1))
        / max(float(np.asarray(arrays["test_lengths"], dtype=np.float64).sum()), 1.0)
        for a, b in _group_bounds(p.width)
    ])
    counts = traces["test__L2__spike"].astype(np.float64).sum(1)
    norms = _parameter_norms_from_state(_checkpoint_state(directory))
    row: dict[str, Any] = {
        "case": case,
        "seed": seed,
        "carrier": CARRIERS[case][0],
        "loss_kind": LOSS_KINDS[case],
        **{
            f"{split}_{name}": value
            for split in SPLITS
            for name, value in native["splits"][split].items()
            if isinstance(value, (int, float))
        },
        **{f"l2_{key}_test_ba": value for key, value in probe.items()},
        **norms,
        "l2_mean_firing_hz": float(l2_rates.mean()),
        "l2_p90_firing_hz": float(np.quantile(l2_rates, 0.90)),
        "l2_p95_firing_hz": float(np.quantile(l2_rates, 0.95)),
        "l2_fraction_ge_10hz": float((l2_rates >= 10.0).mean()),
        "l2_fraction_ge_20hz": float((l2_rates >= 20.0).mean()),
        "l2_mean_total_count": float(counts.sum(1).mean()),
        "l2_mean_count_l2": float(np.sqrt(np.square(counts).sum(1)).mean()),
    }
    row["fixed250_order_gap"] = probe["fixed250_ordered"] - probe["fixed250_shuffled"]
    row["relative10_order_gap"] = probe["relative10_ordered"] - probe["relative10_shuffled"]
    row["relative_native_gap"] = probe["relative10_ordered"] - row["test_ba"]
    row["fixed_native_gap"] = probe["fixed250_ordered"] - row["test_ba"]
    row["test_rate_group_summary"] = rates
    return row


def finalize(config: Config) -> dict[str, Any]:
    p, core_lock, arrays, _ = _core(config)
    protocol = _protocol_lock(config)
    rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        for case in ALL_CASES:
            if case in REFERENCE_CASES:
                directory = _reference_dir(config, case, seed, p)
                locked = protocol["references"][case][str(seed)]
                for name in ("checkpoint.pt", "native.json", "probes.json", "traces.npz"):
                    if file_hash(directory / name) != locked[f"{name}_hash"]:
                        raise ValueError(f"Changed reference artifact: {case} seed {seed} {name}")
            else:
                spec = ExpSpec(case, seed)
                directory = _checkpoint_dir(config, spec)
                expected = _run_provenance(config, spec, core_lock["identity"])
                if not _validate_complete(directory, expected):
                    raise FileNotFoundError(f"Missing complete Exp18.2 run: {case} seed {seed}")
            rows.append(_row_from_directory(directory, case, seed, arrays, p))

    metric_names = (
        "test_ba",
        "l2_mean_firing_hz",
        "l2_p95_firing_hz",
        "l2_fraction_ge_10hz",
        "l2_fraction_ge_20hz",
        "l2_whole_count_test_ba",
        "l2_fixed250_ordered_test_ba",
        "l2_relative10_ordered_test_ba",
        "relative_native_gap",
        "fixed250_order_gap",
        "relative10_order_gap",
        "head_weight_fro",
        "L1_weight_fro",
        "L2_weight_fro",
        "l2_mean_total_count",
        "l2_mean_count_l2",
    )
    paired: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for seed in SEEDS:
        by_case = {row["case"]: row for row in rows if row["seed"] == seed}
        for carrier in ("I", "U"):
            ref = by_case[f"{carrier}_WCCE_REF"]
            for loss_name in ("NWCCE", "MWCCE"):
                cur = by_case[f"{carrier}_{loss_name}"]
                for metric in metric_names:
                    paired.append({
                        "seed": seed,
                        "carrier": carrier,
                        "loss": loss_name,
                        "metric": metric,
                        "delta_vs_wcce": cur[metric] - ref[metric],
                    })
        for loss_name in ("NWCCE", "MWCCE"):
            for metric in metric_names:
                i_delta = by_case[f"I_{loss_name}"][metric] - by_case["I_WCCE_REF"][metric]
                u_delta = by_case[f"U_{loss_name}"][metric] - by_case["U_WCCE_REF"][metric]
                interactions.append({
                    "seed": seed,
                    "loss": loss_name,
                    "metric": metric,
                    "I_delta_vs_wcce": i_delta,
                    "U_delta_vs_wcce": u_delta,
                    "carrier_by_loss_interaction": u_delta - i_delta,
                })

    paired_summary = []
    for carrier in ("I", "U"):
        for loss_name in ("NWCCE", "MWCCE"):
            for metric in metric_names:
                selected = [
                    row["delta_vs_wcce"]
                    for row in paired
                    if row["carrier"] == carrier and row["loss"] == loss_name and row["metric"] == metric
                ]
                paired_summary.append({
                    "carrier": carrier,
                    "loss": loss_name,
                    "metric": metric,
                    "mean_delta_vs_wcce": float(np.mean(selected)),
                    "sd_delta_vs_wcce": float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0,
                })

    interaction_summary = []
    for loss_name in ("NWCCE", "MWCCE"):
        for metric in metric_names:
            selected = [
                row["carrier_by_loss_interaction"]
                for row in interactions
                if row["loss"] == loss_name and row["metric"] == metric
            ]
            interaction_summary.append({
                "loss": loss_name,
                "metric": metric,
                "mean_carrier_by_loss_interaction": float(np.mean(selected)),
            })

    payload = {
        "status": "PASS",
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "core_identity": core_lock["identity"],
        "rows": rows,
        "paired_contrasts": paired,
        "paired_summary": paired_summary,
        "carrier_by_loss_interactions": interactions,
        "interaction_summary": interaction_summary,
    }
    save_json(config.results_dir / "aggregate.json", payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    parser.add_argument("--exp18-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    sub.add_parser("prepare")
    run_p = sub.add_parser("run")
    run_p.add_argument("--case", choices=FORMAL_CASES, required=True)
    run_p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    sub.add_parser("finalize")
    args = parser.parse_args(argv)
    config = config_from_args(args)

    if args.command == "smoke":
        result = smoke(config)
    elif args.command == "prepare":
        result = prepare(config)
    elif args.command == "run":
        result = run_one(config, ExpSpec(args.case, args.seed))
    else:
        result = finalize(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
