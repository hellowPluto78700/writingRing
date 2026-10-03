#!/usr/bin/env python3
"""Exp18.1: localize the useful memory carrier by assigning I/U dynamics per hidden layer."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from core_benchmark_v1.data import loader
from core_benchmark_v1.model import BenchmarkNet, BinarySpike, mean_logits, valid_mask
from core_benchmark_v1.probes import run_probes, temporal_features
from core_benchmark_v1.protocol import Protocol, Run, SPLITS, paired_seed
from core_benchmark_v1.storage import file_hash, load_torch, save_json, save_npz, save_torch, state_hash
from core_benchmark_v1.training import cpu_state, metrics
from scripts import experiment_18_membrane_history as exp18

EXPERIMENT_ID = "experiment_18_1_layerwise_memory_carrier"
PROTOCOL_VERSION = "layerwise_memory_carrier_v1"
SEEDS = (11, 23, 37)
CASES = ("UI", "IU")
REFERENCE_CASES = ("II_REF", "UU_REF")
SHIFTS = ((2, 3, 4), (2, 3, 4))
CARRIERS = {"UI": ("U", "I"), "IU": ("I", "U"), "II_REF": ("I", "I"), "UU_REF": ("U", "U")}
EXP18_RESULTS_REL = Path("notebooks/artifacts/experiment_18_membrane_history/membrane_history_v1")
PRIMARY_AGGREGATIONS = (
    "whole_count", "fixed250_ordered", "fixed250_shuffled",
    "relative10_ordered", "relative10_shuffled",
)


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
    return [ExpSpec(case, seed) for seed in SEEDS for case in CASES]


def _run(spec: ExpSpec) -> Run:
    return Run(spec.case, spec.seed, "18_1_layerwise_memory_carrier", shifts=SHIFTS, objective="wcce")


def _slow_values(shifts: tuple[int, ...], width: int) -> torch.Tensor:
    q, r = divmod(width, len(shifts))
    return torch.tensor(
        [1.0 - 2.0 ** (-shift) for j, shift in enumerate(shifts) for _ in range(q + (j < r))],
        dtype=torch.float32,
    )


def _source_hash() -> str:
    return file_hash(Path(__file__).resolve())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_exp18_reference(config: Config, p: Protocol, core_lock: dict[str, Any]) -> dict[str, Any]:
    lock_path = config.exp18_results_dir / "protocol.lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing Exp18 protocol lock: {lock_path}")
    lock = _read_json(lock_path)
    if lock.get("core_identity") != core_lock["identity"]:
        raise ValueError("Exp18 reference uses a different CoreBenchmark identity")
    references: dict[str, Any] = {}
    for seed in SEEDS:
        directory = config.exp18_results_dir / "runs" / f"U_NORMAL__seed{seed}"
        complete = _read_json(directory / "complete.json")
        if complete.get("status") != "PASS":
            raise ValueError(f"Exp18 U_NORMAL seed {seed} is not complete")
        for name, expected in complete.get("files", {}).items():
            if file_hash(directory / name) != expected:
                raise ValueError(f"Changed Exp18 reference artifact: {directory / name}")
        references[str(seed)] = {
            "directory": str(directory),
            "checkpoint_hash": file_hash(directory / "checkpoint.pt"),
            "native_hash": file_hash(directory / "native.json"),
            "probes_hash": file_hash(directory / "probes.json"),
        }
    return {"protocol_hash": file_hash(lock_path), "runs": references}


def _core(config: Config) -> tuple[Protocol, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    p, lock, arrays = exp18._core(config)
    if tuple(p.seeds) != SEEDS or tuple(tuple(v) for v in SHIFTS) != SHIFTS:
        raise ValueError("CoreBenchmark seed/shift contract changed")
    refs = _validate_exp18_reference(config, p, lock)
    return p, lock, arrays, refs


class HybridMemoryNet(nn.Module):
    """Two-layer paired SNN whose slow pole can live in I or U independently per layer."""

    def __init__(self, spec: ExpSpec, p: Protocol, carriers: tuple[str, str] | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.protocol = p
        self.carriers = carriers or CARRIERS[spec.case]
        if len(self.carriers) != 2 or any(value not in ("I", "U") for value in self.carriers):
            raise ValueError(f"Invalid carrier assignment: {self.carriers}")
        self.layers = nn.ModuleList(
            nn.Linear(p.input_channels if i == 0 else p.width, p.width, bias=False)
            for i in range(2)
        )
        self.head = nn.Linear(p.width, len(p.labels), bias=False)
        self.fast = math.exp(-(1000.0 / p.fs) / p.tau_mem_ms)
        for li, shifts in enumerate(SHIFTS):
            self.register_buffer(f"slow_{li}", _slow_values(shifts, p.width))
        for name, parameter in self.named_parameters():
            generator = torch.Generator().manual_seed(paired_seed(spec.seed, f"init:{name}"))
            bound = 1.0 / math.sqrt(parameter.shape[1])
            with torch.no_grad():
                parameter.uniform_(-bound, bound, generator=generator)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, Any]:
        p = self.protocol
        batch, steps, channels = x.shape
        if channels != p.input_channels or lengths.shape != (batch,) or (lengths < 1).any() or (lengths > steps).any():
            raise ValueError("Invalid input/valid-length geometry")
        syn = [x.new_zeros(batch, p.width) for _ in range(2)]
        mem = [x.new_zeros(batch, p.width) for _ in range(2)]
        spikes: list[list[torch.Tensor]] = [[], []]
        pre_reset: list[list[torch.Tensor]] = [[], []]
        evidence: list[torch.Tensor] = []
        for t in range(int(lengths.max().item())):
            active = (t < lengths)[:, None]
            cur = x[:, t] * active
            for li, linear in enumerate(self.layers):
                slow = getattr(self, f"slow_{li}")
                syn_decay = slow if self.carriers[li] == "I" else self.fast
                mem_decay = self.fast if self.carriers[li] == "I" else slow
                candidate = syn_decay * syn[li] + linear(cur)
                syn[li] = torch.where(active, candidate, syn[li])
                pre = mem_decay * mem[li] + syn[li]
                spike = BinarySpike.apply(pre, p.threshold, p.surrogate_slope)
                mem[li] = torch.where(active, pre - p.threshold * spike, mem[li])
                cur = spike * active
                spikes[li].append(cur)
                pre_reset[li].append(pre * active)
            evidence.append(self.head(cur))

        def stack(values: list[torch.Tensor]) -> torch.Tensor:
            result = torch.stack(values, dim=1)
            return F.pad(result, (0, 0, 0, steps - result.shape[1]))

        return {
            "spike": tuple(stack(values) for values in spikes),
            "pre_reset": tuple(stack(values) for values in pre_reset),
            "evidence": stack(evidence),
            "final_syn": tuple(syn),
            "final_mem": tuple(mem),
        }


def _checkpoint_dir(config: Config, spec: ExpSpec) -> Path:
    return config.results_dir / "runs" / spec.key


def _protocol_lock(config: Config) -> dict[str, Any]:
    path = config.results_dir / "protocol.lock.json"
    if not path.is_file():
        raise FileNotFoundError("Run Exp18.1 prepare before training")
    lock = _read_json(path)
    if lock.get("source_hash") != _source_hash():
        raise ValueError("Exp18.1 source changed after prepare; use a new results directory or rerun prepare")
    return lock


def _run_provenance(config: Config, spec: ExpSpec, core_identity: str) -> dict[str, Any]:
    lock = _protocol_lock(config)
    return {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "source_hash": lock["source_hash"],
        "core_identity": core_identity,
        "case": spec.case,
        "seed": spec.seed,
        "carriers": list(CARRIERS[spec.case]),
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


def _split_eval(model: nn.Module, arrays: dict[str, np.ndarray], p: Protocol, seed: int, split: str) -> dict[str, float]:
    return exp18._split_eval(model, arrays, p, seed, split)


def train(config: Config, spec: ExpSpec) -> HybridMemoryNet:
    p, core_lock, arrays, _ = _core(config)
    expected = _run_provenance(config, spec, core_lock["identity"])
    directory = _checkpoint_dir(config, spec)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.is_file():
        payload = load_torch(checkpoint)
        _validate_checkpoint(payload, expected)
        model = HybridMemoryNet(spec, p)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return model

    model = HybridMemoryNet(spec, p)
    initial = cpu_state(model)
    save_torch(directory / "initial.pt", {**expected, "model_state_dict": initial})
    optimizer = torch.optim.Adam(model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
    best = _split_eval(model, arrays, p, spec.seed, "val")
    best_state, best_epoch = initial, 0
    history: list[dict[str, Any]] = [{
        "epoch": 0, "train_loss": None, "val_ba": best["ba"], "val_mean_logit_ce": best["mean_logit_ce"]
    }]
    train_loader = loader(arrays, "train", p, spec.seed, shuffle=True)
    stopped_epoch = 0
    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(mean_logits(model(x, lengths)["evidence"], lengths), y)
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
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val": best,
        "selection_rule": "native validation BA, then validation mean-logit CE, then earliest epoch",
    })
    return model


def extract(
    model: HybridMemoryNet,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    spec: ExpSpec,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    model.eval()
    traces: dict[str, np.ndarray] = {}
    native: dict[str, Any] = {"case": spec.case, "seed": spec.seed, "splits": {}}
    activity_rows: list[dict[str, Any]] = []
    q, r = divmod(p.width, 3)
    groups = []
    start = 0
    for gi in range(3):
        stop = start + q + (gi < r)
        groups.append((start, stop))
        start = stop
    for split in SPLITS:
        chunks = {f"L{li+1}__{state}": [] for li in range(2) for state in ("spike", "pre_reset")}
        chunks["evidence"] = []
        accum = {
            (li, gi): {"spikes": 0.0, "valid": 0.0, "pre_abs": 0.0, "fire_pre": 0.0, "fires": 0.0, "silent": 0.0, "samples": 0.0}
            for li in range(2) for gi in range(3)
        }
        for x, _y, lengths in loader(arrays, split, p, spec.seed):
            with torch.no_grad():
                out = model(x, lengths)
            chunks["evidence"].append(out["evidence"].numpy())
            mask = valid_mask(lengths, p.steps).numpy()
            for li in range(2):
                spike_np = out["spike"][li].numpy()
                pre_np = out["pre_reset"][li].numpy()
                chunks[f"L{li+1}__spike"].append(spike_np.astype(np.uint8))
                chunks[f"L{li+1}__pre_reset"].append(pre_np.astype(np.float32))
                for gi, (a, b) in enumerate(groups):
                    valid = mask[:, :, None]
                    z, v = spike_np[:, :, a:b], pre_np[:, :, a:b]
                    acc = accum[(li, gi)]
                    acc["spikes"] += float((z * valid).sum())
                    acc["valid"] += float(valid.sum() * (b - a))
                    acc["pre_abs"] += float((np.abs(v) * valid).sum())
                    fired = (z > 0) & valid
                    acc["fires"] += float(fired.sum())
                    if fired.any():
                        acc["fire_pre"] += float((v[fired] / p.threshold).sum())
                    valid_counts = (z * valid).sum(axis=1)
                    acc["silent"] += float((valid_counts == 0).sum())
                    acc["samples"] += float(valid_counts.size)
        for key, values in chunks.items():
            traces[f"{split}__{key}"] = np.concatenate(values)
        y = arrays[f"{split}_y"]
        prediction = traces[f"{split}__evidence"].sum(1).argmax(1)
        native["splits"][split] = {**metrics(y, prediction), "n_samples": len(y)}
        for li in range(2):
            for gi, shift in enumerate(SHIFTS[li]):
                acc = accum[(li, gi)]
                tau_ms = -(1000.0 / p.fs) / math.log(1.0 - 2.0 ** (-shift))
                activity_rows.append({
                    "split": split,
                    "layer": f"L{li+1}",
                    "carrier": model.carriers[li],
                    "shift": shift,
                    "slow_tau_ms": tau_ms,
                    "firing_hz": p.fs * acc["spikes"] / acc["valid"] if acc["valid"] else 0.0,
                    "mean_abs_pre_reset": acc["pre_abs"] / acc["valid"] if acc["valid"] else 0.0,
                    "mean_pre_over_threshold_when_firing": acc["fire_pre"] / acc["fires"] if acc["fires"] else 0.0,
                    "silent_neuron_fraction": acc["silent"] / acc["samples"] if acc["samples"] else 0.0,
                })
    native["train_test_gap"] = native["splits"]["train"]["ba"] - native["splits"]["test"]["ba"]
    activity = {"case": spec.case, "seed": spec.seed, "carriers": list(model.carriers), "rows": activity_rows}
    return traces, native, activity


def _load_model(config: Config, spec: ExpSpec) -> tuple[HybridMemoryNet, Protocol, dict[str, np.ndarray], dict[str, Any]]:
    p, core_lock, arrays, _ = _core(config)
    expected = _run_provenance(config, spec, core_lock["identity"])
    payload = load_torch(_checkpoint_dir(config, spec) / "checkpoint.pt")
    _validate_checkpoint(payload, expected)
    model = HybridMemoryNet(spec, p)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, p, arrays, expected


def evaluate(config: Config, spec: ExpSpec) -> dict[str, Any]:
    model, p, arrays, _ = _load_model(config, spec)
    directory = _checkpoint_dir(config, spec)
    checkpoint_hash = file_hash(directory / "checkpoint.pt")
    traces, native, activity = extract(model, arrays, p, spec)
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
    results = []
    for case in CASES:
        spec = ExpSpec(case, SEEDS[0])
        model = HybridMemoryNet(spec, p)
        optimizer = torch.optim.Adam(model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
        before = cpu_state(model)
        out = model(x, lengths)
        loss = F.cross_entropy(mean_logits(out["evidence"], lengths), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss) or any(param.grad is not None and not torch.isfinite(param.grad).all() for param in model.parameters()):
            raise FloatingPointError(f"{case}: nonfinite smoke loss/gradient")
        optimizer.step()
        after = cpu_state(model)
        if state_hash(before) == state_hash(after):
            raise AssertionError(f"{case}: optimizer step did not change parameters")
        sample_ids = arrays["train_ids"][: len(x)]
        feature = temporal_features(out["spike"][1].detach().numpy(), lengths.numpy(), sample_ids, "whole_count", p)
        if feature.shape != (len(x), p.width):
            raise AssertionError(f"{case}: probe feature path returned {feature.shape}")
        with tempfile.TemporaryDirectory(prefix="exp18_1_smoke_") as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_torch(path, {"model_state_dict": after})
            restored = HybridMemoryNet(spec, p)
            restored.load_state_dict(load_torch(path)["model_state_dict"], strict=True)
            with torch.no_grad():
                logits = mean_logits(restored(x, lengths)["evidence"], lengths)
            if logits.shape != (len(x), len(p.labels)) or not torch.isfinite(logits).all():
                raise AssertionError(f"{case}: checkpoint/evaluation roundtrip failed")
        results.append({"case": case, "carriers": list(CARRIERS[case]), "loss": float(loss.detach())})
    return {
        "status": "PASS",
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "core_identity": core_lock["identity"],
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "cases": results,
    }


def prepare(config: Config) -> dict[str, Any]:
    p, core_lock, _, exp18_refs = _core(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    ii_refs, uu_refs = {}, {}
    for seed in SEEDS:
        core_dir = config.core_results_dir / "runs" / exp18._o0_run(seed, p).key
        ii_refs[str(seed)] = {
            "directory": str(core_dir),
            "checkpoint_hash": file_hash(core_dir / "checkpoint.pt"),
            "native_hash": file_hash(core_dir / "native.json"),
            "probes_hash": file_hash(core_dir / "probes.json"),
        }
        uu_refs[str(seed)] = exp18_refs["runs"][str(seed)]
    payload = {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "source_hash": _source_hash(),
        "core_identity": core_lock["identity"],
        "core_dataset_hash": core_lock["dataset_hash"],
        "core_protocol_hash": core_lock["protocol_hash"],
        "exp18_protocol_hash": exp18_refs["protocol_hash"],
        "seeds": list(SEEDS),
        "formal_cases": list(CASES),
        "carriers": {key: list(value) for key, value in CARRIERS.items()},
        "references": {"II_REF": ii_refs, "UU_REF": uu_refs},
        "shifts": [list(value) for value in SHIFTS],
        "recommended_concurrency": 6,
    }
    path = config.results_dir / "protocol.lock.json"
    if path.is_file():
        previous = _read_json(path)
        if previous != payload:
            raise ValueError("Existing Exp18.1 protocol lock differs; use a new results directory")
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
        "initial.pt", "checkpoint.pt", "history.json", "native.json", "activity.json", "traces.npz",
        "probes.json", "probe_search.json", "probe_decoders.npz", "probe_predictions.npz",
    )
    save_json(directory / "complete.json", {
        **expected,
        "status": "PASS",
        "files": {name: file_hash(directory / name) for name in required},
    })
    return {"run": spec.key, "status": "PASS", "test_ba": native["splits"]["test"]["ba"]}


def _reference_dir(config: Config, case: str, seed: int, p: Protocol) -> Path:
    if case == "II_REF":
        return config.core_results_dir / "runs" / exp18._o0_run(seed, p).key
    if case == "UU_REF":
        return config.exp18_results_dir / "runs" / f"U_NORMAL__seed{seed}"
    return _checkpoint_dir(config, ExpSpec(case, seed))


def _primary_probe(directory: Path) -> dict[str, float]:
    rows = _read_json(directory / "probes.json")["rows"]
    result: dict[str, float] = {}
    for aggregation in PRIMARY_AGGREGATIONS:
        selected = [
            row for row in rows
            if row["layer"] == "L2" and row["state"] == "spike"
            and row["decoder"] == "no_bias" and row["aggregation"] == aggregation
        ]
        if not selected:
            raise ValueError(f"Missing L2 spike no-bias probe {aggregation}: {directory}")
        result[aggregation] = float(np.mean([row["test_ba"] for row in selected]))
    return result


def finalize(config: Config) -> dict[str, Any]:
    p, core_lock, _, _ = _core(config)
    protocol = _protocol_lock(config)
    rows: list[dict[str, Any]] = []
    all_cases = ("II_REF", "UI", "IU", "UU_REF")
    for seed in SEEDS:
        for case in all_cases:
            directory = _reference_dir(config, case, seed, p)
            if case in CASES:
                expected = _run_provenance(config, ExpSpec(case, seed), core_lock["identity"])
                if not _validate_complete(directory, expected):
                    raise FileNotFoundError(f"Missing complete hybrid run: {case} seed {seed}")
            else:
                ref = protocol["references"][case][str(seed)]
                for name in ("checkpoint", "native", "probes"):
                    if file_hash(directory / f"{name}.json") != ref[f"{name}_hash"] if name != "checkpoint" else file_hash(directory / "checkpoint.pt") != ref["checkpoint_hash"]:
                        raise ValueError(f"Changed reference artifact: {case} seed {seed} {name}")
            native = _read_json(directory / "native.json")
            probe = _primary_probe(directory)
            row = {
                "case": case,
                "seed": seed,
                "carriers": list(CARRIERS[case]),
                **{f"{split}_{name}": value for split in SPLITS for name, value in native["splits"][split].items() if isinstance(value, (int, float))},
                **{f"l2_{key}_test_ba": value for key, value in probe.items()},
            }
            row["fixed250_order_gap"] = probe["fixed250_ordered"] - probe["fixed250_shuffled"]
            row["relative10_order_gap"] = probe["relative10_ordered"] - probe["relative10_shuffled"]
            rows.append(row)

    metric_names = [
        "test_ba",
        "l2_whole_count_test_ba",
        "l2_fixed250_ordered_test_ba",
        "l2_fixed250_shuffled_test_ba",
        "l2_relative10_ordered_test_ba",
        "l2_relative10_shuffled_test_ba",
        "fixed250_order_gap",
        "relative10_order_gap",
    ]
    contrasts: list[dict[str, Any]] = []
    for seed in SEEDS:
        by_case = {row["case"]: row for row in rows if row["seed"] == seed}
        for metric in metric_names:
            ii, ui, iu, uu = (by_case[case][metric] for case in all_cases)
            contrasts.append({
                "seed": seed,
                "metric": metric,
                "L1_U_effect_given_L2_I": ui - ii,
                "L1_U_effect_given_L2_U": uu - iu,
                "L2_U_effect_given_L1_I": iu - ii,
                "L2_U_effect_given_L1_U": uu - ui,
                "interaction": uu - ui - iu + ii,
            })
    summary = []
    for metric in metric_names:
        selected = [row for row in contrasts if row["metric"] == metric]
        summary.append({
            "metric": metric,
            **{
                key: float(np.mean([row[key] for row in selected]))
                for key in (
                    "L1_U_effect_given_L2_I", "L1_U_effect_given_L2_U",
                    "L2_U_effect_given_L1_I", "L2_U_effect_given_L1_U", "interaction",
                )
            },
        })
    payload = {
        "status": "PASS",
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "core_identity": core_lock["identity"],
        "rows": rows,
        "factorial_contrasts": contrasts,
        "factorial_summary": summary,
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
    run_p.add_argument("--case", choices=CASES, required=True)
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
