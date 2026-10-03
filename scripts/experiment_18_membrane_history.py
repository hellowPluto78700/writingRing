#!/usr/bin/env python3
"""Exp18: consumable multi-tau membrane history under the CoreBenchmark O0 contract."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from core_benchmark_v1.data import loader
from core_benchmark_v1.model import BenchmarkNet, BinarySpike, mean_logits, valid_mask
from core_benchmark_v1.probes import run_probes
from core_benchmark_v1.protocol import Protocol, Run, SPLITS, paired_seed, runs
from core_benchmark_v1.storage import file_hash, load_torch, save_json, save_npz, save_torch, validate_complete
from core_benchmark_v1.training import cpu_state, metrics
from scripts import experiment_16_prefix_supervised_selective_memory as exp16

EXPERIMENT_ID = "experiment_18_membrane_history"
PROTOCOL_VERSION = "membrane_history_v1"
SEEDS = (11, 23, 37)
CASES = ("U_NORMAL", "U_DETACH")
ALL_CASES = ("I_REF", *CASES)
SHIFTS = ((2, 3, 4), (2, 3, 4))
SUFFIX_STEPS = 16


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
        core_results_dir=(args.core_results or root / exp16.CORE_RESULTS_REL).resolve(),
    )


def specs() -> list[ExpSpec]:
    return [ExpSpec(case, seed) for seed in SEEDS for case in CASES]


def _run(spec: ExpSpec) -> Run:
    return Run(spec.case, spec.seed, "18_membrane_history", shifts=SHIFTS, objective="wcce")


def _o0_run(seed: int, p: Protocol) -> Run:
    return next(run for run in runs(p) if run.case == "O0" and run.seed == seed)


def _core(config: Config) -> tuple[Protocol, dict[str, Any], dict[str, np.ndarray]]:
    p, lock, arrays = exp16._load_core(config)
    if tuple(p.seeds) != SEEDS or p.width != 128 or p.threshold != 0.5:
        raise ValueError("CoreBenchmark O0 geometry/seeds differ from Exp18 contract")
    if p.wcce_reduction != "valid_mean_logits" or p.native_bias:
        raise ValueError("Exp18 requires the CoreBenchmark bias-free valid-mean WCCE contract")
    for seed in SEEDS:
        run = _o0_run(seed, p)
        validate_complete(config.core_results_dir / "runs" / run.key, run, lock)
    return p, lock, arrays


def _slow_values(shifts: tuple[int, ...], width: int) -> torch.Tensor:
    q, r = divmod(width, len(shifts))
    values = [1.0 - 2.0 ** (-shift) for j, shift in enumerate(shifts) for _ in range(q + (j < r))]
    return torch.tensor(values, dtype=torch.float32)


class UHistoryNet(nn.Module):
    """Fast synaptic filter + slow multi-tau membrane; only reset backward differs by case."""

    def __init__(self, spec: ExpSpec, p: Protocol) -> None:
        super().__init__()
        if spec.case not in CASES:
            raise ValueError(spec.case)
        self.spec, self.protocol = spec, p
        self.layers = nn.ModuleList(
            nn.Linear(p.input_channels if i == 0 else p.width, p.width, bias=False)
            for i in range(2)
        )
        self.head = nn.Linear(p.width, len(p.labels), bias=False)
        self.fast_alpha = math.exp(-(1000.0 / p.fs) / p.tau_mem_ms)
        for i, shifts in enumerate(SHIFTS):
            self.register_buffer(f"slow_beta_{i}", _slow_values(shifts, p.width))
        for name, parameter in self.named_parameters():
            generator = torch.Generator().manual_seed(paired_seed(spec.seed, f"init:{name}"))
            bound = 1.0 / math.sqrt(parameter.shape[1])
            with torch.no_grad():
                parameter.uniform_(-bound, bound, generator=generator)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, *, capture_grad: bool = False) -> dict[str, Any]:
        p = self.protocol
        batch, steps, channels = x.shape
        if channels != p.input_channels or lengths.shape != (batch,):
            raise ValueError("Invalid Exp18 input geometry")
        syn = [x.new_zeros(batch, p.width) for _ in range(2)]
        mem = [x.new_zeros(batch, p.width) for _ in range(2)]
        spikes: list[list[torch.Tensor]] = [[], []]
        pre_reset: list[list[torch.Tensor]] = [[], []]
        post_reset: list[list[torch.Tensor]] = [[], []]
        synaptic: list[list[torch.Tensor]] = [[], []]
        pre_nodes: list[list[torch.Tensor]] = [[], []]
        evidence: list[torch.Tensor] = []
        for t in range(int(lengths.max().item())):
            active = (t < lengths)[:, None]
            cur = x[:, t] * active
            for li, linear in enumerate(self.layers):
                syn_candidate = self.fast_alpha * syn[li] + linear(cur)
                syn[li] = torch.where(active, syn_candidate, syn[li])
                pre = getattr(self, f"slow_beta_{li}") * mem[li] + syn[li]
                if capture_grad:
                    pre.retain_grad()
                    pre_nodes[li].append(pre)
                spike = BinarySpike.apply(pre, p.threshold, p.surrogate_slope)
                reset_spike = spike if self.spec.case == "U_NORMAL" else spike.detach()
                new_mem = pre - p.threshold * reset_spike
                mem[li] = torch.where(active, new_mem, mem[li])
                cur = spike * active
                spikes[li].append(cur)
                pre_reset[li].append(pre * active)
                post_reset[li].append(mem[li] * active)
                synaptic[li].append(syn[li] * active)
            evidence.append(self.head(cur))

        def stack(values: list[torch.Tensor]) -> torch.Tensor:
            result = torch.stack(values, dim=1)
            return F.pad(result, (0, 0, 0, steps - result.shape[1]))

        out: dict[str, Any] = {
            "spike": tuple(stack(v) for v in spikes),
            "pre_reset": tuple(stack(v) for v in pre_reset),
            "post_reset": tuple(stack(v) for v in post_reset),
            "synaptic": tuple(stack(v) for v in synaptic),
            "evidence": stack(evidence),
            "final_syn": tuple(syn),
            "final_mem": tuple(mem),
        }
        if capture_grad:
            out["_pre_nodes"] = tuple(pre_nodes)
        return out


def _split_eval(model: nn.Module, arrays: dict[str, np.ndarray], p: Protocol, seed: int, split: str) -> dict[str, float]:
    model.eval()
    ys, predictions = [], []
    ce_sum = 0.0
    with torch.no_grad():
        for x, y, lengths in loader(arrays, split, p, seed):
            logits = mean_logits(model(x, lengths)["evidence"], lengths)
            ys.append(y.numpy())
            predictions.append(logits.argmax(1).numpy())
            ce_sum += float(F.cross_entropy(logits, y, reduction="sum"))
    y_all, pred_all = np.concatenate(ys), np.concatenate(predictions)
    return {**metrics(y_all, pred_all), "mean_logit_ce": ce_sum / len(y_all)}


def _better(candidate: dict[str, float], best: dict[str, float]) -> bool:
    return candidate["ba"] > best["ba"] + 1e-12 or (
        abs(candidate["ba"] - best["ba"]) <= 1e-12
        and candidate["mean_logit_ce"] < best["mean_logit_ce"] - 1e-12
    )


def _checkpoint_dir(config: Config, spec: ExpSpec) -> Path:
    return config.results_dir / "runs" / spec.key


def train(config: Config, spec: ExpSpec) -> UHistoryNet:
    p, lock, arrays = _core(config)
    directory = _checkpoint_dir(config, spec)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    if checkpoint_path.exists():
        payload = load_torch(checkpoint_path)
        model = UHistoryNet(spec, p)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return model
    model = UHistoryNet(spec, p)
    initial = cpu_state(model)
    save_torch(directory / "initial.pt", {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "case": spec.case,
        "seed": spec.seed,
        "core_identity": lock["identity"],
        "model_state_dict": initial,
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
    best = _split_eval(model, arrays, p, spec.seed, "val")
    best_state, best_epoch = initial, 0
    history: list[dict[str, Any]] = [{
        "epoch": 0, "train_loss": None, "val_ba": best["ba"], "val_mean_logit_ce": best["mean_logit_ce"]
    }]
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
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
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
        if _better(val, best):
            best, best_state, best_epoch = dict(val), cpu_state(model), epoch
        stopped_epoch = epoch
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break
    model.load_state_dict(best_state, strict=True)
    save_json(directory / "history.json", {"rows": history})
    save_torch(checkpoint_path, {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "case": spec.case,
        "seed": spec.seed,
        "core_identity": lock["identity"],
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val": best,
        "selection_rule": "native validation BA, then validation mean-logit CE, then earliest epoch",
    })
    return model


def extract(model: UHistoryNet, arrays: dict[str, np.ndarray], p: Protocol, spec: ExpSpec) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model.eval()
    traces: dict[str, np.ndarray] = {}
    native: dict[str, Any] = {"case": spec.case, "seed": spec.seed, "splits": {}, "activity": []}
    groups = ((0, 43), (43, 86), (86, 128))
    for split in SPLITS:
        chunks: dict[str, list[np.ndarray]] = {"evidence": []}
        for state in ("spike", "pre_reset"):
            for li in range(2):
                chunks[f"L{li+1}__{state}"] = []
        tau_acc = {
            (li, gi): {"spikes": 0.0, "valid": 0.0, "pre_abs": 0.0, "pre_ratio_fire": 0.0, "fires": 0.0}
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
                    z = spike_np[:, :, a:b]
                    v = pre_np[:, :, a:b]
                    acc = tau_acc[(li, gi)]
                    acc["spikes"] += float((z * valid).sum())
                    acc["valid"] += float(valid.sum() * (b - a))
                    acc["pre_abs"] += float((np.abs(v) * valid).sum())
                    fired = (z > 0) & valid
                    acc["fires"] += float(fired.sum())
                    if fired.any():
                        acc["pre_ratio_fire"] += float((v[fired] / p.threshold).sum())
        for key, values in chunks.items():
            traces[f"{split}__{key}"] = np.concatenate(values)
        y = arrays[f"{split}_y"]
        scores = traces[f"{split}__evidence"].sum(1)
        prediction = scores.argmax(1)
        native["splits"][split] = {**metrics(y, prediction), "n_samples": len(y)}
        for li in range(2):
            for gi, shift in enumerate(SHIFTS[li]):
                acc = tau_acc[(li, gi)]
                tau_ms = -(1000.0 / p.fs) / math.log(1.0 - 2.0 ** (-shift))
                native["activity"].append({
                    "split": split,
                    "layer": f"L{li+1}",
                    "shift": shift,
                    "tau_mem_ms": tau_ms,
                    "firing_hz": 0.0 if acc["valid"] == 0 else p.fs * acc["spikes"] / acc["valid"],
                    "mean_abs_pre_reset": 0.0 if acc["valid"] == 0 else acc["pre_abs"] / acc["valid"],
                    "mean_pre_over_threshold_when_firing": 0.0 if acc["fires"] == 0 else acc["pre_ratio_fire"] / acc["fires"],
                })
    native["train_test_gap"] = native["splits"]["train"]["ba"] - native["splits"]["test"]["ba"]
    return traces, native


def _load_u_model(config: Config, spec: ExpSpec) -> tuple[UHistoryNet, Protocol, dict[str, np.ndarray]]:
    p, _, arrays = _core(config)
    payload = load_torch(_checkpoint_dir(config, spec) / "checkpoint.pt")
    model = UHistoryNet(spec, p)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, p, arrays


def evaluate(config: Config, spec: ExpSpec) -> dict[str, Any]:
    model, p, arrays = _load_u_model(config, spec)
    directory = _checkpoint_dir(config, spec)
    checkpoint_hash = file_hash(directory / "checkpoint.pt")
    traces, native = extract(model, arrays, p, spec)
    save_npz(directory / "traces.npz", traces)
    save_json(directory / "native.json", native)
    run_probes(directory, traces, arrays, _run(spec), p)
    if file_hash(directory / "checkpoint.pt") != checkpoint_hash:
        raise AssertionError("Evaluation modified selected checkpoint")
    return native


def _suffix_logits(evidence: torch.Tensor, lengths: torch.Tensor, count: int) -> torch.Tensor:
    t = torch.arange(evidence.shape[1], device=evidence.device)[None, :]
    start = torch.clamp(lengths - count, min=0)[:, None]
    mask = (t >= start) & (t < lengths[:, None])
    denom = mask.sum(1).clamp_min(1).to(evidence.dtype)[:, None]
    return (evidence * mask.unsqueeze(-1)).sum(1) / denom


def _gradient_rows(model: nn.Module, arrays: dict[str, np.ndarray], p: Protocol, seed: int, loss_kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    sums: dict[tuple[int, int, int], list[float]] = {}
    for x, y, lengths in loader(arrays, "val", p, seed):
        model.zero_grad(set_to_none=True)
        out = model(x, lengths, capture_grad=True)
        logits = mean_logits(out["evidence"], lengths) if loss_kind == "wcce" else _suffix_logits(out["evidence"], lengths, SUFFIX_STEPS)
        F.cross_entropy(logits, y).backward()
        for li in range(2):
            spikes = out["spike"][li].detach()
            future = torch.flip(torch.cumsum(torch.flip(spikes, dims=[1]), dim=1), dims=[1])
            for t, pre in enumerate(out["_pre_nodes"][li]):
                if pre.grad is None:
                    continue
                active = t < lengths
                if not bool(active.any()):
                    continue
                grad = pre.grad.detach()[active].abs()
                future_t = future[active, t]
                for gi, (a, b) in enumerate(((0, 43), (43, 86), (86, 128))):
                    g = grad[:, a:b].reshape(-1)
                    nreset = future_t[:, a:b].reshape(-1).to(torch.int64)
                    for bucket in range(11):
                        selected = nreset == bucket if bucket < 10 else nreset >= 10
                        if bool(selected.any()):
                            key = (li, gi, bucket)
                            sums.setdefault(key, [0.0, 0.0])
                            sums[key][0] += float(g[selected].sum())
                            sums[key][1] += float(selected.sum())
    for (li, gi, bucket), (total, n) in sorted(sums.items()):
        shift = SHIFTS[li][gi]
        rows.append({
            "loss_kind": loss_kind,
            "layer": f"L{li+1}",
            "shift": shift,
            "tau_ms": -(1000.0 / p.fs) / math.log(1.0 - 2.0 ** (-shift)),
            "future_reset_bucket": "10+" if bucket == 10 else str(bucket),
            "mean_abs_pre_gradient": total / n,
            "n_values": int(n),
        })
    return rows


class IRefDiagnosticNet(BenchmarkNet):
    """O0-equivalent forward with retained pre-reset nodes for diagnostic-only gradients."""

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, *, capture_grad: bool = False) -> dict[str, Any]:
        if not capture_grad:
            return super().forward(x, lengths)
        p = self.protocol
        batch, steps, _ = x.shape
        syn = [x.new_zeros(batch, p.width) for _ in self.layers]
        mem = [x.new_zeros(batch, p.width) for _ in self.layers]
        spikes: list[list[torch.Tensor]] = [[] for _ in self.layers]
        pre_nodes: list[list[torch.Tensor]] = [[] for _ in self.layers]
        evidence: list[torch.Tensor] = []
        for t in range(int(lengths.max().item())):
            active = (t < lengths)[:, None]
            cur = x[:, t] * active
            for li, linear in enumerate(self.layers):
                syn_candidate = getattr(self, f"alpha_{li}") * syn[li] + linear(cur)
                syn[li] = torch.where(active, syn_candidate, syn[li])
                pre = self.betas[li] * mem[li] + syn[li]
                pre.retain_grad()
                pre_nodes[li].append(pre)
                spike = BinarySpike.apply(pre, p.threshold, p.surrogate_slope)
                mem_candidate = pre - p.threshold * spike
                mem[li] = torch.where(active, mem_candidate, mem[li])
                cur = spike * active
                spikes[li].append(cur)
            evidence.append(self.head(cur))
        def stack(values: list[torch.Tensor]) -> torch.Tensor:
            result = torch.stack(values, dim=1)
            return F.pad(result, (0, 0, 0, steps - result.shape[1]))
        return {"spike": tuple(stack(v) for v in spikes), "evidence": stack(evidence), "_pre_nodes": tuple(pre_nodes)}


def diagnose(config: Config, case: str, seed: int) -> dict[str, Any]:
    p, lock, arrays = _core(config)
    if case == "I_REF":
        run = _o0_run(seed, p)
        source_checkpoint = config.core_results_dir / "runs" / run.key / "checkpoint.pt"
        payload = load_torch(source_checkpoint)
        model: nn.Module = IRefDiagnosticNet(run, p)
        model.load_state_dict(payload["model_state_dict"], strict=True)
    else:
        spec = ExpSpec(case, seed)
        model, _, _ = _load_u_model(config, spec)
        source_checkpoint = _checkpoint_dir(config, spec) / "checkpoint.pt"
    rows = _gradient_rows(model, arrays, p, seed, "wcce") + _gradient_rows(model, arrays, p, seed, "suffix")
    directory = config.results_dir / "diagnostics" / f"{case}__seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    result = {
        "case": case,
        "seed": seed,
        "core_identity": lock["identity"],
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_hash": file_hash(source_checkpoint),
        "split": "val",
        "suffix_steps": SUFFIX_STEPS,
        "rows": rows,
    }
    save_json(directory / "gradient_diagnostics.json", result)
    return result


def prepare(config: Config) -> dict[str, Any]:
    p, lock, _ = _core(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    baseline = {}
    for seed in SEEDS:
        run = _o0_run(seed, p)
        directory = config.core_results_dir / "runs" / run.key
        baseline[str(seed)] = {
            "run_key": run.key,
            "checkpoint_hash": file_hash(directory / "checkpoint.pt"),
            "native_hash": file_hash(directory / "native.json"),
            "probes_hash": file_hash(directory / "probes.json"),
        }
    payload = {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "core_dataset_hash": lock["dataset_hash"],
        "core_protocol_hash": lock["protocol_hash"],
        "seeds": list(SEEDS),
        "cases": list(CASES),
        "baseline": baseline,
        "shifts": [list(v) for v in SHIFTS],
        "fast_synaptic_tau_ms": p.tau_mem_ms,
        "fast_synaptic_alpha": math.exp(-(1000.0 / p.fs) / p.tau_mem_ms),
        "threshold": p.threshold,
        "suffix_steps": SUFFIX_STEPS,
        "contract": "CoreBenchmark O0 except slow/fast state-carrier swap and reset-gradient rule",
    }
    save_json(config.results_dir / "protocol.lock.json", payload)
    return payload


def run_one(config: Config, spec: ExpSpec) -> dict[str, Any]:
    train(config, spec)
    native = evaluate(config, spec)
    diagnose(config, spec.case, spec.seed)
    directory = _checkpoint_dir(config, spec)
    required = (
        "initial.pt", "checkpoint.pt", "history.json", "native.json", "traces.npz",
        "probes.json", "probe_search.json", "probe_decoders.npz", "probe_predictions.npz",
    )
    save_json(directory / "complete.json", {
        "status": "PASS",
        "case": spec.case,
        "seed": spec.seed,
        "files": {name: file_hash(directory / name) for name in required},
    })
    return native


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(config: Config) -> dict[str, Any]:
    p, lock, _ = _core(config)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        o0_dir = config.core_results_dir / "runs" / _o0_run(seed, p).key
        ref = _read_json(o0_dir / "native.json")
        rows.append({
            "case": "I_REF",
            "seed": seed,
            **{f"{s}_{k}": v for s in SPLITS for k, v in ref["splits"][s].items() if isinstance(v, (int, float))},
        })
        if not (config.results_dir / "diagnostics" / f"I_REF__seed{seed}" / "gradient_diagnostics.json").is_file():
            raise FileNotFoundError(f"Missing I_REF diagnostic for seed {seed}")
        for case in CASES:
            directory = _checkpoint_dir(config, ExpSpec(case, seed))
            complete = _read_json(directory / "complete.json")
            if complete.get("status") != "PASS":
                raise ValueError(f"Incomplete Exp18 run: {case} seed {seed}")
            for name, expected in complete["files"].items():
                if file_hash(directory / name) != expected:
                    raise ValueError(f"Changed Exp18 artifact: {directory / name}")
            native = _read_json(directory / "native.json")
            rows.append({
                "case": case,
                "seed": seed,
                **{f"{s}_{k}": v for s in SPLITS for k, v in native["splits"][s].items() if isinstance(v, (int, float))},
            })
    paired = []
    for seed in SEEDS:
        by_case = {row["case"]: row for row in rows if row["seed"] == seed}
        paired.append({
            "seed": seed,
            "U_NORMAL_minus_I_REF_test_ba": by_case["U_NORMAL"]["test_ba"] - by_case["I_REF"]["test_ba"],
            "U_DETACH_minus_I_REF_test_ba": by_case["U_DETACH"]["test_ba"] - by_case["I_REF"]["test_ba"],
            "U_DETACH_minus_U_NORMAL_test_ba": by_case["U_DETACH"]["test_ba"] - by_case["U_NORMAL"]["test_ba"],
        })
    payload = {
        "status": "PASS",
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "rows": rows,
        "paired": paired,
    }
    save_json(config.results_dir / "aggregate.json", payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run_p = sub.add_parser("run")
    run_p.add_argument("--case", choices=CASES, required=True)
    run_p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    diag_p = sub.add_parser("diagnose")
    diag_p.add_argument("--case", choices=ALL_CASES, required=True)
    diag_p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    sub.add_parser("finalize")
    args = parser.parse_args(argv)
    config = config_from_args(args)
    if args.command == "prepare":
        result = prepare(config)
    elif args.command == "run":
        result = run_one(config, ExpSpec(args.case, args.seed))
    elif args.command == "diagnose":
        result = diagnose(config, args.case, args.seed)
    else:
        result = finalize(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
