from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from core_benchmark_v1.data import load_cache, loader
from core_benchmark_v1.model import BenchmarkNet, lif_step, mean_logits, sequence_loss
from core_benchmark_v1.probes import AGGREGATIONS, DECODERS, fit_probe, temporal_features
from core_benchmark_v1.protocol import Protocol, Run, SPLITS, digest, paired_seed
from core_benchmark_v1.storage import file_hash, save_json, save_npz, save_torch, validate_checkpoint
from core_benchmark_v1.training import cpu_state, metrics

EXPERIMENT_ID = "experiment_15_context_dependent_write_gate"
PROTOCOL_VERSION = "context_write_gate_v1"
CORE_RESULTS_REL = Path("core_benchmark_v1/results/main")
SEEDS = (11, 23, 37)
SHIFTS = ((2, 3, 4), (2, 3, 4))
CASES = ("C0", "GF", "GJ")
GATED_CASES = ("GF", "GJ")
Q_INIT = 0.9
GRADIENT_EPOCHS = (0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
INTERVENTIONS = ("A0", "A1", "A2", "A3", "A4", "A4b")
A3_SHUFFLE_SEEDS = (101, 211, 307, 401, 503)


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
        root,
        (args.results or default_results_dir(root)).resolve(),
        (args.core_results or root / CORE_RESULTS_REL).resolve(),
    )


def phase1_specs() -> list[ExpSpec]:
    return [ExpSpec(case, seed) for case in CASES for seed in SEEDS]


def phase1_5_specs() -> list[ExpSpec]:
    return [ExpSpec(case, seed) for case in GATED_CASES for seed in SEEDS]


def _core_o0_run(seed: int) -> Run:
    return Run("O0", seed, "01_objective", shifts=SHIFTS, objective="wcce")


def _exp_run(spec: ExpSpec) -> Run:
    return Run(spec.case, spec.seed, "15_context_write_gate", shifts=SHIFTS, objective="wcce")


def _load_core(config: Config) -> tuple[Protocol, dict[str, Any], dict[str, np.ndarray]]:
    lock = json.loads((config.core_results_dir / "protocol.lock.json").read_text(encoding="utf-8"))
    payload = dict(lock["protocol"])
    payload["version"] = Protocol().version
    p = Protocol.from_dict(payload)
    if tuple(p.seeds) != SEEDS:
        raise ValueError(f"CoreBenchmark seeds changed: {p.seeds}")
    if (p.width, p.fs, p.input_channels, p.tau_mem_ms) != (128, 64.0, 30, 22.54):
        raise ValueError("CoreBenchmark geometry differs from Exp15 contract")
    if file_hash(config.core_results_dir / "dataset.npz") != lock["dataset_hash"]:
        raise ValueError("CoreBenchmark dataset hash changed")
    expected = {
        "train": ["user_0", "user_1", "user_11", "user_12", "user_13", "user_14", "user_15",
                  "user_18", "user_19", "user_2", "user_20", "user_5", "user_7", "user_8"],
        "val": ["user_16", "user_4", "user_9"],
        "test": ["user_10", "user_3", "user_6"],
    }
    actual = {key: list(value) for key, value in lock["data_metadata"]["users"].items()}
    if actual != expected:
        raise ValueError(f"CoreBenchmark user split differs from Exp15 contract: {actual}")
    arrays = load_cache(config.core_results_dir, p, lock)
    for seed in SEEDS:
        path = config.core_results_dir / "runs" / f"O0__seed{seed}" / "checkpoint.pt"
        if not path.exists():
            raise FileNotFoundError(path)
    return p, lock, arrays


def prepare(config: Config) -> dict[str, Any]:
    p, lock, _ = _load_core(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "core_dataset_hash": lock["dataset_hash"],
        "seeds": list(SEEDS),
        "cases": list(CASES),
        "q_init": Q_INIT,
        "phase1_count": len(phase1_specs()),
        "phase1_5_count": len(phase1_5_specs()),
        "interventions": list(INTERVENTIONS),
        "a3_shuffle_seeds": list(A3_SHUFFLE_SEEDS),
        "optimizer": {"lr": p.learning_rate, "weight_decay": p.weight_decay},
        "probe_contract": (
            "CoreBenchmark temporal_features + fit_probe; sample-wise shuffled train/val/test "
            "with a separately fit matched decoder for each shuffle replicate"
        ),
        "users": lock["data_metadata"]["users"],
    }
    payload["identity"] = digest(payload)
    save_json(config.results_dir / "protocol.json", payload)
    source_paths = [
        Path(__file__).resolve(),
        config.repo_root / "core_benchmark_v1" / "model.py",
        config.repo_root / "core_benchmark_v1" / "probes.py",
        config.repo_root / "core_benchmark_v1" / "data.py",
        config.repo_root / "core_benchmark_v1" / "training.py",
    ]
    save_json(
        config.results_dir / "source_manifest.json",
        {
            "files": {
                str(path.relative_to(config.repo_root)): file_hash(path)
                for path in source_paths
            }
        },
    )
    save_json(
        config.results_dir / "locked_core_contract.json",
        {
            "core_identity": lock["identity"],
            "core_protocol_hash": lock["protocol_hash"],
            "core_dataset_hash": lock["dataset_hash"],
            "users": lock["data_metadata"]["users"],
            "seeds": list(SEEDS),
            "shifts": [list(value) for value in SHIFTS],
        },
    )
    return payload


class ContextWriteGateNet(BenchmarkNet):
    def __init__(self, run: Run, p: Protocol) -> None:
        super().__init__(run, p)
        self.gate_input = nn.Linear(p.width, 1, bias=False)
        self.gate_history = nn.Linear(p.width, 1, bias=False)
        self.gate_bias = nn.Parameter(
            torch.tensor(math.log(Q_INIT / (1.0 - Q_INIT)), dtype=torch.float32)
        )
        with torch.no_grad():
            self.gate_input.weight.zero_()
            self.gate_history.weight.zero_()

    def load_baseline_state(self, state: dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        for key, value in state.items():
            if key not in own:
                raise KeyError(key)
            own[key].copy_(value)
        self.load_state_dict(own, strict=True)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        *,
        forced_m: torch.Tensor | None = None,
        remove_gate_history: bool = False,
        history_override: torch.Tensor | None = None,
        capture_gate_grad: bool = False,
    ) -> dict[str, Any]:
        p = self.protocol
        batch, steps, channels = x.shape
        if channels != p.input_channels or lengths.shape != (batch,):
            raise ValueError("Invalid input geometry")
        if forced_m is not None and forced_m.shape != (batch, steps):
            raise ValueError("forced_m must be [batch, steps]")
        if history_override is not None and history_override.shape != (batch, steps):
            raise ValueError("history_override must be [batch, steps]")

        syn = [x.new_zeros(batch, p.width) for _ in self.layers]
        mem = [x.new_zeros(batch, p.width) for _ in self.layers]
        spikes: list[list[torch.Tensor]] = [[], []]
        pre_reset: list[list[torch.Tensor]] = [[], []]
        evidence: list[torch.Tensor] = []
        gate_g: list[torch.Tensor] = []
        gate_m: list[torch.Tensor] = []
        gate_a: list[torch.Tensor] = []
        gate_in: list[torch.Tensor] = []
        gate_hist: list[torch.Tensor] = []
        prev_l2 = x.new_zeros(batch, p.width)

        for t in range(int(lengths.max().item())):
            active = (t < lengths)[:, None]
            cur = x[:, t] * active

            candidate0 = self.alpha_0 * syn[0] + self.layers[0](cur)
            syn[0] = torch.where(active, candidate0, syn[0])
            s1, new_mem1, pre1 = lif_step(
                syn[0], mem[0], self.betas[0], p.threshold, p.surrogate_slope
            )
            mem[0] = torch.where(active, new_mem1, mem[0])
            s1 = s1 * active

            input_term = self.gate_input(s1).squeeze(-1)
            history_term = self.gate_history(prev_l2).squeeze(-1)
            used_history = torch.zeros_like(history_term) if remove_gate_history else history_term
            if history_override is not None:
                used_history = history_override[:, t]
            a = input_term + used_history + self.gate_bias
            if capture_gate_grad:
                a.retain_grad()
            g = torch.sigmoid(a)
            multiplier = g / Q_INIT if forced_m is None else forced_m[:, t]

            candidate1 = (
                self.alpha_1 * syn[1] + self.layers[1](s1) * multiplier[:, None]
            )
            syn[1] = torch.where(active, candidate1, syn[1])
            s2, new_mem2, pre2 = lif_step(
                syn[1], mem[1], self.betas[1], p.threshold, p.surrogate_slope
            )
            mem[1] = torch.where(active, new_mem2, mem[1])
            s2 = s2 * active
            prev_l2 = s2

            spikes[0].append(s1)
            spikes[1].append(s2)
            pre_reset[0].append(pre1 * active)
            pre_reset[1].append(pre2 * active)
            evidence.append(self.head(s2))
            gate_g.append(g * active.squeeze(1))
            gate_m.append(multiplier * active.squeeze(1))
            gate_a.append(a)
            gate_in.append(input_term)
            gate_hist.append(history_term)

        def stack3(values: list[torch.Tensor]) -> torch.Tensor:
            result = torch.stack(values, dim=1)
            return F.pad(result, (0, 0, 0, steps - result.shape[1]))

        def stack2(values: list[torch.Tensor]) -> torch.Tensor:
            result = torch.stack(values, dim=1)
            return F.pad(result, (0, steps - result.shape[1]))

        a_stack = stack2(gate_a)

        return {
            "spike": (stack3(spikes[0]), stack3(spikes[1])),
            "pre_reset": (stack3(pre_reset[0]), stack3(pre_reset[1])),
            "evidence": stack3(evidence),
            "final_syn": tuple(syn),
            "final_mem": tuple(mem),
            "gate_g": stack2(gate_g),
            "gate_m": stack2(gate_m),
            "gate_a": a_stack,
            "gate_input_term": stack2(gate_in),
            "gate_history_term": stack2(gate_hist),
            "gate_a_nodes": gate_a if capture_gate_grad else None,
        }


def _load_baseline(
    config: Config, seed: int, p: Protocol, lock: dict[str, Any]
) -> dict[str, torch.Tensor]:
    path = config.core_results_dir / "runs" / f"O0__seed{seed}" / "checkpoint.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(payload, _core_o0_run(seed), lock)
    if not payload.get("training_complete"):
        raise ValueError(f"Incomplete baseline checkpoint: {path}")
    return payload["model_state_dict"]


def _build_model(
    config: Config, spec: ExpSpec, p: Protocol, lock: dict[str, Any]
) -> nn.Module:
    state = _load_baseline(config, spec.seed, p, lock)
    if spec.case == "C0":
        model = BenchmarkNet(_exp_run(spec), p)
        model.load_state_dict(state, strict=True)
        return model

    model = ContextWriteGateNet(_exp_run(spec), p)
    model.load_baseline_state(state)
    if spec.case == "GF":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("gate_"))
    return model


def _split_eval(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
    split: str,
) -> dict[str, float]:
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
    result = metrics(y_all, pred_all)
    result["mean_logit_ce"] = sum(losses) / len(y_all)
    return result


def _gradient_diagnostic(
    model: ContextWriteGateNet,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    x, y, lengths = next(iter(loader(arrays, "train", p, seed, shuffle=False)))
    model.zero_grad(set_to_none=True)
    model.train()
    trajectory = model(x, lengths, capture_gate_grad=True)
    loss = sequence_loss(trajectory["evidence"], lengths, y, "wcce")
    loss.backward()
    row: dict[str, Any] = {"epoch": epoch, "loss": float(loss.detach())}
    parameters = dict(model.named_parameters())
    for name in (
        "gate_input.weight",
        "gate_history.weight",
        "gate_bias",
        "layers.1.weight",
        "head.weight",
    ):
        parameter = parameters.get(name)
        if parameter is None:
            continue
        grad = 0.0 if parameter.grad is None else float(parameter.grad.norm().detach())
        norm = float(parameter.detach().norm())
        row[f"grad__{name}"] = grad
        row[f"relative_grad__{name}"] = grad / (norm + 1e-12)

    nodes = trajectory["gate_a_nodes"]
    if nodes is not None and all(node.grad is not None for node in nodes):
        gradient = torch.stack([node.grad.detach().abs() for node in nodes], dim=1)
        phase_values: list[float] = []
        for bin_index in range(10):
            values: list[torch.Tensor] = []
            for sample_index, length in enumerate(lengths.tolist()):
                lo = int(math.floor(bin_index * length / 10))
                hi = int(math.floor((bin_index + 1) * length / 10))
                if hi > lo:
                    values.append(gradient[sample_index, lo:hi])
            phase_values.append(
                float(torch.cat(values).mean()) if values else float("nan")
            )
        row["gate_a_abs_grad_phase10"] = phase_values
    model.zero_grad(set_to_none=True)
    return row


def _make_optimizer(model: nn.Module, p: Protocol) -> torch.optim.Optimizer:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if isinstance(model, ContextWriteGateNet):
        gate_bias = [parameter for name, parameter in trainable if name == "gate_bias"]
        others = [parameter for name, parameter in trainable if name != "gate_bias"]
        groups: list[dict[str, Any]] = []
        if others:
            groups.append({"params": others, "weight_decay": p.weight_decay})
        if gate_bias:
            groups.append({"params": gate_bias, "weight_decay": 0.0})
        return torch.optim.Adam(groups, lr=p.learning_rate)
    return torch.optim.Adam(
        [parameter for _, parameter in trainable],
        lr=p.learning_rate,
        weight_decay=p.weight_decay,
    )


def train_one(config: Config, spec: ExpSpec) -> dict[str, Any]:
    p, lock, arrays = _load_core(config)
    directory = config.results_dir / "runs" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "checkpoint.pt").exists():
        return {"status": "exists", "run": spec.key}

    model = _build_model(config, spec, p, lock)
    initial = cpu_state(model)
    optimizer = _make_optimizer(model, p)
    train_loader = loader(arrays, "train", p, spec.seed, shuffle=True)
    best = _split_eval(model, arrays, p, spec.seed, "val")
    best_state = initial
    best_epoch = 0
    history = [{
        "epoch": 0,
        "train_loss": None,
        "val_ba": best["ba"],
        "val_mean_logit_ce": best["mean_logit_ce"],
    }]
    gradient_rows: list[dict[str, Any]] = []
    if isinstance(model, ContextWriteGateNet):
        gradient_rows.append(_gradient_diagnostic(model, arrays, p, spec.seed, 0))

    stopped_epoch = 0
    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for x, y, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(model(x, lengths)["evidence"], lengths, y, "wcce")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite loss")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"{spec.key}: nonfinite gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            count += len(y)

        val = _split_eval(model, arrays, p, spec.seed, "val")
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / count,
            "val_ba": val["ba"],
            "val_mean_logit_ce": val["mean_logit_ce"],
        })
        improved = (
            val["ba"] > best["ba"] + 1e-12
            or (
                abs(val["ba"] - best["ba"]) <= 1e-12
                and val["mean_logit_ce"] < best["mean_logit_ce"] - 1e-12
            )
        )
        if improved:
            best = val
            best_state = cpu_state(model)
            best_epoch = epoch
        if isinstance(model, ContextWriteGateNet) and epoch in GRADIENT_EPOCHS:
            gradient_rows.append(
                _gradient_diagnostic(model, arrays, p, spec.seed, epoch)
            )
        stopped_epoch = epoch
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    model.load_state_dict(best_state)
    if isinstance(model, ContextWriteGateNet):
        selected_gradient = _gradient_diagnostic(
            model, arrays, p, spec.seed, best_epoch
        )
        selected_gradient["selected_checkpoint"] = True
        gradient_rows.append(selected_gradient)
    if spec.case == "GF":
        baseline_keys = [key for key in initial if not key.startswith("gate_")]
        if any(not torch.equal(initial[key], best_state[key]) for key in baseline_keys):
            raise AssertionError("GF frozen backbone changed")

    save_json(directory / "train_history.json", {"rows": history})
    save_json(directory / "gate_gradient_diagnostics.json", {"rows": gradient_rows})
    save_torch(
        directory / "checkpoint.pt",
        {
            "experiment": EXPERIMENT_ID,
            "protocol": PROTOCOL_VERSION,
            "case": spec.case,
            "seed": spec.seed,
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val": best,
            "core_checkpoint_hash": file_hash(
                config.core_results_dir / "runs" / f"O0__seed{spec.seed}" / "checkpoint.pt"
            ),
            "selection_rule": (
                "validation native BA, then validation mean-logit CE, epoch0 included"
            ),
        },
    )
    evaluate_one(config, spec, model=model)
    return {
        "status": "PASS",
        "run": spec.key,
        "best_epoch": best_epoch,
        "best_val_ba": best["ba"],
    }


def _extract(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    traces: dict[str, np.ndarray] = {}
    native: dict[str, Any] = {"splits": {}, "users": []}
    gate_arrays: dict[str, np.ndarray] = {}
    model.eval()

    with torch.no_grad():
        for split in SPLITS:
            evidence_chunks: list[np.ndarray] = []
            l2_chunks: list[np.ndarray] = []
            gate_chunks = {
                key: []
                for key in (
                    "gate_g",
                    "gate_m",
                    "gate_input_term",
                    "gate_history_term",
                )
            }
            for x, _, lengths in loader(arrays, split, p, seed):
                trajectory = model(x, lengths)
                evidence_chunks.append(trajectory["evidence"].numpy())
                l2_chunks.append(trajectory["spike"][1].numpy().astype(np.uint8))
                if isinstance(model, ContextWriteGateNet):
                    for key in gate_chunks:
                        gate_chunks[key].append(trajectory[key].numpy())

            evidence = np.concatenate(evidence_chunks)
            l2 = np.concatenate(l2_chunks)
            traces[f"{split}__evidence"] = evidence
            traces[f"{split}__L2__spike"] = l2
            y = arrays[f"{split}_y"]
            prediction = evidence.sum(1).argmax(1)
            native["splits"][split] = {
                **metrics(y, prediction),
                "n_samples": len(y),
            }
            for user in np.unique(arrays[f"{split}_users"]):
                selected = arrays[f"{split}_users"] == user
                native["users"].append({
                    "split": split,
                    "user": str(user),
                    "n": int(selected.sum()),
                    **metrics(y[selected], prediction[selected]),
                })
            if isinstance(model, ContextWriteGateNet):
                for key, chunks in gate_chunks.items():
                    gate_arrays[f"{split}__{key}"] = np.concatenate(chunks)

    native["train_test_gap"] = (
        native["splits"]["train"]["ba"] - native["splits"]["test"]["ba"]
    )
    return traces, native, gate_arrays


def _gate_summary(
    gates: dict[str, np.ndarray], arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        m = gates[f"{split}__gate_m"]
        g = gates[f"{split}__gate_g"]
        input_term = gates[f"{split}__gate_input_term"]
        history_term = gates[f"{split}__gate_history_term"]
        lengths = arrays[f"{split}_lengths"]
        valid = np.arange(m.shape[1])[None, :] < lengths[:, None]
        mv = m[valid]
        gv = g[valid]
        input_valid = input_term[valid]
        history_valid = history_term[valid]
        sequence_means = np.array([m[i, :length].mean() for i, length in enumerate(lengths)])
        within_variances = np.array([m[i, :length].var() for i, length in enumerate(lengths)])
        input_rms = float(np.sqrt(np.mean(input_valid ** 2)))
        history_rms = float(np.sqrt(np.mean(history_valid ** 2)))
        rows.append({
            "split": split,
            "mean_g": float(gv.mean()),
            "std_g": float(gv.std()),
            "p_g_lt_01": float((gv < 0.1).mean()),
            "p_g_gt_09": float((gv > 0.9).mean()),
            "mean_m": float(mv.mean()),
            "std_m": float(mv.std()),
            "mean_abs_m_minus_1": float(np.abs(mv - 1).mean()),
            "rms_m_minus_1": float(np.sqrt(np.mean((mv - 1) ** 2))),
            "p_abs_m_minus_1_gt_005": float((np.abs(mv - 1) > 0.05).mean()),
            "p_abs_m_minus_1_gt_010": float((np.abs(mv - 1) > 0.10).mean()),
            "within_sequence_var_mean": float(within_variances.mean()),
            "between_sequence_mean_var": float(sequence_means.var()),
            "input_term_rms": input_rms,
            "history_term_rms": history_rms,
            "history_input_rms_ratio": history_rms / (input_rms + 1e-12),
            **{
                f"m_q{quantile:02d}": float(np.quantile(mv, quantile / 100))
                for quantile in (5, 25, 50, 75, 95)
            },
        })

        for bin_index in range(10):
            values_m: list[np.ndarray] = []
            values_g: list[np.ndarray] = []
            values_input: list[np.ndarray] = []
            values_history: list[np.ndarray] = []
            for sample_index, length in enumerate(lengths):
                lo = int(math.floor(bin_index * int(length) / 10))
                hi = int(math.floor((bin_index + 1) * int(length) / 10))
                if hi > lo:
                    values_m.append(m[sample_index, lo:hi])
                    values_g.append(g[sample_index, lo:hi])
                    values_input.append(input_term[sample_index, lo:hi])
                    values_history.append(history_term[sample_index, lo:hi])
            input_values = np.concatenate(values_input)
            history_values = np.concatenate(values_history)
            phase_rows.append({
                "split": split,
                "bin": bin_index,
                "mean_m": float(np.concatenate(values_m).mean()),
                "std_m": float(np.concatenate(values_m).std()),
                "mean_g": float(np.concatenate(values_g).mean()),
                "std_g": float(np.concatenate(values_g).std()),
                "input_term_rms": float(np.sqrt(np.mean(input_values ** 2))),
                "history_term_rms": float(np.sqrt(np.mean(history_values ** 2))),
            })
    return {"summary": rows, "phase10": phase_rows}


def _run_l2_probes(
    directory: Path,
    traces: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    spec: ExpSpec,
    p: Protocol,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    for aggregation in AGGREGATIONS:
        shuffle_seeds = p.shuffle_seeds if aggregation.endswith("shuffled") else (-1,)
        for shuffle_seed in shuffle_seeds:
            features = {
                split: temporal_features(
                    traces[f"{split}__L2__spike"],
                    arrays[f"{split}_lengths"],
                    arrays[f"{split}_ids"],
                    aggregation,
                    p,
                    shuffle_seed,
                )
                for split in SPLITS
            }
            for decoder in DECODERS:
                scaler, probe, search = fit_probe(
                    features["train"],
                    arrays["train_y"],
                    features["val"],
                    arrays["val_y"],
                    decoder,
                    p,
                )
                row: dict[str, Any] = {
                    "case": spec.case,
                    "seed": spec.seed,
                    "layer": "L2",
                    "state": "spike",
                    "aggregation": aggregation,
                    "decoder": decoder,
                    "shuffle_seed": shuffle_seed,
                    "C": float(probe.C),
                }
                for split in SPLITS:
                    prediction = probe.predict(
                        scaler.transform(features[split].astype(np.float64))
                    )
                    row.update({
                        f"{split}_{name}": value
                        for name, value in metrics(
                            arrays[f"{split}_y"], prediction
                        ).items()
                    })
                row["train_test_gap"] = row["train_ba"] - row["test_ba"]
                rows.append(row)
                search_rows.extend({
                    "aggregation": aggregation,
                    "decoder": decoder,
                    "shuffle_seed": shuffle_seed,
                    **candidate,
                } for candidate in search)
    save_json(directory / "probes.json", {"rows": rows})
    save_json(directory / "probe_search.json", {"rows": search_rows})
    return rows


def evaluate_one(
    config: Config,
    spec: ExpSpec,
    model: nn.Module | None = None,
) -> dict[str, Any]:
    p, lock, arrays = _load_core(config)
    directory = config.results_dir / "runs" / spec.key
    if model is None:
        payload = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
        model = _build_model(config, spec, p, lock)
        model.load_state_dict(payload["model_state_dict"], strict=True)
    traces, native, gates = _extract(model, arrays, p, spec.seed)
    save_json(directory / "native.json", native)
    save_npz(directory / "traces.npz", traces)
    _run_l2_probes(directory, traces, arrays, spec, p)
    if gates:
        gate_summary = _gate_summary(gates, arrays)
        save_json(directory / "gate_summary.json", gate_summary)
    return {"status": "PASS", "run": spec.key}


def _intervention_forward(
    model: ContextWriteGateNet,
    x: torch.Tensor,
    lengths: torch.Tensor,
    intervention: str,
    sample_ids: np.ndarray,
    shuffle_seed: int = -1,
) -> dict[str, Any]:
    if intervention == "A0":
        return model(x, lengths)

    first = model(x, lengths)
    steps = x.shape[1]
    valid = torch.arange(steps)[None, :] < lengths[:, None]
    if intervention == "A1":
        return model(x, lengths, forced_m=torch.ones_like(first["gate_m"]))
    if intervention == "A2":
        mean = (first["gate_m"] * valid).sum(1) / lengths
        return model(x, lengths, forced_m=mean[:, None].expand(-1, steps))
    if intervention == "A3":
        forced = first["gate_m"].detach().clone()
        for sample_index, length in enumerate(lengths.tolist()):
            rng = np.random.default_rng(
                paired_seed(shuffle_seed, f"exp15:A3:{sample_ids[sample_index]}")
            )
            permutation = torch.from_numpy(rng.permutation(length)).long()
            forced[sample_index, :length] = forced[sample_index, permutation]
        return model(x, lengths, forced_m=forced)
    if intervention == "A4":
        return model(x, lengths, remove_gate_history=True)
    if intervention == "A4b":
        history = first["gate_history_term"]
        mean = (history * valid).sum(1) / lengths
        return model(
            x,
            lengths,
            history_override=mean[:, None].expand(-1, steps),
        )
    raise ValueError(intervention)


def _extract_intervention(
    model: ContextWriteGateNet,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
    intervention: str,
    shuffle_seed: int = -1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    traces: dict[str, np.ndarray] = {}
    native: dict[str, Any] = {"splits": {}, "users": []}
    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            evidence_chunks: list[np.ndarray] = []
            l2_chunks: list[np.ndarray] = []
            offset = 0
            for x, y, lengths in loader(arrays, split, p, seed):
                sample_ids = arrays[f"{split}_ids"][offset:offset + len(y)]
                offset += len(y)
                trajectory = _intervention_forward(
                    model,
                    x,
                    lengths,
                    intervention,
                    sample_ids,
                    shuffle_seed,
                )
                evidence_chunks.append(trajectory["evidence"].numpy())
                l2_chunks.append(trajectory["spike"][1].numpy().astype(np.uint8))
            evidence = np.concatenate(evidence_chunks)
            l2 = np.concatenate(l2_chunks)
            traces[f"{split}__evidence"] = evidence
            traces[f"{split}__L2__spike"] = l2
            y = arrays[f"{split}_y"]
            prediction = evidence.sum(1).argmax(1)
            native["splits"][split] = {
                **metrics(y, prediction),
                "n_samples": len(y),
            }
            for user in np.unique(arrays[f"{split}_users"]):
                selected = arrays[f"{split}_users"] == user
                native["users"].append({
                    "split": split,
                    "user": str(user),
                    "n": int(selected.sum()),
                    **metrics(y[selected], prediction[selected]),
                })
    native["train_test_gap"] = (
        native["splits"]["train"]["ba"] - native["splits"]["test"]["ba"]
    )
    return traces, native


def run_phase1_5(config: Config, spec: ExpSpec) -> dict[str, Any]:
    if spec.case not in GATED_CASES:
        raise ValueError(spec.case)
    p, lock, arrays = _load_core(config)
    run_directory = config.results_dir / "runs" / spec.key
    payload = torch.load(
        run_directory / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = _build_model(config, spec, p, lock)
    model.load_state_dict(payload["model_state_dict"], strict=True)

    root = config.results_dir / "phase1_5" / spec.key
    root.mkdir(parents=True, exist_ok=True)
    for intervention in INTERVENTIONS:
        shuffle_seeds = A3_SHUFFLE_SEEDS if intervention == "A3" else (-1,)
        for shuffle_seed in shuffle_seeds:
            tag = intervention if shuffle_seed < 0 else f"{intervention}__shuffle{shuffle_seed}"
            directory = root / tag
            directory.mkdir(parents=True, exist_ok=True)
            traces, native = _extract_intervention(
                model,
                arrays,
                p,
                spec.seed,
                intervention,
                shuffle_seed,
            )
            save_json(directory / "native.json", native)
            save_npz(directory / "traces.npz", traces)
            _run_l2_probes(
                directory,
                traces,
                arrays,
                ExpSpec(f"{spec.case}_{tag}", spec.seed),
                p,
            )
    return {
        "status": "PASS",
        "run": spec.key,
        "interventions": list(INTERVENTIONS),
    }


def _probe_seed_means(frame: pd.DataFrame) -> pd.DataFrame:
    groups = ["case", "seed", "layer", "state", "aggregation", "decoder"]
    metrics_columns = [
        column for column in frame.columns
        if column.endswith(("_ba", "_accuracy", "_macro_f1", "_gap"))
    ]
    return frame.groupby(
        groups, as_index=False, dropna=False
    )[metrics_columns].mean()


def _probe_gains(seed_means: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["case", "seed", "layer", "state", "decoder"]
    for key, group in seed_means.groupby(groups, dropna=False, sort=False):
        lookup = group.set_index("aggregation")
        for name, left, right in (
            ("G_resolved", "fixed250_ordered", "whole_count"),
            ("G_order", "fixed250_ordered", "fixed250_shuffled"),
            ("G_relative_order", "relative10_ordered", "relative10_shuffled"),
            ("Gap_rel10_whole", "relative10_ordered", "whole_count"),
        ):
            if left in lookup.index and right in lookup.index:
                rows.append({
                    **dict(zip(groups, key)),
                    "gain": name,
                    **{
                        f"{split}_delta": float(
                            lookup.loc[left, f"{split}_ba"]
                            - lookup.loc[right, f"{split}_ba"]
                        )
                        for split in SPLITS
                    },
                })
    return pd.DataFrame(rows)


def finalize(config: Config) -> dict[str, Any]:
    _, lock, _ = _load_core(config)
    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)

    native_rows: list[dict[str, Any]] = []
    per_user_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    gate_phase_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        core_directory = config.core_results_dir / "runs" / f"O0__seed{seed}"
        core_native = json.loads(
            (core_directory / "native.json").read_text(encoding="utf-8")
        )
        baseline_row: dict[str, Any] = {
            "case": "B0",
            "seed": seed,
            "train_test_gap": core_native["train_test_gap"],
        }
        for split in SPLITS:
            for name, value in core_native["splits"][split].items():
                baseline_row[f"{split}_{name}"] = value
        native_rows.append(baseline_row)
        per_user_rows.extend(
            {"case": "B0", "seed": seed, **item}
            for item in core_native["users"]
        )
        core_probes = json.loads(
            (core_directory / "probes.json").read_text(encoding="utf-8")
        )["rows"]
        probe_rows.extend(
            {
                **item,
                "case": "B0",
            }
            for item in core_probes
            if item["layer"] == "L2" and item["state"] == "spike"
        )

    for spec in phase1_specs():
        directory = config.results_dir / "runs" / spec.key
        native = json.loads((directory / "native.json").read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "case": spec.case,
            "seed": spec.seed,
            "train_test_gap": native["train_test_gap"],
        }
        for split in SPLITS:
            for name, value in native["splits"][split].items():
                row[f"{split}_{name}"] = value
        native_rows.append(row)
        per_user_rows.extend(
            {"case": spec.case, "seed": spec.seed, **item}
            for item in native["users"]
        )
        probe_rows.extend(
            json.loads((directory / "probes.json").read_text(encoding="utf-8"))["rows"]
        )
        if (directory / "gate_summary.json").exists():
            gate = json.loads(
                (directory / "gate_summary.json").read_text(encoding="utf-8")
            )
            gate_rows.extend(
                {"case": spec.case, "seed": spec.seed, **item}
                for item in gate["summary"]
            )
            gate_phase_rows.extend(
                {"case": spec.case, "seed": spec.seed, **item}
                for item in gate["phase10"]
            )

    native_frame = pd.DataFrame(native_rows)
    native_frame.to_csv(aggregate / "phase1_native_runs.csv", index=False)
    pd.DataFrame(per_user_rows).to_csv(
        aggregate / "phase1_native_per_user.csv", index=False
    )
    probe_frame = pd.DataFrame(probe_rows)
    probe_frame.to_csv(aggregate / "phase1_probe_runs.csv", index=False)
    probe_seed_means = _probe_seed_means(probe_frame)
    probe_seed_means.to_csv(
        aggregate / "phase1_probe_seed_means.csv", index=False
    )
    _probe_gains(probe_seed_means).to_csv(
        aggregate / "phase1_probe_gains.csv", index=False
    )
    pd.DataFrame(gate_rows).to_csv(
        aggregate / "phase1_gate_summary.csv", index=False
    )
    pd.DataFrame(gate_phase_rows).to_csv(
        aggregate / "phase1_gate_phase10.csv", index=False
    )

    contrasts: list[dict[str, Any]] = []
    for seed in SEEDS:
        current = native_frame[native_frame.seed == seed].set_index("case")
        core_native = json.loads(
            (
                config.core_results_dir
                / "runs"
                / f"O0__seed{seed}"
                / "native.json"
            ).read_text(encoding="utf-8")
        )
        for contrast, left, right in (
            ("GF_minus_B0", "GF", "B0"),
            ("GJ_minus_C0", "GJ", "C0"),
            ("GF_minus_C0", "GF", "C0"),
        ):
            deltas: dict[str, float] = {}
            for split in SPLITS:
                right_value = (
                    core_native["splits"][split]["ba"]
                    if right == "B0"
                    else float(current.loc[right, f"{split}_ba"])
                )
                deltas[f"{split}_delta"] = (
                    float(current.loc[left, f"{split}_ba"]) - right_value
                )
            contrasts.append({
                "seed": seed,
                "contrast": contrast,
                **deltas,
            })
    pd.DataFrame(contrasts).to_csv(
        aggregate / "phase1_paired_contrasts.csv", index=False
    )

    intervention_native_rows: list[dict[str, Any]] = []
    intervention_probe_rows: list[dict[str, Any]] = []
    for spec in phase1_5_specs():
        root = config.results_dir / "phase1_5" / spec.key
        for intervention in INTERVENTIONS:
            tags = (
                [intervention]
                if intervention != "A3"
                else [f"A3__shuffle{seed}" for seed in A3_SHUFFLE_SEEDS]
            )
            for tag in tags:
                directory = root / tag
                native = json.loads(
                    (directory / "native.json").read_text(encoding="utf-8")
                )
                row: dict[str, Any] = {
                    "case": spec.case,
                    "seed": spec.seed,
                    "intervention": intervention,
                    "replicate": tag,
                    "train_test_gap": native["train_test_gap"],
                }
                for split in SPLITS:
                    for name, value in native["splits"][split].items():
                        row[f"{split}_{name}"] = value
                intervention_native_rows.append(row)
                for probe_row in json.loads(
                    (directory / "probes.json").read_text(encoding="utf-8")
                )["rows"]:
                    intervention_probe_rows.append({
                        "parent_case": spec.case,
                        "intervention": intervention,
                        "replicate": tag,
                        **probe_row,
                    })

    intervention_native = pd.DataFrame(intervention_native_rows)
    intervention_native.to_csv(
        aggregate / "phase1_5_native_runs.csv", index=False
    )
    intervention_seed_means = intervention_native.groupby(
        ["case", "seed", "intervention"], as_index=False
    ).mean(numeric_only=True)
    intervention_seed_means.to_csv(
        aggregate / "phase1_5_native_seed_means.csv", index=False
    )
    intervention_deltas: list[dict[str, Any]] = []
    for (case, seed), group in intervention_seed_means.groupby(
        ["case", "seed"], sort=False
    ):
        lookup = group.set_index("intervention")
        if "A0" not in lookup.index:
            continue
        for intervention in ("A1", "A2", "A3", "A4", "A4b"):
            if intervention not in lookup.index:
                continue
            intervention_deltas.append({
                "case": case,
                "seed": seed,
                "contrast": f"A0_minus_{intervention}",
                **{
                    f"{split}_delta": float(
                        lookup.loc["A0", f"{split}_ba"]
                        - lookup.loc[intervention, f"{split}_ba"]
                    )
                    for split in SPLITS
                },
            })
    pd.DataFrame(intervention_deltas).to_csv(
        aggregate / "phase1_5_native_deltas.csv", index=False
    )
    intervention_probe = pd.DataFrame(intervention_probe_rows)
    intervention_probe.to_csv(
        aggregate / "phase1_5_probe_runs.csv", index=False
    )
    intervention_probe_seed_means = intervention_probe.groupby(
        ["parent_case", "seed", "intervention", "layer", "state", "aggregation", "decoder"],
        as_index=False,
        dropna=False,
    )[[column for column in intervention_probe.columns if column.endswith(
        ("_ba", "_accuracy", "_macro_f1", "_gap")
    )]].mean()
    intervention_probe_seed_means.to_csv(
        aggregate / "phase1_5_probe_seed_means.csv", index=False
    )

    report = {
        "status": "PASS",
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "phase1_runs": len(phase1_specs()),
        "phase1_5_tasks": len(phase1_5_specs()),
    }
    save_json(aggregate / "manifest.json", report)
    return report


def configure_cpu() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("plan")
    sub.add_parser("finalize")
    phase1 = sub.add_parser("train-phase1")
    phase1.add_argument("--task-id", type=int, required=True)
    phase15 = sub.add_parser("phase1-5")
    phase15.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args(argv)

    configure_cpu()
    config = config_from_args(args)
    if args.command == "prepare":
        print(json.dumps(prepare(config), indent=2))
    elif args.command == "plan":
        print(json.dumps({
            "phase1": [asdict(spec) | {"key": spec.key} for spec in phase1_specs()],
            "phase1_5": [
                asdict(spec) | {"key": spec.key} for spec in phase1_5_specs()
            ],
        }, indent=2))
    elif args.command == "train-phase1":
        specs = phase1_specs()
        if not 0 <= args.task_id < len(specs):
            parser.error(f"task-id must be in [0, {len(specs) - 1}]")
        print(json.dumps(train_one(config, specs[args.task_id]), indent=2))
    elif args.command == "phase1-5":
        specs = phase1_5_specs()
        if not 0 <= args.task_id < len(specs):
            parser.error(f"task-id must be in [0, {len(specs) - 1}]")
        print(json.dumps(run_phase1_5(config, specs[args.task_id]), indent=2))
    elif args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))


if __name__ == "__main__":
    main()
