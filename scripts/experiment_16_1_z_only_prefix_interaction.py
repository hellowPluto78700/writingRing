"""Exp16.1: single-node Prefix x remember-gate interaction study."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from core_benchmark_v1.data import loader
from core_benchmark_v1.model import BenchmarkNet, lif_step, sequence_loss
from core_benchmark_v1.protocol import Protocol, Run, SPLITS
from core_benchmark_v1.storage import save_json, save_torch
from core_benchmark_v1.training import cpu_state, metrics
from scripts import experiment_14_history_organization as exp14
from scripts import experiment_16_prefix_supervised_selective_memory as exp16


EXPERIMENT_ID = "experiment_16_1_z_only_prefix_interaction"
PROTOCOL_VERSION = "z_only_prefix_interaction_v1"
SEEDS = (11, 23, 37)
PREFIX_LAMBDAS = (0.01, 0.03, 0.05)
GATE_INIT_Q = 0.90
SHIFTS = ((2, 3, 4), (2, 3, 4))
GRADIENT_EPOCHS = (0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
FUNCTIONAL_ABLATIONS = ("learned", "gate_one", "gate_time_mean", "gate_time_shuffle")
SHUFFLE_SEEDS = (101, 211, 307)
MODULE_NAME = "scripts.experiment_16_1_z_only_prefix_interaction"


def lambda_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def prefix_case(prefix: str, value: float) -> str:
    return f"{prefix}_lp{lambda_tag(value)}"


@dataclass(frozen=True)
class ExpSpec:
    case: str
    seed: int
    gate_mode: str
    lambda_prefix: float

    @property
    def key(self) -> str:
        return f"{self.case}__seed{self.seed}"

    @property
    def gated(self) -> bool:
        return self.gate_mode in ("z_only", "z_history")


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
        (args.core_results or root / exp16.CORE_RESULTS_REL).resolve(),
    )


def specs() -> list[ExpSpec]:
    rows: list[ExpSpec] = []
    for seed in SEEDS:
        rows.append(ExpSpec("C0", seed, "none", 0.0))
        rows.extend(
            ExpSpec(prefix_case("P", value), seed, "none", value)
            for value in PREFIX_LAMBDAS
        )
        rows.append(ExpSpec("GZ0", seed, "z_only", 0.0))
        rows.extend(
            ExpSpec(prefix_case("GPZ", value), seed, "z_only", value)
            for value in PREFIX_LAMBDAS
        )
        rows.append(ExpSpec("GZH0", seed, "z_history", 0.0))
        rows.extend(
            ExpSpec(prefix_case("GPZH", value), seed, "z_history", value)
            for value in PREFIX_LAMBDAS
        )
    return rows


def gated_specs() -> list[ExpSpec]:
    return [spec for spec in specs() if spec.gated]


def _run(spec: ExpSpec) -> Run:
    return Run(spec.case, spec.seed, "16_1_prefix_gate", shifts=SHIFTS, objective="wcce")


def prepare(config: Config) -> dict[str, Any]:
    p, lock, _ = exp16._load_core(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    allocation = {
        "hostname": platform.node(),
        "processor": platform.processor(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "core_dataset_hash": lock["dataset_hash"],
        "seeds": list(SEEDS),
        "prefix_lambdas": list(PREFIX_LAMBDAS),
        "prefix_definition": {
            "phases": list(exp16.PREFIX_PHASES),
            "weights": list(exp16.PREFIX_WEIGHTS),
        },
        "checkpoint_selection": "native validation BA, then validation CE, then earliest epoch",
        "gate_init_q": GATE_INIT_Q,
        "gate_modes": {
            "z_only": "g_t=sigmoid(G_z z_t+b_g)",
            "z_history": "g_t=sigmoid(G_z z_t+G_h h_{t-1}+b_g)",
        },
        "task_count": len(specs()),
        "gated_task_count": len(gated_specs()),
        "allocation": allocation,
        "single_node_required": True,
        "test_metrics_never_used_for_training_or_checkpoint_selection": True,
        "width": p.width,
        "tau_mem_ms": p.tau_mem_ms,
    }
    save_json(config.results_dir / "protocol.json", payload)
    save_json(config.results_dir / "allocation.json", allocation)
    return payload


class ContextGateNet(BenchmarkNet):
    """Scalar suppressive gate with optional explicit previous-L2 history."""

    def __init__(self, run: Run, p: Protocol, gate_mode: str) -> None:
        if gate_mode not in ("z_only", "z_history"):
            raise ValueError(gate_mode)
        super().__init__(run, p)
        self.gate_mode = gate_mode
        self.gate_input = nn.Linear(p.width, 1, bias=False)
        self.gate_history = (
            nn.Linear(p.width, 1, bias=False) if gate_mode == "z_history" else None
        )
        self.gate_bias = nn.Parameter(
            torch.tensor(math.log(GATE_INIT_Q / (1.0 - GATE_INIT_Q)), dtype=torch.float32)
        )
        with torch.no_grad():
            self.gate_input.weight.zero_()
            if self.gate_history is not None:
                self.gate_history.weight.zero_()

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        *,
        forced_g: torch.Tensor | None = None,
        reset_at: torch.Tensor | None = None,
        reset_layers: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        p = self.protocol
        batch, steps, channels = x.shape
        if channels != p.input_channels or lengths.shape != (batch,):
            raise ValueError("Invalid input geometry")
        if forced_g is not None and forced_g.shape != (batch, steps):
            raise ValueError("forced_g must be [batch, steps]")
        if reset_at is not None and reset_at.shape != lengths.shape:
            raise ValueError("reset_at must be one boundary per sample")

        syn = [x.new_zeros(batch, p.width) for _ in self.layers]
        mem = [x.new_zeros(batch, p.width) for _ in self.layers]
        spikes: list[list[torch.Tensor]] = [[], []]
        pre_reset: list[list[torch.Tensor]] = [[], []]
        evidence: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        prev_l2 = x.new_zeros(batch, p.width)

        for t in range(int(lengths.max().item())):
            active = (t < lengths)[:, None]
            cur = x[:, t] * active
            if reset_at is not None:
                retain = (reset_at != t)[:, None]
                if 0 in reset_layers:
                    syn[0], mem[0] = syn[0] * retain, mem[0] * retain
                if 1 in reset_layers:
                    syn[1], mem[1] = syn[1] * retain, mem[1] * retain
                    prev_l2 = prev_l2 * retain

            candidate0 = self.alpha_0 * syn[0] + self.layers[0](cur)
            syn[0] = torch.where(active, candidate0, syn[0])
            s1, new_mem1, pre1 = lif_step(
                syn[0], mem[0], self.betas[0], p.threshold, p.surrogate_slope
            )
            mem[0] = torch.where(active, new_mem1, mem[0])
            s1 = s1 * active

            gate_logit = self.gate_input(s1).squeeze(-1) + self.gate_bias
            if self.gate_history is not None:
                gate_logit = gate_logit + self.gate_history(prev_l2).squeeze(-1)
            gate = torch.sigmoid(gate_logit)
            used_gate = gate if forced_g is None else forced_g[:, t]

            candidate1 = self.alpha_1 * syn[1] + self.layers[1](s1) * used_gate[:, None]
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
            gates.append(gate * active.squeeze(1))

        def stack3(values: list[torch.Tensor]) -> torch.Tensor:
            out = torch.stack(values, 1)
            return F.pad(out, (0, 0, 0, steps - out.shape[1]))

        def stack2(values: list[torch.Tensor]) -> torch.Tensor:
            out = torch.stack(values, 1)
            return F.pad(out, (0, steps - out.shape[1]))

        return {
            "spike": (stack3(spikes[0]), stack3(spikes[1])),
            "pre_reset": (stack3(pre_reset[0]), stack3(pre_reset[1])),
            "evidence": stack3(evidence),
            "final_syn": tuple(syn),
            "final_mem": tuple(mem),
            "gate_g": stack2(gates),
        }


def _make_model(spec: ExpSpec, p: Protocol) -> nn.Module:
    if spec.gated:
        return ContextGateNet(_run(spec), p, spec.gate_mode)
    return BenchmarkNet(_run(spec), p)


def _make_optimizer(model: nn.Module, p: Protocol) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=p.learning_rate,
        weight_decay=p.weight_decay,
    )


def _parameter_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    named = dict(model.named_parameters())
    groups = {
        "R": [named["head.weight"]],
        "W_L1": [named["layers.0.weight"]],
        "W_L2": [named["layers.1.weight"]],
    }
    if isinstance(model, ContextGateNet):
        groups["G_z"] = [named["gate_input.weight"]]
        if model.gate_history is not None:
            groups["G_h"] = [named["gate_history.weight"]]
        groups["G_b"] = [named["gate_bias"]]
    return groups


def _grad_vector(
    loss: torch.Tensor,
    params: list[nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
    chunks = []
    for parameter, gradient in zip(params, gradients):
        chunks.append(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
        )
    return torch.cat(chunks) if chunks else torch.zeros(1)


def _gradient_geometry(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    spec: ExpSpec,
    epoch: int,
) -> list[dict[str, Any]]:
    if spec.lambda_prefix <= 0:
        return []
    x, y, lengths = next(iter(loader(arrays, "train", p, spec.seed, shuffle=False)))
    model.train()
    evidence = model(x, lengths)["evidence"]
    wcce = sequence_loss(evidence, lengths, y, "wcce")
    prefix = exp16.prefix_wcce(evidence, lengths, y)
    groups = _parameter_groups(model)
    rows: list[dict[str, Any]] = []
    for index, (name, params) in enumerate(groups.items()):
        gw = _grad_vector(wcce, params, retain_graph=True)
        gp = _grad_vector(prefix, params, retain_graph=index < len(groups) - 1)
        nw, np_ = float(gw.norm()), float(gp.norm())
        cosine = (
            float(F.cosine_similarity(gw[None], gp[None]).item())
            if nw > 0 and np_ > 0
            else float("nan")
        )
        rows.append(
            {
                "case": spec.case,
                "seed": spec.seed,
                "gate_mode": spec.gate_mode,
                "lambda_prefix": spec.lambda_prefix,
                "epoch": epoch,
                "group": name,
                "wcce_grad_norm": nw,
                "prefix_grad_norm": np_,
                "weighted_ratio": spec.lambda_prefix * np_ / (nw + 1e-12),
                "cosine": cosine,
            }
        )
    model.zero_grad(set_to_none=True)
    return rows


def _relative_updates(
    before: dict[str, torch.Tensor],
    model: nn.Module,
    epoch: int,
) -> list[dict[str, Any]]:
    rows = []
    for group, params in _parameter_groups(model).items():
        ids = {id(parameter) for parameter in params}
        numerator = 0.0
        denominator = 0.0
        for name, parameter in model.named_parameters():
            if id(parameter) not in ids:
                continue
            numerator += float((parameter.detach() - before[name]).square().sum())
            denominator += float(before[name].square().sum())
        rows.append(
            {
                "epoch": epoch,
                "group": group,
                "relative_update_norm": math.sqrt(numerator) / (math.sqrt(denominator) + 1e-12),
            }
        )
    return rows


def train_one(config: Config, spec: ExpSpec) -> dict[str, Any]:
    p, _, arrays = exp16._load_core(config)
    directory = config.results_dir / "runs" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    if checkpoint_path.exists():
        return {"status": "exists", "run": spec.key}

    model = _make_model(spec, p)
    optimizer = _make_optimizer(model, p)
    train_loader = loader(arrays, "train", p, spec.seed, shuffle=True)
    best = exp16._split_eval(model, arrays, p, spec.seed, "val")
    best_state = cpu_state(model)
    best_epoch = 0
    history = [
        {
            "epoch": 0,
            "train_loss": None,
            "val_ba": best["ba"],
            "val_mean_logit_ce": best["mean_logit_ce"],
        }
    ]
    gradients = _gradient_geometry(model, arrays, p, spec, 0)
    updates: list[dict[str, Any]] = []

    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        diagnostic_batch = epoch in GRADIENT_EPOCHS
        first_batch = True
        for x, y, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            evidence = model(x, lengths)["evidence"]
            wcce = sequence_loss(evidence, lengths, y, "wcce")
            prefix = exp16.prefix_wcce(evidence, lengths, y)
            loss = wcce + spec.lambda_prefix * prefix
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite loss")
            before = None
            if diagnostic_batch and first_batch:
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"{spec.key}: nonfinite gradient")
            optimizer.step()
            if before is not None:
                updates.extend(_relative_updates(before, model, epoch))
                first_batch = False
            total += float(loss.detach()) * len(y)
            count += len(y)

        val = exp16._split_eval(model, arrays, p, spec.seed, "val")
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / count,
                "val_ba": val["ba"],
                "val_mean_logit_ce": val["mean_logit_ce"],
            }
        )
        if exp16._better(val, best, exp16.SELECTOR_BA):
            best = dict(val)
            best_state = cpu_state(model)
            best_epoch = epoch
        if spec.lambda_prefix > 0 and epoch in GRADIENT_EPOCHS:
            gradients.extend(_gradient_geometry(model, arrays, p, spec, epoch))
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    model.load_state_dict(best_state)
    save_json(directory / "train_history.json", {"rows": history})
    save_json(directory / "gradient_geometry.json", {"rows": gradients})
    save_json(directory / "optimizer_update_diagnostics.json", {"rows": updates})
    save_torch(
        checkpoint_path,
        {
            "experiment": EXPERIMENT_ID,
            "protocol": PROTOCOL_VERSION,
            "case": spec.case,
            "seed": spec.seed,
            "gate_mode": spec.gate_mode,
            "lambda_prefix": spec.lambda_prefix,
            "hostname": platform.node(),
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "stopped_epoch": epoch,
            "best_val": best,
            "selection_rule": "native validation BA, then validation CE, then earliest epoch",
        },
    )
    exp16.evaluate_model(config, spec, model)
    return {
        "status": "PASS",
        "run": spec.key,
        "hostname": platform.node(),
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "best_val": best,
    }


def _load_model(config: Config, spec: ExpSpec) -> tuple[nn.Module, Protocol, dict[str, np.ndarray]]:
    p, _, arrays = exp16._load_core(config)
    payload = torch.load(
        config.results_dir / "runs" / spec.key / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = _make_model(spec, p)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, p, arrays


def _forced_gate(
    model: ContextGateNet,
    x: torch.Tensor,
    lengths: torch.Tensor,
    mode: str,
    sample_ids: np.ndarray,
    shuffle_seed: int,
) -> torch.Tensor | None:
    return exp16._forced_gate(model, x, lengths, mode, sample_ids, shuffle_seed)


def run_ablation(config: Config, spec: ExpSpec) -> dict[str, Any]:
    if not spec.gated:
        raise ValueError("Functional ablation requires a gated run")
    model, p, arrays = _load_model(config, spec)
    model.eval()
    root = config.results_dir / "functional_ablation" / spec.key
    root.mkdir(parents=True, exist_ok=True)

    for mode in FUNCTIONAL_ABLATIONS:
        shuffle_seeds = SHUFFLE_SEEDS if mode == "gate_time_shuffle" else (-1,)
        for shuffle_seed in shuffle_seeds:
            tag = mode if shuffle_seed < 0 else f"{mode}__shuffle{shuffle_seed}"
            directory = root / tag
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "native.json").exists():
                continue
            native: dict[str, Any] = {"splits": {}}
            traces: dict[str, np.ndarray] = {}
            with torch.no_grad():
                for split in SPLITS:
                    evidence_chunks, l1_chunks, l2_chunks = [], [], []
                    offset = 0
                    for x, y, lengths in loader(arrays, split, p, spec.seed):
                        ids = arrays[f"{split}_ids"][offset : offset + len(y)]
                        offset += len(y)
                        forced = _forced_gate(
                            model, x, lengths, mode, ids, shuffle_seed
                        )
                        output = model(x, lengths, forced_g=forced)
                        evidence_chunks.append(output["evidence"].numpy())
                        l1_chunks.append(output["spike"][0].numpy().astype(np.uint8))
                        l2_chunks.append(output["spike"][1].numpy().astype(np.uint8))
                    evidence = np.concatenate(evidence_chunks)
                    traces[f"{split}__L1__spike"] = np.concatenate(l1_chunks)
                    traces[f"{split}__L2__spike"] = np.concatenate(l2_chunks)
                    scores = np.asarray(
                        [
                            evidence[index, :length].mean(0)
                            for index, length in enumerate(arrays[f"{split}_lengths"])
                        ]
                    )
                    prediction = scores.argmax(1)
                    native["splits"][split] = metrics(
                        arrays[f"{split}_y"], prediction
                    )
            save_json(directory / "native.json", native)
            exp16._run_probes(directory, traces, arrays, spec, p)
    return {"status": "PASS", "run": spec.key}


def _native_rows(config: Config) -> pd.DataFrame:
    rows = []
    for spec in specs():
        directory = config.results_dir / "runs" / spec.key
        native = json.loads((directory / "native.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(
            directory / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        row: dict[str, Any] = {
            "case": spec.case,
            "seed": spec.seed,
            "gate_mode": spec.gate_mode,
            "lambda_prefix": spec.lambda_prefix,
            "hostname": checkpoint["hostname"],
            "best_epoch": checkpoint["best_epoch"],
            "stopped_epoch": checkpoint["stopped_epoch"],
            "train_test_gap": native["train_test_gap"],
        }
        for split in SPLITS:
            row.update(
                {
                    f"{split}_{name}": value
                    for name, value in native["splits"][split].items()
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _probe_rows(config: Config) -> pd.DataFrame:
    rows = []
    for spec in specs():
        payload = json.loads(
            (config.results_dir / "runs" / spec.key / "probes.json").read_text(
                encoding="utf-8"
            )
        )
        rows.extend(
            {
                "gate_mode": spec.gate_mode,
                "lambda_prefix": spec.lambda_prefix,
                **row,
            }
            for row in payload["rows"]
        )
    return pd.DataFrame(rows)


def _gate_rows(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, phase_rows = [], []
    for spec in gated_specs():
        payload = json.loads(
            (config.results_dir / "runs" / spec.key / "gate_summary.json").read_text(
                encoding="utf-8"
            )
        )
        summary_rows.extend(
            {
                "case": spec.case,
                "seed": spec.seed,
                "gate_mode": spec.gate_mode,
                "lambda_prefix": spec.lambda_prefix,
                **row,
            }
            for row in payload["summary"]
        )
        phase_rows.extend(
            {
                "case": spec.case,
                "seed": spec.seed,
                "gate_mode": spec.gate_mode,
                "lambda_prefix": spec.lambda_prefix,
                **row,
            }
            for row in payload["phase10"]
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(phase_rows)


def _case_value(
    frame: pd.DataFrame,
    case: str,
    seed: int,
    column: str,
) -> float:
    selected = frame[(frame["case"] == case) & (frame["seed"] == seed)]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {case} seed {seed}, got {len(selected)}")
    return float(selected.iloc[0][column])


def _interaction_rows(
    native: pd.DataFrame,
    collapse: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for value in PREFIX_LAMBDAS:
            p_case = prefix_case("P", value)
            for gate_mode, baseline, prefix in (
                ("z_only", "GZ0", prefix_case("GPZ", value)),
                ("z_history", "GZH0", prefix_case("GPZH", value)),
            ):
                row: dict[str, Any] = {
                    "seed": seed,
                    "lambda_prefix": value,
                    "gate_mode": gate_mode,
                }
                for metric in ("val_ba", "test_ba"):
                    ungated_gain = (
                        _case_value(native, p_case, seed, metric)
                        - _case_value(native, "C0", seed, metric)
                    )
                    gated_gain = (
                        _case_value(native, prefix, seed, metric)
                        - _case_value(native, baseline, seed, metric)
                    )
                    row[f"{metric}_ungated_prefix_gain_pp"] = 100.0 * ungated_gain
                    row[f"{metric}_gated_prefix_gain_pp"] = 100.0 * gated_gain
                    row[f"{metric}_interaction_pp"] = 100.0 * (
                        gated_gain - ungated_gain
                    )
                for metric in (
                    "whole_count",
                    "relative10_ordered",
                    "collapse_gap_pp",
                ):
                    ungated_gain = (
                        _case_value(collapse, p_case, seed, metric)
                        - _case_value(collapse, "C0", seed, metric)
                    )
                    gated_gain = (
                        _case_value(collapse, prefix, seed, metric)
                        - _case_value(collapse, baseline, seed, metric)
                    )
                    scale = 1.0 if metric == "collapse_gap_pp" else 100.0
                    row[f"{metric}_interaction_pp"] = scale * (
                        gated_gain - ungated_gain
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _ablation_rows(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    native_rows, probe_rows = [], []
    for spec in gated_specs():
        root = config.results_dir / "functional_ablation" / spec.key
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            native = json.loads((directory / "native.json").read_text(encoding="utf-8"))
            row: dict[str, Any] = {
                "case": spec.case,
                "seed": spec.seed,
                "gate_mode": spec.gate_mode,
                "lambda_prefix": spec.lambda_prefix,
                "intervention": directory.name,
            }
            for split in SPLITS:
                row.update(
                    {
                        f"{split}_{name}": value
                        for name, value in native["splits"][split].items()
                    }
                )
            native_rows.append(row)
            probes = json.loads(
                (directory / "probes.json").read_text(encoding="utf-8")
            )["rows"]
            probe_rows.extend(
                {
                    "parent_case": spec.case,
                    "parent_seed": spec.seed,
                    "parent_gate_mode": spec.gate_mode,
                    "parent_lambda_prefix": spec.lambda_prefix,
                    "intervention": directory.name,
                    **probe,
                }
                for probe in probes
            )
    return pd.DataFrame(native_rows), pd.DataFrame(probe_rows)


def finalize(config: Config) -> dict[str, Any]:
    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)

    native = _native_rows(config)
    hosts = sorted(set(native["hostname"]))
    if len(hosts) != 1:
        raise AssertionError(f"Exp16.1 requires one physical node; got {hosts}")
    native.to_csv(aggregate / "native_runs.csv", index=False)

    probes = _probe_rows(config)
    probes.to_csv(aggregate / "probe_runs.csv", index=False)
    collapse = exp16._collapse_rows(probes)
    collapse.to_csv(aggregate / "collapse_gap.csv", index=False)

    gate_summary, gate_phase = _gate_rows(config)
    gate_summary.to_csv(aggregate / "gate_summary.csv", index=False)
    gate_phase.to_csv(aggregate / "gate_phase10.csv", index=False)

    interactions = _interaction_rows(native, collapse)
    interactions.to_csv(aggregate / "interactions.csv", index=False)
    interaction_summary = (
        interactions.groupby(["gate_mode", "lambda_prefix"], as_index=False)
        .agg(
            val_interaction_mean_pp=("val_ba_interaction_pp", "mean"),
            val_interaction_std_pp=("val_ba_interaction_pp", "std"),
            test_interaction_mean_pp=("test_ba_interaction_pp", "mean"),
            test_interaction_std_pp=("test_ba_interaction_pp", "std"),
            wholecount_interaction_mean_pp=("whole_count_interaction_pp", "mean"),
            relative10_interaction_mean_pp=(
                "relative10_ordered_interaction_pp",
                "mean",
            ),
            collapse_interaction_mean_pp=(
                "collapse_gap_pp_interaction_pp",
                "mean",
            ),
        )
    )
    interaction_summary.to_csv(
        aggregate / "interaction_summary.csv", index=False
    )

    gradients = []
    updates = []
    for spec in specs():
        directory = config.results_dir / "runs" / spec.key
        gradients.extend(
            {
                "case": spec.case,
                "seed": spec.seed,
                "gate_mode": spec.gate_mode,
                "lambda_prefix": spec.lambda_prefix,
                **row,
            }
            for row in json.loads(
                (directory / "gradient_geometry.json").read_text(encoding="utf-8")
            )["rows"]
        )
        updates.extend(
            {
                "case": spec.case,
                "seed": spec.seed,
                "gate_mode": spec.gate_mode,
                "lambda_prefix": spec.lambda_prefix,
                **row,
            }
            for row in json.loads(
                (directory / "optimizer_update_diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )["rows"]
        )
    pd.DataFrame(gradients).to_csv(
        aggregate / "gradient_geometry.csv", index=False
    )
    pd.DataFrame(updates).to_csv(
        aggregate / "optimizer_update_diagnostics.csv", index=False
    )

    ablation_native, ablation_probe = _ablation_rows(config)
    ablation_native.to_csv(
        aggregate / "functional_ablation_native_runs.csv", index=False
    )
    ablation_probe.to_csv(
        aggregate / "functional_ablation_probe_runs.csv", index=False
    )
    learned = ablation_native[
        ablation_native["intervention"] == "learned"
    ][["case", "seed", "test_ba"]].rename(columns={"test_ba": "learned_test_ba"})
    deltas = ablation_native.merge(learned, on=["case", "seed"], how="left")
    deltas["test_ba_delta_vs_learned_pp"] = 100.0 * (
        deltas["test_ba"] - deltas["learned_test_ba"]
    )
    deltas.to_csv(
        aggregate / "functional_ablation_native_deltas.csv", index=False
    )

    report = {
        "status": "PASS",
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "hostname": hosts[0],
        "training_runs": len(native),
        "gated_runs": len(gated_specs()),
        "interaction_rows": len(interactions),
        "prefix_lambdas": list(PREFIX_LAMBDAS),
        "single_node_verified": True,
    }
    save_json(aggregate / "manifest.json", report)
    return report


def _launch_one(
    config: Config,
    command: str,
    task_id: int,
    log_root: Path,
) -> tuple[int, int]:
    log_root.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--results",
        str(config.results_dir),
        "--core-results",
        str(config.core_results_dir),
        command,
        "--task-id",
        str(task_id),
    ]
    with (log_root / f"task_{task_id}.out").open("w", encoding="utf-8") as out:
        with (log_root / f"task_{task_id}.err").open("w", encoding="utf-8") as err:
            completed = subprocess.run(
                args, stdout=out, stderr=err, check=False
            )
    return task_id, completed.returncode


def launch(
    config: Config,
    *,
    command: str,
    count: int,
    max_workers: int,
) -> dict[str, Any]:
    stage = "train" if command == "train" else "ablation"
    log_root = config.results_dir / "launcher_logs" / stage
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _launch_one, config, command, task_id, log_root
            )
            for task_id in range(count)
        ]
        for future in as_completed(futures):
            task_id, returncode = future.result()
            if returncode:
                failures.append(
                    {"task_id": task_id, "returncode": returncode}
                )
    result = {
        "stage": stage,
        "hostname": platform.node(),
        "max_workers": max_workers,
        "task_count": count,
        "failures": failures,
    }
    save_json(config.results_dir / f"launcher_{stage}.json", result)
    if failures:
        raise RuntimeError(f"{stage} failures: {failures}")
    return result


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
    train = sub.add_parser("train")
    train.add_argument("--task-id", type=int, required=True)
    ablate = sub.add_parser("ablation")
    ablate.add_argument("--task-id", type=int, required=True)
    launch_train = sub.add_parser("launch-train")
    launch_train.add_argument("--max-workers", type=int, default=30)
    launch_ablate = sub.add_parser("launch-ablation")
    launch_ablate.add_argument("--max-workers", type=int, default=30)
    sub.add_parser("finalize")
    args = parser.parse_args(argv)

    configure_cpu()
    config = config_from_args(args)
    if args.command == "prepare":
        print(json.dumps(prepare(config), indent=2))
    elif args.command == "plan":
        print(
            json.dumps(
                {
                    "training": [
                        asdict(spec) | {"key": spec.key}
                        for spec in specs()
                    ],
                    "training_count": len(specs()),
                    "ablation_count": len(gated_specs()),
                },
                indent=2,
            )
        )
    elif args.command == "train":
        if not 0 <= args.task_id < len(specs()):
            parser.error(
                f"task-id must be in [0, {len(specs()) - 1}]"
            )
        print(
            json.dumps(train_one(config, specs()[args.task_id]), indent=2)
        )
    elif args.command == "ablation":
        if not 0 <= args.task_id < len(gated_specs()):
            parser.error(
                f"task-id must be in [0, {len(gated_specs()) - 1}]"
            )
        print(
            json.dumps(
                run_ablation(config, gated_specs()[args.task_id]), indent=2
            )
        )
    elif args.command == "launch-train":
        print(
            json.dumps(
                launch(
                    config,
                    command="train",
                    count=len(specs()),
                    max_workers=args.max_workers,
                ),
                indent=2,
            )
        )
    elif args.command == "launch-ablation":
        print(
            json.dumps(
                launch(
                    config,
                    command="ablation",
                    count=len(gated_specs()),
                    max_workers=args.max_workers,
                ),
                indent=2,
            )
        )
    elif args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))


if __name__ == "__main__":
    main()

