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
from core_benchmark_v1.storage import file_hash, save_json, save_npz, save_torch
from core_benchmark_v1.training import cpu_state, metrics
from scripts import experiment_14_history_organization as exp14


EXPERIMENT_ID = "experiment_16_prefix_supervised_selective_memory"
PROTOCOL_VERSION = "prefix_selective_memory_v1"
CORE_RESULTS_REL = Path("core_benchmark_v1/results/main")
SEEDS = (11, 23, 37)
SHIFTS = ((2, 3, 4), (2, 3, 4))

PREFIX_PHASES = (0.50, 0.75)
PREFIX_WEIGHTS = (0.5, 0.5)
PREFIX_LAMBDAS = (0.0, 0.01, 0.03, 0.05, 0.10)
LOOKAHEAD_EPOCHS = (10, 20, 30, 50)
LOOKAHEAD_STEPS_EPOCHS = 1
VAL_CE_MIN_IMPROVEMENT = 0.005
VAL_BA_RETENTION_PP = 0.5
MIN_SEED_SUPPORT = 2

SELECTOR_BA = "ba_first"
SELECTOR_CE = "ce_min"
PHASE1_CASES = ("C0", "P0", "G0", "GP_J")
PHASE15_CASES = ("GP_stopR", "GP_gate", "GP_noGate")
GATED_CASES = ("G0", "GP_J", *PHASE15_CASES)
PREFIX_CASES = ("P0", "GP_J", *PHASE15_CASES)
PREFIX_ROUTES = {
    "P0": "joint",
    "GP_J": "joint",
    "GP_stopR": "stopR",
    "GP_gate": "gate_only",
    "GP_noGate": "no_gate",
}
GATE_INIT_Q = 0.90
GRADIENT_EPOCHS = (0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
FUNCTIONAL_ABLATIONS = ("learned", "gate_one", "gate_time_mean", "gate_time_shuffle")
SHUFFLE_SEEDS = (101, 211, 307)


@dataclass(frozen=True)
class ExpSpec:
    case: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.case}__seed{self.seed}"


@dataclass(frozen=True)
class LookaheadSpec:
    seed: int
    base_epoch: int
    lambda_prefix: float

    @property
    def key(self) -> str:
        value = f"{self.lambda_prefix:g}".replace(".", "p")
        return f"seed{self.seed}__epoch{self.base_epoch}__lambda{value}"


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


def phase0a_specs() -> list[ExpSpec]:
    return [ExpSpec("C0_trajectory", seed) for seed in SEEDS]


def phase0b_specs() -> list[LookaheadSpec]:
    return [
        LookaheadSpec(seed, epoch, value)
        for seed in SEEDS
        for epoch in LOOKAHEAD_EPOCHS
        for value in PREFIX_LAMBDAS
    ]


def phase1_specs() -> list[ExpSpec]:
    return [ExpSpec(case, seed) for case in PHASE1_CASES for seed in SEEDS]


def phase15_specs() -> list[ExpSpec]:
    return [ExpSpec(case, seed) for case in PHASE15_CASES for seed in SEEDS]


def _run(spec: ExpSpec) -> Run:
    return Run(spec.case, spec.seed, "16_prefix_selective_memory", shifts=SHIFTS, objective="wcce")


def _load_core(config: Config) -> tuple[Protocol, dict[str, Any], dict[str, np.ndarray]]:
    lock = json.loads((config.core_results_dir / "protocol.lock.json").read_text(encoding="utf-8"))
    payload = dict(lock["protocol"])
    payload["version"] = Protocol().version
    p = Protocol.from_dict(payload)
    if tuple(p.seeds) != SEEDS:
        raise ValueError(f"CoreBenchmark seeds changed: {p.seeds}")
    if (p.width, p.fs, p.input_channels, p.tau_mem_ms, p.batch_size) != (128, 64.0, 30, 22.54, 128):
        raise ValueError("CoreBenchmark geometry differs from Exp16 contract")
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
        raise ValueError(f"CoreBenchmark split differs from Exp16 contract: {actual}")
    return p, lock, load_cache(config.core_results_dir, p, lock)


def prepare(config: Config) -> dict[str, Any]:
    p, lock, _ = _load_core(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "core_dataset_hash": lock["dataset_hash"],
        "seeds": list(SEEDS),
        "shifts": [list(v) for v in SHIFTS],
        "width": p.width,
        "tau_mem_ms": p.tau_mem_ms,
        "phase0a": {
            "purpose": "same WCCE trajectory; compare BA-first versus minimum-validation-CE checkpoint selection",
            "tasks": len(phase0a_specs()),
            "selectors": [SELECTOR_BA, SELECTOR_CE],
        },
        "phase0b": {
            "purpose": "validation-only Prefix coefficient calibration by one-epoch lookahead",
            "prefix_phases": list(PREFIX_PHASES),
            "prefix_weights": list(PREFIX_WEIGHTS),
            "lambda_candidates": list(PREFIX_LAMBDAS),
            "base_epochs": list(LOOKAHEAD_EPOCHS),
            "tasks": len(phase0b_specs()),
            "val_ce_min_improvement": VAL_CE_MIN_IMPROVEMENT,
            "val_ba_retention_pp": VAL_BA_RETENTION_PP,
            "min_seed_support": MIN_SEED_SUPPORT,
        },
        "phase1_cases": list(PHASE1_CASES),
        "phase1_tasks": len(phase1_specs()),
        "phase1_5_cases": list(PHASE15_CASES),
        "phase1_5_tasks": len(phase15_specs()),
        "gate": {
            "geometry": "Exp15 scalar context-dependent gate",
            "equation": "g_t=sigmoid(G_z z_t + G_h h_{t-1} + b); L2 input=current*g_t",
            "suppressive_only": True,
            "initial_g": GATE_INIT_Q,
            "no_g_over_q_amplification": True,
        },
        "prefix_routes": PREFIX_ROUTES,
        "functional_ablations": list(FUNCTIONAL_ABLATIONS),
        "selection_leakage_rule": "Phase0 selector/lambda decisions use validation only; test is never read by selection functions.",
    }
    payload["identity"] = digest(payload)
    save_json(config.results_dir / "protocol.json", payload)
    return payload


def _prefix_means(evidence: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cumulative = evidence.cumsum(1)
    batch = torch.arange(evidence.shape[0], device=evidence.device)
    means = []
    for phase in PREFIX_PHASES:
        count = torch.ceil(lengths.float() * phase).long().clamp_min(1)
        means.append(cumulative[batch, count - 1] / count.to(evidence.dtype).unsqueeze(1))
    return means[0], means[1]


def prefix_wcce(evidence: torch.Tensor, lengths: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    a50, a75 = _prefix_means(evidence, lengths)
    return PREFIX_WEIGHTS[0] * F.cross_entropy(a50, y) + PREFIX_WEIGHTS[1] * F.cross_entropy(a75, y)


class SuppressiveContextWriteGateNet(BenchmarkNet):
    """Exp15 scalar context gate with true suppressive multiplier g in (0,1)."""

    def __init__(self, run: Run, p: Protocol) -> None:
        super().__init__(run, p)
        self.gate_input = nn.Linear(p.width, 1, bias=False)
        self.gate_history = nn.Linear(p.width, 1, bias=False)
        self.gate_bias = nn.Parameter(torch.tensor(math.log(GATE_INIT_Q / (1 - GATE_INIT_Q)), dtype=torch.float32))
        with torch.no_grad():
            self.gate_input.weight.zero_()
            self.gate_history.weight.zero_()

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        *,
        forced_g: torch.Tensor | None = None,
        capture_gate_grad: bool = False,
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
        gate_nodes: list[torch.Tensor] = []
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
            s1, new_mem1, pre1 = lif_step(syn[0], mem[0], self.betas[0], p.threshold, p.surrogate_slope)
            mem[0] = torch.where(active, new_mem1, mem[0])
            s1 = s1 * active

            a = self.gate_input(s1).squeeze(-1) + self.gate_history(prev_l2).squeeze(-1) + self.gate_bias
            if capture_gate_grad:
                a.retain_grad()
            g = torch.sigmoid(a)
            used_g = g if forced_g is None else forced_g[:, t]
            candidate1 = self.alpha_1 * syn[1] + self.layers[1](s1) * used_g[:, None]
            syn[1] = torch.where(active, candidate1, syn[1])
            s2, new_mem2, pre2 = lif_step(syn[1], mem[1], self.betas[1], p.threshold, p.surrogate_slope)
            mem[1] = torch.where(active, new_mem2, mem[1])
            s2 = s2 * active
            prev_l2 = s2

            spikes[0].append(s1)
            spikes[1].append(s2)
            pre_reset[0].append(pre1 * active)
            pre_reset[1].append(pre2 * active)
            evidence.append(self.head(s2))
            gates.append(g * active.squeeze(1))
            gate_nodes.append(a)

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
            "gate_a_nodes": gate_nodes if capture_gate_grad else None,
        }


def _make_model(spec: ExpSpec, p: Protocol) -> nn.Module:
    if spec.case in GATED_CASES:
        return SuppressiveContextWriteGateNet(_run(spec), p)
    return BenchmarkNet(_run(spec), p)


def _make_optimizer(model: nn.Module, p: Protocol) -> torch.optim.Optimizer:
    named = [(n, v) for n, v in model.named_parameters() if v.requires_grad]
    if isinstance(model, SuppressiveContextWriteGateNet):
        bias = [v for n, v in named if n == "gate_bias"]
        other = [v for n, v in named if n != "gate_bias"]
        groups: list[dict[str, Any]] = []
        if other:
            groups.append({"params": other, "weight_decay": p.weight_decay})
        if bias:
            groups.append({"params": bias, "weight_decay": 0.0})
        return torch.optim.Adam(groups, lr=p.learning_rate)
    return torch.optim.Adam((v for _, v in named), lr=p.learning_rate, weight_decay=p.weight_decay)


def _split_eval(model: nn.Module, arrays: dict[str, np.ndarray], p: Protocol, seed: int, split: str) -> dict[str, float]:
    model.eval()
    targets, predictions = [], []
    ce = 0.0
    n = 0
    with torch.no_grad():
        for x, y, lengths in loader(arrays, split, p, seed):
            scores = mean_logits(model(x, lengths)["evidence"], lengths)
            targets.append(y.numpy())
            predictions.append(scores.argmax(1).numpy())
            ce += float(F.cross_entropy(scores, y, reduction="sum"))
            n += len(y)
    y = np.concatenate(targets)
    pred = np.concatenate(predictions)
    out = metrics(y, pred)
    out["mean_logit_ce"] = ce / n
    return out


def _better(candidate: dict[str, float], best: dict[str, float], selector: str) -> bool:
    if selector == SELECTOR_CE:
        return candidate["mean_logit_ce"] < best["mean_logit_ce"] - 1e-12 or (
            abs(candidate["mean_logit_ce"] - best["mean_logit_ce"]) <= 1e-12
            and candidate["ba"] > best["ba"] + 1e-12
        )
    if selector == SELECTOR_BA:
        return candidate["ba"] > best["ba"] + 1e-12 or (
            abs(candidate["ba"] - best["ba"]) <= 1e-12
            and candidate["mean_logit_ce"] < best["mean_logit_ce"] - 1e-12
        )
    raise ValueError(selector)


def train_phase0a(config: Config, spec: ExpSpec) -> dict[str, Any]:
    p, _, arrays = _load_core(config)
    root = config.results_dir / "phase0a" / f"seed{spec.seed}"
    root.mkdir(parents=True, exist_ok=True)
    done = root / "trajectory.json"
    if done.exists():
        return {"status": "exists", "seed": spec.seed}

    model = BenchmarkNet(_run(ExpSpec("C0", spec.seed)), p)
    optimizer = _make_optimizer(model, p)
    train_loader = loader(arrays, "train", p, spec.seed, shuffle=True)
    val0 = _split_eval(model, arrays, p, spec.seed, "val")
    best = {SELECTOR_BA: dict(val0), SELECTOR_CE: dict(val0)}
    best_state = {SELECTOR_BA: cpu_state(model), SELECTOR_CE: cpu_state(model)}
    best_epoch = {SELECTOR_BA: 0, SELECTOR_CE: 0}
    history = [{"epoch": 0, "train_loss": None, "val_ba": val0["ba"], "val_mean_logit_ce": val0["mean_logit_ce"]}]
    snapshots = set(LOOKAHEAD_EPOCHS)

    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(model(x, lengths)["evidence"], lengths, y, "wcce")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        val = _split_eval(model, arrays, p, spec.seed, "val")
        history.append({"epoch": epoch, "train_loss": total / n, "val_ba": val["ba"], "val_mean_logit_ce": val["mean_logit_ce"]})
        for selector in (SELECTOR_BA, SELECTOR_CE):
            if _better(val, best[selector], selector):
                best[selector] = dict(val)
                best_state[selector] = cpu_state(model)
                best_epoch[selector] = epoch
        if epoch in snapshots:
            save_torch(root / f"epoch{epoch}.pt", {
                "experiment": EXPERIMENT_ID,
                "stage": "phase0a_snapshot",
                "seed": spec.seed,
                "epoch": epoch,
                "model_state_dict": cpu_state(model),
                "optimizer_state_dict": optimizer.state_dict(),
            })
        # Preserve the original C0 training trajectory exactly: early stopping is
        # still governed by the repository-standard BA-first selector.  C0-CE
        # only re-selects a checkpoint from epochs that the original trajectory
        # actually visited.
        if epoch >= p.min_epochs and epoch - best_epoch[SELECTOR_BA] >= p.patience:
            break

    for selector in (SELECTOR_BA, SELECTOR_CE):
        save_torch(root / f"selected__{selector}.pt", {
            "experiment": EXPERIMENT_ID,
            "stage": "phase0a_selector",
            "seed": spec.seed,
            "selector": selector,
            "model_state_dict": best_state[selector],
            "best_epoch": best_epoch[selector],
            "best_val": best[selector],
            "trajectory_id": f"seed{spec.seed}:shared_wcce",
        })
    save_json(done, {
        "seed": spec.seed,
        "trajectory_id": f"seed{spec.seed}:shared_wcce",
        "selectors": {
            selector: {"best_epoch": best_epoch[selector], "best_val": best[selector]}
            for selector in (SELECTOR_BA, SELECTOR_CE)
        },
        "history": history,
    })
    return {"status": "PASS", "seed": spec.seed, "best_epoch": best_epoch}


def _phase0a_selector_model(config: Config, seed: int, selector: str) -> tuple[nn.Module, Protocol, dict[str, np.ndarray], dict[str, Any]]:
    p, _, arrays = _load_core(config)
    payload = torch.load(config.results_dir / "phase0a" / f"seed{seed}" / f"selected__{selector}.pt", map_location="cpu", weights_only=False)
    model = BenchmarkNet(_run(ExpSpec("C0", seed)), p)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, p, arrays, payload


def finalize_phase0a(config: Config) -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        for selector in (SELECTOR_BA, SELECTOR_CE):
            model, p, arrays, payload = _phase0a_selector_model(config, seed, selector)
            row: dict[str, Any] = {
                "seed": seed,
                "selector": selector,
                "selected_epoch": payload["best_epoch"],
                "trajectory_id": payload["trajectory_id"],
            }
            for split in SPLITS:
                result = _split_eval(model, arrays, p, seed, split)
                row.update({f"{split}_{k}": v for k, v in result.items()})
            rows.append(row)
    frame = pd.DataFrame(rows)
    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    frame.to_csv(aggregate / "phase0a_checkpoint_selection.csv", index=False)

    ba = frame[frame.selector == SELECTOR_BA].set_index("seed")
    ce = frame[frame.selector == SELECTOR_CE].set_index("seed")
    ce_safe = int(((ce["val_ba"] + VAL_BA_RETENTION_PP / 100.0) >= ba["val_ba"]).sum())
    chosen = SELECTOR_CE if ce_safe >= MIN_SEED_SUPPORT else SELECTOR_BA
    decision = {
        "selected_checkpoint_rule": chosen,
        "selection_source": "validation_only",
        "ce_selector_val_ba_retention_seed_count": ce_safe,
        "required_seed_count": MIN_SEED_SUPPORT,
        "test_metrics_not_used_for_decision": True,
    }
    save_json(config.results_dir / "checkpoint_rule_selection.json", decision)
    return decision


def run_phase0b_lookahead(config: Config, spec: LookaheadSpec) -> dict[str, Any]:
    p, _, arrays = _load_core(config)
    out_dir = config.results_dir / "phase0b" / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation.json"
    if out_path.exists():
        return {"status": "exists", "key": spec.key}

    base_path = config.results_dir / "phase0a" / f"seed{spec.seed}" / f"epoch{spec.base_epoch}.pt"
    if not base_path.exists():
        result = {
            "status": "SKIPPED",
            "seed": spec.seed,
            "base_epoch": spec.base_epoch,
            "lambda_prefix": spec.lambda_prefix,
            "reason": "base epoch not reached by the preserved BA-first C0 trajectory",
            "selection_source": "validation_only",
        }
        save_json(out_path, result)
        return result
    payload = torch.load(base_path, map_location="cpu", weights_only=False)
    model = BenchmarkNet(_run(ExpSpec("P0", spec.seed)), p)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = _make_optimizer(model, p)
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    train_loader = loader(arrays, "train", p, spec.seed, shuffle=True)
    for _ in range(LOOKAHEAD_STEPS_EPOCHS):
        model.train()
        for x, y, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            evidence = model(x, lengths)["evidence"]
            wcce = sequence_loss(evidence, lengths, y, "wcce")
            prefix = prefix_wcce(evidence, lengths, y)
            loss = wcce + spec.lambda_prefix * prefix
            loss.backward()
            optimizer.step()

    val = _split_eval(model, arrays, p, spec.seed, "val")
    result = {
        "seed": spec.seed,
        "base_epoch": spec.base_epoch,
        "lambda_prefix": spec.lambda_prefix,
        "val_ba": val["ba"],
        "val_ce": val["mean_logit_ce"],
        "selection_source": "validation_only",
    }
    save_json(out_path, result)
    return {"status": "PASS", **result}


def select_phase0b(config: Config) -> dict[str, Any]:
    rows = []
    for spec in phase0b_specs():
        path = config.results_dir / "phase0b" / spec.key / "validation.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status", "PASS") == "SKIPPED":
            continue
        rows.append(payload)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No Phase0B lookahead points are available")
    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    frame.to_csv(aggregate / "phase0b_prefix_lookahead.csv", index=False)

    base = frame[frame.lambda_prefix == 0.0][["seed", "base_epoch", "val_ba", "val_ce"]].rename(
        columns={"val_ba": "base_val_ba", "val_ce": "base_val_ce"}
    )
    joined = frame.merge(base, on=["seed", "base_epoch"], how="left")
    joined["ce_improvement"] = joined["base_val_ce"] - joined["val_ce"]
    joined["ba_delta_pp"] = 100.0 * (joined["val_ba"] - joined["base_val_ba"])
    joined["eligible"] = (
        (joined["lambda_prefix"] > 0)
        & (joined["ce_improvement"] >= VAL_CE_MIN_IMPROVEMENT)
        & (joined["ba_delta_pp"] >= -VAL_BA_RETENTION_PP)
    )
    joined.to_csv(aggregate / "phase0b_prefix_eligibility.csv", index=False)

    summaries = []
    for value in PREFIX_LAMBDAS[1:]:
        part = joined[joined.lambda_prefix == value]
        seed_support = int(part.groupby("seed")["eligible"].any().sum())
        eligible = part[part.eligible]
        summaries.append({
            "lambda_prefix": value,
            "seed_support": seed_support,
            "eligible_points": int(len(eligible)),
            "mean_ce_improvement_eligible": float(eligible["ce_improvement"].mean()) if len(eligible) else float("nan"),
            "mean_ba_delta_pp_eligible": float(eligible["ba_delta_pp"].mean()) if len(eligible) else float("nan"),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(aggregate / "phase0b_prefix_summary.csv", index=False)
    eligible_values = summary[summary.seed_support >= MIN_SEED_SUPPORT]["lambda_prefix"].tolist()
    selected = float(min(eligible_values)) if eligible_values else 0.0
    decision = {
        "lambda_prefix": selected,
        "status": "selected" if selected > 0 else "no_validation_supported_prefix",
        "rule": "smallest lambda with eligible lookahead on at least 2/3 seeds",
        "selection_source": "validation_only",
        "test_metrics_not_used": True,
    }
    save_json(config.results_dir / "prefix_selection.json", decision)
    return decision


def _selected_protocol(config: Config) -> tuple[str, float]:
    selector = json.loads((config.results_dir / "checkpoint_rule_selection.json").read_text(encoding="utf-8"))["selected_checkpoint_rule"]
    lam = float(json.loads((config.results_dir / "prefix_selection.json").read_text(encoding="utf-8"))["lambda_prefix"])
    return selector, lam


def _parameter_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    named = dict(model.named_parameters())
    groups = {
        "R": [named["head.weight"]],
        "W_L1": [named["layers.0.weight"]],
        "W_L2": [named["layers.1.weight"]],
    }
    if isinstance(model, SuppressiveContextWriteGateNet):
        groups["G_z"] = [named["gate_input.weight"]]
        groups["G_h"] = [named["gate_history.weight"]]
        groups["G_b"] = [named["gate_bias"]]
    return groups


def _grad_vector(loss: torch.Tensor, params: list[nn.Parameter], retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    chunks = []
    for parameter, grad in zip(params, grads):
        chunks.append(torch.zeros_like(parameter).reshape(-1) if grad is None else grad.detach().reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(1)


def _gradient_geometry(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
    lambda_prefix: float,
    epoch: int,
) -> list[dict[str, Any]]:
    x, y, lengths = next(iter(loader(arrays, "train", p, seed, shuffle=False)))
    model.train()
    evidence = model(x, lengths)["evidence"]
    wcce = sequence_loss(evidence, lengths, y, "wcce")
    prefix = prefix_wcce(evidence, lengths, y)
    rows = []
    groups = _parameter_groups(model)
    for index, (name, params) in enumerate(groups.items()):
        gw = _grad_vector(wcce, params, retain_graph=True)
        gp = _grad_vector(prefix, params, retain_graph=index < len(groups) - 1)
        nw = float(gw.norm())
        np_ = float(gp.norm())
        cosine = float(F.cosine_similarity(gw[None], gp[None]).item()) if nw > 0 and np_ > 0 else float("nan")
        rows.append({
            "epoch": epoch,
            "group": name,
            "wcce_grad_norm": nw,
            "prefix_grad_norm": np_,
            "weighted_ratio": lambda_prefix * np_ / (nw + 1e-12),
            "cosine": cosine,
        })
    model.zero_grad(set_to_none=True)
    return rows


def _prefix_target_params(model: nn.Module, route: str) -> list[nn.Parameter]:
    named = list(model.named_parameters())
    gate_names = {"gate_input.weight", "gate_history.weight", "gate_bias"}
    if route == "joint":
        return [p for _, p in named if p.requires_grad]
    if route == "stopR":
        return [p for n, p in named if p.requires_grad and n != "head.weight"]
    if route == "gate_only":
        return [p for n, p in named if p.requires_grad and n in gate_names]
    if route == "no_gate":
        return [p for n, p in named if p.requires_grad and n not in gate_names]
    raise ValueError(route)


def _backward_routed(
    model: nn.Module,
    wcce: torch.Tensor,
    prefix: torch.Tensor,
    lambda_prefix: float,
    route: str,
) -> None:
    wcce.backward(retain_graph=lambda_prefix > 0)
    if lambda_prefix <= 0:
        return
    targets = _prefix_target_params(model, route)
    grads = torch.autograd.grad(prefix, targets, allow_unused=True)
    for parameter, grad in zip(targets, grads):
        if grad is None:
            continue
        addition = lambda_prefix * grad
        parameter.grad = addition.detach().clone() if parameter.grad is None else parameter.grad + addition.detach()


def _gate_summary(gates: dict[str, np.ndarray], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = []
    phase_rows = []
    for split in SPLITS:
        g = gates[f"{split}__gate_g"]
        lengths = arrays[f"{split}_lengths"]
        valid = np.arange(g.shape[1])[None, :] < lengths[:, None]
        gv = g[valid]
        rows.append({
            "split": split,
            "mean_g": float(gv.mean()),
            "std_g": float(gv.std()),
            "p_g_lt_01": float((gv < 0.1).mean()),
            "p_g_gt_09": float((gv > 0.9).mean()),
            "mean_sigmoid_derivative": float((gv * (1 - gv)).mean()),
        })
        for phase in range(10):
            values = []
            for i, length in enumerate(lengths.tolist()):
                lo = int(math.floor(phase * length / 10))
                hi = int(math.floor((phase + 1) * length / 10))
                if hi > lo:
                    values.append(g[i, lo:hi])
            merged = np.concatenate(values) if values else np.asarray([], dtype=np.float32)
            phase_rows.append({
                "split": split,
                "phase10": phase,
                "mean_g": float(merged.mean()) if len(merged) else float("nan"),
                "std_g": float(merged.std()) if len(merged) else float("nan"),
            })
    return {"summary": rows, "phase10": phase_rows}


def _extract(model: nn.Module, arrays: dict[str, np.ndarray], p: Protocol, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    traces: dict[str, np.ndarray] = {}
    native: dict[str, Any] = {"splits": {}, "users": []}
    gates: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            chunks = {"evidence": [], "L1__spike": [], "L2__spike": [], "L1__pre_reset": [], "L2__pre_reset": []}
            gate_chunks = []
            for x, _, lengths in loader(arrays, split, p, seed):
                out = model(x, lengths)
                chunks["evidence"].append(out["evidence"].numpy())
                chunks["L1__spike"].append(out["spike"][0].numpy().astype(np.uint8))
                chunks["L2__spike"].append(out["spike"][1].numpy().astype(np.uint8))
                chunks["L1__pre_reset"].append(out["pre_reset"][0].numpy().astype(np.float32))
                chunks["L2__pre_reset"].append(out["pre_reset"][1].numpy().astype(np.float32))
                if "gate_g" in out:
                    gate_chunks.append(out["gate_g"].numpy())
            for name, values in chunks.items():
                traces[f"{split}__{name}"] = np.concatenate(values)
            if gate_chunks:
                gates[f"{split}__gate_g"] = np.concatenate(gate_chunks)
            evidence = traces[f"{split}__evidence"]
            scores = np.asarray([evidence[i, :length].mean(0) for i, length in enumerate(arrays[f"{split}_lengths"])])
            pred = scores.argmax(1)
            native["splits"][split] = {**metrics(arrays[f"{split}_y"], pred), "mean_logit_ce": float(F.cross_entropy(torch.from_numpy(scores), torch.from_numpy(arrays[f"{split}_y"])).item()), "n_samples": len(pred)}
            for user in np.unique(arrays[f"{split}_users"]):
                chosen = arrays[f"{split}_users"] == user
                native["users"].append({"split": split, "user": str(user), "n": int(chosen.sum()), **metrics(arrays[f"{split}_y"][chosen], pred[chosen])})
    native["train_test_gap"] = native["splits"]["train"]["ba"] - native["splits"]["test"]["ba"]
    return traces, native, gates


def _run_probes(directory: Path, traces: dict[str, np.ndarray], arrays: dict[str, np.ndarray], spec: ExpSpec, p: Protocol) -> list[dict[str, Any]]:
    rows = []
    for layer in ("L1", "L2"):
        for state in ("spike",):
            for aggregation in AGGREGATIONS:
                shuffle_seeds = p.shuffle_seeds if aggregation.endswith("shuffled") else (-1,)
                for shuffle_seed in shuffle_seeds:
                    features = {
                        split: temporal_features(
                            traces[f"{split}__{layer}__{state}"],
                            arrays[f"{split}_lengths"],
                            arrays[f"{split}_ids"],
                            aggregation,
                            p,
                            shuffle_seed,
                        )
                        for split in SPLITS
                    }
                    for decoder in DECODERS:
                        scaler, probe, _ = fit_probe(features["train"], arrays["train_y"], features["val"], arrays["val_y"], decoder, p)
                        row: dict[str, Any] = {
                            "case": spec.case, "seed": spec.seed, "layer": layer, "state": state,
                            "aggregation": aggregation, "decoder": decoder, "shuffle_seed": shuffle_seed,
                        }
                        for split in SPLITS:
                            pred = probe.predict(scaler.transform(features[split].astype(np.float64)))
                            row.update({f"{split}_{k}": v for k, v in metrics(arrays[f"{split}_y"], pred).items()})
                        rows.append(row)
    save_json(directory / "probes.json", {"rows": rows})
    return rows


def _history_retrieval(directory: Path, traces: dict[str, np.ndarray], arrays: dict[str, np.ndarray], spec: ExpSpec) -> None:
    try:
        rows = exp14._retrieval(traces, arrays, exp14.LossSpec(spec.case, spec.seed))
        save_json(directory / "history_retrieval.json", {"rows": rows})
    except Exception as error:
        save_json(directory / "history_retrieval.json", {"status": "unavailable", "reason": str(error), "rows": []})


def evaluate_model(config: Config, spec: ExpSpec, model: nn.Module) -> dict[str, Any]:
    p, _, arrays = _load_core(config)
    directory = config.results_dir / "runs" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    traces, native, gates = _extract(model, arrays, p, spec.seed)
    save_json(directory / "native.json", native)
    save_npz(directory / "traces.npz", traces)
    _run_probes(directory, traces, arrays, spec, p)
    _history_retrieval(directory, traces, arrays, spec)
    try:
        history_rows = exp14._history_generalization(
            model, arrays, exp14.LossSpec(spec.case, spec.seed)
        )
        save_json(directory / "history_generalization.json", {"rows": history_rows})
    except Exception as error:
        save_json(
            directory / "history_generalization.json",
            {"status": "unavailable", "reason": str(error), "rows": []},
        )
    if gates:
        save_json(directory / "gate_summary.json", _gate_summary(gates, arrays))
    return {"status": "PASS", "run": spec.key}


def train_phase1_or_15(config: Config, spec: ExpSpec, phase15: bool = False) -> dict[str, Any]:
    selector, lambda_selected = _selected_protocol(config)
    if phase15 and spec.case not in PHASE15_CASES:
        raise ValueError(spec.case)
    if not phase15 and spec.case not in PHASE1_CASES:
        raise ValueError(spec.case)
    if phase15:
        decision = json.loads((config.results_dir / "phase1_decision.json").read_text(encoding="utf-8"))
        if not decision.get("enable_phase1_5", False):
            return {"status": "SKIPPED", "case": spec.case, "seed": spec.seed, "reason": decision.get("reason")}

    p, _, arrays = _load_core(config)
    directory = config.results_dir / "runs" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "checkpoint.pt").exists():
        return {"status": "exists", "run": spec.key}

    model = _make_model(spec, p)
    optimizer = _make_optimizer(model, p)
    train_loader = loader(arrays, "train", p, spec.seed, shuffle=True)
    use_prefix = spec.case in PREFIX_CASES and lambda_selected > 0
    route = PREFIX_ROUTES.get(spec.case, "joint")
    val0 = _split_eval(model, arrays, p, spec.seed, "val")
    best = dict(val0)
    best_state = cpu_state(model)
    best_epoch = 0
    history = [{"epoch": 0, "train_loss": None, "val_ba": val0["ba"], "val_mean_logit_ce": val0["mean_logit_ce"]}]
    gradient_rows = []
    update_rows = []

    if use_prefix:
        gradient_rows.extend(_gradient_geometry(model, arrays, p, spec.seed, lambda_selected, 0))

    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total = 0.0
        n = 0
        first_batch = True
        for x, y, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(x, lengths)
            wcce = sequence_loss(out["evidence"], lengths, y, "wcce")
            prefix = prefix_wcce(out["evidence"], lengths, y)
            before = None
            if first_batch and epoch in GRADIENT_EPOCHS:
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
            _backward_routed(model, wcce, prefix, lambda_selected if use_prefix else 0.0, route)
            optimizer.step()
            if before is not None:
                for group_name, params in _parameter_groups(model).items():
                    names = {id(p) for p in params}
                    num = 0.0
                    den = 0.0
                    for name, parameter in model.named_parameters():
                        if id(parameter) in names:
                            num += float((parameter.detach() - before[name]).square().sum())
                            den += float(before[name].square().sum())
                    update_rows.append({"epoch": epoch, "group": group_name, "relative_update_norm": math.sqrt(num) / (math.sqrt(den) + 1e-12)})
                first_batch = False
            total += float((wcce + (lambda_selected * prefix if use_prefix else 0)).detach()) * len(y)
            n += len(y)

        val = _split_eval(model, arrays, p, spec.seed, "val")
        history.append({"epoch": epoch, "train_loss": total / n, "val_ba": val["ba"], "val_mean_logit_ce": val["mean_logit_ce"]})
        if _better(val, best, selector):
            best = dict(val)
            best_state = cpu_state(model)
            best_epoch = epoch
        if use_prefix and epoch in GRADIENT_EPOCHS:
            gradient_rows.extend(_gradient_geometry(model, arrays, p, spec.seed, lambda_selected, epoch))
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    model.load_state_dict(best_state)
    save_json(directory / "train_history.json", {"rows": history})
    save_json(directory / "gradient_geometry.json", {"rows": gradient_rows})
    save_json(directory / "optimizer_update_diagnostics.json", {"rows": update_rows})
    save_torch(directory / "checkpoint.pt", {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "case": spec.case,
        "seed": spec.seed,
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val": best,
        "checkpoint_selector": selector,
        "lambda_prefix": lambda_selected if use_prefix else 0.0,
        "prefix_route": route if use_prefix else "none",
    })
    evaluate_model(config, spec, model)
    return {"status": "PASS", "run": spec.key, "best_epoch": best_epoch, "best_val": best}


def _native_row(spec: ExpSpec, native: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"case": spec.case, "seed": spec.seed, "train_test_gap": native["train_test_gap"]}
    for split in SPLITS:
        row.update({f"{split}_{k}": v for k, v in native["splits"][split].items()})
    return row




def _collapse_rows(probes: pd.DataFrame) -> pd.DataFrame:
    selected = probes[
        (probes["layer"] == "L2")
        & (probes["state"] == "spike")
        & (probes["decoder"] == "no_bias")
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    meaned = (
        selected.groupby(["case", "seed", "aggregation"], as_index=False)["test_ba"]
        .mean()
    )
    pivot = meaned.pivot(index=["case", "seed"], columns="aggregation", values="test_ba").reset_index()
    required = {
        "whole_count",
        "fixed250_ordered",
        "fixed250_shuffled",
        "relative10_ordered",
        "relative10_shuffled",
    }
    if not required.issubset(set(pivot.columns)):
        return pd.DataFrame()
    pivot["collapse_gap_pp"] = 100.0 * (pivot["relative10_ordered"] - pivot["whole_count"])
    pivot["fixed250_order_gap_pp"] = 100.0 * (pivot["fixed250_ordered"] - pivot["fixed250_shuffled"])
    pivot["relative10_order_gap_pp"] = 100.0 * (pivot["relative10_ordered"] - pivot["relative10_shuffled"])
    return pivot

def finalize_phase1(config: Config) -> dict[str, Any]:
    native_rows, probe_rows = [], []
    for spec in phase1_specs():
        directory = config.results_dir / "runs" / spec.key
        native_rows.append(_native_row(spec, json.loads((directory / "native.json").read_text(encoding="utf-8"))))
        probe_rows.extend(json.loads((directory / "probes.json").read_text(encoding="utf-8"))["rows"])
    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    native = pd.DataFrame(native_rows)
    probes = pd.DataFrame(probe_rows)
    native.to_csv(aggregate / "phase1_native_runs.csv", index=False)
    probes.to_csv(aggregate / "phase1_probe_runs.csv", index=False)
    collapse = _collapse_rows(probes)
    collapse.to_csv(aggregate / "phase1_collapse_gap.csv", index=False)

    means = native.groupby("case")["val_ba"].mean()
    prefix_gain = float(means["GP_J"] - means["G0"])
    ungated_gain = float(means["P0"] - means["C0"])
    interaction = prefix_gain - ungated_gain
    per_seed = native.pivot(index="seed", columns="case", values="val_ba")
    per_seed_interaction = (
        (per_seed["GP_J"] - per_seed["G0"])
        - (per_seed["P0"] - per_seed["C0"])
    )
    positive_seed_count = int((per_seed_interaction > 0).sum())
    enabled = bool(
        interaction > 0
        and prefix_gain > 0
        and positive_seed_count >= MIN_SEED_SUPPORT
    )
    decision = {
        "metric": "validation native BA interaction",
        "gp_minus_g0": prefix_gain,
        "p0_minus_c0": ungated_gain,
        "interaction": interaction,
        "per_seed_interaction": {
            str(int(seed)): float(value)
            for seed, value in per_seed_interaction.items()
        },
        "positive_interaction_seed_count": positive_seed_count,
        "required_seed_count": MIN_SEED_SUPPORT,
        "enable_phase1_5": enabled,
        "reason": "positive Gate x Prefix validation interaction with >=2/3 seed support" if enabled else "no stable positive Gate x Prefix validation interaction",
        "test_metrics_not_used": True,
    }
    save_json(config.results_dir / "phase1_decision.json", decision)
    return decision


def _forced_gate(model: SuppressiveContextWriteGateNet, x: torch.Tensor, lengths: torch.Tensor, mode: str, sample_ids: np.ndarray, shuffle_seed: int) -> torch.Tensor | None:
    if mode == "learned":
        return None
    with torch.no_grad():
        learned = model(x, lengths)["gate_g"]
    if mode == "gate_one":
        return torch.ones_like(learned)
    if mode == "gate_time_mean":
        valid = torch.arange(learned.shape[1])[None, :] < lengths[:, None]
        mean = (learned * valid).sum(1) / lengths
        return mean[:, None].expand_as(learned)
    if mode == "gate_time_shuffle":
        forced = learned.clone()
        for i, length in enumerate(lengths.tolist()):
            rng = np.random.default_rng(paired_seed(shuffle_seed, f"exp16:gate_shuffle:{sample_ids[i]}"))
            permutation = torch.from_numpy(rng.permutation(length)).long()
            forced[i, :length] = forced[i, permutation]
        return forced
    raise ValueError(mode)


def run_functional_ablation(config: Config, spec: ExpSpec) -> dict[str, Any]:
    if spec.case not in GATED_CASES:
        raise ValueError("Functional ablation requires gated case")
    p, _, arrays = _load_core(config)
    checkpoint_path = config.results_dir / "runs" / spec.key / "checkpoint.pt"
    if not checkpoint_path.exists():
        return {"status": "SKIPPED", "run": spec.key, "reason": "checkpoint unavailable (case may be conditionally disabled)"}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = _make_model(spec, p)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    root = config.results_dir / "functional_ablation" / spec.key
    root.mkdir(parents=True, exist_ok=True)

    for mode in FUNCTIONAL_ABLATIONS:
        seeds = SHUFFLE_SEEDS if mode == "gate_time_shuffle" else (-1,)
        for shuffle_seed in seeds:
            tag = mode if shuffle_seed < 0 else f"{mode}__shuffle{shuffle_seed}"
            native = {"splits": {}}
            traces: dict[str, np.ndarray] = {}
            with torch.no_grad():
                for split in SPLITS:
                    evidence_chunks, l1_chunks, l2_chunks = [], [], []
                    offset = 0
                    for x, y, lengths in loader(arrays, split, p, spec.seed):
                        ids = arrays[f"{split}_ids"][offset:offset + len(y)]
                        offset += len(y)
                        forced = _forced_gate(model, x, lengths, mode, ids, shuffle_seed)
                        out = model(x, lengths, forced_g=forced)
                        evidence_chunks.append(out["evidence"].numpy())
                        l1_chunks.append(out["spike"][0].numpy().astype(np.uint8))
                        l2_chunks.append(out["spike"][1].numpy().astype(np.uint8))
                    evidence = np.concatenate(evidence_chunks)
                    traces[f"{split}__L1__spike"] = np.concatenate(l1_chunks)
                    traces[f"{split}__L2__spike"] = np.concatenate(l2_chunks)
                    scores = np.asarray([evidence[i, :length].mean(0) for i, length in enumerate(arrays[f"{split}_lengths"])])
                    pred = scores.argmax(1)
                    native["splits"][split] = metrics(arrays[f"{split}_y"], pred)
            directory = root / tag
            directory.mkdir(parents=True, exist_ok=True)
            save_json(directory / "native.json", native)
            _run_probes(directory, traces, arrays, ExpSpec(f"{spec.case}_{tag}", spec.seed), p)
    return {"status": "PASS", "run": spec.key}


def finalize(config: Config) -> dict[str, Any]:
    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)

    phase15_native_rows: list[dict[str, Any]] = []
    phase15_probe_rows: list[dict[str, Any]] = []
    for spec in phase15_specs():
        directory = config.results_dir / "runs" / spec.key
        if not (directory / "checkpoint.pt").exists():
            continue
        phase15_native_rows.append(
            _native_row(
                spec,
                json.loads((directory / "native.json").read_text(encoding="utf-8")),
            )
        )
        phase15_probe_rows.extend(
            json.loads((directory / "probes.json").read_text(encoding="utf-8"))["rows"]
        )
    if phase15_native_rows:
        pd.DataFrame(phase15_native_rows).to_csv(
            aggregate / "phase1_5_native_runs.csv", index=False
        )
    if phase15_probe_rows:
        phase15_probes = pd.DataFrame(phase15_probe_rows)
        phase15_probes.to_csv(aggregate / "phase1_5_probe_runs.csv", index=False)
        _collapse_rows(phase15_probes).to_csv(
            aggregate / "phase1_5_collapse_gap.csv", index=False
        )

    ablation_native_rows: list[dict[str, Any]] = []
    ablation_probe_rows: list[dict[str, Any]] = []
    gated_specs = [s for s in (*phase1_specs(), *phase15_specs()) if s.case in GATED_CASES]
    for spec in gated_specs:
        root = config.results_dir / "functional_ablation" / spec.key
        if not root.exists():
            continue
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            native_path = directory / "native.json"
            probes_path = directory / "probes.json"
            if not native_path.exists() or not probes_path.exists():
                continue
            native = json.loads(native_path.read_text(encoding="utf-8"))
            row: dict[str, Any] = {
                "case": spec.case,
                "seed": spec.seed,
                "intervention": directory.name,
            }
            for split in SPLITS:
                row.update({
                    f"{split}_{key}": value
                    for key, value in native["splits"][split].items()
                })
            ablation_native_rows.append(row)
            for probe in json.loads(probes_path.read_text(encoding="utf-8"))["rows"]:
                ablation_probe_rows.append({
                    "parent_case": spec.case,
                    "parent_seed": spec.seed,
                    "intervention": directory.name,
                    **probe,
                })
    if ablation_native_rows:
        ablation_native = pd.DataFrame(ablation_native_rows)
        ablation_native.to_csv(
            aggregate / "functional_ablation_native_runs.csv", index=False
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
    if ablation_probe_rows:
        pd.DataFrame(ablation_probe_rows).to_csv(
            aggregate / "functional_ablation_probe_runs.csv", index=False
        )

    report = {
        "status": "PASS",
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "checkpoint_rule": json.loads((config.results_dir / "checkpoint_rule_selection.json").read_text(encoding="utf-8")),
        "prefix_selection": json.loads((config.results_dir / "prefix_selection.json").read_text(encoding="utf-8")),
        "phase1_decision": json.loads((config.results_dir / "phase1_decision.json").read_text(encoding="utf-8")),
        "phase1_5_completed_runs": len(phase15_native_rows),
        "functional_ablation_rows": len(ablation_native_rows),
    }
    save_json(aggregate / "manifest.json", report)
    return report


def configure_cpu() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exp16 Prefix-supervised selective memory")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("plan")
    p0a = sub.add_parser("train-phase0a")
    p0a.add_argument("--task-id", type=int, required=True)
    sub.add_parser("finalize-phase0a")
    p0b = sub.add_parser("run-phase0b")
    p0b.add_argument("--task-id", type=int, required=True)
    sub.add_parser("select-phase0b")
    p1 = sub.add_parser("train-phase1")
    p1.add_argument("--task-id", type=int, required=True)
    sub.add_parser("finalize-phase1")
    p15 = sub.add_parser("train-phase1-5")
    p15.add_argument("--task-id", type=int, required=True)
    ab = sub.add_parser("functional-ablation")
    ab.add_argument("--task-id", type=int, required=True)
    sub.add_parser("finalize")
    args = parser.parse_args(argv)

    configure_cpu()
    config = config_from_args(args)
    if args.command == "prepare":
        print(json.dumps(prepare(config), indent=2))
    elif args.command == "plan":
        print(json.dumps({
            "phase0a": [asdict(s) | {"key": s.key} for s in phase0a_specs()],
            "phase0b": [asdict(s) | {"key": s.key} for s in phase0b_specs()],
            "phase1": [asdict(s) | {"key": s.key} for s in phase1_specs()],
            "phase1_5": [asdict(s) | {"key": s.key} for s in phase15_specs()],
        }, indent=2))
    elif args.command == "train-phase0a":
        print(json.dumps(train_phase0a(config, phase0a_specs()[args.task_id]), indent=2))
    elif args.command == "finalize-phase0a":
        print(json.dumps(finalize_phase0a(config), indent=2))
    elif args.command == "run-phase0b":
        print(json.dumps(run_phase0b_lookahead(config, phase0b_specs()[args.task_id]), indent=2))
    elif args.command == "select-phase0b":
        print(json.dumps(select_phase0b(config), indent=2))
    elif args.command == "train-phase1":
        print(json.dumps(train_phase1_or_15(config, phase1_specs()[args.task_id]), indent=2))
    elif args.command == "finalize-phase1":
        print(json.dumps(finalize_phase1(config), indent=2))
    elif args.command == "train-phase1-5":
        print(json.dumps(train_phase1_or_15(config, phase15_specs()[args.task_id], phase15=True), indent=2))
    elif args.command == "functional-ablation":
        gated = [s for s in (*phase1_specs(), *phase15_specs()) if s.case in GATED_CASES]
        print(json.dumps(run_functional_ablation(config, gated[args.task_id]), indent=2))
    elif args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))


if __name__ == "__main__":
    main()
