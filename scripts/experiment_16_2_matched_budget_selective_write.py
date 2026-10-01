"""Exp16.2: matched-budget selective-write experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from core_benchmark_v1.model import BenchmarkNet, lif_step, sequence_loss, valid_mask
from core_benchmark_v1.protocol import Protocol, Run, SPLITS, paired_seed
from core_benchmark_v1.storage import save_json, save_npz, save_torch
from core_benchmark_v1.training import cpu_state, metrics
from scripts import experiment_16_prefix_supervised_selective_memory as exp16


EXPERIMENT_ID = "experiment_16_2_matched_budget_selective_write"
PROTOCOL_VERSION = "matched_budget_selective_write_v1"
FORMAL_SEEDS = (11, 23, 37)
CALIBRATION_SEED = 101
RHOS = (0.9, 0.7, 0.5)
LAMBDA_B_CANDIDATES = (0.1, 0.3, 1.0, 3.0)
BUDGET_MAE_TOL = 0.02
BUDGET_COVERAGE_TOL = 0.05
BUDGET_LATE_EPOCHS = 5
GZ0_INIT_Q = 0.9
SHIFTS = ((2, 3, 4), (2, 3, 4))
SHUFFLE_SEEDS = (101, 211, 307)
SHIFT_FRACTIONS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class ExpSpec:
    case: str
    seed: int
    gate_kind: str
    rho: float | None = None
    lambda_budget: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.case}__seed{self.seed}"

    @property
    def constrained(self) -> bool:
        return self.gate_kind == "dynamic" and self.rho is not None

    @property
    def gated(self) -> bool:
        return self.gate_kind in {"static", "dynamic"}


@dataclass(frozen=True)
class CalibrationSpec:
    rho: float
    lambda_budget: float
    seed: int = CALIBRATION_SEED

    @property
    def key(self) -> str:
        return f"rho{rho_tag(self.rho)}__lb{lambda_tag(self.lambda_budget)}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    core_results_dir: Path


def rho_tag(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def lambda_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


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


def _run(case: str, seed: int) -> Run:
    return Run(case, seed, "16_2_matched_budget", shifts=SHIFTS, objective="wcce")


def calibration_specs() -> list[CalibrationSpec]:
    return [
        CalibrationSpec(rho, lambda_budget)
        for rho in RHOS
        for lambda_budget in LAMBDA_B_CANDIDATES
    ]


def formal_specs(lambda_by_rho: dict[float, float] | None = None) -> list[ExpSpec]:
    selected = lambda_by_rho or {rho: 0.0 for rho in RHOS}
    rows: list[ExpSpec] = []
    for seed in FORMAL_SEEDS:
        rows.append(ExpSpec("C0", seed, "none"))
        rows.append(ExpSpec("GZ0", seed, "dynamic"))
        for rho in RHOS:
            tag = int(round(rho * 100))
            rows.append(ExpSpec(f"S{tag}", seed, "static", rho=rho))
            rows.append(ExpSpec(f"G{tag}", seed, "dynamic", rho=rho, lambda_budget=float(selected[rho])))
    return rows


def dynamic_formal_specs(lambda_by_rho: dict[float, float]) -> list[ExpSpec]:
    return [spec for spec in formal_specs(lambda_by_rho) if spec.gate_kind == "dynamic"]


def pair_tasks(lambda_by_rho: dict[float, float] | None = None) -> list[dict[str, Any]]:
    selected = lambda_by_rho or {rho: 0.0 for rho in RHOS}
    order_table = {
        11: {0.9: "SG", 0.7: "GS", 0.5: "SG"},
        23: {0.9: "GS", 0.7: "SG", 0.5: "GS"},
        37: {0.9: "SG", 0.7: "GS", 0.5: "SG"},
    }
    tasks: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        tasks.append({"seed": seed, "rho": None, "order": "C0->GZ0",
                      "specs": [ExpSpec("C0", seed, "none"), ExpSpec("GZ0", seed, "dynamic")]})
        for rho in RHOS:
            tag = int(round(rho * 100))
            s = ExpSpec(f"S{tag}", seed, "static", rho=rho)
            g = ExpSpec(f"G{tag}", seed, "dynamic", rho=rho, lambda_budget=float(selected[rho]))
            if order_table[seed][rho] == "SG":
                task_specs, order = [s, g], "S->G"
            else:
                task_specs, order = [g, s], "G->S"
            tasks.append({"seed": seed, "rho": rho, "order": order, "specs": task_specs})
    return tasks


class MatchedBudgetNet(BenchmarkNet):
    def __init__(self, run: Run, p: Protocol, gate_kind: str, rho: float | None) -> None:
        if gate_kind not in {"static", "dynamic"}:
            raise ValueError(gate_kind)
        super().__init__(run, p)
        self.gate_kind = gate_kind
        self.rho = rho
        if gate_kind == "dynamic":
            init_q = GZ0_INIT_Q if rho is None else rho
            self.gate_input = nn.Linear(p.width, 1, bias=False)
            self.gate_bias = nn.Parameter(torch.tensor(math.log(init_q / (1.0 - init_q)), dtype=torch.float32))
            with torch.no_grad():
                self.gate_input.weight.zero_()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, *, forced_g: torch.Tensor | None = None,
                reset_at: torch.Tensor | None = None, reset_layers: tuple[int, ...] = ()) -> dict[str, Any]:
        p = self.protocol
        batch, steps, channels = x.shape
        if channels != p.input_channels or lengths.shape != (batch,):
            raise ValueError("Invalid input geometry")
        if forced_g is not None and forced_g.shape != (batch, steps):
            raise ValueError("forced_g must be [batch, steps]")

        syn = [x.new_zeros(batch, p.width) for _ in self.layers]
        mem = [x.new_zeros(batch, p.width) for _ in self.layers]
        spikes: list[list[torch.Tensor]] = [[], []]
        pre_reset: list[list[torch.Tensor]] = [[], []]
        syn_trace: list[list[torch.Tensor]] = [[], []]
        mem_trace: list[list[torch.Tensor]] = [[], []]
        evidence: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        write_candidates: list[torch.Tensor] = []
        gated_writes: list[torch.Tensor] = []

        for t in range(int(lengths.max().item())):
            active = (t < lengths)[:, None]
            cur = x[:, t] * active
            if reset_at is not None:
                retain = (reset_at != t)[:, None]
                if 0 in reset_layers:
                    syn[0], mem[0] = syn[0] * retain, mem[0] * retain
                if 1 in reset_layers:
                    syn[1], mem[1] = syn[1] * retain, mem[1] * retain

            candidate0 = self.alpha_0 * syn[0] + self.layers[0](cur)
            syn[0] = torch.where(active, candidate0, syn[0])
            s1, new_mem1, pre1 = lif_step(syn[0], mem[0], self.betas[0], p.threshold, p.surrogate_slope)
            mem[0] = torch.where(active, new_mem1, mem[0])
            s1 = s1 * active

            if self.gate_kind == "static":
                assert self.rho is not None
                gate = x.new_full((batch,), self.rho)
            else:
                gate = torch.sigmoid(self.gate_input(s1).squeeze(-1) + self.gate_bias)
            used_gate = gate if forced_g is None else forced_g[:, t]
            u_t = self.layers[1](s1)
            gu_t = used_gate[:, None] * u_t
            candidate1 = self.alpha_1 * syn[1] + gu_t
            syn[1] = torch.where(active, candidate1, syn[1])
            s2, new_mem2, pre2 = lif_step(syn[1], mem[1], self.betas[1], p.threshold, p.surrogate_slope)
            mem[1] = torch.where(active, new_mem2, mem[1])
            s2 = s2 * active

            spikes[0].append(s1); spikes[1].append(s2)
            pre_reset[0].append(pre1 * active); pre_reset[1].append(pre2 * active)
            syn_trace[0].append(syn[0] * active); syn_trace[1].append(syn[1] * active)
            mem_trace[0].append(mem[0] * active); mem_trace[1].append(mem[1] * active)
            evidence.append(self.head(s2))
            gates.append(gate * active.squeeze(1))
            write_candidates.append(u_t * active)
            gated_writes.append(gu_t * active)

        def stack3(values: list[torch.Tensor]) -> torch.Tensor:
            out = torch.stack(values, dim=1)
            return F.pad(out, (0, 0, 0, steps - out.shape[1]))

        def stack2(values: list[torch.Tensor]) -> torch.Tensor:
            out = torch.stack(values, dim=1)
            return F.pad(out, (0, steps - out.shape[1]))

        return {
            "spike": (stack3(spikes[0]), stack3(spikes[1])),
            "pre_reset": (stack3(pre_reset[0]), stack3(pre_reset[1])),
            "synapse": (stack3(syn_trace[0]), stack3(syn_trace[1])),
            "membrane": (stack3(mem_trace[0]), stack3(mem_trace[1])),
            "evidence": stack3(evidence),
            "final_syn": tuple(syn),
            "final_mem": tuple(mem),
            "gate_g": stack2(gates),
            "write_candidate": stack3(write_candidates),
            "gated_write": stack3(gated_writes),
        }


def _base_shared_state(seed: int, p: Protocol) -> dict[str, torch.Tensor]:
    base = BenchmarkNet(_run("C0", seed), p)
    return {name: value.detach().clone() for name, value in base.state_dict().items()}


def _make_model(spec: ExpSpec, p: Protocol) -> nn.Module:
    model: nn.Module
    if spec.gate_kind == "none":
        model = BenchmarkNet(_run(spec.case, spec.seed), p)
    else:
        model = MatchedBudgetNet(_run(spec.case, spec.seed), p, spec.gate_kind, spec.rho)
    own = model.state_dict()
    for name, value in _base_shared_state(spec.seed, p).items():
        if name in own:
            own[name] = value.clone()
    model.load_state_dict(own, strict=True)
    return model


def _make_optimizer(model: nn.Module, p: Protocol) -> torch.optim.Optimizer:
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    gate_bias = [value for name, value in named if name == "gate_bias"]
    other = [value for name, value in named if name != "gate_bias"]
    groups: list[dict[str, Any]] = []
    if other:
        groups.append({"params": other, "weight_decay": p.weight_decay})
    if gate_bias:
        groups.append({"params": gate_bias, "weight_decay": 0.0})
    return torch.optim.Adam(groups, lr=p.learning_rate)


def _epoch_permutation(n: int, seed: int, epoch: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(
        paired_seed(seed, f"exp16_2:loader:train:epoch:{epoch}")
    )
    return torch.randperm(n, generator=generator)


def _permutation_hash(permutation: torch.Tensor) -> str:
    return hashlib.sha256(permutation.cpu().numpy().astype(np.int64).tobytes()).hexdigest()


def _train_batches(arrays: dict[str, np.ndarray], p: Protocol, seed: int, epoch: int):
    permutation = _epoch_permutation(len(arrays["train_y"]), seed, epoch)
    for start in range(0, len(permutation), p.batch_size):
        chosen = permutation[start:start + p.batch_size].numpy()
        yield (
            torch.from_numpy(arrays["train_x"][chosen]),
            torch.from_numpy(arrays["train_y"][chosen]),
            torch.from_numpy(arrays["train_lengths"][chosen]),
        )


def _budget_per_sample(gates: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = valid_mask(lengths, gates.shape[1]).to(gates.dtype)
    return (gates * mask).sum(1) / lengths.to(gates.dtype)


def _budget_loss(gates: torch.Tensor, lengths: torch.Tensor, rho: float) -> torch.Tensor:
    means = _budget_per_sample(gates, lengths)
    return (means - rho).square().mean()


def _budget_eval(model: nn.Module, arrays: dict[str, np.ndarray], p: Protocol,
                 seed: int, split: str, rho: float) -> dict[str, float]:
    means: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, _, lengths in exp16.loader(arrays, split, p, seed):
            out = model(x, lengths)
            means.append(_budget_per_sample(out["gate_g"], lengths).numpy())
    values = np.concatenate(means)
    errors = np.abs(values - rho)
    return {
        "mean_gate": float(values.mean()),
        "std_sample_mean_gate": float(values.std()),
        "budget_mae": float(errors.mean()),
        "coverage_within_0p05": float((errors <= BUDGET_COVERAGE_TOL).mean()),
    }


def _split_eval(model: nn.Module, arrays: dict[str, np.ndarray], p: Protocol,
                seed: int, split: str) -> dict[str, float]:
    return exp16._split_eval(model, arrays, p, seed, split)


def _better(candidate: dict[str, float], best: dict[str, float]) -> bool:
    return exp16._better(candidate, best, exp16.SELECTOR_BA)


def _initial_forward_assertions(p: Protocol, arrays: dict[str, np.ndarray], seed: int) -> None:
    x = torch.from_numpy(arrays["train_x"][:min(4, len(arrays["train_x"]))])
    lengths = torch.from_numpy(arrays["train_lengths"][:len(x)])
    c0 = _make_model(ExpSpec("C0", seed, "none"), p)
    s100 = MatchedBudgetNet(_run("S100", seed), p, "static", 1.0)
    own = s100.state_dict()
    for name, value in _base_shared_state(seed, p).items():
        if name in own:
            own[name] = value.clone()
    s100.load_state_dict(own, strict=True)
    with torch.no_grad():
        out_c0 = c0(x, lengths)
        out_s100 = s100(x, lengths)
    for key, idx in (("evidence", None), ("spike", 1), ("pre_reset", 1)):
        left = out_c0[key] if idx is None else out_c0[key][idx]
        right = out_s100[key] if idx is None else out_s100[key][idx]
        if not torch.equal(left, right):
            raise AssertionError(f"C0 != S100 for {key}")

    for rho in RHOS:
        tag = int(round(100 * rho))
        s = _make_model(ExpSpec(f"S{tag}", seed, "static", rho=rho), p)
        g = _make_model(ExpSpec(f"G{tag}", seed, "dynamic", rho=rho), p)
        with torch.no_grad():
            out_s = s(x, lengths)
            out_g = g(x, lengths)
        for key, idx in (("evidence", None), ("spike", 1), ("synapse", 1), ("membrane", 1)):
            left = out_s[key] if idx is None else out_s[key][idx]
            right = out_g[key] if idx is None else out_g[key][idx]
            if not torch.equal(left, right):
                raise AssertionError(f"S{tag} != G{tag} at initialization for {key}")


def prepare(config: Config) -> dict[str, Any]:
    p, lock, arrays = exp16._load_core(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    for seed in FORMAL_SEEDS:
        _initial_forward_assertions(p, arrays, seed)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "core_dataset_hash": lock["dataset_hash"],
        "formal_seeds": list(FORMAL_SEEDS),
        "calibration_seed": CALIBRATION_SEED,
        "rhos": list(RHOS),
        "lambda_budget_candidates": list(LAMBDA_B_CANDIDATES),
        "budget_mae_tolerance": BUDGET_MAE_TOL,
        "budget_late_epochs": BUDGET_LATE_EPOCHS,
        "formal_cases_per_seed": ["C0", "GZ0", "S90", "G90", "S70", "G70", "S50", "G50"],
        "formal_run_count": 24,
        "paired_slurm_task_count": 12,
        "phase2_cases": ["GZ0", "G90", "G70", "G50"],
        "checkpoint_selection": "G_rho: val budget MAE<=0.02 then max val BA, min val CE, earliest epoch; others BA-first",
        "loader_contract": "epoch-specific deterministic permutation hash per seed/epoch",
        "paired_initialization": "explicit shared-state clone from one C0 initialization per seed",
        "matched_quantity": "mean gate budget, not memory capacity",
        "test_metrics_never_used_for_training_or_checkpoint_selection": True,
        "width": p.width,
        "tau_mem_ms": p.tau_mem_ms,
    }
    save_json(config.results_dir / "protocol.json", payload)
    return payload


def train_calibration(config: Config, spec: CalibrationSpec) -> dict[str, Any]:
    p, _, arrays = exp16._load_core(config)
    case = f"CAL_G{int(round(spec.rho * 100))}"
    formal = ExpSpec(case, spec.seed, "dynamic", rho=spec.rho, lambda_budget=spec.lambda_budget)
    directory = config.results_dir / "phase0" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / "calibration.json"
    if done.exists():
        return {"status": "exists", "run": spec.key}

    model = _make_model(formal, p)
    optimizer = _make_optimizer(model, p)
    history: list[dict[str, Any]] = []
    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, y, lengths in _train_batches(arrays, p, spec.seed, epoch):
            optimizer.zero_grad(set_to_none=True)
            out = model(x, lengths)
            wcce = sequence_loss(out["evidence"], lengths, y, "wcce")
            budget = _budget_loss(out["gate_g"], lengths, spec.rho)
            loss = wcce + spec.lambda_budget * budget
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        val = _split_eval(model, arrays, p, spec.seed, "val")
        budget_val = _budget_eval(model, arrays, p, spec.seed, "val", spec.rho)
        history.append({
            "epoch": epoch,
            "train_loss": total / count,
            "val_ba": val["ba"],
            "val_mean_logit_ce": val["mean_logit_ce"],
            **budget_val,
            "sampler_hash": _permutation_hash(
                _epoch_permutation(len(arrays["train_y"]), spec.seed, epoch)
            ),
        })

    late = history[-BUDGET_LATE_EPOCHS:]
    late_mae = float(np.mean([row["budget_mae"] for row in late]))
    late_coverage = float(np.mean([row["coverage_within_0p05"] for row in late]))
    payload = {
        "rho": spec.rho,
        "lambda_budget": spec.lambda_budget,
        "seed": spec.seed,
        "late_budget_mae": late_mae,
        "late_coverage_within_0p05": late_coverage,
        "compliant": late_mae <= BUDGET_MAE_TOL,
        "history": history,
    }
    save_json(done, payload)
    return {"status": "PASS", "run": spec.key, **payload}


def finalize_calibration(config: Config) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in calibration_specs():
        path = config.results_dir / "phase0" / spec.key / "calibration.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "rho": float(payload["rho"]),
            "lambda_budget": float(payload["lambda_budget"]),
            "late_budget_mae": float(payload["late_budget_mae"]),
            "late_coverage_within_0p05": float(payload["late_coverage_within_0p05"]),
            "compliant": bool(payload["compliant"]),
        })
    frame = pd.DataFrame(rows)
    common: float | None = None
    for candidate in LAMBDA_B_CANDIDATES:
        subset = frame[frame["lambda_budget"] == candidate]
        if len(subset) == len(RHOS) and bool(subset["compliant"].all()):
            common = float(candidate)
            break
    selected: dict[float, float] = {}
    if common is not None:
        selected = {rho: common for rho in RHOS}
        mode = "common"
    else:
        mode = "per_rho"
        for rho in RHOS:
            subset = frame[(frame["rho"] == rho) & frame["compliant"]].sort_values("lambda_budget")
            if subset.empty:
                raise RuntimeError(f"No compliant lambda_B for rho={rho}")
            selected[rho] = float(subset.iloc[0]["lambda_budget"])
    result = {
        "selection_uses_ba": False,
        "mode": mode,
        "lambda_by_rho": {str(rho): value for rho, value in selected.items()},
        "criterion": f"smallest lambda_B with late-{BUDGET_LATE_EPOCHS} val budget MAE <= {BUDGET_MAE_TOL}",
    }
    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    frame.to_csv(aggregate / "phase0_budget_calibration.csv", index=False)
    save_json(config.results_dir / "phase0_selection.json", result)
    return result


def _selected_lambda_by_rho(config: Config) -> dict[float, float]:
    path = config.results_dir / "phase0_selection.json"
    if not path.exists():
        raise FileNotFoundError("Run finalize-calibration before Phase1")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {float(key): float(value) for key, value in payload["lambda_by_rho"].items()}


def _diagnostics(model: nn.Module, arrays: dict[str, np.ndarray], p: Protocol, seed: int) -> dict[str, Any]:
    if not isinstance(model, MatchedBudgetNet):
        return {}
    output: dict[str, Any] = {"splits": {}}
    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            sample_means: list[float] = []
            temporal_stds: list[float] = []
            p90_p10: list[float] = []
            high: list[float] = []
            low: list[float] = []
            u_norm: list[float] = []
            gu_norm: list[float] = []
            i_norm: list[float] = []
            v_norm: list[float] = []
            l1_rate: list[float] = []
            l2_rate: list[float] = []
            for x, _, lengths in exp16.loader(arrays, split, p, seed):
                out = model(x, lengths)
                for i, length in enumerate(lengths.tolist()):
                    g = out["gate_g"][i, :length]
                    sample_means.append(float(g.mean()))
                    temporal_stds.append(float(g.std(unbiased=False)))
                    p90_p10.append(float(torch.quantile(g, 0.9) - torch.quantile(g, 0.1)))
                    rho_ref = model.rho if model.rho is not None else float(g.mean())
                    high.append(float((g > rho_ref + 0.1).float().mean()))
                    low.append(float((g < rho_ref - 0.1).float().mean()))
                    u_norm.extend(out["write_candidate"][i, :length].norm(dim=1).tolist())
                    gu_norm.extend(out["gated_write"][i, :length].norm(dim=1).tolist())
                    i_norm.extend(out["synapse"][1][i, :length].norm(dim=1).tolist())
                    v_norm.extend(out["membrane"][1][i, :length].norm(dim=1).tolist())
                    l1_rate.append(float(out["spike"][0][i, :length].mean()))
                    l2_rate.append(float(out["spike"][1][i, :length].mean()))
            output["splits"][split] = {
                "mean_sample_gate": float(np.mean(sample_means)),
                "std_sample_mean_gate": float(np.std(sample_means)),
                "mean_within_sample_gate_std": float(np.mean(temporal_stds)),
                "mean_p90_minus_p10": float(np.mean(p90_p10)),
                "p_gate_gt_rho_plus_0p1": float(np.mean(high)),
                "p_gate_lt_rho_minus_0p1": float(np.mean(low)),
                "mean_u_l2": float(np.mean(u_norm)),
                "mean_gu_l2": float(np.mean(gu_norm)),
                "mean_l2_syn_l2": float(np.mean(i_norm)),
                "mean_l2_mem_l2": float(np.mean(v_norm)),
                "l1_firing_rate": float(np.mean(l1_rate)),
                "l2_firing_rate": float(np.mean(l2_rate)),
            }
    output["w2_frobenius"] = float(model.layers[1].weight.detach().norm())
    return output


def _evaluate_model(config: Config, spec: ExpSpec, model: nn.Module) -> None:
    p, _, arrays = exp16._load_core(config)
    directory = config.results_dir / "runs" / spec.key
    traces, native, gates = exp16._extract(model, arrays, p, spec.seed)
    save_json(directory / "native.json", native)
    save_npz(directory / "traces.npz", traces)
    exp16._run_probes(directory, traces, arrays, spec, p)
    if gates:
        save_json(directory / "gate_summary_core.json", exp16._gate_summary(gates, arrays))
    diagnostics = _diagnostics(model, arrays, p, spec.seed)
    if diagnostics:
        save_json(directory / "write_state_diagnostics.json", diagnostics)


def train_one(config: Config, spec: ExpSpec) -> dict[str, Any]:
    p, _, arrays = exp16._load_core(config)
    directory = config.results_dir / "runs" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    if checkpoint_path.exists():
        return {"status": "exists", "run": spec.key}

    model = _make_model(spec, p)
    optimizer = _make_optimizer(model, p)
    best: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, y, lengths in _train_batches(arrays, p, spec.seed, epoch):
            optimizer.zero_grad(set_to_none=True)
            out = model(x, lengths)
            wcce = sequence_loss(out["evidence"], lengths, y, "wcce")
            budget = wcce.new_zeros(())
            if spec.constrained:
                assert spec.rho is not None
                budget = _budget_loss(out["gate_g"], lengths, spec.rho)
            loss = wcce + spec.lambda_budget * budget
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)

        val = _split_eval(model, arrays, p, spec.seed, "val")
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": total / count,
            "val_ba": val["ba"],
            "val_mean_logit_ce": val["mean_logit_ce"],
            "sampler_hash": _permutation_hash(_epoch_permutation(len(arrays["train_y"]), spec.seed, epoch)),
        }
        compliant = True
        if spec.constrained:
            assert spec.rho is not None
            budget_val = _budget_eval(model, arrays, p, spec.seed, "val", spec.rho)
            row.update(budget_val)
            compliant = budget_val["budget_mae"] <= BUDGET_MAE_TOL
            row["budget_compliant"] = compliant
        history.append(row)
        if compliant and (best is None or _better(val, best)):
            best = dict(val)
            best_state = cpu_state(model)
            best_epoch = epoch
        if best_epoch is not None and epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    save_json(directory / "train_history.json", {"rows": history})
    if best_state is None or best is None or best_epoch is None:
        closest = min(history, key=lambda row: row.get("budget_mae", float("inf")))
        save_json(directory / "budget_invalid.json", {
            "status": "BUDGET_INVALID",
            "closest_budget_epoch": closest["epoch"],
            "closest_budget_mae": closest.get("budget_mae"),
        })
        return {"status": "BUDGET_INVALID", "run": spec.key}

    model.load_state_dict(best_state)
    shared_hash = hashlib.sha256(
        b"".join(value.cpu().numpy().tobytes() for value in _base_shared_state(spec.seed, p).values())
    ).hexdigest()
    save_torch(checkpoint_path, {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "case": spec.case,
        "seed": spec.seed,
        "gate_kind": spec.gate_kind,
        "rho": spec.rho,
        "lambda_budget": spec.lambda_budget,
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "best_val": best,
        "selection_rule": "constrained val budget then BA/CE/earliest" if spec.constrained else "val BA/CE/earliest",
        "hostname": platform.node(),
        "shared_init_hash": shared_hash,
    })
    _evaluate_model(config, spec, model)
    return {"status": "PASS", "run": spec.key, "best_epoch": best_epoch, "stopped_epoch": epoch, "best_val": best}


def train_pair_task(config: Config, task_id: int) -> dict[str, Any]:
    selected = _selected_lambda_by_rho(config)
    task = pair_tasks(selected)[task_id]
    results = [train_one(config, spec) for spec in task["specs"]]
    pair_hashes: dict[str, list[str]] = {}
    for spec in task["specs"]:
        history_path = config.results_dir / "runs" / spec.key / "train_history.json"
        if history_path.exists():
            rows = json.loads(history_path.read_text(encoding="utf-8"))["rows"]
            pair_hashes[spec.key] = [row["sampler_hash"] for row in rows]
    if len(pair_hashes) == 2:
        common_epochs = min(len(value) for value in pair_hashes.values())
        values = list(pair_hashes.values())
        if values[0][:common_epochs] != values[1][:common_epochs]:
            raise AssertionError(f"Batch trajectory mismatch in task {task_id}")
    return {
        "task_id": task_id,
        "seed": task["seed"],
        "rho": task["rho"],
        "order": task["order"],
        "hostname": platform.node(),
        "results": results,
        "loader_hash_match": True,
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


def _replay_gate(learned: torch.Tensor, lengths: torch.Tensor, mode: str,
                 ids: np.ndarray, seed_or_fraction: int | float | None) -> torch.Tensor:
    forced = learned.clone()
    if mode == "learned":
        return forced
    if mode == "mean":
        means = _budget_per_sample(learned, lengths)
        return means[:, None].expand_as(learned).clone()
    if mode == "shuffle":
        assert isinstance(seed_or_fraction, int)
        for i, length in enumerate(lengths.tolist()):
            rng = np.random.default_rng(
                paired_seed(seed_or_fraction, f"exp16_2:shuffle:{ids[i]}")
            )
            permutation = torch.from_numpy(rng.permutation(length)).long()
            forced[i, :length] = learned[i, permutation]
        return forced
    if mode == "shift":
        fraction = float(seed_or_fraction)
        for i, length in enumerate(lengths.tolist()):
            delta = int(round(fraction * length)) % length
            forced[i, :length] = torch.roll(learned[i, :length], shifts=-delta)
        return forced
    raise ValueError(mode)


def replay_one(config: Config, spec: ExpSpec) -> dict[str, Any]:
    if spec.gate_kind != "dynamic":
        raise ValueError("Phase2 applies only to dynamic gates")
    checkpoint = config.results_dir / "runs" / spec.key / "checkpoint.pt"
    if not checkpoint.exists():
        return {"status": "SKIPPED", "run": spec.key, "reason": "checkpoint unavailable"}

    model, p, arrays = _load_model(config, spec)
    model.eval()
    root = config.results_dir / "phase2_replay" / spec.key
    root.mkdir(parents=True, exist_ok=True)
    interventions: list[tuple[str, int | float | None]] = [("learned", None), ("mean", None)]
    interventions.extend(("shuffle", seed) for seed in SHUFFLE_SEEDS)
    interventions.extend(("shift", fraction) for fraction in SHIFT_FRACTIONS)

    for mode, parameter in interventions:
        if mode == "learned":
            tag = "learned"
        elif mode == "mean":
            tag = "mean"
        elif mode == "shuffle":
            tag = f"shuffle_{int(parameter)}"
        else:
            tag = f"shift_{int(round(float(parameter) * 100))}"
        directory = root / tag
        directory.mkdir(parents=True, exist_ok=True)
        traces: dict[str, np.ndarray] = {}
        native: dict[str, Any] = {"splits": {}}

        with torch.no_grad():
            for split in SPLITS:
                evidence_chunks: list[np.ndarray] = []
                l1_chunks: list[np.ndarray] = []
                l2_chunks: list[np.ndarray] = []
                offset = 0
                for x, y, lengths in exp16.loader(arrays, split, p, spec.seed):
                    ids = arrays[f"{split}_ids"][offset:offset + len(y)]
                    offset += len(y)
                    learned = model(x, lengths)["gate_g"]
                    forced = _replay_gate(learned, lengths, mode, ids, parameter)
                    out = model(x, lengths, forced_g=forced)
                    evidence_chunks.append(out["evidence"].numpy())
                    l1_chunks.append(out["spike"][0].numpy().astype(np.uint8))
                    l2_chunks.append(out["spike"][1].numpy().astype(np.uint8))

                evidence = np.concatenate(evidence_chunks)
                traces[f"{split}__L1__spike"] = np.concatenate(l1_chunks)
                traces[f"{split}__L2__spike"] = np.concatenate(l2_chunks)
                scores = np.asarray([
                    evidence[i, :length].mean(0)
                    for i, length in enumerate(arrays[f"{split}_lengths"])
                ])
                pred = scores.argmax(1)
                split_metrics = metrics(arrays[f"{split}_y"], pred)
                split_metrics["mean_logit_ce"] = float(
                    F.cross_entropy(
                        torch.from_numpy(scores),
                        torch.from_numpy(arrays[f"{split}_y"]),
                    ).item()
                )
                native["splits"][split] = split_metrics

        save_json(directory / "native.json", native)
        exp16._run_probes(directory, traces, arrays, spec, p)
    return {"status": "PASS", "run": spec.key}


def _probe_metric(probes: pd.DataFrame, *, case: str, seed: int,
                  aggregation: str, layer: str = "L2", decoder: str = "no_bias") -> float:
    rows = probes[
        (probes["case"] == case)
        & (probes["seed"] == seed)
        & (probes["layer"] == layer)
        & (probes["aggregation"] == aggregation)
        & (probes["decoder"] == decoder)
    ]
    if rows.empty:
        return float("nan")
    ordered = rows[rows["shuffle_seed"] == -1]
    chosen = ordered if not ordered.empty else rows
    return float(chosen["test_ba"].mean())


def finalize(config: Config) -> dict[str, Any]:
    selected = _selected_lambda_by_rho(config)
    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    specs = formal_specs(selected)

    native_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    invalid: list[str] = []
    for spec in specs:
        directory = config.results_dir / "runs" / spec.key
        if (directory / "budget_invalid.json").exists():
            invalid.append(spec.key)
            continue
        if not (directory / "checkpoint.pt").exists():
            raise FileNotFoundError(directory / "checkpoint.pt")
        native = json.loads((directory / "native.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
        row = {
            "case": spec.case,
            "seed": spec.seed,
            "rho": spec.rho,
            "lambda_budget": spec.lambda_budget,
            "best_epoch": checkpoint["best_epoch"],
            "stopped_epoch": checkpoint["stopped_epoch"],
            "hostname": checkpoint["hostname"],
        }
        for split in SPLITS:
            for metric_name, value in native["splits"][split].items():
                if isinstance(value, (int, float)):
                    row[f"{split}_{metric_name}"] = value
        row["train_test_gap"] = native["train_test_gap"]
        native_rows.append(row)
        probe_rows.extend(
            json.loads((directory / "probes.json").read_text(encoding="utf-8"))["rows"]
        )

    native_frame = pd.DataFrame(native_rows)
    probe_frame = pd.DataFrame(probe_rows)
    native_frame.to_csv(aggregate / "formal_native_runs.csv", index=False)
    probe_frame.to_csv(aggregate / "formal_probe_runs.csv", index=False)

    delta_rows: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        for rho in RHOS:
            tag = int(round(rho * 100))
            g_case, s_case = f"G{tag}", f"S{tag}"
            if f"{g_case}__seed{seed}" in invalid:
                continue
            g_native = native_frame[(native_frame.case == g_case) & (native_frame.seed == seed)]
            s_native = native_frame[(native_frame.case == s_case) & (native_frame.seed == seed)]
            if g_native.empty or s_native.empty:
                continue
            row = {
                "seed": seed,
                "rho": rho,
                "delta_native_pp": 100.0 * (
                    float(g_native.iloc[0]["test_ba"]) - float(s_native.iloc[0]["test_ba"])
                ),
            }
            for aggregation, short in (
                ("whole_count", "wholecount"),
                ("fixed250_ordered", "fix250"),
                ("relative10_ordered", "relative10"),
            ):
                gv = _probe_metric(probe_frame, case=g_case, seed=seed, aggregation=aggregation)
                sv = _probe_metric(probe_frame, case=s_case, seed=seed, aggregation=aggregation)
                row[f"delta_{short}_pp"] = 100.0 * (gv - sv)
            row["delta_collapse_pp"] = row["delta_relative10_pp"] - row["delta_wholecount_pp"]
            delta_rows.append(row)

    delta_frame = pd.DataFrame(delta_rows)
    delta_frame.to_csv(aggregate / "adaptive_gating_gain_per_seed.csv", index=False)
    if not delta_frame.empty:
        delta_frame.groupby("rho", as_index=False).mean(numeric_only=True).to_csv(
            aggregate / "adaptive_gating_gain_mean.csv", index=False
        )

    replay_rows: list[dict[str, Any]] = []
    for spec in dynamic_formal_specs(selected):
        root = config.results_dir / "phase2_replay" / spec.key
        if not root.exists():
            continue
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            native = json.loads((directory / "native.json").read_text(encoding="utf-8"))
            replay_rows.append({
                "case": spec.case,
                "seed": spec.seed,
                "intervention": directory.name,
                "test_ba": native["splits"]["test"]["ba"],
            })
    replay_frame = pd.DataFrame(replay_rows)
    if not replay_frame.empty:
        replay_frame.to_csv(aggregate / "phase2_replay_native.csv", index=False)
        causal_rows: list[dict[str, Any]] = []
        for (case, seed), group in replay_frame.groupby(["case", "seed"]):
            values = group.set_index("intervention")["test_ba"]
            if "learned" not in values or "mean" not in values:
                continue
            shuffle_names = [name for name in values.index if name.startswith("shuffle_")]
            shift_names = [name for name in values.index if name.startswith("shift_")]
            shuffle_mean = values[shuffle_names].mean()
            shift_mean = values[shift_names].mean()
            causal_rows.append({
                "case": case,
                "seed": seed,
                "delta_variation_pp": 100.0 * (values["learned"] - values["mean"]),
                "delta_shuffle_pp": 100.0 * (values["learned"] - shuffle_mean),
                "delta_alignment_pp": 100.0 * (values["learned"] - shift_mean),
            })
        pd.DataFrame(causal_rows).to_csv(aggregate / "phase2_causal_deltas.csv", index=False)

    summary = {
        "status": "PASS" if not invalid else "PARTIAL_BUDGET_INVALID",
        "budget_invalid_runs": invalid,
        "formal_run_count_expected": 24,
        "formal_run_count_valid": len(native_rows),
        "interpretation": {
            "level1": "G_rho > S_rho: adaptive gating gain under matched mean gate budget",
            "level2": "Learned > Mean: within-sequence temporal nonuniformity is functionally used",
            "level3": "Learned > Shuffle/Shift: specific temporal placement is functionally important",
        },
    }
    save_json(aggregate / "summary.json", summary)
    return summary


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("plan")
    phase0 = sub.add_parser("phase0")
    phase0.add_argument("--task-id", type=int, required=True)
    sub.add_parser("finalize-calibration")
    pair = sub.add_parser("train-pair")
    pair.add_argument("--task-id", type=int, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--task-id", type=int, required=True)
    sub.add_parser("finalize")
    args = parser.parse_args(argv)

    configure_cpu()
    config = config_from_args(args)
    if args.command == "prepare":
        print(json.dumps(prepare(config), indent=2))
    elif args.command == "plan":
        payload: dict[str, Any] = {
            "phase0": [asdict(spec) | {"key": spec.key} for spec in calibration_specs()],
            "phase0_count": len(calibration_specs()),
        }
        selection_path = config.results_dir / "phase0_selection.json"
        if selection_path.exists():
            selected = _selected_lambda_by_rho(config)
            payload["formal"] = [asdict(spec) | {"key": spec.key} for spec in formal_specs(selected)]
            payload["pair_tasks"] = [
                {
                    **{key: value for key, value in task.items() if key != "specs"},
                    "specs": [asdict(spec) for spec in task["specs"]],                }
                for task in pair_tasks(selected)
            ]
        print(json.dumps(payload, indent=2))
    elif args.command == "phase0":
        specs = calibration_specs()
        if not 0 <= args.task_id < len(specs):
            parser.error(f"task-id must be in [0, {len(specs)-1}]")
        print(json.dumps(train_calibration(config, specs[args.task_id]), indent=2))
    elif args.command == "finalize-calibration":
        print(json.dumps(finalize_calibration(config), indent=2))
    elif args.command == "train-pair":
        tasks = pair_tasks(_selected_lambda_by_rho(config))
        if not 0 <= args.task_id < len(tasks):
            parser.error(f"task-id must be in [0, {len(tasks)-1}]")
        print(json.dumps(train_pair_task(config, args.task_id), indent=2))
    elif args.command == "replay":
        specs = dynamic_formal_specs(_selected_lambda_by_rho(config))
        if not 0 <= args.task_id < len(specs):
            parser.error(f"task-id must be in [0, {len(specs)-1}]")
        print(json.dumps(replay_one(config, specs[args.task_id]), indent=2))
    elif args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))


if __name__ == "__main__":
    main()