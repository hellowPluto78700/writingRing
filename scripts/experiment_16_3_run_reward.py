#!/usr/bin/env python3
"""Exp16.3: run-reward shaping for sustained L2 firing.

Question
--------
Does linear repeated-spike reward make one persistent firing episode
artificially valuable, and can diminishing within-run reward reduce redundant
persistent activity while preserving whole-character classification?

All formal cases keep exactly one whole-character label and one sequence-level
cross-entropy. The SNN architecture, split, seeds, optimizer and temporal
constants are unchanged. Only the aggregation of L2 spikes before the existing
bias-free linear head changes.

The experiment also includes two checkpoint-only diagnostics:
1. scale compensation across Exp16.2 tau groups using RMS/absolute drive and
   per-neuron W2 row norms;
2. CoreBenchmark O0/WCCE vs O1/TSCE activity plus run-cap/early-tail probe
   diagnostics. These are descriptive/causal diagnostics and never select a
   formal Exp16.3 training case.
"""
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

from core_benchmark_v1.model import BenchmarkNet, valid_mask
from core_benchmark_v1.probes import fit_probe
from core_benchmark_v1.protocol import Protocol, Run, SPLITS, paired_seed
from core_benchmark_v1.storage import save_json, save_npz, save_torch
from core_benchmark_v1.training import cpu_state, metrics
from scripts import experiment_16_prefix_supervised_selective_memory as exp16
from scripts import experiment_16_2_matched_budget_selective_write as exp16_2
from scripts import analyze_experiment_16_2_sustained_firing_mechanism as mech16_2


EXPERIMENT_ID = "experiment_16_3_run_reward"
PROTOCOL_VERSION = "run_reward_v1"
FORMAL_SEEDS = (11, 23, 37)
SHIFTS = ((2, 3, 4), (2, 3, 4))
SUBLINEAR_KAPPA = 8.0
DIAGNOSTIC_CAPS = (1, 2, 4, 8, 16, 24)
EARLY_TAIL_CAPS = (4, 8, 16)
EXP16_2_RESULTS_REL = Path(
    "notebooks/artifacts/experiment_16_2_matched_budget_selective_write/"
    "matched_budget_selective_write_v1"
)


@dataclass(frozen=True)
class ExpSpec:
    case: str
    seed: int
    reward_kind: str
    reward_param: float | None = None

    @property
    def key(self) -> str:
        return f"{self.case}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    core_results_dir: Path
    exp16_2_results_dir: Path


FORMAL_CASES: tuple[tuple[str, str, float | None], ...] = (
    ("LIN", "linear", None),
    ("CAP4", "cap", 4.0),
    ("CAP8", "cap", 8.0),
    ("CAP16", "cap", 16.0),
    ("CAP24", "cap", 24.0),
    ("SUB8", "sublinear", SUBLINEAR_KAPPA),
    ("EP1", "episode", 1.0),
)


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
        exp16_2_results_dir=(
            args.exp16_2_results or root / EXP16_2_RESULTS_REL
        ).resolve(),
    )


def _run(case: str, seed: int) -> Run:
    return Run(case, seed, "16_3_run_reward", shifts=SHIFTS, objective="wcce")


def formal_specs() -> list[ExpSpec]:
    return [
        ExpSpec(case, seed, reward_kind, reward_param)
        for seed in FORMAL_SEEDS
        for case, reward_kind, reward_param in FORMAL_CASES
    ]


def scale_diagnostic_specs(config: Config) -> list[exp16_2.ExpSpec]:
    selected = exp16_2._selected_lambda_by_rho(
        exp16_2.Config(
            config.repo_root,
            config.exp16_2_results_dir,
            config.core_results_dir,
        )
    )
    return exp16_2.formal_specs(selected)


def objective_diagnostic_specs() -> list[tuple[str, int]]:
    return [
        (case, seed)
        for seed in FORMAL_SEEDS
        for case in ("O0", "O1")
    ]


def _base_shared_state(seed: int, p: Protocol) -> dict[str, torch.Tensor]:
    model = BenchmarkNet(_run("LIN", seed), p)
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _shared_init_hash(seed: int, p: Protocol) -> str:
    state = _base_shared_state(seed, p)
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].cpu().numpy().tobytes())
    return digest.hexdigest()


def _make_model(spec: ExpSpec, p: Protocol) -> BenchmarkNet:
    model = BenchmarkNet(_run(spec.case, spec.seed), p)
    model.load_state_dict(_base_shared_state(spec.seed, p), strict=True)
    return model


def _epoch_permutation(n: int, seed: int, epoch: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(
        paired_seed(seed, f"exp16_3:loader:train:epoch:{epoch}")
    )
    return torch.randperm(n, generator=generator)


def _permutation_hash(permutation: torch.Tensor) -> str:
    return hashlib.sha256(
        permutation.cpu().numpy().astype(np.int64).tobytes()
    ).hexdigest()


def _train_batches(
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
    epoch: int,
):
    permutation = _epoch_permutation(len(arrays["train_y"]), seed, epoch)
    for start in range(0, len(permutation), p.batch_size):
        chosen = permutation[start:start + p.batch_size].numpy()
        yield (
            torch.from_numpy(arrays["train_x"][chosen]),
            torch.from_numpy(arrays["train_y"][chosen]),
            torch.from_numpy(arrays["train_lengths"][chosen]),
        )


def _sublinear_value(run_length: torch.Tensor, kappa: float) -> torch.Tensor:
    return kappa * (1.0 - torch.exp(-run_length / kappa))


def marginal_reward_weights(
    spikes: torch.Tensor,
    lengths: torch.Tensor,
    reward_kind: str,
    reward_param: float | None,
) -> torch.Tensor:
    """Return straight-through marginal reward weights for each spike variable.

    Forward run positions are computed from detached binary spikes. Gradients
    flow through the original spike values multiplied by the marginal reward
    assigned to the *hypothetical current spike* given the detached previous
    run length.

    Thus a silent timestep after a gap still receives the first-spike reward
    gradient; a continuation beyond CAP-K receives zero repeated-spike reward.
    """
    if spikes.ndim != 3:
        raise ValueError("spikes must be [batch,time,neuron]")
    if lengths.shape != (spikes.shape[0],):
        raise ValueError("lengths must align with spikes")

    valid = valid_mask(lengths, spikes.shape[1]).unsqueeze(-1)
    detached = (spikes.detach() > 0.5) & valid
    prev_run = spikes.new_zeros(spikes.shape[0], spikes.shape[2])
    weights: list[torch.Tensor] = []

    for t in range(spikes.shape[1]):
        active = valid[:, t]
        if reward_kind == "linear":
            weight = active.to(spikes.dtype)
        elif reward_kind in {"cap", "episode"}:
            cap = 1.0 if reward_kind == "episode" else float(reward_param)
            if not math.isfinite(cap) or cap <= 0:
                raise ValueError("CAP/episode reward requires positive cap")
            weight = ((prev_run < cap) & active).to(spikes.dtype)
        elif reward_kind == "sublinear":
            kappa = float(reward_param)
            if not math.isfinite(kappa) or kappa <= 0:
                raise ValueError("Sublinear reward requires positive kappa")
            next_run = prev_run + 1.0
            weight = (
                _sublinear_value(next_run, kappa)
                - _sublinear_value(prev_run, kappa)
            ) * active.to(spikes.dtype)
        else:
            raise ValueError(f"Unknown reward kind: {reward_kind}")

        weights.append(weight)
        current = detached[:, t]
        prev_run = torch.where(current, prev_run + 1.0, torch.zeros_like(prev_run))

    return torch.stack(weights, dim=1)


def run_reward_feature(
    spikes: torch.Tensor,
    lengths: torch.Tensor,
    reward_kind: str,
    reward_param: float | None,
) -> torch.Tensor:
    weights = marginal_reward_weights(
        spikes, lengths, reward_kind, reward_param
    )
    return (spikes * weights).sum(1) / lengths.to(spikes.dtype)[:, None]


def reward_logits(
    model: BenchmarkNet,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
    spec: ExpSpec,
) -> torch.Tensor:
    feature = run_reward_feature(
        trajectory["spike"][1],
        lengths,
        spec.reward_kind,
        spec.reward_param,
    )
    return model.head(feature)


def _linear_logits(
    model: BenchmarkNet,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
) -> torch.Tensor:
    feature = run_reward_feature(trajectory["spike"][1], lengths, "linear", None)
    return model.head(feature)


def _better(candidate: dict[str, float], best: dict[str, float]) -> bool:
    return (
        candidate["ba"] > best["ba"] + 1e-12
        or (
            abs(candidate["ba"] - best["ba"]) <= 1e-12
            and candidate["ce"] < best["ce"] - 1e-12
        )
    )


def _split_eval(
    model: BenchmarkNet,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    spec: ExpSpec,
    split: str,
) -> dict[str, float]:
    model.eval()
    ys: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    losses = 0.0
    count = 0
    with torch.no_grad():
        for x, y, lengths in exp16.loader(arrays, split, p, spec.seed):
            out = model(x, lengths)
            logits = reward_logits(model, out, lengths, spec)
            ys.append(y.numpy())
            predictions.append(logits.argmax(1).numpy())
            losses += float(F.cross_entropy(logits, y, reduction="sum"))
            count += len(y)
    y_all = np.concatenate(ys)
    pred_all = np.concatenate(predictions)
    return {
        **metrics(y_all, pred_all),
        "ce": losses / count,
    }


def _longest_runs(spikes: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    result = np.zeros((len(spikes), spikes.shape[2]), dtype=np.int32)
    for i, length in enumerate(lengths.astype(int).tolist()):
        current = np.zeros(spikes.shape[2], dtype=np.int32)
        best = np.zeros(spikes.shape[2], dtype=np.int32)
        for row in spikes[i, :length]:
            current = (current + 1) * row.astype(np.int32)
            best = np.maximum(best, current)
        result[i] = best
    return result


def _episode_counts(spikes: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    result = np.zeros((len(spikes), spikes.shape[2]), dtype=np.float64)
    for i, length in enumerate(lengths.astype(int).tolist()):
        z = spikes[i, :length].astype(np.uint8)
        prev = np.vstack(
            (np.zeros((1, z.shape[1]), dtype=np.uint8), z[:-1])
        )
        result[i] = ((z == 1) & (prev == 0)).sum(axis=0)
    return result


def _activity_summary(
    traces: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    p: Protocol,
    case: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        lengths = arrays[f"{split}_lengths"]
        for layer_index, layer in enumerate(("L1", "L2")):
            z = traces[f"{split}__{layer}__spike"].astype(np.uint8)
            occupancy = np.asarray(
                [z[i, :int(length)].mean(0) for i, length in enumerate(lengths)]
            )
            longest = _longest_runs(z, lengths)
            episodes = _episode_counts(z, lengths)
            alpha = (
                1.0
                - 2.0
                ** -np.asarray(
                    [v for shift in SHIFTS[layer_index] for v in [shift]],
                    dtype=float,
                )
            )
            # BenchmarkNet assigns contiguous near-equal groups. Reconstruct the
            # exact per-neuron alpha vector from the locked shift tuple.
            q, r = divmod(p.width, len(SHIFTS[layer_index]))
            alpha_vec = np.asarray(
                [
                    1.0 - 2.0 ** (-shift)
                    for j, shift in enumerate(SHIFTS[layer_index])
                    for _ in range(q + (j < r))
                ],
                dtype=float,
            )
            unique = sorted(np.unique(alpha_vec))
            labels = {
                unique[0]: "short",
                unique[1]: "medium",
                unique[2]: "long",
            }
            for alpha_value in unique:
                chosen = np.isclose(alpha_vec, alpha_value)
                tau_ms = -(1000.0 / p.fs) / math.log(alpha_value)
                rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "split": split,
                        "layer": layer,
                        "tau_group": labels[alpha_value],
                        "alpha": alpha_value,
                        "tau_ms": tau_ms,
                        "mean_occupancy": float(occupancy[:, chosen].mean()),
                        "p90_neuron_occupancy": float(
                            np.quantile(occupancy[:, chosen].mean(0), 0.90)
                        ),
                        "fraction_neurons_mean_occ_gt_0p5": float(
                            (occupancy[:, chosen].mean(0) > 0.5).mean()
                        ),
                        "mean_longest_run": float(longest[:, chosen].mean()),
                        "p90_longest_run": float(
                            np.quantile(longest[:, chosen], 0.90)
                        ),
                        "mean_episode_count": float(episodes[:, chosen].mean()),
                    }
                )
    return rows


def _extract_and_evaluate(
    config: Config,
    spec: ExpSpec,
    model: BenchmarkNet,
) -> None:
    p, _, arrays = exp16._load_core(config)
    directory = config.results_dir / "runs" / spec.key
    traces: dict[str, np.ndarray] = {}
    native: dict[str, Any] = {
        "case": spec.case,
        "seed": spec.seed,
        "reward_kind": spec.reward_kind,
        "reward_param": spec.reward_param,
        "splits": {},
    }

    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            chunks: dict[str, list[np.ndarray]] = {
                "evidence": [],
                "L1__spike": [],
                "L2__spike": [],
                "L1__pre_reset": [],
                "L2__pre_reset": [],
            }
            reward_scores: list[np.ndarray] = []
            linear_scores: list[np.ndarray] = []
            ys: list[np.ndarray] = []
            for x, y, lengths in exp16.loader(arrays, split, p, spec.seed):
                out = model(x, lengths)
                chunks["evidence"].append(out["evidence"].numpy())
                chunks["L1__spike"].append(
                    out["spike"][0].numpy().astype(np.uint8)
                )
                chunks["L2__spike"].append(
                    out["spike"][1].numpy().astype(np.uint8)
                )
                chunks["L1__pre_reset"].append(
                    out["pre_reset"][0].numpy().astype(np.float32)
                )
                chunks["L2__pre_reset"].append(
                    out["pre_reset"][1].numpy().astype(np.float32)
                )
                reward_scores.append(
                    reward_logits(model, out, lengths, spec).numpy()
                )
                linear_scores.append(_linear_logits(model, out, lengths).numpy())
                ys.append(y.numpy())

            for name, pieces in chunks.items():
                traces[f"{split}__{name}"] = np.concatenate(pieces)

            y_all = np.concatenate(ys)
            reward_np = np.concatenate(reward_scores)
            linear_np = np.concatenate(linear_scores)
            reward_pred = reward_np.argmax(1)
            linear_pred = linear_np.argmax(1)
            native["splits"][split] = {
                "reward": {
                    **metrics(y_all, reward_pred),
                    "ce": float(
                        F.cross_entropy(
                            torch.from_numpy(reward_np),
                            torch.from_numpy(y_all),
                        )
                    ),
                },
                "linear_same_head": {
                    **metrics(y_all, linear_pred),
                    "ce": float(
                        F.cross_entropy(
                            torch.from_numpy(linear_np),
                            torch.from_numpy(y_all),
                        )
                    ),
                },
            }

    native["reward_train_test_gap"] = (
        native["splits"]["train"]["reward"]["ba"]
        - native["splits"]["test"]["reward"]["ba"]
    )
    save_json(directory / "native.json", native)
    save_npz(directory / "traces.npz", traces)
    exp16._run_probes(directory, traces, arrays, spec, p)
    pd.DataFrame(
        _activity_summary(traces, arrays, p, spec.case, spec.seed)
    ).to_csv(directory / "activity.csv", index=False)


def train_one(config: Config, spec: ExpSpec) -> dict[str, Any]:
    p, _, arrays = exp16._load_core(config)
    directory = config.results_dir / "runs" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    if checkpoint_path.exists():
        return {"status": "exists", "run": spec.key}

    model = _make_model(spec, p)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=p.learning_rate,
        weight_decay=p.weight_decay,
    )

    val0 = _split_eval(model, arrays, p, spec, "val")
    best = dict(val0)
    best_state = cpu_state(model)
    best_epoch = 0
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train_loss": None,
            "val_ba": val0["ba"],
            "val_ce": val0["ce"],
        }
    ]

    stopped_epoch = 0
    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, y, lengths in _train_batches(arrays, p, spec.seed, epoch):
            optimizer.zero_grad(set_to_none=True)
            out = model(x, lengths)
            logits = reward_logits(model, out, lengths, spec)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite loss")
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"{spec.key}: nonfinite gradient")
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)

        val = _split_eval(model, arrays, p, spec, "val")
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "val_ba": val["ba"],
            "val_ce": val["ce"],
            "sampler_hash": _permutation_hash(
                _epoch_permutation(len(arrays["train_y"]), spec.seed, epoch)
            ),
        }
        history.append(row)
        if _better(val, best):
            best = dict(val)
            best_state = cpu_state(model)
            best_epoch = epoch

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{spec.key} epoch={epoch} loss={total/count:.5f} "
                f"val_ba={val['ba']:.5f}",
                flush=True,
            )
            save_json(directory / "progress.json", row)

        stopped_epoch = epoch
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    save_json(directory / "train_history.json", {"rows": history})
    model.load_state_dict(best_state)
    save_torch(
        checkpoint_path,
        {
            "experiment": EXPERIMENT_ID,
            "protocol": PROTOCOL_VERSION,
            "case": spec.case,
            "seed": spec.seed,
            "reward_kind": spec.reward_kind,
            "reward_param": spec.reward_param,
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val": best,
            "selection_rule": (
                "native reward-readout validation BA, then native reward CE, "
                "then earliest epoch"
            ),
            "shared_init_hash": _shared_init_hash(spec.seed, p),
            "hostname": platform.node(),
        },
    )
    _extract_and_evaluate(config, spec, model)
    return {
        "status": "PASS",
        "run": spec.key,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val": best,
    }


def _tau_metadata(alpha: np.ndarray, fs: float) -> pd.DataFrame:
    unique = sorted(float(v) for v in np.unique(np.round(alpha, 8)))
    names = ("short", "medium", "long")
    rows = []
    for index, value in enumerate(unique):
        rows.append(
            {
                "alpha": value,
                "tau_group": names[index],
                "tau_ms": -(1000.0 / fs) / math.log(value),
                "one_minus_alpha": 1.0 - value,
            }
        )
    return pd.DataFrame(rows)


def run_scale_diagnostic(
    config: Config,
    task_id: int,
) -> dict[str, Any]:
    specs = scale_diagnostic_specs(config)
    if not 0 <= task_id < len(specs):
        raise IndexError(task_id)
    spec = specs[task_id]
    exp_config = exp16_2.Config(
        config.repo_root,
        config.exp16_2_results_dir,
        config.core_results_dir,
    )
    model, p, arrays = exp16_2._load_model(exp_config, spec)
    metadata = _tau_metadata(
        model.alpha_1.detach().cpu().numpy().astype(np.float64),
        p.fs,
    )
    alpha_vec = model.alpha_1.detach().cpu().numpy().astype(np.float64)
    weight_norm = (
        model.layers[1].weight.detach().cpu().numpy().astype(np.float64)
        ** 2
    ).sum(1) ** 0.5

    total_count = np.zeros(p.width, dtype=np.float64)
    raw_sum = np.zeros(p.width, dtype=np.float64)
    raw_abs = np.zeros(p.width, dtype=np.float64)
    raw_sq = np.zeros(p.width, dtype=np.float64)
    raw_pos = np.zeros(p.width, dtype=np.float64)
    raw_neg = np.zeros(p.width, dtype=np.float64)
    actual_sum = np.zeros(p.width, dtype=np.float64)
    actual_abs = np.zeros(p.width, dtype=np.float64)
    actual_sq = np.zeros(p.width, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for batch_index, (x, _, lengths) in enumerate(
            exp16.loader(arrays, "test", p, spec.seed)
        ):
            trace = mech16_2._trace_batch(
                model,
                x,
                lengths,
                p,
                verify=(batch_index == 0),
            )
            valid = valid_mask(lengths, x.shape[1]).numpy()
            raw = trace["raw_drive"].numpy()
            actual = trace["actual_drive"].numpy()
            for i, length in enumerate(lengths.tolist()):
                rv = raw[i, :length].astype(np.float64)
                av = actual[i, :length].astype(np.float64)
                total_count += length
                raw_sum += rv.sum(0)
                raw_abs += np.abs(rv).sum(0)
                raw_sq += np.square(rv).sum(0)
                raw_pos += np.maximum(rv, 0.0).sum(0)
                raw_neg += np.maximum(-rv, 0.0).sum(0)
                actual_sum += av.sum(0)
                actual_abs += np.abs(av).sum(0)
                actual_sq += np.square(av).sum(0)

    rows = []
    for neuron in range(p.width):
        alpha = float(alpha_vec[neuron])
        group = metadata.iloc[
            int(np.argmin(np.abs(metadata["alpha"].to_numpy() - alpha)))
        ]
        oma = 1.0 - alpha
        raw_rms = math.sqrt(raw_sq[neuron] / total_count[neuron])
        actual_rms = math.sqrt(actual_sq[neuron] / total_count[neuron])
        rows.append(
            {
                "case": spec.case,
                "seed": spec.seed,
                "neuron": neuron,
                "tau_group": group["tau_group"],
                "alpha": alpha,
                "tau_ms": float(group["tau_ms"]),
                "one_minus_alpha": oma,
                "w2_row_norm": weight_norm[neuron],
                "w2_row_norm_over_one_minus_alpha": weight_norm[neuron] / oma,
                "raw_mean": raw_sum[neuron] / total_count[neuron],
                "raw_mean_abs": raw_abs[neuron] / total_count[neuron],
                "raw_rms": raw_rms,
                "raw_positive_part_mean": raw_pos[neuron] / total_count[neuron],
                "raw_negative_part_mean": raw_neg[neuron] / total_count[neuron],
                "raw_rms_over_one_minus_alpha": raw_rms / oma,
                "actual_mean": actual_sum[neuron] / total_count[neuron],
                "actual_mean_abs": actual_abs[neuron] / total_count[neuron],
                "actual_rms": actual_rms,
                "actual_rms_over_one_minus_alpha": actual_rms / oma,
            }
        )

    directory = config.results_dir / "phase0_scale" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "scale.csv", index=False)
    return {"status": "PASS", "run": spec.key, "rows": len(rows)}


def _run_position_features_np(
    spikes: np.ndarray,
    lengths: np.ndarray,
    kind: str,
    param: float | None,
) -> np.ndarray:
    n, _, width = spikes.shape
    out = np.zeros((n, width), dtype=np.float64)
    for i, length in enumerate(lengths.astype(int).tolist()):
        prev_run = np.zeros(width, dtype=np.int32)
        total = np.zeros(width, dtype=np.float64)
        for row in spikes[i, :length].astype(np.uint8):
            if kind == "linear":
                weight = np.ones(width, dtype=np.float64)
            elif kind in {"cap", "episode"}:
                cap = 1.0 if kind == "episode" else float(param)
                weight = (prev_run < cap).astype(np.float64)
            elif kind == "sublinear":
                kappa = float(param)
                weight = kappa * (
                    np.exp(-prev_run / kappa)
                    - np.exp(-(prev_run + 1.0) / kappa)
                )
            else:
                raise ValueError(kind)
            total += row * weight
            prev_run = np.where(row > 0, prev_run + 1, 0)
        out[i] = total / float(length)
    return out


def _early_tail_features_np(
    spikes: np.ndarray,
    lengths: np.ndarray,
    cap: int,
) -> tuple[np.ndarray, np.ndarray]:
    early = _run_position_features_np(spikes, lengths, "cap", float(cap))
    linear = _run_position_features_np(spikes, lengths, "linear", None)
    return early, linear - early


def _probe_feature_set(
    features: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    p: Protocol,
    decoder: str,
) -> dict[str, Any]:
    scaler, model, _ = fit_probe(
        features["train"],
        arrays["train_y"],
        features["val"],
        arrays["val_y"],
        decoder,
        p,
    )
    row: dict[str, Any] = {
        "decoder": decoder,
        "dimension": features["train"].shape[1],
        "C": float(model.C),
    }
    for split in SPLITS:
        pred = model.predict(
            scaler.transform(features[split].astype(np.float64))
        )
        for name, value in metrics(arrays[f"{split}_y"], pred).items():
            row[f"{split}_{name}"] = value
    row["train_test_gap"] = row["train_ba"] - row["test_ba"]
    return row


def run_objective_diagnostic(
    config: Config,
    task_id: int,
) -> dict[str, Any]:
    specs = objective_diagnostic_specs()
    if not 0 <= task_id < len(specs):
        raise IndexError(task_id)
    case, seed = specs[task_id]
    p, _, arrays = exp16._load_core(config)
    trace_path = config.core_results_dir / "runs" / f"{case}__seed{seed}" / "traces.npz"
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    with np.load(trace_path, allow_pickle=False) as payload:
        traces = {
            key: np.asarray(payload[key])
            for key in payload.files
            if "__spike" in key
        }

    activity_rows = _activity_summary(traces, arrays, p, case, seed)
    probe_rows: list[dict[str, Any]] = []
    z = {
        split: traces[f"{split}__L2__spike"].astype(np.uint8)
        for split in SPLITS
    }

    reward_specs: list[tuple[str, str, float | None]] = [
        ("linear", "linear", None),
        *[
            (f"cap{cap}", "cap", float(cap))
            for cap in DIAGNOSTIC_CAPS
        ],
        ("sub8", "sublinear", SUBLINEAR_KAPPA),
    ]
    for name, kind, param in reward_specs:
        features = {
            split: _run_position_features_np(
                z[split],
                arrays[f"{split}_lengths"],
                kind,
                param,
            )
            for split in SPLITS
        }
        for decoder in ("no_bias", "affine"):
            row = _probe_feature_set(features, arrays, p, decoder)
            row.update(
                {
                    "objective_case": case,
                    "seed": seed,
                    "feature": name,
                    "family": "run_reward",
                    "cap": param if kind == "cap" else None,
                }
            )
            probe_rows.append(row)

    for cap in EARLY_TAIL_CAPS:
        early: dict[str, np.ndarray] = {}
        tail: dict[str, np.ndarray] = {}
        concat: dict[str, np.ndarray] = {}
        for split in SPLITS:
            e, t = _early_tail_features_np(
                z[split],
                arrays[f"{split}_lengths"],
                cap,
            )
            early[split] = e
            tail[split] = t
            concat[split] = np.concatenate((e, t), axis=1)
        for feature_name, features in (
            (f"early{cap}", early),
            (f"tail{cap}", tail),
            (f"early_tail{cap}", concat),
        ):
            for decoder in ("no_bias", "affine"):
                row = _probe_feature_set(features, arrays, p, decoder)
                row.update(
                    {
                        "objective_case": case,
                        "seed": seed,
                        "feature": feature_name,
                        "family": "early_tail",
                        "cap": cap,
                    }
                )
                probe_rows.append(row)

    directory = config.results_dir / "phase0_objective" / f"{case}__seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(activity_rows).to_csv(directory / "activity.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(directory / "run_tail_probes.csv", index=False)
    return {
        "status": "PASS",
        "objective_case": case,
        "seed": seed,
        "probe_rows": len(probe_rows),
    }


def prepare(config: Config) -> dict[str, Any]:
    p, lock, arrays = exp16._load_core(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    # Formal cases must start from the exact same parameter state for a seed.
    for seed in FORMAL_SEEDS:
        states = []
        for case, kind, param in FORMAL_CASES:
            model = _make_model(ExpSpec(case, seed, kind, param), p)
            states.append(model.state_dict())
        names = states[0].keys()
        for state in states[1:]:
            if any(not torch.equal(states[0][name], state[name]) for name in names):
                raise AssertionError(f"Formal initialization mismatch for seed {seed}")

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "core_dataset_hash": lock["dataset_hash"],
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_cases": [
            {
                "case": case,
                "reward_kind": kind,
                "reward_param": param,
            }
            for case, kind, param in FORMAL_CASES
        ],
        "formal_run_count": len(formal_specs()),
        "scale_diagnostic_run_count": len(scale_diagnostic_specs(config)),
        "objective_diagnostic_run_count": len(objective_diagnostic_specs()),
        "architecture": {
            "width": p.width,
            "shifts": [list(v) for v in SHIFTS],
            "tau_mem_ms": p.tau_mem_ms,
            "native_head_bias": False,
        },
        "hard_contracts": {
            "one_label_one_sequence_level_ce": True,
            "no_tsce_in_formal_training": True,
            "duration_normalization": "all reward features divided by valid T",
            "same_parameter_initialization_per_seed": True,
            "epoch_specific_deterministic_sampler": True,
            "checkpoint_selection": (
                "native reward-readout validation BA, then validation CE, "
                "then earliest epoch"
            ),
            "test_never_used_for_training_or_selection": True,
            "phase0_diagnostics_never_select_formal_cases": True,
            "reward_gradient_rule": (
                "detached forward run position defines marginal reward; "
                "gradient flows through current spike weighted by that marginal reward"
            ),
        },
        "diagnostics": {
            "scale": (
                "Exp16.2 test-set raw/effective drive signed mean, abs mean, RMS, "
                "positive/negative parts, and W2 row norm by tau"
            ),
            "objective": (
                "CoreBenchmark O0/WCCE vs O1/TSCE activity plus run-cap and "
                "early/tail retrained probes"
            ),
        },
    }
    save_json(config.results_dir / "protocol.json", payload)
    return payload


def _probe_metric(
    frame: pd.DataFrame,
    case: str,
    seed: int,
    aggregation: str,
    decoder: str = "no_bias",
) -> float:
    chosen = frame[
        (frame["case"] == case)
        & (frame["seed"] == seed)
        & (frame["layer"] == "L2")
        & (frame["aggregation"] == aggregation)
        & (frame["decoder"] == decoder)
    ]
    if chosen.empty:
        return float("nan")
    ordered = chosen[chosen["shuffle_seed"] == -1]
    if not ordered.empty:
        chosen = ordered
    return float(chosen["test_ba"].mean())


def finalize(config: Config) -> dict[str, Any]:
    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)

    # Phase0 scale diagnostic.
    scale_frames = []
    for spec in scale_diagnostic_specs(config):
        path = config.results_dir / "phase0_scale" / spec.key / "scale.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        scale_frames.append(pd.read_csv(path))
    scale = pd.concat(scale_frames, ignore_index=True)
    scale.to_csv(aggregate / "scale_compensation_neurons.csv", index=False)
    scale_group = (
        scale.groupby(["case", "seed", "tau_group"], as_index=False)
        .mean(numeric_only=True)
    )
    scale_group.to_csv(aggregate / "scale_compensation_tau.csv", index=False)

    # Phase0 objective/run-tail diagnostic.
    objective_activity = []
    objective_probes = []
    for case, seed in objective_diagnostic_specs():
        directory = config.results_dir / "phase0_objective" / f"{case}__seed{seed}"
        objective_activity.append(pd.read_csv(directory / "activity.csv"))
        objective_probes.append(pd.read_csv(directory / "run_tail_probes.csv"))
    objective_activity_frame = pd.concat(objective_activity, ignore_index=True)
    objective_probe_frame = pd.concat(objective_probes, ignore_index=True)
    objective_activity_frame.to_csv(
        aggregate / "objective_activity.csv", index=False
    )
    objective_probe_frame.to_csv(
        aggregate / "run_tail_probe_diagnostic.csv", index=False
    )

    # Formal training.
    native_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    activity_rows: list[pd.DataFrame] = []
    for spec in formal_specs():
        directory = config.results_dir / "runs" / spec.key
        if not (directory / "checkpoint.pt").exists():
            raise FileNotFoundError(directory / "checkpoint.pt")
        native = json.loads((directory / "native.json").read_text())
        checkpoint = torch.load(
            directory / "checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        row: dict[str, Any] = {
            "case": spec.case,
            "seed": spec.seed,
            "reward_kind": spec.reward_kind,
            "reward_param": spec.reward_param,
            "best_epoch": checkpoint["best_epoch"],
            "stopped_epoch": checkpoint["stopped_epoch"],
            "hostname": checkpoint["hostname"],
        }
        for split in SPLITS:
            for readout in ("reward", "linear_same_head"):
                for metric_name, value in native["splits"][split][readout].items():
                    row[f"{split}_{readout}_{metric_name}"] = value
        row["reward_train_test_gap"] = native["reward_train_test_gap"]
        native_rows.append(row)
        probe_rows.extend(
            json.loads((directory / "probes.json").read_text())["rows"]
        )
        activity_rows.append(pd.read_csv(directory / "activity.csv"))

    native_frame = pd.DataFrame(native_rows)
    probe_frame = pd.DataFrame(probe_rows)
    activity_frame = pd.concat(activity_rows, ignore_index=True)
    native_frame.to_csv(aggregate / "formal_native.csv", index=False)
    probe_frame.to_csv(aggregate / "formal_probes.csv", index=False)
    activity_frame.to_csv(aggregate / "formal_activity.csv", index=False)

    # Primary deltas relative to the exact linear-WCCE parameterization.
    delta_rows = []
    for seed in FORMAL_SEEDS:
        base_native = native_frame[
            (native_frame.case == "LIN") & (native_frame.seed == seed)
        ].iloc[0]
        base_activity = activity_frame[
            (activity_frame.case == "LIN")
            & (activity_frame.seed == seed)
            & (activity_frame.split == "test")
            & (activity_frame.layer == "L2")
        ]
        base_occ = float(base_activity["mean_occupancy"].mean())
        base_run = float(base_activity["mean_longest_run"].mean())
        for spec in [s for s in formal_specs() if s.seed == seed]:
            current = native_frame[
                (native_frame.case == spec.case)
                & (native_frame.seed == seed)
            ].iloc[0]
            current_activity = activity_frame[
                (activity_frame.case == spec.case)
                & (activity_frame.seed == seed)
                & (activity_frame.split == "test")
                & (activity_frame.layer == "L2")
            ]
            row = {
                "case": spec.case,
                "seed": seed,
                "reward_kind": spec.reward_kind,
                "reward_param": spec.reward_param,
                "delta_native_reward_pp": 100.0
                * (
                    float(current["test_reward_ba"])
                    - float(base_native["test_reward_ba"])
                ),
                "delta_linear_same_head_pp": 100.0
                * (
                    float(current["test_linear_same_head_ba"])
                    - float(base_native["test_linear_same_head_ba"])
                ),
                "delta_l2_occupancy": float(
                    current_activity["mean_occupancy"].mean() - base_occ
                ),
                "delta_l2_longest_run": float(
                    current_activity["mean_longest_run"].mean() - base_run
                ),
            }
            for aggregation, short in (
                ("whole_count", "wholecount"),
                ("fixed250_ordered", "fix250"),
                ("relative10_ordered", "relative10"),
            ):
                value = _probe_metric(
                    probe_frame,
                    spec.case,
                    seed,
                    aggregation,
                )
                base_value = _probe_metric(
                    probe_frame,
                    "LIN",
                    seed,
                    aggregation,
                )
                row[f"delta_{short}_pp"] = 100.0 * (value - base_value)
            delta_rows.append(row)

    delta_frame = pd.DataFrame(delta_rows)
    delta_frame.to_csv(
        aggregate / "formal_deltas_vs_linear.csv",
        index=False,
    )
    delta_frame.groupby("case", as_index=False).mean(numeric_only=True).to_csv(
        aggregate / "formal_deltas_vs_linear_mean.csv",
        index=False,
    )

    summary = {
        "status": "PASS",
        "formal_run_count": len(native_rows),
        "formal_run_count_expected": len(formal_specs()),
        "primary_question": (
            "Does diminishing reward within one sustained firing episode reduce "
            "persistent activity while preserving or improving whole-character classification?"
        ),
        "interpretation_contract": {
            "tail_redundancy": (
                "Phase0 early+tail probe ~= early-only supports low incremental "
                "information in the late run tail."
            ),
            "objective_comparison": (
                "O0 vs O1 is descriptive evidence that objective changes the "
                "temporal regime; it does not isolate repeated-count causality."
            ),
            "formal_mechanism": (
                "If diminishing-run-reward cases reduce occupancy/longest-run "
                "without harming test BA and especially improve transfer, that "
                "supports repeated-reward incentive as a causal contributor."
            ),
        },
    }
    save_json(aggregate / "summary.json", summary)
    return summary


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
    parser.add_argument("--exp16-2-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("plan")
    scale = sub.add_parser("diagnose-scale")
    scale.add_argument("--task-id", type=int, required=True)
    objective = sub.add_parser("diagnose-objective")
    objective.add_argument("--task-id", type=int, required=True)
    train = sub.add_parser("train")
    train.add_argument("--task-id", type=int, required=True)
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
                    "formal": [
                        asdict(spec) | {"key": spec.key}
                        for spec in formal_specs()
                    ],
                    "scale_diagnostic": [
                        {"case": spec.case, "seed": spec.seed, "key": spec.key}
                        for spec in scale_diagnostic_specs(config)
                    ],
                    "objective_diagnostic": [
                        {"case": case, "seed": seed}
                        for case, seed in objective_diagnostic_specs()
                    ],
                },
                indent=2,
            )
        )
    elif args.command == "diagnose-scale":
        print(json.dumps(run_scale_diagnostic(config, args.task_id), indent=2))
    elif args.command == "diagnose-objective":
        print(
            json.dumps(
                run_objective_diagnostic(config, args.task_id),
                indent=2,
            )
        )
    elif args.command == "train":
        specs = formal_specs()
        if not 0 <= args.task_id < len(specs):
            parser.error(f"task-id must be in [0, {len(specs)-1}]")
        print(json.dumps(train_one(config, specs[args.task_id]), indent=2))
    elif args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))


if __name__ == "__main__":
    main()
