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
from torch import nn
import torch.nn.functional as F

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_8_1_hidden_quantization_ablation as exp81


EXPERIMENT_ID = "experiment_8_1_1_two_layer_mt_factorial"
PROTOCOL_VERSION = "two_layer_mt_factorial_v1"
SEEDS = exp73.SEEDS
WIDTH = exp73.HIDDEN_WIDTH
SHIFTS = (2, 3, 4)
THRESHOLD = float(exp73.THRESHOLD)
THRESHOLD_MULTIPLIERS = (0.5, 1.0, 1.5)
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
REGULARIZATION = exp73.REGULARIZATION
PROBE_STATES = ("pre_reset", "communication")
PROBE_AGGREGATIONS = ("whole_mean", "fixed250_ordered_mean")
LAYERS = ("l1", "l2")

# 2x2 factorial: L1 coding x L2 coding. All four methods are width matched.
METHOD_CONFIG: dict[str, dict[str, str]] = {
    "bb": {"l1_coding": "binary", "l2_coding": "binary"},
    "mb": {"l1_coding": "hetero3", "l2_coding": "binary"},
    "bm": {"l1_coding": "binary", "l2_coding": "hetero3"},
    "mm": {"l1_coding": "hetero3", "l2_coding": "hetero3"},
}
METHOD_ORDER = tuple(METHOD_CONFIG)
EXPECTED_RUNS = len(METHOD_ORDER) * len(SEEDS)


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.method}__a2_wcce__{REGULARIZATION}__seed{self.seed}"

    @property
    def l1_coding(self) -> str:
        return METHOD_CONFIG[self.method]["l1_coding"]

    @property
    def l2_coding(self) -> str:
        return METHOD_CONFIG[self.method]["l2_coding"]


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(method, seed) for method in METHOD_ORDER for seed in SEEDS]


def validate_spec(spec: RunSpec) -> None:
    if spec.method not in METHOD_CONFIG:
        raise ValueError(spec.method)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _bank_widths(width: int = WIDTH) -> tuple[int, ...]:
    q, r = divmod(width, len(THRESHOLD_MULTIPLIERS))
    return tuple(q + (index < r) for index in range(len(THRESHOLD_MULTIPLIERS)))


def _alpha_vector(coding: str) -> torch.Tensor:
    if coding == "binary":
        return exp50.alpha_vector(WIDTH, SHIFTS)
    if coding == "hetero3":
        return torch.cat([exp50.alpha_vector(width, SHIFTS) for width in _bank_widths()])
    raise ValueError(coding)


def _threshold_vector(coding: str) -> torch.Tensor:
    if coding == "binary":
        return torch.full((WIDTH,), THRESHOLD, dtype=torch.float32)
    if coding == "hetero3":
        return torch.cat(
            [
                torch.full((width,), THRESHOLD * multiplier, dtype=torch.float32)
                for width, multiplier in zip(_bank_widths(), THRESHOLD_MULTIPLIERS, strict=True)
            ]
        )
    raise ValueError(coding)


def _groups(coding: str) -> list[dict[str, float | int]]:
    groups: list[dict[str, float | int]] = []
    start = 0
    if coding == "binary":
        banks: Iterable[tuple[int, float]] = ((WIDTH, 1.0),)
    elif coding == "hetero3":
        banks = zip(_bank_widths(), THRESHOLD_MULTIPLIERS, strict=True)
    else:
        raise ValueError(coding)
    for bank_index, (bank_width, multiplier) in enumerate(banks):
        for local in exp72.shift_groups(SHIFTS, width=int(bank_width)):
            count = int(local["count"])
            groups.append(
                {
                    "bank": int(bank_index),
                    "threshold_multiplier": float(multiplier),
                    "shift": int(local["shift"]),
                    "start": start,
                    "stop": start + count,
                    "count": count,
                }
            )
            start += count
    if start != WIDTH:
        raise RuntimeError((coding, start, WIDTH))
    return groups


def _lif(coding: str, beta: float) -> nn.Module:
    if coding == "binary":
        return exp401.MacroMultiSpikeLIF(
            beta=beta,
            threshold=THRESHOLD,
            max_spikes_per_dt=1,
            surrogate_slope=exp72.SURROGATE_SLOPE,
        )
    if coding == "hetero3":
        return exp81.VectorThresholdBinaryLIF(
            beta,
            _threshold_vector(coding),
            exp72.SURROGATE_SLOPE,
        )
    raise ValueError(coding)


class Exp811Net(nn.Module):
    """A2-compatible 128x128 local SNN with independently chosen L1/L2 coding."""

    def __init__(self, spec: RunSpec, n_classes: int, fs: float) -> None:
        super().__init__()
        validate_spec(spec)
        self.spec = spec
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        self.hidden_linears = nn.ModuleList(
            [
                nn.Linear(exp72.EXPECTED_CHANNELS, WIDTH, bias=False),
                nn.Linear(WIDTH, WIDTH, bias=False),
            ]
        )
        beta_hidden = math.exp(-(1000.0 / fs) / exp72.TAU_MEM_MS)
        self.l1_lif = _lif(spec.l1_coding, beta_hidden)
        self.l2_lif = _lif(spec.l2_coding, beta_hidden)
        self.register_buffer("alpha_0", _alpha_vector(spec.l1_coding))
        self.register_buffer("alpha_1", _alpha_vector(spec.l2_coding))
        self.output_linear = nn.Linear(WIDTH, n_classes, bias=False)

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != exp72.EXPECTED_CHANNELS:
            raise ValueError(channels)
        syn1 = torch.zeros(batch, WIDTH, device=x.device, dtype=x.dtype)
        mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, WIDTH, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(syn2)
        l1_comm: list[torch.Tensor] = []
        l2_comm: list[torch.Tensor] = []
        l1_pre: list[torch.Tensor] = []
        l2_pre: list[torch.Tensor] = []
        evidence: list[torch.Tensor] = []
        for timestep in range(steps):
            syn1 = self.alpha_0 * syn1 + self.hidden_linears[0](x[:, timestep])
            out1, mem1, pre1 = self.l1_lif(syn1, mem1)
            syn2 = self.alpha_1 * syn2 + self.hidden_linears[1](out1)
            out2, mem2, pre2 = self.l2_lif(syn2, mem2)
            l1_comm.append(out1)
            l2_comm.append(out2)
            l1_pre.append(pre1)
            l2_pre.append(pre2)
            evidence.append(self.output_linear(out2))
        return {
            "hidden_communication": (
                torch.stack(l1_comm, dim=1),
                torch.stack(l2_comm, dim=1),
            ),
            "hidden_pre_reset": (
                torch.stack(l1_pre, dim=1),
                torch.stack(l2_pre, dim=1),
            ),
            "evidence": torch.stack(evidence, dim=1),
        }


def new_model(spec: RunSpec, data: exp3.Data) -> Exp811Net:
    return Exp811Net(spec, len(data.labels), data.fs)


def _collect_activity(
    spec: RunSpec,
    state_splits: dict[
        str, dict[str, dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]]]
    ],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for layer, coding in (("l1", spec.l1_coding), ("l2", spec.l2_coding)):
        values, _, lengths = state_splits[layer]["communication"]["test"]
        output[layer] = exp81._communication_stats(values, lengths, _groups(coding), 1)
    return output


def _probe_ba(probes: dict[str, Any], layer: str, state: str, aggregation: str) -> float:
    return exp81._probe_ba(probes, layer, state, aggregation)


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    evaluation_path = _path(config.results_dir, "evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "checkpoints", spec.key, ".pt")
    if evaluation_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(evaluation_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)

    # Same initialization and sample-order seeds across all four factorial cells.
    exp3.seed_all(exp73._e2e_pair_seed(spec.seed, "model_init"))
    model = new_model(spec, data).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY
    )
    train_loader = exp73._raw_loaders(data, spec.seed, config.batch_size, True)["train"]
    eval_loaders = exp73._raw_loaders(data, spec.seed, config.batch_size, False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch, best_ba, best_loss = -1, -1.0, float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum, n_total = 0.0, 0
        for X, y, lengths in train_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            scores = exp81._valid_mean(trajectory["evidence"], lengths)
            loss = F.cross_entropy(scores, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics = exp81._evaluate_linear(model, eval_loaders["train"], device)
        val_metrics = exp81._evaluate_linear(model, eval_loaders["val"], device)
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )
        if exp73._checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE:
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "method_config": METHOD_CONFIG[spec.method],
            "threshold_multipliers": THRESHOLD_MULTIPLIERS,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )

    model.load_state_dict(best_state, strict=True)
    linear_metrics = {
        split: exp81._evaluate_linear(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    lif_metrics = {
        split: exp81._evaluate_lif_transfer(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    state_splits = exp81._extract_state_splits(model, eval_loaders, device)
    probes = exp81._fit_state_probes(state_splits, spec.seed, data.bin_steps)
    activity = _collect_activity(spec, state_splits)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "method_config": METHOD_CONFIG[spec.method],
        "threshold_bank_widths": list(_bank_widths()),
        "threshold_multipliers": list(THRESHOLD_MULTIPLIERS),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "linear_metrics": linear_metrics,
        "lif_transfer_metrics": lif_metrics,
        "representation_probes": probes,
        "activity": activity,
    }
    _save_json(evaluation_path, payload)
    history_path = _path(config.results_dir, "histories", spec.key, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return payload


def _method_row(payload: dict[str, Any]) -> dict[str, Any]:
    probes = payload["representation_probes"]
    row: dict[str, Any] = {
        "method": payload["spec"]["method"],
        "seed": payload["spec"]["seed"],
        "l1_coding": payload["method_config"]["l1_coding"],
        "l2_coding": payload["method_config"]["l2_coding"],
        "parameter_count": payload["parameter_count"],
        "linear_test_ba": payload["linear_metrics"]["test"]["balanced_accuracy"],
        "lif_test_ba": payload["lif_transfer_metrics"]["test"]["balanced_accuracy"],
        "best_epoch": payload["best_epoch"],
    }
    for layer in LAYERS:
        for state in PROBE_STATES:
            for aggregation in PROBE_AGGREGATIONS:
                short = "whole" if aggregation == "whole_mean" else "fixed250"
                row[f"{layer}_{state}_{short}_ba"] = _probe_ba(
                    probes, layer, state, aggregation
                )
    for short in ("whole", "fixed250"):
        row[f"l1_threshold_delta_{short}"] = (
            row[f"l1_communication_{short}_ba"] - row[f"l1_pre_reset_{short}_ba"]
        )
        row[f"l1_to_l2_transform_delta_{short}"] = (
            row[f"l2_pre_reset_{short}_ba"] - row[f"l1_communication_{short}_ba"]
        )
        row[f"l2_threshold_delta_{short}"] = (
            row[f"l2_communication_{short}_ba"] - row[f"l2_pre_reset_{short}_ba"]
        )
        row[f"l1_quantization_gap_{short}"] = -row[f"l1_threshold_delta_{short}"]
        row[f"l2_quantization_gap_{short}"] = -row[f"l2_threshold_delta_{short}"]
    return row


def _activity_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, groups in payload["activity"].items():
        for group in groups:
            rows.append(
                {
                    "method": payload["spec"]["method"],
                    "seed": payload["spec"]["seed"],
                    "layer": layer,
                    **group,
                }
            )
    return rows


CONTRASTS = (
    ("l1_mt_when_l2_binary", "mb", "bb"),
    ("l2_mt_when_l1_binary", "bm", "bb"),
    ("l2_mt_when_l1_mt", "mm", "mb"),
    ("l1_mt_when_l2_mt", "mm", "bm"),
    ("mm_vs_bb", "mm", "bb"),
)

FACTORIAL_METRICS = (
    "linear_test_ba",
    "lif_test_ba",
    "l1_pre_reset_fixed250_ba",
    "l1_communication_fixed250_ba",
    "l2_pre_reset_fixed250_ba",
    "l2_communication_fixed250_ba",
    "l1_threshold_delta_fixed250",
    "l1_to_l2_transform_delta_fixed250",
    "l2_threshold_delta_fixed250",
)


def _paired_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contrast, method_a, method_b in CONTRASTS:
        for seed in SEEDS:
            a = runs[(runs.method == method_a) & (runs.seed == seed)].iloc[0]
            b = runs[(runs.method == method_b) & (runs.seed == seed)].iloc[0]
            row: dict[str, Any] = {
                "contrast": contrast,
                "method_a": method_a,
                "method_b": method_b,
                "seed": seed,
            }
            for metric in FACTORIAL_METRICS:
                row[f"delta_{metric}"] = float(a[metric] - b[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def _factorial_effects(runs: pd.DataFrame) -> pd.DataFrame:
    """Seed-paired 2x2 main effects and L1xL2 interaction."""
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        by_method = {
            method: runs[(runs.method == method) & (runs.seed == seed)].iloc[0]
            for method in METHOD_ORDER
        }
        for metric in FACTORIAL_METRICS:
            bb = float(by_method["bb"][metric])
            mb = float(by_method["mb"][metric])
            bm = float(by_method["bm"][metric])
            mm = float(by_method["mm"][metric])
            rows.extend(
                [
                    {
                        "seed": seed,
                        "metric": metric,
                        "effect": "l1_mt_main_effect",
                        "value": 0.5 * ((mb - bb) + (mm - bm)),
                    },
                    {
                        "seed": seed,
                        "metric": metric,
                        "effect": "l2_mt_main_effect",
                        "value": 0.5 * ((bm - bb) + (mm - mb)),
                    },
                    {
                        "seed": seed,
                        "metric": metric,
                        "effect": "l1_x_l2_interaction",
                        "value": mm - mb - bm + bb,
                    },
                ]
            )
    return pd.DataFrame(rows)


def _flatten_agg_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for spec in run_specs():
        path = _path(config.results_dir, "evaluations", spec.key, ".json")
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp8.1.1 evaluation: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if len(payloads) != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} runs, got {len(payloads)}")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([_method_row(payload) for payload in payloads])
    runs.to_csv(config.results_dir / "method_runs.csv", index=False)

    metric_columns = [
        column
        for column in runs.columns
        if column not in {"method", "seed", "l1_coding", "l2_coding"}
    ]
    summary = runs.groupby("method", sort=False)[metric_columns].agg(["mean", "std"]).reset_index()
    _flatten_agg_columns(summary).to_csv(config.results_dir / "method_summary.csv", index=False)

    contrasts = _paired_contrasts(runs)
    contrasts.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_metrics = [column for column in contrasts if column.startswith("delta_")]
    contrast_summary = contrasts.groupby(
        ["contrast", "method_a", "method_b"], sort=False
    )[contrast_metrics].agg(["mean", "std"]).reset_index()
    _flatten_agg_columns(contrast_summary).to_csv(
        config.results_dir / "contrast_summary.csv", index=False
    )

    effects = _factorial_effects(runs)
    effects.to_csv(config.results_dir / "factorial_effect_runs.csv", index=False)
    effect_summary = effects.groupby(["metric", "effect"], sort=False)["value"].agg(
        ["mean", "std"]
    ).reset_index()
    effect_summary.to_csv(config.results_dir / "factorial_effect_summary.csv", index=False)

    activity_runs = pd.DataFrame(
        [row for payload in payloads for row in _activity_rows(payload)]
    )
    activity_runs.to_csv(config.results_dir / "activity_runs.csv", index=False)
    activity_summary = activity_runs.groupby(
        ["method", "layer", "threshold_multiplier", "shift"], sort=False
    )[
        [
            "mean_value_per_neuron_step",
            "fraction_nonzero",
            "fraction_gt1",
            "fraction_at_cap",
        ]
    ].agg(["mean", "std"]).reset_index()
    _flatten_agg_columns(activity_summary).to_csv(
        config.results_dir / "activity_summary.csv", index=False
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "methods": METHOD_CONFIG,
        "method_order": list(METHOD_ORDER),
        "seeds": list(SEEDS),
        "counts": {
            "methods": len(METHOD_ORDER),
            "seeds": len(SEEDS),
            "parallel_runs": EXPECTED_RUNS,
        },
        "architecture": {
            "l1_width": WIDTH,
            "l2_width": WIDTH,
            "l1_shifts": list(SHIFTS),
            "l2_shifts": list(SHIFTS),
            "threshold": THRESHOLD,
            "threshold_multipliers": list(THRESHOLD_MULTIPLIERS),
        },
        "training": "Exp7.3 A2-compatible end-to-end Linear/WCCE",
        "scope": "2x2 factorial over L1/L2 binary vs fixed heterogeneous-threshold coding",
        "primary_representation_metric": "l2_communication_fixed250_ba",
        "secondary_task_metric": "linear_test_ba",
        "diagnostics": {
            "states": list(PROBE_STATES),
            "aggregations": list(PROBE_AGGREGATIONS),
            "factorial_effects": [
                "l1_mt_main_effect",
                "l2_mt_main_effect",
                "l1_x_l2_interaction",
            ],
        },
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp8.1.1 two-layer MT factorial")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-runs")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--array-task-id", type=int, required=True)
    run_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = find_repo_root(args.repo_root)
    config = Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        max_epochs=args.max_epochs,
    )
    specs = run_specs()
    if args.command == "list-runs":
        for index, spec in enumerate(specs):
            print(index, spec.key)
        return
    if args.command == "run":
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        payload = run_one(specs[args.array_task_id], config, force=args.force)
        print(
            json.dumps(
                {
                    "key": specs[args.array_task_id].key,
                    "test_ba": payload["linear_metrics"]["test"]["balanced_accuracy"],
                },
                indent=2,
            )
        )
        return
    if args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
