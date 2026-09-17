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
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_8_0_3_phase_aware_hierarchical_readout as exp803
from scripts import experiment_8_0_4_phase_aware_hierarchical_readout_mean as exp804


EXPERIMENT_ID = "experiment_8_0_5_frozen_backbone_phase_readout"
PROTOCOL_VERSION = "frozen_backbone_phase_readout_v1"
SEEDS = exp804.SEEDS
ARCHITECTURE = exp804.ARCHITECTURE
ARCHITECTURE_SHIFTS = exp804.ARCHITECTURE_SHIFTS
BACKBONE_METHOD = "l1_l2_timeshared_count"
METHODS = (
    "l2_whole",
    "l1_l2_timeshared",
    "l1_fixed250_l2_whole_true_phase",
    "l1_fixed250_l2_whole_destroyed_phase",
)
EXPECTED_RUNS = len(METHODS) * len(SEEDS)
TRUE_PHASE_METHOD = "l1_fixed250_l2_whole_true_phase"
DESTROYED_PHASE_METHOD = "l1_fixed250_l2_whole_destroyed_phase"


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.method}__{ARCHITECTURE}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp803.exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = exp803.exp73.MAX_EPOCHS


@dataclass
class BaseFeatures:
    l1_count: torch.Tensor
    l2_count: torch.Tensor
    l1_fixed: torch.Tensor
    lengths: torch.Tensor
    y: torch.Tensor

    @property
    def n_bins(self) -> int:
        return int(self.l1_fixed.shape[1])

    @property
    def hidden_width(self) -> int:
        return int(self.l1_count.shape[1])


@dataclass
class HeadFeatures:
    x: torch.Tensor
    y: torch.Tensor
    l1_dim: int
    l2_dim: int
    n_bins: int

    @property
    def feature_dim(self) -> int:
        return int(self.x.shape[1])


def find_repo_root(start: Path | None = None) -> Path:
    return exp804.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / exp804.EXPERIMENT_ID
        / exp804.PROTOCOL_VERSION
    )


def source_checkpoint_path(repo_root: Path, seed: int) -> Path:
    key = f"{BACKBONE_METHOD}__{ARCHITECTURE}__seed{seed}"
    return source_results_dir(repo_root) / "checkpoints" / f"{key}.pt"


def run_specs() -> list[RunSpec]:
    return [RunSpec(method, seed) for method in METHODS for seed in SEEDS]


def validate_spec(spec: RunSpec) -> None:
    if spec.method not in METHODS:
        raise ValueError(spec.method)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _source_model(
    repo_root: Path,
    data: exp803.exp3.Data,
    seed: int,
    device: torch.device,
) -> exp804.Exp804Net:
    path = source_checkpoint_path(repo_root, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Exp8.0.4 source checkpoint for seed {seed}: {path}"
        )
    spec = exp803.RunSpec(BACKBONE_METHOD, seed)
    model = exp804.new_model(spec, data).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


def _extract_base_features(
    model: exp804.Exp804Net,
    loader: Iterable,
    bin_steps: int,
    device: torch.device,
) -> BaseFeatures:
    l1_counts: list[torch.Tensor] = []
    l2_counts: list[torch.Tensor] = []
    l1_fixed: list[torch.Tensor] = []
    lengths_all: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            lengths_d = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            l1 = trajectory["hidden_spikes"][0]
            l2 = trajectory["hidden_spikes"][1]
            valid = exp803.exp80._valid_mask(lengths_d, l1.shape[1]).to(
                l1.dtype
            ).unsqueeze(-1)
            l1_counts.append((l1 * valid).sum(dim=1).cpu())
            l2_counts.append((l2 * valid).sum(dim=1).cpu())
            l1_fixed.append(
                exp803.exp3.fixed_counts(l1, lengths_d, bin_steps).cpu()
            )
            lengths_all.append(lengths.cpu())
            ys.append(y.cpu())

    return BaseFeatures(
        l1_count=torch.cat(l1_counts, dim=0).float(),
        l2_count=torch.cat(l2_counts, dim=0).float(),
        l1_fixed=torch.cat(l1_fixed, dim=0).float(),
        lengths=torch.cat(lengths_all, dim=0).long(),
        y=torch.cat(ys, dim=0).long(),
    )


def phase_destroy_offsets(
    n_samples: int,
    n_bins: int,
    seed: int,
    split: str,
) -> torch.Tensor:
    """Deterministic, sample-specific non-zero cyclic offsets.

    C and D retain the same BxH phase feature dimensionality. D destroys only
    the alignment between absolute sequence phase and phase-feature block.
    """
    if n_bins <= 1:
        return torch.zeros(n_samples, dtype=torch.long)
    split_code = {"train": 101, "val": 211, "test": 307}[split]
    rng = np.random.default_rng(int(seed) * 1009 + split_code)
    return torch.from_numpy(
        rng.integers(1, n_bins, size=n_samples, dtype=np.int64)
    )


def roll_phase_bins(
    values: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"Expected [N,B,H], got {tuple(values.shape)}")
    if len(offsets) != len(values):
        raise ValueError("offset count must match sample count")
    return torch.stack(
        [
            torch.roll(values[i], shifts=int(offsets[i]), dims=0)
            for i in range(len(values))
        ],
        dim=0,
    )


def _global_phase_shift(values: torch.Tensor, shift: int) -> torch.Tensor:
    return torch.roll(values, shifts=int(shift), dims=1)


def build_head_features(
    base: BaseFeatures,
    method: str,
    seed: int,
    split: str,
    *,
    global_phase_shift: int = 0,
) -> HeadFeatures:
    if method not in METHODS:
        raise ValueError(method)

    denom = base.lengths.clamp_min(1).to(torch.float32).unsqueeze(1)
    l2_mean = base.l2_count / denom
    n_bins = base.n_bins
    hidden = base.hidden_width

    if method == "l2_whole":
        return HeadFeatures(
            x=l2_mean,
            y=base.y,
            l1_dim=0,
            l2_dim=hidden,
            n_bins=n_bins,
        )

    if method == "l1_l2_timeshared":
        l1_mean = base.l1_count / denom
        return HeadFeatures(
            x=torch.cat([l1_mean, l2_mean], dim=1),
            y=base.y,
            l1_dim=hidden,
            l2_dim=hidden,
            n_bins=n_bins,
        )

    phase = base.l1_fixed
    if method == TRUE_PHASE_METHOD:
        if global_phase_shift:
            phase = _global_phase_shift(phase, global_phase_shift)
    elif method == DESTROYED_PHASE_METHOD:
        offsets = phase_destroy_offsets(len(base.y), n_bins, seed, split)
        phase = roll_phase_bins(phase, offsets)

    phase_flat = phase.flatten(1) / denom
    return HeadFeatures(
        x=torch.cat([phase_flat, l2_mean], dim=1),
        y=base.y,
        l1_dim=n_bins * hidden,
        l2_dim=hidden,
        n_bins=n_bins,
    )


def head_init_seed(seed: int, method: str) -> int:
    """Pair true/destroyed phase initialization exactly."""
    family = (
        "phase_bank"
        if method in {TRUE_PHASE_METHOD, DESTROYED_PHASE_METHOD}
        else method
    )
    family_code = {
        "l2_whole": 11,
        "l1_l2_timeshared": 23,
        "phase_bank": 37,
    }[family]
    return int(seed) * 10007 + family_code


def new_head(feature_dim: int, n_classes: int, seed: int, method: str) -> nn.Linear:
    torch.manual_seed(head_init_seed(seed, method))
    return nn.Linear(feature_dim, n_classes, bias=False)


def _head_metrics(
    head: nn.Linear,
    features: HeadFeatures,
    device: torch.device,
) -> dict[str, float]:
    head.eval()
    with torch.no_grad():
        x = features.x.to(device)
        y = features.y.to(device)
        scores = head(x)
        loss = F.cross_entropy(scores, y)
        metrics = exp803.exp72._metrics(
            y.cpu().numpy(), scores.argmax(dim=1).cpu().numpy()
        )
    metrics["objective_loss"] = float(loss.cpu())
    return metrics


def _branch_scores(
    head: nn.Linear,
    features: HeadFeatures,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = features.x.to(device)
    weight = head.weight
    full = F.linear(x, weight)
    if features.l1_dim:
        l1 = F.linear(x[:, : features.l1_dim], weight[:, : features.l1_dim])
    else:
        l1 = torch.zeros_like(full)
    l2 = F.linear(
        x[:, features.l1_dim : features.l1_dim + features.l2_dim],
        weight[:, features.l1_dim : features.l1_dim + features.l2_dim],
    )
    return full, l1, l2


def _balanced_accuracy_from_scores(y: torch.Tensor, scores: torch.Tensor) -> float:
    return float(
        exp803.exp72._metrics(
            y.cpu().numpy(), scores.argmax(dim=1).cpu().numpy()
        )["balanced_accuracy"]
    )


def branch_ablation_metrics(
    head: nn.Linear,
    features: HeadFeatures,
    device: torch.device,
) -> dict[str, float]:
    head.eval()
    with torch.no_grad():
        full, l1, l2 = _branch_scores(head, features, device)
        y = features.y.to(device)
        full_ba = _balanced_accuracy_from_scores(y, full)
        l2_ba = _balanced_accuracy_from_scores(y, l2)
        if features.l1_dim:
            l1_ba = _balanced_accuracy_from_scores(y, l1)
            l2_removal_drop = full_ba - l1_ba
        else:
            l1_ba = float("nan")
            l2_removal_drop = float("nan")
        return {
            "full_ba": full_ba,
            "l1_only_ba": l1_ba,
            "l2_only_ba": l2_ba,
            "l1_removal_drop": full_ba - l2_ba,
            "l2_removal_drop": l2_removal_drop,
            "full_equals_branch_sum_max_error": float(
                (full - (l1 + l2)).abs().max().cpu()
            ),
        }


def _phase_shift_sweep(
    head: nn.Linear,
    base_test: BaseFeatures,
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for shift in range(base_test.n_bins):
        features = build_head_features(
            base_test,
            TRUE_PHASE_METHOD,
            seed,
            "test",
            global_phase_shift=shift,
        )
        metrics = _head_metrics(head, features, device)
        rows.append(
            {
                "shift_bins": int(shift),
                "shift_ms": float(shift * 250.0),
                "test_ba": float(metrics["balanced_accuracy"]),
            }
        )
    return rows


def _fit_head(
    spec: RunSpec,
    config: Config,
    train_features: HeadFeatures,
    val_features: HeadFeatures,
    n_classes: int,
    device: torch.device,
) -> tuple[nn.Linear, list[dict[str, float]], int, int]:
    head = new_head(
        train_features.feature_dim,
        n_classes,
        spec.seed,
        spec.method,
    ).to(device)
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=exp803.exp72.LR,
        weight_decay=exp803.exp72.WEIGHT_DECAY,
    )
    generator = torch.Generator().manual_seed(
        head_init_seed(spec.seed, spec.method) + 5003
    )
    loader = DataLoader(
        TensorDataset(train_features.x, train_features.y),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        head.train()
        total_loss = 0.0
        n_total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            n_total += len(y)

        train_metrics = _head_metrics(head, train_features, device)
        val_metrics = _head_metrics(head, val_features, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": total_loss / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )

        if exp803.exp73._checkpoint_improved(
            val_metrics, best_ba, best_loss
        ):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }

        if (
            epoch >= exp803.exp73.MIN_EPOCHS
            and best_epoch > 0
            and epoch - best_epoch >= exp803.exp73.PATIENCE
        ):
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No head checkpoint selected for {spec.key}")
    head.load_state_dict(best_state, strict=True)
    return head, history, best_epoch, stopped_epoch


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    eval_path = _path(config.results_dir, "evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "checkpoints", spec.key, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    data = exp803.exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)
    source_model = _source_model(config.repo_root, data, spec.seed, device)
    loaders = exp803.exp73._raw_loaders(
        data, spec.seed, config.batch_size, False
    )

    base = {
        split: _extract_base_features(
            source_model, loader, int(data.bin_steps), device
        )
        for split, loader in loaders.items()
    }
    features = {
        split: build_head_features(
            base_split, spec.method, spec.seed, split
        )
        for split, base_split in base.items()
    }

    head, history, best_epoch, stopped_epoch = _fit_head(
        spec,
        config,
        features["train"],
        features["val"],
        len(data.labels),
        device,
    )

    metrics = {
        split: _head_metrics(head, split_features, device)
        for split, split_features in features.items()
    }
    ablation = {
        split: branch_ablation_metrics(head, split_features, device)
        for split, split_features in features.items()
    }
    phase_shift = (
        _phase_shift_sweep(head, base["test"], spec.seed, device)
        if spec.method == TRUE_PHASE_METHOD
        else []
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "source_experiment_id": exp804.EXPERIMENT_ID,
            "source_protocol_version": exp804.PROTOCOL_VERSION,
            "source_backbone_method": BACKBONE_METHOD,
            "source_checkpoint": str(
                source_checkpoint_path(config.repo_root, spec.seed)
                .relative_to(config.repo_root)
            ),
            "feature_dim": features["train"].feature_dim,
            "l1_dim": features["train"].l1_dim,
            "l2_dim": features["train"].l2_dim,
            "n_bins": features["train"].n_bins,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            },
        },
        checkpoint_path,
    )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "source_experiment_id": exp804.EXPERIMENT_ID,
        "source_protocol_version": exp804.PROTOCOL_VERSION,
        "source_backbone_method": BACKBONE_METHOD,
        "source_checkpoint": str(
            source_checkpoint_path(config.repo_root, spec.seed)
            .relative_to(config.repo_root)
        ),
        "backbone_frozen": True,
        "score_normalization": "valid_length_mean",
        "feature_dim": features["train"].feature_dim,
        "l1_dim": features["train"].l1_dim,
        "l2_dim": features["train"].l2_dim,
        "n_bins": features["train"].n_bins,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "metrics": metrics,
        "branch_ablation": ablation,
        "phase_shift_sweep": phase_shift,
        "parameter_count": int(sum(p.numel() for p in head.parameters())),
    }
    _save_json(eval_path, payload)

    history_path = _path(config.results_dir, "histories", spec.key, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return payload


def _method_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": payload["spec"]["method"],
        "seed": int(payload["spec"]["seed"]),
        "train_ba": float(payload["metrics"]["train"]["balanced_accuracy"]),
        "val_ba": float(payload["metrics"]["val"]["balanced_accuracy"]),
        "test_ba": float(payload["metrics"]["test"]["balanced_accuracy"]),
        "test_objective_loss": float(payload["metrics"]["test"]["objective_loss"]),
        "best_epoch": int(payload["best_epoch"]),
        "feature_dim": int(payload["feature_dim"]),
        "l1_dim": int(payload["l1_dim"]),
        "l2_dim": int(payload["l2_dim"]),
        "parameter_count": int(payload["parameter_count"]),
    }


def _ablation_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": payload["spec"]["method"],
            "seed": int(payload["spec"]["seed"]),
            "split": split,
            **{key: float(value) for key, value in metrics.items()},
        }
        for split, metrics in payload["branch_ablation"].items()
    ]


def _phase_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": payload["spec"]["method"],
            "seed": int(payload["spec"]["seed"]),
            **{key: float(value) for key, value in row.items()},
        }
        for row in payload["phase_shift_sweep"]
    ]


def _paired_deltas(method_runs: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        (
            "true_phase_vs_destroyed_phase",
            TRUE_PHASE_METHOD,
            DESTROYED_PHASE_METHOD,
        ),
        (
            "true_phase_vs_timeshared",
            TRUE_PHASE_METHOD,
            "l1_l2_timeshared",
        ),
        (
            "timeshared_vs_l2",
            "l1_l2_timeshared",
            "l2_whole",
        ),
    )
    rows: list[dict[str, Any]] = []
    for name, treatment, control in comparisons:
        for seed in SEEDS:
            tr = method_runs[
                (method_runs.method == treatment)
                & (method_runs.seed == seed)
            ].iloc[0]
            ct = method_runs[
                (method_runs.method == control)
                & (method_runs.seed == seed)
            ].iloc[0]
            rows.append(
                {
                    "comparison": name,
                    "seed": int(seed),
                    "test_ba_delta": float(tr.test_ba - ct.test_ba),
                    "val_ba_delta": float(tr.val_ba - ct.val_ba),
                }
            )
    return pd.DataFrame(rows)


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for spec in run_specs():
        path = _path(config.results_dir, "evaluations", spec.key, ".json")
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp8.0.5 evaluation: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if len(payloads) != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} runs, got {len(payloads)}")

    config.results_dir.mkdir(parents=True, exist_ok=True)

    method_runs = pd.DataFrame([_method_row(payload) for payload in payloads])
    method_runs.to_csv(config.results_dir / "method_runs.csv", index=False)
    exp803.exp802._summarize(
        method_runs,
        ["method"],
        [
            "train_ba",
            "val_ba",
            "test_ba",
            "test_objective_loss",
            "best_epoch",
            "feature_dim",
            "l1_dim",
            "l2_dim",
            "parameter_count",
        ],
    ).to_csv(config.results_dir / "method_summary.csv", index=False)

    ablation_runs = pd.DataFrame(
        [row for payload in payloads for row in _ablation_rows(payload)]
    )
    ablation_runs.to_csv(
        config.results_dir / "branch_ablation_runs.csv", index=False
    )
    exp803.exp802._summarize(
        ablation_runs,
        ["method", "split"],
        [
            "full_ba",
            "l1_only_ba",
            "l2_only_ba",
            "l1_removal_drop",
            "l2_removal_drop",
            "full_equals_branch_sum_max_error",
        ],
    ).to_csv(
        config.results_dir / "branch_ablation_summary.csv", index=False
    )

    phase_rows = [
        row for payload in payloads for row in _phase_rows(payload)
    ]
    phase_runs = pd.DataFrame(phase_rows)
    if not phase_runs.empty:
        phase_runs.to_csv(
            config.results_dir / "phase_shift_runs.csv", index=False
        )
        exp803.exp802._summarize(
            phase_runs,
            ["method", "shift_bins", "shift_ms"],
            ["test_ba"],
        ).to_csv(
            config.results_dir / "phase_shift_summary.csv", index=False
        )

    paired = _paired_deltas(method_runs)
    paired.to_csv(config.results_dir / "paired_deltas.csv", index=False)
    exp803.exp802._summarize(
        paired,
        ["comparison"],
        ["test_ba_delta", "val_ba_delta"],
    ).to_csv(config.results_dir / "paired_delta_summary.csv", index=False)

    histories: list[pd.DataFrame] = []
    for spec in run_specs():
        path = _path(config.results_dir, "histories", spec.key, ".csv")
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp8.0.5 history: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "seed", spec.seed)
        frame.insert(0, "method", spec.method)
        histories.append(frame)
    pd.concat(histories, ignore_index=True).to_csv(
        config.results_dir / "history_runs.csv", index=False
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "architecture_shifts": [list(layer) for layer in ARCHITECTURE_SHIFTS],
        "source_experiment_id": exp804.EXPERIMENT_ID,
        "source_protocol_version": exp804.PROTOCOL_VERSION,
        "source_backbone_method": BACKBONE_METHOD,
        "backbone_frozen": True,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "counts": {
            "methods": len(METHODS),
            "seeds": len(SEEDS),
            "parallel_runs": EXPECTED_RUNS,
        },
        "training": (
            "readout-only bias-free linear CE on valid-length-normalized "
            "frozen L1/L2 spike-count features"
        ),
        "primary_comparison": (
            "l1_fixed250_l2_whole_true_phase - "
            "l1_fixed250_l2_whole_destroyed_phase"
        ),
        "phase_destroyed_control": (
            "sample-specific deterministic non-zero cyclic phase offset; "
            "same Bx128 L1 feature bank and same head dimensionality as true phase"
        ),
        "phase_pair_initialization": (
            "true-phase and destroyed-phase heads use the exact same "
            "initialization seed at each backbone seed"
        ),
        "diagnostics": [
            "full/L1-only/L2-only test BA and branch removal drops",
            "true-phase global circular-shift sweep over every phase bin",
        ],
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _resolve_config(args: argparse.Namespace) -> Config:
    root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    out = Path(args.results_dir).resolve() if args.results_dir else results_dir(root)
    return Config(
        repo_root=root,
        results_dir=out,
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        max_epochs=args.max_epochs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--batch-size", type=int, default=exp803.exp72.BATCH_SIZE
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--max-epochs", type=int, default=exp803.exp73.MAX_EPOCHS
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--array-task-id", type=int, default=None)
    run_parser.add_argument("--method", choices=METHODS, default=None)
    run_parser.add_argument("--seed", type=int, choices=SEEDS, default=None)
    run_parser.add_argument("--force", action="store_true")
    sub.add_parser("finalize")

    args = parser.parse_args()
    config = _resolve_config(args)

    if args.command == "finalize":
        print(json.dumps(finalize(config), indent=2, sort_keys=True))
        return

    specs = run_specs()
    if args.array_task_id is not None:
        if not 0 <= args.array_task_id < len(specs):
            raise ValueError(
                f"array-task-id must be in [0, {len(specs)-1}], "
                f"got {args.array_task_id}"
            )
        spec = specs[args.array_task_id]
    else:
        if args.method is None or args.seed is None:
            raise ValueError(
                "Provide --array-task-id or both --method and --seed"
            )
        spec = RunSpec(args.method, args.seed)

    payload = run_one(spec, config, force=args.force)
    print(
        json.dumps(
            {
                "spec": payload["spec"],
                "source_checkpoint": payload["source_checkpoint"],
                "best_epoch": payload["best_epoch"],
                "test_ba": payload["metrics"]["test"]["balanced_accuracy"],
                "branch_ablation": payload["branch_ablation"]["test"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
