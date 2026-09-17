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
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_8_0_local_backbone_tau_sweep as exp80
from scripts import experiment_8_0_2_joint_l1_l2_supervision as exp802


EXPERIMENT_ID = "experiment_8_0_3_phase_aware_hierarchical_readout"
PROTOCOL_VERSION = "phase_aware_hierarchical_readout_v1"
SEEDS = exp73.SEEDS
ARCHITECTURE = "234x234"
ARCHITECTURE_SHIFTS = exp80.ARCHITECTURES[ARCHITECTURE]
METHODS = (
    "l2_only",
    "l1_l2_timeshared",
    "l1_fixed250_l2_whole",
    "l1_capacity_no_phase_l2_whole",
)
EXPECTED_RUNS = len(METHODS) * len(SEEDS)
TARGET_PROBE = "l1fixed250_l2whole"


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
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = exp73.MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp80.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


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


class Exp803Net(exp80.Exp80Net):
    """Exp8.0 local backbone with time-shared or absolute-bin L1 readout."""

    def __init__(
        self,
        method: str,
        n_classes: int,
        fs: float,
        n_steps: int,
        bin_steps: int,
    ) -> None:
        if method not in METHODS:
            raise ValueError(method)
        super().__init__(ARCHITECTURE_SHIFTS, n_classes, fs)
        self.method = method
        self.n_classes = int(n_classes)
        self.n_steps = int(n_steps)
        self.bin_steps = int(bin_steps)
        self.n_bins = int(math.ceil(self.n_steps / self.bin_steps))
        self.l1_timeshared_linear: nn.Linear | None = None
        self.l1_phase_linear: nn.Linear | None = None

        # The inherited backbone and W2 are initialized first. Therefore all
        # shared parameters preserve the Exp7.3/8.0 paired initialization.
        if method == "l1_l2_timeshared":
            self.l1_timeshared_linear = nn.Linear(
                exp80.HIDDEN_WIDTH, n_classes, bias=False
            )
        elif method in {
            "l1_fixed250_l2_whole",
            "l1_capacity_no_phase_l2_whole",
        }:
            self.l1_phase_linear = nn.Linear(
                self.n_bins * exp80.HIDDEN_WIDTH, n_classes, bias=False
            )

    def phase_weight_bank(self) -> torch.Tensor:
        if self.l1_phase_linear is None:
            raise RuntimeError("This method has no phase weight bank")
        return self.l1_phase_linear.weight.view(
            self.n_classes, self.n_bins, exp80.HIDDEN_WIDTH
        ).permute(1, 0, 2)


def new_model(spec: RunSpec, data: exp3.Data) -> Exp803Net:
    return Exp803Net(
        spec.method,
        len(data.labels),
        data.fs,
        int(data.Xtr.shape[1]),
        int(data.bin_steps),
    )


def _l1_branch_evidence(model: Exp803Net, l1: torch.Tensor) -> torch.Tensor:
    if model.method == "l2_only":
        return torch.zeros(
            l1.shape[0], l1.shape[1], model.n_classes,
            dtype=l1.dtype, device=l1.device,
        )
    if model.method == "l1_l2_timeshared":
        if model.l1_timeshared_linear is None:
            raise RuntimeError("Missing time-shared L1 head")
        return model.l1_timeshared_linear(l1)

    bank = model.phase_weight_bank()
    if model.method == "l1_fixed250_l2_whole":
        bin_index = torch.div(
            torch.arange(l1.shape[1], device=l1.device),
            model.bin_steps,
            rounding_mode="floor",
        ).clamp(max=model.n_bins - 1)
        weights_by_t = bank[bin_index]
        return torch.einsum("bth,tkh->btk", l1, weights_by_t)

    if model.method == "l1_capacity_no_phase_l2_whole":
        # Same B x K x H parameters as the phase-aware head but without access
        # to b(t). This is equivalent to feeding Whole(L1) into every bin slot.
        effective_weight = bank.sum(dim=0)
        return F.linear(l1, effective_weight)

    raise ValueError(model.method)


def _branch_evidence(
    model: Exp803Net, trajectory: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    l1 = trajectory["hidden_spikes"][0]
    l2_evidence = trajectory["evidence"]
    return _l1_branch_evidence(model, l1), l2_evidence


def _native_evidence(model: Exp803Net, trajectory: dict[str, Any]) -> torch.Tensor:
    l1_evidence, l2_evidence = _branch_evidence(model, trajectory)
    return l1_evidence + l2_evidence


def _scores(
    model: Exp803Net, trajectory: dict[str, Any], lengths: torch.Tensor
) -> torch.Tensor:
    return exp80._valid_mean(_native_evidence(model, trajectory), lengths)


def _loss(
    model: Exp803Net,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(_scores(model, trajectory, lengths), y)


def _explicit_feature_scores(
    model: Exp803Net,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Count-form implementation of the native per-timestep evidence score."""
    l1 = trajectory["hidden_spikes"][0]
    l2 = trajectory["hidden_spikes"][1]
    valid = exp80._valid_mask(lengths, l2.shape[1]).to(l2.dtype).unsqueeze(-1)
    l2_count = (l2 * valid).sum(dim=1)
    score = F.linear(l2_count, model.output_linear.weight)

    if model.method == "l1_l2_timeshared":
        if model.l1_timeshared_linear is None:
            raise RuntimeError("Missing time-shared L1 head")
        l1_count = (l1 * valid).sum(dim=1)
        score = score + F.linear(l1_count, model.l1_timeshared_linear.weight)
    elif model.method == "l1_fixed250_l2_whole":
        if model.l1_phase_linear is None:
            raise RuntimeError("Missing phase-aware L1 head")
        l1_fixed = exp3.fixed_counts(l1, lengths, model.bin_steps).flatten(1)
        score = score + model.l1_phase_linear(l1_fixed)
    elif model.method == "l1_capacity_no_phase_l2_whole":
        if model.l1_phase_linear is None:
            raise RuntimeError("Missing capacity-control L1 head")
        l1_count = (l1 * valid).sum(dim=1)
        repeated = l1_count.repeat(1, model.n_bins)
        score = score + model.l1_phase_linear(repeated)
    elif model.method != "l2_only":
        raise ValueError(model.method)

    denominator = lengths.clamp_min(1).to(score.dtype).unsqueeze(1)
    return score / denominator


def _evaluate_native(
    model: Exp803Net, loader: Iterable, device: torch.device
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    max_equivalence_error = 0.0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            trajectory = model.forward_trajectory(X)
            scores = _scores(model, trajectory, lengths)
            explicit = _explicit_feature_scores(model, trajectory, lengths)
            max_equivalence_error = max(
                max_equivalence_error,
                float((scores - explicit).abs().max().cpu()),
            )
            loss = F.cross_entropy(scores, y)
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)
    out = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    out["objective_loss"] = loss_sum / max(n_total, 1)
    out["max_accumulator_equivalence_error"] = max_equivalence_error
    return out


def _evaluate_lif_transfer(
    model: Exp803Net, loader: Iterable, device: torch.device
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            lengths_d = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            evidence = _native_evidence(model, trajectory)
            output_spikes = exp80._output_lif_spikes(evidence)
            mask = exp80._valid_mask(lengths_d, output_spikes.shape[1]).to(
                output_spikes.dtype
            ).unsqueeze(-1)
            counts = (output_spikes * mask).sum(dim=1)
            ys.append(y.numpy())
            preds.append(counts.argmax(dim=1).cpu().numpy())
    return exp72._metrics(np.concatenate(ys), np.concatenate(preds))


def _branch_score_diagnostics(
    model: Exp803Net, loader: Iterable, device: torch.device
) -> dict[str, float]:
    l1_scores: list[torch.Tensor] = []
    l2_scores: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for X, _, lengths in loader:
            X = X.to(device)
            lengths_d = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            l1_e, l2_e = _branch_evidence(model, trajectory)
            l1_scores.append(exp80._valid_mean(l1_e, lengths_d).cpu())
            l2_scores.append(exp80._valid_mean(l2_e, lengths_d).cpu())
    l1 = torch.cat(l1_scores, dim=0)
    l2 = torch.cat(l2_scores, dim=0)
    l1_rms = float(torch.sqrt(torch.mean(l1.square())))
    l2_rms = float(torch.sqrt(torch.mean(l2.square())))
    denom = l1_rms + l2_rms
    return {
        "l1_score_rms": l1_rms,
        "l2_score_rms": l2_rms,
        "l1_score_rms_fraction": l1_rms / denom if denom > 0 else 0.0,
    }


def _trained_head_diagnostics(model: Exp803Net) -> list[dict[str, Any]]:
    heads: list[tuple[str, torch.Tensor]] = [("l2", model.output_linear.weight)]
    if model.l1_timeshared_linear is not None:
        heads.append(("l1_timeshared", model.l1_timeshared_linear.weight))
    if model.l1_phase_linear is not None:
        heads.append(("l1_large_bank", model.l1_phase_linear.weight))

    raw: list[dict[str, Any]] = []
    norms: list[float] = []
    for name, tensor in heads:
        weight = tensor.detach().cpu().numpy().astype(np.float64)
        fro = float(np.linalg.norm(weight))
        rms = float(np.sqrt(np.mean(np.square(weight))))
        raw.append({"head": name, "coef_fro_norm": fro, "coef_rms": rms})
        norms.append(fro)
    total = float(sum(norms))
    return [
        {
            **row,
            "coef_norm_fraction": norm / total if total > 0 else 0.0,
        }
        for row, norm in zip(raw, norms, strict=True)
    ]


def _phase_structure_diagnostics(model: Exp803Net) -> dict[str, Any] | None:
    if model.l1_phase_linear is None:
        return None
    bank = model.phase_weight_bank().detach().cpu().numpy().astype(np.float64)
    flat = bank.reshape(model.n_bins, -1)
    norms = np.linalg.norm(flat, axis=1)
    norm_products = norms[:, None] * norms[None, :]
    cosine = np.divide(
        flat @ flat.T,
        norm_products,
        out=np.zeros_like(norm_products),
        where=norm_products > 0,
    )
    pair_values = cosine[np.triu_indices(model.n_bins, k=1)]
    adjacent = np.asarray([cosine[i, i + 1] for i in range(model.n_bins - 1)])
    centered = bank - bank.mean(axis=0, keepdims=True)
    rows = [
        {
            "bin_index": int(i),
            "coef_fro_norm": float(norms[i]),
            "coef_rms": float(np.sqrt(np.mean(np.square(bank[i])))),
            "norm_fraction": float(norms[i] / norms.sum()) if norms.sum() > 0 else 0.0,
        }
        for i in range(model.n_bins)
    ]
    return {
        "bins": rows,
        "mean_pairwise_cosine": float(pair_values.mean()) if pair_values.size else 1.0,
        "std_pairwise_cosine": float(pair_values.std()) if pair_values.size else 0.0,
        "mean_adjacent_cosine": float(adjacent.mean()) if adjacent.size else 1.0,
        "between_bin_weight_rms": float(np.sqrt(np.mean(np.square(centered)))),
    }


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    eval_path = _path(config.results_dir, "evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "checkpoints", spec.key, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)

    exp3.seed_all(exp73._e2e_pair_seed(spec.seed, "model_init"))
    model = new_model(spec, data).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY
    )
    train_loader = exp73._raw_loaders(data, spec.seed, config.batch_size, True)["train"]
    eval_loaders = exp73._raw_loaders(data, spec.seed, config.batch_size, False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_total = 0
        for X, y, lengths in train_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            loss = _loss(model, trajectory, lengths, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate_native(model, eval_loaders["train"], device)
        val_metrics = _evaluate_native(model, eval_loaders["val"], device)
        history.append({
            "epoch": float(epoch),
            "train_ba": float(train_metrics["balanced_accuracy"]),
            "val_ba": float(val_metrics["balanced_accuracy"]),
            "train_loss": train_loss_sum / max(n_total, 1),
            "val_loss": float(val_metrics["objective_loss"]),
        })

        if exp73._checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if (
            epoch >= exp73.MIN_EPOCHS
            and best_epoch > 0
            and epoch - best_epoch >= exp73.PATIENCE
        ):
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
            "architecture": ARCHITECTURE,
            "architecture_shifts": ARCHITECTURE_SHIFTS,
            "bin_steps": int(data.bin_steps),
            "n_bins": int(model.n_bins),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )

    model.load_state_dict(best_state, strict=True)
    native_metrics = {
        split: _evaluate_native(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    lif_metrics = {
        split: _evaluate_lif_transfer(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    probes, overlap, derived = exp802._fit_representation_probes(
        model, eval_loaders, data, spec, device
    )
    head_diagnostics = _trained_head_diagnostics(model)
    phase_structure = _phase_structure_diagnostics(model)
    branch_scores = _branch_score_diagnostics(model, eval_loaders["test"], device)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "architecture": ARCHITECTURE,
        "architecture_shifts": ARCHITECTURE_SHIFTS,
        "bin_steps": int(data.bin_steps),
        "n_bins": int(model.n_bins),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native_metrics,
        "lif_transfer_metrics": lif_metrics,
        "lif_penalty_test_ba": float(
            native_metrics["test"]["balanced_accuracy"]
            - lif_metrics["test"]["balanced_accuracy"]
        ),
        "probes": probes,
        "correctness_overlap": overlap,
        "derived": derived,
        "trained_head_diagnostics": head_diagnostics,
        "phase_structure_diagnostics": phase_structure,
        "test_branch_score_diagnostics": branch_scores,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    _save_json(eval_path, payload)

    history_path = _path(config.results_dir, "histories", spec.key, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return payload


def _method_row(payload: dict[str, Any]) -> dict[str, Any]:
    test = payload["native_metrics"]["test"]
    branch = payload["test_branch_score_diagnostics"]
    return {
        "method": payload["spec"]["method"],
        "seed": int(payload["spec"]["seed"]),
        "native_test_ba": float(test["balanced_accuracy"]),
        "native_val_ba": float(payload["native_metrics"]["val"]["balanced_accuracy"]),
        "lif_test_ba": float(payload["lif_transfer_metrics"]["test"]["balanced_accuracy"]),
        "lif_penalty": float(payload["lif_penalty_test_ba"]),
        "accumulator_equivalence_error": float(test["max_accumulator_equivalence_error"]),
        "l1_score_rms": float(branch["l1_score_rms"]),
        "l2_score_rms": float(branch["l2_score_rms"]),
        "l1_score_rms_fraction": float(branch["l1_score_rms_fraction"]),
        "best_epoch": int(payload["best_epoch"]),
        "parameter_count": int(payload["parameter_count"]),
    }


def _head_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": payload["spec"]["method"],
            "seed": int(payload["spec"]["seed"]),
            **row,
        }
        for row in payload["trained_head_diagnostics"]
    ]


def _phase_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    diag = payload["phase_structure_diagnostics"]
    if diag is None:
        return []
    return [
        {
            "method": payload["spec"]["method"],
            "seed": int(payload["spec"]["seed"]),
            **row,
        }
        for row in diag["bins"]
    ]


def _phase_structure_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    diag = payload["phase_structure_diagnostics"]
    if diag is None:
        return None
    return {
        "method": payload["spec"]["method"],
        "seed": int(payload["spec"]["seed"]),
        "mean_pairwise_cosine": float(diag["mean_pairwise_cosine"]),
        "std_pairwise_cosine": float(diag["std_pairwise_cosine"]),
        "mean_adjacent_cosine": float(diag["mean_adjacent_cosine"]),
        "between_bin_weight_rms": float(diag["between_bin_weight_rms"]),
    }


def _paired_deltas(
    method_runs: pd.DataFrame, probe_runs: pd.DataFrame
) -> pd.DataFrame:
    comparisons = (
        ("phase_vs_capacity", "l1_fixed250_l2_whole", "l1_capacity_no_phase_l2_whole"),
        ("phase_vs_timeshared", "l1_fixed250_l2_whole", "l1_l2_timeshared"),
        ("timeshared_vs_l2", "l1_l2_timeshared", "l2_only"),
    )
    target = probe_runs[probe_runs.feature == TARGET_PROBE][
        ["method", "seed", "test_ba"]
    ].rename(columns={"test_ba": "target_probe_test_ba"})
    merged = method_runs.merge(target, on=["method", "seed"], how="left")
    rows: list[dict[str, Any]] = []
    for name, treatment, control in comparisons:
        for seed in SEEDS:
            tr = merged[(merged.method == treatment) & (merged.seed == seed)].iloc[0]
            ct = merged[(merged.method == control) & (merged.seed == seed)].iloc[0]
            rows.append({
                "comparison": name,
                "seed": int(seed),
                "native_test_ba_delta": float(tr.native_test_ba - ct.native_test_ba),
                "lif_test_ba_delta": float(tr.lif_test_ba - ct.lif_test_ba),
                "target_probe_test_ba_delta": float(
                    tr.target_probe_test_ba - ct.target_probe_test_ba
                ),
            })
    return pd.DataFrame(rows)


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for spec in run_specs():
        path = _path(config.results_dir, "evaluations", spec.key, ".json")
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp8.0.3 evaluation: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if len(payloads) != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} runs, got {len(payloads)}")

    config.results_dir.mkdir(parents=True, exist_ok=True)

    method_runs = pd.DataFrame([_method_row(payload) for payload in payloads])
    method_runs.to_csv(config.results_dir / "method_runs.csv", index=False)
    exp802._summarize(
        method_runs,
        ["method"],
        [
            "native_test_ba",
            "native_val_ba",
            "lif_test_ba",
            "lif_penalty",
            "accumulator_equivalence_error",
            "l1_score_rms",
            "l2_score_rms",
            "l1_score_rms_fraction",
            "best_epoch",
            "parameter_count",
        ],
    ).to_csv(config.results_dir / "method_summary.csv", index=False)

    probe_runs = pd.DataFrame(
        [row for payload in payloads for row in exp802._probe_rows(payload)]
    )
    probe_runs.to_csv(config.results_dir / "probe_runs.csv", index=False)
    exp802._summarize(
        probe_runs,
        ["method", "feature"],
        ["train_ba", "val_ba", "test_ba", "train_test_gap", "feature_dim", "probe_C"],
    ).to_csv(config.results_dir / "probe_summary.csv", index=False)

    fusion_rows = [
        {
            "method": payload["spec"]["method"],
            "seed": int(payload["spec"]["seed"]),
            **{key: float(value) for key, value in payload["derived"].items()},
        }
        for payload in payloads
    ]
    fusion_runs = pd.DataFrame(fusion_rows)
    fusion_runs.to_csv(config.results_dir / "fusion_gain_runs.csv", index=False)
    fusion_metrics = [c for c in fusion_runs.columns if c not in {"method", "seed"}]
    exp802._summarize(fusion_runs, ["method"], fusion_metrics).to_csv(
        config.results_dir / "fusion_gain_summary.csv", index=False
    )

    overlap_runs = pd.DataFrame(
        [row for payload in payloads for row in exp802._overlap_rows(payload)]
    )
    overlap_runs.to_csv(config.results_dir / "correctness_overlap_runs.csv", index=False)
    exp802._summarize(
        overlap_runs,
        ["method", "aggregation", "split"],
        [
            "both_correct",
            "l1_only_correct",
            "l2_only_correct",
            "both_wrong",
            "prediction_disagreement",
            "oracle_union_accuracy",
        ],
    ).to_csv(config.results_dir / "correctness_overlap_summary.csv", index=False)

    coef_runs = pd.DataFrame(
        [row for payload in payloads for row in exp802._coef_rows(payload)]
    )
    coef_runs.to_csv(config.results_dir / "coef_block_runs.csv", index=False)
    exp802._summarize(
        coef_runs,
        ["method", "feature", "block_index", "layer", "aggregation"],
        ["coef_fro_norm", "coef_rms", "coef_norm_fraction"],
    ).to_csv(config.results_dir / "coef_block_summary.csv", index=False)

    head_runs = pd.DataFrame([row for payload in payloads for row in _head_rows(payload)])
    head_runs.to_csv(config.results_dir / "trained_head_runs.csv", index=False)
    exp802._summarize(
        head_runs,
        ["method", "head"],
        ["coef_fro_norm", "coef_rms", "coef_norm_fraction"],
    ).to_csv(config.results_dir / "trained_head_summary.csv", index=False)

    phase_rows = [row for payload in payloads for row in _phase_rows(payload)]
    phase_bin_runs = pd.DataFrame(phase_rows)
    if not phase_bin_runs.empty:
        phase_bin_runs.to_csv(config.results_dir / "phase_bin_runs.csv", index=False)
        exp802._summarize(
            phase_bin_runs,
            ["method", "bin_index"],
            ["coef_fro_norm", "coef_rms", "norm_fraction"],
        ).to_csv(config.results_dir / "phase_bin_summary.csv", index=False)

    structure_rows = [
        row for payload in payloads if (row := _phase_structure_row(payload)) is not None
    ]
    phase_structure_runs = pd.DataFrame(structure_rows)
    if not phase_structure_runs.empty:
        phase_structure_runs.to_csv(
            config.results_dir / "phase_structure_runs.csv", index=False
        )
        exp802._summarize(
            phase_structure_runs,
            ["method"],
            [
                "mean_pairwise_cosine",
                "std_pairwise_cosine",
                "mean_adjacent_cosine",
                "between_bin_weight_rms",
            ],
        ).to_csv(config.results_dir / "phase_structure_summary.csv", index=False)

    paired = _paired_deltas(method_runs, probe_runs)
    paired.to_csv(config.results_dir / "paired_deltas.csv", index=False)
    exp802._summarize(
        paired,
        ["comparison"],
        ["native_test_ba_delta", "lif_test_ba_delta", "target_probe_test_ba_delta"],
    ).to_csv(config.results_dir / "paired_delta_summary.csv", index=False)

    history_frames: list[pd.DataFrame] = []
    for spec in run_specs():
        path = _path(config.results_dir, "histories", spec.key, ".csv")
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp8.0.3 history: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "seed", spec.seed)
        frame.insert(0, "method", spec.method)
        history_frames.append(frame)
    pd.concat(history_frames, ignore_index=True).to_csv(
        config.results_dir / "history_runs.csv", index=False
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "architecture_shifts": [list(layer) for layer in ARCHITECTURE_SHIFTS],
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "counts": {
            "methods": len(METHODS),
            "seeds": len(SEEDS),
            "parallel_runs": EXPECTED_RUNS,
        },
        "primary_comparison": "l1_fixed250_l2_whole - l1_capacity_no_phase_l2_whole",
        "target_posthoc_probe": TARGET_PROBE,
        "training": "A2-compatible end-to-end CE on valid-mean accumulated class evidence",
        "output_lif": {
            "alpha": 0.0,
            "beta": exp80.OUTPUT_BETA,
            "threshold": exp80.THRESHOLD,
            "cap": exp80.OUTPUT_CAP,
        },
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
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=exp73.MAX_EPOCHS)
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
        manifest = finalize(config)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    specs = run_specs()
    if args.array_task_id is not None:
        if not 0 <= args.array_task_id < len(specs):
            raise ValueError(
                f"array-task-id must be in [0, {len(specs)-1}], got {args.array_task_id}"
            )
        spec = specs[args.array_task_id]
    else:
        if args.method is None or args.seed is None:
            raise ValueError("Provide --array-task-id or both --method and --seed")
        spec = RunSpec(args.method, args.seed)

    payload = run_one(spec, config, force=args.force)
    print(json.dumps({
        "spec": payload["spec"],
        "best_epoch": payload["best_epoch"],
        "native_test_ba": payload["native_metrics"]["test"]["balanced_accuracy"],
        "lif_test_ba": payload["lif_transfer_metrics"]["test"]["balanced_accuracy"],
        "equivalence_error": payload["native_metrics"]["test"]["max_accumulator_equivalence_error"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
