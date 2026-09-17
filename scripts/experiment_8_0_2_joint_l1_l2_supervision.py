from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
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
from scripts import experiment_8_0_1_l1_l2_fusion_probe as exp801


EXPERIMENT_ID = "experiment_8_0_2_joint_l1_l2_supervision"
PROTOCOL_VERSION = "joint_l1_l2_supervision_v1"
SEEDS = exp73.SEEDS
ARCHITECTURE = "234x234"
ARCHITECTURE_SHIFTS = exp80.ARCHITECTURES[ARCHITECTURE]
AUX_LAMBDA = 0.1
METHODS = ("l2_only", "l1_l2_joint", "l2_main_l1_aux")
EXPECTED_RUNS = len(METHODS) * len(SEEDS)


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


class Exp802Net(exp80.Exp80Net):
    """Exp8.0 234x234 backbone with an optional direct L1 classification head."""

    def __init__(self, method: str, n_classes: int, fs: float) -> None:
        if method not in METHODS:
            raise ValueError(method)
        super().__init__(ARCHITECTURE_SHIFTS, n_classes, fs)
        self.method = method
        self.l1_output_linear: nn.Linear | None
        if method in {"l1_l2_joint", "l2_main_l1_aux"}:
            # Created after the inherited backbone/W2 head so paired seeds preserve
            # the baseline initialization for all shared parameters.
            self.l1_output_linear = nn.Linear(exp80.HIDDEN_WIDTH, n_classes, bias=False)
        else:
            self.l1_output_linear = None

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        trajectory = super().forward_trajectory(x)
        if self.l1_output_linear is None:
            trajectory["l1_evidence"] = None
        else:
            trajectory["l1_evidence"] = self.l1_output_linear(
                trajectory["hidden_spikes"][0]
            )
        return trajectory


def new_model(spec: RunSpec, data: exp3.Data) -> Exp802Net:
    return Exp802Net(spec.method, len(data.labels), data.fs)


def _native_evidence(model: Exp802Net, trajectory: dict[str, Any]) -> torch.Tensor:
    l2 = trajectory["evidence"]
    if model.method == "l1_l2_joint":
        l1 = trajectory["l1_evidence"]
        if l1 is None:
            raise RuntimeError("Joint method requires L1 evidence")
        return l1 + l2
    return l2


def _scores(model: Exp802Net, trajectory: dict[str, Any], lengths: torch.Tensor) -> torch.Tensor:
    return exp80._valid_mean(_native_evidence(model, trajectory), lengths)


def _loss_terms(
    model: Exp802Net,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    main_scores = _scores(model, trajectory, lengths)
    main_ce = F.cross_entropy(main_scores, y)
    aux_ce: torch.Tensor | None = None
    total = main_ce
    if model.method == "l2_main_l1_aux":
        l1 = trajectory["l1_evidence"]
        if l1 is None:
            raise RuntimeError("Auxiliary method requires L1 evidence")
        l1_scores = exp80._valid_mean(l1, lengths)
        aux_ce = F.cross_entropy(l1_scores, y)
        total = main_ce + AUX_LAMBDA * aux_ce
    return total, main_ce, aux_ce


def _evaluate_native(
    model: Exp802Net, loader: Iterable, device: torch.device
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    objective_sum = 0.0
    main_sum = 0.0
    aux_sum = 0.0
    n_total = 0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            trajectory = model.forward_trajectory(X)
            scores = _scores(model, trajectory, lengths)
            total, main_ce, aux_ce = _loss_terms(model, trajectory, lengths, y)
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(1).cpu().numpy())
            objective_sum += float(total) * len(y)
            main_sum += float(main_ce) * len(y)
            if aux_ce is not None:
                aux_sum += float(aux_ce) * len(y)
            n_total += len(y)
    out = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    out["objective_loss"] = objective_sum / max(n_total, 1)
    out["main_ce"] = main_sum / max(n_total, 1)
    if model.method == "l2_main_l1_aux":
        out["aux_ce"] = aux_sum / max(n_total, 1)
    return out


def _evaluate_lif_transfer(
    model: Exp802Net, loader: Iterable, device: torch.device
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            lengths_device = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            evidence = _native_evidence(model, trajectory)
            out_spikes = exp80._output_lif_spikes(evidence)
            mask = exp80._valid_mask(lengths_device, out_spikes.shape[1]).to(
                out_spikes.dtype
            ).unsqueeze(-1)
            counts = (out_spikes * mask).sum(1)
            ys.append(y.numpy())
            preds.append(counts.argmax(1).cpu().numpy())
    return exp72._metrics(np.concatenate(ys), np.concatenate(preds))


def _fit_representation_probes(
    model: Exp802Net,
    loaders: dict[str, Iterable],
    data: exp3.Data,
    spec: RunSpec,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    extracted = exp80._extract_layer_splits(model, loaders, device)
    probes: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    split_labels = {
        split: extracted["l1"][split][1]
        for split in ("train", "val", "test")
    }

    for feature in exp801.FEATURE_SPECS:
        built = {
            split: exp801._build_features(extracted, feature, split, data.bin_steps)
            for split in ("train", "val", "test")
        }
        probe, preds = exp801._fit_probe_detailed(
            built["train"][0],
            built["train"][1],
            built["val"][0],
            built["val"][1],
            built["test"][0],
            built["test"][1],
            seed=exp3.dseed(spec.seed, EXPERIMENT_ID, spec.method, feature.name),
            blocks=built["train"][2],
        )
        probes[feature.name] = probe
        predictions[feature.name] = preds

    overlap = {
        aggregation: {
            split: exp801._correctness_overlap(
                split_labels[split],
                predictions[f"l1_{aggregation}"][split],
                predictions[f"l2_{aggregation}"][split],
            )
            for split in ("train", "val", "test")
        }
        for aggregation in ("whole", "fixed250")
    }

    def test_ba(name: str) -> float:
        return float(probes[name]["metrics"]["test"]["balanced_accuracy"])

    derived = {
        "whole_fusion_gain_over_best_single": test_ba("l1_l2_whole")
        - max(test_ba("l1_whole"), test_ba("l2_whole")),
        "fixed250_fusion_gain_over_best_single": test_ba("l1_l2_fixed250")
        - max(test_ba("l1_fixed250"), test_ba("l2_fixed250")),
        "l1whole_l2fixed250_gain_over_l2fixed250": test_ba("l1whole_l2fixed250")
        - test_ba("l2_fixed250"),
        "reverse_mixed_gain_over_best_component": test_ba("l1fixed250_l2whole")
        - max(test_ba("l1_fixed250"), test_ba("l2_whole")),
    }
    return probes, overlap, derived


def _head_diagnostics(model: Exp802Net) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    heads: list[tuple[str, nn.Linear | None]] = [
        ("l2", model.output_linear),
        ("l1", model.l1_output_linear),
    ]
    norms: list[float] = []
    raw_rows: list[dict[str, Any]] = []
    for name, head in heads:
        if head is None:
            continue
        weight = head.weight.detach().cpu().numpy().astype(np.float64)
        fro = float(np.linalg.norm(weight))
        rms = float(np.sqrt(np.mean(np.square(weight))))
        norms.append(fro)
        raw_rows.append({"head": name, "coef_fro_norm": fro, "coef_rms": rms})
    total = float(sum(norms))
    for row, norm in zip(raw_rows, norms, strict=True):
        rows.append({
            **row,
            "coef_norm_fraction": norm / total if total > 0 else 0.0,
        })
    return rows


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    eval_path = _path(config.results_dir, "evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "checkpoints", spec.key, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)

    # Preserve the Exp7.3 A2 paired initialization and data-order streams.
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
        loss_sum = 0.0
        main_sum = 0.0
        aux_sum = 0.0
        n_total = 0
        for X, y, lengths in train_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            loss, main_ce, aux_ce = _loss_terms(model, trajectory, lengths, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            main_sum += float(main_ce.detach()) * len(y)
            if aux_ce is not None:
                aux_sum += float(aux_ce.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate_native(model, eval_loaders["train"], device)
        val_metrics = _evaluate_native(model, eval_loaders["val"], device)
        row = {
            "epoch": float(epoch),
            "train_ba": float(train_metrics["balanced_accuracy"]),
            "val_ba": float(val_metrics["balanced_accuracy"]),
            "train_loss": loss_sum / max(n_total, 1),
            "val_loss": float(val_metrics["objective_loss"]),
            "train_main_ce": main_sum / max(n_total, 1),
            "val_main_ce": float(val_metrics["main_ce"]),
        }
        if spec.method == "l2_main_l1_aux":
            row["train_aux_ce"] = aux_sum / max(n_total, 1)
            row["val_aux_ce"] = float(val_metrics["aux_ce"])
        history.append(row)

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
            "aux_lambda": AUX_LAMBDA,
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
    probes, overlap, derived = _fit_representation_probes(
        model, eval_loaders, data, spec, device
    )
    head_diagnostics = _head_diagnostics(model)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "architecture": ARCHITECTURE,
        "architecture_shifts": ARCHITECTURE_SHIFTS,
        "aux_lambda": AUX_LAMBDA,
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
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
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
        "native_test_ba": float(payload["native_metrics"]["test"]["balanced_accuracy"]),
        "native_val_ba": float(payload["native_metrics"]["val"]["balanced_accuracy"]),
        "lif_test_ba": float(payload["lif_transfer_metrics"]["test"]["balanced_accuracy"]),
        "lif_penalty": float(payload["lif_penalty_test_ba"]),
        "best_epoch": int(payload["best_epoch"]),
        "parameter_count": int(payload["parameter_count"]),
    }


def _probe_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method = payload["spec"]["method"]
    seed = int(payload["spec"]["seed"])
    for feature, probe in payload["probes"].items():
        rows.append({
            "method": method,
            "seed": seed,
            "feature": feature,
            "feature_dim": int(probe["feature_dim"]),
            "probe_C": float(probe["probe_C"]),
            "train_ba": float(probe["metrics"]["train"]["balanced_accuracy"]),
            "val_ba": float(probe["metrics"]["val"]["balanced_accuracy"]),
            "test_ba": float(probe["metrics"]["test"]["balanced_accuracy"]),
            "train_test_gap": float(probe["generalization_gap_train_test_ba"]),
        })
    return rows


def _coef_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method = payload["spec"]["method"]
    seed = int(payload["spec"]["seed"])
    for feature, probe in payload["probes"].items():
        for block_index, block in enumerate(probe["blocks"]):
            rows.append({
                "method": method,
                "seed": seed,
                "feature": feature,
                "block_index": block_index,
                "layer": block["layer"],
                "aggregation": block["aggregation"],
                "feature_dim": int(block["feature_dim"]),
                "coef_fro_norm": float(block["coef_fro_norm"]),
                "coef_rms": float(block["coef_rms"]),
                "coef_norm_fraction": float(block["coef_norm_fraction"]),
            })
    return rows


def _overlap_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method = payload["spec"]["method"]
    seed = int(payload["spec"]["seed"])
    for aggregation, split_payload in payload["correctness_overlap"].items():
        for split, metrics in split_payload.items():
            rows.append({
                "method": method,
                "seed": seed,
                "aggregation": aggregation,
                "split": split,
                **{key: float(value) for key, value in metrics.items()},
            })
    return rows


def _head_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": payload["spec"]["method"],
            "seed": int(payload["spec"]["seed"]),
            **row,
        }
        for row in payload["trained_head_diagnostics"]
    ]


def _summarize(
    frame: pd.DataFrame, group_cols: list[str], numeric_cols: list[str]
) -> pd.DataFrame:
    summary = frame.groupby(group_cols, sort=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(value) for value in col if str(value))
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for spec in run_specs():
        path = _path(config.results_dir, "evaluations", spec.key, ".json")
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp8.0.2 evaluation: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if len(payloads) != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} runs, got {len(payloads)}")

    config.results_dir.mkdir(parents=True, exist_ok=True)

    method_runs = pd.DataFrame([_method_row(payload) for payload in payloads])
    method_runs.to_csv(config.results_dir / "method_runs.csv", index=False)
    _summarize(
        method_runs,
        ["method"],
        ["native_test_ba", "native_val_ba", "lif_test_ba", "lif_penalty", "best_epoch", "parameter_count"],
    ).to_csv(config.results_dir / "method_summary.csv", index=False)

    probe_runs = pd.DataFrame([row for payload in payloads for row in _probe_rows(payload)])
    probe_runs.to_csv(config.results_dir / "probe_runs.csv", index=False)
    _summarize(
        probe_runs,
        ["method", "feature"],
        ["train_ba", "val_ba", "test_ba", "train_test_gap", "feature_dim", "probe_C"],
    ).to_csv(config.results_dir / "probe_summary.csv", index=False)

    fusion_rows = []
    for payload in payloads:
        fusion_rows.append({
            "method": payload["spec"]["method"],
            "seed": int(payload["spec"]["seed"]),
            **{key: float(value) for key, value in payload["derived"].items()},
        })
    fusion_runs = pd.DataFrame(fusion_rows)
    fusion_runs.to_csv(config.results_dir / "fusion_gain_runs.csv", index=False)
    fusion_metrics = [col for col in fusion_runs.columns if col not in {"method", "seed"}]
    _summarize(fusion_runs, ["method"], fusion_metrics).to_csv(
        config.results_dir / "fusion_gain_summary.csv", index=False
    )

    overlap_runs = pd.DataFrame([row for payload in payloads for row in _overlap_rows(payload)])
    overlap_runs.to_csv(config.results_dir / "correctness_overlap_runs.csv", index=False)
    overlap_metrics = [
        "both_correct",
        "l1_only_correct",
        "l2_only_correct",
        "both_wrong",
        "prediction_disagreement",
        "oracle_union_accuracy",
    ]
    _summarize(overlap_runs, ["method", "aggregation", "split"], overlap_metrics).to_csv(
        config.results_dir / "correctness_overlap_summary.csv", index=False
    )

    coef_runs = pd.DataFrame([row for payload in payloads for row in _coef_rows(payload)])
    coef_runs.to_csv(config.results_dir / "coef_block_runs.csv", index=False)
    _summarize(
        coef_runs,
        ["method", "feature", "block_index", "layer", "aggregation"],
        ["coef_fro_norm", "coef_rms", "coef_norm_fraction"],
    ).to_csv(config.results_dir / "coef_block_summary.csv", index=False)

    head_runs = pd.DataFrame([row for payload in payloads for row in _head_rows(payload)])
    head_runs.to_csv(config.results_dir / "trained_head_runs.csv", index=False)
    _summarize(
        head_runs,
        ["method", "head"],
        ["coef_fro_norm", "coef_rms", "coef_norm_fraction"],
    ).to_csv(config.results_dir / "trained_head_summary.csv", index=False)

    history_frames: list[pd.DataFrame] = []
    for spec in run_specs():
        path = _path(config.results_dir, "histories", spec.key, ".csv")
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp8.0.2 history: {path}")
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
        "aux_lambda": AUX_LAMBDA,
        "counts": {"methods": len(METHODS), "seeds": len(SEEDS), "parallel_runs": EXPECTED_RUNS},
        "training": "Exp7.3 A2-compatible end-to-end training with supervision topology as the independent variable",
        "output_lif": {
            "alpha": exp80.OUTPUT_ALPHA,
            "beta": exp80.OUTPUT_BETA,
            "threshold": exp80.THRESHOLD,
            "cap": exp80.OUTPUT_CAP,
        },
        "probe_features": list(exp801.FEATURE_NAMES),
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp8.0.2 joint L1/L2 supervision")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=exp73.MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-runs")
    run = sub.add_parser("run")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
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
        print(json.dumps({
            "key": specs[args.array_task_id].key,
            "native_test_ba": payload["native_metrics"]["test"]["balanced_accuracy"],
        }, indent=2))
        return
    if args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
