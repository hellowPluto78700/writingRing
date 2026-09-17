from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts import experiment_8_0_3_phase_aware_hierarchical_readout as exp803


EXPERIMENT_ID = "experiment_8_0_4_phase_aware_hierarchical_readout_mean"
PROTOCOL_VERSION = "phase_aware_hierarchical_readout_mean_v1"
SEEDS = exp803.SEEDS
ARCHITECTURE = exp803.ARCHITECTURE
ARCHITECTURE_SHIFTS = exp803.ARCHITECTURE_SHIFTS
METHODS = exp803.METHODS
EXPECTED_RUNS = len(METHODS) * len(SEEDS)
TARGET_PROBE = exp803.TARGET_PROBE

RunSpec = exp803.RunSpec
Config = exp803.Config


class Exp804Net(exp803.Exp803Net):
    """Exp8.0.3 architecture with valid-length-mean CE training."""


def find_repo_root(start: Path | None = None) -> Path:
    return exp803.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(method, seed) for method in METHODS for seed in SEEDS]


def validate_spec(spec: RunSpec) -> None:
    if spec.method not in METHODS:
        raise ValueError(spec.method)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def new_model(spec: RunSpec, data: exp803.exp3.Data) -> Exp804Net:
    return Exp804Net(
        spec.method,
        len(data.labels),
        data.fs,
        int(data.Xtr.shape[1]),
        int(data.bin_steps),
    )


def _scores(
    model: Exp804Net, trajectory: dict[str, Any], lengths: torch.Tensor
) -> torch.Tensor:
    """A2-style normalized score: mean valid evidence per sequence."""
    evidence = exp803._native_evidence(model, trajectory)
    return exp803.exp80._valid_mean(evidence, lengths)


def _loss(
    model: Exp804Net,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(_scores(model, trajectory, lengths), y)


def _explicit_feature_scores(
    model: Exp804Net,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Explicit Fixed250/Whole score, normalized by valid length.

    Exp8.0.3 proved that the per-timestep accumulator equals the explicit
    count-feature form. Exp8.0.4 keeps the same feature semantics but divides
    the final class score by T_valid before CE, matching the A2 objective.
    """
    count_score = exp803._explicit_feature_scores(model, trajectory, lengths)
    denom = lengths.clamp_min(1).to(count_score.dtype).unsqueeze(1)
    return count_score / denom


def _evaluate_native(
    model: Exp804Net, loader: Iterable, device: torch.device
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
    out = exp803.exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    out["objective_loss"] = loss_sum / max(n_total, 1)
    out["max_accumulator_equivalence_error"] = max_equivalence_error
    return out


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    eval_path = exp803._path(config.results_dir, "evaluations", spec.key, ".json")
    checkpoint_path = exp803._path(config.results_dir, "checkpoints", spec.key, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    data = exp803.exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)

    # Exact same paired initialization stream as Exp8.0.3 / Exp8.0.2.
    exp803.exp3.seed_all(exp803.exp73._e2e_pair_seed(spec.seed, "model_init"))
    model = new_model(spec, data).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=exp803.exp72.LR, weight_decay=exp803.exp72.WEIGHT_DECAY
    )
    train_loader = exp803.exp73._raw_loaders(
        data, spec.seed, config.batch_size, True
    )["train"]
    eval_loaders = exp803.exp73._raw_loaders(
        data, spec.seed, config.batch_size, False
    )

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

        if exp803.exp73._checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if (
            epoch >= exp803.exp73.MIN_EPOCHS
            and best_epoch > 0
            and epoch - best_epoch >= exp803.exp73.PATIENCE
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
            "score_normalization": "valid_length_mean",
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
        split: exp803._evaluate_lif_transfer(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    probes, overlap, derived = exp803.exp802._fit_representation_probes(
        model, eval_loaders, data, spec, device
    )
    head_diagnostics = exp803._trained_head_diagnostics(model)
    phase_structure = exp803._phase_structure_diagnostics(model)
    branch_scores = exp803._branch_score_diagnostics(
        model, eval_loaders["test"], device
    )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "architecture": ARCHITECTURE,
        "architecture_shifts": ARCHITECTURE_SHIFTS,
        "bin_steps": int(data.bin_steps),
        "n_bins": int(model.n_bins),
        "score_normalization": "valid_length_mean",
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
    exp803._save_json(eval_path, payload)

    history_path = exp803._path(config.results_dir, "histories", spec.key, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return payload


def finalize(config: Config) -> dict[str, Any]:
    # Reuse the Exp8.0.3 aggregation schema verbatim so cross-experiment CSVs
    # are directly comparable. All run keys/method semantics are intentionally
    # retained; only the CE score normalization changed.
    exp803.finalize(config)
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
        "primary_comparison": (
            "l1_fixed250_l2_whole_count - "
            "l1_capacity_no_phase_l2_whole_count"
        ),
        "target_posthoc_probe": TARGET_PROBE,
        "training": (
            "end-to-end CE on valid-length-mean class evidence: "
            "mean_t[e_t] over valid timesteps"
        ),
        "single_change_from_exp803": (
            "score = valid_sum(evidence)/T_valid before CE; all architecture, "
            "method, seed, optimizer, checkpoint, probe, and LIF settings retained"
        ),
        "method_names": (
            "retained from Exp8.0.3 for exact paired comparison; the '_count' "
            "suffix describes the feature/readout construction, not CE normalization"
        ),
        "capacity_control_scale": "1/sqrt(n_bins)",
        "output_lif": {
            "alpha": 0.0,
            "beta": exp803.exp80.OUTPUT_BETA,
            "threshold": exp803.exp80.THRESHOLD,
            "cap": exp803.exp80.OUTPUT_CAP,
        },
    }
    exp803._save_json(config.results_dir / "manifest.json", manifest)
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
    parser.add_argument("--batch-size", type=int, default=exp803.exp72.BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=exp803.exp73.MAX_EPOCHS)
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
