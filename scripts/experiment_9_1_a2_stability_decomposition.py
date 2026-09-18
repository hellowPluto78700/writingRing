from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_8_0_local_backbone_tau_sweep as exp80
from scripts import experiment_9_0_within_user_generalization as exp90


EXPERIMENT_ID = "experiment_9_1_a2_stability_decomposition"
PROTOCOL_VERSION = "a2_stability_decomposition_v1"
ARCHITECTURE = "234x234"
ARCHITECTURE_SHIFTS = ((2, 3, 4), (2, 3, 4))
MODEL_SEEDS = (11, 23, 37, 53, 71)
REPRO_SEEDS = (11, 23, 37)
FOLDS = tuple(range(5))
EXPECTED_REPRO_RUNS = len(REPRO_SEEDS)
EXPECTED_FACTORIAL_RUNS = len(FOLDS) * len(MODEL_SEEDS)
COLLAPSE_TRAIN_BA = 0.70
DIAGNOSTIC_EPOCHS = (1, 5, 10, 20, 40, 60, 80, 100)
DIAGNOSTIC_SAMPLES = 64
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
BATCH_SIZE = exp72.BATCH_SIZE
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ReproSpec:
    seed: int

    @property
    def key(self) -> str:
        return f"old_split__seed{self.seed}"


@dataclass(frozen=True)
class FactorialSpec:
    fold: int
    seed: int

    @property
    def key(self) -> str:
        return f"within_fold{self.fold}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    threads: int = 1
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp90.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def repro_specs() -> list[ReproSpec]:
    return [ReproSpec(seed) for seed in REPRO_SEEDS]


def factorial_specs() -> list[FactorialSpec]:
    return [FactorialSpec(fold, seed) for fold in FOLDS for seed in MODEL_SEEDS]


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _split_assignment_path(config: Config) -> Path:
    return config.results_dir / "split" / "within_user_assignment.csv"


def _split_manifest_path(config: Config, fold: int) -> Path:
    return config.results_dir / "split" / f"within_fold{fold}.csv"


def prepare_all(config: Config) -> dict[str, Any]:
    loaded, raw_manifest, _ = exp90._load_manifest(config.repo_root)
    del loaded
    assignment = exp90._assign_within_user_folds(raw_manifest)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    _split_assignment_path(config).parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(_split_assignment_path(config), index=False)

    fold_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        split_manifest = exp90._apply_rotation(assignment, fold)
        split_manifest.to_csv(_split_manifest_path(config, fold), index=False)
        counts = split_manifest.split.value_counts()
        fold_rows.append(
            {
                "fold": fold,
                "train_samples": int(counts.get("train", 0)),
                "val_samples": int(counts.get("val", 0)),
                "test_samples": int(counts.get("test", 0)),
                "train_users": int(
                    split_manifest.loc[split_manifest.split == "train", "user"].nunique()
                ),
                "val_users": int(
                    split_manifest.loc[split_manifest.split == "val", "user"].nunique()
                ),
                "test_users": int(
                    split_manifest.loc[split_manifest.split == "test", "user"].nunique()
                ),
            }
        )
    pd.DataFrame(fold_rows).to_csv(config.results_dir / "fold_summary.csv", index=False)

    old = exp3.prepare_data(config.repo_root)
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Separate A2 optimization instability from fold difficulty and representation loss."
        ),
        "architecture": ARCHITECTURE,
        "architecture_shifts": [list(v) for v in ARCHITECTURE_SHIFTS],
        "reproduction": {
            "split": "original Exp8/Exp3 fixed cross-user split",
            "seeds": list(REPRO_SEEDS),
            "expected_runs": EXPECTED_REPRO_RUNS,
            "train_samples": int(len(old.ytr)),
            "val_samples": int(len(old.yva)),
            "test_samples": int(len(old.yte)),
            "train_users": list(old.split["train_users"]),
            "val_users": list(old.split["val_users"]),
            "test_users": list(old.split["test_users"]),
        },
        "factorial": {
            "cv_mode": "within_user",
            "folds": list(FOLDS),
            "model_seeds": list(MODEL_SEEDS),
            "expected_runs": EXPECTED_FACTORIAL_RUNS,
            "seed_fold_decoupled": True,
        },
        "model_init_seed_contract": "exp73._e2e_pair_seed(seed, 'model_init')",
        "loader_seed_contract": "exp73._raw_loaders(data, seed, ...)",
        "collapse_train_ba_threshold": COLLAPSE_TRAIN_BA,
        "diagnostic_epochs": list(DIAGNOSTIC_EPOCHS),
        "diagnostic_samples": DIAGNOSTIC_SAMPLES,
    }
    _save_json(config.results_dir / "audit.json", audit)
    return audit


def _prepare_within_fold(config: Config, fold: int) -> exp3.Data:
    if fold not in FOLDS:
        raise ValueError(fold)
    loaded, raw_manifest, labels = exp90._load_manifest(config.repo_root)
    assignment_path = _split_assignment_path(config)
    if assignment_path.exists():
        saved = pd.read_csv(assignment_path, usecols=["sample_id", "cv_fold"])
        assignment = raw_manifest.merge(
            saved, on="sample_id", how="inner", validate="one_to_one"
        )
        if len(assignment) != len(raw_manifest):
            raise RuntimeError("Saved Exp9.1 fold assignment does not match current dataset")
    else:
        assignment = exp90._assign_within_user_folds(raw_manifest)
    assignment["cv_fold"] = assignment.cv_fold.astype(int)
    split_manifest = exp90._apply_rotation(assignment, fold)
    data, _ = exp90._to_data(loaded, split_manifest, labels)
    return data


def _make_loaders(data: exp3.Data, seed: int, batch_size: int, shuffle_train: bool):
    # Deliberately reuse the old A2 loader seed contract, with no fold/experiment
    # component mixed into the seed. This makes seed effects comparable to Exp8.0.
    return exp73._raw_loaders(data, seed, batch_size, shuffle_train)


def _evaluate(
    model: exp80.Exp80Net,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    return exp80._evaluate_linear(model, loader, device)


def _gradient_norms(model: exp80.Exp80Net) -> dict[str, float]:
    def norm(parameter: torch.nn.Parameter) -> float:
        if parameter.grad is None:
            return 0.0
        return float(torch.linalg.vector_norm(parameter.grad.detach()).cpu())

    return {
        "grad_l1": norm(model.hidden_linears[0].weight),
        "grad_l2": norm(model.hidden_linears[1].weight),
        "grad_out": norm(model.output_linear.weight),
    }


def _activity_snapshot(
    model: exp80.Exp80Net,
    data: exp3.Data,
    device: torch.device,
    epoch: int,
) -> list[dict[str, Any]]:
    n = min(DIAGNOSTIC_SAMPLES, len(data.ytr))
    X = torch.tensor(data.Xtr[:n], dtype=torch.float32, device=device)
    lengths = torch.tensor(data.ltr[:n], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        trajectory = model.forward_trajectory(X)

    mask = exp80._valid_mask(lengths, trajectory["hidden_spikes"][0].shape[1])
    mask_f = mask.to(torch.float32).unsqueeze(-1)
    valid_seconds = float(lengths.sum().item()) / float(data.fs)
    rows: list[dict[str, Any]] = []
    for li, layer in enumerate(("l1", "l2")):
        spikes = trajectory["hidden_spikes"][li]
        overall_den = max(float(mask_f.sum().item()) * spikes.shape[-1], 1.0)
        rows.append(
            {
                "epoch": epoch,
                "layer": layer,
                "shift": "all",
                "mean_spikes_per_neuron_s": float(
                    (spikes * mask_f).sum().item()
                    / max(valid_seconds * spikes.shape[-1], 1e-12)
                ),
                "dead_neuron_fraction": float(
                    (((spikes * mask_f).sum(dim=(0, 1))) == 0).float().mean().item()
                ),
                "spike_occupancy": float((spikes * mask_f).sum().item() / overall_den),
            }
        )
        for group in exp80.shift_groups(ARCHITECTURE_SHIFTS[li]):
            chunk = spikes[:, :, group["start"] : group["stop"]]
            counts = (chunk * mask_f).sum(dim=(0, 1))
            rows.append(
                {
                    "epoch": epoch,
                    "layer": layer,
                    "shift": int(group["shift"]),
                    "mean_spikes_per_neuron_s": float(
                        counts.sum().item()
                        / max(valid_seconds * int(group["count"]), 1e-12)
                    ),
                    "dead_neuron_fraction": float((counts == 0).float().mean().item()),
                    "spike_occupancy": float(
                        (chunk * mask_f).sum().item()
                        / max(float(mask_f.sum().item()) * int(group["count"]), 1.0)
                    ),
                }
            )
    return rows


def _probe_metric(
    probes: dict[str, Any],
    layer: str,
    aggregation: str,
    split: str,
    metric: str = "balanced_accuracy",
) -> float:
    payload = probes[layer][aggregation]
    if "metrics" in payload:
        return float(payload["metrics"][split][metric])
    return float(payload[split][metric])


def _reference_payload(repo_root: Path, seed: int) -> dict[str, Any] | None:
    path = (
        repo_root
        / "notebooks/artifacts/experiment_8_0_local_backbone_tau_sweep"
        / "local_backbone_tau_sweep_v1/evaluations"
        / f"234x234__a2_wcce__task_only__seed{seed}.json"
    )
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _train_a2(
    data: exp3.Data,
    seed: int,
    config: Config,
    run_key: str,
) -> dict[str, Any]:
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)

    # Exact old A2 model-init seed contract. No fold, experiment ID, or CV mode
    # is mixed into this seed.
    exp3.seed_all(exp73._e2e_pair_seed(seed, "model_init"))
    model = exp80.Exp80Net(ARCHITECTURE_SHIFTS, len(data.labels), data.fs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp72.LR,
        weight_decay=exp72.WEIGHT_DECAY,
    )
    train_loader = _make_loaders(data, seed, config.batch_size, True)["train"]
    eval_loaders = _make_loaders(data, seed, config.batch_size, False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    best_train_ba = -1.0
    epoch_to_70: int | None = None
    history: list[dict[str, float]] = []
    activity_rows: list[dict[str, Any]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_total = 0
        grad_sums = {"grad_l1": 0.0, "grad_l2": 0.0, "grad_out": 0.0}
        grad_batches = 0
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            scores = exp80._valid_mean(trajectory["evidence"], lengths)
            loss = F.cross_entropy(scores, y)
            loss.backward()
            norms = _gradient_norms(model)
            for name, value in norms.items():
                grad_sums[name] += value
            grad_batches += 1
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate(model, eval_loaders["train"], device)
        val_metrics = _evaluate(model, eval_loaders["val"], device)
        train_ba = float(train_metrics["balanced_accuracy"])
        best_train_ba = max(best_train_ba, train_ba)
        if epoch_to_70 is None and train_ba >= COLLAPSE_TRAIN_BA:
            epoch_to_70 = epoch

        row = {
            "epoch": float(epoch),
            "train_ba": train_ba,
            "val_ba": float(val_metrics["balanced_accuracy"]),
            "train_loss": train_loss_sum / max(n_total, 1),
            "val_loss": float(val_metrics["objective_loss"]),
        }
        for name, value in grad_sums.items():
            row[name] = value / max(grad_batches, 1)
        history.append(row)

        if epoch in DIAGNOSTIC_EPOCHS or epoch == config.max_epochs:
            activity_rows.extend(_activity_snapshot(model, data, device, epoch))

        if exp73._checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if (
            epoch >= MIN_EPOCHS
            and best_epoch > 0
            and epoch - best_epoch >= PATIENCE
        ):
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No A2 checkpoint selected for {run_key}")

    final_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    final_optimizer_state = optimizer.state_dict()

    model.load_state_dict(best_state, strict=True)
    metrics = {
        split: _evaluate(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    layer_splits = exp80._extract_layer_splits(model, eval_loaders, device)
    probes = exp80._fit_layer_probes(layer_splits, seed, data.bin_steps)
    final_activity = _activity_snapshot(model, data, device, best_epoch)

    history_path = _path(config.results_dir, "histories", run_key, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    activity_path = _path(config.results_dir, "activity_diagnostics", run_key, ".csv")
    activity_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(activity_rows).to_csv(activity_path, index=False)

    checkpoint_path = _path(config.results_dir, "checkpoints", run_key, ".pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "run_key": run_key,
            "seed": seed,
            "architecture": ARCHITECTURE,
            "architecture_shifts": ARCHITECTURE_SHIFTS,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_loss": best_loss,
            "best_train_ba": best_train_ba,
            "epoch_to_70_train_ba": epoch_to_70,
            "best_model_state_dict": best_state,
            "final_model_state_dict": final_state,
            "final_optimizer_state_dict": final_optimizer_state,
        },
        checkpoint_path,
    )

    return {
        "metrics": metrics,
        "probes": probes,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_ba": best_ba,
        "best_val_loss": best_loss,
        "best_train_ba": best_train_ba,
        "epoch_to_70_train_ba": epoch_to_70,
        "collapsed": bool(best_train_ba < COLLAPSE_TRAIN_BA),
        "final_activity_at_best_checkpoint": final_activity,
    }


def run_reproduction(
    spec: ReproSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    if spec.seed not in REPRO_SEEDS:
        raise ValueError(spec)
    path = _path(config.results_dir, "reproduction_evaluations", spec.key, ".json")
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    data = exp3.prepare_data(config.repo_root)
    result = _train_a2(data, spec.seed, config, spec.key)
    reference = _reference_payload(config.repo_root, spec.seed)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "group": "old_split_reproduction",
        "spec": asdict(spec),
        "architecture": ARCHITECTURE,
        "architecture_shifts": ARCHITECTURE_SHIFTS,
        **result,
        "reference_exp8_linear_metrics": (
            reference.get("linear_metrics") if reference is not None else None
        ),
    }
    _save_json(path, payload)
    return payload


def run_factorial(
    spec: FactorialSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    if spec.fold not in FOLDS or spec.seed not in MODEL_SEEDS:
        raise ValueError(spec)
    path = _path(config.results_dir, "factorial_evaluations", spec.key, ".json")
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    data = _prepare_within_fold(config, spec.fold)
    result = _train_a2(data, spec.seed, config, spec.key)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "group": "within_user_fold_seed_factorial",
        "spec": asdict(spec),
        "architecture": ARCHITECTURE,
        "architecture_shifts": ARCHITECTURE_SHIFTS,
        **result,
    }
    _save_json(path, payload)
    return payload


def _flatten_run(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    probes = payload["probes"]
    spec = payload["spec"]
    row: dict[str, Any] = {
        **spec,
        "best_epoch": int(payload["best_epoch"]),
        "stopped_epoch": int(payload["stopped_epoch"]),
        "best_train_ba": float(payload["best_train_ba"]),
        "epoch_to_70_train_ba": payload["epoch_to_70_train_ba"],
        "collapsed": bool(payload["collapsed"]),
    }
    for split in SPLITS:
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "objective_loss"):
            row[f"{split}_{metric}"] = float(metrics[split][metric])
    for layer in ("l1", "l2"):
        for aggregation in ("whole_count", "fixed250"):
            for split in SPLITS:
                row[
                    f"{layer}_{aggregation}_{split}_ba"
                ] = _probe_metric(probes, layer, aggregation, split)
    row["native_to_l2_whole_test_gap"] = (
        row["l2_whole_count_test_ba"] - row["test_balanced_accuracy"]
    )
    row["native_to_l2_fixed250_test_gap"] = (
        row["l2_fixed250_test_ba"] - row["test_balanced_accuracy"]
    )
    row["l2_minus_l1_whole_test"] = (
        row["l2_whole_count_test_ba"] - row["l1_whole_count_test_ba"]
    )
    row["l2_minus_l1_fixed250_test"] = (
        row["l2_fixed250_test_ba"] - row["l1_fixed250_test_ba"]
    )
    return row


def _aggregate_diagnostics(config: Config, specs: list[FactorialSpec]) -> None:
    history_frames = []
    activity_frames = []
    for spec in specs:
        h = pd.read_csv(_path(config.results_dir, "histories", spec.key, ".csv"))
        h.insert(0, "seed", spec.seed)
        h.insert(0, "fold", spec.fold)
        history_frames.append(h)
        a = pd.read_csv(
            _path(config.results_dir, "activity_diagnostics", spec.key, ".csv")
        )
        a.insert(0, "seed", spec.seed)
        a.insert(0, "fold", spec.fold)
        activity_frames.append(a)
    pd.concat(history_frames, ignore_index=True).to_csv(
        config.results_dir / "training_diagnostics.csv", index=False
    )
    pd.concat(activity_frames, ignore_index=True).to_csv(
        config.results_dir / "activity_diagnostics.csv", index=False
    )


def finalize(config: Config) -> dict[str, Any]:
    repro_payloads = []
    for spec in repro_specs():
        path = _path(config.results_dir, "reproduction_evaluations", spec.key, ".json")
        if not path.exists():
            raise FileNotFoundError(path)
        repro_payloads.append(json.loads(path.read_text(encoding="utf-8")))
    factorial_payloads = []
    for spec in factorial_specs():
        path = _path(config.results_dir, "factorial_evaluations", spec.key, ".json")
        if not path.exists():
            raise FileNotFoundError(path)
        factorial_payloads.append(json.loads(path.read_text(encoding="utf-8")))

    repro_rows: list[dict[str, Any]] = []
    for payload in repro_payloads:
        row = _flatten_run(payload)
        reference = payload.get("reference_exp8_linear_metrics")
        if reference:
            for split in SPLITS:
                row[f"reference_{split}_ba"] = float(
                    reference[split]["balanced_accuracy"]
                )
                row[f"delta_vs_reference_{split}_ba"] = (
                    row[f"{split}_balanced_accuracy"]
                    - row[f"reference_{split}_ba"]
                )
        repro_rows.append(row)
    reproduction = pd.DataFrame(repro_rows)
    reproduction.to_csv(config.results_dir / "reproduction_runs.csv", index=False)

    factorial = pd.DataFrame([_flatten_run(p) for p in factorial_payloads])
    factorial.to_csv(config.results_dir / "factorial_runs.csv", index=False)

    metric_cols = [
        "test_balanced_accuracy",
        "val_balanced_accuracy",
        "best_train_ba",
        "l1_whole_count_test_ba",
        "l1_fixed250_test_ba",
        "l2_whole_count_test_ba",
        "l2_fixed250_test_ba",
        "native_to_l2_whole_test_gap",
        "native_to_l2_fixed250_test_gap",
    ]
    fold_summary = (
        factorial.groupby("fold", sort=True)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    fold_summary.columns = [
        "_".join(str(v) for v in col if str(v))
        if isinstance(col, tuple)
        else str(col)
        for col in fold_summary.columns
    ]
    collapse_by_fold = (
        factorial.groupby("fold", sort=True).collapsed.mean().rename("collapse_rate")
    )
    fold_summary = fold_summary.merge(
        collapse_by_fold.reset_index(), on="fold", how="left"
    )
    fold_summary.to_csv(config.results_dir / "fold_effect_summary.csv", index=False)

    seed_summary = (
        factorial.groupby("seed", sort=True)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    seed_summary.columns = [
        "_".join(str(v) for v in col if str(v))
        if isinstance(col, tuple)
        else str(col)
        for col in seed_summary.columns
    ]
    collapse_by_seed = (
        factorial.groupby("seed", sort=True).collapsed.mean().rename("collapse_rate")
    )
    seed_summary = seed_summary.merge(
        collapse_by_seed.reset_index(), on="seed", how="left"
    )
    seed_summary.to_csv(config.results_dir / "seed_effect_summary.csv", index=False)

    factorial.pivot(index="fold", columns="seed", values="test_balanced_accuracy").to_csv(
        config.results_dir / "test_ba_matrix.csv"
    )
    factorial.pivot(index="fold", columns="seed", values="best_train_ba").to_csv(
        config.results_dir / "best_train_ba_matrix.csv"
    )
    factorial.assign(collapse_int=factorial.collapsed.astype(int)).pivot(
        index="fold", columns="seed", values="collapse_int"
    ).to_csv(config.results_dir / "collapse_matrix.csv")

    successful = factorial[~factorial.collapsed].copy()
    successful.to_csv(config.results_dir / "successful_runs.csv", index=False)
    successful_summary = {
        "n_successful": int(len(successful)),
        "n_total": int(len(factorial)),
        "collapse_rate": float(factorial.collapsed.mean()),
        "successful_test_ba_mean": (
            float(successful.test_balanced_accuracy.mean())
            if not successful.empty
            else None
        ),
        "successful_test_ba_std": (
            float(successful.test_balanced_accuracy.std())
            if len(successful) > 1
            else None
        ),
    }
    _save_json(config.results_dir / "collapse_summary.json", successful_summary)

    probe_cols = [
        "fold",
        "seed",
        "collapsed",
        "test_balanced_accuracy",
        "l1_whole_count_test_ba",
        "l1_fixed250_test_ba",
        "l2_whole_count_test_ba",
        "l2_fixed250_test_ba",
        "native_to_l2_whole_test_gap",
        "native_to_l2_fixed250_test_gap",
        "l2_minus_l1_whole_test",
        "l2_minus_l1_fixed250_test",
    ]
    factorial[probe_cols].to_csv(
        config.results_dir / "native_vs_probe.csv", index=False
    )

    _aggregate_diagnostics(config, factorial_specs())

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "architecture_shifts": [list(v) for v in ARCHITECTURE_SHIFTS],
        "reproduction_runs": EXPECTED_REPRO_RUNS,
        "factorial_runs": EXPECTED_FACTORIAL_RUNS,
        "folds": list(FOLDS),
        "model_seeds": list(MODEL_SEEDS),
        "collapse_train_ba_threshold": COLLAPSE_TRAIN_BA,
        "model_init_seed_contract": "exp73._e2e_pair_seed(seed, 'model_init')",
        "loader_seed_contract": "exp73._raw_loaders(data, seed, ...)",
        "diagnostic_epochs": list(DIAGNOSTIC_EPOCHS),
        "primary_outputs": [
            "reproduction_runs.csv",
            "factorial_runs.csv",
            "test_ba_matrix.csv",
            "best_train_ba_matrix.csv",
            "collapse_matrix.csv",
            "fold_effect_summary.csv",
            "seed_effect_summary.csv",
            "native_vs_probe.csv",
            "training_diagnostics.csv",
            "activity_diagnostics.csv",
            "collapse_summary.json",
        ],
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _resolve_config(args: argparse.Namespace) -> Config:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    output = (
        Path(args.results_dir).resolve()
        if args.results_dir
        else results_dir(repo_root)
    )
    return Config(
        repo_root=repo_root,
        results_dir=output,
        device=args.device,
        threads=args.threads,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp9.1 A2 optimization stability and representation failure decomposition"
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("list-runs")
    repro = sub.add_parser("run-reproduction")
    repro.add_argument("--array-task-id", type=int, required=True)
    repro.add_argument("--force", action="store_true")
    factorial = sub.add_parser("run-factorial")
    factorial.add_argument("--array-task-id", type=int, required=True)
    factorial.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _resolve_config(args)
    if args.command == "prepare":
        print(json.dumps(prepare_all(config), indent=2))
    elif args.command == "list-runs":
        print("Reproduction:")
        for idx, spec in enumerate(repro_specs()):
            print(idx, spec.key)
        print("Factorial:")
        for idx, spec in enumerate(factorial_specs()):
            print(idx, spec.key)
    elif args.command == "run-reproduction":
        specs = repro_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        payload = run_reproduction(
            specs[args.array_task_id], config, force=args.force
        )
        print(
            json.dumps(
                {
                    "key": specs[args.array_task_id].key,
                    "test_ba": payload["metrics"]["test"]["balanced_accuracy"],
                    "best_train_ba": payload["best_train_ba"],
                    "collapsed": payload["collapsed"],
                },
                indent=2,
            )
        )
    elif args.command == "run-factorial":
        specs = factorial_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        payload = run_factorial(
            specs[args.array_task_id], config, force=args.force
        )
        print(
            json.dumps(
                {
                    "key": specs[args.array_task_id].key,
                    "test_ba": payload["metrics"]["test"]["balanced_accuracy"],
                    "best_train_ba": payload["best_train_ba"],
                    "collapsed": payload["collapsed"],
                },
                indent=2,
            )
        )
    elif args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))


if __name__ == "__main__":
    main()
