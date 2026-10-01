#!/usr/bin/env python3
"""Exp17: close the persistent-pathway mechanism story.

Exp17 contains three deliberately narrow blocks:
A. epoch-wise emergence/reliance trajectory for the locked WCCE backbone;
B. dimension-matched persistent-subset class/user decoding on Exp16.2 traces;
C. pruning-specificity controls (high/low/random/readout-weight) on Exp16.2.

Only block A retrains an SNN. Blocks B/C are artifact-only and never modify a
backbone checkpoint. Test data are diagnostic only and never affect training,
checkpoint selection, subset definition, C selection, or pruning ranking.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from core_benchmark_v1.probes import fit_probe
from core_benchmark_v1.protocol import Protocol, SPLITS, paired_seed
from core_benchmark_v1.storage import load_torch, save_json, save_torch
from core_benchmark_v1.training import cpu_state, metrics
from scripts import analyze_experiment_16_2_persistent_neurons as pdiag
from scripts import experiment_16_2_matched_budget_selective_write as exp16_2
from scripts import experiment_16_3_run_reward as exp16_3
from scripts import experiment_16_prefix_supervised_selective_memory as exp16


EXPERIMENT_ID = "experiment_17_persistent_pathway_story"
PROTOCOL_VERSION = "persistent_pathway_story_v1"
FORMAL_SEEDS = (11, 23, 37)
SOURCE_CASES = tuple(pdiag.DEFAULT_CASES)
CHECKPOINT_EVERY = 5
PERSISTENT_QUARTILE = 0.25
LONGITUDINAL_PRUNE_FRACTION = 0.30
SUBSET_FRACTION = 0.30
SUBSET_RANDOM_REPLICATES = 5
PRUNE_FRACTIONS = (0.10, 0.20, 0.30)
PRUNE_RANDOM_REPLICATES = 20
USER_CV_FOLDS = 5
OCCUPANCY_THRESHOLDS = (0.50, 0.80)
EXP16_2_RESULTS_REL = Path(
    "notebooks/artifacts/experiment_16_2_matched_budget_selective_write/"
    "matched_budget_selective_write_v1"
)


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    core_results_dir: Path
    exp16_2_results_dir: Path


@dataclass(frozen=True)
class ArtifactSpec:
    case: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.case}__seed{self.seed}"


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


def trajectory_specs() -> list[int]:
    return list(FORMAL_SEEDS)


def artifact_specs() -> list[ArtifactSpec]:
    return [ArtifactSpec(case, seed) for seed in FORMAL_SEEDS for case in SOURCE_CASES]


def _core(config: Config) -> tuple[Protocol, dict[str, Any], dict[str, np.ndarray]]:
    return exp16._load_core(config)  # Config is duck-typed by _load_core.


def _trajectory_spec(seed: int) -> exp16_3.ExpSpec:
    return exp16_3.ExpSpec("LIN", seed, "linear", None)


def _make_trajectory_model(seed: int, p: Protocol) -> torch.nn.Module:
    return exp16_3._make_model(_trajectory_spec(seed), p)


def _snapshot_path(directory: Path, epoch: int) -> Path:
    return directory / "snapshots" / f"epoch_{epoch:03d}.pt"


def _save_snapshot(directory: Path, epoch: int, state: dict[str, torch.Tensor]) -> None:
    path = _snapshot_path(directory, epoch)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    save_torch(path, {"epoch": epoch, "model_state_dict": state})


def _better(candidate: dict[str, float], best: dict[str, float]) -> bool:
    return (
        candidate["ba"] > best["ba"] + 1e-12
        or (
            abs(candidate["ba"] - best["ba"]) <= 1e-12
            and candidate["ce"] < best["ce"] - 1e-12
        )
    )


def _train_trajectory(config: Config, seed: int) -> dict[str, Any]:
    """Train the locked LIN/WCCE trajectory. Test data are never read here."""
    p, _, arrays = _core(config)
    spec = _trajectory_spec(seed)
    directory = config.results_dir / "trajectory" / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    manifest_path = directory / "snapshot_manifest.json"
    history_path = directory / "train_history.json"
    if checkpoint_path.exists() and manifest_path.exists() and history_path.exists():
        return {"status": "exists", "seed": seed}

    model = _make_trajectory_model(seed, p)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
    )
    roles: dict[int, set[str]] = {0: {"initial"}}
    _save_snapshot(directory, 0, cpu_state(model))

    val0 = exp16_3._split_eval(model, arrays, p, spec, "val")
    best = dict(val0)
    best_state = cpu_state(model)
    best_epoch = 0
    history: list[dict[str, Any]] = [{
        "epoch": 0,
        "train_loss": None,
        "val_ba": val0["ba"],
        "val_ce": val0["ce"],
        "sampler_hash": None,
    }]

    stopped_epoch = 0
    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for x, y, lengths in exp16_3._train_batches(arrays, p, seed, epoch):
            optimizer.zero_grad(set_to_none=True)
            out = model(x, lengths)
            logits = exp16_3.reward_logits(model, out, lengths, spec)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Exp17 seed{seed}: nonfinite loss")
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"Exp17 seed{seed}: nonfinite gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            count += len(y)

        val = exp16_3._split_eval(model, arrays, p, spec, "val")
        row = {
            "epoch": epoch,
            "train_loss": total_loss / count,
            "val_ba": val["ba"],
            "val_ce": val["ce"],
            "sampler_hash": exp16_3._permutation_hash(
                exp16_3._epoch_permutation(len(arrays["train_y"]), seed, epoch)
            ),
        }
        history.append(row)
        if _better(val, best):
            best = dict(val)
            best_state = cpu_state(model)
            best_epoch = epoch

        if epoch % CHECKPOINT_EVERY == 0:
            _save_snapshot(directory, epoch, cpu_state(model))
            roles.setdefault(epoch, set()).add("periodic")
        if epoch == 1 or epoch % 10 == 0:
            save_json(directory / "progress.json", row)
            print(
                f"Exp17 seed{seed} epoch={epoch} loss={total_loss/count:.5f} "
                f"val_ba={val['ba']:.5f}",
                flush=True,
            )

        stopped_epoch = epoch
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    _save_snapshot(directory, best_epoch, best_state)
    roles.setdefault(best_epoch, set()).add("selected_best")
    stopped_state = cpu_state(model)
    _save_snapshot(directory, stopped_epoch, stopped_state)
    roles.setdefault(stopped_epoch, set()).add("stopped")

    save_json(history_path, {"rows": history})
    manifest = {
        "seed": seed,
        "checkpoint_every": CHECKPOINT_EVERY,
        "snapshots": [
            {"epoch": epoch, "roles": sorted(roles.get(epoch, {"periodic"}))}
            for epoch in sorted({0, best_epoch, stopped_epoch, *roles.keys()})
            if _snapshot_path(directory, epoch).exists()
        ],
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
    }
    save_json(manifest_path, manifest)
    save_torch(
        checkpoint_path,
        {
            "experiment": EXPERIMENT_ID,
            "protocol": PROTOCOL_VERSION,
            "seed": seed,
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val": best,
            "selection_rule": "validation BA, then validation CE, then earliest epoch",
            "shared_init_hash": exp16_3._shared_init_hash(seed, p),
        },
    )
    return {
        "status": "PASS",
        "seed": seed,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
    }


def _snapshot_features(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {
        layer: {} for layer in ("L1", "L2")
    }
    native: dict[str, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            layer_counts = {"L1": [], "L2": []}
            layer_longest = {"L1": [], "L2": []}
            ys: list[np.ndarray] = []
            preds: list[np.ndarray] = []
            for x, y, lengths in exp16.loader(arrays, split, p, seed):
                out = model(x, lengths)
                scores = exp16_3._linear_logits(model, out, lengths)
                ys.append(y.numpy())
                preds.append(scores.argmax(1).numpy())
                lengths_np = lengths.numpy()
                for index, layer in enumerate(("L1", "L2")):
                    spikes = out["spike"][index].numpy().astype(np.uint8)
                    layer_counts[layer].append(pdiag._whole_count(spikes, lengths_np))
                    layer_longest[layer].append(pdiag._longest_runs(spikes, lengths_np))
            y_all = np.concatenate(ys)
            pred_all = np.concatenate(preds)
            native[split] = metrics(y_all, pred_all)
            lengths_all = arrays[f"{split}_lengths"].astype(np.float64)
            for layer in ("L1", "L2"):
                counts = np.concatenate(layer_counts[layer])
                longest = np.concatenate(layer_longest[layer])
                result[layer][split] = {
                    "counts": counts,
                    "occupancy": counts / lengths_all[:, None],
                    "longest": longest,
                }
    return result, native


def _native_frozen_ablation(
    model: torch.nn.Module,
    features: dict[str, dict[str, np.ndarray]],
    arrays: dict[str, np.ndarray],
    selected: np.ndarray,
) -> dict[str, dict[str, float]]:
    weight = model.head.weight.detach().cpu().numpy().astype(np.float64)
    train_mean = features["train"]["occupancy"].mean(axis=0)
    out: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        x = features[split]["occupancy"].copy()
        scores_full = x @ weight.T
        full = metrics(arrays[f"{split}_y"], scores_full.argmax(1))
        x[:, selected] = train_mean[selected]
        scores_ablated = x @ weight.T
        ablated = metrics(arrays[f"{split}_y"], scores_ablated.argmax(1))
        out[split] = {
            "full_ba": full["ba"],
            "ablated_ba": ablated["ba"],
            "drop_pp": 100.0 * (full["ba"] - ablated["ba"]),
        }
    return out


def _diagnose_snapshot(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
    epoch: int,
    roles: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    layers, native = _snapshot_features(model, arrays, p, seed)
    row: dict[str, Any] = {
        "seed": seed,
        "epoch": epoch,
        "snapshot_roles": ";".join(roles),
    }
    for split in SPLITS:
        for name, value in native[split].items():
            row[f"{split}_native_{name}"] = value

    for layer in ("L1", "L2"):
        for split in SPLITS:
            occupancy = layers[layer][split]["occupancy"]
            longest = layers[layer][split]["longest"]
            neuron_occ = occupancy.mean(axis=0)
            prefix = f"{layer.lower()}_{split}"
            row[f"{prefix}_mean_occupancy"] = float(neuron_occ.mean())
            row[f"{prefix}_p90_neuron_occupancy"] = float(np.quantile(neuron_occ, 0.90))
            row[f"{prefix}_mean_longest_run"] = float(longest.mean())
            row[f"{prefix}_p90_neuron_longest_run"] = float(
                np.quantile(longest.mean(axis=0), 0.90)
            )
            for threshold in OCCUPANCY_THRESHOLDS:
                tag = int(round(100 * threshold))
                row[f"{prefix}_fraction_neurons_occ_gt_{tag}"] = float(
                    (neuron_occ > threshold).mean()
                )

    train_occ = layers["L2"]["train"]["occupancy"].mean(axis=0)
    ranking = np.argsort(-train_occ, kind="stable")
    q = max(1, int(math.ceil(PERSISTENT_QUARTILE * len(ranking))))
    high = np.sort(ranking[:q])
    low = np.sort(ranking[-q:])

    head_weight = np.linalg.norm(
        model.head.weight.detach().cpu().numpy().astype(np.float64), axis=0
    )
    total_weight = float(head_weight.sum())
    row["persistent_top25_head_weight_share"] = (
        float(head_weight[high].sum() / total_weight) if total_weight > 0 else 0.0
    )
    row["persistent_top25_head_weight_mean"] = float(head_weight[high].mean())
    row["low25_head_weight_mean"] = float(head_weight[low].mean())

    neuron_rows: list[dict[str, Any]] = []
    eta_by_split: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        counts = layers["L2"][split]["counts"]
        class_eta = pdiag._eta_squared(counts, arrays[f"{split}_y"])
        user_eta = pdiag._eta_squared(counts, arrays[f"{split}_users"])
        eta_by_split[split] = class_eta, user_eta
        for subset_name, chosen in (("high25", high), ("low25", low)):
            row[f"{split}_{subset_name}_class_eta2_mean"] = float(
                np.nanmean(class_eta[chosen])
            )
            row[f"{split}_{subset_name}_user_eta2_mean"] = float(
                np.nanmean(user_eta[chosen])
            )

    for neuron in range(len(train_occ)):
        record: dict[str, Any] = {
            "seed": seed,
            "epoch": epoch,
            "neuron": neuron,
            "train_mean_occupancy": float(train_occ[neuron]),
            "head_weight_norm": float(head_weight[neuron]),
            "persistent_quartile": "high25" if neuron in set(high.tolist()) else (
                "low25" if neuron in set(low.tolist()) else "middle50"
            ),
        }
        for split in SPLITS:
            class_eta, user_eta = eta_by_split[split]
            record[f"{split}_class_eta2"] = float(class_eta[neuron])
            record[f"{split}_user_eta2"] = float(user_eta[neuron])
        neuron_rows.append(record)

    counts = {split: layers["L2"][split]["counts"] for split in SPLITS}
    scaler, probe, scaled, score_map, baseline = pdiag._fit_wholecount_probe(
        counts, arrays, p
    )
    k = max(1, int(math.ceil(LONGITUDINAL_PRUNE_FRACTION * len(ranking))))
    selected = np.sort(ranking[:k])
    frozen = {
        split: pdiag._group_frozen_ablation(
            selected,
            split,
            scaled,
            score_map,
            counts,
            arrays,
            scaler,
            probe,
        )
        for split in SPLITS
    }
    retrained = pdiag._fit_reduced_probe(selected, counts, arrays, p)
    native_frozen = _native_frozen_ablation(model, layers["L2"], arrays, selected)
    for split in SPLITS:
        row[f"{split}_wholecount_full_ba"] = baseline[split]["ba"]
        row[f"{split}_wholecount_frozen_remove_high30_ba"] = frozen[split]["ba"]
        row[f"{split}_wholecount_frozen_remove_high30_drop_pp"] = 100.0 * (
            baseline[split]["ba"] - frozen[split]["ba"]
        )
        row[f"{split}_wholecount_retrained_remove_high30_ba"] = retrained[split]["ba"]
        row[f"{split}_wholecount_retrained_remove_high30_drop_pp"] = 100.0 * (
            baseline[split]["ba"] - retrained[split]["ba"]
        )
        row[f"{split}_native_frozen_remove_high30_drop_pp"] = native_frozen[split]["drop_pp"]
    return row, neuron_rows


def run_trajectory_task(config: Config, task_id: int) -> dict[str, Any]:
    specs = trajectory_specs()
    if not 0 <= task_id < len(specs):
        raise IndexError(task_id)
    seed = specs[task_id]
    _train_trajectory(config, seed)

    p, _, arrays = _core(config)
    directory = config.results_dir / "trajectory" / f"seed{seed}"
    manifest = json.loads((directory / "snapshot_manifest.json").read_text(encoding="utf-8"))
    output = directory / "trajectory_metrics.csv"
    neuron_output = directory / "trajectory_neurons.csv"
    if output.exists() and neuron_output.exists():
        return {"status": "exists", "seed": seed}

    summary_rows: list[dict[str, Any]] = []
    neuron_rows: list[dict[str, Any]] = []
    for snapshot in manifest["snapshots"]:
        epoch = int(snapshot["epoch"])
        payload = load_torch(_snapshot_path(directory, epoch))
        model = _make_trajectory_model(seed, p)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        summary, neurons = _diagnose_snapshot(
            model, arrays, p, seed, epoch, list(snapshot["roles"])
        )
        summary_rows.append(summary)
        neuron_rows.extend(neurons)
        print(f"Exp17 trajectory seed{seed}: diagnosed epoch {epoch}", flush=True)

    pd.DataFrame(summary_rows).sort_values("epoch").to_csv(output, index=False)
    pd.DataFrame(neuron_rows).sort_values(["epoch", "neuron"]).to_csv(
        neuron_output, index=False
    )
    return {"status": "PASS", "seed": seed, "snapshots": len(summary_rows)}


def _load_artifact_counts(
    config: Config,
    spec: ArtifactSpec,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    traces = pdiag._load_layer_traces(config.exp16_2_results_dir, spec.case, spec.seed)
    counts = {
        split: pdiag._whole_count(
            traces[split]["L2"], arrays[f"{split}_lengths"]
        )
        for split in SPLITS
    }
    train_occ = counts["train"] / arrays["train_lengths"][:, None].astype(np.float64)
    return counts, train_occ.mean(axis=0)


def _random_subset_indices(
    width: int,
    n_selected: int,
    seed: int,
    case: str,
    family: str,
    replicate: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        paired_seed(seed, f"exp17:{family}:{case}:rep{replicate}")
    )
    return np.sort(rng.choice(width, size=n_selected, replace=False))


def _subset_specs(
    train_mean_occupancy: np.ndarray,
    seed: int,
    case: str,
) -> list[tuple[str, int, np.ndarray]]:
    width = len(train_mean_occupancy)
    k = max(1, int(math.ceil(SUBSET_FRACTION * width)))
    ranking = np.argsort(-train_mean_occupancy, kind="stable")
    rows: list[tuple[str, int, np.ndarray]] = [
        ("high30", -1, np.sort(ranking[:k])),
        ("low30", -1, np.sort(ranking[-k:])),
    ]
    rows.extend(
        (
            "random30",
            replicate,
            _random_subset_indices(width, k, seed, case, "subset", replicate),
        )
        for replicate in range(SUBSET_RANDOM_REPLICATES)
    )
    return rows


def _eval_probe_result(
    features: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    p: Protocol,
) -> dict[str, Any]:
    scaler, model, search = fit_probe(
        features["train"],
        arrays["train_y"],
        features["val"],
        arrays["val_y"],
        "no_bias",
        p,
    )
    row: dict[str, Any] = {
        "C": float(model.C),
        "dimension": int(features["train"].shape[1]),
        "search_rows": len(search),
    }
    for split in SPLITS:
        prediction = model.predict(scaler.transform(features[split].astype(np.float64)))
        for name, value in metrics(arrays[f"{split}_y"], prediction).items():
            row[f"{split}_{name}"] = value
    row["train_test_gap"] = row["train_ba"] - row["test_ba"]
    return row


def _user_cv_fold_ids(users: np.ndarray, seed: int, n_folds: int = USER_CV_FOLDS) -> np.ndarray:
    if n_folds < 3:
        raise ValueError("User CV needs >=3 folds for train/validation/test separation")
    folds = np.full(len(users), -1, dtype=np.int64)
    for user in np.unique(users):
        chosen = np.flatnonzero(users == user)
        if len(chosen) < n_folds:
            raise ValueError(f"User {user!r} has only {len(chosen)} samples for {n_folds}-fold CV")
        rng = np.random.default_rng(paired_seed(seed, f"exp17:user-cv:{user}"))
        shuffled = chosen[rng.permutation(len(chosen))]
        folds[shuffled] = np.arange(len(shuffled), dtype=np.int64) % n_folds
    if (folds < 0).any():
        raise AssertionError("Unassigned user-CV sample")
    return folds


def _class_residualized_triplet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means: dict[Any, np.ndarray] = {}
    for label in np.unique(np.concatenate((y_train, y_val, y_test))):
        selected = y_train == label
        if not np.any(selected):
            raise ValueError(f"Class {label!r} absent from user-CV training fold")
        means[label] = x_train[selected].mean(axis=0)

    def residualize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray([row - means[label] for row, label in zip(x, y)], dtype=np.float64)

    return residualize(x_train, y_train), residualize(x_val, y_val), residualize(x_test, y_test)


def _user_decode_rows(
    x: np.ndarray,
    y_class: np.ndarray,
    users: np.ndarray,
    p: Protocol,
    seed: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    folds = _user_cv_fold_ids(users, seed)
    rows: list[dict[str, Any]] = []
    for test_fold in range(USER_CV_FOLDS):
        val_fold = (test_fold + 1) % USER_CV_FOLDS
        test = folds == test_fold
        val = folds == val_fold
        train = ~(test | val)
        for mode in ("raw", "class_residualized"):
            x_train, x_val, x_test = x[train], x[val], x[test]
            if mode == "class_residualized":
                x_train, x_val, x_test = _class_residualized_triplet(
                    x_train,
                    y_class[train],
                    x_val,
                    y_class[val],
                    x_test,
                    y_class[test],
                )
            scaler, model, _ = fit_probe(
                x_train,
                users[train],
                x_val,
                users[val],
                "no_bias",
                p,
            )
            prediction = model.predict(scaler.transform(x_test.astype(np.float64)))
            result = metrics(users[test], prediction)
            rows.append({
                **metadata,
                "mode": mode,
                "fold": test_fold,
                "C": float(model.C),
                "n_train": int(train.sum()),
                "n_val": int(val.sum()),
                "n_test": int(test.sum()),
                **{f"test_{name}": value for name, value in result.items()},
            })
    return rows


def run_subset_task(config: Config, task_id: int) -> dict[str, Any]:
    specs = artifact_specs()
    if not 0 <= task_id < len(specs):
        raise IndexError(task_id)
    spec = specs[task_id]
    p, _, arrays = _core(config)
    directory = config.results_dir / "subset" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    class_path = directory / "class_decoding.csv"
    user_path = directory / "user_decoding.csv"
    manifest_path = directory / "subset_manifest.csv"
    if class_path.exists() and user_path.exists() and manifest_path.exists():
        return {"status": "exists", "run": spec.key}

    counts, train_mean_occupancy = _load_artifact_counts(config, spec, arrays)
    class_eta = pdiag._eta_squared(counts["train"], arrays["train_y"])
    user_eta = pdiag._eta_squared(counts["train"], arrays["train_users"])
    class_rows: list[dict[str, Any]] = []
    user_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []

    for subset_kind, replicate, selected in _subset_specs(
        train_mean_occupancy, spec.seed, spec.case
    ):
        metadata = {
            "case": spec.case,
            "seed": spec.seed,
            "subset": subset_kind,
            "replicate": replicate,
            "n_neurons": len(selected),
        }
        features = {split: counts[split][:, selected] for split in SPLITS}
        class_row = _eval_probe_result(features, arrays, p)
        class_row.update(metadata)
        class_row["train_mean_occupancy"] = float(train_mean_occupancy[selected].mean())
        class_row["train_mean_class_eta2"] = float(np.nanmean(class_eta[selected]))
        class_row["train_mean_user_eta2"] = float(np.nanmean(user_eta[selected]))
        class_rows.append(class_row)

        user_rows.extend(
            _user_decode_rows(
                counts["train"][:, selected],
                arrays["train_y"],
                arrays["train_users"],
                p,
                spec.seed,
                metadata,
            )
        )
        manifests.append({
            **metadata,
            "selected_neurons": ";".join(str(int(v)) for v in selected.tolist()),
        })

    pd.DataFrame(class_rows).to_csv(class_path, index=False)
    pd.DataFrame(user_rows).to_csv(user_path, index=False)
    pd.DataFrame(manifests).to_csv(manifest_path, index=False)
    return {"status": "PASS", "run": spec.key, "subsets": len(manifests)}


def _pruning_rankings(
    train_mean_occupancy: np.ndarray,
    probe: Any,
) -> dict[str, np.ndarray]:
    high_occ = np.argsort(-train_mean_occupancy, kind="stable")
    low_occ = np.argsort(train_mean_occupancy, kind="stable")
    standardized_weight = np.linalg.norm(
        np.asarray(probe.coef_, dtype=np.float64), axis=0
    )
    high_weight = np.argsort(-standardized_weight, kind="stable")
    return {
        "high_occupancy": high_occ,
        "low_occupancy": low_occ,
        "high_readout_weight": high_weight,
    }


def _pruning_row(
    spec: ArtifactSpec,
    fraction: float,
    ranking_kind: str,
    replicate: int,
    selected: np.ndarray,
    train_mean_occupancy: np.ndarray,
    baseline: dict[str, dict[str, float]],
    frozen: dict[str, dict[str, float]],
    retrained: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "case": spec.case,
        "seed": spec.seed,
        "fraction_removed": fraction,
        "ranking": ranking_kind,
        "replicate": replicate,
        "n_removed": len(selected),
        "selected_train_mean_occupancy": float(train_mean_occupancy[selected].mean()),
        "selected_neurons": ";".join(str(int(v)) for v in selected.tolist()),
    }
    for method, result in (
        ("frozen_mean_replacement", frozen),
        ("retrained_without_neurons", retrained),
    ):
        row = {**base, "method": method}
        for split in SPLITS:
            row[f"{split}_ba"] = result[split]["ba"]
            row[f"{split}_ce"] = result[split]["ce"]
            row[f"{split}_delta_ba_pp_vs_full"] = 100.0 * (
                baseline[split]["ba"] - result[split]["ba"]
            )
            row[f"{split}_delta_ce_vs_full"] = (
                result[split]["ce"] - baseline[split]["ce"]
            )
        rows.append(row)
    return rows


def run_pruning_task(config: Config, task_id: int) -> dict[str, Any]:
    specs = artifact_specs()
    if not 0 <= task_id < len(specs):
        raise IndexError(task_id)
    spec = specs[task_id]
    p, _, arrays = _core(config)
    directory = config.results_dir / "pruning" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "pruning_specificity.csv"
    if output.exists():
        return {"status": "exists", "run": spec.key}

    features, train_mean_occupancy = _load_artifact_counts(config, spec, arrays)
    scaler, probe, scaled, score_map, baseline = pdiag._fit_wholecount_probe(
        features, arrays, p
    )
    rankings = _pruning_rankings(train_mean_occupancy, probe)
    rows: list[dict[str, Any]] = []
    width = features["train"].shape[1]

    for fraction in PRUNE_FRACTIONS:
        k = max(1, int(math.ceil(fraction * width)))
        selections: list[tuple[str, int, np.ndarray]] = []
        for name, ranking in rankings.items():
            selections.append((name, -1, np.sort(ranking[:k])))
        for replicate in range(PRUNE_RANDOM_REPLICATES):
            selections.append((
                "random",
                replicate,
                _random_subset_indices(width, k, spec.seed, spec.case, f"prune-{fraction}", replicate),
            ))

        for ranking_kind, replicate, selected in selections:
            frozen = {
                split: pdiag._group_frozen_ablation(
                    selected,
                    split,
                    scaled,
                    score_map,
                    features,
                    arrays,
                    scaler,
                    probe,
                )
                for split in SPLITS
            }
            retrained = pdiag._fit_reduced_probe(selected, features, arrays, p)
            rows.extend(
                _pruning_row(
                    spec,
                    fraction,
                    ranking_kind,
                    replicate,
                    selected,
                    train_mean_occupancy,
                    baseline,
                    frozen,
                    retrained,
                )
            )

    pd.DataFrame(rows).to_csv(output, index=False)
    return {"status": "PASS", "run": spec.key, "rows": len(rows)}


def prepare(config: Config) -> dict[str, Any]:
    p, lock, _ = _core(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    source_protocol = config.exp16_2_results_dir / "protocol.json"
    if not source_protocol.exists():
        raise FileNotFoundError(source_protocol)
    missing = [
        str(config.exp16_2_results_dir / "runs" / spec.key / "traces.npz")
        for spec in artifact_specs()
        if not (config.exp16_2_results_dir / "runs" / spec.key / "traces.npz").exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Exp17-B/C require finalized Exp16.2 traces; missing: " + ", ".join(missing[:5])
        )

    exp16_2_protocol = json.loads(source_protocol.read_text(encoding="utf-8"))
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "core_dataset_hash": lock["dataset_hash"],
        "source_exp16_2_protocol": exp16_2_protocol.get("protocol_version"),
        "formal_seeds": list(FORMAL_SEEDS),
        "source_cases": list(SOURCE_CASES),
        "trajectory": {
            "objective": "locked LIN/WCCE equivalent from Exp16.3",
            "checkpoint_every_epochs": CHECKPOINT_EVERY,
            "save_initial_best_stopped": True,
            "persistent_quartile": PERSISTENT_QUARTILE,
            "longitudinal_prune_fraction": LONGITUDINAL_PRUNE_FRACTION,
            "metrics": [
                "L1/L2 occupancy and longest-run statistics",
                "class eta-squared and user eta-squared by persistence quartile",
                "native head weight share on top-persistence quartile",
                "native and WholeCount high-persistence removal",
                "retrained WholeCount recovery after high-persistence removal",
            ],
        },
        "subset_decoding": {
            "fraction": SUBSET_FRACTION,
            "sets": ["high30", "low30", "random30"],
            "random_replicates": SUBSET_RANDOM_REPLICATES,
            "class_decoder": "bias-free WholeCount probe; validation-only C selection",
            "user_decoder": (
                f"{USER_CV_FOLDS}-fold within-training-user CV with distinct train/val/test folds; "
                "report raw and class-residualized features"
            ),
        },
        "pruning_specificity": {
            "fractions": list(PRUNE_FRACTIONS),
            "rankings": [
                "high_occupancy",
                "low_occupancy",
                "high_readout_weight",
                "random",
            ],
            "random_replicates": PRUNE_RANDOM_REPLICATES,
            "methods": ["frozen_mean_replacement", "retrained_without_neurons"],
        },
        "hard_contracts": {
            "only_trajectory_retrains_snn": True,
            "subset_and_pruning_are_artifact_only": True,
            "persistence_rankings_use_training_split_only": True,
            "probe_C_selection_uses_validation_only": True,
            "test_never_affects_training_or_selection": True,
            "random_controls_are_dimension_matched": True,
            "trajectory_initialization_and_epoch_sampler_match_exp16_3_LIN": True,
            "no_new_auxiliary_loss_or_gate": True,
        },
        "architecture": {
            "width": p.width,
            "fs": p.fs,
            "shifts": [list(v) for v in exp16_3.SHIFTS],
            "tau_mem_ms": p.tau_mem_ms,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["identity"] = hashlib.sha256(encoded).hexdigest()
    save_json(config.results_dir / "protocol.json", payload)
    return payload


def _read_csvs(paths: list[Path]) -> pd.DataFrame:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Exp17 task outputs: " + ", ".join(missing[:8]))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _random_control_summary(pruning: pd.DataFrame) -> pd.DataFrame:
    group = ["case", "fraction_removed", "ranking", "method"]
    metrics_cols = [
        "train_delta_ba_pp_vs_full",
        "val_delta_ba_pp_vs_full",
        "test_delta_ba_pp_vs_full",
        "test_delta_ce_vs_full",
    ]
    rows: list[dict[str, Any]] = []
    for keys, frame in pruning.groupby(group, sort=True):
        row = dict(zip(group, keys))
        for column in metrics_cols:
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_std"] = float(frame[column].std(ddof=0))
        row["n_rows"] = len(frame)
        rows.append(row)
    return pd.DataFrame(rows)


def _story_summary(
    trajectory: pd.DataFrame,
    subset_class: pd.DataFrame,
    subset_user: pd.DataFrame,
    pruning: pd.DataFrame,
) -> dict[str, Any]:
    trajectory_delta: list[dict[str, Any]] = []
    for seed, frame in trajectory.groupby("seed"):
        frame = frame.sort_values("epoch")
        first = frame.iloc[0]
        last = frame.iloc[-1]
        trajectory_delta.append({
            "seed": int(seed),
            "from_epoch": int(first["epoch"]),
            "to_epoch": int(last["epoch"]),
            "delta_l2_train_occ_gt50": float(
                last["l2_train_fraction_neurons_occ_gt_50"]
                - first["l2_train_fraction_neurons_occ_gt_50"]
            ),
            "delta_persistent_top25_head_weight_share": float(
                last["persistent_top25_head_weight_share"]
                - first["persistent_top25_head_weight_share"]
            ),
            "delta_train_high25_user_eta2": float(
                last["train_high25_user_eta2_mean"]
                - first["train_high25_user_eta2_mean"]
            ),
            "delta_test_native_ba": float(last["test_native_ba"] - first["test_native_ba"]),
            "final_train_native_ba": float(last["train_native_ba"]),
            "final_test_native_ba": float(last["test_native_ba"]),
        })

    class_summary = (
        subset_class.groupby("subset", as_index=False)["test_ba"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    residual_user = subset_user[subset_user["mode"] == "class_residualized"]
    user_summary = (
        residual_user.groupby("subset", as_index=False)["test_ba"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    prune30 = pruning[
        (np.isclose(pruning["fraction_removed"], 0.30))
        & (pruning["method"] == "retrained_without_neurons")
    ]
    pruning_summary = (
        prune30.groupby("ranking", as_index=False)["test_delta_ba_pp_vs_full"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "trajectory_initial_to_last_snapshot": trajectory_delta,
        "subset_class_test_ba": class_summary,
        "class_residualized_user_test_ba": user_summary,
        "prune30_retrained_test_drop_pp": pruning_summary,
        "interpretation_rule": (
            "These are descriptive closure diagnostics. Exp17 does not auto-accept or reject the story; "
            "claim strength must follow the paired seed/control patterns."
        ),
    }


def finalize(config: Config) -> dict[str, Any]:
    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)

    trajectory = _read_csvs([
        config.results_dir / "trajectory" / f"seed{seed}" / "trajectory_metrics.csv"
        for seed in FORMAL_SEEDS
    ])
    trajectory_neurons = _read_csvs([
        config.results_dir / "trajectory" / f"seed{seed}" / "trajectory_neurons.csv"
        for seed in FORMAL_SEEDS
    ])
    subset_class = _read_csvs([
        config.results_dir / "subset" / spec.key / "class_decoding.csv"
        for spec in artifact_specs()
    ])
    subset_user = _read_csvs([
        config.results_dir / "subset" / spec.key / "user_decoding.csv"
        for spec in artifact_specs()
    ])
    pruning = _read_csvs([
        config.results_dir / "pruning" / spec.key / "pruning_specificity.csv"
        for spec in artifact_specs()
    ])

    trajectory.sort_values(["seed", "epoch"]).to_csv(
        aggregate / "trajectory_metrics.csv", index=False
    )
    trajectory_neurons.sort_values(["seed", "epoch", "neuron"]).to_csv(
        aggregate / "trajectory_neurons.csv", index=False
    )
    subset_class.to_csv(aggregate / "subset_class_decoding.csv", index=False)
    subset_user.to_csv(aggregate / "subset_user_decoding.csv", index=False)
    pruning.to_csv(aggregate / "pruning_specificity.csv", index=False)
    _random_control_summary(pruning).to_csv(
        aggregate / "pruning_specificity_summary.csv", index=False
    )

    subset_class_summary = subset_class.groupby(
        ["case", "subset"], as_index=False
    ).agg(
        test_ba_mean=("test_ba", "mean"),
        test_ba_std=("test_ba", "std"),
        train_test_gap_mean=("train_test_gap", "mean"),
        n=("test_ba", "size"),
    )
    subset_class_summary.to_csv(aggregate / "subset_class_summary.csv", index=False)
    subset_user_summary = subset_user.groupby(
        ["case", "subset", "mode"], as_index=False
    ).agg(
        test_ba_mean=("test_ba", "mean"),
        test_ba_std=("test_ba", "std"),
        n=("test_ba", "size"),
    )
    subset_user_summary.to_csv(aggregate / "subset_user_summary.csv", index=False)

    story = _story_summary(trajectory, subset_class, subset_user, pruning)
    save_json(aggregate / "story_summary.json", story)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "trajectory_rows": len(trajectory),
        "trajectory_neuron_rows": len(trajectory_neurons),
        "subset_class_rows": len(subset_class),
        "subset_user_rows": len(subset_user),
        "pruning_rows": len(pruning),
        "outputs": [
            "trajectory_metrics.csv",
            "trajectory_neurons.csv",
            "subset_class_decoding.csv",
            "subset_user_decoding.csv",
            "subset_class_summary.csv",
            "subset_user_summary.csv",
            "pruning_specificity.csv",
            "pruning_specificity_summary.csv",
            "story_summary.json",
        ],
    }
    save_json(aggregate / "manifest.json", manifest)
    return {"status": "PASS", **manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    parser.add_argument("--exp16-2-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    for name in ("trajectory", "subset", "pruning"):
        phase = sub.add_parser(name)
        phase.add_argument("--task-id", type=int, required=True)
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    if args.command == "prepare":
        result = prepare(config)
    elif args.command == "trajectory":
        result = run_trajectory_task(config, args.task_id)
    elif args.command == "subset":
        result = run_subset_task(config, args.task_id)
    elif args.command == "pruning":
        result = run_pruning_task(config, args.task_id)
    elif args.command == "finalize":
        result = finalize(config)
    else:
        raise ValueError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()