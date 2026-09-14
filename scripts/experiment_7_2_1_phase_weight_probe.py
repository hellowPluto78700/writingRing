from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objective_comparison as exp3
from scripts import experiment_7_2_two_layer_tau_training as exp72

EXPERIMENT_ID = "experiment_7_2_1_phase_weight_probe"
PROTOCOL_VERSION = "phase_weight_probe_v1"
SHUFFLE_REPEATS = 3
SOURCE_WHOLE = "l2_wholecount_shared"
SOURCE_ORDERED = "l2_fixed250_ordered"
SOURCE_SHUFFLED = "l2_fixed250_phase_shuffled"
SOURCES = (SOURCE_WHOLE, SOURCE_ORDERED, SOURCE_SHUFFLED)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp72.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[exp72.RunSpec]:
    # Exactly reuse the 84 frozen Exp7.2 checkpoints.
    return exp72.run_specs()


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_path(root: Path, spec: exp72.RunSpec) -> Path:
    return root / "runs" / f"{spec.key}.json"


def _base_config(config: Config) -> exp72.Config:
    return exp72.Config(
        config.repo_root,
        exp72.results_dir(config.repo_root),
        config.device,
        exp72.MAX_EPOCHS,
        config.batch_size,
        config.threads,
    )


def _ordered_probe_path(config: Config, spec: exp72.RunSpec) -> Path:
    return (
        exp72.results_dir(config.repo_root)
        / "probes"
        / "l2_fixed250"
        / f"{spec.key}.json"
    )


def _extract_l2_fixed250(
    model: exp72.TwoLayerTauSNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    bin_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            lengths_d = lengths.to(device)
            tr = model.forward_trajectory(X)
            l2 = tr["hidden_spikes"][-1]
            counts = exp3.fixed_counts(l2, lengths_d, bin_steps)
            xs.append(counts.cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _whole_count_features(fixed250: np.ndarray) -> np.ndarray:
    if fixed250.ndim != 3:
        raise ValueError(f"Expected [N,B,D], got {fixed250.shape}")
    # A shared temporal classifier is exactly W sum_b z_b = sum_b W z_b.
    return fixed250.sum(axis=1)


def _flatten_ordered(fixed250: np.ndarray) -> np.ndarray:
    return fixed250.reshape(fixed250.shape[0], -1)


def _phase_shuffle(fixed250: np.ndarray, seed: int) -> np.ndarray:
    """Destroy absolute phase labels while preserving each sample's bin contents."""
    if fixed250.ndim != 3:
        raise ValueError(f"Expected [N,B,D], got {fixed250.shape}")
    rng = np.random.default_rng(seed)
    out = np.empty_like(fixed250)
    for i in range(len(fixed250)):
        out[i] = fixed250[i, rng.permutation(fixed250.shape[1])]
    return out


def _fit_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    return exp01._fit_linear_probe(
        train_x, train_y, val_x, val_y, test_x, test_y, seed
    )


def _normalize_probe(probe: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "feature_dim": int(probe["feature_dim"]),
        "probe_C": float(probe["probe_C"]),
        "metrics": {split: dict(probe[split]) for split in ("train", "val", "test")},
    }


def _mean_repeat_metrics(
    repeats: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    means: dict[str, dict[str, float]] = {}
    stds: dict[str, dict[str, float]] = {}
    for split in ("train", "val", "test"):
        means[split], stds[split] = {}, {}
        for metric in METRICS:
            values = np.asarray(
                [r["metrics"][split][metric] for r in repeats], dtype=float
            )
            means[split][metric] = float(values.mean())
            stds[split][metric] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
    return means, stds


def evaluate_one(
    spec: exp72.RunSpec,
    data: exp3.Data,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    out_path = _run_path(config.results_dir, spec)
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    base = _base_config(config)
    checkpoint = exp72._path(base.results_dir, "checkpoints", spec, ".pt")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing Exp7.2 checkpoint for {spec.key}: {checkpoint}. "
            "This extension never trains an SNN."
        )

    ordered_path = _ordered_probe_path(config, spec)
    if not ordered_path.exists():
        raise FileNotFoundError(
            f"Missing Exp7.2 ordered Fixed250 probe for {spec.key}: {ordered_path}"
        )

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model, checkpoint_payload = exp72.load_model(spec, data, base)
    split_loaders = exp72.loaders(data, spec.seed, config.batch_size, False)
    fixed = {
        split: _extract_l2_fixed250(model, loader, device, data.bin_steps)
        for split, loader in split_loaders.items()
    }

    whole = {
        split: (_whole_count_features(x), y)
        for split, (x, y) in fixed.items()
    }
    whole_raw = _fit_probe(
        whole["train"][0], whole["train"][1],
        whole["val"][0], whole["val"][1],
        whole["test"][0], whole["test"][1],
        exp3.dseed(spec.seed, EXPERIMENT_ID, spec.key, SOURCE_WHOLE),
    )
    whole_probe = _normalize_probe(whole_raw, SOURCE_WHOLE)

    # Reuse the exact ordered Fixed250 probe already produced by Exp7.2.
    ordered_existing = json.loads(ordered_path.read_text(encoding="utf-8"))
    ordered_probe = {
        "source": SOURCE_ORDERED,
        "feature_dim": int(ordered_existing["feature_dim"]),
        "probe_C": float(ordered_existing["probe_C"]),
        "metrics": ordered_existing["metrics"],
        "reused_from": str(ordered_path.relative_to(config.repo_root)),
    }

    shuffled_repeats: list[dict[str, Any]] = []
    for repeat in range(SHUFFLE_REPEATS):
        shuffled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split, (x, y) in fixed.items():
            shuffle_seed = exp3.dseed(
                spec.seed,
                EXPERIMENT_ID,
                spec.key,
                SOURCE_SHUFFLED,
                split,
                repeat,
            )
            shuffled[split] = (
                _flatten_ordered(_phase_shuffle(x, shuffle_seed)),
                y,
            )
        raw = _fit_probe(
            shuffled["train"][0], shuffled["train"][1],
            shuffled["val"][0], shuffled["val"][1],
            shuffled["test"][0], shuffled["test"][1],
            exp3.dseed(spec.seed, EXPERIMENT_ID, spec.key, SOURCE_SHUFFLED, repeat),
        )
        shuffled_repeats.append(_normalize_probe(raw, SOURCE_SHUFFLED))

    shuffled_mean, shuffled_std = _mean_repeat_metrics(shuffled_repeats)
    shuffled_probe = {
        "source": SOURCE_SHUFFLED,
        "feature_dim": int(shuffled_repeats[0]["feature_dim"]),
        "shuffle_repeats": SHUFFLE_REPEATS,
        "probe_C_values": [float(r["probe_C"]) for r in shuffled_repeats],
        "metrics": shuffled_mean,
        "repeat_std": shuffled_std,
    }

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "reuses_exp7_2_checkpoint": True,
        "snn_retrained": False,
        "checkpoint": str(checkpoint.relative_to(config.repo_root)),
        "checkpoint_best_epoch": int(checkpoint_payload["best_epoch"]),
        "probes": {
            SOURCE_WHOLE: whole_probe,
            SOURCE_ORDERED: ordered_probe,
            SOURCE_SHUFFLED: shuffled_probe,
        },
    }
    _save_json(out_path, payload)
    return payload


def _performance_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    base = {
        "architecture": spec["architecture"],
        "training_family": spec["training_family"],
        "regularization": spec["regularization"],
        "seed": int(spec["seed"]),
    }
    rows: list[dict[str, Any]] = []
    for source, probe in payload["probes"].items():
        for split, metrics in probe["metrics"].items():
            rows.append({
                **base,
                "source": source,
                "split": split,
                **{metric: float(metrics[metric]) for metric in METRICS},
            })
    return rows


def _aggregate(
    df: pd.DataFrame, groups: list[str], values: list[str]
) -> pd.DataFrame:
    grouped = df.groupby(groups, dropna=False)[values]
    return pd.concat(
        [
            grouped.size().rename("n"),
            grouped.mean().add_suffix("_mean"),
            grouped.std(ddof=1).fillna(0).add_suffix("_std"),
        ],
        axis=1,
    ).reset_index()


def finalize(repo_root: Path) -> dict[str, Any]:
    root = results_dir(repo_root)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in run_specs():
        path = _run_path(root, spec)
        if not path.exists():
            missing.append(spec.key)
            continue
        rows.extend(_performance_rows(json.loads(path.read_text(encoding="utf-8"))))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} extension runs; first={missing[:8]}")

    runs = pd.DataFrame(rows)
    runs.to_csv(root / "probe_runs.csv", index=False)
    summary = _aggregate(
        runs,
        ["architecture", "training_family", "regularization", "source", "split"],
        list(METRICS),
    )
    summary.to_csv(root / "probe_summary.csv", index=False)

    deltas: list[dict[str, Any]] = []
    test = runs[runs.split == "test"].copy()
    contrasts = (
        ("ordered_minus_wholecount", SOURCE_ORDERED, SOURCE_WHOLE),
        ("ordered_minus_phase_shuffled", SOURCE_ORDERED, SOURCE_SHUFFLED),
        ("phase_shuffled_minus_wholecount", SOURCE_SHUFFLED, SOURCE_WHOLE),
    )
    for (architecture, family, regularization, seed), group in test.groupby(
        ["architecture", "training_family", "regularization", "seed"]
    ):
        indexed = group.set_index("source")
        for contrast, left, right in contrasts:
            if left not in indexed.index or right not in indexed.index:
                continue
            for metric in METRICS:
                deltas.append({
                    "contrast": contrast,
                    "architecture": architecture,
                    "training_family": family,
                    "regularization": regularization,
                    "seed": int(seed),
                    "metric": metric,
                    "delta": float(indexed.loc[left, metric] - indexed.loc[right, metric]),
                })
    delta_runs = pd.DataFrame(deltas)
    delta_runs.to_csv(root / "paired_delta_runs.csv", index=False)
    delta_summary = _aggregate(
        delta_runs,
        ["contrast", "architecture", "training_family", "regularization", "metric"],
        ["delta"],
    )
    delta_summary.to_csv(root / "paired_deltas.csv", index=False)

    architecture_source = exp72.results_dir(repo_root) / "architecture_table.csv"
    if not architecture_source.exists():
        raise FileNotFoundError(
            f"Run Exp7.2 finalizer first; missing {architecture_source}"
        )
    pd.read_csv(architecture_source).to_csv(root / "architecture_table.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "reuses_exp7_2_checkpoints": True,
        "snn_training_runs": 0,
        "checkpoint_evaluations": len(run_specs()),
        "shuffle_repeats": SHUFFLE_REPEATS,
        "sources": list(SOURCES),
        "primary_contrasts": [
            "ordered_minus_wholecount",
            "ordered_minus_phase_shuffled",
        ],
        "notebook_inputs": [
            "architecture_table.csv",
            "probe_summary.csv",
            "paired_deltas.csv",
        ],
    }
    _save_json(root / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int)
    run.add_argument("--architecture", choices=exp72.ARCHITECTURE_ORDER)
    run.add_argument("--training-family", choices=exp72.TRAINING_FAMILIES)
    run.add_argument("--regularization", choices=exp72.REGULARIZATION_CONDITIONS)
    run.add_argument("--seed", type=int, choices=exp72.TRAIN_SEEDS)
    run.add_argument("--force", action="store_true")

    sub.add_parser("finalize")
    sub.add_parser("list-runs")
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = find_repo_root(args.repo_root)
    config = Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
    )

    if args.command == "list-runs":
        for i, spec in enumerate(run_specs()):
            print(i, spec.key)
        return
    if args.command == "finalize":
        print(json.dumps(finalize(repo_root), indent=2))
        return

    specs = run_specs()
    if args.array_task_id is not None:
        if not 0 <= args.array_task_id < len(specs):
            raise ValueError(args.array_task_id)
        spec = specs[args.array_task_id]
    else:
        spec = exp72.RunSpec(
            args.architecture,
            args.training_family,
            args.regularization,
            args.seed,
        )
        exp72.validate_spec(spec)

    data = exp72.prepare_data(repo_root)
    payload = evaluate_one(spec, data, config, force=args.force)
    print(json.dumps({
        "spec": payload["spec"],
        "snn_retrained": payload["snn_retrained"],
        "sources": list(payload["probes"]),
    }, indent=2))


if __name__ == "__main__":
    main()
