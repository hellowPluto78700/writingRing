from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objective_comparison as exp3
from scripts import experiment_3_0_2_hidden_multitau_architectures as exp302
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_8_0_local_backbone_tau_sweep as exp80


EXPERIMENT_ID = "experiment_8_0_1_l1_l2_fusion_probe"
PROTOCOL_VERSION = "l1_l2_fusion_probe_v1"
SOURCE_EXPERIMENT_ID = exp80.EXPERIMENT_ID
SOURCE_PROTOCOL_VERSION = exp80.PROTOCOL_VERSION
SOURCE_ARCHITECTURE = "234x234"
SEEDS = exp80.SEEDS


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    components: tuple[tuple[str, str], ...]


FEATURE_SPECS = (
    FeatureSpec("l1_whole", (("l1", "whole_count"),)),
    FeatureSpec("l2_whole", (("l2", "whole_count"),)),
    FeatureSpec("l1_l2_whole", (("l1", "whole_count"), ("l2", "whole_count"))),
    FeatureSpec("l1_fixed250", (("l1", "fixed250"),)),
    FeatureSpec("l2_fixed250", (("l2", "fixed250"),)),
    FeatureSpec("l1_l2_fixed250", (("l1", "fixed250"), ("l2", "fixed250"))),
    FeatureSpec("l1whole_l2fixed250", (("l1", "whole_count"), ("l2", "fixed250"))),
    FeatureSpec("l1fixed250_l2whole", (("l1", "fixed250"), ("l2", "whole_count"))),
)
FEATURE_NAMES = tuple(spec.name for spec in FEATURE_SPECS)


@dataclass(frozen=True)
class RunSpec:
    seed: int

    @property
    def key(self) -> str:
        return f"{SOURCE_ARCHITECTURE}__frozen_fusion__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp80.exp72.BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp80.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / SOURCE_EXPERIMENT_ID / SOURCE_PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(seed) for seed in SEEDS]


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _source_spec(seed: int) -> exp80.RunSpec:
    return exp80.RunSpec(SOURCE_ARCHITECTURE, seed)


def _source_checkpoint_path(repo_root: Path, seed: int) -> Path:
    spec = _source_spec(seed)
    return source_results_dir(repo_root) / "checkpoints" / f"{spec.key}.pt"


def _source_eval_path(repo_root: Path, seed: int) -> Path:
    spec = _source_spec(seed)
    return source_results_dir(repo_root) / "evaluations" / f"{spec.key}.json"


def _load_source_model(
    run: RunSpec, data: exp3.Data, config: Config
) -> tuple[exp80.Exp80Net, dict[str, Any], dict[str, Any]]:
    checkpoint_path = _source_checkpoint_path(config.repo_root, run.seed)
    eval_path = _source_eval_path(config.repo_root, run.seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Exp8.0 source checkpoint: {checkpoint_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing Exp8.0 source evaluation: {eval_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    source_eval = json.loads(eval_path.read_text(encoding="utf-8"))
    if checkpoint.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("Source checkpoint experiment mismatch")
    if checkpoint.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise ValueError("Source checkpoint protocol mismatch")
    if tuple(tuple(int(v) for v in layer) for layer in checkpoint["architecture_shifts"]) != exp80.ARCHITECTURES[SOURCE_ARCHITECTURE]:
        raise ValueError("Exp8.0.1 requires the 234x234 Exp8.0 checkpoint")
    model = exp80.new_model(_source_spec(run.seed), data).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint, source_eval


def _feature_spec(name: str) -> FeatureSpec:
    for spec in FEATURE_SPECS:
        if spec.name == name:
            return spec
    raise ValueError(name)


def _build_features(
    extracted: dict[str, dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]]],
    feature_spec: FeatureSpec,
    split: str,
    bin_steps: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    parts: list[np.ndarray] = []
    blocks: list[dict[str, Any]] = []
    labels: np.ndarray | None = None
    start = 0
    for layer, aggregation in feature_spec.components:
        spikes, y, lengths = extracted[layer][split]
        feat = exp80._probe_features(spikes, lengths, bin_steps, aggregation).astype(np.float64)
        if labels is None:
            labels = y
        elif not np.array_equal(labels, y):
            raise RuntimeError("L1/L2 label ordering mismatch")
        stop = start + feat.shape[1]
        blocks.append({
            "layer": layer,
            "aggregation": aggregation,
            "start": start,
            "stop": stop,
            "feature_dim": int(feat.shape[1]),
        })
        parts.append(feat)
        start = stop
    if labels is None:
        raise RuntimeError("No feature components")
    return np.concatenate(parts, axis=1), labels, blocks


def _fit_probe_detailed(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    blocks: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    scaler = StandardScaler().fit(train_x)
    train_z = scaler.transform(train_x)
    val_z = scaler.transform(val_x)
    test_z = scaler.transform(test_x)

    best: tuple[float, float, LogisticRegression] | None = None
    for C in exp302.PROBE_C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=5000,
            solver="lbfgs",
            random_state=seed,
        ).fit(train_z, train_y)
        val_pred = classifier.predict(val_z)
        val_ba = float(balanced_accuracy_score(val_y, val_pred))
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError("No probe candidate selected")

    _, selected_C, classifier = best
    predictions = {
        "train": classifier.predict(train_z),
        "val": classifier.predict(val_z),
        "test": classifier.predict(test_z),
    }
    metrics = {
        "train": exp01._classification_metrics(train_y, predictions["train"]),
        "val": exp01._classification_metrics(val_y, predictions["val"]),
        "test": exp01._classification_metrics(test_y, predictions["test"]),
    }

    coef = np.asarray(classifier.coef_, dtype=np.float64)
    block_diagnostics = []
    norms = []
    for block in blocks:
        block_coef = coef[:, int(block["start"]):int(block["stop"])]
        fro = float(np.linalg.norm(block_coef))
        rms = float(np.sqrt(np.mean(np.square(block_coef)))) if block_coef.size else 0.0
        norms.append(fro)
        block_diagnostics.append({**block, "coef_fro_norm": fro, "coef_rms": rms})
    norm_total = float(sum(norms))
    for row, norm in zip(block_diagnostics, norms, strict=True):
        row["coef_norm_fraction"] = norm / norm_total if norm_total > 0 else 0.0

    payload = {
        "feature_dim": int(train_x.shape[1]),
        "probe_C": selected_C,
        "metrics": metrics,
        "generalization_gap_train_test_ba": float(
            metrics["train"]["balanced_accuracy"] - metrics["test"]["balanced_accuracy"]
        ),
        "blocks": block_diagnostics,
    }
    return payload, predictions


def _correctness_overlap(
    y: np.ndarray,
    pred_l1: np.ndarray,
    pred_l2: np.ndarray,
) -> dict[str, float]:
    c1 = pred_l1 == y
    c2 = pred_l2 == y
    n = max(len(y), 1)
    return {
        "both_correct": float(np.sum(c1 & c2) / n),
        "l1_only_correct": float(np.sum(c1 & ~c2) / n),
        "l2_only_correct": float(np.sum(~c1 & c2) / n),
        "both_wrong": float(np.sum(~c1 & ~c2) / n),
        "prediction_disagreement": float(np.mean(pred_l1 != pred_l2)),
        "oracle_union_accuracy": float(np.mean(c1 | c2)),
    }


def run_one(run: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    eval_path = config.results_dir / "evaluations" / f"{run.key}.json"
    if eval_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)
    model, checkpoint, source_eval = _load_source_model(run, data, config)
    loaders = exp73._raw_loaders(data, run.seed, config.batch_size, False)
    extracted = exp80._extract_layer_splits(model, loaders, device)

    probes: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    split_labels = {
        split: extracted["l1"][split][1]
        for split in ("train", "val", "test")
    }
    for feature in FEATURE_SPECS:
        built = {
            split: _build_features(extracted, feature, split, data.bin_steps)
            for split in ("train", "val", "test")
        }
        blocks = built["train"][2]
        probe, preds = _fit_probe_detailed(
            built["train"][0], built["train"][1],
            built["val"][0], built["val"][1],
            built["test"][0], built["test"][1],
            seed=exp3.dseed(run.seed, EXPERIMENT_ID, feature.name),
            blocks=blocks,
        )
        probes[feature.name] = probe
        predictions[feature.name] = preds

    overlap = {
        aggregation: {
            split: _correctness_overlap(
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
        "whole_fusion_gain_over_best_single": test_ba("l1_l2_whole") - max(test_ba("l1_whole"), test_ba("l2_whole")),
        "fixed250_fusion_gain_over_best_single": test_ba("l1_l2_fixed250") - max(test_ba("l1_fixed250"), test_ba("l2_fixed250")),
        "l1whole_l2fixed250_gain_over_l2fixed250": test_ba("l1whole_l2fixed250") - test_ba("l2_fixed250"),
        "l1whole_l2fixed250_gain_over_best_component": test_ba("l1whole_l2fixed250") - max(test_ba("l1_whole"), test_ba("l2_fixed250")),
        "reverse_mixed_gain_over_best_component": test_ba("l1fixed250_l2whole") - max(test_ba("l1_fixed250"), test_ba("l2_whole")),
    }

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "run": asdict(run),
        "source": {
            "experiment_id": SOURCE_EXPERIMENT_ID,
            "protocol_version": SOURCE_PROTOCOL_VERSION,
            "architecture": SOURCE_ARCHITECTURE,
            "checkpoint": str(_source_checkpoint_path(config.repo_root, run.seed).relative_to(config.repo_root)),
            "best_epoch": int(checkpoint["best_epoch"]),
            "source_linear_test_ba": float(source_eval["linear_metrics"]["test"]["balanced_accuracy"]),
            "frozen_backbone": True,
        },
        "feature_specs": {spec.name: [list(x) for x in spec.components] for spec in FEATURE_SPECS},
        "probes": probes,
        "correctness_overlap": overlap,
        "derived": derived,
    }
    _save_json(eval_path, payload)
    return payload


def _probe_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    seed = int(payload["run"]["seed"])
    for name, probe in payload["probes"].items():
        rows.append({
            "seed": seed,
            "feature": name,
            "feature_dim": int(probe["feature_dim"]),
            "probe_C": float(probe["probe_C"]),
            "train_ba": float(probe["metrics"]["train"]["balanced_accuracy"]),
            "val_ba": float(probe["metrics"]["val"]["balanced_accuracy"]),
            "test_ba": float(probe["metrics"]["test"]["balanced_accuracy"]),
            "train_test_gap": float(probe["generalization_gap_train_test_ba"]),
        })
    return rows


def _block_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    seed = int(payload["run"]["seed"])
    for name, probe in payload["probes"].items():
        for block_index, block in enumerate(probe["blocks"]):
            rows.append({
                "seed": seed,
                "feature": name,
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
    rows = []
    seed = int(payload["run"]["seed"])
    for aggregation, splits in payload["correctness_overlap"].items():
        for split, values in splits.items():
            rows.append({"seed": seed, "aggregation": aggregation, "split": split, **values})
    return rows


def finalize(config: Config) -> dict[str, Any]:
    payloads = []
    for run in run_specs():
        path = config.results_dir / "evaluations" / f"{run.key}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp8.0.1 evaluation: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))

    probe_runs = pd.DataFrame([row for payload in payloads for row in _probe_rows(payload)])
    probe_runs.to_csv(config.results_dir / "probe_runs.csv", index=False)
    metric_cols = ["test_ba", "val_ba", "train_ba", "train_test_gap", "feature_dim", "probe_C"]
    probe_summary = probe_runs.groupby("feature", sort=False)[metric_cols].agg(["mean", "std"]).reset_index()
    probe_summary.columns = [
        "_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col)
        for col in probe_summary.columns
    ]
    probe_summary.to_csv(config.results_dir / "probe_summary.csv", index=False)

    derived_runs = pd.DataFrame([
        {"seed": int(payload["run"]["seed"]), **payload["derived"]}
        for payload in payloads
    ])
    derived_runs.to_csv(config.results_dir / "fusion_gain_runs.csv", index=False)
    derived_summary = derived_runs.drop(columns=["seed"]).agg(["mean", "std"]).T.reset_index()
    derived_summary.columns = ["metric", "mean", "std"]
    derived_summary.to_csv(config.results_dir / "fusion_gain_summary.csv", index=False)

    overlap_runs = pd.DataFrame([row for payload in payloads for row in _overlap_rows(payload)])
    overlap_runs.to_csv(config.results_dir / "correctness_overlap_runs.csv", index=False)
    overlap_metrics = ["both_correct", "l1_only_correct", "l2_only_correct", "both_wrong", "prediction_disagreement", "oracle_union_accuracy"]
    overlap_summary = overlap_runs.groupby(["aggregation", "split"], sort=False)[overlap_metrics].agg(["mean", "std"]).reset_index()
    overlap_summary.columns = [
        "_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col)
        for col in overlap_summary.columns
    ]
    overlap_summary.to_csv(config.results_dir / "correctness_overlap_summary.csv", index=False)

    block_runs = pd.DataFrame([row for payload in payloads for row in _block_rows(payload)])
    block_runs.to_csv(config.results_dir / "coef_block_runs.csv", index=False)
    block_summary = block_runs.groupby(["feature", "block_index", "layer", "aggregation"], sort=False)[
        ["coef_fro_norm", "coef_rms", "coef_norm_fraction"]
    ].agg(["mean", "std"]).reset_index()
    block_summary.columns = [
        "_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col)
        for col in block_summary.columns
    ]
    block_summary.to_csv(config.results_dir / "coef_block_summary.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_protocol_version": SOURCE_PROTOCOL_VERSION,
        "source_architecture": SOURCE_ARCHITECTURE,
        "seeds": list(SEEDS),
        "parallel_runs": len(SEEDS),
        "backbone_training": "none; reuse and freeze Exp8.0 234x234 checkpoints",
        "feature_specs": {spec.name: [list(x) for x in spec.components] for spec in FEATURE_SPECS},
        "probe_protocol": "train-only StandardScaler + validation-selected LogisticRegression C",
        "primary_question": "Do frozen L1 and L2 spike representations carry complementary whole-count and/or ordered Fixed250 information?",
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp8.0.1 frozen L1/L2 fusion probe")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp80.exp72.BATCH_SIZE)
    sub = parser.add_subparsers(dest="command", required=True)
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
    )
    if args.command == "run":
        runs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(runs):
            raise IndexError(args.array_task_id)
        payload = run_one(runs[args.array_task_id], config, force=args.force)
        print(json.dumps({
            "seed": payload["run"]["seed"],
            "derived": payload["derived"],
        }, indent=2))
    else:
        print(json.dumps(finalize(config), indent=2))


if __name__ == "__main__":
    main()
