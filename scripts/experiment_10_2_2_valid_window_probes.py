from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_3_5_hidden_state_information_loss as exp735
from scripts import experiment_10_0_airborne_motion_ablation as exp10
from scripts import experiment_10_2_1_l2_membrane_weighted as exp1021


EXPERIMENT_ID = "experiment_10_2_2_valid_window_probes"
PROTOCOL_VERSION = "exp10_2_1_checkpoints_valid_window_v1"

SOURCE_EXPERIMENT_ID = exp1021.EXPERIMENT_ID
SOURCE_PROTOCOL_VERSION = exp1021.PROTOCOL_VERSION

SUPPORT_VALID = "valid"
SUPPORT_WINDOW = "window"
SUPPORTS = (SUPPORT_VALID, SUPPORT_WINDOW)

LAYERS = ("l1", "l2")
ANALOG_STATES = ("syn_current", "pre_reset", "post_reset")
COMMUNICATION_STATE = "communication"
ANALOG_AGGREGATIONS = ("whole_mean", "fixed250_ordered_mean")
COMMUNICATION_AGGREGATIONS = (
    "whole_mean",
    "fixed250_ordered_mean",
    "whole_count",
    "fixed250_count",
)

C_GRID = tuple(float(v) for v in exp10.C_GRID)
PROBE_MAX_ITER = exp10.PROBE_MAX_ITER
MODEL_SEEDS = exp1021.MODEL_SEEDS
L2_MEM_SHIFTS = exp1021.L2_MEM_SHIFTS
CODINGS = exp1021.CODINGS
EXPECTED_CHECKPOINTS = exp1021.EXPECTED_RUNS
EXPECTED_PROBES_PER_CHECKPOINT = 40
SPLITS = exp1021.SPLITS


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    threads: int = 1
    batch_size: int = exp1021.BATCH_SIZE


def find_repo_root(start: Path | None = None) -> Path:
    return exp1021.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / EXPERIMENT_ID
        / PROTOCOL_VERSION
    )


def source_results_dir(repo_root: Path) -> Path:
    return exp1021.results_dir(repo_root)


def source_config(config: Config) -> exp1021.Config:
    return exp1021.Config(
        repo_root=config.repo_root,
        results_dir=source_results_dir(config.repo_root),
        device=config.device,
        threads=config.threads,
        batch_size=config.batch_size,
        max_epochs=exp1021.MAX_EPOCHS,
    )


def run_specs() -> list[exp1021.RunSpec]:
    return list(exp1021.run_specs())


def probe_names() -> tuple[str, ...]:
    names: list[str] = []
    for layer in LAYERS:
        for state in ANALOG_STATES:
            for support in SUPPORTS:
                for aggregation in ANALOG_AGGREGATIONS:
                    names.append(
                        f"{layer}__{state}__{support}__{aggregation}"
                    )
        for support in SUPPORTS:
            for aggregation in COMMUNICATION_AGGREGATIONS:
                names.append(
                    f"{layer}__{COMMUNICATION_STATE}__{support}__{aggregation}"
                )
    if len(names) != EXPECTED_PROBES_PER_CHECKPOINT:
        raise RuntimeError(
            f"Expected {EXPECTED_PROBES_PER_CHECKPOINT} probes, got {len(names)}"
        )
    if len(set(names)) != len(names):
        raise RuntimeError("Duplicate Exp10.2.2 probe names")
    return tuple(names)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _source_manifest(config: Config) -> dict[str, Any]:
    path = source_results_dir(config.repo_root) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp10.2.1 manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": SOURCE_EXPERIMENT_ID,
        "protocol_version": SOURCE_PROTOCOL_VERSION,
        "status": "PASS",
        "expected_run_count": EXPECTED_CHECKPOINTS,
        "run_count": EXPECTED_CHECKPOINTS,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Exp10.2.1 manifest mismatch for {key}: "
                f"expected {value!r}, got {manifest.get(key)!r}"
            )
    return manifest


def prepare_all(config: Config) -> dict[str, Any]:
    source_manifest = _source_manifest(config)
    source_cfg = source_config(config)

    missing: list[str] = []
    for spec in run_specs():
        checkpoint = exp1021._run_artifacts(source_cfg, spec)["checkpoint"]
        if not checkpoint.exists():
            missing.append(str(checkpoint))
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} Exp10.2.1 checkpoints:\n"
            + "\n".join(missing[:40])
        )

    names = probe_names()
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_protocol_version": SOURCE_PROTOCOL_VERSION,
        "source_manifest": source_manifest,
        "evaluation_only": True,
        "snn_training": False,
        "checkpoint_count": EXPECTED_CHECKPOINTS,
        "probe_count_per_checkpoint": len(names),
        "expected_probe_rows": EXPECTED_CHECKPOINTS * len(names),
        "supports": list(SUPPORTS),
        "layers": list(LAYERS),
        "analog_states": list(ANALOG_STATES),
        "communication_state": COMMUNICATION_STATE,
        "analog_aggregations": list(ANALOG_AGGREGATIONS),
        "communication_aggregations": list(COMMUNICATION_AGGREGATIONS),
        "probe_selection_metric": "validation balanced accuracy",
        "reported_metrics": ["accuracy", "balanced_accuracy", "macro_f1"],
        "primary_contrast": "window_minus_valid",
        "statistical_scope": (
            "one locked cross-user split; seeds 11/23/37 are paired "
            "optimization replicates inherited from Exp10.2.1"
        ),
    }
    _save_json(config.results_dir / "audit.json", audit)
    return audit


def _checkpoint_path(
    config: Config,
    spec: exp1021.RunSpec,
) -> Path:
    return exp1021._run_artifacts(source_config(config), spec)["checkpoint"]


def _load_checkpoint_model(
    config: Config,
    spec: exp1021.RunSpec,
    data: exp3.Data,
    split_hashes: Mapping[str, str],
) -> tuple[exp1021.Exp1021Net, dict[str, Any], Path]:
    path = _checkpoint_path(config, spec)
    if not path.exists():
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise RuntimeError(f"{spec.key}: source experiment mismatch")
    if checkpoint.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise RuntimeError(f"{spec.key}: source protocol mismatch")
    if checkpoint.get("spec") != asdict(spec):
        raise RuntimeError(f"{spec.key}: source spec mismatch")
    if checkpoint.get("split_sample_hashes") != dict(split_hashes):
        raise RuntimeError(f"{spec.key}: source split geometry mismatch")

    device = torch.device(config.device)
    model = exp1021.Exp1021Net(spec, len(data.labels), data.fs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint, path


def _support_lengths(
    lengths: torch.Tensor,
    steps: int,
    support: str,
) -> torch.Tensor:
    if support == SUPPORT_VALID:
        return lengths
    if support == SUPPORT_WINDOW:
        return torch.full_like(lengths, int(steps))
    raise ValueError(support)


def _aggregate(
    values: torch.Tensor,
    lengths: torch.Tensor,
    support: str,
    aggregation: str,
    bin_steps: int,
) -> torch.Tensor:
    support_lengths = _support_lengths(lengths, values.shape[1], support)
    if aggregation in ANALOG_AGGREGATIONS:
        return exp735._aggregate_tensor(
            values,
            support_lengths,
            aggregation,
            bin_steps,
        )
    if aggregation == "whole_count":
        return exp10._masked_sum(values, support_lengths)
    if aggregation == "fixed250_count":
        return (
            exp3.fixed_counts(values, support_lengths, bin_steps)
            .flatten(start_dim=1)
        )
    raise ValueError(aggregation)


def _descriptor(name: str) -> tuple[str, str, str, str]:
    layer, state, support, aggregation = name.split("__", 3)
    return layer, state, support, aggregation


def _collect_probe_features(
    model: exp1021.Exp1021Net,
    data: exp3.Data,
    spec: exp1021.RunSpec,
    config: Config,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
]:
    loaders = exp1021.exp73._raw_loaders(
        data,
        spec.seed,
        config.batch_size,
        False,
    )
    device = torch.device(config.device)
    names = probe_names()
    feature_parts: dict[str, dict[str, list[np.ndarray]]] = {
        split: {name: [] for name in names}
        for split in SPLITS
    }
    label_parts: dict[str, list[np.ndarray]] = {
        split: [] for split in SPLITS
    }

    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            for X, y, lengths in loaders[split]:
                Xd = X.to(device=device, dtype=torch.float32)
                ld = lengths.to(device=device, dtype=torch.long)
                hidden = model.forward_trajectory(Xd)["hidden"]

                for layer in LAYERS:
                    for state in ANALOG_STATES:
                        values = hidden[layer][state]
                        for support in SUPPORTS:
                            for aggregation in ANALOG_AGGREGATIONS:
                                name = (
                                    f"{layer}__{state}__{support}__{aggregation}"
                                )
                                feature_parts[split][name].append(
                                    _aggregate(
                                        values,
                                        ld,
                                        support,
                                        aggregation,
                                        data.bin_steps,
                                    )
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False)
                                )

                    communication = hidden[layer]["spike"]
                    for support in SUPPORTS:
                        for aggregation in COMMUNICATION_AGGREGATIONS:
                            name = (
                                f"{layer}__{COMMUNICATION_STATE}__"
                                f"{support}__{aggregation}"
                            )
                            feature_parts[split][name].append(
                                _aggregate(
                                    communication,
                                    ld,
                                    support,
                                    aggregation,
                                    data.bin_steps,
                                )
                                .cpu()
                                .numpy()
                                .astype(np.float32, copy=False)
                            )

                label_parts[split].append(
                    y.numpy().astype(np.int64, copy=False)
                )

    features = {
        split: {
            name: np.concatenate(parts, axis=0)
            for name, parts in feature_parts[split].items()
        }
        for split in SPLITS
    }
    labels = {
        split: np.concatenate(label_parts[split], axis=0)
        for split in SPLITS
    }
    return features, labels


def _probe_seed(spec: exp1021.RunSpec, probe_name: str) -> int:
    return int(
        exp3.dseed(
            spec.seed,
            EXPERIMENT_ID,
            "probe",
            spec.coding,
            spec.l2_mem_shift,
            probe_name,
        )
    )


def _fit_probes(
    spec: exp1021.RunSpec,
    features: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for name in probe_names():
        train_x = features["train"][name].astype(np.float64, copy=False)
        val_x = features["val"][name].astype(np.float64, copy=False)
        test_x = features["test"][name].astype(np.float64, copy=False)

        scaler = StandardScaler().fit(train_x)
        transformed = {
            "train": scaler.transform(train_x),
            "val": scaler.transform(val_x),
            "test": scaler.transform(test_x),
        }

        random_state = _probe_seed(spec, name)
        best: tuple[float, float, LogisticRegression] | None = None
        candidates: list[dict[str, float]] = []

        for C in C_GRID:
            classifier = LogisticRegression(
                C=C,
                max_iter=PROBE_MAX_ITER,
                solver="lbfgs",
                random_state=random_state,
                fit_intercept=True,
            ).fit(transformed["train"], labels["train"])
            val_pred = classifier.predict(transformed["val"])
            val_metrics = exp10._classification_metrics(
                labels["val"], val_pred
            )
            candidates.append(
                {
                    "C": float(C),
                    "val_balanced_accuracy": float(
                        val_metrics["balanced_accuracy"]
                    ),
                    "val_macro_f1": float(val_metrics["macro_f1"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                }
            )
            val_ba = float(val_metrics["balanced_accuracy"])
            if best is None or val_ba > best[0] + 1e-12:
                best = (val_ba, float(C), classifier)

        if best is None:
            raise RuntimeError(
                f"No probe candidate selected for {spec.key}/{name}"
            )

        selected_val_ba, selected_C, classifier = best
        predictions = {
            split: classifier.predict(transformed[split])
            for split in SPLITS
        }
        metrics = {
            split: exp10._classification_metrics(
                labels[split], predictions[split]
            )
            for split in SPLITS
        }

        layer, state, support, aggregation = _descriptor(name)
        row: dict[str, Any] = {
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "source_protocol_version": SOURCE_PROTOCOL_VERSION,
            "coding": spec.coding,
            "l1_mem_shift": exp1021.L1_MEM_SHIFT,
            "l2_mem_shift": spec.l2_mem_shift,
            "seed": spec.seed,
            "probe": name,
            "layer": layer,
            "state": state,
            "support": support,
            "aggregation": aggregation,
            "feature_dim": int(train_x.shape[1]),
            "selected_C": selected_C,
            "selected_val_balanced_accuracy": selected_val_ba,
            "probe_seed": random_state,
            "candidate_validation_json": json.dumps(
                candidates,
                sort_keys=True,
            ),
        }
        for split in SPLITS:
            for metric, value in metrics[split].items():
                row[f"{split}_{metric}"] = float(value)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_PROBES_PER_CHECKPOINT:
        raise RuntimeError(
            f"{spec.key}: expected {EXPECTED_PROBES_PER_CHECKPOINT} probe rows, "
            f"got {len(frame)}"
        )
    return frame


def _evaluation_artifacts(
    config: Config,
    spec: exp1021.RunSpec,
) -> dict[str, Path]:
    return {
        "probes": (
            config.results_dir
            / "probe_evaluations"
            / f"{spec.key}.csv"
        ),
        "evaluation": (
            config.results_dir
            / "evaluations"
            / f"{spec.key}.json"
        ),
    }


def evaluate_one(
    spec: exp1021.RunSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    artifacts = _evaluation_artifacts(config, spec)
    if (
        not force
        and artifacts["probes"].exists()
        and artifacts["evaluation"].exists()
    ):
        return json.loads(
            artifacts["evaluation"].read_text(encoding="utf-8")
        )

    _source_manifest(config)
    data, frames, _ = exp1021._prepare_data(source_config(config))
    split_hashes = exp1021._split_hashes(frames)

    torch.set_num_threads(config.threads)
    model, checkpoint, checkpoint_path = _load_checkpoint_model(
        config,
        spec,
        data,
        split_hashes,
    )
    features, labels = _collect_probe_features(
        model,
        data,
        spec,
        config,
    )
    probes = _fit_probes(spec, features, labels)

    artifacts["probes"].parent.mkdir(parents=True, exist_ok=True)
    probes.to_csv(artifacts["probes"], index=False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_only": True,
        "snn_training": False,
        "spec": asdict(spec),
        "source": {
            "experiment_id": checkpoint["experiment_id"],
            "protocol_version": checkpoint["protocol_version"],
            "checkpoint": str(
                checkpoint_path.relative_to(config.repo_root)
            ),
            "best_epoch": int(checkpoint["best_epoch"]),
            "stopped_epoch": int(checkpoint["stopped_epoch"]),
            "split_sample_hashes": split_hashes,
        },
        "probe_rows": int(len(probes)),
        "supports": list(SUPPORTS),
        "reported_metrics": [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ],
        "selection_metric": "validation balanced accuracy",
    }
    _save_json(artifacts["evaluation"], payload)
    return payload


def _support_contrasts(probes: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "coding",
        "l1_mem_shift",
        "l2_mem_shift",
        "seed",
        "layer",
        "state",
        "aggregation",
    ]
    metric_cols = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
    ]

    valid = probes[probes.support == SUPPORT_VALID].copy()
    window = probes[probes.support == SUPPORT_WINDOW].copy()
    merged = window.merge(
        valid,
        on=id_cols,
        how="inner",
        suffixes=("_window", "_valid"),
        validate="one_to_one",
    )

    rows: list[dict[str, Any]] = []
    for _, source in merged.iterrows():
        row: dict[str, Any] = {
            col: source[col] for col in id_cols
        }
        row["contrast"] = "window_minus_valid"
        row["feature_dim_valid"] = int(source["feature_dim_valid"])
        row["feature_dim_window"] = int(source["feature_dim_window"])
        row["selected_C_valid"] = float(source["selected_C_valid"])
        row["selected_C_window"] = float(source["selected_C_window"])
        for metric in metric_cols:
            row[f"{metric}_valid"] = float(source[f"{metric}_valid"])
            row[f"{metric}_window"] = float(source[f"{metric}_window"])
            row[f"delta_{metric}"] = float(
                source[f"{metric}_window"]
                - source[f"{metric}_valid"]
            )
        rows.append(row)

    expected = EXPECTED_CHECKPOINTS * (
        EXPECTED_PROBES_PER_CHECKPOINT // len(SUPPORTS)
    )
    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} support contrasts, got {len(rows)}"
        )
    return pd.DataFrame(rows)


def finalize(config: Config) -> dict[str, Any]:
    _source_manifest(config)

    probe_frames: list[pd.DataFrame] = []
    evaluations: list[dict[str, Any]] = []
    missing: list[str] = []

    for spec in run_specs():
        artifacts = _evaluation_artifacts(config, spec)
        for name, path in artifacts.items():
            if not path.exists():
                missing.append(f"{spec.key}:{name}:{path}")
        if artifacts["probes"].exists():
            probe_frames.append(pd.read_csv(artifacts["probes"]))
        if artifacts["evaluation"].exists():
            evaluations.append(
                json.loads(
                    artifacts["evaluation"].read_text(encoding="utf-8")
                )
            )

    if missing:
        raise FileNotFoundError(
            f"Exp10.2.2 incomplete; missing {len(missing)} artifacts:\n"
            + "\n".join(missing[:50])
        )
    if len(probe_frames) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(
            f"Expected {EXPECTED_CHECKPOINTS} probe files, "
            f"got {len(probe_frames)}"
        )

    probes = pd.concat(probe_frames, ignore_index=True)
    expected_rows = EXPECTED_CHECKPOINTS * EXPECTED_PROBES_PER_CHECKPOINT
    if len(probes) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} total probe rows, got {len(probes)}"
        )
    probes.to_csv(config.results_dir / "probe_runs.csv", index=False)

    summary_metrics = [
        "train_accuracy",
        "train_balanced_accuracy",
        "train_macro_f1",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
    ]
    probe_summary = (
        probes.groupby(
            [
                "coding",
                "l2_mem_shift",
                "layer",
                "state",
                "support",
                "aggregation",
            ],
            sort=True,
        )[summary_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    probe_summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in probe_summary.columns
    ]
    probe_summary.to_csv(
        config.results_dir / "probe_summary.csv",
        index=False,
    )

    contrasts = _support_contrasts(probes)
    contrasts.to_csv(
        config.results_dir / "support_contrasts.csv",
        index=False,
    )
    contrast_metrics = [
        c for c in contrasts.columns if c.startswith("delta_")
    ]
    contrast_summary = (
        contrasts.groupby(
            [
                "coding",
                "l2_mem_shift",
                "layer",
                "state",
                "aggregation",
            ],
            sort=True,
        )[contrast_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    contrast_summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in contrast_summary.columns
    ]
    contrast_summary.to_csv(
        config.results_dir / "support_contrast_summary.csv",
        index=False,
    )

    key = contrasts[
        (
            (contrasts.state == "pre_reset")
            & (contrasts.aggregation == "fixed250_ordered_mean")
        )
        |
        (
            (contrasts.state == COMMUNICATION_STATE)
            & (contrasts.aggregation == "fixed250_count")
        )
    ].copy()
    key.to_csv(
        config.results_dir / "key_probe_contrasts.csv",
        index=False,
    )
    key_summary = (
        key.groupby(
            [
                "coding",
                "l2_mem_shift",
                "layer",
                "state",
                "aggregation",
            ],
            sort=True,
        )[contrast_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    key_summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in key_summary.columns
    ]
    key_summary.to_csv(
        config.results_dir / "key_probe_summary.csv",
        index=False,
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS",
        "evaluation_only": True,
        "snn_training": False,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_protocol_version": SOURCE_PROTOCOL_VERSION,
        "checkpoint_count": EXPECTED_CHECKPOINTS,
        "probe_rows": int(len(probes)),
        "support_contrast_rows": int(len(contrasts)),
        "supports": list(SUPPORTS),
        "selection_metric": "validation balanced accuracy",
        "reported_metrics": [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ],
        "primary_outputs": [
            "probe_runs.csv",
            "probe_summary.csv",
            "support_contrasts.csv",
            "support_contrast_summary.csv",
            "key_probe_contrasts.csv",
            "key_probe_summary.csv",
        ],
        "statistical_scope": (
            "one locked cross-user split; seeds are paired optimization "
            "replicates inherited from Exp10.2.1"
        ),
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _resolve_config(args: argparse.Namespace) -> Config:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else find_repo_root()
    )
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
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exp10.2.2 evaluation-only valid-length vs whole-window "
            "hidden-state probes over Exp10.2.1 checkpoints"
        )
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp1021.BATCH_SIZE)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("list-runs")

    evaluate = sub.add_parser("evaluate-one")
    evaluate.add_argument("--array-task-id", type=int, required=True)
    evaluate.add_argument("--force", action="store_true")

    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _resolve_config(args)

    if args.command == "prepare":
        print(json.dumps(prepare_all(config), indent=2, sort_keys=True))
        return

    if args.command == "list-runs":
        for index, spec in enumerate(run_specs()):
            print(index, spec.key)
        return

    if args.command == "evaluate-one":
        specs = run_specs()
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        spec = specs[args.array_task_id]
        payload = evaluate_one(
            spec,
            config,
            force=args.force,
        )
        print(
            json.dumps(
                {
                    "key": spec.key,
                    "probe_rows": payload["probe_rows"],
                    "source_checkpoint": payload["source"]["checkpoint"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "finalize":
        print(json.dumps(finalize(config), indent=2, sort_keys=True))
        return

    raise ValueError(args.command)


if __name__ == "__main__":
    main()
