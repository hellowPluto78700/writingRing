from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_4_0_5_temporal_resolution_event_capacity as exp405
from scripts import experiment_4_0_6_local_mem_raw64 as exp406
from scripts import experiment_4_3_long_term_memory_validation as exp43


EXPERIMENT_ID = "experiment_4_4_1_stage2_memory_sweep"
PROTOCOL_VERSION = "stage2_tau_x_recurrence_v1"
LOCAL_CONDITION = "shift34"
MODEL_KINDS = ("ff", "rsnn")
TAU_MEM_MS = (125, 250, 500, 1000, 2000)
SEEDS = exp406.SEEDS
BATCH_SIZE = exp406.BATCH_SIZE
EPOCHS = exp406.EPOCHS
LR = exp406.LR
WEIGHT_DECAY = exp406.WEIGHT_DECAY
PROBE_MAX_ITER = exp406.PROBE_MAX_ITER

_MULTI_HO = next(item for item in exp405.VARIANTS if item[0] == "multi_ho")
VARIANT, HIDDEN_CAP, OUTPUT_CAP = _MULTI_HO

ModelKind = Literal["ff", "rsnn"]


@dataclass(frozen=True)
class RunSpec:
    model_kind: ModelKind
    tau_mem_ms: int
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.model_kind}__tau{self.tau_mem_ms}ms__"
            f"{LOCAL_CONDITION}__{VARIANT}__seed{self.seed}"
        )


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(model_kind, tau_mem_ms, seed)
        for model_kind in MODEL_KINDS
        for tau_mem_ms in TAU_MEM_MS
        for seed in SEEDS
    ]


def source_spec(seed: int) -> exp406.RunSpec:
    return exp406.RunSpec(
        condition=LOCAL_CONDITION,
        variant=VARIANT,
        hidden_cap=HIDDEN_CAP,
        output_cap=OUTPUT_CAP,
        seed=seed,
    )


def paired_seed(spec: RunSpec, role: str) -> int:
    # Pair all tau / FF-RSNN conditions inside a user split seed.
    return exp406.paired_seed(source_spec(spec.seed), role)


def stage2_beta(tau_mem_ms: float, dt_ms: float) -> float:
    if tau_mem_ms <= 0:
        raise ValueError("tau_mem_ms must be positive")
    return math.exp(-float(dt_ms) / float(tau_mem_ms))


class Stage2SweepDecoder(exp43.Stage2ValidationDecoder):
    """Exp4.3 decoder with configurable Stage-2 membrane time constant."""

    def __init__(
        self,
        model_kind: ModelKind,
        tau_mem_ms: int,
        n_classes: int,
        fs: float = 64.0,
    ) -> None:
        super().__init__(model_kind=model_kind, n_classes=n_classes, fs=fs)
        self.tau_mem_ms = int(tau_mem_ms)
        self.state_lif = exp401.MacroMultiSpikeLIF(
            beta=stage2_beta(self.tau_mem_ms, self.dt_ms),
            threshold=exp405.THRESHOLD,
            max_spikes_per_dt=HIDDEN_CAP,
        )


def prepare_data(repo_root: Path) -> exp405.TemporalData:
    return exp406.prepare_raw64_data(repo_root)


def make_loaders(
    data: exp405.TemporalData,
    spec: RunSpec,
    batch_size: int,
    train_shuffle: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    parts = {
        "train": (data.Xtr, data.ytr, data.vtr),
        "val": (data.Xva, data.yva, data.vva),
        "test": (data.Xte, data.yte, data.vte),
    }
    return {
        split: exp405.loader(
            X,
            y,
            valid,
            batch_size,
            train_shuffle if split == "train" else False,
            paired_seed(spec, f"{split}_loader"),
        )
        for split, (X, y, valid) in parts.items()
    }


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _scaled_linear_probe() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=PROBE_MAX_ITER,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_one(
    spec: RunSpec,
    data: exp405.TemporalData,
    config: Config,
    force: bool,
) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp405.exp40.base.seed_all(paired_seed(spec, "model_init"))
    model = Stage2SweepDecoder(
        model_kind=spec.model_kind,
        tau_mem_ms=spec.tau_mem_ms,
        n_classes=len(data.labels),
        fs=data.fs,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    train_loader = make_loaders(data, spec, config.batch_size, train_shuffle=True)["train"]
    val_loader = make_loaders(data, spec, config.batch_size, train_shuffle=False)["val"]

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for X, y, valid_steps in train_loader:
            X = X.to(device)
            y = y.to(device)
            valid_steps = valid_steps.to(device)
            optimizer.zero_grad(set_to_none=True)
            traj = model.forward_trajectory(X)
            logits = exp405.normalized_output_evidence(
                traj["output_spikes"], valid_steps, model.output_cap
            )
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            n = len(y)
            train_n += n
            train_loss_sum += float(loss.item()) * n
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = exp405.exp40.metrics(
            np.concatenate(train_true), np.concatenate(train_pred)
        )
        val_metrics = exp405.evaluate_model(model, val_loader, device)
        val_ba = float(val_metrics["valid_count"]["balanced_accuracy"])
        val_loss = float(val_metrics["valid_count_loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_n, 1),
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_loss": val_loss,
                "val_balanced_accuracy": val_ba,
            }
        )
        improved = val_ba > best_val_ba + 1e-12 or (
            abs(val_ba - best_val_ba) <= 1e-12 and val_loss < best_val_loss
        )
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_ba,
            "best_val_loss": best_val_loss,
            "model_state_dict": best_state,
            "channel_scale": data.channel_scale,
            "labels": data.labels,
            "split": data.split,
        },
        destination,
    )
    history_file = history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: exp405.TemporalData,
    config: Config,
) -> tuple[Stage2SweepDecoder, dict[str, object]]:
    device = torch.device(config.device)
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp4.4.1 checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint identity mismatch for {spec.key}")
    model = Stage2SweepDecoder(
        model_kind=spec.model_kind,
        tau_mem_ms=spec.tau_mem_ms,
        n_classes=len(data.labels),
        fs=data.fs,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def extract_features(
    model: Stage2SweepDecoder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> exp43.ConditionFeatures:
    return exp43.extract_condition_features(model, data_loader, device)


def evaluate_one(
    spec: RunSpec,
    data: exp405.TemporalData,
    config: Config,
    force: bool,
) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model, checkpoint = load_model(spec, data, config)
    loaders = make_loaders(data, spec, config.batch_size, train_shuffle=False)
    features = {
        split: extract_features(model, loader, device)
        for split, loader in loaders.items()
    }
    hidden_probe = _scaled_linear_probe()
    uend_probe = _scaled_linear_probe()
    hidden_probe.fit(features["train"].hidden_count, features["train"].labels)
    uend_probe.fit(features["train"].uend, features["train"].labels)

    metrics: dict[str, object] = {}
    for split, feature in features.items():
        output_metrics = exp405.exp40.metrics(
            feature.labels, feature.output_logits.argmax(axis=1)
        )
        hidden_metrics = exp405.exp40.metrics(
            feature.labels, hidden_probe.predict(feature.hidden_count)
        )
        uend_metrics = exp405.exp40.metrics(
            feature.labels, uend_probe.predict(feature.uend)
        )
        metrics[split] = {
            "output_whole_count": output_metrics,
            "hidden_whole_count_linear": hidden_metrics,
            "uend_linear": uend_metrics,
            "uend_minus_hidden_count_ba": float(
                uend_metrics["balanced_accuracy"] - hidden_metrics["balanced_accuracy"]
            ),
            "uend_minus_output_count_ba": float(
                uend_metrics["balanced_accuracy"] - output_metrics["balanced_accuracy"]
            ),
        }

    diagnostics = {
        split: exp405.evaluate_model(model, loader, device)
        for split, loader in loaders.items()
    }
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_balanced_accuracy": float(checkpoint["best_val_balanced_accuracy"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "architecture": {
            "input": "Raw64 30-channel scaled weighted events",
            "local": exp406.local_condition_definition(LOCAL_CONDITION, data.fs),
            "local_recurrence": False,
            "stage2_width": exp405.STATE_WIDTH,
            "stage2_tau_mem_ms": spec.tau_mem_ms,
            "stage2_beta": stage2_beta(spec.tau_mem_ms, data.dt_ms),
            "stage2_recurrence": spec.model_kind == "rsnn",
            "output_tau_mem_ms": exp405.OUTPUT_TAU_MEM_MS,
            "variant": VARIANT,
            "hidden_cap": HIDDEN_CAP,
            "output_cap": OUTPUT_CAP,
            "threshold": exp405.THRESHOLD,
            "dt_ms": data.dt_ms,
        },
        "probe_protocol": {
            "fit_source": "normal train-user representations only",
            "classifier": "train-only StandardScaler + balanced LogisticRegression(lbfgs)",
            "temporal_phase_access": False,
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "provenance": {
            "split_seed": int(exp405.exp40.base.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "training_objective": "valid normalized output WholeCount CE",
            "channel_scale_source": "training Fixed250 valid bins only, inherited from Exp4.0.5/4.0.6",
        },
    }
    _save_json(destination, payload)
    return payload


def run_one(
    spec: RunSpec,
    data: exp405.TemporalData,
    config: Config,
    force: bool,
) -> dict[str, object]:
    train_one(spec, data, config, force)
    return evaluate_one(spec, data, config, force)


def _metric_ba(metrics: dict[str, object], readout: str) -> float:
    return float(metrics[readout]["balanced_accuracy"])  # type: ignore[index]


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing required Exp4.4.1 artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for split, metrics in payload["metrics"].items():
            diagnostics = payload["diagnostics"][split]
            rows.append(
                {
                    "model_kind": spec.model_kind,
                    "tau_mem_ms": spec.tau_mem_ms,
                    "seed": spec.seed,
                    "split": split,
                    "output_whole_count_ba": _metric_ba(metrics, "output_whole_count"),
                    "hidden_whole_count_linear_ba": _metric_ba(metrics, "hidden_whole_count_linear"),
                    "uend_linear_ba": _metric_ba(metrics, "uend_linear"),
                    "uend_minus_hidden_count_ba": float(metrics["uend_minus_hidden_count_ba"]),
                    "uend_minus_output_count_ba": float(metrics["uend_minus_output_count_ba"]),
                    "state_events_per_neuron_second": float(
                        diagnostics["state_valid_stats"]["mean_events_per_neuron_second"]
                    ),
                    "output_events_per_neuron_second": float(
                        diagnostics["output_valid_stats"]["mean_events_per_neuron_second"]
                    ),
                    "state_tail_event_fraction": float(diagnostics["state_tail_event_fraction"]),
                    "output_tail_event_fraction": float(diagnostics["output_tail_event_fraction"]),
                }
            )
    runs = pd.DataFrame(rows)
    runs_file = root / "runs.csv"
    runs.to_csv(runs_file, index=False)

    test = runs[runs["split"] == "test"].copy()
    metric_columns = [
        "output_whole_count_ba",
        "hidden_whole_count_linear_ba",
        "uend_linear_ba",
        "uend_minus_hidden_count_ba",
        "uend_minus_output_count_ba",
        "state_events_per_neuron_second",
        "output_events_per_neuron_second",
        "state_tail_event_fraction",
        "output_tail_event_fraction",
    ]
    summary = (
        test.groupby(["model_kind", "tau_mem_ms"])[metric_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    summary_file = root / "summary.csv"
    summary.to_csv(summary_file, index=False)

    paired_rows: list[dict[str, object]] = []
    indexed = test.set_index(["model_kind", "tau_mem_ms", "seed"])
    readouts = {
        "output_whole_count": "output_whole_count_ba",
        "hidden_whole_count_linear": "hidden_whole_count_linear_ba",
        "uend_linear": "uend_linear_ba",
    }
    for tau_mem_ms in TAU_MEM_MS:
        for seed in SEEDS:
            for readout, column in readouts.items():
                rsnn = float(indexed.loc[("rsnn", tau_mem_ms, seed), column])
                ff = float(indexed.loc[("ff", tau_mem_ms, seed), column])
                paired_rows.append(
                    {
                        "tau_mem_ms": tau_mem_ms,
                        "seed": seed,
                        "readout": readout,
                        "delta_ba_rsnn_minus_ff": rsnn - ff,
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired_file = root / "paired_effects.csv"
    paired.to_csv(paired_file, index=False)
    paired_summary = (
        paired.groupby(["tau_mem_ms", "readout"])["delta_ba_rsnn_minus_ff"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    paired_summary_file = root / "paired_effects_summary.csv"
    paired_summary.to_csv(paired_summary_file, index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "With Local128 fixed, how do passive Stage-2 tau_mem and dense recurrence interact "
            "to determine gesture-level memory quality and spike-accessible classification?"
        ),
        "run_count": len(run_specs()),
        "seeds": list(SEEDS),
        "tau_mem_ms": list(TAU_MEM_MS),
        "model_kinds": list(MODEL_KINDS),
        "fixed": {
            "local_condition": LOCAL_CONDITION,
            "stage2_width": exp405.STATE_WIDTH,
            "variant": VARIANT,
            "hidden_cap": HIDDEN_CAP,
            "output_cap": OUTPUT_CAP,
            "threshold": exp405.THRESHOLD,
            "output_tau_mem_ms": exp405.OUTPUT_TAU_MEM_MS,
            "training_objective": "valid normalized output WholeCount CE",
        },
        "primary_architecture_metric": "Uend + train-only scaled Linear probe balanced accuracy",
        "secondary_metrics": [
            "Hidden WholeCount + train-only scaled Linear probe balanced accuracy",
            "Output WholeCount balanced accuracy",
            "Uend-HiddenCount gap",
            "Uend-OutputCount gap",
            "state/output firing rate and tail activity",
        ],
        "files": {
            "runs": runs_file.name,
            "summary": summary_file.name,
            "paired_effects": paired_file.name,
            "paired_effects_summary": paired_summary_file.name,
        },
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    return {
        "runs": runs_file,
        "summary": summary_file,
        "paired_effects": paired_file,
        "paired_effects_summary": paired_summary_file,
        "manifest": manifest_file,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-one")
    run_parser.add_argument("--array-task-id", type=int, required=True)
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--threads", type=int, default=1)
    run_parser.add_argument("--epochs", type=int, default=EPOCHS)
    run_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run_parser.add_argument("--force", action="store_true")

    eval_parser = subparsers.add_parser("eval-one")
    eval_parser.add_argument("--model-kind", choices=MODEL_KINDS, required=True)
    eval_parser.add_argument("--tau-mem-ms", type=int, choices=TAU_MEM_MS, required=True)
    eval_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--threads", type=int, default=1)
    eval_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    eval_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("finalize")
    return parser


def _config_from_args(args: argparse.Namespace, repo_root: Path) -> Config:
    return Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=getattr(args, "device", "cpu"),
        epochs=getattr(args, "epochs", EPOCHS),
        batch_size=getattr(args, "batch_size", BATCH_SIZE),
        threads=getattr(args, "threads", 1),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = exp405.exp40.find_repo_root()
    if args.command == "finalize":
        for name, path in finalize_experiment(repo_root).items():
            print(f"{name}: {path}")
        return

    data = prepare_data(repo_root)
    config = _config_from_args(args, repo_root)
    if args.command == "run-one":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(
                f"array task {args.array_task_id} outside [0, {len(specs) - 1}]"
            )
        spec = specs[args.array_task_id]
        payload = run_one(spec, data, config, args.force)
    else:
        spec = RunSpec(args.model_kind, args.tau_mem_ms, args.seed)
        payload = evaluate_one(spec, data, config, args.force)

    test = payload["metrics"]["test"]
    print(
        f"completed {spec.key}: "
        f"output_count_BA={test['output_whole_count']['balanced_accuracy']:.6f} "
        f"hidden_count_linear_BA={test['hidden_whole_count_linear']['balanced_accuracy']:.6f} "
        f"uend_linear_BA={test['uend_linear']['balanced_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
