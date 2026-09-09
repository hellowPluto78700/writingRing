from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objective_comparison as exp3
from scripts import experiment_5_0_local_evidence_objectives as exp50


EXPERIMENT_ID = "experiment_0_1_1_local234_wholecount"
PROTOCOL_VERSION = "local234_wholecount_v1"
ARCHITECTURE = "local_234x2"
HIDDEN_SHIFTS: tuple[tuple[int, ...], ...] = ((2, 3, 4), (2, 3, 4))
OBJECTIVES = ("whole_count_ce", "timestep_ce")
VARIANTS: tuple[tuple[str, int], ...] = (("binary", 1), ("multi_h", 31))
SEEDS = exp01.SEEDS
OUTPUT_CAP = 1
HIDDEN_WIDTH = 128

BATCH_SIZE = exp01.BATCH_SIZE
EPOCHS = exp01.EPOCHS
LR = exp01.LR
WEIGHT_DECAY = exp01.WEIGHT_DECAY


@dataclass(frozen=True)
class RunSpec:
    objective: str
    variant: str
    hidden_cap: int
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{ARCHITECTURE}__{self.objective}__{self.variant}__"
            f"hcap{self.hidden_cap}__ocap{OUTPUT_CAP}__seed{self.seed}"
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
        RunSpec(objective, variant, hidden_cap, seed)
        for objective in OBJECTIVES
        for variant, hidden_cap in VARIANTS
        for seed in SEEDS
    ]


def parent_spec(spec: RunSpec) -> exp01.RunSpec:
    """Identity adapter used only to inherit Exp0.1's paired seed namespace."""
    return exp01.RunSpec(
        family=exp01.DIRECT_FAMILY,
        architecture=ARCHITECTURE,
        objective=spec.objective,
        variant=spec.variant,
        hidden_cap=spec.hidden_cap,
        seed=spec.seed,
    )


def paired_seed(spec: RunSpec, role: str) -> int:
    return exp01.paired_seed(parent_spec(spec), role)


def prepare_data(repo_root: Path) -> exp3.Data:
    return exp01.prepare_data(repo_root)


def make_loaders(
    data: exp3.Data,
    spec: RunSpec,
    batch_size: int,
    train_shuffle: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    partitions = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    return {
        split: exp3.loader(
            X,
            y,
            lengths,
            batch_size,
            train_shuffle if split == "train" else False,
            paired_seed(spec, f"{split}_loader"),
        )
        for split, (X, y, lengths) in partitions.items()
    }


def new_model(spec: RunSpec, data: exp3.Data) -> exp01.MultiTauHierarchySNN:
    return exp01.MultiTauHierarchySNN(
        layer_shifts=HIDDEN_SHIFTS,
        n_classes=len(data.labels),
        fs=data.fs,
        hidden_cap=spec.hidden_cap,
        output_spiking=True,
    )


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _spec_matches(value: object, spec: RunSpec) -> bool:
    return isinstance(value, dict) and value == spec.__dict__


def checkpoint_complete(root: Path, spec: RunSpec) -> bool:
    path = checkpoint_path(root, spec)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("experiment_id") == EXPERIMENT_ID
        and payload.get("protocol_version") == PROTOCOL_VERSION
        and _spec_matches(payload.get("spec"), spec)
        and isinstance(payload.get("best_epoch"), int)
        and isinstance(payload.get("model_state_dict"), dict)
        and bool(payload.get("model_state_dict"))
    )


def history_complete(root: Path, spec: RunSpec) -> bool:
    path = history_path(root, spec)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    required = {"epoch", "train_objective_loss", "val_output_whole_count_ba", "val_output_whole_count_loss"}
    return bool(not frame.empty and required.issubset(frame.columns))


def _metrics_complete(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    for split in ("train", "val", "test"):
        metrics = value.get(split)
        if not isinstance(metrics, dict):
            return False
        if not {"accuracy", "balanced_accuracy", "macro_f1", "loss"}.issubset(metrics):
            return False
    return True


def evaluation_complete(root: Path, spec: RunSpec) -> bool:
    path = evaluation_path(root, spec)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("experiment_id") == EXPERIMENT_ID
        and payload.get("protocol_version") == PROTOCOL_VERSION
        and _spec_matches(payload.get("spec"), spec)
        and payload.get("primary_readout") == "output_whole_count"
        and _metrics_complete(payload.get("metrics"))
    )


def run_complete(root: Path, spec: RunSpec) -> bool:
    return (
        checkpoint_complete(root, spec)
        and history_complete(root, spec)
        and evaluation_complete(root, spec)
    )


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_deployment(
    model: exp01.MultiTauHierarchySNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for X, y, lengths in data_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            output_spikes = trajectory["output_spikes"]
            if not isinstance(output_spikes, torch.Tensor):
                raise TypeError("Expected output_spikes tensor")
            logits = exp50.deployment_logits(output_spikes, lengths, OUTPUT_CAP)
            ce_logits = exp50.deployment_ce_logits(output_spikes, lengths, OUTPUT_CAP)
            loss = F.cross_entropy(ce_logits, y)
            n = len(y)
            n_total += n
            loss_sum += float(loss.item()) * n
            labels.append(y.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    metrics = _classification_metrics(np.concatenate(labels), np.concatenate(predictions))
    metrics["loss"] = loss_sum / max(n_total, 1)
    return metrics


def architecture_manifest(spec: RunSpec, fs: float, n_classes: int) -> dict[str, object]:
    layers: list[dict[str, object]] = []
    for index, shifts in enumerate(HIDDEN_SHIFTS, start=1):
        layers.append(
            {
                "layer": index,
                "width": HIDDEN_WIDTH,
                "shifts": list(shifts),
                "group_neurons": list(exp50._group_counts(HIDDEN_WIDTH, shifts)),
                "tau_syn_ms": [exp3.tau_ms(shift, fs) for shift in shifts],
                "event_cap": spec.hidden_cap,
            }
        )
    return {
        "input_channels": exp3.EVENT_CHANNELS,
        "sampling_rate_hz": float(fs),
        "hidden_layers": layers,
        "tau_mem_ms": exp01.TAU_MEM_MS,
        "threshold": exp01.THRESHOLD,
        "output": {
            "type": "spiking_class_neurons",
            "neurons": int(n_classes),
            "event_cap": OUTPUT_CAP,
            "readout": "valid whole count",
        },
    }


def train_one(spec: RunSpec, data: exp3.Data, config: Config, force: bool) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if not force and checkpoint_complete(config.results_dir, spec) and history_complete(config.results_dir, spec):
        print(f"train-skip {spec.key}: complete checkpoint + history already exist")
        return destination

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp3.seed_all(paired_seed(spec, "model_init"))
    model = new_model(spec, data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loaders(data, spec, config.batch_size, train_shuffle=True)["train"]
    val_loader = make_loaders(data, spec, config.batch_size, train_shuffle=False)["val"]

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        objective_sum = 0.0
        n_total = 0
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            output_spikes = trajectory["output_spikes"]
            if not isinstance(output_spikes, torch.Tensor):
                raise TypeError("Expected output_spikes tensor")
            loss = exp50.objective_loss(
                output_spikes,
                lengths,
                y,
                spec.objective,
                OUTPUT_CAP,
                data.fs,
            )
            loss.backward()
            optimizer.step()
            n_total += len(y)
            objective_sum += float(loss.item()) * len(y)

        val_metrics = evaluate_deployment(model, val_loader, device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_objective_loss": objective_sum / max(n_total, 1),
                "val_output_whole_count_ba": val_ba,
                "val_output_whole_count_loss": val_loss,
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
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
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
            "best_val_output_whole_count_ba": best_val_ba,
            "best_val_output_whole_count_loss": best_val_loss,
            "model_state_dict": best_state,
            "labels": data.labels,
            "split": data.split,
            "architecture": architecture_manifest(spec, data.fs, len(data.labels)),
            "paired_seed_namespace": "exp0_1_general_comparison",
        },
        destination,
    )
    history_file = history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
) -> tuple[exp01.MultiTauHierarchySNN, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not checkpoint_complete(config.results_dir, spec):
        raise FileNotFoundError(f"Missing or incomplete Exp0.1.1 checkpoint: {path}")
    device = torch.device(config.device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = new_model(spec, data).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _last_hidden_event_rate(
    model: exp01.MultiTauHierarchySNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    fs: float,
) -> float:
    total_events = 0.0
    total_steps = 0.0
    model.eval()
    with torch.no_grad():
        for X, _, lengths in data_loader:
            X = X.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            hidden = trajectory["hidden_spikes"]
            if not isinstance(hidden, tuple):
                raise TypeError("Expected hidden_spikes tuple")
            last = hidden[-1]
            valid = exp50.valid_mask(lengths, last.shape[1]).to(last.dtype).unsqueeze(-1)
            total_events += float((last * valid).sum().item())
            total_steps += float(lengths.sum().item())
    return total_events * float(fs) / max(total_steps * HIDDEN_WIDTH, 1.0)


def evaluate_one(spec: RunSpec, data: exp3.Data, config: Config, force: bool) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if not force and evaluation_complete(config.results_dir, spec):
        print(f"eval-skip {spec.key}: complete evaluation already exists")
        return json.loads(destination.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model, checkpoint = load_model(spec, data, config)
    loaders = make_loaders(data, spec, config.batch_size, train_shuffle=False)
    metrics = {
        split: evaluate_deployment(model, loader, device)
        for split, loader in loaders.items()
    }
    event_rates = {
        split: _last_hidden_event_rate(model, loader, device, data.fs)
        for split, loader in loaders.items()
    }
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "architecture": architecture_manifest(spec, data.fs, len(data.labels)),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_output_whole_count_ba": float(checkpoint["best_val_output_whole_count_ba"]),
        "best_val_output_whole_count_loss": float(checkpoint["best_val_output_whole_count_loss"]),
        "primary_readout": "output_whole_count",
        "metrics": metrics,
        "last_hidden_events_per_neuron_second": event_rates,
        "provenance": {
            "split_seed": int(exp3.SPLIT_SEED),
            "model_seed": spec.seed,
            "paired_seed_namespace": "exp0_1_general_comparison",
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "input_contract": "Exp0.1 Raw64 30-channel unsigned weighted events; no pre-SNN pooling or scaling",
            "checkpoint_selection": "maximum validation Output WholeCount BA; tie-break minimum validation normalized WholeCount CE",
            "readout": "valid Output WholeCount for both training objectives",
        },
    }
    _save_json(destination, payload)
    return payload


def run_one(spec: RunSpec, data: exp3.Data, config: Config, force: bool) -> dict[str, object]:
    if not force and run_complete(config.results_dir, spec):
        print(f"run-skip {spec.key}: checkpoint, history, and evaluation are complete")
        return json.loads(evaluation_path(config.results_dir, spec).read_text(encoding="utf-8"))
    train_one(spec, data, config, force)
    return evaluate_one(spec, data, config, force)


def _row(spec: RunSpec, payload: dict[str, object]) -> dict[str, object]:
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")
    return {
        "architecture": ARCHITECTURE,
        "objective": spec.objective,
        "variant": spec.variant,
        "hidden_cap": spec.hidden_cap,
        "output_cap": OUTPUT_CAP,
        "seed": spec.seed,
        "best_epoch": int(payload["best_epoch"]),
        "train_ba": float(metrics["train"]["balanced_accuracy"]),  # type: ignore[index]
        "val_ba": float(metrics["val"]["balanced_accuracy"]),  # type: ignore[index]
        "test_ba": float(metrics["test"]["balanced_accuracy"]),  # type: ignore[index]
        "test_accuracy": float(metrics["test"]["accuracy"]),  # type: ignore[index]
        "test_macro_f1": float(metrics["test"]["macro_f1"]),  # type: ignore[index]
        "test_last_hidden_events_per_neuron_second": float(
            payload["last_hidden_events_per_neuron_second"]["test"]  # type: ignore[index]
        ),
    }


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    for spec in run_specs():
        if not run_complete(root, spec):
            raise FileNotFoundError(f"Incomplete required Exp0.1.1 run: {spec.key}")
        payload = json.loads(evaluation_path(root, spec).read_text(encoding="utf-8"))
        rows.append(_row(spec, payload))

    runs = pd.DataFrame(rows)
    runs_file = root / "runs.csv"
    runs_file.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(runs_file, index=False)

    summary = (
        runs.groupby(["architecture", "objective", "variant"])[
            ["test_ba", "test_accuracy", "test_macro_f1", "test_last_hidden_events_per_neuron_second"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary_file = root / "summary.csv"
    summary.to_csv(summary_file, index=False)

    indexed = runs.set_index(["objective", "variant", "seed"])
    objective_rows: list[dict[str, object]] = []
    for variant, _ in VARIANTS:
        for seed in SEEDS:
            whole = float(indexed.loc[("whole_count_ce", variant, seed), "test_ba"])
            timestep = float(indexed.loc[("timestep_ce", variant, seed), "test_ba"])
            objective_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "whole_count_ce_test_ba": whole,
                    "timestep_ce_test_ba": timestep,
                    "delta_test_ba_whole_minus_timestep": whole - timestep,
                }
            )
    objective_effects = pd.DataFrame(objective_rows)
    objective_file = root / "paired_objective_effects.csv"
    objective_effects.to_csv(objective_file, index=False)

    capacity_rows: list[dict[str, object]] = []
    for objective in OBJECTIVES:
        for seed in SEEDS:
            binary = float(indexed.loc[(objective, "binary", seed), "test_ba"])
            multi_h = float(indexed.loc[(objective, "multi_h", seed), "test_ba"])
            capacity_rows.append(
                {
                    "objective": objective,
                    "seed": seed,
                    "binary_test_ba": binary,
                    "multi_h_test_ba": multi_h,
                    "delta_test_ba_multi_h_minus_binary": multi_h - binary,
                }
            )
    capacity_effects = pd.DataFrame(capacity_rows)
    capacity_file = root / "paired_capacity_effects.csv"
    capacity_effects.to_csv(capacity_file, index=False)

    parent_comparison = exp01.results_dir(repo_root) / "comparison_summary.csv"
    if not parent_comparison.exists():
        raise FileNotFoundError(f"Missing finalized Exp0.1 comparison: {parent_comparison}")
    combined = pd.read_csv(parent_comparison)
    additions: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        additions.append(
            {
                "system": f"direct_snn:{ARCHITECTURE}:{row.objective}:{row.variant}",
                "family": "direct_snn",
                "architecture": ARCHITECTURE,
                "objective": row.objective,
                "variant": row.variant,
                "mean_test_ba": row.test_ba_mean,
                "sd_test_ba": row.test_ba_std,
                "n": row.test_ba_count,
            }
        )
    combined = pd.concat([combined, pd.DataFrame(additions)], ignore_index=True)
    comparison_file = root / "comparison_with_exp0_1.csv"
    combined.to_csv(comparison_file, index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "parent_experiment": exp01.EXPERIMENT_ID,
        "question": "For the paired Exp0.1 protocol, how does a local (234)->(234)->K direct SNN perform with timestep-CE versus WholeCount-CE when deployment readout is always Output WholeCount?",
        "architecture": [list(shifts) for shifts in HIDDEN_SHIFTS],
        "objectives": list(OBJECTIVES),
        "variants": [
            {"name": name, "hidden_cap": cap, "output_cap": OUTPUT_CAP}
            for name, cap in VARIANTS
        ],
        "seeds": list(SEEDS),
        "run_count": len(run_specs()),
        "resume_contract": {
            "complete_run": "valid checkpoint + nonempty required history + valid evaluation",
            "complete_run_action": "exit task before data loading/training",
            "checkpoint_only_action": "skip training and evaluate existing checkpoint",
            "incomplete_checkpoint_action": "retrain run",
        },
        "paired_seed_namespace": "exp0_1_general_comparison",
        "files": {
            "runs": runs_file.name,
            "summary": summary_file.name,
            "paired_objective_effects": objective_file.name,
            "paired_capacity_effects": capacity_file.name,
            "comparison_with_exp0_1": comparison_file.name,
        },
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    return {
        "runs": runs_file,
        "summary": summary_file,
        "paired_objective_effects": objective_file,
        "paired_capacity_effects": capacity_file,
        "comparison_with_exp0_1": comparison_file,
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

    subparsers.add_parser("finalize")
    return parser


def _config(args: argparse.Namespace, repo_root: Path) -> Config:
    return Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threads=args.threads,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = exp3.find_repo_root()
    if args.command == "finalize":
        for name, path in finalize_experiment(repo_root).items():
            print(f"{name}: {path}")
        return

    specs = run_specs()
    if args.array_task_id < 0 or args.array_task_id >= len(specs):
        raise IndexError(f"array task {args.array_task_id} outside [0, {len(specs) - 1}]")
    spec = specs[args.array_task_id]
    root = results_dir(repo_root)

    if not args.force and run_complete(root, spec):
        print(f"run-skip {spec.key}: checkpoint, history, and evaluation are complete")
        return

    data = prepare_data(repo_root)
    config = _config(args, repo_root)
    payload = run_one(spec, data, config, args.force)
    test = payload["metrics"]["test"]
    print(
        f"completed {spec.key}: readout=output_whole_count "
        f"test_BA={test['balanced_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
