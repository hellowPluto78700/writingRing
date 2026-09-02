from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from scripts import experiment_4_0_fixed250_temporal_snn as exp40


EXPERIMENT_ID = "experiment_4_2_continuous_recurrent_controls"
PROTOCOL_VERSION = "continuous_controls_v1"
MODEL_TYPES = ("ff_ann", "rnn")
HIDDEN_WIDTHS = (64, 128)
OBJECTIVES = ("endpoint_ce", "sum_logits_ce")
SEEDS = exp40.SEEDS
REFERENCE_SNN_CONFIGS = ((64, 1000.0), (128, 250.0))
EXPECTED_RUNS = len(MODEL_TYPES) * len(HIDDEN_WIDTHS) * len(OBJECTIVES) * len(SEEDS)


@dataclass(frozen=True)
class RunSpec:
    model_type: str
    hidden_width: int
    objective: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.model_type}__h{self.hidden_width}__{self.objective}__seed{self.seed}"


def results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / EXPERIMENT_ID
        / PROTOCOL_VERSION
    )


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(model_type, hidden_width, objective, seed)
        for model_type in MODEL_TYPES
        for hidden_width in HIDDEN_WIDTHS
        for objective in OBJECTIVES
        for seed in SEEDS
    ]


def parameter_counts(hidden_width: int, recurrent: bool, n_classes: int) -> dict[str, int]:
    input_hidden = exp40.EVENT_CHANNELS * hidden_width
    recurrent_count = hidden_width * hidden_width if recurrent else 0
    hidden_output = hidden_width * n_classes
    return {
        "input_hidden": int(input_hidden),
        "recurrent": int(recurrent_count),
        "hidden_output": int(hidden_output),
        "total": int(input_hidden + recurrent_count + hidden_output),
    }


def valid_endpoint_logits(logits: torch.Tensor, valid_bins: torch.Tensor) -> torch.Tensor:
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_index, valid_bins - 1]


def valid_sum_logits(logits: torch.Tensor, valid_bins: torch.Tensor) -> torch.Tensor:
    mask = exp40.macro_mask(valid_bins, logits.shape[1]).to(logits.dtype).unsqueeze(-1)
    return (logits * mask).sum(dim=1)


def full_sum_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum(dim=1)


class ContinuousTemporalDecoder(nn.Module):
    """Continuous-state controls over the same scaled Fixed250 vectors.

    Both models use tanh so FF-ANN vs RNN changes only the recurrent state path.
    All padded timesteps execute. FF-ANN returns exactly to zero on padded zero
    input; RNN can retain autonomous state through its recurrent matrix.
    """

    def __init__(self, model_type: str, hidden_width: int, n_classes: int) -> None:
        super().__init__()
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model_type: {model_type}")
        if hidden_width not in HIDDEN_WIDTHS:
            raise ValueError(f"Unsupported hidden width: {hidden_width}")
        self.model_type = model_type
        self.hidden_width = int(hidden_width)
        self.n_classes = int(n_classes)
        self.input_hidden = nn.Linear(exp40.EVENT_CHANNELS, hidden_width, bias=False)
        self.recurrent = (
            nn.Linear(hidden_width, hidden_width, bias=False)
            if model_type == "rnn"
            else None
        )
        self.hidden_output = nn.Linear(hidden_width, n_classes, bias=False)

    def forward_trajectory(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_bins, _ = x.shape
        hidden = torch.zeros(batch, self.hidden_width, device=x.device, dtype=x.dtype)
        hidden_seq: list[torch.Tensor] = []
        logits_seq: list[torch.Tensor] = []
        for b in range(n_bins):
            current = self.input_hidden(x[:, b])
            if self.recurrent is not None:
                current = current + self.recurrent(hidden)
            hidden = torch.tanh(current)
            logits = self.hidden_output(hidden)
            hidden_seq.append(hidden)
            logits_seq.append(logits)
        return torch.stack(logits_seq, dim=1), torch.stack(hidden_seq, dim=1)


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def objective_logits(
    logits: torch.Tensor,
    valid_bins: torch.Tensor,
    objective: str,
) -> torch.Tensor:
    if objective == "endpoint_ce":
        return valid_endpoint_logits(logits, valid_bins)
    if objective == "sum_logits_ce":
        return valid_sum_logits(logits, valid_bins)
    raise ValueError(f"Unknown objective: {objective}")


def evaluate_model(
    model: ContinuousTemporalDecoder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    objective: str,
) -> dict[str, object]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    predictions: dict[str, list[np.ndarray]] = {
        "endpoint_logits": [],
        "valid_sum_logits": [],
        "full_sum_logits": [],
    }
    loss_sum = 0.0
    n_total = 0
    tail_l1 = 0.0
    full_l1 = 0.0
    with torch.no_grad():
        for X, y, valid_bins in data_loader:
            X = X.to(device)
            y = y.to(device)
            valid_bins = valid_bins.to(device)
            logits, _ = model.forward_trajectory(X)
            endpoint = valid_endpoint_logits(logits, valid_bins)
            valid_sum = valid_sum_logits(logits, valid_bins)
            full_sum = full_sum_logits(logits)
            primary = objective_logits(logits, valid_bins, objective)
            loss = F.cross_entropy(primary, y)
            n = len(y)
            loss_sum += float(loss.item()) * n
            n_total += n
            y_true_parts.append(y.cpu().numpy())
            for name, values in (
                ("endpoint_logits", endpoint),
                ("valid_sum_logits", valid_sum),
                ("full_sum_logits", full_sum),
            ):
                predictions[name].append(values.argmax(dim=1).cpu().numpy())
            tail_l1 += float((full_sum - valid_sum).abs().sum().item())
            full_l1 += float(full_sum.abs().sum().item())

    y_true = np.concatenate(y_true_parts)
    result: dict[str, object] = {
        "primary_loss": float(loss_sum / max(n_total, 1)),
        "tail_logit_l1": float(tail_l1),
        "full_logit_l1": float(full_l1),
        "tail_logit_l1_fraction": float(tail_l1 / max(full_l1, exp40.EPS)),
    }
    for name, parts in predictions.items():
        result[name] = exp40.metrics(y_true, np.concatenate(parts))
    return result


def _provenance(
    spec: RunSpec,
    data: exp40.BinnedData,
    config: exp40.Config,
) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_representation_experiment": exp40.EXPERIMENT_ID,
        "model_type": spec.model_type,
        "hidden_width": spec.hidden_width,
        "objective": spec.objective,
        "activation": "tanh",
        "bias": False,
        "seed": spec.seed,
        "split_seed": int(exp40.base.SPLIT_SEED),
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
        "n_padded_bins": data.n_bins,
        "fixed_bin_ms": exp40.FIXED_MS,
        "event_channels": exp40.EVENT_CHANNELS,
        "channel_scale": data.channel_scale.tolist(),
        "state_dynamics": "all padded bins execute; zero input after endpoint; no state freeze",
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": exp40.LR,
        "parameter_counts": parameter_counts(
            spec.hidden_width,
            spec.model_type == "rnn",
            len(data.labels),
        ),
    }


def run_one(
    spec: RunSpec,
    data: exp40.BinnedData,
    config: exp40.Config,
) -> dict[str, object]:
    path = evaluation_path(config.results_dir, spec)
    ckpt = checkpoint_path(config.results_dir, spec)
    if config.resume and path.exists() and ckpt.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp40.base.seed_all(
        exp40.base.dseed(
            spec.seed,
            "exp4_2_model_init",
            spec.model_type,
            spec.hidden_width,
            spec.objective,
        )
    )
    model = ContinuousTemporalDecoder(
        spec.model_type,
        spec.hidden_width,
        len(data.labels),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp40.LR)

    partitions = [
        (data.Xtr, data.ytr, data.btr),
        (data.Xva, data.yva, data.bva),
        (data.Xte, data.yte, data.bte),
    ]
    train_loader = exp40.loader(
        *partitions[0],
        config.batch_size,
        True,
        exp40.base.dseed(spec.seed, spec.key, "train_loader"),
    )
    eval_loaders = [
        exp40.loader(
            *partition,
            config.batch_size,
            False,
            exp40.base.dseed(spec.seed, spec.key, split, "eval_loader"),
        )
        for partition, split in zip(partitions, ("train", "val", "test"), strict=True)
    ]

    primary_readout = "endpoint_logits" if spec.objective == "endpoint_ce" else "valid_sum_logits"
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -np.inf
    best_val_loss = np.inf
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for X, y, valid_bins in train_loader:
            X = X.to(device)
            y = y.to(device)
            valid_bins = valid_bins.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model.forward_trajectory(X)
            primary = objective_logits(logits, valid_bins, spec.objective)
            loss = F.cross_entropy(primary, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(y)
            train_n += len(y)

        val_metrics = evaluate_model(model, eval_loaders[1], device, spec.objective)
        val_ba = float(val_metrics[primary_readout]["balanced_accuracy"])
        val_loss = float(val_metrics["primary_loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss_sum / max(train_n, 1)),
                "val_primary_loss": val_loss,
                "val_primary_ba": val_ba,
            }
        )
        if (val_ba > best_val_ba) or (
            np.isclose(val_ba, best_val_ba) and val_loss < best_val_loss
        ):
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    model.load_state_dict(best_state)
    split_metrics = {
        split: evaluate_model(model, loader, device, spec.objective)
        for split, loader in zip(("train", "val", "test"), eval_loaders, strict=True)
    }
    provenance = _provenance(spec, data, config)

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "provenance": provenance,
            "best_epoch": best_epoch,
            "model_state_dict": best_state,
            "history": history,
        },
        ckpt,
    )
    payload: dict[str, object] = {
        "spec": spec.__dict__,
        "key": spec.key,
        "best_epoch": best_epoch,
        "primary_readout": primary_readout,
        "provenance": provenance,
        "metrics": split_metrics,
        "history": history,
    }
    _save_json(path, payload)
    return payload


def _continuous_row(payload: dict[str, object], split: str) -> dict[str, object]:
    spec = payload["spec"]
    metrics = payload["metrics"][split]
    primary_readout = payload["primary_readout"]
    row: dict[str, object] = {
        "model_type": spec["model_type"],
        "hidden_width": int(spec["hidden_width"]),
        "objective": spec["objective"],
        "seed": int(spec["seed"]),
        "split": split,
        "best_epoch": int(payload["best_epoch"]),
        "primary_readout": primary_readout,
        "parameter_count": int(payload["provenance"]["parameter_counts"]["total"]),
        "tail_logit_l1_fraction": float(metrics["tail_logit_l1_fraction"]),
    }
    for readout in ("endpoint_logits", "valid_sum_logits", "full_sum_logits"):
        for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
            row[f"{readout}_{metric_name}"] = metrics[readout][metric_name]
    row["primary_balanced_accuracy"] = metrics[primary_readout]["balanced_accuracy"]
    row["primary_macro_f1"] = metrics[primary_readout]["macro_f1"]
    return row


def finalize(data: exp40.BinnedData, config: exp40.Config) -> None:
    missing: list[str] = []
    for spec in run_specs():
        path = evaluation_path(config.results_dir, spec)
        if not path.exists():
            missing.append(str(path))

    exp40_root = exp40.results_dir(config.repo_root)
    baseline_path = exp40_root / "baseline" / "fixed250_linear.json"
    if not baseline_path.exists():
        missing.append(str(baseline_path))

    reference_specs = [
        exp40.RunSpec(architecture, width, tau_mem_ms, seed)
        for architecture in ("ff", "rsnn")
        for width, tau_mem_ms in REFERENCE_SNN_CONFIGS
        for seed in SEEDS
    ]
    for spec in reference_specs:
        path = exp40.evaluation_path(exp40_root, spec)
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Cannot finalize Experiment 4.2; missing required artifacts:\n"
            + "\n".join(missing[:20])
        )

    rows: list[dict[str, object]] = []
    for spec in run_specs():
        payload = json.loads(
            evaluation_path(config.results_dir, spec).read_text(encoding="utf-8")
        )
        for split in ("train", "val", "test"):
            rows.append(_continuous_row(payload, split))
    runs = pd.DataFrame(rows)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "runs.csv", index=False)

    test = runs[runs["split"] == "test"].copy()
    summary = (
        test.groupby(["model_type", "hidden_width", "objective"])[
            ["primary_balanced_accuracy", "primary_macro_f1", "tail_logit_l1_fraction"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(config.results_dir / "summary.csv", index=False)

    reference_rows: list[dict[str, object]] = []
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    reference_rows.append(
        {
            "method": "linear",
            "architecture": "linear",
            "hidden_width": np.nan,
            "tau_mem_ms": np.nan,
            "seed": np.nan,
            "test_balanced_accuracy": baseline["metrics"]["test"]["balanced_accuracy"],
            "test_macro_f1": baseline["metrics"]["test"]["macro_f1"],
        }
    )
    for spec in reference_specs:
        payload = json.loads(
            exp40.evaluation_path(exp40_root, spec).read_text(encoding="utf-8")
        )
        metrics = payload["metrics"]["test"]["valid_count"]
        reference_rows.append(
            {
                "method": f"{spec.architecture}_snn",
                "architecture": spec.architecture,
                "hidden_width": spec.hidden_width,
                "tau_mem_ms": spec.tau_mem_ms,
                "seed": spec.seed,
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_macro_f1": metrics["macro_f1"],
            }
        )
    pd.DataFrame(reference_rows).to_csv(config.results_dir / "references.csv", index=False)

    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "expected_runs": EXPECTED_RUNS,
        "model_types": list(MODEL_TYPES),
        "hidden_widths": list(HIDDEN_WIDTHS),
        "objectives": list(OBJECTIVES),
        "seeds": list(SEEDS),
        "split_seed": int(exp40.base.SPLIT_SEED),
        "activation": "tanh for both FF-ANN and RNN",
        "question": (
            "separate recurrent sequence-model capacity from spiking dynamics and count-style readout"
        ),
        "reference_snn_configs": [list(x) for x in REFERENCE_SNN_CONFIGS],
        "linear_reference_test_ba": baseline["metrics"]["test"]["balanced_accuracy"],
    }
    _save_json(config.results_dir / "provenance.json", provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 4.2 continuous recurrent controls")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--epochs", type=int, default=exp40.EPOCHS)
    run.add_argument("--batch-size", type=int, default=exp40.BATCH_SIZE)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--force-retrain", action="store_true")
    final = sub.add_parser("finalize")
    final.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = exp40.find_repo_root()
    root = results_dir(repo_root)
    data = exp40.prepare_binned_data(repo_root)
    if args.command == "run-one":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(
                f"array-task-id {args.array_task_id} outside [0,{len(specs)-1}]"
            )
        spec = specs[args.array_task_id]
        config = exp40.Config(
            repo_root=repo_root,
            results_dir=root,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume=not args.force_retrain,
            threads=args.threads,
        )
        payload = run_one(spec, data, config)
        print(json.dumps({"completed": spec.key, "best_epoch": payload["best_epoch"]}, indent=2))
    else:
        config = exp40.Config(repo_root=repo_root, results_dir=root, device=args.device)
        finalize(data, config)
        print(root / "summary.csv")


if __name__ == "__main__":
    main()
