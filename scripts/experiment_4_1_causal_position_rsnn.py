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
import snntorch as snn
from snntorch import surrogate

from scripts import experiment_4_0_fixed250_temporal_snn as exp40


EXPERIMENT_ID = "experiment_4_1_causal_position_rsnn"
PROTOCOL_VERSION = "causal_position_v1"
POSITION_ENCODINGS = ("scalar", "onehot")
ANCHOR_CONFIGS = ((64, 1000.0), (128, 250.0))
SEEDS = exp40.SEEDS
EXPECTED_NEW_RUNS = len(POSITION_ENCODINGS) * len(ANCHOR_CONFIGS) * len(SEEDS)


@dataclass(frozen=True)
class RunSpec:
    hidden_width: int
    tau_mem_ms: float
    position_encoding: str
    seed: int

    @property
    def key(self) -> str:
        return (
            f"rsnn__h{self.hidden_width}__tau{int(self.tau_mem_ms)}ms__"
            f"pos-{self.position_encoding}__seed{self.seed}"
        )


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
        RunSpec(width, tau_mem_ms, position_encoding, seed)
        for width, tau_mem_ms in ANCHOR_CONFIGS
        for position_encoding in POSITION_ENCODINGS
        for seed in SEEDS
    ]


def add_causal_position_features(
    X: np.ndarray,
    valid_bins: np.ndarray,
    position_encoding: str,
) -> np.ndarray:
    """Append causal absolute-bin position, zeroed after each sample endpoint.

    The position is absolute elapsed time / slot identity relative to the fixed
    padded horizon, never relative to the unknown final gesture duration.
    Position inputs are zero after valid_bins so post-end dynamics remain a
    zero-external-input diagnostic exactly as in Experiment 4.0.
    """
    X = np.asarray(X, dtype=np.float32)
    valid_bins = np.asarray(valid_bins, dtype=np.int64)
    if X.ndim != 3 or X.shape[2] != exp40.EVENT_CHANNELS:
        raise ValueError(f"Expected [N,B,{exp40.EVENT_CHANNELS}], got {X.shape}")
    n_samples, n_bins, _ = X.shape
    valid_mask = np.arange(n_bins)[None, :] < valid_bins[:, None]

    if position_encoding == "scalar":
        position = (np.arange(n_bins, dtype=np.float32) + 1.0) / float(n_bins)
        encoded = np.broadcast_to(position[None, :, None], (n_samples, n_bins, 1)).copy()
        encoded *= valid_mask[:, :, None]
    elif position_encoding == "onehot":
        basis = np.eye(n_bins, dtype=np.float32)
        encoded = np.broadcast_to(basis[None, :, :], (n_samples, n_bins, n_bins)).copy()
        encoded *= valid_mask[:, :, None]
    else:
        raise ValueError(f"Unknown position encoding: {position_encoding}")

    return np.concatenate([X, encoded], axis=2).astype(np.float32, copy=False)


def input_dim(position_encoding: str, n_bins: int) -> int:
    if position_encoding == "scalar":
        return exp40.EVENT_CHANNELS + 1
    if position_encoding == "onehot":
        return exp40.EVENT_CHANNELS + n_bins
    raise ValueError(f"Unknown position encoding: {position_encoding}")


def parameter_counts(
    hidden_width: int,
    n_classes: int,
    position_encoding: str,
    n_bins: int,
) -> dict[str, int]:
    in_dim = input_dim(position_encoding, n_bins)
    input_hidden = in_dim * hidden_width
    recurrent = hidden_width * hidden_width
    hidden_output = hidden_width * n_classes
    return {
        "input_hidden": int(input_hidden),
        "recurrent": int(recurrent),
        "hidden_output": int(hidden_output),
        "total": int(input_hidden + recurrent + hidden_output),
    }


class PositionAwareRSNN(nn.Module):
    def __init__(
        self,
        hidden_width: int,
        tau_mem_ms: float,
        n_classes: int,
        position_encoding: str,
        n_bins: int,
    ) -> None:
        super().__init__()
        if (hidden_width, tau_mem_ms) not in ANCHOR_CONFIGS:
            raise ValueError(f"Unsupported anchor: {(hidden_width, tau_mem_ms)}")
        self.hidden_width = int(hidden_width)
        self.tau_mem_ms = float(tau_mem_ms)
        self.position_encoding = position_encoding
        self.n_classes = int(n_classes)
        self.n_bins = int(n_bins)

        beta_hidden = np.exp(-exp40.FIXED_MS / self.tau_mem_ms)
        beta_output = np.exp(-exp40.FIXED_MS / exp40.OUTPUT_TAU_MEM_MS)
        spike_grad = surrogate.fast_sigmoid(slope=exp40.SURROGATE_SLOPE)

        self.input_hidden = nn.Linear(
            input_dim(position_encoding, n_bins), hidden_width, bias=False
        )
        self.recurrent = nn.Linear(hidden_width, hidden_width, bias=False)
        self.hidden_lif = snn.Leaky(
            beta=float(beta_hidden),
            threshold=exp40.THRESHOLD,
            spike_grad=spike_grad,
            reset_mechanism=exp40.RESET,
        )
        self.hidden_output = nn.Linear(hidden_width, n_classes, bias=False)
        self.output_lif = snn.Leaky(
            beta=float(beta_output),
            threshold=exp40.THRESHOLD,
            spike_grad=spike_grad,
            reset_mechanism=exp40.RESET,
        )

    def forward_trajectory(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_bins, _ = x.shape
        hidden_mem = torch.zeros(batch, self.hidden_width, device=x.device, dtype=x.dtype)
        output_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)
        prev_hidden_spike = torch.zeros_like(hidden_mem)
        hidden_spikes: list[torch.Tensor] = []
        output_spikes: list[torch.Tensor] = []
        output_mems: list[torch.Tensor] = []

        for b in range(n_bins):
            current = self.input_hidden(x[:, b]) + self.recurrent(prev_hidden_spike)
            hidden_spike, hidden_mem = self.hidden_lif(current, hidden_mem)
            output_current = self.hidden_output(hidden_spike)
            output_spike, output_mem = self.output_lif(output_current, output_mem)
            hidden_spikes.append(hidden_spike)
            output_spikes.append(output_spike)
            output_mems.append(output_mem)
            prev_hidden_spike = hidden_spike

        return (
            torch.stack(output_spikes, dim=1),
            torch.stack(output_mems, dim=1),
            torch.stack(hidden_spikes, dim=1),
        )


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def evaluate_model(
    model: PositionAwareRSNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    predictions: dict[str, list[np.ndarray]] = {
        "valid_count": [],
        "full_count": [],
        "valid_membrane": [],
        "full_membrane": [],
    }
    loss_sum = 0.0
    n_total = 0
    tail_spikes = 0.0
    full_spikes = 0.0
    with torch.no_grad():
        for X, y, valid_bins in data_loader:
            X = X.to(device)
            y = y.to(device)
            valid_bins = valid_bins.to(device)
            spikes, membranes, _ = model.forward_trajectory(X)
            valid_count = exp40.valid_whole_count(spikes, valid_bins)
            full_count = exp40.full_whole_count(spikes)
            valid_mem = exp40.valid_final_membrane(membranes, valid_bins)
            full_mem = membranes[:, -1]
            loss = F.cross_entropy(valid_count, y)
            n = len(y)
            loss_sum += float(loss.item()) * n
            n_total += n
            y_true_parts.append(y.cpu().numpy())
            for name, logits in (
                ("valid_count", valid_count),
                ("full_count", full_count),
                ("valid_membrane", valid_mem),
                ("full_membrane", full_mem),
            ):
                predictions[name].append(logits.argmax(dim=1).cpu().numpy())
            tail_spikes += float((full_count - valid_count).sum().item())
            full_spikes += float(full_count.sum().item())

    y_true = np.concatenate(y_true_parts)
    result: dict[str, object] = {
        "valid_count_loss": float(loss_sum / max(n_total, 1)),
        "tail_output_spikes": float(tail_spikes),
        "full_output_spikes": float(full_spikes),
        "tail_spike_fraction": float(tail_spikes / max(full_spikes, exp40.EPS)),
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
        "hidden_width": spec.hidden_width,
        "tau_mem_ms": spec.tau_mem_ms,
        "position_encoding": spec.position_encoding,
        "position_definition": (
            "absolute padded-slot index; scalar=b/Bmax or onehot(slot); valid bins only"
        ),
        "post_endpoint_position": "zero",
        "seed": spec.seed,
        "split_seed": int(exp40.base.SPLIT_SEED),
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
        "n_padded_bins": data.n_bins,
        "event_channels": exp40.EVENT_CHANNELS,
        "fixed_bin_ms": exp40.FIXED_MS,
        "channel_scale": data.channel_scale.tolist(),
        "primary_objective": "valid_whole_count_ce",
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": exp40.LR,
        "parameter_counts": parameter_counts(
            spec.hidden_width,
            len(data.labels),
            spec.position_encoding,
            data.n_bins,
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
    Xtr = add_causal_position_features(data.Xtr, data.btr, spec.position_encoding)
    Xva = add_causal_position_features(data.Xva, data.bva, spec.position_encoding)
    Xte = add_causal_position_features(data.Xte, data.bte, spec.position_encoding)
    partitions = [
        (Xtr, data.ytr, data.btr),
        (Xva, data.yva, data.bva),
        (Xte, data.yte, data.bte),
    ]

    exp40.base.seed_all(
        exp40.base.dseed(
            spec.seed,
            "exp4_1_model_init",
            spec.hidden_width,
            spec.tau_mem_ms,
            spec.position_encoding,
        )
    )
    model = PositionAwareRSNN(
        spec.hidden_width,
        spec.tau_mem_ms,
        len(data.labels),
        spec.position_encoding,
        data.n_bins,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp40.LR)

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
            spikes, _, _ = model.forward_trajectory(X)
            logits = exp40.valid_whole_count(spikes, valid_bins)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(y)
            train_n += len(y)

        val_metrics = evaluate_model(model, eval_loaders[1], device)
        val_ba = float(val_metrics["valid_count"]["balanced_accuracy"])
        val_loss = float(val_metrics["valid_count_loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss_sum / max(train_n, 1)),
                "val_valid_count_loss": val_loss,
                "val_valid_count_ba": val_ba,
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
        split: evaluate_model(model, loader, device)
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
        "provenance": provenance,
        "metrics": split_metrics,
        "history": history,
    }
    _save_json(path, payload)
    return payload


def _row_from_payload(
    payload: dict[str, object],
    split: str,
    position_encoding: str,
    source_experiment: str,
) -> dict[str, object]:
    spec = payload["spec"]
    metrics = payload["metrics"][split]
    row: dict[str, object] = {
        "hidden_width": int(spec["hidden_width"]),
        "tau_mem_ms": float(spec["tau_mem_ms"]),
        "position_encoding": position_encoding,
        "seed": int(spec["seed"]),
        "split": split,
        "source_experiment": source_experiment,
        "best_epoch": int(payload["best_epoch"]),
        "tail_spike_fraction": float(metrics["tail_spike_fraction"]),
    }
    for readout in ("valid_count", "full_count", "valid_membrane", "full_membrane"):
        for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
            row[f"{readout}_{metric_name}"] = metrics[readout][metric_name]
    return row


def finalize(data: exp40.BinnedData, config: exp40.Config) -> None:
    missing: list[str] = []
    for spec in run_specs():
        path = evaluation_path(config.results_dir, spec)
        if not path.exists():
            missing.append(str(path))

    exp40_root = exp40.results_dir(config.repo_root)
    control_specs = [
        exp40.RunSpec("rsnn", width, tau_mem_ms, seed)
        for width, tau_mem_ms in ANCHOR_CONFIGS
        for seed in SEEDS
    ]
    for spec in control_specs:
        path = exp40.evaluation_path(exp40_root, spec)
        if not path.exists():
            missing.append(str(path))

    baseline_path = exp40_root / "baseline" / "fixed250_linear.json"
    if not baseline_path.exists():
        missing.append(str(baseline_path))
    if missing:
        raise FileNotFoundError(
            "Cannot finalize Experiment 4.1; missing required artifacts:\n"
            + "\n".join(missing[:20])
        )

    rows: list[dict[str, object]] = []
    for spec in control_specs:
        payload = json.loads(
            exp40.evaluation_path(exp40_root, spec).read_text(encoding="utf-8")
        )
        for split in ("train", "val", "test"):
            rows.append(_row_from_payload(payload, split, "none", exp40.EXPERIMENT_ID))

    for spec in run_specs():
        payload = json.loads(
            evaluation_path(config.results_dir, spec).read_text(encoding="utf-8")
        )
        for split in ("train", "val", "test"):
            rows.append(
                _row_from_payload(payload, split, spec.position_encoding, EXPERIMENT_ID)
            )

    runs = pd.DataFrame(rows)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "runs.csv", index=False)

    test = runs[runs["split"] == "test"].copy()
    group_cols = ["hidden_width", "tau_mem_ms", "position_encoding"]
    value_cols = [
        "valid_count_balanced_accuracy",
        "valid_count_macro_f1",
        "full_count_balanced_accuracy",
        "tail_spike_fraction",
    ]
    summary = test.groupby(group_cols)[value_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(config.results_dir / "summary.csv", index=False)

    paired = test.pivot_table(
        index=["hidden_width", "tau_mem_ms", "seed"],
        columns="position_encoding",
        values="valid_count_balanced_accuracy",
    ).reset_index()
    for position_encoding in POSITION_ENCODINGS:
        paired[f"{position_encoding}_minus_none"] = (
            paired[position_encoding] - paired["none"]
        )
    paired.to_csv(config.results_dir / "paired_position_effects.csv", index=False)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "new_runs": EXPECTED_NEW_RUNS,
        "reused_exp4_0_control_runs": len(control_specs),
        "anchor_configs": [list(x) for x in ANCHOR_CONFIGS],
        "position_encodings": list(POSITION_ENCODINGS),
        "seeds": list(SEEDS),
        "split_seed": int(exp40.base.SPLIT_SEED),
        "linear_reference_test_ba": baseline["metrics"]["test"]["balanced_accuracy"],
        "causality_guardrail": (
            "position uses absolute elapsed padded-slot index only; no final-duration normalization"
        ),
        "post_endpoint_position": "zero",
    }
    _save_json(config.results_dir / "provenance.json", provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 4.1 causal position RSNN")
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
