from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import snntorch as snn
from snntorch import surrogate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from scripts import experiment_3_0_1_single_tau_objectives as base


EXPERIMENT_ID = "experiment_4_0_fixed250_temporal_snn"
PROTOCOL_VERSION = "fixed250_stateful_decoder_v1"

ARCHITECTURES = ("ff", "rsnn")
HIDDEN_WIDTHS = (32, 64, 128)
TAU_MEM_MS_VALUES = (250.0, 500.0, 1000.0, 2000.0)
SEEDS = (11, 23, 37, 53, 71)
EXPECTED_SNN_RUNS = (
    len(ARCHITECTURES) * len(HIDDEN_WIDTHS) * len(TAU_MEM_MS_VALUES) * len(SEEDS)
)

FIXED_MS = 250.0
EVENT_CHANNELS = 30
OUTPUT_TAU_MEM_MS = 250.0
THRESHOLD = 0.5
SURROGATE_SLOPE = 25.0
RESET = "subtract"
EPS = 1e-6
BATCH_SIZE = 128
EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 0.0
LOGREG_MAX_ITER = 5000


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    hidden_width: int
    tau_mem_ms: float
    seed: int

    @property
    def key(self) -> str:
        tau_tag = str(int(self.tau_mem_ms))
        return (
            f"{self.architecture}__h{self.hidden_width}__"
            f"tau{tau_tag}ms__seed{self.seed}"
        )


@dataclass
class BinnedData:
    Xtr: np.ndarray
    ytr: np.ndarray
    btr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    bva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    bte: np.ndarray
    channel_scale: np.ndarray
    labels: tuple[str, ...]
    split: dict[str, tuple[str, ...]]
    fs: float
    bin_steps: int
    n_bins: int


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    resume: bool = True
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "snn").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate writingRing repository root")


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
        RunSpec(architecture, width, tau_mem_ms, seed)
        for architecture in ARCHITECTURES
        for width in HIDDEN_WIDTHS
        for tau_mem_ms in TAU_MEM_MS_VALUES
        for seed in SEEDS
    ]


def valid_bin_counts_np(lengths: np.ndarray, bin_steps: int, n_bins: int) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64)
    return np.clip((np.maximum(lengths, 1) + bin_steps - 1) // bin_steps, 1, n_bins)


def valid_bin_counts_torch(
    lengths: torch.Tensor,
    bin_steps: int,
    n_bins: int,
) -> torch.Tensor:
    return torch.div(
        lengths.clamp_min(1) + bin_steps - 1,
        bin_steps,
        rounding_mode="floor",
    ).clamp(min=1, max=n_bins)


def fixed250_channel_counts(
    X: np.ndarray,
    lengths: np.ndarray,
    bin_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate raw event channels inside absolute Fixed250 bins.

    Samples after valid_length are explicitly zeroed before summation. The last
    non-empty partial bin is retained. Returned valid-bin counts are endpoint
    indices for the macro-step SNN.
    """
    X = np.asarray(X, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.int64)
    if X.ndim != 3 or X.shape[2] != EVENT_CHANNELS:
        raise ValueError(f"Expected [N,T,{EVENT_CHANNELS}] raw events, got {X.shape}")
    n_samples, T, channels = X.shape
    n_bins = int(math.ceil(T / bin_steps))
    padded_T = n_bins * bin_steps
    masked = np.zeros((n_samples, padded_T, channels), dtype=np.float32)
    for i in range(n_samples):
        n = min(int(lengths[i]), T)
        if n > 0:
            masked[i, :n] = X[i, :n]
    counts = masked.reshape(n_samples, n_bins, bin_steps, channels).sum(axis=2)
    valid_bins = valid_bin_counts_np(lengths, bin_steps, n_bins)
    return counts.astype(np.float32, copy=False), valid_bins


def fit_zero_preserving_channel_scale(
    X_train: np.ndarray,
    valid_bins: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """Fit one std scale per channel using training valid bins only.

    No mean is subtracted, so padded zero bins remain exactly zero after scaling.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    valid_bins = np.asarray(valid_bins, dtype=np.int64)
    rows = [X_train[i, : int(valid_bins[i])] for i in range(len(X_train))]
    valid = np.concatenate(rows, axis=0)
    scale = valid.std(axis=0, ddof=0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > eps), scale, 1.0).astype(np.float32)
    if scale.shape != (EVENT_CHANNELS,):
        raise ValueError(f"Expected {EVENT_CHANNELS} channel scales, got {scale.shape}")
    return scale


def apply_channel_scale(X: np.ndarray, scale: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    return (X / scale[None, None, :]).astype(np.float32, copy=False)


def prepare_binned_data(repo_root: Path) -> BinnedData:
    raw = base.prepare_data(repo_root)
    if raw.Xtr.shape[2] != EVENT_CHANNELS:
        raise ValueError(f"Expected {EVENT_CHANNELS} event channels")
    expected_steps = int(np.rint(FIXED_MS * raw.fs / 1000.0))
    if expected_steps != raw.bin_steps:
        raise ValueError(
            f"Existing Fixed250 contract uses {raw.bin_steps} steps, expected {expected_steps}"
        )

    Xtr, btr = fixed250_channel_counts(raw.Xtr, raw.ltr, raw.bin_steps)
    Xva, bva = fixed250_channel_counts(raw.Xva, raw.lva, raw.bin_steps)
    Xte, bte = fixed250_channel_counts(raw.Xte, raw.lte, raw.bin_steps)
    scale = fit_zero_preserving_channel_scale(Xtr, btr)
    Xtr = apply_channel_scale(Xtr, scale)
    Xva = apply_channel_scale(Xva, scale)
    Xte = apply_channel_scale(Xte, scale)

    return BinnedData(
        Xtr=Xtr,
        ytr=raw.ytr,
        btr=btr,
        Xva=Xva,
        yva=raw.yva,
        bva=bva,
        Xte=Xte,
        yte=raw.yte,
        bte=bte,
        channel_scale=scale,
        labels=raw.labels,
        split=raw.split,
        fs=raw.fs,
        bin_steps=raw.bin_steps,
        n_bins=Xtr.shape[1],
    )


def loader(
    X: np.ndarray,
    y: np.ndarray,
    valid_bins: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(valid_bins, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def macro_mask(valid_bins: torch.Tensor, n_bins: int) -> torch.Tensor:
    return torch.arange(n_bins, device=valid_bins.device)[None, :] < valid_bins[:, None]


def valid_whole_count(spikes: torch.Tensor, valid_bins: torch.Tensor) -> torch.Tensor:
    mask = macro_mask(valid_bins, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
    return (spikes * mask).sum(dim=1)


def full_whole_count(spikes: torch.Tensor) -> torch.Tensor:
    return spikes.sum(dim=1)


def valid_final_membrane(mem: torch.Tensor, valid_bins: torch.Tensor) -> torch.Tensor:
    batch_index = torch.arange(mem.shape[0], device=mem.device)
    return mem[batch_index, valid_bins - 1]


def parameter_counts(hidden_width: int, recurrent: bool, n_classes: int) -> dict[str, int]:
    input_hidden = EVENT_CHANNELS * hidden_width
    recurrent_count = hidden_width * hidden_width if recurrent else 0
    hidden_output = hidden_width * n_classes
    total = input_hidden + recurrent_count + hidden_output
    return {
        "input_hidden": int(input_hidden),
        "recurrent": int(recurrent_count),
        "hidden_output": int(hidden_output),
        "total": int(total),
    }


class TemporalDecoderSNN(nn.Module):
    """Single-hidden-layer temporal SNN over Fixed250 event vectors.

    All samples execute all padded macro timesteps. Inputs after each sample's
    endpoint are zero, but hidden/output membrane dynamics continue normally.
    Valid-length information is used only by the readout helpers.
    """

    def __init__(
        self,
        architecture: str,
        hidden_width: int,
        tau_mem_ms: float,
        n_classes: int,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        if hidden_width not in HIDDEN_WIDTHS:
            raise ValueError(f"Unsupported hidden width: {hidden_width}")
        if tau_mem_ms not in TAU_MEM_MS_VALUES:
            raise ValueError(f"Unsupported tau_mem_ms: {tau_mem_ms}")
        self.architecture = architecture
        self.hidden_width = int(hidden_width)
        self.tau_mem_ms = float(tau_mem_ms)
        self.n_classes = int(n_classes)

        # Macro timestep is exactly one 250 ms input bin.
        beta_hidden = math.exp(-FIXED_MS / self.tau_mem_ms)
        beta_output = math.exp(-FIXED_MS / OUTPUT_TAU_MEM_MS)
        spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

        self.input_hidden = nn.Linear(EVENT_CHANNELS, hidden_width, bias=False)
        self.recurrent = (
            nn.Linear(hidden_width, hidden_width, bias=False)
            if architecture == "rsnn"
            else None
        )
        self.hidden_lif = snn.Leaky(
            beta=beta_hidden,
            threshold=THRESHOLD,
            spike_grad=spike_grad,
            reset_mechanism=RESET,
        )
        self.hidden_output = nn.Linear(hidden_width, n_classes, bias=False)
        self.output_lif = snn.Leaky(
            beta=beta_output,
            threshold=THRESHOLD,
            spike_grad=spike_grad,
            reset_mechanism=RESET,
        )

    def forward_trajectory(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_bins, _ = x.shape
        hidden_mem = torch.zeros(
            batch, self.hidden_width, device=x.device, dtype=x.dtype
        )
        output_mem = torch.zeros(
            batch, self.n_classes, device=x.device, dtype=x.dtype
        )
        prev_hidden_spike = torch.zeros_like(hidden_mem)
        hidden_spikes: list[torch.Tensor] = []
        output_spikes: list[torch.Tensor] = []
        output_mems: list[torch.Tensor] = []

        for b in range(n_bins):
            current = self.input_hidden(x[:, b])
            if self.recurrent is not None:
                current = current + self.recurrent(prev_hidden_spike)
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


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_model(
    model: TemporalDecoderSNN,
    data_loader: DataLoader,
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
            valid_count = valid_whole_count(spikes, valid_bins)
            full_count = full_whole_count(spikes)
            valid_mem = valid_final_membrane(membranes, valid_bins)
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
    out: dict[str, object] = {
        "valid_count_loss": float(loss_sum / max(n_total, 1)),
        "tail_output_spikes": float(tail_spikes),
        "full_output_spikes": float(full_spikes),
        "tail_spike_fraction": float(tail_spikes / max(full_spikes, EPS)),
    }
    for name, parts in predictions.items():
        out[name] = metrics(y_true, np.concatenate(parts))
    return out


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def _provenance(spec: RunSpec, data: BinnedData, config: Config) -> dict[str, object]:
    counts = parameter_counts(
        spec.hidden_width,
        spec.architecture == "rsnn",
        len(data.labels),
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": spec.architecture,
        "hidden_width": int(spec.hidden_width),
        "tau_mem_ms": float(spec.tau_mem_ms),
        "seed": int(spec.seed),
        "split_seed": int(base.SPLIT_SEED),
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
        "event_channels": EVENT_CHANNELS,
        "sampling_rate_hz": float(data.fs),
        "fixed_bin_ms": FIXED_MS,
        "fixed_bin_steps": int(data.bin_steps),
        "n_padded_bins": int(data.n_bins),
        "scaling": "train-valid-bin per-channel std; no mean subtraction",
        "channel_scale": data.channel_scale.tolist(),
        "state_dynamics": "all padded bins execute; zero input after endpoint; no state freeze",
        "primary_objective": "valid_whole_count_ce",
        "secondary_readouts": [
            "full_whole_count",
            "valid_final_output_membrane",
            "full_final_output_membrane",
        ],
        "output_tau_mem_ms": OUTPUT_TAU_MEM_MS,
        "threshold": THRESHOLD,
        "reset": RESET,
        "epochs": int(config.epochs),
        "batch_size": int(config.batch_size),
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "parameter_counts": counts,
    }


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_one(spec: RunSpec, data: BinnedData, config: Config) -> dict[str, object]:
    path = evaluation_path(config.results_dir, spec)
    ckpt = checkpoint_path(config.results_dir, spec)
    provenance = _provenance(spec, data, config)
    if config.resume and path.exists() and ckpt.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    base.seed_all(base.dseed(spec.seed, "exp4_0_model_init", spec.architecture, spec.hidden_width, spec.tau_mem_ms))
    model = TemporalDecoderSNN(
        spec.architecture,
        spec.hidden_width,
        spec.tau_mem_ms,
        len(data.labels),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    partitions = [
        (data.Xtr, data.ytr, data.btr),
        (data.Xva, data.yva, data.bva),
        (data.Xte, data.yte, data.bte),
    ]
    train_loader = loader(
        *partitions[0],
        config.batch_size,
        True,
        base.dseed(spec.seed, spec.key, "train_loader"),
    )
    eval_loaders = [
        loader(
            *partition,
            config.batch_size,
            False,
            base.dseed(spec.seed, spec.key, split, "eval_loader"),
        )
        for partition, split in zip(
            partitions, ("train", "val", "test"), strict=True
        )
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
            logits = valid_whole_count(spikes, valid_bins)
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
        split: evaluate_model(model, dl, device)
        for split, dl in zip(("train", "val", "test"), eval_loaders, strict=True)
    }

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


def run_linear_baseline(data: BinnedData, root: Path) -> dict[str, object]:
    """Fixed250+Linear baseline using the exact same zero-preserving scaled input."""
    path = root / "baseline" / "fixed250_linear.json"
    Xtr = data.Xtr.reshape(len(data.Xtr), -1)
    Xva = data.Xva.reshape(len(data.Xva), -1)
    Xte = data.Xte.reshape(len(data.Xte), -1)
    model = LogisticRegression(
        max_iter=LOGREG_MAX_ITER,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
    )
    model.fit(Xtr, data.ytr)
    split_metrics = {}
    for split, X, y in (
        ("train", Xtr, data.ytr),
        ("val", Xva, data.yva),
        ("test", Xte, data.yte),
    ):
        split_metrics[split] = metrics(y, model.predict(X))
    payload = {
        "method": "fixed250_linear",
        "representation": "same scaled Fixed250 vectors used by every SNN case",
        "extra_centering": False,
        "split_seed": int(base.SPLIT_SEED),
        "channel_scale": data.channel_scale.tolist(),
        "metrics": split_metrics,
        "feature_dim": int(Xtr.shape[1]),
    }
    _save_json(path, payload)
    return payload


def finalize(data: BinnedData, config: Config) -> None:
    specs = run_specs()
    missing = [str(evaluation_path(config.results_dir, spec)) for spec in specs if not evaluation_path(config.results_dir, spec).exists()]
    baseline_path = config.results_dir / "baseline" / "fixed250_linear.json"
    if not baseline_path.exists():
        missing.append(str(baseline_path))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Cannot finalize: {len(missing)} required artifacts are missing.\n{preview}"
        )

    rows: list[dict[str, object]] = []
    for spec in specs:
        payload = json.loads(evaluation_path(config.results_dir, spec).read_text(encoding="utf-8"))
        for split in ("train", "val", "test"):
            m = payload["metrics"][split]
            row: dict[str, object] = {
                "architecture": spec.architecture,
                "hidden_width": spec.hidden_width,
                "tau_mem_ms": spec.tau_mem_ms,
                "seed": spec.seed,
                "split": split,
                "best_epoch": payload["best_epoch"],
                "parameter_count": payload["provenance"]["parameter_counts"]["total"],
                "tail_spike_fraction": m["tail_spike_fraction"],
            }
            for readout in (
                "valid_count",
                "full_count",
                "valid_membrane",
                "full_membrane",
            ):
                for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
                    row[f"{readout}_{metric_name}"] = m[readout][metric_name]
            rows.append(row)

    runs = pd.DataFrame(rows)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "runs.csv", index=False)
    test = runs[runs.split == "test"].copy()
    group_cols = ["architecture", "hidden_width", "tau_mem_ms"]
    value_cols = [
        column
        for column in test.columns
        if column.endswith("balanced_accuracy")
        or column.endswith("macro_f1")
        or column == "tail_spike_fraction"
    ]
    summary = test.groupby(group_cols)[value_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(config.results_dir / "summary.csv", index=False)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "expected_snn_runs": EXPECTED_SNN_RUNS,
        "architectures": list(ARCHITECTURES),
        "hidden_widths": list(HIDDEN_WIDTHS),
        "tau_mem_ms_values": list(TAU_MEM_MS_VALUES),
        "seeds": list(SEEDS),
        "split_seed": int(base.SPLIT_SEED),
        "fixed_bin_ms": FIXED_MS,
        "event_channels": EVENT_CHANNELS,
        "channel_scale": data.channel_scale.tolist(),
        "primary_objective": "valid_whole_count_ce",
        "state_dynamics": "all padded bins execute; readout masking only",
    }
    _save_json(config.results_dir / "provenance.json", provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 4.0 Fixed250 temporal SNN")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--epochs", type=int, default=EPOCHS)
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--force-retrain", action="store_true")
    baseline_parser = sub.add_parser("baseline")
    baseline_parser.add_argument("--force", action="store_true")
    final = sub.add_parser("finalize")
    final.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = find_repo_root()
    root = results_dir(repo_root)
    data = prepare_binned_data(repo_root)
    if args.command == "run-one":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(
                f"array-task-id {args.array_task_id} outside [0,{len(specs)-1}]"
            )
        spec = specs[args.array_task_id]
        config = Config(
            repo_root=repo_root,
            results_dir=root,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume=not args.force_retrain,
            threads=args.threads,
        )
        result = run_one(spec, data, config)
        print(json.dumps({"completed": spec.key, "best_epoch": result["best_epoch"]}, indent=2))
    elif args.command == "baseline":
        path = root / "baseline" / "fixed250_linear.json"
        if path.exists() and not args.force:
            print(path)
        else:
            payload = run_linear_baseline(data, root)
            print(json.dumps(payload["metrics"], indent=2))
    else:
        config = Config(repo_root=repo_root, results_dir=root, device=args.device)
        finalize(data, config)
        print(root / "summary.csv")


if __name__ == "__main__":
    main()
