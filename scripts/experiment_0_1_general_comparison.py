from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from scripts import experiment_3_0_1_single_tau_objective_comparison as exp3
from scripts import experiment_3_0_2_hidden_multitau_architectures as exp302
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50


EXPERIMENT_ID = "experiment_0_1_general_comparison"
PROTOCOL_VERSION = "general_comparison_v1"

DIRECT_FAMILY = "direct_snn"
PROBE_FAMILY = "snn_fixed250_linear"
OBJECTIVES = ("whole_count_ce", "timestep_ce")
VARIANTS: tuple[tuple[str, int], ...] = (
    ("binary", 1),
    ("multi_h", 31),
)
SEEDS = (11, 23, 37, 53, 71)
OUTPUT_CAP = 1
HIDDEN_WIDTH = 128

DIRECT_ARCHITECTURES: dict[str, tuple[tuple[int, ...], ...]] = {
    "short_mid_long": (
        (2, 3),
        (2, 3, 4, 5),
        (2, 3, 4, 5, 6, 7),
    ),
    "short_mid": (
        (2, 3),
        (2, 3, 4, 5),
    ),
    "mid_long": (
        (2, 3, 4, 5),
        (2, 3, 4, 5, 6, 7),
    ),
}
PROBE_ARCHITECTURE = "local_234x2"
PROBE_HIDDEN_SHIFTS: tuple[tuple[int, ...], ...] = (
    (2, 3, 4),
    (2, 3, 4),
)

BATCH_SIZE = exp3.BATCH_SIZE
EPOCHS = exp3.EPOCHS
LR = exp3.LR
WEIGHT_DECAY = 0.0
TAU_MEM_MS = exp3.TAU_MEM_MS
THRESHOLD = exp3.THRESHOLD
SURROGATE_SLOPE = exp3.SURROGATE_SLOPE
FIXED250_MS = 250.0
RELATIVE_BINS = 10


@dataclass(frozen=True)
class RunSpec:
    family: str
    architecture: str
    objective: str
    variant: str
    hidden_cap: int
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.family}__{self.architecture}__{self.objective}__"
            f"{self.variant}__hcap{self.hidden_cap}__seed{self.seed}"
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
    direct = [
        RunSpec(DIRECT_FAMILY, architecture, objective, variant, hidden_cap, seed)
        for architecture in DIRECT_ARCHITECTURES
        for objective in OBJECTIVES
        for variant, hidden_cap in VARIANTS
        for seed in SEEDS
    ]
    probe = [
        RunSpec(PROBE_FAMILY, PROBE_ARCHITECTURE, "timestep_ce", variant, hidden_cap, seed)
        for variant, hidden_cap in VARIANTS
        for seed in SEEDS
    ]
    return direct + probe


def paired_seed(spec: RunSpec, role: str) -> int:
    """Keep initialization/data order paired across objectives and capacities."""
    return exp3.dseed(spec.seed, "exp0_1_general_comparison", role)


def hidden_shifts(spec: RunSpec) -> tuple[tuple[int, ...], ...]:
    if spec.family == DIRECT_FAMILY:
        return DIRECT_ARCHITECTURES[spec.architecture]
    if spec.family == PROBE_FAMILY and spec.architecture == PROBE_ARCHITECTURE:
        return PROBE_HIDDEN_SHIFTS
    raise ValueError(f"Unknown architecture identity: {spec.family}/{spec.architecture}")


def fixed_steps(ms: float, fs: float) -> int:
    return int(np.rint(float(ms) * float(fs) / 1000.0))


class MultiTauHierarchySNN(nn.Module):
    """Raw64 feed-forward multi-tau SNN with optional binary spiking class head."""

    def __init__(
        self,
        layer_shifts: tuple[tuple[int, ...], ...],
        n_classes: int,
        fs: float,
        hidden_cap: int,
        output_spiking: bool,
    ) -> None:
        super().__init__()
        if not layer_shifts:
            raise ValueError("At least one hidden layer is required")
        self.layer_shifts = tuple(tuple(int(value) for value in shifts) for shifts in layer_shifts)
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        self.hidden_cap = int(hidden_cap)
        self.output_spiking = bool(output_spiking)
        beta = exp50.beta_value(self.fs)

        linears: list[nn.Linear] = []
        lifs: list[exp401.MacroMultiSpikeLIF] = []
        input_dim = exp3.EVENT_CHANNELS
        self._alpha_names: list[str] = []
        for index, shifts in enumerate(self.layer_shifts):
            linears.append(nn.Linear(input_dim, HIDDEN_WIDTH, bias=False))
            lifs.append(
                exp401.MacroMultiSpikeLIF(
                    beta=beta,
                    threshold=THRESHOLD,
                    max_spikes_per_dt=self.hidden_cap,
                    surrogate_slope=SURROGATE_SLOPE,
                )
            )
            alpha_name = f"alpha_{index}"
            self.register_buffer(alpha_name, exp50.alpha_vector(HIDDEN_WIDTH, shifts))
            self._alpha_names.append(alpha_name)
            input_dim = HIDDEN_WIDTH

        self.hidden_linears = nn.ModuleList(linears)
        self.hidden_lifs = nn.ModuleList(lifs)
        if self.output_spiking:
            self.output_linear = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=False)
            self.output_lif = exp401.MacroMultiSpikeLIF(
                beta=beta,
                threshold=THRESHOLD,
                max_spikes_per_dt=OUTPUT_CAP,
                surrogate_slope=SURROGATE_SLOPE,
            )
            self.analog_head = None
        else:
            self.output_linear = None
            self.output_lif = None
            self.analog_head = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=True)

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, object]:
        batch, n_steps, channels = x.shape
        if channels != exp3.EVENT_CHANNELS:
            raise ValueError(f"Expected {exp3.EVENT_CHANNELS} input channels, got {channels}")

        syn_states = [
            torch.zeros(batch, HIDDEN_WIDTH, device=x.device, dtype=x.dtype)
            for _ in self.hidden_linears
        ]
        mem_states = [torch.zeros_like(value) for value in syn_states]
        layer_spikes: list[list[torch.Tensor]] = [[] for _ in self.hidden_linears]
        output_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)
        output_spikes: list[torch.Tensor] = []
        analog_logits: list[torch.Tensor] = []

        for step in range(n_steps):
            current = x[:, step]
            for index, (linear, lif) in enumerate(zip(self.hidden_linears, self.hidden_lifs, strict=True)):
                alpha = getattr(self, self._alpha_names[index])
                syn_states[index] = alpha * syn_states[index] + linear(current)
                spike, mem_states[index], _ = lif(syn_states[index], mem_states[index])
                layer_spikes[index].append(spike)
                current = spike

            if self.output_spiking:
                if self.output_linear is None or self.output_lif is None:
                    raise RuntimeError("Missing spiking output modules")
                spike_out, output_mem, _ = self.output_lif(self.output_linear(current), output_mem)
                output_spikes.append(spike_out)
            else:
                if self.analog_head is None:
                    raise RuntimeError("Missing analog head")
                analog_logits.append(self.analog_head(current))

        payload: dict[str, object] = {
            "hidden_spikes": tuple(torch.stack(parts, dim=1) for parts in layer_spikes),
        }
        if self.output_spiking:
            payload["output_spikes"] = torch.stack(output_spikes, dim=1)
        else:
            payload["analog_logits"] = torch.stack(analog_logits, dim=1)
        return payload


def prepare_data(repo_root: Path) -> exp3.Data:
    data = exp3.prepare_data(repo_root)
    if exp3.SPLIT_SEED != 12345:
        raise ValueError(f"Exp0.1 requires split seed 12345, got {exp3.SPLIT_SEED}")
    if fixed_steps(FIXED250_MS, data.fs) != data.bin_steps:
        raise ValueError("Fixed250 bin width disagrees with Exp3 data contract")
    return data


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


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def baseline_path(root: Path) -> Path:
    return root / "raw_baselines.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
) -> dict[str, object]:
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
        val_ba = float(balanced_accuracy_score(val_y, classifier.predict(val_z)))
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError("No linear-probe candidate selected")

    _, selected_C, classifier = best
    return {
        "feature_dim": int(train_x.shape[1]),
        "probe_C": selected_C,
        "train": _classification_metrics(train_y, classifier.predict(train_z)),
        "val": _classification_metrics(val_y, classifier.predict(val_z)),
        "test": _classification_metrics(test_y, classifier.predict(test_z)),
    }


def _direct_metrics(
    model: MultiTauHierarchySNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    ce_sum = 0.0
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
            ce = F.cross_entropy(ce_logits, y)
            n = len(y)
            n_total += n
            ce_sum += float(ce.item()) * n
            labels.append(y.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    metrics = _classification_metrics(np.concatenate(labels), np.concatenate(predictions))
    metrics["loss"] = ce_sum / max(n_total, 1)
    return metrics


def _analog_metrics(
    model: MultiTauHierarchySNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    ce_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for X, y, lengths in data_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            logits_t = trajectory["analog_logits"]
            if not isinstance(logits_t, torch.Tensor):
                raise TypeError("Expected analog_logits tensor")
            segment_logits = exp50.valid_mean(logits_t, lengths)
            ce = F.cross_entropy(segment_logits, y)
            n = len(y)
            n_total += n
            ce_sum += float(ce.item()) * n
            labels.append(y.cpu().numpy())
            predictions.append(segment_logits.argmax(dim=1).cpu().numpy())
    metrics = _classification_metrics(np.concatenate(labels), np.concatenate(predictions))
    metrics["loss"] = ce_sum / max(n_total, 1)
    return metrics


def _last_hidden_event_rate(
    model: MultiTauHierarchySNN,
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


def _new_model(spec: RunSpec, data: exp3.Data) -> MultiTauHierarchySNN:
    return MultiTauHierarchySNN(
        layer_shifts=hidden_shifts(spec),
        n_classes=len(data.labels),
        fs=data.fs,
        hidden_cap=spec.hidden_cap,
        output_spiking=(spec.family == DIRECT_FAMILY),
    )


def train_one(spec: RunSpec, data: exp3.Data, config: Config, force: bool) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp3.seed_all(paired_seed(spec, "model_init"))
    model = _new_model(spec, data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loaders = make_loaders(data, spec, config.batch_size, train_shuffle=True)
    train_loader = loaders["train"]
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
            if spec.family == DIRECT_FAMILY:
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
            else:
                logits_t = trajectory["analog_logits"]
                if not isinstance(logits_t, torch.Tensor):
                    raise TypeError("Expected analog_logits tensor")
                valid = exp50.valid_mask(lengths, logits_t.shape[1])
                targets = y[:, None].expand(len(y), logits_t.shape[1])
                loss = F.cross_entropy(logits_t[valid], targets[valid])
            loss.backward()
            optimizer.step()
            n_total += len(y)
            objective_sum += float(loss.item()) * len(y)

        if spec.family == DIRECT_FAMILY:
            val_metrics = _direct_metrics(model, val_loader, device)
        else:
            val_metrics = _analog_metrics(model, val_loader, device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_objective_loss": objective_sum / max(n_total, 1),
                "val_native_ba": val_ba,
                "val_native_loss": val_loss,
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
            "best_val_native_ba": best_val_ba,
            "best_val_native_loss": best_val_loss,
            "model_state_dict": best_state,
            "labels": data.labels,
            "split": data.split,
            "architecture": architecture_manifest(spec, data.fs, len(data.labels)),
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
) -> tuple[MultiTauHierarchySNN, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp0.1 checkpoint: {path}")
    device = torch.device(config.device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong experiment id in {path}")
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong protocol version in {path}")
    if checkpoint.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint identity mismatch for {spec.key}")
    model = _new_model(spec, data).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _extract_fixed250_last_hidden(
    model: MultiTauHierarchySNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    bin_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in data_loader:
            X = X.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            hidden = trajectory["hidden_spikes"]
            if not isinstance(hidden, tuple):
                raise TypeError("Expected hidden_spikes tuple")
            last = hidden[-1]
            features = exp3.fixed_counts(last, lengths, bin_steps).flatten(start_dim=1)
            matrices.append(features.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(matrices), np.concatenate(labels)


def evaluate_one(
    spec: RunSpec,
    data: exp3.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model, checkpoint = load_model(spec, data, config)
    loaders = make_loaders(data, spec, config.batch_size, train_shuffle=False)

    if spec.family == DIRECT_FAMILY:
        split_metrics = {
            split: _direct_metrics(model, loader, device)
            for split, loader in loaders.items()
        }
        primary = split_metrics
        probe = None
    else:
        native = {
            split: _analog_metrics(model, loader, device)
            for split, loader in loaders.items()
        }
        extracted = {
            split: _extract_fixed250_last_hidden(model, loader, device, data.bin_steps)
            for split, loader in loaders.items()
        }
        probe = _fit_linear_probe(
            extracted["train"][0], extracted["train"][1],
            extracted["val"][0], extracted["val"][1],
            extracted["test"][0], extracted["test"][1],
            exp3.dseed(spec.seed, "exp0_1_fixed250_probe", spec.variant),
        )
        split_metrics = native
        primary = {
            split: probe[split]  # type: ignore[index]
            for split in ("train", "val", "test")
        }

    event_rates = {
        split: _last_hidden_event_rate(model, loader, device, data.fs)
        for split, loader in loaders.items()
    }
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_native_ba": float(checkpoint["best_val_native_ba"]),
        "best_val_native_loss": float(checkpoint["best_val_native_loss"]),
        "architecture": architecture_manifest(spec, data.fs, len(data.labels)),
        "primary_readout": (
            "output_whole_count" if spec.family == DIRECT_FAMILY else "l2_fixed250_ordered_linear"
        ),
        "metrics": primary,
        "native_metrics": split_metrics,
        "fixed250_probe": probe,
        "last_hidden_events_per_neuron_second": event_rates,
        "provenance": {
            "split_seed": int(exp3.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "input_contract": "Raw64 30-channel unsigned weighted events; no pre-SNN pooling or scaling",
            "checkpoint_selection": (
                "validation Output WholeCount BA; tie-break normalized WholeCount CE"
                if spec.family == DIRECT_FAMILY
                else "validation native analog-head BA; tie-break native analog-head CE"
            ),
            "probe_contract": (
                None
                if spec.family == DIRECT_FAMILY
                else "freeze timestep-trained SNN; ordered 250-ms L2 spike counts; train-only StandardScaler; validation-selected LogisticRegression C"
            ),
        },
    }
    _save_json(destination, payload)
    return payload


def run_one(spec: RunSpec, data: exp3.Data, config: Config, force: bool) -> dict[str, object]:
    train_one(spec, data, config, force)
    return evaluate_one(spec, data, config, force)


def _raw_features(
    X: np.ndarray,
    lengths: np.ndarray,
    data: exp3.Data,
    representation: str,
) -> np.ndarray:
    tensor = torch.tensor(X, dtype=torch.float32)
    valid_lengths = torch.tensor(lengths, dtype=torch.long)
    if representation == "fixed250":
        features = exp3.fixed_counts(tensor, valid_lengths, data.bin_steps)
    elif representation == "relative10":
        features = exp3.relative_counts(tensor, valid_lengths, RELATIVE_BINS)
    else:
        raise ValueError(f"Unknown raw representation: {representation}")
    return features.flatten(start_dim=1).numpy()


def run_raw_baselines(repo_root: Path, force: bool = False) -> dict[str, object]:
    root = results_dir(repo_root)
    destination = baseline_path(root)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    data = prepare_data(repo_root)
    partitions = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    results: dict[str, object] = {}
    for representation in ("fixed250", "relative10"):
        built = {
            split: (_raw_features(X, lengths, data, representation), y)
            for split, (X, y, lengths) in partitions.items()
        }
        results[representation] = _fit_linear_probe(
            built["train"][0], built["train"][1],
            built["val"][0], built["val"][1],
            built["test"][0], built["test"][1],
            exp3.dseed(exp3.SPLIT_SEED, "exp0_1_raw_baseline", representation),
        )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "split_seed": int(exp3.SPLIT_SEED),
        "representations": results,
        "protocol": "Raw64 -> ordered aggregation -> flatten -> train-only StandardScaler -> validation-selected LogisticRegression C",
    }
    _save_json(destination, payload)
    return payload


def architecture_manifest(spec: RunSpec, fs: float, n_classes: int) -> dict[str, object]:
    layers = []
    for index, shifts in enumerate(hidden_shifts(spec), start=1):
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
        "tau_mem_ms": TAU_MEM_MS,
        "threshold": THRESHOLD,
        "output": (
            {
                "type": "spiking_class_neurons",
                "neurons": int(n_classes),
                "event_cap": OUTPUT_CAP,
                "readout": "valid whole count",
            }
            if spec.family == DIRECT_FAMILY
            else {
                "type": "temporary_analog_linear_head",
                "neurons": int(n_classes),
                "training_only": True,
                "final_readout": "frozen L2 Fixed250 + retrained Linear",
            }
        ),
    }


def _row_from_payload(spec: RunSpec, payload: dict[str, object]) -> dict[str, object]:
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")
    native = payload["native_metrics"]
    if not isinstance(native, dict):
        raise TypeError("native_metrics must be a dict")
    test = metrics["test"]
    val = metrics["val"]
    train = metrics["train"]
    native_test = native["test"]
    return {
        "family": spec.family,
        "architecture": spec.architecture,
        "objective": spec.objective,
        "variant": spec.variant,
        "hidden_cap": spec.hidden_cap,
        "output_cap": OUTPUT_CAP if spec.family == DIRECT_FAMILY else np.nan,
        "seed": spec.seed,
        "primary_readout": payload["primary_readout"],
        "best_epoch": int(payload["best_epoch"]),
        "train_ba": float(train["balanced_accuracy"]),
        "val_ba": float(val["balanced_accuracy"]),
        "test_ba": float(test["balanced_accuracy"]),
        "test_accuracy": float(test["accuracy"]),
        "test_macro_f1": float(test["macro_f1"]),
        "native_test_ba": float(native_test["balanced_accuracy"]),
        "test_last_hidden_events_per_neuron_second": float(
            payload["last_hidden_events_per_neuron_second"]["test"]  # type: ignore[index]
        ),
    }


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing required Exp0.1 evaluation: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_row_from_payload(spec, payload))

    baseline_file = baseline_path(root)
    if not baseline_file.exists():
        raise FileNotFoundError(f"Missing required raw baseline artifact: {baseline_file}")
    baseline_payload = json.loads(baseline_file.read_text(encoding="utf-8"))

    runs = pd.DataFrame(rows)
    runs_file = root / "runs.csv"
    runs.to_csv(runs_file, index=False)

    summary = (
        runs.groupby(["family", "architecture", "objective", "variant", "primary_readout"])[
            ["test_ba", "test_accuracy", "test_macro_f1", "native_test_ba", "test_last_hidden_events_per_neuron_second"]
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

    indexed = runs.set_index(["family", "architecture", "objective", "variant", "seed"])
    capacity_rows: list[dict[str, object]] = []
    for family, architecture, objective in runs[["family", "architecture", "objective"]].drop_duplicates().itertuples(index=False, name=None):
        for seed in SEEDS:
            multi = float(indexed.loc[(family, architecture, objective, "multi_h", seed), "test_ba"])
            binary = float(indexed.loc[(family, architecture, objective, "binary", seed), "test_ba"])
            capacity_rows.append(
                {
                    "family": family,
                    "architecture": architecture,
                    "objective": objective,
                    "seed": seed,
                    "delta_test_ba_multi_h_minus_binary": multi - binary,
                }
            )
    capacity = pd.DataFrame(capacity_rows)
    capacity_file = root / "paired_capacity_effects.csv"
    capacity.to_csv(capacity_file, index=False)

    objective_rows: list[dict[str, object]] = []
    direct = runs[runs["family"] == DIRECT_FAMILY].set_index(["architecture", "objective", "variant", "seed"])
    for architecture in DIRECT_ARCHITECTURES:
        for variant, _ in VARIANTS:
            for seed in SEEDS:
                timestep = float(direct.loc[(architecture, "timestep_ce", variant, seed), "test_ba"])
                whole = float(direct.loc[(architecture, "whole_count_ce", variant, seed), "test_ba"])
                objective_rows.append(
                    {
                        "architecture": architecture,
                        "variant": variant,
                        "seed": seed,
                        "delta_test_ba_timestep_minus_whole_count": timestep - whole,
                    }
                )
    objective_effects = pd.DataFrame(objective_rows)
    objective_file = root / "paired_objective_effects.csv"
    objective_effects.to_csv(objective_file, index=False)

    architecture_rows: list[dict[str, object]] = []
    architecture_pairs = (
        ("short_mid_long", "short_mid", "add_long_layer"),
        ("mid_long", "short_mid", "shift_coverage_longer"),
    )
    for candidate, baseline, comparison in architecture_pairs:
        for objective in OBJECTIVES:
            for variant, _ in VARIANTS:
                for seed in SEEDS:
                    candidate_ba = float(direct.loc[(candidate, objective, variant, seed), "test_ba"])
                    baseline_ba = float(direct.loc[(baseline, objective, variant, seed), "test_ba"])
                    architecture_rows.append(
                        {
                            "comparison": comparison,
                            "candidate": candidate,
                            "baseline": baseline,
                            "objective": objective,
                            "variant": variant,
                            "seed": seed,
                            "delta_test_ba": candidate_ba - baseline_ba,
                        }
                    )
    architecture_effects = pd.DataFrame(architecture_rows)
    architecture_file = root / "paired_architecture_effects.csv"
    architecture_effects.to_csv(architecture_file, index=False)

    baseline_rows: list[dict[str, object]] = []
    for representation, label in (("fixed250", "Raw Fixed250 + Linear"), ("relative10", "Raw Relative10 + Linear")):
        result = baseline_payload["representations"][representation]
        baseline_rows.append(
            {
                "system": label,
                "representation": representation,
                "feature_dim": int(result["feature_dim"]),
                "probe_C": float(result["probe_C"]),
                "train_ba": float(result["train"]["balanced_accuracy"]),
                "val_ba": float(result["val"]["balanced_accuracy"]),
                "test_ba": float(result["test"]["balanced_accuracy"]),
                "test_accuracy": float(result["test"]["accuracy"]),
                "test_macro_f1": float(result["test"]["macro_f1"]),
            }
        )
    baselines = pd.DataFrame(baseline_rows)
    baselines_file = root / "baseline_results.csv"
    baselines.to_csv(baselines_file, index=False)

    comparison_rows = [
        {
            "system": f"{row.family}:{row.architecture}:{row.objective}:{row.variant}",
            "family": row.family,
            "architecture": row.architecture,
            "objective": row.objective,
            "variant": row.variant,
            "mean_test_ba": row.test_ba_mean,
            "sd_test_ba": row.test_ba_std,
            "n": row.test_ba_count,
        }
        for row in summary.itertuples(index=False)
    ]
    for row in baselines.itertuples(index=False):
        comparison_rows.append(
            {
                "system": row.system,
                "family": "raw_linear_baseline",
                "architecture": row.representation,
                "objective": "posthoc_linear",
                "variant": "deterministic",
                "mean_test_ba": row.test_ba,
                "sd_test_ba": np.nan,
                "n": 1,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison_file = root / "comparison_summary.csv"
    comparison.to_csv(comparison_file, index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": "Can feed-forward hierarchical multi-tau SNNs turn raw spike trains into class evidence that closes the gap to explicit Fixed250/Relative10 temporal decoders?",
        "split_seed": int(exp3.SPLIT_SEED),
        "seeds": list(SEEDS),
        "run_count": len(run_specs()),
        "direct_run_count": 60,
        "probe_run_count": 10,
        "direct_architectures": {key: [list(shifts) for shifts in value] for key, value in DIRECT_ARCHITECTURES.items()},
        "probe_architecture": [list(shifts) for shifts in PROBE_HIDDEN_SHIFTS],
        "objectives": list(OBJECTIVES),
        "variants": [
            {"name": name, "hidden_cap": cap, "output_cap": OUTPUT_CAP}
            for name, cap in VARIANTS
        ],
        "primary_readouts": {
            DIRECT_FAMILY: "Output WholeCount",
            PROBE_FAMILY: "frozen L2 Fixed250 ordered + retrained Linear",
            "raw_baselines": ["Raw Fixed250 + Linear", "Raw Relative10 + Linear"],
        },
        "files": {
            "runs": runs_file.name,
            "summary": summary_file.name,
            "baselines": baselines_file.name,
            "comparison_summary": comparison_file.name,
            "paired_capacity_effects": capacity_file.name,
            "paired_objective_effects": objective_file.name,
            "paired_architecture_effects": architecture_file.name,
        },
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    return {
        "runs": runs_file,
        "summary": summary_file,
        "baselines": baselines_file,
        "comparison_summary": comparison_file,
        "paired_capacity_effects": capacity_file,
        "paired_objective_effects": objective_file,
        "paired_architecture_effects": architecture_file,
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

    baseline_parser = subparsers.add_parser("run-baselines")
    baseline_parser.add_argument("--force", action="store_true")

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
    repo_root = exp3.find_repo_root()
    if args.command == "run-baselines":
        payload = run_raw_baselines(repo_root, args.force)
        for name, result in payload["representations"].items():
            print(f"{name}: test_BA={result['test']['balanced_accuracy']:.6f}")
        return
    if args.command == "finalize":
        for name, path in finalize_experiment(repo_root).items():
            print(f"{name}: {path}")
        return

    specs = run_specs()
    if args.array_task_id < 0 or args.array_task_id >= len(specs):
        raise IndexError(f"array task {args.array_task_id} outside [0, {len(specs) - 1}]")
    spec = specs[args.array_task_id]
    data = prepare_data(repo_root)
    config = _config_from_args(args, repo_root)
    payload = run_one(spec, data, config, args.force)
    test = payload["metrics"]["test"]
    print(
        f"completed {spec.key}: readout={payload['primary_readout']} "
        f"test_BA={test['balanced_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
