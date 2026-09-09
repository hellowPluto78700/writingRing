from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import experiment_0_1_general_comparison as exp01


EXPERIMENT_ID = "experiment_0_1_2_regularized_general_comparison"
PROTOCOL_VERSION = "regularized_general_comparison_v1"
BASE_EXPERIMENT_ID = exp01.EXPERIMENT_ID
BASE_PROTOCOL_VERSION = exp01.PROTOCOL_VERSION

DIRECT_FAMILY = exp01.DIRECT_FAMILY
OBJECTIVES = exp01.OBJECTIVES
VARIANTS = exp01.VARIANTS
SEEDS = exp01.SEEDS
DIRECT_ARCHITECTURES = exp01.DIRECT_ARCHITECTURES
HIDDEN_WIDTH = exp01.HIDDEN_WIDTH
OUTPUT_CAP = exp01.OUTPUT_CAP

REG_TAU = 0.99
REG_LAMBDA_P2 = 0.01
REG_LAMBDA_A1 = 0.1
REG_POTENTIAL_SOURCE = "pre_reset_membrane"
REG_ACTIVITY_NORMALIZATION = "event_count_divided_by_hidden_cap"

BATCH_SIZE = exp01.BATCH_SIZE
EPOCHS = exp01.EPOCHS
LR = exp01.LR
WEIGHT_DECAY = exp01.WEIGHT_DECAY


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    objective: str
    variant: str
    hidden_cap: int
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{DIRECT_FAMILY}__{self.architecture}__{self.objective}__"
            f"{self.variant}__hcap{self.hidden_cap}__reg_on__seed{self.seed}"
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


def base_results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / BASE_EXPERIMENT_ID / BASE_PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(architecture, objective, variant, hidden_cap, seed)
        for architecture in DIRECT_ARCHITECTURES
        for objective in OBJECTIVES
        for variant, hidden_cap in VARIANTS
        for seed in SEEDS
    ]


def base_spec(spec: RunSpec) -> exp01.RunSpec:
    return exp01.RunSpec(
        DIRECT_FAMILY,
        spec.architecture,
        spec.objective,
        spec.variant,
        spec.hidden_cap,
        spec.seed,
    )


def paired_seed(spec: RunSpec, role: str) -> int:
    return exp01.paired_seed(base_spec(spec), role)


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class RegularizedMultiTauHierarchySNN(exp01.MultiTauHierarchySNN):
    """Exp0.1 direct SNN with hidden pre-reset traces for regularization."""

    def __init__(
        self,
        layer_shifts: tuple[tuple[int, ...], ...],
        n_classes: int,
        fs: float,
        hidden_cap: int,
    ) -> None:
        super().__init__(
            layer_shifts=layer_shifts,
            n_classes=n_classes,
            fs=fs,
            hidden_cap=hidden_cap,
            output_spiking=True,
        )

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, object]:
        batch, n_steps, channels = x.shape
        if channels != exp01.exp3.EVENT_CHANNELS:
            raise ValueError(
                f"Expected {exp01.exp3.EVENT_CHANNELS} input channels, got {channels}"
            )

        syn_states = [
            torch.zeros(batch, HIDDEN_WIDTH, device=x.device, dtype=x.dtype)
            for _ in self.hidden_linears
        ]
        mem_states = [torch.zeros_like(value) for value in syn_states]
        layer_spikes: list[list[torch.Tensor]] = [[] for _ in self.hidden_linears]
        layer_pre_reset: list[list[torch.Tensor]] = [[] for _ in self.hidden_linears]
        output_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)
        output_spikes: list[torch.Tensor] = []

        for step in range(n_steps):
            current = x[:, step]
            for index, (linear, lif) in enumerate(
                zip(self.hidden_linears, self.hidden_lifs, strict=True)
            ):
                alpha = getattr(self, self._alpha_names[index])
                syn_states[index] = alpha * syn_states[index] + linear(current)
                spike, mem_states[index], pre_reset = lif(
                    syn_states[index], mem_states[index]
                )
                layer_spikes[index].append(spike)
                layer_pre_reset[index].append(pre_reset)
                current = spike

            if self.output_linear is None or self.output_lif is None:
                raise RuntimeError("Exp0.1.2 requires a spiking output head")
            spike_out, output_mem, _ = self.output_lif(
                self.output_linear(current), output_mem
            )
            output_spikes.append(spike_out)

        return {
            "hidden_spikes": tuple(
                torch.stack(parts, dim=1) for parts in layer_spikes
            ),
            "hidden_pre_reset": tuple(
                torch.stack(parts, dim=1) for parts in layer_pre_reset
            ),
            "output_spikes": torch.stack(output_spikes, dim=1),
        }


def _new_model(spec: RunSpec, data: exp01.exp3.Data) -> RegularizedMultiTauHierarchySNN:
    return RegularizedMultiTauHierarchySNN(
        layer_shifts=DIRECT_ARCHITECTURES[spec.architecture],
        n_classes=len(data.labels),
        fs=data.fs,
        hidden_cap=spec.hidden_cap,
    )


def make_loaders(
    data: exp01.exp3.Data,
    spec: RunSpec,
    batch_size: int,
    train_shuffle: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    return exp01.make_loaders(data, base_spec(spec), batch_size, train_shuffle)


def regularization_terms(
    trajectory: dict[str, object],
    lengths: torch.Tensor,
    hidden_cap: int,
    tau: float = REG_TAU,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return layer-mean P2 and A1 over valid timesteps only.

    P2 uses the cap-independent pre-reset membrane trace. The cumulative
    potential freezes after each sample's valid endpoint. A1 uses hidden
    event count / hidden_cap so binary and multi-H live on the same [0, 1]
    per-step activity scale.
    """
    spikes = trajectory.get("hidden_spikes")
    pre_reset = trajectory.get("hidden_pre_reset")
    if not isinstance(spikes, tuple) or not isinstance(pre_reset, tuple):
        raise TypeError("Expected hidden_spikes and hidden_pre_reset tuples")
    if len(spikes) != len(pre_reset) or not spikes:
        raise ValueError("Hidden trace tuples must be non-empty and aligned")
    if hidden_cap < 1:
        raise ValueError("hidden_cap must be >= 1")
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")

    p2_layers: list[torch.Tensor] = []
    a1_layers: list[torch.Tensor] = []
    for layer_spikes, layer_pre in zip(spikes, pre_reset, strict=True):
        if not isinstance(layer_spikes, torch.Tensor) or not isinstance(
            layer_pre, torch.Tensor
        ):
            raise TypeError("Hidden traces must contain tensors")
        if layer_spikes.shape != layer_pre.shape:
            raise ValueError("Hidden spike and membrane traces must have equal shapes")
        batch, n_steps, width = layer_pre.shape
        valid = exp01.exp50.valid_mask(lengths, n_steps)
        if valid.shape[0] != batch:
            raise ValueError("lengths batch dimension disagrees with trajectory")

        cumulative = torch.zeros(
            batch, width, device=layer_pre.device, dtype=layer_pre.dtype
        )
        for step in range(n_steps):
            valid_step = valid[:, step].unsqueeze(-1)
            updated = float(tau) * cumulative + layer_pre[:, step]
            cumulative = torch.where(valid_step, updated, cumulative)
        p2_layers.append(cumulative.square().mean())

        valid_f = valid.to(layer_spikes.dtype).unsqueeze(-1)
        denominator = valid_f.sum() * float(width)
        if float(denominator.detach().item()) <= 0.0:
            raise ValueError("Regularization requires at least one valid timestep")
        normalized_activity = layer_spikes / float(hidden_cap)
        a1_layers.append((normalized_activity * valid_f).sum() / denominator)

    return torch.stack(p2_layers).mean(), torch.stack(a1_layers).mean()


def _regularization_manifest() -> dict[str, object]:
    return {
        "enabled": True,
        "potential_term": "P2",
        "potential_source": REG_POTENTIAL_SOURCE,
        "potential_layers": "all hidden spiking layers only",
        "activity_term": "A1",
        "activity_layers": "all hidden spiking layers only",
        "activity_normalization": REG_ACTIVITY_NORMALIZATION,
        "output_layer_regularized": False,
        "tau": REG_TAU,
        "lambda_p2": REG_LAMBDA_P2,
        "lambda_a1": REG_LAMBDA_A1,
        "normalization": "mean over neurons within each layer, then mean over layers",
        "valid_length_policy": "update only on valid timesteps; freeze cumulative potential after endpoint",
    }


def train_one(
    spec: RunSpec,
    data: exp01.exp3.Data,
    config: Config,
    force: bool,
) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp01.exp3.seed_all(paired_seed(spec, "model_init"))
    model = _new_model(spec, data).to(device)
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
        task_sum = 0.0
        p2_sum = 0.0
        a1_sum = 0.0
        total_sum = 0.0
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

            task_loss = exp01.exp50.objective_loss(
                output_spikes,
                lengths,
                y,
                spec.objective,
                OUTPUT_CAP,
                data.fs,
            )
            p2, a1 = regularization_terms(trajectory, lengths, spec.hidden_cap)
            total_loss = task_loss + REG_LAMBDA_P2 * p2 + REG_LAMBDA_A1 * a1
            total_loss.backward()
            optimizer.step()

            n = len(y)
            n_total += n
            task_sum += float(task_loss.item()) * n
            p2_sum += float(p2.item()) * n
            a1_sum += float(a1.item()) * n
            total_sum += float(total_loss.item()) * n

        val_metrics = exp01._direct_metrics(model, val_loader, device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        task_mean = task_sum / max(n_total, 1)
        p2_mean = p2_sum / max(n_total, 1)
        a1_mean = a1_sum / max(n_total, 1)
        reg_weighted = REG_LAMBDA_P2 * p2_mean + REG_LAMBDA_A1 * a1_mean
        history.append(
            {
                "epoch": epoch,
                "train_task_loss": task_mean,
                "train_p2": p2_mean,
                "train_a1": a1_mean,
                "train_weighted_p2": REG_LAMBDA_P2 * p2_mean,
                "train_weighted_a1": REG_LAMBDA_A1 * a1_mean,
                "train_total_loss": total_sum / max(n_total, 1),
                "train_reg_to_task_ratio": reg_weighted / max(task_mean, 1e-12),
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
            "regularization": _regularization_manifest(),
            "best_epoch": best_epoch,
            "best_val_native_ba": best_val_ba,
            "best_val_native_loss": best_val_loss,
            "model_state_dict": best_state,
            "labels": data.labels,
            "split": data.split,
            "architecture": exp01.architecture_manifest(
                base_spec(spec), data.fs, len(data.labels)
            ),
        },
        destination,
    )
    history_file = history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: exp01.exp3.Data,
    config: Config,
) -> tuple[RegularizedMultiTauHierarchySNN, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp0.1.2 checkpoint: {path}")
    device = torch.device(config.device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong experiment id in {path}")
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong protocol version in {path}")
    if checkpoint.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint identity mismatch for {spec.key}")
    if checkpoint.get("regularization") != _regularization_manifest():
        raise ValueError(f"Regularization contract mismatch for {spec.key}")
    model = _new_model(spec, data).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _hidden_diagnostics(
    model: RegularizedMultiTauHierarchySNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    fs: float,
    hidden_cap: int,
) -> dict[str, object]:
    n_layers = len(model.hidden_lifs)
    event_totals = [0.0] * n_layers
    normalized_event_totals = [0.0] * n_layers
    active_totals = [0.0] * n_layers
    cap_totals = [0.0] * n_layers
    valid_neuron_steps = [0.0] * n_layers
    pre_abs_totals = [0.0] * n_layers
    neuron_event_totals = [
        torch.zeros(HIDDEN_WIDTH, dtype=torch.float64) for _ in range(n_layers)
    ]

    model.eval()
    with torch.no_grad():
        for X, _, lengths in data_loader:
            X = X.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            spikes = trajectory["hidden_spikes"]
            pre_reset = trajectory["hidden_pre_reset"]
            if not isinstance(spikes, tuple) or not isinstance(pre_reset, tuple):
                raise TypeError("Expected hidden traces")

            for index, (layer_spikes, layer_pre) in enumerate(
                zip(spikes, pre_reset, strict=True)
            ):
                valid = exp01.exp50.valid_mask(lengths, layer_spikes.shape[1])
                valid_f = valid.to(layer_spikes.dtype).unsqueeze(-1)
                n_valid = float(valid.sum().item()) * float(layer_spikes.shape[2])
                valid_neuron_steps[index] += n_valid
                masked_spikes = layer_spikes * valid_f
                event_totals[index] += float(masked_spikes.sum().item())
                normalized_event_totals[index] += float(
                    (masked_spikes / float(hidden_cap)).sum().item()
                )
                active_totals[index] += float(
                    ((layer_spikes > 0) & valid.unsqueeze(-1)).sum().item()
                )
                cap_totals[index] += float(
                    ((layer_spikes >= hidden_cap) & valid.unsqueeze(-1)).sum().item()
                )
                pre_abs_totals[index] += float(
                    (layer_pre.abs() * valid_f).sum().item()
                )
                neuron_event_totals[index] += (
                    masked_spikes.sum(dim=(0, 1)).cpu().double()
                )

    layers: list[dict[str, float | int]] = []
    for index in range(n_layers):
        denom = max(valid_neuron_steps[index], 1.0)
        neuron_seconds = denom / float(fs)
        layers.append(
            {
                "layer": index + 1,
                "mean_events_per_neuron_step": event_totals[index] / denom,
                "mean_normalized_activity_per_neuron_step": (
                    normalized_event_totals[index] / denom
                ),
                "events_per_neuron_second": event_totals[index]
                / max(neuron_seconds, 1e-12),
                "active_step_fraction": active_totals[index] / denom,
                "fraction_at_cap": cap_totals[index] / denom,
                "dead_neuron_fraction": float(
                    (neuron_event_totals[index] == 0).double().mean().item()
                ),
                "mean_abs_pre_reset_membrane": pre_abs_totals[index] / denom,
            }
        )
    return {"layers": layers}


def evaluate_one(
    spec: RunSpec,
    data: exp01.exp3.Data,
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
    metrics = {
        split: exp01._direct_metrics(model, loader, device)
        for split, loader in loaders.items()
    }
    event_rates = {
        split: exp01._last_hidden_event_rate(model, loader, device, data.fs)
        for split, loader in loaders.items()
    }
    diagnostics = {
        split: _hidden_diagnostics(model, loader, device, data.fs, spec.hidden_cap)
        for split, loader in loaders.items()
    }
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "regularization": _regularization_manifest(),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_native_ba": float(checkpoint["best_val_native_ba"]),
        "best_val_native_loss": float(checkpoint["best_val_native_loss"]),
        "architecture": exp01.architecture_manifest(
            base_spec(spec), data.fs, len(data.labels)
        ),
        "primary_readout": "output_whole_count",
        "metrics": metrics,
        "native_metrics": metrics,
        "last_hidden_events_per_neuron_second": event_rates,
        "hidden_diagnostics": diagnostics,
        "provenance": {
            "base_experiment_id": BASE_EXPERIMENT_ID,
            "base_protocol_version": BASE_PROTOCOL_VERSION,
            "paired_reg_off_key": base_spec(spec).key,
            "split_seed": int(exp01.exp3.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "input_contract": (
                "identical to Exp0.1 direct SNN: Raw64 30-channel unsigned weighted events"
            ),
            "checkpoint_selection": (
                "validation Output WholeCount BA; tie-break normalized WholeCount CE"
            ),
        },
    }
    _save_json(destination, payload)
    return payload


def run_one(
    spec: RunSpec,
    data: exp01.exp3.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    train_one(spec, data, config, force)
    return evaluate_one(spec, data, config, force)


def _row_from_payload(spec: RunSpec, payload: dict[str, object]) -> dict[str, object]:
    metrics = payload["metrics"]
    diagnostics = payload["hidden_diagnostics"]
    if not isinstance(metrics, dict) or not isinstance(diagnostics, dict):
        raise TypeError("Malformed evaluation payload")
    train = metrics["train"]
    val = metrics["val"]
    test = metrics["test"]
    test_diag = diagnostics["test"]
    if not isinstance(test_diag, dict):
        raise TypeError("Malformed test diagnostics")
    layers = test_diag["layers"]
    if not isinstance(layers, list) or not layers:
        raise TypeError("Missing hidden layer diagnostics")
    last = layers[-1]
    mean_dead = float(
        np.mean([float(layer["dead_neuron_fraction"]) for layer in layers])
    )
    mean_active = float(
        np.mean([float(layer["active_step_fraction"]) for layer in layers])
    )
    mean_at_cap = float(
        np.mean([float(layer["fraction_at_cap"]) for layer in layers])
    )
    return {
        "family": DIRECT_FAMILY,
        "architecture": spec.architecture,
        "objective": spec.objective,
        "variant": spec.variant,
        "hidden_cap": spec.hidden_cap,
        "output_cap": OUTPUT_CAP,
        "regularization": "on",
        "seed": spec.seed,
        "primary_readout": "output_whole_count",
        "best_epoch": int(payload["best_epoch"]),
        "train_ba": float(train["balanced_accuracy"]),
        "val_ba": float(val["balanced_accuracy"]),
        "test_ba": float(test["balanced_accuracy"]),
        "test_accuracy": float(test["accuracy"]),
        "test_macro_f1": float(test["macro_f1"]),
        "native_test_ba": float(test["balanced_accuracy"]),
        "test_last_hidden_events_per_neuron_second": float(
            payload["last_hidden_events_per_neuron_second"]["test"]
        ),
        "test_mean_dead_neuron_fraction": mean_dead,
        "test_mean_active_step_fraction": mean_active,
        "test_mean_fraction_at_cap": mean_at_cap,
        "test_last_hidden_dead_neuron_fraction": float(
            last["dead_neuron_fraction"]
        ),
        "test_last_hidden_active_step_fraction": float(
            last["active_step_fraction"]
        ),
        "test_last_hidden_fraction_at_cap": float(last["fraction_at_cap"]),
    }


def _load_frozen_base_runs(repo_root: Path) -> pd.DataFrame:
    path = base_results_dir(repo_root) / "runs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen Exp0.1 runs.csv; run/finalize Exp0.1 first: {path}"
        )
    runs = pd.read_csv(path)
    direct = runs[runs["family"] == DIRECT_FAMILY].copy()
    if len(direct) != 60:
        raise ValueError(f"Expected 60 frozen Exp0.1 direct runs, found {len(direct)}")
    expected = {
        (s.architecture, s.objective, s.variant, s.seed)
        for s in run_specs()
    }
    observed = set(
        direct[["architecture", "objective", "variant", "seed"]]
        .itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError("Frozen Exp0.1 direct-run identity does not match Exp0.1.2")
    direct["regularization"] = "off"
    for column in (
        "test_mean_dead_neuron_fraction",
        "test_mean_active_step_fraction",
        "test_mean_fraction_at_cap",
        "test_last_hidden_dead_neuron_fraction",
        "test_last_hidden_active_step_fraction",
        "test_last_hidden_fraction_at_cap",
    ):
        direct[column] = np.nan
    return direct


def _load_frozen_context_runs(repo_root: Path) -> pd.DataFrame:
    path = base_results_dir(repo_root) / "runs.csv"
    runs = pd.read_csv(path)
    context = runs[runs["family"] != DIRECT_FAMILY].copy()
    context["regularization"] = "not_applicable"
    for column in (
        "test_mean_dead_neuron_fraction",
        "test_mean_active_step_fraction",
        "test_mean_fraction_at_cap",
        "test_last_hidden_dead_neuron_fraction",
        "test_last_hidden_active_step_fraction",
        "test_last_hidden_fraction_at_cap",
    ):
        context[column] = np.nan
    return context


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    on_rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing required Exp0.1.2 evaluation: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        on_rows.append(_row_from_payload(spec, payload))

    reg_on = pd.DataFrame(on_rows)
    reg_off = _load_frozen_base_runs(repo_root)
    context = _load_frozen_context_runs(repo_root)
    runs = pd.concat([reg_off, reg_on, context], ignore_index=True, sort=False)
    runs_file = root / "runs.csv"
    root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(runs_file, index=False)

    summary = (
        runs.groupby(
            [
                "family",
                "architecture",
                "objective",
                "variant",
                "regularization",
                "primary_readout",
            ],
            dropna=False,
        )[
            [
                "test_ba",
                "test_accuracy",
                "test_macro_f1",
                "native_test_ba",
                "test_last_hidden_events_per_neuron_second",
            ]
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

    direct = runs[runs["family"] == DIRECT_FAMILY].set_index(
        ["architecture", "objective", "variant", "regularization", "seed"]
    )
    paired_rows: list[dict[str, object]] = []
    for spec in run_specs():
        on = direct.loc[
            (spec.architecture, spec.objective, spec.variant, "on", spec.seed)
        ]
        off = direct.loc[
            (spec.architecture, spec.objective, spec.variant, "off", spec.seed)
        ]
        paired_rows.append(
            {
                "architecture": spec.architecture,
                "objective": spec.objective,
                "variant": spec.variant,
                "hidden_cap": spec.hidden_cap,
                "seed": spec.seed,
                "off_test_ba": float(off["test_ba"]),
                "on_test_ba": float(on["test_ba"]),
                "delta_test_ba_reg_on_minus_off": float(
                    on["test_ba"] - off["test_ba"]
                ),
                "off_last_hidden_events_per_neuron_second": float(
                    off["test_last_hidden_events_per_neuron_second"]
                ),
                "on_last_hidden_events_per_neuron_second": float(
                    on["test_last_hidden_events_per_neuron_second"]
                ),
                "delta_last_hidden_events_per_neuron_second": float(
                    on["test_last_hidden_events_per_neuron_second"]
                    - off["test_last_hidden_events_per_neuron_second"]
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired_file = root / "paired_regularization_effects.csv"
    paired.to_csv(paired_file, index=False)

    paired_index = paired.set_index(
        ["architecture", "objective", "variant", "seed"]
    )
    objective_rows: list[dict[str, object]] = []
    for architecture in DIRECT_ARCHITECTURES:
        for variant, _ in VARIANTS:
            for seed in SEEDS:
                timestep = float(
                    paired_index.loc[
                        (architecture, "timestep_ce", variant, seed),
                        "delta_test_ba_reg_on_minus_off",
                    ]
                )
                whole = float(
                    paired_index.loc[
                        (architecture, "whole_count_ce", variant, seed),
                        "delta_test_ba_reg_on_minus_off",
                    ]
                )
                objective_rows.append(
                    {
                        "architecture": architecture,
                        "variant": variant,
                        "seed": seed,
                        "interaction_reg_delta_timestep_minus_whole_count": (
                            timestep - whole
                        ),
                    }
                )
    objective_interactions = pd.DataFrame(objective_rows)
    objective_file = root / "regularization_objective_interactions.csv"
    objective_interactions.to_csv(objective_file, index=False)

    capacity_rows: list[dict[str, object]] = []
    for architecture in DIRECT_ARCHITECTURES:
        for objective in OBJECTIVES:
            for seed in SEEDS:
                multi = float(
                    paired_index.loc[
                        (architecture, objective, "multi_h", seed),
                        "delta_test_ba_reg_on_minus_off",
                    ]
                )
                binary = float(
                    paired_index.loc[
                        (architecture, objective, "binary", seed),
                        "delta_test_ba_reg_on_minus_off",
                    ]
                )
                capacity_rows.append(
                    {
                        "architecture": architecture,
                        "objective": objective,
                        "seed": seed,
                        "interaction_reg_delta_multi_h_minus_binary": (
                            multi - binary
                        ),
                    }
                )
    capacity_interactions = pd.DataFrame(capacity_rows)
    capacity_file = root / "regularization_capacity_interactions.csv"
    capacity_interactions.to_csv(capacity_file, index=False)

    architecture_rows: list[dict[str, object]] = []
    architecture_pairs = (
        ("short_mid_long", "short_mid", "add_long_layer"),
        ("mid_long", "short_mid", "shift_coverage_longer"),
    )
    for candidate, baseline, comparison in architecture_pairs:
        for objective in OBJECTIVES:
            for variant, _ in VARIANTS:
                for seed in SEEDS:
                    candidate_delta = float(
                        paired_index.loc[
                            (candidate, objective, variant, seed),
                            "delta_test_ba_reg_on_minus_off",
                        ]
                    )
                    baseline_delta = float(
                        paired_index.loc[
                            (baseline, objective, variant, seed),
                            "delta_test_ba_reg_on_minus_off",
                        ]
                    )
                    architecture_rows.append(
                        {
                            "comparison": comparison,
                            "candidate": candidate,
                            "baseline": baseline,
                            "objective": objective,
                            "variant": variant,
                            "seed": seed,
                            "interaction_reg_delta_candidate_minus_baseline": (
                                candidate_delta - baseline_delta
                            ),
                        }
                    )
    architecture_interactions = pd.DataFrame(architecture_rows)
    architecture_file = root / "regularization_architecture_interactions.csv"
    architecture_interactions.to_csv(architecture_file, index=False)

    base_baselines = base_results_dir(repo_root) / "baseline_results.csv"
    if not base_baselines.exists():
        raise FileNotFoundError(
            f"Missing frozen Exp0.1 baseline results: {base_baselines}"
        )
    baselines = pd.read_csv(base_baselines)
    baselines_file = root / "baseline_results.csv"
    baselines.to_csv(baselines_file, index=False)

    comparison_rows = [
        {
            "system": (
                f"{row.family}:{row.architecture}:{row.objective}:"
                f"{row.variant}:reg={row.regularization}"
            ),
            "family": row.family,
            "architecture": row.architecture,
            "objective": row.objective,
            "variant": row.variant,
            "regularization": row.regularization,
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
                "regularization": "not_applicable",
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
        "question": (
            "Does hidden-state P2+A1 regularization improve direct multi-tau SNN "
            "optimization without changing architecture, objective, input, readout, or split?"
        ),
        "base_experiment": {
            "experiment_id": BASE_EXPERIMENT_ID,
            "protocol_version": BASE_PROTOCOL_VERSION,
            "reuse": (
                "60 frozen direct reg-off runs + frozen probe/context and raw baselines"
            ),
        },
        "new_training_run_count": len(run_specs()),
        "regularization": _regularization_manifest(),
        "seeds": list(SEEDS),
        "objectives": list(OBJECTIVES),
        "direct_architectures": {
            key: [list(shifts) for shifts in value]
            for key, value in DIRECT_ARCHITECTURES.items()
        },
        "variants": [
            {"name": name, "hidden_cap": cap, "output_cap": OUTPUT_CAP}
            for name, cap in VARIANTS
        ],
        "files": {
            "runs": runs_file.name,
            "summary": summary_file.name,
            "baselines": baselines_file.name,
            "comparison_summary": comparison_file.name,
            "paired_regularization_effects": paired_file.name,
            "regularization_objective_interactions": objective_file.name,
            "regularization_capacity_interactions": capacity_file.name,
            "regularization_architecture_interactions": architecture_file.name,
        },
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    return {
        "runs": runs_file,
        "summary": summary_file,
        "baselines": baselines_file,
        "comparison_summary": comparison_file,
        "paired_regularization_effects": paired_file,
        "regularization_objective_interactions": objective_file,
        "regularization_capacity_interactions": capacity_file,
        "regularization_architecture_interactions": architecture_file,
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
    repo_root = exp01.exp3.find_repo_root()
    if args.command == "finalize":
        for name, path in finalize_experiment(repo_root).items():
            print(f"{name}: {path}")
        return

    specs = run_specs()
    if args.array_task_id < 0 or args.array_task_id >= len(specs):
        raise IndexError(
            f"array task {args.array_task_id} outside [0, {len(specs) - 1}]"
        )
    spec = specs[args.array_task_id]
    data = exp01.prepare_data(repo_root)
    config = _config_from_args(args, repo_root)
    payload = run_one(spec, data, config, args.force)
    test = payload["metrics"]["test"]
    print(
        f"completed {spec.key}: test_BA={test['balanced_accuracy']:.6f} "
        f"regularization=P2+A1 hidden-only"
    )


if __name__ == "__main__":
    main()
