from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import experiment_0_1_general_comparison as exp01


EXPERIMENT_ID = "experiment_0_1_3_doc_faithful_regularization"
PROTOCOL_VERSION = "doc_faithful_sae_dense_binary_v1"
BASE_EXPERIMENT_ID = exp01.EXPERIMENT_ID
BASE_PROTOCOL_VERSION = exp01.PROTOCOL_VERSION

DIRECT_FAMILY = exp01.DIRECT_FAMILY
OBJECTIVES = exp01.OBJECTIVES
SEEDS = exp01.SEEDS
DIRECT_ARCHITECTURES = exp01.DIRECT_ARCHITECTURES
HIDDEN_WIDTH = exp01.HIDDEN_WIDTH
OUTPUT_CAP = exp01.OUTPUT_CAP

VARIANT = "binary"
HIDDEN_CAP = 1

REG_TAU = 0.99
REG_LAMBDA_P2 = 0.01
REG_LAMBDA_A1 = 0.1
REG_POTENTIAL_SOURCE = "post_reset_membrane"
REG_POTENTIAL_LAYERS = "all_spiking_layers_including_output"
REG_ACTIVITY_LAYERS = "last_hidden_latent_layer_only"
REG_NORMALIZATION = "doc_faithful_sum_no_neuron_or_layer_normalization"

BATCH_SIZE = exp01.BATCH_SIZE
EPOCHS = exp01.EPOCHS
LR = exp01.LR
WEIGHT_DECAY = exp01.WEIGHT_DECAY


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    objective: str
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{DIRECT_FAMILY}__{self.architecture}__{self.objective}__"
            f"{VARIANT}__hcap{HIDDEN_CAP}__doc_reg_on__seed{self.seed}"
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
        RunSpec(architecture, objective, seed)
        for architecture in DIRECT_ARCHITECTURES
        for objective in OBJECTIVES
        for seed in SEEDS
    ]


def base_spec(spec: RunSpec) -> exp01.RunSpec:
    return exp01.RunSpec(
        DIRECT_FAMILY,
        spec.architecture,
        spec.objective,
        VARIANT,
        HIDDEN_CAP,
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


class DocFaithfulRegularizedSNN(exp01.MultiTauHierarchySNN):
    """Exp0.1 binary direct SNN exposing post-reset membrane trajectories."""

    def __init__(
        self,
        layer_shifts: tuple[tuple[int, ...], ...],
        n_classes: int,
        fs: float,
    ) -> None:
        super().__init__(
            layer_shifts=layer_shifts,
            n_classes=n_classes,
            fs=fs,
            hidden_cap=HIDDEN_CAP,
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
        hidden_spikes: list[list[torch.Tensor]] = [[] for _ in self.hidden_linears]
        hidden_post_reset: list[list[torch.Tensor]] = [
            [] for _ in self.hidden_linears
        ]
        output_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)
        output_spikes: list[torch.Tensor] = []
        output_post_reset: list[torch.Tensor] = []

        for step in range(n_steps):
            current = x[:, step]
            for index, (linear, lif) in enumerate(
                zip(self.hidden_linears, self.hidden_lifs, strict=True)
            ):
                alpha = getattr(self, self._alpha_names[index])
                syn_states[index] = alpha * syn_states[index] + linear(current)
                spike, mem_states[index], _ = lif(
                    syn_states[index], mem_states[index]
                )
                hidden_spikes[index].append(spike)
                hidden_post_reset[index].append(mem_states[index])
                current = spike

            if self.output_linear is None or self.output_lif is None:
                raise RuntimeError("Exp0.1.3 requires a spiking output head")
            spike_out, output_mem, _ = self.output_lif(
                self.output_linear(current), output_mem
            )
            output_spikes.append(spike_out)
            output_post_reset.append(output_mem)

        return {
            "hidden_spikes": tuple(
                torch.stack(parts, dim=1) for parts in hidden_spikes
            ),
            "hidden_post_reset": tuple(
                torch.stack(parts, dim=1) for parts in hidden_post_reset
            ),
            "output_spikes": torch.stack(output_spikes, dim=1),
            "output_post_reset": torch.stack(output_post_reset, dim=1),
        }


def _new_model(spec: RunSpec, data: exp01.exp3.Data) -> DocFaithfulRegularizedSNN:
    return DocFaithfulRegularizedSNN(
        layer_shifts=DIRECT_ARCHITECTURES[spec.architecture],
        n_classes=len(data.labels),
        fs=data.fs,
    )


def make_loaders(
    data: exp01.exp3.Data,
    spec: RunSpec,
    batch_size: int,
    train_shuffle: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    return exp01.make_loaders(data, base_spec(spec), batch_size, train_shuffle)


def _cumulative_endpoint(
    membrane: torch.Tensor,
    lengths: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    if membrane.ndim != 3:
        raise ValueError("membrane trajectory must have shape [B, T, N]")
    batch, n_steps, width = membrane.shape
    valid = exp01.exp50.valid_mask(lengths, n_steps)
    if valid.shape[0] != batch:
        raise ValueError("lengths batch dimension disagrees with trajectory")
    cumulative = torch.zeros(
        batch, width, device=membrane.device, dtype=membrane.dtype
    )
    for step in range(n_steps):
        valid_step = valid[:, step].unsqueeze(-1)
        updated = float(tau) * cumulative + membrane[:, step]
        cumulative = torch.where(valid_step, updated, cumulative)
    return cumulative


def regularization_terms(
    trajectory: dict[str, object],
    lengths: torch.Tensor,
    tau: float = REG_TAU,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Return doc-faithful SAE-Dense P2 and A1 for binary Exp0.1 SNNs.

    P2 uses post-reset membrane, accumulates U(t)=tau*U(t-1)+mem(t), and
    directly sums U(T)^2 over batch, neurons, and every spiking layer,
    including the output layer. A1 is the per-sample time-average binary
    spike count of the last hidden (latent) layer, summed across the batch.
    The only adaptation to the fixed-T source definition is valid-length
    masking/freezing at each sample endpoint so padding cannot alter either
    term.
    """
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")

    hidden_spikes = trajectory.get("hidden_spikes")
    hidden_post_reset = trajectory.get("hidden_post_reset")
    output_post_reset = trajectory.get("output_post_reset")
    if not isinstance(hidden_spikes, tuple) or not hidden_spikes:
        raise TypeError("Expected non-empty hidden_spikes tuple")
    if not isinstance(hidden_post_reset, tuple) or not hidden_post_reset:
        raise TypeError("Expected non-empty hidden_post_reset tuple")
    if len(hidden_spikes) != len(hidden_post_reset):
        raise ValueError("Hidden spike and membrane layer counts must match")
    if not isinstance(output_post_reset, torch.Tensor):
        raise TypeError("Expected output_post_reset tensor")

    potential_trajectories = (*hidden_post_reset, output_post_reset)
    p2_components: list[torch.Tensor] = []
    for membrane in potential_trajectories:
        if not isinstance(membrane, torch.Tensor):
            raise TypeError("Post-reset membrane traces must be tensors")
        endpoint = _cumulative_endpoint(membrane, lengths, tau)
        p2_components.append(endpoint.square().sum())
    p2 = torch.stack(p2_components).sum()

    latent_spikes = hidden_spikes[-1]
    if not isinstance(latent_spikes, torch.Tensor):
        raise TypeError("Last hidden spike trace must be a tensor")
    valid = exp01.exp50.valid_mask(lengths, latent_spikes.shape[1])
    valid_f = valid.to(latent_spikes.dtype).unsqueeze(-1)
    lengths_f = lengths.to(latent_spikes.dtype).clamp_min(1.0)
    per_sample_a1 = (latent_spikes * valid_f).sum(dim=(1, 2)) / lengths_f
    a1 = per_sample_a1.sum()
    return p2, a1, tuple(p2_components)


def _regularization_manifest() -> dict[str, object]:
    return {
        "enabled": True,
        "configuration": "SAE-Dense",
        "binary_only": True,
        "tau": REG_TAU,
        "lambda_p1": 0.0,
        "lambda_p2": REG_LAMBDA_P2,
        "lambda_a1": REG_LAMBDA_A1,
        "lambda_l2": 0.0,
        "potential_term": "P2",
        "potential_source": REG_POTENTIAL_SOURCE,
        "potential_layers": REG_POTENTIAL_LAYERS,
        "potential_reduction": "direct_sum_over_batch_neurons_and_layers",
        "activity_term": "A1",
        "activity_layers": REG_ACTIVITY_LAYERS,
        "activity_signal": "binary_spike_output",
        "activity_reduction": (
            "per_sample_sum_over_latent_neurons_and_valid_timesteps_divided_by_valid_T; "
            "sum_over_batch"
        ),
        "normalization": REG_NORMALIZATION,
        "valid_length_policy": (
            "source assumes fixed T; adaptation updates only valid timesteps and freezes "
            "cumulative potential after each sample endpoint"
        ),
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
        weighted_p2_sum = 0.0
        weighted_a1_sum = 0.0
        p2_component_sums: list[float] | None = None
        n_batches = 0

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
            p2, a1, p2_components = regularization_terms(trajectory, lengths)
            weighted_p2 = REG_LAMBDA_P2 * p2
            weighted_a1 = REG_LAMBDA_A1 * a1
            total_loss = task_loss + weighted_p2 + weighted_a1
            total_loss.backward()
            optimizer.step()

            n_batches += 1
            task_sum += float(task_loss.item())
            p2_sum += float(p2.item())
            a1_sum += float(a1.item())
            weighted_p2_sum += float(weighted_p2.item())
            weighted_a1_sum += float(weighted_a1.item())
            total_sum += float(total_loss.item())
            if p2_component_sums is None:
                p2_component_sums = [0.0] * len(p2_components)
            for index, component in enumerate(p2_components):
                p2_component_sums[index] += float(component.item())

        if n_batches == 0:
            raise RuntimeError("Training loader produced no batches")

        val_metrics = exp01._direct_metrics(model, val_loader, device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        task_mean = task_sum / n_batches
        p2_mean = p2_sum / n_batches
        a1_mean = a1_sum / n_batches
        weighted_p2_mean = weighted_p2_sum / n_batches
        weighted_a1_mean = weighted_a1_sum / n_batches
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_task_loss": task_mean,
            "train_p2": p2_mean,
            "train_a1": a1_mean,
            "train_weighted_p2": weighted_p2_mean,
            "train_weighted_a1": weighted_a1_mean,
            "train_total_loss": total_sum / n_batches,
            "train_reg_to_task_ratio": (
                weighted_p2_mean + weighted_a1_mean
            ) / max(task_mean, 1e-12),
            "val_native_ba": val_ba,
            "val_native_loss": val_loss,
        }
        if p2_component_sums is None:
            raise RuntimeError("Missing P2 component history")
        for index, value in enumerate(p2_component_sums[:-1], start=1):
            row[f"train_p2_hidden_{index}"] = value / n_batches
        row["train_p2_output"] = p2_component_sums[-1] / n_batches
        history.append(row)

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
) -> tuple[DocFaithfulRegularizedSNN, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp0.1.3 checkpoint: {path}")
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
    model: DocFaithfulRegularizedSNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    fs: float,
) -> dict[str, object]:
    n_layers = len(model.hidden_lifs)
    event_totals = [0.0] * n_layers
    active_totals = [0.0] * n_layers
    valid_neuron_steps = [0.0] * n_layers
    post_abs_totals = [0.0] * n_layers
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
            post_reset = trajectory["hidden_post_reset"]
            if not isinstance(spikes, tuple) or not isinstance(post_reset, tuple):
                raise TypeError("Expected hidden trajectories")

            for index, (layer_spikes, layer_post) in enumerate(
                zip(spikes, post_reset, strict=True)
            ):
                valid = exp01.exp50.valid_mask(lengths, layer_spikes.shape[1])
                valid_f = valid.to(layer_spikes.dtype).unsqueeze(-1)
                denom = float(valid.sum().item()) * float(layer_spikes.shape[2])
                valid_neuron_steps[index] += denom
                masked_spikes = layer_spikes * valid_f
                event_totals[index] += float(masked_spikes.sum().item())
                active_totals[index] += float(
                    ((layer_spikes > 0) & valid.unsqueeze(-1)).sum().item()
                )
                post_abs_totals[index] += float(
                    (layer_post.abs() * valid_f).sum().item()
                )
                neuron_event_totals[index] += (
                    masked_spikes.sum(dim=(0, 1)).cpu().double()
                )

    layers: list[dict[str, float | int]] = []
    for index in range(n_layers):
        denom = max(valid_neuron_steps[index], 1.0)
        layers.append(
            {
                "layer": index + 1,
                "events_per_neuron_second": event_totals[index]
                / max(denom / float(fs), 1e-12),
                "active_step_fraction": active_totals[index] / denom,
                "dead_neuron_fraction": float(
                    (neuron_event_totals[index] == 0).double().mean().item()
                ),
                "mean_abs_post_reset_membrane": post_abs_totals[index] / denom,
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
        split: _hidden_diagnostics(model, loader, device, data.fs)
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
            "training_contract": (
                "Exp0.1 architecture, optimizer, LR, epochs, batch size, seeds, objectives, "
                "readout, and checkpoint selection are unchanged; only doc-faithful "
                "SAE-Dense regularization is added"
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
    return {
        "family": DIRECT_FAMILY,
        "architecture": spec.architecture,
        "objective": spec.objective,
        "variant": VARIANT,
        "hidden_cap": HIDDEN_CAP,
        "output_cap": OUTPUT_CAP,
        "regularization": "doc_sae_dense_on",
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
        "test_mean_dead_neuron_fraction": float(
            np.mean([float(layer["dead_neuron_fraction"]) for layer in layers])
        ),
        "test_last_hidden_dead_neuron_fraction": float(
            last["dead_neuron_fraction"]
        ),
        "test_mean_active_step_fraction": float(
            np.mean([float(layer["active_step_fraction"]) for layer in layers])
        ),
        "test_last_hidden_active_step_fraction": float(
            last["active_step_fraction"]
        ),
        "test_mean_abs_post_reset_membrane": float(
            np.mean([float(layer["mean_abs_post_reset_membrane"]) for layer in layers])
        ),
        "test_last_hidden_mean_abs_post_reset_membrane": float(
            last["mean_abs_post_reset_membrane"]
        ),
    }


def _load_frozen_binary_direct_runs(repo_root: Path) -> pd.DataFrame:
    path = base_results_dir(repo_root) / "runs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen Exp0.1 runs.csv; run/finalize Exp0.1 first: {path}"
        )
    runs = pd.read_csv(path)
    direct = runs[
        (runs["family"] == DIRECT_FAMILY)
        & (runs["variant"] == VARIANT)
        & (runs["hidden_cap"] == HIDDEN_CAP)
    ].copy()
    if len(direct) != 30:
        raise ValueError(f"Expected 30 frozen Exp0.1 binary direct runs, found {len(direct)}")
    expected = {
        (spec.architecture, spec.objective, VARIANT, spec.seed)
        for spec in run_specs()
    }
    observed = set(
        direct[["architecture", "objective", "variant", "seed"]]
        .itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError("Frozen Exp0.1 binary direct identities do not match Exp0.1.3")
    direct["regularization"] = "off"
    for column in (
        "test_mean_dead_neuron_fraction",
        "test_last_hidden_dead_neuron_fraction",
        "test_mean_active_step_fraction",
        "test_last_hidden_active_step_fraction",
        "test_mean_abs_post_reset_membrane",
        "test_last_hidden_mean_abs_post_reset_membrane",
    ):
        direct[column] = np.nan
    return direct


def _load_binary_probe_context(repo_root: Path) -> pd.DataFrame:
    path = base_results_dir(repo_root) / "runs.csv"
    runs = pd.read_csv(path)
    context = runs[
        (runs["family"] != DIRECT_FAMILY)
        & (runs["variant"] == VARIANT)
    ].copy()
    return context


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    on_rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing required Exp0.1.3 evaluation: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        on_rows.append(_row_from_payload(spec, payload))

    reg_on = pd.DataFrame(on_rows)
    reg_off = _load_frozen_binary_direct_runs(repo_root)
    runs = pd.concat([reg_off, reg_on], ignore_index=True, sort=False)
    root.mkdir(parents=True, exist_ok=True)
    runs_file = root / "runs.csv"
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

    direct = runs.set_index(
        ["architecture", "objective", "regularization", "seed"]
    )
    paired_rows: list[dict[str, object]] = []
    for spec in run_specs():
        on = direct.loc[
            (spec.architecture, spec.objective, "doc_sae_dense_on", spec.seed)
        ]
        off = direct.loc[(spec.architecture, spec.objective, "off", spec.seed)]
        paired_rows.append(
            {
                "architecture": spec.architecture,
                "objective": spec.objective,
                "variant": VARIANT,
                "hidden_cap": HIDDEN_CAP,
                "seed": spec.seed,
                "off_test_ba": float(off["test_ba"]),
                "on_test_ba": float(on["test_ba"]),
                "delta_test_ba_doc_reg_minus_off": float(
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
                "on_test_mean_dead_neuron_fraction": float(
                    on["test_mean_dead_neuron_fraction"]
                ),
                "on_test_last_hidden_dead_neuron_fraction": float(
                    on["test_last_hidden_dead_neuron_fraction"]
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired_file = root / "paired_regularization_effects.csv"
    paired.to_csv(paired_file, index=False)

    paired_index = paired.set_index(["architecture", "objective", "seed"])
    objective_rows: list[dict[str, object]] = []
    for architecture in DIRECT_ARCHITECTURES:
        for seed in SEEDS:
            timestep = float(
                paired_index.loc[
                    (architecture, "timestep_ce", seed),
                    "delta_test_ba_doc_reg_minus_off",
                ]
            )
            whole = float(
                paired_index.loc[
                    (architecture, "whole_count_ce", seed),
                    "delta_test_ba_doc_reg_minus_off",
                ]
            )
            objective_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "interaction_doc_reg_delta_timestep_minus_whole_count": (
                        timestep - whole
                    ),
                }
            )
    objective_file = root / "regularization_objective_interactions.csv"
    pd.DataFrame(objective_rows).to_csv(objective_file, index=False)

    architecture_rows: list[dict[str, object]] = []
    architecture_pairs = (
        ("short_mid_long", "short_mid", "add_long_layer"),
        ("mid_long", "short_mid", "shift_coverage_longer"),
    )
    for candidate, baseline, comparison in architecture_pairs:
        for objective in OBJECTIVES:
            for seed in SEEDS:
                candidate_delta = float(
                    paired_index.loc[
                        (candidate, objective, seed),
                        "delta_test_ba_doc_reg_minus_off",
                    ]
                )
                baseline_delta = float(
                    paired_index.loc[
                        (baseline, objective, seed),
                        "delta_test_ba_doc_reg_minus_off",
                    ]
                )
                architecture_rows.append(
                    {
                        "comparison": comparison,
                        "candidate": candidate,
                        "baseline": baseline,
                        "objective": objective,
                        "seed": seed,
                        "interaction_doc_reg_delta_candidate_minus_baseline": (
                            candidate_delta - baseline_delta
                        ),
                    }
                )
    architecture_file = root / "regularization_architecture_interactions.csv"
    pd.DataFrame(architecture_rows).to_csv(architecture_file, index=False)

    base_baselines = base_results_dir(repo_root) / "baseline_results.csv"
    if not base_baselines.exists():
        raise FileNotFoundError(
            f"Missing frozen Exp0.1 baseline results: {base_baselines}"
        )
    baselines = pd.read_csv(base_baselines)
    baselines_file = root / "baseline_results.csv"
    baselines.to_csv(baselines_file, index=False)

    probe_context = _load_binary_probe_context(repo_root)
    context_file = root / "binary_probe_context.csv"
    probe_context.to_csv(context_file, index=False)

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
    if not probe_context.empty:
        probe_summary = (
            probe_context.groupby(
                ["family", "architecture", "objective", "variant"],
                dropna=False,
            )["test_ba"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        for row in probe_summary.itertuples(index=False):
            comparison_rows.append(
                {
                    "system": (
                        f"{row.family}:{row.architecture}:{row.objective}:"
                        f"{row.variant}:frozen_context"
                    ),
                    "family": row.family,
                    "architecture": row.architecture,
                    "objective": row.objective,
                    "variant": row.variant,
                    "regularization": "not_applicable",
                    "mean_test_ba": row.mean,
                    "sd_test_ba": row.std,
                    "n": row.count,
                }
            )
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
    comparison_file = root / "comparison_summary.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_file, index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Does the source-document SAE-Dense regularizer improve Exp0.1 binary direct "
            "SNNs when every SNN/training parameter is otherwise unchanged?"
        ),
        "scope": (
            "Exp0.1 direct SNN family only; binary hidden spikes only. The Exp0.1 "
            "SNN+Fixed250 Linear probe is frozen context and is not retrained."
        ),
        "base_experiment": {
            "experiment_id": BASE_EXPERIMENT_ID,
            "protocol_version": BASE_PROTOCOL_VERSION,
            "reuse": "30 frozen binary direct reg-off runs + frozen binary probe/raw context",
        },
        "new_training_run_count": len(run_specs()),
        "regularization": _regularization_manifest(),
        "preserved_training_parameters": {
            "hidden_width": HIDDEN_WIDTH,
            "output_cap": OUTPUT_CAP,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seeds": list(SEEDS),
            "objectives": list(OBJECTIVES),
            "checkpoint_selection": (
                "validation Output WholeCount BA; tie-break normalized WholeCount CE"
            ),
        },
        "direct_architectures": {
            key: [list(shifts) for shifts in value]
            for key, value in DIRECT_ARCHITECTURES.items()
        },
        "variant": {
            "name": VARIANT,
            "hidden_cap": HIDDEN_CAP,
            "output_cap": OUTPUT_CAP,
        },
        "files": {
            "runs": runs_file.name,
            "summary": summary_file.name,
            "baselines": baselines_file.name,
            "binary_probe_context": context_file.name,
            "comparison_summary": comparison_file.name,
            "paired_regularization_effects": paired_file.name,
            "regularization_objective_interactions": objective_file.name,
            "regularization_architecture_interactions": architecture_file.name,
        },
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    return {
        "runs": runs_file,
        "summary": summary_file,
        "baselines": baselines_file,
        "binary_probe_context": context_file,
        "comparison_summary": comparison_file,
        "paired_regularization_effects": paired_file,
        "regularization_objective_interactions": objective_file,
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
        f"regularization=doc-faithful SAE-Dense P2+A1"
    )


if __name__ == "__main__":
    main()
