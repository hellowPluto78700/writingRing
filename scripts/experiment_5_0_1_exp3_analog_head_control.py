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

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_0_4_l2_width_representation_capacity as exp304
from scripts import experiment_3_0_5_frozen_representation_accessibility as exp305
from scripts import experiment_5_0_local_evidence_objectives as exp50


EXPERIMENT_ID = "experiment_5_0_1_exp3_analog_head_control"
PROTOCOL_VERSION = "analog_head_three_way_control_v3"
OBJECTIVE = "timestep_ce"
EXP3_EXACT = "exp3_exact_analog"
EXP3_EXP5_STREAM = "exp3_synaptic_exp5stream_analog"
MACRO_EXP5_STREAM = "exp5_macro_exp5stream_analog"
CONDITIONS = (EXP3_EXACT, EXP3_EXP5_STREAM, MACRO_EXP5_STREAM)
SEEDS = base.SEEDS
WIDTH = 128
L1_SHIFTS = (2, 3, 4)
L2_SHIFTS = (2, 3, 4)
PROBE_TYPES = exp305.PROBE_TYPES
PRIMARY_PROBES = (
    "full_count",
    "fixed250_ordered",
    "relative10_ordered",
)
EPOCHS = base.EPOCHS
BATCH_SIZE = base.BATCH_SIZE
LR = base.LR
EXPECTED_RUNS = len(CONDITIONS) * len(SEEDS)


@dataclass(frozen=True)
class RunSpec:
    condition: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.condition}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
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
    return [RunSpec(condition, seed) for condition in CONDITIONS for seed in SEEDS]


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_data(repo_root: Path) -> base.Data:
    return base.prepare_data(repo_root)


class Exp5MacroBinaryAnalogNet(nn.Module):
    """Exp5 binary hidden dynamics with an Exp3-style analog timestep head."""

    def __init__(self, n_classes: int, fs: float) -> None:
        super().__init__()
        self.layer_order = ("L1", "L2")
        self.layer_widths = {"L1": WIDTH, "L2": WIDTH}
        self.layer_shifts = {"L1": L1_SHIFTS, "L2": L2_SHIFTS}
        self.last_layer = "L2"
        self.last_width = WIDTH
        self.fs = float(fs)

        beta = exp50.beta_value(self.fs)
        self.f1 = nn.Linear(base.EVENT_CHANNELS, WIDTH, bias=False)
        self.f2 = nn.Linear(WIDTH, WIDTH, bias=False)
        # Third Linear position mirrors Exp5.0's output_linear draw order. The
        # analog head adds a bias only after its weight has been initialized.
        self.head = nn.Linear(WIDTH, n_classes, bias=True)
        self.l1_lif = exp50.exp401.MacroMultiSpikeLIF(
            beta=beta,
            threshold=base.THRESHOLD,
            max_spikes_per_dt=1,
            surrogate_slope=base.SURROGATE_SLOPE,
        )
        self.l2_lif = exp50.exp401.MacroMultiSpikeLIF(
            beta=beta,
            threshold=base.THRESHOLD,
            max_spikes_per_dt=1,
            surrogate_slope=base.SURROGATE_SLOPE,
        )
        self.register_buffer("alpha1", exp50.alpha_vector(WIDTH, L1_SHIFTS))
        self.register_buffer("alpha2", exp50.alpha_vector(WIDTH, L2_SHIFTS))

    def layer_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, n_steps, channels = x.shape
        if channels != base.EVENT_CHANNELS:
            raise ValueError(f"Expected {base.EVENT_CHANNELS} channels, got {channels}")
        syn1 = torch.zeros(batch, WIDTH, device=x.device, dtype=x.dtype)
        mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, WIDTH, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(syn2)
        seq1: list[torch.Tensor] = []
        seq2: list[torch.Tensor] = []
        for timestep in range(n_steps):
            syn1 = self.alpha1 * syn1 + self.f1(x[:, timestep])
            s1, mem1, _ = self.l1_lif(syn1, mem1)
            syn2 = self.alpha2 * syn2 + self.f2(s1)
            s2, mem2, _ = self.l2_lif(syn2, mem2)
            seq1.append(s1)
            seq2.append(s2)
        return {
            "L1": torch.stack(seq1, dim=1),
            "L2": torch.stack(seq2, dim=1),
        }

    def loss_logits(
        self,
        spikes: torch.Tensor,
        lengths: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_steps, _ = spikes.shape
        logits_t = self.head(spikes)
        valid = base.mask(lengths, n_steps)
        targets = y[:, None].expand(batch, n_steps)
        loss = F.cross_entropy(logits_t[valid], targets[valid])
        weights = valid.to(logits_t.dtype).unsqueeze(-1)
        segment_logits = (logits_t * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return loss, segment_logits


def build_model(data: base.Data, condition: str) -> nn.Module:
    if condition in (EXP3_EXACT, EXP3_EXP5_STREAM):
        model = exp304.L2WidthNet(
            WIDTH,
            OBJECTIVE,
            len(data.labels),
            data.T,
            data.fs,
            data.bin_steps,
        )
        if model.last_layer != "L2":
            raise RuntimeError(f"Expected L2 final layer, got {model.last_layer}")
        return model
    if condition == MACRO_EXP5_STREAM:
        return Exp5MacroBinaryAnalogNet(len(data.labels), data.fs)
    raise ValueError(f"Unknown condition: {condition}")


def _initialize_model(spec: RunSpec, data: base.Data, device: torch.device) -> nn.Module:
    if spec.condition == EXP3_EXACT:
        # Exact historical Exp3.0.3-B / Exp3.0.4 contract.
        base.seed_all(base.dseed(spec.seed, "shared_backbone_init"))
        model = build_model(data, spec.condition).to(device)
        base.seed_all(base.dseed(spec.seed, OBJECTIVE, "head_init"))
        model.head.reset_parameters()  # type: ignore[attr-defined]
        return model

    # Both Exp5-stream controls use exactly the same random stream. Because both
    # constructors have f1, f2, and the 128->12 analog head as their only random
    # parameterized modules in that order, they begin with identical f1/f2/head
    # parameters; only hidden neuron dynamics differ.
    base.seed_all(base.dseed(spec.seed, "exp5_0_paired", "model_init"))
    return build_model(data, spec.condition).to(device)


def _loader_seed(spec: RunSpec, split: str) -> int:
    if spec.condition == EXP3_EXACT:
        return base.dseed(spec.seed, split, "loader")
    return base.dseed(spec.seed, "exp5_0_paired", f"{split}_loader")


def make_loaders(
    data: base.Data,
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
        split: base.loader(
            *partition,
            batch_size,
            train_shuffle if split == "train" else False,
            _loader_seed(spec, split),
        )
        for split, partition in partitions.items()
    }


def evaluate_native(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for X, y, lengths in data_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            spikes = model.layer_features(X)["L2"]  # type: ignore[attr-defined]
            loss, logits = model.loss_logits(spikes, lengths, y)  # type: ignore[attr-defined]
            n = len(y)
            n_total += n
            loss_sum += float(loss.item()) * n
            y_true_parts.append(y.cpu().numpy())
            y_pred_parts.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    out = base.metrics(y_true, y_pred)
    out["loss"] = loss_sum / max(n_total, 1)
    return out


def train_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool,
) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model = _initialize_model(spec, data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_loader = make_loaders(data, spec, config.batch_size, train_shuffle=True)["train"]
    eval_loaders = make_loaders(data, spec, config.batch_size, train_shuffle=False)

    best_val_ba = -np.inf
    best_val_loss = np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []

        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            spikes = model.layer_features(X)["L2"]  # type: ignore[attr-defined]
            loss, logits = model.loss_logits(spikes, lengths, y)  # type: ignore[attr-defined]
            loss.backward()
            optimizer.step()

            n = len(y)
            train_n += n
            train_loss_sum += float(loss.item()) * n
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = base.metrics(np.concatenate(train_true), np.concatenate(train_pred))
        val_metrics = evaluate_native(model, eval_loaders["val"], device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
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
            abs(val_ba - best_val_ba) <= 1e-12 and val_loss < best_val_loss - 1e-12
        )
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    model.load_state_dict(best_state)

    final = {
        split: evaluate_native(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    result = {
        "condition": spec.condition,
        "seed": spec.seed,
        "objective": OBJECTIVE,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_ba,
        "best_val_loss": best_val_loss,
        "native": final,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "provenance": provenance(spec, data, config),
            "result": result,
            "state_dict": best_state,
        },
        destination,
    )
    history_file = history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[nn.Module, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.0.1 checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong experiment id in {path}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong protocol version in {path}")
    if payload.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint identity mismatch for {spec.key}")
    model = build_model(data, spec.condition).to(torch.device(config.device))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def evaluate_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    model, checkpoint = load_model(spec, data, config)
    probe_config = exp305.Config(
        repo_root=config.repo_root,
        results_dir=config.results_dir,
        device=config.device,
        batch_size=config.batch_size,
        resume=False,
        threads=config.threads,
    )
    partitions = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    extracted = {
        split: exp305._partition_features(
            model,  # type: ignore[arg-type]
            *partition,
            split,
            OBJECTIVE,
            spec.seed,
            probe_config,
            data.fs,
            data.bin_steps,
        )
        for split, partition in partitions.items()
    }

    probes: list[dict[str, object]] = []
    for probe_type in PROBE_TYPES:
        if probe_type == "fixed250_pca128":
            source_key = "fixed250_ordered"
            metrics = exp304._pca_probe_metrics(
                extracted["train"][source_key],
                extracted["train"]["y"],
                extracted["val"][source_key],
                extracted["val"]["y"],
                extracted["test"][source_key],
                extracted["test"]["y"],
                base.dseed(spec.seed, OBJECTIVE, probe_type),
            )
        elif probe_type == "relative10_pca128":
            source_key = "relative10_ordered"
            metrics = exp304._pca_probe_metrics(
                extracted["train"][source_key],
                extracted["train"]["y"],
                extracted["val"][source_key],
                extracted["val"]["y"],
                extracted["test"][source_key],
                extracted["test"]["y"],
                base.dseed(spec.seed, OBJECTIVE, probe_type),
            )
        else:
            metrics = exp304._raw_probe_metrics(
                extracted["train"][probe_type],
                extracted["train"]["y"],
                extracted["val"][probe_type],
                extracted["val"]["y"],
                extracted["test"][probe_type],
                extracted["test"]["y"],
                base.dseed(spec.seed, OBJECTIVE, probe_type),
            )
        probes.append({"probe_type": probe_type, **metrics})

    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "condition": spec.condition,
        "seed": spec.seed,
        "objective": OBJECTIVE,
        "best_epoch": int(checkpoint["result"]["best_epoch"]),
        "native": checkpoint["result"]["native"],
        "probes": probes,
        "provenance": provenance(spec, data, config),
    }
    _save_json(destination, payload)
    return payload


def run_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool,
) -> dict[str, object]:
    train_one(spec, data, config, force)
    return evaluate_one(spec, data, config, force)


def provenance(spec: RunSpec, data: base.Data, config: Config) -> dict[str, object]:
    if spec.condition == EXP3_EXACT:
        hidden_contract = "exact Exp3.0.3-B snnTorch.Synaptic L1/L2"
        init_contract = "Exp3 shared_backbone_init + timestep_ce head_init"
        loader_contract = "Exp3 train/val/test loader streams"
        role = "historical Exp3 reproduction gate"
    elif spec.condition == EXP3_EXP5_STREAM:
        hidden_contract = "Exp3 snnTorch.Synaptic L1/L2 under Exp5 paired random/data stream"
        init_contract = "Exp5.0 exp5_0_paired model_init stream"
        loader_contract = "Exp5.0 exp5_0_paired train/val/test loader streams"
        role = "paired Synaptic control for hidden-dynamics comparison"
    else:
        hidden_contract = "exact Exp5.0 binary MacroMultiSpikeLIF hidden dynamics through L2"
        init_contract = "Exp5.0 exp5_0_paired model_init stream"
        loader_contract = "Exp5.0 exp5_0_paired train/val/test loader streams"
        role = "paired Macro control for hidden-dynamics and output-path comparison"
    return {
        "question": (
            "Separate protocol reproduction, hidden-dynamics implementation, and the Exp5 "
            "spiking-output path by using three analog-head controls."
        ),
        "condition": spec.condition,
        "role": role,
        "hidden_contract": hidden_contract,
        "width": WIDTH,
        "l1_shifts": list(L1_SHIFTS),
        "l2_shifts": list(L2_SHIFTS),
        "tau_mem_ms": base.TAU_MEM_MS,
        "threshold": base.THRESHOLD,
        "reset": base.RESET,
        "surrogate_slope": base.SURROGATE_SLOPE,
        "head": "Linear(128, 12, bias=True), no output LIF",
        "objective": OBJECTIVE,
        "checkpoint_selection": "native analog-head validation BA, tie-break validation CE",
        "probe_protocol": "exact Exp3.0.5 StandardScaler + LogisticRegression C-grid/PCA controls",
        "initialization": init_contract,
        "loader_contract": loader_contract,
        "seed": spec.seed,
        "split_seed": base.SPLIT_SEED,
        "train_users": list(data.split["train_users"]),
        "val_users": list(data.split["val_users"]),
        "test_users": list(data.split["test_users"]),
        "labels": list(data.labels),
        "sampling_rate_hz": float(data.fs),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": LR,
    }


def _exp3_reference_paths(repo_root: Path) -> tuple[Path, Path]:
    root = (
        repo_root
        / "notebooks"
        / "artifacts"
        / "experiment_3_0_5_frozen_representation_accessibility"
        / "frozen_access_v1"
    )
    return (
        root / "experiment_3_0_5_probe_results.csv",
        root / "experiment_3_0_5_native_results.csv",
    )


def _exp5_reference_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / "experiment_5_0_local_evidence_objectives"
        / "local_objective_x_event_capacity_v1"
        / "runs.csv"
    )


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing required Exp5.0.1 artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = (PROTOCOL_VERSION, spec.condition, spec.seed)
        actual = (
            str(payload.get("protocol_version")),
            str(payload.get("condition")),
            int(payload.get("seed")),
        )
        if actual != expected:
            raise ValueError(f"Evaluation identity mismatch in {path}: {actual} != {expected}")
        probe_map = {probe["probe_type"]: probe for probe in payload["probes"]}
        row: dict[str, object] = {
            "source": "exp5_0_1_control",
            "condition": spec.condition,
            "seed": spec.seed,
            "native_test_ba": float(payload["native"]["test"]["balanced_accuracy"]),
            "native_val_ba": float(payload["native"]["val"]["balanced_accuracy"]),
            "best_epoch": int(payload["best_epoch"]),
        }
        for probe_type in PROBE_TYPES:
            probe = probe_map[probe_type]
            row[f"{probe_type}_test_ba"] = float(probe["probe_test_balanced_accuracy"])
            row[f"{probe_type}_val_ba"] = float(probe["probe_val_balanced_accuracy"])
        rows.append(row)

    runs = pd.DataFrame(rows).sort_values(["condition", "seed"], ignore_index=True)
    root.mkdir(parents=True, exist_ok=True)
    runs_file = root / "runs.csv"
    runs.to_csv(runs_file, index=False)

    exp3_probe_path, exp3_native_path = _exp3_reference_paths(repo_root)
    exp5_runs_path = _exp5_reference_path(repo_root)
    for path in (exp3_probe_path, exp3_native_path, exp5_runs_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing comparison source artifact: {path}")

    exp3_probe = pd.read_csv(exp3_probe_path)
    exp3_native = pd.read_csv(exp3_native_path)
    exp5_runs = pd.read_csv(exp5_runs_path)

    comparison_rows: list[dict[str, object]] = []
    for _, row in runs.iterrows():
        comparison_rows.append(
            {
                "source": "exp5_0_1_control",
                "variant": str(row["condition"]),
                "seed": int(row["seed"]),
                "native_or_output_ba": float(row["native_test_ba"]),
                "full_count_ba": float(row["full_count_test_ba"]),
                "fixed250_ordered_ba": float(row["fixed250_ordered_test_ba"]),
                "relative10_ordered_ba": float(row["relative10_ordered_test_ba"]),
            }
        )

    exp3_native_ts = exp3_native[exp3_native["objective"] == OBJECTIVE].set_index("seed")
    exp3_probe_ts = exp3_probe[exp3_probe["objective"] == OBJECTIVE]
    for seed in SEEDS:
        by_probe = exp3_probe_ts[exp3_probe_ts["seed"] == seed].set_index("probe_type")
        comparison_rows.append(
            {
                "source": "exp3_0_5_reference",
                "variant": "historical_exp3_analog",
                "seed": seed,
                "native_or_output_ba": float(exp3_native_ts.loc[seed, "test_balanced_accuracy"]),
                "full_count_ba": float(by_probe.loc["full_count", "probe_test_balanced_accuracy"]),
                "fixed250_ordered_ba": float(by_probe.loc["fixed250_ordered", "probe_test_balanced_accuracy"]),
                "relative10_ordered_ba": float(by_probe.loc["relative10_ordered", "probe_test_balanced_accuracy"]),
            }
        )

    exp5_ts = exp5_runs[(exp5_runs["split"] == "test") & (exp5_runs["objective"] == OBJECTIVE)]
    for variant in ("binary", "multi_ho"):
        subset = exp5_ts[exp5_ts["variant"] == variant]
        for _, row in subset.iterrows():
            comparison_rows.append(
                {
                    "source": "exp5_0_spiking_output_reference",
                    "variant": variant,
                    "seed": int(row["seed"]),
                    "native_or_output_ba": float(row["output_whole_count_ba"]),
                    "full_count_ba": float(row["l2_whole_count_linear_ba"]),
                    "fixed250_ordered_ba": float(row["l2_fixed250_ordered_linear_ba"]),
                    "relative10_ordered_ba": float(row["l2_relative10_ordered_linear_ba"]),
                }
            )

    comparison = pd.DataFrame(comparison_rows)
    comparison_file = root / "comparison_runs.csv"
    comparison.to_csv(comparison_file, index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "expected_runs": EXPECTED_RUNS,
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "objective": OBJECTIVE,
        "primary_probes": list(PRIMARY_PROBES),
        "comparison_sources": [
            "Exp3.0.5 historical analog-head reference",
            "Exp5.0 timestep_ce binary spiking-output reference",
            "Exp5.0 timestep_ce multi_ho spiking-output reference",
        ],
        "files": {
            "runs": runs_file.name,
            "comparison_runs": comparison_file.name,
        },
        "notebook_role": "compute mean/SD, paired deltas, and plots from finalized run tables",
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    return {
        "runs": runs_file,
        "comparison_runs": comparison_file,
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
    eval_parser.add_argument("--condition", choices=CONDITIONS, required=True)
    eval_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--threads", type=int, default=1)
    eval_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    eval_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = find_repo_root()
    if args.command == "finalize":
        for name, path in finalize_experiment(repo_root).items():
            print(f"{name}: {path}")
        return

    data = prepare_data(repo_root)
    config = Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        epochs=getattr(args, "epochs", EPOCHS),
        batch_size=args.batch_size,
        threads=args.threads,
    )
    if args.command == "run-one":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(f"array task {args.array_task_id} outside [0, {len(specs) - 1}]")
        spec = specs[args.array_task_id]
        payload = run_one(spec, data, config, args.force)
    else:
        spec = RunSpec(args.condition, args.seed)
        payload = evaluate_one(spec, data, config, args.force)

    probe_map = {probe["probe_type"]: probe for probe in payload["probes"]}
    print(
        f"completed {spec.key}: "
        f"native={payload['native']['test']['balanced_accuracy']:.6f} "
        f"full_count={probe_map['full_count']['probe_test_balanced_accuracy']:.6f} "
        f"fixed250={probe_map['fixed250_ordered']['probe_test_balanced_accuracy']:.6f} "
        f"relative10={probe_map['relative10_ordered']['probe_test_balanced_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
