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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts import experiment_4_0_5_temporal_resolution_event_capacity as exp405


EXPERIMENT_ID = "experiment_4_0_6_local_mem_raw64"
PROTOCOL_VERSION = "raw64_local_mem_shift_ablation_v1"
BASELINE_CONDITION = "beta0"
TRAIN_CONDITIONS = ("shift4", "shift34", "shift234")
ALL_CONDITIONS = (BASELINE_CONDITION, *TRAIN_CONDITIONS)
LOCAL_SHIFT_SETS: dict[str, tuple[int, ...]] = {
    "shift4": (4,),
    "shift34": (3, 4),
    "shift234": (2, 3, 4),
}
VARIANTS = exp405.VARIANTS
SEEDS = exp405.SEEDS
EXPECTED_NEW_RUNS = len(TRAIN_CONDITIONS) * len(VARIANTS) * len(SEEDS)
EXPECTED_BASELINE_PROBES = len(VARIANTS) * len(SEEDS)

LOCAL_WIDTH = exp405.LOCAL_WIDTH
STATE_WIDTH = exp405.STATE_WIDTH
STATE_TAU_MEM_MS = exp405.STATE_TAU_MEM_MS
OUTPUT_TAU_MEM_MS = exp405.OUTPUT_TAU_MEM_MS
THRESHOLD = exp405.THRESHOLD
BATCH_SIZE = exp405.BATCH_SIZE
EPOCHS = exp405.EPOCHS
LR = exp405.LR
WEIGHT_DECAY = exp405.WEIGHT_DECAY
PROBE_MAX_ITER = exp405.PROBE_MAX_ITER
RAW_DT_MS = 1000.0 / 64.0


@dataclass(frozen=True)
class RunSpec:
    condition: str
    variant: str
    hidden_cap: int
    output_cap: int
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.condition}__{self.variant}__"
            f"hcap{self.hidden_cap}__ocap{self.output_cap}__seed{self.seed}"
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
        RunSpec(condition, variant, hidden_cap, output_cap, seed)
        for condition in TRAIN_CONDITIONS
        for variant, hidden_cap, output_cap in VARIANTS
        for seed in SEEDS
    ]


def baseline_specs() -> list[RunSpec]:
    return [
        RunSpec(BASELINE_CONDITION, variant, hidden_cap, output_cap, seed)
        for variant, hidden_cap, output_cap in VARIANTS
        for seed in SEEDS
    ]


def paired_seed(spec: RunSpec, role: str) -> int:
    """Reuse the exact Exp4.0.5 paired random stream for fair beta0 comparisons."""
    return exp405.exp40.base.dseed(spec.seed, "exp4_0_5_paired", role)


def _split_width(width: int, n_groups: int) -> tuple[int, ...]:
    base, remainder = divmod(width, n_groups)
    return tuple(base + (1 if index < remainder else 0) for index in range(n_groups))


def local_condition_definition(condition: str, fs: float = 64.0) -> dict[str, object]:
    if condition == BASELINE_CONDITION:
        return {
            "condition": condition,
            "shifts": (),
            "group_widths": (LOCAL_WIDTH,),
            "betas": (0.0,),
            "tau_mem_ms": (0.0,),
            "role": "Exp4.0.5 Raw64 control: no local temporal memory",
        }
    shifts = LOCAL_SHIFT_SETS.get(condition)
    if shifts is None:
        raise ValueError(f"Unknown local-memory condition: {condition}")
    widths = _split_width(LOCAL_WIDTH, len(shifts))
    betas = tuple(exp405.exp40.base.alpha(shift) for shift in shifts)
    taus = tuple(exp405.exp40.base.tau_ms(shift, fs) for shift in shifts)
    return {
        "condition": condition,
        "shifts": shifts,
        "group_widths": widths,
        "betas": betas,
        "tau_mem_ms": taus,
        "role": "stateful feedforward local LIF memory; no learned local recurrence",
    }


def local_beta_vector(condition: str, fs: float = 64.0) -> torch.Tensor:
    definition = local_condition_definition(condition, fs)
    pieces: list[torch.Tensor] = []
    for beta, width in zip(
        definition["betas"], definition["group_widths"], strict=True
    ):
        pieces.append(torch.full((int(width),), float(beta), dtype=torch.float32))
    vector = torch.cat(pieces)
    if vector.shape != (LOCAL_WIDTH,):
        raise ValueError(f"Expected {LOCAL_WIDTH} local betas, got {vector.shape}")
    return vector


class HeterogeneousMacroMultiSpikeLIF(nn.Module):
    """Exp4.0.1 multi-event LIF with a fixed beta for each local neuron."""

    def __init__(
        self,
        beta_by_neuron: torch.Tensor,
        threshold: float,
        max_spikes_per_dt: int,
    ) -> None:
        super().__init__()
        beta = torch.as_tensor(beta_by_neuron, dtype=torch.float32).reshape(-1)
        if beta.shape != (LOCAL_WIDTH,):
            raise ValueError(f"Expected {LOCAL_WIDTH} beta values, got {beta.shape}")
        if bool(((beta < 0.0) | (beta > 1.0)).any()):
            raise ValueError("beta values must be in [0, 1]")
        self.register_buffer("beta", beta)
        self.threshold = float(threshold)
        self.max_spikes_per_dt = int(max_spikes_per_dt)
        self.surrogate_slope = float(exp405.exp401.SURROGATE_SLOPE)

    def forward(
        self,
        current: torch.Tensor,
        membrane: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_reset = self.beta.to(dtype=membrane.dtype) * membrane + current
        spikes = exp405.exp401.multi_threshold_spike(
            pre_reset,
            self.threshold,
            self.max_spikes_per_dt,
            self.surrogate_slope,
        )
        membrane = pre_reset - spikes * self.threshold
        return spikes, membrane, pre_reset


class LocalMemoryRaw64Decoder(exp405.HierarchicalTemporalDecoder):
    """Raw64 Local128 -> RSNN128 -> output with controlled local membrane memory."""

    def __init__(
        self,
        condition: str,
        n_classes: int,
        hidden_cap: int,
        output_cap: int,
        fs: float = 64.0,
    ) -> None:
        if condition not in TRAIN_CONDITIONS:
            raise ValueError(f"Trainable Exp4.0.6 condition required, got {condition}")
        dt_ms = 1000.0 / float(fs)
        super().__init__(
            n_classes=n_classes,
            dt_ms=dt_ms,
            hidden_cap=hidden_cap,
            output_cap=output_cap,
        )
        self.condition = condition
        self.fs = float(fs)
        self.local_lif = HeterogeneousMacroMultiSpikeLIF(
            beta_by_neuron=local_beta_vector(condition, fs),
            threshold=THRESHOLD,
            max_spikes_per_dt=hidden_cap,
        )


def prepare_raw64_data(repo_root: Path) -> exp405.TemporalData:
    data = exp405.prepare_all_representations(repo_root)["raw64"]
    if not np.isclose(data.dt_ms, RAW_DT_MS):
        raise ValueError(f"Expected Raw64 dt={RAW_DT_MS}, got {data.dt_ms}")
    return data


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


def extract_probe_features(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count_parts: list[np.ndarray] = []
    endpoint_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, valid_steps in data_loader:
            X = X.to(device)
            valid_steps = valid_steps.to(device)
            traj = model.forward_trajectory(X)  # type: ignore[attr-defined]
            hidden_count = exp405.exp40.valid_whole_count(
                traj["state_spikes"], valid_steps
            )
            endpoint = exp405.exp40.valid_final_membrane(
                traj["state_membranes"], valid_steps
            )
            count_parts.append(hidden_count.cpu().numpy())
            endpoint_parts.append(endpoint.cpu().numpy())
            label_parts.append(y.numpy())
    return (
        np.concatenate(count_parts),
        np.concatenate(endpoint_parts),
        np.concatenate(label_parts),
    )


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


def fit_scaled_linear_probes(
    model: nn.Module,
    loaders: dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> dict[str, object]:
    extracted = {
        split: extract_probe_features(model, data_loader, device)
        for split, data_loader in loaders.items()
    }
    train_count, train_endpoint, ytr = extracted["train"]
    probes = {
        "hidden_whole_count": _scaled_linear_probe(),
        "uend": _scaled_linear_probe(),
    }
    probes["hidden_whole_count"].fit(train_count, ytr)
    probes["uend"].fit(train_endpoint, ytr)

    out: dict[str, object] = {
        "temporal_phase_access": False,
        "source_snn_frozen": True,
        "classifier": "train-only StandardScaler + balanced LogisticRegression(lbfgs)",
        "hidden_whole_count_feature_dim": STATE_WIDTH,
        "uend_feature_dim": STATE_WIDTH,
        "metrics": {},
    }
    for split, (hidden_count, endpoint, y) in extracted.items():
        out["metrics"][split] = {
            "hidden_whole_count_linear": exp405.exp40.metrics(
                y, probes["hidden_whole_count"].predict(hidden_count)
            ),
            "uend_linear": exp405.exp40.metrics(
                y, probes["uend"].predict(endpoint)
            ),
        }
    return out


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def baseline_path(root: Path, spec: RunSpec) -> Path:
    return root / "baseline_probes" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def train_one(
    spec: RunSpec,
    data: exp405.TemporalData,
    root: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    threads: int,
    force: bool,
) -> Path:
    destination = checkpoint_path(root, spec)
    if destination.exists() and not force:
        return destination
    if spec.condition not in TRAIN_CONDITIONS:
        raise ValueError("beta0 is a frozen Exp4.0.5 baseline and must not be retrained")

    torch.set_num_threads(threads)
    exp405.exp40.base.seed_all(paired_seed(spec, "model_init"))
    model = LocalMemoryRaw64Decoder(
        condition=spec.condition,
        n_classes=len(data.labels),
        hidden_cap=spec.hidden_cap,
        output_cap=spec.output_cap,
        fs=data.fs,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    train_loader = make_loaders(data, spec, batch_size, train_shuffle=True)["train"]
    val_loader = make_loaders(data, spec, batch_size, train_shuffle=False)["val"]

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
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
            train_loss_sum += float(loss.item()) * n
            train_n += n
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
            "local_condition": local_condition_definition(spec.condition, data.fs),
            "dt_ms": data.dt_ms,
            "state_tau_mem_ms": STATE_TAU_MEM_MS,
            "output_tau_mem_ms": OUTPUT_TAU_MEM_MS,
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
    history_file = history_path(root, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False)
    return destination


def load_best_model(
    spec: RunSpec,
    data: exp405.TemporalData,
    root: Path,
    device: torch.device,
) -> tuple[LocalMemoryRaw64Decoder, dict[str, object]]:
    path = checkpoint_path(root, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp4.0.6 checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint identity mismatch for {spec.key}")
    model = LocalMemoryRaw64Decoder(
        condition=spec.condition,
        n_classes=len(data.labels),
        hidden_cap=spec.hidden_cap,
        output_cap=spec.output_cap,
        fs=data.fs,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_new_run(
    spec: RunSpec,
    data: exp405.TemporalData,
    root: Path,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, object]:
    destination = evaluation_path(root, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    model, checkpoint = load_best_model(spec, data, root, device)
    loaders = make_loaders(data, spec, batch_size, train_shuffle=False)
    metrics = {
        split: exp405.evaluate_model(model, data_loader, device)
        for split, data_loader in loaders.items()
    }
    probes = fit_scaled_linear_probes(model, loaders, device)
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "source": "new Exp4.0.6 training run",
        "architecture": {
            "topology": "Raw64 30 -> Local128(stateful, no recurrence) -> RSNN128 -> 12",
            "local": local_condition_definition(spec.condition, data.fs),
            "state_tau_mem_ms": STATE_TAU_MEM_MS,
            "output_tau_mem_ms": OUTPUT_TAU_MEM_MS,
            "dt_ms": data.dt_ms,
            "state_beta": exp405.state_beta(data.dt_ms),
            "output_beta": exp405.output_beta(data.dt_ms),
            "threshold": THRESHOLD,
            "parameter_counts": exp405.parameter_counts(len(data.labels)),
        },
        "checkpoint": {
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_val_balanced_accuracy": float(
                checkpoint["best_val_balanced_accuracy"]
            ),
            "best_val_loss": float(checkpoint["best_val_loss"]),
        },
        "metrics": metrics,
        "linear_probes": probes,
        "provenance": {
            "input_representation": "raw64",
            "input_information_contract": data.information_contract,
            "channel_scale_source": (
                "training Fixed250 valid bins only, inherited exactly from Exp4.0.5"
            ),
            "paired_random_stream": "exact Exp4.0.5 exp4_0_5_paired seed stream",
            "split_seed": int(exp405.exp40.base.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "readout_training_objective": "valid normalized output WholeCount CE",
            "probe_training": (
                "frozen best SNN; train-only StandardScaler and balanced LogisticRegression"
            ),
        },
    }
    _save_json(destination, payload)
    return payload


def source_exp405_spec(spec: RunSpec) -> exp405.RunSpec:
    if spec.condition != BASELINE_CONDITION:
        raise ValueError("Only beta0 maps to an Exp4.0.5 source checkpoint")
    return exp405.RunSpec(
        representation="raw64",
        variant=spec.variant,
        hidden_cap=spec.hidden_cap,
        output_cap=spec.output_cap,
        seed=spec.seed,
    )


def evaluate_frozen_baseline(
    spec: RunSpec,
    data: exp405.TemporalData,
    repo_root: Path,
    root: Path,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, object]:
    destination = baseline_path(root, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    source_root = exp405.results_dir(repo_root)
    source_spec = source_exp405_spec(spec)
    source_checkpoint = exp405.checkpoint_path(source_root, source_spec)
    if not source_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing Exp4.0.5 Raw64 baseline checkpoint: {source_checkpoint}"
        )
    model, checkpoint = exp405.load_best_model(
        source_spec, data, source_root, device
    )
    loaders = make_loaders(data, spec, batch_size, train_shuffle=False)
    metrics = {
        split: exp405.evaluate_model(model, data_loader, device)
        for split, data_loader in loaders.items()
    }
    probes = fit_scaled_linear_probes(model, loaders, device)
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "source": {
            "experiment_id": exp405.EXPERIMENT_ID,
            "protocol_version": exp405.PROTOCOL_VERSION,
            "checkpoint": str(source_checkpoint.relative_to(repo_root)),
            "best_epoch": int(checkpoint["best_epoch"]),
        },
        "architecture": {
            "topology": "Raw64 30 -> Local128(beta=0,no recurrence) -> RSNN128 -> 12",
            "local": local_condition_definition(BASELINE_CONDITION, data.fs),
            "state_tau_mem_ms": STATE_TAU_MEM_MS,
            "output_tau_mem_ms": OUTPUT_TAU_MEM_MS,
            "dt_ms": data.dt_ms,
            "state_beta": exp405.state_beta(data.dt_ms),
            "output_beta": exp405.output_beta(data.dt_ms),
            "threshold": THRESHOLD,
            "parameter_counts": exp405.parameter_counts(len(data.labels)),
        },
        "checkpoint": {
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_val_balanced_accuracy": float(
                checkpoint["best_val_balanced_accuracy"]
            ),
            "best_val_loss": float(checkpoint["best_val_loss"]),
        },
        "metrics": metrics,
        "linear_probes": probes,
        "provenance": {
            "input_representation": "raw64",
            "reused_frozen_baseline": True,
            "snn_retrained": False,
            "probe_refit": True,
            "probe_training": (
                "frozen Exp4.0.5 checkpoint; train-only StandardScaler and balanced LogisticRegression"
            ),
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
    device = torch.device(config.device)
    train_one(
        spec,
        data,
        config.results_dir,
        device,
        config.epochs,
        config.batch_size,
        config.threads,
        force,
    )
    return evaluate_new_run(
        spec,
        data,
        config.results_dir,
        device,
        config.batch_size,
        force,
    )


def _payload_path(root: Path, spec: RunSpec) -> Path:
    return baseline_path(root, spec) if spec.condition == BASELINE_CONDITION else evaluation_path(root, spec)


def _all_specs() -> list[RunSpec]:
    return [*baseline_specs(), *run_specs()]


def _run_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in _all_specs():
        path = _payload_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        probes = payload["linear_probes"]["metrics"]
        local = payload["architecture"]["local"]
        for split in ("train", "val", "test"):
            metrics = payload["metrics"][split]
            hidden_probe = probes[split]["hidden_whole_count_linear"]
            uend_probe = probes[split]["uend_linear"]
            row: dict[str, object] = {
                "condition": spec.condition,
                "local_shifts": json.dumps(local["shifts"]),
                "local_group_widths": json.dumps(local["group_widths"]),
                "local_tau_mem_ms": json.dumps(local["tau_mem_ms"]),
                "variant": spec.variant,
                "hidden_cap": spec.hidden_cap,
                "output_cap": spec.output_cap,
                "seed": spec.seed,
                "split": split,
                "dt_ms": float(payload["architecture"]["dt_ms"]),
                "valid_count_balanced_accuracy": float(
                    metrics["valid_count"]["balanced_accuracy"]
                ),
                "valid_count_macro_f1": float(metrics["valid_count"]["macro_f1"]),
                "full_count_balanced_accuracy": float(
                    metrics["full_count"]["balanced_accuracy"]
                ),
                "hidden_count_linear_balanced_accuracy": float(
                    hidden_probe["balanced_accuracy"]
                ),
                "hidden_count_linear_macro_f1": float(hidden_probe["macro_f1"]),
                "uend_linear_balanced_accuracy": float(
                    uend_probe["balanced_accuracy"]
                ),
                "uend_linear_macro_f1": float(uend_probe["macro_f1"]),
                "state_tail_event_fraction": float(
                    metrics["state_tail_event_fraction"]
                ),
                "output_tail_event_fraction": float(
                    metrics["output_tail_event_fraction"]
                ),
                "state_mean_events_per_neuron_second": float(
                    metrics["state_valid_stats"]["mean_events_per_neuron_second"]
                ),
                "output_mean_events_per_neuron_second": float(
                    metrics["output_valid_stats"]["mean_events_per_neuron_second"]
                ),
                "local_mean_events_per_neuron_second": float(
                    metrics["local_valid_stats"]["mean_events_per_neuron_second"]
                ),
                "local_fraction_at_cap": float(
                    metrics["local_valid_stats"]["fraction_at_cap"]
                ),
                "state_fraction_at_cap": float(
                    metrics["state_valid_stats"]["fraction_at_cap"]
                ),
                "output_fraction_at_cap": float(
                    metrics["output_valid_stats"]["fraction_at_cap"]
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _readout_columns() -> dict[str, str]:
    return {
        "output_whole_count": "valid_count_balanced_accuracy",
        "hidden_whole_count_linear": "hidden_count_linear_balanced_accuracy",
        "uend_linear": "uend_linear_balanced_accuracy",
    }


def _paired_effects(test: pd.DataFrame) -> pd.DataFrame:
    indexed = test.set_index(["condition", "variant", "seed"])
    rows: list[dict[str, object]] = []
    condition_pairs = (
        ("shift4_minus_beta0", "shift4", "beta0"),
        ("shift34_minus_beta0", "shift34", "beta0"),
        ("shift234_minus_beta0", "shift234", "beta0"),
        ("shift34_minus_shift4", "shift34", "shift4"),
        ("shift234_minus_shift34", "shift234", "shift34"),
    )
    for effect, high, low in condition_pairs:
        for variant, _, _ in VARIANTS:
            for seed in SEEDS:
                for readout, column in _readout_columns().items():
                    delta = float(indexed.loc[(high, variant, seed), column]) - float(
                        indexed.loc[(low, variant, seed), column]
                    )
                    rows.append(
                        {
                            "effect": effect,
                            "condition": "",
                            "variant": variant,
                            "seed": seed,
                            "readout": readout,
                            "delta_balanced_accuracy": delta,
                        }
                    )
    for condition in ALL_CONDITIONS:
        for seed in SEEDS:
            for readout, column in _readout_columns().items():
                binary = float(indexed.loc[(condition, "binary", seed), column])
                multi_h = float(indexed.loc[(condition, "multi_h", seed), column])
                multi_ho = float(indexed.loc[(condition, "multi_ho", seed), column])
                rows.extend(
                    [
                        {
                            "effect": "multi_h_minus_binary",
                            "condition": condition,
                            "variant": "",
                            "seed": seed,
                            "readout": readout,
                            "delta_balanced_accuracy": multi_h - binary,
                        },
                        {
                            "effect": "multi_ho_minus_binary",
                            "condition": condition,
                            "variant": "",
                            "seed": seed,
                            "readout": readout,
                            "delta_balanced_accuracy": multi_ho - binary,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    runs = _run_rows(root)
    expected_rows = (EXPECTED_NEW_RUNS + EXPECTED_BASELINE_PROBES) * 3
    if len(runs) != expected_rows:
        raise ValueError(f"Expected {expected_rows} finalized rows, got {len(runs)}")
    runs_file = root / "runs.csv"
    runs.to_csv(runs_file, index=False)

    test = runs[runs["split"] == "test"].copy()
    summary_columns = [
        "valid_count_balanced_accuracy",
        "hidden_count_linear_balanced_accuracy",
        "uend_linear_balanced_accuracy",
        "local_mean_events_per_neuron_second",
        "state_mean_events_per_neuron_second",
        "output_mean_events_per_neuron_second",
        "state_tail_event_fraction",
        "output_tail_event_fraction",
        "local_fraction_at_cap",
        "state_fraction_at_cap",
        "output_fraction_at_cap",
    ]
    summary = (
        test.groupby(["condition", "variant", "hidden_cap", "output_cap"])[
            summary_columns
        ]
        .agg(["mean", "std"])
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

    effects = _paired_effects(test)
    effects_file = root / "paired_effects.csv"
    effects.to_csv(effects_file, index=False)
    effect_summary = (
        effects.groupby(["effect", "condition", "variant", "readout"], dropna=False)[
            "delta_balanced_accuracy"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    effect_summary_file = root / "paired_effects_summary.csv"
    effect_summary.to_csv(effect_summary_file, index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "expected_new_training_runs": EXPECTED_NEW_RUNS,
        "expected_frozen_baseline_probes": EXPECTED_BASELINE_PROBES,
        "conditions": {
            condition: local_condition_definition(condition, 64.0)
            for condition in ALL_CONDITIONS
        },
        "variants": [
            {"variant": variant, "hidden_cap": hidden, "output_cap": output}
            for variant, hidden, output in VARIANTS
        ],
        "seeds": SEEDS,
        "input": "same scaled Raw64 weighted event sequence as Exp4.0.5",
        "architecture": "30 -> Local128(no recurrence) -> RSNN128 -> 12",
        "readouts": list(_readout_columns()),
        "primary_training_objective": "valid normalized output WholeCount CE",
        "linear_probe": (
            "train-only StandardScaler + balanced LogisticRegression on frozen representations"
        ),
        "files": {
            "runs": runs_file.name,
            "summary": summary_file.name,
            "paired_effects": effects_file.name,
            "paired_effects_summary": effect_summary_file.name,
        },
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    return {
        "runs": runs_file,
        "summary": summary_file,
        "paired_effects": effects_file,
        "paired_effects_summary": effect_summary_file,
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

    baseline_parser = subparsers.add_parser("probe-baseline")
    baseline_parser.add_argument("--array-task-id", type=int, required=True)
    baseline_parser.add_argument("--device", default="cpu")
    baseline_parser.add_argument("--threads", type=int, default=1)
    baseline_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    baseline_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = exp405.exp40.find_repo_root()
    root = results_dir(repo_root)
    if args.command == "finalize":
        paths = finalize_experiment(repo_root)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    data = prepare_raw64_data(repo_root)
    if args.command == "probe-baseline":
        specs = baseline_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(
                f"array task {args.array_task_id} outside [0, {len(specs) - 1}]"
            )
        torch.set_num_threads(args.threads)
        spec = specs[args.array_task_id]
        payload = evaluate_frozen_baseline(
            spec,
            data,
            repo_root,
            root,
            torch.device(args.device),
            args.batch_size,
            args.force,
        )
    else:
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(
                f"array task {args.array_task_id} outside [0, {len(specs) - 1}]"
            )
        spec = specs[args.array_task_id]
        config = Config(
            repo_root=repo_root,
            results_dir=root,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            threads=args.threads,
        )
        payload = run_one(spec, data, config, force=args.force)

    test = payload["metrics"]["test"]["valid_count"]["balanced_accuracy"]
    hcount = payload["linear_probes"]["metrics"]["test"][
        "hidden_whole_count_linear"
    ]["balanced_accuracy"]
    uend = payload["linear_probes"]["metrics"]["test"]["uend_linear"][
        "balanced_accuracy"
    ]
    print(
        f"completed {spec.key}: output_count_BA={test:.6f} "
        f"hidden_count_linear_BA={hcount:.6f} uend_linear_BA={uend:.6f}"
    )


if __name__ == "__main__":
    main()
