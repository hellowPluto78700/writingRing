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
from sklearn.linear_model import LogisticRegression

from scripts import experiment_4_0_fixed250_temporal_snn as exp40
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_4_2_continuous_recurrent_controls as exp42


EXPERIMENT_ID = "experiment_4_0_3_hierarchical_snn"
PROTOCOL_VERSION = "fixed250_hierarchical_multispike_v1"
INPUT_INFORMATION_CONTRACT = "same scaled Fixed250 vectors as Exp4.0/4.0.1/4.2; no sub-bin timing"

SEEDS = exp40.SEEDS
NEW_CONDITIONS = (
    "rsnn176_capacity",
    "local128_rsnn128",
    "rsnn128_rsnn128",
    "local128_ff128",
)
BASELINE_CONDITION = "rsnn128_multispike_reused"
EXPECTED_NEW_RUNS = len(NEW_CONDITIONS) * len(SEEDS)

HIDDEN_CAP = 31
OUTPUT_CAP = 31
STATE_TAU_MEM_MS = 250.0
LOCAL_BETA = 0.0
THRESHOLD = exp401.THRESHOLD
BATCH_SIZE = exp401.BATCH_SIZE
EPOCHS = exp401.EPOCHS
LR = exp401.LR
WEIGHT_DECAY = exp401.WEIGHT_DECAY
PROBE_MAX_ITER = exp40.LOGREG_MAX_ITER


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
    resume: bool = True
    threads: int = 1


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(condition, seed) for condition in NEW_CONDITIONS for seed in SEEDS]


def paired_seed(spec: RunSpec, role: str) -> int:
    """Pair data-order random streams across conditions for each master seed."""
    return exp40.base.dseed(spec.seed, "exp4_0_3_paired", role)


def state_beta() -> float:
    return math.exp(-exp40.FIXED_MS / STATE_TAU_MEM_MS)


def condition_definition(condition: str) -> dict[str, object]:
    if condition == "rsnn176_capacity":
        return {
            "layer1_width": 176,
            "layer1_recurrent": True,
            "layer1_beta": state_beta(),
            "layer2_width": None,
            "layer2_recurrent": False,
            "layer2_beta": None,
            "role": "parameter-matched single-layer recurrent capacity control",
        }
    if condition == "local128_rsnn128":
        return {
            "layer1_width": 128,
            "layer1_recurrent": False,
            "layer1_beta": LOCAL_BETA,
            "layer2_width": 128,
            "layer2_recurrent": True,
            "layer2_beta": state_beta(),
            "role": "primary: local spike-code formation then recurrent temporal integration",
        }
    if condition == "rsnn128_rsnn128":
        return {
            "layer1_width": 128,
            "layer1_recurrent": True,
            "layer1_beta": state_beta(),
            "layer2_width": 128,
            "layer2_recurrent": True,
            "layer2_beta": state_beta(),
            "role": "generic recurrent depth control",
        }
    if condition == "local128_ff128":
        return {
            "layer1_width": 128,
            "layer1_recurrent": False,
            "layer1_beta": LOCAL_BETA,
            "layer2_width": 128,
            "layer2_recurrent": False,
            "layer2_beta": state_beta(),
            "role": "two-layer no-learned-recurrence control",
        }
    raise ValueError(f"Unknown condition: {condition}")


def parameter_counts(condition: str, n_classes: int) -> dict[str, int]:
    definition = condition_definition(condition)
    h1 = int(definition["layer1_width"])
    h2_raw = definition["layer2_width"]
    input_layer1 = exp40.EVENT_CHANNELS * h1
    recurrent1 = h1 * h1 if bool(definition["layer1_recurrent"]) else 0
    if h2_raw is None:
        layer1_layer2 = 0
        recurrent2 = 0
        output = h1 * n_classes
    else:
        h2 = int(h2_raw)
        layer1_layer2 = h1 * h2
        recurrent2 = h2 * h2 if bool(definition["layer2_recurrent"]) else 0
        output = h2 * n_classes
    total = input_layer1 + recurrent1 + layer1_layer2 + recurrent2 + output
    return {
        "input_layer1": int(input_layer1),
        "recurrent1": int(recurrent1),
        "layer1_layer2": int(layer1_layer2),
        "recurrent2": int(recurrent2),
        "output": int(output),
        "total": int(total),
    }


class HierarchicalMacroDecoder(nn.Module):
    """Fixed250 multi-event SNN with explicit representation/memory roles.

    The input is exactly the zero-preserving scaled Fixed250 sequence from
    Exp4.0. No raw or sub-bin timing information is reconstructed here.

    For the primary hierarchy, layer 1 has beta=0 and no learned recurrence,
    so every macro bin is independently converted into a population event code.
    Layer 2 then owns the recurrent temporal state. All variants use the same
    cap-31 MacroMultiSpikeLIF formulation established in Exp4.0.1.
    """

    def __init__(self, condition: str, n_classes: int) -> None:
        super().__init__()
        definition = condition_definition(condition)
        self.condition = condition
        self.n_classes = int(n_classes)
        self.layer1_width = int(definition["layer1_width"])
        self.layer2_width = (
            int(definition["layer2_width"])
            if definition["layer2_width"] is not None
            else None
        )
        self.state_width = self.layer2_width or self.layer1_width

        self.input_layer1 = nn.Linear(
            exp40.EVENT_CHANNELS, self.layer1_width, bias=False
        )
        self.recurrent1 = (
            nn.Linear(self.layer1_width, self.layer1_width, bias=False)
            if bool(definition["layer1_recurrent"])
            else None
        )
        self.layer1_lif = exp401.MacroMultiSpikeLIF(
            beta=float(definition["layer1_beta"]),
            threshold=THRESHOLD,
            max_spikes_per_dt=HIDDEN_CAP,
        )

        if self.layer2_width is None:
            self.layer1_layer2 = None
            self.recurrent2 = None
            self.layer2_lif = None
        else:
            self.layer1_layer2 = nn.Linear(
                self.layer1_width, self.layer2_width, bias=False
            )
            self.recurrent2 = (
                nn.Linear(self.layer2_width, self.layer2_width, bias=False)
                if bool(definition["layer2_recurrent"])
                else None
            )
            self.layer2_lif = exp401.MacroMultiSpikeLIF(
                beta=float(definition["layer2_beta"]),
                threshold=THRESHOLD,
                max_spikes_per_dt=HIDDEN_CAP,
            )

        self.state_output = nn.Linear(self.state_width, n_classes, bias=False)
        self.output_lif = exp401.MacroMultiSpikeLIF(
            beta=math.exp(-exp40.FIXED_MS / exp40.OUTPUT_TAU_MEM_MS),
            threshold=THRESHOLD,
            max_spikes_per_dt=OUTPUT_CAP,
        )

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, n_bins, channels = x.shape
        if channels != exp40.EVENT_CHANNELS:
            raise ValueError(
                f"Expected Fixed250 vectors with {exp40.EVENT_CHANNELS} channels, got {channels}"
            )

        layer1_mem = torch.zeros(
            batch, self.layer1_width, device=x.device, dtype=x.dtype
        )
        prev_layer1_spikes = torch.zeros_like(layer1_mem)
        if self.layer2_width is not None:
            layer2_mem = torch.zeros(
                batch, self.layer2_width, device=x.device, dtype=x.dtype
            )
            prev_layer2_spikes = torch.zeros_like(layer2_mem)
        else:
            layer2_mem = None
            prev_layer2_spikes = None
        output_mem = torch.zeros(
            batch, self.n_classes, device=x.device, dtype=x.dtype
        )

        layer1_spikes_seq: list[torch.Tensor] = []
        layer1_pre_seq: list[torch.Tensor] = []
        state_spikes_seq: list[torch.Tensor] = []
        state_mems_seq: list[torch.Tensor] = []
        state_pre_seq: list[torch.Tensor] = []
        output_spikes_seq: list[torch.Tensor] = []
        output_mems_seq: list[torch.Tensor] = []
        output_pre_seq: list[torch.Tensor] = []

        for b in range(n_bins):
            current1 = self.input_layer1(x[:, b])
            if self.recurrent1 is not None:
                current1 = current1 + self.recurrent1(prev_layer1_spikes)
            spikes1, layer1_mem, pre1 = self.layer1_lif(current1, layer1_mem)

            if self.layer2_width is None:
                state_spikes = spikes1
                state_mem = layer1_mem
                state_pre = pre1
            else:
                if self.layer1_layer2 is None or self.layer2_lif is None:
                    raise RuntimeError("Layer-2 modules are incomplete")
                current2 = self.layer1_layer2(spikes1)
                if self.recurrent2 is not None:
                    if prev_layer2_spikes is None:
                        raise RuntimeError("Missing recurrent layer-2 state")
                    current2 = current2 + self.recurrent2(prev_layer2_spikes)
                if layer2_mem is None:
                    raise RuntimeError("Missing layer-2 membrane state")
                state_spikes, layer2_mem, state_pre = self.layer2_lif(
                    current2, layer2_mem
                )
                state_mem = layer2_mem
                prev_layer2_spikes = state_spikes

            output_current = self.state_output(state_spikes)
            output_spikes, output_mem, output_pre = self.output_lif(
                output_current, output_mem
            )

            layer1_spikes_seq.append(spikes1)
            layer1_pre_seq.append(pre1)
            state_spikes_seq.append(state_spikes)
            state_mems_seq.append(state_mem)
            state_pre_seq.append(state_pre)
            output_spikes_seq.append(output_spikes)
            output_mems_seq.append(output_mem)
            output_pre_seq.append(output_pre)
            prev_layer1_spikes = spikes1

        return {
            "layer1_spikes": torch.stack(layer1_spikes_seq, dim=1),
            "layer1_pre_reset": torch.stack(layer1_pre_seq, dim=1),
            "state_spikes": torch.stack(state_spikes_seq, dim=1),
            "state_membranes": torch.stack(state_mems_seq, dim=1),
            "state_pre_reset": torch.stack(state_pre_seq, dim=1),
            "output_spikes": torch.stack(output_spikes_seq, dim=1),
            "output_membranes": torch.stack(output_mems_seq, dim=1),
            "output_pre_reset": torch.stack(output_pre_seq, dim=1),
        }


def valid_final_state(
    state_membranes: torch.Tensor,
    valid_bins: torch.Tensor,
) -> torch.Tensor:
    return exp40.valid_final_membrane(state_membranes, valid_bins)


def evaluate_model(
    model: HierarchicalMacroDecoder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    prediction_parts: dict[str, list[np.ndarray]] = {
        "valid_count": [],
        "full_count": [],
        "valid_output_membrane": [],
        "full_output_membrane": [],
    }
    loss_sum = 0.0
    n_total = 0
    layer1_stat_sums: dict[str, float] = {}
    state_stat_sums: dict[str, float] = {}
    output_stat_sums: dict[str, float] = {}
    stat_weight = 0
    state_tail_weighted = 0.0
    output_tail_weighted = 0.0

    with torch.no_grad():
        for X, y, valid_bins in data_loader:
            X = X.to(device)
            y = y.to(device)
            valid_bins = valid_bins.to(device)
            trajectories = model.forward_trajectory(X)
            output_spikes = trajectories["output_spikes"]
            output_mems = trajectories["output_membranes"]
            valid_logits = exp401.valid_evidence(
                output_spikes, valid_bins, OUTPUT_CAP
            )
            full_logits = exp401.full_evidence(output_spikes, OUTPUT_CAP)
            valid_output_mem = exp40.valid_final_membrane(output_mems, valid_bins)
            full_output_mem = output_mems[:, -1]
            loss = F.cross_entropy(valid_logits, y)

            n = len(y)
            n_total += n
            loss_sum += float(loss.item()) * n
            y_true_parts.append(y.cpu().numpy())
            for name, values in (
                ("valid_count", valid_logits),
                ("full_count", full_logits),
                ("valid_output_membrane", valid_output_mem),
                ("full_output_membrane", full_output_mem),
            ):
                prediction_parts[name].append(values.argmax(dim=1).cpu().numpy())

            layer1_stats = exp401._distribution_stats(
                trajectories["layer1_spikes"],
                trajectories["layer1_pre_reset"],
                valid_bins,
                HIDDEN_CAP,
            )
            state_stats = exp401._distribution_stats(
                trajectories["state_spikes"],
                trajectories["state_pre_reset"],
                valid_bins,
                HIDDEN_CAP,
            )
            output_stats = exp401._distribution_stats(
                output_spikes,
                trajectories["output_pre_reset"],
                valid_bins,
                OUTPUT_CAP,
            )
            for target, source in (
                (layer1_stat_sums, layer1_stats),
                (state_stat_sums, state_stats),
                (output_stat_sums, output_stats),
            ):
                for key, value in source.items():
                    target[key] = target.get(key, 0.0) + float(value) * n
            state_tail_weighted += exp401._tail_fraction(
                trajectories["state_spikes"], valid_bins
            ) * n
            output_tail_weighted += exp401._tail_fraction(
                output_spikes, valid_bins
            ) * n
            stat_weight += n

    y_true = np.concatenate(y_true_parts)
    result: dict[str, object] = {
        "valid_count_loss": float(loss_sum / max(n_total, 1)),
        "layer1_valid_stats": {
            key: float(value / max(stat_weight, 1))
            for key, value in layer1_stat_sums.items()
        },
        "state_valid_stats": {
            key: float(value / max(stat_weight, 1))
            for key, value in state_stat_sums.items()
        },
        "output_valid_stats": {
            key: float(value / max(stat_weight, 1))
            for key, value in output_stat_sums.items()
        },
        "state_tail_event_fraction": float(
            state_tail_weighted / max(stat_weight, 1)
        ),
        "output_tail_event_fraction": float(
            output_tail_weighted / max(stat_weight, 1)
        ),
    }
    for name, parts in prediction_parts.items():
        result[name] = exp40.metrics(y_true, np.concatenate(parts))
    return result


def extract_endpoint_states(
    model: HierarchicalMacroDecoder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    state_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    with torch.no_grad():
        for X, y, valid_bins in data_loader:
            X = X.to(device)
            valid_bins = valid_bins.to(device)
            trajectories = model.forward_trajectory(X)
            endpoint = valid_final_state(trajectories["state_membranes"], valid_bins)
            state_parts.append(endpoint.cpu().numpy())
            label_parts.append(y.numpy())
    return np.concatenate(state_parts), np.concatenate(label_parts)


def fit_endpoint_state_probe(
    model: HierarchicalMacroDecoder,
    eval_loaders: list[torch.utils.data.DataLoader],
    device: torch.device,
) -> dict[str, object]:
    extracted = [extract_endpoint_states(model, loader, device) for loader in eval_loaders]
    Xtr, ytr = extracted[0]
    probe = LogisticRegression(
        max_iter=PROBE_MAX_ITER,
        class_weight="balanced",
        solver="lbfgs",
    )
    probe.fit(Xtr, ytr)
    metrics_by_split: dict[str, object] = {}
    for split, (X, y) in zip(("train", "val", "test"), extracted, strict=True):
        metrics_by_split[split] = exp40.metrics(y, probe.predict(X))
    return {
        "feature": "causal endpoint membrane of final temporal SNN layer only",
        "feature_dim": int(Xtr.shape[1]),
        "temporal_phase_access": False,
        "classifier": "balanced LogisticRegression(lbfgs)",
        "metrics": metrics_by_split,
    }


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _provenance(
    spec: RunSpec,
    data: exp40.BinnedData,
    config: Config,
) -> dict[str, object]:
    definition = condition_definition(spec.condition)
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "condition": spec.condition,
        "condition_definition": definition,
        "input_information_contract": INPUT_INFORMATION_CONTRACT,
        "seed": spec.seed,
        "split_seed": int(exp40.base.SPLIT_SEED),
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
        "event_channels": exp40.EVENT_CHANNELS,
        "fixed_bin_ms": exp40.FIXED_MS,
        "n_padded_bins": data.n_bins,
        "channel_scale": data.channel_scale.tolist(),
        "hidden_cap": HIDDEN_CAP,
        "output_cap": OUTPUT_CAP,
        "output_readout_normalization": "divide output event counts by 31 before WholeCount CE",
        "primary_objective": "valid normalized fully-spiking WholeCount CE",
        "checkpoint_selection": "validation valid-count BA, validation loss tie-break",
        "state_probe": "post-hoc only; final temporal-layer endpoint membrane; no phase access",
        "state_dynamics": "all padded bins execute; zero input after endpoint; no state freeze",
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "parameter_counts": parameter_counts(spec.condition, len(data.labels)),
    }


def run_one(
    spec: RunSpec,
    data: exp40.BinnedData,
    config: Config,
) -> dict[str, object]:
    path = evaluation_path(config.results_dir, spec)
    ckpt = checkpoint_path(config.results_dir, spec)
    if config.resume and path.exists() and ckpt.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp40.base.seed_all(paired_seed(spec, "model_init"))
    model = HierarchicalMacroDecoder(spec.condition, len(data.labels)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    partitions = [
        (data.Xtr, data.ytr, data.btr),
        (data.Xva, data.yva, data.bva),
        (data.Xte, data.yte, data.bte),
    ]
    train_loader = exp40.loader(
        *partitions[0],
        config.batch_size,
        True,
        paired_seed(spec, "train_loader"),
    )
    eval_loaders = [
        exp40.loader(
            *partition,
            config.batch_size,
            False,
            exp40.base.dseed(spec.seed, "exp4_0_3_eval", split),
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
            trajectories = model.forward_trajectory(X)
            logits = exp401.valid_evidence(
                trajectories["output_spikes"], valid_bins, OUTPUT_CAP
            )
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
    state_probe = fit_endpoint_state_probe(model, eval_loaders, device)
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
        "state_probe": state_probe,
        "history": history,
    }
    _save_json(path, payload)
    return payload


def _exp401_baseline_spec(seed: int) -> exp401.RunSpec:
    return exp401.RunSpec(
        architecture="rsnn",
        hidden_width=128,
        tau_mem_ms=250.0,
        variant="multi_ho",
        hidden_cap=31,
        output_cap=31,
        seed=seed,
    )


def _baseline_hidden_endpoint_states(
    model: exp401.MacroTemporalDecoder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    state_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, valid_bins in data_loader:
            X = X.to(device)
            valid_bins = valid_bins.to(device)
            batch, n_bins, _ = X.shape
            hidden_mem = torch.zeros(
                batch, model.hidden_width, device=device, dtype=X.dtype
            )
            prev_hidden_spikes = torch.zeros_like(hidden_mem)
            hidden_mems: list[torch.Tensor] = []
            for b in range(n_bins):
                current = model.input_hidden(X[:, b])
                if model.recurrent is not None:
                    current = current + model.recurrent(prev_hidden_spikes)
                hidden_spikes, hidden_mem, _ = model.hidden_lif(current, hidden_mem)
                hidden_mems.append(hidden_mem)
                prev_hidden_spikes = hidden_spikes
            sequence = torch.stack(hidden_mems, dim=1)
            endpoint = exp40.valid_final_membrane(sequence, valid_bins)
            state_parts.append(endpoint.cpu().numpy())
            label_parts.append(y.numpy())
    return np.concatenate(state_parts), np.concatenate(label_parts)


def probe_reused_baseline(data: exp40.BinnedData, config: Config, force: bool) -> None:
    path = config.results_dir / "baseline" / "reused_rsnn128_multispike_state_probe.json"
    if path.exists() and not force:
        print(path)
        return

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    partitions = [
        (data.Xtr, data.ytr, data.btr),
        (data.Xva, data.yva, data.bva),
        (data.Xte, data.yte, data.bte),
    ]
    rows: list[dict[str, object]] = []
    source_root = exp401.results_dir(config.repo_root)
    for seed in SEEDS:
        source_spec = _exp401_baseline_spec(seed)
        source_ckpt = exp401.checkpoint_path(source_root, source_spec)
        if not source_ckpt.exists():
            raise FileNotFoundError(f"Missing reused Exp4.0.1 checkpoint: {source_ckpt}")
        checkpoint = torch.load(source_ckpt, map_location=device, weights_only=False)
        model = exp401.MacroTemporalDecoder(
            architecture="rsnn",
            hidden_width=128,
            tau_mem_ms=250.0,
            n_classes=len(data.labels),
            hidden_cap=31,
            output_cap=31,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        loaders = [
            exp40.loader(
                *partition,
                config.batch_size,
                False,
                exp40.base.dseed(seed, "exp4_0_3_baseline_probe", split),
            )
            for partition, split in zip(
                partitions, ("train", "val", "test"), strict=True
            )
        ]
        extracted = [
            _baseline_hidden_endpoint_states(model, loader, device)
            for loader in loaders
        ]
        Xtr, ytr = extracted[0]
        probe = LogisticRegression(
            max_iter=PROBE_MAX_ITER,
            class_weight="balanced",
            solver="lbfgs",
        )
        probe.fit(Xtr, ytr)
        for split, (X, y) in zip(("train", "val", "test"), extracted, strict=True):
            row: dict[str, object] = {
                "condition": BASELINE_CONDITION,
                "seed": seed,
                "split": split,
                "feature_dim": int(Xtr.shape[1]),
            }
            row.update(
                {
                    f"probe_{key}": value
                    for key, value in exp40.metrics(y, probe.predict(X)).items()
                }
            )
            rows.append(row)
    _save_json(
        path,
        {
            "source_experiment": exp401.EXPERIMENT_ID,
            "source_protocol": exp401.PROTOCOL_VERSION,
            "source_condition": "rsnn H128 tau250 multi_ho",
            "feature": "causal endpoint hidden membrane only",
            "temporal_phase_access": False,
            "rows": rows,
        },
    )
    print(path)


def _flatten_new_payload(
    spec: RunSpec,
    payload: dict[str, object],
    split: str,
) -> dict[str, object]:
    metrics = payload["metrics"][split]
    probe_metrics = payload["state_probe"]["metrics"][split]
    row: dict[str, object] = {
        "condition": spec.condition,
        "seed": spec.seed,
        "split": split,
        "best_epoch": payload["best_epoch"],
        "parameter_count": payload["provenance"]["parameter_counts"]["total"],
        "state_probe_feature_dim": payload["state_probe"]["feature_dim"],
        "state_tail_event_fraction": metrics["state_tail_event_fraction"],
        "output_tail_event_fraction": metrics["output_tail_event_fraction"],
    }
    for readout in (
        "valid_count",
        "full_count",
        "valid_output_membrane",
        "full_output_membrane",
    ):
        for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
            row[f"{readout}_{metric_name}"] = metrics[readout][metric_name]
    for prefix, source in (
        ("layer1", metrics["layer1_valid_stats"]),
        ("state", metrics["state_valid_stats"]),
        ("output", metrics["output_valid_stats"]),
    ):
        for name, value in source.items():
            row[f"{prefix}_{name}"] = value
    for name, value in probe_metrics.items():
        row[f"state_probe_{name}"] = value
    return row


def _load_reused_baseline_rows(config: Config) -> list[dict[str, object]]:
    source_runs_path = exp401.results_dir(config.repo_root) / "runs.csv"
    probe_path = (
        config.results_dir / "baseline" / "reused_rsnn128_multispike_state_probe.json"
    )
    if not source_runs_path.exists():
        raise FileNotFoundError(f"Missing Exp4.0.1 finalized runs: {source_runs_path}")
    if not probe_path.exists():
        raise FileNotFoundError(f"Missing reused baseline state probe: {probe_path}")

    source = pd.read_csv(source_runs_path)
    source = source[
        (source["architecture"] == "rsnn")
        & (source["hidden_width"] == 128)
        & np.isclose(source["tau_mem_ms"], 250.0)
        & (source["variant"] == "multi_ho")
        & (source["hidden_cap"] == 31)
        & (source["output_cap"] == 31)
    ].copy()
    if len(source) != len(SEEDS) * 3:
        raise ValueError(
            f"Expected {len(SEEDS) * 3} reused baseline rows, got {len(source)}"
        )
    probe_payload = json.loads(probe_path.read_text(encoding="utf-8"))
    probe_lookup = {
        (int(row["seed"]), str(row["split"])): row
        for row in probe_payload["rows"]
    }

    rows: list[dict[str, object]] = []
    for source_row in source.itertuples(index=False):
        key = (int(source_row.seed), str(source_row.split))
        probe = probe_lookup[key]
        row: dict[str, object] = {
            "condition": BASELINE_CONDITION,
            "seed": int(source_row.seed),
            "split": str(source_row.split),
            "best_epoch": int(source_row.best_epoch),
            "parameter_count": int(source_row.parameter_count),
            "state_probe_feature_dim": int(probe["feature_dim"]),
            "state_tail_event_fraction": float(source_row.hidden_tail_event_fraction),
            "output_tail_event_fraction": float(source_row.output_tail_event_fraction),
        }
        for readout in ("valid_count", "full_count", "valid_membrane", "full_membrane"):
            mapped = {
                "valid_count": "valid_count",
                "full_count": "full_count",
                "valid_membrane": "valid_output_membrane",
                "full_membrane": "full_output_membrane",
            }[readout]
            for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
                row[f"{mapped}_{metric_name}"] = float(
                    getattr(source_row, f"{readout}_{metric_name}")
                )
        for source_prefix, target_prefix in (
            ("hidden", "state"),
            ("output", "output"),
        ):
            for stat_name in (
                "mean_events_per_neuron_step",
                "fraction_zero",
                "fraction_one",
                "fraction_gt1",
                "fraction_ge4",
                "fraction_at_cap",
                "pre_reset_membrane_mean",
                "pre_reset_membrane_max",
            ):
                source_name = f"{source_prefix}_{stat_name}"
                if hasattr(source_row, source_name):
                    row[f"{target_prefix}_{stat_name}"] = float(
                        getattr(source_row, source_name)
                    )
        row["state_probe_accuracy"] = float(probe["probe_accuracy"])
        row["state_probe_balanced_accuracy"] = float(
            probe["probe_balanced_accuracy"]
        )
        row["state_probe_macro_f1"] = float(probe["probe_macro_f1"])
        rows.append(row)
    return rows


def _write_comparison(test: pd.DataFrame, root: Path) -> None:
    summary = (
        test.groupby("condition")[[
            "valid_count_balanced_accuracy",
            "valid_count_macro_f1",
            "state_probe_balanced_accuracy",
            "state_probe_macro_f1",
            "state_tail_event_fraction",
            "output_tail_event_fraction",
        ]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(root / "summary.csv", index=False)

    baseline = test[test["condition"] == BASELINE_CONDITION]
    if len(baseline) != len(SEEDS):
        raise ValueError("Reused baseline must contain exactly five test rows")
    baseline_by_seed = {
        int(row.seed): row for row in baseline.itertuples(index=False)
    }
    effect_rows: list[dict[str, object]] = []
    for condition in NEW_CONDITIONS:
        group = test[test["condition"] == condition]
        for row in group.itertuples(index=False):
            ref = baseline_by_seed[int(row.seed)]
            effect_rows.append(
                {
                    "condition": condition,
                    "seed": int(row.seed),
                    "fully_spiking_ba": float(row.valid_count_balanced_accuracy),
                    "baseline_fully_spiking_ba": float(ref.valid_count_balanced_accuracy),
                    "fully_spiking_delta_vs_baseline": float(
                        row.valid_count_balanced_accuracy
                        - ref.valid_count_balanced_accuracy
                    ),
                    "state_probe_ba": float(row.state_probe_balanced_accuracy),
                    "baseline_state_probe_ba": float(
                        ref.state_probe_balanced_accuracy
                    ),
                    "state_probe_delta_vs_baseline": float(
                        row.state_probe_balanced_accuracy
                        - ref.state_probe_balanced_accuracy
                    ),
                }
            )
    pd.DataFrame(effect_rows).to_csv(root / "paired_effects.csv", index=False)


def _write_external_references(config: Config, test: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    linear_path = exp40.results_dir(config.repo_root) / "baseline" / "fixed250_linear.json"
    if linear_path.exists():
        payload = json.loads(linear_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "reference": "fixed250_linear",
                "mean_test_ba": float(payload["metrics"]["test"]["balanced_accuracy"]),
                "source": str(linear_path.relative_to(config.repo_root)),
            }
        )

    rnn_runs_path = exp42.results_dir(config.repo_root) / "runs.csv"
    if rnn_runs_path.exists():
        rnn = pd.read_csv(rnn_runs_path)
        rnn = rnn[
            (rnn["model_type"] == "rnn")
            & (rnn["hidden_width"] == 128)
            & (rnn["objective"] == "sum_logits_ce")
            & (rnn["split"] == "test")
        ]
        if len(rnn) == len(SEEDS):
            rows.append(
                {
                    "reference": "continuous_rnn128_sum_logits",
                    "mean_test_ba": float(
                        rnn["valid_sum_logits_balanced_accuracy"].mean()
                    ),
                    "source": str(rnn_runs_path.relative_to(config.repo_root)),
                }
            )

    for condition, group in test.groupby("condition"):
        rows.append(
            {
                "reference": str(condition),
                "mean_test_ba": float(group["valid_count_balanced_accuracy"].mean()),
                "source": "Exp4.0.3 finalized runs",
            }
        )
    pd.DataFrame(rows).to_csv(config.results_dir / "references.csv", index=False)


def finalize(data: exp40.BinnedData, config: Config) -> None:
    missing = [
        str(evaluation_path(config.results_dir, spec))
        for spec in run_specs()
        if not evaluation_path(config.results_dir, spec).exists()
    ]
    probe_path = (
        config.results_dir / "baseline" / "reused_rsnn128_multispike_state_probe.json"
    )
    if not probe_path.exists():
        missing.append(str(probe_path))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Cannot finalize: {len(missing)} required artifacts are missing.\n{preview}"
        )

    rows: list[dict[str, object]] = []
    for spec in run_specs():
        payload = json.loads(
            evaluation_path(config.results_dir, spec).read_text(encoding="utf-8")
        )
        for split in ("train", "val", "test"):
            rows.append(_flatten_new_payload(spec, payload, split))
    rows.extend(_load_reused_baseline_rows(config))

    runs = pd.DataFrame(rows)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "runs.csv", index=False)
    test = runs[runs["split"] == "test"].copy()
    _write_comparison(test, config.results_dir)
    _write_external_references(config, test)

    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "input_information_contract": INPUT_INFORMATION_CONTRACT,
        "new_conditions": list(NEW_CONDITIONS),
        "reused_baseline": BASELINE_CONDITION,
        "expected_new_runs": EXPECTED_NEW_RUNS,
        "seeds": list(SEEDS),
        "split_seed": int(exp40.base.SPLIT_SEED),
        "fixed_bin_ms": exp40.FIXED_MS,
        "event_channels": exp40.EVENT_CHANNELS,
        "hidden_cap": HIDDEN_CAP,
        "output_cap": OUTPUT_CAP,
        "state_probe": "post-hoc endpoint state only; no phase access; never used for checkpoint selection",
        "primary_readout": "fully-spiking normalized valid WholeCount",
        "state_dynamics": "all padded bins execute; zero input after endpoint; no state freeze",
        "channel_scale": data.channel_scale.tolist(),
        "parameter_counts": {
            condition: parameter_counts(condition, len(data.labels))
            for condition in NEW_CONDITIONS
        },
    }
    _save_json(config.results_dir / "provenance.json", provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 4.0.3 hierarchical multi-spike Fixed250 SNN"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--epochs", type=int, default=EPOCHS)
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--force-retrain", action="store_true")

    probe = sub.add_parser("probe-baseline")
    probe.add_argument("--device", default="cpu")
    probe.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    probe.add_argument("--threads", type=int, default=1)
    probe.add_argument("--force", action="store_true")

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
                f"array-task-id {args.array_task_id} outside [0,{len(specs) - 1}]"
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
        print(
            json.dumps(
                {"completed": spec.key, "best_epoch": result["best_epoch"]},
                indent=2,
            )
        )
        return

    if args.command == "probe-baseline":
        config = Config(
            repo_root=repo_root,
            results_dir=root,
            device=args.device,
            batch_size=args.batch_size,
            threads=args.threads,
        )
        probe_reused_baseline(data, config, force=args.force)
        return

    config = Config(repo_root=repo_root, results_dir=root, device=args.device)
    finalize(data, config)
    print(root / "summary.csv")


if __name__ == "__main__":
    main()
