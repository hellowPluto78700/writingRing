from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_3_0_2_hidden_multitau_architectures as exp302
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73


EXPERIMENT_ID = "experiment_7_3_9_pretrained_a2_depth_extension"
PROTOCOL_VERSION = "pretrained_a2_depth_extension_v1"

SOURCE_EXPERIMENT_ID = exp73.EXPERIMENT_ID
SOURCE_PROTOCOL_VERSION = exp73.PROTOCOL_VERSION
SOURCE_METHOD = "A2_e2e_linear_wcce"
SOURCE_ARCHITECTURE = exp73.ARCHITECTURE
SOURCE_OBJECTIVE = "wcce"

C0_CASE = "C0_a2_2layer"
C1_CASE = "C1_frozen_a2_train_l3"
C2_CASE = "C2_c1_init_e2e_3layer"
CASES = (C0_CASE, C1_CASE, C2_CASE)
TRAIN_CASES = (C1_CASE, C2_CASE)

ARCHITECTURE_2L = "234x234"
ARCHITECTURE_3L = "234x234x234"
SHIFTS = ((2, 3, 4), (2, 3, 4), (2, 3, 4))
LAYERS_2L = ("L1", "L2")
LAYERS_3L = ("L1", "L2", "L3")
SEEDS = exp73.SEEDS
HIDDEN_WIDTH = exp73.HIDDEN_WIDTH
THRESHOLD = exp73.THRESHOLD
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
BATCH_SIZE = exp72.BATCH_SIZE

PROBE_REPRESENTATIONS = ("wholecount", "fixed250")
PROBE_BIAS_MODES = ("affine", "no_bias")
PROBE_C_GRID = tuple(float(v) for v in exp302.PROBE_C_GRID)
PROBE_MAX_ITER = 5000

EXPECTED_C1_RUNS = len(SEEDS)
EXPECTED_C2_RUNS = len(SEEDS)
EXPECTED_PROBE_TASKS = len(CASES) * len(SEEDS)
EXPECTED_PROBE_ROWS = (
    len(SEEDS) * len(LAYERS_2L) * len(PROBE_REPRESENTATIONS) * len(PROBE_BIAS_MODES)
    + 2
    * len(SEEDS)
    * len(LAYERS_3L)
    * len(PROBE_REPRESENTATIONS)
    * len(PROBE_BIAS_MODES)
)


@dataclass(frozen=True)
class TrainSpec:
    case: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.case}__seed{self.seed}"


@dataclass(frozen=True)
class ProbeSpec:
    case: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.case}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def c1_specs() -> list[TrainSpec]:
    return [TrainSpec(C1_CASE, seed) for seed in SEEDS]


def c2_specs() -> list[TrainSpec]:
    return [TrainSpec(C2_CASE, seed) for seed in SEEDS]


def probe_specs() -> list[ProbeSpec]:
    return [ProbeSpec(case, seed) for case in CASES for seed in SEEDS]


def case_layers(case: str) -> tuple[str, ...]:
    if case == C0_CASE:
        return LAYERS_2L
    if case in (C1_CASE, C2_CASE):
        return LAYERS_3L
    raise ValueError(case)


def validate_train_spec(spec: TrainSpec) -> None:
    if spec.case not in TRAIN_CASES:
        raise ValueError(spec.case)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def validate_probe_spec(spec: ProbeSpec) -> None:
    if spec.case not in CASES:
        raise ValueError(spec.case)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _source_spec(seed: int) -> exp73.E2ESpec:
    return exp73.E2ESpec(seed, "linear", SOURCE_OBJECTIVE)


def _source_checkpoint_path(repo_root: Path, seed: int) -> Path:
    spec = _source_spec(seed)
    return exp73.results_dir(repo_root) / "e2e_checkpoints" / f"{spec.key}.pt"


def _source_evaluation_path(repo_root: Path, seed: int) -> Path:
    spec = _source_spec(seed)
    return exp73.results_dir(repo_root) / "e2e_evaluations" / f"{spec.key}.json"


def _checkpoint_path(root: Path, case: str, seed: int) -> Path:
    return root / case / "checkpoints" / f"{case}__seed{seed}.pt"


def _evaluation_path(root: Path, case: str, seed: int) -> Path:
    return root / case / "evaluations" / f"{case}__seed{seed}.json"


def _history_path(root: Path, case: str, seed: int) -> Path:
    return root / case / "histories" / f"{case}__seed{seed}.csv"


def _probe_path(root: Path, spec: ProbeSpec) -> Path:
    return root / "probe_evaluations" / f"{spec.key}.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_source_model(
    config: Config,
    data: exp3.Data,
    seed: int,
) -> tuple[exp73.Exp73Net, dict[str, Any], Path]:
    checkpoint_path = _source_checkpoint_path(config.repo_root, seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Exp7.3 A2 checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected source checkpoint payload: {checkpoint_path}")
    if payload.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError(f"Unexpected source experiment in {checkpoint_path}")
    if payload.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise ValueError(f"Unexpected source protocol in {checkpoint_path}")
    expected_spec = asdict(_source_spec(seed))
    if payload.get("spec") != expected_spec:
        raise ValueError(
            f"A2 checkpoint identity mismatch for seed {seed}: "
            f"{payload.get('spec')!r} != {expected_spec!r}"
        )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Missing model_state_dict: {checkpoint_path}")

    model = exp73.Exp73Net("linear", len(data.labels), data.fs).to(config.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, payload, checkpoint_path


def prepare_source(config: Config) -> dict[str, Any]:
    data = exp3.prepare_data(config.repo_root)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        model, checkpoint, checkpoint_path = _load_source_model(config, data, seed)
        del model
        evaluation_path = _source_evaluation_path(config.repo_root, seed)
        evaluation = _load_json(evaluation_path)
        if evaluation.get("experiment_id") != SOURCE_EXPERIMENT_ID:
            raise ValueError(f"Unexpected source evaluation experiment: {evaluation_path}")
        if evaluation.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
            raise ValueError(f"Unexpected source evaluation protocol: {evaluation_path}")
        if evaluation.get("method") != SOURCE_METHOD:
            raise ValueError(f"Unexpected source method: {evaluation.get('method')!r}")
        if evaluation.get("spec") != asdict(_source_spec(seed)):
            raise ValueError(f"Source evaluation identity mismatch: {evaluation_path}")
        rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path.relative_to(config.repo_root)),
                "evaluation": str(evaluation_path.relative_to(config.repo_root)),
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_val_ba": float(checkpoint["best_val_ba"]),
                "native_test_ba": float(
                    evaluation["native_metrics"]["test"]["balanced_accuracy"]
                ),
            }
        )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_protocol_version": SOURCE_PROTOCOL_VERSION,
        "source_method": SOURCE_METHOD,
        "source_architecture": SOURCE_ARCHITECTURE,
        "source_objective": SOURCE_OBJECTIVE,
        "seeds": list(SEEDS),
        "sources": rows,
    }
    _save_json(config.results_dir / "source_manifest.json", payload)
    return payload


class DepthExtensionNet(nn.Module):
    def __init__(self, n_classes: int, fs: float) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        self.hidden_linears = nn.ModuleList(
            [
                nn.Linear(exp72.EXPECTED_CHANNELS, HIDDEN_WIDTH, bias=False),
                nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False),
                nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False),
            ]
        )
        beta_hidden = math.exp(-(1000.0 / fs) / exp72.TAU_MEM_MS)
        self.hidden_lifs = nn.ModuleList(
            [
                exp401.MacroMultiSpikeLIF(
                    beta=beta_hidden,
                    threshold=THRESHOLD,
                    max_spikes_per_dt=1,
                    surrogate_slope=exp72.SURROGATE_SLOPE,
                )
                for _ in range(3)
            ]
        )
        for index, shifts in enumerate(SHIFTS):
            self.register_buffer(
                f"alpha_{index}",
                exp50.alpha_vector(HIDDEN_WIDTH, shifts),
            )
        self.output_linear = nn.Linear(HIDDEN_WIDTH, n_classes, bias=False)

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != exp72.EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected {exp72.EXPECTED_CHANNELS} channels, got {channels}"
            )
        syn = [
            torch.zeros(batch, HIDDEN_WIDTH, device=x.device, dtype=x.dtype)
            for _ in range(3)
        ]
        mem = [torch.zeros_like(syn[0]) for _ in range(3)]
        hidden: list[list[torch.Tensor]] = [[], [], []]
        evidence_steps: list[torch.Tensor] = []

        for timestep in range(steps):
            current = x[:, timestep]
            for layer_index in range(3):
                alpha = getattr(self, f"alpha_{layer_index}")
                syn[layer_index] = (
                    alpha * syn[layer_index]
                    + self.hidden_linears[layer_index](current)
                )
                spike, mem[layer_index], _ = self.hidden_lifs[layer_index](
                    syn[layer_index], mem[layer_index]
                )
                hidden[layer_index].append(spike)
                current = spike
            evidence_steps.append(self.output_linear(current))

        return {
            "hidden_spikes": tuple(torch.stack(values, dim=1) for values in hidden),
            "evidence": torch.stack(evidence_steps, dim=1),
        }


def _l3_init_seed(seed: int) -> int:
    return exp3.dseed(seed, EXPERIMENT_ID, "l3_init")


def _copy_a2_into_three_layer(
    model: DepthExtensionNet,
    source: exp73.Exp73Net,
    seed: int,
) -> None:
    with torch.no_grad():
        model.hidden_linears[0].weight.copy_(source.hidden_linears[0].weight)
        model.hidden_linears[1].weight.copy_(source.hidden_linears[1].weight)
        model.output_linear.weight.copy_(source.output_linear.weight)
    exp3.seed_all(_l3_init_seed(seed))
    model.hidden_linears[2].reset_parameters()

    if not torch.equal(model.alpha_0, source.alpha_0):
        raise RuntimeError("L1 alpha mismatch with source A2")
    if not torch.equal(model.alpha_1, source.alpha_1):
        raise RuntimeError("L2 alpha mismatch with source A2")
    if not torch.equal(model.alpha_2, source.alpha_1):
        raise RuntimeError("L3 alpha must match A2 L2 (2,3,4)")


def _configure_trainable(model: DepthExtensionNet, case: str) -> None:
    if case == C1_CASE:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.hidden_linears[2].parameters():
            parameter.requires_grad_(True)
        return
    if case == C2_CASE:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    raise ValueError(case)


def _trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _assert_c1_source_parameters_preserved(
    model: DepthExtensionNet,
    source: exp73.Exp73Net,
) -> None:
    pairs = (
        (model.hidden_linears[0].weight, source.hidden_linears[0].weight, "W1"),
        (model.hidden_linears[1].weight, source.hidden_linears[1].weight, "W2"),
        (model.output_linear.weight, source.output_linear.weight, "Wout"),
    )
    for candidate, reference, name in pairs:
        if not torch.equal(candidate.detach().cpu(), reference.detach().cpu()):
            raise RuntimeError(f"C1 changed frozen source parameter {name}")


def _evaluate_three_layer(
    model: DepthExtensionNet,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            loss, scores = exp73._objective_loss_scores(
                trajectory["evidence"],
                lengths,
                y,
                SOURCE_OBJECTIVE,
            )
            ys.append(y.cpu().numpy())
            predictions.append(scores.argmax(dim=1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)
    metrics = exp72._metrics(np.concatenate(ys), np.concatenate(predictions))
    metrics["objective_loss"] = loss_sum / max(n_total, 1)
    return metrics


def _checkpoint_improved(
    metrics: dict[str, float],
    best_ba: float,
    best_loss: float,
) -> bool:
    return metrics["balanced_accuracy"] > best_ba + 1e-12 or (
        abs(metrics["balanced_accuracy"] - best_ba) <= 1e-12
        and metrics["objective_loss"] < best_loss - 1e-12
    )


def _state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _load_c1_model(
    config: Config,
    data: exp3.Data,
    seed: int,
) -> tuple[DepthExtensionNet, dict[str, Any], Path]:
    path = _checkpoint_path(config.results_dir, C1_CASE, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing C1 checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong C1 experiment id: {path}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong C1 protocol: {path}")
    if payload.get("spec") != asdict(TrainSpec(C1_CASE, seed)):
        raise ValueError(f"C1 checkpoint identity mismatch: {path}")
    model = DepthExtensionNet(len(data.labels), data.fs).to(config.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload, path


def run_c1(
    spec: TrainSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_train_spec(spec)
    if spec.case != C1_CASE:
        raise ValueError(spec.case)

    checkpoint_path = _checkpoint_path(config.results_dir, spec.case, spec.seed)
    evaluation_path = _evaluation_path(config.results_dir, spec.case, spec.seed)
    if checkpoint_path.exists() and evaluation_path.exists() and not force:
        return _load_json(evaluation_path)

    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    source, source_payload, source_path = _load_source_model(config, data, spec.seed)

    model = DepthExtensionNet(len(data.labels), data.fs).to(device)
    _copy_a2_into_three_layer(model, source, spec.seed)
    _configure_trainable(model, C1_CASE)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=exp72.LR,
        weight_decay=exp72.WEIGHT_DECAY,
    )
    train_loader = exp73._raw_loaders(
        data, spec.seed, config.batch_size, True
    )["train"]
    eval_loaders = exp73._raw_loaders(
        data, spec.seed, config.batch_size, False
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_total = 0
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            loss, _ = exp73._objective_loss_scores(
                trajectory["evidence"],
                lengths,
                y,
                SOURCE_OBJECTIVE,
            )
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate_three_layer(model, eval_loaders["train"], device)
        val_metrics = _evaluate_three_layer(model, eval_loaders["val"], device)
        history.append(
            {
                "epoch": epoch,
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )
        if _checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = _state_dict_cpu(model)
        if epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE:
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No C1 checkpoint selected for seed {spec.seed}")

    model.load_state_dict(best_state, strict=True)
    _assert_c1_source_parameters_preserved(model, source)
    native_metrics = {
        split: _evaluate_three_layer(model, loader, device)
        for split, loader in eval_loaders.items()
    }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "architecture": ARCHITECTURE_3L,
            "shifts": SHIFTS,
            "source_a2_spec": asdict(_source_spec(spec.seed)),
            "source_a2_checkpoint": str(source_path.relative_to(config.repo_root)),
            "source_a2_best_epoch": int(source_payload["best_epoch"]),
            "l3_init_seed": _l3_init_seed(spec.seed),
            "trainable_scope": "W3_only",
            "trainable_parameter_count": _trainable_parameter_count(model),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    history_path = _history_path(config.results_dir, spec.case, spec.seed)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "case": spec.case,
        "spec": asdict(spec),
        "architecture": ARCHITECTURE_3L,
        "objective": SOURCE_OBJECTIVE,
        "training_scope": "W3_only",
        "source_a2_checkpoint": str(source_path.relative_to(config.repo_root)),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native_metrics,
        "checkpoint_selection": "validation native BA primary, WCCE loss tiebreak",
        "frozen_parameters_verified": True,
    }
    _save_json(evaluation_path, payload)
    return payload


def run_c2(
    spec: TrainSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_train_spec(spec)
    if spec.case != C2_CASE:
        raise ValueError(spec.case)

    checkpoint_path = _checkpoint_path(config.results_dir, spec.case, spec.seed)
    evaluation_path = _evaluation_path(config.results_dir, spec.case, spec.seed)
    if checkpoint_path.exists() and evaluation_path.exists() and not force:
        return _load_json(evaluation_path)

    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model, c1_payload, c1_path = _load_c1_model(config, data, spec.seed)
    _configure_trainable(model, C2_CASE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp72.LR,
        weight_decay=exp72.WEIGHT_DECAY,
    )
    train_loader = exp73._raw_loaders(
        data, spec.seed, config.batch_size, True
    )["train"]
    eval_loaders = exp73._raw_loaders(
        data, spec.seed, config.batch_size, False
    )

    epoch0_train = _evaluate_three_layer(model, eval_loaders["train"], device)
    epoch0_val = _evaluate_three_layer(model, eval_loaders["val"], device)
    best_state = _state_dict_cpu(model)
    best_epoch = 0
    best_ba = float(epoch0_val["balanced_accuracy"])
    best_loss = float(epoch0_val["objective_loss"])
    stopped_epoch = config.max_epochs
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train_ba": float(epoch0_train["balanced_accuracy"]),
            "val_ba": best_ba,
            "train_loss": float(epoch0_train["objective_loss"]),
            "val_loss": best_loss,
            "checkpoint_source": "C1_selected_checkpoint",
        }
    ]

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_total = 0
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            loss, _ = exp73._objective_loss_scores(
                trajectory["evidence"],
                lengths,
                y,
                SOURCE_OBJECTIVE,
            )
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate_three_layer(model, eval_loaders["train"], device)
        val_metrics = _evaluate_three_layer(model, eval_loaders["val"], device)
        history.append(
            {
                "epoch": epoch,
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
                "checkpoint_source": "e2e_finetune",
            }
        )
        if _checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = _state_dict_cpu(model)
        if epoch >= MIN_EPOCHS and epoch - best_epoch >= PATIENCE:
            stopped_epoch = epoch
            break

    model.load_state_dict(best_state, strict=True)
    native_metrics = {
        split: _evaluate_three_layer(model, loader, device)
        for split, loader in eval_loaders.items()
    }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "architecture": ARCHITECTURE_3L,
            "shifts": SHIFTS,
            "source_c1_checkpoint": str(c1_path.relative_to(config.repo_root)),
            "source_c1_best_epoch": int(c1_payload["best_epoch"]),
            "optimizer_state_reused": False,
            "epoch0_candidate": True,
            "trainable_scope": "W1_W2_W3_Wout",
            "trainable_parameter_count": _trainable_parameter_count(model),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    history_path = _history_path(config.results_dir, spec.case, spec.seed)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "case": spec.case,
        "spec": asdict(spec),
        "architecture": ARCHITECTURE_3L,
        "objective": SOURCE_OBJECTIVE,
        "training_scope": "full_e2e_from_C1",
        "source_c1_checkpoint": str(c1_path.relative_to(config.repo_root)),
        "epoch0_candidate": True,
        "optimizer_state_reused": False,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native_metrics,
        "checkpoint_selection": "C1 epoch-0 candidate plus validation native BA primary, WCCE loss tiebreak",
    }
    _save_json(evaluation_path, payload)
    return payload


def _load_case_model(
    config: Config,
    data: exp3.Data,
    spec: ProbeSpec,
) -> tuple[nn.Module, dict[str, Any]]:
    if spec.case == C0_CASE:
        model, checkpoint, path = _load_source_model(config, data, spec.seed)
        return model, {
            "checkpoint": str(path.relative_to(config.repo_root)),
            "best_epoch": int(checkpoint["best_epoch"]),
            "source": SOURCE_METHOD,
        }

    checkpoint_path = _checkpoint_path(config.results_dir, spec.case, spec.seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(
        checkpoint_path,
        map_location=config.device,
        weights_only=False,
    )
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Wrong checkpoint experiment: {checkpoint_path}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong checkpoint protocol: {checkpoint_path}")
    model = DepthExtensionNet(len(data.labels), data.fs).to(config.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, {
        "checkpoint": str(checkpoint_path.relative_to(config.repo_root)),
        "best_epoch": int(payload["best_epoch"]),
        "source": spec.case,
    }


def _extract_hidden_splits(
    model: nn.Module,
    loaders: dict[str, Any],
    device: torch.device,
    expected_layers: int,
) -> dict[str, tuple[tuple[torch.Tensor, ...], np.ndarray, np.ndarray]]:
    result: dict[
        str,
        tuple[tuple[torch.Tensor, ...], np.ndarray, np.ndarray],
    ] = {}
    model.eval()
    with torch.no_grad():
        for split, loader in loaders.items():
            layer_parts: list[list[torch.Tensor]] = [
                [] for _ in range(expected_layers)
            ]
            labels: list[np.ndarray] = []
            lengths_all: list[np.ndarray] = []
            for X, y, lengths in loader:
                trajectory = model.forward_trajectory(X.to(device))
                hidden = trajectory["hidden_spikes"]
                if not isinstance(hidden, tuple) or len(hidden) != expected_layers:
                    raise RuntimeError(
                        f"{split}: expected {expected_layers} hidden layers, "
                        f"got {type(hidden)!r}/{len(hidden) if isinstance(hidden, tuple) else 'n/a'}"
                    )
                for index, values in enumerate(hidden):
                    layer_parts[index].append(values.cpu())
                labels.append(y.numpy())
                lengths_all.append(lengths.numpy())
            result[split] = (
                tuple(torch.cat(parts, dim=0) for parts in layer_parts),
                np.concatenate(labels),
                np.concatenate(lengths_all),
            )
    return result


def _probe_features(
    spikes: torch.Tensor,
    lengths: np.ndarray,
    bin_steps: int,
    representation: str,
) -> np.ndarray:
    lengths_t = torch.as_tensor(lengths, dtype=torch.long)
    if representation == "wholecount":
        valid = exp50.valid_mask(lengths_t, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
        return (spikes * valid).sum(dim=1).numpy()
    if representation == "fixed250":
        return exp3.fixed_counts(spikes, lengths_t, bin_steps).flatten(1).numpy()
    raise ValueError(representation)


def _fit_no_bias_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    scaler = StandardScaler(with_mean=False).fit(train_x)
    train_z = scaler.transform(train_x)
    val_z = scaler.transform(val_x)
    test_z = scaler.transform(test_x)

    best: tuple[float, float, LogisticRegression] | None = None
    for C in PROBE_C_GRID:
        classifier = LogisticRegression(
            C=C,
            fit_intercept=False,
            max_iter=PROBE_MAX_ITER,
            solver="lbfgs",
            random_state=seed,
        ).fit(train_z, train_y)
        val_ba = float(
            balanced_accuracy_score(val_y, classifier.predict(val_z))
        )
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError("No no-bias probe candidate selected")

    _, selected_C, classifier = best
    return {
        "feature_dim": int(train_x.shape[1]),
        "probe_C": selected_C,
        "fit_intercept": False,
        "scaler_with_mean": False,
        "normalization": "train_only_scale_no_centering",
        "train": exp72._metrics(train_y, classifier.predict(train_z)),
        "val": exp72._metrics(val_y, classifier.predict(val_z)),
        "test": exp72._metrics(test_y, classifier.predict(test_z)),
    }


def _fit_affine_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    probe = exp01._fit_linear_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        seed,
    )
    return {
        **probe,
        "fit_intercept": True,
        "scaler_with_mean": True,
        "normalization": "Exp7.3 train-only StandardScaler mean+scale",
    }


def run_probe(
    spec: ProbeSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_probe_spec(spec)
    destination = _probe_path(config.results_dir, spec)
    if destination.exists() and not force:
        return _load_json(destination)

    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model, checkpoint_meta = _load_case_model(config, data, spec)
    loaders = exp73._raw_loaders(
        data, spec.seed, config.batch_size, False
    )
    layers = case_layers(spec.case)
    extracted = _extract_hidden_splits(
        model,
        loaders,
        device,
        expected_layers=len(layers),
    )

    rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        for representation in PROBE_REPRESENTATIONS:
            features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for split, (hidden, y, lengths) in extracted.items():
                features[split] = (
                    _probe_features(
                        hidden[layer_index],
                        lengths,
                        data.bin_steps,
                        representation,
                    ),
                    y,
                )
            for bias_mode in PROBE_BIAS_MODES:
                seed = exp3.dseed(
                    spec.seed,
                    EXPERIMENT_ID,
                    spec.case,
                    layer,
                    representation,
                    bias_mode,
                )
                fit = _fit_affine_probe if bias_mode == "affine" else _fit_no_bias_probe
                probe = fit(
                    features["train"][0],
                    features["train"][1],
                    features["val"][0],
                    features["val"][1],
                    features["test"][0],
                    features["test"][1],
                    seed,
                )
                row: dict[str, Any] = {
                    "case": spec.case,
                    "seed": spec.seed,
                    "layer": layer,
                    "representation": representation,
                    "bias_mode": bias_mode,
                    "feature_dim": int(probe["feature_dim"]),
                    "selected_C": float(probe["probe_C"]),
                    "fit_intercept": bool(probe["fit_intercept"]),
                    "scaler_with_mean": bool(probe["scaler_with_mean"]),
                    "normalization": str(probe["normalization"]),
                }
                for split in ("train", "val", "test"):
                    metrics = probe[split]
                    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                        row[f"{split}_{metric}"] = float(metrics[metric])
                rows.append(row)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "architecture": (
            ARCHITECTURE_2L if spec.case == C0_CASE else ARCHITECTURE_3L
        ),
        "checkpoint": checkpoint_meta,
        "layers": list(layers),
        "probe_contract": {
            "representations": list(PROBE_REPRESENTATIONS),
            "bias_modes": list(PROBE_BIAS_MODES),
            "affine": "Exp7.3 StandardScaler(mean+scale) + LogisticRegression(intercept)",
            "no_bias": "StandardScaler(with_mean=False) + LogisticRegression(fit_intercept=False)",
            "C_grid": list(PROBE_C_GRID),
            "C_selection": "validation balanced accuracy only",
            "test_not_used_for_selection": True,
        },
        "rows": rows,
    }
    _save_json(destination, payload)
    return payload


def _native_rows(config: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        source_eval = _load_json(_source_evaluation_path(config.repo_root, seed))
        source_ckpt = torch.load(
            _source_checkpoint_path(config.repo_root, seed),
            map_location="cpu",
            weights_only=False,
        )
        rows.append(
            {
                "case": C0_CASE,
                "seed": seed,
                "architecture": ARCHITECTURE_2L,
                "training_scope": "reused_exp7.3_A2",
                "best_epoch": int(source_ckpt["best_epoch"]),
                "train_ba": float(
                    source_eval["native_metrics"]["train"]["balanced_accuracy"]
                ),
                "val_ba": float(
                    source_eval["native_metrics"]["val"]["balanced_accuracy"]
                ),
                "test_ba": float(
                    source_eval["native_metrics"]["test"]["balanced_accuracy"]
                ),
                "test_accuracy": float(
                    source_eval["native_metrics"]["test"]["accuracy"]
                ),
                "test_macro_f1": float(
                    source_eval["native_metrics"]["test"]["macro_f1"]
                ),
            }
        )

    for case in TRAIN_CASES:
        for seed in SEEDS:
            payload = _load_json(
                _evaluation_path(config.results_dir, case, seed)
            )
            native = payload["native_metrics"]
            rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "architecture": ARCHITECTURE_3L,
                    "training_scope": payload["training_scope"],
                    "best_epoch": int(payload["best_epoch"]),
                    "train_ba": float(native["train"]["balanced_accuracy"]),
                    "val_ba": float(native["val"]["balanced_accuracy"]),
                    "test_ba": float(native["test"]["balanced_accuracy"]),
                    "test_accuracy": float(native["test"]["accuracy"]),
                    "test_macro_f1": float(native["test"]["macro_f1"]),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != len(CASES) * len(SEEDS):
        raise RuntimeError(f"Expected 9 native rows, got {len(frame)}")
    return frame


def _probe_rows(config: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in probe_specs():
        payload = _load_json(_probe_path(config.results_dir, spec))
        payload_rows = payload.get("rows")
        if not isinstance(payload_rows, list):
            raise ValueError(f"Malformed probe payload: {spec.key}")
        rows.extend(payload_rows)
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_PROBE_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_PROBE_ROWS} probe rows, got {len(frame)}"
        )
    return frame


def _depth_contrasts(native: pd.DataFrame) -> pd.DataFrame:
    pivot = native.pivot(index="seed", columns="case", values="test_ba")
    rows: list[dict[str, Any]] = []
    comparisons = (
        ("C1_minus_C0_insert_L3", C1_CASE, C0_CASE),
        ("C2_minus_C1_e2e_adaptation", C2_CASE, C1_CASE),
        ("C2_minus_C0_total_depth_gain", C2_CASE, C0_CASE),
    )
    for name, left, right in comparisons:
        for seed in SEEDS:
            rows.append(
                {
                    "contrast": name,
                    "seed": seed,
                    "left_case": left,
                    "right_case": right,
                    "test_ba_delta_pp": 100.0
                    * (float(pivot.loc[seed, left]) - float(pivot.loc[seed, right])),
                }
            )
    return pd.DataFrame(rows)


def _bias_contrasts(probes: pd.DataFrame) -> pd.DataFrame:
    pivot = probes.pivot_table(
        index=["case", "seed", "layer", "representation"],
        columns="bias_mode",
        values="test_balanced_accuracy",
        aggfunc="first",
    ).reset_index()
    pivot["affine_minus_no_bias_pp"] = 100.0 * (
        pivot["affine"] - pivot["no_bias"]
    )
    return pivot


def _temporal_contrasts(probes: pd.DataFrame) -> pd.DataFrame:
    pivot = probes.pivot_table(
        index=["case", "seed", "layer", "bias_mode"],
        columns="representation",
        values="test_balanced_accuracy",
        aggfunc="first",
    ).reset_index()
    pivot["fixed250_minus_wholecount_pp"] = 100.0 * (
        pivot["fixed250"] - pivot["wholecount"]
    )
    return pivot


def _layer_contrasts(probes: pd.DataFrame) -> pd.DataFrame:
    subset = probes[probes["case"].isin((C1_CASE, C2_CASE))]
    pivot = subset.pivot_table(
        index=["case", "seed", "representation", "bias_mode"],
        columns="layer",
        values="test_balanced_accuracy",
        aggfunc="first",
    ).reset_index()
    pivot["L3_minus_L2_pp"] = 100.0 * (pivot["L3"] - pivot["L2"])
    return pivot


def finalize(config: Config) -> dict[str, Any]:
    _ = _load_json(config.results_dir / "source_manifest.json")
    native = _native_rows(config)
    probes = _probe_rows(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    native.to_csv(config.results_dir / "native_runs.csv", index=False)
    native_summary = (
        native.groupby(["case", "architecture", "training_scope"], sort=False)[
            ["test_ba", "test_accuracy", "test_macro_f1", "best_epoch"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    native_summary.columns = [
        "_".join(str(v) for v in column if str(v))
        if isinstance(column, tuple)
        else str(column)
        for column in native_summary.columns
    ]
    native_summary.to_csv(
        config.results_dir / "native_summary.csv",
        index=False,
    )

    probes.to_csv(config.results_dir / "probe_runs.csv", index=False)
    probe_summary = (
        probes.groupby(
            ["case", "layer", "representation", "bias_mode"],
            sort=False,
        )[
            [
                "test_balanced_accuracy",
                "test_accuracy",
                "test_macro_f1",
                "selected_C",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    probe_summary.columns = [
        "_".join(str(v) for v in column if str(v))
        if isinstance(column, tuple)
        else str(column)
        for column in probe_summary.columns
    ]
    probe_summary.to_csv(
        config.results_dir / "probe_summary.csv",
        index=False,
    )

    depth = _depth_contrasts(native)
    bias = _bias_contrasts(probes)
    temporal = _temporal_contrasts(probes)
    layer = _layer_contrasts(probes)
    depth.to_csv(config.results_dir / "depth_contrasts.csv", index=False)
    bias.to_csv(config.results_dir / "bias_contrasts.csv", index=False)
    temporal.to_csv(config.results_dir / "temporal_contrasts.csv", index=False)
    layer.to_csv(config.results_dir / "layer_contrasts.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Can a third 128-neuron (2,3,4) SNN layer improve an already trained "
            "Exp7.3 A2 hierarchy, first with the A2 backbone/output frozen and then "
            "with full end-to-end WCCE adaptation?"
        ),
        "source": {
            "experiment_id": SOURCE_EXPERIMENT_ID,
            "protocol_version": SOURCE_PROTOCOL_VERSION,
            "method": SOURCE_METHOD,
            "architecture": SOURCE_ARCHITECTURE,
            "objective": SOURCE_OBJECTIVE,
        },
        "cases": {
            C0_CASE: "reuse Exp7.3 A2 checkpoint; no retraining",
            C1_CASE: "A2 W1/W2/Wout frozen; insert 128-neuron (2,3,4) L3 and train W3 only",
            C2_CASE: "load selected C1 checkpoint; fresh Adam; epoch 0 is C1 candidate; unfreeze W1/W2/W3/Wout",
        },
        "seeds": list(SEEDS),
        "architecture_3layer": {
            "widths": [HIDDEN_WIDTH, HIDDEN_WIDTH, HIDDEN_WIDTH],
            "shifts": [list(values) for values in SHIFTS],
            "output": "bias-free Linear 128->12",
        },
        "training": {
            "objective": "Exp7.3 A2 WCCE using valid-time mean Linear evidence",
            "max_epochs": MAX_EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "patience": PATIENCE,
            "checkpoint_rule": "validation native BA primary, WCCE loss tiebreak",
            "c2_optimizer_state_reused": False,
            "c2_epoch0_candidate": True,
        },
        "probes": {
            "representations": list(PROBE_REPRESENTATIONS),
            "bias_modes": list(PROBE_BIAS_MODES),
            "affine": "Exp7.3 StandardScaler(mean+scale) + intercept",
            "no_bias": "train-only scale without centering + fit_intercept=False",
            "C_grid": list(PROBE_C_GRID),
        },
        "counts": {
            "new_training_runs": EXPECTED_C1_RUNS + EXPECTED_C2_RUNS,
            "probe_tasks": EXPECTED_PROBE_TASKS,
            "probe_rows": EXPECTED_PROBE_ROWS,
            "native_rows": len(native),
        },
        "files": {
            "native_runs": "native_runs.csv",
            "native_summary": "native_summary.csv",
            "probe_runs": "probe_runs.csv",
            "probe_summary": "probe_summary.csv",
            "depth_contrasts": "depth_contrasts.csv",
            "bias_contrasts": "bias_contrasts.csv",
            "temporal_contrasts": "temporal_contrasts.csv",
            "layer_contrasts": "layer_contrasts.csv",
        },
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp7.3.9 pretrained A2 depth extension"
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--force", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-source")
    for name in ("c1", "c2", "probe"):
        child = subparsers.add_parser(name)
        child.add_argument("--array-task-id", type=int, required=True)
    subparsers.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = (
        args.repo_root.resolve()
        if args.repo_root is not None
        else find_repo_root()
    )
    config = Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        max_epochs=args.max_epochs,
    )

    if args.command == "prepare-source":
        prepare_source(config)
        return
    if args.command == "c1":
        specs = c1_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        run_c1(specs[args.array_task_id], config, force=args.force)
        return
    if args.command == "c2":
        specs = c2_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        run_c2(specs[args.array_task_id], config, force=args.force)
        return
    if args.command == "probe":
        specs = probe_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        run_probe(specs[args.array_task_id], config, force=args.force)
        return
    if args.command == "finalize":
        finalize(config)
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
