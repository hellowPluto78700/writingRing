from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from snntorch import surrogate

from scripts import experiment_5_3_2_1_when_objective_sweep as objective_parent


parent = objective_parent.parent
base = parent.base
exp52 = parent.exp52

EXPERIMENT_ID = "experiment_5_3_2_2_when_width_sweep"
PROTOCOL_VERSION = "rsnn_when_width_v1"
SEEDS = objective_parent.SEEDS
WIDTHS = (16, 32, 64, 128, 256)
WIDTH_CONDITIONS = tuple(f"rsnn_h{width}" for width in WIDTHS)
OBJECTIVE = objective_parent.OBJECTIVE_BY_NAME["joint"]
WHEN_CONDITION = objective_parent.WHEN_CONDITION
LOCAL_WIDTH = parent.LOCAL_WIDTH
SHORT_SHIFT = parent.SHORT_SHIFT
N_PHASES = parent.N_PHASES
THRESHOLD = parent.THRESHOLD
RESET = parent.RESET
SURROGATE_SLOPE = parent.SURROGATE_SLOPE
EPOCHS = objective_parent.parent.EPOCHS
BATCH_SIZE = objective_parent.parent.BATCH_SIZE
LR = objective_parent.parent.LR
WEIGHT_DECAY = objective_parent.parent.WEIGHT_DECAY
SHUFFLE_REPLICATES = objective_parent.parent.SHUFFLE_REPLICATES
SPIKE_WINDOWS_SECONDS = objective_parent.parent.SPIKE_WINDOWS_SECONDS
PHASE_C_GRID = objective_parent.parent.PHASE_C_GRID
PROGRESS_ALPHA_GRID = objective_parent.parent.PROGRESS_ALPHA_GRID
DEAD_FR_THRESHOLD = 1e-4
HIGHLY_ACTIVE_FR_THRESHOLD = 0.10
EPS = 1e-12
COMMON_PROBE_NAMESPACE = "exp5_3_2_1_common_probe"
RESET_PROBE_NAMESPACE = "exp5_3_2_1_state_reset"
SHUFFLE_PROBE_NAMESPACE = "exp5_3_2_1_temporal_shuffle"
EVALUATION_REQUIRED_SECTIONS = (
    "parameter_count",
    "parameter_counts",
    "native",
    "probes",
    "trajectory_metrics",
    "history_attribution",
    "activity",
    "representation",
)


@dataclass(frozen=True)
class RunSpec:
    hidden_width: int
    seed: int

    @property
    def condition(self) -> str:
        return f"rsnn_h{self.hidden_width}"

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
    return parent.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / EXPERIMENT_ID
        / PROTOCOL_VERSION
    )


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def baseline_path(root: Path, seed: int) -> Path:
    return root / "baselines" / f"seed{seed}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_specs() -> list[RunSpec]:
    return [RunSpec(width, seed) for width in WIDTHS for seed in SEEDS]


def _validate_spec(spec: RunSpec) -> None:
    if spec.hidden_width not in WIDTHS:
        raise ValueError(f"Unknown Exp5.3.2.2 width: {spec.hidden_width}")
    if spec.seed not in SEEDS:
        raise ValueError(f"Unknown Exp5.3.2.2 seed: {spec.seed}")


def _parent_config(config: Config) -> parent.Config:
    return parent.Config(
        repo_root=config.repo_root,
        results_dir=parent.results_dir(config.repo_root),
        device=config.device,
        epochs=config.epochs,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _objective_config(config: Config) -> objective_parent.Config:
    return objective_parent.Config(
        repo_root=config.repo_root,
        results_dir=objective_parent.results_dir(config.repo_root),
        device=config.device,
        epochs=config.epochs,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _parent_spec(seed: int) -> parent.RunSpec:
    return parent.RunSpec(WHEN_CONDITION, seed)


def _loader_seed(seed: int, split: str) -> int:
    return base.dseed(seed, "exp5_3_2", split, "loader")


class WidthWhenBranchNet(nn.Module):
    """Width-parametric copy of the Exp5.3.2.1 short-memory one-layer RSNN."""

    def __init__(self, hidden_width: int, fs: float) -> None:
        super().__init__()
        if hidden_width not in WIDTHS:
            raise ValueError(f"Unsupported hidden width: {hidden_width}")
        self.hidden_width = int(hidden_width)
        self.fs = float(fs)
        self.spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

        self.input_projection = nn.Linear(
            LOCAL_WIDTH,
            self.hidden_width,
            bias=False,
        )
        self.recurrent = nn.Linear(
            self.hidden_width,
            self.hidden_width,
            bias=False,
        )
        self.phase_head = nn.Linear(self.hidden_width, N_PHASES, bias=True)
        self.progress_head = nn.Linear(self.hidden_width, 1, bias=True)

        decay = parent.decay_from_shift(SHORT_SHIFT)
        self.register_buffer(
            "alpha_vector",
            torch.full((self.hidden_width,), decay, dtype=torch.float32),
        )
        self.register_buffer(
            "beta_vector",
            torch.full((self.hidden_width,), decay, dtype=torch.float32),
        )

    @staticmethod
    def _reset_membrane(
        membrane: torch.Tensor,
        spike: torch.Tensor,
    ) -> torch.Tensor:
        if RESET == "subtract":
            return membrane - spike * THRESHOLD
        if RESET == "zero":
            return membrane * (1.0 - spike)
        if RESET == "none":
            return membrane
        raise ValueError(f"Unsupported reset mechanism: {RESET}")

    def _initial_state(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (x.shape[0], self.hidden_width)
        synaptic = torch.zeros(shape, dtype=x.dtype, device=x.device)
        membrane = torch.zeros_like(synaptic)
        spike = torch.zeros_like(synaptic)
        return synaptic, membrane, spike

    def _step(
        self,
        local_t: torch.Tensor,
        synaptic: torch.Tensor,
        membrane: torch.Tensor,
        previous_spike: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = self.input_projection(local_t) + self.recurrent(previous_spike)
        synaptic = self.alpha_vector.to(current) * synaptic + current
        membrane = self.beta_vector.to(current) * membrane + synaptic
        spike = self.spike_grad(membrane - THRESHOLD)
        membrane = self._reset_membrane(membrane, spike)
        return synaptic, membrane, spike

    def forward_trajectory(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        reset_state_each_step: bool = False,
    ) -> parent.WhenTrajectory:
        if torch.any(lengths <= 0) or torch.any(lengths > x.shape[1]):
            raise ValueError("Invalid sequence lengths for WHEN trajectory")
        synaptic, membrane, previous_spike = self._initial_state(x)
        synaptic_parts: list[torch.Tensor] = []
        membrane_parts: list[torch.Tensor] = []
        spike_parts: list[torch.Tensor] = []
        for timestep in range(x.shape[1]):
            if reset_state_each_step:
                synaptic = torch.zeros_like(synaptic)
                membrane = torch.zeros_like(membrane)
                previous_spike = torch.zeros_like(previous_spike)
            next_synaptic, next_membrane, next_spike = self._step(
                x[:, timestep],
                synaptic,
                membrane,
                previous_spike,
            )
            valid = (timestep < lengths).unsqueeze(1)
            synaptic = torch.where(valid, next_synaptic, synaptic)
            membrane = torch.where(valid, next_membrane, membrane)
            spike = next_spike * valid.to(next_spike.dtype)
            synaptic_parts.append(synaptic)
            membrane_parts.append(membrane)
            spike_parts.append(spike)
            previous_spike = spike
        return parent.WhenTrajectory(
            synaptic=torch.stack(synaptic_parts, dim=1),
            membranes=torch.stack(membrane_parts, dim=1),
            spikes=torch.stack(spike_parts, dim=1),
        )

    def heads_from_membrane(
        self,
        membranes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase_logits = self.phase_head(membranes)
        progress = torch.sigmoid(self.progress_head(membranes).squeeze(-1))
        return phase_logits, progress

    def loss_components(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        reset_state_each_step: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        parent.WhenTrajectory,
    ]:
        trajectory = self.forward_trajectory(
            x,
            lengths,
            reset_state_each_step=reset_state_each_step,
        )
        phase_logits, progress_pred = self.heads_from_membrane(
            trajectory.membranes
        )
        phase_target, progress_target, valid = parent._phase_progress_targets(
            lengths,
            x.shape[1],
            x.dtype,
        )
        batch_size, steps = phase_target.shape
        phase_step = F.cross_entropy(
            phase_logits.reshape(batch_size * steps, N_PHASES),
            phase_target.reshape(batch_size * steps),
            reduction="none",
        ).reshape(batch_size, steps)
        progress_step = F.smooth_l1_loss(
            progress_pred,
            progress_target,
            reduction="none",
        )
        valid_float = valid.to(x.dtype)
        denom = lengths.to(x.dtype)
        phase_loss = ((phase_step * valid_float).sum(dim=1) / denom).mean()
        progress_loss = ((progress_step * valid_float).sum(dim=1) / denom).mean()
        total_loss = phase_loss + progress_loss
        return (
            total_loss,
            phase_loss,
            progress_loss,
            phase_logits,
            progress_pred,
            trajectory,
        )

    def group_indices(self, state_kind: str) -> dict[str, list[int]]:
        if state_kind not in ("membrane", "synaptic"):
            raise ValueError(f"Unknown state kind: {state_kind}")
        return {}


def _initialize_model(
    hidden_width: int,
    seed: int,
    fs: float,
    device: torch.device,
) -> WidthWhenBranchNet:
    width_key = f"h{hidden_width}"
    base.seed_all(base.dseed(seed, EXPERIMENT_ID, width_key, "constructor"))
    model = WidthWhenBranchNet(hidden_width, fs).to(device)

    for component, module in (
        ("input_projection", model.input_projection),
        ("recurrent", model.recurrent),
        ("phase_head", model.phase_head),
        ("progress_head", model.progress_head),
    ):
        base.seed_all(base.dseed(seed, EXPERIMENT_ID, width_key, component))
        module.reset_parameters()
    return model


def parameter_counts(model: WidthWhenBranchNet) -> dict[str, int]:
    return {
        "input_projection": int(
            sum(parameter.numel() for parameter in model.input_projection.parameters())
        ),
        "recurrent": int(
            sum(parameter.numel() for parameter in model.recurrent.parameters())
        ),
        "phase_head": int(
            sum(parameter.numel() for parameter in model.phase_head.parameters())
        ),
        "progress_head": int(
            sum(parameter.numel() for parameter in model.progress_head.parameters())
        ),
        "trainable_total": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
    }


def _provenance(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    model: WidthWhenBranchNet,
) -> dict[str, object]:
    counts = parameter_counts(model)
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": spec.seed,
        "split_seed": base.SPLIT_SEED,
        "condition": spec.condition,
        "hidden_width": spec.hidden_width,
        "local_width": LOCAL_WIDTH,
        "recurrent": True,
        "shifts_mem": [SHORT_SHIFT],
        "shifts_syn": [SHORT_SHIFT],
        "tau_mem_ms": [parent.tau_ms_from_shift(SHORT_SHIFT, data.fs)],
        "tau_syn_ms": [parent.tau_ms_from_shift(SHORT_SHIFT, data.fs)],
        "threshold": THRESHOLD,
        "reset": RESET,
        "surrogate_slope": SURROGATE_SLOPE,
        "sampling_rate_hz": float(data.fs),
        "source_objective_experiment": objective_parent.EXPERIMENT_ID,
        "source_objective_protocol": objective_parent.PROTOCOL_VERSION,
        "source_when_experiment": parent.EXPERIMENT_ID,
        "source_when_protocol": parent.PROTOCOL_VERSION,
        "source_local_experiment": exp52.EXPERIMENT_ID,
        "source_local_protocol": exp52.PROTOCOL_VERSION,
        "local_source_frozen": True,
        "input_scaling": (
            "reuse the identical frozen Exp5.2 WHAT cache for every width; "
            "no width-dependent rescaling"
        ),
        "objective": OBJECTIVE.name,
        "objective_formula": "L_phase + L_progress",
        "phase_target": (
            "q_t=min(9,floor(10*t/(T-1))); T used only to construct supervision"
        ),
        "progress_target": "p_t=t/(T-1); T used only to construct supervision",
        "causality_contract": (
            "RSNN forward receives z_1:t only; final valid length T is never an "
            "input to the RSNN or either prediction head"
        ),
        "loss_reduction": (
            "mean over valid timesteps within each sample, then mean over samples"
        ),
        "checkpoint_selection": (
            "min validation joint objective loss; tie lower validation progress "
            "sample-balanced MAE; tie higher validation phase BA"
        ),
        "initialization": (
            "deterministic dseed(seed, experiment, width, component); no neuron "
            "prefix copying across widths"
        ),
        "minibatch_order": (
            "paired across widths within seed via the unchanged Exp5.3.2 loader seed"
        ),
        "train_loader_seed": _loader_seed(spec.seed, "train"),
        "val_loader_seed": _loader_seed(spec.seed, "val"),
        "test_loader_seed": _loader_seed(spec.seed, "test"),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "shuffle_replicates": SHUFFLE_REPLICATES,
        "spike_windows_seconds": list(SPIKE_WINDOWS_SECONDS),
        "dead_fr_threshold": DEAD_FR_THRESHOLD,
        "highly_active_fr_threshold": HIGHLY_ACTIVE_FR_THRESHOLD,
        "parameter_counts": counts,
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
    }


def prepare_local_seed(
    seed: int,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Unknown Exp5.3.2.2 seed: {seed}")

    source_config = _objective_config(config)
    objective_parent.prepare_local_seed(
        seed,
        data,
        source_config,
        force=False,
    )
    cache = parent.load_local_cache(seed, data, _parent_config(config))
    for split, (X, lengths) in parent._partitions(data, cache).items():
        if X.shape[0] != len(lengths):
            raise ValueError(
                f"Frozen WHAT cache sample mismatch for seed {seed} split {split}"
            )
        if X.shape[-1] != LOCAL_WIDTH:
            raise ValueError(
                f"Frozen WHAT width mismatch for seed {seed} split {split}: "
                f"{X.shape[-1]} != {LOCAL_WIDTH}"
            )

    source_baseline = objective_parent.baseline_path(
        objective_parent.results_dir(config.repo_root),
        seed,
    )
    if not source_baseline.exists():
        raise FileNotFoundError(
            f"Missing Exp5.3.2.1 baseline artifact after preparation: {source_baseline}"
        )
    baseline_payload = json.loads(source_baseline.read_text(encoding="utf-8"))
    if (
        baseline_payload.get("experiment_id") != objective_parent.EXPERIMENT_ID
        or baseline_payload.get("protocol_version")
        != objective_parent.PROTOCOL_VERSION
        or baseline_payload.get("seed") != seed
    ):
        raise ValueError(f"Exp5.3.2.1 baseline identity mismatch: {source_baseline}")

    destination = baseline_path(config.results_dir, seed)
    if force or not destination.exists():
        _save_json(
            destination,
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "source_experiment_id": objective_parent.EXPERIMENT_ID,
                "source_protocol_version": objective_parent.PROTOCOL_VERSION,
                "source_path": str(source_baseline.relative_to(config.repo_root)),
                "reuse_policy": (
                    "identity-validated reuse only; Exp5.3.2.2 does not refit "
                    "WHAT-only or elapsed-time baselines"
                ),
                "baselines": baseline_payload["baselines"],
            },
        )
    return exp52.local_cache_path(parent.source_results_dir(config.repo_root), seed)


def _evaluate_native(
    model: WidthWhenBranchNet,
    loader: DataLoader,
    device: torch.device,
    reset_state_each_step: bool = False,
) -> dict[str, float]:
    return objective_parent._evaluate_native_objective(
        model,
        loader,
        device,
        OBJECTIVE,
        reset_state_each_step=reset_state_each_step,
    )


def train_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    _validate_spec(spec)
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination

    pconfig = _parent_config(config)
    pspec = _parent_spec(spec.seed)
    cache = parent.load_local_cache(spec.seed, data, pconfig)
    train_loaders = parent._make_loaders(
        data,
        cache,
        pspec,
        pconfig,
        train_shuffle=True,
    )
    eval_loaders = parent._make_loaders(
        data,
        cache,
        pspec,
        pconfig,
        train_shuffle=False,
    )

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _initialize_model(spec.hidden_width, spec.seed, data.fs, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_objective = np.inf
    best_val_progress_mae = np.inf
    best_val_phase_ba = -np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_objective_sum = 0.0
        train_phase_sum = 0.0
        train_progress_sum = 0.0
        train_progress_abs_sum = 0.0
        train_phase_true: list[np.ndarray] = []
        train_phase_pred: list[np.ndarray] = []
        train_n = 0

        for local, lengths in train_loaders["train"]:
            local = local.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, phase_loss, progress_loss, phase_logits, progress_pred, _ = (
                model.loss_components(local, lengths)
            )
            objective_loss = objective_parent._objective_loss(
                OBJECTIVE,
                phase_loss,
                progress_loss,
            )
            objective_loss.backward()
            optimizer.step()

            phase_true, phase_pred, _, _, sample_abs_sum = (
                parent._native_metrics_from_batch(
                    phase_logits.detach(),
                    progress_pred.detach(),
                    lengths,
                )
            )
            n = len(local)
            train_n += n
            train_objective_sum += float(objective_loss.item()) * n
            train_phase_sum += float(phase_loss.item()) * n
            train_progress_sum += float(progress_loss.item()) * n
            train_progress_abs_sum += sample_abs_sum
            train_phase_true.append(phase_true)
            train_phase_pred.append(phase_pred)

        train_phase_ba = float(
            balanced_accuracy_score(
                np.concatenate(train_phase_true),
                np.concatenate(train_phase_pred),
            )
        )
        train_progress_mae = train_progress_abs_sum / max(train_n, 1)
        val_metrics = _evaluate_native(model, eval_loaders["val"], device)
        val_objective = float(val_metrics["objective_loss"])
        val_progress_mae = float(
            val_metrics["progress_sample_balanced_mae"]
        )
        val_phase_ba = float(val_metrics["phase_balanced_accuracy"])

        history.append(
            {
                "epoch": epoch,
                "train_objective_loss": train_objective_sum / max(train_n, 1),
                "train_phase_ce": train_phase_sum / max(train_n, 1),
                "train_progress_loss": train_progress_sum / max(train_n, 1),
                "train_phase_balanced_accuracy": train_phase_ba,
                "train_progress_sample_balanced_mae": train_progress_mae,
                "val_objective_loss": val_objective,
                "val_phase_ce": val_metrics["phase_ce"],
                "val_progress_loss": val_metrics["progress_loss"],
                "val_phase_balanced_accuracy": val_phase_ba,
                "val_progress_sample_balanced_mae": val_progress_mae,
            }
        )

        improved = (
            val_objective < best_val_objective - 1e-12
            or (
                abs(val_objective - best_val_objective) <= 1e-12
                and val_progress_mae < best_val_progress_mae - 1e-12
            )
            or (
                abs(val_objective - best_val_objective) <= 1e-12
                and abs(val_progress_mae - best_val_progress_mae) <= 1e-12
                and val_phase_ba > best_val_phase_ba + 1e-12
            )
        )
        if improved:
            best_val_objective = val_objective
            best_val_progress_mae = val_progress_mae
            best_val_phase_ba = val_phase_ba
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")

    model.load_state_dict(best_state)
    native = {
        split: _evaluate_native(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "provenance": _provenance(spec, data, config, model),
            "result": {
                "best_epoch": best_epoch,
                "best_val_objective_loss": best_val_objective,
                "best_val_progress_mae": best_val_progress_mae,
                "best_val_phase_ba": best_val_phase_ba,
                "native": native,
            },
            "state_dict": best_state,
        },
        destination,
    )
    hpath = history_path(config.results_dir, spec)
    hpath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hpath, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[WidthWhenBranchNet, dict[str, object]]:
    _validate_spec(spec)
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.3.2.2 checkpoint: {path}")
    payload = torch.load(
        path,
        map_location=config.device,
        weights_only=False,
    )
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise ValueError(f"Wrong Exp5.3.2.2 checkpoint identity: {path}")
    if payload.get("spec") != asdict(spec):
        raise ValueError(f"Checkpoint spec mismatch: {path}")
    model = WidthWhenBranchNet(spec.hidden_width, data.fs).to(
        torch.device(config.device)
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _lengths_by_split(data: base.Data) -> dict[str, np.ndarray]:
    return {
        "train": data.ltr,
        "val": data.lva,
        "test": data.lte,
    }


def _sequences_from_flat(
    flat: np.ndarray,
    lengths: np.ndarray,
) -> list[np.ndarray]:
    sequences: list[np.ndarray] = []
    offset = 0
    for raw_length in lengths:
        length = int(raw_length)
        if length <= 0:
            raise ValueError("Probe sequence length must be positive")
        stop = offset + length
        sequences.append(np.asarray(flat[offset:stop], dtype=np.float32))
        offset = stop
    if offset != len(flat):
        raise ValueError(
            f"Flattened probe rows mismatch valid lengths: {offset} != {len(flat)}"
        )
    return sequences


def _fit_phase_probe_model(
    features: dict[str, dict[str, np.ndarray]],
    feature_name: str,
    seed: int,
    namespace: str,
) -> tuple[StandardScaler, LogisticRegression, dict[str, float]]:
    train_x = features["train"][feature_name]
    val_x = features["val"][feature_name]
    test_x = features["test"][feature_name]
    train_y = features["train"]["phase"]
    val_y = features["val"]["phase"]
    test_y = features["test"]["phase"]

    scaler = StandardScaler().fit(train_x)
    train_z = scaler.transform(train_x)
    val_z = scaler.transform(val_x)
    test_z = scaler.transform(test_x)
    random_state = base.dseed(
        seed,
        "exp5_3_2",
        namespace,
        feature_name,
        "phase_probe",
    )

    best: tuple[float, float, LogisticRegression] | None = None
    for C in PHASE_C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=3000,
            solver="lbfgs",
            random_state=random_state,
        ).fit(train_z, train_y)
        val_prediction = classifier.predict(val_z)
        val_ba = float(balanced_accuracy_score(val_y, val_prediction))
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError("No phase probe candidate selected")

    val_ba, C, classifier = best
    metrics: dict[str, float] = {"phase_probe_C": C}
    for split, z, y in (
        ("train", train_z, train_y),
        ("val", val_z, val_y),
        ("test", test_z, test_y),
    ):
        prediction = classifier.predict(z)
        metrics[f"phase_probe_{split}_ba"] = float(
            balanced_accuracy_score(y, prediction)
        )
        metrics[f"phase_probe_{split}_accuracy"] = float(
            accuracy_score(y, prediction)
        )
        metrics[f"phase_probe_{split}_macro_f1"] = float(
            f1_score(y, prediction, average="macro", zero_division=0)
        )
    metrics["phase_probe_val_ba"] = val_ba
    return scaler, classifier, metrics


def _fit_progress_probe_model(
    features: dict[str, dict[str, np.ndarray]],
    feature_name: str,
) -> tuple[StandardScaler, Ridge, dict[str, float]]:
    train_x = features["train"][feature_name]
    val_x = features["val"][feature_name]
    test_x = features["test"][feature_name]
    train_y = features["train"]["progress"]
    val_y = features["val"]["progress"]
    test_y = features["test"]["progress"]

    scaler = StandardScaler().fit(train_x)
    train_z = scaler.transform(train_x)
    val_z = scaler.transform(val_x)
    test_z = scaler.transform(test_x)

    best: tuple[float, float, Ridge] | None = None
    for alpha in PROGRESS_ALPHA_GRID:
        regressor = Ridge(alpha=alpha).fit(train_z, train_y)
        val_prediction = np.clip(regressor.predict(val_z), 0.0, 1.0)
        val_mae = float(mean_absolute_error(val_y, val_prediction))
        if best is None or val_mae < best[0] - 1e-12:
            best = (val_mae, float(alpha), regressor)
    if best is None:
        raise RuntimeError("No progress probe candidate selected")

    val_mae, alpha, regressor = best
    metrics: dict[str, float] = {"progress_probe_alpha": alpha}
    for split, z, y in (
        ("train", train_z, train_y),
        ("val", val_z, val_y),
        ("test", test_z, test_y),
    ):
        prediction = np.clip(regressor.predict(z), 0.0, 1.0)
        metrics[f"progress_probe_{split}_mae"] = float(
            mean_absolute_error(y, prediction)
        )
        metrics[f"progress_probe_{split}_r2"] = float(r2_score(y, prediction))
    metrics["progress_probe_val_mae"] = val_mae
    return scaler, regressor, metrics


def _fit_feature_probe(
    features: dict[str, dict[str, np.ndarray]],
    data: base.Data,
    feature_name: str,
    seed: int,
    namespace: str,
) -> dict[str, object]:
    _, _, phase_metrics = _fit_phase_probe_model(
        features,
        feature_name,
        seed,
        namespace,
    )
    progress_scaler, progress_regressor, progress_metrics = (
        _fit_progress_probe_model(features, feature_name)
    )
    output: dict[str, object] = {
        "feature_type": feature_name,
        "feature_dim": int(features["train"][feature_name].shape[1]),
        **phase_metrics,
        **progress_metrics,
    }
    for split, lengths in _lengths_by_split(data).items():
        sequences = _sequences_from_flat(
            features[split][feature_name],
            lengths,
        )
        trajectory = objective_parent._trajectory_metrics(
            sequences,
            progress_scaler,
            progress_regressor,
        )
        for name, value in trajectory.items():
            output[f"progress_probe_{split}_{name}"] = value
    return output


def _main_probe_rows(
    features: dict[str, dict[str, np.ndarray]],
    data: base.Data,
    seed: int,
) -> list[dict[str, object]]:
    return [
        _fit_feature_probe(
            features,
            data,
            feature_name,
            seed,
            COMMON_PROBE_NAMESPACE,
        )
        for feature_name in (
            "membrane",
            "synaptic",
            "spike",
            "spike250",
            "spike500",
        )
    ]


def _activity_diagnostics(
    spikes: np.ndarray,
    fs: float,
) -> dict[str, float | int]:
    if spikes.ndim != 2 or spikes.shape[0] <= 0 or spikes.shape[1] <= 0:
        raise ValueError("Expected non-empty [valid_steps, neurons] spike matrix")
    neuron_fr = np.asarray(spikes, dtype=np.float64).mean(axis=0)
    mean_fr = float(neuron_fr.mean())
    return {
        "n_valid_timesteps": int(spikes.shape[0]),
        "n_neurons": int(spikes.shape[1]),
        "mean_firing_rate": mean_fr,
        "mean_firing_rate_hz": mean_fr * float(fs),
        "median_neuron_firing_rate": float(np.median(neuron_fr)),
        "dead_neuron_fraction": float(np.mean(neuron_fr < DEAD_FR_THRESHOLD)),
        "highly_active_neuron_fraction": float(
            np.mean(neuron_fr > HIGHLY_ACTIVE_FR_THRESHOLD)
        ),
        "dead_fr_threshold": DEAD_FR_THRESHOLD,
        "highly_active_fr_threshold": HIGHLY_ACTIVE_FR_THRESHOLD,
    }


def _representation_diagnostics(
    membrane: np.ndarray,
) -> dict[str, float | int]:
    if membrane.ndim != 2 or membrane.shape[0] <= 0 or membrane.shape[1] <= 0:
        raise ValueError("Expected non-empty [valid_steps, hidden_width] membrane matrix")
    centered = np.asarray(membrane, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    if centered.shape[0] <= 1:
        eigenvalues = np.zeros(centered.shape[1], dtype=np.float64)
    else:
        covariance = (centered.T @ centered) / float(centered.shape[0] - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
    eigenvalues = np.sort(eigenvalues)[::-1]
    total = float(eigenvalues.sum())
    squared = float(np.square(eigenvalues).sum())
    if total <= EPS:
        effective_dimension = 0.0
        pca90_dimension = 0
    else:
        effective_dimension = total * total / max(squared, EPS)
        cumulative = np.cumsum(eigenvalues) / total
        pca90_dimension = int(np.searchsorted(cumulative, 0.90, side="left") + 1)
    return {
        "n_valid_timesteps": int(membrane.shape[0]),
        "hidden_width": int(membrane.shape[1]),
        "effective_dimension": float(effective_dimension),
        "pca90_dimension": int(pca90_dimension),
        "total_variance": total,
    }


def evaluate_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    _validate_spec(spec)
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))

    pconfig = _parent_config(config)
    pspec = _parent_spec(spec.seed)
    cache = parent.load_local_cache(spec.seed, data, pconfig)
    model, checkpoint = load_model(spec, data, config)
    device = torch.device(config.device)
    loaders = parent._make_loaders(
        data,
        cache,
        pspec,
        pconfig,
        train_shuffle=False,
    )

    native = {
        split: _evaluate_native(model, loader, device)
        for split, loader in loaders.items()
    }
    ordered_features = parent.collect_features(
        model,
        data,
        cache,
        pspec,
        pconfig,
        reset_state_each_step=False,
    )
    probes = _main_probe_rows(ordered_features, data, spec.seed)
    ordered_membrane = next(
        probe for probe in probes if probe["feature_type"] == "membrane"
    )
    activity = _activity_diagnostics(
        ordered_features["test"]["spike"],
        data.fs,
    )
    representation = _representation_diagnostics(
        ordered_features["test"]["membrane"]
    )

    reset_features = parent.collect_features(
        model,
        data,
        cache,
        pspec,
        pconfig,
        reset_state_each_step=True,
    )
    reset_probe = _fit_feature_probe(
        reset_features,
        data,
        "membrane",
        spec.seed,
        RESET_PROBE_NAMESPACE,
    )
    reset_native = {
        split: _evaluate_native(
            model,
            loader,
            device,
            reset_state_each_step=True,
        )
        for split, loader in loaders.items()
    }

    ablations: list[dict[str, object]] = [
        {
            "ablation": "state_reset",
            "replicate": 0,
            "native": reset_native,
            "membrane_probe": reset_probe,
        }
    ]

    for replicate in range(SHUFFLE_REPLICATES):
        shuffled_cache = parent._shuffle_cache(
            data,
            cache,
            spec.seed,
            replicate,
        )
        shuffled_loaders = parent._make_loaders(
            data,
            shuffled_cache,
            pspec,
            pconfig,
            train_shuffle=False,
        )
        shuffled_native = {
            split: _evaluate_native(model, loader, device)
            for split, loader in shuffled_loaders.items()
        }
        shuffled_features = parent.collect_features(
            model,
            data,
            shuffled_cache,
            pspec,
            pconfig,
            reset_state_each_step=False,
        )
        shuffled_probe = _fit_feature_probe(
            shuffled_features,
            data,
            "membrane",
            spec.seed,
            SHUFFLE_PROBE_NAMESPACE,
        )
        ablations.append(
            {
                "ablation": "temporal_shuffle",
                "replicate": replicate,
                "native": shuffled_native,
                "membrane_probe": shuffled_probe,
            }
        )

    shuffle_probes = [
        item["membrane_probe"]
        for item in ablations
        if item["ablation"] == "temporal_shuffle"
    ]
    if len(shuffle_probes) != SHUFFLE_REPLICATES:
        raise RuntimeError(
            f"Expected {SHUFFLE_REPLICATES} temporal shuffles for {spec.key}"
        )

    def test_metric(probe: dict[str, object], name: str) -> float:
        return float(probe[f"progress_probe_test_{name}"])

    shuffle_phase_ba = float(
        np.mean([float(probe["phase_probe_test_ba"]) for probe in shuffle_probes])
    )
    shuffle_progress_mae = float(
        np.mean(
            [
                test_metric(probe, "sample_balanced_mae_mean")
                for probe in shuffle_probes
            ]
        )
    )
    shuffle_spearman = float(
        np.mean([test_metric(probe, "spearman_mean") for probe in shuffle_probes])
    )
    shuffle_violation = float(
        np.mean(
            [
                test_metric(probe, "monotonic_violation_rate_mean")
                for probe in shuffle_probes
            ]
        )
    )
    ordered_progress_mae = test_metric(
        ordered_membrane,
        "sample_balanced_mae_mean",
    )
    reset_progress_mae = test_metric(reset_probe, "sample_balanced_mae_mean")
    ordered_spearman = test_metric(ordered_membrane, "spearman_mean")
    reset_spearman = test_metric(reset_probe, "spearman_mean")
    ordered_violation = test_metric(
        ordered_membrane,
        "monotonic_violation_rate_mean",
    )
    reset_violation = test_metric(reset_probe, "monotonic_violation_rate_mean")
    history_attribution = {
        "ordered": {
            "phase_ba": float(ordered_membrane["phase_probe_test_ba"]),
            "progress_mae": ordered_progress_mae,
            "spearman": ordered_spearman,
            "monotonic_violation": ordered_violation,
        },
        "reset": {
            "phase_ba": float(reset_probe["phase_probe_test_ba"]),
            "progress_mae": reset_progress_mae,
            "spearman": reset_spearman,
            "monotonic_violation": reset_violation,
        },
        "shuffle": {
            "replicates": SHUFFLE_REPLICATES,
            "phase_ba": shuffle_phase_ba,
            "progress_mae": shuffle_progress_mae,
            "spearman": shuffle_spearman,
            "monotonic_violation": shuffle_violation,
        },
        "H_reset_phase_ba": float(ordered_membrane["phase_probe_test_ba"])
        - float(reset_probe["phase_probe_test_ba"]),
        "H_shuffle_phase_ba": float(ordered_membrane["phase_probe_test_ba"])
        - shuffle_phase_ba,
        "H_reset_progress_mae": reset_progress_mae - ordered_progress_mae,
        "H_shuffle_progress_mae": shuffle_progress_mae - ordered_progress_mae,
        "H_reset_spearman": ordered_spearman - reset_spearman,
        "H_shuffle_spearman": ordered_spearman - shuffle_spearman,
    }
    trajectory_metrics = {
        "progress_mae": ordered_progress_mae,
        "spearman": ordered_spearman,
        "monotonic_violation": ordered_violation,
    }
    counts = checkpoint["provenance"]["parameter_counts"]

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "condition": spec.condition,
        "hidden_width": spec.hidden_width,
        "seed": spec.seed,
        "parameter_count": counts["trainable_total"],
        "parameter_counts": counts,
        "objective": asdict(OBJECTIVE),
        "provenance": checkpoint["provenance"],
        "best_epoch": checkpoint["result"]["best_epoch"],
        "native": native,
        "probes": probes,
        "ordered_membrane_probe": ordered_membrane,
        "ablations": ablations,
        "trajectory_metrics": trajectory_metrics,
        "history_attribution": history_attribution,
        "activity": activity,
        "representation": representation,
    }
    _save_json(destination, payload)
    return payload


def run_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    train_one(spec, data, config, force=force)
    return evaluate_one(spec, data, config, force=force)


def _probe_trajectory_columns(
    probe: dict[str, object],
    split: str,
) -> dict[str, object]:
    prefix = f"progress_probe_{split}_"
    names = (
        "sample_balanced_mae_mean",
        "sample_balanced_mae_sd",
        "spearman_mean",
        "spearman_sd",
        "monotonic_violation_rate_mean",
        "monotonic_violation_rate_sd",
    )
    return {
        f"probe_{split}_{name}": probe.get(f"{prefix}{name}", np.nan)
        for name in names
    }


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    run_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    history_gain_rows: list[dict[str, object]] = []
    activity_rows: list[dict[str, object]] = []
    representation_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    local_rows: list[dict[str, object]] = []
    history_parts: list[pd.DataFrame] = []

    specs = run_specs()
    if len(specs) != 25 or len({spec.key for spec in specs}) != 25:
        raise RuntimeError("Exp5.3.2.2 run matrix must contain exactly 25 unique runs")

    for width in WIDTHS:
        if sum(spec.hidden_width == width for spec in specs) != len(SEEDS):
            raise RuntimeError(f"Width {width} does not have exactly five seeds")

    for spec in specs:
        epath = evaluation_path(root, spec)
        if not epath.exists():
            raise FileNotFoundError(f"Missing Exp5.3.2.2 evaluation: {epath}")
        payload = json.loads(epath.read_text(encoding="utf-8"))
        if (
            payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("spec") != asdict(spec)
            or payload.get("hidden_width") != spec.hidden_width
        ):
            raise ValueError(f"Evaluation identity mismatch: {epath}")
        missing_sections = [
            name for name in EVALUATION_REQUIRED_SECTIONS if name not in payload
        ]
        if missing_sections:
            raise ValueError(
                f"Evaluation missing required sections {missing_sections}: {epath}"
            )

        counts = payload["provenance"]["parameter_counts"]
        common = {
            "condition": spec.condition,
            "hidden_width": spec.hidden_width,
            "seed": spec.seed,
            "parameter_count": counts["trainable_total"],
            "recurrent_parameter_count": counts["recurrent"],
        }
        ordered_probe = payload["ordered_membrane_probe"]
        row: dict[str, object] = {
            **common,
            "best_epoch": payload["best_epoch"],
        }
        for split in ("train", "val", "test"):
            for metric, value in payload["native"][split].items():
                row[f"native_{split}_{metric}"] = value
            row[f"probe_u_{split}_phase_ba"] = ordered_probe[
                f"phase_probe_{split}_ba"
            ]
            row[f"probe_u_{split}_progress_mae"] = ordered_probe[
                f"progress_probe_{split}_mae"
            ]
            row.update(_probe_trajectory_columns(ordered_probe, split))
        row["probe_phase_train_val_gap"] = (
            row["probe_u_train_phase_ba"] - row["probe_u_val_phase_ba"]
        )
        row["probe_progress_val_train_gap"] = (
            row["probe_val_sample_balanced_mae_mean"]
            - row["probe_train_sample_balanced_mae_mean"]
        )
        run_rows.append(row)

        for probe in payload["probes"]:
            probe_rows.append({**common, **probe})
        activity_rows.append({**common, **payload["activity"]})
        representation_rows.append({**common, **payload["representation"]})

        def ablation_row(
            ablation: str,
            replicate: int,
            native_metrics: dict[str, dict[str, float]],
            probe: dict[str, object],
        ) -> dict[str, object]:
            return {
                **common,
                "ablation": ablation,
                "replicate": replicate,
                "native_test_phase_ba": native_metrics["test"][
                    "phase_balanced_accuracy"
                ],
                "native_test_progress_mae": native_metrics["test"][
                    "progress_sample_balanced_mae"
                ],
                "probe_test_phase_ba": probe["phase_probe_test_ba"],
                "probe_test_progress_mae": probe["progress_probe_test_mae"],
                **_probe_trajectory_columns(probe, "test"),
            }

        ordered_row = ablation_row(
            "ordered",
            0,
            payload["native"],
            ordered_probe,
        )
        ablation_rows.append(ordered_row)
        reset_entry = next(
            item
            for item in payload["ablations"]
            if item["ablation"] == "state_reset"
        )
        reset_row = ablation_row(
            "state_reset",
            int(reset_entry["replicate"]),
            reset_entry["native"],
            reset_entry["membrane_probe"],
        )
        ablation_rows.append(reset_row)

        shuffle_rows: list[dict[str, object]] = []
        for item in payload["ablations"]:
            if item["ablation"] != "temporal_shuffle":
                continue
            shuffle_row = ablation_row(
                "temporal_shuffle",
                int(item["replicate"]),
                item["native"],
                item["membrane_probe"],
            )
            ablation_rows.append(shuffle_row)
            shuffle_rows.append(shuffle_row)
        if len(shuffle_rows) != SHUFFLE_REPLICATES:
            raise ValueError(
                f"Expected {SHUFFLE_REPLICATES} shuffles for {spec.key}, "
                f"found {len(shuffle_rows)}"
            )

        shuffle_phase_ba = float(
            np.mean([item["probe_test_phase_ba"] for item in shuffle_rows])
        )
        shuffle_progress_mae = float(
            np.mean(
                [
                    item["probe_test_sample_balanced_mae_mean"]
                    for item in shuffle_rows
                ]
            )
        )
        shuffle_spearman = float(
            np.mean([item["probe_test_spearman_mean"] for item in shuffle_rows])
        )
        shuffle_violation = float(
            np.mean(
                [
                    item["probe_test_monotonic_violation_rate_mean"]
                    for item in shuffle_rows
                ]
            )
        )
        history_gain_rows.append(
            {
                **common,
                "ordered_phase_ba": ordered_row["probe_test_phase_ba"],
                "reset_phase_ba": reset_row["probe_test_phase_ba"],
                "shuffle_phase_ba": shuffle_phase_ba,
                "H_reset_phase_ba": ordered_row["probe_test_phase_ba"]
                - reset_row["probe_test_phase_ba"],
                "H_shuffle_phase_ba": ordered_row["probe_test_phase_ba"]
                - shuffle_phase_ba,
                "ordered_progress_mae": ordered_row[
                    "probe_test_sample_balanced_mae_mean"
                ],
                "reset_progress_mae": reset_row[
                    "probe_test_sample_balanced_mae_mean"
                ],
                "shuffle_progress_mae": shuffle_progress_mae,
                "H_reset_progress_mae": reset_row[
                    "probe_test_sample_balanced_mae_mean"
                ]
                - ordered_row["probe_test_sample_balanced_mae_mean"],
                "H_shuffle_progress_mae": shuffle_progress_mae
                - ordered_row["probe_test_sample_balanced_mae_mean"],
                "ordered_spearman": ordered_row["probe_test_spearman_mean"],
                "reset_spearman": reset_row["probe_test_spearman_mean"],
                "shuffle_spearman": shuffle_spearman,
                "H_reset_spearman": ordered_row["probe_test_spearman_mean"]
                - reset_row["probe_test_spearman_mean"],
                "H_shuffle_spearman": ordered_row["probe_test_spearman_mean"]
                - shuffle_spearman,
                "ordered_monotonic_violation": ordered_row[
                    "probe_test_monotonic_violation_rate_mean"
                ],
                "reset_monotonic_violation": reset_row[
                    "probe_test_monotonic_violation_rate_mean"
                ],
                "shuffle_monotonic_violation": shuffle_violation,
            }
        )

        hpath = history_path(root, spec)
        if not hpath.exists():
            raise FileNotFoundError(f"Missing Exp5.3.2.2 history: {hpath}")
        history = pd.read_csv(hpath)
        history.insert(0, "seed", spec.seed)
        history.insert(0, "hidden_width", spec.hidden_width)
        history.insert(0, "condition", spec.condition)
        history_parts.append(history)

    for seed in SEEDS:
        bpath = baseline_path(root, seed)
        if not bpath.exists():
            raise FileNotFoundError(f"Missing Exp5.3.2.2 baseline: {bpath}")
        baseline = json.loads(bpath.read_text(encoding="utf-8"))
        if (
            baseline.get("experiment_id") != EXPERIMENT_ID
            or baseline.get("protocol_version") != PROTOCOL_VERSION
            or baseline.get("seed") != seed
            or baseline.get("source_experiment_id")
            != objective_parent.EXPERIMENT_ID
            or baseline.get("source_protocol_version")
            != objective_parent.PROTOCOL_VERSION
        ):
            raise ValueError(f"Baseline identity mismatch: {bpath}")
        for entry in baseline["baselines"]:
            baseline_rows.append({"seed": seed, **entry})

        reference_path = exp52.local_reference_path(
            parent.source_results_dir(repo_root),
            seed,
        )
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Missing frozen Local reference: {reference_path}"
            )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        for probe in reference["probes"]:
            local_rows.append(
                {
                    "seed": seed,
                    "probe_type": probe["probe_type"],
                    "val_ba": probe["probe_val_balanced_accuracy"],
                    "test_ba": probe["probe_test_balanced_accuracy"],
                    "test_accuracy": probe["probe_test_accuracy"],
                    "test_macro_f1": probe["probe_test_macro_f1"],
                }
            )

    outputs = {
        "runs": root / "runs.csv",
        "histories": root / "histories.csv",
        "probe_runs": root / "probe_runs.csv",
        "ablation_runs": root / "ablation_runs.csv",
        "history_gain_runs": root / "history_gain_runs.csv",
        "activity_runs": root / "activity_runs.csv",
        "representation_runs": root / "representation_runs.csv",
        "baseline_runs": root / "baseline_runs.csv",
        "local_reference": root / "local_reference.csv",
        "manifest": root / "manifest.json",
    }
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(outputs["runs"], index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(
        outputs["histories"],
        index=False,
    )
    pd.DataFrame(probe_rows).to_csv(outputs["probe_runs"], index=False)
    pd.DataFrame(ablation_rows).to_csv(outputs["ablation_runs"], index=False)
    pd.DataFrame(history_gain_rows).to_csv(
        outputs["history_gain_runs"],
        index=False,
    )
    pd.DataFrame(activity_rows).to_csv(outputs["activity_runs"], index=False)
    pd.DataFrame(representation_rows).to_csv(
        outputs["representation_runs"],
        index=False,
    )
    pd.DataFrame(baseline_rows).to_csv(outputs["baseline_runs"], index=False)
    pd.DataFrame(local_rows).to_csv(outputs["local_reference"], index=False)

    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "scientific_question": (
                "whether the supervised causal WHEN ceiling is limited by "
                "single-layer RSNN hidden-state capacity"
            ),
            "expected_runs": len(specs),
            "expected_runs_per_width": len(SEEDS),
            "widths": list(WIDTHS),
            "conditions": list(WIDTH_CONDITIONS),
            "seeds": list(SEEDS),
            "run_order": (
                "width-major: tasks 0-4 H16, 5-9 H32, 10-14 H64, "
                "15-19 H128, 20-24 H256"
            ),
            "fixed_objective": asdict(OBJECTIVE),
            "fixed_when_architecture": (
                "one-layer short-tau recurrent SNN; only hidden width changes"
            ),
            "source_objective_experiment": objective_parent.EXPERIMENT_ID,
            "source_objective_protocol": objective_parent.PROTOCOL_VERSION,
            "source_when_experiment": parent.EXPERIMENT_ID,
            "source_when_protocol": parent.PROTOCOL_VERSION,
            "source_local_experiment": exp52.EXPERIMENT_ID,
            "source_local_protocol": exp52.PROTOCOL_VERSION,
            "checkpoint_selection": (
                "min validation joint objective loss; tie lower progress MAE; "
                "tie higher phase BA"
            ),
            "paired_initialization": (
                "deterministic width-specific component seeds; no prefix copying"
            ),
            "paired_minibatch_order": (
                "same Exp5.3.2 loader seed for every width within a master seed"
            ),
            "causality_contract": (
                "T_i is used only to construct phase/progress supervision; "
                "T_i is not an RSNN/head/deployment input"
            ),
            "evaluation_required_sections": list(EVALUATION_REQUIRED_SECTIONS),
            "primary_metrics": [
                "U phase balanced accuracy",
                "U progress sample-balanced MAE",
                "per-gesture progress Spearman",
                "monotonic violation rate",
                "H_reset and H_shuffle for phase/progress",
                "train/val/test probe generalization",
                "spike-domain phase/progress accessibility",
                "firing-rate utilization",
                "effective dimension and PCA90 dimension",
            ],
            "spike_readouts": [
                "membrane",
                "synaptic",
                "instantaneous spike",
                "trailing 250ms spike count",
                "trailing 500ms spike count",
            ],
            "activity_thresholds": {
                "dead_neuron_fraction": f"FR < {DEAD_FR_THRESHOLD}",
                "highly_active_neuron_fraction": (
                    f"FR > {HIGHLY_ACTIVE_FR_THRESHOLD} spikes/timestep"
                ),
            },
            "representation_diagnostic": (
                "covariance eigenspectrum of all valid test U_t; participation "
                "ratio and number of PCs explaining >=90% variance"
            ),
            "baselines": ["what_only", "elapsed_time_only"],
            "baseline_reuse": (
                "identity-validated Exp5.3.2.1 baseline artifacts; no refitting"
            ),
            "winner_selection": [
                "require mean ProgressMAE < elapsed-time mean",
                "require positive mean H_reset and H_shuffle for both phase and progress",
                "then prefer higher PhaseBA, lower ProgressMAE, larger history gain",
                "then prefer fewer parameters and better spike accessibility",
            ],
            "aggregation_policy": (
                "finalizer requires all 25 evaluations, all 25 histories, all "
                "five baseline artifacts, and only aggregates existing artifacts"
            ),
            "files": {
                name: path.name
                for name, path in outputs.items()
                if name != "manifest"
            },
        },
    )
    return outputs


def _config_from_args(args: argparse.Namespace) -> Config:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else find_repo_root()
    )
    return Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threads=args.threads,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 5.3.2.2 RSNN hidden-width capacity sweep"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo-root", type=str, default=None)
        subparser.add_argument("--device", type=str, default="cpu")
        subparser.add_argument("--threads", type=int, default=1)
        subparser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        subparser.add_argument("--epochs", type=int, default=EPOCHS)
        subparser.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare-local")
    common(prepare)
    prepare.add_argument("--array-task-id", type=int, required=True)

    run = subparsers.add_parser("run-one")
    common(run)
    run.add_argument("--array-task-id", type=int, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--repo-root", type=str, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "finalize":
        repo_root = (
            Path(args.repo_root).resolve()
            if args.repo_root
            else find_repo_root()
        )
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return

    config = _config_from_args(args)
    data = base.prepare_data(config.repo_root)

    if args.command == "prepare-local":
        task_id = int(args.array_task_id)
        if not 0 <= task_id < len(SEEDS):
            raise IndexError(
                f"prepare-local task {task_id} outside 0..{len(SEEDS)-1}"
            )
        print(
            prepare_local_seed(
                SEEDS[task_id],
                data,
                config,
                force=args.force,
            )
        )
        return

    if args.command == "run-one":
        specs = run_specs()
        task_id = int(args.array_task_id)
        if not 0 <= task_id < len(specs):
            raise IndexError(
                f"run-one task {task_id} outside 0..{len(specs)-1}"
            )
        payload = run_one(
            specs[task_id],
            data,
            config,
            force=args.force,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
