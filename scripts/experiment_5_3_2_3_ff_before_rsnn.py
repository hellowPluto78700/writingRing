from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from snntorch import surrogate

from scripts import experiment_5_3_2_2_when_width_sweep as width_parent


objective_parent = width_parent.objective_parent
parent = width_parent.parent
base = width_parent.base
exp52 = width_parent.exp52

EXPERIMENT_ID = "experiment_5_3_2_3_ff_before_rsnn"
PROTOCOL_VERSION = "ff_before_rsnn_v1"
SEEDS = width_parent.SEEDS
RSNN_WIDTH = 64
DIRECT_CONDITION = "direct_rsnn64"
LINEAR_CONTROL_CONDITION = "linear64_rsnn64"
FF64_CONDITION = "ffsnn64_rsnn64"
FF128_CONDITION = "ffsnn128_rsnn64"
TRAINABLE_CONDITIONS = (
    LINEAR_CONTROL_CONDITION,
    FF64_CONDITION,
    FF128_CONDITION,
)
ALL_CONDITIONS = (DIRECT_CONDITION, *TRAINABLE_CONDITIONS)
OBJECTIVE = width_parent.OBJECTIVE
LOCAL_WIDTH = width_parent.LOCAL_WIDTH
SHORT_SHIFT = width_parent.SHORT_SHIFT
N_PHASES = width_parent.N_PHASES
THRESHOLD = width_parent.THRESHOLD
RESET = width_parent.RESET
SURROGATE_SLOPE = width_parent.SURROGATE_SLOPE
EPOCHS = width_parent.EPOCHS
BATCH_SIZE = width_parent.BATCH_SIZE
LR = width_parent.LR
WEIGHT_DECAY = width_parent.WEIGHT_DECAY
SHUFFLE_REPLICATES = width_parent.SHUFFLE_REPLICATES
SPIKE_WINDOWS_SECONDS = width_parent.SPIKE_WINDOWS_SECONDS
EVALUATION_REQUIRED_SECTIONS = width_parent.EVALUATION_REQUIRED_SECTIONS


@dataclass(frozen=True)
class RunSpec:
    condition: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.condition}__seed{self.seed}"

    @property
    def ff_width(self) -> int:
        if self.condition in (LINEAR_CONTROL_CONDITION, FF64_CONDITION):
            return 64
        if self.condition == FF128_CONDITION:
            return 128
        if self.condition == DIRECT_CONDITION:
            return 0
        raise ValueError(f"Unknown Exp5.3.2.3 condition: {self.condition}")

    @property
    def front_end(self) -> str:
        if self.condition == DIRECT_CONDITION:
            return "direct"
        if self.condition == LINEAR_CONTROL_CONDITION:
            return "linear"
        if self.condition in (FF64_CONDITION, FF128_CONDITION):
            return "ff_snn"
        raise ValueError(f"Unknown Exp5.3.2.3 condition: {self.condition}")


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
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def direct_reference_path(root: Path, seed: int) -> Path:
    return root / "direct_references" / f"seed{seed}.json"


def baseline_path(root: Path, seed: int) -> Path:
    return root / "baselines" / f"seed{seed}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_specs() -> list[RunSpec]:
    return [RunSpec(condition, seed) for condition in TRAINABLE_CONDITIONS for seed in SEEDS]


def all_specs() -> list[RunSpec]:
    return [RunSpec(condition, seed) for condition in ALL_CONDITIONS for seed in SEEDS]


def _validate_spec(spec: RunSpec, allow_direct: bool = False) -> None:
    allowed = ALL_CONDITIONS if allow_direct else TRAINABLE_CONDITIONS
    if spec.condition not in allowed:
        raise ValueError(f"Unknown Exp5.3.2.3 condition: {spec.condition}")
    if spec.seed not in SEEDS:
        raise ValueError(f"Unknown Exp5.3.2.3 seed: {spec.seed}")


def _parent_config(config: Config) -> parent.Config:
    return parent.Config(
        repo_root=config.repo_root,
        results_dir=parent.results_dir(config.repo_root),
        device=config.device,
        epochs=config.epochs,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _width_config(config: Config) -> width_parent.Config:
    return width_parent.Config(
        repo_root=config.repo_root,
        results_dir=width_parent.results_dir(config.repo_root),
        device=config.device,
        epochs=config.epochs,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _parent_spec(seed: int) -> parent.RunSpec:
    return width_parent._parent_spec(seed)


def _loader_seed(seed: int, split: str) -> int:
    return width_parent._loader_seed(seed, split)


class FrontEndWhenNet(nn.Module):
    """Fixed RSNN64 with an optional pre-recurrent transform.

    The linear64 and ffsnn64 conditions have identical trainable weight shapes.
    The only architectural difference between them is the short-tau LIF transform
    inserted between the two linear maps.
    """

    def __init__(self, condition: str, fs: float) -> None:
        super().__init__()
        if condition not in TRAINABLE_CONDITIONS:
            raise ValueError(f"Unsupported trainable condition: {condition}")
        self.condition = condition
        self.fs = float(fs)
        self.rsnn_width = RSNN_WIDTH
        self.front_end = "linear" if condition == LINEAR_CONTROL_CONDITION else "ff_snn"
        self.ff_width = 64 if condition in (LINEAR_CONTROL_CONDITION, FF64_CONDITION) else 128
        self.spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

        self.front_projection = nn.Linear(LOCAL_WIDTH, self.ff_width, bias=False)
        self.input_projection = nn.Linear(self.ff_width, RSNN_WIDTH, bias=False)
        self.recurrent = nn.Linear(RSNN_WIDTH, RSNN_WIDTH, bias=False)
        self.phase_head = nn.Linear(RSNN_WIDTH, N_PHASES, bias=True)
        self.progress_head = nn.Linear(RSNN_WIDTH, 1, bias=True)

        decay = parent.decay_from_shift(SHORT_SHIFT)
        self.register_buffer(
            "alpha_vector",
            torch.full((RSNN_WIDTH,), decay, dtype=torch.float32),
        )
        self.register_buffer(
            "beta_vector",
            torch.full((RSNN_WIDTH,), decay, dtype=torch.float32),
        )
        if self.front_end == "ff_snn":
            self.register_buffer(
                "front_alpha_vector",
                torch.full((self.ff_width,), decay, dtype=torch.float32),
            )
            self.register_buffer(
                "front_beta_vector",
                torch.full((self.ff_width,), decay, dtype=torch.float32),
            )

    @staticmethod
    def _reset_membrane(membrane: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        return width_parent.WidthWhenBranchNet._reset_membrane(membrane, spike)

    def _initial_rsnn_state(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (x.shape[0], RSNN_WIDTH)
        synaptic = torch.zeros(shape, dtype=x.dtype, device=x.device)
        membrane = torch.zeros_like(synaptic)
        spike = torch.zeros_like(synaptic)
        return synaptic, membrane, spike

    def _initial_front_state(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (x.shape[0], self.ff_width)
        synaptic = torch.zeros(shape, dtype=x.dtype, device=x.device)
        membrane = torch.zeros_like(synaptic)
        return synaptic, membrane

    def _front_step(
        self,
        local_t: torch.Tensor,
        synaptic: torch.Tensor,
        membrane: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = self.front_projection(local_t)
        synaptic = self.front_alpha_vector.to(current) * synaptic + current
        membrane = self.front_beta_vector.to(current) * membrane + synaptic
        spike = self.spike_grad(membrane - THRESHOLD)
        membrane = self._reset_membrane(membrane, spike)
        return synaptic, membrane, spike

    def _rsnn_step(
        self,
        front_t: torch.Tensor,
        synaptic: torch.Tensor,
        membrane: torch.Tensor,
        previous_spike: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = self.input_projection(front_t) + self.recurrent(previous_spike)
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

        synaptic, membrane, previous_spike = self._initial_rsnn_state(x)
        if self.front_end == "ff_snn":
            front_synaptic, front_membrane = self._initial_front_state(x)
        else:
            front_synaptic = front_membrane = None

        synaptic_parts: list[torch.Tensor] = []
        membrane_parts: list[torch.Tensor] = []
        spike_parts: list[torch.Tensor] = []

        for timestep in range(x.shape[1]):
            if reset_state_each_step:
                synaptic = torch.zeros_like(synaptic)
                membrane = torch.zeros_like(membrane)
                previous_spike = torch.zeros_like(previous_spike)
                if self.front_end == "ff_snn":
                    assert front_synaptic is not None and front_membrane is not None
                    front_synaptic = torch.zeros_like(front_synaptic)
                    front_membrane = torch.zeros_like(front_membrane)

            valid = (timestep < lengths).unsqueeze(1)
            local_t = x[:, timestep]
            if self.front_end == "ff_snn":
                assert front_synaptic is not None and front_membrane is not None
                next_front_syn, next_front_mem, next_front_spike = self._front_step(
                    local_t,
                    front_synaptic,
                    front_membrane,
                )
                front_synaptic = torch.where(valid, next_front_syn, front_synaptic)
                front_membrane = torch.where(valid, next_front_mem, front_membrane)
                front_t = next_front_spike * valid.to(next_front_spike.dtype)
            else:
                front_t = self.front_projection(local_t)

            next_synaptic, next_membrane, next_spike = self._rsnn_step(
                front_t,
                synaptic,
                membrane,
                previous_spike,
            )
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
        phase_logits, progress_pred = self.heads_from_membrane(trajectory.membranes)
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
        return (
            phase_loss + progress_loss,
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


def _initialization_namespace(condition: str) -> str:
    if condition in (LINEAR_CONTROL_CONDITION, FF64_CONDITION):
        return "paired_front64"
    return condition


def _initialize_model(
    condition: str,
    seed: int,
    fs: float,
    device: torch.device,
) -> FrontEndWhenNet:
    if condition not in TRAINABLE_CONDITIONS:
        raise ValueError(f"Cannot initialize non-trainable condition: {condition}")
    namespace = _initialization_namespace(condition)
    base.seed_all(base.dseed(seed, EXPERIMENT_ID, namespace, "constructor"))
    model = FrontEndWhenNet(condition, fs).to(device)
    for component, module in (
        ("front_projection", model.front_projection),
        ("input_projection", model.input_projection),
        ("recurrent", model.recurrent),
        ("phase_head", model.phase_head),
        ("progress_head", model.progress_head),
    ):
        base.seed_all(base.dseed(seed, EXPERIMENT_ID, namespace, component))
        module.reset_parameters()
    return model


def parameter_counts(model: FrontEndWhenNet) -> dict[str, int]:
    return {
        "front_projection": int(sum(p.numel() for p in model.front_projection.parameters())),
        "input_projection": int(sum(p.numel() for p in model.input_projection.parameters())),
        "recurrent": int(sum(p.numel() for p in model.recurrent.parameters())),
        "phase_head": int(sum(p.numel() for p in model.phase_head.parameters())),
        "progress_head": int(sum(p.numel() for p in model.progress_head.parameters())),
        "trainable_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def _provenance(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    model: FrontEndWhenNet,
) -> dict[str, object]:
    counts = parameter_counts(model)
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": spec.seed,
        "split_seed": base.SPLIT_SEED,
        "condition": spec.condition,
        "front_end": spec.front_end,
        "ff_width": spec.ff_width,
        "rsnn_width": RSNN_WIDTH,
        "local_width": LOCAL_WIDTH,
        "front_short_tau": spec.front_end == "ff_snn",
        "recurrent_short_tau": True,
        "shift_mem": SHORT_SHIFT,
        "shift_syn": SHORT_SHIFT,
        "tau_mem_ms": parent.tau_ms_from_shift(SHORT_SHIFT, data.fs),
        "tau_syn_ms": parent.tau_ms_from_shift(SHORT_SHIFT, data.fs),
        "threshold": THRESHOLD,
        "reset": RESET,
        "surrogate_slope": SURROGATE_SLOPE,
        "sampling_rate_hz": float(data.fs),
        "source_width_experiment": width_parent.EXPERIMENT_ID,
        "source_width_protocol": width_parent.PROTOCOL_VERSION,
        "source_objective_experiment": objective_parent.EXPERIMENT_ID,
        "source_objective_protocol": objective_parent.PROTOCOL_VERSION,
        "source_when_experiment": parent.EXPERIMENT_ID,
        "source_when_protocol": parent.PROTOCOL_VERSION,
        "source_local_experiment": exp52.EXPERIMENT_ID,
        "source_local_protocol": exp52.PROTOCOL_VERSION,
        "local_source_frozen": True,
        "objective": OBJECTIVE.name,
        "objective_formula": "L_phase + L_progress",
        "loss_reduction": "mean over valid timesteps within each sample, then mean over samples",
        "causality_contract": (
            "T_i constructs supervision/mask only; local WHAT, front-end, RSNN and heads "
            "never receive final duration"
        ),
        "checkpoint_selection": (
            "min validation joint objective loss; tie lower validation progress "
            "sample-balanced MAE; tie higher validation phase BA"
        ),
        "initialization": (
            "deterministic architecture/component seeds; linear64 and ffsnn64 share "
            "all trainable weight initializations so only the LIF transform differs"
        ),
        "minibatch_order": (
            "paired across trainable architectures within seed via unchanged Exp5.3.2 loader seeds"
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
        "parameter_counts": counts,
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
    }


def _direct_width_spec(seed: int) -> width_parent.RunSpec:
    return width_parent.RunSpec(RSNN_WIDTH, seed)


def prepare_local_seed(
    seed: int,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Unknown Exp5.3.2.3 seed: {seed}")

    wconfig = _width_config(config)
    width_parent.prepare_local_seed(seed, data, wconfig, force=False)
    cache = parent.load_local_cache(seed, data, _parent_config(config))
    for split, (x, lengths) in parent._partitions(data, cache).items():
        if x.shape[0] != len(lengths) or x.shape[-1] != LOCAL_WIDTH:
            raise ValueError(f"Frozen WHAT cache mismatch for seed {seed} split {split}")

    direct_spec = _direct_width_spec(seed)
    source_eval = width_parent.evaluation_path(wconfig.results_dir, direct_spec)
    source_history = width_parent.history_path(wconfig.results_dir, direct_spec)
    source_checkpoint = width_parent.checkpoint_path(wconfig.results_dir, direct_spec)
    for path in (source_eval, source_history, source_checkpoint):
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp5.3.2.2 H64 source artifact: {path}")
    direct_payload = json.loads(source_eval.read_text(encoding="utf-8"))
    if (
        direct_payload.get("experiment_id") != width_parent.EXPERIMENT_ID
        or direct_payload.get("protocol_version") != width_parent.PROTOCOL_VERSION
        or direct_payload.get("hidden_width") != RSNN_WIDTH
        or direct_payload.get("seed") != seed
    ):
        raise ValueError(f"Exp5.3.2.2 H64 evaluation identity mismatch: {source_eval}")

    ref_destination = direct_reference_path(config.results_dir, seed)
    if force or not ref_destination.exists():
        _save_json(
            ref_destination,
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "condition": DIRECT_CONDITION,
                "source_experiment_id": width_parent.EXPERIMENT_ID,
                "source_protocol_version": width_parent.PROTOCOL_VERSION,
                "source_condition": f"rsnn_h{RSNN_WIDTH}",
                "source_hidden_width": RSNN_WIDTH,
                "evaluation_path": str(source_eval.relative_to(config.repo_root)),
                "history_path": str(source_history.relative_to(config.repo_root)),
                "checkpoint_path": str(source_checkpoint.relative_to(config.repo_root)),
                "reuse_policy": "identity-validated reuse; direct RSNN64 is not retrained in Exp5.3.2.3",
            },
        )

    source_baseline = width_parent.baseline_path(wconfig.results_dir, seed)
    if not source_baseline.exists():
        raise FileNotFoundError(f"Missing Exp5.3.2.2 baseline artifact: {source_baseline}")
    baseline_payload = json.loads(source_baseline.read_text(encoding="utf-8"))
    destination = baseline_path(config.results_dir, seed)
    if force or not destination.exists():
        _save_json(
            destination,
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "source_experiment_id": width_parent.EXPERIMENT_ID,
                "source_protocol_version": width_parent.PROTOCOL_VERSION,
                "source_path": str(source_baseline.relative_to(config.repo_root)),
                "baselines": baseline_payload["baselines"],
            },
        )
    return exp52.local_cache_path(parent.source_results_dir(config.repo_root), seed)


def _evaluate_native(
    model: FrontEndWhenNet,
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
    train_loaders = parent._make_loaders(data, cache, pspec, pconfig, train_shuffle=True)
    eval_loaders = parent._make_loaders(data, cache, pspec, pconfig, train_shuffle=False)

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _initialize_model(spec.condition, spec.seed, data.fs, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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
            _, phase_loss, progress_loss, phase_logits, progress_pred, _ = model.loss_components(
                local,
                lengths,
            )
            objective_loss = objective_parent._objective_loss(
                OBJECTIVE,
                phase_loss,
                progress_loss,
            )
            objective_loss.backward()
            optimizer.step()

            phase_true, phase_pred, _, _, sample_abs_sum = parent._native_metrics_from_batch(
                phase_logits.detach(),
                progress_pred.detach(),
                lengths,
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
        val_progress_mae = float(val_metrics["progress_sample_balanced_mae"])
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
    native = {split: _evaluate_native(model, loader, device) for split, loader in eval_loaders.items()}
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
) -> tuple[FrontEndWhenNet, dict[str, object]]:
    _validate_spec(spec)
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.3.2.3 checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("spec") != asdict(spec)
    ):
        raise ValueError(f"Wrong Exp5.3.2.3 checkpoint identity: {path}")
    model = FrontEndWhenNet(spec.condition, data.fs).to(torch.device(config.device))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


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
    loaders = parent._make_loaders(data, cache, pspec, pconfig, train_shuffle=False)

    native = {split: _evaluate_native(model, loader, device) for split, loader in loaders.items()}
    ordered_features = parent.collect_features(
        model,
        data,
        cache,
        pspec,
        pconfig,
        reset_state_each_step=False,
    )
    probes = width_parent._main_probe_rows(ordered_features, data, spec.seed)
    ordered_membrane = next(probe for probe in probes if probe["feature_type"] == "membrane")
    activity = width_parent._activity_diagnostics(ordered_features["test"]["spike"], data.fs)
    representation = width_parent._representation_diagnostics(ordered_features["test"]["membrane"])

    reset_features = parent.collect_features(
        model,
        data,
        cache,
        pspec,
        pconfig,
        reset_state_each_step=True,
    )
    reset_probe = width_parent._fit_feature_probe(
        reset_features,
        data,
        "membrane",
        spec.seed,
        width_parent.RESET_PROBE_NAMESPACE,
    )
    reset_native = {
        split: _evaluate_native(model, loader, device, reset_state_each_step=True)
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
        shuffled_cache = parent._shuffle_cache(data, cache, spec.seed, replicate)
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
        shuffled_probe = width_parent._fit_feature_probe(
            shuffled_features,
            data,
            "membrane",
            spec.seed,
            width_parent.SHUFFLE_PROBE_NAMESPACE,
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
    ordered_progress_mae = float(ordered_membrane["progress_probe_test_sample_balanced_mae_mean"])
    reset_progress_mae = float(reset_probe["progress_probe_test_sample_balanced_mae_mean"])
    shuffle_phase_ba = float(np.mean([float(p["phase_probe_test_ba"]) for p in shuffle_probes]))
    shuffle_progress_mae = float(
        np.mean([float(p["progress_probe_test_sample_balanced_mae_mean"]) for p in shuffle_probes])
    )
    ordered_spearman = float(ordered_membrane["progress_probe_test_spearman_mean"])
    reset_spearman = float(reset_probe["progress_probe_test_spearman_mean"])
    shuffle_spearman = float(
        np.mean([float(p["progress_probe_test_spearman_mean"]) for p in shuffle_probes])
    )
    ordered_violation = float(ordered_membrane["progress_probe_test_monotonic_violation_rate_mean"])
    reset_violation = float(reset_probe["progress_probe_test_monotonic_violation_rate_mean"])
    shuffle_violation = float(
        np.mean([float(p["progress_probe_test_monotonic_violation_rate_mean"]) for p in shuffle_probes])
    )
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
        "H_reset_phase_ba": float(ordered_membrane["phase_probe_test_ba"]) - float(reset_probe["phase_probe_test_ba"]),
        "H_shuffle_phase_ba": float(ordered_membrane["phase_probe_test_ba"]) - shuffle_phase_ba,
        "H_reset_progress_mae": reset_progress_mae - ordered_progress_mae,
        "H_shuffle_progress_mae": shuffle_progress_mae - ordered_progress_mae,
        "H_reset_spearman": ordered_spearman - reset_spearman,
        "H_shuffle_spearman": ordered_spearman - shuffle_spearman,
    }
    trajectory_metrics = {
        split: {
            key.removeprefix(f"progress_probe_{split}_"): value
            for key, value in ordered_membrane.items()
            if key.startswith(f"progress_probe_{split}_")
            and any(
                token in key
                for token in (
                    "sample_balanced_mae",
                    "spearman",
                    "monotonic_violation_rate",
                    "n_gestures",
                )
            )
        }
        for split in ("train", "val", "test")
    }
    counts = parameter_counts(model)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "condition": spec.condition,
        "front_end": spec.front_end,
        "ff_width": spec.ff_width,
        "rsnn_width": RSNN_WIDTH,
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


def _load_direct_payload(repo_root: Path, seed: int) -> tuple[dict[str, object], Path]:
    root = results_dir(repo_root)
    ref_path = direct_reference_path(root, seed)
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing direct RSNN64 reference: {ref_path}")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    if (
        ref.get("experiment_id") != EXPERIMENT_ID
        or ref.get("protocol_version") != PROTOCOL_VERSION
        or ref.get("seed") != seed
        or ref.get("condition") != DIRECT_CONDITION
    ):
        raise ValueError(f"Direct reference identity mismatch: {ref_path}")
    eval_path = repo_root / str(ref["evaluation_path"])
    history_source = repo_root / str(ref["history_path"])
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != width_parent.EXPERIMENT_ID
        or payload.get("protocol_version") != width_parent.PROTOCOL_VERSION
        or payload.get("hidden_width") != RSNN_WIDTH
        or payload.get("seed") != seed
    ):
        raise ValueError(f"Direct source evaluation identity mismatch: {eval_path}")
    return payload, history_source


def _normalized_payload(repo_root: Path, spec: RunSpec) -> tuple[dict[str, object], Path]:
    if spec.condition == DIRECT_CONDITION:
        return _load_direct_payload(repo_root, spec.seed)
    epath = evaluation_path(results_dir(repo_root), spec)
    if not epath.exists():
        raise FileNotFoundError(f"Missing Exp5.3.2.3 evaluation: {epath}")
    payload = json.loads(epath.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("spec") != asdict(spec)
    ):
        raise ValueError(f"Evaluation identity mismatch: {epath}")
    missing = [name for name in EVALUATION_REQUIRED_SECTIONS if name not in payload]
    if missing:
        raise ValueError(f"Evaluation missing required sections {missing}: {epath}")
    return payload, history_path(results_dir(repo_root), spec)


def _condition_metadata(spec: RunSpec, payload: dict[str, object]) -> dict[str, object]:
    if spec.condition == DIRECT_CONDITION:
        counts = payload["provenance"]["parameter_counts"]
        return {
            "condition": spec.condition,
            "front_end": "direct",
            "ff_width": 0,
            "rsnn_width": RSNN_WIDTH,
            "seed": spec.seed,
            "parameter_count": int(counts["trainable_total"]),
            "recurrent_parameter_count": int(counts["recurrent"]),
            "source": "reused_exp5_3_2_2_h64",
        }
    counts = payload["parameter_counts"]
    return {
        "condition": spec.condition,
        "front_end": spec.front_end,
        "ff_width": spec.ff_width,
        "rsnn_width": RSNN_WIDTH,
        "seed": spec.seed,
        "parameter_count": int(counts["trainable_total"]),
        "recurrent_parameter_count": int(counts["recurrent"]),
        "source": "trained_exp5_3_2_3",
    }


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    specs = all_specs()
    if len(specs) != 20 or len({spec.key for spec in specs}) != 20:
        raise RuntimeError("Exp5.3.2.3 comparison matrix must contain exactly 20 unique condition-seed rows")
    for condition in ALL_CONDITIONS:
        if sum(spec.condition == condition for spec in specs) != len(SEEDS):
            raise RuntimeError(f"Condition {condition} does not have exactly five seeds")

    run_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    history_gain_rows: list[dict[str, object]] = []
    activity_rows: list[dict[str, object]] = []
    representation_rows: list[dict[str, object]] = []
    history_parts: list[pd.DataFrame] = []

    for spec in specs:
        payload, hpath = _normalized_payload(repo_root, spec)
        if not hpath.exists():
            raise FileNotFoundError(f"Missing history for {spec.key}: {hpath}")
        common = _condition_metadata(spec, payload)
        ordered_probe = payload["ordered_membrane_probe"]
        row: dict[str, object] = {**common, "best_epoch": payload["best_epoch"]}
        for split in ("train", "val", "test"):
            for metric, value in payload["native"][split].items():
                row[f"native_{split}_{metric}"] = value
            row[f"probe_u_{split}_phase_ba"] = ordered_probe[f"phase_probe_{split}_ba"]
            row[f"probe_u_{split}_progress_mae"] = ordered_probe[f"progress_probe_{split}_mae"]
            row.update(width_parent._probe_trajectory_columns(ordered_probe, split))
        row["probe_phase_train_val_gap"] = row["probe_u_train_phase_ba"] - row["probe_u_val_phase_ba"]
        row["probe_progress_val_train_gap"] = (
            row["probe_val_sample_balanced_mae_mean"] - row["probe_train_sample_balanced_mae_mean"]
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
                "native_test_phase_ba": native_metrics["test"]["phase_balanced_accuracy"],
                "native_test_progress_mae": native_metrics["test"]["progress_sample_balanced_mae"],
                "probe_test_phase_ba": probe["phase_probe_test_ba"],
                "probe_test_progress_mae": probe["progress_probe_test_mae"],
                **width_parent._probe_trajectory_columns(probe, "test"),
            }

        ordered_row = ablation_row("ordered", 0, payload["native"], ordered_probe)
        ablation_rows.append(ordered_row)
        reset_entry = next(item for item in payload["ablations"] if item["ablation"] == "state_reset")
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
            current = ablation_row(
                "temporal_shuffle",
                int(item["replicate"]),
                item["native"],
                item["membrane_probe"],
            )
            ablation_rows.append(current)
            shuffle_rows.append(current)
        if len(shuffle_rows) != SHUFFLE_REPLICATES:
            raise ValueError(f"Expected {SHUFFLE_REPLICATES} shuffles for {spec.key}")

        shuffle_phase = float(np.mean([item["probe_test_phase_ba"] for item in shuffle_rows]))
        shuffle_progress = float(
            np.mean([item["probe_test_sample_balanced_mae_mean"] for item in shuffle_rows])
        )
        shuffle_spearman = float(np.mean([item["probe_test_spearman_mean"] for item in shuffle_rows]))
        history_gain_rows.append(
            {
                **common,
                "ordered_phase_ba": ordered_row["probe_test_phase_ba"],
                "reset_phase_ba": reset_row["probe_test_phase_ba"],
                "shuffle_phase_ba": shuffle_phase,
                "H_reset_phase_ba": ordered_row["probe_test_phase_ba"] - reset_row["probe_test_phase_ba"],
                "H_shuffle_phase_ba": ordered_row["probe_test_phase_ba"] - shuffle_phase,
                "ordered_progress_mae": ordered_row["probe_test_sample_balanced_mae_mean"],
                "reset_progress_mae": reset_row["probe_test_sample_balanced_mae_mean"],
                "shuffle_progress_mae": shuffle_progress,
                "H_reset_progress_mae": reset_row["probe_test_sample_balanced_mae_mean"]
                - ordered_row["probe_test_sample_balanced_mae_mean"],
                "H_shuffle_progress_mae": shuffle_progress - ordered_row["probe_test_sample_balanced_mae_mean"],
                "ordered_spearman": ordered_row["probe_test_spearman_mean"],
                "reset_spearman": reset_row["probe_test_spearman_mean"],
                "shuffle_spearman": shuffle_spearman,
                "H_reset_spearman": ordered_row["probe_test_spearman_mean"] - reset_row["probe_test_spearman_mean"],
                "H_shuffle_spearman": ordered_row["probe_test_spearman_mean"] - shuffle_spearman,
            }
        )

        history = pd.read_csv(hpath)
        history.insert(0, "seed", spec.seed)
        history.insert(0, "ff_width", spec.ff_width)
        history.insert(0, "front_end", spec.front_end)
        history.insert(0, "condition", spec.condition)
        history_parts.append(history)

    baseline_rows: list[dict[str, object]] = []
    local_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        bpath = baseline_path(root, seed)
        if not bpath.exists():
            raise FileNotFoundError(f"Missing Exp5.3.2.3 baseline: {bpath}")
        baseline = json.loads(bpath.read_text(encoding="utf-8"))
        for entry in baseline["baselines"]:
            baseline_rows.append({"seed": seed, **entry})

        reference_path = exp52.local_reference_path(parent.source_results_dir(repo_root), seed)
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing frozen Local reference: {reference_path}")
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
        "paired_deltas": root / "paired_deltas.csv",
        "manifest": root / "manifest.json",
    }
    root.mkdir(parents=True, exist_ok=True)
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(outputs["runs"], index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(outputs["histories"], index=False)
    pd.DataFrame(probe_rows).to_csv(outputs["probe_runs"], index=False)
    pd.DataFrame(ablation_rows).to_csv(outputs["ablation_runs"], index=False)
    pd.DataFrame(history_gain_rows).to_csv(outputs["history_gain_runs"], index=False)
    pd.DataFrame(activity_rows).to_csv(outputs["activity_runs"], index=False)
    pd.DataFrame(representation_rows).to_csv(outputs["representation_runs"], index=False)
    pd.DataFrame(baseline_rows).to_csv(outputs["baseline_runs"], index=False)
    pd.DataFrame(local_rows).to_csv(outputs["local_reference"], index=False)

    direct = runs_df[runs_df["condition"] == DIRECT_CONDITION].set_index("seed")
    delta_rows: list[dict[str, object]] = []
    for condition in TRAINABLE_CONDITIONS:
        candidate = runs_df[runs_df["condition"] == condition].set_index("seed")
        for seed in SEEDS:
            delta_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "delta_phase_ba": candidate.loc[seed, "probe_u_test_phase_ba"]
                    - direct.loc[seed, "probe_u_test_phase_ba"],
                    "delta_progress_mae": candidate.loc[seed, "probe_test_sample_balanced_mae_mean"]
                    - direct.loc[seed, "probe_test_sample_balanced_mae_mean"],
                    "delta_spearman": candidate.loc[seed, "probe_test_spearman_mean"]
                    - direct.loc[seed, "probe_test_spearman_mean"],
                    "delta_violation": candidate.loc[seed, "probe_test_monotonic_violation_rate_mean"]
                    - direct.loc[seed, "probe_test_monotonic_violation_rate_mean"],
                }
            )
    pd.DataFrame(delta_rows).to_csv(outputs["paired_deltas"], index=False)

    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "scientific_question": (
                "whether a pre-recurrent feedforward SNN transform improves the "
                "supervised causal WHEN representation relative to direct RSNN64"
            ),
            "conditions": list(ALL_CONDITIONS),
            "trainable_conditions": list(TRAINABLE_CONDITIONS),
            "seeds": list(SEEDS),
            "expected_train_runs": len(run_specs()),
            "expected_comparison_rows": len(specs),
            "run_order": (
                "condition-major: tasks 0-4 linear64_rsnn64, 5-9 ffsnn64_rsnn64, "
                "10-14 ffsnn128_rsnn64; direct_rsnn64 is identity-validated reuse"
            ),
            "fixed_rsnn_width": RSNN_WIDTH,
            "fixed_objective": asdict(OBJECTIVE),
            "architecture_contract": {
                DIRECT_CONDITION: "WHAT128 -> RSNN64",
                LINEAR_CONTROL_CONDITION: "WHAT128 -> Linear64 -> RSNN64",
                FF64_CONDITION: "WHAT128 -> short-tau FF-SNN64 spikes -> RSNN64",
                FF128_CONDITION: "WHAT128 -> short-tau FF-SNN128 spikes -> RSNN64",
            },
            "linear_control_role": (
                "linear64_rsnn64 has the same trainable weight shapes and paired "
                "initial weights as ffsnn64_rsnn64; the LIF transform is the only difference"
            ),
            "direct_reuse": (
                "Exp5.3.2.2 rsnn_h64 checkpoints/evaluations/histories are reused by "
                "identity-validated references and are never retrained"
            ),
            "checkpoint_selection": (
                "min validation joint objective loss; tie lower progress MAE; tie higher phase BA"
            ),
            "causality_contract": (
                "T_i is used only to construct phase/progress supervision and valid masks; "
                "no model condition receives final duration"
            ),
            "primary_comparisons": [
                "ffsnn64_rsnn64 vs direct_rsnn64",
                "ffsnn64_rsnn64 vs linear64_rsnn64",
                "ffsnn128_rsnn64 vs ffsnn64_rsnn64",
            ],
            "primary_metrics": [
                "U phase balanced accuracy",
                "U progress sample-balanced MAE",
                "per-gesture progress Spearman",
                "monotonic violation rate",
                "H_reset and H_shuffle",
                "train/val/test generalization",
                "spike-domain accessibility",
                "parameter count and firing-rate utilization",
                "effective dimension and PCA90 dimension",
                "paired per-seed deltas against direct RSNN64",
            ],
            "winner_rule": (
                "prefer strict Pareto improvement over direct on mean phase BA and mean progress MAE; "
                "then use history gains, generalization gap, spike accessibility, and parameter cost as diagnostics"
            ),
            "aggregation_policy": (
                "finalizer requires 15 new train/evaluate runs, 5 direct references, and five baseline artifacts; "
                "it only aggregates existing artifacts"
            ),
            "files": {name: path.name for name, path in outputs.items() if name != "manifest"},
        },
    )
    return outputs


def _config_from_args(args: argparse.Namespace) -> Config:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
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
        description="Experiment 5.3.2.3 pre-recurrent FF-SNN vs direct RSNN64"
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
        repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return

    config = _config_from_args(args)
    data = base.prepare_data(config.repo_root)
    if args.command == "prepare-local":
        task_id = int(args.array_task_id)
        if not 0 <= task_id < len(SEEDS):
            raise IndexError(f"prepare-local task {task_id} outside 0..{len(SEEDS)-1}")
        print(prepare_local_seed(SEEDS[task_id], data, config, force=args.force))
        return

    if args.command == "run-one":
        specs = run_specs()
        task_id = int(args.array_task_id)
        if not 0 <= task_id < len(specs):
            raise IndexError(f"run-one task {task_id} outside 0..{len(specs)-1}")
        payload = run_one(specs[task_id], data, config, force=args.force)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
