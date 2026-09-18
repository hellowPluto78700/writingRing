from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_8_1_hidden_quantization_ablation as exp81
from scripts import experiment_8_1_1_two_layer_mt_factorial as exp811
from scripts import experiment_10_0_airborne_motion_ablation as exp10
from scripts import experiment_10_2_1_l2_membrane_weighted as exp1021
from scripts import experiment_10_2_2_valid_window_probes as exp1022


EXPERIMENT_ID = "experiment_11_0_rsnn_history_internalization"
PROTOCOL_VERSION = "d1_l1mem2_rsnn_fusion_internalization_v1"

VARIANT = exp1021.VARIANT
ROTATION = exp1021.ROTATION
MODEL_SEEDS = exp1021.MODEL_SEEDS
WIDTH = exp1021.WIDTH
L1_MEM_SHIFT = exp1021.L1_MEM_SHIFT
L1_SYN_SHIFTS = exp1021.SYN_SHIFTS

L1_INIT_DYNAMICS_ONLY = "dynamics_only"
L1_INIT_PRETRAINED = "pretrained_input"
L1_INIT_MODES = (L1_INIT_DYNAMICS_ONLY, L1_INIT_PRETRAINED)

TOPOLOGY_FF = "ff"
TOPOLOGY_DIAGONAL = "diagonal"
TOPOLOGY_DENSE = "dense"
TOPOLOGIES = (TOPOLOGY_FF, TOPOLOGY_DIAGONAL, TOPOLOGY_DENSE)

CONTEXT_TAU_MS = exp1021.tau_mem_ms_from_shift(2)
FUSION_TAU_MS = CONTEXT_TAU_MS
COMMUNICATION_CAP = 1
GRAD_CLIP_NORM = 1.0

MAX_EPOCHS = exp1021.MAX_EPOCHS
MIN_EPOCHS = exp1021.MIN_EPOCHS
PATIENCE = exp1021.PATIENCE
BATCH_SIZE = exp1021.BATCH_SIZE
SPLITS = exp1021.SPLITS

LAYERS = ("l1", "rsnn", "fusion")
PROBE_ANALOG_STATE = "pre_reset"
PROBE_COMMUNICATION_STATE = "communication"
SUPPORTS = exp1022.SUPPORTS
ANALOG_AGGREGATIONS = exp1022.ANALOG_AGGREGATIONS
COMMUNICATION_AGGREGATIONS = exp1022.COMMUNICATION_AGGREGATIONS
C_GRID = exp1022.C_GRID
PROBE_MAX_ITER = exp1022.PROBE_MAX_ITER

EXPECTED_RUNS = len(L1_INIT_MODES) * len(TOPOLOGIES) * len(MODEL_SEEDS)
EXPECTED_PROBES_PER_RUN = len(LAYERS) * (
    len(SUPPORTS) * len(ANALOG_AGGREGATIONS)
    + len(SUPPORTS) * len(COMMUNICATION_AGGREGATIONS)
)


@dataclass(frozen=True)
class RunSpec:
    l1_init: str
    topology: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.l1_init}__{self.topology}__seed{self.seed}"

    @property
    def variant(self) -> str:
        return VARIANT

    @property
    def rotation(self) -> int:
        return ROTATION


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    threads: int = 1
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp1021.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / EXPERIMENT_ID
        / PROTOCOL_VERSION
    )


def source_results_dir(repo_root: Path) -> Path:
    return exp1021.results_dir(repo_root)


def source_probe_results_dir(repo_root: Path) -> Path:
    return exp1022.results_dir(repo_root)


def source_config(config: Config) -> exp1021.Config:
    return exp1021.Config(
        repo_root=config.repo_root,
        results_dir=source_results_dir(config.repo_root),
        device=config.device,
        threads=config.threads,
        batch_size=config.batch_size,
        max_epochs=exp1021.MAX_EPOCHS,
    )


def source_spec(seed: int) -> exp1021.RunSpec:
    return exp1021.RunSpec(exp1021.CODING_BINARY, 1, seed)


def source_checkpoint_path(config: Config, seed: int) -> Path:
    return exp1021._run_artifacts(source_config(config), source_spec(seed))["checkpoint"]


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(l1_init, topology, seed)
        for l1_init in L1_INIT_MODES
        for topology in TOPOLOGIES
        for seed in MODEL_SEEDS
    ]


def validate_spec(spec: RunSpec) -> None:
    if spec.l1_init not in L1_INIT_MODES:
        raise ValueError(spec.l1_init)
    if spec.topology not in TOPOLOGIES:
        raise ValueError(spec.topology)
    if spec.seed not in MODEL_SEEDS:
        raise ValueError(spec.seed)


def decay_from_tau_ms(tau_ms: float, fs: float = exp72.EXPECTED_FS) -> float:
    if tau_ms <= 0:
        raise ValueError("tau_ms must be positive")
    return math.exp(-(1000.0 / float(fs)) / float(tau_ms))


def recurrent_param_count(topology: str) -> int:
    if topology == TOPOLOGY_FF:
        return 0
    if topology == TOPOLOGY_DIAGONAL:
        return WIDTH
    if topology == TOPOLOGY_DENSE:
        return WIDTH * WIDTH
    raise ValueError(topology)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _run_artifacts(config: Config, spec: RunSpec) -> dict[str, Path]:
    return {
        "checkpoint": _path(config.results_dir, "checkpoints", spec.key, ".pt"),
        "history": _path(config.results_dir, "histories", spec.key, ".csv"),
        "evaluation": _path(config.results_dir, "evaluations", spec.key, ".json"),
        "probes": _path(config.results_dir, "probe_evaluations", spec.key, ".csv"),
    }


def _load_manifest(path: Path, experiment_id: str, protocol_version: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": experiment_id,
        "protocol_version": protocol_version,
        "status": "PASS",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Manifest mismatch at {path} for {key}: "
                f"expected {value!r}, got {manifest.get(key)!r}"
            )
    return manifest


def _prepare_data(
    config: Config,
) -> tuple[exp3.Data, dict[str, pd.DataFrame], pd.DataFrame]:
    return exp1021._prepare_data(source_config(config))


def prepare_all(config: Config) -> dict[str, Any]:
    source_manifest = _load_manifest(
        source_results_dir(config.repo_root) / "manifest.json",
        exp1021.EXPERIMENT_ID,
        exp1021.PROTOCOL_VERSION,
    )
    probe_manifest = _load_manifest(
        source_probe_results_dir(config.repo_root) / "manifest.json",
        exp1022.EXPERIMENT_ID,
        exp1022.PROTOCOL_VERSION,
    )

    data, frames, split_manifest = _prepare_data(config)
    split_hashes = exp1021._split_hashes(frames)

    source_checkpoints: dict[str, str] = {}
    for seed in MODEL_SEEDS:
        path = source_checkpoint_path(config, seed)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing seed-matched Exp10.2.1 source checkpoint: {path}"
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        expected_spec = asdict(source_spec(seed))
        if checkpoint.get("experiment_id") != exp1021.EXPERIMENT_ID:
            raise RuntimeError(f"{path}: source experiment mismatch")
        if checkpoint.get("protocol_version") != exp1021.PROTOCOL_VERSION:
            raise RuntimeError(f"{path}: source protocol mismatch")
        if checkpoint.get("spec") != expected_spec:
            raise RuntimeError(f"{path}: source spec mismatch")
        if checkpoint.get("split_sample_hashes") != split_hashes:
            raise RuntimeError(f"{path}: split geometry mismatch")
        source_checkpoints[str(seed)] = str(path.relative_to(config.repo_root))

    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Can recurrent SNN state internalize the temporal ordering exposed by "
            "external Fixed250 probes so that one time-shared 128x12 weight matrix "
            "can decode a history-aware fusion representation?"
        ),
        "dataset": VARIANT,
        "rotation": ROTATION,
        "source_exp10_2_1_manifest": source_manifest,
        "source_exp10_2_2_manifest": probe_manifest,
        "source_checkpoints": source_checkpoints,
        "source_condition": "binary__l1mem2__l2mem1, seed matched",
        "split_sample_hashes": split_hashes,
        "split_users": {
            split: sorted(frames[split].user.astype(str).unique().tolist())
            for split in SPLITS
        },
        "split_samples": {split: int(len(frames[split])) for split in SPLITS},
        "labels": list(data.labels),
        "architecture": "30 -> L1(128) -> RSNN(128) -> Fusion(128) -> Linear(12)",
        "l1_init_modes": list(L1_INIT_MODES),
        "l1_pretrained_contract": (
            "pretrained_input copies only hidden_linears.0.weight from the "
            "seed-matched Exp10.2.1 binary/l1mem2/l2mem1 best checkpoint; "
            "the copied weight remains trainable"
        ),
        "l1_mem_shift": L1_MEM_SHIFT,
        "l1_tau_mem_ms": exp1021.tau_mem_ms_from_shift(L1_MEM_SHIFT, data.fs),
        "l1_synaptic_shifts": list(L1_SYN_SHIFTS),
        "rsnn_width": WIDTH,
        "rsnn_topologies": list(TOPOLOGIES),
        "rsnn_tau_syn_ms": CONTEXT_TAU_MS,
        "rsnn_tau_mem_ms": CONTEXT_TAU_MS,
        "fusion_width": WIDTH,
        "fusion_recurrent": False,
        "fusion_tau_syn_ms": FUSION_TAU_MS,
        "fusion_tau_mem_ms": FUSION_TAU_MS,
        "fusion_inputs": ["L1 binary local spikes", "RSNN binary context spikes"],
        "objective": "valid-length mean time-shared WCCE only",
        "output": "single shared bias-free 128x12 Linear",
        "communication": "binary spikes in L1, RSNN, and Fusion",
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "supports": list(SUPPORTS),
        "probe_layers": list(LAYERS),
        "probe_states": [PROBE_ANALOG_STATE, PROBE_COMMUNICATION_STATE],
        "probe_count_per_run": EXPECTED_PROBES_PER_RUN,
        "model_seeds": list(MODEL_SEEDS),
        "expected_runs": EXPECTED_RUNS,
        "statistical_scope": (
            "one locked cross-user split; seeds 11/23/37 are paired optimization "
            "replicates, not independent user splits"
        ),
        "primary_mechanism_endpoint": (
            "communication Fixed250-minus-whole probe gap should shrink from L1 "
            "to Fusion while Fusion whole/native BA rises"
        ),
    }
    config.results_dir.mkdir(parents=True, exist_ok=True)
    _save_json(config.results_dir / "audit.json", audit)
    split_manifest.drop(columns=["pi", "si"], errors="ignore").to_csv(
        config.results_dir / "rotation0_sample_manifest.csv", index=False
    )
    return audit


class ZeroRecurrent(nn.Module):
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(spikes)


class DiagonalRecurrent(nn.Module):
    """Trainable self recurrence with no cross-neuron mixing."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(WIDTH))

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        return spikes * self.weight.to(dtype=spikes.dtype)


class Exp110Net(nn.Module):
    """Local L1 -> recurrent context -> feed-forward fusion -> shared Linear."""

    def __init__(
        self,
        spec: RunSpec,
        n_classes: int,
        fs: float,
    ) -> None:
        super().__init__()
        validate_spec(spec)
        self.spec = spec
        self.n_classes = int(n_classes)
        self.fs = float(fs)

        self.l1_input = nn.Linear(exp72.EXPECTED_CHANNELS, WIDTH, bias=False)
        self.rsnn_input = nn.Linear(WIDTH, WIDTH, bias=False)
        self.fusion_local = nn.Linear(WIDTH, WIDTH, bias=False)
        self.fusion_context = nn.Linear(WIDTH, WIDTH, bias=False)
        self.output_linear = nn.Linear(WIDTH, n_classes, bias=False)

        if spec.topology == TOPOLOGY_FF:
            self.recurrent: nn.Module = ZeroRecurrent()
        elif spec.topology == TOPOLOGY_DIAGONAL:
            self.recurrent = DiagonalRecurrent()
        elif spec.topology == TOPOLOGY_DENSE:
            self.recurrent = nn.Linear(WIDTH, WIDTH, bias=False)
        else:
            raise ValueError(spec.topology)

        self.l1_lif = exp401.MacroMultiSpikeLIF(
            beta=exp1021.beta_from_mem_shift(L1_MEM_SHIFT, fs),
            threshold=float(exp73.THRESHOLD),
            max_spikes_per_dt=COMMUNICATION_CAP,
            surrogate_slope=exp72.SURROGATE_SLOPE,
        )
        context_beta = decay_from_tau_ms(CONTEXT_TAU_MS, fs)
        fusion_beta = decay_from_tau_ms(FUSION_TAU_MS, fs)
        self.rsnn_lif = exp401.MacroMultiSpikeLIF(
            beta=context_beta,
            threshold=float(exp73.THRESHOLD),
            max_spikes_per_dt=COMMUNICATION_CAP,
            surrogate_slope=exp72.SURROGATE_SLOPE,
        )
        self.fusion_lif = exp401.MacroMultiSpikeLIF(
            beta=fusion_beta,
            threshold=float(exp73.THRESHOLD),
            max_spikes_per_dt=COMMUNICATION_CAP,
            surrogate_slope=exp72.SURROGATE_SLOPE,
        )

        self.register_buffer("l1_alpha", exp811._alpha_vector("binary"))
        self.register_buffer(
            "rsnn_alpha",
            torch.tensor(decay_from_tau_ms(CONTEXT_TAU_MS, fs), dtype=torch.float32),
        )
        self.register_buffer(
            "fusion_alpha",
            torch.tensor(decay_from_tau_ms(FUSION_TAU_MS, fs), dtype=torch.float32),
        )

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != exp72.EXPECTED_CHANNELS:
            raise ValueError(channels)

        syn_l1 = torch.zeros(batch, WIDTH, device=x.device, dtype=x.dtype)
        mem_l1 = torch.zeros_like(syn_l1)
        syn_rsnn = torch.zeros_like(syn_l1)
        mem_rsnn = torch.zeros_like(syn_l1)
        prev_rsnn = torch.zeros_like(syn_l1)
        syn_fusion = torch.zeros_like(syn_l1)
        mem_fusion = torch.zeros_like(syn_l1)

        states: dict[str, dict[str, list[torch.Tensor]]] = {
            layer: {
                "syn_current": [],
                "pre_reset": [],
                "spike": [],
                "post_reset": [],
            }
            for layer in LAYERS
        }
        rsnn_external: list[torch.Tensor] = []
        rsnn_recurrent: list[torch.Tensor] = []
        evidence: list[torch.Tensor] = []

        for timestep in range(steps):
            syn_l1 = self.l1_alpha * syn_l1 + self.l1_input(x[:, timestep])
            z_t, post_l1, pre_l1 = self.l1_lif(syn_l1, mem_l1)
            mem_l1 = post_l1

            external_t = self.rsnn_input(z_t)
            recurrent_t = self.recurrent(prev_rsnn)
            syn_rsnn = self.rsnn_alpha * syn_rsnn + external_t + recurrent_t
            r_t, post_rsnn, pre_rsnn = self.rsnn_lif(syn_rsnn, mem_rsnn)
            mem_rsnn = post_rsnn
            prev_rsnn = r_t

            fusion_drive = self.fusion_local(z_t) + self.fusion_context(r_t)
            syn_fusion = self.fusion_alpha * syn_fusion + fusion_drive
            q_t, post_fusion, pre_fusion = self.fusion_lif(
                syn_fusion, mem_fusion
            )
            mem_fusion = post_fusion

            for layer, syn, pre, spike, post in (
                ("l1", syn_l1, pre_l1, z_t, post_l1),
                ("rsnn", syn_rsnn, pre_rsnn, r_t, post_rsnn),
                ("fusion", syn_fusion, pre_fusion, q_t, post_fusion),
            ):
                states[layer]["syn_current"].append(syn)
                states[layer]["pre_reset"].append(pre)
                states[layer]["spike"].append(spike)
                states[layer]["post_reset"].append(post)

            rsnn_external.append(external_t)
            rsnn_recurrent.append(recurrent_t)
            evidence.append(self.output_linear(q_t))

        hidden = {
            layer: {
                state: torch.stack(values, dim=1)
                for state, values in layer_states.items()
            }
            for layer, layer_states in states.items()
        }
        return {
            "hidden": hidden,
            "rsnn_external_input": torch.stack(rsnn_external, dim=1),
            "rsnn_recurrent_input": torch.stack(rsnn_recurrent, dim=1),
            "fusion_evidence": torch.stack(evidence, dim=1),
        }


def _reset_linear(module: nn.Linear, seed: int, role: str) -> None:
    exp3.seed_all(exp73._e2e_pair_seed(seed, role))
    module.reset_parameters()


def _initialize_paired(model: Exp110Net, spec: RunSpec) -> None:
    """Match all shared initial weights across the six conditions of one seed."""
    for module, role in (
        (model.l1_input, "exp11_l1_input_init"),
        (model.rsnn_input, "exp11_rsnn_input_init"),
        (model.fusion_local, "exp11_fusion_local_init"),
        (model.fusion_context, "exp11_fusion_context_init"),
        (model.output_linear, "exp11_output_init"),
    ):
        _reset_linear(module, spec.seed, role)

    exp3.seed_all(exp73._e2e_pair_seed(spec.seed, "exp11_recurrent_init"))
    paired_dense = nn.Linear(WIDTH, WIDTH, bias=False)
    paired_weight = paired_dense.weight.detach().clone()
    if spec.topology == TOPOLOGY_DENSE:
        assert isinstance(model.recurrent, nn.Linear)
        with torch.no_grad():
            model.recurrent.weight.copy_(paired_weight)
    elif spec.topology == TOPOLOGY_DIAGONAL:
        assert isinstance(model.recurrent, DiagonalRecurrent)
        with torch.no_grad():
            model.recurrent.weight.copy_(torch.diagonal(paired_weight))


def _load_pretrained_l1(
    model: Exp110Net,
    spec: RunSpec,
    frames: Mapping[str, pd.DataFrame],
    config: Config,
) -> dict[str, Any] | None:
    if spec.l1_init == L1_INIT_DYNAMICS_ONLY:
        return None

    path = source_checkpoint_path(config, spec.seed)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_spec = asdict(source_spec(spec.seed))
    if checkpoint.get("experiment_id") != exp1021.EXPERIMENT_ID:
        raise RuntimeError(f"{path}: source experiment mismatch")
    if checkpoint.get("protocol_version") != exp1021.PROTOCOL_VERSION:
        raise RuntimeError(f"{path}: source protocol mismatch")
    if checkpoint.get("spec") != expected_spec:
        raise RuntimeError(f"{path}: source spec mismatch")
    split_hashes = exp1021._split_hashes(frames)
    if checkpoint.get("split_sample_hashes") != split_hashes:
        raise RuntimeError(f"{path}: source split geometry mismatch")

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"{path}: missing model_state_dict")
    source_weight = state.get("hidden_linears.0.weight")
    if source_weight is None:
        raise RuntimeError(f"{path}: missing hidden_linears.0.weight")
    if tuple(source_weight.shape) != tuple(model.l1_input.weight.shape):
        raise RuntimeError(
            f"{path}: L1 weight shape {tuple(source_weight.shape)} does not match "
            f"{tuple(model.l1_input.weight.shape)}"
        )
    with torch.no_grad():
        model.l1_input.weight.copy_(
            source_weight.to(dtype=model.l1_input.weight.dtype)
        )
    if not model.l1_input.weight.requires_grad:
        raise RuntimeError("Pretrained L1 input weight must remain trainable")
    return {
        "checkpoint": str(path.relative_to(config.repo_root)),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "source_spec": expected_spec,
    }


def _valid_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return exp1021._valid_mean(values, lengths)


def _native_scores(
    model: Exp110Net,
    X: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    evidence = model.forward_trajectory(X)["fusion_evidence"]
    return _valid_mean(evidence, lengths)


def _window_scores(model: Exp110Net, X: torch.Tensor) -> torch.Tensor:
    evidence = model.forward_trajectory(X)["fusion_evidence"]
    return evidence.mean(dim=1)


def _evaluate_scores(
    model: Exp110Net,
    loader: Iterable,
    device: torch.device,
    support: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0

    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            Xd = X.to(device=device, dtype=torch.float32)
            yd = y.to(device)
            ld = lengths.to(device=device, dtype=torch.long)
            if support == exp1022.SUPPORT_VALID:
                scores = _native_scores(model, Xd, ld)
            elif support == exp1022.SUPPORT_WINDOW:
                scores = _window_scores(model, Xd)
            else:
                raise ValueError(support)
            loss = F.cross_entropy(scores, yd)
            ys.append(y.numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    metrics = exp10._classification_metrics(y_true, y_pred)
    metrics["objective_loss"] = loss_sum / max(n_total, 1)
    return metrics, y_true, y_pred


def _evaluate_lif_transfer(
    model: Exp110Net,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            Xd = X.to(device=device, dtype=torch.float32)
            ld = lengths.to(device=device, dtype=torch.long)
            evidence = model.forward_trajectory(Xd)["fusion_evidence"]
            spikes = exp81._output_lif_spikes(evidence)
            mask = exp1021._valid_mask(ld, spikes.shape[1]).to(
                spikes.dtype
            ).unsqueeze(-1)
            scores = (spikes * mask).sum(dim=1)
            ys.append(y.numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
    return exp10._classification_metrics(
        np.concatenate(ys), np.concatenate(preds)
    )


def probe_names() -> tuple[str, ...]:
    names: list[str] = []
    for layer in LAYERS:
        for support in SUPPORTS:
            for aggregation in ANALOG_AGGREGATIONS:
                names.append(
                    f"{layer}__{PROBE_ANALOG_STATE}__{support}__{aggregation}"
                )
        for support in SUPPORTS:
            for aggregation in COMMUNICATION_AGGREGATIONS:
                names.append(
                    f"{layer}__{PROBE_COMMUNICATION_STATE}__"
                    f"{support}__{aggregation}"
                )
    if len(names) != EXPECTED_PROBES_PER_RUN:
        raise RuntimeError(
            f"Expected {EXPECTED_PROBES_PER_RUN} probes, got {len(names)}"
        )
    return tuple(names)


def _descriptor(name: str) -> tuple[str, str, str, str]:
    return tuple(name.split("__", 3))  # type: ignore[return-value]


def _activity_stats(
    values: torch.Tensor,
    lengths: np.ndarray,
    fs: float,
) -> dict[str, float]:
    lengths_t = torch.as_tensor(lengths, dtype=torch.long)
    mask = exp1021._valid_mask(lengths_t, values.shape[1]).to(values.dtype)
    valid = values * mask.unsqueeze(-1)
    valid_neuron_steps = max(
        float(mask.sum().item()) * float(values.shape[2]), 1.0
    )
    valid_events = float(valid.sum().item())
    total_events = float(values.sum().item())
    tail_events = max(total_events - valid_events, 0.0)
    per_neuron = valid.sum(dim=(0, 1))
    return {
        "mean_events_per_neuron_step": valid_events / valid_neuron_steps,
        "mean_events_per_neuron_s": (
            valid_events / valid_neuron_steps
        ) * float(fs),
        "nonzero_fraction": float(
            (((values > 0).to(values.dtype) * mask.unsqueeze(-1)).sum().item())
            / valid_neuron_steps
        ),
        "dead_neuron_fraction": float(
            (per_neuron == 0).to(torch.float32).mean().item()
        ),
        "post_valid_event_fraction": tail_events / max(total_events, 1.0),
    }


def _recurrent_input_diagnostics(
    external: torch.Tensor,
    recurrent: torch.Tensor,
    lengths: np.ndarray,
    model: Exp110Net,
) -> dict[str, float]:
    lengths_t = torch.as_tensor(lengths, dtype=torch.long)
    mask = exp1021._valid_mask(lengths_t, external.shape[1])
    external_valid = external[mask]
    recurrent_valid = recurrent[mask]
    external_mean_abs = (
        float(external_valid.abs().mean().item())
        if external_valid.numel()
        else 0.0
    )
    recurrent_mean_abs = (
        float(recurrent_valid.abs().mean().item())
        if recurrent_valid.numel()
        else 0.0
    )
    if isinstance(model.recurrent, nn.Linear):
        weight_norm = float(model.recurrent.weight.detach().norm().item())
    elif isinstance(model.recurrent, DiagonalRecurrent):
        weight_norm = float(model.recurrent.weight.detach().norm().item())
    else:
        weight_norm = 0.0
    return {
        "mean_abs_external_input": external_mean_abs,
        "mean_abs_recurrent_input": recurrent_mean_abs,
        "recurrent_to_external_abs_ratio": (
            recurrent_mean_abs / max(external_mean_abs, 1e-12)
        ),
        "recurrent_weight_norm": weight_norm,
    }


def _collect_probe_features_and_activity(
    model: Exp110Net,
    data: exp3.Data,
    spec: RunSpec,
    config: Config,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, dict[str, float]],
    dict[str, float],
]:
    device = torch.device(config.device)
    loaders = exp73._raw_loaders(
        data, spec.seed, config.batch_size, False
    )
    names = probe_names()
    feature_parts: dict[str, dict[str, list[np.ndarray]]] = {
        split: {name: [] for name in names}
        for split in SPLITS
    }
    label_parts: dict[str, list[np.ndarray]] = {
        split: [] for split in SPLITS
    }

    test_spikes: dict[str, list[torch.Tensor]] = {
        layer: [] for layer in LAYERS
    }
    test_external: list[torch.Tensor] = []
    test_recurrent: list[torch.Tensor] = []
    test_lengths: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            for X, y, lengths in loaders[split]:
                Xd = X.to(device=device, dtype=torch.float32)
                ld = lengths.to(device=device, dtype=torch.long)
                trajectory = model.forward_trajectory(Xd)
                hidden = trajectory["hidden"]

                for layer in LAYERS:
                    pre = hidden[layer]["pre_reset"]
                    for support in SUPPORTS:
                        for aggregation in ANALOG_AGGREGATIONS:
                            name = (
                                f"{layer}__{PROBE_ANALOG_STATE}__"
                                f"{support}__{aggregation}"
                            )
                            aggregated = exp1022._aggregate(
                                pre,
                                ld,
                                support,
                                aggregation,
                                data.bin_steps,
                            )
                            feature_parts[split][name].append(
                                aggregated.cpu().numpy().astype(
                                    np.float32, copy=False
                                )
                            )

                    communication = hidden[layer]["spike"]
                    for support in SUPPORTS:
                        for aggregation in COMMUNICATION_AGGREGATIONS:
                            name = (
                                f"{layer}__{PROBE_COMMUNICATION_STATE}__"
                                f"{support}__{aggregation}"
                            )
                            aggregated = exp1022._aggregate(
                                communication,
                                ld,
                                support,
                                aggregation,
                                data.bin_steps,
                            )
                            feature_parts[split][name].append(
                                aggregated.cpu().numpy().astype(
                                    np.float32, copy=False
                                )
                            )
                    if split == "test":
                        test_spikes[layer].append(communication.cpu())

                if split == "test":
                    test_external.append(
                        trajectory["rsnn_external_input"].cpu()
                    )
                    test_recurrent.append(
                        trajectory["rsnn_recurrent_input"].cpu()
                    )
                    test_lengths.append(
                        lengths.numpy().astype(np.int64, copy=False)
                    )
                label_parts[split].append(
                    y.numpy().astype(np.int64, copy=False)
                )

    features = {
        split: {
            name: np.concatenate(parts, axis=0)
            for name, parts in feature_parts[split].items()
        }
        for split in SPLITS
    }
    labels = {
        split: np.concatenate(parts, axis=0)
        for split, parts in label_parts.items()
    }
    lengths_np = np.concatenate(test_lengths, axis=0)
    activity = {
        layer: _activity_stats(
            torch.cat(test_spikes[layer], dim=0),
            lengths_np,
            data.fs,
        )
        for layer in LAYERS
    }
    recurrent_diag = _recurrent_input_diagnostics(
        torch.cat(test_external, dim=0),
        torch.cat(test_recurrent, dim=0),
        lengths_np,
        model,
    )
    return features, labels, activity, recurrent_diag


def _probe_seed(spec: RunSpec, name: str) -> int:
    return int(
        exp3.dseed(
            spec.seed,
            EXPERIMENT_ID,
            "probe",
            spec.l1_init,
            spec.topology,
            name,
        )
    )


def _fit_probes(
    spec: RunSpec,
    features: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for name in probe_names():
        train_x = features["train"][name].astype(np.float64, copy=False)
        val_x = features["val"][name].astype(np.float64, copy=False)
        test_x = features["test"][name].astype(np.float64, copy=False)

        scaler = StandardScaler().fit(train_x)
        transformed = {
            "train": scaler.transform(train_x),
            "val": scaler.transform(val_x),
            "test": scaler.transform(test_x),
        }

        random_state = _probe_seed(spec, name)
        best: tuple[float, float, LogisticRegression] | None = None
        candidates: list[dict[str, float]] = []
        for C in C_GRID:
            classifier = LogisticRegression(
                C=C,
                max_iter=PROBE_MAX_ITER,
                solver="lbfgs",
                random_state=random_state,
                fit_intercept=True,
            ).fit(transformed["train"], labels["train"])
            val_pred = classifier.predict(transformed["val"])
            val_metrics = exp10._classification_metrics(
                labels["val"], val_pred
            )
            val_ba = float(val_metrics["balanced_accuracy"])
            candidates.append(
                {
                    "C": float(C),
                    "val_balanced_accuracy": val_ba,
                }
            )
            if best is None or val_ba > best[0] + 1e-12:
                best = (val_ba, float(C), classifier)

        if best is None:
            raise RuntimeError(f"No probe candidate selected for {spec.key}/{name}")

        selected_val_ba, selected_C, classifier = best
        metrics = {
            split: exp10._classification_metrics(
                labels[split],
                classifier.predict(transformed[split]),
            )
            for split in SPLITS
        }
        layer, state, support, aggregation = _descriptor(name)
        rows.append(
            {
                "probe": name,
                "l1_init": spec.l1_init,
                "topology": spec.topology,
                "seed": spec.seed,
                "layer": layer,
                "state": state,
                "support": support,
                "aggregation": aggregation,
                "feature_dim": int(train_x.shape[1]),
                "selected_C": selected_C,
                "selected_val_balanced_accuracy": selected_val_ba,
                "candidate_validation_json": json.dumps(candidates),
                "train_accuracy": metrics["train"]["accuracy"],
                "train_balanced_accuracy": metrics["train"]["balanced_accuracy"],
                "train_macro_f1": metrics["train"]["macro_f1"],
                "val_accuracy": metrics["val"]["accuracy"],
                "val_balanced_accuracy": metrics["val"]["balanced_accuracy"],
                "val_macro_f1": metrics["val"]["macro_f1"],
                "test_accuracy": metrics["test"]["accuracy"],
                "test_balanced_accuracy": metrics["test"]["balanced_accuracy"],
                "test_macro_f1": metrics["test"]["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def _probe_value(frame: pd.DataFrame, name: str) -> float:
    row = frame[frame.probe == name]
    if len(row) != 1:
        raise RuntimeError(f"Expected one row for {name}, got {len(row)}")
    return float(row.iloc[0].test_balanced_accuracy)


def run_one(
    spec: RunSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_spec(spec)
    artifacts = _run_artifacts(config, spec)
    if not force and all(path.exists() for path in artifacts.values()):
        return json.loads(artifacts["evaluation"].read_text(encoding="utf-8"))

    data, frames, _ = _prepare_data(config)
    split_hashes = exp1021._split_hashes(frames)
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)

    model = Exp110Net(spec, len(data.labels), data.fs)
    _initialize_paired(model, spec)
    source_l1 = _load_pretrained_l1(model, spec, frames, config)
    model = model.to(device)

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

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_total = 0
        for X, y, lengths in train_loader:
            Xd = X.to(device=device, dtype=torch.float32)
            yd = y.to(device)
            ld = lengths.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            scores = _native_scores(model, Xd, ld)
            loss = F.cross_entropy(scores, yd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP_NORM
            )
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics, _, _ = _evaluate_scores(
            model,
            eval_loaders["train"],
            device,
            exp1022.SUPPORT_VALID,
        )
        val_metrics, _, _ = _evaluate_scores(
            model,
            eval_loaders["val"],
            device,
            exp1022.SUPPORT_VALID,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )

        if exp73._checkpoint_improved(
            val_metrics, best_ba, best_loss
        ):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if (
            epoch >= MIN_EPOCHS
            and best_epoch > 0
            and epoch - best_epoch >= PATIENCE
        ):
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")

    artifacts["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "split_sample_hashes": split_hashes,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "source_l1": source_l1,
            "model_state_dict": best_state,
        },
        artifacts["checkpoint"],
    )
    artifacts["history"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(artifacts["history"], index=False)

    model.load_state_dict(best_state, strict=True)
    native_metrics: dict[str, dict[str, float]] = {}
    window_metrics: dict[str, dict[str, float]] = {}
    lif_metrics: dict[str, dict[str, float]] = {}
    native_labels: dict[str, np.ndarray] = {}

    for split in SPLITS:
        valid_metrics, y_true, _ = _evaluate_scores(
            model,
            eval_loaders[split],
            device,
            exp1022.SUPPORT_VALID,
        )
        full_metrics, _, _ = _evaluate_scores(
            model,
            eval_loaders[split],
            device,
            exp1022.SUPPORT_WINDOW,
        )
        native_metrics[split] = valid_metrics
        window_metrics[split] = full_metrics
        native_labels[split] = y_true
        lif_metrics[split] = _evaluate_lif_transfer(
            model, eval_loaders[split], device
        )

    features, probe_labels, activity, recurrent_diag = (
        _collect_probe_features_and_activity(
            model, data, spec, config
        )
    )
    for split in SPLITS:
        if not np.array_equal(probe_labels[split], native_labels[split]):
            raise RuntimeError(
                f"{spec.key}/{split}: probe label order mismatch"
            )

    probes = _fit_probes(spec, features, probe_labels)
    artifacts["probes"].parent.mkdir(parents=True, exist_ok=True)
    probes.to_csv(artifacts["probes"], index=False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "dataset": VARIANT,
            "rotation": ROTATION,
            "architecture": (
                "30->L1(128)->RSNN(128)->Fusion(128)->Linear(12)"
            ),
            "l1_init": spec.l1_init,
            "l1_mem_shift": L1_MEM_SHIFT,
            "l1_tau_mem_ms": exp1021.tau_mem_ms_from_shift(
                L1_MEM_SHIFT, data.fs
            ),
            "l1_synaptic_shifts": list(L1_SYN_SHIFTS),
            "rsnn_topology": spec.topology,
            "rsnn_recurrent_param_count": recurrent_param_count(
                spec.topology
            ),
            "rsnn_tau_syn_ms": CONTEXT_TAU_MS,
            "rsnn_tau_mem_ms": CONTEXT_TAU_MS,
            "fusion_tau_syn_ms": FUSION_TAU_MS,
            "fusion_tau_mem_ms": FUSION_TAU_MS,
            "fusion_recurrent": False,
            "objective": "valid-mean time-shared WCCE",
            "readout": "single bias-free 128x12 Linear",
            "gradient_clip_norm": GRAD_CLIP_NORM,
        },
        "source_l1": source_l1,
        "split_sample_hashes": split_hashes,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native_metrics,
        "window_metrics": window_metrics,
        "lif_transfer_metrics": lif_metrics,
        "activity": activity,
        "recurrent_diagnostics": recurrent_diag,
        "probe_count": int(len(probes)),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameter_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }
    _save_json(artifacts["evaluation"], payload)
    return payload


def _run_row(
    payload: Mapping[str, Any],
    probes: pd.DataFrame,
) -> dict[str, Any]:
    spec = payload["spec"]
    row: dict[str, Any] = {
        "l1_init": spec["l1_init"],
        "topology": spec["topology"],
        "seed": int(spec["seed"]),
        "best_epoch": int(payload["best_epoch"]),
        "stopped_epoch": int(payload["stopped_epoch"]),
        "native_train_ba": float(
            payload["native_metrics"]["train"]["balanced_accuracy"]
        ),
        "native_val_ba": float(
            payload["native_metrics"]["val"]["balanced_accuracy"]
        ),
        "native_test_ba": float(
            payload["native_metrics"]["test"]["balanced_accuracy"]
        ),
        "window_test_ba": float(
            payload["window_metrics"]["test"]["balanced_accuracy"]
        ),
        "output_lif_test_ba": float(
            payload["lif_transfer_metrics"]["test"]["balanced_accuracy"]
        ),
        "parameter_count": int(payload["parameter_count"]),
        "trainable_parameter_count": int(
            payload["trainable_parameter_count"]
        ),
        "recurrent_param_count": recurrent_param_count(spec["topology"]),
        **{
            key: float(value)
            for key, value in payload["recurrent_diagnostics"].items()
        },
    }

    for layer in LAYERS:
        valid_fixed = _probe_value(
            probes,
            f"{layer}__communication__valid__fixed250_count",
        )
        valid_whole = _probe_value(
            probes,
            f"{layer}__communication__valid__whole_count",
        )
        window_fixed = _probe_value(
            probes,
            f"{layer}__communication__window__fixed250_count",
        )
        window_whole = _probe_value(
            probes,
            f"{layer}__communication__window__whole_count",
        )
        pre_valid_fixed = _probe_value(
            probes,
            f"{layer}__pre_reset__valid__fixed250_ordered_mean",
        )
        pre_valid_whole = _probe_value(
            probes,
            f"{layer}__pre_reset__valid__whole_mean",
        )
        row[f"{layer}_pre_valid_fixed250_ba"] = pre_valid_fixed
        row[f"{layer}_pre_valid_whole_ba"] = pre_valid_whole
        row[f"{layer}_pre_valid_temporal_gap"] = (
            pre_valid_fixed - pre_valid_whole
        )
        row[f"{layer}_comm_valid_fixed250_ba"] = valid_fixed
        row[f"{layer}_comm_valid_whole_ba"] = valid_whole
        row[f"{layer}_comm_valid_temporal_gap"] = (
            valid_fixed - valid_whole
        )
        row[f"{layer}_comm_window_fixed250_ba"] = window_fixed
        row[f"{layer}_comm_window_whole_ba"] = window_whole
        row[f"{layer}_comm_window_temporal_gap"] = (
            window_fixed - window_whole
        )
    return row


def _contrast_metrics() -> list[str]:
    return [
        "native_test_ba",
        "window_test_ba",
        "output_lif_test_ba",
        "l1_comm_valid_fixed250_ba",
        "l1_comm_valid_whole_ba",
        "l1_comm_valid_temporal_gap",
        "rsnn_comm_valid_fixed250_ba",
        "rsnn_comm_valid_whole_ba",
        "rsnn_comm_valid_temporal_gap",
        "fusion_comm_valid_fixed250_ba",
        "fusion_comm_valid_whole_ba",
        "fusion_comm_valid_temporal_gap",
        "fusion_pre_valid_fixed250_ba",
        "fusion_pre_valid_whole_ba",
        "fusion_pre_valid_temporal_gap",
        "recurrent_to_external_abs_ratio",
    ]


def _paired_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    indexed = runs.set_index(["l1_init", "topology", "seed"])
    rows: list[dict[str, Any]] = []

    def emit(
        contrast: str,
        left: tuple[str, str, int],
        right: tuple[str, str, int],
    ) -> None:
        lrow = indexed.loc[left]
        rrow = indexed.loc[right]
        row: dict[str, Any] = {
            "contrast": contrast,
            "l1_init": (
                left[0] if left[0] == right[0] else "pretrained_vs_dynamics"
            ),
            "topology": (
                left[1] if left[1] == right[1] else f"{left[1]}_vs_{right[1]}"
            ),
            "seed": left[2],
        }
        for metric in _contrast_metrics():
            row[f"delta_{metric}"] = float(lrow[metric] - rrow[metric])
        rows.append(row)

    for seed in MODEL_SEEDS:
        for topology in TOPOLOGIES:
            emit(
                "pretrained_minus_dynamics_only",
                (L1_INIT_PRETRAINED, topology, seed),
                (L1_INIT_DYNAMICS_ONLY, topology, seed),
            )
        for l1_init in L1_INIT_MODES:
            emit(
                "diagonal_minus_ff",
                (l1_init, TOPOLOGY_DIAGONAL, seed),
                (l1_init, TOPOLOGY_FF, seed),
            )
            emit(
                "dense_minus_ff",
                (l1_init, TOPOLOGY_DENSE, seed),
                (l1_init, TOPOLOGY_FF, seed),
            )
            emit(
                "dense_minus_diagonal",
                (l1_init, TOPOLOGY_DENSE, seed),
                (l1_init, TOPOLOGY_DIAGONAL, seed),
            )
    return pd.DataFrame(rows)


def _interaction_rows(runs: pd.DataFrame) -> pd.DataFrame:
    indexed = runs.set_index(["l1_init", "topology", "seed"])
    rows: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        for topology in (TOPOLOGY_DIAGONAL, TOPOLOGY_DENSE):
            row: dict[str, Any] = {
                "interaction": f"pretrain_x_{topology}_vs_ff",
                "topology": topology,
                "seed": seed,
            }
            for metric in _contrast_metrics():
                pretrain_rec = float(
                    indexed.loc[
                        (L1_INIT_PRETRAINED, topology, seed), metric
                    ]
                    - indexed.loc[
                        (L1_INIT_PRETRAINED, TOPOLOGY_FF, seed), metric
                    ]
                )
                random_rec = float(
                    indexed.loc[
                        (L1_INIT_DYNAMICS_ONLY, topology, seed), metric
                    ]
                    - indexed.loc[
                        (L1_INIT_DYNAMICS_ONLY, TOPOLOGY_FF, seed), metric
                    ]
                )
                row[f"interaction_{metric}"] = pretrain_rec - random_rec
            rows.append(row)
    return pd.DataFrame(rows)


def _temporal_gap_rows(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in runs.to_dict("records"):
        for layer in LAYERS:
            for support in SUPPORTS:
                prefix = f"{layer}_comm_{support}"
                rows.append(
                    {
                        "l1_init": source["l1_init"],
                        "topology": source["topology"],
                        "seed": source["seed"],
                        "layer": layer,
                        "support": support,
                        "fixed250_ba": source[f"{prefix}_fixed250_ba"],
                        "whole_ba": source[f"{prefix}_whole_ba"],
                        "temporal_gap": source[
                            f"{prefix}_temporal_gap"
                        ],
                    }
                )
    return pd.DataFrame(rows)


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    probe_frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for spec in run_specs():
        artifacts = _run_artifacts(config, spec)
        for name in ("evaluation", "probes"):
            if not artifacts[name].exists():
                missing.append(f"{spec.key}:{name}:{artifacts[name]}")
        if artifacts["evaluation"].exists():
            payloads.append(
                json.loads(
                    artifacts["evaluation"].read_text(encoding="utf-8")
                )
            )
        if artifacts["probes"].exists():
            probe_frames.append(pd.read_csv(artifacts["probes"]))

    if missing:
        raise FileNotFoundError(
            f"Exp11.0 incomplete; missing {len(missing)} artifacts:\n"
            + "\n".join(missing[:50])
        )
    if len(payloads) != EXPECTED_RUNS or len(probe_frames) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} runs, got {len(payloads)} evaluations "
            f"and {len(probe_frames)} probe files"
        )

    rows = [
        _run_row(payload, probes)
        for payload, probes in zip(payloads, probe_frames, strict=True)
    ]
    runs = pd.DataFrame(rows).sort_values(
        ["l1_init", "topology", "seed"]
    )
    runs.to_csv(config.results_dir / "run_metrics.csv", index=False)

    metric_cols = [
        column
        for column in runs.columns
        if column
        not in {
            "l1_init",
            "topology",
            "seed",
            "best_epoch",
            "stopped_epoch",
            "parameter_count",
            "trainable_parameter_count",
            "recurrent_param_count",
        }
    ]
    summary = (
        runs.groupby(["l1_init", "topology"], sort=False)[metric_cols]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(config.results_dir / "method_summary.csv", index=False)

    contrasts = _paired_contrasts(runs)
    contrasts.to_csv(config.results_dir / "paired_contrasts.csv", index=False)
    delta_cols = [
        column for column in contrasts.columns if column.startswith("delta_")
    ]
    contrast_summary = (
        contrasts.groupby(
            ["contrast", "l1_init", "topology"], sort=False
        )[delta_cols]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    contrast_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in contrast_summary.columns
    ]
    contrast_summary.to_csv(
        config.results_dir / "paired_contrast_summary.csv",
        index=False,
    )

    interactions = _interaction_rows(runs)
    interactions.to_csv(
        config.results_dir / "interaction_runs.csv", index=False
    )
    interaction_cols = [
        column
        for column in interactions.columns
        if column.startswith("interaction_")
    ]
    interaction_summary = (
        interactions.groupby(["interaction", "topology"], sort=False)[
            interaction_cols
        ]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    interaction_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in interaction_summary.columns
    ]
    interaction_summary.to_csv(
        config.results_dir / "interaction_summary.csv",
        index=False,
    )

    all_probes = pd.concat(probe_frames, ignore_index=True)
    all_probes.to_csv(config.results_dir / "probe_runs.csv", index=False)
    probe_summary = (
        all_probes.groupby(
            [
                "l1_init",
                "topology",
                "layer",
                "state",
                "support",
                "aggregation",
            ],
            sort=False,
        )["test_balanced_accuracy"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    probe_summary.to_csv(
        config.results_dir / "probe_summary.csv", index=False
    )

    temporal_gaps = _temporal_gap_rows(runs)
    temporal_gaps.to_csv(
        config.results_dir / "temporal_gap_runs.csv", index=False
    )
    temporal_gap_summary = (
        temporal_gaps.groupby(
            ["l1_init", "topology", "layer", "support"], sort=False
        )[["fixed250_ba", "whole_ba", "temporal_gap"]]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    temporal_gap_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in temporal_gap_summary.columns
    ]
    temporal_gap_summary.to_csv(
        config.results_dir / "temporal_gap_summary.csv",
        index=False,
    )

    activity_rows: list[dict[str, Any]] = []
    for payload in payloads:
        spec = payload["spec"]
        for layer, metrics in payload["activity"].items():
            activity_rows.append(
                {
                    "l1_init": spec["l1_init"],
                    "topology": spec["topology"],
                    "seed": int(spec["seed"]),
                    "layer": layer,
                    **metrics,
                }
            )
    activity = pd.DataFrame(activity_rows)
    activity.to_csv(config.results_dir / "activity_runs.csv", index=False)
    activity_metrics = [
        "mean_events_per_neuron_step",
        "mean_events_per_neuron_s",
        "nonzero_fraction",
        "dead_neuron_fraction",
        "post_valid_event_fraction",
    ]
    activity_summary = (
        activity.groupby(
            ["l1_init", "topology", "layer"], sort=False
        )[activity_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    activity_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in activity_summary.columns
    ]
    activity_summary.to_csv(
        config.results_dir / "activity_summary.csv", index=False
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS",
        "dataset": VARIANT,
        "rotation": ROTATION,
        "l1_init_modes": list(L1_INIT_MODES),
        "topologies": list(TOPOLOGIES),
        "model_seeds": list(MODEL_SEEDS),
        "run_count": int(len(runs)),
        "expected_run_count": EXPECTED_RUNS,
        "probe_rows": int(len(all_probes)),
        "expected_probe_rows": EXPECTED_RUNS * EXPECTED_PROBES_PER_RUN,
        "rsnn_tau_syn_ms": CONTEXT_TAU_MS,
        "rsnn_tau_mem_ms": CONTEXT_TAU_MS,
        "fusion_tau_syn_ms": FUSION_TAU_MS,
        "fusion_tau_mem_ms": FUSION_TAU_MS,
        "objective": "valid-length mean time-shared WCCE",
        "primary_outputs": [
            "run_metrics.csv",
            "method_summary.csv",
            "paired_contrasts.csv",
            "paired_contrast_summary.csv",
            "interaction_runs.csv",
            "interaction_summary.csv",
            "probe_runs.csv",
            "probe_summary.csv",
            "temporal_gap_runs.csv",
            "temporal_gap_summary.csv",
            "activity_runs.csv",
            "activity_summary.csv",
        ],
        "statistical_scope": (
            "one locked cross-user split; seeds are paired optimization "
            "replicates, not independent user splits"
        ),
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _resolve_config(args: argparse.Namespace) -> Config:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else find_repo_root()
    )
    output = (
        Path(args.results_dir).resolve()
        if args.results_dir
        else results_dir(repo_root)
    )
    return Config(
        repo_root=repo_root,
        results_dir=output,
        device=args.device,
        threads=args.threads,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exp11.0: internalize Fixed250 temporal information with "
            "L1 -> RSNN context -> feed-forward fusion -> shared Linear"
        )
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("list-runs")
    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _resolve_config(args)

    if args.command == "prepare":
        print(json.dumps(prepare_all(config), indent=2, sort_keys=True))
        return
    if args.command == "list-runs":
        for index, spec in enumerate(run_specs()):
            print(index, spec.key)
        return
    if args.command == "run-one":
        specs = run_specs()
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        spec = specs[args.array_task_id]
        payload = run_one(spec, config, force=args.force)
        print(
            json.dumps(
                {
                    "key": spec.key,
                    "best_epoch": payload["best_epoch"],
                    "native_test_ba": payload["native_metrics"]["test"][
                        "balanced_accuracy"
                    ],
                    "fusion_temporal_gap": _probe_value(
                        pd.read_csv(_run_artifacts(config, spec)["probes"]),
                        "fusion__communication__valid__fixed250_count",
                    )
                    - _probe_value(
                        pd.read_csv(_run_artifacts(config, spec)["probes"]),
                        "fusion__communication__valid__whole_count",
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "finalize":
        print(json.dumps(finalize(config), indent=2, sort_keys=True))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
