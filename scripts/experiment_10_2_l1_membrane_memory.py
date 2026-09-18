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
from torch import nn
import torch.nn.functional as F

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_6_0_multiscale_phase_evidence as exp60
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_8_1_hidden_quantization_ablation as exp81
from scripts import experiment_8_1_1_two_layer_mt_factorial as exp811
from scripts import experiment_9_0_within_user_generalization as exp90
from scripts import experiment_10_0_airborne_motion_ablation as exp10
from scripts import experiment_10_1_a2_backbone_coding_supervision as exp101


EXPERIMENT_ID = "experiment_10_2_l1_membrane_memory"
PROTOCOL_VERSION = "d1_bb_l1_mem_shift_sweep_v1"

VARIANT = exp101.VARIANT_POSTENCODE
CODING = exp101.CODING_BB
OBJECTIVE = exp101.OBJECTIVE_BASELINE
ROTATION = 0
MODEL_SEEDS = (11, 23, 37)

L1_MEM_SHIFTS = (1, 2, 3, 4)
L2_MEM_SHIFT = 1
BASELINE_L1_MEM_SHIFT = 1

WIDTH = exp101.WIDTH
SYN_SHIFTS = exp101.SHIFTS
MAX_EPOCHS = exp101.MAX_EPOCHS
MIN_EPOCHS = exp101.MIN_EPOCHS
PATIENCE = exp101.PATIENCE
BATCH_SIZE = exp101.BATCH_SIZE
SPLITS = exp101.SPLITS
HIDDEN_LAYERS = exp101.HIDDEN_LAYERS
HIDDEN_STATES = exp101.HIDDEN_STATES

EXPECTED_BASELINE_RUNS = len(MODEL_SEEDS)
EXPECTED_LONG_E2E_RUNS = (len(L1_MEM_SHIFTS) - 1) * len(MODEL_SEEDS)
EXPECTED_E2E_RUNS = len(L1_MEM_SHIFTS) * len(MODEL_SEEDS)
EXPECTED_REPLAY_RUNS = len(L1_MEM_SHIFTS) * len(MODEL_SEEDS)

STAGE_REPLAY = "frozen_replay"
STAGE_E2E = "e2e_retrain"


@dataclass(frozen=True)
class TrainSpec:
    l1_mem_shift: int
    seed: int

    @property
    def key(self) -> str:
        return f"train__l1mem{self.l1_mem_shift}__seed{self.seed}"

    @property
    def variant(self) -> str:
        return VARIANT

    @property
    def rotation(self) -> int:
        return ROTATION

    @property
    def coding(self) -> str:
        return CODING

    @property
    def objective(self) -> str:
        return OBJECTIVE

    @property
    def l1_coding(self) -> str:
        return "binary"

    @property
    def l2_coding(self) -> str:
        return "binary"


@dataclass(frozen=True)
class ReplaySpec:
    target_l1_mem_shift: int
    seed: int

    @property
    def source_l1_mem_shift(self) -> int:
        return BASELINE_L1_MEM_SHIFT

    @property
    def key(self) -> str:
        return (
            f"replay__source{self.source_l1_mem_shift}"
            f"__target{self.target_l1_mem_shift}__seed{self.seed}"
        )

    @property
    def variant(self) -> str:
        return VARIANT

    @property
    def rotation(self) -> int:
        return ROTATION

    @property
    def coding(self) -> str:
        return CODING

    @property
    def objective(self) -> str:
        return OBJECTIVE

    @property
    def l1_coding(self) -> str:
        return "binary"

    @property
    def l2_coding(self) -> str:
        return "binary"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    threads: int = 1
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp101.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def beta_from_mem_shift(
    shift: int,
    fs: float = exp72.EXPECTED_FS,
) -> float:
    shift = int(shift)
    if shift == BASELINE_L1_MEM_SHIFT:
        # Preserve the exact current A2 contract. Existing A2 code stores
        # tau_mem as 22.54 ms and derives beta from it, giving ~0.4999676 at
        # 64 Hz rather than a hard-coded 0.5.
        return float(
            math.exp(
                -(1000.0 / float(fs)) / float(exp72.TAU_MEM_MS)
            )
        )
    return exp60.decay_from_shift(shift)


def tau_mem_ms_from_shift(
    shift: int,
    fs: float = exp72.EXPECTED_FS,
) -> float:
    shift = int(shift)
    if shift == BASELINE_L1_MEM_SHIFT:
        return float(exp72.TAU_MEM_MS)
    return exp60.tau_ms_from_shift(shift, float(fs))


def baseline_specs() -> list[TrainSpec]:
    return [TrainSpec(BASELINE_L1_MEM_SHIFT, seed) for seed in MODEL_SEEDS]


def long_e2e_specs() -> list[TrainSpec]:
    return [
        TrainSpec(shift, seed)
        for shift in L1_MEM_SHIFTS
        if shift != BASELINE_L1_MEM_SHIFT
        for seed in MODEL_SEEDS
    ]


def e2e_specs() -> list[TrainSpec]:
    return [
        TrainSpec(shift, seed)
        for shift in L1_MEM_SHIFTS
        for seed in MODEL_SEEDS
    ]


def replay_specs() -> list[ReplaySpec]:
    return [
        ReplaySpec(shift, seed)
        for shift in L1_MEM_SHIFTS
        for seed in MODEL_SEEDS
    ]


def validate_train_spec(spec: TrainSpec) -> None:
    if spec.l1_mem_shift not in L1_MEM_SHIFTS:
        raise ValueError(spec.l1_mem_shift)
    if spec.seed not in MODEL_SEEDS:
        raise ValueError(spec.seed)


def validate_replay_spec(spec: ReplaySpec) -> None:
    if spec.target_l1_mem_shift not in L1_MEM_SHIFTS:
        raise ValueError(spec.target_l1_mem_shift)
    if spec.seed not in MODEL_SEEDS:
        raise ValueError(spec.seed)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _fold_assignment_path(config: Config) -> Path:
    return config.results_dir / "split" / "cross_user_fold_assignment.csv"


def _rotation_manifest_path(config: Config) -> Path:
    return config.results_dir / "split" / f"rotation{ROTATION}.csv"


def prepare_all(config: Config) -> dict[str, Any]:
    loaded, manifest, labels = exp101._load_variant_manifest(
        config.repo_root, VARIANT
    )
    del loaded
    config.results_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / "split").mkdir(parents=True, exist_ok=True)

    canonical = manifest.drop(columns=["pi", "si"]).sort_values("sample_id")
    canonical.to_csv(
        config.results_dir / "canonical_sample_manifest.csv", index=False
    )

    assignment = exp90._assign_cross_user_folds(manifest)
    assignment.to_csv(_fold_assignment_path(config), index=False)
    split_manifest = exp90._apply_rotation(assignment, ROTATION)
    split_manifest.drop(columns=["pi", "si"]).to_csv(
        _rotation_manifest_path(config), index=False
    )

    roles = exp90._rotation_fold_roles(ROTATION)
    users = {
        split: sorted(
            split_manifest.loc[split_manifest.split == split, "user"]
            .astype(str)
            .unique()
            .tolist()
        )
        for split in SPLITS
    }
    counts = split_manifest.split.value_counts()
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Can longer L1 membrane memory convert subthreshold analog evidence "
            "into binary spike evidence without destroying downstream temporal "
            "representation?"
        ),
        "dataset": "D1 postencode_mask only",
        "variant": VARIANT,
        "coding": CODING,
        "objective": OBJECTIVE,
        "readout": "time_shared",
        "rotation": ROTATION,
        "fold_roles": {str(k): v for k, v in roles.items()},
        "split_users": users,
        "split_samples": {
            split: int(counts.get(split, 0)) for split in SPLITS
        },
        "labels": list(labels),
        "l1_mem_shifts": list(L1_MEM_SHIFTS),
        "l1_beta": {
            str(shift): beta_from_mem_shift(shift)
            for shift in L1_MEM_SHIFTS
        },
        "l1_tau_mem_ms": {
            str(shift): tau_mem_ms_from_shift(shift)
            for shift in L1_MEM_SHIFTS
        },
        "l2_mem_shift": L2_MEM_SHIFT,
        "l2_beta": beta_from_mem_shift(L2_MEM_SHIFT),
        "l2_tau_mem_ms": tau_mem_ms_from_shift(L2_MEM_SHIFT),
        "synaptic_shifts": [list(SYN_SHIFTS), list(SYN_SHIFTS)],
        "model_seeds": list(MODEL_SEEDS),
        "stage_a": {
            "source": "Exp10.2 shift_mem=1 baseline checkpoints",
            "frozen_replay_runs": EXPECTED_REPLAY_RUNS,
            "trained_parameters_changed": False,
        },
        "stage_b": {
            "e2e_runs_total": EXPECTED_E2E_RUNS,
            "baseline_runs": EXPECTED_BASELINE_RUNS,
            "long_mem_retrain_runs": EXPECTED_LONG_E2E_RUNS,
        },
        "paired_randomness": (
            "For a fixed seed, model initialization and train-loader order are "
            "identical across all L1 membrane shifts."
        ),
    }
    _save_json(config.results_dir / "audit.json", audit)
    return audit


def _prepare_data(
    config: Config,
) -> tuple[exp3.Data, dict[str, pd.DataFrame], pd.DataFrame]:
    return exp101._prepare_run_data(config, VARIANT, ROTATION)


class Exp102Net(nn.Module):
    """A2 BB backbone with controlled L1 membrane decay and fixed L2 beta=0.5."""

    def __init__(
        self,
        l1_mem_shift: int,
        n_classes: int,
        fs: float,
    ) -> None:
        super().__init__()
        if l1_mem_shift not in L1_MEM_SHIFTS:
            raise ValueError(l1_mem_shift)
        self.l1_mem_shift = int(l1_mem_shift)
        self.l2_mem_shift = int(L2_MEM_SHIFT)
        self.n_classes = int(n_classes)
        self.fs = float(fs)

        self.hidden_linears = nn.ModuleList(
            [
                nn.Linear(exp72.EXPECTED_CHANNELS, WIDTH, bias=False),
                nn.Linear(WIDTH, WIDTH, bias=False),
            ]
        )
        self.output_linear = nn.Linear(WIDTH, n_classes, bias=False)

        self.l1_lif = exp811._lif(
            "binary", beta_from_mem_shift(l1_mem_shift, fs)
        )
        self.l2_lif = exp811._lif(
            "binary", beta_from_mem_shift(L2_MEM_SHIFT, fs)
        )
        self.register_buffer("alpha_0", exp811._alpha_vector("binary"))
        self.register_buffer("alpha_1", exp811._alpha_vector("binary"))

        # Compatibility with Exp10.1 diagnostic helpers.
        self.spec: TrainSpec | ReplaySpec | None = None

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != exp72.EXPECTED_CHANNELS:
            raise ValueError(channels)

        syn1 = torch.zeros(batch, WIDTH, device=x.device, dtype=x.dtype)
        mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, WIDTH, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(syn2)

        states: dict[str, dict[str, list[torch.Tensor]]] = {
            layer: {state: [] for state in HIDDEN_STATES}
            for layer in HIDDEN_LAYERS
        }
        evidence: list[torch.Tensor] = []

        for timestep in range(steps):
            syn1 = self.alpha_0 * syn1 + self.hidden_linears[0](x[:, timestep])
            out1, post1, pre1 = self.l1_lif(syn1, mem1)
            mem1 = post1

            syn2 = self.alpha_1 * syn2 + self.hidden_linears[1](out1)
            out2, post2, pre2 = self.l2_lif(syn2, mem2)
            mem2 = post2

            states["l1"]["syn_current"].append(syn1)
            states["l1"]["pre_reset"].append(pre1)
            states["l1"]["spike"].append(out1)
            states["l1"]["post_reset"].append(post1)
            states["l2"]["syn_current"].append(syn2)
            states["l2"]["pre_reset"].append(pre2)
            states["l2"]["spike"].append(out2)
            states["l2"]["post_reset"].append(post2)
            evidence.append(self.output_linear(out2))

        hidden = {
            layer: {
                state: torch.stack(values, dim=1)
                for state, values in state_map.items()
            }
            for layer, state_map in states.items()
        }
        return {
            "hidden": hidden,
            "l2_evidence": torch.stack(evidence, dim=1),
            "l1_evidence": None,
        }


def _native_scores(
    model: Exp102Net,
    X: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    trajectory = model.forward_trajectory(X)
    return exp101._valid_mean(trajectory["l2_evidence"], lengths)


def _evaluate_native(
    model: Exp102Net,
    loader: Iterable,
    device: torch.device,
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
            ld = lengths.to(device)
            scores = _native_scores(model, Xd, ld)
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
    model: Exp102Net,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            Xd = X.to(device=device, dtype=torch.float32)
            ld = lengths.to(device)
            trajectory = model.forward_trajectory(Xd)
            spikes = exp81._output_lif_spikes(trajectory["l2_evidence"])
            mask = exp101._valid_mask(ld, spikes.shape[1]).to(
                spikes.dtype
            ).unsqueeze(-1)
            scores = (spikes * mask).sum(dim=1)
            ys.append(y.numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
    return exp10._classification_metrics(
        np.concatenate(ys), np.concatenate(preds)
    )


def _extended_activity(
    model: Exp102Net,
    data: exp3.Data,
    seed: int,
    config: Config,
) -> dict[str, list[dict[str, float | int]]]:
    loader = exp73._raw_loaders(
        data, seed, config.batch_size, False
    )["test"]
    device = torch.device(config.device)
    chunks: dict[str, list[torch.Tensor]] = {"l1": [], "l2": []}
    lengths_all: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for X, _, lengths in loader:
            trajectory = model.forward_trajectory(
                X.to(device=device, dtype=torch.float32)
            )
            for layer in HIDDEN_LAYERS:
                chunks[layer].append(
                    trajectory["hidden"][layer]["spike"].cpu()
                )
            lengths_all.append(
                lengths.numpy().astype(np.int64, copy=False)
            )

    lengths_np = np.concatenate(lengths_all)
    lengths_t = torch.as_tensor(lengths_np, dtype=torch.long)
    output: dict[str, list[dict[str, float | int]]] = {}

    for layer in HIDDEN_LAYERS:
        spikes = torch.cat(chunks[layer], dim=0)
        mask = exp101._valid_mask(lengths_t, spikes.shape[1]).to(
            spikes.dtype
        ).unsqueeze(-1)
        valid_steps = float(mask[:, :, 0].sum().item())
        rows: list[dict[str, float | int]] = []
        for group in exp811._groups("binary"):
            start, stop = int(group["start"]), int(group["stop"])
            chunk = spikes[:, :, start:stop]
            counts = (chunk * mask).sum(dim=(0, 1))
            firing_fraction = counts / max(valid_steps, 1.0)
            rows.append(
                {
                    **group,
                    "mean_value_per_neuron_step": float(
                        firing_fraction.mean().item()
                    ),
                    "fraction_nonzero": float(
                        (counts > 0).float().mean().item()
                    ),
                    "dead_neuron_fraction": float(
                        (counts == 0).float().mean().item()
                    ),
                    "high_rate_neuron_fraction_ge_0p5": float(
                        (firing_fraction >= 0.5).float().mean().item()
                    ),
                    "saturated_neuron_fraction_ge_0p95": float(
                        (firing_fraction >= 0.95).float().mean().item()
                    ),
                    "mean_spikes_per_neuron_s": float(
                        firing_fraction.mean().item() * float(data.fs)
                    ),
                }
            )
        output[layer] = rows
    return output


def _probe_frame(
    model: Exp102Net,
    data: exp3.Data,
    frames: dict[str, pd.DataFrame],
    spec: TrainSpec | ReplaySpec,
    stage: str,
    l1_mem_shift: int,
    config: Config,
) -> pd.DataFrame:
    model.spec = spec
    features, labels, _ = exp101._collect_probe_features_and_activity(
        model, data, spec, config
    )
    test_actions = frames["test"].action.to_numpy(
        dtype=np.int64, copy=True
    )
    frame = exp101._fit_probes(
        spec, features, labels, test_actions
    )
    frame.insert(1, "stage", stage)
    frame.insert(2, "l1_mem_shift", int(l1_mem_shift))
    frame.insert(3, "l1_beta", beta_from_mem_shift(l1_mem_shift))
    frame.insert(
        4,
        "l1_tau_mem_ms",
        tau_mem_ms_from_shift(l1_mem_shift, data.fs),
    )
    return frame


def _train_artifacts(
    config: Config,
    spec: TrainSpec,
) -> dict[str, Path]:
    return {
        "checkpoint": _path(
            config.results_dir, "checkpoints", spec.key, ".pt"
        ),
        "history": _path(
            config.results_dir, "histories", spec.key, ".csv"
        ),
        "evaluation": _path(
            config.results_dir, "evaluations", spec.key, ".json"
        ),
        "probes": _path(
            config.results_dir, "probe_evaluations", spec.key, ".csv"
        ),
    }


def _replay_artifacts(
    config: Config,
    spec: ReplaySpec,
) -> dict[str, Path]:
    return {
        "evaluation": _path(
            config.results_dir,
            "replay_evaluations",
            spec.key,
            ".json",
        ),
        "probes": _path(
            config.results_dir,
            "replay_probe_evaluations",
            spec.key,
            ".csv",
        ),
    }


def run_train(
    spec: TrainSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_train_spec(spec)
    artifacts = _train_artifacts(config, spec)
    if not force and all(path.exists() for path in artifacts.values()):
        return json.loads(
            artifacts["evaluation"].read_text(encoding="utf-8")
        )

    data, frames, _ = _prepare_data(config)
    split_hashes = exp101._split_hashes(frames)
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)

    model_init_seed = exp73._e2e_pair_seed(spec.seed, "model_init")
    exp3.seed_all(model_init_seed)
    model = Exp102Net(
        spec.l1_mem_shift, len(data.labels), data.fs
    ).to(device)
    model.spec = spec
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
            ld = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = _native_scores(model, Xd, ld)
            loss = F.cross_entropy(scores, yd)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics, _, _ = _evaluate_native(
            model, eval_loaders["train"], device
        )
        val_metrics, _, _ = _evaluate_native(
            model, eval_loaders["val"], device
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(
                    train_metrics["balanced_accuracy"]
                ),
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
            "model_init_seed": int(model_init_seed),
            "split_sample_hashes": split_hashes,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "l1_mem_shift": spec.l1_mem_shift,
            "l1_beta": beta_from_mem_shift(spec.l1_mem_shift),
            "l1_tau_mem_ms": tau_mem_ms_from_shift(
                spec.l1_mem_shift, data.fs
            ),
            "l2_mem_shift": L2_MEM_SHIFT,
            "model_state_dict": best_state,
        },
        artifacts["checkpoint"],
    )
    artifacts["history"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(artifacts["history"], index=False)

    model.load_state_dict(best_state, strict=True)
    native_metrics: dict[str, dict[str, float]] = {}
    lif_metrics: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        metrics, _, _ = _evaluate_native(
            model, eval_loaders[split], device
        )
        native_metrics[split] = metrics
        lif_metrics[split] = _evaluate_lif_transfer(
            model, eval_loaders[split], device
        )

    probes = _probe_frame(
        model,
        data,
        frames,
        spec,
        STAGE_E2E,
        spec.l1_mem_shift,
        config,
    )
    artifacts["probes"].parent.mkdir(parents=True, exist_ok=True)
    probes.to_csv(artifacts["probes"], index=False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "stage": STAGE_E2E,
        "spec": asdict(spec),
        "contract": {
            "dataset": VARIANT,
            "coding": CODING,
            "objective": OBJECTIVE,
            "readout": "time_shared",
            "synaptic_shifts": [list(SYN_SHIFTS), list(SYN_SHIFTS)],
            "l1_mem_shift": spec.l1_mem_shift,
            "l1_beta": beta_from_mem_shift(spec.l1_mem_shift),
            "l1_tau_mem_ms": tau_mem_ms_from_shift(
                spec.l1_mem_shift, data.fs
            ),
            "l2_mem_shift": L2_MEM_SHIFT,
            "l2_beta": beta_from_mem_shift(L2_MEM_SHIFT),
            "l2_tau_mem_ms": tau_mem_ms_from_shift(
                L2_MEM_SHIFT, data.fs
            ),
        },
        "model_init_seed": int(model_init_seed),
        "split_sample_hashes": split_hashes,
        "parameter_count": int(
            sum(p.numel() for p in model.parameters())
        ),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native_metrics,
        "lif_transfer_metrics": lif_metrics,
        "activity": _extended_activity(
            model, data, spec.seed, config
        ),
    }
    _save_json(artifacts["evaluation"], payload)
    return payload


def run_replay(
    spec: ReplaySpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_replay_spec(spec)
    artifacts = _replay_artifacts(config, spec)
    if not force and all(path.exists() for path in artifacts.values()):
        return json.loads(
            artifacts["evaluation"].read_text(encoding="utf-8")
        )

    source_spec = TrainSpec(
        BASELINE_L1_MEM_SHIFT, spec.seed
    )
    source_artifacts = _train_artifacts(config, source_spec)
    if not source_artifacts["checkpoint"].exists():
        raise FileNotFoundError(
            "Frozen replay requires the Exp10.2 shift_mem=1 baseline "
            f"checkpoint: {source_artifacts['checkpoint']}"
        )

    data, frames, _ = _prepare_data(config)
    split_hashes = exp101._split_hashes(frames)
    checkpoint = torch.load(
        source_artifacts["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("Replay source checkpoint experiment mismatch")
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Replay source checkpoint protocol mismatch")
    if checkpoint.get("spec") != asdict(source_spec):
        raise RuntimeError("Replay source checkpoint spec mismatch")
    if checkpoint.get("split_sample_hashes") != split_hashes:
        raise RuntimeError("Replay source split geometry mismatch")

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model = Exp102Net(
        spec.target_l1_mem_shift,
        len(data.labels),
        data.fs,
    ).to(device)
    model.spec = spec
    # LIF beta is not a learned/state_dict parameter, so loading the
    # shift=1 weights into this target-shift model changes only L1 membrane
    # dynamics while keeping every learned weight frozen.
    model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )

    eval_loaders = exp73._raw_loaders(
        data, spec.seed, config.batch_size, False
    )
    native_metrics: dict[str, dict[str, float]] = {}
    lif_metrics: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        metrics, _, _ = _evaluate_native(
            model, eval_loaders[split], device
        )
        native_metrics[split] = metrics
        lif_metrics[split] = _evaluate_lif_transfer(
            model, eval_loaders[split], device
        )

    probes = _probe_frame(
        model,
        data,
        frames,
        spec,
        STAGE_REPLAY,
        spec.target_l1_mem_shift,
        config,
    )
    artifacts["probes"].parent.mkdir(parents=True, exist_ok=True)
    probes.to_csv(artifacts["probes"], index=False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "stage": STAGE_REPLAY,
        "spec": asdict(spec),
        "source_train_spec": asdict(source_spec),
        "source_best_epoch": int(checkpoint["best_epoch"]),
        "contract": {
            "dataset": VARIANT,
            "coding": CODING,
            "objective": OBJECTIVE,
            "readout": "time_shared",
            "learned_weights_frozen": True,
            "source_l1_mem_shift": BASELINE_L1_MEM_SHIFT,
            "target_l1_mem_shift": spec.target_l1_mem_shift,
            "target_l1_beta": beta_from_mem_shift(
                spec.target_l1_mem_shift
            ),
            "target_l1_tau_mem_ms": tau_mem_ms_from_shift(
                spec.target_l1_mem_shift, data.fs
            ),
            "l2_mem_shift": L2_MEM_SHIFT,
        },
        "model_init_seed": int(checkpoint["model_init_seed"]),
        "split_sample_hashes": split_hashes,
        "parameter_count": int(
            sum(p.numel() for p in model.parameters())
        ),
        "native_metrics": native_metrics,
        "lif_transfer_metrics": lif_metrics,
        "activity": _extended_activity(
            model, data, spec.seed, config
        ),
    }
    _save_json(artifacts["evaluation"], payload)
    return payload


def _probe_value(
    frame: pd.DataFrame,
    probe: str,
) -> float:
    row = frame[frame.probe == probe]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected one probe row for {probe}, got {len(row)}"
        )
    return float(row.iloc[0].test_balanced_accuracy)


def _run_row(
    payload: Mapping[str, Any],
    probes: pd.DataFrame,
) -> dict[str, Any]:
    stage = str(payload["stage"])
    if stage == STAGE_E2E:
        shift = int(payload["spec"]["l1_mem_shift"])
        seed = int(payload["spec"]["seed"])
    elif stage == STAGE_REPLAY:
        shift = int(payload["spec"]["target_l1_mem_shift"])
        seed = int(payload["spec"]["seed"])
    else:
        raise ValueError(stage)

    l1_pre = _probe_value(
        probes, "l1__pre_reset__fixed250_ordered_mean"
    )
    l1_spike = _probe_value(
        probes, "l1__spike__fixed250_count"
    )
    l2_pre = _probe_value(
        probes, "l2__pre_reset__fixed250_ordered_mean"
    )
    l2_spike = _probe_value(
        probes, "l2__spike__fixed250_count"
    )
    l2_whole = _probe_value(
        probes, "l2__spike__whole_count"
    )

    native = payload["native_metrics"]
    row: dict[str, Any] = {
        "stage": stage,
        "l1_mem_shift": shift,
        "l1_beta": beta_from_mem_shift(shift),
        "l1_tau_mem_ms": tau_mem_ms_from_shift(shift),
        "l2_mem_shift": L2_MEM_SHIFT,
        "seed": seed,
        "native_train_ba": float(
            native["train"]["balanced_accuracy"]
        ),
        "native_val_ba": float(
            native["val"]["balanced_accuracy"]
        ),
        "native_test_ba": float(
            native["test"]["balanced_accuracy"]
        ),
        "lif_test_ba": float(
            payload["lif_transfer_metrics"]["test"][
                "balanced_accuracy"
            ]
        ),
        "l1_pre_reset_fixed250_ba": l1_pre,
        "l1_spike_fixed250_ba": l1_spike,
        "l1_quantization_delta": l1_spike - l1_pre,
        "l2_pre_reset_fixed250_ba": l2_pre,
        "l2_spike_fixed250_ba": l2_spike,
        "l2_quantization_delta": l2_spike - l2_pre,
        "l2_spike_whole_ba": l2_whole,
        "l2_temporal_ordering_gain": l2_spike - l2_whole,
    }
    if stage == STAGE_E2E:
        row["best_epoch"] = int(payload["best_epoch"])
        row["stopped_epoch"] = int(payload["stopped_epoch"])
    else:
        row["best_epoch"] = int(payload["source_best_epoch"])
        row["stopped_epoch"] = np.nan
    return row


def _activity_rows(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    stage = str(payload["stage"])
    if stage == STAGE_E2E:
        shift = int(payload["spec"]["l1_mem_shift"])
        seed = int(payload["spec"]["seed"])
    else:
        shift = int(payload["spec"]["target_l1_mem_shift"])
        seed = int(payload["spec"]["seed"])

    rows: list[dict[str, Any]] = []
    for layer, groups in payload["activity"].items():
        for group in groups:
            rows.append(
                {
                    "stage": stage,
                    "l1_mem_shift": shift,
                    "seed": seed,
                    "layer": layer,
                    **group,
                }
            )
    return rows


def _load_stage(
    config: Config,
    stage: str,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    payloads: list[dict[str, Any]] = []
    probes: list[pd.DataFrame] = []
    if stage == STAGE_E2E:
        specs: list[TrainSpec | ReplaySpec] = list(e2e_specs())
        artifact_fn = _train_artifacts
    elif stage == STAGE_REPLAY:
        specs = list(replay_specs())
        artifact_fn = _replay_artifacts
    else:
        raise ValueError(stage)

    missing: list[str] = []
    for spec in specs:
        artifacts = artifact_fn(config, spec)  # type: ignore[arg-type]
        for name in ("evaluation", "probes"):
            if not artifacts[name].exists():
                missing.append(f"{spec.key}:{name}:{artifacts[name]}")
        if artifacts["evaluation"].exists():
            payloads.append(
                json.loads(
                    artifacts["evaluation"].read_text(
                        encoding="utf-8"
                    )
                )
            )
        if artifacts["probes"].exists():
            probes.append(pd.read_csv(artifacts["probes"]))
    if missing:
        raise FileNotFoundError(
            f"{stage} incomplete; missing {len(missing)} artifacts:\n"
            + "\n".join(missing[:30])
        )
    return payloads, probes


def _load_stage_a(
    config: Config,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    payloads: list[dict[str, Any]] = []
    probes: list[pd.DataFrame] = []
    missing: list[str] = []
    for spec in baseline_specs():
        artifacts = _train_artifacts(config, spec)
        for name in ("evaluation", "probes"):
            if not artifacts[name].exists():
                missing.append(f"{spec.key}:{name}:{artifacts[name]}")
    replay_payloads, replay_probes = _load_stage(
        config, STAGE_REPLAY
    )
    if missing:
        raise FileNotFoundError(
            "Stage A baseline artifacts missing:\n"
            + "\n".join(missing)
        )
    payloads.extend(replay_payloads)
    probes.extend(replay_probes)
    return payloads, probes


def _summarize_runs(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "native_train_ba",
        "native_val_ba",
        "native_test_ba",
        "lif_test_ba",
        "l1_pre_reset_fixed250_ba",
        "l1_spike_fixed250_ba",
        "l1_quantization_delta",
        "l2_pre_reset_fixed250_ba",
        "l2_spike_fixed250_ba",
        "l2_quantization_delta",
        "l2_spike_whole_ba",
        "l2_temporal_ordering_gain",
    ]
    summary = (
        runs.groupby(
            ["stage", "l1_mem_shift", "l1_beta", "l1_tau_mem_ms"],
            sort=True,
        )[metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def _shift_contrasts(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "native_test_ba",
        "l1_pre_reset_fixed250_ba",
        "l1_spike_fixed250_ba",
        "l1_quantization_delta",
        "l2_pre_reset_fixed250_ba",
        "l2_spike_fixed250_ba",
        "l2_spike_whole_ba",
        "l2_temporal_ordering_gain",
    ]
    rows: list[dict[str, Any]] = []
    for stage in sorted(runs.stage.unique()):
        cell = runs[runs.stage == stage]
        indexed = cell.set_index(["seed", "l1_mem_shift"])
        for seed in MODEL_SEEDS:
            baseline = indexed.loc[(seed, BASELINE_L1_MEM_SHIFT)]
            for shift in L1_MEM_SHIFTS:
                if shift == BASELINE_L1_MEM_SHIFT:
                    continue
                current = indexed.loc[(seed, shift)]
                row: dict[str, Any] = {
                    "stage": stage,
                    "seed": seed,
                    "l1_mem_shift": shift,
                    "vs_shift": BASELINE_L1_MEM_SHIFT,
                }
                for metric in metrics:
                    row[f"{metric}_delta"] = float(
                        current[metric] - baseline[metric]
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _write_aggregate(
    config: Config,
    runs: pd.DataFrame,
    activity: pd.DataFrame,
    prefix: str,
) -> None:
    runs.to_csv(
        config.results_dir / f"{prefix}_runs.csv", index=False
    )
    _summarize_runs(runs).to_csv(
        config.results_dir / f"{prefix}_summary.csv", index=False
    )
    contrasts = _shift_contrasts(runs)
    contrasts.to_csv(
        config.results_dir / f"{prefix}_shift_contrasts.csv",
        index=False,
    )
    contrast_metrics = [
        c for c in contrasts.columns if c.endswith("_delta")
    ]
    contrast_summary = (
        contrasts.groupby(
            ["stage", "l1_mem_shift"], sort=True
        )[contrast_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    contrast_summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in contrast_summary.columns
    ]
    contrast_summary.to_csv(
        config.results_dir / f"{prefix}_shift_contrast_summary.csv",
        index=False,
    )

    activity.to_csv(
        config.results_dir / f"{prefix}_activity_runs.csv",
        index=False,
    )
    activity_summary = (
        activity.groupby(
            ["stage", "l1_mem_shift", "layer", "shift"],
            sort=True,
        )[
            [
                "mean_value_per_neuron_step",
                "dead_neuron_fraction",
                "high_rate_neuron_fraction_ge_0p5",
                "saturated_neuron_fraction_ge_0p95",
                "mean_spikes_per_neuron_s",
            ]
        ]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    activity_summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in activity_summary.columns
    ]
    activity_summary.to_csv(
        config.results_dir / f"{prefix}_activity_summary.csv",
        index=False,
    )


def finalize_stage_a(config: Config) -> dict[str, Any]:
    payloads, probe_frames = _load_stage_a(config)
    if len(payloads) != EXPECTED_REPLAY_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_REPLAY_RUNS} replay evaluations, "
            f"got {len(payloads)}"
        )
    rows = [
        _run_row(payload, probes)
        for payload, probes in zip(payloads, probe_frames, strict=True)
    ]
    runs = pd.DataFrame(rows).sort_values(
        ["l1_mem_shift", "seed"]
    )
    activity = pd.DataFrame(
        [
            row
            for payload in payloads
            for row in _activity_rows(payload)
        ]
    )
    _write_aggregate(config, runs, activity, "stage_a_replay")

    sanity_rows: list[dict[str, Any]] = []
    replay_shift1 = runs[
        runs.l1_mem_shift == BASELINE_L1_MEM_SHIFT
    ].set_index("seed")
    for spec in baseline_specs():
        train_eval = json.loads(
            _train_artifacts(config, spec)["evaluation"].read_text(
                encoding="utf-8"
            )
        )
        replay_row = replay_shift1.loc[spec.seed]
        sanity_rows.append(
            {
                "seed": spec.seed,
                "train_shift1_native_test_ba": float(
                    train_eval["native_metrics"]["test"][
                        "balanced_accuracy"
                    ]
                ),
                "replay_shift1_native_test_ba": float(
                    replay_row.native_test_ba
                ),
                "absolute_delta": abs(
                    float(
                        train_eval["native_metrics"]["test"][
                            "balanced_accuracy"
                        ]
                    )
                    - float(replay_row.native_test_ba)
                ),
            }
        )
    pd.DataFrame(sanity_rows).to_csv(
        config.results_dir / "stage_a_shift1_replay_sanity.csv",
        index=False,
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "stage": "A",
        "status": "PASS",
        "run_count": int(len(runs)),
        "expected_run_count": EXPECTED_REPLAY_RUNS,
        "primary_metric": "l1_quantization_delta",
        "interpretation": (
            "A less-negative l1_quantization_delta indicates that frozen "
            "longer membrane dynamics preserve more linearly decodable "
            "L1 spike information relative to L1 pre-reset state."
        ),
    }
    _save_json(config.results_dir / "stage_a_manifest.json", manifest)
    return manifest


def finalize(config: Config) -> dict[str, Any]:
    # Full finalization also materializes the Stage-A-only artifacts so the
    # artifact contract is identical whether the user runs Stage A first or
    # submits the complete experiment directly.
    finalize_stage_a(config)

    replay_payloads, replay_probes = _load_stage(
        config, STAGE_REPLAY
    )
    e2e_payloads, e2e_probes = _load_stage(
        config, STAGE_E2E
    )
    if len(replay_payloads) != EXPECTED_REPLAY_RUNS:
        raise RuntimeError(len(replay_payloads))
    if len(e2e_payloads) != EXPECTED_E2E_RUNS:
        raise RuntimeError(len(e2e_payloads))

    replay_runs = pd.DataFrame(
        [
            _run_row(payload, probes)
            for payload, probes in zip(
                replay_payloads, replay_probes, strict=True
            )
        ]
    )
    e2e_runs = pd.DataFrame(
        [
            _run_row(payload, probes)
            for payload, probes in zip(
                e2e_payloads, e2e_probes, strict=True
            )
        ]
    )
    runs = pd.concat(
        [replay_runs, e2e_runs], ignore_index=True
    ).sort_values(["stage", "l1_mem_shift", "seed"])

    activity = pd.DataFrame(
        [
            row
            for payload in [*replay_payloads, *e2e_payloads]
            for row in _activity_rows(payload)
        ]
    )
    _write_aggregate(config, runs, activity, "all")

    stage_comparison = e2e_runs.merge(
        replay_runs,
        on=["l1_mem_shift", "seed"],
        suffixes=("_e2e", "_replay"),
    )
    for metric in (
        "native_test_ba",
        "l1_pre_reset_fixed250_ba",
        "l1_spike_fixed250_ba",
        "l1_quantization_delta",
        "l2_spike_fixed250_ba",
    ):
        stage_comparison[f"{metric}_e2e_minus_replay"] = (
            stage_comparison[f"{metric}_e2e"]
            - stage_comparison[f"{metric}_replay"]
        )
    stage_comparison.to_csv(
        config.results_dir / "e2e_minus_replay.csv",
        index=False,
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS",
        "dataset": VARIANT,
        "rotation": ROTATION,
        "l1_mem_shifts": list(L1_MEM_SHIFTS),
        "l2_mem_shift": L2_MEM_SHIFT,
        "model_seeds": list(MODEL_SEEDS),
        "replay_runs": EXPECTED_REPLAY_RUNS,
        "e2e_runs": EXPECTED_E2E_RUNS,
        "readout": "time_shared_only",
        "phase_aware_readout": False,
        "primary_questions": [
            "Does frozen longer L1 membrane memory make l1_quantization_delta less negative?",
            "After end-to-end retraining, does L1 spike Fixed250 BA rise without reducing L1 pre-reset BA?",
            "Do longer membrane shifts improve L2 temporal representation and native BA without saturation?",
        ],
        "statistical_scope": (
            "one locked cross-user split; seeds 11/23/37 are paired "
            "optimization replicates, not independent user splits"
        ),
        "primary_outputs": [
            "stage_a_replay_runs.csv",
            "stage_a_replay_summary.csv",
            "all_runs.csv",
            "all_summary.csv",
            "all_shift_contrasts.csv",
            "all_shift_contrast_summary.csv",
            "all_activity_runs.csv",
            "all_activity_summary.csv",
            "e2e_minus_replay.csv",
        ],
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
            "Exp10.2 L1 membrane-memory sweep: frozen replay plus "
            "end-to-end D1 BB retraining"
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

    baseline = sub.add_parser("run-baseline")
    baseline.add_argument("--array-task-id", type=int, required=True)
    baseline.add_argument("--force", action="store_true")

    long_train = sub.add_parser("run-long-e2e")
    long_train.add_argument("--array-task-id", type=int, required=True)
    long_train.add_argument("--force", action="store_true")

    replay = sub.add_parser("run-replay")
    replay.add_argument("--array-task-id", type=int, required=True)
    replay.add_argument("--force", action="store_true")

    sub.add_parser("finalize-stage-a")
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _resolve_config(args)

    if args.command == "prepare":
        print(json.dumps(prepare_all(config), indent=2, sort_keys=True))
        return

    if args.command == "list-runs":
        print("Baseline train:")
        for index, spec in enumerate(baseline_specs()):
            print(index, spec.key)
        print("Long-memory E2E:")
        for index, spec in enumerate(long_e2e_specs()):
            print(index, spec.key)
        print("Frozen replay:")
        for index, spec in enumerate(replay_specs()):
            print(index, spec.key)
        return

    if args.command == "run-baseline":
        specs = baseline_specs()
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        payload = run_train(
            specs[args.array_task_id],
            config,
            force=args.force,
        )
        print(
            json.dumps(
                {
                    "key": specs[args.array_task_id].key,
                    "test_ba": payload["native_metrics"]["test"][
                        "balanced_accuracy"
                    ],
                },
                indent=2,
            )
        )
        return

    if args.command == "run-long-e2e":
        specs = long_e2e_specs()
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        payload = run_train(
            specs[args.array_task_id],
            config,
            force=args.force,
        )
        print(
            json.dumps(
                {
                    "key": specs[args.array_task_id].key,
                    "test_ba": payload["native_metrics"]["test"][
                        "balanced_accuracy"
                    ],
                },
                indent=2,
            )
        )
        return

    if args.command == "run-replay":
        specs = replay_specs()
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        payload = run_replay(
            specs[args.array_task_id],
            config,
            force=args.force,
        )
        print(
            json.dumps(
                {
                    "key": specs[args.array_task_id].key,
                    "test_ba": payload["native_metrics"]["test"][
                        "balanced_accuracy"
                    ],
                },
                indent=2,
            )
        )
        return

    if args.command == "finalize-stage-a":
        print(
            json.dumps(
                finalize_stage_a(config),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "finalize":
        print(
            json.dumps(
                finalize(config),
                indent=2,
                sort_keys=True,
            )
        )
        return

    raise ValueError(args.command)


if __name__ == "__main__":
    main()
