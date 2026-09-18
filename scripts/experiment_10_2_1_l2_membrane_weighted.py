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
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_7_3_5_hidden_state_information_loss as exp735
from scripts import experiment_8_1_hidden_quantization_ablation as exp81
from scripts import experiment_8_1_1_two_layer_mt_factorial as exp811
from scripts import experiment_9_0_within_user_generalization as exp90
from scripts import experiment_10_0_airborne_motion_ablation as exp10
from scripts import experiment_10_1_a2_backbone_coding_supervision as exp101
from scripts import experiment_10_2_l1_membrane_memory as exp102


EXPERIMENT_ID = "experiment_10_2_1_l2_membrane_weighted"
PROTOCOL_VERSION = "d1_l1mem2_l2mem123_binary_weighted31_v1"

VARIANT = exp101.VARIANT_POSTENCODE
ROTATION = 0
MODEL_SEEDS = (11, 23, 37)

CODING_BINARY = "binary"
CODING_WEIGHTED = "weighted31"
CODINGS = (CODING_BINARY, CODING_WEIGHTED)
COMMUNICATION_CAP = {
    CODING_BINARY: 1,
    CODING_WEIGHTED: exp81.WEIGHTED_CAP,
}

L1_MEM_SHIFT = 2
L2_MEM_SHIFTS = (1, 2, 3)
WIDTH = exp101.WIDTH
SYN_SHIFTS = exp101.SHIFTS
OBJECTIVE = exp101.OBJECTIVE_BASELINE

MAX_EPOCHS = exp101.MAX_EPOCHS
MIN_EPOCHS = exp101.MIN_EPOCHS
PATIENCE = exp101.PATIENCE
BATCH_SIZE = exp101.BATCH_SIZE
SPLITS = exp101.SPLITS
HIDDEN_LAYERS = exp101.HIDDEN_LAYERS
HIDDEN_STATES = exp101.HIDDEN_STATES
EXPECTED_RUNS = len(CODINGS) * len(L2_MEM_SHIFTS) * len(MODEL_SEEDS)


@dataclass(frozen=True)
class RunSpec:
    coding: str
    l2_mem_shift: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.coding}__l1mem{L1_MEM_SHIFT}__l2mem{self.l2_mem_shift}__seed{self.seed}"

    @property
    def variant(self) -> str:
        return VARIANT

    @property
    def rotation(self) -> int:
        return ROTATION

    @property
    def cap(self) -> int:
        return COMMUNICATION_CAP[self.coding]


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    threads: int = 1
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp102.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(coding, l2_mem_shift, seed)
        for coding in CODINGS
        for l2_mem_shift in L2_MEM_SHIFTS
        for seed in MODEL_SEEDS
    ]


def validate_spec(spec: RunSpec) -> None:
    if spec.coding not in CODINGS:
        raise ValueError(spec.coding)
    if spec.l2_mem_shift not in L2_MEM_SHIFTS:
        raise ValueError(spec.l2_mem_shift)
    if spec.seed not in MODEL_SEEDS:
        raise ValueError(spec.seed)


def beta_from_mem_shift(shift: int, fs: float = exp72.EXPECTED_FS) -> float:
    return exp102.beta_from_mem_shift(shift, fs)


def tau_mem_ms_from_shift(shift: int, fs: float = exp72.EXPECTED_FS) -> float:
    return exp102.tau_mem_ms_from_shift(shift, fs)


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
    loaded, manifest, labels = exp101._load_variant_manifest(config.repo_root, VARIANT)
    del loaded

    config.results_dir.mkdir(parents=True, exist_ok=True)
    split_root = config.results_dir / "split"
    split_root.mkdir(parents=True, exist_ok=True)

    canonical = (
        manifest.drop(columns=["pi", "si"])
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    canonical.to_csv(config.results_dir / "canonical_sample_manifest.csv", index=False)

    assignment = exp90._assign_cross_user_folds(manifest)
    assignment.to_csv(_fold_assignment_path(config), index=False)
    split_manifest = exp90._apply_rotation(assignment, ROTATION)
    split_manifest.drop(columns=["pi", "si"]).to_csv(
        _rotation_manifest_path(config), index=False
    )

    roles = exp90._rotation_fold_roles(ROTATION)
    counts = split_manifest.split.value_counts()
    users = {
        split: sorted(
            split_manifest.loc[split_manifest.split == split, "user"]
            .astype(str)
            .unique()
            .tolist()
        )
        for split in SPLITS
    }

    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "With the Exp10.2 best L1 membrane shift fixed at 2, test whether "
            "L2 also benefits from longer membrane memory and whether integer "
            "weighted hidden communication preserves more analog evidence than "
            "binary communication."
        ),
        "dataset": "D1 postencode_mask only",
        "variant": VARIANT,
        "rotation": ROTATION,
        "fold_roles": {str(k): v for k, v in roles.items()},
        "split_users": users,
        "split_samples": {split: int(counts.get(split, 0)) for split in SPLITS},
        "labels": list(labels),
        "architecture": "30->128->128->12",
        "synaptic_shifts": [list(SYN_SHIFTS), list(SYN_SHIFTS)],
        "objective": "L2 valid-mean time-shared WCCE",
        "readout": "time-shared only; no phase-aware readout",
        "l1_mem_shift": L1_MEM_SHIFT,
        "l1_beta": beta_from_mem_shift(L1_MEM_SHIFT),
        "l1_tau_mem_ms": tau_mem_ms_from_shift(L1_MEM_SHIFT),
        "l2_mem_shifts": list(L2_MEM_SHIFTS),
        "l2_beta": {
            str(shift): beta_from_mem_shift(shift) for shift in L2_MEM_SHIFTS
        },
        "l2_tau_mem_ms": {
            str(shift): tau_mem_ms_from_shift(shift) for shift in L2_MEM_SHIFTS
        },
        "codings": {
            CODING_BINARY: {
                "l1_cap": 1,
                "l2_cap": 1,
                "communication": "binary event count in {0,1}",
            },
            CODING_WEIGHTED: {
                "l1_cap": exp81.WEIGHTED_CAP,
                "l2_cap": exp81.WEIGHTED_CAP,
                "communication": (
                    "integer macro-step event count clip(floor(max(U_pre,0)/theta),"
                    "0,31); reset subtracts count*theta"
                ),
            },
        },
        "weighted_normalization": "none; raw integer communication counts enter downstream weights",
        "model_seeds": list(MODEL_SEEDS),
        "expected_runs": EXPECTED_RUNS,
        "paired_randomness": (
            "For a fixed seed, model initialization and DataLoader ordering are "
            "identical across coding and L2 membrane conditions."
        ),
        "primary_contrasts": [
            "binary: L2 shift2-shift1 and shift3-shift1",
            "weighted31: L2 shift2-shift1 and shift3-shift1",
            "weighted31-binary at each L2 shift",
            "coding x L2-memory interactions",
        ],
    }
    _save_json(config.results_dir / "audit.json", audit)
    return audit


def _prepare_data(
    config: Config,
) -> tuple[exp3.Data, dict[str, pd.DataFrame], pd.DataFrame]:
    loaded, manifest, labels = exp101._load_variant_manifest(config.repo_root, VARIANT)
    canonical_path = config.results_dir / "canonical_sample_manifest.csv"
    if not canonical_path.exists():
        raise FileNotFoundError(f"Missing {canonical_path}; run prepare first")
    canonical = pd.read_csv(canonical_path)
    exp10._assert_paired_geometry(
        canonical,
        manifest,
        reference_name="prepared_canonical",
        candidate_name=VARIANT,
    )
    saved = pd.read_csv(_fold_assignment_path(config), usecols=["sample_id", "cv_fold"])
    assignment = manifest.merge(saved, on="sample_id", how="inner", validate="one_to_one")
    if len(assignment) != len(manifest):
        raise RuntimeError("Prepared fold assignment does not match current D1 dataset")
    assignment["cv_fold"] = assignment.cv_fold.astype(int)
    split_manifest = exp90._apply_rotation(assignment, ROTATION)
    data, frames = exp90._to_data(loaded, split_manifest, labels)
    return data, frames, split_manifest


class Exp1021Net(nn.Module):
    """Exp10.2 backbone with L1 shift2 and controlled L2 memory/coding."""

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

        self.hidden_linears = nn.ModuleList(
            [
                nn.Linear(exp72.EXPECTED_CHANNELS, WIDTH, bias=False),
                nn.Linear(WIDTH, WIDTH, bias=False),
            ]
        )
        self.output_linear = nn.Linear(WIDTH, n_classes, bias=False)

        cap = spec.cap
        self.l1_lif = exp401.MacroMultiSpikeLIF(
            beta=beta_from_mem_shift(L1_MEM_SHIFT, fs),
            threshold=float(exp73.THRESHOLD),
            max_spikes_per_dt=cap,
            surrogate_slope=exp72.SURROGATE_SLOPE,
        )
        self.l2_lif = exp401.MacroMultiSpikeLIF(
            beta=beta_from_mem_shift(spec.l2_mem_shift, fs),
            threshold=float(exp73.THRESHOLD),
            max_spikes_per_dt=cap,
            surrogate_slope=exp72.SURROGATE_SLOPE,
        )
        # Coding changes only the event cap. Synaptic time constants are paired.
        self.register_buffer("alpha_0", exp811._alpha_vector("binary"))
        self.register_buffer("alpha_1", exp811._alpha_vector("binary"))

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
        }


def _valid_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return exp101._valid_mask(lengths, steps)


def _valid_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return exp101._valid_mean(values, lengths)


def _native_scores(
    model: Exp1021Net,
    X: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    trajectory = model.forward_trajectory(X)
    return _valid_mean(trajectory["l2_evidence"], lengths)


def _evaluate_native(
    model: Exp1021Net,
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
    model: Exp1021Net,
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
            evidence = model.forward_trajectory(Xd)["l2_evidence"]
            spikes = exp81._output_lif_spikes(evidence)
            mask = _valid_mask(ld, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
            scores = (spikes * mask).sum(dim=1)
            ys.append(y.numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
    return exp10._classification_metrics(
        np.concatenate(ys), np.concatenate(preds)
    )


def _split_hashes(frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    return exp101._split_hashes(frames)


def _evidence_scale_metrics(
    model: Exp1021Net,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    evidence_abs_sum = 0.0
    evidence_l2_sum = 0.0
    valid_step_count = 0.0
    score_l2_sum = 0.0
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for X, _, lengths in loader:
            Xd = X.to(device=device, dtype=torch.float32)
            ld = lengths.to(device)
            evidence = model.forward_trajectory(Xd)["l2_evidence"]
            mask = _valid_mask(ld, evidence.shape[1])
            valid_evidence = evidence[mask]
            if valid_evidence.numel():
                evidence_abs_sum += float(valid_evidence.abs().mean(dim=1).sum().item())
                evidence_l2_sum += float(valid_evidence.norm(dim=1).sum().item())
                valid_step_count += float(valid_evidence.shape[0])
            scores = _valid_mean(evidence, ld)
            score_l2_sum += float(scores.norm(dim=1).sum().item())
            sample_count += int(scores.shape[0])

    weight = model.output_linear.weight.detach()
    return {
        "output_weight_fro_norm": float(weight.norm().item()),
        "output_weight_mean_abs": float(weight.abs().mean().item()),
        "mean_abs_evidence_per_valid_step": (
            evidence_abs_sum / max(valid_step_count, 1.0)
        ),
        "mean_l2_evidence_norm_per_valid_step": (
            evidence_l2_sum / max(valid_step_count, 1.0)
        ),
        "mean_native_score_l2_norm": (
            score_l2_sum / max(sample_count, 1)
        ),
    }


def _collect_features_and_activity(
    model: Exp1021Net,
    data: exp3.Data,
    frames: dict[str, pd.DataFrame],
    spec: RunSpec,
    config: Config,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, list[dict[str, float | int]]],
]:
    device = torch.device(config.device)
    loaders = exp73._raw_loaders(data, spec.seed, config.batch_size, False)
    names = exp10.probe_names()
    feature_parts: dict[str, dict[str, list[np.ndarray]]] = {
        split: {name: [] for name in names} for split in SPLITS
    }
    label_parts: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    test_comm: dict[str, list[torch.Tensor]] = {"l1": [], "l2": []}
    test_lengths: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            for X, y, lengths in loaders[split]:
                Xd = X.to(device=device, dtype=torch.float32)
                ld = lengths.to(device=device, dtype=torch.long)
                trajectory = model.forward_trajectory(Xd)
                hidden = trajectory["hidden"]

                feature_parts[split]["input__events__whole_count"].append(
                    exp10._masked_sum(Xd, ld).cpu().numpy().astype(np.float32, copy=False)
                )
                feature_parts[split]["input__events__fixed250_count"].append(
                    exp3.fixed_counts(Xd, ld, data.bin_steps)
                    .flatten(start_dim=1)
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

                for layer in HIDDEN_LAYERS:
                    for state in HIDDEN_STATES:
                        values = hidden[layer][state]
                        for aggregation in exp10.STATE_AGGREGATIONS:
                            key = f"{layer}__{state}__{aggregation}"
                            aggregated = exp735._aggregate_tensor(
                                values, ld, aggregation, data.bin_steps
                            )
                            feature_parts[split][key].append(
                                aggregated.cpu().numpy().astype(np.float32, copy=False)
                            )

                    communication = hidden[layer]["spike"]
                    feature_parts[split][f"{layer}__spike__whole_count"].append(
                        exp10._masked_sum(communication, ld)
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    feature_parts[split][f"{layer}__spike__fixed250_count"].append(
                        exp3.fixed_counts(communication, ld, data.bin_steps)
                        .flatten(start_dim=1)
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    if split == "test":
                        test_comm[layer].append(communication.cpu())

                if split == "test":
                    test_lengths.append(lengths.numpy().astype(np.int64, copy=False))
                label_parts[split].append(y.numpy().astype(np.int64, copy=False))

    features = {
        split: {
            name: np.concatenate(parts, axis=0)
            for name, parts in feature_parts[split].items()
        }
        for split in SPLITS
    }
    labels = {
        split: np.concatenate(label_parts[split], axis=0)
        for split in SPLITS
    }

    activity = _communication_activity(
        {
            layer: torch.cat(test_comm[layer], dim=0)
            for layer in HIDDEN_LAYERS
        },
        np.concatenate(test_lengths, axis=0),
        spec,
        data.fs,
    )
    return features, labels, activity


def _communication_activity(
    communications: dict[str, torch.Tensor],
    lengths: np.ndarray,
    spec: RunSpec,
    fs: float,
) -> dict[str, list[dict[str, float | int]]]:
    lengths_t = torch.as_tensor(lengths, dtype=torch.long)
    output: dict[str, list[dict[str, float | int]]] = {}

    for layer in HIDDEN_LAYERS:
        values = communications[layer]
        mask = _valid_mask(lengths_t, values.shape[1]).to(values.dtype).unsqueeze(-1)
        valid_steps = float(mask[:, :, 0].sum().item())
        rows: list[dict[str, float | int]] = []
        for group in exp72.shift_groups(SYN_SHIFTS, width=WIDTH):
            start, stop = int(group["start"]), int(group["stop"])
            chunk = values[:, :, start:stop]
            valid = chunk * mask
            counts_per_neuron = valid.sum(dim=(0, 1))
            denom_neuron_steps = max(valid_steps * float(stop - start), 1.0)
            nonzero_events = ((chunk > 0).to(values.dtype) * mask).sum().item()
            multi_events = ((chunk >= 2).to(values.dtype) * mask).sum().item()
            large_events = ((chunk >= 4).to(values.dtype) * mask).sum().item()
            cap_events = ((chunk >= spec.cap).to(values.dtype) * mask).sum().item()
            mean_count = float(valid.sum().item() / denom_neuron_steps)

            rows.append(
                {
                    "shift": int(group["shift"]),
                    "start": start,
                    "stop": stop,
                    "count": int(group["count"]),
                    "communication_cap": int(spec.cap),
                    "mean_count_per_neuron_step": mean_count,
                    "mean_events_per_neuron_s": mean_count * float(fs),
                    "nonzero_communication_fraction": float(
                        nonzero_events / denom_neuron_steps
                    ),
                    "multi_event_fraction_ge_2": float(
                        multi_events / denom_neuron_steps
                    ),
                    "large_event_fraction_ge_4": float(
                        large_events / denom_neuron_steps
                    ),
                    "cap_hit_fraction": float(cap_events / denom_neuron_steps),
                    "dead_neuron_fraction": float(
                        (counts_per_neuron == 0).float().mean().item()
                    ),
                }
            )
        output[layer] = rows
    return output


def _fit_probes(
    spec: RunSpec,
    features: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    test_actions: np.ndarray,
) -> pd.DataFrame:
    frame = exp10._fit_all_probes(spec, features, labels, test_actions)
    frame.insert(1, "coding", spec.coding)
    frame.insert(2, "l1_mem_shift", L1_MEM_SHIFT)
    frame.insert(3, "l2_mem_shift", spec.l2_mem_shift)
    return frame


def _run_artifacts(config: Config, spec: RunSpec) -> dict[str, Path]:
    return {
        "checkpoint": _path(config.results_dir, "checkpoints", spec.key, ".pt"),
        "history": _path(config.results_dir, "histories", spec.key, ".csv"),
        "evaluation": _path(config.results_dir, "evaluations", spec.key, ".json"),
        "probes": _path(config.results_dir, "probe_evaluations", spec.key, ".csv"),
    }


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
    split_hashes = _split_hashes(frames)
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)

    model_init_seed = exp73._e2e_pair_seed(spec.seed, "model_init")
    exp3.seed_all(model_init_seed)
    model = Exp1021Net(spec, len(data.labels), data.fs).to(device)

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
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )

        if exp73._checkpoint_improved(val_metrics, best_ba, best_loss):
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
            "l1_mem_shift": L1_MEM_SHIFT,
            "l2_mem_shift": spec.l2_mem_shift,
            "communication_cap": spec.cap,
            "model_state_dict": best_state,
        },
        artifacts["checkpoint"],
    )
    artifacts["history"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(artifacts["history"], index=False)

    model.load_state_dict(best_state, strict=True)
    native_metrics: dict[str, dict[str, float]] = {}
    lif_metrics: dict[str, dict[str, float]] = {}
    native_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        metrics, y_true, y_pred = _evaluate_native(
            model, eval_loaders[split], device
        )
        native_metrics[split] = metrics
        native_arrays[split] = (y_true, y_pred)
        lif_metrics[split] = _evaluate_lif_transfer(
            model, eval_loaders[split], device
        )

    features, probe_labels, activity = _collect_features_and_activity(
        model, data, frames, spec, config
    )
    for split in SPLITS:
        if not np.array_equal(probe_labels[split], native_arrays[split][0]):
            raise RuntimeError(f"{spec.key}/{split}: probe label order mismatch")

    test_actions = frames["test"].action.to_numpy(dtype=np.int64, copy=True)
    probes = _fit_probes(spec, features, probe_labels, test_actions)
    artifacts["probes"].parent.mkdir(parents=True, exist_ok=True)
    probes.to_csv(artifacts["probes"], index=False)

    scale_metrics = _evidence_scale_metrics(
        model, eval_loaders["test"], device
    )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "dataset": VARIANT,
            "architecture": "30->128->128->12",
            "coding": spec.coding,
            "l1_communication_cap": spec.cap,
            "l2_communication_cap": spec.cap,
            "l1_mem_shift": L1_MEM_SHIFT,
            "l1_beta": beta_from_mem_shift(L1_MEM_SHIFT, data.fs),
            "l1_tau_mem_ms": tau_mem_ms_from_shift(L1_MEM_SHIFT, data.fs),
            "l2_mem_shift": spec.l2_mem_shift,
            "l2_beta": beta_from_mem_shift(spec.l2_mem_shift, data.fs),
            "l2_tau_mem_ms": tau_mem_ms_from_shift(spec.l2_mem_shift, data.fs),
            "synaptic_shifts": [list(SYN_SHIFTS), list(SYN_SHIFTS)],
            "objective": OBJECTIVE,
            "readout": "time_shared",
            "weighted_count_normalization": "none",
        },
        "model_init_seed": int(model_init_seed),
        "split_sample_hashes": split_hashes,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native_metrics,
        "lif_transfer_metrics": lif_metrics,
        "scale_metrics": scale_metrics,
        "activity": activity,
        "probe_count": int(len(probes)),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    _save_json(artifacts["evaluation"], payload)
    return payload


def _probe_value(frame: pd.DataFrame, name: str) -> float:
    row = frame[frame.probe == name]
    if len(row) != 1:
        raise RuntimeError(f"Expected one row for {name}, got {len(row)}")
    return float(row.iloc[0].test_balanced_accuracy)


def _run_row(payload: Mapping[str, Any], probes: pd.DataFrame) -> dict[str, Any]:
    spec = payload["spec"]
    l1_pre = _probe_value(probes, "l1__pre_reset__fixed250_ordered_mean")
    l1_comm = _probe_value(probes, "l1__spike__fixed250_count")
    l2_pre = _probe_value(probes, "l2__pre_reset__fixed250_ordered_mean")
    l2_comm = _probe_value(probes, "l2__spike__fixed250_count")
    l2_whole = _probe_value(probes, "l2__spike__whole_count")

    row = {
        "coding": spec["coding"],
        "l1_mem_shift": L1_MEM_SHIFT,
        "l2_mem_shift": int(spec["l2_mem_shift"]),
        "l1_tau_mem_ms": tau_mem_ms_from_shift(L1_MEM_SHIFT),
        "l2_tau_mem_ms": tau_mem_ms_from_shift(int(spec["l2_mem_shift"])),
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
        "lif_test_ba": float(
            payload["lif_transfer_metrics"]["test"]["balanced_accuracy"]
        ),
        "l1_pre_reset_fixed250_ba": l1_pre,
        "l1_communication_fixed250_ba": l1_comm,
        "l1_communication_delta": l1_comm - l1_pre,
        "l2_pre_reset_fixed250_ba": l2_pre,
        "l2_communication_fixed250_ba": l2_comm,
        "l2_communication_delta": l2_comm - l2_pre,
        "l2_communication_whole_ba": l2_whole,
        "l2_temporal_ordering_gain": l2_comm - l2_whole,
        "output_weight_fro_norm": float(
            payload["scale_metrics"]["output_weight_fro_norm"]
        ),
        "output_weight_mean_abs": float(
            payload["scale_metrics"]["output_weight_mean_abs"]
        ),
        "mean_abs_evidence_per_valid_step": float(
            payload["scale_metrics"]["mean_abs_evidence_per_valid_step"]
        ),
        "mean_l2_evidence_norm_per_valid_step": float(
            payload["scale_metrics"]["mean_l2_evidence_norm_per_valid_step"]
        ),
        "mean_native_score_l2_norm": float(
            payload["scale_metrics"]["mean_native_score_l2_norm"]
        ),
    }
    return row


def _activity_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    rows: list[dict[str, Any]] = []
    for layer, groups in payload["activity"].items():
        for group in groups:
            rows.append(
                {
                    "coding": spec["coding"],
                    "l2_mem_shift": int(spec["l2_mem_shift"]),
                    "seed": int(spec["seed"]),
                    "layer": layer,
                    **group,
                }
            )
    return rows


def _paired_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "native_test_ba",
        "lif_test_ba",
        "l1_pre_reset_fixed250_ba",
        "l1_communication_fixed250_ba",
        "l1_communication_delta",
        "l2_pre_reset_fixed250_ba",
        "l2_communication_fixed250_ba",
        "l2_communication_delta",
        "l2_communication_whole_ba",
        "l2_temporal_ordering_gain",
        "output_weight_fro_norm",
        "output_weight_mean_abs",
        "mean_abs_evidence_per_valid_step",
        "mean_l2_evidence_norm_per_valid_step",
        "mean_native_score_l2_norm",
    ]
    indexed = runs.set_index(["coding", "l2_mem_shift", "seed"])
    rows: list[dict[str, Any]] = []

    def emit(
        contrast: str,
        left: tuple[str, int, int],
        right: tuple[str, int, int],
    ) -> None:
        lrow = indexed.loc[left]
        rrow = indexed.loc[right]
        row: dict[str, Any] = {
            "contrast": contrast,
            "coding": left[0] if left[0] == right[0] else "weighted31_vs_binary",
            "l2_mem_shift": left[1] if left[1] == right[1] else left[1],
            "seed": left[2],
        }
        for metric in metrics:
            row[f"delta_{metric}"] = float(lrow[metric] - rrow[metric])
        rows.append(row)

    for seed in MODEL_SEEDS:
        for coding in CODINGS:
            for shift in (2, 3):
                emit(
                    f"l2mem{shift}_minus_l2mem1",
                    (coding, shift, seed),
                    (coding, 1, seed),
                )
        for shift in L2_MEM_SHIFTS:
            emit(
                "weighted31_minus_binary",
                (CODING_WEIGHTED, shift, seed),
                (CODING_BINARY, shift, seed),
            )

    return pd.DataFrame(rows)


def _interaction_rows(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "native_test_ba",
        "l1_communication_fixed250_ba",
        "l2_communication_fixed250_ba",
        "l2_communication_whole_ba",
        "l2_temporal_ordering_gain",
        "output_weight_fro_norm",
        "mean_abs_evidence_per_valid_step",
        "mean_native_score_l2_norm",
    ]
    indexed = runs.set_index(["coding", "l2_mem_shift", "seed"])
    rows: list[dict[str, Any]] = []

    for seed in MODEL_SEEDS:
        for shift in (2, 3):
            row: dict[str, Any] = {
                "interaction": f"coding_x_l2mem{shift}",
                "l2_mem_shift": shift,
                "seed": seed,
            }
            for metric in metrics:
                weighted_mem_effect = float(
                    indexed.loc[(CODING_WEIGHTED, shift, seed), metric]
                    - indexed.loc[(CODING_WEIGHTED, 1, seed), metric]
                )
                binary_mem_effect = float(
                    indexed.loc[(CODING_BINARY, shift, seed), metric]
                    - indexed.loc[(CODING_BINARY, 1, seed), metric]
                )
                row[f"interaction_{metric}"] = (
                    weighted_mem_effect - binary_mem_effect
                )
            rows.append(row)
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
                json.loads(artifacts["evaluation"].read_text(encoding="utf-8"))
            )
        if artifacts["probes"].exists():
            probe_frames.append(pd.read_csv(artifacts["probes"]))

    if missing:
        raise FileNotFoundError(
            f"Exp10.2.1 incomplete; missing {len(missing)} artifacts:\n"
            + "\n".join(missing[:40])
        )
    if len(payloads) != EXPECTED_RUNS or len(probe_frames) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} runs, got {len(payloads)} evaluations "
            f"and {len(probe_frames)} probe files"
        )

    runs = pd.DataFrame(
        [
            _run_row(payload, probes)
            for payload, probes in zip(payloads, probe_frames, strict=True)
        ]
    ).sort_values(["coding", "l2_mem_shift", "seed"])
    runs.to_csv(config.results_dir / "run_metrics.csv", index=False)

    metric_cols = [
        "native_train_ba",
        "native_val_ba",
        "native_test_ba",
        "lif_test_ba",
        "l1_pre_reset_fixed250_ba",
        "l1_communication_fixed250_ba",
        "l1_communication_delta",
        "l2_pre_reset_fixed250_ba",
        "l2_communication_fixed250_ba",
        "l2_communication_delta",
        "l2_communication_whole_ba",
        "l2_temporal_ordering_gain",
    ]
    summary = (
        runs.groupby(
            ["coding", "l2_mem_shift", "l2_tau_mem_ms"], sort=True
        )[metric_cols]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    summary.to_csv(config.results_dir / "method_summary.csv", index=False)

    contrasts = _paired_contrasts(runs)
    contrasts.to_csv(config.results_dir / "paired_contrasts.csv", index=False)
    delta_cols = [c for c in contrasts.columns if c.startswith("delta_")]
    contrast_summary = (
        contrasts.groupby(
            ["contrast", "coding", "l2_mem_shift"], sort=False
        )[delta_cols]
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
        config.results_dir / "paired_contrast_summary.csv", index=False
    )

    interactions = _interaction_rows(runs)
    interactions.to_csv(config.results_dir / "interaction_runs.csv", index=False)
    interaction_cols = [
        c for c in interactions.columns if c.startswith("interaction_")
    ]
    interaction_summary = (
        interactions.groupby(["interaction", "l2_mem_shift"], sort=True)[
            interaction_cols
        ]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    interaction_summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in interaction_summary.columns
    ]
    interaction_summary.to_csv(
        config.results_dir / "interaction_summary.csv", index=False
    )

    activity = pd.DataFrame(
        [
            row
            for payload in payloads
            for row in _activity_rows(payload)
        ]
    )
    activity.to_csv(config.results_dir / "activity_runs.csv", index=False)
    activity_metrics = [
        "mean_count_per_neuron_step",
        "mean_events_per_neuron_s",
        "nonzero_communication_fraction",
        "multi_event_fraction_ge_2",
        "large_event_fraction_ge_4",
        "cap_hit_fraction",
        "dead_neuron_fraction",
    ]
    activity_summary = (
        activity.groupby(
            ["coding", "l2_mem_shift", "layer", "shift"], sort=True
        )[activity_metrics]
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
        config.results_dir / "activity_summary.csv", index=False
    )

    all_probes = pd.concat(probe_frames, ignore_index=True)
    all_probes.to_csv(config.results_dir / "probe_runs.csv", index=False)
    probe_summary = (
        all_probes.groupby(
            ["coding", "l2_mem_shift", "probe"], sort=False
        )["test_balanced_accuracy"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    probe_summary.to_csv(config.results_dir / "probe_summary.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS",
        "dataset": VARIANT,
        "rotation": ROTATION,
        "l1_mem_shift": L1_MEM_SHIFT,
        "l2_mem_shifts": list(L2_MEM_SHIFTS),
        "codings": list(CODINGS),
        "model_seeds": list(MODEL_SEEDS),
        "run_count": int(len(runs)),
        "expected_run_count": EXPECTED_RUNS,
        "readout": "time_shared_only",
        "phase_aware_readout": False,
        "weighted_cap": int(exp81.WEIGHTED_CAP),
        "statistical_scope": (
            "one locked cross-user split; seeds 11/23/37 are paired "
            "optimization replicates, not independent user splits"
        ),
        "primary_outputs": [
            "run_metrics.csv",
            "method_summary.csv",
            "paired_contrasts.csv",
            "paired_contrast_summary.csv",
            "interaction_runs.csv",
            "interaction_summary.csv",
            "activity_runs.csv",
            "activity_summary.csv",
            "probe_runs.csv",
            "probe_summary.csv",
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
            "Exp10.2.1: D1 L1-mem2 fixed, L2 membrane 1/2/3 x "
            "binary/weighted31 hidden communication"
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
