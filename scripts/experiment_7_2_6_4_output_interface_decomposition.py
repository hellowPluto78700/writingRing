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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_2_6_output_readout_loss_shaping as exp726
from scripts import experiment_7_2_6_1_sum_vs_mean_ce as exp7261
from scripts import experiment_7_2_6_3_linear_vs_lif_frozen_l2 as exp7263


EXPERIMENT_ID = "experiment_7_2_6_4_output_interface_decomposition"
PROTOCOL_VERSION = "output_interface_decomposition_v1"
ARCHITECTURE = exp726.ARCHITECTURE
REGULARIZATION = exp72.TASK_ONLY
SEEDS = tuple(exp726.SEEDS)
HIDDEN_WIDTH = exp726.HIDDEN_WIDTH
N_CLASSES = exp726.N_CLASSES
THRESHOLD = float(exp726.THRESHOLD)
OUTPUT_CAP = 1
OUTPUT_ALPHA = 0.0
LIF_BETA = 0.5
MAX_EPOCHS = exp726.MAX_EPOCHS
MIN_EPOCHS = exp726.MIN_EPOCHS
PATIENCE = exp726.PATIENCE

FLUSH_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
SOFTMAX_TEMPS = (0.25, 0.5, 1.0, 2.0)
C_OBJECTIVES = ("raw_wcce", "softmax_vote_wcce", "tsce")
D_HEAD_TYPES = ("linear", "lif")
CE_GAINS = (1.0, 2.0, 5.0, 10.0)
D_LR_CONTROL_GAIN = 5.0
D_LR_CONTROL_SCALE = 0.2


@dataclass(frozen=True)
class SourceSpec:
    seed: int

    @property
    def baseline_spec(self) -> exp7261.RunSpec:
        return exp7261.RunSpec(self.seed)

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__{REGULARIZATION}__seed{self.seed}"


@dataclass(frozen=True)
class CSpec:
    seed: int
    objective: str

    @property
    def source(self) -> SourceSpec:
        return SourceSpec(self.seed)

    @property
    def key(self) -> str:
        return f"{self.source.key}__C_{self.objective}"


@dataclass(frozen=True)
class DSpec:
    seed: int
    head_type: str
    ce_gain: float
    lr_scale: float = 1.0

    @property
    def source(self) -> SourceSpec:
        return SourceSpec(self.seed)

    @property
    def key(self) -> str:
        gain = f"{self.ce_gain:g}".replace(".", "p")
        lr = f"{self.lr_scale:g}".replace(".", "p")
        return f"{self.source.key}__D_{self.head_type}__g{gain}__lr{lr}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp726.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_specs() -> list[SourceSpec]:
    return [SourceSpec(seed) for seed in SEEDS]


def c_specs() -> list[CSpec]:
    return [CSpec(seed, objective) for seed in SEEDS for objective in C_OBJECTIVES]


def d_specs() -> list[DSpec]:
    specs = [
        DSpec(seed, head_type, gain, 1.0)
        for seed in SEEDS
        for head_type in D_HEAD_TYPES
        for gain in CE_GAINS
    ]
    specs.extend(DSpec(seed, "lif", D_LR_CONTROL_GAIN, D_LR_CONTROL_SCALE) for seed in SEEDS)
    return specs


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _source_config(config: Config) -> exp7261.Config:
    return exp7261.Config(
        config.repo_root,
        exp7261.results_dir(config.repo_root),
        config.device,
        config.batch_size,
        config.threads,
        config.max_epochs,
    )


def _pair_seed(seed: int, role: str) -> int:
    # Reuse the Exp7.2.6.3 pairing seed exactly. This makes C0 Raw-WCCE and
    # D gain=1 directly reproducible against the established paired-head baseline,
    # while condition/gain remain excluded from initialization and loader order.
    return exp7263._head_pair_seed(seed, role)


def validate_source(spec: SourceSpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def validate_c(spec: CSpec) -> None:
    validate_source(spec.source)
    if spec.objective not in C_OBJECTIVES:
        raise ValueError(spec.objective)


def validate_d(spec: DSpec) -> None:
    validate_source(spec.source)
    if spec.head_type not in D_HEAD_TYPES:
        raise ValueError(spec.head_type)
    if spec.ce_gain not in CE_GAINS:
        raise ValueError(spec.ce_gain)
    if spec.lr_scale not in (1.0, D_LR_CONTROL_SCALE):
        raise ValueError(spec.lr_scale)
    if spec.lr_scale != 1.0 and not (
        spec.head_type == "lif" and spec.ce_gain == D_LR_CONTROL_GAIN
    ):
        raise ValueError("Only the LIF gain=5 LR/5 control is part of the protocol")


# -----------------------------------------------------------------------------
# Frozen L2 cache: rebuilt per Exp7.2.6.4 from the Exp7.2.6.1 source.
# -----------------------------------------------------------------------------

def _cache_path(config: Config, spec: SourceSpec) -> Path:
    return _path(config.results_dir, "frozen_l2_cache", spec.key, ".npz")


def _cache_meta_path(config: Config, spec: SourceSpec) -> Path:
    return _path(config.results_dir, "frozen_l2_cache", spec.key, ".json")


def prepare_cache(spec: SourceSpec, config: Config, force: bool = False) -> Path:
    validate_source(spec)
    destination = _cache_path(config, spec)
    metadata_path = _cache_meta_path(config, spec)
    if destination.exists() and metadata_path.exists() and not force:
        return destination

    source_checkpoint = exp7261._path(
        exp7261.results_dir(config.repo_root), "checkpoints", spec.baseline_spec, ".pt"
    )
    source_eval = exp7261._path(
        exp7261.results_dir(config.repo_root), "evaluations", spec.baseline_spec, ".json"
    )
    missing = [str(p) for p in (source_checkpoint, source_eval) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Exp7.2.6.4 requires Exp7.2.6.1 Mean-CE bias-free artifacts. "
            f"Missing for seed {spec.seed}: {missing}"
        )

    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model, checkpoint = exp7261._load_mean_model(spec.baseline_spec, data, _source_config(config))
    if model.output_linear.bias is not None:
        raise RuntimeError("Exp7.2.6.4 source must be the bias-free Exp7.2.6.1 model")
    loaders = exp726._e2e_loaders(data, spec.baseline_spec.c0_spec, config.batch_size, False)

    arrays: dict[str, np.ndarray] = {
        "source_w": model.output_linear.weight.detach().cpu().numpy().astype(np.float32)
    }
    split_meta: dict[str, Any] = {}
    for split, loader in loaders.items():
        l2, y, lengths = exp726._extract_l2_generic(model, loader, device)
        l2_np = l2.numpy()
        if not np.all((l2_np == 0) | (l2_np == 1)):
            raise RuntimeError(f"{spec.key}/{split}: L2 cache must be binary")
        arrays[f"{split}_l2"] = l2_np.astype(np.uint8, copy=False)
        arrays[f"{split}_y"] = y.astype(np.int64, copy=False)
        arrays[f"{split}_lengths"] = lengths.astype(np.int64, copy=False)
        split_meta[split] = {
            "shape": list(l2_np.shape),
            "n_samples": int(len(y)),
            "mean_firing_fraction_full": float(l2_np.mean()),
            "mean_valid_length": float(lengths.mean()),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    _save_json(
        metadata_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "source_experiment": exp7261.EXPERIMENT_ID,
            "source_protocol": exp7261.PROTOCOL_VERSION,
            "source_checkpoint": str(source_checkpoint.relative_to(config.repo_root)),
            "source_best_epoch": int(checkpoint["best_epoch"]),
            "source_contract": "Exp7.2.6.1 Mean-CE, bias=False, task_only",
            "frozen_l1_l2": True,
            "splits": split_meta,
        },
    )
    return destination


def _load_cache(spec: SourceSpec, config: Config) -> dict[str, Any]:
    path = _cache_path(config, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp7.2.6.4 cache: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {
            "source_w": z["source_w"].astype(np.float64),
            **{
                split: (
                    z[f"{split}_l2"].copy(),
                    z[f"{split}_y"].copy(),
                    z[f"{split}_lengths"].copy(),
                )
                for split in ("train", "val", "test")
            },
        }


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return exp72._metrics(y, scores.argmax(axis=1))


def _masked_evidence(
    l2: np.ndarray, lengths: np.ndarray, W: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    evidence = l2.astype(np.float64) @ W.T
    mask = np.arange(l2.shape[1])[None, :] < lengths[:, None]
    return evidence, mask


def _softmax_np(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(temperature)
    shifted = values / float(temperature)
    shifted = shifted - shifted.max(axis=-1, keepdims=True)
    expv = np.exp(shifted)
    return expv / expv.sum(axis=-1, keepdims=True)


def _masked_sum(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (values * mask[..., None]).sum(axis=1)


def _softmax_vote_scores(
    l2: np.ndarray,
    lengths: np.ndarray,
    W: np.ndarray,
    temperature: float,
) -> np.ndarray:
    evidence, mask = _masked_evidence(l2, lengths, W)
    return _masked_sum(_softmax_np(evidence, temperature), mask)


def _positive_only_scores(
    l2: np.ndarray,
    lengths: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    evidence, mask = _masked_evidence(l2, lengths, W)
    return _masked_sum(np.maximum(evidence, 0.0), mask)


# -----------------------------------------------------------------------------
# Part A: output bandwidth and residual discharge, no training.
# -----------------------------------------------------------------------------

def _flush_from_state(
    counts: np.ndarray,
    residual: np.ndarray,
    beta: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if steps < 0:
        raise ValueError(steps)
    mem = residual.astype(np.float64, copy=True)
    out = counts.astype(np.float64, copy=True)
    extra = np.zeros_like(out)
    for _ in range(int(steps)):
        pre = float(beta) * mem
        spikes = np.minimum(
            np.floor(np.maximum(pre, 0.0) / THRESHOLD),
            float(OUTPUT_CAP),
        )
        mem = pre - THRESHOLD * spikes
        out += spikes
        extra += spikes
    return out, mem, extra


def _infinite_if_counts(counts: np.ndarray, residual: np.ndarray) -> np.ndarray:
    fireable = np.floor((np.maximum(residual, 0.0) + 1e-12) / THRESHOLD)
    return counts.astype(np.float64) + fireable


def _flush_diagnostics(
    initial_residual: np.ndarray,
    final_residual: np.ndarray,
    extra: np.ndarray,
    beta: float,
) -> dict[str, float]:
    positive = np.maximum(initial_residual, 0.0)
    negative = np.minimum(initial_residual, 0.0)
    spike_charge = THRESHOLD * float(extra.sum())
    positive_charge = float(positive.sum())
    fireable_charge = THRESHOLD * float(
        np.floor((positive + 1e-12) / THRESHOLD).sum()
    )
    payload = {
        "beta": float(beta),
        "mean_abs_U_T": float(np.abs(initial_residual).mean()),
        "mean_abs_U_TK": float(np.abs(final_residual).mean()),
        "positive_residual_sum": positive_charge,
        "negative_residual_sum": float(negative.sum()),
        "extra_spikes_sum": float(extra.sum()),
        "mean_extra_spikes_per_sample": float(extra.sum(axis=1).mean()),
        "discharge_ratio_positive": (
            spike_charge / positive_charge if positive_charge > 0 else 0.0
        ),
    }
    if beta == 1.0:
        payload["fireable_discharge_ratio"] = (
            spike_charge / fireable_charge if fireable_charge > 0 else 0.0
        )
    return payload


def _zero_tail(l2: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    out = l2.copy()
    for i, length in enumerate(lengths):
        out[i, int(length):] = 0
    return out


def run_part_a(spec: SourceSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_source(spec)
    out_path = _path(config.results_dir, "part_a_evaluations", spec.key, ".json")
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    cache = _load_cache(spec, config)
    W = cache["source_w"]
    test_l2, test_y, test_lengths = cache["test"]

    analog = exp726._analog_scores(test_l2, test_lengths, W, scale=1.0)
    if1 = exp726._simulate_unipolar(
        test_l2, test_lengths, W, beta=1.0, cap=OUTPUT_CAP, scale=1.0
    )
    lif05 = exp726._simulate_unipolar(
        test_l2, test_lengths, W, beta=LIF_BETA, cap=OUTPUT_CAP, scale=1.0
    )

    references = {
        "analog": _metrics(test_y, analog),
        "if_beta1_charge": _metrics(test_y, if1["charge_scores"]),
        "if_beta1_count": _metrics(test_y, if1["counts"]),
        "lif_beta05_count": _metrics(test_y, lif05["counts"]),
    }

    flush_rows: list[dict[str, Any]] = []
    for beta, base in ((1.0, if1), (LIF_BETA, lif05)):
        for steps in FLUSH_STEPS:
            counts_k, mem_k, extra = _flush_from_state(
                base["counts"], base["residual"], beta, steps
            )
            flush_rows.append(
                {
                    "beta": float(beta),
                    "flush_steps": int(steps),
                    **_metrics(test_y, counts_k),
                    **_flush_diagnostics(
                        base["residual"], mem_k, extra, beta
                    ),
                }
            )

    infinite_counts = _infinite_if_counts(if1["counts"], if1["residual"])
    infinite_fireable = np.floor(
        (np.maximum(if1["residual"], 0.0) + 1e-12) / THRESHOLD
    )
    infinite_row = {
        "beta": 1.0,
        **_metrics(test_y, infinite_counts),
        "mean_extra_spikes_per_sample": float(
            infinite_fireable.sum(axis=1).mean()
        ),
        "positive_residual_sum": float(np.maximum(if1["residual"], 0.0).sum()),
        "negative_residual_sum": float(np.minimum(if1["residual"], 0.0).sum()),
    }

    full_lengths = np.full(len(test_lengths), test_l2.shape[1], dtype=np.int64)
    zero_tail = _zero_tail(test_l2, test_lengths)
    padded_rows: list[dict[str, Any]] = []
    for tail_name, padded_l2 in (
        ("zero_l2_tail", zero_tail),
        ("native_l2_tail", test_l2),
    ):
        analog_padded = exp726._analog_scores(
            padded_l2, full_lengths, W, scale=1.0
        )
        padded_rows.append(
            {
                "tail": tail_name,
                "beta": None,
                "readout": "analog",
                **_metrics(test_y, analog_padded),
            }
        )
        for beta in (1.0, LIF_BETA):
            sim = exp726._simulate_unipolar(
                padded_l2,
                full_lengths,
                W,
                beta=beta,
                cap=OUTPUT_CAP,
                scale=1.0,
            )
            padded_rows.append(
                {
                    "tail": tail_name,
                    "beta": float(beta),
                    "readout": "spike_count",
                    **_metrics(test_y, sim["counts"]),
                    "mean_abs_final_membrane": float(
                        sim["diagnostics"]["mean_abs_final_membrane"]
                    ),
                    "mean_total_output_spikes_per_sample": float(
                        sim["diagnostics"]["mean_total_output_spikes_per_sample"]
                    ),
                }
            )

    sanity = {
        "analog_charge_prediction_equal": bool(
            np.array_equal(analog.argmax(1), if1["charge_scores"].argmax(1))
        ),
        "analog_charge_max_score_error": float(
            np.max(np.abs(analog - if1["charge_scores"]))
        ),
        "zero_tail_analog_prediction_equal": bool(
            np.array_equal(
                analog.argmax(1),
                exp726._analog_scores(zero_tail, full_lengths, W).argmax(1),
            )
        ),
    }
    if not sanity["analog_charge_prediction_equal"]:
        raise RuntimeError(f"{spec.key}: beta=1 charge sanity failed")
    if not sanity["zero_tail_analog_prediction_equal"]:
        raise RuntimeError(f"{spec.key}: zero-tail analog sanity failed")

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "frozen_l1_l2": True,
            "frozen_source_w": True,
            "output_cap": OUTPUT_CAP,
            "input_gain": 1.0,
            "flush_steps": list(FLUSH_STEPS),
        },
        "references": references,
        "zero_input_flush": flush_rows,
        "infinite_if_flush": infinite_row,
        "padded_tail": padded_rows,
        "sanity": sanity,
    }
    _save_json(out_path, payload)
    return payload


# -----------------------------------------------------------------------------
# Part B: evidence semantics, same frozen L2 and same frozen W, no training.
# -----------------------------------------------------------------------------

def _evidence_diagnostics(
    l2: np.ndarray, lengths: np.ndarray, W: np.ndarray
) -> dict[str, float]:
    evidence, mask = _masked_evidence(l2, lengths, W)
    selected = evidence[mask]
    negative = selected[selected < 0]
    positive = selected[selected > 0]
    return {
        "negative_evidence_fraction": float((selected < 0).mean()),
        "positive_evidence_fraction": float((selected > 0).mean()),
        "mean_negative_evidence": float(negative.mean()) if negative.size else 0.0,
        "mean_positive_evidence": float(positive.mean()) if positive.size else 0.0,
        "mean_abs_evidence": float(np.abs(selected).mean()),
    }


def run_part_b(spec: SourceSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_source(spec)
    out_path = _path(config.results_dir, "part_b_evaluations", spec.key, ".json")
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    cache = _load_cache(spec, config)
    W = cache["source_w"]
    rows: list[dict[str, Any]] = []
    split_scores: dict[str, dict[str, np.ndarray]] = {}

    for split in ("val", "test"):
        l2, y, lengths = cache[split]
        raw = exp726._analog_scores(l2, lengths, W, scale=1.0)
        positive = _positive_only_scores(l2, lengths, W)
        lif = exp726._simulate_unipolar(
            l2, lengths, W, beta=LIF_BETA, cap=OUTPUT_CAP, scale=1.0
        )["counts"]
        split_scores[split] = {
            "raw_signed": raw,
            "positive_only": positive,
            "lif_beta05": lif,
        }
        for condition, scores in split_scores[split].items():
            rows.append(
                {
                    "split": split,
                    "condition": condition,
                    "temperature": None,
                    **_metrics(y, scores),
                }
            )
        for temperature in SOFTMAX_TEMPS:
            scores = _softmax_vote_scores(l2, lengths, W, temperature)
            split_scores[split][f"softmax_vote_{temperature:g}"] = scores
            rows.append(
                {
                    "split": split,
                    "condition": "softmax_vote",
                    "temperature": float(temperature),
                    **_metrics(y, scores),
                }
            )

    val_softmax = [
        row for row in rows
        if row["split"] == "val" and row["condition"] == "softmax_vote"
    ]
    selected_tau_row = sorted(
        val_softmax,
        key=lambda row: (
            -float(row["balanced_accuracy"]),
            abs(math.log2(float(row["temperature"]))),
            float(row["temperature"]),
        ),
    )[0]
    selected_tau = float(selected_tau_row["temperature"])
    test_l2, test_y, test_lengths = cache["test"]
    selected_test = _metrics(
        test_y,
        _softmax_vote_scores(test_l2, test_lengths, W, selected_tau),
    )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "frozen_l1_l2": True,
            "frozen_source_w": True,
            "no_training": True,
            "softmax_temperatures": list(SOFTMAX_TEMPS),
        },
        "rows": rows,
        "validation_selected_temperature": selected_tau,
        "validation_selected_softmax_test_metrics": selected_test,
        "evidence_diagnostics": {},
    }
    for split in ("val", "test"):
        l2, _, lengths = cache[split]
        payload["evidence_diagnostics"][split] = _evidence_diagnostics(
            l2, lengths, W
        )
    _save_json(out_path, payload)
    return payload


# -----------------------------------------------------------------------------
# Shared frozen-head training utilities for Part C and Part D.
# -----------------------------------------------------------------------------

def _cached_loader(
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    l2, y, lengths = split
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(
            torch.from_numpy(l2),
            torch.from_numpy(y),
            torch.from_numpy(lengths),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _cached_loaders(
    cache: dict[str, Any],
    seed: int,
    batch_size: int,
    shuffle_train: bool,
) -> dict[str, DataLoader]:
    return {
        split: _cached_loader(
            cache[split],
            batch_size,
            shuffle_train if split == "train" else False,
            _pair_seed(seed, f"{split}_loader"),
        )
        for split in ("train", "val", "test")
    }


def _valid_mean_torch(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return exp7263._valid_mean(values, lengths)


class LinearEvidenceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_linear = nn.Linear(HIDDEN_WIDTH, N_CLASSES, bias=False)

    def forward(self, l2: torch.Tensor) -> torch.Tensor:
        return self.output_linear(l2)


def _c_loss_scores_from_evidence(
    evidence: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    objective: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if objective == "raw_wcce":
        scores = _valid_mean_torch(evidence, lengths)
        return F.cross_entropy(scores, y), scores

    probs = torch.softmax(evidence, dim=-1)
    vote_scores = _valid_mean_torch(probs, lengths)
    if objective == "softmax_vote_wcce":
        true_prob = vote_scores.gather(1, y[:, None]).squeeze(1).clamp_min(1e-12)
        return (-torch.log(true_prob)).mean(), vote_scores

    if objective == "tsce":
        log_probs = F.log_softmax(evidence, dim=-1)
        true_log = log_probs.gather(
            2, y[:, None, None].expand(-1, evidence.shape[1], 1)
        ).squeeze(2)
        mask = (
            torch.arange(evidence.shape[1], device=evidence.device)[None, :]
            < lengths[:, None]
        )
        per_sample = -(
            true_log * mask.to(true_log.dtype)
        ).sum(dim=1) / lengths.clamp_min(1).to(true_log.dtype)
        return per_sample.mean(), vote_scores

    raise ValueError(objective)


def _evaluate_c_native(
    model: LinearEvidenceHead,
    loader: Iterable,
    objective: str,
    device: torch.device,
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for l2, y, lengths in loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            loss, scores = _c_loss_scores_from_evidence(
                model(l2), y, lengths, objective
            )
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n += len(y)
    metrics = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    return {**metrics, "objective_loss": loss_sum / max(n, 1)}


def _cross_evaluate_w(
    cache: dict[str, Any],
    W: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for split in ("train", "val", "test"):
        l2, y, lengths = cache[split]
        analog = exp726._analog_scores(l2, lengths, W, scale=1.0)
        softmax_vote = _softmax_vote_scores(l2, lengths, W, 1.0)
        lif = exp726._simulate_unipolar(
            l2, lengths, W, beta=LIF_BETA, cap=OUTPUT_CAP, scale=1.0
        )["counts"]
        result[split] = {
            "analog": _metrics(y, analog),
            "softmax_vote": _metrics(y, softmax_vote),
            "lif_beta05": _metrics(y, lif),
        }
    return result


def _checkpoint_improved(
    val_metrics: dict[str, float],
    best_ba: float,
    best_loss: float,
) -> bool:
    return (
        val_metrics["balanced_accuracy"] > best_ba + 1e-12
        or (
            abs(val_metrics["balanced_accuracy"] - best_ba) <= 1e-12
            and val_metrics["objective_loss"] < best_loss
        )
    )


def run_part_c(spec: CSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_c(spec)
    eval_path = _path(config.results_dir, "part_c_evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "part_c_checkpoints", spec.key, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    cache = _load_cache(spec.source, config)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp3.seed_all(_pair_seed(spec.seed, "model_init"))
    model = LinearEvidenceHead().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY
    )
    train_loader = _cached_loaders(
        cache, spec.seed, config.batch_size, True
    )["train"]
    eval_loaders = _cached_loaders(
        cache, spec.seed, config.batch_size, False
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
        n = 0
        for l2, y, lengths in train_loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _c_loss_scores_from_evidence(
                model(l2), y, lengths, spec.objective
            )
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n += len(y)

        train_metrics = _evaluate_c_native(
            model, eval_loaders["train"], spec.objective, device
        )
        val_metrics = _evaluate_c_native(
            model, eval_loaders["val"], spec.objective, device
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )
        if _checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        if (
            epoch >= MIN_EPOCHS
            and best_epoch > 0
            and epoch - best_epoch >= PATIENCE
        ):
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No Part C checkpoint selected for {spec.key}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "model_state_dict": best_state,
            "pairing": "objective excluded from model-init and loader seeds",
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state, strict=True)
    native_metrics = {
        split: _evaluate_c_native(
            model, loader, spec.objective, device
        )
        for split, loader in eval_loaders.items()
    }
    W = (
        model.output_linear.weight.detach().cpu().numpy().astype(np.float64)
    )
    cross = _cross_evaluate_w(cache, W)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "frozen_l1_l2": True,
            "trainable_parameters": HIDDEN_WIDTH * N_CLASSES,
            "bias": False,
            "softmax_vote_temperature": 1.0,
        },
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native_metrics,
        "cross_evaluation": cross,
    }
    _save_json(eval_path, payload)
    hist = pd.DataFrame(history)
    hist_path = _path(config.results_dir, "part_c_histories", spec.key, ".csv")
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(hist_path, index=False)
    return payload


# -----------------------------------------------------------------------------
# Part D: CE-gain vs LIF gradient geometry, only W is trainable.
# -----------------------------------------------------------------------------

def _d_loss_scores(
    model: exp7263.FrozenReadoutHead,
    l2: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    ce_gain: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    trajectory = model.forward_trajectory(l2)
    values = (
        trajectory["evidence"]
        if model.mode == exp7263.HEAD_LINEAR
        else trajectory["spikes"]
    )
    scores = _valid_mean_torch(values, lengths)
    logits = float(ce_gain) * scores
    return F.cross_entropy(logits, y), scores


def _evaluate_d_native(
    model: exp7263.FrozenReadoutHead,
    loader: Iterable,
    ce_gain: float,
    device: torch.device,
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for l2, y, lengths in loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            loss, scores = _d_loss_scores(
                model, l2, y, lengths, ce_gain
            )
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n += len(y)
    metrics = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    return {**metrics, "objective_loss": loss_sum / max(n, 1)}


def run_part_d(spec: DSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_d(spec)
    eval_path = _path(config.results_dir, "part_d_evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "part_d_checkpoints", spec.key, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    cache = _load_cache(spec.source, config)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp3.seed_all(_pair_seed(spec.seed, "model_init"))
    model = exp7263.FrozenReadoutHead(spec.head_type).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp72.LR * spec.lr_scale,
        weight_decay=exp72.WEIGHT_DECAY,
    )
    train_loader = _cached_loaders(
        cache, spec.seed, config.batch_size, True
    )["train"]
    eval_loaders = _cached_loaders(
        cache, spec.seed, config.batch_size, False
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
        n = 0
        for l2, y, lengths in train_loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _d_loss_scores(
                model, l2, y, lengths, spec.ce_gain
            )
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n += len(y)

        train_metrics = _evaluate_d_native(
            model, eval_loaders["train"], spec.ce_gain, device
        )
        val_metrics = _evaluate_d_native(
            model, eval_loaders["val"], spec.ce_gain, device
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )
        if _checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        if (
            epoch >= MIN_EPOCHS
            and best_epoch > 0
            and epoch - best_epoch >= PATIENCE
        ):
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No Part D checkpoint selected for {spec.key}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "model_state_dict": best_state,
            "pairing": "head/gain excluded from model-init and loader seeds",
            "base_lr": exp72.LR,
            "effective_lr": exp72.LR * spec.lr_scale,
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state, strict=True)
    native_metrics = {
        split: _evaluate_d_native(
            model, loader, spec.ce_gain, device
        )
        for split, loader in eval_loaders.items()
    }
    W = (
        model.output_linear.weight.detach().cpu().numpy().astype(np.float64)
    )
    cross = _cross_evaluate_w(cache, W)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "frozen_l1_l2": True,
            "trainable_parameters": HIDDEN_WIDTH * N_CLASSES,
            "bias": False,
            "input_gain": 1.0,
            "ce_gain_only_before_ce": True,
        },
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native_metrics,
        "cross_evaluation": cross,
    }
    _save_json(eval_path, payload)
    hist = pd.DataFrame(history)
    hist_path = _path(config.results_dir, "part_d_histories", spec.key, ".csv")
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(hist_path, index=False)
    return payload


def _gradient_vector(
    mode: str,
    W0: torch.Tensor,
    l2: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    gain: float,
) -> tuple[np.ndarray, float]:
    model = exp7263.FrozenReadoutHead(mode).to(l2.device)
    with torch.no_grad():
        model.output_linear.weight.copy_(W0.to(l2.device))
    model.zero_grad(set_to_none=True)
    loss, _ = _d_loss_scores(model, l2, y, lengths, gain)
    loss.backward()
    grad = model.output_linear.weight.grad.detach().cpu().numpy().astype(np.float64)
    return grad, float(loss.detach().cpu())


def run_gradient_diagnostics(
    spec: SourceSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_source(spec)
    out_path = _path(config.results_dir, "gradient_evaluations", spec.key, ".json")
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    cache = _load_cache(spec, config)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp3.seed_all(_pair_seed(spec.seed, "model_init"))
    template = exp7263.FrozenReadoutHead("linear")
    W0 = template.output_linear.weight.detach().cpu().clone()

    loader = _cached_loaders(
        cache, spec.seed, config.batch_size, True
    )["train"]
    l2, y, lengths = next(iter(loader))
    l2 = l2.to(device=device, dtype=torch.float32)
    y = y.to(device)
    lengths = lengths.to(device)

    rows: list[dict[str, Any]] = []
    for gain in CE_GAINS:
        g_lin, loss_lin = _gradient_vector(
            "linear", W0, l2, y, lengths, gain
        )
        g_lif, loss_lif = _gradient_vector(
            "lif", W0, l2, y, lengths, gain
        )
        lin_norm = float(np.linalg.norm(g_lin))
        lif_norm = float(np.linalg.norm(g_lif))
        denom = lin_norm * lif_norm
        global_cos = (
            float(np.dot(g_lin.ravel(), g_lif.ravel()) / denom)
            if denom > 0
            else 0.0
        )
        row: dict[str, Any] = {
            "ce_gain": float(gain),
            "linear_loss": loss_lin,
            "lif_loss": loss_lif,
            "gradient_norm_linear": lin_norm,
            "gradient_norm_lif": lif_norm,
            "norm_ratio_lif_over_linear": (
                lif_norm / lin_norm if lin_norm > 0 else 0.0
            ),
            "global_cosine": global_cos,
        }
        for class_idx in range(N_CLASSES):
            a = g_lin[class_idx]
            b = g_lif[class_idx]
            d = float(np.linalg.norm(a) * np.linalg.norm(b))
            row[f"class{class_idx}_cosine"] = (
                float(np.dot(a, b) / d) if d > 0 else 0.0
            )
        rows.append(row)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "same_initial_w": True,
            "same_diagnostic_batch": True,
            "frozen_l1_l2": True,
            "gains": list(CE_GAINS),
        },
        "rows": rows,
    }
    _save_json(out_path, payload)
    return payload


# -----------------------------------------------------------------------------
# Aggregation-only finalizer.
# -----------------------------------------------------------------------------

def _aggregate(
    frame: pd.DataFrame,
    groups: list[str],
    values: list[str],
) -> pd.DataFrame:
    grouped = frame.groupby(groups, dropna=False)[values]
    return pd.concat(
        [
            grouped.size().rename("n"),
            grouped.mean().add_suffix("_mean"),
            grouped.std(ddof=1).fillna(0).add_suffix("_std"),
        ],
        axis=1,
    ).reset_index()


def finalize(config: Config) -> dict[str, Any]:
    missing: list[str] = []
    for spec in source_specs():
        for kind in (
            "part_a_evaluations",
            "part_b_evaluations",
            "gradient_evaluations",
        ):
            p = _path(config.results_dir, kind, spec.key, ".json")
            if not p.exists():
                missing.append(str(p))
    for spec in c_specs():
        p = _path(config.results_dir, "part_c_evaluations", spec.key, ".json")
        if not p.exists():
            missing.append(str(p))
    for spec in d_specs():
        p = _path(config.results_dir, "part_d_evaluations", spec.key, ".json")
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Exp7.2.6.4 incomplete; missing:\n" + "\n".join(missing)
        )

    root = config.results_dir

    a_ref_rows: list[dict[str, Any]] = []
    a_flush_rows: list[dict[str, Any]] = []
    a_inf_rows: list[dict[str, Any]] = []
    a_padded_rows: list[dict[str, Any]] = []
    for spec in source_specs():
        payload = json.loads(
            _path(
                root, "part_a_evaluations", spec.key, ".json"
            ).read_text(encoding="utf-8")
        )
        for condition, metrics in payload["references"].items():
            a_ref_rows.append(
                {"seed": spec.seed, "condition": condition, **metrics}
            )
        for row in payload["zero_input_flush"]:
            a_flush_rows.append({"seed": spec.seed, **row})
        a_inf_rows.append(
            {"seed": spec.seed, **payload["infinite_if_flush"]}
        )
        for row in payload["padded_tail"]:
            a_padded_rows.append({"seed": spec.seed, **row})

    b_rows: list[dict[str, Any]] = []
    b_selected_rows: list[dict[str, Any]] = []
    b_diag_rows: list[dict[str, Any]] = []
    for spec in source_specs():
        payload = json.loads(
            _path(
                root, "part_b_evaluations", spec.key, ".json"
            ).read_text(encoding="utf-8")
        )
        for row in payload["rows"]:
            b_rows.append({"seed": spec.seed, **row})
        b_selected_rows.append(
            {
                "seed": spec.seed,
                "selected_temperature": payload[
                    "validation_selected_temperature"
                ],
                **payload["validation_selected_softmax_test_metrics"],
            }
        )
        for split, diag in payload["evidence_diagnostics"].items():
            b_diag_rows.append(
                {"seed": spec.seed, "split": split, **diag}
            )

    c_rows: list[dict[str, Any]] = []
    for spec in c_specs():
        payload = json.loads(
            _path(
                root, "part_c_evaluations", spec.key, ".json"
            ).read_text(encoding="utf-8")
        )
        for split, readouts in payload["cross_evaluation"].items():
            for readout, metrics in readouts.items():
                c_rows.append(
                    {
                        "seed": spec.seed,
                        "training_objective": spec.objective,
                        "evaluation_readout": readout,
                        "split": split,
                        "best_epoch": payload["best_epoch"],
                        **metrics,
                    }
                )

    d_rows: list[dict[str, Any]] = []
    for spec in d_specs():
        payload = json.loads(
            _path(
                root, "part_d_evaluations", spec.key, ".json"
            ).read_text(encoding="utf-8")
        )
        for split, readouts in payload["cross_evaluation"].items():
            for readout, metrics in readouts.items():
                d_rows.append(
                    {
                        "seed": spec.seed,
                        "head_type": spec.head_type,
                        "ce_gain": spec.ce_gain,
                        "lr_scale": spec.lr_scale,
                        "evaluation_readout": readout,
                        "split": split,
                        "best_epoch": payload["best_epoch"],
                        **metrics,
                    }
                )

    grad_rows: list[dict[str, Any]] = []
    for spec in source_specs():
        payload = json.loads(
            _path(
                root, "gradient_evaluations", spec.key, ".json"
            ).read_text(encoding="utf-8")
        )
        for row in payload["rows"]:
            grad_rows.append({"seed": spec.seed, **row})

    frames = {
        "part_a_reference_runs.csv": pd.DataFrame(a_ref_rows),
        "part_a_flush_runs.csv": pd.DataFrame(a_flush_rows),
        "part_a_infinite_flush_runs.csv": pd.DataFrame(a_inf_rows),
        "part_a_padded_tail_runs.csv": pd.DataFrame(a_padded_rows),
        "part_b_evidence_semantics_runs.csv": pd.DataFrame(b_rows),
        "part_b_selected_softmax_runs.csv": pd.DataFrame(b_selected_rows),
        "part_b_evidence_diagnostics_runs.csv": pd.DataFrame(b_diag_rows),
        "part_c_objective_training_runs.csv": pd.DataFrame(c_rows),
        "part_d_gain_sweep_runs.csv": pd.DataFrame(d_rows),
        "part_d_gradient_geometry_runs.csv": pd.DataFrame(grad_rows),
    }
    for name, frame in frames.items():
        frame.to_csv(root / name, index=False)

    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1"]
    _aggregate(
        frames["part_a_reference_runs.csv"],
        ["condition"],
        metric_cols,
    ).to_csv(root / "part_a_reference_summary.csv", index=False)
    _aggregate(
        frames["part_a_flush_runs.csv"],
        ["beta", "flush_steps"],
        metric_cols
        + [
            "mean_abs_U_T",
            "mean_abs_U_TK",
            "positive_residual_sum",
            "negative_residual_sum",
            "mean_extra_spikes_per_sample",
            "discharge_ratio_positive",
            "fireable_discharge_ratio",
        ],
    ).to_csv(root / "part_a_flush_summary.csv", index=False)
    _aggregate(
        frames["part_a_infinite_flush_runs.csv"],
        ["beta"],
        metric_cols + ["mean_extra_spikes_per_sample"],
    ).to_csv(root / "part_a_infinite_flush_summary.csv", index=False)
    _aggregate(
        frames["part_a_padded_tail_runs.csv"],
        ["tail", "beta", "readout"],
        metric_cols,
    ).to_csv(root / "part_a_padded_tail_summary.csv", index=False)
    _aggregate(
        frames["part_b_evidence_semantics_runs.csv"],
        ["split", "condition", "temperature"],
        metric_cols,
    ).to_csv(root / "part_b_evidence_semantics_summary.csv", index=False)
    _aggregate(
        frames["part_b_selected_softmax_runs.csv"],
        ["selected_temperature"],
        metric_cols,
    ).to_csv(root / "part_b_selected_softmax_summary.csv", index=False)
    _aggregate(
        frames["part_c_objective_training_runs.csv"],
        ["training_objective", "evaluation_readout", "split"],
        metric_cols,
    ).to_csv(root / "part_c_objective_training_summary.csv", index=False)
    _aggregate(
        frames["part_d_gain_sweep_runs.csv"],
        ["head_type", "ce_gain", "lr_scale", "evaluation_readout", "split"],
        metric_cols,
    ).to_csv(root / "part_d_gain_sweep_summary.csv", index=False)

    grad_value_cols = [
        "gradient_norm_linear",
        "gradient_norm_lif",
        "norm_ratio_lif_over_linear",
        "global_cosine",
    ] + [f"class{i}_cosine" for i in range(N_CLASSES)]
    _aggregate(
        frames["part_d_gradient_geometry_runs.csv"],
        ["ce_gain"],
        grad_value_cols,
    ).to_csv(root / "part_d_gradient_geometry_summary.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "regularization": REGULARIZATION,
        "seeds": list(SEEDS),
        "source": "Exp7.2.6.1 Mean-CE bias=False task_only",
        "frozen_l1_l2_throughout": True,
        "no_e2e_retraining": True,
        "part_a_question": "Is beta=1 count failure a finite output-bandwidth/residual-discharge problem?",
        "part_b_question": "How much do signed negative evidence and evidence normalization matter for a fixed good W?",
        "part_c_question": "How do Raw-WCCE, softmax-vote WCCE, and TSCE shape W on the same frozen L2?",
        "part_d_question": "Is W_lif degradation CE-scale conditioning or LIF/surrogate-gradient geometry?",
        "cache_tasks": len(source_specs()),
        "part_a_tasks": len(source_specs()),
        "part_b_tasks": len(source_specs()),
        "part_c_training_tasks": len(c_specs()),
        "part_d_training_tasks": len(d_specs()),
        "gradient_tasks": len(source_specs()),
        "flush_steps": list(FLUSH_STEPS),
        "softmax_temperatures": list(SOFTMAX_TEMPS),
        "ce_gains": list(CE_GAINS),
        "notebook_contract": "analysis-only; reads finalized aggregate CSV/JSON artifacts only",
    }
    _save_json(root / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exp7.2.6.4 frozen-L2 output-interface decomposition"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, n in (
        ("prepare-cache", len(source_specs())),
        ("part-a", len(source_specs())),
        ("part-b", len(source_specs())),
        ("part-c", len(c_specs())),
        ("part-d", len(d_specs())),
        ("gradient", len(source_specs())),
    ):
        p = sub.add_parser(name)
        p.add_argument(
            "--array-task-id",
            type=int,
            required=True,
            choices=range(n),
        )
        p.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = find_repo_root(Path.cwd())
    config = Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        max_epochs=args.max_epochs,
    )
    config.results_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare-cache":
        print(
            prepare_cache(
                source_specs()[args.array_task_id],
                config,
                args.force,
            )
        )
    elif args.command == "part-a":
        payload = run_part_a(
            source_specs()[args.array_task_id], config, args.force
        )
        print(json.dumps(payload["references"], indent=2, sort_keys=True))
    elif args.command == "part-b":
        payload = run_part_b(
            source_specs()[args.array_task_id], config, args.force
        )
        print(
            json.dumps(
                {
                    "selected_tau": payload[
                        "validation_selected_temperature"
                    ],
                    "selected_test": payload[
                        "validation_selected_softmax_test_metrics"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "part-c":
        payload = run_part_c(
            c_specs()[args.array_task_id], config, args.force
        )
        print(
            json.dumps(
                payload["cross_evaluation"]["test"],
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "part-d":
        payload = run_part_d(
            d_specs()[args.array_task_id], config, args.force
        )
        print(
            json.dumps(
                payload["cross_evaluation"]["test"],
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "gradient":
        payload = run_gradient_diagnostics(
            source_specs()[args.array_task_id], config, args.force
        )
        print(json.dumps(payload["rows"], indent=2, sort_keys=True))
    elif args.command == "finalize":
        print(json.dumps(finalize(config), indent=2, sort_keys=True))
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
