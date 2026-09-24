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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from snn.accel_reconstruction_eval.datasets import load_acceleration_data
from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_3_0_2_hidden_multitau_architectures as exp302
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_7_3_9_pretrained_a2_depth_extension as exp739
from scripts import experiment_12_1_primitive_capacity_fragmentation as exp121


EXPERIMENT_ID = "experiment_13_hierarchical_temporal_representation"
PROTOCOL_VERSION = "hierarchical_temporal_representation_v2"

SOURCE_METHOD = "A2_e2e_linear_wcce"
SOURCE_OBJECTIVE = "wcce"
SEEDS = exp73.SEEDS
HIDDEN_WIDTH = exp73.HIDDEN_WIDTH
BATCH_SIZE = exp72.BATCH_SIZE

SUBEXPERIMENTS = (
    "A1_recurrence",
    "A2_history_truncation",
    "A3_layer_reset",
    "B_context_expression",
    "C_stroke_organization",
    "D_stroke_scrambling",
    "E_cross_tau_decoding",
)

A1_MODEL_KINDS = ("a2", "random_a2", "c2")
TRUNC_MODEL_KINDS = ("a2", "c2")
HISTORY_MS = (50, 100, 250, 500, 750, 1000)
PHASES = (0.25, 0.50, 0.75, 1.00)
RESET_PHASES = (0.25, 0.50, 0.75)
RESET_CASES = ("intact", "reset_l1", "reset_l2", "reset_both")
RESET_DELAYS_STEPS = (0, 1, 2, 4, 8, 16, 32)
RECURRENCE_LAGS_STEPS = (1, 2, 4, 8, 16, 32, 48, 64)
SPIKE_SMOOTH_STEPS = 4

ANALYSIS_STATES = ("spike50", "syn_current", "pre_reset")
MECHANISTIC_STATES = ("syn_current", "pre_reset", "spike")
PROBE_BIAS_MODES = ("no_bias", "affine")
PROBE_C_GRID = tuple(float(v) for v in exp302.PROBE_C_GRID)
PROBE_MAX_ITER = 5000

BOUNDARY_OFFSETS_STEPS = tuple(range(-16, 17))
WITHIN_ACROSS_LAGS_STEPS = (1, 2, 4, 8, 16)

SCRAMBLE_GAP_MS = (0, 50, 100, 250)
SCRAMBLE_SCALES = ("stroke", "multi_stroke")
SCRAMBLE_ONSET_EXCLUDE_STEPS = 2

IMPULSE_TAIL_STEPS = 64
REAL_MOTIF_STEPS = 6

EXPECTED_A1_TASKS = len(SEEDS) * len(A1_MODEL_KINDS)
EXPECTED_A2_TASKS = len(SEEDS) * len(TRUNC_MODEL_KINDS) * len(HISTORY_MS)
EXPECTED_A3_TASKS = len(SEEDS) * len(RESET_CASES)
EXPECTED_B_TASKS = EXPECTED_A2_TASKS
EXPECTED_C_TASKS = len(SEEDS)
EXPECTED_D_TASKS = len(SEEDS) * len(SCRAMBLE_GAP_MS) * len(SCRAMBLE_SCALES)
EXPECTED_E_TASKS = EXPECTED_A1_TASKS


@dataclass(frozen=True)
class A1Spec:
    seed: int
    model_kind: str

    @property
    def key(self) -> str:
        return f"{self.model_kind}__seed{self.seed}"


@dataclass(frozen=True)
class A2Spec:
    seed: int
    model_kind: str
    history_ms: int

    @property
    def key(self) -> str:
        return f"{self.model_kind}__h{self.history_ms}ms__seed{self.seed}"


@dataclass(frozen=True)
class A3Spec:
    seed: int
    reset_case: str

    @property
    def key(self) -> str:
        return f"{self.reset_case}__seed{self.seed}"


@dataclass(frozen=True)
class BSpec:
    seed: int
    model_kind: str
    history_ms: int

    @property
    def key(self) -> str:
        return f"{self.model_kind}__h{self.history_ms}ms__seed{self.seed}"


@dataclass(frozen=True)
class CSpec:
    seed: int

    @property
    def key(self) -> str:
        return f"seed{self.seed}"


@dataclass(frozen=True)
class DSpec:
    seed: int
    gap_ms: int
    scale: str

    @property
    def key(self) -> str:
        return f"{self.scale}__gap{self.gap_ms}ms__seed{self.seed}"


@dataclass(frozen=True)
class ESpec:
    seed: int
    model_kind: str

    @property
    def key(self) -> str:
        return f"{self.model_kind}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / EXPERIMENT_ID
        / PROTOCOL_VERSION
    )


def _subdir(config: Config, name: str) -> Path:
    if name not in SUBEXPERIMENTS:
        raise ValueError(name)
    return config.results_dir / name


def a1_specs() -> list[A1Spec]:
    return [
        A1Spec(seed, model_kind)
        for seed in SEEDS
        for model_kind in A1_MODEL_KINDS
    ]


def a2_specs() -> list[A2Spec]:
    return [
        A2Spec(seed, model_kind, history_ms)
        for seed in SEEDS
        for model_kind in TRUNC_MODEL_KINDS
        for history_ms in HISTORY_MS
    ]


def a3_specs() -> list[A3Spec]:
    return [
        A3Spec(seed, reset_case)
        for seed in SEEDS
        for reset_case in RESET_CASES
    ]


def b_specs() -> list[BSpec]:
    return [
        BSpec(seed, model_kind, history_ms)
        for seed in SEEDS
        for model_kind in TRUNC_MODEL_KINDS
        for history_ms in HISTORY_MS
    ]


def c_specs() -> list[CSpec]:
    return [CSpec(seed) for seed in SEEDS]


def d_specs() -> list[DSpec]:
    return [
        DSpec(seed, gap_ms, scale)
        for seed in SEEDS
        for gap_ms in SCRAMBLE_GAP_MS
        for scale in SCRAMBLE_SCALES
    ]


def e_specs() -> list[ESpec]:
    return [
        ESpec(seed, model_kind)
        for seed in SEEDS
        for model_kind in A1_MODEL_KINDS
    ]


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _source_spec(seed: int) -> exp73.E2ESpec:
    return exp73.E2ESpec(seed, "linear", SOURCE_OBJECTIVE)


def _a2_checkpoint_path(repo_root: Path, seed: int) -> Path:
    return (
        exp73.results_dir(repo_root)
        / "e2e_checkpoints"
        / f"{_source_spec(seed).key}.pt"
    )


def _c2_checkpoint_path(repo_root: Path, seed: int) -> Path:
    return (
        exp739.results_dir(repo_root)
        / exp739.C2_CASE
        / "checkpoints"
        / f"{exp739.C2_CASE}__seed{seed}.pt"
    )


def _split_arrays(
    data: exp3.Data,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }


def _load_model(
    config: Config,
    data: exp3.Data,
    seed: int,
    model_kind: str,
) -> torch.nn.Module:
    device = torch.device(config.device)
    if model_kind == "a2":
        path = _a2_checkpoint_path(config.repo_root, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("experiment_id") != exp73.EXPERIMENT_ID:
            raise ValueError(f"Unexpected A2 experiment id in {path}")
        if payload.get("protocol_version") != exp73.PROTOCOL_VERSION:
            raise ValueError(f"Unexpected A2 protocol in {path}")
        if payload.get("spec") != asdict(_source_spec(seed)):
            raise ValueError(f"A2 checkpoint identity mismatch: {path}")
        model = exp73.Exp73Net("linear", len(data.labels), data.fs).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
    elif model_kind == "c2":
        path = _c2_checkpoint_path(config.repo_root, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("experiment_id") != exp739.EXPERIMENT_ID:
            raise ValueError(f"Unexpected C2 experiment id in {path}")
        if payload.get("protocol_version") != exp739.PROTOCOL_VERSION:
            raise ValueError(f"Unexpected C2 protocol in {path}")
        expected = asdict(exp739.TrainSpec(exp739.C2_CASE, seed))
        if payload.get("spec") != expected:
            raise ValueError(f"C2 checkpoint identity mismatch: {path}")
        model = exp739.DepthExtensionNet(len(data.labels), data.fs).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
    elif model_kind == "random_a2":
        exp3.seed_all(exp3.dseed(seed, EXPERIMENT_ID, "random_a2"))
        model = exp73.Exp73Net("linear", len(data.labels), data.fs).to(device)
    else:
        raise ValueError(model_kind)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _layer_names(model: torch.nn.Module) -> tuple[str, ...]:
    return tuple(f"L{index + 1}" for index in range(len(model.hidden_linears)))


def _zero_states(
    model: torch.nn.Module,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    width = int(model.hidden_linears[0].out_features)
    syn = [
        torch.zeros(batch, width, dtype=dtype, device=device)
        for _ in model.hidden_linears
    ]
    mem = [torch.zeros_like(value) for value in syn]
    return syn, mem


def _replay(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    reset_steps: torch.Tensor | None = None,
    reset_layers: tuple[int, ...] = (),
) -> dict[str, dict[str, torch.Tensor]]:
    batch, steps, _ = x.shape
    layers = _layer_names(model)
    syn, mem = _zero_states(model, batch, x.dtype, x.device)
    collected: dict[str, dict[str, list[torch.Tensor]]] = {
        layer: {
            "syn_current": [],
            "pre_reset": [],
            "post_reset": [],
            "spike": [],
        }
        for layer in layers
    }

    if reset_steps is not None:
        reset_steps = reset_steps.to(device=x.device, dtype=torch.long)
        if reset_steps.shape != (batch,):
            raise ValueError("reset_steps must have shape [batch]")

    for timestep in range(steps):
        if reset_steps is not None and reset_layers:
            mask = reset_steps == timestep
            if bool(mask.any()):
                for layer_index in reset_layers:
                    syn[layer_index] = syn[layer_index].clone()
                    mem[layer_index] = mem[layer_index].clone()
                    syn[layer_index][mask] = 0
                    mem[layer_index][mask] = 0

        current = x[:, timestep]
        for layer_index, layer in enumerate(layers):
            alpha = getattr(model, f"alpha_{layer_index}")
            syn[layer_index] = (
                alpha * syn[layer_index]
                + model.hidden_linears[layer_index](current)
            )
            spike, post_reset, pre_reset = model.hidden_lifs[layer_index](
                syn[layer_index], mem[layer_index]
            )
            mem[layer_index] = post_reset
            collected[layer]["syn_current"].append(syn[layer_index])
            collected[layer]["pre_reset"].append(pre_reset)
            collected[layer]["post_reset"].append(post_reset)
            collected[layer]["spike"].append(spike)
            current = spike

    return {
        layer: {
            state: torch.stack(values, dim=1)
            for state, values in state_map.items()
        }
        for layer, state_map in collected.items()
    }


def _rolling_sum(values: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return values
    cumsum = torch.cumsum(values, dim=1)
    prefix = torch.zeros_like(cumsum)
    prefix[:, window:] = cumsum[:, :-window]
    return cumsum - prefix


def _state_views(
    replay: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer, states in replay.items():
        out[layer] = {
            "syn_current": states["syn_current"],
            "pre_reset": states["pre_reset"],
            "post_reset": states["post_reset"],
            "spike": states["spike"],
            "spike50": _rolling_sum(states["spike"], SPIKE_SMOOTH_STEPS),
        }
    return out


def _phase_steps(lengths: np.ndarray, phase: float) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64)
    if not (0.0 < phase <= 1.0):
        raise ValueError(phase)
    return np.floor((lengths - 1).clip(min=0) * phase).astype(np.int64)


def _gather_steps(values: torch.Tensor, steps: np.ndarray) -> np.ndarray:
    indices = torch.as_tensor(steps, dtype=torch.long, device=values.device)
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, indices].detach().cpu().numpy().astype(np.float32)


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    numerator = np.sum(a * b, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 1e-12,
    )


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = b - b.mean(axis=-1, keepdims=True)
    return _row_cosine(ac, bc)


def _row_normalized_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a, axis=-1) + 1e-12
    return np.linalg.norm(a - b, axis=-1) / denominator


def _similarity_summary(
    a: np.ndarray,
    b: np.ndarray,
) -> dict[str, float]:
    corr = _row_corr(a, b)
    cosine = _row_cosine(a, b)
    nl2 = _row_normalized_l2(a, b)
    return {
        "corr_mean": float(np.nanmean(corr)),
        "corr_std": float(np.nanstd(corr)),
        "cosine_mean": float(np.nanmean(cosine)),
        "cosine_std": float(np.nanstd(cosine)),
        "normalized_l2_mean": float(np.nanmean(nl2)),
        "normalized_l2_std": float(np.nanstd(nl2)),
        "count": int(np.sum(np.isfinite(corr))),
    }


def _iter_batches(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
) -> Iterable[tuple[int, int, torch.Tensor, np.ndarray, np.ndarray]]:
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        yield (
            start,
            end,
            torch.tensor(X[start:end], dtype=torch.float32),
            np.asarray(y[start:end], dtype=np.int64),
            np.asarray(lengths[start:end], dtype=np.int64),
        )


def prepare_source(config: Config) -> dict[str, Any]:
    data, split_frames = exp121.prepare_data_with_frames(config.repo_root)
    stroke_index = exp121._load_stroke_index(config.repo_root, split_frames)
    sources: list[dict[str, Any]] = []
    for seed in SEEDS:
        for model_kind in ("a2", "c2"):
            model = _load_model(config, data, seed, model_kind)
            del model
            path = (
                _a2_checkpoint_path(config.repo_root, seed)
                if model_kind == "a2"
                else _c2_checkpoint_path(config.repo_root, seed)
            )
            sources.append(
                {
                    "seed": seed,
                    "model_kind": model_kind,
                    "checkpoint": str(path.relative_to(config.repo_root)),
                }
            )
    stroke_counts = {
        split: int(
            sum(
                len(
                    stroke_index.get(
                        (
                            str(row.user),
                            str(row.action),
                            int(row.source_segment_index),
                        ),
                        (),
                    )
                )
                for row in frame.itertuples(index=False)
            )
        )
        for split, frame in split_frames.items()
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_method": SOURCE_METHOD,
        "source_objective": SOURCE_OBJECTIVE,
        "seeds": list(SEEDS),
        "subexperiments": list(SUBEXPERIMENTS),
        "sources": sources,
        "stroke_counts": stroke_counts,
        "task_counts": {
            "A1": EXPECTED_A1_TASKS,
            "A2": EXPECTED_A2_TASKS,
            "A3": EXPECTED_A3_TASKS,
            "B": EXPECTED_B_TASKS,
            "C": EXPECTED_C_TASKS,
            "D": EXPECTED_D_TASKS,
            "E": EXPECTED_E_TASKS,
        },
    }
    _save_json(config.results_dir / "source_manifest.json", payload)
    return payload


def _a1_paths(config: Config, spec: A1Spec) -> tuple[Path, Path, Path]:
    root = _subdir(config, "A1_recurrence")
    return (
        root / "recurrence_runs" / f"{spec.key}.csv",
        root / "activity_runs" / f"{spec.key}.csv",
        root / "anchors" / f"{spec.key}.npz",
    )


def _activity_rows(
    views: dict[str, dict[str, torch.Tensor]],
    lengths: np.ndarray,
    split: str,
    spec: A1Spec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, states in views.items():
        for state in ANALYSIS_STATES:
            values = states[state].detach().cpu().numpy()
            valid_mask = (
                np.arange(values.shape[1])[None, :]
                < lengths[:, None]
            )
            valid = values[valid_mask]
            if len(valid) == 0:
                continue
            row = {
                "seed": spec.seed,
                "model_kind": spec.model_kind,
                "split": split,
                "layer": layer,
                "state": state,
                "mean_state_norm": float(np.mean(np.linalg.norm(valid, axis=-1))),
                "mean_per_neuron_variance": float(np.mean(np.var(valid, axis=0))),
                "nonzero_fraction": float(np.mean(valid != 0)),
            }
            if state == "spike50":
                raw = states["spike"].detach().cpu().numpy()[valid_mask]
                row["firing_rate_per_step"] = float(np.mean(raw))
            else:
                row["firing_rate_per_step"] = float("nan")
            rows.append(row)
    return rows


def _recurrence_rows(
    views: dict[str, dict[str, torch.Tensor]],
    lengths: np.ndarray,
    split: str,
    spec: A1Spec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, states in views.items():
        for state in ANALYSIS_STATES:
            values = states[state].detach().cpu().numpy()
            for lag in RECURRENCE_LAGS_STEPS:
                left: list[np.ndarray] = []
                right: list[np.ndarray] = []
                for sample_index, length in enumerate(lengths):
                    n = int(length)
                    if n <= lag:
                        continue
                    left.append(values[sample_index, : n - lag])
                    right.append(values[sample_index, lag:n])
                if not left:
                    continue
                a = np.concatenate(left, axis=0)
                b = np.concatenate(right, axis=0)
                summary = _similarity_summary(a, b)
                rows.append(
                    {
                        "seed": spec.seed,
                        "model_kind": spec.model_kind,
                        "split": split,
                        "layer": layer,
                        "state": state,
                        "lag_steps": lag,
                        **summary,
                    }
                )
    return rows


def run_a1(
    spec: A1Spec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    if spec.seed not in SEEDS or spec.model_kind not in A1_MODEL_KINDS:
        raise ValueError(spec)
    recurrence_path, activity_path, anchor_path = _a1_paths(config, spec)
    if (
        recurrence_path.exists()
        and activity_path.exists()
        and (anchor_path.exists() or spec.model_kind == "random_a2")
        and not force
    ):
        return {"status": "exists", "key": spec.key}

    data = exp3.prepare_data(config.repo_root)
    model = _load_model(config, data, spec.seed, spec.model_kind)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)

    recurrence_rows: list[dict[str, Any]] = []
    activity_rows: list[dict[str, Any]] = []
    anchor_parts: dict[str, list[np.ndarray]] = {}
    anchor_meta: dict[str, np.ndarray] = {}

    for split, (X, y, lengths) in _split_arrays(data).items():
        for start, end, Xb, yb, lb in _iter_batches(
            X, y, lengths, config.batch_size
        ):
            replay = _replay(model, Xb.to(device))
            views = _state_views(replay)
            if split == "test":
                recurrence_rows.extend(_recurrence_rows(views, lb, split, spec))
                activity_rows.extend(_activity_rows(views, lb, split, spec))
            if spec.model_kind != "random_a2":
                for phase in PHASES:
                    steps = _phase_steps(lb, phase)
                    phase_key = f"p{int(round(phase * 100)):03d}"
                    for layer, states in views.items():
                        for state in ANALYSIS_STATES:
                            key = f"{split}__{layer}__{state}__{phase_key}"
                            anchor_parts.setdefault(key, []).append(
                                _gather_steps(states[state], steps)
                            )
                anchor_meta.setdefault(f"{split}__y", []).append(yb)
                anchor_meta.setdefault(f"{split}__lengths", []).append(lb)

    if spec.model_kind != "random_a2":
        arrays: dict[str, np.ndarray] = {
            key: np.concatenate(parts, axis=0)
            for key, parts in anchor_parts.items()
        }
        for key, parts in anchor_meta.items():
            arrays[key] = np.concatenate(parts, axis=0)
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(anchor_path, **arrays)

    _save_csv(recurrence_path, recurrence_rows)
    _save_csv(activity_path, activity_rows)
    return {
        "status": "completed",
        "key": spec.key,
        "recurrence_rows": len(recurrence_rows),
        "activity_rows": len(activity_rows),
    }


def _suffix_batch_for_steps(
    X: np.ndarray,
    sample_indices: np.ndarray,
    anchor_steps: np.ndarray,
    history_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    anchor_steps = np.asarray(anchor_steps, dtype=np.int64)
    if len(sample_indices) != len(anchor_steps):
        raise ValueError("sample_indices and anchor_steps length mismatch")
    out = np.zeros(
        (len(sample_indices), history_steps, X.shape[2]),
        dtype=np.float32,
    )
    lengths = np.zeros(len(sample_indices), dtype=np.int64)
    for row, (sample_index, anchor) in enumerate(
        zip(sample_indices, anchor_steps, strict=True)
    ):
        end = int(anchor) + 1
        start = max(0, end - history_steps)
        segment = X[int(sample_index), start:end]
        n = len(segment)
        out[row, :n] = segment
        lengths[row] = n
    return out, lengths


def _suffix_features(
    model: torch.nn.Module,
    X: np.ndarray,
    sample_indices: np.ndarray,
    anchor_steps: np.ndarray,
    history_steps: int,
    config: Config,
) -> dict[str, dict[str, np.ndarray]]:
    device = torch.device(config.device)
    output: dict[str, dict[str, list[np.ndarray]]] = {}
    for start in range(0, len(sample_indices), config.batch_size):
        end = min(start + config.batch_size, len(sample_indices))
        suffix, suffix_lengths = _suffix_batch_for_steps(
            X,
            sample_indices[start:end],
            anchor_steps[start:end],
            history_steps,
        )
        replay = _replay(model, torch.tensor(suffix, dtype=torch.float32, device=device))
        views = _state_views(replay)
        gather = suffix_lengths - 1
        for layer, states in views.items():
            output.setdefault(layer, {})
            for state in ANALYSIS_STATES:
                output[layer].setdefault(state, []).append(
                    _gather_steps(states[state], gather)
                )
    return {
        layer: {
            state: np.concatenate(parts, axis=0)
            for state, parts in state_map.items()
        }
        for layer, state_map in output.items()
    }


def _stroke_anchor_records(
    frame: pd.DataFrame,
    stroke_index: dict[tuple[str, str, int], tuple[tuple[int, int, int], ...]],
    lengths: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index, row in enumerate(frame.itertuples(index=False)):
        strokes = stroke_index.get(
            (
                str(row.user),
                str(row.action),
                int(row.source_segment_index),
            ),
            (),
        )
        valid = int(lengths[sample_index])
        for stroke_id, start, end in strokes:
            start = max(0, min(int(start), valid - 1))
            end = max(start, min(int(end), valid - 1))
            rows.append(
                {
                    "sample_index": sample_index,
                    "stroke_index": int(stroke_id),
                    "anchor_kind": "mid_stroke",
                    "anchor_step": int((start + end) // 2),
                }
            )
            rows.append(
                {
                    "sample_index": sample_index,
                    "stroke_index": int(stroke_id),
                    "anchor_kind": "stroke_end",
                    "anchor_step": int(end),
                }
            )
    return rows


def _full_features_at_records(
    model: torch.nn.Module,
    X: np.ndarray,
    lengths: np.ndarray,
    records: list[dict[str, Any]],
    config: Config,
) -> dict[str, dict[str, np.ndarray]]:
    if not records:
        return {}
    device = torch.device(config.device)
    by_sample: dict[int, list[int]] = {}
    for record_index, record in enumerate(records):
        by_sample.setdefault(int(record["sample_index"]), []).append(record_index)

    output: dict[str, dict[str, list[np.ndarray | None]]] = {}
    for sample_start in range(0, len(X), config.batch_size):
        sample_end = min(sample_start + config.batch_size, len(X))
        active = [
            sample_index
            for sample_index in range(sample_start, sample_end)
            if sample_index in by_sample
        ]
        if not active:
            continue
        xb = torch.tensor(X[active], dtype=torch.float32, device=device)
        replay = _replay(model, xb)
        views = _state_views(replay)
        local_index = {sample_index: i for i, sample_index in enumerate(active)}
        for layer, states in views.items():
            output.setdefault(layer, {})
            for state in ANALYSIS_STATES:
                output[layer].setdefault(state, [None] * len(records))
                values = states[state].detach().cpu().numpy()
                for sample_index in active:
                    row_index = local_index[sample_index]
                    for record_index in by_sample[sample_index]:
                        anchor = int(records[record_index]["anchor_step"])
                        output[layer][state][record_index] = values[
                            row_index, anchor
                        ].astype(np.float32)
    finalized: dict[str, dict[str, np.ndarray]] = {}
    for layer, states in output.items():
        finalized[layer] = {}
        for state, values in states.items():
            if any(value is None for value in values):
                raise RuntimeError("Missing full feature for stroke anchor")
            finalized[layer][state] = np.stack(
                [np.asarray(value) for value in values],
                axis=0,
            )
    return finalized


def _a2_paths(config: Config, spec: A2Spec) -> tuple[Path, Path]:
    root = _subdir(config, "A2_history_truncation")
    return (
        root / "similarity_runs" / f"{spec.key}.csv",
        root / "features" / f"{spec.key}.npz",
    )


def run_a2(
    spec: A2Spec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    if (
        spec.seed not in SEEDS
        or spec.model_kind not in TRUNC_MODEL_KINDS
        or spec.history_ms not in HISTORY_MS
    ):
        raise ValueError(spec)
    similarity_path, feature_path = _a2_paths(config, spec)
    if similarity_path.exists() and feature_path.exists() and not force:
        return {"status": "exists", "key": spec.key}

    data, split_frames = exp121.prepare_data_with_frames(config.repo_root)
    stroke_index = exp121._load_stroke_index(config.repo_root, split_frames)
    model = _load_model(config, data, spec.seed, spec.model_kind)
    history_steps = max(1, int(round(spec.history_ms * data.fs / 1000.0)))
    torch.set_num_threads(config.threads)

    full_anchor_path = _a1_paths(
        config,
        A1Spec(spec.seed, spec.model_kind),
    )[2]
    if not full_anchor_path.exists():
        raise FileNotFoundError(
            f"A2 requires A1 full-history anchors: {full_anchor_path}"
        )
    full_anchor = np.load(full_anchor_path)

    feature_arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    for split, (X, y, lengths) in _split_arrays(data).items():
        sample_indices = np.arange(len(X), dtype=np.int64)
        feature_arrays[f"{split}__y"] = y.astype(np.int64)
        feature_arrays[f"{split}__lengths"] = lengths.astype(np.int64)
        for phase in PHASES:
            phase_key = f"p{int(round(phase * 100)):03d}"
            anchor_steps = _phase_steps(lengths, phase)
            truncated = _suffix_features(
                model,
                X,
                sample_indices,
                anchor_steps,
                history_steps,
                config,
            )
            for layer, states in truncated.items():
                for state, values in states.items():
                    key = f"{split}__{layer}__{state}__{phase_key}"
                    feature_arrays[key] = values
                    reference = np.asarray(full_anchor[key], dtype=np.float32)
                    rows.append(
                        {
                            "seed": spec.seed,
                            "model_kind": spec.model_kind,
                            "history_ms": spec.history_ms,
                            "history_steps": history_steps,
                            "split": split,
                            "anchor_type": "phase",
                            "anchor_value": phase,
                            "layer": layer,
                            "state": state,
                            **_similarity_summary(reference, values),
                        }
                    )

        if split == "test":
            records = _stroke_anchor_records(
                split_frames[split],
                stroke_index,
                lengths,
            )
            if records:
                record_indices = np.asarray(
                    [int(record["sample_index"]) for record in records],
                    dtype=np.int64,
                )
                record_steps = np.asarray(
                    [int(record["anchor_step"]) for record in records],
                    dtype=np.int64,
                )
                full = _full_features_at_records(
                    model,
                    X,
                    lengths,
                    records,
                    config,
                )
                truncated = _suffix_features(
                    model,
                    X,
                    record_indices,
                    record_steps,
                    history_steps,
                    config,
                )
                for layer in full:
                    for state in ANALYSIS_STATES:
                        for anchor_kind in ("mid_stroke", "stroke_end"):
                            mask = np.asarray(
                                [
                                    record["anchor_kind"] == anchor_kind
                                    for record in records
                                ],
                                dtype=bool,
                            )
                            if not bool(mask.any()):
                                continue
                            rows.append(
                                {
                                    "seed": spec.seed,
                                    "model_kind": spec.model_kind,
                                    "history_ms": spec.history_ms,
                                    "history_steps": history_steps,
                                    "split": split,
                                    "anchor_type": "stroke",
                                    "anchor_value": anchor_kind,
                                    "layer": layer,
                                    "state": state,
                                    **_similarity_summary(
                                        full[layer][state][mask],
                                        truncated[layer][state][mask],
                                    ),
                                }
                            )

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_path, **feature_arrays)
    _save_csv(similarity_path, rows)
    return {
        "status": "completed",
        "key": spec.key,
        "similarity_rows": len(rows),
    }


def _reset_layer_indices(reset_case: str) -> tuple[int, ...]:
    if reset_case == "intact":
        return ()
    if reset_case == "reset_l1":
        return (0,)
    if reset_case == "reset_l2":
        return (1,)
    if reset_case == "reset_both":
        return (0, 1)
    raise ValueError(reset_case)


def _a3_path(config: Config, spec: A3Spec) -> Path:
    return (
        _subdir(config, "A3_layer_reset")
        / "recovery_runs"
        / f"{spec.key}.csv"
    )


def run_a3(
    spec: A3Spec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    if spec.seed not in SEEDS or spec.reset_case not in RESET_CASES:
        raise ValueError(spec)
    path = _a3_path(config, spec)
    if path.exists() and not force:
        return {"status": "exists", "key": spec.key}

    data = exp3.prepare_data(config.repo_root)
    model = _load_model(config, data, spec.seed, "a2")
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    X, _, lengths = _split_arrays(data)["test"]
    reset_layers = _reset_layer_indices(spec.reset_case)
    rows: list[dict[str, Any]] = []

    for _, _, Xb, _, lb in _iter_batches(
        X,
        np.zeros(len(X), dtype=np.int64),
        lengths,
        config.batch_size,
    ):
        xb = Xb.to(device)
        intact = _state_views(_replay(model, xb))
        for phase in RESET_PHASES:
            reset_steps_np = _phase_steps(lb, phase)
            reset_steps = torch.as_tensor(
                reset_steps_np,
                dtype=torch.long,
                device=device,
            )
            perturbed = _state_views(
                _replay(
                    model,
                    xb,
                    reset_steps=reset_steps,
                    reset_layers=reset_layers,
                )
            )
            for delay in RESET_DELAYS_STEPS:
                gather_steps = reset_steps_np + delay
                valid = gather_steps < lb
                if not bool(valid.any()):
                    continue
                for layer in intact:
                    for state in ANALYSIS_STATES:
                        ref = _gather_steps(
                            intact[layer][state],
                            np.minimum(gather_steps, lb - 1),
                        )[valid]
                        alt = _gather_steps(
                            perturbed[layer][state],
                            np.minimum(gather_steps, lb - 1),
                        )[valid]
                        rows.append(
                            {
                                "seed": spec.seed,
                                "reset_case": spec.reset_case,
                                "phase": phase,
                                "delay_steps": delay,
                                "delay_ms": 1000.0 * delay / data.fs,
                                "layer": layer,
                                "state": state,
                                **_similarity_summary(ref, alt),
                            }
                        )

    _save_csv(path, rows)
    return {"status": "completed", "key": spec.key, "rows": len(rows)}


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def _fit_probe_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    bias_mode: str,
) -> tuple[StandardScaler, LogisticRegression, float]:
    if bias_mode == "no_bias":
        scaler = StandardScaler(with_mean=False).fit(train_x)
        fit_intercept = False
    elif bias_mode == "affine":
        scaler = StandardScaler().fit(train_x)
        fit_intercept = True
    else:
        raise ValueError(bias_mode)

    train_z = scaler.transform(train_x)
    val_z = scaler.transform(val_x)
    best: tuple[float, float, LogisticRegression] | None = None
    for C in PROBE_C_GRID:
        classifier = LogisticRegression(
            C=C,
            fit_intercept=fit_intercept,
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
        raise RuntimeError("No probe candidate selected")
    return scaler, best[2], best[1]


def _eval_probe(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    return _metrics(y, classifier.predict(scaler.transform(X)))


def _b_path(config: Config, spec: BSpec) -> Path:
    return (
        _subdir(config, "B_context_expression")
        / "probe_runs"
        / f"{spec.key}.csv"
    )


def run_b(
    spec: BSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    if (
        spec.seed not in SEEDS
        or spec.model_kind not in TRUNC_MODEL_KINDS
        or spec.history_ms not in HISTORY_MS
    ):
        raise ValueError(spec)
    path = _b_path(config, spec)
    if path.exists() and not force:
        return {"status": "exists", "key": spec.key}

    full_path = _a1_paths(config, A1Spec(spec.seed, spec.model_kind))[2]
    trunc_path = _a2_paths(
        config,
        A2Spec(spec.seed, spec.model_kind, spec.history_ms),
    )[1]
    if not full_path.exists() or not trunc_path.exists():
        raise FileNotFoundError(
            f"B requires A1/A2 features: {full_path} / {trunc_path}"
        )
    full = np.load(full_path)
    trunc = np.load(trunc_path)

    layers = (
        ("L1", "L2")
        if spec.model_kind == "a2"
        else ("L1", "L2", "L3")
    )
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for state in ANALYSIS_STATES:
            for phase in PHASES:
                phase_key = f"p{int(round(phase * 100)):03d}"
                full_x = {
                    split: np.asarray(
                        full[f"{split}__{layer}__{state}__{phase_key}"],
                        dtype=np.float64,
                    )
                    for split in ("train", "val", "test")
                }
                trunc_x = {
                    split: np.asarray(
                        trunc[f"{split}__{layer}__{state}__{phase_key}"],
                        dtype=np.float64,
                    )
                    for split in ("train", "val", "test")
                }
                labels = {
                    split: np.asarray(full[f"{split}__y"], dtype=np.int64)
                    for split in ("train", "val", "test")
                }
                for bias_mode in PROBE_BIAS_MODES:
                    probe_seed = exp3.dseed(
                        spec.seed,
                        EXPERIMENT_ID,
                        "B",
                        layer,
                        state,
                        phase_key,
                        bias_mode,
                    )
                    shared_scaler, shared_classifier, shared_C = _fit_probe_model(
                        full_x["train"],
                        labels["train"],
                        full_x["val"],
                        labels["val"],
                        probe_seed,
                        bias_mode,
                    )
                    full_test = _eval_probe(
                        shared_scaler,
                        shared_classifier,
                        full_x["test"],
                        labels["test"],
                    )
                    trunc_shared = _eval_probe(
                        shared_scaler,
                        shared_classifier,
                        trunc_x["test"],
                        labels["test"],
                    )

                    retrained_scaler, retrained_classifier, retrained_C = _fit_probe_model(
                        trunc_x["train"],
                        labels["train"],
                        trunc_x["val"],
                        labels["val"],
                        probe_seed,
                        bias_mode,
                    )
                    trunc_retrained = _eval_probe(
                        retrained_scaler,
                        retrained_classifier,
                        trunc_x["test"],
                        labels["test"],
                    )
                    rows.append(
                        {
                            "seed": spec.seed,
                            "model_kind": spec.model_kind,
                            "history_ms": spec.history_ms,
                            "layer": layer,
                            "state": state,
                            "phase": phase,
                            "bias_mode": bias_mode,
                            "shared_C": shared_C,
                            "retrained_C": retrained_C,
                            "full_test_ba": full_test["balanced_accuracy"],
                            "trunc_shared_test_ba": trunc_shared["balanced_accuracy"],
                            "trunc_retrained_test_ba": trunc_retrained[
                                "balanced_accuracy"
                            ],
                            "context_drop_shared_pp": 100.0
                            * (
                                full_test["balanced_accuracy"]
                                - trunc_shared["balanced_accuracy"]
                            ),
                            "information_loss_retrained_pp": 100.0
                            * (
                                full_test["balanced_accuracy"]
                                - trunc_retrained["balanced_accuracy"]
                            ),
                            "geometry_shift_pp": 100.0
                            * (
                                trunc_retrained["balanced_accuracy"]
                                - trunc_shared["balanced_accuracy"]
                            ),
                            "full_test_accuracy": full_test["accuracy"],
                            "trunc_shared_test_accuracy": trunc_shared["accuracy"],
                            "trunc_retrained_test_accuracy": trunc_retrained["accuracy"],
                            "full_test_macro_f1": full_test["macro_f1"],
                            "trunc_shared_test_macro_f1": trunc_shared["macro_f1"],
                            "trunc_retrained_test_macro_f1": trunc_retrained["macro_f1"],
                            "fit_intercept": bias_mode == "affine",
                            "scaler_with_mean": bias_mode == "affine",
                        }
                    )
    _save_csv(path, rows)
    return {"status": "completed", "key": spec.key, "rows": len(rows)}


def _motion_arrays(
    repo_root: Path,
    data: exp3.Data,
    split_frames: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    loaded = load_acceleration_data(
        exp121._dataset_roots(repo_root),
        repository_root=repo_root,
        require_reconstruction=False,
    )
    output: dict[str, np.ndarray] = {}
    for split, frame in split_frames.items():
        values = np.zeros((len(frame), data.T, exp3.TOTAL_CHANNELS), dtype=np.float32)
        for index, row in enumerate(frame.itertuples(index=False)):
            source = np.asarray(
                loaded.packages[int(row.pi)].padded_spike_imu[int(row.si)],
                dtype=np.float32,
            )
            n = min(
                len(source),
                data.T,
                int(row.valid),
            )
            values[index, :n] = source[:n, : exp3.TOTAL_CHANNELS]
        output[split] = values
    return output


def _stroke_key(row: Any) -> tuple[str, str, int]:
    return (
        str(row.user),
        str(row.action),
        int(row.source_segment_index),
    )


def _boundary_steps(
    row: Any,
    stroke_index: dict[tuple[str, str, int], tuple[tuple[int, int, int], ...]],
    valid: int,
) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for _, start, end in stroke_index.get(_stroke_key(row), ()):
        if 0 <= start < valid:
            result.append(("press", int(start)))
        if 0 <= end < valid:
            result.append(("lift", int(end)))
    return result


def _motion_features(sample: np.ndarray, valid: int) -> np.ndarray:
    values = np.asarray(sample[:valid], dtype=np.float64)
    events = np.sum(np.abs(values[:, : exp3.EVENT_CHANNELS]), axis=1)
    accel = np.linalg.norm(values[:, exp3.EVENT_CHANNELS : exp3.EVENT_CHANNELS + 3], axis=1)
    gyro = np.linalg.norm(values[:, exp3.EVENT_CHANNELS + 3 : exp3.TOTAL_CHANNELS], axis=1)
    derivative = np.zeros(valid, dtype=np.float64)
    if valid > 1:
        derivative[1:] = np.linalg.norm(
            np.diff(values[:, exp3.EVENT_CHANNELS : exp3.TOTAL_CHANNELS], axis=0),
            axis=1,
        )
    return np.stack([events, accel, gyro, derivative], axis=1)


def _pseudo_boundaries(
    sample: np.ndarray,
    valid: int,
    real_steps: list[int],
    seed: int,
) -> tuple[list[int], list[int]]:
    if not real_steps:
        return [], []
    features = _motion_features(sample, valid)
    blocked = np.zeros(valid, dtype=bool)
    for boundary in real_steps:
        lo = max(0, boundary - 4)
        hi = min(valid, boundary + 5)
        blocked[lo:hi] = True
    blocked[:2] = True
    blocked[max(0, valid - 2) :] = True
    candidates = np.flatnonzero(~blocked)
    if len(candidates) == 0:
        return [], []

    scale = np.std(features[candidates], axis=0)
    scale[scale < 1e-9] = 1.0
    motion_matched: list[int] = []
    for boundary in real_steps:
        distance = np.linalg.norm(
            (features[candidates] - features[boundary]) / scale,
            axis=1,
        )
        motion_matched.append(int(candidates[int(np.argmin(distance))]))

    rng = np.random.default_rng(seed)
    random_steps = rng.choice(
        candidates,
        size=len(real_steps),
        replace=len(candidates) < len(real_steps),
    ).astype(int).tolist()
    return motion_matched, random_steps


def _transition_value(values: np.ndarray, timestep: int) -> float:
    if timestep <= 0 or timestep >= len(values):
        return float("nan")
    return float(1.0 - _row_corr(values[timestep:timestep+1], values[timestep-1:timestep])[0])


def _c_paths(config: Config, spec: CSpec) -> tuple[Path, Path]:
    root = _subdir(config, "C_stroke_organization")
    return (
        root / "boundary_runs" / f"{spec.key}.csv",
        root / "within_across_runs" / f"{spec.key}.csv",
    )


def run_c(
    spec: CSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    if spec.seed not in SEEDS:
        raise ValueError(spec)
    boundary_path, within_path = _c_paths(config, spec)
    if boundary_path.exists() and within_path.exists() and not force:
        return {"status": "exists", "key": spec.key}

    data, split_frames = exp121.prepare_data_with_frames(config.repo_root)
    stroke_index = exp121._load_stroke_index(config.repo_root, split_frames)
    motion = _motion_arrays(config.repo_root, data, split_frames)
    model = _load_model(config, data, spec.seed, "a2")
    device = torch.device(config.device)
    X, _, lengths = _split_arrays(data)["test"]
    frame = split_frames["test"]
    full_motion = motion["test"]
    boundary_rows: list[dict[str, Any]] = []
    within_rows: list[dict[str, Any]] = []

    for sample_index in range(len(X)):
        valid = int(lengths[sample_index])
        row = frame.iloc[sample_index]
        real = _boundary_steps(row, stroke_index, valid)
        if not real:
            continue
        replay = _state_views(
            _replay(
                model,
                torch.tensor(
                    X[sample_index : sample_index + 1],
                    dtype=torch.float32,
                    device=device,
                ),
            )
        )
        states = {
            layer: {
                state: values[state][0].detach().cpu().numpy()
                for state in ANALYSIS_STATES
            }
            for layer, values in replay.items()
        }
        real_steps = [step for _, step in real]
        matched, random_steps = _pseudo_boundaries(
            full_motion[sample_index],
            valid,
            real_steps,
            exp3.dseed(spec.seed, EXPERIMENT_ID, "boundary", sample_index),
        )
        classes: list[tuple[str, list[tuple[str, int]]]] = [
            ("real", real),
            (
                "motion_matched",
                [(real[i][0], matched[i]) for i in range(min(len(real), len(matched)))],
            ),
            (
                "random",
                [(real[i][0], random_steps[i]) for i in range(min(len(real), len(random_steps)))],
            ),
        ]
        for boundary_class, boundaries in classes:
            for boundary_type, boundary in boundaries:
                for offset in BOUNDARY_OFFSETS_STEPS:
                    timestep = boundary + offset
                    if timestep <= 0 or timestep >= valid:
                        continue
                    for layer, state_map in states.items():
                        for state, values in state_map.items():
                            boundary_rows.append(
                                {
                                    "seed": spec.seed,
                                    "sample_index": sample_index,
                                    "boundary_class": boundary_class,
                                    "boundary_type": boundary_type,
                                    "offset_steps": offset,
                                    "offset_ms": 1000.0 * offset / data.fs,
                                    "layer": layer,
                                    "state": state,
                                    "transition": _transition_value(values, timestep),
                                }
                            )

        stroke_id = np.full(valid, -1, dtype=np.int64)
        for stroke, start, end in stroke_index.get(_stroke_key(row), ()):
            lo = max(0, int(start))
            hi = min(valid - 1, int(end))
            if hi >= lo:
                stroke_id[lo : hi + 1] = int(stroke)
        for lag in WITHIN_ACROSS_LAGS_STEPS:
            if valid <= lag:
                continue
            left_id = stroke_id[:-lag]
            right_id = stroke_id[lag:]
            within_mask = (left_id >= 0) & (left_id == right_id)
            across_mask = (left_id >= 0) & (right_id >= 0) & (left_id != right_id)
            for relation, mask in (
                ("within", within_mask),
                ("across", across_mask),
            ):
                if not bool(mask.any()):
                    continue
                left_indices = np.flatnonzero(mask)
                right_indices = left_indices + lag
                for layer, state_map in states.items():
                    for state, values in state_map.items():
                        similarities = _row_corr(
                            values[left_indices],
                            values[right_indices],
                        )
                        within_rows.append(
                            {
                                "seed": spec.seed,
                                "sample_index": sample_index,
                                "relation": relation,
                                "lag_steps": lag,
                                "lag_ms": 1000.0 * lag / data.fs,
                                "layer": layer,
                                "state": state,
                                "corr_mean": float(np.nanmean(similarities)),
                                "count": int(np.sum(np.isfinite(similarities))),
                            }
                        )

    _save_csv(boundary_path, boundary_rows)
    _save_csv(within_path, within_rows)
    return {
        "status": "completed",
        "key": spec.key,
        "boundary_rows": len(boundary_rows),
        "within_rows": len(within_rows),
    }


def _scramble_order(n_strokes: int, scale: str, seed: int) -> list[int]:
    if scale == "stroke":
        if n_strokes < 2:
            return list(range(n_strokes))
        rng = np.random.default_rng(seed)
        order = np.arange(n_strokes)
        for _ in range(16):
            candidate = rng.permutation(n_strokes)
            if not np.array_equal(candidate, order):
                return candidate.astype(int).tolist()
        return np.roll(order, 1).astype(int).tolist()
    if scale == "multi_stroke":
        if n_strokes < 4:
            return list(range(n_strokes))
        blocks = [
            list(range(start, min(start + 2, n_strokes)))
            for start in range(0, n_strokes, 2)
        ]
        if len(blocks) < 2:
            return list(range(n_strokes))
        rng = np.random.default_rng(seed)
        block_order = np.arange(len(blocks))
        for _ in range(16):
            candidate = rng.permutation(len(blocks))
            if not np.array_equal(candidate, block_order):
                return [
                    stroke
                    for block_index in candidate
                    for stroke in blocks[int(block_index)]
                ]
        return [
            stroke
            for block_index in np.roll(block_order, 1)
            for stroke in blocks[int(block_index)]
        ]
    raise ValueError(scale)


def _assemble_strokes(
    strokes: list[np.ndarray],
    order: list[int],
    gap_steps: int,
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    pieces: list[np.ndarray] = []
    positions: dict[int, tuple[int, int]] = {}
    cursor = 0
    channels = strokes[0].shape[1]
    for item_index, stroke_id in enumerate(order):
        if item_index > 0 and gap_steps > 0:
            pieces.append(np.zeros((gap_steps, channels), dtype=np.float32))
            cursor += gap_steps
        stroke = np.asarray(strokes[stroke_id], dtype=np.float32)
        start = cursor
        end = cursor + len(stroke)
        positions[int(stroke_id)] = (start, end)
        pieces.append(stroke)
        cursor = end
    return np.concatenate(pieces, axis=0), positions


def _d_path(config: Config, spec: DSpec) -> Path:
    return (
        _subdir(config, "D_stroke_scrambling")
        / "sensitivity_runs"
        / f"{spec.key}.csv"
    )


def run_d(
    spec: DSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    if (
        spec.seed not in SEEDS
        or spec.gap_ms not in SCRAMBLE_GAP_MS
        or spec.scale not in SCRAMBLE_SCALES
    ):
        raise ValueError(spec)
    path = _d_path(config, spec)
    if path.exists() and not force:
        return {"status": "exists", "key": spec.key}

    data, split_frames = exp121.prepare_data_with_frames(config.repo_root)
    stroke_index = exp121._load_stroke_index(config.repo_root, split_frames)
    model = _load_model(config, data, spec.seed, "a2")
    device = torch.device(config.device)
    X, _, lengths = _split_arrays(data)["test"]
    frame = split_frames["test"]
    gap_steps = int(round(spec.gap_ms * data.fs / 1000.0))
    rows: list[dict[str, Any]] = []

    for sample_index in range(len(X)):
        row = frame.iloc[sample_index]
        valid = int(lengths[sample_index])
        intervals = [
            (int(stroke), max(0, int(start)), min(valid - 1, int(end)))
            for stroke, start, end in stroke_index.get(_stroke_key(row), ())
            if int(end) >= int(start)
        ]
        intervals = [
            value for value in intervals if value[2] >= value[1]
        ]
        if len(intervals) < 2:
            continue
        if spec.scale == "multi_stroke" and len(intervals) < 4:
            continue
        strokes = [
            np.asarray(X[sample_index, start : end + 1], dtype=np.float32)
            for _, start, end in intervals
        ]
        original_order = list(range(len(strokes)))
        permuted_order = _scramble_order(
            len(strokes),
            spec.scale,
            exp3.dseed(
                spec.seed,
                EXPERIMENT_ID,
                "scramble",
                spec.scale,
                spec.gap_ms,
                sample_index,
            ),
        )
        if permuted_order == original_order:
            continue
        original_x, original_pos = _assemble_strokes(
            strokes,
            original_order,
            gap_steps,
        )
        permuted_x, permuted_pos = _assemble_strokes(
            strokes,
            permuted_order,
            gap_steps,
        )
        max_steps = max(len(original_x), len(permuted_x))
        original_pad = np.zeros((1, max_steps, X.shape[2]), dtype=np.float32)
        permuted_pad = np.zeros_like(original_pad)
        original_pad[0, : len(original_x)] = original_x
        permuted_pad[0, : len(permuted_x)] = permuted_x
        original = _state_views(
            _replay(
                model,
                torch.tensor(original_pad, dtype=torch.float32, device=device),
            )
        )
        permuted = _state_views(
            _replay(
                model,
                torch.tensor(permuted_pad, dtype=torch.float32, device=device),
            )
        )
        for stroke_id in original_order:
            o_start, o_end = original_pos[stroke_id]
            p_start, p_end = permuted_pos[stroke_id]
            length = min(o_end - o_start, p_end - p_start)
            start_offset = min(SCRAMBLE_ONSET_EXCLUDE_STEPS, max(0, length - 1))
            if length - start_offset <= 0:
                continue
            for layer in original:
                for state in ANALYSIS_STATES:
                    o = (
                        original[layer][state][
                            0,
                            o_start + start_offset : o_start + length,
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    p = (
                        permuted[layer][state][
                            0,
                            p_start + start_offset : p_start + length,
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    similarity = _row_corr(o, p)
                    rows.append(
                        {
                            "seed": spec.seed,
                            "sample_index": sample_index,
                            "scale": spec.scale,
                            "gap_ms": spec.gap_ms,
                            "gap_steps": gap_steps,
                            "stroke_index": int(intervals[stroke_id][0]),
                            "layer": layer,
                            "state": state,
                            "corr_mean": float(np.nanmean(similarity)),
                            "context_sensitivity": float(
                                1.0 - np.nanmean(similarity)
                            ),
                            "count": int(np.sum(np.isfinite(similarity))),
                        }
                    )

    _save_csv(path, rows)
    return {"status": "completed", "key": spec.key, "rows": len(rows)}


def _select_real_motifs(
    data: exp3.Data,
) -> list[tuple[str, np.ndarray]]:
    X, y, lengths = _split_arrays(data)["train"]
    motifs: list[tuple[str, np.ndarray]] = []
    for class_index, label in enumerate(data.labels):
        best_score = -float("inf")
        best: np.ndarray | None = None
        sample_indices = np.flatnonzero(y == class_index)
        for sample_index in sample_indices:
            valid = int(lengths[sample_index])
            if valid < REAL_MOTIF_STEPS:
                continue
            sample = X[sample_index, :valid]
            event_energy = np.sum(np.abs(sample), axis=1)
            window = np.convolve(
                event_energy,
                np.ones(REAL_MOTIF_STEPS, dtype=np.float64),
                mode="valid",
            )
            start = int(np.argmax(window))
            score = float(window[start])
            if score > best_score:
                best_score = score
                best = np.asarray(
                    sample[start : start + REAL_MOTIF_STEPS],
                    dtype=np.float32,
                )
        if best is not None:
            motifs.append((str(label), best))
    return motifs


def _response_metrics(
    values: np.ndarray,
    fs: float,
    stimulus_id: str,
    stimulus_type: str,
    seed: int,
    model_kind: str,
    layer: str,
    state: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    absolute = np.abs(values) if state != "spike" else values
    tail_start = max(0, values.shape[0] - 8)
    for neuron in range(values.shape[1]):
        response = absolute[:, neuron]
        peak_index = int(np.argmax(response))
        peak = float(response[peak_index])
        if peak <= 1e-12:
            fwhm_steps = 0
            duration_steps = 0
            peak_to_tail = 0.0
        else:
            fwhm_steps = int(np.sum(response >= 0.5 * peak))
            duration_steps = int(np.sum(response >= 0.1 * peak))
            tail = float(np.mean(response[tail_start:]))
            peak_to_tail = peak / (tail + 1e-9)
        rows.append(
            {
                "seed": seed,
                "model_kind": model_kind,
                "stimulus_type": stimulus_type,
                "stimulus_id": stimulus_id,
                "layer": layer,
                "state": state,
                "neuron": neuron,
                "peak_latency_steps": peak_index,
                "peak_latency_ms": 1000.0 * peak_index / fs,
                "peak_amplitude": peak,
                "fwhm_steps": fwhm_steps,
                "fwhm_ms": 1000.0 * fwhm_steps / fs,
                "duration_steps": duration_steps,
                "duration_ms": 1000.0 * duration_steps / fs,
                "peak_to_tail_ratio": peak_to_tail,
            }
        )
    return rows


def _e_paths(config: Config, spec: ESpec) -> tuple[Path, Path]:
    root = _subdir(config, "E_cross_tau_decoding")
    return (
        root / "neuron_runs" / f"{spec.key}.csv",
        root / "population_response" / f"{spec.key}.npz",
    )


def run_e(
    spec: ESpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    if spec.seed not in SEEDS or spec.model_kind not in A1_MODEL_KINDS:
        raise ValueError(spec)
    metrics_path, response_path = _e_paths(config, spec)
    if metrics_path.exists() and response_path.exists() and not force:
        return {"status": "exists", "key": spec.key}

    data = exp3.prepare_data(config.repo_root)
    model = _load_model(config, data, spec.seed, spec.model_kind)
    device = torch.device(config.device)
    rows: list[dict[str, Any]] = []
    population: dict[str, np.ndarray] = {}

    unit_stimuli: list[tuple[str, np.ndarray]] = []
    for channel in range(exp3.EVENT_CHANNELS):
        stimulus = np.zeros(
            (1 + IMPULSE_TAIL_STEPS, exp3.EVENT_CHANNELS),
            dtype=np.float32,
        )
        stimulus[0, channel] = 1.0
        unit_stimuli.append((f"channel_{channel}", stimulus))

    motifs = _select_real_motifs(data)
    motif_stimuli: list[tuple[str, np.ndarray]] = []
    for label, motif in motifs:
        stimulus = np.zeros(
            (len(motif) + IMPULSE_TAIL_STEPS, exp3.EVENT_CHANNELS),
            dtype=np.float32,
        )
        stimulus[: len(motif)] = motif
        motif_stimuli.append((label, stimulus))

    for stimulus_type, stimuli in (
        ("unit_impulse", unit_stimuli),
        ("real_motif", motif_stimuli),
    ):
        response_accumulator: dict[tuple[str, str], list[np.ndarray]] = {}
        for stimulus_id, stimulus in stimuli:
            replay = _replay(
                model,
                torch.tensor(
                    stimulus[None, ...],
                    dtype=torch.float32,
                    device=device,
                ),
            )
            for layer, states in replay.items():
                for state in MECHANISTIC_STATES:
                    values = (
                        states[state][0].detach().cpu().numpy().astype(np.float32)
                    )
                    rows.extend(
                        _response_metrics(
                            values,
                            data.fs,
                            stimulus_id,
                            stimulus_type,
                            spec.seed,
                            spec.model_kind,
                            layer,
                            state,
                        )
                    )
                    response_accumulator.setdefault((layer, state), []).append(
                        np.abs(values) if state != "spike" else values
                    )
        for (layer, state), values in response_accumulator.items():
            population[
                f"{stimulus_type}__{layer}__{state}"
            ] = np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)

    _save_csv(metrics_path, rows)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(response_path, **population)
    return {"status": "completed", "key": spec.key, "rows": len(rows)}


def _concat_required(
    paths: list[Path],
    expected: int,
    label: str,
) -> pd.DataFrame:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{label}: missing {len(missing)} required artifacts; first={missing[0]}"
        )
    if len(paths) != expected:
        raise RuntimeError(f"{label}: expected {expected} paths, got {len(paths)}")

    frames: list[pd.DataFrame] = []
    empty_paths: list[Path] = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            empty_paths.append(path)
            continue
        if frame.empty and len(frame.columns) == 0:
            empty_paths.append(path)
            continue
        frames.append(frame)

    if not frames:
        raise ValueError(
            f"{label}: all {len(paths)} required artifacts are empty; "
            "no eligible rows were produced"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined.attrs["empty_artifact_count"] = len(empty_paths)
    combined.attrs["empty_artifacts"] = [str(path) for path in empty_paths]
    return combined


def finalize(config: Config) -> dict[str, Any]:
    source_manifest = config.results_dir / "source_manifest.json"
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)

    a1_recurrence = _concat_required(
        [_a1_paths(config, spec)[0] for spec in a1_specs()],
        EXPECTED_A1_TASKS,
        "A1 recurrence",
    )
    a1_activity = _concat_required(
        [_a1_paths(config, spec)[1] for spec in a1_specs()],
        EXPECTED_A1_TASKS,
        "A1 activity",
    )
    a2 = _concat_required(
        [_a2_paths(config, spec)[0] for spec in a2_specs()],
        EXPECTED_A2_TASKS,
        "A2",
    )
    a3 = _concat_required(
        [_a3_path(config, spec) for spec in a3_specs()],
        EXPECTED_A3_TASKS,
        "A3",
    )
    b = _concat_required(
        [_b_path(config, spec) for spec in b_specs()],
        EXPECTED_B_TASKS,
        "B",
    )
    c_boundary = _concat_required(
        [_c_paths(config, spec)[0] for spec in c_specs()],
        EXPECTED_C_TASKS,
        "C boundary",
    )
    c_within = _concat_required(
        [_c_paths(config, spec)[1] for spec in c_specs()],
        EXPECTED_C_TASKS,
        "C within/across",
    )
    d = _concat_required(
        [_d_path(config, spec) for spec in d_specs()],
        EXPECTED_D_TASKS,
        "D",
    )
    e = _concat_required(
        [_e_paths(config, spec)[0] for spec in e_specs()],
        EXPECTED_E_TASKS,
        "E",
    )

    finalized = {
        "A1_recurrence": (
            a1_recurrence,
            ["model_kind", "layer", "state", "lag_steps"],
            ["corr_mean", "cosine_mean", "normalized_l2_mean"],
        ),
        "A1_activity": (
            a1_activity,
            ["model_kind", "layer", "state"],
            [
                "mean_state_norm",
                "mean_per_neuron_variance",
                "nonzero_fraction",
                "firing_rate_per_step",
            ],
        ),
        "A2_history_truncation": (
            a2,
            [
                "model_kind",
                "history_ms",
                "anchor_type",
                "anchor_value",
                "layer",
                "state",
            ],
            ["corr_mean", "cosine_mean", "normalized_l2_mean"],
        ),
        "A3_layer_reset": (
            a3,
            ["reset_case", "phase", "delay_steps", "layer", "state"],
            ["corr_mean", "cosine_mean", "normalized_l2_mean"],
        ),
        "B_context_expression": (
            b,
            [
                "model_kind",
                "history_ms",
                "layer",
                "state",
                "phase",
                "bias_mode",
            ],
            [
                "full_test_ba",
                "trunc_shared_test_ba",
                "trunc_retrained_test_ba",
                "context_drop_shared_pp",
                "information_loss_retrained_pp",
                "geometry_shift_pp",
            ],
        ),
        "C_boundary": (
            c_boundary,
            [
                "boundary_class",
                "boundary_type",
                "offset_steps",
                "layer",
                "state",
            ],
            ["transition"],
        ),
        "C_within_across": (
            c_within,
            ["relation", "lag_steps", "layer", "state"],
            ["corr_mean"],
        ),
        "D_stroke_scrambling": (
            d,
            ["scale", "gap_ms", "layer", "state"],
            ["corr_mean", "context_sensitivity"],
        ),
        "E_cross_tau_decoding": (
            e,
            ["model_kind", "stimulus_type", "layer", "state"],
            [
                "peak_latency_ms",
                "peak_amplitude",
                "fwhm_ms",
                "duration_ms",
                "peak_to_tail_ratio",
            ],
        ),
    }

    files: dict[str, str] = {}
    for name, (frame, group_columns, metrics) in finalized.items():
        if name.startswith("A1_"):
            root = _subdir(config, "A1_recurrence")
        elif name.startswith("C_"):
            root = _subdir(config, "C_stroke_organization")
        else:
            subexp = {
                "A2_history_truncation": "A2_history_truncation",
                "A3_layer_reset": "A3_layer_reset",
                "B_context_expression": "B_context_expression",
                "D_stroke_scrambling": "D_stroke_scrambling",
                "E_cross_tau_decoding": "E_cross_tau_decoding",
            }[name]
            root = _subdir(config, subexp)
        runs_path = root / f"{name}_all_runs.csv"
        frame.to_csv(runs_path, index=False)
        summary = (
            frame.groupby(group_columns, dropna=False, sort=False)[metrics]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary.columns = [
            "_".join(str(value) for value in column if str(value))
            if isinstance(column, tuple)
            else str(column)
            for column in summary.columns
        ]
        summary_path = root / f"{name}_summary.csv"
        summary.to_csv(summary_path, index=False)
        files[f"{name}_runs"] = str(runs_path.relative_to(config.results_dir))
        files[f"{name}_summary"] = str(summary_path.relative_to(config.results_dir))

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Do deeper multi-tau SNN layers merely filter over longer time scales, "
            "or do they causally transform local temporal evidence into a current, "
            "communicated, stroke-organized contextual representation?"
        ),
        "hypotheses": {
            "H0": "temporal filtering only",
            "H1": "history exists internally but is not effectively expressed through spikes",
            "H2": "history becomes a communicated higher-level contextual representation",
        },
        "paper_style_primary_state": "spike50",
        "mechanistic_states": ["syn_current", "pre_reset"],
        "subexperiments": list(SUBEXPERIMENTS),
        "task_counts": {
            "A1": EXPECTED_A1_TASKS,
            "A2": EXPECTED_A2_TASKS,
            "A3": EXPECTED_A3_TASKS,
            "B": EXPECTED_B_TASKS,
            "C": EXPECTED_C_TASKS,
            "D": EXPECTED_D_TASKS,
            "E": EXPECTED_E_TASKS,
        },
        "history_ms": list(HISTORY_MS),
        "phases": list(PHASES),
        "scramble_gap_ms": list(SCRAMBLE_GAP_MS),
        "scramble_scales": list(SCRAMBLE_SCALES),
        "files": files,
        "empty_artifacts": {
            "D_stroke_scrambling": int(d.attrs.get("empty_artifact_count", 0)),
        },
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp13 hierarchical temporal representation validation"
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-source")
    for name in ("a1", "a2", "a3", "b", "c", "d", "e"):
        child = sub.add_parser(name)
        child.add_argument("--array-task-id", type=int, required=True)
    sub.add_parser("finalize")
    sub.add_parser("list-runs")
    return parser.parse_args()


def _select_task(specs: list[Any], task_id: int) -> Any:
    if task_id < 0 or task_id >= len(specs):
        raise IndexError(task_id)
    return specs[task_id]


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
    )
    if args.command == "prepare-source":
        prepare_source(config)
    elif args.command == "a1":
        run_a1(_select_task(a1_specs(), args.array_task_id), config, args.force)
    elif args.command == "a2":
        run_a2(_select_task(a2_specs(), args.array_task_id), config, args.force)
    elif args.command == "a3":
        run_a3(_select_task(a3_specs(), args.array_task_id), config, args.force)
    elif args.command == "b":
        run_b(_select_task(b_specs(), args.array_task_id), config, args.force)
    elif args.command == "c":
        run_c(_select_task(c_specs(), args.array_task_id), config, args.force)
    elif args.command == "d":
        run_d(_select_task(d_specs(), args.array_task_id), config, args.force)
    elif args.command == "e":
        run_e(_select_task(e_specs(), args.array_task_id), config, args.force)
    elif args.command == "finalize":
        finalize(config)
    elif args.command == "list-runs":
        payload = {
            "A1": [asdict(spec) for spec in a1_specs()],
            "A2": [asdict(spec) for spec in a2_specs()],
            "A3": [asdict(spec) for spec in a3_specs()],
            "B": [asdict(spec) for spec in b_specs()],
            "C": [asdict(spec) for spec in c_specs()],
            "D": [asdict(spec) for spec in d_specs()],
            "E": [asdict(spec) for spec in e_specs()],
        }
        print(json.dumps(payload, indent=2))
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
