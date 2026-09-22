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

from snn.accel_reconstruction_eval.datasets import load_acceleration_data
from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_12_0_local_primitive_bottleneck as exp12


EXPERIMENT_ID = "experiment_12_1_primitive_capacity_fragmentation"
PROTOCOL_VERSION = "primitive_capacity_fragmentation_v1"
ARCHITECTURE = exp73.ARCHITECTURE
SHIFTS = exp73.SHIFTS
SEEDS = exp73.SEEDS
HIDDEN_WIDTH = exp73.HIDDEN_WIDTH
PRIMITIVE_WIDTHS = (16, 32, 64, 128)
TEMPORAL_MODES = ("t0", "ema")
BETA = exp12.BETA
EPS = exp12.EPS
TEMPERATURE = exp12.TEMPERATURE
SPARSITY_QUANTILES = exp12.SPARSITY_QUANTILES
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
MIN_SUSTAINED_RUN = 3
SHORT_LAG_STEPS = (16, 32)
R2_REPRESENTATIONS = (
    "r2_mh_qraw",
    "r2_mh_qnorm",
    "r2_ms_qraw",
    "r2_ms_qnorm",
)


@dataclass(frozen=True)
class RunSpec:
    temporal_mode: str
    primitive_width: int
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{ARCHITECTURE}__{self.temporal_mode}__k{self.primitive_width}"
            f"__seed{self.seed}"
        )


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


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


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(mode, width, seed)
        for mode in TEMPORAL_MODES
        for width in PRIMITIVE_WIDTHS
        for seed in SEEDS
    ]


def _validate_spec(spec: RunSpec) -> None:
    if spec.temporal_mode not in TEMPORAL_MODES:
        raise ValueError(spec.temporal_mode)
    if spec.primitive_width not in PRIMITIVE_WIDTHS:
        raise ValueError(spec.primitive_width)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _dataset_roots(repo_root: Path) -> list[Path]:
    base = repo_root / "outputs"
    return [
        base
        / f"action{action}_wavelets_0e5_1_2_4_8_sr_64"
        / "low-pass"
        / "aligned-board-events"
        / "segmentation_padded"
        for action in (0, 1)
    ]


def _annotation_path(repo_root: Path, user: str, action: str) -> Path:
    return (
        repo_root
        / "outputs"
        / f"action{action}_wavelets_0e5_1_2_4_8_sr_64"
        / "low-pass"
        / "aligned-board-events_writing_motion_ablation"
        / "original_reference"
        / "annotations"
        / user
        / f"action_{action}"
        / f"{user}_action_{action}_writing_intervals.csv"
    )


def _as_bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _output_to_source_segment(package: Any) -> dict[int, int]:
    manifest = pd.read_csv(package.padding_manifest_path)
    required = {"segment_index", "output_segment_index", "exported"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            f"{package.padding_manifest_path}: missing columns {', '.join(missing)}"
        )
    exported = manifest[_as_bool_series(manifest["exported"])].copy()
    mapping: dict[int, int] = {}
    for row in exported.itertuples(index=False):
        output_index = int(row.output_segment_index)
        source_index = int(row.segment_index)
        if output_index in mapping:
            raise ValueError(
                f"{package.padding_manifest_path}: duplicate output segment "
                f"{output_index}"
            )
        mapping[output_index] = source_index
    expected = set(range(package.segment_count))
    if set(mapping) != expected:
        raise ValueError(
            f"{package.padding_manifest_path}: output segment mapping does not "
            "match padded array indices"
        )
    return mapping


def prepare_data_with_frames(
    repo_root: Path,
) -> tuple[exp3.Data, dict[str, pd.DataFrame]]:
    data = exp3.prepare_data(repo_root)
    loaded = load_acceleration_data(
        _dataset_roots(repo_root),
        repository_root=repo_root,
        require_reconstruction=False,
    )
    rows: list[dict[str, Any]] = []
    keep = set(exp3.LABELS)
    for package_index, package in enumerate(loaded.packages):
        source_map = _output_to_source_segment(package)
        for segment_index, label_value in enumerate(package.labels.astype(str)):
            label = str(label_value)
            if label not in keep:
                continue
            rows.append(
                {
                    "pi": package_index,
                    "si": segment_index,
                    "source_segment_index": source_map[segment_index],
                    "user": str(package.user),
                    "action": str(package.action),
                    "label": label,
                    "valid": int(package.valid_lengths[segment_index]),
                    "pad": int(package.padded_spike_imu.shape[1]),
                }
            )
    manifest = pd.DataFrame(rows)
    labels = tuple(sorted(manifest["label"].unique().tolist()))
    if labels != data.labels:
        raise ValueError(f"Exp3/data labels disagree: {labels} vs {data.labels}")
    class_to_idx = {label: index for index, label in enumerate(labels)}
    manifest["y"] = manifest["label"].map(class_to_idx).astype(int)

    split_frames = {
        "train": manifest[
            manifest["user"].isin(set(data.split["train_users"]))
        ].reset_index(drop=True),
        "val": manifest[
            manifest["user"].isin(set(data.split["val_users"]))
        ].reset_index(drop=True),
        "test": manifest[
            manifest["user"].isin(set(data.split["test_users"]))
        ].reset_index(drop=True),
    }
    expected = {
        "train": (data.ytr, data.ltr),
        "val": (data.yva, data.lva),
        "test": (data.yte, data.lte),
    }
    for split, frame in split_frames.items():
        y, lengths = expected[split]
        frame_y = frame["y"].to_numpy(dtype=np.int64, copy=True)
        frame_lengths = np.minimum(
            frame["valid"].to_numpy(dtype=np.int64, copy=True), data.T
        )
        if not np.array_equal(frame_y, y):
            raise ValueError(f"{split}: sample-label order disagrees with Exp3 data")
        if not np.array_equal(frame_lengths, lengths):
            raise ValueError(f"{split}: valid-length order disagrees with Exp3 data")
    return data, split_frames


def _load_stroke_index(
    repo_root: Path,
    split_frames: dict[str, pd.DataFrame],
) -> dict[tuple[str, str, int], tuple[tuple[int, int, int], ...]]:
    user_actions = {
        (str(row.user), str(row.action))
        for frame in split_frames.values()
        for row in frame.itertuples(index=False)
    }
    index: dict[tuple[str, str, int], list[tuple[int, int, int]]] = {}
    for user, action in sorted(user_actions):
        path = _annotation_path(repo_root, user, action)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing weak stroke-boundary annotation required by Exp12.1: {path}"
            )
        frame = pd.read_csv(path)
        required = {
            "segment_index",
            "stroke_index",
            "press_segment_local_index",
            "lift_segment_local_index",
            "retained_in_segment",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(missing)}")
        frame = frame[_as_bool_series(frame["retained_in_segment"])].copy()
        for row in frame.itertuples(index=False):
            if pd.isna(row.press_segment_local_index) or pd.isna(
                row.lift_segment_local_index
            ):
                continue
            segment = int(row.segment_index)
            stroke = int(row.stroke_index)
            start = int(row.press_segment_local_index)
            end = int(row.lift_segment_local_index)
            if start < 0 or end < start:
                raise ValueError(f"{path}: invalid stroke interval {start}:{end}")
            index.setdefault((user, action, segment), []).append(
                (stroke, start, end)
            )
    return {
        key: tuple(sorted(values, key=lambda value: (value[0], value[1], value[2])))
        for key, values in index.items()
    }


class PrimitiveCapacityNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        fs: float,
        temporal_mode: str,
        primitive_width: int,
    ) -> None:
        super().__init__()
        if temporal_mode not in TEMPORAL_MODES:
            raise ValueError(temporal_mode)
        if primitive_width not in PRIMITIVE_WIDTHS:
            raise ValueError(primitive_width)
        self.temporal_mode = temporal_mode
        self.primitive_width = int(primitive_width)
        self.backbone = exp73.Exp73Net("linear", n_classes, fs)
        self.backbone.output_linear.requires_grad_(False)
        self.primitive_projection = nn.Linear(
            HIDDEN_WIDTH, self.primitive_width, bias=False
        )
        self.classifier = nn.Linear(self.primitive_width, n_classes, bias=False)

    def load_a2_backbone(self, payload: dict[str, Any]) -> None:
        self.backbone.load_state_dict(payload["model_state_dict"], strict=True)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def _temporal_integrate(self, z: torch.Tensor) -> torch.Tensor:
        if self.temporal_mode == "t0":
            return z
        state = torch.zeros_like(z[:, 0])
        out: list[torch.Tensor] = []
        for t in range(z.shape[1]):
            state = BETA * state + (1.0 - BETA) * z[:, t]
            out.append(state)
        return torch.stack(out, dim=1)

    @staticmethod
    def primitive_components(
        h: torch.Tensor,
        projection: nn.Linear,
    ) -> dict[str, torch.Tensor]:
        s = projection(h)
        centered = s - s.mean(dim=-1, keepdim=True)
        spread = exp12._zero_preserving_rms(centered, dim=-1)
        normalized = centered / spread.unsqueeze(-1).clamp_min(EPS)
        q_raw = F.softmax(s / TEMPERATURE, dim=-1)
        q_norm = F.softmax(normalized / TEMPERATURE, dim=-1)
        magnitude_h = exp12._zero_preserving_rms(h, dim=-1)
        magnitude_s = exp12._zero_preserving_rms(s, dim=-1)
        raw_top2 = torch.topk(s, k=2, dim=-1).values
        norm_top2 = torch.topk(normalized, k=2, dim=-1).values
        return {
            "s": s,
            "centered": centered,
            "spread": spread,
            "normalized": normalized,
            "q_raw": q_raw,
            "q_norm": q_norm,
            "magnitude_h": magnitude_h,
            "magnitude_s": magnitude_s,
            "confidence_raw": raw_top2[..., 0] - raw_top2[..., 1],
            "confidence_norm": norm_top2[..., 0] - norm_top2[..., 1],
            "winner_raw": q_raw.argmax(dim=-1),
            "winner_norm": q_norm.argmax(dim=-1),
        }

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        backbone_tr = self.backbone.forward_trajectory(x)
        z = backbone_tr["hidden_spikes"][-1]
        h = self._temporal_integrate(z)
        primitive = self.primitive_components(h, self.primitive_projection)
        evidence = self.classifier(primitive["s"])
        return {
            "hidden_spikes": backbone_tr["hidden_spikes"],
            "z": z,
            "h": h,
            "primitive": primitive,
            "evidence": evidence,
        }


def _a2_checkpoint_path(repo_root: Path, seed: int) -> Path:
    return exp12._a2_checkpoint_path(repo_root, seed)


def _load_model(
    spec: RunSpec,
    config: Config,
) -> tuple[PrimitiveCapacityNet, exp3.Data, dict[str, pd.DataFrame]]:
    data, frames = prepare_data_with_frames(config.repo_root)
    checkpoint_path = _a2_checkpoint_path(config.repo_root, spec.seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing Exp7.3 A2 checkpoint required by Exp12.1: {checkpoint_path}"
        )
    payload = torch.load(
        checkpoint_path, map_location=config.device, weights_only=False
    )
    model = PrimitiveCapacityNet(
        len(data.labels),
        data.fs,
        spec.temporal_mode,
        spec.primitive_width,
    ).to(config.device)
    model.load_a2_backbone(payload)
    return model, data, frames


def _loss_scores(
    evidence: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = exp12._valid_sum(evidence, lengths)
    return F.cross_entropy(scores, y), scores


def _evaluate(
    model: PrimitiveCapacityNet,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            tr = model.forward_trajectory(X)
            loss, scores = _loss_scores(tr["evidence"], lengths, y)
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)
    out = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    out["objective_loss"] = loss_sum / max(n_total, 1)
    return out


def _extract_split(
    model: PrimitiveCapacityNet,
    loader: Iterable,
    device: torch.device,
) -> dict[str, np.ndarray]:
    keys = (
        "z",
        "h",
        "s",
        "normalized",
        "q_raw",
        "q_norm",
        "magnitude_h",
        "magnitude_s",
        "confidence_raw",
        "confidence_norm",
        "winner_raw",
        "winner_norm",
    )
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    ys: list[np.ndarray] = []
    lengths_all: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            tr = model.forward_trajectory(X.to(device))
            p = tr["primitive"]
            chunks["z"].append(tr["z"].cpu().numpy())
            chunks["h"].append(tr["h"].cpu().numpy())
            for key in keys[2:]:
                chunks[key].append(p[key].cpu().numpy())
            ys.append(y.numpy())
            lengths_all.append(lengths.numpy())
    out = {key: np.concatenate(values) for key, values in chunks.items()}
    out["y"] = np.concatenate(ys)
    out["lengths"] = np.concatenate(lengths_all)
    return out


def _aggregate_valid(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    valid = np.arange(values.shape[1])[None, :] < lengths[:, None]
    return (values * valid[..., None]).sum(axis=1)


def _fit_representation_probe(
    name: str,
    arrays: dict[str, np.ndarray],
    split_data: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    features = {
        split: _aggregate_valid(values, split_data[split]["lengths"])
        for split, values in arrays.items()
    }
    probe = exp01._fit_linear_probe(
        features["train"],
        labels["train"],
        features["val"],
        labels["val"],
        features["test"],
        labels["test"],
        seed,
    )
    return {
        "representation": name,
        "dimension": int(next(iter(arrays.values())).shape[-1]),
        "train_ba": float(probe["train"]["balanced_accuracy"]),
        "val_ba": float(probe["val"]["balanced_accuracy"]),
        "test_ba": float(probe["test"]["balanced_accuracy"]),
        "test_accuracy": float(probe["test"]["accuracy"]),
        "test_macro_f1": float(probe["test"]["macro_f1"]),
    }


def _representation_arrays(
    split_data: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {
        "r0_direct_l2": {},
        "r0a_integrated_128d": {},
        "r0b_analog_k": {},
        "r1_raw_what": {},
        "r1_norm_what": {},
        "r2_mh_qraw": {},
        "r2_mh_qnorm": {},
        "r2_ms_qraw": {},
        "r2_ms_qnorm": {},
        "r2_mh_qraw_hard": {},
        "r2_mh_qnorm_hard": {},
        "r2_ms_qraw_hard": {},
        "r2_ms_qnorm_hard": {},
    }
    for split, values in split_data.items():
        q_raw = values["q_raw"]
        q_norm = values["q_norm"]
        winner_raw = values["winner_raw"]
        winner_norm = values["winner_norm"]
        width = q_raw.shape[-1]
        eye = np.eye(width, dtype=np.float32)
        hard_raw = eye[winner_raw]
        hard_norm = eye[winner_norm]
        mh = values["magnitude_h"][..., None]
        ms = values["magnitude_s"][..., None]
        out["r0_direct_l2"][split] = values["z"]
        out["r0a_integrated_128d"][split] = values["h"]
        out["r0b_analog_k"][split] = values["s"]
        out["r1_raw_what"][split] = q_raw
        out["r1_norm_what"][split] = q_norm
        out["r2_mh_qraw"][split] = mh * q_raw
        out["r2_mh_qnorm"][split] = mh * q_norm
        out["r2_ms_qraw"][split] = ms * q_raw
        out["r2_ms_qnorm"][split] = ms * q_norm
        out["r2_mh_qraw_hard"][split] = mh * hard_raw
        out["r2_mh_qnorm_hard"][split] = mh * hard_norm
        out["r2_ms_qraw_hard"][split] = ms * hard_raw
        out["r2_ms_qnorm_hard"][split] = ms * hard_norm
    return out


def _effective_k(winner: np.ndarray, lengths: np.ndarray, width: int) -> float:
    valid = np.arange(winner.shape[1])[None, :] < lengths[:, None]
    counts = np.bincount(winner[valid], minlength=width).astype(np.float64)
    probs = counts / max(float(counts.sum()), 1.0)
    nz = probs > 0
    return float(np.exp(-(probs[nz] * np.log(probs[nz])).sum()))


def _mean_js(q: np.ndarray, lengths: np.ndarray) -> float:
    values: list[float] = []
    for i, length_value in enumerate(lengths):
        stop = int(length_value)
        if stop < 2:
            continue
        a = np.clip(q[i, : stop - 1], EPS, 1.0)
        b = np.clip(q[i, 1:stop], EPS, 1.0)
        m = 0.5 * (a + b)
        js = 0.5 * (
            np.sum(a * (np.log(a) - np.log(m)), axis=-1)
            + np.sum(b * (np.log(b) - np.log(m)), axis=-1)
        )
        values.extend(js.tolist())
    return float(np.mean(values)) if values else float("nan")


def _rle(sequence: np.ndarray) -> list[tuple[int, int, int]]:
    if len(sequence) == 0:
        return []
    runs: list[tuple[int, int, int]] = []
    start = 0
    current = int(sequence[0])
    for index in range(1, len(sequence)):
        value = int(sequence[index])
        if value == current:
            continue
        runs.append((current, start, index))
        current = value
        start = index
    runs.append((current, start, len(sequence)))
    return runs


def _stroke_intervals_for_sample(
    row: Any,
    stroke_index: dict[tuple[str, str, int], tuple[tuple[int, int, int], ...]],
    valid_length: int,
) -> tuple[tuple[int, int, int], ...]:
    key = (str(row.user), str(row.action), int(row.source_segment_index))
    intervals = stroke_index.get(key, ())
    clipped: list[tuple[int, int, int]] = []
    for stroke, start, end in intervals:
        start_i = max(0, int(start))
        end_i = min(int(end), valid_length - 1)
        if start_i <= end_i:
            clipped.append((int(stroke), start_i, end_i))
    return tuple(clipped)


def _stroke_fragmentation_metrics(
    winner: np.ndarray,
    q: np.ndarray,
    lengths: np.ndarray,
    frame: pd.DataFrame,
    stroke_index: dict[tuple[str, str, int], tuple[tuple[int, int, int], ...]],
) -> tuple[dict[str, float], np.ndarray]:
    width = q.shape[-1]
    transition_counts = np.zeros((width, width), dtype=np.int64)
    stroke_count = 0
    gesture_count = len(frame)
    gestures_with_stroke = 0
    unique_slots: list[int] = []
    purities: list[float] = []
    switch_counts: list[int] = []
    run_lengths: list[int] = []
    js_values: list[float] = []
    fragmented = 0
    sustained_fragmented = 0
    sustained_switches = 0
    flicker_switches = 0
    total_switches = 0

    for sample_index, row in enumerate(frame.itertuples(index=False)):
        stop = int(lengths[sample_index])
        intervals = _stroke_intervals_for_sample(row, stroke_index, stop)
        if intervals:
            gestures_with_stroke += 1
        for _, start, end in intervals:
            seq = winner[sample_index, start : end + 1]
            probs = q[sample_index, start : end + 1]
            if len(seq) == 0:
                continue
            stroke_count += 1
            counts = np.bincount(seq, minlength=width)
            unique_slots.append(int(np.count_nonzero(counts)))
            purities.append(float(counts.max() / max(len(seq), 1)))
            runs = _rle(seq)
            lengths_here = [run_end - run_start for _, run_start, run_end in runs]
            run_lengths.extend(lengths_here)
            switches = max(len(runs) - 1, 0)
            switch_counts.append(switches)
            total_switches += switches
            if switches > 0:
                fragmented += 1
            sustained_here = 0
            for run_index in range(len(runs) - 1):
                left = runs[run_index]
                right = runs[run_index + 1]
                transition_counts[left[0], right[0]] += 1
                left_length = left[2] - left[1]
                right_length = right[2] - right[1]
                if (
                    left_length >= MIN_SUSTAINED_RUN
                    and right_length >= MIN_SUSTAINED_RUN
                ):
                    sustained_here += 1
            sustained_switches += sustained_here
            if sustained_here > 0:
                sustained_fragmented += 1
            for run_index in range(1, len(runs) - 1):
                prev_run = runs[run_index - 1]
                run = runs[run_index]
                next_run = runs[run_index + 1]
                run_length = run[2] - run[1]
                if (
                    run_length <= 2
                    and prev_run[0] == next_run[0]
                    and run[0] != prev_run[0]
                ):
                    flicker_switches += 2
            if len(probs) >= 2:
                a = np.clip(probs[:-1], EPS, 1.0)
                b = np.clip(probs[1:], EPS, 1.0)
                m = 0.5 * (a + b)
                js = 0.5 * (
                    np.sum(a * (np.log(a) - np.log(m)), axis=-1)
                    + np.sum(b * (np.log(b) - np.log(m)), axis=-1)
                )
                js_values.extend(js.tolist())

    transition_steps = max(sum(lengths - 1 for lengths in [
        [end - start + 1 for _, start, end in _stroke_intervals_for_sample(
            row,
            stroke_index,
            int(lengths[index]),
        )]
        for index, row in enumerate(frame.itertuples(index=False))
    ]), 0)
    metrics = {
        "gesture_count": float(gesture_count),
        "gestures_with_stroke_fraction": float(
            gestures_with_stroke / max(gesture_count, 1)
        ),
        "stroke_count": float(stroke_count),
        "mean_slots_per_stroke": float(np.mean(unique_slots))
        if unique_slots
        else float("nan"),
        "median_slots_per_stroke": float(np.median(unique_slots))
        if unique_slots
        else float("nan"),
        "mean_dominant_slot_purity": float(np.mean(purities))
        if purities
        else float("nan"),
        "mean_switches_per_stroke": float(np.mean(switch_counts))
        if switch_counts
        else float("nan"),
        "fragmented_stroke_fraction": float(fragmented / max(stroke_count, 1)),
        "sustained_fragmented_stroke_fraction": float(
            sustained_fragmented / max(stroke_count, 1)
        ),
        "sustained_switches_per_stroke": float(
            sustained_switches / max(stroke_count, 1)
        ),
        "flicker_switches_per_stroke": float(
            flicker_switches / max(stroke_count, 1)
        ),
        "winner_run_length_mean": float(np.mean(run_lengths))
        if run_lengths
        else float("nan"),
        "winner_run_length_median": float(np.median(run_lengths))
        if run_lengths
        else float("nan"),
        "predicted_runs_per_true_stroke": float(
            (total_switches + stroke_count) / max(stroke_count, 1)
        ),
        "within_stroke_js_mean": float(np.mean(js_values))
        if js_values
        else float("nan"),
        "within_stroke_switch_rate_per_transition": float(
            total_switches / max(transition_steps, 1)
        ),
    }
    return metrics, transition_counts


def _svd_diagnostics(model: PrimitiveCapacityNet) -> dict[str, Any]:
    weight = model.primitive_projection.weight.detach().cpu().double().numpy()
    singular = np.linalg.svd(weight, compute_uv=False)
    if len(singular) == 0:
        raise RuntimeError("primitive projection has no singular values")
    max_s = float(singular.max())
    tolerance = max(weight.shape) * np.finfo(np.float64).eps * max_s
    rank = int(np.count_nonzero(singular > tolerance))
    squared = singular**2
    stable_rank = float(squared.sum() / max(float(squared.max()), EPS))
    total = float(singular.sum())
    if total > 0:
        p = singular / total
        p = p[p > 0]
        effective_rank = float(np.exp(-(p * np.log(p)).sum()))
    else:
        effective_rank = 0.0
    nonzero = singular[singular > tolerance]
    condition = float(nonzero.max() / nonzero.min()) if len(nonzero) else float("inf")
    return {
        "rank": rank,
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "condition_number_nonzero": condition,
        "singular_values": singular.tolist(),
    }


def _peak_mask(
    scores: np.ndarray,
    winner: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return exp12._peak_mask(scores, winner, lengths, threshold)


def _event_diagnostics(
    masks: dict[str, np.ndarray],
    split_data: dict[str, dict[str, np.ndarray]],
    split_frames: dict[str, pd.DataFrame],
    stroke_index: dict[tuple[str, str, int], tuple[tuple[int, int, int], ...]],
    winner_key: str,
) -> dict[str, float]:
    mask = masks["test"]
    values = split_data["test"]
    lengths = values["lengths"]
    winner = values[winner_key]
    valid = np.arange(mask.shape[1])[None, :] < lengths[:, None]
    active = mask & valid
    events = active.sum(axis=1)
    stroke_total = 0
    event_inside_stroke = 0
    cross_pairs: dict[int, int] = {lag: 0 for lag in SHORT_LAG_STEPS}
    eligible_pairs: dict[int, int] = {lag: 0 for lag in SHORT_LAG_STEPS}
    for i, row in enumerate(split_frames["test"].itertuples(index=False)):
        stop = int(lengths[i])
        intervals = _stroke_intervals_for_sample(row, stroke_index, stop)
        stroke_total += len(intervals)
        positions = np.flatnonzero(active[i, :stop])
        inside = np.zeros(stop, dtype=bool)
        for _, start, end in intervals:
            inside[start : end + 1] = True
        event_inside_stroke += int(inside[positions].sum()) if len(positions) else 0
        for left, right in zip(positions[:-1], positions[1:], strict=True):
            delta = int(right - left)
            for lag in SHORT_LAG_STEPS:
                if delta <= lag:
                    eligible_pairs[lag] += 1
                    if winner[i, left] != winner[i, right]:
                        cross_pairs[lag] += 1
    out = {
        "active_fraction_test": float(active.sum() / max(valid.sum(), 1)),
        "events_per_sample_test_mean": float(events.mean()),
        "zero_event_fraction_test": float(np.mean(events == 0)),
        "events_per_true_stroke_test": float(active.sum() / max(stroke_total, 1)),
        "event_inside_stroke_fraction_test": float(
            event_inside_stroke / max(int(active.sum()), 1)
        ),
    }
    for lag in SHORT_LAG_STEPS:
        out[f"cross_primitive_pair_fraction_le_{lag}_steps"] = float(
            cross_pairs[lag] / max(eligible_pairs[lag], 1)
        )
    return out


def _r4_rows(
    spec: RunSpec,
    split_data: dict[str, dict[str, np.ndarray]],
    split_frames: dict[str, pd.DataFrame],
    stroke_index: dict[tuple[str, str, int], tuple[tuple[int, int, int], ...]],
    labels: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    definitions = (
        ("r2_mh_qraw", "q_raw", "s", "winner_raw", "confidence_raw", "magnitude_h"),
        (
            "r2_mh_qnorm",
            "q_norm",
            "normalized",
            "winner_norm",
            "confidence_norm",
            "magnitude_h",
        ),
        ("r2_ms_qraw", "q_raw", "s", "winner_raw", "confidence_raw", "magnitude_s"),
        (
            "r2_ms_qnorm",
            "q_norm",
            "normalized",
            "winner_norm",
            "confidence_norm",
            "magnitude_s",
        ),
    )
    rows: list[dict[str, Any]] = []
    for representation, q_key, score_key, winner_key, confidence_key, magnitude_key in definitions:
        train = split_data["train"]
        train_valid = (
            np.arange(train[confidence_key].shape[1])[None, :]
            < train["lengths"][:, None]
        )
        train_confidence = train[confidence_key][train_valid]
        for quantile in SPARSITY_QUANTILES:
            threshold = float(np.quantile(train_confidence, quantile))
            masks = {
                split: _peak_mask(
                    values[score_key],
                    values[winner_key],
                    values["lengths"],
                    threshold,
                )
                for split, values in split_data.items()
            }
            routed = {
                split: values[magnitude_key][..., None] * values[q_key]
                for split, values in split_data.items()
            }
            features = {
                split: exp12._aggregate_masked(
                    routed[split], masks[split], values["lengths"]
                )
                for split, values in split_data.items()
            }
            probe = exp01._fit_linear_probe(
                features["train"],
                labels["train"],
                features["val"],
                labels["val"],
                features["test"],
                labels["test"],
                exp3.dseed(
                    spec.seed,
                    EXPERIMENT_ID,
                    spec.key,
                    "r4",
                    representation,
                    str(quantile),
                ),
            )
            row = {
                "representation": representation,
                "quantile": float(quantile),
                "threshold": threshold,
                "val_ba": float(probe["val"]["balanced_accuracy"]),
                "test_ba": float(probe["test"]["balanced_accuracy"]),
                "test_accuracy": float(probe["test"]["accuracy"]),
                "test_macro_f1": float(probe["test"]["macro_f1"]),
            }
            row.update(
                _event_diagnostics(
                    masks,
                    split_data,
                    split_frames,
                    stroke_index,
                    winner_key,
                )
            )
            rows.append(row)
    return rows


def run_one(
    spec: RunSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    _validate_spec(spec)
    eval_path = _path(config.results_dir, "evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "checkpoints", spec.key, ".pt")
    representation_path = _path(
        config.results_dir, "representation_probes", spec.key, ".csv"
    )
    r4_path = _path(config.results_dir, "r4", spec.key, ".csv")
    transition_path = _path(
        config.results_dir, "transition_matrices", spec.key, ".npz"
    )
    if (
        eval_path.exists()
        and checkpoint_path.exists()
        and representation_path.exists()
        and r4_path.exists()
        and transition_path.exists()
        and not force
        and exp12._checkpoint_model_state_is_finite(checkpoint_path)
    ):
        return json.loads(eval_path.read_text(encoding="utf-8"))

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp3.seed_all(
        exp3.dseed(
            spec.seed,
            EXPERIMENT_ID,
            spec.temporal_mode,
            spec.primitive_width,
            "analog_capacity",
        )
    )
    model, data, split_frames = _load_model(spec, config)
    stroke_index = _load_stroke_index(config.repo_root, split_frames)
    optimizer = torch.optim.Adam(
        list(model.primitive_projection.parameters())
        + list(model.classifier.parameters()),
        lr=exp72.LR,
        weight_decay=exp72.WEIGHT_DECAY,
    )
    train_loader = exp73._raw_loaders(
        data, spec.seed, config.batch_size, shuffle_train=True
    )["train"]
    eval_loaders = exp73._raw_loaders(
        data, spec.seed, config.batch_size, shuffle_train=False
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
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            tr = model.forward_trajectory(X)
            loss, _ = _loss_scores(tr["evidence"], lengths, y)
            exp12._assert_finite_tensor("training loss", loss)
            loss.backward()
            exp12._assert_finite_trainable_state(model, gradients=True)
            optimizer.step()
            exp12._assert_finite_trainable_state(model, gradients=False)
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate(model, eval_loaders["train"], device)
        val_metrics = _evaluate(model, eval_loaders["val"], device)
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["objective_loss"])
        if not math.isfinite(val_ba) or not math.isfinite(val_loss):
            raise RuntimeError(
                f"Non-finite validation metric for {spec.key}: "
                f"balanced_accuracy={val_ba}, objective_loss={val_loss}"
            )
        if exp73._checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = val_ba
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE:
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "model_state_dict": best_state,
            "source_a2_checkpoint": str(_a2_checkpoint_path(config.repo_root, spec.seed)),
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state, strict=True)

    native = {
        split: _evaluate(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    split_data = {
        split: _extract_split(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    for split, frame in split_frames.items():
        if len(frame) != len(split_data[split]["y"]):
            raise ValueError(f"{split}: frame/trajectory sample count mismatch")
        if not np.array_equal(
            frame["y"].to_numpy(dtype=np.int64), split_data[split]["y"]
        ):
            raise ValueError(f"{split}: frame/trajectory sample order mismatch")

    labels = {split: values["y"] for split, values in split_data.items()}
    arrays = _representation_arrays(split_data)
    representation_rows = [
        _fit_representation_probe(
            name,
            values,
            split_data,
            labels,
            exp3.dseed(spec.seed, EXPERIMENT_ID, spec.key, "probe", name),
        )
        for name, values in arrays.items()
    ]
    representation_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(representation_rows).to_csv(representation_path, index=False)

    fragmentation: dict[str, dict[str, dict[str, float]]] = {}
    transition_payload: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        fragmentation[split] = {}
        for what, q_key, winner_key in (
            ("raw", "q_raw", "winner_raw"),
            ("norm", "q_norm", "winner_norm"),
        ):
            metrics, counts = _stroke_fragmentation_metrics(
                split_data[split][winner_key],
                split_data[split][q_key],
                split_data[split]["lengths"],
                split_frames[split],
                stroke_index,
            )
            metrics["primitive_effective_k"] = _effective_k(
                split_data[split][winner_key],
                split_data[split]["lengths"],
                spec.primitive_width,
            )
            metrics["global_q_js_mean"] = _mean_js(
                split_data[split][q_key],
                split_data[split]["lengths"],
            )
            fragmentation[split][what] = metrics
            transition_payload[f"{split}_{what}_within_stroke_counts"] = counts
            row_sums = counts.sum(axis=1, keepdims=True)
            transition_payload[f"{split}_{what}_within_stroke_probability"] = np.divide(
                counts,
                row_sums,
                out=np.zeros_like(counts, dtype=np.float64),
                where=row_sums > 0,
            )
    transition_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(transition_path, **transition_payload)

    r4_rows = _r4_rows(
        spec,
        split_data,
        split_frames,
        stroke_index,
        labels,
    )
    r4_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(r4_rows).to_csv(r4_path, index=False)

    svd = _svd_diagnostics(model)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "architecture": ARCHITECTURE,
            "shifts": [list(value) for value in SHIFTS],
            "random_seeds": list(SEEDS),
            "primitive_widths": list(PRIMITIVE_WIDTHS),
            "temporal_modes": list(TEMPORAL_MODES),
            "backbone_start": "Exp7.3 A2 pretrained checkpoint",
            "backbone_trainable": False,
            "training_representation": "analog s_t = W_p h_t",
            "aggregation": "valid_timestep_sum",
            "classifier_bias": False,
            "primitive_projection_bias": False,
            "stroke_annotations": (
                "weak press/lift boundaries used for diagnosis only; "
                "no semantic stroke identity labels"
            ),
            "no_fixed_or_sliding_window_in_primitive_assignment": True,
        },
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native,
        "svd": svd,
        "fragmentation": fragmentation,
        "representation_probe_csv": str(representation_path),
        "r4_csv": str(r4_path),
        "transition_matrix_npz": str(transition_path),
    }
    _save_json(eval_path, payload)
    history_path = _path(config.results_dir, "histories", spec.key, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return payload


def _find_representation_row(frame: pd.DataFrame, name: str) -> pd.Series:
    rows = frame[frame["representation"] == name]
    if len(rows) != 1:
        raise ValueError(f"Expected one {name} row, got {len(rows)}")
    return rows.iloc[0]


def _run_row(
    spec: RunSpec,
    payload: dict[str, Any],
    representation_frame: pd.DataFrame,
) -> dict[str, Any]:
    r0a = _find_representation_row(representation_frame, "r0a_integrated_128d")
    r0b = _find_representation_row(representation_frame, "r0b_analog_k")
    raw = payload["fragmentation"]["test"]["raw"]
    norm = payload["fragmentation"]["test"]["norm"]
    test = payload["native_metrics"]["test"]
    return {
        "temporal_mode": spec.temporal_mode,
        "primitive_width": spec.primitive_width,
        "seed": spec.seed,
        "best_epoch": int(payload["best_epoch"]),
        "analog_native_test_ba": float(test["balanced_accuracy"]),
        "r0a_val_ba": float(r0a["val_ba"]),
        "r0a_test_ba": float(r0a["test_ba"]),
        "r0b_val_ba": float(r0b["val_ba"]),
        "r0b_test_ba": float(r0b["test_ba"]),
        "projection_gap_val_pp": 100.0 * float(r0b["val_ba"] - r0a["val_ba"]),
        "projection_gap_test_pp": 100.0 * float(r0b["test_ba"] - r0a["test_ba"]),
        "wp_rank": int(payload["svd"]["rank"]),
        "wp_stable_rank": float(payload["svd"]["stable_rank"]),
        "wp_effective_rank": float(payload["svd"]["effective_rank"]),
        "raw_effective_k": float(raw["primitive_effective_k"]),
        "raw_slots_per_stroke": float(raw["mean_slots_per_stroke"]),
        "raw_stroke_purity": float(raw["mean_dominant_slot_purity"]),
        "raw_within_stroke_switches": float(raw["mean_switches_per_stroke"]),
        "raw_sustained_fragmented_fraction": float(
            raw["sustained_fragmented_stroke_fraction"]
        ),
        "norm_effective_k": float(norm["primitive_effective_k"]),
        "norm_slots_per_stroke": float(norm["mean_slots_per_stroke"]),
        "norm_stroke_purity": float(norm["mean_dominant_slot_purity"]),
        "norm_within_stroke_switches": float(norm["mean_switches_per_stroke"]),
        "norm_sustained_fragmented_fraction": float(
            norm["sustained_fragmented_stroke_fraction"]
        ),
    }


def finalize(config: Config) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    representation_frames: list[pd.DataFrame] = []
    r4_frames: list[pd.DataFrame] = []
    for spec in run_specs():
        eval_path = _path(config.results_dir, "evaluations", spec.key, ".json")
        rep_path = _path(
            config.results_dir, "representation_probes", spec.key, ".csv"
        )
        r4_path = _path(config.results_dir, "r4", spec.key, ".csv")
        transition_path = _path(
            config.results_dir, "transition_matrices", spec.key, ".npz"
        )
        for path in (eval_path, rep_path, r4_path, transition_path):
            if not path.exists():
                raise FileNotFoundError(path)
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        rep = pd.read_csv(rep_path)
        r4 = pd.read_csv(r4_path)
        run_rows.append(_run_row(spec, payload, rep))
        for frame in (rep, r4):
            frame.insert(0, "seed", spec.seed)
            frame.insert(0, "primitive_width", spec.primitive_width)
            frame.insert(0, "temporal_mode", spec.temporal_mode)
        representation_frames.append(rep)
        r4_frames.append(r4)

    runs = pd.DataFrame(run_rows)
    runs.to_csv(config.results_dir / "runs.csv", index=False)
    summary = (
        runs.groupby(["temporal_mode", "primitive_width"], sort=False)
        .agg(
            r0a_test_ba_mean=("r0a_test_ba", "mean"),
            r0b_test_ba_mean=("r0b_test_ba", "mean"),
            projection_gap_test_pp_mean=("projection_gap_test_pp", "mean"),
            wp_effective_rank_mean=("wp_effective_rank", "mean"),
            raw_effective_k_mean=("raw_effective_k", "mean"),
            raw_slots_per_stroke_mean=("raw_slots_per_stroke", "mean"),
            raw_stroke_purity_mean=("raw_stroke_purity", "mean"),
            raw_within_stroke_switches_mean=("raw_within_stroke_switches", "mean"),
            raw_sustained_fragmented_fraction_mean=(
                "raw_sustained_fragmented_fraction",
                "mean",
            ),
            norm_effective_k_mean=("norm_effective_k", "mean"),
            norm_slots_per_stroke_mean=("norm_slots_per_stroke", "mean"),
            norm_stroke_purity_mean=("norm_stroke_purity", "mean"),
            norm_within_stroke_switches_mean=("norm_within_stroke_switches", "mean"),
            norm_sustained_fragmented_fraction_mean=(
                "norm_sustained_fragmented_fraction",
                "mean",
            ),
        )
        .reset_index()
    )
    summary.to_csv(config.results_dir / "capacity_fragmentation_summary.csv", index=False)

    representations = pd.concat(representation_frames, ignore_index=True)
    representations.to_csv(
        config.results_dir / "representation_runs.csv", index=False
    )
    representation_summary = (
        representations.groupby(
            ["temporal_mode", "primitive_width", "representation"],
            sort=False,
        )
        .agg(
            val_ba_mean=("val_ba", "mean"),
            val_ba_std=("val_ba", "std"),
            test_ba_mean=("test_ba", "mean"),
            test_ba_std=("test_ba", "std"),
        )
        .reset_index()
    )
    representation_summary.to_csv(
        config.results_dir / "representation_summary.csv", index=False
    )

    r4 = pd.concat(r4_frames, ignore_index=True)
    r4.to_csv(config.results_dir / "r4_runs.csv", index=False)
    r4_summary = (
        r4.groupby(
            ["temporal_mode", "primitive_width", "representation", "quantile"],
            sort=False,
        )
        .agg(
            val_ba_mean=("val_ba", "mean"),
            test_ba_mean=("test_ba", "mean"),
            events_per_sample_test_mean=("events_per_sample_test_mean", "mean"),
            events_per_true_stroke_test_mean=(
                "events_per_true_stroke_test",
                "mean",
            ),
            cross_primitive_pair_fraction_le_16_steps_mean=(
                "cross_primitive_pair_fraction_le_16_steps",
                "mean",
            ),
            cross_primitive_pair_fraction_le_32_steps_mean=(
                "cross_primitive_pair_fraction_le_32_steps",
                "mean",
            ),
        )
        .reset_index()
    )
    r4_summary.to_csv(config.results_dir / "r4_summary.csv", index=False)

    soft = representation_summary[
        representation_summary["representation"].isin(R2_REPRESENTATIONS)
    ].copy()
    best_index = soft.groupby(
        ["temporal_mode", "primitive_width"], sort=False
    )["val_ba_mean"].idxmax()
    best_soft = soft.loc[best_index].sort_values(
        ["temporal_mode", "primitive_width"]
    )
    best_soft.to_csv(config.results_dir / "best_softmax_by_k.csv", index=False)
    best_k_index = best_soft.groupby("temporal_mode", sort=False)[
        "val_ba_mean"
    ].idxmax()
    k_cls = best_soft.loc[best_k_index].copy()
    k_cls.to_csv(config.results_dir / "k_cls_selection.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS",
        "architecture": ARCHITECTURE,
        "shifts": [list(value) for value in SHIFTS],
        "seeds": list(SEEDS),
        "primitive_widths": list(PRIMITIVE_WIDTHS),
        "temporal_modes": list(TEMPORAL_MODES),
        "run_count": len(run_specs()),
        "array_strategy": (
            "one independent (temporal_mode, K, seed) run per one-CPU Slurm task"
        ),
        "weak_stroke_supervision": (
            "press/lift start/end and stroke count are used only for "
            "fragmentation evaluation; stroke semantic identity is unavailable"
        ),
        "notes": {
            "raw_removed": "Exp12.1 retains T0 and EMA only.",
            "backbone": "Exp7.3 A2 L1/L2 is identical and frozen for every run.",
            "r4": (
                "R4 remains a post-hoc same-primitive confidence-peak probe; "
                "no refractory and no selection training."
            ),
            "k_sparse": (
                "No automatic K_sparse is declared because the acceptable BA "
                "tolerance is intentionally not hard-coded; inspect the finalized "
                "BA/event/fragmentation Pareto tables."
            ),
            "no_rsnn": (
                "Exp12.1 diagnoses capacity and within-stroke phase fragmentation "
                "before adding any recurrent contextualizer."
            ),
        },
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exp12.1 primitive-capacity sweep with weak stroke-boundary "
            "fragmentation diagnosis"
        )
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--force", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--array-task-id", type=int, required=True)
    sub.add_parser("finalize")
    sub.add_parser("list-runs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    config = Config(
        repo_root=root,
        results_dir=results_dir(root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        max_epochs=args.max_epochs,
    )
    if args.command == "list-runs":
        for index, spec in enumerate(run_specs()):
            print(index, spec.key)
        return
    if args.command == "run":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        run_one(specs[args.array_task_id], config, force=args.force)
        return
    if args.command == "finalize":
        finalize(config)
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
