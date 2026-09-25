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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_3_9_pretrained_a2_depth_extension as exp739
from scripts import experiment_12_1_primitive_capacity_fragmentation as exp121
from scripts import experiment_13_hierarchical_temporal_representation as exp13


EXPERIMENT_ID = "experiment_13_1_abstraction_generalization"
PROTOCOL_VERSION = "abstraction_generalization_v1"

SEEDS = exp13.SEEDS
MODEL_KINDS = ("c1", "c2")
ANALYSIS_STATES = exp13.ANALYSIS_STATES
PHASES = (0.25, 0.50, 0.75, 1.00)
H_PHASES = (0.50, 0.75, 1.00)
HISTORY_MS = exp13.HISTORY_MS
H_COMMON_HISTORY_STEPS = 64
ALIGNMENT_STEPS = 64
DTW_BAND_FRACTION = 0.20
PAIR_CAP_PER_STRATUM = 1
USER_FOLDS = 3
H_ID_FOLDS = 3
USER_PERMUTATIONS = 100
PROBE_C_GRID = exp13.PROBE_C_GRID
PROBE_MAX_ITER = exp13.PROBE_MAX_ITER
BATCH_SIZE = exp13.BATCH_SIZE

SUBEXPERIMENTS = (
    "F_cross_user_geometry",
    "G_user_leakage",
    "H_history_generalization",
)


@dataclass(frozen=True)
class CacheSpec:
    model_kind: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.model_kind}__seed{self.seed}"


@dataclass(frozen=True)
class MetricSpec:
    model_kind: str
    seed: int
    state: str

    @property
    def key(self) -> str:
        return f"{self.model_kind}__{self.state}__seed{self.seed}"


@dataclass(frozen=True)
class HFeatureSpec:
    seed: int
    history_ms: int

    @property
    def key(self) -> str:
        return f"c1__h{self.history_ms}ms__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = BATCH_SIZE
    threads: int = 1
    user_permutations: int = USER_PERMUTATIONS


def find_repo_root(start: Path | None = None) -> Path:
    return exp13.find_repo_root(start)


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


def cache_specs() -> list[CacheSpec]:
    return [
        CacheSpec(model_kind, seed)
        for seed in SEEDS
        for model_kind in MODEL_KINDS
    ]


def metric_specs() -> list[MetricSpec]:
    return [
        MetricSpec(model_kind, seed, state)
        for seed in SEEDS
        for model_kind in MODEL_KINDS
        for state in ANALYSIS_STATES
    ]


def h_feature_specs() -> list[HFeatureSpec]:
    return [
        HFeatureSpec(seed, history_ms)
        for seed in SEEDS
        for history_ms in HISTORY_MS
    ]


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _save_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def _exp739_config(config: Config) -> exp739.Config:
    return exp739.Config(
        repo_root=config.repo_root,
        results_dir=exp739.results_dir(config.repo_root),
        device=config.device,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _exp13_config(config: Config) -> exp13.Config:
    return exp13.Config(
        repo_root=config.repo_root,
        results_dir=exp13.results_dir(config.repo_root),
        device=config.device,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _model_checkpoint_path(
    config: Config,
    model_kind: str,
    seed: int,
) -> Path:
    root = exp739.results_dir(config.repo_root)
    case = exp739.C1_CASE if model_kind == "c1" else exp739.C2_CASE
    return root / case / "checkpoints" / f"{case}__seed{seed}.pt"


def _load_model(
    config: Config,
    data: exp3.Data,
    model_kind: str,
    seed: int,
) -> torch.nn.Module:
    if model_kind == "c1":
        model, payload, path = exp739._load_c1_model(
            _exp739_config(config),
            data,
            seed,
        )
        expected = asdict(exp739.TrainSpec(exp739.C1_CASE, seed))
        if payload.get("spec") != expected:
            raise ValueError(f"C1 checkpoint identity mismatch: {path}")
    elif model_kind == "c2":
        model = exp13._load_model(
            _exp13_config(config),
            data,
            seed,
            "c2",
        )
    else:
        raise ValueError(model_kind)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _sample_manifest(
    split_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, frame in split_frames.items():
        for sample_index, row in enumerate(frame.itertuples(index=False)):
            user = str(row.user)
            action = str(row.action)
            source_segment_index = int(row.source_segment_index)
            rows.append(
                {
                    "split": split,
                    "sample_index": sample_index,
                    "sample_id": (
                        f"{user}__a{action}__seg{source_segment_index}"
                    ),
                    "user": user,
                    "action": action,
                    "label": str(row.label),
                    "y": int(row.y),
                    "source_segment_index": source_segment_index,
                    "valid_length": int(row.valid),
                }
            )
    return pd.DataFrame(rows)


def _make_h_folds(train_manifest: pd.DataFrame) -> pd.DataFrame:
    out = train_manifest.copy()
    out["h_fold"] = -1
    for _, indexes in out.groupby(["user", "label"]).groups.items():
        ordered = sorted(
            list(indexes),
            key=lambda index: str(out.loc[index, "sample_id"]),
        )
        for order, index in enumerate(ordered):
            out.loc[index, "h_fold"] = order % H_ID_FOLDS
    if bool((out["h_fold"] < 0).any()):
        raise RuntimeError("Missing H fold assignment")
    return out


def _pair_row(
    domain: str,
    relation: str,
    a: dict[str, Any],
    b: dict[str, Any],
) -> dict[str, Any]:
    return {
        "domain": domain,
        "relation": relation,
        "i": int(a["sample_index"]),
        "j": int(b["sample_index"]),
        "sample_i": str(a["sample_id"]),
        "sample_j": str(b["sample_id"]),
        "user_i": str(a["user"]),
        "user_j": str(b["user"]),
        "label_i": str(a["label"]),
        "label_j": str(b["label"]),
        "length_i": int(a["valid_length"]),
        "length_j": int(b["valid_length"]),
    }


def _make_pair_manifest(
    sample_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        exp3.dseed(0, EXPERIMENT_ID, "pair_manifest")
    )
    rows: list[dict[str, Any]] = []
    for domain in ("train", "test"):
        frame = sample_manifest[
            sample_manifest["split"] == domain
        ].copy()
        frame = frame.sort_values("sample_id").reset_index(drop=True)
        by_user = {
            user: group.reset_index(drop=True)
            for user, group in frame.groupby("user", sort=True)
        }

        for user, group in by_user.items():
            for _, part in group.groupby("label", sort=True):
                records = part.to_dict("records")
                candidates = [
                    (records[i], records[j])
                    for i in range(len(records))
                    for j in range(i + 1, len(records))
                ]
                for a, b in candidates[:PAIR_CAP_PER_STRATUM]:
                    rows.append(_pair_row(domain, "SC_SU", a, b))

        users = sorted(by_user)
        for u_index, user_a in enumerate(users):
            for user_b in users[u_index + 1 :]:
                group_a = by_user[user_a]
                group_b = by_user[user_b]
                common_labels = sorted(
                    set(group_a["label"]) & set(group_b["label"])
                )
                for label in common_labels:
                    a_part = group_a[group_a["label"] == label]
                    b_part = group_b[group_b["label"] == label]
                    candidates = []
                    for a in a_part.to_dict("records"):
                        for b in b_part.to_dict("records"):
                            diff = abs(
                                int(a["valid_length"])
                                - int(b["valid_length"])
                            )
                            candidates.append((diff, a, b))
                    candidates.sort(
                        key=lambda item: (
                            item[0],
                            str(item[1]["sample_id"]),
                            str(item[2]["sample_id"]),
                        )
                    )
                    for _, a, b in candidates[:PAIR_CAP_PER_STRATUM]:
                        rows.append(_pair_row(domain, "SC_CU", a, b))
                        negative = group_b[group_b["label"] != label].copy()
                        if negative.empty:
                            continue
                        negative["duration_delta"] = (
                            negative["valid_length"].astype(int)
                            - int(a["valid_length"])
                        ).abs()
                        best_delta = int(negative["duration_delta"].min())
                        top = negative[
                            negative["duration_delta"] == best_delta
                        ].sort_values("sample_id")
                        pick = int(
                            rng.integers(0, max(1, min(len(top), 3)))
                        )
                        rows.append(
                            _pair_row(
                                domain,
                                "DC_CU",
                                a,
                                top.iloc[pick].to_dict(),
                            )
                        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Pair manifest is empty")
    return frame


def prepare_source(config: Config) -> dict[str, Any]:
    data, split_frames = exp121.prepare_data_with_frames(config.repo_root)
    sample_manifest = _sample_manifest(split_frames)
    _save_csv(config.results_dir / "sample_manifest.csv", sample_manifest)

    for split, (_, y, lengths) in exp13._split_arrays(data).items():
        frame = sample_manifest[sample_manifest["split"] == split]
        if not np.array_equal(
            frame["y"].to_numpy(dtype=np.int64),
            np.asarray(y, dtype=np.int64),
        ):
            raise ValueError(f"{split}: sample manifest label order mismatch")
        expected_lengths = np.minimum(
            frame["valid_length"].to_numpy(dtype=np.int64),
            data.T,
        )
        if not np.array_equal(
            expected_lengths,
            np.asarray(lengths, dtype=np.int64),
        ):
            raise ValueError(f"{split}: sample manifest length order mismatch")

    _save_csv(
        config.results_dir / "pair_manifest.csv",
        _make_pair_manifest(sample_manifest),
    )
    _save_csv(
        config.results_dir / "h_fold_manifest.csv",
        _make_h_folds(
            sample_manifest[
                sample_manifest["split"] == "train"
            ].copy()
        ),
    )

    sources: list[dict[str, Any]] = []
    for seed in SEEDS:
        for model_kind in MODEL_KINDS:
            path = _model_checkpoint_path(config, model_kind, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            model = _load_model(config, data, model_kind, seed)
            del model
            sources.append(
                {
                    "model_kind": model_kind,
                    "seed": seed,
                    "checkpoint": str(path.relative_to(config.repo_root)),
                }
            )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "models": list(MODEL_KINDS),
        "seeds": list(SEEDS),
        "analysis_states": list(ANALYSIS_STATES),
        "alignment_steps": ALIGNMENT_STEPS,
        "dtw_band_fraction": DTW_BAND_FRACTION,
        "history_ms": list(HISTORY_MS),
        "h_phases": list(H_PHASES),
        "user_permutations": config.user_permutations,
        "sources": sources,
        "task_counts": {
            "F_cache": len(cache_specs()),
            "F_metric": len(metric_specs()),
            "G": len(metric_specs()),
            "H_feature": len(h_feature_specs()),
            "H": len(metric_specs()),
        },
    }
    _save_json(config.results_dir / "source_manifest.json", payload)
    return payload


def _cache_path(config: Config, spec: CacheSpec) -> Path:
    return (
        _subdir(config, "F_cross_user_geometry")
        / "trajectories"
        / f"{spec.key}.npz"
    )


def run_f_cache(
    spec: CacheSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    path = _cache_path(config, spec)
    if path.exists() and not force:
        return {"status": "exists", "key": spec.key}

    data = exp3.prepare_data(config.repo_root)
    model = _load_model(config, data, spec.model_kind, spec.seed)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)

    arrays: dict[str, np.ndarray] = {}
    for split, (X, y, lengths) in exp13._split_arrays(data).items():
        state_parts: dict[str, list[np.ndarray]] = {}
        for _, _, Xb, _, _ in exp13._iter_batches(
            X,
            y,
            lengths,
            config.batch_size,
        ):
            with torch.no_grad():
                replay = exp13._replay(model, Xb.to(device))
                views = exp13._state_views(replay)
            for layer, layer_states in views.items():
                for state in ANALYSIS_STATES:
                    key = f"{split}__{layer}__{state}"
                    state_parts.setdefault(key, []).append(
                        layer_states[state]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
        for key, parts in state_parts.items():
            arrays[key] = np.concatenate(parts, axis=0)
        arrays[f"{split}__y"] = np.asarray(y, dtype=np.int64)
        arrays[f"{split}__lengths"] = np.asarray(
            lengths,
            dtype=np.int64,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"status": "completed", "key": spec.key}


def _resample_trajectory(
    values: np.ndarray,
    valid: int,
    steps: int = ALIGNMENT_STEPS,
) -> np.ndarray:
    valid = max(1, min(int(valid), values.shape[0]))
    source = np.asarray(values[:valid], dtype=np.float32)
    if valid == 1:
        return np.repeat(source, steps, axis=0)
    x_old = np.linspace(0.0, 1.0, valid)
    x_new = np.linspace(0.0, 1.0, steps)
    out = np.empty((steps, source.shape[1]), dtype=np.float32)
    for dim in range(source.shape[1]):
        out[:, dim] = np.interp(x_new, x_old, source[:, dim])
    return out


def _resample_split(
    values: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            _resample_trajectory(values[index], int(lengths[index]))
            for index in range(len(values))
        ],
        axis=0,
    )


def _cosine_cost_batch(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=2, keepdims=True)
    b_norm = np.linalg.norm(b, axis=2, keepdims=True)
    a_unit = np.divide(
        a,
        np.maximum(a_norm, 1e-12),
        out=np.zeros_like(a),
    )
    b_unit = np.divide(
        b,
        np.maximum(b_norm, 1e-12),
        out=np.zeros_like(b),
    )
    similarity = np.einsum("bid,bjd->bij", a_unit, b_unit)
    cost = 1.0 - np.clip(similarity, -1.0, 1.0)
    both_zero = (
        (a_norm[:, :, 0] <= 1e-12)[:, :, None]
        & (b_norm[:, :, 0] <= 1e-12)[:, None, :]
    )
    cost[both_zero] = 0.0
    return cost.astype(np.float32, copy=False)


def _batched_bounded_dtw(
    a: np.ndarray,
    b: np.ndarray,
    band_fraction: float = DTW_BAND_FRACTION,
) -> np.ndarray:
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("DTW inputs must match [batch,time,dim]")
    batch, steps, _ = a.shape
    if batch == 0:
        return np.empty(0, dtype=np.float32)
    band = max(1, int(math.ceil(float(band_fraction) * steps)))
    cost = _cosine_cost_batch(a, b)
    inf = np.float32(np.inf)
    dp = np.full((batch, steps, steps), inf, dtype=np.float32)
    plen = np.zeros((batch, steps, steps), dtype=np.int16)
    dp[:, 0, 0] = cost[:, 0, 0]
    plen[:, 0, 0] = 1

    for i in range(steps):
        lo = max(0, i - band)
        hi = min(steps, i + band + 1)
        for j in range(lo, hi):
            if i == 0 and j == 0:
                continue
            candidates = np.full((3, batch), inf, dtype=np.float32)
            lengths = np.zeros((3, batch), dtype=np.int16)
            if i > 0:
                candidates[0] = dp[:, i - 1, j]
                lengths[0] = plen[:, i - 1, j]
            if j > 0:
                candidates[1] = dp[:, i, j - 1]
                lengths[1] = plen[:, i, j - 1]
            if i > 0 and j > 0:
                candidates[2] = dp[:, i - 1, j - 1]
                lengths[2] = plen[:, i - 1, j - 1]
            choice = np.argmin(candidates, axis=0)
            cols = np.arange(batch)
            previous = candidates[choice, cols]
            previous_length = lengths[choice, cols]
            valid = np.isfinite(previous)
            dp[valid, i, j] = (
                previous[valid] + cost[valid, i, j]
            )
            plen[valid, i, j] = previous_length[valid] + 1

    total = dp[:, -1, -1]
    count = plen[:, -1, -1]
    if bool((count <= 0).any()):
        raise RuntimeError("DTW path did not reach endpoint")
    return total / count.astype(np.float32)


def _fixed_alignment_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Fixed-alignment inputs must match")
    a_norm = np.linalg.norm(a, axis=2)
    b_norm = np.linalg.norm(b, axis=2)
    dot = np.sum(a * b, axis=2)
    denom = np.maximum(a_norm * b_norm, 1e-12)
    similarity = np.clip(dot / denom, -1.0, 1.0)
    both_zero = (a_norm <= 1e-12) & (b_norm <= 1e-12)
    distance = 1.0 - similarity
    distance[both_zero] = 0.0
    return distance.mean(axis=1)


def _pair_distances(
    trajectories: np.ndarray,
    pair_frame: pd.DataFrame,
    alignment: str,
    batch_size: int = 256,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, len(pair_frame), batch_size):
        part = pair_frame.iloc[start : start + batch_size]
        left = trajectories[
            part["i"].to_numpy(dtype=np.int64)
        ]
        right = trajectories[
            part["j"].to_numpy(dtype=np.int64)
        ]
        if alignment == "fixed":
            values = _fixed_alignment_distance(left, right)
        elif alignment == "dtw":
            values = _batched_bounded_dtw(left, right)
        else:
            raise ValueError(alignment)
        out.append(np.asarray(values, dtype=np.float32))
    return (
        np.concatenate(out)
        if out
        else np.empty(0, dtype=np.float32)
    )


def _medoid_indexes(
    trajectories: np.ndarray,
    frame: pd.DataFrame,
    alignment: str,
) -> list[int]:
    medoids: list[int] = []
    for _, group in frame.groupby(["user", "label"], sort=True):
        indexes = group["sample_index"].to_numpy(dtype=np.int64)
        if len(indexes) == 1:
            medoids.append(int(indexes[0]))
            continue
        candidates = trajectories[indexes]
        scores = np.zeros(len(indexes), dtype=np.float64)
        for i in range(len(indexes)):
            left = np.repeat(
                candidates[i : i + 1],
                len(indexes),
                axis=0,
            )
            distances = (
                _fixed_alignment_distance(left, candidates)
                if alignment == "fixed"
                else _batched_bounded_dtw(left, candidates)
            )
            scores[i] = float(distances.sum())
        medoids.append(int(indexes[int(np.argmin(scores))]))
    return medoids


def _retrieval_rows(
    trajectories: dict[str, np.ndarray],
    sample_manifest: pd.DataFrame,
    spec: MetricSpec,
    layer: str,
    alignment: str,
) -> list[dict[str, Any]]:
    train_meta = sample_manifest[
        sample_manifest["split"] == "train"
    ].reset_index(drop=True)
    test_meta = sample_manifest[
        sample_manifest["split"] == "test"
    ].reset_index(drop=True)
    train_values = trajectories["train"]
    test_values = trajectories["test"]
    medoid_indexes = _medoid_indexes(
        train_values,
        train_meta,
        alignment,
    )
    gallery = train_values[medoid_indexes]
    gallery_meta = train_meta.iloc[
        medoid_indexes
    ].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for query_index in range(len(test_values)):
        query = np.repeat(
            test_values[query_index : query_index + 1],
            len(gallery),
            axis=0,
        )
        distances = (
            _fixed_alignment_distance(query, gallery)
            if alignment == "fixed"
            else _batched_bounded_dtw(query, gallery)
        )
        best = int(np.argmin(distances))
        query_row = test_meta.iloc[query_index]
        gallery_row = gallery_meta.iloc[best]
        rows.append(
            {
                "model_kind": spec.model_kind,
                "seed": spec.seed,
                "state": spec.state,
                "layer": layer,
                "alignment": alignment,
                "query_index": query_index,
                "query_user": str(query_row.user),
                "query_label": str(query_row.label),
                "pred_label": str(gallery_row.label),
                "nearest_user": str(gallery_row.user),
                "distance": float(distances[best]),
                "correct": (
                    str(query_row.label) == str(gallery_row.label)
                ),
            }
        )
    return rows


def _f_metric_paths(
    config: Config,
    spec: MetricSpec,
) -> tuple[Path, Path]:
    root = _subdir(config, "F_cross_user_geometry")
    return (
        root / "pair_runs" / f"{spec.key}.csv",
        root / "retrieval_runs" / f"{spec.key}.csv",
    )


def run_f_metric(
    spec: MetricSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    pair_path, retrieval_path = _f_metric_paths(config, spec)
    retrieval_required = spec.state == "spike50"
    if (
        pair_path.exists()
        and (retrieval_path.exists() or not retrieval_required)
        and not force
    ):
        return {"status": "exists", "key": spec.key}

    cache = np.load(
        _cache_path(
            config,
            CacheSpec(spec.model_kind, spec.seed),
        ),
        allow_pickle=False,
    )
    pair_manifest = pd.read_csv(
        config.results_dir / "pair_manifest.csv"
    )
    sample_manifest = pd.read_csv(
        config.results_dir / "sample_manifest.csv"
    )
    pair_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []

    for layer in ("L1", "L2", "L3"):
        resampled: dict[str, np.ndarray] = {}
        for split in ("train", "test"):
            values = np.asarray(
                cache[f"{split}__{layer}__{spec.state}"],
                dtype=np.float32,
            )
            lengths = np.asarray(
                cache[f"{split}__lengths"],
                dtype=np.int64,
            )
            resampled[split] = _resample_split(values, lengths)

        for domain in ("train", "test"):
            domain_pairs = pair_manifest[
                pair_manifest["domain"] == domain
            ].reset_index(drop=True)
            for alignment in ("fixed", "dtw"):
                distances = _pair_distances(
                    resampled[domain],
                    domain_pairs,
                    alignment,
                )
                for row, distance in zip(
                    domain_pairs.to_dict("records"),
                    distances,
                    strict=True,
                ):
                    pair_rows.append(
                        {
                            **row,
                            "model_kind": spec.model_kind,
                            "seed": spec.seed,
                            "state": spec.state,
                            "layer": layer,
                            "alignment": alignment,
                            "distance": float(distance),
                        }
                    )

        if retrieval_required:
            for alignment in ("fixed", "dtw"):
                retrieval_rows.extend(
                    _retrieval_rows(
                        resampled,
                        sample_manifest,
                        spec,
                        layer,
                        alignment,
                    )
                )

    _save_csv(pair_path, pair_rows)
    if retrieval_required:
        _save_csv(retrieval_path, retrieval_rows)
    return {
        "status": "completed",
        "key": spec.key,
        "pair_rows": len(pair_rows),
        "retrieval_rows": len(retrieval_rows),
    }


def _phase_vectors(
    cache: Any,
    split: str,
    layer: str,
    state: str,
    phase: float,
) -> np.ndarray:
    values = np.asarray(cache[f"{split}__{layer}__{state}"])
    lengths = np.asarray(
        cache[f"{split}__lengths"],
        dtype=np.int64,
    )
    steps = exp13._phase_steps(lengths, phase)
    return exp13._gather_steps(
        torch.from_numpy(values),
        steps,
    )


def _fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
    *,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler(with_mean=True)
    train_scaled = scaler.fit_transform(x_train)
    classifier = LogisticRegression(
        C=float(C),
        fit_intercept=False,
        class_weight="balanced",
        max_iter=PROBE_MAX_ITER,
        random_state=int(seed),
        solver="lbfgs",
    )
    classifier.fit(train_scaled, y_train)
    return scaler, classifier


def _residualize_by_class(
    x_train: np.ndarray,
    class_train: np.ndarray,
    x_other: np.ndarray,
    class_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(x_train, dtype=np.float64)
    other = np.asarray(x_other, dtype=np.float64)
    train_out = train.copy()
    other_out = other.copy()
    for label in np.unique(class_train):
        mask_train = class_train == label
        centroid = train[mask_train].mean(axis=0)
        train_out[mask_train] -= centroid
        mask_other = class_other == label
        other_out[mask_other] -= centroid
    return train_out, other_out


def _choose_user_c(
    x: np.ndarray,
    users: np.ndarray,
    characters: np.ndarray,
    seed: int,
) -> float:
    splitter = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=seed,
    )
    best_C = float(PROBE_C_GRID[0])
    best_ba = -math.inf
    for C in PROBE_C_GRID:
        scores: list[float] = []
        for train_index, val_index in splitter.split(x, users):
            xtr, xva = _residualize_by_class(
                x[train_index],
                characters[train_index],
                x[val_index],
                characters[val_index],
            )
            scaler, classifier = _fit_logistic(
                xtr,
                users[train_index],
                C,
                seed=seed,
            )
            pred = classifier.predict(scaler.transform(xva))
            scores.append(
                balanced_accuracy_score(
                    users[val_index],
                    pred,
                )
            )
        score = float(np.mean(scores))
        if score > best_ba:
            best_ba = score
            best_C = float(C)
    return best_C


def _g_path(config: Config, spec: MetricSpec) -> Path:
    return (
        _subdir(config, "G_user_leakage")
        / "probe_runs"
        / f"{spec.key}.csv"
    )


def run_g(
    spec: MetricSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    path = _g_path(config, spec)
    if path.exists() and not force:
        return {"status": "exists", "key": spec.key}

    cache = np.load(
        _cache_path(
            config,
            CacheSpec(spec.model_kind, spec.seed),
        ),
        allow_pickle=False,
    )
    manifest = pd.read_csv(
        config.results_dir / "sample_manifest.csv"
    )
    all_meta = pd.concat(
        [
            manifest[
                manifest["split"] == split
            ].reset_index(drop=True)
            for split in ("train", "val", "test")
        ],
        ignore_index=True,
    )
    user_names = sorted(all_meta["user"].unique().tolist())
    user_to_idx = {
        user: index for index, user in enumerate(user_names)
    }
    users = all_meta["user"].map(
        user_to_idx
    ).to_numpy(dtype=np.int64)
    characters = all_meta["y"].to_numpy(dtype=np.int64)
    outer = StratifiedKFold(
        n_splits=USER_FOLDS,
        shuffle=True,
        random_state=exp3.dseed(
            spec.seed,
            EXPERIMENT_ID,
            "G_outer",
            spec.model_kind,
            spec.state,
        ),
    )
    outer_splits = list(
        outer.split(np.zeros(len(users)), users)
    )
    rows: list[dict[str, Any]] = []

    for layer in ("L1", "L2", "L3"):
        for phase in PHASES:
            x = np.concatenate(
                [
                    _phase_vectors(
                        cache,
                        split,
                        layer,
                        spec.state,
                        phase,
                    )
                    for split in ("train", "val", "test")
                ],
                axis=0,
            ).astype(np.float64)

            for fold, (train_index, test_index) in enumerate(
                outer_splits
            ):
                probe_seed = exp3.dseed(
                    spec.seed,
                    EXPERIMENT_ID,
                    "G",
                    spec.model_kind,
                    spec.state,
                    layer,
                    phase,
                    fold,
                )
                selected_C = _choose_user_c(
                    x[train_index],
                    users[train_index],
                    characters[train_index],
                    probe_seed,
                )
                xtr, xte = _residualize_by_class(
                    x[train_index],
                    characters[train_index],
                    x[test_index],
                    characters[test_index],
                )
                scaler, classifier = _fit_logistic(
                    xtr,
                    users[train_index],
                    selected_C,
                    seed=probe_seed,
                )
                pred = classifier.predict(
                    scaler.transform(xte)
                )
                real_ba = balanced_accuracy_score(
                    users[test_index],
                    pred,
                )

                rng = np.random.default_rng(probe_seed)
                perm_scores: list[float] = []
                train_char = characters[train_index]
                for _ in range(config.user_permutations):
                    perm_users = users[train_index].copy()
                    for character in np.unique(train_char):
                        mask = np.flatnonzero(
                            train_char == character
                        )
                        perm_users[mask] = rng.permutation(
                            perm_users[mask]
                        )
                    perm_scaler, perm_classifier = _fit_logistic(
                        xtr,
                        perm_users,
                        selected_C,
                        seed=probe_seed,
                    )
                    perm_pred = perm_classifier.predict(
                        perm_scaler.transform(xte)
                    )
                    perm_scores.append(
                        balanced_accuracy_score(
                            users[test_index],
                            perm_pred,
                        )
                    )

                perm_mean = float(np.mean(perm_scores))
                rows.append(
                    {
                        "model_kind": spec.model_kind,
                        "seed": spec.seed,
                        "state": spec.state,
                        "layer": layer,
                        "phase": phase,
                        "fold": fold,
                        "selected_C": selected_C,
                        "real_user_ba": float(real_ba),
                        "perm_user_ba_mean": perm_mean,
                        "perm_user_ba_std": float(
                            np.std(perm_scores)
                        ),
                        "excess_user_ba": float(
                            real_ba - perm_mean
                        ),
                        "n_train": len(train_index),
                        "n_test": len(test_index),
                    }
                )

    _save_csv(path, rows)
    return {
        "status": "completed",
        "key": spec.key,
        "rows": len(rows),
    }


def _h_feature_path(
    config: Config,
    spec: HFeatureSpec,
) -> Path:
    return (
        _subdir(config, "H_history_generalization")
        / "c1_features"
        / f"{spec.key}.npz"
    )


def run_h_feature(
    spec: HFeatureSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    path = _h_feature_path(config, spec)
    if path.exists() and not force:
        return {"status": "exists", "key": spec.key}

    data = exp3.prepare_data(config.repo_root)
    model = _load_model(config, data, "c1", spec.seed)
    device = torch.device(config.device)
    arrays: dict[str, np.ndarray] = {}

    for split, (X, y, lengths) in exp13._split_arrays(data).items():
        arrays[f"{split}__y"] = np.asarray(y, dtype=np.int64)
        arrays[f"{split}__lengths"] = np.asarray(
            lengths,
            dtype=np.int64,
        )
        for phase in H_PHASES:
            steps = exp13._phase_steps(lengths, phase)
            history_steps = max(
                1,
                int(round(spec.history_ms * float(data.fs) / 1000.0)),
            )
            features = exp13._suffix_features(
                model,
                X,
                np.arange(len(X), dtype=np.int64),
                steps,
                history_steps,
                _exp13_config(config),
            )
            phase_key = f"p{int(round(phase * 100)):03d}"
            for layer, states in features.items():
                for state in ANALYSIS_STATES:
                    arrays[
                        f"{split}__{layer}__{state}__{phase_key}"
                    ] = np.asarray(
                        states[state],
                        dtype=np.float32,
                    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"status": "completed", "key": spec.key}


def _c2_history_feature_path(
    config: Config,
    seed: int,
    history_ms: int,
) -> Path:
    spec = exp13.A2Spec(seed, "c2", history_ms)
    return exp13._a2_paths(
        _exp13_config(config),
        spec,
    )[1]


def _history_feature_path(
    config: Config,
    model_kind: str,
    seed: int,
    history_ms: int,
) -> Path:
    if model_kind == "c1":
        return _h_feature_path(
            config,
            HFeatureSpec(seed, history_ms),
        )
    if model_kind == "c2":
        return _c2_history_feature_path(
            config,
            seed,
            history_ms,
        )
    raise ValueError(model_kind)


def _full_phase_matrix(
    cache: Any,
    split: str,
    layer: str,
    state: str,
    phase: float,
) -> np.ndarray:
    return np.asarray(
        _phase_vectors(cache, split, layer, state, phase),
        dtype=np.float64,
    )


def _truncated_phase_matrix(
    path: Path,
    split: str,
    layer: str,
    state: str,
    phase: float,
) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    phase_key = f"p{int(round(phase * 100)):03d}"
    key = f"{split}__{layer}__{state}__{phase_key}"
    if key not in payload:
        raise KeyError(f"Missing {key} in {path}")
    return np.asarray(payload[key], dtype=np.float64)


def _choose_character_c_idcv(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> float:
    best_C = float(PROBE_C_GRID[0])
    best_ba = -math.inf
    for C in PROBE_C_GRID:
        scores: list[float] = []
        for fold in range(H_ID_FOLDS):
            train_mask = folds != fold
            val_mask = folds == fold
            if not bool(val_mask.any()):
                continue
            scaler, classifier = _fit_logistic(
                x[train_mask],
                y[train_mask],
                C,
                seed=seed,
            )
            pred = classifier.predict(
                scaler.transform(x[val_mask])
            )
            scores.append(
                balanced_accuracy_score(
                    y[val_mask],
                    pred,
                )
            )
        if not scores:
            raise RuntimeError("No ID-CV folds available")
        score = float(np.mean(scores))
        if score > best_ba:
            best_ba = score
            best_C = float(C)
    return best_C


def _evaluate_id_ood(
    train_x: np.ndarray,
    train_y: np.ndarray,
    folds: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    C: float,
    seed: int,
) -> tuple[float, float]:
    id_true: list[np.ndarray] = []
    id_pred: list[np.ndarray] = []
    for fold in range(H_ID_FOLDS):
        train_mask = folds != fold
        val_mask = folds == fold
        if not bool(val_mask.any()):
            continue
        scaler, classifier = _fit_logistic(
            train_x[train_mask],
            train_y[train_mask],
            C,
            seed=seed,
        )
        pred = classifier.predict(
            scaler.transform(train_x[val_mask])
        )
        id_true.append(train_y[val_mask])
        id_pred.append(pred)
    id_ba = balanced_accuracy_score(
        np.concatenate(id_true),
        np.concatenate(id_pred),
    )

    scaler, classifier = _fit_logistic(
        train_x,
        train_y,
        C,
        seed=seed,
    )
    test_pred = classifier.predict(
        scaler.transform(test_x)
    )
    ood_ba = balanced_accuracy_score(test_y, test_pred)
    return float(id_ba), float(ood_ba)


def _h_path(config: Config, spec: MetricSpec) -> Path:
    return (
        _subdir(config, "H_history_generalization")
        / "probe_runs"
        / f"{spec.key}.csv"
    )


def run_h(
    spec: MetricSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    path = _h_path(config, spec)
    if path.exists() and not force:
        return {"status": "exists", "key": spec.key}

    cache = np.load(
        _cache_path(
            config,
            CacheSpec(spec.model_kind, spec.seed),
        ),
        allow_pickle=False,
    )
    sample_manifest = pd.read_csv(
        config.results_dir / "sample_manifest.csv"
    )
    h_folds = pd.read_csv(
        config.results_dir / "h_fold_manifest.csv"
    )
    train_meta = sample_manifest[
        sample_manifest["split"] == "train"
    ].reset_index(drop=True)
    test_meta = sample_manifest[
        sample_manifest["split"] == "test"
    ].reset_index(drop=True)
    if not np.array_equal(
        train_meta["sample_id"].to_numpy(),
        h_folds["sample_id"].to_numpy(),
    ):
        raise ValueError("H fold manifest order mismatch")

    fold_values = h_folds["h_fold"].to_numpy(dtype=np.int64)
    train_y = train_meta["y"].to_numpy(dtype=np.int64)
    test_y = test_meta["y"].to_numpy(dtype=np.int64)
    train_lengths = np.asarray(
        cache["train__lengths"],
        dtype=np.int64,
    )
    test_lengths = np.asarray(
        cache["test__lengths"],
        dtype=np.int64,
    )

    rows: list[dict[str, Any]] = []
    for layer in ("L1", "L2", "L3"):
        for phase in H_PHASES:
            train_steps = exp13._phase_steps(
                train_lengths,
                phase,
            )
            test_steps = exp13._phase_steps(
                test_lengths,
                phase,
            )
            train_eligible = (
                train_steps >= H_COMMON_HISTORY_STEPS
            )
            test_eligible = (
                test_steps >= H_COMMON_HISTORY_STEPS
            )
            if int(train_eligible.sum()) < len(
                np.unique(train_y)
            ):
                continue
            if int(test_eligible.sum()) < len(
                np.unique(test_y)
            ):
                continue

            full_train = _full_phase_matrix(
                cache,
                "train",
                layer,
                spec.state,
                phase,
            )[train_eligible]
            full_test = _full_phase_matrix(
                cache,
                "test",
                layer,
                spec.state,
                phase,
            )[test_eligible]
            y_train = train_y[train_eligible]
            y_test = test_y[test_eligible]
            folds = fold_values[train_eligible]
            probe_seed = exp3.dseed(
                spec.seed,
                EXPERIMENT_ID,
                "H",
                spec.model_kind,
                spec.state,
                layer,
                phase,
            )
            selected_C = _choose_character_c_idcv(
                full_train,
                y_train,
                folds,
                probe_seed,
            )

            conditions: list[
                tuple[
                    str,
                    int | None,
                    np.ndarray,
                    np.ndarray,
                ]
            ] = [
                ("full", None, full_train, full_test)
            ]
            for history_ms in HISTORY_MS:
                source = _history_feature_path(
                    config,
                    spec.model_kind,
                    spec.seed,
                    history_ms,
                )
                if not source.exists():
                    raise FileNotFoundError(source)
                train_x = _truncated_phase_matrix(
                    source,
                    "train",
                    layer,
                    spec.state,
                    phase,
                )[train_eligible]
                test_x = _truncated_phase_matrix(
                    source,
                    "test",
                    layer,
                    spec.state,
                    phase,
                )[test_eligible]
                conditions.append(
                    (
                        f"h{history_ms}",
                        history_ms,
                        train_x,
                        test_x,
                    )
                )

            condition_rows: list[dict[str, Any]] = []
            for (
                condition,
                history_ms,
                train_x,
                test_x,
            ) in conditions:
                id_ba, ood_ba = _evaluate_id_ood(
                    train_x,
                    y_train,
                    folds,
                    test_x,
                    y_test,
                    selected_C,
                    probe_seed,
                )
                condition_rows.append(
                    {
                        "model_kind": spec.model_kind,
                        "seed": spec.seed,
                        "state": spec.state,
                        "layer": layer,
                        "phase": phase,
                        "primary_phase": phase >= 0.75,
                        "condition": condition,
                        "history_ms": history_ms,
                        "selected_C": selected_C,
                        "id_ba": id_ba,
                        "ood_ba": ood_ba,
                        "gap_pp": 100.0 * (
                            id_ba - ood_ba
                        ),
                        "eligible_train_n": int(
                            train_eligible.sum()
                        ),
                        "eligible_test_n": int(
                            test_eligible.sum()
                        ),
                    }
                )

            reference = next(
                row
                for row in condition_rows
                if row["history_ms"] == 50
            )
            for row in condition_rows:
                row["id_gain_pp_vs_50"] = 100.0 * (
                    float(row["id_ba"])
                    - float(reference["id_ba"])
                )
                row["ood_gain_pp_vs_50"] = 100.0 * (
                    float(row["ood_ba"])
                    - float(reference["ood_ba"])
                )
                row["excess_seen_gain_pp"] = (
                    float(row["id_gain_pp_vs_50"])
                    - float(row["ood_gain_pp_vs_50"])
                )
                rows.append(row)

    _save_csv(path, rows)
    return {
        "status": "completed",
        "key": spec.key,
        "rows": len(rows),
    }


def _concat_required(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError("No files to aggregate")
    return pd.concat(frames, ignore_index=True)


def _summarize_f_pairs(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "model_kind",
        "seed",
        "state",
        "layer",
        "domain",
        "alignment",
        "relation",
    ]
    per_seed = (
        frame.groupby(group_cols, as_index=False)
        .agg(
            distance_mean=("distance", "mean"),
            distance_median=("distance", "median"),
            pair_count=("distance", "size"),
        )
    )
    pivot = per_seed.pivot_table(
        index=[
            "model_kind",
            "seed",
            "state",
            "layer",
            "domain",
            "alignment",
        ],
        columns="relation",
        values="distance_mean",
    ).reset_index()
    if {"SC_CU", "DC_CU"}.issubset(pivot.columns):
        pivot["cross_user_ratio"] = (
            pivot["DC_CU"]
            / np.maximum(pivot["SC_CU"], 1e-12)
        )
    return pivot


def _summarize_retrieval(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, part in frame.groupby(
        [
            "model_kind",
            "seed",
            "state",
            "layer",
            "alignment",
        ],
        sort=True,
    ):
        labels = sorted(
            set(part["query_label"]) | set(part["pred_label"])
        )
        label_to_idx = {
            label: index
            for index, label in enumerate(labels)
        }
        true = part["query_label"].map(
            label_to_idx
        ).to_numpy(dtype=np.int64)
        pred = part["pred_label"].map(
            label_to_idx
        ).to_numpy(dtype=np.int64)
        rows.append(
            {
                "model_kind": keys[0],
                "seed": keys[1],
                "state": keys[2],
                "layer": keys[3],
                "alignment": keys[4],
                "retrieval_ba": (
                    balanced_accuracy_score(true, pred)
                ),
                "query_count": len(part),
            }
        )
    return pd.DataFrame(rows)


def finalize(config: Config) -> dict[str, Any]:
    source_manifest = (
        config.results_dir / "source_manifest.json"
    )
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)

    f_pairs = _concat_required(
        _f_metric_paths(config, spec)[0]
        for spec in metric_specs()
    )
    f_pair_seed = _summarize_f_pairs(f_pairs)
    _save_csv(
        _subdir(config, "F_cross_user_geometry")
        / "F_pair_all_runs.csv",
        f_pairs,
    )
    _save_csv(
        _subdir(config, "F_cross_user_geometry")
        / "F_pair_per_seed_summary.csv",
        f_pair_seed,
    )
    f_pair_summary = (
        f_pair_seed.groupby(
            [
                "model_kind",
                "state",
                "layer",
                "domain",
                "alignment",
            ],
            as_index=False,
        )
        .agg(
            cross_user_ratio_mean=(
                "cross_user_ratio",
                "mean",
            ),
            cross_user_ratio_std=(
                "cross_user_ratio",
                "std",
            ),
            seed_count=("cross_user_ratio", "size"),
        )
    )
    _save_csv(
        _subdir(config, "F_cross_user_geometry")
        / "F_pair_summary.csv",
        f_pair_summary,
    )

    retrieval_paths = [
        _f_metric_paths(config, spec)[1]
        for spec in metric_specs()
        if spec.state == "spike50"
    ]
    f_retrieval = _concat_required(retrieval_paths)
    f_retrieval_seed = _summarize_retrieval(
        f_retrieval
    )
    _save_csv(
        _subdir(config, "F_cross_user_geometry")
        / "F_retrieval_all_runs.csv",
        f_retrieval,
    )
    _save_csv(
        _subdir(config, "F_cross_user_geometry")
        / "F_retrieval_per_seed_summary.csv",
        f_retrieval_seed,
    )
    f_retrieval_summary = (
        f_retrieval_seed.groupby(
            [
                "model_kind",
                "state",
                "layer",
                "alignment",
            ],
            as_index=False,
        )
        .agg(
            retrieval_ba_mean=("retrieval_ba", "mean"),
            retrieval_ba_std=("retrieval_ba", "std"),
            seed_count=("retrieval_ba", "size"),
        )
    )
    _save_csv(
        _subdir(config, "F_cross_user_geometry")
        / "F_retrieval_summary.csv",
        f_retrieval_summary,
    )

    g_all = _concat_required(
        _g_path(config, spec)
        for spec in metric_specs()
    )
    _save_csv(
        _subdir(config, "G_user_leakage")
        / "G_user_leakage_all_runs.csv",
        g_all,
    )
    g_seed = (
        g_all.groupby(
            [
                "model_kind",
                "seed",
                "state",
                "layer",
                "phase",
            ],
            as_index=False,
        )
        .agg(
            real_user_ba=("real_user_ba", "mean"),
            perm_user_ba_mean=(
                "perm_user_ba_mean",
                "mean",
            ),
            excess_user_ba=("excess_user_ba", "mean"),
        )
    )
    g_summary = (
        g_seed.groupby(
            ["model_kind", "state", "layer", "phase"],
            as_index=False,
        )
        .agg(
            real_user_ba_mean=("real_user_ba", "mean"),
            real_user_ba_std=("real_user_ba", "std"),
            excess_user_ba_mean=(
                "excess_user_ba",
                "mean",
            ),
            excess_user_ba_std=(
                "excess_user_ba",
                "std",
            ),
            seed_count=("seed", "size"),
        )
    )
    _save_csv(
        _subdir(config, "G_user_leakage")
        / "G_user_leakage_summary.csv",
        g_summary,
    )

    h_all = _concat_required(
        _h_path(config, spec)
        for spec in metric_specs()
    )
    _save_csv(
        _subdir(config, "H_history_generalization")
        / "H_history_all_runs.csv",
        h_all,
    )
    h_summary = (
        h_all.groupby(
            [
                "model_kind",
                "state",
                "layer",
                "phase",
                "primary_phase",
                "condition",
                "history_ms",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            id_ba_mean=("id_ba", "mean"),
            id_ba_std=("id_ba", "std"),
            ood_ba_mean=("ood_ba", "mean"),
            ood_ba_std=("ood_ba", "std"),
            gap_pp_mean=("gap_pp", "mean"),
            id_gain_pp_vs_50_mean=(
                "id_gain_pp_vs_50",
                "mean",
            ),
            ood_gain_pp_vs_50_mean=(
                "ood_gain_pp_vs_50",
                "mean",
            ),
            excess_seen_gain_pp_mean=(
                "excess_seen_gain_pp",
                "mean",
            ),
            excess_seen_gain_pp_std=(
                "excess_seen_gain_pp",
                "std",
            ),
            seed_count=("seed", "size"),
        )
    )
    _save_csv(
        _subdir(config, "H_history_generalization")
        / "H_history_summary.csv",
        h_summary,
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "primary_depth_control": (
            "C1_frozen_A2_backbone_train_L3"
        ),
        "secondary_replication": (
            "C2_end_to_end_three_layer"
        ),
        "F": {
            "question": (
                "How does cross-user task geometry vary "
                "with depth?"
            ),
            "primary_metrics": [
                "DC_CU / SC_CU distance ratio",
                (
                    "cross-user nearest-medoid "
                    "balanced accuracy"
                ),
            ],
            "alignments": [
                "fixed_normalized_time",
                "bounded_DTW",
            ],
        },
        "G": {
            "question": (
                "How much user identity remains after "
                "character residualization?"
            ),
            "primary_metric": (
                "user-ID BA minus within-character "
                "permutation BA"
            ),
        },
        "H": {
            "question": (
                "Does added history benefit seen users "
                "more than held-out users?"
            ),
            "primary_metric": (
                "ID gain minus OOD gain relative to 50 ms"
            ),
            "primary_phases": [0.75, 1.0],
        },
        "artifacts": {
            "F_pair_summary": (
                "F_cross_user_geometry/"
                "F_pair_summary.csv"
            ),
            "F_retrieval_summary": (
                "F_cross_user_geometry/"
                "F_retrieval_summary.csv"
            ),
            "G_summary": (
                "G_user_leakage/"
                "G_user_leakage_summary.csv"
            ),
            "H_summary": (
                "H_history_generalization/"
                "H_history_summary.csv"
            ),
        },
    }
    _save_json(
        config.results_dir / "manifest.json",
        manifest,
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exp13.1 abstraction and cross-user "
            "generalization"
        )
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--user-permutations",
        type=int,
        default=USER_PERMUTATIONS,
    )
    parser.add_argument("--force", action="store_true")
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    subparsers.add_parser("prepare-source")
    for name in (
        "f-cache",
        "f-metric",
        "g",
        "h-feature",
        "h",
    ):
        child = subparsers.add_parser(name)
        child.add_argument(
            "--array-task-id",
            type=int,
            required=True,
        )
    subparsers.add_parser("finalize")
    subparsers.add_parser("list-runs")
    return parser.parse_args()


def _resolve_spec(items: list[Any], index: int) -> Any:
    if index < 0 or index >= len(items):
        raise IndexError(index)
    return items[index]


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
        user_permutations=args.user_permutations,
    )

    if args.command == "prepare-source":
        prepare_source(config)
        return
    if args.command == "f-cache":
        run_f_cache(
            _resolve_spec(
                cache_specs(),
                args.array_task_id,
            ),
            config,
            force=args.force,
        )
        return
    if args.command == "f-metric":
        run_f_metric(
            _resolve_spec(
                metric_specs(),
                args.array_task_id,
            ),
            config,
            force=args.force,
        )
        return
    if args.command == "g":
        run_g(
            _resolve_spec(
                metric_specs(),
                args.array_task_id,
            ),
            config,
            force=args.force,
        )
        return
    if args.command == "h-feature":
        run_h_feature(
            _resolve_spec(
                h_feature_specs(),
                args.array_task_id,
            ),
            config,
            force=args.force,
        )
        return
    if args.command == "h":
        run_h(
            _resolve_spec(
                metric_specs(),
                args.array_task_id,
            ),
            config,
            force=args.force,
        )
        return
    if args.command == "finalize":
        finalize(config)
        return
    if args.command == "list-runs":
        payload = {
            "f_cache": [
                asdict(spec) for spec in cache_specs()
            ],
            "f_metric": [
                asdict(spec) for spec in metric_specs()
            ],
            "g": [
                asdict(spec) for spec in metric_specs()
            ],
            "h_feature": [
                asdict(spec) for spec in h_feature_specs()
            ],
            "h": [
                asdict(spec) for spec in metric_specs()
            ],
        }
        print(json.dumps(payload, indent=2))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
