from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from snn.accel_reconstruction_eval.datasets import load_acceleration_data
from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_3_0_2_hidden_multitau_architectures as exp302
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_8_0_local_backbone_tau_sweep as exp80


EXPERIMENT_ID = "experiment_9_0_within_user_generalization"
PROTOCOL_VERSION = "rotating_grouped_cv_v2"
METHOD_RAW250 = "raw250_linear"
METHOD_A2 = "a2_234x234"
METHODS = (METHOD_RAW250, METHOD_A2)
CV_WITHIN = "within_user"
CV_CROSS = "cross_user"
CV_MODES = (CV_WITHIN, CV_CROSS)
N_FOLDS = 5
ROTATIONS = tuple(range(N_FOLDS))
MODEL_SEEDS = (11, 23, 37, 53, 71)
FOLD_ASSIGNMENT_SEED = 11
EXPECTED_RUNS = len(CV_MODES) * len(METHODS) * len(ROTATIONS)
TARGET_FRACTIONS = {"train": 0.60, "val": 0.20, "test": 0.20}
SPLITS = ("train", "val", "test")
ARCHITECTURE = "234x234"
ARCHITECTURE_SHIFTS = ((2, 3, 4), (2, 3, 4))
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
BATCH_SIZE = exp72.BATCH_SIZE


@dataclass(frozen=True)
class RunSpec:
    cv_mode: str
    method: str
    rotation: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.cv_mode}__{self.method}__rotation{self.rotation}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    threads: int = 1
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS


@dataclass
class PreparedSplit:
    cv_mode: str
    rotation: int
    data: exp3.Data
    frames: dict[str, pd.DataFrame]
    full_manifest: pd.DataFrame
    coverage_summary: pd.DataFrame
    coverage_metrics: dict[str, float]



def find_repo_root(start: Path | None = None) -> Path:
    return exp80.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for cv_mode in CV_MODES:
        for method in METHODS:
            for rotation in ROTATIONS:
                specs.append(RunSpec(cv_mode, method, rotation, MODEL_SEEDS[rotation]))
    return specs


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _stable_seed(seed: int, *parts: object) -> int:
    text = "|".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def _dataset_roots(repo_root: Path) -> list[Path]:
    return [
        repo_root / "outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded",
        repo_root / "outputs/action1_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded",
    ]


def _load_manifest(repo_root: Path):
    loaded = load_acceleration_data(
        _dataset_roots(repo_root),
        repository_root=repo_root,
        require_reconstruction=False,
    )
    fs_values = {float(meta.sampling_rate_hz) for meta in loaded.producer_metadatas}
    if fs_values != {exp3.EXPECTED_FS}:
        raise ValueError(f"Exp9.0 requires exactly {exp3.EXPECTED_FS:g} Hz, got {fs_values}")
    for root, metadata in zip(loaded.padded_roots, loaded.producer_metadatas, strict=True):
        raw = metadata.raw
        if raw.get("event_representation") != "unsigned":
            raise ValueError(f"{root}: expected unsigned events")
        if raw.get("event_feature_schema") != "custom_wavelet_polarity_split_abs_events_v1":
            raise ValueError(f"{root}: unexpected event feature schema")
        if raw.get("event_channel_count") != exp3.EVENT_CHANNELS:
            raise ValueError(f"{root}: expected {exp3.EVENT_CHANNELS} event channels")
        if metadata.channel_count != exp3.TOTAL_CHANNELS:
            raise ValueError(f"{root}: expected {exp3.TOTAL_CHANNELS} total channels")

    rows: list[dict[str, Any]] = []
    keep = set(exp3.LABELS)
    for pi, package in enumerate(loaded.packages):
        source_trial = f"{package.user}/action_{package.action}/{package.stem}"
        for si, label_value in enumerate(package.labels.astype(str)):
            label = str(label_value)
            if label not in keep:
                continue
            rows.append(
                {
                    "pi": int(pi),
                    "si": int(si),
                    "user": str(package.user),
                    "label": label,
                    "valid": int(package.valid_lengths[si]),
                    "pad": int(package.padded_spike_imu.shape[1]),
                    "source_trial": source_trial,
                    "sample_id": f"{source_trial}/segment_{si}",
                }
            )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("Exp9.0 found no eligible 12-class samples")
    labels = tuple(sorted(manifest.label.unique().tolist()))
    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    manifest["y"] = manifest.label.map(class_to_idx).astype(int)
    if manifest.sample_id.duplicated().any():
        raise RuntimeError("Exp9.0 sample_id values must be unique")
    return loaded, manifest, labels


def _assign_within_user_folds(manifest: pd.DataFrame) -> pd.DataFrame:
    """Sparse-safe 5-fold segment stratification performed inside each user."""
    pieces: list[pd.DataFrame] = []
    for user, frame in manifest.groupby("user", sort=True):
        frame = frame.copy()
        if len(frame) < N_FOLDS:
            raise RuntimeError(
                f"{user} has only {len(frame)} segments; {N_FOLDS}-fold CV is impossible"
            )

        fold_total = np.zeros(N_FOLDS, dtype=np.int64)
        fold_by_label: dict[str, np.ndarray] = {}
        assigned: dict[int, int] = {}

        label_groups = sorted(
            frame.groupby("label", sort=True),
            key=lambda item: (-len(item[1]), str(item[0])),
        )
        for label, group in label_groups:
            label = str(label)
            label_counts = fold_by_label.setdefault(
                label, np.zeros(N_FOLDS, dtype=np.int64)
            )
            indices = group.index.to_numpy(dtype=np.int64, copy=True)
            rng = np.random.default_rng(
                _stable_seed(FOLD_ASSIGNMENT_SEED, CV_WITHIN, user, label)
            )
            rng.shuffle(indices)
            for row_index in indices:
                sample_id = str(frame.loc[row_index, "sample_id"])
                fold = min(
                    range(N_FOLDS),
                    key=lambda candidate: (
                        int(label_counts[candidate]),
                        int(fold_total[candidate]),
                        _stable_seed(
                            FOLD_ASSIGNMENT_SEED,
                            CV_WITHIN,
                            user,
                            label,
                            sample_id,
                            candidate,
                        ),
                    ),
                )
                assigned[int(row_index)] = int(fold)
                label_counts[fold] += 1
                fold_total[fold] += 1

        frame["cv_fold"] = [assigned[int(index)] for index in frame.index]
        pieces.append(frame)

    out = pd.concat(pieces, ignore_index=True)
    user_fold_counts = out.groupby(["user", "cv_fold"]).size().unstack(fill_value=0)
    expected_columns = set(range(N_FOLDS))
    if set(user_fold_counts.columns) != expected_columns:
        raise RuntimeError("Within-user fold assignment did not create all five folds")
    if (user_fold_counts == 0).any().any():
        raise RuntimeError(
            "Every user must contribute at least one segment to every within-user fold"
        )
    return out



def _assign_cross_user_folds(manifest: pd.DataFrame) -> pd.DataFrame:
    """Balanced 5-fold user grouping with exactly four users/fold for 20 users."""
    user_sizes = (
        manifest.groupby("user")
        .size()
        .rename("n")
        .reset_index()
    )
    if len(user_sizes) < N_FOLDS:
        raise RuntimeError(f"Need at least {N_FOLDS} users for cross-user CV")

    max_users_per_fold = int(math.ceil(len(user_sizes) / N_FOLDS))
    fold_totals = np.zeros(N_FOLDS, dtype=np.int64)
    fold_users: list[list[str]] = [[] for _ in range(N_FOLDS)]
    user_to_fold: dict[str, int] = {}

    ordered = sorted(
        user_sizes.itertuples(index=False),
        key=lambda row: (
            -int(row.n),
            _stable_seed(FOLD_ASSIGNMENT_SEED, CV_CROSS, str(row.user)),
        ),
    )
    for row in ordered:
        user = str(row.user)
        n = int(row.n)
        candidates = [
            fold
            for fold in range(N_FOLDS)
            if len(fold_users[fold]) < max_users_per_fold
        ]
        if not candidates:
            raise RuntimeError("No cross-user fold has remaining capacity")
        fold = min(
            candidates,
            key=lambda candidate: (
                int(fold_totals[candidate]),
                len(fold_users[candidate]),
                _stable_seed(FOLD_ASSIGNMENT_SEED, CV_CROSS, user, candidate),
            ),
        )
        user_to_fold[user] = fold
        fold_users[fold].append(user)
        fold_totals[fold] += n

    out = manifest.copy()
    out["cv_fold"] = out.user.map(user_to_fold).astype(int)
    if out.groupby("user").cv_fold.nunique().max() != 1:
        raise AssertionError("A user appeared in more than one cross-user fold")
    if len(user_sizes) % N_FOLDS == 0:
        counts = out[["user", "cv_fold"]].drop_duplicates().groupby("cv_fold").size()
        expected = len(user_sizes) // N_FOLDS
        if set(counts.index) != set(range(N_FOLDS)) or not (counts == expected).all():
            raise RuntimeError("Balanced cross-user assignment did not preserve equal user counts")
    return out



def _build_fold_assignment(manifest: pd.DataFrame, cv_mode: str) -> pd.DataFrame:
    if cv_mode == CV_WITHIN:
        return _assign_within_user_folds(manifest)
    if cv_mode == CV_CROSS:
        return _assign_cross_user_folds(manifest)
    raise ValueError(cv_mode)


def _rotation_fold_roles(rotation: int) -> dict[int, str]:
    if rotation not in ROTATIONS:
        raise ValueError(rotation)
    test_fold = rotation
    val_fold = (rotation + 1) % N_FOLDS
    return {
        fold: ("test" if fold == test_fold else "val" if fold == val_fold else "train")
        for fold in range(N_FOLDS)
    }


def _apply_rotation(fold_manifest: pd.DataFrame, rotation: int) -> pd.DataFrame:
    roles = _rotation_fold_roles(rotation)
    out = fold_manifest.copy()
    out["split"] = out.cv_fold.map(roles)
    if out.split.isna().any():
        raise RuntimeError("Unassigned rotation split")
    for split in SPLITS:
        if not (out.split == split).any():
            raise RuntimeError(f"Empty {split} split for rotation {rotation}")
    return out


def _coverage_summary(split_manifest: pd.DataFrame) -> pd.DataFrame:
    total = split_manifest.groupby(["user", "label"]).size().rename("total")
    counts = (
        split_manifest.groupby(["user", "label", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SPLITS, fill_value=0)
    )
    out = pd.concat([total, counts], axis=1).reset_index()
    out["pair_present_in_train"] = out.train > 0
    out["pair_present_in_val"] = out.val > 0
    out["pair_present_in_test"] = out.test > 0
    out["test_pair_seen_in_train"] = (~out.pair_present_in_test) | out.pair_present_in_train
    return out


def _coverage_metrics(split_manifest: pd.DataFrame) -> dict[str, float]:
    train = split_manifest[split_manifest.split == "train"]
    test = split_manifest[split_manifest.split == "test"]
    train_users = set(train.user.astype(str))
    train_pairs = set(zip(train.user.astype(str), train.label.astype(str)))
    if test.empty:
        raise RuntimeError("Empty test split")
    same_user = [str(row.user) in train_users for row in test.itertuples(index=False)]
    same_pair = [
        (str(row.user), str(row.label)) in train_pairs
        for row in test.itertuples(index=False)
    ]
    return {
        "test_same_user_seen_fraction": float(np.mean(same_user)),
        "test_same_user_class_seen_fraction": float(np.mean(same_pair)),
        "test_samples": float(len(test)),
    }


def _fold_summary(fold_manifest: pd.DataFrame, cv_mode: str) -> pd.DataFrame:
    rows = []
    for fold in range(N_FOLDS):
        frame = fold_manifest[fold_manifest.cv_fold == fold]
        rows.append(
            {
                "cv_mode": cv_mode,
                "cv_fold": fold,
                "n_samples": int(len(frame)),
                "n_users": int(frame.user.nunique()),
                "n_classes": int(frame.label.nunique()),
                "users": "|".join(sorted(frame.user.astype(str).unique().tolist())),
            }
        )
    return pd.DataFrame(rows)


def _build_events(loaded, frame: pd.DataFrame, T: int) -> np.ndarray:
    out = np.zeros((len(frame), T, exp3.EVENT_CHANNELS), dtype=np.float32)
    for row_index, row in enumerate(frame.itertuples(index=False)):
        x = np.asarray(
            loaded.packages[int(row.pi)].padded_spike_imu[int(row.si), :, : exp3.EVENT_CHANNELS],
            dtype=np.float32,
        )
        n = min(len(x), T, int(row.valid))
        out[row_index, :n] = x[:n]
    return out


def _to_data(
    loaded,
    split_manifest: pd.DataFrame,
    labels: tuple[str, ...],
) -> tuple[exp3.Data, dict[str, pd.DataFrame]]:
    fs = float(loaded.producer_metadatas[0].sampling_rate_hz)
    original_T = int(split_manifest.pad.max())
    bin_steps = int(np.rint(exp3.FIXED_MS * fs / 1000.0))
    n_bins = int(math.ceil(original_T / bin_steps))
    T = n_bins * bin_steps
    if bin_steps != 16:
        raise ValueError(f"Exp9.0 requires 250 ms = 16 samples, got {bin_steps}")
    frames = {
        split: split_manifest[split_manifest.split == split]
        .sort_values("sample_id")
        .reset_index(drop=True)
        for split in SPLITS
    }
    arrays: list[np.ndarray] = []
    for split in SPLITS:
        frame = frames[split]
        arrays.extend(
            [
                _build_events(loaded, frame, T),
                frame.y.to_numpy(dtype=np.int64, copy=True),
                np.minimum(frame.valid.to_numpy(dtype=np.int64, copy=True), T).copy(),
            ]
        )
    split_info = {
        f"{split}_users": tuple(sorted(frames[split].user.astype(str).unique().tolist()))
        for split in SPLITS
    }
    return exp3.Data(*arrays, labels, fs, T, bin_steps, n_bins, split_info), frames


def _assignment_path(root: Path, cv_mode: str) -> Path:
    return root / "fold_assignments" / f"{cv_mode}.csv"


def _rotation_manifest_path(root: Path, cv_mode: str, rotation: int) -> Path:
    return root / "manifests" / f"{cv_mode}__rotation{rotation}.csv"


def _coverage_path(root: Path, cv_mode: str, rotation: int) -> Path:
    return root / "manifests" / f"{cv_mode}__rotation{rotation}__coverage.csv"


def _load_or_build_assignment(
    config: Config,
    raw_manifest: pd.DataFrame,
    cv_mode: str,
) -> pd.DataFrame:
    path = _assignment_path(config.results_dir, cv_mode)
    if path.exists():
        saved = pd.read_csv(path, usecols=["sample_id", "cv_fold"])
        if saved.sample_id.duplicated().any():
            raise RuntimeError(f"Duplicate sample_id in {path}")
        merged = raw_manifest.merge(saved, on="sample_id", how="inner", validate="one_to_one")
        if len(merged) != len(raw_manifest):
            raise RuntimeError(
                f"Saved fold assignment {path} does not match current dataset: "
                f"{len(merged)} vs {len(raw_manifest)} samples"
            )
        merged["cv_fold"] = merged.cv_fold.astype(int)
        return merged
    return _build_fold_assignment(raw_manifest, cv_mode)


def prepare_rotation(
    config: Config,
    cv_mode: str,
    rotation: int,
    write_artifacts: bool = False,
) -> PreparedSplit:
    loaded, raw_manifest, labels = _load_manifest(config.repo_root)
    assignment = _load_or_build_assignment(config, raw_manifest, cv_mode)
    split_manifest = _apply_rotation(assignment, rotation)
    coverage = _coverage_summary(split_manifest)
    coverage_metrics = _coverage_metrics(split_manifest)
    data, frames = _to_data(loaded, split_manifest, labels)
    if write_artifacts:
        manifest_path = _rotation_manifest_path(config.results_dir, cv_mode, rotation)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        split_manifest.to_csv(manifest_path, index=False)
        coverage.to_csv(_coverage_path(config.results_dir, cv_mode, rotation), index=False)
    return PreparedSplit(
        cv_mode=cv_mode,
        rotation=rotation,
        data=data,
        frames=frames,
        full_manifest=split_manifest,
        coverage_summary=coverage,
        coverage_metrics=coverage_metrics,
    )


def prepare_all(config: Config) -> dict[str, Any]:
    loaded, raw_manifest, _ = _load_manifest(config.repo_root)
    del loaded
    config.results_dir.mkdir(parents=True, exist_ok=True)
    assignment_dir = config.results_dir / "fold_assignments"
    manifest_dir = config.results_dir / "manifests"
    assignment_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    fold_frames: list[pd.DataFrame] = []
    rotation_rows: list[dict[str, Any]] = []
    for cv_mode in CV_MODES:
        assignment = _build_fold_assignment(raw_manifest, cv_mode)
        assignment.to_csv(_assignment_path(config.results_dir, cv_mode), index=False)
        fold_frames.append(_fold_summary(assignment, cv_mode))
        for rotation in ROTATIONS:
            split_manifest = _apply_rotation(assignment, rotation)
            split_manifest.to_csv(
                _rotation_manifest_path(config.results_dir, cv_mode, rotation),
                index=False,
            )
            coverage = _coverage_summary(split_manifest)
            coverage.to_csv(
                _coverage_path(config.results_dir, cv_mode, rotation),
                index=False,
            )
            metrics = _coverage_metrics(split_manifest)
            counts = split_manifest.split.value_counts()
            users = {
                split: int(split_manifest.loc[split_manifest.split == split, "user"].nunique())
                for split in SPLITS
            }
            rotation_rows.append(
                {
                    "cv_mode": cv_mode,
                    "rotation": rotation,
                    "test_fold": rotation,
                    "val_fold": (rotation + 1) % N_FOLDS,
                    "train_samples": int(counts.get("train", 0)),
                    "val_samples": int(counts.get("val", 0)),
                    "test_samples": int(counts.get("test", 0)),
                    "train_users": users["train"],
                    "val_users": users["val"],
                    "test_users": users["test"],
                    **metrics,
                }
            )

    fold_summary = pd.concat(fold_frames, ignore_index=True)
    fold_summary.to_csv(config.results_dir / "fold_summary.csv", index=False)
    rotation_summary = pd.DataFrame(rotation_rows)
    rotation_summary.to_csv(config.results_dir / "rotation_summary.csv", index=False)

    within_assignment = pd.read_csv(_assignment_path(config.results_dir, CV_WITHIN))
    within_user_fold = (
        within_assignment.groupby(["user", "cv_fold"])
        .size()
        .rename("n")
        .reset_index()
    )
    within_user_fold.to_csv(
        config.results_dir / "within_user_fold_counts.csv",
        index=False,
    )

    cross_assignment = pd.read_csv(_assignment_path(config.results_dir, CV_CROSS))
    cross_users = (
        cross_assignment[["user", "cv_fold"]]
        .drop_duplicates()
        .sort_values(["cv_fold", "user"])
        .reset_index(drop=True)
    )
    cross_users.to_csv(config.results_dir / "cross_user_fold_users.csv", index=False)

    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "total_samples": int(len(raw_manifest)),
        "total_users": int(raw_manifest.user.nunique()),
        "total_classes": int(raw_manifest.label.nunique()),
        "source_trials": int(raw_manifest.source_trial.nunique()),
        "n_folds": N_FOLDS,
        "rotations": list(ROTATIONS),
        "model_seeds": list(MODEL_SEEDS),
        "target_split": TARGET_FRACTIONS,
        "within_user_split_unit": "segment",
        "within_user_source_trial_grouping": False,
        "cross_user_split_unit": "user",
        "cross_user_user_grouping": True,
        "within_user_note": (
            "Segments are split within each user because the dataset has only 1-2 "
            "source trials per user; grouping source trials would make within-user "
            "5-fold CV impossible."
        ),
        "expected_runs": EXPECTED_RUNS,
    }
    _save_json(config.results_dir / "audit.json", audit)
    return audit


def _make_loaders(data: exp3.Data, seed: int, batch_size: int, shuffle_train: bool):
    parts = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    return {
        split: exp3.loader(
            X,
            y,
            lengths,
            batch_size,
            shuffle_train if split == "train" else False,
            _stable_seed(seed, EXPERIMENT_ID, split, "loader"),
        )
        for split, (X, y, lengths) in parts.items()
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _per_user_rows(spec: RunSpec, split: str, frame: pd.DataFrame, y_true: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    users = frame.user.to_numpy(dtype=object)
    for user in sorted(frame.user.unique().tolist()):
        mask = users == user
        metrics = _classification_metrics(y_true[mask], pred[mask])
        rows.append({"cv_mode": spec.cv_mode, "method": spec.method, "rotation": spec.rotation, "seed": spec.seed, "split": split, "user": user, "n": int(mask.sum()), **metrics})
    return rows


def _per_class_rows(spec: RunSpec, split: str, labels: tuple[str, ...], y_true: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=np.arange(len(labels)), zero_division=0
    )
    return [
        {
            "cv_mode": spec.cv_mode,
            "method": spec.method,
            "rotation": spec.rotation,
            "seed": spec.seed,
            "split": split,
            "class_index": idx,
            "class_label": labels[idx],
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx in range(len(labels))
    ]


def _diagnostic_tables(spec: RunSpec, prepared: PreparedSplit, predictions: dict[str, tuple[np.ndarray, np.ndarray]]):
    user_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusions: dict[str, list[list[int]]] = {}
    for split in SPLITS:
        y_true, pred = predictions[split]
        frame = prepared.frames[split]
        if len(frame) != len(y_true):
            raise RuntimeError(f"Prediction/frame length mismatch for {split}")
        user_rows.extend(_per_user_rows(spec, split, frame, y_true, pred))
        class_rows.extend(_per_class_rows(spec, split, prepared.data.labels, y_true, pred))
        confusions[split] = confusion_matrix(y_true, pred, labels=np.arange(len(prepared.data.labels))).astype(int).tolist()
    return pd.DataFrame(user_rows), pd.DataFrame(class_rows), confusions


def _fit_raw250(prepared: PreparedSplit, spec: RunSpec):
    data = prepared.data
    parts = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    built = {
        split: (exp01._raw_features(X, lengths, data, "fixed250"), y)
        for split, (X, y, lengths) in parts.items()
    }
    scaler = StandardScaler().fit(built["train"][0])
    z = {split: scaler.transform(values[0]) for split, values in built.items()}
    best: tuple[float, float, LogisticRegression] | None = None
    for C in exp302.PROBE_C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=5000,
            solver="lbfgs",
            random_state=_stable_seed(spec.seed, EXPERIMENT_ID, spec.cv_mode, spec.rotation, "raw250", C),
        ).fit(z["train"], built["train"][1])
        val_pred = classifier.predict(z["val"])
        val_ba = float(balanced_accuracy_score(built["val"][1], val_pred))
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError("No Raw250 LogisticRegression candidate selected")
    _, selected_C, classifier = best
    predictions = {split: (built[split][1], classifier.predict(z[split])) for split in SPLITS}
    metrics = {split: _classification_metrics(*predictions[split]) for split in SPLITS}
    return selected_C, metrics, predictions


def _evaluate_a2(model: exp80.Exp80Net, loader: Iterable, device: torch.device):
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum, n_total = 0.0, 0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            tr = model.forward_trajectory(X)
            scores = exp80._valid_mean(tr["evidence"], lengths)
            loss = F.cross_entropy(scores, y)
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)
    y_true = np.concatenate(ys)
    pred = np.concatenate(preds)
    metrics = _classification_metrics(y_true, pred)
    metrics["objective_loss"] = loss_sum / max(n_total, 1)
    return metrics, (y_true, pred)


def _train_a2(prepared: PreparedSplit, spec: RunSpec, config: Config):
    data = prepared.data
    device = torch.device(config.device)
    exp3.seed_all(_stable_seed(spec.seed, EXPERIMENT_ID, spec.cv_mode, spec.rotation, "a2_model_init"))
    model = exp80.Exp80Net(ARCHITECTURE_SHIFTS, len(data.labels), data.fs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY)
    train_loader = _make_loaders(data, spec.seed, config.batch_size, True)["train"]
    eval_loaders = _make_loaders(data, spec.seed, config.batch_size, False)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch, best_ba, best_loss = -1, -1.0, float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum, n_total = 0.0, 0
        for X, y, lengths in train_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            tr = model.forward_trajectory(X)
            scores = exp80._valid_mean(tr["evidence"], lengths)
            loss = F.cross_entropy(scores, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)
        train_metrics, _ = _evaluate_a2(model, eval_loaders["train"], device)
        val_metrics, _ = _evaluate_a2(model, eval_loaders["val"], device)
        history.append({
            "epoch": float(epoch),
            "train_ba": float(train_metrics["balanced_accuracy"]),
            "val_ba": float(val_metrics["balanced_accuracy"]),
            "train_loss": train_loss_sum / max(n_total, 1),
            "val_loss": float(val_metrics["objective_loss"]),
        })
        if exp73._checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE:
            stopped_epoch = epoch
            break
    if best_state is None:
        raise RuntimeError(f"No A2 checkpoint selected for {spec.key}")
    model.load_state_dict(best_state, strict=True)
    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, loader in eval_loaders.items():
        metrics[split], predictions[split] = _evaluate_a2(model, loader, device)
    return model, best_state, best_epoch, stopped_epoch, best_ba, best_loss, history, metrics, predictions


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    if (
        spec.cv_mode not in CV_MODES
        or spec.method not in METHODS
        or spec.rotation not in ROTATIONS
        or spec.seed != MODEL_SEEDS[spec.rotation]
    ):
        raise ValueError(spec)
    eval_path = _path(config.results_dir, "evaluations", spec.key, ".json")
    if eval_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))
    torch.set_num_threads(config.threads)
    prepared = prepare_rotation(config, spec.cv_mode, spec.rotation, write_artifacts=False)
    best_epoch = None
    stopped_epoch = None
    selected_C = None
    if spec.method == METHOD_RAW250:
        selected_C, metrics, predictions = _fit_raw250(prepared, spec)
        parameter_count = int(
            prepared.data.n_bins * exp3.EVENT_CHANNELS * len(prepared.data.labels)
        )
    else:
        (
            model,
            best_state,
            best_epoch,
            stopped_epoch,
            best_ba,
            best_loss,
            history,
            metrics,
            predictions,
        ) = _train_a2(prepared, spec, config)
        parameter_count = int(sum(p.numel() for p in model.parameters()))
        ckpt = _path(config.results_dir, "checkpoints", spec.key, ".pt")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "spec": asdict(spec),
                "architecture": ARCHITECTURE,
                "architecture_shifts": ARCHITECTURE_SHIFTS,
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "best_val_ba": best_ba,
                "best_val_objective_loss": best_loss,
                "model_state_dict": best_state,
            },
            ckpt,
        )
        hist_path = _path(config.results_dir, "histories", spec.key, ".csv")
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(hist_path, index=False)

    per_user, per_class, confusions = _diagnostic_tables(spec, prepared, predictions)
    per_user_path = _path(config.results_dir, "per_user", spec.key, ".csv")
    per_user_path.parent.mkdir(parents=True, exist_ok=True)
    per_user.to_csv(per_user_path, index=False)
    per_class_path = _path(config.results_dir, "per_class", spec.key, ".csv")
    per_class_path.parent.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(per_class_path, index=False)

    prediction_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        y_true, pred = predictions[split]
        frame = prepared.frames[split]
        for index, row in enumerate(frame.itertuples(index=False)):
            prediction_rows.append(
                {
                    "cv_mode": spec.cv_mode,
                    "method": spec.method,
                    "rotation": spec.rotation,
                    "seed": spec.seed,
                    "split": split,
                    "sample_id": str(row.sample_id),
                    "user": str(row.user),
                    "class_label": str(row.label),
                    "y_true": int(y_true[index]),
                    "y_pred": int(pred[index]),
                }
            )
    prediction_path = _path(config.results_dir, "predictions", spec.key, ".csv")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "target_split": TARGET_FRACTIONS,
        "split_unit": "segment" if spec.cv_mode == CV_WITHIN else "user",
        "architecture": ARCHITECTURE if spec.method == METHOD_A2 else None,
        "architecture_shifts": ARCHITECTURE_SHIFTS if spec.method == METHOD_A2 else None,
        "selected_probe_C": selected_C,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "parameter_count": parameter_count,
        "metrics": metrics,
        "confusion_matrices": confusions,
        "labels": list(prepared.data.labels),
        "counts": {split: int(len(prepared.frames[split])) for split in SPLITS},
        "split_users": {
            split: sorted(prepared.frames[split].user.astype(str).unique().tolist())
            for split in SPLITS
        },
        "coverage_metrics": prepared.coverage_metrics,
    }
    _save_json(eval_path, payload)
    return payload



def _summary_frame(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.groupby(groups, sort=False)[metrics].agg(["mean", "std"]).reset_index()
    out.columns = [
        "_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col)
        for col in out.columns
    ]
    return out


def _pooled_oof_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = predictions[predictions.split == "test"].copy()
    metric_rows: list[dict[str, Any]] = []
    user_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    for (cv_mode, method), frame in test.groupby(["cv_mode", "method"], sort=False):
        if frame.sample_id.duplicated().any():
            duplicated = frame.loc[frame.sample_id.duplicated(), "sample_id"].head().tolist()
            raise RuntimeError(
                f"OOF test samples repeated for {cv_mode}/{method}: {duplicated}"
            )
        metrics = _classification_metrics(
            frame.y_true.to_numpy(dtype=np.int64),
            frame.y_pred.to_numpy(dtype=np.int64),
        )
        metric_rows.append(
            {
                "cv_mode": cv_mode,
                "method": method,
                "n": int(len(frame)),
                **metrics,
            }
        )

        for user, user_frame in frame.groupby("user", sort=True):
            user_metrics = _classification_metrics(
                user_frame.y_true.to_numpy(dtype=np.int64),
                user_frame.y_pred.to_numpy(dtype=np.int64),
            )
            user_rows.append(
                {
                    "cv_mode": cv_mode,
                    "method": method,
                    "user": user,
                    "n": int(len(user_frame)),
                    **user_metrics,
                }
            )

        labels = sorted(frame.class_label.unique().tolist())
        label_to_y = (
            frame[["class_label", "y_true"]]
            .drop_duplicates()
            .set_index("class_label")
            .y_true.to_dict()
        )
        y_indices = [int(label_to_y[label]) for label in labels]
        precision, recall, f1, support = precision_recall_fscore_support(
            frame.y_true.to_numpy(dtype=np.int64),
            frame.y_pred.to_numpy(dtype=np.int64),
            labels=y_indices,
            zero_division=0,
        )
        for idx, label in enumerate(labels):
            class_rows.append(
                {
                    "cv_mode": cv_mode,
                    "method": method,
                    "class_label": label,
                    "class_index": y_indices[idx],
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1": float(f1[idx]),
                    "support": int(support[idx]),
                }
            )

        matrix = confusion_matrix(
            frame.y_true.to_numpy(dtype=np.int64),
            frame.y_pred.to_numpy(dtype=np.int64),
            labels=y_indices,
        )
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                confusion_rows.append(
                    {
                        "cv_mode": cv_mode,
                        "method": method,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": int(matrix[i, j]),
                    }
                )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(user_rows),
        pd.DataFrame(class_rows),
        pd.DataFrame(confusion_rows),
    )


def finalize(config: Config) -> dict[str, Any]:
    specs = run_specs()
    payloads = []
    for spec in specs:
        path = _path(config.results_dir, "evaluations", spec.key, ".json")
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp9.0 evaluation: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if len(payloads) != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} runs, got {len(payloads)}")

    run_rows = []
    for payload in payloads:
        spec = payload["spec"]
        for split in SPLITS:
            metrics = payload["metrics"][split]
            run_rows.append(
                {
                    "cv_mode": spec["cv_mode"],
                    "method": spec["method"],
                    "rotation": int(spec["rotation"]),
                    "seed": int(spec["seed"]),
                    "split": split,
                    "accuracy": float(metrics["accuracy"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "objective_loss": float(metrics.get("objective_loss", np.nan)),
                    "n": int(payload["counts"][split]),
                    "best_epoch": payload.get("best_epoch"),
                    "parameter_count": int(payload["parameter_count"]),
                    "test_same_user_seen_fraction": float(
                        payload["coverage_metrics"]["test_same_user_seen_fraction"]
                    ),
                    "test_same_user_class_seen_fraction": float(
                        payload["coverage_metrics"]["test_same_user_class_seen_fraction"]
                    ),
                }
            )
    runs = pd.DataFrame(run_rows)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "metric_runs.csv", index=False)
    _summary_frame(
        runs,
        ["cv_mode", "method", "split"],
        ["accuracy", "balanced_accuracy", "macro_f1", "objective_loss", "n"],
    ).to_csv(config.results_dir / "metric_summary.csv", index=False)

    users = pd.concat(
        [pd.read_csv(_path(config.results_dir, "per_user", spec.key, ".csv")) for spec in specs],
        ignore_index=True,
    )
    users.to_csv(config.results_dir / "per_user_runs.csv", index=False)
    _summary_frame(
        users,
        ["cv_mode", "method", "split", "user"],
        ["accuracy", "balanced_accuracy", "macro_f1", "n"],
    ).to_csv(config.results_dir / "per_user_summary.csv", index=False)

    classes = pd.concat(
        [pd.read_csv(_path(config.results_dir, "per_class", spec.key, ".csv")) for spec in specs],
        ignore_index=True,
    )
    classes.to_csv(config.results_dir / "per_class_runs.csv", index=False)
    _summary_frame(
        classes,
        ["cv_mode", "method", "split", "class_index", "class_label"],
        ["precision", "recall", "f1", "support"],
    ).to_csv(config.results_dir / "per_class_summary.csv", index=False)

    predictions = pd.concat(
        [pd.read_csv(_path(config.results_dir, "predictions", spec.key, ".csv")) for spec in specs],
        ignore_index=True,
    )
    predictions.to_csv(config.results_dir / "prediction_runs.csv", index=False)
    oof_test = predictions[predictions.split == "test"].copy()
    oof_test.to_csv(config.results_dir / "oof_test_predictions.csv", index=False)

    expected_samples = int(
        pd.read_csv(_assignment_path(config.results_dir, CV_WITHIN)).sample_id.nunique()
    )
    for (cv_mode, method), frame in oof_test.groupby(["cv_mode", "method"], sort=False):
        if len(frame) != expected_samples or frame.sample_id.nunique() != expected_samples:
            raise RuntimeError(
                f"Incomplete OOF coverage for {cv_mode}/{method}: "
                f"{len(frame)} rows, {frame.sample_id.nunique()} unique, "
                f"expected {expected_samples}"
            )

    oof_metrics, oof_users, oof_classes, oof_confusion = _pooled_oof_tables(predictions)
    oof_metrics.to_csv(config.results_dir / "oof_test_metrics.csv", index=False)
    oof_users.to_csv(config.results_dir / "oof_per_user_test.csv", index=False)
    oof_classes.to_csv(config.results_dir / "oof_per_class_test.csv", index=False)
    oof_confusion.to_csv(config.results_dir / "confusion_summary.csv", index=False)

    gap_rows = []
    for method in METHODS:
        within = oof_metrics[
            (oof_metrics.cv_mode == CV_WITHIN) & (oof_metrics.method == method)
        ].iloc[0]
        cross = oof_metrics[
            (oof_metrics.cv_mode == CV_CROSS) & (oof_metrics.method == method)
        ].iloc[0]
        gap_rows.append(
            {
                "method": method,
                "within_user_oof_accuracy": float(within.accuracy),
                "within_user_oof_ba": float(within.balanced_accuracy),
                "within_user_oof_macro_f1": float(within.macro_f1),
                "cross_user_oof_accuracy": float(cross.accuracy),
                "cross_user_oof_ba": float(cross.balanced_accuracy),
                "cross_user_oof_macro_f1": float(cross.macro_f1),
                "user_gap_accuracy": float(within.accuracy - cross.accuracy),
                "user_gap_ba": float(within.balanced_accuracy - cross.balanced_accuracy),
                "user_gap_macro_f1": float(within.macro_f1 - cross.macro_f1),
            }
        )
    pd.DataFrame(gap_rows).to_csv(
        config.results_dir / "within_vs_cross_user.csv",
        index=False,
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "cv_modes": list(CV_MODES),
        "methods": list(METHODS),
        "n_folds": N_FOLDS,
        "rotations": list(ROTATIONS),
        "model_seeds": list(MODEL_SEEDS),
        "parallel_runs": EXPECTED_RUNS,
        "target_split": TARGET_FRACTIONS,
        "within_user": {
            "split_unit": "segment",
            "stratified_within_each_user": True,
            "source_trial_grouping": False,
        },
        "cross_user": {
            "split_unit": "user",
            "user_grouping": True,
            "test_users_unseen": True,
        },
        "primary_metric": "pooled out-of-fold balanced_accuracy",
        "secondary_metrics": ["accuracy", "macro_f1", "fold mean/std"],
        "a2_architecture": ARCHITECTURE,
        "a2_architecture_shifts": [list(v) for v in ARCHITECTURE_SHIFTS],
        "raw250": (
            "Raw64 events -> ordered 250-ms counts -> StandardScaler(train only) -> "
            "validation-selected LogisticRegression C"
        ),
        "a2": (
            "Exp7.3 A2-compatible 30->128->128->12, shifts (234)(234), "
            "shared Linear/WCCE, validation BA checkpoint selection"
        ),
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _resolve_config(args: argparse.Namespace) -> Config:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    out = Path(args.results_dir).resolve() if args.results_dir else results_dir(repo_root)
    return Config(
        repo_root=repo_root,
        results_dir=out,
        device=args.device,
        threads=args.threads,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp9.0 rotating within-user and cross-user CV")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("list-runs")
    run = sub.add_parser("run")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _resolve_config(args)
    specs = run_specs()
    if args.command == "prepare":
        print(json.dumps(prepare_all(config), indent=2))
        return
    if args.command == "list-runs":
        for index, spec in enumerate(specs):
            print(index, spec.key)
        return
    if args.command == "run":
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        spec = specs[args.array_task_id]
        payload = run_one(spec, config, force=args.force)
        print(json.dumps({"key": spec.key, "test_ba": payload["metrics"]["test"]["balanced_accuracy"], "test_macro_f1": payload["metrics"]["test"]["macro_f1"]}, indent=2))
        return
    if args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
