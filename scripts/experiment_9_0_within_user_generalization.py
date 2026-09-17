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
PROTOCOL_VERSION = "within_user_segment_generalization_v1"
METHOD_RAW250 = "raw250_linear"
METHOD_A2 = "a2_234x234"
METHODS = (METHOD_RAW250, METHOD_A2)
SPLIT_SEEDS = (11, 23, 37, 53, 71)
EXPECTED_RUNS = len(METHODS) * len(SPLIT_SEEDS)
INSUFFICIENT_POLICIES = ("keep_all", "error", "exclude_pair", "exclude_user", "exclude_class")
TARGET_FRACTIONS = {"train": 0.60, "val": 0.20, "test": 0.20}
SPLITS = ("train", "val", "test")
ARCHITECTURE = "234x234"
ARCHITECTURE_SHIFTS = ((2, 3, 4), (2, 3, 4))
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
BATCH_SIZE = exp72.BATCH_SIZE
TRIAL_SEARCH_ATTEMPTS = 20000


@dataclass(frozen=True)
class RunSpec:
    method: str
    split_seed: int

    @property
    def key(self) -> str:
        return f"{self.method}__splitseed{self.split_seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    threads: int = 1
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    insufficient_policy: str = "keep_all"


@dataclass
class PreparedSplit:
    data: exp3.Data
    frames: dict[str, pd.DataFrame]
    full_manifest: pd.DataFrame
    pair_summary: pd.DataFrame
    trial_summary: pd.DataFrame
    excluded: dict[str, list[str]]


def find_repo_root(start: Path | None = None) -> Path:
    return exp80.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(method, seed) for method in METHODS for seed in SPLIT_SEEDS]


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _stable_seed(seed: int, *parts: object) -> int:
    text = "|".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def _allocate_counts(n: int, seed: int = 0) -> dict[str, int]:
    """Largest-remainder 60/20/20 allocation with >=1 item per split when n>=3."""
    if n < 3:
        return {"train": max(n, 0), "val": 0, "test": 0}
    ideal = {split: TARGET_FRACTIONS[split] * n for split in SPLITS}
    counts = {split: int(math.floor(ideal[split])) for split in SPLITS}
    counts["train"] = max(counts["train"], 1)
    counts["val"] = max(counts["val"], 1)
    counts["test"] = max(counts["test"], 1)
    while sum(counts.values()) > n:
        candidates = [s for s in SPLITS if counts[s] > 1]
        if not candidates:
            raise RuntimeError(f"Could not allocate {n} samples across three splits")
        split = max(candidates, key=lambda s: counts[s] - ideal[s])
        counts[split] -= 1
    rng = np.random.default_rng(seed)
    while sum(counts.values()) < n:
        deficits = {split: ideal[split] - counts[split] for split in SPLITS}
        best = max(deficits.values())
        tied = [split for split in SPLITS if abs(deficits[split] - best) <= 1e-12]
        split = tied[int(rng.integers(len(tied)))]
        counts[split] += 1
    return counts


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
    return loaded, manifest, labels


def _initial_pair_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    all_users = sorted(manifest.user.unique().tolist())
    all_labels = sorted(manifest.label.unique().tolist())
    index = pd.MultiIndex.from_product([all_users, all_labels], names=["user", "label"])
    counts = manifest.groupby(["user", "label"]).size().reindex(index, fill_value=0)
    rows = []
    for (user, label), total in counts.items():
        total = int(total)
        target = _allocate_counts(total, _stable_seed(0, user, label))
        rows.append(
            {
                "user": user,
                "label": label,
                "total": total,
                "target_train": target["train"],
                "target_val": target["val"],
                "target_test": target["test"],
                "strict_three_way_eligible": bool(total >= 3),
                "status": "eligible" if total >= 3 else "insufficient",
            }
        )
    return pd.DataFrame(rows)


def _present_pair_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (user, label), frame in manifest.groupby(["user", "label"], sort=True):
        total = int(len(frame))
        target = _allocate_counts(total, _stable_seed(0, user, label))
        rows.append(
            {
                "user": str(user),
                "label": str(label),
                "total": total,
                "target_train": target["train"],
                "target_val": target["val"],
                "target_test": target["test"],
                "strict_three_way_eligible": bool(total >= 3),
                "status": "eligible" if total >= 3 else "insufficient",
            }
        )
    return pd.DataFrame(rows)


def _apply_insufficient_policy(
    manifest: pd.DataFrame,
    pair_summary: pd.DataFrame,
    policy: str,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if policy not in INSUFFICIENT_POLICIES:
        raise ValueError(policy)
    bad = pair_summary[~pair_summary.strict_three_way_eligible]
    excluded = {"users": [], "classes": [], "pairs": []}
    if bad.empty or policy == "keep_all":
        return manifest.copy(), excluded
    excluded["pairs"] = [f"{r.user}:{r.label}" for r in bad.itertuples(index=False)]
    if policy == "error":
        raise RuntimeError(
            "Exp9.0 has (user,class) pairs with fewer than 3 samples. "
            "Inspect manifests/insufficient_user_class_pairs.csv, then rerun with "
            "--insufficient-policy keep_all, exclude_pair, exclude_user, or exclude_class."
        )
    if policy == "exclude_pair":
        bad_pairs = {(str(r.user), str(r.label)) for r in bad.itertuples(index=False)}
        keep_mask = [
            (str(row.user), str(row.label)) not in bad_pairs
            for row in manifest.itertuples(index=False)
        ]
        return manifest.loc[keep_mask].reset_index(drop=True), excluded
    if policy == "exclude_user":
        users = sorted(bad.user.astype(str).unique().tolist())
        excluded["users"] = users
        return manifest[~manifest.user.isin(users)].reset_index(drop=True), excluded
    classes = sorted(bad.label.astype(str).unique().tolist())
    excluded["classes"] = classes
    return manifest[~manifest.label.isin(classes)].reset_index(drop=True), excluded


def _assignment_score(
    frame: pd.DataFrame,
    trial_assignment: dict[str, str],
    target: dict[tuple[str, str], int],
) -> tuple[float, bool]:
    split = frame.source_trial.map(trial_assignment)
    if split.isna().any():
        return float("inf"), False
    work = frame.assign(_split=split.to_numpy())
    counts = work.groupby(["label", "_split"]).size()
    labels = sorted(frame.label.unique().tolist())
    feasible = True
    score = 0.0
    for label in labels:
        label_total = int((frame.label == label).sum())
        for part in SPLITS:
            actual = int(counts.get((label, part), 0))
            wanted = int(target[(label, part)])
            score += abs(actual - wanted) / max(wanted, 1)
            # Three-way per-(user,class) coverage is impossible when fewer than
            # three samples exist. For eligible pairs, treat missing coverage as
            # a strong preference rather than a hard feasibility constraint,
            # because source-trial isolation can still make exact coverage
            # impossible (e.g. all samples from one trial).
            if label_total >= 3 and actual < 1:
                score += 10.0
    total = len(frame)
    split_counts = work._split.value_counts()
    for part in SPLITS:
        actual_fraction = float(split_counts.get(part, 0)) / max(total, 1)
        score += 0.25 * abs(actual_fraction - TARGET_FRACTIONS[part])
    return score, feasible


def _assign_user_trials(frame: pd.DataFrame, seed: int) -> dict[str, str]:
    trials = sorted(frame.source_trial.unique().tolist())
    if len(trials) < 3:
        raise RuntimeError(
            f"User {frame.user.iloc[0]} has only {len(trials)} source trials; "
            "source-trial-isolated train/val/test is impossible."
        )
    targets: dict[tuple[str, str], int] = {}
    for label, group in frame.groupby("label", sort=True):
        alloc = _allocate_counts(len(group), _stable_seed(seed, frame.user.iloc[0], label))
        for part in SPLITS:
            targets[(str(label), part)] = int(alloc[part])

    trial_alloc = _allocate_counts(len(trials), _stable_seed(seed, frame.user.iloc[0], "trials"))
    counts = [trial_alloc["train"], trial_alloc["val"], trial_alloc["test"]]
    rng = np.random.default_rng(_stable_seed(seed, frame.user.iloc[0], "assignment_search"))
    best_score = float("inf")
    best: dict[str, str] | None = None
    trial_array = np.asarray(trials, dtype=object)
    for _ in range(TRIAL_SEARCH_ATTEMPTS):
        perm = rng.permutation(trial_array)
        candidate: dict[str, str] = {}
        cursor = 0
        for part, n_part in zip(SPLITS, counts, strict=True):
            for trial in perm[cursor : cursor + n_part]:
                candidate[str(trial)] = part
            cursor += n_part
        score, feasible = _assignment_score(frame, candidate, targets)
        if feasible and score < best_score - 1e-12:
            best_score = score
            best = candidate
            if score <= 1e-12:
                break
    if best is None:
        raise RuntimeError(
            f"Could not find a source-trial-isolated 60/20/20 split for user "
            f"{frame.user.iloc[0]}."
        )
    return best


def _build_split_manifest(
    manifest: pd.DataFrame,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    trial_rows = []
    for user, frame in manifest.groupby("user", sort=True):
        assignment = _assign_user_trials(frame.reset_index(drop=True), split_seed)
        part = frame.copy()
        part["split"] = part.source_trial.map(assignment)
        if part.split.isna().any():
            raise RuntimeError(f"Unassigned source trial for {user}")
        pieces.append(part)
        for trial, split in sorted(assignment.items()):
            trial_rows.append({"split_seed": split_seed, "user": user, "source_trial": trial, "split": split})
    out = pd.concat(pieces, ignore_index=True)
    by_trial = out.groupby("source_trial").split.nunique()
    if int(by_trial.max()) != 1:
        raise AssertionError("A source trial appears in more than one split")
    for split in SPLITS:
        if not (out.split == split).any():
            raise RuntimeError(f"Empty split: {split}")
    return out, pd.DataFrame(trial_rows)


def _pair_actual_summary(split_manifest: pd.DataFrame, pair_summary: pd.DataFrame) -> pd.DataFrame:
    actual = (
        split_manifest.groupby(["user", "label", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for part in SPLITS:
        if part not in actual.columns:
            actual[part] = 0
    merged = pair_summary.merge(actual, on=["user", "label"], how="left")
    for part in SPLITS:
        merged[part] = merged[part].fillna(0).astype(int)
        merged[f"actual_{part}"] = merged[part]
        merged[f"delta_{part}"] = merged[f"actual_{part}"] - merged[f"target_{part}"]
    merged["actual_three_way_coverage"] = (
        (merged.actual_train >= 1) & (merged.actual_val >= 1) & (merged.actual_test >= 1)
    )
    return merged.drop(columns=list(SPLITS))


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


def _to_data(loaded, split_manifest: pd.DataFrame, labels: tuple[str, ...]) -> tuple[exp3.Data, dict[str, pd.DataFrame]]:
    fs = float(loaded.producer_metadatas[0].sampling_rate_hz)
    original_T = int(split_manifest.pad.max())
    bin_steps = int(np.rint(exp3.FIXED_MS * fs / 1000.0))
    n_bins = int(math.ceil(original_T / bin_steps))
    T = n_bins * bin_steps
    if bin_steps != 16:
        raise ValueError(f"Exp9.0 requires 250 ms = 16 samples, got {bin_steps}")
    frames = {
        split: split_manifest[split_manifest.split == split].sort_values("sample_id").reset_index(drop=True)
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
    all_users = tuple(sorted(split_manifest.user.unique().tolist()))
    split_info = {"train_users": all_users, "val_users": all_users, "test_users": all_users}
    return exp3.Data(*arrays, labels, fs, T, bin_steps, n_bins, split_info), frames


def _audit_paths(root: Path, split_seed: int) -> dict[str, Path]:
    base = root / "manifests"
    return {
        "manifest": base / f"split_seed{split_seed}.csv",
        "summary": base / f"manifest_summary_seed{split_seed}.csv",
        "trials": base / f"source_trial_summary_seed{split_seed}.csv",
    }


def prepare_split(config: Config, split_seed: int, write_artifacts: bool = True) -> PreparedSplit:
    loaded, raw_manifest, _ = _load_manifest(config.repo_root)
    initial_summary = _initial_pair_summary(raw_manifest)
    insufficient = initial_summary[~initial_summary.strict_three_way_eligible].copy()
    if write_artifacts:
        manifest_dir = config.results_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        insufficient.to_csv(manifest_dir / "insufficient_user_class_pairs.csv", index=False)
        initial_summary.to_csv(manifest_dir / "initial_pair_summary.csv", index=False)
    filtered, excluded = _apply_insufficient_policy(raw_manifest, initial_summary, config.insufficient_policy)
    if filtered.empty:
        raise RuntimeError("Insufficient-data policy removed all samples")
    active_summary = (
        _present_pair_summary(filtered)
        if config.insufficient_policy == "exclude_pair"
        else _initial_pair_summary(filtered)
    )
    split_manifest, trial_summary = _build_split_manifest(filtered, split_seed)
    actual_summary = _pair_actual_summary(split_manifest, active_summary)
    actual_summary["coverage_expected"] = actual_summary.strict_three_way_eligible
    actual_summary["coverage_met_when_expected"] = (
        (~actual_summary.coverage_expected) | actual_summary.actual_three_way_coverage
    )
    data, frames = _to_data(loaded, split_manifest, tuple(sorted(filtered.label.unique())))
    if write_artifacts:
        paths = _audit_paths(config.results_dir, split_seed)
        paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
        split_manifest.to_csv(paths["manifest"], index=False)
        actual_summary.to_csv(paths["summary"], index=False)
        trial_summary.to_csv(paths["trials"], index=False)
    return PreparedSplit(data, frames, split_manifest, actual_summary, trial_summary, excluded)


def prepare_all(config: Config) -> dict[str, Any]:
    loaded, raw_manifest, _ = _load_manifest(config.repo_root)
    del loaded
    initial = _initial_pair_summary(raw_manifest)
    manifest_dir = config.results_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    initial.to_csv(manifest_dir / "initial_pair_summary.csv", index=False)
    insufficient = initial[~initial.strict_three_way_eligible].copy()
    insufficient.to_csv(manifest_dir / "insufficient_user_class_pairs.csv", index=False)
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "insufficient_policy": config.insufficient_policy,
        "total_samples": int(len(raw_manifest)),
        "total_users": int(raw_manifest.user.nunique()),
        "total_classes": int(raw_manifest.label.nunique()),
        "total_user_class_pairs": int(len(initial)),
        "insufficient_user_class_pairs": int(len(insufficient)),
        "affected_users": sorted(insufficient.user.astype(str).unique().tolist()),
        "affected_classes": sorted(insufficient.label.astype(str).unique().tolist()),
        "target_split": TARGET_FRACTIONS,
        "source_trial_isolation": True,
    }
    _save_json(manifest_dir / "audit.json", audit)
    prepared = []
    for seed in SPLIT_SEEDS:
        try:
            prepared.append(prepare_split(config, seed, write_artifacts=True))
        except Exception as error:
            _save_json(
                manifest_dir / f"split_failure_seed{seed}.json",
                {"split_seed": int(seed), "error_type": type(error).__name__, "error": str(error)},
            )
            raise
    audit["prepared_split_seeds"] = list(SPLIT_SEEDS)
    audit["active_samples_per_seed"] = {
        str(seed): int(len(prep.full_manifest)) for seed, prep in zip(SPLIT_SEEDS, prepared, strict=True)
    }
    audit["excluded"] = prepared[0].excluded if prepared else {}
    _save_json(manifest_dir / "audit.json", audit)
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
        rows.append({"method": spec.method, "split_seed": spec.split_seed, "split": split, "user": user, "n": int(mask.sum()), **metrics})
    return rows


def _per_class_rows(spec: RunSpec, split: str, labels: tuple[str, ...], y_true: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=np.arange(len(labels)), zero_division=0
    )
    return [
        {
            "method": spec.method,
            "split_seed": spec.split_seed,
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
            random_state=_stable_seed(spec.split_seed, EXPERIMENT_ID, "raw250", C),
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
    exp3.seed_all(_stable_seed(spec.split_seed, EXPERIMENT_ID, "a2_model_init"))
    model = exp80.Exp80Net(ARCHITECTURE_SHIFTS, len(data.labels), data.fs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY)
    train_loader = _make_loaders(data, spec.split_seed, config.batch_size, True)["train"]
    eval_loaders = _make_loaders(data, spec.split_seed, config.batch_size, False)
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
    if spec.method not in METHODS or spec.split_seed not in SPLIT_SEEDS:
        raise ValueError(spec)
    eval_path = _path(config.results_dir, "evaluations", spec.key, ".json")
    if eval_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))
    torch.set_num_threads(config.threads)
    prepared = prepare_split(config, spec.split_seed, write_artifacts=False)
    best_epoch = None
    stopped_epoch = None
    selected_C = None
    if spec.method == METHOD_RAW250:
        selected_C, metrics, predictions = _fit_raw250(prepared, spec)
        parameter_count = int(prepared.data.n_bins * exp3.EVENT_CHANNELS * len(prepared.data.labels))
    else:
        model, best_state, best_epoch, stopped_epoch, best_ba, best_loss, history, metrics, predictions = _train_a2(prepared, spec, config)
        parameter_count = int(sum(p.numel() for p in model.parameters()))
        ckpt = _path(config.results_dir, "checkpoints", spec.key, ".pt")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
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
        }, ckpt)
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
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "insufficient_policy": config.insufficient_policy,
        "target_split": TARGET_FRACTIONS,
        "source_trial_isolation": True,
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
        "users": sorted(prepared.full_manifest.user.unique().tolist()),
        "excluded": prepared.excluded,
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


def _cross_user_reference(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_path = repo_root / "notebooks/artifacts/experiment_0_1_general_comparison/general_comparison_v1/raw_baselines.json"
    if raw_path.exists():
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        raw = payload.get("representations", {}).get("fixed250", {})
        if isinstance(raw, dict) and isinstance(raw.get("test"), dict):
            rows.append({
                "method": METHOD_RAW250,
                "source": str(raw_path.relative_to(repo_root)),
                "cross_user_test_ba": float(raw["test"]["balanced_accuracy"]),
                "cross_user_test_accuracy": float(raw["test"]["accuracy"]),
                "cross_user_test_macro_f1": float(raw["test"]["macro_f1"]),
            })
    exp80_root = repo_root / "notebooks/artifacts/experiment_8_0_local_backbone_tau_sweep/local_backbone_tau_sweep_v1/evaluations"
    a2_values = []
    if exp80_root.exists():
        for path in sorted(exp80_root.glob("234x234__a2_wcce__*__seed*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            test = payload.get("linear_metrics", {}).get("test", {})
            if isinstance(test, dict) and "balanced_accuracy" in test:
                a2_values.append(test)
    if a2_values:
        rows.append({
            "method": METHOD_A2,
            "source": str(exp80_root.relative_to(repo_root)),
            "cross_user_test_ba": float(np.mean([v["balanced_accuracy"] for v in a2_values])),
            "cross_user_test_accuracy": float(np.mean([v["accuracy"] for v in a2_values])),
            "cross_user_test_macro_f1": float(np.mean([v["macro_f1"] for v in a2_values])),
        })
    return rows


def finalize(config: Config) -> dict[str, Any]:
    payloads = []
    for spec in run_specs():
        path = _path(config.results_dir, "evaluations", spec.key, ".json")
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp9.0 evaluation: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if len(payloads) != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} runs, got {len(payloads)}")
    run_rows = []
    for payload in payloads:
        for split in SPLITS:
            metrics = payload["metrics"][split]
            run_rows.append({
                "method": payload["spec"]["method"],
                "split_seed": int(payload["spec"]["split_seed"]),
                "split": split,
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "objective_loss": float(metrics.get("objective_loss", np.nan)),
                "n": int(payload["counts"][split]),
                "best_epoch": payload.get("best_epoch"),
                "parameter_count": int(payload["parameter_count"]),
            })
    runs = pd.DataFrame(run_rows)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "metric_runs.csv", index=False)
    _summary_frame(runs, ["method", "split"], ["accuracy", "balanced_accuracy", "macro_f1", "objective_loss", "n"]).to_csv(config.results_dir / "metric_summary.csv", index=False)
    users = pd.concat([pd.read_csv(_path(config.results_dir, "per_user", spec.key, ".csv")) for spec in run_specs()], ignore_index=True)
    users.to_csv(config.results_dir / "per_user_runs.csv", index=False)
    _summary_frame(users, ["method", "split", "user"], ["accuracy", "balanced_accuracy", "macro_f1", "n"]).to_csv(config.results_dir / "per_user_summary.csv", index=False)
    classes = pd.concat([pd.read_csv(_path(config.results_dir, "per_class", spec.key, ".csv")) for spec in run_specs()], ignore_index=True)
    classes.to_csv(config.results_dir / "per_class_runs.csv", index=False)
    _summary_frame(classes, ["method", "split", "class_index", "class_label"], ["precision", "recall", "f1", "support"]).to_csv(config.results_dir / "per_class_summary.csv", index=False)
    confusion_rows = []
    for method in METHODS:
        method_payloads = [p for p in payloads if p["spec"]["method"] == method]
        labels = method_payloads[0]["labels"]
        for split in SPLITS:
            matrices = np.asarray([p["confusion_matrices"][split] for p in method_payloads], dtype=float)
            mean_matrix = matrices.mean(axis=0)
            for i, true_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    confusion_rows.append({"method": method, "split": split, "true_label": true_label, "pred_label": pred_label, "mean_count": float(mean_matrix[i, j])})
    pd.DataFrame(confusion_rows).to_csv(config.results_dir / "confusion_summary.csv", index=False)
    within_test = runs[runs.split == "test"].groupby("method")[["balanced_accuracy", "accuracy", "macro_f1"]].agg(["mean", "std"])
    gap_rows = []
    refs = {row["method"]: row for row in _cross_user_reference(config.repo_root)}
    for method in METHODS:
        row: dict[str, Any] = {"method": method}
        if method in within_test.index:
            row["within_user_test_ba_mean"] = float(within_test.loc[method, ("balanced_accuracy", "mean")])
            row["within_user_test_ba_std"] = float(within_test.loc[method, ("balanced_accuracy", "std")])
            row["within_user_test_macro_f1_mean"] = float(within_test.loc[method, ("macro_f1", "mean")])
        ref = refs.get(method)
        if ref:
            row.update(ref)
            row["user_gap_ba"] = row["within_user_test_ba_mean"] - row["cross_user_test_ba"]
        gap_rows.append(row)
    pd.DataFrame(gap_rows).to_csv(config.results_dir / "within_vs_cross_user.csv", index=False)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "methods": list(METHODS),
        "split_seeds": list(SPLIT_SEEDS),
        "parallel_runs": EXPECTED_RUNS,
        "target_split": TARGET_FRACTIONS,
        "source_trial_isolation": True,
        "insufficient_policy": config.insufficient_policy,
        "a2_architecture": ARCHITECTURE,
        "a2_architecture_shifts": [list(v) for v in ARCHITECTURE_SHIFTS],
        "primary_metric": "balanced_accuracy",
        "secondary_metrics": ["accuracy", "macro_f1"],
        "raw250": "Raw64 events -> ordered 250-ms counts -> StandardScaler(train only) -> validation-selected LogisticRegression C",
        "a2": "Exp7.3 A2-compatible 30->128->128->12, shifts (234)(234), shared Linear/WCCE, validation BA checkpoint selection",
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
        insufficient_policy=args.insufficient_policy,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp9.0 within-user unseen-segment generalization")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--insufficient-policy", choices=INSUFFICIENT_POLICIES, default="keep_all")
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
