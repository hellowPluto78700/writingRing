from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


EXPERIMENT_ID = "withGyro_experiment_0_1_temporal_representation_probe"
PROTOCOL_VERSION = "linear_angular_accel_60event_v2"

SPLIT_SEEDS = (11, 23, 37, 53, 71)
N_TRAIN_USERS = 12
N_VAL_USERS = 4
N_TEST_USERS = 4

FIXED_DURATION_MS = (
    50.0,
    150.0,
    250.0,
    350.0,
    450.0,
    550.0,
    650.0,
    750.0,
    850.0,
    950.0,
    1050.0,
)
RELATIVE_N_BINS = (1, 2, 4, 6, 8, 10, 12, 16, 20)
CLASSIFIERS = ("linear", "5nn")

EVENT_CHANNEL_COUNT = 60
TOTAL_CHANNEL_COUNT = 66
EXPECTED_SAMPLING_RATE_HZ = 64.0
EXPECTED_PADDED_LENGTH = 256
EXPECTED_FEATURE_SCHEMA = (
    "linear_accel_angular_accel_polarity_split_wavelet_events_plus_imu_v1"
)
EXPECTED_EVENT_FEATURE_SCHEMA = (
    "linear_accel_angular_accel_polarity_split_abs_events_v1"
)
EXPECTED_EVENT_REPRESENTATION = "unsigned"
INCLUDED_LABELS = ("A", "B", "C", "D", "E", "X", "G", "H", "I", "J", "K", "L")

KNN_K = 5
LOGREG_MAX_ITER = 5000
STD_EPS = 1e-8

DATASET_RELATIVE_ROOT = Path(
    "outputs/action0_wavelets_0e5_1_2_4_8_sr_64_with_gyro_0e5_1_2_4_8/"
    "low-pass/aligned-board-events"
)


@dataclass(frozen=True)
class RunSpec:
    representation_family: str
    value: float | int
    split_seed: int

    @property
    def condition(self) -> str:
        if self.representation_family == "fixed_duration":
            return f"fixed_{int(round(float(self.value))):04d}ms"
        if self.representation_family == "relative_progress":
            return f"relative_{int(self.value):02d}bin"
        raise ValueError(f"Unknown representation family: {self.representation_family}")

    @property
    def key(self) -> str:
        return f"{self.condition}__split{self.split_seed}"


@dataclass(slots=True)
class Package:
    user: str
    action: str
    directory: Path
    padded_spike_imu: np.ndarray
    labels: np.ndarray
    valid_lengths: np.ndarray
    valid_mask: np.ndarray


@dataclass(slots=True)
class Cohort:
    manifest: pd.DataFrame
    packages: list[Package]
    fs: float
    padded_length: int
    labels: tuple[str, ...]
    dataset_root: Path


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "AGENTS.md").is_file() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate writingRing repository root")


def results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / "withGyro"
        / "experiment_0_1_temporal_representation_probe"
        / PROTOCOL_VERSION
    )


def dataset_root(repo_root: Path) -> Path:
    return repo_root / DATASET_RELATIVE_ROOT


def derive_seed(seed: int, *parts: object) -> int:
    text = "|".join(map(str, (seed, *parts)))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def run_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for duration_ms in FIXED_DURATION_MS:
        specs.extend(
            RunSpec("fixed_duration", duration_ms, split_seed)
            for split_seed in SPLIT_SEEDS
        )
    for n_bins in RELATIVE_N_BINS:
        specs.extend(
            RunSpec("relative_progress", n_bins, split_seed)
            for split_seed in SPLIT_SEEDS
        )
    return specs


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _exact_one(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern))
    if len(found) != 1:
        raise ValueError(
            f"Expected exactly one {pattern!r} under {directory}; found {len(found)}"
        )
    return found[0]


def _validate_contract(payload: Mapping[str, Any], *, context: str) -> None:
    expected = {
        "feature_schema": EXPECTED_FEATURE_SCHEMA,
        "event_feature_schema": EXPECTED_EVENT_FEATURE_SCHEMA,
        "event_representation": EXPECTED_EVENT_REPRESENTATION,
        "event_channel_count": EVENT_CHANNEL_COUNT,
        "channel_count": TOTAL_CHANNEL_COUNT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{context}: expected {key}={value!r}, got {payload.get(key)!r}"
            )
    rate = float(payload.get("sampling_rate_hz", 0.0))
    if not math.isclose(rate, EXPECTED_SAMPLING_RATE_HZ, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{context}: expected sampling_rate_hz={EXPECTED_SAMPLING_RATE_HZ}, got {rate}"
        )


def load_cohort(repo_root: Path) -> Cohort:
    combination_root = dataset_root(repo_root)
    dataset_summary_path = combination_root / "angular_accel66_dataset_summary.json"
    if not dataset_summary_path.is_file():
        raise FileNotFoundError(
            "Missing finalized Angular66 dataset summary. Finish the preprocessing finalizer first: "
            f"{dataset_summary_path}"
        )
    dataset_summary = _load_json(dataset_summary_path)
    if dataset_summary.get("status") != "PASS":
        raise ValueError(f"Angular66 dataset is not PASS: {dataset_summary_path}")
    _validate_contract(dataset_summary, context=str(dataset_summary_path))

    padded_root = combination_root / "segmentation_padded"
    root_padding_summary_path = padded_root / "padding_dataset_summary.json"
    if not root_padding_summary_path.is_file():
        raise FileNotFoundError(f"Missing padding dataset summary: {root_padding_summary_path}")
    root_padding_summary = _load_json(root_padding_summary_path)
    _validate_contract(root_padding_summary, context=str(root_padding_summary_path))
    target_length = int(root_padding_summary.get("target_length", 0))
    if target_length != EXPECTED_PADDED_LENGTH:
        raise ValueError(
            f"Expected padded length {EXPECTED_PADDED_LENGTH}, got {target_length}"
        )

    summary_paths = sorted(padded_root.glob("user_*/action_0/*_padding_summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No action-0 padded packages found below {padded_root}")

    packages: list[Package] = []
    rows: list[dict[str, object]] = []
    keep = set(INCLUDED_LABELS)
    for summary_path in summary_paths:
        directory = summary_path.parent
        package_summary = _load_json(summary_path)
        _validate_contract(package_summary, context=str(summary_path))

        user = directory.parent.name
        action = directory.name.removeprefix("action_")
        if action != "0":
            raise ValueError(f"Expected action 0 package, got {directory}")

        padded_path = _exact_one(directory, "*_paddedSpikeIMU.npy")
        labels_path = _exact_one(directory, "*_labels.npy")
        lengths_path = _exact_one(directory, "*_valid_lengths.npy")
        mask_path = _exact_one(directory, "*_valid_mask.npy")

        padded = np.load(padded_path, allow_pickle=False, mmap_mode="r")
        labels = np.load(labels_path, allow_pickle=False)
        valid_lengths = np.load(lengths_path, allow_pickle=False)
        valid_mask = np.load(mask_path, allow_pickle=False)

        if padded.ndim != 3 or padded.shape[1:] != (
            EXPECTED_PADDED_LENGTH,
            TOTAL_CHANNEL_COUNT,
        ):
            raise ValueError(
                f"Unexpected padded shape {padded.shape} in {padded_path}; "
                f"expected (S, {EXPECTED_PADDED_LENGTH}, {TOTAL_CHANNEL_COUNT})"
            )
        if labels.shape != (len(padded),):
            raise ValueError(f"Label count mismatch in {directory}")
        if valid_lengths.shape != (len(padded),):
            raise ValueError(f"Valid-length count mismatch in {directory}")
        if valid_mask.shape != padded.shape[:2]:
            raise ValueError(f"Valid-mask shape mismatch in {directory}")
        if np.any(valid_lengths <= 0) or np.any(valid_lengths > EXPECTED_PADDED_LENGTH):
            raise ValueError(f"Invalid valid lengths in {directory}")

        package_index = len(packages)
        packages.append(
            Package(
                user=user,
                action=action,
                directory=directory,
                padded_spike_imu=padded,
                labels=labels,
                valid_lengths=valid_lengths,
                valid_mask=valid_mask,
            )
        )
        for segment_index, label_value in enumerate(labels.astype(str)):
            label = str(label_value)
            if label not in keep:
                continue
            rows.append(
                {
                    "package_index": package_index,
                    "segment_index": segment_index,
                    "user": user,
                    "action": action,
                    "label": label,
                    "valid_length": int(valid_lengths[segment_index]),
                    "sample_id": f"{user}/action_{action}/{segment_index}",
                }
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("The 60-event Angular66 cohort contains no requested labels")
    labels = tuple(sorted(manifest.label.unique().tolist()))
    expected_labels = tuple(sorted(INCLUDED_LABELS))
    if labels != expected_labels:
        raise ValueError(f"Expected labels {expected_labels}, got {labels}")
    if manifest.user.nunique() != N_TRAIN_USERS + N_VAL_USERS + N_TEST_USERS:
        raise ValueError(f"Expected 20 users, got {manifest.user.nunique()}")
    if manifest.sample_id.duplicated().any():
        raise ValueError("Duplicate sample identities detected")
    if int(manifest.valid_length.min()) < max(RELATIVE_N_BINS):
        raise ValueError(
            "At least one gesture is shorter than the maximum relative-bin count"
        )

    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    manifest["label_idx"] = manifest.label.map(class_to_idx).astype(int)
    return Cohort(
        manifest=manifest,
        packages=packages,
        fs=EXPECTED_SAMPLING_RATE_HZ,
        padded_length=EXPECTED_PADDED_LENGTH,
        labels=labels,
        dataset_root=combination_root,
    )


def make_user_split(manifest: pd.DataFrame, split_seed: int) -> dict[str, pd.DataFrame]:
    if split_seed not in SPLIT_SEEDS:
        raise ValueError(f"Unknown split seed: {split_seed}")
    users = sorted(manifest.user.unique().tolist())
    expected_users = N_TRAIN_USERS + N_VAL_USERS + N_TEST_USERS
    if len(users) != expected_users:
        raise ValueError(f"Expected {expected_users} users, got {len(users)}")
    rng = np.random.default_rng(derive_seed(split_seed, "user_split"))
    perm = np.asarray(users, dtype=object)
    rng.shuffle(perm)
    train_users = set(perm[:N_TRAIN_USERS].tolist())
    val_users = set(perm[N_TRAIN_USERS : N_TRAIN_USERS + N_VAL_USERS].tolist())
    test_users = set(perm[-N_TEST_USERS:].tolist())
    if train_users & val_users or train_users & test_users or val_users & test_users:
        raise RuntimeError("User-disjoint split construction failed")
    return {
        "train": manifest[manifest.user.isin(train_users)].reset_index(drop=True),
        "val": manifest[manifest.user.isin(val_users)].reset_index(drop=True),
        "test": manifest[manifest.user.isin(test_users)].reset_index(drop=True),
    }


def _split_users(parts: Mapping[str, pd.DataFrame]) -> dict[str, list[str]]:
    return {
        f"{name}_users": sorted(frame.user.unique().tolist())
        for name, frame in parts.items()
    }


def _sample_hash(parts: Mapping[str, pd.DataFrame]) -> str:
    payload: list[str] = []
    for split_name in ("train", "val", "test"):
        payload.extend(
            f"{split_name}:{sample_id}"
            for sample_id in parts[split_name].sample_id.tolist()
        )
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return frame.label_idx.to_numpy(dtype=np.int64, copy=True)


def _sample_events(cohort: Cohort, row: object) -> np.ndarray:
    package = cohort.packages[int(getattr(row, "package_index"))]
    segment_index = int(getattr(row, "segment_index"))
    valid_length = min(int(getattr(row, "valid_length")), cohort.padded_length)
    events = np.asarray(
        package.padded_spike_imu[segment_index, :valid_length, :EVENT_CHANNEL_COUNT],
        dtype=np.float32,
    )
    if events.ndim != 2 or events.shape[1] != EVENT_CHANNEL_COUNT:
        raise ValueError("Unexpected event matrix shape")
    if not np.isfinite(events).all() or np.any(events < 0):
        raise ValueError("60-channel event matrix must be finite and non-negative")
    return events


def fixed_duration_features(
    cohort: Cohort,
    frame: pd.DataFrame,
    requested_ms: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    samples_per_bin = int(np.rint(float(requested_ms) * cohort.fs / 1000.0))
    if samples_per_bin <= 0:
        raise ValueError(f"Invalid fixed duration {requested_ms} ms")
    n_bins = int(np.ceil(cohort.padded_length / samples_per_bin))
    padded_target = n_bins * samples_per_bin
    feature_rows: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        values = np.zeros(
            (padded_target, EVENT_CHANNEL_COUNT),
            dtype=np.float32,
        )
        events = _sample_events(cohort, row)
        values[: len(events)] = events
        counts = values.reshape(n_bins, samples_per_bin, EVENT_CHANNEL_COUNT).sum(axis=1)
        feature_rows.append(counts.reshape(-1))
    features = np.stack(feature_rows).astype(np.float64)
    meta = {
        "representation_family": "fixed_duration",
        "condition": f"fixed_{int(round(float(requested_ms))):04d}ms",
        "requested_duration_ms": float(requested_ms),
        "samples_per_bin": int(samples_per_bin),
        "actual_duration_ms": float(samples_per_bin * 1000.0 / cohort.fs),
        "n_bins": int(n_bins),
        "feature_dim": int(features.shape[1]),
    }
    return features, _labels(frame), meta


def relative_progress_features(
    cohort: Cohort,
    frame: pd.DataFrame,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    feature_rows: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        events = _sample_events(cohort, row)
        if n_bins > len(events):
            raise ValueError(
                f"Relative {n_bins}-bin representation exceeds valid length {len(events)}"
            )
        chunks = np.array_split(events, n_bins, axis=0)
        counts = np.stack([chunk.sum(axis=0) for chunk in chunks], axis=0)
        feature_rows.append(counts.reshape(-1))
    features = np.stack(feature_rows).astype(np.float64)
    meta = {
        "representation_family": "relative_progress",
        "condition": f"relative_{int(n_bins):02d}bin",
        "requested_duration_ms": None,
        "samples_per_bin": None,
        "actual_duration_ms": None,
        "n_bins": int(n_bins),
        "feature_dim": int(features.shape[1]),
    }
    return features, _labels(frame), meta


def build_features(
    cohort: Cohort,
    frame: pd.DataFrame,
    spec: RunSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if spec.representation_family == "fixed_duration":
        return fixed_duration_features(cohort, frame, float(spec.value))
    if spec.representation_family == "relative_progress":
        return relative_progress_features(cohort, frame, int(spec.value))
    raise ValueError(f"Unknown representation family: {spec.representation_family}")


def fit_standardizer(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_x, dtype=np.float64)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < STD_EPS] = 1.0
    return mean, std


def apply_standardizer(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - mean[None, :]) / std[None, :]


def train_models(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    linear = LogisticRegression(
        solver="lbfgs",
        max_iter=LOGREG_MAX_ITER,
        random_state=seed,
    )
    linear.fit(train_x, train_y)
    knn = KNeighborsClassifier(n_neighbors=KNN_K)
    knn.fit(train_x, train_y)
    return {"linear": linear, "5nn": knn}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_models(
    models: Mapping[str, object],
    splits: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    spec: RunSpec,
    meta: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for classifier in CLASSIFIERS:
        model = models[classifier]
        for split_name in ("val", "test"):
            x, y = splits[split_name]
            pred = model.predict(x)
            rows.append(
                {
                    "experiment": EXPERIMENT_ID,
                    "protocol_version": PROTOCOL_VERSION,
                    "split_seed": spec.split_seed,
                    "representation_family": meta["representation_family"],
                    "condition": meta["condition"],
                    "requested_duration_ms": meta["requested_duration_ms"],
                    "samples_per_bin": meta["samples_per_bin"],
                    "actual_duration_ms": meta["actual_duration_ms"],
                    "n_bins": meta["n_bins"],
                    "feature_dim": meta["feature_dim"],
                    "classifier": classifier,
                    "eval_split": split_name,
                    **classification_metrics(y, pred),
                }
            )
    return rows


def _artifact_path(root: Path, spec: RunSpec) -> Path:
    return root / "runs" / f"{spec.key}.json"


def run_one(
    spec: RunSpec,
    cohort: Cohort,
    root: Path,
    *,
    force: bool = False,
) -> dict[str, object]:
    output_path = _artifact_path(root, spec)
    if output_path.exists() and not force:
        payload = _load_json(output_path)
        identity = (
            payload.get("protocol_version"),
            payload.get("run_key"),
            int(payload.get("split_seed", -1)),
        )
        expected = (PROTOCOL_VERSION, spec.key, spec.split_seed)
        if identity != expected:
            raise ValueError(f"Cached run identity mismatch: {identity} != {expected}")
        return payload

    parts = make_user_split(cohort.manifest, spec.split_seed)
    built: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {
        split_name: build_features(cohort, frame, spec)
        for split_name, frame in parts.items()
    }
    train_x, train_y, train_meta = built["train"]
    val_x, val_y, val_meta = built["val"]
    test_x, test_y, test_meta = built["test"]
    if train_meta != val_meta or train_meta != test_meta:
        raise RuntimeError("Representation metadata differs across splits")

    mean, std = fit_standardizer(train_x)
    train_z = apply_standardizer(train_x, mean, std)
    val_z = apply_standardizer(val_x, mean, std)
    test_z = apply_standardizer(test_x, mean, std)
    model_seed = derive_seed(spec.split_seed, spec.condition, "classifier")
    models = train_models(train_z, train_y, seed=model_seed)
    result_rows = evaluate_models(
        models,
        {
            "val": (val_z, val_y),
            "test": (test_z, test_y),
        },
        spec=spec,
        meta=train_meta,
    )

    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "run_key": spec.key,
        "split_seed": spec.split_seed,
        "representation_family": spec.representation_family,
        "condition": spec.condition,
        "value": spec.value,
        "dataset_root": str(cohort.dataset_root.resolve()),
        "sample_count": int(len(cohort.manifest)),
        "user_count": int(cohort.manifest.user.nunique()),
        "sampling_rate_hz": float(cohort.fs),
        "event_channel_count": EVENT_CHANNEL_COUNT,
        "total_channel_count": TOTAL_CHANNEL_COUNT,
        "event_channel_slice": [0, EVENT_CHANNEL_COUNT],
        "linear_acceleration_event_slice": [0, 30],
        "angular_acceleration_event_slice": [30, 60],
        "raw_imu_excluded": True,
        "sample_hash": _sample_hash(parts),
        **_split_users(parts),
        "representation": dict(train_meta),
        "classifiers": list(CLASSIFIERS),
        "results": result_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    payloads: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for spec in run_specs():
        path = _artifact_path(root, spec)
        if not path.is_file():
            raise FileNotFoundError(f"Missing run artifact: {path}")
        payload = _load_json(path)
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"Protocol mismatch in {path}")
        if payload.get("run_key") != spec.key:
            raise ValueError(f"Run identity mismatch in {path}")
        if int(payload.get("split_seed", -1)) != spec.split_seed:
            raise ValueError(f"Split-seed mismatch in {path}")
        payloads.append(payload)
        result_rows.extend(payload["results"])
        for split_name in ("train", "val", "test"):
            for user in payload[f"{split_name}_users"]:
                split_rows.append(
                    {
                        "split_seed": spec.split_seed,
                        "condition": spec.condition,
                        "split": split_name,
                        "user": user,
                    }
                )

    expected_runs = len(run_specs())
    if len(payloads) != expected_runs:
        raise ValueError(f"Expected {expected_runs} runs, got {len(payloads)}")
    expected_result_rows = expected_runs * len(CLASSIFIERS) * 2
    if len(result_rows) != expected_result_rows:
        raise ValueError(
            f"Expected {expected_result_rows} result rows, got {len(result_rows)}"
        )

    results = pd.DataFrame(result_rows).sort_values(
        ["representation_family", "condition", "split_seed", "classifier", "eval_split"]
    ).reset_index(drop=True)
    test = results[results.eval_split == "test"].copy()
    group_columns = [
        "representation_family",
        "condition",
        "requested_duration_ms",
        "samples_per_bin",
        "actual_duration_ms",
        "n_bins",
        "feature_dim",
        "classifier",
    ]
    summary = (
        test.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            n_splits=("split_seed", "nunique"),
            mean_test_balanced_accuracy=("balanced_accuracy", "mean"),
            sd_test_balanced_accuracy=("balanced_accuracy", "std"),
            mean_test_accuracy=("accuracy", "mean"),
            sd_test_accuracy=("accuracy", "std"),
            mean_test_macro_f1=("macro_f1", "mean"),
            sd_test_macro_f1=("macro_f1", "std"),
        )
        .sort_values(["representation_family", "classifier", "n_bins"])
        .reset_index(drop=True)
    )
    if not (summary.n_splits == len(SPLIT_SEEDS)).all():
        raise ValueError("At least one finalized condition is missing split seeds")

    split_assignments = pd.DataFrame(split_rows).drop_duplicates().sort_values(
        ["split_seed", "condition", "split", "user"]
    )

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results": root / "experiment_0_1_results.csv",
        "summary": root / "experiment_0_1_summary.csv",
        "split_assignments": root / "experiment_0_1_split_assignments.csv",
        "provenance": root / "provenance.json",
    }
    results.to_csv(outputs["results"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    split_assignments.to_csv(outputs["split_assignments"], index=False)

    cohort_payload = payloads[0]
    outputs["provenance"].write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "dataset_root": cohort_payload["dataset_root"],
                "sample_count": cohort_payload["sample_count"],
                "user_count": cohort_payload["user_count"],
                "sampling_rate_hz": EXPECTED_SAMPLING_RATE_HZ,
                "feature_schema": EXPECTED_FEATURE_SCHEMA,
                "event_feature_schema": EXPECTED_EVENT_FEATURE_SCHEMA,
                "event_channel_count": EVENT_CHANNEL_COUNT,
                "total_channel_count": TOTAL_CHANNEL_COUNT,
                "event_channel_slice": [0, 60],
                "linear_acceleration_event_slice": [0, 30],
                "angular_acceleration_event_slice": [30, 60],
                "raw_imu_excluded": True,
                "split_seeds": list(SPLIT_SEEDS),
                "fixed_duration_ms": list(FIXED_DURATION_MS),
                "fixed_duration_rule": "start at 50 ms, increment by 100 ms, include values <= 1050 ms",
                "relative_n_bins": list(RELATIVE_N_BINS),
                "classifiers": list(CLASSIFIERS),
                "standardization": "per-feature z-score fit on training users only",
                "fixed_binning": "channel-wise weighted event sums in fixed physical-duration bins over the common 256-sample padded timeline",
                "relative_binning": "channel-wise weighted event sums after np.array_split of each gesture valid prefix",
                "expected_runs": expected_runs,
                "slurm_unit": "one split-seed x representation-condition per CPU task; Linear and 5-NN share the same standardized features inside the task",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "WithGyro Experiment 0.1: 60-event fixed-duration and relative-progress "
            "temporal representation probe"
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    sub.add_parser("describe")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = find_repo_root()
    if args.command == "describe":
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "protocol_version": PROTOCOL_VERSION,
                    "run_count": len(run_specs()),
                    "split_seeds": list(SPLIT_SEEDS),
                    "fixed_duration_ms": list(FIXED_DURATION_MS),
                    "relative_n_bins": list(RELATIVE_N_BINS),
                    "event_channel_count": EVENT_CHANNEL_COUNT,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "finalize":
        outputs = finalize_experiment(repo_root)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return

    specs = run_specs()
    task_id = int(args.array_task_id)
    if not 0 <= task_id < len(specs):
        raise SystemExit(f"array-task-id must be in [0, {len(specs) - 1}], got {task_id}")
    cohort = load_cohort(repo_root)
    payload = run_one(
        specs[task_id],
        cohort,
        results_dir(repo_root),
        force=bool(args.force),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
