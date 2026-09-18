from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from snn.accel_reconstruction_eval.datasets import load_acceleration_data
from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_2_6_output_readout_loss_shaping as exp726
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_7_3_5_hidden_state_information_loss as exp735
from scripts import experiment_9_0_within_user_generalization as exp90


EXPERIMENT_ID = "experiment_10_0_airborne_motion_ablation"
PROTOCOL_VERSION = "a2_cross_user_5fold_v1"

VARIANT_ORIGINAL = "original"
VARIANT_POSTENCODE = "postencode_mask"
VARIANT_REENCODE = "masked_accel_reencode"
VARIANTS = (VARIANT_ORIGINAL, VARIANT_POSTENCODE, VARIANT_REENCODE)

ROTATIONS = tuple(range(5))
MODEL_SEEDS = (11, 23, 37)
EXPECTED_RUNS = len(VARIANTS) * len(ROTATIONS) * len(MODEL_SEEDS)

ARCHITECTURE = exp73.ARCHITECTURE
ARCHITECTURE_SHIFTS = exp73.SHIFTS
HIDDEN_WIDTH = exp73.HIDDEN_WIDTH
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
BATCH_SIZE = exp72.BATCH_SIZE
SPLITS = ("train", "val", "test")

HIDDEN_LAYERS = ("l1", "l2")
HIDDEN_STATES = ("syn_current", "pre_reset", "spike", "post_reset")
STATE_AGGREGATIONS = ("whole_mean", "fixed250_ordered_mean")
COUNT_AGGREGATIONS = ("whole_count", "fixed250_count")
C_GRID = tuple(float(v) for v in exp735.C_GRID)
PROBE_MAX_ITER = exp735.MAX_ITER

if not np.isclose(float(exp726.THRESHOLD), float(exp73.THRESHOLD)):
    raise RuntimeError("Exp10.0 requires Exp7.2.6 and Exp7.3 threshold contracts to match")


@dataclass(frozen=True)
class RunSpec:
    variant: str
    rotation: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.variant}__rotation{self.rotation}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    threads: int = 1
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp90.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(variant, rotation, seed)
        for variant in VARIANTS
        for rotation in ROTATIONS
        for seed in MODEL_SEEDS
    ]


def probe_names() -> tuple[str, ...]:
    names = [
        "input__events__whole_count",
        "input__events__fixed250_count",
    ]
    for layer in HIDDEN_LAYERS:
        for state in HIDDEN_STATES:
            for aggregation in STATE_AGGREGATIONS:
                names.append(f"{layer}__{state}__{aggregation}")
        for aggregation in COUNT_AGGREGATIONS:
            names.append(f"{layer}__spike__{aggregation}")
    if len(names) != 22 or len(set(names)) != len(names):
        raise RuntimeError(f"Unexpected Exp10.0 probe inventory: {len(names)}")
    return tuple(names)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _sha_rows(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _variant_roots(repo_root: Path, variant: str) -> list[Path]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    base = {
        0: repo_root
        / "outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass",
        1: repo_root
        / "outputs/action1_wavelets_0e5_1_2_4_8_sr_64/low-pass",
    }
    if variant == VARIANT_ORIGINAL:
        return [
            base[action] / "aligned-board-events/segmentation_padded"
            for action in (0, 1)
        ]
    return [
        base[action]
        / "aligned-board-events_writing_motion_ablation"
        / variant
        / "segmentation_padded"
        for action in (0, 1)
    ]


def _validate_loaded(loaded: Any, roots: list[Path]) -> None:
    fs_values = {float(meta.sampling_rate_hz) for meta in loaded.producer_metadatas}
    if fs_values != {exp3.EXPECTED_FS}:
        raise ValueError(
            f"Exp10.0 requires exactly {exp3.EXPECTED_FS:g} Hz, got {fs_values}"
        )
    for root, metadata in zip(roots, loaded.producer_metadatas, strict=True):
        raw = metadata.raw
        if raw.get("event_representation") != "unsigned":
            raise ValueError(f"{root}: expected unsigned events")
        if raw.get("event_feature_schema") != "custom_wavelet_polarity_split_abs_events_v1":
            raise ValueError(f"{root}: unexpected event feature schema")
        if raw.get("event_channel_count") != exp3.EVENT_CHANNELS:
            raise ValueError(f"{root}: expected {exp3.EVENT_CHANNELS} event channels")
        if metadata.channel_count != exp3.TOTAL_CHANNELS:
            raise ValueError(f"{root}: expected {exp3.TOTAL_CHANNELS} total channels")


def _load_variant_manifest(
    repo_root: Path,
    variant: str,
) -> tuple[Any, pd.DataFrame, tuple[str, ...]]:
    roots = _variant_roots(repo_root, variant)
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Exp10.0 {variant} dataset roots are missing:\n" + "\n".join(missing)
        )
    loaded = load_acceleration_data(
        roots,
        repository_root=repo_root,
        require_reconstruction=False,
    )
    _validate_loaded(loaded, roots)

    rows: list[dict[str, Any]] = []
    keep = set(exp3.LABELS)
    for pi, package in enumerate(loaded.packages):
        action = int(package.action)
        source_trial = f"{package.user}/action_{action}/{package.stem}"
        for si, label_value in enumerate(package.labels.astype(str)):
            label = str(label_value)
            if label not in keep:
                continue
            rows.append(
                {
                    "pi": int(pi),
                    "si": int(si),
                    "user": str(package.user),
                    "action": action,
                    "label": label,
                    "valid": int(package.valid_lengths[si]),
                    "pad": int(package.padded_spike_imu.shape[1]),
                    "source_trial": source_trial,
                    "sample_id": f"{source_trial}/segment_{si}",
                }
            )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError(f"Exp10.0 {variant} found no eligible 12-class samples")
    if manifest.sample_id.duplicated().any():
        duplicate = manifest.loc[manifest.sample_id.duplicated(), "sample_id"].iloc[0]
        raise RuntimeError(f"Duplicate sample_id in {variant}: {duplicate}")
    labels = tuple(sorted(manifest.label.unique().tolist()))
    expected_labels = tuple(sorted(set(exp3.LABELS)))
    if labels != expected_labels:
        raise RuntimeError(
            f"{variant}: expected labels {expected_labels}, got {labels}"
        )
    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    manifest["y"] = manifest.label.map(class_to_idx).astype(int)
    return loaded, manifest, labels


_GEOMETRY_COLUMNS = (
    "sample_id",
    "user",
    "action",
    "label",
    "valid",
    "pad",
    "source_trial",
)


def _geometry_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.loc[:, list(_GEOMETRY_COLUMNS)]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )


def _assert_paired_geometry(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    reference_name: str,
    candidate_name: str,
) -> None:
    left = _geometry_frame(reference)
    right = _geometry_frame(candidate)
    if len(left) != len(right):
        raise RuntimeError(
            f"{reference_name}/{candidate_name}: sample count mismatch "
            f"{len(left)} != {len(right)}"
        )
    for column in _GEOMETRY_COLUMNS:
        if not left[column].equals(right[column]):
            mismatch = np.flatnonzero(
                left[column].astype(str).to_numpy()
                != right[column].astype(str).to_numpy()
            )
            index = int(mismatch[0]) if len(mismatch) else -1
            raise RuntimeError(
                f"{reference_name}/{candidate_name}: geometry mismatch in "
                f"{column} at sorted row {index}"
            )


def _fold_assignment_path(config: Config) -> Path:
    return config.results_dir / "split" / "cross_user_fold_assignment.csv"


def _user_fold_path(config: Config) -> Path:
    return config.results_dir / "split" / "cross_user_fold_users.csv"


def _rotation_manifest_path(config: Config, rotation: int) -> Path:
    return config.results_dir / "split" / f"rotation{rotation}.csv"


def prepare_all(config: Config) -> dict[str, Any]:
    loaded_by_variant: dict[str, Any] = {}
    manifests: dict[str, pd.DataFrame] = {}
    labels_by_variant: dict[str, tuple[str, ...]] = {}
    for variant in VARIANTS:
        loaded, manifest, labels = _load_variant_manifest(config.repo_root, variant)
        loaded_by_variant[variant] = loaded
        manifests[variant] = manifest
        labels_by_variant[variant] = labels
    del loaded_by_variant

    reference = manifests[VARIANT_ORIGINAL]
    for variant in VARIANTS[1:]:
        _assert_paired_geometry(
            reference,
            manifests[variant],
            reference_name=VARIANT_ORIGINAL,
            candidate_name=variant,
        )
        if labels_by_variant[variant] != labels_by_variant[VARIANT_ORIGINAL]:
            raise RuntimeError(f"{variant}: class mapping mismatch")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    split_root = config.results_dir / "split"
    split_root.mkdir(parents=True, exist_ok=True)

    canonical = reference.drop(columns=["pi", "si"]).sort_values("sample_id")
    canonical.to_csv(config.results_dir / "canonical_sample_manifest.csv", index=False)

    assignment = exp90._assign_cross_user_folds(reference)
    assignment.to_csv(_fold_assignment_path(config), index=False)
    (
        assignment[["user", "cv_fold"]]
        .drop_duplicates()
        .sort_values(["cv_fold", "user"])
        .to_csv(_user_fold_path(config), index=False)
    )

    rotation_rows: list[dict[str, Any]] = []
    for rotation in ROTATIONS:
        split_manifest = exp90._apply_rotation(assignment, rotation)
        split_manifest.drop(columns=["pi", "si"]).to_csv(
            _rotation_manifest_path(config, rotation),
            index=False,
        )
        counts = split_manifest.split.value_counts()
        rotation_rows.append(
            {
                "rotation": rotation,
                "test_fold": rotation,
                "val_fold": (rotation + 1) % len(ROTATIONS),
                "train_samples": int(counts.get("train", 0)),
                "val_samples": int(counts.get("val", 0)),
                "test_samples": int(counts.get("test", 0)),
                "train_users": int(
                    split_manifest.loc[split_manifest.split == "train", "user"].nunique()
                ),
                "val_users": int(
                    split_manifest.loc[split_manifest.split == "val", "user"].nunique()
                ),
                "test_users": int(
                    split_manifest.loc[split_manifest.split == "test", "user"].nunique()
                ),
            }
        )
    pd.DataFrame(rotation_rows).to_csv(
        config.results_dir / "rotation_summary.csv", index=False
    )

    roots = {
        variant: [str(path.resolve()) for path in _variant_roots(config.repo_root, variant)]
        for variant in VARIANTS
    }
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Measure explicit airborne-period evidence and airborne-induced encoder-context "
            "effects under a fixed Exp7.3 A2 classifier."
        ),
        "variants": list(VARIANTS),
        "variant_definitions": {
            VARIANT_ORIGINAL: "D0 = Encoder(a)",
            VARIANT_POSTENCODE: "D1 = m * Encoder(a)",
            VARIANT_REENCODE: "D2 = m * Encoder(m * a)",
        },
        "dataset_roots": roots,
        "combined_actions": [0, 1],
        "total_samples": int(len(reference)),
        "total_users": int(reference.user.nunique()),
        "labels": list(labels_by_variant[VARIANT_ORIGINAL]),
        "rotations": list(ROTATIONS),
        "model_seeds": list(MODEL_SEEDS),
        "expected_runs": EXPECTED_RUNS,
        "cross_user_split_unit": "user",
        "split_protocol": "Exp9.0 cross-user 5-fold assignment and rotations",
        "architecture": ARCHITECTURE,
        "architecture_shifts": [list(v) for v in ARCHITECTURE_SHIFTS],
        "training_contract": "Exp7.3 A2: end-to-end Linear/WCCE, task-only, no bias",
        "model_init_seed_contract": "exp73._e2e_pair_seed(seed, 'model_init'); variant excluded",
        "loader_seed_contract": "exp73._raw_loaders(data, seed, ...); variant excluded",
        "probe_count_per_run": len(probe_names()),
        "probe_names": list(probe_names()),
        "probe_C_grid": list(C_GRID),
    }
    _save_json(config.results_dir / "audit.json", audit)
    return audit


def _load_saved_assignment(config: Config, manifest: pd.DataFrame) -> pd.DataFrame:
    path = _fold_assignment_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Exp10.0 split assignment {path}; run the prepare stage first"
        )
    saved = pd.read_csv(path, usecols=["sample_id", "cv_fold"])
    if saved.sample_id.duplicated().any():
        raise RuntimeError(f"Duplicate sample_id in {path}")
    merged = manifest.merge(saved, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(manifest):
        raise RuntimeError(
            f"Saved fold assignment does not match current dataset: "
            f"{len(merged)} != {len(manifest)}"
        )
    merged["cv_fold"] = merged.cv_fold.astype(int)
    return merged


def _prepare_run_data(
    config: Config,
    variant: str,
    rotation: int,
) -> tuple[exp3.Data, dict[str, pd.DataFrame], pd.DataFrame]:
    loaded, manifest, labels = _load_variant_manifest(config.repo_root, variant)
    assignment = _load_saved_assignment(config, manifest)
    split_manifest = exp90._apply_rotation(assignment, rotation)
    data, frames = exp90._to_data(loaded, split_manifest, labels)
    return data, frames, split_manifest


def _split_hashes(frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    return {
        split: _sha_rows(frames[split].sample_id.astype(str).tolist())
        for split in SPLITS
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def _subgroup_test_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    actions: np.ndarray,
) -> dict[str, dict[str, float] | None]:
    if len(y_true) != len(actions):
        raise RuntimeError("Prediction/action alignment mismatch")
    result: dict[str, dict[str, float] | None] = {}
    for action in (0, 1):
        mask = np.asarray(actions, dtype=np.int64) == action
        result[f"action{action}"] = (
            _classification_metrics(y_true[mask], y_pred[mask])
            if np.any(mask)
            else None
        )
    return result


def _evaluate_native(
    model: exp73.Exp73Net,
    loader: Iterable,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            loss, scores = exp73._objective_loss_scores(
                trajectory["evidence"],
                lengths,
                y,
                "wcce",
            )
            labels.append(y.cpu().numpy())
            predictions.append(scores.argmax(dim=1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            count += len(y)
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    metrics = _classification_metrics(y_true, y_pred)
    metrics["objective_loss"] = loss_sum / max(count, 1)
    return metrics, y_true, y_pred


def _masked_sum(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = exp3.mask(lengths, values.shape[1]).to(values.dtype).unsqueeze(-1)
    return (values * valid).sum(dim=1)


def _probe_seed(spec: RunSpec, probe_name: str) -> int:
    return int(
        exp3.dseed(
            spec.seed,
            EXPERIMENT_ID,
            "probe",
            spec.rotation,
            probe_name,
        )
    )


def _collect_probe_features_and_lif(
    model: exp73.Exp73Net,
    data: exp3.Data,
    spec: RunSpec,
    config: Config,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, dict[str, float]],
    dict[str, np.ndarray],
]:
    device = torch.device(config.device)
    loaders = exp73._raw_loaders(data, spec.seed, config.batch_size, False)
    names = probe_names()
    feature_parts: dict[str, dict[str, list[np.ndarray]]] = {
        split: {name: [] for name in names} for split in SPLITS
    }
    labels_parts: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    lif_pred_parts: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    W = model.output_linear.weight.detach().cpu().numpy().astype(np.float64)

    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            for X, y, lengths in loaders[split]:
                Xd = X.to(device=device, dtype=torch.float32)
                ld = lengths.to(device=device, dtype=torch.long)
                hidden = exp735._hidden_state_trajectory(model, Xd)

                feature_parts[split]["input__events__whole_count"].append(
                    _masked_sum(Xd, ld).cpu().numpy().astype(np.float32, copy=False)
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
                        for aggregation in STATE_AGGREGATIONS:
                            key = f"{layer}__{state}__{aggregation}"
                            aggregated = exp735._aggregate_tensor(
                                values,
                                ld,
                                aggregation,
                                data.bin_steps,
                            )
                            feature_parts[split][key].append(
                                aggregated.cpu().numpy().astype(np.float32, copy=False)
                            )
                    spikes = hidden[layer]["spike"]
                    feature_parts[split][f"{layer}__spike__whole_count"].append(
                        _masked_sum(spikes, ld)
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    feature_parts[split][f"{layer}__spike__fixed250_count"].append(
                        exp3.fixed_counts(spikes, ld, data.bin_steps)
                        .flatten(start_dim=1)
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )

                l2 = hidden["l2"]["spike"].cpu().numpy().astype(np.uint8, copy=False)
                lengths_np = lengths.numpy().astype(np.int64, copy=False)
                lif = exp726._simulate_unipolar(
                    l2,
                    lengths_np,
                    W,
                    beta=exp73.LIF_BETA,
                    cap=exp73.OUTPUT_CAP,
                    scale=1.0,
                )
                lif_pred_parts[split].append(lif["counts"].argmax(axis=1))
                labels_parts[split].append(y.numpy().astype(np.int64, copy=False))

    features = {
        split: {
            name: np.concatenate(parts, axis=0)
            for name, parts in feature_parts[split].items()
        }
        for split in SPLITS
    }
    labels = {
        split: np.concatenate(labels_parts[split], axis=0) for split in SPLITS
    }
    lif_predictions = {
        split: np.concatenate(lif_pred_parts[split], axis=0) for split in SPLITS
    }
    lif_metrics = {
        split: _classification_metrics(labels[split], lif_predictions[split])
        for split in SPLITS
    }
    return features, labels, lif_metrics, lif_predictions


def _probe_descriptor(name: str) -> tuple[str, str, str]:
    source, state, aggregation = name.split("__", 2)
    return source, state, aggregation


def _fit_all_probes(
    spec: RunSpec,
    features: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    test_actions: np.ndarray,
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
        candidate_rows: list[dict[str, float]] = []
        for C in C_GRID:
            classifier = LogisticRegression(
                C=C,
                max_iter=PROBE_MAX_ITER,
                solver="lbfgs",
                random_state=random_state,
                fit_intercept=True,
            ).fit(transformed["train"], labels["train"])
            val_pred = classifier.predict(transformed["val"])
            val_ba = float(balanced_accuracy_score(labels["val"], val_pred))
            candidate_rows.append({"C": float(C), "val_ba": val_ba})
            if best is None or val_ba > best[0] + 1e-12:
                best = (val_ba, float(C), classifier)
        if best is None:
            raise RuntimeError(f"No probe candidate selected for {spec.key}/{name}")

        selected_val_ba, selected_C, classifier = best
        predictions = {
            split: classifier.predict(transformed[split]) for split in SPLITS
        }
        split_metrics = {
            split: _classification_metrics(labels[split], predictions[split])
            for split in SPLITS
        }
        action_metrics = _subgroup_test_metrics(
            labels["test"],
            predictions["test"],
            test_actions,
        )
        source, state, aggregation = _probe_descriptor(name)
        row: dict[str, Any] = {
            "variant": spec.variant,
            "rotation": spec.rotation,
            "seed": spec.seed,
            "probe": name,
            "source": source,
            "state": state,
            "aggregation": aggregation,
            "feature_dim": int(train_x.shape[1]),
            "selected_C": selected_C,
            "selected_val_ba": selected_val_ba,
            "probe_seed": random_state,
            "candidate_validation_json": json.dumps(candidate_rows, sort_keys=True),
        }
        for split in SPLITS:
            for metric, value in split_metrics[split].items():
                row[f"{split}_{metric}"] = value
        for action in (0, 1):
            metrics = action_metrics[f"action{action}"]
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                row[f"test_action{action}_{metric}"] = (
                    float(metrics[metric]) if metrics is not None else np.nan
                )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != len(probe_names()):
        raise RuntimeError(
            f"{spec.key}: expected {len(probe_names())} probes, got {len(frame)}"
        )
    return frame


def _run_artifacts(config: Config, spec: RunSpec) -> dict[str, Path]:
    return {
        "checkpoint": _path(config.results_dir, "checkpoints", spec.key, ".pt"),
        "history": _path(config.results_dir, "histories", spec.key, ".csv"),
        "evaluation": _path(config.results_dir, "evaluations", spec.key, ".json"),
        "probes": _path(config.results_dir, "probe_evaluations", spec.key, ".csv"),
    }


def _checkpoint_improved(
    metrics: Mapping[str, float],
    best_ba: float,
    best_loss: float,
) -> bool:
    return exp73._checkpoint_improved(dict(metrics), best_ba, best_loss)


def run_one(
    spec: RunSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    if spec.variant not in VARIANTS or spec.rotation not in ROTATIONS or spec.seed not in MODEL_SEEDS:
        raise ValueError(spec)
    artifacts = _run_artifacts(config, spec)
    if (
        not force
        and all(path.exists() for path in artifacts.values())
    ):
        return json.loads(artifacts["evaluation"].read_text(encoding="utf-8"))

    data, frames, split_manifest = _prepare_run_data(
        config,
        spec.variant,
        spec.rotation,
    )
    del split_manifest
    split_hashes = _split_hashes(frames)

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model_init_seed = exp73._e2e_pair_seed(spec.seed, "model_init")
    exp3.seed_all(model_init_seed)
    model = exp73.Exp73Net("linear", len(data.labels), data.fs).to(device)
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
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            loss, _ = exp73._objective_loss_scores(
                trajectory["evidence"],
                lengths,
                y,
                "wcce",
            )
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
        if _checkpoint_improved(val_metrics, best_ba, best_loss):
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
        raise RuntimeError(f"No A2 checkpoint selected for {spec.key}")

    artifacts["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "variant": spec.variant,
            "architecture": ARCHITECTURE,
            "architecture_shifts": ARCHITECTURE_SHIFTS,
            "training_method": "Exp7.3 A2_e2e_linear_wcce",
            "readout": "linear",
            "objective": "wcce",
            "regularization": "task_only",
            "bias": False,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "model_init_seed": int(model_init_seed),
            "split_sample_hashes": split_hashes,
            "model_state_dict": best_state,
        },
        artifacts["checkpoint"],
    )
    artifacts["history"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(artifacts["history"], index=False)

    model.load_state_dict(best_state, strict=True)
    native_metrics: dict[str, dict[str, float]] = {}
    native_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        metrics, y_true, y_pred = _evaluate_native(
            model,
            eval_loaders[split],
            device,
        )
        native_metrics[split] = metrics
        native_arrays[split] = (y_true, y_pred)
    test_actions = frames["test"].action.to_numpy(dtype=np.int64, copy=True)
    native_test_by_action = _subgroup_test_metrics(
        native_arrays["test"][0],
        native_arrays["test"][1],
        test_actions,
    )

    features, probe_labels, lif_metrics, lif_predictions = (
        _collect_probe_features_and_lif(model, data, spec, config)
    )
    for split in SPLITS:
        if not np.array_equal(probe_labels[split], native_arrays[split][0]):
            raise RuntimeError(f"{spec.key}/{split}: probe label order mismatch")
    lif_test_by_action = _subgroup_test_metrics(
        probe_labels["test"],
        lif_predictions["test"],
        test_actions,
    )
    probe_frame = _fit_all_probes(
        spec,
        features,
        probe_labels,
        test_actions,
    )
    artifacts["probes"].parent.mkdir(parents=True, exist_ok=True)
    probe_frame.to_csv(artifacts["probes"], index=False)

    feature_shapes = {
        split: {
            name: list(features[split][name].shape)
            for name in probe_names()
        }
        for split in SPLITS
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "architecture": ARCHITECTURE,
            "shifts": [list(v) for v in ARCHITECTURE_SHIFTS],
            "source_method": "Exp7.3 A2_e2e_linear_wcce",
            "readout": "linear",
            "objective": "wcce",
            "regularization": "task_only",
            "bias": False,
            "ce_gain": exp73.CE_GAIN,
            "max_epochs": config.max_epochs,
            "min_epochs": MIN_EPOCHS,
            "patience": PATIENCE,
            "checkpoint_metric": "native validation balanced accuracy",
            "checkpoint_tiebreak": "native validation objective loss",
            "same_w_lif_beta": exp73.LIF_BETA,
            "same_w_lif_threshold": exp73.THRESHOLD,
            "same_w_lif_cap": exp73.OUTPUT_CAP,
        },
        "dataset_roots": [
            str(path.resolve())
            for path in _variant_roots(config.repo_root, spec.variant)
        ],
        "labels": list(data.labels),
        "fs_hz": float(data.fs),
        "timesteps": int(data.T),
        "bin_steps": int(data.bin_steps),
        "n_bins": int(data.n_bins),
        "split": {key: list(value) for key, value in data.split.items()},
        "split_sample_hashes": split_hashes,
        "model_init_seed": int(model_init_seed),
        "loader_seed_contract": "exp73._raw_loaders(data, seed, ...); variant excluded",
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_ba": best_ba,
        "best_val_objective_loss": best_loss,
        "native_metrics": native_metrics,
        "native_test_by_action": native_test_by_action,
        "same_w_lif_metrics": lif_metrics,
        "same_w_lif_test_by_action": lif_test_by_action,
        "probe_count": int(len(probe_frame)),
        "probe_names": list(probe_names()),
        "probe_feature_shapes": feature_shapes,
        "probe_evaluation_csv": str(artifacts["probes"].relative_to(config.repo_root)),
        "checkpoint": str(artifacts["checkpoint"].relative_to(config.repo_root)),
        "history": str(artifacts["history"].relative_to(config.repo_root)),
    }
    _save_json(artifacts["evaluation"], payload)
    return payload


def _action_metric(
    payload: Mapping[str, Any],
    block: str,
    action: int,
    metric: str,
) -> float:
    values = payload[block][f"action{action}"]
    if values is None:
        return float("nan")
    return float(values[metric])


def _run_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    spec = payload["spec"]
    native = payload["native_metrics"]
    lif = payload["same_w_lif_metrics"]
    row: dict[str, Any] = {
        "variant": spec["variant"],
        "rotation": int(spec["rotation"]),
        "seed": int(spec["seed"]),
        "best_epoch": int(payload["best_epoch"]),
        "stopped_epoch": int(payload["stopped_epoch"]),
        "model_init_seed": int(payload["model_init_seed"]),
        "train_sample_hash": payload["split_sample_hashes"]["train"],
        "val_sample_hash": payload["split_sample_hashes"]["val"],
        "test_sample_hash": payload["split_sample_hashes"]["test"],
    }
    for split in SPLITS:
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "objective_loss"):
            row[f"native_{split}_{metric}"] = float(native[split][metric])
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            row[f"lif_{split}_{metric}"] = float(lif[split][metric])
    for action in (0, 1):
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            row[f"native_test_action{action}_{metric}"] = _action_metric(
                payload, "native_test_by_action", action, metric
            )
            row[f"lif_test_action{action}_{metric}"] = _action_metric(
                payload, "same_w_lif_test_by_action", action, metric
            )
    return row


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in out.columns
    ]
    return out


def _paired_delta_rows(
    runs: pd.DataFrame,
    metric_columns: tuple[str, ...],
) -> pd.DataFrame:
    pairs = (
        ("D0_minus_D1", VARIANT_ORIGINAL, VARIANT_POSTENCODE),
        ("D1_minus_D2", VARIANT_POSTENCODE, VARIANT_REENCODE),
        ("D0_minus_D2", VARIANT_ORIGINAL, VARIANT_REENCODE),
    )
    rows: list[dict[str, Any]] = []
    for rotation in ROTATIONS:
        for seed in MODEL_SEEDS:
            cell = runs[(runs.rotation == rotation) & (runs.seed == seed)]
            indexed = cell.set_index("variant")
            if set(indexed.index) != set(VARIANTS):
                raise RuntimeError(
                    f"Missing paired variants for rotation={rotation}, seed={seed}"
                )
            for name, left, right in pairs:
                row: dict[str, Any] = {
                    "contrast": name,
                    "left_variant": left,
                    "right_variant": right,
                    "rotation": rotation,
                    "seed": seed,
                }
                for metric in metric_columns:
                    row[f"{metric}_delta"] = float(
                        indexed.loc[left, metric] - indexed.loc[right, metric]
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _validate_pairing(runs: pd.DataFrame) -> None:
    for rotation in ROTATIONS:
        for seed in MODEL_SEEDS:
            cell = runs[(runs.rotation == rotation) & (runs.seed == seed)]
            if set(cell.variant) != set(VARIANTS):
                raise RuntimeError(
                    f"Incomplete variant cell rotation={rotation}, seed={seed}"
                )
            for field in (
                "model_init_seed",
                "train_sample_hash",
                "val_sample_hash",
                "test_sample_hash",
            ):
                if cell[field].nunique(dropna=False) != 1:
                    raise RuntimeError(
                        f"Pairing mismatch {field} at rotation={rotation}, seed={seed}"
                    )


def _probe_paired_deltas(probes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = (
        ("D0_minus_D1", VARIANT_ORIGINAL, VARIANT_POSTENCODE),
        ("D1_minus_D2", VARIANT_POSTENCODE, VARIANT_REENCODE),
        ("D0_minus_D2", VARIANT_ORIGINAL, VARIANT_REENCODE),
    )
    for rotation in ROTATIONS:
        for seed in MODEL_SEEDS:
            for probe in probe_names():
                cell = probes[
                    (probes.rotation == rotation)
                    & (probes.seed == seed)
                    & (probes.probe == probe)
                ].set_index("variant")
                if set(cell.index) != set(VARIANTS):
                    raise RuntimeError(
                        f"Incomplete probe pairing: rotation={rotation}, seed={seed}, "
                        f"probe={probe}"
                    )
                for name, left, right in pairs:
                    rows.append(
                        {
                            "contrast": name,
                            "left_variant": left,
                            "right_variant": right,
                            "rotation": rotation,
                            "seed": seed,
                            "probe": probe,
                            "test_ba_delta": float(
                                cell.loc[left, "test_balanced_accuracy"]
                                - cell.loc[right, "test_balanced_accuracy"]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    probe_frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for spec in run_specs():
        artifacts = _run_artifacts(config, spec)
        for name, path in artifacts.items():
            if not path.exists():
                missing.append(f"{spec.key}:{name}:{path}")
        if artifacts["evaluation"].exists():
            payloads.append(
                json.loads(artifacts["evaluation"].read_text(encoding="utf-8"))
            )
        if artifacts["probes"].exists():
            frame = pd.read_csv(artifacts["probes"])
            if len(frame) != len(probe_names()):
                raise RuntimeError(
                    f"{spec.key}: expected {len(probe_names())} probes, got {len(frame)}"
                )
            probe_frames.append(frame)
    if missing:
        raise FileNotFoundError(
            f"Exp10.0 is incomplete; missing {len(missing)} artifacts:\n"
            + "\n".join(missing[:20])
        )
    if len(payloads) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} evaluations, got {len(payloads)}"
        )

    runs = pd.DataFrame([_run_row(payload) for payload in payloads]).sort_values(
        ["variant", "rotation", "seed"]
    )
    _validate_pairing(runs)
    runs.to_csv(config.results_dir / "run_metrics.csv", index=False)

    descriptive_metrics = [
        "native_test_balanced_accuracy",
        "lif_test_balanced_accuracy",
        "native_test_action0_balanced_accuracy",
        "native_test_action1_balanced_accuracy",
        "lif_test_action0_balanced_accuracy",
        "lif_test_action1_balanced_accuracy",
    ]
    run_summary = (
        runs.groupby("variant", sort=False)[descriptive_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _flatten_columns(run_summary).to_csv(
        config.results_dir / "variant_run_summary.csv", index=False
    )

    split_level = (
        runs.groupby(["variant", "rotation"], sort=False)[descriptive_metrics]
        .mean()
        .reset_index()
    )
    split_level.to_csv(config.results_dir / "variant_split_level.csv", index=False)
    split_summary = (
        split_level.groupby("variant", sort=False)[descriptive_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _flatten_columns(split_summary).to_csv(
        config.results_dir / "variant_summary.csv", index=False
    )

    delta_metrics = tuple(descriptive_metrics)
    paired = _paired_delta_rows(runs, delta_metrics)
    paired.to_csv(config.results_dir / "paired_variant_deltas.csv", index=False)
    paired_split = (
        paired.groupby(
            ["contrast", "left_variant", "right_variant", "rotation"],
            sort=False,
        )[[f"{metric}_delta" for metric in delta_metrics]]
        .mean()
        .reset_index()
    )
    paired_split.to_csv(
        config.results_dir / "paired_variant_split_deltas.csv", index=False
    )
    paired_summary = (
        paired_split.groupby(
            ["contrast", "left_variant", "right_variant"], sort=False
        )[[f"{metric}_delta" for metric in delta_metrics]]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _flatten_columns(paired_summary).to_csv(
        config.results_dir / "paired_variant_summary.csv", index=False
    )

    probes = pd.concat(probe_frames, ignore_index=True).sort_values(
        ["probe", "variant", "rotation", "seed"]
    )
    expected_probe_rows = EXPECTED_RUNS * len(probe_names())
    if len(probes) != expected_probe_rows:
        raise RuntimeError(
            f"Expected {expected_probe_rows} probe rows, got {len(probes)}"
        )
    probes.to_csv(config.results_dir / "probe_runs.csv", index=False)

    probe_metrics = [
        "test_balanced_accuracy",
        "test_accuracy",
        "test_macro_f1",
        "test_action0_balanced_accuracy",
        "test_action1_balanced_accuracy",
    ]
    probe_split = (
        probes.groupby(["variant", "rotation", "probe"], sort=False)[probe_metrics]
        .mean()
        .reset_index()
    )
    probe_split.to_csv(config.results_dir / "probe_split_level.csv", index=False)
    probe_summary = (
        probe_split.groupby(["variant", "probe"], sort=False)[probe_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _flatten_columns(probe_summary).to_csv(
        config.results_dir / "probe_summary.csv", index=False
    )

    probe_deltas = _probe_paired_deltas(probes)
    probe_deltas.to_csv(
        config.results_dir / "probe_paired_variant_deltas.csv", index=False
    )
    probe_delta_split = (
        probe_deltas.groupby(["contrast", "rotation", "probe"], sort=False)[
            "test_ba_delta"
        ]
        .mean()
        .reset_index()
    )
    probe_delta_split.to_csv(
        config.results_dir / "probe_paired_variant_split_deltas.csv", index=False
    )
    probe_delta_summary = (
        probe_delta_split.groupby(["contrast", "probe"], sort=False)[
            "test_ba_delta"
        ]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    probe_delta_summary.to_csv(
        config.results_dir / "probe_paired_variant_summary.csv", index=False
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS",
        "variants": list(VARIANTS),
        "variant_definitions": {
            VARIANT_ORIGINAL: "D0 = Encoder(a)",
            VARIANT_POSTENCODE: "D1 = m * Encoder(a)",
            VARIANT_REENCODE: "D2 = m * Encoder(m * a)",
        },
        "combined_actions": [0, 1],
        "rotations": list(ROTATIONS),
        "model_seeds": list(MODEL_SEEDS),
        "run_count": int(len(runs)),
        "expected_run_count": EXPECTED_RUNS,
        "probe_count_per_run": len(probe_names()),
        "probe_run_count": int(len(probes)),
        "architecture": ARCHITECTURE,
        "architecture_shifts": [list(v) for v in ARCHITECTURE_SHIFTS],
        "training_method": "Exp7.3 A2_e2e_linear_wcce",
        "primary_metric": "native_test_balanced_accuracy",
        "primary_statistical_unit": (
            "5 cross-user rotations; three seeds are averaged within each rotation"
        ),
        "paired_contrasts": ["D0_minus_D1", "D1_minus_D2", "D0_minus_D2"],
        "interpretation": {
            "D0_minus_D1": "explicit airborne/reposition timestep evidence",
            "D1_minus_D2": (
                "airborne acceleration influence on writing-period representation "
                "through encoder temporal context"
            ),
            "D0_minus_D2": "total effect of removing airborne acceleration from representation",
        },
        "primary_outputs": [
            "run_metrics.csv",
            "variant_split_level.csv",
            "variant_summary.csv",
            "paired_variant_deltas.csv",
            "paired_variant_split_deltas.csv",
            "paired_variant_summary.csv",
            "probe_runs.csv",
            "probe_split_level.csv",
            "probe_summary.csv",
            "probe_paired_variant_deltas.csv",
            "probe_paired_variant_split_deltas.csv",
            "probe_paired_variant_summary.csv",
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
            "Exp10.0 D0/D1/D2 airborne-motion ablation with Exp7.3 A2, "
            "5 cross-user rotations, and hidden-state probes"
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
                    "same_w_lif_test_ba": payload["same_w_lif_metrics"]["test"][
                        "balanced_accuracy"
                    ],
                    "probe_count": payload["probe_count"],
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
