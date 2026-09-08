from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_5_5_non_snn_history_experts as exp55


base = exp55.base
exp543 = exp55.exp543

EXPERIMENT_ID = "experiment_5_5_3_teacher_weight_routing_decomposition"
PROTOCOL_VERSION = "teacher_weight_routing_decomposition_v1"
SEEDS = exp55.SEEDS
WHAT_WIDTH = exp55.WHAT_WIDTH
N_CLASSES = exp55.N_CLASSES
RELATIVE10 = "relative10"
FIXED250 = "fixed250"
FAMILIES = (RELATIVE10, FIXED250)
HARD_TEACHER = "hard_teacher_frozen"
SOFT_TEACHER = "soft_teacher_frozen"
HARD_RANDOM = "hard_random_train"
SOFT_RANDOM = "soft_random_train"
SOFT_TEACHER_INIT = "soft_teacher_init_retrain"
SOFT_LINEAR_REFIT = "soft_linear_refit"
TRAIN_CONDITIONS = (HARD_RANDOM, SOFT_RANDOM, SOFT_TEACHER_INIT)
ALL_CONDITIONS = (
    HARD_TEACHER,
    SOFT_TEACHER,
    HARD_RANDOM,
    SOFT_RANDOM,
    SOFT_TEACHER_INIT,
    SOFT_LINEAR_REFIT,
)
RELATIVE_STATES = 10
SOFT_SIGMA_MULTIPLIER = 1.5
PROBE_C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
EPOCHS = exp55.EPOCHS
PATIENCE = exp55.PATIENCE
BATCH_SIZE = exp55.BATCH_SIZE
LR = exp55.LR
WEIGHT_DECAY = exp55.WEIGHT_DECAY
GRAD_CLIP = exp55.GRAD_CLIP
EQUIVALENCE_ATOL = 1e-6


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    patience: int = PATIENCE
    batch_size: int = BATCH_SIZE
    threads: int = 1


@dataclass(frozen=True)
class TrainSpec:
    family: str
    condition: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.family}__{self.condition}__seed{self.seed}"


@dataclass(frozen=True)
class LinearSpec:
    family: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.family}__{SOFT_LINEAR_REFIT}__seed{self.seed}"


def find_repo_root(start: Path | None = None) -> Path:
    return exp55.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def teacher_npz_path(root: Path, family: str, seed: int) -> Path:
    return root / "teachers" / f"{family}__seed{seed}.npz"


def teacher_json_path(root: Path, family: str, seed: int) -> Path:
    return root / "teacher_metadata" / f"{family}__seed{seed}.json"


def frozen_evaluation_path(root: Path, family: str, seed: int) -> Path:
    return root / "frozen_evaluations" / f"{family}__seed{seed}.json"


def checkpoint_path(root: Path, spec: TrainSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def history_path(root: Path, spec: TrainSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def train_evaluation_path(root: Path, spec: TrainSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def linear_evaluation_path(root: Path, spec: LinearSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def linear_model_path(root: Path, spec: LinearSpec) -> Path:
    return root / "linear_models" / f"{spec.key}.npz"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def train_specs() -> list[TrainSpec]:
    return [
        TrainSpec(family, condition, seed)
        for family in FAMILIES
        for condition in TRAIN_CONDITIONS
        for seed in SEEDS
    ]


def linear_specs() -> list[LinearSpec]:
    return [LinearSpec(family, seed) for family in FAMILIES for seed in SEEDS]


def _validate_family(family: str) -> None:
    if family not in FAMILIES:
        raise ValueError(f"Unknown Exp5.5.3 teacher family: {family}")


def _validate_train_spec(spec: TrainSpec) -> None:
    _validate_family(spec.family)
    if spec.condition not in TRAIN_CONDITIONS or spec.seed not in SEEDS:
        raise ValueError(f"Invalid Exp5.5.3 train spec: {spec}")


def _exp55_config(config: Config) -> exp55.Config:
    return exp55.Config(
        repo_root=config.repo_root,
        results_dir=exp55.results_dir(config.repo_root),
        device=config.device,
        epochs=config.epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _labels_lengths(data: base.Data, split: str) -> tuple[np.ndarray, np.ndarray]:
    return {
        "train": (data.ytr, data.ltr),
        "val": (data.yva, data.lva),
        "test": (data.yte, data.lte),
    }[split]


def _what_for_seed(
    seed: int,
    data: base.Data,
    config: Config,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    source_config = _exp55_config(config)
    exp55.prepare_reference_seed(seed, data, source_config, force=False)
    return exp55._load_what_arrays(seed, data, source_config)


def _relative_features(
    what: np.ndarray,
    lengths: np.ndarray,
    n_states: int = RELATIVE_STATES,
) -> np.ndarray:
    values = np.asarray(what, dtype=np.float64)
    out = np.zeros((len(values), n_states, values.shape[-1]), dtype=np.float64)
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        if length <= 0:
            continue
        positions = np.arange(length, dtype=np.int64)
        bins = np.minimum(n_states - 1, (positions * n_states) // length)
        np.add.at(out[row], bins, values[row, :length])
    return out.reshape(len(values), n_states * values.shape[-1])


def _fixed_features(
    what: np.ndarray,
    lengths: np.ndarray,
    data: base.Data,
) -> tuple[np.ndarray, int, int]:
    features, bin_steps, n_bins = exp543.fixed250_features(what, lengths, data)
    return np.asarray(features, dtype=np.float64), int(bin_steps), int(n_bins)


def _fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    namespace: str,
) -> dict[str, object]:
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_x)
    val_z = scaler.transform(val_x)
    test_z = scaler.transform(test_x)
    best: tuple[float, float, LogisticRegression] | None = None
    for C in PROBE_C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=max(exp543.LOGREG_MAX_ITER, 5000),
            solver="lbfgs",
            random_state=base.dseed(seed, EXPERIMENT_ID, namespace, "linear"),
        )
        classifier.fit(train_z, train_y)
        val_ba = float(balanced_accuracy_score(val_y, classifier.predict(val_z)))
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError("No Exp5.5.3 linear candidate selected")
    val_ba, C, classifier = best
    expected_classes = np.arange(N_CLASSES, dtype=np.int64)
    if not np.array_equal(classifier.classes_, expected_classes):
        raise RuntimeError("Exp5.5.3 class ordering changed")
    offline = {
        "train": classifier.decision_function(train_z),
        "val": classifier.decision_function(val_z),
        "test": classifier.decision_function(test_z),
    }
    labels = {"train": train_y, "val": val_y, "test": test_y}
    metrics = {
        split: exp55._metrics_from_logits(labels[split], np.asarray(logits))
        for split, logits in offline.items()
    }
    effective_weight = classifier.coef_.astype(np.float64) / scaler.scale_[None, :]
    effective_bias = classifier.intercept_.astype(np.float64) - effective_weight @ scaler.mean_.astype(np.float64)
    return {
        "C": C,
        "scaler": scaler,
        "classifier": classifier,
        "offline": offline,
        "metrics": metrics,
        "effective_weight": effective_weight,
        "effective_bias": effective_bias,
    }


def _routing_numpy(
    lengths: np.ndarray,
    steps: int,
    family: str,
    routing: str,
    n_states: int,
    sampling_rate_hz: float,
    bin_steps: int,
) -> np.ndarray:
    _validate_family(family)
    if routing not in ("hard", "soft"):
        raise ValueError(routing)
    q = np.zeros((len(lengths), steps, n_states), dtype=np.float64)
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        if length <= 0:
            continue
        time = np.arange(length, dtype=np.float64)
        if family == RELATIVE10:
            coordinate = time * n_states / float(length)
            centers = np.arange(n_states, dtype=np.float64) + 0.5
            sigma = SOFT_SIGMA_MULTIPLIER
            hard_index = np.minimum(n_states - 1, (np.arange(length) * n_states) // length)
        else:
            coordinate = time / float(sampling_rate_hz)
            bin_seconds = float(bin_steps / sampling_rate_hz)
            centers = (np.arange(n_states, dtype=np.float64) + 0.5) * bin_seconds
            sigma = SOFT_SIGMA_MULTIPLIER * bin_seconds
            hard_index = np.minimum(n_states - 1, np.arange(length) // bin_steps)
        if routing == "hard":
            q[row, np.arange(length), hard_index] = 1.0
        else:
            logits = -0.5 * ((coordinate[:, None] - centers[None, :]) / sigma) ** 2
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            q[row, :length] = probs
    return q


def _routed_features_numpy(
    what: np.ndarray,
    lengths: np.ndarray,
    family: str,
    routing: str,
    n_states: int,
    sampling_rate_hz: float,
    bin_steps: int,
) -> np.ndarray:
    q = _routing_numpy(
        lengths,
        what.shape[1],
        family,
        routing,
        n_states,
        sampling_rate_hz,
        bin_steps,
    )
    return np.einsum(
        "ntk,ntd->nkd",
        q,
        np.asarray(what, dtype=np.float64),
        optimize=True,
    )


def _teacher_logits_numpy(
    what: np.ndarray,
    lengths: np.ndarray,
    weight_bank: np.ndarray,
    bias: np.ndarray,
    family: str,
    routing: str,
    sampling_rate_hz: float,
    bin_steps: int,
) -> np.ndarray:
    features = _routed_features_numpy(
        what,
        lengths,
        family,
        routing,
        int(weight_bank.shape[0]),
        sampling_rate_hz,
        bin_steps,
    )
    return np.einsum("nkd,kcd->nc", features, weight_bank, optimize=True) + bias


def _margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    true = values[np.arange(len(values)), labels]
    masked = values.copy()
    masked[np.arange(len(values)), labels] = -np.inf
    return true - masked.max(axis=1)


def _distortion_metrics(
    hard_logits: np.ndarray,
    soft_logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    delta = np.asarray(soft_logits) - np.asarray(hard_logits)
    hard_pred = np.asarray(hard_logits).argmax(axis=1)
    soft_pred = np.asarray(soft_logits).argmax(axis=1)
    return {
        "mean_logit_l2": float(np.linalg.norm(delta, axis=1).mean()),
        "mean_logit_abs": float(np.abs(delta).mean()),
        "prediction_flip_rate": float(np.mean(hard_pred != soft_pred)),
        "mean_true_margin_change": float((_margin(soft_logits, labels) - _margin(hard_logits, labels)).mean()),
    }


def _teacher_payload(
    family: str,
    seed: int,
    weight_bank: np.ndarray,
    bias: np.ndarray,
    bin_steps: int,
    metrics: dict[str, object],
    equivalence: dict[str, object],
    source_meta: dict[str, object],
    C: float | None,
) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "family": family,
        "seed": seed,
        "split_seed": base.SPLIT_SEED,
        "source_experiment_id": exp55.EXPERIMENT_ID,
        "source_protocol_version": exp55.PROTOCOL_VERSION,
        "source_what_definition": "frozen Local-SNN L2 spikes from the Exp5.4/5.5 fusion cache",
        "source_cache_seed": int(source_meta.get("seed", seed)),
        "n_states": int(weight_bank.shape[0]),
        "what_width": int(weight_bank.shape[-1]),
        "n_classes": int(weight_bank.shape[1]),
        "bin_steps": int(bin_steps),
        "soft_sigma_multiplier": SOFT_SIGMA_MULTIPLIER,
        "linear_C": C,
        "metrics": metrics,
        "equivalence": equivalence,
        "test_used_for_model_selection": False,
    }


def prepare_teacher_seed(
    seed: int,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, Path]:
    if seed not in SEEDS:
        raise ValueError(seed)
    what, source_meta = _what_for_seed(seed, data, config)
    outputs: dict[str, Path] = {}
    for family in FAMILIES:
        npz_path = teacher_npz_path(config.results_dir, family, seed)
        json_path = teacher_json_path(config.results_dir, family, seed)
        frozen_path = frozen_evaluation_path(config.results_dir, family, seed)
        if npz_path.exists() and json_path.exists() and frozen_path.exists() and not force:
            outputs[family] = json_path
            continue

        if family == RELATIVE10:
            features = {
                split: _relative_features(what[split], _labels_lengths(data, split)[1])
                for split in ("train", "val", "test")
            }
            fit = _fit_linear_probe(
                features["train"],
                data.ytr,
                features["val"],
                data.yva,
                features["test"],
                data.yte,
                seed,
                "relative10_teacher",
            )
            weight_bank = np.asarray(fit["effective_weight"], dtype=np.float64).reshape(
                N_CLASSES, RELATIVE_STATES, WHAT_WIDTH
            ).transpose(1, 0, 2)
            bias = np.asarray(fit["effective_bias"], dtype=np.float64)
            bin_steps = 0
            metrics = fit["metrics"]
            offline = fit["offline"]
            C: float | None = float(fit["C"])
            scaler = fit["scaler"]
            classifier = fit["classifier"]
            save_extra = {
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "classifier_coef": classifier.coef_,
                "classifier_intercept": classifier.intercept_,
                "classes": classifier.classes_,
            }
        else:
            source_config = _exp55_config(config)
            source_npz_path = exp55.reference_model_path(source_config.results_dir, seed)
            source_json_path = exp55.reference_path(source_config.results_dir, seed)
            if not source_npz_path.exists() or not source_json_path.exists():
                raise FileNotFoundError("Exp5.5 Fixed250 reference was not prepared")
            source_npz = np.load(source_npz_path)
            source_json = json.loads(source_json_path.read_text(encoding="utf-8"))
            weight_bank = np.asarray(source_npz["weight_bins"], dtype=np.float64)
            bias = np.asarray(source_npz["bias"], dtype=np.float64)
            bin_steps = int(np.asarray(source_npz["bin_steps"]).reshape(-1)[0])
            metrics = source_json["fixed250"]["metrics"]
            offline: dict[str, np.ndarray] = {}
            C = 1.0
            save_extra = {
                "scaler_mean": np.asarray(source_npz["scaler_mean"]),
                "scaler_scale": np.asarray(source_npz["scaler_scale"]),
                "classes": np.asarray(source_npz["classes"]),
            }
            for split in ("train", "val", "test"):
                feature_values, check_steps, check_bins = _fixed_features(
                    what[split], _labels_lengths(data, split)[1], data
                )
                if check_steps != bin_steps or check_bins != weight_bank.shape[0]:
                    raise RuntimeError("Fixed250 teacher layout changed")
                offline[split] = np.einsum(
                    "nkd,kcd->nc",
                    feature_values.reshape(len(feature_values), check_bins, WHAT_WIDTH),
                    weight_bank,
                    optimize=True,
                ) + bias

        equivalence: dict[str, object] = {}
        hard_logits: dict[str, np.ndarray] = {}
        soft_logits: dict[str, np.ndarray] = {}
        labels_by_split = {"train": data.ytr, "val": data.yva, "test": data.yte}
        for split in ("train", "val", "test"):
            _, lengths = _labels_lengths(data, split)
            hard = _teacher_logits_numpy(
                what[split],
                lengths,
                weight_bank,
                bias,
                family,
                "hard",
                float(data.fs),
                bin_steps,
            )
            soft = _teacher_logits_numpy(
                what[split],
                lengths,
                weight_bank,
                bias,
                family,
                "soft",
                float(data.fs),
                bin_steps,
            )
            hard_logits[split] = hard
            soft_logits[split] = soft
            delta = float(np.max(np.abs(np.asarray(offline[split]) - hard)))
            identical = bool(np.array_equal(np.asarray(offline[split]).argmax(1), hard.argmax(1)))
            equivalence[f"{split}_max_abs_offline_vs_hard_streaming"] = delta
            equivalence[f"{split}_predictions_identical"] = identical
            if delta > EQUIVALENCE_ATOL or not identical:
                raise RuntimeError(
                    f"{family} seed {seed} teacher equivalence failed on {split}: {delta:.3e}"
                )

        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            weight_bank=weight_bank,
            bias=bias,
            bin_steps=np.asarray([bin_steps], dtype=np.int64),
            n_states=np.asarray([weight_bank.shape[0]], dtype=np.int64),
            **save_extra,
        )
        metadata = _teacher_payload(
            family,
            seed,
            weight_bank,
            bias,
            bin_steps,
            metrics,
            equivalence,
            source_meta,
            C,
        )
        _save_json(json_path, metadata)
        frozen_metrics = {
            routing: {
                split: exp55._metrics_from_logits(labels_by_split[split], logits_by_split[split])
                for split in ("train", "val", "test")
            }
            for routing, logits_by_split in (("hard", hard_logits), ("soft", soft_logits))
        }
        distortion = {
            split: _distortion_metrics(hard_logits[split], soft_logits[split], labels_by_split[split])
            for split in ("train", "val", "test")
        }
        _save_json(
            frozen_path,
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "family": family,
                "seed": seed,
                "hard_teacher": frozen_metrics["hard"],
                "soft_teacher": frozen_metrics["soft"],
                "hard_to_soft_logit_distortion": distortion,
                "teacher_equivalence": equivalence,
                "test_used_for_model_selection": False,
            },
        )
        outputs[family] = json_path
    return outputs


def load_teacher(
    family: str,
    seed: int,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, object]]:
    _validate_family(family)
    npz_path = teacher_npz_path(config.results_dir, family, seed)
    json_path = teacher_json_path(config.results_dir, family, seed)
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"Missing Exp5.5.3 teacher for {family} seed {seed}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if (
        metadata.get("experiment_id") != EXPERIMENT_ID
        or metadata.get("protocol_version") != PROTOCOL_VERSION
        or metadata.get("family") != family
        or metadata.get("seed") != seed
    ):
        raise ValueError(f"Exp5.5.3 teacher identity mismatch: {json_path}")
    arrays = np.load(npz_path)
    return (
        np.asarray(arrays["weight_bank"], dtype=np.float32),
        np.asarray(arrays["bias"], dtype=np.float32),
        int(np.asarray(arrays["bin_steps"]).reshape(-1)[0]),
        metadata,
    )


class RoutedWeightBank(nn.Module):
    """Fixed temporal routing followed by a trainable phase-conditioned WHAT weight bank."""

    def __init__(
        self,
        family: str,
        routing: str,
        n_states: int,
        bin_steps: int,
        sampling_rate_hz: float,
    ) -> None:
        super().__init__()
        _validate_family(family)
        if routing not in ("hard", "soft"):
            raise ValueError(routing)
        self.family = family
        self.routing = routing
        self.n_states = int(n_states)
        self.bin_steps = int(bin_steps)
        self.sampling_rate_hz = float(sampling_rate_hz)
        if self.family == FIXED250 and self.bin_steps <= 0:
            raise ValueError("Fixed250 routing requires positive bin_steps")
        self.weight_bank = nn.Parameter(torch.empty(self.n_states, N_CLASSES, WHAT_WIDTH))
        self.class_bias = nn.Parameter(torch.zeros(N_CLASSES))

    def reset_random(self) -> None:
        nn.init.xavier_uniform_(self.weight_bank)
        nn.init.zeros_(self.class_bias)

    def initialize_teacher(self, weight_bank: np.ndarray, bias: np.ndarray) -> None:
        if tuple(weight_bank.shape) != tuple(self.weight_bank.shape):
            raise ValueError("Teacher weight-bank shape mismatch")
        with torch.no_grad():
            self.weight_bank.copy_(torch.as_tensor(weight_bank, dtype=self.weight_bank.dtype))
            self.class_bias.copy_(torch.as_tensor(bias, dtype=self.class_bias.dtype))

    def routing_probabilities(
        self,
        lengths: torch.Tensor,
        steps: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch = len(lengths)
        device = lengths.device
        positions = torch.arange(steps, device=device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        if self.family == RELATIVE10:
            denom = lengths.clamp_min(1).to(dtype).unsqueeze(1)
            coordinate = positions.to(dtype).unsqueeze(0) * float(self.n_states) / denom
            centers = torch.arange(self.n_states, device=device, dtype=dtype) + 0.5
            hard_index = torch.div(
                positions.unsqueeze(0) * self.n_states,
                lengths.clamp_min(1).unsqueeze(1),
                rounding_mode="floor",
            ).clamp(max=self.n_states - 1)
            sigma = float(SOFT_SIGMA_MULTIPLIER)
        else:
            coordinate = positions.to(dtype).unsqueeze(0).expand(batch, -1) / self.sampling_rate_hz
            bin_seconds = float(self.bin_steps / self.sampling_rate_hz)
            centers = (torch.arange(self.n_states, device=device, dtype=dtype) + 0.5) * bin_seconds
            hard_index = torch.div(
                positions.unsqueeze(0).expand(batch, -1),
                self.bin_steps,
                rounding_mode="floor",
            ).clamp(max=self.n_states - 1)
            sigma = float(SOFT_SIGMA_MULTIPLIER * bin_seconds)
        if self.routing == "hard":
            q = torch.zeros(batch, steps, self.n_states, device=device, dtype=dtype)
            q.scatter_(2, hard_index.unsqueeze(-1), 1.0)
        else:
            logits = -0.5 * ((coordinate.unsqueeze(-1) - centers.view(1, 1, -1)) / sigma) ** 2
            q = torch.softmax(logits, dim=-1)
        return q * valid.to(dtype).unsqueeze(-1)

    def forward(self, what: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        q = self.routing_probabilities(lengths, what.shape[1], what.dtype)
        bank_evidence = torch.einsum("btd,kcd->btkc", what, self.weight_bank)
        evidence = (q.unsqueeze(-1) * bank_evidence).sum(dim=2)
        return evidence.sum(dim=1) + self.class_bias


def _loader(
    what: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(what, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def _loaders(
    what: dict[str, np.ndarray],
    data: base.Data,
    seed: int,
    config: Config,
    train_shuffle: bool,
    splits: tuple[str, ...],
) -> dict[str, DataLoader]:
    output: dict[str, DataLoader] = {}
    for split in splits:
        labels, lengths = _labels_lengths(data, split)
        output[split] = _loader(
            what[split],
            labels,
            lengths,
            config.batch_size,
            train_shuffle if split == "train" else False,
            base.dseed(seed, EXPERIMENT_ID, "loader", split),
        )
    return output


def _evaluate_model(
    model: RoutedWeightBank,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    labels_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    with torch.no_grad():
        for what, labels, lengths in loader:
            logits = model(what.to(device), lengths.to(device))
            labels_parts.append(labels.numpy())
            logits_parts.append(logits.cpu().numpy())
    return exp55._metrics_from_logits(np.concatenate(labels_parts), np.concatenate(logits_parts))


def _weight_diagnostics(
    model: RoutedWeightBank,
    teacher_weight: np.ndarray,
    teacher_bias: np.ndarray,
) -> dict[str, float]:
    learned = model.weight_bank.detach().cpu().numpy().astype(np.float64)
    learned_bias = model.class_bias.detach().cpu().numpy().astype(np.float64)
    teacher = np.asarray(teacher_weight, dtype=np.float64)
    teacher_bias64 = np.asarray(teacher_bias, dtype=np.float64)
    delta = learned - teacher
    denom = max(float(np.linalg.norm(teacher)), 1e-12)
    a = learned.reshape(-1)
    b = teacher.reshape(-1)
    cosine = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return {
        "weight_frobenius_distance_to_teacher": float(np.linalg.norm(delta)),
        "weight_relative_distance_to_teacher": float(np.linalg.norm(delta) / denom),
        "weight_cosine_to_teacher": cosine,
        "bias_l2_distance_to_teacher": float(np.linalg.norm(learned_bias - teacher_bias64)),
    }


def _initialize_train_model(
    spec: TrainSpec,
    data: base.Data,
    config: Config,
    teacher_weight: np.ndarray,
    teacher_bias: np.ndarray,
    bin_steps: int,
) -> RoutedWeightBank:
    routing = "hard" if spec.condition == HARD_RANDOM else "soft"
    seed = base.dseed(spec.seed, EXPERIMENT_ID, spec.family, "paired_random_init")
    base.seed_all(seed)
    model = RoutedWeightBank(
        spec.family,
        routing,
        int(teacher_weight.shape[0]),
        bin_steps,
        float(data.fs),
    ).to(config.device)
    if spec.condition == SOFT_TEACHER_INIT:
        model.initialize_teacher(teacher_weight, teacher_bias)
    else:
        model.reset_random()
    return model


def run_train_one(
    spec: TrainSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    _validate_train_spec(spec)
    destination = train_evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    weight_bank, bias, bin_steps, _ = load_teacher(spec.family, spec.seed, config)
    what, source_meta = _what_for_seed(spec.seed, data, config)
    train_loaders = _loaders(what, data, spec.seed, config, True, ("train",))
    eval_loaders = _loaders(what, data, spec.seed, config, False, ("train", "val", "test"))
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _initialize_train_model(spec, data, config, weight_bank, bias, bin_steps)

    epoch0_train = _evaluate_model(model, eval_loaders["train"], device)
    epoch0_val = _evaluate_model(model, eval_loaders["val"], device)
    best_ba = float(epoch0_val["balanced_accuracy"])
    best_loss = float(epoch0_val["loss"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int]] = [
        {
            "epoch": 0,
            "train_loss": epoch0_train["loss"],
            "train_balanced_accuracy": epoch0_train["balanced_accuracy"],
            "val_loss": epoch0_val["loss"],
            "val_balanced_accuracy": epoch0_val["balanced_accuracy"],
        }
    ]
    stale = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb, lengths in train_loaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, lengths)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            n = len(yb)
            train_loss_sum += float(loss.item()) * n
            train_count += n
            train_true.append(yb.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())
        val = _evaluate_model(model, eval_loaders["val"], device)
        train_ba = float(balanced_accuracy_score(np.concatenate(train_true), np.concatenate(train_pred)))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_count, 1),
                "train_balanced_accuracy": train_ba,
                "val_loss": val["loss"],
                "val_balanced_accuracy": val["balanced_accuracy"],
            }
        )
        val_ba = float(val["balanced_accuracy"])
        val_loss = float(val["loss"])
        improved = val_ba > best_ba + 1e-12 or (
            abs(val_ba - best_ba) <= 1e-12 and val_loss < best_loss - 1e-12
        )
        if improved:
            best_ba = val_ba
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    model.load_state_dict(best_state)
    metrics = {
        split: _evaluate_model(model, eval_loaders[split], device)
        for split in ("train", "val", "test")
    }
    diagnostics = _weight_diagnostics(model, weight_bank, bias)
    checkpoint = checkpoint_path(config.results_dir, spec)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_ba,
            "best_val_loss": best_loss,
            "state_dict": best_state,
        },
        checkpoint,
    )
    hist = history_path(config.results_dir, spec)
    hist.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hist, index=False)
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "seed": spec.seed,
        "family": spec.family,
        "condition": spec.condition,
        "routing": "hard" if spec.condition == HARD_RANDOM else "soft",
        "initialization": "teacher" if spec.condition == SOFT_TEACHER_INIT else "random",
        "best_epoch": best_epoch,
        "stopped_after_epoch": int(history[-1]["epoch"]),
        "initial_train_balanced_accuracy": epoch0_train["balanced_accuracy"],
        "initial_val_balanced_accuracy": epoch0_val["balanced_accuracy"],
        "train": metrics["train"],
        "val": metrics["val"],
        "test": metrics["test"],
        "weight_diagnostics": diagnostics,
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "source_cache_seed": int(source_meta.get("seed", spec.seed)),
        "selection_policy": "epoch 0 and trained epochs compete by validation balanced accuracy, tie-broken by validation CE",
        "test_used_for_checkpoint_selection": False,
    }
    _save_json(destination, payload)
    return payload


def run_linear_one(
    spec: LinearSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    _validate_family(spec.family)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    destination = linear_evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    teacher_weight, _, bin_steps, _ = load_teacher(spec.family, spec.seed, config)
    what, source_meta = _what_for_seed(spec.seed, data, config)
    n_states = int(teacher_weight.shape[0])
    feature_mats: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        _, lengths = _labels_lengths(data, split)
        feature_mats[split] = _routed_features_numpy(
            what[split],
            lengths,
            spec.family,
            "soft",
            n_states,
            float(data.fs),
            bin_steps,
        ).reshape(len(what[split]), n_states * WHAT_WIDTH)
    fit = _fit_linear_probe(
        feature_mats["train"],
        data.ytr,
        feature_mats["val"],
        data.yva,
        feature_mats["test"],
        data.yte,
        spec.seed,
        f"{spec.family}_soft_linear_refit",
    )
    weight_bank = np.asarray(fit["effective_weight"], dtype=np.float64).reshape(
        N_CLASSES, n_states, WHAT_WIDTH
    ).transpose(1, 0, 2)
    bias = np.asarray(fit["effective_bias"], dtype=np.float64)
    model_path = linear_model_path(config.results_dir, spec)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler = fit["scaler"]
    classifier = fit["classifier"]
    np.savez_compressed(
        model_path,
        weight_bank=weight_bank,
        bias=bias,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classifier_coef=classifier.coef_,
        classifier_intercept=classifier.intercept_,
        classes=classifier.classes_,
        C=np.asarray([fit["C"]], dtype=np.float64),
    )
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "family": spec.family,
        "condition": SOFT_LINEAR_REFIT,
        "seed": spec.seed,
        "routing": "soft",
        "n_states": n_states,
        "linear_C": fit["C"],
        "feature_dim": int(n_states * WHAT_WIDTH),
        "train": fit["metrics"]["train"],
        "val": fit["metrics"]["val"],
        "test": fit["metrics"]["test"],
        "source_cache_seed": int(source_meta.get("seed", spec.seed)),
        "selection_policy": "C selected by validation balanced accuracy only; test is evaluated once after selection",
        "test_used_for_checkpoint_selection": False,
    }
    _save_json(destination, payload)
    return payload


def _sem(values: np.ndarray) -> float:
    return 0.0 if len(values) <= 1 else float(values.std(ddof=1) / math.sqrt(len(values)))


def _run_row(
    family: str,
    condition: str,
    seed: int,
    payload: dict[str, object],
    trainable_parameter_count: int,
) -> dict[str, object]:
    return {
        "family": family,
        "condition": condition,
        "seed": seed,
        "best_epoch": payload.get("best_epoch"),
        "trainable_parameter_count": trainable_parameter_count,
        "val_balanced_accuracy": payload["val"]["balanced_accuracy"],
        "val_accuracy": payload["val"]["accuracy"],
        "val_macro_f1": payload["val"]["macro_f1"],
        "val_loss": payload["val"]["loss"],
        "test_balanced_accuracy": payload["test"]["balanced_accuracy"],
        "test_accuracy": payload["test"]["accuracy"],
        "test_macro_f1": payload["test"]["macro_f1"],
        "test_loss": payload["test"]["loss"],
    }


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    run_rows: list[dict[str, object]] = []
    equivalence_rows: list[dict[str, object]] = []
    distortion_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    per_run: dict[tuple[str, str, int], dict[str, object]] = {}
    n_states_by_family: dict[str, int] = {}

    for family in FAMILIES:
        for seed in SEEDS:
            teacher_path = teacher_json_path(root, family, seed)
            frozen_path = frozen_evaluation_path(root, family, seed)
            if not teacher_path.exists() or not frozen_path.exists():
                raise FileNotFoundError(
                    f"Finalizer will not regenerate missing teacher artifacts for {family} seed {seed}"
                )
            teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            n_states = int(teacher["n_states"])
            previous = n_states_by_family.setdefault(family, n_states)
            if previous != n_states:
                raise ValueError(f"Teacher state count changed across seeds for {family}")
            for split in ("train", "val", "test"):
                equivalence_rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "split": split,
                        "max_abs_offline_vs_hard_streaming": teacher["equivalence"][f"{split}_max_abs_offline_vs_hard_streaming"],
                        "predictions_identical": teacher["equivalence"][f"{split}_predictions_identical"],
                    }
                )
                distortion_rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "split": split,
                        **frozen["hard_to_soft_logit_distortion"][split],
                    }
                )
            for condition, key in ((HARD_TEACHER, "hard_teacher"), (SOFT_TEACHER, "soft_teacher")):
                payload = {
                    "train": frozen[key]["train"],
                    "val": frozen[key]["val"],
                    "test": frozen[key]["test"],
                    "best_epoch": None,
                    "trainable_parameter_count": 0,
                }
                per_run[(family, condition, seed)] = payload
                run_rows.append(_run_row(family, condition, seed, payload, trainable_parameter_count=0))

    for spec in train_specs():
        path = train_evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Finalizer will not retrain missing run: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("spec") != asdict(spec)
            or payload.get("test_used_for_checkpoint_selection") is not False
        ):
            raise ValueError(f"Invalid Exp5.5.3 train artifact: {path}")
        per_run[(spec.family, spec.condition, spec.seed)] = payload
        run_rows.append(
            _run_row(
                spec.family,
                spec.condition,
                spec.seed,
                payload,
                trainable_parameter_count=int(payload["trainable_parameter_count"]),
            )
        )
        weight_rows.append(
            {
                "family": spec.family,
                "condition": spec.condition,
                "seed": spec.seed,
                **payload["weight_diagnostics"],
            }
        )

    for spec in linear_specs():
        path = linear_evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Finalizer will not refit missing linear run: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("family") != spec.family
            or payload.get("seed") != spec.seed
            or payload.get("condition") != SOFT_LINEAR_REFIT
            or payload.get("test_used_for_checkpoint_selection") is not False
        ):
            raise ValueError(f"Invalid Exp5.5.3 linear artifact: {path}")
        per_run[(spec.family, SOFT_LINEAR_REFIT, spec.seed)] = payload
        count = n_states_by_family[spec.family] * N_CLASSES * WHAT_WIDTH + N_CLASSES
        run_rows.append(
            _run_row(
                spec.family,
                SOFT_LINEAR_REFIT,
                spec.seed,
                payload,
                trainable_parameter_count=count,
            )
        )

    runs = pd.DataFrame(run_rows).sort_values(["family", "condition", "seed"])
    summary_rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for condition in ALL_CONDITIONS:
            group = runs[(runs.family == family) & (runs.condition == condition)].sort_values("seed")
            if len(group) != len(SEEDS):
                raise RuntimeError(f"Incomplete Exp5.5.3 summary group: {family}/{condition}")
            test_ba = group.test_balanced_accuracy.to_numpy(dtype=float)
            val_ba = group.val_balanced_accuracy.to_numpy(dtype=float)
            summary_rows.append(
                {
                    "family": family,
                    "condition": condition,
                    "n_runs": int(len(group)),
                    "mean_val_balanced_accuracy": float(val_ba.mean()),
                    "sd_val_balanced_accuracy": float(val_ba.std(ddof=1)),
                    "mean_test_balanced_accuracy": float(test_ba.mean()),
                    "sd_test_balanced_accuracy": float(test_ba.std(ddof=1)),
                    "sem_test_balanced_accuracy": _sem(test_ba),
                    "mean_test_accuracy": float(group.test_accuracy.mean()),
                    "mean_test_macro_f1": float(group.test_macro_f1.mean()),
                }
            )

    comparisons = (
        (SOFT_TEACHER, HARD_TEACHER, "softness_penalty"),
        (SOFT_TEACHER_INIT, SOFT_TEACHER, "teacher_init_retrain_gain"),
        (SOFT_TEACHER_INIT, SOFT_RANDOM, "teacher_initialization_benefit"),
        (HARD_RANDOM, HARD_TEACHER, "hard_optimization_gap"),
        (SOFT_LINEAR_REFIT, SOFT_RANDOM, "soft_linear_minus_soft_random"),
        (SOFT_LINEAR_REFIT, HARD_TEACHER, "soft_representation_gap_to_hard_teacher"),
        (SOFT_TEACHER_INIT, HARD_TEACHER, "soft_teacher_init_gap_to_hard_teacher"),
    )
    paired_rows: list[dict[str, object]] = []
    recovery_rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for seed in SEEDS:
            for left, right, name in comparisons:
                left_ba = float(per_run[(family, left, seed)]["test"]["balanced_accuracy"])
                right_ba = float(per_run[(family, right, seed)]["test"]["balanced_accuracy"])
                paired_rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "comparison": name,
                        "left_condition": left,
                        "right_condition": right,
                        "delta_test_balanced_accuracy": left_ba - right_ba,
                    }
                )
            hard = float(per_run[(family, HARD_TEACHER, seed)]["test"]["balanced_accuracy"])
            soft = float(per_run[(family, SOFT_TEACHER, seed)]["test"]["balanced_accuracy"])
            retrained = float(per_run[(family, SOFT_TEACHER_INIT, seed)]["test"]["balanced_accuracy"])
            denominator = hard - soft
            recovery_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "hard_teacher_test_ba": hard,
                    "soft_teacher_test_ba": soft,
                    "soft_teacher_init_retrain_test_ba": retrained,
                    "hard_to_soft_loss": denominator,
                    "retraining_recovery_fraction": (
                        (retrained - soft) / denominator if denominator > 1e-12 else np.nan
                    ),
                }
            )

    outputs = {
        "teacher_equivalence": root / "teacher_equivalence.csv",
        "runs": root / "runs.csv",
        "summary": root / "summary.csv",
        "paired_deltas": root / "paired_deltas.csv",
        "recovery": root / "recovery.csv",
        "logit_distortion": root / "logit_distortion.csv",
        "weight_drift": root / "weight_drift.csv",
        "manifest": root / "manifest.json",
    }
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(equivalence_rows).to_csv(outputs["teacher_equivalence"], index=False)
    runs.to_csv(outputs["runs"], index=False)
    pd.DataFrame(summary_rows).to_csv(outputs["summary"], index=False)
    pd.DataFrame(paired_rows).to_csv(outputs["paired_deltas"], index=False)
    pd.DataFrame(recovery_rows).to_csv(outputs["recovery"], index=False)
    pd.DataFrame(distortion_rows).to_csv(outputs["logit_distortion"], index=False)
    pd.DataFrame(weight_rows).to_csv(outputs["weight_drift"], index=False)
    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seeds": list(SEEDS),
            "families": list(FAMILIES),
            "conditions": list(ALL_CONDITIONS),
            "source_experiment": exp55.EXPERIMENT_ID,
            "source_protocol": exp55.PROTOCOL_VERSION,
            "source_what_frozen": True,
            "n_states_by_family": n_states_by_family,
            "soft_sigma_multiplier": SOFT_SIGMA_MULTIPLIER,
            "equivalence_atol": EQUIVALENCE_ATOL,
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4), grad clip 1.0",
            "linear_refit": "train-only StandardScaler + LogisticRegression(lbfgs), C selected on validation BA",
            "selection_policy": "epoch 0 and trained epochs are validation-selected within each train run; test never selects checkpoints, C, routing, or conditions",
            "multi_cpu_policy": "5 teacher tasks -> 30 independent train/evaluate tasks plus 10 independent soft-linear-refit tasks -> one artifact-only finalizer",
            "notebook_policy": "analysis-only; reads finalized CSV/JSON artifacts and never trains, refits, or regenerates missing artifacts",
        },
    )
    return outputs


def _config_from_args(args: argparse.Namespace) -> Config:
    root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    return Config(
        repo_root=root,
        results_dir=results_dir(root),
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        threads=args.threads,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 5.5.3 teacher-weight routing decomposition"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repo-root", default=None)
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=1)
        command.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        command.add_argument("--epochs", type=int, default=EPOCHS)
        command.add_argument("--patience", type=int, default=PATIENCE)
        command.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare-teachers")
    add_common(prepare)
    prepare.add_argument("--array-task-id", type=int, required=True)

    train = subparsers.add_parser("run-train")
    add_common(train)
    train.add_argument("--array-task-id", type=int, required=True)

    linear = subparsers.add_parser("run-linear")
    add_common(linear)
    linear.add_argument("--array-task-id", type=int, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--repo-root", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "finalize":
        root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
        for name, path in finalize_experiment(root).items():
            print(f"{name}: {path}")
        return

    config = _config_from_args(args)
    data = base.prepare_data(config.repo_root)
    task = int(args.array_task_id)
    if args.command == "prepare-teachers":
        if task < 0 or task >= len(SEEDS):
            raise IndexError(f"prepare-teachers task {task} outside 0..{len(SEEDS)-1}")
        for family, path in prepare_teacher_seed(SEEDS[task], data, config, force=args.force).items():
            print(f"{family}: {path}")
        return
    if args.command == "run-train":
        specs = train_specs()
        if task < 0 or task >= len(specs):
            raise IndexError(f"run-train task {task} outside 0..{len(specs)-1}")
        print(json.dumps(run_train_one(specs[task], data, config, force=args.force), indent=2))
        return
    if args.command == "run-linear":
        specs = linear_specs()
        if task < 0 or task >= len(specs):
            raise IndexError(f"run-linear task {task} outside 0..{len(specs)-1}")
        print(json.dumps(run_linear_one(specs[task], data, config, force=args.force), indent=2))
        return
    raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
