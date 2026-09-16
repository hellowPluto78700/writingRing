from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from scripts import experiment_7_3_2_affine_probe_bridge as exp732
from scripts import experiment_7_3_training_strategy_decomposition as exp73


EXPERIMENT_ID = "experiment_7_3_7_wholecount_bias_transfer"
PROTOCOL_VERSION = "wholecount_bias_transfer_v1"
ARCHITECTURE = exp73.ARCHITECTURE
SEEDS = exp73.SEEDS
BACKBONE_OBJECTIVE = "wcce"
LINEAR_METHODS = ("L0_linear_no_bias", "L1_linear_affine")
METHODS = (
    "L0_linear_no_bias",
    "L1_linear_affine",
    "L2_same_w_lif_count",
    "L3_same_w_lif_continuous_bias",
    "L4_same_w_lif_integer_bias",
)
C_GRID = exp732.C_GRID
MAX_ITER = exp732.MAX_ITER
LIF_BETA = exp73.LIF_BETA
THRESHOLD = exp73.THRESHOLD
OUTPUT_CAP = exp73.OUTPUT_CAP
CALIBRATION_LR = 5e-2
CALIBRATION_MAX_EPOCHS = 500
CALIBRATION_MIN_EPOCHS = 20
CALIBRATION_PATIENCE = 80
GAIN_LOG_MIN = -8.0
GAIN_LOG_MAX = 4.0


@dataclass(frozen=True)
class RunSpec:
    seed: int

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__seed{self.seed}"

    @property
    def backbone(self) -> exp73.BackboneSpec:
        return exp73.BackboneSpec(self.seed, BACKBONE_OBJECTIVE)


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp73.exp72.BATCH_SIZE
    threads: int = 1
    calibration_max_epochs: int = CALIBRATION_MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(seed) for seed in SEEDS]


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _exp73_config(config: Config) -> exp73.Config:
    return exp73.Config(
        repo_root=config.repo_root,
        results_dir=exp73.results_dir(config.repo_root),
        device=config.device,
        batch_size=config.batch_size,
        threads=config.threads,
        max_epochs=exp73.MAX_EPOCHS,
    )


def _load_cache(config: Config, spec: RunSpec) -> dict[str, Any]:
    cache = exp73._load_stage2_cache(spec.backbone, _exp73_config(config))
    meta = cache["metadata"]
    if meta.get("source_method") != "A2_e2e_linear_wcce":
        raise RuntimeError(
            f"{spec.key}: expected A2_e2e_linear_wcce cache, got {meta.get('source_method')}"
        )
    if not bool(meta.get("frozen_l1_l2")):
        raise RuntimeError(f"{spec.key}: Exp7.3 cache is not marked frozen")
    return cache


def _wholecount_features(
    split: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    l2, y, lengths = split
    x = l2.astype(np.float64, copy=False)
    positions = np.arange(x.shape[1], dtype=np.int64)[None, :]
    mask = (positions < lengths[:, None]).astype(np.float64)[..., None]
    return (x * mask).sum(axis=1), y.astype(np.int64, copy=False)


def _fold_scaled_affine(
    coef: np.ndarray,
    intercept: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_weight = np.asarray(coef, dtype=np.float64) / np.asarray(scale, dtype=np.float64)[None, :]
    raw_bias = np.asarray(intercept, dtype=np.float64).copy()
    return raw_weight, raw_bias


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return exp73._metrics(y, scores)


def _evaluate_raw_linear(
    features: dict[str, tuple[np.ndarray, np.ndarray]],
    raw_weight: np.ndarray,
    raw_bias: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        split: _metrics(y, x @ raw_weight.T + raw_bias[None, :])
        for split, (x, y) in features.items()
    }


def _linear_seed(seed: int, fit_intercept: bool) -> int:
    role = "p5_affine" if fit_intercept else "p4_no_bias"
    return int(exp73._probe_seed(seed, f"exp7_3_7_{role}"))


def _fit_linear_case(
    features: dict[str, tuple[np.ndarray, np.ndarray]],
    scaler: StandardScaler,
    seed: int,
    fit_intercept: bool,
) -> dict[str, Any]:
    transformed = {
        split: (scaler.transform(x), y)
        for split, (x, y) in features.items()
    }
    candidates: list[dict[str, Any]] = []
    best: tuple[float, float, LogisticRegression] | None = None
    classifier_seed = _linear_seed(seed, fit_intercept)
    for C in C_GRID:
        classifier = LogisticRegression(
            C=float(C),
            max_iter=MAX_ITER,
            solver="lbfgs",
            random_state=classifier_seed,
            fit_intercept=fit_intercept,
        ).fit(transformed["train"][0], transformed["train"][1])
        val_pred = classifier.predict(transformed["val"][0])
        val_ba = float(balanced_accuracy_score(transformed["val"][1], val_pred))
        candidates.append(
            {
                "C": float(C),
                "val_balanced_accuracy": val_ba,
                "n_iter_max": int(np.asarray(classifier.n_iter_).max()),
            }
        )
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError("No LogisticRegression candidate selected")

    selected_val_ba, selected_C, classifier = best
    expected_classes = np.arange(len(np.unique(features["train"][1])), dtype=np.int64)
    if not np.array_equal(classifier.classes_, expected_classes):
        raise RuntimeError(
            f"Unexpected class order: {classifier.classes_.tolist()} vs {expected_classes.tolist()}"
        )
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    intercept = (
        np.asarray(classifier.intercept_, dtype=np.float64)
        if fit_intercept
        else np.zeros(classifier.coef_.shape[0], dtype=np.float64)
    )
    raw_weight, raw_bias = _fold_scaled_affine(classifier.coef_, intercept, scale)
    metrics = _evaluate_raw_linear(features, raw_weight, raw_bias)
    return {
        "selected_C": selected_C,
        "selected_val_balanced_accuracy": selected_val_ba,
        "candidate_validation": candidates,
        "metrics": metrics,
        "raw_weight": raw_weight,
        "raw_bias": raw_bias,
        "scale": scale,
        "weight_l2": float(np.linalg.norm(raw_weight)),
        "bias_l2": float(np.linalg.norm(raw_bias)),
        "bias_span": float(raw_bias.max() - raw_bias.min()),
    }


def _lif_counts(
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    raw_weight: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    l2, y, lengths = split
    n_classes, n_features = raw_weight.shape
    if l2.shape[-1] != n_features:
        raise ValueError((l2.shape, raw_weight.shape))
    model = exp73.Stage2Head("lif", n_classes).to(device)
    with torch.no_grad():
        model.output_linear.weight.copy_(
            torch.from_numpy(raw_weight).to(
                device=device, dtype=model.output_linear.weight.dtype
            )
        )
    model.eval()
    counts_all: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            stop = min(start + batch_size, len(y))
            xb = torch.from_numpy(l2[start:stop]).to(device=device, dtype=torch.float32)
            lb = torch.from_numpy(lengths[start:stop]).to(device=device, dtype=torch.long)
            trajectory = model.forward_trajectory(xb)
            spikes = trajectory["output_spikes"]
            valid = exp73._valid_mask(lb, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
            counts = (spikes * valid).sum(dim=1)
            counts_all.append(counts.cpu().numpy().astype(np.float64, copy=False))
    return np.concatenate(counts_all, axis=0), y.astype(np.int64, copy=False)


class CountBiasCalibrator(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.log_gain = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.raw_q = nn.Parameter(torch.zeros(n_classes, dtype=torch.float32))

    def q(self) -> torch.Tensor:
        return self.raw_q - self.raw_q.mean()

    def gain(self) -> torch.Tensor:
        return torch.exp(self.log_gain.clamp(GAIN_LOG_MIN, GAIN_LOG_MAX))

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        return self.gain() * (counts + self.q())


def _evaluate_calibrator(
    model: CountBiasCalibrator,
    counts: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        scores = model(counts)
        out = _metrics(y.cpu().numpy(), scores.cpu().numpy())
        out["objective_loss"] = float(F.cross_entropy(scores, y))
    return out


def _fit_count_bias(
    counts: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
    device: torch.device,
    max_epochs: int,
) -> dict[str, Any]:
    n_classes = int(counts["train"][0].shape[1])
    exp73.exp3.seed_all(exp73._stage2_pair_seed(seed, "exp7_3_7_count_bias"))
    model = CountBiasCalibrator(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CALIBRATION_LR)
    tensors = {
        split: (
            torch.from_numpy(x).to(device=device, dtype=torch.float32),
            torch.from_numpy(y).to(device=device, dtype=torch.long),
        )
        for split, (x, y) in counts.items()
    }

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = max_epochs
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_x, train_y = tensors["train"]
        optimizer.zero_grad(set_to_none=True)
        train_scores = model(train_x)
        loss = F.cross_entropy(train_scores, train_y)
        loss.backward()
        optimizer.step()

        train_eval = _evaluate_calibrator(model, *tensors["train"])
        val_eval = _evaluate_calibrator(model, *tensors["val"])
        history.append(
            {
                "epoch": epoch,
                "optimization_train_loss": float(loss.detach()),
                "train_ba": train_eval["balanced_accuracy"],
                "val_ba": val_eval["balanced_accuracy"],
                "train_loss": train_eval["objective_loss"],
                "val_loss": val_eval["objective_loss"],
                "gain": float(model.gain().detach()),
                "q_l2": float(torch.linalg.vector_norm(model.q()).detach()),
                "q_span": float((model.q().max() - model.q().min()).detach()),
            }
        )
        if exp73._checkpoint_improved(val_eval, best_ba, best_loss):
            best_ba = float(val_eval["balanced_accuracy"])
            best_loss = float(val_eval["objective_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if (
            epoch >= CALIBRATION_MIN_EPOCHS
            and best_epoch > 0
            and epoch - best_epoch >= CALIBRATION_PATIENCE
        ):
            stopped_epoch = epoch
            break
    if best_state is None:
        raise RuntimeError("No count-bias checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        q_centered = model.q().cpu().numpy().astype(np.float64)
        gain = float(model.gain().cpu())
    q_nonnegative = q_centered - float(q_centered.min())
    q_integer = np.rint(q_nonnegative).astype(np.int64)
    return {
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_ba": best_ba,
        "best_val_loss": best_loss,
        "gain": gain,
        "q_centered": q_centered,
        "q_nonnegative": q_nonnegative,
        "q_integer": q_integer,
        "history": history,
    }


def _evaluate_lif_methods(
    counts: dict[str, tuple[np.ndarray, np.ndarray]],
    q_centered: np.ndarray,
    q_integer: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for split, (x, y) in counts.items():
        result[split] = {
            "L2_same_w_lif_count": _metrics(y, x),
            "L3_same_w_lif_continuous_bias": _metrics(
                y, x + q_centered[None, :]
            ),
            "L4_same_w_lif_integer_bias": _metrics(
                y, x + q_integer.astype(np.float64)[None, :]
            ),
        }
    return result


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    evaluation_path = config.results_dir / "evaluations" / f"{spec.key}.json"
    checkpoint_path = config.results_dir / "checkpoints" / f"{spec.key}.npz"
    history_path = config.results_dir / "histories" / f"{spec.key}.csv"
    if (
        evaluation_path.exists()
        and checkpoint_path.exists()
        and history_path.exists()
        and not force
    ):
        return json.loads(evaluation_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    cache = _load_cache(config, spec)
    features = {
        split: _wholecount_features(cache[split])
        for split in ("train", "val", "test")
    }
    scaler = StandardScaler(with_mean=False, with_std=True).fit(features["train"][0])
    no_bias = _fit_linear_case(features, scaler, spec.seed, fit_intercept=False)
    affine = _fit_linear_case(features, scaler, spec.seed, fit_intercept=True)

    lif_counts = {
        split: _lif_counts(
            cache[split], affine["raw_weight"], config.batch_size, device
        )
        for split in ("train", "val", "test")
    }
    calibration = _fit_count_bias(
        lif_counts,
        spec.seed,
        device,
        config.calibration_max_epochs,
    )
    lif_metrics = _evaluate_lif_methods(
        lif_counts,
        calibration["q_centered"],
        calibration["q_integer"],
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint_path,
        linear_no_bias_raw_weight=no_bias["raw_weight"],
        linear_affine_raw_weight=affine["raw_weight"],
        linear_affine_raw_bias=affine["raw_bias"],
        wholecount_scale=np.asarray(scaler.scale_, dtype=np.float64),
        lif_count_bias_centered=calibration["q_centered"],
        lif_count_bias_nonnegative=calibration["q_nonnegative"],
        lif_count_bias_integer=calibration["q_integer"],
        lif_count_calibration_gain=np.asarray([calibration["gain"]], dtype=np.float64),
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(calibration["history"]).to_csv(history_path, index=False)

    linear_payload = {}
    for name, result in (
        ("L0_linear_no_bias", no_bias),
        ("L1_linear_affine", affine),
    ):
        linear_payload[name] = {
            "selected_C": result["selected_C"],
            "selected_val_balanced_accuracy": result["selected_val_balanced_accuracy"],
            "candidate_validation": result["candidate_validation"],
            "metrics": result["metrics"],
            "weight_l2": result["weight_l2"],
            "bias_l2": result["bias_l2"],
            "bias_span": result["bias_span"],
        }

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "architecture": ARCHITECTURE,
            "backbone_source": "A2_e2e_linear_wcce",
            "frozen_l1_l2": True,
            "stage_a_feature": "valid whole-count Z=sum_t z_t",
            "stage_a_scaler": "train-only per-feature std, no centering; folded into raw W",
            "stage_a_trainable": "only Linear W and optional 12-class bias",
            "stage_a_solver": "sklearn LogisticRegression lbfgs with L2; C selected by validation BA",
            "stage_b_weight_transfer": "exact raw W from L1_linear_affine; no LIF W training",
            "lif_beta": LIF_BETA,
            "threshold": THRESHOLD,
            "output_cap": OUTPUT_CAP,
            "lif_input_bias": "none; linear bias is not injected into LIF dynamics",
            "stage_c_trainable": "only 12D count-space q plus positive CE-only scalar gain",
            "stage_c_selection": "validation BA primary, validation CE loss tiebreak",
            "deployment_continuous": "argmax(output_spike_count + q)",
            "deployment_integer": "argmax(output_spike_count + round(q - min(q)))",
            "test_untouched_until_reporting": True,
        },
        "linear": linear_payload,
        "lif_metrics": lif_metrics,
        "count_bias": {
            "best_epoch": calibration["best_epoch"],
            "stopped_epoch": calibration["stopped_epoch"],
            "best_val_ba": calibration["best_val_ba"],
            "best_val_loss": calibration["best_val_loss"],
            "ce_only_gain": calibration["gain"],
            "q_centered": calibration["q_centered"].tolist(),
            "q_nonnegative": calibration["q_nonnegative"].tolist(),
            "q_integer": calibration["q_integer"].tolist(),
        },
    }
    _save_json(evaluation_path, payload)
    return payload


def _method_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(payload["spec"]["seed"])
    rows: list[dict[str, Any]] = []
    for method in LINEAR_METHODS:
        result = payload["linear"][method]
        rows.append(
            {
                "method": method,
                "seed": seed,
                "train_ba": float(result["metrics"]["train"]["balanced_accuracy"]),
                "val_ba": float(result["metrics"]["val"]["balanced_accuracy"]),
                "test_ba": float(result["metrics"]["test"]["balanced_accuracy"]),
                "selected_C": float(result["selected_C"]),
                "weight_l2": float(result["weight_l2"]),
                "bias_l2": float(result["bias_l2"]),
                "bias_span": float(result["bias_span"]),
            }
        )
    for method in METHODS[2:]:
        metrics = payload["lif_metrics"]
        rows.append(
            {
                "method": method,
                "seed": seed,
                "train_ba": float(metrics["train"][method]["balanced_accuracy"]),
                "val_ba": float(metrics["val"][method]["balanced_accuracy"]),
                "test_ba": float(metrics["test"][method]["balanced_accuracy"]),
                "selected_C": np.nan,
                "weight_l2": float(payload["linear"]["L1_linear_affine"]["weight_l2"]),
                "bias_l2": (
                    float(np.linalg.norm(payload["count_bias"]["q_centered"]))
                    if method == "L3_same_w_lif_continuous_bias"
                    else 0.0
                ),
                "bias_span": (
                    float(
                        np.max(payload["count_bias"]["q_centered"])
                        - np.min(payload["count_bias"]["q_centered"])
                    )
                    if method == "L3_same_w_lif_continuous_bias"
                    else (
                        float(
                            np.max(payload["count_bias"]["q_integer"])
                            - np.min(payload["count_bias"]["q_integer"])
                        )
                        if method == "L4_same_w_lif_integer_bias"
                        else 0.0
                    )
                ),
            }
        )
    return rows


def _gap_row(payload: dict[str, Any]) -> dict[str, Any]:
    seed = int(payload["spec"]["seed"])
    linear0 = float(payload["linear"]["L0_linear_no_bias"]["metrics"]["test"]["balanced_accuracy"])
    linear1 = float(payload["linear"]["L1_linear_affine"]["metrics"]["test"]["balanced_accuracy"])
    lif = float(payload["lif_metrics"]["test"]["L2_same_w_lif_count"]["balanced_accuracy"])
    continuous = float(payload["lif_metrics"]["test"]["L3_same_w_lif_continuous_bias"]["balanced_accuracy"])
    integer = float(payload["lif_metrics"]["test"]["L4_same_w_lif_integer_bias"]["balanced_accuracy"])
    return {
        "seed": seed,
        "linear_bias_gain_pp": 100.0 * (linear1 - linear0),
        "linear_to_lif_loss_pp": 100.0 * (linear1 - lif),
        "continuous_bias_recovery_pp": 100.0 * (continuous - lif),
        "integer_bias_recovery_pp": 100.0 * (integer - lif),
        "integer_vs_continuous_pp": 100.0 * (integer - continuous),
        "remaining_gap_after_continuous_pp": 100.0 * (linear1 - continuous),
        "remaining_gap_after_integer_pp": 100.0 * (linear1 - integer),
    }


def _source_reproduction(config: Config, method_runs: pd.DataFrame) -> dict[str, Any]:
    path = (
        config.repo_root
        / "notebooks/artifacts/experiment_7_3_2_affine_probe_bridge"
        / "affine_probe_bridge_v1/method_runs.csv"
    )
    result: dict[str, Any] = {
        "source": str(path.relative_to(config.repo_root)),
        "available": path.exists(),
    }
    if not path.exists():
        return result
    source = pd.read_csv(path)
    mapping = {
        "L0_linear_no_bias": "P4_wholecount_scale_no_center_no_bias",
        "L1_linear_affine": "P5_wholecount_scale_no_center_bias",
    }
    deltas: dict[str, list[float]] = {}
    for method, case in mapping.items():
        left = method_runs[method_runs.method == method][["seed", "test_ba"]]
        right = source[
            (source.backbone_objective == "wcce") & (source.case == case)
        ][["seed", "test_ba"]]
        merged = left.merge(right, on="seed", suffixes=("_exp737", "_exp732"))
        deltas[method] = (
            100.0 * (merged.test_ba_exp737 - merged.test_ba_exp732)
        ).tolist()
    result["paired_test_ba_delta_pp"] = deltas
    result["max_abs_delta_pp"] = max(
        abs(v) for values in deltas.values() for v in values
    )
    return result


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    missing: list[Path] = []
    for spec in run_specs():
        path = config.results_dir / "evaluations" / f"{spec.key}.json"
        if path.exists():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing Exp7.3.7 evaluations:\n" + "\n".join(str(p) for p in missing)
        )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    method_runs = pd.DataFrame(
        [row for payload in payloads for row in _method_rows(payload)]
    ).sort_values(["method", "seed"])
    method_runs.to_csv(config.results_dir / "method_runs.csv", index=False)
    numeric = [c for c in method_runs.columns if c not in {"method", "seed"}]
    method_summary = method_runs.groupby("method", sort=False)[numeric].agg(["mean", "std", "count"])
    method_summary.columns = [f"{name}_{stat}" for name, stat in method_summary.columns]
    method_summary.reset_index().to_csv(config.results_dir / "method_summary.csv", index=False)

    gap_runs = pd.DataFrame([_gap_row(payload) for payload in payloads]).sort_values("seed")
    gap_runs.to_csv(config.results_dir / "gap_runs.csv", index=False)
    gap_summary = gap_runs.drop(columns=["seed"]).agg(["mean", "std", "count"]).T.reset_index()
    gap_summary.columns = ["contrast", "delta_pp_mean", "delta_pp_std", "count"]
    gap_summary.to_csv(config.results_dir / "gap_summary.csv", index=False)

    bias_rows: list[dict[str, Any]] = []
    for payload in payloads:
        seed = int(payload["spec"]["seed"])
        checkpoint = np.load(
            config.results_dir / "checkpoints" / f"{ARCHITECTURE}__seed{seed}.npz",
            allow_pickle=False,
        )
        linear_bias = checkpoint["linear_affine_raw_bias"]
        q_centered = checkpoint["lif_count_bias_centered"]
        q_nonnegative = checkpoint["lif_count_bias_nonnegative"]
        q_integer = checkpoint["lif_count_bias_integer"]
        for class_index in range(len(linear_bias)):
            bias_rows.append(
                {
                    "seed": seed,
                    "class_index": class_index,
                    "linear_bias_logit": float(linear_bias[class_index]),
                    "lif_count_bias_centered": float(q_centered[class_index]),
                    "lif_count_bias_nonnegative": float(q_nonnegative[class_index]),
                    "lif_count_bias_integer": int(q_integer[class_index]),
                }
            )
    bias_vectors = pd.DataFrame(bias_rows).sort_values(["class_index", "seed"])
    bias_vectors.to_csv(config.results_dir / "bias_vectors.csv", index=False)
    bias_numeric = [c for c in bias_vectors.columns if c not in {"seed", "class_index"}]
    bias_summary = bias_vectors.groupby("class_index")[bias_numeric].agg(["mean", "std", "count"])
    bias_summary.columns = [f"{name}_{stat}" for name, stat in bias_summary.columns]
    bias_summary.reset_index().to_csv(config.results_dir / "bias_vector_summary.csv", index=False)

    histories: list[pd.DataFrame] = []
    for spec in run_specs():
        path = config.results_dir / "histories" / f"{spec.key}.csv"
        df = pd.read_csv(path)
        df.insert(0, "seed", spec.seed)
        histories.append(df)
    history_runs = pd.concat(histories, ignore_index=True)
    history_runs.to_csv(config.results_dir / "calibration_history_runs.csv", index=False)
    history_numeric = [c for c in history_runs.columns if c not in {"seed", "epoch"}]
    history_summary = history_runs.groupby("epoch", sort=True)[history_numeric].agg(["mean", "std", "count"])
    history_summary.columns = [f"{name}_{stat}" for name, stat in history_summary.columns]
    history_summary.reset_index().to_csv(
        config.results_dir / "calibration_history_summary.csv", index=False
    )

    reproduction = _source_reproduction(config, method_runs)
    _save_json(config.results_dir / "source_reproduction_checks.json", reproduction)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "logical_training_runs": len(SEEDS),
        "frozen_backbone": "Exp7.3 A2_e2e_linear_wcce L2 cache",
        "stage_a": "whole-count scale-only LogisticRegression; L0 no bias and L1 affine",
        "stage_b": "transfer L1 raw W unchanged to beta=0.5 LIF; no input bias and no LIF W training",
        "stage_c": "fit only class-specific output-count q plus positive CE-only gain; report continuous and integer virtual bias counts",
        "multi_cpu_contract": "one CPU per seed task; each task runs Stage A->B->C->evaluate; afterok finalizer aggregates only",
        "notebook_contract": "analysis-only; method-level aggregation, no per-run result dump",
        "source_reproduction": reproduction,
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp7.3.7 whole-count bias transfer")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp73.exp72.BATCH_SIZE)
    parser.add_argument(
        "--calibration-max-epochs", type=int, default=CALIBRATION_MAX_EPOCHS
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
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
        calibration_max_epochs=args.calibration_max_epochs,
    )
    if args.command == "run":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        run_one(specs[args.array_task_id], config, force=args.force)
    elif args.command == "finalize":
        finalize(config)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
