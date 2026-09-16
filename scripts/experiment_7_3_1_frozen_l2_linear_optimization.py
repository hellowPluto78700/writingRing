from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73


EXPERIMENT_ID = "experiment_7_3_1_frozen_l2_linear_optimization"
PROTOCOL_VERSION = "frozen_l2_linear_optimization_v1"
ARCHITECTURE = exp73.ARCHITECTURE
SEEDS = exp73.SEEDS
BACKBONE_OBJECTIVES = ("tsce", "wcce")
HEAD_OBJECTIVES = ("tsce", "wcce")
CASES = (
    "C0_adam100",
    "C1_adam500",
    "C2_lbfgs_noreg",
    "C3_lbfgs_reg",
    "C4_scale_lbfgs_reg",
    "Ref_scale_lbfgs_converged",
)
REG_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
ADAM_EPOCHS = 500
ADAM_C0_EPOCH_LIMIT = 100
LBFGS_MAX_EVAL = 100
REF_MAX_EVAL = 5000
LBFGS_HISTORY_SIZE = 50
CE_GAIN = 1.0
LIF_BETA = exp73.LIF_BETA
THRESHOLD = exp73.THRESHOLD
OUTPUT_CAP = exp73.OUTPUT_CAP
HIDDEN_WIDTH = exp73.HIDDEN_WIDTH
BIAS = False
SCALING_EPS = 1e-6


@dataclass(frozen=True)
class BaseSpec:
    seed: int
    backbone_objective: str
    objective: str

    @property
    def key(self) -> str:
        return (
            f"{ARCHITECTURE}__backbone_{self.backbone_objective}__"
            f"head_{self.objective}__seed{self.seed}"
        )


@dataclass(frozen=True)
class CandidateSpec:
    family: str
    base: BaseSpec
    reg_lambda: float

    @property
    def key(self) -> str:
        lam = f"{self.reg_lambda:.0e}".replace("+", "")
        return f"{self.base.key}__{self.family}__lambda{lam}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def base_specs() -> list[BaseSpec]:
    return [
        BaseSpec(seed, backbone_objective, objective)
        for seed in SEEDS
        for backbone_objective in BACKBONE_OBJECTIVES
        for objective in HEAD_OBJECTIVES
    ]


def candidate_specs(family: str) -> list[CandidateSpec]:
    if family not in {"C3_lbfgs_reg", "C4_scale_lbfgs_reg", "Ref_scale_lbfgs_converged"}:
        raise ValueError(family)
    return [
        CandidateSpec(family, base, reg_lambda)
        for base in base_specs()
        for reg_lambda in REG_LAMBDAS
    ]


def validate_base(spec: BaseSpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    if spec.backbone_objective not in BACKBONE_OBJECTIVES:
        raise ValueError(spec.backbone_objective)
    if spec.objective not in HEAD_OBJECTIVES:
        raise ValueError(spec.objective)


def validate_candidate(spec: CandidateSpec) -> None:
    validate_base(spec.base)
    if spec.reg_lambda not in REG_LAMBDAS:
        raise ValueError(spec.reg_lambda)
    if spec.family not in {"C3_lbfgs_reg", "C4_scale_lbfgs_reg", "Ref_scale_lbfgs_converged"}:
        raise ValueError(spec.family)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _exp73_cache_paths(repo_root: Path, spec: BaseSpec) -> tuple[Path, Path]:
    backbone = exp73.BackboneSpec(spec.seed, spec.backbone_objective)
    root = exp73.results_dir(repo_root) / "stage2_l2_cache"
    return root / f"{backbone.key}.npz", root / f"{backbone.key}.json"


def _load_cache(repo_root: Path, spec: BaseSpec) -> dict[str, Any]:
    npz_path, meta_path = _exp73_cache_paths(repo_root, spec)
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Exp7.3 frozen-L2 cache missing for {spec.key}: {npz_path} / {meta_path}"
        )
    with np.load(npz_path, allow_pickle=False) as z:
        result: dict[str, Any] = {
            split: (
                z[f"{split}_l2"].copy(),
                z[f"{split}_y"].copy(),
                z[f"{split}_lengths"].copy(),
            )
            for split in ("train", "val", "test")
        }
    result["metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return result


def _valid_mask_np(lengths: np.ndarray, n_steps: int) -> np.ndarray:
    return np.arange(n_steps, dtype=np.int64)[None, :] < lengths[:, None]


def _objective_features(
    split: tuple[np.ndarray, np.ndarray, np.ndarray], objective: str
) -> tuple[np.ndarray, np.ndarray]:
    l2, y, lengths = split
    x = l2.astype(np.float32, copy=False)
    if objective == "wcce":
        mask = _valid_mask_np(lengths, x.shape[1]).astype(np.float32)[..., None]
        summed = (x * mask).sum(axis=1)
        features = summed / np.maximum(lengths, 1).astype(np.float32)[:, None]
        return features.astype(np.float32, copy=False), y.astype(np.int64, copy=False)
    if objective == "tsce":
        valid = _valid_mask_np(lengths, x.shape[1])
        features = x[valid]
        targets = np.broadcast_to(y[:, None], valid.shape)[valid]
        return features.astype(np.float32, copy=False), targets.astype(np.int64, copy=False)
    raise ValueError(objective)


def _feature_scale(train_x: np.ndarray) -> np.ndarray:
    sigma = train_x.astype(np.float64).std(axis=0, ddof=0)
    sigma = np.maximum(sigma, SCALING_EPS)
    return sigma.astype(np.float32)


def _fold_scale_weight(weight_scaled: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return weight_scaled / sigma[None, :]


def _initial_weight(seed: int, n_classes: int) -> torch.Tensor:
    exp3.seed_all(exp73._stage2_pair_seed(seed, "model_init"))
    model = exp73.Stage2Head("linear", n_classes)
    return model.output_linear.weight.detach().cpu().clone()


def _objective_loss_from_weight(
    weight: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    reg_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(x, weight, bias=None)
    task = F.cross_entropy(CE_GAIN * logits, y)
    total = task + 0.5 * float(reg_lambda) * weight.square().sum()
    return task, total


def _objective_metrics(
    weight: np.ndarray,
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    objective: str,
    reg_lambda: float,
    sigma: np.ndarray | None = None,
) -> dict[str, float]:
    x, y = _objective_features(split, objective)
    if sigma is not None:
        x = x / sigma[None, :]
    wt = torch.from_numpy(weight.astype(np.float32, copy=False))
    xt = torch.from_numpy(x.astype(np.float32, copy=False))
    yt = torch.from_numpy(y.astype(np.int64, copy=False))
    with torch.no_grad():
        task, total = _objective_loss_from_weight(wt, xt, yt, reg_lambda)
    return {"task_ce": float(task.item()), "total_objective": float(total.item())}


def _cross_metrics(
    cache: dict[str, Any], weight_raw: np.ndarray
) -> dict[str, dict[str, dict[str, float]]]:
    splits = {split: cache[split] for split in ("train", "val", "test")}
    return exp73._cross_evaluate_w(splits, weight_raw.astype(np.float64, copy=False))


def _checkpoint_improved(
    val_ba: float,
    val_task_ce: float,
    best_ba: float,
    best_loss: float,
) -> bool:
    return val_ba > best_ba + 1e-12 or (
        abs(val_ba - best_ba) <= 1e-12 and val_task_ce < best_loss
    )


def _payload_from_weight(
    spec: BaseSpec,
    case: str,
    cache: dict[str, Any],
    weight_train_space: np.ndarray,
    weight_raw: np.ndarray,
    optimizer: str,
    reg_lambda: float,
    sigma: np.ndarray | None,
    budget: dict[str, Any],
    optimization: dict[str, Any],
) -> dict[str, Any]:
    objective_metrics = {
        split: _objective_metrics(
            weight_train_space,
            cache[split],
            spec.objective,
            reg_lambda,
            sigma,
        )
        for split in ("train", "val", "test")
    }
    cross = _cross_metrics(cache, weight_raw)
    source = cache["metadata"]
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "case": case,
        "spec": asdict(spec),
        "contract": {
            "architecture": ARCHITECTURE,
            "frozen_l1_l2": True,
            "source_exp73_method": source["source_method"],
            "source_exp73_best_epoch": source["source_best_epoch"],
            "bias": BIAS,
            "ce_gain": CE_GAIN,
            "training_objective": spec.objective,
            "linear_sequence_readout": "valid evidence sum",
            "same_w_lif_substitution": True,
            "lif_beta": LIF_BETA,
            "threshold": THRESHOLD,
            "output_cap": OUTPUT_CAP,
            "no_lif_retraining": True,
        },
        "optimizer": optimizer,
        "reg_lambda": float(reg_lambda),
        "scaling": {
            "mode": "scale_only" if sigma is not None else "none",
            "folded_into_raw_weight": sigma is not None,
            "sigma_min": float(np.min(sigma)) if sigma is not None else None,
            "sigma_max": float(np.max(sigma)) if sigma is not None else None,
        },
        "budget": budget,
        "optimization": optimization,
        "objective_metrics": objective_metrics,
        "cross_evaluation": cross,
        "weight_norm_train_space": float(np.linalg.norm(weight_train_space)),
        "weight_norm_raw": float(np.linalg.norm(weight_raw)),
        "legacy_l2_wholecount_probe_test_ba": float(
            source["representation_probes"]["l2_wholecount_linear"]["metrics"]["test"][
                "balanced_accuracy"
            ]
        ),
        "legacy_l2_fixed250_probe_test_ba": float(
            source["representation_probes"]["l2_fixed250_linear"]["metrics"]["test"][
                "balanced_accuracy"
            ]
        ),
    }


def run_adam(spec: BaseSpec, config: Config, force: bool = False) -> list[dict[str, Any]]:
    validate_base(spec)
    out_paths = {
        "C0_adam100": _path(config.results_dir, "evaluations", f"{spec.key}__C0_adam100", ".json"),
        "C1_adam500": _path(config.results_dir, "evaluations", f"{spec.key}__C1_adam500", ".json"),
    }
    if all(path.exists() for path in out_paths.values()) and not force:
        return [json.loads(out_paths[case].read_text(encoding="utf-8")) for case in out_paths]

    cache = _load_cache(config.repo_root, spec)
    n_classes = int(np.max(cache["train"][1])) + 1
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp3.seed_all(exp73._stage2_pair_seed(spec.seed, "model_init"))
    model = exp73.Stage2Head("linear", n_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY
    )
    train_loader = exp73._cached_loaders(
        cache, spec.seed, config.batch_size, True
    )["train"]
    eval_loaders = exp73._cached_loaders(cache, spec.seed, config.batch_size, False)
    exp73_spec = exp73.Stage2Spec(
        spec.seed, spec.backbone_objective, "linear", spec.objective
    )

    best: dict[str, dict[str, Any]] = {
        "C0_adam100": {"ba": -1.0, "loss": float("inf"), "epoch": -1, "state": None},
        "C1_adam500": {"ba": -1.0, "loss": float("inf"), "epoch": -1, "state": None},
    }
    history: list[dict[str, float]] = []

    for epoch in range(1, ADAM_EPOCHS + 1):
        model.train()
        for l2, y, lengths in train_loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            tr = model.forward_trajectory(l2)
            loss, _ = exp73._objective_loss_scores(
                tr["evidence"], lengths, y, spec.objective
            )
            loss.backward()
            optimizer.step()

        train_metrics = exp73._evaluate_stage2_native(
            exp73_spec, model, eval_loaders["train"], device
        )
        val_metrics = exp73._evaluate_stage2_native(
            exp73_spec, model, eval_loaders["val"], device
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_task_ce": float(train_metrics["objective_loss"]),
                "val_task_ce": float(val_metrics["objective_loss"]),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
            }
        )
        for case, limit in (("C0_adam100", ADAM_C0_EPOCH_LIMIT), ("C1_adam500", ADAM_EPOCHS)):
            if epoch > limit:
                continue
            if _checkpoint_improved(
                float(val_metrics["balanced_accuracy"]),
                float(val_metrics["objective_loss"]),
                float(best[case]["ba"]),
                float(best[case]["loss"]),
            ):
                best[case] = {
                    "ba": float(val_metrics["balanced_accuracy"]),
                    "loss": float(val_metrics["objective_loss"]),
                    "epoch": epoch,
                    "state": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                }

    history_path = _path(config.results_dir, "histories", f"{spec.key}__adam500", ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)

    payloads: list[dict[str, Any]] = []
    for case, limit in (("C0_adam100", ADAM_C0_EPOCH_LIMIT), ("C1_adam500", ADAM_EPOCHS)):
        state = best[case]["state"]
        if state is None:
            raise RuntimeError(f"No Adam checkpoint selected for {spec.key}/{case}")
        model.load_state_dict(state, strict=True)
        weight_raw = model.output_linear.weight.detach().cpu().numpy().astype(np.float64)
        payload = _payload_from_weight(
            spec,
            case,
            cache,
            weight_raw,
            weight_raw,
            "adam",
            0.0,
            None,
            {
                "unit": "epochs",
                "limit": limit,
                "shared_trajectory_total_epochs": ADAM_EPOCHS,
            },
            {
                "selected_epoch": int(best[case]["epoch"]),
                "selected_val_ba": float(best[case]["ba"]),
                "selected_val_task_ce": float(best[case]["loss"]),
                "history_path": str(history_path.relative_to(config.repo_root)),
            },
        )
        _save_json(out_paths[case], payload)
        payloads.append(payload)
    return payloads


def _run_lbfgs_weight(
    spec: BaseSpec,
    cache: dict[str, Any],
    reg_lambda: float,
    use_scale: bool,
    max_eval: int,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any], list[dict[str, float]]]:
    n_classes = int(np.max(cache["train"][1])) + 1
    train_x, train_y = _objective_features(cache["train"], spec.objective)
    sigma = _feature_scale(train_x) if use_scale else None
    if sigma is not None:
        train_x = train_x / sigma[None, :]

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    weight = torch.nn.Parameter(_initial_weight(spec.seed, n_classes).to(device))
    x = torch.from_numpy(train_x).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(train_y).to(device=device, dtype=torch.long)
    optimizer = torch.optim.LBFGS(
        [weight],
        lr=1.0,
        max_iter=max_eval,
        max_eval=max_eval,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )
    trace: list[dict[str, float]] = []
    eval_count = 0

    def closure() -> torch.Tensor:
        nonlocal eval_count
        optimizer.zero_grad(set_to_none=True)
        task, total = _objective_loss_from_weight(weight, x, y, reg_lambda)
        total.backward()
        eval_count += 1
        trace.append(
            {
                "objective_eval": float(eval_count),
                "train_task_ce": float(task.detach().cpu()),
                "train_total_objective": float(total.detach().cpu()),
            }
        )
        return total

    optimizer.step(closure)
    weight_train = weight.detach().cpu().numpy().astype(np.float64)
    if sigma is None:
        weight_raw = weight_train.copy()
    else:
        weight_raw = _fold_scale_weight(weight_train, sigma.astype(np.float64))
    optimization = {
        "objective_evals": int(eval_count),
        "lbfgs_max_eval": int(max_eval),
        "line_search": "strong_wolfe",
        "history_size": LBFGS_HISTORY_SIZE,
    }
    return weight_train, weight_raw, sigma, optimization, trace


def run_lbfgs_noreg(
    spec: BaseSpec, config: Config, force: bool = False
) -> dict[str, Any]:
    validate_base(spec)
    case = "C2_lbfgs_noreg"
    eval_path = _path(config.results_dir, "evaluations", f"{spec.key}__{case}", ".json")
    if eval_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))
    cache = _load_cache(config.repo_root, spec)
    weight_train, weight_raw, sigma, optimization, trace = _run_lbfgs_weight(
        spec, cache, 0.0, False, LBFGS_MAX_EVAL, config
    )
    trace_path = _path(config.results_dir, "histories", f"{spec.key}__{case}", ".csv")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace).to_csv(trace_path, index=False)
    optimization["history_path"] = str(trace_path.relative_to(config.repo_root))
    payload = _payload_from_weight(
        spec,
        case,
        cache,
        weight_train,
        weight_raw,
        "lbfgs",
        0.0,
        sigma,
        {"unit": "full_objective_evaluations", "limit": LBFGS_MAX_EVAL},
        optimization,
    )
    _save_json(eval_path, payload)
    return payload


def run_candidate(
    spec: CandidateSpec, config: Config, force: bool = False
) -> dict[str, Any]:
    validate_candidate(spec)
    eval_path = _path(config.results_dir, "candidates", spec.key, ".json")
    if eval_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))
    cache = _load_cache(config.repo_root, spec.base)
    use_scale = spec.family in {"C4_scale_lbfgs_reg", "Ref_scale_lbfgs_converged"}
    max_eval = REF_MAX_EVAL if spec.family == "Ref_scale_lbfgs_converged" else LBFGS_MAX_EVAL
    weight_train, weight_raw, sigma, optimization, trace = _run_lbfgs_weight(
        spec.base,
        cache,
        spec.reg_lambda,
        use_scale,
        max_eval,
        config,
    )
    trace_path = _path(config.results_dir, "histories", spec.key, ".csv")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace).to_csv(trace_path, index=False)
    optimization["history_path"] = str(trace_path.relative_to(config.repo_root))
    payload = _payload_from_weight(
        spec.base,
        spec.family,
        cache,
        weight_train,
        weight_raw,
        "lbfgs",
        spec.reg_lambda,
        sigma,
        {"unit": "full_objective_evaluations", "limit": max_eval},
        optimization,
    )
    payload["candidate_key"] = spec.key
    _save_json(eval_path, payload)
    return payload


def _load_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_candidate(config: Config, base: BaseSpec, family: str) -> dict[str, Any]:
    candidates = [
        _load_required(_path(config.results_dir, "candidates", spec.key, ".json"))
        for spec in candidate_specs(family)
        if spec.base == base
    ]
    if len(candidates) != len(REG_LAMBDAS):
        raise RuntimeError(f"Incomplete {family} sweep for {base.key}")
    candidates.sort(
        key=lambda payload: (
            -float(payload["cross_evaluation"]["val"]["analog"]["balanced_accuracy"]),
            float(payload["objective_metrics"]["val"]["task_ce"]),
            float(payload["reg_lambda"]),
        )
    )
    selected = dict(candidates[0])
    selected["selection"] = {
        "criterion": "highest validation Analog BA; validation task CE then smaller lambda tiebreak",
        "candidate_lambdas": list(REG_LAMBDAS),
        "selected_lambda": float(selected["reg_lambda"]),
    }
    return selected


def _row(payload: dict[str, Any]) -> dict[str, Any]:
    spec = payload["spec"]
    analog = payload["cross_evaluation"]["test"]["analog"]["balanced_accuracy"]
    lif = payload["cross_evaluation"]["test"]["lif_beta05"]["balanced_accuracy"]
    return {
        "seed": int(spec["seed"]),
        "backbone_objective": spec["backbone_objective"],
        "head_objective": spec["objective"],
        "case": payload["case"],
        "optimizer": payload["optimizer"],
        "reg_lambda": float(payload["reg_lambda"]),
        "scaling_mode": payload["scaling"]["mode"],
        "budget_unit": payload["budget"]["unit"],
        "budget_limit": int(payload["budget"]["limit"]),
        "objective_evals": payload["optimization"].get("objective_evals"),
        "selected_epoch": payload["optimization"].get("selected_epoch"),
        "train_task_ce": float(payload["objective_metrics"]["train"]["task_ce"]),
        "val_task_ce": float(payload["objective_metrics"]["val"]["task_ce"]),
        "train_total_objective": float(
            payload["objective_metrics"]["train"]["total_objective"]
        ),
        "val_total_objective": float(
            payload["objective_metrics"]["val"]["total_objective"]
        ),
        "analog_test_ba": float(analog),
        "lif_test_ba": float(lif),
        "analog_to_lif_gap_pp": 100.0 * (float(analog) - float(lif)),
        "legacy_l2_wholecount_probe_test_ba": float(
            payload["legacy_l2_wholecount_probe_test_ba"]
        ),
        "legacy_l2_fixed250_probe_test_ba": float(
            payload["legacy_l2_fixed250_probe_test_ba"]
        ),
        "weight_norm_raw": float(payload["weight_norm_raw"]),
    }


def _contrast_frames(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for backbone in BACKBONE_OBJECTIVES:
        for objective in HEAD_OBJECTIVES:
            subset = runs[
                (runs.backbone_objective == backbone) & (runs.head_objective == objective)
            ]
            pivot = subset.pivot(index="seed", columns="case", values=["analog_test_ba", "lif_test_ba"])
            for left, right, name in (
                ("C1_adam500", "C0_adam100", "C1_minus_C0_extra_adam_budget"),
                ("C2_lbfgs_noreg", "C0_adam100", "C2_minus_C0_lbfgs_vs_adam100"),
                ("C3_lbfgs_reg", "C2_lbfgs_noreg", "C3_minus_C2_regularization"),
                ("C4_scale_lbfgs_reg", "C3_lbfgs_reg", "C4_minus_C3_feature_scaling"),
                ("Ref_scale_lbfgs_converged", "C4_scale_lbfgs_reg", "Ref_minus_C4_convergence_budget"),
            ):
                for seed in SEEDS:
                    rows.append(
                        {
                            "contrast": name,
                            "backbone_objective": backbone,
                            "head_objective": objective,
                            "seed": seed,
                            "left_case": left,
                            "right_case": right,
                            "analog_delta_pp": 100.0
                            * (float(pivot.loc[seed, ("analog_test_ba", left)]) - float(pivot.loc[seed, ("analog_test_ba", right)])),
                            "lif_delta_pp": 100.0
                            * (float(pivot.loc[seed, ("lif_test_ba", left)]) - float(pivot.loc[seed, ("lif_test_ba", right)])),
                        }
                    )
    for backbone in BACKBONE_OBJECTIVES:
        for case in CASES:
            subset = runs[(runs.backbone_objective == backbone) & (runs.case == case)]
            pivot = subset.pivot(index="seed", columns="head_objective", values=["analog_test_ba", "lif_test_ba"])
            for seed in SEEDS:
                rows.append(
                    {
                        "contrast": "wcce_minus_tsce_objective",
                        "backbone_objective": backbone,
                        "head_objective": "wcce_minus_tsce",
                        "seed": seed,
                        "left_case": case,
                        "right_case": case,
                        "analog_delta_pp": 100.0
                        * (float(pivot.loc[seed, ("analog_test_ba", "wcce")]) - float(pivot.loc[seed, ("analog_test_ba", "tsce")])),
                        "lif_delta_pp": 100.0
                        * (float(pivot.loc[seed, ("lif_test_ba", "wcce")]) - float(pivot.loc[seed, ("lif_test_ba", "tsce")])),
                    }
                )
    contrast_runs = pd.DataFrame(rows)
    contrast_summary = (
        contrast_runs.groupby(
            ["contrast", "backbone_objective", "head_objective", "left_case", "right_case"],
            sort=False,
        )[["analog_delta_pp", "lif_delta_pp"]]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    contrast_summary.columns = [
        "_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col)
        for col in contrast_summary.columns
    ]
    return contrast_runs, contrast_summary


def finalize(config: Config) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for base in base_specs():
        for case in ("C0_adam100", "C1_adam500", "C2_lbfgs_noreg"):
            payload = _load_required(
                _path(config.results_dir, "evaluations", f"{base.key}__{case}", ".json")
            )
            rows.append(_row(payload))
        for family in ("C3_lbfgs_reg", "C4_scale_lbfgs_reg", "Ref_scale_lbfgs_converged"):
            selected = _selected_candidate(config, base, family)
            selected_path = _path(config.results_dir, "selected", f"{base.key}__{family}", ".json")
            _save_json(selected_path, selected)
            rows.append(_row(selected))

    runs = pd.DataFrame(rows)
    expected = len(SEEDS) * len(BACKBONE_OBJECTIVES) * len(HEAD_OBJECTIVES) * len(CASES)
    if len(runs) != expected:
        raise RuntimeError(f"Expected {expected} logical rows, got {len(runs)}")
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "method_runs.csv", index=False)

    group_cols = ["backbone_objective", "head_objective", "case", "optimizer", "scaling_mode"]
    metric_cols = [
        "reg_lambda",
        "train_task_ce",
        "val_task_ce",
        "analog_test_ba",
        "lif_test_ba",
        "analog_to_lif_gap_pp",
        "legacy_l2_wholecount_probe_test_ba",
        "legacy_l2_fixed250_probe_test_ba",
        "weight_norm_raw",
    ]
    summary = runs.groupby(group_cols, sort=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(config.results_dir / "method_summary.csv", index=False)

    contrast_runs, contrast_summary = _contrast_frames(runs)
    contrast_runs.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_summary.to_csv(config.results_dir / "contrast_summary.csv", index=False)

    selected_lambdas = runs[runs.case.isin({"C3_lbfgs_reg", "C4_scale_lbfgs_reg", "Ref_scale_lbfgs_converged"})][
        ["seed", "backbone_objective", "head_objective", "case", "reg_lambda"]
    ].copy()
    selected_lambdas.to_csv(config.results_dir / "selected_regularization.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "source_experiment": exp73.EXPERIMENT_ID,
        "source_protocol": exp73.PROTOCOL_VERSION,
        "seeds": list(SEEDS),
        "backbone_objectives": list(BACKBONE_OBJECTIVES),
        "head_objectives": list(HEAD_OBJECTIVES),
        "cases": list(CASES),
        "reg_lambdas": list(REG_LAMBDAS),
        "adam_epochs": ADAM_EPOCHS,
        "adam_c0_epoch_limit": ADAM_C0_EPOCH_LIMIT,
        "lbfgs_max_eval": LBFGS_MAX_EVAL,
        "ref_max_eval": REF_MAX_EVAL,
        "logical_rows": expected,
        "physical_tasks": {
            "adam_shared_C0_C1": len(base_specs()),
            "C2_lbfgs_noreg": len(base_specs()),
            "C3_candidates": len(candidate_specs("C3_lbfgs_reg")),
            "C4_candidates": len(candidate_specs("C4_scale_lbfgs_reg")),
            "Ref_candidates": len(candidate_specs("Ref_scale_lbfgs_converged")),
        },
        "selection_rule": "C3/C4/Ref select lambda by highest validation Analog BA, then validation task CE, then smaller lambda; test is untouched until reporting.",
        "deployment_contract": "Every selected Linear W is evaluated unchanged through the same beta=0.5, threshold=0.5, cap=1 output LIF spike-count simulator; no LIF retraining or gain/threshold/beta sweep.",
        "legacy_probe_note": "Exp7.3 StandardScaler+intercept probes are reported only as auxiliary accessibility references, not as exact same-W LIF-deployable heads.",
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp7.3.1 frozen-L2 Linear optimization and same-W LIF substitution")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("adam", "lbfgs-noreg"):
        child = sub.add_parser(name)
        child.add_argument("--array-task-id", type=int, required=True)
    for name in ("c3", "c4", "ref"):
        child = sub.add_parser(name)
        child.add_argument("--array-task-id", type=int, required=True)
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    config = Config(root, results_dir(root), args.device, args.batch_size, args.threads)
    if args.command == "adam":
        specs = base_specs()
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        run_adam(specs[args.array_task_id], config, force=args.force)
        return
    if args.command == "lbfgs-noreg":
        specs = base_specs()
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        run_lbfgs_noreg(specs[args.array_task_id], config, force=args.force)
        return
    if args.command in {"c3", "c4", "ref"}:
        family = {
            "c3": "C3_lbfgs_reg",
            "c4": "C4_scale_lbfgs_reg",
            "ref": "Ref_scale_lbfgs_converged",
        }[args.command]
        specs = candidate_specs(family)
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        run_candidate(specs[args.array_task_id], config, force=args.force)
        return
    if args.command == "finalize":
        finalize(config)
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
