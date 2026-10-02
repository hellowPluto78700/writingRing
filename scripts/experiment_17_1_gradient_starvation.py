#!/usr/bin/env python3
"""Exp17.1: persistent-route gradient starvation and low-rate rescue."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from core_benchmark_v1.protocol import Protocol, paired_seed
from core_benchmark_v1.storage import load_torch, save_json, save_torch
from core_benchmark_v1.training import cpu_state, metrics
from scripts import experiment_16_3_run_reward as exp16_3
from scripts import experiment_16_prefix_supervised_selective_memory as exp16

EXPERIMENT_ID = "experiment_17_1_gradient_starvation"
PROTOCOL_VERSION = "gradient_starvation_v1"
FORMAL_SEEDS = (11, 23, 37)
CALIBRATION_SEED = 101
BRANCH_EPOCH = 20
GROUP_FRACTION = 0.25
CASES = ("C0", "C1", "C2", "C3")
CASE_LABELS = {"C0": "baseline", "C1": "low_rate_aux", "C2": "high_rate_aux", "C3": "random_aux"}
GRADIENT_RATIO_TARGET = 1.0
LAMBDA_MIN = 1e-4
LAMBDA_MAX = 100.0
DIAGNOSTIC_EVERY = 5


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    core_results_dir: Path


@dataclass(frozen=True)
class FormalSpec:
    case: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.case}__seed{self.seed}"


def find_repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in (path, *path.parents):
        if (parent / "AGENTS.md").exists() and (parent / "core_benchmark_v1").exists():
            return parent
    raise FileNotFoundError("Repository root not found")


def default_results_dir(root: Path) -> Path:
    return root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def config_from_args(args: argparse.Namespace) -> Config:
    root = find_repo_root()
    return Config(
        repo_root=root,
        results_dir=(args.results or default_results_dir(root)).resolve(),
        core_results_dir=(args.core_results or root / exp16.CORE_RESULTS_REL).resolve(),
    )


def formal_specs() -> list[FormalSpec]:
    return [FormalSpec(case, seed) for seed in FORMAL_SEEDS for case in CASES]


def _core(config: Config) -> tuple[Protocol, dict[str, Any], dict[str, np.ndarray]]:
    return exp16._load_core(config)


def _base_spec(seed: int) -> exp16_3.ExpSpec:
    return exp16_3.ExpSpec("LIN", seed, "linear", None)


def _make_model(seed: int, p: Protocol) -> nn.Module:
    return exp16_3._make_model(_base_spec(seed), p)


def _aux_head(seed: int, p: Protocol, tag: str) -> nn.Linear:
    head = nn.Linear(p.width, len(p.labels), bias=False)
    generator = torch.Generator().manual_seed(paired_seed(seed, f"exp17_1:aux_head:{tag}"))
    bound = 1.0 / math.sqrt(p.width)
    with torch.no_grad():
        head.weight.uniform_(-bound, bound, generator=generator)
    return head


def _feature(spikes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = (
        torch.arange(spikes.shape[1], device=spikes.device)[None, :] < lengths[:, None]
    ).unsqueeze(-1)
    return (spikes * mask).sum(1) / lengths.to(spikes.dtype)[:, None]


def _main_logits(model: nn.Module, out: dict[str, Any], lengths: torch.Tensor) -> torch.Tensor:
    return model.head(_feature(out["spike"][1], lengths))


def _subset_feature(out: dict[str, Any], lengths: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    feature = _feature(out["spike"][1], lengths)
    index = torch.as_tensor(indices, dtype=torch.long, device=feature.device)
    masked = torch.zeros_like(feature)
    masked[:, index] = feature[:, index]
    return masked


def _occupancy_from_arrays(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
    split: str = "train",
) -> np.ndarray:
    model.eval()
    total = np.zeros(p.width, dtype=np.float64)
    valid_steps = 0
    with torch.no_grad():
        for x, _y, lengths in exp16.loader(arrays, split, p, seed):
            out = model(x, lengths)
            spikes = out["spike"][1]
            mask = (
                torch.arange(spikes.shape[1])[None, :] < lengths[:, None]
            ).unsqueeze(-1)
            total += (spikes * mask).sum((0, 1)).cpu().numpy()
            valid_steps += int(lengths.sum())
    if valid_steps <= 0:
        raise RuntimeError("No valid timesteps for occupancy")
    return total / valid_steps


def _group_size(width: int) -> int:
    return max(1, int(math.ceil(GROUP_FRACTION * width)))


def _group_indices(occupancy: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    if occupancy.ndim != 1:
        raise ValueError("occupancy must be one-dimensional")
    k = _group_size(len(occupancy))
    order = np.argsort(occupancy, kind="mergesort")
    low = np.sort(order[:k])
    high = np.sort(order[-k:])
    generator = np.random.default_rng(paired_seed(seed, "exp17_1:random_group"))
    random = np.sort(generator.choice(len(occupancy), size=k, replace=False))
    return {"low": low, "high": high, "random": random}


def _group_for_case(groups: dict[str, np.ndarray], case: str) -> np.ndarray | None:
    mapping = {"C1": "low", "C2": "high", "C3": "random"}
    if case == "C0":
        return None
    if case not in mapping:
        raise ValueError(case)
    return groups[mapping[case]]


def _new_optimizer(model: nn.Module, p: Protocol, aux: nn.Module | None = None):
    params = list(model.parameters()) + ([] if aux is None else list(aux.parameters()))
    return torch.optim.Adam(params, lr=p.learning_rate, weight_decay=p.weight_decay)


def _train_to_epoch(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
    start_epoch: int,
    end_epoch: int,
) -> list[dict[str, float]]:
    optimizer = _new_optimizer(model, p)
    history = []
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for x, y, lengths in exp16_3._train_batches(arrays, p, seed, epoch):
            optimizer.zero_grad(set_to_none=True)
            out = model(x, lengths)
            loss = F.cross_entropy(_main_logits(model, out, lengths), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            n += len(y)
        history.append({"epoch": epoch, "train_loss": loss_sum / n})
    return history


def _bootstrap_path(config: Config, seed: int) -> Path:
    return config.results_dir / "bootstrap" / f"seed{seed}" / "epoch020.pt"


def _bootstrap(config: Config, seed: int) -> dict[str, Any]:
    p, _, arrays = _core(config)
    path = _bootstrap_path(config, seed)
    groups_path = path.parent / "groups.json"
    if path.exists() and groups_path.exists():
        return {"status": "exists", "seed": seed}
    model = _make_model(seed, p)
    optimizer = _new_optimizer(model, p)
    history = []
    for epoch in range(1, BRANCH_EPOCH + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for x, y, lengths in exp16_3._train_batches(arrays, p, seed, epoch):
            optimizer.zero_grad(set_to_none=True)
            out = model(x, lengths)
            loss = F.cross_entropy(_main_logits(model, out, lengths), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            n += len(y)
        history.append({"epoch": epoch, "train_loss": loss_sum / n})
    occupancy = _occupancy_from_arrays(model, arrays, p, seed, "train")
    groups = _group_indices(occupancy, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_torch(path, {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "seed": seed,
        "epoch": BRANCH_EPOCH,
        "model_state_dict": cpu_state(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "shared_init_hash": exp16_3._shared_init_hash(seed, p),
    })
    save_json(groups_path, {
        "seed": seed,
        "epoch": BRANCH_EPOCH,
        "group_fraction": GROUP_FRACTION,
        "occupancy": occupancy.tolist(),
        "low": groups["low"].tolist(),
        "high": groups["high"].tolist(),
        "random": groups["random"].tolist(),
        "source": "training-split L2 mean occupancy only",
    })
    save_json(path.parent / "history.json", {"rows": history})
    return {"status": "PASS", "seed": seed}


def _load_bootstrap(config: Config, seed: int, p: Protocol):
    ckpt = load_torch(_bootstrap_path(config, seed))
    model = _make_model(seed, p)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    raw = json.loads((_bootstrap_path(config, seed).parent / "groups.json").read_text())
    groups = {key: np.asarray(raw[key], dtype=np.int64) for key in ("low", "high", "random")}
    return model, groups, ckpt["optimizer_state_dict"]


def _gradient_pair(
    model: nn.Module,
    aux: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    indices: np.ndarray,
) -> tuple[float, float, float]:
    model.zero_grad(set_to_none=True)
    aux.zero_grad(set_to_none=True)
    out = model(x, lengths)
    F.cross_entropy(_main_logits(model, out, lengths), y).backward()
    main_grad = model.layers[1].weight.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    aux.zero_grad(set_to_none=True)
    out = model(x, lengths)
    F.cross_entropy(aux(_subset_feature(out, lengths, indices)), y).backward()
    aux_grad = model.layers[1].weight.grad.detach().clone()

    idx = torch.as_tensor(indices, dtype=torch.long)
    g_main = main_grad[idx].reshape(-1)
    g_aux = aux_grad[idx].reshape(-1)
    n_main = float(g_main.norm())
    n_aux = float(g_aux.norm())
    cosine = float(torch.dot(g_main, g_aux) / max(n_main * n_aux, 1e-12))
    model.zero_grad(set_to_none=True)
    aux.zero_grad(set_to_none=True)
    return n_main, n_aux, cosine


def _calibrate(config: Config) -> dict[str, Any]:
    p, _, arrays = _core(config)
    seed = CALIBRATION_SEED
    model = _make_model(seed, p)
    _train_to_epoch(model, arrays, p, seed, 0, BRANCH_EPOCH)
    occupancy = _occupancy_from_arrays(model, arrays, p, seed, "train")
    groups = _group_indices(occupancy, seed)
    aux = _aux_head(seed, p, "calibration_low")

    rows = []
    for batch_idx, (x, y, lengths) in enumerate(
        exp16_3._train_batches(arrays, p, seed, BRANCH_EPOCH + 1)
    ):
        n_main, n_aux, cosine = _gradient_pair(model, aux, x, y, lengths, groups["low"])
        rows.append({
            "batch": batch_idx,
            "main_low_grad_norm": n_main,
            "aux_low_grad_norm": n_aux,
            "cosine": cosine,
        })
        if batch_idx >= 7:
            break
    main = float(np.mean([row["main_low_grad_norm"] for row in rows]))
    aux_norm = float(np.mean([row["aux_low_grad_norm"] for row in rows]))
    raw_lambda = GRADIENT_RATIO_TARGET * main / max(aux_norm, 1e-12)
    selected = float(np.clip(raw_lambda, LAMBDA_MIN, LAMBDA_MAX))
    payload = {
        "seed": seed,
        "branch_epoch": BRANCH_EPOCH,
        "target_gradient_ratio": GRADIENT_RATIO_TARGET,
        "mean_main_low_grad_norm": main,
        "mean_aux_low_grad_norm": aux_norm,
        "raw_lambda": raw_lambda,
        "selected_lambda": selected,
        "lambda_bounds": [LAMBDA_MIN, LAMBDA_MAX],
        "selection_uses_performance": False,
        "batches": rows,
    }
    path = config.results_dir / "calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, payload)
    return payload


def _selected_lambda(config: Config) -> float:
    path = config.results_dir / "calibration.json"
    if not path.exists():
        raise FileNotFoundError("Run calibrate before formal continuation")
    return float(json.loads(path.read_text())["selected_lambda"])


def _split_eval(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
    split: str,
) -> dict[str, float]:
    model.eval()
    ys = []
    predictions = []
    losses = 0.0
    residual_norm_sum = 0.0
    margin_sum = 0.0
    count = 0
    with torch.no_grad():
        for x, y, lengths in exp16.loader(arrays, split, p, seed):
            out = model(x, lengths)
            logits = _main_logits(model, out, lengths)
            probabilities = logits.softmax(1)
            targets = F.one_hot(y, num_classes=logits.shape[1]).to(logits.dtype)
            residual_norm_sum += float((probabilities - targets).norm(dim=1).sum())
            true_logit = logits.gather(1, y[:, None]).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, y[:, None], -torch.inf)
            margin_sum += float((true_logit - masked.max(1).values).sum())
            ys.append(y.numpy())
            predictions.append(logits.argmax(1).numpy())
            losses += float(F.cross_entropy(logits, y, reduction="sum"))
            count += len(y)
    y_all = np.concatenate(ys)
    pred_all = np.concatenate(predictions)
    return {
        **metrics(y_all, pred_all),
        "ce": losses / count,
        "mean_residual_norm": residual_norm_sum / count,
        "mean_margin": margin_sum / count,
    }


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    overall = values.mean(axis=0)
    total = ((values - overall) ** 2).sum(axis=0)
    between = np.zeros(values.shape[1], dtype=np.float64)
    for label in np.unique(labels):
        group = values[labels == label]
        between += len(group) * (group.mean(axis=0) - overall) ** 2
    result = np.zeros_like(total)
    valid = total > 1e-12
    result[valid] = between[valid] / total[valid]
    return result


def _representation_stats(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    seed: int,
    groups: dict[str, np.ndarray],
) -> dict[str, float]:
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for x, y, lengths in exp16.loader(arrays, "train", p, seed):
            out = model(x, lengths)
            features.append(_feature(out["spike"][1], lengths).cpu().numpy())
            labels.append(y.numpy())
    feature = np.concatenate(features)
    y = np.concatenate(labels)
    eta = _eta_squared(feature, y)
    user_eta = _eta_squared(feature, arrays["train_users"])
    occupancy = feature.mean(axis=0)
    result = {}
    for name in ("low", "high", "random"):
        idx = groups[name]
        result[f"{name}_occupancy"] = float(occupancy[idx].mean())
        result[f"{name}_class_eta2"] = float(eta[idx].mean())
        result[f"{name}_user_eta2"] = float(user_eta[idx].mean())
        weights = model.head.weight.detach().cpu().numpy()[:, idx]
        result[f"{name}_head_norm"] = float(np.linalg.norm(weights))
        contribution = feature[:, idx] @ weights.T
        result[f"{name}_logit_norm"] = float(np.linalg.norm(contribution, axis=1).mean())
    result["occupancy_class_eta2_corr"] = (
        float(np.corrcoef(occupancy, eta)[0, 1])
        if np.std(occupancy) > 0 and np.std(eta) > 0 else 0.0
    )
    return result


def _run_formal(config: Config, spec: FormalSpec) -> dict[str, Any]:
    p, _, arrays = _core(config)
    model, groups, optimizer_state = _load_bootstrap(config, spec.seed, p)
    lam = _selected_lambda(config)
    selected = _group_for_case(groups, spec.case)
    aux = None if selected is None else _aux_head(spec.seed, p, spec.case)
    optimizer = _new_optimizer(model, p)
    optimizer.load_state_dict(optimizer_state)
    if aux is not None:
        optimizer.add_param_group({
            "params": list(aux.parameters()),
            "lr": p.learning_rate,
            "weight_decay": p.weight_decay,
        })

    directory = config.results_dir / "formal" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = _split_eval(model, arrays, p, spec.seed, "val")
    best_epoch = BRANCH_EPOCH
    best_state = cpu_state(model)

    for epoch in range(BRANCH_EPOCH + 1, p.max_epochs + 1):
        model.train()
        if aux is not None:
            aux.train()
        total = main_total = aux_total = 0.0
        n = 0
        grad_diag = None
        for batch_idx, (x, y, lengths) in enumerate(
            exp16_3._train_batches(arrays, p, spec.seed, epoch)
        ):
            if (
                selected is not None
                and batch_idx == 0
                and (epoch == BRANCH_EPOCH + 1 or epoch % DIAGNOSTIC_EVERY == 0)
            ):
                n_main, n_aux, cosine = _gradient_pair(model, aux, x, y, lengths, selected)
                effective_total = math.sqrt(max(
                    n_main * n_main
                    + (lam * n_aux) * (lam * n_aux)
                    + 2.0 * lam * n_main * n_aux * cosine,
                    0.0,
                ))
                grad_diag = {
                    "main_group_grad_norm": n_main,
                    "aux_group_grad_norm": n_aux,
                    "main_aux_grad_cosine": cosine,
                    "effective_group_grad_norm": effective_total,
                }

            optimizer.zero_grad(set_to_none=True)
            out = model(x, lengths)
            main_loss = F.cross_entropy(_main_logits(model, out, lengths), y)
            if aux is None:
                aux_loss = main_loss.new_zeros(())
                loss = main_loss
            else:
                aux_loss = F.cross_entropy(aux(_subset_feature(out, lengths, selected)), y)
                loss = main_loss + lam * aux_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            main_total += float(main_loss.detach()) * len(y)
            aux_total += float(aux_loss.detach()) * len(y)
            n += len(y)

        val = _split_eval(model, arrays, p, spec.seed, "val")
        row = {
            "epoch": epoch,
            "case": spec.case,
            "train_total_loss": total / n,
            "train_main_loss": main_total / n,
            "train_aux_loss": aux_total / n,
            "lambda_aux": 0.0 if aux is None else lam,
            "val_ba": val["ba"],
            "val_ce": val["ce"],
        }
        if grad_diag is not None:
            row.update(grad_diag)
        if epoch == BRANCH_EPOCH + 1 or epoch % DIAGNOSTIC_EVERY == 0:
            row.update(_representation_stats(model, arrays, p, spec.seed, groups))
        history.append(row)

        if (
            val["ba"] > best_val["ba"] + 1e-12
            or (
                abs(val["ba"] - best_val["ba"]) <= 1e-12
                and val["ce"] < best_val["ce"] - 1e-12
            )
        ):
            best_val = dict(val)
            best_epoch = epoch
            best_state = cpu_state(model)

        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    model.load_state_dict(best_state, strict=True)
    final = {
        "case": spec.case,
        "case_label": CASE_LABELS[spec.case],
        "seed": spec.seed,
        "branch_epoch": BRANCH_EPOCH,
        "best_epoch": best_epoch,
        "lambda_aux": 0.0 if aux is None else lam,
        "selection_rule": "validation BA, then validation CE, then earliest epoch",
        "train": _split_eval(model, arrays, p, spec.seed, "train"),
        "val": _split_eval(model, arrays, p, spec.seed, "val"),
        "test": _split_eval(model, arrays, p, spec.seed, "test"),
        "representation": _representation_stats(model, arrays, p, spec.seed, groups),
    }
    save_json(directory / "history.json", {"rows": history})
    save_json(directory / "metrics.json", final)
    save_torch(directory / "checkpoint.pt", {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "case": spec.case,
        "seed": spec.seed,
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "lambda_aux": final["lambda_aux"],
        "groups": {k: v.tolist() for k, v in groups.items()},
    })
    return final


def _aggregate(config: Config) -> dict[str, Any]:
    rows = []
    for spec in formal_specs():
        path = config.results_dir / "formal" / spec.key / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        rows.append({
            "case": spec.case,
            "seed": spec.seed,
            "best_epoch": payload["best_epoch"],
            "train_ba": payload["train"]["ba"],
            "val_ba": payload["val"]["ba"],
            "test_ba": payload["test"]["ba"],
            "low_occupancy": payload["representation"]["low_occupancy"],
            "high_occupancy": payload["representation"]["high_occupancy"],
            "low_class_eta2": payload["representation"]["low_class_eta2"],
            "high_class_eta2": payload["representation"]["high_class_eta2"],
            "low_user_eta2": payload["representation"]["low_user_eta2"],
            "high_user_eta2": payload["representation"]["high_user_eta2"],
            "low_head_norm": payload["representation"]["low_head_norm"],
            "high_head_norm": payload["representation"]["high_head_norm"],
            "low_logit_norm": payload["representation"]["low_logit_norm"],
            "high_logit_norm": payload["representation"]["high_logit_norm"],
            "occupancy_class_eta2_corr": payload["representation"]["occupancy_class_eta2_corr"],
        })
    summary = {"rows": rows, "cases": {}}
    for case in CASES:
        selected = [row for row in rows if row["case"] == case]
        summary["cases"][case] = {
            key: {
                "mean": float(np.mean([row[key] for row in selected])),
                "std": float(np.std([row[key] for row in selected], ddof=1)),
            }
            for key in (
                "train_ba", "val_ba", "test_ba", "low_occupancy",
                "high_occupancy", "low_class_eta2", "high_class_eta2",
                "low_user_eta2", "high_user_eta2",
                "low_head_norm", "high_head_norm", "low_logit_norm",
                "high_logit_norm", "occupancy_class_eta2_corr",
            )
        }
    out = config.results_dir / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate")
    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("--seed", type=int, required=True)
    run = sub.add_parser("run")
    run.add_argument("--case", choices=CASES, required=True)
    run.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    sub.add_parser("aggregate")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = config_from_args(args)
    if args.command == "calibrate":
        print(json.dumps(_calibrate(config), indent=2))
    elif args.command == "bootstrap":
        if args.seed not in FORMAL_SEEDS:
            raise ValueError(f"bootstrap seed must be one of {FORMAL_SEEDS}")
        print(json.dumps(_bootstrap(config, args.seed), indent=2))
    elif args.command == "run":
        print(json.dumps(_run_formal(config, FormalSpec(args.case, args.seed)), indent=2))
    elif args.command == "aggregate":
        print(json.dumps(_aggregate(config), indent=2))
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

[executed on device: acd20ea31325 (425a23ad-a806-44e3-abed-ce7b563d3969)]