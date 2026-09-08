from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts import experiment_5_5_1_regularized_semantic_when as exp551


base = exp551.base

EXPERIMENT_ID = "experiment_5_5_1_1_routing_diagnostics"
PROTOCOL_VERSION = "routing_diagnostics_v1"
SEEDS = exp551.SEEDS
TEMPERATURES = (1.0, 0.8, 0.6, 0.4, 0.25)
HARD_VARIANT = "hard_top1"
SHUFFLE_REPLICATES = 5
FUNCTIONAL_MAX_POINTS = 50_000
CONFIDENCE_THRESHOLDS = (0.3, 0.4, 0.5, 0.6)


def find_repo_root(start: Path | None = None) -> Path:
    return exp551.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / EXPERIMENT_ID
        / PROTOCOL_VERSION
    )


def source_results_dir(repo_root: Path) -> Path:
    return exp551.results_dir(repo_root)


def seed_path(root: Path, seed: int) -> Path:
    return root / "per_seed" / f"seed{seed}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def variant_name(temperature: float) -> str:
    return f"tau_{temperature:.2f}"


def transform_q(
    q: torch.Tensor,
    lengths: torch.Tensor,
    *,
    temperature: float | None = None,
    hard_top1: bool = False,
) -> torch.Tensor:
    if hard_top1 == (temperature is not None):
        raise ValueError("Specify exactly one of temperature or hard_top1")
    valid = exp551.sequence_mask(lengths, q.shape[1]).to(q.dtype).unsqueeze(-1)
    if hard_top1:
        index = q.argmax(dim=-1)
        transformed = F.one_hot(index, num_classes=q.shape[-1]).to(q.dtype)
    else:
        assert temperature is not None
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        transformed = torch.softmax(torch.log(q.clamp(min=1e-12)) / temperature, dim=-1)
    return transformed * valid


def confidence_metrics(q: np.ndarray, lengths: np.ndarray) -> dict[str, float | int]:
    valid_rows = [
        np.asarray(q[row, : int(length)], dtype=np.float64)
        for row, length in enumerate(lengths)
        if int(length) > 0
    ]
    if not valid_rows:
        raise ValueError("No valid q rows")
    values = np.concatenate(valid_rows, axis=0)
    ordered = np.sort(values, axis=1)
    max_q = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -(values * np.log(np.clip(values, 1e-12, None))).sum(axis=1)
    payload: dict[str, float | int] = {
        "n_timesteps": int(len(values)),
        "mean_max_q": float(max_q.mean()),
        "median_max_q": float(np.median(max_q)),
        "p10_max_q": float(np.quantile(max_q, 0.10)),
        "p25_max_q": float(np.quantile(max_q, 0.25)),
        "p75_max_q": float(np.quantile(max_q, 0.75)),
        "p90_max_q": float(np.quantile(max_q, 0.90)),
        "mean_top1_top2_margin": float(margin.mean()),
        "median_top1_top2_margin": float(np.median(margin)),
        "mean_entropy": float(entropy.mean()),
        "normalized_entropy": float(entropy.mean() / math.log(values.shape[1])),
        "mean_effective_states": float(np.exp(entropy).mean()),
    }
    for threshold in CONFIDENCE_THRESHOLDS:
        payload[f"fraction_max_q_ge_{threshold:.1f}"] = float(np.mean(max_q >= threshold))
    return payload


def _metrics_from_logits(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    return exp551.exp55._metrics_from_logits(labels, logits)


def _q_variant_rows(
    *,
    model: exp551.RegularizedSemanticWhen,
    trajectory: dict[str, np.ndarray],
    split: str,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    what = torch.tensor(trajectory["what"], dtype=torch.float32, device=device)
    q = torch.tensor(trajectory["q"], dtype=torch.float32, device=device)
    lengths = torch.tensor(trajectory["lengths"], dtype=torch.long, device=device)
    labels = np.asarray(trajectory["labels"], dtype=np.int64)
    metric_rows: list[dict[str, object]] = []
    shuffle_rows: list[dict[str, object]] = []

    variants: list[tuple[str, torch.Tensor]] = []
    for temperature in TEMPERATURES:
        variants.append(
            (
                variant_name(temperature),
                transform_q(q, lengths, temperature=temperature),
            )
        )
    variants.append((HARD_VARIANT, transform_q(q, lengths, hard_top1=True)))

    for name, q_variant in variants:
        with torch.no_grad():
            logits = model.logits_from_q(what, q_variant, lengths).cpu().numpy()
        q_numpy = q_variant.cpu().numpy()
        confidence = confidence_metrics(q_numpy, trajectory["lengths"])
        metric_rows.append(
            {
                "seed": seed,
                "split": split,
                "variant": name,
                **_metrics_from_logits(labels, logits),
                **confidence,
            }
        )

        for replicate in range(SHUFFLE_REPLICATES):
            rng = np.random.default_rng(
                base.dseed(seed, EXPERIMENT_ID, "shuffle", split, name, replicate)
            )
            shuffled = q_numpy.copy()
            for row, length_value in enumerate(trajectory["lengths"]):
                length = int(length_value)
                if length > 1:
                    shuffled[row, :length] = q_numpy[row, rng.permutation(length)]
            shuffled_tensor = torch.tensor(shuffled, dtype=torch.float32, device=device)
            with torch.no_grad():
                shuffled_logits = model.logits_from_q(what, shuffled_tensor, lengths).cpu().numpy()
            shuffle_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "variant": name,
                    "replicate": replicate,
                    **_metrics_from_logits(labels, shuffled_logits),
                }
            )
    return metric_rows, shuffle_rows


def expert_weight_similarity_rows(
    model: exp551.RegularizedSemanticWhen,
    seed: int,
) -> list[dict[str, object]]:
    weight = model.expert_weight.detach().cpu().numpy().astype(np.float64)
    flat = weight.reshape(weight.shape[0], -1)
    rows: list[dict[str, object]] = []
    for left in range(weight.shape[0]):
        for right in range(left + 1, weight.shape[0]):
            a = flat[left]
            b = flat[right]
            denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
            distance = float(np.linalg.norm(a - b))
            scale = max(0.5 * float(np.linalg.norm(a) + np.linalg.norm(b)), 1e-12)
            rows.append(
                {
                    "seed": seed,
                    "expert_a": left,
                    "expert_b": right,
                    "weight_cosine_similarity": float(np.dot(a, b) / denom),
                    "weight_l2_distance": distance,
                    "weight_relative_l2_distance": distance / scale,
                }
            )
    return rows


def _sample_active_what(
    trajectory: dict[str, np.ndarray],
    *,
    seed: int,
    split: str,
) -> tuple[np.ndarray, float]:
    rows: list[np.ndarray] = []
    valid_count = 0
    active_count = 0
    for sample, length_value in enumerate(trajectory["lengths"]):
        length = int(length_value)
        values = trajectory["what"][sample, :length].astype(np.float32)
        valid_count += length
        active = values[np.linalg.norm(values, axis=1) > 0.0]
        active_count += len(active)
        if len(active):
            rows.append(active)
    if not rows:
        raise ValueError(f"No active WHAT vectors for seed={seed} split={split}")
    values = np.concatenate(rows, axis=0)
    if len(values) > FUNCTIONAL_MAX_POINTS:
        rng = np.random.default_rng(base.dseed(seed, EXPERIMENT_ID, "functional", split))
        idx = np.sort(rng.choice(len(values), size=FUNCTIONAL_MAX_POINTS, replace=False))
        values = values[idx]
    return values, float(active_count / max(valid_count, 1))


def expert_functional_diversity_rows(
    model: exp551.RegularizedSemanticWhen,
    trajectory: dict[str, np.ndarray],
    *,
    seed: int,
    split: str,
) -> list[dict[str, object]]:
    what, active_fraction = _sample_active_what(trajectory, seed=seed, split=split)
    weight = model.expert_weight.detach().cpu().numpy().astype(np.float64)
    evidence = np.einsum("nd,kcd->nkc", what.astype(np.float64), weight)
    rows: list[dict[str, object]] = []
    for left in range(weight.shape[0]):
        for right in range(left + 1, weight.shape[0]):
            a = evidence[:, left]
            b = evidence[:, right]
            diff = a - b
            diff_l2 = np.linalg.norm(diff, axis=1)
            a_l2 = np.linalg.norm(a, axis=1)
            b_l2 = np.linalg.norm(b, axis=1)
            denom = np.maximum(a_l2 * b_l2, 1e-12)
            cosine = (a * b).sum(axis=1) / denom
            scale = np.maximum(0.5 * (a_l2 + b_l2), 1e-12)
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "expert_a": left,
                    "expert_b": right,
                    "n_active_what": int(len(what)),
                    "active_what_fraction": active_fraction,
                    "mean_evidence_l1_difference": float(np.abs(diff).sum(axis=1).mean()),
                    "mean_evidence_l2_difference": float(diff_l2.mean()),
                    "mean_relative_evidence_l2_difference": float((diff_l2 / scale).mean()),
                    "mean_evidence_cosine_similarity": float(cosine.mean()),
                    "top_support_disagreement_rate": float(
                        np.mean(a.argmax(axis=1) != b.argmax(axis=1))
                    ),
                }
            )
    return rows


def _load_selected_ordered(
    *,
    seed: int,
    data: Any,
    config: exp551.Config,
) -> tuple[exp551.RegularizedSemanticWhen, dict[str, object], str]:
    selection = exp551.load_selection(config)
    recipe_name = str(selection["selected_recipe"])
    recipe = exp551.RECIPES[recipe_name]
    spec = exp551.ScreenSpec(recipe_name, seed)
    model, checkpoint = exp551._load_checkpoint(
        exp551.screen_checkpoint_path(config.results_dir, spec),
        seed=seed,
        mode=exp551.ORDERED,
        recipe=recipe,
        data=data,
        config=config,
    )
    return model, checkpoint, recipe_name


def run_seed(
    seed: int,
    data: Any,
    config: exp551.Config,
    output_root: Path,
    *,
    force: bool = False,
) -> Path:
    if seed not in SEEDS:
        raise ValueError(seed)
    destination = seed_path(output_root, seed)
    if destination.exists() and not force:
        return destination

    final_path = exp551.final_evaluation_path(config.results_dir, seed)
    if not final_path.exists():
        raise FileNotFoundError(
            f"Exp5.5.1.1 requires finalized Exp5.5.1 seed artifact: {final_path}"
        )
    final_payload = json.loads(final_path.read_text(encoding="utf-8"))
    if final_payload.get("test_used_for_recipe_selection") is not False:
        raise ValueError(f"Invalid Exp5.5.1 source final artifact: {final_path}")

    model, checkpoint, recipe_name = _load_selected_ordered(seed=seed, data=data, config=config)
    what, _ = exp551._load_what_splits(seed, data, config, ("val", "test"))
    loaders = exp551._make_loaders(
        what,
        data,
        seed,
        config,
        ("val", "test"),
        False,
    )
    device = torch.device(config.device)
    trajectories = {
        split: exp551._collect_trajectory(model, loaders[split], device)
        for split in ("val", "test")
    }

    confidence_rows: list[dict[str, object]] = []
    temperature_rows: list[dict[str, object]] = []
    shuffle_rows: list[dict[str, object]] = []
    functional_rows: list[dict[str, object]] = []
    for split in ("val", "test"):
        confidence_rows.append(
            {
                "seed": seed,
                "split": split,
                **confidence_metrics(trajectories[split]["q"], trajectories[split]["lengths"]),
            }
        )
        variant_rows, variant_shuffle_rows = _q_variant_rows(
            model=model,
            trajectory=trajectories[split],
            split=split,
            seed=seed,
            device=device,
        )
        temperature_rows.extend(variant_rows)
        shuffle_rows.extend(variant_shuffle_rows)
        functional_rows.extend(
            expert_functional_diversity_rows(
                model,
                trajectories[split],
                seed=seed,
                split=split,
            )
        )

    tau_one_test = next(
        row
        for row in temperature_rows
        if row["split"] == "test" and row["variant"] == variant_name(1.0)
    )
    source_test_ba = float(final_payload["ordered"]["test"]["balanced_accuracy"])
    delta = abs(float(tau_one_test["balanced_accuracy"]) - source_test_ba)
    if delta > 1e-12:
        raise RuntimeError(
            f"tau=1 diagnostic changed source predictions for seed {seed}: BA delta={delta:.3e}"
        )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "source_experiment_id": exp551.EXPERIMENT_ID,
        "source_protocol_version": exp551.PROTOCOL_VERSION,
        "selected_recipe": recipe_name,
        "checkpoint_model_seed": checkpoint["model_seed"],
        "diagnostic_only": True,
        "no_training": True,
        "no_temperature_selection": True,
        "test_used_for_model_or_recipe_selection": False,
        "tau1_source_test_ba_abs_delta": delta,
        "confidence": confidence_rows,
        "temperature_sweep": temperature_rows,
        "temperature_shuffled_q": shuffle_rows,
        "expert_weight_similarity": expert_weight_similarity_rows(model, seed),
        "expert_functional_diversity": functional_rows,
    }
    _save_json(destination, payload)
    return destination


def _sem(values: np.ndarray) -> float:
    return 0.0 if len(values) <= 1 else float(values.std(ddof=1) / math.sqrt(len(values)))


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    payloads: list[dict[str, object]] = []
    for seed in SEEDS:
        path = seed_path(root, seed)
        if not path.exists():
            raise FileNotFoundError(f"Finalizer will not run missing diagnostic seed: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("seed") != seed
            or payload.get("no_training") is not True
            or payload.get("no_temperature_selection") is not True
        ):
            raise ValueError(f"Invalid Exp5.5.1.1 seed artifact: {path}")
        payloads.append(payload)

    confidence = pd.DataFrame(
        [row for payload in payloads for row in payload["confidence"]]
    )
    temperature = pd.DataFrame(
        [row for payload in payloads for row in payload["temperature_sweep"]]
    )
    shuffled = pd.DataFrame(
        [row for payload in payloads for row in payload["temperature_shuffled_q"]]
    )
    weight_similarity = pd.DataFrame(
        [row for payload in payloads for row in payload["expert_weight_similarity"]]
    )
    functional = pd.DataFrame(
        [row for payload in payloads for row in payload["expert_functional_diversity"]]
    )

    tau_one = temperature[temperature["variant"] == variant_name(1.0)][
        ["seed", "split", "balanced_accuracy"]
    ].rename(columns={"balanced_accuracy": "tau1_balanced_accuracy"})
    temperature = temperature.merge(tau_one, on=["seed", "split"], how="left")
    temperature["delta_balanced_accuracy_vs_tau1"] = (
        temperature["balanced_accuracy"] - temperature["tau1_balanced_accuracy"]
    )

    shuffled_mean = (
        shuffled.groupby(["seed", "split", "variant"], as_index=False)["balanced_accuracy"]
        .mean()
        .rename(columns={"balanced_accuracy": "mean_shuffled_balanced_accuracy"})
    )
    alignment = temperature[
        ["seed", "split", "variant", "balanced_accuracy"]
    ].merge(shuffled_mean, on=["seed", "split", "variant"], how="left")
    alignment["ordered_minus_shuffled_balanced_accuracy"] = (
        alignment["balanced_accuracy"] - alignment["mean_shuffled_balanced_accuracy"]
    )

    confidence_summary_rows: list[dict[str, object]] = []
    confidence_metrics_names = [
        "mean_max_q",
        "median_max_q",
        "mean_top1_top2_margin",
        "median_top1_top2_margin",
        "mean_entropy",
        "normalized_entropy",
        "mean_effective_states",
        *(f"fraction_max_q_ge_{threshold:.1f}" for threshold in CONFIDENCE_THRESHOLDS),
    ]
    for split in ("val", "test"):
        group = confidence[confidence["split"] == split]
        for metric in confidence_metrics_names:
            values = group[metric].to_numpy(dtype=float)
            confidence_summary_rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "sem": _sem(values),
                }
            )

    temperature_summary_rows: list[dict[str, object]] = []
    for (split, variant), group in temperature.groupby(["split", "variant"], sort=False):
        ba = group["balanced_accuracy"].to_numpy(dtype=float)
        delta = group["delta_balanced_accuracy_vs_tau1"].to_numpy(dtype=float)
        temperature_summary_rows.append(
            {
                "split": split,
                "variant": variant,
                "mean_balanced_accuracy": float(ba.mean()),
                "sd_balanced_accuracy": float(ba.std(ddof=1)),
                "sem_balanced_accuracy": _sem(ba),
                "mean_delta_balanced_accuracy_vs_tau1": float(delta.mean()),
                "nonnegative_delta_seed_count": int(np.count_nonzero(delta >= 0.0)),
                "mean_max_q": float(group["mean_max_q"].mean()),
                "mean_top1_top2_margin": float(group["mean_top1_top2_margin"].mean()),
                "mean_entropy": float(group["mean_entropy"].mean()),
                "mean_effective_states": float(group["mean_effective_states"].mean()),
            }
        )

    expert_summary_rows: list[dict[str, object]] = []
    weight_metrics = (
        "weight_cosine_similarity",
        "weight_l2_distance",
        "weight_relative_l2_distance",
    )
    for metric in weight_metrics:
        per_seed = weight_similarity.groupby("seed")[metric].mean().to_numpy(dtype=float)
        expert_summary_rows.append(
            {
                "scope": "weight",
                "split": "all",
                "metric": metric,
                "mean": float(per_seed.mean()),
                "sd": float(per_seed.std(ddof=1)),
                "sem": _sem(per_seed),
            }
        )
    functional_metrics = (
        "mean_evidence_l1_difference",
        "mean_evidence_l2_difference",
        "mean_relative_evidence_l2_difference",
        "mean_evidence_cosine_similarity",
        "top_support_disagreement_rate",
        "active_what_fraction",
    )
    for split in ("val", "test"):
        frame = functional[functional["split"] == split]
        for metric in functional_metrics:
            per_seed = frame.groupby("seed")[metric].mean().to_numpy(dtype=float)
            expert_summary_rows.append(
                {
                    "scope": "functional",
                    "split": split,
                    "metric": metric,
                    "mean": float(per_seed.mean()),
                    "sd": float(per_seed.std(ddof=1)),
                    "sem": _sem(per_seed),
                }
            )

    outputs = {
        "confidence_metrics": root / "confidence_metrics.csv",
        "confidence_summary": root / "confidence_summary.csv",
        "temperature_sweep": root / "temperature_sweep.csv",
        "temperature_summary": root / "temperature_summary.csv",
        "temperature_shuffled_q": root / "temperature_shuffled_q.csv",
        "temperature_alignment": root / "temperature_alignment.csv",
        "expert_weight_similarity": root / "expert_weight_similarity.csv",
        "expert_functional_diversity": root / "expert_functional_diversity.csv",
        "expert_summary": root / "expert_summary.csv",
        "manifest": root / "manifest.json",
    }
    confidence.to_csv(outputs["confidence_metrics"], index=False)
    pd.DataFrame(confidence_summary_rows).to_csv(outputs["confidence_summary"], index=False)
    temperature.to_csv(outputs["temperature_sweep"], index=False)
    pd.DataFrame(temperature_summary_rows).to_csv(outputs["temperature_summary"], index=False)
    shuffled.to_csv(outputs["temperature_shuffled_q"], index=False)
    alignment.to_csv(outputs["temperature_alignment"], index=False)
    weight_similarity.to_csv(outputs["expert_weight_similarity"], index=False)
    functional.to_csv(outputs["expert_functional_diversity"], index=False)
    pd.DataFrame(expert_summary_rows).to_csv(outputs["expert_summary"], index=False)
    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "source_experiment_id": exp551.EXPERIMENT_ID,
            "source_protocol_version": exp551.PROTOCOL_VERSION,
            "seeds": list(SEEDS),
            "temperatures": list(TEMPERATURES),
            "hard_variant": HARD_VARIANT,
            "shuffle_replicates": SHUFFLE_REPLICATES,
            "functional_max_points": FUNCTIONAL_MAX_POINTS,
            "diagnostic_only": True,
            "no_training": True,
            "no_temperature_selection": True,
            "test_policy": "val/test are reported for post-hoc diagnosis only; no temperature or architecture is selected in Exp5.5.1.1",
            "multi_cpu_policy": "5 independent seed diagnostic tasks -> one artifact-only finalizer",
            "notebook_policy": "analysis-only; reads finalized CSV/JSON artifacts and never runs diagnostics or regenerates missing artifacts",
        },
    )
    return outputs


def _config(repo_root: Path, device: str, threads: int) -> exp551.Config:
    return exp551.Config(
        repo_root=repo_root,
        results_dir=source_results_dir(repo_root),
        device=device,
        epochs=exp551.EPOCHS,
        patience=exp551.PATIENCE,
        batch_size=exp551.BATCH_SIZE,
        threads=threads,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exp5.5.1.1 post-hoc routing diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-seed")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--repo-root", default=None)
    run.add_argument("--device", default="cpu")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--force", action="store_true")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--repo-root", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    if args.command == "finalize":
        for name, path in finalize_experiment(root).items():
            print(f"{name}: {path}")
        return

    task = int(args.array_task_id)
    if task < 0 or task >= len(SEEDS):
        raise IndexError(f"run-seed array task {task} outside 0..{len(SEEDS) - 1}")
    torch.set_num_threads(int(args.threads))
    data = base.prepare_data(root)
    config = _config(root, str(args.device), int(args.threads))
    path = run_seed(SEEDS[task], data, config, results_dir(root), force=bool(args.force))
    print(path)


if __name__ == "__main__":
    main()
