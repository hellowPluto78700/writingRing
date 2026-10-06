"""Read-only dissection of the three O0 baseline checkpoints.

No optimization is performed. Historical v1.0 checkpoints are loaded by weight
after explicit geometry/case/seed checks so that current v1.1 source identity
does not invalidate the locked production artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import BenchmarkNet, lif_step
from .protocol import Protocol, Run
from .storage import load_torch

SEEDS = (11, 23, 37)
CASE = "O0"
REQUIRED_LOCK = {
    "input_channels": 30,
    "width": 128,
    "steps": 256,
    "fs": 64.0,
    "threshold": 0.5,
    "tau_mem_ms": 22.54,
}
O0_SHIFTS = ((2, 3, 4), (2, 3, 4))


def _load_locked_data(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    lock = json.loads((root / "protocol.lock.json").read_text(encoding="utf-8"))
    locked = lock["protocol"]
    for key, expected in REQUIRED_LOCK.items():
        if locked.get(key) != expected:
            raise ValueError(f"Historical baseline geometry mismatch for {key}: {locked.get(key)!r}")
    if tuple(locked.get("labels", ())) != Protocol().labels:
        raise ValueError("Historical label order differs from the baseline evaluator")
    with np.load(root / "dataset.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return lock, arrays


def _checkpoint_run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError("Checkpoint is missing run metadata")
    return run


def load_o0(root: Path, seed: int) -> tuple[BenchmarkNet, dict[str, Any]]:
    path = root / "runs" / f"{CASE}__seed{seed}" / "checkpoint.pt"
    payload = load_torch(path)
    run_meta = _checkpoint_run_payload(payload)
    if run_meta.get("case") != CASE or int(run_meta.get("seed", -1)) != seed:
        raise ValueError(f"Not the requested O0 seed{seed} checkpoint")
    if tuple(tuple(v) for v in run_meta.get("shifts", ())) != O0_SHIFTS:
        raise ValueError("Checkpoint is not the locked two-layer 234/234 baseline")
    if run_meta.get("objective") != "wcce" or not payload.get("training_complete"):
        raise ValueError("Checkpoint is not a completed O0 WCCE training checkpoint")
    p = Protocol()
    run = Run(CASE, seed, "01_objective", shifts=O0_SHIFTS, objective="wcce")
    model = BenchmarkNet(run, p)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


@torch.no_grad()
def infer_split(model: BenchmarkNet, arrays: dict[str, np.ndarray], split: str, batch_size: int = 128) -> dict[str, np.ndarray]:
    x = arrays[f"{split}_x"]
    y = arrays[f"{split}_y"]
    lengths = arrays[f"{split}_lengths"]
    evidence_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for start in range(0, len(y), batch_size):
        stop = min(len(y), start + batch_size)
        xb = torch.from_numpy(x[start:stop])
        lb = torch.from_numpy(lengths[start:stop])
        evidence = model(xb, lb)["evidence"]
        mask = torch.arange(evidence.shape[1])[None, :] < lb[:, None]
        scores = (evidence * mask[..., None]).sum(1)
        evidence_parts.append(evidence.numpy())
        score_parts.append(scores.numpy())
        pred_parts.append(scores.argmax(1).numpy())
    return {
        "evidence": np.concatenate(evidence_parts),
        "scores": np.concatenate(score_parts),
        "prediction": np.concatenate(pred_parts),
        "y": y.copy(),
    }


def confusion(y: np.ndarray, pred: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(raw, (y.astype(int), pred.astype(int)), 1)
    denom = raw.sum(axis=1, keepdims=True)
    normalized = np.divide(raw, denom, out=np.zeros_like(raw, dtype=np.float64), where=denom != 0)
    return raw, normalized


def select_classes(test_normalized: dict[int, np.ndarray], labels: tuple[str, ...]) -> dict[str, Any]:
    mean_cm = np.mean(np.stack([test_normalized[s] for s in SEEDS]), axis=0)
    recalls = np.diag(mean_cm)
    order = np.lexsort((np.arange(len(labels)), recalls))
    lower_median = order[(len(order) - 1) // 2]
    chosen = {
        "highest": int(order[-1]),
        "middle": int(lower_median),
        "lowest": int(order[0]),
    }
    return {
        "indices": chosen,
        "labels": {role: labels[idx] for role, idx in chosen.items()},
        "mean_test_row_normalized_confusion": mean_cm,
        "mean_recall": {labels[i]: float(recalls[i]) for i in range(len(labels))},
        "selection_rule": "highest, lower-median (6th of 12), and lowest diagonal recall from the mean of the three row-normalized O0 test confusion matrices",
    }


def _representative_index(
    arrays: dict[str, np.ndarray],
    split: str,
    class_idx: int,
    prediction: np.ndarray,
    prefer_error: bool,
) -> int:
    y = arrays[f"{split}_y"]
    lengths = arrays[f"{split}_lengths"]
    candidates = np.flatnonzero(y == class_idx)
    if not len(candidates):
        raise ValueError(f"No samples for class={class_idx} in {split}")
    if prefer_error:
        preferred = candidates[prediction[candidates] != y[candidates]]
    else:
        preferred = candidates[prediction[candidates] == y[candidates]]
    if not len(preferred):
        preferred = candidates
    median = float(np.median(lengths[preferred]))
    distances = np.abs(lengths[preferred].astype(float) - median)
    # Stable sample index breaks equal-length ties.
    return int(preferred[np.lexsort((preferred, distances))[0]])


@torch.no_grad()
def trace_one(model: BenchmarkNet, x_np: np.ndarray, valid_length: int) -> dict[str, np.ndarray]:
    p = model.protocol
    x = torch.from_numpy(x_np[:valid_length]).unsqueeze(0)
    syn = [x.new_zeros(1, p.width) for _ in model.layers]
    mem = [x.new_zeros(1, p.width) for _ in model.layers]
    spikes = [[] for _ in model.layers]
    currents = [[] for _ in model.layers]
    membranes = [[] for _ in model.layers]
    evidence: list[torch.Tensor] = []
    for t in range(valid_length):
        cur = x[:, t]
        for li, linear in enumerate(model.layers):
            syn[li] = getattr(model, f"alpha_{li}") * syn[li] + linear(cur)
            spike, mem[li], _ = lif_step(syn[li], mem[li], model.betas[li], p.threshold, p.surrogate_slope)
            currents[li].append(syn[li].squeeze(0).clone())
            membranes[li].append(mem[li].squeeze(0).clone())
            spikes[li].append(spike.squeeze(0).clone())
            cur = spike
        evidence.append(model.head(cur).squeeze(0))
    result = {
        "input": x.squeeze(0).numpy(),
        "output_evidence": torch.stack(evidence).numpy(),
    }
    for li in range(len(model.layers)):
        result[f"L{li+1}_spike"] = torch.stack(spikes[li]).numpy()
        result[f"L{li+1}_I"] = torch.stack(currents[li]).numpy()
        result[f"L{li+1}_U"] = torch.stack(membranes[li]).numpy()
    result["output_accumulator"] = np.cumsum(result["output_evidence"], axis=0)
    return result


def _save_matrix_csv(path: Path, values: np.ndarray, row_axis: str = "timestep") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, delimiter=",", fmt="%.8g",
               header=",".join([f"c{i}" for i in range(values.shape[1])]), comments="")


def _save_confusion_csv(path: Path, matrix: np.ndarray, labels: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def _tau_boundaries(run_shifts: tuple[int, ...], width: int) -> tuple[list[int], list[tuple[int, slice]]]:
    q, r = divmod(width, len(run_shifts))
    start = 0
    groups = []
    boundaries = []
    for j, shift in enumerate(run_shifts):
        size = q + (j < r)
        groups.append((shift, slice(start, start + size)))
        start += size
        if start < width:
            boundaries.append(start)
    return boundaries, groups


def plot_confusion(path: Path, matrix: np.ndarray, labels: tuple[str, ...], title: str, normalized: bool) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, aspect="equal", interpolation="nearest", vmin=0, vmax=(1 if normalized else None))
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set(xlabel="Predicted class", ylabel="True class", title=title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _heat(ax: Any, values_txn: np.ndarray, title: str, ylabel: str, boundaries: list[int] | None = None,
          signed: bool = True) -> None:
    data = values_txn.T
    vmax = float(np.quantile(np.abs(data), 0.995)) if data.size else 1.0
    if vmax == 0:
        vmax = 1.0
    if signed:
        image = ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest",
                          cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    else:
        image = ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest",
                          cmap="viridis", vmin=0, vmax=max(float(data.max(initial=0)), 1.0))
    if boundaries:
        for boundary in boundaries:
            ax.axhline(boundary - 0.5, linewidth=0.8, alpha=0.7)
    ax.set(ylabel=ylabel, title=title)
    return image


def plot_raster_panel(path: Path, trace: dict[str, np.ndarray], labels: tuple[str, ...], meta: dict[str, Any]) -> None:
    boundaries, _ = _tau_boundaries((2, 3, 4), 128)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    _heat(axes[0], trace["input"], "Input raster", "input channel", signed=False)
    _heat(axes[1], trace["L1_spike"], "L1 spike raster", "L1 neuron", boundaries, signed=False)
    _heat(axes[2], trace["L2_spike"], "L2 spike raster", "L2 neuron", boundaries, signed=False)
    im = _heat(axes[3], trace["output_evidence"], "Accumulator input e(t) by output class", "class", signed=True)
    axes[3].set_yticks(np.arange(len(labels)), labels)
    axes[3].set_xlabel("Valid timestep")
    fig.colorbar(im, ax=axes[3], fraction=0.025, pad=0.01, label="instantaneous evidence")
    fig.suptitle(
        f"{meta['checkpoint']} | {meta['split']} {meta['user']} | true={meta['true_label']} "
        f"pred={meta['predicted_label']} | T={meta['valid_length']} | margin={meta['margin']:.3f}",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_state_panel(path: Path, trace: dict[str, np.ndarray], state: str, meta: dict[str, Any]) -> None:
    boundaries, _ = _tau_boundaries((2, 3, 4), 128)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    im1 = _heat(axes[0], trace[f"L1_{state}"], f"L1 {state}", "L1 neuron", boundaries, signed=True)
    im2 = _heat(axes[1], trace[f"L2_{state}"], f"L2 {state}", "L2 neuron", boundaries, signed=True)
    axes[1].set_xlabel("Valid timestep")
    fig.colorbar(im1, ax=axes[0], fraction=0.025, pad=0.01)
    fig.colorbar(im2, ax=axes[1], fraction=0.025, pad=0.01)
    fig.suptitle(f"{meta['checkpoint']} | {meta['split']} | {meta['true_label']} | {state}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_weight_heatmap(path: Path, weight: np.ndarray, title: str, y_boundaries: list[int] | None = None,
                        x_boundaries: list[int] | None = None) -> None:
    vmax = max(float(np.quantile(np.abs(weight), 0.995)), 1e-12)
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(weight, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for boundary in y_boundaries or []:
        ax.axhline(boundary - 0.5, linewidth=0.8)
    for boundary in x_boundaries or []:
        ax.axvline(boundary - 0.5, linewidth=0.8)
    ax.set(title=title, xlabel="source", ylabel="destination")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_histogram(path: Path, groups: list[tuple[str, np.ndarray]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in groups:
        ax.hist(values.ravel(), bins=80, density=True, histtype="step", linewidth=1.4, label=label)
    ax.set(xlabel="Weight", ylabel="Density", title=title)
    if len(groups) > 1:
        ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_weights(model: BenchmarkNet, out: Path, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    boundaries, groups = _tau_boundaries((2, 3, 4), model.protocol.width)
    matrices = {
        "input_to_L1": model.layers[0].weight.detach().cpu().numpy(),
        "L1_to_L2": model.layers[1].weight.detach().cpu().numpy(),
        "L2_to_output": model.head.weight.detach().cpu().numpy(),
    }
    for name, weight in matrices.items():
        np.savetxt(out / f"{name}.csv", weight, delimiter=",", fmt="%.8g")
        yb = boundaries if name != "L2_to_output" else None
        xb = boundaries if name in ("L1_to_L2", "L2_to_output") else None
        plot_weight_heatmap(out / f"{name}_heatmap.png", weight, f"seed{seed}: {name}", yb, xb)
        plot_histogram(out / f"{name}_histogram.png", [("all weights", weight)], f"seed{seed}: {name} overall distribution")
        if name == "input_to_L1":
            tau = [(f"dst shift {shift}", weight[sl, :]) for shift, sl in groups]
        elif name == "L1_to_L2":
            tau = [(f"dst shift {shift}", weight[sl, :]) for shift, sl in groups]
            with (out / "L1_to_L2_tau_block_summary.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["source_shift", "destination_shift", "count", "mean", "std", "mean_abs", "rms"])
                for dst_shift, dst_slice in groups:
                    for src_shift, src_slice in groups:
                        block = weight[dst_slice, src_slice]
                        writer.writerow([
                            src_shift, dst_shift, block.size, float(block.mean()), float(block.std()),
                            float(np.abs(block).mean()), float(np.sqrt(np.mean(np.square(block)))),
                        ])
        else:
            tau = [(f"src shift {shift}", weight[:, sl]) for shift, sl in groups]
        plot_histogram(out / f"{name}_tau_group_histogram.png", tau, f"seed{seed}: {name} by tau group")


def _jsonable_selection(selection: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in selection.items() if not isinstance(v, np.ndarray)}


def evaluate(root: Path, output: Path) -> dict[str, Any]:
    lock, arrays = _load_locked_data(root)
    labels = tuple(lock["protocol"]["labels"])
    output.mkdir(parents=True, exist_ok=True)

    models: dict[int, BenchmarkNet] = {}
    inferences: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    test_norm: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        model, _ = load_o0(root, seed)
        models[seed] = model
        inferences[seed] = {}
        seed_dir = output / f"seed{seed}"
        confusion_dir = seed_dir / "confusion"
        for split in ("train", "test"):
            result = infer_split(model, arrays, split)
            inferences[seed][split] = result
            raw, norm = confusion(result["y"], result["prediction"], len(labels))
            if split == "test":
                test_norm[seed] = norm
            _save_confusion_csv(confusion_dir / f"{split}_raw.csv", raw, labels)
            _save_confusion_csv(confusion_dir / f"{split}_row_normalized.csv", norm, labels)
            plot_confusion(confusion_dir / f"{split}_raw.png", raw, labels, f"O0 seed{seed} {split} confusion", False)
            plot_confusion(confusion_dir / f"{split}_row_normalized.png", norm, labels, f"O0 seed{seed} {split} row-normalized confusion", True)
        export_weights(model, seed_dir / "weights", seed)

    selection = select_classes(test_norm, labels)
    np.savetxt(output / "mean_test_row_normalized_confusion.csv",
               selection["mean_test_row_normalized_confusion"], delimiter=",", fmt="%.8g")
    (output / "class_selection.json").write_text(
        json.dumps(_jsonable_selection(selection), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        model = models[seed]
        for role in ("highest", "middle", "lowest"):
            class_idx = selection["indices"][role]
            for split in ("train", "test"):
                result = inferences[seed][split]
                prefer_error = role == "lowest" and split == "test"
                idx = _representative_index(arrays, split, class_idx, result["prediction"], prefer_error)
                valid_length = int(arrays[f"{split}_lengths"][idx])
                scores = result["scores"][idx]
                pred = int(result["prediction"][idx])
                sorted_scores = np.sort(scores)
                margin = float(sorted_scores[-1] - sorted_scores[-2])
                meta = {
                    "checkpoint": f"O0__seed{seed}",
                    "seed": seed,
                    "role": role,
                    "split": split,
                    "index": idx,
                    "sample_id": str(arrays[f"{split}_ids"][idx]),
                    "user": str(arrays[f"{split}_users"][idx]),
                    "true_class": int(class_idx),
                    "true_label": labels[class_idx],
                    "predicted_class": pred,
                    "predicted_label": labels[pred],
                    "correct": bool(pred == class_idx),
                    "valid_length": valid_length,
                    "margin": margin,
                    "selection_preferred_error": prefer_error,
                }
                trace = trace_one(model, arrays[f"{split}_x"][idx], valid_length)
                sample_dir = output / f"seed{seed}" / "samples" / f"{role}_{labels[class_idx]}" / split
                sample_dir.mkdir(parents=True, exist_ok=True)
                plot_raster_panel(sample_dir / "raster_panel.png", trace, labels, meta)
                plot_state_panel(sample_dir / "I_heatmaps.png", trace, "I", meta)
                plot_state_panel(sample_dir / "U_heatmaps.png", trace, "U", meta)
                for key, value in trace.items():
                    _save_matrix_csv(sample_dir / f"{key}.csv", value)
                (sample_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                manifest_rows.append(meta)

    with (output / "sample_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "status": "PASS",
        "training_performed": False,
        "source_results_root": str(root.resolve()),
        "baseline_case": CASE,
        "seeds": list(SEEDS),
        "selected_classes": selection["labels"],
        "selection_rule": selection["selection_rule"],
        "sample_count": len(manifest_rows),
        "output_root": str(output.resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the three CoreBenchmark O0 baseline checkpoints without training.")
    parser.add_argument("--root", type=Path, default=Path("core_benchmark_v1/results/main"),
                        help="Historical finalized CoreBenchmark result root.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: <root>/baseline_evaluation).")
    args = parser.parse_args()
    out = args.output or (args.root / "baseline_evaluation")
    print(json.dumps(evaluate(args.root.resolve(), out.resolve()), indent=2))


if __name__ == "__main__":
    main()