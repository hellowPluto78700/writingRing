from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73


EXPERIMENT_ID = "experiment_8_0_local_backbone_tau_sweep"
PROTOCOL_VERSION = "local_backbone_tau_sweep_v1"
SEEDS = exp73.SEEDS
HIDDEN_WIDTH = exp73.HIDDEN_WIDTH
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
THRESHOLD = exp73.THRESHOLD
OUTPUT_BETA = 0.5
OUTPUT_ALPHA = 0.0
OUTPUT_CAP = 1
REGULARIZATION = exp73.REGULARIZATION
ARCHITECTURES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "234x234": ((2, 3, 4), (2, 3, 4)),
    "123x234": ((1, 2, 3), (2, 3, 4)),
    "123x123": ((1, 2, 3), (1, 2, 3)),
    "123x345": ((1, 2, 3), (3, 4, 5)),
    "234x345": ((2, 3, 4), (3, 4, 5)),
}
ARCHITECTURE_ORDER = tuple(ARCHITECTURES)
PROBE_AGGREGATIONS = ("whole_count", "fixed250")
LAYERS = ("l1", "l2")
EXPECTED_RUNS = len(ARCHITECTURES) * len(SEEDS)


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.architecture}__a2_wcce__{REGULARIZATION}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(architecture, seed) for architecture in ARCHITECTURE_ORDER for seed in SEEDS]


def validate_spec(spec: RunSpec) -> None:
    if spec.architecture not in ARCHITECTURES:
        raise ValueError(spec.architecture)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _valid_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.arange(steps, device=lengths.device)[None, :] < lengths[:, None]


def _valid_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _valid_mask(lengths, values.shape[1]).to(values.dtype).unsqueeze(-1)
    return (values * mask).sum(1) / lengths.clamp_min(1).to(values.dtype).unsqueeze(1)


def _group_counts(width: int, shifts: tuple[int, ...]) -> tuple[int, ...]:
    q, r = divmod(width, len(shifts))
    return tuple(q + (i < r) for i in range(len(shifts)))


def shift_groups(shifts: tuple[int, ...]) -> list[dict[str, int]]:
    groups, start = [], 0
    for shift, count in zip(shifts, _group_counts(HIDDEN_WIDTH, shifts), strict=True):
        groups.append({"shift": int(shift), "start": start, "stop": start + count, "count": count})
        start += count
    return groups


class Exp80Net(nn.Module):
    def __init__(self, shifts: tuple[tuple[int, ...], tuple[int, ...]], n_classes: int, fs: float) -> None:
        super().__init__()
        self.shifts = tuple(tuple(int(v) for v in layer) for layer in shifts)
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        self.hidden_linears = nn.ModuleList([
            nn.Linear(exp72.EXPECTED_CHANNELS, HIDDEN_WIDTH, bias=False),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False),
        ])
        beta_hidden = math.exp(-(1000.0 / fs) / exp72.TAU_MEM_MS)
        self.hidden_lifs = nn.ModuleList([
            exp401.MacroMultiSpikeLIF(
                beta=beta_hidden,
                threshold=THRESHOLD,
                max_spikes_per_dt=1,
                surrogate_slope=exp72.SURROGATE_SLOPE,
            )
            for _ in range(2)
        ])
        self.register_buffer("alpha_0", exp50.alpha_vector(HIDDEN_WIDTH, self.shifts[0]))
        self.register_buffer("alpha_1", exp50.alpha_vector(HIDDEN_WIDTH, self.shifts[1]))
        self.output_linear = nn.Linear(HIDDEN_WIDTH, n_classes, bias=False)

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != exp72.EXPECTED_CHANNELS:
            raise ValueError(channels)
        syn = [torch.zeros(batch, HIDDEN_WIDTH, device=x.device, dtype=x.dtype) for _ in range(2)]
        mem = [torch.zeros_like(syn[0]), torch.zeros_like(syn[1])]
        hidden: list[list[torch.Tensor]] = [[], []]
        evidence: list[torch.Tensor] = []
        for t in range(steps):
            cur = x[:, t]
            for li in range(2):
                alpha = getattr(self, f"alpha_{li}")
                syn[li] = alpha * syn[li] + self.hidden_linears[li](cur)
                spk, mem[li], _ = self.hidden_lifs[li](syn[li], mem[li])
                hidden[li].append(spk)
                cur = spk
            evidence.append(self.output_linear(cur))
        return {
            "hidden_spikes": tuple(torch.stack(v, dim=1) for v in hidden),
            "evidence": torch.stack(evidence, dim=1),
        }


def new_model(spec: RunSpec, data: exp3.Data) -> Exp80Net:
    return Exp80Net(ARCHITECTURES[spec.architecture], len(data.labels), data.fs)


def _evaluate_linear(model: Exp80Net, loader: Iterable, device: torch.device) -> dict[str, float]:
    ys, preds = [], []
    loss_sum, n_total = 0.0, 0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            tr = model.forward_trajectory(X)
            scores = _valid_mean(tr["evidence"], lengths)
            loss = F.cross_entropy(scores, y)
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)
    out = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    out["objective_loss"] = loss_sum / max(n_total, 1)
    return out


def _output_lif_spikes(evidence: torch.Tensor) -> torch.Tensor:
    batch, steps, n_classes = evidence.shape
    mem = torch.zeros(batch, n_classes, device=evidence.device, dtype=evidence.dtype)
    lif = exp401.MacroMultiSpikeLIF(
        beta=OUTPUT_BETA,
        threshold=THRESHOLD,
        max_spikes_per_dt=OUTPUT_CAP,
        surrogate_slope=exp72.SURROGATE_SLOPE,
    ).to(evidence.device)
    spikes = []
    for t in range(steps):
        # alpha_out = 0: direct current injection I_t = W z_t.
        spk, mem, _ = lif(evidence[:, t], mem)
        spikes.append(spk)
    return torch.stack(spikes, dim=1)


def _evaluate_lif_transfer(model: Exp80Net, loader: Iterable, device: torch.device) -> dict[str, float]:
    ys, preds = [], []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X, lengths = X.to(device), lengths.to(device)
            tr = model.forward_trajectory(X)
            out_spikes = _output_lif_spikes(tr["evidence"])
            mask = _valid_mask(lengths, out_spikes.shape[1]).to(out_spikes.dtype).unsqueeze(-1)
            counts = (out_spikes * mask).sum(1)
            ys.append(y.numpy())
            preds.append(counts.argmax(1).cpu().numpy())
    return exp72._metrics(np.concatenate(ys), np.concatenate(preds))


def _extract_layer_splits(model: Exp80Net, loaders: dict[str, Iterable], device: torch.device) -> dict[str, dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]]]:
    out = {layer: {} for layer in LAYERS}
    model.eval()
    with torch.no_grad():
        for split, loader in loaders.items():
            layer_chunks = {layer: [] for layer in LAYERS}
            ys, ls = [], []
            for X, y, lengths in loader:
                tr = model.forward_trajectory(X.to(device))
                for li, layer in enumerate(LAYERS):
                    layer_chunks[layer].append(tr["hidden_spikes"][li].cpu())
                ys.append(y.numpy())
                ls.append(lengths.numpy())
            y_arr, l_arr = np.concatenate(ys), np.concatenate(ls)
            for layer in LAYERS:
                out[layer][split] = (torch.cat(layer_chunks[layer], 0), y_arr, l_arr)
    return out


def _probe_features(spikes: torch.Tensor, lengths: np.ndarray, bin_steps: int, aggregation: str) -> np.ndarray:
    lt = torch.as_tensor(lengths, dtype=torch.long)
    if aggregation == "whole_count":
        mask = _valid_mask(lt, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
        return (spikes * mask).sum(1).numpy()
    if aggregation == "fixed250":
        return exp3.fixed_counts(spikes, lt, bin_steps).flatten(1).numpy()
    raise ValueError(aggregation)


def _fit_layer_probes(splits: dict[str, dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]]], seed: int, bin_steps: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer in LAYERS:
        result[layer] = {}
        for aggregation in PROBE_AGGREGATIONS:
            features = {
                split: (_probe_features(spikes, lengths, bin_steps, aggregation), y)
                for split, (spikes, y, lengths) in splits[layer].items()
            }
            probe = exp01._fit_linear_probe(
                features["train"][0], features["train"][1],
                features["val"][0], features["val"][1],
                features["test"][0], features["test"][1],
                seed=exp3.dseed(seed, EXPERIMENT_ID, layer, aggregation),
            )
            result[layer][aggregation] = probe
    return result


def _activity_stats(spikes: torch.Tensor, lengths: np.ndarray, groups: list[dict[str, int]], fs: float) -> list[dict[str, float | int]]:
    lt = torch.as_tensor(lengths, dtype=torch.long)
    mask = _valid_mask(lt, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
    valid_seconds = float(lt.sum()) / float(fs)
    rows = []
    for group in groups:
        chunk = spikes[:, :, group["start"]:group["stop"]]
        counts = (chunk * mask).sum(dim=(0, 1))
        rate = float(counts.sum()) / max(valid_seconds * group["count"], 1e-12)
        dead = float((counts == 0).float().mean())
        rows.append({**group, "mean_spikes_per_neuron_s": rate, "dead_neuron_fraction": dead})
    return rows


def _collect_activity(model: Exp80Net, loaders: dict[str, Iterable], data: exp3.Data, device: torch.device) -> dict[str, Any]:
    test = _extract_layer_splits(model, {"test": loaders["test"]}, device)
    result = {}
    for li, layer in enumerate(LAYERS):
        spikes, _, lengths = test[layer]["test"]
        result[layer] = _activity_stats(spikes, lengths, shift_groups(ARCHITECTURES[next(k for k,v in ARCHITECTURES.items() if v == model.shifts)][li]), data.fs)
    # output activity
    all_spikes, all_lengths = [], []
    with torch.no_grad():
        for X, _, lengths in loaders["test"]:
            tr = model.forward_trajectory(X.to(device))
            all_spikes.append(_output_lif_spikes(tr["evidence"]).cpu())
            all_lengths.append(lengths.numpy())
    out_spikes = torch.cat(all_spikes, 0)
    out_lengths = np.concatenate(all_lengths)
    result["output"] = _activity_stats(out_spikes, out_lengths, [{"shift": 0, "start": 0, "stop": out_spikes.shape[2], "count": out_spikes.shape[2]}], data.fs)
    return result


def _representative_test_sample(data: exp3.Data) -> tuple[int, int]:
    median = float(np.median(data.lte.astype(float)))
    idx = int(np.argmin(np.abs(data.lte.astype(float) - median)))
    return idx, int(data.lte[idx])


def _raster(arr: np.ndarray, path: Path, valid: int, title: str, groups: list[dict[str, int]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(arr.T, aspect="auto", interpolation="nearest", origin="lower")
    ax.axvline(valid, linestyle="--", linewidth=1)
    if groups:
        for group in groups[:-1]:
            ax.axhline(group["stop"] - 0.5, linestyle=":", linewidth=0.8)
        title += "\n" + ", ".join(f"s{g['shift']}:{g['start']}-{g['stop']-1}" for g in groups)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_rasters(spec: RunSpec, model: Exp80Net, data: exp3.Data, config: Config) -> dict[str, Any]:
    idx, valid = _representative_test_sample(data)
    device = torch.device(config.device)
    sample = torch.tensor(data.Xte[idx:idx+1], dtype=torch.float32, device=device)
    sample[:, valid:] = 0
    with torch.no_grad():
        tr = model.forward_trajectory(sample)
        out_spikes = _output_lif_spikes(tr["evidence"])
    raster_dir = config.results_dir / "rasters"
    paths = {}
    for li, layer in enumerate(LAYERS):
        path = raster_dir / f"{spec.key}__{layer.upper()}.png"
        _raster(tr["hidden_spikes"][li][0].cpu().numpy(), path, valid, f"{spec.key} — {layer.upper()}", shift_groups(ARCHITECTURES[spec.architecture][li]))
        paths[layer] = str(path.relative_to(config.repo_root))
    output_path = raster_dir / f"{spec.key}__output.png"
    _raster(out_spikes[0].cpu().numpy(), output_path, valid, f"{spec.key} — output LIF")
    paths["output"] = str(output_path.relative_to(config.repo_root))
    trace_path = raster_dir / f"{spec.key}__trace.npz"
    np.savez_compressed(
        trace_path,
        l1=tr["hidden_spikes"][0][0].cpu().numpy().astype(np.uint8),
        l2=tr["hidden_spikes"][1][0].cpu().numpy().astype(np.uint8),
        output=out_spikes[0].cpu().numpy().astype(np.uint8),
        valid_length=np.asarray(valid, dtype=np.int64),
        label=np.asarray(int(data.yte[idx]), dtype=np.int64),
    )
    return {"split": "test", "sample_index": idx, "valid_length": valid, "label": str(data.labels[int(data.yte[idx])]), "paths": paths, "trace_npz": str(trace_path.relative_to(config.repo_root))}


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    eval_path = _path(config.results_dir, "evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "checkpoints", spec.key, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)

    # Strict A2 pairing: identical initialization/data-order seed streams across architectures.
    exp3.seed_all(exp73._e2e_pair_seed(spec.seed, "model_init"))
    model = new_model(spec, data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY)
    train_loader = exp73._raw_loaders(data, spec.seed, config.batch_size, True)["train"]
    eval_loaders = exp73._raw_loaders(data, spec.seed, config.batch_size, False)

    best_state = None
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
            scores = _valid_mean(tr["evidence"], lengths)
            loss = F.cross_entropy(scores, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate_linear(model, eval_loaders["train"], device)
        val_metrics = _evaluate_linear(model, eval_loaders["val"], device)
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
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "architecture_shifts": ARCHITECTURES[spec.architecture],
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_ba": best_ba,
        "best_val_objective_loss": best_loss,
        "model_state_dict": best_state,
        "output_lif": {"alpha": OUTPUT_ALPHA, "beta": OUTPUT_BETA, "threshold": THRESHOLD, "cap": OUTPUT_CAP},
    }, checkpoint_path)

    model.load_state_dict(best_state, strict=True)
    linear_metrics = {split: _evaluate_linear(model, loader, device) for split, loader in eval_loaders.items()}
    lif_metrics = {split: _evaluate_lif_transfer(model, loader, device) for split, loader in eval_loaders.items()}
    layer_splits = _extract_layer_splits(model, eval_loaders, device)
    probes = _fit_layer_probes(layer_splits, spec.seed, data.bin_steps)
    activity = _collect_activity(model, eval_loaders, data, device)
    raster = _save_rasters(spec, model, data, config)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "architecture_shifts": ARCHITECTURES[spec.architecture],
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "linear_metrics": linear_metrics,
        "lif_transfer_metrics": lif_metrics,
        "lif_penalty_test_ba": float(linear_metrics["test"]["balanced_accuracy"] - lif_metrics["test"]["balanced_accuracy"]),
        "representation_probes": probes,
        "activity": activity,
        "raster": raster,
    }
    _save_json(eval_path, payload)
    history_path = _path(config.results_dir, "histories", spec.key, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return payload


def _probe_ba(payload: dict[str, Any], layer: str, aggregation: str) -> float:
    probe = payload["representation_probes"][layer][aggregation]
    if "metrics" in probe:
        return float(probe["metrics"]["test"]["balanced_accuracy"])
    if "test" in probe and isinstance(probe["test"], dict):
        return float(probe["test"]["balanced_accuracy"])
    return float(probe["test_balanced_accuracy"])


def _method_row(payload: dict[str, Any]) -> dict[str, Any]:
    l1_whole = _probe_ba(payload, "l1", "whole_count")
    l1_f250 = _probe_ba(payload, "l1", "fixed250")
    l2_whole = _probe_ba(payload, "l2", "whole_count")
    l2_f250 = _probe_ba(payload, "l2", "fixed250")
    return {
        "architecture": payload["spec"]["architecture"],
        "seed": payload["spec"]["seed"],
        "linear_test_ba": payload["linear_metrics"]["test"]["balanced_accuracy"],
        "lif_test_ba": payload["lif_transfer_metrics"]["test"]["balanced_accuracy"],
        "lif_penalty": payload["lif_penalty_test_ba"],
        "l1_whole_ba": l1_whole,
        "l1_fixed250_ba": l1_f250,
        "l2_whole_ba": l2_whole,
        "l2_fixed250_ba": l2_f250,
        "l1_temporal_gain": l1_f250 - l1_whole,
        "l2_temporal_gain": l2_f250 - l2_whole,
        "l2_minus_l1_whole": l2_whole - l1_whole,
        "l2_minus_l1_fixed250": l2_f250 - l1_f250,
        "best_epoch": payload["best_epoch"],
    }


def _activity_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for layer, groups in payload["activity"].items():
        for group in groups:
            rows.append({"architecture": payload["spec"]["architecture"], "seed": payload["spec"]["seed"], "layer": layer, **group})
    return rows


def _paired_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("fast_L1", "123x234", "234x234"),
        ("L2_short_to_mid", "123x234", "123x123"),
        ("L2_mid_to_long", "123x345", "123x234"),
        ("fast_L1_with_long_L2", "123x345", "234x345"),
    ]
    metrics = ["linear_test_ba", "lif_test_ba", "l1_whole_ba", "l1_fixed250_ba", "l2_whole_ba", "l2_fixed250_ba"]
    rows = []
    for name, a, b in comparisons:
        for seed in SEEDS:
            ar = runs[(runs.architecture == a) & (runs.seed == seed)].iloc[0]
            br = runs[(runs.architecture == b) & (runs.seed == seed)].iloc[0]
            row = {"contrast": name, "architecture_a": a, "architecture_b": b, "seed": seed}
            for metric in metrics:
                row[f"delta_{metric}"] = float(ar[metric] - br[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def finalize(config: Config) -> dict[str, Any]:
    payloads = []
    for spec in run_specs():
        path = _path(config.results_dir, "evaluations", spec.key, ".json")
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp8.0 evaluation: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if len(payloads) != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} runs, got {len(payloads)}")

    runs = pd.DataFrame([_method_row(payload) for payload in payloads])
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "method_runs.csv", index=False)
    metrics = [c for c in runs.columns if c not in {"architecture", "seed"}]
    summary = runs.groupby("architecture", sort=False)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col) for col in summary.columns]
    summary.to_csv(config.results_dir / "method_summary.csv", index=False)

    contrast_runs = _paired_contrasts(runs)
    contrast_runs.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_metrics = [c for c in contrast_runs.columns if c.startswith("delta_")]
    contrast_summary = contrast_runs.groupby(["contrast", "architecture_a", "architecture_b"], sort=False)[contrast_metrics].agg(["mean", "std"]).reset_index()
    contrast_summary.columns = ["_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col) for col in contrast_summary.columns]
    contrast_summary.to_csv(config.results_dir / "contrast_summary.csv", index=False)

    activity_runs = pd.DataFrame([row for payload in payloads for row in _activity_rows(payload)])
    activity_runs.to_csv(config.results_dir / "activity_runs.csv", index=False)
    activity_summary = activity_runs.groupby(["architecture", "layer", "shift"], sort=False)[["mean_spikes_per_neuron_s", "dead_neuron_fraction"]].agg(["mean", "std"]).reset_index()
    activity_summary.columns = ["_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col) for col in activity_summary.columns]
    activity_summary.to_csv(config.results_dir / "activity_summary.csv", index=False)

    raster_rows = []
    for payload in payloads:
        base = {"architecture": payload["spec"]["architecture"], "seed": payload["spec"]["seed"], "split": payload["raster"]["split"], "sample_index": payload["raster"]["sample_index"], "valid_length": payload["raster"]["valid_length"], "label": payload["raster"]["label"], "trace_npz": payload["raster"]["trace_npz"]}
        for layer, path in payload["raster"]["paths"].items():
            raster_rows.append({**base, "layer": layer, "raster_png": path})
    pd.DataFrame(raster_rows).to_csv(config.results_dir / "raster_index.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architectures": {k: [list(v[0]), list(v[1])] for k, v in ARCHITECTURES.items()},
        "seeds": list(SEEDS),
        "counts": {"architectures": len(ARCHITECTURES), "seeds": len(SEEDS), "parallel_runs": EXPECTED_RUNS},
        "training": "Exp7.3 A2-compatible end-to-end Linear/WCCE",
        "output_lif": {"alpha": OUTPUT_ALPHA, "beta": OUTPUT_BETA, "threshold": THRESHOLD, "cap": OUTPUT_CAP},
        "representative_raster_sample": payloads[0]["raster"],
        "tau_ms": {str(s): exp72.tau_ms_from_shift(s) for s in (1, 2, 3, 4, 5)},
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp8.0 local-backbone temporal-scale sweep")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-runs")
    run = sub.add_parser("run")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = find_repo_root(args.repo_root)
    config = Config(repo_root=repo_root, results_dir=results_dir(repo_root), device=args.device, batch_size=args.batch_size, threads=args.threads, max_epochs=args.max_epochs)
    specs = run_specs()
    if args.command == "list-runs":
        for i, spec in enumerate(specs):
            print(i, spec.key)
        return
    if args.command == "run":
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        payload = run_one(specs[args.array_task_id], config, force=args.force)
        print(json.dumps({"key": specs[args.array_task_id].key, "test_ba": payload["linear_metrics"]["test"]["balanced_accuracy"]}, indent=2))
        return
    if args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
