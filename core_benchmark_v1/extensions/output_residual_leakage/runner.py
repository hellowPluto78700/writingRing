"""CLI, training, evaluation, probes and mechanism diagnostics for the extension."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core_benchmark_v1.data import load_cache, loader, prepare as prepare_core
from core_benchmark_v1.model import BenchmarkNet
from core_benchmark_v1.probes import fit_probe, run_probes, temporal_features
from core_benchmark_v1.protocol import Protocol, Run, SPLITS
from core_benchmark_v1.storage import load_lock, load_torch, save_json, save_npz, save_torch, state_hash
from core_benchmark_v1.training import cpu_state, metrics
from . import EXTENSION_ID, EXTENSION_VERSION
from .model import extension_loss, residual_leakage_output
from .protocol import BETAS, OBJECTIVES, SEEDS, ExtensionRun, runs


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def configure_cpu() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("CoreBenchmark extensions require Python 3.11")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def core_protocol(smoke: bool = False) -> Protocol:
    if not smoke:
        return Protocol()
    return replace(Protocol(), profile="smoke", seeds=(11,), width=6, input_channels=3,
                   total_channels=9, steps=32, labels=("A", "B", "C"), batch_size=9,
                   max_epochs=1, min_epochs=1, patience=1, c_grid=(0.1,),
                   probe_max_iter=1000, shuffle_seeds=(101,), relative_bins=4,
                   lags=(1, 2), history_ms=(125.0, 250.0))


def core_run(spec: ExtensionRun) -> Run:
    # Case/objective names do not participate in Core paired parameter streams.
    # Keeping the standard 234/234 geometry gives exact Core initialization per seed.
    return Run(spec.key, spec.seed, "08_output_residual_leakage", objective="wcce")


def extension_manifest() -> dict[str, Any]:
    return {
        "extension_id": EXTENSION_ID,
        "version": EXTENSION_VERSION,
        "betas": BETAS,
        "objectives": OBJECTIVES,
        "seeds": SEEDS,
        "runs": [r.as_dict() | {"key": r.key} for r in runs()],
        "core_contract_modified": False,
    }


def prepare(root: Path, *, synthetic: bool = False) -> dict[str, Any]:
    p = core_protocol(smoke=synthetic)
    lock = prepare_core(repo_root(), root, p, synthetic=synthetic)
    manifest = extension_manifest()
    if synthetic:
        manifest["seeds"] = (11,)
        manifest["betas"] = (0.0, 0.5, 1.0)
    save_json(root / "extension_manifest.json", manifest)
    return {"core_identity": lock["identity"], **manifest}


def available_runs(root: Path) -> list[ExtensionRun]:
    manifest = json.loads((root / "extension_manifest.json").read_text())
    allowed_seeds = tuple(manifest["seeds"])
    allowed_betas = tuple(float(v) for v in manifest["betas"])
    return [r for r in runs() if r.seed in allowed_seeds and r.beta in allowed_betas]


def validate_checkpoint(payload: dict[str, Any], spec: ExtensionRun, core_identity: str) -> None:
    if payload.get("extension_version") != EXTENSION_VERSION or payload.get("core_identity") != core_identity:
        raise ValueError(f"Extension checkpoint identity mismatch: {spec.key}")
    if payload.get("run") != spec.as_dict():
        raise ValueError(f"Extension run mismatch: {spec.key}")


def forward_output(model: BenchmarkNet, x: torch.Tensor, lengths: torch.Tensor, spec: ExtensionRun, p: Protocol) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    hidden = model(x, lengths)
    output = residual_leakage_output(hidden["evidence"], lengths, beta=spec.beta,
                                     threshold=p.threshold, slope=p.surrogate_slope)
    return hidden, output


@torch.no_grad()
def evaluate_split(model: BenchmarkNet, arrays: dict[str, np.ndarray], split: str, spec: ExtensionRun, p: Protocol) -> dict[str, Any]:
    model.eval()
    ys, total_pred, spike_pred, analog_pred = [], [], [], []
    loss_sum = 0.0
    residual_norm, drift_norm, backlog1, backlog2, backlog4 = [], [], [], [], []
    for x, y, lengths in loader(arrays, split, p, spec.seed):
        hidden, out = forward_output(model, x, lengths, spec, p)
        loss = extension_loss(out, lengths, y, spec.objective)
        loss_sum += float(loss) * len(y)
        ys.append(y.numpy())
        total_pred.append(out["total_score"].argmax(1).numpy())
        spike_pred.append(out["spike_score"].argmax(1).numpy())
        analog_pred.append(out["analog_score"].argmax(1).numpy())
        residual_norm.append(out["final_residual"].abs().sum(1).numpy())
        drift_norm.append(out["leakage_drift"].abs().sum(1).numpy())
        mask = (torch.arange(p.steps)[None, :] < lengths[:, None]).unsqueeze(-1)
        r = out["residual"]
        denom = mask.sum().item() * r.shape[-1]
        backlog1.append(float(((r > p.threshold) & mask).sum()) / max(denom, 1))
        backlog2.append(float(((r > 2 * p.threshold) & mask).sum()) / max(denom, 1))
        backlog4.append(float(((r > 4 * p.threshold) & mask).sum()) / max(denom, 1))
    y = np.concatenate(ys)
    total = np.concatenate(total_pred)
    spike = np.concatenate(spike_pred)
    analog = np.concatenate(analog_pred)
    return {
        **metrics(y, total),
        "objective_ce": loss_sum / len(y),
        "spike_only_ba": metrics(y, spike)["ba"],
        "analog_counterfactual_ba": metrics(y, analog)["ba"],
        "total_vs_spike_disagreement": float((total != spike).mean()),
        "total_vs_analog_disagreement": float((total != analog).mean()),
        "final_residual_l1": float(np.concatenate(residual_norm).mean()),
        "leakage_drift_l1": float(np.concatenate(drift_norm).mean()),
        "backlog_gt_1theta": float(np.mean(backlog1)),
        "backlog_gt_2theta": float(np.mean(backlog2)),
        "backlog_gt_4theta": float(np.mean(backlog4)),
        "n_samples": len(y),
    }


def evaluate_validation(model: BenchmarkNet, arrays: dict[str, np.ndarray], spec: ExtensionRun, p: Protocol) -> dict[str, Any]:
    return evaluate_split(model, arrays, "val", spec, p)


def train(root: Path, spec: ExtensionRun, p: Protocol, lock: dict[str, Any], arrays: dict[str, np.ndarray]) -> BenchmarkNet:
    directory = root / "runs" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    run = core_run(spec)
    if checkpoint_path.exists():
        payload = load_torch(checkpoint_path)
        validate_checkpoint(payload, spec, lock["identity"])
        model = BenchmarkNet(run, p)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return model
    model = BenchmarkNet(run, p)
    initial = cpu_state(model)
    parameter_hashes = {name: state_hash({name: tensor}) for name, tensor in model.named_parameters()}
    save_torch(directory / "initial.pt", {"extension_version": EXTENSION_VERSION, "core_identity": lock["identity"],
                                           "run": spec.as_dict(), "model_state_dict": initial,
                                           "initial_parameter_hashes": parameter_hashes})
    optimizer = torch.optim.Adam(model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
    train_batches = loader(arrays, "train", p, spec.seed, shuffle=True)
    best = evaluate_validation(model, arrays, spec, p)
    best_state, best_epoch = initial, 0
    history = [{"epoch": 0, "train_loss": None, "val_ba": best["ba"], "val_objective_ce": best["objective_ce"]}]
    epoch = 0
    for epoch in range(1, p.max_epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for x, y, lengths in train_batches:
            optimizer.zero_grad(set_to_none=True)
            _, out = forward_output(model, x, lengths, spec, p)
            loss = extension_loss(out, lengths, y, spec.objective)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite loss")
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise FloatingPointError(f"{spec.key}: nonfinite gradient")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            count += len(y)
        val = evaluate_validation(model, arrays, spec, p)
        history.append({"epoch": epoch, "train_loss": loss_sum / count,
                        "val_ba": val["ba"], "val_objective_ce": val["objective_ce"]})
        improved = val["ba"] > best["ba"] + 1e-12 or (
            abs(val["ba"] - best["ba"]) <= 1e-12 and val["objective_ce"] < best["objective_ce"] - 1e-12)
        if improved:
            best, best_epoch, best_state = val, epoch, cpu_state(model)
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break
    model.load_state_dict(best_state)
    save_json(directory / "history.json", {"rows": history})
    save_torch(checkpoint_path, {"extension_version": EXTENSION_VERSION, "core_identity": lock["identity"],
        "run": spec.as_dict(), "model_state_dict": best_state, "training_complete": True,
        "best_epoch": best_epoch, "stopped_epoch": epoch, "best_val": best,
        "initial_parameter_hashes": parameter_hashes,
        "selection_rule": "validation native BA, then extension objective CE, then earliest epoch"})
    return model

@torch.no_grad()
def extract(model: BenchmarkNet, arrays: dict[str, np.ndarray], spec: ExtensionRun, p: Protocol) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model.eval()
    run = core_run(spec)
    traces: dict[str, np.ndarray] = {}
    native: dict[str, Any] = {"run_key": spec.key, "objective": spec.objective, "beta": spec.beta,
                              "seed": spec.seed, "splits": {}, "users": [], "activity": []}
    for split in SPLITS:
        chunks: dict[str, list[np.ndarray]] = {"evidence": [], "output_spike": [], "output_residual": [],
                                               "output_pre_reset": [], "output_trajectory": []}
        for state in ("spike", "pre_reset"):
            for li in range(len(run.shifts)):
                chunks[f"L{li+1}__{state}"] = []
        ys, total_preds, spike_preds, analog_preds = [], [], [], []
        for x, y, lengths in loader(arrays, split, p, spec.seed):
            hidden, out = forward_output(model, x, lengths, spec, p)
            chunks["evidence"].append(hidden["evidence"].numpy())
            chunks["output_spike"].append(out["spike"].numpy().astype(np.uint8))
            chunks["output_residual"].append(out["residual"].numpy().astype(np.float32))
            chunks["output_pre_reset"].append(out["pre_reset"].numpy().astype(np.float32))
            chunks["output_trajectory"].append(out["trajectory"].numpy().astype(np.float32))
            for state in ("spike", "pre_reset"):
                for li in range(len(run.shifts)):
                    value = hidden[state][li].numpy()
                    chunks[f"L{li+1}__{state}"].append(value.astype(np.uint8 if state == "spike" else np.float32))
            ys.append(y.numpy())
            total_preds.append(out["total_score"].argmax(1).numpy())
            spike_preds.append(out["spike_score"].argmax(1).numpy())
            analog_preds.append(out["analog_score"].argmax(1).numpy())
        for key, pieces in chunks.items():
            traces[f"{split}__{key}"] = np.concatenate(pieces)
        y = np.concatenate(ys)
        total_pred = np.concatenate(total_preds)
        spike_pred = np.concatenate(spike_preds)
        analog_pred = np.concatenate(analog_preds)
        split_metrics = evaluate_split(model, arrays, split, spec, p)
        split_metrics["spike_prediction_ba"] = metrics(y, spike_pred)["ba"]
        split_metrics["analog_prediction_ba"] = metrics(y, analog_pred)["ba"]
        native["splits"][split] = split_metrics
        traces[f"{split}__native_prediction"] = total_pred
        traces[f"{split}__spike_only_prediction"] = spike_pred
        traces[f"{split}__analog_prediction"] = analog_pred
        for user in np.unique(arrays[f"{split}_users"]):
            selected = arrays[f"{split}_users"] == user
            native["users"].append({"split": split, "user": str(user), "n": int(selected.sum()),
                                     **metrics(y[selected], total_pred[selected])})
        lengths = arrays[f"{split}_lengths"]
        mask = np.arange(p.steps)[None, :] < lengths[:, None]
        duration_s = float(lengths.sum()) / p.fs
        for li in range(len(run.shifts)):
            z3 = traces[f"{split}__L{li+1}__spike"]
            flat = z3[mask]
            rates = z3.sum(axis=(0, 1)) / max(duration_s, 1e-12)
            native["activity"].append({"split": split, "layer": f"L{li+1}",
                "mean_hz": float(rates.mean()), "median_hz": float(np.median(rates)),
                "p10_hz": float(np.percentile(rates, 10)), "p25_hz": float(np.percentile(rates, 25)),
                "p50_hz": float(np.percentile(rates, 50)), "p75_hz": float(np.percentile(rates, 75)),
                "p90_hz": float(np.percentile(rates, 90)), "p95_hz": float(np.percentile(rates, 95)),
                "max_hz": float(rates.max()), "fraction_lt1hz": float((rates < 1).mean()),
                "fraction_lt5hz": float((rates < 5).mean()), "fraction_gt10hz": float((rates > 10).mean()),
                "fraction_gt20hz": float((rates > 20).mean()), "fraction_gt30hz": float((rates > 30).mean()),
                "nonzero_fraction": float((flat != 0).mean())})
    native["train_test_gap"] = native["splits"]["train"]["ba"] - native["splits"]["test"]["ba"]
    return traces, native


def run_lengths(z: np.ndarray, lengths: np.ndarray, fs: float) -> dict[str, float]:
    values: list[int] = []
    spike_runs_ge = {2: 0, 4: 0, 8: 0, 16: 0}
    total_spikes = 0
    for i, length in enumerate(lengths):
        for j in range(z.shape[2]):
            seq = z[i, :int(length), j].astype(bool)
            padded = np.r_[False, seq, False].astype(np.int8)
            delta = np.diff(padded)
            starts, ends = np.flatnonzero(delta == 1), np.flatnonzero(delta == -1)
            runs = (ends - starts).tolist()
            values.extend(runs)
            total_spikes += int(seq.sum())
            for threshold in spike_runs_ge:
                spike_runs_ge[threshold] += sum(v for v in runs if v >= threshold)
    if not values:
        values = [0]
    return {"mean_run_steps": float(np.mean(values)), "median_run_steps": float(np.median(values)),
            "p90_run_steps": float(np.percentile(values, 90)), "max_run_steps": int(max(values)),
            "p90_run_ms": float(np.percentile(values, 90) * 1000 / fs),
            **{f"spike_fraction_in_runs_ge{k}": float(v / max(total_spikes, 1)) for k, v in spike_runs_ge.items()}}


def mechanism_diagnostics(traces: dict[str, np.ndarray], arrays: dict[str, np.ndarray], spec: ExtensionRun, p: Protocol) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    train_lengths = arrays["train_lengths"]
    for layer in ("L1", "L2"):
        for split in SPLITS:
            z = traces[f"{split}__{layer}__spike"]
            rows.append({"layer": layer, "split": split, **run_lengths(z, arrays[f"{split}_lengths"], p.fs)})
    output_rows = []
    for split in SPLITS:
        z = traces[f"{split}__output_spike"]
        lengths = arrays[f"{split}_lengths"]
        mask = np.arange(p.steps)[None, :] < lengths[:, None]
        residual = traces[f"{split}__output_residual"]
        spike_score_l1 = np.abs(p.threshold * z).sum(axis=(1, 2))
        final_r = residual[np.arange(len(lengths)), lengths - 1]
        output_rows.append({"split": split, "spikes_per_valid_timestep": float(z[mask].sum(1).mean()),
                            "final_residual_l1": float(np.abs(final_r).sum(1).mean()),
                            "residual_to_spike_l1_ratio": float(np.mean(np.abs(final_r).sum(1) / (spike_score_l1 + 1e-12))),
                            **run_lengths(z, lengths, p.fs)})
    return {"run_key": spec.key, "rows": rows, "output": output_rows}


def group_probe_rows(traces: dict[str, np.ndarray], arrays: dict[str, np.ndarray], spec: ExtensionRun, p: Protocol) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in ("L1", "L2"):
        z_train = traces[f"train__{layer}__spike"]
        rates = z_train.sum(axis=(0, 1)) / (arrays["train_lengths"].sum() / p.fs)
        order = np.argsort(rates)
        groups = {
            "low30": order[:max(1, int(np.ceil(0.30 * len(order))))],
            "high30": order[-max(1, int(np.ceil(0.30 * len(order)))):],
        }
        for group, indices in groups.items():
            for aggregation in ("whole_count", "fixed250_ordered", "relative10_ordered"):
                features = {s: temporal_features(traces[f"{s}__{layer}__spike"][:, :, indices], arrays[f"{s}_lengths"],
                                                 arrays[f"{s}_ids"], aggregation, p) for s in SPLITS}
                scaler, model, _ = fit_probe(features["train"], arrays["train_y"], features["val"], arrays["val_y"], "no_bias", p)
                row = {"run_key": spec.key, "layer": layer, "group": group, "aggregation": aggregation,
                       "n_neurons": len(indices), "rate_mean_hz": float(rates[indices].mean()), "C": float(model.C)}
                for split in SPLITS:
                    pred = model.predict(scaler.transform(features[split].astype(np.float64)))
                    row.update({f"{split}_{k}": v for k, v in metrics(arrays[f"{split}_y"], pred).items()})
                rows.append(row)
    return rows

def gradient_profile(model: BenchmarkNet, arrays: dict[str, np.ndarray], spec: ExtensionRun, p: Protocol) -> dict[str, Any]:
    model.eval()
    bins = 10
    sums = np.zeros(bins, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)
    for x, y, lengths in loader(arrays, "train", p, spec.seed):
        model.zero_grad(set_to_none=True)
        hidden = model(x, lengths)
        evidence = hidden["evidence"]
        evidence.retain_grad()
        out = residual_leakage_output(evidence, lengths, beta=spec.beta, threshold=p.threshold, slope=p.surrogate_slope)
        loss = extension_loss(out, lengths, y, spec.objective)
        loss.backward()
        grad = evidence.grad.detach().norm(dim=2).cpu().numpy()
        for i, length in enumerate(lengths.cpu().numpy()):
            for t in range(int(length)):
                b = min(bins - 1, int(bins * t / max(int(length), 1)))
                sums[b] += grad[i, t]
                counts[b] += 1
    return {"run_key": spec.key, "normalized_time_bins": bins,
            "mean_dloss_devidence_norm": (sums / np.maximum(counts, 1)).tolist(),
            "counts": counts.tolist()}


def load_selected_model(root: Path, spec: ExtensionRun, p: Protocol, lock: dict[str, Any]) -> BenchmarkNet:
    payload = load_torch(root / "runs" / spec.key / "checkpoint.pt")
    validate_checkpoint(payload, spec, lock["identity"])
    if not payload.get("training_complete"):
        raise ValueError(f"Incomplete checkpoint: {spec.key}")
    model = BenchmarkNet(core_run(spec), p)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def run_formal(root: Path, spec: ExtensionRun) -> dict[str, Any]:
    configure_cpu()
    p, lock = load_lock(root, repo_root())
    arrays = load_cache(root, p, lock)
    model = train(root, spec, p, lock, arrays)
    traces, native = extract(model, arrays, spec, p)
    directory = root / "runs" / spec.key
    save_npz(directory / "traces.npz", traces)
    save_json(directory / "native.json", native)
    save_json(directory / "train_complete.json", {"status": "PASS", "run": spec.as_dict(), "core_identity": lock["identity"]})
    return {"run": spec.key, "status": "PASS", "test_ba": native["splits"]["test"]["ba"]}


def run_analysis(root: Path, spec: ExtensionRun) -> dict[str, Any]:
    configure_cpu()
    p, lock = load_lock(root, repo_root())
    arrays = load_cache(root, p, lock)
    directory = root / "runs" / spec.key
    if not (directory / "train_complete.json").exists():
        raise FileNotFoundError(f"Train/eval incomplete: {spec.key}")
    model = load_selected_model(root, spec, p, lock)
    with np.load(directory / "traces.npz", allow_pickle=False) as data:
        traces = {name: data[name] for name in data.files}
    run_probes(directory, traces, arrays, core_run(spec), p)
    save_json(directory / "mechanism.json", mechanism_diagnostics(traces, arrays, spec, p))
    save_json(directory / "group_probes.json", {"rows": group_probe_rows(traces, arrays, spec, p)})
    save_json(directory / "gradient_profile.json", gradient_profile(model, arrays, spec, p))
    save_json(directory / "analysis_complete.json", {"status": "PASS", "run": spec.as_dict(), "core_identity": lock["identity"]})
    return {"run": spec.key, "status": "PASS"}


def evaluate_from_l2(head: nn.Linear, z: np.ndarray, y: np.ndarray, lengths: np.ndarray,
                     spec: ExtensionRun, p: Protocol, keep: np.ndarray) -> dict[str, float]:
    zt = torch.from_numpy(z.astype(np.float32)).clone()
    mask = torch.zeros(zt.shape[-1], dtype=zt.dtype)
    mask[torch.from_numpy(keep.astype(np.int64))] = 1
    zt *= mask
    with torch.no_grad():
        evidence = head(zt)
        out = residual_leakage_output(evidence, torch.from_numpy(lengths), beta=spec.beta,
                                      threshold=p.threshold, slope=p.surrogate_slope)
        pred = out["total_score"].argmax(1).numpy()
    return metrics(y, pred)


def retrain_head(z: dict[str, np.ndarray], arrays: dict[str, np.ndarray], source: nn.Linear,
                 keep: np.ndarray, spec: ExtensionRun, p: Protocol) -> nn.Linear:
    head = nn.Linear(source.in_features, source.out_features, bias=False)
    head.load_state_dict(source.state_dict())
    mask = torch.zeros(source.in_features)
    mask[torch.from_numpy(keep.astype(np.int64))] = 1
    optimizer = torch.optim.Adam(head.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
    ds = TensorDataset(torch.from_numpy(z["train"].astype(np.float32)), torch.from_numpy(arrays["train_y"]),
                       torch.from_numpy(arrays["train_lengths"]))
    gen = torch.Generator().manual_seed(spec.seed + 91827)
    dl = DataLoader(ds, batch_size=p.batch_size, shuffle=True, generator=gen, num_workers=0)

    def val_stats() -> tuple[float, float]:
        with torch.no_grad():
            zz = torch.from_numpy(z["val"].astype(np.float32)) * mask
            lengths = torch.from_numpy(arrays["val_lengths"])
            out = residual_leakage_output(head(zz), lengths, beta=spec.beta, threshold=p.threshold, slope=p.surrogate_slope)
            pred = out["total_score"].argmax(1).numpy()
            loss = float(extension_loss(out, lengths, torch.from_numpy(arrays["val_y"]), spec.objective))
            return metrics(arrays["val_y"], pred)["ba"], loss

    best_ba, best_loss = val_stats()
    best_state = cpu_state(head)
    best_epoch = 0
    for epoch in range(1, p.max_epochs + 1):
        head.train()
        for zz, y, lengths in dl:
            optimizer.zero_grad(set_to_none=True)
            out = residual_leakage_output(head(zz * mask), lengths, beta=spec.beta, threshold=p.threshold, slope=p.surrogate_slope)
            loss = extension_loss(out, lengths, y, spec.objective)
            loss.backward()
            head.weight.grad *= mask[None, :]
            optimizer.step()
        ba, loss = val_stats()
        if ba > best_ba + 1e-12 or (abs(ba - best_ba) <= 1e-12 and loss < best_loss - 1e-12):
            best_ba, best_loss, best_epoch, best_state = ba, loss, epoch, cpu_state(head)
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break
    head.load_state_dict(best_state)
    return head


def run_pruning(root: Path, spec: ExtensionRun) -> dict[str, Any]:
    configure_cpu()
    p, lock = load_lock(root, repo_root())
    arrays = load_cache(root, p, lock)
    directory = root / "runs" / spec.key
    model = load_selected_model(root, spec, p, lock)
    with np.load(directory / "traces.npz", allow_pickle=False) as data:
        z = {s: data[f"{s}__L2__spike"] for s in SPLITS}
    rates = z["train"].sum(axis=(0, 1)) / (arrays["train_lengths"].sum() / p.fs)
    order = np.argsort(rates)
    rows: list[dict[str, Any]] = []
    all_idx = np.arange(len(rates))
    for group in ("high", "low"):
        for fraction in (0.1, 0.2, 0.3):
            n = max(1, int(np.ceil(fraction * len(rates))))
            removed = order[-n:] if group == "high" else order[:n]
            keep = np.setdiff1d(all_idx, removed)
            row: dict[str, Any] = {"run_key": spec.key, "group": group, "fraction": fraction,
                                   "removed_n": n, "removed_mean_hz": float(rates[removed].mean())}
            for split in SPLITS:
                vals = evaluate_from_l2(model.head, z[split], arrays[f"{split}_y"], arrays[f"{split}_lengths"], spec, p, keep)
                row.update({f"fixedW_{split}_{k}": v for k, v in vals.items()})
            adapted = retrain_head(z, arrays, model.head, keep, spec, p)
            for split in SPLITS:
                vals = evaluate_from_l2(adapted, z[split], arrays[f"{split}_y"], arrays[f"{split}_lengths"], spec, p, keep)
                row.update({f"retrainW_{split}_{k}": v for k, v in vals.items()})
            rows.append(row)
    save_json(directory / "pruning.json", {"rows": rows})
    save_json(directory / "pruning_complete.json", {"status": "PASS", "run": spec.as_dict()})
    return {"run": spec.key, "status": "PASS", "rows": len(rows)}

def finalize(root: Path, *, require_pruning: bool = True) -> dict[str, Any]:
    import pandas as pd
    specs = available_runs(root)
    native_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    firing_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    pruning_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    for spec in specs:
        d = root / "runs" / spec.key
        required = ["train_complete.json", "analysis_complete.json", "native.json", "probes.json",
                    "group_probes.json", "mechanism.json", "gradient_profile.json"]
        if require_pruning:
            required += ["pruning_complete.json", "pruning.json"]
        missing = [name for name in required if not (d / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{spec.key}: missing {missing}")
        native = json.loads((d / "native.json").read_text())
        for split, values in native["splits"].items():
            native_rows.append({"run_key": spec.key, **spec.as_dict(), "split": split, **values})
        firing_rows.extend({"run_key": spec.key, **spec.as_dict(), **row} for row in native["activity"])
        probes = json.loads((d / "probes.json").read_text())["rows"]
        probe_rows.extend({**spec.as_dict(), **row} for row in probes)
        groups = json.loads((d / "group_probes.json").read_text())["rows"]
        group_rows.extend({**spec.as_dict(), **row} for row in groups)
        mech = json.loads((d / "mechanism.json").read_text())
        run_rows.extend({"run_key": spec.key, **spec.as_dict(), **row} for row in mech["rows"])
        if require_pruning:
            pruning = json.loads((d / "pruning.json").read_text())["rows"]
            pruning_rows.extend({**spec.as_dict(), **row} for row in pruning)
        grad = json.loads((d / "gradient_profile.json").read_text())
        for b, (value, count) in enumerate(zip(grad["mean_dloss_devidence_norm"], grad["counts"])):
            gradient_rows.append({"run_key": spec.key, **spec.as_dict(), "time_bin": b,
                                  "mean_dloss_devidence_norm": value, "count": count})
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    tables = {
        "native_runs.csv": native_rows,
        "probe_runs.csv": probe_rows,
        "group_probe_runs.csv": group_rows,
        "firing_runs.csv": firing_rows,
        "run_length_runs.csv": run_rows,
        "gradient_runs.csv": gradient_rows,
    }
    if require_pruning:
        tables["pruning_runs.csv"] = pruning_rows
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(aggregate / name, index=False)
    native_df = pd.DataFrame(native_rows)
    test = native_df[native_df["split"] == "test"]
    numeric = [c for c in ("ba", "accuracy", "macro_f1", "spike_only_ba", "analog_counterfactual_ba",
                            "total_vs_spike_disagreement", "total_vs_analog_disagreement", "final_residual_l1",
                            "leakage_drift_l1", "backlog_gt_1theta", "backlog_gt_2theta", "backlog_gt_4theta") if c in test]
    summary = test.groupby(["objective", "beta"])[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "") if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(aggregate / "native_summary.csv", index=False)
    manifest = {"status": "PASS", "extension_id": EXTENSION_ID, "version": EXTENSION_VERSION,
                "expected_runs": len(specs), "completed_runs": len(specs), "require_pruning": require_pruning,
                "tables": sorted(tables) + ["native_summary.csv"]}
    save_json(aggregate / "manifest.json", manifest)
    return manifest


def resolve_spec(root: Path, run_key: str | None, task_id: int | None) -> ExtensionRun:
    specs = available_runs(root)
    if run_key is not None:
        lookup = {r.key: r for r in specs}
        if run_key not in lookup:
            raise ValueError(f"Unknown run: {run_key}")
        return lookup[run_key]
    if task_id is None or not 0 <= task_id < len(specs):
        raise ValueError(f"task-id must be in [0, {len(specs)-1}]")
    return specs[task_id]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=repo_root() / "core_benchmark_v1/results/output_residual_leakage_v1")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    prep = sub.add_parser("prepare")
    prep.add_argument("--synthetic", action="store_true")
    for name in ("run", "analyze", "prune"):
        cmd = sub.add_parser(name)
        choose = cmd.add_mutually_exclusive_group(required=True)
        choose.add_argument("--run-key")
        choose.add_argument("--task-id", type=int)
    final = sub.add_parser("finalize")
    final.add_argument("--no-pruning", action="store_true")
    sub.add_parser("smoke")
    args = parser.parse_args(argv)
    root = args.results.expanduser().resolve()
    if args.command == "plan":
        print(json.dumps(extension_manifest(), indent=2))
        return
    if args.command == "prepare":
        configure_cpu()
        print(json.dumps(prepare(root, synthetic=args.synthetic), indent=2))
        return
    if args.command == "smoke":
        configure_cpu()
        prepare(root, synthetic=True)
        for spec in available_runs(root):
            run_formal(root, spec)
            run_analysis(root, spec)
            run_pruning(root, spec)
        print(json.dumps(finalize(root), indent=2))
        return
    if args.command == "finalize":
        print(json.dumps(finalize(root, require_pruning=not args.no_pruning), indent=2))
        return
    spec = resolve_spec(root, args.run_key, args.task_id)
    fn = {"run": run_formal, "analyze": run_analysis, "prune": run_pruning}[args.command]
    print(json.dumps(fn(root, spec), indent=2))


if __name__ == "__main__":
    main()