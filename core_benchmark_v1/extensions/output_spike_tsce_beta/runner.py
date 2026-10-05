"""Training, evaluation and aggregation for spike-only TSCE output beta sweeps."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from core_benchmark_v1.data import load_cache, loader, prepare as prepare_core
from core_benchmark_v1.model import BenchmarkNet, valid_sum
from core_benchmark_v1.protocol import Protocol, Run, SPLITS
from core_benchmark_v1.storage import (
    load_lock,
    load_torch,
    save_json,
    save_npz,
    save_torch,
    state_hash,
)
from core_benchmark_v1.training import cpu_state, metrics

from . import EXTENSION_ID, EXTENSION_VERSION
from .model import native_count, spike_tsce_loss, spike_tsce_output
from .protocol import BETAS, DRAIN_BETA, MAX_DRAIN_STEPS, MODES, SEEDS, ExtensionRun, runs


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
    return replace(
        Protocol(),
        profile="smoke",
        seeds=(11,),
        width=6,
        input_channels=3,
        total_channels=9,
        steps=32,
        labels=("A", "B", "C"),
        batch_size=9,
        max_epochs=1,
        min_epochs=1,
        patience=1,
        c_grid=(0.1,),
        probe_max_iter=1000,
        shuffle_seeds=(101,),
        relative_bins=4,
        lags=(1, 2),
        history_ms=(125.0, 250.0),
    )


def core_run(spec: ExtensionRun) -> Run:
    # Core parameter RNG is keyed only by protocol version, seed and parameter
    # role. Case/mode/beta therefore do not perturb paired initialization.
    return Run(spec.key, spec.seed, "08_output_spike_tsce_beta", objective="wcce")


def extension_manifest() -> dict[str, Any]:
    return {
        "extension_id": EXTENSION_ID,
        "version": EXTENSION_VERSION,
        "betas": BETAS,
        "modes": MODES,
        "seeds": SEEDS,
        "objective": "core_style_per_sample_tsce_on_output_spikes",
        "selection": "native spike-count validation BA then native mean-count CE",
        "drain_beta": DRAIN_BETA,
        "max_drain_steps": MAX_DRAIN_STEPS,
        "positive_only_spikes": True,
        "output_spike_cap": 1,
        "post_valid_new_evidence": False,
        "drain_adds_tsce_timesteps": False,
        "drain_assignment": "all endpoint drain spikes are added to the final valid TSCE timestep",
        "analog_used_for_training": False,
        "analog_used_for_selection": False,
        "analog_counterfactual_is_diagnostic_only": True,
        "runs": [run.as_dict() | {"key": run.key} for run in runs()],
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
    allowed_seeds = tuple(int(value) for value in manifest["seeds"])
    allowed_betas = tuple(float(value) for value in manifest["betas"])
    allowed_modes = tuple(manifest.get("modes", MODES))
    return [
        run
        for run in runs()
        if run.seed in allowed_seeds
        and run.beta in allowed_betas
        and run.mode in allowed_modes
    ]


def validate_checkpoint(payload: dict[str, Any], spec: ExtensionRun, core_identity: str) -> None:
    if payload.get("extension_version") != EXTENSION_VERSION:
        raise ValueError(f"Extension version mismatch: {spec.key}")
    if payload.get("core_identity") != core_identity:
        raise ValueError(f"Core identity mismatch: {spec.key}")
    if payload.get("run") != spec.as_dict():
        raise ValueError(f"Extension run mismatch: {spec.key}")


def forward_output(
    model: BenchmarkNet,
    x: torch.Tensor,
    lengths: torch.Tensor,
    spec: ExtensionRun,
    p: Protocol,
    *,
    include_drain: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    hidden = model(x, lengths)
    output = spike_tsce_output(
        hidden["evidence"],
        lengths,
        beta=spec.beta,
        threshold=p.threshold,
        slope=p.surrogate_slope,
        max_drain_steps=MAX_DRAIN_STEPS,
        include_drain=include_drain,
    )
    if include_drain and bool(output["cap_hit"].item()):
        raise RuntimeError(
            f"{spec.key}: required output drain exceeds MAX_DRAIN_STEPS={MAX_DRAIN_STEPS}"
        )
    return hidden, output


def _sample_diagnostics(output: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    final_u = output["final_membrane"].detach()
    valid_count = output["valid_count"].detach()
    drain_count = output["drain_count"].detach()
    total_count = output["total_count"].detach()
    required = output["required_steps_per_unit"].detach()
    return {
        "valid_spikes": valid_count.sum(1).cpu().numpy(),
        "drain_spikes": drain_count.sum(1).cpu().numpy(),
        "total_spikes": total_count.sum(1).cpu().numpy(),
        "drain_fraction": (
            drain_count.sum(1) / total_count.sum(1).clamp_min(1)
        ).cpu().numpy(),
        "k_required": required.max(1).values.cpu().numpy(),
        "positive_residual_l1": final_u.clamp_min(0).sum(1).cpu().numpy(),
        "negative_residual_l1": (-final_u.clamp_max(0)).sum(1).cpu().numpy(),
    }


@torch.no_grad()
def evaluate_split(
    model: BenchmarkNet,
    arrays: dict[str, np.ndarray],
    split: str,
    spec: ExtensionRun,
    p: Protocol,
) -> dict[str, Any]:
    model.eval()
    ys: list[np.ndarray] = []
    native_predictions: list[np.ndarray] = []
    valid_predictions: list[np.ndarray] = []
    drained_predictions: list[np.ndarray] = []
    analog_predictions: list[np.ndarray] = []
    diagnostics: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "valid_spikes",
            "drain_spikes",
            "total_spikes",
            "drain_fraction",
            "k_required",
            "positive_residual_l1",
            "negative_residual_l1",
        )
    }
    objective_sum = 0.0
    native_ce_sum = 0.0
    valid_ce_sum = 0.0
    drained_ce_sum = 0.0
    count = 0

    for x, y, lengths in loader(arrays, split, p, spec.seed):
        hidden, output = forward_output(model, x, lengths, spec, p, include_drain=True)
        selected_count = native_count(output, spec.mode)
        analog_score = valid_sum(hidden["evidence"], lengths)
        objective_sum += float(spike_tsce_loss(output, lengths, y, spec.mode)) * len(y)
        native_ce_sum += float(
            F.cross_entropy(selected_count / lengths[:, None], y, reduction="sum")
        )
        valid_ce_sum += float(
            F.cross_entropy(output["valid_mean_logits"], y, reduction="sum")
        )
        drained_ce_sum += float(
            F.cross_entropy(output["drained_mean_logits"], y, reduction="sum")
        )
        count += len(y)
        ys.append(y.cpu().numpy())
        native_predictions.append(selected_count.argmax(1).cpu().numpy())
        valid_predictions.append(output["valid_count"].argmax(1).cpu().numpy())
        drained_predictions.append(output["total_count"].argmax(1).cpu().numpy())
        analog_predictions.append(analog_score.argmax(1).cpu().numpy())
        sample = _sample_diagnostics(output)
        for key, value in sample.items():
            diagnostics[key].append(value)

    y_all = np.concatenate(ys)
    native = np.concatenate(native_predictions)
    valid = np.concatenate(valid_predictions)
    drained = np.concatenate(drained_predictions)
    analog = np.concatenate(analog_predictions)
    native_metrics = metrics(y_all, native)
    valid_metrics = metrics(y_all, valid)
    drained_metrics = metrics(y_all, drained)
    analog_metrics = metrics(y_all, analog)
    result: dict[str, Any] = {
        **native_metrics,
        "native_ba": native_metrics["ba"],
        "valid_only_ba": valid_metrics["ba"],
        "drained_ba": drained_metrics["ba"],
        "analog_counterfactual_ba": analog_metrics["ba"],
        "drain_gain_ba": drained_metrics["ba"] - valid_metrics["ba"],
        "objective_tsce": objective_sum / count,
        "native_mean_logit_ce": native_ce_sum / count,
        "valid_mean_logit_ce": valid_ce_sum / count,
        "drained_mean_logit_ce": drained_ce_sum / count,
        "native_vs_analog_disagreement": float((native != analog).mean()),
        "drained_vs_valid_disagreement": float((drained != valid).mean()),
        "n_samples": count,
        "cap_hit_fraction": 0.0,
    }
    flat = {key: np.concatenate(parts) for key, parts in diagnostics.items()}
    result.update(
        {
            "valid_spike_count_mean": float(flat["valid_spikes"].mean()),
            "drain_spike_count_mean": float(flat["drain_spikes"].mean()),
            "total_spike_count_mean": float(flat["total_spikes"].mean()),
            "drain_spike_fraction_mean": float(flat["drain_fraction"].mean()),
            "k_required_mean": float(flat["k_required"].mean()),
            "k_required_median": float(np.median(flat["k_required"])),
            "k_required_p90": float(np.percentile(flat["k_required"], 90)),
            "k_required_p99": float(np.percentile(flat["k_required"], 99)),
            "k_required_max": int(flat["k_required"].max(initial=0)),
            "positive_residual_l1_mean": float(flat["positive_residual_l1"].mean()),
            "negative_residual_l1_mean": float(flat["negative_residual_l1"].mean()),
        }
    )
    return result


def evaluate_validation(
    model: BenchmarkNet,
    arrays: dict[str, np.ndarray],
    spec: ExtensionRun,
    p: Protocol,
) -> dict[str, Any]:
    return evaluate_split(model, arrays, "val", spec, p)


def train(
    root: Path,
    spec: ExtensionRun,
    p: Protocol,
    lock: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> BenchmarkNet:
    directory = root / "runs" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    run = core_run(spec)
    if checkpoint_path.exists():
        payload = load_torch(checkpoint_path)
        validate_checkpoint(payload, spec, lock["identity"])
        if not payload.get("training_complete"):
            raise ValueError(f"Incomplete checkpoint: {spec.key}")
        model = BenchmarkNet(run, p)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return model

    model = BenchmarkNet(run, p)
    initial = cpu_state(model)
    parameter_hashes = {
        name: state_hash({name: tensor}) for name, tensor in model.named_parameters()
    }
    save_torch(
        directory / "initial.pt",
        {
            "extension_version": EXTENSION_VERSION,
            "core_identity": lock["identity"],
            "run": spec.as_dict(),
            "model_state_dict": initial,
            "initial_parameter_hashes": parameter_hashes,
        },
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
    )
    train_batches = loader(arrays, "train", p, spec.seed, shuffle=True)
    best = evaluate_validation(model, arrays, spec, p)
    best_state, best_epoch = initial, 0
    history = [
        {
            "epoch": 0,
            "train_loss": None,
            "val_native_ba": best["native_ba"],
            "val_native_mean_logit_ce": best["native_mean_logit_ce"],
            "val_valid_only_ba": best["valid_only_ba"],
            "val_drained_ba": best["drained_ba"],
            "val_analog_counterfactual_ba": best["analog_counterfactual_ba"],
        }
    ]
    epoch = 0
    for epoch in range(1, p.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for x, y, lengths in train_batches:
            optimizer.zero_grad(set_to_none=True)
            _, output = forward_output(
                model,
                x,
                lengths,
                spec,
                p,
                include_drain=spec.mode == "drain",
            )
            loss = spike_tsce_loss(output, lengths, y, spec.mode)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite spike TSCE loss")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"{spec.key}: nonfinite spike TSCE gradient")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            count += len(y)

        val = evaluate_validation(model, arrays, spec, p)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / count,
                "val_native_ba": val["native_ba"],
                "val_native_mean_logit_ce": val["native_mean_logit_ce"],
                "val_valid_only_ba": val["valid_only_ba"],
                "val_drained_ba": val["drained_ba"],
                "val_analog_counterfactual_ba": val["analog_counterfactual_ba"],
            }
        )
        improved = val["native_ba"] > best["native_ba"] + 1e-12 or (
            abs(val["native_ba"] - best["native_ba"]) <= 1e-12
            and val["native_mean_logit_ce"] < best["native_mean_logit_ce"] - 1e-12
        )
        if improved:
            best = val
            best_epoch = epoch
            best_state = cpu_state(model)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{spec.key} epoch={epoch} train_loss={loss_sum/count:.5f} "
                f"val_native_ba={val['native_ba']:.5f} "
                f"val_valid_ba={val['valid_only_ba']:.5f} "
                f"val_drained_ba={val['drained_ba']:.5f}",
                flush=True,
            )
            save_json(directory / "progress.json", history[-1])
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    model.load_state_dict(best_state)
    save_json(directory / "history.json", {"rows": history})
    save_torch(
        checkpoint_path,
        {
            "extension_version": EXTENSION_VERSION,
            "core_identity": lock["identity"],
            "run": spec.as_dict(),
            "model_state_dict": best_state,
            "training_complete": True,
            "best_epoch": best_epoch,
            "stopped_epoch": epoch,
            "best_val": best,
            "initial_parameter_hashes": parameter_hashes,
            "trainable_parameters": sum(value.numel() for value in model.parameters()),
            "training_objective": "per_sample_valid_mean_output_spike_tsce",
            "output_beta": spec.beta,
            "mode": spec.mode,
            "drain_beta": DRAIN_BETA,
            "output_threshold": p.threshold,
            "output_spike_cap": 1,
            "positive_only_spikes": True,
            "post_valid_new_evidence": False,
            "max_drain_steps": MAX_DRAIN_STEPS,
            "drain_adds_tsce_timesteps": False,
            "selection_rule": (
                "validation native spike-count BA, then native mean-count CE, then earliest epoch"
            ),
        },
    )
    return model


@torch.no_grad()
def extract(
    model: BenchmarkNet,
    arrays: dict[str, np.ndarray],
    spec: ExtensionRun,
    p: Protocol,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model.eval()
    run = core_run(spec)
    traces: dict[str, np.ndarray] = {}
    native: dict[str, Any] = {
        "run_key": spec.key,
        "mode": spec.mode,
        "beta": spec.beta,
        "seed": spec.seed,
        "objective": "spike_tsce",
        "selection_interface": "spike_count_only",
        "analog_counterfactual_role": "diagnostic_only",
        "drain_beta": DRAIN_BETA,
        "positive_only_spikes": True,
        "splits": {},
        "users": [],
        "activity": [],
    }
    for split in SPLITS:
        native["splits"][split] = evaluate_split(model, arrays, split, spec, p)
        chunks: dict[str, list[np.ndarray]] = {
            "evidence": [],
            "output_spike": [],
            "output_pre_reset": [],
            "output_final_membrane": [],
            "output_valid_count": [],
            "output_drain_count": [],
            "output_drained_trajectory": [],
        }
        for state in ("spike", "pre_reset"):
            for li in range(len(run.shifts)):
                chunks[f"L{li+1}__{state}"] = []
        ys: list[np.ndarray] = []
        native_predictions: list[np.ndarray] = []
        for x, y, lengths in loader(arrays, split, p, spec.seed):
            hidden, output = forward_output(model, x, lengths, spec, p, include_drain=True)
            chunks["evidence"].append(hidden["evidence"].cpu().numpy().astype(np.float32))
            chunks["output_spike"].append(output["spike"].cpu().numpy().astype(np.uint8))
            chunks["output_pre_reset"].append(output["pre_reset"].cpu().numpy().astype(np.float32))
            chunks["output_final_membrane"].append(
                output["final_membrane"].cpu().numpy().astype(np.float32)
            )
            chunks["output_valid_count"].append(
                output["valid_count"].cpu().numpy().astype(np.float32)
            )
            chunks["output_drain_count"].append(
                output["drain_count"].cpu().numpy().astype(np.float32)
            )
            chunks["output_drained_trajectory"].append(
                output["drained_trajectory"].cpu().numpy().astype(np.float32)
            )
            for state in ("spike", "pre_reset"):
                for li in range(len(run.shifts)):
                    value = hidden[state][li].cpu().numpy()
                    chunks[f"L{li+1}__{state}"].append(
                        value.astype(np.uint8 if state == "spike" else np.float32)
                    )
            ys.append(y.cpu().numpy())
            native_predictions.append(native_count(output, spec.mode).argmax(1).cpu().numpy())

        for key, pieces in chunks.items():
            traces[f"{split}__{key}"] = np.concatenate(pieces)
        y_all = np.concatenate(ys)
        prediction_all = np.concatenate(native_predictions)
        traces[f"{split}__native_prediction"] = prediction_all

        for user in np.unique(arrays[f"{split}_users"]):
            selected = arrays[f"{split}_users"] == user
            native["users"].append(
                {
                    "split": split,
                    "user": str(user),
                    "n": int(selected.sum()),
                    **metrics(y_all[selected], prediction_all[selected]),
                }
            )

        lengths_all = arrays[f"{split}_lengths"]
        mask = np.arange(p.steps)[None, :] < lengths_all[:, None]
        duration_s = float(lengths_all.sum()) / p.fs
        for li in range(len(run.shifts)):
            spikes = traces[f"{split}__L{li+1}__spike"]
            flat = spikes[mask]
            rates = spikes.sum(axis=(0, 1)) / max(duration_s, 1e-12)
            native["activity"].append(
                {
                    "split": split,
                    "layer": f"L{li+1}",
                    "mean_hz": float(rates.mean()),
                    "median_hz": float(np.median(rates)),
                    "p90_hz": float(np.percentile(rates, 90)),
                    "max_hz": float(rates.max()),
                    "fraction_lt1hz": float((rates < 1).mean()),
                    "fraction_lt5hz": float((rates < 5).mean()),
                    "fraction_gt10hz": float((rates > 10).mean()),
                    "fraction_gt20hz": float((rates > 20).mean()),
                    "fraction_gt30hz": float((rates > 30).mean()),
                    "nonzero_fraction": float((flat != 0).mean()),
                }
            )

    native["train_test_gap"] = (
        native["splits"]["train"]["native_ba"]
        - native["splits"]["test"]["native_ba"]
    )
    return traces, native


def _complete_is_valid(root: Path, spec: ExtensionRun, core_identity: str) -> bool:
    directory = root / "runs" / spec.key
    path = directory / "complete.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    required = ("checkpoint.pt", "history.json", "native.json", "traces.npz")
    return (
        payload.get("status") == "PASS"
        and payload.get("extension_version") == EXTENSION_VERSION
        and payload.get("core_identity") == core_identity
        and payload.get("run") == spec.as_dict()
        and all((directory / name).is_file() for name in required)
    )


def run_formal(root: Path, spec: ExtensionRun) -> dict[str, Any]:
    configure_cpu()
    p, lock = load_lock(root, repo_root())
    if _complete_is_valid(root, spec, lock["identity"]):
        native = json.loads((root / "runs" / spec.key / "native.json").read_text())
        return {
            "run": spec.key,
            "status": "already_complete",
            "test_native_ba": native["splits"]["test"]["native_ba"],
        }
    arrays = load_cache(root, p, lock)
    model = train(root, spec, p, lock, arrays)
    traces, native = extract(model, arrays, spec, p)
    directory = root / "runs" / spec.key
    save_npz(directory / "traces.npz", traces)
    save_json(directory / "native.json", native)
    save_json(
        directory / "complete.json",
        {
            "status": "PASS",
            "extension_version": EXTENSION_VERSION,
            "run": spec.as_dict(),
            "core_identity": lock["identity"],
            "required_artifacts": [
                "checkpoint.pt",
                "history.json",
                "native.json",
                "traces.npz",
            ],
        },
    )
    return {
        "run": spec.key,
        "status": "PASS",
        "test_native_ba": native["splits"]["test"]["native_ba"],
        "test_valid_only_ba": native["splits"]["test"]["valid_only_ba"],
        "test_drained_ba": native["splits"]["test"]["drained_ba"],
        "test_analog_counterfactual_ba": native["splits"]["test"]["analog_counterfactual_ba"],
    }


def smoke(root: Path) -> dict[str, Any]:
    """Cheap real-data preflight for both modes and beta endpoints."""
    configure_cpu()
    p, lock = load_lock(root, repo_root())
    arrays = load_cache(root, p, lock)
    batch = next(iter(loader(arrays, "train", p, 11, shuffle=True)))
    x, y, lengths = batch
    rows: list[dict[str, Any]] = []
    smoke_dir = root / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        for beta in (0.0, 1.0):
            spec = ExtensionRun(mode, beta, 11)
            model = BenchmarkNet(core_run(spec), p)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
            )
            optimizer.zero_grad(set_to_none=True)
            _, output = forward_output(
                model,
                x,
                lengths,
                spec,
                p,
                include_drain=mode == "drain",
            )
            loss = spike_tsce_loss(output, lengths, y, mode)
            loss.backward()
            checked = (model.head.weight, model.layers[0].weight, model.layers[1].weight)
            if any(value.grad is None or not torch.isfinite(value.grad).all() for value in checked):
                raise RuntimeError(f"{spec.key}: smoke gradient did not reach the full network")
            optimizer.step()
            payload = {
                "extension_version": EXTENSION_VERSION,
                "core_identity": lock["identity"],
                "run": spec.as_dict(),
                "model_state_dict": cpu_state(model),
                "training_complete": False,
            }
            checkpoint = smoke_dir / f"{spec.key}.pt"
            save_torch(checkpoint, payload)
            restored = load_torch(checkpoint)
            if restored["run"] != spec.as_dict():
                raise RuntimeError(f"{spec.key}: smoke checkpoint round trip failed")
            restored_model = BenchmarkNet(core_run(spec), p)
            restored_model.load_state_dict(restored["model_state_dict"], strict=True)
            restored_model.eval()
            with torch.no_grad():
                _, evaluated = forward_output(
                    restored_model, x, lengths, spec, p, include_drain=True
                )
                prediction = native_count(evaluated, mode).argmax(1).cpu().numpy()
                batch_metrics = metrics(y.cpu().numpy(), prediction)
            rows.append(
                {
                    "run": spec.key,
                    "loss": float(loss.detach()),
                    "batch_ba": batch_metrics["ba"],
                    "native_count_mean": float(
                        native_count(evaluated, mode).sum(1).mean()
                    ),
                }
            )
    result = {
        "status": "PASS",
        "core_identity": lock["identity"],
        "cases": rows,
    }
    save_json(smoke_dir / "smoke.json", result)
    return result


def finalize(root: Path) -> dict[str, Any]:
    import pandas as pd

    p, lock = load_lock(root, repo_root())
    del p
    specs = available_runs(root)
    native_rows: list[dict[str, Any]] = []
    firing_rows: list[dict[str, Any]] = []
    for spec in specs:
        if not _complete_is_valid(root, spec, lock["identity"]):
            raise FileNotFoundError(f"{spec.key}: missing or stale required completion artifacts")
        directory = root / "runs" / spec.key
        checkpoint = load_torch(directory / "checkpoint.pt")
        validate_checkpoint(checkpoint, spec, lock["identity"])
        native = json.loads((directory / "native.json").read_text())
        for split, values in native["splits"].items():
            native_rows.append(
                {"run_key": spec.key, **spec.as_dict(), "split": split, **values}
            )
        firing_rows.extend(
            {"run_key": spec.key, **spec.as_dict(), **row}
            for row in native["activity"]
        )

    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(native_rows).to_csv(aggregate / "native_runs.csv", index=False)
    pd.DataFrame(firing_rows).to_csv(aggregate / "firing_runs.csv", index=False)

    native_df = pd.DataFrame(native_rows)
    test = native_df[native_df["split"] == "test"]
    numeric = [
        column
        for column in (
            "native_ba",
            "valid_only_ba",
            "drained_ba",
            "analog_counterfactual_ba",
            "drain_gain_ba",
            "objective_tsce",
            "native_mean_logit_ce",
            "native_vs_analog_disagreement",
            "drained_vs_valid_disagreement",
            "valid_spike_count_mean",
            "drain_spike_count_mean",
            "drain_spike_fraction_mean",
            "k_required_mean",
            "k_required_p90",
            "k_required_p99",
            "k_required_max",
            "positive_residual_l1_mean",
            "negative_residual_l1_mean",
        )
        if column in test
    ]
    summary = test.groupby(["mode", "beta"])[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(value) for value in column if value != "")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(aggregate / "native_summary.csv", index=False)
    manifest = {
        "status": "PASS",
        "extension_id": EXTENSION_ID,
        "version": EXTENSION_VERSION,
        "core_identity": lock["identity"],
        "expected_runs": len(specs),
        "completed_runs": len(specs),
        "tables": ["native_runs.csv", "native_summary.csv", "firing_runs.csv"],
    }
    save_json(aggregate / "manifest.json", manifest)
    return manifest


def resolve_spec(root: Path, run_key: str | None, task_id: int | None) -> ExtensionRun:
    specs = available_runs(root)
    if run_key is not None:
        lookup = {run.key: run for run in specs}
        if run_key not in lookup:
            raise ValueError(f"Unknown run: {run_key}")
        return lookup[run_key]
    if task_id is None or not 0 <= task_id < len(specs):
        raise ValueError(f"task-id must be in [0, {len(specs)-1}]")
    return specs[task_id]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=repo_root() / "core_benchmark_v1/results/output_spike_tsce_beta_v1",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    prep = sub.add_parser("prepare")
    prep.add_argument("--synthetic", action="store_true")
    run_parser = sub.add_parser("run")
    choose = run_parser.add_mutually_exclusive_group(required=True)
    choose.add_argument("--run-key")
    choose.add_argument("--task-id", type=int)
    sub.add_parser("smoke")
    sub.add_parser("finalize")
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
        print(json.dumps(smoke(root), indent=2))
        return
    if args.command == "finalize":
        print(json.dumps(finalize(root), indent=2))
        return

    spec = resolve_spec(root, args.run_key, args.task_id)
    print(json.dumps(run_formal(root, spec), indent=2))


if __name__ == "__main__":
    main()
