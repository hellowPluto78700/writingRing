"""Training, evaluation, probes and aggregation for the spike-drain extension."""
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
from core_benchmark_v1.model import BenchmarkNet
from core_benchmark_v1.probes import run_probes
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
from .model import drained_spike_output, drain_loss
from .protocol import BETAS, DRAIN_BETA, MAX_DRAIN_STEPS, SEEDS, ExtensionRun, runs


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
    # Core paired initialization is keyed by seed/parameter role, not case name.
    return Run(spec.key, spec.seed, "08_output_spike_drain", objective="wcce")


def extension_manifest() -> dict[str, Any]:
    return {
        "extension_id": EXTENSION_ID,
        "version": EXTENSION_VERSION,
        "betas": BETAS,
        "seeds": SEEDS,
        "objective": "wcce",
        "drain_beta": DRAIN_BETA,
        "max_drain_steps": MAX_DRAIN_STEPS,
        "positive_only_spikes": True,
        "post_valid_new_evidence": False,
        "strict_reference": {
            "case": "R_LIF_E2E",
            "beta": 0.5,
            "source": "CoreBenchmark v1 readout control",
            "retrained_here": False,
        },
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
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    hidden = model(x, lengths)
    output = drained_spike_output(
        hidden["evidence"],
        lengths,
        beta=spec.beta,
        threshold=p.threshold,
        slope=p.surrogate_slope,
        max_drain_steps=MAX_DRAIN_STEPS,
    )
    if bool(output["cap_hit"].item()):
        raise RuntimeError(
            f"{spec.key}: required output drain exceeds MAX_DRAIN_STEPS={MAX_DRAIN_STEPS}"
        )
    return hidden, output


def _sample_diagnostics(out: dict[str, torch.Tensor], p: Protocol) -> dict[str, np.ndarray]:
    final_u = out["final_membrane"].detach()
    valid_count = out["valid_count"].detach()
    drain_count = out["drain_count"].detach()
    total_count = out["total_count"].detach()
    required = out["required_steps_per_unit"].detach()
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
    drained_pred: list[np.ndarray] = []
    valid_pred: list[np.ndarray] = []
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
    loss_sum = 0.0
    count = 0
    for x, y, lengths in loader(arrays, split, p, spec.seed):
        _, out = forward_output(model, x, lengths, spec, p)
        loss_sum += float(
            F.cross_entropy(out["drained_mean_logits"], y, reduction="sum")
        )
        count += len(y)
        ys.append(y.numpy())
        drained_pred.append(out["total_count"].argmax(1).cpu().numpy())
        valid_pred.append(out["valid_count"].argmax(1).cpu().numpy())
        sample = _sample_diagnostics(out, p)
        for key, value in sample.items():
            diagnostics[key].append(value)

    y_all = np.concatenate(ys)
    drained = np.concatenate(drained_pred)
    valid = np.concatenate(valid_pred)
    result = {
        **metrics(y_all, drained),
        "drained_ba": metrics(y_all, drained)["ba"],
        "valid_only_ba": metrics(y_all, valid)["ba"],
        "drain_gain_ba": metrics(y_all, drained)["ba"] - metrics(y_all, valid)["ba"],
        "drained_mean_logit_ce": loss_sum / count,
        "prediction_disagreement": float((drained != valid).mean()),
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
            "val_ba": best["drained_ba"],
            "val_drained_mean_logit_ce": best["drained_mean_logit_ce"],
            "val_valid_only_ba": best["valid_only_ba"],
            "val_drain_spike_fraction": best["drain_spike_fraction_mean"],
            "val_k_required_p99": best["k_required_p99"],
        }
    ]
    epoch = 0
    for epoch in range(1, p.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for x, y, lengths in train_batches:
            optimizer.zero_grad(set_to_none=True)
            _, out = forward_output(model, x, lengths, spec, p)
            loss = drain_loss(out, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite drain loss")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"{spec.key}: nonfinite drain gradient")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            count += len(y)

        val = evaluate_validation(model, arrays, spec, p)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / count,
                "val_ba": val["drained_ba"],
                "val_drained_mean_logit_ce": val["drained_mean_logit_ce"],
                "val_valid_only_ba": val["valid_only_ba"],
                "val_drain_spike_fraction": val["drain_spike_fraction_mean"],
                "val_k_required_p99": val["k_required_p99"],
            }
        )
        improved = val["drained_ba"] > best["drained_ba"] + 1e-12 or (
            abs(val["drained_ba"] - best["drained_ba"]) <= 1e-12
            and val["drained_mean_logit_ce"]
            < best["drained_mean_logit_ce"] - 1e-12
        )
        if improved:
            best = val
            best_epoch = epoch
            best_state = cpu_state(model)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{spec.key} epoch={epoch} train_loss={loss_sum/count:.5f} "
                f"val_drained_ba={val['drained_ba']:.5f} "
                f"val_valid_only_ba={val['valid_only_ba']:.5f}",
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
            "trainable_parameters": sum(v.numel() for v in model.parameters()),
            "output_beta": spec.beta,
            "drain_beta": DRAIN_BETA,
            "output_threshold": p.threshold,
            "output_spike_cap": 1,
            "positive_only_spikes": True,
            "post_valid_new_evidence": False,
            "max_drain_steps": MAX_DRAIN_STEPS,
            "selection_rule": (
                "validation drained BA, then drained mean-logit CE, then earliest epoch"
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
        "beta": spec.beta,
        "seed": spec.seed,
        "objective": "wcce",
        "drain_beta": DRAIN_BETA,
        "positive_only_spikes": True,
        "splits": {},
        "users": [],
        "activity": [],
    }
    for split in SPLITS:
        chunks: dict[str, list[np.ndarray]] = {
            "evidence": [],
            "output_spike": [],
            "output_pre_reset": [],
            "output_final_membrane": [],
            "output_valid_count": [],
            "output_drain_count": [],
        }
        for state in ("spike", "pre_reset"):
            for li in range(len(run.shifts)):
                chunks[f"L{li+1}__{state}"] = []
        ys: list[np.ndarray] = []
        drained_preds: list[np.ndarray] = []
        valid_preds: list[np.ndarray] = []
        for x, y, lengths in loader(arrays, split, p, spec.seed):
            hidden, out = forward_output(model, x, lengths, spec, p)
            chunks["evidence"].append(hidden["evidence"].cpu().numpy())
            chunks["output_spike"].append(out["spike"].cpu().numpy().astype(np.uint8))
            chunks["output_pre_reset"].append(out["pre_reset"].cpu().numpy().astype(np.float32))
            chunks["output_final_membrane"].append(out["final_membrane"].cpu().numpy().astype(np.float32))
            chunks["output_valid_count"].append(out["valid_count"].cpu().numpy().astype(np.float32))
            chunks["output_drain_count"].append(out["drain_count"].cpu().numpy().astype(np.float32))
            for state in ("spike", "pre_reset"):
                for li in range(len(run.shifts)):
                    value = hidden[state][li].cpu().numpy()
                    chunks[f"L{li+1}__{state}"].append(
                        value.astype(np.uint8 if state == "spike" else np.float32)
                    )
            ys.append(y.numpy())
            drained_preds.append(out["total_count"].argmax(1).cpu().numpy())
            valid_preds.append(out["valid_count"].argmax(1).cpu().numpy())

        for key, pieces in chunks.items():
            traces[f"{split}__{key}"] = np.concatenate(pieces)
        y_all = np.concatenate(ys)
        drained_all = np.concatenate(drained_preds)
        valid_all = np.concatenate(valid_preds)
        native["splits"][split] = evaluate_split(model, arrays, split, spec, p)
        traces[f"{split}__native_prediction"] = drained_all
        traces[f"{split}__valid_only_prediction"] = valid_all

        for user in np.unique(arrays[f"{split}_users"]):
            selected = arrays[f"{split}_users"] == user
            native["users"].append(
                {
                    "split": split,
                    "user": str(user),
                    "n": int(selected.sum()),
                    **metrics(y_all[selected], drained_all[selected]),
                }
            )

        lengths = arrays[f"{split}_lengths"]
        mask = np.arange(p.steps)[None, :] < lengths[:, None]
        duration_s = float(lengths.sum()) / p.fs
        for li in range(len(run.shifts)):
            z3 = traces[f"{split}__L{li+1}__spike"]
            flat = z3[mask]
            rates = z3.sum(axis=(0, 1)) / max(duration_s, 1e-12)
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
        native["splits"]["train"]["drained_ba"]
        - native["splits"]["test"]["drained_ba"]
    )
    return traces, native


def load_selected_model(
    root: Path,
    spec: ExtensionRun,
    p: Protocol,
    lock: dict[str, Any],
) -> BenchmarkNet:
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
    save_json(
        directory / "train_complete.json",
        {"status": "PASS", "run": spec.as_dict(), "core_identity": lock["identity"]},
    )
    return {
        "run": spec.key,
        "status": "PASS",
        "test_drained_ba": native["splits"]["test"]["drained_ba"],
        "test_valid_only_ba": native["splits"]["test"]["valid_only_ba"],
    }


def run_postprocess(root: Path, spec: ExtensionRun) -> dict[str, Any]:
    configure_cpu()
    p, lock = load_lock(root, repo_root())
    arrays = load_cache(root, p, lock)
    directory = root / "runs" / spec.key
    if not (directory / "train_complete.json").exists():
        raise FileNotFoundError(f"Train/eval incomplete: {spec.key}")
    if (directory / "analysis_complete.json").exists():
        return {"run": spec.key, "status": "already_complete"}
    with np.load(directory / "traces.npz", allow_pickle=False) as data:
        traces = {name: data[name] for name in data.files}
    run_probes(directory, traces, arrays, core_run(spec), p)
    save_json(
        directory / "analysis_complete.json",
        {"status": "PASS", "run": spec.as_dict(), "core_identity": lock["identity"]},
    )
    return {"run": spec.key, "status": "PASS"}


def finalize(root: Path) -> dict[str, Any]:
    import pandas as pd

    specs = available_runs(root)
    native_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    firing_rows: list[dict[str, Any]] = []
    for spec in specs:
        directory = root / "runs" / spec.key
        required = (
            "train_complete.json",
            "analysis_complete.json",
            "native.json",
            "probes.json",
        )
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{spec.key}: missing {missing}")
        native = json.loads((directory / "native.json").read_text())
        for split, values in native["splits"].items():
            native_rows.append(
                {"run_key": spec.key, **spec.as_dict(), "split": split, **values}
            )
        firing_rows.extend(
            {"run_key": spec.key, **spec.as_dict(), **row}
            for row in native["activity"]
        )
        probes = json.loads((directory / "probes.json").read_text())["rows"]
        probe_rows.extend({**spec.as_dict(), **row} for row in probes)

    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(native_rows).to_csv(aggregate / "native_runs.csv", index=False)
    pd.DataFrame(firing_rows).to_csv(aggregate / "firing_runs.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(aggregate / "probe_runs.csv", index=False)

    native_df = pd.DataFrame(native_rows)
    test = native_df[native_df["split"] == "test"]
    numeric = [
        column
        for column in (
            "drained_ba",
            "valid_only_ba",
            "drain_gain_ba",
            "drained_mean_logit_ce",
            "prediction_disagreement",
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
    summary = test.groupby(["beta"])[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(x) for x in column if x != "")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(aggregate / "native_summary.csv", index=False)
    manifest = {
        "status": "PASS",
        "extension_id": EXTENSION_ID,
        "version": EXTENSION_VERSION,
        "expected_runs": len(specs),
        "completed_runs": len(specs),
        "strict_reference": "Core R_LIF_E2E beta=0.5; not retrained",
        "tables": [
            "native_runs.csv",
            "native_summary.csv",
            "firing_runs.csv",
            "probe_runs.csv",
        ],
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
        default=repo_root() / "core_benchmark_v1/results/output_spike_drain_v1",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    prep = sub.add_parser("prepare")
    prep.add_argument("--synthetic", action="store_true")
    for name in ("run", "postprocess"):
        command = sub.add_parser(name)
        choose = command.add_mutually_exclusive_group(required=True)
        choose.add_argument("--run-key")
        choose.add_argument("--task-id", type=int)
    sub.add_parser("finalize")
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
            run_postprocess(root, spec)
        print(json.dumps(finalize(root), indent=2))
        return
    if args.command == "finalize":
        print(json.dumps(finalize(root), indent=2))
        return

    spec = resolve_spec(root, args.run_key, args.task_id)
    fn = {"run": run_formal, "postprocess": run_postprocess}[args.command]
    print(json.dumps(fn(root, spec), indent=2))


if __name__ == "__main__":
    main()
