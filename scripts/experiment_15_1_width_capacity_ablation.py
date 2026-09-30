from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from core_benchmark_v1.data import load_cache, loader
from core_benchmark_v1.model import BenchmarkNet, sequence_loss
from core_benchmark_v1.protocol import Protocol, Run, SPLITS, digest
from core_benchmark_v1.storage import file_hash, save_json, save_npz, save_torch
from core_benchmark_v1.training import cpu_state
from scripts import experiment_15_context_dependent_write_gate as exp15


EXPERIMENT_ID = "experiment_15_1_width_capacity_ablation"
PROTOCOL_VERSION = "width_capacity_v1"
CORE_RESULTS_REL = Path("core_benchmark_v1/results/main")
EXP15_RESULTS_REL = Path(
    "notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1"
)
WIDTHS = (64, 96, 128)
TRAIN_WIDTHS = (64, 96)
SEEDS = (11, 23, 37)
SHIFTS = ((2, 3, 4), (2, 3, 4))
CASES = ("B0", "C0", "GF", "GJ")
CONTINUATION_CASES = ("C0", "GF", "GJ")
GATED_CASES = ("GF", "GJ")
Q_INIT = exp15.Q_INIT
INTERVENTIONS = exp15.INTERVENTIONS
A3_SHUFFLE_SEEDS = exp15.A3_SHUFFLE_SEEDS


@dataclass(frozen=True)
class WidthSpec:
    width: int
    case: str
    seed: int

    @property
    def key(self) -> str:
        return f"H{self.width}__{self.case}__seed{self.seed}"

    @property
    def run_key(self) -> str:
        return f"{self.case}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    core_results_dir: Path
    exp15_results_dir: Path


@dataclass(frozen=True)
class WidthProtocol(Protocol):
    """Production CoreBenchmark protocol with width as the only unlocked geometry axis."""

    def validate(self) -> None:
        # Reuse all generic Protocol validation by validating an otherwise-identical
        # base Protocol under the relaxed smoke geometry gate. This does not imply
        # synthetic data: Exp15.1 always consumes the frozen production dataset.
        relaxed = replace(Protocol(**asdict(self)), profile="smoke")
        relaxed.validate()

        if self.profile != "production":
            raise ValueError("Exp15.1 width protocols must derive from production CoreBenchmark")
        if self.seeds != SEEDS:
            raise ValueError("Exp15.1 seeds are locked")
        if self.width not in WIDTHS:
            raise ValueError(f"Exp15.1 width must be one of {WIDTHS}")
        if (
            self.fs != 64.0
            or self.input_channels != 30
            or self.total_channels != 36
            or self.steps != 256
        ):
            raise ValueError("Exp15.1 unlocks width only; all other production geometry is locked")


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
        core_results_dir=(args.core_results or root / CORE_RESULTS_REL).resolve(),
        exp15_results_dir=(args.exp15_results or root / EXP15_RESULTS_REL).resolve(),
    )


def baseline_specs() -> list[WidthSpec]:
    return [WidthSpec(width, "B0", seed) for width in TRAIN_WIDTHS for seed in SEEDS]


def phase1_specs() -> list[WidthSpec]:
    return [
        WidthSpec(width, case, seed)
        for width in TRAIN_WIDTHS
        for case in CONTINUATION_CASES
        for seed in SEEDS
    ]


def phase1_5_specs() -> list[WidthSpec]:
    return [
        WidthSpec(width, case, seed)
        for width in TRAIN_WIDTHS
        for case in GATED_CASES
        for seed in SEEDS
    ]


def _run(spec: WidthSpec) -> Run:
    return Run(
        spec.case,
        spec.seed,
        "15_1_width_capacity",
        shifts=SHIFTS,
        objective="wcce",
    )


def _load_core(
    config: Config,
) -> tuple[Protocol, dict[str, Any], dict[str, np.ndarray]]:
    lock = json.loads(
        (config.core_results_dir / "protocol.lock.json").read_text(encoding="utf-8")
    )
    payload = dict(lock["protocol"])
    payload["version"] = Protocol().version
    core = Protocol.from_dict(payload)
    if tuple(core.seeds) != SEEDS or core.width != 128:
        raise ValueError("Exp15.1 requires the locked CoreBenchmark 128-wide production protocol")
    if file_hash(config.core_results_dir / "dataset.npz") != lock["dataset_hash"]:
        raise ValueError("CoreBenchmark dataset hash changed")
    arrays = load_cache(config.core_results_dir, core, lock)
    return core, lock, arrays


def make_width_protocol(core: Protocol, width: int) -> WidthProtocol:
    if width not in WIDTHS:
        raise ValueError(width)
    payload = asdict(core)
    payload["width"] = width
    protocol = WidthProtocol(**payload)
    protocol.validate()
    return protocol


def _require_exp15_reference(config: Config) -> dict[str, Any]:
    manifest_path = config.exp15_results_dir / "aggregate" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Exp15 reference artifacts are required for H128 reuse: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("phase1_runs") != 9:
        raise ValueError("Exp15 reference manifest is not a completed PASS artifact")
    required = (
        "phase1_native_runs.csv",
        "phase1_native_per_user.csv",
        "phase1_probe_runs.csv",
        "phase1_gate_summary.csv",
        "phase1_gate_phase10.csv",
        "phase1_5_native_runs.csv",
        "phase1_5_probe_runs.csv",
    )
    missing = [
        name
        for name in required
        if not (config.exp15_results_dir / "aggregate" / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Exp15 reference aggregate is incomplete: {missing}")
    return manifest


def prepare(config: Config) -> dict[str, Any]:
    core, lock, _ = _load_core(config)
    exp15_manifest = _require_exp15_reference(config)
    if exp15_manifest["core_identity"] != lock["identity"]:
        raise ValueError(
            "Exp15 H128 reference and current CoreBenchmark do not share the same core identity"
        )
    for width in WIDTHS:
        make_width_protocol(core, width)

    config.results_dir.mkdir(parents=True, exist_ok=True)
    parameter_rows = parameter_counts()
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "scientific_question": (
            "Does reducing both hidden SNN layers from 128 neurons to 96 or 64 "
            "reduce over-capacity and make the unchanged Exp15 gate more useful?"
        ),
        "widths": list(WIDTHS),
        "trained_widths": list(TRAIN_WIDTHS),
        "reused_width": 128,
        "seeds": list(SEEDS),
        "cases": list(CASES),
        "continuation_cases": list(CONTINUATION_CASES),
        "q_init": Q_INIT,
        "core_identity": lock["identity"],
        "core_dataset_hash": lock["dataset_hash"],
        "exp15_reference_manifest_hash": file_hash(
            config.exp15_results_dir / "aggregate" / "manifest.json"
        ),
        "exp15_reference_core_identity": exp15_manifest["core_identity"],
        "baseline_tasks": len(baseline_specs()),
        "phase1_tasks": len(phase1_specs()),
        "phase1_5_tasks": len(phase1_5_specs()),
        "interventions": list(INTERVENTIONS),
        "parameter_counts": parameter_rows,
        "preserved_contract": (
            "Exp15 architecture, WCCE, gate equation/init, optimizer, checkpoint "
            "selection, split, tau, readout, probes and interventions are unchanged; "
            "only L1/L2 hidden width varies."
        ),
    }
    payload["identity"] = digest(payload)
    save_json(config.results_dir / "protocol.json", payload)

    source_paths = [
        Path(__file__).resolve(),
        config.repo_root / "scripts" / "experiment_15_context_dependent_write_gate.py",
        config.repo_root / "core_benchmark_v1" / "model.py",
        config.repo_root / "core_benchmark_v1" / "probes.py",
        config.repo_root / "core_benchmark_v1" / "data.py",
        config.repo_root / "core_benchmark_v1" / "training.py",
    ]
    save_json(
        config.results_dir / "source_manifest.json",
        {
            "files": {
                str(path.relative_to(config.repo_root)): file_hash(path)
                for path in source_paths
            }
        },
    )
    exp15_aggregate = config.exp15_results_dir / "aggregate"
    exp15_reference_files = {
        name: file_hash(exp15_aggregate / name)
        for name in (
            "manifest.json",
            "phase1_native_runs.csv",
            "phase1_native_per_user.csv",
            "phase1_probe_runs.csv",
            "phase1_gate_summary.csv",
            "phase1_gate_phase10.csv",
            "phase1_5_native_runs.csv",
            "phase1_5_probe_runs.csv",
        )
    }
    save_json(
        config.results_dir / "reference_manifest.json",
        {
            "core_identity": lock["identity"],
            "core_dataset_hash": lock["dataset_hash"],
            "exp15_manifest": exp15_manifest,
            "exp15_reference_files": exp15_reference_files,
        },
    )
    return payload


def parameter_counts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference = 128 * 128 + 42 * 128
    for width in WIDTHS:
        baseline = width * width + 42 * width
        gate = 2 * width + 1
        rows.append(
            {
                "width": width,
                "baseline_parameters": baseline,
                "gate_parameters": gate,
                "gated_total_parameters": baseline + gate,
                "baseline_fraction_of_h128": baseline / reference,
            }
        )
    return rows


def _run_dir(config: Config, spec: WidthSpec) -> Path:
    return config.results_dir / "runs" / f"H{spec.width}" / spec.run_key


def _baseline_checkpoint(config: Config, width: int, seed: int) -> Path:
    return _run_dir(config, WidthSpec(width, "B0", seed)) / "checkpoint.pt"


def _load_width_baseline_state(
    config: Config, width: int, seed: int
) -> dict[str, torch.Tensor]:
    path = _baseline_checkpoint(config, width, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("experiment") != EXPERIMENT_ID
        or payload.get("protocol") != PROTOCOL_VERSION
        or payload.get("width") != width
        or payload.get("case") != "B0"
        or payload.get("seed") != seed
    ):
        raise ValueError(f"Width baseline checkpoint identity mismatch: {path}")
    return payload["model_state_dict"]


def _build_model(
    config: Config,
    spec: WidthSpec,
    p: WidthProtocol,
) -> nn.Module:
    run = _run(spec)
    if spec.case == "B0":
        return BenchmarkNet(run, p)

    state = _load_width_baseline_state(config, spec.width, spec.seed)
    if spec.case == "C0":
        model = BenchmarkNet(run, p)
        model.load_state_dict(state, strict=True)
        return model

    if spec.case not in GATED_CASES:
        raise ValueError(spec.case)
    model = exp15.ContextWriteGateNet(run, p)
    model.load_baseline_state(state)
    if spec.case == "GF":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("gate_"))
    return model


def _evaluate_spec(
    config: Config,
    spec: WidthSpec,
    p: WidthProtocol,
    arrays: dict[str, np.ndarray],
    model: nn.Module,
) -> dict[str, Any]:
    directory = _run_dir(config, spec)
    traces, native, gates = exp15._extract(model, arrays, p, spec.seed)
    native["width"] = spec.width
    native["case"] = spec.case
    native["seed"] = spec.seed
    save_json(directory / "native.json", native)
    save_npz(directory / "traces.npz", traces)
    exp15._run_l2_probes(
        directory,
        traces,
        arrays,
        exp15.ExpSpec(spec.case, spec.seed),
        p,
    )
    if gates:
        gate_summary = exp15._gate_summary(gates, arrays)
        save_json(directory / "gate_summary.json", gate_summary)
    return {"status": "PASS", "run": spec.key}


def train_one(config: Config, spec: WidthSpec) -> dict[str, Any]:
    if spec.width not in TRAIN_WIDTHS:
        raise ValueError("H128 is reference-only in Exp15.1")

    core, _, arrays = _load_core(config)
    p = make_width_protocol(core, spec.width)
    directory = _run_dir(config, spec)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    if checkpoint_path.exists():
        return {"status": "exists", "run": spec.key}

    model = _build_model(config, spec, p)
    initial = cpu_state(model)
    optimizer = exp15._make_optimizer(model, p)
    train_loader = loader(arrays, "train", p, spec.seed, shuffle=True)

    best = exp15._split_eval(model, arrays, p, spec.seed, "val")
    best_state = initial
    best_epoch = 0
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train_loss": None,
            "val_ba": best["ba"],
            "val_mean_logit_ce": best["mean_logit_ce"],
        }
    ]
    gradient_rows: list[dict[str, Any]] = []
    if isinstance(model, exp15.ContextWriteGateNet):
        gradient_rows.append(
            exp15._gradient_diagnostic(model, arrays, p, spec.seed, 0)
        )

    stopped_epoch = 0
    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for x, y, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(model(x, lengths)["evidence"], lengths, y, "wcce")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite loss")
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"{spec.key}: nonfinite gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            count += len(y)

        val = exp15._split_eval(model, arrays, p, spec.seed, "val")
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / count,
                "val_ba": val["ba"],
                "val_mean_logit_ce": val["mean_logit_ce"],
            }
        )
        improved = val["ba"] > best["ba"] + 1e-12 or (
            abs(val["ba"] - best["ba"]) <= 1e-12
            and val["mean_logit_ce"] < best["mean_logit_ce"] - 1e-12
        )
        if improved:
            best = val
            best_state = cpu_state(model)
            best_epoch = epoch

        if (
            isinstance(model, exp15.ContextWriteGateNet)
            and epoch in exp15.GRADIENT_EPOCHS
        ):
            gradient_rows.append(
                exp15._gradient_diagnostic(model, arrays, p, spec.seed, epoch)
            )

        stopped_epoch = epoch
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    model.load_state_dict(best_state)
    if isinstance(model, exp15.ContextWriteGateNet):
        selected_gradient = exp15._gradient_diagnostic(
            model, arrays, p, spec.seed, best_epoch
        )
        selected_gradient["selected_checkpoint"] = True
        gradient_rows.append(selected_gradient)

    if spec.case == "GF":
        baseline_keys = [key for key in initial if not key.startswith("gate_")]
        if any(
            not torch.equal(initial[key], best_state[key])
            for key in baseline_keys
        ):
            raise AssertionError("GF frozen backbone changed")

    save_json(directory / "train_history.json", {"rows": history})
    save_json(
        directory / "gate_gradient_diagnostics.json",
        {"rows": gradient_rows},
    )
    save_torch(
        checkpoint_path,
        {
            "experiment": EXPERIMENT_ID,
            "protocol": PROTOCOL_VERSION,
            "width": spec.width,
            "case": spec.case,
            "seed": spec.seed,
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val": best,
            "parent_baseline_hash": (
                None
                if spec.case == "B0"
                else file_hash(
                    _baseline_checkpoint(config, spec.width, spec.seed)
                )
            ),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "selection_rule": (
                "native validation BA, then validation mean-logit CE, epoch0 included"
            ),
        },
    )
    _evaluate_spec(config, spec, p, arrays, model)
    return {
        "status": "PASS",
        "run": spec.key,
        "best_epoch": best_epoch,
        "best_val_ba": best["ba"],
    }


def _load_selected_model(
    config: Config,
    spec: WidthSpec,
    p: WidthProtocol,
) -> nn.Module:
    payload = torch.load(
        _run_dir(config, spec) / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("width") != spec.width or payload.get("case") != spec.case:
        raise ValueError(f"Checkpoint identity mismatch: {spec.key}")
    model = _build_model(config, spec, p)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def run_phase1_5(config: Config, spec: WidthSpec) -> dict[str, Any]:
    if spec.width not in TRAIN_WIDTHS or spec.case not in GATED_CASES:
        raise ValueError(spec)

    core, _, arrays = _load_core(config)
    p = make_width_protocol(core, spec.width)
    model = _load_selected_model(config, spec, p)

    root = config.results_dir / "phase1_5" / f"H{spec.width}" / spec.run_key
    root.mkdir(parents=True, exist_ok=True)
    for intervention in INTERVENTIONS:
        shuffle_seeds = (
            A3_SHUFFLE_SEEDS if intervention == "A3" else (-1,)
        )
        for shuffle_seed in shuffle_seeds:
            tag = (
                intervention
                if shuffle_seed < 0
                else f"{intervention}__shuffle{shuffle_seed}"
            )
            directory = root / tag
            directory.mkdir(parents=True, exist_ok=True)
            traces, native = exp15._extract_intervention(
                model,
                arrays,
                p,
                spec.seed,
                intervention,
                shuffle_seed,
            )
            native["width"] = spec.width
            native["case"] = spec.case
            native["seed"] = spec.seed
            save_json(directory / "native.json", native)
            save_npz(directory / "traces.npz", traces)
            exp15._run_l2_probes(
                directory,
                traces,
                arrays,
                exp15.ExpSpec(spec.case, spec.seed),
                p,
            )
    return {
        "status": "PASS",
        "run": spec.key,
        "interventions": list(INTERVENTIONS),
    }


def _native_row(
    width: int, case: str, seed: int, native: dict[str, Any]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "width": width,
        "case": case,
        "seed": seed,
        "train_test_gap": native["train_test_gap"],
    }
    for split in SPLITS:
        for name, value in native["splits"][split].items():
            row[f"{split}_{name}"] = value
    return row


def _add_width(frame: pd.DataFrame, width: int) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "width", width)
    return result


def _read_exp15(config: Config, name: str) -> pd.DataFrame:
    return pd.read_csv(config.exp15_results_dir / "aggregate" / name)


def finalize(config: Config) -> dict[str, Any]:
    exp15_manifest = _require_exp15_reference(config)
    core, lock, _ = _load_core(config)
    if exp15_manifest["core_identity"] != lock["identity"]:
        raise ValueError(
            "Exp15 H128 reference and current CoreBenchmark do not share the same core identity"
        )
    make_width_protocol(core, 128)
    reference_lock = json.loads(
        (config.results_dir / "reference_manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in reference_lock["exp15_reference_files"].items():
        actual = file_hash(config.exp15_results_dir / "aggregate" / name)
        if actual != expected:
            raise ValueError(f"Exp15 reference artifact changed after prepare: {name}")
    aggregate = config.results_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)

    native_rows: list[dict[str, Any]] = []
    per_user_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    gate_phase_rows: list[dict[str, Any]] = []

    for width in TRAIN_WIDTHS:
        for case in CASES:
            for seed in SEEDS:
                spec = WidthSpec(width, case, seed)
                directory = _run_dir(config, spec)
                if not (directory / "native.json").exists():
                    raise FileNotFoundError(directory / "native.json")
                native = json.loads(
                    (directory / "native.json").read_text(encoding="utf-8")
                )
                native_rows.append(_native_row(width, case, seed, native))
                per_user_rows.extend(
                    {
                        "width": width,
                        "case": case,
                        "seed": seed,
                        **item,
                    }
                    for item in native["users"]
                )
                probe_rows.extend(
                    {
                        "width": width,
                        **item,
                    }
                    for item in json.loads(
                        (directory / "probes.json").read_text(encoding="utf-8")
                    )["rows"]
                )
                if (directory / "gate_summary.json").exists():
                    gate = json.loads(
                        (directory / "gate_summary.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    gate_rows.extend(
                        {
                            "width": width,
                            "case": case,
                            "seed": seed,
                            **item,
                        }
                        for item in gate["summary"]
                    )
                    gate_phase_rows.extend(
                        {
                            "width": width,
                            "case": case,
                            "seed": seed,
                            **item,
                        }
                        for item in gate["phase10"]
                    )

    # H128 is immutable Exp15 output and is never retrained in Exp15.1.
    native_128 = _add_width(_read_exp15(config, "phase1_native_runs.csv"), 128)
    native_frame = pd.concat(
        [pd.DataFrame(native_rows), native_128],
        ignore_index=True,
    )
    native_frame.to_csv(aggregate / "phase1_native_runs.csv", index=False)

    per_user_128 = _add_width(
        _read_exp15(config, "phase1_native_per_user.csv"), 128
    )
    per_user_frame = pd.concat(
        [pd.DataFrame(per_user_rows), per_user_128],
        ignore_index=True,
    )
    per_user_frame.to_csv(
        aggregate / "phase1_native_per_user.csv", index=False
    )

    probe_128 = _add_width(
        _read_exp15(config, "phase1_probe_runs.csv"), 128
    )
    probe_frame = pd.concat(
        [pd.DataFrame(probe_rows), probe_128],
        ignore_index=True,
    )
    probe_frame.to_csv(
        aggregate / "phase1_probe_runs.csv", index=False
    )
    probe_seed_means = probe_frame.groupby(
        [
            "width",
            "case",
            "seed",
            "layer",
            "state",
            "aggregation",
            "decoder",
        ],
        as_index=False,
        dropna=False,
    )[
        [
            column
            for column in probe_frame.columns
            if column.endswith(
                ("_ba", "_accuracy", "_macro_f1", "_gap")
            )
        ]
    ].mean()
    probe_seed_means.to_csv(
        aggregate / "phase1_probe_seed_means.csv", index=False
    )

    gain_rows: list[pd.DataFrame] = []
    for width in WIDTHS:
        current = probe_seed_means[probe_seed_means.width == width].drop(
            columns="width"
        )
        gains = exp15._probe_gains(current)
        gains.insert(0, "width", width)
        gain_rows.append(gains)
    pd.concat(gain_rows, ignore_index=True).to_csv(
        aggregate / "phase1_probe_gains.csv", index=False
    )

    gate_frame = pd.concat(
        [
            pd.DataFrame(gate_rows),
            _add_width(_read_exp15(config, "phase1_gate_summary.csv"), 128),
        ],
        ignore_index=True,
    )
    gate_frame.to_csv(aggregate / "phase1_gate_summary.csv", index=False)

    gate_phase_frame = pd.concat(
        [
            pd.DataFrame(gate_phase_rows),
            _add_width(_read_exp15(config, "phase1_gate_phase10.csv"), 128),
        ],
        ignore_index=True,
    )
    gate_phase_frame.to_csv(
        aggregate / "phase1_gate_phase10.csv", index=False
    )

    paired_contrasts: list[dict[str, Any]] = []
    for width in WIDTHS:
        for seed in SEEDS:
            current = native_frame[
                (native_frame.width == width) & (native_frame.seed == seed)
            ].set_index("case")
            for contrast, left, right in (
                ("GF_minus_B0", "GF", "B0"),
                ("GJ_minus_C0", "GJ", "C0"),
                ("GF_minus_C0", "GF", "C0"),
            ):
                paired_contrasts.append(
                    {
                        "width": width,
                        "seed": seed,
                        "contrast": contrast,
                        **{
                            f"{split}_delta": float(
                                current.loc[left, f"{split}_ba"]
                                - current.loc[right, f"{split}_ba"]
                            )
                            for split in SPLITS
                        },
                    }
                )
    pd.DataFrame(paired_contrasts).to_csv(
        aggregate / "phase1_paired_contrasts.csv", index=False
    )

    native_summary = native_frame.groupby(
        ["width", "case"], as_index=False
    ).agg(
        train_ba_mean=("train_ba", "mean"),
        train_ba_std=("train_ba", "std"),
        val_ba_mean=("val_ba", "mean"),
        val_ba_std=("val_ba", "std"),
        test_ba_mean=("test_ba", "mean"),
        test_ba_std=("test_ba", "std"),
        train_test_gap_mean=("train_test_gap", "mean"),
        train_test_gap_std=("train_test_gap", "std"),
    )
    native_summary.to_csv(
        aggregate / "width_native_summary.csv", index=False
    )

    probe_width_summary = probe_seed_means.groupby(
        ["width", "case", "aggregation", "decoder"],
        as_index=False,
    ).agg(
        train_ba_mean=("train_ba", "mean"),
        val_ba_mean=("val_ba", "mean"),
        test_ba_mean=("test_ba", "mean"),
        test_ba_std=("test_ba", "std"),
    )
    probe_width_summary.to_csv(
        aggregate / "width_probe_summary.csv", index=False
    )

    pd.DataFrame(parameter_counts()).to_csv(
        aggregate / "parameter_counts.csv", index=False
    )

    intervention_native_rows: list[dict[str, Any]] = []
    intervention_probe_rows: list[dict[str, Any]] = []
    for width in TRAIN_WIDTHS:
        for case in GATED_CASES:
            for seed in SEEDS:
                spec = WidthSpec(width, case, seed)
                root = (
                    config.results_dir
                    / "phase1_5"
                    / f"H{width}"
                    / spec.run_key
                )
                for intervention in INTERVENTIONS:
                    tags = (
                        [intervention]
                        if intervention != "A3"
                        else [
                            f"A3__shuffle{shuffle_seed}"
                            for shuffle_seed in A3_SHUFFLE_SEEDS
                        ]
                    )
                    for tag in tags:
                        directory = root / tag
                        native = json.loads(
                            (directory / "native.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        row: dict[str, Any] = {
                            "width": width,
                            "case": case,
                            "seed": seed,
                            "intervention": intervention,
                            "replicate": tag,
                            "train_test_gap": native["train_test_gap"],
                        }
                        for split in SPLITS:
                            for name, value in native["splits"][
                                split
                            ].items():
                                row[f"{split}_{name}"] = value
                        intervention_native_rows.append(row)
                        intervention_probe_rows.extend(
                            {
                                "width": width,
                                "parent_case": case,
                                "intervention": intervention,
                                "replicate": tag,
                                **probe_row,
                            }
                            for probe_row in json.loads(
                                (directory / "probes.json").read_text(
                                    encoding="utf-8"
                                )
                            )["rows"]
                        )

    intervention_native = pd.concat(
        [
            pd.DataFrame(intervention_native_rows),
            _add_width(
                _read_exp15(config, "phase1_5_native_runs.csv"),
                128,
            ),
        ],
        ignore_index=True,
    )
    intervention_native.to_csv(
        aggregate / "phase1_5_native_runs.csv", index=False
    )
    intervention_seed_means = intervention_native.groupby(
        ["width", "case", "seed", "intervention"],
        as_index=False,
    ).mean(numeric_only=True)
    intervention_seed_means.to_csv(
        aggregate / "phase1_5_native_seed_means.csv", index=False
    )

    intervention_deltas: list[dict[str, Any]] = []
    for (width, case, seed), group in intervention_seed_means.groupby(
        ["width", "case", "seed"],
        sort=False,
    ):
        lookup = group.set_index("intervention")
        if "A0" not in lookup.index:
            continue
        for intervention in ("A1", "A2", "A3", "A4", "A4b"):
            if intervention not in lookup.index:
                continue
            intervention_deltas.append(
                {
                    "width": width,
                    "case": case,
                    "seed": seed,
                    "contrast": f"A0_minus_{intervention}",
                    **{
                        f"{split}_delta": float(
                            lookup.loc["A0", f"{split}_ba"]
                            - lookup.loc[
                                intervention, f"{split}_ba"
                            ]
                        )
                        for split in SPLITS
                    },
                }
            )
    pd.DataFrame(intervention_deltas).to_csv(
        aggregate / "phase1_5_native_deltas.csv", index=False
    )

    intervention_probe = pd.concat(
        [
            pd.DataFrame(intervention_probe_rows),
            _add_width(
                _read_exp15(config, "phase1_5_probe_runs.csv"),
                128,
            ),
        ],
        ignore_index=True,
    )
    intervention_probe.to_csv(
        aggregate / "phase1_5_probe_runs.csv", index=False
    )
    intervention_probe_seed_means = intervention_probe.groupby(
        [
            "width",
            "parent_case",
            "seed",
            "intervention",
            "layer",
            "state",
            "aggregation",
            "decoder",
        ],
        as_index=False,
        dropna=False,
    )[
        [
            column
            for column in intervention_probe.columns
            if column.endswith(
                ("_ba", "_accuracy", "_macro_f1", "_gap")
            )
        ]
    ].mean()
    intervention_probe_seed_means.to_csv(
        aggregate / "phase1_5_probe_seed_means.csv", index=False
    )

    report = {
        "status": "PASS",
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL_VERSION,
        "widths": list(WIDTHS),
        "trained_widths": list(TRAIN_WIDTHS),
        "reused_width": 128,
        "baseline_tasks": len(baseline_specs()),
        "phase1_tasks": len(phase1_specs()),
        "phase1_5_tasks": len(phase1_5_specs()),
        "exp15_reference_manifest_hash": file_hash(
            config.exp15_results_dir / "aggregate" / "manifest.json"
        ),
    }
    save_json(aggregate / "manifest.json", report)
    return report


def configure_cpu() -> None:
    exp15.configure_cpu()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    parser.add_argument("--exp15-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("plan")
    sub.add_parser("finalize")

    baseline = sub.add_parser("train-baseline")
    baseline.add_argument("--task-id", type=int, required=True)
    phase1 = sub.add_parser("train-phase1")
    phase1.add_argument("--task-id", type=int, required=True)
    phase15 = sub.add_parser("phase1-5")
    phase15.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args(argv)

    configure_cpu()
    config = config_from_args(args)
    if args.command == "prepare":
        print(json.dumps(prepare(config), indent=2))
    elif args.command == "plan":
        print(
            json.dumps(
                {
                    "baseline": [
                        asdict(spec) | {"key": spec.key}
                        for spec in baseline_specs()
                    ],
                    "phase1": [
                        asdict(spec) | {"key": spec.key}
                        for spec in phase1_specs()
                    ],
                    "phase1_5": [
                        asdict(spec) | {"key": spec.key}
                        for spec in phase1_5_specs()
                    ],
                    "reused": [
                        {
                            "width": 128,
                            "source": "Exp15",
                            "cases": list(CASES),
                            "seeds": list(SEEDS),
                        }
                    ],
                },
                indent=2,
            )
        )
    elif args.command == "train-baseline":
        specs = baseline_specs()
        if not 0 <= args.task_id < len(specs):
            parser.error(f"task-id must be in [0, {len(specs) - 1}]")
        print(json.dumps(train_one(config, specs[args.task_id]), indent=2))
    elif args.command == "train-phase1":
        specs = phase1_specs()
        if not 0 <= args.task_id < len(specs):
            parser.error(f"task-id must be in [0, {len(specs) - 1}]")
        print(json.dumps(train_one(config, specs[args.task_id]), indent=2))
    elif args.command == "phase1-5":
        specs = phase1_5_specs()
        if not 0 <= args.task_id < len(specs):
            parser.error(f"task-id must be in [0, {len(specs) - 1}]")
        print(
            json.dumps(
                run_phase1_5(config, specs[args.task_id]),
                indent=2,
            )
        )
    elif args.command == "finalize":
        print(json.dumps(finalize(config), indent=2))


if __name__ == "__main__":
    main()
