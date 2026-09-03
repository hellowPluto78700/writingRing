from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_4_0_5_temporal_resolution_event_capacity as exp405
from scripts import experiment_4_0_6_local_mem_raw64 as exp406
from scripts import experiment_4_3_long_term_memory_validation as exp43
from scripts import experiment_4_4_1_stage2_memory_sweep as exp441


EXPERIMENT_ID = "experiment_4_4_3_multitau_memory"
PROTOCOL_VERSION = "stage2_multitau_dense_v1"
LOCAL_CONDITION = exp441.LOCAL_CONDITION
SEEDS = exp441.SEEDS
BATCH_SIZE = exp441.BATCH_SIZE
EPOCHS = exp441.EPOCHS
LR = exp441.LR
WEIGHT_DECAY = exp441.WEIGHT_DECAY
VARIANT = exp441.VARIANT
HIDDEN_CAP = exp441.HIDDEN_CAP
OUTPUT_CAP = exp441.OUTPUT_CAP
MULTITAU_PROFILES: dict[str, tuple[int, ...]] = {
    "tau250_500": (250, 500),
    "tau125_250_500": (125, 250, 500),
}
BASELINE_SINGLE_TAUS = (250, 500)


@dataclass(frozen=True)
class RunSpec:
    profile: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.profile}__dense__{LOCAL_CONDITION}__{VARIANT}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(profile, seed) for profile in MULTITAU_PROFILES for seed in SEEDS]


def paired_seed(spec: RunSpec, role: str) -> int:
    return exp441.paired_seed(
        exp441.RunSpec(model_kind="rsnn", tau_mem_ms=250, seed=spec.seed),
        role,
    )


def split_width(width: int, groups: int) -> tuple[int, ...]:
    base, remainder = divmod(width, groups)
    return tuple(base + (1 if index < remainder else 0) for index in range(groups))


def profile_definition(profile: str, dt_ms: float) -> dict[str, object]:
    taus = MULTITAU_PROFILES.get(profile)
    if taus is None:
        raise ValueError(f"Unknown multi-tau profile: {profile}")
    widths = split_width(exp405.STATE_WIDTH, len(taus))
    betas = tuple(exp441.stage2_beta(tau, dt_ms) for tau in taus)
    return {
        "profile": profile,
        "tau_mem_ms": taus,
        "group_widths": widths,
        "betas": betas,
    }


def profile_beta_vector(profile: str, dt_ms: float) -> torch.Tensor:
    definition = profile_definition(profile, dt_ms)
    pieces = [
        torch.full((int(width),), float(beta), dtype=torch.float32)
        for beta, width in zip(definition["betas"], definition["group_widths"], strict=True)
    ]
    vector = torch.cat(pieces)
    if vector.shape != (exp405.STATE_WIDTH,):
        raise ValueError(f"Expected {exp405.STATE_WIDTH} Stage-2 betas, got {vector.shape}")
    return vector


class HeterogeneousStage2LIF(nn.Module):
    """Stage-2 multi-event LIF with a fixed beta per neuron."""

    def __init__(self, beta_by_neuron: torch.Tensor) -> None:
        super().__init__()
        beta = torch.as_tensor(beta_by_neuron, dtype=torch.float32).reshape(-1)
        if beta.shape != (exp405.STATE_WIDTH,):
            raise ValueError(f"Expected {exp405.STATE_WIDTH} beta values, got {beta.shape}")
        if bool(((beta < 0.0) | (beta > 1.0)).any()):
            raise ValueError("beta values must be in [0, 1]")
        self.register_buffer("beta", beta)
        self.threshold = float(exp405.THRESHOLD)
        self.max_spikes_per_dt = int(HIDDEN_CAP)
        self.surrogate_slope = float(exp401.SURROGATE_SLOPE)

    def forward(
        self,
        current: torch.Tensor,
        membrane: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_reset = self.beta.to(dtype=membrane.dtype) * membrane + current
        spikes = exp401.multi_threshold_spike(
            pre_reset,
            self.threshold,
            self.max_spikes_per_dt,
            self.surrogate_slope,
        )
        membrane = pre_reset - spikes * self.threshold
        return spikes, membrane, pre_reset


class MultiTauStage2Decoder(exp441.Stage2SweepDecoder):
    """Dense RSNN with heterogeneous Stage-2 membrane time constants."""

    def __init__(
        self,
        profile: str,
        n_classes: int,
        fs: float = 64.0,
    ) -> None:
        # Build the exact paired Dense-RSNN weight stack first. state_lif has no
        # trainable parameters, so replacing it does not perturb shared weight initialization.
        super().__init__(
            model_kind="rsnn",
            tau_mem_ms=250,
            n_classes=n_classes,
            fs=fs,
        )
        self.profile = profile
        self.state_lif = HeterogeneousStage2LIF(profile_beta_vector(profile, self.dt_ms))


def prepare_data(repo_root: Path) -> exp405.TemporalData:
    return exp441.prepare_data(repo_root)


def make_loaders(
    data: exp405.TemporalData,
    spec: RunSpec,
    batch_size: int,
    train_shuffle: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    parts = {
        "train": (data.Xtr, data.ytr, data.vtr),
        "val": (data.Xva, data.yva, data.vva),
        "test": (data.Xte, data.yte, data.vte),
    }
    return {
        split: exp405.loader(
            X,
            y,
            valid,
            batch_size,
            train_shuffle if split == "train" else False,
            paired_seed(spec, f"{split}_loader"),
        )
        for split, (X, y, valid) in parts.items()
    }


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def train_one(
    spec: RunSpec,
    data: exp405.TemporalData,
    config: Config,
    force: bool,
) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp405.exp40.base.seed_all(paired_seed(spec, "model_init"))
    model = MultiTauStage2Decoder(
        profile=spec.profile,
        n_classes=len(data.labels),
        fs=data.fs,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loaders(data, spec, config.batch_size, train_shuffle=True)["train"]
    val_loader = make_loaders(data, spec, config.batch_size, train_shuffle=False)["val"]

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for X, y, valid_steps in train_loader:
            X = X.to(device)
            y = y.to(device)
            valid_steps = valid_steps.to(device)
            optimizer.zero_grad(set_to_none=True)
            traj = model.forward_trajectory(X)
            logits = exp405.normalized_output_evidence(
                traj["output_spikes"], valid_steps, model.output_cap
            )
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            n = len(y)
            train_n += n
            train_loss_sum += float(loss.item()) * n
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = exp405.exp40.metrics(
            np.concatenate(train_true), np.concatenate(train_pred)
        )
        val_metrics = exp405.evaluate_model(model, val_loader, device)
        val_ba = float(val_metrics["valid_count"]["balanced_accuracy"])
        val_loss = float(val_metrics["valid_count_loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_n, 1),
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_loss": val_loss,
                "val_balanced_accuracy": val_ba,
            }
        )
        improved = val_ba > best_val_ba + 1e-12 or (
            abs(val_ba - best_val_ba) <= 1e-12 and val_loss < best_val_loss
        )
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "profile_definition": profile_definition(spec.profile, data.dt_ms),
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_ba,
            "best_val_loss": best_val_loss,
            "model_state_dict": best_state,
            "channel_scale": data.channel_scale,
            "labels": data.labels,
            "split": data.split,
        },
        destination,
    )
    history_file = history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: exp405.TemporalData,
    config: Config,
) -> tuple[MultiTauStage2Decoder, dict[str, object]]:
    device = torch.device(config.device)
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp4.4.3 checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint identity mismatch for {spec.key}")
    model = MultiTauStage2Decoder(
        profile=spec.profile,
        n_classes=len(data.labels),
        fs=data.fs,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_one(
    spec: RunSpec,
    data: exp405.TemporalData,
    config: Config,
    force: bool,
) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model, checkpoint = load_model(spec, data, config)
    loaders = make_loaders(data, spec, config.batch_size, train_shuffle=False)
    features = {
        split: exp43.extract_condition_features(model, loader, device)
        for split, loader in loaders.items()
    }
    hidden_probe, uend_probe = exp43._fit_frozen_probes(features["train"])
    metrics = {
        split: exp43._condition_metrics(feature, hidden_probe, uend_probe)
        for split, feature in features.items()
    }
    for split, split_metrics in metrics.items():
        split_metrics["uend_minus_hidden_count_ba"] = float(
            split_metrics["uend_linear"]["balanced_accuracy"]
            - split_metrics["hidden_whole_count_linear"]["balanced_accuracy"]
        )
        split_metrics["uend_minus_output_count_ba"] = float(
            split_metrics["uend_linear"]["balanced_accuracy"]
            - split_metrics["output_whole_count"]["balanced_accuracy"]
        )

    diagnostics = {
        split: exp405.evaluate_model(model, loader, device)
        for split, loader in loaders.items()
    }
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "profile_definition": profile_definition(spec.profile, data.dt_ms),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_balanced_accuracy": float(checkpoint["best_val_balanced_accuracy"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "architecture": {
            "input": "Raw64 30-channel scaled weighted events",
            "local": exp406.local_condition_definition(LOCAL_CONDITION, data.fs),
            "stage2_width": exp405.STATE_WIDTH,
            "stage2_topology": "dense",
            "stage2_tau_profile": profile_definition(spec.profile, data.dt_ms),
            "output_tau_mem_ms": exp405.OUTPUT_TAU_MEM_MS,
            "variant": VARIANT,
            "hidden_cap": HIDDEN_CAP,
            "output_cap": OUTPUT_CAP,
            "threshold": exp405.THRESHOLD,
            "dt_ms": data.dt_ms,
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "probe_protocol": {
            "fit_source": "train-user representations only",
            "classifier": "train-only StandardScaler + balanced LogisticRegression(lbfgs)",
            "temporal_phase_access": False,
        },
        "provenance": {
            "split_seed": int(exp405.exp40.base.SPLIT_SEED),
            "training_objective": "valid normalized output WholeCount CE",
            "parameter_count_change_vs_single_tau": 0,
            "shared_initialization": (
                "same paired dense-RSNN weight initialization as Exp4.4.1; only fixed Stage-2 beta values differ"
            ),
        },
    }
    _save_json(destination, payload)
    return payload


def run_one(
    spec: RunSpec,
    data: exp405.TemporalData,
    config: Config,
    force: bool,
) -> dict[str, object]:
    train_one(spec, data, config, force)
    return evaluate_one(spec, data, config, force)


def _new_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in run_specs():
        payload = json.loads(evaluation_path(root, spec).read_text(encoding="utf-8"))
        definition = payload["profile_definition"]
        for split, metrics in payload["metrics"].items():
            diagnostics = payload["diagnostics"][split]
            rows.append(
                {
                    "profile": spec.profile,
                    "is_heterogeneous": True,
                    "tau_profile_ms": "/".join(str(value) for value in definition["tau_mem_ms"]),
                    "group_widths": "/".join(str(value) for value in definition["group_widths"]),
                    "seed": spec.seed,
                    "split": split,
                    "output_whole_count_ba": metrics["output_whole_count"]["balanced_accuracy"],
                    "hidden_whole_count_linear_ba": metrics["hidden_whole_count_linear"]["balanced_accuracy"],
                    "uend_linear_ba": metrics["uend_linear"]["balanced_accuracy"],
                    "uend_minus_hidden_count_ba": metrics["uend_minus_hidden_count_ba"],
                    "uend_minus_output_count_ba": metrics["uend_minus_output_count_ba"],
                    "state_events_per_neuron_second": diagnostics["state_valid_stats"]["mean_events_per_neuron_second"],
                    "output_events_per_neuron_second": diagnostics["output_valid_stats"]["mean_events_per_neuron_second"],
                    "state_tail_event_fraction": diagnostics["state_tail_event_fraction"],
                    "output_tail_event_fraction": diagnostics["output_tail_event_fraction"],
                    "artifact_source": EXPERIMENT_ID,
                }
            )
    return pd.DataFrame(rows)


def _baseline_rows(repo_root: Path) -> pd.DataFrame:
    source = exp441.results_dir(repo_root) / "runs.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing required Exp4.4.1 baseline artifact: {source}")
    frame = pd.read_csv(source)
    frame = frame[(frame["model_kind"] == "rsnn") & frame["tau_mem_ms"].isin(BASELINE_SINGLE_TAUS)].copy()
    frame["profile"] = frame["tau_mem_ms"].map(lambda value: f"single{int(value)}")
    frame["is_heterogeneous"] = False
    frame["tau_profile_ms"] = frame["tau_mem_ms"].astype(int).astype(str)
    frame["group_widths"] = str(exp405.STATE_WIDTH)
    frame["artifact_source"] = exp441.EXPERIMENT_ID
    keep = [
        "profile",
        "is_heterogeneous",
        "tau_profile_ms",
        "group_widths",
        "seed",
        "split",
        "output_whole_count_ba",
        "hidden_whole_count_linear_ba",
        "uend_linear_ba",
        "uend_minus_hidden_count_ba",
        "uend_minus_output_count_ba",
        "state_events_per_neuron_second",
        "output_events_per_neuron_second",
        "state_tail_event_fraction",
        "output_tail_event_fraction",
        "artifact_source",
    ]
    return frame[keep]


def _paired_effects(runs: pd.DataFrame) -> pd.DataFrame:
    test = runs[runs["split"] == "test"].set_index(["profile", "seed"])
    readouts = {
        "output_whole_count": "output_whole_count_ba",
        "hidden_whole_count_linear": "hidden_whole_count_linear_ba",
        "uend_linear": "uend_linear_ba",
    }
    rows: list[dict[str, object]] = []
    for profile in MULTITAU_PROFILES:
        for baseline in ("single250", "single500"):
            for seed in SEEDS:
                for readout, column in readouts.items():
                    rows.append(
                        {
                            "effect": f"{profile}_minus_{baseline}",
                            "profile": profile,
                            "baseline": baseline,
                            "seed": seed,
                            "readout": readout,
                            "delta_ba": float(
                                test.loc[(profile, seed), column]
                                - test.loc[(baseline, seed), column]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing required Exp4.4.3 artifact: {path}")

    runs = pd.concat([_baseline_rows(repo_root), _new_rows(root)], ignore_index=True)
    runs_file = root / "runs.csv"
    runs.to_csv(runs_file, index=False)

    test = runs[runs["split"] == "test"]
    metric_columns = [
        "output_whole_count_ba",
        "hidden_whole_count_linear_ba",
        "uend_linear_ba",
        "uend_minus_hidden_count_ba",
        "uend_minus_output_count_ba",
        "state_events_per_neuron_second",
        "output_events_per_neuron_second",
        "state_tail_event_fraction",
        "output_tail_event_fraction",
    ]
    summary = test.groupby(["profile", "is_heterogeneous", "tau_profile_ms", "group_widths"])[metric_columns].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary_file = root / "summary.csv"
    summary.to_csv(summary_file, index=False)

    effects = _paired_effects(runs)
    effects_file = root / "paired_effects.csv"
    effects.to_csv(effects_file, index=False)
    effects_summary = effects.groupby(["effect", "profile", "baseline", "readout"])["delta_ba"].agg(["mean", "std", "count"]).reset_index()
    effects_summary_file = root / "paired_effects_summary.csv"
    effects_summary.to_csv(effects_summary_file, index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Can heterogeneous Stage-2 membrane timescales improve the Dense-RSNN gesture-level state "
            "beyond the best homogeneous single-tau architecture without adding parameters?"
        ),
        "multitau_profiles": {key: list(value) for key, value in MULTITAU_PROFILES.items()},
        "baseline_single_taus": list(BASELINE_SINGLE_TAUS),
        "new_training_runs": len(run_specs()),
        "reused_exp441_baseline_runs": len(BASELINE_SINGLE_TAUS) * len(SEEDS),
        "seeds": list(SEEDS),
        "fixed": {
            "local_condition": LOCAL_CONDITION,
            "stage2_width": exp405.STATE_WIDTH,
            "stage2_topology": "dense",
            "variant": VARIANT,
            "threshold": exp405.THRESHOLD,
            "output_tau_mem_ms": exp405.OUTPUT_TAU_MEM_MS,
            "training_objective": "valid normalized output WholeCount CE",
        },
        "parameter_note": "Only fixed per-neuron beta/tau assignment changes; trainable parameter count is unchanged.",
        "primary_comparison": "each heterogeneous profile versus single250 paired by seed",
        "files": {
            "runs": runs_file.name,
            "summary": summary_file.name,
            "paired_effects": effects_file.name,
            "paired_effects_summary": effects_summary_file.name,
        },
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    return {
        "runs": runs_file,
        "summary": summary_file,
        "paired_effects": effects_file,
        "paired_effects_summary": effects_summary_file,
        "manifest": manifest_file,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-one")
    run_parser.add_argument("--array-task-id", type=int, required=True)
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--threads", type=int, default=1)
    run_parser.add_argument("--epochs", type=int, default=EPOCHS)
    run_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run_parser.add_argument("--force", action="store_true")

    eval_parser = subparsers.add_parser("eval-one")
    eval_parser.add_argument("--profile", choices=tuple(MULTITAU_PROFILES), required=True)
    eval_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--threads", type=int, default=1)
    eval_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    eval_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = exp405.exp40.find_repo_root()
    if args.command == "finalize":
        for name, path in finalize_experiment(repo_root).items():
            print(f"{name}: {path}")
        return

    data = prepare_data(repo_root)
    config = Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        epochs=getattr(args, "epochs", EPOCHS),
        batch_size=args.batch_size,
        threads=args.threads,
    )
    if args.command == "run-one":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(f"array task {args.array_task_id} outside [0, {len(specs) - 1}]")
        spec = specs[args.array_task_id]
        payload = run_one(spec, data, config, args.force)
    else:
        spec = RunSpec(args.profile, args.seed)
        payload = evaluate_one(spec, data, config, args.force)

    test = payload["metrics"]["test"]
    print(
        f"completed {spec.key}: "
        f"output_BA={test['output_whole_count']['balanced_accuracy']:.6f} "
        f"hidden_BA={test['hidden_whole_count_linear']['balanced_accuracy']:.6f} "
        f"uend_BA={test['uend_linear']['balanced_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
