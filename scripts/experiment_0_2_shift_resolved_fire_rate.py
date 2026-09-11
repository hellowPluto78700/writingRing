from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import experiment_0_2_endpoint_tail_regularization as exp02


ANALYSIS_ID = "experiment_0_2_shift_resolved_fire_rate"
ANALYSIS_VERSION = "shift_fire_rate_v1"
PROFILES = (exp02.FROZEN_PROFILE,) + exp02.PROFILES
# Exp0.1/0.2 naming treats s2-s5 (~54-492 ms at 64 Hz) as short/mid and
# introduces s6/s7 (~1-2 s) in the long-timescale final hidden layer.
LONG_SHIFT_MIN = 6

RATE_COLUMNS = (
    "valid_firing_rate_hz",
    "tail_stage1_firing_rate_hz",
    "tail_stage2_firing_rate_hz",
    "tail_stage3_firing_rate_hz",
    "tail_firing_rate_hz",
    "tail_to_valid_rate_ratio",
)


@dataclass(frozen=True)
class EvalSpec:
    architecture: str
    objective: str
    profile: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.architecture}__{self.objective}__{self.profile}__seed{self.seed}"

    def exp02_spec(self) -> exp02.RunSpec:
        return exp02.RunSpec(self.architecture, self.objective, self.profile, self.seed)


def eval_specs() -> list[EvalSpec]:
    return [
        EvalSpec(architecture, objective, profile, seed)
        for architecture in exp02.DIRECT_ARCHITECTURES
        for objective in exp02.OBJECTIVES
        for profile in PROFILES
        for seed in exp02.SEEDS
    ]


def artifact_root(repo_root: Path) -> Path:
    return exp02.results_dir(repo_root)


def analysis_root(repo_root: Path) -> Path:
    return artifact_root(repo_root) / ANALYSIS_ID / ANALYSIS_VERSION


def per_checkpoint_path(repo_root: Path, spec: EvalSpec) -> Path:
    return analysis_root(repo_root) / "per_checkpoint" / f"{spec.key}.csv"


def _shift_slices(shifts: tuple[int, ...], width: int) -> list[tuple[int, int, int]]:
    """Use the exact final-hidden shift allocation already used by Exp0.2 diagnostics."""
    return list(exp02._shift_slices(width, shifts))


def _spike_stats(
    spikes: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    stop: int,
    fs: float,
) -> tuple[float, float, int, int]:
    """Return Hz/neuron, spikes/neuron, masked sample-steps and neuron count."""
    if spikes.ndim != 3 or mask.ndim != 2 or spikes.shape[:2] != mask.shape:
        raise ValueError("Expected spikes [B,T,N] and mask [B,T]")
    if not (0 <= start < stop <= spikes.shape[2]):
        raise ValueError(f"Invalid neuron slice [{start}:{stop}) for width {spikes.shape[2]}")
    selected = spikes[:, :, start:stop]
    mask_f = mask.to(selected.dtype).unsqueeze(-1)
    spike_count = float((selected * mask_f).sum().item())
    timestep_count = int(mask.sum().item())
    neuron_count = int(stop - start)
    neuron_steps = timestep_count * neuron_count
    neuron_seconds = neuron_steps / float(fs)
    firing_rate_hz = spike_count / neuron_seconds if neuron_seconds > 0 else float("nan")
    spikes_per_neuron = spike_count / neuron_count if neuron_count > 0 else float("nan")
    return firing_rate_hz, spikes_per_neuron, timestep_count, neuron_count


def _load_checkpoint_model(
    repo_root: Path,
    data: object,
    spec: EvalSpec,
    device_name: str,
    batch_size: int,
    threads: int,
) -> exp02.DiagnosticMultiTauHierarchySNN:
    """Reuse Exp0.2 identity checks for both new and frozen Exp0.1 checkpoints."""
    run_spec = spec.exp02_spec()
    config = exp02.Config(
        repo_root=repo_root,
        results_dir=artifact_root(repo_root),
        device=device_name,
        batch_size=batch_size,
        threads=threads,
    )
    if spec.profile == exp02.FROZEN_PROFILE:
        model, _, _ = exp02._load_frozen_model(run_spec, data, config)  # type: ignore[arg-type]
    else:
        model, _ = exp02._load_new_model(run_spec, data, config)  # type: ignore[arg-type]
    return model


def _alpha_and_tau_ms(shift: int, fs: float) -> tuple[float, float]:
    alpha = 1.0 - 2.0 ** (-int(shift))
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"Invalid alpha for shift {shift}: {alpha}")
    tau_ms = -1000.0 / (float(fs) * math.log(alpha))
    return alpha, tau_ms


def evaluate_one(
    repo_root: Path,
    spec: EvalSpec,
    *,
    device_name: str = "cpu",
    batch_size: int | None = None,
    threads: int = 1,
) -> pd.DataFrame:
    torch.set_num_threads(max(1, int(threads)))
    device = torch.device(device_name)
    data = exp02.prepare_data(repo_root)
    effective_batch_size = batch_size or exp02.BATCH_SIZE
    model = _load_checkpoint_model(
        repo_root,
        data,
        spec,
        device_name=device_name,
        batch_size=effective_batch_size,
        threads=threads,
    )
    loaders = exp02.make_loaders(data, spec.exp02_spec(), effective_batch_size, False)
    loader = loaders["test"]

    final_shifts = tuple(int(value) for value in exp02.DIRECT_ARCHITECTURES[spec.architecture][-1])
    slices = _shift_slices(final_shifts, exp02.HIDDEN_WIDTH)
    accum: dict[int, dict[str, float]] = {
        shift: {
            "valid_spikes": 0.0,
            "valid_neuron_steps": 0.0,
            "stage1_spikes": 0.0,
            "stage1_neuron_steps": 0.0,
            "stage2_spikes": 0.0,
            "stage2_neuron_steps": 0.0,
            "stage3_spikes": 0.0,
            "stage3_neuron_steps": 0.0,
        }
        for shift, _, _ in slices
    }
    n_samples = 0

    with torch.no_grad():
        for X, _, lengths in loader:
            X = X.to(device)
            lengths = lengths.to(device)
            rollout = exp02.endpoint_rollout_input(X, lengths, data.fs)
            trajectory = model.forward_trajectory(rollout)
            hidden = trajectory.get("hidden_spikes")
            if not isinstance(hidden, tuple) or not hidden:
                raise TypeError("Expected hidden_spikes tuple")
            final_hidden = hidden[-1]
            n_steps = final_hidden.shape[1]
            valid_mask = exp02.exp01.exp50.valid_mask(lengths, n_steps)
            stage_masks = exp02.tail_stage_masks(lengths, n_steps, data.fs)

            for shift, start, stop in slices:
                selected = final_hidden[:, :, start:stop]
                for name, mask in (
                    ("valid", valid_mask),
                    ("stage1", stage_masks[0]),
                    ("stage2", stage_masks[1]),
                    ("stage3", stage_masks[2]),
                ):
                    mask_f = mask.to(selected.dtype).unsqueeze(-1)
                    accum[shift][f"{name}_spikes"] += float((selected * mask_f).sum().item())
                    accum[shift][f"{name}_neuron_steps"] += float(mask.sum().item()) * float(stop - start)
            n_samples += len(X)

    if n_samples <= 0:
        raise RuntimeError("Cannot evaluate an empty test loader")

    rows: list[dict[str, object]] = []
    b1, b2, b3 = exp02.tail_stage_boundaries(data.fs)
    for shift, start, stop in slices:
        stats = accum[shift]

        def rate(name: str) -> float:
            neuron_seconds = stats[f"{name}_neuron_steps"] / float(data.fs)
            return stats[f"{name}_spikes"] / neuron_seconds if neuron_seconds > 0 else float("nan")

        valid_rate = rate("valid")
        stage_rates = [rate(f"stage{index}") for index in (1, 2, 3)]
        tail_spikes = sum(stats[f"stage{index}_spikes"] for index in (1, 2, 3))
        tail_neuron_steps = sum(stats[f"stage{index}_neuron_steps"] for index in (1, 2, 3))
        tail_neuron_seconds = tail_neuron_steps / float(data.fs)
        tail_rate = tail_spikes / tail_neuron_seconds if tail_neuron_seconds > 0 else float("nan")
        neuron_count = stop - start
        alpha, tau_syn_ms = _alpha_and_tau_ms(shift, data.fs)

        rows.append(
            {
                "architecture": spec.architecture,
                "objective": spec.objective,
                "profile": spec.profile,
                "seed": spec.seed,
                "shift": shift,
                "alpha": alpha,
                "tau_syn_ms": tau_syn_ms,
                "tau_group": "long" if shift >= LONG_SHIFT_MIN else "short_mid",
                "neuron_start": start,
                "neuron_stop": stop,
                "neuron_count": neuron_count,
                "n_test_samples": n_samples,
                "fs_hz": float(data.fs),
                "tail_stage1_steps": b1,
                "tail_stage2_steps": b2 - b1,
                "tail_stage3_steps": b3 - b2,
                "valid_firing_rate_hz": valid_rate,
                "tail_stage1_firing_rate_hz": stage_rates[0],
                "tail_stage2_firing_rate_hz": stage_rates[1],
                "tail_stage3_firing_rate_hz": stage_rates[2],
                "tail_firing_rate_hz": tail_rate,
                "valid_spikes_per_neuron_sample": stats["valid_spikes"] / (neuron_count * n_samples),
                "tail_stage1_spikes_per_neuron_sample": stats["stage1_spikes"] / (neuron_count * n_samples),
                "tail_stage2_spikes_per_neuron_sample": stats["stage2_spikes"] / (neuron_count * n_samples),
                "tail_stage3_spikes_per_neuron_sample": stats["stage3_spikes"] / (neuron_count * n_samples),
                "tail_spikes_per_neuron_sample": tail_spikes / (neuron_count * n_samples),
                "tail_to_valid_rate_ratio": tail_rate / valid_rate if valid_rate > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def run_one(repo_root: Path, array_task_id: int, device: str, batch_size: int | None, threads: int) -> Path:
    specs = eval_specs()
    if not 0 <= array_task_id < len(specs):
        raise IndexError(f"array_task_id must be in [0,{len(specs) - 1}], got {array_task_id}")
    spec = specs[array_task_id]
    frame = evaluate_one(repo_root, spec, device_name=device, batch_size=batch_size, threads=threads)
    path = per_checkpoint_path(repo_root, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"[exp0.2-shift-fr] task={array_task_id} spec={spec.key} rows={len(frame)} -> {path}")
    return path


def _paired_vs_none(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["profile"] == exp02.FROZEN_PROFILE].copy()
    base_cols = ["architecture", "objective", "seed", "shift"] + list(RATE_COLUMNS)
    baseline = baseline[base_cols].rename(columns={column: f"none_{column}" for column in RATE_COLUMNS})
    merged = summary.merge(
        baseline,
        on=["architecture", "objective", "seed", "shift"],
        how="left",
        validate="many_to_one",
    )
    if merged[[f"none_{column}" for column in RATE_COLUMNS]].isna().all(axis=1).any():
        raise RuntimeError("Missing matched frozen Exp0.1 baseline for at least one per-shift row")

    for column in RATE_COLUMNS:
        base = f"none_{column}"
        merged[f"delta_{column}_vs_none"] = merged[column] - merged[base]
        merged[f"pct_change_{column}_vs_none"] = np.where(
            merged[base].abs() > 0,
            100.0 * (merged[column] - merged[base]) / merged[base],
            np.nan,
        )

    # Positive means tail firing was suppressed more strongly than valid-region firing.
    valid_pct = merged["pct_change_valid_firing_rate_hz_vs_none"]
    merged["tail_specific_suppression_pp"] = valid_pct - merged["pct_change_tail_firing_rate_hz_vs_none"]
    for stage in (1, 2, 3):
        merged[f"tail_stage{stage}_specific_suppression_pp"] = (
            valid_pct - merged[f"pct_change_tail_stage{stage}_firing_rate_hz_vs_none"]
        )
    return merged


def _weighted_group_mean(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float]:
    weights = frame["neuron_count"].to_numpy(dtype=float)
    result: dict[str, float] = {}
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        result[column] = float(np.average(values[finite], weights=weights[finite])) if finite.any() else float("nan")
    return result


def _long_tau_summary(summary: pd.DataFrame) -> pd.DataFrame:
    long_rows = summary[summary["shift"] >= LONG_SHIFT_MIN].copy()
    if long_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["architecture", "objective", "profile", "seed"]
    for key, frame in long_rows.groupby(keys, sort=False):
        row = dict(zip(keys, key, strict=True))
        row["long_shifts"] = ",".join(str(int(v)) for v in sorted(frame["shift"].unique()))
        row["long_neuron_count"] = int(frame["neuron_count"].sum())
        row.update(_weighted_group_mean(frame, RATE_COLUMNS))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_long_vs_none(long_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = long_summary[long_summary["profile"] == exp02.FROZEN_PROFILE].copy()
    keys = ["architecture", "objective", "seed"]
    baseline = baseline[keys + list(RATE_COLUMNS)].rename(
        columns={column: f"none_{column}" for column in RATE_COLUMNS}
    )
    merged = long_summary.merge(baseline, on=keys, how="left", validate="many_to_one")
    for column in RATE_COLUMNS:
        base = f"none_{column}"
        merged[f"delta_{column}_vs_none"] = merged[column] - merged[base]
        merged[f"pct_change_{column}_vs_none"] = np.where(
            merged[base].abs() > 0,
            100.0 * (merged[column] - merged[base]) / merged[base],
            np.nan,
        )
    merged["tail_specific_suppression_pp"] = (
        merged["pct_change_valid_firing_rate_hz_vs_none"]
        - merged["pct_change_tail_firing_rate_hz_vs_none"]
    )
    return merged


def _comparison(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    comparison = summary.groupby(group_cols)[list(RATE_COLUMNS)].agg(["mean", "std"]).reset_index()
    comparison.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in comparison.columns
    ]
    return comparison


def finalize(repo_root: Path) -> None:
    specs = eval_specs()
    root = analysis_root(repo_root)
    expected = {per_checkpoint_path(repo_root, spec) for spec in specs}
    existing = set((root / "per_checkpoint").glob("*.csv"))
    missing = sorted(str(path) for path in expected - existing)
    extras = sorted(str(path) for path in existing - expected)
    if missing or extras:
        raise RuntimeError(f"Per-checkpoint artifact mismatch: missing={missing[:5]} extras={extras[:5]}")

    frames = [pd.read_csv(per_checkpoint_path(repo_root, spec)) for spec in specs]
    summary = pd.concat(frames, ignore_index=True)
    key_cols = ["architecture", "objective", "profile", "seed", "shift"]
    if summary.duplicated(key_cols).any():
        raise RuntimeError("Duplicate architecture/objective/profile/seed/shift rows detected")

    root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / "shift_fire_rate_summary.csv", index=False)
    _comparison(summary, ["architecture", "objective", "profile", "shift"]).to_csv(
        root / "shift_fire_rate_comparison.csv", index=False
    )

    paired = _paired_vs_none(summary)
    paired.to_csv(root / "shift_fire_rate_vs_none.csv", index=False)

    long_summary = _long_tau_summary(summary)
    if not long_summary.empty:
        long_summary.to_csv(root / "shift_fire_rate_long_tau_summary.csv", index=False)
        _comparison(long_summary, ["architecture", "objective", "profile"]).to_csv(
            root / "shift_fire_rate_long_tau_comparison.csv", index=False
        )
        _paired_long_vs_none(long_summary).to_csv(root / "shift_fire_rate_long_tau_vs_none.csv", index=False)

    fs_values = summary["fs_hz"].dropna().unique()
    if len(fs_values) != 1:
        raise RuntimeError(f"Expected one sampling rate, got {fs_values}")
    fs = float(fs_values[0])
    stage_steps = [
        int(summary["tail_stage1_steps"].iloc[0]),
        int(summary["tail_stage2_steps"].iloc[0]),
        int(summary["tail_stage3_steps"].iloc[0]),
    ]
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "analysis_version": ANALYSIS_VERSION,
        "source_experiment": exp02.EXPERIMENT_ID,
        "source_protocol": exp02.PROTOCOL_VERSION,
        "checkpoint_count": len(specs),
        "profiles": list(PROFILES),
        "architectures": list(exp02.DIRECT_ARCHITECTURES),
        "objectives": list(exp02.OBJECTIVES),
        "seeds": list(exp02.SEEDS),
        "long_shift_min": LONG_SHIFT_MIN,
        "long_group_definition": "configured final-hidden shifts s6/s7 (shift >= 6); all per-shift rows remain primary",
        "firing_rate_denominator": "spike count / neuron-seconds",
        "fs_hz": fs,
        "tail_stage_lengths_steps": stage_steps,
        "tail_stage_boundaries_ms_requested": list(exp02.TAIL_STAGE_MS),
        "post_endpoint_input": "explicit zero input via Exp0.2 endpoint_rollout_input",
        "training": "none; checkpoint-only post-evaluation",
        "tail_specific_suppression_pp": "pct_change_valid_FR_vs_none - pct_change_tail_FR_vs_none; positive means preferential tail suppression",
    }
    (root / "shift_fire_rate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[exp0.2-shift-fr] finalized {len(summary)} per-shift rows from {len(specs)} checkpoints -> {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp0.2 shift-resolved final-hidden firing-rate post-evaluation"
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.command == "run-one":
        run_one(repo_root, args.array_task_id, args.device, args.batch_size, args.threads)
    elif args.command == "finalize":
        finalize(repo_root)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
