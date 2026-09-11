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
LONG_SHIFT_MIN = 5


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
    # Preserve the same deterministic equal-width allocation used by Exp0.2 rasters.
    if hasattr(exp02, "_shift_slices"):
        return list(exp02._shift_slices(shifts, width))  # type: ignore[attr-defined]
    base, remainder = divmod(width, len(shifts))
    start = 0
    result: list[tuple[int, int, int]] = []
    for index, shift in enumerate(shifts):
        count = base + (1 if index < remainder else 0)
        stop = start + count
        result.append((int(shift), start, stop))
        start = stop
    if start != width:
        raise RuntimeError("Shift slices do not span the final hidden layer")
    return result


def _spike_stats(
    spikes: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    stop: int,
    fs: float,
) -> tuple[float, float, int, int]:
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
    seconds_per_neuron = neuron_steps / float(fs)
    firing_rate_hz = spike_count / seconds_per_neuron if seconds_per_neuron > 0 else float("nan")
    spikes_per_neuron = spike_count / neuron_count if neuron_count > 0 else float("nan")
    return firing_rate_hz, spikes_per_neuron, timestep_count, neuron_count


def _load_checkpoint_model(
    repo_root: Path,
    data: object,
    spec: EvalSpec,
    device: torch.device,
) -> exp02.DiagnosticMultiTauHierarchySNN:
    run_spec = spec.exp02_spec()
    model = exp02._new_model(run_spec, data).to(device)  # type: ignore[arg-type]
    if spec.profile == exp02.FROZEN_PROFILE:
        base_path = exp02.base_results_dir(repo_root) / "checkpoints" / f"{exp02.base_spec(run_spec).key}.pt"
        checkpoint = torch.load(base_path, map_location=device)
    else:
        path = exp02.checkpoint_path(artifact_root(repo_root), run_spec)
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


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
    model = _load_checkpoint_model(repo_root, data, spec, device)
    loaders = exp02.make_loaders(data, spec.exp02_spec(), batch_size or exp02.BATCH_SIZE, False)
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

    with torch.no_grad():
        for X, y, lengths in loader:
            del y
            X = X.to(device)
            lengths = lengths.to(device)
            rollout = exp02.endpoint_rollout_input(X, lengths, data.fs)
            trajectory = model.forward_trajectory(rollout)
            hidden = trajectory.get("hidden_spikes")
            if not isinstance(hidden, tuple) or not hidden:
                raise TypeError("Expected hidden_spikes tuple")
            final_hidden = hidden[-1]
            n_steps = final_hidden.shape[1]
            t = torch.arange(n_steps, device=device).unsqueeze(0)
            valid_mask = t < lengths.unsqueeze(1)
            stage_masks = exp02.tail_stage_masks(lengths, n_steps, data.fs)

            for shift, start, stop in slices:
                for name, mask in (
                    ("valid", valid_mask),
                    ("stage1", stage_masks[0]),
                    ("stage2", stage_masks[1]),
                    ("stage3", stage_masks[2]),
                ):
                    selected = final_hidden[:, :, start:stop]
                    mask_f = mask.to(selected.dtype).unsqueeze(-1)
                    accum[shift][f"{name}_spikes"] += float((selected * mask_f).sum().item())
                    accum[shift][f"{name}_neuron_steps"] += float(mask.sum().item()) * float(stop - start)

    rows: list[dict[str, object]] = []
    b1, b2, b3 = exp02.tail_stage_boundaries(data.fs)
    for shift, start, stop in slices:
        stats = accum[shift]

        def rate(name: str) -> float:
            denom = stats[f"{name}_neuron_steps"] / float(data.fs)
            return stats[f"{name}_spikes"] / denom if denom > 0 else float("nan")

        valid_rate = rate("valid")
        stage_rates = [rate(f"stage{index}") for index in (1, 2, 3)]
        tail_spikes = sum(stats[f"stage{index}_spikes"] for index in (1, 2, 3))
        tail_neuron_steps = sum(stats[f"stage{index}_neuron_steps"] for index in (1, 2, 3))
        tail_rate = tail_spikes / (tail_neuron_steps / float(data.fs)) if tail_neuron_steps > 0 else float("nan")
        neuron_count = stop - start
        rows.append(
            {
                "architecture": spec.architecture,
                "objective": spec.objective,
                "profile": spec.profile,
                "seed": spec.seed,
                "shift": shift,
                "tau_group": "long" if shift >= LONG_SHIFT_MIN else "short_mid",
                "neuron_start": start,
                "neuron_stop": stop,
                "neuron_count": neuron_count,
                "fs_hz": float(data.fs),
                "tail_stage1_steps": b1,
                "tail_stage2_steps": b2 - b1,
                "tail_stage3_steps": b3 - b2,
                "valid_firing_rate_hz": valid_rate,
                "tail_stage1_firing_rate_hz": stage_rates[0],
                "tail_stage2_firing_rate_hz": stage_rates[1],
                "tail_stage3_firing_rate_hz": stage_rates[2],
                "tail_firing_rate_hz": tail_rate,
                "valid_spikes_per_neuron": stats["valid_spikes"] / neuron_count,
                "tail_stage1_spikes_per_neuron": stats["stage1_spikes"] / neuron_count,
                "tail_stage2_spikes_per_neuron": stats["stage2_spikes"] / neuron_count,
                "tail_stage3_spikes_per_neuron": stats["stage3_spikes"] / neuron_count,
                "tail_spikes_per_neuron": tail_spikes / neuron_count,
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
    value_cols = [
        "valid_firing_rate_hz",
        "tail_stage1_firing_rate_hz",
        "tail_stage2_firing_rate_hz",
        "tail_stage3_firing_rate_hz",
        "tail_firing_rate_hz",
        "tail_to_valid_rate_ratio",
    ]
    base_cols = ["architecture", "objective", "seed", "shift"] + value_cols
    baseline = baseline[base_cols].rename(columns={column: f"none_{column}" for column in value_cols})
    merged = summary.merge(baseline, on=["architecture", "objective", "seed", "shift"], how="left", validate="many_to_one")
    for column in value_cols:
        base = f"none_{column}"
        merged[f"delta_{column}_vs_none"] = merged[column] - merged[base]
        merged[f"pct_change_{column}_vs_none"] = np.where(
            merged[base].abs() > 0,
            100.0 * (merged[column] - merged[base]) / merged[base],
            np.nan,
        )
    return merged


def _long_tau_summary(summary: pd.DataFrame) -> pd.DataFrame:
    long_rows = summary[summary["shift"] >= LONG_SHIFT_MIN].copy()
    if long_rows.empty:
        return pd.DataFrame()
    keys = ["architecture", "objective", "profile", "seed"]
    numeric = [
        "valid_firing_rate_hz",
        "tail_stage1_firing_rate_hz",
        "tail_stage2_firing_rate_hz",
        "tail_stage3_firing_rate_hz",
        "tail_firing_rate_hz",
        "tail_to_valid_rate_ratio",
    ]
    pooled = long_rows.groupby(keys, as_index=False)[numeric].mean()
    shifts = (
        long_rows.groupby(keys)["shift"]
        .apply(lambda values: ",".join(str(int(v)) for v in sorted(set(values))))
        .reset_index(name="long_shifts")
    )
    pooled = pooled.merge(shifts, on=keys, how="left", validate="one_to_one")
    return pooled


def finalize(repo_root: Path) -> None:
    specs = eval_specs()
    expected = {per_checkpoint_path(repo_root, spec) for spec in specs}
    existing = set((analysis_root(repo_root) / "per_checkpoint").glob("*.csv"))
    missing = sorted(str(path) for path in expected - existing)
    extras = sorted(str(path) for path in existing - expected)
    if missing or extras:
        raise RuntimeError(f"Per-checkpoint artifact mismatch: missing={missing[:5]} extras={extras[:5]}")

    frames = [pd.read_csv(per_checkpoint_path(repo_root, spec)) for spec in specs]
    summary = pd.concat(frames, ignore_index=True)
    key_cols = ["architecture", "objective", "profile", "seed", "shift"]
    if summary.duplicated(key_cols).any():
        raise RuntimeError("Duplicate architecture/objective/profile/seed/shift rows detected")

    out = analysis_root(repo_root)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "shift_fire_rate_summary.csv", index=False)

    metric_cols = [
        "valid_firing_rate_hz",
        "tail_stage1_firing_rate_hz",
        "tail_stage2_firing_rate_hz",
        "tail_stage3_firing_rate_hz",
        "tail_firing_rate_hz",
        "tail_to_valid_rate_ratio",
    ]
    group_cols = ["architecture", "objective", "profile", "shift"]
    comparison = summary.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    comparison.columns = ["_".join(part for part in column if part) if isinstance(column, tuple) else column for column in comparison.columns]
    comparison.to_csv(out / "shift_fire_rate_comparison.csv", index=False)

    paired = _paired_vs_none(summary)
    paired.to_csv(out / "shift_fire_rate_vs_none.csv", index=False)

    long_summary = _long_tau_summary(summary)
    if not long_summary.empty:
        long_summary.to_csv(out / "shift_fire_rate_long_tau_summary.csv", index=False)

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
        "long_group_definition": "final-hidden configured shifts >= 5, while all per-shift rows remain available",
        "firing_rate_denominator": "spike count / neuron-seconds",
        "tail_stage_boundaries_steps": list(exp02.tail_stage_boundaries(exp02.exp01.exp3.SAMPLE_FREQ if hasattr(exp02.exp01.exp3, 'SAMPLE_FREQ') else 64.0)),
        "tail_stage_boundaries_ms_requested": list(exp02.TAIL_STAGE_MS),
        "post_endpoint_input": "explicit zero input via Exp0.2 endpoint_rollout_input",
        "training": "none; checkpoint-only post-evaluation",
    }
    (out / "shift_fire_rate_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[exp0.2-shift-fr] finalized {len(summary)} per-shift rows from {len(specs)} checkpoints -> {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp0.2 shift-resolved final-hidden firing-rate post-evaluation")
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
