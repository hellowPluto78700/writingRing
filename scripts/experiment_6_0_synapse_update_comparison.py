from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts import experiment_6_0_legacy_synapse as legacy
from scripts import experiment_6_0_multiscale_phase_evidence as base


NORMALIZED = "normalized_unit_dc"
LEGACY = legacy.SYNAPSE_MODE
SYNAPSE_MODES = (NORMALIZED, LEGACY)
MODE_LABELS = {
    NORMALIZED: "Normalized: alpha I + (1-alpha) Wx",
    LEGACY: "Legacy: alpha I + Wx",
}
OBJECTIVE_LABELS = {
    "whole_count": "WholeCount",
    "whole_count_hce": "WholeCount + HCE",
    "whole_count_contextual_gain": "WholeCount + Contextual Gain",
}


def _load_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Exp6.0 artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mode_root(repo_root: Path, synapse_mode: str) -> Path:
    if synapse_mode == NORMALIZED:
        return base.results_dir(repo_root)
    if synapse_mode == LEGACY:
        return legacy.legacy_results_dir(repo_root)
    raise ValueError(f"Unknown synapse mode: {synapse_mode}")


def _combined_rows(repo_root: Path) -> tuple[pd.DataFrame, list[int]]:
    rows: list[dict[str, object]] = []
    sample_indices: list[int] = []
    for synapse_mode in SYNAPSE_MODES:
        root = _mode_root(repo_root, synapse_mode)
        for spec in base.run_specs():
            payload = _load_payload(base.evaluation_path(root, spec))
            row = base._row_from_evaluation(spec, payload)
            row["synapse_mode"] = synapse_mode
            row["synaptic_update"] = (
                "I_t = alpha*I_{t-1} + (1-alpha)*W*x_t"
                if synapse_mode == NORMALIZED
                else legacy.SYNAPTIC_UPDATE
            )
            rows.append(row)
            activity = payload.get("activity")
            if not isinstance(activity, dict):
                raise TypeError("Exp6.0 evaluation activity must be a dict")
            sample_indices.append(int(activity["validation_index"]))
    return pd.DataFrame(rows), sample_indices


def _paired_deltas(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["objective", "mem_shift", "seed"]
    metrics = [
        "val_ba",
        "test_ba",
        "test_accuracy",
        "test_macro_f1",
        "hidden_s4_fr_valid",
        "hidden_s5_fr_valid",
        "hidden_s6_fr_valid",
        "hidden_s4_fr_tail",
        "hidden_s5_fr_tail",
        "hidden_s6_fr_tail",
    ]
    norm = runs[runs["synapse_mode"] == NORMALIZED][keys + metrics].copy()
    old = runs[runs["synapse_mode"] == LEGACY][keys + metrics].copy()
    merged = norm.merge(old, on=keys, suffixes=("_normalized", "_legacy"), validate="one_to_one")
    for metric in metrics:
        merged[f"{metric}_legacy_minus_normalized"] = (
            merged[f"{metric}_legacy"] - merged[f"{metric}_normalized"]
        )
    return merged


def _summary_plots(
    runs: pd.DataFrame,
    deltas: pd.DataFrame,
    baseline_test_ba: float,
    root: Path,
) -> None:
    plot_dir = root / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, filename in (
        ("test_ba", "Test balanced accuracy", "test_ba_vs_mem_shift.png"),
        ("val_ba", "Validation balanced accuracy", "val_ba_vs_mem_shift.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
        for ax, synapse_mode in zip(axes, SYNAPSE_MODES, strict=True):
            mode_rows = runs[runs["synapse_mode"] == synapse_mode]
            for objective in base.OBJECTIVES:
                subset = mode_rows[mode_rows["objective"] == objective]
                grouped = subset.groupby("mem_shift")[metric]
                means = grouped.mean().reindex(base.MEM_SHIFTS)
                stds = grouped.std(ddof=1).reindex(base.MEM_SHIFTS).fillna(0.0)
                ax.errorbar(
                    base.MEM_SHIFTS,
                    means.to_numpy(),
                    yerr=stds.to_numpy(),
                    marker="o",
                    capsize=4,
                    label=OBJECTIVE_LABELS[objective],
                )
            if metric == "test_ba":
                ax.axhline(
                    baseline_test_ba,
                    linestyle="--",
                    label="Raw Fixed250 + Linear",
                )
            ax.set_title(MODE_LABELS[synapse_mode])
            ax.set_xticks(base.MEM_SHIFTS)
            ax.set_xlabel("Hidden membrane shift")
            ax.set_ylim(0.0, 1.0)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for row_index, synapse_mode in enumerate(SYNAPSE_MODES):
        mode_rows = runs[runs["synapse_mode"] == synapse_mode]
        for col_index, shift_name in enumerate(("s4", "s5", "s6")):
            ax = axes[row_index, col_index]
            column = f"hidden_{shift_name}_fr_valid"
            for objective in base.OBJECTIVES:
                subset = mode_rows[mode_rows["objective"] == objective]
                grouped = subset.groupby("mem_shift")[column]
                means = grouped.mean().reindex(base.MEM_SHIFTS)
                stds = grouped.std(ddof=1).reindex(base.MEM_SHIFTS).fillna(0.0)
                ax.errorbar(
                    base.MEM_SHIFTS,
                    means.to_numpy(),
                    yerr=stds.to_numpy(),
                    marker="o",
                    capsize=4,
                    label=OBJECTIVE_LABELS[objective],
                )
            ax.set_title(f"{MODE_LABELS[synapse_mode]} | {shift_name}")
            ax.set_xticks(base.MEM_SHIFTS)
            ax.set_xlabel("Hidden membrane shift")
    axes[0, 0].set_ylabel("Valid-region spike fraction")
    axes[1, 0].set_ylabel("Valid-region spike fraction")
    axes[0, -1].legend()
    fig.tight_layout()
    fig.savefig(
        plot_dir / "hidden_firing_rate_vs_mem_shift.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    delta_col = "test_ba_legacy_minus_normalized"
    for objective in base.OBJECTIVES:
        subset = deltas[deltas["objective"] == objective]
        grouped = subset.groupby("mem_shift")[delta_col]
        means = grouped.mean().reindex(base.MEM_SHIFTS)
        stds = grouped.std(ddof=1).reindex(base.MEM_SHIFTS).fillna(0.0)
        ax.errorbar(
            base.MEM_SHIFTS,
            means.to_numpy(),
            yerr=stds.to_numpy(),
            marker="o",
            capsize=4,
            label=OBJECTIVE_LABELS[objective],
        )
    ax.axhline(0.0, linestyle="--")
    ax.set_xticks(base.MEM_SHIFTS)
    ax.set_xlabel("Hidden membrane shift")
    ax.set_ylabel("Test BA: legacy - normalized")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        plot_dir / "paired_synapse_update_test_ba_delta.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(fig)


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = base.results_dir(repo_root)
    baseline_payload = _load_payload(base.baseline_path(root))
    baseline_test_ba = float(
        baseline_payload["baseline"]["metrics"]["test"]["balanced_accuracy"]
    )

    runs, sample_indices = _combined_rows(repo_root)
    if len(runs) != 54:
        raise ValueError(f"Exp6.0 comparison requires 54 SNN runs, got {len(runs)}")
    if len(set(sample_indices)) != 1:
        raise ValueError(
            "Normalized and legacy Exp6.0 workers did not use the same visualization sample"
        )

    root.mkdir(parents=True, exist_ok=True)
    runs_file = root / "runs.csv"
    runs.to_csv(runs_file, index=False)

    summary = (
        runs.groupby(["synapse_mode", "objective", "mem_shift", "tau_mem_ms"])[
            [
                "val_ba",
                "test_ba",
                "test_accuracy",
                "test_macro_f1",
                "hidden_s4_fr_valid",
                "hidden_s5_fr_valid",
                "hidden_s6_fr_valid",
                "hidden_s4_fr_tail",
                "hidden_s5_fr_tail",
                "hidden_s6_fr_tail",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary_file = root / "summary.csv"
    summary.to_csv(summary_file, index=False)

    deltas = _paired_deltas(runs)
    delta_file = root / "paired_synapse_update_deltas.csv"
    deltas.to_csv(delta_file, index=False)

    history_rows: list[dict[str, object]] = []
    for synapse_mode in SYNAPSE_MODES:
        mode_root = _mode_root(repo_root, synapse_mode)
        for spec in base.run_specs():
            history_rows.append(
                {
                    "synapse_mode": synapse_mode,
                    "objective": spec.objective,
                    "mem_shift": spec.mem_shift,
                    "seed": spec.seed,
                    "history_csv": str(
                        base.history_path(mode_root, spec).relative_to(repo_root)
                    ),
                    "learning_curve_png": str(
                        base.learning_curve_path(mode_root, spec).relative_to(repo_root)
                    ),
                    "raster_png": str(
                        base.raster_path(mode_root, spec).relative_to(repo_root)
                    ),
                }
            )
    history_index = pd.DataFrame(history_rows)
    history_index_file = root / "history_index.csv"
    history_index.to_csv(history_index_file, index=False)

    first_spec = base.run_specs()[0]
    first_payload = _load_payload(base.evaluation_path(root, first_spec))
    first_activity = first_payload["activity"]
    if not isinstance(first_activity, dict):
        raise TypeError("Exp6.0 activity payload must be a dict")
    sample_manifest = {
        "validation_index": int(first_activity["validation_index"]),
        "valid_length": int(first_activity["valid_length"]),
        "label_index": int(first_activity["true_label_index"]),
        "label": str(first_activity["true_label"]),
        "selection_rule": "same deterministic validation sample for all 54 paired runs; selected before inspecting model results",
        "visualization_steps": base.VIS_STEPS,
    }
    sample_file = root / "visualization_sample.json"
    base._save_json(sample_file, sample_manifest)

    manifest = {
        "experiment_id": base.EXPERIMENT_ID,
        "protocol_version": base.PROTOCOL_VERSION,
        "runs": len(runs),
        "runs_per_synapse_mode": len(base.run_specs()),
        "synapse_modes": {
            NORMALIZED: "I_t = alpha*I_{t-1} + (1-alpha)*W*x_t",
            LEGACY: legacy.SYNAPTIC_UPDATE,
        },
        "paired_randomness": "model initialization and loader random streams are identical across synapse-update modes for matched objective/mem_shift/seed",
        "epochs": base.EPOCHS,
        "training_seeds": list(base.TRAIN_SEEDS),
        "user_split_seed": int(base.exp3.SPLIT_SEED),
        "objectives": list(base.OBJECTIVES),
        "hidden_mem_shifts": list(base.MEM_SHIFTS),
        "hidden_syn_shifts": list(base.SYN_SHIFTS),
        "hidden_group_counts": list(base.GROUP_COUNTS),
        "binary_spikes": True,
        "multi_cpu_policy": "27 one-core normalized SNN tasks + 27 one-core legacy SNN tasks + one one-core Fixed250 baseline; one artifact-only comparison finalizer",
        "notebook_policy": "analysis-only; notebook reads finalized CSV/JSON/PNG artifacts and never trains",
        "baseline_test_ba": baseline_test_ba,
    }
    manifest_file = root / "manifest.json"
    base._save_json(manifest_file, manifest)
    _summary_plots(runs, deltas, baseline_test_ba, root)

    return {
        "runs": runs_file,
        "summary": summary_file,
        "paired_deltas": delta_file,
        "history_index": history_index_file,
        "visualization_sample": sample_file,
        "manifest": manifest_file,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize the paired Exp6.0 synaptic-update comparison"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("finalize")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = base.find_repo_root()
    if args.command == "finalize":
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
