from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "snn").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate writingRing repository root")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_3_0_3_l3_bottleneck_ablation import (  # noqa: E402
    EVAL_ARCHITECTURES,
    EXPECTED_EVAL_RUNS,
    EXPERIMENT_ID,
    OBJECTIVES,
    PROTOCOL_VERSION,
    SEEDS,
    aggregate_mean_sd,
    architecture_rows,
    eval_specs,
    evaluation_path,
    last_layer_name,
    native_result_and_history,
    results_dir,
)


def main() -> None:
    root = results_dir(REPO_ROOT)
    missing: list[str] = []
    native_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    layer_probe_rows: list[dict[str, object]] = []
    subgroup_probe_rows: list[dict[str, object]] = []
    firing_rows: list[dict[str, object]] = []

    for architecture, objective, seed in eval_specs():
        path = evaluation_path(root, architecture, objective, seed)
        if not path.exists():
            missing.append(str(path.relative_to(REPO_ROOT)))
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = (PROTOCOL_VERSION, architecture, objective, seed)
        actual = (
            str(payload.get("protocol_version")),
            str(payload.get("architecture")),
            str(payload.get("objective")),
            int(payload.get("seed")),
        )
        if actual != expected:
            raise ValueError(f"Evaluation identity mismatch for {path}: {actual} != {expected}")

        native, history = native_result_and_history(
            REPO_ROOT, root, architecture, objective, seed
        )
        native_rows.append(native)
        history_rows.extend(history)

        common = {
            "architecture": architecture,
            "objective": objective,
            "seed": int(seed),
            "last_layer": last_layer_name(architecture),
        }
        for row in payload["layer_probes"]:
            layer_probe_rows.append({**common, **row})
        for row in payload["subgroup_probes"]:
            subgroup_probe_rows.append({**common, **row})
        for row in payload["firing_rates"]:
            firing_rows.append({**common, **row})

    if missing:
        print(f"Completed evaluations: {EXPECTED_EVAL_RUNS - len(missing)}/{EXPECTED_EVAL_RUNS}")
        print("Missing evaluation artifacts:")
        for item in missing:
            print(" -", item)
        raise SystemExit(2)

    native = pd.DataFrame(native_rows).sort_values(["objective", "architecture", "seed"])
    history = pd.DataFrame(history_rows).sort_values(
        ["objective", "architecture", "seed", "epoch"]
    )
    layer_probes = pd.DataFrame(layer_probe_rows).sort_values(
        ["objective", "architecture", "layer", "probe_type", "seed"]
    )
    subgroup_probes = pd.DataFrame(subgroup_probe_rows)
    if not subgroup_probes.empty:
        subgroup_probes = subgroup_probes.sort_values(
            ["objective", "architecture", "layer", "shift", "probe_type", "seed"]
        )
    firing = pd.DataFrame(firing_rows).sort_values(
        ["objective", "architecture", "split", "scope", "layer", "shift", "seed"]
    )

    if len(native) != EXPECTED_EVAL_RUNS:
        raise RuntimeError((len(native), EXPECTED_EVAL_RUNS))
    if set(native["architecture"].unique()) != set(EVAL_ARCHITECTURES):
        raise ValueError("Missing architecture in native results")
    if set(native["objective"].unique()) != set(OBJECTIVES):
        raise ValueError("Missing objective in native results")
    if set(native["seed"].unique()) != set(SEEDS):
        raise ValueError("Missing seed in native results")

    native_summary = aggregate_mean_sd(
        native,
        ["objective", "architecture", "last_layer"],
        [
            "test_balanced_accuracy",
            "test_macro_f1",
            "test_accuracy",
            "val_balanced_accuracy",
            "train_balanced_accuracy",
            "train_test_ba_gap",
            "best_epoch",
        ],
    )
    layer_probe_summary = aggregate_mean_sd(
        layer_probes,
        ["objective", "architecture", "last_layer", "layer", "probe_type"],
        [
            "feature_dim",
            "probe_val_balanced_accuracy",
            "probe_test_balanced_accuracy",
            "probe_test_macro_f1",
            "probe_test_accuracy",
        ],
    )
    if subgroup_probes.empty:
        subgroup_probe_summary = pd.DataFrame()
    else:
        subgroup_probe_summary = aggregate_mean_sd(
            subgroup_probes,
            [
                "objective",
                "architecture",
                "last_layer",
                "layer",
                "shift",
                "tau_syn_ms",
                "group_neurons",
                "probe_type",
            ],
            [
                "feature_dim",
                "probe_val_balanced_accuracy",
                "probe_test_balanced_accuracy",
                "probe_test_macro_f1",
                "probe_test_accuracy",
            ],
        )
    firing_summary = aggregate_mean_sd(
        firing,
        [
            "objective",
            "architecture",
            "last_layer",
            "split",
            "scope",
            "layer",
            "shift",
        ],
        ["neurons", "firing_rate"],
    )

    # Diagnostic deltas: L3 probe minus L2 probe for architectures with L3,
    # and last-layer fixed250 probe minus native BA for all architectures.
    fixed_probe = layer_probes[layer_probes["probe_type"] == "fixed250"].copy()
    l2 = fixed_probe[fixed_probe["layer"] == "L2"][
        ["architecture", "objective", "seed", "probe_test_balanced_accuracy"]
    ].rename(columns={"probe_test_balanced_accuracy": "l2_fixed250_probe_ba"})
    last_rows = []
    for architecture in EVAL_ARCHITECTURES:
        layer = last_layer_name(architecture)
        part = fixed_probe[
            (fixed_probe["architecture"] == architecture) & (fixed_probe["layer"] == layer)
        ].copy()
        part = part[
            ["architecture", "objective", "seed", "probe_test_balanced_accuracy"]
        ].rename(columns={"probe_test_balanced_accuracy": "last_fixed250_probe_ba"})
        last_rows.append(part)
    last_probe = pd.concat(last_rows, ignore_index=True)
    diagnostics = native[
        ["architecture", "objective", "seed", "test_balanced_accuracy", "last_layer"]
    ].merge(l2, on=["architecture", "objective", "seed"], how="left").merge(
        last_probe, on=["architecture", "objective", "seed"], how="left"
    )
    diagnostics["last_minus_l2_fixed250_probe_ba"] = (
        diagnostics["last_fixed250_probe_ba"] - diagnostics["l2_fixed250_probe_ba"]
    )
    diagnostics.loc[
        diagnostics["last_layer"] == "L2", "last_minus_l2_fixed250_probe_ba"
    ] = 0.0
    diagnostics["readout_utilization_gap"] = (
        diagnostics["last_fixed250_probe_ba"] - diagnostics["test_balanced_accuracy"]
    )
    diagnostic_summary = aggregate_mean_sd(
        diagnostics,
        ["objective", "architecture", "last_layer"],
        [
            "l2_fixed250_probe_ba",
            "last_fixed250_probe_ba",
            "last_minus_l2_fixed250_probe_ba",
            "test_balanced_accuracy",
            "readout_utilization_gap",
        ],
    )

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "experiment_3_0_3_architectures.csv": pd.DataFrame(architecture_rows()),
        "experiment_3_0_3_native_results.csv": native,
        "experiment_3_0_3_history.csv": history,
        "experiment_3_0_3_layer_probes.csv": layer_probes,
        "experiment_3_0_3_tau_subgroup_probes.csv": subgroup_probes,
        "experiment_3_0_3_firing_rates.csv": firing,
        "experiment_3_0_3_diagnostics.csv": diagnostics,
        "experiment_3_0_3_native_summary.csv": native_summary,
        "experiment_3_0_3_layer_probe_summary.csv": layer_probe_summary,
        "experiment_3_0_3_tau_subgroup_probe_summary.csv": subgroup_probe_summary,
        "experiment_3_0_3_firing_rate_summary.csv": firing_summary,
        "experiment_3_0_3_diagnostic_summary.csv": diagnostic_summary,
    }
    for filename, frame in outputs.items():
        path = root / filename
        frame.to_csv(path, index=False)
        print("Wrote:", path.relative_to(REPO_ROOT), f"rows={len(frame)}")

    print("Experiment:", EXPERIMENT_ID)
    print("Protocol:", PROTOCOL_VERSION)
    print(f"Completed evaluations: {len(native)}/{EXPECTED_EVAL_RUNS}")


if __name__ == "__main__":
    main()
