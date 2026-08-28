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

from scripts.experiment_3_0_4_l2_width_representation_capacity import (  # noqa: E402
    EVAL_WIDTHS,
    EXPECTED_EVAL_RUNS,
    EXPERIMENT_ID,
    OBJECTIVES,
    PROTOCOL_VERSION,
    SEEDS,
    aggregate_mean_sd,
    configuration_rows,
    eval_specs,
    evaluation_path,
    native_result_and_history,
    results_dir,
)


def _probe_column(
    frame: pd.DataFrame,
    layer: str,
    probe_type: str,
    value_column: str,
    output_name: str,
) -> pd.DataFrame:
    return frame[
        (frame["layer"] == layer)
        & (frame["probe_type"] == probe_type)
    ][["l2_width", "objective", "seed", value_column]].rename(
        columns={value_column: output_name}
    )


def main() -> None:
    root = results_dir(REPO_ROOT)
    missing: list[str] = []
    native_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    layer_probe_rows: list[dict[str, object]] = []
    subgroup_probe_rows: list[dict[str, object]] = []
    firing_rows: list[dict[str, object]] = []

    for width, objective, seed in eval_specs():
        path = evaluation_path(root, width, objective, seed)
        if not path.exists():
            missing.append(str(path.relative_to(REPO_ROOT)))
            continue

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = (PROTOCOL_VERSION, width, objective, seed)
        actual = (
            str(payload.get("protocol_version")),
            int(payload.get("l2_width")),
            str(payload.get("objective")),
            int(payload.get("seed")),
        )
        if actual != expected:
            raise ValueError(
                f"Evaluation identity mismatch for {path}: {actual} != {expected}"
            )

        native, history = native_result_and_history(
            REPO_ROOT,
            root,
            width,
            objective,
            seed,
        )
        native_rows.append(native)
        history_rows.extend(history)

        common = {
            "l2_width": int(width),
            "objective": objective,
            "seed": int(seed),
            "last_layer": "L2",
        }
        for row in payload["layer_probes"]:
            layer_probe_rows.append({**common, **row})
        for row in payload["subgroup_probes"]:
            subgroup_probe_rows.append({**common, **row})
        for row in payload["firing_rates"]:
            firing_rows.append({**common, **row})

    if missing:
        print(
            f"Completed evaluations: {EXPECTED_EVAL_RUNS - len(missing)}/"
            f"{EXPECTED_EVAL_RUNS}"
        )
        print("Missing evaluation artifacts:")
        for item in missing:
            print(" -", item)
        raise SystemExit(2)

    native = pd.DataFrame(native_rows).sort_values(
        ["objective", "l2_width", "seed"]
    )
    history = pd.DataFrame(history_rows).sort_values(
        ["objective", "l2_width", "seed", "epoch"]
    )
    layer_probes = pd.DataFrame(layer_probe_rows).sort_values(
        ["objective", "l2_width", "layer", "probe_type", "seed"]
    )
    subgroup_probes = pd.DataFrame(subgroup_probe_rows).sort_values(
        ["objective", "l2_width", "layer", "shift", "probe_type", "seed"]
    )
    firing = pd.DataFrame(firing_rows).sort_values(
        ["objective", "l2_width", "split", "scope", "layer", "shift", "seed"]
    )

    if len(native) != EXPECTED_EVAL_RUNS:
        raise RuntimeError((len(native), EXPECTED_EVAL_RUNS))
    if set(native["l2_width"].unique()) != set(EVAL_WIDTHS):
        raise ValueError("Missing L2 width in native results")
    if set(native["objective"].unique()) != set(OBJECTIVES):
        raise ValueError("Missing objective in native results")
    if set(native["seed"].unique()) != set(SEEDS):
        raise ValueError("Missing seed in native results")

    native_summary = aggregate_mean_sd(
        native,
        ["objective", "l2_width", "last_layer"],
        [
            "test_balanced_accuracy",
            "test_macro_f1",
            "test_accuracy",
            "val_balanced_accuracy",
            "train_balanced_accuracy",
            "train_test_ba_gap",
            "best_epoch",
            "head_feature_dim",
            "head_parameter_count",
        ],
    )
    layer_probe_summary = aggregate_mean_sd(
        layer_probes,
        ["objective", "l2_width", "last_layer", "layer", "probe_type"],
        [
            "input_feature_dim",
            "feature_dim",
            "pca_explained_variance_ratio_sum",
            "probe_val_balanced_accuracy",
            "probe_test_balanced_accuracy",
            "probe_test_macro_f1",
            "probe_test_accuracy",
        ],
    )
    subgroup_probe_summary = aggregate_mean_sd(
        subgroup_probes,
        [
            "objective",
            "l2_width",
            "last_layer",
            "layer",
            "shift",
            "tau_syn_ms",
            "group_neurons",
            "probe_type",
        ],
        [
            "input_feature_dim",
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
            "l2_width",
            "last_layer",
            "split",
            "scope",
            "layer",
            "shift",
        ],
        [
            "neurons",
            "firing_rate",
            "total_spikes_per_timestep",
        ],
    )

    diagnostics = native[
        [
            "l2_width",
            "objective",
            "seed",
            "test_balanced_accuracy",
            "train_test_ba_gap",
            "head_feature_dim",
            "head_parameter_count",
        ]
    ].copy()

    for layer, prefix in (("L1", "l1"), ("L2", "l2")):
        raw = _probe_column(
            layer_probes,
            layer,
            "fixed250",
            "probe_test_balanced_accuracy",
            f"{prefix}_fixed250_raw_probe_ba",
        )
        pca = _probe_column(
            layer_probes,
            layer,
            "fixed250_pca128",
            "probe_test_balanced_accuracy",
            f"{prefix}_fixed250_pca128_probe_ba",
        )
        diagnostics = diagnostics.merge(
            raw,
            on=["l2_width", "objective", "seed"],
            how="left",
        ).merge(
            pca,
            on=["l2_width", "objective", "seed"],
            how="left",
        )

    l2_pca_variance = _probe_column(
        layer_probes,
        "L2",
        "fixed250_pca128",
        "pca_explained_variance_ratio_sum",
        "l2_pca128_explained_variance_ratio_sum",
    )
    diagnostics = diagnostics.merge(
        l2_pca_variance,
        on=["l2_width", "objective", "seed"],
        how="left",
    )

    l2_test_firing = firing[
        (firing["split"] == "test")
        & (firing["scope"] == "whole_layer")
        & (firing["layer"] == "L2")
    ][
        [
            "l2_width",
            "objective",
            "seed",
            "firing_rate",
            "total_spikes_per_timestep",
        ]
    ].rename(
        columns={
            "firing_rate": "l2_test_firing_rate",
            "total_spikes_per_timestep": "l2_test_total_spikes_per_timestep",
        }
    )
    diagnostics = diagnostics.merge(
        l2_test_firing,
        on=["l2_width", "objective", "seed"],
        how="left",
    )

    diagnostics["raw_l2_minus_l1_fixed250_probe_ba"] = (
        diagnostics["l2_fixed250_raw_probe_ba"]
        - diagnostics["l1_fixed250_raw_probe_ba"]
    )
    diagnostics["pca128_l2_minus_l1_fixed250_probe_ba"] = (
        diagnostics["l2_fixed250_pca128_probe_ba"]
        - diagnostics["l1_fixed250_pca128_probe_ba"]
    )
    diagnostics["l2_raw_minus_pca128_fixed250_probe_ba"] = (
        diagnostics["l2_fixed250_raw_probe_ba"]
        - diagnostics["l2_fixed250_pca128_probe_ba"]
    )

    diagnostic_summary = aggregate_mean_sd(
        diagnostics,
        ["objective", "l2_width"],
        [
            "test_balanced_accuracy",
            "train_test_ba_gap",
            "head_feature_dim",
            "head_parameter_count",
            "l1_fixed250_raw_probe_ba",
            "l2_fixed250_raw_probe_ba",
            "raw_l2_minus_l1_fixed250_probe_ba",
            "l1_fixed250_pca128_probe_ba",
            "l2_fixed250_pca128_probe_ba",
            "pca128_l2_minus_l1_fixed250_probe_ba",
            "l2_raw_minus_pca128_fixed250_probe_ba",
            "l2_pca128_explained_variance_ratio_sum",
            "l2_test_firing_rate",
            "l2_test_total_spikes_per_timestep",
        ],
    )

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "experiment_3_0_4_configurations.csv": pd.DataFrame(configuration_rows()),
        "experiment_3_0_4_native_results.csv": native,
        "experiment_3_0_4_history.csv": history,
        "experiment_3_0_4_layer_probes.csv": layer_probes,
        "experiment_3_0_4_tau_subgroup_probes.csv": subgroup_probes,
        "experiment_3_0_4_firing_rates.csv": firing,
        "experiment_3_0_4_diagnostics.csv": diagnostics,
        "experiment_3_0_4_native_summary.csv": native_summary,
        "experiment_3_0_4_layer_probe_summary.csv": layer_probe_summary,
        "experiment_3_0_4_tau_subgroup_probe_summary.csv": subgroup_probe_summary,
        "experiment_3_0_4_firing_rate_summary.csv": firing_summary,
        "experiment_3_0_4_diagnostic_summary.csv": diagnostic_summary,
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
