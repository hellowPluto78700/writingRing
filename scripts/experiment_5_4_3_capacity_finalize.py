from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import experiment_5_4_3_elapsed_readout_capacity as exp543


def finalize_capacity(repo_root: Path) -> dict[str, Path]:
    """Finalize Stage A without blocking the preregistered Stage-B diagnostic.

    The main Exp5.4.3 finalizer handles the normal case where at least one
    Stage-A capacity improves frozen WHAT. If every K/r configuration fails
    that minimal support criterion, this wrapper still records the complete
    validation map and forces the direct-full-matrix Stage-B diagnostic.
    Test remains unopened in either path.
    """
    root = exp543.results_dir(repo_root)
    specs = exp543.capacity_specs()
    if len(specs) != 45:
        raise RuntimeError("Stage-A mapping must contain 45 runs")

    payloads = {spec.key: exp543._load_eval(root, spec) for spec in specs}
    rows, summaries = exp543._summary_from_payloads(specs, payloads)
    eligible = [item for item in summaries if bool(item["eligible"])]
    if eligible:
        return exp543.finalize_capacity(repo_root)

    references = exp543._reference_rows(root)
    preliminary = max(
        summaries,
        key=lambda item: (
            float(item["mean_val_balanced_accuracy"]),
            -int(item["trainable_parameter_count"]),
        ),
    )

    outputs = {
        "reference_runs": root / "reference_runs.csv",
        "capacity_runs": root / "capacity_runs.csv",
        "capacity_paired_deltas": root / "capacity_paired_deltas.csv",
        "capacity_recovery": root / "capacity_recovery.csv",
        "capacity_summary": root / "capacity_summary.csv",
        "capacity_selection": exp543.capacity_selection_path(root),
    }
    pd.DataFrame(references).to_csv(outputs["reference_runs"], index=False)
    pd.DataFrame(rows).to_csv(outputs["capacity_runs"], index=False)
    pd.DataFrame(
        [
            {
                key: row[key]
                for key in (
                    "seed",
                    "parameterization",
                    "n_banks",
                    "rank",
                    "delta_val_ba_vs_base",
                    "gap_to_fixed250_val_ba",
                )
            }
            for row in rows
        ]
    ).to_csv(outputs["capacity_paired_deltas"], index=False)
    pd.DataFrame(
        [
            {
                key: row[key]
                for key in (
                    "seed",
                    "parameterization",
                    "n_banks",
                    "rank",
                    "oracle_gap_recovery_fraction",
                )
            }
            for row in rows
        ]
    ).to_csv(outputs["capacity_recovery"], index=False)
    pd.DataFrame(summaries).to_csv(outputs["capacity_summary"], index=False)

    exp543._save_json(
        outputs["capacity_selection"],
        {
            "experiment_id": exp543.EXPERIMENT_ID,
            "protocol_version": exp543.PROTOCOL_VERSION,
            "stage": "capacity",
            "status": "extension_required",
            "preliminary_selection": preliminary,
            "preliminary_selection_supported": False,
            "extension_required": True,
            "extension_mode": "direct_matrix",
            "extension_reason": (
                "No Stage-A K/r configuration had positive mean paired validation "
                "BA with at least 4/5 nonnegative seeds; run the direct full-matrix "
                "diagnostic before opening test."
            ),
            "capacity_summary": summaries,
            "strong_recovery_definition": {
                "mean_recovery_fraction_at_least": exp543.STRONG_RECOVERY_FRACTION,
                "mean_gap_to_fixed250_val_ba_at_most": exp543.STRONG_FIXED_GAP,
            },
            "only_validation_selected": True,
            "test_not_used_for_selection": True,
        },
    )
    return outputs


def main() -> None:
    repo_root = exp543.find_repo_root()
    outputs = finalize_capacity(repo_root)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
