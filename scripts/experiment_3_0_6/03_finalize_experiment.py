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

from scripts import experiment_3_0_6_causal_temporal_decoding as exp  # noqa: E402


def main() -> None:
    outputs = exp.finalize_experiment(REPO_ROOT)

    # Enforce a strictly validation-only decoder choice. The core finalizer
    # writes all aggregate artifacts first; this wrapper owns the durable
    # selection contract consumed by Experiment 3.0.7.
    runs = pd.read_csv(outputs["runs"])
    causal = runs[runs["decoder"].isin(exp.DECODERS)].copy()
    selection = (
        causal.groupby("decoder", sort=True)
        .agg(
            mean_val_early_recognition_auc=("val_early_recognition_auc", "mean"),
        )
        .reset_index()
        .sort_values(
            ["mean_val_early_recognition_auc", "decoder"],
            ascending=[False, True],
            ignore_index=True,
        )
    )
    if selection.empty:
        raise RuntimeError("No causal decoder available for validation-only selection")
    winner = str(selection.iloc[0]["decoder"])
    selection_path = exp.results_dir(REPO_ROOT) / "selected_causal_decoder.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_id": exp.EXPERIMENT_ID,
                "protocol_version": exp.PROTOCOL_VERSION,
                "selected_decoder": winner,
                "source_objective": exp.selected_objective(REPO_ROOT),
                "selection_metric": "mean validation Early Recognition AUC",
                "tie_breaker": "decoder name only; test metrics are never used for selection",
                "seeds": exp.SEEDS,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    selection_csv = exp.results_dir(REPO_ROOT) / "experiment_3_0_6_validation_selection.csv"
    selection.to_csv(selection_csv, index=False)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"validation_selection: {selection_csv}")
    print(f"selected_decoder: {selection_path}")


if __name__ == "__main__":
    main()
