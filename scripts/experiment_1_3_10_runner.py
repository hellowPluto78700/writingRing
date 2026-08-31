from __future__ import annotations

"""Experiment 1.3.10 entry point with the full architecture sweep.

The core implementation lives in experiment_1_3_10_stacked_bin_snn_ablation.py.
This runner defines the experiment-level architecture grid so the Slurm array,
finalizer, and notebook all use the same 60-run factorial design.
"""

import experiment_1_3_10_stacked_bin_snn_ablation as experiment


ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "1h128": (128,),
    "1h256": (256,),
    "1h512": (512,),
    "1h1024": (1024,),
    "2h128": (128, 128),
}

experiment.ARCHITECTURES = ARCHITECTURES
experiment.EXPECTED_RUNS = (
    len(experiment.SPLIT_SEEDS)
    * len(ARCHITECTURES)
    * len(experiment.OBJECTIVES)
    * len(experiment.TRAIN_REGIMES)
)


if __name__ == "__main__":
    experiment.main()
