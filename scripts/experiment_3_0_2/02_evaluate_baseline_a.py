from __future__ import annotations

from pathlib import Path
import sys

import torch


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "snn").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate writingRing repository root")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_3_0_1_single_tau_objectives import prepare_data  # noqa: E402
from scripts.experiment_3_0_2_hidden_multitau_architectures import (  # noqa: E402
    BATCH_SIZE,
    Config,
    OBJECTIVES,
    PROTOCOL_VERSION,
    SEEDS,
    results_dir,
    run_evaluation_one,
)


def main() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    root = results_dir(REPO_ROOT)
    config = Config(
        repo_root=REPO_ROOT,
        results_dir=root,
        device="cpu",
        batch_size=BATCH_SIZE,
        resume=True,
        threads=1,
    )
    data = prepare_data(REPO_ROOT)

    print("Protocol:", PROTOCOL_VERSION)
    print("Baseline A: reuse Experiment 3.0.1 shift=3 checkpoints")
    completed = 0
    for objective in OBJECTIVES:
        for seed in SEEDS:
            print("Evaluating:", "A", objective, seed)
            payload = run_evaluation_one("A", objective, seed, data, config)
            completed += 1
            print(
                "  completed:",
                f"layer_probes={len(payload['layer_probes'])}",
                f"subgroup_probes={len(payload['subgroup_probes'])}",
                f"firing_rows={len(payload['firing_rates'])}",
            )

    print(f"Completed baseline evaluations: {completed}/{len(OBJECTIVES) * len(SEEDS)}")


if __name__ == "__main__":
    main()
