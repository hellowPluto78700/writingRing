from __future__ import annotations

import argparse
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
from scripts.experiment_3_0_4_l2_width_representation_capacity import (  # noqa: E402
    BASELINE_WIDTH,
    BATCH_SIZE,
    Config,
    OBJECTIVES,
    PROTOCOL_VERSION,
    SEEDS,
    results_dir,
    run_evaluation_one,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the reused Experiment 3.0.3-B W128 baseline."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data = prepare_data(REPO_ROOT)
    config = Config(
        repo_root=REPO_ROOT,
        results_dir=results_dir(REPO_ROOT),
        device=args.device,
        batch_size=int(args.batch_size),
        resume=not args.force,
        threads=1,
    )

    completed = 0
    for objective in OBJECTIVES:
        for seed in SEEDS:
            result = run_evaluation_one(
                BASELINE_WIDTH,
                objective,
                seed,
                data,
                config,
            )
            completed += 1
            print(
                f"[{completed}/9] W{BASELINE_WIDTH} {objective} seed={seed}:",
                f"layer_probes={len(result['layer_probes'])}",
            )

    print("Protocol:", PROTOCOL_VERSION)
    print("Baseline evaluations completed:", completed)


if __name__ == "__main__":
    main()
