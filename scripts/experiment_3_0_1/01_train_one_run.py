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

from scripts.experiment_3_0_1_single_tau_objectives import (  # noqa: E402
    Config,
    EPOCHS,
    EXPECTED_RUNS,
    EXPERIMENT_ID,
    OBJECTIVES,
    PROTOCOL_VERSION,
    SEEDS,
    SHIFTS,
    prepare_data,
    protocol_results_dir,
    run_one,
)


def run_specs() -> list[tuple[int, str, int]]:
    return [
        (shift, objective, seed)
        for shift in SHIFTS
        for objective in OBJECTIVES
        for seed in SEEDS
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train exactly one Experiment 3.0.1 run."
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--array-task-id",
        type=int,
        help=f"Map Slurm array index 0..{EXPECTED_RUNS - 1} to one run.",
    )
    selector.add_argument(
        "--shift",
        type=int,
        choices=SHIFTS,
        help="Explicit homogeneous shift_syn for L1/L2/L3.",
    )
    parser.add_argument("--objective", choices=OBJECTIVES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore an existing matching checkpoint for this protocol.",
    )
    return parser.parse_args()


def resolve_run(args: argparse.Namespace) -> tuple[int, str, int, int | None]:
    specs = run_specs()
    if len(specs) != EXPECTED_RUNS:
        raise RuntimeError((len(specs), EXPECTED_RUNS))

    if args.array_task_id is not None:
        task_id = int(args.array_task_id)
        if task_id < 0 or task_id >= len(specs):
            raise ValueError(
                f"array task id {task_id} outside valid range 0..{len(specs) - 1}"
            )
        if args.objective is not None or args.seed is not None:
            raise ValueError(
                "Do not pass --objective/--seed together with --array-task-id."
            )
        shift, objective, seed = specs[task_id]
        return shift, objective, seed, task_id

    if args.objective is None or args.seed is None:
        raise ValueError(
            "Explicit --shift mode also requires --objective and --seed."
        )
    return int(args.shift), str(args.objective), int(args.seed), None


def main() -> None:
    args = parse_args()
    shift, objective, seed, task_id = resolve_run(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    results_dir = protocol_results_dir(REPO_ROOT)
    config = Config(
        repo_root=REPO_ROOT,
        results_dir=results_dir,
        device=args.device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        resume=not args.force_retrain,
        threads=1,
    )

    print("Repository root:", REPO_ROOT)
    print("Experiment:", EXPERIMENT_ID)
    print("Protocol:", PROTOCOL_VERSION)
    print("Results dir:", results_dir.relative_to(REPO_ROOT))
    if task_id is not None:
        print(f"Array task: {task_id}/{EXPECTED_RUNS - 1}")
    print(
        "Run:",
        f"shift={shift}",
        f"objective={objective}",
        f"seed={seed}",
        f"epochs={args.epochs}",
        f"batch_size={args.batch_size}",
        f"device={args.device}",
    )

    data = prepare_data(REPO_ROOT)
    print("Split:", data.split)
    result = run_one(shift, objective, seed, data, config)

    print(
        "Completed:",
        f"best_epoch={result['best_epoch']}",
        f"val_BA={result['val_balanced_accuracy']:.6f}",
        f"test_BA={result['test_balanced_accuracy']:.6f}",
        f"probe_test_BA={result['probe_test_balanced_accuracy']:.6f}",
    )


if __name__ == "__main__":
    main()
