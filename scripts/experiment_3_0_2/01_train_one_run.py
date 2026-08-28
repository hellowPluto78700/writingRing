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
from scripts.experiment_3_0_2_hidden_multitau_architectures import (  # noqa: E402
    BATCH_SIZE,
    Config,
    EPOCHS,
    EXPECTED_TRAIN_RUNS,
    OBJECTIVES,
    PROTOCOL_VERSION,
    SEEDS,
    TRAIN_ARCHITECTURES,
    architecture_shifts,
    results_dir,
    run_train_one,
    train_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Experiment 3.0.2 run.")
    parser.add_argument("--array-task-id", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = train_specs()
    if len(specs) != EXPECTED_TRAIN_RUNS:
        raise RuntimeError((len(specs), EXPECTED_TRAIN_RUNS))
    task_id = int(args.array_task_id)
    if task_id < 0 or task_id >= len(specs):
        raise ValueError(f"array task id {task_id} outside 0..{len(specs)-1}")
    architecture, objective, seed = specs[task_id]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    root = results_dir(REPO_ROOT)
    config = Config(
        repo_root=REPO_ROOT,
        results_dir=root,
        device=args.device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        resume=not args.force_retrain,
        threads=1,
    )
    data = prepare_data(REPO_ROOT)

    print("Protocol:", PROTOCOL_VERSION)
    print("Task:", task_id, "/", EXPECTED_TRAIN_RUNS - 1)
    print("Architecture:", architecture, architecture_shifts(architecture))
    print("Objective:", objective)
    print("Seed:", seed)
    result = run_train_one(architecture, objective, seed, data, config)
    print(
        "Completed:",
        f"best_epoch={result['best_epoch']}",
        f"val_BA={result['val_balanced_accuracy']:.6f}",
        f"test_BA={result['test_balanced_accuracy']:.6f}",
    )


if __name__ == "__main__":
    main()
