from __future__ import annotations

import argparse
from pathlib import Path

import torch


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "snn").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate writingRing repository root")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train exactly one Experiment 1.3.9 development run."
    )
    parser.add_argument("--condition", choices=("con250", "con500"), required=True)
    parser.add_argument("--lambda-con", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore an existing matching checkpoint and train again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = find_repo_root()
    script_dir = repo / "scripts/experiment_1_3_9_phase_aware_contrastive"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    namespace: dict[str, object] = {"__name__": "__main__"}
    setup_path = script_dir / "01_setup.py"
    model_path = script_dir / "02_model_training.py"
    exec(compile(setup_path.read_text(), str(setup_path), "exec"), namespace)

    namespace["DEVICE"] = torch.device(args.device)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        namespace["NUM_EPOCHS"] = int(args.epochs)
    namespace["RESUME_EXISTING"] = not args.force_retrain
    namespace["SAVE_CHECKPOINTS"] = True

    exec(compile(model_path.read_text(), str(model_path), "exec"), namespace)

    lambda_grid = tuple(float(v) for v in namespace["LAMBDA_CON_GRID"])
    if not any(abs(args.lambda_con - v) <= 1e-12 for v in lambda_grid):
        raise ValueError(
            f"lambda={args.lambda_con:g} is not in configured LAMBDA_CON_GRID={lambda_grid}"
        )

    seeds = tuple(int(v) for v in namespace["SEEDS"])
    if args.seed not in seeds:
        raise ValueError(f"seed={args.seed} is not in configured SEEDS={seeds}")

    checkpoint_dir = Path(namespace["CHECKPOINT_DIR"])
    checkpoint_path = checkpoint_dir / (
        f"dev_{args.condition}_lambda_{args.lambda_con:g}_seed_{args.seed}.pt"
    )

    print(
        "Single development run:",
        f"condition={args.condition}",
        f"lambda={args.lambda_con:g}",
        f"seed={args.seed}",
        f"epochs={namespace['NUM_EPOCHS']}",
        f"device={namespace['DEVICE']}",
    )
    print("Checkpoint:", checkpoint_path)

    payload = namespace["train_run"](
        args.condition,
        args.seed,
        args.lambda_con,
        checkpoint_path,
    )
    print(
        "Completed:",
        f"best_epoch={payload['best_epoch']}",
        f"best_val_BA={payload['val_best']['balanced_accuracy']:.6f}",
    )


if __name__ == "__main__":
    main()
