from __future__ import annotations

from pathlib import Path

import torch


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "snn").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate writingRing repository root")


def main() -> None:
    repo = find_repo_root()
    script_dir = repo / "scripts/experiment_1_3_9_phase_aware_contrastive"
    namespace: dict[str, object] = {"__name__": "__main__"}

    setup_path = script_dir / "01_setup.py"
    model_path = script_dir / "02_model_training.py"
    aggregate_path = script_dir / "03_run_experiment.py"

    exec(compile(setup_path.read_text(), str(setup_path), "exec"), namespace)
    namespace["DEVICE"] = torch.device("cpu")
    namespace["RESUME_EXISTING"] = True
    namespace["SAVE_CHECKPOINTS"] = True
    exec(compile(model_path.read_text(), str(model_path), "exec"), namespace)

    checkpoint_dir = Path(namespace["CHECKPOINT_DIR"])
    conditions = ("con250", "con500")
    lambdas = tuple(float(v) for v in namespace["LAMBDA_CON_GRID"])
    seeds = tuple(int(v) for v in namespace["SEEDS"])

    missing: list[Path] = []
    for condition in conditions:
        for lambda_con in lambdas:
            for seed in seeds:
                ckpt = checkpoint_dir / (
                    f"dev_{condition}_lambda_{lambda_con:g}_seed_{seed}.pt"
                )
                if not ckpt.exists():
                    missing.append(ckpt)

    if missing:
        names = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Experiment 1.3.9 finalization requires all 24 development "
            "checkpoints. Missing checkpoints:\n" + names +
            "\nRe-run the failed/missing Slurm array tasks before finalizing."
        )

    print(f"Verified all {len(conditions) * len(lambdas) * len(seeds)} development checkpoints.")
    print("Finalization is load/evaluate-only; no development run may be trained here.")

    # The existing aggregation script calls train_run for each development point.
    # Because every checkpoint was verified above and RESUME_EXISTING=True,
    # train_run only validates and loads those checkpoints; it cannot fall back
    # to training a missing run in this workflow.
    exec(compile(aggregate_path.read_text(), str(aggregate_path), "exec"), namespace)


if __name__ == "__main__":
    main()
