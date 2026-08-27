from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
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
    EXPECTED_RUNS,
    EXPERIMENT_ID,
    OBJECTIVES,
    SEEDS,
    SHIFTS,
    aggregate,
)


def checkpoint_path(results_dir: Path, shift: int, objective: str, seed: int) -> Path:
    return results_dir / "checkpoints" / f"shift{shift}__{objective}__seed{seed}.pt"


def main() -> None:
    results_dir = REPO_ROOT / "notebooks" / "artifacts" / EXPERIMENT_ID
    rows: list[dict] = []
    history_rows: list[dict] = []
    missing: list[str] = []

    for shift in SHIFTS:
        for objective in OBJECTIVES:
            for seed in SEEDS:
                path = checkpoint_path(results_dir, shift, objective, seed)
                if not path.exists():
                    missing.append(str(path.relative_to(REPO_ROOT)))
                    continue

                payload = torch.load(path, map_location="cpu", weights_only=False)
                result = payload.get("result")
                if not isinstance(result, dict):
                    raise ValueError(f"Checkpoint has no result dict: {path}")

                expected = (shift, objective, seed)
                actual = (
                    int(result.get("shift")),
                    str(result.get("objective")),
                    int(result.get("seed")),
                )
                if actual != expected:
                    raise ValueError(
                        f"Checkpoint identity mismatch for {path}: {actual} != {expected}"
                    )

                history = result.get("history", [])
                row = {key: value for key, value in result.items() if key != "history"}
                rows.append(row)
                for epoch_row in history:
                    history_rows.append(
                        {
                            "shift": shift,
                            "objective": objective,
                            "seed": seed,
                            **epoch_row,
                        }
                    )

    if missing:
        print(f"Completed checkpoints: {len(rows)}/{EXPECTED_RUNS}")
        print("Missing checkpoints:")
        for item in missing:
            print(" -", item)
        raise SystemExit(2)

    results = pd.DataFrame(rows).sort_values(["objective", "shift", "seed"])
    if len(results) != EXPECTED_RUNS:
        raise RuntimeError((len(results), EXPECTED_RUNS))

    history = pd.DataFrame(history_rows).sort_values(
        ["objective", "shift", "seed", "epoch"]
    )
    summary = aggregate(results)

    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "experiment_3_0_1_results.csv"
    history_path = results_dir / "experiment_3_0_1_history.csv"
    summary_path = results_dir / "experiment_3_0_1_summary.csv"
    results.to_csv(results_path, index=False)
    history.to_csv(history_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Completed checkpoints: {len(results)}/{EXPECTED_RUNS}")
    print("Wrote:", results_path.relative_to(REPO_ROOT))
    print("Wrote:", history_path.relative_to(REPO_ROOT))
    print("Wrote:", summary_path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
