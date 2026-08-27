#!/usr/bin/env bash
#SBATCH --job-name=wr139-cpu-bench
#SBATCH --partition=cpu
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=unity_cpu_benchmark_%A_%a.out
#SBATCH --error=unity_cpu_benchmark_%A_%a.err

set -euo pipefail

# Unity benchmark for experiment 1.3.9.
# Three Slurm array tasks run seeds 11, 23, and 101 independently. With enough
# free CPU nodes the seeds execute concurrently, which is the comparison target
# against the one-GPU sequential benchmark.
#
# Submit from anywhere inside the repository:
#   sbatch scripts/bash_script/SNN_Bash/benchmark_exp_1_3_9_cpu_unity.bash
#
# Optional overrides:
#   sbatch --export=ALL,WR_BENCH_EPOCHS=40,WR_BENCH_CONDITION=con500,WR_BENCH_LAMBDA=0.1 \
#     scripts/bash_script/SNN_Bash/benchmark_exp_1_3_9_cpu_unity.bash
#
# The default is intentionally one CPU core per seed. Small 30->128->128->64
# matrix operations often do not benefit from many BLAS threads. If this is
# competitive, a second benchmark can test --cpus-per-task=4 separately.

SEEDS=(11 23 101)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#SEEDS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID" >&2
  exit 2
fi
SEED="${SEEDS[$TASK_ID]}"

JOB_START_EPOCH="$(date +%s)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu

# Force the CPU benchmark to stay on CPU and prevent hidden BLAS oversubscription.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

export WRITINGRING_REPO_ROOT="$REPO_ROOT"
export WR_BENCH_DEVICE="cpu"
export WR_BENCH_SEED="$SEED"
export WR_BENCH_EPOCHS="${WR_BENCH_EPOCHS:-40}"
export WR_BENCH_CONDITION="${WR_BENCH_CONDITION:-con500}"
export WR_BENCH_LAMBDA="${WR_BENCH_LAMBDA:-0.1}"
export WR_BENCH_JOB_START_EPOCH="$JOB_START_EPOCH"

RESULT_DIR="$REPO_ROOT/notebooks/artifacts/experiment_1_3_9_phase_aware_cross_user_contrastive_local_features/benchmark_unity/cpu"
mkdir -p "$RESULT_DIR"
export WR_BENCH_RESULT_DIR="$RESULT_DIR"

printf 'Node: %s\n' "$(hostname)"
printf 'Array job/task: %s/%s\n' "${SLURM_ARRAY_JOB_ID:-none}" "$TASK_ID"
printf 'Seed: %s\n' "$SEED"
printf 'Config: condition=%s lambda=%s epochs=%s cpus=%s\n' \
  "$WR_BENCH_CONDITION" "$WR_BENCH_LAMBDA" "$WR_BENCH_EPOCHS" "${SLURM_CPUS_PER_TASK:-1}"

python - <<'PY'
from __future__ import annotations

import csv
import json
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

repo = Path(os.environ["WRITINGRING_REPO_ROOT"]).resolve()
result_dir = Path(os.environ["WR_BENCH_RESULT_DIR"]).resolve()
result_dir.mkdir(parents=True, exist_ok=True)

condition = os.environ["WR_BENCH_CONDITION"]
lambda_con = float(os.environ["WR_BENCH_LAMBDA"])
epochs = int(os.environ["WR_BENCH_EPOCHS"])
seed = int(os.environ["WR_BENCH_SEED"])
job_id = os.environ.get("SLURM_JOB_ID", "interactive")
array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", job_id)
array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
job_start_epoch = float(os.environ.get("WR_BENCH_JOB_START_EPOCH", time.time()))

cpu_threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
torch.set_num_threads(cpu_threads)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

namespace: dict[str, object] = {"__name__": "__main__"}
setup_t0 = time.perf_counter()
setup_path = repo / "scripts/experiment_1_3_9_phase_aware_contrastive/01_setup.py"
model_path = repo / "scripts/experiment_1_3_9_phase_aware_contrastive/02_model_training.py"
exec(compile(setup_path.read_text(), str(setup_path), "exec"), namespace)

# Benchmark-specific overrides. Do not resume or save normal experiment checkpoints.
namespace["DEVICE"] = torch.device("cpu")
namespace["NUM_EPOCHS"] = epochs
namespace["RESUME_EXISTING"] = False
namespace["SAVE_CHECKPOINTS"] = False
namespace["RUN_VARIANT"] = "benchmark_unity_cpu"
exec(compile(model_path.read_text(), str(model_path), "exec"), namespace)
setup_seconds = time.perf_counter() - setup_t0

original_run_epoch = namespace["run_epoch"]
epoch_times: dict[int, dict[str, float]] = {}

def timed_run_epoch(model, loader, condition_arg, lambda_arg, optimizer=None, epoch=None):
    training = optimizer is not None
    t0 = time.perf_counter()
    out = original_run_epoch(model, loader, condition_arg, lambda_arg, optimizer, epoch)
    elapsed = time.perf_counter() - t0

    # During train_run each epoch has one train call followed by one validation call.
    # Ignore the extra best-checkpoint train/val evaluation after the training loop.
    if epoch is not None and 1 <= int(epoch) <= epochs:
        rec = epoch_times.setdefault(int(epoch), {})
        if training:
            rec["train_seconds"] = elapsed
        elif "train_seconds" in rec and "val_seconds" not in rec:
            rec["val_seconds"] = elapsed
    return out

namespace["run_epoch"] = timed_run_epoch
train_run = namespace["train_run"]
warmup_epochs = int(namespace["CONTRASTIVE_WARMUP_EPOCHS"])
ramp_epochs = int(namespace["CONTRASTIVE_RAMP_EPOCHS"])

def phase_for_epoch(epoch: int) -> str:
    if epoch <= warmup_epochs:
        return "warmup"
    if epoch <= warmup_epochs + ramp_epochs:
        return "ramp"
    return "full_contrastive"

seed_t0 = time.perf_counter()
payload = train_run(
    condition,
    seed,
    lambda_con,
    result_dir / f"unused_cpu_seed_{seed}.pt",
)
seed_wall_seconds = time.perf_counter() - seed_t0
job_end_epoch = time.time()

epoch_rows: list[dict[str, object]] = []
for epoch in range(1, epochs + 1):
    rec = epoch_times.get(epoch, {})
    train_s = float(rec.get("train_seconds", float("nan")))
    val_s = float(rec.get("val_seconds", float("nan")))
    epoch_rows.append({
        "platform": "cpu",
        "array_job_id": array_job_id,
        "array_task_id": array_task_id,
        "job_id": job_id,
        "node": socket.gethostname(),
        "seed": seed,
        "condition": condition,
        "lambda_con": lambda_con,
        "cpu_threads": cpu_threads,
        "epoch": epoch,
        "phase": phase_for_epoch(epoch),
        "train_seconds": train_s,
        "val_seconds": val_s,
        "epoch_seconds": train_s + val_s,
    })

valid_epoch_seconds = [r["epoch_seconds"] for r in epoch_rows if np.isfinite(r["epoch_seconds"])]
phase_means = {}
for phase in ("warmup", "ramp", "full_contrastive"):
    vals = [r["epoch_seconds"] for r in epoch_rows if r["phase"] == phase and np.isfinite(r["epoch_seconds"])]
    phase_means[phase] = float(np.mean(vals)) if vals else float("nan")

summary = {
    "platform": "cpu",
    "array_job_id": array_job_id,
    "array_task_id": array_task_id,
    "job_id": job_id,
    "node": socket.gethostname(),
    "cpu_model": platform.processor(),
    "cpu_threads": cpu_threads,
    "torch_version": torch.__version__,
    "seed": seed,
    "condition": condition,
    "lambda_con": lambda_con,
    "epochs": epochs,
    "setup_seconds": setup_seconds,
    "seed_wall_seconds": seed_wall_seconds,
    "mean_epoch_seconds": float(np.mean(valid_epoch_seconds)),
    "warmup_mean_epoch_seconds": phase_means["warmup"],
    "ramp_mean_epoch_seconds": phase_means["ramp"],
    "full_contrastive_mean_epoch_seconds": phase_means["full_contrastive"],
    "best_epoch": int(payload["best_epoch"]),
    "best_val_ba": float(payload["best_val_ba"]),
    "job_start_epoch": job_start_epoch,
    "job_end_epoch": job_end_epoch,
}

stamp = f"array{array_job_id}_task{array_task_id}_seed{seed}"
epoch_csv = result_dir / f"cpu_epoch_times_{stamp}.csv"
summary_csv = result_dir / f"cpu_summary_{stamp}.csv"
metadata_json = result_dir / f"cpu_metadata_{stamp}.json"

with epoch_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
    writer.writeheader()
    writer.writerows(epoch_rows)

with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)

metadata = {
    **summary,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "platform_string": platform.platform(),
    "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
}
metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")

print("\nCPU benchmark complete")
print(f"seed={seed} setup={setup_seconds:.3f}s wall={seed_wall_seconds:.3f}s")
print(
    f"mean_epoch={summary['mean_epoch_seconds']:.3f}s "
    f"warmup={summary['warmup_mean_epoch_seconds']:.3f}s "
    f"ramp={summary['ramp_mean_epoch_seconds']:.3f}s "
    f"full={summary['full_contrastive_mean_epoch_seconds']:.3f}s"
)
print("epoch timing:", epoch_csv)
print("summary:", summary_csv)
print("metadata:", metadata_json)
PY

JOB_END_EPOCH="$(date +%s)"
TOTAL_SHELL_SECONDS="$((JOB_END_EPOCH - JOB_START_EPOCH))"
printf 'slurm_job_wall_seconds=%s\n' "$TOTAL_SHELL_SECONDS" | tee "$RESULT_DIR/cpu_slurm_wall_array${SLURM_ARRAY_JOB_ID:-interactive}_task${TASK_ID}_seed${SEED}.txt"
