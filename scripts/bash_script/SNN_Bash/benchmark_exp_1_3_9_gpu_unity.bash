#!/usr/bin/env bash
#SBATCH --job-name=wr139-gpu-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=unity_gpu_benchmark_%j.out
#SBATCH --error=unity_gpu_benchmark_%j.err

set -euo pipefail

# Unity benchmark for experiment 1.3.9.
# One GPU runs seeds 11, 23, and 101 sequentially so that the measured wall time
# can be compared with the three-way CPU array benchmark.
#
# Submit from anywhere inside the repository:
#   sbatch scripts/bash_script/SNN_Bash/benchmark_exp_1_3_9_gpu_unity.bash
#
# Optional overrides:
#   sbatch --export=ALL,WR_BENCH_EPOCHS=40,WR_BENCH_CONDITION=con500,WR_BENCH_LAMBDA=0.1 \
#     scripts/bash_script/SNN_Bash/benchmark_exp_1_3_9_gpu_unity.bash
#
# The default 40 epochs intentionally cover:
#   epochs 1-25  : CE-only warm-up
#   epochs 26-35 : contrastive ramp
#   epochs 36-40 : full contrastive objective

JOB_START_EPOCH="$(date +%s)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WRITINGRING_REPO_ROOT="$REPO_ROOT"
export WR_BENCH_DEVICE="cuda"
export WR_BENCH_SEEDS="${WR_BENCH_SEEDS:-11,23,101}"
export WR_BENCH_EPOCHS="${WR_BENCH_EPOCHS:-40}"
export WR_BENCH_CONDITION="${WR_BENCH_CONDITION:-con500}"
export WR_BENCH_LAMBDA="${WR_BENCH_LAMBDA:-0.1}"
export WR_BENCH_JOB_START_EPOCH="$JOB_START_EPOCH"

RESULT_DIR="$REPO_ROOT/notebooks/artifacts/experiment_1_3_9_phase_aware_cross_user_contrastive_local_features/benchmark_unity/gpu"
mkdir -p "$RESULT_DIR"
export WR_BENCH_RESULT_DIR="$RESULT_DIR"

printf 'Node: %s\n' "$(hostname)"
printf 'Job: %s\n' "${SLURM_JOB_ID:-none}"
printf 'Config: condition=%s lambda=%s epochs=%s seeds=%s\n' \
  "$WR_BENCH_CONDITION" "$WR_BENCH_LAMBDA" "$WR_BENCH_EPOCHS" "$WR_BENCH_SEEDS"
nvidia-smi

python - <<'PY'
from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
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
seeds = tuple(int(x) for x in os.environ["WR_BENCH_SEEDS"].split(",") if x.strip())
job_id = os.environ.get("SLURM_JOB_ID", "interactive")
job_start_epoch = float(os.environ.get("WR_BENCH_JOB_START_EPOCH", time.time()))

if not torch.cuda.is_available():
    raise RuntimeError("GPU benchmark requested, but torch.cuda.is_available() is False")

host_threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
torch.set_num_threads(host_threads)
try:
    torch.set_num_interop_threads(min(2, host_threads))
except RuntimeError:
    pass

namespace: dict[str, object] = {"__name__": "__main__"}
setup_t0 = time.perf_counter()
setup_path = repo / "scripts/experiment_1_3_9_phase_aware_contrastive/01_setup.py"
model_path = repo / "scripts/experiment_1_3_9_phase_aware_contrastive/02_model_training.py"
exec(compile(setup_path.read_text(), str(setup_path), "exec"), namespace)

# Benchmark-specific overrides. Do not resume or save normal experiment checkpoints.
namespace["DEVICE"] = torch.device("cuda")
namespace["NUM_EPOCHS"] = epochs
namespace["RESUME_EXISTING"] = False
namespace["SAVE_CHECKPOINTS"] = False
namespace["RUN_VARIANT"] = "benchmark_unity_gpu"
exec(compile(model_path.read_text(), str(model_path), "exec"), namespace)
setup_seconds = time.perf_counter() - setup_t0

original_run_epoch = namespace["run_epoch"]
epoch_times: dict[int, dict[str, float]] = {}

def sync_device() -> None:
    torch.cuda.synchronize()

def timed_run_epoch(model, loader, condition_arg, lambda_arg, optimizer=None, epoch=None):
    training = optimizer is not None
    sync_device()
    t0 = time.perf_counter()
    out = original_run_epoch(model, loader, condition_arg, lambda_arg, optimizer, epoch)
    sync_device()
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

summary_rows: list[dict[str, object]] = []
epoch_rows: list[dict[str, object]] = []
all_seed_t0 = time.perf_counter()

for seed in seeds:
    epoch_times.clear()
    sync_device()
    seed_t0 = time.perf_counter()
    payload = train_run(
        condition,
        seed,
        lambda_con,
        result_dir / f"unused_gpu_seed_{seed}.pt",
    )
    sync_device()
    seed_wall_seconds = time.perf_counter() - seed_t0

    local_rows = []
    for epoch in range(1, epochs + 1):
        rec = epoch_times.get(epoch, {})
        train_s = float(rec.get("train_seconds", float("nan")))
        val_s = float(rec.get("val_seconds", float("nan")))
        epoch_s = train_s + val_s
        row = {
            "platform": "gpu",
            "job_id": job_id,
            "node": socket.gethostname(),
            "seed": seed,
            "condition": condition,
            "lambda_con": lambda_con,
            "epoch": epoch,
            "phase": phase_for_epoch(epoch),
            "train_seconds": train_s,
            "val_seconds": val_s,
            "epoch_seconds": epoch_s,
        }
        local_rows.append(row)
        epoch_rows.append(row)

    valid_epoch_seconds = [r["epoch_seconds"] for r in local_rows if np.isfinite(r["epoch_seconds"])]
    by_phase = {}
    for phase in ("warmup", "ramp", "full_contrastive"):
        vals = [r["epoch_seconds"] for r in local_rows if r["phase"] == phase and np.isfinite(r["epoch_seconds"])]
        by_phase[phase] = float(np.mean(vals)) if vals else float("nan")

    summary_rows.append({
        "platform": "gpu",
        "job_id": job_id,
        "node": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "seed": seed,
        "condition": condition,
        "lambda_con": lambda_con,
        "epochs": epochs,
        "setup_seconds_shared": setup_seconds,
        "seed_wall_seconds": seed_wall_seconds,
        "mean_epoch_seconds": float(np.mean(valid_epoch_seconds)),
        "warmup_mean_epoch_seconds": by_phase["warmup"],
        "ramp_mean_epoch_seconds": by_phase["ramp"],
        "full_contrastive_mean_epoch_seconds": by_phase["full_contrastive"],
        "best_epoch": int(payload["best_epoch"]),
        "best_val_ba": float(payload["best_val_ba"]),
    })

all_seed_wall_seconds = time.perf_counter() - all_seed_t0
job_end_epoch = time.time()

# Add the observed sequential three-seed wall time to every summary row for easy comparison.
for row in summary_rows:
    row["all_seeds_sequential_wall_seconds"] = all_seed_wall_seconds
    row["job_start_epoch"] = job_start_epoch
    row["job_end_epoch"] = job_end_epoch

stamp = f"job{job_id}"
epoch_csv = result_dir / f"gpu_epoch_times_{stamp}.csv"
summary_csv = result_dir / f"gpu_summary_{stamp}.csv"
metadata_json = result_dir / f"gpu_metadata_{stamp}.json"

with epoch_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
    writer.writeheader()
    writer.writerows(epoch_rows)

with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

try:
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
except Exception as exc:
    smi = f"unavailable: {exc}"

metadata = {
    "platform": "gpu",
    "job_id": job_id,
    "node": socket.gethostname(),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "condition": condition,
    "lambda_con": lambda_con,
    "epochs": epochs,
    "seeds": seeds,
    "setup_seconds_shared": setup_seconds,
    "all_seeds_sequential_wall_seconds": all_seed_wall_seconds,
    "job_start_epoch": job_start_epoch,
    "job_end_epoch": job_end_epoch,
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_compute_capability": torch.cuda.get_device_capability(0),
    "nvidia_smi": smi,
    "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
}
metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")

print("\nGPU benchmark complete")
print(f"setup seconds: {setup_seconds:.3f}")
print(f"all seeds sequential wall seconds: {all_seed_wall_seconds:.3f}")
for row in summary_rows:
    print(
        f"seed={row['seed']} wall={row['seed_wall_seconds']:.3f}s "
        f"mean_epoch={row['mean_epoch_seconds']:.3f}s "
        f"warmup={row['warmup_mean_epoch_seconds']:.3f}s "
        f"ramp={row['ramp_mean_epoch_seconds']:.3f}s "
        f"full={row['full_contrastive_mean_epoch_seconds']:.3f}s"
    )
print("epoch timing:", epoch_csv)
print("summary:", summary_csv)
print("metadata:", metadata_json)
PY

JOB_END_EPOCH="$(date +%s)"
TOTAL_SHELL_SECONDS="$((JOB_END_EPOCH - JOB_START_EPOCH))"
printf 'slurm_job_wall_seconds=%s\n' "$TOTAL_SHELL_SECONDS" | tee "$RESULT_DIR/gpu_slurm_wall_job${SLURM_JOB_ID:-interactive}.txt"
