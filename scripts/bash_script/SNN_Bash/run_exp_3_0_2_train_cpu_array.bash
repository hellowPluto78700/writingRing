#!/usr/bin/env bash
#SBATCH --job-name=wr302-train-eval
#SBATCH --partition=cpu
#SBATCH --array=0-35%50
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --output=unity_exp302_train_eval_%A_%a.out
#SBATCH --error=unity_exp302_train_eval_%A_%a.err

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= 36 )); then
  echo "Invalid array task id: $TASK_ID" >&2
  exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

printf 'Node: %s\n' "$(hostname)"
printf 'Train+eval array job/task: %s/%s\n' "${SLURM_ARRAY_JOB_ID:-none}" "$TASK_ID"
printf 'CPUs per task: %s\n' "${SLURM_CPUS_PER_TASK:-1}"

python -u scripts/experiment_3_0_2/01_train_one_run.py \
  --array-task-id "$TASK_ID" \
  --device cpu
