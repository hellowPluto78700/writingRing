#!/usr/bin/env bash
#SBATCH --job-name=wr139-array
#SBATCH --partition=cpu
#SBATCH --array=0-23%18
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=unity_exp139_array_%A_%a.out
#SBATCH --error=unity_exp139_array_%A_%a.err

set -euo pipefail

CONDITIONS=(con250 con500)
LAMBDAS=(0 0.01 0.03 0.1)
SEEDS=(11 23 101)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= 24 )); then
  echo "Invalid array task id: $TASK_ID" >&2
  exit 2
fi

SEED_IDX=$(( TASK_ID % 3 ))
LAMBDA_IDX=$(( (TASK_ID / 3) % 4 ))
CONDITION_IDX=$(( TASK_ID / 12 ))

SEED="${SEEDS[$SEED_IDX]}"
LAMBDA="${LAMBDAS[$LAMBDA_IDX]}"
CONDITION="${CONDITIONS[$CONDITION_IDX]}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu

# Keep each independent run single-threaded. The benchmark for this experiment
# showed that task-level CPU parallelism is more effective than one GPU running
# seeds sequentially.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

printf 'Node: %s\n' "$(hostname)"
printf 'Array job/task: %s/%s\n' "${SLURM_ARRAY_JOB_ID:-none}" "$TASK_ID"
printf 'Run: condition=%s lambda=%s seed=%s\n' "$CONDITION" "$LAMBDA" "$SEED"

python -u scripts/experiment_1_3_9_phase_aware_contrastive/03_train_one_run.py \
  --device cpu \
  --condition "$CONDITION" \
  --lambda-con "$LAMBDA" \
  --seed "$SEED"
