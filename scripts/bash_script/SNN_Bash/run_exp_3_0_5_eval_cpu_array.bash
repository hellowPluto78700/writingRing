#!/usr/bin/env bash
#SBATCH --job-name=wr305-eval
#SBATCH --partition=cpu
#SBATCH --array=0-8%50
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=unity_exp305_eval_%A_%a.out
#SBATCH --error=unity_exp305_eval_%A_%a.err

set -euo pipefail
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= 9 )); then echo "Invalid array task id: $TASK_ID" >&2; exit 2; fi
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u -m scripts.experiment_3_0_5_frozen_representation_accessibility run-one --array-task-id "$TASK_ID" --device cpu
