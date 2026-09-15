#!/usr/bin/env bash
#SBATCH --job-name=exp7_2_6_2_bias
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=exp7_2_6_2_bias_%A_%a.out
#SBATCH --error=exp7_2_6_2_bias_%A_%a.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
module load conda/latest
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx writingring-gpu; then
    conda activate writingring-gpu
else
    conda activate writingring-viz
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
python -u -m scripts.experiment_7_2_6_2_mean_ce_bias \
  --device cpu --threads 1 run-bias --array-task-id "$TASK_ID"
