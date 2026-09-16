#!/usr/bin/env bash
#SBATCH --job-name=exp7_3_1_c4
#SBATCH --partition=cpu
#SBATCH --array=0-59%50
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp7_3_1_c4_%A_%a.out
#SBATCH --error=logs/exp7_3_1_c4_%A_%a.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

module load conda/latest
eval "$(conda shell.bash hook)"
if ! conda activate writingring-gpu; then
  conda activate writingring-viz
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m scripts.experiment_7_3_1_frozen_l2_linear_optimization \
  --repo-root "$REPO_ROOT" --device cpu --threads 1 \
  c4 --array-task-id "$SLURM_ARRAY_TASK_ID"
