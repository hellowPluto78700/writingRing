#!/usr/bin/env bash
#SBATCH --job-name=exp7_3_cache
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=10:00:00
#SBATCH --output=exp7_3_cache_%A_%a.out
#SBATCH --error=exp7_3_cache_%A_%a.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
module load conda/latest
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx writingring-gpu; then conda activate writingring-gpu; else conda activate writingring-viz; fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u -m scripts.experiment_7_3_training_strategy_decomposition \
  --device cpu --threads 1 prepare-stage2-cache --array-task-id "${SLURM_ARRAY_TASK_ID:?}"
