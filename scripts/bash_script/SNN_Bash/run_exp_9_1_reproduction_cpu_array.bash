#!/usr/bin/env bash
#SBATCH --job-name=exp9_1_rep
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=exp9_1_rep_%A_%a.out
#SBATCH --error=exp9_1_rep_%A_%a.err

set -eo pipefail
source /etc/profile
set -u
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
python -u -m scripts.experiment_9_1_a2_stability_decomposition \
  --device cpu --threads 1 run-reproduction \
  --array-task-id "${SLURM_ARRAY_TASK_ID:?}"
