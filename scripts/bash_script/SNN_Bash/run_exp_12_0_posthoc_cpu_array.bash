#!/usr/bin/env bash
#SBATCH --job-name=exp12_0_post
#SBATCH --array=0-26%27
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --deadline=2026-09-23T05:00:00
#SBATCH --output=exp12_0_post_%A_%a.out
#SBATCH --error=exp12_0_post_%A_%a.err

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
python -u -m scripts.experiment_12_0_local_primitive_bottleneck \
  --device cpu --threads 1 posthoc \
  --array-task-id "${SLURM_ARRAY_TASK_ID:?}"
