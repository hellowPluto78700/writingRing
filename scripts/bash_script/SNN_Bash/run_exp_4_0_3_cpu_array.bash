#!/usr/bin/env bash
#SBATCH --job-name=exp4_0_3
#SBATCH --array=0-19%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=08:00:00
#SBATCH --output=exp4_0_3_%A_%a.out
#SBATCH --error=exp4_0_3_%A_%a.err

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
python -u -m scripts.experiment_4_0_3_hierarchical_snn run-one \
    --array-task-id "$TASK_ID" \
    --device cpu \
    --threads 1
