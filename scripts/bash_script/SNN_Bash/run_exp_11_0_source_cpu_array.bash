#!/usr/bin/env bash
#SBATCH --job-name=exp11_0_src
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=exp11_0_src_%A_%a.out
#SBATCH --error=exp11_0_src_%A_%a.err

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
force_args=()
if [[ "${EXP11_0_FORCE_SOURCE:-0}" == "1" ]]; then force_args+=(--force); fi
python -u -m scripts.experiment_11_0_rsnn_history_internalization   --device cpu --threads 1 train-source-one   --array-task-id "${SLURM_ARRAY_TASK_ID:?}"   "${force_args[@]}"
