#!/usr/bin/env bash
#SBATCH --job-name=exp10_2_replay
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=exp10_2_replay_%A_%a.out
#SBATCH --error=exp10_2_replay_%A_%a.err

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
if [[ "${EXP10_2_FORCE:-0}" == "1" ]]; then force_args+=(--force); fi
python -u -m scripts.experiment_10_2_l1_membrane_memory \
  --device cpu --threads 1 run-replay \
  --array-task-id "${SLURM_ARRAY_TASK_ID:?}" \
  "${force_args[@]}"
