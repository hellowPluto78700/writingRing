#!/usr/bin/env bash
#SBATCH --job-name=exp7_2_6_3_mech
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=exp7_2_6_3_mech_%A_%a.out
#SBATCH --error=exp7_2_6_3_mech_%A_%a.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
module load conda/latest
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx writingring-gpu; then conda activate writingring-gpu; else conda activate writingring-viz; fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u -m scripts.experiment_7_2_6_3_linear_vs_lif_frozen_l2 \
  --device cpu --threads 1 mechanism --array-task-id "${SLURM_ARRAY_TASK_ID:?}"
