#!/usr/bin/env bash
#SBATCH --job-name=exp8_0_1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=exp8_0_1_%A_%a.out
#SBATCH --error=exp8_0_1_%A_%a.err

set -eo pipefail
source /etc/profile
set -u
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
module load conda/latest
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx writingring-gpu; then conda activate writingring-gpu; else conda activate writingring-viz; fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u -m scripts.experiment_8_0_1_l1_l2_fusion_probe \
  --device cpu --threads 1 run --array-task-id "${SLURM_ARRAY_TASK_ID:?}"
