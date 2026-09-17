#!/usr/bin/env bash
#SBATCH --job-name=exp7_4
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=1
#SBATCH --mem=14G
#SBATCH --time=30:00:00
#SBATCH --output=exp7_4_%A_%a.out
#SBATCH --error=exp7_4_%A_%a.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
source /etc/profile
module load conda/latest
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx writingring-gpu; then conda activate writingring-gpu; else conda activate writingring-viz; fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u -m scripts.experiment_7_4_latent_softmax_evidence \
  --device cpu --threads 1 run --array-task-id "${SLURM_ARRAY_TASK_ID:?}"
