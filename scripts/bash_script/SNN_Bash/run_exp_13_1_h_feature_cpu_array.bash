#!/usr/bin/env bash
#SBATCH --job-name=exp13_1_hf
#SBATCH --array=0-17%18
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=exp13_1_hf_%A_%a.out
#SBATCH --error=exp13_1_hf_%A_%a.err

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

python -u -m scripts.experiment_13_1_abstraction_generalization   --device cpu --threads 1 h-feature   --array-task-id "${SLURM_ARRAY_TASK_ID:?}"
