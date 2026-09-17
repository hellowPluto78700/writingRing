#!/usr/bin/env bash
#SBATCH --job-name=exp8_1_1_fin
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=exp8_1_1_fin_%j.out
#SBATCH --error=exp8_1_1_fin_%j.err

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
python -u -m scripts.experiment_8_1_1_two_layer_mt_factorial \
  --device cpu --threads 1 finalize
