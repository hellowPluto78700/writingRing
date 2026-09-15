#!/usr/bin/env bash
#SBATCH --job-name=exp7_2_6_4_fin
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=exp7_2_6_4_fin_%j.out
#SBATCH --error=exp7_2_6_4_fin_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
module load conda/latest
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx writingring-gpu; then conda activate writingring-gpu; else conda activate writingring-viz; fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u -m scripts.experiment_7_2_6_4_output_interface_decomposition --device cpu --threads 1 finalize
