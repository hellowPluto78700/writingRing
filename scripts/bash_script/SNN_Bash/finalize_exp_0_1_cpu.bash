#!/usr/bin/env bash
#SBATCH --job-name=wr01-final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=unity_exp01_finalize_%j.out
#SBATCH --error=unity_exp01_finalize_%j.err

set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
python -u -m scripts.experiment_0_1_growing_prefix_relative10 finalize
