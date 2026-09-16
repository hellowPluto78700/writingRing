#!/usr/bin/env bash
#SBATCH --job-name=exp7_3_2_finalize
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/exp7_3_2_finalize_%j.out
#SBATCH --error=logs/exp7_3_2_finalize_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

module load conda/latest
eval "$(conda shell.bash hook)"
if ! conda activate writingring-gpu; then
  conda activate writingring-viz
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m scripts.experiment_7_3_2_affine_probe_bridge \
  --repo-root "$REPO_ROOT" --threads 1 finalize
