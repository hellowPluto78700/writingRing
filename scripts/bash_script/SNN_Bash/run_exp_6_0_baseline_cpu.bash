#!/usr/bin/env bash
#SBATCH --job-name=exp6_0_base
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=exp6_0_baseline_%j.out
#SBATCH --error=exp6_0_baseline_%j.err

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
python -u -m scripts.experiment_6_0_multiscale_phase_evidence \
  --device cpu \
  --threads 1 \
  baseline
