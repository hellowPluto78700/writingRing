#!/bin/bash
#SBATCH --job-name=exp1_3_10_finalize
#SBATCH --output=logs/exp1_3_10_finalize_%j.out
#SBATCH --error=logs/exp1_3_10_finalize_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

module load conda/latest

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx writingring-gpu; then
    conda activate writingring-gpu
elif conda env list | awk '{print $1}' | grep -qx writingring-viz; then
    conda activate writingring-viz
else
    echo "Neither writingring-gpu nor writingring-viz Conda environment is available on this compute node." >&2
    exit 2
fi

LABELS="${LABELS:-A,B,C,D,E,X,G,H,I,J,K,L}"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
which python

python scripts/experiment_1_3_10_runner.py \
    --labels "$LABELS" \
    --device cpu \
    --threads 1 \
    --finalize-only \
    --require-complete
