#!/bin/bash
#SBATCH --job-name=exp1_3_10_linear
#SBATCH --output=logs/exp1_3_10_linear_%j.out
#SBATCH --error=logs/exp1_3_10_linear_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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
echo "Experiment 1.3.10 Fixed250+Linear baseline | labels=${LABELS}"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
which python

nice -n 10 python scripts/experiment_1_3_10_fixed250_linear_baseline.py \
    --labels "$LABELS"
