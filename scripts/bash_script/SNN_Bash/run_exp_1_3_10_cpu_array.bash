#!/bin/bash
#SBATCH --job-name=exp1_3_10
#SBATCH --output=logs/exp1_3_10_%A_%a.out
#SBATCH --error=logs/exp1_3_10_%A_%a.err
#SBATCH --array=0-59%50
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=03:00:00

set -euo pipefail

THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
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

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
LABELS="${LABELS:-A,B,C,D,E,X,G,H,I,J,K,L}"

echo "Experiment 1.3.10 task ${TASK_ID}/59 | CPUs=${THREADS} | labels=${LABELS}"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
which python

nice -n 10 python scripts/experiment_1_3_10_runner.py \
    --run-index "$TASK_ID" \
    --labels "$LABELS" \
    --device cpu \
    --threads "$THREADS"
