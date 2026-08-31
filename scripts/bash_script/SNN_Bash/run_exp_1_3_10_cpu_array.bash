#!/bin/bash
#SBATCH --job-name=exp1_3_10
#SBATCH --output=logs/exp1_3_10_%A_%a.out
#SBATCH --error=logs/exp1_3_10_%A_%a.err
#SBATCH --array=0-35%18
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=03:00:00

set -euo pipefail

THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export MPLBACKEND=Agg

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
mkdir -p logs

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
LABELS="${LABELS:-A,B,C,D,E,X,G,H,I,J,K,L}"

echo "Experiment 1.3.10 task ${TASK_ID}/35 | CPUs=${THREADS} | labels=${LABELS}"

nice -n 10 python scripts/experiment_1_3_10_stacked_bin_snn_ablation.py \
    --run-index "$TASK_ID" \
    --labels "$LABELS" \
    --device cpu \
    --threads "$THREADS" \
    --settle-seconds 15 \
    --finalize-if-ready
