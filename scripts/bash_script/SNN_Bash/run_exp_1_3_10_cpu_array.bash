#!/bin/bash
#SBATCH --job-name=exp1_3_10
#SBATCH --output=logs/exp1_3_10_%A_%a.out
#SBATCH --error=logs/exp1_3_10_%A_%a.err
#SBATCH --array=0-35%36
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

PYTHON_BIN="${WRITINGRING_PYTHON:-}"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "WRITINGRING_PYTHON is missing or not executable. Submit with submit_exp_1_3_10_cpu.bash from an activated writingring environment." >&2
    exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
LABELS="${LABELS:-A,B,C,D,E,X,G,H,I,J,K,L}"

echo "Experiment 1.3.10 task ${TASK_ID}/35 | CPUs=${THREADS} | labels=${LABELS}"
echo "Python: ${PYTHON_BIN}"

nice -n 10 "$PYTHON_BIN" scripts/experiment_1_3_10_stacked_bin_snn_ablation.py \
    --run-index "$TASK_ID" \
    --labels "$LABELS" \
    --device cpu \
    --threads "$THREADS"
