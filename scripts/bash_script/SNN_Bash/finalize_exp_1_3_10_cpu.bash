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

PYTHON_BIN="${WRITINGRING_PYTHON:-}"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "WRITINGRING_PYTHON is missing or not executable. Submit with submit_exp_1_3_10_cpu.bash from an activated writingring environment." >&2
    exit 2
fi

LABELS="${LABELS:-A,B,C,D,E,X,G,H,I,J,K,L}"
echo "Python: ${PYTHON_BIN}"

"$PYTHON_BIN" scripts/experiment_1_3_10_stacked_bin_snn_ablation.py \
    --labels "$LABELS" \
    --device cpu \
    --threads 1 \
    --finalize-only \
    --require-complete
