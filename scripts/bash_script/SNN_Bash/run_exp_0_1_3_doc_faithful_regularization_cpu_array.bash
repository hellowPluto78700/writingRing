#!/usr/bin/env bash
#SBATCH --job-name=exp0_1_3_docreg
#SBATCH --array=0-29%30
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/exp0_1_3_docreg_%A_%a.out
#SBATCH --error=outputs/exp0_1_3_docreg_%A_%a.err

set -euo pipefail

REPO_ROOT="${WRITINGRING_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p outputs

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

python -u -m scripts.experiment_0_1_3_doc_faithful_regularization \
    run-one \
    --array-task-id "${SLURM_ARRAY_TASK_ID}" \
    --device cpu \
    --threads 1
