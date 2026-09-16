#!/usr/bin/env bash
#SBATCH --job-name=exp7_3_5_probe
#SBATCH --partition=cpu
#SBATCH --array=0-47%48
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/exp7_3_5_probe_%A_%a.out
#SBATCH --error=logs/exp7_3_5_probe_%A_%a.err

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

python -m scripts.experiment_7_3_5_hidden_state_information_loss \
  --repo-root "$REPO_ROOT" --threads 1 --device cpu \
  probe --array-task-id "$SLURM_ARRAY_TASK_ID"
