#!/usr/bin/env bash
#SBATCH --job-name=wr341
#SBATCH --partition=cpu
#SBATCH --array=0-24%25
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=unity_exp341_array_%A_%a.out
#SBATCH --error=unity_exp341_array_%A_%a.err

set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
python -u -m scripts.experiment_3_4_1_raw_causal_temporal_readout run-one --array-task-id "$TASK_ID"
