#!/usr/bin/env bash
#SBATCH --job-name=wr34-causal
#SBATCH --partition=cpu
#SBATCH --array=0-8%9
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=unity_exp34_%A_%a.out
#SBATCH --error=unity_exp34_%A_%a.err

set -euo pipefail
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= 9 )); then
  echo "Invalid array task id: $TASK_ID" >&2
  exit 2
fi
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
python -u -m scripts.experiment_3_4_causal_fixed250_objectives run-one --array-task-id "$TASK_ID" --device cpu
