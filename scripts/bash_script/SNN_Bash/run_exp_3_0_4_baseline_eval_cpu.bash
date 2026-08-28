#!/usr/bin/env bash
#SBATCH --job-name=wr304-w128-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=unity_exp304_w128_eval_%j.out
#SBATCH --error=unity_exp304_w128_eval_%j.err

set -euo pipefail

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

printf 'Node: %s\n' "$(hostname)"
printf 'CPUs per task: %s\n' "${SLURM_CPUS_PER_TASK:-1}"

python -u scripts/experiment_3_0_4/02_evaluate_baseline_w128.py --device cpu
