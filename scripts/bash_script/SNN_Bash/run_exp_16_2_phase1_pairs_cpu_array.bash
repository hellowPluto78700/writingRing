#!/usr/bin/env bash
#SBATCH --job-name=exp16_2_p1
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu || conda activate writingring-viz
cd "${REPO_ROOT:-$PWD}"
python -m scripts.experiment_16_2_matched_budget_selective_write train-pair --task-id "${SLURM_ARRAY_TASK_ID}"