#!/usr/bin/env bash
#SBATCH --job-name=exp15_p15
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=12:00:00
set -eo pipefail
source /etc/profile
set -u
cd "${REPO_ROOT:-$PWD}"
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu 2>/dev/null || conda activate writingring-viz
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
python -m scripts.experiment_15_context_dependent_write_gate phase1-5 --task-id "${SLURM_ARRAY_TASK_ID}"
