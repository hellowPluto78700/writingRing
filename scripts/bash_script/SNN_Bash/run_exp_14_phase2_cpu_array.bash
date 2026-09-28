#!/usr/bin/env bash
#SBATCH --job-name=exp14_p2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
set -eo pipefail
source /etc/profile
set -u
cd "${REPO_ROOT:-$PWD}"
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu 2>/dev/null || conda activate writingring-viz
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
python -m scripts.experiment_14_history_organization train-phase2 --task-id "${SLURM_ARRAY_TASK_ID}"
