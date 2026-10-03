#!/usr/bin/env bash
#SBATCH --job-name=exp18_run
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
set -eo pipefail
source /etc/profile
set -u
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu 2>/dev/null || conda activate writingring-viz
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "${REPO_ROOT:-$PWD}"
seeds=(11 23 37)
cases=(U_NORMAL U_DETACH)
seed_idx=$((SLURM_ARRAY_TASK_ID / 2))
case_idx=$((SLURM_ARRAY_TASK_ID % 2))
python -m scripts.experiment_18_membrane_history run --seed "${seeds[$seed_idx]}" --case "${cases[$case_idx]}"
