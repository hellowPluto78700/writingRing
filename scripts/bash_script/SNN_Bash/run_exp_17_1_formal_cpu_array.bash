#!/usr/bin/env bash
#SBATCH --job-name=exp17_1_formal
#SBATCH --array=0-11%12
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
cases=(C0 C1 C2 C3)
seed_idx=$((SLURM_ARRAY_TASK_ID / 4))
case_idx=$((SLURM_ARRAY_TASK_ID % 4))
python -m scripts.experiment_17_1_gradient_starvation run --seed "${seeds[$seed_idx]}" --case "${cases[$case_idx]}"
