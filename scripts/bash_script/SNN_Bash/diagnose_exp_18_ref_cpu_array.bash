#!/usr/bin/env bash
#SBATCH --job-name=exp18_refdiag
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
set -eo pipefail
source /etc/profile
set -u
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu 2>/dev/null || conda activate writingring-viz
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "${REPO_ROOT:-$PWD}"
seeds=(11 23 37)
python -m scripts.experiment_18_membrane_history diagnose --case I_REF --seed "${seeds[$SLURM_ARRAY_TASK_ID]}"
