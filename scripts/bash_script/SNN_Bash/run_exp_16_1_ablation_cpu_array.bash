#!/usr/bin/env bash
#SBATCH --job-name=exp16_1_ablate
#SBATCH --array=0-23%24
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -eo pipefail
source /etc/profile
set -u
cd "${REPO_ROOT:-$PWD}"
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu 2>/dev/null || conda activate writingring-viz
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
python -m scripts.experiment_16_1_z_only_prefix_interaction ablation --task-id "${SLURM_ARRAY_TASK_ID}"
