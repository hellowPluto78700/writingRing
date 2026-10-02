#!/usr/bin/env bash
#SBATCH --job-name=exp17_1_final
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -eo pipefail
source /etc/profile
set -u
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu 2>/dev/null || conda activate writingring-viz
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "${REPO_ROOT:-$PWD}"
python -m scripts.experiment_17_1_gradient_starvation aggregate
