#!/usr/bin/env bash
#SBATCH --job-name=exp17_final
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
set -eo pipefail
source /etc/profile
set -u
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu 2>/dev/null || conda activate writingring-viz
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "${REPO_ROOT:-$PWD}"
python -m scripts.experiment_17_persistent_pathway_story finalize