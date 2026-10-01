#!/usr/bin/env bash
#SBATCH --job-name=exp16_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=100G
#SBATCH --time=02:00:00

set -eo pipefail
source /etc/profile
set -u

cd "${REPO_ROOT:-$PWD}"
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu 2>/dev/null || conda activate writingring-viz

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

WORKERS="${SLURM_CPUS_PER_TASK:-30}"
echo "Exp16.1 host=$(hostname) workers=${WORKERS} job=${SLURM_JOB_ID:-none}"

python -m scripts.experiment_16_1_z_only_prefix_interaction prepare
python -m scripts.experiment_16_1_z_only_prefix_interaction launch-train --max-workers "${WORKERS}"
python -m scripts.experiment_16_1_z_only_prefix_interaction launch-ablation --max-workers "${WORKERS}"
python -m scripts.experiment_16_1_z_only_prefix_interaction finalize
