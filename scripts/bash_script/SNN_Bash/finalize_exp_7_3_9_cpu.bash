#!/usr/bin/env bash
#SBATCH --job-name=exp7_3_9_final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-exp7_3_9_final-%A.out
#SBATCH --error=slurm-exp7_3_9_final-%A.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load conda/latest
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx 'writingring-gpu'; then
  conda activate writingring-gpu
else
  conda activate writingring-viz
fi

cd "${SLURM_SUBMIT_DIR}"
python -m scripts.experiment_7_3_9_pretrained_a2_depth_extension \
  --device cpu \
  --threads 1 \
  finalize
