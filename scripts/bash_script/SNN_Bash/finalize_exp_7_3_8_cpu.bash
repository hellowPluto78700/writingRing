#!/usr/bin/env bash
#SBATCH --job-name=exp7_3_8_finalize
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-exp7_3_8-finalize-%j.out
#SBATCH --error=slurm-exp7_3_8-finalize-%j.err

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
python -m scripts.experiment_7_3_8_lif_gain_sweep \
  --device cpu \
  --threads 1 \
  finalize
