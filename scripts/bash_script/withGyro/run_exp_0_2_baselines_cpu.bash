#!/usr/bin/env bash
#SBATCH --job-name=wr-gyro02-base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=unity_withgyro_exp02_baselines_%j.out
#SBATCH --error=unity_withgyro_exp02_baselines_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

WRITINGRING_CONDA_PREFIX="${WRITINGRING_CONDA_PREFIX:-/work/pi_jgummeso_umass_edu/${USER}/.conda/envs/writingring-gpu}"
export WRITINGRING_CONDA_PREFIX

module load conda/latest
eval "$(conda shell.bash hook)"
if [[ ! -d "$WRITINGRING_CONDA_PREFIX" ]]; then
    echo "Missing writingring-gpu Conda prefix: $WRITINGRING_CONDA_PREFIX" >&2
    exit 2
fi
conda activate "$WRITINGRING_CONDA_PREFIX"
if [[ "${CONDA_PREFIX:-}" != "$WRITINGRING_CONDA_PREFIX" ]]; then
    echo "Activated unexpected Conda prefix: ${CONDA_PREFIX:-<unset>}" >&2
    exit 2
fi
python -c "import numpy, pandas, sklearn, torch" >/dev/null
printf '[withGyro exp0.2 baselines] conda=%s python=%s\n' \
    "$CONDA_PREFIX" "$(command -v python)"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python -u -m scripts.with_gyro_experiment_0_2_nonlinear_temporal_decoder_probe run-baselines
