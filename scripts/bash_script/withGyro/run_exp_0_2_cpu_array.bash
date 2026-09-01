#!/usr/bin/env bash
#SBATCH --job-name=wr-gyro02
#SBATCH --partition=cpu
#SBATCH --array=0-119%50
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=unity_withgyro_exp02_%A_%a.out
#SBATCH --error=unity_withgyro_exp02_%A_%a.err

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= 120 )); then
    echo "Invalid array task id: $TASK_ID" >&2
    exit 2
fi

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
printf '[withGyro exp0.2] task=%s conda=%s python=%s\n' \
    "$TASK_ID" "$CONDA_PREFIX" "$(command -v python)"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python -u -m scripts.with_gyro_experiment_0_2_nonlinear_temporal_decoder_probe \
    run-one \
    --array-task-id "$TASK_ID"
