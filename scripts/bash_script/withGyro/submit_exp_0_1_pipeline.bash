#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${1:-$PWD}"
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
python -c "import numpy, pandas, sklearn" >/dev/null
python -m scripts.with_gyro_experiment_0_1_temporal_representation_probe describe >/dev/null
printf '[withGyro exp0.1] preflight conda=%s python=%s\n' \
    "$CONDA_PREFIX" "$(command -v python)"

ARRAY_JOB=$(
    sbatch \
        --export="ALL,WRITINGRING_CONDA_PREFIX=${WRITINGRING_CONDA_PREFIX}" \
        scripts/bash_script/withGyro/run_exp_0_1_cpu_array.bash \
    | awk '{print $4}'
)
FINAL_JOB=$(
    sbatch \
        --dependency="afterok:${ARRAY_JOB}" \
        --export="ALL,WRITINGRING_CONDA_PREFIX=${WRITINGRING_CONDA_PREFIX}" \
        scripts/bash_script/withGyro/finalize_exp_0_1_cpu.bash \
    | awk '{print $4}'
)

printf 'WithGyro Experiment 0.1 submitted\n'
printf '  conda:     %s\n' "$WRITINGRING_CONDA_PREFIX"
printf '  array:     %s\n' "$ARRAY_JOB"
printf '  finalizer: %s\n' "$FINAL_JOB"
printf '  tasks:     55 = 5 split seeds x (2 fixed-duration + 9 relative-progress conditions)\n'
printf '  max CPU concurrency: 50\n'
printf '\nMonitor with:\n'
printf '  squeue -u %s\n' "$USER"
printf '  sacct -j %s,%s --format=JobID,JobName,State,Elapsed,ExitCode\n' "$ARRAY_JOB" "$FINAL_JOB"
