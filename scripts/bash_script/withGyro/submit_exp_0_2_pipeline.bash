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
python -c "import numpy, pandas, sklearn, torch" >/dev/null
python -m scripts.with_gyro_experiment_0_2_nonlinear_temporal_decoder_probe describe >/dev/null
printf '[withGyro exp0.2] preflight conda=%s python=%s\n' \
    "$CONDA_PREFIX" "$(command -v python)"

BASELINE_JOB=$(
    sbatch \
        --export="ALL,WRITINGRING_CONDA_PREFIX=${WRITINGRING_CONDA_PREFIX}" \
        scripts/bash_script/withGyro/run_exp_0_2_baselines_cpu.bash \
    | awk '{print $4}'
)
ARRAY_JOB=$(
    sbatch \
        --dependency="afterok:${BASELINE_JOB}" \
        --export="ALL,WRITINGRING_CONDA_PREFIX=${WRITINGRING_CONDA_PREFIX}" \
        scripts/bash_script/withGyro/run_exp_0_2_cpu_array.bash \
    | awk '{print $4}'
)
FINAL_JOB=$(
    sbatch \
        --dependency="afterok:${ARRAY_JOB}" \
        --export="ALL,WRITINGRING_CONDA_PREFIX=${WRITINGRING_CONDA_PREFIX}" \
        scripts/bash_script/withGyro/finalize_exp_0_2_cpu.bash \
    | awk '{print $4}'
)

printf 'WithGyro Experiment 0.2 submitted\n'
printf '  conda:      %s\n' "$WRITINGRING_CONDA_PREFIX"
printf '  baselines:  %s\n' "$BASELINE_JOB"
printf '  array:      %s\n' "$ARRAY_JOB"
printf '  finalizer:  %s\n' "$FINAL_JOB"
printf '  baselines:  30 = 3 channel sets x 2 representations x 5 split seeds\n'
printf '  neural:     120 = 3 channel sets x 2 representations x 4 neural decoders x 5 split seeds\n'
printf '  decoders:   local, transition, local_transition, gru\n'
printf '  max CPU concurrency: 50\n'
printf '\nMonitor with:\n'
printf '  squeue -u %s\n' "$USER"
printf '  sacct -j %s,%s,%s --format=JobID,JobName,State,Elapsed,ExitCode\n' \
    "$BASELINE_JOB" "$ARRAY_JOB" "$FINAL_JOB"
