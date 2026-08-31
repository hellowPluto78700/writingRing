#!/bin/bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

ARRAY_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_1_3_10_cpu_array.bash"
FINALIZER_SCRIPT="scripts/bash_script/SNN_Bash/finalize_exp_1_3_10_cpu.bash"

ARRAY_JOB_ID="$(sbatch --parsable "$ARRAY_SCRIPT")"
FINALIZER_JOB_ID="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB_ID}" "$FINALIZER_SCRIPT")"

echo "Submitted Experiment 1.3.10 array job: ${ARRAY_JOB_ID}"
echo "Submitted afterok finalizer job: ${FINALIZER_JOB_ID}"
