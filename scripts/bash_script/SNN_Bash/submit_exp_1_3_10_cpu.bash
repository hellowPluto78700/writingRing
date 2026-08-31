#!/bin/bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

ARRAY_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_1_3_10_cpu_array.bash"
BASELINE_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_1_3_10_fixed250_linear_baseline.bash"
FINALIZER_SCRIPT="scripts/bash_script/SNN_Bash/finalize_exp_1_3_10_cpu.bash"

# Keep the comma-separated label list as one environment-variable value.
# Do not embed it in sbatch --export=..., because Slurm uses commas to
# separate exported variables and would truncate LABELS to the first label.
export LABELS="${LABELS:-A,B,C,D,E,X,G,H,I,J,K,L}"

ARRAY_JOB_ID="$(
    sbatch --parsable \
        --export=ALL \
        "$ARRAY_SCRIPT"
)"
BASELINE_JOB_ID="$(
    sbatch --parsable \
        --export=ALL \
        "$BASELINE_SCRIPT"
)"
FINALIZER_JOB_ID="$(
    sbatch --parsable \
        --dependency="afterok:${ARRAY_JOB_ID}:${BASELINE_JOB_ID}" \
        --export=ALL \
        "$FINALIZER_SCRIPT"
)"

echo "Labels: ${LABELS}"
echo "Submitted Experiment 1.3.10 SNN array job: ${ARRAY_JOB_ID}"
echo "Submitted Fixed250+Linear baseline job: ${BASELINE_JOB_ID}"
echo "Submitted afterok finalizer job: ${FINALIZER_JOB_ID}"
