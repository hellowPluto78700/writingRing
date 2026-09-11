#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_0_2_shift_fire_rate_cpu_array.bash"
FINALIZE_SCRIPT="scripts/bash_script/SNN_Bash/finalize_exp_0_2_shift_fire_rate_cpu.bash"

ARRAY_JOB_ID="$(sbatch --parsable --export=ALL "$ARRAY_SCRIPT")"
FINALIZE_JOB_ID="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB_ID}" --export=ALL "$FINALIZE_SCRIPT")"

echo "Exp0.2 shift-resolved firing-rate array job: ${ARRAY_JOB_ID}"
echo "Exp0.2 shift-resolved firing-rate finalizer job: ${FINALIZE_JOB_ID} (afterok:${ARRAY_JOB_ID})"
