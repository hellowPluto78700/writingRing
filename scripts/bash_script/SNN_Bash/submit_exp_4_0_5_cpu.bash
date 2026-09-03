#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

ARRAY_JOB="$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_4_0_5_cpu_array.bash)"
FINAL_JOB="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_4_0_5_cpu.bash)"

echo "Exp4.0.5 array job: ${ARRAY_JOB}"
echo "Exp4.0.5 finalizer job: ${FINAL_JOB}"
echo "Finalizer dependency: afterok:${ARRAY_JOB}"
