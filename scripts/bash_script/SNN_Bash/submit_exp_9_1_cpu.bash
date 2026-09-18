#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

prepare_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_9_1_cpu.bash)
repro_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_9_1_reproduction_cpu_array.bash)
factorial_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_9_1_factorial_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${repro_job}:${factorial_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_9_1_cpu.bash)

echo "Exp9.1 prepare: ${prepare_job}"
echo "Exp9.1 reproduction CPU array: ${repro_job}"
echo "Exp9.1 5x5 factorial CPU array: ${factorial_job}"
echo "Exp9.1 finalizer: ${finalizer_job}"
