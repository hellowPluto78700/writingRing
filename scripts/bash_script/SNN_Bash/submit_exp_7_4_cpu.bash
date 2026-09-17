#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

array_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_4_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${array_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_7_4_cpu.bash)

echo "Exp7.4 training array: ${array_job}"
echo "Exp7.4 finalizer: ${finalizer_job}"
