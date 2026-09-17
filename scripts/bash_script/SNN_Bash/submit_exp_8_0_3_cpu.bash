#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

array_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_8_0_3_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${array_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_8_0_3_cpu.bash)

echo "Exp8.0.3 training array: ${array_job}"
echo "Exp8.0.3 finalizer: ${finalizer_job}"
