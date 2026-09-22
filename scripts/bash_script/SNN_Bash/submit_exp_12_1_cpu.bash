#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

array_job=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_12_1_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${array_job}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_12_1_cpu.bash)

echo "Exp12.1 24-run CPU array: ${array_job}"
echo "Exp12.1 finalizer: ${finalizer_job}"
