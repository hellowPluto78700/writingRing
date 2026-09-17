#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

run_job=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_8_1_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${run_job}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_8_1_cpu.bash)

echo "Exp8.1 run array: ${run_job}"
echo "Exp8.1 finalizer: ${finalizer_job}"
