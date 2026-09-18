#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

prepare_job=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/prepare_exp_10_0_cpu.bash)
array_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_10_0_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${array_job}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_10_0_cpu.bash)

echo "Exp10.0 prepare: ${prepare_job}"
echo "Exp10.0 A2 array: ${array_job}"
echo "Exp10.0 finalizer: ${finalizer_job}"
