#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

prepare_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_10_2_1_cpu.bash)
array_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_10_2_1_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${array_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_10_2_1_cpu.bash)

echo "Exp10.2.1 prepare: ${prepare_job}"
echo "Exp10.2.1 18-run CPU array: ${array_job}"
echo "Exp10.2.1 finalizer: ${finalizer_job}"
