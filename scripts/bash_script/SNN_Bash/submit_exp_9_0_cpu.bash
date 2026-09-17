#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT
export EXP9_INSUFFICIENT_POLICY="${EXP9_INSUFFICIENT_POLICY:-keep_all}"

prepare_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_9_0_cpu.bash)
array_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_9_0_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${array_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_9_0_cpu.bash)

echo "Exp9.0 manifest/audit: ${prepare_job}"
echo "Exp9.0 CPU array: ${array_job}"
echo "Exp9.0 finalizer: ${finalizer_job}"
echo "Insufficient-pair policy: ${EXP9_INSUFFICIENT_POLICY}"
