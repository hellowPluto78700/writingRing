#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

e2e_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_e2e_cpu_array.bash)
cache_job=$(sbatch --parsable --dependency="afterok:${e2e_job}" --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_7_3_stage2_cache_cpu_array.bash)
stage2_job=$(sbatch --parsable --dependency="afterok:${cache_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_stage2_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${e2e_job}:${stage2_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_7_3_cpu.bash)

echo "Exp7.3 E2E array: ${e2e_job}"
echo "Exp7.3 Stage2 cache array: ${cache_job}"
echo "Exp7.3 Stage2 array: ${stage2_job}"
echo "Exp7.3 finalizer: ${finalizer_job}"
