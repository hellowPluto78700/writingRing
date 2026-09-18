#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

# Stage B assumes Stage A baseline/replay artifacts already exist.
# Re-running prepare is cheap and keeps the saved split manifest synchronized.
prepare_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_10_2_cpu.bash)
long_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_10_2_long_e2e_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${long_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_10_2_cpu.bash)

echo "Exp10.2 Stage-B prepare: ${prepare_job}"
echo "Exp10.2 Stage-B long-memory E2E array: ${long_job}"
echo "Exp10.2 full finalizer: ${finalizer_job}"
