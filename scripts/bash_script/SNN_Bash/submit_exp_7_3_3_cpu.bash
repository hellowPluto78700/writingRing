#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

bridge_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_3_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${bridge_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_7_3_3_cpu.bash)

echo "Exp7.3.3 affine-LIF array: ${bridge_job}"
echo "Exp7.3.3 finalizer: ${finalizer_job}"
