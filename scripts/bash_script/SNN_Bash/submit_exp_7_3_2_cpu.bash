#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"
export REPO_ROOT

bridge_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_2_cpu_array.bash)
finalizer_job=$(sbatch --parsable --export=ALL \
  --dependency="afterok:${bridge_job}" \
  scripts/bash_script/SNN_Bash/finalize_exp_7_3_2_cpu.bash)

echo "Exp7.3.2 bridge array: ${bridge_job}"
echo "Exp7.3.2 finalizer: ${finalizer_job}"
