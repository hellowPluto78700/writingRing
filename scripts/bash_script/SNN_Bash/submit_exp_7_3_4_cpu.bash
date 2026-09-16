#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

train_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_4_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${train_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_7_3_4_cpu.bash)

echo "Exp7.3.4 LIF fine-tuning array: ${train_job}"
echo "Exp7.3.4 finalizer: ${finalizer_job}"
