#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

EVAL_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_7_2_4_cpu_array.bash"
FINAL_SCRIPT="scripts/bash_script/SNN_Bash/finalize_exp_7_2_4_cpu.bash"

eval_job="$(sbatch --parsable --export=ALL "$EVAL_SCRIPT")"
final_job="$(sbatch --parsable --dependency="afterok:${eval_job}" --export=ALL "$FINAL_SCRIPT")"

echo "Exp7.2.4 evaluation array: ${eval_job}"
echo "Exp7.2.4 finalizer: ${final_job}"
