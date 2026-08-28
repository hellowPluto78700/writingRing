#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="$(pwd)"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

EXP341_ARRAY="$(sbatch --parsable scripts/bash_script/SNN_Bash/run_exp_3_4_1_cpu_array.bash)"
echo "Experiment 3.4.1 array: ${EXP341_ARRAY}"
EXP341_FINAL="$(sbatch --parsable --dependency=afterok:${EXP341_ARRAY} scripts/bash_script/SNN_Bash/finalize_exp_3_4_1_cpu.bash)"
echo "Experiment 3.4.1 finalizer: ${EXP341_FINAL}"
