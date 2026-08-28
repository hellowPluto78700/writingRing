#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${1:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

EXP01_ARRAY=$(sbatch scripts/bash_script/SNN_Bash/run_exp_0_1_cpu_array.bash | awk '{print $4}')
EXP01_FINAL=$(sbatch --dependency=afterok:${EXP01_ARRAY} scripts/bash_script/SNN_Bash/finalize_exp_0_1_cpu.bash | awk '{print $4}')

printf 'Experiment 0.1 submitted\n'
printf '  split array: %s\n' "$EXP01_ARRAY"
printf '  finalizer:   %s\n' "$EXP01_FINAL"
printf '\nMonitor with:\n'
printf '  squeue -u %s\n' "$USER"
printf '  sacct -j %s,%s --format=JobID,JobName,State,Elapsed,ExitCode\n' "$EXP01_ARRAY" "$EXP01_FINAL"
