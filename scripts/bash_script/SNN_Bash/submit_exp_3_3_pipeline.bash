#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${1:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

EXP33_ARRAY=$(sbatch scripts/bash_script/SNN_Bash/run_exp_3_3_cpu_array.bash | awk '{print $4}')
EXP33_FINAL=$(sbatch --dependency=afterok:${EXP33_ARRAY} scripts/bash_script/SNN_Bash/finalize_exp_3_3_cpu.bash | awk '{print $4}')

printf 'Experiment 3.3 submitted\n'
printf '  residual array: %s\n' "$EXP33_ARRAY"
printf '  finalizer:      %s\n' "$EXP33_FINAL"
printf '\nMonitor with:\n'
printf '  squeue -u %s\n' "$USER"
printf '  sacct -j %s,%s --format=JobID,JobName,State,Elapsed,ExitCode\n' "$EXP33_ARRAY" "$EXP33_FINAL"
