#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${1:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

EXP31_ARRAY=$(sbatch scripts/bash_script/SNN_Bash/run_exp_3_1_cpu_array.bash | awk '{print $4}')
EXP31_FINAL=$(sbatch --dependency=afterok:${EXP31_ARRAY} scripts/bash_script/SNN_Bash/finalize_exp_3_1_cpu.bash | awk '{print $4}')

printf 'Experiment 3.1 submitted\n'
printf '  frozen matched eval array: %s\n' "$EXP31_ARRAY"
printf '  finalizer:                %s\n' "$EXP31_FINAL"
printf '\nMonitor with:\n'
printf '  squeue -u %s\n' "$USER"
printf '  sacct -j %s,%s --format=JobID,JobName,State,Elapsed,ExitCode\n' "$EXP31_ARRAY" "$EXP31_FINAL"
