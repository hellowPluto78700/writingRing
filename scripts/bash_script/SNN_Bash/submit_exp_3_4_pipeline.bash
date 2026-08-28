#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${1:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

EXP34_ARRAY=$(sbatch scripts/bash_script/SNN_Bash/run_exp_3_4_cpu_array.bash | awk '{print $4}')
EXP34_FINAL=$(sbatch --dependency=afterok:${EXP34_ARRAY} scripts/bash_script/SNN_Bash/finalize_exp_3_4_cpu.bash | awk '{print $4}')

printf 'Experiment 3.4 submitted\n'
printf '  training array: %s\n' "$EXP34_ARRAY"
printf '  finalizer:      %s\n' "$EXP34_FINAL"
printf '\nMonitor with:\n'
printf '  squeue -u %s\n' "$USER"
printf '  sacct -j %s,%s --format=JobID,JobName,State,Elapsed,ExitCode\n' "$EXP34_ARRAY" "$EXP34_FINAL"
