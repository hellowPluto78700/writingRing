#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${1:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

TRAIN_JOB=$(sbatch scripts/bash_script/SNN_Bash/run_exp_3_0_2_train_cpu_array.bash | awk '{print $4}')
EVAL_JOB=$(sbatch \
  --dependency=afterok:${TRAIN_JOB} \
  scripts/bash_script/SNN_Bash/run_exp_3_0_2_eval_cpu_array.bash \
  | awk '{print $4}')
FINAL_JOB=$(sbatch \
  --dependency=afterok:${EVAL_JOB} \
  scripts/bash_script/SNN_Bash/finalize_exp_3_0_2_cpu.bash \
  | awk '{print $4}')

printf 'Experiment 3.0.2 pipeline submitted\n'
printf '  training array:   %s\n' "$TRAIN_JOB"
printf '  evaluation array: %s\n' "$EVAL_JOB"
printf '  finalizer:        %s\n' "$FINAL_JOB"
printf '\nMonitor with:\n'
printf '  squeue -u %s\n' "$USER"
printf '  sacct -j %s,%s,%s --format=JobID,JobName,State,Elapsed,ExitCode\n' \
  "$TRAIN_JOB" "$EVAL_JOB" "$FINAL_JOB"
