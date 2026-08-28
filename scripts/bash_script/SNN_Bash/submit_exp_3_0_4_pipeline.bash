#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${1:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

TRAIN_EVAL_JOB=$(sbatch \
  scripts/bash_script/SNN_Bash/run_exp_3_0_4_train_cpu_array.bash \
  | awk '{print $4}')
BASELINE_EVAL_JOB=$(sbatch \
  scripts/bash_script/SNN_Bash/run_exp_3_0_4_baseline_eval_cpu.bash \
  | awk '{print $4}')
FINAL_JOB=$(sbatch \
  --dependency=afterok:${TRAIN_EVAL_JOB}:${BASELINE_EVAL_JOB} \
  scripts/bash_script/SNN_Bash/finalize_exp_3_0_4_cpu.bash \
  | awk '{print $4}')

printf 'Experiment 3.0.4 pipeline submitted\n'
printf '  W32/W64/W256 train+eval array: %s\n' "$TRAIN_EVAL_JOB"
printf '  W128 baseline eval:             %s\n' "$BASELINE_EVAL_JOB"
printf '  finalizer:                      %s\n' "$FINAL_JOB"
printf '\nMonitor with:\n'
printf '  squeue -u %s\n' "$USER"
printf '  sacct -j %s,%s,%s --format=JobID,JobName,State,Elapsed,ExitCode\n' \
  "$TRAIN_EVAL_JOB" "$BASELINE_EVAL_JOB" "$FINAL_JOB"
