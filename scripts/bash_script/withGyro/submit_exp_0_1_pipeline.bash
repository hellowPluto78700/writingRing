#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${1:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch scripts/bash_script/withGyro/run_exp_0_1_cpu_array.bash | awk '{print $4}')
FINAL_JOB=$(
    sbatch \
        --dependency="afterok:${ARRAY_JOB}" \
        scripts/bash_script/withGyro/finalize_exp_0_1_cpu.bash \
    | awk '{print $4}'
)

printf 'WithGyro Experiment 0.1 submitted\n'
printf '  array:     %s\n' "$ARRAY_JOB"
printf '  finalizer: %s\n' "$FINAL_JOB"
printf '  tasks:     55 = 5 split seeds x (2 fixed-duration + 9 relative-progress conditions)\n'
printf '  max CPU concurrency: 50\n'
printf '\nMonitor with:\n'
printf '  squeue -u %s\n' "$USER"
printf '  sacct -j %s,%s --format=JobID,JobName,State,Elapsed,ExitCode\n' "$ARRAY_JOB" "$FINAL_JOB"
