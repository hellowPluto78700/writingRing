#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
WORKER_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_7_2_1_phase_probe_cpu_array.bash"
FINAL_SCRIPT="scripts/bash_script/SNN_Bash/finalize_exp_7_2_1_phase_probe_cpu.bash"
worker_job="$(sbatch --parsable --export=ALL "$WORKER_SCRIPT")"
final_job="$(sbatch --parsable --dependency="afterok:${worker_job}" --export=ALL "$FINAL_SCRIPT")"
printf 'Exp7.2.1 frozen-checkpoint probe array: %s\n' "$worker_job"
printf 'Exp7.2.1 aggregate finalizer: %s\n' "$final_job"
