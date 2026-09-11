#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT

BASELINE_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_6_0_baseline_cpu.bash"
MAIN_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_6_0_cpu_array.bash"
FINAL_SCRIPT="scripts/bash_script/SNN_Bash/finalize_exp_6_0_cpu.bash"

baseline_job="$(sbatch --parsable --export=ALL "$BASELINE_SCRIPT")"
main_job="$(sbatch --parsable --export=ALL "$MAIN_SCRIPT")"
final_job="$(sbatch --parsable --dependency="afterok:${baseline_job}:${main_job}" --export=ALL "$FINAL_SCRIPT")"

printf 'Exp6.0 Fixed250 baseline job: %s\n' "$baseline_job"
printf 'Exp6.0 SNN array: %s\n' "$main_job"
printf 'Exp6.0 finalizer: %s\n' "$final_job"
