#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
NEW_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_7_2_3_new_cpu_array.bash"
REUSE_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_7_2_3_reuse_cpu_array.bash"
FINAL_SCRIPT="scripts/bash_script/SNN_Bash/finalize_exp_7_2_3_cpu.bash"

new_job="$(sbatch --parsable --export=ALL "$NEW_SCRIPT")"
reuse_job="$(sbatch --parsable --export=ALL "$REUSE_SCRIPT")"
final_job="$(sbatch --parsable --dependency="afterok:${new_job}:${reuse_job}" --export=ALL "$FINAL_SCRIPT")"

printf 'Exp7.2.3 new-training array: %s\n' "$new_job"
printf 'Exp7.2.3 reuse-eval array: %s\n' "$reuse_job"
printf 'Exp7.2.3 finalizer: %s\n' "$final_job"
