#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT

BASELINE_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_7_0_baseline_cpu.bash"
LOCAL_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_7_0_local_cpu_array.bash"
HIERARCHICAL_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_7_0_hierarchical_cpu_array.bash"
FINAL_SCRIPT="scripts/bash_script/SNN_Bash/finalize_exp_7_0_cpu.bash"

baseline_job="$(sbatch --parsable --export=ALL "$BASELINE_SCRIPT")"
local_job="$(sbatch --parsable --export=ALL "$LOCAL_SCRIPT")"
hierarchical_job="$(sbatch --parsable --export=ALL "$HIERARCHICAL_SCRIPT")"
final_job="$(sbatch --parsable --dependency="afterok:${baseline_job}:${local_job}:${hierarchical_job}" --export=ALL "$FINAL_SCRIPT")"

printf 'Exp7.0 raw Fixed250+Linear baseline job: %s\n' "$baseline_job"
printf 'Exp7.0 local-SNN baseline array: %s\n' "$local_job"
printf 'Exp7.0 hierarchical SNN array: %s\n' "$hierarchical_job"
printf 'Exp7.0 artifact-only finalizer: %s\n' "$final_job"
