#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT

REF_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_0_2_2_reference_cpu.bash"
WARMUP_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_0_2_2_warmup_cpu_array.bash"
MAIN_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_0_2_2_capacity_preserving_cpu_array.bash"
FINAL_SCRIPT="scripts/bash_script/SNN_Bash/finalize_exp_0_2_2_capacity_preserving_cpu.bash"

ref_job="$(sbatch --parsable --export=ALL "$REF_SCRIPT")"
warmup_job="$(sbatch --parsable --export=ALL "$WARMUP_SCRIPT")"
main_job="$(sbatch --parsable --dependency="afterok:${ref_job}:${warmup_job}" --export=ALL "$MAIN_SCRIPT")"
final_job="$(sbatch --parsable --dependency="afterok:${main_job}" --export=ALL "$FINAL_SCRIPT")"

printf 'Exp0.2.2 reference job: %s\n' "$ref_job"
printf 'Exp0.2.2 warmup array: %s\n' "$warmup_job"
printf 'Exp0.2.2 main array: %s\n' "$main_job"
printf 'Exp0.2.2 finalizer: %s\n' "$final_job"
