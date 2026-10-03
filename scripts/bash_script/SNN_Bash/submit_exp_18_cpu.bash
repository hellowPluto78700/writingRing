#!/usr/bin/env bash
set -eo pipefail
source /etc/profile
set -u

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

prepare_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_18_cpu.bash)
run_job=$(sbatch --parsable --dependency=afterok:"$prepare_job" --export=ALL scripts/bash_script/SNN_Bash/run_exp_18_cpu_array.bash)
ref_job=$(sbatch --parsable --dependency=afterok:"$prepare_job" --export=ALL scripts/bash_script/SNN_Bash/diagnose_exp_18_ref_cpu_array.bash)
final_job=$(sbatch --parsable --dependency=afterok:"$run_job":"$ref_job" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_18_cpu.bash)

printf 'prepare=%s run=%s refdiag=%s finalize=%s\n' "$prepare_job" "$run_job" "$ref_job" "$final_job"
