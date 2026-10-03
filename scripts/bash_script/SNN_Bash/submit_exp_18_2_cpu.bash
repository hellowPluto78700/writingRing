#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

smoke_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/smoke_exp_18_2_cpu.bash)
prepare_job=$(sbatch --parsable --dependency=afterok:"$smoke_job" --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_18_2_cpu.bash)
run_job=$(sbatch --parsable --dependency=afterok:"$prepare_job" --export=ALL scripts/bash_script/SNN_Bash/run_exp_18_2_cpu_array.bash)
final_job=$(sbatch --parsable --dependency=afterok:"$run_job" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_18_2_cpu.bash)

printf 'smoke=%s prepare=%s run=%s finalize=%s\n' "$smoke_job" "$prepare_job" "$run_job" "$final_job"
