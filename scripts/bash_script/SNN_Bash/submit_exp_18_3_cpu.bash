#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

smoke_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/smoke_exp_18_3_cpu.bash)
prepare_job=$(sbatch --parsable --dependency=afterok:"$smoke_job" --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_18_3_cpu.bash)
diag_job=$(sbatch --parsable --dependency=afterok:"$prepare_job" --export=ALL scripts/bash_script/SNN_Bash/run_exp_18_3_cpu_array.bash)
final_job=$(sbatch --parsable --dependency=afterok:"$diag_job" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_18_3_cpu.bash)

printf 'smoke=%s prepare=%s diagnose=%s finalize=%s\n' "$smoke_job" "$prepare_job" "$diag_job" "$final_job"
