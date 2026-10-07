#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

smoke_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/smoke_exp_19_cpu.bash)
prepare_job=$(sbatch --parsable --dependency=afterok:"$smoke_job" --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_19_cpu.bash)
cal_job=$(sbatch --parsable --dependency=afterok:"$prepare_job" --export=ALL scripts/bash_script/SNN_Bash/run_exp_19_calibration_cpu_array.bash)
select_job=$(sbatch --parsable --dependency=afterok:"$cal_job" --export=ALL scripts/bash_script/SNN_Bash/select_exp_19_cpu.bash)
final_job=$(sbatch --parsable --dependency=afterok:"$select_job" --export=ALL scripts/bash_script/SNN_Bash/run_exp_19_final_cpu_array.bash)
aggregate_job=$(sbatch --parsable --dependency=afterok:"$final_job" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_19_cpu.bash)

printf 'smoke=%s prepare=%s calibration=%s select=%s final=%s finalize=%s\n' \
  "$smoke_job" "$prepare_job" "$cal_job" "$select_job" "$final_job" "$aggregate_job"

[executed on device: acd20ea31325 (425a23ad-a806-44e3-abed-ce7b563d3969)]