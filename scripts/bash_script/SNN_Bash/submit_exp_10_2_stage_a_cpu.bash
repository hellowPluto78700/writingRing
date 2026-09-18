#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

prepare_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_10_2_cpu.bash)
baseline_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_10_2_baseline_cpu_array.bash)
replay_job=$(sbatch --parsable --dependency="afterok:${baseline_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_10_2_replay_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${replay_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_10_2_stage_a_cpu.bash)

echo "Exp10.2 prepare: ${prepare_job}"
echo "Exp10.2 baseline train array: ${baseline_job}"
echo "Exp10.2 frozen replay array: ${replay_job}"
echo "Exp10.2 Stage-A finalizer: ${finalizer_job}"
