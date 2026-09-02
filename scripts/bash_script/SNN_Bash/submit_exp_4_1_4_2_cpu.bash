#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

EXP41_ARRAY=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_4_1_cpu_array.bash)
EXP42_ARRAY=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_4_2_cpu_array.bash)
EXP41_FINAL=$(sbatch --parsable --export=ALL --dependency="afterok:${EXP41_ARRAY}" scripts/bash_script/SNN_Bash/finalize_exp_4_1_cpu.bash)
EXP42_FINAL=$(sbatch --parsable --export=ALL --dependency="afterok:${EXP42_ARRAY}" scripts/bash_script/SNN_Bash/finalize_exp_4_2_cpu.bash)

echo "Experiment 4.1 array job: ${EXP41_ARRAY} (max 20 concurrent CPU tasks)"
echo "Experiment 4.2 array job: ${EXP42_ARRAY} (max 30 concurrent CPU tasks)"
echo "Combined experiment concurrency cap: 50 CPU tasks"
echo "Experiment 4.1 finalizer job: ${EXP41_FINAL}"
echo "Experiment 4.2 finalizer job: ${EXP42_FINAL}"
