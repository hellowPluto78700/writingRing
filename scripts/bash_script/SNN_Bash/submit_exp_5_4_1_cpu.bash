#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

PREP_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_5_4_1_source_cpu_array.bash)
RUN_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_5_4_1_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --dependency="afterok:${RUN_JOB}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_5_4_1_cpu.bash)

echo "Exp5.4.1 source-validation array: ${PREP_JOB} (5 tasks)"
echo "Exp5.4.1 train/evaluate array: ${RUN_JOB} (5 tasks)"
echo "Exp5.4.1 finalizer: ${FINAL_JOB}"
