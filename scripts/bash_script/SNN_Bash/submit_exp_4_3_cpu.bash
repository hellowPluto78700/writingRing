#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

FF_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_4_3_ff_cpu_array.bash)
RSNN_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/eval_exp_4_3_rsnn_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${FF_JOB}:${RSNN_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_4_3_cpu.bash)

echo "Experiment 4.3 FF training array: ${FF_JOB} (5 paired runs, max 5 concurrent CPU tasks)"
echo "Experiment 4.3 frozen RSNN evaluation array: ${RSNN_JOB} (5 tasks, max 5 concurrent CPU tasks)"
echo "Experiment 4.3 finalizer: ${FINAL_JOB} (afterok on both arrays)"
