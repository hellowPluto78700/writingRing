#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

TRAIN_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_4_0_6_cpu_array.bash)
BASE_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/probe_exp_4_0_6_baseline_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${TRAIN_JOB}:${BASE_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_4_0_6_cpu.bash)

echo "Experiment 4.0.6 training array: ${TRAIN_JOB} (45 new runs, max 45 concurrent CPU tasks)"
echo "Experiment 4.0.6 frozen beta0 probe array: ${BASE_JOB} (15 tasks, max 15 concurrent CPU tasks)"
echo "Experiment 4.0.6 finalizer: ${FINAL_JOB} (afterok on both arrays)"
