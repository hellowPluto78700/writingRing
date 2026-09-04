#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_5_0_1_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_5_0_1_cpu.bash)

echo "Experiment 5.0.1 analog-head control array: ${ARRAY_JOB} (9 runs, max 9 concurrent CPU tasks)"
echo "Experiment 5.0.1 finalizer: ${FINAL_JOB} (afterok on array)"
