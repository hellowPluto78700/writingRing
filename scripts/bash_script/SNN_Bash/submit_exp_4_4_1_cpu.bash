#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_4_4_1_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_4_4_1_cpu.bash)

echo "Experiment 4.4.1 training/evaluation array: ${ARRAY_JOB} (50 runs, max 50 concurrent CPU tasks)"
echo "Experiment 4.4.1 finalizer: ${FINAL_JOB} (afterok on array)"
