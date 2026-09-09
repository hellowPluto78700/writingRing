#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_0_1_1_local234_wholecount_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_0_1_1_local234_wholecount_cpu.bash)

echo "Experiment 0.1.1 array: ${ARRAY_JOB} (20 runs, max 20 concurrent CPU tasks)"
echo "Experiment 0.1.1 finalizer: ${FINAL_JOB} (afterok on array)"
