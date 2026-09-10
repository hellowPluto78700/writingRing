#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_0_1_4_parallel_regularization_ablation_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_0_1_4_parallel_regularization_ablation_cpu.bash)

echo "Exp0.1.4 regularization ablation array: ${ARRAY_JOB} (180 runs, 30 epochs each, max 50 concurrent CPU tasks)"
echo "Exp0.1.4 finalizer: ${FINAL_JOB} (afterok:${ARRAY_JOB}; aggregation only)"
