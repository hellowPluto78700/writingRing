#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_0_1_5_gradient_calibrated_all_fixes_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_0_1_5_gradient_calibrated_all_fixes_cpu.bash)

echo "Exp0.1.5 gradient-calibrated all-fixes array: ${ARRAY_JOB} (90 runs, max 50 concurrent CPU tasks)"
echo "Exp0.1.5 finalizer: ${FINAL_JOB} (afterok; aggregation only; frozen Exp0.1 no-reg is reused)"
