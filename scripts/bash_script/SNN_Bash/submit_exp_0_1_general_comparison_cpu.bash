#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_0_1_general_comparison_cpu_array.bash)
BASELINE_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_0_1_general_comparison_baselines_cpu.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}:${BASELINE_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_0_1_general_comparison_cpu.bash)

echo "Exp0.1 General Comparison SNN array: ${ARRAY_JOB} (70 runs, max 50 concurrent CPU tasks)"
echo "Exp0.1 General Comparison raw baselines: ${BASELINE_JOB}"
echo "Exp0.1 General Comparison finalizer: ${FINAL_JOB} (afterok on array + baselines)"
