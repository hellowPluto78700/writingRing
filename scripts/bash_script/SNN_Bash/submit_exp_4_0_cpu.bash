#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_4_0_cpu_array.bash)
LINEAR_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_4_0_linear_baseline_cpu.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}:${LINEAR_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_4_0_cpu.bash)

echo "Experiment 4.0 SNN array job: ${ARRAY_JOB}"
echo "Experiment 4.0 Linear baseline job: ${LINEAR_JOB}"
echo "Experiment 4.0 finalizer job: ${FINAL_JOB}"
