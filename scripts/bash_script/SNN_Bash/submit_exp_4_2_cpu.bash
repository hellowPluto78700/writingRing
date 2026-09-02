#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_4_2_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_4_2_cpu.bash)

echo "Experiment 4.2 continuous-control array job: ${ARRAY_JOB}"
echo "Experiment 4.2 finalizer job: ${FINAL_JOB}"
