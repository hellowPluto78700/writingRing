#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_4_1_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_4_1_cpu.bash)

echo "Experiment 4.1 position-aware RSNN array job: ${ARRAY_JOB}"
echo "Experiment 4.1 finalizer job: ${FINAL_JOB}"
