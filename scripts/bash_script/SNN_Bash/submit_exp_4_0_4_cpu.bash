#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB_ID="$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_4_0_4_cpu_array.bash)"
FINAL_JOB_ID="$(sbatch --parsable --dependency=afterok:${ARRAY_JOB_ID} --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_4_0_4_cpu.bash)"

echo "Exp4.0.4 probe array job: ${ARRAY_JOB_ID}"
echo "Exp4.0.4 finalizer job: ${FINAL_JOB_ID} (afterok:${ARRAY_JOB_ID})"
