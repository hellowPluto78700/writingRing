#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

LOCAL_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_5_3_2_3_local_cpu_array.bash)
RUN_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${LOCAL_JOB}" scripts/bash_script/SNN_Bash/run_exp_5_3_2_3_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${RUN_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_5_3_2_3_cpu.bash)

echo "Exp5.3.2.3 preparation array: ${LOCAL_JOB}"
echo "Exp5.3.2.3 15 train/evaluate runs: ${RUN_JOB}"
echo "Exp5.3.2.3 aggregation-only finalizer: ${FINAL_JOB}"
