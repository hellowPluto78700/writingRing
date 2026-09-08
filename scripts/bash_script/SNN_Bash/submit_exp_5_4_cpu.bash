#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

LOCAL_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_5_4_fusion_cpu_array.bash)
RUN_JOB=$(sbatch --parsable --dependency="afterok:${LOCAL_JOB}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_5_4_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --dependency="afterok:${RUN_JOB}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_5_4_cpu.bash)

echo "Exp5.4 submitted"
echo "  fusion-input preparation: ${LOCAL_JOB} (5 tasks)"
echo "  fusion train/evaluate:      ${RUN_JOB} (30 train/evaluate runs)"
echo "  aggregation finalizer:      ${FINAL_JOB}"
echo "Dependencies: afterok:${LOCAL_JOB} -> afterok:${RUN_JOB} -> ${FINAL_JOB}"
