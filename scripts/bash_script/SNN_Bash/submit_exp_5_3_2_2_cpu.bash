#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

LOCAL_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_5_3_2_2_local_cpu_array.bash)
RUN_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${LOCAL_JOB}" scripts/bash_script/SNN_Bash/run_exp_5_3_2_2_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${RUN_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_5_3_2_2_cpu.bash)

echo "Exp5.3.2.2 frozen-WHAT/baseline validation: ${LOCAL_JOB} (5 seeds)"
echo "Exp5.3.2.2 RSNN width array: ${RUN_JOB} (25 runs, max 25 concurrent CPU tasks)"
echo "Exp5.3.2.2 finalizer: ${FINAL_JOB} (afterok on all 25 runs)"
