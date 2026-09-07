#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

LOCAL_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_5_3_1_local_cpu_array.bash)
RUN_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${LOCAL_JOB}" scripts/bash_script/SNN_Bash/run_exp_5_3_1_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${RUN_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_5_3_1_cpu.bash)

echo "Exp5.3.1 frozen-local preparation: ${LOCAL_JOB} (5 seeds)"
echo "Exp5.3.1 factorized attribution array: ${RUN_JOB} (20 runs, max 20 concurrent CPU tasks)"
echo "Exp5.3.1 finalizer: ${FINAL_JOB} (afterok on all 20 runs)"
