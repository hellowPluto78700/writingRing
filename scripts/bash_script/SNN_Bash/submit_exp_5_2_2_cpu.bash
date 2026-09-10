#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

LOCAL_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_5_2_2_local_cpu_array.bash)
RUN_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${LOCAL_JOB}" scripts/bash_script/SNN_Bash/run_exp_5_2_2_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${RUN_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_5_2_2_cpu.bash)

echo "Exp5.2.2 frozen-local preparation: ${LOCAL_JOB} (3 tasks, same Exp5.1 local checkpoints/caches)"
echo "Exp5.2.2 decoder array: ${RUN_JOB} (60 runs, max 50 concurrent one-CPU tasks, afterok:${LOCAL_JOB})"
echo "Exp5.2.2 finalizer: ${FINAL_JOB} (afterok:${RUN_JOB}; aggregation only)"
