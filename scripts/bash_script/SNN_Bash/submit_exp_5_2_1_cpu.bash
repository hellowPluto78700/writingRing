#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

RUN_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_5_2_1_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${RUN_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_5_2_1_cpu.bash)

echo "Exp5.2.1 FF factorial array: ${RUN_JOB} (20 runs, max 20 concurrent one-CPU tasks)"
echo "Exp5.2.1 finalizer: ${FINAL_JOB} (afterok on all 20 tasks)"
echo "Frozen Local L2 caches are reused from finalized Exp5.2 artifacts; each task validates its source cache before training."
