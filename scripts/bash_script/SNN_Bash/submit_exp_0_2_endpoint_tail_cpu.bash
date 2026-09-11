#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_0_2_endpoint_tail_cpu_array.bash)
FROZEN_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/eval_exp_0_2_frozen_exp01_cpu.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}:${FROZEN_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_0_2_endpoint_tail_cpu.bash)

echo "Exp0.2 new-run array: ${ARRAY_JOB} (72 runs, max 50 concurrent CPU tasks)"
echo "Exp0.2 frozen Exp0.1 evaluation: ${FROZEN_JOB} (18 checkpoints, sequential single CPU)"
echo "Exp0.2 finalizer: ${FINAL_JOB} (afterok both jobs; aggregation only)"
