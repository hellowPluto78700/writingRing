#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_0_1_3_doc_faithful_regularization_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_0_1_3_doc_faithful_regularization_cpu.bash)

echo "Exp0.1.3 doc-faithful binary direct-SNN array: ${ARRAY_JOB} (30 new reg-on runs, max 30 concurrent CPU tasks)"
echo "Exp0.1.3 finalizer: ${FINAL_JOB} (afterok on regularized array; frozen Exp0.1 artifacts are reused)"
