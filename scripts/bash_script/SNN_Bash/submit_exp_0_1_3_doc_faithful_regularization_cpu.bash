#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${WRITINGRING_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p outputs

BASE_RUNS="notebooks/artifacts/experiment_0_1_general_comparison/general_comparison_v1/runs.csv"
BASE_BASELINES="notebooks/artifacts/experiment_0_1_general_comparison/general_comparison_v1/baseline_results.csv"
if [[ ! -f "$BASE_RUNS" || ! -f "$BASE_BASELINES" ]]; then
    echo "Exp0.1 finalized artifacts are required before Exp0.1.3." >&2
    echo "Missing one of: $BASE_RUNS, $BASE_BASELINES" >&2
    exit 1
fi

ARRAY_JOB=$(sbatch --parsable \
    scripts/bash_script/SNN_Bash/run_exp_0_1_3_doc_faithful_regularization_cpu_array.bash)
FINALIZER_JOB=$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB}" \
    scripts/bash_script/SNN_Bash/finalize_exp_0_1_3_doc_faithful_regularization_cpu.bash)

echo "Exp0.1.3 array job: ${ARRAY_JOB}"
echo "Exp0.1.3 finalizer job: ${FINALIZER_JOB} (afterok:${ARRAY_JOB})"
