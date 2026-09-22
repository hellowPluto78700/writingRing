#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

dense_job=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_12_0_dense_cpu_array.bash)
post_job=$(sbatch --parsable --dependency="afterok:${dense_job}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_12_0_posthoc_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${post_job}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_12_0_cpu.bash)

echo "Exp12.0 dense 36-run CPU array: ${dense_job}"
echo "Exp12.0 post-hoc 27-run CPU array: ${post_job}"
echo "Exp12.0 finalizer: ${finalizer_job}"
