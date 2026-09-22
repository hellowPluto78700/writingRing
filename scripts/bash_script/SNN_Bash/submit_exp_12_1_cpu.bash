#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
SLURM_MAX_CONCURRENCY="${SLURM_MAX_CONCURRENCY:-24}"
if (( SLURM_MAX_CONCURRENCY < 1 || SLURM_MAX_CONCURRENCY > 50 )); then
  echo "SLURM_MAX_CONCURRENCY must be in [1, 50]" >&2
  exit 2
fi

cd "$REPO_ROOT"
export REPO_ROOT

array_job=$(sbatch --parsable --export=ALL \
  --array="0-23%${SLURM_MAX_CONCURRENCY}" \
  scripts/bash_script/SNN_Bash/run_exp_12_1_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${array_job}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_12_1_cpu.bash)

echo "Exp12.1 24-run CPU array: ${array_job} (max concurrent=${SLURM_MAX_CONCURRENCY})"
echo "Exp12.1 finalizer: ${finalizer_job}"
