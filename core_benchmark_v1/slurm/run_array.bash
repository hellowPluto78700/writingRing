#!/usr/bin/env bash
#SBATCH --job-name=core_v1_run
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
set -euo pipefail
source "${REPO_ROOT:?}/core_benchmark_v1/slurm/common.bash"
python -u -m core_benchmark_v1 --results "$CORE_RESULTS" run \
  --phase "${CORE_PHASE:?}" --task-id "${SLURM_ARRAY_TASK_ID:?}"
