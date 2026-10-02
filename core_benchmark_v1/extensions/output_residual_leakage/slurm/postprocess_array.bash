#!/usr/bin/env bash
#SBATCH --array=0-65%50
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$ROOT/core_benchmark_v1/extensions/output_residual_leakage/slurm/common.bash"
python -m core_benchmark_v1.extensions.output_residual_leakage --results "$RESULTS" postprocess --task-id "$SLURM_ARRAY_TASK_ID"
