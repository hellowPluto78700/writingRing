#!/usr/bin/env bash
#SBATCH --array=0-65%50
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
set -euo pipefail
source "$(dirname "$0")/common.bash"
python -m core_benchmark_v1.extensions.output_residual_leakage --results "$RESULTS" prune --task-id "$SLURM_ARRAY_TASK_ID"