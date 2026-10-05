#!/usr/bin/env bash
#SBATCH --array=0-65%33
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -eo pipefail
ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$ROOT/core_benchmark_v1/extensions/output_spike_tsce_beta/slurm/common.bash"
python -m core_benchmark_v1.extensions.output_spike_tsce_beta --results "$RESULTS" run --task-id "$SLURM_ARRAY_TASK_ID"
