#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -eo pipefail
ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$ROOT/core_benchmark_v1/extensions/output_spike_drain/slurm/common.bash"
python -m core_benchmark_v1.extensions.output_spike_drain --results "$RESULTS" finalize
