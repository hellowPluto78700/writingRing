#!/usr/bin/env bash
#SBATCH --job-name=core_v1_prepare
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -euo pipefail
source "${REPO_ROOT:?}/core_benchmark_v1/slurm/common.bash"
python -u -m core_benchmark_v1 --config "$CORE_CONFIG" --results "$CORE_RESULTS" prepare
