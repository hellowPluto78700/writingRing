#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source "$(dirname "$0")/common.bash"
python -m core_benchmark_v1.extensions.output_residual_leakage --results "$RESULTS" finalize