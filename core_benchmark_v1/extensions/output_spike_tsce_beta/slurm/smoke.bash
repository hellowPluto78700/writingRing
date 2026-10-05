#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -eo pipefail
ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$ROOT/core_benchmark_v1/extensions/output_spike_tsce_beta/slurm/common.bash"
python -c 'import sys, torch, numpy, pandas; assert sys.version_info[:2] == (3, 11); print(sys.version); print(torch.__version__)'
python -m core_benchmark_v1.extensions.output_spike_tsce_beta --results "$RESULTS" smoke
