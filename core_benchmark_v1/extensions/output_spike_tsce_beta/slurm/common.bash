#!/usr/bin/env bash
set -eo pipefail
ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export REPO_ROOT="$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/bash_script/SNN_Bash/slurm_cpu_env.bash"
set -u
cd "$REPO_ROOT"
RESULTS="${CORE_SPIKE_TSCE_BETA_RESULTS:-core_benchmark_v1/results/output_spike_tsce_beta_v1}"
export RESULTS
