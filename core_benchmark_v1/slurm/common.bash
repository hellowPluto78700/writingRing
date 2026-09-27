#!/usr/bin/env bash
# Sourced by every compute-node job; never depend on the submit shell's Python.
set -eo pipefail
# Parent launchers enable nounset. Unity's /etc/profile reads variables that may
# legitimately be unset, so suspend nounset only while the system profile loads.
set +u
source /etc/profile
set -u
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
module load conda/latest
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx writingring-gpu; then
  conda activate writingring-gpu
elif conda env list | awk '{print $1}' | grep -qx writingring-viz; then
  conda activate writingring-viz
else
  echo 'Neither writingring-gpu nor writingring-viz exists on this node' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CORE_CONFIG="${CORE_CONFIG:-$REPO_ROOT/core_benchmark_v1/00_protocol/default.json}"
CORE_RESULTS="${CORE_RESULTS:-$REPO_ROOT/core_benchmark_v1/results/main}"
export REPO_ROOT CORE_CONFIG CORE_RESULTS
