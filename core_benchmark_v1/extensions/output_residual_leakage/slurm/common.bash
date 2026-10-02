#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export REPO_ROOT
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu 2>/dev/null || conda activate writingring-viz
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "$REPO_ROOT"
RESULTS="${CORE_RESIDUAL_RESULTS:-core_benchmark_v1/results/output_residual_leakage_v1}"
