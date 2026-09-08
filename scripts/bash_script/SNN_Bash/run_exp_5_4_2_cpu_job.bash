#!/usr/bin/env bash
set -euo pipefail

# Slurm --wrap generates a /bin/sh wrapper on Unity. This worker is always
# launched explicitly with `bash -l` so the cluster module initialization is
# available before loading Conda.
if ! type module >/dev/null 2>&1; then
  for init_script in \
    /etc/profile.d/modules.sh \
    /usr/share/Modules/init/bash \
    /etc/profile.d/lmod.sh
  do
    if [[ -r "$init_script" ]]; then
      # Cluster init scripts are not guaranteed to be nounset-safe.
      set +u
      # shellcheck disable=SC1090
      source "$init_script"
      set -u
      break
    fi
  done
fi

if ! type module >/dev/null 2>&1; then
  echo "ERROR: environment-modules/Lmod is unavailable in the Slurm job shell." >&2
  echo "Expected a login bash shell or a standard Modules init script." >&2
  exit 10
fi

module load conda/latest

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'module load conda/latest' did not put conda on PATH." >&2
  exit 11
fi

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx writingring-gpu; then
  conda activate writingring-gpu
elif conda env list | awk '{print $1}' | grep -qx writingring-viz; then
  conda activate writingring-viz
else
  echo "ERROR: neither writingring-gpu nor writingring-viz Conda environment exists." >&2
  exit 12
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment-subcommand> [args ...]" >&2
  exit 2
fi

SUBCOMMAND="$1"
shift
exec python -u -m scripts.experiment_5_4_2_phase_conditioned_readout "$SUBCOMMAND" "$@"
