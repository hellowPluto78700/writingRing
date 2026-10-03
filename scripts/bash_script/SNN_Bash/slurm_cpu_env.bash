#!/usr/bin/env bash
# Shared Unity CPU experiment environment bootstrap.
# Intentionally avoids sourcing /etc/profile wholesale because Unity's
# z05-lmod-purge.sh can fail in batch shells when LMOD_DO_PURGE is unset.

_slurm_cpu_env_restore_nounset=0
case "$-" in
  *u*)
    _slurm_cpu_env_restore_nounset=1
    set +u
    ;;
esac

if ! type module >/dev/null 2>&1; then
  # Minimal Lmod initialization used on Unity.
  # Lmod's own init scripts may read optional variables such as FPATH, so
  # nounset is disabled only while the environment bootstrap executes.
  # shellcheck disable=SC1091
  source /etc/profile.d/z00-lmod-profile.sh
  # shellcheck disable=SC1091
  source /etc/profile.d/z03-lmod-vars.sh
  # shellcheck disable=SC1091
  source /etc/profile.d/z06-lmod-modulepath.sh
fi

module load conda/latest
eval "$(conda shell.bash hook)"
if ! conda activate writingring-gpu 2>/dev/null; then
  conda activate writingring-viz
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

if [ "$_slurm_cpu_env_restore_nounset" -eq 1 ]; then
  set -u
fi
unset _slurm_cpu_env_restore_nounset
