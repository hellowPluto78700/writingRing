#!/usr/bin/env bash
set -Eeuo pipefail

# Pass one or more completed action roots explicitly. With no arguments, use
# the current full-rate Action 0/Action 1 pair as the convenience default.
if (($# == 0)); then
  DATASET_ROOTS=(
    "${ACTION0_ROOT:-outputs/action0_wavelets_0e5_1_2_4_8/low-pass/aligned-board-events}"
    "${ACTION1_ROOT:-outputs/action1_wavelets_0e5_1_2_4_8/low-pass/aligned-board-events}"
  )
else
  DATASET_ROOTS=("$@")
fi

for dataset_root in "${DATASET_ROOTS[@]}"; do
  python scripts/reconstruct_padded_spike_accel.py \
    "$dataset_root" \
    --overwrite
done
