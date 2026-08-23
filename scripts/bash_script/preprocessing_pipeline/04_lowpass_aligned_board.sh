#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="data"
ACTION=0
POST_ENCODE_TRANSFORM="PolaritySplitAbs"  #none, AbsRectify, PolaritySplitAbs
RESAMPLE_RATE_HZ="${RESAMPLE_RATE_HZ:-64}"  # none, or a lower rate such as 64
OUTPUT_BASE="outputs/action0_wavelets_0e5_1_2_4_8_sr_64"
PIPELINE_MODE="${PIPELINE_MODE:-overwrite}"  # "continue" or "overwrite"
# Five Custom Wavelet bands. Leave unset to use custom_wavelet.json defaults.
#ENCODER_FREQUENCIES_HZ="1 2 4 8 16"
ENCODER_FREQUENCIES_HZ="${ENCODER_FREQUENCIES_HZ:-}"

export DATA_ROOT
export ACTION
export POST_ENCODE_TRANSFORM
export RESAMPLE_RATE_HZ
export PIPELINE_MODE
export ENCODER_FREQUENCIES_HZ

# shellcheck source=_common.bash
source "$SCRIPT_DIR/_common.bash"

run_action0_pipeline low-pass aligned-board-events
