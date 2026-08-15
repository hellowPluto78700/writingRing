#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="data"
ACTION=0
POST_ENCODE_TRANSFORM="AbsRectify"  #None, AbsRectify
OUTPUT_BASE="outputs/action0_rectified"
# Five Custom Wavelet bands. Leave unset to use custom_wavelet.json defaults.
# Example: ENCODER_FREQUENCIES_HZ="1 2 4 8 16"
ENCODER_FREQUENCIES_HZ="${ENCODER_FREQUENCIES_HZ:-}"

export DATA_ROOT
export ACTION
export POST_ENCODE_TRANSFORM
export ENCODER_FREQUENCIES_HZ
#export OUTPUT_BASE
# shellcheck source=_common.bash
source "$SCRIPT_DIR/_common.bash"

run_action0_pipeline low-pass label
