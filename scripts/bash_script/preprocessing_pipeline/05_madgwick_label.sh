#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Five Custom Wavelet bands. Leave unset to use custom_wavelet.json defaults.
# Example: ENCODER_FREQUENCIES_HZ="1 2 4 8 16"
ENCODER_FREQUENCIES_HZ="${ENCODER_FREQUENCIES_HZ:-}"
export ENCODER_FREQUENCIES_HZ

# shellcheck source=_common.bash
source "$SCRIPT_DIR/_common.bash"

run_action0_pipeline madgwick label
