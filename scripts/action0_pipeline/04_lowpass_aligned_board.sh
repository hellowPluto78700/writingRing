#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="data"
ACTION=0
POST_ENCODE_TRANSFORM="AbsRectify"  #None, AbsRectify
OUTPUT_BASE="outputs/action0_rectified"
#PIPELINE_MODE = "continue"  # "continue" or "overwrite"

export DATA_ROOT
export ACTION
export POST_ENCODE_TRANSFORM

# shellcheck source=_common.bash
source "$SCRIPT_DIR/_common.bash"

run_action0_pipeline low-pass aligned-board-events
