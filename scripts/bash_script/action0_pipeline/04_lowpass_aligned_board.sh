#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="data"
ACTION=1
POST_ENCODE_TRANSFORM="none"  #none, AbsRectify
OUTPUT_BASE="outputs/action1_rectified"
PIPELINE_MODE="overwrite"  # "continue" or "overwrite"

export DATA_ROOT
export ACTION
export POST_ENCODE_TRANSFORM
export PIPELINE_MODE

# shellcheck source=_common.bash
source "$SCRIPT_DIR/_common.bash"

run_action0_pipeline low-pass aligned-board-events
