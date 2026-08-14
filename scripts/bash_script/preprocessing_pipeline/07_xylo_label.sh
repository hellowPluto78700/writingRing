#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.bash
source "$SCRIPT_DIR/_common.bash"

run_action0_pipeline xylo-rotate-and-remove-gravity label
