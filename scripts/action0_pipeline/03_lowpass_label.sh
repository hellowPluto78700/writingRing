#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.bash
source "$SCRIPT_DIR/_common_fixed.bash"

run_action0_pipeline low-pass label
