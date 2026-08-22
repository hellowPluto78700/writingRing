#!/usr/bin/env bash
set -Eeuo pipefail

# Thin wrapper for:
#   scripts/build_polarity_split_variant.py
#
# Usage:
#
#   bash scripts/bash_script/preprocessing_pipeline/build_polarity_split_variant.bash \
#       SOURCE_COMBINATION_ROOT \
#       [OUTPUT_ROOT]
#
# Default output:
#   <SOURCE_COMBINATION_ROOT>/segmentation_padded_polarity_split
#
# Environment:
#   OVERWRITE_DEST=0|1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

SOURCE_ROOT="${1:-outputs/action1_wavelets_1_2_4_8_16/low-pass/aligned-board-events}"
OUTPUT_ROOT="${2:-}"
OVERWRITE_DEST="${OVERWRITE_DEST:-0}"

fail() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  build_polarity_split_variant.bash SOURCE_COMBINATION_ROOT [OUTPUT_ROOT]

SOURCE_COMBINATION_ROOT must be a completed WritingRing Action 0 or Action 1
combination root whose basename is:
  label
or:
  aligned-board-events

The source must contain the canonical 21-channel signed SpikeIMU padded
representation and must not already use AbsRectify.

Default OUTPUT_ROOT:
  SOURCE_COMBINATION_ROOT/segmentation_padded_polarity_split

Environment:
  OVERWRITE_DEST=0|1
EOF
}

[[ -n "$SOURCE_ROOT" ]] || {
    usage
    exit 2
}

case "$OVERWRITE_DEST" in
    0|1) ;;
    *) fail "OVERWRITE_DEST must be 0 or 1" ;;
esac

[[ -d "$SOURCE_ROOT" ]] ||
    fail "source root is not a directory: $SOURCE_ROOT"

PYTHON_HELPER="$REPO_ROOT/scripts/build_polarity_split_variant.py"

[[ -f "$PYTHON_HELPER" ]] ||
    fail "missing Python helper: $PYTHON_HELPER"

if command -v conda >/dev/null 2>&1; then
    if conda env list | grep -q '^writingring-gpu '; then
        PYTHON_CMD=(conda run --no-capture-output -n writingring-gpu python)
    elif conda env list | grep -q '^writingring-viz '; then
        PYTHON_CMD=(conda run --no-capture-output -n writingring-viz python)
    else
        PYTHON_CMD=(python)
    fi
else
    PYTHON_CMD=(python)
fi

ARGS=(
    "$PYTHON_HELPER"
    --source-root "$SOURCE_ROOT"
)

if [[ -n "$OUTPUT_ROOT" ]]; then
    ARGS+=(--output-root "$OUTPUT_ROOT")
fi

if [[ "$OVERWRITE_DEST" == "1" ]]; then
    ARGS+=(--overwrite)
fi

cd "$REPO_ROOT"
"${PYTHON_CMD[@]}" "${ARGS[@]}"
