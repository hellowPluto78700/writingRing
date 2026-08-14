#!/usr/bin/env bash
set -Eeuo pipefail

# ------------------------------------------------------------
# Re-pad two actions to the same target length.
#
# Logic:
#   1. Read current target_length from both padded summaries.
#   2. Take the larger target_length.
#   3. Re-run padding ONLY for the action with the smaller target.
#   4. Read from variable-length segmentation/, NOT segmentation_padded/.
#
# Usage:
#
#   bash scripts/bash_script/repad_actions_to_common_target.sh
#
# Or explicitly provide the two pipeline roots:
#
#   bash scripts/bash_script/repad_actions_to_common_target.sh \
#       outputs/action0_rectified/low-pass/aligned-board-events \
#       outputs/action1_rectified/low-pass/aligned-board-events
#
# Optional environment variables:
#
#   CONDA_ENV=writingring-gpu
#   SAMPLING_RATE=200
#   PADDING_VALUE=0.0
# ------------------------------------------------------------

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

ACTION0_ROOT="${1:-outputs/action0_rectified/low-pass/aligned-board-events}"
ACTION1_ROOT="${2:-outputs/action1_rectified/low-pass/aligned-board-events}"

CONDA_ENV="${CONDA_ENV:-writingring-gpu}"
SAMPLING_RATE="${SAMPLING_RATE:-200}"
PADDING_VALUE="${PADDING_VALUE:-0.0}"

ACTION0_SEGMENT_ROOT="$ACTION0_ROOT/segmentation"
ACTION1_SEGMENT_ROOT="$ACTION1_ROOT/segmentation"

ACTION0_PADDED_ROOT="$ACTION0_ROOT/segmentation_padded"
ACTION1_PADDED_ROOT="$ACTION1_ROOT/segmentation_padded"

ACTION0_SUMMARY="$ACTION0_PADDED_ROOT/padding_dataset_summary.json"
ACTION1_SUMMARY="$ACTION1_PADDED_ROOT/padding_dataset_summary.json"


die() {
    echo "error: $*" >&2
    exit 1
}


read_target_length() {
    local summary="$1"

    [[ -f "$summary" ]] || die "padding summary not found: $summary"

    python3 - "$summary" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

target = data.get("target_length")

if not isinstance(target, int) or target <= 0:
    raise SystemExit(
        f"invalid or missing target_length in {path}: {target!r}"
    )

print(target)
PY
}


echo "Reading current padding targets..."

ACTION0_TARGET="$(read_target_length "$ACTION0_SUMMARY")"
ACTION1_TARGET="$(read_target_length "$ACTION1_SUMMARY")"

echo
echo "Action 0 padding target: $ACTION0_TARGET"
echo "Action 1 padding target: $ACTION1_TARGET"
echo


# Already identical: nothing to do.
if (( ACTION0_TARGET == ACTION1_TARGET )); then
    echo "Padding targets are already identical."
    echo "Common target length: $ACTION0_TARGET"
    exit 0
fi


# Decide which action needs re-padding.
if (( ACTION0_TARGET < ACTION1_TARGET )); then
    SMALL_ACTION="action0"
    OLD_TARGET="$ACTION0_TARGET"
    NEW_TARGET="$ACTION1_TARGET"

    SEGMENT_ROOT="$ACTION0_SEGMENT_ROOT"
    PADDED_ROOT="$ACTION0_PADDED_ROOT"
else
    SMALL_ACTION="action1"
    OLD_TARGET="$ACTION1_TARGET"
    NEW_TARGET="$ACTION0_TARGET"

    SEGMENT_ROOT="$ACTION1_SEGMENT_ROOT"
    PADDED_ROOT="$ACTION1_PADDED_ROOT"
fi


[[ -d "$SEGMENT_ROOT" ]] || die "segmentation root not found: $SEGMENT_ROOT"

echo "$SMALL_ACTION has the smaller padding target."
echo "Old target: $OLD_TARGET"
echo "New target: $NEW_TARGET"
echo
echo "Input : $SEGMENT_ROOT"
echo "Output: $PADDED_ROOT"
echo


conda run --no-capture-output -n "$CONDA_ENV" \
    python scripts/pad_segmented_imu.py \
    --input-root "$SEGMENT_ROOT" \
    --output-root "$PADDED_ROOT" \
    --target-length "$NEW_TARGET" \
    --sampling-rate "$SAMPLING_RATE" \
    --padding-value "$PADDING_VALUE" \
    --overwrite


echo
echo "Re-padding completed."


# Verify the newly published target.
UPDATED_TARGET="$(read_target_length "$PADDED_ROOT/padding_dataset_summary.json")"

if (( UPDATED_TARGET != NEW_TARGET )); then
    die "verification failed: expected target=$NEW_TARGET, got target=$UPDATED_TARGET"
fi

echo "Verified $SMALL_ACTION target length: $UPDATED_TARGET"
echo
echo "Final common padding target: $NEW_TARGET"
echo "Action 0: $NEW_TARGET"
echo "Action 1: $NEW_TARGET"

