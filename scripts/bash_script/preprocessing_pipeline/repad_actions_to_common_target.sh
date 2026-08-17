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
#   ENCODER_FREQUENCIES_HZ="1 2 4 8 16"  # validates existing inputs only
# ------------------------------------------------------------

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

ACTION0_ROOT="${1:-outputs/action0_rectified_default_wavelets_absrectify/low-pass/aligned-board-events}"
ACTION1_ROOT="${2:-outputs/action1_rectified_default_wavelets_absrectify/low-pass/aligned-board-events}"

CONDA_ENV="${CONDA_ENV:-writingring-gpu}"
SAMPLING_RATE="${SAMPLING_RATE:-200}"
PADDING_VALUE="${PADDING_VALUE:-0.0}"
ENCODER_FREQUENCIES_HZ="${ENCODER_FREQUENCIES_HZ:-}"

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


validate_encoder_identity() {
    local summary_a="$1"
    local summary_b="$2"
    local requested_frequencies="$3"

    python3 - "$summary_a" "$summary_b" "$requested_frequencies" <<'PY'
import json
import math
import sys
from pathlib import Path


def identity(path: Path) -> tuple[dict[str, object], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = payload.get("spike_encoder")
    fingerprint = payload.get("spike_encoder_spec_sha256")
    if not isinstance(spec, dict) or not isinstance(fingerprint, str):
        raise SystemExit(f"missing spike encoder identity in {path}; regenerate from encode")
    return spec, fingerprint


first_path, second_path = map(Path, sys.argv[1:3])
requested = sys.argv[3]
first_spec, first_hash = identity(first_path)
second_spec, second_hash = identity(second_path)
if first_hash != second_hash or first_spec != second_spec:
    raise SystemExit(
        "encoder identity mismatch between padding roots: "
        f"{first_path}={first_hash}, {second_path}={second_hash}"
    )
if requested:
    try:
        values = [float(value) for value in requested.split()]
    except ValueError as error:
        raise SystemExit("ENCODER_FREQUENCIES_HZ must contain numeric values") from error
    if len(values) != 5 or not all(math.isfinite(value) and value > 0 for value in values):
        raise SystemExit("ENCODER_FREQUENCIES_HZ must contain exactly five positive finite values")
    if values != first_spec.get("frequencies_hz"):
        raise SystemExit(
            "ENCODER_FREQUENCIES_HZ does not match existing encoded inputs; "
            "repad does not re-encode data"
        )
PY
}


echo "Reading current padding targets..."

validate_encoder_identity "$ACTION0_SUMMARY" "$ACTION1_SUMMARY" "$ENCODER_FREQUENCIES_HZ"

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
