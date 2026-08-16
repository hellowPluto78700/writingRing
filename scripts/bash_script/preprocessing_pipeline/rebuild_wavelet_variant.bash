#!/usr/bin/env bash
set -Eeuo pipefail

# Tested against writingRing main commit:
# 8bf56c7264b5707ec2aa81a2adf452a50d8e4d76
#
# This script creates a new five-band Custom Wavelet variant from an existing
# completed pipeline combination root. It intentionally does NOT rerun Ring
# preprocessing or alignment estimation. Alignment is reused only after the
# helper proves that the alignment signal channels (SpikeIMU[:, 15:21]) are
# byte-for-byte equal between the old and new encodings.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
 HELPER_SCRIPT="${HELPER_SCRIPT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)/wavelet_variant_reuse.py}"

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
# Repository checkout root.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"

# Raw data root used by segment_ring_imu.py.
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"

# Existing completed combination root, e.g.:
# outputs/action1_rectified/low-pass/aligned-board-events
SOURCE_COMBINATION_ROOT="${SOURCE_COMBINATION_ROOT:-$REPO_ROOT/outputs/action1_rectified_new_wavelets/low-pass/aligned-board-events}"

# New variant combination root. Must differ from SOURCE_COMBINATION_ROOT.
DEST_COMBINATION_ROOT="${DEST_COMBINATION_ROOT:-$REPO_ROOT/outputs/action1_rectified_default_wavelets/low-pass/aligned-board-events}"

ACTION="${ACTION:-1}"
BOUNDARY_MODE="${BOUNDARY_MODE:-aligned-board-events}"
SAMPLING_RATE="${SAMPLING_RATE:-200}"
ENCODER_FREQUENCIES_HZ="${ENCODER_FREQUENCIES_HZ:-0.5 1 2 4 8}"
ENCODER_SETTINGS="${ENCODER_SETTINGS:-$REPO_ROOT/configs/spike_encoding/custom_wavelet.json}"
# Keep as inherit to preserve the source transform. You may explicitly set
# none or AbsRectify, but doing so will fail validation if it changes any
# non-frequency encoder semantics relative to the source variant.
POST_ENCODE_TRANSFORM="${POST_ENCODE_TRANSFORM:-inherit}"

# Set to 1 to delete an existing destination root before starting.
OVERWRITE_DEST="${OVERWRITE_DEST:-1}"

# If source has segmentation_padded, rebuild destination padding with the exact
# same target length. Set to 0 to skip padding even if the source has it.
REBUILD_PADDING="${REBUILD_PADDING:-1}"

# Segmentation defaults copied from the repository pipeline contract.
SEGMENT_VERIFICATION_DPI="${SEGMENT_VERIFICATION_DPI:-200}"

# ---------------------------------------------------------------------------
# INTERNALS
# ---------------------------------------------------------------------------
fail() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

note() {
    printf '[wavelet-reuse] %s\n' "$1"
}

[[ -d "$REPO_ROOT" ]] || fail "REPO_ROOT is not a directory: $REPO_ROOT"
[[ -f "$REPO_ROOT/scripts/encode_spikes.py" ]] || fail "not a writingRing checkout: $REPO_ROOT"
[[ -f "$HELPER_SCRIPT" ]] || fail "helper script is missing: $HELPER_SCRIPT"
[[ -f "$ENCODER_SETTINGS" ]] || fail "encoder settings are missing: $ENCODER_SETTINGS"
[[ -d "$DATA_ROOT" ]] || fail "DATA_ROOT is not a directory: $DATA_ROOT"
[[ -d "$SOURCE_COMBINATION_ROOT" ]] || fail "source combination root is missing"
[[ "$SOURCE_COMBINATION_ROOT" != "$DEST_COMBINATION_ROOT" ]] || fail "source and destination roots must differ"
case "$BOUNDARY_MODE" in
    label|aligned-board-events) ;;
    *) fail "BOUNDARY_MODE must be label or aligned-board-events" ;;
esac
case "$OVERWRITE_DEST" in
    0|1) ;;
    *) fail "OVERWRITE_DEST must be 0 or 1" ;;
esac
case "$REBUILD_PADDING" in
    0|1) ;;
    *) fail "REBUILD_PADDING must be 0 or 1" ;;
esac

read -r -a ENCODER_FREQUENCIES <<<"$ENCODER_FREQUENCIES_HZ"
[[ "${#ENCODER_FREQUENCIES[@]}" -eq 5 ]] || fail "ENCODER_FREQUENCIES_HZ must contain exactly five values"

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

cd "$REPO_ROOT"

note "preflight source reuse contract"
"${PYTHON_CMD[@]}" "$HELPER_SCRIPT" preflight \
    --source-root "$SOURCE_COMBINATION_ROOT" \
    --action "$ACTION" \
    --boundary-mode "$BOUNDARY_MODE"

if [[ -e "$DEST_COMBINATION_ROOT" ]]; then
    if [[ "$OVERWRITE_DEST" != "1" ]]; then
        fail "destination already exists; set OVERWRITE_DEST=1 to replace it: $DEST_COMBINATION_ROOT"
    fi
    case "$DEST_COMBINATION_ROOT" in
        /|"$REPO_ROOT"|"$SOURCE_COMBINATION_ROOT") fail "refusing unsafe destination removal" ;;
    esac
    note "removing existing destination root"
    rm -rf -- "$DEST_COMBINATION_ROOT"
fi
mkdir -p "$DEST_COMBINATION_ROOT"

SOURCE_PREPROCESS_ROOT="$SOURCE_COMBINATION_ROOT/preprocessedIMU"
DEST_PREPROCESS_ROOT="$DEST_COMBINATION_ROOT/preprocessedIMU"

DEST_SPIKE_OUTPUT_ROOT="$DEST_COMBINATION_ROOT/spikeEncoding"
DEST_SPIKE_ROOT="$DEST_SPIKE_OUTPUT_ROOT/custom-wavelet"
DEST_SEGMENT_ROOT="$DEST_COMBINATION_ROOT/segmentation"
DEST_PADDING_ROOT="$DEST_COMBINATION_ROOT/segmentation_padded"
DEST_ALIGNMENT_OFFSET_ROOT="$DEST_COMBINATION_ROOT/alignment/offsets"

note "reuse preprocessedIMU by symbolic links"

[[ -d "$SOURCE_PREPROCESS_ROOT" ]] || \
    fail "source preprocessedIMU root is missing: $SOURCE_PREPROCESS_ROOT"

rm -rf -- "$DEST_PREPROCESS_ROOT"
mkdir -p "$DEST_PREPROCESS_ROOT"

SYMLINK_COUNT=0

while IFS= read -r -d '' src; do
    rel="${src#"$SOURCE_PREPROCESS_ROOT"/}"
    dst="$DEST_PREPROCESS_ROOT/$rel"

    mkdir -p "$(dirname -- "$dst")"

    ln -s "$(realpath "$src")" "$dst"
    SYMLINK_COUNT=$((SYMLINK_COUNT + 1))
done < <(find "$SOURCE_PREPROCESS_ROOT" -type f -print0)

note "preprocessedIMU symlink tree created: $SYMLINK_COUNT files"

if [[ "$POST_ENCODE_TRANSFORM" == "inherit" ]]; then
    POST_ENCODE_TRANSFORM="$("${PYTHON_CMD[@]}" "$HELPER_SCRIPT" source-transform --source-root "$SOURCE_COMBINATION_ROOT" --action "$ACTION")"
fi
case "$POST_ENCODE_TRANSFORM" in
    none|AbsRectify) ;;
    *) fail "POST_ENCODE_TRANSFORM must be inherit, none, or AbsRectify" ;;
esac

note "encode new Custom Wavelet variant: $ENCODER_FREQUENCIES_HZ Hz, transform=$POST_ENCODE_TRANSFORM"
"${PYTHON_CMD[@]}" scripts/encode_spikes.py \
    --input-root "$DEST_PREPROCESS_ROOT" \
    --pattern '*_preprocessedIMU.npy' \
    --output-root "$DEST_SPIKE_OUTPUT_ROOT" \
    --encoder custom-wavelet \
    --encoder-settings "$ENCODER_SETTINGS" \
    --encoder-frequencies-hz "${ENCODER_FREQUENCIES[@]}" \
    --post-encode-transform "$POST_ENCODE_TRANSFORM" \
    --overwrite

note "validate new encoding before reusing downstream artifacts"
"${PYTHON_CMD[@]}" "$HELPER_SCRIPT" check-encoding \
    --source-root "$SOURCE_COMBINATION_ROOT" \
    --dest-root "$DEST_COMBINATION_ROOT" \
    --action "$ACTION" \
    --frequencies "${ENCODER_FREQUENCIES[@]}"

if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
    note "prove trailing IMU invariance and rebind existing alignment provenance"
    "${PYTHON_CMD[@]}" "$HELPER_SCRIPT" rebind-alignment \
        --source-root "$SOURCE_COMBINATION_ROOT" \
        --dest-root "$DEST_COMBINATION_ROOT" \
        --action "$ACTION" \
        --frequencies "${ENCODER_FREQUENCIES[@]}" \
        --overwrite
fi

note "re-materialize segmentation from the reused boundary/alignment contract"
mkdir -p "$DEST_SEGMENT_ROOT"
USER_COUNT=0
for user_dir in "$DEST_SPIKE_ROOT"/*; do
    [[ -d "$user_dir" ]] || continue
    user="$(basename -- "$user_dir")"
    if [[ ! -d "$user_dir/$ACTION" && ! -d "$user_dir/action_$ACTION" ]]; then
        continue
    fi
    USER_COUNT=$((USER_COUNT + 1))
    segment_args=(
        "${PYTHON_CMD[@]}" scripts/segment_ring_imu.py
        --data-root "$DATA_ROOT"
        --user "$user"
        --action "$ACTION"
        --input-kind spike-imu
        --spike-root "$DEST_SPIKE_ROOT"
        --boundary-mode "$BOUNDARY_MODE"
        --sampling-rate "$SAMPLING_RATE"
        --output-root "$DEST_SEGMENT_ROOT"
        --overwrite
    )
    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        segment_args+=(
            --alignment-offset-root "$DEST_ALIGNMENT_OFFSET_ROOT"
            --pre-press-context-seconds 0.2
            --post-lift-context-seconds 0.2
            --maximum-segment-duration-seconds 5.0
            --carry-in-press-lookback-seconds 0.2
            --missing-event-policy skip
            --crossing-touch-policy accept_until_next_press
            --recording-error-policy skip
            --verification-panel-seconds 10
            --verification-dpi "$SEGMENT_VERIFICATION_DPI"
            --overwrite-verification
        )
    fi
    note "segment user=$user action=$ACTION"
    "${segment_args[@]}"
done
[[ "$USER_COUNT" -gt 0 ]] || fail "no users found under new spike root: $DEST_SPIKE_ROOT"

if [[ "$REBUILD_PADDING" == "1" ]]; then
    SOURCE_PADDING_TARGET="$("${PYTHON_CMD[@]}" "$HELPER_SCRIPT" padding-target --source-root "$SOURCE_COMBINATION_ROOT")"
    if [[ "$SOURCE_PADDING_TARGET" != "NONE" ]]; then
        note "rebuild padding with reused target length: $SOURCE_PADDING_TARGET"
        "${PYTHON_CMD[@]}" scripts/pad_segmented_imu.py \
            --input-root "$DEST_SEGMENT_ROOT" \
            --output-root "$DEST_PADDING_ROOT" \
            --target-length "$SOURCE_PADDING_TARGET" \
            --sampling-rate "$SAMPLING_RATE" \
            --padding-value 0.0 \
            --overwrite
    else
        note "source has no padded dataset; padding skipped"
    fi
else
    note "padding rebuild disabled by REBUILD_PADDING=0"
fi

note "final contract verification"
"${PYTHON_CMD[@]}" "$HELPER_SCRIPT" verify \
    --source-root "$SOURCE_COMBINATION_ROOT" \
    --dest-root "$DEST_COMBINATION_ROOT" \
    --action "$ACTION" \
    --boundary-mode "$BOUNDARY_MODE" \
    --frequencies "${ENCODER_FREQUENCIES[@]}"

note "PASS"
note "new variant: $DEST_COMBINATION_ROOT"
note "validation report: $DEST_COMBINATION_ROOT/wavelet_variant_validation.json"
