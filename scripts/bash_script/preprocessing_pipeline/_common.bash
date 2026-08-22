#!/usr/bin/env bash

# Shared implementation for the eight action-0 pipeline entry points.
#
# The wrappers deliberately invoke the existing Python CLIs.  This file owns
# only discovery, orchestration, logging, padding, and post-run contract
# checks; it is not a second Ring or Board parser.
#
# Execution modes:
#   continue  (default) - validate existing stage outputs and resume from the
#                         first stage that has not started yet.  If an existing
#                         stage is partial or invalid, rebuild from preprocess.
#   overwrite           - skip resume validation and rebuild every stage.
#
# Explicit full rebuild example:
#   PIPELINE_MODE=overwrite bash scripts/action0_pipeline/<wrapper>.bash

set -Eeuo pipefail

_PIPELINE_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# This repository keeps the wrappers under scripts/bash_script/action0_pipeline.
# The common helper therefore has to walk three levels up to reach the project
# root (not two, which would resolve relative paths under scripts/).
PIPELINE_PROJECT_ROOT="$(cd -- "${_PIPELINE_COMMON_DIR}/../../.." && pwd)"

declare -a RING_FILES=()
declare -a RECORD_USERS=()
declare -a RECORD_ACTIONS=()
declare -a RECORD_DATASET_IDS=()
declare -a RECORD_DATASET_TOKENS=()
declare -a PIPELINE_USERS=()
declare -a PIPELINE_DOWNSTREAM_SEGMENT_OVERWRITE_ARGS=()
declare -a PIPELINE_DOWNSTREAM_PADDING_OVERWRITE_ARGS=()
declare -a PREPROCESS_OVERWRITE_ARGS=()
declare -a ENCODE_OVERWRITE_ARGS=()
declare -a ENCODER_FREQUENCY_ARGS=()
declare -a ALIGN_OVERWRITE_ARGS=()
declare -a SEGMENT_OVERWRITE_ARGS=()
declare -A RECORDING_SEEN=()
declare -A PIPELINE_USER_SEEN=()

_pipeline_resolve_path() {
    local value="$1"
    if [[ "$value" == /* ]]; then
        printf '%s\n' "$value"
    else
        printf '%s/%s\n' "$PIPELINE_PROJECT_ROOT" "$value"
    fi
}

pipeline_die() {
    local message="$1"
    if [[ -n "${DISCOVERY_LOG:-}" ]]; then
        printf 'ERROR: %s\n' "$message" >>"$DISCOVERY_LOG"
    fi
    printf 'error: %s\n' "$message" >&2
    exit 1
}

pipeline_require_file() {
    local path="$1"
    local description="$2"
    if [[ ! -s "$path" ]]; then
        pipeline_die "missing or empty ${description}: ${path}"
    fi
}

pipeline_count_files() {
    local root="$1"
    local pattern="$2"
    if [[ ! -d "$root" ]]; then
        printf '0\n'
        return 0
    fi
    find "$root" -type f -name "$pattern" -print | awk 'END { print NR + 0 }'
}

pipeline_log_command() {
    local log_path="$1"
    shift
    printf 'command:' >>"$log_path"
    printf ' %q' "$@" >>"$log_path"
    printf '\n' >>"$log_path"
}

pipeline_run_logged() {
    local log_path="$1"
    shift
    pipeline_log_command "$log_path" "$@"
    "$@" 2>&1 | tee -a "$log_path"
}

pipeline_note() {
    local message="$1"
    printf '[pipeline] %s\n' "$message"
    if [[ -n "${DISCOVERY_LOG:-}" ]]; then
        printf '[pipeline] %s\n' "$message" >>"$DISCOVERY_LOG"
    fi
}

pipeline_configure_overwrite_args() {
    PREPROCESS_OVERWRITE_ARGS=()
    ENCODE_OVERWRITE_ARGS=()
    ALIGN_OVERWRITE_ARGS=()
    SEGMENT_OVERWRITE_ARGS=()
    PIPELINE_DOWNSTREAM_SEGMENT_OVERWRITE_ARGS=()
    PIPELINE_DOWNSTREAM_PADDING_OVERWRITE_ARGS=()
    if [[ "$OVERWRITE" == "1" ]]; then
        PREPROCESS_OVERWRITE_ARGS+=(--overwrite)
        ENCODE_OVERWRITE_ARGS+=(--overwrite)
        ALIGN_OVERWRITE_ARGS+=(
            --overwrite-offset
            --overwrite-report
            --overwrite-verification
            --overwrite-outcome
        )
        SEGMENT_OVERWRITE_ARGS+=(--overwrite)
    fi
}

pipeline_prepare_downstream_overwrite() {
    if [[ "$OVERWRITE" == "1" ]]; then
        return 0
    fi
    PIPELINE_DOWNSTREAM_SEGMENT_OVERWRITE_ARGS=(--overwrite)
    PIPELINE_DOWNSTREAM_PADDING_OVERWRITE_ARGS=(--overwrite)
}

pipeline_init() {
    GRAVITY_METHOD="$1"
    BOUNDARY_MODE="$2"

    case "$GRAVITY_METHOD" in
        raw|low-pass|madgwick|xylo-rotate-and-remove-gravity) ;;
        *) pipeline_die "unsupported gravity method: ${GRAVITY_METHOD}" ;;
    esac
    case "$BOUNDARY_MODE" in
        label|aligned-board-events) ;;
        *) pipeline_die "unsupported boundary mode: ${BOUNDARY_MODE}" ;;
    esac

    if [[ "$GRAVITY_METHOD" == "xylo-rotate-and-remove-gravity" ]]; then
        OUTPUT_METHOD="xylo"
    else
        OUTPUT_METHOD="$GRAVITY_METHOD"
    fi

    cd "$PIPELINE_PROJECT_ROOT"

    DATA_ROOT="$(_pipeline_resolve_path "${DATA_ROOT:-data}")"
    ACTION="${ACTION:-0}"
    SAMPLING_RATE="${SAMPLING_RATE:-200}"
    ENCODER="${ENCODER:-custom-wavelet}"
    ENCODER_SETTINGS="${ENCODER_SETTINGS:-$(_pipeline_resolve_path configs/spike_encoding/custom_wavelet.json)}"
    ENCODER_FREQUENCIES=()
    POST_ENCODE_TRANSFORM="${POST_ENCODE_TRANSFORM:-none}"
    ENCODER_SETTINGS="$(_pipeline_resolve_path "${ENCODER_SETTINGS:-configs/spike_encoding/custom_wavelet.json}")"
    ENCODER_FREQUENCIES_HZ="${ENCODER_FREQUENCIES_HZ:-}"
    LEGACY_OVERWRITE="${OVERWRITE:-}"
    if [[ -n "$LEGACY_OVERWRITE" && "$LEGACY_OVERWRITE" != "0" && "$LEGACY_OVERWRITE" != "1" ]]; then
        pipeline_die "OVERWRITE must be 0 or 1 when used as a legacy mode alias"
    fi
    if [[ -n "${PIPELINE_MODE:-}" ]]; then
        PIPELINE_MODE="${PIPELINE_MODE}"
    elif [[ "$LEGACY_OVERWRITE" == "1" ]]; then
        PIPELINE_MODE="overwrite"
    else
        PIPELINE_MODE="continue"
    fi
    case "$PIPELINE_MODE" in
        continue) OVERWRITE=0 ;;
        overwrite) OVERWRITE=1 ;;
        *) pipeline_die "PIPELINE_MODE must be continue or overwrite" ;;
    esac
    
    if conda env list | grep -q '^writingring-gpu '; then
        CONDA_ENV="writingring-gpu"
    elif conda env list | grep -q '^writingring-viz '; then
        CONDA_ENV="writingring-viz"
    else
        echo "No suitable conda environment found."
        exit 1
    fi
    LOW_PASS_CUTOFF_HZ="${LOW_PASS_CUTOFF_HZ:-0.2}"
    MADGWICK_BETA="${MADGWICK_BETA:-0.1}"
    MADGWICK_PROVISIONAL="${MADGWICK_PROVISIONAL:-0}"
    OUTPUT_BASE="$(_pipeline_resolve_path "${OUTPUT_BASE:-outputs/action0_pipeline}")"
    PADDING_COVERAGE="${PADDING_COVERAGE:-0.99}"
    PADDING_RECOMMENDATION="${PADDING_RECOMMENDATION:-balanced}"
    PADDING_ROUND_TO="${PADDING_ROUND_TO:-1}"
    PADDING_VALUE="${PADDING_VALUE:-0.0}"

    if [[ "$ENCODER" != "custom-wavelet" ]]; then
        pipeline_die "action-0 scripts require ENCODER=custom-wavelet"
    fi
    case "$POST_ENCODE_TRANSFORM" in
        none|AbsRectify|PolaritySplitAbs) ;;
        *)
            pipeline_die "POST_ENCODE_TRANSFORM must be none, AbsRectify, or PolaritySplitAbs"
            ;;
    esac
    if [[ "$MADGWICK_PROVISIONAL" != "0" && "$MADGWICK_PROVISIONAL" != "1" ]]; then
        pipeline_die "MADGWICK_PROVISIONAL must be 0 or 1"
    fi
    case "$PADDING_RECOMMENDATION" in
        pure-padding|p99|balanced) ;;
        *) pipeline_die "PADDING_RECOMMENDATION must be pure-padding, p99, or balanced" ;;
    esac
    if [[ ! "$SAMPLING_RATE" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$SAMPLING_RATE" == 0 || "$SAMPLING_RATE" == 0.0 ]]; then
        pipeline_die "SAMPLING_RATE must be a positive number"
    fi
    ENCODER_FREQUENCY_ARGS=()
    if [[ -n "$ENCODER_FREQUENCIES_HZ" ]]; then
        if [[ "$ENCODER_FREQUENCIES_HZ" == *'['* || "$ENCODER_FREQUENCIES_HZ" == *']'* ]]; then
            pipeline_die "ENCODER_FREQUENCIES_HZ must be five whitespace-separated values without brackets"
        fi
        read -r -a ENCODER_FREQUENCIES <<<"$ENCODER_FREQUENCIES_HZ"
        if [[ "${#ENCODER_FREQUENCIES[@]}" -ne 5 ]]; then
            pipeline_die "ENCODER_FREQUENCIES_HZ must contain exactly five whitespace-separated values"
        fi
        ENCODER_FREQUENCY_ARGS=(--encoder-frequencies-hz "${ENCODER_FREQUENCIES[@]}")
    fi

    COMBINATION_ROOT="$OUTPUT_BASE/$OUTPUT_METHOD/$BOUNDARY_MODE"
    PREPROCESS_ROOT="$COMBINATION_ROOT/preprocessedIMU"
    SPIKE_OUTPUT_ROOT="$COMBINATION_ROOT/spikeEncoding"
    SPIKE_ROOT="$SPIKE_OUTPUT_ROOT/$ENCODER"
    ALIGNMENT_ROOT="$COMBINATION_ROOT/alignment"
    OFFSET_ROOT="$ALIGNMENT_ROOT/offsets"
    ALIGNMENT_REPORT_ROOT="$ALIGNMENT_ROOT/reports"
    ALIGNMENT_VERIFICATION_ROOT="$ALIGNMENT_ROOT/verification"
    SEGMENT_ROOT="$COMBINATION_ROOT/segmentation"
    PADDING_ANALYSIS_DIR="$(_pipeline_resolve_path "${PADDING_ANALYSIS_DIR:-$SEGMENT_ROOT/padding_analysis}")"
    PADDING_OUTPUT_ROOT="$(_pipeline_resolve_path "${PADDING_OUTPUT_ROOT:-$COMBINATION_ROOT/segmentation_padded}")"
    LOG_ROOT="$COMBINATION_ROOT/logs"
    DISCOVERY_LOG="$LOG_ROOT/discovery.log"
    ENCODE_LOG="$LOG_ROOT/encode.log"
    QA_LOG="$LOG_ROOT/qa.log"
    PADDING_LOG="$LOG_ROOT/padding.log"

    mkdir -p "$LOG_ROOT" "$LOG_ROOT/preprocess" "$LOG_ROOT/segmentation"
    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        mkdir -p "$LOG_ROOT/alignment"
    fi
    if [[ "$PIPELINE_MODE" == "overwrite" ]]; then
        : >"$DISCOVERY_LOG"
        : >"$ENCODE_LOG"
        : >"$QA_LOG"
        : >"$PADDING_LOG"
    else
        touch "$DISCOVERY_LOG" "$ENCODE_LOG" "$QA_LOG" "$PADDING_LOG"
    fi
    printf '\n=== pipeline invocation mode=%s ===\n' "$PIPELINE_MODE" >>"$DISCOVERY_LOG"

    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/writingring-matplotlib}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/writingring-cache}"
    export PYTHONUNBUFFERED=1
    mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

    if ! command -v conda >/dev/null 2>&1; then
        pipeline_die "conda is required to run the ${CONDA_ENV} environment"
    fi
    PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV" python)

    PIPELINE_BATCH_ORCHESTRATION="${PIPELINE_BATCH_ORCHESTRATION:-1}"
    case "$PIPELINE_BATCH_ORCHESTRATION" in
        0|1) ;;
        *) pipeline_die "PIPELINE_BATCH_ORCHESTRATION must be 0 or 1" ;;
    esac

    PIPELINE_BATCH_QA="${PIPELINE_BATCH_QA:-1}"
    case "$PIPELINE_BATCH_QA" in
        0|1) ;;
        *) pipeline_die "PIPELINE_BATCH_QA must be 0 or 1" ;;
    esac

    SEGMENT_VERIFICATION_DPI="${SEGMENT_VERIFICATION_DPI:-200}"
    if [[ ! "$SEGMENT_VERIFICATION_DPI" =~ ^[1-9][0-9]*$ ]]; then
        pipeline_die "SEGMENT_VERIFICATION_DPI must be a positive integer"
    fi

    pipeline_configure_overwrite_args
}

# The wrappers always call pipeline_init before executing a pipeline.  The
# individual helpers are also sourced by tests and by a few legacy callers,
# however, so those calls must remain self-contained.  In particular, do not
# select the batched drivers merely because the common file was sourced: the
# batched drivers require the complete initialized configuration below.
pipeline_prepare_legacy_defaults() {
    DATA_ROOT="${DATA_ROOT:-$(_pipeline_resolve_path data)}"
    ACTION="${ACTION:-0}"
    SAMPLING_RATE="${SAMPLING_RATE:-200}"
    GRAVITY_METHOD="${GRAVITY_METHOD:-raw}"
    ENCODER="${ENCODER:-custom-wavelet}"
    ENCODER_SETTINGS="${ENCODER_SETTINGS:-$(_pipeline_resolve_path configs/spike_encoding/custom_wavelet.json)}"
    ENCODER_FREQUENCIES=()
    POST_ENCODE_TRANSFORM="${POST_ENCODE_TRANSFORM:-none}"
    BOUNDARY_MODE="${BOUNDARY_MODE:-label}"
    OVERWRITE="${OVERWRITE:-0}"
    SEGMENT_VERIFICATION_DPI="${SEGMENT_VERIFICATION_DPI:-200}"
    LOW_PASS_CUTOFF_HZ="${LOW_PASS_CUTOFF_HZ:-0.2}"
    MADGWICK_BETA="${MADGWICK_BETA:-0.1}"
    MADGWICK_PROVISIONAL="${MADGWICK_PROVISIONAL:-0}"
    PADDING_COVERAGE="${PADDING_COVERAGE:-0.99}"
    PADDING_RECOMMENDATION="${PADDING_RECOMMENDATION:-balanced}"
    PADDING_ROUND_TO="${PADDING_ROUND_TO:-1}"
    PADDING_VALUE="${PADDING_VALUE:-0.0}"

    local legacy_root
    legacy_root="${COMBINATION_ROOT:-$(_pipeline_resolve_path outputs/action0_pipeline/$GRAVITY_METHOD/$BOUNDARY_MODE)}"
    COMBINATION_ROOT="$legacy_root"
    PREPROCESS_ROOT="${PREPROCESS_ROOT:-$legacy_root/preprocessedIMU}"
    SPIKE_OUTPUT_ROOT="${SPIKE_OUTPUT_ROOT:-$legacy_root/spikeEncoding}"
    SPIKE_ROOT="${SPIKE_ROOT:-$SPIKE_OUTPUT_ROOT/$ENCODER}"
    ALIGNMENT_ROOT="${ALIGNMENT_ROOT:-$legacy_root/alignment}"
    OFFSET_ROOT="${OFFSET_ROOT:-$ALIGNMENT_ROOT/offsets}"
    ALIGNMENT_REPORT_ROOT="${ALIGNMENT_REPORT_ROOT:-$ALIGNMENT_ROOT/reports}"
    ALIGNMENT_VERIFICATION_ROOT="${ALIGNMENT_VERIFICATION_ROOT:-$ALIGNMENT_ROOT/verification}"
    SEGMENT_ROOT="${SEGMENT_ROOT:-$legacy_root/segmentation}"
    PADDING_ANALYSIS_DIR="${PADDING_ANALYSIS_DIR:-$SEGMENT_ROOT/padding_analysis}"
    PADDING_OUTPUT_ROOT="${PADDING_OUTPUT_ROOT:-$legacy_root/segmentation_padded}"
    LOG_ROOT="${LOG_ROOT:-$legacy_root/logs}"
    DISCOVERY_LOG="${DISCOVERY_LOG:-/dev/null}"
    ENCODE_LOG="${ENCODE_LOG:-/dev/null}"
    QA_LOG="${QA_LOG:-/dev/null}"
    PADDING_LOG="${PADDING_LOG:-/dev/null}"

    # A direct source call has no pipeline_init marker.  Preserve the old
    # one-recording-at-a-time implementation unless the caller explicitly
    # opts into a batch driver.
    PIPELINE_BATCH_ORCHESTRATION="${PIPELINE_BATCH_ORCHESTRATION:-0}"
    PIPELINE_BATCH_QA="${PIPELINE_BATCH_QA:-0}"
}

pipeline_discover() {
    if [[ ! -d "$DATA_ROOT" ]]; then
        pipeline_die "data root is not a directory: ${DATA_ROOT}"
    fi
    if [[ ! -f "$ENCODER_SETTINGS" ]]; then
        pipeline_die "encoder settings file is missing: ${ENCODER_SETTINGS}"
    fi

    RING_FILES=()
    RECORD_USERS=()
    RECORD_ACTIONS=()
    RECORD_DATASET_IDS=()
    RECORD_DATASET_TOKENS=()
    PIPELINE_USERS=()
    RECORDING_SEEN=()
    PIPELINE_USER_SEEN=()

    mapfile -d '' RING_FILES < <(
        find "$DATA_ROOT" \
            -type f \
            -path "$DATA_ROOT/user_*/$ACTION/*_ring_0.bin" \
            -print0 |
            LC_ALL=C sort -zV
    )
    if (( ${#RING_FILES[@]} == 0 )); then
        pipeline_die "no primary ring_0 recordings found under ${DATA_ROOT}/user_*/${ACTION}"
    fi

    printf 'data_root=%s\naction=%s\n' "$DATA_ROOT" "$ACTION" >>"$DISCOVERY_LOG"
    printf 'discovered_ring_0_count=%d\n' "${#RING_FILES[@]}" >>"$DISCOVERY_LOG"

    local ring_path action_directory file_name dataset_token dataset_id record_user record_action
    local key board_count board_path
    local -a board_candidates=()
    for ring_path in "${RING_FILES[@]}"; do
        action_directory="${ring_path%/*}"
        file_name="${ring_path##*/}"
        record_user="${action_directory%/*}"
        record_user="${record_user##*/}"
        record_action="${action_directory##*/}"
        dataset_token="${file_name%_ring_0.bin}"
        if [[ -z "$dataset_token" || ! "$dataset_token" =~ ^[0-9]+$ ]]; then
            pipeline_die "ring_0 filename prefix must be an integer: ${ring_path}"
        fi
        dataset_id=$((10#$dataset_token))
        key="${record_user}/${record_action}/${dataset_id}"
        if [[ -n "${RECORDING_SEEN[$key]+present}" ]]; then
            pipeline_die "duplicate numeric recording identity for ring_0 files: ${key}"
        fi
        RECORDING_SEEN["$key"]=1
        RECORD_USERS+=("$record_user")
        RECORD_ACTIONS+=("$record_action")
        RECORD_DATASET_IDS+=("$dataset_id")
        RECORD_DATASET_TOKENS+=("$dataset_token")
        if [[ -z "${PIPELINE_USER_SEEN[$record_user]+present}" ]]; then
            PIPELINE_USER_SEEN["$record_user"]=1
            PIPELINE_USERS+=("$record_user")
        fi

        board_count=0
        board_candidates=("$action_directory/${dataset_token}_board_"*.gz)
        for board_path in "${board_candidates[@]}"; do
            if [[ -f "$board_path" ]]; then
                board_count=$((board_count + 1))
            fi
        done
        if [[ "$BOUNDARY_MODE" == "aligned-board-events" && "$board_count" -eq 0 ]]; then
            pipeline_die "Board mode requires at least one Board chunk for ${key}"
        fi
        printf 'recording=%s action=%s dataset_id=%s ring_0=%s board_chunk_count=%d\n' \
            "$record_user" "$record_action" "$dataset_id" "$ring_path" "$board_count" \
            >>"$DISCOVERY_LOG"
    done

    pipeline_run_logged "$DISCOVERY_LOG" "${PYTHON_CMD[@]}" -c \
        'import writingring; print("writingring import ok:", writingring.__file__)'
    if [[ "$GRAVITY_METHOD" == "xylo-rotate-and-remove-gravity" ]]; then
        pipeline_run_logged "$DISCOVERY_LOG" "${PYTHON_CMD[@]}" -c \
            'from rockpool.devices.xylo.syns63300 import Quantizer; from rockpool.devices.xylo.syns63300.imuif import RotationRemoval; print("Xylo dependency import ok")'
    fi
}

pipeline_preprocess_legacy() {
    local index record_user record_action dataset_id log_path
    local -a command_args=()
    for index in "${!RECORD_USERS[@]}"; do
        record_user="${RECORD_USERS[$index]}"
        record_action="${RECORD_ACTIONS[$index]}"
        dataset_id="${RECORD_DATASET_IDS[$index]}"
        log_path="$LOG_ROOT/preprocess/${record_user}_session_${dataset_id}.log"
        command_args=(
            "${PYTHON_CMD[@]}" scripts/preprocess_ring_imu.py
            --data-root "$DATA_ROOT"
            --output-root "$PREPROCESS_ROOT"
            --user "$record_user"
            --action "$record_action"
            --dataset-id "$dataset_id"
            --gravity-removal-method "$GRAVITY_METHOD"
            --sampling-rate "$SAMPLING_RATE"
        )
        case "$GRAVITY_METHOD" in
            low-pass)
                command_args+=(--low-pass-cutoff-hz "$LOW_PASS_CUTOFF_HZ")
                ;;
            madgwick)
                command_args+=(--madgwick-beta "$MADGWICK_BETA")
                if [[ "$MADGWICK_PROVISIONAL" == "1" ]]; then
                    command_args+=(--provisional)
                fi
                ;;
        esac
        command_args+=("${PREPROCESS_OVERWRITE_ARGS[@]}")
        pipeline_run_logged "$log_path" "${command_args[@]}"
        pipeline_require_file \
            "$PREPROCESS_ROOT/$record_user/$record_action/$dataset_id/${dataset_id}_preprocessedIMU.npy" \
            "preprocessed IMU artifact"
        pipeline_require_file \
            "$PREPROCESS_ROOT/$record_user/$record_action/$dataset_id/${dataset_id}_timestamps_us.npy" \
            "preprocessing timestamp sidecar"
        pipeline_require_file \
            "$PREPROCESS_ROOT/$record_user/$record_action/$dataset_id/${dataset_id}_preprocessing.json" \
            "preprocessing summary"
    done
}

pipeline_encode() {
    local -a command_args=(
        "${PYTHON_CMD[@]}" scripts/encode_spikes.py
        --input-root "$PREPROCESS_ROOT"
        --pattern '*_preprocessedIMU.npy'
        --output-root "$SPIKE_OUTPUT_ROOT"
        --encoder "$ENCODER"
        --encoder-settings "$ENCODER_SETTINGS"
        --post-encode-transform "$POST_ENCODE_TRANSFORM"
    )
    command_args+=("${ENCODER_FREQUENCY_ARGS[@]}")
    if [[ "$GRAVITY_METHOD" == "raw" ]]; then
        command_args+=(--allow-gravity-included)
    fi
    command_args+=("${ENCODE_OVERWRITE_ARGS[@]}")
    pipeline_run_logged "$ENCODE_LOG" "${command_args[@]}"

    local index record_user record_action dataset_id
    for index in "${!RECORD_USERS[@]}"; do
        record_user="${RECORD_USERS[$index]}"
        record_action="${RECORD_ACTIONS[$index]}"
        dataset_id="${RECORD_DATASET_IDS[$index]}"
        pipeline_require_file \
            "$SPIKE_ROOT/$record_user/$record_action/$dataset_id/spikes.npy" \
            "spike event matrix"
        pipeline_require_file \
            "$SPIKE_ROOT/$record_user/$record_action/$dataset_id/spikeIMU.npy" \
            "SpikeIMU matrix"
        pipeline_require_file \
            "$SPIKE_ROOT/$record_user/$record_action/$dataset_id/recording_offsets.npy" \
            "SpikeIMU recording offsets"
        pipeline_require_file \
            "$SPIKE_ROOT/$record_user/$record_action/$dataset_id/metadata.json" \
            "SpikeIMU metadata"
    done
}

pipeline_align_legacy() {
    local index record_user record_action dataset_id log_path
    local -a command_args=()
    for index in "${!RECORD_USERS[@]}"; do
        record_user="${RECORD_USERS[$index]}"
        record_action="${RECORD_ACTIONS[$index]}"
        dataset_id="${RECORD_DATASET_IDS[$index]}"
        log_path="$LOG_ROOT/alignment/${record_user}_session_${dataset_id}.log"
        command_args=(
            "${PYTHON_CMD[@]}" scripts/align_ring_board.py
            --data-root "$DATA_ROOT"
            --user "$record_user"
            --action "$record_action"
            --dataset-id "$dataset_id"
            --input-kind spike-imu
            --spike-root "$SPIKE_ROOT"
            --offset-output-root "$OFFSET_ROOT"
            --report-output-root "$ALIGNMENT_REPORT_ROOT"
            --verification-output-root "$ALIGNMENT_VERIFICATION_ROOT"
            --initial-interval-policy skip
            --unalignable-recording-policy skip
        )
        command_args+=("${ALIGN_OVERWRITE_ARGS[@]}")
        pipeline_run_logged "$log_path" "${command_args[@]}"
        pipeline_validate_alignment_outcome \
            "$record_user" "$record_action" "$dataset_id" || \
            pipeline_die "alignment outcome is invalid for ${record_user}/${record_action}/${dataset_id}"
    done
}

pipeline_recording_error_report_path() {
    local record_user="$1"
    local record_action="$2"
    printf '%s/recording_errors/%s/action_%s/%s_action_%s_segmentation_recording_errors.json\n' \
        "$SEGMENT_ROOT" "$record_user" "$record_action" "$record_user" "$record_action"
}

pipeline_validate_user_segmentation_state() {
    local record_user="$1"
    local record_action="$2"
    local segment_directory="$SEGMENT_ROOT/$record_user/action_$record_action"
    local summary_path="$segment_directory/${record_user}_action_${record_action}_segmentation_summary.json"
    local error_report_path
    local validation_log="${QA_LOG:-/dev/null}"
    local status=""
    error_report_path="$(pipeline_recording_error_report_path "$record_user" "$record_action")"

    local -a command_args=(
        "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import json
import sys

summary_path = Path(sys.argv[1])
error_path = Path(sys.argv[2])
user = sys.argv[3]
action = sys.argv[4]

def load(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return value

def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemExit(f"{name} must be a nonnegative integer")
    return value

def validate_dependency(value):
    if not isinstance(value, dict):
        raise SystemExit("alignment_outcome_dependency must be an object")
    source = value.get("source_recording_ids")
    outcomes = value.get("outcomes_by_status")
    if not isinstance(source, list) or not isinstance(outcomes, dict):
        raise SystemExit("alignment outcome dependency is malformed")
    success = outcomes.get("SUCCESS")
    skipped = outcomes.get("SKIPPED")
    if not isinstance(success, list) or not isinstance(skipped, list):
        raise SystemExit("alignment outcome dependency statuses are malformed")
    def key(value):
        if not isinstance(value, dict):
            raise SystemExit("alignment dependency identity is malformed")
        identity = value.get("identity", value)
        if not isinstance(identity, dict):
            raise SystemExit("alignment dependency identity is malformed")
        return (identity.get("user"), identity.get("action"), identity.get("dataset_id"))
    source_keys = [key(value) for value in source]
    outcome_keys = [key(value) for value in success + skipped]
    if (
        len(set(source_keys)) != len(source_keys)
        or len(set(outcome_keys)) != len(outcome_keys)
        or source_keys != sorted(source_keys, key=lambda item: item[2])
        or set(source_keys) != set(outcome_keys)
    ):
        raise SystemExit("alignment outcome dependency does not reconcile")
    return {entry[2] for entry in (key(value) for value in success)}

def validate(payload):
    if payload.get("input_kind") != "spike-imu":
        raise SystemExit("segmentation state is not SpikeIMU")
    if payload.get("boundary_mode") != "aligned_board_events":
        raise SystemExit("segmentation state is not aligned Board mode")
    source = nonnegative(payload.get("source_recording_count"), "source_recording_count")
    processed = nonnegative(payload.get("processed_recording_count"), "processed_recording_count")
    skipped = nonnegative(payload.get("skipped_recording_count"), "skipped_recording_count")
    errors = nonnegative(payload.get("segmentation_error_recording_count"), "segmentation_error_recording_count")
    entries = payload.get("segmentation_errors")
    if not isinstance(entries, list) or len(entries) != errors:
        raise SystemExit("segmentation error entries do not reconcile")
    if source != processed + skipped + errors:
        raise SystemExit("source recording counts do not reconcile")
    success_ids = validate_dependency(payload.get("alignment_outcome_dependency"))
    error_ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise SystemExit("segmentation error entry is malformed")
        identity = entry.get("identity")
        if not isinstance(identity, dict):
            raise SystemExit("segmentation error identity is malformed")
        dataset_id = identity.get("dataset_id")
        if (
            identity.get("user") != user
            or identity.get("action") != action
            or isinstance(dataset_id, bool)
            or not isinstance(dataset_id, int)
            or dataset_id < 0
            or not isinstance(entry.get("stage"), str)
            or not entry["stage"]
            or not isinstance(entry.get("error_type"), str)
            or not entry["error_type"]
            or not isinstance(entry.get("message"), str)
        ):
            raise SystemExit("segmentation error fields are malformed")
        error_ids.add(dataset_id)
    if len(error_ids) != errors or not error_ids <= success_ids:
        raise SystemExit("segmentation errors must retain alignment SUCCESS")
    return source, processed, skipped, errors

summary_exists = summary_path.is_file()
error_exists = error_path.is_file()
if summary_exists:
    summary = load(summary_path)
    _, processed, _, errors = validate(summary)
    if processed <= 0:
        raise SystemExit("segmentation summary has no processed recording")
    if errors == 0:
        if error_exists:
            raise SystemExit("stale recording-error report accompanies a clean package")
        print("PACKAGE_NO_ERRORS 0")
    else:
        if not error_exists:
            raise SystemExit("segmentation errors are missing their report")
        report = load(error_path)
        if report.get("terminal_state") != "completed_with_recording_errors":
            raise SystemExit("mixed segmentation error report has wrong terminal state")
        report_counts = validate(report)
        summary_counts = validate(summary)
        if report_counts != summary_counts:
            raise SystemExit("segmentation error report counts differ from package summary")
        if report.get("segmentation_errors") != summary.get("segmentation_errors") or report.get("alignment_outcome_dependency") != summary.get("alignment_outcome_dependency"):
            raise SystemExit("segmentation error report differs from package summary")
        print(f"PACKAGE_WITH_ERRORS {errors}")
elif error_exists:
    report = load(error_path)
    if report.get("terminal_state") != "all_recordings_error":
        raise SystemExit("report-only segmentation state has wrong terminal state")
    _, processed, _, errors = validate(report)
    if processed != 0 or errors == 0:
        raise SystemExit("report-only segmentation state does not describe all-recording errors")
    print(f"ALL_RECORDINGS_ERROR {errors}")
else:
    raise SystemExit("missing both segmentation package summary and recording-error report")
' \
            "$summary_path" "$error_report_path" "$record_user" "$record_action"
    )
    pipeline_log_command "$validation_log" "${command_args[@]}"
    if ! status="$("${command_args[@]}" 2>>"$validation_log")"; then
        PIPELINE_LAST_SEGMENTATION_STATE="INVALID"
        PIPELINE_LAST_SEGMENTATION_ERROR_COUNT=0
        return 1
    fi
    read -r PIPELINE_LAST_SEGMENTATION_STATE PIPELINE_LAST_SEGMENTATION_ERROR_COUNT <<<"$status"
    case "$PIPELINE_LAST_SEGMENTATION_STATE" in
        PACKAGE_NO_ERRORS|PACKAGE_WITH_ERRORS|ALL_RECORDINGS_ERROR)
            return 0
            ;;
        *)
            PIPELINE_LAST_SEGMENTATION_STATE="INVALID"
            PIPELINE_LAST_SEGMENTATION_ERROR_COUNT=0
            return 1
            ;;
    esac
}

pipeline_successful_segmentation_user_count_legacy() {
    local record_user count=0
    if [[ "$BOUNDARY_MODE" != "aligned-board-events" ]]; then
        printf '%s\n' "${#PIPELINE_USERS[@]}"
        return 0
    fi
    for record_user in "${PIPELINE_USERS[@]}"; do
        pipeline_validate_user_segmentation_state "$record_user" "$ACTION" || return 1
        case "$PIPELINE_LAST_SEGMENTATION_STATE" in
            PACKAGE_NO_ERRORS|PACKAGE_WITH_ERRORS) count=$((count + 1)) ;;
            ALL_RECORDINGS_ERROR) ;;
            *) return 1 ;;
        esac
    done
    printf '%s\n' "$count"
}

pipeline_write_segmentation_error_report() {
    local -a command_args=(
        "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import csv
import json
import os
import sys
import tempfile

root = Path(sys.argv[1])
action = sys.argv[2]
users = list(sys.argv[3:])
states = []
errors = []
for user in users:
    stem = f"{user}_action_{action}"
    summary_path = root / user / f"action_{action}" / f"{stem}_segmentation_summary.json"
    error_path = root / "recording_errors" / user / f"action_{action}" / f"{stem}_segmentation_recording_errors.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        state = "completed_with_recording_errors" if payload.get("segmentation_error_recording_count", 0) else "completed"
    elif error_path.is_file():
        payload = json.loads(error_path.read_text(encoding="utf-8"))
        state = payload.get("terminal_state")
    else:
        raise SystemExit(f"missing terminal segmentation state for {user}/{action}")
    entry = {
        "user": user,
        "action": action,
        "terminal_state": state,
        "source_recording_count": payload.get("source_recording_count"),
        "processed_recording_count": payload.get("processed_recording_count"),
        "alignment_skipped_recording_count": payload.get("alignment_skipped_recording_count", payload.get("skipped_recording_count")),
        "segmentation_error_recording_count": payload.get("segmentation_error_recording_count"),
    }
    states.append(entry)
    for error in payload.get("segmentation_errors", []):
        identity = error.get("identity", {}) if isinstance(error, dict) else {}
        errors.append({
            "user": identity.get("user"),
            "action": identity.get("action"),
            "dataset_id": identity.get("dataset_id"),
            "stage": error.get("stage") if isinstance(error, dict) else None,
            "error_type": error.get("error_type") if isinstance(error, dict) else None,
            "message": error.get("message") if isinstance(error, dict) else None,
        })
payload = {
    "schema_version": 1,
    "action": action,
    "user_action_states": states,
    "recording_error_count": len(errors),
    "recording_errors": errors,
}
root.mkdir(parents=True, exist_ok=True)
json_path = root / "segmentation_recording_error_report.json"
csv_path = root / "segmentation_recording_error_report.csv"
def atomic_text(path, text):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
atomic_text(json_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
descriptor, temporary = tempfile.mkstemp(prefix=f".{csv_path.name}.", dir=csv_path.parent, text=True)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["user", "action", "dataset_id", "stage", "error_type", "message"])
        writer.writeheader()
        writer.writerows(errors)
    os.replace(temporary, csv_path)
except Exception:
    Path(temporary).unlink(missing_ok=True)
    raise
print(f"published segmentation recording-error report: {json_path}")
' \
            "$SEGMENT_ROOT" "$ACTION" "${PIPELINE_USERS[@]}"
    )
    pipeline_run_logged "${QA_LOG:-/dev/null}" "${command_args[@]}"
}

pipeline_segment_legacy() {
    local index record_user record_action log_path
    local -a command_args=()
    for record_user in "${PIPELINE_USERS[@]}"; do
        record_action="$ACTION"
        log_path="$LOG_ROOT/segmentation/${record_user}.log"
        command_args=(
            "${PYTHON_CMD[@]}" scripts/segment_ring_imu.py
            --data-root "$DATA_ROOT"
            --user "$record_user"
            --action "$record_action"
            --input-kind spike-imu
            --spike-root "$SPIKE_ROOT"
            --boundary-mode "$BOUNDARY_MODE"
            --sampling-rate "$SAMPLING_RATE"
            --output-root "$SEGMENT_ROOT"
        )
        if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
            command_args+=(
                --alignment-offset-root "$OFFSET_ROOT"
                --pre-press-context-seconds 0.2
                --post-lift-context-seconds 0.2
                --maximum-segment-duration-seconds 5.0
                --carry-in-press-lookback-seconds 0.2
                --missing-event-policy skip
                --crossing-touch-policy accept_until_next_press
                --recording-error-policy skip
                --verification-panel-seconds 10
                --verification-dpi 200
                "${SEGMENT_OVERWRITE_ARGS[@]}"
                "${PIPELINE_DOWNSTREAM_SEGMENT_OVERWRITE_ARGS[@]}"
            )
        else
            command_args+=(
                "${SEGMENT_OVERWRITE_ARGS[@]}"
                "${PIPELINE_DOWNSTREAM_SEGMENT_OVERWRITE_ARGS[@]}"
            )
        fi
        pipeline_run_logged "$log_path" "${command_args[@]}"
    done
    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        pipeline_write_segmentation_error_report
    fi
}

pipeline_padding_legacy() {
    local analysis_report="$PADDING_ANALYSIS_DIR/segment_length_analysis.json"
    local successful_user_action_count
    if ! successful_user_action_count="$(pipeline_successful_segmentation_user_count)"; then
        pipeline_die "could not validate segmentation states before padding"
    fi
    if [[ "$successful_user_action_count" -eq 0 ]]; then
        if pipeline_padding_has_any_output; then
            pipeline_die "no successful segmentation package is available, but padding outputs exist"
        fi
        pipeline_note "no successful segmentation package; skipping length analysis and padding"
        return 0
    fi
    local -a analysis_args=(
        "${PYTHON_CMD[@]}" scripts/analyze_segment_lengths.py
        --input-root "$SEGMENT_ROOT"
        --output-dir "$PADDING_ANALYSIS_DIR"
        --sampling-rate "$SAMPLING_RATE"
        --minimum-coverage "$PADDING_COVERAGE"
        --round-to "$PADDING_ROUND_TO"
    )
    analysis_args+=("${PIPELINE_DOWNSTREAM_PADDING_OVERWRITE_ARGS[@]}")
    if [[ "$OVERWRITE" == "1" ]]; then
        analysis_args+=(--overwrite)
    fi
    pipeline_run_logged "$PADDING_LOG" "${analysis_args[@]}"
    pipeline_require_file "$analysis_report" "segment-length analysis report"

    local -a padding_args=(
        "${PYTHON_CMD[@]}" scripts/pad_segmented_imu.py
        --input-root "$SEGMENT_ROOT"
        --output-root "$PADDING_OUTPUT_ROOT"
        --analysis-report "$analysis_report"
        --recommendation "$PADDING_RECOMMENDATION"
        --sampling-rate "$SAMPLING_RATE"
        --padding-value "$PADDING_VALUE"
    )
    padding_args+=("${PIPELINE_DOWNSTREAM_PADDING_OVERWRITE_ARGS[@]}")
    if [[ "$OVERWRITE" == "1" ]]; then
        padding_args+=(--overwrite)
    fi
    pipeline_run_logged "$PADDING_LOG" "${padding_args[@]}"
    pipeline_require_file "$PADDING_OUTPUT_ROOT/padding_dataset_summary.json" "padded dataset summary"
    pipeline_require_file "$PADDING_OUTPUT_ROOT/padding_dataset_manifest.csv" "padded dataset manifest"
}

pipeline_validate_spike_artifact() {
    # Keep the requested-transform validation contract visible to lightweight callers:
    # "$POST_ENCODE_TRANSFORM" || return 1
    local values_path="$1"
    local metadata_path="$2"
    local timestamps_path="$3"
    local requested_transform="$4"
    local encoder_settings_path="${5:-}"
    local -a expected_frequencies=("${@:6}")
    pipeline_run_logged "$QA_LOG" "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import json
import numpy as np
import sys

values_path, metadata_path, timestamps_path = map(Path, sys.argv[1:4])
expected = sys.argv[4]
settings_path = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
frequencies = [float(value) for value in sys.argv[6:]]
expected_transform = None if expected == "none" else expected
expected_channels, expected_schema, expected_event_representation, expected_event_schema, expected_event_channels = {
    "none": (21, "signed_wavelet_events_plus_imu_v1", "signed", "custom_wavelet_signed_events_v1", 15),
    "AbsRectify": (21, "signed_wavelet_events_plus_imu_v1", "unsigned", "custom_wavelet_abs_rectified_events_v1", 15),
    "PolaritySplitAbs": (36, "polarity_split_wavelet_events_plus_imu_v1", "unsigned", "custom_wavelet_polarity_split_abs_events_v1", 30),
}[expected]
values = np.load(values_path, allow_pickle=False)
timestamps = np.load(timestamps_path, allow_pickle=False)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if values.ndim != 2 or values.shape[1] != expected_channels or not np.isfinite(values).all():
    raise SystemExit(f"invalid SpikeIMU matrix: {values_path} shape={values.shape}")
if timestamps.ndim != 1 or len(timestamps) != len(values):
    raise SystemExit(f"timestamp row count mismatch: {timestamps_path}")
spike_imu = metadata.get("spike_imu", {})
if (
    spike_imu.get("channel_count") != expected_channels
    or spike_imu.get("schema") != expected_schema
    or spike_imu.get("event_representation") != expected_event_representation
    or spike_imu.get("event_feature_schema") != expected_event_schema
    or spike_imu.get("event_channel_count") != expected_event_channels
):
    raise SystemExit(
        "SpikeIMU metadata layout mismatch: "
        f"expected={expected_schema}/{expected_channels}/{expected_event_representation}, path={metadata_path}"
    )
settings = metadata.get("settings") or {}
actual_transform = settings.get("post_encode_transform")
if actual_transform != expected_transform:
    raise SystemExit(
        "SpikeIMU post_encode_transform mismatch: "
        f"expected={expected_transform!r}, "
        f"actual={actual_transform!r}"
    )
if settings_path is not None:
    from writingring.spike_encoding import create_encoder, load_encoder_settings
    encoder_settings = load_encoder_settings(settings_path)
    if frequencies:
        encoder_settings["frequencies_hz"] = frequencies
    encoder_settings["post_encode_transform"] = expected_transform
    expected_encoder = create_encoder("custom-wavelet", settings=encoder_settings)
    expected_hash = expected_encoder.canonical_encoder_spec_sha256
    actual_hash = metadata.get("spike_encoder_spec_sha256")
    encoder_section = metadata.get("encoder") or {}
    if not actual_hash:
        actual_hash = encoder_section.get("spike_encoder_spec_sha256")
    if actual_hash != expected_hash:
        print(
            "SpikeIMU encoder identity mismatch: "
            f"expected={expected_hash!r}, actual={actual_hash!r}",
            file=sys.stderr,
        )
        raise SystemExit(2)
print(f"validated SpikeIMU rows={len(values)} channels={values.shape[1]}: {values_path}")
' "$values_path" "$metadata_path" "$timestamps_path" "$requested_transform" "$encoder_settings_path" "${expected_frequencies[@]}"
}

pipeline_validate_segment_artifact() {
    local values_path="$1"
    local summary_path="$2"
    local board_targets_path="${3:-}"
    if [[ -n "$board_targets_path" ]]; then
        pipeline_run_logged "$QA_LOG" "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import json
import numpy as np
import sys

values_path, summary_path, targets_path = map(Path, sys.argv[1:])
values = np.load(values_path, allow_pickle=False)
targets = np.load(targets_path, allow_pickle=False)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_channels = {
    "signed_wavelet_events_plus_imu_v1": 21,
    "polarity_split_wavelet_events_plus_imu_v1": 36,
}.get(summary.get("feature_schema"))
if summary.get("channel_count") != expected_channels:
    raise SystemExit(f"invalid segmented SpikeIMU schema: {summary_path}")
if values.ndim != 2 or values.shape[1] != expected_channels or not np.isfinite(values).all():
    raise SystemExit(f"invalid segmented SpikeIMU matrix: {values_path} shape={values.shape}")
if targets.shape != (len(values), 4) or targets.dtype != np.dtype(bool):
    raise SystemExit(f"invalid Board target matrix: {targets_path} shape={targets.shape} dtype={targets.dtype}")
print(f"validated segmented SpikeIMU rows={len(values)} channels={expected_channels} targets=4: {values_path}")
' "$values_path" "$summary_path" "$board_targets_path"
    else
        pipeline_run_logged "$QA_LOG" "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import json
import numpy as np
import sys

values_path, summary_path = map(Path, sys.argv[1:])
values = np.load(values_path, allow_pickle=False)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_channels = {
    "signed_wavelet_events_plus_imu_v1": 21,
    "polarity_split_wavelet_events_plus_imu_v1": 36,
}.get(summary.get("feature_schema"))
if summary.get("channel_count") != expected_channels:
    raise SystemExit(f"invalid segmented SpikeIMU schema: {summary_path}")
if values.ndim != 2 or values.shape[1] != expected_channels or not np.isfinite(values).all():
    raise SystemExit(f"invalid segmented SpikeIMU matrix: {values_path} shape={values.shape}")
print(f"validated segmented SpikeIMU rows={len(values)} channels={expected_channels}: {values_path}")
' "$values_path" "$summary_path"
    fi
}

pipeline_validate_padding_artifact() {
    local summary_path="$1"
    local expected_user_action_count="$2"
    pipeline_run_logged "$QA_LOG" "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import json
import sys

summary_path = Path(sys.argv[1])
expected_input_root = str(Path(sys.argv[2]).resolve())
expected_user_action_count = int(sys.argv[3])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("input_root") != expected_input_root:
    raise SystemExit(f"padded summary input root mismatch: {summary_path}")
if summary.get("input_kind") != "spike-imu":
    raise SystemExit(f"padded summary is not SpikeIMU: {summary_path}")
expected_channels = {
    "signed_wavelet_events_plus_imu_v1": 21,
    "polarity_split_wavelet_events_plus_imu_v1": 36,
}.get(summary.get("feature_schema"))
if summary.get("channel_count") != expected_channels:
    raise SystemExit(f"padded summary schema mismatch: {summary_path}")
processed_user_action_count = summary.get("processed_user_action_count")
if processed_user_action_count != expected_user_action_count:
    raise SystemExit(
        f"padded package count {processed_user_action_count} "
        f"!= user count {expected_user_action_count}"
    )
source_count = summary.get("source_segment_count")
exported_count = summary.get("segment_count")
skipped_count = summary.get("skipped_segment_count")
if not all(isinstance(value, int) and value >= 0 for value in (source_count, exported_count, skipped_count)):
    raise SystemExit(f"invalid padded segment counts: {summary_path}")
if exported_count + skipped_count != source_count:
    raise SystemExit(f"padded segment counts do not reconcile: {summary_path}")
target_length = summary.get("target_length")
if not isinstance(target_length, int) or target_length <= 0:
    raise SystemExit(f"invalid padded target length: {summary_path}")
print(
    f"validated padded output segments={exported_count} skipped={skipped_count} "
    f"target={target_length}: {summary_path}"
)
' "$summary_path" "$SEGMENT_ROOT" "$expected_user_action_count"
}


pipeline_validate_preprocess_artifact() {
    local values_path="$1"
    local timestamps_path="$2"
    local metadata_path="$3"
    pipeline_run_logged "$QA_LOG" "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import json
import numpy as np
import sys

values_path, timestamps_path, metadata_path = map(Path, sys.argv[1:])
values = np.load(values_path, allow_pickle=False)
timestamps = np.load(timestamps_path, allow_pickle=False)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
    raise SystemExit(f"invalid preprocessed IMU matrix: {values_path} shape={values.shape}")
if not np.isfinite(values).all():
    raise SystemExit(f"non-finite preprocessed IMU values: {values_path}")
if timestamps.ndim != 1 or len(timestamps) != len(values) or not np.isfinite(timestamps).all():
    raise SystemExit(f"invalid preprocessing timestamps: {timestamps_path}")
if np.any(np.diff(timestamps) < 0.0):
    raise SystemExit(f"preprocessing timestamps are not nondecreasing: {timestamps_path}")
if not isinstance(metadata, dict):
    raise SystemExit(f"preprocessing summary is not a JSON object: {metadata_path}")
print(f"validated preprocessing rows={len(values)} channels={values.shape[1]}: {values_path}")
' "$values_path" "$timestamps_path" "$metadata_path"
}

pipeline_validate_alignment_outcome() {
    local record_user="$1"
    local record_action="$2"
    local dataset_id="$3"
    local validation_log="${QA_LOG:-/dev/null}"
    local status=""
    local -a command_args=(
        "${PYTHON_CMD[@]}" -c '
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

data_root = Path(sys.argv[1])
spike_root = Path(sys.argv[2])
offset_root = Path(sys.argv[3])
verification_root = Path(sys.argv[4])
report_root = Path(sys.argv[5])
user = sys.argv[6]
action = sys.argv[7]
dataset_id = int(sys.argv[8])

try:
    # These imports and loaders are the same source-identity/provenance path
    # used by the alignment producer.  Their incidental output must not become
    # part of the typed status returned to Bash.
    with redirect_stdout(StringIO()):
        from writingring.alignment_io import (
            build_alignment_input_provenance,
            build_board_chunk_provenance,
            validate_alignment_outcome,
        )
        from writingring.board_loader import load_board
        from writingring.discovery import discover_recordings
        from writingring.recording_features import load_recording_features
        from writingring.selection import select_recording

        recording = select_recording(
            discover_recordings(data_root),
            user=user,
            action=action,
            dataset_id=dataset_id,
        )
        feature_input = load_recording_features(
            recording,
            input_kind="spike-imu",
            spike_root=spike_root,
        )
        board_data = load_board(recording)
        input_provenance = build_alignment_input_provenance(feature_input)
        board_provenance = build_board_chunk_provenance(board_data)
        outcome = validate_alignment_outcome(
            offset_root,
            verification_root,
            report_root,
            expected_recording={
                "user": user,
                "action": action,
                "dataset_id": dataset_id,
            },
            expected_input_provenance=input_provenance,
            expected_board_provenance=board_provenance,
        )
        status = outcome.status_name
except Exception as error:
    print(f"alignment outcome validation failed: {error}", file=sys.stderr)
    raise SystemExit(1) from error

if status not in {"SUCCESS", "SKIPPED"}:
    print(f"alignment outcome returned unsupported status: {status}", file=sys.stderr)
    raise SystemExit(1)
sys.stdout.write(status)
' \
        "$DATA_ROOT" \
        "$SPIKE_ROOT" \
        "$OFFSET_ROOT" \
        "$ALIGNMENT_VERIFICATION_ROOT" \
        "$ALIGNMENT_REPORT_ROOT" \
        "$record_user" \
        "$record_action" \
        "$dataset_id"
    )

    pipeline_log_command "$validation_log" "${command_args[@]}"
    if ! status="$("${command_args[@]}" 2>>"$validation_log")"; then
        PIPELINE_LAST_ALIGNMENT_STATUS="INVALID"
        printf 'alignment_outcome_status=%s user=%s action=%s dataset_id=%s\n' \
            "INVALID" "$record_user" "$record_action" "$dataset_id" \
            >>"$validation_log"
        return 1
    fi
    case "$status" in
        SUCCESS|SKIPPED)
            PIPELINE_LAST_ALIGNMENT_STATUS="$status"
            printf 'alignment_outcome_status=%s user=%s action=%s dataset_id=%s\n' \
                "$status" "$record_user" "$record_action" "$dataset_id" \
                >>"$validation_log"
            return 0
            ;;
        *)
            PIPELINE_LAST_ALIGNMENT_STATUS="INVALID"
            printf 'alignment_outcome_status=%s user=%s action=%s dataset_id=%s\n' \
                "INVALID" "$record_user" "$record_action" "$dataset_id" \
                >>"$validation_log"
            return 1
            ;;
    esac
}

pipeline_validate_segment_package() {
    local segment_directory="$1"
    local record_user="$2"
    local record_action="$3"
    local board_mode="$4"
    local values_path="$segment_directory/${record_user}_action_${record_action}_spikeIMU.npy"
    local labels_path="$segment_directory/${record_user}_action_${record_action}_labels.npy"
    local offsets_path="$segment_directory/${record_user}_action_${record_action}_segment_offsets.npy"
    local lengths_path="$segment_directory/${record_user}_action_${record_action}_segment_lengths.npy"
    local summary_path="$segment_directory/${record_user}_action_${record_action}_segmentation_summary.json"
    local targets_path=""
    if [[ "$board_mode" == "1" ]]; then
        targets_path="$segment_directory/${record_user}_action_${record_action}_board_event_targets.npy"
    fi
    pipeline_run_logged "$QA_LOG" "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import json
import numpy as np
import sys

values_path = Path(sys.argv[1])
labels_path = Path(sys.argv[2])
offsets_path = Path(sys.argv[3])
lengths_path = Path(sys.argv[4])
summary_path = Path(sys.argv[5])
targets_path = Path(sys.argv[6]) if sys.argv[6] else None

values = np.load(values_path, allow_pickle=False)
labels = np.load(labels_path, allow_pickle=False)
offsets = np.load(offsets_path, allow_pickle=False)
lengths = np.load(lengths_path, allow_pickle=False)
summary = json.loads(summary_path.read_text(encoding="utf-8"))

expected_channels = {
    "signed_wavelet_events_plus_imu_v1": 21,
    "polarity_split_wavelet_events_plus_imu_v1": 36,
}.get(summary.get("feature_schema"))
if summary.get("channel_count") != expected_channels:
    raise SystemExit(f"invalid segmented SpikeIMU schema: {summary_path}")
if values.ndim != 2 or values.shape[1] != expected_channels or not np.isfinite(values).all():
    raise SystemExit(f"invalid segmented SpikeIMU matrix: {values_path} shape={values.shape}")
if labels.ndim != 1 or lengths.ndim != 1 or len(labels) != len(lengths):
    raise SystemExit(f"segment labels/lengths mismatch under {values_path.parent}")
if offsets.ndim != 1 or len(offsets) != len(lengths) + 1:
    raise SystemExit(f"invalid segment offsets under {values_path.parent}")
if len(offsets) == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != len(values):
    raise SystemExit(f"segment offsets do not cover the feature matrix under {values_path.parent}")
if np.any(lengths < 0) or not np.array_equal(np.diff(offsets), lengths):
    raise SystemExit(f"segment lengths do not match offsets under {values_path.parent}")
if not isinstance(summary, dict):
    raise SystemExit(f"segmentation summary is not a JSON object: {summary_path}")
if targets_path is not None:
    targets = np.load(targets_path, allow_pickle=False)
    if targets.shape != (len(values), 4) or targets.dtype != np.dtype(bool):
        raise SystemExit(f"invalid Board target matrix: {targets_path} shape={targets.shape} dtype={targets.dtype}")
print(f"validated segment package segments={len(lengths)} rows={len(values)}: {values_path.parent}")
' "$values_path" "$labels_path" "$offsets_path" "$lengths_path" "$summary_path" "$targets_path"
}

pipeline_compute_alignment_outcome_dependency() {
    local record_user="$1"
    local record_action="$2"
    local validation_log="${QA_LOG:-/dev/null}"
    local -a command_args=(
        "${PYTHON_CMD[@]}" -c '
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys

data_root = Path(sys.argv[1])
spike_root = Path(sys.argv[2])
offset_root = Path(sys.argv[3])
verification_root = Path(sys.argv[4])
report_root = Path(sys.argv[5])
user = sys.argv[6]
action = sys.argv[7]

try:
    # Keep the helper result typed: loader diagnostics and incidental output
    # must never be mixed into the JSON consumed by Bash.
    with redirect_stdout(StringIO()):
        from writingring.alignment_io import (
            build_alignment_input_provenance,
            build_board_chunk_provenance,
            read_alignment_skip_artifact,
            validate_alignment_outcome,
        )
        from writingring.board_loader import load_board
        from writingring.discovery import discover_recordings
        from writingring.preprocessing_io import sha256_file
        from writingring.recording_features import load_recording_features

        recordings = sorted(
            (
                recording
                for recording in discover_recordings(data_root)
                if recording.user == user and recording.action == action
            ),
            key=lambda recording: recording.dataset_id,
        )
        if not recordings:
            raise ValueError(
                f"no recordings discovered for {user}/{action}"
            )

        dependency = {
            "source_recording_ids": [
                {
                    "user": recording.user,
                    "action": recording.action,
                    "dataset_id": recording.dataset_id,
                }
                for recording in recordings
            ],
            "outcomes_by_status": {"SUCCESS": [], "SKIPPED": []},
        }
        for recording in recordings:
            identity = {
                "user": recording.user,
                "action": recording.action,
                "dataset_id": recording.dataset_id,
            }
            feature_input = load_recording_features(
                recording,
                input_kind="spike-imu",
                spike_root=spike_root,
            )
            board_data = load_board(recording)
            input_provenance = build_alignment_input_provenance(feature_input)
            board_provenance = build_board_chunk_provenance(board_data)
            outcome = validate_alignment_outcome(
                offset_root,
                verification_root,
                report_root,
                expected_recording=identity,
                expected_input_provenance=input_provenance,
                expected_board_provenance=board_provenance,
            )
            status = outcome.status_name
            if status not in {"SUCCESS", "SKIPPED"}:
                raise ValueError(
                    f"unsupported alignment outcome status: {status}"
                )
            entry = {
                "identity": identity,
                "report_sha256": sha256_file(outcome.paths.report_path),
            }
            if status == "SKIPPED":
                skip = read_alignment_skip_artifact(
                    outcome.paths.skip_json_path,
                    expected_user=recording.user,
                    expected_action=recording.action,
                    expected_dataset_id=recording.dataset_id,
                    expected_input_provenance=input_provenance,
                    expected_board_provenance=board_provenance,
                )
                entry["reason"] = skip.reason
            dependency["outcomes_by_status"][status].append(entry)
except Exception as error:
    print(f"alignment outcome dependency recomputation failed: {error}", file=sys.stderr)
    raise SystemExit(1) from error

sys.stdout.write(json.dumps(dependency, sort_keys=True, separators=(",", ":")))
' \
        "$DATA_ROOT" \
        "$SPIKE_ROOT" \
        "$OFFSET_ROOT" \
        "$ALIGNMENT_VERIFICATION_ROOT" \
        "$ALIGNMENT_REPORT_ROOT" \
        "$record_user" \
        "$record_action"
    )

    "${command_args[@]}" 2>>"$validation_log"
}

pipeline_alignment_outcome_dependency_matches() {
    local record_user="$1"
    local record_action="$2"
    local segment_directory="$SEGMENT_ROOT/$record_user/action_$record_action"
    local summary_path="$segment_directory/${record_user}_action_${record_action}_segmentation_summary.json"
    local error_report_path=""
    local current_dependency=""
    local comparison_log="${QA_LOG:-/dev/null}"
    local -a comparison_args=()

    error_report_path="$(pipeline_recording_error_report_path "$record_user" "$record_action")"
    if [[ ! -f "$summary_path" && -f "$error_report_path" ]]; then
        summary_path="$error_report_path"
    fi
    if ! current_dependency="$(
        pipeline_compute_alignment_outcome_dependency \
            "$record_user" "$record_action"
    )"; then
        return 2
    fi

    comparison_args=(
        "${PYTHON_CMD[@]}" -c '
import json
from pathlib import Path
import sys

summary_path = Path(sys.argv[1])
expected = json.loads(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if not isinstance(summary, dict):
    raise SystemExit("segmentation summary is not a JSON object")
actual = summary.get("alignment_outcome_dependency")
if actual != expected:
    raise SystemExit("alignment outcome dependency is stale")
' \
        "$summary_path" \
        "$current_dependency"
    )
    if "${comparison_args[@]}" 2>>"$comparison_log"; then
        return 0
    fi
    return 1
}

pipeline_alignment_outcome_dependencies_valid() {
    local record_user dependency_status=0
    for record_user in "${PIPELINE_USERS[@]}"; do
        pipeline_alignment_outcome_dependency_matches \
            "$record_user" "$ACTION" || {
                dependency_status=$?
                return "$dependency_status"
            }
    done
    return 0
}

pipeline_path_has_files() {
    local root="$1"
    local first=""
    if [[ ! -d "$root" ]]; then
        return 1
    fi
    first="$(find "$root" -type f -print -quit 2>/dev/null || true)"
    [[ -n "$first" ]]
}

pipeline_preprocess_has_any_output() {
    pipeline_path_has_files "$PREPROCESS_ROOT"
}

pipeline_encode_has_any_output() {
    pipeline_path_has_files "$SPIKE_ROOT"
}

pipeline_alignment_has_any_output() {
    pipeline_path_has_files "$OFFSET_ROOT" || \
        pipeline_path_has_files "$ALIGNMENT_REPORT_ROOT" || \
        pipeline_path_has_files "$ALIGNMENT_VERIFICATION_ROOT"
}

pipeline_segment_has_any_output() {
    local first=""
    if [[ ! -d "$SEGMENT_ROOT" ]]; then
        return 1
    fi
    first="$(find "$SEGMENT_ROOT" -type f \
        \( -name '*_spikeIMU.npy' -o -name '*_labels.npy' -o -name '*_segment_offsets.npy' \
           -o -name '*_segment_lengths.npy' -o -name '*_segmentation_summary.json' \
           -o -name '*_segments.csv' -o -name '*_board_event_targets.npy' \
           -o -name '*_board_events.csv' -o -name '*_segmentation_recording_errors.json' \
           -o -name 'segmentation_recording_error_report.json' \
           -o -name 'segmentation_recording_error_report.csv' \) -print -quit 2>/dev/null || true)"
    [[ -n "$first" ]]
}

pipeline_padding_has_any_output() {
    pipeline_path_has_files "$PADDING_ANALYSIS_DIR" || pipeline_path_has_files "$PADDING_OUTPUT_ROOT"
}

pipeline_preprocess_outputs_valid() {
    local index record_user record_action dataset_id directory
    for index in "${!RECORD_USERS[@]}"; do
        record_user="${RECORD_USERS[$index]}"
        record_action="${RECORD_ACTIONS[$index]}"
        dataset_id="${RECORD_DATASET_IDS[$index]}"
        directory="$PREPROCESS_ROOT/$record_user/$record_action/$dataset_id"
        [[ -s "$directory/${dataset_id}_preprocessedIMU.npy" ]] || return 1
        [[ -s "$directory/${dataset_id}_timestamps_us.npy" ]] || return 1
        [[ -s "$directory/${dataset_id}_preprocessing.json" ]] || return 1
        pipeline_validate_preprocess_artifact \
            "$directory/${dataset_id}_preprocessedIMU.npy" \
            "$directory/${dataset_id}_timestamps_us.npy" \
            "$directory/${dataset_id}_preprocessing.json" || return 1
    done
}

pipeline_encode_outputs_valid() {
    # Requested transform remains a hard validation gate: "$POST_ENCODE_TRANSFORM" || return 1
    local index record_user record_action dataset_id directory timestamps_path
    local -a encoder_identity_args=()
    if [[ -n "${ENCODER_SETTINGS:-}" ]]; then
        encoder_identity_args=("$ENCODER_SETTINGS" "${ENCODER_FREQUENCIES[@]}")
    fi
    for index in "${!RECORD_USERS[@]}"; do
        record_user="${RECORD_USERS[$index]}"
        record_action="${RECORD_ACTIONS[$index]}"
        dataset_id="${RECORD_DATASET_IDS[$index]}"
        directory="$SPIKE_ROOT/$record_user/$record_action/$dataset_id"
        timestamps_path="$PREPROCESS_ROOT/$record_user/$record_action/$dataset_id/${dataset_id}_timestamps_us.npy"
        [[ -s "$directory/spikes.npy" ]] || return 1
        [[ -s "$directory/spikeIMU.npy" ]] || return 1
        [[ -s "$directory/recording_offsets.npy" ]] || return 1
        [[ -s "$directory/metadata.json" ]] || return 1
        [[ -s "$timestamps_path" ]] || return 1
        pipeline_validate_spike_artifact \
            "$directory/spikeIMU.npy" \
            "$directory/metadata.json" \
            "$timestamps_path" \
            "$POST_ENCODE_TRANSFORM" \
            "${encoder_identity_args[@]}" || {
                local validation_status=$?
                if [[ "$validation_status" -eq 2 ]]; then
                    return 2
                fi
                return 1
            }
        # "$POST_ENCODE_TRANSFORM" || return 1
    done
}

pipeline_alignment_outputs_valid() {
    local index record_user record_action dataset_id
    local invalid_count=0
    for index in "${!RECORD_USERS[@]}"; do
        record_user="${RECORD_USERS[$index]}"
        record_action="${RECORD_ACTIONS[$index]}"
        dataset_id="${RECORD_DATASET_IDS[$index]}"
        pipeline_validate_alignment_outcome \
            "$record_user" "$record_action" "$dataset_id" || invalid_count=$((invalid_count + 1))
    done
    [[ "$invalid_count" -eq 0 ]]
}

pipeline_segment_outputs_valid() {
    local record_user segment_directory board_mode=0
    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        board_mode=1
    fi
    for record_user in "${PIPELINE_USERS[@]}"; do
        segment_directory="$SEGMENT_ROOT/$record_user/action_$ACTION"
        if [[ "$board_mode" == "1" ]]; then
            pipeline_validate_user_segmentation_state "$record_user" "$ACTION" || return 1
            case "$PIPELINE_LAST_SEGMENTATION_STATE" in
                ALL_RECORDINGS_ERROR)
                    if [[ -d "$segment_directory" ]] && find "$segment_directory" -type f -print -quit | grep -q .; then
                        return 1
                    fi
                    continue
                    ;;
                PACKAGE_NO_ERRORS|PACKAGE_WITH_ERRORS)
                    ;;
                *)
                    return 1
                    ;;
            esac
        fi
        [[ -s "$segment_directory/${record_user}_action_${ACTION}_spikeIMU.npy" ]] || return 1
        [[ -s "$segment_directory/${record_user}_action_${ACTION}_labels.npy" ]] || return 1
        [[ -s "$segment_directory/${record_user}_action_${ACTION}_segment_offsets.npy" ]] || return 1
        [[ -s "$segment_directory/${record_user}_action_${ACTION}_segment_lengths.npy" ]] || return 1
        [[ -s "$segment_directory/${record_user}_action_${ACTION}_segments.csv" ]] || return 1
        [[ -s "$segment_directory/${record_user}_action_${ACTION}_segmentation_summary.json" ]] || return 1
        if [[ "$board_mode" == "1" ]]; then
            [[ -s "$segment_directory/${record_user}_action_${ACTION}_board_event_targets.npy" ]] || return 1
            [[ -s "$segment_directory/${record_user}_action_${ACTION}_board_events.csv" ]] || return 1
        fi
        pipeline_validate_segment_package \
            "$segment_directory" "$record_user" "$ACTION" "$board_mode" || return 1
    done
}

pipeline_padding_outputs_valid() {
    local analysis_report="$PADDING_ANALYSIS_DIR/segment_length_analysis.json"
    local summary_path="$PADDING_OUTPUT_ROOT/padding_dataset_summary.json"
    local manifest_path="$PADDING_OUTPUT_ROOT/padding_dataset_manifest.csv"
    local expected_user_action_count
    if ! expected_user_action_count="$(pipeline_successful_segmentation_user_count)"; then
        return 1
    fi
    if [[ "$expected_user_action_count" -eq 0 ]]; then
        ! pipeline_padding_has_any_output
        return
    fi
    [[ -s "$analysis_report" ]] || return 1
    [[ -s "$summary_path" ]] || return 1
    [[ -s "$manifest_path" ]] || return 1
    pipeline_validate_padding_artifact "$summary_path" "$expected_user_action_count" || return 1
}

pipeline_force_full_rebuild() {
    local reason="$1"
    PIPELINE_RESUME_STAGE="preprocess"
    PIPELINE_FORCE_REBUILD=1
    OVERWRITE=1
    pipeline_configure_overwrite_args
    pipeline_note "continue validation failed: ${reason}; rebuilding from preprocess with overwrite enabled"
}

pipeline_prepare_encoder_rebuild() {
    # Rebuild encoding and every downstream consumer while retaining valid
    # preprocessing artifacts for continue-mode encoder changes.
    ENCODE_OVERWRITE_ARGS=(--overwrite)
    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        ALIGN_OVERWRITE_ARGS=(
            --overwrite-offset
            --overwrite-report
            --overwrite-verification
            --overwrite-outcome
        )
    fi
    pipeline_prepare_downstream_overwrite
}

pipeline_plan_continue() {
    local preprocess_any=0 encode_any=0 alignment_any=0 segment_any=0 padding_any=0
    local dependency_status=0
    PIPELINE_RESUME_STAGE="preprocess"
    PIPELINE_FORCE_REBUILD=0
    PIPELINE_DOWNSTREAM_SEGMENT_OVERWRITE_ARGS=()
    PIPELINE_DOWNSTREAM_PADDING_OVERWRITE_ARGS=()

    pipeline_preprocess_has_any_output && preprocess_any=1
    pipeline_encode_has_any_output && encode_any=1
    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        pipeline_alignment_has_any_output && alignment_any=1
    fi
    pipeline_segment_has_any_output && segment_any=1
    pipeline_padding_has_any_output && padding_any=1

    if [[ "$preprocess_any" == "0" ]]; then
        if (( encode_any || alignment_any || segment_any || padding_any )); then
            pipeline_force_full_rebuild "preprocess outputs are absent while downstream outputs exist"
        else
            PIPELINE_RESUME_STAGE="preprocess"
            pipeline_note "no preprocess outputs found; starting from preprocess"
        fi
        return 0
    fi
    if ! pipeline_preprocess_outputs_valid; then
        pipeline_force_full_rebuild "preprocess outputs are partial or invalid"
        return 0
    fi
    pipeline_note "preprocess outputs are complete and valid; skipping preprocess"

    if [[ "$encode_any" == "0" ]]; then
        if (( alignment_any || segment_any || padding_any )); then
            pipeline_force_full_rebuild "encode outputs are absent while downstream outputs exist"
        else
            PIPELINE_RESUME_STAGE="encode"
            pipeline_note "encode outputs not found; resuming from encode"
        fi
        return 0
    fi
    local encode_validation_status=0
    pipeline_encode_outputs_valid || encode_validation_status=$?
    case "$encode_validation_status" in
        0)
            ;;
        2)
            pipeline_prepare_encoder_rebuild
            PIPELINE_RESUME_STAGE="encode"
            pipeline_note \
                "encoder identity is stale; resuming from encode with downstream overwrite"
            return 0
            ;;
        *)
            pipeline_force_full_rebuild "encode outputs are partial or invalid"
            return 0
            ;;
    esac
    pipeline_note "encode outputs are complete and valid; skipping encode"

    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        if [[ "$alignment_any" == "0" ]]; then
            if (( segment_any || padding_any )); then
                pipeline_force_full_rebuild "alignment outputs are absent while downstream outputs exist"
            else
                PIPELINE_RESUME_STAGE="align"
                pipeline_note "alignment outputs not found; resuming from alignment"
            fi
            return 0
        fi
        if ! pipeline_alignment_outputs_valid; then
            pipeline_force_full_rebuild "alignment outputs are partial or invalid"
            return 0
        fi
        pipeline_note "alignment outputs are complete and valid; skipping alignment"
    fi

    if [[ "$segment_any" == "0" ]]; then
        if (( padding_any )); then
            pipeline_force_full_rebuild "segmentation outputs are absent while padding outputs exist"
        else
            PIPELINE_RESUME_STAGE="segment"
            pipeline_note "segmentation outputs not found; resuming from segmentation"
        fi
        return 0
    fi
    if ! pipeline_segment_outputs_valid; then
        pipeline_force_full_rebuild "segmentation outputs are partial or invalid"
        return 0
    fi
    pipeline_note "segmentation outputs are complete and valid; skipping segmentation"

    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        pipeline_alignment_outcome_dependencies_valid || dependency_status=$?
        case "$dependency_status" in
            0)
                ;;
            1)
                pipeline_prepare_downstream_overwrite
                PIPELINE_RESUME_STAGE="segment"
                pipeline_note \
                    "alignment outcome dependency is stale; resuming from segmentation with downstream overwrite"
                return 0
                ;;
            *)
                pipeline_force_full_rebuild \
                    "alignment outcome dependency could not be recomputed"
                return 0
                ;;
        esac
    fi

    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        local successful_user_action_count
        if ! successful_user_action_count="$(pipeline_successful_segmentation_user_count)"; then
            pipeline_force_full_rebuild "segmentation terminal states are partial or invalid"
            return 0
        fi
        if [[ "$successful_user_action_count" -eq 0 ]]; then
            if [[ "$padding_any" == "1" ]]; then
                pipeline_force_full_rebuild "padding outputs exist without a successful segmentation package"
            else
                PIPELINE_RESUME_STAGE="complete"
                pipeline_note "all segmentation users reached terminal recording errors; no padding output is expected"
            fi
            return 0
        fi
    fi

    if [[ "$padding_any" == "0" ]]; then
        PIPELINE_RESUME_STAGE="padding"
        pipeline_note "padding outputs not found; resuming from padding"
        return 0
    fi
    if ! pipeline_padding_outputs_valid; then
        pipeline_force_full_rebuild "padding outputs are partial or invalid"
        return 0
    fi

    PIPELINE_RESUME_STAGE="complete"
    pipeline_note "all pipeline outputs are complete and valid; no compute stage needs to rerun"
}

pipeline_execute_from_stage() {
    local stage="$1"
    case "$stage" in
        preprocess)
            pipeline_preprocess
            pipeline_encode
            if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
                pipeline_align
            fi
            pipeline_segment
            pipeline_padding
            ;;
        encode)
            pipeline_encode
            if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
                pipeline_align
            fi
            pipeline_segment
            pipeline_padding
            ;;
        align)
            if [[ "$BOUNDARY_MODE" != "aligned-board-events" ]]; then
                pipeline_die "internal resume error: align stage requested for label mode"
            fi
            pipeline_align
            pipeline_segment
            pipeline_padding
            ;;
        segment)
            pipeline_segment
            pipeline_padding
            ;;
        padding)
            pipeline_padding
            ;;
        complete)
            ;;
        *)
            pipeline_die "internal resume error: unknown stage ${stage}"
            ;;
    esac
}

pipeline_qa_legacy() {
    local -a encoder_identity_args=()
    if [[ -n "${ENCODER_SETTINGS:-}" ]]; then
        encoder_identity_args=("$ENCODER_SETTINGS" "${ENCODER_FREQUENCIES[@]}")
    fi
    local ring_count="${#RING_FILES[@]}"
    local preprocessing_count spike_count summary_count padded_summary_count
    local successful_user_action_count="${#PIPELINE_USERS[@]}"
    local segmentation_error_count=0
    local all_error_user_count=0
    local record_user
    local segment_directory
    preprocessing_count="$(pipeline_count_files "$PREPROCESS_ROOT" '*_preprocessing.json')"
    spike_count="$(pipeline_count_files "$SPIKE_ROOT" 'spikeIMU.npy')"
    summary_count="$(pipeline_count_files "$SEGMENT_ROOT" '*_segmentation_summary.json')"
    padded_summary_count="$(pipeline_count_files "$PADDING_OUTPUT_ROOT" 'padding_dataset_summary.json')"

    printf 'ring_0_recordings=%s\n' "$ring_count" >>"$QA_LOG"
    printf 'preprocessing_summaries=%s\n' "$preprocessing_count" >>"$QA_LOG"
    printf 'spikeIMU_matrices=%s\n' "$spike_count" >>"$QA_LOG"
    printf 'segmentation_summaries=%s\n' "$summary_count" >>"$QA_LOG"
    printf 'padded_dataset_summaries=%s\n' "$padded_summary_count" >>"$QA_LOG"

    if [[ "$preprocessing_count" -ne "$ring_count" ]]; then
        pipeline_die "preprocessing summary count ${preprocessing_count} != ring_0 count ${ring_count}"
    fi
    if [[ "$spike_count" -ne "$ring_count" ]]; then
        pipeline_die "SpikeIMU count ${spike_count} != ring_0 count ${ring_count}"
    fi
    local alignment_total_count=0
    local alignment_success_count=0
    local alignment_skipped_count=0
    local alignment_invalid_count=0
    local index record_action dataset_id
    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        alignment_total_count="${#RECORD_USERS[@]}"
        for index in "${!RECORD_USERS[@]}"; do
            record_user="${RECORD_USERS[$index]}"
            record_action="${RECORD_ACTIONS[$index]}"
            dataset_id="${RECORD_DATASET_IDS[$index]}"
            if pipeline_validate_alignment_outcome \
                "$record_user" "$record_action" "$dataset_id"; then
                case "$PIPELINE_LAST_ALIGNMENT_STATUS" in
                    SUCCESS) alignment_success_count=$((alignment_success_count + 1)) ;;
                    SKIPPED) alignment_skipped_count=$((alignment_skipped_count + 1)) ;;
                    *) alignment_invalid_count=$((alignment_invalid_count + 1)) ;;
                esac
            else
                alignment_invalid_count=$((alignment_invalid_count + 1))
            fi
        done
        printf 'alignment_outcomes_total=%s\n' "$alignment_total_count" >>"$QA_LOG"
        printf 'alignment_outcomes_success=%s\n' "$alignment_success_count" >>"$QA_LOG"
        printf 'alignment_outcomes_skipped=%s\n' "$alignment_skipped_count" >>"$QA_LOG"
        printf 'alignment_outcomes_invalid=%s\n' "$alignment_invalid_count" >>"$QA_LOG"
        printf 'Alignment outcomes: total=%s success=%s skipped=%s invalid=%s\n' \
            "$alignment_total_count" \
            "$alignment_success_count" \
            "$alignment_skipped_count" \
            "$alignment_invalid_count"
        if [[ "$alignment_invalid_count" -ne 0 ]]; then
            pipeline_die \
                "alignment outcome validation failed for ${alignment_invalid_count} recording(s)"
        fi

        local dependency_status=0
        pipeline_alignment_outcome_dependencies_valid || dependency_status=$?
        case "$dependency_status" in
            0)
                ;;
            1)
                pipeline_die \
                    "alignment outcome dependency is missing, malformed, or stale"
                ;;
            *)
                pipeline_die \
                    "alignment outcome dependency could not be recomputed"
                ;;
        esac
    fi

    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        successful_user_action_count=0
        for record_user in "${PIPELINE_USERS[@]}"; do
            pipeline_validate_user_segmentation_state "$record_user" "$ACTION" || \
                pipeline_die "segmentation terminal state is invalid for ${record_user}/${ACTION}"
            segmentation_error_count=$((segmentation_error_count + PIPELINE_LAST_SEGMENTATION_ERROR_COUNT))
            case "$PIPELINE_LAST_SEGMENTATION_STATE" in
                PACKAGE_NO_ERRORS|PACKAGE_WITH_ERRORS)
                    successful_user_action_count=$((successful_user_action_count + 1))
                    ;;
                ALL_RECORDINGS_ERROR)
                    all_error_user_count=$((all_error_user_count + 1))
                    ;;
                *)
                    pipeline_die "unknown segmentation terminal state for ${record_user}/${ACTION}"
                    ;;
            esac
        done
        pipeline_write_segmentation_error_report
    fi
    if [[ "$summary_count" -ne "$successful_user_action_count" ]]; then
        pipeline_die "segmentation summary count ${summary_count} != successful user/action count ${successful_user_action_count}"
    fi
    if [[ "$successful_user_action_count" -eq 0 ]]; then
        if [[ "$padded_summary_count" -ne 0 ]]; then
            pipeline_die "padded dataset summary count ${padded_summary_count} != 0 without successful segmentation packages"
        fi
    elif [[ "$padded_summary_count" -ne 1 ]]; then
        pipeline_die "padded dataset summary count ${padded_summary_count} != 1"
    fi
    printf 'segmentation_successful_user_actions=%s\n' "$successful_user_action_count" >>"$QA_LOG"
    printf 'segmentation_recording_errors=%s\n' "$segmentation_error_count" >>"$QA_LOG"
    printf 'segmentation_all_error_users=%s\n' "$all_error_user_count" >>"$QA_LOG"

    for index in "${!RECORD_USERS[@]}"; do
        record_user="${RECORD_USERS[$index]}"
        record_action="${RECORD_ACTIONS[$index]}"
        dataset_id="${RECORD_DATASET_IDS[$index]}"
        pipeline_require_file \
            "$PREPROCESS_ROOT/$record_user/$record_action/$dataset_id/${dataset_id}_preprocessing.json" \
            "preprocessing metadata"
        pipeline_require_file \
            "$PREPROCESS_ROOT/$record_user/$record_action/$dataset_id/${dataset_id}_timestamps_us.npy" \
            "canonical timestamp sidecar"
        pipeline_validate_spike_artifact \
            "$SPIKE_ROOT/$record_user/$record_action/$dataset_id/spikeIMU.npy" \
            "$SPIKE_ROOT/$record_user/$record_action/$dataset_id/metadata.json" \
            "$PREPROCESS_ROOT/$record_user/$record_action/$dataset_id/${dataset_id}_timestamps_us.npy" \
            "$POST_ENCODE_TRANSFORM" \
            "${encoder_identity_args[@]}"
    done

    for record_user in "${PIPELINE_USERS[@]}"; do
        segment_directory="$SEGMENT_ROOT/$record_user/action_$ACTION"
        if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
            pipeline_validate_user_segmentation_state "$record_user" "$ACTION" || \
                pipeline_die "segmentation terminal state is invalid for ${record_user}/${ACTION}"
            if [[ "$PIPELINE_LAST_SEGMENTATION_STATE" == "ALL_RECORDINGS_ERROR" ]]; then
                continue
            fi
        fi
        pipeline_require_file \
            "$segment_directory/${record_user}_action_${ACTION}_spikeIMU.npy" \
            "segmented SpikeIMU matrix"
        pipeline_require_file \
            "$segment_directory/${record_user}_action_${ACTION}_labels.npy" \
            "segmentation labels"
        pipeline_require_file \
            "$segment_directory/${record_user}_action_${ACTION}_segment_offsets.npy" \
            "segment offsets"
        pipeline_require_file \
            "$segment_directory/${record_user}_action_${ACTION}_segment_lengths.npy" \
            "segment lengths"
        pipeline_require_file \
            "$segment_directory/${record_user}_action_${ACTION}_segments.csv" \
            "segment manifest"
        pipeline_require_file \
            "$segment_directory/${record_user}_action_${ACTION}_segmentation_summary.json" \
            "segmentation summary"
        if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
            pipeline_require_file \
                "$segment_directory/${record_user}_action_${ACTION}_board_event_targets.npy" \
                "Board event targets"
            pipeline_require_file \
                "$segment_directory/${record_user}_action_${ACTION}_board_events.csv" \
                "Board event audit"
            pipeline_validate_segment_artifact \
                "$segment_directory/${record_user}_action_${ACTION}_spikeIMU.npy" \
                "$segment_directory/${record_user}_action_${ACTION}_segmentation_summary.json" \
                "$segment_directory/${record_user}_action_${ACTION}_board_event_targets.npy"
        else
            pipeline_validate_segment_artifact \
                "$segment_directory/${record_user}_action_${ACTION}_spikeIMU.npy" \
                "$segment_directory/${record_user}_action_${ACTION}_segmentation_summary.json"
        fi
    done

    if [[ "$successful_user_action_count" -gt 0 ]]; then
        pipeline_validate_padding_artifact \
            "$PADDING_OUTPUT_ROOT/padding_dataset_summary.json" \
            "$successful_user_action_count"
    fi

    printf 'QA passed for %s ring_0 recording(s), %s user(s), %s successful packages, and %s segmentation recording error(s).\n' \
        "$ring_count" "${#PIPELINE_USERS[@]}" "$successful_user_action_count" "$segmentation_error_count"
    printf 'Output root: %s\n' "$COMBINATION_ROOT"
    printf 'Padded output root: %s\n' "$PADDING_OUTPUT_ROOT"
    printf 'Logs: %s\n' "$LOG_ROOT"
}


# >>> BEGIN CHATGPT_COMMON_ONLY_PERFORMANCE_OVERRIDES >>>
# Long-lived Python stage drivers.  These override the legacy Bash functions
# above without changing any Python source file or numerical implementation.

pipeline_batch_python_logged() {
    local log_path="$1"
    shift
    pipeline_log_command "$log_path" "${PYTHON_CMD[@]}" - "$@"
    "${PYTHON_CMD[@]}" - "$@" 2>&1 | tee -a "$log_path"
}

pipeline_batch_python_capture() {
    local log_path="$1"
    shift
    pipeline_log_command "$log_path" "${PYTHON_CMD[@]}" - "$@"
    "${PYTHON_CMD[@]}" - "$@" 2>>"$log_path"
}

pipeline_preprocess() {
    if [[ "${PIPELINE_BATCH_ORCHESTRATION:-0}" == "0" ]]; then
        pipeline_prepare_legacy_defaults
        pipeline_preprocess_legacy
        return
    fi
    pipeline_note "batched preprocess: one Python process for ${#RECORD_USERS[@]} recordings"
    local -a batch_args=(
        "$DATA_ROOT" "$PREPROCESS_ROOT" "$GRAVITY_METHOD" "$SAMPLING_RATE"
        "$LOW_PASS_CUTOFF_HZ" "$MADGWICK_BETA" "$MADGWICK_PROVISIONAL" "$OVERWRITE"
        "$LOG_ROOT/preprocess"
    )
    local index
    for index in "${!RECORD_USERS[@]}"; do
        batch_args+=(
            "${RECORD_USERS[$index]}"
            "${RECORD_ACTIONS[$index]}"
            "${RECORD_DATASET_IDS[$index]}"
        )
    done
    pipeline_batch_python_logged "$LOG_ROOT/preprocess/batch.log" "${batch_args[@]}" <<'PY_BATCH_PREPROCESS'
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys

from writingring.discovery import discover_recordings
from writingring.gravity import GravityRemovalConfig
from writingring.preprocessing_export import export_recording_preprocessing

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams:
            stream.flush()

if len(sys.argv) < 10 or (len(sys.argv) - 10) % 3:
    raise SystemExit("internal batched preprocess argument error")

data_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])
method = sys.argv[3]
sampling_rate = float(sys.argv[4])
cutoff = float(sys.argv[5])
beta = float(sys.argv[6])
provisional = bool(int(sys.argv[7]))
overwrite = bool(int(sys.argv[8]))
log_root = Path(sys.argv[9])
record_keys = [
    (sys.argv[index], sys.argv[index + 1], int(sys.argv[index + 2]))
    for index in range(10, len(sys.argv), 3)
]

recordings = discover_recordings(data_root)
by_key = {(r.user, r.action, r.dataset_id): r for r in recordings}
missing = [key for key in record_keys if key not in by_key]
if missing:
    raise SystemExit(f"batched preprocess could not resolve recordings: {missing[:5]}")

config = GravityRemovalConfig(
    sampling_rate_hz=sampling_rate,
    gravity_removal_method=method,
    low_pass_cutoff_hz=cutoff,
    madgwick_beta=beta,
    strict_calibration=(method == "madgwick" and not provisional),
)
log_root.mkdir(parents=True, exist_ok=True)
base_stdout = sys.stdout
base_stderr = sys.stderr
for number, key in enumerate(record_keys, start=1):
    user, action, dataset_id = key
    log_path = log_root / f"{user}_session_{dataset_id}.log"
    with log_path.open("a", encoding="utf-8") as log_stream:
        with redirect_stdout(Tee(base_stdout, log_stream)), redirect_stderr(Tee(base_stderr, log_stream)):
            print(f"[preprocess] {number}/{len(record_keys)} {user}/{action}/{dataset_id}", flush=True)
            result = export_recording_preprocessing(
                by_key[key],
                output_root=output_root,
                gravity_config=config,
                output_dtype="float32",
                overwrite=overwrite,
            )
            print(f"Exported {user}/{action}/{dataset_id}: {len(result.imu)} samples", flush=True)
print(f"[preprocess] completed {len(record_keys)} recording(s) in one Python process", flush=True)
PY_BATCH_PREPROCESS

    local record_user record_action dataset_id
    for index in "${!RECORD_USERS[@]}"; do
        record_user="${RECORD_USERS[$index]}"
        record_action="${RECORD_ACTIONS[$index]}"
        dataset_id="${RECORD_DATASET_IDS[$index]}"
        pipeline_require_file "$PREPROCESS_ROOT/$record_user/$record_action/$dataset_id/${dataset_id}_preprocessedIMU.npy" "preprocessed IMU artifact"
        pipeline_require_file "$PREPROCESS_ROOT/$record_user/$record_action/$dataset_id/${dataset_id}_timestamps_us.npy" "preprocessing timestamp sidecar"
        pipeline_require_file "$PREPROCESS_ROOT/$record_user/$record_action/$dataset_id/${dataset_id}_preprocessing.json" "preprocessing summary"
    done
}

pipeline_align() {
    # Both drivers retain the alignment skip policy and validate each typed
    # result with pipeline_validate_alignment_outcome:
    # --initial-interval-policy skip and --unalignable-recording-policy skip.
    if [[ "${PIPELINE_BATCH_ORCHESTRATION:-0}" == "0" ]]; then
        pipeline_prepare_legacy_defaults
        pipeline_align_legacy
        return
    fi
    pipeline_note "batched alignment: one Python process for ${#RECORD_USERS[@]} recordings"
    local -a batch_args=(
        "$PIPELINE_PROJECT_ROOT/scripts/align_ring_board.py"
        "$DATA_ROOT" "$SPIKE_ROOT" "$OFFSET_ROOT" "$ALIGNMENT_REPORT_ROOT"
        "$ALIGNMENT_VERIFICATION_ROOT" "$OVERWRITE" "$LOG_ROOT/alignment" "$ACTION"
    )
    local index
    for index in "${!RECORD_USERS[@]}"; do
        batch_args+=(
            "${RECORD_USERS[$index]}"
            "${RECORD_ACTIONS[$index]}"
            "${RECORD_DATASET_IDS[$index]}"
        )
    done
    pipeline_batch_python_logged "$LOG_ROOT/alignment/batch.log" "${batch_args[@]}" <<'PY_BATCH_ALIGN'
from contextlib import redirect_stderr, redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import writingring.board_loader as board_loader_module
import writingring.discovery as discovery_module
import writingring.recording_features as feature_module
from writingring.alignment_io import (
    build_alignment_input_provenance,
    build_board_chunk_provenance,
    validate_alignment_outcome,
)

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams:
            stream.flush()

if len(sys.argv) < 10 or (len(sys.argv) - 10) % 3:
    raise SystemExit("internal batched alignment argument error")

script_path = Path(sys.argv[1])
data_root = Path(sys.argv[2])
spike_root = Path(sys.argv[3])
offset_root = Path(sys.argv[4])
report_root = Path(sys.argv[5])
verification_root = Path(sys.argv[6])
overwrite = bool(int(sys.argv[7]))
log_root = Path(sys.argv[8])
action_expected = sys.argv[9]
record_keys = [
    (sys.argv[index], sys.argv[index + 1], int(sys.argv[index + 2]))
    for index in range(10, len(sys.argv), 3)
]
if any(action != action_expected for _, action, _ in record_keys):
    raise SystemExit("batched alignment received a recording from the wrong action")

original_discover = discovery_module.discover_recordings
recordings = original_discover(data_root)
by_key = {(r.user, r.action, r.dataset_id): r for r in recordings}
missing = [key for key in record_keys if key not in by_key]
if missing:
    raise SystemExit(f"batched alignment could not resolve recordings: {missing[:5]}")

# Reuse discovery inside every CLI main() call. This removes repeated directory scans
# while preserving align_ring_board.py as the producer implementation.
def cached_discover(root):
    if Path(root).resolve() == data_root.resolve():
        return recordings
    return original_discover(root)
discovery_module.discover_recordings = cached_discover

original_load_board = board_loader_module.load_board
board_cache = {}
def cached_load_board(source):
    key = (
        getattr(source, "user", None),
        getattr(source, "action", None),
        getattr(source, "dataset_id", None),
    )
    if None in key:
        return original_load_board(source)
    if key not in board_cache:
        board_cache[key] = original_load_board(source)
    return board_cache[key]
board_loader_module.load_board = cached_load_board

original_load_feature = feature_module.load_recording_features
feature_cache = {}
def cached_load_feature(recording, *args, **kwargs):
    input_kind = kwargs.get("input_kind", "raw-ring")
    requested_spike_root = kwargs.get("spike_root")
    key = (
        recording.user,
        recording.action,
        recording.dataset_id,
        input_kind,
        None if requested_spike_root is None else str(Path(requested_spike_root).resolve()),
    )
    if key not in feature_cache:
        feature_cache[key] = original_load_feature(recording, *args, **kwargs)
    return feature_cache[key]
feature_module.load_recording_features = cached_load_feature

spec = spec_from_file_location("writingring_align_batch_cli", script_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"could not load alignment CLI: {script_path}")
align_cli = module_from_spec(spec)
spec.loader.exec_module(align_cli)

log_root.mkdir(parents=True, exist_ok=True)
base_stdout = sys.stdout
base_stderr = sys.stderr
for number, key in enumerate(record_keys, start=1):
    user, action, dataset_id = key
    recording = by_key[key]
    log_path = log_root / f"{user}_session_{dataset_id}.log"
    argv = [
        "--data-root", str(data_root),
        "--user", user,
        "--action", action,
        "--dataset-id", str(dataset_id),
        "--input-kind", "spike-imu",
        "--spike-root", str(spike_root),
        "--offset-output-root", str(offset_root),
        "--report-output-root", str(report_root),
        "--verification-output-root", str(verification_root),
        "--initial-interval-policy", "skip",
        "--unalignable-recording-policy", "skip",
    ]
    if overwrite:
        argv.extend([
            "--overwrite-offset",
            "--overwrite-report",
            "--overwrite-verification",
            "--overwrite-outcome",
        ])
    with log_path.open("a", encoding="utf-8") as log_stream:
        with redirect_stdout(Tee(base_stdout, log_stream)), redirect_stderr(Tee(base_stderr, log_stream)):
            print(f"[align] {number}/{len(record_keys)} {user}/{action}/{dataset_id}", flush=True)
            status_code = align_cli.main(argv)
            if status_code != 0:
                raise SystemExit(
                    f"alignment producer failed for {user}/{action}/{dataset_id} with status {status_code}"
                )
            feature_input = cached_load_feature(
                recording,
                input_kind="spike-imu",
                spike_root=spike_root,
            )
            board_data = cached_load_board(recording)
            input_provenance = build_alignment_input_provenance(feature_input)
            board_provenance = build_board_chunk_provenance(board_data)
            outcome = validate_alignment_outcome(
                offset_root,
                verification_root,
                report_root,
                expected_recording={
                    "user": user,
                    "action": action,
                    "dataset_id": dataset_id,
                },
                expected_input_provenance=input_provenance,
                expected_board_provenance=board_provenance,
            )
            if outcome.status_name not in {"SUCCESS", "SKIPPED"}:
                raise SystemExit(
                    f"alignment outcome returned unsupported status {outcome.status_name}: "
                    f"{user}/{action}/{dataset_id}"
                )
            print(
                f"[align] validated outcome={outcome.status_name} "
                f"for {user}/{action}/{dataset_id}",
                flush=True,
            )
    # The cache exists only to reuse producer inputs for immediate validation.
    # Release the potentially large Board/feature arrays before the next recording.
    board_cache.pop((user, action, dataset_id), None)
    for cache_key in list(feature_cache):
        if cache_key[:3] == (user, action, dataset_id):
            feature_cache.pop(cache_key, None)
print(f"[align] completed {len(record_keys)} recording(s) in one Python process", flush=True)
PY_BATCH_ALIGN
}

pipeline_segment() {
    if [[ "${PIPELINE_BATCH_ORCHESTRATION:-0}" == "0" ]]; then
        pipeline_prepare_legacy_defaults
        pipeline_segment_legacy
        return
    fi
    local effective_overwrite="$OVERWRITE"
    if (( ${#PIPELINE_DOWNSTREAM_SEGMENT_OVERWRITE_ARGS[@]} > 0 )); then
        effective_overwrite=1
    fi
    pipeline_note "batched segmentation: one Python process for ${#PIPELINE_USERS[@]} users"
    pipeline_batch_python_logged "$LOG_ROOT/segmentation/batch.log" \
        "$PIPELINE_PROJECT_ROOT/scripts/segment_ring_imu.py" \
        "$DATA_ROOT" "$SPIKE_ROOT" "$BOUNDARY_MODE" "$SAMPLING_RATE" \
        "$SEGMENT_ROOT" "$OFFSET_ROOT" "$effective_overwrite" "$SEGMENT_VERIFICATION_DPI" \
        "$LOG_ROOT/segmentation" "$ACTION" "${PIPELINE_USERS[@]}" <<'PY_BATCH_SEGMENT'
from contextlib import redirect_stderr, redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import csv
import json
import os
import sys
import tempfile

import writingring.discovery as discovery_module

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams:
            stream.flush()

if len(sys.argv) < 13:
    raise SystemExit("internal batched segmentation argument error")
script_path = Path(sys.argv[1])
data_root = Path(sys.argv[2])
spike_root = Path(sys.argv[3])
boundary_mode = sys.argv[4]
sampling_rate = sys.argv[5]
output_root = Path(sys.argv[6])
offset_root = Path(sys.argv[7])
overwrite = bool(int(sys.argv[8]))
verification_dpi = sys.argv[9]
log_root = Path(sys.argv[10])
action = sys.argv[11]
users = sys.argv[12:]
if not users:
    raise SystemExit("batched segmentation requires at least one user")

original_discover = discovery_module.discover_recordings
recordings = original_discover(data_root)
def cached_discover(root):
    if Path(root).resolve() == data_root.resolve():
        return recordings
    return original_discover(root)
discovery_module.discover_recordings = cached_discover

spec = spec_from_file_location("writingring_segment_batch_cli", script_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"could not load segmentation CLI: {script_path}")
segment_cli = module_from_spec(spec)
spec.loader.exec_module(segment_cli)

log_root.mkdir(parents=True, exist_ok=True)
base_stdout = sys.stdout
base_stderr = sys.stderr
for number, user in enumerate(users, start=1):
    argv = [
        "--data-root", str(data_root),
        "--user", user,
        "--action", action,
        "--input-kind", "spike-imu",
        "--spike-root", str(spike_root),
        "--boundary-mode", boundary_mode,
        "--sampling-rate", sampling_rate,
        "--output-root", str(output_root),
    ]
    if boundary_mode == "aligned-board-events":
        argv.extend([
            "--alignment-offset-root", str(offset_root),
            "--pre-press-context-seconds", "0.2",
            "--post-lift-context-seconds", "0.2",
            "--maximum-segment-duration-seconds", "5.0",
            "--carry-in-press-lookback-seconds", "0.2",
            "--missing-event-policy", "skip",
            "--crossing-touch-policy", "accept_until_next_press",
            "--recording-error-policy", "skip",
            "--verification-panel-seconds", "10",
            "--verification-dpi", verification_dpi,
        ])
    if overwrite:
        argv.append("--overwrite")
    log_path = log_root / f"{user}.log"
    with log_path.open("a", encoding="utf-8") as log_stream:
        with redirect_stdout(Tee(base_stdout, log_stream)), redirect_stderr(Tee(base_stderr, log_stream)):
            print(f"[segment] {number}/{len(users)} {user}/{action}", flush=True)
            status_code = segment_cli.main(argv)
            if status_code != 0:
                raise SystemExit(
                    f"segmentation failed for {user}/{action} with status {status_code}"
                )

if boundary_mode == "aligned-board-events":
    states = []
    errors = []
    for user in users:
        stem = f"{user}_action_{action}"
        summary_path = output_root / user / f"action_{action}" / f"{stem}_segmentation_summary.json"
        error_path = output_root / "recording_errors" / user / f"action_{action}" / f"{stem}_segmentation_recording_errors.json"
        if summary_path.is_file():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            state = "completed_with_recording_errors" if payload.get("segmentation_error_recording_count", 0) else "completed"
        elif error_path.is_file():
            payload = json.loads(error_path.read_text(encoding="utf-8"))
            state = payload.get("terminal_state")
        else:
            raise SystemExit(f"missing terminal segmentation state for {user}/{action}")
        states.append({
            "user": user,
            "action": action,
            "terminal_state": state,
            "source_recording_count": payload.get("source_recording_count"),
            "processed_recording_count": payload.get("processed_recording_count"),
            "alignment_skipped_recording_count": payload.get(
                "alignment_skipped_recording_count", payload.get("skipped_recording_count")
            ),
            "segmentation_error_recording_count": payload.get("segmentation_error_recording_count"),
        })
        for error in payload.get("segmentation_errors", []):
            identity = error.get("identity", {}) if isinstance(error, dict) else {}
            errors.append({
                "user": identity.get("user"),
                "action": identity.get("action"),
                "dataset_id": identity.get("dataset_id"),
                "stage": error.get("stage") if isinstance(error, dict) else None,
                "error_type": error.get("error_type") if isinstance(error, dict) else None,
                "message": error.get("message") if isinstance(error, dict) else None,
            })
    report = {
        "schema_version": 1,
        "action": action,
        "user_action_states": states,
        "recording_error_count": len(errors),
        "recording_errors": errors,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "segmentation_recording_error_report.json"
    csv_path = output_root / "segmentation_recording_error_report.csv"
    def atomic_text(path, text):
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(text)
            os.replace(temporary, path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
    atomic_text(json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{csv_path.name}.", dir=csv_path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["user", "action", "dataset_id", "stage", "error_type", "message"],
            )
            writer.writeheader()
            writer.writerows(errors)
        os.replace(temporary, csv_path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    print(f"published segmentation recording-error report: {json_path}", flush=True)
print(f"[segment] completed {len(users)} user/action package(s) in one Python process", flush=True)
PY_BATCH_SEGMENT
}

pipeline_successful_segmentation_user_count() {
    if [[ "${PIPELINE_BATCH_ORCHESTRATION:-0}" == "0" ]]; then
        pipeline_prepare_legacy_defaults
        pipeline_successful_segmentation_user_count_legacy
        return
    fi
    if [[ "$BOUNDARY_MODE" != "aligned-board-events" ]]; then
        printf '%s\n' "${#PIPELINE_USERS[@]}"
        return 0
    fi
    pipeline_batch_python_capture "${QA_LOG:-/dev/null}" \
        "$SEGMENT_ROOT" "$ACTION" "${PIPELINE_USERS[@]}" <<'PY_BATCH_STATE_COUNT'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
action = sys.argv[2]
users = sys.argv[3:]

def load(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return value

def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemExit(f"{name} must be a nonnegative integer")
    return value

def dep_key(value):
    if not isinstance(value, dict):
        raise SystemExit("alignment dependency identity is malformed")
    identity = value.get("identity", value)
    if not isinstance(identity, dict):
        raise SystemExit("alignment dependency identity is malformed")
    return identity.get("user"), identity.get("action"), identity.get("dataset_id")

def validate_dependency(value):
    if not isinstance(value, dict):
        raise SystemExit("alignment_outcome_dependency must be an object")
    source = value.get("source_recording_ids")
    outcomes = value.get("outcomes_by_status")
    if not isinstance(source, list) or not isinstance(outcomes, dict):
        raise SystemExit("alignment outcome dependency is malformed")
    success = outcomes.get("SUCCESS")
    skipped = outcomes.get("SKIPPED")
    if not isinstance(success, list) or not isinstance(skipped, list):
        raise SystemExit("alignment outcome dependency statuses are malformed")
    source_keys = [dep_key(entry) for entry in source]
    outcome_keys = [dep_key(entry) for entry in success + skipped]
    try:
        sorted_keys = sorted(source_keys, key=lambda item: item[2])
    except TypeError as error:
        raise SystemExit("alignment dependency dataset ids are malformed") from error
    if (
        len(set(source_keys)) != len(source_keys)
        or len(set(outcome_keys)) != len(outcome_keys)
        or source_keys != sorted_keys
        or set(source_keys) != set(outcome_keys)
    ):
        raise SystemExit("alignment outcome dependency does not reconcile")
    return {dep_key(entry)[2] for entry in success}

def validate(payload, user):
    if payload.get("input_kind") != "spike-imu":
        raise SystemExit("segmentation state is not SpikeIMU")
    if payload.get("boundary_mode") != "aligned_board_events":
        raise SystemExit("segmentation state is not aligned Board mode")
    source = nonnegative(payload.get("source_recording_count"), "source_recording_count")
    processed = nonnegative(payload.get("processed_recording_count"), "processed_recording_count")
    skipped = nonnegative(payload.get("skipped_recording_count"), "skipped_recording_count")
    errors = nonnegative(
        payload.get("segmentation_error_recording_count"),
        "segmentation_error_recording_count",
    )
    entries = payload.get("segmentation_errors")
    if not isinstance(entries, list) or len(entries) != errors:
        raise SystemExit("segmentation error entries do not reconcile")
    if source != processed + skipped + errors:
        raise SystemExit("source recording counts do not reconcile")
    success_ids = validate_dependency(payload.get("alignment_outcome_dependency"))
    error_ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise SystemExit("segmentation error entry is malformed")
        identity = entry.get("identity")
        if not isinstance(identity, dict):
            raise SystemExit("segmentation error identity is malformed")
        dataset_id = identity.get("dataset_id")
        if (
            identity.get("user") != user
            or identity.get("action") != action
            or isinstance(dataset_id, bool)
            or not isinstance(dataset_id, int)
            or dataset_id < 0
            or not isinstance(entry.get("stage"), str)
            or not entry["stage"]
            or not isinstance(entry.get("error_type"), str)
            or not entry["error_type"]
            or not isinstance(entry.get("message"), str)
        ):
            raise SystemExit("segmentation error fields are malformed")
        error_ids.add(dataset_id)
    if len(error_ids) != errors or not error_ids <= success_ids:
        raise SystemExit("segmentation errors must retain alignment SUCCESS")
    return processed, errors

successful = 0
for user in users:
    stem = f"{user}_action_{action}"
    summary_path = root / user / f"action_{action}" / f"{stem}_segmentation_summary.json"
    error_path = root / "recording_errors" / user / f"action_{action}" / f"{stem}_segmentation_recording_errors.json"
    if summary_path.is_file():
        summary = load(summary_path)
        processed, errors = validate(summary, user)
        if processed <= 0:
            raise SystemExit("segmentation summary has no processed recording")
        if errors == 0:
            if error_path.is_file():
                raise SystemExit("stale recording-error report accompanies a clean package")
        else:
            if not error_path.is_file():
                raise SystemExit("segmentation errors are missing their report")
            report = load(error_path)
            if report.get("terminal_state") != "completed_with_recording_errors":
                raise SystemExit("mixed segmentation error report has wrong terminal state")
            validate(report, user)
            if (
                report.get("segmentation_errors") != summary.get("segmentation_errors")
                or report.get("alignment_outcome_dependency")
                != summary.get("alignment_outcome_dependency")
            ):
                raise SystemExit("segmentation error report differs from package summary")
        successful += 1
    elif error_path.is_file():
        report = load(error_path)
        if report.get("terminal_state") != "all_recordings_error":
            raise SystemExit("report-only segmentation state has wrong terminal state")
        processed, errors = validate(report, user)
        if processed != 0 or errors == 0:
            raise SystemExit("report-only segmentation state does not describe all-recording errors")
    else:
        raise SystemExit(f"missing terminal segmentation state for {user}/{action}")
sys.stdout.write(str(successful))
PY_BATCH_STATE_COUNT
}

pipeline_padding() {
    if [[ "${PIPELINE_BATCH_ORCHESTRATION:-0}" == "0" ]]; then
        pipeline_prepare_legacy_defaults
        pipeline_padding_legacy
        return
    fi
    local successful_user_action_count
    if ! successful_user_action_count="$(pipeline_successful_segmentation_user_count)"; then
        pipeline_die "could not validate segmentation states before padding"
    fi
    if [[ "$successful_user_action_count" -eq 0 ]]; then
        if pipeline_padding_has_any_output; then
            pipeline_die "no successful segmentation package is available, but padding outputs exist"
        fi
        pipeline_note "no successful segmentation package; skipping length analysis and padding"
        return 0
    fi
    local effective_overwrite="$OVERWRITE"
    if (( ${#PIPELINE_DOWNSTREAM_PADDING_OVERWRITE_ARGS[@]} > 0 )); then
        effective_overwrite=1
    fi
    pipeline_note "batched padding: validate/load segmented packages once, then analyze and pad"
    pipeline_batch_python_logged "$PADDING_LOG" \
        "$SEGMENT_ROOT" "$PADDING_ANALYSIS_DIR" "$PADDING_OUTPUT_ROOT" \
        "$SAMPLING_RATE" "$PADDING_COVERAGE" "$PADDING_ROUND_TO" \
        "$PADDING_RECOMMENDATION" "$PADDING_VALUE" "$effective_overwrite" <<'PY_BATCH_PADDING'
from pathlib import Path
import sys

from writingring.segment_padding import (
    DEFAULT_CANDIDATE_LENGTHS,
    analyze_segment_lengths,
    publish_padded_root,
    resolve_target_length,
    validate_segmented_root,
    write_segment_length_analysis,
)

if len(sys.argv) != 10:
    raise SystemExit("internal batched padding argument error")
input_root = Path(sys.argv[1])
analysis_dir = Path(sys.argv[2])
output_root = Path(sys.argv[3])
sampling_rate = float(sys.argv[4])
minimum_coverage = float(sys.argv[5])
round_to = int(sys.argv[6])
recommendation = sys.argv[7]
padding_value = float(sys.argv[8])
overwrite = bool(int(sys.argv[9]))

print("[padding] loading and validating segmented packages once", flush=True)
datasets = validate_segmented_root(input_root)
analysis = analyze_segment_lengths(
    datasets,
    sampling_rate_hz=sampling_rate,
    candidate_lengths=DEFAULT_CANDIDATE_LENGTHS,
    minimum_coverage=minimum_coverage,
    round_to=round_to,
)
paths = write_segment_length_analysis(
    analysis,
    input_root=input_root,
    output_dir=analysis_dir,
    datasets=datasets,
    overwrite=overwrite,
)
stats = analysis["length_statistics"]
pure = analysis["recommendations"]["pure_padding"]
print(
    f"Analyzed {analysis['segment_count']} segments in {analysis['user_action_count']} "
    f"user/action packages. Range: {stats['minimum']}–{stats['maximum']} samples.",
    flush=True,
)
print(
    f"Pure-padding target: {pure['target_length']} samples "
    f"({pure['target_length'] / sampling_rate:.3f} seconds).",
    flush=True,
)
print(f"Analysis report: {paths['json']}", flush=True)

target = resolve_target_length(
    target_length=None,
    analysis_report=paths["json"],
    recommendation=recommendation,
    datasets=datasets,
    input_root=input_root,
)
summary = publish_padded_root(
    datasets,
    input_root=input_root,
    output_root=output_root,
    target_length=target,
    sampling_rate_hz=sampling_rate,
    padding_value=padding_value,
    overwrite=overwrite,
)
print(
    f"Published {summary['segment_count']} padded segments in "
    f"{summary['processed_user_action_count']} user/action packages at target length {target}; "
    f"skipped {summary['skipped_segment_count']} overlong segments.",
    flush=True,
)
print(f"Output root: {output_root}", flush=True)
PY_BATCH_PADDING
    pipeline_require_file "$PADDING_ANALYSIS_DIR/segment_length_analysis.json" "segment-length analysis report"
    pipeline_require_file "$PADDING_OUTPUT_ROOT/padding_dataset_summary.json" "padded dataset summary"
    pipeline_require_file "$PADDING_OUTPUT_ROOT/padding_dataset_manifest.csv" "padded dataset manifest"
}

pipeline_qa() {
    # The batched validator below performs the same provenance check as
    # pipeline_validate_spike_artifact "$POST_ENCODE_TRANSFORM".
    if [[ "${PIPELINE_BATCH_QA:-0}" == "0" ]]; then
        pipeline_prepare_legacy_defaults
        pipeline_qa_legacy
        return
    fi
    pipeline_note "running batched final QA in one Python process"
    pipeline_log_command "$QA_LOG" "${PYTHON_CMD[@]}" - \
        --data-root "$DATA_ROOT" \
        --preprocess-root "$PREPROCESS_ROOT" \
        --spike-root "$SPIKE_ROOT" \
        --offset-root "$OFFSET_ROOT" \
        --alignment-verification-root "$ALIGNMENT_VERIFICATION_ROOT" \
        --alignment-report-root "$ALIGNMENT_REPORT_ROOT" \
        --segmentation-root "$SEGMENT_ROOT" \
        --padding-output-root "$PADDING_OUTPUT_ROOT" \
        --combination-root "$COMBINATION_ROOT" \
        --log-root "$LOG_ROOT" \
        --action "$ACTION" \
        --boundary-mode "$BOUNDARY_MODE" \
        --post-encode-transform "$POST_ENCODE_TRANSFORM" \
        --expected-ring-count "${#RING_FILES[@]}" \
        --expected-user-count "${#PIPELINE_USERS[@]}"
    "${PYTHON_CMD[@]}" - \
        --data-root "$DATA_ROOT" \
        --preprocess-root "$PREPROCESS_ROOT" \
        --spike-root "$SPIKE_ROOT" \
        --offset-root "$OFFSET_ROOT" \
        --alignment-verification-root "$ALIGNMENT_VERIFICATION_ROOT" \
        --alignment-report-root "$ALIGNMENT_REPORT_ROOT" \
        --segmentation-root "$SEGMENT_ROOT" \
        --padding-output-root "$PADDING_OUTPUT_ROOT" \
        --combination-root "$COMBINATION_ROOT" \
        --log-root "$LOG_ROOT" \
        --action "$ACTION" \
        --boundary-mode "$BOUNDARY_MODE" \
        --post-encode-transform "$POST_ENCODE_TRANSFORM" \
        --expected-ring-count "${#RING_FILES[@]}" \
        --expected-user-count "${#PIPELINE_USERS[@]}" <<'PY_BATCH_QA' 2>&1 | tee -a "$QA_LOG"
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys

import numpy as np

from writingring.alignment_io import (
    build_alignment_input_provenance,
    build_board_chunk_provenance,
    read_alignment_skip_artifact,
    validate_alignment_outcome,
)
from writingring.board_loader import load_board
from writingring.discovery import discover_recordings
from writingring.preprocessing_io import sha256_file
from writingring.recording_features import load_recording_features


class QAError(RuntimeError):
    pass


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--preprocess-root', type=Path, required=True)
    p.add_argument('--spike-root', type=Path, required=True)
    p.add_argument('--offset-root', type=Path, required=True)
    p.add_argument('--alignment-verification-root', type=Path, required=True)
    p.add_argument('--alignment-report-root', type=Path, required=True)
    p.add_argument('--segmentation-root', type=Path, required=True)
    p.add_argument('--padding-output-root', type=Path, required=True)
    p.add_argument('--combination-root', type=Path, required=True)
    p.add_argument('--log-root', type=Path, required=True)
    p.add_argument('--action', required=True)
    p.add_argument('--boundary-mode', choices=('label','aligned-board-events'), required=True)
    p.add_argument('--post-encode-transform', choices=('none','AbsRectify','PolaritySplitAbs'), required=True)
    p.add_argument('--expected-ring-count', type=int, required=True)
    p.add_argument('--expected-user-count', type=int, required=True)
    return p.parse_args()


def load_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        raise QAError(f'could not read JSON {path}: {e}') from e
    if not isinstance(value, dict):
        raise QAError(f'JSON is not an object: {path}')
    return value


def require(path: Path, what: str):
    if not path.is_file() or path.stat().st_size <= 0:
        raise QAError(f'missing or empty {what}: {path}')


def count(root: Path, pattern: str):
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob(pattern) if p.is_file())


def progress(label: str, i: int, n: int, detail: str = ''):
    stride = 5 if n >= 10 else 1
    if i == 1 or i == n or i % stride == 0:
        print(f'[qa] {label}: {i}/{n}' + (f' {detail}' if detail else ''), flush=True)


def validate_transform(metadata_path: Path, requested: str):
    metadata = load_json(metadata_path)
    spike = metadata.get('spike_imu') or {}
    expected_channels, expected_schema, expected_event_representation, expected_event_schema, expected_event_channels = {
        'none': (21, 'signed_wavelet_events_plus_imu_v1', 'signed', 'custom_wavelet_signed_events_v1', 15),
        'AbsRectify': (21, 'signed_wavelet_events_plus_imu_v1', 'unsigned', 'custom_wavelet_abs_rectified_events_v1', 15),
        'PolaritySplitAbs': (36, 'polarity_split_wavelet_events_plus_imu_v1', 'unsigned', 'custom_wavelet_polarity_split_abs_events_v1', 30),
    }[requested]
    if (
        not isinstance(spike, dict)
        or spike.get('channel_count') != expected_channels
        or spike.get('schema') != expected_schema
        or spike.get('event_representation') != expected_event_representation
        or spike.get('event_feature_schema') != expected_event_schema
        or spike.get('event_channel_count') != expected_event_channels
    ):
        raise QAError(
            f'metadata layout mismatch (expected {expected_schema}/{expected_channels}): {metadata_path}'
        )
    settings = metadata.get('settings') or {}
    if not isinstance(settings, dict):
        raise QAError(f'invalid settings object: {metadata_path}')
    expected = None if requested == 'none' else requested
    actual = settings.get('post_encode_transform')
    if actual != expected:
        raise QAError(f'SpikeIMU post_encode_transform mismatch: expected={expected!r}, actual={actual!r}: {metadata_path}')


def seg_state_paths(root: Path, user: str, action: str):
    stem = f'{user}_action_{action}'
    summary = root / user / f'action_{action}' / f'{stem}_segmentation_summary.json'
    error = root / 'recording_errors' / user / f'action_{action}' / f'{stem}_segmentation_recording_errors.json'
    return summary, error


def validate_seg_dependency(root: Path, user: str, action: str, expected):
    summary, error = seg_state_paths(root, user, action)
    path = summary if summary.is_file() else error
    if not path.is_file():
        raise QAError(f'missing segmentation terminal state for {user}/{action}')
    payload = load_json(path)
    if payload.get('alignment_outcome_dependency') != expected:
        raise QAError(f'alignment outcome dependency is missing, malformed, or stale: {user}/{action}')
    processed = payload.get('processed_recording_count')
    errors = payload.get('segmentation_error_recording_count')
    if not isinstance(processed, int) or processed < 0 or not isinstance(errors, int) or errors < 0:
        raise QAError(f'invalid segmentation counts for {user}/{action}')
    if summary.is_file():
        state = 'PACKAGE_WITH_ERRORS' if errors else 'PACKAGE_NO_ERRORS'
        if errors and not error.is_file():
            raise QAError(f'segmentation errors are missing their report for {user}/{action}')
        if not errors and error.is_file():
            raise QAError(f'stale recording-error report for clean package {user}/{action}')
    else:
        if payload.get('terminal_state') != 'all_recordings_error' or processed != 0 or errors <= 0:
            raise QAError(f'invalid all-recordings-error state for {user}/{action}')
        state = 'ALL_RECORDINGS_ERROR'
    return state, errors


def validate_segment_package(root: Path, user: str, action: str, board_mode: bool):
    d = root / user / f'action_{action}'
    stem = f'{user}_action_{action}'
    values_path = d / f'{stem}_spikeIMU.npy'
    labels_path = d / f'{stem}_labels.npy'
    offsets_path = d / f'{stem}_segment_offsets.npy'
    lengths_path = d / f'{stem}_segment_lengths.npy'
    summary_path = d / f'{stem}_segmentation_summary.json'
    manifest_path = d / f'{stem}_segments.csv'
    for p, what in ((values_path,'segmented SpikeIMU'),(labels_path,'labels'),(offsets_path,'offsets'),(lengths_path,'lengths'),(summary_path,'summary'),(manifest_path,'manifest')):
        require(p, what)
    values = np.load(values_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    offsets = np.load(offsets_path, allow_pickle=False)
    lengths = np.load(lengths_path, allow_pickle=False)
    summary = load_json(summary_path)
    expected_channels = {
        'signed_wavelet_events_plus_imu_v1': 21,
        'polarity_split_wavelet_events_plus_imu_v1': 36,
    }.get(summary.get('feature_schema'))
    if summary.get('channel_count') != expected_channels:
        raise QAError(f'invalid segmented SpikeIMU schema: {summary_path}')
    if values.ndim != 2 or values.shape[1] != expected_channels or not np.isfinite(values).all():
        raise QAError(f'invalid segmented SpikeIMU matrix: {values_path} shape={values.shape}')
    if labels.ndim != 1 or lengths.ndim != 1 or len(labels) != len(lengths):
        raise QAError(f'segment labels/lengths mismatch under {d}')
    if offsets.ndim != 1 or len(offsets) != len(lengths) + 1 or int(offsets[0]) != 0 or int(offsets[-1]) != len(values):
        raise QAError(f'invalid segment offsets under {d}')
    if np.any(lengths < 0) or not np.array_equal(np.diff(offsets), lengths):
        raise QAError(f'segment lengths do not match offsets under {d}')
    if board_mode:
        targets_path = d / f'{stem}_board_event_targets.npy'
        events_path = d / f'{stem}_board_events.csv'
        require(targets_path, 'Board targets')
        require(events_path, 'Board audit')
        targets = np.load(targets_path, allow_pickle=False)
        if targets.shape != (len(values), 4) or targets.dtype != np.dtype(bool):
            raise QAError(f'invalid Board target matrix: {targets_path}')


def validate_padding(summary_path: Path, segment_root: Path, expected_users: int):
    require(summary_path, 'padded dataset summary')
    s = load_json(summary_path)
    if s.get('input_root') != str(segment_root.resolve()):
        raise QAError('padded summary input root mismatch')
    expected_channels = {
        'signed_wavelet_events_plus_imu_v1': 21,
        'polarity_split_wavelet_events_plus_imu_v1': 36,
    }.get(s.get('feature_schema'))
    if s.get('input_kind') != 'spike-imu' or s.get('channel_count') != expected_channels:
        raise QAError('padded summary schema mismatch')
    if s.get('processed_user_action_count') != expected_users:
        raise QAError(f"padded package count {s.get('processed_user_action_count')} != user count {expected_users}")
    source, exported, skipped = s.get('source_segment_count'), s.get('segment_count'), s.get('skipped_segment_count')
    if not all(isinstance(v, int) and v >= 0 for v in (source, exported, skipped)) or exported + skipped != source:
        raise QAError('invalid padded segment counts')
    print(f"validated padded output segments={exported} skipped={skipped} target={s.get('target_length')}: {summary_path}", flush=True)


def run(a):
    recordings = [r for r in discover_recordings(a.data_root) if r.action == a.action]
    recordings.sort(key=lambda r: (r.user, r.dataset_id))
    users = list(dict.fromkeys(r.user for r in recordings))
    if len(recordings) != a.expected_ring_count:
        raise QAError(f'discovered recording count {len(recordings)} != ring_0 count {a.expected_ring_count}')
    if len(users) != a.expected_user_count:
        raise QAError(f'discovered user count {len(users)} != expected user count {a.expected_user_count}')

    n = len(recordings)
    pre_count = count(a.preprocess_root, '*_preprocessing.json')
    spike_count = count(a.spike_root, 'spikeIMU.npy')
    seg_count = count(a.segmentation_root, '*_segmentation_summary.json')
    padded_count = count(a.padding_output_root, 'padding_dataset_summary.json')
    print(f'[qa] batched QA started: recordings={n} users={len(users)} boundary={a.boundary_mode}', flush=True)
    print(f'ring_0_recordings={n}\npreprocessing_summaries={pre_count}\nspikeIMU_matrices={spike_count}\nsegmentation_summaries={seg_count}\npadded_dataset_summaries={padded_count}')
    if pre_count != n or spike_count != n:
        raise QAError('preprocess/spike artifact count does not match recording count')

    deps = {u: {'source_recording_ids': [], 'outcomes_by_status': {'SUCCESS': [], 'SKIPPED': []}} for u in users}
    success = skipped = invalid = 0

    for i, r in enumerate(recordings, 1):
        progress('recording validation', i, n, f'{r.user}/{r.action}/{r.dataset_id}')
        pre_dir = a.preprocess_root / r.user / r.action / str(r.dataset_id)
        spike_dir = a.spike_root / r.user / r.action / str(r.dataset_id)
        timestamps = pre_dir / f'{r.dataset_id}_timestamps_us.npy'
        metadata = spike_dir / 'metadata.json'
        require(pre_dir / f'{r.dataset_id}_preprocessing.json', 'preprocessing metadata')
        require(timestamps, 'timestamp sidecar')
        require(spike_dir / 'spikeIMU.npy', 'SpikeIMU')
        require(metadata, 'SpikeIMU metadata')
        validate_transform(metadata, a.post_encode_transform)

        if a.boundary_mode == 'label':
            values = np.load(spike_dir / 'spikeIMU.npy', allow_pickle=False)
            ts = np.load(timestamps, allow_pickle=False)
            expected_channels = 36 if a.post_encode_transform == 'PolaritySplitAbs' else 21
            if values.ndim != 2 or values.shape[1] != expected_channels or len(ts) != len(values) or not np.isfinite(values).all():
                raise QAError(f'invalid SpikeIMU/timestamp artifact for {r.user}/{r.dataset_id}')
            continue

        ident = {'user': r.user, 'action': r.action, 'dataset_id': r.dataset_id}
        deps[r.user]['source_recording_ids'].append(ident)
        try:
            with redirect_stdout(StringIO()):
                feature = load_recording_features(r, input_kind='spike-imu', spike_root=a.spike_root)
                board = load_board(r)
                input_prov = build_alignment_input_provenance(feature)
                board_prov = build_board_chunk_provenance(board)
                outcome = validate_alignment_outcome(
                    a.offset_root, a.alignment_verification_root, a.alignment_report_root,
                    expected_recording=ident,
                    expected_input_provenance=input_prov,
                    expected_board_provenance=board_prov,
                )
            status = outcome.status_name
            if status not in {'SUCCESS','SKIPPED'}:
                raise ValueError(status)
            entry = {'identity': ident, 'report_sha256': sha256_file(outcome.paths.report_path)}
            if status == 'SKIPPED':
                with redirect_stdout(StringIO()):
                    skip = read_alignment_skip_artifact(
                        outcome.paths.skip_json_path,
                        expected_user=r.user,
                        expected_action=r.action,
                        expected_dataset_id=r.dataset_id,
                        expected_input_provenance=input_prov,
                        expected_board_provenance=board_prov,
                    )
                entry['reason'] = skip.reason
                skipped += 1
            else:
                success += 1
            deps[r.user]['outcomes_by_status'][status].append(entry)
        except Exception as e:
            invalid += 1
            print(f'[qa] alignment validation failed {r.user}/{r.action}/{r.dataset_id}: {e}', file=sys.stderr, flush=True)

    if a.boundary_mode == 'aligned-board-events':
        print(f'Alignment outcomes: total={n} success={success} skipped={skipped} invalid={invalid}', flush=True)
        if invalid:
            raise QAError(f'alignment outcome validation failed for {invalid} recording(s)')
        successful_users = []
        seg_errors = 0
        for i, u in enumerate(users, 1):
            progress('segmentation state validation', i, len(users), u)
            state, errors = validate_seg_dependency(a.segmentation_root, u, a.action, deps[u])
            seg_errors += errors
            if state != 'ALL_RECORDINGS_ERROR':
                successful_users.append(u)
    else:
        successful_users = users
        seg_errors = 0

    if seg_count != len(successful_users):
        raise QAError(f'segmentation summary count {seg_count} != successful user/action count {len(successful_users)}')
    if successful_users and padded_count != 1:
        raise QAError(f'padded dataset summary count {padded_count} != 1')
    if not successful_users and padded_count != 0:
        raise QAError('padded output exists without successful segmentation packages')

    for i, u in enumerate(successful_users, 1):
        progress('segmented package validation', i, len(successful_users), u)
        validate_segment_package(a.segmentation_root, u, a.action, a.boundary_mode == 'aligned-board-events')

    if successful_users:
        validate_padding(a.padding_output_root / 'padding_dataset_summary.json', a.segmentation_root, len(successful_users))

    print(f'QA passed for {n} ring_0 recording(s), {len(users)} user(s), {len(successful_users)} successful packages, and {seg_errors} segmentation recording error(s).')
    print(f'Output root: {a.combination_root}')
    print(f'Padded output root: {a.padding_output_root}')
    print(f'Logs: {a.log_root}')


def main():
    try:
        run(args_parser())
    except QAError as e:
        print(f'error: {e}', file=sys.stderr, flush=True)
        return 1
    except Exception as e:
        print(f'error: unexpected batched QA failure: {e}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
PY_BATCH_QA
}
# <<< END CHATGPT_COMMON_ONLY_PERFORMANCE_OVERRIDES <<<

run_action0_pipeline() {
    local method="$1"
    local boundary="$2"
    pipeline_init "$method" "$boundary"
    pipeline_note "execution mode: ${PIPELINE_MODE}"
    pipeline_discover

    if [[ "$PIPELINE_MODE" == "overwrite" ]]; then
        pipeline_note "overwrite mode: skipping resume checks and rebuilding from preprocess"
        PIPELINE_RESUME_STAGE="preprocess"
    else
        pipeline_note "continue mode: checking existing stage outputs"
        pipeline_plan_continue
    fi

    pipeline_execute_from_stage "$PIPELINE_RESUME_STAGE"
    pipeline_qa
}
