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
PIPELINE_PROJECT_ROOT="$(cd -- "${_PIPELINE_COMMON_DIR}/../.." && pwd)"

declare -a RING_FILES=()
declare -a RECORD_USERS=()
declare -a RECORD_ACTIONS=()
declare -a RECORD_DATASET_IDS=()
declare -a RECORD_DATASET_TOKENS=()
declare -a PIPELINE_USERS=()
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
    if [[ "$OVERWRITE" == "1" ]]; then
        PREPROCESS_OVERWRITE_ARGS+=(--overwrite)
        ENCODE_OVERWRITE_ARGS+=(--overwrite)
        ALIGN_OVERWRITE_ARGS+=(--overwrite-offset --overwrite-report --overwrite-verification)
        SEGMENT_OVERWRITE_ARGS+=(--overwrite)
    fi
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
    ENCODER_SETTINGS="$(_pipeline_resolve_path "${ENCODER_SETTINGS:-configs/spike_encoding/custom_wavelet.json}")"
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
    CONDA_ENV="${CONDA_ENV:-writingring-viz}"
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

    pipeline_configure_overwrite_args
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

pipeline_preprocess() {
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
    )
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

pipeline_align() {
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
        )
        command_args+=("${ALIGN_OVERWRITE_ARGS[@]}")
        pipeline_run_logged "$log_path" "${command_args[@]}"
        pipeline_require_file \
            "$OFFSET_ROOT/$record_user/action_${record_action}/${dataset_id}_ring_board_offset.txt" \
            "alignment offset"
        pipeline_require_file \
            "$ALIGNMENT_REPORT_ROOT/$record_user/action_${record_action}/${dataset_id}_alignment_report.json" \
            "alignment report"
        pipeline_require_file \
            "$ALIGNMENT_VERIFICATION_ROOT/$record_user/action_${record_action}/${dataset_id}_alignment_verification.png" \
            "alignment verification image"
    done
}

pipeline_segment() {
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
                --missing-event-policy skip
                --crossing-touch-policy accept_until_next_press
                --verification-panel-seconds 10
                --verification-dpi 200
                "${SEGMENT_OVERWRITE_ARGS[@]}"
            )
        else
            command_args+=("${SEGMENT_OVERWRITE_ARGS[@]}")
        fi
        pipeline_run_logged "$log_path" "${command_args[@]}"
    done
}

pipeline_padding() {
    local analysis_report="$PADDING_ANALYSIS_DIR/segment_length_analysis.json"
    local -a analysis_args=(
        "${PYTHON_CMD[@]}" scripts/analyze_segment_lengths.py
        --input-root "$SEGMENT_ROOT"
        --output-dir "$PADDING_ANALYSIS_DIR"
        --sampling-rate "$SAMPLING_RATE"
        --minimum-coverage "$PADDING_COVERAGE"
        --round-to "$PADDING_ROUND_TO"
    )
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
    if [[ "$OVERWRITE" == "1" ]]; then
        padding_args+=(--overwrite)
    fi
    pipeline_run_logged "$PADDING_LOG" "${padding_args[@]}"
    pipeline_require_file "$PADDING_OUTPUT_ROOT/padding_dataset_summary.json" "padded dataset summary"
    pipeline_require_file "$PADDING_OUTPUT_ROOT/padding_dataset_manifest.csv" "padded dataset manifest"
}

pipeline_validate_spike_artifact() {
    local values_path="$1"
    local metadata_path="$2"
    local timestamps_path="$3"
    pipeline_run_logged "$QA_LOG" "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import json
import numpy as np
import sys

values_path, metadata_path, timestamps_path = map(Path, sys.argv[1:])
values = np.load(values_path, allow_pickle=False)
timestamps = np.load(timestamps_path, allow_pickle=False)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if values.ndim != 2 or values.shape[1] != 21 or not np.isfinite(values).all():
    raise SystemExit(f"invalid SpikeIMU matrix: {values_path} shape={values.shape}")
if timestamps.ndim != 1 or len(timestamps) != len(values):
    raise SystemExit(f"timestamp row count mismatch: {timestamps_path}")
spike_imu = metadata.get("spike_imu", {})
if spike_imu.get("channel_count") != 21:
    raise SystemExit(f"metadata does not declare 21 SpikeIMU channels: {metadata_path}")
print(f"validated SpikeIMU rows={len(values)} channels={values.shape[1]}: {values_path}")
' "$values_path" "$metadata_path" "$timestamps_path"
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
json.loads(summary_path.read_text(encoding="utf-8"))
if values.ndim != 2 or values.shape[1] != 21 or not np.isfinite(values).all():
    raise SystemExit(f"invalid segmented SpikeIMU matrix: {values_path} shape={values.shape}")
if targets.shape != (len(values), 4) or targets.dtype != np.dtype(bool):
    raise SystemExit(f"invalid Board target matrix: {targets_path} shape={targets.shape} dtype={targets.dtype}")
print(f"validated segmented SpikeIMU rows={len(values)} channels=21 targets=4: {values_path}")
' "$values_path" "$summary_path" "$board_targets_path"
    else
        pipeline_run_logged "$QA_LOG" "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import json
import numpy as np
import sys

values_path, summary_path = map(Path, sys.argv[1:])
values = np.load(values_path, allow_pickle=False)
json.loads(summary_path.read_text(encoding="utf-8"))
if values.ndim != 2 or values.shape[1] != 21 or not np.isfinite(values).all():
    raise SystemExit(f"invalid segmented SpikeIMU matrix: {values_path} shape={values.shape}")
print(f"validated segmented SpikeIMU rows={len(values)} channels=21: {values_path}")
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
if summary.get("feature_schema") != "signed_wavelet_events_plus_imu_v1":
    raise SystemExit(f"padded summary feature schema mismatch: {summary_path}")
if summary.get("channel_count") != 21:
    raise SystemExit(f"padded summary channel count is not 21: {summary_path}")
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

pipeline_validate_alignment_artifact() {
    local offset_path="$1"
    local report_path="$2"
    pipeline_run_logged "$QA_LOG" "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import json
import math
import sys

offset_path, report_path = map(Path, sys.argv[1:])
try:
    offset = float(offset_path.read_text(encoding="utf-8").strip())
except (OSError, ValueError) as error:
    raise SystemExit(f"invalid alignment offset {offset_path}: {error}") from error
if not math.isfinite(offset):
    raise SystemExit(f"alignment offset is not finite: {offset_path}")
report = json.loads(report_path.read_text(encoding="utf-8"))
if not isinstance(report, dict):
    raise SystemExit(f"alignment report is not a JSON object: {report_path}")
print(f"validated alignment offset={offset}: {offset_path}")
' "$offset_path" "$report_path"
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

if values.ndim != 2 or values.shape[1] != 21 or not np.isfinite(values).all():
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
           -o -name '*_board_events.csv' \) -print -quit 2>/dev/null || true)"
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
    local index record_user record_action dataset_id directory timestamps_path
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
            "$timestamps_path" || return 1
    done
}

pipeline_alignment_outputs_valid() {
    local index record_user record_action dataset_id offset_path report_path verification_path
    for index in "${!RECORD_USERS[@]}"; do
        record_user="${RECORD_USERS[$index]}"
        record_action="${RECORD_ACTIONS[$index]}"
        dataset_id="${RECORD_DATASET_IDS[$index]}"
        offset_path="$OFFSET_ROOT/$record_user/action_${record_action}/${dataset_id}_ring_board_offset.txt"
        report_path="$ALIGNMENT_REPORT_ROOT/$record_user/action_${record_action}/${dataset_id}_alignment_report.json"
        verification_path="$ALIGNMENT_VERIFICATION_ROOT/$record_user/action_${record_action}/${dataset_id}_alignment_verification.png"
        [[ -s "$offset_path" ]] || return 1
        [[ -s "$report_path" ]] || return 1
        [[ -s "$verification_path" ]] || return 1
        pipeline_validate_alignment_artifact "$offset_path" "$report_path" || return 1
    done
}

pipeline_segment_outputs_valid() {
    local record_user segment_directory board_mode=0
    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        board_mode=1
    fi
    for record_user in "${PIPELINE_USERS[@]}"; do
        segment_directory="$SEGMENT_ROOT/$record_user/action_$ACTION"
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
    [[ -s "$analysis_report" ]] || return 1
    [[ -s "$summary_path" ]] || return 1
    [[ -s "$manifest_path" ]] || return 1
    pipeline_validate_padding_artifact "$summary_path" "${#PIPELINE_USERS[@]}" || return 1
}

pipeline_force_full_rebuild() {
    local reason="$1"
    PIPELINE_RESUME_STAGE="preprocess"
    PIPELINE_FORCE_REBUILD=1
    OVERWRITE=1
    pipeline_configure_overwrite_args
    pipeline_note "continue validation failed: ${reason}; rebuilding from preprocess with overwrite enabled"
}

pipeline_plan_continue() {
    local preprocess_any=0 encode_any=0 alignment_any=0 segment_any=0 padding_any=0
    PIPELINE_RESUME_STAGE="preprocess"
    PIPELINE_FORCE_REBUILD=0

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
    if ! pipeline_encode_outputs_valid; then
        pipeline_force_full_rebuild "encode outputs are partial or invalid"
        return 0
    fi
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

pipeline_qa() {
    local ring_count="${#RING_FILES[@]}"
    local preprocessing_count spike_count offset_count summary_count padded_summary_count
    preprocessing_count="$(pipeline_count_files "$PREPROCESS_ROOT" '*_preprocessing.json')"
    spike_count="$(pipeline_count_files "$SPIKE_ROOT" 'spikeIMU.npy')"
    summary_count="$(pipeline_count_files "$SEGMENT_ROOT" '*_segmentation_summary.json')"
    padded_summary_count="$(pipeline_count_files "$PADDING_OUTPUT_ROOT" 'padding_dataset_summary.json')"
    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        offset_count="$(pipeline_count_files "$OFFSET_ROOT" '*_ring_board_offset.txt')"
    else
        offset_count=0
    fi

    printf 'ring_0_recordings=%s\n' "$ring_count" >>"$QA_LOG"
    printf 'preprocessing_summaries=%s\n' "$preprocessing_count" >>"$QA_LOG"
    printf 'spikeIMU_matrices=%s\n' "$spike_count" >>"$QA_LOG"
    printf 'alignment_offsets=%s\n' "$offset_count" >>"$QA_LOG"
    printf 'segmentation_summaries=%s\n' "$summary_count" >>"$QA_LOG"
    printf 'padded_dataset_summaries=%s\n' "$padded_summary_count" >>"$QA_LOG"

    if [[ "$preprocessing_count" -ne "$ring_count" ]]; then
        pipeline_die "preprocessing summary count ${preprocessing_count} != ring_0 count ${ring_count}"
    fi
    if [[ "$spike_count" -ne "$ring_count" ]]; then
        pipeline_die "SpikeIMU count ${spike_count} != ring_0 count ${ring_count}"
    fi
    if [[ "$BOUNDARY_MODE" == "aligned-board-events" && "$offset_count" -ne "$ring_count" ]]; then
        pipeline_die "alignment offset count ${offset_count} != ring_0 count ${ring_count}"
    fi
    if [[ "$summary_count" -ne "${#PIPELINE_USERS[@]}" ]]; then
        pipeline_die "segmentation summary count ${summary_count} != user count ${#PIPELINE_USERS[@]}"
    fi
    if [[ "$padded_summary_count" -ne 1 ]]; then
        pipeline_die "padded dataset summary count ${padded_summary_count} != 1"
    fi

    local index record_user record_action dataset_id segment_directory
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
            "$PREPROCESS_ROOT/$record_user/$record_action/$dataset_id/${dataset_id}_timestamps_us.npy"
    done

    for record_user in "${PIPELINE_USERS[@]}"; do
        segment_directory="$SEGMENT_ROOT/$record_user/action_$ACTION"
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

    pipeline_validate_padding_artifact \
        "$PADDING_OUTPUT_ROOT/padding_dataset_summary.json" \
        "${#PIPELINE_USERS[@]}"

    printf 'QA passed for %s ring_0 recording(s), %s user(s).\n' "$ring_count" "${#PIPELINE_USERS[@]}"
    printf 'Output root: %s\n' "$COMBINATION_ROOT"
    printf 'Padded output root: %s\n' "$PADDING_OUTPUT_ROOT"
    printf 'Logs: %s\n' "$LOG_ROOT"
}

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
