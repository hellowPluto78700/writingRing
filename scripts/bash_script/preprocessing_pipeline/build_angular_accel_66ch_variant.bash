#!/usr/bin/env bash
set -Eeuo pipefail

# Build the 64 Hz 66-channel linear/angular-acceleration SpikeIMU variant.
#
# Source contract:
#   completed 64 Hz polarity-split pipeline root
#   30 linear-acceleration events + trailing 6 IMU = 36 channels
#
# Output contract:
#   30 linear-acceleration events
# + 30 angular-acceleration events
# + trailing 6 IMU
# = 66 channels
#
# Modes:
#   MODE=local    one local process per user, capped by JOBS (default)
#   MODE=submit   submit one Slurm array task per user + afterok finalizer
#   MODE=worker   internal/user-specific worker
#   MODE=finalize internal/final aggregation
#
# Usage:
#   bash scripts/bash_script/preprocessing_pipeline/build_angular_accel_66ch_variant.bash \
#       SOURCE_COMBINATION_ROOT [OUTPUT_ROOT]
#
# Default OUTPUT_ROOT is a sibling named <SOURCE_COMBINATION_ROOT>_angular_accel66.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
PYTHON_HELPER="${REPO_ROOT}/scripts/build_angular_accel_66ch_variant.py"

SOURCE_ROOT="${1:-}"
OUTPUT_ROOT="${2:-}"
MODE="${MODE:-local}"
OVERWRITE_DEST="${OVERWRITE_DEST:-0}"
SAMPLING_RATE_HZ="${SAMPLING_RATE_HZ:-64}"
FREQUENCIES_HZ="${FREQUENCIES_HZ:-0.5 1 2 4 8}"
MAX_FILTER_TIME_S="${MAX_FILTER_TIME_S:-0.3}"
JOBS="${JOBS:-}"
SLURM_MAX_CONCURRENCY="${SLURM_MAX_CONCURRENCY:-50}"
SLURM_TIME="${SLURM_TIME:-02:00:00}"
SLURM_MEM="${SLURM_MEM:-4G}"
USER_NAME="${USER_NAME:-}"
PYTHON_BIN="${PYTHON_BIN:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

fail() {
    printf 'error: %s\n' "$1" >&2
    exit 2
}

usage() {
    cat <<'USAGE'
Usage:
  MODE=local bash scripts/bash_script/preprocessing_pipeline/build_angular_accel_66ch_variant.bash \
      SOURCE_COMBINATION_ROOT [OUTPUT_ROOT]

  MODE=submit bash scripts/bash_script/preprocessing_pipeline/build_angular_accel_66ch_variant.bash \
      SOURCE_COMBINATION_ROOT [OUTPUT_ROOT]

Environment:
  MODE=local|submit|worker|finalize     default: local
  JOBS=<positive integer>              local concurrent users; default: min(nproc, 50)
  OVERWRITE_DEST=0|1                   replace this derived variant when 1
  PYTHON_BIN=/path/to/python           optional explicit interpreter override
  SAMPLING_RATE_HZ=64                  fixed contract; non-64 values are rejected
  FREQUENCIES_HZ="0.5 1 2 4 8"        exactly five Custom Wavelet frequencies
  MAX_FILTER_TIME_S=0.3                extrema max-filter duration
  SLURM_MAX_CONCURRENCY=50             maximum simultaneous user tasks, hard-capped at 50
  SLURM_TIME=02:00:00
  SLURM_MEM=4G

Python selection is fail-fast. The launcher first honors PYTHON_BIN, then the
currently active Conda environment, then the current python on PATH, and then
tries the writingring-gpu and writingring-viz Conda environments. A candidate
is accepted only when it can import and start the AngularAccel66 helper.

The source root must already contain a completed 64 Hz 36-channel
PolaritySplitAbs pipeline, including recording-level spikeEncoding,
segmentation, and segmentation_padded artifacts. Alignment and segmentation
boundaries are reused; they are not recomputed.
USAGE
}

[[ -n "$SOURCE_ROOT" ]] || {
    usage
    exit 2
}

case "$MODE" in
    local|submit|worker|finalize) ;;
    *) fail "MODE must be local, submit, worker, or finalize" ;;
esac
case "$OVERWRITE_DEST" in
    0|1) ;;
    *) fail "OVERWRITE_DEST must be 0 or 1" ;;
esac
[[ "$SLURM_MAX_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] ||
    fail "SLURM_MAX_CONCURRENCY must be a positive integer"
(( SLURM_MAX_CONCURRENCY <= 50 )) ||
    fail "SLURM_MAX_CONCURRENCY may not exceed the repository default cap of 50"

cd "$REPO_ROOT"
[[ -f "$PYTHON_HELPER" ]] || fail "missing Python helper: $PYTHON_HELPER"

if [[ "$SOURCE_ROOT" != /* ]]; then
    SOURCE_ROOT="$REPO_ROOT/$SOURCE_ROOT"
fi
SOURCE_ROOT="${SOURCE_ROOT%/}"
[[ -d "$SOURCE_ROOT" ]] || fail "source root is not a directory: $SOURCE_ROOT"

if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT="${SOURCE_ROOT}_angular_accel66"
elif [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"
fi
OUTPUT_ROOT="${OUTPUT_ROOT%/}"
[[ "$OUTPUT_ROOT" != "$SOURCE_ROOT" ]] || fail "output root must differ from source root"

python_candidate_works() {
    local candidate="$1"
    [[ -n "$candidate" && -x "$candidate" ]] || return 1
    "$candidate" "$PYTHON_HELPER" --help >/dev/null 2>&1
}

select_python() {
    local candidate=""
    local env_name=""

    if [[ -n "$PYTHON_BIN" ]]; then
        if [[ "$PYTHON_BIN" == */* ]]; then
            candidate="$PYTHON_BIN"
        else
            candidate="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
        fi
        python_candidate_works "$candidate" ||
            fail "PYTHON_BIN does not provide the required WritingRing dependencies: $PYTHON_BIN"
        PYTHON_BIN="$candidate"
        return
    fi

    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        candidate="${CONDA_PREFIX}/bin/python"
        if python_candidate_works "$candidate"; then
            PYTHON_BIN="$candidate"
            return
        fi
    fi

    candidate="$(command -v python 2>/dev/null || true)"
    if python_candidate_works "$candidate"; then
        PYTHON_BIN="$candidate"
        return
    fi

    if command -v conda >/dev/null 2>&1; then
        for env_name in writingring-gpu writingring-viz; do
            candidate="$(
                conda run --no-capture-output -n "$env_name" \
                    python -c 'import sys; print(sys.executable)' 2>/dev/null \
                    | tr -d '\r' \
                    | tail -n 1
            )"
            if python_candidate_works "$candidate"; then
                PYTHON_BIN="$candidate"
                return
            fi
        done
    fi

    fail "could not find a Python interpreter with the AngularAccel66 dependencies. Activate writingring-gpu/writingring-viz, or set PYTHON_BIN=/path/to/env/bin/python"
}

select_python
export PYTHON_BIN
PYTHON_CMD=("$PYTHON_BIN")
printf '[angular66] python=%s\n' "$PYTHON_BIN"

read -r -a FREQUENCY_ARGS <<<"$FREQUENCIES_HZ"
[[ "${#FREQUENCY_ARGS[@]}" -eq 5 ]] ||
    fail "FREQUENCIES_HZ must contain exactly five whitespace-separated values"

common_args=(
    --source-root "$SOURCE_ROOT"
    --output-root "$OUTPUT_ROOT"
    --sampling-rate-hz "$SAMPLING_RATE_HZ"
    --frequencies-hz "${FREQUENCY_ARGS[@]}"
    --max-filter-time-s "$MAX_FILTER_TIME_S"
)
overwrite_args=()
if [[ "$OVERWRITE_DEST" == "1" ]]; then
    overwrite_args+=(--overwrite)
fi

list_users() {
    "${PYTHON_CMD[@]}" "$PYTHON_HELPER" list-users --source-root "$SOURCE_ROOT"
}

run_worker() {
    local user="$1"
    "${PYTHON_CMD[@]}" "$PYTHON_HELPER" build-user \
        "${common_args[@]}" \
        --user "$user" \
        "${overwrite_args[@]}"
}

run_finalizer() {
    "${PYTHON_CMD[@]}" "$PYTHON_HELPER" finalize \
        "${common_args[@]}" \
        "${overwrite_args[@]}"
}

USERS_OUTPUT=""
if ! USERS_OUTPUT="$(list_users)"; then
    fail "user discovery failed with Python interpreter $PYTHON_BIN; see the Python traceback above"
fi
[[ -n "$USERS_OUTPUT" ]] || fail "no users discovered in source root: $SOURCE_ROOT"
mapfile -t USERS <<<"$USERS_OUTPUT"

case "$MODE" in
    worker)
        if [[ -z "$USER_NAME" ]]; then
            [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] ||
                fail "worker mode requires USER_NAME or SLURM_ARRAY_TASK_ID"
            [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] ||
                fail "SLURM_ARRAY_TASK_ID must be a non-negative integer"
            (( SLURM_ARRAY_TASK_ID < ${#USERS[@]} )) ||
                fail "SLURM_ARRAY_TASK_ID is outside the discovered user list"
            USER_NAME="${USERS[$SLURM_ARRAY_TASK_ID]}"
        fi
        printf '[angular66] worker user=%s source=%s output=%s\n' \
            "$USER_NAME" "$SOURCE_ROOT" "$OUTPUT_ROOT"
        run_worker "$USER_NAME"
        ;;

    finalize)
        printf '[angular66] finalizing %d users into %s\n' "${#USERS[@]}" "$OUTPUT_ROOT"
        run_finalizer
        ;;

    local)
        if [[ -z "$JOBS" ]]; then
            if command -v nproc >/dev/null 2>&1; then
                JOBS="$(nproc)"
            else
                JOBS=1
            fi
            (( JOBS > 50 )) && JOBS=50
        fi
        [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "JOBS must be a positive integer"
        (( JOBS <= 50 )) || fail "JOBS may not exceed 50"
        (( JOBS > ${#USERS[@]} )) && JOBS="${#USERS[@]}"

        printf '[angular66] local mode: users=%d concurrent=%d\n' "${#USERS[@]}" "$JOBS"
        mkdir -p "$OUTPUT_ROOT/logs"
        pids=()
        running=0
        status=0
        for user in "${USERS[@]}"; do
            (
                printf '[angular66] start %s\n' "$user"
                run_worker "$user"
            ) >"$OUTPUT_ROOT/logs/${user}.log" 2>&1 &
            pids+=("$!")
            ((running += 1))
            if (( running >= JOBS )); then
                if ! wait -n; then
                    status=1
                fi
                ((running -= 1))
            fi
        done
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                if ! wait "$pid"; then
                    status=1
                fi
            fi
        done
        (( status == 0 )) || fail "one or more user workers failed; inspect $OUTPUT_ROOT/logs"
        run_finalizer
        ;;

    submit)
        command -v sbatch >/dev/null 2>&1 || fail "sbatch is required for MODE=submit"
        mkdir -p "$OUTPUT_ROOT/logs"
        array_last=$((${#USERS[@]} - 1))
        export_spec="ALL,PYTHON_BIN=${PYTHON_BIN},MODE=worker,OVERWRITE_DEST=${OVERWRITE_DEST},SAMPLING_RATE_HZ=${SAMPLING_RATE_HZ},FREQUENCIES_HZ=${FREQUENCIES_HZ},MAX_FILTER_TIME_S=${MAX_FILTER_TIME_S}"
        worker_job="$({
            sbatch --parsable \
                --job-name=wr-angular66 \
                --array="0-${array_last}%${SLURM_MAX_CONCURRENCY}" \
                --cpus-per-task=1 \
                --mem="$SLURM_MEM" \
                --time="$SLURM_TIME" \
                --output="$OUTPUT_ROOT/logs/slurm-angular66-%A_%a.out" \
                --error="$OUTPUT_ROOT/logs/slurm-angular66-%A_%a.err" \
                --export="$export_spec" \
                "$SCRIPT_PATH" "$SOURCE_ROOT" "$OUTPUT_ROOT"
        } | tail -n 1)"
        worker_job="${worker_job%%;*}"
        [[ "$worker_job" =~ ^[0-9]+$ ]] || fail "could not parse worker Slurm job id"

        finalize_export="ALL,PYTHON_BIN=${PYTHON_BIN},MODE=finalize,OVERWRITE_DEST=${OVERWRITE_DEST},SAMPLING_RATE_HZ=${SAMPLING_RATE_HZ},FREQUENCIES_HZ=${FREQUENCIES_HZ},MAX_FILTER_TIME_S=${MAX_FILTER_TIME_S}"
        final_job="$({
            sbatch --parsable \
                --job-name=wr-angular66-final \
                --dependency="afterok:${worker_job}" \
                --cpus-per-task=1 \
                --mem="$SLURM_MEM" \
                --time="$SLURM_TIME" \
                --output="$OUTPUT_ROOT/logs/slurm-angular66-final-%j.out" \
                --error="$OUTPUT_ROOT/logs/slurm-angular66-final-%j.err" \
                --export="$finalize_export" \
                "$SCRIPT_PATH" "$SOURCE_ROOT" "$OUTPUT_ROOT"
        } | tail -n 1)"
        final_job="${final_job%%;*}"
        [[ "$final_job" =~ ^[0-9]+$ ]] || fail "could not parse finalizer Slurm job id"

        printf '[angular66] submitted worker array job %s (%d users, max %s concurrent)\n' \
            "$worker_job" "${#USERS[@]}" "$SLURM_MAX_CONCURRENCY"
        printf '[angular66] submitted afterok finalizer job %s\n' "$final_job"
        ;;
esac
