#!/usr/bin/env bash
set -Eeuo pipefail

# Build D0/D1/D2 writing-motion ablation datasets from a completed 64 Hz
# aligned-board-events source pipeline.
#
# D0: source reference               Encoder(a)
# D1: post-encode hard mask          m * Encoder(a)
# D2: masked accel + re-encode       m * Encoder(m * a)
#
# Parallelism follows the existing AngularAccel66 builder:
#   MODE=local    one process per user, capped by JOBS
#   MODE=submit   one Slurm array task per user + afterok finalizer
#   MODE=worker   internal single-user worker
#   MODE=finalize internal aggregate/final validation

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-}"
EXPLICIT_REPO_ROOT="${WRITINGRING_REPO_ROOT:-}"

resolve_repo_root() {
    local candidate="" root=""
    local -a candidates=()
    [[ -n "$EXPLICIT_REPO_ROOT" ]] && candidates+=("$EXPLICIT_REPO_ROOT")
    [[ -n "$SUBMIT_DIR" ]] && candidates+=("$SUBMIT_DIR")
    candidates+=("$PWD" "$SCRIPT_DIR")
    for candidate in "${candidates[@]}"; do
        [[ -n "$candidate" ]] || continue
        if root="$(git -C "$candidate" rev-parse --show-toplevel 2>/dev/null)"; then
            if [[ -f "$root/scripts/build_writing_motion_variants.py" ]]; then
                printf '%s\n' "$root"
                return 0
            fi
        fi
    done
    return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
    printf 'error: could not locate writingRing repository; set WRITINGRING_REPO_ROOT\n' >&2
    exit 2
fi
export WRITINGRING_REPO_ROOT="$REPO_ROOT"
SCRIPT_PATH="$REPO_ROOT/scripts/bash_script/preprocessing_pipeline/build_writing_motion_variants.bash"
PYTHON_HELPER="$REPO_ROOT/scripts/build_writing_motion_variants.py"

SOURCE_ROOT="${1:-}"
OUTPUT_ROOT="${2:-}"
MODE="${MODE:-local}"
JOBS="${JOBS:-}"
OVERWRITE_DEST="${OVERWRITE_DEST:-0}"
WRITE_MASK_VERIFICATION="${WRITE_MASK_VERIFICATION:-1}"
VERIFICATION_DPI="${VERIFICATION_DPI:-200}"
SLURM_MAX_CONCURRENCY="${SLURM_MAX_CONCURRENCY:-50}"
SLURM_PARTITION="${SLURM_PARTITION:-cpu}"
SLURM_TIME="${SLURM_TIME:-02:00:00}"
SLURM_MEM="${SLURM_MEM:-8G}"
USER_NAME="${USER_NAME:-}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

fail() { printf 'error: %s\n' "$1" >&2; exit 2; }

[[ -n "$SOURCE_ROOT" ]] || fail "usage: $0 SOURCE_COMBINATION_ROOT [OUTPUT_ROOT]"
case "$MODE" in local|submit|worker|finalize) ;; *) fail "MODE must be local, submit, worker, or finalize" ;; esac
case "$OVERWRITE_DEST" in 0|1) ;; *) fail "OVERWRITE_DEST must be 0 or 1" ;; esac
case "$WRITE_MASK_VERIFICATION" in 0|1) ;; *) fail "WRITE_MASK_VERIFICATION must be 0 or 1" ;; esac
[[ "$SLURM_MAX_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || fail "SLURM_MAX_CONCURRENCY must be positive"
(( SLURM_MAX_CONCURRENCY <= 50 )) || fail "SLURM_MAX_CONCURRENCY may not exceed 50"
[[ "$VERIFICATION_DPI" =~ ^[1-9][0-9]*$ ]] || fail "VERIFICATION_DPI must be positive"

cd "$REPO_ROOT"
if command -v module >/dev/null 2>&1; then module load conda/latest; fi
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate writingring-gpu
fi
python "$PYTHON_HELPER" --help >/dev/null

[[ "$SOURCE_ROOT" = /* ]] || SOURCE_ROOT="$REPO_ROOT/$SOURCE_ROOT"
SOURCE_ROOT="${SOURCE_ROOT%/}"
[[ -d "$SOURCE_ROOT" ]] || fail "source root is not a directory: $SOURCE_ROOT"
if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT="${SOURCE_ROOT}_writing_motion_ablation"
elif [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"
fi
OUTPUT_ROOT="${OUTPUT_ROOT%/}"
[[ "$OUTPUT_ROOT" != "$SOURCE_ROOT" ]] || fail "output root must differ from source root"
mkdir -p "$OUTPUT_ROOT/logs"

common_args=(--source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT")
overwrite_args=(); [[ "$OVERWRITE_DEST" == 1 ]] && overwrite_args+=(--overwrite)
verification_args=(--verification-dpi "$VERIFICATION_DPI")
[[ "$WRITE_MASK_VERIFICATION" == 0 ]] && verification_args+=(--no-verification)

mapfile -t USERS < <(python "$PYTHON_HELPER" list-users --source-root "$SOURCE_ROOT")
(("${#USERS[@]}" > 0)) || fail "no users discovered"

run_worker() {
    python "$PYTHON_HELPER" build-user "${common_args[@]}" --user "$1" "${overwrite_args[@]}" "${verification_args[@]}"
}
run_finalizer() {
    python "$PYTHON_HELPER" finalize "${common_args[@]}" "${overwrite_args[@]}"
}

case "$MODE" in
worker)
    if [[ -z "$USER_NAME" ]]; then
        [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || fail "worker requires USER_NAME or SLURM_ARRAY_TASK_ID"
        (( SLURM_ARRAY_TASK_ID < ${#USERS[@]} )) || fail "SLURM_ARRAY_TASK_ID out of range"
        USER_NAME="${USERS[$SLURM_ARRAY_TASK_ID]}"
    fi
    run_worker "$USER_NAME"
    ;;
finalize)
    run_finalizer
    ;;
local)
    if [[ -z "$JOBS" ]]; then
        JOBS="$(command -v nproc >/dev/null 2>&1 && nproc || printf '1')"
        (( JOBS > 50 )) && JOBS=50
    fi
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "JOBS must be positive"
    (( JOBS > ${#USERS[@]} )) && JOBS="${#USERS[@]}"
    printf '[writing-motion] users=%d concurrent=%d\n' "${#USERS[@]}" "$JOBS"
    running=0 status=0
    for user in "${USERS[@]}"; do
        (run_worker "$user") >"$OUTPUT_ROOT/logs/${user}.log" 2>&1 &
        ((running += 1))
        if (( running >= JOBS )); then
            if ! wait -n; then status=1; fi
            ((running -= 1))
        fi
    done
    while (( running > 0 )); do
        if ! wait -n; then status=1; fi
        ((running -= 1))
    done
    (( status == 0 )) || fail "one or more user workers failed; inspect $OUTPUT_ROOT/logs"
    run_finalizer
    ;;
submit)
    command -v sbatch >/dev/null 2>&1 || fail "sbatch is required for MODE=submit"
    array_last=$((${#USERS[@]} - 1))
    export_spec="ALL,WRITINGRING_REPO_ROOT=${REPO_ROOT},MODE=worker,OVERWRITE_DEST=${OVERWRITE_DEST},WRITE_MASK_VERIFICATION=${WRITE_MASK_VERIFICATION},VERIFICATION_DPI=${VERIFICATION_DPI}"
    worker_job="$({ sbatch --parsable \
        --job-name=wr-writing-motion --partition="$SLURM_PARTITION" \
        --array="0-${array_last}%${SLURM_MAX_CONCURRENCY}" --cpus-per-task=1 \
        --mem="$SLURM_MEM" --time="$SLURM_TIME" --chdir="$REPO_ROOT" \
        --output="$OUTPUT_ROOT/logs/slurm-writing-motion-%A_%a.out" \
        --error="$OUTPUT_ROOT/logs/slurm-writing-motion-%A_%a.err" \
        --export="$export_spec" "$SCRIPT_PATH" "$SOURCE_ROOT" "$OUTPUT_ROOT"; } | tail -n 1)"
    worker_job="${worker_job%%;*}"
    [[ "$worker_job" =~ ^[0-9]+$ ]] || fail "could not parse worker job id"
    final_export="ALL,WRITINGRING_REPO_ROOT=${REPO_ROOT},MODE=finalize,OVERWRITE_DEST=${OVERWRITE_DEST},WRITE_MASK_VERIFICATION=${WRITE_MASK_VERIFICATION},VERIFICATION_DPI=${VERIFICATION_DPI}"
    final_job="$({ sbatch --parsable \
        --job-name=wr-writing-motion-final --partition="$SLURM_PARTITION" \
        --dependency="afterok:${worker_job}" --cpus-per-task=1 --mem=4G \
        --time=01:00:00 --chdir="$REPO_ROOT" \
        --output="$OUTPUT_ROOT/logs/slurm-writing-motion-final-%j.out" \
        --error="$OUTPUT_ROOT/logs/slurm-writing-motion-final-%j.err" \
        --export="$final_export" "$SCRIPT_PATH" "$SOURCE_ROOT" "$OUTPUT_ROOT"; } | tail -n 1)"
    final_job="${final_job%%;*}"
    printf '[writing-motion] submitted array=%s finalizer=%s\n' "$worker_job" "$final_job"
    ;;
esac
