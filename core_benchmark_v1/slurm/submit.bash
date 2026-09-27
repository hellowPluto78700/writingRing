#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
export REPO_ROOT
CORE_CONFIG="${CORE_CONFIG:-$REPO_ROOT/core_benchmark_v1/00_protocol/default.json}"
CORE_RESULTS="${CORE_RESULTS:-$REPO_ROOT/core_benchmark_v1/results/main}"
SLURM_MAX_CONCURRENCY="${SLURM_MAX_CONCURRENCY:-50}"
DRY_RUN="${DRY_RUN:-0}"
if ! [[ "$SLURM_MAX_CONCURRENCY" =~ ^[0-9]+$ ]] || (( SLURM_MAX_CONCURRENCY < 1 || SLURM_MAX_CONCURRENCY > 50 )); then
  echo 'SLURM_MAX_CONCURRENCY must be an integer in [1, 50]' >&2
  exit 2
fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  echo 'DRY_RUN must be 0 or 1' >&2
  exit 2
fi
if [[ "$DRY_RUN" == 0 ]]; then
  source "$REPO_ROOT/core_benchmark_v1/slurm/common.bash"
fi
# plan imports only the Python standard library; dry-run does not require Conda.
read -r phase1 phase2 phase3 < <(python3 -m core_benchmark_v1 --config "$CORE_CONFIG" plan --counts)
export CORE_CONFIG CORE_RESULTS
if [[ "$DRY_RUN" == 0 ]]; then
  mkdir -p "$CORE_RESULTS/slurm_logs"
fi
submit() {
  local label="$1"; shift
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'sbatch ' >&2
    printf '%q ' "$@" >&2
    printf '\n' >&2
    printf 'DRY_%s\n' "$label"
  else
    local result
    result=$(sbatch --parsable "$@")
    printf '%s\n' "${result%%;*}"
  fi
}
# One array contains all independent factor-isolation runs. Later phases exist
# only for real checkpoint dependencies, not objective/tau winner selection.
prepare_id=$(submit prepare --export=ALL \
  --output="$CORE_RESULTS/slurm_logs/prepare_%j.out" \
  --error="$CORE_RESULTS/slurm_logs/prepare_%j.err" \
  "$REPO_ROOT/core_benchmark_v1/slurm/prepare.bash")
previous="$prepare_id"
for phase in 1 2 3; do
  case "$phase" in
    1) count="$phase1";;
    2) count="$phase2";;
    3) count="$phase3";;
  esac
  if (( count == 0 )); then continue; fi
  cap="$SLURM_MAX_CONCURRENCY"
  if (( count < cap )); then cap="$count"; fi
  export CORE_PHASE="$phase"
  job=$(submit "phase${phase}" --export=ALL --dependency="afterok:${previous}" \
    --array="0-$((count - 1))%${cap}" \
    --output="$CORE_RESULTS/slurm_logs/phase${phase}_%A_%a.out" \
    --error="$CORE_RESULTS/slurm_logs/phase${phase}_%A_%a.err" \
    "$REPO_ROOT/core_benchmark_v1/slurm/run_array.bash")
  echo "phase ${phase}: ${job} (${count} tasks, concurrency ${cap})"
  previous="$job"
done
finalizer=$(submit finalize --export=ALL --dependency="afterok:${previous}" \
  --output="$CORE_RESULTS/slurm_logs/finalize_%j.out" \
  --error="$CORE_RESULTS/slurm_logs/finalize_%j.err" \
  "$REPO_ROOT/core_benchmark_v1/slurm/finalize.bash")
echo "prepare: ${prepare_id}"
echo "finalizer: ${finalizer}"
echo "results: ${CORE_RESULTS}"
