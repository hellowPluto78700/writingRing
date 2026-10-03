#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../../../.." && pwd)}"
export CORE_SPIKE_DRAIN_RESULTS="${CORE_SPIKE_DRAIN_RESULTS:-core_benchmark_v1/results/output_spike_drain_v1}"
prepare=$(sbatch --parsable --export=ALL "$HERE/prepare.bash")
train=$(sbatch --parsable --dependency="afterok:$prepare" --export=ALL "$HERE/run_array.bash")
postprocess=$(sbatch --parsable --dependency="afterok:$train" --export=ALL "$HERE/postprocess_array.bash")
final=$(sbatch --parsable --dependency="afterok:$postprocess" --export=ALL "$HERE/finalize.bash")
printf 'prepare=%s train=%s postprocess=%s finalize=%s\n' "$prepare" "$train" "$postprocess" "$final"
