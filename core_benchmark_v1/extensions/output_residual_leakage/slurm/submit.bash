#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../../../.." && pwd)}"
export CORE_RESIDUAL_RESULTS="${CORE_RESIDUAL_RESULTS:-core_benchmark_v1/results/output_residual_leakage_v1}"
prepare=$(sbatch --parsable --export=ALL "$HERE/prepare.bash")
train=$(sbatch --parsable --dependency="afterok:$prepare" --export=ALL "$HERE/run_array.bash")
analyze=$(sbatch --parsable --dependency="afterok:$train" --export=ALL "$HERE/analyze_array.bash")
prune=$(sbatch --parsable --dependency="afterok:$train" --export=ALL "$HERE/prune_array.bash")
final=$(sbatch --parsable --dependency="afterok:$analyze:$prune" --export=ALL "$HERE/finalize.bash")
printf 'prepare=%s train=%s analyze=%s prune=%s finalize=%s\n' "$prepare" "$train" "$analyze" "$prune" "$final"