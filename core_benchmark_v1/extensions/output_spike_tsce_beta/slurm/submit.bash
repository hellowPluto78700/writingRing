#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../../../.." && pwd)}"
export CORE_SPIKE_TSCE_BETA_RESULTS="${CORE_SPIKE_TSCE_BETA_RESULTS:-core_benchmark_v1/results/output_spike_tsce_beta_v1}"
prepare=$(sbatch --parsable --export=ALL "$HERE/prepare.bash")
smoke=$(sbatch --parsable --dependency="afterok:$prepare" --export=ALL "$HERE/smoke.bash")
train=$(sbatch --parsable --dependency="afterok:$smoke" --export=ALL "$HERE/run_array.bash")
final=$(sbatch --parsable --dependency="afterok:$train" --export=ALL "$HERE/finalize.bash")
printf 'prepare=%s smoke=%s train=%s finalize=%s\n' "$prepare" "$smoke" "$train" "$final"
