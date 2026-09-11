#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT

BASELINE_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_6_0_baseline_cpu.bash"
NORMALIZED_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_6_0_cpu_array.bash"
LEGACY_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_6_0_legacy_cpu_array.bash"
FINAL_SCRIPT="scripts/bash_script/SNN_Bash/finalize_exp_6_0_cpu.bash"

baseline_job="$(sbatch --parsable --export=ALL "$BASELINE_SCRIPT")"
normalized_job="$(sbatch --parsable --export=ALL "$NORMALIZED_SCRIPT")"
legacy_job="$(sbatch --parsable --export=ALL "$LEGACY_SCRIPT")"
final_job="$(sbatch --parsable --dependency="afterok:${baseline_job}:${normalized_job}:${legacy_job}" --export=ALL "$FINAL_SCRIPT")"

printf 'Exp6.0 Fixed250 baseline job: %s\n' "$baseline_job"
printf 'Exp6.0 normalized-synapse SNN array: %s\n' "$normalized_job"
printf 'Exp6.0 legacy-synapse SNN array: %s\n' "$legacy_job"
printf 'Exp6.0 paired comparison finalizer: %s\n' "$final_job"
