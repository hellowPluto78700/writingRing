#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is not available on PATH. Run this launcher from a Slurm login node." >&2
  exit 127
fi

PREP_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_5_4_2_source_cpu_array.bash)
SCREEN_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_5_4_2_screen_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --dependency="afterok:${SCREEN_JOB}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_5_4_2_screen_cpu.bash)

echo "Exp5.4.2 source/cache array: ${PREP_JOB} (5 tasks)"
echo "Exp5.4.2 mechanism screen array: ${SCREEN_JOB} (50 tasks, max 50 concurrent)"
echo "Exp5.4.2 validation-only screen finalizer: ${FINAL_JOB}"
echo "Inspect screen_selection.json before submitting Stage B."
