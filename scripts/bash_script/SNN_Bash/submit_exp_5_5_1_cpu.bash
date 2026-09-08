#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is not available on PATH. Run this launcher on a Slurm login node." >&2
  exit 127
fi

SOURCE_JOB=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/validate_exp_5_5_1_source_cpu_array.bash)
SCREEN_JOB=$(sbatch --parsable --dependency="afterok:${SOURCE_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_5_1_screen_cpu_array.bash)
SELECT_JOB=$(sbatch --parsable --dependency="afterok:${SCREEN_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/select_exp_5_5_1_screen_cpu.bash)
RESET_JOB=$(sbatch --parsable --dependency="afterok:${SELECT_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_5_1_reset_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --dependency="afterok:${RESET_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_5_1_final_cpu_array.bash)
FINALIZER_JOB=$(sbatch --parsable --dependency="afterok:${FINAL_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_5_5_1_cpu.bash)

echo "Exp5.5.1 source validation: ${SOURCE_JOB} (5 tasks)"
echo "Exp5.5.1 ordered regularization screen: ${SCREEN_JOB} (15 tasks, max 15 concurrent)"
echo "Exp5.5.1 validation-only selector: ${SELECT_JOB}"
echo "Exp5.5.1 matched reset: ${RESET_JOB} (5 tasks)"
echo "Exp5.5.1 final test/diagnostics: ${FINAL_JOB} (5 tasks)"
echo "Exp5.5.1 artifact-only finalizer: ${FINALIZER_JOB}"
