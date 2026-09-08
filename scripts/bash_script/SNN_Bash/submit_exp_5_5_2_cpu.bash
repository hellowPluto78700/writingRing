#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is not available on PATH. Run this launcher on a Slurm login node." >&2
  exit 127
fi

SOURCE_JOB=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/validate_exp_5_5_2_source_cpu_array.bash)
SCREEN_JOB=$(sbatch --parsable --dependency="afterok:${SOURCE_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_5_2_clock_screen_cpu_array.bash)
SELECT_JOB=$(sbatch --parsable --dependency="afterok:${SCREEN_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/select_exp_5_5_2_clock_screen_cpu.bash)
CLOCK_FINAL_JOB=$(sbatch --parsable --dependency="afterok:${SELECT_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_5_2_clock_final_cpu_array.bash)
ORACLE_FINAL_JOB=$(sbatch --parsable --dependency="afterok:${SELECT_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_5_2_oracle_final_cpu_array.bash)
FINALIZER_JOB=$(sbatch --parsable --dependency="afterok:${CLOCK_FINAL_JOB}:${ORACLE_FINAL_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_5_5_2_cpu.bash)

echo "Exp5.5.2 source validation: ${SOURCE_JOB} (5 tasks)"
echo "Exp5.5.2 Clock alpha screen: ${SCREEN_JOB} (60 tasks, max 50 concurrent)"
echo "Exp5.5.2 validation-only selector: ${SELECT_JOB}"
echo "Exp5.5.2 Clock selected-checkpoint final evaluation: ${CLOCK_FINAL_JOB} (20 tasks)"
echo "Exp5.5.2 Oracle selected-alpha train/evaluate: ${ORACLE_FINAL_JOB} (20 tasks)"
echo "Exp5.5.2 artifact-only finalizer: ${FINALIZER_JOB}"
