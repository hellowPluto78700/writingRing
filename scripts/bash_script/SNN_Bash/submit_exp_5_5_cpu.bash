#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is not available on PATH. Run this launcher on a Slurm login node." >&2
  exit 127
fi

PREP_JOB=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/prepare_exp_5_5_reference_cpu_array.bash)
RUN_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_5_cpu_array.bash)
FINALIZER_JOB=$(sbatch --parsable --dependency="afterok:${RUN_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_5_5_cpu.bash)

echo "Exp5.5 reference preparation: ${PREP_JOB} (5 tasks)"
echo "Exp5.5 mechanism runs: ${RUN_JOB} (25 tasks, max 25 concurrent)"
echo "Exp5.5 artifact-only finalizer: ${FINALIZER_JOB}"
