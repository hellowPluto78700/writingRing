#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is not available on PATH. Run this launcher on a Slurm login node." >&2
  exit 127
fi

SOURCE_JOB=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/validate_exp_5_5_4_source_cpu_array.bash)
WHEN_JOB=$(sbatch --parsable --dependency="afterok:${SOURCE_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_5_4_when_cpu_array.bash)
FINALIZER_JOB=$(sbatch --parsable --dependency="afterok:${WHEN_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_5_5_4_cpu.bash)

echo "Exp5.5.4 source/teacher validation: ${SOURCE_JOB} (5 tasks)"
echo "Exp5.5.4 GRU64/RSNN64 progress train->evaluate: ${WHEN_JOB} (10 tasks, max 10 concurrent)"
echo "Exp5.5.4 artifact-only finalizer: ${FINALIZER_JOB}"
