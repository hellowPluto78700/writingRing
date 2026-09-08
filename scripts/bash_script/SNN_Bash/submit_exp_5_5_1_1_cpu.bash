#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is not available on PATH. Run this launcher on a Slurm login node." >&2
  exit 127
fi

DIAG_JOB=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_5_1_1_diagnostic_cpu_array.bash)
FINALIZER_JOB=$(sbatch --parsable --dependency="afterok:${DIAG_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_5_5_1_1_cpu.bash)

echo "Exp5.5.1.1 routing diagnostics: ${DIAG_JOB} (5 tasks, max 5 concurrent)"
echo "Exp5.5.1.1 artifact-only finalizer: ${FINALIZER_JOB}"
