#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is not available on PATH. Run this launcher on a Slurm login node." >&2
  exit 127
fi

PREP_JOB=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/prepare_exp_5_4_3_reference_cpu_array.bash)
CAPACITY_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_4_3_capacity_cpu_array.bash)
FINALIZER_JOB=$(sbatch --parsable --dependency="afterok:${CAPACITY_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_5_4_3_capacity_cpu.bash)

echo "Exp5.4.3 paired reference prep: ${PREP_JOB} (5 tasks)"
echo "Exp5.4.3 Stage-A capacity map: ${CAPACITY_JOB} (45 tasks, max 45 concurrent)"
echo "Exp5.4.3 validation-only Stage-A finalizer: ${FINALIZER_JOB}"
echo "Inspect capacity_selection.json, then run submit_exp_5_4_3_extension_cpu.bash."
