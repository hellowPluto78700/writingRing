#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is not available on PATH. Run this launcher on a Slurm login node." >&2
  exit 127
fi

TEACHER_JOB=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/prepare_exp_5_5_3_teachers_cpu_array.bash)
TRAIN_JOB=$(sbatch --parsable --dependency="afterok:${TEACHER_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_5_3_train_cpu_array.bash)
LINEAR_JOB=$(sbatch --parsable --dependency="afterok:${TEACHER_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_5_3_linear_cpu_array.bash)
FINALIZER_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}:${LINEAR_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_5_5_3_cpu.bash)

echo "Exp5.5.3 teacher preparation/frozen evaluation: ${TEACHER_JOB} (5 tasks)"
echo "Exp5.5.3 routed weight-bank train/evaluate: ${TRAIN_JOB} (30 tasks, max 30 concurrent)"
echo "Exp5.5.3 soft routed Linear refit: ${LINEAR_JOB} (10 tasks)"
echo "Exp5.5.3 artifact-only finalizer: ${FINALIZER_JOB}"
