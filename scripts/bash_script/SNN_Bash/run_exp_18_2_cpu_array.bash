#!/usr/bin/env bash
#SBATCH --job-name=exp18_2_run
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source scripts/bash_script/SNN_Bash/slurm_cpu_env.bash

seeds=(11 23 37)
cases=(I_NWCCE I_MWCCE U_NWCCE U_MWCCE)
seed_idx=$((SLURM_ARRAY_TASK_ID / 4))
case_idx=$((SLURM_ARRAY_TASK_ID % 4))

python -m scripts.experiment_18_2_loss_geometry run \
  --seed "${seeds[$seed_idx]}" \
  --case "${cases[$case_idx]}"
