#!/usr/bin/env bash
#SBATCH --job-name=exp19_cal
#SBATCH --array=0-19%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
source scripts/bash_script/SNN_Bash/slurm_cpu_env.bash
python -m scripts.experiment_19_cross_user_generalization run-calibration-index --index "$SLURM_ARRAY_TASK_ID"
