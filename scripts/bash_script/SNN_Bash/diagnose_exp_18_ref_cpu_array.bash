#!/usr/bin/env bash
#SBATCH --job-name=exp18_refdiag
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source scripts/bash_script/SNN_Bash/slurm_cpu_env.bash
seeds=(11 23 37)
python -m scripts.experiment_18_membrane_history diagnose --case I_REF --seed "${seeds[$SLURM_ARRAY_TASK_ID]}"
