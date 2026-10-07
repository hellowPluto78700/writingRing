#!/usr/bin/env bash
#SBATCH --job-name=exp19_prepare
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
source scripts/bash_script/SNN_Bash/slurm_cpu_env.bash
python -m scripts.experiment_19_cross_user_generalization prepare
