#!/usr/bin/env bash
#SBATCH --job-name=exp18_3_smoke
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source scripts/bash_script/SNN_Bash/slurm_cpu_env.bash
python -m scripts.experiment_18_3_discriminative_coordinate smoke
