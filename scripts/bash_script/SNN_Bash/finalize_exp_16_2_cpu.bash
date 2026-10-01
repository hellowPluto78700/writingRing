#!/usr/bin/env bash
#SBATCH --job-name=exp16_2_final
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
set -euo pipefail
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate writingring-gpu || conda activate writingring-viz
cd "${REPO_ROOT:-$PWD}"
python -m scripts.experiment_16_2_matched_budget_selective_write finalize