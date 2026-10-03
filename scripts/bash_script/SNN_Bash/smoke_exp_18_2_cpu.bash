#!/usr/bin/env bash
#SBATCH --job-name=exp18_2_smoke
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source scripts/bash_script/SNN_Bash/slurm_cpu_env.bash
python - <<'PY'
import sys
import numpy
import torch
print({"python": sys.version.split()[0], "numpy": numpy.__version__, "torch": torch.__version__})
PY
python -m scripts.experiment_18_2_loss_geometry smoke
