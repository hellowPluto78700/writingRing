#!/usr/bin/env bash
#SBATCH --job-name=exp7_2_6_prep
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=exp7_2_6_prep_%A_%a.out
#SBATCH --error=exp7_2_6_prep_%A_%a.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
module load conda/latest
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx writingring-gpu; then
    conda activate writingring-gpu
else
    conda activate writingring-viz
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

# Exp7.2.6 uses the Exp7.2.5 alpha=0, beta=.5, WholeCount E2E solution as
# the paired C2/frozen-L2 anchor. Bootstrap that exact anchor here so Exp7.2.6
# does not require a separately completed Exp7.2.5 sweep. Existing checkpoints
# and evaluations are reused by Exp7.2.5 run_e2e without retraining.
REGULARIZATIONS=(task_only task_plus_reg)
SEEDS=(11 23 37)
REG_INDEX=$((TASK_ID / 3))
SEED_INDEX=$((TASK_ID % 3))
REGULARIZATION="${REGULARIZATIONS[$REG_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

echo "Ensuring Exp7.2.5 C2 anchor: regularization=${REGULARIZATION} seed=${SEED}"
python -u -m scripts.experiment_7_2_5_output_synaptic_alpha \
  --device cpu --threads 1 \
  run-e2e \
  --architecture 234x234 \
  --regularization "$REGULARIZATION" \
  --seed "$SEED" \
  --objective whole_count_ce \
  --output-alpha 0.0

echo "Preparing Exp7.2.6 frozen L2 cache for array task ${TASK_ID}"
python -u -m scripts.experiment_7_2_6_output_readout_loss_shaping \
  --device cpu --threads 1 prepare-frozen --array-task-id "$TASK_ID"
