#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

COMMON='module load conda/latest; eval "$(conda shell.bash hook)"; if conda env list | awk '\''{print $1}'\'' | grep -qx writingring-gpu; then conda activate writingring-gpu; else conda activate writingring-viz; fi; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1; cd '"$REPO_ROOT"'; '

PREP_JOB=$(sbatch --parsable \
  --job-name=exp5_4_2_src --array=0-4%5 --cpus-per-task=1 --mem=16G --time=24:00:00 \
  --output=exp5_4_2_src_%A_%a.out --error=exp5_4_2_src_%A_%a.err \
  --wrap="${COMMON} python -u -m scripts.experiment_5_4_2_phase_conditioned_readout prepare-source --array-task-id \${SLURM_ARRAY_TASK_ID} --device cpu --threads 1")

SCREEN_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" \
  --job-name=exp5_4_2_scr --array=0-49%50 --cpus-per-task=1 --mem=12G --time=24:00:00 \
  --output=exp5_4_2_scr_%A_%a.out --error=exp5_4_2_scr_%A_%a.err \
  --wrap="${COMMON} python -u -m scripts.experiment_5_4_2_phase_conditioned_readout screen-run-one --array-task-id \${SLURM_ARRAY_TASK_ID} --device cpu --threads 1")

FINAL_JOB=$(sbatch --parsable --dependency="afterok:${SCREEN_JOB}" \
  --job-name=exp5_4_2_sfin --cpus-per-task=1 --mem=4G --time=01:00:00 \
  --output=exp5_4_2_sfin_%j.out --error=exp5_4_2_sfin_%j.err \
  --wrap="${COMMON} python -u -m scripts.experiment_5_4_2_phase_conditioned_readout screen-finalize")

echo "Exp5.4.2 source/cache array: ${PREP_JOB} (5 tasks)"
echo "Exp5.4.2 mechanism screen array: ${SCREEN_JOB} (50 tasks, max 50 concurrent)"
echo "Exp5.4.2 validation-only screen finalizer: ${FINAL_JOB}"
echo "Inspect screen_selection.json before submitting Stage B."
