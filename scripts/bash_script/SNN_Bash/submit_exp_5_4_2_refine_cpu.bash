#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
SELECTION="notebooks/artifacts/experiment_5_4_2_phase_conditioned_readout/phase_conditioned_readout_v1/screen_selection.json"
if [[ ! -f "$SELECTION" ]]; then
  echo "Missing $SELECTION. Run the screen workflow first." >&2
  exit 2
fi

read -r CONDITION MECHANISM <<<"$(python3 - "$SELECTION" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding='utf-8'))
if p.get('selected_condition') is None:
    raise SystemExit('screen found no supported candidate; refinement intentionally blocked')
print(p['selected_condition'], p['selected_mechanism'])
PY
)"

if [[ "$MECHANISM" == "bank" ]]; then
  REFINE_COUNT=45
elif [[ "$MECHANISM" == "bilinear" ]]; then
  REFINE_COUNT=15
else
  echo "Invalid selected mechanism: $MECHANISM" >&2
  exit 3
fi
REFINE_MAX=$((REFINE_COUNT - 1))

COMMON='module load conda/latest; eval "$(conda shell.bash hook)"; if conda env list | awk '\''{print $1}'\'' | grep -qx writingring-gpu; then conda activate writingring-gpu; else conda activate writingring-viz; fi; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1; cd '"$REPO_ROOT"'; '

REFINE_JOB=$(sbatch --parsable \
  --job-name=exp5_4_2_ref --array="0-${REFINE_MAX}%50" --cpus-per-task=1 --mem=12G --time=24:00:00 \
  --output=exp5_4_2_ref_%A_%a.out --error=exp5_4_2_ref_%A_%a.err \
  --wrap="${COMMON} python -u -m scripts.experiment_5_4_2_phase_conditioned_readout refine-run-one --array-task-id \${SLURM_ARRAY_TASK_ID} --device cpu --threads 1")

SELECT_JOB=$(sbatch --parsable --dependency="afterok:${REFINE_JOB}" \
  --job-name=exp5_4_2_rfin --cpus-per-task=1 --mem=4G --time=01:00:00 \
  --output=exp5_4_2_rfin_%j.out --error=exp5_4_2_rfin_%j.err \
  --wrap="${COMMON} python -u -m scripts.experiment_5_4_2_phase_conditioned_readout refine-finalize")

TEST_JOB=$(sbatch --parsable --dependency="afterok:${SELECT_JOB}" \
  --job-name=exp5_4_2_test --array=0-4%5 --cpus-per-task=1 --mem=12G --time=12:00:00 \
  --output=exp5_4_2_test_%A_%a.out --error=exp5_4_2_test_%A_%a.err \
  --wrap="${COMMON} python -u -m scripts.experiment_5_4_2_phase_conditioned_readout final-run-one --array-task-id \${SLURM_ARRAY_TASK_ID} --device cpu --threads 1")

FINAL_JOB=$(sbatch --parsable --dependency="afterok:${TEST_JOB}" \
  --job-name=exp5_4_2_fin --cpus-per-task=1 --mem=4G --time=01:00:00 \
  --output=exp5_4_2_fin_%j.out --error=exp5_4_2_fin_%j.err \
  --wrap="${COMMON} python -u -m scripts.experiment_5_4_2_phase_conditioned_readout finalize")

echo "Selected: ${CONDITION} (${MECHANISM})"
echo "Refine array: ${REFINE_JOB} (${REFINE_COUNT} tasks, max 50 concurrent)"
echo "Validation-only selector: ${SELECT_JOB}"
echo "Locked final-test array: ${TEST_JOB} (5 tasks)"
echo "Aggregation-only finalizer: ${FINAL_JOB}"
