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

REFINE_JOB=$(sbatch --parsable --array="0-${REFINE_MAX}%50" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_4_2_refine_cpu_array.bash)
SELECT_JOB=$(sbatch --parsable --dependency="afterok:${REFINE_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_5_4_2_refine_cpu.bash)
TEST_JOB=$(sbatch --parsable --dependency="afterok:${SELECT_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_4_2_final_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --dependency="afterok:${TEST_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_5_4_2_cpu.bash)

echo "Selected: ${CONDITION} (${MECHANISM})"
echo "Refine array: ${REFINE_JOB} (${REFINE_COUNT} tasks, max 50 concurrent)"
echo "Validation-only selector: ${SELECT_JOB}"
echo "Locked final-test array: ${TEST_JOB} (5 tasks)"
echo "Aggregation-only finalizer: ${FINAL_JOB}"
