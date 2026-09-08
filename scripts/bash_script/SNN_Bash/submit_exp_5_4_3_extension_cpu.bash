#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is not available on PATH. Run this launcher on a Slurm login node." >&2
  exit 127
fi

ROOT="notebooks/artifacts/experiment_5_4_3_elapsed_readout_capacity/elapsed_readout_capacity_v1"
CAPACITY_SELECTION="${ROOT}/capacity_selection.json"
SELECTION="${ROOT}/selection.json"

if [[ ! -f "$CAPACITY_SELECTION" ]]; then
  echo "Missing ${CAPACITY_SELECTION}. Run Stage A first." >&2
  exit 2
fi

if [[ -f "$SELECTION" ]]; then
  echo "Stage A already locked a supported selection; skipping extension."
  TEST_JOB=$(sbatch --parsable --export=ALL \
    scripts/bash_script/SNN_Bash/run_exp_5_4_3_final_cpu_array.bash)
  FINAL_JOB=$(sbatch --parsable --dependency="afterok:${TEST_JOB}" --export=ALL \
    scripts/bash_script/SNN_Bash/finalize_exp_5_4_3_cpu.bash)
  echo "Locked final-test array: ${TEST_JOB} (5 tasks)"
  echo "Aggregation-only finalizer: ${FINAL_JOB}"
  exit 0
fi

read -r STATUS MODE <<<"$(python3 - "$CAPACITY_SELECTION" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding='utf-8'))
print(p.get('status', ''), p.get('extension_mode', ''))
PY
)"

if [[ "$STATUS" != "extension_required" ]]; then
  echo "Stage-A selection is neither locked nor extension_required: status=${STATUS}" >&2
  exit 3
fi
if [[ "$MODE" != "higher_k" && "$MODE" != "direct_matrix" ]]; then
  echo "Invalid extension mode: ${MODE}" >&2
  exit 4
fi

EXT_JOB=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_4_3_extension_cpu_array.bash)
SELECT_JOB=$(sbatch --parsable --dependency="afterok:${EXT_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_5_4_3_extension_cpu.bash)
TEST_JOB=$(sbatch --parsable --dependency="afterok:${SELECT_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/run_exp_5_4_3_final_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --dependency="afterok:${TEST_JOB}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_5_4_3_cpu.bash)

echo "Extension mode: ${MODE}"
echo "Stage-B diagnostic array: ${EXT_JOB} (15 tasks, max 15 concurrent)"
echo "Validation-only extension selector: ${SELECT_JOB}"
echo "Locked final-test array: ${TEST_JOB} (5 tasks)"
echo "Aggregation-only finalizer: ${FINAL_JOB}"
