#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

artifact_root="notebooks/artifacts/experiment_10_2_l1_membrane_memory/d1_bb_l1_mem_shift_sweep_v1"
stage_a_manifest="${artifact_root}/stage_a_manifest.json"

# Stage B consumes Stage-A baseline/replay artifacts. Refuse to submit any
# Stage-B jobs until the Stage-A finalizer has completed successfully; without
# this gate, the Stage-B finalizer can race the still-running replay array.
if [[ ! -f "$stage_a_manifest" ]]; then
    echo "Exp10.2 Stage B not submitted: missing $stage_a_manifest" >&2
    echo "Run/finish submit_exp_10_2_stage_a_cpu.bash first and wait for Stage-A PASS." >&2
    exit 2
fi
if ! grep -q '"experiment_id": "experiment_10_2_l1_membrane_memory"' "$stage_a_manifest" \
    || ! grep -q '"protocol_version": "d1_bb_l1_mem_shift_sweep_v1"' "$stage_a_manifest" \
    || ! grep -q '"expected_run_count": 12' "$stage_a_manifest" \
    || ! grep -q '"run_count": 12' "$stage_a_manifest" \
    || ! grep -q '"status": "PASS"' "$stage_a_manifest"; then
    echo "Exp10.2 Stage B not submitted: Stage-A manifest is not a matching PASS." >&2
    echo "Inspect $stage_a_manifest and the Stage-A finalizer log before retrying." >&2
    exit 2
fi

echo "Exp10.2 Stage-A preflight: PASS"

# Re-running prepare is cheap and keeps the saved split manifest synchronized.
prepare_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_10_2_cpu.bash)
long_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_10_2_long_e2e_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${long_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_10_2_cpu.bash)

echo "Exp10.2 Stage-B prepare: ${prepare_job}"
echo "Exp10.2 Stage-B long-memory E2E array: ${long_job}"
echo "Exp10.2 full finalizer: ${finalizer_job}"
