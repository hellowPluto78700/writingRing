#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

extract_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/extract_exp_7_3_5_cpu_array.bash)
probe_job=$(sbatch --parsable --dependency="afterok:${extract_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_5_probe_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${probe_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_7_3_5_cpu.bash)

echo "Exp7.3.5 extraction array: ${extract_job}"
echo "Exp7.3.5 probe array: ${probe_job}"
echo "Exp7.3.5 finalizer: ${finalizer_job}"
