#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "${REPO_ROOT}"
export REPO_ROOT

prepare_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_7_3_9_source_cpu.bash)
c1_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_9_c1_cpu_array.bash)
c2_job=$(sbatch --parsable --dependency="afterok:${c1_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_9_c2_cpu_array.bash)
probe_job=$(sbatch --parsable --dependency="afterok:${c2_job}" --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_9_probe_cpu_array.bash)
finalizer_job=$(sbatch --parsable --dependency="afterok:${probe_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_7_3_9_cpu.bash)

echo "Exp7.3.9 source validation: ${prepare_job}"
echo "Exp7.3.9 C1 array: ${c1_job}"
echo "Exp7.3.9 C2 array: ${c2_job}"
echo "Exp7.3.9 probe array: ${probe_job}"
echo "Exp7.3.9 finalizer: ${finalizer_job}"
