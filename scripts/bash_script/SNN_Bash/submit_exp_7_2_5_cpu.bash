#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

prep_job="$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_7_2_5_frozen_l2_cpu_array.bash)"
frozen_job="$(sbatch --parsable --export=ALL --dependency="afterok:${prep_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_5_frozen_head_cpu_array.bash)"
e2e_job="$(sbatch --parsable --export=ALL --dependency="afterok:${frozen_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_5_e2e_cpu_array.bash)"
final_job="$(sbatch --parsable --export=ALL --dependency="afterok:${e2e_job}" scripts/bash_script/SNN_Bash/finalize_exp_7_2_5_cpu.bash)"

echo "Exp7.2.5 frozen-L2 cache array: ${prep_job}"
echo "Exp7.2.5 frozen-head array: ${frozen_job}"
echo "Exp7.2.5 paired E2E array: ${e2e_job}"
echo "Exp7.2.5 finalizer: ${final_job}"
