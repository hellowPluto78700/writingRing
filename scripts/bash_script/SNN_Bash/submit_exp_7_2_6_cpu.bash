#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

prep_job="$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_7_2_6_frozen_l2_cpu_array.bash)"
head_job="$(sbatch --parsable --export=ALL --dependency="afterok:${prep_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_frozen_head_cpu_array.bash)"
mechanism_job="$(sbatch --parsable --export=ALL --dependency="afterok:${head_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_mechanism_cpu_array.bash)"
e2e_job="$(sbatch --parsable --export=ALL --dependency="afterok:${mechanism_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_e2e_cpu_array.bash)"
probe_job="$(sbatch --parsable --export=ALL --dependency="afterok:${e2e_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_probe_cpu_array.bash)"
final_job="$(sbatch --parsable --export=ALL --dependency="afterok:${probe_job}" scripts/bash_script/SNN_Bash/finalize_exp_7_2_6_cpu.bash)"

echo "Exp7.2.6 frozen-L2 cache array: ${prep_job}"
echo "Exp7.2.6 frozen-head array: ${head_job}"
echo "Exp7.2.6 mechanism array: ${mechanism_job}"
echo "Exp7.2.6 E2E array: ${e2e_job}"
echo "Exp7.2.6 probe array: ${probe_job}"
echo "Exp7.2.6 finalizer: ${final_job}"
