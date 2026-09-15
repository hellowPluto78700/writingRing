#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

bias_job="$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_2_6_2_bias_cpu_array.bash)"
probe_job="$(sbatch --parsable --export=ALL --dependency="afterok:${bias_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_2_probe_cpu_array.bash)"
final_job="$(sbatch --parsable --export=ALL --dependency="afterok:${probe_job}" scripts/bash_script/SNN_Bash/finalize_exp_7_2_6_2_cpu.bash)"

echo "Exp7.2.6.2 bias E2E array: ${bias_job}"
echo "Exp7.2.6.2 L2 probe array: ${probe_job}"
echo "Exp7.2.6.2 finalizer: ${final_job}"
