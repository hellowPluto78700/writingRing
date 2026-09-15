#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

mean_job="$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_2_6_1_mean_ce_cpu_array.bash)"
probe_job="$(sbatch --parsable --export=ALL --dependency="afterok:${mean_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_1_probe_cpu_array.bash)"
final_job="$(sbatch --parsable --export=ALL --dependency="afterok:${probe_job}" scripts/bash_script/SNN_Bash/finalize_exp_7_2_6_1_cpu.bash)"

echo "Exp7.2.6.1 mean-CE training array: ${mean_job}"
echo "Exp7.2.6.1 probe array: ${probe_job}"
echo "Exp7.2.6.1 finalizer: ${final_job}"
