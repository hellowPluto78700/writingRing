#!/usr/bin/env bash
set -euo pipefail

train_job=$(sbatch --parsable scripts/bash_script/SNN_Bash/run_exp_7_3_7_cpu_array.bash)
final_job=$(sbatch --parsable --dependency="afterok:${train_job}" scripts/bash_script/SNN_Bash/finalize_exp_7_3_7_cpu.bash)

echo "Exp7.3.7 training array: ${train_job}"
echo "Exp7.3.7 finalizer: ${final_job}"
