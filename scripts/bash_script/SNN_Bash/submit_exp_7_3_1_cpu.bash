#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

adam_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_1_adam_cpu_array.bash)
c2_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_1_lbfgs_noreg_cpu_array.bash)
c3_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_1_c3_cpu_array.bash)
c4_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_1_c4_cpu_array.bash)
ref_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_7_3_1_ref_cpu_array.bash)
finalizer_job=$(sbatch --parsable \
  --dependency="afterok:${adam_job}:${c2_job}:${c3_job}:${c4_job}:${ref_job}" \
  --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_7_3_1_cpu.bash)

echo "Exp7.3.1 Adam shared C0/C1 array: ${adam_job}"
echo "Exp7.3.1 C2 LBFGS array: ${c2_job}"
echo "Exp7.3.1 C3 candidate array: ${c3_job}"
echo "Exp7.3.1 C4 candidate array: ${c4_job}"
echo "Exp7.3.1 Ref candidate array: ${ref_job}"
echo "Exp7.3.1 finalizer: ${finalizer_job}"
