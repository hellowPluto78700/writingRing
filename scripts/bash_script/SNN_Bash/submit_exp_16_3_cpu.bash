#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
cd "$REPO_ROOT"
jid_prepare=$(sbatch --parsable scripts/bash_script/SNN_Bash/prepare_exp_16_3_cpu.bash)
jid_scale=$(sbatch --parsable --dependency=afterok:${jid_prepare} scripts/bash_script/SNN_Bash/run_exp_16_3_scale_cpu_array.bash)
jid_obj=$(sbatch --parsable --dependency=afterok:${jid_prepare} scripts/bash_script/SNN_Bash/run_exp_16_3_objective_cpu_array.bash)
jid_train=$(sbatch --parsable --dependency=afterok:${jid_prepare} scripts/bash_script/SNN_Bash/run_exp_16_3_train_cpu_array.bash)
jid_final=$(sbatch --parsable --dependency=afterok:${jid_scale}:${jid_obj}:${jid_train} scripts/bash_script/SNN_Bash/finalize_exp_16_3_cpu.bash)
printf 'prepare=%s\nscale=%s\nobjective=%s\ntrain=%s\nfinalize=%s\n' "$jid_prepare" "$jid_scale" "$jid_obj" "$jid_train" "$jid_final"
