#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
cd "$REPO_ROOT"
jid_prepare=$(sbatch --parsable scripts/bash_script/SNN_Bash/prepare_exp_16_1_cpu.bash)
jid_train=$(sbatch --parsable --dependency=afterok:${jid_prepare} scripts/bash_script/SNN_Bash/run_exp_16_1_train_cpu_array.bash)
jid_ablate=$(sbatch --parsable --dependency=afterok:${jid_train} scripts/bash_script/SNN_Bash/run_exp_16_1_ablation_cpu_array.bash)
jid_final=$(sbatch --parsable --dependency=afterok:${jid_ablate} scripts/bash_script/SNN_Bash/finalize_exp_16_1_cpu.bash)
printf 'prepare=%s\ntrain=%s\nablation=%s\nfinalize=%s\n' "$jid_prepare" "$jid_train" "$jid_ablate" "$jid_final"
