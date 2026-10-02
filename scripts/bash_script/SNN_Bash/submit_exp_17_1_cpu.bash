#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
cd "$REPO_ROOT"
jid_cal=$(sbatch --parsable scripts/bash_script/SNN_Bash/calibrate_exp_17_1_cpu.bash)
jid_boot=$(sbatch --parsable --dependency=afterok:${jid_cal} scripts/bash_script/SNN_Bash/run_exp_17_1_bootstrap_cpu_array.bash)
jid_formal=$(sbatch --parsable --dependency=afterok:${jid_boot} scripts/bash_script/SNN_Bash/run_exp_17_1_formal_cpu_array.bash)
jid_final=$(sbatch --parsable --dependency=afterok:${jid_formal} scripts/bash_script/SNN_Bash/finalize_exp_17_1_cpu.bash)
printf 'calibrate=%s
bootstrap=%s
formal=%s
finalize=%s
' "$jid_cal" "$jid_boot" "$jid_formal" "$jid_final"
