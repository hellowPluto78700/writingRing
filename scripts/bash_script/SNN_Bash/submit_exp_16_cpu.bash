#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
cd "$REPO_ROOT"

jid_prepare=$(sbatch --parsable scripts/bash_script/SNN_Bash/prepare_exp_16_cpu.bash)
jid_p0a=$(sbatch --parsable --dependency=afterok:${jid_prepare} scripts/bash_script/SNN_Bash/run_exp_16_phase0a_cpu_array.bash)
jid_p0a_fin=$(sbatch --parsable --dependency=afterok:${jid_p0a} scripts/bash_script/SNN_Bash/finalize_exp_16_phase0a_cpu.bash)
jid_p0b=$(sbatch --parsable --dependency=afterok:${jid_p0a_fin} scripts/bash_script/SNN_Bash/run_exp_16_phase0b_cpu_array.bash)
jid_p0b_sel=$(sbatch --parsable --dependency=afterok:${jid_p0b} scripts/bash_script/SNN_Bash/select_exp_16_phase0b_cpu.bash)
jid_p1=$(sbatch --parsable --dependency=afterok:${jid_p0b_sel} scripts/bash_script/SNN_Bash/run_exp_16_phase1_cpu_array.bash)
jid_p1_fin=$(sbatch --parsable --dependency=afterok:${jid_p1} scripts/bash_script/SNN_Bash/finalize_exp_16_phase1_cpu.bash)
jid_p15=$(sbatch --parsable --dependency=afterok:${jid_p1_fin} scripts/bash_script/SNN_Bash/run_exp_16_phase1_5_cpu_array.bash)
jid_ablate=$(sbatch --parsable --dependency=afterok:${jid_p15} scripts/bash_script/SNN_Bash/run_exp_16_functional_ablation_cpu_array.bash)
jid_final=$(sbatch --parsable --dependency=afterok:${jid_ablate} scripts/bash_script/SNN_Bash/finalize_exp_16_cpu.bash)

printf 'prepare=%s\nphase0a=%s\nphase0a_finalize=%s\nphase0b=%s\nphase0b_select=%s\nphase1=%s\nphase1_finalize=%s\nphase1_5=%s\nfunctional_ablation=%s\nfinalize=%s\n' \
  "$jid_prepare" "$jid_p0a" "$jid_p0a_fin" "$jid_p0b" "$jid_p0b_sel" "$jid_p1" "$jid_p1_fin" "$jid_p15" "$jid_ablate" "$jid_final"
