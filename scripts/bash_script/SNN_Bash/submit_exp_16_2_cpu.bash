#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
cd "$REPO_ROOT"
jid_prepare=$(sbatch --parsable scripts/bash_script/SNN_Bash/prepare_exp_16_2_cpu.bash)
jid_p0=$(sbatch --parsable --dependency=afterok:${jid_prepare} scripts/bash_script/SNN_Bash/run_exp_16_2_phase0_cpu_array.bash)
jid_p0fin=$(sbatch --parsable --dependency=afterok:${jid_p0} scripts/bash_script/SNN_Bash/finalize_exp_16_2_phase0_cpu.bash)
jid_p1=$(sbatch --parsable --dependency=afterok:${jid_p0fin} scripts/bash_script/SNN_Bash/run_exp_16_2_phase1_pairs_cpu_array.bash)
jid_p2=$(sbatch --parsable --dependency=afterok:${jid_p1} scripts/bash_script/SNN_Bash/run_exp_16_2_phase2_replay_cpu_array.bash)
jid_final=$(sbatch --parsable --dependency=afterok:${jid_p2} scripts/bash_script/SNN_Bash/finalize_exp_16_2_cpu.bash)
printf 'prepare=%s\nphase0=%s\ncalibration=%s\nphase1=%s\nphase2=%s\nfinalize=%s\n' "$jid_prepare" "$jid_p0" "$jid_p0fin" "$jid_p1" "$jid_p2" "$jid_final"
