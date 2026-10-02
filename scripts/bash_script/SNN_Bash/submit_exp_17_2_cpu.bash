#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
cd "$REPO_ROOT"
jid_prepare=$(sbatch --parsable scripts/bash_script/SNN_Bash/prepare_exp_17_2_cpu.bash)
jid_replay=$(sbatch --parsable --dependency=afterok:${jid_prepare} scripts/bash_script/SNN_Bash/run_exp_17_2_replay_cpu_array.bash)
jid_horizon=$(sbatch --parsable --dependency=afterok:${jid_prepare} scripts/bash_script/SNN_Bash/run_exp_17_2_horizon_cpu_array.bash)
jid_final=$(sbatch --parsable --dependency=afterok:${jid_replay}:${jid_horizon} scripts/bash_script/SNN_Bash/finalize_exp_17_2_cpu.bash)
printf 'prepare=%s\nreplay=%s\nhorizon=%s\nfinalize=%s\n' "$jid_prepare" "$jid_replay" "$jid_horizon" "$jid_final"
