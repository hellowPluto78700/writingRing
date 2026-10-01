#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
cd "$REPO_ROOT"
jid_prepare=$(sbatch --parsable scripts/bash_script/SNN_Bash/prepare_exp_17_cpu.bash)
jid_traj=$(sbatch --parsable --dependency=afterok:${jid_prepare} scripts/bash_script/SNN_Bash/run_exp_17_trajectory_cpu_array.bash)
jid_subset=$(sbatch --parsable --dependency=afterok:${jid_prepare} scripts/bash_script/SNN_Bash/run_exp_17_subset_cpu_array.bash)
jid_prune=$(sbatch --parsable --dependency=afterok:${jid_prepare} scripts/bash_script/SNN_Bash/run_exp_17_pruning_cpu_array.bash)
jid_final=$(sbatch --parsable --dependency=afterok:${jid_traj}:${jid_subset}:${jid_prune} scripts/bash_script/SNN_Bash/finalize_exp_17_cpu.bash)
printf 'prepare=%s\ntrajectory=%s\nsubset=%s\npruning=%s\nfinalize=%s\n' "$jid_prepare" "$jid_traj" "$jid_subset" "$jid_prune" "$jid_final"