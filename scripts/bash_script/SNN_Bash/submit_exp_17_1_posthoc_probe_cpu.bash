#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
cd "$REPO_ROOT"
jid_probe=$(sbatch --parsable scripts/bash_script/SNN_Bash/run_exp_17_1_posthoc_probe_cpu_array.bash)
jid_final=$(sbatch --parsable --dependency=afterok:${jid_probe} scripts/bash_script/SNN_Bash/finalize_exp_17_1_posthoc_probe_cpu.bash)
printf 'probe=%s\nfinalize=%s\n' "$jid_probe" "$jid_final"
