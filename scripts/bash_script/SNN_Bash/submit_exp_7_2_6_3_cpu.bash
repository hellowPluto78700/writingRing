#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT
cache_job="$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_7_2_6_3_cache_cpu_array.bash)"
mech_job="$(sbatch --parsable --export=ALL --dependency="afterok:${cache_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_3_mechanism_cpu_array.bash)"
head_job="$(sbatch --parsable --export=ALL --dependency="afterok:${cache_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_3_head_cpu_array.bash)"
cross_job="$(sbatch --parsable --export=ALL --dependency="afterok:${head_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_3_crosscheck_cpu_array.bash)"
final_job="$(sbatch --parsable --export=ALL --dependency="afterok:${mech_job}:${cross_job}" scripts/bash_script/SNN_Bash/finalize_exp_7_2_6_3_cpu.bash)"
printf 'Exp7.2.6.3 cache: %s\nmechanism: %s\nheads: %s\ncrosscheck: %s\nfinalizer: %s\n' "$cache_job" "$mech_job" "$head_job" "$cross_job" "$final_job"
