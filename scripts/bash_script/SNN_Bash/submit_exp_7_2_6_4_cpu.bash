#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export REPO_ROOT

cache_job="$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_7_2_6_4_cache_cpu_array.bash)"
part_a_job="$(sbatch --parsable --export=ALL --dependency="afterok:${cache_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_4_part_a_cpu_array.bash)"
part_b_job="$(sbatch --parsable --export=ALL --dependency="afterok:${cache_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_4_part_b_cpu_array.bash)"
part_c_job="$(sbatch --parsable --export=ALL --dependency="afterok:${cache_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_4_part_c_cpu_array.bash)"
part_d_job="$(sbatch --parsable --export=ALL --dependency="afterok:${cache_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_4_part_d_cpu_array.bash)"
grad_job="$(sbatch --parsable --export=ALL --dependency="afterok:${cache_job}" scripts/bash_script/SNN_Bash/run_exp_7_2_6_4_gradient_cpu_array.bash)"
final_job="$(sbatch --parsable --export=ALL --dependency="afterok:${part_a_job}:${part_b_job}:${part_c_job}:${part_d_job}:${grad_job}" scripts/bash_script/SNN_Bash/finalize_exp_7_2_6_4_cpu.bash)"

printf 'Exp7.2.6.4 cache: %s\nPart A: %s\nPart B: %s\nPart C: %s\nPart D: %s\ngradient: %s\nfinalizer: %s\n' \
  "$cache_job" "$part_a_job" "$part_b_job" "$part_c_job" "$part_d_job" "$grad_job" "$final_job"
