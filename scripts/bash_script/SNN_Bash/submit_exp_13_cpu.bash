#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
SLURM_MAX_CONCURRENCY="${SLURM_MAX_CONCURRENCY:-20}"
if (( SLURM_MAX_CONCURRENCY < 1 || SLURM_MAX_CONCURRENCY > 50 )); then
  echo "SLURM_MAX_CONCURRENCY must be in [1, 50]" >&2
  exit 2
fi

cd "$REPO_ROOT"
export REPO_ROOT

cap_array() {
  local count="$1"
  if (( SLURM_MAX_CONCURRENCY < count )); then
    echo "$SLURM_MAX_CONCURRENCY"
  else
    echo "$count"
  fi
}

prepare_job=$(sbatch --parsable --export=ALL \
  scripts/bash_script/SNN_Bash/prepare_exp_13_source_cpu.bash)

a1_conc=$(cap_array 9)
a1_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL \
  --array="0-8%${a1_conc}" \
  scripts/bash_script/SNN_Bash/run_exp_13_a1_cpu_array.bash)

a2_conc=$(cap_array 36)
a2_job=$(sbatch --parsable --dependency="afterok:${a1_job}" --export=ALL \
  --array="0-35%${a2_conc}" \
  scripts/bash_script/SNN_Bash/run_exp_13_a2_cpu_array.bash)

a3_conc=$(cap_array 12)
a3_job=$(sbatch --parsable --dependency="afterok:${a2_job}" --export=ALL \
  --array="0-11%${a3_conc}" \
  scripts/bash_script/SNN_Bash/run_exp_13_a3_cpu_array.bash)

b_conc=$(cap_array 36)
b_job=$(sbatch --parsable --dependency="afterok:${a3_job}" --export=ALL \
  --array="0-35%${b_conc}" \
  scripts/bash_script/SNN_Bash/run_exp_13_b_cpu_array.bash)

c_conc=$(cap_array 3)
c_job=$(sbatch --parsable --dependency="afterok:${b_job}" --export=ALL \
  --array="0-2%${c_conc}" \
  scripts/bash_script/SNN_Bash/run_exp_13_c_cpu_array.bash)

d_conc=$(cap_array 24)
d_job=$(sbatch --parsable --dependency="afterok:${c_job}" --export=ALL \
  --array="0-23%${d_conc}" \
  scripts/bash_script/SNN_Bash/run_exp_13_d_cpu_array.bash)

e_conc=$(cap_array 9)
e_job=$(sbatch --parsable --dependency="afterok:${d_job}" --export=ALL \
  --array="0-8%${e_conc}" \
  scripts/bash_script/SNN_Bash/run_exp_13_e_cpu_array.bash)

finalizer_job=$(sbatch --parsable --dependency="afterok:${e_job}" --export=ALL \
  scripts/bash_script/SNN_Bash/finalize_exp_13_cpu.bash)

echo "Exp13 source validation: ${prepare_job}"
echo "Exp13 A1 recurrence array: ${a1_job} (9 tasks, max concurrent=${a1_conc})"
echo "Exp13 A2 history-truncation array: ${a2_job} (36 tasks, max concurrent=${a2_conc})"
echo "Exp13 A3 layer-reset array: ${a3_job} (12 tasks, max concurrent=${a3_conc})"
echo "Exp13 B context-expression array: ${b_job} (36 tasks, max concurrent=${b_conc})"
echo "Exp13 C stroke-organization array: ${c_job} (3 tasks, max concurrent=${c_conc})"
echo "Exp13 D stroke-scrambling array: ${d_job} (24 tasks, max concurrent=${d_conc})"
echo "Exp13 E cross-tau array: ${e_job} (9 tasks, max concurrent=${e_conc})"
echo "Exp13 finalizer: ${finalizer_job}"
