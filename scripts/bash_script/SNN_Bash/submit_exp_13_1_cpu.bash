#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
SLURM_MAX_CONCURRENCY="${SLURM_MAX_CONCURRENCY:-20}"
USER_PERMUTATIONS="${USER_PERMUTATIONS:-100}"
if (( SLURM_MAX_CONCURRENCY < 1 || SLURM_MAX_CONCURRENCY > 50 )); then
  echo "SLURM_MAX_CONCURRENCY must be in [1, 50]" >&2
  exit 2
fi
if (( USER_PERMUTATIONS < 1 )); then
  echo "USER_PERMUTATIONS must be >= 1" >&2
  exit 2
fi

cd "$REPO_ROOT"
export REPO_ROOT USER_PERMUTATIONS

cap_array() {
  local count="$1"
  if (( SLURM_MAX_CONCURRENCY < count )); then
    echo "$SLURM_MAX_CONCURRENCY"
  else
    echo "$count"
  fi
}

prepare_job=$(sbatch --parsable --export=ALL   scripts/bash_script/SNN_Bash/prepare_exp_13_1_source_cpu.bash)

fc_conc=$(cap_array 6)
f_cache_job=$(sbatch --parsable   --dependency="afterok:${prepare_job}" --export=ALL   --array="0-5%${fc_conc}"   scripts/bash_script/SNN_Bash/run_exp_13_1_f_cache_cpu_array.bash)

fm_conc=$(cap_array 18)
f_metric_job=$(sbatch --parsable   --dependency="afterok:${f_cache_job}" --export=ALL   --array="0-17%${fm_conc}"   scripts/bash_script/SNN_Bash/run_exp_13_1_f_metric_cpu_array.bash)

g_conc=$(cap_array 18)
g_job=$(sbatch --parsable   --dependency="afterok:${f_metric_job}" --export=ALL   --array="0-17%${g_conc}"   scripts/bash_script/SNN_Bash/run_exp_13_1_g_cpu_array.bash)

hf_conc=$(cap_array 18)
h_feature_job=$(sbatch --parsable   --dependency="afterok:${g_job}" --export=ALL   --array="0-17%${hf_conc}"   scripts/bash_script/SNN_Bash/run_exp_13_1_h_feature_cpu_array.bash)

h_conc=$(cap_array 18)
h_job=$(sbatch --parsable   --dependency="afterok:${h_feature_job}" --export=ALL   --array="0-17%${h_conc}"   scripts/bash_script/SNN_Bash/run_exp_13_1_h_cpu_array.bash)

finalizer_job=$(sbatch --parsable   --dependency="afterok:${h_job}" --export=ALL   scripts/bash_script/SNN_Bash/finalize_exp_13_1_cpu.bash)

echo "Exp13.1 source validation: ${prepare_job}"
echo "Exp13.1 F trajectory cache: ${f_cache_job} (6 tasks, max concurrent=${fc_conc})"
echo "Exp13.1 F geometry: ${f_metric_job} (18 tasks, max concurrent=${fm_conc})"
echo "Exp13.1 G user leakage: ${g_job} (18 tasks, max concurrent=${g_conc})"
echo "Exp13.1 H C1 history features: ${h_feature_job} (18 tasks, max concurrent=${hf_conc})"
echo "Exp13.1 H ID/OOD probes: ${h_job} (18 tasks, max concurrent=${h_conc})"
echo "Exp13.1 finalizer: ${finalizer_job}"
