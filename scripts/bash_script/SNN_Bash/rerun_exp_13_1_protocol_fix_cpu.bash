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
export FORCE=1

cap_array() {
  local count="$1"
  if (( SLURM_MAX_CONCURRENCY < count )); then
    echo "$SLURM_MAX_CONCURRENCY"
  else
    echo "$count"
  fi
}

prepare_job=$(sbatch --parsable --export=ALL   scripts/bash_script/SNN_Bash/prepare_exp_13_1_source_cpu.bash)

fm_conc=$(cap_array 18)
f_metric_job=$(sbatch --parsable   --dependency="afterok:${prepare_job}" --export=ALL   --array="0-17%${fm_conc}"   scripts/bash_script/SNN_Bash/run_exp_13_1_f_metric_cpu_array.bash)

h_conc=$(cap_array 18)
h_job=$(sbatch --parsable   --dependency="afterok:${f_metric_job}" --export=ALL   --array="0-17%${h_conc}"   scripts/bash_script/SNN_Bash/run_exp_13_1_h_cpu_array.bash)

finalizer_job=$(sbatch --parsable   --dependency="afterok:${h_job}" --export=ALL   scripts/bash_script/SNN_Bash/finalize_exp_13_1_cpu.bash)

echo "Exp13.1 protocol-fix source refresh: ${prepare_job}"
echo "Exp13.1 protocol-fix F geometry: ${f_metric_job}"
echo "Exp13.1 protocol-fix H probes: ${h_job}"
echo "Exp13.1 protocol-fix finalizer: ${finalizer_job}"
