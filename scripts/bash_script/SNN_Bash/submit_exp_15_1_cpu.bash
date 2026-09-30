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

b0_conc="$SLURM_MAX_CONCURRENCY"
if (( b0_conc > 6 )); then b0_conc=6; fi
p1_conc="$SLURM_MAX_CONCURRENCY"
if (( p1_conc > 18 )); then p1_conc=18; fi
p15_conc="$SLURM_MAX_CONCURRENCY"
if (( p15_conc > 12 )); then p15_conc=12; fi

prepare_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_15_1_cpu.bash)
baseline_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL --array="0-5%${b0_conc}" scripts/bash_script/SNN_Bash/run_exp_15_1_baseline_cpu_array.bash)
phase1_job=$(sbatch --parsable --dependency="afterok:${baseline_job}" --export=ALL --array="0-17%${p1_conc}" scripts/bash_script/SNN_Bash/run_exp_15_1_phase1_cpu_array.bash)
phase15_job=$(sbatch --parsable --dependency="afterok:${phase1_job}" --export=ALL --array="0-11%${p15_conc}" scripts/bash_script/SNN_Bash/run_exp_15_1_phase1_5_cpu_array.bash)
final_job=$(sbatch --parsable --dependency="afterok:${phase15_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_15_1_cpu.bash)

echo "Exp15.1 prepare: ${prepare_job}"
echo "Exp15.1 width-matched B0: ${baseline_job} (6 tasks, max concurrent=${b0_conc})"
echo "Exp15.1 phase1: ${phase1_job} (18 tasks, max concurrent=${p1_conc})"
echo "Exp15.1 phase1.5: ${phase15_job} (12 tasks, max concurrent=${p15_conc})"
echo "Exp15.1 finalizer: ${final_job}"
