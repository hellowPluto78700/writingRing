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
p1_conc="$SLURM_MAX_CONCURRENCY"
if (( p1_conc > 9 )); then p1_conc=9; fi
p15_conc="$SLURM_MAX_CONCURRENCY"
if (( p15_conc > 6 )); then p15_conc=6; fi
prepare_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_15_cpu.bash)
phase1_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL --array="0-8%${p1_conc}" scripts/bash_script/SNN_Bash/run_exp_15_phase1_cpu_array.bash)
phase15_job=$(sbatch --parsable --dependency="afterok:${phase1_job}" --export=ALL --array="0-5%${p15_conc}" scripts/bash_script/SNN_Bash/run_exp_15_phase1_5_cpu_array.bash)
final_job=$(sbatch --parsable --dependency="afterok:${phase15_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_15_cpu.bash)
echo "Exp15 prepare: ${prepare_job}"
echo "Exp15 phase1: ${phase1_job} (9 tasks, max concurrent=${p1_conc})"
echo "Exp15 phase1.5: ${phase15_job} (6 tasks, max concurrent=${p15_conc})"
echo "Exp15 finalizer: ${final_job}"
