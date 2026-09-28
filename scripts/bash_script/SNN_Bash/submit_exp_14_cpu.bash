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
phase1_conc="$SLURM_MAX_CONCURRENCY"
if (( phase1_conc > 42 )); then phase1_conc=42; fi
phase2_conc="$SLURM_MAX_CONCURRENCY"
if (( phase2_conc > 6 )); then phase2_conc=6; fi
prepare_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_14_cpu.bash)
phase1_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL --array="0-41%${phase1_conc}" scripts/bash_script/SNN_Bash/run_exp_14_phase1_cpu_array.bash)
select_job=$(sbatch --parsable --dependency="afterok:${phase1_job}" --export=ALL scripts/bash_script/SNN_Bash/select_exp_14_cpu.bash)
phase2_job=$(sbatch --parsable --dependency="afterok:${select_job}" --export=ALL --array="0-5%${phase2_conc}" scripts/bash_script/SNN_Bash/run_exp_14_phase2_cpu_array.bash)
final_job=$(sbatch --parsable --dependency="afterok:${phase2_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_14_cpu.bash)
echo "Exp14 prepare: ${prepare_job}"
echo "Exp14 phase1: ${phase1_job} (42 tasks, max concurrent=${phase1_conc})"
echo "Exp14 lambda selection: ${select_job}"
echo "Exp14 phase2 C4: ${phase2_job} (6 tasks, max concurrent=${phase2_conc})"
echo "Exp14 finalizer: ${final_job}"
