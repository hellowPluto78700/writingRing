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
if (( p1_conc > 24 )); then p1_conc=24; fi
p2_conc="$SLURM_MAX_CONCURRENCY"
if (( p2_conc > 12 )); then p2_conc=12; fi
eval_conc="$SLURM_MAX_CONCURRENCY"
if (( eval_conc > 21 )); then eval_conc=21; fi
prepare_job=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_14_1_cpu.bash)
phase1_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL --array="0-23%${p1_conc}" scripts/bash_script/SNN_Bash/run_exp_14_1_phase1_cpu_array.bash)
select_job=$(sbatch --parsable --dependency="afterok:${phase1_job}" --export=ALL scripts/bash_script/SNN_Bash/select_exp_14_1_cpu.bash)
phase2_job=$(sbatch --parsable --dependency="afterok:${select_job}" --export=ALL --array="0-11%${p2_conc}" scripts/bash_script/SNN_Bash/run_exp_14_1_phase2_cpu_array.bash)
eval_job=$(sbatch --parsable --dependency="afterok:${phase2_job}" --export=ALL --array="0-20%${eval_conc}" scripts/bash_script/SNN_Bash/eval_exp_14_1_final_cpu_array.bash)
final_job=$(sbatch --parsable --dependency="afterok:${eval_job}" --export=ALL scripts/bash_script/SNN_Bash/finalize_exp_14_1_cpu.bash)
echo "Exp14.1 prepare: ${prepare_job}"
echo "Exp14.1 phase1: ${phase1_job} (24 tasks, max concurrent=${p1_conc})"
echo "Exp14.1 validation-only selection: ${select_job}"
echo "Exp14.1 phase2: ${phase2_job} (12 tasks, max concurrent=${p2_conc})"
echo "Exp14.1 final evaluation: ${eval_job} (21 tasks, max concurrent=${eval_conc})"
echo "Exp14.1 finalizer: ${final_job}"
