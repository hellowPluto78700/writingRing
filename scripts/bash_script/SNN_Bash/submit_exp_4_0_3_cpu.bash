#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

ARRAY_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/run_exp_4_0_3_cpu_array.bash)
PROBE_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/probe_exp_4_0_3_baseline_cpu.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}:${PROBE_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_4_0_3_cpu.bash)

echo "Experiment 4.0.3 array job: ${ARRAY_JOB} (20 new runs, max 20 concurrent CPU tasks)"
echo "Experiment 4.0.3 reused-baseline probe job: ${PROBE_JOB} (1 CPU task)"
echo "Experiment 4.0.3 finalizer job: ${FINAL_JOB} (afterok on both jobs)"
