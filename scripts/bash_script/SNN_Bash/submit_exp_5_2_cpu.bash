#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

LOCAL_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_5_2_local_cpu_array.bash)
DECODER_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${LOCAL_JOB}" scripts/bash_script/SNN_Bash/run_exp_5_2_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${DECODER_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_5_2_cpu.bash)

echo "Exp5.2 frozen-local preparation: ${LOCAL_JOB} (5 seeds)"
echo "Exp5.2 FF/RSNN tauR array: ${DECODER_JOB} (50 runs, max 50 concurrent CPU tasks)"
echo "Exp5.2 finalizer: ${FINAL_JOB} (afterok on all decoder tasks)"
