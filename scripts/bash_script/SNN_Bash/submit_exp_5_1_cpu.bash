#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"

LOCAL_JOB=$(sbatch --parsable --export=ALL scripts/bash_script/SNN_Bash/prepare_exp_5_1_local_cpu_array.bash)
DECODER_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${LOCAL_JOB}" scripts/bash_script/SNN_Bash/run_exp_5_1_decoder_cpu_array.bash)
PROBE_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${LOCAL_JOB}" scripts/bash_script/SNN_Bash/run_exp_5_1_interface_probe_cpu_array.bash)
FINAL_JOB=$(sbatch --parsable --export=ALL --dependency="afterok:${DECODER_JOB}:${PROBE_JOB}" scripts/bash_script/SNN_Bash/finalize_exp_5_1_cpu.bash)

echo "Exp5.1 frozen-local preparation: ${LOCAL_JOB} (3 seeds)"
echo "Exp5.1 temporal decoder array: ${DECODER_JOB} (48 runs, max 48 concurrent CPU tasks)"
echo "Exp5.1 interface probe array: ${PROBE_JOB} (6 runs)"
echo "Exp5.1 finalizer: ${FINAL_JOB} (afterok on decoder + interface probes)"
