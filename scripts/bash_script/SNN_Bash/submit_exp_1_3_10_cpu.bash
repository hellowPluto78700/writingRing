#!/bin/bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

ARRAY_SCRIPT="scripts/bash_script/SNN_Bash/run_exp_1_3_10_cpu_array.bash"
FINALIZER_SCRIPT="scripts/bash_script/SNN_Bash/finalize_exp_1_3_10_cpu.bash"
LABELS="${LABELS:-A,B,C,D,E,X,G,H,I,J,K,L}"

# Capture the exact Python executable from the already activated environment.
# On Unity, `conda` is commonly a shell function created by `module load` and
# is not reliably available inside the non-interactive Slurm job shell.
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="$(command -v python || true)"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "No usable Python executable found. Activate writingring-gpu or writingring-viz before submitting." >&2
    exit 2
fi

if ! "$PYTHON_BIN" -c "import torch, snntorch" >/dev/null 2>&1; then
    echo "Selected Python does not provide the required torch/snntorch packages: $PYTHON_BIN" >&2
    echo "Activate writingring-gpu (preferred) or writingring-viz, then resubmit." >&2
    exit 2
fi

echo "Using Python: ${PYTHON_BIN}"

ARRAY_JOB_ID="$(
    sbatch --parsable \
        --export=ALL,WRITINGRING_PYTHON="$PYTHON_BIN",LABELS="$LABELS" \
        "$ARRAY_SCRIPT"
)"
FINALIZER_JOB_ID="$(
    sbatch --parsable \
        --dependency="afterok:${ARRAY_JOB_ID}" \
        --export=ALL,WRITINGRING_PYTHON="$PYTHON_BIN",LABELS="$LABELS" \
        "$FINALIZER_SCRIPT"
)"

echo "Submitted Experiment 1.3.10 array job: ${ARRAY_JOB_ID}"
echo "Submitted afterok finalizer job: ${FINALIZER_JOB_ID}"
