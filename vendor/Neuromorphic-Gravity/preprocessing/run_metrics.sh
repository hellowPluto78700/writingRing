#!/bin/bash

#SBATCH --job-name=neurobench-metrics     # Job name
#SBATCH --time=08:00:00                   # Max run time

# Load required modules
module load conda/latest

# Activate conda environment
conda activate P2P-classifier

# Path configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/spike_encoder_neurobench_utils.py"
BASE_PATH='/work/pi_sunghoonlee_umass_edu/Ignacio'

# ── Option A: auto-scan (recommended) ───────────────────────────────────────
# Scans every existing spikesEncoder_* directory under BASE_PATH and computes
# metrics for all found dataset / subject / encoder combinations in one go.

python "$SCRIPT" \
    --compute_metrics \
    --base_path "$BASE_PATH"

# ── Option B: explicit per-combination SLURM jobs ───────────────────────────
# Uncomment the block below (and comment out Option A above) to submit each
# dataset / subject / encoder triple as an independent SLURM job.
#
# declare -A DATASET_SUBJECTS=(
#     [pamap]="$(seq 1 9)"
#     [wisdm]="$(seq 1 36)"
#     [opportunity]="$(seq 1 4)"
#     [mhealth]="$(seq 1 10)"
#     [umahand]="$(seq 1 25)"
#     [capture24]="$(seq 1 151)"
#     [adl]="$(seq 1 30)"
#     [realworld]="$(seq 0 13)"
# )
#
# spike_encoders=(
#     threshold
#     step_forward
#     moving_window
#     zero_cross_step_forward
#     time_to_first_spike
#     phase
#     bsa
#     hough_spiker
#     modified_hough_spiker
#     burst
#     poisson_rate
# )
#
# for dataset in "${!DATASET_SUBJECTS[@]}"; do
#     for subj in ${DATASET_SUBJECTS[$dataset]}; do
#         for encoder in "${spike_encoders[@]}"; do
#             sbatch --job-name="metrics-${dataset}-${subj}-${encoder}" \
#                    --partition=cpu-preempt \
#                    --time=1:00:00 \
#                    --cpus-per-task=4 \
#                    --mem=4G \
#                    --wrap="python $SCRIPT \
#                                --dataset $dataset \
#                                --subject_ID $subj \
#                                --spike_encoder $encoder \
#                                --base_path $BASE_PATH"
#         done
#     done
# done
