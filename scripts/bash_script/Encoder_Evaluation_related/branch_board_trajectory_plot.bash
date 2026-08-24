#!/bin/bash

  python scripts/plot_board_segment_trajectories.py \
    --data-root data \
    --action 0 \
    --segmentation-root \
    outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation \
    --output-root outputs/plotting_verification \
    --overwrite