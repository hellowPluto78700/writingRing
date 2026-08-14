#!/bin/bash

python scripts/plot_board_trajectory_window.py \
  --data-root data \
  --user user_20 \
  --action 1 \
  --recording 0 \
  --start-s 12 \
  --end-s 14 \
  --output outputs/plotting_verification/user20_action1_record0_12_14s.png