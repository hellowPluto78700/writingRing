#!/bin/bash

python scripts/plot_board_trajectory_window.py \
  --data-root data \
  --user user_0 \
  --recording 2 \
  --start-s 7 \
  --end-s 9 \
  --output outputs/plotting_verification/user0_record2_7_9s.png