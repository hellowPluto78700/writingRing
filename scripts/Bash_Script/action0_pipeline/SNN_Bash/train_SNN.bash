python -m snn.train_action0 \
  --pipeline_root outputs/action0_pipeline \
  --dataset_variant lowpass \
  --sample_freq 200 \
  --train_users user_0 user_1 user_2 user_3 user_4 user_5 user_6 user_7 user_8 user_9 user_10 user_11 user_12 user_13 user_14 \
  --val_users user_15 user_16 user_17 \
  --test_users user_18 user_19 user_20 \
  --neurons_network 24 24 24 \
  --shift_syn 2 \
  --shift_mem 1 \
  --batch_size 128 \
  --num_epochs 30 \
  --learning_rate 5e-4 \
  --spike_regularization 0 \
  --random_seed 12345 \
  --use_gpu \
  --save_model
  