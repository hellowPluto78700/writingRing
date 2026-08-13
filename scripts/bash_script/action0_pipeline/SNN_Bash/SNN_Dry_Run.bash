python -m snn.train_action0 \
  --pipeline_root outputs/action0_pipeline \
  --dataset_variant lowpass \
  --sample_freq 200 \
  --train_users user_0 \
  --val_users user_1 \
  --test_users user_2 \
  --batch_size 1 \
  --dry_run