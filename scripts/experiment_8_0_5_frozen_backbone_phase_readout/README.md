# Experiment 8.0.5 — Frozen backbone phase readout

This experiment freezes the completed Exp8.0.4 `l1_l2_timeshared_count` backbone for each seed and trains only new bias-free linear readouts.

Primary methods:

- `l2_whole`
- `l1_l2_timeshared`
- `l1_fixed250_l2_whole_true_phase`
- `l1_fixed250_l2_whole_destroyed_phase`

The destroyed-phase control uses deterministic sample-specific non-zero cyclic offsets, preserving the same `B x 128` L1 phase feature dimension and the same linear-head parameter count as the true-phase method while removing consistent absolute phase alignment.

Run all 12 CPU jobs plus the dependent finalizer:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_8_0_5_cpu.bash
```

Run one job manually:

```bash
python -m scripts.experiment_8_0_5_frozen_backbone_phase_readout \
  --device cpu --threads 1 run \
  --method l1_fixed250_l2_whole_true_phase --seed 11
```

Finalize:

```bash
python -m scripts.experiment_8_0_5_frozen_backbone_phase_readout \
  --device cpu --threads 1 finalize
```

Artifacts:

```text
notebooks/artifacts/
  experiment_8_0_5_frozen_backbone_phase_readout/
    frozen_backbone_phase_readout_v1/
```

The notebook is aggregation-only:

```text
notebooks/experiment_8_0_5_frozen_backbone_phase_readout.ipynb
```
