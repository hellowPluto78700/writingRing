# Exp8.0.1 frozen L1/L2 fusion probe

Exp8.0.1 reuses and freezes the three Exp8.0 `234x234` checkpoints. It does not retrain the SNN.

## Submit on Unity

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_8_0_1_cpu.bash
```

This submits three CPU array tasks (seeds 11, 23, 37) and an `afterok` finalizer.

## Per-seed work

Each task extracts L1/L2 spike trajectories and evaluates eight repository-standard linear probes:

```text
l1_whole
l2_whole
l1_l2_whole
l1_fixed250
l2_fixed250
l1_l2_fixed250
l1whole_l2fixed250
l1fixed250_l2whole
```

It also records L1/L2 correctness overlap and per-block fusion-probe coefficient diagnostics.

## Final artifacts

Artifacts are written to:

```text
notebooks/artifacts/experiment_8_0_1_l1_l2_fusion_probe/l1_l2_fusion_probe_v1/
```

Finalizer outputs:

```text
probe_runs.csv
probe_summary.csv
fusion_gain_runs.csv
fusion_gain_summary.csv
correctness_overlap_runs.csv
correctness_overlap_summary.csv
coef_block_runs.csv
coef_block_summary.csv
manifest.json
```

Aggregation notebook:

```text
notebooks/experiment_8_0_1_l1_l2_fusion_probe.ipynb
```

The notebook is aggregation-only and does not recompute the SNN or probes.
