# Exp8.0.2 — Joint L1/L2 Supervision

Exp8.0.2 keeps the `234x234` two-layer SNN backbone fixed and changes only the training supervision/readout topology:

```text
l2_only
l1_l2_joint
l2_main_l1_aux (lambda=0.1)
```

Each method is trained for seeds `11, 23, 37`, giving 9 independent CPU runs.

## Submit on Unity

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_8_0_2_cpu.bash
```

The submit wrapper launches:

```text
9-way CPU array
  -> train/evaluate each (method, seed)
  -> dependent finalizer via afterok
```

Each array task uses one CPU core and performs checkpoint selection, native Linear evaluation, same-W output-LIF transfer, and frozen L1/L2 fusion probes before writing per-run artifacts.

## Outputs

Finalized artifacts are written under:

```text
notebooks/artifacts/experiment_8_0_2_joint_l1_l2_supervision/joint_l1_l2_supervision_v1/
```

Important files:

```text
method_runs.csv
method_summary.csv
probe_runs.csv
probe_summary.csv
fusion_gain_runs.csv
fusion_gain_summary.csv
correctness_overlap_runs.csv
correctness_overlap_summary.csv
coef_block_runs.csv
coef_block_summary.csv
trained_head_runs.csv
trained_head_summary.csv
history_runs.csv
manifest.json
```

The analysis-only notebook is:

```text
notebooks/experiment_8_0_2_joint_l1_l2_supervision.ipynb
```

The main interpretation is not only whether the joint native head improves accuracy, but whether joint/auxiliary supervision makes frozen `L1 + L2` probes consistently outperform the best single layer.
