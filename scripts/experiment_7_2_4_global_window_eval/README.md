# Exp7.2.4 — Global-Window Evaluation of Exp7.2.3

Exp7.2.4 is an **evaluation-only** extension of Exp7.2.3. It never retrains A1-S4 backbones or native classifier heads. It reuses all 96 trained/reused Exp7.2.3 checkpoints and asks what information remains linearly accessible from L2 when the evaluator does **not** know the valid endpoint.

## Scientific question

Exp7.2.3 measured frozen-L2 accessibility with valid-length masking:

- `l2_wholecount_linear`: valid-time L2 count, 128-D
- `l2_fixed250_linear`: 16 ordered 250-ms valid-time bins, 2048-D

Exp7.2.4 adds the matched no-mask controls:

- `l2_global_wholecount_linear`: sum all 256 L2 timesteps, 128-D
- `l2_global_fixed250_linear`: split the full 256-step window into 16 absolute 16-step bins and flatten, 2048-D

The global feature builders intentionally do not accept a `lengths` argument. Thus post-end residual SNN activity is included by construction.

At 64 Hz, 250 ms is exactly 16 timesteps and 256 timesteps form exactly 16 Fixed250 bins.

## Checkpoint coverage

The run mapping is identical to the full Exp7.2.3 evaluation set:

- architectures: `234x234`, `34x345`
- families: A1, A2, F1, F2, S1, S2, S3, S4
- regularization: `task_only`, `task_plus_reg`
- seeds: `11`, `23`, `37`

Total checkpoint evaluations:

```text
2 architectures × 8 families × 2 regularizers × 3 seeds = 96
```

Training runs in Exp7.2.4: **0**.

## Probe protocol

The classifier protocol is deliberately matched to Exp7.2.3:

```text
frozen L2 feature
-> train-only StandardScaler
-> LogisticRegression
-> validation-selected C
-> one final test evaluation
```

For each global probe, the logistic-regression random seed is paired to the corresponding Exp7.2.3 valid-length probe seed. The only intended representation-level change is therefore whether the L2 aggregation knows the valid endpoint.

## Paired contrasts

The finalizer computes seed-level paired deltas before aggregation:

- `valid_phase_gain = F250_valid - WC_valid`
- `global_phase_gain = F250_global - WC_global`
- `wc_global_minus_valid = WC_global - WC_valid`
- `f250_global_minus_valid = F250_global - F250_valid`

Positive `global_minus_valid` means using the complete 256-step window helped; negative values indicate that post-end activity and/or loss of endpoint knowledge hurt the probe.

## L2 tail diagnostics

The evaluation also measures L2 activity after each true endpoint. These diagnostics use `lengths` only for analysis; they are not used by the global probe feature builders.

Per split, Exp7.2.4 records:

- total valid-region spikes
- total post-end tail spikes
- tail fraction of all L2 spikes
- valid-region spikes / neuron / second
- tail spikes / neuron / second

It also reports post-end rates in offset windows:

- 0-250 ms
- 250-500 ms
- 500-750 ms
- 750-1000 ms
- >=1000 ms until timestep 256

## Multi-CPU execution

Submit with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_2_4_cpu.bash
```

The submitter launches:

1. one `0-95%50` Slurm array; each one-CPU task loads exactly one existing checkpoint, runs full-window inference, fits both global probes, computes tail diagnostics, and writes one independent evaluation JSON;
2. one `afterok` finalizer that only aggregates existing Exp7.2.4 evaluation JSON plus the already-finalized Exp7.2.3 `performance_runs.csv`.

No task trains or modifies a checkpoint. Missing evaluations or missing finalized Exp7.2.3 run-level results are fatal errors.

## Finalized artifacts

The finalizer writes:

- `manifest.json`
- `global_window_probe_runs.csv`
- `global_window_probe_summary.csv`
- `global_vs_valid_delta_runs.csv`
- `global_vs_valid_delta_summary.csv`
- `l2_tail_activity_runs.csv`
- `l2_tail_activity_summary.csv`
- `l2_tail_offset_runs.csv`
- `l2_tail_offset_summary.csv`
- `report_table_task_only.csv`
- `report_table_task_plus_reg.csv`

The two `report_table_*.csv` files collapse the two architectures and three seeds (six observations per family) and contain both the pre-existing valid-length columns and the new global-window columns.

## Notebook reporting order

`notebooks/experiment_7_2_4_global_window_eval.ipynb` is analysis-only and reports results in this order:

1. `task_only` — valid-length table
2. `task_only` — global-window table
3. `task_plus_reg` — valid-length table
4. `task_plus_reg` — global-window table
5. architecture-specific global-vs-valid deltas
6. L2 tail-activity diagnostics

The notebook reads finalized summary/report CSVs only. It never loads checkpoints, per-run JSON, run-level CSVs, or launches training/evaluation jobs.
