# Exp7.2.3 — Objective × Temporal Head × Output Dynamics

This experiment is a focused mechanism/decomposition study over exactly two Exp7.2 backbones:

- `234x234`: L1 shifts `(2,3,4)`, L2 shifts `(2,3,4)`
- `34x345`: L1 shifts `(3,4)`, L2 shifts `(3,4,5)`

Each family uses seeds `{11,23,37}` and regularization `{task_only, task_plus_reg}`. Therefore each family has 12 runs.

## Training families

| ID | Family | Head | Objective | Output dynamics | Source |
|---|---|---|---|---|---|
| A1 | `a1_shared_tsce` | shared analog Linear | timestep CE | analog | reuse Exp7.2 `local_tsce` |
| A2 | `a2_shared_wholecount` | shared analog Linear | sequence WholeCount CE | analog | new |
| F1 | `f1_fixed250_tsce` | 16 absolute Fixed250 heads | bin-level dense CE | analog | new |
| F2 | `f2_fixed250_wholecount` | 16 absolute Fixed250 heads | sequence WholeCount CE | analog | new |
| S1 | `s1_spike_wc_beta05` | shared output projection | WholeCount CE | LIF beta=0.5 | reuse Exp7.2 `e2e_wc` |
| S2 | `s2_spike_tsce_beta05` | shared output projection | timestep CE | LIF beta=0.5 | new |
| S3 | `s3_spike_wc_beta10` | shared output projection | WholeCount CE | IF/LIF beta=1.0 | new |
| S4 | `s4_spike_tsce_beta10` | shared output projection | timestep CE | IF/LIF beta=1.0 | new |

A1/S1 are never retrained. The new-training set is 6 families × 2 backbones × 2 regularizers × 3 seeds = 72 runs. Reused evaluation is 2 families × 2 × 2 × 3 = 24 checkpoints. Final evaluation coverage is 96 checkpoints.

## Fixed250 head semantics

At 64 Hz, 250 ms is 16 timesteps. L2 binary spikes are aggregated into 16 absolute bins. Each bin has its own trainable `W_j` and bias.

F1 applies CE to each valid bin and weights each bin loss by its valid sample fraction. F2 first forms the same phase-specific logits and then computes a valid-length weighted sequence logit before CE. This makes A2 vs F2 a shared-W versus phase-specific-W comparison under sequence-level CE.

## Unified frozen-L2 probes

Every family, including reused checkpoints, receives both post-hoc frozen probes:

- `l2_wholecount_linear`: 128-D valid L2 count + train-only StandardScaler + validation-selected LogisticRegression
- `l2_fixed250_linear`: 16×128 = 2048-D ordered Fixed250 count + the same probe protocol

Their difference is the post-hoc temporal-position accessibility gain.

## Spiking output decomposition

S1-S4 additionally evaluate the same native output weight matrix under:

- leakless analog valid accumulation
- leakless analog full-256 accumulation
- LIF beta=0.5 valid/full
- beta=1.0 valid/full

This separates output-weight quality, threshold/reset compression, membrane leak, and post-endpoint tail dynamics.

## Multi-CPU execution

Submit with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_2_3_cpu.bash
```

The submitter launches:

1. a 72-task `0-71%50` one-CPU array where each task trains one new run, selects its best checkpoint, immediately evaluates it, fits both frozen probes, and writes independent artifacts;
2. a 24-task reuse-only array for A1/S1 checkpoint evaluation;
3. one `afterok` finalizer depending on both arrays.

The finalizer aggregates only existing per-run artifacts. Missing evaluations are an error; it does not retrain or silently regenerate them.

## Aggregate outputs

The finalizer writes:

- `manifest.json`
- `architecture_table.csv`
- `performance_runs.csv`
- `performance_summary.csv`
- `representation_probe_runs.csv`
- `representation_probe_summary.csv`
- `paired_delta_runs.csv`
- `paired_delta_summary.csv`
- `output_dynamics_runs.csv`
- `output_dynamics_summary.csv`

The notebook `notebooks/experiment_7_2_3_objective_temporal_dynamics.ipynb` is analysis-only. It reads only summary-level finalized CSVs, never checkpoints, histories, per-run JSON, or run-level CSVs.
