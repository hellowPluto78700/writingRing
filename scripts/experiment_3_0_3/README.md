# Experiment 3.0.3 — L3 bottleneck ablation

## Question

Experiment 3.0.2 showed that architecture C, with L1/L2 using shifts `(2,3,4)`, often produced the strongest frozen linear-probe representation at L2, while the current 64-neuron single-tau L3 could reduce probe performance.

Experiment 3.0.3 separates three possible causes of this degradation:

1. additional spiking depth;
2. width compression from 128 to 64 neurons;
3. temporal-scale compression from `(2,3,4)` to single shift `3`.

## Fixed backbone

All conditions keep:

- L1: 128 neurons, shifts `(2,3,4)`;
- L2: 128 neurons, shifts `(2,3,4)`;
- 64 Hz, 30 unsigned event channels;
- the Experiment 3.0.1/3.0.2 user-disjoint split and protocol constants.

## Architectures

| ID | L3 | Purpose |
|---|---|---|
| A | 64 neurons, shift `(3)` | Reuse Experiment 3.0.2 architecture C baseline |
| B | No L3 | Test whether the third spiking layer is necessary |
| C | 128 neurons, shift `(3)` | Width control: remove only 128 -> 64 compression |
| D | 64 neurons, shifts `(2,3,4)` | Tau-family control: remove only single-tau compression |
| E | 128 neurons, shifts `(2,3,4)` | Remove both width and tau-family compression |

Key causal comparisons:

- A vs B: current L3 versus no L3;
- A vs C: width 64 versus 128 with single shift3;
- A vs D: single-tau versus multi-tau at width 64;
- C vs E: single-tau versus multi-tau at width 128;
- D vs E: width effect within multi-tau L3;
- B vs E: closest comparison of two versus three spiking layers without width or tau-family compression.

## Objectives and seeds

Objectives:

- `timestep_ce`
- `relative10_sequence_ce`
- `fixed250_sequence_ce`

Seeds:

- 11
- 23
- 101

A contributes 9 reused runs. B/C/D/E add:

```text
4 architectures x 3 objectives x 3 seeds = 36 new trainings
```

## Evaluation

Every frozen model is evaluated with:

- whole-layer full-count linear probes;
- whole-layer fixed250 linear probes;
- whole-layer firing rates;
- tau-subgroup full-count/fixed250 probes and firing rates for genuinely multi-tau layers.

For No-L3 architecture B, only L1 and L2 are evaluated and the native task head is attached directly to L2.

The finalizer also writes two diagnostic quantities:

```text
last_minus_l2_fixed250_probe_ba
```

For architectures with L3 this is the last-layer fixed250 probe BA minus the L2 fixed250 probe BA. Negative values indicate representation degradation across L2 -> L3. For No-L3 it is defined as zero.

```text
readout_utilization_gap
```

This is last-layer fixed250 probe BA minus native test BA and measures how much frozen representation quality is not captured by the native task readout.

## Multi-CPU execution

This experiment follows the repository default multi-CPU workflow.

- B/C/D/E: one independent run per Slurm array task;
- one CPU core per task;
- each task performs `train -> select best checkpoint -> evaluate -> save artifacts`;
- array concurrency is capped at 50 with `0-35%50`;
- A baseline: one CPU sequentially evaluates the 9 reused checkpoints;
- finalizer runs only after both job groups finish successfully.

Maximum simultaneous CPU cores in the formal pipeline are therefore 37.

No GPU is used by the formal pipeline.

## Submit

From the repository root:

```bash
git pull
bash scripts/bash_script/SNN_Bash/submit_exp_3_0_3_pipeline.bash
```

Monitor with:

```bash
squeue -u $USER
```

## Artifacts

Finalized outputs are written under:

```text
notebooks/artifacts/
  experiment_3_0_3_l3_bottleneck_ablation/
    l3_bottleneck_v1/
```

including:

- `experiment_3_0_3_architectures.csv`
- `experiment_3_0_3_native_results.csv`
- `experiment_3_0_3_history.csv`
- `experiment_3_0_3_layer_probes.csv`
- `experiment_3_0_3_tau_subgroup_probes.csv`
- `experiment_3_0_3_firing_rates.csv`
- `experiment_3_0_3_diagnostics.csv`
- corresponding mean/SD summary CSVs.
