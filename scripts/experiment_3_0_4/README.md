# Experiment 3.0.4 — L2 width and representation capacity

## Question

Experiment 3.0.3 identified the two-layer multi-tau network as the strongest current backbone:

- L1: 128 neurons, shifts `(2,3,4)`;
- L2: 128 neurons, shifts `(2,3,4)`;
- no L3.

Experiment 3.0.4 tests whether the strong L2 result depends on L2 width itself, and whether the large fixed250 feature vector gives the downstream linear probe extra capacity that can be mistaken for better representation quality.

The experiment therefore separates:

1. SNN width/capacity;
2. raw downstream feature dimensionality;
3. dimension-matched representation quality;
4. firing/compute cost.

## Fixed backbone

All conditions keep:

- input: 30 unsigned event channels;
- L1: 128 neurons, shifts `(2,3,4)`;
- L2 shifts: `(2,3,4)`;
- no L3;
- 64 Hz data and the existing user-disjoint split;
- the same membrane, threshold, reset, optimizer, checkpoint-selection, and loss definitions as Experiments 3.0.1–3.0.3.

Only L2 width changes:

```text
32, 64, 128, 256
```

The W128 condition is exactly Experiment 3.0.3 architecture B and is reused without retraining.

## Objectives and seeds

Objectives:

- `timestep_ce`
- `relative10_sequence_ce`
- `fixed250_sequence_ce`

Seeds:

- 11
- 23
- 101

New trainings:

```text
3 new widths x 3 objectives x 3 seeds = 27 runs
```

W128 contributes 9 reused runs, so final evaluation contains 36 conditions.

## Representation probes

For both L1 and L2, every frozen checkpoint is evaluated with:

1. `full_count`
2. raw `fixed250`
3. `fixed250_pca128`

Raw fixed250 dimensions for L2 are:

| L2 width | Raw fixed250 dimension |
|---:|---:|
| 32 | 512 |
| 64 | 1024 |
| 128 | 2048 |
| 256 | 4096 |

The PCA control is fit on the training split only:

```text
raw fixed250
 -> StandardScaler fit on train
 -> PCA(128) fit on train
 -> StandardScaler fit on train PCA features
 -> LogisticRegression
```

The logistic-regression C value is selected using validation balanced accuracy and test is evaluated once.

This creates an apples-to-apples 128-dimensional comparison across all L2 widths.

## Key diagnostics

The finalizer writes:

```text
raw_l2_minus_l1_fixed250_probe_ba
```

This is the raw fixed250 L2 probe BA minus the raw fixed250 L1 probe BA.

```text
pca128_l2_minus_l1_fixed250_probe_ba
```

This is the dimension-matched PCA128 L2 probe BA minus the PCA128 L1 probe BA and is the cleaner depth/representation-gain diagnostic.

```text
l2_raw_minus_pca128_fixed250_probe_ba
```

This measures how much apparent probe performance is associated with allowing the raw L2 feature dimension to grow rather than compressing every width to the same 128-dimensional space.

The finalizer also records:

- PCA128 explained variance ratio;
- native train-test BA gap;
- native head feature dimension;
- native head parameter count;
- L2 firing rate per neuron;
- L2 total spikes per valid timestep.

The last quantity is important because wider layers can have lower per-neuron firing rate while still generating more total spike activity.

## Tau subgroup diagnostics

L1 and L2 continue to use deterministic nearly-even neuron allocation across shifts `(2,3,4)`.

For every subgroup, the experiment records:

- full-count frozen linear probe;
- fixed250 frozen linear probe;
- firing rate;
- total spikes per timestep.

This can reveal whether a tau subgroup saturates at a much smaller neuron count than the full L2 width, which would motivate a later non-uniform tau-allocation experiment.

## Multi-CPU execution

This experiment follows `AGENTS.md` task-level multi-CPU rules.

- W32/W64/W256: one independent run per Slurm array task;
- one CPU core per task;
- each task performs `train -> best checkpoint -> evaluate -> save artifacts`;
- array: `0-26%50`;
- W128 baseline: one CPU sequentially evaluates 9 reused checkpoints;
- finalizer starts only after both job groups complete with `afterok`;
- notebook is analysis-only.

Maximum simultaneous experiment CPU cores are therefore 28.

## Submit

From repository root:

```bash
git pull
python -m pytest -q \
  tests/test_repository_source_syntax.py \
  tests/test_experiment_3_0_4_contract.py

bash scripts/bash_script/SNN_Bash/submit_exp_3_0_4_pipeline.bash
```

Monitor with:

```bash
squeue -u $USER
```

## Artifacts

Finalized outputs are written under:

```text
notebooks/artifacts/
  experiment_3_0_4_l2_width_representation_capacity/
    l2_width_v1/
```

including native results, histories, raw/PCA layer probes, tau-subgroup probes, firing-rate summaries, and width diagnostics.
