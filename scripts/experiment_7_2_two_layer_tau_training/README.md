# Experiment 7.2 — Two-layer multi-tau SNN training and persistence study

## Scientific question

Exp7.2 studies how the synaptic-timescale composition of a two-layer feed-forward SNN changes hidden representation quality, persistent firing, and final classification under two genuinely different optimization strategies:

1. `e2e_wc`: two hidden SNN layers + short-tau 12-neuron output LIF, trained end-to-end with valid-length WholeCount CE.
2. `local_tsce`: two hidden SNN layers + temporary analog timestep classifier, trained like the Exp7.0 local baseline; the temporary head is discarded and a new Fixed250 LogisticRegression readout is fitted on frozen L2 spikes.

Both are compared with and without the Exp7.1-style all-valid-masked anti-persistence regularizer applied to both hidden layers.

## Fixed backbone

```text
Raw64 30-channel spike train
 -> hidden L1: 128 binary LIF
 -> hidden L2: 128 binary LIF
```

Fixed across the experiment:

- 64 Hz, 256 padded timesteps, split seed 12345
- hidden width 128/128, binary spike cap 1
- legacy synapse: `I_t = alpha I_{t-1} + W x_t`
- Exp7.0 tau_mem, threshold, surrogate, optimizer, LR
- no recurrence and no hidden bias
- seeds 11, 23, 37
- max 100 epochs, min gate 20, patience 30

At 64 Hz: shift 2≈54 ms, shift 3≈117 ms, shift 4≈242 ms, shift 5≈492 ms.

## Architecture sweep

| ID | L1 | L2 | Purpose |
|---|---|---|---|
| `234x234` | (2,3,4) | (2,3,4) | Exp7.0 local reference |
| `34x234` | (3,4) | (2,3,4) | remove s2 from L1 |
| `34x34` | (3,4) | (3,4) | both layers local/mid |
| `34x345` | (3,4) | (3,4,5) | introduce s5 into L2 |
| `34x45` | (3,4) | (4,5) | shift L2 toward longer dynamics |
| `4x4` | (4) | (4) | homogeneous mid-timescale control |
| `5x5` | (5) | (5) | homogeneous long-timescale control |

The main controlled path is `34x34 -> 34x345 -> 34x45`: L1 stays fixed at `(3,4)` while L2 is progressively lengthened.

Multi-tau groups are contiguous, not shuffled: `(3,4)=64/64`; three-shift groups use `43/43/42`. Raster plots draw these boundaries and annotate exact neuron ranges.

## Training family A — end-to-end SNN

```text
Raw30 -> L1 -> L2 -> Linear(128->12, no bias) -> short-tau output LIF -> WholeCount CE
```

For class `k`, `c_k = sum_{t<Tvalid} S_out[t,k]`. The whole hidden backbone, output projection, and output LIF are optimized jointly.

After selecting the best checkpoint, the network is frozen and three diagnostic Fixed250 LogisticRegression probes are fitted:

- `l2_fixed250`: L2 spikes, 16x128=2048 features
- `pre_output_fixed250`: 12-D class evidence before output LIF, 16x12=192 features
- `post_output_fixed250`: 12 output spike channels after output LIF, 16x12=192 features

These probes never feed gradients back into the SNN. They localize whether information loss occurs in hidden representation, the 128->12 projection, or output-LIF spike conversion / streaming readout.

## Training family B — Exp7.0-style local SNN + Linear

Training path:

```text
Raw30 -> L1 -> L2 -> temporary analog 12-class timestep head -> valid-timestep CE
```

The checkpoint is chosen by validation native balanced accuracy; lower validation native CE is the tie-break. The temporary head is then discarded. L1/L2 are frozen, L2 spikes are aggregated into ordered 250-ms bins (16 steps per bin, 16 bins), flattened to 2048 D, standardized using train only, and classified with validation-selected LogisticRegression. This retrained linear readout is the formal deployment metric for the local family.

## Regularization

Two paired conditions are used for every architecture/family/seed:

- `task_only`: task loss only
- `task_plus_reg`: task loss + all-valid-masked rate and anti-persistence losses

For each hidden layer:

```math
L_rate = E[z]
L_3,2 = E[ReLU(count_3 - 2)^2]
L_8,4 = E[ReLU(count_8 - 4)^2]
```

The two hidden layers are averaged, and `L_persist = L_3,2 + 0.5 L_8,4`. Sliding windows contribute only if the complete window lies inside the valid interval. Output neurons are never regularized. The regularizer has a 10-epoch linear warmup.

For every regularized architecture/family/seed, five deterministic train batches calibrate lambda values against both hidden input matrices. Initial target gradient ratios are 2.5% for rate and 5% for persistence. Calibration is separate for WholeCount and timestep-CE because the task gradients differ.

## Pairing

Within a seed, initialization and minibatch-order streams are independent of architecture/family/regularization. Because both hidden linears are constructed first for every model, hidden initial weights are paired across the 84 runs.

## Run matrix

```text
7 architectures x 2 training families x 2 regularization conditions x 3 seeds = 84 SNN training runs
```

Frozen probes do not add SNN training runs.

## Diagnostics

Every run writes unique checkpoints, histories, calibration JSON, evaluation JSON, training curves, rasters, probe results, and shift-resolved activity CSVs.

Rasters use one deterministic validation sample across all runs. Every run saves L1 and L2; E2E runs also save the 12-neuron output raster. Hidden plots include shift-group boundaries and the valid-length marker.

Full-test shift-resolved diagnostics include firing fraction, spikes/neuron/s, active-neuron/sample fraction, 3-step and 8-step persistence violation fractions, mean/P95 maximum contiguous run length, and global maximum run length.

## Final aggregation

The dependency finalizer creates notebook-safe aggregate artifacts:

```text
architecture_table.csv
performance_summary.csv
dynamics_summary.csv
paired_deltas.csv
calibration_summary.csv
manifest.json
```

It also retains per-run debugging tables (`performance_runs.csv`, `dynamics_runs.csv`, `paired_delta_runs.csv`, `calibration_runs.csv`). The notebook must not consume these individual-run tables.

Primary paired contrasts include:

- `task_plus_reg - task_only`
- local Fixed250+Linear minus E2E WholeCount
- E2E L2 probe minus E2E WholeCount
- E2E pre-output probe minus post-output probe

## Notebook policy

`notebooks/experiment_7_2_two_layer_tau_training.ipynb` is analysis-only. It reads only the five aggregate CSVs above. It does not train models, load checkpoints, inspect individual histories/evaluations/rasters, or display individual seed runs.

## Multi-CPU execution

Submit on Unity with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_2_cpu.bash
```

This launches an 84-task single-CPU Slurm array (`0-83`) with a concurrency cap of 24, then an `afterok` aggregate finalizer. BLAS/OpenMP thread counts are forced to 1 to prevent nested oversubscription.

Inspect the task mapping with:

```bash
python -m scripts.experiment_7_2_two_layer_tau_training list-runs
```
