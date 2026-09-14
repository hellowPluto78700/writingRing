# Exp7.2.2 — Frozen-L2 Output Readout Decomposition

## Scientific question

Exp7.2.1 showed that a frozen L2 whole-count representation followed by one shared affine classifier can retain most of the useful classification performance. Exp7.2.2 asks where the remaining E2E loss occurs:

1. Is the learned E2E output matrix itself suboptimal?
2. Or does converting signed analog class evidence into short-tau binary LIF spikes destroy information?

The experiment therefore freezes every Exp7.2 checkpoint and changes only the readout.

## Frozen checkpoint grid

The run grid is exactly the 84 Exp7.2 checkpoints:

- 7 hidden tau architectures
- 2 training families: `e2e_wc`, `local_tsce`
- 2 regularization conditions: `task_only`, `task_plus_reg`
- 3 seeds: 11, 23, 37

There are **zero SNN-backbone training runs** in Exp7.2.2. Each worker performs frozen inference and fits only a 128D multinomial LogisticRegression probe on train data, with validation-selected C.

## Readout A/B/C decomposition

For E2E checkpoints only:

### `native_e2e_lif`

The original Exp7.2 output path:

```text
L2 spikes -> native Linear(128 -> 12, no bias) -> binary LIF -> valid whole spike count
```

The output LIF has no synaptic state (`alpha_syn = 0`), short membrane state (`tau_mem = 22.54 ms`, beta about 0.5 at 64 Hz), threshold 0.5, cap 1, and subtractive reset.

### `native_w_analog_sum`

Uses the **same native E2E W** but removes the LIF conversion:

```text
score = sum_t W_e2e z_t = W_e2e sum_t z_t
```

Therefore:

```text
native_w_analog_sum - native_e2e_lif
```

isolates the loss caused by output LIF conversion under the same learned matrix.

### `probe_analog_sum`

Fits the best whole-count affine readout on the same frozen L2:

```text
score = W_probe sum_t z_t + b_probe
```

The train-only StandardScaler is algebraically folded into `W_probe` and `b_probe`, so this is exactly implementable as a streaming affine evidence accumulator. The bias is applied once at sequence end.

For E2E checkpoints:

```text
probe_analog_sum - native_w_analog_sum
```

measures the output-weight optimization gap, while

```text
probe_analog_sum - native_e2e_lif
```

is the total readout gap.

## Fixed-W probe dynamics ablation

All conditions below use the **same fitted `W_probe`, b_probe`**. No threshold rescaling or weight retraining is performed; this keeps the attribution clean.

### `probe_analog_sum`

Exact non-spiking reference:

```text
A_t = A_(t-1) + W_probe z_t
score = A_T + b_probe
```

This must exactly reproduce the frozen-L2 whole-count LogisticRegression predictions.

### `probe_if_beta1_cap1`

No membrane leak, but keep binary threshold/reset:

```text
U^-_t = U_(t-1) + W_probe z_t
S_t = 1[U^-_t >= 0.5]
U_t = U^-_t - 0.5 S_t
score = sum_t S_t + b_probe
```

`analog_sum - if_beta1_cap1` measures the combined cost of thresholding, nonnegative spike communication, reset and binary magnitude quantization without leak.

### `probe_lif_beta05_cap1`

Current-style short LIF:

```text
U^-_t = beta U_(t-1) + W_probe z_t
beta = exp(-dt / 22.54 ms) ~= 0.5 at 64 Hz
cap = 1
```

`if_beta1_cap1 - lif_beta05_cap1` isolates the additional effect of membrane leak under the same W, threshold and event cap.

### `probe_lif_beta05_cap31`

Same short LIF but allows up to 31 threshold crossings per timestep. The contrast

```text
probe_lif_beta05_cap31 - probe_lif_beta05_cap1
```

measures cap-1 / magnitude saturation.

### `probe_bipolar_lif_beta05_cap1`

Split signed evidence into two spike banks:

```text
e_pos = max(Wz, 0)
e_neg = max(-Wz, 0)
score = count_pos - count_neg + b_probe
```

Both banks use the same short beta-about-0.5, threshold 0.5, cap-1 LIF. This tests whether inability to communicate negative class evidence is a major bottleneck.

## Diagnostics

On the test split the worker records, where applicable:

- negative evidence fraction
- mean absolute affine evidence
- output events per neuron-second
- mean events per valid output position
- potential multi-threshold-crossing fraction (`pre_reset >= 2 * threshold`)
- zero-spike sample fraction
- mean absolute final membrane
- total output events per sample
- positive/negative event fractions for bipolar readout

## Primary paired contrasts

The finalizer reports:

- `native_analog_minus_native_lif`
- `probe_analog_minus_native_analog`
- `probe_analog_minus_native_lif`
- `analog_minus_if`
- `if_minus_lif`
- `multispike_minus_lif`
- `bipolar_minus_lif`

Balanced accuracy is the primary metric; accuracy and macro-F1 are retained.

## Multi-CPU strategy

Use one CPU process per frozen Exp7.2 checkpoint:

```text
84 array tasks = 7 architectures x 2 training families x 2 regularization conditions x 3 seeds
```

The Slurm array is capped at 50 simultaneous workers. Each worker requests one CPU, disables CUDA, and forces BLAS/OpenMP libraries to one thread. This avoids nested oversubscription when many LogisticRegression fits run concurrently.

The finalizer is a separate one-CPU dependency job and only runs after the full worker array succeeds.

Submit with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_2_2_cpu.bash
```

## Aggregate notebook

`notebooks/experiment_7_2_2_output_readout_decomposition.ipynb` reads aggregate outputs only:

- `architecture_table.csv`
- `readout_summary.csv`
- `paired_deltas.csv`
- `diagnostics_summary.csv`

Per-run JSON, checkpoints and raw run CSVs are intentionally not displayed in the notebook.
