# Experiment 4.0 — Fixed250 temporal SNN decoder

## Question

Can a stateful SNN replace the flattened Linear decoder after the raw 30-channel weighted event train has already been compressed into ordered 250 ms channel-wise sums?

The representation is fixed before any classifier-specific branch:

```text
raw 30-channel weighted events
  -> 250 ms channel-wise aggregation
  -> train-only per-channel zero-preserving scaling
  -> ordered Fixed250 vectors
  -> Linear / FF-SNN / RSNN
```

The SNN is therefore tested as a temporal decoder, not as an additional local feature extractor.

## Shared Fixed250 input

At 64 Hz, one 250 ms bin contains 16 raw samples. For event channel `c` and absolute bin `b`:

```text
z[b,c] = sum_t X[t,c], t in bin b and t < valid_length
```

The last non-empty partial bin is retained. Padded bins after the endpoint are all zero.

One channel scale is fit from training users only, using valid Fixed250 bins only:

```text
scale[c] = std(train valid z[:,c])
z_scaled[b,c] = z[b,c] / scale[c]
```

No mean is subtracted. Therefore exact zero inputs, including padded bins, remain zero. The same 30 scales are applied to every temporal position and to train/validation/test for all methods.

## Methods

### Linear baseline

The scaled Fixed250 sequence is flattened and passed to multinomial Logistic Regression. No second centered StandardScaler is applied.

### FF-SNN

```text
30 -> H LIF -> 12 output LIF
```

The hidden layer receives the current Fixed250 vector. There is no recurrent matrix. Temporal state comes from the hidden LIF membrane.

### RSNN

The same single-hidden-layer network adds a dense hidden-to-hidden recurrent matrix:

```text
I_b = W_in z_b + W_rec S_(b-1)
```

For width `H`, recurrence adds `H^2` trainable weights.

## Dynamics and variable length

Every sample executes all padded macro timesteps. After the true endpoint, the input vectors are zero but hidden and output membrane dynamics continue normally. State is never frozen by the valid-length mask.

Valid length is used only to extract readouts from the same complete trajectory:

- `valid_count`: output spike count from bin 1 through the final valid bin;
- `full_count`: output spike count through all padded bins;
- `valid_membrane`: output membrane at the final valid bin;
- `full_membrane`: output membrane at the final padded bin.

The primary training objective is cross entropy on `valid_count`. `full_count`, `valid_membrane`, and `full_membrane` are diagnostic readouts from the selected checkpoint and do not cause a second forward run.

## Sweep

```text
architecture: ff, rsnn
hidden width: 32, 64, 128
hidden tau_mem: 250, 500, 1000, 2000 ms
training seeds: 11, 23, 37, 53, 71
fixed user split seed: inherited from Experiment 3.0.1
```

This gives 120 independent SNN runs. The output layer uses a fixed 250 ms membrane time constant so the hidden-memory sweep is not duplicated in the readout neuron dynamics.

For 12 classes and 30 input channels, ignoring biases because all Linear maps are bias-free:

| H | FF parameters | recurrent matrix | RSNN parameters |
|---:|---:|---:|---:|
| 32 | 1,344 | 1,024 | 2,368 |
| 64 | 2,688 | 4,096 | 6,784 |
| 128 | 5,376 | 16,384 | 21,760 |

## Multi-CPU execution

The implementation follows `AGENTS.md` task-level multi-CPU rules.

- one Slurm array task per SNN configuration/seed;
- one CPU core per task;
- array concurrency capped at 50;
- each task performs train -> checkpoint selection -> train/val/test evaluation -> per-run artifact write;
- the Linear baseline is a separate one-core job;
- the finalizer has `afterok` dependencies on both the SNN array and Linear job;
- the finalizer only aggregates existing artifacts and fails if any required artifact is missing;
- the notebook is analysis-only.

Submit everything with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_0_cpu.bash
```

## Artifacts

```text
notebooks/artifacts/
  experiment_4_0_fixed250_temporal_snn/
    fixed250_stateful_decoder_v1/
      checkpoints/
      evaluations/
      baseline/fixed250_linear.json
      runs.csv
      summary.csv
      provenance.json
```

## Required checks

Before merging, run:

```bash
python -m pytest -q tests/test_experiment_4_0_contract.py
python -m pytest -q tests/test_repository_source_syntax.py
```
