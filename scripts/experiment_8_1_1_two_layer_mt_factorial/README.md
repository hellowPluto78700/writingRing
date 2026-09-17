# Experiment 8.1.1 — Two-layer multi-threshold factorial

## Question

Exp8.1 showed that fixed heterogeneous thresholds improve the linearly accessible L1 Fixed250 representation, but that improvement largely disappears after the original uniform-binary L2. Exp8.1.1 isolates whether the downstream L2 threshold code is the bottleneck.

The experiment asks:

1. Does making **L2 multi-threshold (MT3)** improve L2 local representation when L1 remains binary?
2. Does making **L2 MT3** specifically preserve the richer representation produced by an MT3 L1?
3. Is there a positive **L1 MT × L2 MT interaction**, i.e. does the benefit of L2 MT depend on L1 already being MT?

This experiment intentionally does **not** change the long-range readout, recurrent structure, objective, width, or synaptic time constants.

## Fixed training/architecture contract

All four factorial cells follow the Exp7.3 A2 contract:

- input: raw 30-channel spike train
- L1 width: 128
- L2 width: 128
- L1 synaptic shifts: `(2, 3, 4)`
- L2 synaptic shifts: `(2, 3, 4)`
- membrane dynamics, surrogate slope, optimizer, batch size: inherited from the existing A2/Exp8.1 implementation
- output: bias-free `Linear(128 -> 12)`
- training objective: valid-length mean evidence + cross entropy (A2 WCCE)
- end-to-end training of L1, L2, and linear head
- task-only loss; no auxiliary representation loss
- seeds: `11, 23, 37`
- max epochs / min epochs / patience: inherited from Exp7.3 A2

The MT threshold multipliers are fixed to:

```text
0.5 x theta0
1.0 x theta0
1.5 x theta0
```

Within each threshold bank the 128-neuron population is again divided across shifts `(2,3,4)`. Thresholds are fixed per neuron and never adapt with time.

## 2 x 2 factorial

| Method | L1 coding | L2 coding | Purpose |
|---|---|---|---|
| `bb` | uniform binary | uniform binary | A2/Exp8.1-style control |
| `mb` | MT3 | uniform binary | reproduces the Exp8.1 L1-MT condition |
| `bm` | uniform binary | MT3 | tests L2 threshold bottleneck alone |
| `mm` | MT3 | MT3 | tests whether MT must propagate through both hidden layers |

All methods are `128 -> 128`; therefore no width/capacity confound is introduced.

## Representation diagnostics

For both L1 and L2, the finalizer fits frozen linear probes to:

- pre-reset membrane state `U^-`
- communicated spike state

with two aggregations:

- whole valid-sequence mean
- ordered Fixed250 mean

This provides the local information path:

```text
L1 U^- -> L1 communication -> L2 U^- -> L2 communication
```

The finalizer reports:

- `l1_threshold_delta_* = BA(L1 communication) - BA(L1 U^-)`
- `l1_to_l2_transform_delta_* = BA(L2 U^-) - BA(L1 communication)`
- `l2_threshold_delta_* = BA(L2 communication) - BA(L2 U^-)`

The sign is intentionally preserved: thresholding can either lose or improve linear accessibility.

The primary representation endpoint is:

```text
l2_communication_fixed250_ba
```

The native A2 test BA is retained as a secondary task endpoint.

## Factorial effects

For every seed and every key metric, Exp8.1.1 computes:

```text
L1 MT main effect = 0.5 * [(MB - BB) + (MM - BM)]
L2 MT main effect = 0.5 * [(BM - BB) + (MM - MB)]
interaction        = MM - MB - BM + BB
```

The most important mechanistic comparison is:

```text
MM - MB
```

because it asks whether switching only L2 from binary to MT recovers/preserves the representation already produced by an MT L1.

A positive interaction on `l2_communication_fixed250_ba` would support the hypothesis that heterogeneous L1 information benefits specifically from heterogeneous downstream threshold coding.

## Activity diagnostics

For every layer, threshold multiplier, and synaptic shift the experiment records:

- mean communicated value per neuron-step
- fraction nonzero
- fraction > 1 (expected to be zero for all binary/MT conditions)
- fraction at cap

This is especially important for L2 because Exp8.1 showed high L2 firing rates. If the `0.5 x theta0` L2 bank becomes nearly saturated, the result should be interpreted as a threshold-range problem rather than evidence against L2 threshold heterogeneity in general.

## Parallel execution

There are:

```text
4 methods x 3 seeds = 12 independent training runs
```

The CPU runner maps one run to one Slurm array task and pins each task to one CPU thread. A dependent finalizer runs only after the full array succeeds.

Submit with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_8_1_1_cpu.bash
```

Artifacts are written to:

```text
notebooks/artifacts/experiment_8_1_1_two_layer_mt_factorial/two_layer_mt_factorial_v1/
```

The finalizer creates:

- `method_runs.csv`
- `method_summary.csv`
- `contrast_runs.csv`
- `contrast_summary.csv`
- `factorial_effect_runs.csv`
- `factorial_effect_summary.csv`
- `activity_runs.csv`
- `activity_summary.csv`
- `manifest.json`

## Notebook contract

`notebooks/experiment_8_1_1_two_layer_mt_factorial.ipynb` is analysis-only. It reads finalized artifacts and never trains models. The notebook focuses on method-level L1/L2 information flow, seed-paired contrasts, factorial main/interaction effects, and threshold/tau activity diagnostics.
