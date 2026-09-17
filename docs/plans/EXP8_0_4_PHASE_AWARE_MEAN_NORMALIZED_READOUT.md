# Exp8.0.4 — Phase-aware hierarchical readout with mean-normalized CE

## Question

Exp8.0.3 showed that phase-specific L1 mappings can outperform a parameter-count-matched no-phase control, but its L2-only native BA collapsed after changing the end-to-end objective from A2 valid-length mean logits to raw count logits.

Exp8.0.4 asks one controlled question:

> Does restoring valid-length-mean CE recover the local backbone while preserving the benefit of phase-aware L1 evidence?

## Single change from Exp8.0.3

Exp8.0.3 trained with

```text
score = sum_t e_t
loss  = CE(score, y)
```

Exp8.0.4 trains with

```text
score = sum_t e_t / T_valid
loss  = CE(score, y)
```

The per-timestep evidence stream is unchanged. Therefore a frozen model has the same argmax under sum and mean scores, while the CE temperature / gradient scaling during training is changed.

## Backbone

Fixed for every run:

- architecture: `234x234`
- L1 shifts: `(2,3,4)`
- L2 shifts: `(2,3,4)`
- width: 128 / 128
- binary hidden spikes
- same optimizer, data split, initialization pairing, early stopping and checkpoint rule as Exp8.0.3
- seeds: `11, 23, 37`

## Methods

Method identifiers are intentionally retained from Exp8.0.3 to preserve exact paired comparisons. The `_count` suffix describes the feature/readout construction, not the CE normalization used in Exp8.0.4.

1. `l2_only_count`
   - `e_t = W2 z2_t`

2. `l1_l2_timeshared_count`
   - `e_t = W1 z1_t + W2 z2_t`

3. `l1_fixed250_l2_whole_count`
   - phase-aware L1 head
   - `e_t = W1,b(t) z1_t + W2 z2_t`

4. `l1_capacity_no_phase_l2_whole_count`
   - same stored L1 bank size as method 3
   - no phase access
   - effective time-shared weight is `sum_b W1,b / sqrt(B)`

The primary comparison remains method 3 minus method 4.

## Readout equivalence

For the phase-aware method,

```text
sum_t e_t
= W1 Fixed250(z1) + W2 Whole(z2)
```

Exp8.0.4 trains on

```text
[W1 Fixed250(z1) + W2 Whole(z2)] / T_valid
```

The contract test verifies that the per-timestep implementation and the explicit feature-form implementation agree numerically after normalization.

## Output LIF

The same raw per-timestep evidence stream is transferred to the existing output LIF without rescaling:

- alpha = 0
- beta = 0.5
- threshold = 0.5
- cap = 1
- final readout = output spike count

This keeps the Linear-to-LIF compatibility question identical to Exp8.0.3.

## Diagnostics

For each run save:

- native train/val/test BA and CE
- same-evidence output-LIF BA and penalty
- L1/L2 branch score RMS
- trained head norms
- phase-bank cosine / between-bin dispersion
- Exp8.0.1-style frozen probes for L1, L2 and fusion variants
- correctness overlap / oracle union
- training history

The main cross-experiment sanity check is:

```text
Exp8.0.4 l2_only_count ≈ Exp8.0.2 / A2 L2-only
```

If that recovers while phase-aware remains above the no-phase control, then Exp8.0.3's low native BA was a loss-normalization artifact rather than evidence against phase-aware supervision.

## Parallel execution

`4 methods x 3 seeds = 12 independent CPU jobs`.

Slurm array:

```text
0-11%12
```

Each task uses one CPU and forces OMP/MKL/OpenBLAS/NumExpr to one thread. The finalizer is submitted with `afterok` and aggregates all 12 completed runs.

Submit with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_8_0_4_cpu.bash
```

## Notebook aggregation

`notebooks/experiment_8_0_4_phase_aware_hierarchical_readout_mean.ipynb` only reads finalized CSV/JSON artifacts. It does not train models or refit probes.
