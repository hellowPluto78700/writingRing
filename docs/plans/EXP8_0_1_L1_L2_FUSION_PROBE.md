# Exp8.0.1 — Frozen L1/L2 Information-Complementarity Probe

## Motivation

Exp8.0 showed that the original `234x234` backbone remains the strongest overall A2 configuration, but L1 and L2 expose different accessibility patterns:

- L1 whole-count can be stronger than L2 whole-count.
- L2 ordered Fixed250 can be stronger than L1 ordered Fixed250.
- Longer L2 time constants (`345`) degrade both the native classifier and frozen representation probes.

Exp8.0.1 therefore does **not** train another SNN. It freezes the three Exp8.0 `234x234` checkpoints and asks whether L1 and L2 carry complementary information that a skip/multi-level feature path could exploit.

## Source backbone

Use only Exp8.0:

```text
architecture = 234x234
seeds = 11, 23, 37
```

For every seed:

```text
30 input channels
  -> frozen L1 (128 binary spikes)
  -> frozen L2 (128 binary spikes)
```

No hidden weights, thresholds, time constants, or output-head parameters are changed. The Exp8.0 Linear head is not used for the fusion probes.

## Main probe families

All probes use the repository-standard protocol:

```text
train-only StandardScaler
  -> LogisticRegression
  -> C selected by validation balanced accuracy
```

The main feature sets are:

| Probe | Components | Purpose |
|---|---|---|
| `l1_whole` | L1 whole count | L1 aggregate control |
| `l2_whole` | L2 whole count | L2 aggregate control |
| `l1_l2_whole` | `[L1 whole ; L2 whole]` | same-timescale whole-count fusion |
| `l1_fixed250` | L1 ordered Fixed250 | L1 temporal control |
| `l2_fixed250` | L2 ordered Fixed250 | L2 temporal control |
| `l1_l2_fixed250` | `[L1 F250 ; L2 F250]` | same-timescale ordered fusion |
| `l1whole_l2fixed250` | `[L1 whole ; L2 F250]` | primary mixed hypothesis: L1 aggregate + L2 phase-aware information |
| `l1fixed250_l2whole` | `[L1 F250 ; L2 whole]` | symmetric mixed control |

The main scientific question is not whether doubling dimension can fit training data better. The critical endpoint is whether fusion improves **test balanced accuracy after validation-selected regularization**, consistently across paired seeds.

## Primary derived metrics

For each seed:

```math
G_{whole}=BA([L1_{whole};L2_{whole}])-
\max(BA(L1_{whole}),BA(L2_{whole}))
```

```math
G_{F250}=BA([L1_{F250};L2_{F250}])-
\max(BA(L1_{F250}),BA(L2_{F250}))
```

The main mixed hypothesis is:

```math
G_{mixed}=BA([L1_{whole};L2_{F250}])-BA(L2_{F250})
```

and it is also compared with the better of its two components.

If `G_mixed > 0` consistently across seeds, that supports the interpretation that L1 retains useful aggregate/local-motion evidence that is not fully preserved in L2, while L2 contributes ordered/phase-aware information.

## Complementarity diagnostic

A raw concatenation gain can be hard to interpret by itself. Therefore Exp8.0.1 also compares the separately trained L1 and L2 probe predictions on the same test examples.

For both whole-count and Fixed250, report:

- both correct;
- L1-only correct;
- L2-only correct;
- both wrong;
- prediction disagreement rate;
- oracle union accuracy (`L1 correct OR L2 correct`).

Large nonzero `L1-only` and `L2-only` fractions are direct evidence that the two layers make different errors and contain complementary class information.

## Fusion-weight diagnostic

For every multi-block probe, store the fitted classifier coefficient diagnostics separately for each feature block:

- Frobenius norm;
- RMS coefficient magnitude;
- fraction of total block norm.

Because features are standardized using training statistics, block RMS is useful for determining whether the fused probe actually uses both L1 and L2 rather than ignoring one side of the concatenation.

## Generalization diagnostic

For every probe record:

```text
train BA
validation BA
test BA
train-test BA gap
selected C
feature dimension
```

This is mandatory because the Fixed250 features are high-dimensional and Exp8.0 already showed a large train/test decodability gap. A fusion gain that appears only on train but not on validation/test does not support a skip-connection interpretation.

## Compute strategy

There is no SNN training in Exp8.0.1. Use three independent CPU array tasks, one per source seed:

```text
3 frozen checkpoints x 1 analysis task = 3 CPU jobs
```

Each task:

1. loads the corresponding Exp8.0 `234x234` checkpoint;
2. freezes the network and extracts L1/L2 spikes for train/val/test;
3. constructs all eight feature families;
4. fits validation-selected linear probes;
5. computes complementarity and coefficient-block diagnostics;
6. writes one per-seed JSON.

A dependent finalizer aggregates all three seeds.

## Final artifacts

```text
probe_runs.csv
probe_summary.csv
fusion_gain_runs.csv
fusion_gain_summary.csv
correctness_overlap_runs.csv
correctness_overlap_summary.csv
coef_block_runs.csv
coef_block_summary.csv
manifest.json
```

The notebook reads only finalized artifacts and does not retrain or re-extract the SNN.

## Interpretation

### Outcome A: fusion improves test BA consistently

This supports an explicit L1→head skip / multi-level fusion path in the next trainable architecture:

```text
L1 spike -----> fusion/readout
   |
   v
L2 spike -----> fusion/readout
```

It would indicate that L2 transforms useful temporal information but does not preserve all useful L1 evidence.

### Outcome B: concat does not improve over L2

Then the apparent L1/L2 probe crossover in Exp8.0 is probably not sufficiently complementary to justify a larger skip path. The next local-feature experiment should focus on quantization/threshold coding or generalization rather than residual fusion.

### Outcome C: train improves strongly but test does not

Then the main bottleneck is generalization/high-dimensional temporal overfitting rather than missing multi-level capacity. This would prioritize augmentation/invariance experiments before architectural fusion.
