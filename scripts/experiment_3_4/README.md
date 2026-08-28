# Experiment 3.4 — Causal Fixed250 Objective Comparison

## Question

Can fixed-duration causal-prefix supervision improve SNN classification balanced accuracy while preserving real-time causality?

## Hypothesis

Progressive Fixed250 prefix supervision will improve streaming evidence accumulation, and may improve final balanced accuracy, relative to parameter-matched whole-segment Fixed250 supervision.

## Fixed protocol

- Input: 30 unsigned event channels at 64 Hz.
- User split seed: 12345.
- Training seeds: 11, 23, 101.
- Backbone: selected Phase-C Architecture B, `30 -> 128 -> 128`, no L3.
- L1 shifts: `(2, 3, 4)`.
- L2 shifts: `(2, 3, 4)`.
- Fixed bin: 250 ms = 16 samples.
- Optimizer/training defaults reuse Experiment 3.0.1.
- Relative10 is not used by training or inference; it remains an offline oracle/reference only.

## Objectives

1. `timestep_ce`: shared `128 -> 12` head at every valid timestep; timestep CE terms are averaged. Final segment logits are the mean valid timestep logits.
2. `whole_fixed250_ce`: valid L2 spikes are counted in ordered 250-ms bins, flattened, and passed to one `2048 -> 12` linear head. One final CE term per segment.
3. `causal_fixed250_prefix_ce`: uses exactly the same Fixed250 head shape and paired head initialization as `whole_fixed250_ce`. The head is viewed as position-specific blocks `W_k`; prefix logits are `h_k=b+sum_{j<=k} W_j z_j`. Non-final prefixes contribute a prefix loss, while the final non-empty bin contributes the endpoint loss.

For causal training:

`L = 0.5 * L_prefix + 0.5 * L_final`

when at least one non-final prefix exists; otherwise `L=L_final`.

## Partial final bin

The last non-empty Fixed250 bin is retained even when shorter than 250 ms. Samples beyond `valid_length` are masked to zero before counting. The partial final bin is used for the final endpoint prediction and is excluded from the non-final prefix-loss average, so it is not double-counted.

## Causality

A periodic prefix at 250, 500, 750, ... ms only depends on samples observed by that time. No relative-progress normalization and no future duration information are used to construct those prefix predictions.

## Validation standard

Primary comparison: `causal_fixed250_prefix_ce` vs parameter-matched `whole_fixed250_ce`.

Support requires both:

1. mean paired final test BA gain across seeds is positive;
2. causal final test BA exceeds whole-segment Fixed250 in at least 2/3 seeds.

Streaming BA from 500–1500 ms is a secondary diagnostic, not part of the primary decision rule.

## Slurm execution

Nine independent runs are mapped one-to-one to CPU array tasks:

- tasks 0–2: `timestep_ce`, seeds 11/23/101;
- tasks 3–5: `whole_fixed250_ce`, seeds 11/23/101;
- tasks 6–8: `causal_fixed250_prefix_ce`, seeds 11/23/101.

Each task uses one CPU core and performs train -> best validation-final-BA checkpoint selection -> final and streaming evaluation -> per-run artifact write.

Submit with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_3_4_pipeline.bash
```

The finalizer runs with `afterok` and only aggregates existing artifacts.
