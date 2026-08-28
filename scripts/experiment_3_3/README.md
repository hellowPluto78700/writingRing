# Experiment 3.3 — Does the SNN linearize nonlinear temporal structure?

## Question

Experiment 3.1 showed that the frozen SNN-L2 representation is substantially more
linearly decodable than matched raw events under `fixed250_ordered`, while
Experiment 3.2 showed that nonlinear Local/Transition residuals can improve
fixed-duration raw-event decoding.

Experiment 3.3 joins those two observations:

> Has the frozen SNN-L2 already converted part of the nonlinear temporal
> structure in raw events into linearly accessible features?

The experiment is a readout probe only. The source SNN is frozen and is never
updated by Experiment 3.3.

## Pre-registered hypothesis

Primary representation: **Fixed250**.

Primary nonlinear decoder: **Local+Transition**.

Let

- `Delta_raw = BA(Raw + Local+Transition) - BA(Raw + Linear)`
- `Delta_snn = BA(SNN-L2 + Local+Transition) - BA(SNN-L2 + Linear)`

The primary hypothesis is that the frozen SNN has already performed part of the
nonlinear temporal transformation needed by Raw:

1. mean `SNN-L2 Linear BA - Raw Linear BA > 0`;
2. `Delta_raw > 0`;
3. mean `Delta_snn - Delta_raw < 0`;
4. at least 2 of the 3 frozen SNN seeds satisfy `Delta_snn < Delta_raw`.

All four conditions must hold for the finalizer to mark the primary hypothesis
as supported. `relative10` is a secondary phase-normalized control and does not
determine the primary Fixed250 decision.

## Representation and decoder protocol

The data split, selected source objective, frozen W128 SNN checkpoints, and
linear-probe protocol are inherited from Experiment 3.1.

Sources:

- `raw`: 30 event channels.
- `snn_l2`: 128 frozen L2 spike channels from each source-SNN seed
  `(11, 23, 101)`.

Temporal representations:

- `fixed250`: ordered 250 ms bins, 16 samples/bin at 64 Hz; the final partial
  valid bin is retained and padded bins are masked.
- `relative10`: ten relative-progress bins; all ten bins are valid.

Decoders:

- `Linear`: exact Experiment 3.1 StandardScaler + validation-selected logistic
  regression baseline.
- `Local`: shared `channels -> rank16 -> GELU` within each bin, followed by a
  bias-free linear residual head.
- `Transition`: adjacent-bin low-rank bilinear feature
  `(P z_b) * (Q z_{b+1})`, followed by a bias-free linear residual head.
- `Local+Transition`: sum of both residual branches.

Every nonlinear model is

`frozen Linear logits + zero-initialized nonlinear residual`.

Epoch 0 must therefore reproduce the frozen Linear logits exactly.

## Run matrix

Raw is deterministic on the single Experiment 3.1 split, so it is trained once
for each `(temporal representation, residual decoder)`:

- `2 representations * 3 decoders = 6 Raw runs`.

SNN-L2 is repeated for the three independently trained frozen source SNN seeds:

- `3 SNN seeds * 2 representations * 3 decoders = 18 SNN runs`.

Total: **24 independent runs**.

Each run is one Slurm array task using one CPU core. The finalizer is an
`afterok` aggregation-only job.

## Training

Residual training matches Experiment 3.2:

- rank: 16
- batch size: 64
- AdamW, learning rate `1e-3`, weight decay `1e-4`
- max epochs: 200
- min epochs: 20
- early-stop patience: 25
- gradient clipping: 1.0
- validation balanced accuracy selects the checkpoint
- validation CE breaks BA ties
- epoch 0 is a valid checkpoint

The residual initialization seed is independent of source and source-SNN seed,
so variation across the SNN runs comes from the frozen SNN representation, not
from a different residual initialization.

## Integrity checks

The finalizer fails if:

- any of the 24 run artifacts is missing;
- run identity is inconsistent;
- the repeated Linear baseline differs across residual decoders;
- Experiment 3.3 does not reproduce the corresponding Experiment 3.1 ordered
  Linear probe (`C`, validation BA, test BA, test macro-F1, and test accuracy)
  within `1e-10`.

## Outputs

Finalized artifacts are written to:

`notebooks/artifacts/experiment_3_3_snn_nonlinear_accessibility/matched_raw_snn_residual_v1/`

including:

- `experiment_3_3_results.csv`
- `experiment_3_3_summary.csv`
- `experiment_3_3_paired_accessibility.csv`
- `experiment_3_3_paired_summary.csv`
- `experiment_3_3_linear_reproduction.csv`
- `experiment_3_3_conclusion.json`
- `provenance.json`

The notebook `notebooks/experiment_3_3_snn_nonlinear_accessibility.ipynb` is
analysis-only.

## Submit on Unity

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_3_3_pipeline.bash
```

The submission creates a 24-task CPU array and an `afterok` finalizer.

## Interpretation guardrail

SNN-L2 has 128 channels while Raw has 30. Therefore a **smaller** nonlinear
residual gain on SNN-L2 is conservative evidence that the SNN has made useful
temporal information more linearly accessible. A larger SNN residual gain is
capacity-confounded and should not by itself be interpreted as evidence that
SNN-L2 contains intrinsically more nonlinear information.
