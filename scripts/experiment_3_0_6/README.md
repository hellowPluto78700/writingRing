# Experiment 3.0.6 — Frozen causal temporal decoding

## Question

Can the discriminative information already present in the frozen SNN be decoded causally without knowing the final gesture duration?

## Source backbone

Experiment 3.0.5 first chooses one W128 source objective using validation-only common-probe performance. Experiment 3.0.6 then freezes that SNN completely for all causal-head comparisons.

No SNN training occurs in the primary Experiment 3.0.6 comparison.

## 250 ms causal features

At 64 Hz, each checkpoint is 16 samples = 250 ms. L2 spikes are counted per fixed bin:

```text
z_b in R^128
```

A prediction at checkpoint `b` cannot use any bin after `b`. Final gesture duration is used only to mask valid training prefixes and to define the terminal partial-bin/final prediction; it is never an input feature.

## Decoders

- `current250`: classify the current causal L2 activity only. Because the SNN state itself is continuous, this is not a reset local-window model.
- `cumulative250`: classify the cumulative 128D spike count through the current checkpoint.
- `prefix250_uniform`: concatenate all observed absolute-time bins into a fixed 16x128 vector, zero future bins, and train with uniform per-prefix weight.
- `prefix250_weighted`: same causal prefix representation, but prefix `b` receives absolute elapsed-time weight proportional to `b`.

For every gesture, prefix weights are normalized to sum to one before fitting. This prevents longer gestures from receiving more total training weight solely because they contain more checkpoints.

The linear prefix decoder is deployment-equivalent to an incremental logit accumulator: future bins are never required and the full 2048D feature does not have to be stored online.

## Model selection

Each decoder/seed task selects LogisticRegression `C` using validation Early Recognition AUC. Test curves are evaluated only after `C` is fixed.

The durable decoder choice consumed by Experiment 3.0.7 is strictly validation-only: mean validation Early Recognition AUC across seeds, with decoder name as the deterministic tie-breaker. Test metrics never participate in decoder selection.

## Evaluation

For each fixed elapsed checkpoint the complete test cohort is retained. If a gesture has already ended, its final available prediction is held rather than dropping it from later checkpoints.

Reported metrics include:

- BA / accuracy / Macro-F1 versus elapsed time;
- Early Recognition AUC;
- final BA;
- sustained `t90` and `t95`.

A reused `timestep_ce` W128 native baseline is evaluated by the finalizer for reference.

## Multi-CPU execution

```text
4 causal decoders x 3 seeds = 12 independent tasks
```

One Slurm array task handles one decoder/seed pair with one CPU core. The finalizer aggregates artifacts only and writes the validation-only `selected_causal_decoder.json`.

Artifacts are written under:

```text
notebooks/artifacts/
  experiment_3_0_6_causal_temporal_decoding/
    frozen_causal_v1/
```
