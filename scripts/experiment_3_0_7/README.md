# Experiment 3.0.7 — Calibrated online early decision

## Question

When does the best causal classifier have enough evidence to make a reliable irreversible letter prediction, and what is the accuracy-latency-spike-cost tradeoff?

## Source model

Experiment 3.0.7 uses the validation-selected causal decoder from Experiment 3.0.6 and the same frozen W128 SNN checkpoints. It does not retrain the SNN or causal classifier.

## Confidence calibration

For each seed, all valid validation-prefix logits are used to fit one scalar temperature by minimizing validation negative log-likelihood over a fixed temperature grid. Test data never participates in calibration.

## Decision policies

Thresholds:

```text
0.50, 0.60, 0.70, 0.80, 0.90, 0.95
```

Stability requirements:

- `M=1`: one qualifying checkpoint;
- `M=2`: two consecutive qualifying predictions = 250 ms persistence;
- `M=3`: three consecutive qualifying predictions = 500 ms persistence.

A qualifying sequence must keep the same predicted label and keep calibrated confidence above threshold at every required checkpoint.

Once a prediction is committed it is irreversible. If no policy commits strictly before gesture offset, the terminal partial-bin prediction is forced at the true gesture end and the sample is marked as a forced decision.

## Metrics

Classification:

- balanced accuracy;
- accuracy;
- Macro-F1.

Latency:

- mean and median decision latency;
- P(T <= 500 ms), P(T <= 750 ms), P(T <= 1000 ms), P(T <= 1500 ms).

Reliability:

- early-exit coverage;
- forced-decision rate;
- premature wrong-decision rate;
- conditional early-exit accuracy.

Neuromorphic activity proxies:

- input-event + L1 + L2 spikes/events accumulated until decision;
- spikes per correctly classified gesture;
- estimated SynOps until decision.

The SynOps value is an activity-based compute proxy, not a hardware energy measurement.

## Model-independent fixed-time curve

The experiment also reports forced decision performance at each fixed 250 ms checkpoint with the complete cohort retained and ended gestures holding their final prediction. This separates information accumulation from the stopping policy.

## Operating-point selection

All threshold/stability policies are evaluated on validation and test, but the single reported operating point is selected on **validation only** by:

1. maximize mean validation balanced accuracy across seeds;
2. minimize mean validation decision latency.

The complete pre-specified test frontier remains available for analysis.

## Multi-CPU execution

```text
3 seeds = 3 independent tasks
```

Each task evaluates all thresholds and stability settings for one seed on one CPU core. The finalizer aggregates existing artifacts only.

Artifacts are written under:

```text
notebooks/artifacts/
  experiment_3_0_7_online_early_decision/
    calibrated_commit_v1/
```
