# Experiment 0.1 — Frozen full-Relative10 Linear on growing prefixes

## Question

If the gesture start is known but the endpoint is not, how early can a Linear classifier trained only on complete Relative10 gestures produce useful predictions when Relative10 is recomputed from the currently observed prefix?

This experiment does not train an SNN and does not train a prefix-aware classifier. It measures distribution shift and robustness of the existing complete-gesture Relative10 + Linear solution.

## Cohort and paired user splits

Experiment 0.1 reuses the Experiment 3.2 / Experiment 1.3.3 cohort and split contract:

- 64 Hz unsigned event representation;
- 30 event channels;
- 853 samples, 20 users, 12 classes;
- Action0 + Action1;
- global padded length 256;
- split seeds `(11, 23, 37, 53, 71)`;
- 12 train users / 4 validation users / 4 test users;
- user-disjoint split construction through `derive_seed(split_seed, "user_split")`.

## Training contract

For each split seed, train exactly one complete-gesture Relative10 Linear baseline:

```text
complete training gesture
-> np.array_split(valid sequence, 10)
-> sum each temporal chunk
-> flatten to 300D
-> fit train-only per-feature z-score
-> LogisticRegression(solver="lbfgs", max_iter=5000)
```

The scaler and Linear classifier are then frozen.

The finalizer requires the complete-gesture test Balanced Accuracy to reproduce the saved Experiment 1.3.3 `relative_10bin` Linear result for the same split to absolute tolerance `1e-10`.

## Growing-prefix evaluation

Assume the gesture start is available, but the true endpoint is not used for prediction.

At every 10 newly observed samples:

```text
10 samples  -> Relative10(current prefix) -> frozen scaler -> frozen Linear
20 samples  -> Relative10(current prefix) -> frozen scaler -> frozen Linear
30 samples  -> Relative10(current prefix) -> frozen scaler -> frozen Linear
...
```

At 64 Hz, one evaluation tick is:

```text
10 / 64 s = 156.25 ms
```

Relative10 is recomputed over the entire currently observed prefix at every tick. Thus:

```text
10 samples -> 1 sample per relative bin
20 samples -> 2 samples per relative bin
30 samples -> 3 samples per relative bin
...
```

No sample after the current prefix is used to build the representation.

## Important interpretation

The classifier is **not** trained on prefixes. It only sees complete-gesture Relative10 during fitting.

Therefore this experiment asks:

> How robust is the full-gesture phase-normalized Linear solution to the representation shift induced by an incomplete but growing prefix?

A poor early-prefix curve does not prove that causal prefix classification is impossible. It indicates that a prefix-aware classifier or another causal representation may be required.

## Curve semantics

At a tick with `k` observed samples, only gestures whose reference valid length is at least `k` are evaluated. Each row records:

- Balanced Accuracy;
- accuracy;
- macro-F1;
- number of active gestures;
- active-gesture coverage;
- observed samples;
- observed time in milliseconds;
- mean samples per Relative10 bin (`k / 10`).

The reference endpoint is used only to decide whether a gesture has enough recorded samples to contribute at a given tick; it is not supplied to the representation or classifier.

## Multi-CPU execution

There are five independent tasks, one per user split seed:

```text
array task 0 -> split 11
array task 1 -> split 23
array task 2 -> split 37
array task 3 -> split 53
array task 4 -> split 71
```

Each task uses one CPU core and performs:

```text
load cohort
-> construct one user-disjoint split
-> fit full-gesture Relative10 scaler + Linear
-> evaluate full train/val/test metrics
-> evaluate growing-prefix train/val/test curves
-> save run JSON + frozen model
```

The finalizer is submitted with `afterok` and aggregates existing run artifacts only.

CPU thread counts are fixed to one to avoid hidden oversubscription.

## Outputs

```text
notebooks/artifacts/
  experiment_0_1_growing_prefix_relative10/
    frozen_full_linear_v1/
      runs/
      models/
      experiment_0_1_results.csv
      experiment_0_1_prefix_curves.csv
      experiment_0_1_summary.csv
      experiment_0_1_prefix_summary.csv
      experiment_0_1_baseline_parity.csv
      provenance.json
```

## Validation

```bash
python -m pytest -q tests/test_experiment_0_1_contract.py
python -m pytest -q tests/test_repository_source_syntax.py
```

## Submit on Unity

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_0_1_pipeline.bash
```
