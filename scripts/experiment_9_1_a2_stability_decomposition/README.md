# Exp9.1 — A2 Optimization Stability and Representation Failure Decomposition

## Goal

Exp9.0 showed that A2 can reproduce its historical ~56–57% BA when training converges, but some fold/initialization combinations collapse to much lower train and test BA. Exp9.1 separates:

1. implementation/protocol drift;
2. fold difficulty;
3. model-initialization sensitivity;
4. fold × initialization interaction;
5. representation loss after successful optimization.

The A2 architecture is fixed:

```text
30 -> L1(128, shifts 2/3/4) -> L2(128, shifts 2/3/4) -> Linear(12)
```

The native objective remains valid-time mean evidence followed by cross-entropy. No architecture, tau, threshold, loss, regularizer, or augmentation sweep is introduced.

## Stage A — old-split reproduction

Three runs use the original Exp8/Exp3 fixed split and exact old A2 seed contracts:

```text
seeds = 11, 23, 37
model init = exp73._e2e_pair_seed(seed, "model_init")
loader order = exp73._raw_loaders(data, seed, ...)
```

These runs are compared directly with committed Exp8.0 A2 references. They answer whether the current implementation still reproduces the historical A2 baseline.

## Stage B — 5 fold × 5 seed factorial

Only the Exp9.0 within-user segment-level folds are used. Every fold is trained with every seed:

```text
5 folds x 5 model seeds = 25 A2 trainings
```

This removes the Exp9.0 confound where one fold was tied to one seed.

The same model initialization for a given seed is reused across all folds. Fold identity is never mixed into the model-init seed.

## Collapse diagnostic

A run is tagged as an optimization collapse when:

```text
best train BA < 0.70
```

This is a diagnostic label only; collapsed runs remain in all artifacts.

Each run records:

- train/val/test Accuracy, BA, Macro-F1 and CE;
- best train BA;
- epoch first reaching 70% train BA;
- best/stopped epoch;
- per-epoch train/val loss and BA;
- per-epoch L1/L2/output gradient norms;
- L1/L2 firing rate, dead-neuron fraction and spike occupancy at epochs 1, 5, 10, 20, 40, 60, 80 and 100.

The checkpoint also stores the final epoch model and optimizer state so a later Exp9.1 continuation study can resume collapsed runs without retraining the first 100 epochs.

## Frozen representation probes

At the selected checkpoint, L1 and L2 are frozen and evaluated with post-hoc Linear probes:

- L1 whole-count;
- L1 Fixed250;
- L2 whole-count;
- L2 Fixed250.

The key comparisons are:

```text
native A2 vs L2 whole-count
native A2 vs L2 Fixed250
L1 probe vs L2 probe
```

They distinguish backbone failure, L2 representation destruction, and native-readout bottlenecks.

## Multi-CPU strategy

Submission graph:

```text
prepare
   |-----------------------------|
   v                             v
3-way old-split array      25-way fold×seed array
   |                             |
   |-------------|---------------|
                 v
              finalizer
```

All array tasks use one CPU and set OpenMP/MKL/OpenBLAS/NumExpr threads to one.

Run:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_9_1_cpu.bash
```

List task identities:

```bash
python -m scripts.experiment_9_1_a2_stability_decomposition list-runs
```

## Main outputs

Artifacts are written under:

```text
notebooks/artifacts/experiment_9_1_a2_stability_decomposition/
  a2_stability_decomposition_v1/
```

Important aggregate files:

- `reproduction_runs.csv`
- `factorial_runs.csv`
- `test_ba_matrix.csv`
- `best_train_ba_matrix.csv`
- `collapse_matrix.csv`
- `fold_effect_summary.csv`
- `seed_effect_summary.csv`
- `successful_runs.csv`
- `collapse_summary.json`
- `native_vs_probe.csv`
- `training_diagnostics.csv`
- `activity_diagnostics.csv`

The notebook is aggregation-only and never retrains a model.
