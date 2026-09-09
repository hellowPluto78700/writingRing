# Experiment 5.5.4 — Causal WHEN Phase Recovery with Frozen Relative10 Teacher

## Scientific question

Exp5.5.3 established an exactly streaming-equivalent Relative10 teacher. Exp5.5.4 freezes that classifier and asks only:

> How much of the oracle Relative10 performance can be recovered when the true phase is replaced by a causal WHEN prediction from frozen Local-SNN WHAT history?

The causal chain is:

```text
frozen WHAT m_t
  -> causal WHEN (GRU64 or direct RSNN64)
  -> predicted progress r_hat_t
  -> hard or local2 Relative10 routing
  -> frozen Exp5.5.3 W_k and bias
  -> whole-gesture evidence sum
```

Classification CE never trains or selects the WHEN model.

## Frozen sources

For each seed `(11, 23, 37, 53, 71)` the experiment reuses:

- frozen Local-SNN L2 WHAT trajectory `m_t in R^128` from the Exp5.4/5.5 fusion cache;
- the Exp5.5.3 Relative10 teacher `W in R^(10 x 12 x 128)` and class bias.

The source-validation stage refuses to regenerate either source. It also applies the fatal gate that oracle hard replay must match Exp5.5.3 hard streaming logits to `1e-6` and have identical predictions.

## WHEN architectures

The primary comparison is state-width matched, not parameter matched.

### `gru64_progress`

```text
WHAT128 -> GRU64 -> Linear(64,1) -> sigmoid -> progress
```

The progress head reads the current causal state `h_t`.

### `rsnn64_progress`

```text
WHAT128 -> Linear(128,64) -> recurrent LIF64 -> membrane_t -> Linear(64,1) -> sigmoid
```

The RSNN copies the direct short-memory RSNN dynamics used in Exp5.3.2.x: trainable input and recurrent weights, short fixed decay, surrogate spike gradient, and membrane-state progress readout. There is no FF-SNN front-end.

## R10-aligned supervision

For a valid length `T` and timestep `t`:

```text
r*_t = t / T
k*_t = floor(10 * t / T)
```

Thus `floor(10*r*_t)` exactly matches the Relative10 routing rule used by Exp5.5.3. `T` is used only to construct targets and masks; it is never supplied to GRU/RSNN.

## Training objective

Only sample-balanced progress regression is trained:

```text
per_sample = mean_valid_t SmoothL1(r_hat_t, r*_t)
loss = mean_samples(per_sample)
```

with SmoothL1 `beta=0.1`.

There is no classification CE, phase CE, sticky loss, confidence loss, monotonic loss, expert loss, or teacher fine-tuning.

## Checkpoint selection

Epoch 0 and trained epochs are eligible. Selection is based only on WHEN validation metrics:

1. minimum validation sample-balanced progress MAE;
2. tie: lower validation sample-balanced SmoothL1;
3. tie: higher validation Spearman.

Test classification and test progress never select checkpoints.

## Evaluation

For every seed the source stage records two oracle controls and every trained WHEN run records two predicted controls:

| WHEN | routing | frozen teacher |
|---|---|---|
| oracle | hard | Relative10 |
| oracle | local2 | Relative10 |
| GRU64 | hard | Relative10 |
| GRU64 | local2 | Relative10 |
| RSNN64 | hard | Relative10 |
| RSNN64 | local2 | Relative10 |

`local2` interpolates only between the two nearest Relative10 expert centers. Outside the first/last center it clamps to the endpoint expert. It never performs broad RBF mixing.

## WHEN metrics

Each split records:

- sample-balanced progress MAE;
- sample-balanced SmoothL1;
- progress R2;
- timestep Spearman;
- sample-balanced 10-bin accuracy;
- sample-balanced mean absolute bin error;
- sample-balanced `|bin error| <= 1` accuracy;
- progress backtrack rate `P(r_hat[t+1] < r_hat[t])`;
- phase backtrack rate `P(k_hat[t+1] < k_hat[t])`.

Test trajectories additionally save a 10x10 phase confusion matrix.

## Downstream diagnostics

For every predicted WHEN run the experiment saves per-sample routing disagreement and margin degradation:

```text
routing disagreement = mean_t [k_hat_t != k*_t]
margin delta = predicted true-class margin - oracle true-class margin
```

The source stage also saves phase sensitivity:

```text
D[k,j] = E || W_k m_t - W_j m_t ||_2
```

conditioned on the oracle true phase `k`.

## Multi-CPU execution

The implementation follows `AGENTS.md` task-level execution:

```text
5 source/teacher validation tasks
            |
          afterok
            |
10 WHEN train -> best checkpoint -> evaluate tasks
(GRU64 / RSNN64 x 5 seeds, max 10 concurrent)
            |
          afterok
            |
1 artifact-only finalizer
```

Each Slurm task uses one CPU core and initializes Conda locally.

From the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_5_4_cpu.bash
```

## Finalized artifacts

```text
notebooks/artifacts/
  experiment_5_5_4_causal_when_phase_recovery/
    causal_when_phase_recovery_v1/
      source_validation/
      checkpoints/
      histories/
      evaluations/
      trajectories/
      routing_disagreement/
      margin_degradation/
      phase_sensitivity/
      runs.csv
      summary.csv
      paired_deltas.csv
      phase_metrics.csv
      phase_confusion.csv
      routing_disagreement.csv
      margin_degradation.csv
      phase_sensitivity.csv
      manifest.json
```

The notebook `notebooks/experiment_5_5_4_causal_when_phase_recovery.ipynb` is analysis-only and reads finalized artifacts. It does not train, submit Slurm jobs, select checkpoints, or regenerate missing files.
