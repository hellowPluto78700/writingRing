# Experiment 5.5.3 - Teacher-weight routing decomposition

## Scientific question

Experiment 5.5.3 isolates the unresolved fusion bottleneck from Experiments 5.5-5.5.2:

```text
Given a perfect temporal coordinate, why does the current dynamic-weight
WHAT readout still underperform the proven Fixed250 / Relative10 Linear readouts?
```

The experiment deliberately does **not** train or evaluate a GRU/RSNN WHEN branch. It freezes the same Local-SNN WHAT trajectory used by Experiment 5.5 and decomposes the gap into two candidate causes:

1. **routing softness** - replacing a hard temporal assignment with a smooth mixture may blur useful phase-specific class mappings;
2. **weight learning / optimization** - the Exp5.5 AdamW training path may fail to recover the strong weight bank already learned by `StandardScaler -> LogisticRegression`.

The experiment uses two independent temporal teacher families:

- `relative10`: normalized relative progress, 10 hard relative bins;
- `fixed250`: absolute elapsed time, 250 ms bins (16 bins at the current padded 64 Hz layout).

The teacher weights are fusion/readout parameters, not WHEN-branch parameters. They map current frozen WHAT to class evidence under a temporal state.

## Frozen source

Every condition consumes the exact Experiment 5.5 frozen WHAT cache:

```text
Raw64 30-channel weighted events
  -> frozen Local-SNN
  -> L2 spike trajectory m_t in R^128
  -> Exp5.5.3 temporal routing / weight-bank readout
```

The Local-SNN is never fine-tuned in this experiment. Seeds remain:

```text
(11, 23, 37, 53, 71)
```

All seeds reuse the existing user-disjoint split (`split_seed=12345`). These are paired stochastic/model seeds on the same split, not five independent user splits.

## Why a bin-trained Linear weight bank can be replayed per timestep

For Relative10, the original probe is

```math
z_k = \sum_{t \in B_k} m_t,
```

```math
L = b + \sum_k W_k z_k.
```

Because the readout is linear,

```math
W_k \left(\sum_{t \in B_k}m_t\right)
=
\sum_{t \in B_k}W_km_t.
```

Therefore the same classifier can be written as a streaming evidence accumulator:

```math
L = b + \sum_t W_{k(t)}m_t.
```

`W_k` is therefore a phase-conditioned WHAT-to-class evidence matrix. It is not part of the WHEN branch.

The same identity holds for Fixed250.

## Stage 0 - teacher construction and fatal equivalence gate

### Relative10 teacher

For each seed, the frozen WHAT trajectory is converted to the same hard Relative10 count representation used elsewhere in the repository. A train-only `StandardScaler` and multinomial `LogisticRegression(lbfgs)` are fitted. `C` is selected only by validation balanced accuracy from:

```text
1e-3, 1e-2, 1e-1, 1, 10
```

The standardized classifier is folded back into raw-count space:

```math
W^{raw} = W / \sigma,
```

```math
b^{raw} = b - W^{raw}\mu.
```

The flattened classifier is reshaped to

```text
[K, classes, WHAT] = [10, 12, 128].
```

### Fixed250 teacher

The experiment reuses Experiment 5.5's already validated Fixed250 raw-space reference. That source reference was produced from:

```text
Fixed250 frozen-WHAT counts
  -> train-only StandardScaler
  -> LogisticRegression(lbfgs)
  -> exact raw-space weight conversion
```

and already contains a streaming-equivalence check. Exp5.5.3 performs its own hard-routing replay check again.

### Fatal equivalence contract

For every family, seed, and split:

```math
\max |L_{offline} - L_{hard\ streaming}| < 10^{-6}
```

and predictions must be identical.

If this contract fails, teacher preparation raises immediately. Downstream training is not allowed to reinterpret an invalid teacher.

## Routing definitions

### Relative10 hard routing

The hard routing exactly follows the repository Relative10 bin assignment:

```math
k_t = \left\lfloor \frac{10t}{T} \right\rfloor.
```

### Relative10 soft routing

The same discrete coordinate is expressed in bin units:

```math
u_t = \frac{10t}{T}.
```

RBF centers are aligned to the hard bins:

```math
c_k = k + 0.5,
```

with fixed

```math
\sigma = 1.5\ \text{bins}.
```

The normalized routing is

```math
q_{t,k}=softmax_k\left[-\frac12\left(\frac{u_t-c_k}{\sigma}\right)^2\right].
```

This keeps the hard and soft coordinate geometry aligned and changes primarily the routing softness.

### Fixed250 hard routing

At 64 Hz, one 250 ms bin is 16 timesteps:

```math
k_t = \left\lfloor \frac{t}{16} \right\rfloor.
```

### Fixed250 soft routing

The RBF centers are the physical centers of the 250 ms hard bins:

```text
0.125 s, 0.375 s, 0.625 s, ...
```

with

```math
\sigma = 1.5 \times 0.25\text{ s}.
```

Padding is always masked. No evidence is accumulated after a sample's valid length.

## Shared evidence path

All hard/soft routed models use the same weight-bank computation:

```math
E_{t,k}=W_km_t,
```

```math
e_t=\sum_k q_{t,k}E_{t,k},
```

```math
L=b+\sum_{t<T}e_t.
```

There is no timestep loss, progress loss, GRU state loss, auxiliary phase loss, or Local-SNN update.

## Conditions

Each teacher family contains six final conditions.

| Condition | Routing | Weight source | Training |
|---|---|---|---|
| `hard_teacher_frozen` | hard | proven Linear teacher | none |
| `soft_teacher_frozen` | soft | same proven Linear teacher | none |
| `hard_random_train` | hard | paired random initialization | AdamW whole-gesture CE |
| `soft_random_train` | soft | same paired random initialization | AdamW whole-gesture CE |
| `soft_teacher_init_retrain` | soft | teacher initialization | AdamW whole-gesture CE |
| `soft_linear_refit` | soft | train-only StandardScaler + LogisticRegression | validation-selected C |

The first two are produced during teacher preparation. The three neural training conditions are independent Slurm tasks. The soft Linear refit is a separate independent task.

### Why `hard_random_train` matters

If hard routing plus random-init AdamW cannot recover the hard teacher, routing softness is not necessary to explain the failure. The optimization path itself is insufficient.

### Why `soft_linear_refit` matters

With fixed soft routing,

```math
\phi_k = \sum_t q_{t,k}m_t
```

and

```math
L=b+\sum_k W_k\phi_k.
```

This is still a linear classification problem in the soft routed features. Fitting the same `StandardScaler -> LogisticRegression` protocol therefore provides an optimizer-independent reference for the information retained by soft routing.

## Training protocol

The three PyTorch conditions use the Experiment 5.5 optimizer protocol:

```text
optimizer: AdamW
learning rate: 1e-3
weight decay: 1e-4
grad clip: 1.0
max epochs: 200
patience: 25
```

The objective is only final whole-gesture cross entropy.

Epoch 0 participates in checkpoint selection. This is mandatory for `soft_teacher_init_retrain`: if fine-tuning makes a strong initialization worse, the original initialization remains a valid validation-selected checkpoint.

Checkpoint selection is:

1. maximum validation balanced accuracy;
2. tie-break minimum validation CE.

Test metrics never select epochs, C, routing, conditions, or teacher families.

## Primary attribution metrics

### Softness penalty

```math
\Delta_{soft}
=
BA(soft\ teacher\ frozen)-BA(hard\ teacher\ frozen).
```

### Teacher-init retraining gain

```math
\Delta_{retrain}
=
BA(soft\ teacher\ init\ retrain)-BA(soft\ teacher\ frozen).
```

### Recovery fraction

When the hard teacher is better than the frozen soft teacher:

```math
R_{recover}
=
\frac{BA(soft\ teacher\ init\ retrain)-BA(soft\ teacher\ frozen)}
{BA(hard\ teacher\ frozen)-BA(soft\ teacher\ frozen)}.
```

### Initialization benefit

```math
\Delta_{init}
=
BA(soft\ teacher\ init\ retrain)-BA(soft\ random\ train).
```

### Hard optimization gap

```math
\Delta_{hard-opt}
=
BA(hard\ random\ train)-BA(hard\ teacher\ frozen).
```

### Soft optimizer gap

```math
\Delta_{soft-opt}
=
BA(soft\ linear\ refit)-BA(soft\ random\ train).
```

## Mechanistic diagnostics

For the frozen teacher hard -> soft change, the experiment stores:

- mean per-sample logit L2 distortion;
- mean absolute logit distortion;
- prediction flip rate;
- mean true-class margin change.

For every trained weight bank, it also stores:

- Frobenius distance to the teacher bank;
- relative Frobenius distance;
- flattened cosine similarity to the teacher bank;
- class-bias L2 distance to the teacher bias.

These measurements distinguish a small soft-routing adaptation from a wholesale replacement of the teacher basis.

## Multi-CPU execution

The repository's task-level multi-CPU policy is used exactly:

```text
5 teacher preparation / frozen-evaluation tasks
                |
              afterok
                |
      +---------+---------+
      |                   |
30 train->evaluate      10 soft-linear-refit
array tasks             array tasks
      |                   |
      +---------+---------+
                |
              afterok
                |
       1 artifact-only finalizer
```

The 30 training tasks are:

```text
2 families x 3 neural conditions x 5 seeds
```

The 10 Linear tasks are:

```text
2 families x 5 seeds
```

Each array task uses one CPU core. The training array is capped below the repository maximum of 50 concurrent CPU tasks. BLAS/OpenMP thread counts are fixed to one.

Each training task performs:

```text
train -> validation checkpoint selection -> evaluate the selected checkpoint -> write per-run artifacts
```

There is no separate second evaluation array.

## Slurm entry point

Submit the complete dependency graph with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_5_3_cpu.bash
```

## Durable artifacts

Final root:

```text
notebooks/artifacts/
  experiment_5_5_3_teacher_weight_routing_decomposition/
    teacher_weight_routing_decomposition_v1/
```

Per-seed artifacts include:

```text
teachers/
teacher_metadata/
frozen_evaluations/
checkpoints/
histories/
evaluations/
linear_models/
```

The artifact-only finalizer writes:

```text
teacher_equivalence.csv
runs.csv
summary.csv
paired_deltas.csv
recovery.csv
logit_distortion.csv
weight_drift.csv
manifest.json
```

The finalizer never trains, refits, or regenerates missing artifacts.

## Notebook policy

`notebooks/experiment_5_5_3_teacher_weight_routing_decomposition.ipynb` is analysis-only.

It reads only finalized CSV/JSON artifacts and produces:

1. the complete BA ladder for both teacher families;
2. paired hard-teacher -> soft-teacher changes;
3. soft teacher-init recovery;
4. hard/soft optimization comparisons;
5. hard -> soft logit distortion and margin diagnostics.

The notebook does not import the experiment module, create a model, fit a classifier, launch Slurm, or regenerate a missing run.

## Interpretation contract

The experiment is successful if it attributes the existing fusion gap, not only if balanced accuracy increases.

- `soft_teacher_frozen ~= hard_teacher_frozen`: routing softness is not the primary problem.
- `soft_teacher_frozen << hard_teacher_frozen`: soft interpolation conflicts with the proven phase-specific mapping.
- `soft_teacher_init_retrain ~= hard_teacher_frozen`: soft routing can work after soft-aware adaptation.
- `soft_teacher_init_retrain >> soft_random_train`: initialization / optimization is a major bottleneck.
- `hard_random_train << hard_teacher_frozen`: the neural training path cannot recover even the hard Linear solution.
- `soft_linear_refit >> soft_random_train`: the soft representation retains useful information, but the current AdamW path fails to exploit it.
- `soft_linear_refit << hard_teacher_frozen`: soft temporal mixing itself removes useful discriminative structure.

Only after this gate is resolved should the perfect/oracle temporal coordinate be replaced again by a learned GRU/RSNN WHEN source.
