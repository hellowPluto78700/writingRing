# Experiment 17 — Persistent-pathway story closure

## Purpose

Exp17 is a closure experiment, not a new modeling branch. It tests the three remaining arrows in the current mechanism story:

1. whether optimization progressively increases reliance on persistent L2 dimensions;
2. whether persistent dimensions simultaneously carry transferable class information and user/context information;
3. whether the prune-and-retrain effect is specific to persistence-ranked dimensions rather than generic dimensionality reduction.

The experiment adds no gate, auxiliary loss, new tau sweep, deeper layer, or new task objective.

## Evidence inherited from earlier experiments

Exp17 assumes the already-finalized evidence chain:

- CoreBenchmark / Exp13: L2 strongly depends on history and internalizes part of explicit temporal order;
- Exp13.1: more historical information is not equivalent to more transferable information;
- Exp14/14.1: cross-user/phase representation supervision alone does not reliably improve native accumulation;
- Exp15/15.1: a learnable write gate is not naturally driven toward selective writing by WCCE;
- Exp16.2: high-occupancy dimensions are used by the current WholeCount readout, but much of the loss after removal is recoverable by readout retraining;
- Exp16.3: reducing repeated-spike reward does not suppress persistence and can increase sustained firing while harming classification.

Exp17 therefore does not retest memory capacity or whether persistence exists. It tests how the persistent pathway emerges, what information it contains, and whether its apparent pruning effect is persistence-specific.

# 17-A — Epoch-wise emergence and reliance trajectory

## Question

Does standard WCCE training progressively make the persistent pathway more dominant, and if so, is dominance caused by more persistent neurons, stronger readout reliance, or both?

## Training contract

- exact two-layer 128-neuron CoreBenchmark backbone;
- shifts `((2,3,4),(2,3,4))`;
- seeds 11, 23, 37;
- exact LIN/WCCE readout used by Exp16.3;
- exact Exp16.3 shared initialization and epoch-specific deterministic sampler;
- validation BA, then validation CE, then earliest epoch selects the best checkpoint;
- standard early stopping is preserved;
- test is never read during training or checkpoint selection.

Snapshots are saved at epoch 0, every 5 epochs, the selected-best epoch, and the stopped epoch.

## Snapshot diagnostics

Each snapshot records:

- train/val/test native BA, accuracy and macro-F1;
- L1/L2 mean occupancy, P90 neuron occupancy, fraction of neurons with mean occupancy >0.5 and >0.8;
- mean and P90 neuron longest firing run;
- L2 class eta-squared and user eta-squared per neuron;
- high-persistence top quartile versus low-persistence bottom quartile eta-squared summaries;
- current native-head L2 weight norm per neuron;
- fraction of total head-weight norm allocated to the top-persistence quartile;
- top-30%-persistence frozen ablation on the current native head;
- top-30%-persistence frozen WholeCount-probe ablation;
- top-30%-persistence removal followed by a newly trained bias-free WholeCount probe.

Persistence ranking always uses **training-split mean occupancy only**.

## What would support the optimization-shortcut story?

The strongest pattern would be increasing persistent-pathway reliance, for example:

`persistent top-quartile head-weight share ↑`

and/or

`fraction of persistent neurons ↑`

while train BA continues improving and test BA plateaus. Increasing user eta-squared specifically in the high-persistence quartile would further support increasing contextual entanglement.

The experiment deliberately distinguishes two mechanisms:

- more neurons become persistent;
- the same persistent neurons become more heavily exploited by the readout.

# 17-B — Persistent-subset information decomposition

## Question

Are high-persistence neurons simply nuisance neurons, simply class neurons, or a mixed representation?

This block is **artifact-only** and reads finalized Exp16.2 L2 spike traces for all eight cases and seeds 11/23/37. No SNN is retrained.

## Dimension-matched subsets

For each case/seed, rank L2 neurons by training mean occupancy and construct:

- `high30`: top 30%;
- `low30`: bottom 30%;
- `random30`: five deterministic random 30% subsets.

Every subset has the same number of dimensions.

## Class decoding

Train the standard bias-free WholeCount probe on each subset. C is selected only on validation data. Report train/val/test BA and train-test gap.

A non-chance `high30` class decoder establishes that persistent dimensions contain class information rather than being pure nuisance.

## User decoding with class control

User identity is evaluated only within the Core training users, because validation/test users are disjoint identities.

Use deterministic five-fold user-stratified CV. For every test fold:

- one distinct fold is validation;
- the remaining three folds are training;
- C is selected using only the validation fold.

Two feature modes are reported:

- `raw`;
- `class_residualized`: subtract each class centroid estimated **only from the CV training fold** before user decoding.

The class-residualized mode asks whether user information remains after first-order class structure has been removed. It avoids requiring dense user-by-class cells, which the dataset does not provide.

The mixed-representation story is supported if `high30` remains class-decodable while retaining stronger class-residualized user decoding than `low30` and dimension-matched random controls.

# 17-C — Pruning specificity controls

## Question

When pruning and retraining improves or preserves OOD performance, is that specifically associated with high-persistence dimensions, or is it generic feature-count regularization?

This block is also artifact-only and uses the same Exp16.2 traces.

For removal fractions 10%, 20%, and 30%, compare:

- `high_occupancy`: highest training mean occupancy;
- `low_occupancy`: lowest training mean occupancy;
- `high_readout_weight`: largest standardized bias-free WholeCount coefficient norm;
- `random`: 20 deterministic dimension-matched random controls.

For every mask evaluate:

1. `frozen_mean_replacement`: keep the fitted full WholeCount probe frozen and replace removed dimensions with their training mean;
2. `retrained_without_neurons`: remove the dimensions and fit a new bias-free WholeCount probe.

The important comparison is not merely whether high-persistence pruning can improve test BA. It is whether its retrained behavior differs systematically from the matched random-removal distribution.

# Hard anti-leakage contracts

- all persistence rankings use train data only;
- all readout-weight rankings use a probe fitted with train and validation only;
- random masks depend only on case, model seed, replicate, and requested dimension;
- class-probe C selection uses validation only;
- user-probe class centroids are estimated from the CV training fold only;
- user-probe C selection uses the CV validation fold only;
- test labels never affect SNN training, checkpoint selection, ranking, subset selection, or C selection;
- B/C never retrain or modify SNN checkpoints.

# Slurm layout

One CPU per independent task:

- prepare: 1 job;
- 17-A trajectory: 3 tasks, one per seed;
- 17-B subset decomposition: 24 tasks, one per Exp16.2 case/seed;
- 17-C pruning controls: 24 tasks, one per Exp16.2 case/seed;
- finalizer: aggregate-only after all tasks succeed.

Submit from the Unity repository root:

```bash
git pull origin main
bash scripts/bash_script/SNN_Bash/submit_exp_17_cpu.bash
```

Artifacts are written to:

`notebooks/artifacts/experiment_17_persistent_pathway_story/persistent_pathway_story_v1/`

The `aggregate/` directory contains the final trajectory, subset-decoding, pruning-control, and story-summary tables.

# Stop criterion for this story

The diagnosis story is considered experimentally closed if the evidence jointly establishes:

1. persistent-pathway reliance emerges or increases during ordinary WCCE optimization;
2. high-persistence dimensions carry both class information and class-controlled user/context information;
3. high-persistence pruning/retraining behaves differently from dimension-matched random pruning.

If one of these conditions fails, the corresponding causal sentence must be weakened rather than adding unrelated experiments.