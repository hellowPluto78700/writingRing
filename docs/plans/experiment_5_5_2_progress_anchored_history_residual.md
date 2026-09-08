# Experiment 5.5.2 — Progress-Anchored History-Residual Routing

## Goal

Determine whether past WHAT history provides class-relevant routing information beyond a strong temporal coordinate and beyond current/previous local WHAT, while preserving the already-trained Exp5.5 Clock/Oracle expert banks.

## Required behavior

### Frozen source anchors

For each stochastic seed `(11, 23, 37, 53, 71)`, reuse the matching Exp5.5 source checkpoint independently for:

- `clock`
- `oracle_progress`

Freeze both the eight full expert matrices `W_k in R^(12 x 128)` and the 12-D class bias. Never mix a Clock routing source with an Oracle expert bank or vice versa.

The new model must expose the same pre-softmax RBF routing logits used by Exp5.5. For anchor logits `a_t^A`, residual routing is

`q_t = softmax(a_t^A + delta_z_t)`

with

`delta_z_t = alpha * tanh(R h_t*)`,

so a zero residual projection reproduces the source anchor exactly rather than approximately reconstructing logits from `log(q + eps)`.

### Residual controls

Screen four residual modes:

1. `coordinate`: only the anchor coordinate; tiny `1 -> 64 -> 8` MLP.
2. `reset`: current `m_t` only through a reset GRU64.
3. `lag1`: previous `m_(t-1)` only through a reset GRU64; no recurrent state.
4. `ordered`: accumulated past through GRU64, but current routing uses `h_(t-1)` so current `m_t` cannot change the routing used to interpret itself.

`lag1` and `ordered` must have zero residual at `t=0` because no past WHAT exists yet.

All residual heads are zero-initialized. Residual amplitude is structurally bounded by `|delta_z_t,k| <= alpha`.

### Objective and frozen contracts

Use whole-gesture classification CE only. Do not add progress, confidence, sticky-state, balance, diversity, or expert-training losses. Preserve Exp5.5 optimizer settings: AdamW, `lr=1e-3`, `weight_decay=1e-4`, gradient clipping at 1, up to 200 epochs with patience 25, and best checkpoint by validation balanced accuracy with validation CE as tie-breaker.

### Source equivalence gate

Before any screen run for a seed, validate both source anchors on validation data only. With a zero residual projection, require:

- max absolute q difference <= `1e-6`;
- max absolute gesture-logit difference <= `1e-6`;
- identical predictions.

Failure is fatal. Test is not loaded for this gate.

### Alpha selection

Screen `alpha in {0.25, 0.5, 1.0}` on Clock only. Each residual mode selects its own alpha using validation only.

For each mode/alpha, compute paired validation BA deltas against the matching Exp5.5 Clock seed. An alpha is eligible when:

- mean paired delta > 0; and
- at least 4/5 seed deltas are nonnegative.

Within the eligible pool, choose the smallest alpha whose mean validation BA is within one SEM of the best mean validation BA. If none is eligible, use the same one-SEM/smallest-alpha fallback over all three alphas and mark rescue unsupported for that mode.

Oracle and test must not participate in alpha selection.

### Final protocol

After `selection.json` is fixed:

- Clock final jobs reuse the selected screen checkpoints and do not retrain them.
- Oracle final jobs train each residual mode with that mode's Clock-selected alpha, then evaluate the selected checkpoint.
- Final evaluation opens train/val/test only after selection is fixed.

The central contrasts are:

- `ordered - anchor`
- `ordered - reset`
- `ordered - coordinate`
- `ordered - lag1`

The first asks whether history residual helps the anchor; the next two exclude current-WHAT and flexible-coordinate explanations; `ordered - lag1` tests whether accumulated history provides information beyond only the previous local state.

### Ordered diagnostics

For each selected Ordered model, save:

- residual magnitude and saturation;
- `L1(q - q_anchor)` and `KL(q || q_anchor)`;
- anchor routing argmax-change fraction;
- relative effective-weight correction and anchor/residual effective-weight cosine;
- relative evidence correction;
- residual-only train-fit LogisticRegression probes from mean and final `delta_z`;
- five within-gesture circular-shift residual replicates (primary temporal-alignment control);
- five within-gesture random-permutation residual replicates (secondary aggressive control).

The alignment ablations must keep anchor logits and the residual value distribution fixed and only break the temporal alignment of `delta_z` with current WHAT.

For Clock Ordered, report whether the routing center-of-mass proxy moves closer to true offline relative progress. This is diagnostic only and is never used for training or selection.

### Same-WHAT / same-coordinate diagnostic

For selected Ordered models, search test timesteps for pairs satisfying:

- WHAT cosine similarity >= 0.95;
- WHAT norm ratio <= 1.10;
- different samples and different labels;
- Oracle relative-progress difference <= 0.05, or Clock elapsed-time difference <= 0.125 s.

Report residual/q/effective-weight differences, top-class support, and anchor-to-residual correct-class margin changes. This diagnostic characterizes whether similar local WHAT at similar anchor coordinates can be interpreted differently under different past histories.

### Generalization reporting

For each anchor/mode/seed, report:

`generalization_gap = (BA_val_residual - BA_val_anchor) - (BA_test_residual - BA_test_anchor)`.

The five seeds share one fixed user-disjoint split; they are paired stochastic replicates, not five independent user-split experiments.

## Multi-CPU execution contract

Use the repository one-core task-level Slurm pattern:

```text
5 source-validation tasks
        -> afterok
60 Clock screen tasks
   4 modes x 3 alphas x 5 seeds
   max 50 concurrent
        -> afterok
1 validation-only selector
        -> afterok
20 Clock final-evaluation tasks       20 Oracle train->evaluate tasks
   4 modes x 5 seeds                     4 modes x 5 seeds
        \                                 /
         ----------- afterok ------------
                      |
             1 artifact-only finalizer
                      |
              analysis-only notebook
```

Every compute task requests one CPU core, initializes Conda locally, prefers `writingring-gpu` with `writingring-viz` fallback, disables CUDA, and sets OpenMP/BLAS/NumExpr threads to one. The finalizer aggregates existing artifacts only and fails on missing runs.

## Artifact contract

Root:

`notebooks/artifacts/experiment_5_5_2_progress_anchored_history_residual/progress_anchored_history_residual_v1/`

Durable outputs include:

- `source_references/`
- `screen_checkpoints/`, `screen_histories/`, `screen_evaluations/`
- `screen_runs.csv`, `screen_summary.csv`, `screen_paired_deltas.csv`
- `selection.json`
- `final_checkpoints/`, `final_histories/`, `final_evaluations/`
- `trajectories/`, `same_what/`
- `final_runs.csv`, `final_summary.csv`, `final_paired_deltas.csv`
- `residual_diagnostics.csv`
- `residual_alignment.csv`
- `residual_only_probes.csv`
- `generalization_gap.csv`
- `clock_progress_alignment.csv`
- `same_what_same_coordinate.csv`
- `manifest.json`

The experiment notebook is analysis-only: it reads finalized CSV/JSON artifacts, visualizes and summarizes them, and never trains, selects alpha, submits Slurm jobs, or regenerates missing artifacts.

## Acceptance criteria

1. Frozen source expert banks/class biases cannot receive gradients.
2. Clock/Oracle anchor routing formulas are source-equivalent at zero residual.
3. Ordered current routing is unchanged when only current `m_t` changes; later routing may change.
4. Reset current routing is independent of earlier timesteps.
5. Lag1 current routing depends on `m_(t-1)` but not `m_t` or older history.
6. Ordered and Lag1 residuals are zero at `t=0`.
7. Padding suffix changes cannot change logits.
8. `|delta_z| <= alpha`.
9. Screen/source stages do not construct test loaders.
10. Oracle/test do not influence alpha selection.
11. Clock selected checkpoints are reused, not retrained, in final evaluation.
12. Finalizer is artifact-only.
13. Slurm mapping is exactly 5 source + 60 screen + 1 selector + 20 Clock final + 20 Oracle final + 1 finalizer with <=50 concurrent experiment tasks.
14. Notebook is analysis-only and depends on finalized artifacts.
15. Repository syntax test and focused Exp5.5.2 contract test are included in CI.
