# Exp7.3 Training Strategy Decomposition

## Goal

Determine whether the best final LIF classifier is obtained by joint E2E training or by decoupling hidden-representation learning from final projection learning, while holding the final deployment interface fixed.

## Required behavior

- Use the `234x234` two-hidden-layer SNN backbone with task-only training and seeds 11/23/37.
- Implement four E2E methods crossing Linear/LIF training readout with TSCE/WCCE.
- Reuse the Linear-TSCE and Linear-WCCE E2E checkpoints as the two Stage-1 representation sources.
- For two-stage methods, discard the Stage-1 W, freeze L1/L2, cache L2, and train a fresh W under the 2x2 crossing of Linear/LIF training readout and TSCE/WCCE.
- Use the same fresh W initialization and training order across all Stage-2 conditions within a seed.
- Re-evaluate every method with one common final LIF(beta=0.5, cap=1) spike-count implementation.
- Preserve same-W Analog evaluation and frozen-L2 WholeCount/Fixed250 probes as diagnostics.
- Use native validation BA for checkpoint selection, with native objective loss as tiebreak.
- Finalize only after all required E2E, cache, and Stage-2 tasks exist; missing artifacts are errors.

## Preserved contracts

- Input preprocessing/data split/channel semantics are inherited unchanged from the current Exp7.2 pipeline.
- Hidden synaptic/membrane dynamics and shift assignments match the repository's `234x234` implementation.
- Hidden/output matrices are bias-free.
- CE gain is fixed to 1 for the full Exp7.3 main matrix.
- Test data never participates in model or hyperparameter selection.
- Finalizer aggregates existing artifacts only.
- Notebook is analysis-only.

## Execution graph

```text
12 E2E array tasks
    -> 6 frozen-L2 cache/probe tasks sourced only from A1/A2
        -> 24 Stage-2 W-training array tasks
            -> one finalizer
```

One CPU core is used per array task. Array concurrency remains below the repository cap of 50.

## Acceptance criteria

- `e2e_specs()` contains exactly 12 tasks.
- `backbone_specs()` contains exactly 6 cache tasks.
- `stage2_specs()` contains exactly 24 tasks.
- Finalized `method_runs.csv` contains exactly 36 method/seed rows representing 12 distinct methods and 3 seeds each.
- Every row contains same-W Analog BA, common final-LIF BA, and both L2 probe BAs.
- Slurm dependencies enforce E2E -> cache -> Stage2 -> finalizer.
- Aggregate notebook reads final CSV/JSON artifacts only.
- Repository source syntax and the Exp7.3 focused contract test pass in CI.

## Replan triggers

Replan only if the canonical Exp7.2 `234x234` dynamics, label encoding, or final LIF threshold/cap contract materially changes before execution.
