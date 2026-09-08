# Experiment 5.5.1.1 — Routing diagnostic extension TaskSpec

## Goal

Diagnose whether Exp5.5.1 is limited primarily by diffuse semantic routing or by insufficient specialization of the eight full expert matrices, without retraining or selecting a new model.

## Required behavior

- Reuse the finalized Exp5.5.1 selected Ordered-GRU checkpoint for each seed `(11, 23, 37, 53, 71)`.
- Do not train, fine-tune, backpropagate, or modify Exp5.5.1 checkpoints.
- Quantify q confidence using max probability, top1-top2 margin, entropy, effective state count, and confidence-threshold fractions.
- Apply post-hoc routing transforms at temperatures `(1.0, 0.8, 0.6, 0.4, 0.25)` plus hard top-1 routing.
- Report classification metrics for every transform on validation and test, but do not choose a temperature in this experiment.
- For every transform, shuffle q within each valid gesture for five deterministic replicates and report the loss of WHAT/q temporal alignment.
- Quantify expert specialization using pairwise full-weight cosine/L2 distance and functional class-evidence diversity on active WHAT vectors.
- Verify that temperature 1.0 reproduces the finalized Exp5.5.1 Ordered test balanced accuracy for every seed.
- Run one independent diagnostic task per seed and aggregate only existing per-seed artifacts in the finalizer.
- Keep the notebook analysis-only.

## Preserved contracts

- Source model is Exp5.5.1 `regularized_semantic_when_v1`; no source checkpoint is rewritten.
- WHAT width 128, GRU width 64, K=8, 12 classes, strict-history Ordered routing, and full `12 x 128` experts remain unchanged.
- Padding timesteps do not participate in q transforms, shuffles, confidence statistics, or evidence aggregation.
- Post-hoc temperature/hard routing is diagnostic only and does not become a selected test condition.
- One CPU core per independent Slurm task; BLAS/OpenMP thread counts fixed to one.

## Acceptance criteria

- Temperature 1.0 preserves q and reproduces source test balanced accuracy for every seed.
- Lower temperature increases q sharpness on a controlled synthetic q fixture; hard routing is valid one-hot on valid timesteps and zero on padding.
- Expert-weight diagnostics return all 28 unordered expert pairs for K=8.
- Identical expert weights produce zero functional evidence difference in a focused test fixture.
- Per-seed diagnostic runner contains no optimizer, training loop, or backward call.
- Slurm execution is `5 seed tasks -> afterok -> one artifact-only finalizer` with one CPU per task.
- Finalizer does not run diagnostics or regenerate missing per-seed artifacts.
- Notebook reads finalized artifacts only and never launches Slurm or diagnostic execution.

## Validation

Run:

```bash
python -m pytest -q tests/test_repository_source_syntax.py
python -m pytest -q tests/test_experiment_5_5_1_1_contract.py
```

Include the focused contract test in GitHub CI.
