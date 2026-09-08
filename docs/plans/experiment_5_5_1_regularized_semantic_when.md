# Experiment 5.5.1 — Regularized semantic WHEN TaskSpec

## Goal

Test whether weak auxiliary supervision can organize the existing Exp5.5 ordered-GRU history representation into a useful semantic WHEN state without replacing semantic routing with progress gating.

## Required behavior

- Reuse the frozen Exp5.5 WHAT representation and existing Exp5.5 baselines.
- Preserve strict-history ordered routing: current `q_t` depends on `h_(t-1)` and therefore on prior WHAT only.
- Add a separate progress head from current `h_t`; predicted progress is used only in an auxiliary loss and must not enter q, expert weights, or logits.
- Screen three nested ordered-GRU regularization recipes across five seeds using validation only.
- Persist validation selection to `selection.json` before any new final test evaluation.
- Train a capacity-matched reset-GRU using the selected recipe.
- Run final ordered/reset test diagnostics only after selection and matched-reset artifacts exist.
- Aggregate existing artifacts in a finalizer; finalizer must not train, select, or regenerate missing runs.
- Keep the notebook analysis-only.

## Preserved contracts

- WHAT width 128, GRU width 64, K=8, 12 classes.
- Full independent `12 x 128` experts and one final 12-D class bias.
- Whole-gesture valid-timestep evidence sum.
- AdamW `lr=1e-3`, weight decay `1e-4`, grad clip 1.0, max 200 epochs, patience 25.
- Best checkpoint by validation balanced accuracy, tie-broken by validation classification CE.
- Seeds `(11, 23, 37, 53, 71)` and existing fixed split contract.
- One CPU core per independent Slurm task and thread environment variables fixed to one.

## Acceptance criteria

- Contract tests prove strict-history routing, padding invariance, auxiliary masking, and gradient routing.
- Progress-only loss has gradients to GRU/progress head but not directly to state head, experts, or class bias.
- Sticky/confidence losses have gradients to GRU/state head but not directly to experts, class bias, or progress head.
- Classification CE has gradients to GRU/state head/experts/class bias but not progress head.
- Screen tasks do not construct test loaders and record `test_evaluated=false`.
- `selection.json` records a validation-only recipe choice.
- Slurm dependency chain is source validation -> 15 ordered screen tasks -> selector -> 5 matched reset tasks -> 5 final test tasks -> finalizer.
- Notebook consumes finalized CSV/JSON artifacts only.

## Validation

Run repository source syntax checks and `tests/test_experiment_5_5_1_contract.py`; include the new contract test in GitHub CI.
