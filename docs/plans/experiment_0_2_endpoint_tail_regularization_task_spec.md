# Exp0.2 Endpoint Tail Regularization — TaskSpec

## Goal

Implement Exp0.2 to compare classification objectives and hidden-state regularization mechanisms for the Exp0.1 binary direct-SNN backbones while explicitly measuring and penalizing endpoint-aligned post-gesture firing tails.

## Required behavior

- Reuse the Exp0.1 direct binary SNN architecture, data split, optimizer settings, and deployment WholeCount readout.
- Backbones: `short_mid`, `mid_long`, `short_mid_long`.
- Objectives: `whole_count_ce`, `timestep_ce`.
- New regularization profiles: `a1`, `a1_p2`, `tail`, `a1_tail`.
- Seeds: `11`, `23`, `37` on the single existing user-disjoint split (`12345`).
- Train every new run for exactly 50 epochs; checkpoint selection is validation WholeCount balanced accuracy with WholeCount CE as tie-break.
- Reuse the matching Exp0.1 binary checkpoints for the `none` profile instead of retraining them.
- A1 and all-fixes P2 must preserve the Exp0.1.5 definitions. P2 is used only in the `a1_p2` profile.
- Endpoint-tail loss uses a fixed endpoint-relative horizon of approximately 600 ms, split at approximately 200/400/600 ms, with stage weights `(1, 2, 4)` normalized by their sum.
- Post-end input is explicitly forced to zero before the extra rollout. Classification losses/readout use only timesteps `< valid_length`; tail loss uses only endpoint-relative post-end timesteps.
- Tail regularization acts on all hidden layers, normalized within each layer and then averaged across hidden layers. The output layer is diagnostic only and is not regularized.
- Every regularized profile uses one-time hidden-gradient calibration to a target ratio of 5% over five deterministic train-only calibration batches, followed by the existing 10-epoch linear warmup.
- Evaluation for every new and frozen checkpoint must include train/val/test classification metrics and endpoint-tail diagnostics.
- Every evaluated checkpoint must save a fixed-test-sample raster image containing the final hidden layer and output layer over the valid segment plus the endpoint-relative tail horizon. Final-hidden neurons are grouped by their configured synaptic shift.

## Preserved contracts

- Input channel count and semantics are unchanged from Exp0.1.
- Binary hidden and output spike caps remain 1.
- Exp0.1 forward dynamics are unchanged: no `(1-alpha)` normalization is added to the actual synaptic recurrence.
- WholeCount deployment ignores every post-end rollout timestep.
- Exp0.1 frozen checkpoints are never modified.
- Existing Exp0.1/Exp0.1.5 artifacts are read-only dependencies.

## Execution contract

- 72 new runs = 3 backbones x 2 objectives x 4 regularization profiles x 3 seeds.
- One Slurm array task per new run, one CPU core per task, array concurrency capped at 50.
- Each array task performs train -> select/save best checkpoint -> evaluate -> save per-run artifacts.
- 18 frozen Exp0.1 evaluations = 3 backbones x 2 objectives x 3 seeds, executed by one separate single-CPU job.
- Finalizer runs only after both the new-run array and frozen-evaluation job succeed, and only aggregates existing artifacts.
- Notebook is analysis-only and reads finalized artifacts.

## Acceptance criteria

- Run-spec keys are unique and enumerate exactly 72 trainable runs and 18 frozen evaluations.
- Endpoint rollout zeroes each sample from its own valid endpoint onward and provides at least the configured fixed tail horizon for every sample.
- Tail loss is invariant to unrelated remaining padding length and excludes valid-region spikes.
- Frozen Exp0.1 evaluations use the matching architecture/objective/seed binary checkpoint.
- Finalized artifacts include per-run summary, seed-aggregated comparison, paired deltas versus frozen Exp0.1, calibration summary, concatenated histories, shift-resolved tail diagnostics, raster index, and manifest.
- Slurm scripts follow repository Conda and one-core thread controls.
- Focused contract tests cover run mapping, rollout/tail masks, regularizer composition, frozen reuse, aggregation-only finalizer, Slurm layout, and analysis-only notebook.

## Validation

- `python -m pytest -q tests/test_experiment_0_2_endpoint_tail_regularization_contract.py`
- `python -m pytest -q tests/test_repository_source_syntax.py`

## Replan triggers

Replan only if the Exp0.1 checkpoint schema, Exp0.1 data split contract, or all-fixes P2/A1 interface has changed incompatibly on the target branch.