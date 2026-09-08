# Exp5.4.1 constrained conjunction residual — frozen TaskSpec

Status: implemented

## Goal

Test whether the frozen Exp5.3.2.3 WHEN spikes provide **incremental** letter-classification value when they are prevented from acting as an independent classifier and may only add a residual correction through a conjunction with the already learned Exp5.4 WHAT-only evidence spikes.

## Context

Exp5.4 showed that WHAT/WHEN temporal alignment matters inside the trained Fusion-LIF model, but the jointly trained Fusion-LIF did not consistently outperform WHAT-only, elapsed-time, reset-WHEN, or the parameter-matched linear control. WHEN-only also retained substantial letter identity. Exp5.4.1 therefore constrains information routing rather than adding capacity.

## Required behavior

- Reuse the frozen Exp5.4 `what_only_lif` checkpoint for each seed as the base classifier.
- Reuse the frozen Exp5.4 fusion cache, including 128D WHAT spikes and 64D ordered/reset WHEN spikes.
- The base WHAT path and its 128D Fusion spike code are frozen.
- A new residual context path may read **only** the frozen base Fusion spikes and frozen WHEN spikes; it must not read raw 128D WHAT spikes directly.
- The residual path must contain a structural conjunction such that zeroing either the base Fusion spikes or WHEN spikes produces zero conjunction spikes and zero residual class evidence.
- The residual output projection is zero-initialized so epoch 0 is exactly the frozen WHAT-only classifier.
- Epoch 0 is a legal checkpoint candidate; training may keep the base checkpoint if validation balanced accuracy does not improve.
- Final logits are `base_logits + residual_evidence_sum`; no second class bias is introduced.
- Final duration is never a model/context input. Valid length only masks padded timesteps and selects the endpoint.
- Evaluate the trained residual model with ordered WHEN, reset-WHEN, zero WHEN, shuffled WHEN, circularly shifted WHEN, and zero base-Fusion-spike context ablations.

## Preserved contracts

- Seeds remain `(11, 23, 37, 53, 71)` with the same user-disjoint split.
- WHAT and WHEN source models/caches are frozen and identity-validated.
- No temporal binning, flattening, attention, or new recurrent classifier is introduced.
- The context/conjunction path has no temporal recurrence or long-memory state; history can enter only through frozen WHEN spikes.
- Base WHAT classification remains available unchanged when the residual correction is zero.
- Finalizer aggregates existing per-run artifacts only.

## Allowed write scope

Exp5.4.1 runner, README, analysis-only notebook, Slurm scripts, focused contract test, CI test list, and this TaskSpec.

## Acceptance criteria

- Exactly five train/evaluate tasks, one per seed.
- All parameters of the reused Exp5.4 WHAT-only base model have `requires_grad=False`.
- Residual output weights are exactly zero at initialization and initial logits equal frozen base logits.
- With WHEN set to zero, conjunction spikes and residual logits are exactly zero.
- With base Fusion spikes set to zero, conjunction spikes and residual logits are exactly zero.
- Epoch 0 baseline is included in checkpoint selection.
- Slurm uses five one-core source-validation tasks, five one-core train/evaluate tasks, `afterok` dependencies, and an aggregation-only finalizer.
- Notebook is analysis-only and reports paired deltas vs the Exp5.4 WHAT-only baseline plus alignment/history ablations.
- Repository source syntax and focused Exp5.4.1 contracts pass in CI.

## Replan triggers

Replan only if the committed Exp5.4 WHAT-only checkpoint/cache identity changes, the base Fusion spike width is no longer 128, the selected WHEN spike width is no longer 64, or the deployment requirement changes away from fixed-synapse spike-domain contextual correction.
