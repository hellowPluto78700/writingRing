# Exp5.4.1 direct WHAT + constrained conjunction residual — frozen TaskSpec

Status: implemented

## Goal

Test whether frozen Exp5.3.2.3 WHEN spikes provide **incremental** letter-classification value on top of the original frozen WHAT L2 spike representation when WHEN is structurally prevented from becoming an independent classifier.

## Context

Exp5.4 showed WHAT/WHEN alignment sensitivity but no consistent final-BA win over simpler controls. The first Exp5.4.1 draft reused Exp5.4 `what_only_lif`, inserting an additional 128-neuron Fusion-LIF before the base readout. That extra learned WHAT transform is removed so the experiment now operates directly on the canonical WHAT branch output.

## Required behavior

- Reuse the frozen Exp5.4 fusion cache only as a source of 128D WHAT L2 spikes and 64D ordered/reset WHEN spikes.
- Do **not** reuse the Exp5.4 `what_only_lif` classifier or its Fusion spike representation as the base.
- Stage A trains a direct WHAT readout: bias-free `Linear(128, 12)` per timestep, valid-time evidence sum, and one final class bias.
- Stage A uses final whole-sequence CE and validation-BA checkpoint selection.
- Stage B freezes every Stage-A base parameter.
- Stage B residual path reads the same original 128D WHAT spikes and frozen 64D WHEN spikes.
- The residual path must contain a structural conjunction such that zeroing either residual WHAT input or WHEN input produces zero conjunction spikes and zero residual class evidence.
- The residual output projection is exactly zero-initialized, so Stage-B epoch 0 exactly equals the frozen direct WHAT base.
- Stage-B epoch 0 is a legal checkpoint candidate.
- Final logits are `direct_what_base_logits + residual_evidence_sum`; no second class bias is introduced.
- Final duration is never a model/context input. Valid length only masks padded timesteps and selects the endpoint.
- Evaluate ordered WHEN, reset-WHEN, zero WHEN, shuffled WHEN, circularly shifted WHEN, and residual-WHAT-zero ablations.

## Preserved contracts

- Seeds remain `(11, 23, 37, 53, 71)` with the same user-disjoint split.
- WHAT and WHEN source networks/caches are frozen and identity-validated.
- The canonical WHAT representation is the 128D binary Local-SNN L2 spike trajectory.
- There is no additional Fusion-LIF, recurrent classifier, temporal binning, flattening, attention, or phase oracle between WHAT and the direct base readout.
- The residual conjunction path has no temporal recurrence, synaptic state, or membrane memory across timesteps; history can enter only through frozen WHEN spikes.
- The direct WHAT base remains available unchanged whenever the residual correction is zero.
- Finalizer aggregates existing per-run artifacts only.

## Allowed write scope

Exp5.4.1 runner, README, analysis-only notebook, Slurm scripts, focused contract test, CI test list if needed, and this TaskSpec.

## Acceptance criteria

- Exactly five input-preparation tasks, five direct-WHAT-base training tasks, five residual train/evaluate tasks, and one aggregation-only finalizer.
- Direct base architecture is exactly `WHAT128 -> Linear12(bias=False) -> valid-time sum -> one class bias`.
- No Exp5.4 `FusionClassifier` is instantiated as the base classifier.
- All trained direct-base parameters have `requires_grad=False` during residual training.
- Residual output weights are exactly zero at initialization and initial logits equal frozen direct-base logits.
- With WHEN set to zero, conjunction spikes and residual logits are exactly zero.
- With residual WHAT input set to zero, conjunction spikes and residual logits are exactly zero.
- Residual epoch 0 is included in checkpoint selection.
- Slurm uses one CPU per array task and `afterok` dependencies in the order prep -> base -> residual -> finalizer.
- Notebook is analysis-only and reports paired deltas vs the direct WHAT base plus alignment/history ablations.
- Repository source syntax and focused Exp5.4.1 contracts pass in CI.

## Replan triggers

Replan only if the canonical WHAT L2 width is no longer 128, the selected WHEN spike width is no longer 64, the Exp5.4 fusion-cache identity changes incompatibly, or the deployment requirement changes away from fixed-synapse spike-domain contextual correction.
