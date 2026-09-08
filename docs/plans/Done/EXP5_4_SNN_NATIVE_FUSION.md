# Exp5.4 SNN-native fusion — frozen TaskSpec

Status: implemented

## Goal

Test whether the frozen Exp5.3.2.3 history-dependent WHEN spikes provide downstream letter-classification value by contextualizing frozen Local-SNN WHAT spikes in a small fixed-synapse Fusion SNN.

## Required behavior

- Use frozen 128D Local-SNN WHAT spikes.
- Use frozen Exp5.3.2.3 `ffsnn128_rsnn64` 64D WHEN spikes.
- Primary fusion is a 128-neuron short-memory, non-recurrent LIF layer with fixed trainable `W_what` and `W_when`.
- Use a bias-free fixed `W_out` from Fusion spikes to 12D class evidence.
- Sum evidence over valid timesteps only; add one class bias after accumulation; train with final whole-sequence CE only.
- Compare six paired conditions: WHAT-only, WHEN-only, WHAT+elapsed, WHAT+reset-WHEN, WHAT+WHEN linear, WHAT+WHEN Fusion LIF.
- Evaluate zero/shuffle/circular-shift WHEN ablations on the trained primary Fusion-LIF checkpoint.
- Use five seeds `(11, 23, 37, 53, 71)` and task-level Slurm arrays.

## Preserved contracts

- Final gesture duration is never a model or context input.
- Valid length only masks padding and identifies the endpoint.
- WHAT and WHEN source checkpoints remain frozen and are identity-validated.
- No temporal binning, flattening, attention, or recurrent classifier is introduced after WHAT/WHEN.
- Fusion has no recurrent weights; its short temporal state cannot replace the WHEN branch's long/order-dependent role.
- All six conditions use paired per-seed fusion initialization and minibatch order.
- Finalizer aggregates existing per-run artifacts only.

## Allowed write scope

Experiment 5.4 runner, README, analysis notebook, Slurm scripts, focused contract test, CI test list, and this completed TaskSpec.

## Acceptance criteria

- Exactly 30 unique train/evaluate run specs: six conditions by five seeds.
- Fusion-LIF and linear control have identical trainable parameter shapes and paired initialization.
- `W_out` is bias-free and class bias is added once after accumulation.
- Frozen-cache metadata validates Exp5.3.2.3 protocol/condition/seed identity.
- Padding changes cannot alter valid evidence accumulation.
- Slurm uses 5 preparation tasks, 30 one-core train/evaluate tasks, `afterok` dependencies, and an aggregation-only finalizer.
- Notebook is analysis-only and reads final CSV/JSON artifacts.
- Repository source syntax and focused Exp5.4 contract checks pass in CI.

## Replan triggers

Replan only if the frozen Exp5.3.2.3 source checkpoint interface changes, the source WHAT cache is no longer 128D binary L2 spikes, or the deployment requirement changes away from fixed-synapse SNN-native fusion.
