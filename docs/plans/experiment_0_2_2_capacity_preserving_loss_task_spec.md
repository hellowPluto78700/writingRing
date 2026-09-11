# Exp0.2.2 Capacity-Preserving Anti-Persistent-Firing Loss — TaskSpec

## Goal

Test whether long-timescale final-hidden neurons can avoid sustained/persistent firing without being globally silenced, while preserving WholeCount classification performance.

## Required behavior

- Train only `mid_long` and `short_mid_long` direct binary SNN backbones from Exp0.2.
- Use `whole_count_ce` as the only task objective.
- Use seeds `11, 23, 37` and the existing user-disjoint split seed `12345`.
- Train every run for exactly 50 epochs.
- Use five conditions:
  - `wc_only`
  - `sat`
  - `relative_tail`
  - `sat_relative_tail`
  - `sat_relative_tail_capacity`
- Epochs 1-5 are a shared WC-only warmup for each `(backbone, seed)` pair. Save model, optimizer and RNG state at epoch 5 and fork all five conditions from that exact state.
- Regularization applies only to the final hidden layer's `s6` and `s7` neurons.
- Saturation loss penalizes only excessive 250-ms causal firing occupancy above a train-derived healthy-reference threshold.
- Relative-tail loss penalizes endpoint-relative tail/valid firing-rate ratios for the 0-200, 200-400 and 400-600 ms stages, using stage weights `(1, 2, 4)` and train-derived healthy-reference targets.
- Capacity-floor loss is used only in `sat_relative_tail_capacity`; it prevents each long-timescale population's valid-region firing rate from falling below 70% of its own shared epoch-5 warmup reference.
- Healthy-reference saturation and relative-tail targets are extracted only from the train split using the existing strong `short_mid + whole_count_ce + none` reference checkpoint; no test data may influence these targets.
- At the end of epoch 5, calibrate each non-baseline condition's regularizer once so its gradient norm on final-hidden incoming weights for s6/s7 is 5% of the WholeCount task-gradient norm over five deterministic train-only calibration batches. Fail loudly or record an invalid calibration if the regularizer gradient is effectively zero; do not silently create an unbounded scale.
- Regularizer scale is zero for epochs 1-5, ramps linearly over epochs 6-15, and is fully enabled for epochs 16-50.
- Checkpoint selection uses highest validation WholeCount balanced accuracy, tie-broken by lower validation WholeCount CE.
- Test evaluation occurs only once after training, using the selected validation checkpoint.

## Per-epoch history and diagnostics

After every training epoch, perform deterministic no-shuffle train and validation evaluation and record:

- `train_balanced_accuracy`
- `val_balanced_accuracy`
- `train_task_ce`
- `val_task_ce`
- minibatch-mean optimization components:
  - `train_optim_task_loss`
  - `train_optim_raw_reg_loss`
  - `train_optim_weighted_reg_loss`
  - `train_optim_total_loss`
  - raw component losses `sat`, `relative_tail`, `capacity` when applicable
- regularizer warmup scale and calibrated kappa
- train and validation s6+s7 valid firing rate, tail firing rate and tail/valid ratio
- output-layer utilization by final-hidden shift, including per-shift mean L2 outgoing-weight norm and the fraction of output-weight energy assigned to s6+s7.

## Plots

For every `(backbone, condition, seed)` run, save one 2x3 training diagnostic figure with:

1. Train BA and Val BA versus epoch, with epoch-5 calibration/fork boundary, epoch-15 full-regularizer boundary and selected best epoch marked.
2. Train WholeCount CE and Val WholeCount CE versus epoch.
3. Train optimization task loss, weighted regularizer contribution and total loss on the same axes. For composite conditions, component weighted regularizers may be shown as secondary thin traces.
4. Long-population valid and tail firing rates versus epoch.
5. Long-population tail/valid firing-rate ratio versus epoch.
6. Output-weight utilization versus epoch, including s6/s7 outgoing-weight norms and long-population output-weight energy fraction.

The notebook is analysis-only and reads finalized CSV/JSON/PNG artifacts.

## Preserved contracts

- Reuse Exp0.2/Exp0.1 direct-SNN forward dynamics unchanged. Do not add `(1-alpha)` normalization, recurrence, reset/gating, pooling, or architectural changes.
- Keep binary hidden/output spike caps at 1.
- Classification readout always uses valid WholeCount only; endpoint tail timesteps never enter classification loss or deployment readout.
- Tail rollout explicitly zeroes input at each sample's valid endpoint.
- Existing Exp0.1/Exp0.2 checkpoints and artifacts are read-only dependencies.
- Test split must not be evaluated during epoch-by-epoch training or reference-target extraction.

## Execution contract

- Healthy-reference extraction: one single-CPU preprocessing job.
- Shared warmup: 6 Slurm array tasks = 2 backbones x 3 seeds, one CPU each, capped at 6 concurrent tasks.
- Main training: 30 Slurm array tasks = 2 backbones x 5 conditions x 3 seeds, one CPU each, capped at 30 concurrent tasks. Each task loads the matching epoch-5 warmup checkpoint, trains epochs 6-50, selects/evaluates its own best checkpoint, saves per-run artifacts and the requested training-curve figure.
- Finalizer runs only after reference extraction, warmup array and main array succeed; it only aggregates existing artifacts and never trains or regenerates missing runs.
- Every compute job initializes Conda locally and sets BLAS/OpenMP thread counts to one.

## Acceptance criteria

- Exactly 6 unique warmup specs and 30 unique main specs are generated.
- The five conditions for the same `(backbone, seed)` load byte-identical epoch-5 model/optimizer/RNG state before condition-specific training.
- Saturation, relative-tail and capacity losses inspect only final-hidden s6/s7 neuron slices.
- Healthy thresholds/targets are derived from train split only and persisted in a reference JSON artifact.
- Capacity floor uses the matching warmup run's s6/s7 valid firing reference and `eta=0.7`.
- No test evaluation occurs before final selected-checkpoint evaluation.
- History contains all requested train/val BA, task loss, optimization loss, regularizer, firing-rate and output-weight-utilization fields for all 50 epochs.
- Plot panel 3 contains task loss, weighted regularizer contribution and total loss on the same axes.
- Finalized artifacts include per-run summary, seed-aggregated comparison, long-format histories, per-shift firing/output-weight diagnostics, reference targets and manifest.
- Slurm scripts follow repository one-core task-level parallelism and dependency rules.
- Focused contract tests cover run mapping, shared warmup/fork identity contract, long-neuron slicing, loss definitions, no-test-during-training behavior, history schema, plot contract and Slurm topology.

## Allowed write scope

Experiment 0.2.2 source, focused tests, experiment-specific Slurm scripts, experiment README/notebook and the TaskSpec. Do not modify dataset files, vendor code, Exp0.1/Exp0.2 historical artifacts or unrelated experiments.

## Validation

- `python -m pytest -q tests/test_experiment_0_2_2_capacity_preserving_loss_contract.py`
- `python -m pytest -q tests/test_repository_source_syntax.py`

## Replan triggers

Replan only if the current Exp0.2 checkpoint/model loader, direct-SNN shift allocation, split contract or WholeCount objective interface is incompatible with this experiment.