# Exp17 persistent-pathway story closure — task specification

## Scientific goal

Close the remaining causal gaps in the current SNN story without introducing a new architecture or objective.

## Task A — Epoch-wise emergence and reliance

Implement one locked LIN/WCCE trajectory per seed 11/23/37. Preserve Exp16.3 initialization, deterministic epoch sampler, optimizer, early stopping, and validation-only selection. Save epoch 0, every fifth epoch, selected-best epoch, and stopped epoch.

PASS if each snapshot produces train/val/test native metrics, L1/L2 persistence metrics, per-neuron class/user eta-squared, top-persistence-quartile head-weight share, and top-30% high-persistence frozen/retrained ablations.

Preserved contracts: test never affects training/selection; persistence ranking uses train occupancy only; no gate/auxiliary loss/new tau/depth.

## Task B — Persistent-subset information decomposition

Reuse finalized Exp16.2 L2 traces for all eight formal cases and all three seeds. Construct dimension-matched high30, low30 and deterministic random30 subsets from train occupancy.

PASS if every subset has class WholeCount decoding plus within-training-user raw and class-residualized user decoding with disjoint train/validation/test CV folds and validation-only C selection.

Preserved contracts: no SNN retraining; class residual centroids are estimated from the CV training fold only; random subsets are deterministic and dimension matched.

## Task C — Pruning specificity

For 10/20/30% removal compare high occupancy, low occupancy, high standardized WholeCount readout weight, and 20 deterministic random masks. Evaluate both frozen mean-replacement and retrained bias-free WholeCount heads.

PASS if every case/seed/fraction/ranking writes paired full-relative train/val/test BA and CE deltas and the finalizer aggregates random-control mean/std.

Preserved contracts: rankings use training-side quantities only; no SNN retraining; random masks are dimension matched.

## Task D — Finalization and execution contract

Provide one-CPU Slurm arrays: 3 trajectory tasks, 24 subset tasks, 24 pruning tasks, plus prepare and aggregate-only finalizer. Finalizer must fail on missing task outputs rather than regenerate them.

PASS if experiment-specific tests and repository source-syntax checks pass in the Unity `writingring-gpu` environment.