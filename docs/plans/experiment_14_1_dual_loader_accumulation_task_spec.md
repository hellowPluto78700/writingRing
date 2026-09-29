# Exp14.1 Dual-Loader Transferable Accumulation — TaskSpec

## Goal

Test whether the cross-user phase alignment benefit from Exp14 can be preserved under the original CoreBenchmark task-sampling distribution, and whether prefix-level shared evidence supervision can make the improved L2 representation easier to decode with one temporal accumulator without erasing useful phase-dependent information.

## Required behavior

- Reuse the finalized CoreBenchmark production dataset, user split, seeds `(11, 23, 37)`, two-layer width-128 multi-tau SNN, `tau_mem=22.54 ms`, shifts `(2,3,4)` in both layers, threshold `0.5`, unnormalized synapses, and bias-free analog output head.
- Preserve the CoreBenchmark random shuffled task loader for all WCCE and Prefix-WCCE updates.
- Use a separate deterministic class/user-balanced auxiliary loader only for the Exp14 phase-conditioned cross-user SupCon loss: 8 classes x 4 users/class x 4 samples/user = 128 samples.
- Never compute WCCE or Prefix-WCCE on the structured auxiliary batch.
- Phase loss is unchanged from Exp14 C2: L2 pre-reset states at normalized phases `(0.25, 0.50, 0.75, 1.00)`, training-only `128->64->ReLU->32` projection, L2 normalization, and positives restricted to same class / different user.
- Prefix-WCCE uses the native shared evidence head and exactly two prefixes: 50% and 75% of each valid sequence. For each prefix, average native evidence over the prefix and apply one CE. The prefix loss is `1/3 * CE(prefix50) + 2/3 * CE(prefix75)`.
- Prefix-WCCE must not include 25% or 100%; final 100% supervision remains the ordinary WCCE term.
- Auxiliary weights are zero through epoch 10, ramp linearly through epoch 30, then plateau.
- Phase sweep: `lambda_phase in {0.01, 0.03, 0.06, 0.10}`.
- Prefix sweep: `lambda_prefix in {0.10, 0.25, 0.50}`.
- C0 is a null dual-loader control with both auxiliary weights zero and must reproduce the CoreBenchmark O0 training path for the same seed.
- Hyperparameter selection uses training/validation information only. Test arrays must not be read by phase-1 selection code.
- Phase lambda selection first enforces native validation BA >= C0 mean minus 1 pp, then maximizes validation L2 spike cross-user retrieval; ties within 0.5 pp choose the smaller lambda.
- Prefix lambda selection enforces native validation BA >= C0 mean minus 1 pp and validation relative10 no-bias BA >= C0 mean minus 1 pp, then maximizes validation L2 spike whole-count no-bias BA; ties choose smaller collapse gap and then smaller lambda.
- Combined cases use independently selected lambdas: full/full, half-phase/full-prefix, full-phase/half-prefix, and half/half. These are preregistered sensitivity cases; test metrics never select among them.
- Final test evaluation is performed only after hyperparameter selection, on C0, selected phase-only, selected prefix-only, and all four combined cases.
- Do not persist full timestep trace archives; retain checkpoints so traces can be regenerated.

## Diagnostics

- Native train/validation/test BA, accuracy and macro-F1 for final selected cases.
- L1/L2 CoreBenchmark temporal probes, with L2 spike whole-count no-bias and relative10 no-bias as primary accumulation diagnostics.
- `collapse_gap_pp = 100 * (relative10_no_bias_ba - wholecount_no_bias_ba)` for validation selection and final test reporting.
- L2 spike cross-user retrieval is primary for phase-loss selection; L2 pre-reset retrieval and geometry are secondary.
- Existing Exp14 history-generalization analysis remains secondary for final selected cases.
- Record gradient norms and cosine similarities among WCCE, phase-CU, and Prefix-WCCE on L2 parameters at deterministic diagnostic points; diagnostics are observational only and must not dynamically change weights or optimizer behavior.

## Execution contract

- Prepare job freezes/validates the CoreBenchmark dependency and writes the Exp14.1 protocol manifest.
- Phase-1 array: 24 one-CPU tasks = 3 C0 + 12 phase-only + 9 prefix-only runs.
- Selection job aggregates phase-1 validation artifacts only and writes selected lambdas.
- Phase-2 array: 12 one-CPU tasks = 4 combined cases x 3 seeds.
- Final-evaluation array: 21 one-CPU tasks = 3 C0 + 3 selected phase-only + 3 selected prefix-only + 12 combined checkpoints. It is created only after selection and phase-2 training, and performs each selected checkpoint's test/probe/diagnostic evaluation without retraining.
- Finalizer aggregates existing final-evaluation artifacts only; it never retrains or evaluates models.
- Slurm array concurrency is capped at 50 and every compute job initializes Conda locally with BLAS/OpenMP threads set to one.

## Preserved contracts

- Do not modify CoreBenchmark, Exp13, or Exp14 source/artifacts.
- Do not alter SNN dynamics, architecture, tau, output-head geometry, dataset, split, seeds, or native WCCE reduction.
- Do not use a memory bank, dynamic loss weighting, PCGrad, GradNorm, or test-set hyperparameter selection.
- The phase projection head is training-only and discarded at inference.

## Acceptance criteria

- Exactly 24 phase-1 specs and 12 phase-2 combined specs are generated.
- C0 model initialization matches Core O0 for each seed, and the C0 training step path uses the same Core shuffled task batches and WCCE objective without auxiliary forward/backward contribution.
- Prefix-WCCE applies one CE to the 50% prefix mean and one CE to the 75% prefix mean with fixed weights `(1/3, 2/3)`.
- Phase auxiliary batches contain same-class cross-user positives and never enter the WCCE/Prefix-WCCE terms.
- Selection code uses no `test_*` arrays or test metrics.
- Final artifacts report native metrics, probes, collapse gap, cross-user retrieval/geometry, history generalization, gradient diagnostics, selection metadata and a manifest.
- Focused tests cover run mapping, dual-loader isolation, prefix loss math, C0/Core initialization contract, selection constraints, no-test selection behavior, and Slurm topology.

## Allowed write scope

Exp14.1 source, README, TaskSpec, focused tests, experiment-specific Slurm scripts and analysis notebook. Do not modify shared CoreBenchmark or historical experiment code.

## Validation

- `python -m pytest -q tests/test_experiment_14_1_dual_loader_accumulation_contract.py`
- `python -m pytest -q tests/test_repository_source_syntax.py`

## Replan triggers

Replan only if the current CoreBenchmark loader/checkpoint contract, Exp14 phase-loss implementation, or Core probe interface cannot support the required dual-loader and validation-only selection semantics without modifying shared code.
