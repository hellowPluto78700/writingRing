# Exp14.1 Dual-Loader Transferable Accumulation — TaskSpec

## Goal

Test whether the cross-user phase alignment benefit from Exp14 can be preserved while restoring the original CoreBenchmark task-loss sampling distribution, and whether prefix-level shared evidence supervision makes the native shared readout class-decodable across multiple temporal horizons without erasing useful phase-dependent information. Prefix success is not, by itself, interpreted as proof of intrinsic temporal accumulatability; the evidence-organization diagnostics are required for that stronger interpretation.

## Required behavior

- Reuse the finalized CoreBenchmark production dataset, user split, seeds `(11, 23, 37)`, two-layer width-128 multi-tau SNN, `tau_mem=22.54 ms`, shifts `(2,3,4)` in both layers, threshold `0.5`, unnormalized synapses, and bias-free analog output head.
- Preserve the CoreBenchmark random shuffled **task-loss sampling distribution** for all WCCE and Prefix-WCCE updates. The overall backbone optimization distribution is intentionally changed by the auxiliary Phase-CU gradient.
- Use a separate deterministic user-diverse auxiliary loader only for the phase-conditioned cross-user SupCon loss: 8 classes x 8 distinct eligible users/class x 1 unique segment/(class,user) = 64 samples. A user is eligible for a class iff that class-user cell has at least one train sample; zero-sample cells are skipped only for that class.
- Never compute WCCE or Prefix-WCCE on the structured auxiliary batch.
- Phase loss is unchanged from Exp14 C2: L2 pre-reset states at normalized phases `(0.25, 0.50, 0.75, 1.00)`, training-only `128->64->ReLU->32` projection, L2 normalization, and positives restricted to same class / different user.
- Prefix-WCCE uses the native shared evidence head and exactly two prefixes: 50% and 75% of each valid sequence. For each prefix, average native evidence over the prefix and apply one CE. The first-pass prefix loss is assumption-light and symmetric: `1/2 * CE(prefix50) + 1/2 * CE(prefix75)`.
- Prefix-WCCE must not include 25% or 100%; final 100% supervision remains the ordinary WCCE term.
- Auxiliary weights are zero through epoch 10, ramp linearly through epoch 30, then plateau.
- Phase sweep: `lambda_phase in {0.01, 0.03, 0.06, 0.10}`.
- Prefix sweep: `lambda_prefix in {0.10, 0.25, 0.50}`.
- C0 is a null control with both auxiliary weights zero. It must bypass auxiliary sampling entirely, execute the current CoreBenchmark O0 task-training semantics for the same seed, and record the first three deterministic task-batch sample-ID groups for debugging. Exact tensor equality to the historical frozen O0 checkpoint is not a hard requirement; that checkpoint is retained as a provenance-validated, non-blocking state/epoch/validation-metric sanity reference.
- Hyperparameter selection uses training/validation information only. Test arrays must not be read by phase-1 selection code.
- Phase lambda selection first enforces native validation BA >= C0 mean minus 1 pp, then maximizes validation L2 spike cross-user retrieval; ties within 0.5 pp choose the smaller lambda.
- Prefix lambda selection enforces native validation BA >= C0 mean minus 1 pp and validation relative10 no-bias BA >= C0 mean minus 1 pp, then maximizes validation L2 spike whole-count no-bias BA; ties choose smaller collapse gap and then smaller lambda.
- A loss family with no validation-eligible candidate is recorded as `status=no_eligible_candidate`; its threshold is never relaxed and no candidate is forced.
- Combined cases use independently selected lambdas only when both families have `status=selected`: full/full, half-phase/full-prefix, full-phase/half-prefix, and half/half. Otherwise combined is `not_applicable`. These are preregistered sensitivity cases; test metrics never select among them.
- Final test evaluation is performed only after hyperparameter selection: always C0, plus each family with a validation-selected candidate, plus combined cases only when both families are selected. Fixed array slots with no mapped case must return `SKIPPED` with exit code 0.
- Do not persist full timestep trace archives; retain checkpoints so traces can be regenerated.

## Diagnostics

- Native train/validation/test BA, accuracy and macro-F1 for final selected cases.
- For C0, record the historical frozen O0 checkpoint hash, state-hash/key match, mismatched parameter names, best-epoch match, and validation BA/CE deltas as non-blocking sanity diagnostics; checkpoint identity/run provenance remains a hard validation.
- L1/L2 CoreBenchmark temporal probes, with L2 spike whole-count no-bias and relative10 no-bias as primary accumulation diagnostics.
- `collapse_gap_pp = 100 * (relative10_no_bias_ba - wholecount_no_bias_ba)` for validation selection and final test reporting.
- L2 spike cross-user retrieval is primary for phase-loss selection; L2 pre-reset retrieval and geometry are secondary.
- Existing Exp14 history-generalization analysis remains secondary for final selected cases.
- Directly evaluate native-prefix BA at 50%, 75%, and 100% on train/validation/test; 100% equals the native full-sequence readout.
- Diagnose accumulated evidence organization with true-class margin trajectories, cosine similarity between 50%/75% and 75%/100% cumulative evidence, monotonic-margin fraction, true-class-support sign-reversal rate, and a cancellation ratio. These diagnostics distinguish earlier discriminability from progressively compatible accumulation.
- Record gradient norms and cosine similarities among WCCE, phase-CU, and Prefix-WCCE on L2 parameters at deterministic diagnostic points. WCCE/Prefix gradients use four fixed Core-distribution task batches; Phase-CU gradients use four fixed structured auxiliary batches; average each objective's gradients before computing cosine similarities. Diagnostics are observational only and must not dynamically change weights or optimizer behavior.

## Execution contract

- Prepare job freezes/validates the CoreBenchmark dependency, validates the complete auxiliary sampler schedule across all seeds and epochs, writes a per-class sampler-capacity report, and writes the Exp14.1 protocol manifest.
- Phase-1 array: 24 one-CPU tasks = 3 C0 + 12 phase-only + 9 prefix-only runs.
- Selection job aggregates phase-1 validation artifacts only and writes selected lambdas.
- Phase-2 array reserves 12 one-CPU slots = 4 combined cases x 3 seeds. If combined is not applicable, all reserved slots return `SKIPPED` successfully.
- Final-evaluation array reserves 21 one-CPU slots = 3 C0 + up to 3 selected phase-only + up to 3 selected prefix-only + up to 12 combined checkpoints. It is created only after selection; slots beyond the dynamically selected case list return `SKIPPED` successfully.
- Finalizer aggregates existing final-evaluation artifacts only; it never retrains or evaluates models.
- Slurm array concurrency is capped at 50 and every compute job initializes Conda locally with BLAS/OpenMP threads set to one.

## Preserved contracts

- Do not modify CoreBenchmark, Exp13, or Exp14 source/artifacts.
- Do not alter SNN dynamics, architecture, tau, output-head geometry, dataset, split, seeds, or native WCCE reduction.
- Do not use a memory bank, dynamic loss weighting, PCGrad, GradNorm, a new monotonic/cancellation training loss, or test-set hyperparameter selection.
- The phase projection head is training-only and discarded at inference.

## Acceptance criteria

- Exactly 24 phase-1 specs and 12 phase-2 combined specs are generated.
- C0 model initialization matches the current Core O0 for each seed, the C0 training/optimizer path uses the same Core shuffled task batches and WCCE objective without auxiliary forward/backward contribution, and the historical O0 checkpoint can differ in tensors or selected epoch without failing the run when its provenance is valid.
- Prefix-WCCE applies one CE to the 50% prefix mean and one CE to the 75% prefix mean with fixed equal weights `(1/2, 1/2)`.
- Phase auxiliary batches contain no duplicate segment ID, exactly 8 selected classes, exactly 8 distinct users per selected class, one segment per selected class-user cell, and therefore exactly 7 same-class/different-user positives for every anchor. They never enter the WCCE/Prefix-WCCE terms.
- Selection code uses no `test_*` arrays or test metrics.
- Final artifacts report native metrics, native-prefix metrics, evidence-organization diagnostics, probes, collapse gap, cross-user retrieval/geometry, history generalization, gradient diagnostics, selection metadata and a manifest.
- Focused tests cover run mapping, C0 auxiliary bypass, task-batch trace determinism, auxiliary-batch uniqueness/positive validity, prefix loss math, C0/Core initialization contract, selection constraints, no-test selection behavior, evidence diagnostics, and Slurm topology.

## Allowed write scope

Exp14.1 source, README, TaskSpec, focused tests, experiment-specific Slurm scripts and analysis notebook. Do not modify shared CoreBenchmark or historical experiment code.

## Validation

- `python -m pytest -q tests/test_experiment_14_1_dual_loader_accumulation_contract.py`
- `python -m pytest -q tests/test_repository_source_syntax.py`

## Replan triggers

Replan only if the current CoreBenchmark loader/checkpoint contract, Exp14 phase-loss implementation, or Core probe interface cannot support the required dual-loader and validation-only selection semantics without modifying shared code.
