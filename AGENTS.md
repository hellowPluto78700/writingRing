# Project Instructions

## Repository rules

* Use Python 3.11.
* Use `pathlib.Path` and type hints.
* Prefer Conda env `writingring-gpu`; fall back to `writingring-viz`.
* Use Matplotlib.
* Never modify `data_sample/**` or `vendor/**`.
* Do not inspect `vendor/**` unless explicitly required.
* Ignore `*_ring_1.bin`.
* Order board chunks by numeric chunk index.
* Official pickle classes are under `core/sensel_lib/`.
* Never infer undocumented units, schemas, timestamps, or channel meanings.

Knowledge priority:

1. `docs/notes/**`
2. `README.md`
3. current code and tests
4. `docs/plans/**`
5. FROZEN TaskSpec for the active task

When sources materially disagree, resolve the conflict instead of guessing.

## Mandatory fast checks before delivery

For any change that creates or edits Python or Bash source, run the relevant fast checks before claiming the change is ready:

```bash
python -m pytest -q tests/test_repository_source_syntax.py
```

This test compiles repository Python sources and runs `bash -n` on repository Bash scripts, so truncated or malformed source must be caught before delivery.

For Experiment 3.0.2 changes, also run:

```bash
python -m pytest -q tests/test_experiment_3_0_2_contract.py
```

For other experiment-specific changes, add or update focused contract tests when run mapping, architecture definitions, protocol constants, checkpoint identity, or durable experiment behavior changes.

Do not describe a change as tested or passing if the checks were not actually executed. If the current environment cannot execute them, say explicitly that the change was only statically reviewed and leave the GitHub CI result as the remaining verification gate.

## Default multi-CPU experiment execution

Unless the user explicitly requests another execution model or the workload has a concrete reason not to parallelize this way, use task-level multi-CPU execution for experiment sweeps.

The default pattern is:

```text
independent run dimensions
(seed / objective / architecture / configuration / ablation)
        -> one Slurm array task per independent run
        -> one CPU core per task
        -> train the run
        -> select/save its best checkpoint
        -> immediately evaluate that same checkpoint when evaluation depends only on that run
        -> write per-run artifacts
all required tasks complete
        -> one finalizer/aggregator job
        -> analysis-only notebook for tables and plots
```

Required defaults:

* Prefer Slurm arrays for independent experiment runs.
* Use one CPU core per independent run unless profiling shows the run materially benefits from more cores.
* Cap array concurrency at 50 tasks by default, e.g. `#SBATCH --array=0-N%50`.
* Never request more than 50 simultaneously running experiment CPU tasks without explicit user approval.
* Split sweeps across seeds, objectives, architectures, configurations, or other independent conditions instead of running them sequentially in one process.
* When a run's evaluation depends only on its own checkpoint, perform `train -> evaluate -> save artifacts` in the same Slurm task rather than creating a barrier followed by a second full evaluation array.
* Keep training and evaluation as separate Python functions/modules even when the Slurm task executes them consecutively, so evaluation can be rerun without retraining.
* Reused/frozen baselines that require no training may be evaluated by a small separate job; use one CPU sequentially when the baseline set is small unless there is a measured reason to parallelize it.
* Use Slurm `afterok` dependencies for finalizers that require multiple job groups to complete.
* Finalizers should aggregate existing per-run artifacts only; they should not retrain models or silently regenerate missing runs.
* Experiment notebooks should be analysis-only whenever practical: read finalized CSV/JSON artifacts, aggregate, rank, and plot. Do not make the notebook the primary training or multiprocessing driver.
* Set CPU thread environment variables such as `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` for one-core-per-task jobs to avoid hidden oversubscription.
* Size walltime and memory for the complete atomic task, including evaluation after training.

Valid reasons to deviate include a workload that is demonstrably GPU-bound, requires materially more memory per run, has unavoidable shared mutable state, has a true serial dependency between runs, cannot safely write independent artifacts, or has benchmark evidence that another execution model is better. Document the reason for the deviation in the experiment README or runner comments.

For new experiment implementations, treat this multi-CPU pattern as the repository default rather than an experiment-specific optimization.

# Roles

* PRIMARY: intent, routing, planning, orchestration, documentation.
* `luna_probe`: targeted read-only investigation.
* `luna_worker`: implementation and local repair loop.
* `luna_verifier`: independent verification.

PRIMARY may directly inspect code and tests. Do not spawn Probe just to answer a simple implementation question.

# Task routing

## Routing priority

FAST_FIX is the default workflow.

PRIMARY MUST use FAST_FIX when:

* the requested behavior is explicit;
* the change is localized;
* no unresolved repository fact blocks implementation;
* no schema/file-format/public-API/persistent-artifact contract must change.

When these conditions hold:

* MUST NOT spawn luna_probe;
* MUST NOT create or freeze a TaskSpec;
* MUST NOT spawn luna_verifier.

The fact that code has callers or consumers does NOT by itself require STANDARD.

Use STANDARD only when PRIMARY can name a concrete cross-module contract
that may change or a concrete unknown that must be resolved before implementation.

Do not Probe merely to increase confidence.

## FAST_FIX

Use FAST_FIX when all of the following are true:

* requested behavior is explicit;
* the change is localized;
* no unresolved repository fact blocks implementation;
* no schema, file-format, public API, persistent artifact, timestamp,
  channel, or other durable contract must change;
* no project-level architecture decision is required.

```text
PRIMARY -> worker -> done
```

When these conditions hold:

MUST NOT spawn luna_probe;
MUST NOT create or freeze a TaskSpec;
MUST NOT spawn luna_verifier;
MUST NOT update WORKBOARD;
MUST NOT run full pytest by default.

The existence of callers, consumers, or multiple touched files/modules does
NOT by itself require STANDARD.

Run focused tests and relevant checks.

Escalate only when PRIMARY or worker can name a concrete contract ambiguity,
unresolved repository fact, broader scope requirement, or meaningful
regression risk.

## STANDARD

Use STANDARD only when at least one of the following is true:

a concrete cross-module or producer/consumer contract may change;
an unresolved repository fact must be established before safe implementation;
the behavior crosses a durable interface whose invariant must be preserved;
the implementation has material regression risk that justifies independent
verification.

```text
[probe only if a concrete unresolved fact exists] -> freeze -> worker -> verifier
```

Probe is not a mandatory workflow stage.

PRIMARY may inspect code, tests, and documentation directly and freeze a task
without Probe when the required repository facts are already established.

Freeze WHAT must be true, not HOW to implement it.

A STANDARD TaskSpec should contain only:

Goal
Context, only when necessary
Required behavior
Preserved contracts
Allowed write scope
Acceptance criteria
Validation
Dependencies, only when real
Replan triggers

Do not freeze helper names, internal class/function structure, algorithms,
or exact modified files unless those details are themselves contractual.

## HIGH_RISK

Use STANDARD plus stronger verification/integration validation for changes
involving:

schemas or file formats;
public APIs;
pipeline contracts;
timestamps or channel semantics;
persisted artifacts;
broad architecture or migrations.

HIGH_RISK always requires independent verification.

Full pytest or E2E is still required only when justified by the affected
risk surface.

# Worker rules

Worker owns the normal loop:

```text
inspect -> implement -> test -> fix -> retest -> self-review
```

Ordinary bugs, missed edge cases, local refactors, and test fixes do not require replanning.

Use `NEEDS_REPLAN` only when correct implementation requires a material TaskSpec, contract, dependency, architecture, or write-scope change.

Use `BLOCKED` for genuine environment, data, access, or tool failures.

# Verification rules

Verifier checks the complete relevant task surface before returning.

* `PASS`: implementation satisfies the TaskSpec.
* `FAIL`: TaskSpec is valid; implementation needs repair.
* `REPLAN`: TaskSpec itself is materially wrong or stale.

Batch all currently discoverable blocking findings into one FAIL.

Do not FAIL for style preferences, optional cleanup, or speculative risks.

`UNVERIFIED` test evidence alone is not a failure.

# Validation

Run validation from cheap to expensive:

```text
focused checks -> focused tests -> integration tests -> full pytest -> E2E
```

Do not run the same validation twice for the same implementation state.

Do not run full pytest after every small task unless risk justifies it.

For multi-task plans, prefer focused validation per task and full regression near integration/final completion.

Do not repeatedly rerun an unchanged expensive command after environment failure.

# Plans and documentation

Use planning files only for substantial STANDARD/HIGH_RISK work.

Prefer behavior-oriented tasks over file-by-file micro-tasks.

Use:

* `docs/notes/**` for durable verified technical knowledge;
* `docs/plans/**` for substantial planning/orchestration;
* `WORKBOARD.md` only when persistent multi-task state is useful.

FAST_FIX normally requires none of these.

## Plan quality

Prefer 1-5 behavior-oriented tasks for substantial work.
Use more only when the requested behavior genuinely contains more independent
implementation units.

Each task must be independently implementable and independently verifiable.

Acceptance criteria must be objectively PASS/FAIL.

Preserved contracts must name the invariants that actually matter to the task.
Do not use vague requirements such as:

* ensure correctness;
* preserve semantics;
* handle edge cases;
* keep behavior robust.

Instead state the concrete invariant, for example:

* both test views contain identical sample identities and labels;
* validation rejects mismatched cohorts rather than intersecting them;
* existing artifact filenames and channel meanings remain unchanged.

Allowed write scope should prevent scope creep without predicting every file
the worker may need to touch.

Ordinary implementation bugs, local refactors, test fixture changes, missed
edge cases, and implementation choices inside the frozen behavior and write
scope are NOT reasons to replan.

# Continuous execution

When asked to complete a plan, continue while runnable work remains.

Do not stop merely because a Probe, Worker, Verifier, or individual task finished.

Stop only when:

* requested work is complete;
* user input or a user-level decision is required;
* external access/data/environment blocks progress;
* an unrecoverable tool failure occurs;
* the user asks to stop.

# Recovery

On a new session, read only the state relevant to the active task.

Do not repeat completed investigation unless intervening changes may have invalidated it.
