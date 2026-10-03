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
5. active-task instructions or plan, when one exists

When sources materially disagree, resolve the conflict instead of guessing.

## Impact-driven CI and test selection

CI and test validation MUST be selected from the files changed in the current
task and the behavior those files can affect.

Default rule:

```text
changed files -> dependency / behavior impact -> relevant CI checks
```

Required behavior:

* Run the smallest set of CI checks or tests that meaningfully validates the
  current change.
* Run a CI check only when the modified files can affect the code, behavior,
  contract, artifact, or workflow covered by that check.
* Do not run unrelated CI checks merely because they exist in the repository.
* Prefer file-, module-, or experiment-specific checks over broad repository
  checks when the change is isolated to that scope.
* Documentation-only changes do not require code CI unless a documentation
  check specifically covers the changed files.
* A change isolated to one experiment should normally run that experiment's
  focused contract/tests, not unrelated experiment checks.
* Expand validation when shared/core code is modified, including utilities,
  dataset loaders, model components, training/evaluation infrastructure, or
  other code imported by multiple experiments.
* Also expand validation when dependency/environment/packaging/CI
  configuration changes, when the impact surface is uncertain, when a focused
  failure suggests a broader regression, or when the user explicitly requests
  broader/full validation.
* Select checks by transitive impact, not only by filename. A shared module
  change may require multiple downstream checks even if the active task belongs
  to only one experiment.
* Before claiming completion, report which relevant checks were actually run.
  If broader checks were intentionally not run because they are outside the
  affected surface, do not imply that the full CI suite was executed.

## Mandatory fast checks before delivery

For any change that creates or edits Python or Bash source, the repository source-syntax check is relevant and must be run before claiming the change is ready:

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
* Treat 50 concurrent CPU tasks as a repository ceiling, not a target. Each experiment README must declare a recommended concurrency based on run cost and scheduler/fairshare behavior.
* Never request more than 50 simultaneously running experiment CPU tasks without explicit user approval.
* Split sweeps across seeds, objectives, architectures, configurations, or other independent conditions instead of running them sequentially in one process.
* When required per-run post-training work depends only on that run, is lightweight relative to training, and is resource-compatible with training, keep it in the same Slurm task: `train -> select checkpoint -> evaluate -> required per-run analysis -> complete.json`. Do not create separate CPU stages merely for organizational convenience.
* Separate post-training jobs are appropriate for cross-run aggregation, reused references without a training task, materially different resource requirements, or independently expensive diagnostics. Every deviation must be stated in the experiment README.
* Keep training and evaluation/analysis as separate Python functions/modules even when one Slurm task executes them consecutively, so valid checkpoints can be re-analyzed without retraining.
* Reused/frozen baselines that require no training may be evaluated by a small separate job; use one CPU sequentially when the baseline set is small unless there is a measured reason to parallelize it.
* Use Slurm `afterok` dependencies for finalizers that require multiple job groups to complete.
* Finalizers should aggregate existing per-run artifacts only; they should not retrain models or silently regenerate missing runs.
* Experiment notebooks should be analysis-only whenever practical: read finalized CSV/JSON artifacts, aggregate, rank, and plot. Do not make the notebook the primary training or multiprocessing driver.
* Every Slurm compute node/job must initialize its environment locally. For CPU experiment jobs, use the repository's validated `scripts/bash_script/SNN_Bash/slurm_cpu_env.bash` bootstrap when available; it initializes only the required Lmod pieces, loads `conda/latest`, activates `writingring-gpu` with documented fallback to `writingring-viz`, and sets one-core thread limits.
* Do not `source /etc/profile` wholesale in Slurm experiment jobs. Unity's `/etc/profile.d/z05-lmod-purge.sh` reads `LMOD_DO_PURGE` unsafely under some batch environments and has caused immediate array-wide bootstrap failures. If a different bootstrap is required, document and smoke-test it.
* Do not rely on Conda activation inherited from the login/submit shell, and do not pass the submit shell's absolute Python executable path to compute nodes as the environment contract.
* Do not place comma-separated values such as label lists directly inside `sbatch --export=...`, because Slurm uses commas as variable separators. Export the complete value in the submit shell first, then submit with `--export=ALL` so the value is inherited intact.
* Set CPU thread environment variables such as `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` for one-core-per-task jobs to avoid hidden oversubscription.
* Size walltime and memory for the complete atomic task, including evaluation after training.

Valid reasons to deviate include a workload that is demonstrably GPU-bound, requires materially more memory per run, has unavoidable shared mutable state, has a true serial dependency between runs, cannot safely write independent artifacts, or has benchmark evidence that another execution model is better. Document the reason for the deviation in the experiment README or runner comments.

For new experiment implementations, treat this multi-CPU pattern as the repository default rather than an experiment-specific optimization.

### Required execution contract for every experiment README

Every newly implemented experiment must contain an explicit execution contract in its README before formal Slurm submission. At minimum it must declare:

* required per-run artifacts/analyses that gate completion;
* optional/exploratory artifacts that do not gate completion;
* the exact Slurm DAG and which steps are co-located in one task;
* any heavy diagnostic that is intentionally split into a separate job, with the reason;
* whether reused/reference artifacts require a separate evaluation/diagnostic job;
* the files/coordinates the finalizer requires;
* recommended array concurrency;
* the compute-node environment/bootstrap contract;
* the environment smoke command/job and experiment smoke command/job;
* resume/skip behavior and provenance checks used before reusing checkpoints or completed runs.

A Slurm job reaching `COMPLETED` is not the experiment-level completion criterion. A run is complete only when its README-declared required artifacts have been produced and a valid `complete.json` (or an explicitly documented equivalent) verifies them. Finalizers must fail closed on missing required artifacts.

Runs should be restartable from artifacts rather than from manually maintained workflow-state markers:

```text
valid complete.json + matching provenance -> skip the run
valid checkpoint + matching provenance    -> skip training, continue required analysis
missing or mismatched provenance          -> do not silently reuse artifacts
```

Artifact provenance should include enough information to reject stale results after meaningful implementation changes (experiment/protocol version, seed/case identity, dataset/core-protocol identity, and source/hash or equivalent repository revision).

### Mandatory preflight before formal experiment arrays

Formal Slurm arrays for a new or materially changed experiment must be gated by two cheap checks:

1. **Environment smoke on a compute node:** initialize the exact environment bootstrap used by formal jobs, activate the intended Conda environment, import critical dependencies, and verify the expected Python/runtime.
2. **Experiment smoke:** load the real locked protocol/data path, construct each materially distinct model case, run at least one small forward/loss/backward/optimizer step, exercise checkpoint save/load, and run a minimal evaluation/probe path when those components are part of the formal run.

The smoke path must use the same bootstrap and core code paths as the formal tasks. Do not replace it with a login-node-only import test. If smoke fails, do not submit the formal array until the failure is understood or the user explicitly requests a bypass.

# Agent execution model

PRIMARY owns end-to-end task execution.

PRIMARY is responsible for understanding intent, inspecting repository state,
planning when useful, implementing changes or delegating implementation,
running relevant validation, reviewing results, and documenting durable
findings when appropriate.

PRIMARY may use subagents when they improve efficiency, parallelism,
specialization, or independent verification. Subagent use is optional unless
the active runtime itself requires delegation.

Possible specialist roles include:

* targeted read-only investigation;
* implementation and local repair;
* independent verification;
* experiment or result analysis.

PRIMARY decides:

* whether a subagent is needed;
* which role to use;
* how many subagents to use;
* whether work should run sequentially or in parallel;
* whether independent verification is worthwhile;
* whether a written plan or persistent task state is useful.

Do not delegate merely to satisfy a workflow. PRIMARY remains responsible for
the final result even when work is delegated.

PRIMARY may directly inspect code, tests, documentation, experiment outputs,
and repository history. It may also implement and validate changes directly.

When subagents are used, give them the minimum context and scope required for
their task. Avoid redundant investigation, duplicate validation, or artificial
handoffs that do not improve the outcome.

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

Use planning files only when the work is substantial enough that persistent planning or orchestration state is useful.

Prefer behavior-oriented tasks over file-by-file micro-tasks.

Use:

* `docs/notes/**` for durable verified technical knowledge;
* `docs/plans/**` for substantial planning/orchestration;
* `WORKBOARD.md` only when persistent multi-task state is useful.

Simple or localized tasks normally require none of these.

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
edge cases, and implementation choices that remain within the requested
behavior are not reasons to create or rewrite a plan.

# Continuous execution

When asked to complete a plan, continue while runnable work remains.

Do not stop merely because a subagent or individual task finished.

Stop only when:

* requested work is complete;
* user input or a user-level decision is required;
* external access/data/environment blocks progress;
* an unrecoverable tool failure occurs;
* the user asks to stop.

# Recovery

On a new session, read only the state relevant to the active task.

Do not repeat completed investigation unless intervening changes may have invalidated it.