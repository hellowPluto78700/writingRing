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

# Roles

* PRIMARY: intent, routing, planning, orchestration, documentation.
* `luna_probe`: targeted read-only investigation.
* `luna_worker`: implementation and local repair loop.
* `luna_verifier`: independent verification.

PRIMARY may directly inspect code and tests. Do not spawn Probe just to answer a simple implementation question.

# Task routing

Use the cheapest safe workflow.

## FAST_FIX

Use for localized, low-risk changes with clear behavior and no durable contract change.

```text
PRIMARY -> worker -> done
```

No Probe, frozen TaskSpec, Verifier, WORKBOARD update, or full pytest by default.

Run focused tests and relevant checks.

Escalate if contract ambiguity, broader scope, or meaningful regression risk appears.

## STANDARD

Use when multiple modules or producer/consumer contracts are involved.

```text
probe -> freeze -> worker -> verifier
```

Freeze only behavior, contracts, allowed paths, acceptance criteria, validation, and replan triggers.

Do not freeze unnecessary implementation details.

## HIGH_RISK

Use STANDARD plus stronger integration/full validation for changes involving:

* schemas or file formats;
* public APIs;
* pipeline contracts;
* timestamps/channels;
* artifacts;
* broad architecture or migrations.

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
