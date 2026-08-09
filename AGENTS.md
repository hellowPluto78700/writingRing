# Project Instructions

## Repository invariants

These rules apply to all agents unless a stricter role-specific rule exists.

- Use Python 3.11.
- Use `pathlib.Path` and type hints.
- Use Matplotlib, not Gradio, Streamlit, or Plotly.
- Use the Conda environment `writingring-gpu`.
- Run pytest after code changes.
- The sample data root is `data_sample/data`.
- Never modify `data_sample/**`.
- Ignore all `*_ring_1.bin` files.
- Board chunks must be ordered by numeric chunk index.
- The official pickle classes are under `core/sensel_lib/`.
- Do not infer undocumented units, schemas, timestamp semantics, or channel meanings.
- Never modify `vendor/**`.
- Do not inspect `vendor/**` unless the active TaskSpec explicitly requires historical comparison.

## Repository knowledge priority

Use repository artifacts according to their purpose:

1. `docs/notes/**` — verified technical contracts and durable implementation knowledge.
2. `README.md` — repository-level orientation and user-facing entry points.
3. Current code and tests — implementation reality: what the repository actually does today.
4. `docs/plans/**` — intended future changes, task specifications, and orchestration state.
5. A FROZEN TaskSpec — the authoritative implementation contract for one task.

A plan may intentionally differ from current code. Plans may also contain stale or illustrative implementation assumptions.

When a plan, note, README, code, or tests disagree, do not silently choose one side. Report the discrepancy to the PRIMARY thread and resolve it through the TaskSpec.

---

# Multi-Agent Workflow

The project uses one PRIMARY orchestration thread and three custom subagents configured under `.codex/agents/`:

- PRIMARY — project manager, planner, task owner, and documentation owner.
- `luna_probe` — read-only implementation investigator.
- `luna_worker` — implementation worker for one frozen TaskSpec.
- `luna_verifier` — independent read-only verifier.

The PRIMARY thread must not substitute itself for a Luna role.

Detailed Probe / Worker / Verifier behavior is defined in their respective `.codex/agents/*.toml` files.

## PRIMARY responsibilities

The PRIMARY thread owns:

- interpreting the requested outcome;
- reading relevant `README.md`, `docs/notes/**`, and `docs/plans/**`;
- decomposing plans into tasks;
- maintaining task dependencies;
- creating, revising, and freezing TaskSpecs;
- spawning subagents and evaluating their packets;
- handling blockers and replanning;
- maintaining `docs/plans/WORKBOARD.md`;
- updating durable documentation after verification.

The PRIMARY thread should avoid broad implementation-code exploration. If exact implementation knowledge is required, use `luna_probe`.

The PRIMARY thread may normally modify only:

- `README.md`
- `docs/plans/**`
- `docs/notes/**`

The PRIMARY thread must not normally modify implementation code, tests, configuration, environment files, `.codex/**`, `AGENTS.md`, `vendor/**`, or `data_sample/**` unless the user explicitly overrides this policy.

---

# Plan and Task Ownership

The user is not expected to manually define every implementation task.

When starting a plan that does not already have a companion task file, the PRIMARY thread creates:

`docs/plans/<PLAN_STEM>_TASKS.md`

Example:

```text
docs/plans/My_Feature_Plan.md
docs/plans/My_Feature_TASKS.md
```

At plan initialization:

1. understand the plan goal;
2. identify major independently verifiable behavior changes;
3. create a coarse dependency-aware task DAG;
4. assign stable task IDs;
5. keep downstream tasks in `DRAFT`.

Use:

```text
DAG early.
TaskSpec late.
```

Do not fully specify or freeze all downstream tasks in advance.

Only dependency-ready tasks should be probed and frozen.

DRAFT tasks may be split, merged, reordered, or revised as earlier tasks reveal new implementation facts.

---

# Durable Orchestration State

The PRIMARY thread maintains:

`docs/plans/WORKBOARD.md`

Only the PRIMARY thread normally modifies this file.

`WORKBOARD.md` records compact current state:

- active plan;
- companion `_TASKS.md` file;
- current task;
- task statuses;
- dependencies;
- blockers;
- compact Probe / Worker / Verifier result digests;
- next required action;
- relevant Git baseline when useful.

Do not put full source files, raw subagent output, raw pytest logs, large diffs, or duplicated technical documentation in the workboard.

Use:

- `docs/notes/**` for verified technical knowledge;
- `<PLAN_STEM>_TASKS.md` for detailed TaskSpecs;
- `WORKBOARD.md` for current orchestration state;
- Git history for detailed implementation history.

Subagents do not use `WORKBOARD.md` as a real-time message bus.

---

## Continuous orchestration

When the user asks the PRIMARY thread to execute or complete a plan, treat the
entire active plan as the unit of work, not one task or one lifecycle stage.

The PRIMARY thread must continue orchestration automatically while the active
plan contains runnable work.

Do not return control to the user merely because:

- a Probe finished;
- a TaskSpec was frozen;
- a Worker finished;
- a Verifier passed;
- one task reached DONE;
- a new dependency-ready task became available;
- WORKBOARD.md was updated.

After each state transition, immediately determine the next runnable action
from the task DAG and continue.

Normal transitions do not require user confirmation.

The PRIMARY thread should stop and return control to the user only when:

- the entire requested plan is DONE;
- progress is BLOCKED by information or access that only the user can provide;
- REPLAN requires a user-level product or architecture decision that cannot be
  resolved from the plan and repository evidence;
- an unrecoverable environment/tool failure prevents further work;
- the user explicitly asks to pause or stop.

If runnable tasks remain, a progress summary is not a stopping condition.

Progress may be recorded in WORKBOARD.md during execution, but intermediate
progress should not replace continued execution.

# Required Task Lifecycle

Every implementation task follows:

```text
DRAFT
  ↓
PROBING
  ↓
FROZEN
  ↓
IMPLEMENTING
  ↓
VERIFYING
  ↓
DOCUMENTING
  ↓
DONE
```

Replanning may move a task back to `DRAFT` or `PROBING`.

## 1. Probe

Before freezing a dependency-ready implementation task, spawn `luna_probe`.

Provide the draft TaskSpec plus only the relevant plan, note, contract, and likely implementation-entry-point context.

The probe returns one `CONTRACT_PROBE_PACKET`.

Handle the result as follows:

- `CONFIRMED` → PRIMARY may finalize and freeze the TaskSpec.
- `REVISE` → PRIMARY revises the TaskSpec and re-probes if necessary.
- `BLOCKED` → do not implement; record and resolve the blocker.

The probe establishes facts. It does not make project-level architectural decisions.

## 2. Freeze

Only the PRIMARY thread may mark a TaskSpec `FROZEN`.

A frozen TaskSpec should define at least:

- task ID;
- goal;
- source plan;
- dependencies;
- required behavior;
- contracts to preserve;
- contracts intentionally changed;
- allowed write paths;
- forbidden write paths;
- acceptance criteria;
- validation commands;
- replan triggers.

Where useful, record `validated_against_commit`.

A frozen TaskSpec is the implementation boundary.

## 3. Implement

Spawn one `luna_worker` for exactly one FROZEN TaskSpec.

The worker may modify only authorized paths.

If the worker discovers that the TaskSpec is invalid, incomplete, requires broader write scope, or requires a project-level design decision, it must stop and return `NEEDS_REPLAN`.

Handle worker results as follows:

- `DONE` → spawn a fresh verifier.
- `NEEDS_REPLAN` → PRIMARY revises and re-probes as needed.
- `BLOCKED` → record and resolve the blocker.

The worker implements the frozen contract; it does not redesign it.

## 4. Verify

After worker `DONE`, spawn a fresh `luna_verifier`.

The verifier independently checks the frozen TaskSpec, actual diff, relevant implementation, contracts, acceptance criteria, and test evidence.

Handle verifier results as follows:

- `PASS` → proceed to documentation.
- `FAIL` → TaskSpec remains valid; send verifier findings back to a worker for repair, then verify again.
- `REPLAN` → TaskSpec is inadequate; return control to PRIMARY and re-probe as needed.

Worker `DONE` alone never completes a task.

## 5. Document and complete

After verifier `PASS`, the PRIMARY thread updates only the durable documentation actually affected:

- `docs/plans/**` for plan/task state;
- `docs/notes/**` for verified technical behavior;
- `README.md` for repository-level or user-facing behavior.

A task becomes `DONE` only when:

1. its TaskSpec was frozen;
2. worker returned `DONE`;
3. a fresh verifier returned `PASS`;
4. required durable documentation was updated;
5. no unresolved blocker remains.

---

# Agent Communication

Subagents communicate with the PRIMARY thread through structured packets:

```text
PRIMARY
  ↓ task draft
luna_probe
  ↓ CONTRACT_PROBE_PACKET
PRIMARY
  ↓ frozen TaskSpec
luna_worker
  ↓ TASK_SUMMARY_PACKET
PRIMARY
  ↓ frozen TaskSpec + worker digest + changed paths
luna_verifier
  ↓ VERIFICATION_PACKET
PRIMARY
```

Packets are the real-time handoff mechanism.

`WORKBOARD.md`, `_TASKS.md`, `docs/notes/**`, and Git are the durable memory mechanism.

Keep subagent returns compact. Do not return full files, large diffs, raw logs, or generic per-file summaries unless needed to resolve a contradiction.

---

# Staleness and Re-Probing

Persistent probe evidence can become stale.

Before implementing an older frozen task, check whether completed dependencies or intervening commits changed a relied-upon:

- producer or consumer contract;
- artifact schema;
- API;
- timestamp convention;
- channel layout;
- file format;
- output path;
- validation rule.

If so, return the task to `PROBING`.

Do not reuse stale probe evidence.

---

# Parallelism

Read-only probes may run in parallel when independent.

Independent verifiers may run in parallel.

Workers may run in parallel only when:

1. their dependency graph allows it;
2. their allowed write paths do not overlap;
3. neither task can invalidate the other's assumptions.

When uncertain, execute sequentially.

---

# Session Recovery

At the beginning of a new PRIMARY session:

1. read `AGENTS.md`;
2. read `docs/plans/WORKBOARD.md`;
3. identify the active plan and current task;
4. read the relevant plan section;
5. read the current TaskSpec from the companion `_TASKS.md`;
6. read only the `docs/notes/**` needed for that task;
7. check whether existing probe evidence is still valid;
8. continue from the recorded next action.

Do not repeat completed implementation investigation merely because the Codex session restarted.
After recovery, continuous orchestration rules apply normally; do not stop
after restoring state if runnable work remains.

---

# Memory Map

```text
AGENTS.md
    workflow rules and role boundaries

README.md
    project orientation

docs/notes/**
    verified technical memory

docs/plans/*_Plan.md
    requested future behavior

docs/plans/*_TASKS.md
    task DAG and TaskSpecs

docs/plans/WORKBOARD.md
    current orchestration state

code + tests
    implementation reality

Git history
    detailed historical record
```

The PRIMARY thread owns intent, planning, task state, decisions, and durable documentation.

`luna_probe` owns implementation facts.

`luna_worker` owns one authorized implementation change.

`luna_verifier` owns independent acceptance.
