# WritingRing Workboard

This file is the PRIMARY thread's durable orchestration state.

It records active plans, task state, dependencies, and the next required
action.

It is NOT a technical design document, implementation log, test log, or
replacement for `docs/notes/**`.

Only the PRIMARY thread normally updates this file.

## Status values

- DRAFT
- PROBING
- READY_TO_FREEZE
- FROZEN
- IMPLEMENTING
- VERIFYING
- NEEDS_REPLAN
- BLOCKED
- DOCUMENTING
- DONE

## Active plan

Plan:
`docs/plans/<plan-name>.md`

Goal:
<short goal>

Plan state:
ACTIVE

Last orchestration update:
<commit or timestamp if useful>

## Task DAG

| Task | Goal | Depends on | Status | Next action |
| --- | --- | --- | --- | --- |
| T001 | Validate producer contract | — | DONE | — |
| T002 | Implement dataset loader | T001 | VERIFYING | luna_verifier |
| T003 | Implement masked loss | T001 | FROZEN | luna_worker |
| T004 | Training engine | T002,T003 | BLOCKED | wait |
| T005 | CLI integration | T004 | DRAFT | probe later |

## Task records

### T002 — Action0 dataset loader

Status: VERIFYING

Source plan:

- `docs/plans/...`

Dependencies:

- T001

Frozen TaskSpec:

- Goal: ...
- Required behavior: ...
- Contracts preserved: ...
- Allowed write paths:
  - `snn/action0_dataset.py`
  - `tests/test_action0_dataset.py`
- Acceptance criteria:
  - ...
- Replan triggers:
  - ...

Latest probe:

- verdict: CONFIRMED
- key contracts:
  - padded tensor is `(S, T_pad, 21)`
  - valid mask is `(S, T_pad)`
- unresolved issues: none

Latest worker:

- status: DONE
- changed:
  - `snn/action0_dataset.py`
  - `tests/test_action0_dataset.py`
- focused tests: PASS
- full pytest: PASS
- deviations: none

Latest verifier:

- status: PENDING

Next action:

- spawn `luna_verifier` for T002

## Blockers

None.
