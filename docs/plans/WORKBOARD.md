# Workboard

## Active plan

- **Plan:** `docs/plans/TODO/10_spike_user_exlude_refine_plan.md`
- **TaskSpec:** frozen incrementally after each dependency-ready `luna_probe`
  investigation.
- **State:** ACTIVE — restoring the user-exclusion experiment workflow.

## Current DAG state

| Task | Dependencies | State | Notes |
| --- | --- | --- | --- |
| T1 | — | VERIFIED PASS | Config contract independently verified. |
| T2 | T1 | VERIFIED PASS | Split-layer exclusion independently verified. |
| T3 | T2 | VERIFIED PASS | A runner exclusion forwarding independently verified. |
| T4 | T3 | VERIFIED PASS | A checkpoint/provenance cohort contract independently verified. |
| T5 | T4 | VERIFIED PASS | B cohort inheritance independently verified. |
| T6 | T4 | VERIFIED PASS | C cohort inheritance independently verified. |
| T7 | T4 | VERIFIED PASS | D cohort inheritance independently verified. |
| T8 | T5, T6, T7 | VERIFIED PASS | Notebook controls independently verified. |
| T9 | T8 | DRAFT / PROBE NEXT | Run cross-experiment and regression validation. |

Each dependency-ready task follows `luna_probe → PRIMARY frozen TaskSpec → execution → luna_verifier`.
PRIMARY owns this workboard and all project-documentation edits. Documentation-only
tasks are implemented by PRIMARY; only implementation tasks use `luna_worker`.
