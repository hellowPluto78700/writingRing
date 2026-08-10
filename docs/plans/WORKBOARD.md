# Workboard

## Active plan

- **Plan:** `docs/plans/TODO/07_Plan_Skip_Low_Confidence_Alignment_Recordings.md`
- **Task file:** `docs/plans/TODO/07_Plan_Skip_Low_Confidence_Alignment_Recordings_TASKS.md`
- **Baseline:** `2b05be416b4893eb8fbf596a6ac29a8e43b639a9`
- **Current task:** T6
- **Status:** BLOCKED — persistent full-pipeline terminal evidence

## Current DAG state

| Task | State | Depends on | Compact evidence |
| --- | --- | --- | --- |
| T1 | DONE | — | Probe CONFIRMED; worker DONE; fresh verifier PASS. Worker: 33 focused / 583 full passed, 1 skipped; verifier direct smoke/diff PASS (pytest unavailable read-only). |
| T2 | DONE | T1 | Revised Probe; worker; fresh verifier PASS. Full-suite rerun unavailable in verifier environment; final validation retained for T6. |
| T3 | DONE | T2 | Repair worker DONE; fresh verifier PASS. Authentic T1 nested confidence minima publish valid skips. |
| T4 | DONE | T3 | Repaired worker DONE; fresh verifier PASS. 32 focused / 612 full passed, 1 skipped; bash -n/diff PASS. |
| T5 | DONE | T2 | Probe CONFIRMED; worker DONE; fresh verifier PASS. Generic consumer remains source-unchanged. |
| T6 | BLOCKED | T4, T5 | User decision recorded; temporary tree correctly excludes only user_4/action_0/dataset_0, but environment repeatedly terminates full pipeline before terminal QA/padding. |

## Verified contract digest

- Prior completed plans establish the current `SUCCESS`/`SKIPPED` outcome,
  sibling-root, validated-provenance, recording-skip accounting, and aligned
  `alignment_outcome_dependency` contracts.
- The plan must retain standalone strict failure, add only structured
  confidence-derived skips, and never create consumer-visible success
  artifacts for a skipped recording.
- Plan-body references to a `03_...` plan/task filename are reconciled to the
  user-selected `07_...` companion task file; no duplicate plan is created.

## Next required action

Run the full Action0 pipeline in a persistent environment against the prepared
temporary acceptance input tree (excluding only user_4/action_0/dataset_0),
retain exit-0 QA/padding evidence, then request a fresh T6 verifier.
