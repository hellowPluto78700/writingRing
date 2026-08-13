# Workboard

## Active plan

- **Plan:** `docs/plans/TODO/09_Board_Assisted_Segmentation_Follow_up_Plan.md`
- **TaskSpec:** frozen incrementally after each dependency-ready implementation investigation.
- **State:** COMPLETE — Board-assisted segmentation follow-up independently verified.

## Current DAG state

| Task | Dependencies | State | Notes |
| --- | --- | --- | --- |
| T0 | — | DONE | Core `board_event_segmentation.py` replacement supplied by prior work; Plan 09 treats it as complete unless current validation finds a defect. |
| T1 | T0 | VERIFIED PASS | CLI/config integration and focused CLI suite independently verified. |
| T2 | T0 | VERIFIED PASS | Revision 2 fixture repair independently verified; no production contract changed. |
| T3 | T1 | VERIFIED PASS | Action0 shared wrapper integration independently verified. |
| T4 | T1, T2, T3 | VERIFIED PASS | Documentation contract independently verified against source. |
| T5 | T1, T2, T3, T4 | VERIFIED PASS | Cross-layer verification: 266 focused tests passed, 1 skipped; full pytest: 627 passed, 1 skipped. |

Each task follows `luna_probe → PRIMARY frozen TaskSpec → execution → luna_verifier`; PRIMARY owns all Workboard and project-documentation edits. T1 and T2 are independent after T0 and may proceed concurrently.
