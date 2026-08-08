# Workboard

## Active plan

- **Plan:** `docs/plans/TODO/00_Documentation_Contract_Repair_Plan.md`
- **Task file:** `docs/plans/TODO/00_Documentation_Contract_Repair_TASKS.md`
- **Baseline commit:** `34a79fc6a6d5942e082cc23ab6be4e5c20dd63db`
- **Current task:** none
- **Status:** DONE

## Task state

| Task | State | Depends on |
| --- | --- | --- |
| T001 | DONE | — |
| T002 | DONE | T001 |
| T003 | DONE | T001 |
| T004 | DONE | T001 |
| T005 | DONE | T001 |
| T006 | DONE — DATA_FORMAT only; reports deferred | T001, T003 |
| T007 | DONE | T002–T006 |
| T008 | NOT REQUIRED | no implementation discrepancy found |

## Focus and next action

- Probe digest: the revised baseline is confirmed. It distinguishes
  work-axis-only alignment artifacts, scoped defaults, generic versus public
  schemas, and mask/state semantics; no implementation discrepancy was found.
- Focus: source representation and rate wording; 9/15/21 schemas; alignment
  endpoint work axis and canonical projection status; segmentation
  combinations; scoped defaults; historical-report scope.
- Blockers: none recorded.
- Primary execution digest: recorded the shape matrix, timestamp vocabulary,
  scoped defaults, and explicit alignment-projection/mask distinctions in the
  frozen task file; README and notes remain untouched.
- Verifier digest: PASS. T001 scope and all frozen contracts were independently
  confirmed; 87 targeted tests passed. Environment note: the verifier observed
  Python 3.10.20 in `writingring-viz`, although repository policy requires
  3.11; no T001 artifact depends on changing that environment.
- Probe digest: T002 is confirmed. README may describe the public entry-point
  flow and link to notes; it must distinguish optional separate Action-0
  training from the wrapper pipeline.
- Primary execution digest: README was rewritten within the frozen scope to
  be an orientation layer with validated note links and current entry points.
- Verifier digest: PASS. README's source/rate/action terminology, pipeline,
  entry points, link targets, and scope were independently confirmed.
- Probe digest: T003 confirmed endpoint reconstruction for both input kinds;
  schema-v2 `offset_us` is work-axis-domain and canonical projection failure
  does not invalidate a declared work-axis artifact.
- Primary execution digest: repaired only ALIGNMENT_OUTPUTS and the derived
  alignment section of DATA_FORMAT.
- Verifier digest: PASS. 66 focused tests passed, 1 real-sample test skipped;
  endpoint/projection/consumer documentation matches current behavior.
- Plan reconciliation: PROJECT_REPORT and VENDOR_WINDOWING_REPORT are deferred
  / out of scope, therefore excluded from T006 and T007 blocking criteria.
- Probe digest: T004 confirmed public 9/15/21 schemas, independent selectors,
  Board validation/no-fallback behavior, and stale raw/six-nine/strict-axis
  phrases requiring documentation repair.
- Primary execution digest: repaired T004's frozen note scope; no runtime
  artifacts changed.
- Verifier digest: PASS. 41 targeted tests passed, 1 skipped; raw/spike widths,
  selector behavior, and Board fallback rules match the repaired notes.
- Probe digest: T005 confirmed current mode/default/legacy alias semantics.
- Primary execution digest: repaired SEGMENTATIONS_BASH_SCRIPTS only.
- Verifier digest: PASS. Pipeline/dataset/model focused checks passed.
- Probe digest: T006 requires a narrow DATA_FORMAT repair (0.10-second
  stationary default, scoped defaults, and clearer four-way classification).
- Re-probe digest: T006 revised scope confirmed; deferred reports stay
  untouched.
- Primary execution digest: corrected DATA_FORMAT classification/defaults.
- Verifier digest: PASS. 47 focused tests passed; no deferred report changed.
- Probe digest: T007 found no scoped contradiction; deferred reports are
  excluded, and remaining stale-term hits are contextually correct.
- Verifier digest: PASS. 102 focused tests passed and 1 skipped; no scoped
  contradiction or implementation replan was found. Environment note remains:
  verifier observed Python 3.10.20 despite the repository's Python 3.11 rule.
- Plan result: DONE. T001–T007 passed their required probe/freeze/PRIMARY
  documentation/verifier lifecycle; T008 is not required. Deferred historical
  reports remain unmodified and out of scope.
