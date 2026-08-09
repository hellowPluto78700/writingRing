# Workboard

## Active plan

- **Plan:** `docs/plans/TODO/04_Action0_SNN_Training_Notebook_Plan.md`
- **Task file:** `docs/plans/TODO/04_Action0_SNN_Training_Notebook_TASKS.md`
- **Baseline commit:** `93d93578ad0537054eae7843ed93b574c29b1a91`
- **Current task:** T003 — Existing-training integration
- **Status:** VERIFYING

## Task state

| Task | State | Depends on |
| --- | --- | --- |
| T001 | DONE | — |
| T002 | DONE | T001 |
| T003 | VERIFYING | T002 |
| T004 | DRAFT | T003 |
| T005 | DRAFT | T004 |

## Focus and next action

- DAG established from the requested Action0 notebook plan. T001 is the sole
  dependency-ready task; T002–T005 remain DRAFT until their dependencies pass
  fresh verification.
- T001 probe digest: REVISE. Current HEAD is
  `93d93578ad0537054eae7843ed93b574c29b1a91`; the foundation must use the
  padded root, one global class map, metadata/rate and split-length validators,
  and dataset `len`/`class_distribution`. It must not call the full trainer or
  re-pad data. PRIMARY revised the draft and requires a fresh probe.
- T001 re-probe digest: CONFIRMED. The revised notebook-only boundary is
  viable using existing root/metadata/mapping/dataset APIs and the three
  trainer validators in the confirmed order. Real existing low-pass packages
  constructed as 104/151/98 segments, padded length 1024, 52 global classes;
  focused Action0 tests passed 19.
- T001 worker digest: DONE. Added only the frozen foundation notebook with
  centralized configuration, ordered existing-validator reuse, shared mapping,
  and split summaries. Focused pytest: 19 passed; full pytest: 527 passed/1
  skipped; direct notebook execution and structural audit passed. nbconvert is
  unavailable in `writingring-gpu`. `git diff --check` found only untouched
  pre-existing whitespace in `_common.bash:147`.
- T001 verifier digest: PASS. Independent review confirmed only the frozen
  notebook changed; configuration/order, padded 21-channel data, shared
  mapping, model slice, metadata/rate/split-length validation, and
  foundation-only exclusion all hold. Direct execution produced 1,751/349/300
  segments, padded length 1,024, and 52 global classes. No README/note update
  is appropriate until the complete user-facing notebook workflow exists.
- T002 probe digest: REVISE. The dataset has no canonical timestamps, so the
  plan must name the x-axis as relative nominal time based only on valid sample
  indices and the declared rate. PRIMARY revised the distribution, table,
  mask, and write-scope contract; a fresh probe is required.
- T002 re-probe digest: CONFIRMED. The existing notebook variables and dataset
  APIs support the revised percentage/table, valid-only 15-row figure, and
  truthful relative nominal sample-index time without contract changes.
- T002 worker digest: DONE. Only the frozen notebook changed; it adds the
  global-order percentage table, three split figures, and valid-only 15-channel
  figure with declared-rate nominal time. Focused pytest: 19 passed; full
  pytest: 527 passed/1 skipped; direct execution and structural audit passed.
  nbconvert is unavailable in `writingring-gpu`; known unrelated whitespace in
  `_common.bash:147` remains untouched.
- T002 verifier digest: PASS. The actual notebook alone changed; global-order
  table/three figures/inverse label map/valid-only 15-channel figure and
  declared-rate nominal-time provenance all satisfy the frozen contract.
  Real execution rendered four figures from 52 classes and 1,751/349/300
  split lengths; no timestamp, padding, or implementation contract changed.
- T003 probe digest: CONFIRMED with required TaskSpec detail. Existing in-memory
  APIs permit notebook-only integration, but the frozen contract must pin
  helper reuse, RNG-relevant train preview, all eight history metrics, and
  in-memory strict-greater best validation selection. PRIMARY revised and will
  re-probe before freezing.
- T003 re-probe digest: CONFIRMED with final freeze details: exact Action0
  engine/model imports and calls, `num_workers=0`, existing validation plus
  three-layer guard, and Python-float eight-metric history only. PRIMARY
  revised and requires one final focused re-probe.
- T003 final re-probe digest: CONFIRMED. Precise imports/signatures/order,
  three-layer guard, `num_workers=0`, uncapped eight-float histories, and
  strict in-memory earliest-tie best state are implementation-safe.
- T003 worker digest: DONE. Only the notebook changed; existing SynNet engine,
  seeded preview, uncapped eight-metric histories, and strict in-memory best
  state are integrated. Focused pytest 19 passed; full pytest 527 passed/1
  skipped; structural audit and real one-epoch execution passed. nbconvert is
  unavailable and known unrelated `_common.bash` whitespace remains untouched.
- Baseline hygiene: pre-existing edits to `AGENTS.md`, Action0 shell scripts,
  and `snn/models/action0/` are user-owned baseline state. They are outside
  this plan unless a later frozen TaskSpec explicitly authorizes a path.
- T005 is a PRIMARY validation/documentation task. T001–T004 are potential
  implementation tasks and require the probe → freeze → worker → verifier
  lifecycle.
- T003 implementation is complete. Next action: fresh verifier checks it;
  T004–T005 remain DRAFT.
