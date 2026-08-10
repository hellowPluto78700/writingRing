# Workboard

## Active plan

- **Plan:** `docs/plans/TODO/01_Board_Assist_Alignment_Correction_Plan.md`
- **Task file:** `docs/plans/TODO/01_Board_Assist_Alignment_Correction_TASKS.md`
- **Baseline commit:** `a69476c4f3ae713335b6867678cd35c48a04ef3d`
- **Current task:** T2 — Alignment outcome contract and CLI
- **Status:** T1 DONE; T2 repair complete and re-verifying

## Task state

| Task | State | Depends on |
| --- | --- | --- |
| T1 | DONE | — |
| T2 | VERIFYING | T1 |
| T3 | DRAFT | T2 |
| T4 | DRAFT | T2 |
| T5 | DRAFT | T3, T4 |

## Focus and next action

- The 01 Board-Assisted Alignment Correction plan is active. Its companion
  task file did not exist, so PRIMARY established the T1 → T2 → (T3, T4) →
  T5 DAG at baseline `a69476c4f3ae713335b6867678cd35c48a04ef3d`. T1's initial
  probe returned REVISE: current code accepts pairs when only the press is
  pre-jump and later timestamp filtering can leak post-jump stale rows; also,
  no-jump recordings with valid pairs are current normal successes, not hard
  failures. PRIMARY revised T1 to use positional `[0, j)` containment for
  both pair endpoints and every selected derived table, preserve normal
  no-jump success, and require structured diagnostics. Outcome artifacts, CLI
  policy, Action0, and segmentation remain downstream. Next action: focused
  T1 re-probe CONFIRMED the revised positional boundary, typed diagnostic, and
  two-file implementation/test scope. T1 is FROZEN against
  `a69476c4f3ae713335b6867678cd35c48a04ef3d`. Worker returned DONE with only
  `event_alignment.py` and `test_event_alignment.py` changed: positional
  prefix enforcement, both-endpoint pair eligibility, identity-aware stale-tail
  filtering, and `InitialIntervalNoUsablePairError`. Fresh verifier PASS:
  scope/contract/acceptance checks passed, it reran 51 focused tests and full
  pytest passed 530 with one skip in `writingring-gpu`; no undocumented change.
  T1 is DONE. T2 initial probe returned REVISE: the repository has only
  success TXT/report/PNG artifacts, no Board content provenance, no unified
  outcome reader, and non-atomic cross-artifact publication. PRIMARY revised
  T2 with an exact skip sibling JSON, report-as-final-manifest schema,
  raw/SpikeIMU plus ordered Board-hash provenance, strict completion validator,
  transition gate, and narrow `--initial-interval-policy`. Fresh re-probe
  CONFIRMED. PRIMARY froze literal `alignment_skip` schema v1, the eight T1
  diagnostic keys, report artifact digest entries, validator behavior, and
  allowed paths. Worker returned DONE with only `alignment_io.py`, the
  alignment CLI, and package exports changed: validated terminal outcomes,
  provenance, atomic publication, transition authorization, and explicit skip
  policy. In `writingring-gpu`, focused tests passed 35/1 skipped and full
  pytest passed 530/1 skipped; diff check passed. Fresh verifier returned FAIL
  while keeping the TaskSpec valid: schema versions accept bool/float aliases,
  conflicting timestamp digest aliases normalize away, completed publication
  may omit current provenance, pathless Board doubles get fabricated hashes,
  and frozen T2 regressions were not added. Next action: same worker repairs
  only the frozen T2 scope, then fresh verification.

## Prior controller notebook plan archive

- The controller-notebook label-contract plan is DONE. Initial probe found
  that the untracked controller notebook incorrectly calls action-directory
  IDs labels/classes; canonical labels are full, case-preserved timestamp
  sidecar texts, and `wrong` is an invalid start marker. Re-probe CONFIRMED a
  stronger contract: canonical parsing plus raw-line provenance, per-sidecar
  parse-error reporting, and no action fallback. Worker returned DONE with
  only the controller notebook changed: action IDs are nonsemantic identities;
  timestamp-marker audit preserves parser values plus raw-line provenance;
  missing and parse-error sidecars are separate; and `wrong` is non-class.
  It cleared stale outputs. Full-data audit found 606 recordings, 13,356
  markers, 257 `wrong` markers, no missing sidecars, and no parse errors.
  Focused discovery/segmentation tests passed 32; full pytest passed 527 with
  one skip; JSON/AST checks and `git diff --check` passed. Fresh verifier PASS:
  all semantics/diagnostics/scope rules hold; outputs are clear and exports
  remain default-off. It observed an existing optional-export import issue
  only when a manual switch is enabled, outside this frozen task. T001 is DONE.

## Prior Action0 notebook plan archive

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
- T003 required replanning after a user-run notebook cell raised
  `FileNotFoundError` for the relative `outputs/action0_pipeline` root.
  `probe_path_fix_final` returned CONFIRMED against
  `51293739773d6a192ad8cdec18857d6ec4b52319`: with the required editable
  install, `Path(snn.__file__).resolve().parent.parent` is the repository root
  and safely anchors the existing producer root independently of kernel cwd.
  The frozen amendment authorizes only the notebook import/configuration repair
  and clearing stale failed-run outputs; it forbids producer/output changes.
  The worker returned DONE: only the notebook changed, with explicit `import
  snn`, an editable-package-derived absolute pipeline root, and stale failed
  notebook outputs cleared. From a non-repository cwd it loaded the current
  metadata at the expected padded root; structural audit passed; focused tests
  passed 19; full pytest passed 527 with one skip; and `git diff --check`
  passed. Fresh verifier returned FAIL despite functional acceptance: it found
  an unrelated empty markdown cell plus execution/kernel metadata changes that
  violate the frozen source-preservation boundary. The path repair and all
  data/training contracts passed. Next action: return those narrow findings to
  the same worker for notebook-only repair, then request a fresh verification.
  Repair worker returned DONE: the empty cell and unrelated execution/kernel
  metadata are restored to baseline; the notebook diff is now limited to the
  `snn` import, absolute root expression, and stale-output cleanup. Structural
  audit, focused tests (19 passed), full pytest (527 passed/1 skipped), and
  `git diff --check` passed. Fresh verifier PASS: the final diff is limited to
  the explicit `snn` import and package-derived root expression; no stale
  errors, metadata churn, empty cells, producer mutation, or training-contract
  change remains. The data cells executed from `/tmp` with split lengths
  1,751/349/300 and 52 classes. T003 is DONE.
- User requested selectable post-encode-transform channel views. Initial
  T003A probe returned REVISE: `AbsRectify` is an upstream representation, not
  a plotting operation. The draft freezes user-facing `"None"`/`"AbsRectify"`
  selection, normalization to metadata semantics, and the observed roots
  `outputs/action0_pipeline`/`outputs/action0_rectified` so datasets, view,
  and training share one representation. Next action: fresh probe of this
  revised contract. Fresh probe CONFIRMED the precise two-root map, strict
  option/absence handling, direct dataset plot, and pre-construction selection
  order. Worker returned DONE with only the notebook changed: strict
  `"None"`/`"AbsRectify"` selection, repository-root-derived observed bases,
  missing-root failure, display/rerun guidance, and unchanged direct dataset
  plotting. Both choices ran from `/tmp`; each loaded 21-channel/1,024-frame/
  200 Hz data with 1,751/349/300 split lengths, and AbsRectify was nonnegative
  while signed data retained negatives. Focused tests passed 19; full pytest
  passed 527 with one skip; JSON audit and `git diff --check` passed. Fresh
  verifier PASS: exact choices/roots/early errors and shared view-training
  representation all hold; signed samples retain negatives while AbsRectify
  samples are nonnegative. T003A is DONE. T004–T005 remain DRAFT and are
  outside this user-requested transform-selection change.
