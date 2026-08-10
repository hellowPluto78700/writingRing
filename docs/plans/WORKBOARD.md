# Workboard

## Active plan

- **Plan:** `docs/plans/TODO/06_Action0_Notebook_Subset_Experiment_Plan.md`
- **Task file:** `docs/plans/TODO/06_Action0_Notebook_Subset_Experiment_TASKS.md`
- **Baseline commit:** `51293739773d6a192ad8cdec18857d6ec4b52319`
- **Current task:** T002 — Notebook experiment documentation
- **Status:** DONE

## Task state

| Task | State | Depends on |
| --- | --- | --- |
| T001 | DONE | — |
| T002 | DONE | T001 |

## Focus and next action

- A new Action0 notebook subset-experiment plan is active. It preserves CLI
  full-label training while completing the user-selected reproducible
  common-label experiment, explicit figure policy, best-model restoration, and
  final split evaluation. T001 initial probe returned REVISE: it must select
  the positive-label intersection, use non-mutating evaluation mode after
  best-state restoration, fix cwd-relative automatic figure writes, and clear
  stale notebook evidence. Final re-probe CONFIRMED the exact algorithms,
  wrapper invariants, eight-metric results, and default-off figure policy.
  T001 is FROZEN. Worker returned DONE with only the notebook changed:
  reproducible positive-intersection subset mapping and invariants, strict
  best-state restoration, non-mutating final split evaluation, eight-metric
  table/provenance, and default-off repository-anchored figure saving. Both
  transform roots and invalid size handling were verified; focused tests passed
  19, full pytest passed 527 with one skip, direct reduced execution passed,
  and no figures were written by default. Fresh verifier PASS: all frozen
  selection/wrapper/restored-state/final-evaluation/figure-policy rules hold.
  T001 is DONE. T002 updated README and Action0 training documentation to
  distinguish CLI full-label training from the verified notebook subset
  experiment; no producer/encoding/padding/Bash contract changed. Markdown
  linkage and `git diff --check` passed. The requested plan is DONE.

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
