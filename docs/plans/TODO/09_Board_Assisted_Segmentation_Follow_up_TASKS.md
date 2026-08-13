# Board-Assisted Segmentation Follow-up — Frozen TaskSpecs

**Lifecycle:** COMPLETE — T1 through T5 independently verified after the T2
revision-2 fixture-scope replan.

This file records only frozen, dependency-ready work. The Plan remains the
source of task ordering. A worker must return `NEEDS_REPLAN` instead of
expanding an allowed write scope or changing a contract below.

## T1 — CLI/config integration

- **Status:** FROZEN
- **Dependencies:** T0 complete.
- **Allowed writes:** `scripts/segment_ring_imu.py`,
  `tests/test_segment_ring_imu_cli.py`.
- **Required behavior:** In `aligned-board-events` mode, expose and explicitly
  forward Board-only final-segment maximum duration (default `5.0` seconds)
  and carry-in press lookback (default `0.2` seconds) to
  `BoardEventSegmentationConfig`, converting seconds to the existing
  microsecond configuration fields. Continue explicitly forwarding the
  `0.2`-second pre-press and post-lift contexts. The label-mode construction
  path and behavior must remain unchanged; the two Board-only options are
  invalid in label mode.
- **Acceptance:** Focused CLI tests prove default forwarding, explicit
  overrides, and the existing label-mode behavior. No changes to Board core
  logic or Action0 scripts.
- **Validation:** `python -m pytest tests/test_segment_ring_imu_cli.py` in the
  preferred usable project Conda environment.
- **Replan triggers:** A required behavior cannot be implemented within the
  allowed files; current core config lacks the receiving fields; or preserving
  label mode requires a contract change.

## T2 — Core Board-segmentation regression tests

- **Status:** REPLANNED AND FROZEN (revision 2)
- **Dependencies:** T0 complete.
- **Allowed writes:** `tests/test_board_event_segmentation.py`,
  `tests/test_spike_imu_segmentation.py`, and only if required for a
  pre-existing label-only assertion, `tests/test_segmentation.py`. Do not modify
  `src/writingring/board_event_segmentation.py` unless verification establishes
  a core defect and PRIMARY replans.
- **Required behavior:** Cover the Board-only replacement contract: a label
  gap greater than five seconds can export when the final Board window is at
  most five seconds; a longer final Board window skips with
  `final_segment_duration_gt_5s`; carry-in ownership, normal start, and
  midpoint start; no carry-in for a preceding lift or a press older than the
  `0.2`-second lookback; empty-leading recovery searches strictly from the
  label to the first Board frame, selects the most-prominent alignment-grade
  peak, and has no fallback; every exported Board sample has duration at most
  five seconds. Keep the label-only greater-than-five-second rule covered.
- **Acceptance:** Stale crossing-touch expectations are revised to distinguish
  the new carry-in case from historical crossing cases. Board fixtures provide
  the validation metadata needed by the current production contract. Tests do
  not alter alignment, loader, timestamp, or channel semantics.
- **Validation:** `python -m pytest tests/test_board_event_segmentation.py
  tests/test_segmentation.py tests/test_spike_imu_segmentation.py` in the
  preferred usable project Conda environment.
- **Replan triggers:** A test exposes a core T0 defect; coverage requires a
  contract change or a file beyond the allowed scope; or fixture behavior
  reveals undocumented Board data semantics.

### T2 revision 2 — SpikeIMU Board fixture compatibility

T5 final verification found four aligned SpikeIMU tests whose `load_board`
mocks lack the `BoardData.validation.empty_leading_chunk_indices` metadata
required by the completed T0 core contract. This is a fixture-contract repair,
not a production behavior change. Update those test fixtures to provide the
appropriate empty-leading index metadata, preserve their existing intended
rate/export assertions, and run the amended T2 validation. No other scope or
contract changes are authorized.

## T3 — Action0 integration

- **Status:** FROZEN
- **Dependencies:** T1 independently verified.
- **Allowed writes:** `scripts/bash_script/action0_pipeline/_common.bash`,
  `tests/test_action0_pipeline_scripts.py`.
- **Required behavior:** The shared `pipeline_segment` aligned-Board argument
  branch explicitly passes `--maximum-segment-duration-seconds 5.0` and
  `--carry-in-press-lookback-seconds 0.2` beside the existing `0.2`-second
  context arguments. This one shared change must cover raw, low-pass,
  Madgwick, and Xylo aligned-Board wrappers. The separate label argument
  branch must not receive Board-only flags. Repair the focused test's stale
  `scripts/action0_pipeline` path to the repository's existing
  `scripts/bash_script/action0_pipeline` location, then assert aligned versus
  label command construction.
- **Acceptance:** All four aligned wrappers share the new explicit contract;
  all label wrappers remain unchanged; focused Action0 script tests pass.
- **Validation:** `bash -n scripts/bash_script/action0_pipeline/_common.bash`
  and `python -m pytest tests/test_action0_pipeline_scripts.py` in the
  preferred usable project Conda environment.
- **Replan triggers:** Correct integration requires modifying individual
  wrappers, CLI/core behavior, another script tree, or an Action0 contract
  beyond the stated Board-only parameters.

## T4 — Documentation contract update

- **Status:** FROZEN
- **Dependencies:** T1, T2, and T3 independently verified.
- **Executor:** PRIMARY (documentation-only task; Luna does not author project
  or orchestration documentation).
- **Allowed writes:** `docs/notes/BOARD_EVENT_GUIDED_SEGMENTATION.md`,
  `docs/notes/IMU_SEGMENTATION.md`,
  `docs/notes/SPIKE_SEGMENTATION_PIPELINE.md`, and
  `docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md`.
- **Required behavior:** Replace obsolete Board-mode label-gap and
  press-timestamp-ownership wording with the verified final-window, carry-in,
  and empty-leading recovery contract. Scope the old greater-than-five-second
  label-gap rule to label mode. Record that SpikeIMU recovery scores only
  channels `15:21`, searches strictly between the label and first Board frame,
  uses the alignment-grade peak detector and documented selection tie-break,
  and has no fallback. State Action0's explicit shared aligned-Board arguments
  (contexts `0.2`, maximum duration `5.0`, carry-in lookback `0.2`) while
  retaining the label-branch distinction.
- **Preserved facts:** Alignment offset creation, offset/time-domain,
  canonical timestamp, Board loading, channel, and provenance contracts do
  not change.
- **Acceptance:** All four notes agree with current core, CLI, and Action0
  contracts; no documentation makes a new behavioral claim.
- **Validation:** Review the exact modified passages against current source;
  run `git diff --check`.
- **Replan triggers:** Accurate documentation requires a changed implementation
  contract, a non-listed documentation file, or an undocumented assumption.

## T5 — Final verification

- **Status:** FROZEN
- **Dependencies:** T1–T4 independently verified.
- **Executor:** independent Luna verifier (verification-only task; no worker
  write scope).
- **Allowed writes:** none.
- **Required checks:** Confirm the final Board contract across core, CLI,
  Action0, tests, and the four notes: final resolved Board windows are at most
  five seconds while label-only retains its label-interval rule; carry-in and
  empty-leading recovery remain covered; and alignment/Board-loader behavior
  has no regression. Run Bash syntax validation, one combined focused suite,
  the full Python test suite, and `git diff --check`. Treat pre-existing
  notebook backups/deletions and `*.old` files as out of scope.
- **Acceptance:** All required validation passes and the modified
  documentation agrees with the source. Optional Xylo tests may be skipped
  when their optional runtime is unavailable; this is not a Plan 09 failure.
- **Replan triggers:** A final test exposes a defect requiring a changed TaskSpec,
  contract, or write scope. A normal implementation/test repair stays within
  the task that owns it; halt and report it rather than silently expanding T5.
