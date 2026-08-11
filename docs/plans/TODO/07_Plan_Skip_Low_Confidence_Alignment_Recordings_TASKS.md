# Board Segmentation Recording Fault Isolation — TaskSpec

## P0 — confirmed probe

The aligned consumer already distinguishes provenance-valid alignment
`SUCCESS` and `SKIPPED` outcomes.  It loads a feature input and Board data,
validates the alignment outcome/provenance, then performs successful-outcome
offset/feature validation, label/event preparation, Board segmentation, and
verification before transactionally aggregating and publishing one user/action
package.  Any exception in the latter per-recording sequence currently aborts
the entire user/action; an all-`SKIPPED` alignment user deliberately remains a
hard failure before aggregation.

The Action0 wrapper runs one aligned segmentation CLI per user.  Its resume
and QA rules currently require one package and one padded package per user,
so they interpret a report-only all-error user as a partial stage.  Padding
discovers only completed variable-length packages, so it needs no algorithmic
change if report-only error states do not look like packages.

## F0 — FROZEN TaskSpec

### Goal

Add opt-in recording-level isolation for failures that arise while preparing
the Board-event segmentation of an alignment-`SUCCESS` recording.  A failed
recording is a new segmentation terminal state; it is never rewritten as an
alignment `SKIPPED` outcome.  Action0 opts into this policy, continues through
all users, and produces reconcilable per-user and run-level reports.

### Required behavior

- `BoardEventSegmentationConfig` and the aligned CLI expose
  `recording_error_policy=error|skip`, defaulting to strict `error`.
  Label mode does not accept this option.  Action0 aligned mode explicitly
  passes `skip`.
- Only a known, recording-local Board segmentation validation failure after a
  provenance-valid alignment `SUCCESS` may be quarantined.  The permitted
  stages are timestamp-label interpretation, Board event preparation, Board
  event-table alignment, and Board boundary/slice construction.  The report
  records identity, `stage`, `error_type`, and message.
- Alignment outcome/provenance validation, malformed/stale success artifacts,
  feature/source loading, common-rate validation, verification rendering or
  PNG publication, aggregate invariants, aggregate publication/filesystem
  failures, and unexpected exceptions remain hard failures.  A caught
  recording-local error with an `OSError` or alignment-artifact error in its
  cause chain remains a hard failure.
- A mixed user/action publishes its normal package containing only processed
  recordings, plus a separate recording-error sidecar.  Its summary records
  processed, alignment-skipped, and segmentation-error counts and details;
  `source_recording_count == processed_recording_count +
  skipped_recording_count + segmentation_error_recording_count`.
- `alignment_outcome_dependency` remains a pure alignment contract: its
  `SUCCESS` list includes both processed and segmentation-error recordings,
  and its `SKIPPED` list includes only validated alignment skips.  Its union
  is the discovered source list.
- If an opt-in user has no processed recording solely because one or more
  alignment-`SUCCESS` recordings reached the permitted segmentation-error
  state, publish no empty variable-length package.  Publish a report-only
  terminal user state under the segmentation root instead.  Retain the
  existing hard failure for a user containing only alignment `SKIPPED`
  outcomes.
- Action0 writes a root JSON report and CSV after every user has completed.
  It contains each user/action terminal state and every error with at least
  `user`, `action`, `dataset_id`, `stage`, `error_type`, and `message`.
  Report-only users are excluded from length analysis, padding, and training
  inputs.
- Continue-mode and QA recognize valid report-only users.  They require
  packages only for users with processed recordings, validate count
  reconciliation, and compare padding packages with successful user/action
  packages rather than all discovered users.  Missing/malformed states remain
  structural invalidity; current full-rebuild behavior is retained.

### Contracts intentionally preserved

Board boundaries, segment slicing, alignment matching/classification,
alignment `SUCCESS`/`SKIPPED` publication, preprocessing, Spike encoding, and
padding algorithms are unchanged.  Standalone aligned segmentation stays
strict unless the new option is supplied.

### Allowed write paths

- `src/writingring/board_event_segmentation.py`
- `scripts/segment_ring_imu.py`
- `scripts/action0_pipeline/_common.bash`
- `tests/test_board_event_segmentation.py`
- `tests/test_segment_ring_imu_cli.py`
- `tests/test_action0_pipeline_scripts.py`
- `docs/notes/BOARD_EVENT_GUIDED_SEGMENTATION.md`
- `docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md`
- this TaskSpec and `docs/plans/WORKBOARD.md`

No files below `data_sample/**` or `vendor/**` may be changed or inspected.

### Validation

- Focused Board segmentation, CLI, and Action0-wrapper tests.
- Bash syntax check and `git diff --check`.
- Relevant padding tests to verify that completed-package-only discovery still
  excludes report-only terminal users.
- A regression run that keeps existing strict hard-failure behavior.

### Replan triggers

Any need to change alignment statuses or provenance validation, Board boundary
or padding algorithms, allow arbitrary exceptions to be swallowed, or add a
consumer-facing empty segmentation package requires replanning.
