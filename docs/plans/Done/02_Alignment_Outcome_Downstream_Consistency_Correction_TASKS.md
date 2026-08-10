# Alignment Outcome Downstream Consistency Correction — TaskSpecs

## Plan state

- Source plan: `docs/plans/TODO/02_Alignment_Outcome_Downstream_Consistency_Correction_Plan.md`
- Baseline: `a1cd88be4ab9d9f4bca3babd1019da0adc5c1ed8`
- Status: active

```text
T6 Downstream outcome dependency contract
        |
        +------------------+
        |                  |
        v                  v
T7 Action0 stale     T8 SpikeIMU skip/rate
reuse correction     correction
        \                  /
         \                /
          v              v
        T9 End-to-end transition verification
```

| Task | State | Depends on | Evidence |
| --- | --- | --- | --- |
| T6 | DONE | — | Probe; worker; fresh verifier PASS; 43 focused / 572 full passed, 1 skipped. |
| T7 | DONE | T6 | Revised probe; worker; fresh verifier PASS; bash -n / 30 focused / 581 full passed, 1 skipped. |
| T8 | DONE | T6 | Probe; worker; fresh verifier PASS; 45 focused / 574 full passed, 1 skipped. |
| T9 | DONE | T7, T8 | PRIMARY validation and fresh final verifier PASS; bash -n / 75 focused / 581 full passed, 1 skipped. |

## T6 — Downstream outcome dependency contract (FROZEN)

- Goal: publish a deterministic, validated alignment-outcome dependency in
  every aligned-Board segmentation summary, so consumers can prove it matches
  the current source/SUCCESS/SKIPPED partition rather than merely trusting
  structural artifact validity.
- Source plan: `02_Alignment_Outcome_Downstream_Consistency_Correction_Plan.md`.
- Dependencies: none.
- Validated against commit:
  `a1cd88be4ab9d9f4bca3babd1019da0adc5c1ed8`.
- Allowed write paths:

  ```text
  src/writingring/board_event_segmentation.py
  tests/test_board_event_segmentation.py
  tests/test_spike_imu_segmentation.py
  ```

- Forbidden write paths: every other path, including `docs/**`, `README.md`,
  `scripts/**`, configuration, `vendor/**`, and `data_sample/**`.
- Required behavior:

  1. In `boundary-mode=aligned-board-events` summaries only, add an
     `alignment_outcome_dependency` object with this exact additive shape:

     ```json
     {
       "source_recording_ids": [
         {"user": "user_0", "action": "0", "dataset_id": 0}
       ],
       "outcomes_by_status": {
         "SUCCESS": [
           {
             "identity": {"user": "user_0", "action": "0", "dataset_id": 0},
             "report_sha256": "64 lowercase hexadecimal characters"
           }
         ],
         "SKIPPED": [
           {
             "identity": {"user": "user_0", "action": "0", "dataset_id": 1},
             "reason": "validated skip reason",
             "report_sha256": "64 lowercase hexadecimal characters"
           }
         ]
       }
     }
     ```

  2. Derive `source_recording_ids` from the authoritative selected
     `Recording` objects, and derive each status entry only after
     `validate_alignment_outcome()` has validated current feature and Board
     provenance.  Use `sha256_file(outcome.paths.report_path)` for the report
     digest.  Do not read raw report JSON or infer anything from offset files.
  3. Lists use the established numeric `dataset_id` recording order.  The
     source IDs equal the disjoint union of the two status partitions;
     `SUCCESS` contains exactly processed recordings, and `SKIPPED` contains
     exactly validated recording skips.
  4. Preserve the output schema version and all existing summary fields.
     `recording_skips` remains the existing human/audit record; it and the
     label-level `skipped_segment_count` remain independent from the new
     dependency object.  Label-mode summaries must not require or emit this
     aligned-only contract.
  5. Do not change alignment artifact schemas, outcome decisions, skip
     reasons, Board sorting/repair, segmentation or padding algorithms, or
     resume behavior.  Legacy aligned summaries lacking this object are not a
     compatibility obligation for this task; T7 will treat them as stale.
- Acceptance criteria:

  - mixed `SUCCESS`/`SKIPPED` raw- and SpikeIMU aligned segmentation summaries
    assert the complete source list, both status lists, skip reason, report
    digest format/value, deterministic numeric ordering, and partition/count
    invariants;
  - tests show recording-level skips remain distinct from
    `skipped_segment_count`;
  - existing label-mode summary behavior remains covered and unchanged;
  - all existing focused segmentation tests pass, and the full Python 3.11
    pytest suite passes in `writingring-gpu` (fall back to `writingring-viz`
    only if `writingring-gpu` is unavailable).
- Validation commands:

  ```bash
  conda run --no-capture-output -n writingring-gpu python -m pytest -q \
    tests/test_board_event_segmentation.py tests/test_spike_imu_segmentation.py
  conda run --no-capture-output -n writingring-gpu python -m pytest -q
  ```

- Replan triggers: no available public validated report path/digest; required
  summary shape conflicts with an established consumer; or satisfying the
  contract requires writes outside the allowed paths.

#### Completion record

The frozen additive contract was implemented only in the authorized
segmentation module and tests.  A fresh verifier PASS confirmed field shape,
validated derivation, partition/order/count invariants, skip independence,
unchanged schema/version, label-mode exclusion, and Action0 consumer
sufficiency.  Focused pytest passed 43 with 1 skipped; full Python 3.11 pytest
passed 572 with 1 skipped in `writingring-gpu`.

## T7 — Action0 stale downstream reuse correction (FROZEN)

- Goal: make aligned-Board Action0 `continue` and final QA reconcile T6's
  exact outcome dependency, so stale segmentation invalidates padding
  transitively while leaving valid alignment reusable.
- Source plan: `02_Alignment_Outcome_Downstream_Consistency_Correction_Plan.md`.
- Dependencies: T6 DONE.
- Validated against commit:
  `a1cd88be4ab9d9f4bca3babd1019da0adc5c1ed8`, plus T6's independently
  verified, uncommitted summary-contract changes.
- Allowed write paths:

  ```text
  scripts/action0_pipeline/_common.bash
  tests/test_action0_pipeline_scripts.py
  ```

- Forbidden write paths: every other path, including `docs/**`, `README.md`,
  `src/**`, configuration, `vendor/**`, and `data_sample/**`.
- Required behavior:

  1. For each selected aligned user, recompute T6's exact
     `alignment_outcome_dependency` in an inline Python helper called by
     `_common.bash`.  It must use numeric `discover_recordings()` order,
     current SpikeIMU feature and numeric Board provenance,
     `validate_alignment_outcome()`, `read_alignment_skip_artifact()` for
     SKIPPED reasons, and `sha256_file(outcome.paths.report_path)`.  Bash must
     consume only the helper result and must not parse outcome artifacts.
  2. In continue planning, first retain the existing alignment validation and
     structural segmentation validation.  If the summary is unreadable,
     missing, or a non-object, retain the existing structural-invalid/full
     preprocess-rebuild policy.  If it is structurally readable but its
     `alignment_outcome_dependency` is missing, legacy, malformed, or unequal
     to the current exact object, classify it as downstream stale—not alignment
     stale.
  3. For downstream stale state, do not rerun preprocess, encoding, or
     alignment.  Plan from `segment`, replace the existing segmentation output
     and rerun padding.  Use a dedicated internal downstream-overwrite path so
     only the segmentation and padding publishers receive their existing
     overwrite flags; do not set or broaden the external/global `OVERWRITE`
     semantics.  `pipeline_execute_from_stage segment` must therefore publish
     both fresh segmentation and fresh padded artifacts successfully.
  4. Final aligned QA must perform the identical dependency reconciliation
     after current outcome validation/counting and before reporting success.
     It must reject missing, malformed, or unequal dependency fields.  Padding
     remains transitively invalidated via segmentation; do not add alignment
     dependency schema to padding outputs or alter standalone padding behavior.
  5. Preserve all valid current stage-level resume, validation, and outcome
     failure behavior.  Do not add per-recording resume or modify alignment
     artifact/outcome schema.
- Acceptance criteria:

  - planning coverage proves each of `SUCCESS→SKIPPED`, `SKIPPED→SUCCESS`, and
    a valid report-byte/digest change avoids `PIPELINE_RESUME_STAGE=complete`,
    selects `segment`, and causes only segmentation/padding overwrite calls;
  - the `SUCCESS→SKIPPED` regression retains old segmentation/padding until
    planning, then runs fresh segmentation/padding and ultimately exposes only
    the remaining SUCCESS recording; T9 will independently exercise the
    end-to-end artifact contents;
  - final QA repeats the comparison, catching a stale dependency that planning
    did not (or could not) retain;
  - readable legacy/malformed dependency versus unreadable/non-object summary
    have the distinct specified rebuild outcomes;
  - focused Action0 tests and the full Python 3.11 pytest suite pass in
    `writingring-gpu` (fall back to `writingring-viz` only if unavailable).
- Validation commands:

  ```bash
  conda run --no-capture-output -n writingring-gpu python -m pytest -q \
    tests/test_action0_pipeline_scripts.py
  conda run --no-capture-output -n writingring-gpu python -m pytest -q
  ```

- Replan triggers: the public outcome contract cannot rebuild the exact T6
  object; a required behavior needs a write outside the allowed paths; or
  preserving publisher safety requires a product-level resume-policy change.

#### Completion record

The frozen public-contract reconciliation and dedicated downstream overwrite
path were implemented in the two authorized Action0 paths. A fresh verifier
PASS confirmed numeric exact-object construction, dependency versus structural
staleness classification, segment/padding-only replacement, repeated final-QA
validation, and unchanged global/standalone behavior. `bash -n` and focused
pytest passed 30; full Python 3.11 pytest passed 581 with 1 skipped in
`writingring-gpu`.

## T8 — SpikeIMU SKIPPED sampling-rate correction (FROZEN)

- Goal: prevent a final recording-level `SKIPPED` SpikeIMU outcome from
  failing requested sampling-rate validation before outcome classification,
  while preserving all provenance and `SUCCESS` rate guarantees.
- Source plan: `02_Alignment_Outcome_Downstream_Consistency_Correction_Plan.md`.
- Dependencies: T6 DONE.
- Validated against commit:
  `a1cd88be4ab9d9f4bca3babd1019da0adc5c1ed8`, plus T6's independently
  verified, uncommitted summary-contract changes.
- Allowed write paths:

  ```text
  src/writingring/board_event_segmentation.py
  tests/test_spike_imu_segmentation.py
  ```

- Forbidden write paths: every other path, including `docs/**`, `README.md`,
  `scripts/**`, configuration, `vendor/**`, and `data_sample/**`.
- Required behavior:

  1. When aligned Board segmentation uses SpikeIMU inputs, load each feature
     artifact sufficiently to validate its canonical identity, schema/content,
     metadata/timestamp provenance, and metadata rate, but do not reject it
     against the caller's requested sampling rate before outcome validation.
  2. Load Board provenance and validate the current alignment outcome.  A
     validated `SKIPPED` recording must still validate feature and Board
     provenance and retain the current skip handling, then exit before labels,
     Board events, verification, segment aggregation, caller-requested rate
     rejection, and common-rate aggregation.
  3. For every final `SUCCESS`, apply the existing requested sampling-rate
     semantics exactly: finite positive request and exact absolute tolerance
     `1e-12`.  Preserve the existing SUCCESS-only common-rate check using
     `validate_common_feature_sampling_rate()` and the same tolerance.
  4. Preserve T6's aligned summary dependency fields unchanged, all existing
     output/cleanup behavior, numeric discovery ordering, and failure behavior
     for malformed/nonpositive metadata rates and stale hashes.  Do not move
     mixed-SUCCESS common-rate failure into a new preflight stage.
- Acceptance criteria:

  - with requested rate 200 Hz, `SUCCESS@200 Hz + SKIPPED@100 Hz` succeeds
    with `processed_recording_count == 1`, `skipped_recording_count == 1`, and
    top-level `sampling_rate_hz == 200`;
  - `SUCCESS@200 Hz + SUCCESS@100 Hz` still fails;
  - a sole `SUCCESS` whose rate differs from the requested rate still fails;
  - existing outcome/provenance validation and T6 dependency-summary tests
    remain green;
  - focused SpikeIMU tests and the full Python 3.11 pytest suite pass in
    `writingring-gpu` (fall back to `writingring-viz` only if unavailable).
- Validation commands:

  ```bash
  conda run --no-capture-output -n writingring-gpu python -m pytest -q \
    tests/test_spike_imu_segmentation.py
  conda run --no-capture-output -n writingring-gpu python -m pytest -q
  ```

- Replan triggers: a change outside the allowed paths is required; deferring
  the caller-requested check also defers malformed/nonpositive rate or
  provenance errors; or a required consumer contract differs from T6.

#### Completion record

The frozen SUCCESS-only requested-rate order was implemented within the two
authorized paths. A fresh verifier PASS confirmed preserved feature/Board
provenance and rate validity, SKIPPED exit ordering, exact SUCCESS requested
and common-rate semantics, compatible T6 summary output, and cleanup/order
behavior. Focused pytest passed 45 with 1 skipped; full Python 3.11 pytest
passed 574 with 1 skipped in `writingring-gpu`.

## T9 — End-to-end outcome-transition verification (FROZEN)

- Goal: independently validate that outcome transitions propagate through
  Action0 continuation, segmentation, padding, and QA, then leave all durable
  state consistent with the verified implementation.
- Source plan: `02_Alignment_Outcome_Downstream_Consistency_Correction_Plan.md`.
- Dependencies: T7 DONE; T8 DONE.
- Owner: PRIMARY validation/documentation, followed by a fresh
  `luna_verifier`. This is not an implementation task and has no worker write
  scope.
- Required validation:

  ```bash
  bash -n scripts/action0_pipeline/_common.bash
  conda run --no-capture-output -n writingring-gpu python -m pytest -q \
    tests/test_action0_pipeline_scripts.py \
    tests/test_board_event_segmentation.py \
    tests/test_spike_imu_segmentation.py
  conda run --no-capture-output -n writingring-gpu python -m pytest -q
  git diff --check
  ```

  Use `writingring-viz` only if `writingring-gpu` is unavailable.
- Acceptance criteria:

  - focused evidence covers `SUCCESS + SUCCESS -> SUCCESS + SKIPPED`, the
    reverse transition, and a report-digest-only mutation; each rejects stale
    downstream state, starts at `segment`, regenerates padding, and final QA
    applies the same contract;
  - SpikeIMU `SUCCESS@200 + SKIPPED@100` at requested 200 succeeds with only
    one processed recording, whereas mixed/sole mismatched SUCCESS rates fail;
  - no stale downstream package can pass final QA; stage-level resume remains
    intact and no padding schema or standalone behavior changes;
  - focused and full Python 3.11 pytest pass, durable notes agree with code,
    and the plan/task/workboard state accurately records the lifecycle;
  - a fresh final verifier PASS confirms the complete T6–T9 contract.
- Allowed PRIMARY write paths: `docs/plans/**` and affected `docs/notes/**`.
- Forbidden paths: implementation, tests, scripts, configuration,
  `vendor/**`, and `data_sample/**`.
- Replan triggers: test or final verification exposes an unmet behavior
  requiring an implementation change outside an already completed frozen
  TaskSpec, or a product-level decision is required.

#### Completion record

PRIMARY ran `bash -n`, the frozen cross-module regression suite (75 passed, 1
skipped), the full Python 3.11 suite (581 passed, 1 skipped), and `git diff
--check` in `writingring-gpu`. The affected segmentation, Action0, and
SpikeIMU durable notes were reconciled. A fresh final verifier PASS confirmed
the complete T6–T9 dependency, targeted-rebuild, rate-ordering, transition,
scope, documentation, and lifecycle contract.
