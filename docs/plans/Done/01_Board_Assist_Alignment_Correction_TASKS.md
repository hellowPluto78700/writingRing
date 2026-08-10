# Board-Assisted Alignment Outcome Follow-up Correction — TaskSpecs

## Plan state

- Source plan: `docs/plans/TODO/01_Board_Assist_Alignment_Correction_Plan.md`
- Original baseline: `a69476c4f3ae713335b6867678cd35c48a04ef3d`
- Follow-up baseline: `f0e519d16fcd59f774f2000c8fe737ff3ac09763`
- Status: active final validation and documentation

```text
T1R Event-prefix correction
        |
        v
T2R Outcome-contract correction
      /   \
     v     v
T3 Action0  T4 Board segmentation
      \   /
       v v
T5 End-to-end verification and state cleanup
```

| Task | State | Depends on | Evidence |
| --- | --- | --- | --- |
| T1R | DONE | — | revised probe; worker; fresh verifier PASS |
| T2R | DONE | T1R | probe; repair worker; fresh verifier PASS |
| T3 | DONE | T2R | probe; worker; fresh verifier PASS |
| T4 | DONE | T2R | revised probe; worker; fresh verifier PASS |
| T5 | DONE | T3, T4 | PRIMARY validation/documentation; fresh final verifier PASS |

This file intentionally retains only the active follow-up lifecycle. The
superseded initial T1/T2 TaskSpecs and their stale downstream outlines have
been removed; their material verified contracts appear below.

## T1R — Event-prefix correction (DONE)

- Goal: only globally valid press/lift pairs whose two endpoints are inside
  the initial positional monotonic prefix, and only their eligible events,
  may participate in alignment.
- Scope: `src/writingring/event_alignment.py` and
  `tests/test_event_alignment.py`.
- Corrected contract: derive usable pair IDs from prefix-filtered valid pairs
  and allowlist only `valid_touch` events with those IDs. A cross-boundary pair
  contributes neither endpoint; invalid/transient audit rows and timestamp-only
  fixture fallback remain intact. Later timestamp overlap cannot reintroduce
  frames, identity-bearing contacts, events, or pairs.
- Evidence: initial probe `REVISE`, revised probe `CONFIRMED`, worker DONE,
  fresh verifier PASS. Focused pytest: 52 passed. Full pytest then: 543 passed,
  1 skipped in `writingring-gpu`.
- Durable note: `docs/notes/ALIGNMENT_OUTPUTS.md` records this prefix-event
  boundary.

## T2R — Outcome-contract correction (DONE)

- Goal: correct terminal-state authorization, normalize malformed outcomes,
  and prove the semantics of a SKIPPED artifact.
- Scope: `src/writingring/alignment_io.py` and
  `tests/test_alignment_io.py`.
- Corrected contract: only SUCCESS↔SKIPPED needs `overwrite_outcome`.
  FAILED→SUCCESS/SKIPPED uses ordinary target overwrite flags, cleans stale
  opposite artifacts, and ends conflict-free. Public malformed outcome/skip
  reads—including absent manifests, wrong types, conversion overflow, and
  underlying export errors—raise `AlignmentOutcomeError`. A skip proves its
  eight diagnostics, positive global pairs, zero usable prefix pairs, a raw
  backward jump, and contiguous prefix/jump identities.
- Evidence: probe `CONFIRMED`; one verifier FAIL repaired in scope; fresh
  verifier PASS. Focused pytest: 59 passed, 1 skipped. Full pytest then:
  558 passed, 1 skipped in `writingring-gpu`.
- Durable note: `docs/notes/ALIGNMENT_OUTPUTS.md` records completion,
  publication, overwrite, and error contracts.

## T3 — Action0 outcome integration (DONE)

- Goal: Action0 aligned-board stages consume only the T2R Python completed
  outcome contract.
- Scope: `scripts/action0_pipeline/_common.bash` and
  `tests/test_action0_pipeline_scripts.py`.
- Corrected contract: alignment uses `--initial-interval-policy skip`; rebuild
  authorization includes `--overwrite-outcome`; stage validation and QA invoke
  `validate_alignment_outcome` with current SpikeIMU and numeric Board
  provenance for every authoritative recording. Bash does not parse outcome
  artifacts and only counts returned SUCCESS/SKIPPED status. Missing, stale,
  malformed, conflicting, and FAILED outcomes invalidate the whole stage;
  continue remains stage-level rebuild with no per-recording resume.
- Evidence: probe `CONFIRMED`, worker DONE, fresh verifier PASS. Focused
  pytest: 23 passed. Stable full pytest: 571 passed, 1 skipped.
- Durable note: `docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md` records skip-aware
  outcome validation and QA accounting.

## T4 — Board segmentation outcome integration (DONE)

- Goal: validated SUCCESS records segment normally; validated SKIPPED records
  are provenance-checked then omitted; every other outcome is a hard failure.
- Scope: `src/writingring/board_event_segmentation.py`,
  `tests/test_board_event_segmentation.py`, and
  `tests/test_spike_imu_segmentation.py`.
- Corrected contract: given `<alignment_root>/offsets`, derive sibling
  `reports` and `verification` roots. Load feature and numeric Board
  provenance before status validation, but defer labels, Board event work,
  rate aggregation, and aggregate artifacts until SUCCESS. SKIPPED details use
  the public skip reader. Zero SUCCESS recordings fails before empty aggregate
  access. Summary contains `source_recording_count`,
  `processed_recording_count`, `skipped_recording_count`, `recording_skips`,
  and preserved `skipped_segment_count`.
- Evidence: initial probe `REVISE`, revised probe `CONFIRMED`, worker DONE,
  fresh verifier PASS. Focused pytest: 42 passed, 1 skipped. Stable full
  pytest: 571 passed, 1 skipped.
- Durable note: `docs/notes/BOARD_EVENT_GUIDED_SEGMENTATION.md` records
  sibling roots, skip behavior, all-skipped failure, and summary accounting.

## T5 — End-to-end verification and orchestration cleanup (DONE)

- Dependencies: T3 DONE; T4 DONE.
- Owner: PRIMARY validation/documentation, followed by a fresh
  `luna_verifier`. This is not an implementation task and has no worker write
  scope.
- Goal: verify the complete aligned-board contract and leave durable state
  consistent with the actual lifecycle.
- Required validation:

  ```bash
  conda run --no-capture-output -n writingring-gpu python -m pytest -q \
    tests/test_event_alignment.py tests/test_alignment_io.py \
    tests/test_action0_pipeline_scripts.py tests/test_board_event_segmentation.py \
    tests/test_spike_imu_segmentation.py
  conda run --no-capture-output -n writingring-gpu python -m pytest -q
  ```

  Use `writingring-viz` only if `writingring-gpu` is unavailable.
- Acceptance criteria:

  - regression evidence covers mixed SUCCESS/SKIPPED through Action0 QA,
    segmentation, and padding-facing outputs;
  - non-allowed skip, stale, partial, FAILED, and conflicting outcomes hard
    fail at their consumer boundary;
  - full Python 3.11 pytest passes;
  - `ALIGNMENT_OUTPUTS.md`, `SEGMENTATIONS_BASH_SCRIPTS.md`, and
    `BOARD_EVENT_GUIDED_SEGMENTATION.md` agree with implementation;
  - this TaskSpec and `WORKBOARD.md` have one unambiguous active lifecycle and
    no superseded TaskSpecs or stale DRAFT state;
  - a fresh final verifier PASS confirms the DAG contract.
- Allowed PRIMARY write paths: `docs/plans/**` and affected `docs/notes/**`.
- Forbidden paths: implementation, tests, scripts, configuration,
  `vendor/**`, and `data_sample/**`.
- Replan triggers: end-to-end evidence exposes an unmet behavior that requires
  implementation changes outside a completed task's frozen scope.

#### Completion record

PRIMARY ran the frozen combined regression suite (137 passed, 1 skipped) and
full Python 3.11 pytest (571 passed, 1 skipped) in `writingring-gpu`; `git
diff --check` passed. The active task file and workboard were reduced to the
real T1R–T5 lifecycle, and the three affected durable notes were reconciled.
Fresh final verification PASS confirmed the complete prefix/outcome/Action0/
segmentation/padding contract, test evidence, scope, and documentation.
