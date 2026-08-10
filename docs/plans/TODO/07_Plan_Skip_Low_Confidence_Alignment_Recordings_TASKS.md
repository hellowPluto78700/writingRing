# Task DAG: Skip Low-Confidence Alignment Recordings

## Plan identity

- **Source plan:** `docs/plans/TODO/07_Plan_Skip_Low_Confidence_Alignment_Recordings.md`
- **Initialized against:** `2b05be416b4893eb8fbf596a6ac29a8e43b639a9`
- **Naming note:** The source plan's requested `03_Alignment_Unalignable_Recording_Skip_*`
  filenames are superseded by this plan's user-selected `07_...` identity.  Do
  not create duplicate `03_...` planning files.

## Dependency graph

```text
T1 (structured confidence diagnostics)
 └─ T2 (semantic SKIPPED validation)
     ├─ T3 (CLI classification and publication)
     │   └─ T4 (Action0 skip-policy integration)
     └─ T5 (downstream SKIPPED consumption)
          └─ T6 (end-to-end validation and durable documentation) ← T4, T5
```

## Current task states

| Task | State | Depends on | Ownership | Scope |
| --- | --- | --- | --- | --- |
| T1 | DONE | — | luna_worker | Probe CONFIRMED; worker DONE; fresh verifier PASS. |
| T2 | DONE | T1 | luna_worker | Revised Probe CONFIRMED; worker DONE; fresh verifier PASS. |
| T3 | DONE | T2 | luna_worker | Revised Probe CONFIRMED; repaired worker DONE; fresh verifier PASS. |
| T4 | DONE | T3 | luna_worker | Probe CONFIRMED; repaired worker DONE; fresh verifier PASS. |
| T5 | DONE | T2 | luna_worker | Probe CONFIRMED; worker DONE; fresh verifier PASS. |
| T6 | BLOCKED | T4, T5 | PRIMARY documentation; Luna verification | Revised acceptance is frozen; persistent full-pipeline terminal evidence remains unavailable. |

Only a dependency-ready task may be probed.  Frozen TaskSpecs are added below
only after their Probe confirms the relevant implementation contract.

## Task drafts

### T1 — Structured alignment confidence diagnostics

- **Goal:** Expose machine-readable confidence-check outcomes from alignment
  without changing thresholds, matching, ranking, or peak detection.
- **Expected implementation surface:** `src/writingring/event_alignment.py`,
  `tests/test_event_alignment.py`.
- **Plan acceptance focus:** classify insufficient valid pairs, low event
  coverage, and the equality boundary; report the named failed checks.

#### FROZEN TaskSpec

- **Task ID:** T1
- **Source plan:** `07_Plan_Skip_Low_Confidence_Alignment_Recordings.md`
- **Validated against commit:** `2b05be416b4893eb8fbf596a6ac29a8e43b639a9`
- **Dependencies:** none
- **Goal:** Make failed alignment confidence checks machine-readable in the
  alignment report while preserving all current matching and success semantics.
- **Required behavior:**
  - Add a report `confidence_checks` mapping with, in this order,
    `minimum_valid_touch_pairs` and `minimum_event_coverage_ratio`. Each
    entry exposes `passed`, `actual`, and `minimum` using the existing
    total-valid-pair count, valid-event coverage ratio, and configured limits.
  - Add `failed_confidence_checks`, ordered as the above mapping, containing
    exactly the names whose existing comparison fails.
  - Preserve the existing warning behavior and `alignment_success` result.
    The current `>=` boundaries remain passing: exact event coverage `0.40`
    passes when the pair threshold is met.
  - Do not change configured thresholds (5 and 0.40), valid-count
    definitions, matching, candidate ranking, peak detection, result
    propagation, or unrelated report fields.
- **Contracts intentionally changed:** The alignment report gains the two
  additive structured diagnostic fields for downstream consumption.
- **Contracts to preserve:** Existing `alignment_success`, warning text and
  conditions, report propagation, and all alignment algorithm behavior.
- **Allowed write paths:**
  - `src/writingring/event_alignment.py`
  - `tests/test_event_alignment.py`
- **Forbidden write paths:** `README.md`, `docs/**`, `AGENTS.md`, `.codex/**`,
  `vendor/**`, `data_sample/**`, and every path outside the allowed set.
- **Acceptance criteria:**
  - Adequate valid pairs plus event coverage below 0.40 gives unsuccessful
    alignment and `failed_confidence_checks == ["minimum_event_coverage_ratio"]`.
  - Three valid pairs with minimum five includes
    `minimum_valid_touch_pairs` in the deterministic failed list.
  - Adequate pairs and coverage exactly 0.40 passes the coverage check and
    retains successful alignment when no other existing condition fails.
  - The pair check uses `total_valid_touch_pair_count`, never
    `fully_matched_touch_pair_count`.
- **Validation commands:**
  - `conda run --no-capture-output -n writingring-gpu python -m pytest tests/test_event_alignment.py`
    (use `writingring-viz` only if `writingring-gpu` is unavailable)
  - `conda run --no-capture-output -n writingring-gpu python -m pytest`
    (same fallback)
  - `git diff --check`
- **Replan triggers:** Any need to change matching/count definitions,
  thresholds, report schema outside the named additive fields, or a write
  outside the allowed paths.

### T2 — Semantic validation for confidence-derived skips

- **Goal:** Add the two confidence-insufficiency skip reasons and validate
  their diagnostics semantically rather than by reason string alone.
- **Expected implementation surface:** `src/writingring/alignment_io.py`,
  `tests/test_alignment_io.py`.
- **Revised draft requirements (Probe revalidation required):**
  - Validate diagnostics by skip reason; retain the exact existing
    `initial_interval_no_usable_pair` diagnostic schema and all current
    provenance/success-artifact rules unchanged.
  - Both new reason-specific schemas carry only the plan-required diagnostics
    plus `minimum_valid_touch_pairs` for `insufficient_event_coverage`; this
    supplies the explicit threshold needed to prove the plan's required
    `total_valid_touch_pair_count >= minimum_valid_touch_pairs` invariant.
  - Require JSON numeric counts to be finite non-negative integers, minima to
    be finite positive integers, ratios to be finite in `[0, 1]`, and totals
    that appear as denominators to be positive. Require every reported
    coverage ratio to agree with its matched/total counts using
    `math.isclose(..., rel_tol=0.0, abs_tol=1e-12)`.
  - Require `failed_confidence_checks` to be an ordered, duplicate-free list
    of the T1 check names. `insufficient_valid_touch_pairs` must contain
    `minimum_valid_touch_pairs`; `insufficient_event_coverage` must equal
    `["minimum_event_coverage_ratio"]`, matching the plan's CLI-only coverage
    classification.
  - Do not change skip schema version or the existing initial-interval
    artifact factory. T3, not T2, owns production of the new artifacts.

#### FROZEN TaskSpec

- **Task ID:** T2
- **Source plan:** `07_Plan_Skip_Low_Confidence_Alignment_Recordings.md`
- **Validated against:** `2b05be416b4893eb8fbf596a6ac29a8e43b639a9` plus T1's uncommitted,
  verifier-passed additive diagnostics.
- **Dependencies:** T1 DONE.
- **Goal:** Permit and semantically prove the two confidence-derived
  `SKIPPED` reasons without weakening existing outcome validation.
- **Required behavior:**
  - Extend accepted completed-outcome reasons with
    `insufficient_valid_touch_pairs` and `insufficient_event_coverage`.
    Dispatch diagnostics validation by reason. Preserve schema version 1,
    top-level fields, provenance/manifest requirements, success-artifact
    exclusion, publication behavior, and the exact eight-key
    `initial_interval_no_usable_pair` schema unchanged.
  - For insufficient pairs require exactly the plan diagnostics and prove
    `0 < total_valid_touch_pair_count < minimum_valid_touch_pairs`.
  - For insufficient coverage require exactly the plan diagnostics plus
    `minimum_valid_touch_pairs`; prove pair sufficiency, positive total-event
    count, `event_coverage_ratio < minimum_event_coverage_ratio`, and ratio
    coherence with matched/total.
  - Validate numeric counts as finite non-boolean non-negative integers,
    integer minima as finite positive integers, ratio/minimum-ratio values as
    finite values in `[0, 1]`, and coverage ratios with
    `math.isclose(rel_tol=0.0, abs_tol=1e-12)`. Validate analogous listed
    press/lift counts and ratios where present.
  - Require ordered duplicate-free T1 check names in
    `failed_confidence_checks`: insufficient-pairs contains its named check;
    insufficient-coverage is exactly `["minimum_event_coverage_ratio"]`.
  - Retain `make_alignment_skip_artifact()` as initial-error-only; T3 will
    construct the new artifact values directly.
- **Allowed write paths:** `src/writingring/alignment_io.py`,
  `tests/test_alignment_io.py`.
- **Forbidden write paths:** `README.md`, `docs/**`, `AGENTS.md`, `.codex/**`,
  `vendor/**`, `data_sample/**`, and all paths outside the allowed list.
- **Acceptance criteria:** Legal artifacts for both new reasons validate;
  coverage at/above minimum, incoherent event or press/lift coverage,
  pair-count at/above its minimum, unknown reason, wrong/missing diagnostics,
  invalid numerics, or malformed failed-check lists fail. Existing legal and
  invalid initial-interval artifacts retain current behavior.
- **Validation commands:** focused `tests/test_alignment_io.py`, full pytest,
  and `git diff --check`, using `writingring-gpu` first and `writingring-viz`
  only if needed.
- **Replan triggers:** any required schema-version/top-level-field change,
  initial diagnostic change, changed provenance/publication behavior, new
  production path outside T3, or writes outside the two permitted files.

### T3 — Explicit CLI unalignable-recording policy

- **Goal:** Add the strict-by-default `error|skip` policy and publish skips
  only for the structured, explicitly permitted classifications.
- **Expected implementation surface:** `scripts/align_ring_board.py` and its
  current CLI test module.
- **Revised draft (re-probe required):** Preserve the legacy
  `--initial-interval-policy` and its initial-error factory behavior. An
  initial-interval exception skips when either legacy policy or the new
  `--unalignable-recording-policy` is `skip`; confidence failures skip only
  under the new policy. Classify only unsuccessful `SequenceAlignmentResult`
  reports, use their structured `failed_confidence_checks`, project exact T2
  reason diagnostics into a directly constructed artifact, and fall through
  to current FAILED/nonzero behavior if construction/validation cannot prove a
  legal skip. Do not broaden arbitrary pre-result exception reporting.

#### FROZEN TaskSpec

- **Task ID:** T3; **Dependencies:** T1/T2 DONE.
- **Goal:** Add strict-by-default `--unalignable-recording-policy error|skip`.
- **Required:** Preserve legacy initial policy/factory; initial typed error
  skips if either policy is skip. For an unsuccessful `SequenceAlignmentResult`
  only, read structured failed list: pair has precedence; coverage requires the
  exact singleton list. With new policy `skip`, project exact T2 diagnostics
  into directly constructed skip artifacts and publish SKIPPED. Any malformed
  diagnostics or other failures use existing FAILED/nonzero behavior. Skips
  have no offset/verification; report and skip JSON remain valid.
- **Allowed:** `scripts/align_ring_board.py`, `tests/test_spike_imu_segmentation.py`.
- **Forbidden:** all other paths, including docs and `alignment_io.py`.
- **Acceptance:** default low coverage is FAILED/nonzero/report-only; explicit
  coverage/pair skips exit 0 with correct reason/no success artifacts;
  unexpected and malformed lists fail; legacy initial path remains valid.
- **Validation:** focused CLI tests, full pytest, diff check. **Replan:** any
  need to alter factory/schema or catch arbitrary pre-result exceptions.

### T4 — Action0 policy integration

- **Goal:** Make aligned Action0 runs opt into the CLI skip policy and retain
  stage/QA behavior for mixed terminal outcomes.
- **Expected implementation surface:** `scripts/action0_pipeline/_common.bash`,
  `tests/test_action0_pipeline_scripts.py`.

#### FROZEN TaskSpec

- **Task ID:** T4; **Dependencies:** T2/T3/T5 DONE.
- **Goal:** Have every aligned Action0 entrypoint explicitly allow all three
  legal recording-level skip outcomes.
- **Required:** In shared `pipeline_align()`, retain
  `--initial-interval-policy skip` and add exactly
  `--unalignable-recording-policy skip`. Preserve all existing outcome
  validator, overwrite, continue, QA, and all-skipped behavior. Tests must
  assert both policy arguments and mixed SUCCESS/SKIPPED continuation without
  changing Python alignment or segmentation production code.
- **Allowed write paths:** `scripts/action0_pipeline/_common.bash`,
  `tests/test_action0_pipeline_scripts.py`.
- **Forbidden:** all other paths, including aligned entrypoint scripts and
  Python sources.
- **Acceptance/validation:** focused Action0 scripts tests, full pytest,
  `bash -n scripts/action0_pipeline/_common.bash`, and diff check.
- **Replan:** entrypoint-specific behavior, validator/QA contract change, or
  any needed write outside the two allowed paths.

### T5 — Downstream consumption of new valid skips

- **Goal:** Confirm Board-event segmentation accepts and omits the two new
  validated SKIPPED outcomes while preserving strict failure for invalid or
  all-skipped input.
- **Expected implementation surface:** `src/writingring/board_event_segmentation.py`,
  `tests/test_board_event_segmentation.py`.

#### FROZEN TaskSpec

- **Task ID:** T5; **Dependencies:** T2 DONE.
- **Goal:** Lock in existing generic downstream handling of the new validated
  coverage skip without adding classification logic.
- **Required behavior:** Add tests that construct a valid
  `AlignmentSkipArtifact` directly (the factory remains initial-only) for
  `insufficient_event_coverage`; assert mixed SUCCESS/skipped results process
  only success and summarize processed/skipped as 1/1; assert all such skips
  hard-fail before aggregation/publication.
- **Allowed write paths:** `tests/test_board_event_segmentation.py` only.
- **Forbidden:** `src/writingring/board_event_segmentation.py` and all paths
  outside the allowed test.
- **Acceptance:** Preserve generic validated-outcome consumption and all
  existing initial-skip tests. Run focused test, full pytest, diff check.
- **Replan:** Any production code need, direct reason branching, factory
  expansion, or scope expansion.

### T6 — End-to-end evidence and durable documentation

- **Goal:** Run the frozen plan's required validation, capture verified
  behavior in the relevant notes, and complete planning/workboard state.
- **Expected documentation surface:** `docs/notes/ALIGNMENT_OUTPUTS.md`,
  `docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md`, this task file, and
  `docs/plans/WORKBOARD.md`.
- **Execution note:** PRIMARY performs documentation changes. A fresh
  `luna_verifier` independently checks the final, frozen integration/document
  contract after PRIMARY's documentation update.

#### FROZEN TaskSpec

- **Task ID:** T6; **Dependencies:** T1–T5 DONE.
- **Goal:** Record the completed behavior durably and verify the real
  low-coverage recording plus complete regression suite.
- **PRIMARY-only write paths:** `docs/notes/ALIGNMENT_OUTPUTS.md`,
  `docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md`, this task file, and
  `docs/plans/WORKBOARD.md`.
- **Required documentation:** enumerate all three legal skip reasons; state
  that every skip has a report and skip JSON but never an offset TXT or
  verification PNG; distinguish strict standalone defaults from Action0's
  explicit dual skip-policy invocation and its validated downstream omission/
  QA accounting.
- **Required validation:** direct user_7/action0/dataset1 SpikeIMU run with
  explicit skip (12/44 coverage, 22 valid pairs, expected coverage reason/no
  success artifacts), standalone default failure, focused regression suite,
  full pytest, Bash syntax, and diff check. Then run
  `PIPELINE_MODE=continue bash scripts/action0_pipeline/04_lowpass_aligned_board.sh`
  against available `data` if its ignored generated outputs can be updated.
- **Expected external/generated paths:** `data/user_7/0/1` and existing
  `outputs/action0_rectified/low-pass/aligned-board-events`; never alter
  `data_sample/**` or source data.
- **Replan/blocker:** unavailable/malformed real data, output overwrite
  conflict without applicable flags, any mismatch in required diagnostic
  values, or documentation requiring an undocumented claim.

#### PRIMARY evidence

- Probe-confirmed real `data/user_7/0/1` explicit skip: `SKIPPED`,
  `insufficient_event_coverage`, 12/44 (`0.2727272727`) against `0.4`, 22
  valid pairs, zero fully matched pairs, report plus skip JSON only.
- Probe-confirmed standalone-default counterpart: `FAILED`, exit 2, report
  only, and no offset/verification artifact.
- T4's post-repair validation used `writingring-gpu`: 32 focused passed and
  612 full passed, 1 skipped; `bash -n` and `git diff --check` passed. PRIMARY
  repeated the focused Action0 test (32 passed) and final Bash/diff checks.
- `PIPELINE_MODE=continue` was started for the available low-pass aligned
  Action0 root; its environment-captured output confirmed valid existing
  preprocessing reuse before the execution channel returned incomplete output.
- A later fresh verifier again found no terminal QA/padding evidence. Repeated
  full-pipeline attempts were interrupted by the execution environment; an
  isolated `user_7` real-data outcome and all test evidence remain valid.

#### Revised acceptance decision — 2026-08-10

`user_4/action_0/dataset_0` is excluded from this plan's **real-data Action0
acceptance dataset only**:

```text
KNOWN DATA/REPRESENTATION COLLISION
alignment = SUCCESS
segmentation = strict failure
no runtime skip/recovery policy authorized
```

Its source labels remain strictly increasing, but the distinct `wrong` and
`f` marker timestamps collapse through the established
`searchsorted(..., side="left")` mapping onto the same alignment-work-axis
sample boundary. The production pipeline must retain that hard failure. This
is neither an alignment failure nor an authorized `SKIPPED` reason; no code
may special-case the recording. The final real-data run uses a temporary,
read-only-derived acceptance input tree that omits only its `0_ring_0.bin`
discovery member, leaving the source `data/` tree unchanged. Future collision
support, if needed, requires a separate plan defining same-index marker audit
and interval semantics.

The temporary acceptance tree was constructed successfully and omitted only
that discovery member. Its full Action0 process was again terminated by the
current execution service before a QA/padding terminal marker could be
recorded. This is an execution-environment limitation, not acceptance of the
collision or a production recovery policy.
