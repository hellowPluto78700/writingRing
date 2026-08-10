# Board-Assisted Alignment Correction — Task DAG and TaskSpecs

## Plan state

- Source plan: `docs/plans/TODO/01_Board_Assist_Alignment_Correction_Plan.md`
- Baseline commit: `a69476c4f3ae713335b6867678cd35c48a04ef3d`
- Status: active

## Dependency graph

```text
T1 Board initial-interval semantics
        |
        v
T2 Alignment outcome contract and CLI
      /   \
     v     v
T3 Action0  T4 Board-segmentation outcome consumer
      \   /
       v v
T5 End-to-end verification and durable documentation
```

| Task | State | Depends on | Owner |
| --- | --- | --- | --- |
| T1 | DONE | — | luna_probe → luna_worker → luna_verifier; PRIMARY documents |
| T2 | DONE | T1 | luna_probe → luna_worker → luna_verifier; PRIMARY documents |
| T3 | PROBING | T2 | luna_probe → luna_worker → luna_verifier; PRIMARY documents |
| T4 | PROBING | T2 | luna_probe → luna_worker → luna_verifier; PRIMARY documents |
| T5 | DRAFT | T3, T4 | PRIMARY validation/documentation, with fresh luna_verifier |

Only T1 is dependency-ready. Downstream TaskSpecs will be written and frozen
only after their dependencies complete and their implementation facts have
been freshly probed.

## T1 — Board initial-interval semantics (FROZEN)

- Source plan: `01_Board_Assist_Alignment_Correction_Plan.md`
- Dependencies: none
- Validated against commit: `a69476c4f3ae713335b6867678cd35c48a04ef3d`
- Probe evidence: initial probe `REVISE`; fresh re-probe `CONFIRMED`

### Goal

Implement the sole newly skippable condition: the complete numerically ordered
Board recording contains one or more globally valid press/lift pairs and has a
first backward timestamp jump, but no pair has both endpoints in the initial
monotonic frame interval before that jump. T1 itself raises a typed diagnostic
exception; T2 alone decides whether the caller publishes a SKIPPED outcome.
All other board-pair and timestamp failures retain their current behavior.

### Dependency and scope hypotheses to probe

- Depends on no plan task.
- This task does not create a final `SUCCESS`/`SKIPPED` artifact, alter CLI
  policy, modify Action0 orchestration, or change Board segmentation. Those
  changes belong to T2–T4.

### Required behavior

- The first backward jump is determined from the existing numeric
  chunk/frame order as the first post-jump global-frame position `j`; the
  initial interval is positional `[0, j)`. It is never timestamp-sorted,
  repaired, or extended into a later epoch.
- A usable pair has both its press and lift global-frame indices in `[0, j)`.
  The final pre-jump frame `j - 1` is included; the first post-jump frame `j`
  is excluded. A cross-boundary pair is never usable.
- Once the initial interval is selected, all selected frames and derived
  events/pairs must remain within that positional prefix before any timestamp
  range filtering. Where production contacts carry global frame identity, they
  are restricted by the same prefix. This prevents post-jump stale timestamp
  overlap from re-entering the selected result.
- Introduce a typed subclass of `EventAlignmentError` in
  `writingring.event_alignment` for this exact condition. Its stable,
  structured diagnostics expose first-jump previous/next global-frame
  identities and raw timestamps, exclusive prefix boundary position, inclusive
  last pre-jump frame, total globally valid-pair count, and usable-prefix
  valid-pair count. It must be distinguishable by T2 without parsing a
  message.
- No globally valid pair, malformed Board data, invalid event tables, and
  downstream matching failures continue to use their existing hard failures.
  A recording with no backward jump and valid pairs retains its existing
  normal-success interval behavior; no backward jump and no valid pairs
  retains the existing failure.

### Contracts to preserve

- Numeric Board chunk order and raw frame order remain immutable input order.
- No timestamp sorting, repair, post-jump epoch selection, or broad exception
  to skip conversion.
- Alignment work-axis, canonical timestamp, peak detection, coverage, and
  provenance contracts remain unchanged.
- Valid-touch duration semantics remain frame-count based; T1 must not infer
  timestamp validity from a duration that can cross a jump.

### Allowed write paths

- `src/writingring/event_alignment.py`
- `tests/test_event_alignment.py`

### Forbidden write paths

- `vendor/**`, `data_sample/**`, `scripts/**`, `src/writingring/alignment_io.py`,
  `src/writingring/__init__.py`, segmentation modules, Action0 scripts,
  README, `docs/**`, configuration, environment files, and unrelated tests.

### Acceptance criteria

- Focused regression tests cover pre-jump usable, post-jump-only,
  cross-boundary-only, mixed usable/nonusable, no-global-pair, no-jump normal
  success, stale-tail timestamp overlap, and malformed/order-preserving cases.
- The exact typed exception occurs only when global valid-pair count is
  positive, a first backward jump exists, and usable-prefix valid-pair count
  is zero. It carries the frozen diagnostic fields.
- No-jump recordings with valid pairs retain normal successful interval
  selection. Zero-global-pair, malformed/nonexistent/duplicate frame identity,
  and invalid event-table paths retain existing hard failures.
- The selected frame/event/pair tables and identity-bearing contact table have
  no post-prefix rows even when their timestamps overlap the selected window.
- No code outside the allowed paths changes.

### Validation commands

```bash
conda run --no-capture-output -n writingring-gpu python -m pytest -q tests/test_event_alignment.py tests/test_board_loader.py
conda run --no-capture-output -n writingring-gpu python -m pytest -q
```

If `writingring-gpu` is unavailable, use `writingring-viz` with the same
commands. The worker must report which environment was used.
- Full repository `pytest` passes in the project Python 3.11 Conda
  environment after implementation.
- A fresh verifier confirms write scope and preserved contracts.

### Replan triggers

- Existing code cannot represent the typed diagnostic without changing the
  public outcome/artifact layer.
- Correct stale-tail exclusion requires a producer/consumer/API path beyond
  the stated T1 write scope.
- Correct testing requires an unplanned public schema or path change.

### Completion record

T1 completed after its initial `REVISE` and fresh `CONFIRMED` re-probe. The
worker changed only `src/writingring/event_alignment.py` and
`tests/test_event_alignment.py`. It added positional prefix enforcement,
both-endpoint pair eligibility, identity-aware stale-tail filtering, and the
typed `InitialIntervalNoUsablePairError`. Fresh verification passed the frozen
scope and contract checks, reran 51 focused tests, recorded full pytest as 530
passed/1 skipped in `writingring-gpu`, and found no undocumented behavior.
No user-facing durable note is updated yet because the public outcome artifact
and CLI policy are intentionally deferred to T2.

## T2 — Alignment outcome contract and CLI (FROZEN)

- Source plan: `01_Board_Assist_Alignment_Correction_Plan.md`
- Dependencies: T1 DONE
- Validated against: `a69476c4f3ae713335b6867678cd35c48a04ef3d` plus verified
  uncommitted T1 implementation/tests
- Probe evidence: initial `REVISE`; fresh re-probe `CONFIRMED`

### Goal

Build the authoritative Python representation, validation, publication, and
CLI policy for exactly three terminal alignment states:

```text
SUCCESS  → validated offset artifact
SKIPPED  → validated skip artifact for T1's exact typed condition only
FAILED   → report only; not a completed outcome
```

The standalone alignment CLI remains error-by-default. Only an explicit skip
policy may catch T1's `InitialIntervalNoUsablePairError` and publish SKIPPED.

### Dependency and implementation hypotheses to probe

- T1 is DONE; its typed diagnostic is the only candidate skip source.
- Likely implementation paths include `src/writingring/alignment_io.py`,
  `scripts/align_ring_board.py`, `src/writingring/__init__.py`, and focused
  alignment I/O/CLI tests.
- Determine the actual current offset/report schema, publication ordering,
  provenance validators, overwrite behavior, and package export conventions.
- T2 must centralize Python outcome validation in `alignment_io.py`; Bash and
  downstream consumers are deferred to T3/T4.

### Required behavior to validate before freezing

#### Exact outcome files and schemas

- Keep the existing success TXT path unchanged. Add a mutually exclusive skip
  artifact at the sibling offset path
  `<offset_root>/<user>/action_<action>/<dataset>_ring_board_skip.json`.
  Its JSON is strict (`allow_nan=False`) and declares
  `alignment_skip_schema_version: 1`, `artifact_kind: "alignment_skip"`, recording identity,
  `reason="initial_interval_no_usable_pair"`, the eight T1 diagnostic fields,
  input provenance, and its own artifact kind.
- Keep the existing report path unchanged and make the report the last-written,
  authoritative outcome manifest. Add `alignment_outcome_schema_version: 1`,
  `alignment_status: "success"|"failed"|"skipped"`, and an
  `outcome_artifacts` manifest of filename and SHA-256 digests. Success
  reports retain legacy success booleans, which must agree with the status.
  Existing schema-v2 work-axis success remains SUCCESS even if canonical
  projection is unavailable.
- A valid SUCCESS requires exactly one valid existing offset TXT, a strict
  status-success report that hashes that TXT, and a nonempty verification PNG
  whose digest is also in the report. A valid SKIPPED requires exactly one
  strict skip JSON and a status-skipped report hashing it; it has no offset or
  verification PNG. A report with `failed` is never completed.
- The copied T1 diagnostics mapping has exactly these required keys:
  `previous_global_frame_index`, `next_global_frame_index`,
  `previous_timestamp_raw`, `next_timestamp_raw`,
  `prefix_boundary_position`, `last_pre_jump_global_frame_index`,
  `total_global_valid_pair_count`, and
  `usable_prefix_valid_pair_count`.
- Each `outcome_artifacts` entry is an object with `filename` (base name only)
  and lowercase 64-hex `sha256`; success declares `offset_txt` and
  `verification_png`, skipped declares `skip_json`, and failed declares none.
- Validator input must support both current CLI input kinds. SpikeIMU
  provenance consists of the existing values/metadata/canonical-timestamp
  SHA-256 fields plus identity/schema/rate. Raw-ring provenance consists of
  input kind, recording identity, raw `ring_0` content SHA-256, and canonical
  timestamp-array SHA-256. Both outcomes also contain Board provenance:
  the numeric-ordered sequence of `{chunk_index, sha256}` for the actual Board
  chunk files. Paths may be diagnostic only, never a replacement for a digest.

#### Python API and validation boundary

- `alignment_io.py` owns dataclasses and read/write/validate helpers for
  outcome status, skip artifact, current input provenance, outcome paths, and
  complete-outcome validation. `read_alignment_offset_txt` remains a
  compatibility reader for success TXT, but no longer defines completion.
- The authoritative validator requires expected recording identity, current
  feature provenance, and current Board provenance. It rejects absent,
  malformed/nonfinite, unsupported-version, stale, partial, status-mismatched,
  digest-mismatched, and success+skip-conflicting files. Legacy offset/report
  files are readable only through existing compatibility APIs and cannot pass
  the new completed-outcome validator without v1 outcome manifest fields.
- The public completed-outcome validator receives resolved outcome roots plus
  expected identity, input provenance, and Board provenance, returns a typed
  `SUCCESS` or `SKIPPED` outcome object only, and raises the existing
  alignment-I/O exception family for every non-completion. It must not return
  FAILED as a completion value.
- New public types/helpers needed by callers are re-exported through
  `writingring.__all__`; unrelated event-extraction exports are not added in
  this task.

#### Publication and overwrite boundary

- Before writing, preflight every target and transition. Same-state replacement
  follows the existing artifact-specific overwrite permissions; a
  success↔skip transition additionally requires explicit
  `--overwrite-outcome` authorization. No unauthorized invocation may delete,
  replace, or invalidate a prior completed outcome.
- Stage and atomically replace individual TXT/JSON/PNG artifacts; stage the
  strict report manifest last. On an authorized transition, remove obsolete
  opposite-state artifact(s) before publishing the final report. Thus a crash
  can leave only an invalid/incomplete stage, never a reader-valid new
  completed outcome. The validator treats every intermediate/missing manifest
  combination as invalid.

#### CLI policy boundary

- Add `--initial-interval-policy error|skip`, default `error`. Only `skip`
  narrowly catches T1's `InitialIntervalNoUsablePairError` and publishes the
  SKIPPED artifact/report via the outcome API. `error` retains the existing
  nonzero failure behavior and must not publish SKIPPED.
- Existing broad `EventAlignmentError`, `AlignmentFailureError`, load errors,
  and unexpected exceptions remain error/FAILED behavior; no catch-all can
  create SKIPPED. Existing success behavior remains compatible while publishing
  the added outcome-manifest fields.

#### Tests and validation expected before freezing

- Focused tests cover strict skip/report schemas, current SpikeIMU and raw
  provenance, stale Board chunk, missing/partial/malformed files, digest and
  status mismatch, conflict, projection-without-canonical-offset success,
  authorized/denied success↔skip transitions, atomic-manifest ordering, public
  export, and error-vs-skip CLI policy.
- The exact test files and CLI flag wiring are confirmed by the re-probe; full
  pytest remains required after implementation.

### Allowed write paths

- `src/writingring/alignment_io.py`
- `scripts/align_ring_board.py`
- `src/writingring/__init__.py`
- `tests/test_alignment_io.py`
- `tests/test_alignment_verification.py`
- `tests/test_spike_imu_segmentation.py`

### Forbidden write paths

- `src/writingring/event_alignment.py`, `src/writingring/board_event_segmentation.py`,
  `scripts/action0_pipeline/**`, all other scripts/modules/tests, `vendor/**`,
  `data_sample/**`, README, and `docs/**`.

### Acceptance and validation

- Preserve legacy offset TXT reader/writer compatibility and do not change the
  low-level offset validator to require outcome fields.
- No status/artifact/provenance/validator logic is implemented in Bash or
  Board segmentation.
- In `writingring-gpu` (or `writingring-viz` only if unavailable), run the
  focused authorized alignment tests, then `python -m pytest -q`.
- `git diff --check` passes and a fresh verifier confirms the entire frozen
  schema, policy, provenance, overwrite, and write-scope contract.

### Replan triggers

- Current artifact layout cannot express mutually exclusive outcome validity
  without broader producer/consumer ownership.
- Required provenance or publication guarantees require an unplanned schema or
  path owner.
- The CLI's current error/report behavior conflicts with the plan's explicit
  policy boundary.

## T2–T5 (DRAFT outlines)

- **T2:** Centralize validated success/skip/report outcomes in `alignment_io`,
  add explicit CLI skip policy, provenance, publication, overwrite, and
  relevant exports. Dependency satisfied; currently probing.
- **T3:** Make Action0 aligned-board orchestration call the T2 Python outcome
  validator for continue/rebuild/QA accounting. Freeze only after T2.
- **T4:** Make Board-event segmentation consume T2 outcomes, omit verified
  recording-level skips, expose counts, and reject all-skipped aggregates.
  Freeze only after T2.
- **T5:** Primary-owned end-to-end validation plus durable documentation after
  T3 and T4 pass; a fresh verifier independently checks the final DAG contract.
