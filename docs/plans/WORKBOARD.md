# Workboard

## Active plan

- **Plan:** none — most recently completed:
  `docs/plans/Done/03_Spike_Post_Encode_Transform_Plan.md`
- **Task file:** `docs/plans/Done/03_Spike_Post_Encode_Transform_TASKS.md`
- **Baseline commit:** `d45ae108d54641a406bd0a582baaaac52e6a1aad`
- **Current task:** none
- **Status:** DONE

## Task state

| Task | State | Depends on |
| --- | --- | --- |
| T001 | DONE | — |
| T002 | DONE | T001 |
| T003 | DONE | T002 |
| T004 | DONE | T001, T002, T003 |

## Focus and next action

- Completed focus: `AbsRectify` is applied after complete occurrence-aligned
  signed encoding while preserving sparsity, rows, channels, timestamps, and
  the six trailing IMU values.
- Baseline hygiene: inherited uncommitted plan relocations and
  `scripts/action0_pipeline/SNN_Bash/` are outside this plan unless a frozen
  TaskSpec explicitly includes them.
- Documentation gate: do not modify README or `docs/notes/**` before T001
  contract baseline has passed independent verification.
- Probe digest: CONFIRMED. The transform can be post-crop without detector or
  row/sparsity changes; no direct consumer interprets the legacy schema name
  as a mandatory polarity guarantee. Publication metadata must explicitly
  describe rectification while retaining the legacy layout schema.
- Worker digest: DONE. Strict settings, post-crop rectification, dynamic
  representation, and truthful publication metadata landed only in frozen
  paths. Python 3.11.15 fallback: focused 30 passed, full 512 passed/1 skipped;
  `writingring-viz` is unavailable with `NoWritableEnvsDirError`.
- T001 verifier digest: PASS. PRIMARY documented only verified transform and
  representation behavior after acceptance; README remains unchanged. The
  Python 3.11.15 fallback has 30 focused passed and 512 passed/1 known skip;
  `writingring-viz` remains unavailable with `NoWritableEnvsDirError`.
- T002 probe digest: CONFIRMED. Omitted CLI can preserve settings while
  explicit values override; `_common.bash` has one encoding call site inherited
  by all wrappers. PRIMARY froze Custom-Wavelet-only explicit option handling
  and final-representation status wording.
- T002 worker digest: DONE. The frozen option precedence, Custom-Wavelet-only
  explicit validation, Bash final authority, logging wiring, and truthful CLI
  status landed in scope. Python 3.11.15 fallback: 516 passed/1 known skip;
  `writingring-viz` remains unavailable.
- T002 verifier digest: PASS. PRIMARY documented verified CLI/Bash controls;
  defaults and wrappers are unchanged. Fallback Python 3.11.15 evidence is
  516 passed/1 known skip; `writingring-viz` remains unavailable.
- T003 probe digest: CONFIRMED. Missing/null transform metadata is legacy
  `None`; mismatch can safely use the current invalid-output full-rebuild
  policy. Both continue and final-QA validator call sites require the expected
  transform.
- T003 worker digest: DONE. Artifact transform provenance now participates in
  both existing validation sites; mismatch uses the existing preprocess/full
  rebuild. Python 3.11.15 fallback: focused 60 passed, full 527 passed/1 known
  skip; `writingring-viz` remains unavailable.
- T003 verifier digest: PASS. PRIMARY documented the verified legacy/mismatch
  continue behavior; fallback Python 3.11.15 evidence is 61 focused passed and
  527 passed/1 known skip, while `writingring-viz` remains unavailable.
- T004 probe digest: CONFIRMED. No code repair is indicated; final acceptance
  needs a temporary same-input fixture for tail/timestamp provenance alongside
  focused/full test evidence and truthful environment classification.
- T004 execution digest: temporary producer-backed paired matrix PASS. Signed
  and rectified event values/masks/rows/channels, SpikeIMU tails, timestamp
  provenance, legacy schema, metadata, and final statistics meet the frozen
  acceptance criteria. Fallback Python 3.11.15: 61 focused passed; 527
  passed/1 known skip full. Bash syntax and `git diff --check` passed.
  `writingring-viz` cannot be entered (`NoWritableEnvsDirError`), an
  environment-access limitation rather than test evidence.
- T004 verifier digest: PASS. Independent review confirmed the paired
  producer-backed artifacts, compatibility/provenance invariants, test
  evidence, and environment classification; no blocker or replan is needed.
- Next action: none — plan complete and records archived under
  `docs/plans/Done/`.
