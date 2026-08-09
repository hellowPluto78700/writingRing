# Spike Post-Encode Transform — Task DAG and TaskSpecs

## Plan identity

- **Source plan:** `docs/plans/Done/03_Spike_Post_Encode_Transform_Plan.md`
- **Baseline commit:** `d45ae108d54641a406bd0a582baaaac52e6a1aad`
- **Primary durable contracts:** `docs/notes/SPIKE_ENCODING.md`,
  `docs/notes/OCCURRENCE_ALIGNED_SPIKE_ENCODING.md`, and
  `docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md`

## Dependency DAG

```text
T001 Encoder post-transform contract
  ↓
T002 CLI and Action-0 pipeline control
  ↓
T003 Continue-mode artifact compatibility
  ↓
T004 Integration validation, documentation, and closure
```

| Task | Goal | Dependencies | State |
| --- | --- | --- | --- |
| T001 | Encode a validated post-encode transform while preserving occurrence-aligned detector behavior and truthful metadata. | — | DONE |
| T002 | Expose the transform through the encoding CLI and the Action-0 shared Bash control. | T001 | DONE |
| T003 | Reject stale encoded artifacts when the requested transform differs in continue mode. | T002 | DONE |
| T004 | Independently validate the end-to-end matrix, update durable documentation, and close the plan. | T001, T002, T003 | DONE |

## T001 — Draft TaskSpec

- **Goal:** Establish the encoder-level `none` / `AbsRectify` contract after
  occurrence alignment, including metadata and downstream compatibility.
- **Source:** plan §§1–3; **dependencies:** none.
- **Probe scope:** Custom Wavelet settings/encoder/runner/publication, their
  focused tests, and only direct SpikeIMU metadata consumers.
- **Contracts to preserve:** signed local-extrema detector behavior, complete
  recording reset boundaries, occurrence rows, channel ordering/count, source
  timestamps, and trailing six SpikeIMU IMU channels.
- **Expected decision:** whether the legacy
  `signed_wavelet_events_plus_imu_v1` schema can remain a 21-channel layout
  identifier while actual event polarity is made explicit in representation
  metadata.
- **Anticipated write paths:** encoder settings/runner/publication modules and
  their focused tests, as confirmed by the probe. README and notes are
  forbidden until T001 has passed its verifier.
- **Replan triggers:** a direct consumer treats the word `signed` in the
  legacy schema as a mandatory polarity guarantee; applying a transform after
  occurrence alignment cannot preserve the stated invariants; or a required
  public contract needs a schema migration beyond this plan.

## T001 — FROZEN TaskSpec

- **Goal:** Add a strict Custom Wavelet post-encode transform that supports
  only signed preservation (`None`) and occurrence-aligned absolute
  rectification (`"AbsRectify"`).
- **Source:** plan §§1–3; **dependencies:** none; **validated against:**
  `d45ae108d54641a406bd0a582baaaac52e6a1aad` plus the T001 probe.
- **Required behavior:**
  - Add `post_encode_transform: str | None = None` to
    `CustomWaveletSettings`; accept only `None` and exact `"AbsRectify"`, and
    reject aliases, case variants, booleans, integers, and unknown values.
  - Apply `np.abs` only to the final cropped occurrence-aligned `(N, C)`
    result, before it becomes read-only. Leave `step()` and the extrema
    detector signed.
  - Report `signed_sparse_wavelet_extrema` for `None` and
    `abs_rectified_sparse_wavelet_extrema` for `AbsRectify`; record both the
    transform and event representation in encoder/publication metadata.
  - Retain `spike_imu.schema="signed_wavelet_events_plus_imu_v1"` as the
    legacy 21-channel layout identifier, add explicit `event_representation`,
    and make polarity metadata match the actual values.
- **Contracts to preserve:** complete-recording reset/crop behavior,
  occurrence rows, detector extrema signs, source timestamp sidecars, event
  channel names/order/count, runner computation over final returned values,
  exact source columns `3:9` in trailing SpikeIMU channels, and all existing
  no-transform output behavior.
- **Intentionally changed contract:** rectified metadata truthfully declares
  absolute event representation and no polarity preservation; legacy layout
  schema remains stable.
- **Allowed writes:**
  `src/writingring/spike_encoding/encoders/custom_wavelet.py`,
  `src/writingring/spike_encoding/publication.py`,
  `tests/test_custom_wavelet_settings.py`,
  `tests/test_custom_wavelet_encoder.py`, and
  `tests/test_spike_encoding_publication.py`.
- **Forbidden writes:** runner, registry, CLI/Bash, all consumer modules,
  README, `docs/notes/**`, vendor, sample data, configuration, environment,
  and `scripts/action0_pipeline/SNN_Bash/**`.
- **Acceptance:** focused tests prove strict setting validation; signed versus
  rectified values equal `np.abs(signed)`; nonzero masks, shape, channel names,
  rows, and trailing six SpikeIMU channels are unchanged; rectified values are
  nonnegative; statistics and representation/polarity metadata describe final
  values; existing signed detector expectations remain unchanged.
- **Validation:** run the three focused test files and full `pytest` in the
  repository-required environment; record Python-version limitations rather
  than silently treating them as success.
- **Replan triggers:** any write outside scope; a direct consumer needs a
  schema migration; transform cannot remain post-crop; or existing signed
  detector expectations would need alteration.

### T001 execution record

Worker completed the frozen encoder/publication/test scope. Focused tests
passed 30 and the full suite passed 512 with one known skip in the available
Python 3.11.15 `writingring-gpu` fallback; `git diff --check` passed. The
repository-required `writingring-viz` command was attempted but its Conda
environment was unavailable (`NoWritableEnvsDirError`), so this remains an
environment-evidence limitation for independent verification.

### T001 verification and documentation

Fresh verifier PASS confirmed the frozen behavior and scope. PRIMARY updated
the two affected durable encoder contracts: default signed behavior, optional
post-crop `AbsRectify`, explicit event representation/provenance, unchanged
legacy layout schema, and row/channel/sparsity preservation. README remained
unchanged. `writingring-viz` test evidence is still unavailable because its
Conda environment raises `NoWritableEnvsDirError`; the independently verified
Python 3.11.15 fallback evidence is recorded above.

## T002 — Draft TaskSpec

- **Goal:** Add explicit CLI and Action-0 Bash transform selection with the
  plan-defined precedence and strict fail-fast validation.
- **Dependencies:** T001 (DONE).
- **Probe scope:** `scripts/encode_spikes.py`, `_common.bash`, their existing
  tests, and call sites only.
- **Replan triggers:** CLI/settings precedence conflicts with existing public
  configuration semantics or pipeline input/output contracts.

## T003 — Draft TaskSpec

- **Goal:** Make current post-encode-transform provenance part of continue
  validation without redesigning stage-resume policy.
- **Dependencies:** T002 (DONE).
- **Probe scope:** `_common.bash` encode validation and continue-mode tests.
- **Replan triggers:** current invalid-artifact handling cannot safely rebuild
  or the artifact metadata lacks a backward-compatible no-transform reading.

## T002 — FROZEN TaskSpec

- **Goal:** Make the T001 transform explicitly selectable through
  `encode_spikes.py` and the common Action-0 pipeline control without changing
  default behavior.
- **Source:** plan §4; **dependencies:** T001 (DONE); **validated against:**
  T001 verified working tree and T002 probe.
- **Required behavior:**
  - Add `--post-encode-transform` with choices `none` and `AbsRectify`, default
    `None`. Omission preserves JSON/settings/default behavior; explicit `none`
    writes `None`; explicit `AbsRectify` writes `"AbsRectify"`, after settings
    load and before encoder creation.
  - The option is Custom-Wavelet-specific: when explicitly passed with another
    encoder, fail clearly rather than silently ignoring it. Omitted behavior
    for another encoder remains unchanged.
  - Replace hard-coded signed CLI result wording with the final reported
    representation.
  - Add `POST_ENCODE_TRANSFORM="${POST_ENCODE_TRANSFORM:-none}"` to
    `pipeline_init()`, validate only `none|AbsRectify` through the established
    fail-fast path, and pass the explicit CLI flag from the sole
    `pipeline_encode()` call site. The existing logged shell-quoted command is
    the audit record; all wrappers inherit it without edits.
- **Contracts to preserve:** existing settings parsing, sampling-rate handling,
  output publication, eight wrapper entrypoints, all defaults, and T001
  no-transform behavior. T003 exclusively owns continue-artifact provenance
  checks and resume policy.
- **Allowed writes:** `scripts/encode_spikes.py`,
  `scripts/action0_pipeline/_common.bash`,
  `tests/test_encode_spikes_cli.py`, and
  `tests/test_action0_pipeline_scripts.py`.
- **Forbidden writes:** wrapper scripts, encoder/publication modules, docs,
  README, configuration, consumer modules, vendor, sample data, environments,
  and `scripts/action0_pipeline/SNN_Bash/**`.
- **Acceptance:** omitted CLI leaves settings/default unchanged; explicit
  `none` overrides a configured transform; explicit `AbsRectify` overrides to
  rectification; invalid CLI and Bash values fail; the pipeline log command
  includes the flag; result status names final representation; all eight
  wrappers retain common wiring.
- **Validation:** focused CLI/pipeline tests, T001 focused encoder/publication
  tests, full `pytest`, and `git diff --check`; record unavailable
  `writingring-viz` evidence separately from Python 3.11 fallback evidence.
- **Replan triggers:** a non-Custom encoder requires transform semantics, CLI
  precedence conflicts with current settings behavior, or a required change to
  pipeline resume policy.

### T002 execution record

Worker completed the frozen CLI/Bash/test scope. CLI/Bash focused tests passed
20, T001 focused tests passed 30, and the full fallback Python 3.11.15 suite
passed 516 with one known skip; Bash syntax and `git diff --check` passed.
`writingring-viz` remains unavailable with `NoWritableEnvsDirError`.

### T002 verification and documentation

Fresh verifier PASS confirmed strict CLI precedence, Custom-Wavelet-only
explicit handling, Bash default/final authority, auditable command wiring, and
unchanged wrappers. PRIMARY documented only those confirmed CLI/Bash controls
in the affected encoder and pipeline notes; README remained unchanged. The
same `writingring-viz` environment limitation remains recorded separately from
fallback Python 3.11 evidence.

## T004 — Draft TaskSpec

- **Goal:** Execute the final verified matrix and document only confirmed
  behavior.
- **Dependencies:** T001–T003 (DONE).
- **Expected ownership:** PRIMARY runs validation/documentation; no Luna
  worker unless a separately frozen implementation repair is required.
- **Replan triggers:** required integration evidence reveals a contract or
  implementation defect.

## T004 — FROZEN TaskSpec

- **Goal:** Independently validate the complete `none`/`AbsRectify` path with
  temporary isolated inputs, record only verified results, and close the plan.
- **Source:** plan §§7–9; **dependencies:** T001–T003 (DONE); **validated
  against:** verified T001–T003 working tree and T004 probe.
- **Execution ownership:** PRIMARY performs validation and documentation only;
  no `luna_worker` is authorized because no implementation change is expected.
- **Required evidence:**
  - Same-input `none` and `AbsRectify` fixture proves event values equal
    `abs(signed)`, exact nonzero mask/shape/rows/channel names, nonnegative
    rectified values, final statistics, and representation/polarity/transform
    metadata.
  - Published SpikeIMU proves exact `[:, 15:21]` equality between variants and
    with source `[:, 3:9]`, unchanged timestamp values/path/digest, and stable
    legacy layout schema.
  - CLI omission/explicit/invalid behavior, Bash invalid fail-fast, logged
    command flag, same-transform reuse, changed-transform mismatch diagnostic,
    and existing preprocess/full-rebuild handling are exercised by focused
    tests.
  - Run focused T001–T003 tests, full pytest, Bash syntax, and `git diff
    --check`; attempt the repository-required `writingring-viz` environment
    and record any failure separately from Python 3.11 fallback evidence.
- **Allowed writes:** `docs/plans/**`, `docs/notes/**` only if an already
  verified behavior is missing, and temporary `/tmp` fixture artifacts.
- **Forbidden writes:** README, all implementation/tests/configuration,
  `outputs/**`, vendor, sample data, environments, and
  `scripts/action0_pipeline/SNN_Bash/**`.
- **Closure:** update plan/task/workboard state and relocate the plan and task
  record to `docs/plans/Done/` only after fresh verifier PASS and synchronized
  pointers. Keep README unchanged.
- **Acceptance:** all matrix evidence passes or is accurately classified;
  temporary fixture leaves no repository artifact; documentation contains no
  unsupported claim; fresh verifier PASS.
- **Replan triggers:** fixture disproves the stated invariant, a required
  test fails due implementation, required docs conflict with verified fact, or
  final path relocation leaves stale pointers.

### T004 execution record

PRIMARY completed the frozen, temporary producer-backed same-input matrix on
Python 3.11.15 in `writingring-gpu`: the signed and `AbsRectify` event arrays
matched exactly by `abs`, had identical masks, rows, and channels, and the
rectified output had no negative values. Both published `(N, 21)` SpikeIMU
artifacts preserved the same source IMU tail `[:, 3:9]`, timestamp path and
SHA-256 digest, and the legacy layout schema, while metadata and final
statistics truthfully distinguished the two representations. The focused
T001--T003 suite passed **61**, and the full suite passed **527** with the one
known unavailable-real-artifact skip. Bash syntax and `git diff --check`
passed. The required `writingring-viz` environment could not be entered:
Conda reported `NoWritableEnvsDirError`; this is an environment-access
limitation, not a Python 3.11, dependency, or implementation test failure.

### T004 verification and completion

Fresh verifier PASS independently confirmed the producer-backed matrix,
metadata/provenance and legacy-layout compatibility, CLI/Bash and
continue-rebuild coverage, and the recorded environment classification. No
additional README or note change was needed beyond the verified T001--T003
updates. The task DAG is complete and this plan/task record is now archived in
`docs/plans/Done/`.

## T003 — FROZEN TaskSpec

- **Goal:** Make requested post-encode-transform provenance a required part of
  continue-mode SpikeIMU artifact validity, using the existing invalid-output
  rebuild policy.
- **Source:** plan §5; **dependencies:** T002 (DONE); **validated against:**
  T001/T002 verified working tree and T003 probe.
- **Required behavior:**
  - Extend `pipeline_validate_spike_artifact` to receive requested transform,
    normalize shell `none` to Python `None`, read
    `metadata.settings.post_encode_transform`, and fail with expected/actual
    values when they differ.
  - Treat missing settings/key and JSON `null` as legacy `None`. Thus legacy or
    no-transform artifact plus `none` remains valid, while any no-transform
    form plus `AbsRectify`, or `AbsRectify` plus `none`, is invalid.
  - Pass the requested transform at both existing validation sites:
    continue-mode `pipeline_encode_outputs_valid` and final `pipeline_qa`.
  - On mismatch, retain existing behavior: invalid encode output forces the
    established preprocess/full-rebuild path. Do not create transform-specific
    resume stages or change rebuild policy.
- **Contracts to preserve:** existing finite `(N,21)`/timestamp/channel checks,
  T002 shell validation, same-transform resume behavior, metadata authority at
  `settings.post_encode_transform`, and all downstream pipeline stages.
- **Allowed writes:** `scripts/action0_pipeline/_common.bash` and
  `tests/test_action0_pipeline_scripts.py` only.
- **Forbidden writes:** encoder/CLI/publication, wrappers, docs/README,
  consumer modules, vendor, sample data, configuration, environments, and
  `scripts/action0_pipeline/SNN_Bash/**`.
- **Acceptance:** tests cover missing/null/matching/mismatching metadata,
  both validator call sites, same-transform reuse, mismatch invalidation and
  existing preprocess/full-rebuild result; existing pipeline tests remain
  green.
- **Validation:** focused pipeline tests, T001/T002 focused tests, full
  `pytest`, Bash syntax, and `git diff --check`; keep required-environment
  unavailability distinct from fallback Python 3.11 evidence.
- **Replan triggers:** canonical metadata location changes, existing rebuild
  policy cannot handle invalid encoding safely, or a change requires any
  resume-policy redesign.

### T003 execution record

Worker completed the frozen Bash/test scope. Focused pipeline tests passed 60;
the full fallback Python 3.11.15 suite passed 527 with one known skip; Bash
syntax and `git diff --check` passed. `writingring-viz` remains unavailable
with `NoWritableEnvsDirError`.

### T003 verification and documentation

Fresh verifier PASS confirmed canonical transform provenance, legacy
no-transform compatibility, both validation sites, and unchanged full-rebuild
handling. PRIMARY documented only this verified continue contract in the
pipeline note; README remained unchanged. `writingring-viz` evidence remains
unavailable separately from fallback Python 3.11 evidence.
