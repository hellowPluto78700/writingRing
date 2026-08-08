# Documentation Contract Repair — Task DAG and TaskSpecs

## Plan identity

- Source plan: `docs/plans/TODO/00_Documentation_Contract_Repair_Plan.md`
- Active documentation task file: this file
- Baseline commit: `34a79fc6a6d5942e082cc23ab6be4e5c20dd63db`

## Dependency DAG

```text
T001
 ├── T002
 ├── T003
 ├── T004
 ├── T005
 └── T006 (DATA_FORMAT only; historical reports deferred)
       \
        +--> T007
T002 --------/
T003 -------/
T004 ------/
T005 -----/

T008 is conditional on an implementation-contract discrepancy.
```

| Task | Goal | Dependencies | State |
| --- | --- | --- | --- |
| T001 | Establish verified documentation contract baseline | — | DONE |
| T002 | Repair README orientation contract | T001 | DONE |
| T003 | Repair alignment and data-format contracts | T001 | DONE |
| T004 | Repair segmentation and spike contracts | T001 | DONE |
| T005 | Repair Action-0 orchestration documentation | T001 | DONE |
| T006 | DATA_FORMAT residual contract classification | T001 | DONE |
| T007 | Verify documentation consistency across repaired files | T002–T006 | DONE |
| T008 | Replan genuine implementation inconsistency | Conditional | NOT REQUIRED |

## T001 execution record — baseline established

| Layer | Verified public contract |
| --- | --- |
| Raw Ring source | `N × 7` native-endian float64 rows: six sensor values and a timestamp; `ring_0` only |
| Preprocessed IMU | `N × 9`: selected acceleration in `g`, the same values in `m/s²`, then gyro |
| Custom-wavelet events | `N × 15`: three axes by five configured frequencies |
| Custom-wavelet SpikeIMU | `N × 21`: events plus source-preprocessed `[:, 3:9]` |
| Raw-ring segmentation | variable-length contiguous `total_samples × 9` output plus offsets/lengths |
| SpikeIMU segmentation | variable-length contiguous `total_samples × 21` output plus offsets/lengths |
| Padded SpikeIMU | `segments × T × 21`, right padded with valid-length and valid-prefix-mask metadata |
| Action-0 input | padded SpikeIMU slice `[:, :, 0:15]` |

Timestamp vocabulary is frozen as follows:

```text
canonical timestamps     source/provenance and segmentation axis
alignment work axis      endpoint-reconstructed strictly increasing coordinate
work-axis offset         match result in that declared coordinate
canonical projection     separately reported projection status/value
```

The normal Action-0 wrapper has `PIPELINE_MODE=continue`,
`MADGWICK_PROVISIONAL=0`, nominal `SAMPLING_RATE=200`, coverage `0.99`,
balanced selection, rounding `1`, and padding value `0`. These are wrapper
controls, not raw-acquisition guarantees; standalone padding keeps its own
default selection scope.

T001 is ready for independent verification. README and `docs/notes/**` were
not modified while establishing this baseline.

## T002 — FROZEN TaskSpec

- **Goal:** Make README a concise, current repository entry point without
  duplicating technical-note contracts.
- **Source plan:** §10 T002; **dependencies:** T001 (DONE);
  **validated against:** `34a79fc6a6d5942e082cc23ab6be4e5c20dd63db`.
- **Verified behavior:** primary Ring input is `*_ring_0.bin`, native-endian
  float64 `N × 7` binary rows; timestamp microseconds are a repository
  interpretation and 200 Hz is a processing/default setting. Current flow is
  preprocessing (9) → optional custom-wavelet events (15)/SpikeIMU (21) →
  optional alignment → label or Board-event segmentation → analysis/padding →
  separate optional Action-0 SynNet training on `0:15`.
- **Contracts to preserve:** `ring_1` is ignored; action directories are
  identifiers without verified human semantics; README stays orientation-only;
  Action-0 shell wrappers end at padded artifacts and module training is
  separate.
- **Documentation changes:** remove NumPy-file/acquisition/action-label
  claims; replace obsolete plotting-only orientation with current capability
  flow and links to the relevant notes; qualify vendor utilities as historical
  upstream tools if mentioned.
- **Allowed write paths:** `README.md`, this task file, and
  `docs/plans/WORKBOARD.md`.
- **Forbidden write paths:** `docs/notes/**`, implementation, tests,
  configuration, `vendor/**`, and `data_sample/**`.
- **Acceptance:** raw binary wording; no universal 200-Hz or action-semantic
  claim; clear current pipeline and entry points; valid, link-oriented detail
  references; fresh verifier PASS.
- **Validation:** README link/path inspection; targeted stale-term scan;
  independent verifier review.
- **Replan triggers:** a needed undocumented upstream guarantee, code/test
  disagreement, or any required write outside the allowed paths.

### T002 execution record

README now describes raw `ring_0` input as float64 binary rows rather than a
NumPy file, qualifies both the 200-Hz processing setting and action IDs, and
links its orientation-level pipeline to technical notes. It distinguishes the
wrapper pipeline from separate optional Action-0 training.

## T003 — FROZEN TaskSpec

- **Goal:** Repair alignment timestamp/offset terminology while keeping the
  DATA_FORMAT source reference separate from derived behavior.
- **Source plan:** §10 T003; **dependencies:** T001 (DONE);
  **validated against:** `34a79fc6a6d5942e082cc23ab6be4e5c20dd63db`.
- **Verified behavior:** both public inputs use endpoint-reconstructed work
  axes; feature sampling rate is metadata and effective work-axis rate is
  endpoint-derived; `offset_domain=alignment_work_axis` makes `offset_us` a
  work-axis value, while `canonical_offset_us` exists only after successful
  projection. Work alignment may succeed even when canonical projection fails;
  its declared-domain TXT/report/PNG remain publishable and segmentation uses
  `offset_domain`/`boundary_offset_us` accordingly.
- **Documentation changes:** replace obsolete strict/reconstruction and direct
  canonical-export wording; state mapping direction, domains, artifact fields,
  output consequences, and source-vs-derived status.
- **Allowed write paths:** `docs/notes/ALIGNMENT_OUTPUTS.md`,
  `docs/notes/DATA_FORMAT.md`, this task file, and WORKBOARD.
- **Forbidden paths:** implementation, tests, configuration, README,
  unrelated notes, `vendor/**`, and `data_sample/**`.
- **Acceptance:** current sections contain no obsolete implementation claim;
  all stated domains/rates/output conditions match code/tests; verifier PASS.
- **Validation:** targeted term scan and independent verifier review.
- **Replan triggers:** code/test disagreement or required writes outside scope.

### T003 execution record

The alignment note now uses endpoint reconstruction for both inputs and
documents schema-v2 work-axis offset artifacts, projection status, and their
consumer behavior. DATA_FORMAT identifies these as derived outputs rather than
raw source facts.

## T004 — Draft TaskSpec

- **Goal:** Make segmentation/spike notes agree on public 9/15/21 schemas,
  input kinds, boundary modes, and raw-mode terminology.
- **Source plan:** §10 T004; **dependencies:** T001 (DONE).
- **Anticipated write paths after freeze:** BOARD_EVENT_GUIDED_SEGMENTATION,
  IMU_SEGMENTATION, OCCURRENCE_ALIGNED_SPIKE_ENCODING, and small necessary
  corrections in SPIKE_ENCODING/SPIKE_SEGMENTATION_PIPELINE/SEGMENT_PADDING/
  GRAVITY_TO_SPIKE_PIPELINE, plus task/workboard records.
- **Replan trigger:** code/test disagreement or an edit outside these paths.

## T004 — FROZEN TaskSpec

- **Goal:** Align public segmentation and spike notes with the T001 schemas
  and selector contracts.
- **Source plan:** §10 T004; **dependencies:** T001 (DONE);
  **validated against:** `34a79fc6a6d5942e082cc23ab6be4e5c20dd63db`.
- **Verified behavior:** raw-ring preprocessing/segmentation is 9-channel even
  for `raw`; Custom Wavelet provides 15 events plus source `[:, 3:9]` as the
  21-channel SpikeIMU; input kind and boundary mode are independent; Board
  mode requires a matching alignment artifact and has no label fallback.
- **Documentation changes:** remove six-channel raw-output and six/nine-only
  segmenter claims; correct stale strict-work-axis wording; retain current
  Board target/validation and variable-length/padding boundaries.
- **Allowed write paths:** `docs/notes/BOARD_EVENT_GUIDED_SEGMENTATION.md`,
  `docs/notes/IMU_SEGMENTATION.md`,
  `docs/notes/OCCURRENCE_ALIGNED_SPIKE_ENCODING.md`,
  `docs/notes/SPIKE_ENCODING.md`, `docs/notes/SPIKE_SEGMENTATION_PIPELINE.md`,
  this task file, and WORKBOARD.
- **Forbidden paths:** implementation, tests, README, `vendor/**`, sample
  data, and unrelated notes.
- **Acceptance:** public width/selector/alignment terminology is consistent;
  fresh verifier PASS.
- **Validation:** stale-term scan and independent verifier review.
- **Replan triggers:** code/test disagreement or scope expansion.

### T004 execution record

Repaired raw-mode output wording, restored public SpikeIMU segmenter support,
and updated dependent alignment terminology without changing accurate Board,
padding, or gravity-to-spike detail.

## T006 — revised draft

`PROJECT_REPORT.md` and `VENDOR_WINDOWING_REPORT.md` are explicitly deferred
and out of scope under the updated source plan. T006 is limited to any
remaining DATA_FORMAT classification needed after T003, and those deferred
reports are excluded from T006 and T007 acceptance scans.

### T006 revised requirements

- Correct DATA_FORMAT's stationary-search default to 0.10 seconds.
- Distinguish lower-level gravity defaults from public preprocessing-CLI
  defaults, without making either an acquisition-rate claim.
- Make its source facts, sample observations, repository interpretations, and
  repository policies visibly distinct; do not edit deferred reports.

## T006 — FROZEN TaskSpec

- **Goal:** Finish DATA_FORMAT's source/observation/interpretation/policy
  classification without touching deferred historical reports.
- **Dependencies:** T001 and T003 (DONE); **validated against:** baseline.
- **Required corrections:** stationary search default is 0.10 seconds;
  lower-level gravity defaults and preprocessing-CLI defaults are distinct;
  classification headings identify source facts, sample observations,
  repository interpretations, and policies.
- **Allowed paths:** `docs/notes/DATA_FORMAT.md`, task file, WORKBOARD.
- **Forbidden paths:** PROJECT_REPORT, VENDOR_WINDOWING_REPORT, runtime,
  tests, README, vendor, and sample data.
- **Acceptance:** no acquisition/timestamp guarantee is introduced and a fresh
  verifier PASS confirms the narrow change.

### T006 execution record

DATA_FORMAT now names its four epistemic categories, corrects the stationary
search default, and scopes preprocessing versus gravity defaults. Deferred
historical reports were not modified.

## T007 — FROZEN TaskSpec and execution record

- **Goal:** Verify the scoped documentation system after T002–T006.
- **Dependencies:** T002–T006 (DONE); **validated against:** current repairs.
- **Scope:** README and current technical notes only. `PROJECT_REPORT.md` and
  `VENDOR_WINDOWING_REPORT.md` are deferred/out of scope and explicitly
  excluded from scans and blocking criteria.
- **Required invariant review:** raw source versus preprocessed output;
  acquisition versus processing rate; canonical versus work axis; raw mode
  versus six signals; 15/21 and `0:15`; selectors; padding order; offset
  convention; and `PIPELINE_MODE` wording.
- **Result:** probe found no remaining scoped contradiction. Contextual
  `rawIMU`, 6/9, 200/201, legacy, and OVERWRITE hits are correctly classified.
- **Allowed write paths:** task/workboard only; no documentation edit required.
- **Acceptance:** independent verifier PASS; then mark T008 NOT REQUIRED and
  the plan DONE.

## T005 — FROZEN TaskSpec

- **Goal:** Correct Action-0 wrapper defaults and resume behavior.
- **Dependencies:** T001 (DONE); **validated against:** baseline commit.
- **Verified behavior:** `PIPELINE_MODE=continue` is primary; `overwrite`
  rebuilds from preprocessing; legacy `OVERWRITE` applies only if PIPELINE_MODE
  is absent; `MADGWICK_PROVISIONAL=0`; wrapper padding is `.99/balanced/1/0`.
- **Allowed paths:** `docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md`, this task
  file, and WORKBOARD. **Forbidden:** all runtime files and unrelated notes.
- **Acceptance:** default, precedence, rebuild semantics, wrapper matrix, and
  21-artifact/15-model relationship are accurate; verifier PASS.

## T001 — FROZEN TaskSpec

- **Goal:** Establish the verified common documentation vocabulary and
  baseline for Ring source data, preprocessing, spike artifacts, alignment,
  segmentation, padding, Action-0 input, and orchestration controls.
- **Source plan:** §7 and §10 T001 of
  `docs/plans/TODO/00_Documentation_Contract_Repair_Plan.md`.
- **Dependencies:** none.
- **Validated against commit:** `34a79fc6a6d5942e082cc23ab6be4e5c20dd63db`.
- **Verified behavior to record:**
  - Public Ring loading accepts only native-endian float64 `ring_0` rows of
    seven values; raw timestamp microseconds are a repository interpretation,
    not an upstream guarantee.
  - The public preprocessing path emits nine features; the public
    custom-wavelet path emits fifteen events and twenty-one-channel SpikeIMU.
  - Alignment preserves canonical timestamps, uses an endpoint-reconstructed
    work axis, and separately declares the work-axis offset and canonical
    projection status.
  - `raw-ring|spike-imu` and `label|aligned-board-events` are independent
    selectors; their public outputs are respectively nine and twenty-one
    channels and remain variable length before padding.
  - Padding right-pads SpikeIMU and creates valid-prefix masks; Action-0
    consumes the first fifteen channels, with masks applied to loss/count/
    regularization aggregation rather than recurrent-state evolution.
  - Pipeline controls default to `PIPELINE_MODE=continue`,
    `MADGWICK_PROVISIONAL=0`, nominal `SAMPLING_RATE=200`, and the Action-0
    wrapper's balanced padding recommendation.
- **Contracts to preserve:** no upstream acquisition-rate, timestamp-unit,
  or action-semantic guarantee is inferred; no runtime behavior changes.
- **Documentation distinctions intentionally changed:**
  - `15`/`21` widths describe the public custom-wavelet pipeline, not all
    generic helpers.
  - A failed canonical projection leaves a work-axis-domain alignment artifact
    with explicit status; it is not a canonical offset.
  - Analysis, standalone padding CLI, and Action-0 wrapper defaults are
    documented by their own scopes.
  - CLI preprocessing's low-pass default is not conflated with the lower-level
    gravity configuration's Madgwick default.
- **Allowed write paths:**
  `docs/plans/TODO/00_Documentation_Contract_Repair_TASKS.md` and
  `docs/plans/WORKBOARD.md` only.
- **Forbidden write paths:** `README.md`, `docs/notes/**`, implementation,
  tests, configuration, `vendor/**`, and `data_sample/**`.
- **Acceptance criteria:** record the confirmed shape and timestamp-domain
  baseline, including all scoped distinctions above; obtain a fresh
  `luna_verifier` PASS before marking T001 done.
- **Validation:** frozen-TaskSpec review against both CONFIRMED T001 probe
  packets and independent verifier review.
- **Replan triggers:** code/test disagreement; a needed undocumented upstream
  guarantee; or any baseline statement that would require a runtime change.

Downstream TaskSpecs remain intentionally unfrozen until T001 is verified.
