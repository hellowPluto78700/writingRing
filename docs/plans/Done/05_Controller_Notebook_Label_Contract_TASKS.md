# Controller Notebook Label Contract — Task DAG and TaskSpecs

## Plan identity

- **Source plan:** `docs/plans/TODO/05_Controller_Notebook_Label_Contract_Plan.md`
- **Primary contracts:** `README.md`, `docs/notes/DATA_FORMAT.md`, and
  `docs/notes/IMU_SEGMENTATION.md`

| Task | Goal | Dependencies | Execution owner | State |
| --- | --- | --- | --- | --- |
| T001 | Correct action-ID semantics and add a read-only timestamp-marker audit. | — | luna_worker after freeze | DONE |

## T001 — Draft TaskSpec

- **Goal:** Repair false action-directory-as-label/class statements in
  `notebooks/writingRing_preprocessing_spike_controller.ipynb` and make its
  label diagnostics use canonical timestamp sidecars.
- **Initial probe:** REVISE. The notebook currently calls `recording.action`
  a label/class; the revised contract must explicitly separate action IDs from
  timestamp labels and must not turn raw marker counts into segment/class
  targets.
- **Required behavior:**
  - Rename all action-derived notebook table, plot, export, and snapshot fields
    from `label`/`class` to `action_id` (or clearly qualified action directory
    ID). State that its semantic meaning is undocumented.
  - Add a read-only timestamp-marker audit using
    `writingring.segmentation.load_timestamp_labels()` for every available
    `recording.timestamp_path`. Preserve each marker's full original text,
    case, timestamp, source line, and recording identity; report missing
    sidecars separately and do not fall back to action IDs.
  - Display raw timestamp-marker totals and a marker-text distribution with an
    explicit `is_wrong_start_marker` indication. State that it is not a
    segmentation-export, Action0 class, or Board-contact-label distribution;
    `wrong` is case-insensitively an invalid segment-start marker.
  - Keep preprocessing/spike encoder controls and exports unchanged and off by
    default. Do not run segmentation or change its boundary rules.
- **Contracts to preserve:** primary `ring_0` selection; data hierarchy;
  action IDs as nonsemantic directory identifiers; timestamp-label parsing and
  preservation; no undocumented timestamp/unit inference; no data/vendor
  mutation.
- **Allowed writes:** `notebooks/writingRing_preprocessing_spike_controller.ipynb`
  only. **Forbidden:** every other path, including outputs, data sample,
  source, tests, scripts, configurations, vendor, and environments.
- **Acceptance/validation:** notebook JSON remains valid; no source claim
  equates action ID with label/class; sample/full data marker audit preserves
  text/case/source location; `wrong` is not presented as a class; missing
  sidecars are reported; focused segmentation tests and full pytest pass;
  `git diff --check` passes.
- **Replan triggers:** requested classification semantics cannot be derived
  from timestamp labels alone, marker audit requires segmentation/export, a
  malformed sidecar needs a recovery policy, or the notebook must change a
  non-notebook path.

## T001 — FROZEN TaskSpec: action-ID and timestamp-marker repair

- **Goal:** Make the controller notebook truthful about recording identity and
  timestamp labels while retaining its preprocessing/spike-controller role.
- **Source/dependencies:** user request; final T001 probe CONFIRMED against
  `51293739773d6a192ad8cdec18857d6ec4b52319`.
- **Action identity repair:** replace every source, markdown, displayed-table,
  export-table, and snapshot use of action-derived `label`/`class` with
  `action_id` (or explicitly named action-directory identifier). Replace
  action-derived “class” totals/distributions with action-ID totals/coverage.
  State clearly that action-ID meaning is undocumented; do not infer names.
- **Timestamp-marker audit:** add a read-only section after recording
  inventory. For every discovered recording, report missing
  `recording.timestamp_path` separately; otherwise call canonical
  `load_timestamp_labels()`. For each successfully parsed marker display
  `user`, `action_id`, `dataset_id`, sidecar path, `timestamp_us`,
  `source_line_number`, `timestamp_marker_text`, `raw_sidecar_line`, and
  `is_wrong_start_marker`. `timestamp_marker_text` is the parser's
  case-preserved label; `raw_sidecar_line` is the exact source line minus only
  its line terminator, retrieved by one-based source line number without
  manually reparsing marker semantics.
- **Error/meaning rules:** catch `SegmentLabelParseError` per sidecar and
  display a separate parse-error diagnostic, then continue auditing all other
  recordings. The raw-marker summary/distribution must be explicitly labelled
  as diagnostic only—not a segmentation export, Action0 target/class map, or
  Board-contact label distribution. Flag `timestamp_marker_text.casefold() ==
  "wrong"`; explain that this is an invalid segment-start marker, not a class.
  Do not call segmentation APIs, apply `searchsorted`, infer eligibility, or
  generate artifacts.
- **Notebook hygiene:** revise the opening/inventory markdown to distinguish
  action ID, timestamp-marker text, Action0 discovered segment classes, and
  Board contact fields. Clear existing notebook outputs/execution counts that
  retain the incorrect action-as-class claims; do not change unrelated source
  or optional export controls.
- **Contracts to preserve:** `ring_0` discovery, optional timestamp sidecar
  handling, exact parser validation/case preservation, preprocessing/spike
  encoder behavior, default-off exports, no timestamp/unit inference, and no
  data/output/vendor/source/test modification.
- **Allowed writes:** `notebooks/writingRing_preprocessing_spike_controller.ipynb`
  only. **Forbidden:** all other paths, including outputs, data sample, source,
  tests, scripts, config, README/docs, vendor, and environments.
- **Acceptance/validation:** notebook JSON/source audit confirms no action ID
  is called class/label; valid sidecars retain parser text/case and raw line
  provenance; missing and malformed sidecars remain visible without fallback;
  `wrong` is non-class diagnostic; optional exports remain false; focused
  discovery/segmentation tests and full pytest pass; `git diff --check` passes.
- **Replan triggers:** requirement for byte-exact file preservation beyond
  line-content provenance, raw marker inspection becomes a segment/class
  distribution, parser recovery needs a new project policy, or any write
  outside the notebook is required.

### T001 execution, verification, and documentation

The worker changed only the controller notebook, corrected all action-ID
terminology, added the canonical timestamp-marker/raw-line audit, and cleared
stale outputs. Real-data execution audited 606 recordings and 13,356 markers;
257 markers case-folded to `wrong`, with no missing sidecars or parse errors.
Focused discovery/segmentation tests passed 32; the full suite passed 527 with
one known skip; JSON/AST and whitespace audits passed. A fresh verifier
returned **PASS**, independently confirming the required semantic separation,
parser/diagnostic behavior, no segmentation or artifact operations, and
default-off exports. It noted an existing optional-export import issue when a
manual export switch is enabled; this is default-off and outside the frozen
label-contract scope. No README or technical-note update is needed because the
repair brings the notebook into alignment with existing documented contracts.
