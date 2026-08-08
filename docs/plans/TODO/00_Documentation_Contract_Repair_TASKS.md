# Documentation Contract Repair Plan

## 1. Status

* **Plan status:** DRAFT
* **Plan type:** Documentation contract repair
* **Primary owner:** PRIMARY
* **Companion task file:** `docs/plans/Documentation_Contract_Repair_TASKS.md`
* **Workboard:** `docs/plans/WORKBOARD.md`
* **Implementation code changes:** Not planned
* **Repository behavior changes:** Not planned
* **Primary objective:** Make `README.md` and `docs/notes/**` accurately describe the repository's current implemented contracts without silently changing implementation semantics.

This plan is a documentation-repair plan. It does not authorize changes to runtime behavior merely to make the existing documentation true.

If execution discovers that current code and tests disagree, or that the documentation describes an intended behavior that should supersede the implementation, that discrepancy must be handled through replanning and a separate implementation TaskSpec rather than being folded into this documentation repair.

---

## 2. Motivation

The repository has evolved beyond its original inspection and visualization scope. Current code now includes, among other capabilities:

* Ring recording discovery and loading;
* canonical nine-channel IMU preprocessing;
* multiple gravity-removal modes;
* spike encoding;
* occurrence-aligned spike encoding;
* Ring–Board alignment;
* label-driven segmentation;
* Board-event-guided segmentation;
* fixed-length segment analysis and padding;
* Action-0 SynNet training;
* orchestration scripts for the segmentation pipeline.

Several existing documentation files were written against earlier implementation phases. As a result, parts of `README.md` and `docs/notes/**` now mix:

1. verified source-format facts;
2. observed properties of sample datasets;
3. processing defaults;
4. current implementation contracts;
5. historical implementation behavior;
6. future-plan assumptions.

Those categories need to be separated explicitly.

The most important known mismatches are:

* README describes raw Ring data as being in “NumPy format”, although the source files are raw binary float64 streams read through NumPy rather than `.npy` files.
* README presents `200 Hz` as a source-data sampling fact, while the actual acquisition-rate contract is not verified and sample observations are approximately `201 Hz`; `200 Hz` is currently a processing/default assumption in several pipelines.
* README assigns semantic meanings to action IDs although those meanings are not established by the verified data-format contract.
* alignment documentation describes an obsolete strict-timestamp reconstruction strategy, while current alignment uses endpoint reconstruction for its work axis.
* alignment documentation does not consistently distinguish work-axis offset estimation from projection back to the canonical Ring timestamp domain.
* some segmentation notes describe raw mode as exporting the original six IMU channels, while the canonical preprocessed raw-mode feature representation is nine channels.
* occurrence-aligned spike documentation still states that the segmenter only supports six- or nine-channel input, while the current segmenter supports the 21-channel SpikeIMU representation.
* segmentation shell-script documentation states an outdated `MADGWICK_PROVISIONAL=1` default.
* shell-script documentation does not fully describe the current `PIPELINE_MODE=continue|overwrite` resume behavior.
* `PROJECT_REPORT.md` describes an older project phase as though it were the current/final repository architecture.
* portions of `VENDOR_WINDOWING_REPORT.md` that discuss the surrounding project state are historical even though its vendor-code observations remain useful.

The documentation therefore needs a coordinated contract repair rather than isolated wording patches.

---

## 3. Repository Authority Model

This plan follows the repository knowledge hierarchy defined by `AGENTS.md`.

The relevant authority levels are:

1. **`docs/notes/**`**

   * durable verified technical contracts and implementation knowledge;
   * should describe verified behavior, not unverified assumptions.

2. **`README.md`**

   * repository orientation and user-facing entry points;
   * should summarize stable repository behavior and point readers to detailed notes.

3. **current code and tests**

   * implementation reality;
   * used to validate whether a documentation statement is currently true.

4. **`docs/plans/**`**

   * future work, intended changes, orchestration, and task specifications;
   * may intentionally differ from current implementation.

5. **frozen TaskSpec**

   * authoritative implementation contract for the task to which it applies.

When a README, note, plan, source file, and/or test disagree, the repair process must not silently choose whichever statement is easiest to document.

The discrepancy must first be classified as one of:

* documentation stale;
* documentation intentionally historical;
* implementation stale;
* test stale;
* unresolved contract ambiguity.

Only the first two classes are directly within this plan.

---

## 4. Goals

This plan has seven primary goals.

### G1. Establish one coherent documentation vocabulary

Use consistent terminology for:

* raw Ring source data;
* preprocessed IMU features;
* spike event features;
* SpikeIMU features;
* canonical timestamps;
* alignment work-axis timestamps;
* alignment offsets;
* segmentation boundary sources;
* padded model inputs.

### G2. Separate source facts from processing assumptions

In particular:

* do not describe a processing nominal sampling rate as a verified acquisition rate;
* do not describe inferred timestamp units as upstream format guarantees;
* do not describe observed sample behavior as universal file-format behavior.

### G3. Make channel-count contracts unambiguous

The documentation should consistently distinguish:

| Artifact                                |        Current contract |
| --------------------------------------- | ----------------------: |
| Raw Ring source record                  | 7 float64 values/sample |
| Raw IMU measurements inside Ring source |  6 signal values/sample |
| Preprocessed IMU feature matrix         |              9 channels |
| Spike event matrix                      |             15 channels |
| SpikeIMU feature matrix                 |             21 channels |
| Segmented preprocessed/raw-mode IMU     |              9 channels |
| Segmented SpikeIMU                      |             21 channels |
| Padded SpikeIMU                         |             21 channels |
| Action-0 SynNet model input             | first 15 event channels |

The phrase “raw six-channel output” must not be used for the canonical preprocessed raw-mode output.

### G4. Correct the alignment contract

Documentation must accurately represent:

* canonical timestamps;
* the reconstructed alignment work axis;
* endpoint reconstruction;
* metadata/provenance sampling rate versus alignment endpoint-derived rate;
* work-axis offset estimation;
* projection of that offset into the canonical timestamp domain;
* export requirements and failure behavior.

### G5. Make segmentation and spike documentation internally consistent

Documentation must consistently describe the current combinations of:

* `raw-ring` versus `spike-imu`;
* `label` versus `aligned-board-events`;
* 9-channel versus 21-channel feature matrices;
* Board event targets;
* timestamp and sampling-rate validation;
* skip rules and output artifacts.

### G6. Distinguish current documentation from historical reports

Reports created for earlier phases should remain useful historical records without presenting old scope statements as the current repository architecture.

### G7. Establish a repeatable documentation verification gate

After repair, a targeted repository-wide contract scan should detect obvious regressions such as:

* stale channel counts;
* obsolete alignment strategy names;
* obsolete pipeline defaults;
* historical “current project does not support X” statements.

---

## 5. Non-Goals

This plan does **not** attempt to:

* change Ring binary parsing;
* change timestamp interpretation;
* establish an undocumented upstream acquisition-rate guarantee;
* establish undocumented action-ID semantics;
* change preprocessing algorithms;
* change gravity-removal algorithms;
* change spike encoding;
* change Ring–Board alignment behavior;
* change segmentation behavior;
* change padding behavior;
* change SNN architecture or training;
* modify tests merely to match documentation;
* refactor implementation code;
* modify sample data;
* modify vendor code;
* reproduce or modernize every historical report;
* infer undocumented physical units, channel meanings, timestamp semantics, or acquisition conventions.

Any such requirement must be replanned separately.

---

## 6. Allowed and Forbidden Write Scope

PRIMARY owns the documentation repair.

### 6.1 Allowed write paths

The default allowed paths for this plan are:

```text
README.md
docs/notes/**
docs/plans/Documentation_Contract_Repair_Plan.md
docs/plans/Documentation_Contract_Repair_TASKS.md
docs/plans/WORKBOARD.md
```

A frozen TaskSpec should normally narrow this further to the exact files required by that task.

### 6.2 Forbidden write paths

Unless the user explicitly overrides the scope and a separate implementation task is created, this plan must not modify:

```text
src/**
scripts/**
snn/**
tests/**
configs/**
AGENTS.md
.codex/**
vendor/**
data_sample/**
```

### 6.3 Vendor inspection restriction

`vendor/**` must not be re-inspected as part of routine documentation repair.

Existing verified historical findings may be retained.

If a task specifically requires new comparison against vendor behavior, that requirement must be written explicitly into the corresponding TaskSpec before inspection.

---

## 7. Contract Baseline

Before editing individual documents, the execution must freeze a common documentation baseline.

The baseline is organized into six contract groups.

---

## 7.1 Raw Ring Source Contract

Current documentation should distinguish the source container from the logical signal channels.

### Source representation

A Ring source file is a raw binary stream consumed as native-endian float64 values and reshaped into rows of seven values.

It is not a NumPy `.npy` file merely because NumPy is used to read it.

Conceptually:

```text
Ring source row:
[
    acceleration_x,
    acceleration_y,
    acceleration_z,
    gyroscope_x,
    gyroscope_y,
    gyroscope_z,
    timestamp
]
```

Current repository support selects:

```text
*_ring_0.bin
```

as the supported Ring recording source.

`*_ring_1.bin` must not be presented as an equivalent supported input stream.

### Sampling-rate wording

Documentation must separate:

* **verified raw format facts**;
* **observed sample timing**;
* **processing/default nominal sampling rate**.

`200 Hz` must not be stated as a verified universal acquisition-rate property unless upstream evidence establishes that contract.

Where relevant, documentation may state that current processing pipelines commonly use a nominal/default rate of `200 Hz`.

### Timestamp wording

Documentation must not elevate inferred timestamp units into a guaranteed upstream storage contract.

If microsecond interpretation is required by current processing, state that as the current repository interpretation/contract at the relevant layer.

### Action IDs

Action IDs may be documented as directory/action identifiers.

Human semantic labels for action IDs must not be stated as verified source-format facts unless a reliable repository contract establishes them.

---

## 7.2 Preprocessed IMU Contract

Canonical preprocessed IMU data is a nine-channel feature representation.

The schema is:

```text
0: acceleration_x_g
1: acceleration_y_g
2: acceleration_z_g

3: acceleration_x_m_s2
4: acceleration_y_m_s2
5: acceleration_z_m_s2

6: gyroscope_x
7: gyroscope_y
8: gyroscope_z
```

Exact human-readable names should follow the implementation metadata where a document needs canonical field names.

Supported preprocessing/gravity-removal modes include the currently implemented modes:

* raw;
* low-pass;
* Madgwick;
* Xylo.

### Raw mode

`raw` means that gravity removal is bypassed.

It does **not** mean that the pipeline emits a six-column source matrix.

Raw mode still emits the canonical nine-channel preprocessed representation.

For raw mode, the acceleration-with-gravity measurements are preserved in the acceleration portion of that nine-channel schema, with corresponding `g` and `m/s²` representations.

Documentation must therefore avoid statements such as:

> raw mode exports the original six Ring IMU channels

when referring to preprocessed or segmented pipeline output.

If the six original measurement values are discussed, they must be described explicitly as the six sensor-signal values contained inside the seven-value raw Ring source row.

---

## 7.3 Spike Feature Contract

### Spike events

The current custom-wavelet spike-event representation contains:

```text
15 event channels
```

generated from:

```text
3 acceleration axes × 5 configured frequencies
```

with the currently documented default frequencies:

```text
0.5, 1, 2, 4, 8 Hz
```

where applicable.

### SpikeIMU

The current SpikeIMU representation contains:

```text
21 channels
```

formed conceptually as:

```text
15 spike-event channels
+
6 IMU feature channels
```

The six appended channels correspond to the source preprocessed matrix slice:

```text
source[:, 3:9]
```

That is:

* three acceleration channels in `m/s²`;
* three gyroscope channels.

The 21-channel SpikeIMU contract must be used consistently across:

* spike encoding notes;
* segmentation notes;
* padding notes;
* model-input notes.

### Action-0 model input

The Action-0 SynNet baseline consumes only:

```text
SpikeIMU[:, :, 0:15]
```

as model features.

The remaining six channels are retained in the SpikeIMU artifact but are not part of the current baseline model input.

---

## 7.4 Alignment Contract

Alignment documentation must distinguish two timestamp domains.

### Canonical timestamp axis

The canonical Ring/SpikeIMU timestamps are preserved as source/provenance timestamps.

They are the timestamp domain to which exported alignment offsets ultimately need to refer.

### Alignment work axis

The current alignment implementation constructs a strictly increasing work axis from the canonical endpoints.

For a canonical timestamp vector with `N` rows:

```text
work_axis = linspace(
    canonical_timestamp[0],
    canonical_timestamp[-1],
    N
)
```

subject to the implementation's timestamp validation requirements.

The current strategy name is:

```text
endpoint_reconstruction
```

Documentation must not describe current behavior as:

```text
strict_reconstruction
```

unless discussing historical behavior.

### Feature sampling-rate metadata

A supplied feature sampling rate is metadata/provenance.

It does not determine the current alignment work-axis spacing.

The work-axis rate is derived from the canonical endpoint duration and sample count.

Documentation must not conflate:

```text
feature sampling rate
```

with:

```text
alignment endpoint-derived work-axis rate
```

### Offset domains

Alignment produces an estimated offset in the alignment work-axis domain.

The documentation must distinguish:

```text
best work-axis offset
```

from:

```text
projected canonical timestamp offset
```

The exported reusable alignment offset is the canonical-domain projection, when projection succeeds.

### Offset convention

Where the current implementation uses the mapping:

```text
ring_timestamp_us = board_timestamp_us + offset_us
```

documentation should use that convention consistently.

### Export failure behavior

A numerically finite work-axis match is not by itself sufficient to claim that a reusable canonical alignment offset was exported.

The relevant documentation must reflect canonical projection status and the implementation's successful-alignment requirements.

---

## 7.5 Segmentation Contract

Segmentation now has two independent dimensions:

### Input kind

```text
raw-ring
spike-imu
```

### Boundary mode

```text
label
aligned-board-events
```

Documentation should not conflate these two choices.

For example:

```text
spike-imu + label
```

is valid independently of:

```text
spike-imu + aligned-board-events
```

### Raw-Ring segmentation feature width

When raw Ring input goes through the current preprocessing path, segmented IMU features use the canonical nine-channel representation.

### SpikeIMU segmentation feature width

SpikeIMU segmentation uses the current 21-channel SpikeIMU representation.

The documentation must remove the obsolete statement that the segmenter accepts only six- or nine-channel input.

### Label semantics

In label mode, labels represent segmentation boundaries according to the current implementation contract.

The documentation should preserve distinctions between:

* label boundaries;
* Board press/lift events;
* transient detection signals.

### Board-event-guided segmentation

Aligned Board mode requires a valid alignment artifact according to the current implementation contract.

There is no silent fallback to unaligned operation.

The documentation should preserve the distinction between:

* valid Board press/lift targets;
* transient press/lift targets;
* feature channels used to detect alignment;
* segment boundary logic.

### Output naming

If artifact names such as `rawIMU` remain part of the public output format, documentation should explain that this is an artifact/file naming convention.

It must not imply that the stored matrix is the original six-signal raw Ring source format.

---

## 7.6 Padding and Action-0 Training Contract

### Variable-length segmentation

Current segmentation emits variable-length segments represented using concatenated data plus offsets/length metadata.

Padding/truncation is a separate downstream stage.

### Fixed-length padding

The padding stage:

* selects a target length;
* right-pads shorter or equal-length segments;
* does not truncate longer segments;
* skips overflow segments according to current policy;
* produces valid-length metadata;
* produces a valid-prefix mask;
* retains the 21-channel SpikeIMU representation for SpikeIMU input.

### Default target selection

Where the relevant note documents current defaults, preserve the currently implemented defaults including:

```text
minimum coverage = 0.99
selection mode = balanced
rounding = 1
padding value = 0
```

subject to verification against implementation during the task probe.

### Action-0 model input and masking

The current Action-0 SynNet pipeline consumes:

```text
padded SpikeIMU: 21 channels
model input: first 15 event channels
```

Loss and metrics use the valid mask to exclude padded output time steps.

Documentation should also preserve the implementation nuance that the current mask does not terminate recurrent state evolution at padded steps; it masks the output aggregation/loss path according to the current implementation.

---

## 8. Documents in Scope

The following files are known to require review under this plan.

### Primary repair targets

```text
README.md
docs/notes/ALIGNMENT_OUTPUTS.md
docs/notes/BOARD_EVENT_GUIDED_SEGMENTATION.md
docs/notes/DATA_FORMAT.md
docs/notes/IMU_SEGMENTATION.md
docs/notes/OCCURRENCE_ALIGNED_SPIKE_ENCODING.md
docs/notes/PROJECT_REPORT.md
docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md
docs/notes/VENDOR_WINDOWING_REPORT.md
```

### Verification-only / expected-minimal-change targets

These notes currently appear substantially consistent with the implementation but must still participate in cross-document verification:

```text
docs/notes/ACTION0_SNN_TRAINING.md
docs/notes/GRAVITY_TO_SPIKE_PIPELINE.md
docs/notes/SEGMENT_PADDING.md
docs/notes/SPIKE_ENCODING.md
docs/notes/SPIKE_SEGMENTATION_PIPELINE.md
docs/notes/XYLO_GRAVITY_REMOVAL.md
```

“Verification-only” does not forbid an edit if the common terminology repair requires a small correction, but broad rewriting is not planned.

---

## 9. Task DAG

The documentation repair is decomposed as:

```text
T001
 ├── T002
 ├── T003
 ├── T004
 ├── T005
 └── T006
       \
        +----> T007
T002 --------/
T003 -------/
T004 ------/
T005 -----/

T008 is conditional and is created/replanned only if T001–T007
discover a genuine implementation-contract problem.
```

Equivalent dependency summary:

| Task | Goal                                               | Depends on  | Initial state |
| ---- | -------------------------------------------------- | ----------- | ------------- |
| T001 | Establish verified documentation baseline          | —           | DRAFT         |
| T002 | Repair README                                      | T001        | DRAFT         |
| T003 | Repair alignment contracts                         | T001        | DRAFT         |
| T004 | Repair segmentation and spike contracts            | T001        | DRAFT         |
| T005 | Repair Action-0 orchestration documentation        | T001        | DRAFT         |
| T006 | Separate historical reports from current contracts | T001        | DRAFT         |
| T007 | Cross-document contract verification               | T002–T006   | DRAFT         |
| T008 | Replan genuine implementation inconsistencies      | conditional | DRAFT         |

After T001 is complete, T002–T006 are logically parallel where the execution environment supports independent work.

T007 must wait for all documentation repair tasks.

---

# 10. Task Details

## T001 — Establish Documentation Contract Baseline

### Goal

Establish the verified vocabulary and invariants used by all downstream documentation tasks.

### Required investigation

Verify the current implementation contract for:

1. Ring source representation;
2. supported Ring stream;
3. timestamp handling;
4. preprocessing output schema;
5. gravity-removal method semantics;
6. spike event schema;
7. SpikeIMU schema;
8. alignment timestamp domains;
9. alignment strategy and offset projection;
10. segmentation input kinds;
11. segmentation boundary modes;
12. segmentation output schemas;
13. fixed-length padding;
14. Action-0 model input;
15. orchestration defaults.

### Required baseline matrix

The task must either confirm or revise the following table:

| Layer                               | Expected shape       |
| ----------------------------------- | -------------------- |
| Raw Ring source                     | `N × 7`              |
| Preprocessed IMU                    | `N × 9`              |
| Spike events                        | `N × 15`             |
| SpikeIMU                            | `N × 21`             |
| Segmented preprocessed/raw-mode IMU | `total_samples × 9`  |
| Segmented SpikeIMU                  | `total_samples × 21` |
| Padded SpikeIMU                     | `segments × T × 21`  |
| Action-0 SynNet input               | `segments × T × 15`  |

### Timestamp baseline

Confirm or revise:

```text
canonical timestamps
    = immutable source/provenance and segmentation timestamp axis

alignment work axis
    = endpoint-reconstructed strictly increasing coordinate used for alignment

alignment work-axis offset
    = matching result in work-axis coordinates

exported alignment offset
    = projected Board-to-canonical-Ring offset
```

### Output

T001 should produce a compact `CONTRACT_PROBE_PACKET` suitable for freezing T002–T006 TaskSpecs.

The packet should identify every statement as one of:

```text
CONFIRMED
REVISE
BLOCKED
```

### Replan condition

If implementation and tests disagree on any contract that materially affects the documentation, T001 must not declare the conflicting contract confirmed.

The affected downstream task remains blocked pending PRIMARY decision.

---

## T002 — Repair README

### Goal

Turn `README.md` into a concise, trustworthy current entry point without duplicating the technical notes.

### Required changes

#### A. Raw Ring format wording

Replace wording equivalent to:

```text
Ring data is stored in NumPy format
```

with language that correctly describes raw binary float64 source files read using NumPy.

Clearly distinguish:

```text
7 values/source row
=
6 IMU measurements + timestamp
```

#### B. Sampling-rate wording

Remove any implication that `200 Hz` is a verified universal acquisition-rate fact.

Prefer wording equivalent to:

```text
Current processing pipelines commonly use a nominal/default 200 Hz
configuration; the raw acquisition-rate contract is not established here.
```

Do not add a hard `201 Hz` acquisition guarantee merely because that rate was observed in sample data.

#### C. Action-ID semantics

Remove or explicitly qualify unverified semantic mappings for action IDs.

README may still describe the directory structure:

```text
data/{user}/{action}/...
```

without claiming undocumented action meanings.

#### D. Current repository capabilities

Update the high-level architecture so the README no longer appears to describe only the original plotting/inspection phase.

A concise current flow should cover:

```text
Ring recording
    ↓
preprocessing / gravity handling
    ↓
optional spike encoding
    ↓
optional Ring–Board alignment
    ↓
label or Board-event-guided segmentation
    ↓
segment-length analysis / fixed-length padding
    ↓
Action-0 SynNet training
```

The README should stay orientation-focused.

Detailed algorithmic contracts should link to the relevant `docs/notes/**` files instead of being duplicated.

#### E. Vendor scripts

If vendor plotting utilities are mentioned, make clear that they live under `vendor/WritingRing/` and are historical/upstream utilities rather than the primary implementation architecture.

### Acceptance criteria

* README does not call raw Ring `.bin` data a NumPy file format.
* README does not claim `200 Hz` as a verified raw acquisition contract.
* README does not present unverified action semantics as fact.
* README exposes the current preprocessing → spike/alignment → segmentation → padding → SNN pipeline at orientation level.
* README terminology agrees with T001.

---

## T003 — Repair Alignment Documentation

### Primary files

```text
docs/notes/ALIGNMENT_OUTPUTS.md
docs/notes/DATA_FORMAT.md
```

Additional alignment references elsewhere may be edited if required for consistency.

### Goal

Make timestamp-domain and offset-domain behavior explicit and consistent with the current alignment implementation.

### Required changes

#### A. Replace obsolete work-axis reconstruction description

Remove current-contract language that says duplicate timestamps specifically trigger rate-driven strict reconstruction.

Document the current rule:

```text
alignment work axis is reconstructed from canonical endpoints
for both raw-ring and spike-imu inputs
```

using the implementation's endpoint reconstruction behavior.

Current strategy terminology:

```text
endpoint_reconstruction
```

#### B. Separate feature sampling rate from work-axis rate

Make clear:

```text
feature sampling rate
    = metadata / feature provenance

endpoint-derived rate
    = implied by canonical endpoints and row count for alignment coordinates
```

A supplied feature sampling rate does not currently define alignment-axis spacing.

#### C. Preserve canonical timestamps

Document that canonical timestamps are not overwritten merely to manufacture a strictly increasing alignment coordinate.

The alignment work axis is a separate coordinate.

#### D. Distinguish two offsets

Use explicit terminology for:

1. work-axis matching result;
2. projected canonical-domain alignment offset.

Avoid generic wording such as “the best offset is directly exported” if the actual reusable artifact depends on canonical projection.

#### E. Offset mapping

State the current convention consistently:

```text
ring_timestamp_us = board_timestamp_us + offset_us
```

where appropriate.

#### F. DATA_FORMAT derived-processing section

Keep durable raw-source facts separate from later derived processing behavior.

Do not mix alignment implementation details into the raw binary-format contract without clearly labeling them as downstream processing.

### Acceptance criteria

* No current-behavior section uses `strict_reconstruction` as the implemented strategy.
* Both raw-ring and spike-imu alignment describe endpoint reconstruction consistently.
* Feature rate and endpoint-derived work-axis rate are distinguished.
* Canonical timestamps and work-axis timestamps are distinct concepts.
* Work-axis offset and canonical exported offset are distinct concepts.
* Alignment export semantics match the actual projection requirement.

---

## T004 — Repair Segmentation and Spike Documentation

### Primary files

```text
docs/notes/BOARD_EVENT_GUIDED_SEGMENTATION.md
docs/notes/IMU_SEGMENTATION.md
docs/notes/OCCURRENCE_ALIGNED_SPIKE_ENCODING.md
```

### Verification targets

```text
docs/notes/SPIKE_ENCODING.md
docs/notes/SPIKE_SEGMENTATION_PIPELINE.md
docs/notes/SEGMENT_PADDING.md
docs/notes/GRAVITY_TO_SPIKE_PIPELINE.md
```

### Goal

Make all feature-width, input-kind, boundary-mode, and spike-segmentation statements consistent.

### Required changes

#### A. Remove “raw mode exports six channels”

Replace such wording with the canonical distinction:

```text
Raw Ring source:
    six IMU measurements + timestamp

raw preprocessing mode:
    preserves gravity but still emits the nine-channel
    canonical preprocessed representation
```

#### B. Correct SpikeIMU segmenter support

Remove the obsolete statement that the segmenter accepts only six- or nine-channel input.

Document current support for:

```text
--input-kind spike-imu
```

and the 21-channel SpikeIMU representation.

#### C. Unify channel terminology

Use the following meanings consistently:

```text
raw Ring source       N × 7
preprocessed IMU      N × 9
spike events          N × 15
SpikeIMU              N × 21
```

Do not use “raw IMU” without enough context to distinguish source measurements from an output artifact whose filename contains `rawIMU`.

#### D. Separate input kind from boundary mode

Documentation should explain independently:

```text
input kind:
    raw-ring | spike-imu

boundary mode:
    label | aligned-board-events
```

#### E. Board-event-guided behavior

Preserve/document the verified current semantics for:

* alignment required;
* no silent unaligned fallback;
* Board press/lift target generation;
* transient target generation;
* crossing behavior;
* missing-event behavior;
* label validation/skip rules;
* feature sampling-rate validation.

Do not unnecessarily rewrite already accurate detailed sections.

#### F. Occurrence-aligned encoding

Preserve the verified occurrence-aligned custom-wavelet behavior, including current feature widths.

Only repair claims made obsolete by the newer segmentation support unless another discrepancy is found by T001.

### Acceptance criteria

* No note claims canonical raw-mode preprocessing/segmentation outputs six channels.
* SpikeIMU is consistently documented as 21 channels.
* The segmenter is documented as supporting `spike-imu`.
* Input kind and boundary mode are clearly independent.
* The relationship between 15 event channels and six appended IMU channels is consistent across notes.
* Padding and downstream model notes remain consistent with the repaired schema.

---

## T005 — Repair Action-0 Orchestration Documentation

### Primary file

```text
docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md
```

### Verification file

```text
docs/notes/ACTION0_SNN_TRAINING.md
```

### Goal

Correct orchestration defaults and resume behavior without changing pipeline execution.

### Required changes

#### A. Madgwick provisional default

Correct the documented default to the current implementation value:

```text
MADGWICK_PROVISIONAL=0
```

unless T001 finds the implementation has changed before this task is frozen.

#### B. Pipeline mode

Make the current primary control explicit:

```text
PIPELINE_MODE=continue
PIPELINE_MODE=overwrite
```

Describe `continue` as the normal resume/validation mode.

It should communicate that existing complete outputs may be retained while invalid/partial stage results may be rebuilt according to current script behavior.

#### C. Legacy OVERWRITE handling

Document `OVERWRITE` as a legacy compatibility input rather than the primary current execution interface if that remains the current implementation.

#### D. Preserve verified defaults

Keep the currently implemented padding defaults aligned:

```text
coverage threshold: 0.99
selection policy: balanced
rounding: 1
padding value: 0
```

Also preserve the correct processing defaults for the wrapper matrix after verification.

### Acceptance criteria

* `MADGWICK_PROVISIONAL` default matches implementation.
* `PIPELINE_MODE` is the primary documented resume/rebuild control.
* `OVERWRITE` is correctly labeled according to current compatibility semantics.
* The eight-wrapper matrix remains correct.
* Action-0 training documentation remains consistent with padded 21-channel input artifacts and 15-channel model features.

---

## T006 — Historical Reports and DATA_FORMAT Cleanup

### Primary files

```text
docs/notes/PROJECT_REPORT.md
docs/notes/VENDOR_WINDOWING_REPORT.md
docs/notes/DATA_FORMAT.md
```

### Goal

Preserve historical research value without allowing historical statements to masquerade as current repository contracts.

### A. PROJECT_REPORT.md

`PROJECT_REPORT.md` originated from an earlier documentation task focused on the local data inspection and visualization project.

It should not be continuously rewritten into a completely different report every time the repository gains capabilities.

Instead:

1. add a prominent historical-snapshot notice;
2. identify the approximate project phase/scope represented by the report;
3. qualify statements such as:

   * “current implementation”;
   * “final report”;
   * “non-goals”;
   * command-count/test-count statements;
4. explicitly state that later repository capabilities are documented elsewhere;
5. link readers to the current README and relevant technical notes.

Statements saying that the project does not implement alignment, reusable segmentation, machine-learning preprocessing/training, etc. must not remain presented as current repository facts.

The body may otherwise remain a useful record of the inspection/visualization phase.

### B. VENDOR_WINDOWING_REPORT.md

Preserve verified vendor findings such as:

* rolling/FIFO window behavior;
* lack of emitted ML sliding windows where that was the finding;
* plotting behavior;
* historical lack of explicit Ring–Board synchronization in the vendor implementation.

However:

* mark project-state conclusions as historical where necessary;
* mark “future work” written for that earlier repository phase as historical;
* avoid asserting that the present repository still lacks capabilities added later.

Do not re-inspect `vendor/**` unless the frozen TaskSpec explicitly authorizes historical comparison.

### C. DATA_FORMAT.md

Reorganize or clarify the document so a reader can distinguish:

#### Source-format facts

Examples:

```text
binary row width
column layout
supported ring stream
```

#### Observed dataset properties

Examples:

```text
approximately observed sample interval/rate
timestamp duplication seen in sample files
```

#### Repository interpretation

Examples:

```text
timestamp unit expected by current processing
```

#### Derived processing contracts

Examples:

```text
preprocessed 9-channel representation
alignment work axis
offset projection
```

Historical findings should not be phrased as immutable upstream-format guarantees.

### Acceptance criteria

* `PROJECT_REPORT.md` can no longer reasonably be mistaken for the current complete architecture report.
* Historical non-goals are explicitly historical.
* Vendor findings remain usable without implying the present project is unchanged.
* `DATA_FORMAT.md` cleanly separates raw source facts, observations, repository interpretations, and downstream derived representations.

---

## T007 — Cross-Document Contract Verification

### Goal

Verify the repaired documentation as one system rather than merely checking each file independently.

### Required stale-term scan

Search the user-facing documentation for variants of at least:

```text
6 channels
six channels
9 channels
15 channels
21 channels

200 Hz
201 Hz

strict_reconstruction
endpoint_reconstruction

MADGWICK_PROVISIONAL

segmenter still accepts
only six
only nine

rawIMU
raw IMU

current implementation
current project
does not implement
non-goals
final report

OVERWRITE
PIPELINE_MODE
```

Each hit must be classified rather than mechanically replaced.

A historical report may legitimately contain old terminology if it is clearly labeled as historical.

### Required invariant review

Verify all of the following.

#### I1. Raw source is not preprocessed output

```text
raw Ring source != preprocessed IMU feature matrix
```

#### I2. Acquisition rate is not processing rate

```text
processing nominal/default rate != verified upstream acquisition-rate contract
```

#### I3. Canonical timestamps are not alignment work-axis timestamps

```text
canonical timestamp axis != reconstructed work axis
```

#### I4. Alignment signal and segmentation boundary are different concepts

A transient feature used for alignment does not automatically define segment boundaries.

#### I5. Raw gravity mode does not mean six-column output

```text
gravity-removal method raw
!=
six-channel canonical output
```

#### I6. Current SpikeIMU width is 21

All current-contract notes must agree.

#### I7. Action-0 consumes event channels

Current baseline model input is:

```text
channels 0:15
```

rather than the full 21 channels.

#### I8. Padding is downstream of variable-length segmentation

Segmentation should not be documented as implicitly padding or truncating the current variable-length artifacts.

#### I9. Board alignment offset convention is consistent

Documents that state the mapping should agree on:

```text
ring_timestamp_us = board_timestamp_us + offset_us
```

#### I10. Historical docs are explicitly historical

Old project-state claims must not be indistinguishable from current contracts.

### Optional test sanity check

Because this plan is documentation-only, `AGENTS.md` does not require a full test run solely because Markdown changed.

However, when the execution environment is available, a final sanity test may run:

```bash
conda run -n writingring-viz python -m pytest -q
```

The result should be reported exactly as observed.

A test run must never be claimed if it was not actually executed.

### Acceptance criteria

T007 passes only when:

* all scoped docs use compatible schemas and terminology;
* every known stale statement is repaired, intentionally historical, or explicitly documented as unresolved;
* no documentation edit accidentally claims an implementation behavior unsupported by code/tests;
* no implementation changes were smuggled into the documentation task.

---

## T008 — Conditional Implementation Replan

### Trigger

T008 is not automatically executed.

It is triggered only if investigation or verification finds a discrepancy of one of these forms:

```text
code and tests disagree
```

or:

```text
verified durable note and tests agree,
but current implementation violates that contract
```

or:

```text
the requested documentation repair would require changing
runtime behavior to make the documentation true
```

### Required action

Do not “fix” the discrepancy by rewriting the durable contract to match whichever implementation happens to exist.

Instead:

1. record the discrepancy;
2. block the affected documentation task if necessary;
3. return the task to PRIMARY;
4. create or update a separate implementation plan;
5. create a dependency-aware implementation TaskSpec;
6. follow the normal probe → freeze → worker → verifier lifecycle.

T008 implementation work is outside the default write scope of this plan.

---

# 11. Task Lifecycle

Each task follows the repository lifecycle:

```text
DRAFT
  ↓
PROBING
  ↓
FROZEN
  ↓
IMPLEMENTING
  ↓
VERIFYING
  ↓
DOCUMENTING
  ↓
DONE
```

For documentation-only tasks, “IMPLEMENTING” means applying the authorized documentation edits.

A task may move backward when evidence invalidates its TaskSpec.

For example:

```text
VERIFYING
  ↓
PROBING
```

is valid if verification exposes an unverified assumption.

---

## 12. Probe Requirements

Before freezing a dependency-ready task, the assigned probe must return enough evidence to remove implementation ambiguity.

The probe result should be compact and should not contain unnecessary source dumps.

Recommended structure:

```text
CONTRACT_PROBE_PACKET

Task:
Status: CONFIRMED | REVISE | BLOCKED

Verified contracts:
- ...

Documentation mismatches:
- ...

Relevant implementation locations:
- ...

Relevant tests:
- ...

Historical context:
- ...

Unresolved questions:
- ...

Recommended TaskSpec changes:
- ...
```

PRIMARY uses this packet to freeze the TaskSpec.

A probe must not directly convert an uncertain inference into durable documentation.

---

# 13. Frozen TaskSpec Requirements

The companion `Documentation_Contract_Repair_TASKS.md` should contain the executable TaskSpecs.

Each frozen TaskSpec should contain at minimum:

```text
Task ID
Goal
Source plan
Dependencies
Verified behavior
Contracts to preserve
Contracts to change in documentation
Allowed write paths
Forbidden write paths
Acceptance criteria
Validation commands/checks
Replan triggers
Relevant verified commit, if useful
```

The plan intentionally keeps the DAG coarse.

Detailed implementation instructions should be frozen only when a task becomes dependency-ready.

In other words:

```text
DAG early.
TaskSpec late.
```

---

# 14. Documentation Editing Rules

All documentation edits under this plan must follow these rules.

## 14.1 Prefer explicit epistemic status

Use wording that tells the reader whether a statement is:

* verified format contract;
* repository assumption;
* processing default;
* observed sample property;
* historical behavior.

Avoid phrases that erase that distinction.

## 14.2 Do not invent upstream guarantees

Do not infer undocumented:

* sampling frequency;
* timestamp unit;
* action semantics;
* channel semantics;
* device behavior.

## 14.3 Avoid duplicated algorithm specifications

Detailed behavior should live in the most relevant technical note.

README should summarize and link.

This reduces future documentation drift.

## 14.4 Preserve useful historical knowledge

Do not delete old investigation reports simply because the project advanced.

Label their temporal scope instead.

## 14.5 Prefer schema tables

Where channel widths or artifacts are important, use compact tables rather than prose that repeatedly says “six”, “nine”, “fifteen”, or “twenty-one” without naming the artifact.

## 14.6 Use precise “raw” terminology

The word `raw` is overloaded in the repository.

When ambiguity exists, use one of:

```text
raw Ring source
raw sensor measurements
raw preprocessing mode
raw-mode preprocessed IMU
rawIMU artifact
```

rather than an unqualified `raw IMU`.

## 14.7 Keep canonical/work-axis terminology stable

Use:

```text
canonical timestamps
alignment work axis
work-axis offset
projected canonical offset
```

consistently.

Do not alternate between multiple undocumented names for the same coordinate.

---

# 15. Validation Strategy

Documentation repair has three levels of validation.

## Level 1 — File-local validation

For each edited document:

* links remain valid where locally checkable;
* examples use correct artifact names;
* channel-count claims are internally consistent;
* flags/defaults agree within the file;
* current and historical language is clearly separated.

## Level 2 — Code-contract validation

For every technical claim edited because of implementation behavior, verify it against the relevant implementation and, where available, tests.

Do not rely on an older plan as evidence that behavior currently exists.

## Level 3 — Cross-document validation

Run T007 to ensure the repaired repository documentation tells one coherent story.

---

# 16. Known Contract Corrections to Preserve

The following corrections are already known and should be treated as required probe targets rather than rediscovered as assumptions.

| Area                   | Stale/ambiguous description               | Expected repair direction                                           |                                |
| ---------------------- | ----------------------------------------- | ------------------------------------------------------------------- | ------------------------------ |
| Ring source            | “NumPy format”                            | raw float64 binary read using NumPy                                 |                                |
| Ring source rate       | “sampled at 200 Hz”                       | acquisition rate unverified; distinguish processing default         |                                |
| Action IDs             | fixed action semantics                    | remove/qualify unless verified                                      |                                |
| Preprocessing raw mode | original six-channel output               | canonical nine-channel output with gravity preserved                |                                |
| Alignment work axis    | duplicate-dependent strict reconstruction | endpoint reconstruction for current inputs                          |                                |
| Alignment strategy     | `strict_reconstruction`                   | `endpoint_reconstruction`                                           |                                |
| Alignment rate         | supplied feature rate drives work axis    | feature rate is metadata; endpoint-derived work coordinate          |                                |
| Alignment offset       | best offset directly exported             | distinguish work-axis estimate and canonical projection             |                                |
| Spike segmentation     | segmenter only supports 6/9 channels      | current `spike-imu` support with 21 channels                        |                                |
| SpikeIMU               | ambiguous feature width                   | 15 event + 6 IMU = 21                                               |                                |
| Bash defaults          | `MADGWICK_PROVISIONAL=1`                  | current default expected `0`                                        |                                |
| Resume mode            | `OVERWRITE` primary                       | `PIPELINE_MODE=continue                                             | overwrite`, legacy`OVERWRITE` |
| Project report         | current/final architecture                | historical inspection/visualization snapshot                        |                                |
| Vendor report          | old project-state claims appear current   | preserve vendor findings, mark surrounding project state historical |                                |

Every row must still be checked against current implementation before its corresponding TaskSpec is frozen.

---

# 17. Risk Management

## Risk 1 — Turning observations into guarantees

Example:

```text
sample data appears close to 201 Hz
```

must not become:

```text
the device samples at exactly 201 Hz
```

Mitigation:

* explicitly label observations;
* avoid false precision.

## Risk 2 — Repairing docs to match a bug

Code is implementation reality, but it is not automatically the intended durable contract.

Mitigation:

* compare relevant tests and durable notes;
* trigger T008 when implementation intent is genuinely disputed.

## Risk 3 — Historical reports becoming misleading

Simply leaving old reports untouched can make repository search results misleading.

Mitigation:

* add clear historical banners;
* qualify temporal statements;
* link to current contracts.

## Risk 4 — Excessive README detail

Moving every technical contract into README would recreate drift.

Mitigation:

* README remains an orientation layer;
* technical details remain in notes.

## Risk 5 — “Raw” terminology ambiguity

The repository contains several legitimate meanings of “raw”.

Mitigation:

* qualify the noun;
* use schema dimensions next to the term where useful.

## Risk 6 — Vendor scope violation

Re-auditing vendor code without authorization would violate repository workflow expectations.

Mitigation:

* use previously verified findings;
* require an explicit TaskSpec if renewed vendor comparison becomes necessary.

---

# 18. WORKBOARD Integration

At plan activation, `docs/plans/WORKBOARD.md` should stop being only a template/example and record this actual plan.

At minimum, the active section should contain:

```text
Plan:
docs/plans/Documentation_Contract_Repair_Plan.md

Task file:
docs/plans/Documentation_Contract_Repair_TASKS.md

Goal:
Repair README and docs/notes so current technical contracts agree
with verified implementation behavior while preserving clearly
marked historical reports.
```

The board should then track:

* current task ID;
* task status;
* dependencies;
* blockers;
* compact probe digest;
* compact worker digest;
* compact verifier digest;
* next action;
* useful Git baseline where relevant.

The workboard should not contain:

* raw source dumps;
* entire patches;
* complete tool logs;
* full test logs;
* full TaskSpec copies.

Those belong in their appropriate execution context.

---

# 19. Suggested Initial WORKBOARD State

After plan initialization and before T001 probing:

```text
Active plan:
Documentation_Contract_Repair_Plan

Current task:
T001

Status:
PROBING

Dependencies:
none

Goal:
Establish the verified cross-document contract baseline.

Known focus:
- raw binary vs NumPy-file wording
- acquisition rate vs processing rate
- 9/15/21 channel schemas
- endpoint-reconstructed alignment work axis
- work-axis vs canonical offset
- spike-imu segmentation support
- pipeline resume/default settings
- historical report labeling

Next action:
Run implementation contract probe and freeze T001 findings before
freezing T002-T006.
```

---

# 20. Completion Criteria

The plan is complete only when all of the following are true:

1. T001 has established a verified contract baseline.
2. README accurately describes the current repository at orientation level.
3. raw Ring data is not misdescribed as a NumPy `.npy` format.
4. `200 Hz` is not presented as an unverified universal acquisition-rate fact.
5. undocumented action semantics are removed or explicitly qualified.
6. preprocessed raw mode is consistently documented as nine-channel output.
7. spike events are consistently documented as 15 channels.
8. SpikeIMU is consistently documented as 21 channels.
9. current segmentation documentation acknowledges `spike-imu`.
10. alignment documentation uses the current endpoint-reconstruction contract.
11. feature sampling-rate metadata is distinguished from the alignment work-axis rate.
12. canonical timestamps are distinguished from the alignment work axis.
13. work-axis alignment estimates are distinguished from exported canonical-domain offsets.
14. segmentation input kind is distinguished from boundary mode.
15. fixed-length padding is documented as a separate downstream stage.
16. Action-0 model input is documented as the first 15 event channels.
17. shell-script defaults match current implementation.
18. `PIPELINE_MODE` resume semantics are documented.
19. `PROJECT_REPORT.md` is clearly historical rather than a current final architecture report.
20. historical portions of `VENDOR_WINDOWING_REPORT.md` are clearly scoped.
21. `DATA_FORMAT.md` separates raw facts, observations, repository interpretations, and derived processing.
22. T007 finds no unresolved high-confidence cross-document contract contradiction.
23. Any implementation discrepancy discovered during the work has been replanned rather than hidden through documentation wording.
24. `WORKBOARD.md` records the final task state and no unresolved documentation blocker remains.

When these conditions hold:

```text
T001 DONE
T002 DONE
T003 DONE
T004 DONE
T005 DONE
T006 DONE
T007 DONE
```

and T008 is either:

```text
NOT REQUIRED
```

or has been split into a separate active implementation plan.

The documentation repair plan can then be marked:

```text
DONE
```

---

# 21. Expected End State

After completion, a reader should be able to navigate the repository without encountering contradictory answers to basic questions such as:

> Is Ring input a `.npy` file?

No. It is raw binary data consumed as float64 values.

> How many values are in one raw Ring source row?

Seven: six IMU measurements plus a timestamp.

> Is the device acquisition rate guaranteed to be 200 Hz?

The documentation does not claim that without upstream evidence. Current processing commonly uses a nominal/default 200 Hz configuration.

> How many channels are in the preprocessed IMU representation?

Nine.

> Does raw gravity mode return six channels?

No. It retains gravity while using the canonical nine-channel feature schema.

> How many spike-event channels are there?

Fifteen in the current custom-wavelet configuration.

> How many channels are in SpikeIMU?

Twenty-one.

> Can the segmenter consume SpikeIMU?

Yes, through the current `spike-imu` input path.

> Are alignment timestamps the original canonical timestamps?

Not necessarily. Alignment uses a separate endpoint-reconstructed work axis while retaining canonical timestamps separately.

> Is the alignment matching offset automatically the reusable exported offset?

The documentation distinguishes the work-axis matching result from its projection into the canonical timestamp domain.

> Does segmentation itself create the final fixed-length tensors?

No. Variable-length segmentation and downstream fixed-length padding are separate stages.

> Does the Action-0 baseline train on all 21 SpikeIMU channels?

No. The current baseline model consumes the first 15 event channels, with the padding mask used by the training/evaluation path.

> Is `PROJECT_REPORT.md` the current complete description of the repository?

No. It is retained as a historical report for an earlier inspection/visualization phase and points readers to current documentation.

That is the documentation contract this plan is intended to establish.
