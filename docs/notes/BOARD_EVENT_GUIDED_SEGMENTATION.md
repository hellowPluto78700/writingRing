# Aligned Board-event-guided IMU segmentation

`scripts/segment_ring_imu.py` supports two strictly separated boundary modes:

```text
label                 # default; timestamp-label boundaries only
aligned-board-events  # explicit; labels, Board events, and a saved offset
```

The default mode remains documented in [IMU_SEGMENTATION.md](IMU_SEGMENTATION.md).
It does not read Board data or alignment offsets.

The same Board boundary implementation supports `--input-kind spike-imu`.
In that mode `--spike-root` supplies the published `(N, 21)` matrix and the
canonical timestamp source; preprocessing/gravity flags are invalid.

## Run aligned mode

Create a validated Ring--Board outcome for every recording first. The aligned
outcome family uses sibling roots:

```text
outputs/alignment/
├── offsets/
├── reports/
└── verification/
```

Segmentation receives the `offsets/` root and derives the sibling report and
verification roots. It validates current feature and numeric-order Board
provenance before loading labels or preparing Board events. `SUCCESS` records
are segmented normally. A provenance-valid `SKIPPED` record is omitted without
requiring a timestamp label, and does not enter sampling-rate or aggregate
calculations. `FAILED`, missing, stale, malformed, or conflicting outcomes
hard fail.

For SpikeIMU inputs, loading before outcome validation still verifies feature
identity/content, metadata/timestamps, and metadata-rate validity, but does
not reject the caller-requested rate yet. Once the outcome is validated, only
`SUCCESS` recordings are checked against that requested rate and included in
the common-rate contract. Thus a provenance-valid `SKIPPED` recording with a
different rate cannot reject an otherwise valid SUCCESS partition.

Then run:

```bash
conda run --no-capture-output -n writingring-viz \
    python scripts/segment_ring_imu.py \
    --data-root data \
    --user user_0 \
    --action 0 \
    --boundary-mode aligned-board-events \
    --alignment-offset-root outputs/alignment/offsets
```

The command never estimates an offset. An invalid outcome never falls back to
`label`. If every selected recording is provenance-valid `SKIPPED`, the command
fails explicitly before aggregate output construction.
Use `--overwrite` to replace an existing complete aligned output set.

For `--input-kind spike-imu`, all selected artifacts are loaded before staging
is created. Each processed (`SUCCESS`) artifact is checked against an explicitly supplied
`--sampling-rate` when present, then the metadata rates are compared across the
user/action with absolute tolerance `1e-12`. Mixed rates fail with both
conflicting dataset IDs and rates, and no aggregate or staging directory is
published. A successful aggregate records the validated common rate in its
top-level `sampling_rate_hz`; recording-level skips do not participate in that
comparison. Raw-ring mode retains its existing processing and
outcome-validation order.

## Boundary semantics

`timestamp.txt` continues to provide the class and identifies the label that
owns a touch from its press timestamp. Within an eligible interval, the final window starts 0.2 s
before the first valid Board press and ends 0.2 s after the last valid Board
lift. Change those values explicitly:

```bash
--pre-press-context-seconds 0.2 \
--post-lift-context-seconds 0.2
```

Transient touches do not define boundaries but are preserved in target
channels. Incomplete touches remain in the Board-event audit CSV but do not
write a target. The default crossing policy is `accept_until_next_press`: a
touch with `press < next_label <= lift < next_first_valid_press` remains owned
by the press label, writes its normal press/lift targets, and may define that
segment's final lift boundary. A crossing lift at or after the next label's
first valid press, or one with no reliable next valid press, is rejected and
remains audit-only. The CSV records `crosses_next_label_timestamp`,
`crossing_resolution`, and `assigned_label_index` for each Board event.

When neighboring 0.2 s contexts overlap, their split is the midpoint between
the current segment's last lift and the following segment's first press. This
retains both event cores instead of forcibly clipping at the label timestamp.
Labels still use the existing `wrong`, minimum 0.1-second, and maximum
5-second validity rules.

The audit CSV distinguishes target assignment from boundary construction:

```text
boundary_role = start  # first eligible valid press
boundary_role = end    # last eligible valid lift
boundary_role = none   # every other event
```

Only `start` and `end` have `used_for_segment_boundary = true`. Intermediate
valid touches can still have event targets.

The optional aligned-only controls are:

```text
--missing-event-policy skip
--crossing-touch-policy accept_until_next_press
--recording-error-policy error|skip
--verification-panel-seconds 10
--verification-dpi 200
```

`skip` remains available as an explicit compatibility policy and rejects all
crossing touches. `clip-at-label` is not implemented; this prevents a silent
mixture of boundary definitions.

`--recording-error-policy` is independent of the alignment outcome. Its
default, `error`, preserves standalone strict behavior. With explicit `skip`,
a provenance-valid alignment `SUCCESS` recording that fails a known
recording-local label/event-preparation or Board-boundary segmentation check
is recorded and omitted while later recordings continue. This does not turn
the alignment outcome into `SKIPPED`.

Stale or malformed provenance/outcomes or success artifacts, source-feature
loading and rate validation, verification rendering/PNG output, aggregate
validation/publication, filesystem failures, and unexpected exceptions remain
hard errors even with this policy.

## Outputs

For low-pass mode, results are published transactionally into
`outputs/boardAssistSegmentedIMU_LowPassFilterin/user_0/action_0/` by default.
For Madgwick mode, the root is
`outputs/boardAssistSegmentedIMU_Madgwick/user_0/action_0/`. An explicit
`--output-root` replaces the method-specific root directly; no additional
`aligned_board_events` directory is added.

To export raw-mode preprocessed IMU features without gravity removal, use:

```bash
python scripts/segment_ring_imu.py \
  --data-root data --user user_0 --action 0 \
  --boundary-mode aligned-board-events \
  --alignment-offset-root outputs/alignment/offsets \
  --gravity-removal-method raw
```

Without `--output-root`, this publishes under
`outputs/boardAssistSegmentedIMU_RawIMU/user_0/action_0/`. Raw keeps gravity
in the measured acceleration; all modes publish the canonical nine channels.

```text
outputs/boardAssistSegmentedIMU_LowPassFilterin/user_0/action_0/
├── user_0_action_0_rawIMU.npy
├── user_0_action_0_labels.npy
├── user_0_action_0_segment_offsets.npy
├── user_0_action_0_segment_lengths.npy
├── user_0_action_0_board_event_targets.npy
├── user_0_action_0_segments.csv
├── user_0_action_0_board_events.csv
├── user_0_action_0_segmentation_summary.json
└── {dataset_id}_ring_0_segmentation_verification.png
```

`rawIMU.npy` remains contiguous `(total_samples, 9)` preprocessing-feature storage. The four-column
boolean `board_event_targets.npy` has the same row count and channel order:

```text
valid_press, valid_lift, transient_press, transient_lift
```

For SpikeIMU mode the feature file is named `*_spikeIMU.npy`, has shape
`(total_samples, 21)`, and `board_event_targets.npy` has exactly the same row
count. The output summary records `feature_schema`, channel slices
`[0,15]`, `[15,18]`, `[18,21]`, and per-recording feature/metadata/timestamp
hashes, plus the single validated common `sampling_rate_hz`. Before Board
event preparation or slicing, the validated outcome must prove the same
recording identity and all of those hashes; a raw-ring or stale SpikeIMU
outcome is rejected.

Each verification image covers the full Ring recording in vertically stacked
10-second panels. It shows the caller-final segment boundaries only, plus the
aligned Board event lines, labels, skipped labels, and transient score. If its
PNG cannot be written and validated, no partial aggregate output is published.

The segmentation summary retains `skipped_segment_count` for segment-level
policy skips and additionally records `source_recording_count`,
`processed_recording_count`, `skipped_recording_count`, and `recording_skips`.
Each recording-skip entry includes its identity, validated reason, and
diagnostics.

For aligned-Board summaries only, `alignment_outcome_dependency` is the
machine-readable downstream reuse contract.  Its ordered
`source_recording_ids` list comes from the authoritative discovered `ring_0`
recordings.  `outcomes_by_status.SUCCESS` contains each processed recording's
identity and SHA-256 of its validated alignment report; `SKIPPED` contains the
same identity and report digest plus the validated skip reason.  The two lists
are disjoint and their union is exactly the source list.  This dependency is
separate from both the audit-oriented `recording_skips` and label-policy
`skipped_segment_count`; it is absent from label-mode summaries.  Consumers
must treat a missing or unequal dependency as stale, rather than infer outcome
state from offset files or artifact counts.

When recording-error skip mode omits an alignment-`SUCCESS` recording, the
normal user/action summary additionally contains
`alignment_skipped_recording_count`, `segmentation_error_recording_count`, and
`segmentation_errors`. Each error has recording identity, `stage`,
`error_type`, and message, and the count invariant is:

```text
source_recording_count
  = processed_recording_count
  + alignment_skipped_recording_count
  + segmentation_error_recording_count
```

`alignment_outcome_dependency.outcomes_by_status.SUCCESS` still includes an
omitted recording: it describes alignment only, not segmentation usability.
For a user with at least one processed recording, a matching sidecar is also
published under:

```text
<output-root>/recording_errors/<user>/action_<action>/
  <user>_action_<action>_segmentation_recording_errors.json
```

If every alignment-`SUCCESS` recording reaches that permitted terminal error,
there is no empty IMU/target/manifest package under
`<output-root>/<user>/action_<action>/`; only that report-only terminal state
is published. A user containing only alignment `SKIPPED` outcomes retains the
existing strict error before aggregation.
