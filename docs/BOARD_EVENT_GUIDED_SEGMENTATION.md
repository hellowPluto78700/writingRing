# Aligned Board-event-guided IMU segmentation

`scripts/segment_ring_imu.py` supports two strictly separated boundary modes:

```text
label                 # default; timestamp-label boundaries only
aligned-board-events  # explicit; labels, Board events, and a saved offset
```

The default mode remains documented in [IMU_SEGMENTATION.md](IMU_SEGMENTATION.md).
It does not read Board data or alignment offsets.

## Run aligned mode

Create a successful Ring--Board offset for every recording first. The offset
root must follow the alignment export layout:

```text
outputs/alignment/offsets/
└── user_0/
    └── action_0/
        ├── 0_ring_board_offset.txt
        ├── 1_ring_board_offset.txt
        └── ...
```

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

The command never estimates an offset. A missing, unsuccessful, malformed, or
identity-mismatched offset fails the run; it does not fall back to `label`.
Use `--overwrite` to replace an existing complete aligned output set.

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
--verification-panel-seconds 10
--verification-dpi 200
```

`skip` remains available as an explicit compatibility policy and rejects all
crossing touches. `clip-at-label` is not implemented; this prevents a silent
mixture of boundary definitions.

## Outputs

For low-pass mode, results are published transactionally into
`outputs/boardAssistSegmentedIMU_LowPassFilterin/user_0/action_0/` by default.
For Madgwick mode, the root is
`outputs/boardAssistSegmentedIMU_Madgwick/user_0/action_0/`. An explicit
`--output-root` replaces the method-specific root directly; no additional
`aligned_board_events` directory is added.

To export the original six Ring IMU channels without gravity removal, use:

```bash
python scripts/segment_ring_imu.py \
  --data-root data --user user_0 --action 0 \
  --boundary-mode aligned-board-events \
  --alignment-offset-root outputs/alignment/offsets \
  --gravity-removal-method raw
```

Without `--output-root`, this publishes under
`outputs/boardAssistSegmentedIMU_RawIMU/user_0/action_0/`. Raw keeps gravity
in the measured acceleration; all modes still publish the same nine channels.

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

Each verification image covers the full Ring recording in vertically stacked
10-second panels. It shows the caller-final segment boundaries only, plus the
aligned Board event lines, labels, skipped labels, and transient score. If its
PNG cannot be written and validated, no partial aggregate output is published.
