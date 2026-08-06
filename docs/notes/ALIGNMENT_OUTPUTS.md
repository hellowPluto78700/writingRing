# Ring--Board alignment outputs

Successful constant-offset alignment uses exactly one convention:

```text
ring_timestamp_us = board_timestamp_us + offset_us
```

`offset_us` is always in microseconds. A positive value means the mapped Ring
timestamp is later than the corresponding Board timestamp. Raw Ring, Board,
and label timestamps are never modified; aligned values are derived only for
matching and plotting.

The alignment pipeline has three distinct timestamp/offset domains:

```text
canonical timestamps → provenance and downstream segmentation
strict work timestamps → peak detection, candidate ranking, and matching
exported offset → Board-to-canonical Ring mapping
```

`SequenceAlignmentResult.best_offset_us` is the work-axis offset. Before a TXT
offset is written, matched peak indices project the work-axis coordinate
difference into canonical time using a robust median. The public
`AlignmentOffset.offset_us` is this projected canonical offset; it is never the
unprojected work-axis value.

## Sampling-rate and work-axis rules

The alignment API records an effective sampling rate for the work axis without
changing the canonical timestamp artifact:

- `raw-ring` always uses the historical endpoint-derived rate and reconstructs
  a strict work axis, even when the raw timestamps are irregular or contain
  duplicates. A supplied rate does not replace this raw-ring behavior.
- `spike-imu` uses its supplied metadata rate when duplicate timestamps require
  reconstruction. If no rate is supplied, it infers
  `(sample_count - 1) * 1_000_000 / (last_timestamp - first_timestamp)` after
  validating at least two finite, nondecreasing timestamps with positive
  duration.
- A strictly increasing SpikeIMU canonical vector is copied as the work axis;
  a duplicate-containing vector receives an equal-length strict reconstruction.
  If the rate cannot be inferred, the API raises `AlignmentOffsetExportError`
  rather than leaking an assertion failure.

The work axis is internal to peak detection and matching. Provenance hashes,
canonical offset projection, and downstream segmentation continue to use the
original canonical timestamps. `alignment_time_axis.sampling_rate_hz` records
the effective rate used by alignment.

## Outputs

For `user_0`, action `0`, dataset `0`, the alignment CLI writes:

```text
data/user_0/0/0_ring_board_offset.txt
outputs/alignment/reports/user_0/action_0/0_alignment_report.json
outputs/alignmentVerification/user_0/action_0/0_alignment_verification.png
```

The TXT file is UTF-8 `key=value` data. It includes the mapping direction,
unit, `offset_us`, display-only `offset_ms`, and alignment coverage metadata.
Readers must use `offset_us`; an unsuccessful alignment cannot be exported.

The verification image has six vertically stacked ten-second panels. It shows the
Ring transient score, Board press/lift events after applying the exact same
offset, and labels from `{dataset_id}_label.txt` or fallback
`{dataset_id}_timestamp.txt`. The label time domain is recorded in both the
image subtitle and report. `shared` and `ring` label domains are not shifted;
`board` labels receive the same Board-to-Ring offset as Board events.

Existing offset and verification files are rejected by default. Pass the
corresponding explicit overwrite flag only when replacing the artifact is
intended.

## SpikeIMU alignment provenance

Use `--input-kind spike-imu --spike-root PATH` to align the published SpikeIMU
artifact. The canonical timestamp vector is retained exactly, including
duplicate timestamps, for provenance and downstream segmentation. Alignment
uses a separate same-length strict work axis and computes the transient score
only from channels `15:21`. The offset and JSON report record:

```text
alignment_signal_source=spike-imu
feature_schema=signed_wavelet_events_plus_imu_v1
feature_values_sha256=<digest>
feature_metadata_sha256=<digest>
timestamp_sha256=<digest>
transient_channel_indices=15,16,17,18,19,20
spike_event_channels_used=false
```

For duplicate canonical timestamps, the report also records:

```json
{
  "timestamp_source": {
    "ordering": "nondecreasing",
    "duplicate_step_count": 123
  },
  "alignment_time_axis": {
    "strategy": "strict_reconstruction",
    "work_axis_offset_us": 123.0,
    "canonical_timestamps_modified": false
  },
  "canonical_offset_projection": {
    "projection_delta_us": -5000.0,
    "projection_delta_mad_us": 0.0,
    "projection_delta_min_us": -5000.0,
    "projection_delta_max_us": -5000.0,
    "projection_delta_range_us": 0.0,
    "exported_offset_us": -4877.0,
    "contributing_match_count": 18
  }
}
```

The raw-ring compatibility path uses its endpoint-derived sampling rate for
the strict work axis for both duplicate and irregular timestamp streams.
SpikeIMU uses the sampling rate declared by its metadata when reconstruction is
needed, and copies a strict canonical axis otherwise. A projection range above
half a sampling interval is reported as a warning; a projection that exceeds
the configured constant-offset guard is rejected before an offset is written.
Neither path changes the canonical timestamp artifact or its SHA-256 hash.

Consumers must match these values before SpikeIMU Board-assisted segmentation
or label-verification overlay. A label overlay is diagnostic only and does
not change the label segmentation artifacts.

## CLI

```bash
python scripts/align_ring_board.py \
  --data-root data_sample/data \
  --user user_0 \
  --action 0 \
  --dataset-id 0 \
  --verification-output-root outputs/alignmentVerification \
  --label-time-domain shared
```

By default, the offset TXT is written beside the selected `*_ring_0.bin` in
the data directory. Supply `--offset-output-root PATH` only when a separate,
structured export root is wanted. Useful options are `--label-path`,
`--verification-start-seconds`, `--overwrite-offset`, and
`--overwrite-verification`. On an unsuccessful
alignment, the report is retained but no offset TXT or verification PNG is
created.
