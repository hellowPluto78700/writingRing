# SpikeIMU segmentation pipeline

This note records the PR1/PR3 input and label-segmentation contract plus the
PR2/PR4 alignment and Board-assisted consumer contract. It is a consumer of
the complete gravity-to-spike artifacts; it does not perform gravity removal.

| feature input | label boundaries | Board-assisted boundaries |
|---|---|---|
| `raw-ring` | supported, legacy default | supported, legacy 9-channel output |
| `spike-imu` | 21-channel canonical artifact | 21-channel artifact plus four Board targets |

`boundary-mode` chooses the boundary source; `input-kind` chooses the matrix
and canonical timestamp artifact. Segmentation never reconstructs or resamples
that canonical axis. Alignment uses a separate endpoint-reconstructed strictly
increasing work axis for peak detection and matching while canonical
timestamps remain immutable.

## Canonical input

Preprocessing publishes one immutable row-aligned set per recording:

```text
<data_id>_preprocessedIMU.npy
<data_id>_timestamps_us.npy
<data_id>_preprocessing.json
```

Batch Custom Wavelet encoding publishes a recording namespace containing
`spikeIMU.npy` and `metadata.json`. The metadata references the same canonical
timestamp NPY and records its SHA-256 digest. The SpikeIMU loader verifies:

- recording identity (`user`, `action`, `data_id`);
- `signed_wavelet_events_plus_imu_v1`, shape `(N, 21)`, finite values, and
  sample count;
- the 21-element units declaration, canonical trailing IMU units, and the six
  trailing channel names;
- the SpikeIMU and timestamp SHA-256 digests;
- finite, nondecreasing timestamps in microseconds; duplicate timestamps are
  retained and accepted.

`RecordingFeatureInput` is the common immutable handoff for raw Ring and
SpikeIMU consumers. Raw input remains compatible with the existing in-memory
preprocessing path. SpikeIMU input is loaded as-is.

## Sampling-rate contract

The optional CLI `--sampling-rate` is an expected-rate check applied to each
SpikeIMU artifact while it is loaded. It does not resample or rewrite feature
values or timestamps. For a user/action aggregation, the segmentation code
loads all selected SpikeIMU inputs first and then requires one common metadata
rate, comparing every recording to the first with absolute tolerance `1e-12`.
Mismatch errors identify both dataset IDs and their declared rates.

This preflight runs before label slicing, Board loading, aggregation, or
staging publication. Consequently, a mixed-rate action is rejected without
publishing a partial label or aligned-Board output, and a successful summary's
top-level `sampling_rate_hz` is the validated common rate. Raw-ring
segmentation keeps its existing per-recording processing behavior.

## Label mode

The segmentation CLI separates the feature source from the boundary source:

```text
--input-kind raw-ring|spike-imu
--boundary-mode label|aligned-board-events
```

For PR3, `spike-imu + label` loads `<user>/<action>/<data_id>/spikeIMU.npy`
under `--spike-root`, reads the original timestamp label file, and slices:

```text
start = searchsorted(timestamps_us, label_i_timestamp, side="left")
stop  = searchsorted(timestamps_us, label_(i+1)_timestamp, side="left")
segment = spikeIMU[start:stop, :]
```

The output retains 21 channels in a contiguous `*_spikeIMU.npy`; offsets and
lengths remain the authoritative variable-length boundaries. The manifest and
summary carry input kind, schema, units, timestamp provenance, and feature
hashes.

The five Custom Wavelet frequencies are configurable with
`ENCODER_FREQUENCIES_HZ="1 2 4 8 16"` (or CLI
`--encoder-frequencies-hz 1 2 4 8 16`). Event channels remain index-based;
their frequency/width mapping comes from `spike_encoder` metadata. Segmentation
and padding propagate the encoder spec and hash, and reject mixed encoder
identities before publishing or combining data. Artifacts missing that identity
must be regenerated.

Optional downstream reconstruction can derive a separate `(N, 3)` acceleration
array from the first 15 event channels, or a `(S, T_pad, 3)` array after the
completed package has been right-padded. Both forms reconstruct each source
segment from offsets rather than convolving across the aggregate array. See
[SEGMENTED_SPIKE_ACCEL_RECONSTRUCTION.md](SEGMENTED_SPIKE_ACCEL_RECONSTRUCTION.md).

Gravity-removal flags are rejected in SpikeIMU mode. An explicit sampling rate
is a per-recording metadata consistency check, followed by the action-level
common-rate check above. `spike-imu + aligned-board-events` requires the
matching provenance-bearing alignment offset described below.

## Verification overlay

Label verification uses the exact same `RecordingFeatureInput` and segmentation
result as the export path. For SpikeIMU, transient scoring is computed from
`spikeIMU[:, 15:21]`; event channels `0:15` cannot affect the score. A Board
overlay is opt-in and changes only the verification context and rendered
figure. It cannot change label start/stop indices, segment offsets, labels,
lengths, or exported values.

When the overlay is enabled for SpikeIMU, the offset is accepted only after
its recording identity, signal source, feature schema, SpikeIMU values and
metadata hashes, transient-channel contract, and canonical timestamp hash
match the current `RecordingFeatureInput`. Legacy raw-Ring offsets and stale
SpikeIMU offsets are rejected before Board alignment or rendering.

## Board-assisted SpikeIMU mode

`scripts/align_ring_board.py --input-kind spike-imu` loads one published
SpikeIMU artifact, uses `values[:, 15:21]` for transient detection, and writes
the canonical timestamp hash into the alignment report and offset. The
segmentation CLI then validates that offset before loading Board events and
publishes:

```text
*_spikeIMU.npy                 (total_samples, 21)
*_board_event_targets.npy     (total_samples, 4)
*_labels.npy
*_segment_offsets.npy
*_segment_lengths.npy
*_segments.csv
*_board_events.csv
*_segmentation_summary.json
```

`board_event_targets` has the fixed order
`valid_press, valid_lift, transient_press, transient_lift`. The summary marks
`alignment_input_hash_match_verified=true` only after values, metadata, and
timestamp hashes match for every recording. Raw-ring offsets, offsets from an
older SpikeIMU artifact, and offsets with a different timestamp hash fail
before Board-assisted outputs are published. The sampling-rate preflight also
completes before staging; mixed-rate SpikeIMU recordings cannot produce an
aggregate whose single top-level rate describes only the first recording.

For an exportable first label whose leading `board_0` chunk is serialized
empty, Board-assisted segmentation may recover its start from a transient
peak. This is not a search over wavelet event channels: it scores only the
trailing SpikeIMU transient input `spikeIMU[:, 15:21]`. The candidate peak must
lie strictly between the label timestamp and the first aligned Board frame,
and must be accepted by the same alignment-grade
`detect_transient_peak_regions` detector used for alignment. Among eligible
peaks, selection is greatest prominence, then greatest height, then earliest
timestamp. No qualifying peak means no recovery fallback; normal Board-event
availability rules continue to decide the candidate.

## Compatibility and migration

Omitting both selectors keeps the existing raw-ring label workflow. Existing
raw-ring Board-assist exports retain the nine-channel `*_rawIMU.npy` naming
and legacy gravity summaries. SpikeIMU outputs use `*_spikeIMU.npy`; they are
not interchangeable with raw-ring outputs, and a raw-ring alignment offset
must never be reused for SpikeIMU overlay or Board-assisted segmentation.
