# SpikeIMU label segmentation

This note records the PR1/PR3 input and label-segmentation contract. It is a
consumer of the complete gravity-to-spike artifacts; it does not perform
gravity removal or implement SpikeIMU Board-assisted segmentation.

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

Gravity-removal flags are rejected in SpikeIMU mode. An explicit sampling rate
is a metadata consistency check only. `spike-imu + aligned-board-events` is
rejected until the separate Board-assist implementation supplies compatible
alignment provenance.

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
