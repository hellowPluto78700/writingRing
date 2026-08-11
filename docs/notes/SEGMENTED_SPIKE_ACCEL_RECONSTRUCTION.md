# Segment-wise SpikeIMU acceleration reconstruction

`scripts/reconstruct_segmented_spike_accel.py` derives a separate acceleration
artifact from a completed, variable-length SpikeIMU segmentation package. It
does not load raw Ring or Board data, alter the segmentation boundaries, or
modify, overwrite, or append channels to `*_spikeIMU.npy`.

The script accepts either a dataset root containing `segmentation/` or the
`segmentation/` directory itself. It discovers each requested user's existing
`action_*` packages beneath that root.

## Run it

For selected users:

```bash
python scripts/reconstruct_segmented_spike_accel.py \
  outputs/action0_rectified/low-pass/aligned-board-events \
  --users user_0 user_3
```

Numeric user identifiers are normalized to the `user_<id>` form:

```bash
python scripts/reconstruct_segmented_spike_accel.py \
  outputs/action0_rectified/low-pass/aligned-board-events \
  --users 0 3
```

To process every existing `user_*` directory under the segmentation root:

```bash
python scripts/reconstruct_segmented_spike_accel.py \
  outputs/action0_rectified/low-pass/aligned-board-events \
  --users all
```

`all` cannot be combined with explicit users. Missing explicit users are
reported and skipped when at least one requested user exists. For every
selected user, all complete matching `action_*` packages are considered.

## Required input package

Each package must contain the following files, where `<prefix>` is
`<user>_action_<action>`:

```text
<prefix>_spikeIMU.npy              # finite, nonempty (N, 21)
<prefix>_labels.npy                # (segment_count,)
<prefix>_segment_offsets.npy       # (segment_count + 1,)
<prefix>_segment_lengths.npy       # (segment_count,)
<prefix>_segmentation_summary.json
```

Offsets are authoritative: they must start at zero, end at `N`, be strictly
increasing, and have differences exactly equal to `segment_lengths`. The
summary must provide a positive `sampling_rate_hz` unless an explicit
`--sampling-rate-hz` override is supplied. When the summary declares the
SpikeIMU identity fields, they must be `input_kind=spike-imu` and
`feature_schema=signed_wavelet_events_plus_imu_v1`; other declared values are
rejected.

The reconstruction frequencies require an integer number of samples per band:
`sampling_rate_hz / frequency_hz` must be integral for every configured
frequency. `--sampling-rate-hz` changes that reconstruction rate only after
this check; it does not resample the source artifact.

## Reconstruction semantics

Only `spikeIMU[:, 0:15]` is used. Those event channels are interpreted as
three axes with five frequency bands per axis, in axis-major,
frequency-minor order. The script uses the same Custom Wavelet reconstruction
method as the comparison notebook:

```text
frequencies_hz = [0.5, 1, 2, 4, 8]

for each segment:
    reconstruct spikeIMU[start:stop, 0:15] independently
    convolve each event band with the reversed accelerationWavelet kernel
    sum the five bands for each axis, dividing each by 2.5
    convert the result from g to m/s² by multiplying by 9.80665
```

The segment loop is deliberate:

```text
segment 0 events  → segment 0 reconstructed acceleration
segment 1 events  → segment 1 reconstructed acceleration
...               → concatenate at the original offsets
```

Convolution never crosses a segment boundary; outside-segment context is zero.
The output therefore has exactly one row for every source SpikeIMU row, while
a spike at the end of one segment cannot produce reconstruction energy in the
next segment. IMU channels `15:21` are not used for reconstruction.

Published event amplitudes are used as stored. If a source artifact was
post-encode `AbsRectify` transformed, the lost event polarity cannot be
recovered by this script. The reconstruction implementation was compared
value-for-value with the notebook method on the supplied test data
(`max_abs_diff = 0.0`).

## Outputs and provenance

The script publishes three derived files beside the input package:

```text
<prefix>_reconstructed_accel_m_s2.npy
<prefix>_reconstructed_accel_metadata.json
<prefix>_reconstructed_accel_segments.csv
```

The NPY array has shape `(N, 3)`, is row-aligned with the source
`*_spikeIMU.npy`, and uses channel order:

```text
0  reconstructed acceleration x (m/s²)
1  reconstructed acceleration y (m/s²)
2  reconstructed acceleration z (m/s²)
```

The default stored dtype is `float64`; pass `--output-dtype float32` to store
`float32` instead. The CSV has one row per segment, including its label,
aggregate row range, length, duration, and event activity statistics.

The JSON metadata records the reconstruction method, frequencies, sampling
rate and its source, scale divisor `2.5`, standard gravity `9.80665`, channel
names and units, the explicit segment-wise boundary semantics, and SHA-256
digests for the source SpikeIMU, labels, offsets, lengths, segmentation
summary, reconstructed array, and reconstruction CSV.

Existing complete outputs are reused only after their metadata and output
hashes match the current source package. A changed source segmentation is
reported as stale rather than silently reused. To replace an existing complete
or partial output set, rerun with `--overwrite`:

```bash
python scripts/reconstruct_segmented_spike_accel.py \
  outputs/action0_rectified/low-pass/aligned-board-events \
  --users user_0 user_3 \
  --overwrite
```

Each output file is atomically replaced. If publishing a reconstruction set
fails, the script removes its reconstruction files instead of retaining a
partial set. A package-level error is reported while the script continues with
other discovered packages; the final exit status is nonzero if any package
failed.
