# Variable-length and padded SpikeIMU acceleration reconstruction

Two scripts derive acceleration from completed SpikeIMU segmentation outputs:

- `scripts/reconstruct_segmented_spike_accel.py` reconstructs every selected
  variable-length package into a row-aligned `(N, 3)` array.
- `scripts/reconstruct_padded_spike_accel.py` reconstructs the original
  variable-length segments and then applies the existing padded package's
  right-padding layout, producing `(S, T_pad, 3)`.

Neither script loads raw Ring or Board data, alters segmentation boundaries, or
modifies, overwrites, or appends channels to `*_spikeIMU.npy`.

## Variable-length reconstruction

`reconstruct_segmented_spike_accel.py` accepts either a dataset root containing
`segmentation/` or the `segmentation/` directory itself. It discovers each
requested user's existing `action_*` packages beneath that root.

### Run it

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

### Required input package

Each package must contain the following files, where `<prefix>` is
`<user>_action_<action>`:

```text
<prefix>_spikeIMU.npy              # finite, nonempty (N, 21) or (N, 36)
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
`feature_schema` and `channel_count` must identify one of the supported
contracts:

```text
signed_wavelet_events_plus_imu_v1
  15 signed event channels + 6 IMU channels = 21 total channels

polarity_split_wavelet_events_plus_imu_v1
  30 pairwise positive/negative event channels + 6 IMU channels = 36 total channels
```

The 36-channel event order is
`e0_pos, e0_neg_abs, e1_pos, e1_neg_abs, ..., e14_pos, e14_neg_abs`.
Missing `feature_schema` remains a legacy alias for the 21-channel signed
layout only; a 36-channel package must declare its schema explicitly.

The reconstruction uses the published `spike_encoder.wavelet_widths_samples`
and frequencies, not hard-coded defaults or a newly inferred width. Under the
current encoder rule, 200 Hz / 16 Hz is truncated to width 12, so the canonical
five-band example `[1, 2, 4, 8, 16]` uses `[200, 100, 50, 25, 12]`. Artifacts
without a verified encoder spec/hash must be regenerated. The encoder can be
selected with `ENCODER_FREQUENCIES_HZ="1 2 4 8 16"` or
`--encoder-frequencies-hz 1 2 4 8 16`.

### Reconstruction semantics

The schema-specific event prefix is decoded first into canonical signed
`(T, 15)` events. The 21-channel signed layout is passed through unchanged;
the 36-channel layout is decoded as `positive - negative_abs` for each pair.
The downstream reconstruction therefore always interprets three axes with five
frequency bands per axis, in axis-major, frequency-minor order. The script
uses the same Custom Wavelet reconstruction method as the comparison notebook:

```text
frequencies_hz = [0.5, 1, 2, 4, 8]

for each segment:
    decode source SpikeIMU events to signed spikeIMU[start:stop, 0:15]
    reconstruct the canonical signed events independently
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

### Outputs and provenance

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

## Padded reconstruction

`reconstruct_padded_spike_accel.py` accepts the dataset combination root, not
an individual user or `segmentation/` directory. It automatically scans every
complete `user_*/action_*` variable-length package under:

```text
<dataset-root>/segmentation/
```

and requires its corresponding package under:

```text
<dataset-root>/segmentation_padded/
```

Run it after the repository padding pipeline has completed:

```bash
python scripts/reconstruct_padded_spike_accel.py \
  outputs/action0_rectified/low-pass/aligned-board-events
```

`--sampling-rate-hz` and `--output-dtype float32|float64` have the same
meaning as for variable-length reconstruction. The padded script has no user
filter: every discoverable complete source package is attempted. It continues
after a package-level error, then returns a nonzero status if any package
failed.

The convenience wrapper
`scripts/Bash_Script/Encoder_Evaluation_related/reconstruct_spike_sequence.bash`
runs the same command for
`outputs/action0_rectified/low-pass/aligned-board-events` with `--overwrite`.
It therefore replaces every existing padded reconstruction in that root.

### Matching and boundary contract

For each user/action, the script requires the variable-length SpikeIMU,
labels, offsets, lengths, and segmentation summary, plus the matching padded
SpikeIMU, labels, valid lengths, valid mask, padding manifest, and padding
summary. It validates that the padded package is canonical right padding with
`overflow_policy=skip`.

The padded reconstruction reads and validates the source segmentation schema
and padded-package schema independently. Thus signed-to-signed (21→21),
signed-to-polarity-split (21→36), and polarity-split-to-polarity-split
(36→36) packages are valid as long as each package declares its own matching
schema and channel count. Only the source schema is decoded for reconstruction;
the padded schema controls shape and provenance validation.

The reconstruction itself always reads the original unpadded
the schema-decoded canonical signed events selected by the authoritative
source offsets. The padding manifest maps the source `segment_index` to each retained
`output_segment_index`; `valid_lengths`, labels, and the boolean `valid_mask`
must agree with that mapping. The output has the same segment and time axes as
`*_paddedSpikeIMU.npy`:

```text
reconstructed[output_segment_index, :original_length, :] = segment reconstruction
reconstructed[output_segment_index, original_length:, :] = 0.0
```

Consequently, convolution neither crosses a segment boundary nor sees padded
samples. Source segments omitted by the padding manifest, including overlong
segments under its skip policy, remain absent from the padded reconstruction.

### Outputs and provenance

The script writes these files next to the matching padded SpikeIMU package:

```text
<prefix>_padded_reconstructed_accel_m_s2.npy
<prefix>_padded_reconstructed_accel_metadata.json
```

The array has shape `(S, T_pad, 3)`, where its three channels are reconstructed
x/y/z acceleration in m/s². Its padded region is exactly zero and its segment
axis, labels, valid lengths, and valid mask align with
`*_paddedSpikeIMU.npy`; it is a separate artifact, not extra SpikeIMU
channels.

Metadata records the Custom Wavelet settings, source and padded package
SHA-256 digests, target length, right-padding semantics, zero padding value,
and the fact that reconstruction occurs before padding. If either padded
reconstruction output already exists, the script fails unless `--overwrite` is
given; unlike the variable-length tool, it does not reuse existing output.
