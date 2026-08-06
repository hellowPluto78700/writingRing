# Timestamp-label and Board-assisted Ring IMU segmentation

The CLI has two independent selectors:

```text
--input-kind raw-ring|spike-imu
--boundary-mode label|aligned-board-events
```

`boundary-mode=label` uses only timestamp-label files. `boundary-mode=aligned-board-events`
requires a previously exported successful Ring--Board offset and derives
boundaries from aligned Board press/lift events. The `--input-kind` selector
chooses the matrix being sliced; it never changes the boundary policy.

For the default `raw-ring` input, `scripts/segment_ring_imu.py` first removes
the gravity contribution from each primary Ring recording, then exports
variable-length IMU segments for one `user` and one `action`. It uses only
`{dataset_id}_ring_0.bin`; `ring_1` is never read.

```bash
conda run --no-capture-output -n writingring-viz \
    python scripts/segment_ring_imu.py \
    --data-root data \
    --user user_0 \
    --action 0
```

The default gravity method is the existing causal second-order Butterworth
low-pass implementation (`low-pass`, 0.2 Hz cutoff). Its default output root
is `outputs/segmentedIMU_LowPassFiltering`. Select the existing Madgwick
sensor-fusion implementation with `--gravity-removal-method madgwick`; its
default root is `outputs/segmentedIMU_Madgwick`.

Use `--gravity-removal-method raw` to bypass gravity removal and export the
original six Ring IMU channels. Its default root is
`outputs/segmentedIMU_RawIMU`:

```bash
python scripts/segment_ring_imu.py \
    --data-root data --user user_0 --action 0 \
    --gravity-removal-method raw
```

```bash
python scripts/segment_ring_imu.py \
    --data-root data --user user_0 --action 0 \
    --gravity-removal-method madgwick \
    --madgwick-beta 0.1
```

`--low-pass-cutoff-hz` adjusts the low-pass cutoff, and `--sampling-rate` sets
the processing assumption for either estimator. Low-pass exports remain
available if the existing calibration diagnostics fail because its filter does
not use the calibration estimate. Madgwick is strict by default; `--provisional`
permits its export when stationary-calibration checks fail. Use `--output-root`
only to explicitly override the method-specific destination.

The command does not overwrite existing artifacts by default. Use
`--overwrite` to replace the known artifacts for that user/action. This also
removes the old fixed-length `valid_lengths` and `valid_mask` files from a
previous exporter run.

## Optional label-only verification

Label mode does not create figures by default and does not load Board data or
alignment offsets. Add `--write-label-verification` to write one full-recording
PNG per dataset beside the label-mode output arrays:

```bash
python scripts/segment_ring_imu.py \
    --data-root data --user user_0 --action 0 \
    --write-label-verification
```

This figure shows the six-axis transient score, timestamp labels, skipped label
annotations, and the final label-only segment backgrounds. It does not change
the exported boundaries or arrays.

To display Board event markers for visual comparison only, add both
`--overlay-aligned-board-events` and a structured `--alignment-offset-root`:

```bash
python scripts/segment_ring_imu.py \
    --data-root data --user user_0 --action 0 \
    --write-label-verification \
    --overlay-aligned-board-events \
    --alignment-offset-root outputs/alignment/offsets
```

The overlay requires valid saved offsets and Board data, but it never changes
label-mode segments. `--verification-panel-seconds` and `--verification-dpi`
adjust either label or aligned verification figures.

## SpikeIMU consumer mode

`--input-kind spike-imu` requires `--spike-root` and consumes the published
`spikeIMU.npy` plus its metadata and canonical timestamp sidecar. It performs
no gravity removal, resampling, timestamp rewriting, or event-channel
alignment during segmentation. Label mode slices all 21 channels with the same
`searchsorted(canonical_timestamps_us, ..., side="left")` rule. Aligned Board
mode additionally requires an offset whose feature, metadata, and timestamp
hashes match the loaded SpikeIMU artifact. An explicit `--sampling-rate` is
checked per recording. When a user/action selects multiple SpikeIMU recordings,
their metadata rates must also match within `1e-12`; this is checked before
label or aligned-Board aggregation, so mixed-rate actions do not publish
partial outputs and successful summaries contain one common `sampling_rate_hz`.

Spike label verification scores only channels `15:21`; channels `0:15` are
signed wavelet event channels and cannot affect segmentation. The optional
Board overlay is diagnostic only: it can reject a stale or raw-ring offset,
but it cannot alter values, offsets, lengths, labels, or manifest boundaries.

## Segment semantics

Each nonempty, non-comment line in `{dataset_id}_timestamp.txt` has a numeric
timestamp followed by its label. Labels preserve their original case and the
rest of the line after the timestamp. Timestamps must be finite and strictly
increasing.

For consecutive label timestamps `t_i` and `t_(i+1)`, the segment contains:

```text
[t_i, t_(i+1))
```

Both boundaries use:

```python
np.searchsorted(ring_timestamps_us, timestamp, side="left")
```

Thus a segment begins at the first Ring sample whose timestamp is at least its
label timestamp and ends at the frame immediately before the next label's
first selected Ring sample. The last label extends through the final Ring
sample. Duplicated Ring timestamps at a boundary belong entirely to the later
segment, so no sample is duplicated across segments.

### Invalid label starts

A marker is retained as a boundary but not exported as a segment start when
its label is `wrong` (case-insensitive), or when its adjacent timestamp pair
violates one of these limits. Under the dataset's observed microsecond
interpretation, the limits are 0.1 seconds = 100,000 timestamp units and
5 seconds = 5,000,000 timestamp units.

For every adjacent pair `t_i`, `t_(i+1)`, with
`delta = t_(i+1) - t_i`:

- `delta < 100,000`: both `t_i` and `t_(i+1)` are invalid starts;
- `100,000 <= delta <= 5,000,000`: `t_i` is eligible to generate its normal
  segment;
- `delta > 5,000,000`: `t_i` is invalid, and processing can restart at
  `t_(i+1)` if that marker is otherwise valid.

The last marker is checked against the final Ring timestamp. If that interval
is longer than 5,000,000 units, the last marker is invalid too. Invalid
markers remain boundaries: an earlier valid segment still ends immediately
before the next marker, but no segment starts at an invalid marker. The JSON
summary records skipped-marker counts and reasons.

The upstream `vendor/WritingRing/ring_plot.py` maps markers to a nearest Ring
row for visualization. This exporter intentionally uses left-side
`searchsorted` because a segment start must not select a sample before its
label marker. Raw Ring decoding otherwise follows the upstream semantics:
native-endian `float64`, reshaped to seven columns, with the first six columns
used as IMU features.

## Variable-length export

There is no 600-sample limit, zero-padding, mask, or truncation. Each segment
retains every preprocessed IMU frame in its label interval, in this channel
order:

```text
acceleration_x_g, acceleration_y_g, acceleration_z_g,
acceleration_x, acceleration_y, acceleration_z,
gyro_x, gyro_y, gyro_z
```

The first three channels are g, channels 3--5 are the same processed
acceleration in m/s², and the final triplet is gyroscope in rad/s. They satisfy
`acceleration_m_s2 = acceleration_g * 9.80665`. Raw preserves measured
acceleration including gravity; the summary and manifest expose that choice as
`acceleration_semantics` alongside the selected preprocessing method.

Variable-length arrays cannot form a regular numeric `(N, L, 9)` NPY file.
Instead, `rawIMU.npy` is one contiguous numeric `(total_samples, 9)` array,
and `segment_offsets.npy` identifies each segment:

```python
segment_i = raw_imu[offsets[i] : offsets[i + 1]]
```

`segment_lengths.npy` is equivalent to `np.diff(offsets)` and is included for
quick inspection. All three arrays are loadable with `allow_pickle=False`.

## Outputs and ordering

For `--user user_0 --action 0`, paths are:

```text
outputs/segmentedIMU/user_0/action_0/
├── user_0_action_0_rawIMU.npy           # preprocessing features (total_samples, 9)
├── user_0_action_0_labels.npy           # (N,), original strings
├── user_0_action_0_segment_offsets.npy  # (N + 1,)
├── user_0_action_0_segment_lengths.npy  # (N,)
├── user_0_action_0_segments.csv
└── user_0_action_0_segmentation_summary.json
```

Segments are ordered by numeric `dataset_id`, then source-line label order
within each dataset. The CSV records source paths, label boundaries, Ring
sample indices, and exact segment lengths for auditability.

Use `--output-dtype float64` to retain the stored Ring precision. Add
`--exclude-last-label` only when a downstream experiment intentionally omits
the final label.
