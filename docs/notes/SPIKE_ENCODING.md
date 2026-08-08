# Spike encoding

`scripts/encode_spikes.py` is a read-only consumer of one complete,
preprocessed Ring IMU recording. Gravity removal, resampling, segmentation,
and label-boundary handling happen outside the encoding layer.

The preprocessing timestamp sidecar is part of the handoff contract:
`preprocessedIMU[i]`, `timestamps_us[i]`, and the published `spikeIMU[i]`
describe the same row. Encoding validates the declared source timestamp
SHA-256 before publication and carries the verified digest into SpikeIMU
metadata. Consumers must reuse this canonical vector; they must not sort,
deduplicate, resample, or replace it with an endpoint-reconstructed axis for
segmentation. Alignment may use a separate endpoint-reconstructed strictly
increasing work axis internally, but
its provenance hash always remains the canonical source hash.

## Source contract

The preferred source is:

```text
<user>/<action>/<data_id>/<data_id>_preprocessedIMU.npy
<user>/<action>/<data_id>/<data_id>_preprocessing.json
```

The NPY must be a finite numeric `(N, 9)` array with this exact order:

```text
acceleration_x_g, acceleration_y_g, acceleration_z_g,
acceleration_x, acceleration_y, acceleration_z,
gyro_x, gyro_y, gyro_z
```

Columns `0:3` are body-frame acceleration in `g`; columns `3:6` are the same
signal in `m/s²`; columns `6:9` are gyroscope values in `rad/s`. The loader
requires:

```text
imu[:, 3:6] == imu[:, 0:3] * 9.80665
```

within the documented floating-point tolerance. This rejects legacy
`acceleration + gyro + magnetometer` arrays even when they happen to have nine
columns. The encoder consumes only `imu[:, :3]` and the Custom Wavelet
publication preserves `imu[:, 3:9]` as its trailing six channels.

The summary `units` field is an ordered nine-element list and must equal:

```text
["g", "g", "g", "m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s"]
```

The preprocessing summary must declare `schema_version: 1`, the nine
`channel_names`, `sample_count`, `sampling_rate_hz`,
`standard_gravity_m_s2: 9.80665`, `acceleration_semantics`,
`gravity_removal_method`, `gravity_removed`, and `units`, plus recording
identity, source NPY path, and source NPY SHA-256. The consumer cross-checks
sample count, channel names, source path, digest, recording identity when
supplied, and sampling rate. `raw`/gravity-included summaries are rejected by
default; use
`--allow-gravity-included` only for an intentional measured-acceleration run.
That flag does not waive the complete metadata contract: the summary must
explicitly identify `raw` measured acceleration with `gravity_removed=false`.

Published `metadata.json` uses the `spike_encoding_v3` contract and carries the
source IMU digest, canonical timestamp path/digest when available, timestamp
unit, recording identity, gravity-removal semantics, and row-alignment flags.
When the preprocessing summary declares a timestamp digest, the encoder hashes
the actual timestamp file immediately before publication and records that the
source hash was verified; a replacement file with the same length is rejected.

The low-level Xylo API returns only `(N, 3)` acceleration. That is not a valid
spike handoff. Use the complete `(N, 9)` result from
`preprocess_ring_imu(...).imu` or the preprocessing export CLI.

## Single-recording command

```bash
python scripts/encode_spikes.py \
  --input-imu outputs/preprocessedIMU/user_0/0/0/0_preprocessedIMU.npy \
  --encoder custom-wavelet \
  --encoder-settings configs/spike_encoding/custom_wavelet.json
```

Custom Wavelet treats the entire file as one recording and uses
`boundaries=[0, N]`. It resets exactly once. Directory names such as
`user/action/data_id` organize artifacts only; they never create internal
encoder boundaries. Sequence offsets, labels, and segment manifests cannot
reset this encoder. An optional timestamps NPY is validation-only.

## Batch command

```bash
python scripts/encode_spikes.py \
  --input-root outputs/preprocessedIMU \
  --pattern '*_preprocessedIMU.npy' \
  --output-root outputs/spikeEncoding \
  --encoder custom-wavelet \
  --encoder-settings configs/spike_encoding/custom_wavelet.json
```

Each input is loaded and encoded independently with a fresh encoder. Relative
recording directories are retained:

```text
outputs/spikeEncoding/
└── custom-wavelet/
    └── <user>/<action>/<data_id>/
        ├── spikes.npy
        ├── spikeIMU.npy
        ├── recording_offsets.npy
        ├── sequences.csv
        └── metadata.json
```

Batch publication accepts an explicit output root and never assumes it is the
input directory. Each recording remains atomic and existing output is
protected unless `--overwrite` is supplied. A destination containing
unrelated files is never removed.

## Custom Wavelet settings and outputs

The included `configs/spike_encoding/custom_wavelet.json` uses frequencies
`[0.5, 1, 2, 4, 8]` Hz at 200 Hz, so the output has 15 channels in
axis-major/frequency-minor order. Sampling rate is used to derive wavelet
widths and the extrema window; it never resamples data. If settings and the
preprocessing summary both provide `sampling_rate_hz`, they must match
exactly. If neither provides it, encoding fails.

Custom Wavelet emits signed local-extrema amplitudes, not binary spike trains.
It reflect-pads by half of its odd extrema window, compensates the fixed
extrema-confirmation latency, and returns events to their occurrence rows. At
the default settings the half-window is 30 samples (0.15 seconds) on each
side. IIR phase/group delay and warmup are not compensated.

`spikes.npy` has shape `(N, 15)`. `spikeIMU.npy` has shape `(N, 21)`:

```text
15 signed event channels
3 acceleration channels in m/s²
3 gyroscope channels in rad/s
```

The six trailing values are copied row-for-row from source columns `3:9`;
they are processed acceleration when the source method is low-pass, Madgwick,
or Xylo, and measured acceleration only when raw mode was explicitly allowed.
`recording_offsets.npy` is `[0, N]`. `metadata.json` records the source
contract, encoder settings, output schema, reset boundary, and row alignment.
For the canonical preprocessing handoff, its `spike_imu` section also records
the 21-channel schema, event units, trailing IMU units, source columns `3:9`,
sample count, and the copied timestamp provenance.

The downstream segmentation matrix is:

```text
columns 0:15   signed wavelet events
columns 15:18  acceleration x/y/z (m/s²)
columns 18:21  gyro x/y/z (rad/s)
```

Alignment and Board-assisted segmentation use only `15:21` for transient
detection and require the SpikeIMU values, metadata, and canonical timestamp
hashes to match the saved alignment offset.

## Legacy segmentation sources

Older `*_rawIMU.npy` segmentation exports remain readable when they satisfy the
same nine-channel physical contract. A legacy segmentation summary may be
passed with `--input-summary`; labels and offsets are provenance only and are
not used as Custom Wavelet boundaries. Segmentation and occurrence-aligned
workflows are optional downstream consumers, not part of the complete
gravity-to-spike handoff.
