# Gravity-to-spike pipeline

The complete-recording pipeline is:

```text
Ring *_ring_0.bin
  → preprocess_ring_imu()
  → (N, 9) preprocessed artifact + canonical timestamps + JSON summary
  → one independent Custom Wavelet encoding
  → (N, 15) events and (N, 21) spike IMU
```

The preprocessing export is a file-backed handoff for one complete recording,
not a segmentation or windowing step. Its stable namespace is:

```text
outputs/preprocessedIMU/<user>/<action>/<data_id>/
├── <data_id>_preprocessedIMU.npy
├── <data_id>_timestamps_us.npy
└── <data_id>_preprocessing.json
```

The feature and timestamp arrays have the same row count and preserve the
source Ring order. Timestamps are validated as finite and nondecreasing;
duplicate values are retained. No timestamp sorting, deduplication, or
resampling is performed.

The exporter uses the repository's validated Ring loader. As in the upstream
`vendor/WritingRing/ring_plot.py`, Ring samples are native-endian float64 rows
with seven values: six IMU values followed by the stored timestamp. This
implementation intentionally keeps the upstream primary-file choice
(`*_ring_0.bin`) but adds structured validation, preserves the raw timestamp
outside the preprocessing artifact, and does not infer undocumented physical
units. The exporter never reads or modifies `*_ring_1.bin`.

## Artifact contract

The nine columns are fixed:

```text
acceleration_x_g, acceleration_y_g, acceleration_z_g,
acceleration_x, acceleration_y, acceleration_z,
gyro_x, gyro_y, gyro_z
```

The first triplet is the selected preprocessing result in `g`; the second is
the same signal in `m/s²`; the final triplet is gyro in `rad/s`. Every reader
checks finite values and the relation `m/s² = g × 9.80665`. The summary also
records method, semantics, sampling rate, row count, channel names, recording
identity, and the artifact digest. Its ordered `units` list must be exactly
`["g", "g", "g", "m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s"]`.

`raw` remains available for producing an explicitly gravity-included artifact,
but spike encoding rejects its summary by default. `--allow-gravity-included`
is an explicit opt-in at the consumer boundary.

The preprocessing summary also records the canonical timestamp path and
SHA-256, the source `ring_0.bin` path and SHA-256, `gravity_removed`, the
standard gravity constant `9.80665`, output dtype, and the recording identity.
Complete `*_preprocessedIMU.npy` artifacts require this metadata contract;
loaders reject missing or inconsistent fields before encoding. `raw` is the
explicit measured-acceleration mode, while the other methods declare
gravity-removed acceleration semantics.

The preprocessing CLI defaults to a 200 Hz sampling-rate assumption and
supports `low-pass`, `madgwick`, and Xylo rotation/gravity removal in addition
to `raw`. Madgwick can be exported provisionally when stationary calibration
checks fail. All discovered recordings are exported in numeric recording order,
or one exact `user/action/data_id` can be selected; `--overwrite` is required
to replace an existing artifact namespace.
The exporter stages the NPY, timestamp sidecar, and JSON summary together,
reloads and validates them, then atomically publishes the complete directory;
it refuses to overwrite unexpected files in an existing namespace.

## Boundaries and provenance

One exported NPY file is one complete recording. Custom Wavelet uses
`[0, N]`, resets exactly once, and never consults label or segment offsets.
Batch encoding creates a fresh encoder for every file and mirrors the relative
`user/action/data_id` path beneath the requested output root. Output publication
is staged, reloaded, verified, and atomically replaced only with `--overwrite`.
Each recording is independent: encoder state resets once at `[0, N]`, and one
bad or existing destination cannot silently merge with another recording.

The bottom-level Xylo rotation/gravity function returns `(N, 3)` acceleration.
The handoff deliberately uses the wrapper's `(N, 9)` result so the consumer
can preserve the m/s² and gyro channels and prove their semantics in metadata.
