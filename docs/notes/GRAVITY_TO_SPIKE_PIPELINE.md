# Gravity-to-spike pipeline

The complete-recording pipeline is:

```text
Ring *_ring_0.bin
  → preprocess_ring_imu()
  → (N, 9) preprocessed artifact + JSON summary
  → one independent Custom Wavelet encoding
  → (N, 15) events and (N, 21) spike IMU
```

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

## Boundaries and provenance

One exported NPY file is one complete recording. Custom Wavelet uses
`[0, N]`, resets exactly once, and never consults label or segment offsets.
Batch encoding creates a fresh encoder for every file and mirrors the relative
`user/action/data_id` path beneath the requested output root. Output publication
is staged, reloaded, verified, and atomically replaced only with `--overwrite`.

The bottom-level Xylo rotation/gravity function returns `(N, 3)` acceleration.
The handoff deliberately uses the wrapper's `(N, 9)` result so the consumer
can preserve the m/s² and gyro channels and prove their semantics in metadata.
