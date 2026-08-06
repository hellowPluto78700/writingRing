# Xylo rotation and gravity removal

`--gravity-removal-method xylo-rotate-and-remove-gravity` processes Ring
acceleration without resampling or changing timestamps. Ring input remains
the upstream six-signal source schema in m/s². Its acceleration is converted
to g before the Rockpool Xylo `Quantizer` and `RotationRemoval` stages. The
result is converted back to m/s² only after Xylo gravity removal.

Xylo support is optional. Install the tested Rockpool/Xylo runtime before
selecting this method:

```bash
pip install -e ".[xylo]"
```

Published installations can use `pip install "writingring[xylo]"` instead.
The raw, low-pass, and Madgwick methods do not import or require Rockpool.

The Xylo simulator itself provides fixed-point rotation removal, not a named
linear-acceleration output. The project therefore subtracts an explicit,
causal power-of-two moving gravity baseline after its rotated g-domain output.
This is an intentional project-level processing step; it does not change the
upstream Ring parser or claim undocumented Ring units.

Every preprocessing method returns these nine channels:

```text
acceleration_x_g, acceleration_y_g, acceleration_z_g,
acceleration_x, acceleration_y, acceleration_z,
gyro_x, gyro_y, gyro_z
```

For Xylo, the g triplet is canonical and the m/s² triplet is exactly the
canonical values multiplied by `9.80665`. The output summary uses
`acceleration_semantics: xylo_gravity_removed_acceleration`.

## Spike encoding handoff

The valid handoff is the complete result:

```python
preprocess_ring_imu(...).imu  # shape (N, 9)
```

The low-level value below is acceleration-only and cannot be passed directly
to `scripts/encode_spikes.py`:

```python
xylo_rotate_and_remove_gravity(...).linear_acceleration_g  # shape (N, 3)
```

Preprocessing preserves `N`, retains the Ring gyroscope triplet, and writes
the same g/m/s² dual-acceleration schema used by the spike loader. A complete
file is one recording; `user/action/data_id` directories are organizational
only and do not reset encoder state. The recommended files are
`<data_id>_preprocessedIMU.npy` and `<data_id>_preprocessing.json`; the summary
must identify the method, sampling rate, units, channel names, sample count,
recording identity, source path, and NPY digest.
