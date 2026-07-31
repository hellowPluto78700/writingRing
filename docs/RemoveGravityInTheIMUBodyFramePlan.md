# Plan: remove the gravity contribution in the IMU body frame

## Implementation status

Implemented in:

- `src/writingring/gravity.py`;
- `src/writingring/plotting.py`;
- `scripts/plot_ring_linear_acceleration.py`;
- public exports from `src/writingring/__init__.py`;
- `tests/test_gravity.py`, `tests/test_gravity_plotting.py`, and
  `tests/test_gravity_cli.py`; and
- `README.md`, `docs/DATA_FORMAT.md`, and `docs/PROJECT_REPORT.md`.

The completed implementation follows the offline bidirectional estimator,
strict/provisional calibration policy, raw-data preservation, assumed-profile
labeling, and acceptance criteria below. The complete suite passes with
`158 passed`.

## 1. Objective

Add an optional, reusable preprocessing step that estimates the acceleration
contribution associated with gravity in the Ring IMU's sensor/body frame and
subtracts it from the measured acceleration:

```text
measured acceleration in body axes
→ estimated gravity contribution in body axes
→ linear acceleration in body axes
```

The feature must preserve the seven decoded Ring columns, expose every
assumption used by the estimate, and report when the available data cannot
support a trustworthy result. It must not change Ring loading, infer
undocumented units, repair timestamps, use `ring_1`, or claim Ring–Board
synchronization.

In this plan, “body frame” initially means the three axes of the selected
`ring_0` sensor after an explicitly configured axis transform. With the
identity transform, the result must be described as being in the raw Ring
sensor axes. A physical ring/body mounting frame cannot be claimed until its
axis mapping is documented.

## 2. Evidence and constraints

The design is based on `AGENTS.md`, `README.md`,
`docs/PROJECT_REPORT.md`, `docs/DATA_FORMAT.md`, and the relevant upstream
Ring implementation:

- `vendor/WritingRing/ring_plot.py` decodes native-endian `float64` rows in
  the order `acc_x`, `acc_y`, `acc_z`, `gyr_x`, `gyr_y`, `gyr_z`,
  `timestamp`.
- `vendor/WritingRing/core/imu_data.py` uses the same field order.
  `IMUData.scale()` divides acceleration by `9.8`, converts gyroscope values
  with `value / pi * 180`, and flips the y and z signs. These operations are
  evidence for a possible convention, not a declared unit or frame contract.
- The current loader intentionally preserves raw values and labels the Ring
  timestamp's microsecond interpretation as inferred.
- Ring timestamps are nondecreasing but contain many duplicates. They are not
  suitable as direct per-sample integration intervals without a separately
  designed reconstruction or resampling policy.
- The current spectral workflow uses a caller-visible nominal sampling rate
  (200 Hz by default) as an analysis assumption. Gravity estimation should
  follow the same explicit-assumption pattern.
- The data contains no orientation quaternion, rotation matrix, or
  magnetometer channel. Roll and pitch relative to gravity can be estimated
  from the accelerometer and gyroscope, but absolute yaw cannot. Yaw is not
  needed to express gravity in body axes.
- Physical units, gyro scale, acceleration sign, sensor-to-body mounting, and
  a guaranteed nominal sample rate remain undocumented.
- A small sample inspection found acceleration magnitudes near `9.8`, which
  is consistent with the upstream scaling hint, but this observation is not
  a format contract. The beginnings of the sample recordings are not
  reliably stationary, so the first N samples must not be accepted silently
  as calibration data.

Consequently, subtracting a constant from one acceleration axis is invalid:
the gravity contribution rotates among body axes as the Ring rotates.
Likewise, a low-pass filter alone would confuse sustained linear acceleration
with gravity and lag during rotation. The primary method should propagate a
body-frame gravity vector with the gyroscope and use the accelerometer only
for bounded drift correction.

## 3. Scope

### In scope

- The primary `ring_0` acceleration and gyroscope channels.
- A deterministic NumPy implementation with typed configuration, result, and
  diagnostic models.
- An explicit nominal sampling rate.
- Optional, explicit axis and unit conversion.
- Gyroscope propagation plus confidence-gated accelerometer correction.
- Calibration validation and provenance.
- Derived body-frame gravity and linear-acceleration columns.
- Unit tests, sample-data diagnostics, Matplotlib comparison plots, CLI
  integration, and documentation.

### Out of scope

- Parsing changes or any modification under `data_sample/` or
  `vendor/WritingRing/`.
- Use of `*_ring_1.bin`.
- Inferring a physical sensor mounting from the plane-direction constants in
  `IMUData`.
- Absolute heading, magnetometer fusion, navigation, position, or velocity.
- Ring–Board synchronization.
- Timestamp repair, resampling, or replacing raw timestamps.
- Claiming that the result is ground-truth linear acceleration.
- Silently choosing `9.8`, radians/second, a stationary interval, or the
  upstream y/z sign flips.

## 4. Mathematical convention

Use column vectors. Let:

- `a_b[i]` be measured acceleration in configured body axes;
- `omega_b[i]` be angular velocity in the same right-handed axes, converted
  explicitly to radians/second;
- `g_b[i]` be the estimated stationary/gravity contribution expressed in
  body axes;
- `l_b[i] = a_b[i] - g_b[i]` be estimated linear acceleration.

This project should call `g_b` the **gravity contribution** or **stationary
acceleration contribution**, rather than the physical gravitational
acceleration vector. Accelerometers measure specific force, and the sign of
that measurement is not documented here. Initializing the estimate from a
validated stationary measurement makes subtraction yield approximately zero
at rest without asserting an unsupported sign convention.

For a vector fixed in the world but represented in a rotating body frame:

```text
d(g_b)/dt = -omega_b × g_b
```

Propagate one sample with an exact Rodrigues rotation:

```text
omega_interval = (omega_b[i] + omega_b[i + 1]) / 2
g_pred[i + 1] = R(-omega_interval * dt) @ g_b[i]
```

When the accelerometer sample passes the correction gate, correct the
predicted direction toward the normalized measured acceleration:

```text
g_acc = gravity_magnitude * a_b / ||a_b||
base_weight = 1 - exp(-dt / correction_time_constant)
g_corrected = normalize(
    (1 - confidence * base_weight) * g_pred
    + confidence * base_weight * g_acc
) * gravity_magnitude
```

When the gate rejects a sample, use `g_pred` without accelerometer correction.
Normalize after propagation/correction to prevent numerical magnitude drift.
Use the small-angle limit when the Rodrigues angle is close to zero.

Define acceleration confidence deterministically as
`max(0, 1 - relative_norm_error / acceleration_tolerance)`. If an angular
rate gate is configured, calculate an equivalent bounded rate confidence and
multiply the two values. A zero value rejects correction; values in `(0, 1]`
soften the correction near a gate boundary. The gate, thresholds, and
confidence formula must be included in result diagnostics. Rejected samples
are normal during motion, not loader errors.

This is an offline analysis. Anchor the gravity vector at the midpoint of the
validated calibration interval, propagate forward to the end, and propagate
backward to sample zero using the inverse interval rotations and the same
confidence rules. This avoids pretending that an arbitrary first sample was
stationary while retaining one derived row per input row. Document that this
bidirectional result is noncausal and is not suitable for a real-time
controller. A future streaming estimator should have a separate API and
require stationary initialization before emitting calibrated output.

## 5. Configuration and calibration contract

Create immutable, slotted dataclasses in a new
`src/writingring/gravity.py` module.

### `GravityRemovalConfig`

Include at least:

- `sampling_rate_hz`: positive finite nominal processing rate;
- `acceleration_scale_to_working_units`: positive finite multiplier applied
  before calibration and subtraction;
- `acceleration_unit_label`: caller-supplied working-unit label and
  provenance, defaulting to `raw acceleration units`;
- `gyro_scale_to_rad_s`: positive finite multiplier, with no hidden unit
  conversion;
- `axis_transform`: finite right-handed orthonormal `3 x 3` matrix applied to
  both accelerometer and gyroscope vectors;
- `correction_time_constant_s`: positive finite complementary correction
  time constant;
- `acceleration_gate_relative_tolerance`: allowable relative difference from
  calibrated gravity magnitude;
- optional `angular_rate_gate_rad_s`;
- inclusive calibration start and exclusive calibration stop sample indices;
- strictness policy controlling whether failed calibration is an error or an
  explicitly provisional result.

Do not put mutable NumPy arrays directly into equality-sensitive dataclass
fields without defining copying and validation behavior.

### `GravityCalibration`

Store:

- inclusive start and exclusive stop sample indices;
- estimated gravity magnitude;
- initial stationary-acceleration direction;
- gyroscope bias in radians/second;
- acceleration-norm and gyroscope-norm dispersion metrics;
- sample count and pass/fail status;
- warnings and the source of each value (`measured`, `caller supplied`, or
  `assumed`).

Calibration must use an explicitly selected interval. Validate that it has
enough finite samples, sufficiently stable acceleration magnitude, and low
enough angular-rate dispersion. Use robust statistics such as componentwise
medians and median absolute deviation. Do not automatically certify the
first second as stationary.

Allow a caller to supply the gravity magnitude and gyro bias when an external
calibration is available. Caller-supplied values must remain distinguishable
from values estimated from this recording.

### Profiles

If a convenience profile is added for the upstream scaling hints, name it
clearly (for example, `upstream_suggested`) and record that it assumes:

```text
acceleration scale to working units: 1 / 9.8 (assumed g)
gyroscope scale to rad/s:            1 (assumed raw rad/s)
axis transform:                       diag(1, -1, -1)
```

The implementation should perform all fusion internally in radians/second.
An identity acceleration scale should remain available to keep derived values
in raw acceleration units.
The profile must be opt-in and described as an assumption, not as confirmed
metadata. Prefer direct explicit configuration over a profile in the core
API.

## 6. Public result and API

Add:

```python
def calibrate_gravity_removal(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    *,
    config: GravityRemovalConfig,
) -> GravityCalibration:
    ...


def remove_gravity_in_body_frame(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    *,
    config: GravityRemovalConfig,
    calibration: GravityCalibration,
) -> GravityRemovalResult:
    ...
```

Keep the numerical core independent of Pandas and `RingData`. Add a thin
adapter:

```python
def process_ring_gravity(
    ring_data: RingData,
    *,
    config: GravityRemovalConfig,
    calibration: GravityCalibration,
) -> GravityRemovalResult:
    ...
```

### `GravityRemovalResult`

Contain read-only arrays with shape `(N, 3)`:

- configured body-frame acceleration;
- configured and bias-corrected body-frame angular velocity;
- estimated gravity contribution;
- estimated linear acceleration;
- accelerometer-correction mask and confidence values.

Also contain the exact configuration, calibration, and aggregate diagnostics:
sample count, corrected/rejected counts, finite status, residual norm during
the calibration interval, and warnings.

Provide a method that returns a new DataFrame with explicit derived names:

```text
acc_body_x, acc_body_y, acc_body_z
gravity_body_x, gravity_body_y, gravity_body_z
linear_acc_body_x, linear_acc_body_y, linear_acc_body_z
gravity_correction_used, gravity_correction_confidence
```

Do not mutate `RingData.dataframe`. Preserve `acc_*`, `gyr_*`, `timestamp`,
the named `sample_index`, and `relative_time_inferred_s` unchanged.

Use typed errors under `GravityRemovalError` for invalid shape, non-finite
configuration, invalid rotation matrices, bad calibration bounds, failed
strict calibration, and non-finite samples. Error messages should identify
the field and offending condition.

## 7. Time handling

The first implementation should use a fixed:

```text
dt = 1 / sampling_rate_hz
```

The nominal rate is a processing assumption and must appear in diagnostics,
CLI output, and plot annotations. Do not calculate samplewise `dt` directly
from the current raw Ring timestamps because duplicate values would produce
zero-duration integration steps and positive jumps represent batched timing,
not a verified per-sample clock.

A later variable-time mode is acceptable only after a separate timestamp
reconstruction/resampling design is documented and tested. It must not be
smuggled into gravity removal.

## 8. Implementation sequence

### Phase 1: lock the convention with synthetic tests

Before running on sample data:

1. Define the axis, rotation, accelerometer-sign, and subtraction conventions
   in module docstrings.
2. Implement configuration validation and Rodrigues rotation.
3. Test a stationary sensor in several fixed orientations.
4. Test a sensor rotating at a known angular rate with zero linear
   acceleration.
5. Verify that the sign in `d(g_b)/dt = -omega_b × g_b` produces the expected
   body-frame vector.

This phase prevents a visually plausible but sign-inverted filter.

### Phase 2: calibration and estimator

1. Add strict calibration interval validation and choose its midpoint as the
   offline estimator anchor.
2. Estimate robust gravity magnitude/direction and gyro bias only from a
   validated interval.
3. Implement forward and inverse/backward gyro propagation, confidence
   gating, bounded correction, and norm preservation.
4. Return immutable result arrays and complete diagnostics.
5. Ensure no input array or `RingData` object is mutated.

### Phase 3: Ring adapter and visualization

Add a Matplotlib comparison function with shared x axes:

- raw/configured body-frame acceleration;
- estimated gravity contribution;
- resulting linear acceleration;
- a compact indication of correction-gate acceptance.

Support the existing `sample_index` and inferred-time display choices without
creating or repairing a time column. Labels must use “raw acceleration units”
unless the caller explicitly supplies a documented unit. Titles or footers
must state the nominal rate, axis transform/profile, calibration interval,
and provisional warnings.

Do not replace the current raw `plot_ring_imu` or acceleration PSD plots.
Gravity-removed signals are a separate, opt-in product. If spectral analysis
is later applied to `linear_acc_body_*`, the output filename, legend, and
summary must distinguish it from the existing raw-acceleration PSD.

### Phase 4: CLI

Prefer a separate script, such as
`scripts/plot_ring_linear_acceleration.py`, over adding implicit behavior to
the existing commands. Require explicit selection of the recording and
calibration interval. Expose:

- nominal sampling rate;
- gyro scale/unit choice;
- axis transform or named opt-in profile;
- calibration start/stop;
- correction time constant and gate thresholds;
- strict/provisional calibration policy;
- output path, `--overwrite`, and `--no-show`.

Print a concise processing summary and return status `2` for expected
configuration, calibration, loading, plotting, or filesystem failures.
Select Matplotlib's Agg backend before importing pyplot in noninteractive
mode, matching the existing scripts.

### Phase 5: documentation and acceptance

Update:

- `README.md` with an assumption-forward usage example;
- `docs/DATA_FORMAT.md` only to distinguish raw fields from the new derived
  fields—do not present the derived result as part of the file format;
- `docs/PROJECT_REPORT.md` with the module/API, transformations, diagnostics,
  tests, and limitations;
- the package public API only after the lower-level contract is stable.

Document the intentional upstream difference: the upstream implementation
scales and plots IMU values but does not estimate orientation or remove
gravity. The new feature preserves upstream decoding semantics while adding
an optional derived analysis.

## 9. Test plan

### Unit tests

- Reject arrays other than finite numeric `(N, 3)` acceleration and gyro
  pairs of equal nonzero length.
- Reject invalid rates, scales, thresholds, calibration ranges, and
  non-orthonormal or left-handed transforms.
- Verify exact identity and 90/180-degree Rodrigues rotations.
- At rest in multiple orientations, estimate near-zero linear acceleration.
- Under constant known rotation and no translation, track the rotating
  body-frame gravity vector and retain near-zero linear acceleration.
- Add known body-frame linear acceleration and recover it within a stated
  tolerance.
- Add constant gyro bias and verify calibration removes it.
- Exercise high-dynamic intervals and verify the correction gate rejects
  misleading acceleration samples.
- Exercise long propagation intervals and verify the gravity estimate retains
  its calibrated magnitude.
- Verify strict calibration rejects motion and provisional mode emits a
  structured warning.
- Verify inputs and the source Ring DataFrame remain byte-for-byte/value
  unchanged.
- Verify deterministic results and read-only output arrays.

### Plot and CLI tests

- Check panel content, labels, warning footer, calibration annotation, and
  both supported x-axis modes.
- Check output parent creation, overwrite protection, Agg mode, and figure
  closure.
- Check that only `ring_0` is loaded.
- Check that expected failures are concise and omit tracebacks.
- Check that raw and gravity-removed PSD products cannot be confused.

### Sample-data acceptance

Use `data_sample/data` read-only. For each `ring_0` recording:

1. run diagnostics with explicitly recorded assumptions;
2. select and document a calibration interval rather than assuming the
   initial samples are stationary;
3. confirm all output arrays are finite and have the original row count;
4. inspect correction acceptance, gravity magnitude stability, calibration
   residuals, and gyro drift warnings;
5. compare raw acceleration, estimated gravity contribution, and residual
   acceleration visually;
6. record results as empirical observations, not proof of the unit/frame
   convention.

Hash or otherwise verify that `data_sample/` and `vendor/WritingRing/` remain
unchanged during acceptance work.

Run the complete suite after every code change:

```bash
conda run --no-capture-output -n writingring-viz python -m pytest -q
```

## 10. Acceptance criteria

The feature is complete only when:

- raw Ring loading and all existing tests remain unchanged in behavior;
- the mathematical frame/sign convention is documented and protected by
  synthetic rotating-body tests;
- every scale, axis, rate, and calibration assumption is explicit and
  serialized in diagnostics;
- no default silently certifies undocumented units or a stationary interval;
- the estimator returns same-length, finite, non-mutating derived arrays;
- stationary synthetic cases produce near-zero residual acceleration;
- known synthetic linear acceleration is recovered within declared numeric
  tolerances;
- dynamic acceleration cannot freely overwrite the gravity estimate;
- plots and CLI output distinguish raw, configured-body, gravity, and linear
  acceleration;
- sample-data output is labeled provisional unless the unit, frame, and
  calibration assumptions are independently confirmed;
- no files under `data_sample/` or `vendor/WritingRing/` are modified; and
- the full pytest suite passes in the `writingring-viz` Conda environment.

## 11. Main risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Gyro units or signs are wrong | Require explicit scale and axis transform; protect the convention with synthetic tests; mark the upstream profile as assumed |
| No stationary calibration interval exists | Fail strict calibration or accept external calibration; never silently use the first samples |
| Linear acceleration is mistaken for gravity | Gate correction by norm/rate consistency and expose confidence and acceptance |
| Gyro bias causes drift | Estimate bias only from validated stationary data and report long correction gaps |
| Duplicate timestamps corrupt integration | Use an explicit fixed nominal rate and defer timestamp reconstruction |
| Sensor axes are called a physical body frame | Label identity-mapped results as raw sensor axes; require a mounting transform for physical body labels |
| Accelerometer sign is confused with physical gravity | Estimate the stationary measurement contribution and define subtraction operationally |
| Filtering hides provenance | Preserve raw columns and serialize configuration and calibration diagnostics |
| Results appear more certain than the evidence | Put provisional status and assumptions in APIs, CLI summaries, plots, and documentation |

## 12. Decision gate before implementation

The algorithm can be implemented and validated synthetically now, but
sample-data results cannot be promoted from provisional to physically
calibrated until at least these facts are confirmed or deliberately supplied
as processing assumptions:

1. gyroscope raw unit and sign convention;
2. acceleration raw unit and stationary sign convention;
3. sensor-to-ring/body mounting transform;
4. nominal per-sample rate or an approved timestamp reconstruction;
5. a stationary calibration interval or external calibration values.

If those facts remain unavailable, the implementation should still be useful
for controlled experiments, but its public language must remain “estimated,”
“configured,” and “provisional.”
