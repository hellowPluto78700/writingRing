# Preprocessing pipeline entry points

This directory contains the shell entry points for WritingRing preprocessing and derived feature datasets. The numbered scripts run the canonical raw/low-pass/Madgwick/Xylo pipelines. The variant builders consume an already completed pipeline and preserve its timestamp, alignment, segmentation, and padding geometry unless their documentation explicitly says otherwise.

## 64 Hz linear + angular-acceleration SpikeIMU (66 channels)

Use `build_angular_accel_66ch_variant.bash` when the source pipeline is already complete at **64 Hz** and its recording-level SpikeIMU is the canonical **36-channel PolaritySplitAbs** layout:

```text
30 linear-acceleration Custom-Wavelet event channels
+ 3 acceleration channels in m/s^2
+ 3 gyroscope channels in rad/s
= 36 channels
```

The builder does not rerun gravity removal, resampling, alignment, or boundary detection. For each complete recording it derives angular acceleration from the existing gyroscope signal, runs the same Custom Wavelet encoder, and then reuses the already validated segmentation and padding geometry.

### Signal transform

For the complete recording, before segmentation:

```text
gyroscope omega(t) [rad/s]
    -> numpy.gradient(..., dt=1/64, edge_order=2)
angular acceleration alpha(t) [rad/s^2]
    -> Custom Wavelet
       frequencies = 0.5, 1, 2, 4, 8 Hz
       max_filter_time_s = 0.3
    -> PolaritySplitAbs
30 unsigned angular-acceleration event channels
```

The derivative is evaluated on the **whole recording**, not independently inside each segment. This prevents artificial derivative discontinuities at letter boundaries and preserves the original row count and timestamp indexing.

At 64 Hz the five Custom-Wavelet widths are 128, 64, 32, 16, and 8 samples. `max_filter_time_s=0.3` remains the extrema-detection window parameter; it is not a 0.3 s cap on the wavelet width.

### 66-channel layout

The durable layout is:

```text
[0:30)   linear-acceleration PolaritySplitAbs events
[30:60)  angular-acceleration PolaritySplitAbs events
[60:63)  acceleration x/y/z in m/s^2
[63:66)  gyroscope x/y/z in rad/s
```

Feature schema:

```text
linear_accel_angular_accel_polarity_split_wavelet_events_plus_imu_v1
```

Event schema:

```text
linear_accel_angular_accel_polarity_split_abs_events_v1
```

The final six channels deliberately remain the original physical IMU channels. This preserves the existing trailing-IMU invariant used by transient/alignment logic.

### Geometry reuse contract

The builder treats alignment, segmentation, and padding geometry as immutable inputs:

- **Alignment is not recomputed.** If the source has an `alignment/` tree, the finalizer copies it byte-for-byte as geometry/provenance evidence. It intentionally does not rewrite old alignment hashes to pretend that the 66-channel feature matrix was aligned again.
- **Segmentation boundaries are not recomputed.** The source `segments.csv` supplies `dataset_id`, `start_sample_index`, and `stop_sample_index_exclusive`; the new 66-channel recording is sliced with those exact indices.
- Labels, segment offsets, segment lengths, and Board-event targets are copied and verified unchanged.
- **Padding target/geometry is not recomputed.** The existing padding manifest determines which segment occupies each padded output row. `valid_lengths`, `valid_mask`, labels, and optional Board-event targets remain unchanged.
- Padding in the new angular-event channels uses the same stored padding value as the source dataset.

The derived output therefore remains dependent on the source pipeline for canonical timestamp provenance. Do not delete or move source timestamp artifacts after building the variant unless the metadata paths are deliberately migrated and revalidated.

### Output tree

The output mirrors the interfaces used by the normal pipeline:

```text
<output-root>/
├── spikeEncoding/custom-wavelet/<user>/<action>/<dataset_id>/
│   ├── spikeIMU.npy              # (N, 66)
│   ├── spikes.npy                # (N, 60)
│   ├── angularAcceleration.npy   # (N, 3), rad/s^2
│   ├── metadata.json
│   └── ... copied recording-offset/sequence artifacts
├── segmentation/<user>/action_<action>/
│   ├── <user>_action_<action>_spikeIMU.npy
│   ├── labels / offsets / lengths / manifests
│   └── segmentation summary
├── segmentation_padded/<user>/action_<action>/
│   ├── *_paddedSpikeIMU.npy      # (segments, T, 66)
│   ├── valid lengths / mask / labels
│   └── padding summary / manifest
├── user_reports/<user>.json
├── alignment/                    # copied only when present in source
└── angular_accel66_dataset_summary.json
```

### Run locally: one user per CPU

```bash
MODE=local JOBS=20 \
  bash scripts/bash_script/preprocessing_pipeline/build_angular_accel_66ch_variant.bash \
  outputs/<completed-64hz-combination-root>
```

The default output is a sibling named:

```text
outputs/<completed-64hz-combination-root>_angular_accel66
```

Use an explicit output root as the second positional argument when desired:

```bash
MODE=local JOBS=20 \
  bash scripts/bash_script/preprocessing_pipeline/build_angular_accel_66ch_variant.bash \
  <source-root> <output-root>
```

`JOBS` is capped at 50. Every worker exports `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` so one user task does not silently oversubscribe CPU cores.

### Run on Unity with a Slurm array

The recommended cluster path is one array task per user and one CPU core per task:

```bash
MODE=submit \
SLURM_MAX_CONCURRENCY=50 \
SLURM_TIME=02:00:00 \
SLURM_MEM=4G \
  bash scripts/bash_script/preprocessing_pipeline/build_angular_accel_66ch_variant.bash \
  <source-root> <output-root>
```

The submit mode creates:

```text
user_0  -> one array task / one CPU
user_1  -> one array task / one CPU
...
all user tasks PASS
  -> afterok finalizer job
```

The finalizer only aggregates/validates worker outputs; it does not regenerate missing users. `SLURM_MAX_CONCURRENCY` is hard-capped at 50 in accordance with the repository execution policy.

### Two completed datasets

Run the script independently for each completed 64 Hz combination root. For example:

```bash
MODE=submit bash scripts/bash_script/preprocessing_pipeline/build_angular_accel_66ch_variant.bash \
  outputs/<dataset-A-root> outputs/<dataset-A-root>_angular_accel66

MODE=submit bash scripts/bash_script/preprocessing_pipeline/build_angular_accel_66ch_variant.bash \
  outputs/<dataset-B-root> outputs/<dataset-B-root>_angular_accel66
```

This keeps the two source datasets and their provenance isolated.

### Overwrite and restart behavior

By default the builder refuses to replace already published derived user packages. To intentionally rebuild the derived variant:

```bash
OVERWRITE_DEST=1 MODE=local \
  bash scripts/bash_script/preprocessing_pipeline/build_angular_accel_66ch_variant.bash \
  <source-root> <output-root>
```

Each user is published through staging and rename. A failed user does not publish a partially built user package. In Slurm mode the finalizer uses `afterok`, so it will not run when any user task fails.

### Validation artifacts

Each worker checks, among other invariants:

- source is 64 Hz, 36-channel PolaritySplitAbs;
- source frequency list and max-filter setting match the requested defaults;
- angular acceleration preserves `(N, 3)` and finite values;
- angular event output is `(N, 30)` and nonnegative;
- first 30 linear-acceleration event channels are bitwise unchanged;
- final six physical IMU channels are bitwise unchanged;
- segment geometry and Board targets match the source;
- padding valid lengths/mask/labels match the source.

Per-user results are written to `user_reports/`; the finalizer writes `angular_accel66_dataset_summary.json`.

### Parameters

Defaults are intentionally fixed to the planned dataset:

```text
SAMPLING_RATE_HZ=64
FREQUENCIES_HZ="0.5 1 2 4 8"
MAX_FILTER_TIME_S=0.3
```

The Python helper rejects a non-64 Hz sampling rate. Frequency and max-filter overrides are accepted only when the source 36-channel encoder metadata declares the same settings, preventing silent feature-schema drift.

No smoothing is inserted before the gyroscope derivative. If later analysis shows high-frequency angular-acceleration events dominate, evaluate derivative filtering as a separate ablation rather than silently changing this dataset contract.

## Existing polarity-split variant

`build_polarity_split_variant.bash` remains the builder for converting a completed signed 21-channel padded SpikeIMU dataset into the 36-channel polarity-split padded representation. The new 66-channel builder does not replace or modify that existing workflow.
