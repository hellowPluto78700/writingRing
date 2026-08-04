# WritingRing local inspection and visualization

This project provides a local, reusable workflow for inspecting and
visualizing the WritingRing dataset. It includes:

- recording discovery by user, action, and numeric dataset ID;
- Ring binary loading with structured validation;
- Board chunk loading with structured validation;
- reusable Matplotlib plots;
- command-line listing, inspection, and plotting tools; and
- a Jupyter notebook for interactive exploration.

The implementation is intentionally conservative: it preserves the stored
data and reports anomalies without silently repairing them. See
[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for the durable, source-backed
format reference.

## Ring--Board alignment export and verification

`scripts/align_ring_board.py` estimates one constant offset only when the
sequence matcher succeeds, writes a directed microsecond TXT record, and
generates a six-panel verification image. The shared convention is
`ring_timestamp_us = board_timestamp_us + offset_us`; both artifacts use the
same `best_offset_us` and never change raw timestamps.

```bash
python scripts/align_ring_board.py \
  --data-root data_sample/data \
  --user user_0 --action 0 --dataset-id 0 \
  --verification-output-root outputs/alignmentVerification \
  --label-time-domain shared
```

See [docs/ALIGNMENT_OUTPUTS.md](docs/ALIGNMENT_OUTPUTS.md) for the output
layout, label-time-domain behavior, and overwrite policy.

## Timestamp-label IMU segmentation

Remove gravity and export variable-length primary-Ring IMU segments for all
datasets of one user/action, using each dataset's timestamp labels as
half-open boundaries. The default is 0.2 Hz low-pass filtering and writes to
`outputs/segmentedIMU_LowPassFiltering`:

```bash
python scripts/segment_ring_imu.py \
  --data-root data_sample/data \
  --user user_0 --action 0 \
  --output-root outputs/segmentedIMU
```

The exporter writes contiguous gravity-removed IMU plus segment offsets and
lengths, string labels, an audit CSV, and JSON summary. Use
`--gravity-removal-method madgwick` for sensor fusion; it writes to
`outputs/segmentedIMU_Madgwick` by default. It is strict about
malformed/out-of-range labels and does not overwrite by default. See
[docs/IMU_SEGMENTATION.md](docs/IMU_SEGMENTATION.md) for segment boundary,
duplicate-timestamp, and overwrite semantics.

## Important data rules

Recordings are organized using this hierarchy:

```text
data root/
└── user/
    └── action/
        ├── {dataset_id}_ring_0.bin
        ├── {dataset_id}_ring_1.bin
        ├── {dataset_id}_timestamp.txt
        └── {dataset_id}_board_{chunk_index}.gz
```

The project applies the following rules:

- Only `*_ring_0.bin` is used as the primary Ring input.
- `*_ring_1.bin` is retained as optional discovery metadata and is never
  selected as the primary Ring payload.
- Board files are matched to the same dataset ID and ordered by the numeric
  chunk index, not by lexical filename order or timestamps.
- Board chunks are deserialized with `compress_pickle.load` after importing
  the official pickle classes from `core.sensel_lib.frame_data`.
- Physical units for Ring signals and Board coordinates and force remain
  undocumented. Values are labeled as raw, stored, or undocumented.
- Ring timestamp microseconds are an observed and inferred interpretation,
  not a confirmed upstream contract. The raw timestamp is preserved.
- Ring and Board streams are not assumed to be synchronized.
- Dataset 0 has a backward Board timestamp boundary between chunks `6` and
  `7`.
- No Board chunk is silently dropped, reordered, repaired, split, or sorted
  by timestamp. Dataset 0 chunks `0` through `15`, including its empty leading
  chunk and older tail, remain represented.

## Environment setup

Create and activate the Python 3.11 Conda environment, then install the
project editably:

```bash
conda create -n writingring-viz python=3.11 -y
conda activate writingring-viz
python -m pip install -e .
```

Testing and notebook tooling are optional extras:

```bash
python -m pip install -e ".[test]"
python -m pip install -e ".[notebook]"
```

Install both for development and acceptance work:

```bash
python -m pip install -e ".[test,notebook]"
```

The runtime installation includes NumPy, Pandas, Matplotlib, and
`compress-pickle`. The notebook extra includes JupyterLab, `ipykernel`,
`nbconvert`, and `nbformat`.

## Dataset setup

Place the local dataset below a configurable data root. The included sample
layout is expected at:

```text
data_sample/data/user_0/0/
```

Within that action directory, filenames identify dataset IDs and Board chunk
indices. Discovery does not require the user, action, or dataset ID to be
`user_0`, `0`, or `0`.

The raw dataset, downloaded archives, and generated `outputs/` are local
artifacts and are not intended to be committed to Git. Do not modify
`data_sample/` or the read-only upstream reference under
`vendor/WritingRing/`.

To publish every `downloads/**/data.zip` archive into the documented full-data
layout without changing the protected sample, run:

```bash
python scripts/extract_downloaded_data.py --output-root data
```

The extractor keeps only the archive's `data/user_<id>/<action>/...` payload,
skips macOS metadata, rejects unsafe paths and destination collisions, stages
the extraction before publishing it as `data/`, and never overwrites an
existing root. For the included archive, use `--data-root data` with the
inspection and plotting commands.

## Command-line usage

All commands below are run from the project root after editable installation.

List recordings:

```bash
python scripts/list_recordings.py \
    --data-root data_sample/data
```

Inspect one recording and its validation reports:

```bash
python scripts/inspect_recording.py \
    --data-root data_sample/data \
    --user user_0 \
    --action 0 \
    --dataset-id 0
```

Add `--json-output PATH` to save the same inspection information as strict
JSON.

Generate the standard noninteractive plots and summary:

```bash
python scripts/plot_recording.py \
    --data-root data_sample/data \
    --user user_0 \
    --action 0 \
    --dataset-id 0 \
    --output-dir outputs/dataset_0 \
    --no-show
```

The plotting command writes:

```text
ring_imu.png
touch_trajectory.png
board_force_time.png
summary.json
```

Only these four expected files may be replaced when the same explicit output
directory is reused; unrelated files are left alone.

Ring time-axis choices are:

- `inferred_time` (default): uses the existing
  `relative_time_inferred_s` column and visibly identifies the unconfirmed
  microsecond interpretation;
- `sample_index`: uses the loaded DataFrame's named sample index.

Select them with `--ring-time-axis inferred_time` or
`--ring-time-axis sample_index`.

Board force time-axis choices are:

- `frame_index` (default): uses the global frame index in numeric chunk order;
- `raw_timestamp`: uses stored Board timestamps without sorting or repair.

Select them with `--board-time-axis frame_index` or
`--board-time-axis raw_timestamp`. Neither mode claims synchronization with
the Ring stream.

## Ring acceleration PSD comparison

Generate one combined windowed power-spectral-density figure for a selected
primary Ring recording:

```bash
python scripts/plot_ring_accel_spectrum.py \
  --data-root data_sample/data \
  --user user_0 \
  --action 0 \
  --dataset-id 0 \
  --sampling-rate 200 \
  --window-seconds 1 \
  --overlap 0.5 \
  --aggregate mean \
  --frequency-min 0 \
  --frequency-max 30 \
  --output outputs/dataset_0/ring_accel_psd_overlay.png \
  --no-show
```

The figure analyzes the complete selected `ring_0` stream and contains three
vertically stacked panels for acceleration X, Y, and Z. Gray curves are the
PSDs of individual complete windows; the highlighted curve is the selected
mean or median PSD. Defaults use a nominal 200 Hz analysis rate, one-second
windows, 50% overlap, and therefore 200-sample windows, 100-sample hops, and
1 Hz frequency resolution. An incomplete final tail is dropped without
padding.

The 200 Hz rate is an explicit processing assumption, not a confirmed
upstream sampling-rate contract. Acceleration physical units are undocumented,
so PSD values are labeled in raw acceleration units²/Hz. This whole-recording
analysis does not use marker intervals, load Board data, synchronize Ring and
Board, repair timestamps, or resample the signal. Use `--overwrite` to replace
an existing output file explicitly.

To analyze the estimated body-frame linear acceleration instead of the raw
acceleration, add `--remove-gravity`:

```bash
python scripts/plot_ring_accel_spectrum.py \
  --data-root data_sample/data \
  --user user_0 \
  --action 0 \
  --dataset-id 0 \
  --remove-gravity \
  --provisional \
  --output outputs/dataset_0/ring_linear_acceleration_psd.png \
  --no-show
```

Raw acceleration remains the default. With `--remove-gravity`, the script uses
the same offline bidirectional gravity estimator and automatic stationary
search as `plot_ring_linear_acceleration.py`, then applies the selected PSD or
frequency-support analysis to the resulting linear acceleration. The default
search evaluates 0.10-second windows at a nominal 200 Hz, assumes acceleration
in `m/s^2` and gyro values in `rad/s`, and is only a calibration aid—not proof
of physical stationarity. Use `--calibration-start` and `--calibration-stop`
together to override the automatic interval. Strict processing errors when no
candidate passes; `--provisional` analyzes the best candidate and reports its
failed checks. This workflow preserves the source recording and is offline.

## Body-frame gravity-contribution removal

The optional offline gravity workflow estimates the stationary acceleration
contribution in explicitly configured Ring sensor/body axes and subtracts it
from measured acceleration. It preserves every raw Ring column and returns
same-length derived acceleration, gravity-contribution, linear-acceleration,
gate-confidence, calibration, and diagnostic values.

By default, the workflow assumes acceleration is in `m/s^2`, gyroscope values
are in `rad/s`, and processing is at a nominal 200 Hz. It automatically
selects the lowest-scoring passing 0.10-second stationary candidate using
acceleration magnitude/variation and gyroscope activity. The candidate is a
processing aid, not proof of physical stationarity; sustained constant linear
acceleration may resemble a stationary interval.

```bash
python scripts/plot_ring_linear_acceleration.py \
  --data-root data_sample/data \
  --user user_0 \
  --action 0 \
  --dataset-id 0 \
  --output outputs/dataset_0/ring_linear_acceleration.png \
  --no-show
```

Use `--stationary-duration-s`, `--stationary-stride-s`, and
`--expected-gravity` to control the search. Supply both
`--calibration-start` and `--calibration-stop` to override automatic
selection. Strict mode errors when no candidate passes; `--provisional`
continues with the best candidate and visibly reports the failed checks.

The identity axis transform means raw Ring sensor axes, not a confirmed
physical ring mounting frame. Use `--axis-transform` with nine row-major
values only when a right-handed sensor-to-body transform is known.

An opt-in `--profile upstream_suggested` applies the unconfirmed hints from
the upstream `IMUData.scale()` implementation: acceleration divided by
`9.8`, raw gyroscope treated as radians/second, and y/z axis sign flips.
Plots and summaries label that profile as assumed. Its scaled `g` values are
not eligible for automatic m/s² stationary search, so use manual calibration
bounds with that profile. `--provisional` permits exploratory output when
stationary calibration checks fail, while preserving prominent warnings;
strict calibration is the default.

Select the estimator with `--gravity-removal-method`. The default `madgwick`
method performs IMU-only sensor fusion; tune its gradient-descent gain with
`--madgwick-beta` (default `0.1`). For acceleration-only removal, select
`--gravity-removal-method low-pass`; `--low-pass-cutoff-hz` controls its
cutoff (default `0.2` Hz). This method applies one causal second-order
Butterworth IIR low-pass SOS/biquad independently to each configured
acceleration axis and subtracts that output as the gravity contribution.

The Madgwick estimator uses a fixed nominal processing rate rather than
duplicate-rich Ring timestamps. It anchors at the calibration interval
midpoint and propagates in both directions, so its result is noncausal and
intended for offline inspection—not real-time control, navigation, or
ground-truth motion reconstruction. See
[docs/RemoveGravityInTheIMUBodyFramePlan.md](docs/RemoveGravityInTheIMUBodyFramePlan.md)
for its conventions, limitations, and acceptance criteria.

## Notebook usage

Install the notebook extra and register the active Conda environment as a
kernel:

```bash
conda activate writingring-viz
python -m ipykernel install --user \
    --name writingring-viz \
    --display-name "Python (writingring-viz)"
jupyter lab notebooks/explore_recording.ipynb
```

Choose the `Python (writingring-viz)` kernel if Jupyter does not select it
automatically. The notebook imports the installed `writingring` package and
reuses its discovery, selection, loading, validation, summary, and plotting
APIs. It does not duplicate loader or deserialization logic.

For a noninteractive execution check:

```bash
jupyter nbconvert \
    --to notebook \
    --execute notebooks/explore_recording.ipynb \
    --output explore_recording.executed.ipynb
```

## Python API example

The public package API supports the same workflow:

```python
from pathlib import Path

from writingring import (
    GravityRemovalConfig,
    discover_recordings,
    select_recording,
    load_ring,
    load_board,
    process_ring_gravity,
    plot_ring_imu,
    plot_ring_gravity_removal,
    plot_touch_trajectory,
)

recordings = discover_recordings(Path("data_sample/data"))

recording = select_recording(
    recordings,
    user="user_0",
    action="0",
    dataset_id=0,
)

ring = load_ring(recording)
board = load_board(recording)

plot_ring_imu(ring)
plot_touch_trajectory(board)

gravity_config = GravityRemovalConfig(
)
gravity_result = process_ring_gravity(ring, config=gravity_config)
plot_ring_gravity_removal(ring, gravity_result)
```

Set `calibration_start_sample` and `calibration_stop_sample` on the config to
override automatic selection with a manually verified interval.

Use `show=False` and an `output_path` for noninteractive plotting. The caller
should close returned figures after saving when they are no longer needed.

## Tests

Install the test extra and run:

```bash
python -m pytest -q
```

The latest complete acceptance run verified `174 passed`.

## Project structure

```text
src/writingring/       reusable discovery, loading, validation, summaries,
                       selection, gravity/spectral analysis, and Matplotlib
                       plotting
scripts/               command-line entry scripts
tests/                 focused unit and workflow tests
notebooks/             installed-package interactive exploration
docs/                  durable data-format documentation
vendor/WritingRing/    read-only upstream author reference
core/                  official pickle compatibility namespace
data_sample/           local raw sample data (ignored by Git)
outputs/               generated plots and JSON summaries (ignored by Git)
```

`vendor/WritingRing` is the unmodified upstream reference used to verify data
semantics. The root `core` package preserves the official class import path
needed to deserialize existing Board pickles. `src/writingring` contains the
new modular and tested implementation.

## Known limitations

- Physical units for Ring signals and Board coordinates, contact force, and
  related values are undocumented.
- Ring timestamps support an inferred microsecond interpretation, but the
  Ring writer and a confirmed upstream unit contract are unavailable.
- Ring–Board clock synchronization and offset guarantees are not confirmed;
  the project does not synchronize the streams.
- Gravity removal depends on explicit sampling-rate, unit, axis, and
  stationary-calibration assumptions. Results remain provisional unless
  those conventions are independently confirmed.
- Dataset 0 contains a backward Board timestamp boundary from chunk `6` to
  chunk `7`; all numeric chunks remain retained in filename order.
- Dense contact and force plots can overplot because the plotting API
  intentionally preserves all rows.
- Board deserialization depends on the repository-level `core` package for
  the official `core.sensel_lib.frame_data` pickle classes.
