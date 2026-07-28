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
    discover_recordings,
    select_recording,
    load_ring,
    load_board,
    plot_ring_imu,
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
```

Use `show=False` and an `output_path` for noninteractive plotting. The caller
should close returned figures after saving when they are no longer needed.

## Tests

Install the test extra and run:

```bash
python -m pytest -q
```

The latest complete acceptance run verified `128 passed`.

## Project structure

```text
src/writingring/       reusable discovery, loading, validation, summaries,
                       selection, spectral analysis, and Matplotlib plotting
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
- Dataset 0 contains a backward Board timestamp boundary from chunk `6` to
  chunk `7`; all numeric chunks remain retained in filename order.
- Dense contact and force plots can overplot because the plotting API
  intentionally preserves all rows.
- Board deserialization depends on the repository-level `core` package for
  the official `core.sensel_lib.frame_data` pickle classes.
