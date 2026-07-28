# Progress

The upstream format reference remains in
[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Completed phases

- [x] Upstream data-format inspection
- [x] Recording discovery
- [x] Ring loading and validation
- [x] Board loading and validation
- [x] Matplotlib plotting
- [x] Remaining CLI scripts
- [x] Jupyter notebook
- [x] Full workflow documentation and acceptance run
- [x] Final technical project report

## Board loading phase

### Files created or changed

- Created `src/writingring/board_loader.py`.
- Created `tests/test_board_loader.py`.
- Updated `src/writingring/__init__.py` to export the board API.
- Updated `pyproject.toml` to declare the required `compress-pickle`
  dependency.
- Updated `PROGRESS.md`.
- Kept `src/writingring/discovery.py`,
  `src/writingring/ring_loader.py`, and their existing tests unchanged.
- Did not change `docs/DATA_FORMAT.md`; implementation confirmed the existing
  board format and dataset `0` anomaly without revealing a conflict.
- Did not change `vendor/WritingRing` or `data_sample`.

### API implemented

```python
load_board(
    source: Recording | str | Path | Sequence[str | Path],
) -> BoardData
```

- A `Recording` contributes its `board_chunk_paths` exactly in discovery
  order.
- Explicit paths are also consumed exactly as supplied.
- Paths must have one dataset ID and nondecreasing numeric chunk indices.
- Timestamp values are never used to sort, split, repair, or discard chunks.
- Before deserialization, `FrameData` and `ContactData` are imported from
  `core.sensel_lib.frame_data`.
- Every chunk is loaded with the author's operation:

  ```python
  frames = compress_pickle.load(path)
  ```

- `BoardData` contains:
  - the source recording, when supplied;
  - the exact chunk paths;
  - one structured `BoardChunkReport` per successful chunk;
  - frame-level and contact-level Pandas DataFrames;
  - `BoardValidationReport`;
  - aggregate warnings.

The frame DataFrame retains every frame, including frames with no contacts.
It records dataset/chunk indices, frame index within the chunk, global frame
index, raw frame timestamp, contact count, and force-array shape/dtype
metadata.

The contact DataFrame records confirmed contact fields. Stored `y` is retained
as `y_raw`; `y_display = 1.0 - y_raw` is added only as the confirmed upstream
display transformation. No physical units are assigned.

### Validation policy

Path-level errors:

- `MissingBoardChunksError` when no chunk paths are supplied;
- `BoardPathError` for missing/non-regular files, malformed board filenames,
  mixed dataset IDs, or decreasing numeric chunk order;
- `BoardClassImportError` when the official pickle classes are unavailable.

Chunk-level failures raise a `BoardChunkError` subtype that carries the failed
chunk's structured report:

- `EmptyBoardChunkFileError` for a zero-byte file;
- `BoardDeserializationError` for invalid gzip/pickle content;
- `UnexpectedBoardContainerError` for a non-list top-level object;
- `UnexpectedBoardObjectError` for unexpected frame/contact/container types or
  nonnumeric required values;
- `MissingBoardAttributeError` for missing confirmed frame/contact fields.

Successfully deserialized `[]` is not an error. It is retained as an empty
chunk and warned. No failed, empty, duplicate, old, or anomalous chunk is
silently skipped.

The aggregate report includes:

- chunk, frame, contact, and contact-free-frame counts;
- missing, duplicate, empty, and empty-leading chunk indices;
- numeric-order first/last frame timestamps;
- within-chunk duplicate/backward timestamp checks;
- timestamp boundary reports for every adjacent supplied path;
- explicit cross-chunk backward-boundary reports;
- finite x, raw-y, and force ranges and non-finite counts;
- warnings.

### Commands run and exact results

- Focused board tests:

  ```bash
  conda run --no-capture-output -n writingring-viz \
      python -m pytest tests/test_board_loader.py -q
  ```

  Exact final result: `21 passed in 0.43s`.

- Full suite:

  ```bash
  conda run --no-capture-output -n writingring-viz \
      python -m pytest -q
  ```

  Exact final result: `59 passed in 0.24s`.

- Reusable-API real-data inspection with payload-call tracking:

  ```bash
  PYTHONPATH=src conda run --no-capture-output \
      -n writingring-viz python -
  ```

  Result: four recordings loaded. Exactly 39 board paths were passed to
  `compress_pickle.load`, in discovery order. No ring payload, including
  `ring_1`, was opened.

### Real-data results

| Dataset | Chunks | Empty | Frames | Contacts | First timestamp in numeric order | Last timestamp in numeric order | Within-chunk backward steps | Cross-chunk backward boundaries | Missing indices |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 16 | 0 | 14365 | 5800 | 1720476262355268 | 1720475983195457 | 0 | `6->7`, delta `-387876490` | none |
| 1 | 8 | 0 | 6724 | 3414 | 1720476340060284 | 1720476391394439 | 0 | none | none |
| 2 | 7 | 0 | 5631 | 3001 | 1720476425943282 | 1720476469056724 | 0 | none | none |
| 3 | 8 | 0 | 6612 | 3396 | 1720476502079573 | 1720476552609021 | 0 | none | none |

Finite contact ranges:

| Dataset | x range | raw y range | force range |
| ---: | --- | --- | --- |
| 0 | 0.3011548913043478 to 0.7125339673913044 | 0.13278245192307692 to 0.8388221153846154 | 3.0 to 280.875 |
| 1 | 0.3413722826086957 to 0.7802479619565217 | 0.1739783653846154 to 0.9256911057692307 | 3.0 to 183.75 |
| 2 | 0.34110054347826085 to 0.7793817934782609 | 0.13221153846153846 to 0.8595252403846154 | 3.0 to 269.375 |
| 3 | 0.24500679347826088 to 0.7327615489130435 | 0.1390925480769231 to 0.8790264423076923 | 3.0 to 211.875 |

Warnings:

- Dataset 0:
  - empty chunk `0` retained;
  - empty leading chunk `0` retained;
  - 8,654 frames contain no contacts;
  - backward boundary `6->7`, delta `-387876490`.
- Dataset 1:
  - empty/leading chunk `0` retained;
  - 3,383 frames contain no contacts.
- Dataset 2:
  - empty/leading chunk `0` retained;
  - 2,658 frames contain no contacts.
- Dataset 3:
  - empty/leading chunk `0` retained;
  - 3,318 frames contain no contacts.

No non-finite x, raw-y, or force values and no within-chunk backward timestamp
steps were observed.

### Dataset 0 anomaly

The returned chunk indices are exactly:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
```

All chunks, including empty chunk `0` and older chunks `7` through `15`, are
retained. The boundary report records:

```text
previous chunk: 6
next chunk: 7
previous last timestamp: 1720476305908521
next first timestamp: 1720475918032031
delta: -387876490
```

No timestamp reordering, repair, automatic split, or data removal occurred.

### Unresolved issues

- The reason every recording has an empty initial chunk remains unknown.
- Dataset `0`'s older chunk tail is retained; any later user-facing treatment
  remains a plotting/product decision, not a loader correction.
- Physical units for contact positions, force, area, axes, deltas, and force
  arrays remain undocumented.
- Ring/board synchronization guarantees remain unknown and were not
  implemented.
- The official pickle classes currently rely on the repository-level `core`
  namespace being importable.

### Next recommended phase

Implement only reusable Matplotlib plotting from the already loaded ring and
board tables. Preserve raw/inferred distinctions, show contact-free recordings
gracefully, and make dataset `0`'s validation warnings visible. Do not add
synchronization, CLI expansion, or notebook work unless separately requested.

## Matplotlib plotting phase

### Files created or changed

- Created `src/writingring/plotting.py`.
- Created `tests/test_plotting.py`.
- Updated `src/writingring/__init__.py` to export the plotting functions,
  concise-warning helper, and plotting-specific exceptions.
- Updated `pyproject.toml` to declare the Matplotlib runtime dependency.
- Created 12 real-data verification PNGs under
  `outputs/plotting_verification/`.
- Updated `PROGRESS.md`.
- Kept discovery, ring loading, board loading, and all of their existing tests
  unchanged.
- Did not update `docs/DATA_FORMAT.md`; plotting revealed no new format
  evidence or conflict.
- Did not modify `vendor/WritingRing` or `data_sample`.

### Plotting API

```python
plot_ring_imu(
    ring_data: RingData,
    *,
    time_axis: str = "inferred_time",
    output_path: str | Path | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure

plot_touch_trajectory(
    board_data: BoardData,
    *,
    output_path: str | Path | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure

plot_board_force_over_time(
    board_data: BoardData,
    *,
    time_axis: str = "frame_index",
    output_path: str | Path | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure
```

All functions return an open `Figure`, save when requested, create missing
parent output directories, and call `plt.show()` only when `show=True`.
Interactive callers retain the returned figure. Noninteractive callers own
the figure and should call `matplotlib.pyplot.close(figure)` after saving or
inspection.

Plotting-only failures use `PlottingError` subtypes:

- `UnsupportedTimeAxisError`;
- `MissingPlotColumnError`;
- `InferredTimeUnavailableError`;
- `PlotOutputError`;
- `MalformedPlotDataError`.

### Time-axis and data-order policies

- Ring `sample_index` uses the loaded DataFrame index.
- Ring `inferred_time` uses only the existing
  `relative_time_inferred_s` column and visibly labels its microsecond basis
  as inferred and unconfirmed. Absence of that column raises
  `InferredTimeUnavailableError`; no 200 Hz or other synthetic axis is
  substituted.
- Board force `frame_index` uses `global_frame_index`.
- Board force `raw_timestamp` uses `frame_timestamp_raw` in existing contact
  row order.
- Board and Ring timestamps are not synchronized or presented as
  synchronized.
- No plotting function sorts contacts, reorders chunks, repairs timestamps,
  splits an anomaly, drops rows, or modifies an input DataFrame.

### Warning-display policy

Warnings are bounded, deduplicated, wrapped, and displayed in reserved figure
footer space. The structured summary includes, when present:

- duplicate Ring timestamp steps;
- the unconfirmed inferred Ring time interpretation;
- empty Board chunk indices;
- contact-free frame counts;
- cross-chunk backward timestamp boundaries and deltas.

Dataset `0` touch and force figures visibly state:

```text
Backward timestamp boundary 6->7 (delta=-387876490.0); order retained
```

### Commands run and exact results

- Initial focused-test attempt:

  ```bash
  conda run --no-capture-output -n writingring-viz \
      python -m pytest tests/test_plotting.py -q
  ```

  Result: collection error because Matplotlib was not installed:
  `ModuleNotFoundError: No module named 'matplotlib'`.

- Installed the declared required dependency:

  ```bash
  conda install -n writingring-viz matplotlib -y
  ```

  Result: completed successfully; Matplotlib `3.11.1` was installed.

- Focused plotting tests after installation:

  ```bash
  conda run --no-capture-output -n writingring-viz \
      python -m pytest tests/test_plotting.py -q
  ```

  Result: `15 passed in 0.78s`.

- A warning-layout refinement initially had one extra positional argument to
  `Figure.text`; its focused run reported `11 failed, 4 passed in 0.83s`.
  The typo was corrected immediately. Exact final focused result:
  `15 passed in 1.02s`.

- Real-data image generation used the reusable library API under the Agg
  backend through a temporary `conda run ... python -c` inspection. Payload
  calls were tracked. Result:

  - opened exactly `0_ring_0.bin` through `3_ring_0.bin`;
  - opened all 39 board chunks in the exact discovery order;
  - opened no `ring_1` payload;
  - asserted plotted touch/raw-timestamp offsets equal the contact-table row
    order;
  - asserted Dataset `0` chunks `7..15` all remain represented;
  - asserted the Dataset `0` `6->7` warning and all empty-chunk warnings are
    present in figure text.

- Full suite:

  ```bash
  conda run --no-capture-output -n writingring-viz \
      python -m pytest -q
  ```

  Exact result: `74 passed in 0.70s`.

- Read-only tree hashes before and after plotting were identical:

  ```text
  data_sample:        9cdc8a0c34bc955270ab282220b7fbd9e64edfe3a38bbe58601b2d297e0e9ff8
  vendor/WritingRing: 315cc77dee0a30cb23750bb0a61804b165eb388080e6d659b10a867902dd5cd8
  ```

### Real-data outputs

| Dataset | Numeric chunk indices | Contacts plotted | Empty chunks disclosed | Backward boundaries disclosed |
| ---: | --- | ---: | --- | --- |
| 0 | `0..15` | 5800 | `0` | `6->7`, delta `-387876490.0` |
| 1 | `0..7` | 3414 | `0` | none |
| 2 | `0..6` | 3001 | `0` | none |
| 3 | `0..7` | 3396 | `0` | none |

Every generated file exists and has nonzero size:

| Output | Bytes |
| --- | ---: |
| `dataset_0_ring_imu.png` | 147703 |
| `dataset_0_touch_trajectory.png` | 219289 |
| `dataset_0_board_force.png` | 101862 |
| `dataset_1_ring_imu.png` | 175315 |
| `dataset_1_touch_trajectory.png` | 211472 |
| `dataset_1_board_force.png` | 115710 |
| `dataset_2_ring_imu.png` | 143163 |
| `dataset_2_touch_trajectory.png` | 208214 |
| `dataset_2_board_force.png` | 97866 |
| `dataset_3_ring_imu.png` | 169497 |
| `dataset_3_touch_trajectory.png` | 218722 |
| `dataset_3_board_force.png` | 111864 |

Visual inspection of Dataset `0`'s touch and force PNGs confirmed the warning
footer is readable. The raw-timestamp force figure retains two separated
timestamp regions, visibly reflecting the unrepaired `6->7` jump.

### Visual and data limitations

- Dense touch trajectories and force samples can overplot; the reusable plots
  intentionally preserve every row rather than aggregate or discard data.
- Dataset `0`'s old tail is shown in the same figures because no split or
  anomaly-removal policy is authorized.
- Raw board timestamps use large stored values, so Matplotlib applies its
  standard axis offset notation.
- Ring time in seconds remains an inferred interpretation, not a confirmed
  upstream unit contract.
- Physical signal/force units remain undocumented and are labeled as raw
  values.
- No synchronization guarantee exists between Board and Ring clocks.

### Unresolved issues

- A later product/workflow phase must decide how users should interact with
  Dataset `0`'s retained older tail; plotting does not make that decision.
- The physical units and independent Ring timestamp unit contract remain
  unknown.
- Highly dense recordings may eventually benefit from an explicitly
  authorized visualization/downsampling policy, but none was introduced here.

### Next recommended phase

Implement only the next explicitly requested workflow phase, likely the
remaining inspection CLI surface. Reuse these plotting APIs and preserve their
raw/inferred distinctions. Do not add synchronization or a notebook until
those phases are separately requested.

## Command-line inspection and plotting phase

### Files created or modified

- Created `src/writingring/selection.py` with exact recording selection and
  typed selection errors.
- Created `src/writingring/inspection.py` with the common structured summary,
  readable text formatting, strict JSON conversion, and JSON writing.
- Created `scripts/inspect_recording.py`.
- Created `scripts/plot_recording.py`.
- Created `tests/test_selection.py`.
- Created `tests/test_cli.py`.
- Updated `src/writingring/__init__.py` to export the selection and inspection
  helpers.
- Updated `pyproject.toml` only as needed to include the official repository
  `core` and `core.sensel_lib` namespaces in an editable installation.
- Updated `PROGRESS.md`.
- Generated verification outputs under `outputs/cli_verification/dataset_0`
  through `dataset_3`.
- Kept `src/writingring/discovery.py`, `ring_loader.py`, `board_loader.py`,
  and `plotting.py` unchanged.
- Kept all preexisting scripts and tests unchanged.
- Did not update `docs/DATA_FORMAT.md`; the CLI phase revealed no new format
  evidence.
- Did not modify `vendor/WritingRing` or `data_sample`.

### Reusable selection and summary APIs

```python
select_recording(
    recordings: Iterable[Recording],
    *,
    user: str,
    action: str,
    dataset_id: int,
) -> Recording
```

Selection is exact and does not change discovery behavior. Empty user/action
values and negative or non-integer dataset IDs raise
`InvalidRecordingSelectorError`; zero matches raise
`RecordingNotFoundError`; unexpected duplicate identities raise
`MultipleRecordingsMatchedError`.

```python
build_recording_summary(
    recording: Recording,
    ring_data: RingData,
    board_data: BoardData,
) -> dict[str, Any]
```

Both CLIs use this single summary source. `to_jsonable` recursively converts
dataclasses, Paths, enums, NumPy arrays/scalars, mappings, and sequences.
Non-finite floating-point values become explicit JSON strings (`NaN`,
`Infinity`, or `-Infinity`) so `write_json` can enforce `allow_nan=False`.

### CLI usage

Readable inspection:

```bash
python scripts/inspect_recording.py \
    --data-root data_sample/data \
    --user user_0 \
    --action 0 \
    --dataset-id 0
```

Add `--json-output PATH` to write the same structured information as strict
JSON.

Plot generation:

```bash
python scripts/plot_recording.py \
    --data-root data_sample/data \
    --user user_0 \
    --action 0 \
    --dataset-id 0 \
    --output-dir outputs/cli_verification/dataset_0 \
    --no-show
```

Plotting options:

- `--ring-time-axis`: `sample_index` or `inferred_time`; default
  `inferred_time`.
- `--board-time-axis`: `frame_index` or `raw_timestamp`; default
  `frame_index`.
- `--show` / `--no-show`; default is `--show`.

The choices and defaults are imported directly from
`writingring.plotting`. In `--no-show` mode the script selects Agg before
importing pyplot-dependent project code and closes every returned figure.
Interactive mode preserves the plotting API's existing show behavior and does
not close figures prematurely.

The plotting CLI creates the requested output directory and writes:

```text
ring_imu.png
touch_trajectory.png
board_force_time.png
summary.json
```

Reusing an explicitly selected output directory may replace only those four
expected paths. Unrelated files are left untouched; this policy is documented
in CLI help and `summary.json`.

### Error handling

Both CLIs use concise `error: ...` messages and return status `2` for expected
discovery, selection, Ring-loading, Board-loading, plotting, output, and JSON
failures. Standard argparse errors handle malformed numeric selectors with
status `2`. Expected failures do not print tracebacks.

### Editable installation

The required command was run:

```bash
conda run --no-capture-output -n writingring-viz \
    python -m pip install -e .
```

The first sandboxed attempt could not download isolated build metadata and
could not write Conda site-packages. The approved external rerun succeeded.

The first real inspection then revealed that the initial `src`-only package
configuration did not expose the official pickle classes when a script was
launched from `scripts/`. It failed clearly with:

```text
error: official pickle classes are not importable from
core.sensel_lib.frame_data
```

The minimum packaging fix explicitly included `writingring`, `core`, and
`core.sensel_lib`. Reinstalling editably succeeded. From `/tmp`, without a
`PYTHONPATH` setting, both imports resolve to the project:

```text
/home/ted/project/writingring-viz/src/writingring/__init__.py
/home/ted/project/writingring-viz/core/sensel_lib/frame_data.py
```

No loader code was changed to obtain this result.

### Tests and exact results

Focused command:

```bash
conda run --no-capture-output -n writingring-viz \
    python -m pytest tests/test_selection.py tests/test_cli.py -q
```

Initial implemented result: `15 passed in 1.43s`.

After adding explicit nonexistent-root and process-level malformed-selector
coverage, exact final result: `17 passed in 1.46s`.

Full suite:

```bash
conda run --no-capture-output -n writingring-viz \
    python -m pytest -q
```

Exact result: `91 passed in 1.64s`.

Focused tests cover:

- exact, missing, duplicate, and malformed selection;
- inspection text and strict JSON;
- nonexistent roots and loader failure;
- plotting output generation and valid summary JSON;
- Agg/no-show behavior and figure closure;
- exact Ring and Board time-axis forwarding;
- Dataset `0`-style `6->7` warnings;
- concise plotting failures and output-directory failures;
- primary `ring_0`-only payload access;
- complete numeric Board path order with no dropping.

The existing listing command was rerun after installation and still reported
the same four recordings:

```bash
conda run --no-capture-output -n writingring-viz \
    python scripts/list_recordings.py --data-root data_sample/data
```

### Real-data command results

The requested Dataset `0` inspection succeeded after the packaging fix:

```bash
conda run --no-capture-output -n writingring-viz \
    python scripts/inspect_recording.py \
    --data-root data_sample/data \
    --user user_0 \
    --action 0 \
    --dataset-id 0
```

It reported:

- Ring shape `(10247, 7)` and 10,247 samples;
- 7,148 duplicate and zero backward Ring timestamp steps;
- 51.003585 inferred seconds and 200.88783955088647 inferred Hz;
- 16 Board chunks with indices `0..15`;
- empty chunk `0`;
- 14,365 frames, 5,800 contacts, and 8,654 contact-free frames;
- Board boundary `6->7`, delta `-387876490.0`;
- no missing Board chunk indices.

The plotting command was run separately for datasets `0`, `1`, `2`, and `3`
with `--no-show`. All returned status `0` and printed their four output paths.

A tracked reusable-CLI verification then asserted:

- only `0_ring_0.bin` through `3_ring_0.bin` were passed to `numpy.fromfile`;
- no `ring_1` payload was opened;
- all 39 Board paths were passed to `compress_pickle.load` in exact discovery
  order;
- summary chunk sequences are `0..15`, `0..7`, `0..6`, and `0..7`;
- Dataset `0` JSON contains the `6->7` boundary;
- all JSON files parse;
- every expected output is nonempty;
- every summary explicitly states that Ring and Board are not assumed
  synchronized.

Visual inspection of
`outputs/cli_verification/dataset_0/touch_trajectory.png` confirmed its
warning footer visibly includes:

```text
Backward timestamp boundary 6->7 (delta=-387876490.0); order retained
```

### Generated verification outputs

| Dataset | File | Bytes |
| ---: | --- | ---: |
| 0 | `ring_imu.png` | 147703 |
| 0 | `touch_trajectory.png` | 219289 |
| 0 | `board_force_time.png` | 128803 |
| 0 | `summary.json` | 5984 |
| 1 | `ring_imu.png` | 175315 |
| 1 | `touch_trajectory.png` | 211472 |
| 1 | `board_force_time.png` | 115601 |
| 1 | `summary.json` | 4613 |
| 2 | `ring_imu.png` | 143163 |
| 2 | `touch_trajectory.png` | 208214 |
| 2 | `board_force_time.png` | 98323 |
| 2 | `summary.json` | 4493 |
| 3 | `ring_imu.png` | 169497 |
| 3 | `touch_trajectory.png` | 218722 |
| 3 | `board_force_time.png` | 111629 |
| 3 | `summary.json` | 4605 |

The read-only tree hashes remained identical before and after the CLI runs:

```text
data_sample:        9cdc8a0c34bc955270ab282220b7fbd9e64edfe3a38bbe58601b2d297e0e9ff8
vendor/WritingRing: 315cc77dee0a30cb23750bb0a61804b165eb388080e6d659b10a867902dd5cd8
```

### Unresolved issues

- Dataset `0`'s older Board tail remains retained and disclosed; the CLI does
  not choose a product policy for it.
- Physical units remain undocumented.
- The Ring timestamp microsecond interpretation remains inferred rather than
  a confirmed upstream contract.
- Ring–Board synchronization guarantees remain unknown and no synchronization
  was performed.
- Dense figures retain all points and may overplot, consistent with the
  plotting API.

### Next recommended phase

This phase's recommended Jupyter notebook follow-up is completed in the
subsequent section.

## Jupyter notebook phase

### Files created or changed

- Created `notebooks/explore_recording.ipynb`.
- Created the notebook verification figures under
  `outputs/notebook_exploration/`.
- Updated `PROGRESS.md`.
- Did not change discovery, selection, loader, inspection-summary, plotting,
  or CLI code.
- Did not change `docs/DATA_FORMAT.md`; notebook execution revealed no new
  format evidence or conflict.

### Notebook sections

The notebook:

1. imports only public APIs from the editably installed `writingring`
   package;
2. resolves the project data and notebook-output directories without setting
   `PYTHONPATH`;
3. discovers all recordings and displays a concise Pandas discovery table;
4. selects `user_0`, action `0`, dataset `0` through `select_recording`;
5. loads the primary Ring payload and all Board chunks through `load_ring`
   and `load_board`;
6. builds and prints the shared `build_recording_summary` result;
7. previews the Ring, Board-frame, and Board-contact tables with `.head()`;
8. displays structured Ring validation, channel statistics, Board validation,
   and chunk reports;
9. creates and saves Ring inferred-time and sample-index plots, the touch
   trajectory, and Board-force plots using frame index and raw timestamp;
10. demonstrates independent Board numeric-chunk filtering and Ring
    sample-index slicing without modifying the loaded data;
11. runs explicit output and anomaly checks; and
12. ends with the known timestamp, empty-chunk, physical-unit, and
    synchronization limitations.

The Board and Ring filtering examples explicitly state that separate filters
do not establish synchronization.

### Structure and execution verification

`nbformat`, `nbconvert`, and `ipykernel` were installed into the existing
`writingring-viz` Conda environment because they were not initially present.
No project dependency or packaging metadata was changed for this
environment-only verification.

Programmatic validation used `nbformat.validate`, parsed every code cell with
`ast.parse`, and checked the required public API calls and limitation text.
Exact result:

```text
valid notebook: 30 cells, 16 code cells, 14 markdown cells
```

The editable import check, with no `PYTHONPATH` setting, resolved to:

```text
/home/ted/project/writingring-viz/src/writingring/__init__.py
```

Jupyter configuration, runtime files, IPython state, and Matplotlib caches
were directed to `/tmp` because the home directory is read-only. The final
execution command was:

```bash
env \
  JUPYTER_CONFIG_DIR=/tmp/writingring-jupyter-config \
  JUPYTER_DATA_DIR=/tmp/writingring-jupyter-data \
  JUPYTER_RUNTIME_DIR=/tmp/writingring-jupyter-runtime \
  IPYTHONDIR=/tmp/writingring-ipython \
  MPLCONFIGDIR=/tmp/writingring-matplotlib \
  XDG_CACHE_HOME=/tmp/writingring-cache \
  MPLBACKEND=Agg \
  conda run --no-capture-output -n writingring-viz \
  jupyter nbconvert \
    --to notebook \
    --execute notebooks/explore_recording.ipynb \
    --output explore_recording.executed.ipynb \
    --output-dir /tmp/writingring-notebook-execution \
    --ExecutePreprocessor.timeout=300
```

The sandboxed first attempt could not create Jupyter's localhost kernel
socket. The approved execution completed successfully and wrote:

```text
/tmp/writingring-notebook-execution/explore_recording.executed.ipynb
```

The executed artifact passed schema validation with all 16 code cells
executed and zero error outputs. Its checks confirmed:

- primary payload `0_ring_0.bin` was loaded; Ring 1 remained metadata only;
- all 16 Board chunks remained in numeric order with indices `0..15`;
- the visible Dataset `0` warning was
  `backward timestamp boundary 6->7: delta=-387876490.0`;
- no timestamp repair or Ring–Board synchronization was applied; and
- every expected saved figure existed and was nonempty.

Two pre-final execution runs exposed notebook-only naming mistakes
(`frame_count` versus summary wording and the named `sample_index` index
versus a column). The notebook was corrected to use the existing public data
representations; no reusable API semantics changed.

### Generated output files

| File | Bytes |
| --- | ---: |
| `outputs/notebook_exploration/ring_imu_inferred_time.png` | 147703 |
| `outputs/notebook_exploration/ring_imu_sample_index.png` | 157730 |
| `outputs/notebook_exploration/touch_trajectory.png` | 219289 |
| `outputs/notebook_exploration/board_force_frame_index.png` | 128803 |
| `outputs/notebook_exploration/board_force_raw_timestamp.png` | 101862 |

The read-only tree hashes were identical before and after execution:

```text
data_sample:        9cdc8a0c34bc955270ab282220b7fbd9e64edfe3a38bbe58601b2d297e0e9ff8
vendor/WritingRing: 315cc77dee0a30cb23750bb0a61804b165eb388080e6d659b10a867902dd5cd8
```

### Full test result

Command:

```bash
conda run --no-capture-output -n writingring-viz \
    python -m pytest -q
```

Exact result:

```text
91 passed in 1.73s
```

### Limitations

- Dataset `0`'s `6->7` backward Board timestamp boundary remains retained and
  disclosed; no timestamp was repaired and no chunk was dropped, reordered,
  or split.
- Empty leading Board chunk `0` remains represented and disclosed.
- The Ring microsecond interpretation remains inferred, not a confirmed
  upstream timestamp contract.
- Ring signal and Board coordinate/force physical units remain undocumented.
- Ring–Board synchronization remains unconfirmed and was not implemented or
  implied by the independent filtering examples.
- The notebook intentionally uses concise previews rather than embedding
  complete large DataFrames.

### Next recommended phase

This phase's recommended final documentation and acceptance follow-up is
completed in the subsequent section.

## Final workflow documentation and acceptance phase

### Files created or changed

- Created project-root `README.md`.
- Created project-root `.gitignore`.
- Added a minimal `notebook` optional dependency group to `pyproject.toml`;
  the existing runtime dependencies and `test` extra remain unchanged.
- Updated `PROGRESS.md`.
- Generated final acceptance artifacts under
  `outputs/final_acceptance/`.
- Did not change discovery, selection, Ring loading, Board loading,
  inspection summaries, plotting, CLI, notebook, or data-format semantics.
- Did not change `docs/DATA_FORMAT.md`; final acceptance revealed no new
  format evidence or conflict.

### README sections

The README now documents:

- project purpose and supported discovery/loading/validation/plotting
  workflows;
- the user/action/dataset-ID hierarchy and conservative data rules;
- primary `ring_0` handling and metadata-only `ring_1`;
- numeric Board ordering and official pickle-class dependency;
- environment, editable installation, testing, and notebook setup;
- local dataset placement and generated-output policy;
- listing, inspection, and plotting CLI examples;
- Ring and Board time-axis choices;
- installed-package notebook usage;
- a concise public Python API example;
- the final project structure and the distinction between
  `vendor/WritingRing`, root `core`, and `src/writingring`; and
- all remaining timestamp, unit, synchronization, anomaly, and overplotting
  limitations.

### `.gitignore` policy

The ignore policy covers:

```text
__pycache__/
*.py[cod]
.pytest_cache/
.ipynb_checkpoints/
outputs/
*.egg-info/
build/
dist/
data_sample/
downloads/
*.zip
*.tar
*.tar.gz
*.tgz
```

`data_sample/` is an untracked 71 MB local raw-data tree and `downloads/`
contains an untracked 9.2 GB source archive, so neither is intended for Git.
Generated future `outputs/` are ignored. Previously tracked
`outputs/plotting_verification/` files were not removed or untracked.
Source, tests, docs, the notebook, `vendor/WritingRing`, and root `core` are
not ignored.

### Packaging and installation

The existing `test` extra remains:

```text
pytest>=8
```

The new minimal `notebook` extra contains:

```text
ipykernel>=6
jupyterlab>=4
nbconvert>=7
nbformat>=5
```

Editable installation command:

```bash
conda run --no-capture-output -n writingring-viz \
    python -m pip install -e .
```

Result: the editable wheel built and `writingring 0.1.0` installed
successfully. From `/tmp`, with no `PYTHONPATH` setting, imports resolved to:

```text
/home/ted/project/writingring-viz/src/writingring/__init__.py
/home/ted/project/writingring-viz/core/sensel_lib/frame_data.py
```

### Exact acceptance commands and results

Recording listing:

```bash
conda run --no-capture-output -n writingring-viz \
    python scripts/list_recordings.py \
    --data-root data_sample/data
```

Result: four recordings were discovered for `user_0`, action `0`, with
dataset IDs `0`, `1`, `2`, and `3`. Their Board chunk counts and ranges were
`16 (0..15)`, `8 (0..7)`, `7 (0..6)`, and `8 (0..7)`.

Dataset `0` inspection:

```bash
conda run --no-capture-output -n writingring-viz \
    python scripts/inspect_recording.py \
    --data-root data_sample/data \
    --user user_0 \
    --action 0 \
    --dataset-id 0 \
    --json-output outputs/final_acceptance/dataset_0_inspection.json
```

Result: status `0`; strict JSON was written. The report retains Board chunks
`0..15`, empty chunk `0`, 14,365 frames, 5,800 contacts, and the backward
boundary `6->7` with delta `-387876490.0`.

Plotting acceptance commands:

```bash
conda run --no-capture-output -n writingring-viz \
    python scripts/plot_recording.py \
    --data-root data_sample/data \
    --user user_0 --action 0 --dataset-id 0 \
    --output-dir outputs/final_acceptance/dataset_0 --no-show

conda run --no-capture-output -n writingring-viz \
    python scripts/plot_recording.py \
    --data-root data_sample/data \
    --user user_0 --action 0 --dataset-id 1 \
    --output-dir outputs/final_acceptance/dataset_1 --no-show

conda run --no-capture-output -n writingring-viz \
    python scripts/plot_recording.py \
    --data-root data_sample/data \
    --user user_0 --action 0 --dataset-id 2 \
    --output-dir outputs/final_acceptance/dataset_2 --no-show

conda run --no-capture-output -n writingring-viz \
    python scripts/plot_recording.py \
    --data-root data_sample/data \
    --user user_0 --action 0 --dataset-id 3 \
    --output-dir outputs/final_acceptance/dataset_3 --no-show
```

Result: all four commands returned status `0` and produced their expected
three PNGs and one strict JSON summary.

Final notebook execution:

```bash
env \
  JUPYTER_CONFIG_DIR=/tmp/writingring-jupyter-config \
  JUPYTER_DATA_DIR=/tmp/writingring-jupyter-data \
  JUPYTER_RUNTIME_DIR=/tmp/writingring-jupyter-runtime \
  IPYTHONDIR=/tmp/writingring-ipython \
  MPLCONFIGDIR=/tmp/writingring-matplotlib \
  XDG_CACHE_HOME=/tmp/writingring-cache \
  MPLBACKEND=Agg \
  conda run --no-capture-output -n writingring-viz \
  jupyter nbconvert \
    --to notebook \
    --execute notebooks/explore_recording.ipynb \
    --output explore_recording.executed.ipynb \
    --output-dir outputs/final_acceptance \
    --ExecutePreprocessor.timeout=300
```

Result: status `0`;
`outputs/final_acceptance/explore_recording.executed.ipynb` contains all 16
executed code cells and zero error outputs. Its output preserves all 16
Dataset `0` chunks, displays the `6->7` warning, and explicitly states that
no timestamp repair or synchronization was applied.

Final test command:

```bash
conda run --no-capture-output -n writingring-viz \
    python -m pytest -q
```

Exact result:

```text
91 passed in 1.67s
```

### Generated final acceptance artifacts

| Artifact | Bytes |
| --- | ---: |
| `outputs/final_acceptance/dataset_0_inspection.json` | 5354 |
| `outputs/final_acceptance/dataset_0/ring_imu.png` | 147703 |
| `outputs/final_acceptance/dataset_0/touch_trajectory.png` | 219289 |
| `outputs/final_acceptance/dataset_0/board_force_time.png` | 128803 |
| `outputs/final_acceptance/dataset_0/summary.json` | 5984 |
| `outputs/final_acceptance/dataset_1/ring_imu.png` | 175315 |
| `outputs/final_acceptance/dataset_1/touch_trajectory.png` | 211472 |
| `outputs/final_acceptance/dataset_1/board_force_time.png` | 115601 |
| `outputs/final_acceptance/dataset_1/summary.json` | 4613 |
| `outputs/final_acceptance/dataset_2/ring_imu.png` | 143163 |
| `outputs/final_acceptance/dataset_2/touch_trajectory.png` | 208214 |
| `outputs/final_acceptance/dataset_2/board_force_time.png` | 98323 |
| `outputs/final_acceptance/dataset_2/summary.json` | 4493 |
| `outputs/final_acceptance/dataset_3/ring_imu.png` | 169497 |
| `outputs/final_acceptance/dataset_3/touch_trajectory.png` | 218722 |
| `outputs/final_acceptance/dataset_3/board_force_time.png` | 111629 |
| `outputs/final_acceptance/dataset_3/summary.json` | 4605 |
| `outputs/final_acceptance/explore_recording.executed.ipynb` | 91156 |

All five JSON files parsed as strict JSON. Every listed artifact exists and
has nonzero size.

### Acceptance invariants

A reusable-API acceptance run tracked every payload call while loading all
four recordings:

```text
recordings=4 ring_payloads=4 ring_1_payloads=0 board_chunks=39
dataset_0_chunks=0..15 boundary=6->7 delta=-387876490.0 order_retained=true
```

This verifies:

- only `0_ring_0.bin` through `3_ring_0.bin` were opened as Ring payloads;
- no `ring_1` payload was opened;
- all 39 Board paths were deserialized in exact discovery order;
- Dataset `0` retains chunks `0` through `15`;
- no timestamp was repaired and no Board chunk was dropped, reordered,
  split, or sorted by timestamp;
- JSON summaries state that Ring and Board are not assumed synchronized;
- no physical unit was inferred; and
- the notebook and CLI outputs retain the known warnings and limitations.

The read-only tree hashes were identical before and after final acceptance:

```text
data_sample:        9cdc8a0c34bc955270ab282220b7fbd9e64edfe3a38bbe58601b2d297e0e9ff8
vendor/WritingRing: 315cc77dee0a30cb23750bb0a61804b165eb388080e6d659b10a867902dd5cd8
```

### Remaining research and data limitations

- Ring timestamp microseconds remain an inferred interpretation rather than
  a confirmed upstream contract.
- Ring acceleration/gyro and Board coordinate/force physical units remain
  undocumented.
- Ring–Board synchronization and clock-offset guarantees remain unknown; no
  synchronization was implemented.
- Dataset `0`'s older Board tail and the cause of the `6->7` jump remain
  unexplained; the complete numeric sequence is retained.
- The cause of every recording's empty leading Board chunk remains unknown.
- Dense plots may overplot because all data rows are preserved.
- Board deserialization continues to depend on the repository-level `core`
  compatibility namespace.

### Final status

All eight planned project phases are complete. Documentation, editable
installation, CLI inspection and plotting, notebook execution, strict JSON
validation, payload-order tracking, source-integrity hashing, and the full
test suite passed. No synchronization or machine-learning phase was started.

## Final technical project report documentation

### Documentation-only files changed

- Created `docs/PROJECT_REPORT.md`.
- Updated `PROGRESS.md`.
- Did not change `README.md`, `docs/DATA_FORMAT.md`, `pyproject.toml`,
  implementation modules, CLI scripts, tests, notebook behavior,
  `vendor/WritingRing`, or `data_sample`.

### Report path and sections

The comprehensive report is:

```text
docs/PROJECT_REPORT.md
```

It contains all 18 required sections:

1. executive summary;
2. original upstream implementation;
3. confirmed dataset organization;
4. project goals and design principles;
5. final project architecture;
6. module-by-module explanation;
7. important data structures;
8. function call and dependency flow;
9. end-to-end workflows;
10. data transformations;
11. validation and error handling;
12. Dataset `0` case study;
13. testing strategy;
14. packaging and environment;
15. current implementation effect;
16. known limitations and non-goals;
17. future extension points; and
18. usage quick reference.

The report contains three Mermaid diagrams:

- the project architecture and layer relationships;
- the raw-file-to-interface data flow; and
- the exact major function-call flow used by the CLI tools.

### Direct evidence reviewed

The report was verified directly against:

- `vendor/WritingRing/ring_plot.py`;
- `vendor/WritingRing/board_plot.py`;
- `vendor/WritingRing/core/imu_data.py`;
- `vendor/WritingRing/core/window.py`;
- `vendor/WritingRing/core/sensel_lib/frame_data.py`;
- `vendor/WritingRing/core/sensel_lib/board.py`;
- every module under `src/writingring`;
- all three scripts under `scripts`;
- every test file under `tests`;
- `pyproject.toml`;
- all 30 cells in `notebooks/explore_recording.ipynb`; and
- the previously executed notebook artifact under
  `outputs/final_acceptance/`.

The audit used executable source as the primary implementation evidence
rather than relying only on earlier progress notes. It identified and
documented the source-grounded detail that `BoardData` retains frame-table
force-array shape/dtype metadata but does not retain the dense force arrays
themselves.

### Verification commands and results

Public API verification inspected `writingring.__all__` and runtime
signatures:

```bash
env \
  MPLCONFIGDIR=/tmp/writingring-matplotlib \
  XDG_CACHE_HOME=/tmp/writingring-cache \
  conda run --no-capture-output -n writingring-viz \
  python -c "<public API inspection>"
```

Result: all 50 exported names resolved. Every class and function named as a
public package API in the report exists with the documented signature.

CLI parser verification:

```bash
conda run --no-capture-output -n writingring-viz \
    python scripts/list_recordings.py --help

conda run --no-capture-output -n writingring-viz \
    python scripts/inspect_recording.py --help

conda run --no-capture-output -n writingring-viz \
    python scripts/plot_recording.py --help
```

Result: each command returned help successfully. The documented required
arguments, optional inspection JSON path, plotting time-axis choices, and
`--show`/`--no-show` behavior match the current parsers.

Root/upstream compatibility comparison:

```bash
diff -qr core vendor/WritingRing/core
```

Exact difference:

```text
Only in core/sensel_lib: __pycache__
```

Notebook verification read:

```text
outputs/final_acceptance/explore_recording.executed.ipynb
```

Result: the file exists, is nonempty, contains all 16 executed code cells,
and has zero error outputs.

Report structure/name/diagram validation checked:

- all numbered sections `1` through `18`;
- all required source paths;
- every named primary public API against `writingring.__all__`;
- upstream class/function declarations against the author files;
- exactly three Mermaid blocks; and
- every Mermaid edge against a node defined in its diagram.

Exact result:

```text
report verification passed: 18 sections, 3 Mermaid diagrams, all required paths and named APIs present
```

Full test command:

```bash
conda run --no-capture-output -n writingring-viz \
    python -m pytest -q
```

Exact result:

```text
91 passed in 2.11s
```

The source-tree hashes remained:

```text
data_sample:        9cdc8a0c34bc955270ab282220b7fbd9e64edfe3a38bbe58601b2d297e0e9ff8
vendor/WritingRing: 315cc77dee0a30cb23750bb0a61804b165eb388080e6d659b10a867902dd5cd8
```

### Statements that could not be verified

The report explicitly lists, rather than resolves, the following unavailable
facts:

- an independently documented Ring byte order beyond NumPy's native-endian
  behavior;
- the Ring writer clock source, guaranteed timestamp unit, and nominal
  sampling rate;
- physical units for Ring signals and Board coordinate/force-related values;
- the purpose of `ring_1` and why upstream plotting selects `ring_0`;
- action-ID meanings;
- a Ring–Board shared clock, offset, or synchronization guarantee;
- the cause of empty initial Board chunks;
- the cause of Dataset `0`'s older tail and `6->7` backward boundary;
- the reason for the `FrameData` timestamp-docstring/writer discrepancy; and
- the reason `Board.FPS` differs from observed sample timing.

No other report statement remained unverified after the source, path, API,
CLI, notebook, diagram, and test checks.

### Final documentation status

The project report is complete and serves both as a new-developer
architecture guide and as a technical appendix for later research. This
phase added no implementation feature and did not begin synchronization,
segmentation, force-array analysis, or machine-learning work.

## Vendor windowing investigation

- [x] Completed the source- and sample-backed investigation requested in
  `docs/WINDOWING_ANALYSIS_TASK.md`.
- Created the complete report at
  `docs/VENDOR_WINDOWING_REPORT.md`.
- Analyzed `vendor/WritingRing/board_plot.py`, `ring_plot.py`,
  `core/window.py`, `core/imu_data.py`, and the relevant Sensel Board,
  frame-data, binding, and register-map modules.
- Main finding: the vendor defines a list-backed bounded FIFO rolling-buffer
  class, but no vendor call site imports or instantiates it. The actual Ring
  and Board plotting paths do not implement true fixed-length sliding-window
  segmentation. Board files are persistence chunks, and adjacent marker
  intervals are variable-duration timestamp filters.
- Unresolved questions include the Ring `0`/`1` meanings and Ring clock
  contract, a guaranteed Ring–Board synchronization relationship, the cause
  of empty initial Board chunks and dataset `0`'s older tail, and whether the
  unpopulated `Board.frames` history was an omission or intentional.
- No implementation, vendor source, or sample-data file was changed.

## Ring acceleration windowed PSD phase

### Files added or changed

- Added `src/writingring/spectral.py` with reusable numerical windowing,
  one-sided PSD calculation, aggregation, validation, and combined plotting.
- Added `scripts/plot_ring_accel_spectrum.py` as a thin discovery, selection,
  Ring-loading, plotting, and reporting adapter.
- Added `tests/test_spectral.py` and `tests/test_spectral_cli.py`.
- Updated `src/writingring/__init__.py` with the stable spectral API.
- Updated `README.md` with the PSD workflow, interpretation, CLI usage, and
  non-goals.
- Updated `PROGRESS.md`.
- Generated the sample verification image at
  `outputs/dataset_0/ring_accel_psd_overlay.png`.
- Kept all existing discovery, selection, Ring-loading, and plotting behavior
  unchanged. Did not modify `vendor/WritingRing`, `data_sample`, Board
  loading, marker handling, synchronization, timestamp handling, or
  resampling behavior.

### Public API and implementation decisions

The exported API is:

```python
WindowedPSD
SpectralAnalysisError
InsufficientSpectralSamplesError
InvalidFrequencyRangeError
UnsupportedAggregateError
SpectralPlotError
compute_windowed_psd
plot_ring_acceleration_psd_overlay
```

- `compute_windowed_psd` accepts one finite one-dimensional signal, calculates
  `round(sampling_rate_hz * window_seconds)` samples per window and
  `round(window_size * (1 - overlap_ratio))` samples per hop, and emits only
  complete windows at the specified starts. It drops the incomplete tail
  without padding.
- Each segment is copied to float64, mean-detrended, multiplied by
  `numpy.hanning`, transformed with `numpy.fft.rfft`, and normalized by
  `sampling_rate_hz * sum(hann**2)`.
- One-sided scaling doubles interior bins only: `1:-1` for even lengths and
  `1:` for odd lengths, preserving DC and the even-length Nyquist bin.
- Mean and median aggregation are calculated independently at each frequency
  bin. Result arrays are read-only, and neither the input array nor the loaded
  Ring DataFrame is modified.
- Plotting analyzes exactly `acc_x`, `acc_y`, and `acc_z`. It creates one
  three-row, one-column figure with shared frequency axes, shared logarithmic
  y limits, gray complete-window curves, and a blue selected-aggregate curve.
- Exact PSD zeros are replaced only in temporary display arrays with a floor
  derived from the smallest positive selected PSD. Stored PSD values remain
  unchanged.
- The default 200 Hz rate is explicitly labeled nominal. Acceleration PSD
  units are labeled `raw acceleration units²/Hz`.
- The CLI reuses `discover_recordings`, `select_recording`, and `load_ring`.
  It imports plotting code only after selecting Agg for `--no-show`, never
  loads Board data, and refuses to replace an existing output unless
  `--overwrite` is supplied.

### Commands executed and results

Focused tests:

```bash
conda run --no-capture-output -n writingring-viz \
    python -m pytest tests/test_spectral.py tests/test_spectral_cli.py -q
```

Initial result after the first test-helper correction cycle:
`1 failed, 36 passed`. The remaining failure was a frequency-validation
subclass mismatch; finite-range conversion errors were then consistently
wrapped as `InvalidFrequencyRangeError`.

Exact final focused result:

```text
37 passed in 1.43s
```

Complete suite:

```bash
conda run --no-capture-output -n writingring-viz \
    python -m pytest -q
```

Exact result:

```text
128 passed in 2.72s
```

Sample-data CLI verification:

```bash
conda run --no-capture-output -n writingring-viz \
    python scripts/plot_ring_accel_spectrum.py \
    --data-root data_sample/data \
    --user user_0 \
    --action 0 \
    --dataset-id 0 \
    --output outputs/dataset_0/ring_accel_psd_overlay.png \
    --no-show \
    --overwrite
```

Calculated summary:

```text
Samples: 10247
Nominal sampling rate: 200 Hz
Window duration: 1 s
Window size: 200 samples
Overlap: 50%
Hop size: 100 samples
Windows: 101
Frequency resolution: 1 Hz
Displayed range: 0–30 Hz
Aggregate: mean
```

The output is one nonempty 313,582-byte PNG at 1,100 × 1,000 pixels.
Programmatic figure verification found exactly three axes titled Acceleration
X, Y, and Z; 102 lines per axis (101 gray window curves plus one aggregate);
logarithmic y scales; shared x axes; and common 0–30 Hz limits. Visual
inspection confirmed the required vertically stacked layout, correct order,
visible per-window overlays, highlighted mean curves, neutral units, and
readable title/labels.

### Unresolved limitations

- The nominal 200 Hz analysis rate remains a processing assumption; the
  upstream Ring writer and a confirmed sampling-rate contract are unavailable.
- Acceleration physical units remain undocumented.
- Fixed sample windows do not compensate for timestamp duplicates, sampling
  jitter, gaps, drift, or recording-specific effective rate.
- The complete recording is analyzed without marker segmentation.
- Ring and Board remain unsynchronized; Board data is not loaded.
- No resampling, interpolation, zero-padding, timestamp repair, gyroscope
  analysis, spectrogram, or machine-learning export was added.

## Frequency-support dry run

Experiment command:

```bash
conda run --no-capture-output -n writingring-viz \
  python scripts/plot_ring_accel_spectrum.py \
  --data-root data_sample/data \
  --user user_0 \
  --action 0 \
  --dataset-id 0 \
  --plot-mode support \
  --sampling-rate 200 \
  --window-seconds 1 \
  --overlap 0.5 \
  --frequency-min 1 \
  --frequency-max 100 \
  --high-power-quantile 0.90 \
  --minimum-support 20 \
  --frequency-smoothing-bins 3 \
  --top-k-labels 5 \
  --output outputs/dataset_0/ring_accel_frequency_support.png \
  --no-show \
  --overwrite
```

Output: `outputs/dataset_0/ring_accel_frequency_support.png` (115,778-byte,
1,100 × 1,000 PNG). The run analyzed 10,247 samples as 101 complete
200-sample windows with a 100-sample hop and 1 Hz frequency resolution.

Top results, ranked by support percentage and then conditional median
relative-power strength:

- X: 2 Hz (100.00%, 0.279526), 1 Hz (100.00%, 0.247421), 3 Hz
  (99.01%, 0.145219), 4 Hz (94.06%, 0.060683), 5 Hz (84.16%, 0.034496).
- Y: 2 Hz (97.03%, 0.196745), 3 Hz (96.04%, 0.129158), 4 Hz
  (96.04%, 0.069669), 1 Hz (95.05%, 0.157494), 5 Hz (79.21%, 0.037473).
- Z: 3 Hz (91.09%, 0.110451), 2 Hz (90.10%, 0.129282), 4 Hz
  (85.15%, 0.068298), 1 Hz (77.23%, 0.091631), 5 Hz (72.28%, 0.050389).

No experiment runtime problem occurred. Direct assertions confirmed normalized
relative-power rows, support counts matching the Boolean masks, bounded
percentages, and exactly three linear scatter-only axes with 0–100% y-limits.
The complete suite passed: `130 passed in 2.91s`. Interpretation remains
limited by the nominal 200 Hz assumption, fixed sample windows, undocumented
acceleration units, and this single-recording exploratory scope.
