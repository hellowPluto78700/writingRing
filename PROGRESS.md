# Progress

The upstream format reference remains in
[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Completed phases

- [x] Upstream data-format inspection
- [x] Recording discovery
- [x] Ring loading and validation
- [x] Board loading and validation
- [x] Matplotlib plotting
- [ ] Remaining CLI scripts
- [ ] Jupyter notebook
- [ ] Full workflow documentation and acceptance run

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
