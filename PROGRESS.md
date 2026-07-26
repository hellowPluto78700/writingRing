# Progress

The upstream format reference remains in
[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Completed phases

- [x] Upstream data-format inspection
- [x] Recording discovery
- [x] Ring loading and validation
- [x] Board loading and validation
- [ ] Matplotlib plotting
- [ ] Remaining CLI scripts
- [ ] Jupyter notebook
- [ ] Full workflow documentation and acceptance run

## Files created or changed

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

## API implemented

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

## Validation policy

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

## Commands run and exact results

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

## Real-data results

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

## Dataset 0 anomaly

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

## Unresolved issues

- The reason every recording has an empty initial chunk remains unknown.
- Dataset `0`'s older chunk tail is retained; any later user-facing treatment
  remains a plotting/product decision, not a loader correction.
- Physical units for contact positions, force, area, axes, deltas, and force
  arrays remain undocumented.
- Ring/board synchronization guarantees remain unknown and were not
  implemented.
- The official pickle classes currently rely on the repository-level `core`
  namespace being importable.

## Next recommended phase

Implement only reusable Matplotlib plotting from the already loaded ring and
board tables. Preserve raw/inferred distinctions, show contact-free recordings
gracefully, and make dataset `0`'s validation warnings visible. Do not add
synchronization, CLI expansion, or notebook work unless separately requested.
