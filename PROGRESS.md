# Data-format inspection progress

The durable findings are documented in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Inspection status

- [x] Read `AGENTS.md`, `TASK.md`, `PROGRESS.md`, and
  `docs/DATA_FORMAT_TASK.md`.
- [x] Read all required upstream files:
  `ring_plot.py`, `board_plot.py`, `core/imu_data.py`, `core/window.py`, and
  `core/sensel_lib/frame_data.py`.
- [x] Read the additional upstream board writer and Sensel binding files needed
  to trace timestamps, chunking, coordinates, force, and contact fields:
  `core/sensel_lib/board.py`, `sensel.py`, and `sensel_register_map.py`.
- [x] Listed and categorized every file under
  `data_sample/data/user_0/0`.
- [x] Inspected `ring_0` with the author's exact
  `np.fromfile(..., dtype=np.float64).reshape(-1, 7)` method.
- [x] Inspected `ring_1` with the same shape interpretation for comparison
  only.
- [x] Loaded `board_0`, `board_1`, consecutive chunks, and all sample chunk
  boundaries with the author's `compress_pickle.load` method.
- [x] Recorded actual top-level, frame, force-array, contact-container, and
  contact types and fields.
- [x] Confirmed ring column order from `ring_plot.py` and `IMUData`.
- [x] Confirmed the stored and display board coordinate transformations.
- [x] Inspected all four `timestamp.txt` files and traced both upstream uses.
- [x] Compared root and vendor `core` directories.
- [x] Verified numerical versus lexical board ordering.
- [x] Confirmed that unknown units and clock/synchronization questions remain
  explicitly unresolved.
- [x] Created `docs/DATA_FORMAT.md`.
- [x] No loader, plotter, CLI, notebook, or test implementation was added.

## Commands run and results

- `conda run -n writingring-viz python --version`
  - Result: Python 3.11.15.
- `diff -qr core vendor/WritingRing/core`
  - Result: only `core/sensel_lib/__pycache__` differs; source files match.
- `find data_sample/data/user_0/0 ...`
  - Result: four dataset IDs, eight ring binaries, four timestamp files, and
    39 board chunks were categorized.
- Author-equivalent NumPy ring inspection in the `writingring-viz` environment
  - Result: all files reshape to seven float64 columns; `ring_0` rates are
    approximately 200.8-201.1 Hz under the microsecond interpretation;
    timestamps are nondecreasing with duplicates.
- `file`, `xxd`, `gzip -t`, gzip-prefix inspection, and
  `compress_pickle.load`
  - Result: board files are valid gzip-compressed protocol-4 pickles; the
    loaded object is a list of official `FrameData` objects.
- Numeric all-boundary board inspection
  - Result: datasets 1-3 are chronological after an empty chunk 0; dataset 0
    jumps backward 387,876,490 µs between chunks 6 and 7.
- SHA-256 aggregate checks of `vendor/WritingRing` and `data_sample`
  - Baseline results:
    `315cc77dee0a30cb23750bb0a61804b165eb388080e6d659b10a867902dd5cd8`
    and
    `9cdc8a0c34bc955270ab282220b7fbd9e64edfe3a38bbe58601b2d297e0e9ff8`.

## Unresolved issues

- Ring index physical meanings and why `ring_1` is omitted upstream.
- Ring byte-order contract, independent timestamp clock source, nominal sample
  rate, and physical signal units.
- Physical units for board coordinates, force, force arrays, area, axes, and
  deltas.
- Exact synchronization guarantees among ring, board, and marker timestamps.
- Cause and desired handling of the empty chunk-zero files, delayed board
  starts, and dataset `0`'s older chunks `7..15`.
- Action-ID semantics.
- Upstream inconsistencies: `FrameData` mentions `perf_counter` while the
  writer uses `time.time`; `Board.FPS` is 50 while sample frame timing is about
  131 Hz.

## Next step

Implement recording discovery only, using user/action/dataset membership,
`ring_0`, and numeric board-chunk parsing. Keep parsing separate from loading,
and preserve the documented dataset `0` anomaly for later validation-policy
work.
