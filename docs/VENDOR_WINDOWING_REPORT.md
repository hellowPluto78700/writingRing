# Vendor WritingRing Windowing Analysis

## 1. Executive Summary

**Confirmed:** The supplied vendor implementation does not perform true
sliding-window segmentation for either Ring or Board data. It defines a
generic `Window` class, but no other vendor file imports, constructs, or calls
that class. Searches for `Window`, its accessors, fixed-length slicing,
interpolation, resampling, padding, model-input construction, and related
buffering terms found no external `Window` call site.

`core/window.py` implements a list-backed, bounded FIFO rolling buffer when it
is populated exclusively through `push()`: each push appends the newest item,
and one oldest item is removed when the configured limit is exceeded. It is
not a circular-array ring buffer. The class can also be initialized directly
with an over-length list, so the bound is not a universal class invariant.
`head()` and `tail()` are shallow list slices wrapped in new `Window` objects;
they are extraction helpers, not an automatic segmentation pipeline.

The actual vendor paths are separate offline plotters:

- Ring: one complete `ring_0` file is decoded as native-endian float64 rows of
  seven values, text markers are mapped to nearest Ring row indices, and all
  six signal columns are plotted.
- Board: all `*.gz` files in an action directory are independently loaded,
  frames are filtered to one randomly selected adjacent-marker interval, and
  all contacts in that variable-duration interval are accumulated for one
  scatter plot.

The Board acquisition code does use temporary buffering, but for persistence,
not algorithmic windowing. Frames pass through a multiprocessing queue and
are dumped in lists of at most 1,000 frames. A stop dumps the final list,
including an empty list. Those files are recording chunks, not model windows.

Text timestamps provide a shared marker domain used independently by the two
plotters. There is nearest-timestamp marker matching for Ring and inclusive
timestamp filtering for Board, but no direct Board-to-Ring matching, clock
offset estimation, interpolation, resampling, paired-window construction, or
sampling-rate correction.

Important limitations are the unused `Window` class, unsorted action-wide
Board globbing, lack of dataset-prefix isolation in `board_plot.py`, silent
per-file Board error skipping, no verified Ring writer or clock contract, and
acquisition code that never appends generated frames to `Board.frames`.

## 2. Scope and Method

Files read directly:

- `AGENTS.md`, `TASK.md`, `PROGRESS.md`, `README.md`;
- `docs/DATA_FORMAT.md` and `docs/WINDOWING_ANALYSIS_TASK.md`;
- `vendor/WritingRing/ring_plot.py`;
- `vendor/WritingRing/board_plot.py`;
- `vendor/WritingRing/core/window.py`;
- `vendor/WritingRing/core/imu_data.py`;
- `vendor/WritingRing/core/sensel_lib/board.py`;
- `vendor/WritingRing/core/sensel_lib/frame_data.py`;
- `vendor/WritingRing/core/sensel_lib/sensel.py`;
- `vendor/WritingRing/core/sensel_lib/sensel_register_map.py`;
- the current modules under `src/writingring/`, relevant scripts, and tests.

The complete vendor file tree was enumerated. Recursive source searches covered
`Window` imports/construction, `push`, `head`, `tail`, `first`, `last`, `get`,
array/list slicing, queues, buffers, histories, timestamps, interpolation,
resampling, padding, truncation, synchronization, and model inputs. A
dedicated import/construction search returned only the `Window` definition and
its own `map()` factory.

Read-only Python 3.11 inspection in the `writingring-viz` Conda environment
measured Ring files, marker files, deserialized Board chunks, chunk timestamp
boundaries, marker-to-Ring indices, and marker-interval Board counts under
`data_sample/data/user_0/0/`. A temporary script was placed in `/tmp`. No
vendor or sample file was changed.

Line references below refer to the inspected source as it existed during this
investigation.

## 3. Terminology

- **Sample:** One Ring row: six IMU signal values plus one timestamp.
- **Frame:** One `FrameData` Board scan containing a dense force array, a
  timestamp, and zero or more contacts.
- **Contact:** One `ContactData` record attached to a frame.
- **Recording:** In this project, files sharing user, action, and the leading
  numeric filename prefix. The vendor code does not define a recording object.
- **Chunk:** One persisted Board list written at a numeric `segment` boundary,
  normally 1,000 frames or the remaining list on stop.
- **Buffer:** Temporary retained data, such as the Board save queue/list or a
  hypothetical `Window` instance.
- **Rolling window:** A bounded recent-history container updated as new items
  arrive. `Window.push()` has this behavior.
- **Sliding window:** A sequence of fixed-length segments emitted from a
  stream, normally with a defined stride. No such pipeline is present.
- **Stride:** The number of input samples between successive emitted windows.
- **Overlap:** Samples shared by successive emitted windows, commonly
  `length - stride`. Neither stride nor algorithmic overlap is defined here.

An adjacent text-marker interval is variable-duration timestamp-based trial
segmentation. It is not a fixed-length sliding window. A Board file is a
persistence batch. It is also not a sliding window.

## 4. Window Class Analysis

### Construction and representation

`Window(window_length, window=None)` stores `window_length` unchanged and uses
either a new empty Python list or the exact supplied list
(`core/window.py:9-12`). There is no validation that:

- the length is a positive integer;
- a supplied list fits the configured length;
- elements have a common type or shape;
- elements contain timestamps;
- timestamps are ordered;
- samples have a common rate.

The constructor does not copy or trim a supplied list. External mutation of
that list therefore mutates the `Window`.

### Update behavior

The complete normal update rule is (`core/window.py:14-17`):

```text
push(new_item):
    append new_item at the end
    if current length > window_length:
        remove exactly one item at index 0
```

Thus old samples are discarded and the new sample is retained. Starting from
an empty list and a positive integer length `L`, capacity warms from zero to
`L`, then stays at `L`. Each later push gives a one-sample rolling update.
The class itself emits no segment and defines no stride.

Because only one item is removed, a constructor-supplied over-length list
remains over length after a push. With length zero, a push appends and then
removes the item. Negative lengths also do not produce meaningful bounded
storage. These are consequences of missing validation, not documented modes.

### Access and mutation methods

| Method | Confirmed behavior |
| --- | --- |
| `clear()` | Calls `list.clear()`; configured length is retained (`19-20`). |
| `first()` | Returns element `0`; empty input raises `IndexError` (`22-23`). |
| `last()` | Returns element `-1`; empty input raises `IndexError` (`25-26`). |
| `get(index)` | Uses normal Python indexing, including negative indices; an invalid index raises `IndexError` (`28-29`). |
| `head(length)` | Wraps `self.window[:length]` in a new `Window` with the original configured length (`31-32`). |
| `tail(length)` | Wraps `self.window[-length:]` in a new `Window` with the original configured length (`34-35`). |
| `capacity()` | Returns the current element count, not remaining capacity (`37-38`). |
| `empty()` | Returns whether current count is zero (`40-41`). |
| `full()` | Tests equality with `window_length`, not `>=` (`43-44`). |

Requests longer than current capacity return every current item. `head(0)`
returns empty, but `tail(0)` returns the whole list because Python evaluates
`-0` as `0`. Negative requests inherit Python slice semantics and are not
rejected.

`head()` and `tail()` create independent list objects via slicing, but the
elements are shallow references. Pushing or clearing the returned window does
not change the source list; mutating a shared mutable element is visible
through both windows.

### Other helpers

`sum()` maps an optional function and uses built-in sum (`46-47`). `count()`
counts mapped results equal to `True` (`49-50`). `map()` returns another
`Window` with a new list and the same configured length (`52-53`).
`argmax()` linearly finds the first maximum, but returns integer `0` when empty
instead of its annotated tuple (`55-64`). `to_numpy()` concatenates elements
along axis zero; it fails on an empty list and requires compatible array
shapes (`66-67`). `to_numpy_inside()` calls each element's `to_numpy()`
(`69-70`). `feature()` computes global mean, minimum, maximum, skew-like, and
kurtosis-like values over `np.array(window)` without axis selection
(`72-83`); it is not invoked by vendor processing.

`set_to_last_value()` iterates exactly `window_length - 1` positions and
copies or assigns the final element into those positions (`85-91`). It assumes
a sufficiently populated window and fails for an empty or short one. For
objects with `assigned_by`, `IMUData.assigned_by()` copies signals while
retaining each target object's original timestamp (`imu_data.py:65-75`).

### Technical classification

**Confirmed:** The best description is a **list-backed bounded FIFO rolling
buffer**, conditional on positive length and push-only population from empty.
It can support a sliding recent-history view, but is not itself an emitted
sliding-window dataset. It is not a ring buffer in the data-structure sense:
there is no circular storage or write pointer, and `pop(0)` shifts a list.

## 5. Window Call-Site Inventory

No external call site exists.

| File | Window object | Length | Stored data | Push trigger | Consumer | Full-window requirement |
| --- | --- | ---: | --- | --- | --- | --- |
| All vendor files outside `core/window.py` | None | N/A | N/A | N/A | N/A | N/A |

The only constructor expressions after the class declaration are internal:
`head()` and `tail()` wrap slices (`window.py:31-35`), and `map()` wraps mapped
items (`52-53`). No vendor caller invokes those methods. No configured Window
length, push frequency, clear trigger, full-window gate, emitted overlap,
stride, or timestamp policy therefore exists.

## 6. Ring Data Pipeline

The actual pipeline is:

```text
explicit *_ring_0.bin path
→ np.fromfile(path, dtype=np.float64)
→ reshape all values to (-1, 7)
→ six raw signal columns plus per-row timestamp
→ derive *_timestamp.txt by string replacement
→ parse each marker timestamp as float
→ nearest Ring-timestamp row via argmin(abs(ring_timestamp - marker))
→ plot all six raw channels against row index
→ draw marker lines at nearest row indices
```

`ring_plot.py:6-35` performs this entire path. One record is seven float64
values, or 56 bytes:

```text
acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, timestamp
```

The channel order is corroborated by `IMUData.to_numpy_with_timestamp()`
(`imu_data.py:59-63`). `IMUData` offers object construction, norms, direction,
scaling, array conversion, and grouped data (`imu_data.py:7-96`), but
`ring_plot.py` does not import or instantiate it. Consequently:

- no `IMUData.scale()` transform is applied by the plotting path;
- no feature extraction is applied;
- no `Window` is populated;
- no fixed sampling rate or sample-index-derived time is constructed;
- no Ring windows, overlap, stride, padding, or incomplete-window policy
  exists.

Timing comes from the seventh value in every binary row. The external marker
file is used for annotation alignment, not to construct sample timestamps.
The nearest-neighbor operation maps each text marker to Ring sample index; it
does not resample the Ring signal.

## 7. Board Data Pipeline

### Acquisition and persistence

The acquisition path is:

```text
Sensel read/get-frame calls
→ host timestamp int(time.time() * 1e6)
→ copy dense force array and multiply it by 0.2
→ FrameData(force_array, timestamp)
→ normalize each contact x/230 and y/130
→ ContactData with contact fields and frame_id
→ multiprocessing q_frames while recording
→ save-process Python list
→ compress_pickle.dump at 1,000 frames or stop
→ {supplied_base}_{segment}.gz
```

Evidence is `board.py:15-55, 60-79, 133-165`. The queue decouples acquisition
from disk compression. The save-process list is a persistence batch:
`MAX_FRAME_SIZE = 1000`; on reaching it, the entire list is passed to a child
dump process, the list is replaced, and `segment` increments. `"stop"` dumps
the remaining list even if it is empty, then resets the segment and list.
`"change_filename"` discards the current list without dumping and resets the
segment. `"flush"` is explicitly a no-op.

`FrameData` retains `force_array`, timestamp, and a contact list
(`frame_data.py:56-80`). `ContactData` retains ID, state, x, y, area, force,
major/minor axes, coordinate/force/area deltas, label, and frame ID
(`frame_data.py:3-43`).

Two acquisition limitations affect any buffering interpretation:

1. `_run()` loops over available device frames, but frame conversion is
   dedented after that loop (`board.py:138-148`). If multiple frames are
   available, only the last fetched device frame is converted; if none is
   available, local timestamp state is unsafe.
2. `_initFrame()` creates `self.frames = []`, and `_sync()`, `getFrame()`,
   `getNewFrame()`, and `getFrameTime()` consume it
   (`board.py:114, 128-131, 178-192`), but the supplied `_run()` never appends
   the generated frame to `self.frames`. It only queues the frame for saving
   (`157-159`). Therefore this apparent live history is not an operational
   rolling buffer in the shown path.

### Offline visualization

The offline pipeline is:

```text
action directory
→ randomly choose one *_timestamp.txt
→ randomly choose one adjacent marker pair [start, end]
→ glob every action-level *.gz, without sorting or prefix filtering
→ compress_pickle.load each file into a list of FrameData
→ retain frames whose timestamp is inclusively within [start, end]
→ for every retained frame, extract each contact's x, 1-y, and force
→ append those values to plot-wide lists
→ one force-colored scatter plot
```

Evidence is `board_plot.py:8-9, 23-63, 66-101`. A `.gz` contains multiple
frames, not one frame or one precomputed window. Each frame carries its dense
matrix and contact records.

Board visualization does not retain area, axes, deltas, IDs, states, frame
IDs, dense force arrays, or frame timestamps in its final plot arrays. Frames
with no contacts contribute no scatter points. Deserialization or extraction
errors are caught per file, printed, and skipped (`48-49`). Missing timestamp
files skip an action (`71-73`). There is no explicit missing-frame detection,
rolling frame window, cross-frame contact grouping by ID, overlap, padding, or
final partial-window handling.

The marker interval is timestamp-based, variable length, and selected once.
It is trial/label-range filtering, not fixed-sample windowing.

## 8. File-Level Chunking

The leading number associates the sample filenames operationally: Ring
plotting replaces `ring_0.bin` with `timestamp.txt`, and project/sample
inspection uses that number as the dataset ID. However, the supplied Board
writer receives an arbitrary base filename and does not parse or attach a
dataset ID. Its Board plotter ignores leading prefixes and scans all action
gzip files. The semantic origin of the leading number is therefore
**unresolved in the vendor writer**, though filename membership is strongly
supported by the dataset and Ring path derivation.

The Board suffix is **confirmed** as the persistence `segment` counter
(`board.py:19-44`). It starts at zero, increments after each 1,000-frame dump,
and resets on stop or filename change. One Board file is one recording batch,
not one algorithmic window. Normal intermediate files hold 1,000 frames; the
last can be shorter or empty.

The Ring suffix is **unresolved** as a device/body-location meaning. The sample
contains valid `ring_0` and `ring_1` streams, but the vendor plotter accepts
and reads only a supplied `ring_0` path. No Ring writer is supplied. The
suffix cannot be established as a temporal chunk index, and Board segment
indices do not correspond directly to Ring suffixes.

The offline Ring plot does not concatenate Ring files. The Board plot
processes each gzip independently but accumulates qualifying contacts into one
plot. It does not concatenate frames into an ordered sequence and does not
carry `Window` state across files. Glob order can affect processing order but
not the final scatter geometry.

Board persistence list state resets after each file dump while acquisition
continues; this is a storage boundary, not a trial/window reset. A stop or
filename change resets `segment` to zero. Numeric file boundaries do not
define any downstream sliding-window boundary because no downstream sliding
windows exist.

## 9. Time Synchronization

Three timing mechanisms are present:

- Ring rows contain a seventh timestamp value; its writer, clock source,
  guaranteed unit, and nominal rate are not supplied.
- Board acquisition assigns host wall-clock microseconds with
  `int(time.time() * 1e6)` (`board.py:140`).
- Text files contain timestamp/label markers. Ring plotting maps each marker
  to its nearest Ring timestamp (`ring_plot.py:9-17`). Board plotting uses two
  adjacent marker values as inclusive frame-filter bounds
  (`board_plot.py:75-97`).

Thus text markers are calculation inputs, not metadata only. Sample values
show that Ring, Board, and marker timestamps are numerically comparable.
Nevertheless, the vendor code does not:

- directly compare a Board timestamp to a Ring timestamp;
- estimate or apply an inter-stream offset;
- interpolate, resample, or nearest-match Board frames to Ring samples;
- crop one complete stream to the other's extent;
- model clock drift or sampling-rate stability;
- create paired Board/Ring windows.

**Confirmed:** synchronization is marker-mediated and stream-specific, not
explicit Board-to-Ring synchronization. The Ring plot uses nearest-neighbor
marker placement; the Board plot uses range filtering. These operations occur
instead of windowing, not before or after a shared windowing stage.

A marker interval can conceptually identify Ring rows and Board frames in the
same labeled time range, but the vendor program never constructs that paired
object. Whether both devices truly share clock origin and stability is
**unresolved**.

## 10. Window Parameters

| Stream | Mechanism | Length | Sampling rate | Duration | Stride | Overlap | Reset condition |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Generic `Window` | Push-only bounded FIFO history; unused | Constructor argument | Unknown | Unknown | Not defined | Not defined | Explicit `clear()` only |
| Ring plotting | Whole-file array | Entire file | No code-declared rate | Whole file | N/A | N/A | New function call/file |
| Ring marker range | Nearest marker-to-row annotation | One index per marker | N/A | N/A | N/A | N/A | New function call |
| Board save list | Persistence chunk buffer | 1,000 max normally | `Board.FPS=50` exists but does not establish saved-sample rate | Do not infer | 1,000-frame dump boundary, not ML stride | None as files; not algorithmic windows | Full batch, stop, or filename change |
| Board plot range | Inclusive adjacent-marker filter | Variable frame count | No reliable declared plot-input rate | `end_timestamp - start_timestamp` | Not defined; one random range | Not defined | New selected interval/call |
| `Board.frames` | Intended live list API, but never populated in shown `_run()` | N/A operationally | N/A | N/A | N/A | N/A | Initialized in `_initFrame()` |

Although `Board.FPS = 50`, `_sync()` multiplies a microsecond difference by 50
and checks whether it is below 1 (`board.py:128-131`), and `self.frames` is
empty in the shown producer path. The constant is not sufficient evidence for
a 50 Hz saved stream or window duration.

## 11. Worked Example

Recording prefix `0` under `data_sample/data/user_0/0/` contains:

- `0_ring_0.bin`: 573,832 bytes / 56 bytes per row = 10,247 rows;
- `0_ring_1.bin`: 10,155 rows, not read by the vendor Ring plot;
- `0_timestamp.txt`: 26 timestamp/label markers;
- `0_board_0.gz` through `0_board_15.gz`.

The Ring plot reads all 10,247 `ring_0` rows. Its stored timestamps span
`1720476254890600` to `1720476305894185`, a 51.003585-second difference under
the sample-supported microsecond interpretation. There are 3,099 unique
timestamps, 7,148 duplicate consecutive steps, and no backward step. The
first three marker-to-row mappings are 362, 865, and 1,166; the last is 9,892.
No buffer update or extracted Ring window occurs.

Board chunk measurements in numeric filename order:

| Chunk(s) | Frames | Timestamp behavior |
| --- | ---: | --- |
| `0` | 0 | Serialized empty list |
| `1..5` | 1,000 each | Strictly increasing within and across boundaries |
| `6` | 701 | Ends at `1720476305908521` |
| `7..14` | 1,000 each | Older internally continuous tail |
| `15` | 664 | Ends at `1720475983195457` |

There are 14,365 total frames and 5,800 contacts. Chunks `1..6` overlap the
Ring time range and contain 5,701 frames. The `6 -> 7` boundary jumps backward
from `1720476305908521` to `1720475918032031`, delta `-387876490`. Numeric
chunk index therefore does not guarantee chronological continuity.

For an example marker interval, the vendor-compatible action-wide filter from
marker `t=1720476260694202` to the next marker
`f=1720476262687873` retains 44 Board frames and 24 contacts. The first two
marker intervals retain zero frames because the nonempty matching Board
sequence begins later. The plotter would choose only one of the 25 adjacent
intervals at random, scan all action-level gzip files (including other
prefixes), accumulate x, `1-y`, and force, and emit one scatter plot. It
creates no fixed-length Board window.

## 12. Edge Cases and Limitations

- Empty `Window.first()`, `last()`, and `get()` fail through normal list
  indexing; empty NumPy concatenation and feature calculation are also unsafe.
- Constructor-supplied Window data can exceed the configured limit; short or
  empty data can break `set_to_last_value()`.
- `tail(0)` unexpectedly returns all elements.
- Short Ring files are not padded or windowed. A non-multiple-of-seven value
  count fails at reshape; an empty array later makes nearest-marker reduction
  fail.
- Ring timestamps may repeat. The sample shows thousands of duplicate steps.
- The Ring plot requires the derived timestamp file and does not handle its
  absence or malformed lines.
- Board stop can serialize an incomplete or empty final chunk. The code makes
  empty files possible, but does not prove why every sample chunk zero is
  empty.
- Board filename change discards any currently buffered unsaved list.
- `multiprocessing.Queue.empty()` is not a synchronization guarantee, so the
  save loop has race sensitivity.
- Multiple available Sensel frames can be fetched but only the last converted,
  potentially dropping intermediate frames.
- The shown live `Board.frames` accessors cannot operate because `_run()` does
  not append to that list.
- Board plotting neither sorts chunks nor restricts them to the chosen
  timestamp file's prefix.
- Any Board file load/extraction error is printed and skipped, allowing
  incomplete visualization without structured failure.
- Inclusive adjacent intervals would both include a frame exactly on their
  shared marker if all intervals were generated. The vendor generates only
  one random interval per action invocation.
- Different Ring/Board rates, clock offsets, drift, timestamp gaps, missing
  chunks, stale chunks, and different stream extents are not corrected.
- There is no incomplete-final-window concept because no algorithmic windows
  are emitted.

## 13. Confirmed Findings vs Inferences

### Confirmed findings

| Finding | Evidence |
| --- | --- |
| `Window` has no external vendor use | Recursive import/construction/method searches; only `window.py` definitions/internal factories matched |
| Push-only use is bounded FIFO recent history | `window.py:14-17` |
| No Ring scaling/windowing is applied | Complete `ring_plot.py:6-35`; no `IMUData`/`Window` import |
| Ring record is seven float64 values | `ring_plot.py:7-8`; `imu_data.py:62-63`; sample byte/count measurements |
| Markers map to nearest Ring rows | `ring_plot.py:9-17` |
| Board save chunks are 1,000-frame list batches | `board.py:19-44` |
| Board frames contain matrix, timestamp, contacts | `frame_data.py:56-80`; deserialized sample |
| Board plot filters one adjacent marker interval | `board_plot.py:71-98` |
| Board plot is action-wide, unsorted, and not prefix-isolated | `board_plot.py:26, 33-49` |
| No direct Board/Ring synchronization exists | Complete vendor call-flow/search audit |
| Dataset 0 has a backward `6 -> 7` boundary | Sample measurement |

### Inferred interpretations

| Interpretation | Basis |
| --- | --- |
| Leading numeric prefix is a capture/dataset identifier | Ring filename replacement, co-located sample naming, project discovery semantics; no complete vendor recorder is supplied |
| Ring timestamps use the same broad time domain as markers/Board | Magnitudes and successful marker matching in sample; Ring writer is absent |
| Text-marker intervals represent labeled trials/actions | Each line pairs time and character, and Board plot names output from the selected label; experimental semantics are not documented |
| `Board.frames` was intended as a live history/consumer queue | `_sync`, `getFrame`, `getNewFrame`, and `getFrameTime` reference it, but producer append is absent |

### Unresolved questions

| Question | Missing evidence |
| --- | --- |
| What do Ring suffixes 0 and 1 physically mean? | No Ring writer or device metadata |
| What is the guaranteed Ring timestamp unit/clock/rate? | Only reader and observed sample are supplied |
| Do Ring and Board share a clock with bounded offset/drift? | No synchronization contract or calibration |
| Why is every initial Board chunk empty? | Writer permits empty stop dumps, but capture-controller call sequence is absent |
| Why does dataset 0 contain an older tail? | No session/control log |
| Why is `Window` shipped but unused? | No callers or project history in supplied source |
| Was the missing `self.frames.append(frame)` intentional? | No tests or acquisition documentation |

## 14. Implications for Our Reimplementation

The current project already reproduces or preserves the useful verified
semantics:

- discovery groups user/action/dataset prefix and numerically sorts Board
  chunk indices (`src/writingring/discovery.py`);
- only `ring_0` is loaded as native-endian float64 and reshaped to seven
  columns (`src/writingring/ring_loader.py`);
- Ring raw timestamps and duplicate-step validation are retained;
- Board chunks use `compress_pickle.load` with official pickle classes and
  retain frame/contact order (`src/writingring/board_loader.py`);
- empty chunks/frames and the dataset 0 backward boundary are reported rather
  than silently dropped;
- stored y and upstream display `1-y` are separated;
- Matplotlib plots preserve loaded order and do not claim synchronization
  (`src/writingring/plotting.py`).

Intentional differences improve correctness:

- current discovery isolates Board chunks by dataset prefix and numeric index;
  vendor Board plotting globs all action gzip files without sorting;
- current loaders validate and report failures; vendor plotting prints and
  skips Board file errors;
- current APIs represent complete recordings deterministically; vendor Board
  plotting randomly chooses a timestamp file and interval;
- current plotting exposes warnings and explicit raw/inferred timing; vendor
  plots do not expose continuity problems.

Compatibility does **not** require adopting the unused vendor `Window`,
action-wide Board mixing, random selection, or acquisition bugs. Compatibility
does require preserving the seven-value Ring schema, Board pickle classes and
contact transform, integer Board segment semantics, text-marker behavior when
marker visualization is added, and stored ordering unless a separate,
explicit policy is selected.

No window-generation API currently exists under `src/writingring`. A future
implementation should be separate from loading and plotting, should accept
explicit stream/clock policies, and should not call 1,000-frame persistence
chunks “windows.”

Additional tests should cover:

- exact sample-based window length/stride/overlap and final-tail policy;
- windows crossing Board file boundaries without assuming timestamp
  continuity;
- explicit handling of dataset 0's backward boundary;
- repeated Ring timestamps and rate estimation;
- marker interval endpoint inclusion;
- missing markers/Board chunks and empty contact intervals;
- independent versus paired stream windows;
- synthetic known offsets, drift, interpolation, and resampling;
- non-mutation of loaded source tables.

## 15. Recommended Next Steps

### Compatibility-preserving changes

1. Add a reusable text-marker loader that preserves raw timestamps and labels.
2. Add deterministic marker-to-Ring nearest-index and marker-to-Board
   inclusive-range helpers matching vendor semantics.
3. Keep file discovery/loading independent from segmentation.

### Correctness fixes

1. Define anomaly policies for backward timestamp boundaries before permitting
   windows to cross them.
2. Require explicit behavior for gaps, duplicates, short input, and final
   incomplete segments.
3. Keep persistence chunk boundaries as metadata; concatenate only under a
   validated recording policy.

### Synchronization improvements

1. Establish the Ring clock source/unit and whether clocks share an epoch.
2. Measure per-recording offset and drift using markers or another calibration
   source.
3. Make interpolation/nearest-neighbor rules and acceptable timing error
   explicit; retain original timestamps and matching residuals.

### Machine-learning window generation

1. Introduce a new tested window specification with sample/time length,
   stride, alignment anchor, and tail policy.
2. Define Ring-only and Board-only windows first.
3. Add paired multimodal windows only after synchronization guarantees exist.
4. Record provenance: source rows, chunk indices, time bounds, gaps, and
   anomaly flags for every emitted window.

### Testing and validation

1. Use synthetic streams for exact boundary mathematics.
2. Add sample-data regression measurements for all four prefixes.
3. Test file-boundary independence and dataset-prefix isolation.
4. Verify that no algorithm silently reorders, repairs, or drops source data.

## Appendix A. Source Evidence

| Conclusion | Source |
| --- | --- |
| Window storage and push eviction | `vendor/WritingRing/core/window.py:9-17` |
| Window accessors and shallow slice construction | `window.py:19-44` |
| Window aggregation/features/fill helper | `window.py:46-91` |
| IMU fields, scale, timestamp conversion | `vendor/WritingRing/core/imu_data.py:7-75` |
| Ring decode, marker match, plot | `vendor/WritingRing/ring_plot.py:6-35` |
| Board offline load/filter/scatter | `vendor/WritingRing/board_plot.py:8-63` |
| Random marker-file and interval selection | `board_plot.py:66-101` |
| Board save batching and reset controls | `vendor/WritingRing/core/sensel_lib/board.py:15-55` |
| Board device initialization/queues | `board.py:60-120` |
| Board timestamp, matrix, contact conversion | `board.py:128-165` |
| Inoperative `self.frames` consumers | `board.py:114, 128-131, 178-192` |
| Contact and frame stored fields | `vendor/WritingRing/core/sensel_lib/frame_data.py:3-80` |
| Sensel raw frame/contact structures | `vendor/WritingRing/core/sensel_lib/sensel.py:34-107` |
| Device frame-read wrappers | `sensel.py:193-212` |

## Appendix B. Sample Data Measurements

### Ring files

| Prefix | Ring index | Rows | Bytes | First timestamp | Last timestamp | Difference under µs interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 10,247 | 573,832 | 1720476254890600 | 1720476305894185 | 51.003585 s |
| 0 | 1 | 10,155 | 568,680 | 1720476254890689 | 1720476305910700 | 51.020011 s |
| 1 | 0 | 11,812 | 661,472 | 1720476332594540 | 1720476391386683 | 58.792143 s |
| 1 | 1 | 11,693 | 654,808 | 1720476332594540 | 1720476391353295 | 58.758755 s |
| 2 | 0 | 10,158 | 568,848 | 1720476418482738 | 1720476469059301 | 50.576563 s |
| 3 | 0 | 11,651 | 652,456 | 1720476494662869 | 1720476552591235 | 57.928366 s |

Every record is 56 bytes. All four `ring_0` files are nondecreasing but contain
duplicate consecutive timestamp values. Prefix 0 has 3,099 unique timestamps
and 7,148 duplicate steps.

### Board files

| Prefix | Named chunks | Empty chunk | Total frames | Contacts | Matching chronological nonempty sequence | Sequence frames |
| ---: | --- | ---: | ---: | ---: | --- | ---: |
| 0 | `0..15` | 0 | 14,365 | 5,800 | `1..6` | 5,701 |
| 1 | `0..7` | 0 | 6,724 | 3,414 | `1..7` | 6,724 |
| 2 | `0..6` | 0 | 5,631 | 3,001 | `1..6` | 5,631 |
| 3 | `0..7` | 0 | 6,612 | 3,396 | `1..7` | 6,612 |

All nonempty inspected chunks have strictly increasing internal frame
timestamps. Prefixes 1-3 have positive adjacent numeric boundaries. Prefix 0
has the single backward numeric boundary:

```text
chunk 6 last:  1720476305908521
chunk 7 first: 1720475918032031
delta:         -387876490
```

All four marker files contain 26 strictly increasing timestamp/character
lines. Every marker falls inside its matching `ring_0` extent. The matching
Board sequences begin after the first three markers, so only 23 marker values
fall inside those sequences. These measurements support marker-based range
selection but do not establish a synchronized Board/Ring clock contract.
