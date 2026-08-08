# WritingRing data format

This document is the durable format reference for the local visualization
project. It separates facts supported by the upstream source or sample data
from unknowns and proposed behavior for future code.

The inspection used Python 3.11.15 from the `writingring-viz` Conda
environment. `diff -qr core vendor/WritingRing/core` reported only
`core/sensel_lib/__pycache__`; the source files under the two `core`
directories are identical. Pickles name the official classes under
`core.sensel_lib.frame_data`.

Source: command `diff -qr core vendor/WritingRing/core`

## 1. Directory hierarchy

The confirmed hierarchy is:

```text
data/
└── user_{user_id}/
    └── {action_id}/
        ├── {dataset_id}_ring_0.bin
        ├── {dataset_id}_ring_1.bin
        ├── {dataset_id}_timestamp.txt
        └── {dataset_id}_board_{chunk_index}.gz
```

- **User ID** is the directory-level identifier that the upstream code calls
  `user`. The sample contains `user_0`. The upstream files do not document
  participant metadata or the meaning of the numeric suffix.
- **Action ID** is the immediate subdirectory under a user. The upstream code
  calls these paths `action_dir`. The sample inspected here contains action
  `0`; the semantic meaning of an action number is not documented.
- **Dataset ID** is the integer filename prefix, such as the first `0` in
  `0_ring_0.bin`. It distinguishes captures inside one user/action directory.
- **Board chunk index** is the integer after `_board_`. The board writer starts
  `segment` at zero and increments it after each 1,000-frame dump.
- **Ring index** is the integer in `_ring_0.bin` or `_ring_1.bin`. Two streams
  exist in every sample recording, but their physical device/body-location
  meanings are not documented in the supplied upstream source.

Source: `vendor/WritingRing/board_plot.py`, `plot_board_data`, lines 66-75  
Source: `vendor/WritingRing/core/sensel_lib/board.py`, `save_frame`, lines 19-44  
Verified from: `data_sample/data/user_0/0`

The inspected directory contains four dataset IDs (`0` through `3`), four
timestamp files, eight ring files, and 39 board chunks. Named chunk indices
are contiguous within each dataset: `0..15`, `0..7`, `0..6`, and `0..7`,
respectively.

Verified from: file listing under `data_sample/data/user_0/0`

## 2. Recording membership rules

For this project, files belong to the same nominal recording when they have
the same user directory, action directory, and dataset-ID prefix. Thus
`1_ring_0.bin`, `1_ring_1.bin`, `1_timestamp.txt`, and every
`1_board_{chunk}.gz` are filename members of dataset `1`.

The association is partly explicit upstream: `ring_plot.py` derives the
timestamp path by replacing `ring_0.bin` with `timestamp.txt`. Board chunk
names are produced by appending a numeric segment to a supplied board save
filename. The upstream board plotter itself does **not** group gzip files by
dataset ID; it globs every `*.gz` file in the action directory and relies on
timestamp filtering.

Source: `vendor/WritingRing/ring_plot.py`, `plot_ring_data`, lines 6-16  
Source: `vendor/WritingRing/core/sensel_lib/board.py`, `save_frame`, lines 19-44  
Source: `vendor/WritingRing/board_plot.py`, `visualize_touch_data`, lines 23-43

Use `{dataset_id}_ring_0.bin` for the planned implementation.
`ring_plot.py` loads only `ring_0`, and the project instructions explicitly
require ignoring `*_ring_1.bin`. Inspection shows that `ring_1` is a valid
second seven-column float64 stream with a similar time range, not a duplicate
or malformed copy. The supplied upstream source never states why it is
excluded or what it physically represents.

Source: `vendor/WritingRing/ring_plot.py`, `plot_ring_data`, lines 6-8  
Source: `AGENTS.md`  
Verified from: `0_ring_0.bin`, `0_ring_1.bin`, and corresponding files for
datasets `1` through `3`

Filename membership alone does not guarantee temporal membership. Dataset
`0` contains an observed stale/older tail: chunks `1..6` overlap the dataset
`0` ring interval, while chunks `7..15` are approximately 388 seconds earlier
at the `6 -> 7` boundary. Future code must detect this rather than silently
asserting that all same-prefix chunks are one continuous sequence.

Verified from: `0_board_1.gz` through `0_board_15.gz` and `0_ring_0.bin`

## 3. Ring binary format

The concise upstream-equivalent loader is:

```python
raw = np.fromfile(path, dtype=np.float64)
data = raw.reshape(-1, 7)
```

Source: `vendor/WritingRing/ring_plot.py`, `plot_ring_data`, lines 6-8

The result is a two-dimensional NumPy array with seven float64 columns in this
exact order:

| Column | Field |
| ---: | --- |
| 0 | `acc_x` |
| 1 | `acc_y` |
| 2 | `acc_z` |
| 3 | `gyr_x` |
| 4 | `gyr_y` |
| 5 | `gyr_z` |
| 6 | `timestamp` |

The order is confirmed independently by both the plot labels and
`IMUData.to_numpy_with_timestamp()`.

Source: `vendor/WritingRing/ring_plot.py`, `plot_ring_data`, lines 14-25  
Source: `vendor/WritingRing/core/imu_data.py`, class `IMUData`, lines 7-20
and 59-63

`np.float64` is native-endian in the author's loader. On the inspected
little-endian host it is `<f8`, and all sample files decode into plausible,
finite values. The file carries no byte-order marker, and the upstream source
does not specify cross-platform endianness. Therefore the sample is confirmed
little-endian-compatible, but a portable on-disk byte-order contract remains
undocumented.

Verified from: all eight ring files under `data_sample/data/user_0/0`

The four `ring_0` samples decode as follows:

| Dataset | Shape | First timestamp | Last timestamp | Duration under µs interpretation | Effective sample rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `(10247, 7)` | 1720476254890600 | 1720476305894185 | 51.003585 s | 200.888 Hz |
| 1 | `(11812, 7)` | 1720476332594540 | 1720476391386683 | 58.792143 s | 200.894 Hz |
| 2 | `(10158, 7)` | 1720476418482738 | 1720476469059301 | 50.576563 s | 200.824 Hz |
| 3 | `(11651, 7)` | 1720476494662869 | 1720476552591235 | 57.928366 s | 201.110 Hz |

Verified from: `0_ring_0.bin`, `1_ring_0.bin`, `2_ring_0.bin`, and
`3_ring_0.bin`

The upstream source does not declare an expected ring sampling frequency.
Approximately 201 Hz is an observation from these files, conditional on the
microsecond timestamp interpretation, not a declared device rate.

Ring timestamps are stored in the final float64 column. They are
integer-valued, large absolute-looking values and are nondecreasing, but not
strictly increasing. Dataset `0`, for example, has 10,247 rows but only 3,099
unique timestamps; consecutive groups contain 1 to 13 rows. A strict
monotonicity requirement would reject valid sample data.

Verified from: all four `*_ring_0.bin` files

The values align with the board and text-marker timestamp domain. Board
generation explicitly uses wall-clock microseconds, but no ring acquisition
writer is included. Treat the ring values operationally as microseconds for
duration/alignment while retaining the ring timestamp unit/origin as not
independently documented.

For a nonempty stream with `N` rows:

```text
duration_seconds = (timestamp[-1] - timestamp[0]) / 1_000_000
sample_rate_hz = (N - 1) / duration_seconds
```

This duration-based rate is preferable to `1 / median(diff(timestamp))`
because the sample contains many zero differences. The timestamps are not
relative: the stored values are not reset to zero at a recording start.

The upstream `IMUData.scale()` divides acceleration fields by `9.8` and
converts gyro fields with `value / pi * 180`, with sign flips on selected
axes. Those transformations suggest conventional physical quantities, but
the upstream source does not explicitly state the raw acceleration or gyro
units. Their physical units must remain marked unknown.

Source: `vendor/WritingRing/core/imu_data.py`, `IMUData.scale`, lines 49-51

A malformed ring file cannot be reshaped if its float count is not divisible
by seven; the author's reshape raises `ValueError`. Explicit checks for file
existence, byte length divisible by 8, value count divisible by 7, finite
values, a nonempty stream, and nondecreasing timestamps are recommended before
future plotting.

## 4. Board chunk format

Board chunks are gzip-compressed Python pickle files. The inspected chunks
begin with gzip magic bytes `1f 8b`; after decompression,
`0_board_0.gz` and `0_board_1.gz` begin with pickle protocol-4 bytes
`80 04`. The writer passes a list to `compress_pickle.dump`, and the author
loads each chunk with:

```python
frames = compress_pickle.load(path)
```

Source: `vendor/WritingRing/core/sensel_lib/board.py`,
`compress_dump_to_file`, lines 15-17  
Source: `vendor/WritingRing/board_plot.py`, `load_data`, lines 8-9  
Verified from: `0_board_0.gz` and `0_board_1.gz`

| Object | Python type | Confirmed fields | Meaning |
| --- | --- | --- | --- |
| Loaded container | `builtins.list` | list operations | Ordered frame objects within one chunk |
| Frame | `core.sensel_lib.frame_data.FrameData` | `force_array`, `timestamp`, `contacts` | One board scan |
| Force array | `numpy.ndarray` | shape `(105, 185)`, dtype `float64` | Dense board pressure/force grid |
| Contacts container | `builtins.list` | zero or more contacts | Contacts detected in the frame |
| Contact | `core.sensel_lib.frame_data.ContactData` | fields listed below | One tracked contact |

Verified from: `0_board_1.gz`

The contact instance fields and actual scalar Python types are:

| Field | Python type | Upstream role |
| --- | --- | --- |
| `id` | `int` | Sensel contact ID |
| `state` | `int` | 1=start, 2=move, 3=end |
| `x`, `y` | `float` | Stored normalized coordinates |
| `area` | `float` | Sensel area value; unit undocumented |
| `force` | `float` | Sensel `total_force`; unit undocumented |
| `major`, `minor` | `float` | Contact axes; units undocumented |
| `delta_x`, `delta_y` | `float` | Sensel coordinate deltas; units undocumented |
| `delta_force`, `delta_area` | `float` | Sensel deltas; units undocumented |
| `label` | `int` | Writer supplies `0`; class documents -1/0/1 meanings |
| `frame_id` | `int` | Board runner's frame counter at contact creation |

Source: `vendor/WritingRing/core/sensel_lib/frame_data.py`, class
`ContactData`, lines 3-29  
Source: `vendor/WritingRing/core/sensel_lib/board.py`, `Board._run`, lines
150-164  
Verified from: first contact in `0_board_1.gz`

Frames with no detected contact are normal `FrameData` objects whose
`contacts` field is an empty list. A whole chunk may also be empty:
every sample dataset's `{dataset_id}_board_0.gz` loads as `[]`. In
`0_board_1.gz`, 577 of 1,000 frames have no contacts.

Verified from: `0_board_0.gz`, `0_board_1.gz`, `1_board_0.gz`,
`2_board_0.gz`, and `3_board_0.gz`

The writer's normal maximum chunk size is 1,000 frames. Current matching
sequences have 1,000-frame intermediate chunks and a shorter final chunk.
Within every nonempty inspected chunk, frame timestamps are strictly
increasing. Across the temporally matching sequences, boundary deltas are
positive and approximately 6-9 ms.

Source: `vendor/WritingRing/core/sensel_lib/board.py`, `save_frame`, lines
19-44  
Verified from: all 39 chunks under `data_sample/data/user_0/0`

The observed sequences are:

| Dataset | Empty chunk | Sequence overlapping ring | Frames in overlapping sequence | Observed boundary behavior |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 1-6 | 5,701 | continuous through 6; chunk 7 jumps backward 387,876,490 µs |
| 1 | 0 | 1-7 | 6,724 | strictly increasing |
| 2 | 0 | 1-6 | 5,631 | strictly increasing |
| 3 | 0 | 1-7 | 6,612 | strictly increasing |

Dataset `0` chunks `7..15` are internally and mutually continuous but precede
the ring recording and chunks `1..6`. Their filenames are contiguous, but
their contents are not part of the same chronological sequence as chunks
`1..6`.

Verified from: all `*_board_*.gz` and `*_ring_0.bin` files under
`data_sample/data/user_0/0`

## 5. Board coordinate and force interpretation

The board writer applies these storage transformations:

```text
x_stored = sensel_contact.x_pos / 230.0
y_stored = sensel_contact.y_pos / 130.0
force_stored = sensel_contact.total_force
force_array_stored = sensel_force_array * 0.2
```

Source: `vendor/WritingRing/core/sensel_lib/board.py`, `Board.MAX_X`,
`Board.MAX_Y`, and `Board._run`, lines 60-63 and 142-155

The board plotter uses `contact.x` unchanged and flips the vertical coordinate:

```text
x_display = x_stored
y_display = 1.0 - y_stored
color = contact.force
```

Source: `vendor/WritingRing/board_plot.py`, `visualize_touch_data`, lines
40-47

Thus stored and display `y` are distinct. The stored `x` and `y` are
dimensionless normalized coordinates as produced by the writer; the physical
units of the pre-normalized Sensel positions and the rationale/units for
constants `230.0` and `130.0` are not stated. Do not label them millimeters
without another source.

In dataset `0`, observed contacts have `x` from approximately `0.3012` to
`0.7125`, `y` from `0.1328` to `0.8388`, force from `3.0` to `280.875`,
states `1, 2, 3`, and IDs `0, 1`. These are sample ranges, not format bounds.

Verified from: all dataset `0` board chunks

Contact `id` is copied directly from the Sensel contact. The class describes
it as a unique contact/keystroke integer with an upper limit of 16. Frame
timestamps are assigned separately at scan time; `frame_id` is attached to
contacts but is not a field on `FrameData`.

Source: `vendor/WritingRing/core/sensel_lib/frame_data.py`, class
`ContactData`, lines 5-15  
Source: `vendor/WritingRing/core/sensel_lib/board.py`, `Board._run`, lines
138-164

## 6. Timestamp handling

### Ring timestamps

The final float64 column is used directly for nearest-marker matching. Values
are nondecreasing with duplicate groups and are not reset to zero. No ring
writer is included, so its clock source and unit are not independently
declared upstream.

Source: `vendor/WritingRing/ring_plot.py`, `plot_ring_data`, lines 7-17  
Verified from: all four `*_ring_0.bin` files

### Board timestamps

The board writer records `int(time.time() * 1e6)`, an integer wall-clock
timestamp in microseconds. Loaded sample values are strictly increasing within
each valid sequence. The `FrameData` docstring says `time.perf_counter`, but
the executed writer code uses `time.time`; the writer and actual epoch-scale
values are the stronger evidence.

Source: `vendor/WritingRing/core/sensel_lib/board.py`, `Board._run`, lines
133-148  
Source: `vendor/WritingRing/core/sensel_lib/frame_data.py`, `FrameData`
docstring, lines 56-70  
Verified from: all sample board chunks

The author defines `Board.FPS = 50`, but the inspected matching sequences are
approximately 130.6-131.0 frames/s by timestamp duration. The constant alone
must not be presented as the observed data rate.

Source: `vendor/WritingRing/core/sensel_lib/board.py`, `Board`, lines 60-63
and `_sync`, lines 128-131  
Verified from: matching board sequences for datasets `0` through `3`

### Timestamp text files

Each inspected file has 26 lines:

```text
{integer_timestamp} {single_character_label}
```

The timestamps are strictly increasing. Dataset `0` uses lowercase labels;
dataset `1` uses the uppercase version in the same order. Datasets `2` and `3`
similarly form a lowercase/uppercase pair with a different character order.

Verified from: `0_timestamp.txt`, `1_timestamp.txt`, `2_timestamp.txt`, and
`3_timestamp.txt`

The ring plotter derives the text path from `ring_0`, reads the first token as
`float`, finds the nearest ring timestamp for each marker, and plots a vertical
line at that row index. The board plotter reads marker timestamps as `int`,
selects two adjacent markers, and filters board frames inclusively between
them.

Source: `vendor/WritingRing/ring_plot.py`, `plot_ring_data`, lines 9-27  
Source: `vendor/WritingRing/board_plot.py`, `plot_board_data`, lines 71-98
and `visualize_touch_data`, lines 33-43

All text markers fall inside their matching `ring_0` time range. Only 23 of 26
markers fall inside each matching board sequence because all four nonempty
board sequences begin after the first three markers. Dataset `0` is further
complicated by the older chunk tail described above.

Verified from: all ring, timestamp, and board files under
`data_sample/data/user_0/0`

Ring, board, and marker values are numerically comparable at microsecond scale,
and the author uses the marker file independently against both streams.
However, the upstream source contains no explicit ring/board clock
synchronization or offset correction. It is not verified that ring and board
acquisition used exactly the same clock origin; do not infer tighter
synchronization from similar durations or timestamp magnitudes.

## 7. Upstream data-loading flow

The author supplies two separate plotting flows, not one combined recording
loader.

Ring flow:

```text
receive a ring_0 path
→ np.fromfile(..., dtype=np.float64)
→ reshape to (-1, 7)
→ derive the matching timestamp.txt path by string replacement
→ read marker timestamps
→ map each marker to the nearest ring timestamp row
→ plot the first six columns
```

Source: `vendor/WritingRing/ring_plot.py`, `plot_ring_data`, lines 6-35

Board flow:

```text
select a user
→ enumerate action directories
→ randomly select one timestamp file in an action directory
→ randomly select one adjacent marker interval
→ glob every *.gz file in that action directory
→ compress_pickle.load each file in glob order
→ filter frames to the marker interval
→ extract x, 1-y, and force from contacts
→ scatter plot
```

Source: `vendor/WritingRing/board_plot.py`, `plot_board_data` and
`visualize_touch_data`, lines 23-98

The upstream board flow neither restricts gzip files to the selected timestamp
file's dataset ID nor sorts them. Matching by dataset prefix and sorting
numerically are intentional requirements of the new project, not behavior
already implemented by the upstream plotter.

## 8. File ordering rules

For one nominal recording, first select only:

```text
{dataset_id}_board_{chunk_index}.gz
```

Then parse `chunk_index` as an integer and sort by that integer in ascending
order. Do not sort by the whole filename string.

Lexical sorting places `board_10` immediately after `board_1` and before
`board_2`. Actual dataset `0` demonstrates this:

```text
lexical:  board_0, board_1, board_10, ..., board_2, ...
numeric:  board_0, board_1, board_2, ..., board_10, ...
```

The numeric rule follows the writer's integer `segment` counter.

Source: `vendor/WritingRing/core/sensel_lib/board.py`, `save_frame`, lines
19-44  
Verified from: filenames under `data_sample/data/user_0/0`

Numeric sorting establishes intended segment order but does not prove temporal
continuity. Validate timestamps after sorting; dataset `0` has a backward jump
at the otherwise numerically consecutive `6 -> 7` boundary.

## 9. Validation rules

The classifications below distinguish behavior imposed by upstream code,
properties merely observed in this sample, and new defensive checks.

| Validation | Classification | Evidence or recommended behavior |
| --- | --- | --- |
| Ring value count divisible by 7 | Required by upstream code | `reshape(-1, 7)` raises otherwise |
| Ring file decodes as float64 | Required by upstream code | `np.fromfile(..., dtype=np.float64)` |
| Ring values are finite | Observed from actual data | All eight sample ring files are finite |
| Ring timestamps are nondecreasing | Observed from actual data | Accept duplicates; all `ring_0` samples contain them |
| Ring file exists and is nonempty | Additional defensive validation proposed | Report a clear recording/path error before `fromfile`/indexing |
| Ring byte length divisible by 8 | Additional defensive validation proposed | Reject partial float64 values |
| Timestamp file exists | Required by upstream ring code | `open()` fails; upstream board flow instead skips actions with no timestamp file |
| Timestamp rows contain a numeric timestamp and label | Required by upstream code | Both plotters index the first token; board plotter also indexes the second |
| Text-marker timestamps are strictly increasing | Observed from actual data | True for all four sample files |
| At least one board chunk exists | Additional defensive validation proposed | Report a missing recording component |
| Chunk indices are parsed and sorted numerically | Additional defensive validation proposed/project requirement | Upstream board glob is unsorted |
| Missing or duplicate chunk indices | Additional defensive validation proposed | Report explicitly; names are contiguous in the sample |
| Empty chunk list | Observed from actual data | Treat as valid-but-empty; every `board_0` is `[]` |
| Board deserialization succeeds | Required by upstream code | Upstream catches per-file exceptions and continues |
| Loaded top-level object is a list of frames | Additional defensive validation proposed | Confirmed for inspected chunks; reject or clearly report incompatible shapes |
| Expected frame/contact fields exist | Additional defensive validation proposed | Validate fields listed in section 4 before extraction |
| Frames without contacts | Observed from actual data | Valid; represent as `contacts == []`, not an error |
| Force array shape/dtype | Observed from actual data | All dataset `0` frames inspected were `(105, 185)` float64; warn or fail clearly if incompatible |
| Within-chunk timestamp monotonicity | Observed from actual data and defensive validation proposed | All nonempty chunks are strict; validate at load time |
| Cross-chunk timestamp continuity | Additional defensive validation proposed | Detect backward jumps, duplicates, and unexpectedly large gaps |
| Chunk time range overlaps the nominal ring/marker range | Additional defensive validation proposed | Dataset `0` proves filename matching is insufficient |
| Unexpected fields | Additional defensive validation proposed | Permit documented fields; warn or fail intentionally for schema drift |

Source: `vendor/WritingRing/ring_plot.py`, lines 6-17  
Source: `vendor/WritingRing/board_plot.py`, lines 26-49 and 71-84  
Verified from: all sample files under `data_sample/data/user_0/0`

## 10. Source facts, sample observations, repository interpretation, and policy

### Source-format facts and sample observations

The following list intentionally combines source-backed format facts with
observations explicitly marked as sample-specific. Repository policies appear
in their own section below.

- Ring files are read as native-endian NumPy float64 and reshaped to seven
  columns.
- Ring columns are acceleration x/y/z, gyro x/y/z, then timestamp.
- The author plots `ring_0`; the physical meaning of either ring index is not
  documented.
- Board chunks are gzip-compressed protocol-4 Python pickles loaded by
  `compress_pickle.load`.
- A loaded chunk is a list of `FrameData`; contacts are lists of
  `ContactData`.
- Board frames contain a `(105, 185)` float64 force array, integer timestamp,
  and contact list in the inspected sample.
- Stored board coordinates divide Sensel x/y positions by `230.0`/`130.0`;
  the plotter displays `y` as `1-y`.
- Board timestamps are wall-clock microseconds from `time.time() * 1e6`.
- Board chunks are normally capped at 1,000 frames and are named with an
  incrementing integer segment.
- Every sample `board_0` is empty.
- Dataset `0` has a backward timestamp discontinuity between chunks 6 and 7;
  chunks 7-15 do not overlap that dataset's ring/marker interval.
- Root and vendor `core` source files match; root contains only an additional
  `__pycache__` directory.

### Still unknown

- The physical meanings/locations of ring indices `0` and `1`.
- Why the upstream plots only `ring_0`.
- The independently documented byte order and clock source for ring files.
- The nominal ring sampling frequency; approximately 201 Hz is observed only.
- Explicit physical units for ring acceleration and gyro values.
- Explicit physical units for board coordinates before normalization, force,
  force-array values, area, axes, and deltas.
- The semantic meaning of each action ID.
- Why all four chunk-zero files are empty and why nonempty board capture starts
  after the first three markers.
- Whether dataset `0` chunks `7..15` should be discarded, quarantined, or
  exposed as a separate anomalous sequence.
- Whether ring, board, and marker writers used exactly the same wall clock and
  what synchronization guarantees, if any, existed.
- Why the `FrameData` docstring mentions `time.perf_counter` while the writer
  uses `time.time`.
- Why `Board.FPS` is 50 while the sample board sequences are approximately
  131 frames/s.

### Repository interpretations and policies

- Discover recordings by user/action/dataset-ID filename membership.
- Load only `ring_0`; record `ring_1` as ignored metadata rather than treating
  it as malformed.
- Select matching board files by dataset prefix and sort on the parsed integer
  chunk index.
- Preserve both `y_raw` and `y_display = 1-y_raw` so coordinate transforms are
  explicit and reversible.
- Preserve frame-level force arrays separately from contact-level records.
- Treat empty frames and empty chunks as valid data conditions.
- Validate timestamps at every chunk boundary and prominently report backward
  jumps. Do not silently drop dataset `0`'s older chunks until an anomaly
  policy is chosen.
- Use duration-based ring sampling-rate estimates and accept nondecreasing
  timestamps with duplicates.
- Surface missing timestamp files and discontinuous/missing chunks explicitly;
  the exact warning-versus-error policy remains an implementation decision.

### Policy choices that differ from upstream or sample behavior

- `TASK.md` requires matching and numerically ordered board chunks. The
  upstream board plotter instead globs all action-level gzip files without
  filtering by dataset ID or sorting. The task requirement is an intentional
  correction.
- A simple assumption that numerically consecutive chunks form one continuous
  recording conflicts with actual dataset `0`, which jumps backward at chunk
  7.
- Ignoring `ring_1` matches the only supplied ring plotting path and the
  project instruction, but the upstream source does not explicitly explain or
  mandate the omission.
- No upstream source confirms an expected ring frequency. The sample supports
  only an observed approximately 201 Hz effective rate under the microsecond
  interpretation.

## 11. Derived gravity-removal and preprocessing data is not part of the file format

`src/writingring/gravity.py` provides an optional offline analysis above the
raw seven-column Ring loader. It does not change or extend the binary format.
The raw `ring_0.bin` row remains the seven-value source record:

```text
acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, timestamp
```

The following names are derived in memory and are never decoded as named
fields from `ring_0.bin`:

```text
acc_body_x, acc_body_y, acc_body_z
angular_velocity_body_x_rad_s
angular_velocity_body_y_rad_s
angular_velocity_body_z_rad_s
gravity_body_x, gravity_body_y, gravity_body_z
linear_acc_body_x, linear_acc_body_y, linear_acc_body_z
gravity_correction_used
gravity_correction_confidence
```

`preprocess_ring_imu()` builds a separate, fixed nine-channel derived schema
for segmentation:

```text
acceleration_x_g, acceleration_y_g, acceleration_z_g,
acceleration_x, acceleration_y, acceleration_z,
gyro_x, gyro_y, gyro_z
```

The first acceleration triplet is always the selected preprocessing result in
g, and the second is the same result in m/s². The relationship is
`acceleration_m_s2 = acceleration_g * 9.80665`. With the `raw` preprocessing
method, the selected acceleration is the measured Ring acceleration and still
contains gravity. With `low-pass`, `madgwick`, or
`xylo-rotate-and-remove-gravity`, it is a processed gravity-removed result.
The gyro columns likewise represent the method's selected gyro output, not a
new field in the source binary.

The segmentation exporter writes those nine channels to `rawIMU.npy` for
compatibility. That file is a contiguous aggregation of selected samples and
does not contain the source timestamp column; `segment_offsets.npy`, the
segment manifest, and the source Ring data retain the boundary/time
information separately. The name `rawIMU.npy` therefore does not imply that
its acceleration columns are raw measurements.

The lower-level gravity configuration defaults to a nominal 200 Hz rate,
Madgwick processing, acceleration in `m/s^2`, gyroscope values in `rad/s`,
the identity axis transform, and expected gravity of `9.80665 m/s^2`. The
public preprocessing CLI instead defaults to low-pass processing. Both are
repository processing defaults, not acquisition-rate guarantees. The stationary
search default proposes the best passing 0.10-second interval from robust
acceleration and gyro metrics; callers may override bounds explicitly. The
identity transform means raw Ring sensor axes; a physical sensor-to-ring
mounting transform remains undocumented.

The opt-in `upstream_suggested` profile is based on
`IMUData.scale()`—acceleration divided by `9.8`, raw gyroscope treated as
radians/second for fusion, and y/z sign flips. It remains an assumption rather
than confirmed file metadata. Raw `acc_*`, `gyr_*`, and `timestamp` columns
are preserved unchanged.

The default Madgwick IMU-only method uses a configurable beta gain of `0.1`.
Because Ring timestamps contain duplicate groups, it uses
`dt = 1 / sampling_rate_hz` and does not derive per-sample integration steps
from the timestamp column. It anchors the gravity contribution in an
explicitly validated stationary interval and propagates forward and backward.
Its output is therefore noncausal, estimated, and suitable for offline
inspection rather than navigation or real-time control. The alternative
low-pass method applies a causal second-order Butterworth IIR SOS/biquad to
each acceleration axis, with a configurable cutoff of `0.2 Hz` by default.

This analysis intentionally differs from the upstream implementation:
upstream scales and plots IMU channels but does not estimate orientation,
run Madgwick fusion, low-pass acceleration for gravity estimation, propagate
a body-frame gravity contribution, or subtract it.

## 12. Alignment exports are derived artifacts

Alignment TXT, report, and verification PNG artifacts are downstream
processing outputs; they do not alter Ring, Board, or marker files. Canonical
Ring timestamps remain source/provenance values. Matching uses a separate
endpoint-reconstructed work axis, and its endpoint-derived effective rate is
not a device-acquisition claim.

The mapping convention is `ring_timestamp_us = board_timestamp_us + offset_us`.
Current schema-v2 output declares the offset domain: `offset_us` is the
work-axis offset when `offset_domain=alignment_work_axis`; a separately
reported `canonical_offset_us` exists only after successful canonical
projection. A successful work-axis alignment may therefore publish a
declared-domain artifact even when canonical projection is not representable.
Consumers must honor the declared domain rather than treating every exported
offset as canonical. See [ALIGNMENT_OUTPUTS.md](ALIGNMENT_OUTPUTS.md) for the
full output contract and CLI.
