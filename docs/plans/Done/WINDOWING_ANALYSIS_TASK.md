# WritingRing Original Windowing Analysis Task

## Objective

Analyze how the original WritingRing implementation under:

```text
vendor/WritingRing/
```

organizes, buffers, aligns, and windows sensor data.

The goal is to produce a complete technical report explaining exactly how the original authors convert raw board and ring data into windowed data structures.

This is an investigation and documentation task only.

Do not modify the original source files under `vendor/WritingRing/`.

---

## Primary Scope

Inspect all relevant files under:

```text
vendor/WritingRing/
├── board_plot.py
├── ring_plot.py
└── core
    ├── imu_data.py
    ├── window.py
    └── sensel_lib
        ├── board.py
        ├── frame_data.py
        ├── sensel.py
        └── sensel_register_map.py
```

Pay particular attention to:

```text
vendor/WritingRing/core/window.py
vendor/WritingRing/core/imu_data.py
vendor/WritingRing/board_plot.py
vendor/WritingRing/ring_plot.py
```

However, do not assume that windowing is implemented only in `window.py`.

Search the entire `vendor/WritingRing/` directory for all code that:

* imports `Window`
* creates a `Window` object
* calls `push()`
* calls `head()`, `tail()`, `first()`, `last()`, or `get()`
* slices arrays or lists to create fixed-length segments
* buffers sensor samples
* maintains rolling histories
* synchronizes board and ring measurements
* groups data by timestamps
* performs interpolation, resampling, padding, truncation, or alignment
* constructs model inputs or visualization windows

---

## Project instructions and existing documentation

* `AGENTS.md`
* `TASK.md`
* `PROGRESS.md`
* `README.md`, if it exists
* `docs/notes/DATA_FORMAT.md`

## Required Investigation

### 1. Window class implementation

Explain the complete behavior of the `Window` class in:

```text
vendor/WritingRing/core/window.py
```

Document:

* constructor arguments
* internal data structure
* meaning of `window_length`
* behavior when the number of elements exceeds `window_length`
* whether old or new samples are discarded
* behavior of `push()`
* behavior of `clear()`
* behavior of `first()`
* behavior of `last()`
* behavior of `get()`
* behavior of `head()`
* behavior of `tail()`
* behavior of `capacity()`
* behavior of `empty()`, if present
* whether the class enforces type, shape, timestamp, or sampling-rate consistency
* whether returned `head()` and `tail()` windows share data or contain copies
* edge cases, such as empty windows or requests longer than the current capacity

Determine whether this class should technically be described as:

* a sliding window
* a rolling buffer
* a bounded FIFO queue
* a ring buffer
* or another data structure

Explain the choice precisely.

---

### 2. Complete call-site analysis

Find every use of `Window` in `vendor/WritingRing/`.

For every call site, report:

* file path
* function or code block
* approximate line range
* object name
* configured window length
* type of data stored
* how frequently data is pushed
* what causes the window to be cleared
* how the contents are later consumed
* whether the window must be full before being used
* whether overlapping windows are produced
* whether a stride is defined
* whether timestamps are stored with samples

Create a table similar to:

| File | Window object | Length | Stored data | Push trigger | Consumer | Full-window requirement |
| ---- | ------------: | -----: | ----------- | ------------ | -------- | ----------------------- |

If a `Window` class exists but is not used anywhere, state that explicitly and provide evidence.

---

### 3. Ring data path

Trace the complete ring-data flow from input to final use.

At minimum, inspect:

```text
vendor/WritingRing/ring_plot.py
vendor/WritingRing/core/imu_data.py
```

Explain:

1. How ring binary data is read.
2. The binary record structure.
3. How raw bytes are converted into sensor values.
4. Which IMU channels are available.
5. Whether timestamps are contained in each ring record.
6. Whether timing is inferred from:

   * sample index,
   * fixed sampling rate,
   * external timestamp files,
   * packet timestamps,
   * or another mechanism.
7. Whether samples are accumulated in a `Window`.
8. The window length, if any.
9. Whether windows overlap.
10. Whether any preprocessing occurs before or after windowing.
11. How windowed ring data is used by plotting or downstream processing.

Provide a step-by-step pipeline such as:

```text
ring binary file
→ binary decoding
→ raw IMU fields
→ scaling or conversion
→ timestamp construction
→ buffering
→ window extraction
→ plotting or downstream use
```

Replace this example with the actual pipeline found in the code.

---

### 4. Board data path

Trace the complete board-data flow from input to final use.

At minimum, inspect:

```text
vendor/WritingRing/board_plot.py
vendor/WritingRing/core/sensel_lib/frame_data.py
vendor/WritingRing/core/sensel_lib/board.py
```

Explain:

1. How `.gz` board files are opened and decoded.
2. Whether each file contains:

   * one board frame,
   * multiple board frames,
   * contact records,
   * force matrices,
   * or another structure.
3. How frame timestamps are represented.
4. How contacts are extracted.
5. Whether board frames are placed into a rolling window.
6. Whether contacts from multiple frames are grouped into temporal windows.
7. Whether force, position, area, and timestamp are retained.
8. Whether missing frames are handled.
9. Whether board windows overlap.
10. How board data is used by visualization or downstream processing.

Provide the exact pipeline:

```text
board .gz files
→ decompression
→ frame decoding
→ contact extraction
→ timestamp processing
→ buffering/windowing
→ plotting or downstream use
```

---

### 5. Relationship between file numbering and windowing

Investigate filenames such as:

```text
0_board_0.gz
0_board_1.gz
...
0_board_15.gz

0_ring_0.bin
0_ring_1.bin
0_timestamp.txt
```

Determine:

* what the first number means
* what the board-file suffix means
* what the ring-file suffix means
* whether one file corresponds to one window
* whether one file corresponds to one recording chunk
* whether files must be concatenated before windowing
* whether boundaries between files affect windows
* whether the window state is reset between files
* whether board and ring chunk indexes correspond directly

Do not infer these relationships only from filenames. Verify them using code.

If the code does not establish a relationship, clearly distinguish:

* directly confirmed behavior
* likely interpretation
* unresolved behavior

---

### 6. Time synchronization and alignment

Determine whether the original implementation aligns board and ring data.

Investigate:

* `*_timestamp.txt`
* timestamps inside board frames
* inferred timestamps for ring samples
* host/computer timestamps
* sensor/device timestamps
* chunk start and end times

Answer:

1. Are board and ring streams placed on a common time axis?
2. Is synchronization performed before windowing or after windowing?
3. Is interpolation used?
4. Is nearest-neighbor matching used?
5. Is one stream cropped to match the other?
6. Is there an explicit synchronization offset?
7. Are timestamp files only metadata, or are they used in calculations?
8. Can one board window be associated with one ring window?
9. What assumptions are made about sampling rate and clock stability?
10. What synchronization limitations exist?

If no synchronization is implemented in the vendor code, state that directly.

---

### 7. Window semantics

For each window or buffer discovered, determine:

* window length in samples
* approximate duration in seconds, if sampling rate is known
* input sampling rate
* output/update rate
* stride
* overlap
* initialization behavior
* warm-up period
* boundary behavior
* final incomplete-window behavior
* whether windows are sample-based or time-based
* whether windows contain raw data or processed features
* whether windows are generated offline or maintained during streaming
* whether windows are used for plotting only or model inference

Where possible, calculate:

```text
window duration = window length / sampling frequency
overlap = window length - stride
overlap ratio = overlap / window length
```

Do not calculate duration when the sampling frequency is not supported by the code or documentation.

---

### 8. Distinguish buffering from dataset segmentation

Explicitly separate the following concepts:

1. Temporary buffering during file reading.
2. Rolling history used for visualization.
3. Sliding windows used as machine-learning inputs.
4. Recording chunks represented by separate files.
5. Trial-level or session-level segmentation.
6. Timestamp-based board/ring alignment.

Do not call every list, file chunk, or buffer a “window.”

For each mechanism, explain what it actually represents.

---

### 9. Verify against sample data

Use the sample recording under:

```text
data_sample/data/user_0/0/
```

Only perform non-destructive inspection.

Use sample data to verify claims where practical, including:

* number of ring records per `.bin` file
* ring record size
* number of board frames per `.gz` file
* board timestamp ranges
* ring sample counts
* whether adjacent chunks are continuous
* whether timestamps reset at file boundaries
* whether chunk indexes correspond to time order
* whether board and ring durations are similar
* whether a supposed window length matches observed data

Prefer small inspection scripts or existing project utilities.

Do not rewrite or delete sample data.

Any temporary scripts should be placed outside `vendor/WritingRing/`, for example:

```text
scripts/
```

or:

```text
/tmp/
```

---

## Evidence Requirements

Every major conclusion must be supported by one or more of:

* source file path and line range
* function/class name
* relevant code excerpt
* observed sample-data result
* calculated result derived from code constants

Use short code excerpts only where they improve clarity.

Do not paste entire source files into the report.

Label conclusions using the following confidence categories:

* **Confirmed**: directly established by code or measured sample data.
* **Inferred**: strongly suggested, but not explicitly established.
* **Unresolved**: insufficient evidence in the vendor implementation.

---

## Required Report Structure

Write the final report to:

```text
docs/notes/VENDOR_WINDOWING_REPORT.md
```

Use the following structure.

# Vendor WritingRing Windowing Analysis

## 1. Executive Summary

Summarize:

* whether true sliding-window segmentation exists
* what the `Window` class actually does
* where it is used
* how ring data is buffered
* how board data is buffered
* whether board and ring data are aligned
* the most important limitations

## 2. Scope and Method

List:

* files inspected
* searches performed
* sample data inspected
* scripts or commands used

## 3. Terminology

Define:

* sample
* frame
* contact
* recording
* chunk
* buffer
* rolling window
* sliding window
* stride
* overlap

Use definitions appropriate to this codebase.

## 4. Window Class Analysis

Explain the full implementation of `core/window.py`.

Include pseudocode describing its update behavior.

## 5. Window Call-Site Inventory

Provide a complete call-site table.

## 6. Ring Data Pipeline

Describe raw file decoding through final consumption.

Include relevant data shapes, field types, and timing assumptions.

## 7. Board Data Pipeline

Describe `.gz` decoding through final consumption.

Include frame/contact structures and timing behavior.

## 8. File-Level Chunking

Explain the meaning of board, ring, and timestamp files.

Distinguish file chunks from algorithmic windows.

## 9. Time Synchronization

Explain whether and how board/ring streams are synchronized.

## 10. Window Parameters

Provide a summary table:

| Stream | Mechanism | Length | Sampling rate | Duration | Stride | Overlap | Reset condition |
| ------ | --------- | -----: | ------------: | -------: | -----: | ------: | --------------- |

Use `N/A` or `Unknown` rather than guessing.

## 11. Worked Example

Trace one concrete sample recording, preferably recording prefix `0`.

Show:

* files read
* sample/frame counts
* timestamps
* buffer updates
* extracted windows, if any
* final output

## 12. Edge Cases and Limitations

Discuss:

* empty buffers
* short recordings
* file boundaries
* timestamp gaps
* dropped samples
* missing board files
* different numbers of ring and board chunks
* inconsistent sampling rates
* lack of synchronization
* incomplete final windows

## 13. Confirmed Findings vs Inferences

Provide separate tables for:

* confirmed findings
* inferred interpretations
* unresolved questions

## 14. Implications for Our Reimplementation

Compare the vendor behavior with the current project implementation under:

```text
core/
src/writingring/
scripts/
```

Do not modify these files.

Explain:

* which vendor behaviors are already reproduced
* which behaviors differ
* which behavior should be preserved for compatibility
* which behavior should be improved
* what additional tests should be written

## 15. Recommended Next Steps

Provide prioritized recommendations for implementing a robust windowing pipeline.

Separate recommendations into:

* compatibility-preserving changes
* correctness fixes
* synchronization improvements
* machine-learning window generation
* testing and validation

## Appendix A. Source Evidence

Provide a concise mapping from conclusions to source paths and line ranges.

## Appendix B. Sample Data Measurements

Report measured counts, durations, timestamp ranges, and chunk boundaries.

---

## Additional Deliverable

Update:

```text
PROGRESS.md
```

Add a brief entry containing:

* investigation completed
* report path
* files analyzed
* main finding about whether vendor code implements true sliding windows
* unresolved questions

Do not rewrite unrelated content in `PROGRESS.md`.

---

## Constraints

* Do not modify any file under `vendor/WritingRing/`.
* Do not refactor code.
* Do not implement a new windowing system.
* Do not overwrite existing reports.
* Do not make unsupported assumptions.
* Do not treat plotting history as ML segmentation unless the code proves it.
* Do not rely only on class names.
* Trace actual data flow and call sites.
* Clearly distinguish source-code facts from interpretations.
* Preserve all existing sample data and outputs.

---

## Completion Criteria

The task is complete only when:

1. Every `Window` call site in `vendor/WritingRing/` has been identified.
2. Ring and board data paths have been traced separately.
3. File chunking has been distinguished from sliding-window segmentation.
4. Window length, stride, overlap, and reset behavior have been documented wherever available.
5. Board/ring synchronization behavior has been established.
6. Claims have source-code or sample-data evidence.
7. `docs/notes/VENDOR_WINDOWING_REPORT.md` has been created.
8. `PROGRESS.md` has been updated.
9. No vendor source file has been modified.
