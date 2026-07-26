Read `AGENTS.md` and `TASK.md` first.

Before writing or modifying any implementation code, inspect the dataset author's original code under `vendor/WritingRing`.

Treat the upstream code as the primary reference for understanding the dataset format, but verify important assumptions against the actual sample data.

Read these files completely:

* `vendor/WritingRing/ring_plot.py`
* `vendor/WritingRing/board_plot.py`
* `vendor/WritingRing/core/imu_data.py`
* `vendor/WritingRing/core/window.py`
* `vendor/WritingRing/core/sensel_lib/frame_data.py`

Read these files only if they are referenced or needed to understand the data structures:

* `vendor/WritingRing/core/sensel_lib/board.py`
* `vendor/WritingRing/core/sensel_lib/sensel.py`
* `vendor/WritingRing/core/sensel_lib/sensel_register_map.py`

Also perform the following inspection:

1. Run:

```bash
diff -qr core vendor/WritingRing/core
```

Report whether the root `core/` directory differs from the upstream copy.

1. List and categorize the files under:

```text
data_sample/data/user_0/0
```

1. Inspect at least one `*_ring_0.bin` file using the exact loading method used by the author.

2. Inspect at least one `*_ring_1.bin` file only to determine how it differs from `ring_0`, but do not use it in the planned implementation unless the author code explicitly requires it.

3. Load at least one `*_board_0.gz` file using the exact method used by the author.

4. Print and record the actual Python types and available attributes for:

* the object returned by `compress_pickle.load`;
* one board frame;
* the frame's contacts container;
* one contact;
* any other nested object needed to interpret the data.

1. Inspect more than one consecutive board chunk, such as:

```text
0_board_0.gz
0_board_1.gz
```

Determine whether frame timestamps and ordering continue correctly across chunk boundaries.

1. Inspect the corresponding `*_timestamp.txt` file and determine how, or whether, the upstream code uses it.

## Required documentation output

Create or update:

```text
docs/DATA_FORMAT.md
```

This file must serve as the durable technical reference for all future dataset-loading work.

Document the following sections.

### 1. Directory hierarchy

Describe the confirmed hierarchy:

```text
data/
└── user_{user_id}/
    └── {action_id}/
        └── recording files
```

Explain the meanings of:

* user ID;
* action ID;
* dataset ID;
* board chunk index;
* ring index.

### 2. Recording membership rules

Explain exactly how files are grouped into one recording.

For example, document whether these files belong to the same recording:

```text
1_ring_0.bin
1_ring_1.bin
1_timestamp.txt
1_board_0.gz
1_board_1.gz
...
```

Explicitly document:

* which filename component represents the dataset ID;
* how board chunks are associated with a dataset ID;
* which ring file should be used;
* whether `ring_1.bin` should be ignored and why.

### 3. Ring binary format

Document, based on the upstream source and actual sample data:

* binary dtype;
* byte order, if identifiable;
* number of values per sample;
* reshape operation;
* exact column order;
* timestamp column location;
* expected sampling frequency;
* timestamp behavior;
* whether timestamps are absolute or relative;
* whether timestamps are measured in seconds;
* how duration and sampling frequency should be computed;
* validation conditions for malformed files;
* any units explicitly stated by the author;
* any units that remain undocumented.

Include a concise reference loader example derived from the upstream behavior, but do not copy unnecessary plotting code.

Example form:

```python
raw = np.fromfile(path, dtype=...)
data = raw.reshape(-1, ...)
```

### 4. Board chunk format

Document:

* compression and serialization format;
* exact loading function;
* type of the top-level loaded object;
* whether it is a list, array, iterator, or another container;
* frame object type;
* frame attributes;
* contact container type;
* contact object type;
* contact attributes;
* how empty-contact frames are represented;
* whether board chunks form one continuous recording;
* required chunk ordering;
* behavior at chunk boundaries.

Include a table such as:

| Object           | Python type | Confirmed fields | Meaning |
| ---------------- | ----------- | ---------------- | ------- |
| Loaded container | ...         | ...              | ...     |
| Frame            | ...         | ...              | ...     |
| Contact          | ...         | ...              | ...     |

### 5. Board coordinate and force interpretation

Document exactly how the author uses:

* `x`;
* `y`;
* `force`;
* contact ID;
* frame timestamp.

Explicitly identify every coordinate transformation.

For example, determine whether the display coordinate is:

```python
y_display = 1.0 - y_raw
```

Clearly distinguish:

* stored coordinate;
* transformed display coordinate;
* documented physical unit;
* unknown or undocumented unit.

### 6. Timestamp handling

Document separately:

* ring timestamp behavior;
* board timestamp behavior;
* `timestamp.txt` contents;
* units, where confirmed;
* timestamp origins, where confirmed;
* monotonicity;
* continuity across board chunks;
* whether ring and board timestamps share the same origin and scale;
* whether synchronization is explicitly implemented by the author.

Do not infer synchronization merely because durations appear similar.

### 7. Upstream data-loading flow

Summarize the author's actual loading flow in order.

For example:

```text
select user/action/dataset ID
→ locate ring file
→ read binary ring samples
→ reshape ring data
→ locate matching board chunks
→ sort board chunks
→ deserialize frames
→ extract contacts
→ apply coordinate transformation
→ filter or plot by timestamp
```

Only include steps confirmed by the code.

### 8. File ordering rules

Document how filenames must be sorted.

Explain why normal lexical sorting is incorrect for:

```text
board_1
board_2
board_10
```

Record the exact numerical sorting rule that the new implementation should use.

### 9. Validation rules

Record recommended validation checks derived from the upstream code and actual data, including:

* ring file length divisible by the sample width;
* finite values;
* timestamp monotonicity;
* missing ring file;
* missing timestamp file;
* missing board chunks;
* missing board chunk indices;
* board deserialization failures;
* frames without contacts;
* unexpected object fields;
* timestamp discontinuities between chunks.

Label each validation rule as one of:

* required by upstream code;
* observed from actual data;
* additional defensive validation proposed for the new implementation.

### 10. Confirmed facts, unknowns, and design decisions

Maintain three explicit lists:

#### Confirmed facts

Facts directly supported by either:

* upstream source code; or
* actual sample-data inspection.

#### Still unknown

Items that could not be confirmed, such as undocumented physical units or timestamp origins.

#### Proposed implementation decisions

Choices for the new project that are not necessarily made by the upstream author, such as:

* using Pandas DataFrames;
* retaining both `y_raw` and `y_display`;
* returning frame-level and contact-level tables;
* producing warnings instead of failing on a missing timestamp file.

Do not present proposed decisions as properties of the original dataset.

## Evidence requirements

For every important data-format claim in `docs/DATA_FORMAT.md`, include its evidence source.

Use one of these formats:

```text
Source: vendor/WritingRing/ring_plot.py, function or relevant line range
```

```text
Source: vendor/WritingRing/core/sensel_lib/frame_data.py, class FrameData
```

```text
Verified from: data_sample/data/user_0/0/0_board_0.gz
```

When possible, include:

* source file path;
* class or function name;
* relevant line range;
* sample file used for verification.

Do not write unsupported conclusions.

## Progress tracking

Update `PROGRESS.md` separately.

`PROGRESS.md` should contain only:

* which inspection steps were completed;
* the files inspected;
* unresolved issues;
* the next task;
* commands run and their results.

Do not duplicate the full data-format documentation in `PROGRESS.md`. Instead, link to:

```text
docs/DATA_FORMAT.md
```

## Restrictions

* Treat `vendor/WritingRing` as read-only.
* Do not modify any file under `vendor/WritingRing`.
* Do not modify files under `data_sample`.
* Do not implement the new loaders or plotting code yet.
* Do not blindly copy the upstream scripts.
* Do not guess undocumented units.
* Do not assume ring and board timestamps are synchronized without evidence.
* Do not mark a fact as confirmed unless it is supported by source code or actual sample inspection.

Before finishing this inspection phase:

1. Confirm that `docs/DATA_FORMAT.md` has been created.
2. Summarize the most important confirmed format facts.
3. List all unresolved questions.
4. Identify any assumptions in `TASK.md` that conflict with the upstream implementation.
5. Update `PROGRESS.md` with the inspection status and next recommended implementation step.

## Completion checklist

Before finishing, verify:

* [ ] All required upstream files were read.
* [ ] Root and vendor core directories were compared.
* [ ] One ring_0 file was inspected.
* [ ] One ring_1 file was inspected for comparison only.
* [ ] At least two consecutive board chunks were inspected.
* [ ] timestamp.txt was inspected.
* [ ] Actual frame and contact fields were recorded.
* [ ] Ring column order was confirmed.
* [ ] Board chunk ordering was confirmed.
* [ ] Coordinate transformations were confirmed.
* [ ] Unknown units were clearly marked as unknown.
* [ ] docs/DATA_FORMAT.md was created or updated.
* [ ] PROGRESS.md was updated.
* [ ] No vendor or sample-data file was modified.
