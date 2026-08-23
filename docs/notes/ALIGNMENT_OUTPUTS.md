# Ring--Board alignment outputs

Alignment preserves the canonical Ring/SpikeIMU timestamp vector for
provenance and segmentation. It uses a separate, strictly increasing
**alignment work axis** for matching:

```text
work_axis = linspace(canonical_timestamp[0], canonical_timestamp[-1], N)
strategy = endpoint_reconstruction
```

This is the current behavior for both `raw-ring` and `spike-imu` inputs. The
feature sampling rate is provenance metadata; it does not determine work-axis
spacing. The reported effective work-axis rate is derived from the canonical
endpoints and sample count. Neither rate establishes an upstream acquisition
rate or timestamp-unit guarantee.

When a Board timestamp has its first backward jump, alignment considers only
the positional initial monotonic prefix. A valid press/lift pair is eligible
only if both endpoints are in that prefix. Events from a cross-boundary valid
pair are not alignment events, including the pre-jump press; invalid and
transient event rows remain audit diagnostics. Timestamp overlap from a later
epoch cannot extend the selected prefix.

## Timestamp and offset domains

```text
canonical timestamps      immutable source/provenance and segmentation axis
alignment work axis       endpoint-reconstructed matching coordinate
work-axis offset          matching result in that coordinate
canonical projection      separately computed offset/status, when representable
```

The mapping convention is:

```text
ring_timestamp_us = board_timestamp_us + offset_us
```

Current schema-v2 artifacts declare `offset_domain=alignment_work_axis`.
Their `offset_us` and `work_axis_offset_us` are work-axis values.
`canonical_offset_us` is present only when matched peaks can project the
work-axis result into canonical timestamps; the report records
`canonical_offset_projection_success` and its diagnostics separately.

Canonical timestamps are not overwritten to make matching strict. Projection
uses the canonical-minus-work differences at matched peak indices. Large
variation is first warned and then rejected according to the configured
constant-offset guard.

## Success and publication

The report is the final authoritative completion manifest. A completed
recording is exactly one of `SUCCESS` or `SKIPPED`: success has the offset TXT
and verification PNG declared by the report, while skipped has its sibling
skip JSON and no success artifacts. A `FAILED` report is diagnostic only and
is never a completed outcome. Consumers must use the Python outcome validator
with current input provenance, rather than infer completion from a TXT or JSON
file alone.

Only a `SUCCESS`/`SKIPPED` replacement is a completed-state transition and
requires explicit outcome-transition authorization. Publishing a new completed
outcome after `FAILED` follows the ordinary overwrite permission for the new
target artifacts; stale opposite artifacts are removed so the final manifest
is conflict-free. Malformed reports, artifacts, or skip diagnostics fail
through `AlignmentOutcomeError`.

The only validated skip reasons are:

```text
initial_interval_no_usable_pair
insufficient_valid_touch_pairs
insufficient_event_coverage
```

The initial-interval reason proves the backward-jump/prefix condition,
positive global valid-pair count, zero usable-prefix valid pairs, and coherent
prefix/jump frame diagnostics. The confidence-derived reasons instead carry
validated count, threshold, coverage, and failed-check diagnostics proving the
corresponding insufficient evidence. Every `SKIPPED` outcome publishes its
authoritative report and sibling skip JSON, and never publishes an offset TXT
or verification PNG.

A failed work-axis match retains its report but writes no offset TXT or
verification PNG. A successful work-axis match whose canonical projection
fails still writes its declared work-axis TXT, report, and verification PNG;
it is usable only through its stated `offset_domain`, not as a canonical
offset. Board-event segmentation honors `offset_domain` and
`boundary_offset_us` while retaining canonical feature timestamps for slicing
and provenance.

Reports and manifests expose the input identity and hashes, feature-rate
metadata, timestamp provenance, work-axis strategy/rate/sample count,
`best_offset_us`, `work_axis_offset_us`, `canonical_offset_us` when available,
the declared domain, and projection diagnostics/status. Legacy
canonical-domain offsets remain a compatibility input; they are not the
current schema-v2 CLI output contract.

The CLI report also records the `peak_detection` sample-window provenance.
Its `sampling_rate_hz` is the feature rate used for alignment, while the
three reported sample windows are scaled from the 200 Hz reference detector
configuration. Prominence thresholds and boundary fractions are unchanged by
this scaling.

## CLI

```bash
python scripts/align_ring_board.py \
  --data-root data_sample/data \
  --user user_0 --action 0 --dataset-id 0 \
  --verification-output-root outputs/alignmentVerification
```

Use `--input-kind spike-imu --spike-root PATH` for a published SpikeIMU
artifact. Existing artifacts are protected unless an explicit overwrite option
is supplied. See [DATA_FORMAT.md](DATA_FORMAT.md) for source-format facts and
their distinction from this derived processing contract.
