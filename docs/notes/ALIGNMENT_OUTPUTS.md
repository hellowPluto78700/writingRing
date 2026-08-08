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
