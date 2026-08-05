# Ring--Board alignment outputs

Successful constant-offset alignment uses exactly one convention:

```text
ring_timestamp_us = board_timestamp_us + offset_us
```

`offset_us` is always in microseconds. A positive value means the mapped Ring
timestamp is later than the corresponding Board timestamp. Raw Ring, Board,
and label timestamps are never modified; aligned values are derived only for
matching and plotting.

## Outputs

For `user_0`, action `0`, dataset `0`, the alignment CLI writes:

```text
data/user_0/0/0_ring_board_offset.txt
outputs/alignment/reports/user_0/action_0/0_alignment_report.json
outputs/alignmentVerification/user_0/action_0/0_alignment_verification.png
```

The TXT file is UTF-8 `key=value` data. It includes the mapping direction,
unit, `offset_us`, display-only `offset_ms`, and alignment coverage metadata.
Readers must use `offset_us`; an unsuccessful alignment cannot be exported.

The verification image has six vertically stacked ten-second panels. It shows the
Ring transient score, Board press/lift events after applying the exact same
offset, and labels from `{dataset_id}_label.txt` or fallback
`{dataset_id}_timestamp.txt`. The label time domain is recorded in both the
image subtitle and report. `shared` and `ring` label domains are not shifted;
`board` labels receive the same Board-to-Ring offset as Board events.

Existing offset and verification files are rejected by default. Pass the
corresponding explicit overwrite flag only when replacing the artifact is
intended.

## CLI

```bash
python scripts/align_ring_board.py \
  --data-root data_sample/data \
  --user user_0 \
  --action 0 \
  --dataset-id 0 \
  --verification-output-root outputs/alignmentVerification \
  --label-time-domain shared
```

By default, the offset TXT is written beside the selected `*_ring_0.bin` in
the data directory. Supply `--offset-output-root PATH` only when a separate,
structured export root is wanted. Useful options are `--label-path`,
`--verification-start-seconds`, `--overwrite-offset`, and
`--overwrite-verification`. On an unsuccessful
alignment, the report is retained but no offset TXT or verification PNG is
created.
