# TaskSpec — Polarity-split post-encode transform

## Goal

Add a new Custom Wavelet post-encode transform:

`PolaritySplitAbs`

It converts each of the current 15 signed event channels into separate positive
and negative-magnitude channels:

pos = max(x, 0)
neg = max(-x, 0)

Result:

15 signed event channels -> 30 non-negative event channels.

## Context

This must be a first-class `POST_ENCODE_TRANSFORM` in the normal preprocessing
pipeline, alongside the existing:

- `none`
- `AbsRectify`

Do not use the post-hoc `build_polarity_split_variant.py` approach.

## Required behavior

1. Support `PolaritySplitAbs` through `_common.bash` and `encode_spikes.py`.

2. Apply the transform after signed wavelet extrema/event generation:
   - `none`: 15 signed channels
   - `AbsRectify`: 15 abs-rectified channels
   - `PolaritySplitAbs`: 30 non-negative channels

3. Use pairwise output ordering:

   `x0_pos, x0_neg_abs, x1_pos, x1_neg_abs, ..., z4_pos, z4_neg_abs`

4. For `PolaritySplitAbs`, publish:

   - `spikes.npy`: `(N, 30)`
   - `spikeIMU.npy`: `(N, 36)`
   - final 6 channels remain accel XYZ + gyro XYZ

5. Downstream segmentation/padding/validation must support the variable event
   channel count instead of assuming 15 events / 21 total channels.

6. Metadata and encoder identity must explicitly describe the new representation,
   channel names/count, and transform.

## Preserved contracts

- `none` remains exactly 15 signed events + 6 IMU = 21 channels.
- `AbsRectify` remains exactly 15 rectified events + 6 IMU = 21 channels.
- Existing numerical outputs for `none` and `AbsRectify` must not change.
- For `PolaritySplitAbs`:

  `pos - neg == original signed event`

- Positive and negative output channels are always non-negative.
- Event nonzero count is preserved across the polarity split.
- The trailing six accel/gyro channels are numerically unchanged.
- Event timing, extrema occurrence indices, segmentation boundaries, and padding
  geometry are unchanged by the transform.

## Allowed write scope

May modify spike encoding, preprocessing pipeline orchestration, downstream
SpikeIMU contract/validation, relevant SNN dataset loading, tests, and directly
related documentation.

Do not modify:

- `data_sample/**`
- `vendor/**`
- unrelated experiment/model code

## Acceptance criteria

PASS only if all are true:

1. `POST_ENCODE_TRANSFORM=PolaritySplitAbs` runs through the normal pipeline.
2. Polarity-split event output has 30 channels and SpikeIMU has 36.
3. All 30 event channels are non-negative.
4. Reconstructing `pos - neg` exactly reproduces the original signed 15-channel
   event matrix.
5. The final six IMU channels are identical to the signed representation.
6. Segmentation and padding successfully consume and publish the 36-channel
   representation.
7. `PIPELINE_MODE=continue` recognizes valid PolaritySplitAbs outputs instead of
   rebuilding them unnecessarily.
8. Existing `none` and `AbsRectify` tests remain passing.
9. SNN/data loaders do not silently truncate the new 30 event channels to 15.

## Validation

Run in this order:

1. focused transform/unit tests
2. encoding/publication tests
3. segmentation + padding contract tests
4. pipeline continue-mode tests
5. small end-to-end PolaritySplitAbs smoke test
6. relevant regression tests for `none` and `AbsRectify`

Independent `luna_verifier` must verify the complete affected pipeline contract.

## Replan triggers

Return `NEEDS_REPLAN` only if implementation reveals that:

- the documented SpikeIMU channel/layout contract materially differs from the
  current code/tests;
- supporting variable event-channel counts requires a broader artifact/schema
  migration than described above;
- an existing public/persisted contract cannot support 21- and 36-channel
  representations simultaneously without a design decision.
