# Plan: Alignment Outcome Downstream Consistency Correction

## Goal

Keep aligned-Board downstream artifacts consistent with the current alignment
outcome partition.  When a recording changes between `SUCCESS` and `SKIPPED`,
Action0 `continue` must reject stale segmentation and padding rather than
declaring the pipeline complete.  Also ensure a SpikeIMU recording ultimately
validated as `SKIPPED` does not participate in sampling-rate validation.

## In scope

- `scripts/action0_pipeline/_common.bash`
- `src/writingring/board_event_segmentation.py`
- `tests/test_action0_pipeline_scripts.py`
- `tests/test_board_event_segmentation.py`
- `tests/test_spike_imu_segmentation.py`
- affected durable notes and this plan's task/workboard state

The plan does not change alignment algorithms, outcome schemas or skip
reasons, timestamp repair, per-recording Action0 resume, padding algorithms,
SNN, notebooks, `vendor/**`, or `data_sample/**`.

## DAG

```text
T6 Downstream outcome dependency contract
        |
        +------------------+
        |                  |
        v                  v
T7 Action0 stale     T8 SpikeIMU skip/rate
reuse correction     correction
        \                  /
         \                /
          v              v
        T9 End-to-end transition verification
```

T10, a standalone alignment-root CLI cleanup, is expressly optional and does
not block T6–T9.  It is deferred unless the main-path work exposes a concrete
consumer incompatibility that cannot be resolved under the current Action0
canonical sibling layout.

## Completion invariant

```text
alignment outcome changes
    -> every downstream artifact that depends on it becomes stale
    -> continue never treats stale segmentation or padding as COMPLETE
```

The plan is complete only after T6–T9 are `DONE`, a fresh final verifier
passes, the full Python 3.11 test suite passes, and durable documentation is
reconciled.

## Completion record

T6–T9 are DONE. Fresh final verification passed after `bash -n`, 75 focused
regressions (1 skipped), and the full Python 3.11 suite (581 passed, 1
skipped) in `writingring-gpu`. The optional T10 standalone-root cleanup was
not required for the canonical Action0 contract and remains deferred.
