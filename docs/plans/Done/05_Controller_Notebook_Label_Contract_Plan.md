# Plan: Controller Notebook Label Contract Repair

## Goal

Correct `notebooks/writingRing_preprocessing_spike_controller.ipynb` so it
does not treat action-directory IDs as semantic labels or classes, and gives a
truthful read-only audit of timestamp label markers.

## Intended behavior

```text
recording inventory: user / action_id / dataset_id
    separate from
timestamp-marker inventory: timestamp-file text labels and source locations
```

The notebook remains a preprocessing and spike-encoding controller. It does
not run segmentation merely to derive a class distribution, and it does not
alter producer, timestamp, or segmentation artifacts.

## Task DAG

```text
T001 Label-contract notebook repair
```

T001 requires probe → freeze → worker → verifier before completion.
