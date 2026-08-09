# Action0 SNN Training Notebook — Task DAG and TaskSpecs

## Plan identity

- **Source plan:** `docs/plans/TODO/04_Action0_SNN_Training_Notebook_Plan.md`
- **Baseline commit:** `93d93578ad0537054eae7843ed93b574c29b1a91`
- **Primary durable contracts:** `docs/notes/ACTION0_SNN_TRAINING.md`,
  `docs/notes/SEGMENT_PADDING.md`, and
  `docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md`

## Dependency DAG

```text
T001 Notebook foundation
  ↓
T002 Pre-training visualizations
  ↓
T003 Existing-training integration
  ↓
T004 Result visualizations and final test evaluation
  ↓
T005 End-to-end verification and documentation
```

| Task | Goal | Dependencies | Execution owner | State |
| --- | --- | --- | --- | --- |
| T001 | Create the notebook foundation and load the existing Action0 padded datasets. | — | luna_worker after freeze | DRAFT |
| T002 | Add label-distribution and unpadded 15-channel segment visualizations. | T001 | luna_worker after freeze | DRAFT |
| T003 | Integrate the existing SynNet training loop and epoch-history capture. | T002 | luna_worker after freeze | DRAFT |
| T004 | Plot training history, restore the selected model, and present final split metrics. | T003 | luna_worker after freeze | DRAFT |
| T005 | Execute the complete notebook, verify preserved SNN contracts, and document verified results. | T004 | PRIMARY (validation/documentation only) | DRAFT |

## T001 — Draft TaskSpec

- **Goal:** Establish a reproducible notebook configuration and connect it to
  the existing Action0 `segmentation_padded` producer packages.
- **Source:** plan T001; **dependencies:** none.
- **Probe scope:** current Action0 training entry point, dataset construction
  and metadata/class discovery APIs, related tests, notebook conventions, and
  the direct producer contract in `ACTION0_SNN_TRAINING.md`.
- **Required outcome:** one configuration cell; producer-root resolution;
  producer metadata and shared class-map discovery; train/validation/test
  dataset construction; and a displayed summary of split segment/label counts
  plus global class count.
- **Revised implementation boundary:** use
  `snn.action0_dataset.resolve_dataset_root`,
  `resolve_segmentation_root`, `load_padding_dataset_metadata`,
  `discover_class_to_idx`, and `Action0SegmentDataset`. Reuse the existing
  trainer validation helpers `_validate_user_splits`,
  `_validate_sampling_rate`, and `_validate_split_padded_lengths` directly;
  do not invoke `train_action0.main()` and do not duplicate/re-pad data.
- **Configuration requirement:** one early cell declares the pipeline root,
  dataset variant, sample rate, disjoint train/validation/test user lists,
  and the plan's later training parameters without yet running training.
- **Summary requirement:** validate metadata/rate before class discovery and
  dataset construction; display each dataset's `len`, `class_distribution`,
  padded length, and the global class count. Do not use producer diagnostic
  counts as split counts.
- **Contracts to preserve:** padded-package validation; 21-channel producer
  schema; model-facing channels `0:15`; one global mapping shared by the three
  disjoint user splits; metadata sampling-rate validation; and no changes to
  producer, dataset, loss, network, or prediction behavior.
- **Expected write path:** `notebooks/action0_snn_training.ipynb` only.
  Implementation, tests, configs, outputs, and all other paths are forbidden.
- **Replan triggers:** current data construction cannot be reused from a
  notebook without a public API decision; the plan's illustrative path or
  summary assumptions conflict with the producer contract; or a required
  implementation surface lies outside the confirmed write set.

### T001 probe revision record

The initial probe returned **REVISE** because the plan record had an invalid
full baseline hash and did not specify the safe foundation-only reuse path.
The revised draft records current HEAD, direct existing dataset/validator
reuse, metadata-before-construction ordering, one concrete notebook path, and
the no-test-write boundary. A fresh probe is required before freezing.

## T001 — FROZEN TaskSpec

- **Goal:** Create the Action0 training notebook foundation using the existing
  padded Action0 dataset contract, without starting training or changing any
  implementation contract.
- **Source:** plan T001; **dependencies:** none; **validated against:**
  `93d93578ad0537054eae7843ed93b574c29b1a91` and the fresh CONFIRMED probe.
- **Required behavior:**
  - Add exactly `notebooks/action0_snn_training.ipynb`, with an early single
    configuration cell containing `PIPELINE_ROOT`, `DATASET_VARIANT`,
    `SAMPLE_RATE`, disjoint `TRAIN_USERS`/`VAL_USERS`/`TEST_USERS`, and all
    later plan-owned training settings: `NEURONS_NETWORK`, `SHIFT_SYN`,
    `SHIFT_MEM`, `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`,
    `SPIKE_REGULARIZATION`, `RANDOM_SEED`, and `USE_GPU`.
  - Build the data foundation in this exact contract order: call
    `_validate_user_splits`; resolve the selected variant with
    `resolve_dataset_root` and then `resolve_segmentation_root`; load producer
    metadata; validate `SAMPLE_RATE` with `_validate_sampling_rate`; discover
    one global mapping via `discover_class_to_idx`; build all three
    `Action0SegmentDataset` instances with that shared mapping; then call
    `_validate_split_padded_lengths`.
  - Display the resolved padded root and producer metadata needed for the
    notebook, plus train/validation/test `len(dataset)`, each dataset's
    `class_distribution`, each padded length, and `len(class_to_idx)`.
  - Keep the notebook foundation-only: do not call `train_action0.main`,
    construct a model, re-pad/copy data, create directories, save outputs, or
    modify producer packages.
- **Contracts to preserve:** training consumes only `segmentation_padded`;
  padded `(S, T_pad, 21)` producer validation; model-facing features are
  channels `0:15`; one deterministic global label mapping; required right
  padding and metadata sampling-rate checks; no changes to loss, prediction,
  `SynNet`, dataset, producer, split semantics, or package APIs.
- **Contracts intentionally changed:** none.
- **Allowed writes:** `notebooks/action0_snn_training.ipynb` only.
- **Forbidden writes:** all other paths, including `outputs/**`, SNN source,
  tests, configuration, scripts, README, docs/notes, data sample, vendor, and
  environments. Existing user changes are baseline and must not be altered.
- **Acceptance criteria:** the notebook imports existing APIs directly;
  configuration is centralized; its displayed values derive from dataset APIs
  rather than producer counters; all configured splits construct from the
  real existing padded root; the shared mapping has the global class count;
  no training/model or producer mutation appears; and the specified validation
  order is visible and executable.
- **Validation commands:** run
  `conda run --no-capture-output -n writingring-gpu python -m pytest -q
  tests/test_snn_action0_dataset.py tests/test_snn_action0_smoke.py`, then the
  repository-required `pytest -q` in `writingring-gpu`; if it does not exist,
  repeat with `writingring-viz` and record the environment limitation. Execute
  a temporary copy of the notebook through nbconvert only when the selected
  environment has the notebook tooling; otherwise accurately record why that
  evidence is unavailable. Run `git diff --check`.
- **Replan triggers:** any required API/path lies outside the allowed write
  set; metadata or class-map contract differs from probe evidence; notebook
  execution requires re-padding/output generation; or the foundation would
  change training, loss, prediction, producer, or dataset behavior.

### T001 execution, verification, and documentation

`luna_worker` added only `notebooks/action0_snn_training.ipynb`. The notebook
uses the frozen configuration and validation ordering, validates and displays
the existing padded datasets without model/training/output mutation, and keeps
the later training parameters centralized. Focused Action0 tests passed 19;
the full suite passed 527 with one known skip; direct notebook execution and
its structural audit passed. `writingring-gpu` lacks nbconvert, so nbconvert
execution evidence is unavailable; the direct execution result is separately
recorded rather than substituted.

A fresh `luna_verifier` returned **PASS**: it confirmed that T001 attributable
changes are limited to the notebook, all frozen dataset contracts and summary
sources are preserved, direct execution produced train/validation/test counts
of 1,751/349/300 with padded length 1,024 and 52 global classes, and no
undocumented behavior change occurred. No README or durable technical-note
change is appropriate before the notebook's complete user-facing workflow is
implemented; this task file and `WORKBOARD.md` record the verified state.

## T002 — Draft TaskSpec

- **Goal:** Add pre-training label-distribution and one valid-only 15-channel
  segment visualization to the T001 notebook foundation.
- **Dependencies:** T001 (DONE).
- **Probe scope:** frozen T001 notebook surface, dataset sample/mask contract,
  existing plotting dependencies/conventions, and relevant tests.
- **Revised time contract:** the padded dataset has no canonical timestamp
  vector. Plot x as *relative nominal segment time* computed only as
  `valid_indices / producer_metadata.sampling_rate_hz`, label that provenance
  clearly, and do not load, infer, or reconstruct acquisition timestamps.
- **Revised distribution contract:** obtain counts from each dataset's
  `class_distribution` and denominator from `len(dataset)`; use the global
  deterministic `class_to_idx` label order, zero for a label absent from a
  split, and percentage `100.0 * count / len(dataset)`. Display an unrounded
  `Label | Train % | Val % | Test %` table.
- **Revised required outcome:** three independent Matplotlib percentage
  figures (train/validation/test) and one 15-row × 1-column Matplotlib figure
  for deterministic real `train_dataset[0]`, plotting only
  `np.flatnonzero(valid_mask.numpy())` values for channels 0–14. Display the
  original label using an inverse class map.
- **Expected write path:** `notebooks/action0_snn_training.ipynb` only. No
  source, test, data, output, configuration, or documentation write is
  authorized.
- **Replan triggers:** the dataset does not expose enough information to plot
  the specified nominal-time and valid frames without changing a
  producer/dataset contract, the plan requires absolute/canonical timestamps,
  or the visualization requires a non-Matplotlib dependency.

### T002 probe revision record

The initial probe returned **REVISE** because “time” required an explicit
provenance-safe interpretation. The revised draft uses only sample indices and
the declared producer rate; it does not assert timestamp semantics. It also
freezes exact percentage/table/mask behavior and notebook-only scope. A fresh
probe is required before freezing.

## T002 — FROZEN TaskSpec

- **Goal:** Add plan-required pre-training visualizations without changing
  producer, dataset, or training contracts.
- **Source/dependencies:** plan T002; T001 (DONE); validated by T002 re-probe.
- **Required behavior:** only in `notebooks/action0_snn_training.ipynb`, use
  global `list(class_to_idx)` order and each dataset's `class_distribution` /
  `len(dataset)` to compute unrounded `100.0 * count / len(dataset)` values;
  display `Label`, `Train %`, `Val %`, `Test %` and three independent
  Matplotlib percentage figures. For `train_dataset[0]`, inverse-map the
  label, use `np.flatnonzero(valid_mask.numpy())`, and plot exactly channels
  0–14 on a 15-row × 1-column figure using only valid values. x equals
  `valid_indices / producer_metadata.sampling_rate_hz` and is explicitly
  labeled relative nominal sample-index time based on declared rate.
- **Contracts to preserve:** T001 foundation/order; padded 21-channel contract,
  model features `0:15`, global class map, right-padding mask, declared rate,
  and all model/loss/prediction/dataset/producer behavior. Do not load, infer,
  or reconstruct timestamps; do not display padding.
- **Contracts intentionally changed:** none.
- **Allowed writes:** `notebooks/action0_snn_training.ipynb` only.
- **Forbidden writes:** all others, including outputs, SNN source, tests,
  scripts, configuration, README/docs, vendor, sample data, environments, and
  user-owned baseline changes.
- **Acceptance:** requested table/three figures/valid-only 15-channel figure
  render from existing dataset APIs; axis provenance is truthful; no padding,
  timestamps, or implementation contract changes occur.
- **Validation:** focused and full pytest in `writingring-gpu` (fall back only
  if necessary), notebook JSON/structure audit, temporary-copy nbconvert if
  available, and `git diff --check` with unrelated whitespace recorded.
- **Replan triggers:** need for canonical timestamps, T001 contract changes,
  out-of-scope write, or new metric/dependency/producer-dataset behavior.

### T002 execution, verification, and documentation

`luna_worker` changed only the notebook. It added the global-order unrounded
distribution table, three split figures, and a valid-only 15-channel figure
whose x-axis truthfully names declared-rate relative nominal sample-index time.
Focused tests passed 19 and full pytest passed 527 with one known skip; direct
execution/structural audit passed; nbconvert remains unavailable in
`writingring-gpu`. Fresh verifier **PASS** independently confirmed the frozen
scope, computations, no-timestamp/no-padding rule, and four rendered figures.
No README or technical-note update is appropriate until the complete training
workflow is verified; this task record and `WORKBOARD.md` are the durable
orchestration documentation.

## T003 — Draft TaskSpec

- **Goal:** Reuse the existing `Action0SegmentDataset`, `SynNet`,
  `MaskedCrossEntropySpkReg`, and `run_epoch` path from the notebook, with
  complete train/validation epoch histories and best-validation selection.
- **Dependencies:** T002 (DONE).
- **Probe scope:** current Action0 trainer/model/epoch APIs, metrics contract,
  checkpoint state contract, focused tests, and frozen notebook state.
- **Revised reuse contract:** reuse `_set_random_seed`, `_resolve_device`, and
  `_create_dataloaders`; retain the trainer's pre-training
  `next(iter(train_loader))` preview so seeded train-shuffle consumption stays
  consistent; validate settings before model/optimizer construction. Reuse
  `SynNet`, `MaskedCrossEntropySpkReg`, and `run_epoch` directly; do not call
  `train_action0.main()` or recompute loss, predictions, or metrics.
- **Revised history/state contract:** create
  `history={"epochs": [], "train": {...}, "val": {...}}` where each split
  records float lists for `loss`, `accuracy`, `balanced_accuracy`, `macro_f1`,
  `weighted_f1`, `mean_output_spikes`, `mean_total_spikes`, and
  `zero_output_spike_fraction`. Append after every complete train/validation
  pair for zero-based epochs. Maintain only in-memory
  `best_state=deepcopy(model.state_dict())`, `best_epoch`, and
  `best_val_balanced_accuracy`, updating strictly on higher validation balanced
  accuracy; never write a checkpoint/output directory.
- **Expected write path:** `notebooks/action0_snn_training.ipynb` only.
- **Replan triggers:** reuse would change loss/prediction semantics, selected
  best-model metric differs from the plan's validation balanced accuracy, or
  history capture needs a public API/contract redesign.

### T003 probe revision record

The probe returned **CONFIRMED** with required freeze detail. This revision
pins private helper use, RNG-relevant preview behavior, complete history keys,
and in-memory earliest-tie best-state selection. A fresh probe is required
before freezing.

### T003 second probe revision record

The re-probe confirmed the design but requires these frozen details: use
`SynNet` from `snn.utils_architectures` and `run_epoch` from
`snn.action0_engine` only; create loaders with `num_workers=0`; pass keyword
`expected_num_classes=len(class_to_idx)` and no max-batch cap; invoke
`_validate_args` with the Action0 fields plus an explicit exactly-three-valid-
layers check; and convert only the named eight returned metrics to Python
`float`. A final focused re-probe is required before freezing.

## T003 — FROZEN TaskSpec

- **Goal:** Integrate the existing Action0 SynNet trainer in the notebook and
  retain complete in-memory epoch history/best validation state.
- **Source/dependencies:** plan T003; T002 (DONE); final probe CONFIRMED.
- **Required behavior:** only in the notebook, import `SynNet` from
  `snn.utils_architectures`, `MaskedCrossEntropySpkReg` from
  `snn.action0_losses`, and `run_epoch` from `snn.action0_engine`. Before
  model construction, check `len(NEURONS_NETWORK) == 3` and invoke
  `_validate_args` with Action0 network/rate/optimizer/shift/worker/max-batch/
  split fields. Then use `_set_random_seed`, `_resolve_device`, and
  `_create_dataloaders(..., num_workers=0, ...)`, consume exactly one preview
  `next(iter(train_loader))`, create existing SynNet (input 15/output
  `len(class_to_idx)`), criterion, and Adam optimizer. For every zero-based
  epoch use uncapped existing `run_epoch` train then validation calls with
  `expected_num_classes=len(class_to_idx)`. Store only Python-float lists for
  `loss`, `accuracy`, `balanced_accuracy`, `macro_f1`, `weighted_f1`,
  `mean_output_spikes`, `mean_total_spikes`, and
  `zero_output_spike_fraction` under train/val history. Set in-memory
  best-state only on strict higher validation balanced accuracy using
  `deepcopy(model.state_dict())`; keep earliest ties; do not restore it yet.
- **Contracts to preserve:** all current model/loss/masking/prediction/metric
  semantics, T001/T002 data foundation, seeded train preview behavior, and
  15-channel input/exact class width.
- **Forbidden behavior:** `train_action0.main`, legacy run_epoch/Rockpool,
  raw metric/prediction/loss code, capped epochs, checkpoint/output writes,
  W&B, model restoration, or any non-notebook write.
- **Allowed writes:** `notebooks/action0_snn_training.ipynb` only.
- **Acceptance/validation:** histories have equal `NUM_EPOCHS` lengths;
  best-state behavior follows strict validation balanced accuracy; frozen
  imports/order hold; focused loss/model/smoke plus full pytest run in the
  prescribed environment, with structural audit and a tiny temporary
  execution; `git diff --check` records known unrelated whitespace separately.
- **Replan triggers:** any API drift, need for checkpoint persistence, altered
  loss/prediction/metric behavior, or write outside scope.

## T004 — Draft TaskSpec

- **Goal:** Use the T003 history to render the five required training curves,
  restore the best validation model, independently evaluate train/validation/
  test, and display the requested final metrics table.
- **Dependencies:** T003 (DONE).
- **Probe scope:** frozen notebook/training history, best-state representation,
  evaluation return contract, and relevant tests.
- **Replan triggers:** final evaluation cannot reuse the existing metric path,
  restoration semantics are not safe to reproduce in the notebook, or a
  requested figure/table implies a new metric.

## T005 — Draft TaskSpec

- **Goal:** Validate a clean end-to-end notebook execution and close the plan
  without changing implementation behavior.
- **Dependencies:** T004 (DONE).
- **Probe scope:** complete frozen notebook, required execution environment,
  direct SNN contract tests, and durable documentation affected by the
  notebook's verified user-facing behavior.
- **Execution ownership:** PRIMARY validation and documentation only; no
  `luna_worker` is authorized unless verification exposes a separately scoped
  implementation defect.
- **Replan triggers:** clean execution or contract tests expose an
  implementation defect, environment evidence cannot support the claimed
  acceptance result, or required durable docs disagree with verified behavior.
