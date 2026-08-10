# Action0 Notebook Subset Experiment — Task DAG and TaskSpecs

## Plan identity

- **Source plan:** `docs/plans/TODO/06_Action0_Notebook_Subset_Experiment_Plan.md`
- **Primary contracts:** `README.md`, `docs/notes/ACTION0_SNN_TRAINING.md`,
  `docs/notes/SPIKE_ENCODING.md`, and current Action0 dataset/trainer APIs.

| Task | Goal | Dependencies | Execution owner | State |
| --- | --- | --- | --- | --- |
| T001 | Complete the reproducible common-label notebook experiment and final evaluation. | — | luna_worker after freeze | DONE |
| T002 | Document the verified notebook experiment and its distinction from CLI training. | T001 | PRIMARY | DONE |

## T001 — Draft TaskSpec

- **Goal:** Harden the user-modified Action0 notebook as a supported,
  reproducible selected-label experiment.
- **Required behavior:** choose exactly `NUM_SELECTED_LABELS` labels using
  `LABEL_SELECTION_SEED` from the intersection of labels with positive counts
  in the full train, validation, and test datasets; build one sorted contiguous
  selected mapping shared by all selected datasets/loaders/model/results.
- **Result behavior:** retain existing loss/metric semantics, restore the
  strict-best in-memory validation state before final uncapped train/validation/
  test `run_epoch` evaluation, and display the final metrics table. Make figure
  saving explicit and repository-anchored rather than cwd-relative.
- **Contracts to preserve:** existing post-transform root selection, padded
  producer validation, user-disjoint splits, 21-channel schema, model inputs
  `0:15`, engine/loss/prediction semantics, and the canonical CLI's full-label
  mapping. No producer/dataset/source API change.
- **Likely write scope:** `notebooks/action0_snn_training.ipynb` and notebook
  figure artifacts only if explicitly configured. No other implementation,
  data, producer, test, or environment paths.
- **Replan triggers:** wrapper cannot be made compatible with existing loader/
  engine behavior, selected labels lack split coverage, figure output policy
  needs a non-notebook write, or final evaluation needs new metrics/APIs.

### T001 probe revision record

The initial probe returned **REVISE**. The frozen contract must explicitly use
positive `class_distribution` intersection (rather than global-map sampling),
validate selection size/nonempty selected splits, and avoid
`run_epoch(split="train")` for final evaluation because it mutates weights.
It must also pin the existing eight metrics, a default-off repository-anchored
figure policy, and removal of stale executed outputs. A fresh probe is
required.

## T001 — FROZEN TaskSpec: reproducible common-label experiment

- **Goal:** Complete the notebook's selected-label Action0 experiment and
  produce final, non-mutating best-model results without changing the CLI or
  producer contracts.
- **Source/dependencies:** user-approved plan; final T001 probe CONFIRMED
  against `51293739773d6a192ad8cdec18857d6ec4b52319`.
- **Selection order:** build the three full datasets with the canonical full
  mapping; calculate the sorted intersection of labels with positive
  `class_distribution` counts in train, validation, and test; reject
  `NUM_SELECTED_LABELS <= 0` or a count greater than the intersection; sample
  without replacement with `np.random.default_rng(LABEL_SELECTION_SEED)`, sort
  the sampled labels, then enumerate one contiguous selected mapping. The
  notebook must display the selected labels/mapping and selection provenance.
- **Wrapper invariants:** the selected wrapper filters the full datasets,
  remaps every old label to the shared selected mapping, preserves
  `padded_length`, exposes only selected-label counts, and asserts positive
  coverage for every selected label in every split, nonempty selected splits,
  consistent retained-count totals, and `(T, 15)`/long/bool sample shapes.
  Use that same mapping/dataset for plots, loaders, model output width, history,
  and all final results.
- **Training/final evaluation:** preserve the existing strict-greater
  validation balanced-accuracy best-state rule. After training call
  `model.load_state_dict(best_state, strict=True)`, then run uncapped final
  train and validation evaluation as `split="val", optimizer=None` and final
  test evaluation as `split="test", optimizer=None`. Never run final train
  data with `split="train"`. Display a Train/Validation/Test table containing
  exactly the existing eight metrics: loss, accuracy, balanced accuracy,
  macro F1, weighted F1, mean output spikes, mean total spikes, and zero-output
  spike fraction. Display provenance: transform, variant, selected roots,
  sample rate, selection seed/count, selected labels/mapping, best epoch, and
  best validation balanced accuracy.
- **Figure policy:** define `SAVE_FIGURES = False` in the central configuration
  cell. Always show figures in the notebook; create/save only when this setting
  is true, using the repository-derived `notebooks/figures` directory. Do not
  touch the existing untracked figure files during implementation or validation.
- **Contracts to preserve:** selected transform root before data construction;
  padded 21-channel/1,024-frame/200-Hz/right-padding validation; user-disjoint
  splits; model input 0:15; existing SynNet/loss/prediction/metric semantics;
  canonical CLI full-label mapping; no output/provenance inference changes.
- **Allowed writes:** `notebooks/action0_snn_training.ipynb` only. Clear stale
  execution outputs and counts that no longer describe source behavior.
  **Forbidden writes:** all other paths, including figures, outputs, producer
  data, SNN source, tests, scripts, config, README/docs, vendor, and
  environments.
- **Acceptance/validation:** both transform roots retain existing producer
  validation; fixed seed produces the expected selected common-label mapping;
  invalid selection size fails before wrapper/model; every selected label is in
  every split; final evaluation leaves the restored state unchanged; final
  table/provenance uses exactly existing metrics; no files are written with
  `SAVE_FIGURES=False`; focused Action0 tests, full pytest, notebook
  JSON/structure/direct execution audits, and `git diff --check` pass.
- **Replan triggers:** a common-label selection cannot be built without source
  API changes, exact output state cannot be preserved through final evaluation,
  a new metric is needed, or a write outside the notebook is required.

### T001 execution and verification

The worker changed only `notebooks/action0_snn_training.ipynb`. Fresh verifier
**PASS** confirmed the frozen common-label selection/mapping/wrapper
invariants, strict restored best state, non-mutating final evaluations, exact
eight-metric table/provenance, and default-off repository-anchored figures.
Fixed seed selected `J,K,O,U,e,f,h,j,m,y`; invalid sizes 0 and 53 failed before
construction; both transform roots passed existing producer validation.
Focused Action0 tests passed 24; full pytest passed 527 with one known skip;
reduced direct execution from repository and `/tmp` passed. `nbformat` and
`nbconvert` are unavailable, so JSON/AST/structure audits are the recorded
notebook evidence.

## T002 — FROZEN TaskSpec: notebook experiment documentation

- **Goal:** Document the verified notebook subset experiment without changing
  the canonical CLI or producer contract descriptions.
- **Dependencies/source:** T001 (DONE); user-approved plan and verified T001.
- **Required changes:** in `docs/notes/ACTION0_SNN_TRAINING.md`, add a concise
  notebook-only section distinguishing CLI full-variant mapping from the
  reproducible common-label subset workflow, transform-root selection,
  selected mapping/provenance, best-state final evaluation, and default-off
  figures. In `README.md`, make the notebook and its Action0 documentation
  discoverable.
- **Forbidden documentation changes:** do not change `SPIKE_ENCODING.md`,
  `SEGMENT_PADDING.md`, Bash-pipeline contracts, CLI defaults, producer schema,
  or assert that notebook paths are a generic CLI API.
- **Allowed writes:** `README.md`, `docs/notes/ACTION0_SNN_TRAINING.md`,
  this task file, and `docs/plans/WORKBOARD.md` only.
- **Acceptance:** documentation makes the CLI/notebook distinction explicit;
  preserves all established producer/CLI claims; matches actual verified
  notebook behavior; Markdown links resolve; and `git diff --check` passes.

### T002 documentation and completion

PRIMARY updated `README.md` to expose the notebook alongside the CLI baseline
and `docs/notes/ACTION0_SNN_TRAINING.md` to document the verified
common-label experiment. The note explicitly preserves the CLI's full variant
mapping and all producer/encoding/padding contracts; it records subset
selection provenance, transform-root behavior, restored-state evaluation, and
default-off figures. `git diff --check` passed and the documentation links
resolve locally. No change was required in spike encoding, padding, or shell
pipeline notes. T002 is DONE.
