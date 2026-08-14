# Plan: Experiment A/B/C/D Single-Action and Multi-Action Dataset Input

## Plan identity

- **Repository:** `hellowPluto78700/writingRing`
- **Validated against commit:** `380a9fb74ae0ac53579fcb6b8c57db6c47340237`
- **Workflow authority:** `AGENTS.md`
- **Agent roles:** `PRIMARY`, `luna_probe`, `luna_worker`, `luna_verifier`
- **Risk class:** **HIGH_RISK**
- **Reason for HIGH_RISK:** the change crosses the shared acceleration dataset loader, all four experiment runners/notebooks, and the persisted Experiment A checkpoint/provenance contract used by Experiments B/C/D.

## Goal

Allow all four acceleration CNN experiment notebooks and their runners to consume:

1. **Action 0 only**
2. **Action 1 only**
3. **Action 0 + Action 1 together**

without physically merging or copying the two action datasets.

Action 0 and Action 1 remain in separate existing output directories. When both are selected, the shared dataset layer exposes one logical in-memory dataset containing packages from both roots.

The experiment semantics must remain:

- **Experiment A:** raw acceleration only.
- **Experiment B:** frozen Experiment A CNN; raw reference plus reconstruction evaluation.
- **Experiment C:** reconstruction-only training/evaluation.
- **Experiment D:** mixed raw + reconstruction training; separate raw and reconstruction testing.

User-disjoint splitting remains by `user`, not by `(user, action)`. Therefore all samples for a given user, across every selected action, must belong to the same train/validation/test split.

---

## Non-goals

This plan does **not**:

- merge Action 0 and Action 1 into a new on-disk dataset;
- rename or rewrite existing padded package files;
- change raw acceleration channel semantics (`paddedSpikeIMU[..., 15:18]`);
- change reconstruction file semantics;
- change the CNN architecture, loss, optimizer, representation metrics, or experiment protocols;
- change preprocessing, segmentation, padding, or reconstruction producers;
- introduce cross-action resampling or automatic temporal-length conversion;
- modify `data_sample/**` or `vendor/**`.

---

## Required user-facing input model

All four notebooks must expose one central dataset-root configuration that supports one or two roots.

Conceptually:

```python
DATASET_ROOTS = [
    ACTION0_ROOT,
]

# or

DATASET_ROOTS = [
    ACTION1_ROOT,
]

# or

DATASET_ROOTS = [
    ACTION0_ROOT,
    ACTION1_ROOT,
]
```

The roots remain independent physical directories.

For a two-root run:

```text
action0 root ──┐
               ├── shared loader ──> one logical package/manifest view
action1 root ──┘
```

No `combined_raw/`, `combined_reconstruction/`, or other merged output directory is created.

Existing single-root runner/CLI usage must remain supported.

---

## Core invariants

### Dataset identity

For every loaded sample, the manifest must retain at least:

- user identity;
- action identity;
- label;
- segment identity;
- package identity;
- valid length;
- reconstruction availability where applicable.

Action 0 and Action 1 samples must remain distinguishable in provenance.

Sample IDs must be globally unique across all selected roots. A duplicate sample ID or duplicate `(user, action)` package identity must fail explicitly rather than silently overwrite or deduplicate data.

### User split

Train/validation/test assignment remains user-disjoint.

For any user `U`:

```text
U/action0 and U/action1
```

must always inherit the same split when both actions are selected.

No split operation may treat Action 0 and Action 1 as independent user populations.

### Metadata compatibility

When more than one root is selected, the shared loader must reject incompatible producer contracts before training/evaluation.

The selected roots must agree on the contracts required to batch and interpret samples, including:

- `input_kind`;
- feature schema;
- channel count;
- acceleration channel meaning;
- sampling rate;
- right-padding convention;
- `target_length`.

The first implementation must **not** silently batch roots with different padded temporal lengths. If `target_length` differs, fail with an actionable error naming the incompatible roots/values.

A future dynamic-padding feature is out of scope.

### Reconstruction requirement

- Experiment A may load roots without reconstruction artifacts.
- Experiments B, C, and D require complete reconstruction artifacts for every selected package needed by their protocol.
- Missing reconstruction in any selected action must fail explicitly; the implementation must not silently drop that action/package.

### No silent cohort intersection

When a selected root is missing an action, user, label, or sample required by the authoritative Experiment A cohort, downstream experiments must reject the mismatch rather than silently intersect/filter to the available subset.

---

# Agent workflow

This work follows `AGENTS.md`.

```text
PRIMARY
  |
  | inspect current contracts
  | freeze TaskSpec
  v
luna_worker
  |
  | inspect -> implement -> focused tests -> fix -> retest -> self-review
  v
luna_verifier
  |
  | independent contract + acceptance verification
  v
PRIMARY integration/documentation
```

`luna_probe` is **not mandatory**.

Use `luna_probe` only if a concrete unresolved repository fact blocks freezing or implementation. Examples:

- the checkpoint serializer cannot safely carry the required cohort metadata;
- a notebook bypasses the shared runner in a way not visible from current source;
- an existing test fixture encodes a conflicting persisted-artifact contract.

Do not invoke Probe merely for confidence.

---

# Task DAG

```text
T001 Multi-root dataset and Experiment A cohort contract
  ↓
T002 Experiments B/C/D multi-root enforcement
  ↓
T003 A/B/C/D notebook configuration and integration acceptance
```

---

# T001 — FROZEN TaskSpec: multi-root dataset and Experiment A cohort contract

## Goal

Extend the shared acceleration experiment data path and Experiment A runner so one experiment run may use one selected action root or two compatible action roots, while preserving existing single-root behavior and establishing an authoritative multi-action cohort contract for downstream experiments.

## Context

Current experiment runners use a single dataset root. The shared loader resolves one padded dataset and returns one package collection plus one sample manifest.

Current sample identity already carries action information, and current train/validation/test assignment is user-based. The new behavior should therefore combine roots at the shared dataset layer rather than duplicating merge logic in notebooks.

## Required behavior

1. The shared acceleration data-loading surface accepts either:
   - one dataset root; or
   - a sequence containing the selected Action 0 / Action 1 roots.

2. A single selected root preserves current behavior.

3. Multiple selected roots are fully loaded and validated independently, then exposed as one logical dataset:
   - package indices are valid and unique in the combined view;
   - sample manifest rows point to the correct combined package;
   - action identity is preserved;
   - source-root provenance is preserved.

4. Multi-root input rejects:
   - an empty root selection;
   - duplicate root arguments resolving to the same padded dataset;
   - duplicate package identities;
   - duplicate sample IDs;
   - incompatible producer metadata;
   - incompatible `target_length`.

5. Experiment A performs cohort filtering, user-disjoint split construction, raw-train normalization, training, and evaluation over the complete selected logical dataset.

6. Experiment A checkpoint/provenance records enough durable cohort identity for B/C/D to determine whether they loaded the same action/sample cohort.

7. The new Experiment A cohort metadata must include, at minimum:
   - selected action IDs;
   - selected sample count;
   - a deterministic digest of the canonical selected sample IDs after A cohort restriction and before source-domain duplication;
   - existing train/validation/test users;
   - existing `class_to_idx`;
   - existing user-exclusion / eligibility metadata.

8. The cohort digest must be path-independent:
   - moving the same dataset to a different filesystem location must not change cohort identity;
   - root paths may be recorded separately as provenance, but must not define sample identity.

9. Existing single-root Experiment A calls and existing Action 0 defaults continue to work.

10. New Experiment A provenance records all selected root arguments, resolved padded roots, and selected actions rather than collapsing them into one ambiguous root string.

## Preserved contracts

- raw acceleration remains `paddedSpikeIMU[..., 15:18]`;
- model input remains `(B, 3, T)`;
- user-disjoint split semantics remain unchanged;
- normalization is still fit only on valid raw train time points for A;
- right-padding masks remain canonical;
- label mapping semantics remain unchanged;
- A remains the authoritative source of split/class/cohort metadata for B/C/D;
- existing artifact filenames remain unchanged unless an additional metadata field/file is strictly needed;
- no source dataset directory is modified.

## Allowed write scope

Implementation may change only the experiment evaluation surface needed for this behavior, including:

- shared acceleration reconstruction evaluation modules;
- Experiment A runner;
- focused experiment tests/fixtures;
- Experiment A notebook only if required to exercise the new runner API during this task.

Do not modify preprocessing, segmentation, padding, reconstruction producer logic, `data_sample/**`, or `vendor/**`.

## Acceptance criteria

PASS only if all are true:

1. Action 0-only loading succeeds.
2. Action 1-only loading succeeds.
3. Action 0 + Action 1 loading succeeds when metadata is compatible.
4. Combined manifest contains both action IDs and globally unique sample IDs.
5. A user present in both actions has one and only one split assignment.
6. Normalization is computed from all valid selected raw-train samples, not one root only.
7. Duplicate roots fail before training.
8. incompatible metadata or `target_length` fails before training.
9. New A checkpoint contains deterministic action/sample cohort identity.
10. Existing single-root A behavior remains supported.
11. No combined dataset directory is created.

## Validation

Run cheap-to-expensive:

1. focused unit tests for root normalization, merge/reindex, duplicates, compatibility, and cohort digest;
2. focused Experiment A integration tests for action0-only, action1-only, both;
3. existing acceleration reconstruction tests affected by the loader/checkpoint contract;
4. `git diff --check`.

Do not run full pytest yet unless focused failures indicate a broader shared contract issue.

## Dependencies

None.

## Replan triggers

Return `NEEDS_REPLAN` only if correct implementation requires:

- changing padded package schemas/files;
- changing acceleration channel semantics;
- supporting unequal padded lengths through dynamic batching;
- changing user-disjoint split semantics;
- replacing the A-authoritative cohort model;
- materially expanding into preprocessing/segmentation/reconstruction producers.

---

# T002 — FROZEN TaskSpec: Experiments B/C/D multi-root cohort enforcement

## Goal

Make Experiments B, C, and D accept the same one-or-two-root input model as Experiment A while guaranteeing that they evaluate/train on the exact action/sample cohort authorized by the A checkpoint.

## Required behavior

1. B/C/D accept one or two selected dataset roots through the same public input contract used by A.

2. For all three experiments, dataset packages are fully loaded and validated before checkpoint cohort restriction.

3. B/C/D recover the authoritative:
   - train users;
   - validation users;
   - test users;
   - class mapping;
   - selected action set;
   - canonical sample cohort identity
   from the A checkpoint.

4. B/C/D reject action mismatch.

Examples:

```text
A checkpoint: action0
B request:    action0 + action1
=> reject
```

```text
A checkpoint: action0 + action1
C request:    action0 only
=> reject
```

5. B/C/D reject sample-cohort mismatch even when user and label sets happen to match.

6. A legacy A checkpoint that predates action/sample cohort metadata:
   - remains usable for the legacy single-root path when current compatibility rules are satisfied;
   - must not authorize a two-root run;
   - a two-root run with a legacy checkpoint fails with an actionable message instructing the user to regenerate Experiment A for the selected roots.

7. Experiment B:
   - uses A model weights;
   - uses A raw-train normalization;
   - uses raw train/validation reference embeddings;
   - tests raw and reconstruction views over the same selected logical cohort;
   - performs no reconstruction fine-tuning/refit.

8. Experiment C:
   - uses A only for cohort/split/class authority;
   - does not reuse A model weights or normalization;
   - fits reconstruction normalization from reconstruction train samples across all selected actions.

9. Experiment D:
   - uses A only for cohort/split/class authority;
   - creates one mixed training domain from both raw and reconstruction for every logical selected training sample across all selected actions;
   - retains separate raw and reconstruction test views over identical logical test sample identities.

10. Provenance for B/C/D records all selected roots/actions and the validated A cohort digest.

## Preserved contracts

- B remains frozen-model evaluation.
- C remains reconstruction-trained from scratch.
- D remains mixed-domain training from scratch.
- B uses A normalization; C/D fit their protocol-specific normalization.
- representation reference domains remain unchanged.
- paired raw/reconstruction evaluation retains one-to-one logical sample alignment.
- B/C/D do not independently redefine the A cohort.
- existing single-root Action 0 workflow remains supported.

## Allowed write scope

Implementation may change:

- Experiment B/C/D runners;
- shared cohort/checkpoint helpers needed by A-D;
- focused B/C/D integration tests and fixtures.

Do not modify model architecture, representation metric definitions, preprocessing producers, or source data.

## Acceptance criteria

PASS only if:

1. B/C/D each run through their setup path with action0-only.
2. B/C/D each run through their setup path with action1-only when paired with an A checkpoint produced from action1.
3. B/C/D each accept action0+action1 when paired with an A checkpoint produced from the same combined cohort.
4. Wrong selected action set is rejected.
5. Missing selected action is rejected.
6. Same users/labels but different canonical sample cohort is rejected.
7. multi-root use with a legacy A checkpoint is rejected with a regenerate-A message.
8. legacy single-root checkpoint behavior remains supported.
9. B raw/reconstruction test views have identical logical sample IDs and labels.
10. D raw/reconstruction test views have identical logical sample IDs and labels.
11. No experiment silently drops a package to satisfy checkpoint compatibility.

## Validation

1. focused checkpoint compatibility tests;
2. focused B/C/D multi-root integration tests;
3. paired-view identity tests;
4. existing acceleration reconstruction exclusion/label-selection integration tests;
5. `git diff --check`.

## Dependencies

T001 must be DONE and independently verified before T002 is frozen for execution.

## Replan triggers

Return `NEEDS_REPLAN` if:

- A checkpoint cannot provide path-independent sample cohort identity without changing unrelated artifact semantics;
- downstream experiments require different logical sample definitions;
- reconstruction packages do not preserve one-to-one padded segment identity with raw packages;
- correct support requires changing the A/B/C/D scientific protocol.

---

# T003 — FROZEN TaskSpec: notebook configuration and final integration

## Goal

Make the four notebooks thin, consistent configuration/presentation layers for single-action or two-action execution, then verify the complete experiment surface.

## Required behavior

1. Each notebook has one obvious configuration block for the selected dataset roots.

2. Each notebook supports:
   - Action 0 only;
   - Action 1 only;
   - Action 0 + Action 1.

3. The notebooks pass selected roots to the runner/shared API and do not implement their own package concatenation, manifest mutation, split logic, or normalization.

4. Notebook configuration explains that:
   - roots stay physically separate;
   - two-root input is logically combined in memory;
   - B/C/D must use an A checkpoint produced from the same selected action/sample cohort.

5. Existing experiment scientific semantics and plots remain unchanged except for provenance/reporting needed to show the selected roots/actions.

6. Stale executed notebook outputs that claim an incompatible single-root configuration are cleared or regenerated according to the repository's existing notebook policy.

7. Durable documentation is updated only where needed so users can discover the supported input modes and the regenerate-A requirement for changing the action cohort.

## Preserved contracts

- notebooks remain presentation/configuration layers rather than duplicate implementations;
- runner results/artifact locations remain the authoritative experiment outputs;
- plots and metrics preserve current definitions;
- source action directories remain untouched;
- existing default behavior remains Action 0-only unless the project intentionally chooses another documented default during implementation.

## Allowed write scope

- four Experiment A/B/C/D notebooks;
- directly relevant experiment documentation in `README.md` and/or `docs/notes/**`;
- focused notebook/configuration tests or structural checks;
- plan/workboard bookkeeping only if PRIMARY decides persistent task state is useful.

No unrelated notebook cleanup.

## Acceptance criteria

PASS only if:

1. all four notebooks expose the same dataset-root selection concept;
2. no notebook contains a private Action 0/Action 1 merge implementation;
3. action0-only configuration resolves correctly;
4. action1-only configuration resolves correctly;
5. two-root configuration resolves correctly;
6. B/C/D surface a clear checkpoint-cohort error if A was generated for a different action selection;
7. provenance shown/saved by each experiment names the actual selected actions;
8. notebook JSON is valid and stale contradictory outputs are absent;
9. focused A-D tests pass;
10. full repository pytest passes, except already-known documented skips/failures unrelated to this change;
11. `git diff --check` passes.

## Validation

Run, in order:

1. notebook JSON/structure checks;
2. focused A-D experiment tests;
3. focused multi-root integration/smoke tests;
4. existing acceleration CNN experiment regression tests;
5. full `pytest`;
6. `git diff --check`.

Use actual representative Action 0/Action 1 data for an end-to-end smoke run only if it is available locally and the cost is reasonable. Lack of large local data is `UNVERIFIED` evidence, not automatically a failure, provided fixture-based contract tests cover the frozen requirements.

## Dependencies

T001 -> T002 -> T003.

## Replan triggers

Return `NEEDS_REPLAN` if:

- notebooks cannot remain thin wrappers because required runner functionality is absent;
- supporting both actions requires physical source-data rewriting;
- real Action 0/Action 1 producer metadata proves fundamentally incompatible in a way not handled by explicit rejection;
- notebook requirements imply a change to the scientific A/B/C/D protocols.

---

# Independent verification gates

Because this plan changes persisted cohort metadata and cross-experiment producer/consumer behavior, every implementation TaskSpec must receive independent verification.

The verifier should judge the frozen behavior, not implementation style.

Critical verifier checks:

```text
single root remains compatible
two roots remain physically separate
combined logical manifest has unique sample IDs
same user never crosses splits between actions
A checkpoint records selected action/sample cohort
B/C/D require exact A cohort
legacy checkpoint cannot silently authorize multi-root
B and D paired raw/recon views retain identical logical sample identities
no reconstruction/domain protocol changes
no source data rewritten
```

Verifier verdicts:

- `PASS`: all frozen requirements satisfied.
- `FAIL`: implementation defect repairable inside frozen behavior/write scope.
- `REPLAN`: frozen contract itself is materially wrong/stale.

A `FAIL` returns to `luna_worker` for repair and focused re-validation; it does not automatically require a new plan.

---

# Expected final user workflow

## Action 0 only

```python
DATASET_ROOTS = [
    ACTION0_ROOT,
]
```

Run A first, then B/C/D using the A checkpoint from that run.

## Action 1 only

```python
DATASET_ROOTS = [
    ACTION1_ROOT,
]
```

Run A first, then B/C/D using the A checkpoint from that run.

## Action 0 + Action 1

```python
DATASET_ROOTS = [
    ACTION0_ROOT,
    ACTION1_ROOT,
]
```

Run A first. Its checkpoint becomes authoritative for the combined cohort.

Then run B/C/D with the same two selected roots and the combined-cohort A checkpoint.

Conceptually:

```text
ACTION0_ROOT ──┐
               ├── logical shared dataset ──> user split/cohort
ACTION1_ROOT ──┘                              |
                                              ├── A: RAW
                                              ├── B: RAW -> frozen CNN -> RECON
                                              ├── C: RECON
                                              └── D: RAW + RECON mixed
```

The source directories are never merged on disk.

---

# Completion definition

The plan is complete only when:

- one-root and two-root input are supported across A/B/C/D runners and notebooks;
- user-disjoint split semantics remain intact across actions;
- incompatible root metadata is rejected before training;
- A persists an authoritative action/sample cohort contract;
- B/C/D validate exact compatibility with A rather than silently intersecting data;
- legacy single-root usage remains supported;
- documentation explains the three supported input modes;
- independent verification passes the frozen TaskSpecs;
- final regression validation passes.
