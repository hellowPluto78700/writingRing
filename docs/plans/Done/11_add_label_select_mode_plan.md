# Plan — Experiment Cohort Selection and Downstream Inheritance

**CLASSIFICATION: HIGH_RISK**

This change redefines the Experiment A cohort-selection contract and makes that cohort authoritative for downstream B/C/D experiments. It crosses experiment pipeline boundaries and therefore requires independent contract verification plus focused integration validation.

## Goal

Experiment A supports explicit `EXCLUDED_USERS` and `INCLUDED_LABELS`.

After full dataset/package loading and validation, A must:

1. exclude configured users;
2. select configured labels;
3. build the user-disjoint split;
4. compute normalization from the final training cohort;
5. train and evaluate the CNN.

B/C/D must inherit A's resulting split users and class mapping instead of independently redefining the cohort.

For this task, **same cohort** means:

- identical train/val/test user assignment;
- identical selected label set;
- identical `class_to_idx` mapping;
- identical corresponding segment identity, defined by `package_index + segment_index`, wherever experiments operate on paired views of the same samples.

Domain-specific representations, model weights, and normalization sources may still differ where the existing experiment protocol requires them to differ.

## Global constraints

- Processing order is fixed:

  ```text
  load + validate
  -> exclude users
  -> select labels
  -> split
  -> normalization
  -> training / evaluation
  ```

- Full dataset/package validation remains before cohort filtering.
- User exclusion happens before label selection.
- Every requested `INCLUDED_LABELS` entry must exist in the **post-exclusion** segment pool. If any requested label has no surviving segment after user exclusion, fail explicitly; never silently intersect requested labels with surviving labels.
- Excluded users must contribute zero segments to train, val, test, or downstream evaluation.
- Users with no surviving segments after filtering must not enter the split.
- Explicit train/val/test users that are unavailable after filtering must cause an explicit failure; do not silently remove them.
- `class_to_idx` must cover exactly the final selected label set.
- Do not modify underlying padded/reconstruction arrays or renumber `sample_id`, `package_index`, or `segment_index`.
- Do not change CNN architecture, optimizer/training algorithm, normalization algorithm, embedding metrics, retrieval metrics, clustering metrics, or geometry metrics.
- B keeps frozen A weights and A normalization.
- C/D keep fresh training and their existing normalization protocol.
- Existing A checkpoint fields `train_users`, `val_users`, `test_users`, and `class_to_idx` are the intended downstream cohort authority.
- Do not extend checkpoint schema unless those existing fields are proven insufficient to reconstruct the required downstream cohort.

## Task graph

```text
T1 -> T2
```

## Tasks

### T1 — Apply cohort selection before Experiment A splitting

#### Goal

Make Experiment A define the final cohort before user splitting so that all downstream computation naturally uses only the selected users and labels.

#### Required behavior

- A runner/notebook exposes `EXCLUDED_USERS` and `INCLUDED_LABELS`.
- Dataset/package loading and validation complete before filtering.
- Apply user exclusion first.
- Apply label selection to the post-exclusion segment pool.
- Every requested label must have at least one surviving segment after user exclusion; otherwise fail explicitly.
- The split user pool contains only users with at least one surviving selected segment.
- Normalization uses only final training-cohort segments.
- `class_to_idx` is generated from the final selected label set.
- `INCLUDED_LABELS=None` and empty user exclusion preserve current behavior.
- Existing explicit split validation remains active on the filtered cohort.

#### Preserved contracts

- User-disjoint split semantics remain unchanged.
- Existing split validation such as `require_all_labels_in_all_splits` continues to apply to the filtered cohort.
- Dataset segment lookup continues to use original `package_index + segment_index`.
- Filtering must not renumber or rewrite underlying segment identity.
- No padded/reconstruction artifact format changes are introduced.

#### Allowed write scope

- `snn/accel_reconstruction_eval/**`
- Experiment A runner/notebook
- focused tests directly related to A cohort selection

The worker may modify additional local helper/test files inside the same experiment surface when required by the implementation, provided no unrelated behavior is changed.

#### Acceptance criteria

- With no filtering configured, existing A split and class-mapping behavior is unchanged.
- An excluded user appears zero times in the final A manifest and zero times in train/val/test.
- On successful filtering, final manifest label set equals `INCLUDED_LABELS` exactly.
- A requested label with no post-exclusion surviving segment causes an explicit failure.
- A user with no surviving segment after filtering does not enter train/val/test.
- An explicit split referencing a user unavailable after filtering causes an explicit failure.
- `class_to_idx.keys()` equals the final manifest label set exactly.
- Surviving segment `package_index + segment_index` identities are unchanged by filtering.
- Normalization is computed only from the final training cohort.

#### Validation

Run focused dataset/split/A-runner tests covering:

- no filtering;
- user exclusion only;
- label selection only;
- combined user exclusion + label selection;
- requested label removed by exclusion;
- unknown requested label;
- explicit split conflict after filtering;
- identity preservation for surviving segments.

Do not run full-repository pytest for T1.

#### Dependencies

- None

#### Replan only if

- Manifest-level filtering cannot drive the existing Dataset while preserving segment identity.
- Correct implementation requires changing a persisted artifact/schema contract.
- Existing split APIs cannot express the required post-filter explicit user split without a material public/cross-module contract change.

---

### T2 — Enforce Experiment A cohort authority in B/C/D

#### Goal

Make B/C/D reconstruct and enforce the cohort defined by A instead of independently selecting users or labels.

#### Required behavior

- B/C/D treat A checkpoint `train_users`, `val_users`, `test_users`, and `class_to_idx` as authoritative cohort inputs.
- After normal loading/validation, downstream experiments retain only users referenced by A's split and labels referenced by A's `class_to_idx`.
- Downstream code then rebuilds the split using A's explicit train/val/test users and A's class mapping.
- B/C/D must not independently redefine the cohort through separate user or label overrides.
- Any required A user or label that cannot be satisfied by the current dataset causes an explicit failure.
- Downstream must never silently drop users, labels, or segments to make the cohort valid.
- B/C/D preserve existing domain-specific training and evaluation behavior.
- Where two experiment views are intended to represent the same samples, paired sample membership and `package_index + segment_index` identity must remain aligned.

#### Preserved contracts

- Existing A checkpoint files remain usable without migration.
- No new checkpoint field is required unless existing fields are proven insufficient.
- B continues to use frozen A weights and A normalization.
- C/D continue to train fresh CNNs and use their existing normalization source.
- B/D raw/reconstruction paired sample identity and ordering remain unchanged.
- Existing domain-specific representations may differ; cohort membership may not.
- Existing evaluation metrics and model architecture remain unchanged.

#### Allowed write scope

- Experiment B/C/D runners/notebooks
- shared cohort-filtering/inheritance helpers required by B/C/D
- focused tests directly related to downstream cohort inheritance

The worker may modify local shared helpers/tests needed to express the same contract once, rather than duplicating independent B/C/D implementations.

#### Acceptance criteria

- The same A checkpoint drives identical train-user, val-user, and test-user assignments in A/B/C/D.
- A/B/C/D use identical selected label sets.
- A/B/C/D use identical `class_to_idx` mappings.
- B/C/D manifests contain no user outside A's split-user union.
- B/C/D manifests contain no label outside A's `class_to_idx`.
- Any required A user unavailable in the current downstream dataset causes an explicit failure.
- Any required A label unavailable in the current downstream dataset causes an explicit failure.
- Downstream cohort reconstruction does not silently intersect, reorder, or redefine A's cohort.
- Where paired views correspond to the same samples, surviving `package_index + segment_index` identities match exactly.
- B's frozen-weight and A-normalization contract remains unchanged.
- C/D still train fresh CNNs and preserve their existing normalization protocol.
- Existing A checkpoints require no migration.

#### Validation

Run focused B/C/D runner tests using one shared A-checkpoint fixture to verify:

- split-user inheritance;
- class-mapping inheritance;
- rejection of missing required users;
- rejection of missing required labels;
- no independent downstream cohort override;
- paired sample identity where applicable.

Then run one lightweight A -> B/C/D integration fixture asserting:

- identical split-user assignments;
- identical selected labels;
- identical `class_to_idx`;
- expected paired segment identity where applicable.

Do not run full-repository pytest unless focused integration reveals a broader regression surface.

#### Dependencies

- T1

#### Replan only if

- Existing A checkpoint `train_users`, `val_users`, `test_users`, and `class_to_idx` cannot uniquely establish the required downstream split and label space.
- Exact downstream cohort preservation requires additional persisted identity information not available from the existing checkpoint and dataset contracts.
- Correct downstream enforcement requires a checkpoint/schema migration or another durable public contract change.

## Final validation

Use one small fixture containing:

- at least one excluded user;
- at least one unselected label;
- multiple surviving users;
- multiple surviving selected labels;
- enough samples to exercise train/val/test inheritance.

Verify end-to-end that:

- A filters before splitting;
- A's final train/val/test users and `class_to_idx` are the downstream authority;
- B/C/D reproduce those split users and class mapping exactly;
- downstream manifests contain no out-of-cohort users or labels;
- paired views preserve required segment identity;
- B retains frozen A weights + A normalization;
- C/D retain fresh training + their existing normalization source.

Run focused acceleration-reconstruction experiment regression tests relevant to the changed surfaces.

Full-repository pytest is not required by default.

## Out of scope

- Cohort audit/provenance reporting.
- Checkpoint schema expansion unless T2 proves existing fields insufficient.
- A new rule requiring every user to contain every selected label.
- CNN architecture changes.
- Optimizer or training-algorithm changes.
- Embedding, retrieval, geometry, clustering, or other evaluation-metric changes.
- Artifact format migration.
